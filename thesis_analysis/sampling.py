#!/usr/bin/env python3
"""
Sample the VAE latent space, decode z -> macro circuit graphs, expand them to
primitive circuit graphs, plot examples with NetworkX, and quantify topological
validity.

Validity definition used here:
    A decoded circuit is topologically valid if its expanded primitive graph is
    non-empty, has no self-loops, and is one connected component. This directly
    tests whether latent samples decode to a single physical circuit instead of
    disconnected node clusters.

Example:
    python latent_space_circuit_sampling.py --checkpoint best_vae.pt --n-samples 1000 --n-examples 8
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import importlib.util
import pickle
import types
import numpy as np
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import torch
import os

THIS_FILE = Path(__file__).resolve()

def find_repo_root() -> Path:
    candidates = [THIS_FILE.parent, *THIS_FILE.parents, Path.cwd(), *Path.cwd().parents]
    for c in candidates:
        if (c / "inference_circuit_elements.py").exists() and (c / "src").exists():
            return c.resolve()
    raise RuntimeError("Could not find repo root: expected inference_circuit_elements.py and src/.")


REPO_ROOT = find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
os.chdir(REPO_ROOT)  # loaders use paths like data/Qubit.txt

from circuit2graph import CQEDTopology, SubgType, SUBG_DEFS, expand_topology, graphlize  # noqa: E402
from circuit2graph.constraints import exists_valid_macro_graph, is_compatible  # noqa: E402
try:  # noqa: E402
    from graph2qultra.topology_to_qultra import topology_to_net
    from graph2qultra.qultra_workflow import import_qultra, qelements_to_qultra_net, expected_mode_count
except Exception as _qultra_import_exc:  # noqa: E402
    topology_to_net = None
    import_qultra = None
    qelements_to_qultra_net = None
    expected_mode_count = None
    _QULTRA_IMPORT_ERROR = _qultra_import_exc
else:
    _QULTRA_IMPORT_ERROR = None
_DECODER_SPEC = importlib.util.spec_from_file_location("vae_decoder_local", REPO_ROOT / "src" / "vae_model" / "decoder.py")
if _DECODER_SPEC is None or _DECODER_SPEC.loader is None:
    raise RuntimeError("Could not load src/vae_model/decoder.py")
_decoder_module = importlib.util.module_from_spec(_DECODER_SPEC)
_DECODER_SPEC.loader.exec_module(_decoder_module)
TransformerTopologyDecoder = _decoder_module.TransformerTopologyDecoder
ParamDecoder = _decoder_module.ParamDecoder


MODEL_CONFIG_KEYS = {
    "nz", "hv_dim", "inner_hidden", "inner_layers", "outer_hidden", "outer_layers",
    "d", "nhead", "tf_layers", "max_nodes", "hs", "ggnn_rounds", "max_nodes_dec",
    "param_hidden", "param_layers", "spec_d", "spec_layers", "cg_hidden", "dropout",
    "beta_start", "beta_max", "warmup_steps", "attrs_scale", "align_scale",
    "nce_scale", "cg_scale", "spec_recon_scale", "tau", "class_weight_end",
}


def _install_minimal_data_loader_package_for_scalers() -> None:
    """Make checkpoint scaler unpickling work without importing torch_geometric."""
    if "data_loader" not in sys.modules:
        pkg = types.ModuleType("data_loader")
        pkg.__path__ = [str(REPO_ROOT / "src" / "data_loader")]
        sys.modules["data_loader"] = pkg

    if "data_loader.processing" not in sys.modules:
        proc = types.ModuleType("data_loader.processing")

        class ParamScaler:
            def inverse_transform(self, Y_scaled, attr_names=None):
                """Minimal checkpoint-unpickle compatible inverse scaler.

                Supports both the old dataset-specific signature
                inverse_transform(Y) and the global-scaler signature
                inverse_transform(Y, attr_names).  When named statistics are
                present, columns are selected by attr_names; otherwise it falls
                back to the vector statistics stored in the checkpoint.
                """
                Y_scaled = np.asarray(Y_scaled, dtype=np.float64)

                if attr_names is not None:
                    attr_names = [str(a).rsplit("_n", 1)[0] for a in attr_names]

                    # Common global-scaler layouts: dictionaries keyed by attr.
                    for mean_name, std_name in [
                        ("mean_by_attr", "std_by_attr"),
                        ("mean_by_attr_", "std_by_attr_"),
                        ("means", "stds"),
                        ("mean_dict", "std_dict"),
                    ]:
                        if hasattr(self, mean_name) and hasattr(self, std_name):
                            mean_obj = getattr(self, mean_name)
                            std_obj = getattr(self, std_name)
                            try:
                                mean = np.asarray([mean_obj[a] for a in attr_names], dtype=np.float64)
                                std = np.asarray([std_obj[a] for a in attr_names], dtype=np.float64)
                                return 10.0 ** (Y_scaled * std + mean)
                            except Exception:
                                pass

                    # Common vector layout plus an attribute-name list.
                    for names_name in ["attr_names", "attr_names_", "columns", "columns_"]:
                        if hasattr(self, names_name) and hasattr(self, "mean_") and hasattr(self, "std_"):
                            names = [str(a) for a in getattr(self, names_name)]
                            index = {name: i for i, name in enumerate(names)}
                            try:
                                idx = [index[a] for a in attr_names]
                                mean = np.asarray(self.mean_, dtype=np.float64)[idx]
                                std = np.asarray(self.std_, dtype=np.float64)[idx]
                                return 10.0 ** (Y_scaled * std + mean)
                            except Exception:
                                pass

                return 10.0 ** (Y_scaled * self.std_ + self.mean_)

        class ObsScaler:
            pass

        class DatasetScalers:
            def inverse_targets(self, Y_scaled):
                return self.param_scaler.inverse_transform(Y_scaled)

        ParamScaler.__module__ = "data_loader.processing"
        ObsScaler.__module__ = "data_loader.processing"
        DatasetScalers.__module__ = "data_loader.processing"
        proc.ParamScaler = ParamScaler
        proc.ObsScaler = ObsScaler
        proc.DatasetScalers = DatasetScalers
        sys.modules["data_loader.processing"] = proc


def load_checkpoint_scalers(checkpoint: dict) -> dict[str, Any]:
    """Load checkpoint scalers, supporting both old per-dataset and new global formats."""
    _install_minimal_data_loader_package_for_scalers()

    scalers: dict[str, Any] = {}

    raw = checkpoint.get("scalers")
    if raw is not None:
        try:
            loaded = pickle.loads(raw)
            if isinstance(loaded, dict):
                scalers.update(loaded)
        except Exception:
            pass

    raw_global = checkpoint.get("global_scaler")
    if raw_global is not None:
        try:
            scalers["__global__"] = pickle.loads(raw_global)
        except Exception:
            pass

    return scalers


def _dataset_attr_profiles() -> dict[str, dict[str, Any]]:
    """Return the canonical compressed parameter order for each dataset scaler."""
    profiles: dict[str, dict[str, Any]] = {}
    try:
        _install_minimal_data_loader_package_for_scalers()
        from data_loader.schema import DATASETS  # type: ignore
    except Exception:
        return profiles

    for ds_name, defn in DATASETS.items():
        try:
            compressed = graphlize(defn.topology_fn())
            node_types = [int(n.subg_type) for n in compressed._nodes]
            attr_names: list[str] = []
            for n in compressed._nodes:
                attr_names.extend([a for a in SUBG_DEFS[n.subg_type].attrs if a != "dir"])
            profiles[ds_name] = {
                "node_types": node_types,
                "attr_names": attr_names,
                "n_attrs": len(attr_names),
            }
        except Exception:
            continue
    return profiles



def _flat_attr_names_from_graph(g: SimpleNamespace) -> list[str]:
    """Flatten non-dir attribute names in the same order as ParamDecoder outputs."""
    names: list[str] = []
    for st_int in getattr(g, "node_types", []):
        st = SubgType(int(st_int))
        for attr_name in SUBG_DEFS[st].attrs:
            if attr_name != "dir":
                names.append(str(attr_name))
    return names


def _inverse_transform_named_param_scaler(scaler: Any, row: np.ndarray, attr_names: list[str]) -> np.ndarray:
    """Robust inverse transform for dataset-specific and global ParamScaler variants."""
    clean_names = [str(a).rsplit("_n", 1)[0] for a in attr_names]

    # Preferred global-scaler path.
    try:
        return np.asarray(scaler.inverse_transform(row, clean_names), dtype=np.float64)
    except TypeError:
        pass
    except Exception:
        pass

    # Old dataset-specific path.
    return np.asarray(scaler.inverse_transform(row), dtype=np.float64)


class LatentCircuitSampler:
    """Minimal inference wrapper that only loads decoder + parameter decoder.

    This avoids importing the encoder stack at sampling time, so the script can run
    even in environments where torch_geometric is not installed.
    """

    def __init__(self, checkpoint: dict, device: torch.device):
        config = checkpoint.get("config", {})
        self.scalers = load_checkpoint_scalers(checkpoint)
        self.scaler_profiles = _dataset_attr_profiles()
        self.nz = int(config.get("nz", 128))
        self.decoder = TransformerTopologyDecoder(
            nz=self.nz,
            d=int(config.get("hs", 512)),
            n_layers=int(config.get("ggnn_rounds", 3)),
            max_nodes=int(config.get("max_nodes_dec", 8)),
            dropout=float(config.get("dropout", 0.1)),
        ).to(device)
        self.param_decoder = ParamDecoder(
            nz=self.nz,
            hidden=int(config.get("param_hidden", 128)),
            sage_layers=int(config.get("param_layers", 3)),
            dropout=float(config.get("dropout", 0.1)),
        ).to(device)

        state = {k.replace("._orig_mod.", "."): v for k, v in checkpoint["model_state"].items()}
        decoder_state = {k.removeprefix("decoder."): v for k, v in state.items() if k.startswith("decoder.")}
        param_state = {k.removeprefix("param_decoder."): v for k, v in state.items() if k.startswith("param_decoder.")}
        self.decoder.load_state_dict(decoder_state, strict=True)
        self.param_decoder.load_state_dict(param_state, strict=True)
        self.decoder.eval()
        self.param_decoder.eval()

    @torch.no_grad()
    def decode(self, z: torch.Tensor, stochastic: bool) -> list[SimpleNamespace]:
        graphs = self.decoder.decode(z, stochastic=stochastic)
        attrs_per_graph = self.param_decoder.predict(z, graphs)
        for g, attrs in zip(graphs, attrs_per_graph):
            for i, st_int in enumerate(g.node_types):
                if "dir" in SUBG_DEFS[SubgType(int(st_int))].attrs:
                    attrs[i]["dir"] = float(getattr(g, "direction", [0.0] * len(g.node_types))[i])
            g.attrs = attrs
        return graphs

    def flat_scaled_attrs(self, g: SimpleNamespace) -> list[float]:
        row: list[float] = []
        for i, st_int in enumerate(getattr(g, "node_types", [])):
            node_attrs = g.attrs[i] if hasattr(g, "attrs") and i < len(g.attrs) else {}
            for attr_name in SUBG_DEFS[SubgType(int(st_int))].attrs:
                if attr_name == "dir":
                    continue
                row.append(float(node_attrs.get(attr_name, float("nan"))))
        return row

    def _select_param_scaler(self, g: SimpleNamespace) -> tuple[str | None, Any | None, str]:
        # New global-scaler checkpoint: use the same scaler for every decoded
        # topology and select columns by attribute name during inverse_transform.
        if "__global__" in self.scalers:
            return "__global__", self.scalers["__global__"].param_scaler, "global_scaler_named_attrs"

        node_types = [int(t) for t in getattr(g, "node_types", [])]
        n_attrs = len(self.flat_scaled_attrs(g))

        # Legacy checkpoint: best case, the sampled macro topology exactly matches a known
        # training compressed-topology signature, so the scaler columns are
        # unambiguous.
        exact = []
        for ds_name, profile in self.scaler_profiles.items():
            if profile.get("node_types") == node_types and profile.get("n_attrs") == n_attrs:
                exact.append(ds_name)
        if len(exact) == 1 and exact[0] in self.scalers:
            return exact[0], self.scalers[exact[0]].param_scaler, "exact_topology_signature"

        # Fallback: only use a scaler when the parameter-vector length is unique.
        # If several datasets have the same length, using one would silently mix
        # scaler columns, so we mark the sample as not physically rescalable.
        by_len = [
            ds_name for ds_name, profile in self.scaler_profiles.items()
            if profile.get("n_attrs") == n_attrs and ds_name in self.scalers
        ]
        if len(by_len) == 1:
            return by_len[0], self.scalers[by_len[0]].param_scaler, "unique_parameter_count"
        if len(by_len) > 1:
            return None, None, "ambiguous_scaler_for_parameter_count:" + ",".join(sorted(by_len))
        return None, None, "no_scaler_for_parameter_count"

    def rescale_graph_attrs_to_physical(self, g: SimpleNamespace) -> tuple[SimpleNamespace, dict[str, Any]]:
        """Convert decoded scaled attributes back to original physical units.

        The ParamDecoder predicts the same scaled targets used during training:
        log10-physical values standardized by the dataset's ParamScaler. This
        function applies the corresponding inverse scaler before expansion,
        validation, plotting, and Qultra conversion.
        """
        ds_name, scaler, scaler_status = self._select_param_scaler(g)
        scaled_row = self.flat_scaled_attrs(g)
        info = {
            "param_scaler_dataset": ds_name or "",
            "param_scaler_status": scaler_status,
            "physical_rescale_ok": False,
            "physical_rescale_error": "",
        }
        if scaler is None:
            info["physical_rescale_error"] = scaler_status
            return g, info
        try:
            row = np.asarray(scaled_row, dtype=np.float64).reshape(1, -1)
            attr_names = _flat_attr_names_from_graph(g)
            physical_row = _inverse_transform_named_param_scaler(scaler, row, attr_names)[0].tolist()
        except Exception as exc:
            info["physical_rescale_error"] = f"{type(exc).__name__}: {exc}"
            return g, info

        out = SimpleNamespace()
        out.node_types = list(getattr(g, "node_types", []))
        out.edges = [tuple(e) for e in getattr(g, "edges", [])]
        out.direction = list(getattr(g, "direction", [0.0] * len(out.node_types)))
        out.attrs = []
        cursor = 0
        for i, st_int in enumerate(out.node_types):
            st = SubgType(int(st_int))
            old_attrs = g.attrs[i] if hasattr(g, "attrs") and i < len(g.attrs) else {}
            d: dict[str, Any] = {}
            for attr_name in SUBG_DEFS[st].attrs:
                if attr_name == "dir":
                    d["dir"] = float(old_attrs.get("dir", out.direction[i] if i < len(out.direction) else 0.0))
                    continue
                d[attr_name] = float(physical_row[cursor]) if cursor < len(physical_row) else float("nan")
                cursor += 1
            out.attrs.append(d)
        info["physical_rescale_ok"] = True
        return out, info


def load_latent_sampler(checkpoint_path: Path, device: torch.device) -> LatentCircuitSampler:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return LatentCircuitSampler(checkpoint, device)


def graph_ns_to_cqed_topology(g: SimpleNamespace, name: str) -> CQEDTopology:
    """Convert the decoder SimpleNamespace graph into the repo CQEDTopology class."""
    topo = CQEDTopology(name)
    nodes = []
    for i, st_int in enumerate(getattr(g, "node_types", [])):
        st = SubgType(int(st_int))
        attrs = dict(g.attrs[i]) if hasattr(g, "attrs") and i < len(g.attrs) else {}
        nodes.append(topo.add_node(st, attrs, label=f"{st.name}_{i}"))

    for u, v in getattr(g, "edges", []):
        u, v = int(u), int(v)
        if 0 <= u < len(nodes) and 0 <= v < len(nodes) and u != v:
            topo.add_edge(nodes[u], nodes[v])
    return topo


def fmt_attr_value(v: Any) -> str:
    if isinstance(v, (float, int)):
        if not math.isfinite(float(v)):
            return str(v)
        return f"{float(v):.2e}"
    return str(v)


def topology_to_nx(topology: CQEDTopology) -> nx.Graph:
    G = nx.Graph(name=topology.name)
    for node in topology._nodes:
        type_name = node.subg_type.name
        attrs = dict(node.attrs)
        attrs_text = "\n".join(f"{k}={fmt_attr_value(v)}" for k, v in attrs.items())
        label = f"{node.node_id}: {type_name}"
        if node.label:
            label += f"\n{node.label}"
        if attrs_text:
            label += f"\n{attrs_text}"
        G.add_node(
            node.node_id,
            label=label,
            subg_type=node.subg_type,
            type_name=type_name,
            attrs=attrs,
        )
    for u, v in topology._edges:
        if u != v:
            G.add_edge(int(u), int(v))
    return G




def _is_macro_graph_connected(n_nodes: int, edges: list[tuple[int, int]]) -> bool:
    if n_nodes <= 1:
        return n_nodes == 1
    adj = [[] for _ in range(n_nodes)]
    for u, v in edges:
        u, v = int(u), int(v)
        if 0 <= u < n_nodes and 0 <= v < n_nodes and u != v:
            adj[u].append(v)
            adj[v].append(u)
    seen = set()
    stack = [0]
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        stack.extend(v for v in adj[u] if v not in seen)
    return len(seen) == n_nodes


def analyze_macro_physical_constraints(g: SimpleNamespace) -> dict[str, Any]:
    """Final macro-level constraint analysis for sampled decoded graphs."""
    node_types = [int(t) for t in getattr(g, "node_types", [])]
    directions = [float(d) for d in getattr(g, "direction", [0.0] * len(node_types))]
    if len(directions) < len(node_types):
        directions = directions + [0.0] * (len(node_types) - len(directions))
    directions = directions[:len(node_types)]
    edges = [(int(u), int(v)) for u, v in getattr(g, "edges", [])]

    reasons: list[str] = []
    n = len(node_types)
    if n == 0:
        reasons.append("empty_macro_graph")

    try:
        macro_completion_exists = bool(exists_valid_macro_graph(node_types, directions))
    except Exception as exc:
        macro_completion_exists = False
        reasons.append(f"macro_completion_check_error:{type(exc).__name__}")

    if n > 0 and not macro_completion_exists:
        reasons.append("no_connected_physical_macro_completion")

    bad_edges: list[str] = []
    for u, v in edges:
        if not (0 <= u < n and 0 <= v < n) or u == v:
            bad_edges.append(f"{u}-{v}:invalid_indices")
            continue
        try:
            ok = bool(is_compatible(node_types[u], directions[u], node_types[v], directions[v]))
        except Exception:
            ok = False
        if not ok:
            bad_edges.append(f"{u}-{v}")
    if bad_edges:
        reasons.append("macro_edge_not_physical")

    actual_connected = _is_macro_graph_connected(n, edges)
    if n > 1 and not actual_connected:
        reasons.append("macro_graph_disconnected")

    is_valid = len(reasons) == 0
    return {
        "macro_completion_exists": macro_completion_exists,
        "macro_actual_connected": actual_connected,
        "macro_edges_all_physical": len(bad_edges) == 0,
        "macro_bad_edges": ";".join(bad_edges),
        "is_macro_physical_valid": is_valid,
        "macro_invalid_reasons": ";".join(reasons),
    }

def analyze_primitive_topology(topology: CQEDTopology) -> dict[str, Any]:
    G = topology_to_nx(topology)
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    n_components = nx.number_connected_components(G) if n_nodes else 0
    connected = bool(n_nodes > 0 and nx.is_connected(G))
    has_self_loops = bool(nx.number_of_selfloops(G) > 0)
    valid = bool(n_nodes > 0 and connected and not has_self_loops)
    return {
        "n_primitive_nodes": n_nodes,
        "n_primitive_edges": n_edges,
        "n_components": n_components,
        "is_connected": connected,
        "has_self_loops": has_self_loops,
        "is_topologically_valid": valid,
    }



PHYSICAL_TYPES = {SubgType.TRANSMON, SubgType.RESONATOR, SubgType.FEEDLINE}
COUPLER_TYPES = {SubgType.C_COUPLER, SubgType.I_COUPLER}
SUPPORTED_QULTRA_PRIMITIVE_TYPES = PHYSICAL_TYPES | COUPLER_TYPES
POSITIVE_ATTR_KEYS = {"L", "C", "Cc", "Cc_qr", "Cc_rf", "length", "l", "D"}


def _node_by_id(topology: CQEDTopology) -> dict[int, Any]:
    return {int(n.node_id): n for n in topology._nodes}


def _adjacency(topology: CQEDTopology) -> dict[int, list[int]]:
    adj = {int(n.node_id): [] for n in topology._nodes}
    for u, v in topology._edges:
        u, v = int(u), int(v)
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    return adj


def _is_positive_finite(value: Any) -> bool:
    try:
        x = float(value)
    except Exception:
        return False
    return math.isfinite(x) and x > 0.0


def _fail(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def analyze_superconducting_circuit_validity(topology: CQEDTopology, base_valid: bool, physical_rescale_ok: bool = True) -> dict[str, Any]:
    """Conservative physical-validity test not used during training.

    The test is stricter than plain graph connectivity. It checks that the
    expanded primitive topology looks like a realizable superconducting circuit:
    physical nodes (T/R/F) are coupled only through capacitive/inductive
    couplers, couplers are not directly chained together, couplers have the
    expected degree, I-couplers connect exactly resonator-feedline pairs, all
    primitive node types are supported by the Qultra converter, and known
    physical parameters are finite and positive.
    """
    reasons: list[str] = []
    nodes = _node_by_id(topology)
    adj = _adjacency(topology)

    if not base_valid:
        _fail(reasons, "base_topology_invalid")
    if not physical_rescale_ok:
        _fail(reasons, "physical_parameter_rescale_unavailable")

    unsupported = [n.subg_type.name for n in topology._nodes if n.subg_type not in SUPPORTED_QULTRA_PRIMITIVE_TYPES]
    if unsupported:
        _fail(reasons, "unsupported_primitive_type")

    n_modes = sum(1 for n in topology._nodes if n.subg_type in {SubgType.TRANSMON, SubgType.RESONATOR})
    if n_modes < 1:
        _fail(reasons, "no_mode_bearing_element")

    for u, v in topology._edges:
        if int(u) == int(v):
            _fail(reasons, "self_loop")
            continue
        a, b = nodes.get(int(u)), nodes.get(int(v))
        if a is None or b is None:
            _fail(reasons, "edge_references_missing_node")
            continue
        if a.subg_type in PHYSICAL_TYPES and b.subg_type in PHYSICAL_TYPES:
            _fail(reasons, "direct_physical_node_edge")
        if a.subg_type in COUPLER_TYPES and b.subg_type in COUPLER_TYPES:
            _fail(reasons, "direct_coupler_coupler_edge")

    for node in topology._nodes:
        st = node.subg_type
        degree = len(adj.get(int(node.node_id), []))
        if st in COUPLER_TYPES and degree != 2:
            _fail(reasons, "coupler_degree_not_two")
        if st == SubgType.C_COUPLER:
            neigh_types = [nodes[nid].subg_type for nid in adj.get(int(node.node_id), []) if nid in nodes]
            if len(neigh_types) != 2 or any(nt not in PHYSICAL_TYPES for nt in neigh_types):
                _fail(reasons, "capacitive_coupler_not_between_physical_nodes")
        elif st == SubgType.I_COUPLER:
            neigh_types = [nodes[nid].subg_type for nid in adj.get(int(node.node_id), []) if nid in nodes]
            if set(neigh_types) != {SubgType.RESONATOR, SubgType.FEEDLINE} or len(neigh_types) != 2:
                _fail(reasons, "inductive_coupler_not_resonator_feedline")
        if st == SubgType.FEEDLINE and degree > 2:
            _fail(reasons, "feedline_degree_greater_than_two")
        for key, value in dict(getattr(node, "attrs", {})).items():
            if key in POSITIVE_ATTR_KEYS and not _is_positive_finite(value):
                _fail(reasons, "non_positive_or_non_finite_physical_parameter")

    is_valid = len(reasons) == 0
    return {
        "is_superconducting_circuit_valid": is_valid,
        "superconducting_invalid_reasons": ";".join(reasons),
    }


def analyze_qultra_analyzability(topology: CQEDTopology, run_qultra: bool, f_min: float, f_max: float) -> dict[str, Any]:
    """Check whether the primitive circuit can be converted/analyzed by Qultra.

    By default this performs the repo's topology_to_net transformation, which is
    the stable Qultra-front-end validity test. With --run-qultra it also imports
    Qultra and instantiates QCircuit in the requested frequency window.
    """
    if topology_to_net is None:
        return {
            "is_qultra_analyzable": False,
            "qultra_stage": "import_graph2qultra",
            "qultra_expected_modes": 0,
            "qultra_error": f"{type(_QULTRA_IMPORT_ERROR).__name__}: {_QULTRA_IMPORT_ERROR}",
        }
    try:
        qelements = topology_to_net(topology)
        n_expected = expected_mode_count(topology) if expected_mode_count is not None else 0
        if n_expected < 1:
            raise ValueError("Qultra topology has zero expected physical modes")
        if run_qultra:
            if import_qultra is None or qelements_to_qultra_net is None:
                raise RuntimeError(f"Qultra imports unavailable: {_QULTRA_IMPORT_ERROR}")
            qu = import_qultra()
            qnet = qelements_to_qultra_net(qelements, qu)
            circuit = qu.QCircuit(qnet, f_min, f_max)
            try:
                _ = circuit.mode_frequencies()
            finally:
                del circuit
        return {
            "is_qultra_analyzable": True,
            "qultra_stage": "QCircuit" if run_qultra else "topology_to_net",
            "qultra_expected_modes": int(n_expected),
            "qultra_error": "",
        }
    except Exception as exc:
        return {
            "is_qultra_analyzable": False,
            "qultra_stage": "QCircuit" if run_qultra else "topology_to_net",
            "qultra_expected_modes": 0,
            "qultra_error": f"{type(exc).__name__}: {exc}",
        }


def save_percent_bar(labels: list[str], values: list[float], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.3, 4.5))
    ax.bar(labels, values)
    ax.set_ylabel("Decoded samples (%)")
    ax.set_ylim(0, 100)
    ax.set_title(title, fontsize=10)
    for idx, pct in enumerate(values):
        ax.text(idx, 2, f"{pct:.1f}%", ha="center", va="bottom", fontsize=11, color="black")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_topology(ax: plt.Axes, topology: CQEDTopology, title: str) -> None:
    G = topology_to_nx(topology)
    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "empty graph", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.axis("off")
        return

    pos = nx.spring_layout(G, seed=7, k=1.1) if G.number_of_nodes() > 2 else nx.shell_layout(G)
    colors = [SUBG_DEFS[G.nodes[n]["subg_type"]].color for n in G.nodes]
    labels = {n: G.nodes[n]["label"] for n in G.nodes}

    nx.draw_networkx_edges(G, pos, ax=ax, width=1.8, alpha=0.75)
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=colors, node_size=2300,
        edgecolors="black", linewidths=0.8, alpha=0.95,
    )
    nx.draw_networkx_labels(
        G, pos, labels=labels, ax=ax, font_size=6.5,
        font_color="black", font_weight="bold",
    )
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def save_example_grid(expanded_topologies: list[CQEDTopology], rows: list[dict[str, Any]], out_path: Path) -> None:
    n = len(expanded_topologies)
    if n == 0:
        return
    n_cols = min(4, n)
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 4.6 * n_rows))
    axes_flat = list(getattr(axes, "flat", [axes]))
    for ax in axes_flat[n:]:
        ax.axis("off")
    for i, topo in enumerate(expanded_topologies):
        status = "valid" if rows[i]["is_topologically_valid"] else "invalid"
        draw_topology(axes_flat[i], topo, f"Sample {rows[i]['sample_id']} - {status}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_validity_bar(n_valid: int, n_total: int, out_path: Path) -> None:
    valid_pct = 100.0 * n_valid / max(n_total, 1)
    invalid_pct = 100.0 - valid_pct
    save_percent_bar(
        ["Topologically valid", "Invalid / disconnected"],
        [valid_pct, invalid_pct],
        out_path,
        "Validity: non-empty, connected, and no self-loops.",
    )


def write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "best_vae.pt")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "pres_plot/latent_sampling_output")
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--n-examples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true", help="Use argmax decoding instead of stochastic decoding.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--torch-threads", type=int, default=1, help="CPU Torch threads. 1 is usually fastest for autoregressive sampling.")
    parser.add_argument("--run-qultra", action="store_true", help="After topology_to_net, instantiate Qultra QCircuit and call mode_frequencies().")
    parser.add_argument("--f-min", type=float, default=1.0, help="Minimum frequency for optional Qultra QCircuit analysis.")
    parser.add_argument("--f-max", type=float, default=9.0, help="Maximum frequency for optional Qultra QCircuit analysis.")
    args = parser.parse_args()
    if args.torch_threads and args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model = load_latent_sampler(args.checkpoint, device)
    rows: list[dict[str, Any]] = []
    example_topologies: list[CQEDTopology] = []
    example_rows: list[dict[str, Any]] = []

    sampled = 0
    with torch.no_grad():
        while sampled < args.n_samples:
            current = min(args.batch_size, args.n_samples - sampled)
            z = torch.randn(current, model.nz, device=device)
            decoded = model.decode(z, stochastic=not args.deterministic)

            for local_i, g in enumerate(decoded):
                sample_id = sampled + local_i
                row: dict[str, Any] = {
                    "sample_id": sample_id,
                    "n_macro_nodes": len(getattr(g, "node_types", [])),
                    "n_macro_edges": len(getattr(g, "edges", [])),
                    "macro_node_types": ";".join(SubgType(int(t)).name for t in getattr(g, "node_types", [])),
                    "macro_edges": ";".join(f"{int(u)}-{int(v)}" for u, v in getattr(g, "edges", [])),
                    "macro_directions": ";".join(f"{float(d):.0f}" for d in getattr(g, "direction", [])),
                    "expansion_error": "",
                }
                row.update(analyze_macro_physical_constraints(g))
                try:
                    g_physical, rescale_info = model.rescale_graph_attrs_to_physical(g)
                    row.update(rescale_info)
                    macro_topology = graph_ns_to_cqed_topology(g_physical, name=f"sample_{sample_id}_macro")
                    primitive_topology = expand_topology(macro_topology, validate=False)
                    row.update(analyze_primitive_topology(primitive_topology))
                    row.update(analyze_superconducting_circuit_validity(
                        primitive_topology, bool(row["is_topologically_valid"]), bool(row.get("physical_rescale_ok", False))
                    ))
                    row.update(analyze_qultra_analyzability(
                        primitive_topology, run_qultra=args.run_qultra, f_min=args.f_min, f_max=args.f_max
                    ))
                    row["is_physical_valid"] = bool(
                        row.get("is_macro_physical_valid")
                        and row.get("is_superconducting_circuit_valid")
                        and row.get("is_qultra_analyzable")
                    )
                    row["physical_invalid_reasons"] = ";".join([
                        str(row.get("macro_invalid_reasons", "") or ""),
                        str(row.get("superconducting_invalid_reasons", "") or ""),
                        "" if row.get("is_qultra_analyzable") else "not_qultra_analyzable",
                    ]).strip(";")
                    if len(example_topologies) < args.n_examples:
                        example_topologies.append(primitive_topology)
                        example_rows.append(row.copy())
                except Exception as exc:
                    row.update({
                        "n_primitive_nodes": 0,
                        "n_primitive_edges": 0,
                        "n_components": 0,
                        "is_connected": False,
                        "has_self_loops": False,
                        "is_topologically_valid": False,
                        "is_superconducting_circuit_valid": False,
                        "superconducting_invalid_reasons": "expansion_error",
                        "is_qultra_analyzable": False,
                        "qultra_stage": "expansion",
                        "qultra_expected_modes": 0,
                        "qultra_error": f"{type(exc).__name__}: {exc}",
                        "param_scaler_dataset": row.get("param_scaler_dataset", ""),
                        "param_scaler_status": row.get("param_scaler_status", ""),
                        "physical_rescale_ok": bool(row.get("physical_rescale_ok", False)),
                        "physical_rescale_error": row.get("physical_rescale_error", ""),
                        "expansion_error": f"{type(exc).__name__}: {exc}",
                        "is_physical_valid": False,
                        "physical_invalid_reasons": ";".join([
                            str(row.get("macro_invalid_reasons", "") or ""),
                            "expansion_error",
                        ]).strip(";"),
                    })
                rows.append(row)
            sampled += current

    n_total = len(rows)
    n_valid = sum(1 for r in rows if r["is_topologically_valid"])
    n_sc_valid = sum(1 for r in rows if r.get("is_superconducting_circuit_valid"))
    n_qultra = sum(1 for r in rows if r.get("is_qultra_analyzable"))
    n_macro_valid = sum(1 for r in rows if r.get("is_macro_physical_valid"))
    n_physical_valid = sum(1 for r in rows if r.get("is_physical_valid"))
    macro_reason_counts = Counter()
    physical_reason_counts = Counter()
    sc_reason_counts = Counter()
    qultra_reason_counts = Counter()
    for r in rows:
        if not r.get("is_macro_physical_valid"):
            for reason in str(r.get("macro_invalid_reasons", "unknown") or "unknown").split(";"):
                macro_reason_counts[reason] += 1
        if not r.get("is_physical_valid"):
            for reason in str(r.get("physical_invalid_reasons", "unknown") or "unknown").split(";"):
                physical_reason_counts[reason] += 1
        if not r.get("is_superconducting_circuit_valid"):
            for reason in str(r.get("superconducting_invalid_reasons", "unknown") or "unknown").split(";"):
                sc_reason_counts[reason] += 1
        if not r.get("is_qultra_analyzable"):
            qerr = str(r.get("qultra_error", "unknown") or "unknown")
            qultra_reason_counts[qerr.split(":", 1)[0]] += 1

    summary = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "n_samples": n_total,
        "n_valid": n_valid,
        "n_invalid": n_total - n_valid,
        "valid_percent": 100.0 * n_valid / max(n_total, 1),
        "invalid_percent": 100.0 * (n_total - n_valid) / max(n_total, 1),
        "n_macro_physical_valid": n_macro_valid,
        "n_macro_physical_invalid": n_total - n_macro_valid,
        "macro_physical_valid_percent": 100.0 * n_macro_valid / max(n_total, 1),
        "macro_physical_invalid_percent": 100.0 * (n_total - n_macro_valid) / max(n_total, 1),
        "n_physical_valid": n_physical_valid,
        "n_physical_invalid": n_total - n_physical_valid,
        "physical_valid_percent": 100.0 * n_physical_valid / max(n_total, 1),
        "physical_invalid_percent": 100.0 * (n_total - n_physical_valid) / max(n_total, 1),
        "n_superconducting_valid": n_sc_valid,
        "n_superconducting_invalid": n_total - n_sc_valid,
        "superconducting_valid_percent": 100.0 * n_sc_valid / max(n_total, 1),
        "superconducting_invalid_percent": 100.0 * (n_total - n_sc_valid) / max(n_total, 1),
        "n_qultra_analyzable": n_qultra,
        "n_qultra_not_analyzable": n_total - n_qultra,
        "qultra_analyzable_percent": 100.0 * n_qultra / max(n_total, 1),
        "qultra_not_analyzable_percent": 100.0 * (n_total - n_qultra) / max(n_total, 1),
        "macro_physical_validity_definition": "generated macro-node set admits a connected graph of physical macro-edges; actual sampled macro graph is connected; every sampled macro edge is physically compatible in its stored orientation",
        "physical_validity_definition": "macro physical validity AND strict superconducting primitive validity AND Qultra analyzability",
        "macro_invalid_reason_counts": dict(macro_reason_counts),
        "physical_invalid_reason_counts": dict(physical_reason_counts),
        "validity_definition": "expanded primitive graph is non-empty, connected, and has no self-loops",
        "superconducting_validity_definition": "physical attributes inverse-scaled to original units before expansion/checks; base-valid, supported T/R/F/C/I primitive types, no direct T/R/F edge, no coupler-coupler edge, coupler degree two, I-coupler is R-F, positive finite physical parameters, at least one mode",
        "qultra_definition": "physical attributes inverse-scaled to original units, then topology_to_net succeeds; with --run-qultra, QCircuit construction and mode_frequencies also succeed",
        "superconducting_invalid_reason_counts": dict(sc_reason_counts),
        "qultra_failure_counts": dict(qultra_reason_counts),
        "stochastic_decoding": not args.deterministic,
        "random_latent_sampling": "z ~ N(0, I) independently for every sample; stochastic decoder unless --deterministic is passed",
        "seed": args.seed,
        "run_qultra": bool(args.run_qultra),
        "f_min": args.f_min,
        "f_max": args.f_max,
    }

    write_csv(rows, args.out_dir / "latent_sampling_topology_analysis.csv")
    (args.out_dir / "latent_sampling_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_example_grid(example_topologies, example_rows, args.out_dir / "decoded_primitive_examples.png")
    save_validity_bar(n_valid, n_total, args.out_dir / "topological_validity_percent.png")
    save_percent_bar(
        ["SC-circuit valid", "Invalid by SC rules"],
        [100.0 * n_sc_valid / max(n_total, 1), 100.0 * (n_total - n_sc_valid) / max(n_total, 1)],
        args.out_dir / "superconducting_circuit_validity_percent.png",
        "Strict SC validity: connected, T/R/F only via couplers; no coupler-coupler edges; valid coupler degrees/params.",
    )
    save_percent_bar(
        ["Qultra analyzable", "Not analyzable"],
        [100.0 * n_qultra / max(n_total, 1), 100.0 * (n_total - n_qultra) / max(n_total, 1)],
        args.out_dir / "qultra_analyzability_percent.png",
        "Qultra analyzability: primitive circuit transforms to Qultra net" + (" and QCircuit runs." if args.run_qultra else "."),
    )
    save_percent_bar(
        ["Macro physical valid", "Macro invalid"],
        [100.0 * n_macro_valid / max(n_total, 1), 100.0 * (n_total - n_macro_valid) / max(n_total, 1)],
        args.out_dir / "macro_physical_validity_percent.png",
        "Macro constraints: connectable macro-node set, connected sampled macro graph, physical sampled edges.",
    )
    save_percent_bar(
        ["Final physical valid", "Final invalid"],
        [100.0 * n_physical_valid / max(n_total, 1), 100.0 * (n_total - n_physical_valid) / max(n_total, 1)],
        args.out_dir / "final_physical_validity_percent.png",
        "Final validity: macro constraints + strict primitive SC validity + Qultra analyzability.",
    )

    print(json.dumps(summary, indent=2))
    print(f"Wrote: {args.out_dir / 'decoded_primitive_examples.png'}")
    print(f"Wrote: {args.out_dir / 'topological_validity_percent.png'}")
    print(f"Wrote: {args.out_dir / 'superconducting_circuit_validity_percent.png'}")
    print(f"Wrote: {args.out_dir / 'qultra_analyzability_percent.png'}")
    print(f"Wrote: {args.out_dir / 'macro_physical_validity_percent.png'}")
    print(f"Wrote: {args.out_dir / 'final_physical_validity_percent.png'}")
    print(f"Wrote: {args.out_dir / 'latent_sampling_topology_analysis.csv'}")


if __name__ == "__main__":
    main()
