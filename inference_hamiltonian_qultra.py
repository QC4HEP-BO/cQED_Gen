from __future__ import annotations

import argparse
import inspect
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import json
import gc
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

try:
    import networkx as nx
    _HAS_NX = True
except ImportError:
    _HAS_NX = False
    print("WARN: networkx non installato — i plot di grafo saranno disabilitati.")

REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT  = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vae_model.vae import GraphVAE
from data_loader.loader_vae import load_all_datasets_vae, load_inference_datasets_vae
from data_loader.schema import train_datasets
from vae_model.decoder import data_to_graph_ns
from data_loader.schema import DATASETS, OBS_SLOTS
from circuit2graph import SubgType, SUBG_DEFS, CQEDTopology, expand_topology

UNITS: dict[str, str] = {
    "L": "H",
    "C": "F",
    "length": "m",
    "Cc": "F",
    "Cc_qr": "F",
    "Cc_rf": "F",
    "L2": "H",
    "C2": "F",
    "D": "µm",
    "l": "m",
}

DS_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2"]

DEBUG_QULTRA_DATASET: str | None = None
DEBUG_QULTRA_FAILURES: int = 0
DEBUG_QULTRA_TRACEBACK: bool = False
_DEBUG_QULTRA_COUNTS: dict[str, int] = defaultdict(int)

# Qultra workflow helpers. These files should live in the repo root.
try:
    from graph2qultra.qultra_workflow import import_qultra, qelements_to_qultra_net, expected_mode_count
    from graph2qultra.topology_to_qultra import topology_to_net
    _HAS_QULTRA_WORKFLOW = True
except Exception as _qultra_import_exc:
    import_qultra = None
    qelements_to_qultra_net = None
    expected_mode_count = None
    topology_to_net = None
    _HAS_QULTRA_WORKFLOW = False
    _QULTRA_IMPORT_ERROR = _qultra_import_exc


def _unit(attr: str) -> str:
    """Look up the physical unit for a (possibly suffixed) attribute name.

    Suffixed names take the form ``<base>_n<node_index>`` (e.g. ``length_n2``).
    Strip the suffix before looking up in UNITS so that duplicate attributes
    produced by _unique_attr_names_from_graph still get the right unit string.
    """
    unit = UNITS.get(attr)
    if unit is not None:
        return unit
    # Try stripping the node-index suffix added for duplicate attrs.
    base = attr.rsplit("_n", 1)[0] if "_n" in attr else attr
    return UNITS.get(base, "")



def _unique_attr_names_from_graph(g):
    """Return display names in the exact same order as sample.y/scaler columns.

    ``dir`` is a discrete topological/classification variable in the current
    model, not a continuous physical parameter. Therefore it is intentionally
    skipped here so the returned names match sample.y and the parameter scaler.
    """
    names = []
    seen_bases: set[str] = set()
    for node_i, st_int in enumerate(g.node_types):
        for attr_name in SUBG_DEFS[SubgType(st_int)].attrs:
            if attr_name == "dir":
                continue
            if attr_name not in seen_bases:
                seen_bases.add(attr_name)
                display = attr_name
            else:
                display = f"{attr_name}_n{node_i}"
            names.append(display)
    return names



def _attr_node_indices_from_graph(g):
    """Return the node index for each continuous flattened attribute column."""
    node_indices = []
    for node_i, st_int in enumerate(g.node_types):
        for attr_name in SUBG_DEFS[SubgType(st_int)].attrs:
            if attr_name == "dir":
                continue
            node_indices.append(node_i)
    return node_indices

def _scaled_attr_row_from_pred_graph(g_pred):
    """Flatten predicted continuous attrs by node and SUBG_DEFS order.

    ``dir`` is predicted by the topological decoder and stored in attrs only for
    graph expansion/Qultra. It must not be part of the scaled parameter row.
    """
    row = []
    for node_i, st_int in enumerate(g_pred.node_types):
        attrs_ordered = SUBG_DEFS[SubgType(st_int)].attrs
        node_attrs = g_pred.attrs[node_i] if node_i < len(g_pred.attrs) else {}
        for a in attrs_ordered:
            if a == "dir":
                continue
            row.append(float(node_attrs.get(a, float("nan"))))
    return row



def _get_attr_perm_indices(sample, n_attrs: int) -> list[list[int]]:
    """Return saved symmetry-induced parameter permutations for one sample.

    The loader stores these permutations from the primitive true topology.
    Identity-only samples therefore return exactly [range(n_attrs)].
    """
    raw = getattr(sample, "attr_perm_indices", None)
    if raw is None:
        return [list(range(n_attrs))]

    # torch_geometric may keep Python lists as-is, but be permissive.
    if torch.is_tensor(raw):
        raw = raw.detach().cpu().tolist()

    perms: list[list[int]] = []
    for perm in raw:
        if torch.is_tensor(perm):
            perm = perm.detach().cpu().tolist()
        perm = [int(i) for i in perm]
        if len(perm) == n_attrs and sorted(perm) == list(range(n_attrs)):
            perms.append(perm)

    return perms or [list(range(n_attrs))]


def _best_pred_row_aligned_to_true_order(true_row, pred_row, perms: list[list[int]]):
    """Align prediction to the canonical true/scaler column order using saved perms.

    Training computes min MSE(pred, true[perm]) in scaled space.  For reporting
    physical metrics we must keep columns in the original scaler order.  Therefore,
    after choosing the best perm, we move the prediction back with the inverse
    mapping: aligned_pred[perm[k]] = pred[k].
    """
    true_arr = np.asarray(true_row, dtype=np.float64)
    pred_arr = np.asarray(pred_row, dtype=np.float64)
    if len(perms) <= 1 or true_arr.shape != pred_arr.shape or not np.all(np.isfinite(pred_arr)):
        return pred_arr.tolist(), 0, len(perms)

    best_i = 0
    best_loss = float("inf")
    for i, perm in enumerate(perms):
        idx = np.asarray(perm, dtype=np.int64)
        cand_true = true_arr[idx]
        mask = np.isfinite(cand_true) & np.isfinite(pred_arr)
        if not np.any(mask):
            loss = float("inf")
        else:
            d = pred_arr[mask] - cand_true[mask]
            loss = float(np.mean(d * d))
        if loss < best_loss:
            best_loss = loss
            best_i = i

    best_perm = perms[best_i]
    aligned = np.full_like(pred_arr, np.nan, dtype=np.float64)
    for pred_k, true_k in enumerate(best_perm):
        aligned[int(true_k)] = pred_arr[pred_k]
    return aligned.tolist(), best_i, len(perms)


def _build_model_from_config(config: dict) -> GraphVAE:
    sig = inspect.signature(GraphVAE.__init__)
    accepted = set(sig.parameters.keys()) - {"self"}

    candidate_kwargs = {
        "nz": config["nz"],
        "hv_dim": config["hv_dim"],
        "inner_hidden": config["inner_hidden"],
        "inner_layers": config["inner_layers"],
        "outer_hidden": config["outer_hidden"],
        "outer_layers": config["outer_layers"],
        "d": config["d"],
        "nhead": config["nhead"],
        "tf_layers": config["tf_layers"],
        "max_nodes": config["max_nodes"],
        "hs": config["hs"],
        "ggnn_rounds": config["ggnn_rounds"],
        "max_nodes_dec": config["max_nodes_dec"],
        "param_hidden": config["param_hidden"],
        "param_layers": config["param_layers"],
        "spec_d": config.get("spec_d", 64),
        "spec_layers": config.get("spec_layers", 3),
        "cg_hidden": config.get("cg_hidden", 64),
        "dropout": config.get("dropout", 0.1),
        "beta_start": config.get("beta_start", 0.0),
        "beta_max": config.get("beta_max", 1.0),
        "warmup_steps": config.get("warmup_steps", 5000),
        "attrs_scale": config.get("attrs_scale", 1.0),
        "align_scale": config.get("align_scale", 0.5),
        "nce_scale": config.get("nce_scale", 0.5),
        "cg_scale": config.get("cg_scale", 0.5),
        "spec_recon_scale": config.get("spec_recon_scale", 0.3),
        "tau": config.get("tau", 0.1),
        "class_weight_end": config.get("class_weight_end", 1.0),
    }

    kwargs = {k: v for k, v in candidate_kwargs.items() if k in accepted}
    return GraphVAE(**kwargs)


def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    scalers = pickle.loads(ckpt["scalers"]) if "scalers" in ckpt else {}

    model = _build_model_from_config(config).to(device)

    # load_state_dict_full does not exist on GraphVAE — calling it raises
    # AttributeError, not RuntimeError, so the old except clause never fired
    # and the model kept random initialisation weights.  Load directly.
    # Normalizza le chiavi: torch.compile salva con '_orig_mod.' nel path.
    # Es: 'encoder._orig_mod.inner_enc.lin_in.weight' -> 'encoder.inner_enc.lin_in.weight'
    raw_state = ckpt["model_state"]
    clean_state = {k.replace("._orig_mod.", "."): v for k, v in raw_state.items()}

    missing, unexpected = model.load_state_dict(clean_state, strict=True)
    if missing:
        print(f"  ATTENZIONE: {len(missing)} chiavi mancanti nel checkpoint: {missing[:5]}{'…' if len(missing)>5 else ''}")
    if unexpected:
        print(f"  ATTENZIONE: {len(unexpected)} chiavi inattese nel checkpoint: {unexpected[:5]}{'…' if len(unexpected)>5 else ''}")

    if hasattr(model, "_step"):
        model._step = ckpt.get("step", 0)

    # Sanity check: if weights look like random init the checkpoint did not load.
    w = next(model.parameters())
    w_std = w.std().item()
    if w_std < 1e-4 or w_std > 2.0:
        print(f"  ATTENZIONE: std pesi = {w_std:.4f} — possibile errore nel caricamento del checkpoint")
    else:
        print(f"  Pesi caricati OK (std primo layer = {w_std:.4f})")

    model.eval()
    return model, scalers, config


def topo_match(true_types, true_edges, pred_types, pred_edges) -> dict:
    n_true = len(true_types)
    n_pred = len(pred_types)
    n_min = min(n_true, n_pred)
    n_correct_nodes = sum(1 for i in range(n_min) if true_types[i] == pred_types[i])

    true_edge_set = {frozenset(e) for e in true_edges}
    pred_edge_set = {frozenset(e) for e in pred_edges}
    tp = len(true_edge_set & pred_edge_set)
    fp = len(pred_edge_set - true_edge_set)
    fn = len(true_edge_set - pred_edge_set)

    exact = (
        n_true == n_pred and n_correct_nodes == n_true and tp == len(true_edge_set) and fp == 0
    )
    return {
        "exact": exact,
        "n_true": n_true,
        "n_pred": n_pred,
        "n_correct_nodes": n_correct_nodes,
        "edge_tp": tp,
        "edge_fp": fp,
        "edge_fn": fn,
    }


def _get_dataset_id_mapping(data: list) -> dict[str, int]:
    seen = []
    for s in data:
        if s.dataset_name not in seen:
            seen.append(s.dataset_name)
    return {name: i for i, name in enumerate(seen)}


def _decode_supports_ds_ids(model: GraphVAE) -> bool:
    try:
        return "ds_ids" in inspect.signature(model.decode).parameters
    except (TypeError, ValueError):
        return False


def _decode_graphs(model: GraphVAE, z: torch.Tensor, samples: list, stochastic: bool, ds_map: dict[str, int]):
    if _decode_supports_ds_ids(model):
        ds_ids = torch.tensor([ds_map[s.dataset_name] for s in samples], dtype=torch.long, device=z.device)
        return model.decode(z, stochastic=stochastic, ds_ids=ds_ids), True
    return model.decode(z, stochastic=stochastic), False


@torch.no_grad()
def run_circuit_encoder(vae: GraphVAE, data: list, scalers: dict, device: torch.device, batch_size: int = 128, stochastic: bool = False, use_permutation_eval: bool = True) -> tuple[dict, bool, dict[str, int]]:
    vae.eval()
    by_ds: dict[str, list] = {}
    for s in data:
        by_ds.setdefault(s.dataset_name, []).append(s)

    ds_map = _get_dataset_id_mapping(data)
    decode_used_ds_ids = False
    out: dict = {}

    for ds_name, samples in by_ds.items():
        scaler = scalers[ds_name].param_scaler
        flat_attrs = None  # set from first true graph, matching sample.y/scaler order
        graph_template = None
        attr_node_indices = None

        topo_results = []
        true_scaled_rows = []
        pred_scaled_rows = []
        perm_eval_stats = {"identity_only": 0, "nontrivial": 0, "best_nonidentity": 0}

        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            z, _, _ = vae.encode(batch, scalers)
            graphs_pred, used_ds_ids = _decode_graphs(vae, z, batch, stochastic, ds_map)
            decode_used_ds_ids = decode_used_ds_ids or used_ds_ids

            for s, g_pred in zip(batch, graphs_pred):
                g_true = data_to_graph_ns(s, scaler)
                tm = topo_match(g_true.node_types, g_true.edges, g_pred.node_types, g_pred.edges)
                topo_results.append(tm)
                if flat_attrs is None:
                    flat_attrs = _unique_attr_names_from_graph(g_true)
                    graph_template = g_true
                    attr_node_indices = _attr_node_indices_from_graph(g_true)

                true_row = s.y.tolist()
                pred_row = [float("nan")] * len(flat_attrs)
                if tm["exact"] and g_pred.attrs:
                    pred_row = _scaled_attr_row_from_pred_graph(g_pred)
                    if len(pred_row) != len(flat_attrs):
                        pred_row = [float("nan")] * len(flat_attrs)

                perms = _get_attr_perm_indices(s, len(true_row)) if use_permutation_eval else [list(range(len(true_row)))]
                if len(perms) <= 1:
                    perm_eval_stats["identity_only"] += 1
                    aligned_pred_row = pred_row
                else:
                    perm_eval_stats["nontrivial"] += 1
                    aligned_pred_row, best_perm_i, _ = _best_pred_row_aligned_to_true_order(true_row, pred_row, perms)
                    if best_perm_i != 0:
                        perm_eval_stats["best_nonidentity"] += 1

                true_scaled_rows.append(true_row)
                pred_scaled_rows.append(aligned_pred_row)

        y_true_mat = np.array(true_scaled_rows, dtype=np.float64)
        y_pred_mat = np.array(pred_scaled_rows, dtype=np.float64)
        y_true_phys = scaler.inverse_transform(y_true_mat)
        y_pred_phys = scaler.inverse_transform(y_pred_mat)

        param_node_index = {
            attr: int(attr_node_indices[k])
            for k, attr in enumerate(flat_attrs)
            if not (attr == "dir" or attr.startswith("dir_n"))
        }
        out[ds_name] = {
            "topo_results": topo_results,
            "params": {attr: {"true": y_true_phys[:, k], "pred": y_pred_phys[:, k]}
                       for k, attr in enumerate(flat_attrs)
                       if not (attr == "dir" or attr.startswith("dir_n"))},
            "param_node_index": param_node_index,
            "graph_template": graph_template,
            "perm_eval_stats": perm_eval_stats,
        }

    return out, decode_used_ds_ids, ds_map


@torch.no_grad()
def run_spec_encoder(vae: GraphVAE, data: list, scalers: dict, device: torch.device, batch_size: int = 128, stochastic: bool = False, use_permutation_eval: bool = True) -> tuple[dict, bool, dict[str, int]]:
    vae.eval()
    by_ds: dict[str, list] = {}
    for s in data:
        by_ds.setdefault(s.dataset_name, []).append(s)

    ds_map = _get_dataset_id_mapping(data)
    decode_used_ds_ids = False
    out: dict = {}

    for ds_name, samples in by_ds.items():
        scaler = scalers[ds_name].param_scaler
        flat_attrs = None  # set from first true graph, matching sample.y/scaler order
        graph_template = None
        attr_node_indices = None

        topo_results = []
        true_scaled_rows = []
        pred_scaled_rows = []
        perm_eval_stats = {"identity_only": 0, "nontrivial": 0, "best_nonidentity": 0}

        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            obs_vals = torch.stack([s.obs_vals for s in batch]).to(device)
            obs_mask = torch.stack([s.obs_mask for s in batch]).to(device)
            z_s, _, _ = vae.spec_encoder.encode(obs_vals, obs_mask)
            graphs_pred, used_ds_ids = _decode_graphs(vae, z_s, batch, stochastic, ds_map)
            decode_used_ds_ids = decode_used_ds_ids or used_ds_ids

            for s, g_pred in zip(batch, graphs_pred):
                g_true = data_to_graph_ns(s, scaler)
                tm = topo_match(g_true.node_types, g_true.edges, g_pred.node_types, g_pred.edges)
                topo_results.append(tm)
                if flat_attrs is None:
                    flat_attrs = _unique_attr_names_from_graph(g_true)
                    graph_template = g_true
                    attr_node_indices = _attr_node_indices_from_graph(g_true)

                true_row = s.y.tolist()
                pred_row = [float("nan")] * len(flat_attrs)
                if tm["exact"] and g_pred.attrs:
                    pred_row = _scaled_attr_row_from_pred_graph(g_pred)
                    if len(pred_row) != len(flat_attrs):
                        pred_row = [float("nan")] * len(flat_attrs)

                perms = _get_attr_perm_indices(s, len(true_row)) if use_permutation_eval else [list(range(len(true_row)))]
                if len(perms) <= 1:
                    perm_eval_stats["identity_only"] += 1
                    aligned_pred_row = pred_row
                else:
                    perm_eval_stats["nontrivial"] += 1
                    aligned_pred_row, best_perm_i, _ = _best_pred_row_aligned_to_true_order(true_row, pred_row, perms)
                    if best_perm_i != 0:
                        perm_eval_stats["best_nonidentity"] += 1

                true_scaled_rows.append(true_row)
                pred_scaled_rows.append(aligned_pred_row)

        y_true_mat = np.array(true_scaled_rows, dtype=np.float64)
        y_pred_mat = np.array(pred_scaled_rows, dtype=np.float64)
        y_true_phys = scaler.inverse_transform(y_true_mat)
        y_pred_phys = scaler.inverse_transform(y_pred_mat)

        param_node_index = {
            attr: int(attr_node_indices[k])
            for k, attr in enumerate(flat_attrs)
            if not (attr == "dir" or attr.startswith("dir_n"))
        }
        out[ds_name] = {
            "topo_results": topo_results,
            "params": {attr: {"true": y_true_phys[:, k], "pred": y_pred_phys[:, k]}
                       for k, attr in enumerate(flat_attrs)
                       if not (attr == "dir" or attr.startswith("dir_n"))},
            "param_node_index": param_node_index,
            "graph_template": graph_template,
            "perm_eval_stats": perm_eval_stats,
        }

    return out, decode_used_ds_ids, ds_map


def r2_score(true: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(true) & np.isfinite(pred)
    if mask.sum() < 2:
        return float("nan")
    t, p = true[mask], pred[mask]
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - np.mean(t)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def rmse(true: np.ndarray, pred: np.ndarray) -> float:
    mask = np.isfinite(true) & np.isfinite(pred)
    if mask.sum() < 1:
        return float("nan")
    return float(np.sqrt(np.mean((true[mask] - pred[mask]) ** 2)))


def compute_topo_metrics(topo_results: list[dict]) -> dict:
    if not topo_results:
        return {}
    n = len(topo_results)
    exact_acc = sum(r["exact"] for r in topo_results) / n
    total_nodes_true = sum(r["n_true"] for r in topo_results)
    total_correct = sum(r["n_correct_nodes"] for r in topo_results)
    node_acc = total_correct / max(total_nodes_true, 1)
    tp = sum(r["edge_tp"] for r in topo_results)
    fp = sum(r["edge_fp"] for r in topo_results)
    fn = sum(r["edge_fn"] for r in topo_results)
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if np.isfinite(prec) and np.isfinite(rec) and (prec + rec) > 0 else float("nan")
    n_mae = np.mean([abs(r["n_pred"] - r["n_true"]) for r in topo_results])
    return {
        "topology_accuracy": exact_acc,
        "node_type_accuracy": node_acc,
        "edge_precision": prec,
        "edge_recall": rec,
        "edge_f1": f1,
        "n_nodes_mae": float(n_mae),
        "n_samples": n,
        "n_exact": sum(r["exact"] for r in topo_results),
    }


def compute_param_metrics(params: dict[str, dict[str, np.ndarray]]) -> dict[str, dict]:
    out = {}
    for attr, arrs in params.items():
        t = arrs["true"]
        p = arrs["pred"]
        mask = np.isfinite(t) & np.isfinite(p) & (t > 0) & (p > 0)
        n_valid = int(mask.sum())
        if n_valid < 2:
            out[attr] = {"r2": float("nan"), "rmse": float("nan"), "n_valid": n_valid}
            continue
        use_log = (t[mask].max() / t[mask].min()) > 100
        r2 = r2_score(np.log10(t[mask]), np.log10(p[mask])) if use_log else r2_score(t[mask], p[mask])
        rms = rmse(t[mask], p[mask])
        out[attr] = {"r2": r2, "rmse": rms, "n_valid": n_valid, "log_scale": use_log}
    return out


def _fmt(v: float, decimals: int = 4) -> str:
    if not np.isfinite(v):
        return "  n/a  "
    if abs(v) >= 1e4 or (abs(v) < 1e-3 and v != 0):
        return f"{v:10.3e}"
    return f"{v:{10}.{decimals}f}"


def print_report(branch_name: str, results: dict, decode_used_ds_ids: bool, ds_map: dict[str, int]) -> None:
    sep = "=" * 80
    print()
    print(sep)
    print(f"  RAMO: {branch_name}")
    print(sep)
    print(f"  decode con ds_ids: {decode_used_ds_ids}")
    print(f"  mapping dataset->id: {ds_map}")

    for ds_name in sorted(results):
        r = results[ds_name]
        tm_metrics = compute_topo_metrics(r["topo_results"])
        pm_metrics = compute_param_metrics(r["params"])

        print(f"\n  Dataset: {ds_name}  (N={tm_metrics.get('n_samples', 0)})")
        print("  " + "-" * 60)
        print("  Metriche topologiche:")
        print(f"    Topology Accuracy  : {tm_metrics['topology_accuracy']:.4f}  ({tm_metrics['n_exact']}/{tm_metrics['n_samples']} corrette)")
        print(f"    Node-type Accuracy : {tm_metrics['node_type_accuracy']:.4f}")
        print(f"    Edge Precision     : {_fmt(tm_metrics['edge_precision'])}")
        print(f"    Edge Recall        : {_fmt(tm_metrics['edge_recall'])}")
        print(f"    Edge F1            : {_fmt(tm_metrics['edge_f1'])}")
        print(f"    nNodes MAE         : {tm_metrics['n_nodes_mae']:.3f}")
        ps = r.get("perm_eval_stats", {})
        if ps:
            print(
                "  Permutazioni parametri eval: "
                f"identity_only={ps.get('identity_only', 0)}  "
                f"nontrivial={ps.get('nontrivial', 0)}  "
                f"best_nonidentity={ps.get('best_nonidentity', 0)}"
            )
        print("  Metriche parametri  (solo topologia corretta):")
        print(f"    {'Attr':<10} {'R²':>10} {'RMSE':>14} {'N_valid':>8}  scala")
        print("    " + "-" * 48)
        for attr, m in pm_metrics.items():
            unit = _unit(attr)
            log_label = "(log10)" if m.get("log_scale") else "      "
            print(f"    {attr:<10} {_fmt(m['r2']):>10} {_fmt(m['rmse']):>10} {unit:<4} {m['n_valid']:>6}  {log_label}")
    print()


def _fmt_title(v: float) -> str:
    if not np.isfinite(v):
        return "n/a"
    if abs(v) >= 1000 or (abs(v) < 0.01 and v != 0):
        return f"{v:.2e}"
    return f"{v:.3g}"


def plot_scatter(results: dict, out_path: str, n_samples: int, branch: str, ckpt_path: str) -> None:
    ds_names = sorted(results)
    all_attrs = []
    seen = set()
    for ds in ds_names:
        for attr in results[ds]["params"]:
            if attr not in seen:
                all_attrs.append(attr)
                seen.add(attr)

    n_rows = len(ds_names)
    n_cols = len(all_attrs)
    if n_rows == 0 or n_cols == 0:
        print("  Nessun dato da plottare.")
        return

    fig = plt.figure(figsize=(3.6 * n_cols, 3.8 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.55, wspace=0.40)

    for row_i, ds_name in enumerate(ds_names):
        color = DS_COLORS[row_i % len(DS_COLORS)]
        params = results[ds_name]["params"]
        for col_j, attr in enumerate(all_attrs):
            ax = fig.add_subplot(gs[row_i, col_j])
            if attr not in params:
                ax.axis("off")
                continue
            t_phys = params[attr]["true"]
            p_phys = params[attr]["pred"]
            mask = np.isfinite(t_phys) & np.isfinite(p_phys) & (t_phys > 0) & (p_phys > 0)
            t = t_phys[mask]
            p = p_phys[mask]
            if len(t) == 0:
                ax.text(0.5, 0.5, "nessun dato\n(topo errata)", ha="center", va="center", transform=ax.transAxes, fontsize=7.5, color="gray")
                ax.set_title(f"{ds_name[:22]}\n{attr}", fontsize=7)
                continue
            if len(t) > n_samples:
                idx = np.random.choice(len(t), n_samples, replace=False)
                t = t[idx]
                p = p[idx]
            use_log = (t.max() / t.min()) > 100
            if use_log:
                tv, pv = np.log10(t), np.log10(p)
                ax_label = f"log10 {attr} [{_unit(attr)}]"
            else:
                tv, pv = t, p
                ax_label = f"{attr} [{_unit(attr)}]"
            ax.scatter(tv, pv, s=5, alpha=0.30, color=color, linewidths=0, rasterized=True)
            lo = min(tv.min(), pv.min())
            hi = max(tv.max(), pv.max())
            pad = (hi - lo) * 0.05 or 0.1
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", lw=0.8, ls="--", zorder=3)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_aspect("equal", adjustable="box")
            r2 = r2_score(tv, pv)
            rms = rmse(t, p)
            unit = _unit(attr)
            short_ds = ds_name.replace("_", " ")
            if len(short_ds) > 26:
                short_ds = short_ds[:24] + "…"
            ax.set_title(f"{short_ds} — {attr}\nR²={_fmt_title(r2)}   RMSE={_fmt_title(rms)} {unit}\nN={len(t)}", fontsize=7, pad=2)
            ax.set_xlabel(f"vero\n{ax_label}", fontsize=6.5)
            ax.set_ylabel("predetto", fontsize=6.5)
            ax.tick_params(labelsize=6)

    fig.suptitle(f"GraphVAE — predetto vs. vero  |  ramo={branch}  |  {Path(ckpt_path).name}", fontsize=11, y=1.01)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Plot salvato -> {out_path}")


def plot_topo_breakdown(results: dict, out_path: str, branch: str, ckpt_path: str) -> None:
    ds_names = sorted(results)
    n_ds = len(ds_names)
    if n_ds == 0:
        return
    fig, axes = plt.subplots(n_ds, 2, figsize=(9, 2.8 * n_ds), squeeze=False)
    fig.suptitle(f"GraphVAE — analisi errori topologici  |  ramo={branch}  |  {Path(ckpt_path).name}", fontsize=10, y=1.01)

    for row_i, ds_name in enumerate(ds_names):
        topo_results = results[ds_name]["topo_results"]
        tm = compute_topo_metrics(topo_results)
        color = DS_COLORS[row_i % len(DS_COLORS)]

        ax_left = axes[row_i][0]
        metric_names = ["Topology\nAccuracy", "Node-type\nAccuracy", "Edge\nPrecision", "Edge\nRecall", "Edge\nF1"]
        metric_vals = [tm["topology_accuracy"], tm["node_type_accuracy"], tm["edge_precision"], tm["edge_recall"], tm["edge_f1"]]
        bars = ax_left.bar(range(len(metric_names)), [v if np.isfinite(v) else 0 for v in metric_vals], color=color, alpha=0.8, edgecolor="white")
        ax_left.set_xticks(range(len(metric_names)))
        ax_left.set_xticklabels(metric_names, fontsize=7.5)
        ax_left.set_ylim(0, 1.05)
        ax_left.set_ylabel("valore metrica", fontsize=8)
        ax_left.set_title(f"{ds_name.replace('_', ' ')} (N={tm['n_samples']})", fontsize=9, pad=4)
        for bar, val in zip(bars, metric_vals):
            if np.isfinite(val):
                ax_left.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)
        ax_left.spines[["top", "right"]].set_visible(False)

        ax_right = axes[row_i][1]
        delta_n = [r["n_pred"] - r["n_true"] for r in topo_results]
        bins = np.arange(min(delta_n) - 0.5, max(delta_n) + 1.5)
        ax_right.hist(delta_n, bins=bins, color=color, alpha=0.75, edgecolor="white", linewidth=0.4)
        ax_right.axvline(0, color="black", lw=1.2, ls="--")
        ax_right.set_xlabel("nNodes predetti - nNodes veri", fontsize=8)
        ax_right.set_ylabel("conteggio", fontsize=8)
        ax_right.set_title(f"Distribuzione errore nNodes\nMAE={tm['n_nodes_mae']:.3f}", fontsize=9, pad=4)
        ax_right.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Plot topologia salvato -> {out_path}")



# ===========================================================================
# Graph drawing utilities (networkx)
# ===========================================================================

_SUBGTYPE_LABEL: dict[int, str] = {}
_SUBGTYPE_COLOR: dict[int, str] = {}

def _init_node_style():
    if _SUBGTYPE_LABEL:
        return
    _STYLE = {
        "TRANSMON":  ("T",   "#4C78A8"),
        "RESONATOR": ("R",   "#54A24B"),
        "C_COUPLER": ("Cc",  "#F58518"),
        "I_COUPLER": ("Ic",  "#E45756"),
        "FEEDLINE":  ("F",   "#B279A2"),
        "TCT":       ("TCT", "#76B7B2"),
        "RCT":       ("RCT", "#EDC948"),
        "RC":        ("RC",  "#FF9DA7"),
        "RI":        ("RI",  "#9C755F"),
    }
    for st in SubgType:
        name = st.name
        label, color = _STYLE.get(name, (name[:3], "#AAAAAA"))
        _SUBGTYPE_LABEL[int(st)] = label
        _SUBGTYPE_COLOR[int(st)] = color

_init_node_style()


def _graph_ns_to_nx(g):
    G = nx.Graph()
    for i, st_int in enumerate(g.node_types):
        label = _SUBGTYPE_LABEL.get(st_int, "?")
        attr_parts = []
        if hasattr(g, "attrs") and i < len(g.attrs) and g.attrs[i]:
            for k, v in g.attrs[i].items():
                attr_parts.append(f"{k}={v:.2e}" if isinstance(v, float) else f"{k}={v}")
        G.add_node(i, st_int=st_int, label=label, attr_str="\n".join(attr_parts))
    for u, v in g.edges:
        G.add_edge(u, v)
    return G


def _draw_single_graph(ax, g, title="", show_attrs=True, title_color="black"):
    if not _HAS_NX:
        ax.text(0.5, 0.5, "networkx non disponibile", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="red")
        return
    if not g.node_types:
        ax.text(0.5, 0.5, "(grafo vuoto)", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="gray")
        ax.set_title(title, fontsize=8, color=title_color, pad=3)
        ax.axis("off")
        return
    G   = _graph_ns_to_nx(g)
    n   = len(G.nodes)
    pos = nx.spring_layout(G, seed=42) if n > 2 else nx.shell_layout(G)
    node_colors = [_SUBGTYPE_COLOR.get(G.nodes[i]["st_int"], "#AAAAAA") for i in G.nodes]
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#555555", width=1.5, alpha=0.7)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=800, alpha=0.92)
    if show_attrs:
        labels = {i: G.nodes[i]["label"] + ("\n" + G.nodes[i]["attr_str"] if G.nodes[i]["attr_str"] else "") for i in G.nodes}
    else:
        labels = {i: G.nodes[i]["label"] for i in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=6, font_color="white", font_weight="bold")
    ax.set_title(title, fontsize=8, color=title_color, pad=3)
    ax.axis("off")


def _cqed_topology_to_nx(topology: CQEDTopology):
    G = nx.Graph()
    nodes = list(getattr(topology, "_nodes", []))
    for n in nodes:
        st = getattr(n, "subg_type", None)
        st_int = int(st) if st is not None else -1
        st_name = getattr(st, "name", str(st))
        attr_parts = []
        for k, v in getattr(n, "attrs", {}).items():
            if k == "dir":
                attr_parts.append(f"{k}={v}")
            elif isinstance(v, float):
                attr_parts.append(f"{k}={v:.2e}")
            else:
                attr_parts.append(f"{k}={v}")
        label = f"{getattr(n, 'node_id', '?')}: {st_name}"
        if getattr(n, "label", None):
            label += f"\n{n.label}"
        if attr_parts:
            label += "\n" + "\n".join(attr_parts[:4])
        G.add_node(getattr(n, "node_id", len(G.nodes)), st_int=st_int, label=label)
    for u, v in getattr(topology, "_edges", []):
        G.add_edge(int(u), int(v))
    return G


def _draw_cqed_topology(ax, topology: CQEDTopology | None, title: str = "", title_color: str = "black") -> None:
    if not _HAS_NX:
        ax.text(0.5, 0.5, "networkx non disponibile", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="red")
        ax.axis("off")
        return
    if topology is None or not getattr(topology, "_nodes", None):
        ax.text(0.5, 0.5, "(grafo vuoto)", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="gray")
        ax.set_title(title, fontsize=9, color=title_color)
        ax.axis("off")
        return
    G = _cqed_topology_to_nx(topology)
    pos = nx.spring_layout(G, seed=42) if len(G.nodes) > 2 else nx.shell_layout(G)
    node_colors = [_SUBGTYPE_COLOR.get(G.nodes[i].get("st_int", -1), "#AAAAAA") for i in G.nodes]
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#444444", width=1.7, alpha=0.78)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=1450, alpha=0.94,
                           edgecolors="white", linewidths=1.2)
    nx.draw_networkx_labels(G, pos, labels={i: G.nodes[i]["label"] for i in G.nodes},
                            ax=ax, font_size=6.0, font_weight="bold")
    ax.set_title(title, fontsize=9, color=title_color, pad=4)
    ax.axis("off")




def _try_expand_topology_for_plot(topology: CQEDTopology | None) -> tuple[CQEDTopology | None, str | None]:
    """Expand a macro CQEDTopology for NetworkX debug plots without crashing evaluation."""
    if topology is None:
        return None, "topology_none"
    try:
        return expand_topology(topology, validate=False), None
    except Exception as exc:
        return None, f"expand_failed: {type(exc).__name__}: {exc}"

def _make_debug_topologies_for_sample(s, g_pred, param_scaler, obs_scaler=None, qu=None,
                                      f_min: float = 1.0, f_max: float = 9.0,
                                      use_permutation_eval: bool = True) -> tuple[CQEDTopology, CQEDTopology | None, CQEDTopology | None, str | None]:
    """Build TRUE macro, aligned PRED macro and expanded PRED primitive for debug plots."""
    g_true_scaled = data_to_graph_ns(s, None)
    true_topo = _graph_ns_to_cqed_topology(g_true_scaled, name=f"{s.dataset_name}_true_macro")
    tm = topo_match(g_true_scaled.node_types, g_true_scaled.edges, g_pred.node_types, g_pred.edges)

    pred_topo = None
    expanded = None
    reason = None
    if not tm["exact"]:
        pred_topo = _graph_ns_to_cqed_topology(g_pred, name=f"{s.dataset_name}_pred_macro_raw")
        try:
            expanded = expand_topology(pred_topo, validate=False)
        except Exception:
            expanded = None
        return true_topo, pred_topo, expanded, "topology_not_exact"

    true_row_scaled = s.y.tolist()
    pred_row_scaled = _scaled_attr_row_from_pred_graph(g_pred) if getattr(g_pred, "attrs", None) else []
    if len(pred_row_scaled) != len(true_row_scaled):
        pred_topo = _graph_ns_to_cqed_topology(g_pred, name=f"{s.dataset_name}_pred_macro_raw")
        try:
            expanded = expand_topology(pred_topo, validate=False)
        except Exception:
            expanded = None
        return true_topo, pred_topo, expanded, "attr_length_mismatch"

    if use_permutation_eval:
        perms = _get_attr_perm_indices(s, len(true_row_scaled))
        pred_row_scaled, _, _ = _best_pred_row_aligned_to_true_order(true_row_scaled, pred_row_scaled, perms)

    pred_row_phys = _inverse_scaled_row_safe(param_scaler, pred_row_scaled)
    g_pred_phys = _graph_from_flat_attrs_like(g_true_scaled, pred_row_phys, "pred_aligned")
    pred_topo = _graph_ns_to_cqed_topology(g_pred_phys, name=f"{s.dataset_name}_pred_macro_aligned")
    try:
        expanded = expand_topology(pred_topo, validate=False)
    except Exception as exc:
        return true_topo, pred_topo, None, f"expand_failed: {type(exc).__name__}: {exc}"

    if qu is not None and obs_scaler is not None:
        pred_obs = _simulate_topology_arrays(pred_topo, qu, f_min, f_max)
        if not pred_obs.get("ok"):
            reason = pred_obs.get("error") or "pred_qultra_failed"
    return true_topo, pred_topo, expanded, reason


@torch.no_grad()
def plot_expansion_debug_cases(vae, data, scalers, results: dict, out_dir: Path, device,
                               branch: str = "circuit", batch_size: int = 128,
                               stochastic: bool = False, use_permutation_eval: bool = True,
                               f_min: float = 1.0, f_max: float = 9.0,
                               seed: int = 42, max_scan_per_dataset: int = 50) -> None:
    """For each dataset, plot TRUE macro, final PRED macro, and expanded PRED primitive.

    If a dataset has failed samples, the first failed sample encountered is plotted.
    If no failed sample is found, one deterministic pseudo-random valid sample is plotted.
    """
    if not _HAS_NX:
        print("  plot_expansion_debug_cases: networkx non disponibile, skip.")
        return
    qu = None
    if _HAS_QULTRA_WORKFLOW:
        try:
            qu = import_qultra()
        except Exception:
            qu = None

    by_ds: dict[str, list] = {}
    for s in data:
        by_ds.setdefault(s.dataset_name, []).append(s)

    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir) / "expansion_debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    vae.eval()

    for ds_name, samples in sorted(by_ds.items()):
        dataset_scalers = scalers[ds_name]
        param_scaler = dataset_scalers.param_scaler
        obs_scaler = getattr(dataset_scalers, "obs_scaler", None)
        preferred_error = bool(results.get(ds_name, {}).get("fail_reasons"))
        chosen = None
        fallback = None
        scan_samples = samples[:max_scan_per_dataset]
        if not preferred_error and len(samples) > 0:
            start = int(rng.integers(0, max(1, len(samples))))
            scan_samples = samples[start:start + max_scan_per_dataset] + samples[:max(0, max_scan_per_dataset - len(samples[start:start + max_scan_per_dataset]))]

        for i in range(0, len(scan_samples), batch_size):
            batch = scan_samples[i:i + batch_size]
            if branch == "spec":
                obs_v = torch.stack([s.obs_vals for s in batch]).to(device)
                obs_m = torch.stack([s.obs_mask for s in batch]).to(device)
                z, _, _ = vae.spec_encoder.encode(obs_v, obs_m)
            else:
                z, _, _ = vae.encode(batch, scalers)
            graphs_pred, _ = _decode_graphs(vae, z, batch, stochastic, _get_dataset_id_mapping(data))
            for sample, g_pred in zip(batch, graphs_pred):
                true_topo, pred_topo, expanded, reason = _make_debug_topologies_for_sample(
                    sample, g_pred, param_scaler, obs_scaler=obs_scaler, qu=qu,
                    f_min=f_min, f_max=f_max, use_permutation_eval=use_permutation_eval,
                )
                item = (true_topo, pred_topo, expanded, reason, sample)
                if fallback is None:
                    fallback = item
                if reason is not None:
                    chosen = item
                    break
            if chosen is not None:
                break

        if chosen is None:
            chosen = fallback
        if chosen is None:
            continue

        true_topo, pred_topo, expanded, reason, sample = chosen
        true_expanded, true_exp_reason = _try_expand_topology_for_plot(true_topo)
        if expanded is None and pred_topo is not None:
            expanded, pred_exp_reason = _try_expand_topology_for_plot(pred_topo)
            if reason is None and pred_exp_reason is not None:
                reason = pred_exp_reason
        is_error = reason is not None

        # Pannello NetworkX: macro TRUE, primitiva TRUE espansa, macro PRED, primitiva PRED espansa.
        # Il terzo/quarto pannello sono quelli più utili per verificare l'assegnazione dei link
        # dopo expand_topology().
        fig, axes = plt.subplots(1, 4, figsize=(20, 5.0))
        _draw_cqed_topology(axes[0], true_topo, "macro originale / TRUE")
        _draw_cqed_topology(axes[1], true_expanded, "TRUE primitivo espanso",
                            title_color="#E74C3C" if true_exp_reason else "#2C3E50")
        _draw_cqed_topology(axes[2], pred_topo, "macro finale / PRED",
                            title_color="#E74C3C" if is_error else "#27AE60")
        _draw_cqed_topology(axes[3], expanded, "PRED primitivo espanso",
                            title_color="#E74C3C" if is_error else "#27AE60")
        reason_text = reason if reason is not None else "nessun errore trovato: campione scelto a caso"
        if true_exp_reason:
            reason_text += f" | true_expand={true_exp_reason}"
        fig.suptitle(
            f"Expansion debug | {ds_name} | ramo={branch} | {reason_text}",
            fontsize=11, y=1.03,
        )
        plt.tight_layout()
        out_path = out_dir / f"{_safe_name(ds_name)}_{branch}_primitive_expansion_debug.png"
        plt.savefig(out_path, dpi=145, bbox_inches="tight")
        plt.close()
        print(f"  Plot primitive expansion debug salvato -> {out_path}")



@torch.no_grad()
def plot_topology_errors(vae, data, scalers, ds_name, out_path, device,
                         n_show=20, batch_size=128, stochastic=False, branch="circuit"):
    if not _HAS_NX:
        print("  plot_topology_errors: networkx non disponibile, skip.")
        return
    samples = [s for s in data if s.dataset_name == ds_name]
    if not samples:
        print(f"  Nessun campione trovato per dataset '{ds_name}', skip.")
        return
    pairs = []
    vae.eval()
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        if branch == "spec":
            obs_v = torch.stack([s.obs_vals for s in batch]).to(device)
            obs_m = torch.stack([s.obs_mask for s in batch]).to(device)
            z, _, _ = vae.spec_encoder.encode(obs_v, obs_m)
        else:
            z, _, _ = vae.encode(batch, scalers)
        graphs_pred = vae.decode(z, stochastic=stochastic)
        scaler = scalers[ds_name].param_scaler
        for s, g_pred in zip(batch, graphs_pred):
            g_true = data_to_graph_ns(s, scaler)
            pairs.append((g_true, g_pred))
        if len(pairs) >= n_show:
            break
    pairs = pairs[:n_show]
    n_cols = min(len(pairs), 10)
    n_rows_groups = (len(pairs) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows_groups * 2, n_cols, figsize=(2.8 * n_cols, 3.2 * n_rows_groups * 2))
    if n_rows_groups * 2 == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows_groups * 2 == 1:
        axes = [list(axes)]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]
    else:
        axes = [list(row) for row in axes]
    for r in axes:
        for ax in r:
            ax.axis("off")
    for idx, (g_true, g_pred) in enumerate(pairs):
        group = idx // n_cols
        col   = idx % n_cols
        tm    = topo_match(g_true.node_types, g_true.edges, g_pred.node_types, g_pred.edges)
        pred_color = "#27AE60" if tm["exact"] else "#E74C3C"
        _draw_single_graph(axes[group * 2][col],     g_true,  title=f"TRUE #{idx+1}",  show_attrs=False)
        _draw_single_graph(axes[group * 2 + 1][col], g_pred,
                           title=f"PRED #{idx+1}\nnodes:{g_pred.node_types}",
                           show_attrs=False, title_color=pred_color)
    n_exact = sum(topo_match(t.node_types, t.edges, p.node_types, p.edges)["exact"] for t, p in pairs)
    fig.suptitle(f"Topologia: {ds_name.replace('_', ' ')}  |  ramo={branch}\n"
                 f"Corretti: {n_exact}/{len(pairs)}  (mai visto in training)",
                 fontsize=10, y=1.01)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Plot topologia errori salvato -> {out_path}")


@torch.no_grad()
def plot_latent_samples(vae, out_path, device, n_show=20, stochastic=True, seed=42):
    if not _HAS_NX:
        print("  plot_latent_samples: networkx non disponibile, skip.")
        return
    torch.manual_seed(seed)
    z      = torch.randn(n_show, vae.nz, device=device)
    graphs = vae.decode(z, stochastic=stochastic)
    topo_counter = {}
    for g in graphs:
        key = str(sorted(g.node_types))
        topo_counter[key] = topo_counter.get(key, 0) + 1
    n_cols = min(n_show, 5)
    n_rows = (n_show + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 4.0 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [list(axes)]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]
    else:
        axes = [list(row) for row in axes]
    for r in axes:
        for ax in r:
            ax.axis("off")
    for idx, g in enumerate(graphs):
        row = idx // n_cols
        col = idx % n_cols
        type_names = [_SUBGTYPE_LABEL.get(t, "?") for t in g.node_types]
        _draw_single_graph(axes[row][col], g,
                           title=f"sample #{idx+1}\n[{', '.join(type_names)}]",
                           show_attrs=True)
    summary_lines = [f"{k}: {v}" for k, v in sorted(topo_counter.items(), key=lambda x: -x[1])]
    summary = "Topologie: " + "  |  ".join(summary_lines[:6])
    fig.suptitle(f"GraphVAE — campionamento dal latent space  z ~ N(0,I)\n{summary}",
                 fontsize=10, y=1.01)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Plot campionamento latent space salvato -> {out_path}")



def _node_r2_summary_for_dataset(ds_result: dict) -> dict[int, dict]:
    """Aggregate parameter R² values by node for one dataset result."""
    param_node_index = ds_result.get("param_node_index", {})
    by_node: dict[int, list[tuple[str, float, int]]] = defaultdict(list)
    for attr, arrs in ds_result.get("params", {}).items():
        if attr not in param_node_index:
            continue
        t = np.asarray(arrs["true"], dtype=np.float64)
        p = np.asarray(arrs["pred"], dtype=np.float64)
        mask = np.isfinite(t) & np.isfinite(p) & (t > 0) & (p > 0)
        n_valid = int(mask.sum())
        if n_valid < 2:
            r2 = float("nan")
        else:
            use_log = (t[mask].max() / t[mask].min()) > 100
            r2 = r2_score(np.log10(t[mask]), np.log10(p[mask])) if use_log else r2_score(t[mask], p[mask])
        by_node[int(param_node_index[attr])].append((attr, r2, n_valid))

    out = {}
    for node_i, vals in by_node.items():
        finite = [r2 for _attr, r2, _n in vals if np.isfinite(r2)]
        out[node_i] = {
            "mean_r2": float(np.mean(finite)) if finite else float("nan"),
            "attrs": vals,
        }
    return out


def _r2_to_color(r2: float) -> str:
    """Small traffic-light palette for node-level R² diagnostics."""
    if not np.isfinite(r2):
        return "#BDBDBD"
    if r2 >= 0.90:
        return "#2E7D32"
    if r2 >= 0.70:
        return "#7CB342"
    if r2 >= 0.40:
        return "#F9A825"
    if r2 >= 0.00:
        return "#EF6C00"
    return "#C62828"


def _draw_r2_graph(ax, g, node_summary: dict[int, dict], title: str = "") -> None:
    if not _HAS_NX:
        ax.text(0.5, 0.5, "networkx non disponibile", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="red")
        ax.axis("off")
        return
    if g is None or not getattr(g, "node_types", None):
        ax.text(0.5, 0.5, "nessun grafo template", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="gray")
        ax.axis("off")
        return

    G = nx.Graph()
    for i, st_int in enumerate(g.node_types):
        base_label = _SUBGTYPE_LABEL.get(int(st_int), str(st_int))
        summary = node_summary.get(i, {"mean_r2": float("nan"), "attrs": []})
        mean_r2 = summary["mean_r2"]
        attr_lines = []
        for attr, r2, n_valid in summary.get("attrs", []):
            attr_lines.append(f"{attr}:{_fmt_title(r2)}")
        if len(attr_lines) > 4:
            attr_lines = attr_lines[:4] + ["…"]
        label = f"{i}: {base_label}\nR²={_fmt_title(mean_r2)}"
        if attr_lines:
            label += "\n" + "\n".join(attr_lines)
        G.add_node(i, label=label, mean_r2=mean_r2)
    for u, v in g.edges:
        G.add_edge(int(u), int(v))

    pos = nx.spring_layout(G, seed=42) if len(G.nodes) > 2 else nx.shell_layout(G)
    node_colors = [_r2_to_color(G.nodes[i]["mean_r2"]) for i in G.nodes]
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#555555", width=1.7, alpha=0.70)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=1500, alpha=0.95,
                           edgecolors="white", linewidths=1.2)
    nx.draw_networkx_labels(G, pos, labels={i: G.nodes[i]["label"] for i in G.nodes},
                            ax=ax, font_size=6.2, font_weight="bold")
    ax.set_title(title, fontsize=8.5, pad=4)
    ax.axis("off")


def plot_param_r2_graph(results: dict, out_path: str, branch: str, ckpt_path: str) -> None:
    """Draw one NetworkX template graph per dataset with node-level parameter R².

    Each node shows the mean R² across its physical parameters plus the per-attr
    R² values. Metrics are computed only where predictions are finite, so wrong
    topologies naturally appear as missing/low-valid parameter predictions.
    """
    if not _HAS_NX:
        print("  plot_param_r2_graph: networkx non disponibile, skip.")
        return
    ds_names = sorted(results)
    if not ds_names:
        return

    n_cols = min(3, len(ds_names))
    n_rows = (len(ds_names) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 4.0 * n_rows), squeeze=False)
    for row in axes:
        for ax in row:
            ax.axis("off")

    for idx, ds_name in enumerate(ds_names):
        ax = axes[idx // n_cols][idx % n_cols]
        ds_result = results[ds_name]
        summary = _node_r2_summary_for_dataset(ds_result)
        tm = compute_topo_metrics(ds_result.get("topo_results", []))
        title_ds = ds_name.replace("_", " ")
        if len(title_ds) > 32:
            title_ds = title_ds[:30] + "…"
        title = (f"{title_ds}\n"
                 f"topo acc={tm.get('topology_accuracy', float('nan')):.3f} | "
                 f"exact={tm.get('n_exact', 0)}/{tm.get('n_samples', 0)}")
        _draw_r2_graph(ax, ds_result.get("graph_template"), summary, title=title)

    legend_items = [
        ("≥0.90", "#2E7D32"), ("0.70–0.90", "#7CB342"),
        ("0.40–0.70", "#F9A825"), ("0–0.40", "#EF6C00"),
        ("<0", "#C62828"), ("n/a", "#BDBDBD"),
    ]
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=lab,
                          markerfacecolor=col, markersize=8)
               for lab, col in legend_items]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8,
               title="R² medio per nodo", title_fontsize=8)
    fig.suptitle(f"GraphVAE — NetworkX con R² sui nodi  |  ramo={branch}  |  {Path(ckpt_path).name}",
                 fontsize=11, y=1.02)
    plt.tight_layout(rect=(0, 0.07, 1, 1))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  Plot NetworkX R² nodi salvato -> {out_path}")


# ===========================================================================
# Qultra observable evaluation utilities
# ===========================================================================

def _graph_ns_to_cqed_topology(g, name: str = "graph") -> CQEDTopology:
    """Convert decoder SimpleNamespace graph -> CQEDTopology with current attrs."""
    topo = CQEDTopology(name)
    nodes = []
    for i, st_int in enumerate(g.node_types):
        st = SubgType(int(st_int))
        attrs = dict(g.attrs[i]) if hasattr(g, "attrs") and i < len(g.attrs) else {}
        label = f"{st.name}_{i}"
        nodes.append(topo.add_node(st, attrs, label=label))
    for u, v in getattr(g, "edges", []):
        if 0 <= int(u) < len(nodes) and 0 <= int(v) < len(nodes) and int(u) != int(v):
            try:
                topo.add_edge(nodes[int(u)], nodes[int(v)])
            except Exception:
                pass
    return topo


def _graph_from_flat_attrs_like(g_template, flat_values, name: str):
    """Build graph namespace with physical attrs from a flat continuous row.

    ``flat_values`` follows sample.y/scaler order and therefore excludes ``dir``.
    Direction is copied from the template graph attrs so Qultra expansion still
    receives attrs["dir"] = +/-1 for directional macro-nodes.
    """
    out = SimpleNamespace()
    out.node_types = list(g_template.node_types)
    out.edges = [tuple(e) for e in g_template.edges]
    out.attrs = []
    cursor = 0
    flat_values = list(flat_values)
    for node_i, st_int in enumerate(out.node_types):
        st = SubgType(int(st_int))
        d = {}
        template_attrs = g_template.attrs[node_i] if hasattr(g_template, "attrs") and node_i < len(g_template.attrs) else {}
        for attr_name in SUBG_DEFS[st].attrs:
            if attr_name == "dir":
                d["dir"] = float(template_attrs.get("dir", 0.0))
                continue
            d[attr_name] = float(flat_values[cursor]) if cursor < len(flat_values) else float("nan")
            cursor += 1
        out.attrs.append(d)
    return out


def _inverse_scaled_row_safe(scaler, row):
    row = np.asarray(row, dtype=np.float64).reshape(1, -1)
    return scaler.inverse_transform(row)[0]


def _to_float_array(values, n_expected: int) -> np.ndarray:
    """Convert list/array possibly containing '-' to float array padded with NaN."""
    out = []
    if values is None:
        values = []
    try:
        iterable = list(np.asarray(values, dtype=object).ravel())
    except Exception:
        iterable = []
    for v in iterable[:n_expected]:
        try:
            out.append(float(v))
        except Exception:
            out.append(np.nan)
    while len(out) < n_expected:
        out.append(np.nan)
    return np.asarray(out, dtype=np.float64)


def _to_float_matrix(values, n_expected: int) -> np.ndarray:
    if n_expected <= 0:
        return np.zeros((0, 0), dtype=np.float64)
    mat = np.full((n_expected, n_expected), np.nan, dtype=np.float64)
    if values is None:
        return mat
    try:
        arr = np.asarray(values, dtype=object)
        if arr.ndim == 1:
            # flat chi vector -> square when possible
            side = int(np.sqrt(arr.size))
            if side * side == arr.size:
                arr = arr.reshape(side, side)
            else:
                flat = arr.ravel()[: n_expected * n_expected]
                for k, v in enumerate(flat):
                    try:
                        mat.ravel()[k] = float(v)
                    except Exception:
                        pass
                return mat
        r = min(n_expected, arr.shape[0])
        c = min(n_expected, arr.shape[1])
        for i in range(r):
            for j in range(c):
                try:
                    mat[i, j] = float(arr[i, j])
                except Exception:
                    pass
    except Exception:
        pass
    return mat


def _sample_true_observables_from_dataset(s, obs_scaler) -> dict:
    """Return true observables already stored in the dataset, in raw physical units.

    The loader stores `obs_vals` scaled and `obs_mask` as presence flags.  This
    function inverts the observable scaler and reconstructs the compact arrays
    used by the qultra comparison:

      - frequencies: [f_1, f_2, ...] in the same mode order used by the dataset
      - kappa:       [kappa_1, kappa_2, ...], NaN where absent
      - chi:         symmetric matrix from chi_ij slots, NaN where absent

    This avoids re-running qultra on the true circuit, because the dataset was
    generated with those observables already computed.
    """
    vals_scaled = np.asarray(s.obs_vals.detach().cpu().numpy(), dtype=np.float64)
    mask = np.asarray(s.obs_mask.detach().cpu().numpy(), dtype=np.float64) > 0.5

    raw = np.full_like(vals_scaled, np.nan, dtype=np.float64)
    raw[mask] = vals_scaled[mask] * obs_scaler.std_[mask] + obs_scaler.mean_[mask]

    def slot_value(slot: str) -> float:
        if slot not in OBS_SLOTS:
            return np.nan
        idx = OBS_SLOTS.index(slot)
        if idx >= len(raw) or not mask[idx]:
            return np.nan
        return float(raw[idx])

    # Frequencies define the mode count and the dataset mode ordering.
    f_items = []
    for slot in OBS_SLOTS:
        if slot.startswith("f_"):
            try:
                k = int(slot.split("_", 1)[1])
            except Exception:
                continue
            v = slot_value(slot)
            if np.isfinite(v):
                f_items.append((k, v))
    f_items.sort(key=lambda x: x[0])
    frequencies = np.asarray([v for _, v in f_items], dtype=np.float64)
    n_modes = int(len(frequencies))

    kappa = np.full(n_modes, np.nan, dtype=np.float64)
    for i in range(n_modes):
        v = slot_value(f"kappa_{i + 1}")
        if np.isfinite(v):
            kappa[i] = v

    chi = np.full((n_modes, n_modes), np.nan, dtype=np.float64)
    for slot in OBS_SLOTS:
        if not slot.startswith("chi_"):
            continue
        try:
            ij = slot.split("_", 1)[1]
            if len(ij) != 2:
                continue
            i = int(ij[0]) - 1
            j = int(ij[1]) - 1
        except Exception:
            continue
        if i < n_modes and j < n_modes:
            v = slot_value(slot)
            if np.isfinite(v):
                chi[i, j] = v
                chi[j, i] = v

    return {
        "ok": True,
        "error": None,
        "n_expected": n_modes,
        "frequencies": frequencies,
        "kappa": kappa,
        "chi": chi,
        "source": "dataset",
    }



def _safe_obj_name(obj) -> str:
    try:
        return getattr(obj, "name", None) or getattr(obj, "_name", None) or str(obj)
    except Exception:
        return str(obj)


def _describe_graph_ns(g, title: str = "GraphNS") -> str:
    lines = [f"# {title}"]
    lines.append("nodes:")
    for i, st_int in enumerate(getattr(g, "node_types", [])):
        try:
            st_name = SubgType(int(st_int)).name
        except Exception:
            st_name = str(st_int)
        attrs = getattr(g, "attrs", [])
        a = attrs[i] if i < len(attrs) else {}
        lines.append(f"  {i}: {st_name} attrs={a}")
    lines.append("edges:")
    for e in getattr(g, "edges", []):
        lines.append(f"  {tuple(e)}")
    return "\n".join(lines)


def _describe_cqed_topology_text(topology: CQEDTopology, title: str = "CQEDTopology") -> str:
    lines = [f"# {title}: {_safe_obj_name(topology)}"]
    nodes = list(getattr(topology, "_nodes", []))
    edges = list(getattr(topology, "_edges", []))
    adj = {getattr(n, "node_id", i): [] for i, n in enumerate(nodes)}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    lines.append("nodes:")
    for n in nodes:
        st = getattr(getattr(n, "subg_type", None), "name", str(getattr(n, "subg_type", None)))
        lines.append(
            f"  id={getattr(n,'node_id', '?')} label={getattr(n,'label','')} "
            f"type={st} attrs={getattr(n,'attrs',{})} neighbors={adj.get(getattr(n,'node_id', '?'), [])}"
        )
    lines.append("edges:")
    for e in edges:
        lines.append(f"  {tuple(e)}")
    return "\n".join(lines)


def _qelements_to_text(qelements) -> str:
    lines = ["net = ["]
    for el in qelements or []:
        for line in str(el).splitlines():
            lines.append("    " + line)
        lines[-1] += ","
    lines.append("]")
    return "\n".join(lines)


def _maybe_print_qultra_failure_debug(dataset_name: str, s, g_true_scaled, g_pred, g_pred_phys, pred_topo, pred_obs):
    global _DEBUG_QULTRA_COUNTS
    if DEBUG_QULTRA_FAILURES <= 0:
        return
    if DEBUG_QULTRA_DATASET and dataset_name != DEBUG_QULTRA_DATASET:
        return
    count = _DEBUG_QULTRA_COUNTS[dataset_name]
    if count >= DEBUG_QULTRA_FAILURES:
        return
    _DEBUG_QULTRA_COUNTS[dataset_name] = count + 1

    print("\n" + "!" * 90)
    print(f"DEBUG QULTRA FAILURE #{count + 1} | dataset={dataset_name}")
    print(f"sample index/id: {getattr(s, 'idx', None) or getattr(s, 'sample_idx', None) or 'n/a'}")
    print(f"error: {pred_obs.get('error')}")
    print(f"stage: {pred_obs.get('stage')}")
    print("\n" + _describe_graph_ns(g_true_scaled, "TRUE graph scaled/template"))
    print("\n" + _describe_graph_ns(g_pred, "PRED graph scaled raw"))
    print("\n" + _describe_graph_ns(g_pred_phys, "PRED graph physical aligned"))
    print("\n" + _describe_cqed_topology_text(pred_topo, "PRED macro CQEDTopology"))
    if pred_obs.get("primitive_description"):
        print("\n" + pred_obs["primitive_description"])
    if pred_obs.get("net_string"):
        print("\n# qultra net used")
        print(pred_obs["net_string"])
    if DEBUG_QULTRA_TRACEBACK and pred_obs.get("traceback"):
        print("\n# traceback")
        print(pred_obs["traceback"])
    print("!" * 90 + "\n")

def _simulate_topology_arrays(topology: CQEDTopology, qu, f_min: float, f_max: float) -> dict:
    """Run qultra and return numeric arrays with NaN for missing values.

    Debug note: on failure, the returned dict contains `stage`, `error`,
    `primitive_description`, `net_string`, and `traceback` when available.
    """
    out = {
        "ok": False,
        "error": None,
        "stage": None,
        "n_expected": 0,
        "frequencies": np.array([], dtype=float),
        "kappa": np.array([], dtype=float),
        "chi": np.zeros((0, 0), dtype=float),
        "primitive_description": None,
        "net_string": None,
        "traceback": None,
    }
    circuit = None
    try:
        out["stage"] = "expand_topology"
        primitive = expand_topology(topology, validate=False)
        out["primitive_description"] = _describe_cqed_topology_text(primitive, "PRED primitive CQEDTopology")

        out["stage"] = "topology_to_net"
        qelements = topology_to_net(primitive)
        out["net_string"] = _qelements_to_text(qelements)

        out["stage"] = "qelements_to_qultra_net"
        qnet = qelements_to_qultra_net(qelements, qu)
        n_expected = int(expected_mode_count(primitive))
        out["n_expected"] = n_expected

        out["stage"] = "QCircuit"
        circuit = qu.QCircuit(qnet, f_min, f_max)

        out["stage"] = "mode_frequencies"
        try:
            freqs_raw = circuit.mode_frequencies()
        except Exception:
            freqs_raw = []

        out["stage"] = "run_epr"
        try:
            chi_raw, _ = circuit.run_epr()
        except Exception:
            chi_raw = None

        out["stage"] = "kappa"
        try:
            kappa_raw = circuit.kappa()
        except Exception:
            kappa_raw = []

        out.update({
            "ok": True,
            "stage": "done",
            "frequencies": _to_float_array(freqs_raw, n_expected),
            "kappa": _to_float_array(kappa_raw, n_expected),
            "chi": _to_float_matrix(chi_raw, n_expected),
        })
        return out
    except Exception as exc:
        out["error"] = f"{out.get('stage')}: {type(exc).__name__}: {exc}"
        out["traceback"] = traceback.format_exc()
        return out
    finally:
        try:
            del circuit
        except Exception:
            pass
        gc.collect()

def _linear_sum_assignment_small(cost: np.ndarray) -> list[tuple[int, int]]:
    """Assignment helper: scipy if available, otherwise exact for small n, greedy fallback."""
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(cost)
        return list(zip(r.tolist(), c.tolist()))
    except Exception:
        pass
    n, m = cost.shape
    k = min(n, m)
    if max(n, m) <= 9:
        import itertools
        best = None
        best_val = float("inf")
        if n <= m:
            for cols in itertools.permutations(range(m), n):
                val = sum(cost[i, cols[i]] for i in range(n))
                if val < best_val:
                    best_val = val
                    best = [(i, cols[i]) for i in range(n)]
        else:
            for rows in itertools.permutations(range(n), m):
                val = sum(cost[rows[j], j] for j in range(m))
                if val < best_val:
                    best_val = val
                    best = [(rows[j], j) for j in range(m)]
        return best or []
    # greedy fallback
    pairs = []
    used_r, used_c = set(), set()
    flat = [(cost[i, j], i, j) for i in range(n) for j in range(m)]
    for _, i, j in sorted(flat, key=lambda x: x[0]):
        if i not in used_r and j not in used_c:
            pairs.append((i, j))
            used_r.add(i); used_c.add(j)
            if len(pairs) == k:
                break
    return pairs


def _align_pred_modes_to_true(true_obs: dict, pred_obs: dict) -> dict:
    """Align predicted modes to true modes using frequency assignment, then permute chi/kappa."""
    tf = np.asarray(true_obs["frequencies"], dtype=np.float64)
    pf = np.asarray(pred_obs["frequencies"], dtype=np.float64)
    n_true = len(tf)
    af = np.full(n_true, np.nan, dtype=np.float64)
    ak = np.full(n_true, np.nan, dtype=np.float64)
    ac = np.full((n_true, n_true), np.nan, dtype=np.float64)
    valid_t = np.where(np.isfinite(tf))[0]
    valid_p = np.where(np.isfinite(pf))[0]
    mapping: dict[int, int] = {}
    if len(valid_t) and len(valid_p):
        cost = np.abs(tf[valid_t, None] - pf[valid_p][None, :])
        for ii, jj in _linear_sum_assignment_small(cost):
            mapping[int(valid_t[ii])] = int(valid_p[jj])
    for ti, pj in mapping.items():
        af[ti] = pf[pj]
        pk = np.asarray(pred_obs.get("kappa", []), dtype=np.float64)
        if pj < len(pk):
            ak[ti] = pk[pj]
    pc = np.asarray(pred_obs.get("chi", np.zeros((0, 0))), dtype=np.float64)
    for ti, pi in mapping.items():
        for tj, pj in mapping.items():
            if pi < pc.shape[0] and pj < pc.shape[1]:
                ac[ti, tj] = pc[pi, pj]
    return {"frequencies": af, "kappa": ak, "chi": ac, "mode_mapping": mapping}


def _flatten_physics_records(records: list[dict]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    t_freq, p_freq, t_kap, p_kap, t_chi, p_chi = [], [], [], [], [], []
    for r in records:
        t = r["true"]
        p = r["pred_aligned"]
        t_freq.extend(np.asarray(t["frequencies"], dtype=float).ravel())
        p_freq.extend(np.asarray(p["frequencies"], dtype=float).ravel())
        t_kap.extend(np.asarray(t["kappa"], dtype=float).ravel())
        p_kap.extend(np.asarray(p["kappa"], dtype=float).ravel())
        t_chi.extend(np.asarray(t["chi"], dtype=float).ravel())
        p_chi.extend(np.asarray(p["chi"], dtype=float).ravel())
    return {
        "frequency": (np.asarray(t_freq), np.asarray(p_freq)),
        "kappa": (np.asarray(t_kap), np.asarray(p_kap)),
        "chi": (np.asarray(t_chi), np.asarray(p_chi)),
    }


def compute_observable_metrics(records: list[dict]) -> dict[str, dict]:
    out = {}
    for name, (t, p) in _flatten_physics_records(records).items():
        mask = np.isfinite(t) & np.isfinite(p)
        n_valid = int(mask.sum())
        if n_valid < 2:
            out[name] = {"r2": float("nan"), "rmse": float("nan"), "mae": float("nan"), "n_valid": n_valid}
            continue
        d = p[mask] - t[mask]
        out[name] = {
            "r2": r2_score(t[mask], p[mask]),
            "rmse": float(np.sqrt(np.mean(d * d))),
            "mae": float(np.mean(np.abs(d))),
            "n_valid": n_valid,
        }
    return out


def plot_observable_scatter(results: dict, out_path: str, n_samples: int, branch: str, ckpt_path: str) -> None:
    """Plot true-vs-pred scatter for qultra observables.

    True observables come from the dataset; predicted observables come from
    qultra simulation of the predicted graph. Predicted modes are already
    aligned to true modes before flattening.
    """
    ds_names = [ds for ds in sorted(results) if results[ds].get("physics_records")]
    observables = ["frequency", "chi", "kappa"]

    if not ds_names:
        print("  Nessun record qultra valido da plottare.")
        return

    n_rows = len(ds_names)
    n_cols = len(observables)
    fig = plt.figure(figsize=(4.2 * n_cols, 3.8 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.55, wspace=0.38)

    for row_i, ds_name in enumerate(ds_names):
        color = DS_COLORS[row_i % len(DS_COLORS)]
        flat = _flatten_physics_records(results[ds_name].get("physics_records", []))
        metrics = results[ds_name].get("observable_metrics", {})

        for col_i, obs in enumerate(observables):
            ax = fig.add_subplot(gs[row_i, col_i])
            t, p = flat.get(obs, (np.array([]), np.array([])))
            t = np.asarray(t, dtype=np.float64)
            p = np.asarray(p, dtype=np.float64)
            mask = np.isfinite(t) & np.isfinite(p)
            t = t[mask]
            p = p[mask]

            short_ds = ds_name.replace("_", " ")
            if len(short_ds) > 32:
                short_ds = short_ds[:30] + "…"

            if len(t) == 0:
                ax.text(0.5, 0.5, "nessun dato valido", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="gray")
                ax.set_title(f"{short_ds} — {obs}", fontsize=8)
                ax.axis("off")
                continue

            if len(t) > n_samples:
                idx = np.random.choice(len(t), n_samples, replace=False)
                t = t[idx]
                p = p[idx]

            # Use log scale only for strictly-positive quantities spanning many decades.
            use_log = bool(len(t) and np.all(t > 0) and np.all(p > 0) and (max(t.max(), p.max()) / max(min(t.min(), p.min()), 1e-300) > 100))
            if use_log:
                tv = np.log10(t)
                pv = np.log10(p)
                axis_label = f"log10 {obs}"
            else:
                tv = t
                pv = p
                axis_label = obs

            ax.scatter(tv, pv, s=7, alpha=0.35, color=color, linewidths=0, rasterized=True)
            lo = min(tv.min(), pv.min())
            hi = max(tv.max(), pv.max())
            pad = (hi - lo) * 0.05 or 0.1
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", lw=0.8, ls="--", zorder=3)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_aspect("equal", adjustable="box")

            m = metrics.get(obs, {"r2": float("nan"), "rmse": float("nan"), "mae": float("nan"), "n_valid": 0})
            ax.set_title(
                f"{short_ds} — {obs}\n"
                f"R²={_fmt_title(m.get('r2', float('nan')))}  "
                f"RMSE={_fmt_title(m.get('rmse', float('nan')))}  "
                f"N={m.get('n_valid', 0)}",
                fontsize=8, pad=3,
            )
            ax.set_xlabel(f"vero dataset\n{axis_label}", fontsize=7)
            ax.set_ylabel(f"predetto qultra\n{axis_label}", fontsize=7)
            ax.tick_params(labelsize=6)

    fig.suptitle(
        f"GraphVAE — osservabili qultra predette vs vere | ramo={branch} | {Path(ckpt_path).name}",
        fontsize=11, y=1.01,
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  Plot osservabili salvato -> {out_path}")



def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(text))


def _observable_variable_pairs(records: list[dict], observable: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return true/pred arrays grouped by individual observable variable.

    Examples of returned keys:
      - frequency: f_1, f_2, ...
      - kappa:     kappa_1, kappa_2, ...
      - chi:       chi_11, chi_12, ...

    The predicted arrays are already mode-aligned using frequency matching, so
    chi_ij and kappa_i refer to the same true mode indices.
    """
    buckets: dict[str, tuple[list[float], list[float]]] = {}

    def add(key: str, t_val, p_val) -> None:
        if key not in buckets:
            buckets[key] = ([], [])
        try:
            tv = float(t_val)
        except Exception:
            tv = np.nan
        try:
            pv = float(p_val)
        except Exception:
            pv = np.nan
        buckets[key][0].append(tv)
        buckets[key][1].append(pv)

    for rec in records:
        true = rec.get("true", {})
        pred = rec.get("pred_aligned", {})

        if observable == "frequency":
            t = np.asarray(true.get("frequencies", []), dtype=np.float64).ravel()
            pr = np.asarray(pred.get("frequencies", []), dtype=np.float64).ravel()
            n = max(len(t), len(pr))
            for i in range(n):
                add(f"f_{i + 1}", t[i] if i < len(t) else np.nan, pr[i] if i < len(pr) else np.nan)

        elif observable == "kappa":
            t = np.asarray(true.get("kappa", []), dtype=np.float64).ravel()
            pr = np.asarray(pred.get("kappa", []), dtype=np.float64).ravel()
            n = max(len(t), len(pr))
            for i in range(n):
                add(f"kappa_{i + 1}", t[i] if i < len(t) else np.nan, pr[i] if i < len(pr) else np.nan)

        elif observable == "chi":
            t = np.asarray(true.get("chi", np.zeros((0, 0))), dtype=np.float64)
            pr = np.asarray(pred.get("chi", np.zeros((0, 0))), dtype=np.float64)
            n = max(t.shape[0] if t.ndim == 2 else 0, pr.shape[0] if pr.ndim == 2 else 0)
            for i in range(n):
                for j in range(i, n):
                    tv = t[i, j] if t.ndim == 2 and i < t.shape[0] and j < t.shape[1] else np.nan
                    pv = pr[i, j] if pr.ndim == 2 and i < pr.shape[0] and j < pr.shape[1] else np.nan
                    add(f"chi_{i + 1}{j + 1}", tv, pv)

    return {k: (np.asarray(v[0], dtype=np.float64), np.asarray(v[1], dtype=np.float64))
            for k, v in buckets.items()}


def compute_observable_variable_metrics(records: list[dict]) -> dict[str, dict[str, dict]]:
    """Metrics for every f_i, chi_ij and kappa_i separately."""
    out: dict[str, dict[str, dict]] = {}
    for observable in ("frequency", "chi", "kappa"):
        out[observable] = {}
        for var, (t, p) in _observable_variable_pairs(records, observable).items():
            mask = np.isfinite(t) & np.isfinite(p)
            n_valid = int(mask.sum())
            if n_valid < 2:
                out[observable][var] = {
                    "r2": float("nan"), "rmse": float("nan"),
                    "mae": float("nan"), "n_valid": n_valid,
                }
                continue
            d = p[mask] - t[mask]
            out[observable][var] = {
                "r2": r2_score(t[mask], p[mask]),
                "rmse": float(np.sqrt(np.mean(d * d))),
                "mae": float(np.mean(np.abs(d))),
                "n_valid": n_valid,
            }
    return out


def _plot_one_variable_scatter(ax, true_vals: np.ndarray, pred_vals: np.ndarray,
                               title: str, color: str, x_label: str = "vero", y_label: str = "predetto") -> None:
    mask = np.isfinite(true_vals) & np.isfinite(pred_vals)
    t = np.asarray(true_vals, dtype=np.float64)[mask]
    p = np.asarray(pred_vals, dtype=np.float64)[mask]
    if len(t) == 0:
        ax.text(0.5, 0.5, "nessun dato", ha="center", va="center",
                transform=ax.transAxes, fontsize=7, color="gray")
        ax.set_title(title, fontsize=7.5)
        ax.axis("off")
        return

    use_log = bool(np.all(t > 0) and np.all(p > 0) and
                   (max(t.max(), p.max()) / max(min(t.min(), p.min()), 1e-300) > 100))
    if use_log:
        tv = np.log10(t)
        pv = np.log10(p)
        x_label = "log10 " + x_label
        y_label = "log10 " + y_label
    else:
        tv, pv = t, p

    ax.scatter(tv, pv, s=8, alpha=0.35, color=color, linewidths=0, rasterized=True)
    lo = min(tv.min(), pv.min())
    hi = max(tv.max(), pv.max())
    pad = (hi - lo) * 0.05 or 0.1
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", lw=0.75, ls="--", zorder=3)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=7.5, pad=2)
    ax.set_xlabel(x_label, fontsize=6.5)
    ax.set_ylabel(y_label, fontsize=6.5)
    ax.tick_params(labelsize=6)


def plot_observable_variable_scatters(results: dict, out_dir: Path, branch: str, ckpt_path: str) -> None:
    """Plot true-vs-pred separately for every f_i, chi_ij and kappa_i."""
    ds_names = [ds for ds in sorted(results) if results[ds].get("physics_records")]
    if not ds_names:
        print("  Nessun record qultra valido per plot per-variabile.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    for observable in ("frequency", "chi", "kappa"):
        # Union of variables across datasets, ordered naturally by suffix.
        var_names: list[str] = []
        for ds in ds_names:
            pairs = _observable_variable_pairs(results[ds].get("physics_records", []), observable)
            for v in pairs:
                if v not in var_names:
                    var_names.append(v)

        if not var_names:
            continue

        n_rows = len(ds_names)
        n_cols = len(var_names)
        # Keep large chi grids readable.
        fig = plt.figure(figsize=(3.0 * n_cols, 2.9 * n_rows))
        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.62, wspace=0.45)

        for row_i, ds_name in enumerate(ds_names):
            color = DS_COLORS[row_i % len(DS_COLORS)]
            records = results[ds_name].get("physics_records", [])
            pairs = _observable_variable_pairs(records, observable)
            metrics = compute_observable_variable_metrics(records).get(observable, {})
            short_ds = ds_name.replace("_", " ")
            if len(short_ds) > 24:
                short_ds = short_ds[:22] + "…"

            for col_i, var in enumerate(var_names):
                ax = fig.add_subplot(gs[row_i, col_i])
                t, p = pairs.get(var, (np.array([]), np.array([])))
                m = metrics.get(var, {"r2": float("nan"), "rmse": float("nan"), "n_valid": 0})
                title = f"{short_ds}\n{var}  R²={_fmt_title(m.get('r2', float('nan')))}  N={m.get('n_valid', 0)}"
                _plot_one_variable_scatter(ax, t, p, title, color,
                                           x_label=f"true {var}", y_label=f"pred {var}")

        fig.suptitle(
            f"GraphVAE — {observable} per variabile | ramo={branch} | {Path(ckpt_path).name}",
            fontsize=11, y=1.01,
        )
        out_path = out_dir / f"inference_{branch}_{observable}_by_variable.png"
        plt.savefig(out_path, dpi=145, bbox_inches="tight")
        plt.close()
        print(f"  Plot {observable} per variabile salvato -> {out_path}")


def _record_observable_series(rec: dict, observable: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return labels, true, pred arrays for one sample and one observable."""
    true = rec.get("true", {})
    pred = rec.get("pred_aligned", {})
    labels: list[str] = []
    t_vals: list[float] = []
    p_vals: list[float] = []

    if observable == "frequency":
        t = np.asarray(true.get("frequencies", []), dtype=np.float64).ravel()
        p = np.asarray(pred.get("frequencies", []), dtype=np.float64).ravel()
        n = max(len(t), len(p))
        for i in range(n):
            labels.append(f"f_{i + 1}")
            t_vals.append(t[i] if i < len(t) else np.nan)
            p_vals.append(p[i] if i < len(p) else np.nan)

    elif observable == "kappa":
        t = np.asarray(true.get("kappa", []), dtype=np.float64).ravel()
        p = np.asarray(pred.get("kappa", []), dtype=np.float64).ravel()
        n = max(len(t), len(p))
        for i in range(n):
            labels.append(f"kappa_{i + 1}")
            t_vals.append(t[i] if i < len(t) else np.nan)
            p_vals.append(p[i] if i < len(p) else np.nan)

    elif observable == "chi":
        t = np.asarray(true.get("chi", np.zeros((0, 0))), dtype=np.float64)
        p = np.asarray(pred.get("chi", np.zeros((0, 0))), dtype=np.float64)
        n = max(t.shape[0] if t.ndim == 2 else 0, p.shape[0] if p.ndim == 2 else 0)
        for i in range(n):
            for j in range(i, n):
                labels.append(f"chi_{i + 1}{j + 1}")
                t_vals.append(t[i, j] if t.ndim == 2 and i < t.shape[0] and j < t.shape[1] else np.nan)
                p_vals.append(p[i, j] if p.ndim == 2 and i < p.shape[0] and j < p.shape[1] else np.nan)

    return labels, np.asarray(t_vals, dtype=np.float64), np.asarray(p_vals, dtype=np.float64)




def _rel_err_percent(true: np.ndarray, pred: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    """Absolute relative error in percent, robust to zero-valued observables."""
    true = np.asarray(true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    return np.abs(pred - true) / np.maximum(np.abs(true), eps) * 100.0


def _collect_dataset_variables(records: list[dict]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Collect f_i, kappa_i and upper-triangle chi_ij into aligned matrices.

    Returns labels, true_matrix, pred_matrix with shape (n_records, n_variables).
    Missing values are NaN.  Chi uses only i <= j to avoid double-counting the
    symmetric lower triangle.
    """
    if not records:
        return [], np.zeros((0, 0)), np.zeros((0, 0))

    var_order: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for observable in ("frequency", "kappa", "chi"):
        pairs = _observable_variable_pairs(records, observable)
        for key in pairs:
            token = (observable, key)
            if token not in seen:
                seen.add(token)
                var_order.append(token)

    labels = [key for _obs, key in var_order]
    true_cols: list[np.ndarray] = []
    pred_cols: list[np.ndarray] = []
    for observable, key in var_order:
        pairs = _observable_variable_pairs(records, observable)
        t, p = pairs.get(key, (np.full(len(records), np.nan), np.full(len(records), np.nan)))
        true_cols.append(np.asarray(t, dtype=np.float64))
        pred_cols.append(np.asarray(p, dtype=np.float64))

    true_mat = np.vstack(true_cols).T if true_cols else np.zeros((len(records), 0))
    pred_mat = np.vstack(pred_cols).T if pred_cols else np.zeros((len(records), 0))
    return labels, true_mat, pred_mat


def _plot_evaluator_style_hist(ds_name: str, labels: list[str], true_mat: np.ndarray,
                               pred_mat: np.ndarray, out_dir: Path, branch: str) -> None:
    n = len(labels)
    if n == 0:
        return
    ncols = min(n, 5)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.6 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for j, (ax, label) in enumerate(zip(axes_flat, labels)):
        t = true_mat[:, j]
        p = pred_mat[:, j]
        mask = np.isfinite(t) & np.isfinite(p)
        if mask.sum() == 0:
            ax.text(0.5, 0.5, "nessun dato", ha="center", va="center", transform=ax.transAxes, fontsize=8, color="gray")
            ax.set_title(label, fontsize=9)
            ax.axis("off")
            continue
        d = _rel_err_percent(t[mask], p[mask])
        d = d[np.isfinite(d)]
        if d.size == 0:
            ax.text(0.5, 0.5, "nessun errore finito", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="gray")
            ax.set_title(label, fontsize=9)
            ax.axis("off")
            continue
        # np.histogram can fail with fixed high bin counts when all values are
        # identical or the finite range is numerically degenerate.  Use a robust,
        # adaptive bin count for tiny debug subsets such as --max-qultra-per-dataset 10.
        if d.size == 1 or np.nanmax(d) == np.nanmin(d):
            hist_bins = 1
        else:
            hist_bins = int(min(50, max(1, d.size)))
        ax.hist(d, bins=hist_bins, alpha=0.82, edgecolor="white", linewidth=0.4)
        ax.axvline(float(np.mean(d)), color="black", lw=1.5, ls="--", label=f"mean={np.mean(d):.1f}%")
        ax.axvline(float(np.percentile(d, 95)), color="red", lw=1.2, ls=":", label=f"p95={np.percentile(d, 95):.1f}%")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Relative error (%)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for ax in axes_flat[n:]:
        ax.set_visible(False)
    fig.suptitle(f"{ds_name} [{branch}] — observable error histograms", fontsize=11)
    plt.tight_layout()
    out_path = out_dir / f"{_safe_name(ds_name)}_{branch}_obs_hist.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def _plot_evaluator_style_scatter(ds_name: str, labels: list[str], true_mat: np.ndarray,
                                  pred_mat: np.ndarray, out_dir: Path, branch: str) -> None:
    n = len(labels)
    if n == 0:
        return
    ncols = min(n, 5)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.9 * nrows), squeeze=False)
    axes_flat = axes.ravel()

    for j, (ax, label) in enumerate(zip(axes_flat, labels)):
        t = true_mat[:, j]
        p = pred_mat[:, j]
        mask = np.isfinite(t) & np.isfinite(p)
        if mask.sum() == 0:
            ax.text(0.5, 0.5, "nessun dato", ha="center", va="center", transform=ax.transAxes, fontsize=8, color="gray")
            ax.set_title(label, fontsize=9)
            ax.axis("off")
            continue
        tv = t[mask]
        pv = p[mask]
        ax.scatter(tv, pv, s=6, alpha=0.35, linewidths=0, rasterized=True)
        mn = min(float(tv.min()), float(pv.min()))
        mx = max(float(tv.max()), float(pv.max()))
        pad = (mx - mn) * 0.05 or 0.1
        ax.plot([mn - pad, mx + pad], [mn - pad, mx + pad], "k--", lw=1.0)
        ax.set_xlim(mn - pad, mx + pad)
        ax.set_ylim(mn - pad, mx + pad)
        try:
            r2 = r2_score(tv, pv)
        except Exception:
            r2 = float("nan")
        ax.set_xlabel(f"{label} true", fontsize=7)
        ax.set_ylabel(f"{label} pred", fontsize=7)
        ax.set_title(f"{label}  R²={_fmt_title(r2)}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    for ax in axes_flat[n:]:
        ax.set_visible(False)
    fig.suptitle(f"{ds_name} [{branch}] — true vs pred per observable", fontsize=11)
    plt.tight_layout()
    out_path = out_dir / f"{_safe_name(ds_name)}_{branch}_obs_scatter.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def _plot_evaluator_style_sample_panel(ds_name: str, labels: list[str], true_mat: np.ndarray,
                                       pred_mat: np.ndarray, out_dir: Path, branch: str,
                                       n_show: int = 10, seed: int = 42) -> None:
    n_vars = len(labels)
    if n_vars == 0 or true_mat.shape[0] == 0:
        return
    rng = np.random.default_rng(seed)
    finite_rows = np.where(np.any(np.isfinite(true_mat) & np.isfinite(pred_mat), axis=1))[0]
    if len(finite_rows) == 0:
        return
    idx = np.sort(rng.choice(finite_rows, size=min(int(n_show), len(finite_rows)), replace=False))
    ncols = min(n_vars, 5)
    nrows = int(np.ceil(n_vars / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.6 * nrows), squeeze=False)
    axes_flat = axes.ravel()
    xs = np.arange(len(idx))

    for j, (ax, label) in enumerate(zip(axes_flat, labels)):
        t = true_mat[idx, j]
        p = pred_mat[idx, j]
        ax.plot(xs, t, "o-", lw=1.4, ms=4, label="true")
        ax.plot(xs, p, "s--", lw=1.4, ms=4, alpha=0.75, label="pred")
        mask = np.isfinite(t) & np.isfinite(p)
        rel = np.full(len(idx), np.nan, dtype=np.float64)
        rel[mask] = _rel_err_percent(t[mask], p[mask])
        for k, rv in enumerate(rel):
            if np.isfinite(rv) and np.isfinite(p[k]):
                ax.annotate(f"{rv:.1f}%", (k, p[k]), textcoords="offset points", xytext=(0, 6),
                            ha="center", fontsize=6, color="dimgray")
        ax.set_title(label, fontsize=8)
        ax.set_xlabel("sample", fontsize=7)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(int(i)) for i in idx], rotation=45, ha="right", fontsize=6)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(fontsize=7)

    for ax in axes_flat[n_vars:]:
        ax.set_visible(False)
    fig.suptitle(f"{ds_name} [{branch}] — {len(idx)} random samples", fontsize=11)
    plt.tight_layout()
    out_path = out_dir / f"{_safe_name(ds_name)}_{branch}_sample_panel.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_evaluator_style_observable_reports(results: dict, out_dir: Path, branch: str,
                                            n_random: int = 10, seed: int = 42) -> None:
    """Create evaluator_vae-style plots while keeping the corrected evaluation logic.

    For each dataset this saves:
      - <dataset>_<branch>_obs_hist.png
      - <dataset>_<branch>_obs_scatter.png
      - <dataset>_<branch>_sample_panel.png
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for ds_name in sorted(results):
        records = list(results[ds_name].get("physics_records", []))
        if not records:
            continue
        labels, true_mat, pred_mat = _collect_dataset_variables(records)
        if not labels:
            continue
        _plot_evaluator_style_hist(ds_name, labels, true_mat, pred_mat, out_dir, branch)
        _plot_evaluator_style_scatter(ds_name, labels, true_mat, pred_mat, out_dir, branch)
        _plot_evaluator_style_sample_panel(ds_name, labels, true_mat, pred_mat, out_dir, branch,
                                           n_show=n_random, seed=seed)


def _kappa_swap_metrics(true_mat: np.ndarray, pred_mat: np.ndarray) -> dict:
    """Compare normal and swapped kappa ordering for a two-mode dataset."""
    true_mat = np.asarray(true_mat, dtype=np.float64)
    pred_mat = np.asarray(pred_mat, dtype=np.float64)
    if true_mat.ndim != 2 or pred_mat.ndim != 2 or true_mat.shape[1] < 2 or pred_mat.shape[1] < 2:
        return {"normal_mae": float("nan"), "swapped_mae": float("nan"), "normal_rmse": float("nan"), "swapped_rmse": float("nan")}

    def _mae_rmse(p):
        mask = np.isfinite(true_mat[:, :2]) & np.isfinite(p[:, :2])
        if not np.any(mask):
            return float("nan"), float("nan")
        d = p[:, :2][mask] - true_mat[:, :2][mask]
        return float(np.mean(np.abs(d))), float(np.sqrt(np.mean(d * d)))

    mae_n, rmse_n = _mae_rmse(pred_mat[:, :2])
    mae_s, rmse_s = _mae_rmse(pred_mat[:, [1, 0]])
    return {"normal_mae": mae_n, "swapped_mae": mae_s, "normal_rmse": rmse_n, "swapped_rmse": rmse_s}


def plot_inductive_kappa_debug(results: dict, out_dir: Path, branch: str, ckpt_path: str,
                               dataset_name: str = "Qubit_resonator_feedline_inductive_nognd",
                               n_show: int = 10, seed: int = 42) -> None:
    """Debug the two kappa values for the inductive qubit-resonator-feedline case.

    The figure intentionally shows both the normal comparison and the comparison
    obtained by swapping pred(kappa_1) <-> pred(kappa_2).  If the swapped scatter
    and error numbers look much better, the mismatch is likely an ordering issue
    rather than a model-quality issue.
    """
    records = list(results.get(dataset_name, {}).get("physics_records", []))
    if not records:
        print(f"  Nessun record qultra valido per debug kappa: {dataset_name} [{branch}]")
        return

    rng = np.random.default_rng(seed)
    candidates = []
    for i, rec in enumerate(records):
        labels, t, p = _record_observable_series(rec, "kappa")
        if len(labels) >= 2 and np.any(np.isfinite(t[:2]) | np.isfinite(p[:2])):
            candidates.append((i, labels[:2], t[:2], p[:2]))
    if not candidates:
        print(f"  Nessun kappa finito per debug kappa: {dataset_name} [{branch}]")
        return

    chosen_idx = np.sort(rng.choice(len(candidates), size=min(int(n_show), len(candidates)), replace=False))
    chosen = [candidates[int(i)] for i in chosen_idx]
    sample_ids = [int(c[0]) for c in chosen]
    true_mat = np.vstack([c[2] for c in chosen])
    pred_mat = np.vstack([c[3] for c in chosen])
    pred_swap = pred_mat[:, [1, 0]] if pred_mat.shape[1] >= 2 else pred_mat
    metrics = _kappa_swap_metrics(true_mat, pred_mat)

    xs = np.arange(len(chosen))
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), squeeze=False)

    for j, label in enumerate(["kappa_1", "kappa_2"]):
        ax = axes[0][j]
        ax.plot(xs, true_mat[:, j], marker="o", lw=1.4, label=f"true {label}")
        ax.plot(xs, pred_mat[:, j], marker="x", lw=1.4, label=f"pred {label}")
        ax.plot(xs, pred_swap[:, j], marker="s", lw=1.1, ls="--", alpha=0.75, label=f"pred swapped -> {label}")
        ax.set_title(f"{label}: vero vs predetto", fontsize=10)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(i) for i in sample_ids], rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("physics_records index", fontsize=8)
        ax.set_ylabel(label, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    def _scatter_panel(ax, pred, title):
        t = true_mat[:, :2].ravel()
        p = pred[:, :2].ravel()
        mask = np.isfinite(t) & np.isfinite(p)
        if not np.any(mask):
            ax.text(0.5, 0.5, "nessun dato finito", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            return
        tv = t[mask]
        pv = p[mask]
        ax.scatter(tv, pv, s=28, alpha=0.75)
        mn = min(float(tv.min()), float(pv.min()))
        mx = max(float(tv.max()), float(pv.max()))
        pad = (mx - mn) * 0.08 or 0.01
        ax.plot([mn - pad, mx + pad], [mn - pad, mx + pad], "k--", lw=1.0)
        ax.set_xlim(mn - pad, mx + pad)
        ax.set_ylim(mn - pad, mx + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("true kappa", fontsize=8)
        ax.set_ylabel("pred kappa", fontsize=8)
        ax.grid(True, alpha=0.3)

    _scatter_panel(axes[1][0], pred_mat, f"ordine normale | MAE={_fmt_title(metrics['normal_mae'])} RMSE={_fmt_title(metrics['normal_rmse'])}")
    _scatter_panel(axes[1][1], pred_swap, f"pred invertiti | MAE={_fmt_title(metrics['swapped_mae'])} RMSE={_fmt_title(metrics['swapped_rmse'])}")

    verdict = ""
    if np.isfinite(metrics["normal_mae"]) and np.isfinite(metrics["swapped_mae"]):
        if metrics["swapped_mae"] < 0.7 * metrics["normal_mae"]:
            verdict = " | sospetto: kappa predetti invertiti"
        elif metrics["normal_mae"] < 0.7 * metrics["swapped_mae"]:
            verdict = " | ordine normale migliore"
        else:
            verdict = " | inversione non risolve chiaramente"

    fig.suptitle(
        f"Debug kappa inductive | {dataset_name} | ramo={branch} | {Path(ckpt_path).name}{verdict}",
        fontsize=12,
    )
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"inference_{branch}_{_safe_name(dataset_name)}_kappa_swap_debug.png"
    plt.savefig(out_path, dpi=145, bbox_inches="tight")
    plt.close()
    print(f"  Plot debug kappa inductive salvato -> {out_path}")

def plot_random_sample_observable_comparison(results: dict, out_dir: Path, branch: str,
                                             ckpt_path: str, n_random: int = 10, seed: int = 42) -> None:
    """For random samples, plot expected and predicted values for every variable.

    One figure is saved per dataset.  Rows are random samples; columns are
    frequency, chi and kappa.  Each panel overlays true and predicted series.
    """
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ds_name in sorted(results):
        records = list(results[ds_name].get("physics_records", []))
        if not records:
            continue
        n_show = min(int(n_random), len(records))
        if n_show <= 0:
            continue
        idx = rng.choice(len(records), size=n_show, replace=False)
        selected = [records[int(i)] for i in idx]

        observables = ["frequency", "chi", "kappa"]
        fig = plt.figure(figsize=(5.0 * len(observables), 2.35 * n_show))
        gs = gridspec.GridSpec(n_show, len(observables), figure=fig, hspace=0.85, wspace=0.32)

        for row_i, rec in enumerate(selected):
            for col_i, obs in enumerate(observables):
                ax = fig.add_subplot(gs[row_i, col_i])
                labels, t, p = _record_observable_series(rec, obs)
                mask = np.isfinite(t) | np.isfinite(p)
                labels = [lab for lab, keep in zip(labels, mask) if keep]
                t = t[mask]
                p = p[mask]

                if len(labels) == 0:
                    ax.text(0.5, 0.5, "nessun dato", ha="center", va="center",
                            transform=ax.transAxes, fontsize=7, color="gray")
                    ax.axis("off")
                    continue

                x = np.arange(len(labels))
                ax.plot(x, t, marker="o", lw=1.0, label="atteso")
                ax.plot(x, p, marker="x", lw=1.0, label="predetto")
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=6)
                ax.tick_params(axis="y", labelsize=6)
                if row_i == 0:
                    ax.set_title(obs, fontsize=8.5)
                if col_i == 0:
                    ax.set_ylabel(f"sample {int(idx[row_i])}", fontsize=7)
                if row_i == 0 and col_i == len(observables) - 1:
                    ax.legend(fontsize=7, loc="best")
                ax.grid(True, alpha=0.25, lw=0.5)

        fig.suptitle(
            f"GraphVAE — 10 sample random: atteso vs predetto | {ds_name} | ramo={branch} | {Path(ckpt_path).name}",
            fontsize=11, y=1.005,
        )
        out_path = out_dir / f"inference_{branch}_{_safe_name(ds_name)}_random_samples_observables.png"
        plt.savefig(out_path, dpi=145, bbox_inches="tight")
        plt.close()
        print(f"  Plot random sample osservabili salvato -> {out_path}")

def _maybe_physics_eval_sample(s, g_pred, param_scaler, obs_scaler, qu, f_min: float, f_max: float, use_permutation_eval: bool = True):
    """Return one observable comparison record, or None if topology/attrs are unusable.

    The true observables are read directly from the dataset (`s.obs_vals` +
    `s.obs_mask`, inverse-transformed with `obs_scaler`).  Only the predicted
    graph is expanded and simulated with qultra.  This roughly halves the qultra
    cost and keeps the comparison grounded in the exact observables used to train
    the spec branch.
    """
    g_true_scaled = data_to_graph_ns(s, None)
    tm = topo_match(g_true_scaled.node_types, g_true_scaled.edges, g_pred.node_types, g_pred.edges)
    if not tm["exact"]:
        return None, tm, "topology_not_exact"

    true_row_scaled = s.y.tolist()
    pred_row_scaled = _scaled_attr_row_from_pred_graph(g_pred) if getattr(g_pred, "attrs", None) else []
    if len(pred_row_scaled) != len(true_row_scaled):
        return None, tm, "attr_length_mismatch"

    if use_permutation_eval:
        perms = _get_attr_perm_indices(s, len(true_row_scaled))
        pred_row_scaled, _, _ = _best_pred_row_aligned_to_true_order(true_row_scaled, pred_row_scaled, perms)

    pred_row_phys = _inverse_scaled_row_safe(param_scaler, pred_row_scaled)
    g_pred_phys = _graph_from_flat_attrs_like(g_true_scaled, pred_row_phys, "pred_aligned")
    pred_topo = _graph_ns_to_cqed_topology(g_pred_phys, name=f"{s.dataset_name}_pred")

    true_obs = _sample_true_observables_from_dataset(s, obs_scaler)
    pred_obs = _simulate_topology_arrays(pred_topo, qu, f_min, f_max)
    if not pred_obs["ok"]:
        _maybe_print_qultra_failure_debug(s.dataset_name, s, g_true_scaled, g_pred, g_pred_phys, pred_topo, pred_obs)
        return None, tm, "pred_qultra_failed"

    pred_aligned = _align_pred_modes_to_true(true_obs, pred_obs)
    rec = {
        "dataset_name": s.dataset_name,
        "true": true_obs,
        "pred_raw": pred_obs,
        "pred_aligned": pred_aligned,
        "n_modes_true": int(true_obs["n_expected"]),
        "n_modes_pred": int(pred_obs["n_expected"]),
    }
    return rec, tm, None


@torch.no_grad()
def run_circuit_encoder_observables(vae: GraphVAE, data: list, scalers: dict, device: torch.device,
                                    batch_size: int = 128, stochastic: bool = False,
                                    use_permutation_eval: bool = True,
                                    f_min: float = 1.0, f_max: float = 9.0,
                                    max_qultra_per_dataset: int | None = None) -> tuple[dict, bool, dict[str, int]]:
    vae.eval()
    if not _HAS_QULTRA_WORKFLOW:
        raise RuntimeError(f"Impossibile importare workflow qultra: {_QULTRA_IMPORT_ERROR}")
    qu = import_qultra()
    by_ds: dict[str, list] = {}
    for s in data:
        by_ds.setdefault(s.dataset_name, []).append(s)
    ds_map = _get_dataset_id_mapping(data)
    decode_used_ds_ids = False
    out = {}
    for ds_name, samples in by_ds.items():
        dataset_scalers = scalers[ds_name]
        scaler = dataset_scalers.param_scaler
        obs_scaler = dataset_scalers.obs_scaler
        topo_results = []
        physics_records = []
        fail_reasons = defaultdict(int)
        evaluated = 0
        for i in range(0, len(samples), batch_size):
            if max_qultra_per_dataset is not None and evaluated >= max_qultra_per_dataset:
                break
            batch = samples[i:i + batch_size]
            z, _, _ = vae.encode(batch, scalers)
            graphs_pred, used_ds_ids = _decode_graphs(vae, z, batch, stochastic, ds_map)
            decode_used_ds_ids = decode_used_ds_ids or used_ds_ids
            for s, g_pred in zip(batch, graphs_pred):
                if max_qultra_per_dataset is not None and evaluated >= max_qultra_per_dataset:
                    break
                rec, tm, reason = _maybe_physics_eval_sample(s, g_pred, scaler, obs_scaler, qu, f_min, f_max, use_permutation_eval)
                topo_results.append(tm)
                if rec is not None:
                    physics_records.append(rec)
                else:
                    fail_reasons[reason or "unknown"] += 1
                evaluated += 1
        out[ds_name] = {
            "topo_results": topo_results,
            "physics_records": physics_records,
            "observable_metrics": compute_observable_metrics(physics_records),
            "fail_reasons": dict(fail_reasons),
        }
    return out, decode_used_ds_ids, ds_map


@torch.no_grad()
def run_spec_encoder_observables(vae: GraphVAE, data: list, scalers: dict, device: torch.device,
                                 batch_size: int = 128, stochastic: bool = False,
                                 use_permutation_eval: bool = True,
                                 f_min: float = 1.0, f_max: float = 9.0,
                                 max_qultra_per_dataset: int | None = None) -> tuple[dict, bool, dict[str, int]]:
    vae.eval()
    if not _HAS_QULTRA_WORKFLOW:
        raise RuntimeError(f"Impossibile importare workflow qultra: {_QULTRA_IMPORT_ERROR}")
    qu = import_qultra()
    by_ds: dict[str, list] = {}
    for s in data:
        by_ds.setdefault(s.dataset_name, []).append(s)
    ds_map = _get_dataset_id_mapping(data)
    decode_used_ds_ids = False
    out = {}
    for ds_name, samples in by_ds.items():
        dataset_scalers = scalers[ds_name]
        scaler = dataset_scalers.param_scaler
        obs_scaler = dataset_scalers.obs_scaler
        topo_results = []
        physics_records = []
        fail_reasons = defaultdict(int)
        evaluated = 0
        for i in range(0, len(samples), batch_size):
            if max_qultra_per_dataset is not None and evaluated >= max_qultra_per_dataset:
                break
            batch = samples[i:i + batch_size]
            obs_vals = torch.stack([s.obs_vals for s in batch]).to(device)
            obs_mask = torch.stack([s.obs_mask for s in batch]).to(device)
            z_s, _, _ = vae.spec_encoder.encode(obs_vals, obs_mask)
            graphs_pred, used_ds_ids = _decode_graphs(vae, z_s, batch, stochastic, ds_map)
            decode_used_ds_ids = decode_used_ds_ids or used_ds_ids
            for s, g_pred in zip(batch, graphs_pred):
                if max_qultra_per_dataset is not None and evaluated >= max_qultra_per_dataset:
                    break
                rec, tm, reason = _maybe_physics_eval_sample(s, g_pred, scaler, obs_scaler, qu, f_min, f_max, use_permutation_eval)
                topo_results.append(tm)
                if rec is not None:
                    physics_records.append(rec)
                else:
                    fail_reasons[reason or "unknown"] += 1
                evaluated += 1
        out[ds_name] = {
            "topo_results": topo_results,
            "physics_records": physics_records,
            "observable_metrics": compute_observable_metrics(physics_records),
            "fail_reasons": dict(fail_reasons),
        }
    return out, decode_used_ds_ids, ds_map


def print_observable_report(branch_name: str, results: dict, decode_used_ds_ids: bool, ds_map: dict[str, int]) -> None:
    sep = "=" * 80
    print() ; print(sep) ; print(f"  RAMO: {branch_name}") ; print(sep)
    print(f"  decode con ds_ids: {decode_used_ds_ids}")
    print(f"  mapping dataset->id: {ds_map}")
    for ds_name in sorted(results):
        r = results[ds_name]
        tm = compute_topo_metrics(r.get("topo_results", []))
        om = r.get("observable_metrics", {})
        n_phys = len(r.get("physics_records", []))
        print(f"\n  Dataset: {ds_name}  (N_topo={tm.get('n_samples', 0)} | N_qultra_valid={n_phys})")
        print("  " + "-" * 60)
        print("  Metriche topologiche:")
        print(f"    Topology Accuracy  : {tm.get('topology_accuracy', float('nan')):.4f}  ({tm.get('n_exact', 0)}/{tm.get('n_samples', 0)} corrette)")
        print(f"    Node-type Accuracy : {tm.get('node_type_accuracy', float('nan')):.4f}")
        print(f"    Edge F1            : {_fmt(tm.get('edge_f1', float('nan')))}")
        print("  Metriche osservabili: true dal dataset, pred da qultra; matching dei modi sulle frequenze:")
        print(f"    {'Observable':<12} {'R²':>10} {'RMSE':>14} {'MAE':>14} {'N_valid':>8}")
        print("    " + "-" * 64)
        for obs in ("frequency", "chi", "kappa"):
            m = om.get(obs, {"r2": float('nan'), "rmse": float('nan'), "mae": float('nan'), "n_valid": 0})
            print(f"    {obs:<12} {_fmt(m['r2']):>10} {_fmt(m['rmse']):>14} {_fmt(m['mae']):>14} {m['n_valid']:>8}")
        fr = r.get("fail_reasons", {})
        if fr:
            print(f"  Campioni non valutati con qultra: {fr}")
    print()


def _jsonable_physics_results(results: dict) -> dict:
    out = {}
    for ds, r in results.items():
        records = r.get("physics_records", [])
        out[ds] = {
            "observable_metrics": r.get("observable_metrics", {}),
            "observable_variable_metrics": compute_observable_variable_metrics(records),
            "fail_reasons": r.get("fail_reasons", {}),
            "n_valid_qultra": len(records),
            "topology_metrics": compute_topo_metrics(r.get("topo_results", [])),
        }
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference + evaluation del GraphVAE")
    p.add_argument("--ckpt", default="checkpoints/vae_best.pt")
    p.add_argument("--split", default="test", choices=["test", "val", "train"])
    p.add_argument("--n-samples", type=int, default=750)
    p.add_argument("--out-dir", default="plots")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--stochastic", action="store_true")
    p.add_argument(
        "--inference-only", action="store_true",
        help=(
            "Carica TUTTI i dataset (inclusi include_train=False) e valuta anche "
            "le topologie mai viste in training (es. Three_qubit_capacitive_line). "
            "I scalers per questi dataset vengono fittati sui loro stessi dati, "
            "separatamente dagli scalers del training."
        ),
    )
    p.add_argument("--only-circuit", action="store_true")
    p.add_argument("--only-spec", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot-topo-errors", action="store_true")
    p.add_argument("--no-plot-r2-graph", action="store_true",
        help="Disabilita il plot NetworkX del grafo con R² aggregato sui nodi.")
    p.add_argument("--topo-errors-dataset", default="Three_qubit_capacitive_line")
    p.add_argument("--topo-errors-n", type=int, default=20)
    p.add_argument("--plot-latent-samples", action="store_true")
    p.add_argument("--latent-n", type=int, default=20)
    p.add_argument("--only-latent", action="store_true",
        help="Esegue solo il campionamento dal latent space, salta rami A/B e caricamento dataset.")
    p.add_argument("--no-permutation-eval", action="store_true",
        help="Disabilita l'allineamento dei parametri tramite permutazioni salvate durante la valutazione.")
    p.add_argument("--f-min", type=float, default=1.0, help="Frequenza minima per qultra/QCircuit.")
    p.add_argument("--f-max", type=float, default=9.0, help="Frequenza massima per qultra/QCircuit.")
    p.add_argument("--max-qultra-per-dataset", type=int, default=None,
        help="Limita il numero di campioni simulati con qultra per dataset; utile per debug/RAM.")
    p.add_argument("--save-observable-json", default="observable_metrics.json",
        help="File JSON riassuntivo con metriche freq/chi/kappa.")
    p.add_argument("--random-observable-samples", type=int, default=10,
        help="Numero di sample random da plottare con serie atteso vs predetto per ogni variabile.")
    p.add_argument("--no-plot-inductive-kappa-debug", action="store_true",
        help="Disabilita il debug plot dei due kappa per Qubit_resonator_feedline_inductive_nognd.")
    p.add_argument("--inductive-kappa-debug-dataset", default="Qubit_resonator_feedline_inductive_nognd",
        help="Dataset da usare per il debug kappa swap.")
    p.add_argument("--debug-qultra-dataset", default=None,
        help="Stampa debug dettagliato solo per questo dataset quando qultra fallisce, es. Three_qubit_capacitive_star.")
    p.add_argument("--debug-qultra-failures", type=int, default=0,
        help="Numero massimo di fallimenti qultra da stampare per dataset.")
    p.add_argument("--debug-qultra-traceback", action="store_true",
        help="Nel debug dei fallimenti stampa anche il traceback completo.")
    p.add_argument("--no-plot-expansion-debug", action="store_true",
                   help="Disabilita i pannelli NetworkX macro TRUE / macro PRED / PRED espanso per ogni dataset.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    global DEBUG_QULTRA_DATASET, DEBUG_QULTRA_FAILURES, DEBUG_QULTRA_TRACEBACK
    DEBUG_QULTRA_DATASET = args.debug_qultra_dataset
    DEBUG_QULTRA_FAILURES = int(args.debug_qultra_failures or 0)
    DEBUG_QULTRA_TRACEBACK = bool(args.debug_qultra_traceback)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Split: {args.split}  |  stochastic={args.stochastic}\n")

    print("Caricamento checkpoint...")
    model, scalers, cfg = load_checkpoint(args.ckpt, device)
    print(f"  step={model._step}  nz={model.nz}  beta={model.current_beta():.4f}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  parametri totali: {n_params:,}")

    if args.only_latent:
        print("\n--only-latent: skip caricamento dataset.")
        data = []
    elif args.inference_only:
        print("\nCaricamento dataset (VAE loader)...")
        # Load ALL datasets (train + inference-only), each with its own scaler.
        # The split argument is ignored: each dataset is evaluated in full.
        inference_only_names = [
            k for k, v in __import__("data_loader.schema", fromlist=["DATASETS"])
            .DATASETS.items() if not v.include_train
        ]
        print(f"  modalità --inference-only: carico anche {inference_only_names}")
        all_samples, all_scalers_inf = load_inference_datasets_vae(
            seed      = cfg.get("seed", 42),
            max_nodes = cfg.get("max_nodes", 12),
        )
        # For train-datasets, use the checkpoint scalers.
        # For inference-only datasets, use the freshly fitted scalers.
        merged_scalers = dict(scalers)  # checkpoint scalers (train datasets)
        for ds_name, sc in all_scalers_inf.items():
            if ds_name not in merged_scalers:
                merged_scalers[ds_name] = sc  # inference-only dataset scaler
        scalers = merged_scalers
        # Flatten all samples into a single list for the runner functions.
        data = [s for samples in all_samples.values() for s in samples]
        print(f"  campioni totali (tutti i dataset): {len(data)}")
    else:
        train_data, val_data, test_data, _ = load_all_datasets_vae(
            train_frac=cfg.get("train_frac", 0.70),
            val_frac=cfg.get("val_frac", 0.15),
            seed=cfg.get("seed", 42),
            max_nodes=cfg.get("max_nodes", 12),
        )
        data = {"train": train_data, "val": val_data, "test": test_data}[args.split]
        print(f"  campioni nello split '{args.split}': {len(data)}")

    out_dir = Path(args.out_dir)

    if not args.only_spec and not args.only_latent:
        print("\n[Ramo A] Circuit encoder  G -> z^c -> decoder...")
        t0 = time.time()
        results_circ, used_ds_ids_c, ds_map_c = run_circuit_encoder_observables(
            model, data, scalers, device, batch_size=args.batch_size, stochastic=args.stochastic,
            use_permutation_eval=not args.no_permutation_eval, f_min=args.f_min, f_max=args.f_max,
            max_qultra_per_dataset=args.max_qultra_per_dataset)
        print(f"  completato in {time.time() - t0:.1f}s")
        print_observable_report("Circuit Encoder  (G -> z^c -> Ghat)", results_circ, used_ds_ids_c, ds_map_c)
        plot_topo_breakdown(results_circ, str(out_dir / "inference_circuit_enc_topo.png"), "circuit-enc", args.ckpt)
        plot_observable_scatter(results_circ, str(out_dir / "inference_circuit_enc_observables.png"), args.n_samples, "circuit-enc", args.ckpt)
        plot_observable_variable_scatters(results_circ, out_dir, "circuit-enc", args.ckpt)
        plot_random_sample_observable_comparison(results_circ, out_dir, "circuit-enc", args.ckpt,
                                                 n_random=args.random_observable_samples, seed=args.seed)
        if not args.no_plot_inductive_kappa_debug:
            plot_inductive_kappa_debug(results_circ, out_dir, "circuit-enc", args.ckpt,
                                       dataset_name=args.inductive_kappa_debug_dataset,
                                       n_show=args.random_observable_samples, seed=args.seed)
        plot_evaluator_style_observable_reports(results_circ, out_dir / "eval_style", "circuit",
                                                n_random=args.random_observable_samples, seed=args.seed)
        if not args.no_plot_expansion_debug:
            plot_expansion_debug_cases(
                model, data, scalers, results_circ, out_dir, device,
                branch="circuit", batch_size=args.batch_size, stochastic=args.stochastic,
                use_permutation_eval=not args.no_permutation_eval,
                f_min=args.f_min, f_max=args.f_max, seed=args.seed,
            )
        with open(out_dir / ("circuit_" + args.save_observable_json), "w") as jf:
            json.dump(_jsonable_physics_results(results_circ), jf, indent=2)

    if not args.only_circuit and not args.only_latent:
        print("\n[Ramo B] Spec encoder  obs -> z^s -> decoder...")
        t0 = time.time()
        results_spec, used_ds_ids_s, ds_map_s = run_spec_encoder_observables(
            model, data, scalers, device, batch_size=args.batch_size, stochastic=args.stochastic,
            use_permutation_eval=not args.no_permutation_eval, f_min=args.f_min, f_max=args.f_max,
            max_qultra_per_dataset=args.max_qultra_per_dataset)
        print(f"  completato in {time.time() - t0:.1f}s")
        print_observable_report("Spec Encoder  (obs -> z^s -> Ghat)", results_spec, used_ds_ids_s, ds_map_s)
        plot_topo_breakdown(results_spec, str(out_dir / "inference_spec_enc_topo.png"), "spec-enc", args.ckpt)
        plot_observable_scatter(results_spec, str(out_dir / "inference_spec_enc_observables.png"), args.n_samples, "spec-enc", args.ckpt)
        plot_observable_variable_scatters(results_spec, out_dir, "spec-enc", args.ckpt)
        plot_random_sample_observable_comparison(results_spec, out_dir, "spec-enc", args.ckpt,
                                                 n_random=args.random_observable_samples, seed=args.seed)
        if not args.no_plot_inductive_kappa_debug:
            plot_inductive_kappa_debug(results_spec, out_dir, "spec-enc", args.ckpt,
                                       dataset_name=args.inductive_kappa_debug_dataset,
                                       n_show=args.random_observable_samples, seed=args.seed)
        plot_evaluator_style_observable_reports(results_spec, out_dir / "eval_style", "spec",
                                                n_random=args.random_observable_samples, seed=args.seed)
        if not args.no_plot_expansion_debug:
            plot_expansion_debug_cases(
                model, data, scalers, results_spec, out_dir, device,
                branch="spec", batch_size=args.batch_size, stochastic=args.stochastic,
                use_permutation_eval=not args.no_permutation_eval,
                f_min=args.f_min, f_max=args.f_max, seed=args.seed,
            )
        with open(out_dir / ("spec_" + args.save_observable_json), "w") as jf:
            json.dump(_jsonable_physics_results(results_spec), jf, indent=2)

    if args.plot_topo_errors:
        if not args.inference_only and not args.only_latent:
            print("\nATTENZIONE: --plot-topo-errors richiede --inference-only.")
        else:
            ds_err = args.topo_errors_dataset
            print(f"\n[Plot topologia errori] dataset={ds_err}  n={args.topo_errors_n}")
            for branch_name in (["circuit"] if args.only_circuit else
                                ["spec"]    if args.only_spec    else
                                ["circuit", "spec"]):
                plot_topology_errors(
                    vae=model, data=data, scalers=scalers, ds_name=ds_err,
                    out_path=str(out_dir / f"topo_errors_{ds_err}_{branch_name}.png"),
                    device=device, n_show=args.topo_errors_n,
                    batch_size=args.batch_size, stochastic=args.stochastic,
                    branch=branch_name,
                )

    if args.plot_latent_samples:
        print(f"\n[Plot latent samples]  n={args.latent_n}  stochastic={args.stochastic}")
        plot_latent_samples(
            vae=model, out_path=str(out_dir / "latent_samples.png"),
            device=device, n_show=args.latent_n,
            stochastic=args.stochastic, seed=args.seed,
        )

    print("\nDone.\n")


if __name__ == "__main__":
    main()
