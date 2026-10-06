#!/usr/bin/env python3
"""
Section 4.3.3 - Coherent ablation study for cQED-Gen.

Run from the repository root, e.g.

    python analysis_4_3_3_ablation_study_revised.py --plan core

DESIGN PRINCIPLE
================
This ablation is tied to the quantities already analysed in the thesis:
  (i) latent-space organisation and circuit/specification alignment;
  (ii) unconditional generation validity and novelty;
  (iii) seen-circuit inverse design, H -> z_s -> G, evaluated structurally and
        through QuLTRA;
  (iv) decoder-side superconducting-circuit constraints.

Latent-space optimisation (CEM) is intentionally NOT mixed into the training
objective ablation, because it is an inference-time refinement stage already
studied separately on unseen/stress-test targets.

All trainable variants are trained for EXACTLY 100 epochs by default. The full
baseline is retrained with the same 100-epoch budget, seed and data split as the
ablated models; the long production checkpoint is used only as a configuration
source.

STUDY A - TRAINING OBJECTIVES
=============================
  Full-100ep
  w/o KL            beta_max = 0
  w/o alignment     align_scale = 0
  w/o InfoNCE       nce_scale = 0
  w/o classifier    cg_scale = 0

Metrics:
  Latent space:
    paired cos(z_c,z_s), shuffled negative cosine, cosine margin,
    paired ||z_c-z_s||_2.
  Unconditional generation:
    validity, novelty.
  Seen-circuit inverse design:
    G->z_c->G exact typed-topology reconstruction,
    H->z_s->G exact typed-topology match,
    QuLTRA evaluability and physical Hamiltonian success A_H^(epsilon).

STUDY B - DECODER PHYSICS CONSTRAINTS
=====================================
No retraining. The same Full-100ep checkpoint is decoded with:
  full, no_node_mask, no_edge_constraints, no_degree_tracker, none.
The independent post-generation validator remains strict.

STUDY C - OPTIONAL LATENT DIMENSION
===================================
A compact sensitivity sweep only:
  d_z in {32, 64, 128}
with the same 100-epoch budget.

Outputs:
  ablation_objectives.csv / .tex
  figure41_ablation_losses.pdf
  figure41b_latent_alignment_ablation.pdf
  physical_constraints_ablation.csv / .tex
  figure_physical_constraints_ablation.pdf
  latent_dimension_sensitivity.csv / .tex (optional)
  figure_latent_dimension_sensitivity.pdf (optional)
  ablation_study_summary.json
"""


from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Repository discovery / imports
# -----------------------------------------------------------------------------


def find_repo_root() -> Path:
    here = Path.cwd().resolve()
    candidates = [here, Path(__file__).resolve().parent]
    for start in candidates:
        p = start
        for _ in range(5):
            if (p / "train_vae.py").exists() and (p / "src" / "vae_model" / "vae.py").exists():
                return p
            p = p.parent
    raise FileNotFoundError(
        "Repository root not found. Run this script from the cQED_Gen base directory."
    )


REPO_ROOT = find_repo_root()
import sys
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import train_vae  # noqa: E402
import inference_circuit_elements as ice  # noqa: E402
try:
    import inference_hamiltonian_qultra as ihq  # noqa: E402
    _QULTRA_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover
    ihq = None
    _QULTRA_IMPORT_ERROR = exc

from data_loader.loader_vae import load_all_datasets_vae  # noqa: E402
from data_loader.schema import train_datasets  # noqa: E402
from vae_model.decoder import data_to_graph_ns  # noqa: E402
import vae_model.decoder as decoder_mod  # noqa: E402
from circuit2graph.definitions import SubgType, SUBG_DEFS  # noqa: E402
from circuit2graph.topology import CQEDTopology  # noqa: E402
from circuit2graph.expansion import expand_topology  # noqa: E402
from circuit2graph.constraints import exists_valid_macro_graph, is_compatible  # noqa: E402


PHYSICAL_TYPES = frozenset({SubgType.TRANSMON, SubgType.RESONATOR, SubgType.FEEDLINE})
COUPLER_TYPES = frozenset({SubgType.C_COUPLER, SubgType.I_COUPLER})


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_mean(x: Iterable[float]) -> float:
    a = np.asarray(list(x), dtype=float)
    a = a[np.isfinite(a)]
    return float(np.mean(a)) if a.size else float("nan")


def safe_std(x: Iterable[float], ddof: int = 1) -> float:
    a = np.asarray(list(x), dtype=float)
    a = a[np.isfinite(a)]
    if a.size <= ddof:
        return 0.0 if a.size == 1 else float("nan")
    return float(np.std(a, ddof=ddof))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def json_dump(obj: Any, path: Path) -> None:
    def conv(x: Any) -> Any:
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, dict):
            return {str(k): conv(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [conv(v) for v in x]
        return x
    path.write_text(json.dumps(conv(obj), indent=2, sort_keys=True), encoding="utf-8")


def fmt_pm(mean: float, std: float, percent: bool = True) -> str:
    if not np.isfinite(mean):
        return "--"
    if percent:
        return f"{100.0*mean:.1f} \\pm {100.0*std:.1f}"
    return f"{mean:.4g} \\pm {std:.4g}"


def latex_escape(s: str) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%")


# -----------------------------------------------------------------------------
# Strict validity / novelty, intentionally independent from decoder masking
# -----------------------------------------------------------------------------


def graph_ns_to_macro_topology(g: SimpleNamespace, name: str) -> CQEDTopology:
    topo = CQEDTopology(name)
    dirs = list(getattr(g, "direction", []))
    nodes = []
    for i, st_int in enumerate(getattr(g, "node_types", [])):
        st = SubgType(int(st_int))
        attrs: dict[str, float] = {}
        for attr in SUBG_DEFS[st].attrs:
            if attr == "dir":
                d = dirs[i] if i < len(dirs) else 1.0
                attrs[attr] = 1.0 if float(d) >= 0 else -1.0
            else:
                attrs[attr] = 1.0
        nodes.append(topo.add_node(st, attrs, label=f"{st.name}_{i}"))
    for u, v in getattr(g, "edges", []):
        u, v = int(u), int(v)
        if 0 <= u < len(nodes) and 0 <= v < len(nodes) and u != v:
            topo.add_edge(nodes[u], nodes[v])
    return topo


def is_macro_connected(n_nodes: int, edges: list[tuple[int, int]]) -> bool:
    if n_nodes <= 0:
        return False
    if n_nodes == 1:
        return True
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    G.add_edges_from((int(u), int(v)) for u, v in edges if int(u) != int(v))
    return nx.is_connected(G)


def macro_structural_validity(g: SimpleNamespace) -> tuple[bool, list[str]]:
    node_types = [int(x) for x in getattr(g, "node_types", [])]
    directions = [float(x) for x in getattr(g, "direction", [0.0] * len(node_types))]
    if len(directions) < len(node_types):
        directions += [0.0] * (len(node_types) - len(directions))
    edges = [(int(u), int(v)) for u, v in getattr(g, "edges", [])]
    reasons: list[str] = []
    n = len(node_types)
    if n == 0:
        return False, ["empty_macro_graph"]
    try:
        if not exists_valid_macro_graph(node_types, directions):
            reasons.append("no_valid_macro_completion")
    except Exception:
        reasons.append("macro_completion_error")
    if not is_macro_connected(n, edges):
        reasons.append("macro_disconnected")
    for u, v in edges:
        if not (0 <= u < n and 0 <= v < n) or u == v:
            reasons.append("invalid_macro_edge")
            continue
        try:
            if not is_compatible(node_types[u], directions[u], node_types[v], directions[v]):
                reasons.append("incompatible_macro_edge")
        except Exception:
            reasons.append("incompatible_macro_edge")
    return len(reasons) == 0, sorted(set(reasons))


def primitive_graph(topo: CQEDTopology) -> nx.Graph:
    G = nx.Graph()
    for n in topo._nodes:
        G.add_node(int(n.node_id), subg_type=int(n.subg_type))
    for u, v in topo._edges:
        G.add_edge(int(u), int(v))
    return G


def primitive_structural_validity(topo: CQEDTopology) -> tuple[bool, list[str]]:
    G = primitive_graph(topo)
    reasons: list[str] = []
    if G.number_of_nodes() == 0:
        return False, ["empty_primitive_graph"]
    if nx.number_of_selfloops(G) > 0:
        reasons.append("self_loop")
    if not nx.is_connected(G):
        reasons.append("primitive_disconnected")
    by_id = {int(n.node_id): n for n in topo._nodes}
    for u, v in topo._edges:
        a = by_id[int(u)].subg_type
        b = by_id[int(v)].subg_type
        if a in PHYSICAL_TYPES and b in PHYSICAL_TYPES:
            reasons.append("direct_physical_edge")
        if a in COUPLER_TYPES and b in COUPLER_TYPES:
            reasons.append("coupler_coupler_edge")
    for n in topo._nodes:
        st = n.subg_type
        deg = G.degree[int(n.node_id)]
        if st in COUPLER_TYPES and deg != 2:
            reasons.append("coupler_degree_not_two")
        if st == SubgType.I_COUPLER:
            nbr = [by_id[int(j)].subg_type for j in G.neighbors(int(n.node_id))]
            if len(nbr) != 2 or set(nbr) != {SubgType.RESONATOR, SubgType.FEEDLINE}:
                reasons.append("inductive_coupler_invalid_neighbors")
        if st == SubgType.FEEDLINE and deg > 2:
            reasons.append("feedline_degree_gt_two")
    return len(reasons) == 0, sorted(set(reasons))


def analyze_generated_topology(g: SimpleNamespace, name: str) -> tuple[bool, CQEDTopology | None, list[str]]:
    macro_ok, macro_reasons = macro_structural_validity(g)
    try:
        macro = graph_ns_to_macro_topology(g, name=f"{name}_macro")
        primitive = expand_topology(macro, validate=False)
        prim_ok, prim_reasons = primitive_structural_validity(primitive)
        return bool(macro_ok and prim_ok), primitive, sorted(set(macro_reasons + prim_reasons))
    except Exception as exc:
        return False, None, sorted(set(macro_reasons + [f"expansion:{type(exc).__name__}"]))


def topology_signature(G: nx.Graph) -> tuple[int, int, tuple[int, ...]]:
    types = tuple(sorted(int(G.nodes[n]["subg_type"]) for n in G.nodes))
    return G.number_of_nodes(), G.number_of_edges(), types


def make_training_topology_index() -> dict[tuple[int, int, tuple[int, ...]], list[tuple[str, nx.Graph]]]:
    out: dict[tuple[int, int, tuple[int, ...]], list[tuple[str, nx.Graph]]] = defaultdict(list)
    for name, defn in train_datasets().items():
        G = primitive_graph(defn.topology_fn())
        out[topology_signature(G)].append((name, G))
    return dict(out)


def match_training_topology(primitive: CQEDTopology, index: dict) -> str | None:
    G = primitive_graph(primitive)
    candidates = index.get(topology_signature(G), [])
    nm = nx.algorithms.isomorphism.categorical_node_match("subg_type", -1)
    for family, target in candidates:
        if nx.is_isomorphic(G, target, node_match=nm):
            return family
    return None


def typed_graph(g: Any) -> nx.Graph:
    G = nx.Graph()
    for i, t in enumerate(getattr(g, "node_types", [])):
        G.add_node(int(i), subg_type=int(t))
    G.add_edges_from((int(u), int(v)) for u, v in getattr(g, "edges", []) if int(u) != int(v))
    return G


def exact_typed_isomorphic(g_true: Any, g_pred: Any) -> bool:
    A, B = typed_graph(g_true), typed_graph(g_pred)
    if A.number_of_nodes() != B.number_of_nodes() or A.number_of_edges() != B.number_of_edges():
        return False
    nm = nx.algorithms.isomorphism.categorical_node_match("subg_type", -1)
    return bool(nx.is_isomorphic(A, B, node_match=nm))


# -----------------------------------------------------------------------------
# Constraint-ablation context manager
# -----------------------------------------------------------------------------


@contextlib.contextmanager
def decoder_constraint_mode(mode: str):
    """Temporarily patch ONLY decoder-side inference masks.

    The strict post-generation validator above is never patched.
    """
    valid_modes = {"full", "no_node_mask", "no_edge_constraints", "no_degree_tracker", "none"}
    if mode not in valid_modes:
        raise ValueError(f"Unknown constraint mode {mode!r}")
    if mode == "full":
        yield
        return

    originals = {
        "valid_next_macro_types": decoder_mod.valid_next_macro_types,
        "exists_valid_macro_graph": decoder_mod.exists_valid_macro_graph,
        "build_forbidden_edge_mask": decoder_mod.build_forbidden_edge_mask,
        "choose_compatible_macro_edge_orientation": decoder_mod.choose_compatible_macro_edge_orientation,
        "outer_ports": decoder_mod.outer_ports,
    }

    def all_next(node_types, directions, candidates=None, **kwargs):
        return list(candidates if candidates is not None else range(decoder_mod.N_SUBTYPES))

    def always_exists(*args, **kwargs):
        return True

    def no_forbidden(node_types, directions):
        n = len(node_types)
        return [False] * (n * (n - 1) // 2)

    def direct_orientation(u, v, node_types, directions):
        return int(u), int(v)

    def no_coupler_ports(subg_type, direction):
        return SubgType.TRANSMON, SubgType.TRANSMON

    try:
        if mode in {"no_node_mask", "none"}:
            decoder_mod.valid_next_macro_types = all_next
            decoder_mod.exists_valid_macro_graph = always_exists
        if mode in {"no_edge_constraints", "none"}:
            decoder_mod.build_forbidden_edge_mask = no_forbidden
            decoder_mod.choose_compatible_macro_edge_orientation = direct_orientation
            decoder_mod.outer_ports = no_coupler_ports
        elif mode == "no_degree_tracker":
            decoder_mod.outer_ports = no_coupler_ports
        yield
    finally:
        for k, v in originals.items():
            setattr(decoder_mod, k, v)


# -----------------------------------------------------------------------------
# Loading / balanced samples
# -----------------------------------------------------------------------------


def load_reference_config(checkpoint: Path) -> dict[str, Any]:
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = dict(ck.get("config", {}))
    if not cfg:
        cfg = dict(train_vae.DEFAULT_CONFIG)
    return cfg


def balanced_limit(samples: list[Any], per_family: int, seed: int) -> list[Any]:
    if per_family <= 0:
        return list(samples)
    by: dict[str, list[Any]] = defaultdict(list)
    for s in samples:
        by[s.dataset_name].append(s)
    rng = np.random.default_rng(seed)
    out = []
    for name in sorted(by):
        xs = by[name]
        if len(xs) <= per_family:
            out.extend(xs)
        else:
            idx = rng.choice(len(xs), size=per_family, replace=False)
            out.extend(xs[int(i)] for i in idx)
    return out


@torch.no_grad()
def latent_alignment_metrics(model, samples: list[Any], scalers: dict, device: torch.device, batch_size: int, per_family: int, seed: int) -> dict[str, float]:
    """Paired circuit/specification latent diagnostics on the fixed test set."""
    subset = balanced_limit(samples, per_family, seed)
    if not subset:
        return {k: float("nan") for k in [
            "latent_cos_pos_mean", "latent_cos_pos_std",
            "latent_cos_neg_mean", "latent_cos_neg_std",
            "latent_cos_margin_mean", "latent_cos_margin_std",
            "latent_l2_pair_mean", "latent_l2_pair_std"]} | {"latent_n": 0}

    zc_all, zs_all = [], []
    model.eval()
    for start in range(0, len(subset), batch_size):
        batch = subset[start:start + batch_size]
        zc, _, _ = model.encode(batch, scalers)
        obs_vals = torch.stack([s.obs_vals for s in batch]).to(device)
        obs_mask = torch.stack([s.obs_mask for s in batch]).to(device)
        zs, _, _ = model.spec_encoder.encode(obs_vals, obs_mask)
        zc_all.append(zc.detach())
        zs_all.append(zs.detach())
    zc = torch.cat(zc_all, dim=0)
    zs = torch.cat(zs_all, dim=0)
    pos = F.cosine_similarity(zc, zs, dim=-1)
    l2 = torch.linalg.vector_norm(zc-zs, dim=-1)
    n = int(zc.shape[0])
    if n > 1:
        rng = np.random.default_rng(seed+991)
        perm = rng.permutation(n)
        if np.any(perm == np.arange(n)):
            perm = np.roll(perm, 1)
        perm_t = torch.tensor(perm, dtype=torch.long, device=zc.device)
        neg = F.cosine_similarity(zc[perm_t], zs, dim=-1)
    else:
        neg = torch.full_like(pos, float("nan"))
    margin = pos-neg
    def ms(t):
        a=t.detach().float().cpu().numpy(); a=a[np.isfinite(a)]
        if not a.size: return float("nan"), float("nan")
        return float(a.mean()), float(a.std(ddof=1)) if a.size>1 else 0.0
    pm,ps=ms(pos); nm,ns=ms(neg); mm,msd=ms(margin); lm,ls=ms(l2)
    return {
        "latent_cos_pos_mean":pm, "latent_cos_pos_std":ps,
        "latent_cos_neg_mean":nm, "latent_cos_neg_std":ns,
        "latent_cos_margin_mean":mm, "latent_cos_margin_std":msd,
        "latent_l2_pair_mean":lm, "latent_l2_pair_std":ls, "latent_n":n}


# -----------------------------------------------------------------------------
# Training orchestration
# -----------------------------------------------------------------------------


TRAINABLE_CFG_KEYS = set(train_vae.DEFAULT_CONFIG.keys())


def clean_training_cfg(reference_cfg: dict[str, Any], overrides: dict[str, Any], save_path: Path, args) -> dict[str, Any]:
    cfg = dict(train_vae.DEFAULT_CONFIG)
    cfg.update({k: v for k, v in reference_cfg.items() if k in TRAINABLE_CFG_KEYS})
    cfg.update(overrides)
    cfg["save_path"] = str(save_path)
    cfg["plot_path"] = str(save_path.with_suffix(".png"))
    cfg["num_workers"] = 0
    cfg["val_every"] = 1
    cfg["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        cfg["batch_size"] = int(args.batch_size)
    if args.no_compile:
        cfg["use_compile"] = False
    if args.no_amp:
        cfg["use_amp"] = False
    return cfg


def train_variant(name: str, overrides: dict[str, Any], reference_cfg: dict[str, Any], out_dir: Path, args) -> Path:
    ckpt = out_dir / "checkpoints" / f"{name}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    if ckpt.exists() and not args.force_retrain:
        print(f"[reuse] {name}: {ckpt}")
        return ckpt

    cfg = clean_training_cfg(reference_cfg, overrides, ckpt, args)
    print("\n" + "=" * 90)
    print(f"TRAINING VARIANT: {name}")
    print(f"Overrides: {overrides}")
    print("=" * 90)

    log_path = out_dir / "training_logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # train_vae.train is the repository training function itself, so no duplicate
    # implementation of data loading / optimizer / scheduler is introduced here.
    with log_path.open("w", encoding="utf-8") as log:
        old_out, old_err = sys.stdout, sys.stderr
        try:
            class Tee:
                def __init__(self, *files): self.files = files
                def write(self, x):
                    for f in self.files:
                        f.write(x); f.flush()
                def flush(self):
                    for f in self.files: f.flush()
            sys.stdout = Tee(old_out, log)
            sys.stderr = Tee(old_err, log)
            model = train_vae.train(cfg)
        finally:
            sys.stdout, sys.stderr = old_out, old_err

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not ckpt.exists():
        raise FileNotFoundError(f"Training finished but checkpoint was not written: {ckpt}")
    return ckpt


# -----------------------------------------------------------------------------
# Prior sampling: validity / novelty
# -----------------------------------------------------------------------------


@torch.no_grad()
def prior_sampling_metrics(model, device, mode: str, n_runs: int, samples_per_run: int, batch_size: int, seed: int) -> dict[str, Any]:
    train_index = make_training_topology_index()
    run_rows = []
    reason_total = Counter()

    with decoder_constraint_mode(mode):
        for run in range(n_runs):
            run_seed = seed + 10000 + run
            set_seed(run_seed)
            n_valid = n_novel = n_total = 0
            reasons = Counter()
            for start in range(0, samples_per_run, batch_size):
                n = min(batch_size, samples_per_run - start)
                z = torch.randn(n, model.nz, device=device)
                graphs = model.decoder.decode(z, stochastic=True)
                for j, g in enumerate(graphs):
                    n_total += 1
                    valid, prim, why = analyze_generated_topology(g, f"prior_{run}_{start+j}")
                    if valid and prim is not None:
                        n_valid += 1
                        if match_training_topology(prim, train_index) is None:
                            n_novel += 1
                    else:
                        reasons.update(why)
                        reason_total.update(why)
            row = {
                "run": run + 1,
                "A_valid": n_valid / max(n_total, 1),
                "A_novel_all": n_novel / max(n_total, 1),
                "A_novel_given_valid": n_novel / max(n_valid, 1),
                "n_generated": n_total,
                "n_valid": n_valid,
                "n_novel": n_novel,
                "invalid_reasons": json.dumps(dict(reasons), sort_keys=True),
            }
            run_rows.append(row)

    valid = [r["A_valid"] for r in run_rows]
    novel = [r["A_novel_all"] for r in run_rows]
    novel_v = [r["A_novel_given_valid"] for r in run_rows]
    return {
        "A_valid_mean": safe_mean(valid),
        "A_valid_std": safe_std(valid),
        "A_novel_mean": safe_mean(novel),
        "A_novel_std": safe_std(novel),
        "A_novel_given_valid_mean": safe_mean(novel_v),
        "A_novel_given_valid_std": safe_std(novel_v),
        "prior_runs": run_rows,
        "invalid_reasons": dict(reason_total),
    }


# -----------------------------------------------------------------------------
# Topology accuracy / match, both branches
# -----------------------------------------------------------------------------


@torch.no_grad()
def topology_outcomes(model, samples: list[Any], scalers: dict, device, mode: str, batch_size: int) -> dict[str, list[int]]:
    model.eval()
    out_c: list[int] = []
    out_s: list[int] = []
    with decoder_constraint_mode(mode):
        for start in range(0, len(samples), batch_size):
            batch = samples[start:start + batch_size]
            zc, _, _ = model.encode(batch, scalers)
            obs_vals = torch.stack([s.obs_vals for s in batch]).to(device)
            obs_mask = torch.stack([s.obs_mask for s in batch]).to(device)
            zs, _, _ = model.spec_encoder.encode(obs_vals, obs_mask)
            # Topology metrics do not need continuous parameter prediction.
            gc = model.decoder.decode(zc, stochastic=False)
            gs = model.decoder.decode(zs, stochastic=False)
            for s, pc, ps in zip(batch, gc, gs):
                gtrue = data_to_graph_ns(s, scalers[s.dataset_name].param_scaler)
                out_c.append(int(exact_typed_isomorphic(gtrue, pc)))
                out_s.append(int(exact_typed_isomorphic(gtrue, ps)))
    return {"circuit": out_c, "spec": out_s}


def repeated_binary_summary(outcomes: list[int], n_runs: int, samples_per_run: int, seed: int) -> tuple[float, float, list[float]]:
    x = np.asarray(outcomes, dtype=float)
    if x.size == 0:
        return float("nan"), float("nan"), []
    rng = np.random.default_rng(seed)
    vals = []
    m = min(samples_per_run, x.size)
    for _ in range(n_runs):
        idx = rng.choice(x.size, size=m, replace=False if m <= x.size else True)
        vals.append(float(np.mean(x[idx])))
    return safe_mean(vals), safe_std(vals), vals


# -----------------------------------------------------------------------------
# QuLTRA physical success without requiring topology equality
# -----------------------------------------------------------------------------


def inverse_graph_attrs(g: Any, param_scaler) -> SimpleNamespace:
    out = SimpleNamespace()
    out.node_types = list(getattr(g, "node_types", []))
    out.edges = [tuple(e) for e in getattr(g, "edges", [])]
    out.direction = list(getattr(g, "direction", []))
    out.attrs = []
    for i, st_int in enumerate(out.node_types):
        st = SubgType(int(st_int))
        src = dict(g.attrs[i]) if hasattr(g, "attrs") and i < len(g.attrs) else {}
        names, scaled_vals = [], []
        for name in SUBG_DEFS[st].attrs:
            if name == "dir":
                continue
            names.append(name)
            scaled_vals.append(float(src.get(name, float("nan"))))
        if names:
            phys = param_scaler.inverse_transform(np.asarray(scaled_vals, dtype=float), names)
            d = {name: float(v) for name, v in zip(names, phys)}
        else:
            d = {}
        if "dir" in SUBG_DEFS[st].attrs:
            if i < len(out.direction):
                d["dir"] = 1.0 if float(out.direction[i]) >= 0 else -1.0
            else:
                d["dir"] = 1.0 if float(src.get("dir", 1.0)) >= 0 else -1.0
        out.attrs.append(d)
    return out


def rel_percent(true: float, pred: float, floor: float = 1e-30) -> float:
    return 100.0 * abs(float(pred) - float(true)) / max(abs(float(true)), float(floor))


def hamiltonian_distance(true_obs: dict, pred_aligned: dict, args) -> tuple[float, dict[str, float]]:
    errors_by_group: dict[str, list[float]] = defaultdict(list)

    tf = np.asarray(true_obs.get("frequencies", []), dtype=float)
    pf = np.asarray(pred_aligned.get("frequencies", []), dtype=float)
    for i in range(len(tf)):
        if not np.isfinite(tf[i]):
            continue
        if i >= len(pf) or not np.isfinite(pf[i]):
            return float("inf"), {"missing_frequency": float("inf")}
        errors_by_group["frequency"].append(rel_percent(tf[i], pf[i]))

    tk = np.asarray(true_obs.get("kappa", []), dtype=float)
    pk = np.asarray(pred_aligned.get("kappa", []), dtype=float)
    for i in range(len(tk)):
        if not np.isfinite(tk[i]) or abs(tk[i]) < args.kappa_floor:
            continue
        if i >= len(pk) or not np.isfinite(pk[i]):
            return float("inf"), {"missing_kappa": float("inf")}
        errors_by_group["kappa"].append(rel_percent(tk[i], pk[i]))

    tc = np.asarray(true_obs.get("chi", np.zeros((0, 0))), dtype=float)
    pc = np.asarray(pred_aligned.get("chi", np.zeros((0, 0))), dtype=float)
    n_true = tc.shape[0] if tc.ndim == 2 else 0
    for i in range(n_true):
        for j in range(i, n_true):
            if not np.isfinite(tc[i, j]) or abs(tc[i, j]) < args.chi_floor:
                continue
            if pc.ndim != 2 or i >= pc.shape[0] or j >= pc.shape[1] or not np.isfinite(pc[i, j]):
                return float("inf"), {"missing_chi": float("inf")}
            group = "self_kerr" if i == j else "cross_kerr"
            errors_by_group[group].append(rel_percent(tc[i, j], pc[i, j]))

    tolerances = {
        "frequency": args.eps_frequency,
        "self_kerr": args.eps_self_kerr,
        "cross_kerr": args.eps_cross_kerr,
        "kappa": args.eps_kappa,
    }
    normalized = []
    max_group_error = {}
    for group, vals in errors_by_group.items():
        if vals:
            e = float(max(vals))
            max_group_error[group] = e
            normalized.append(e / max(tolerances[group], 1e-12))
    if not normalized:
        return float("inf"), max_group_error
    return float(max(normalized)), max_group_error


@torch.no_grad()
def physical_success_outcomes(model, samples: list[Any], scalers: dict, device, mode: str, batch_size: int, args) -> dict[str, Any]:
    if args.no_qultra:
        return {"outcomes": [], "n_attempted": 0, "n_simulated": 0, "failures": {"disabled": 1}, "group_errors": {}}
    if ihq is None:
        raise RuntimeError(f"Could not import inference_hamiltonian_qultra.py: {_QULTRA_IMPORT_ERROR}")
    qu = ihq.import_qultra()

    by_family: dict[str, list[Any]] = defaultdict(list)
    for s in samples:
        by_family[s.dataset_name].append(s)
    rng = np.random.default_rng(args.seed + 777)
    selected = []
    for fam in sorted(by_family):
        xs = by_family[fam]
        n = len(xs) if args.qultra_max_per_family <= 0 else min(len(xs), args.qultra_max_per_family)
        if n == len(xs):
            selected.extend(xs)
        else:
            idx = rng.choice(len(xs), size=n, replace=False)
            selected.extend(xs[int(i)] for i in idx)

    outcomes: list[int] = []
    failures = Counter()
    group_error_values: dict[str, list[float]] = defaultdict(list)
    n_simulated = 0

    model.eval()
    with decoder_constraint_mode(mode):
        for start in range(0, len(selected), batch_size):
            batch = selected[start:start + batch_size]
            obs_vals = torch.stack([s.obs_vals for s in batch]).to(device)
            obs_mask = torch.stack([s.obs_mask for s in batch]).to(device)
            zs, _, _ = model.spec_encoder.encode(obs_vals, obs_mask)
            graphs = model.decode(zs, stochastic=False)
            for s, g in zip(batch, graphs):
                try:
                    ds_scaler = scalers[s.dataset_name]
                    g_phys = inverse_graph_attrs(g, ds_scaler.param_scaler)
                    topo = ihq._graph_ns_to_cqed_topology(g_phys, name=f"ablation_{s.dataset_name}")
                    pred = ihq._simulate_topology_arrays(topo, qu, args.f_min, args.f_max)
                    if not pred.get("ok", False):
                        outcomes.append(0)
                        failures[f"qultra:{pred.get('stage', 'unknown')}"] += 1
                        continue
                    true_obs = ihq._sample_true_observables_from_dataset(s, ds_scaler.obs_scaler)
                    aligned = ihq._align_pred_modes_to_true(true_obs, pred)
                    dH, group_errors = hamiltonian_distance(true_obs, aligned, args)
                    n_simulated += 1
                    outcomes.append(int(dH <= 1.0))
                    for k, v in group_errors.items():
                        group_error_values[k].append(float(v))
                except Exception as exc:
                    outcomes.append(0)
                    failures[type(exc).__name__] += 1

    return {
        "outcomes": outcomes,
        "n_attempted": len(selected),
        "n_simulated": n_simulated,
        "failures": dict(failures),
        "group_errors": {k: v for k, v in group_error_values.items()},
    }


# -----------------------------------------------------------------------------
# One checkpoint -> all thesis metrics
# -----------------------------------------------------------------------------


@dataclass
class EvalResult:
    row: dict[str, Any]
    details: dict[str, Any]


def evaluate_checkpoint(checkpoint: Path, label: str, category: str, constraint_mode: str, test_data: list[Any], device, args) -> EvalResult:
    print("\n" + "-" * 90)
    print(f"EVALUATE: {label} | constraints={constraint_mode} | {checkpoint}")
    print("-" * 90)
    model, scalers, cfg = ice.load_checkpoint(str(checkpoint), device)
    model.eval()

    topo_pool = balanced_limit(test_data, args.topology_pool_per_family, args.seed + 13)
    top_out = topology_outcomes(model, topo_pool, scalers, device, constraint_mode, args.eval_batch_size)
    latent = latent_alignment_metrics(model, test_data, scalers, device, args.eval_batch_size, args.latent_pool_per_family, args.seed + 404)
    g_mean, g_std, g_runs = repeated_binary_summary(top_out["circuit"], args.metric_runs, args.metric_samples_per_run, args.seed + 101)
    h_mean, h_std, h_runs = repeated_binary_summary(top_out["spec"], args.metric_runs, args.metric_samples_per_run, args.seed + 202)

    prior = prior_sampling_metrics(
        model, device, constraint_mode,
        n_runs=args.prior_runs,
        samples_per_run=args.prior_samples_per_run,
        batch_size=args.eval_batch_size,
        seed=args.seed,
    )

    phys = physical_success_outcomes(model, test_data, scalers, device, constraint_mode, args.eval_batch_size, args)
    ah_mean, ah_std, ah_runs = repeated_binary_summary(
        phys["outcomes"], args.metric_runs, args.metric_samples_per_run, args.seed + 303
    ) if phys["outcomes"] else (float("nan"), float("nan"), [])

    row = {
        "category": category,
        "label": label,
        "checkpoint": str(checkpoint),
        "constraint_mode": constraint_mode,
        "nz": cfg.get("nz"),
        "beta_max": cfg.get("beta_max"),
        "align_scale": cfg.get("align_scale"),
        "nce_scale": cfg.get("nce_scale"),
        "tau": cfg.get("tau"),
        "cg_scale": cfg.get("cg_scale"),
        "A_valid_mean": prior["A_valid_mean"],
        "A_valid_std": prior["A_valid_std"],
        "A_novel_mean": prior["A_novel_mean"],
        "A_novel_std": prior["A_novel_std"],
        "A_topo_G_mean": g_mean,
        "A_topo_G_std": g_std,
        "A_topo_H_mean": h_mean,
        "A_topo_H_std": h_std,
        "A_H_epsilon_mean": ah_mean,
        "A_H_epsilon_std": ah_std,
        "A_qultra_ok": (float(phys["n_simulated"]) / max(int(phys["n_attempted"]), 1) if not args.no_qultra else float("nan")),
        "qultra_attempted": phys["n_attempted"],
        "qultra_simulated": phys["n_simulated"],
        **latent,
        "eps_frequency_percent": args.eps_frequency,
        "eps_self_kerr_percent": args.eps_self_kerr,
        "eps_cross_kerr_percent": args.eps_cross_kerr,
        "eps_kappa_percent": args.eps_kappa,
        "chi_floor": args.chi_floor,
        "kappa_floor": args.kappa_floor,
    }
    details = {
        "row": row,
        "topology_runs_G": g_runs,
        "topology_runs_H": h_runs,
        "prior": prior,
        "physical_success_runs": ah_runs,
        "physical_success": phys,
        "latent_alignment": latent,
    }

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return EvalResult(row=row, details=details)


# -----------------------------------------------------------------------------
# Experiment definitions
# -----------------------------------------------------------------------------


CORE_ABLATIONS = {
    "full_100ep": {},
    "no_KL": {"beta_start": 0.0, "beta_max": 0.0, "warmup_steps": 0},
    "no_alignment": {"align_scale": 0.0},
    "no_InfoNCE": {"nce_scale": 0.0},
    "no_CG": {"cg_scale": 0.0},
}

LATENT_DIM_VALUES = [32, 64, 128]

REFERENCE_KEYS = {
    "nz": 128,
    "beta_max": 1e-4,
    "align_scale": 0.5,
    "nce_scale": 0.5,
    "tau": 0.1,
    "cg_scale": 0.5,
}



def is_reference_override(overrides: dict[str, Any], ref_cfg: dict[str, Any]) -> bool:
    for k, v in overrides.items():
        rv = ref_cfg.get(k)
        if isinstance(v, float) or isinstance(rv, float):
            if not math.isclose(float(v), float(rv), rel_tol=1e-12, abs_tol=1e-15):
                return False
        elif v != rv:
            return False
    return True


def value_slug(v: Any) -> str:
    s = f"{v:g}" if isinstance(v, float) else str(v)
    return s.replace("-", "m").replace(".", "p").replace("+", "p")


# -----------------------------------------------------------------------------
# Tables / plots
# -----------------------------------------------------------------------------


MAIN_METRIC_COLUMNS = [
    ("A_valid_mean", "A_valid_std", "Validity"),
    ("A_novel_mean", "A_novel_std", "Novelty"),
    ("A_topo_H_mean", "A_topo_H_std", r"$A_{\rm topo}^{H\to G}$"),
    ("A_H_epsilon_mean", "A_H_epsilon_std", r"$A_H^{(\epsilon)}$"),
]


def plot_metric_bars(rows: list[dict[str, Any]], out_pdf: Path, title: str) -> None:
    if not rows: return
    labels=[r["label"] for r in rows]; x=np.arange(len(labels),dtype=float); width=0.18
    fig,ax=plt.subplots(figsize=(max(8.0,1.45*len(labels)+4.0),5.0))
    for j,(mkey,skey,mlabel) in enumerate(MAIN_METRIC_COLUMNS):
        vals=np.asarray([r.get(mkey,np.nan) for r in rows],dtype=float)
        errs=np.asarray([r.get(skey,np.nan) for r in rows],dtype=float)
        off=(j-(len(MAIN_METRIC_COLUMNS)-1)/2.0)*width
        ax.bar(x+off,vals,width=width,label=mlabel,yerr=errs,capsize=2)
    ax.set_xticks(x); ax.set_xticklabels(labels,rotation=18,ha="right"); ax.set_ylim(0,1.05)
    ax.set_ylabel("Fraction"); ax.set_title(title); ax.grid(axis="y",alpha=0.25); ax.legend(ncol=2,fontsize=9)
    fig.tight_layout(); out_pdf.parent.mkdir(parents=True,exist_ok=True); fig.savefig(out_pdf,bbox_inches="tight"); plt.close(fig)


def plot_latent_alignment(rows: list[dict[str, Any]], out_pdf: Path) -> None:
    if not rows: return
    labels=[r["label"] for r in rows]; x=np.arange(len(labels),dtype=float); w=0.24
    pos=np.asarray([r.get("latent_cos_pos_mean",np.nan) for r in rows]); pe=np.asarray([r.get("latent_cos_pos_std",np.nan) for r in rows])
    neg=np.asarray([r.get("latent_cos_neg_mean",np.nan) for r in rows]); ne=np.asarray([r.get("latent_cos_neg_std",np.nan) for r in rows])
    mar=np.asarray([r.get("latent_cos_margin_mean",np.nan) for r in rows]); me=np.asarray([r.get("latent_cos_margin_std",np.nan) for r in rows])
    fig,ax=plt.subplots(figsize=(max(8.2,1.5*len(labels)+4.0),5.0))
    ax.bar(x-w,pos,w,yerr=pe,capsize=2,label=r"paired $\cos(z_c,z_s)$")
    ax.bar(x,neg,w,yerr=ne,capsize=2,label=r"shuffled $\cos(z_c,z_s)$")
    ax.bar(x+w,mar,w,yerr=me,capsize=2,label="cosine margin")
    ax.axhline(0,linewidth=1); ax.set_xticks(x); ax.set_xticklabels(labels,rotation=18,ha="right")
    ax.set_ylabel("Cosine similarity / margin"); ax.set_title("Ablation of circuit--specification latent alignment"); ax.grid(axis="y",alpha=0.25); ax.legend(ncol=3,fontsize=9)
    fig.tight_layout(); fig.savefig(out_pdf,bbox_inches="tight"); plt.close(fig)


def plot_latent_dimension_sensitivity(rows: list[dict[str, Any]], out_pdf: Path) -> None:
    if not rows: return
    rows=sorted(rows,key=lambda r:int(r["nz"])); xs=np.asarray([int(r["nz"]) for r in rows],dtype=float)
    fig,ax=plt.subplots(figsize=(7.5,4.9))
    for mkey,skey,label in MAIN_METRIC_COLUMNS:
        y=np.asarray([float(r.get(mkey,np.nan)) for r in rows]); er=np.asarray([float(r.get(skey,np.nan)) for r in rows])
        ax.errorbar(xs,y,yerr=er,marker="o",capsize=2,label=label)
    ax.set_xlabel("Latent dimension $d_z$"); ax.set_ylabel("Fraction"); ax.set_ylim(0,1.05); ax.set_title("Sensitivity to latent-space dimensionality (100 epochs)"); ax.grid(alpha=0.25); ax.legend(ncol=2,fontsize=8)
    fig.tight_layout(); fig.savefig(out_pdf,bbox_inches="tight"); plt.close(fig)



def write_ablation_tex(rows: list[dict[str, Any]], path: Path, caption: str, label: str) -> None:
    lines=[r"\begin{table}[t]",r"    \centering",r"    \footnotesize",r"    \setlength{\tabcolsep}{3pt}",r"    \renewcommand{\arraystretch}{1.10}",r"    \resizebox{\textwidth}{!}{%",r"    \begin{tabular}{lccccccc}",r"        \toprule",r"        \textbf{Configuration} & \textbf{Validity} & \textbf{Novelty} & $A_{\mathrm{topo}}^{G\to G}$ & $A_{\mathrm{topo}}^{H\to G}$ & $A_H^{(\epsilon)}$ & $\cos(z_c,z_s)$ & \textbf{Cos. margin} \\",r"        \midrule"]
    for r in rows:
        vals=[latex_escape(r["label"]),f"${fmt_pm(r['A_valid_mean'],r['A_valid_std'])}\\%$",f"${fmt_pm(r['A_novel_mean'],r['A_novel_std'])}\\%$",f"${fmt_pm(r['A_topo_G_mean'],r['A_topo_G_std'])}\\%$",f"${fmt_pm(r['A_topo_H_mean'],r['A_topo_H_std'])}\\%$",f"${fmt_pm(r['A_H_epsilon_mean'],r['A_H_epsilon_std'])}\\%$" if np.isfinite(r['A_H_epsilon_mean']) else "--",f"${r.get('latent_cos_pos_mean',float('nan')):.3f}$",f"${r.get('latent_cos_margin_mean',float('nan')):.3f}$"]
        lines.append("        "+" & ".join(vals)+r" \\")
    lines += [r"        \bottomrule",r"    \end{tabular}}",f"    \\caption{{{caption}}}",f"    \\label{{{label}}}",r"\end{table}"]
    path.write_text("\n".join(lines)+"\n",encoding="utf-8")



def write_latent_dimension_tex(rows: list[dict[str, Any]], path: Path) -> None:
    rows=sorted(rows,key=lambda r:int(r["nz"]))
    lines=[r"\begin{table}[t]",r"    \centering",r"    \small",r"    \begin{tabular}{cccccc}",r"        \toprule",r"$d_z$ & \textbf{Validity} & \textbf{Novelty} & $A_{\mathrm{topo}}^{H\to G}$ & $A_H^{(\epsilon)}$ & $\cos(z_c,z_s)$ \\",r"        \midrule"]
    for r in rows:
        vals=[str(int(r["nz"])),f"${100*r['A_valid_mean']:.1f}$",f"${100*r['A_novel_mean']:.1f}$",f"${100*r['A_topo_H_mean']:.1f}$",f"${100*r['A_H_epsilon_mean']:.1f}$" if np.isfinite(r['A_H_epsilon_mean']) else "--",f"${r.get('latent_cos_pos_mean',float('nan')):.3f}$"]
        lines.append("        "+" & ".join(vals)+r" \\")
    lines += [r"        \bottomrule",r"    \end{tabular}",r"    \caption{Compact sensitivity study of the latent-space dimension. All models are trained for the same 100-epoch budget; all remaining hyperparameters are kept fixed.}",r"    \label{tab:latent_dimension_sensitivity}",r"\end{table}"]
    path.write_text("\n".join(lines)+"\n",encoding="utf-8")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description="cQED-Gen Section 4.3.3 coherent ablation study")
    p.add_argument("--reference",default="best_vae_global.pt")
    p.add_argument("--out-dir",default="analysis_results/4_3_3_ablation_study")
    p.add_argument("--plan",choices=["core","constraints","latent_dim","all"],default="core")
    p.add_argument("--force-retrain",action="store_true")
    p.add_argument("--epochs",type=int,default=100)
    p.add_argument("--batch-size",type=int,default=None)
    p.add_argument("--no-compile",action="store_true")
    p.add_argument("--no-amp",action="store_true")
    p.add_argument("--eval-batch-size",type=int,default=128)
    p.add_argument("--topology-pool-per-family",type=int,default=1000)
    p.add_argument("--latent-pool-per-family",type=int,default=250)
    p.add_argument("--metric-runs",type=int,default=10)
    p.add_argument("--metric-samples-per-run",type=int,default=500)
    p.add_argument("--prior-runs",type=int,default=10)
    p.add_argument("--prior-samples-per-run",type=int,default=1000)
    p.add_argument("--qultra-max-per-family",type=int,default=100)
    p.add_argument("--no-qultra",action="store_true")
    p.add_argument("--f-min",type=float,default=1.0)
    p.add_argument("--f-max",type=float,default=9.0)
    p.add_argument("--eps-frequency",type=float,default=1.0)
    p.add_argument("--eps-self-kerr",type=float,default=5.0)
    p.add_argument("--eps-cross-kerr",type=float,default=10.0)
    p.add_argument("--eps-kappa",type=float,default=10.0)
    p.add_argument("--chi-floor",type=float,default=0.5)
    p.add_argument("--kappa-floor",type=float,default=1e-6)
    p.add_argument("--seed",type=int,default=42)
    return p.parse_args()



def main() -> None:
    args=parse_args(); set_seed(args.seed)
    if int(args.epochs)!=100: print(f"[warning] Thesis protocol uses 100 epochs; requested {args.epochs}.")
    out_dir=(REPO_ROOT/args.out_dir).resolve(); out_dir.mkdir(parents=True,exist_ok=True)
    reference=(REPO_ROOT/args.reference).resolve()
    if not reference.exists(): raise FileNotFoundError(reference)
    ref_cfg=load_reference_config(reference); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Repository : {REPO_ROOT}\nDevice     : {device}\nConfig src : {reference}\nOutput     : {out_dir}\nTrain budget per variant: {args.epochs} epochs")
    _,_,test_data,_=load_all_datasets_vae(train_frac=ref_cfg.get("train_frac",0.70),val_frac=ref_cfg.get("val_frac",0.15),seed=ref_cfg.get("seed",42),max_nodes=ref_cfg.get("max_nodes",12))
    all_details={}; objective_rows=[]; constraint_rows=[]; latent_dim_rows=[]
    full_ckpt=out_dir/"checkpoints"/"full_100ep.pt"

    if args.plan in {"core","all"}:
        labels={"full_100ep":"Full (100 ep.)","no_KL":"w/o KL","no_alignment":"w/o alignment","no_InfoNCE":"w/o InfoNCE","no_CG":"w/o classifier guidance"}
        for name,overrides in CORE_ABLATIONS.items():
            ckpt=train_variant(name,overrides,ref_cfg,out_dir,args)
            res=evaluate_checkpoint(ckpt,labels[name],"objective_ablation","full",test_data,device,args)
            objective_rows.append(res.row); all_details[f"objective:{name}"]=res.details
        write_csv(objective_rows,out_dir/"ablation_objectives.csv")
        write_ablation_tex(objective_rows,out_dir/"ablation_objectives.tex","Ablation of the principal cQED-Gen training objectives under a common 100-epoch budget. The reported quantities reproduce the latent-space, unconditional-generation and seen-circuit inverse-design diagnostics used in the main analysis.","tab:ablation_losses")
        plot_metric_bars(objective_rows,out_dir/"figure41_ablation_losses.pdf","Ablation of cQED-Gen training objectives (100 epochs)")
        plot_latent_alignment(objective_rows,out_dir/"figure41b_latent_alignment_ablation.pdf")

    if args.plan in {"constraints","latent_dim"} and not full_ckpt.exists(): full_ckpt=train_variant("full_100ep",{},ref_cfg,out_dir,args)

    if args.plan in {"core","constraints","all"}:
        if not full_ckpt.exists(): full_ckpt=train_variant("full_100ep",{},ref_cfg,out_dir,args)
        modes=["full","no_node_mask","no_edge_constraints","no_degree_tracker","none"]
        labels={"full":"Full constraints","no_node_mask":"No node feasibility","no_edge_constraints":"No edge compatibility","no_degree_tracker":"No coupler-degree tracker","none":"No generation constraints"}
        for mode in modes:
            res=evaluate_checkpoint(full_ckpt,labels[mode],"constraint_ablation",mode,test_data,device,args)
            constraint_rows.append(res.row); all_details[f"constraints:{mode}"]=res.details
        write_csv(constraint_rows,out_dir/"physical_constraints_ablation.csv")
        write_ablation_tex(constraint_rows,out_dir/"physical_constraints_ablation.tex","Ablation of the decoder-side superconducting-circuit constraints using the same Full-100ep checkpoint. Constraint masks are removed only at generation time, while the independent post-generation validity and QuLTRA evaluations remain unchanged.","tab:physics_constraints_ablation")
        plot_metric_bars(constraint_rows,out_dir/"figure_physical_constraints_ablation.pdf","Ablation of physics-based generation constraints")

    if args.plan in {"latent_dim","all"}:
        if not full_ckpt.exists(): full_ckpt=train_variant("full_100ep",{},ref_cfg,out_dir,args)
        ref_nz=int(ref_cfg.get("nz",128))
        for nz in LATENT_DIM_VALUES:
            if nz==ref_nz: ckpt=full_ckpt; label=f"d_z={nz} (full)"
            else: ckpt=train_variant(f"latent_dim_{nz}",{"nz":int(nz)},ref_cfg,out_dir,args); label=f"d_z={nz}"
            res=evaluate_checkpoint(ckpt,label,"latent_dimension_sensitivity","full",test_data,device,args)
            row=dict(res.row); row["nz"]=int(nz); latent_dim_rows.append(row); all_details[f"latent_dim:{nz}"]=res.details
        write_csv(latent_dim_rows,out_dir/"latent_dimension_sensitivity.csv"); write_latent_dimension_tex(latent_dim_rows,out_dir/"latent_dimension_sensitivity.tex"); plot_latent_dimension_sensitivity(latent_dim_rows,out_dir/"figure_latent_dimension_sensitivity.pdf")

    json_dump({"reference_checkpoint_used_for_config_only":str(reference),"reference_config":ref_cfg,"analysis_config":vars(args),"protocol":{"training_epochs_per_variant":int(args.epochs),"same_split_for_all_variants":True,"full_baseline_retrained_for_fair_budget":True,"cem_excluded_from_training_ablation":True,"reason_cem_excluded":"CEM is an inference-time physics-guided refinement already analysed separately on unseen/stress-test targets."},"objective_ablation":objective_rows,"constraint_ablation":constraint_rows,"latent_dimension_sensitivity":latent_dim_rows,"details":all_details,"hamiltonian_success_definition":{"d_H":"max over active groups of (max relative error in group / group tolerance)","success":"d_H <= 1","eps_frequency_percent":args.eps_frequency,"eps_self_kerr_percent":args.eps_self_kerr,"eps_cross_kerr_percent":args.eps_cross_kerr,"eps_kappa_percent":args.eps_kappa,"chi_floor":args.chi_floor,"kappa_floor":args.kappa_floor}},out_dir/"ablation_study_summary.json")
    print("\nDone. Main outputs:")
    for name in ["figure41_ablation_losses.pdf","figure41b_latent_alignment_ablation.pdf","ablation_objectives.csv","ablation_objectives.tex","figure_physical_constraints_ablation.pdf","physical_constraints_ablation.csv","physical_constraints_ablation.tex","figure_latent_dimension_sensitivity.pdf","latent_dimension_sensitivity.csv","latent_dimension_sensitivity.tex","ablation_study_summary.json"]:
        p=out_dir/name
        if p.exists(): print(f"  {p}")



if __name__ == "__main__":
    main()
