#!/usr/bin/env python3
"""
analysis_4_2_3_latent_space.py
================================
Latent-space analysis for Sec. 4.2.3 of the cQED-Gen thesis.

Run this file from the BASE DIRECTORY of the repository, i.e. next to
``train_vae.py``, ``best_vae_global.pt`` and ``src/``.

The script is intentionally built around the current repository API and reuses
its data loader, GraphVAE encoders, decoder and physical topology constraints.
It performs four complementary analyses:

1. Latent-space visualization (Figure 40)
   - deterministic circuit posterior means mu_c from the training split;
   - balanced sampling across the eight training circuit families;
   - 2-D t-SNE projection;
   - faded z ~ N(0,I) prior samples included in the same t-SNE fit;
   - selected prior points are decoded and shown as primitive circuit examples.

2. Geometry in the ORIGINAL latent space (d_z = 128 in the final run)
   - d_intra: mean within-family distance to the family centroid;
   - d_inter: mean pairwise distance between family centroids;
   - R_sep = d_inter / d_intra;
   - values are computed directly in d_z dimensions, never in t-SNE space;
   - the same diagnostics are also produced for the Hamiltonian/spec branch.

3. Cross-modal alignment diagnostics
   - paired ||mu_c - mu_s||_2 and cosine similarity;
   - exact paired cross-modal retrieval Top-1 / Top-5 / Top-10 / MRR;
   - family-level Top-1 retrieval (useful because inverse design is one-to-many);
   - optional joint circuit/Hamiltonian t-SNE PDF.

4. Generative prior sampling
   - repeated z ~ N(0,I) sampling;
   - topology decoding with the repository decoder;
   - expansion to primitive circuits;
   - structural/physical validity using the same cQED compatibility rules used
     by the repository (connected graph, no self-loops, physical/coupler
     compatibility, valid coupler degrees);
   - novelty by primitive, type-preserving graph isomorphism against ALL
     topologies explicitly included in training;
   - mean +/- standard deviation over independent sampling runs.

Default outputs
---------------
analysis_results/4_2_3_latent_space/
    figure40_latent_tsne.pdf
    figure40b_cross_modal_tsne.pdf
    latent_geometry_summary.csv
    latent_family_geometry.csv
    latent_centroid_distances.csv
    cross_modal_retrieval_summary.csv
    latent_sampling_per_run.csv
    latent_sampling_summary.json
    figure40_tsne_coordinates.csv
    latent_analysis_summary.json

Example
-------
    python analysis_4_2_3_latent_space.py

A lighter diagnostic run:
    python analysis_4_2_3_latent_space.py \
        --tsne-per-family 150 \
        --geometry-per-family 300 \
        --retrieval-per-family 200 \
        --sampling-runs 2 \
        --samples-per-run 200

Notes
-----
- The final/global-scaler checkpoint is used by default.
- Posterior MEANS are used for latent geometry. This is deliberate: t-SNE and
  d_intra/d_inter should describe the learned representation, not one random
  reparameterization draw.
- t-SNE is ONLY a visualization. All quantitative distances are calculated in
  the original d_z-dimensional space.
- Novelty is topological: a generated VALID primitive graph is novel if it is
  not type-isomorphic to any primitive topology explicitly used in training.
- A_novel_all uses all decoded samples as denominator, matching the equation
  N_novel / N_generated in the thesis. The script additionally reports
  A_novel_given_valid, which conditions novelty on structural validity.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Repository discovery/imports
# ---------------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()


def find_repo_root() -> Path:
    candidates = [Path.cwd(), THIS_FILE.parent, *THIS_FILE.parents, *Path.cwd().parents]
    for c in candidates:
        if (c / "train_vae.py").exists() and (c / "src").exists():
            return c.resolve()
    raise RuntimeError(
        "Could not find repository root. Run this script from the cQED-Gen base directory "
        "(the directory containing train_vae.py and src/)."
    )


REPO_ROOT = find_repo_root()
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
os.chdir(REPO_ROOT)

# Repository imports. These are intentionally imported from the actual project
# rather than duplicated here.
from data_loader.loader_vae import load_all_datasets_vae  # noqa: E402
from data_loader.schema import train_datasets  # noqa: E402
from circuit2graph import CQEDTopology, SubgType, SUBG_DEFS, expand_topology  # noqa: E402
from circuit2graph.constraints import exists_valid_macro_graph, is_compatible  # noqa: E402
from vae_model.vae import GraphVAE  # noqa: E402


FAMILY_NAMES = {
    "Qubit": "Qubit",
    "Resonator": "Resonator",
    "Qubit_resonator_feedline_capacitive": "Q-R-F (capacitive)",
    "Qubit_resonator_feedline_inductive_nognd": "Q-R-F (inductive)",
    "Qubit_resonator_resonator": "Q-R-R",
    "Resonator_qubit_resonator": "R-Q-R",
    "Two_qubit_with_capacitive_coupling": "Two-qubit capacitive",
    "Three_qubit_capacitive_star": "Three-qubit star",
}

SHORT_NODE_LABEL = {
    SubgType.TRANSMON: "T",
    SubgType.RESONATOR: "R",
    SubgType.C_COUPLER: "C",
    SubgType.I_COUPLER: "I",
    SubgType.FEEDLINE: "F",
}

PHYSICAL_TYPES = {SubgType.TRANSMON, SubgType.RESONATOR, SubgType.FEEDLINE}
COUPLER_TYPES = {SubgType.C_COUPLER, SubgType.I_COUPLER}


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def json_float(x: Any) -> Any:
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def safe_mean(x: np.ndarray) -> float:
    return float(np.mean(x)) if x.size else float("nan")


def safe_std(x: np.ndarray, ddof: int = 1) -> float:
    if x.size <= ddof:
        return 0.0
    return float(np.std(x, ddof=ddof))


def pretty_family(name: str) -> str:
    return FAMILY_NAMES.get(name, name.replace("_", " "))


# ---------------------------------------------------------------------------
# Checkpoint/model loading
# ---------------------------------------------------------------------------


def build_model_from_config(config: dict[str, Any]) -> GraphVAE:
    sig = inspect.signature(GraphVAE.__init__)
    accepted = set(sig.parameters.keys()) - {"self"}
    candidate_kwargs = {
        "nz": config.get("nz", 128),
        "hv_dim": config.get("hv_dim", 32),
        "inner_hidden": config.get("inner_hidden", 32),
        "inner_layers": config.get("inner_layers", 2),
        "outer_hidden": config.get("outer_hidden", 64),
        "outer_layers": config.get("outer_layers", 3),
        "d": config.get("d", 64),
        "nhead": config.get("nhead", 4),
        "tf_layers": config.get("tf_layers", 4),
        "max_nodes": config.get("max_nodes", 12),
        "hs": config.get("hs", 512),
        "ggnn_rounds": config.get("ggnn_rounds", 3),
        "max_nodes_dec": config.get("max_nodes_dec", 8),
        "param_hidden": config.get("param_hidden", 128),
        "param_layers": config.get("param_layers", 3),
        "spec_d": config.get("spec_d", 128),
        "spec_layers": config.get("spec_layers", 4),
        "cg_hidden": config.get("cg_hidden", 64),
        "dropout": config.get("dropout", 0.1),
        "beta_start": config.get("beta_start", 0.0),
        "beta_max": config.get("beta_max", 1e-4),
        "warmup_steps": config.get("warmup_steps", 10_000),
        "attrs_scale": config.get("attrs_scale", 0.01),
        "align_scale": config.get("align_scale", 0.5),
        "nce_scale": config.get("nce_scale", 0.5),
        "cg_scale": config.get("cg_scale", 0.5),
        "spec_recon_scale": config.get("spec_recon_scale", 0.3),
        "tau": config.get("tau", 0.1),
        "class_weight_end": config.get("class_weight_end", 1.0),
        "lambda_phys": config.get("lambda_phys", 0.0),
    }
    kwargs = {k: v for k, v in candidate_kwargs.items() if k in accepted}
    return GraphVAE(**kwargs)


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[GraphVAE, dict[str, Any], dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = dict(ckpt.get("config", {}))
    model = build_model_from_config(config).to(device)
    model.load_state_dict_full({
        "model_state": ckpt["model_state"],
        "step": ckpt.get("step", 0),
    })
    model.eval()
    return model, config, ckpt


# ---------------------------------------------------------------------------
# Split selection and balanced subsampling
# ---------------------------------------------------------------------------


def group_by_family(samples: Iterable[Any]) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for sample in samples:
        grouped[str(sample.dataset_name)].append(sample)
    return dict(grouped)


def balanced_subset(
    samples: list[Any],
    n_per_family: int | None,
    seed: int,
) -> list[Any]:
    grouped = group_by_family(samples)
    rng = np.random.default_rng(seed)
    out: list[Any] = []
    for family in sorted(grouped):
        items = grouped[family]
        if n_per_family is None or n_per_family <= 0 or len(items) <= n_per_family:
            chosen = list(items)
        else:
            idx = rng.choice(len(items), size=n_per_family, replace=False)
            chosen = [items[int(i)] for i in idx]
        out.extend(chosen)
    rng.shuffle(out)
    return out


# ---------------------------------------------------------------------------
# Latent extraction
# ---------------------------------------------------------------------------


@dataclass
class LatentCollection:
    mu_c: np.ndarray
    logvar_c: np.ndarray
    mu_s: np.ndarray
    logvar_s: np.ndarray
    labels: np.ndarray


@torch.no_grad()
def extract_latents(
    model: GraphVAE,
    samples: list[Any],
    scalers: dict[str, Any],
    device: torch.device,
    batch_size: int,
    verbose_prefix: str = "",
) -> LatentCollection:
    """Extract deterministic posterior parameters for both latent branches."""
    model.eval()
    mu_c_all: list[np.ndarray] = []
    lv_c_all: list[np.ndarray] = []
    mu_s_all: list[np.ndarray] = []
    lv_s_all: list[np.ndarray] = []
    labels: list[str] = []

    total = len(samples)
    for start in range(0, total, batch_size):
        batch = samples[start:start + batch_size]

        # Circuit branch: model.encode() returns z=mu in eval mode, plus mu/logvar.
        _z_c, mu_c, logvar_c = model.encode(batch, scalers)

        obs_vals = torch.stack([s.obs_vals for s in batch], dim=0).to(device)
        obs_mask = torch.stack([s.obs_mask for s in batch], dim=0).to(device)
        _z_s, mu_s, logvar_s = model.spec_encoder.encode(obs_vals, obs_mask)

        mu_c_all.append(mu_c.detach().cpu().numpy())
        lv_c_all.append(logvar_c.detach().cpu().numpy())
        mu_s_all.append(mu_s.detach().cpu().numpy())
        lv_s_all.append(logvar_s.detach().cpu().numpy())
        labels.extend(str(s.dataset_name) for s in batch)

        if total >= 2000 and (start == 0 or (start // batch_size + 1) % 20 == 0):
            print(f"  {verbose_prefix}{min(start + len(batch), total)}/{total} encoded")

    return LatentCollection(
        mu_c=np.concatenate(mu_c_all, axis=0),
        logvar_c=np.concatenate(lv_c_all, axis=0),
        mu_s=np.concatenate(mu_s_all, axis=0),
        logvar_s=np.concatenate(lv_s_all, axis=0),
        labels=np.asarray(labels, dtype=object),
    )


# ---------------------------------------------------------------------------
# Original-dimensional latent geometry
# ---------------------------------------------------------------------------


def geometry_for_branch(Z: np.ndarray, labels: np.ndarray, branch: str, split_name: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    families = sorted(set(labels.tolist()))
    centroids: dict[str, np.ndarray] = {}
    intra_by_family: dict[str, float] = {}
    family_rows: list[dict[str, Any]] = []

    for family in families:
        mask = labels == family
        Zk = Z[mask]
        centroid = Zk.mean(axis=0)
        centroids[family] = centroid
        distances = np.linalg.norm(Zk - centroid[None, :], axis=1)
        d_k = float(distances.mean())
        intra_by_family[family] = d_k
        family_rows.append({
            "split": split_name,
            "branch": branch,
            "family": family,
            "family_pretty": pretty_family(family),
            "n_samples": int(mask.sum()),
            "d_intra_family": d_k,
            "distance_to_centroid_std": safe_std(distances),
            "centroid_norm": float(np.linalg.norm(centroid)),
        })

    d_intra = float(np.mean(list(intra_by_family.values())))

    pair_distances: list[float] = []
    pair_rows: list[dict[str, Any]] = []
    for i, a in enumerate(families):
        for b in families[i + 1:]:
            d = float(np.linalg.norm(centroids[a] - centroids[b]))
            pair_distances.append(d)
            pair_rows.append({
                "split": split_name,
                "branch": branch,
                "family_a": a,
                "family_a_pretty": pretty_family(a),
                "family_b": b,
                "family_b_pretty": pretty_family(b),
                "centroid_distance": d,
            })

    d_inter = float(np.mean(pair_distances)) if pair_distances else float("nan")
    r_sep = d_inter / d_intra if d_intra > 0 else float("nan")

    summary = {
        "split": split_name,
        "branch": branch,
        "n_samples": int(Z.shape[0]),
        "n_families": len(families),
        "latent_dim": int(Z.shape[1]),
        "d_intra": d_intra,
        "d_inter": d_inter,
        "R_sep": r_sep,
        "mean_latent_norm": float(np.linalg.norm(Z, axis=1).mean()),
        "std_latent_norm": float(np.linalg.norm(Z, axis=1).std(ddof=1)),
    }
    return summary, family_rows, pair_rows


def cross_modal_centroid_rows(latents: LatentCollection, split_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in sorted(set(latents.labels.tolist())):
        mask = latents.labels == family
        c = latents.mu_c[mask].mean(axis=0)
        s = latents.mu_s[mask].mean(axis=0)
        rows.append({
            "split": split_name,
            "branch": "cross_modal",
            "family": family,
            "family_pretty": pretty_family(family),
            "n_samples": int(mask.sum()),
            "centroid_c_to_s_l2": float(np.linalg.norm(c - s)),
            "centroid_c_to_s_cosine": float(np.dot(c, s) / (np.linalg.norm(c) * np.linalg.norm(s) + 1e-12)),
        })
    return rows


# ---------------------------------------------------------------------------
# Cross-modal retrieval
# ---------------------------------------------------------------------------


def retrieval_direction(
    query: np.ndarray,
    database: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
) -> dict[str, float]:
    """Strict paired retrieval plus a family-level retrieval metric.

    Query i and database i are the exact positive pair. Exact paired retrieval is
    deliberately strict and may underestimate quality in a one-to-many inverse
    problem. Family Top-1 is therefore reported alongside it.
    """
    Q = query / np.clip(np.linalg.norm(query, axis=1, keepdims=True), 1e-12, None)
    D = database / np.clip(np.linalg.norm(database, axis=1, keepdims=True), 1e-12, None)
    N = Q.shape[0]

    ranks: list[int] = []
    family_hits = 0
    paired_cosines: list[float] = []

    D_t = torch.from_numpy(D.astype(np.float32))
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        q = torch.from_numpy(Q[start:end].astype(np.float32))
        sim = q @ D_t.T
        sim_np = sim.numpy()

        local = np.arange(end - start)
        pos_indices = np.arange(start, end)
        pos = sim_np[local, pos_indices]
        paired_cosines.extend(pos.tolist())

        # rank = 1 + number of candidates with strictly larger cosine similarity.
        rank = 1 + np.sum(sim_np > pos[:, None], axis=1)
        ranks.extend(rank.astype(int).tolist())

        top1 = np.argmax(sim_np, axis=1)
        family_hits += int(np.sum(labels[top1] == labels[start:end]))

    r = np.asarray(ranks, dtype=np.int64)
    pc = np.asarray(paired_cosines, dtype=np.float64)
    return {
        "n_queries": float(N),
        "paired_top1": float(np.mean(r <= 1)),
        "paired_top5": float(np.mean(r <= 5)),
        "paired_top10": float(np.mean(r <= 10)),
        "paired_mrr": float(np.mean(1.0 / r)),
        "paired_median_rank": float(np.median(r)),
        "family_top1": float(family_hits / max(N, 1)),
        "paired_cosine_mean": float(pc.mean()),
        "paired_cosine_median": float(np.median(pc)),
    }


def compute_cross_modal_metrics(latents: LatentCollection, batch_size: int, split_name: str) -> list[dict[str, Any]]:
    l2 = np.linalg.norm(latents.mu_c - latents.mu_s, axis=1)
    c_n = latents.mu_c / np.clip(np.linalg.norm(latents.mu_c, axis=1, keepdims=True), 1e-12, None)
    s_n = latents.mu_s / np.clip(np.linalg.norm(latents.mu_s, axis=1, keepdims=True), 1e-12, None)
    cos = np.sum(c_n * s_n, axis=1)

    common = {
        "split": split_name,
        "paired_l2_mean": float(l2.mean()),
        "paired_l2_median": float(np.median(l2)),
        "paired_l2_p90": float(np.percentile(l2, 90)),
        "paired_cosine_mean_direct": float(cos.mean()),
        "paired_cosine_median_direct": float(np.median(cos)),
    }

    s_to_c = retrieval_direction(latents.mu_s, latents.mu_c, latents.labels, batch_size)
    c_to_s = retrieval_direction(latents.mu_c, latents.mu_s, latents.labels, batch_size)

    return [
        {"direction": "H_to_G_latent (spec->circuit)", **common, **s_to_c},
        {"direction": "G_to_H_latent (circuit->spec)", **common, **c_to_s},
    ]


# ---------------------------------------------------------------------------
# Topology helpers for prior sampling and decoded examples
# ---------------------------------------------------------------------------


def graph_ns_to_macro_topology(g: SimpleNamespace, name: str) -> CQEDTopology:
    """Convert decoder topology output to CQEDTopology using placeholder attrs.

    Physical attribute values are irrelevant for the topology-only validity and
    novelty analysis. Direction, however, matters for macro-node expansion and is
    therefore copied from the topological decoder.
    """
    topo = CQEDTopology(name)
    nodes = []
    dirs = list(getattr(g, "direction", []))
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
    if n_nodes == 0:
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
        reasons.append("empty_macro_graph")
        return False, reasons

    try:
        if not exists_valid_macro_graph(node_types, directions):
            reasons.append("no_valid_macro_completion")
    except Exception as exc:
        reasons.append(f"macro_completion_error:{type(exc).__name__}")

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
    for node in topo._nodes:
        G.add_node(int(node.node_id), subg_type=int(node.subg_type))
    for u, v in topo._edges:
        G.add_edge(int(u), int(v))
    return G


def primitive_structural_validity(topo: CQEDTopology) -> tuple[bool, list[str]]:
    G = primitive_graph(topo)
    reasons: list[str] = []
    if G.number_of_nodes() == 0:
        reasons.append("empty_primitive_graph")
        return False, reasons
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

    for node in topo._nodes:
        st = node.subg_type
        deg = G.degree[int(node.node_id)]
        if st in COUPLER_TYPES and deg != 2:
            reasons.append("coupler_degree_not_two")
        if st == SubgType.C_COUPLER:
            nbr_types = [by_id[int(n)].subg_type for n in G.neighbors(int(node.node_id))]
            if len(nbr_types) != 2 or any(t not in PHYSICAL_TYPES for t in nbr_types):
                reasons.append("capacitive_coupler_invalid_neighbors")
        if st == SubgType.I_COUPLER:
            nbr_types = [by_id[int(n)].subg_type for n in G.neighbors(int(node.node_id))]
            if len(nbr_types) != 2 or set(nbr_types) != {SubgType.RESONATOR, SubgType.FEEDLINE}:
                reasons.append("inductive_coupler_invalid_neighbors")
        if st == SubgType.FEEDLINE and deg > 2:
            reasons.append("feedline_degree_gt_two")

    return len(reasons) == 0, sorted(set(reasons))


def analyze_decoded_topology(g: SimpleNamespace, name: str) -> tuple[dict[str, Any], CQEDTopology | None]:
    macro_ok, macro_reasons = macro_structural_validity(g)
    row: dict[str, Any] = {
        "macro_valid": macro_ok,
        "macro_invalid_reasons": ";".join(macro_reasons),
        "n_macro_nodes": len(getattr(g, "node_types", [])),
        "n_macro_edges": len(getattr(g, "edges", [])),
        "primitive_valid": False,
        "primitive_invalid_reasons": "",
        "valid": False,
        "expansion_error": "",
    }
    try:
        macro = graph_ns_to_macro_topology(g, name=f"{name}_macro")
        primitive = expand_topology(macro, validate=False)
        prim_ok, prim_reasons = primitive_structural_validity(primitive)
        row["primitive_valid"] = prim_ok
        row["primitive_invalid_reasons"] = ";".join(prim_reasons)
        row["valid"] = bool(macro_ok and prim_ok)
        row["n_primitive_nodes"] = len(primitive._nodes)
        row["n_primitive_edges"] = len(primitive._edges)
        return row, primitive
    except Exception as exc:
        row["expansion_error"] = f"{type(exc).__name__}: {exc}"
        return row, None


def topology_signature(G: nx.Graph) -> tuple[int, int, tuple[int, ...]]:
    types = tuple(sorted(int(G.nodes[n]["subg_type"]) for n in G.nodes))
    return G.number_of_nodes(), G.number_of_edges(), types


def make_training_topology_index() -> dict[tuple[int, int, tuple[int, ...]], list[tuple[str, nx.Graph]]]:
    index: dict[tuple[int, int, tuple[int, ...]], list[tuple[str, nx.Graph]]] = defaultdict(list)
    for name, defn in train_datasets().items():
        topo = defn.topology_fn()
        G = primitive_graph(topo)
        index[topology_signature(G)].append((name, G))
    return dict(index)


def match_training_topology(
    primitive: CQEDTopology,
    index: dict[tuple[int, int, tuple[int, ...]], list[tuple[str, nx.Graph]]],
) -> str | None:
    G = primitive_graph(primitive)
    candidates = index.get(topology_signature(G), [])
    nm = nx.algorithms.isomorphism.categorical_node_match("subg_type", -1)
    for family, target in candidates:
        if nx.is_isomorphic(G, target, node_match=nm):
            return family
    return None


# ---------------------------------------------------------------------------
# Prior-sampling statistics
# ---------------------------------------------------------------------------


@torch.no_grad()
def decode_topologies(model: GraphVAE, z: torch.Tensor, stochastic: bool) -> list[SimpleNamespace]:
    # Use the topology decoder directly. Parameter regression is unnecessary for
    # the structural validity/novelty metrics and would make the analysis slower.
    return model.decoder.decode(z, stochastic=stochastic)


def run_prior_sampling(
    model: GraphVAE,
    device: torch.device,
    n_runs: int,
    samples_per_run: int,
    batch_size: int,
    seed: int,
    stochastic: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_index = make_training_topology_index()
    run_rows: list[dict[str, Any]] = []
    overall_reason_counts = Counter()

    for run in range(n_runs):
        run_seed = seed + 10_000 + run
        torch.manual_seed(run_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(run_seed)

        n_valid = 0
        n_novel = 0
        n_seen = 0
        n_total = 0
        reason_counts = Counter()
        novel_family_size_counts = Counter()

        for start in range(0, samples_per_run, batch_size):
            current = min(batch_size, samples_per_run - start)
            z = torch.randn(current, model.nz, device=device)
            decoded = decode_topologies(model, z, stochastic=stochastic)
            for j, g in enumerate(decoded):
                row, primitive = analyze_decoded_topology(g, name=f"run{run}_sample{start+j}")
                n_total += 1
                if row["valid"] and primitive is not None:
                    n_valid += 1
                    matched = match_training_topology(primitive, train_index)
                    if matched is None:
                        n_novel += 1
                        novel_family_size_counts[len(primitive._nodes)] += 1
                    else:
                        n_seen += 1
                else:
                    reasons = [
                        *str(row.get("macro_invalid_reasons", "")).split(";"),
                        *str(row.get("primitive_invalid_reasons", "")).split(";"),
                    ]
                    if row.get("expansion_error"):
                        reasons.append("expansion_error")
                    for reason in reasons:
                        if reason:
                            reason_counts[reason] += 1
                            overall_reason_counts[reason] += 1

        A_valid = n_valid / max(n_total, 1)
        A_novel_all = n_novel / max(n_total, 1)
        A_novel_valid = n_novel / max(n_valid, 1)
        run_rows.append({
            "run": run + 1,
            "seed": run_seed,
            "n_generated": n_total,
            "n_valid": n_valid,
            "n_novel": n_novel,
            "n_seen_training_topology": n_seen,
            "A_valid": A_valid,
            "A_novel_all": A_novel_all,
            "A_novel_given_valid": A_novel_valid,
            "invalid_reason_counts": json.dumps(dict(reason_counts), sort_keys=True),
            "novel_primitive_node_count_histogram": json.dumps(dict(novel_family_size_counts), sort_keys=True),
        })
        print(
            f"  prior run {run+1}/{n_runs}: valid={A_valid:.4f}, "
            f"novel/all={A_novel_all:.4f}, novel|valid={A_novel_valid:.4f}"
        )

    valid_arr = np.asarray([r["A_valid"] for r in run_rows], dtype=float)
    novel_all_arr = np.asarray([r["A_novel_all"] for r in run_rows], dtype=float)
    novel_valid_arr = np.asarray([r["A_novel_given_valid"] for r in run_rows], dtype=float)

    summary = {
        "n_runs": n_runs,
        "samples_per_run": samples_per_run,
        "n_generated_total": int(sum(int(r["n_generated"]) for r in run_rows)),
        "n_valid_total": int(sum(int(r["n_valid"]) for r in run_rows)),
        "n_novel_total": int(sum(int(r["n_novel"]) for r in run_rows)),
        "A_valid_mean": safe_mean(valid_arr),
        "A_valid_std": safe_std(valid_arr),
        "A_novel_all_mean": safe_mean(novel_all_arr),
        "A_novel_all_std": safe_std(novel_all_arr),
        "A_novel_given_valid_mean": safe_mean(novel_valid_arr),
        "A_novel_given_valid_std": safe_std(novel_valid_arr),
        "stochastic_decoder": bool(stochastic),
        "validity_definition": (
            "decoded macro graph admits a connected physically compatible macro graph; actual macro graph is connected; "
            "all sampled macro edges are compatible; expanded primitive graph is non-empty, connected, self-loop free; "
            "physical nodes are connected only through couplers; couplers have degree two; inductive couplers join R-F"
        ),
        "novelty_definition": (
            "valid generated primitive topology is not type-preserving graph-isomorphic to any primitive topology "
            "explicitly included in training"
        ),
        "A_novel_all_denominator": "all decoded samples (N_generated), matching thesis equation",
        "invalid_reason_counts_total": dict(overall_reason_counts),
    }
    return run_rows, summary


# ---------------------------------------------------------------------------
# Figure 40: t-SNE + decoded prior examples
# ---------------------------------------------------------------------------


def run_tsne(X: np.ndarray, perplexity: float, seed: int) -> np.ndarray:
    from sklearn.manifold import TSNE

    n = X.shape[0]
    effective = min(float(perplexity), max(5.0, (n - 1) / 3.0))
    tsne = TSNE(
        n_components=2,
        perplexity=effective,
        init="pca",
        learning_rate="auto",
        metric="euclidean",
        random_state=seed,
    )
    return tsne.fit_transform(X)


def choose_spread_indices(coords: np.ndarray, candidates: list[int], n_select: int) -> list[int]:
    if not candidates:
        return []
    if len(candidates) <= n_select:
        return list(candidates)

    C = coords[np.asarray(candidates)]
    center = C.mean(axis=0)
    first_local = int(np.argmax(np.linalg.norm(C - center[None, :], axis=1)))
    selected_local = [first_local]
    while len(selected_local) < n_select:
        sel_coords = C[np.asarray(selected_local)]
        min_dist = np.min(
            np.linalg.norm(C[:, None, :] - sel_coords[None, :, :], axis=2),
            axis=1,
        )
        min_dist[np.asarray(selected_local)] = -1.0
        selected_local.append(int(np.argmax(min_dist)))
    return [candidates[i] for i in selected_local]


def draw_primitive_graph(ax: plt.Axes, topo: CQEDTopology | None, title: str) -> None:
    if topo is None or len(topo._nodes) == 0:
        ax.text(0.5, 0.5, "invalid", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
        return
    G = nx.Graph()
    labels: dict[int, str] = {}
    colors: list[Any] = []
    for node in topo._nodes:
        nid = int(node.node_id)
        G.add_node(nid, subg_type=node.subg_type)
        labels[nid] = SHORT_NODE_LABEL.get(node.subg_type, node.subg_type.name)
    G.add_edges_from((int(u), int(v)) for u, v in topo._edges)

    if G.number_of_nodes() <= 2:
        pos = nx.shell_layout(G)
    else:
        pos = nx.spring_layout(G, seed=7, k=1.0)
    for n in G.nodes:
        st = G.nodes[n]["subg_type"]
        colors.append(SUBG_DEFS[st].color)
    nx.draw_networkx_edges(G, pos, ax=ax, width=1.3, alpha=0.75)
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=colors, node_size=580,
        edgecolors="black", linewidths=0.6,
    )
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=7, font_weight="bold")
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def make_figure40(
    model: GraphVAE,
    latents: LatentCollection,
    device: torch.device,
    out_pdf: Path,
    out_coords_csv: Path,
    prior_points: int,
    n_examples: int,
    perplexity: float,
    seed: int,
) -> dict[str, Any]:
    set_seed(seed)
    n_data = latents.mu_c.shape[0]

    z_prior = torch.randn(prior_points, model.nz, device=device)
    z_prior_np = z_prior.detach().cpu().numpy()
    X = np.concatenate([latents.mu_c, z_prior_np], axis=0)
    coords = run_tsne(X, perplexity=perplexity, seed=seed)
    coords_data = coords[:n_data]
    coords_prior = coords[n_data:]

    # Decode prior points deterministically for a clean latent-point -> topology map.
    prior_decoded: list[SimpleNamespace] = []
    with torch.no_grad():
        for start in range(0, prior_points, 128):
            prior_decoded.extend(decode_topologies(model, z_prior[start:start + 128], stochastic=False))

    train_index = make_training_topology_index()
    prior_analysis: list[dict[str, Any]] = []
    prior_topos: list[CQEDTopology | None] = []
    valid_candidate_indices: list[int] = []
    for i, g in enumerate(prior_decoded):
        row, primitive = analyze_decoded_topology(g, name=f"figure40_prior_{i}")
        matched = match_training_topology(primitive, train_index) if (row["valid"] and primitive is not None) else None
        row["matched_training_family"] = matched or ""
        row["novel"] = bool(row["valid"] and matched is None)
        prior_analysis.append(row)
        prior_topos.append(primitive)
        if row["valid"]:
            valid_candidate_indices.append(i)

    # Prefer novel candidates, then fill from any valid candidate.
    novel_candidates = [i for i in valid_candidate_indices if prior_analysis[i]["novel"]]
    selected = choose_spread_indices(coords_prior, novel_candidates, min(n_examples, len(novel_candidates)))
    if len(selected) < n_examples:
        remaining = [i for i in valid_candidate_indices if i not in selected]
        extra = choose_spread_indices(coords_prior, remaining, n_examples - len(selected))
        selected.extend(extra)

    families = sorted(set(latents.labels.tolist()))
    cmap = plt.get_cmap("tab10")
    colors = {f: cmap(i % cmap.N) for i, f in enumerate(families)}

    fig = plt.figure(figsize=(10.8, 9.0))
    gs = fig.add_gridspec(2, max(n_examples, 1), height_ratios=[3.7, 1.35], hspace=0.22, wspace=0.18)
    ax = fig.add_subplot(gs[0, :])

    for family in families:
        mask = latents.labels == family
        ax.scatter(
            coords_data[mask, 0], coords_data[mask, 1],
            s=10, alpha=0.60, linewidths=0,
            color=colors[family], label=pretty_family(family),
        )
    ax.scatter(
        coords_prior[:, 0], coords_prior[:, 1],
        s=8, alpha=0.12, linewidths=0,
        color="black", label=r"Prior samples $z\sim\mathcal{N}(0,I)$",
    )

    for number, idx in enumerate(selected, start=1):
        xy = coords_prior[idx]
        ax.scatter([xy[0]], [xy[1]], marker="*", s=150, color="black", edgecolors="white", linewidths=0.7, zorder=5)
        ax.annotate(str(number), xy=(xy[0], xy[1]), xytext=(5, 5), textcoords="offset points", fontsize=9, fontweight="bold")

    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.set_title("cQED-Gen latent space: circuit posterior means and prior samples")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False, fontsize=8)
    ax.grid(False)

    for col in range(max(n_examples, 1)):
        ax_g = fig.add_subplot(gs[1, col])
        if col < len(selected):
            idx = selected[col]
            status = "novel" if prior_analysis[idx]["novel"] else "seen topology"
            draw_primitive_graph(ax_g, prior_topos[idx], f"{col+1}. decoded prior sample\n{status}")
        else:
            ax_g.axis("off")

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    coord_rows: list[dict[str, Any]] = []
    for i in range(n_data):
        coord_rows.append({
            "source": "encoded_circuit_mu",
            "family": str(latents.labels[i]),
            "family_pretty": pretty_family(str(latents.labels[i])),
            "tsne_x": float(coords_data[i, 0]),
            "tsne_y": float(coords_data[i, 1]),
            "selected_example": False,
            "valid": True,
            "novel": False,
        })
    for i in range(prior_points):
        coord_rows.append({
            "source": "prior",
            "family": "",
            "family_pretty": "",
            "tsne_x": float(coords_prior[i, 0]),
            "tsne_y": float(coords_prior[i, 1]),
            "selected_example": i in selected,
            "valid": bool(prior_analysis[i]["valid"]),
            "novel": bool(prior_analysis[i]["novel"]),
            "matched_training_family": prior_analysis[i]["matched_training_family"],
        })
    write_csv(coord_rows, out_coords_csv)

    return {
        "n_encoded_points": int(n_data),
        "n_prior_points": int(prior_points),
        "n_selected_examples": int(len(selected)),
        "selected_prior_indices": selected,
        "selected_novel_count": int(sum(bool(prior_analysis[i]["novel"]) for i in selected)),
        "prior_valid_fraction_in_figure": float(np.mean([bool(r["valid"]) for r in prior_analysis])) if prior_analysis else float("nan"),
    }


def make_cross_modal_tsne(latents: LatentCollection, out_pdf: Path, perplexity: float, seed: int) -> None:
    n = latents.mu_c.shape[0]
    X = np.concatenate([latents.mu_c, latents.mu_s], axis=0)
    coords = run_tsne(X, perplexity=perplexity, seed=seed)
    cxy, sxy = coords[:n], coords[n:]

    families = sorted(set(latents.labels.tolist()))
    cmap = plt.get_cmap("tab10")
    colors = {f: cmap(i % cmap.N) for i, f in enumerate(families)}

    fig, ax = plt.subplots(figsize=(9.5, 7.0))
    for family in families:
        mask = latents.labels == family
        ax.scatter(cxy[mask, 0], cxy[mask, 1], s=10, alpha=0.40, linewidths=0, color=colors[family])
        ax.scatter(sxy[mask, 0], sxy[mask, 1], s=12, alpha=0.60, marker="x", linewidths=0.8, color=colors[family])

    # Legend: family colors + modality markers.
    family_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=5, color=colors[f], label=pretty_family(f))
        for f in families
    ]
    modality_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=5, color="black", label="Circuit encoder"),
        plt.Line2D([0], [0], marker="x", linestyle="", markersize=6, color="black", label="Hamiltonian encoder"),
    ]
    leg1 = ax.legend(handles=family_handles, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4, frameon=False, fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=modality_handles, loc="upper right", frameon=False, fontsize=8)
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.set_title("Cross-modal organization of the shared latent space")
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI/main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=Path("best_vae_global.pt"))
    p.add_argument("--out-dir", type=Path, default=Path("analysis_results/4_2_3_latent_space"))
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--batch-size", type=int, default=128, help="Encoder and decoder batch size")
    p.add_argument("--seed", type=int, default=42)

    # Figure 40 / geometry.
    p.add_argument("--tsne-per-family", type=int, default=500, help="Training samples per family in Figure 40")
    p.add_argument("--tsne-prior-points", type=int, default=300, help="Faded prior samples included in Figure 40")
    p.add_argument("--tsne-perplexity", type=float, default=40.0)
    p.add_argument("--tsne-examples", type=int, default=4, help="Decoded prior examples shown under Figure 40")
    p.add_argument("--geometry-per-family", type=int, default=1500, help="Training samples per family for d_intra/d_inter")

    # Cross-modal retrieval uses held-out test data by default.
    p.add_argument("--retrieval-per-family", type=int, default=500, help="Test samples per family for cross-modal retrieval")
    p.add_argument("--retrieval-batch-size", type=int, default=256)
    p.add_argument("--skip-cross-modal-tsne", action="store_true")

    # Prior sampling.
    p.add_argument("--sampling-runs", type=int, default=10)
    p.add_argument("--samples-per-run", type=int, default=1000)
    p.add_argument("--deterministic-sampling", action="store_true", help="Use argmax decoder; default is stochastic generation")
    p.add_argument("--skip-prior-sampling", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    checkpoint_path = (REPO_ROOT / args.checkpoint).resolve() if not args.checkpoint.is_absolute() else args.checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print("=" * 78)
    print("4.2.3 LATENT SPACE ANALYSIS")
    print("=" * 78)
    print(f"repo       : {REPO_ROOT}")
    print(f"checkpoint : {checkpoint_path}")
    print(f"device     : {device}")

    model, config, ckpt = load_model(checkpoint_path, device)
    print(f"latent dim : {model.nz}")
    print(f"checkpoint epoch: {ckpt.get('epoch', '?')}")

    # Exact training loader: same GLOBAL scaler, reservoir sampling and splits.
    print("\nReconstructing the final train/validation/test partitions...")
    train_list, val_list, test_list, scalers = load_all_datasets_vae(
        train_frac=float(config.get("train_frac", 0.70)),
        val_frac=float(config.get("val_frac", 0.15)),
        seed=int(config.get("seed", args.seed)),
        max_nodes=int(config.get("max_nodes", 12)),
    )
    print(f"split sizes: train={len(train_list)}, val={len(val_list)}, test={len(test_list)}")

    # ------------------------------------------------------------------
    # Figure 40
    # ------------------------------------------------------------------
    print("\n[1/4] Figure 40: balanced training latent t-SNE...")
    tsne_samples = balanced_subset(train_list, args.tsne_per_family, seed=args.seed + 1)
    tsne_latents = extract_latents(
        model, tsne_samples, scalers, device, args.batch_size, verbose_prefix="t-SNE: "
    )
    fig40_summary = make_figure40(
        model=model,
        latents=tsne_latents,
        device=device,
        out_pdf=args.out_dir / "figure40_latent_tsne.pdf",
        out_coords_csv=args.out_dir / "figure40_tsne_coordinates.csv",
        prior_points=args.tsne_prior_points,
        n_examples=args.tsne_examples,
        perplexity=args.tsne_perplexity,
        seed=args.seed,
    )

    # ------------------------------------------------------------------
    # Original-dimensional geometry
    # ------------------------------------------------------------------
    print("\n[2/4] Original-dimensional geometry (d_intra, d_inter, R_sep)...")
    geometry_samples = balanced_subset(train_list, args.geometry_per_family, seed=args.seed + 2)
    geometry_latents = extract_latents(
        model, geometry_samples, scalers, device, args.batch_size, verbose_prefix="geometry: "
    )

    geom_c, fam_c, pair_c = geometry_for_branch(geometry_latents.mu_c, geometry_latents.labels, "circuit_mu_c", "train")
    geom_s, fam_s, pair_s = geometry_for_branch(geometry_latents.mu_s, geometry_latents.labels, "hamiltonian_mu_s", "train")
    cross_fam = cross_modal_centroid_rows(geometry_latents, "train")

    write_csv([geom_c, geom_s], args.out_dir / "latent_geometry_summary.csv")
    write_csv(fam_c + fam_s + cross_fam, args.out_dir / "latent_family_geometry.csv")
    write_csv(pair_c + pair_s, args.out_dir / "latent_centroid_distances.csv")

    print(
        f"  circuit branch: d_intra={geom_c['d_intra']:.6f}, "
        f"d_inter={geom_c['d_inter']:.6f}, R_sep={geom_c['R_sep']:.6f}"
    )
    print(
        f"  spec branch   : d_intra={geom_s['d_intra']:.6f}, "
        f"d_inter={geom_s['d_inter']:.6f}, R_sep={geom_s['R_sep']:.6f}"
    )

    # ------------------------------------------------------------------
    # Cross-modal alignment/retrieval on held-out test samples
    # ------------------------------------------------------------------
    print("\n[3/4] Cross-modal alignment and retrieval on held-out test samples...")
    retrieval_samples = balanced_subset(test_list, args.retrieval_per_family, seed=args.seed + 3)
    retrieval_latents = extract_latents(
        model, retrieval_samples, scalers, device, args.batch_size, verbose_prefix="retrieval: "
    )
    retrieval_rows = compute_cross_modal_metrics(retrieval_latents, args.retrieval_batch_size, "test")
    write_csv(retrieval_rows, args.out_dir / "cross_modal_retrieval_summary.csv")

    if not args.skip_cross_modal_tsne:
        make_cross_modal_tsne(
            retrieval_latents,
            args.out_dir / "figure40b_cross_modal_tsne.pdf",
            perplexity=args.tsne_perplexity,
            seed=args.seed + 4,
        )

    # ------------------------------------------------------------------
    # Repeated prior sampling
    # ------------------------------------------------------------------
    sampling_summary: dict[str, Any] = {}
    if not args.skip_prior_sampling:
        print("\n[4/4] Repeated generative prior sampling...")
        run_rows, sampling_summary = run_prior_sampling(
            model=model,
            device=device,
            n_runs=args.sampling_runs,
            samples_per_run=args.samples_per_run,
            batch_size=args.batch_size,
            seed=args.seed,
            stochastic=not args.deterministic_sampling,
        )
        write_csv(run_rows, args.out_dir / "latent_sampling_per_run.csv")
        (args.out_dir / "latent_sampling_summary.json").write_text(
            json.dumps(sampling_summary, indent=2, default=json_float), encoding="utf-8"
        )
    else:
        print("\n[4/4] Prior sampling skipped.")

    # Save reusable latent vectors for audit/replotting.
    np.savez_compressed(
        args.out_dir / "latent_vectors_geometry.npz",
        mu_c=geometry_latents.mu_c,
        logvar_c=geometry_latents.logvar_c,
        mu_s=geometry_latents.mu_s,
        logvar_s=geometry_latents.logvar_s,
        labels=geometry_latents.labels.astype(str),
    )

    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": ckpt.get("epoch"),
        "device": str(device),
        "seed": args.seed,
        "latent_dim": int(model.nz),
        "split_sizes": {"train": len(train_list), "val": len(val_list), "test": len(test_list)},
        "figure40": fig40_summary,
        "geometry_circuit": geom_c,
        "geometry_hamiltonian": geom_s,
        "cross_modal_retrieval": retrieval_rows,
        "prior_sampling": sampling_summary,
        "methodology": {
            "figure40_latent_vector": "circuit posterior mean mu_c, model.eval()",
            "geometry_space": "original latent dimension, not t-SNE",
            "geometry_split": "training split, balanced by circuit family",
            "retrieval_split": "held-out test split, balanced by circuit family",
            "prior": "standard normal N(0,I)",
            "novelty": "primitive type-preserving graph isomorphism against training topologies",
        },
    }
    (args.out_dir / "latent_analysis_summary.json").write_text(
        json.dumps(summary, indent=2, default=json_float), encoding="utf-8"
    )

    print("\nDone. Main outputs:")
    for name in [
        "figure40_latent_tsne.pdf",
        "figure40b_cross_modal_tsne.pdf",
        "latent_geometry_summary.csv",
        "latent_family_geometry.csv",
        "latent_centroid_distances.csv",
        "cross_modal_retrieval_summary.csv",
        "latent_sampling_per_run.csv",
        "latent_sampling_summary.json",
        "latent_analysis_summary.json",
    ]:
        p = args.out_dir / name
        if p.exists():
            print(f"  {p}")


if __name__ == "__main__":
    main()
