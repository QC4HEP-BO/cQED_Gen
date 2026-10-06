from __future__ import annotations

"""
Level-1 constraint-aware inference experiments.

This script does two things:

1. Held-out Three_qubit_capacitive_line test:
   - loads the inference-only Three_qubit_capacitive_line dataset
   - encodes its Hamiltonian/observables with the spec encoder
   - decodes with required_primitives={"transmon": 3}
   - plots N predicted macro-circuits with NetworkX, next to the true macro circuit
   - reports topology match, parameter comparison when topology is exact, and optional Qultra validation

2. Latent sampling with primitive constraints:
   - samples z ~ N(0, I)
   - decodes several batches with different primitive constraints
   - plots generated macro-circuits with NetworkX

Example:
    python inference_threequbit_constraint.py \
        --checkpoint runs/my_ckpt.pt \
        --out-dir outputs/threequbit_constraint \
        --n-pred 12 \
        --guidance-strength 1.5 \
        --stochastic
"""

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    import networkx as nx
    HAS_NX = True
except Exception:
    nx = None
    HAS_NX = False

REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from circuit2graph import SubgType, SUBG_DEFS
from circuit2graph.definitions import MACRO_SIGNATURES
from data_loader.loader_vae import load_inference_datasets_vae
from vae_model.decoder import data_to_graph_ns

# Reuse the mature helpers already present in the repo.
from inference_hamiltonian_qultra import (  # noqa: E402
    load_checkpoint,
    topo_match,
    _unique_attr_names_from_graph,
    _scaled_attr_row_from_pred_graph,
    _best_pred_row_aligned_to_true_order,
    _get_attr_perm_indices,
    _inverse_scaled_row_safe,
    _graph_from_flat_attrs_like,
    _graph_ns_to_cqed_topology,
    _simulate_topology_arrays,
    _sample_true_observables_from_dataset,
    _HAS_QULTRA_WORKFLOW,
    import_qultra,
)

DATASET_NAME = "Three_qubit_capacitive_line"
UNIT = {
    "L": "H", "L2": "H", "C": "F", "C2": "F", "Cc": "F", "Cc_qr": "F",
    "Cc_rf": "F", "length": "m", "D": "um", "l": "m",
}


def primitive_count(g) -> dict[str, int]:
    c = Counter()
    for st_int in getattr(g, "node_types", []):
        try:
            c.update(MACRO_SIGNATURES.get(SubgType(int(st_int)), {}))
        except Exception:
            pass
    return dict(c)


def graph_label(g) -> str:
    if not getattr(g, "node_types", None):
        return "empty"
    names = []
    for st_int in g.node_types:
        try:
            names.append(SubgType(int(st_int)).name)
        except Exception:
            names.append(str(st_int))
    edges = ",".join(f"{u}-{v}" for u, v in getattr(g, "edges", [])) or "no_edges"
    return " | ".join(names) + f"\nedges: {edges}\nprim: {primitive_count(g)}"


def draw_graph(ax, g, title: str, title_color: str = "black", show_attrs: bool = False) -> None:
    ax.set_title(title, fontsize=8, color=title_color)
    ax.axis("off")
    if not HAS_NX:
        ax.text(0.5, 0.5, "networkx not installed", ha="center", va="center")
        return
    if not getattr(g, "node_types", None):
        ax.text(0.5, 0.5, "empty graph", ha="center", va="center")
        return
    G = nx.Graph()
    for i, st_int in enumerate(g.node_types):
        try:
            name = SubgType(int(st_int)).name
        except Exception:
            name = str(st_int)
        label = f"{i}\n{name}"
        if show_attrs and getattr(g, "attrs", None) and i < len(g.attrs):
            attrs = []
            for k, v in list(g.attrs[i].items())[:4]:
                if isinstance(v, (float, int)):
                    attrs.append(f"{k}={v:.2e}")
                else:
                    attrs.append(f"{k}={v}")
            if attrs:
                label += "\n" + "\n".join(attrs)
        G.add_node(i, label=label)
    for u, v in getattr(g, "edges", []):
        G.add_edge(int(u), int(v))
    pos = nx.spring_layout(G, seed=7) if len(G.nodes) > 2 else nx.shell_layout(G)
    nx.draw_networkx_edges(G, pos, ax=ax, width=1.5, alpha=0.75)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=1100, alpha=0.9)
    nx.draw_networkx_labels(G, pos, labels={i: G.nodes[i]["label"] for i in G.nodes}, ax=ax, font_size=6)


def attach_predicted_attrs(model, z: torch.Tensor, graphs: list) -> list:
    attrs_per_graph = model.param_decoder.predict(z, graphs)
    for g, attrs in zip(graphs, attrs_per_graph):
        for i, st_int in enumerate(g.node_types):
            if "dir" in SUBG_DEFS[SubgType(int(st_int))].attrs:
                attrs[i]["dir"] = float(getattr(g, "direction", [0.0] * len(g.node_types))[i])
        g.attrs = attrs
    return graphs


@torch.no_grad()
def decode_from_observables(model, sample, required_primitives, guidance_strength: float, stochastic: bool, device):
    obs_vals = sample.obs_vals.unsqueeze(0).to(device)
    obs_mask = sample.obs_mask.unsqueeze(0).to(device)
    z, _, _ = model.spec_encoder.encode(obs_vals, obs_mask)
    graphs = model.decoder.decode(
        z,
        stochastic=stochastic,
        required_primitives=required_primitives,
        guidance_strength=guidance_strength,
    )
    graphs = attach_predicted_attrs(model, z, graphs)
    return graphs[0], z


def compare_params_if_possible(sample, g_pred, scaler, use_permutation_eval: bool = True) -> tuple[list[dict], str | None]:
    g_true_scaled = data_to_graph_ns(sample, None)
    tm = topo_match(g_true_scaled.node_types, g_true_scaled.edges, g_pred.node_types, g_pred.edges)
    if not tm["exact"]:
        return [], "topology_not_exact"

    true_scaled = sample.y.tolist()
    pred_scaled = _scaled_attr_row_from_pred_graph(g_pred) if getattr(g_pred, "attrs", None) else []
    if len(pred_scaled) != len(true_scaled):
        return [], f"attr_length_mismatch true={len(true_scaled)} pred={len(pred_scaled)}"

    if use_permutation_eval:
        perms = _get_attr_perm_indices(sample, len(true_scaled))
        pred_scaled, best_perm_i, n_perm = _best_pred_row_aligned_to_true_order(true_scaled, pred_scaled, perms)
    else:
        best_perm_i, n_perm = 0, 1

    true_phys = _inverse_scaled_row_safe(scaler, true_scaled)
    pred_phys = _inverse_scaled_row_safe(scaler, pred_scaled)
    names = _unique_attr_names_from_graph(g_true_scaled)
    rows = []
    for name, t, p in zip(names, true_phys, pred_phys):
        err = float(p - t) if np.isfinite(t) and np.isfinite(p) else float("nan")
        rel = float(err / t) if np.isfinite(err) and t != 0 else float("nan")
        base = name.rsplit("_n", 1)[0]
        rows.append({
            "attr": name,
            "unit": UNIT.get(base, ""),
            "true": float(t),
            "pred": float(p),
            "abs_err": err,
            "rel_err": rel,
            "best_perm_i": int(best_perm_i),
            "n_perms": int(n_perm),
        })
    return rows, None


def validate_with_qultra(sample, g_pred, scaler, obs_scaler, f_min: float, f_max: float, use_permutation_eval: bool = True) -> dict:
    out = {"ok": False, "skipped": False, "reason": None, "metrics": {}}
    if not _HAS_QULTRA_WORKFLOW:
        out.update({"skipped": True, "reason": "qultra workflow imports unavailable"})
        return out
    try:
        qu = import_qultra()
    except Exception as exc:
        out.update({"skipped": True, "reason": f"qultra import failed: {type(exc).__name__}: {exc}"})
        return out

    g_true_scaled = data_to_graph_ns(sample, None)
    tm = topo_match(g_true_scaled.node_types, g_true_scaled.edges, g_pred.node_types, g_pred.edges)
    if not tm["exact"]:
        out["reason"] = "topology_not_exact"
        return out

    true_scaled = sample.y.tolist()
    pred_scaled = _scaled_attr_row_from_pred_graph(g_pred) if getattr(g_pred, "attrs", None) else []
    if len(pred_scaled) != len(true_scaled):
        out["reason"] = "attr_length_mismatch"
        return out
    if use_permutation_eval:
        perms = _get_attr_perm_indices(sample, len(true_scaled))
        pred_scaled, _, _ = _best_pred_row_aligned_to_true_order(true_scaled, pred_scaled, perms)

    pred_phys = _inverse_scaled_row_safe(scaler, pred_scaled)
    g_pred_phys = _graph_from_flat_attrs_like(g_true_scaled, pred_phys, "pred_threequbit_constraint")
    pred_topo = _graph_ns_to_cqed_topology(g_pred_phys, name="pred_threequbit_constraint")
    true_obs = _sample_true_observables_from_dataset(sample, obs_scaler)
    pred_obs = _simulate_topology_arrays(pred_topo, qu, f_min, f_max)
    if not pred_obs.get("ok"):
        out["reason"] = pred_obs.get("error") or pred_obs.get("stage") or "qultra_failed"
        return out

    # Simple observable comparison without mode permutation search; this is a diagnostic only.
    metrics = {}
    for key in ["frequencies", "kappa"]:
        t = np.asarray(true_obs.get(key, []), dtype=float).ravel()
        p = np.asarray(pred_obs.get(key, []), dtype=float).ravel()
        n = min(len(t), len(p))
        if n:
            metrics[f"{key}_rmse"] = float(np.sqrt(np.mean((t[:n] - p[:n]) ** 2)))
    t_chi = np.asarray(true_obs.get("chi", []), dtype=float)
    p_chi = np.asarray(pred_obs.get("chi", []), dtype=float)
    n0 = min(t_chi.shape[0] if t_chi.ndim == 2 else 0, p_chi.shape[0] if p_chi.ndim == 2 else 0)
    if n0:
        metrics["chi_rmse"] = float(np.sqrt(np.mean((t_chi[:n0, :n0] - p_chi[:n0, :n0]) ** 2)))
    out.update({"ok": True, "reason": None, "metrics": metrics})
    return out


def save_param_table(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("No parameter comparison available. Usually this means topology was not exact.\n")
        return
    lines = []
    header = f"{'attr':<12} {'true':>14} {'pred':>14} {'abs_err':>14} {'rel_err':>12} unit"
    lines.append(header)
    lines.append("-" * len(header))
    for r in rows:
        lines.append(
            f"{r['attr']:<12} {r['true']:>14.6e} {r['pred']:>14.6e} "
            f"{r['abs_err']:>14.6e} {r['rel_err']:>12.3e} {r['unit']}"
        )
    lines.append(f"\npermutation: best {rows[0]['best_perm_i']} / {rows[0]['n_perms']} candidates")
    path.write_text("\n".join(lines) + "\n")


def plot_threequbit_predictions(true_g, pred_records: list[dict], out_path: Path, show_attrs: bool = False) -> None:
    n = len(pred_records)
    cols = min(4, max(1, n))
    rows = math.ceil((n + 1) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.6 * rows), squeeze=False)
    axes_flat = axes.ravel()
    draw_graph(axes_flat[0], true_g, "TRUE held-out macro", title_color="black", show_attrs=show_attrs)
    for ax, rec in zip(axes_flat[1:], pred_records):
        tm = rec["topology_match"]
        color = "green" if tm["exact"] else "red"
        title = (
            f"pred #{rec['idx']} | exact={tm['exact']}\n"
            f"nodes {tm['n_pred']}/{tm['n_true']} edge F/P/R={tm['edge_tp']}/{tm['edge_fp']}/{tm['edge_fn']}\n"
            f"{primitive_count(rec['graph'])}"
        )
        draw_graph(ax, rec["graph"], title, title_color=color, show_attrs=show_attrs)
    for ax in axes_flat[n + 1:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_latent_samples(records_by_constraint: dict[str, list], out_path: Path, show_attrs: bool = False) -> None:
    labels = list(records_by_constraint.keys())
    n_rows = len(labels)
    n_cols = max(len(v) for v in records_by_constraint.values()) if labels else 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 3.3 * n_rows), squeeze=False)
    for r, label in enumerate(labels):
        graphs = records_by_constraint[label]
        for c in range(n_cols):
            ax = axes[r, c]
            if c >= len(graphs):
                ax.axis("off")
                continue
            g = graphs[c]
            draw_graph(ax, g, f"{label}\n{primitive_count(g)}", show_attrs=show_attrs)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def parse_constraints(items: list[str]) -> list[tuple[str, dict[str, int]]]:
    if not items:
        return [
            ("1T", {"transmon": 1}),
            ("2T", {"transmon": 2}),
            ("3T", {"transmon": 3}),
            ("1R", {"resonator": 1}),
            ("2R", {"resonator": 2}),
            ("2T+1R", {"transmon": 2, "resonator": 1}),
            ("1T+1R", {"transmon": 1, "resonator": 1}),
        ]
    parsed = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid constraint {item!r}. Use label=json, e.g. 3T='{ '{' }\"transmon\":3{ '}' }'")
        label, raw = item.split("=", 1)
        parsed.append((label, json.loads(raw)))
    return parsed


@torch.no_grad()
def run_latent_sampling(model, constraints: list[tuple[str, dict[str, int]]], n_per_constraint: int,
                        guidance_strength: float, stochastic: bool, device):
    out = {}
    for label, req in constraints:
        z = torch.randn(n_per_constraint, model.nz, device=device)
        graphs = model.decoder.decode(
            z,
            stochastic=stochastic,
            required_primitives=req,
            guidance_strength=guidance_strength,
        )
        graphs = attach_predicted_attrs(model, z, graphs)
        out[label] = graphs
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to trained checkpoint .pt/.pth")
    ap.add_argument("--out-dir", default="outputs/threequbit_constraint", help="Output directory")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample-index", type=int, default=0, help="Index in Three_qubit_capacitive_line.txt")
    ap.add_argument("--n-pred", type=int, default=12, help="Number of constrained predictions from the same H")
    ap.add_argument("--n-latent", type=int, default=4, help="Latent samples per constraint")
    ap.add_argument("--n-stress-latent", type=int, default=5, help="Latent samples per high-count stress constraint")
    ap.add_argument("--skip-stress-latent", action="store_true", help="Skip extra latent stress plot for 5T / 5T+5R / 4T+5R")
    ap.add_argument("--guidance-strength", type=float, default=1.5)
    ap.add_argument("--stochastic", action="store_true", help="Use stochastic decoding; recommended for N alternative candidates")
    ap.add_argument("--max-nodes", type=int, default=12)
    ap.add_argument("--f-min", type=float, default=1.0)
    ap.add_argument("--f-max", type=float, default=9.0)
    ap.add_argument("--show-attrs", action="store_true")
    ap.add_argument("--constraint", action="append", default=[], help="Latent constraint label=json; repeatable")
    ap.add_argument("--skip-qultra", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    model, ckpt_scalers, config = load_checkpoint(args.checkpoint, device)
    model.eval()

    print("Loading inference datasets, including held-out Three_qubit_capacitive_line...")
    samples_by_ds, inference_scalers = load_inference_datasets_vae(seed=args.seed, max_nodes=args.max_nodes)
    if DATASET_NAME not in samples_by_ds:
        raise RuntimeError(f"Dataset {DATASET_NAME!r} not found. Available: {sorted(samples_by_ds)}")
    samples = samples_by_ds[DATASET_NAME]
    if not samples:
        raise RuntimeError(f"Dataset {DATASET_NAME!r} is empty")
    sample = samples[min(max(args.sample_index, 0), len(samples) - 1)]
    scaler = inference_scalers[DATASET_NAME].param_scaler
    obs_scaler = inference_scalers[DATASET_NAME].obs_scaler
    true_g = data_to_graph_ns(sample, scaler)
    true_g_scaled = data_to_graph_ns(sample, None)

    req = {"transmon": 3, "coupler":2}
    pred_records = []
    all_param_rows = []
    qultra_results = []
    for i in range(args.n_pred):
        g_pred, _ = decode_from_observables(
            model,
            sample,
            required_primitives=req,
            guidance_strength=args.guidance_strength,
            stochastic=args.stochastic or args.n_pred > 1,
            device=device,
        )
        tm = topo_match(true_g_scaled.node_types, true_g_scaled.edges, g_pred.node_types, g_pred.edges)
        param_rows, param_reason = compare_params_if_possible(sample, g_pred, scaler)
        qres = {"skipped": True, "reason": "--skip-qultra"}
        if not args.skip_qultra:
            qres = validate_with_qultra(sample, g_pred, scaler, obs_scaler, args.f_min, args.f_max)
        pred_records.append({
            "idx": i,
            "graph": g_pred,
            "topology_match": tm,
            "primitive_count": primitive_count(g_pred),
            "param_reason": param_reason,
            "qultra": qres,
            "label": graph_label(g_pred),
        })
        if param_rows and not all_param_rows:
            all_param_rows = param_rows
        qultra_results.append(qres)

    plot_threequbit_predictions(true_g, pred_records, out_dir / "threequbit_constrained_predictions.png", show_attrs=args.show_attrs)
    save_param_table(all_param_rows, out_dir / "threequbit_param_comparison.txt")

    summary = {
        "dataset": DATASET_NAME,
        "sample_index": int(args.sample_index),
        "required_primitives": req,
        "guidance_strength": float(args.guidance_strength),
        "stochastic": bool(args.stochastic or args.n_pred > 1),
        "n_pred": int(args.n_pred),
        "true_graph": graph_label(true_g_scaled),
        "predictions": [
            {
                "idx": r["idx"],
                "topology_match": r["topology_match"],
                "primitive_count": r["primitive_count"],
                "param_reason": r["param_reason"],
                "qultra": r["qultra"],
                "label": r["label"],
            }
            for r in pred_records
        ],
    }
    (out_dir / "threequbit_summary.json").write_text(json.dumps(summary, indent=2))

    exact = sum(1 for r in pred_records if r["topology_match"]["exact"])
    satisfy = sum(1 for r in pred_records if r["primitive_count"].get("transmon", 0) >= 3)
    qultra_ok = sum(1 for r in pred_records if r["qultra"].get("ok"))
    print("\nHeld-out Three_qubit_capacitive_line")
    print(f"  predictions: {len(pred_records)}")
    print(f"  satisfy transmon>=3: {satisfy}/{len(pred_records)}")
    print(f"  exact topology: {exact}/{len(pred_records)}")
    print(f"  qultra ok: {qultra_ok}/{len(pred_records)}")
    print(f"  graph plot: {out_dir / 'threequbit_constrained_predictions.png'}")
    print(f"  param table: {out_dir / 'threequbit_param_comparison.txt'}")
    print(f"  summary: {out_dir / 'threequbit_summary.json'}")

    constraints = parse_constraints(args.constraint)
    latent_records = run_latent_sampling(
        model,
        constraints,
        n_per_constraint=args.n_latent,
        guidance_strength=args.guidance_strength,
        stochastic=True if args.n_latent > 1 else args.stochastic,
        device=device,
    )
    plot_latent_samples(latent_records, out_dir / "latent_constraint_samples.png", show_attrs=args.show_attrs)
    latent_summary = {
        label: [
            {"primitive_count": primitive_count(g), "label": graph_label(g)}
            for g in graphs
        ]
        for label, graphs in latent_records.items()
    }
    (out_dir / "latent_constraint_samples.json").write_text(json.dumps(latent_summary, indent=2))
    print(f"\nLatent constraint sampling plot: {out_dir / 'latent_constraint_samples.png'}")
    print(f"Latent constraint sampling summary: {out_dir / 'latent_constraint_samples.json'}")

    if not args.skip_stress_latent:
        stress_constraints = [
            ("5T", {"transmon": 5}),
            ("5T+5R", {"transmon": 5, "resonator": 5}),
            ("4T+5R", {"transmon": 4, "resonator": 5}),
        ]
        stress_records = run_latent_sampling(
            model,
            stress_constraints,
            n_per_constraint=args.n_stress_latent,
            guidance_strength=args.guidance_strength,
            stochastic=True,
            device=device,
        )
        plot_latent_samples(
            stress_records,
            out_dir / "latent_high_count_stress_samples.png",
            show_attrs=args.show_attrs,
        )
        stress_summary = {
            label: [
                {"primitive_count": primitive_count(g), "label": graph_label(g)}
                for g in graphs
            ]
            for label, graphs in stress_records.items()
        }
        (out_dir / "latent_high_count_stress_samples.json").write_text(json.dumps(stress_summary, indent=2))
        print(f"High-count latent stress plot: {out_dir / 'latent_high_count_stress_samples.png'}")
        print(f"High-count latent stress summary: {out_dir / 'latent_high_count_stress_samples.json'}")


if __name__ == "__main__":
    main()
