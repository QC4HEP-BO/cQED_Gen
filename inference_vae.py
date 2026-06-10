from __future__ import annotations

import argparse
import inspect
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

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
from data_loader.schema import DATASETS
from circuit2graph import SubgType, SUBG_DEFS

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



def _unique_attr_names_from_graph(g):
    """Return display names in the exact same order as sample.y/scaler columns.

    Important for TCT: the compressed TCT node has attrs
    [L, C, Cc, L2, C2].  Using DATASETS[ds].block_params or a single dict
    would relabel/drop duplicated L/C columns.
    """
    names = []
    counts = {}
    for st_int in g.node_types:
        for attr_name in SUBG_DEFS[SubgType(st_int)].attrs:
            base = attr_name
            if base in ("L2", "C2"):
                display = base
            elif base in counts:
                counts[base] += 1
                display = f"{base}{counts[base]}"
            else:
                counts[base] = 1
                display = base
            names.append(display)
    return names


def _scaled_attr_row_from_pred_graph(g_pred):
    """Flatten predicted attrs by node and SUBG_DEFS order, preserving duplicates."""
    row = []
    for node_i, st_int in enumerate(g_pred.node_types):
        attrs_ordered = SUBG_DEFS[SubgType(st_int)].attrs
        node_attrs = g_pred.attrs[node_i] if node_i < len(g_pred.attrs) else {}
        row.extend(float(node_attrs.get(a, float("nan"))) for a in attrs_ordered)
    return row

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
def run_circuit_encoder(vae: GraphVAE, data: list, scalers: dict, device: torch.device, batch_size: int = 128, stochastic: bool = False) -> tuple[dict, bool, dict[str, int]]:
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

        topo_results = []
        true_scaled_rows = []
        pred_scaled_rows = []

        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            z, _, _ = vae.encode(batch, scalers)
            graphs_pred, used_ds_ids = _decode_graphs(vae, z, batch, stochastic, ds_map)
            decode_used_ds_ids = decode_used_ds_ids or used_ds_ids

            for s, g_pred in zip(batch, graphs_pred):
                g_true = data_to_graph_ns(s, scaler)
                tm = topo_match(g_true.node_types, g_true.edges, g_pred.node_types, g_pred.edges)
                topo_results.append(tm)
                true_scaled_rows.append(s.y.tolist())
                if flat_attrs is None:
                    flat_attrs = _unique_attr_names_from_graph(g_true)
                if tm["exact"] and g_pred.attrs:
                    row = _scaled_attr_row_from_pred_graph(g_pred)
                    if len(row) != len(flat_attrs):
                        row = [float("nan")] * len(flat_attrs)
                    pred_scaled_rows.append(row)
                else:
                    pred_scaled_rows.append([float("nan")] * len(flat_attrs))

        y_true_mat = np.array(true_scaled_rows, dtype=np.float64)
        y_pred_mat = np.array(pred_scaled_rows, dtype=np.float64)
        y_true_phys = scaler.inverse_transform(y_true_mat)
        y_pred_phys = scaler.inverse_transform(y_pred_mat)

        out[ds_name] = {
            "topo_results": topo_results,
            "params": {attr: {"true": y_true_phys[:, k], "pred": y_pred_phys[:, k]} for k, attr in enumerate(flat_attrs)},
        }

    return out, decode_used_ds_ids, ds_map


@torch.no_grad()
def run_spec_encoder(vae: GraphVAE, data: list, scalers: dict, device: torch.device, batch_size: int = 128, stochastic: bool = False) -> tuple[dict, bool, dict[str, int]]:
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

        topo_results = []
        true_scaled_rows = []
        pred_scaled_rows = []

        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            obs_vals = torch.stack([s.obs_vals for s in batch]).to(device)
            obs_mask = torch.stack([s.obs_mask for s in batch]).to(device)
            z_s, _, _ = vae.spec_encoder.encode(obs_vals, obs_mask)
            graphs_pred, used_ds_ids = _decode_graphs(vae, z_s, batch, stochastic, ds_map)
            decode_used_ds_ids = decode_used_ds_ids or used_ds_ids

            for s, g_pred in zip(batch, graphs_pred):
                scaler_b = scalers[s.dataset_name].param_scaler
                g_true = data_to_graph_ns(s, scaler_b)
                tm = topo_match(g_true.node_types, g_true.edges, g_pred.node_types, g_pred.edges)
                topo_results.append(tm)
                true_scaled_rows.append(s.y.tolist())
                if flat_attrs is None:
                    flat_attrs = _unique_attr_names_from_graph(g_true)
                if tm["exact"] and g_pred.attrs:
                    row = _scaled_attr_row_from_pred_graph(g_pred)
                    if len(row) != len(flat_attrs):
                        row = [float("nan")] * len(flat_attrs)
                    pred_scaled_rows.append(row)
                else:
                    pred_scaled_rows.append([float("nan")] * len(flat_attrs))

        y_true_mat = np.array(true_scaled_rows, dtype=np.float64)
        y_pred_mat = np.array(pred_scaled_rows, dtype=np.float64)
        y_true_phys = scaler.inverse_transform(y_true_mat)
        y_pred_phys = scaler.inverse_transform(y_pred_mat)

        out[ds_name] = {
            "topo_results": topo_results,
            "params": {attr: {"true": y_true_phys[:, k], "pred": y_pred_phys[:, k]} for k, attr in enumerate(flat_attrs)},
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
        print("  Metriche parametri  (solo topologia corretta):")
        print(f"    {'Attr':<10} {'R²':>10} {'RMSE':>14} {'N_valid':>8}  scala")
        print("    " + "-" * 48)
        for attr, m in pm_metrics.items():
            unit = UNITS.get(attr, "")
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
                ax_label = f"log10 {attr} [{UNITS.get(attr, '?')}]"
            else:
                tv, pv = t, p
                ax_label = f"{attr} [{UNITS.get(attr, '?')}]"
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
            unit = UNITS.get(attr, "")
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
    p.add_argument("--topo-errors-dataset", default="Three_qubit_capacitive_line")
    p.add_argument("--topo-errors-n", type=int, default=20)
    p.add_argument("--plot-latent-samples", action="store_true")
    p.add_argument("--latent-n", type=int, default=20)
    p.add_argument("--only-latent", action="store_true",
        help="Esegue solo il campionamento dal latent space, salta rami A/B e caricamento dataset.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
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
        results_circ, used_ds_ids_c, ds_map_c = run_circuit_encoder(model, data, scalers, device, batch_size=args.batch_size, stochastic=args.stochastic)
        print(f"  completato in {time.time() - t0:.1f}s")
        print_report("Circuit Encoder  (G -> z^c -> Ghat)", results_circ, used_ds_ids_c, ds_map_c)
        plot_scatter(results_circ, str(out_dir / "inference_circuit_enc_params.png"), args.n_samples, "circuit-enc", args.ckpt)
        plot_topo_breakdown(results_circ, str(out_dir / "inference_circuit_enc_topo.png"), "circuit-enc", args.ckpt)

    if not args.only_circuit and not args.only_latent:
        print("\n[Ramo B] Spec encoder  obs -> z^s -> decoder...")
        t0 = time.time()
        results_spec, used_ds_ids_s, ds_map_s = run_spec_encoder(model, data, scalers, device, batch_size=args.batch_size, stochastic=args.stochastic)
        print(f"  completato in {time.time() - t0:.1f}s")
        print_report("Spec Encoder  (obs -> z^s -> Ghat)", results_spec, used_ds_ids_s, ds_map_s)
        plot_scatter(results_spec, str(out_dir / "inference_spec_enc_params.png"), args.n_samples, "spec-enc", args.ckpt)
        plot_topo_breakdown(results_spec, str(out_dir / "inference_spec_enc_topo.png"), "spec-enc", args.ckpt)

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
