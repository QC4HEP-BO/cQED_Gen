
#!/usr/bin/env python3
"""
analysis_4_2_1_training_validation_loss.py
==========================================

Analysis for Sec. 4.2.1 (Training and Validation Loss) of the cQED-Gen thesis.

Run this script from the BASE DIRECTORY of the cQED-Gen repository, e.g.

    python analysis_4_2_1_training_validation_loss.py

By default it reads ``best_vae_global.pt`` and writes the results to

    analysis_results/4_2_1_training_validation_loss/

It produces two thesis-ready PDF figures:

1. ``figure35_training_validation_loss.pdf``
   Complete training and validation objective versus epoch. The best validation
   checkpoint is marked explicitly. Raw and smoothed curves are both shown.

2. ``figure36_validation_loss_per_family.pdf``
   Total validation objective evaluated independently for each training circuit
   family at the selected (best-validation) checkpoint. Error bars are the
   standard error across validation batches. The global validation loss obtained
   by re-evaluating the complete mixed validation set is shown as a dashed line.

It also writes CSV/TXT files with all numerical quantities needed in the text:

- ``training_validation_loss_history.csv``
- ``training_validation_summary.csv``
- ``loss_improvement_intervals.csv``
- ``validation_loss_per_family.csv``
- ``analysis_summary.txt``

Important methodological note
-----------------------------
The checkpoint stores the global train/validation loss history, so Figure 35 is
reconstructed exactly from the recorded run. Per-family historical curves are NOT
stored in the checkpoint. Figure 36 therefore evaluates the *selected checkpoint*
on each validation-family subset, which is the quantity described by the thesis
caption ("at the selected global checkpoint"). No retraining is performed.

The family-specific evaluation uses exactly the same reservoir sampling, split
fractions, seed, and saved GLOBAL scaler as the production run. The full objective
is evaluated without changing its loss weights.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import pickle
import random
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Display names used in plots/tables
# -----------------------------------------------------------------------------

DISPLAY_NAMES = OrderedDict([
    ("Qubit", "Qubit"),
    ("Resonator", "Resonator"),
    ("Qubit_resonator_feedline_capacitive", "Q-R-F\n(capacitive)"),
    ("Qubit_resonator_feedline_inductive_nognd", "Q-R-F\n(inductive)"),
    ("Qubit_resonator_resonator", "Q-R-R"),
    ("Resonator_qubit_resonator", "R-Q-R"),
    ("Two_qubit_with_capacitive_coupling", "2Q\ncapacitive"),
    ("Three_qubit_capacitive_star", "3Q star"),
])


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def _torch_load(path: Path):
    """Compatibility wrapper across PyTorch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average with shortened windows at the boundaries."""
    values = np.asarray(values, dtype=float)
    if window <= 1 or len(values) <= 2:
        return values.copy()
    window = min(int(window), len(values))
    half = window // 2
    out = np.empty_like(values, dtype=float)
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        out[i] = float(np.mean(values[lo:hi]))
    return out


def _validation_epochs(n_train: int, n_val: int, val_every: int) -> np.ndarray:
    """Reconstruct epoch indices at which validation was recorded."""
    if n_val == n_train:
        return np.arange(1, n_train + 1, dtype=int)

    val_every = max(1, int(val_every))
    epochs = [1]
    epochs.extend(e for e in range(2, n_train + 1) if e % val_every == 0)

    # A resumed run can make the exact schedule ambiguous from the checkpoint
    # alone. Use the known schedule when possible, otherwise fail loudly rather
    # than silently assigning the validation values to wrong epochs.
    if len(epochs) != n_val:
        raise RuntimeError(
            "Could not reconstruct validation epochs unambiguously: "
            f"len(train_losses)={n_train}, len(val_losses)={n_val}, "
            f"val_every={val_every}. The production checkpoint should store one "
            "validation value per epoch for this analysis."
        )
    return np.asarray(epochs, dtype=int)


def _fmt(x: float, digits: int = 6) -> str:
    if not np.isfinite(x):
        return "nan"
    if x == 0:
        return "0"
    if abs(x) < 1e-3 or abs(x) >= 1e4:
        return f"{x:.4e}"
    return f"{x:.{digits}f}"


def _write_dict_csv(path: Path, rows: list[tuple[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for key, value in rows:
            w.writerow([key, value])


# -----------------------------------------------------------------------------
# Figure 35 + numerical summary from the checkpoint history
# -----------------------------------------------------------------------------

def analyse_global_history(
    ckpt: dict,
    out_dir: Path,
    smooth_window: int,
) -> dict:
    train = np.asarray(ckpt.get("train_losses", []), dtype=float)
    val = np.asarray(ckpt.get("val_losses", []), dtype=float)
    cfg = ckpt.get("config", {})

    if train.size == 0 or val.size == 0:
        raise RuntimeError(
            "Checkpoint does not contain train_losses / val_losses. "
            "Use the production checkpoint saved by train_vae.py."
        )

    train_epochs = np.arange(1, len(train) + 1, dtype=int)
    val_epochs = _validation_epochs(
        n_train=len(train),
        n_val=len(val),
        val_every=int(cfg.get("val_every", 1)),
    )

    best_val_index = int(np.argmin(val))
    best_epoch = int(val_epochs[best_val_index])
    best_val = float(val[best_val_index])
    train_at_best = float(train[best_epoch - 1])
    gap_best = best_val - train_at_best

    final_epoch = int(train_epochs[-1])
    train_1 = float(train[0])
    train_final = float(train[-1])
    val_1 = float(val[0])
    val_final = float(val[-1])

    epoch50 = min(50, final_epoch)
    train_50 = float(train[epoch50 - 1])

    train_reduction_1_50 = 100.0 * (1.0 - train_50 / train_1)
    train_reduction_1_final = 100.0 * (1.0 - train_final / train_1)
    val_reduction_1_best = 100.0 * (1.0 - best_val / val_1)
    val_reduction_1_final = 100.0 * (1.0 - val_final / val_1)

    denom = train_1 - train_final
    frac_total_by_50 = (
        100.0 * (train_1 - train_50) / denom if abs(denom) > 0 else float("nan")
    )

    # Improvement rates in two useful phases: 1->50 and 50->best.
    def interval_stats(series: np.ndarray, a: int, b: int, label: str, branch: str):
        if b <= a or a < 1 or b > len(series):
            return None
        y0 = float(series[a - 1])
        y1 = float(series[b - 1])
        n = b - a
        rel = 100.0 * (1.0 - y1 / y0) if y0 != 0 else float("nan")
        abs_per_epoch = (y0 - y1) / n
        comp_rel_per_epoch = (
            100.0 * (1.0 - (y1 / y0) ** (1.0 / n))
            if y0 > 0 and y1 > 0 else float("nan")
        )
        return {
            "branch": branch,
            "interval": label,
            "start_epoch": a,
            "end_epoch": b,
            "start_loss": y0,
            "end_loss": y1,
            "absolute_reduction": y0 - y1,
            "relative_reduction_pct": rel,
            "mean_absolute_reduction_per_epoch": abs_per_epoch,
            "compound_relative_reduction_per_epoch_pct": comp_rel_per_epoch,
        }

    intervals = []
    for item in [
        interval_stats(train, 1, epoch50, f"1-{epoch50}", "train"),
        interval_stats(train, epoch50, best_epoch, f"{epoch50}-{best_epoch}", "train"),
        interval_stats(train, 1, best_epoch, f"1-{best_epoch}", "train"),
        interval_stats(train, 1, final_epoch, f"1-{final_epoch}", "train"),
    ]:
        if item is not None:
            intervals.append(item)

    # Validation intervals are exact only when there is one value per epoch.
    if len(val) == len(train):
        for item in [
            interval_stats(val, 1, epoch50, f"1-{epoch50}", "validation"),
            interval_stats(val, epoch50, best_epoch, f"{epoch50}-{best_epoch}", "validation"),
            interval_stats(val, 1, best_epoch, f"1-{best_epoch}", "validation"),
            interval_stats(val, 1, final_epoch, f"1-{final_epoch}", "validation"),
        ]:
            if item is not None:
                intervals.append(item)

    with (out_dir / "loss_improvement_intervals.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        fieldnames = [
            "branch", "interval", "start_epoch", "end_epoch", "start_loss",
            "end_loss", "absolute_reduction", "relative_reduction_pct",
            "mean_absolute_reduction_per_epoch",
            "compound_relative_reduction_per_epoch_pct",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(intervals)

    summary_rows = [
        ("checkpoint_recorded_final_epoch", int(ckpt.get("epoch", final_epoch))),
        ("history_final_epoch", final_epoch),
        ("best_validation_epoch", best_epoch),
        ("best_validation_loss", best_val),
        ("training_loss_epoch_1", train_1),
        (f"training_loss_epoch_{epoch50}", train_50),
        ("training_loss_at_best_validation_epoch", train_at_best),
        ("training_loss_final_epoch", train_final),
        ("validation_loss_epoch_1", val_1),
        ("validation_loss_final_epoch", val_final),
        ("train_validation_gap_at_best", gap_best),
        ("absolute_train_validation_gap_at_best", abs(gap_best)),
        ("training_reduction_epoch1_to_epoch50_pct", train_reduction_1_50),
        ("training_reduction_epoch1_to_final_pct", train_reduction_1_final),
        ("fraction_of_total_training_reduction_reached_by_epoch50_pct", frac_total_by_50),
        ("validation_reduction_epoch1_to_best_pct", val_reduction_1_best),
        ("validation_reduction_epoch1_to_final_pct", val_reduction_1_final),
        ("checkpoint_best_val_loss_field", float(ckpt.get("best_val_loss", float("nan")))),
    ]
    _write_dict_csv(out_dir / "training_validation_summary.csv", summary_rows)

    # Full history table. Leave validation blank at non-validation epochs.
    val_by_epoch = {int(e): float(v) for e, v in zip(val_epochs, val)}
    with (out_dir / "training_validation_loss_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "validation_loss", "train_val_gap"])
        for e, t in zip(train_epochs, train):
            vv = val_by_epoch.get(int(e), None)
            gap = None if vv is None else vv - float(t)
            w.writerow([int(e), float(t), "" if vv is None else vv, "" if gap is None else gap])

    # Figure 35
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(train_epochs, train, linewidth=0.9, alpha=0.35, label="Training (raw)")
    ax.plot(val_epochs, val, linewidth=0.9, alpha=0.35, label="Validation (raw)")

    train_sm = _smooth(train, smooth_window)
    val_sm = _smooth(val, smooth_window)
    if smooth_window > 1:
        ax.plot(train_epochs, train_sm, linewidth=2.0,
                label=f"Training ({smooth_window}-epoch moving average)")
        ax.plot(val_epochs, val_sm, linewidth=2.0,
                label=f"Validation ({smooth_window}-epoch moving average)")

    ax.axvline(best_epoch, linestyle="--", linewidth=1.2,
               label=f"Best validation checkpoint (epoch {best_epoch})")
    ax.scatter([best_epoch], [best_val], s=35, zorder=5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Total objective $\mathcal{L}_{\mathrm{tot}}$")
    ax.set_xlim(1, final_epoch)
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "figure35_training_validation_loss.pdf", bbox_inches="tight")
    plt.close(fig)

    return {
        "final_epoch": final_epoch,
        "best_epoch": best_epoch,
        "best_val": best_val,
        "train_1": train_1,
        "train_50": train_50,
        "train_final": train_final,
        "train_at_best": train_at_best,
        "gap_best": gap_best,
        "train_reduction_1_50": train_reduction_1_50,
        "train_reduction_1_final": train_reduction_1_final,
        "frac_total_by_50": frac_total_by_50,
        "val_reduction_1_best": val_reduction_1_best,
        "val_reduction_1_final": val_reduction_1_final,
    }


# -----------------------------------------------------------------------------
# Production split reconstruction using the checkpoint's saved GLOBAL scaler
# -----------------------------------------------------------------------------

def _load_saved_scalers(ckpt: dict):
    if "scalers" in ckpt:
        return pickle.loads(ckpt["scalers"])
    if "global_scaler" in ckpt:
        gs = pickle.loads(ckpt["global_scaler"])
        return {"__global__": gs}
    raise RuntimeError("Checkpoint does not contain saved scaler information.")


def build_validation_split_from_checkpoint(
    ckpt: dict,
    global_scaler,
):
    """Rebuild exactly the per-family validation subsets from the production seed.

    For speed, only the rows that end up in the validation partition are
    graphlized. This is exactly equivalent to the production loader for the
    current training datasets because graph size is topology-defined and all
    included families satisfy ``max_nodes`` independently of their parameters.
    """
    try:
        from data_loader.schema import train_datasets, OBS_PARSERS
        from data_loader.loader_vae import (
            _read_raw,
            _fill_attrs,
            _compute_attr_perm_indices,
            _extract_params,
            _extract_param_attr_names,
            _topo_to_data_vae,
        )
        from circuit2graph import graphlize
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Could not import the repository data-loader dependencies. Run this script "
            "inside the same Python environment used to train cQED-Gen (in particular, "
            "torch_geometric must be installed)."
        ) from exc

    cfg = ckpt.get("config", {})
    seed = int(cfg.get("seed", 42))
    train_frac = float(cfg.get("train_frac", 0.70))
    val_frac = float(cfg.get("val_frac", 0.15))
    max_nodes = int(cfg.get("max_nodes", 12))

    rng = random.Random(seed)
    np.random.seed(seed)

    definitions = train_datasets()
    raw_by_family = OrderedDict()

    # First consume the reservoir-sampling RNG exactly as in the production
    # loader, for every training family, before any split shuffle is performed.
    for ds_name, defn in definitions.items():
        raw_entries = _read_raw(defn.path, ds_name, defn.n_samples, rng)
        if not raw_entries:
            raise RuntimeError(f"No rows loaded for {ds_name} from {defn.path}")

        # In the production loader max_nodes filtering occurs before splitting.
        # For the present fixed-topology datasets the compressed graph size does
        # not depend on the sampled physical parameters, so one topology check is
        # sufficient to establish whether every row survives the filter.
        template_compressed = graphlize(defn.topology_fn())
        if max_nodes and len(template_compressed._nodes) > max_nodes:
            raise RuntimeError(
                f"Dataset {ds_name} has {len(template_compressed._nodes)} macro-nodes, "
                f"above max_nodes={max_nodes}. Fast exact validation reconstruction "
                "cannot be used for a partially filtered family."
            )
        raw_by_family[ds_name] = raw_entries

    val_by_family = OrderedDict()
    global_val = []

    # Now reproduce the same per-family shuffle and split, but graphlize only the
    # validation rows instead of all 10k selected rows.
    for ds_name, raw_entries in raw_by_family.items():
        defn = definitions[ds_name]
        obs_parser = OBS_PARSERS[ds_name]
        rng.shuffle(raw_entries)

        n = len(raw_entries)
        n_test = max(1, int(round(n * (1.0 - train_frac - val_frac))))
        n_val = max(1, int(round(n * val_frac)))
        n_train = n - n_test - n_val
        selected = raw_entries[n_train:n_train + n_val]

        raw_template = defn.topology_fn()
        val_samples = []
        for attrs, obs_kw in selected:
            topo = _fill_attrs(raw_template, attrs)
            compressed = graphlize(topo)
            attr_perm_indices = _compute_attr_perm_indices(topo, compressed)

            y_raw = np.asarray(_extract_params(compressed), dtype=np.float64)
            y_names = _extract_param_attr_names(compressed)
            y_scaled = global_scaler.param_scaler.transform(y_raw, y_names)

            obs_raw, obs_mask = obs_parser(**obs_kw)
            obs_scaled = global_scaler.obs_scaler.transform_row(obs_raw, obs_mask)

            sample = _topo_to_data_vae(
                compressed,
                y_scaled,
                obs_scaled,
                obs_mask,
                ds_name,
                attr_perm_indices=attr_perm_indices,
            )
            val_samples.append(sample)

        val_by_family[ds_name] = val_samples
        global_val.extend(val_samples)

    # Same final mixed-validation shuffle as the production loader. Kept for
    # reproducibility, even though Figure 36 uses the stored best global loss as
    # its dashed reference.
    rng.shuffle(global_val)
    return val_by_family, global_val


# -----------------------------------------------------------------------------
# Exact loss evaluation at the best-checkpoint model
# -----------------------------------------------------------------------------

def _build_model_from_checkpoint(ckpt: dict, device: torch.device):
    try:
        from vae_model.vae import GraphVAE
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Could not import cQED-Gen model dependencies. Run this script inside "
            "the environment used for training (torch_geometric is required)."
        ) from exc

    cfg = ckpt["config"]
    model = GraphVAE(
        nz=cfg["nz"],
        hv_dim=cfg["hv_dim"],
        inner_hidden=cfg["inner_hidden"],
        inner_layers=cfg["inner_layers"],
        outer_hidden=cfg["outer_hidden"],
        outer_layers=cfg["outer_layers"],
        d=cfg["d"],
        nhead=cfg["nhead"],
        tf_layers=cfg["tf_layers"],
        max_nodes=cfg["max_nodes"],
        hs=cfg["hs"],
        ggnn_rounds=cfg["ggnn_rounds"],
        max_nodes_dec=cfg["max_nodes_dec"],
        param_hidden=cfg["param_hidden"],
        param_layers=cfg["param_layers"],
        spec_d=cfg["spec_d"],
        spec_layers=cfg["spec_layers"],
        cg_hidden=cfg["cg_hidden"],
        dropout=cfg["dropout"],
        beta_start=cfg["beta_start"],
        beta_max=cfg["beta_max"],
        warmup_steps=cfg["warmup_steps"],
        attrs_scale=cfg["attrs_scale"],
        align_scale=cfg["align_scale"],
        nce_scale=cfg["nce_scale"],
        cg_scale=cfg["cg_scale"],
        spec_recon_scale=cfg["spec_recon_scale"],
        tau=cfg["tau"],
        class_weight_end=cfg["class_weight_end"],
    ).to(device)

    payload = {
        "model_state": ckpt["model_state"],
        "step": ckpt.get("step", 0),
    }
    model.load_state_dict_full(payload)
    model.eval()
    return model


def _make_loader(samples, batch_size: int):
    try:
        from torch.utils.data import DataLoader
        from train_vae import collate_vae
    except ModuleNotFoundError as exc:
        raise RuntimeError("Could not import train_vae.collate_vae.") from exc
    return DataLoader(
        samples,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_vae,
        num_workers=0,
    )


@torch.no_grad()
def _evaluate_loader_batch_losses(
    model,
    samples,
    scalers,
    device: torch.device,
    batch_size: int,
    use_amp: bool,
):
    loader = _make_loader(samples, batch_size=batch_size)
    losses = []

    autocast_enabled = bool(use_amp) and device.type == "cuda"
    model.eval()

    for batch in loader:
        obs_vals = batch["obs_vals"].to(device)
        obs_mask = batch["obs_mask"].to(device)
        with torch.amp.autocast("cuda", enabled=autocast_enabled):
            loss, _ = model(
                samples=batch["samples"],
                scalers=scalers,
                obs_vals=obs_vals,
                obs_mask=obs_mask,
            )
        losses.append(float(loss.detach().cpu()))

    arr = np.asarray(losses, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    sem = float(std / math.sqrt(len(arr))) if len(arr) > 0 else float("nan")
    return mean, std, sem, arr


def analyse_validation_per_family(
    ckpt: dict,
    out_dir: Path,
    device: torch.device,
    eval_batch_size: int | None,
) -> dict:
    scalers = _load_saved_scalers(ckpt)
    global_scaler = scalers.get("__global__", None) if isinstance(scalers, dict) else None
    if global_scaler is None:
        # Older checkpoints may map every training family to the same scaler.
        if isinstance(scalers, dict) and scalers:
            global_scaler = next(iter(scalers.values()))
        else:
            raise RuntimeError("Could not recover the saved global scaler.")

    val_by_family, global_val = build_validation_split_from_checkpoint(ckpt, global_scaler)
    model = _build_model_from_checkpoint(ckpt, device)

    cfg = ckpt.get("config", {})
    batch_size = int(eval_batch_size or cfg.get("batch_size", 128))
    use_amp = bool(cfg.get("use_amp", True))

    # The exact global validation loss at the selected checkpoint is already
    # stored by the training loop. Using it here also preserves the original
    # mixed-batch composition used during training.
    global_mean = float(ckpt.get("best_val_loss", float("nan")))
    global_std = float("nan")
    global_sem = float("nan")

    rows = []
    for ds_name, samples in val_by_family.items():
        print(f"    evaluating {ds_name} ({len(samples)} validation samples)...")
        mean, std, sem, batch_losses = _evaluate_loader_batch_losses(
            model=model,
            samples=samples,
            scalers=scalers,
            device=device,
            batch_size=batch_size,
            use_amp=use_amp,
        )
        n_macro = int(samples[0].enc_n_circuit) if samples else 0
        rows.append({
            "dataset_name": ds_name,
            "display_name": DISPLAY_NAMES.get(ds_name, ds_name.replace("_", " ")),
            "n_validation_samples": len(samples),
            "n_validation_batches": len(batch_losses),
            "n_macro_nodes": n_macro,
            "mean_validation_loss": mean,
            "std_across_batches": std,
            "sem_across_batches": sem,
            "min_batch_loss": float(np.min(batch_losses)),
            "max_batch_loss": float(np.max(batch_losses)),
        })

    with (out_dir / "validation_loss_per_family.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        fieldnames = list(rows[0].keys()) if rows else []
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # Figure 36: one bar per family, global mixed-validation loss as reference.
    labels = [r["display_name"] for r in rows]
    means = np.asarray([r["mean_validation_loss"] for r in rows], dtype=float)
    sems = np.asarray([r["sem_across_batches"] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    x = np.arange(len(rows))
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    colors = [cycle[i % len(cycle)] for i in range(len(rows))] if cycle else None
    ax.bar(x, means, yerr=sems, capsize=3, linewidth=0.7, color=colors)
    ax.axhline(
        global_mean,
        linestyle="--",
        linewidth=1.25,
        label=f"Global mixed validation loss = {global_mean:.4f}",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(r"Total validation objective $\mathcal{L}_{\mathrm{tot}}$")
    ax.set_xlabel("Circuit family")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "figure36_validation_loss_per_family.pdf", bbox_inches="tight")
    plt.close(fig)

    return {
        "global_validation_loss_reevaluated": global_mean,
        "global_validation_loss_std_batches": global_std,
        "global_validation_loss_sem_batches": global_sem,
        "rows": rows,
    }



# -----------------------------------------------------------------------------
# Sec. 4.2.2 - epoch-wise loss-component recording and analysis
# -----------------------------------------------------------------------------

# These two coefficients are fixed defaults of TransformerTopologyDecoder.loss
# in the current repository implementation (decoder.py).
LAMBDA_TYPE = 0.50
LAMBDA_POSITION = 0.05


class TrainingComponentRecorder:
    """Collect mean loss components per epoch without changing training semantics.

    The production ``train_vae.py`` returns a component dictionary for every
    training batch and every validation pass, but the original loop only prints
    these values and does not persist them. This recorder is attached through
    lightweight runtime hooks when this script is invoked with
    ``--train-with-analysis``. The model, optimizer and gradients are untouched.
    """

    def __init__(self, existing_csv: Path | None = None):
        self.rows: dict[tuple[int, str], dict[str, float | int | str]] = {}
        self.current_train_epoch: int | None = None
        self.train_sums: defaultdict[str, float] = defaultdict(float)
        self.train_batches = 0
        self.val_loader_id: int | None = None
        self.test_loader_id: int | None = None
        self.history_path: Path | None = existing_csv

        if existing_csv is not None and existing_csv.is_file():
            with existing_csv.open('r', newline='', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    epoch = int(row['epoch'])
                    split = row['split']
                    parsed: dict[str, float | int | str] = {
                        'epoch': epoch,
                        'split': split,
                        'n_batches': int(float(row.get('n_batches', 0) or 0)),
                    }
                    for k, v in row.items():
                        if k in {'epoch', 'split', 'n_batches'} or v in ('', None):
                            continue
                        try:
                            parsed[k] = float(v)
                        except ValueError:
                            parsed[k] = v
                    self.rows[(epoch, split)] = parsed

    @staticmethod
    def _numeric_components(comp: dict) -> dict[str, float]:
        out: dict[str, float] = {}
        for k, v in comp.items():
            try:
                if torch.is_tensor(v):
                    if v.numel() != 1:
                        continue
                    x = float(v.detach().cpu().item())
                else:
                    x = float(v)
                if np.isfinite(x):
                    out[k] = x
            except (TypeError, ValueError):
                continue
        return out

    def _finalize_train(self) -> None:
        if self.current_train_epoch is None or self.train_batches <= 0:
            return
        row: dict[str, float | int | str] = {
            'epoch': int(self.current_train_epoch),
            'split': 'train',
            'n_batches': int(self.train_batches),
        }
        for k, total in self.train_sums.items():
            row[k] = float(total / self.train_batches)
        self.rows[(int(self.current_train_epoch), 'train')] = row
        self.train_sums = defaultdict(float)
        self.train_batches = 0

    def observe_train(self, epoch: int, comp: dict) -> None:
        epoch = int(epoch)
        if self.current_train_epoch is None:
            self.current_train_epoch = epoch
        elif epoch != self.current_train_epoch:
            self._finalize_train()
            self.current_train_epoch = epoch

        vals = self._numeric_components(comp)
        for k, v in vals.items():
            self.train_sums[k] += v
        self.train_batches += 1

    def observe_validation(self, epoch: int, comp: dict, n_batches: int) -> None:
        # Validation is called after all train batches of the same epoch.
        self._finalize_train()
        self.current_train_epoch = int(epoch)
        row: dict[str, float | int | str] = {
            'epoch': int(epoch),
            'split': 'validation',
            'n_batches': int(n_batches),
        }
        row.update(self._numeric_components(comp))
        self.rows[(int(epoch), 'validation')] = row

    def finalize(self) -> None:
        self._finalize_train()

    def save_csv(self, path: Path | None = None) -> None:
        self.finalize()
        path = path or self.history_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [self.rows[k] for k in sorted(self.rows, key=lambda t: (t[0], t[1]))]
        if not rows:
            return
        fixed = ['epoch', 'split', 'n_batches']
        extras = sorted({k for r in rows for k in r.keys()} - set(fixed))
        with path.open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fixed + extras)
            w.writeheader()
            w.writerows(rows)


def _default_component_history_path(checkpoint: Path) -> Path:
    return checkpoint.with_name(f'{checkpoint.stem}_loss_components.csv')


def _load_component_history(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open('r', newline='', encoding='utf-8') as f:
        for raw in csv.DictReader(f):
            row: dict[str, object] = {
                'epoch': int(raw['epoch']),
                'split': raw['split'],
                'n_batches': int(float(raw.get('n_batches', 0) or 0)),
            }
            for k, v in raw.items():
                if k in {'epoch', 'split', 'n_batches'} or v in ('', None):
                    continue
                try:
                    row[k] = float(v)
                except ValueError:
                    row[k] = v
            rows.append(row)
    return rows


def _derived_component_row(row: dict, cfg: dict) -> dict[str, float]:
    """Convert raw logged terms into the quantities used in Sec. 4.2.2."""
    g = lambda k, d=0.0: float(row.get(k, d) or d)

    attrs_scale = float(cfg.get('attrs_scale', 0.01))
    spec_scale = float(cfg.get('spec_recon_scale', 0.30))
    align_scale = float(cfg.get('align_scale', 0.50))
    nce_scale = float(cfg.get('nce_scale', 0.50))
    cg_scale = float(cfg.get('cg_scale', 0.50))

    # Reconstruction branches exactly as they enter the implemented objective.
    loss_R_c = g('loss_topo_c') + attrs_scale * g('loss_attrs_c')
    loss_R_s_raw = g('loss_topo_s') + attrs_scale * g('loss_attrs_s')

    # Structural-only topology pieces. In the circuit branch, the repository's
    # logged loss_topo_c additionally contains beta*KL (and optionally L_phys).
    # These explicit terms are used for Fig. 39 so that it matches Eq. 4.13.
    topo_type_c = LAMBDA_TYPE * g('loss_t_c')
    topo_pos_c = LAMBDA_POSITION * g('loss_p_c')
    topo_edge_c = g('loss_e_c')
    topo_dir_c = g('loss_dir_c')
    topo_struct_c = topo_type_c + topo_pos_c + topo_edge_c + topo_dir_c

    topo_type_s = LAMBDA_TYPE * g('loss_t_s')
    topo_pos_s = LAMBDA_POSITION * g('loss_p_s')
    topo_edge_s = g('loss_e_s')
    topo_dir_s = g('loss_dir_s')
    topo_struct_s = topo_type_s + topo_pos_s + topo_edge_s + topo_dir_s

    beta = g('beta')
    L_KL = g('L_KL')
    L_C = g('L_C')

    out = {
        'loss_total_logged': g('loss_total'),
        'R_c': loss_R_c,
        'R_s_raw': loss_R_s_raw,
        'weighted_R_c': loss_R_c,
        'weighted_R_s': spec_scale * loss_R_s_raw,
        'weighted_align': align_scale * g('loss_align'),
        'weighted_NCE': nce_scale * g('loss_nce'),
        'weighted_CG': cg_scale * g('loss_cg'),
        'topology_c': g('loss_topo_c'),
        'weighted_attrs_c': attrs_scale * g('loss_attrs_c'),
        'topology_s': g('loss_topo_s'),
        'weighted_attrs_s': attrs_scale * g('loss_attrs_s'),
        'type_c': topo_type_c,
        'position_c': topo_pos_c,
        'edge_c': topo_edge_c,
        'direction_c': topo_dir_c,
        'topology_structural_c': topo_struct_c,
        'type_s': topo_type_s,
        'position_s': topo_pos_s,
        'edge_s': topo_edge_s,
        'direction_s': topo_dir_s,
        'topology_structural_s': topo_struct_s,
        'beta': beta,
        'circuit_KL_weighted': beta * g('loss_kl'),
        'align_KL_weighted': align_scale * beta * L_KL,
        'align_C_weighted': align_scale * L_C,
        'L_KL_raw': L_KL,
        'L_C_raw': L_C,
    }
    out['weighted_total_reconstructed'] = (
        out['weighted_R_c'] + out['weighted_R_s'] + out['weighted_align']
        + out['weighted_NCE'] + out['weighted_CG']
    )
    out['total_reconstruction_residual'] = (
        out['weighted_total_reconstructed'] - out['loss_total_logged']
    )
    return out


def _auto_log_scale(ax, series: list[np.ndarray], threshold: float = 80.0) -> None:
    vals = np.concatenate([np.asarray(x, dtype=float).ravel() for x in series])
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size and float(vals.max() / vals.min()) >= threshold:
        ax.set_yscale('log')


def _e90_epoch(epochs: np.ndarray, values: np.ndarray, smooth_window: int) -> float:
    """Epoch at which 90% of the net first-to-final change is first reached."""
    if len(values) < 2:
        return float('nan')
    sm = _smooth(values, smooth_window)
    start = float(sm[0])
    final = float(np.median(sm[max(0, len(sm) - max(3, len(sm)//10)):]))
    if not np.isfinite(start) or not np.isfinite(final) or abs(start-final) < 1e-15:
        return float(epochs[0])
    target = start + 0.90 * (final - start)
    if final < start:
        idx = np.flatnonzero(sm <= target)
    else:
        idx = np.flatnonzero(sm >= target)
    return float(epochs[idx[0]]) if len(idx) else float('nan')


def analyse_loss_components(
    ckpt: dict,
    history_path: Path,
    out_dir: Path,
    smooth_window: int,
    preferred_split: str = 'validation',
) -> dict:
    """Generate Figs. 37-39 and quantitative component summaries."""
    rows = _load_component_history(history_path)
    available_splits = {str(r['split']) for r in rows}
    split = preferred_split if preferred_split in available_splits else 'train'
    selected = sorted(
        [r for r in rows if r['split'] == split],
        key=lambda r: int(r['epoch']),
    )
    if not selected:
        raise RuntimeError(f'No {split} component history found in {history_path}')

    cfg = ckpt.get('config', {})
    epochs = np.asarray([int(r['epoch']) for r in selected], dtype=int)
    derived = [_derived_component_row(r, cfg) for r in selected]

    # Save a fully derived history so every plotted number can be inspected.
    derived_keys = list(derived[0].keys())
    with (out_dir / f'loss_components_{split}_derived.csv').open(
        'w', newline='', encoding='utf-8'
    ) as f:
        w = csv.writer(f)
        w.writerow(['epoch', 'split'] + derived_keys)
        for e, d in zip(epochs, derived):
            w.writerow([int(e), split] + [d[k] for k in derived_keys])

    def arr(key: str) -> np.ndarray:
        return np.asarray([d[key] for d in derived], dtype=float)

    best_epoch = int(_validation_epochs(
        len(ckpt.get('train_losses', [])),
        len(ckpt.get('val_losses', [])),
        int(cfg.get('val_every', 1)),
    )[int(np.argmin(np.asarray(ckpt.get('val_losses', [0.0]), dtype=float)))])
    best_idx = int(np.argmin(np.abs(epochs - best_epoch)))
    final_idx = len(epochs) - 1

    beta_max = float(cfg.get('beta_max', 0.0))
    beta_arr = arr('beta')
    warm_candidates = np.flatnonzero(beta_arr >= 0.99 * beta_max) if beta_max > 0 else np.array([], dtype=int)
    warmup_epoch = int(epochs[warm_candidates[0]]) if len(warm_candidates) else None

    # --- Figure 37: weighted terms that actually enter L_tot -----------------
    weighted_keys = [
        ('weighted_R_c', r'$\mathcal{L}_{R}^{(c)}$'),
        ('weighted_R_s', r'$\lambda_s\mathcal{L}_{R}^{(s)}$'),
        ('weighted_align', r'$\lambda_{\mathrm{align}}\mathcal{L}_{\mathrm{align}}$'),
        ('weighted_NCE', r'$\lambda_{\mathrm{NCE}}\mathcal{L}_{\mathrm{NCE}}$'),
        ('weighted_CG', r'$\lambda_{\mathrm{CG}}\mathcal{L}_{\mathrm{CG}}$'),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    plotted = []
    for key, label in weighted_keys:
        y = arr(key)
        plotted.append(y)
        ax.plot(epochs, y, linewidth=0.65, alpha=0.22)
        ax.plot(epochs, _smooth(y, smooth_window), linewidth=1.8, label=label)
    if warmup_epoch is not None:
        ax.axvline(warmup_epoch, linestyle='--', linewidth=1.0,
                   label=f'KL warm-up end (epoch {warmup_epoch})')
    ax.axvline(best_epoch, linestyle=':', linewidth=1.0,
               label=f'Best validation checkpoint (epoch {best_epoch})')
    _auto_log_scale(ax, plotted)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Weighted loss contribution')
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / 'figure37_weighted_loss_components.pdf', bbox_inches='tight')
    plt.close(fig)

    # Supplementary split of alignment into beta L_KL and L_C. This directly
    # supports the timescale discussion in the final paragraph of Sec. 4.2.2.
    fig, ax = plt.subplots(figsize=(7.6, 4.7))
    latent_keys = [
        ('align_KL_weighted', r'$\lambda_{\mathrm{align}}\,\beta\mathcal{L}_{\mathrm{KL}}$'),
        ('align_C_weighted', r'$\lambda_{\mathrm{align}}\mathcal{L}_{C}$'),
        ('weighted_NCE', r'$\lambda_{\mathrm{NCE}}\mathcal{L}_{\mathrm{NCE}}$'),
        ('weighted_CG', r'$\lambda_{\mathrm{CG}}\mathcal{L}_{\mathrm{CG}}$'),
    ]
    plotted = []
    for key, label in latent_keys:
        y = arr(key)
        plotted.append(y)
        ax.plot(epochs, y, linewidth=0.65, alpha=0.22)
        ax.plot(epochs, _smooth(y, smooth_window), linewidth=1.8, label=label)
    if warmup_epoch is not None:
        ax.axvline(warmup_epoch, linestyle='--', linewidth=1.0,
                   label=f'KL warm-up end (epoch {warmup_epoch})')
    ax.axvline(best_epoch, linestyle=':', linewidth=1.0,
               label=f'Best validation checkpoint (epoch {best_epoch})')
    _auto_log_scale(ax, plotted)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Weighted latent-space contribution')
    ax.grid(True, alpha=0.2)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / 'figure37b_latent_alignment_components.pdf', bbox_inches='tight')
    plt.close(fig)

    # --- Figure 38: topology vs continuous attributes, branch by branch ------
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), sharex=True)
    recon_panels = [
        (axes[0], 'Circuit branch', 'topology_c', 'weighted_attrs_c',
         r'$\mathcal{L}_{\mathrm{topo}}^{(c)}$',
         r'$\lambda_{\mathrm{attr}}\mathcal{L}_{\mathrm{attr}}^{(c)}$'),
        (axes[1], 'Hamiltonian branch', 'topology_s', 'weighted_attrs_s',
         r'$\mathcal{L}_{\mathrm{topo}}^{(s)}$',
         r'$\lambda_{\mathrm{attr}}\mathcal{L}_{\mathrm{attr}}^{(s)}$'),
    ]
    for ax, title, k1, k2, l1, l2 in recon_panels:
        y1, y2 = arr(k1), arr(k2)
        ax.plot(epochs, y1, linewidth=0.65, alpha=0.22)
        ax.plot(epochs, y2, linewidth=0.65, alpha=0.22)
        ax.plot(epochs, _smooth(y1, smooth_window), linewidth=1.8, label=l1)
        ax.plot(epochs, _smooth(y2, smooth_window), linewidth=1.8, label=l2)
        ax.axvline(best_epoch, linestyle=':', linewidth=0.9)
        _auto_log_scale(ax, [y1, y2])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Epoch')
        ax.grid(True, alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel('Reconstruction loss contribution')
    fig.tight_layout()
    fig.savefig(out_dir / 'figure38_reconstruction_loss_decomposition.pdf', bbox_inches='tight')
    plt.close(fig)

    # --- Figure 39: weighted structural topology pieces ----------------------
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.0), sharex=True)
    for ax, suffix, title in [
        (axes[0], 'c', 'Circuit branch'),
        (axes[1], 's', 'Hamiltonian branch'),
    ]:
        keys = [
            (f'type_{suffix}', r'$\lambda_t\mathcal{L}_t$'),
            (f'position_{suffix}', r'$\lambda_p\mathcal{L}_p$'),
            (f'edge_{suffix}', r'$\mathcal{L}_e$'),
            (f'direction_{suffix}', r'$\mathcal{L}_d$'),
        ]
        for key, label in keys:
            y = arr(key)
            ax.plot(epochs, y, linewidth=0.65, alpha=0.20)
            ax.plot(epochs, _smooth(y, smooth_window), linewidth=1.7, label=label)
        ax.axvline(best_epoch, linestyle=':', linewidth=0.9)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Epoch')
        ax.grid(True, alpha=0.2)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel('Weighted topological loss contribution')
    fig.tight_layout()
    fig.savefig(out_dir / 'figure39_topology_loss_components.pdf', bbox_inches='tight')
    plt.close(fig)

    # --- Quantitative summaries used to write the thesis ---------------------
    total_best = arr('loss_total_logged')[best_idx]
    summary_rows: list[dict[str, object]] = []
    for key, label in weighted_keys:
        y = arr(key)
        start = float(y[0])
        best = float(y[best_idx])
        final = float(y[final_idx])
        summary_rows.append({
            'component': key,
            'latex_label': label,
            'split': split,
            'start_epoch': int(epochs[0]),
            'start_value': start,
            'best_checkpoint_epoch': int(epochs[best_idx]),
            'best_checkpoint_value': best,
            'final_epoch': int(epochs[final_idx]),
            'final_value': final,
            'reduction_start_to_best_pct': (100.0*(1.0-best/start) if start != 0 else float('nan')),
            'share_of_total_at_best_pct': (100.0*best/total_best if total_best != 0 else float('nan')),
            'epoch_90pct_net_change': _e90_epoch(epochs, y, smooth_window),
        })

    with (out_dir / 'loss_component_summary.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    convergence_rows: list[dict[str, object]] = []
    for key in [
        'topology_c', 'weighted_attrs_c', 'topology_s', 'weighted_attrs_s',
        'type_c', 'position_c', 'edge_c', 'direction_c',
        'type_s', 'position_s', 'edge_s', 'direction_s',
        'align_KL_weighted', 'align_C_weighted', 'weighted_NCE', 'weighted_CG',
    ]:
        y = arr(key)
        convergence_rows.append({
            'component': key,
            'epoch_90pct_net_change': _e90_epoch(epochs, y, smooth_window),
            'start_value': float(y[0]),
            'best_checkpoint_value': float(y[best_idx]),
            'final_value': float(y[final_idx]),
        })
    with (out_dir / 'loss_component_convergence.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(convergence_rows[0].keys()))
        w.writeheader()
        w.writerows(convergence_rows)

    # Dominant topology contribution at the selected checkpoint.
    topo_c_best = {
        'type': arr('type_c')[best_idx],
        'position': arr('position_c')[best_idx],
        'edge': arr('edge_c')[best_idx],
        'direction': arr('direction_c')[best_idx],
    }
    topo_s_best = {
        'type': arr('type_s')[best_idx],
        'position': arr('position_s')[best_idx],
        'edge': arr('edge_s')[best_idx],
        'direction': arr('direction_s')[best_idx],
    }
    dom_c = max(topo_c_best, key=topo_c_best.get)
    dom_s = max(topo_s_best, key=topo_s_best.get)

    residual = np.max(np.abs(arr('total_reconstruction_residual')))
    report = [
        'cQED-Gen Sec. 4.2.2 - Loss Components Evolution',
        '=' * 62,
        f'History source: {history_path}',
        f'Plotted split: {split}',
        f'Best validation checkpoint: epoch {best_epoch}',
        f'Nearest component-history epoch: {int(epochs[best_idx])}',
        f'KL warm-up end from logged beta: {warmup_epoch}',
        f'Max |reconstructed total - logged total|: {residual:.3e}',
        '',
        'WEIGHTED CONTRIBUTIONS AT THE SELECTED CHECKPOINT',
    ]
    for item in summary_rows:
        report.append(
            f"  {item['component']:<18s} value={item['best_checkpoint_value']:.6g}  "
            f"share={item['share_of_total_at_best_pct']:.2f}%  "
            f"e90={item['epoch_90pct_net_change']}"
        )
    report.extend([
        '',
        f'Dominant circuit-branch topology term at best: {dom_c} ({topo_c_best[dom_c]:.6g})',
        f'Dominant Hamiltonian-branch topology term at best: {dom_s} ({topo_s_best[dom_s]:.6g})',
        f"Circuit reconstruction at best: {arr('R_c')[best_idx]:.6g}",
        f"Hamiltonian reconstruction (raw, before lambda_s) at best: {arr('R_s_raw')[best_idx]:.6g}",
        f"Hamiltonian/circuit reconstruction ratio at best: "
        f"{arr('R_s_raw')[best_idx]/arr('R_c')[best_idx] if arr('R_c')[best_idx] != 0 else float('nan'):.4f}",
        '',
        'Implementation note: loss_topo_c in the repository includes the circuit-branch beta*KL term.',
        'Figure 39 is therefore built explicitly from lambda_t*L_t + lambda_p*L_p + L_e + L_d,',
        'matching the structural decomposition written in Eq. 4.13 of the thesis.',
    ])
    (out_dir / 'loss_components_analysis_summary.txt').write_text('\n'.join(report) + '\n', encoding='utf-8')

    return {
        'split': split,
        'best_epoch': best_epoch,
        'history_best_epoch': int(epochs[best_idx]),
        'warmup_epoch': warmup_epoch,
        'dominant_topology_c': dom_c,
        'dominant_topology_s': dom_s,
        'max_total_residual': float(residual),
    }


def run_training_with_component_logging(
    training_argv: list[str],
    resume_checkpoint: Path | None,
) -> tuple[Path, Path]:
    """Run the repository trainer while persisting epoch-wise component means.

    All unknown CLI options are forwarded to ``train_vae.py``. Example:

        python analysis_4_2_1_training_validation_loss.py --train-with-analysis \\
            --epochs 500 --val_every 1 --save_path best_vae_global.pt

    This performs one normal training run and records the component history needed
    for Figs. 37-39. No second training is required.
    """
    import train_vae as tv

    saved_argv = list(sys.argv)
    try:
        sys.argv = ['train_vae.py'] + list(training_argv)
        targs = tv._parse_args()
    finally:
        sys.argv = saved_argv

    if getattr(targs, 'eval_only', False):
        raise ValueError('--eval_only is not compatible with --train-with-analysis.')
    if getattr(targs, 'num_workers', None) not in (None, 0):
        raise ValueError('num_workers must be 0 for this repository training run.')

    overrides = tv._args_to_overrides(targs)
    resume = resume_checkpoint
    if resume is None and getattr(targs, 'checkpoint', None):
        resume = Path(targs.checkpoint)

    if resume is not None:
        resume = resume.resolve()
        existing = _torch_load(resume)
        cfg_for_paths = {**existing.get('config', {}), **overrides}
    else:
        cfg_for_paths = {**tv.DEFAULT_CONFIG, **overrides}

    save_path = Path(cfg_for_paths.get('save_path', 'best_vae.pt'))
    if not save_path.is_absolute():
        save_path = Path.cwd() / save_path
    history_path = _default_component_history_path(save_path)
    recorder = TrainingComponentRecorder(history_path if history_path.is_file() else None)
    recorder.history_path = history_path

    orig_train_step = tv.train_step
    orig_eval_loop = tv.eval_loop
    orig_run_loop = tv._run_training_loop
    orig_save_checkpoint = tv.save_checkpoint

    def train_step_hook(*args, **kwargs):
        loss, comp = orig_train_step(*args, **kwargs)
        model = args[0] if args else kwargs['model']
        epoch = int(getattr(model.param_decoder, 'current_epoch', 0))
        recorder.observe_train(epoch, comp)
        return loss, comp

    def eval_loop_hook(model, loader, *args, **kwargs):
        loss, comp = orig_eval_loop(model, loader, *args, **kwargs)
        if recorder.val_loader_id is not None and id(loader) == recorder.val_loader_id:
            epoch = int(getattr(model.param_decoder, 'current_epoch', 0))
            recorder.observe_validation(epoch, comp, len(loader))
        return loss, comp

    def run_loop_hook(model, train_loader, val_loader, test_loader, *args, **kwargs):
        recorder.val_loader_id = id(val_loader)
        recorder.test_loader_id = id(test_loader)
        try:
            return orig_run_loop(model, train_loader, val_loader, test_loader, *args, **kwargs)
        finally:
            recorder.finalize()
            recorder.save_csv(history_path)

    def save_checkpoint_hook(*args, **kwargs):
        result = orig_save_checkpoint(*args, **kwargs)
        recorder.save_csv(history_path)
        return result

    tv.train_step = train_step_hook
    tv.eval_loop = eval_loop_hook
    tv._run_training_loop = run_loop_hook
    tv.save_checkpoint = save_checkpoint_hook

    try:
        if resume is not None:
            tv.resume_train(str(resume), config_override=overrides)
        else:
            tv.train(overrides)
    finally:
        tv.train_step = orig_train_step
        tv.eval_loop = orig_eval_loop
        tv._run_training_loop = orig_run_loop
        tv.save_checkpoint = orig_save_checkpoint
        recorder.finalize()
        recorder.save_csv(history_path)

    if not save_path.is_file():
        raise FileNotFoundError(f'Training finished but checkpoint was not found: {save_path}')
    return save_path, history_path


# -----------------------------------------------------------------------------
# Report
# -----------------------------------------------------------------------------

def write_text_report(out_dir: Path, global_stats: dict, family_stats: dict | None) -> None:
    lines = []
    lines.append("cQED-Gen Sec. 4.2.1 - Training/Validation Loss Analysis")
    lines.append("=" * 66)
    lines.append("")
    lines.append("GLOBAL TRAINING HISTORY")
    lines.append(f"  Final recorded epoch: {global_stats['final_epoch']}")
    lines.append(f"  Best validation checkpoint: epoch {global_stats['best_epoch']}")
    lines.append(f"  Best validation loss: {_fmt(global_stats['best_val'])}")
    lines.append(f"  Training loss epoch 1: {_fmt(global_stats['train_1'])}")
    lines.append(f"  Training loss epoch 50: {_fmt(global_stats['train_50'])}")
    lines.append(f"  Training loss final: {_fmt(global_stats['train_final'])}")
    lines.append(f"  Training loss at best validation epoch: {_fmt(global_stats['train_at_best'])}")
    lines.append(f"  Signed train-validation gap at best: {_fmt(global_stats['gap_best'])}")
    lines.append(
        f"  Training reduction epoch 1 -> 50: {global_stats['train_reduction_1_50']:.2f}%"
    )
    lines.append(
        f"  Training reduction epoch 1 -> final: {global_stats['train_reduction_1_final']:.2f}%"
    )
    lines.append(
        f"  Fraction of total training reduction already reached by epoch 50: "
        f"{global_stats['frac_total_by_50']:.2f}%"
    )
    lines.append(
        f"  Validation reduction epoch 1 -> best: {global_stats['val_reduction_1_best']:.2f}%"
    )
    lines.append("")

    if family_stats is not None:
        lines.append("VALIDATION LOSS BY CIRCUIT FAMILY (SELECTED CHECKPOINT)")
        lines.append(
            f"  Global mixed-validation loss re-evaluated: "
            f"{family_stats['global_validation_loss_reevaluated']:.6f}"
        )
        for r in family_stats["rows"]:
            one_line_name = r["display_name"].replace("\n", " ")
            lines.append(
                f"  {one_line_name:24s}  loss={r['mean_validation_loss']:.6f}  "
                f"SEM={r['sem_across_batches']:.6f}  macro-nodes={r['n_macro_nodes']}"
            )
        lines.append("")
        lines.append(
            "Note: family-specific losses are evaluated in family-homogeneous validation "
            "batches. The global dashed-line reference is re-evaluated on the original "
            "mixed validation partition. Because the total objective includes batch-level "
            "terms (notably InfoNCE and batch-dependent class weights), the family values "
            "should be interpreted as family-conditioned validation objectives rather than "
            "an additive decomposition of the global loss."
        )

    (out_dir / "analysis_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(
        description=(
            'Generate Sec. 4.2.1-4.2.2 training analyses, or launch one '
            'instrumented training run and produce all loss figures automatically.'
        )
    )
    p.add_argument(
        '--checkpoint',
        type=Path,
        default=Path('best_vae_global.pt'),
        help='Production checkpoint for post-hoc analysis (default: best_vae_global.pt)',
    )
    p.add_argument(
        '--output-dir',
        type=Path,
        default=Path('analysis_results/4_2_training_losses'),
        help='Output directory',
    )
    p.add_argument(
        '--smooth-window',
        type=int,
        default=9,
        help='Moving-average window for loss figures (default: 9 epochs)',
    )
    p.add_argument(
        '--device',
        choices=['auto', 'cpu', 'cuda'],
        default='auto',
        help='Device for per-family checkpoint evaluation',
    )
    p.add_argument(
        '--eval-batch-size',
        type=int,
        default=None,
        help='Override validation evaluation batch size (default: checkpoint batch size)',
    )
    p.add_argument(
        '--history-only',
        action='store_true',
        help='Skip the expensive per-family checkpoint evaluation (Figure 36)',
    )
    p.add_argument(
        '--component-history',
        type=Path,
        default=None,
        help=(
            'Epoch-wise component CSV. By default the script looks for '
            '<checkpoint_stem>_loss_components.csv beside the checkpoint.'
        ),
    )
    p.add_argument(
        '--component-split',
        choices=['validation', 'train'],
        default='validation',
        help='Split used for Figs. 37-39 (default: validation)',
    )
    p.add_argument(
        '--train-with-analysis',
        action='store_true',
        help=(
            'Launch train_vae.py through lightweight logging hooks, then produce '
            'Figs. 35-39 from that single run. Unknown CLI options are forwarded '
            'to train_vae.py.'
        ),
    )
    p.add_argument(
        '--resume-checkpoint',
        type=Path,
        default=None,
        help='Resume checkpoint when using --train-with-analysis.',
    )
    return p.parse_known_args()


def _resolve_device(name: str) -> torch.device:
    if name == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('--device cuda requested but CUDA is not available.')
        return torch.device('cuda')
    if name == 'cpu':
        return torch.device('cpu')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main() -> None:
    args, training_argv = parse_args()

    repo_root = Path.cwd().resolve()
    src_root = repo_root / 'src'
    if src_root.is_dir():
        sys.path.insert(0, str(src_root))
    sys.path.insert(0, str(repo_root))

    out_dir = args.output_dir
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optional one-run mode: train normally while recording all component means.
    if args.train_with_analysis:
        print('Running one instrumented cQED-Gen training run...')
        checkpoint, component_history = run_training_with_component_logging(
            training_argv=training_argv,
            resume_checkpoint=(
                args.resume_checkpoint if args.resume_checkpoint is None
                else (args.resume_checkpoint if args.resume_checkpoint.is_absolute()
                      else repo_root / args.resume_checkpoint)
            ),
        )
        print(f'\nTraining checkpoint : {checkpoint}')
        print(f'Component history   : {component_history}')
    else:
        if training_argv:
            raise ValueError(
                'Unknown arguments were supplied in post-hoc mode: '
                + ' '.join(training_argv)
                + '. Use --train-with-analysis if these are train_vae.py options.'
            )
        checkpoint = args.checkpoint
        if not checkpoint.is_absolute():
            checkpoint = repo_root / checkpoint
        component_history = args.component_history
        if component_history is not None and not component_history.is_absolute():
            component_history = repo_root / component_history
        if component_history is None:
            component_history = _default_component_history_path(checkpoint)

    if not checkpoint.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')

    print(f'\nRepository root : {repo_root}')
    print(f'Checkpoint      : {checkpoint}')
    print(f'Output directory: {out_dir}')

    ckpt = _torch_load(checkpoint)

    print('\n[1/4] Global training/validation history (Figure 35)...')
    global_stats = analyse_global_history(
        ckpt=ckpt,
        out_dir=out_dir,
        smooth_window=max(1, int(args.smooth_window)),
    )
    print(
        f"  best validation checkpoint = epoch {global_stats['best_epoch']} "
        f"(loss={global_stats['best_val']:.6f})"
    )
    print(
        f"  train loss: epoch 1={global_stats['train_1']:.6f}, "
        f"epoch 50={global_stats['train_50']:.6f}, "
        f"final={global_stats['train_final']:.6f}"
    )
    print(
        f"  total training-loss reduction = "
        f"{global_stats['train_reduction_1_final']:.2f}%"
    )

    family_stats = None
    if not args.history_only:
        device = _resolve_device(args.device)
        print(f'\n[2/4] Validation loss by circuit family on {device} (Figure 36)...')
        family_stats = analyse_validation_per_family(
            ckpt=ckpt,
            out_dir=out_dir,
            device=device,
            eval_batch_size=args.eval_batch_size,
        )
        print(
            '  global validation reference = '
            f"{family_stats['global_validation_loss_reevaluated']:.6f}"
        )
        for r in family_stats['rows']:
            print(
                f"  {r['dataset_name']:<46s} "
                f"{r['mean_validation_loss']:.6f} +/- {r['sem_across_batches']:.6f} (SEM)"
            )
    else:
        print('\n[2/4] Figure 36 skipped (--history-only).')

    component_stats = None
    if component_history.is_file():
        print(f'\n[3/4] Loss-component evolution from {component_history.name} (Figures 37-39)...')
        component_stats = analyse_loss_components(
            ckpt=ckpt,
            history_path=component_history,
            out_dir=out_dir,
            smooth_window=max(1, int(args.smooth_window)),
            preferred_split=args.component_split,
        )
        print(f"  plotted split             = {component_stats['split']}")
        print(f"  KL warm-up end            = {component_stats['warmup_epoch']}")
        print(f"  dominant topology (G->G)  = {component_stats['dominant_topology_c']}")
        print(f"  dominant topology (H->G)  = {component_stats['dominant_topology_s']}")
    else:
        print('\n[3/4] Figures 37-39 not generated.')
        print(
            '  This checkpoint predates epoch-wise component logging and contains only '
            'train_losses/val_losses. True component evolution cannot be reconstructed '
            'from the final weights alone.'
        )
        print(
            '  For the next/final run use this same file with --train-with-analysis; '
            'it records the component history during that one training run.'
        )

    print('\n[4/4] Writing text summary...')
    write_text_report(out_dir, global_stats, family_stats)

    print('\nGenerated files:')
    for p in sorted(out_dir.iterdir()):
        print(f'  - {p.name}')


if __name__ == '__main__':
    main()
