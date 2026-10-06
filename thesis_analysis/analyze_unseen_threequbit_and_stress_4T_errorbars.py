#!/usr/bin/env python3
"""
Unseen-circuit + Hamiltonian stress-test analysis for cQED-Gen.

This script is intentionally SPECIFICATION-BRANCH ONLY:

    Hamiltonian/specifications -> SpecEncoder -> z_s -> topology decoder
                               -> ParamDecoder -> physical circuit -> QuLTRA

The circuit encoder is never used.

Experiments
===========
1) Held-out Three_qubit_capacitive_line (INCLUDE_TRAIN=False)
   - sample up to --n-unseen target Hamiltonians from the held-out dataset;
   - direct one-shot inference from the specification branch;
   - compare the QuLTRA Hamiltonian of the generated circuit with the target,
     independently of whether the generated topology matches the reference;
   - repeat with CEM latent-space optimization. CEM may cross decoder decision
     boundaries, so both topology and continuous circuit parameters can change.

   The target uses all observables available for the held-out dataset:
       f_1, f_2, f_3,
       chi_11, chi_22, chi_33, chi_12, chi_13, chi_23.

   Predicted modes are aligned to target modes by minimum frequency distance,
   then the same mapping is used for chi. This is crucial when a different
   topology produces a different raw QuLTRA mode ordering.

2) Random/plausible Hamiltonian stress test
   Targets are constructed from empirical training-distribution Hamiltonians,
   slightly jittered and clipped to the 5-95% empirical interval of each slot.
   This avoids arbitrary out-of-range values while still producing new targets.

   Frequency-only stages:
       F1: f_1                       + require >=1 transmon
       F2: f_1,f_2                   + require >=2 transmons, >=1 coupler
       F3: f_1,f_2,f_3               + require >=3 transmons, >=2 couplers
       F4: f_1,f_2,f_3               + require >=4 transmons, >=3 couplers
           (the fourth transmon is deliberately left without its own requested
            modal frequency: this is an extra structural/OOD stress condition)

   Frequency + chi stages:
       F1C: f_1, chi_11
       F2C: f_1,f_2, chi_11,chi_22,chi_12
       F3C: f_1,f_2,f_3, chi_11,chi_22,chi_33,chi_12,chi_23
       F4C: same three-mode Hamiltonian request as F3C
            + require >=4 transmons, >=3 couplers

   Stress-test SUCCESS is deliberately frequency-only:
       max_i |f_i^pred - f_i^target| / |f_i^target| < 10%
   (configurable with --frequency-epsilon-pct).

   When chi values are supplied they participate in conditioning and in the CEM
   objective, but they do NOT change the stress-test success definition.

Default runtime is deliberately modest: 20 held-out targets and 5 targets per
stress case, with CEM population 12 x 6 iterations. The number of target
Hamiltonians in each study is hard-capped at 100.

Run from the repository root, for example:

    python analyze_unseen_threequbit_and_stress.py \
        --ckpt best_vae_global.pt \
        --n-unseen 20 \
        --n-stress-per-case 5 \
        --out-dir results/unseen_and_stress

Useful faster smoke test:

    python analyze_unseen_threequbit_and_stress.py \
        --n-unseen 3 --n-stress-per-case 1 \
        --cem-population 6 --cem-iters 2

Outputs
=======
CSV:
  unseen_direct.csv
  unseen_cem.csv
  unseen_pairs.csv
  unseen_observable_summary.csv
  unseen_cem_iterations.csv
  stress_targets.csv
  stress_direct.csv
  stress_cem.csv
  stress_pairs.csv
  stress_summary.csv
  stress_cem_iterations.csv

PDF:
  unseen_observable_errors.pdf
  unseen_success_10pct.pdf
  unseen_cem_convergence.pdf
  stress_frequency_success_10pct.pdf
  stress_combined_single_figure.pdf

JSON:
  analysis_summary.json

Notes
=====
- The default checkpoint is best_vae_global.pt because an unseen topology may
  have a different parameter-vector layout. A global attribute-aware ParamScaler
  is required to map ParamDecoder outputs back to physical units.
- The decoder primitive constraints are semantic/minimum-count guidance, not an
  exact topology template. No linear-topology match is required for success.
- QuLTRA is the physical evaluator. If QuLTRA is unavailable, the script exits
  with a clear error rather than silently using a neural surrogate.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Repository discovery and imports
# -----------------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()


def find_repo_root() -> Path:
    candidates = [Path.cwd(), THIS_FILE.parent, *THIS_FILE.parents]
    for c in candidates:
        if (c / "src").is_dir() and (
            (c / "inference_hamiltonian_qultra_fixed.py").exists()
            or (c / "inference_hamiltonian_qultra.py").exists()
        ):
            return c.resolve()
    raise RuntimeError(
        "Could not find the cQED-Gen repository root. Run this script from the "
        "repository root or place it in the repository root."
    )


REPO_ROOT = find_repo_root()
SRC_ROOT = REPO_ROOT / "src"
for p in (str(REPO_ROOT), str(SRC_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(REPO_ROOT)

from circuit2graph import SubgType, SUBG_DEFS  # noqa: E402
from circuit2graph.definitions import MACRO_SIGNATURES  # noqa: E402
from data_loader.schema import (  # noqa: E402
    DATASETS,
    ROW_PARSERS,
    OBS_PARSERS,
    OBS_SLOTS,
    OBS_IDX,
    N_OBS_SLOTS,
    train_datasets,
)
from data_loader.datasets._base import is_header  # noqa: E402


UNSEEN_DATASET = "Three_qubit_capacitive_line"
DEFAULT_EPS_PCT = 10.0


def import_qinf():
    """Import the repository's mature QuLTRA inference helpers.

    The uploaded repository currently contains the *_fixed.py version while
    older scripts still refer to inference_hamiltonian_qultra.py. Support both.
    """
    candidates = [
        REPO_ROOT / "inference_hamiltonian_qultra_fixed.py",
        REPO_ROOT / "inference_hamiltonian_qultra.py",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise RuntimeError("No inference_hamiltonian_qultra(_fixed).py found.")
    spec = importlib.util.spec_from_file_location("cqed_qultra_inference_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# -----------------------------------------------------------------------------
# Small IO helpers
# -----------------------------------------------------------------------------


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def json_safe(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def percentile(values: list[float], q: float) -> float:
    a = np.asarray([v for v in values if math.isfinite(float(v))], dtype=float)
    return float(np.percentile(a, q)) if a.size else float("nan")


def wilson_interval(values: list[bool], z: float = 1.959963984540054) -> tuple[float, float, float]:
    """Wilson 95% interval for a Bernoulli success rate.

    Returns (p, lower, upper), all in [0, 1]. This is preferred to a normal
    approximation here because some stress-test cells may contain only a small
    number of targets and can have observed rates close to 0% or 100%.
    """
    if not values:
        return float("nan"), float("nan"), float("nan")
    a = np.asarray([bool(v) for v in values], dtype=float)
    n = int(a.size)
    p = float(a.mean())
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def bootstrap_median_ci(
    values: list[float],
    n_bootstrap: int = 4000,
    seed: int = 12345,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Target-level bootstrap interval for a median.

    NaN/Inf values are discarded, matching the median convention used by the
    existing result tables. The bootstrap is deterministic for a fixed seed.
    """
    a = np.asarray(
        [float(v) for v in values if v is not None and math.isfinite(float(v))],
        dtype=float,
    )
    if a.size == 0:
        return float("nan"), float("nan"), float("nan")
    med = float(np.median(a))
    if a.size == 1:
        return med, med, med

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(int(n_bootstrap), a.size))
    meds = np.median(a[idx], axis=1)
    lo = float(np.quantile(meds, alpha / 2.0))
    hi = float(np.quantile(meds, 1.0 - alpha / 2.0))
    return med, lo, hi


def asymmetric_errors(
    centers: list[float],
    lows: list[float],
    highs: list[float],
    scale: float = 1.0,
) -> np.ndarray:
    """Return matplotlib-compatible asymmetric y-error array."""
    c = np.asarray(centers, dtype=float) * scale
    lo = np.asarray(lows, dtype=float) * scale
    hi = np.asarray(highs, dtype=float) * scale
    return np.vstack([np.maximum(0.0, c - lo), np.maximum(0.0, hi - c)])


def fmt_seq(values: Any) -> str:
    try:
        arr = np.asarray(values, dtype=float).ravel()
    except Exception:
        return ""
    return ";".join(f"{x:.12g}" for x in arr)


# -----------------------------------------------------------------------------
# Dataset reading in RAW observable units
# -----------------------------------------------------------------------------


def raw_observable_rows(
    ds_name: str,
    max_rows: int,
    seed: int,
) -> list[dict[str, float]]:
    """Reservoir-sample raw observable rows without fitting a new scaler."""
    if ds_name not in DATASETS:
        raise KeyError(f"Unknown dataset {ds_name!r}. Available: {list(DATASETS)}")

    defn = DATASETS[ds_name]
    path = Path(defn.path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(path)

    row_parser = ROW_PARSERS[ds_name]
    obs_parser = OBS_PARSERS[ds_name]
    rng = random.Random(seed)
    reservoir: list[dict[str, float]] = []
    seen = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if is_header(line):
                continue
            attrs, obs_kw = row_parser(line)
            if attrs is None or obs_kw is None:
                continue
            vals, mask = obs_parser(**obs_kw)
            target = {
                slot: float(vals[i])
                for i, slot in enumerate(OBS_SLOTS)
                if i < len(mask) and float(mask[i]) > 0.5 and math.isfinite(float(vals[i]))
            }
            if not target:
                continue
            seen += 1
            if len(reservoir) < max_rows:
                reservoir.append(target)
            else:
                j = rng.randint(0, seen - 1)
                if j < max_rows:
                    reservoir[j] = target

    if not reservoir:
        raise RuntimeError(f"No observable rows parsed from {path}")
    rng.shuffle(reservoir)
    return reservoir


def unseen_targets(n: int, seed: int) -> list[dict[str, float]]:
    rows = raw_observable_rows(UNSEEN_DATASET, max_rows=max(n, 1), seed=seed)
    wanted = {
        "f_1", "f_2", "f_3",
        "chi_11", "chi_22", "chi_33",
        "chi_12", "chi_13", "chi_23",
    }
    out = []
    for row in rows:
        t = {k: v for k, v in row.items() if k in wanted}
        if all(k in t for k in ["f_1", "f_2", "f_3"]):
            out.append(t)
    return out[:n]


def build_training_observable_pool(max_per_dataset: int, seed: int) -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    for i, ds_name in enumerate(train_datasets().keys()):
        rows = raw_observable_rows(ds_name, max_rows=max_per_dataset, seed=seed + 101 * i)
        for row in rows:
            pool.append({"dataset": ds_name, "obs": row})
    return pool


# -----------------------------------------------------------------------------
# Scaling and target Hamiltonian representation
# -----------------------------------------------------------------------------


def get_global_scaler(scalers: dict[str, Any]):
    if "__global__" in scalers:
        return scalers["__global__"]
    raise RuntimeError(
        "This analysis requires the global attribute-aware scaler. "
        "Use best_vae_global.pt (recommended) or a checkpoint containing scalers['__global__']."
    )


def target_to_encoder_tensors(
    target: dict[str, float],
    obs_scaler,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    vals = np.zeros(N_OBS_SLOTS, dtype=np.float64)
    mask = np.zeros(N_OBS_SLOTS, dtype=np.float64)
    for slot, value in target.items():
        if slot not in OBS_IDX:
            continue
        j = OBS_IDX[slot]
        vals[j] = float(value)
        mask[j] = 1.0

    # Important for the requested stress test: missing chi slots stay masked.
    if hasattr(obs_scaler, "transform_row"):
        scaled = obs_scaler.transform_row(vals, mask)
    else:
        mean = np.asarray(obs_scaler.mean_, dtype=float)
        std = np.asarray(obs_scaler.std_, dtype=float)
        scaled = np.zeros_like(vals)
        active = mask > 0.5
        scaled[active] = (vals[active] - mean[active]) / np.maximum(std[active], 1e-12)

    v = torch.tensor(scaled, dtype=torch.float32, device=device).unsqueeze(0)
    m = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)
    return v, m


@torch.no_grad()
def encode_spec(model, target: dict[str, float], obs_scaler, device: torch.device) -> torch.Tensor:
    vals, mask = target_to_encoder_tensors(target, obs_scaler, device)
    z, _, _ = model.spec_encoder.encode(vals, mask)
    return z[0].detach()


def target_as_qultra_arrays(target: dict[str, float]) -> dict[str, Any]:
    f_ids = []
    for k in target:
        if k.startswith("f_"):
            try:
                f_ids.append(int(k.split("_", 1)[1]))
            except Exception:
                pass
    n = max(f_ids) if f_ids else 0
    freqs = np.full(n, np.nan, dtype=float)
    for i in range(1, n + 1):
        if f"f_{i}" in target:
            freqs[i - 1] = float(target[f"f_{i}"])

    chi = np.full((n, n), np.nan, dtype=float)
    for slot, value in target.items():
        if not slot.startswith("chi_"):
            continue
        ij = slot.split("_", 1)[1]
        if len(ij) != 2 or not ij.isdigit():
            continue
        i, j = int(ij[0]) - 1, int(ij[1]) - 1
        if 0 <= i < n and 0 <= j < n:
            chi[i, j] = float(value)
            chi[j, i] = float(value)

    return {
        "ok": True,
        "frequencies": freqs,
        "chi": chi,
        "kappa": np.full(n, np.nan, dtype=float),
        "n_expected": n,
    }


# -----------------------------------------------------------------------------
# Graph -> physical parameters -> QuLTRA
# -----------------------------------------------------------------------------


def flat_attr_names(g: SimpleNamespace) -> list[str]:
    out: list[str] = []
    for t in getattr(g, "node_types", []):
        st = SubgType(int(t))
        out.extend(a for a in SUBG_DEFS[st].attrs if a != "dir")
    return out


def flat_scaled_attrs(g: SimpleNamespace) -> np.ndarray:
    out: list[float] = []
    for i, t in enumerate(getattr(g, "node_types", [])):
        st = SubgType(int(t))
        attrs = g.attrs[i] if hasattr(g, "attrs") and i < len(g.attrs) else {}
        for name in SUBG_DEFS[st].attrs:
            if name != "dir":
                out.append(float(attrs.get(name, float("nan"))))
    return np.asarray(out, dtype=float)


def graph_to_physical(g: SimpleNamespace, param_scaler) -> SimpleNamespace:
    names = flat_attr_names(g)
    scaled = flat_scaled_attrs(g)
    if len(names):
        physical = param_scaler.inverse_transform(scaled, names)
    else:
        physical = np.asarray([], dtype=float)

    out = SimpleNamespace(
        node_types=list(getattr(g, "node_types", [])),
        edges=[tuple(e) for e in getattr(g, "edges", [])],
        direction=list(getattr(g, "direction", [0.0] * len(getattr(g, "node_types", [])))),
        attrs=[],
    )
    cursor = 0
    for i, t in enumerate(out.node_types):
        st = SubgType(int(t))
        old = g.attrs[i] if hasattr(g, "attrs") and i < len(g.attrs) else {}
        d: dict[str, float] = {}
        for name in SUBG_DEFS[st].attrs:
            if name == "dir":
                d[name] = float(old.get("dir", out.direction[i] if i < len(out.direction) else 0.0))
            else:
                d[name] = float(physical[cursor])
                cursor += 1
        out.attrs.append(d)
    return out


def primitive_counts(g: SimpleNamespace) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for t in getattr(g, "node_types", []):
        try:
            counts.update(MACRO_SIGNATURES[SubgType(int(t))])
        except Exception:
            pass
    return dict(counts)


def requirements_met(g: SimpleNamespace, required: dict[str, int]) -> bool:
    c = primitive_counts(g)
    return all(int(c.get(k, 0)) >= int(v) for k, v in required.items())


def topology_signature(g: SimpleNamespace) -> str:
    types = tuple(int(x) for x in getattr(g, "node_types", []))
    edges = tuple(sorted(tuple(sorted((int(u), int(v)))) for u, v in getattr(g, "edges", [])))
    dirs = tuple(int(np.sign(float(x))) for x in getattr(g, "direction", []))
    return json.dumps({"types": types, "edges": edges, "dir": dirs}, separators=(",", ":"))


def graph_summary(g: SimpleNamespace) -> dict[str, Any]:
    counts = primitive_counts(g)
    return {
        "n_macro_nodes": len(getattr(g, "node_types", [])),
        "n_macro_edges": len(getattr(g, "edges", [])),
        "macro_node_types": ";".join(SubgType(int(t)).name for t in getattr(g, "node_types", [])),
        "macro_edges": ";".join(f"{int(u)}-{int(v)}" for u, v in getattr(g, "edges", [])),
        "primitive_transmon": int(counts.get("transmon", 0)),
        "primitive_resonator": int(counts.get("resonator", 0)),
        "primitive_coupler": int(counts.get("coupler", 0)),
        "primitive_feedline": int(counts.get("feedline", 0)),
        "primitive_inductor": int(counts.get("inductor", 0)),
        "topology_signature": topology_signature(g),
    }


# -----------------------------------------------------------------------------
# Hamiltonian comparison after mode alignment
# -----------------------------------------------------------------------------


def compare_target_to_sim(
    qinf,
    target: dict[str, float],
    sim: dict[str, Any],
    eps_pct: float,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "freq_success_10pct": False,
        "full_requested_success_10pct": False,
        "freq_mean_rel_err_pct": float("nan"),
        "freq_max_rel_err_pct": float("nan"),
        "chi_mean_rel_err_pct": float("nan"),
        "chi_max_rel_err_pct": float("nan"),
        "chi_diag_mean_rel_err_pct": float("nan"),
        "chi_offdiag_mean_rel_err_pct": float("nan"),
        "requested_chi_count": sum(1 for k in target if k.startswith("chi_")),
    }
    if not sim.get("ok"):
        return out

    true_obs = target_as_qultra_arrays(target)
    aligned = qinf._align_pred_modes_to_true(true_obs, sim)
    out["target_freqs"] = fmt_seq(true_obs["frequencies"])
    out["pred_freqs_raw"] = fmt_seq(sim.get("frequencies", []))
    out["pred_freqs_aligned"] = fmt_seq(aligned.get("frequencies", []))
    out["mode_mapping_json"] = json.dumps(
        {str(k): int(v) for k, v in aligned.get("mode_mapping", {}).items()}, sort_keys=True
    )

    f_errors: list[float] = []
    chi_errors: list[float] = []
    chi_diag: list[float] = []
    chi_off: list[float] = []

    af = np.asarray(aligned.get("frequencies", []), dtype=float)
    for slot, target_value in sorted(target.items()):
        if not slot.startswith("f_"):
            continue
        i = int(slot.split("_", 1)[1]) - 1
        pred = float(af[i]) if 0 <= i < len(af) else float("nan")
        rel = (
            100.0 * abs(pred - float(target_value)) / max(abs(float(target_value)), 1e-12)
            if math.isfinite(pred) else float("nan")
        )
        out[f"target_{slot}"] = float(target_value)
        out[f"pred_{slot}"] = pred
        out[f"rel_err_pct_{slot}"] = rel
        if math.isfinite(rel):
            f_errors.append(rel)

    ac = np.asarray(aligned.get("chi", np.zeros((0, 0))), dtype=float)
    for slot, target_value in sorted(target.items()):
        if not slot.startswith("chi_"):
            continue
        ij = slot.split("_", 1)[1]
        i, j = int(ij[0]) - 1, int(ij[1]) - 1
        pred = float(ac[i, j]) if ac.ndim == 2 and i < ac.shape[0] and j < ac.shape[1] else float("nan")
        # Random stress chi targets are explicitly selected to be non-zero.
        # Held-out 3Q-line chi entries are also non-zero. Keep a tiny floor only
        # as numerical protection.
        rel = (
            100.0 * abs(pred - float(target_value)) / max(abs(float(target_value)), 1e-9)
            if math.isfinite(pred) else float("nan")
        )
        out[f"target_{slot}"] = float(target_value)
        out[f"pred_{slot}"] = pred
        out[f"rel_err_pct_{slot}"] = rel
        if math.isfinite(rel):
            chi_errors.append(rel)
            (chi_diag if i == j else chi_off).append(rel)

    n_freq_requested = sum(1 for k in target if k.startswith("f_"))
    out["freq_valid_count"] = len(f_errors)
    out["freq_mean_rel_err_pct"] = float(np.mean(f_errors)) if f_errors else float("nan")
    out["freq_max_rel_err_pct"] = float(np.max(f_errors)) if f_errors else float("nan")
    out["chi_mean_rel_err_pct"] = float(np.mean(chi_errors)) if chi_errors else float("nan")
    out["chi_max_rel_err_pct"] = float(np.max(chi_errors)) if chi_errors else float("nan")
    out["chi_diag_mean_rel_err_pct"] = float(np.mean(chi_diag)) if chi_diag else float("nan")
    out["chi_offdiag_mean_rel_err_pct"] = float(np.mean(chi_off)) if chi_off else float("nan")

    freq_success = (
        len(f_errors) == n_freq_requested
        and n_freq_requested > 0
        and all(e < eps_pct for e in f_errors)
    )
    requested_errors = list(f_errors) + list(chi_errors)
    expected_requested = n_freq_requested + int(out["requested_chi_count"])
    full_success = (
        len(requested_errors) == expected_requested
        and expected_requested > 0
        and all(e < eps_pct for e in requested_errors)
    )
    out["freq_success_10pct"] = bool(freq_success)
    out["full_requested_success_10pct"] = bool(full_success)
    return out


def cem_objective(row: dict[str, Any], target: dict[str, float], requirement_penalty: float) -> float:
    """Worst-group + small mean refinement, always in relative-error percent."""
    if not row.get("ok_qultra", False):
        return 5000.0
    loss = 0.0
    if not row.get("requirements_met", True):
        loss += float(requirement_penalty)

    groups: list[float] = []
    f = row.get("freq_max_rel_err_pct")
    if f is None or not math.isfinite(float(f)):
        return loss + 2000.0
    groups.append(float(f))

    if any(k.startswith("chi_") for k in target):
        c = row.get("chi_max_rel_err_pct")
        if c is None or not math.isfinite(float(c)):
            return loss + 1500.0
        groups.append(float(c))

    return float(loss + max(groups) + 0.2 * float(np.mean(groups)))


# -----------------------------------------------------------------------------
# Batched decode + physical evaluation
# -----------------------------------------------------------------------------


@torch.no_grad()
def decode_and_evaluate(
    model,
    qinf,
    qu,
    z_batch: torch.Tensor,
    param_scaler,
    target: dict[str, float],
    required: dict[str, int],
    guidance_strength: float,
    stochastic: bool,
    f_min: float,
    f_max: float,
    eps_pct: float,
    requirement_penalty: float,
    iteration: int,
) -> list[dict[str, Any]]:
    req_arg = required if required and guidance_strength > 0 else None
    graphs = model.decoder.decode(
        z_batch,
        stochastic=stochastic,
        required_primitives=req_arg,
        guidance_strength=float(guidance_strength),
    )
    attrs_per_graph = model.param_decoder.predict(z_batch, graphs)
    rows: list[dict[str, Any]] = []

    for k, (g, attrs) in enumerate(zip(graphs, attrs_per_graph)):
        for i, t in enumerate(getattr(g, "node_types", [])):
            st = SubgType(int(t))
            if "dir" in SUBG_DEFS[st].attrs:
                attrs[i]["dir"] = float(getattr(g, "direction", [0.0] * len(g.node_types))[i])
        g.attrs = attrs

        row: dict[str, Any] = {
            "candidate_id": int(k),
            "iteration": int(iteration),
            "ok_qultra": False,
            "qultra_stage": "",
            "qultra_error": "",
            "requirements_json": json.dumps(required, sort_keys=True),
            "guidance_strength": float(guidance_strength),
        }
        row.update(graph_summary(g))
        row["requirements_met"] = bool(requirements_met(g, required))

        try:
            g_phys = graph_to_physical(g, param_scaler)
            topo = qinf._graph_ns_to_cqed_topology(g_phys, name=f"spec_unseen_{iteration}_{k}")
            sim = qinf._simulate_topology_arrays(topo, qu, f_min, f_max)
            row["ok_qultra"] = bool(sim.get("ok"))
            row["qultra_stage"] = str(sim.get("stage", ""))
            row["qultra_error"] = str(sim.get("error") or "")
            row["n_qultra_modes"] = len(np.asarray(sim.get("frequencies", []), dtype=float).ravel())
            row.update(compare_target_to_sim(qinf, target, sim, eps_pct))
        except Exception as exc:
            row["ok_qultra"] = False
            row["qultra_error"] = f"{type(exc).__name__}: {exc}"
            row.update(compare_target_to_sim(qinf, target, {"ok": False}, eps_pct))

        row["objective_loss"] = cem_objective(row, target, requirement_penalty)
        rows.append(row)
    return rows


# -----------------------------------------------------------------------------
# One target: direct inference + CEM latent/topology optimization
# -----------------------------------------------------------------------------


def run_direct_target(
    model,
    qinf,
    qu,
    target: dict[str, float],
    obs_scaler,
    param_scaler,
    device: torch.device,
    required: dict[str, int],
    args,
) -> tuple[torch.Tensor, dict[str, Any]]:
    z0 = encode_spec(model, target, obs_scaler, device)
    row = decode_and_evaluate(
        model=model,
        qinf=qinf,
        qu=qu,
        z_batch=z0.unsqueeze(0),
        param_scaler=param_scaler,
        target=target,
        required=required,
        guidance_strength=args.guidance_strength,
        stochastic=args.stochastic_decoder,
        f_min=args.f_min,
        f_max=args.f_max,
        eps_pct=args.frequency_epsilon_pct,
        requirement_penalty=args.requirement_penalty,
        iteration=-1,
    )[0]
    row["branch"] = "direct"
    return z0, row


def run_cem_target(
    model,
    qinf,
    qu,
    z0: torch.Tensor,
    direct_row: dict[str, Any],
    target: dict[str, float],
    param_scaler,
    required: dict[str, int],
    args,
    target_id: int,
    study: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    mu = z0.detach().clone()
    sigma = torch.full_like(mu, float(args.cem_init_sigma))
    best_row = dict(direct_row)
    best_loss = float(best_row.get("objective_loss", float("inf")))
    logs: list[dict[str, Any]] = []
    cumulative_evals = 1

    for it in range(int(args.cem_iters)):
        pop = int(args.cem_population)
        eps = torch.randn(pop, mu.numel(), device=mu.device)
        z_pop = mu.unsqueeze(0) + eps * sigma.unsqueeze(0)
        # Include current CEM center explicitly for monotonic best-so-far tracking.
        z_pop[0] = mu
        if args.cem_z_clip > 0:
            z_pop = torch.clamp(z_pop, -float(args.cem_z_clip), float(args.cem_z_clip))

        rows = decode_and_evaluate(
            model=model,
            qinf=qinf,
            qu=qu,
            z_batch=z_pop,
            param_scaler=param_scaler,
            target=target,
            required=required,
            guidance_strength=args.guidance_strength,
            stochastic=args.stochastic_decoder,
            f_min=args.f_min,
            f_max=args.f_max,
            eps_pct=args.frequency_epsilon_pct,
            requirement_penalty=args.requirement_penalty,
            iteration=it,
        )
        losses = np.asarray([float(r["objective_loss"]) for r in rows], dtype=float)
        order = np.argsort(losses)
        elite_n = max(1, min(int(args.cem_elite), len(rows)))
        elite_idx = order[:elite_n]
        elite_z = z_pop[torch.tensor(elite_idx, dtype=torch.long, device=mu.device)]

        elite_mean = elite_z.mean(dim=0)
        elite_std = elite_z.std(dim=0, unbiased=False).clamp(min=float(args.cem_min_sigma))
        m = float(args.cem_momentum)
        mu = m * mu + (1.0 - m) * elite_mean
        sigma = torch.maximum(
            torch.full_like(sigma, float(args.cem_min_sigma)),
            float(args.cem_sigma_decay) * (m * sigma + (1.0 - m) * elite_std),
        )
        if args.cem_z_clip > 0:
            mu = torch.clamp(mu, -float(args.cem_z_clip), float(args.cem_z_clip))

        candidate = rows[int(order[0])]
        if float(candidate["objective_loss"]) < best_loss:
            best_loss = float(candidate["objective_loss"])
            best_row = dict(candidate)

        cumulative_evals += len(rows)
        logs.append({
            "study": study,
            "target_id": int(target_id),
            "iteration": int(it),
            "cumulative_qultra_candidates": int(cumulative_evals),
            "iteration_best_loss": float(np.min(losses)),
            "iteration_median_loss": float(np.median(losses)),
            "best_so_far_loss": float(best_loss),
            "sigma_mean": float(sigma.mean().detach().cpu().item()),
            "iteration_qultra_ok_rate": float(np.mean([bool(r.get("ok_qultra")) for r in rows])),
            "iteration_freq_success_rate": float(np.mean([bool(r.get("freq_success_10pct")) for r in rows])),
        })

    best_row["branch"] = "cem"
    return best_row, logs


# -----------------------------------------------------------------------------
# Unseen Three-qubit-linear study
# -----------------------------------------------------------------------------


def pair_row(
    target_id: int,
    target: dict[str, float],
    direct: dict[str, Any],
    cem: dict[str, Any],
    study: str,
    case_name: str = "",
) -> dict[str, Any]:
    return {
        "study": study,
        "case": case_name,
        "target_id": int(target_id),
        "target_json": json.dumps(target, sort_keys=True),
        "direct_ok_qultra": bool(direct.get("ok_qultra")),
        "cem_ok_qultra": bool(cem.get("ok_qultra")),
        "direct_freq_success_10pct": bool(direct.get("freq_success_10pct")),
        "cem_freq_success_10pct": bool(cem.get("freq_success_10pct")),
        "direct_full_success_10pct": bool(direct.get("full_requested_success_10pct")),
        "cem_full_success_10pct": bool(cem.get("full_requested_success_10pct")),
        "direct_freq_max_rel_err_pct": direct.get("freq_max_rel_err_pct", float("nan")),
        "cem_freq_max_rel_err_pct": cem.get("freq_max_rel_err_pct", float("nan")),
        "direct_chi_max_rel_err_pct": direct.get("chi_max_rel_err_pct", float("nan")),
        "cem_chi_max_rel_err_pct": cem.get("chi_max_rel_err_pct", float("nan")),
        "direct_objective_loss": direct.get("objective_loss", float("nan")),
        "cem_objective_loss": cem.get("objective_loss", float("nan")),
        "direct_requirements_met": bool(direct.get("requirements_met", False)),
        "cem_requirements_met": bool(cem.get("requirements_met", False)),
        "direct_topology_signature": direct.get("topology_signature", ""),
        "cem_topology_signature": cem.get("topology_signature", ""),
        "topology_switched": direct.get("topology_signature") != cem.get("topology_signature"),
        "direct_macro_node_types": direct.get("macro_node_types", ""),
        "cem_macro_node_types": cem.get("macro_node_types", ""),
    }


def observable_summary_rows(direct_rows: list[dict], cem_rows: list[dict], slots: list[str]) -> list[dict]:
    out = []
    for branch_idx, (branch, rows) in enumerate([("direct", direct_rows), ("cem", cem_rows)]):
        for slot_idx, slot in enumerate(slots):
            vals = []
            for r in rows:
                v = r.get(f"rel_err_pct_{slot}")
                try:
                    if math.isfinite(float(v)):
                        vals.append(float(v))
                except Exception:
                    pass
            group = "frequency" if slot.startswith("f_") else (
                "chi_diag" if slot.startswith("chi_") and slot[-2] == slot[-1] else "chi_offdiag"
            )
            med, med_lo, med_hi = bootstrap_median_ci(
                vals,
                seed=17000 + 1000 * branch_idx + slot_idx,
            )
            out.append({
                "branch": branch,
                "observable": slot,
                "group": group,
                "n": len(vals),
                "median_rel_err_pct": med,
                "median_ci95_low_pct": med_lo,
                "median_ci95_high_pct": med_hi,
                "p90_rel_err_pct": percentile(vals, 90),
                "mean_rel_err_pct": float(np.mean(vals)) if vals else float("nan"),
            })
    return out


def run_unseen_study(model, qinf, qu, obs_scaler, param_scaler, device, args, out_dir: Path):
    targets = unseen_targets(args.n_unseen, args.seed)
    required = {"transmon": 3}
    direct_rows: list[dict[str, Any]] = []
    cem_rows: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    logs: list[dict[str, Any]] = []

    print(f"\n[UNSEEN] {UNSEEN_DATASET}: {len(targets)} target Hamiltonians")
    for i, target in enumerate(targets):
        z0, direct = run_direct_target(
            model, qinf, qu, target, obs_scaler, param_scaler, device, required, args
        )
        direct.update({"study": "unseen_three_qubit_linear", "target_id": i, "target_json": json.dumps(target, sort_keys=True)})
        cem, its = run_cem_target(
            model, qinf, qu, z0, direct, target, param_scaler, required, args,
            target_id=i, study="unseen_three_qubit_linear",
        )
        cem.update({"study": "unseen_three_qubit_linear", "target_id": i, "target_json": json.dumps(target, sort_keys=True)})
        direct_rows.append(direct)
        cem_rows.append(cem)
        logs.extend(its)
        pairs.append(pair_row(i, target, direct, cem, "unseen_three_qubit_linear"))

        print(
            f"  {i+1:03d}/{len(targets):03d} "
            f"direct fmax={direct.get('freq_max_rel_err_pct', float('nan')):.2f}% "
            f"chi={direct.get('chi_max_rel_err_pct', float('nan')):.2f}% | "
            f"CEM fmax={cem.get('freq_max_rel_err_pct', float('nan')):.2f}% "
            f"chi={cem.get('chi_max_rel_err_pct', float('nan')):.2f}%"
        )

    slots = [
        "f_1", "f_2", "f_3",
        "chi_11", "chi_22", "chi_33", "chi_12", "chi_13", "chi_23",
    ]
    obs_summary = observable_summary_rows(direct_rows, cem_rows, slots)

    write_csv(out_dir / "unseen_direct.csv", direct_rows)
    write_csv(out_dir / "unseen_cem.csv", cem_rows)
    write_csv(out_dir / "unseen_pairs.csv", pairs)
    write_csv(out_dir / "unseen_observable_summary.csv", obs_summary)
    write_csv(out_dir / "unseen_cem_iterations.csv", logs)

    return targets, direct_rows, cem_rows, pairs, obs_summary, logs


# -----------------------------------------------------------------------------
# Plausible random stress targets
# -----------------------------------------------------------------------------


STRESS_CASES = [
    {
        "name": "F1_T1",
        "slots": ["f_1"],
        "required": {"transmon": 1},
        "description": "1 frequency; >=1 transmon",
        "source_datasets": ["Qubit"],
    },
    {
        "name": "F2_T2C1",
        "slots": ["f_1", "f_2"],
        "required": {"transmon": 2, "coupler": 1},
        "description": "2 frequencies; >=2 transmons, >=1 coupler",
        "source_datasets": ["Two_qubit_with_capacitive_coupling"],
    },
    {
        "name": "F3_T3C2",
        "slots": ["f_1", "f_2", "f_3"],
        "required": {"transmon": 3, "coupler": 2},
        "description": "3 frequencies; >=3 transmons, >=2 couplers",
        "source_datasets": ["Three_qubit_capacitive_star"],
    },
    {
        "name": "F3_T4C3",
        "slots": ["f_1", "f_2", "f_3"],
        "required": {"transmon": 4, "coupler": 3},
        "description": (
            "3 requested frequencies; >=4 transmons, >=3 couplers "
            "(fourth transmon intentionally unconstrained by a target frequency)"
        ),
        "source_datasets": ["Three_qubit_capacitive_star"],
    },
    {
        "name": "F1CHI_T1",
        "slots": ["f_1", "chi_11"],
        "required": {"transmon": 1},
        "description": "1 frequency + self-chi; >=1 transmon",
        "source_datasets": ["Qubit"],
    },
    {
        "name": "F2CHI_T2C1",
        "slots": ["f_1", "f_2", "chi_11", "chi_22", "chi_12"],
        "required": {"transmon": 2, "coupler": 1},
        "description": "2 frequencies + chi; >=2 transmons, >=1 coupler",
        "source_datasets": ["Two_qubit_with_capacitive_coupling"],
    },
    {
        "name": "F3CHI_T3C2",
        "slots": ["f_1", "f_2", "f_3", "chi_11", "chi_22", "chi_33", "chi_12", "chi_23"],
        "required": {"transmon": 3, "coupler": 2},
        "description": "3 frequencies + selected chi; >=3 transmons, >=2 couplers",
        "source_datasets": ["Three_qubit_capacitive_star"],
    },
    {
        "name": "F3CHI_T4C3",
        "slots": ["f_1", "f_2", "f_3", "chi_11", "chi_22", "chi_33", "chi_12", "chi_23"],
        "required": {"transmon": 4, "coupler": 3},
        "description": (
            "3 frequencies + selected chi; >=4 transmons, >=3 couplers "
            "(fourth transmon intentionally unconstrained by a target frequency)"
        ),
        "source_datasets": ["Three_qubit_capacitive_star"],
    },
]


def eligible_pool_rows(pool: list[dict[str, Any]], slots: list[str]) -> list[dict[str, Any]]:
    eligible = []
    requested_f_ids = [int(s.split("_", 1)[1]) for s in slots if s.startswith("f_")]
    requested_modes = max(requested_f_ids) if requested_f_ids else 0
    for item in pool:
        obs = item["obs"]
        if not all(s in obs and math.isfinite(float(obs[s])) for s in slots):
            continue

        # Keep the stress hierarchy clean: F1 comes from genuinely one-mode
        # training examples, F2 from two-mode examples, and F3 from three-mode
        # examples. This avoids calling a truncated 3-mode Hamiltonian a
        # ``one-frequency'' target.
        available_modes = sum(
            1 for s, v in obs.items()
            if s.startswith("f_") and math.isfinite(float(v))
        )
        if requested_modes and available_modes != requested_modes:
            continue

        # Requested chi values should be meaningfully non-zero so relative error
        # has a sensible interpretation.
        if any(s.startswith("chi_") and abs(float(obs[s])) < 1e-8 for s in slots):
            continue
        eligible.append(item)
    return eligible


def make_plausible_random_targets(
    pool: list[dict[str, Any]],
    case: dict[str, Any],
    n: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    source_names = set(case.get("source_datasets", []))
    source_pool = [x for x in pool if not source_names or x.get("dataset") in source_names]
    eligible = eligible_pool_rows(source_pool, case["slots"])
    if not eligible:
        raise RuntimeError(f"No training rows support stress case {case['name']} slots={case['slots']}")

    # Empirical clipping bounds keep perturbations inside occupied training ranges.
    bounds: dict[str, tuple[float, float]] = {}
    for slot in case["slots"]:
        vals = np.asarray([float(x["obs"][slot]) for x in eligible], dtype=float)
        bounds[slot] = (float(np.percentile(vals, 5)), float(np.percentile(vals, 95)))

    out = []
    for _ in range(n):
        base = eligible[int(rng.integers(0, len(eligible)))]
        target: dict[str, float] = {}
        for slot in case["slots"]:
            x = float(base["obs"][slot])
            jitter = float(rng.uniform(0.97, 1.03)) if slot.startswith("f_") else float(rng.uniform(0.90, 1.10))
            lo, hi = bounds[slot]
            if lo > hi:
                lo, hi = hi, lo
            target[slot] = float(np.clip(x * jitter, lo, hi))
        out.append({
            "target": target,
            "base_dataset": base["dataset"],
            "required": dict(case["required"]),
            "case": case["name"],
            "description": case["description"],
        })
    return out


def run_stress_study(model, qinf, qu, obs_scaler, param_scaler, device, args, out_dir: Path):
    print("\n[STRESS] Building empirical training-observable pool...")
    pool = build_training_observable_pool(args.stress_pool_per_dataset, args.seed + 10000)
    rng = np.random.default_rng(args.seed + 20000)

    target_rows: list[dict[str, Any]] = []
    direct_rows: list[dict[str, Any]] = []
    cem_rows: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    logs: list[dict[str, Any]] = []
    global_id = 0

    for case in STRESS_CASES:
        cases = make_plausible_random_targets(pool, case, args.n_stress_per_case, rng)
        print(f"  case {case['name']}: {len(cases)} targets | {case['description']}")
        for j, item in enumerate(cases):
            target = item["target"]
            required = item["required"]
            target_rows.append({
                "target_id": global_id,
                "case": case["name"],
                "base_dataset": item["base_dataset"],
                "description": item["description"],
                "requirements_json": json.dumps(required, sort_keys=True),
                "target_json": json.dumps(target, sort_keys=True),
            })

            z0, direct = run_direct_target(
                model, qinf, qu, target, obs_scaler, param_scaler, device, required, args
            )
            direct.update({
                "study": "random_hamiltonian_stress",
                "case": case["name"],
                "target_id": global_id,
                "base_dataset": item["base_dataset"],
                "target_json": json.dumps(target, sort_keys=True),
            })

            if args.stress_cem:
                cem, its = run_cem_target(
                    model, qinf, qu, z0, direct, target, param_scaler, required, args,
                    target_id=global_id, study=f"stress:{case['name']}",
                )
                logs.extend(its)
            else:
                cem = dict(direct)
                cem["branch"] = "cem_skipped"

            cem.update({
                "study": "random_hamiltonian_stress",
                "case": case["name"],
                "target_id": global_id,
                "base_dataset": item["base_dataset"],
                "target_json": json.dumps(target, sort_keys=True),
            })

            direct_rows.append(direct)
            cem_rows.append(cem)
            pairs.append(pair_row(
                global_id, target, direct, cem, "random_hamiltonian_stress", case["name"]
            ))
            global_id += 1

    # SUCCESS in this table is intentionally frequency-only at epsilon=10%.
    summary: list[dict[str, Any]] = []
    for case in STRESS_CASES:
        name = case["name"]
        d = [r for r in direct_rows if r.get("case") == name]
        c = [r for r in cem_rows if r.get("case") == name]
        d_success = [bool(r.get("freq_success_10pct")) for r in d]
        c_success = [bool(r.get("freq_success_10pct")) for r in c]
        dp, dlo, dhi = wilson_interval(d_success)
        cp, clo, chi = wilson_interval(c_success)

        d_freq_errors = [float(r.get("freq_max_rel_err_pct", float("nan"))) for r in d]
        c_freq_errors = [float(r.get("freq_max_rel_err_pct", float("nan"))) for r in c]
        dmed, dmed_lo, dmed_hi = bootstrap_median_ci(
            d_freq_errors, seed=31000 + len(summary)
        )
        cmed, cmed_lo, cmed_hi = bootstrap_median_ci(
            c_freq_errors, seed=41000 + len(summary)
        )

        summary.append({
            "case": name,
            "description": case["description"],
            "n_targets": len(d),
            "n_requested_slots": len(case["slots"]),
            "n_requested_frequencies": sum(s.startswith("f_") for s in case["slots"]),
            "n_requested_chi": sum(s.startswith("chi_") for s in case["slots"]),
            "requirements_json": json.dumps(case["required"], sort_keys=True),
            "source_datasets": ";".join(case.get("source_datasets", [])),
            "success_definition": f"all requested frequencies within {args.frequency_epsilon_pct:g}%",
            "direct_frequency_success_rate": dp,
            "direct_frequency_success_ci95_low": dlo,
            "direct_frequency_success_ci95_high": dhi,
            "cem_frequency_success_rate": cp,
            "cem_frequency_success_ci95_low": clo,
            "cem_frequency_success_ci95_high": chi,
            "direct_median_freq_max_rel_err_pct": dmed,
            "direct_median_freq_max_rel_err_ci95_low_pct": dmed_lo,
            "direct_median_freq_max_rel_err_ci95_high_pct": dmed_hi,
            "cem_median_freq_max_rel_err_pct": cmed,
            "cem_median_freq_max_rel_err_ci95_low_pct": cmed_lo,
            "cem_median_freq_max_rel_err_ci95_high_pct": cmed_hi,
            # Structural diagnostics are reported separately from the deliberately
            # frequency-only stress-test success criterion.
            "direct_requirements_met_rate": float(np.mean([bool(r.get("requirements_met")) for r in d])) if d else float("nan"),
            "cem_requirements_met_rate": float(np.mean([bool(r.get("requirements_met")) for r in c])) if c else float("nan"),
            "direct_joint_frequency_and_requirements_rate": float(np.mean([
                bool(r.get("freq_success_10pct")) and bool(r.get("requirements_met")) for r in d
            ])) if d else float("nan"),
            "cem_joint_frequency_and_requirements_rate": float(np.mean([
                bool(r.get("freq_success_10pct")) and bool(r.get("requirements_met")) for r in c
            ])) if c else float("nan"),
            "direct_qultra_ok_rate": float(np.mean([bool(r.get("ok_qultra")) for r in d])) if d else float("nan"),
            "cem_qultra_ok_rate": float(np.mean([bool(r.get("ok_qultra")) for r in c])) if c else float("nan"),
        })

    write_csv(out_dir / "stress_targets.csv", target_rows)
    write_csv(out_dir / "stress_direct.csv", direct_rows)
    write_csv(out_dir / "stress_cem.csv", cem_rows)
    write_csv(out_dir / "stress_pairs.csv", pairs)
    write_csv(out_dir / "stress_summary.csv", summary)
    write_csv(out_dir / "stress_cem_iterations.csv", logs)
    return target_rows, direct_rows, cem_rows, pairs, summary, logs


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------


def plot_unseen_observable_summary(rows: list[dict[str, Any]], path: Path) -> None:
    obs = []
    for r in rows:
        if r["observable"] not in obs:
            obs.append(r["observable"])
    direct = {r["observable"]: r for r in rows if r["branch"] == "direct"}
    cem = {r["observable"]: r for r in rows if r["branch"] == "cem"}
    x = np.arange(len(obs), dtype=float)

    d_med = [direct.get(o, {}).get("median_rel_err_pct", np.nan) for o in obs]
    d_lo = [direct.get(o, {}).get("median_ci95_low_pct", np.nan) for o in obs]
    d_hi = [direct.get(o, {}).get("median_ci95_high_pct", np.nan) for o in obs]
    c_med = [cem.get(o, {}).get("median_rel_err_pct", np.nan) for o in obs]
    c_lo = [cem.get(o, {}).get("median_ci95_low_pct", np.nan) for o in obs]
    c_hi = [cem.get(o, {}).get("median_ci95_high_pct", np.nan) for o in obs]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.errorbar(
        x, d_med,
        yerr=asymmetric_errors(d_med, d_lo, d_hi),
        marker="o", capsize=3, linewidth=1.6,
        label="Direct median (95% bootstrap CI)",
    )
    ax.plot(
        x,
        [direct.get(o, {}).get("p90_rel_err_pct", np.nan) for o in obs],
        marker="x", linestyle="--", linewidth=1.2,
        label="Direct P90",
    )
    # Deliberately no CEM P90: the optimized branch is summarized by its median
    # plus target-level bootstrap uncertainty.
    ax.errorbar(
        x, c_med,
        yerr=asymmetric_errors(c_med, c_lo, c_hi),
        marker="o", capsize=3, linewidth=1.6,
        label="CEM median (95% bootstrap CI)",
    )
    ax.axhline(10.0, linestyle=":", linewidth=1.2, label="10% reference")
    ax.set_xticks(x)
    ax.set_xticklabels(obs, rotation=40, ha="right")
    ax.set_ylabel("Relative error (%)")
    ax.set_title("Held-out Three-qubit linear: Hamiltonian error by observable")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_unseen_success(direct: list[dict], cem: list[dict], path: Path, eps_pct: float) -> None:
    labels = ["Frequency success", "Full requested H success", "QuLTRA OK"]
    d_bool = [
        [bool(r.get("freq_success_10pct")) for r in direct],
        [bool(r.get("full_requested_success_10pct")) for r in direct],
        [bool(r.get("ok_qultra")) for r in direct],
    ]
    c_bool = [
        [bool(r.get("freq_success_10pct")) for r in cem],
        [bool(r.get("full_requested_success_10pct")) for r in cem],
        [bool(r.get("ok_qultra")) for r in cem],
    ]

    d_stats = [wilson_interval(v) for v in d_bool]
    c_stats = [wilson_interval(v) for v in c_bool]
    dvals = [s[0] for s in d_stats]
    cvals = [s[0] for s in c_stats]

    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.bar(
        x - w / 2, 100 * np.asarray(dvals), width=w, label="Direct",
        yerr=asymmetric_errors(
            dvals, [s[1] for s in d_stats], [s[2] for s in d_stats], scale=100.0
        ),
        capsize=4,
    )
    ax.bar(
        x + w / 2, 100 * np.asarray(cvals), width=w, label="CEM",
        yerr=asymmetric_errors(
            cvals, [s[1] for s in c_stats], [s[2] for s in c_stats], scale=100.0
        ),
        capsize=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Rate (%)")
    ax.set_title(f"Held-out Three-qubit linear (epsilon={eps_pct:g}%)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_cem_convergence(logs: list[dict], path: Path, title: str) -> None:
    if not logs:
        return
    by_it: defaultdict[int, list[float]] = defaultdict(list)
    by_it_eval: defaultdict[int, list[float]] = defaultdict(list)
    for r in logs:
        by_it[int(r["iteration"])].append(float(r["best_so_far_loss"]))
        by_it_eval[int(r["iteration"])].append(float(r["cumulative_qultra_candidates"]))
    its = sorted(by_it)

    x = [float(np.median(by_it_eval[i])) for i in its]
    y, lo, hi = [], [], []
    for i in its:
        med, med_lo, med_hi = bootstrap_median_ci(
            by_it[i], seed=52000 + int(i)
        )
        y.append(med)
        lo.append(med_lo)
        hi.append(med_hi)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(x, y, marker="o", label="Median best-so-far objective")
    ax.fill_between(x, lo, hi, alpha=0.18, label="95% bootstrap CI")
    ax.set_xlabel("Candidate physical evaluations per target")
    ax.set_ylabel("Median best-so-far objective")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_stress_summary(summary: list[dict], path: Path, eps_pct: float) -> None:
    if not summary:
        return
    labels = [r["case"] for r in summary]
    d = np.asarray([r["direct_frequency_success_rate"] for r in summary], dtype=float)
    c = np.asarray([r["cem_frequency_success_rate"] for r in summary], dtype=float)

    dlo = [r["direct_frequency_success_ci95_low"] for r in summary]
    dhi = [r["direct_frequency_success_ci95_high"] for r in summary]
    clo = [r["cem_frequency_success_ci95_low"] for r in summary]
    chi = [r["cem_frequency_success_ci95_high"] for r in summary]

    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(12, 5.7))
    ax.bar(
        x - w / 2, 100 * d, width=w, label="Direct",
        yerr=asymmetric_errors(d.tolist(), dlo, dhi, scale=100.0),
        capsize=4,
    )
    ax.bar(
        x + w / 2, 100 * c, width=w, label="CEM",
        yerr=asymmetric_errors(c.tolist(), clo, chi, scale=100.0),
        capsize=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Frequency success rate (%)")
    ax.set_title(
        f"Random plausible Hamiltonian stress test: |delta f|/f < {eps_pct:g}% "
        "(95% Wilson intervals)"
    )
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_stress_combined(
    summary: list[dict],
    pairs: list[dict[str, Any]],
    path: Path,
    eps_pct: float,
) -> None:
    """Single thesis-ready figure combining success and frequency-error scaling.

    The last point is intentionally an OOD structural stress case: the
    specification contains only three modal frequencies (and selected three-mode
    Kerr terms in the chi-conditioned branch), while the decoder is asked for at
    least four transmons and three couplers.
    """
    if not summary or not pairs:
        return

    smap = {r["case"]: r for r in summary}
    freq_cases = ["F1_T1", "F2_T2C1", "F3_T3C2", "F3_T4C3"]
    chi_cases = ["F1CHI_T1", "F2CHI_T2C1", "F3CHI_T3C2", "F3CHI_T4C3"]
    labels = ["1T", "2T+1C", "3T+2C", "4T+3C\\n(3 target f)"]
    x = np.arange(len(labels), dtype=float)

    def success_series(cases: list[str], branch: str):
        center, low, high = [], [], []
        for case in cases:
            r = smap[case]
            center.append(float(r[f"{branch}_frequency_success_rate"]))
            low.append(float(r[f"{branch}_frequency_success_ci95_low"]))
            high.append(float(r[f"{branch}_frequency_success_ci95_high"]))
        return center, low, high

    def error_series(cases: list[str], branch: str):
        center, low, high = [], [], []
        for j, case in enumerate(cases):
            vals = [
                float(r.get(f"{branch}_freq_max_rel_err_pct", float("nan")))
                for r in pairs
                if r.get("case") == case
            ]
            med, lo, hi = bootstrap_median_ci(
                vals,
                seed=63000 + (1000 if branch == "cem" else 0) + 100 * j
                + (5000 if "CHI" in case else 0),
            )
            center.append(med)
            low.append(lo)
            high.append(hi)
        return center, low, high

    fd, fdlo, fdhi = success_series(freq_cases, "direct")
    fc, fclo, fchi = success_series(freq_cases, "cem")
    cd, cdlo, cdhi = success_series(chi_cases, "direct")
    cc, cclo, cchi = success_series(chi_cases, "cem")

    fde, fdelo, fdehi = error_series(freq_cases, "direct")
    fce, fcelo, fcehi = error_series(freq_cases, "cem")
    cde, cdelo, cdehi = error_series(chi_cases, "direct")
    cce, ccelo, ccehi = error_series(chi_cases, "cem")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), constrained_layout=True)

    # Left: success rate with Wilson intervals.
    ax = axes[0]
    w = 0.18
    ax.bar(
        x - 1.5 * w, 100 * np.asarray(fd), width=w, label="Freq only - Direct",
        yerr=asymmetric_errors(fd, fdlo, fdhi, scale=100.0), capsize=3,
    )
    ax.bar(
        x - 0.5 * w, 100 * np.asarray(fc), width=w, label="Freq only - CEM",
        yerr=asymmetric_errors(fc, fclo, fchi, scale=100.0), capsize=3,
    )
    ax.bar(
        x + 0.5 * w, 100 * np.asarray(cd), width=w, label=r"Freq + $\chi$ - Direct",
        yerr=asymmetric_errors(cd, cdlo, cdhi, scale=100.0), capsize=3,
    )
    ax.bar(
        x + 1.5 * w, 100 * np.asarray(cc), width=w, label=r"Freq + $\chi$ - CEM",
        yerr=asymmetric_errors(cc, cclo, cchi, scale=100.0), capsize=3,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Frequency success rate (%)")
    ax.set_xlabel("Minimum requested topology complexity")
    ax.set_title("(a) Frequency success (95% Wilson CI)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(ncol=2)

    # Right: target-level median max frequency error with bootstrap CI.
    ax = axes[1]
    ax.errorbar(
        x, fde, yerr=asymmetric_errors(fde, fdelo, fdehi),
        marker="o", linewidth=1.8, capsize=3, label="Freq only - Direct",
    )
    ax.errorbar(
        x, fce, yerr=asymmetric_errors(fce, fcelo, fcehi),
        marker="o", linestyle="--", linewidth=1.8, capsize=3, label="Freq only - CEM",
    )
    ax.errorbar(
        x, cde, yerr=asymmetric_errors(cde, cdelo, cdehi),
        marker="s", linewidth=1.8, capsize=3, label=r"Freq + $\chi$ - Direct",
    )
    ax.errorbar(
        x, cce, yerr=asymmetric_errors(cce, ccelo, ccehi),
        marker="s", linestyle="--", linewidth=1.8, capsize=3, label=r"Freq + $\chi$ - CEM",
    )
    ax.axhline(eps_pct, linestyle=":", linewidth=1.3, label=f"{eps_pct:g}% threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Median maximum frequency error (%)")
    ax.set_xlabel("Minimum requested topology complexity")
    ax.set_title("(b) Frequency error (95% bootstrap CI)")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.suptitle(
        "Unified random-Hamiltonian stress test: increasing structural constraints",
        fontsize=15,
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# CLI and main
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate held-out 3Q-linear Hamiltonians and random plausible Hamiltonian stress tests."
    )
    p.add_argument("--ckpt", default=str(REPO_ROOT / "best_vae_global.pt"))
    p.add_argument("--out-dir", default=str(REPO_ROOT / "results" / "unseen_and_stress"))
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--n-unseen", type=int, default=20, help="Held-out 3Q-linear targets; hard cap 100.")
    p.add_argument("--n-stress-per-case", type=int, default=5,
                   help="Targets per stress case; total across the eight stress cases is capped at 100.")
    p.add_argument("--stress-pool-per-dataset", type=int, default=800, help="Training rows used to construct plausible stress targets.")
    p.add_argument("--frequency-epsilon-pct", type=float, default=DEFAULT_EPS_PCT)

    p.add_argument("--guidance-strength", type=float, default=8.0,
                   help="Soft semantic primitive-count guidance used identically in direct and CEM branches.")
    p.add_argument("--stochastic-decoder", action="store_true",
                   help="Use stochastic topology decoding. Deterministic is the default for reproducibility.")

    p.add_argument("--cem-population", type=int, default=12)
    p.add_argument("--cem-elite", type=int, default=3)
    p.add_argument("--cem-iters", type=int, default=6)
    p.add_argument("--cem-init-sigma", type=float, default=0.8)
    p.add_argument("--cem-min-sigma", type=float, default=0.05)
    p.add_argument("--cem-sigma-decay", type=float, default=0.90)
    p.add_argument("--cem-momentum", type=float, default=0.65)
    p.add_argument("--cem-z-clip", type=float, default=4.0)
    p.add_argument("--requirement-penalty", type=float, default=300.0,
                   help="Added to CEM objective when minimum primitive requirements are not met.")

    p.add_argument("--f-min", type=float, default=1.0)
    p.add_argument("--f-max", type=float, default=9.0)
    p.add_argument("--no-stress-cem", dest="stress_cem", action="store_false",
                   help="Run only direct inference for the random stress test.")
    p.set_defaults(stress_cem=True)
    return p.parse_args()


def validate_args(args) -> None:
    if not 1 <= args.n_unseen <= 100:
        raise ValueError("--n-unseen must be between 1 and 100.")
    if args.n_stress_per_case < 1:
        raise ValueError("--n-stress-per-case must be >=1.")
    if args.n_stress_per_case * len(STRESS_CASES) > 100:
        raise ValueError(
            f"Stress targets are capped at 100 total: {len(STRESS_CASES)} cases x "
            f"{args.n_stress_per_case} = {len(STRESS_CASES) * args.n_stress_per_case}. "
            f"Use --n-stress-per-case <= {100 // len(STRESS_CASES)}."
        )
    if args.cem_population < 2:
        raise ValueError("--cem-population must be >=2.")
    if not 1 <= args.cem_elite <= args.cem_population:
        raise ValueError("--cem-elite must be in [1, cem-population].")
    if args.cem_iters < 1:
        raise ValueError("--cem-iters must be >=1.")
    if args.frequency_epsilon_pct <= 0:
        raise ValueError("--frequency-epsilon-pct must be >0.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    qinf = import_qinf()
    if not getattr(qinf, "_HAS_QULTRA_WORKFLOW", False):
        err = getattr(qinf, "_QULTRA_IMPORT_ERROR", "unknown import error")
        raise RuntimeError(f"QuLTRA workflow is unavailable: {err}")

    print("=" * 90)
    print("cQED-Gen unseen + stress analysis")
    print("=" * 90)
    print(f"repo:       {REPO_ROOT}")
    print(f"checkpoint: {args.ckpt}")
    print(f"device:     {device}")
    print(f"out:        {out_dir}")
    print(f"epsilon:    {args.frequency_epsilon_pct:g}% (frequency success)")

    model, scalers, cfg = qinf.load_checkpoint(args.ckpt, device)
    model.eval()
    gs = get_global_scaler(scalers)
    obs_scaler = gs.obs_scaler
    param_scaler = gs.param_scaler
    if getattr(param_scaler, "mode", None) != "attr":
        raise RuntimeError(
            "The checkpoint ParamScaler is not in global attribute mode. "
            "Use best_vae_global.pt for unseen-topology evaluation."
        )
    qu = qinf.import_qultra()

    unseen = run_unseen_study(model, qinf, qu, obs_scaler, param_scaler, device, args, out_dir)
    _, unseen_direct, unseen_cem, unseen_pairs, unseen_obs_summary, unseen_logs = unseen

    stress = run_stress_study(model, qinf, qu, obs_scaler, param_scaler, device, args, out_dir)
    stress_targets, stress_direct, stress_cem, stress_pairs, stress_summary, stress_logs = stress

    plot_unseen_observable_summary(unseen_obs_summary, out_dir / "unseen_observable_errors.pdf")
    plot_unseen_success(unseen_direct, unseen_cem, out_dir / "unseen_success_10pct.pdf", args.frequency_epsilon_pct)
    plot_cem_convergence(
        unseen_logs,
        out_dir / "unseen_cem_convergence.pdf",
        "Held-out Three-qubit linear: CEM latent/topology refinement",
    )
    plot_stress_summary(
        stress_summary,
        out_dir / "stress_frequency_success_10pct.pdf",
        args.frequency_epsilon_pct,
    )
    plot_stress_combined(
        stress_summary,
        stress_pairs,
        out_dir / "stress_combined_single_figure.pdf",
        args.frequency_epsilon_pct,
    )

    topology_switch_rate = float(np.mean([bool(r["topology_switched"]) for r in unseen_pairs])) if unseen_pairs else float("nan")
    summary = {
        "checkpoint": str(args.ckpt),
        "seed": args.seed,
        "frequency_success_epsilon_pct": args.frequency_epsilon_pct,
        "methodology": {
            "branch": "specification encoder only; circuit encoder unused",
            "unseen_dataset": UNSEEN_DATASET,
            "unseen_success_independent_of_reference_topology": True,
            "mode_alignment": "minimum absolute frequency assignment; same permutation applied to chi",
            "unseen_cem_objective": "worst requested Hamiltonian group relative error + 0.2 mean + requirement penalty",
            "stress_target_generation": "training observable rows + bounded jitter + clipping to empirical 5-95 percentiles",
            "stress_success": f"all requested frequencies have relative error < {args.frequency_epsilon_pct:g}%",
            "stress_chi_role": "conditioning + CEM objective only; excluded from stress success criterion",
        },
        "unseen": {
            "n_targets": len(unseen_direct),
            "direct_frequency_success_rate": float(np.mean([bool(r.get("freq_success_10pct")) for r in unseen_direct])) if unseen_direct else None,
            "cem_frequency_success_rate": float(np.mean([bool(r.get("freq_success_10pct")) for r in unseen_cem])) if unseen_cem else None,
            "direct_full_H_success_rate": float(np.mean([bool(r.get("full_requested_success_10pct")) for r in unseen_direct])) if unseen_direct else None,
            "cem_full_H_success_rate": float(np.mean([bool(r.get("full_requested_success_10pct")) for r in unseen_cem])) if unseen_cem else None,
            "topology_switch_rate_direct_to_cem": topology_switch_rate,
            "direct_median_freq_max_rel_err_pct": percentile([float(r.get("freq_max_rel_err_pct", float("nan"))) for r in unseen_direct], 50),
            "cem_median_freq_max_rel_err_pct": percentile([float(r.get("freq_max_rel_err_pct", float("nan"))) for r in unseen_cem], 50),
            "direct_median_chi_max_rel_err_pct": percentile([float(r.get("chi_max_rel_err_pct", float("nan"))) for r in unseen_direct], 50),
            "cem_median_chi_max_rel_err_pct": percentile([float(r.get("chi_max_rel_err_pct", float("nan"))) for r in unseen_cem], 50),
        },
        "stress": stress_summary,
        "cem": {
            "population": args.cem_population,
            "elite": args.cem_elite,
            "iterations": args.cem_iters,
            "init_sigma": args.cem_init_sigma,
            "min_sigma": args.cem_min_sigma,
            "sigma_decay": args.cem_sigma_decay,
            "momentum": args.cem_momentum,
            "z_clip": args.cem_z_clip,
        },
    }
    (out_dir / "analysis_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2), encoding="utf-8"
    )

    print("\nDone. Main outputs:")
    for name in [
        "unseen_direct.csv",
        "unseen_cem.csv",
        "unseen_pairs.csv",
        "unseen_observable_summary.csv",
        "unseen_observable_errors.pdf",
        "unseen_success_10pct.pdf",
        "unseen_cem_convergence.pdf",
        "stress_targets.csv",
        "stress_summary.csv",
        "stress_frequency_success_10pct.pdf",
        "stress_combined_single_figure.pdf",
        "analysis_summary.json",
    ]:
        print(f"  {out_dir / name}")


if __name__ == "__main__":
    main()
