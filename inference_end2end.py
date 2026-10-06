#!/usr/bin/env python3
"""
Latent-space optimization with QuLTRA in the loop.

Goal
----
The user fixes any subset of Hamiltonian observables from the terminal
(e.g. f_1, f_2, f_3, chi_11, chi_13). Missing observables are filled in the
scaled space, then the spec encoder gives an initial latent code z0. Around z0
we run a simple Cross-Entropy Method (CEM) optimizer:

    z -> decoder -> param_decoder -> physical circuit -> QuLTRA -> frequencies

The loss compares all user-specified Hamiltonian observables available from
QuLTRA: frequencies, chi entries, and kappa entries. QuLTRA failures receive a
configurable penalty. The optimizer keeps the best elite candidates
and recenters the latent search distribution.  Optionally, ``--quantum-metal``
passes the best expanded primitive graph directly to graph2metal and writes a
Quantum Metal layout-zero design.

Example
-------
python optimize_latent_with_qultra.py \
  --ckpt new_best_vae.pt \
  --f_1 5.0 \
  --f_2 5.35 \
  --f_3 8.0 \
  --chi_11 0.040 \
  --chi_33 0.001 \
  --chi_13 0.001 \
  --required-transmon 2 \
  --required-resonator 1 \
  --guidance-strength 10 \
  --population 64 \
  --elite 8 \
  --iters 20 \
  --stochastic \
  --print-best \
  --plot-best \
  --quantum-metal \
  --out-dir pres_plot/latent_qultra_optimization
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import networkx as nx
    _HAS_NX = True
except Exception:
    nx = None
    _HAS_NX = False


THIS_FILE = Path(__file__).resolve()


def find_repo_root() -> Path:
    candidates = [THIS_FILE.parent, *THIS_FILE.parents, Path.cwd(), *Path.cwd().parents]
    for c in candidates:
        if (c / "inference_hamiltonian_qultra.py").exists() and (c / "src").exists():
            return c.resolve()
        if (c / "inference_circuit_elements.py").exists() and (c / "src").exists():
            return c.resolve()
    raise RuntimeError("Could not find repo root: expected inference_hamiltonian_qultra.py/inference_circuit_elements.py and src/.")


REPO_ROOT = find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
os.chdir(REPO_ROOT)

from circuit2graph import SubgType, SUBG_DEFS, expand_topology  # noqa: E402
from circuit2graph.constraints import exists_valid_macro_graph, is_compatible  # noqa: E402
from data_loader.schema import OBS_SLOTS, OBS_IDX, N_OBS_SLOTS  # noqa: E402


def import_qultra_inference():
    path = REPO_ROOT / "inference_hamiltonian_qultra.py"
    if not path.exists():
        raise RuntimeError(f"Cannot find {path}")
    spec = importlib.util.spec_from_file_location("inference_hamiltonian_qultra", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _fmt_float(x: Any, ndigits: int = 5) -> str:
    try:
        v = float(x)
    except Exception:
        return str(x)
    if not math.isfinite(v):
        return str(v)
    if abs(v) >= 1e4 or (0 < abs(v) < 1e-3):
        return f"{v:.3e}"
    return f"{v:.{ndigits}g}"


def add_observable_arguments(parser: argparse.ArgumentParser) -> None:
    """Add both --f_1 and --f1 style arguments for every OBS_SLOT."""
    added: set[str] = set()
    for slot in OBS_SLOTS:
        canonical = "--" + slot
        alias = "--" + slot.replace("_", "")
        dest = "obs_" + slot
        if canonical not in added:
            parser.add_argument(canonical, type=float, default=None, dest=dest)
            added.add(canonical)
        if alias != canonical and alias not in added:
            parser.add_argument(alias, type=float, default=None, dest=dest)
            added.add(alias)


def user_observables_from_args(args: argparse.Namespace) -> dict[str, float]:
    obs = {}
    for slot in OBS_SLOTS:
        v = getattr(args, "obs_" + slot, None)
        if v is not None:
            obs[slot] = float(v)
    return obs


def raw_to_scaled_value(slot: str, value: float, obs_scaler) -> float:
    idx = OBS_IDX[slot]
    mean = np.asarray(getattr(obs_scaler, "mean_", np.zeros(N_OBS_SLOTS)), dtype=np.float64)
    std = np.asarray(getattr(obs_scaler, "std_", np.ones(N_OBS_SLOTS)), dtype=np.float64)
    return float((float(value) - mean[idx]) / max(float(std[idx]), 1e-12))


def scaled_to_raw_value(slot: str, scaled: float, obs_scaler) -> float:
    idx = OBS_IDX[slot]
    mean = np.asarray(getattr(obs_scaler, "mean_", np.zeros(N_OBS_SLOTS)), dtype=np.float64)
    std = np.asarray(getattr(obs_scaler, "std_", np.ones(N_OBS_SLOTS)), dtype=np.float64)
    return float(float(scaled) * max(float(std[idx]), 1e-12) + mean[idx])


def build_complete_scaled_observables(
    fixed_raw_obs: dict[str, float],
    obs_scaler,
    rng: np.random.Generator,
    random_scaled_std: float,
    random_scaled_clip: float,
    include_all_slots: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], dict[str, float]]:
    """Build a complete obs vector.

    User-specified values are raw/physical and are internally scaled.
    Missing values are sampled directly in scaled space.
    """
    vals_scaled = np.zeros(N_OBS_SLOTS, dtype=np.float64)
    mask = np.zeros(N_OBS_SLOTS, dtype=np.float64)
    raw_debug: dict[str, float] = {}
    scaled_debug: dict[str, float] = {}

    # Mode dimension inferred from specified frequencies, if any.
    specified_freq_indices = []
    for slot in fixed_raw_obs:
        if slot.startswith("f_"):
            try:
                specified_freq_indices.append(int(slot.split("_", 1)[1]))
            except Exception:
                pass
    n_modes = max(specified_freq_indices) if specified_freq_indices else 0

    for slot in OBS_SLOTS:
        if slot in fixed_raw_obs:
            scaled = raw_to_scaled_value(slot, fixed_raw_obs[slot], obs_scaler)
            idx = OBS_IDX[slot]
            vals_scaled[idx] = scaled
            mask[idx] = 1.0
            raw_debug[slot] = float(fixed_raw_obs[slot])
            scaled_debug[slot] = float(scaled)
            continue

        include = include_all_slots
        if not include and n_modes > 0:
            if slot.startswith("kappa_"):
                i = int(slot.split("_", 1)[1])
                include = i <= n_modes
            elif slot.startswith("chi_"):
                ij = slot.split("_", 1)[1]
                i, j = int(ij[0]), int(ij[1])
                include = i <= n_modes and j <= n_modes

        if include:
            scaled = float(rng.normal(0.0, float(random_scaled_std)))
            if random_scaled_clip and random_scaled_clip > 0:
                scaled = float(np.clip(scaled, -float(random_scaled_clip), float(random_scaled_clip)))
            idx = OBS_IDX[slot]
            vals_scaled[idx] = scaled
            mask[idx] = 1.0
            scaled_debug[slot] = float(scaled)
            raw_debug[slot] = scaled_to_raw_value(slot, scaled, obs_scaler)

    return vals_scaled, mask, raw_debug, scaled_debug


def flat_attr_names_from_graph(g: SimpleNamespace) -> list[str]:
    names: list[str] = []
    for st_int in getattr(g, "node_types", []):
        st = SubgType(int(st_int))
        for attr in SUBG_DEFS[st].attrs:
            if attr != "dir":
                names.append(attr)
    return names


def flat_scaled_attrs_from_graph(g: SimpleNamespace) -> list[float]:
    row: list[float] = []
    for i, st_int in enumerate(getattr(g, "node_types", [])):
        st = SubgType(int(st_int))
        attrs = g.attrs[i] if hasattr(g, "attrs") and i < len(g.attrs) else {}
        for attr in SUBG_DEFS[st].attrs:
            if attr == "dir":
                continue
            row.append(float(attrs.get(attr, float("nan"))))
    return row


def graph_with_physical_attrs(g: SimpleNamespace, global_param_scaler) -> tuple[SimpleNamespace, dict[str, Any]]:
    attr_names = flat_attr_names_from_graph(g)
    scaled = np.asarray(flat_scaled_attrs_from_graph(g), dtype=np.float64)
    info = {
        "physical_rescale_ok": False,
        "physical_rescale_error": "",
        "n_param_attrs": int(len(attr_names)),
        "param_attr_names": ";".join(attr_names),
    }
    if len(attr_names) == 0:
        out = SimpleNamespace(
            node_types=list(getattr(g, "node_types", [])),
            edges=[tuple(e) for e in getattr(g, "edges", [])],
            direction=list(getattr(g, "direction", [0.0] * len(getattr(g, "node_types", [])))),
            attrs=[{} for _ in getattr(g, "node_types", [])],
        )
        info["physical_rescale_ok"] = True
        return out, info
    try:
        physical = global_param_scaler.inverse_transform(scaled.reshape(1, -1), attr_names)[0]
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
        old = g.attrs[i] if hasattr(g, "attrs") and i < len(g.attrs) else {}
        d: dict[str, Any] = {}
        for attr in SUBG_DEFS[st].attrs:
            if attr == "dir":
                d["dir"] = float(old.get("dir", out.direction[i] if i < len(out.direction) else 0.0))
            else:
                d[attr] = float(physical[cursor]) if cursor < len(physical) else float("nan")
                cursor += 1
        out.attrs.append(d)
    info["physical_rescale_ok"] = True
    return out, info


def macro_graph_summary(g: SimpleNamespace) -> dict[str, Any]:
    return {
        "n_macro_nodes": len(getattr(g, "node_types", [])),
        "n_macro_edges": len(getattr(g, "edges", [])),
        "macro_node_types": ";".join(SubgType(int(t)).name for t in getattr(g, "node_types", [])),
        "macro_edges": ";".join(f"{int(u)}-{int(v)}" for u, v in getattr(g, "edges", [])),
        "macro_directions": ";".join(f"{float(d):.0f}" for d in getattr(g, "direction", [])),
    }




def _is_macro_graph_connected(n_nodes: int, edges: list[tuple[int, int]]) -> bool:
    """Return True iff the actual decoded macro graph is connected."""
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
    """Strict macro-level validity used by the optimizer.

    This is the final hard check corresponding to the new constraint-aware
    decoder workflow:
      1. the set of generated macro-nodes must admit at least one connected
         graph made only of physical macro-edges;
      2. the actual sampled macro-edges must all be physically compatible;
      3. the actual sampled macro graph must be connected.

    It does not replace QuLTRA.  It rejects topologies before the scalar CEM
    loss can prefer a numerically good but physically meaningless candidate.
    """
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
        # Edges are stored in the exact order generated by the decoder.
        # The edge-generation mask should have allowed this orientation.
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
        "is_physical_valid": is_valid,
        "physical_invalid_reasons": ";".join(reasons),
    }

def _fixed_slots_by_group(fixed_raw_obs: dict[str, float]) -> dict[str, dict[str, float]]:
    """Split user-fixed observables into f/chi/kappa groups."""
    groups = {"f": {}, "chi": {}, "kappa": {}}
    for slot, value in sorted(fixed_raw_obs.items()):
        if slot.startswith("f_"):
            groups["f"][slot] = float(value)
        elif slot.startswith("chi_"):
            groups["chi"][slot] = float(value)
        elif slot.startswith("kappa_"):
            groups["kappa"][slot] = float(value)
    return groups


def _sort_sim_observables(sim: dict[str, Any]) -> dict[str, Any]:
    """Sort simulated observables by increasing frequency.

    The same permutation is applied to chi and kappa, so chi_13 after sorting
    means the coupling between the 1st and 3rd sorted-frequency modes.
    """
    f = np.asarray(sim.get("frequencies", []), dtype=np.float64).ravel()
    out: dict[str, Any] = {"frequencies": f}
    if f.size == 0:
        out["chi"] = np.zeros((0, 0), dtype=np.float64)
        out["kappa"] = np.asarray([], dtype=np.float64)
        out["frequency_sort_perm"] = []
        return out

    perm = np.argsort(f)
    out["frequencies"] = f[perm]
    out["frequency_sort_perm"] = [int(i) for i in perm.tolist()]

    chi = np.asarray(sim.get("chi", np.zeros((0, 0))), dtype=np.float64)
    if chi.ndim == 2 and chi.shape[0] >= f.size and chi.shape[1] >= f.size:
        out["chi"] = chi[np.ix_(perm, perm)]
    else:
        out["chi"] = np.zeros((0, 0), dtype=np.float64)

    kappa = np.asarray(sim.get("kappa", []), dtype=np.float64).ravel()
    if kappa.size >= f.size:
        out["kappa"] = kappa[perm]
    else:
        out["kappa"] = np.asarray([], dtype=np.float64)

    return out


def _pred_value_for_slot(slot: str, sorted_obs: dict[str, Any]) -> float:
    if slot.startswith("f_"):
        i = int(slot.split("_", 1)[1]) - 1
        f = np.asarray(sorted_obs.get("frequencies", []), dtype=np.float64).ravel()
        return float(f[i]) if 0 <= i < len(f) else float("nan")

    if slot.startswith("kappa_"):
        i = int(slot.split("_", 1)[1]) - 1
        k = np.asarray(sorted_obs.get("kappa", []), dtype=np.float64).ravel()
        return float(k[i]) if 0 <= i < len(k) else float("nan")

    if slot.startswith("chi_"):
        ij = slot.split("_", 1)[1]
        if len(ij) == 2 and ij.isdigit():
            i, j = int(ij[0]) - 1, int(ij[1]) - 1
            chi = np.asarray(sorted_obs.get("chi", np.zeros((0, 0))), dtype=np.float64)
            if chi.ndim == 2 and 0 <= i < chi.shape[0] and 0 <= j < chi.shape[1]:
                return float(chi[i, j])
    return float("nan")


def compare_requested_observables(fixed_raw_obs: dict[str, float], sim: dict[str, Any]) -> dict[str, Any]:
    """Compare all user-requested f/chi/kappa slots against QuLTRA output.

    Frequencies are sorted ascending; chi and kappa are permuted with the same
    frequency ordering. Only user-specified slots contribute to the reported
    group errors and later to the optimization loss.
    """
    sorted_obs = _sort_sim_observables(sim)
    groups = _fixed_slots_by_group(fixed_raw_obs)
    out: dict[str, Any] = {
        "frequency_sort_perm": ";".join(map(str, sorted_obs.get("frequency_sort_perm", []))),
        "pred_freqs_sorted": ";".join(f"{x:.12g}" for x in np.asarray(sorted_obs.get("frequencies", []), dtype=np.float64).ravel()),
        "pred_kappa_sorted": ";".join(f"{x:.12g}" for x in np.asarray(sorted_obs.get("kappa", []), dtype=np.float64).ravel()),
    }

    # Keep the old frequency fields for backwards compatibility and plots.
    f_targets = [v for _, v in sorted(groups["f"].items(), key=lambda kv: int(kv[0].split("_")[1]))]
    f_pred = np.asarray(sorted_obs.get("frequencies", []), dtype=np.float64).ravel()
    old_freq = compare_frequencies(groups["f"], f_pred)
    out.update(old_freq)

    all_rel_errors: list[float] = []
    for group_name, slots in groups.items():
        rel_errors: list[float] = []
        abs_errors: list[float] = []
        for slot, target in sorted(slots.items()):
            pred = _pred_value_for_slot(slot, sorted_obs)
            target_v = float(target)
            abs_err = abs(pred - target_v) if math.isfinite(pred) else float("nan")
            rel_err = 100.0 * abs_err / max(abs(target_v), 1e-30) if math.isfinite(abs_err) else float("nan")

            out[f"target_{slot}"] = target_v
            out[f"pred_{slot}"] = pred
            out[f"abs_err_{slot}"] = abs_err
            out[f"rel_err_pct_{slot}"] = rel_err

            if math.isfinite(rel_err):
                rel_errors.append(float(rel_err))
                all_rel_errors.append(float(rel_err))
            if math.isfinite(abs_err):
                abs_errors.append(float(abs_err))

        out[f"requested_{group_name}_count"] = int(len(slots))
        out[f"valid_{group_name}_errors"] = int(len(rel_errors))
        out[f"mean_rel_err_pct_{group_name}"] = float(np.mean(rel_errors)) if rel_errors else float("nan")
        out[f"max_rel_err_pct_{group_name}"] = float(np.max(rel_errors)) if rel_errors else float("nan")
        out[f"mean_abs_err_{group_name}"] = float(np.mean(abs_errors)) if abs_errors else float("nan")
        out[f"max_abs_err_{group_name}"] = float(np.max(abs_errors)) if abs_errors else float("nan")

    out["mean_rel_err_pct_all_requested"] = float(np.mean(all_rel_errors)) if all_rel_errors else float("nan")
    out["max_rel_err_pct_all_requested"] = float(np.max(all_rel_errors)) if all_rel_errors else float("nan")
    return out


def compare_frequencies(target_freq_obs: dict[str, float], pred_freqs: np.ndarray) -> dict[str, Any]:
    """Compare only user-specified f_i slots."""
    targets = []
    for slot, value in sorted(target_freq_obs.items(), key=lambda kv: int(kv[0].split("_")[1])):
        targets.append(float(value))

    target_arr = np.sort(np.asarray(targets, dtype=np.float64))
    pred_arr = np.sort(np.asarray(pred_freqs, dtype=np.float64).ravel())
    n = len(target_arr)
    out: dict[str, Any] = {
        "n_target_freqs": int(n),
        "n_pred_freqs": int(len(pred_arr)),
        "target_freqs_sorted": ";".join(f"{x:.12g}" for x in target_arr),
        "pred_freqs_sorted": ";".join(f"{x:.12g}" for x in pred_arr),
    }
    if n == 0:
        out.update({
            "freq_match_ok": True,
            "freq_mae": 0.0,
            "freq_max_abs_error": 0.0,
            "freq_mean_rel_error_pct": 0.0,
            "freq_max_rel_error_pct": 0.0,
        })
        return out
    if len(pred_arr) < n:
        out.update({
            "freq_match_ok": False,
            "freq_mae": float("nan"),
            "freq_max_abs_error": float("nan"),
            "freq_mean_rel_error_pct": float("nan"),
            "freq_max_rel_error_pct": float("nan"),
        })
        return out

    pred_use = pred_arr[:n]
    abs_err = np.abs(pred_use - target_arr)
    rel = 100.0 * abs_err / np.maximum(np.abs(target_arr), 1e-30)
    out.update({
        "freq_match_ok": True,
        "matched_pred_freqs_sorted": ";".join(f"{x:.12g}" for x in pred_use),
        "freq_mae": float(np.mean(abs_err)),
        "freq_max_abs_error": float(np.max(abs_err)),
        "freq_mean_rel_error_pct": float(np.mean(rel)),
        "freq_max_rel_error_pct": float(np.max(rel)),
    })
    for i in range(n):
        out[f"target_f_{i+1}"] = float(target_arr[i])
        out[f"pred_f_{i+1}"] = float(pred_use[i])
        out[f"abs_err_f_{i+1}"] = float(abs_err[i])
        out[f"rel_err_pct_f_{i+1}"] = float(rel[i])
    return out


def _finite_or_nan(v: Any) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else float("nan")
    except Exception:
        return float("nan")


def _group_loss(row: dict[str, Any], group: str, reduction: str) -> float:
    """Return group loss from already-computed requested observable errors."""
    if int(row.get(f"requested_{group}_count", 0) or 0) <= 0:
        return 0.0

    key = f"{reduction}_rel_err_pct_{group}"
    value = _finite_or_nan(row.get(key))
    if math.isfinite(value):
        return float(value)

    # If requested but not computable, punish it strongly but less than total QuLTRA failure.
    return float("nan")


def candidate_loss(row: dict[str, Any], args: argparse.Namespace) -> float:
    """Scalar loss for CEM. Lower is better.

    The loss includes every observable explicitly fixed by the user:
      - f_i through frequency error;
      - chi_ij through chi error;
      - kappa_i through kappa error.

    By default each group contributes its max relative error among requested
    slots. This is stricter than mean and avoids hiding a single bad chi/kappa.
    """
    if not row.get("is_physical_valid", False):
        return float(getattr(args, "physical_invalid_penalty", args.fail_penalty))

    if not row.get("ok_qultra", False):
        return float(args.fail_penalty)

    n_target = int(row.get("n_target_freqs", 0) or 0)
    n_pred = int(row.get("n_pred_freqs", 0) or 0)
    if n_target > 0 and n_pred < n_target:
        return float(args.missing_mode_penalty)

    missing_requested = 0
    terms: list[tuple[str, float, float]] = []
    for group, weight in [
        ("f", float(args.freq_loss_weight)),
        ("chi", float(args.chi_loss_weight)),
        ("kappa", float(args.kappa_loss_weight)),
    ]:
        if weight == 0.0:
            continue
        if int(row.get(f"requested_{group}_count", 0) or 0) <= 0:
            continue
        value = _group_loss(row, group, args.loss_reduction)
        if math.isfinite(value):
            terms.append((group, weight, float(value)))
        else:
            missing_requested += int(row.get(f"requested_{group}_count", 0) or 0)

    if not terms and missing_requested == 0:
        # No requested observable was scoreable; this should only happen when
        # the user did not specify any f/chi/kappa slot.
        return float(args.fail_penalty)

    weighted_values = [w * v for _, w, v in terms]
    denom = sum(abs(w) for _, w, _ in terms)

    if args.loss_aggregation == "worst_plus_mean":
        # Main term: the worst requested group dominates.
        # Refinement term: a small weighted mean still rewards improving all groups.
        worst = max(weighted_values, default=0.0)
        mean = sum(weighted_values) / max(denom, 1e-12)
        loss = worst + float(args.loss_mean_weight) * mean
    elif args.loss_aggregation == "weighted_mean":
        loss = sum(weighted_values) / max(denom, 1e-12)
    elif args.loss_aggregation == "weighted_sum":
        loss = sum(weighted_values)
    elif args.loss_aggregation == "max":
        loss = max(weighted_values, default=0.0)
    else:
        raise ValueError(f"Unknown loss_aggregation={args.loss_aggregation}")

    if missing_requested:
        loss += float(args.missing_observable_penalty) * float(missing_requested)

    # Extra penalties are optional; useful for steering output complexity.
    if args.target_modes > 0:
        loss += float(args.mode_count_weight) * abs(n_pred - int(args.target_modes))
    elif n_target > 0:
        loss += float(args.mode_count_weight) * abs(n_pred - n_target)

    return float(loss)


def print_best_candidate(row: dict[str, Any], prefix: str = "BEST") -> None:
    print("\n" + "=" * 90)
    print(prefix)
    print("=" * 90)
    print(f"sample_id: {row.get('sample_id')}")
    print(f"iteration: {row.get('iteration')}  candidate: {row.get('candidate_id')}")
    print(f"loss: {row.get('loss')}")
    print(f"ok_qultra: {row.get('ok_qultra')}  stage={row.get('qultra_stage')}  err={row.get('qultra_error')}")
    print(f"macro nodes: {row.get('macro_node_types')}")
    print(f"macro edges: {row.get('macro_edges')}")
    print(f"param attrs: {row.get('param_attr_names')}")
    print(f"target freqs: {row.get('target_freqs_sorted')}")
    print(f"pred freqs:   {row.get('pred_freqs_sorted')}")
    print(f"matched pred: {row.get('matched_pred_freqs_sorted')}")
    print(f"freq mean/max rel err %:  {row.get('mean_rel_err_pct_f')} / {row.get('max_rel_err_pct_f')}")
    print(f"chi mean/max rel err %:   {row.get('mean_rel_err_pct_chi')} / {row.get('max_rel_err_pct_chi')}")
    print(f"kappa mean/max rel err %: {row.get('mean_rel_err_pct_kappa')} / {row.get('max_rel_err_pct_kappa')}")
    print(f"all requested mean/max %: {row.get('mean_rel_err_pct_all_requested')} / {row.get('max_rel_err_pct_all_requested')}")
    print("=" * 90)


def _graph_ns_to_nx(g: SimpleNamespace, physical: bool = False):
    G = nx.Graph()
    for i, st_int in enumerate(getattr(g, "node_types", [])):
        st = SubgType(int(st_int))
        attrs = {}
        if hasattr(g, "attrs") and i < len(g.attrs):
            attrs = dict(g.attrs[i] or {})
        label = f"{i}: {st.name}"
        if "dir" in SUBG_DEFS[st].attrs:
            d = attrs.get("dir", getattr(g, "direction", [0.0] * len(getattr(g, "node_types", [])))[i])
            label += f"\ndir={_fmt_float(d)}"
        if physical and attrs:
            shown = []
            for k, v in attrs.items():
                if k == "dir":
                    continue
                shown.append(f"{k}={_fmt_float(v, 3)}")
            if shown:
                label += "\n" + "\n".join(shown[:4])
                if len(shown) > 4:
                    label += "\n..."
        G.add_node(i, label=label, subg_type=st)
    for u, v in getattr(g, "edges", []):
        u, v = int(u), int(v)
        if u != v:
            G.add_edge(u, v)
    return G


def _topology_to_nx_debug(topo):
    G = nx.Graph()
    for node in getattr(topo, "_nodes", []):
        st = node.subg_type
        attrs = dict(getattr(node, "attrs", {}) or {})
        label = f"{int(node.node_id)}: {st.name}"
        shown = []
        for k, v in attrs.items():
            if k == "dir":
                continue
            shown.append(f"{k}={_fmt_float(v, 3)}")
        if shown:
            label += "\n" + "\n".join(shown[:3])
            if len(shown) > 3:
                label += "\n..."
        G.add_node(int(node.node_id), label=label, subg_type=st)
    for u, v in getattr(topo, "_edges", []):
        u, v = int(u), int(v)
        if u != v:
            G.add_edge(u, v)
    return G


def _draw_nx_graph(ax, G, title: str):
    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "empty graph", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
        return
    try:
        colors = [SUBG_DEFS[G.nodes[n]["subg_type"]].color for n in G.nodes]
    except Exception:
        colors = ["lightgray" for _ in G.nodes]
    pos = nx.spring_layout(G, seed=7, k=1.2) if G.number_of_nodes() > 2 else nx.shell_layout(G)
    nx.draw_networkx_edges(G, pos, ax=ax, width=1.4, alpha=0.75)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=1250,
                           edgecolors="black", linewidths=0.7, alpha=0.95)
    nx.draw_networkx_labels(G, pos, labels={n: G.nodes[n]["label"] for n in G.nodes},
                            ax=ax, font_size=5.7, font_color="black", font_weight="bold")
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def save_best_plot(best_record: dict[str, Any], out_dir: Path) -> None:
    if not _HAS_NX or not best_record:
        return
    g_phys = best_record.get("_g_physical")
    prim = best_record.get("_primitive_topology")
    if g_phys is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    G_macro = _graph_ns_to_nx(g_phys, physical=True)
    _draw_nx_graph(
        axes[0],
        G_macro,
        f"Best macro graph\nloss={_fmt_float(best_record.get('loss'))}, err={_fmt_float(best_record.get('freq_mean_rel_error_pct'))}%",
    )
    if prim is not None:
        G_prim = _topology_to_nx_debug(prim)
        _draw_nx_graph(axes[1], G_prim, "Best expanded primitive graph")
    else:
        axes[1].text(0.5, 0.5, "No primitive expansion", ha="center", va="center", transform=axes[1].transAxes)
        axes[1].axis("off")
    fig.tight_layout()
    path = out_dir / "best_candidate_networkx.png"
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {path}")


def save_loss_curve(iteration_summaries: list[dict[str, Any]], out_dir: Path) -> None:
    if not iteration_summaries:
        return
    x = [int(r["iteration"]) for r in iteration_summaries]
    best = [float(r["best_loss"]) for r in iteration_summaries]
    med = [float(r["median_loss"]) for r in iteration_summaries]
    ok = [float(r["qultra_ok_percent"]) for r in iteration_summaries]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(x, best, marker="o", label="Best loss")
    ax.plot(x, med, marker="o", label="Median loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    ax2 = ax.twinx()
    ax2.plot(x, ok, marker="s", linestyle="--", label="QuLTRA OK %")
    ax2.set_ylabel("QuLTRA OK (%)")
    ax2.set_ylim(0, 100)
    ax2.legend(loc="upper right")
    fig.tight_layout()
    path = out_dir / "optimization_loss_curve.png"
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {path}")


QUANTUM_METAL_NOTICE = (
    "Quantum-Metal compulsory.\n"
    "This is only a layout-zero design generated from the lumped circuit. "
    "Electromagnetic optimization and validation are required before fabrication."
)


def _quantum_metal_compatibility_check(primitive_topology: Any) -> None:
    """Reject primitive graphs that graph2metal cannot represent faithfully."""
    nodes = list(getattr(primitive_topology, "_nodes", []))
    if not nodes:
        raise RuntimeError(
            "Quantum Metal generation requires the expanded primitive CQEDTopology, "
            "but the best candidate has no primitive nodes."
        )

    unsupported = []
    for node in nodes:
        kind = getattr(getattr(node, "subg_type", None), "name", str(getattr(node, "subg_type", "")))
        if str(kind).upper() == "I_COUPLER":
            unsupported.append(int(getattr(node, "node_id", -1)))
    if unsupported:
        raise RuntimeError(
            "Quantum Metal layout-zero generation is compulsory because --quantum-metal "
            "was set, but graph2metal does not yet implement inductive couplers. "
            f"Unsupported I_COUPLER node IDs: {unsupported}."
        )


def generate_quantum_metal_from_best(
    best_rich: dict[str, Any] | None,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, Any]:
    """Create the layout-zero design for the best G-VAE/QuLTRA candidate."""
    if not args.quantum_metal:
        return {"enabled": False, "status": "DISABLED"}
    if not best_rich:
        raise RuntimeError(
            "Quantum Metal layout-zero generation is compulsory because --quantum-metal "
            "was set, but no best candidate is available."
        )

    primitive_topology = best_rich.get("_primitive_topology")
    if primitive_topology is None:
        raise RuntimeError(
            "Quantum Metal layout-zero generation is compulsory because --quantum-metal "
            "was set, but primitive expansion failed for the best candidate."
        )
    _quantum_metal_compatibility_check(primitive_topology)

    try:
        from graph2metal.graph_model import SynthesisConfig
        from graph2metal.plot_layout import generate_layout_zero
    except Exception as exc:
        raise RuntimeError(
            "Quantum Metal layout-zero generation is compulsory because --quantum-metal "
            "was set, but the integrated src/graph2metal package could not be imported."
        ) from exc

    layout_dir = (
        Path(args.quantum_metal_out_dir)
        if args.quantum_metal_out_dir
        else out_dir / "quantum_metal_layout_zero"
    )
    config = SynthesisConfig(
        chip_width_um=float(args.quantum_metal_chip_width_um),
        chip_height_um=float(args.quantum_metal_chip_height_um),
    )

    try:
        result = generate_layout_zero(
            primitive_topology,
            layout_dir,
            config=config,
            strict=not bool(args.quantum_metal_non_strict),
            strict_geometry_check=not bool(args.quantum_metal_allow_qgeometry_conflicts),
            plan_only=False,
        )
    except Exception as exc:
        raise RuntimeError(
            "Quantum Metal layout-zero generation is compulsory because --quantum-metal "
            "was set, and graph2metal failed to produce the design."
        ) from exc

    skipped = list(result.translation.graph.skipped_nodes)
    if skipped:
        skipped_ids = [int(node.node_id) for node in skipped]
        raise RuntimeError(
            "Quantum Metal layout-zero generation is compulsory, but graph2metal skipped "
            f"unsupported circuit nodes: {skipped_ids}."
        )
    if result.layout_pdf is None:
        raise RuntimeError(
            "Quantum Metal layout-zero generation was requested, but no layout_zero.pdf was produced."
        )

    print("\n" + QUANTUM_METAL_NOTICE)
    print(f"Quantum Metal layout zero: {result.layout_pdf}")
    print(f"Graph2metal layout plan:   {result.layout_plan_json}")
    print(f"Quantum Metal manifest:   {result.render_json}")

    return {
        "enabled": True,
        "status": "PASS",
        "notice": QUANTUM_METAL_NOTICE.replace("\n", " "),
        "output_dir": str(layout_dir),
        "source_graph_json": str(result.source_graph_json),
        "layout_plan_json": str(result.layout_plan_json),
        "layout_pdf": str(result.layout_pdf),
        "render_json": str(result.render_json),
        "coupler_zoom_pdfs": {
            str(key): str(value)
            for key, value in (result.artifacts.coupler_zoom_pdfs if result.artifacts else {}).items()
        },
        "layout_valid": bool(result.translation.valid),
        "synthesis_attempt": int(result.translation.plan.synthesis_attempt),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optimize latent z with QuLTRA in the loop.")
    p.add_argument("--ckpt", default=str(REPO_ROOT / "best_vae.pt"))
    p.add_argument("--out-dir", default=str(REPO_ROOT / "pres_plot" / "latent_qultra_optimization"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    # User observables are added dynamically from OBS_SLOTS.
    add_observable_arguments(p)

    p.add_argument("--population", type=int, default=64)
    p.add_argument("--elite", type=int, default=8)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--init-sigma", type=float, default=1.0)
    p.add_argument("--min-sigma", type=float, default=0.05)
    p.add_argument("--sigma-decay", type=float, default=0.90)
    p.add_argument("--cem-momentum", type=float, default=0.65, help="0 = jump to elite mean, 1 = keep old mean.")
    p.add_argument("--z-clip", type=float, default=4.0, help="Clip optimized latent coordinates. 0 disables.")
    p.add_argument("--initial-random-completions", type=int, default=8, help="Try multiple random completions before choosing z0.")

    p.add_argument("--random-scaled-std", type=float, default=1.0)
    p.add_argument("--random-scaled-clip", type=float, default=2.5)
    p.add_argument("--include-all-slots", action="store_true", help="Randomly fill every missing OBS_SLOT, not only slots compatible with specified frequency dimension.")

    p.add_argument("--stochastic", action="store_true", help="Use stochastic decoder. Default deterministic.")
    p.add_argument("--guidance-strength", type=float, default=0.0)
    p.add_argument("--required-transmon", type=int, default=0)
    p.add_argument("--required-resonator", type=int, default=0)
    p.add_argument("--required-coupler", type=int, default=0)
    p.add_argument("--required-feedline", type=int, default=0)
    p.add_argument("--required-inductor", type=int, default=0)

    p.add_argument("--f-min", type=float, default=1.0)
    p.add_argument("--f-max", type=float, default=9.0)
    p.add_argument("--target-modes", type=int, default=0, help="Optional target number of simulated modes. 0 uses number of fixed frequencies.")
    p.add_argument("--fail-penalty", type=float, default=1e6)
    p.add_argument("--physical-invalid-penalty", type=float, default=1e6,
                   help="CEM loss assigned to candidates failing final macro/physical validity checks.")
    p.add_argument("--missing-mode-penalty", type=float, default=1e5)
    p.add_argument("--mode-count-weight", type=float, default=10.0)
    p.add_argument("--freq-loss-weight", type=float, default=1.0, help="Weight of requested f_i errors in the CEM loss.")
    p.add_argument("--chi-loss-weight", type=float, default=1.0, help="Weight of requested chi_ij errors in the CEM loss.")
    p.add_argument("--kappa-loss-weight", type=float, default=1.0, help="Weight of requested kappa_i errors in the CEM loss.")
    p.add_argument("--loss-reduction", choices=["mean", "max"], default="max", help="Use mean or max relative error inside each requested group.")
    p.add_argument("--loss-aggregation", choices=["worst_plus_mean", "weighted_mean", "weighted_sum", "max"], default="worst_plus_mean", help="How to combine f/chi/kappa group losses. Default: max group error + mean_weight * weighted mean.")
    p.add_argument("--loss-mean-weight", type=float, default=0.2, help="Only used with --loss-aggregation worst_plus_mean: loss = max(weighted group errors) + this * weighted mean.")
    p.add_argument("--missing-observable-penalty", type=float, default=1e4, help="Penalty per requested chi/kappa/f slot that QuLTRA cannot provide.")

    qm = p.add_argument_group("optional Quantum Metal layout-zero output")
    qm.add_argument(
        "--quantum-metal",
        action="store_true",
        default=False,
        help=(
            "Generate a compulsory Quantum Metal layout-zero design for the best "
            "expanded lumped graph. Default: off."
        ),
    )
    qm.add_argument(
        "--quantum-metal-out-dir",
        default=None,
        help="Output directory. Default: <out-dir>/quantum_metal_layout_zero.",
    )
    qm.add_argument("--quantum-metal-chip-width-um", type=float, default=12000.0)
    qm.add_argument("--quantum-metal-chip-height-um", type=float, default=10000.0)
    qm.add_argument(
        "--quantum-metal-non-strict",
        action="store_true",
        help="Allow a graph2metal LayoutPlan with reported planner violations.",
    )
    qm.add_argument(
        "--quantum-metal-allow-qgeometry-conflicts",
        action="store_true",
        help="Do not fail on post-build Quantum Metal copper-overlap checks.",
    )

    p.add_argument("--print-every-iter", action="store_true")
    p.add_argument("--print-best", action="store_true")
    p.add_argument("--plot-best", action="store_true")
    return p.parse_args()


@torch.no_grad()
def decode_and_evaluate_population(
    model,
    qinf,
    qu,
    z_batch: torch.Tensor,
    param_scaler,
    fixed_raw_obs: dict[str, float],
    required_primitives: dict[str, int],
    args: argparse.Namespace,
    iteration: int,
    start_sample_id: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return serializable rows and rich records with graph objects."""
    if required_primitives and args.guidance_strength > 0:
        graphs = model.decoder.decode(
            z_batch,
            stochastic=bool(args.stochastic),
            required_primitives=required_primitives,
            guidance_strength=float(args.guidance_strength),
        )
    else:
        graphs = model.decoder.decode(z_batch, stochastic=bool(args.stochastic))

    attrs_per_graph = model.param_decoder.predict(z_batch, graphs)
    rows: list[dict[str, Any]] = []
    rich: list[dict[str, Any]] = []

    for k, (g, attrs) in enumerate(zip(graphs, attrs_per_graph)):
        sample_id = start_sample_id + k
        for i, st_int in enumerate(getattr(g, "node_types", [])):
            st = SubgType(int(st_int))
            if "dir" in SUBG_DEFS[st].attrs:
                attrs[i]["dir"] = float(getattr(g, "direction", [0.0] * len(g.node_types))[i])
        g.attrs = attrs

        row: dict[str, Any] = {
            "sample_id": int(sample_id),
            "iteration": int(iteration),
            "candidate_id": int(k),
            "ok_qultra": False,
            "qultra_error": "",
            "qultra_stage": "",
            "required_primitives_json": json.dumps(required_primitives, sort_keys=True),
            "guidance_strength": float(args.guidance_strength),
            "stochastic": bool(args.stochastic),
        }
        row.update(macro_graph_summary(g))
        row.update(analyze_macro_physical_constraints(g))

        g_phys = None
        primitive_topology = None
        try:
            g_phys, rescale_info = graph_with_physical_attrs(g, param_scaler)
            row.update(rescale_info)
            if not rescale_info.get("physical_rescale_ok"):
                raise RuntimeError(rescale_info.get("physical_rescale_error", "physical rescale failed"))

            pred_topo = qinf._graph_ns_to_cqed_topology(g_phys, name=f"latent_opt_{iteration}_{k}")
            try:
                primitive_topology = expand_topology(pred_topo, validate=False)
                row["primitive_expansion_ok"] = True
                row["n_primitive_nodes"] = len(getattr(primitive_topology, "_nodes", []))
                row["n_primitive_edges"] = len(getattr(primitive_topology, "_edges", []))
            except Exception as exc:
                row["primitive_expansion_ok"] = False
                row["primitive_expansion_error"] = f"{type(exc).__name__}: {exc}"
                row["is_physical_valid"] = False
                prev_reasons = str(row.get("physical_invalid_reasons", "") or "")
                row["physical_invalid_reasons"] = ";".join([x for x in [prev_reasons, "primitive_expansion_failed"] if x])

            sim = qinf._simulate_topology_arrays(pred_topo, qu, args.f_min, args.f_max)
            row["ok_qultra"] = bool(sim.get("ok"))
            row["qultra_stage"] = str(sim.get("stage", ""))
            row["qultra_error"] = str(sim.get("error") or "")
            row["n_expected_modes"] = int(sim.get("n_expected", 0) or 0)
            if sim.get("ok"):
                row.update(compare_requested_observables(fixed_raw_obs, sim))
            else:
                row.update(compare_requested_observables(fixed_raw_obs, {}))
        except Exception as exc:
            row["ok_qultra"] = False
            row["qultra_error"] = f"{type(exc).__name__}: {exc}"
            row["is_physical_valid"] = False
            prev_reasons = str(row.get("physical_invalid_reasons", "") or "")
            row["physical_invalid_reasons"] = ";".join([x for x in [prev_reasons, f"evaluation_exception:{type(exc).__name__}"] if x])
            row.update(compare_requested_observables(fixed_raw_obs, {}))

        row["loss"] = candidate_loss(row, args)
        rows.append(row)
        rich.append({
            "row": dict(row),
            "_g_scaled": g,
            "_g_physical": g_phys,
            "_primitive_topology": primitive_topology,
        })

    return rows, rich


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.quantum_metal:
        print(QUANTUM_METAL_NOTICE)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    fixed_raw_obs = user_observables_from_args(args)
    target_freq_obs = {k: v for k, v in fixed_raw_obs.items() if k.startswith("f_")}
    target_chi_obs = {k: v for k, v in fixed_raw_obs.items() if k.startswith("chi_")}
    target_kappa_obs = {k: v for k, v in fixed_raw_obs.items() if k.startswith("kappa_")}

    required_primitives = {}
    for name, value in [
        ("transmon", args.required_transmon),
        ("resonator", args.required_resonator),
        ("coupler", args.required_coupler),
        ("feedline", args.required_feedline),
        ("inductor", args.required_inductor),
    ]:
        if int(value) > 0:
            required_primitives[name] = int(value)

    qinf = import_qultra_inference()
    print(f"Repo root: {REPO_ROOT}")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Fixed raw observables: {json.dumps(fixed_raw_obs, sort_keys=True)}")
    print(f"Target frequency slots: {json.dumps(target_freq_obs, sort_keys=True)}")
    print(f"Target chi slots: {json.dumps(target_chi_obs, sort_keys=True)}")
    print(f"Target kappa slots: {json.dumps(target_kappa_obs, sort_keys=True)}")
    print(f"Required primitives: {json.dumps(required_primitives, sort_keys=True)}")

    model, scalers, cfg = qinf.load_checkpoint(args.ckpt, device)
    model.eval()

    if "__global__" not in scalers:
        raise RuntimeError(
            "This script requires a checkpoint with global_scaler. "
            f"Available scaler keys: {list(scalers.keys())}"
        )
    global_scaler = scalers["__global__"]
    obs_scaler = global_scaler.obs_scaler
    param_scaler = global_scaler.param_scaler
    qu = qinf.import_qultra()

    # Build several random completions and choose the initial z0 that gives the
    # best immediate decoded candidate.
    init_rows_all: list[dict[str, Any]] = []
    init_rich_all: list[dict[str, Any]] = []
    z0_candidates = []
    obs_debug_rows = []

    for c in range(max(1, int(args.initial_random_completions))):
        vals_scaled, mask, raw_debug, scaled_debug = build_complete_scaled_observables(
            fixed_raw_obs=fixed_raw_obs,
            obs_scaler=obs_scaler,
            rng=rng,
            random_scaled_std=args.random_scaled_std,
            random_scaled_clip=args.random_scaled_clip,
            include_all_slots=bool(args.include_all_slots),
        )
        obs_scaled_t = torch.tensor(vals_scaled, dtype=torch.float32, device=device).unsqueeze(0)
        obs_mask_t = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            z_s, _, _ = model.spec_encoder.encode(obs_scaled_t, obs_mask_t)
        z0_candidates.append(z_s[0].detach().clone())
        obs_debug_rows.append({
            "completion_id": int(c),
            "raw_observables_json": json.dumps(raw_debug, sort_keys=True),
            "scaled_observables_json": json.dumps(scaled_debug, sort_keys=True),
        })
        rows, rich = decode_and_evaluate_population(
            model=model,
            qinf=qinf,
            qu=qu,
            z_batch=z_s,
            param_scaler=param_scaler,
            fixed_raw_obs=fixed_raw_obs,
            required_primitives=required_primitives,
            args=args,
            iteration=-1,
            start_sample_id=c,
        )
        for r in rows:
            r["initial_completion_id"] = int(c)
        init_rows_all.extend(rows)
        init_rich_all.extend(rich)

    best_init_idx = int(np.argmin([float(r["loss"]) for r in init_rows_all])) if init_rows_all else 0
    best_init_row = init_rows_all[best_init_idx]
    best_completion_id = int(best_init_row.get("initial_completion_id", 0))
    mu = z0_candidates[best_completion_id].detach().clone()

    print(f"Initial best completion: {best_completion_id}, loss={best_init_row.get('loss')}")
    if args.print_best:
        print_best_candidate(best_init_row, prefix="BEST INITIAL")

    sigma = torch.full_like(mu, float(args.init_sigma))
    if args.z_clip and args.z_clip > 0:
        mu = torch.clamp(mu, -float(args.z_clip), float(args.z_clip))

    all_rows: list[dict[str, Any]] = []
    all_rows.extend(init_rows_all)
    best_rich = init_rich_all[best_init_idx] if init_rich_all else None
    best_row = dict(best_init_row) if best_init_row else None
    sample_counter = len(init_rows_all)

    iteration_summaries: list[dict[str, Any]] = []

    for it in range(int(args.iters)):
        eps = torch.randn(int(args.population), mu.numel(), device=device)
        z_pop = mu.unsqueeze(0) + eps * sigma.unsqueeze(0)
        if args.z_clip and args.z_clip > 0:
            z_pop = torch.clamp(z_pop, -float(args.z_clip), float(args.z_clip))

        rows, rich = decode_and_evaluate_population(
            model=model,
            qinf=qinf,
            qu=qu,
            z_batch=z_pop,
            param_scaler=param_scaler,
            fixed_raw_obs=fixed_raw_obs,
            required_primitives=required_primitives,
            args=args,
            iteration=it,
            start_sample_id=sample_counter,
        )
        sample_counter += len(rows)
        all_rows.extend(rows)

        losses = np.asarray([float(r["loss"]) for r in rows], dtype=np.float64)
        order = np.argsort(losses)
        elite_n = max(1, min(int(args.elite), len(rows)))
        elite_idx = order[:elite_n]
        elite_z = z_pop[torch.tensor(elite_idx, dtype=torch.long, device=device)]

        elite_mean = elite_z.mean(dim=0)
        elite_std = elite_z.std(dim=0).clamp(min=float(args.min_sigma))

        momentum = float(args.cem_momentum)
        mu = momentum * mu + (1.0 - momentum) * elite_mean
        sigma = torch.maximum(
            torch.tensor(float(args.min_sigma), device=device),
            float(args.sigma_decay) * (momentum * sigma + (1.0 - momentum) * elite_std),
        )
        if args.z_clip and args.z_clip > 0:
            mu = torch.clamp(mu, -float(args.z_clip), float(args.z_clip))

        iter_best_i = int(order[0])
        iter_best_row = rows[iter_best_i]
        iter_best_rich = rich[iter_best_i]
        if best_row is None or float(iter_best_row["loss"]) < float(best_row["loss"]):
            best_row = dict(iter_best_row)
            best_rich = iter_best_rich

        ok_count = sum(1 for r in rows if r.get("ok_qultra"))
        comparable_count = sum(1 for r in rows if r.get("freq_match_ok"))
        iteration_summary = {
            "iteration": int(it),
            "best_loss": float(np.min(losses)),
            "median_loss": float(np.median(losses)),
            "mean_loss": float(np.mean(losses[np.isfinite(losses)])),
            "global_best_loss": float(best_row["loss"]) if best_row else float("nan"),
            "qultra_ok": int(ok_count),
            "qultra_ok_percent": 100.0 * ok_count / max(len(rows), 1),
            "freq_comparable": int(comparable_count),
            "freq_comparable_percent": 100.0 * comparable_count / max(len(rows), 1),
            "sigma_mean": float(sigma.mean().detach().cpu().item()),
            "sigma_max": float(sigma.max().detach().cpu().item()),
            "iter_best_macro_node_types": iter_best_row.get("macro_node_types", ""),
            "iter_best_qultra_error": iter_best_row.get("qultra_error", ""),
            "iter_best_mean_rel_err_pct_f": iter_best_row.get("mean_rel_err_pct_f", float("nan")),
            "iter_best_max_rel_err_pct_f": iter_best_row.get("max_rel_err_pct_f", float("nan")),
            "iter_best_mean_rel_err_pct_chi": iter_best_row.get("mean_rel_err_pct_chi", float("nan")),
            "iter_best_max_rel_err_pct_chi": iter_best_row.get("max_rel_err_pct_chi", float("nan")),
            "iter_best_mean_rel_err_pct_kappa": iter_best_row.get("mean_rel_err_pct_kappa", float("nan")),
            "iter_best_max_rel_err_pct_kappa": iter_best_row.get("max_rel_err_pct_kappa", float("nan")),
            "iter_best_mean_rel_err_pct_all_requested": iter_best_row.get("mean_rel_err_pct_all_requested", float("nan")),
            "iter_best_max_rel_err_pct_all_requested": iter_best_row.get("max_rel_err_pct_all_requested", float("nan")),
        }
        iteration_summaries.append(iteration_summary)

        if args.print_every_iter:
            print(
                f"iter={it:03d} best={iteration_summary['best_loss']:.6g} "
                f"global={iteration_summary['global_best_loss']:.6g} "
                f"ok={iteration_summary['qultra_ok_percent']:.1f}% "
                f"sigma={iteration_summary['sigma_mean']:.3g} "
                f"topo={iteration_summary['iter_best_macro_node_types']}"
            )

    write_csv(out_dir / "latent_qultra_optimization_candidates.csv", all_rows)
    write_csv(out_dir / "latent_qultra_optimization_iterations.csv", iteration_summaries)
    write_csv(out_dir / "latent_qultra_optimization_initial_completions.csv", obs_debug_rows)

    quantum_metal_result = generate_quantum_metal_from_best(best_rich, args, out_dir)

    # Remove non-serializable objects from best rich record before JSON.
    best_for_json = dict(best_row or {})
    summary = {
        "checkpoint": str(args.ckpt),
        "device": str(device),
        "fixed_raw_observables": fixed_raw_obs,
        "target_frequency_slots": target_freq_obs,
        "target_chi_slots": target_chi_obs,
        "target_kappa_slots": target_kappa_obs,
        "required_primitives": required_primitives,
        "loss_weights": {
            "f": float(args.freq_loss_weight),
            "chi": float(args.chi_loss_weight),
            "kappa": float(args.kappa_loss_weight),
            "reduction": args.loss_reduction,
            "aggregation": args.loss_aggregation,
            "mean_weight": float(args.loss_mean_weight),
        },
        "population": int(args.population),
        "elite": int(args.elite),
        "iters": int(args.iters),
        "initial_random_completions": int(args.initial_random_completions),
        "best": best_for_json,
        "quantum_metal": quantum_metal_result,
        "definitions": {
            "optimization": "Cross-Entropy Method in latent space around z0 = spec_encoder(obs).",
            "failure_handling": "QuLTRA failures receive fail_penalty and are kept in CSV with error reason.",
            "objective": "User-specified f_i, chi_ij, and kappa_i slots are compared when available from QuLTRA. Frequencies/chi/kappa are sorted/permuted consistently by increasing frequency. Default loss is worst requested group error plus a small weighted-mean refinement, so one bad kappa/chi cannot be hidden by good frequencies. Optional mode-count penalty is added.",
            "missing_observables": "User-specified observables are raw physical values and are scaled internally. Missing observables are sampled directly in scaled space.",
            "quantum_metal_layout_zero": "Optional graph2metal output is only a design zero generated from the lumped circuit and requires electromagnetic optimization and validation before fabrication.",
        },
    }
    (out_dir / "latent_qultra_optimization_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.print_best and best_row:
        print_best_candidate(best_row, prefix="FINAL BEST")

    if args.plot_best and best_rich:
        # Merge graph objects into row for plotting.
        plot_record = dict(best_rich)
        plot_record.update(best_rich["row"])
        save_best_plot(plot_record, out_dir)

    save_loss_curve(iteration_summaries, out_dir)

    print("\nDone.")
    print(f"Wrote: {out_dir / 'latent_qultra_optimization_candidates.csv'}")
    print(f"Wrote: {out_dir / 'latent_qultra_optimization_iterations.csv'}")
    print(f"Wrote: {out_dir / 'latent_qultra_optimization_initial_completions.csv'}")
    print(f"Wrote: {out_dir / 'latent_qultra_optimization_summary.json'}")
    if best_row:
        print_best_candidate(best_row, prefix="BEST SUMMARY")


if __name__ == "__main__":
    main()
