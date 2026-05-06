"""Dataset schema for the cQED loaders.

This module has all the function to read the datasets and to build both
observables and circuit for each dataset.

How to add a new dataset?

# 0. Add any new observable that is needed.
# 1. Add the raw data file to the data/ directory.
# 2. Write a topology builder (_build_<name>).
# 3. Write a row parser (_parse_row_<name>), add to ROW_PARSERS.
# 4. Write an observable parser (obs_<name>).
# 5. Add a DatasetDef entry to DATASETS.
# 6. Add the observable parser to OBS_PARSERS.

"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, List

import numpy as np

from circuit2graph import CQEDTopology, SubgType


# ===========================================================================
# Observable vocabulary
# ===========================================================================
#
# HOW TO ADD A NEW OBSERVABLE
# ----------------------------
# 1. Append the new name to OBS_SLOTS (keep the existing order).
# 2. N_OBS_SLOTS and OBS_IDX are derived automatically.
# 3. Update or add a parser function below that populates the new slot.
# 4. Register the parser in DATASETS / OBS_PARSERS below.

OBS_SLOTS: list[str] = [
    "f_1",
    "f_2",
    "kappa_1",
    "kappa_2",
    "chi_11",
    "chi_22",
    "chi_12",
]

N_OBS_SLOTS: int             = len(OBS_SLOTS)
OBS_IDX:     dict[str, int]  = {name: i for i, name in enumerate(OBS_SLOTS)}


def _empty_obs() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.zeros(N_OBS_SLOTS, dtype=np.float64),
        np.zeros(N_OBS_SLOTS, dtype=np.float64),
    )


# ===========================================================================
# Per-dataset observable parsers
# ===========================================================================
#
# HOW TO ADD A NEW OBSERVABLE PARSER
# -------------------------------------
# Write a function with signature:
#
#     def obs_<name>(freqs_res, freqs_qb, kappa, chi, **kwargs)
#         -> tuple[np.ndarray, np.ndarray]:   # (obs_vals, obs_mask)
#
# Both arrays have length N_OBS_SLOTS.  Set mask[i] = 1.0 only for slots
# that are actually measured in this dataset.

def obs_qubit(
    freqs_res: list, freqs_qb: list, kappa: list, chi: list, **_
) -> tuple[np.ndarray, np.ndarray]:
    """
    Single transmon.
    Active slots: f_1, chi_11
    """
    vals, mask = _empty_obs()
    if freqs_qb:
        vals[OBS_IDX["f_1"]]    = freqs_qb[0];  mask[OBS_IDX["f_1"]]    = 1.0
    if chi:
        vals[OBS_IDX["chi_11"]] = chi[0];        mask[OBS_IDX["chi_11"]] = 1.0
    return vals, mask


def obs_resonator(
    freqs_res: list, freqs_qb: list, kappa: list, chi: list, **_
) -> tuple[np.ndarray, np.ndarray]:
    """
    Single CPW resonator.
    Active slots: f_1
    """
    vals, mask = _empty_obs()
    if freqs_res:
        vals[OBS_IDX["f_1"]] = freqs_res[0];  mask[OBS_IDX["f_1"]] = 1.0
    return vals, mask


def obs_qrf(
    freqs_res: list, freqs_qb: list, kappa: list, chi: list, **_
) -> tuple[np.ndarray, np.ndarray]:
    """
    Qubit-Resonator-Feedline (capacitive or inductive).
    Active slots: f_1, f_2, kappa_1, kappa_2, chi_11, chi_22, chi_12
    """
    vals, mask = _empty_obs()
    if freqs_res:
        vals[OBS_IDX["f_1"]]     = freqs_res[0]; mask[OBS_IDX["f_1"]]     = 1.0
    if freqs_qb:
        vals[OBS_IDX["f_2"]]     = freqs_qb[0];  mask[OBS_IDX["f_2"]]     = 1.0
    if len(kappa) > 0:
        vals[OBS_IDX["kappa_1"]] = kappa[0];     mask[OBS_IDX["kappa_1"]] = 1.0
    if len(kappa) > 1:
        vals[OBS_IDX["kappa_2"]] = kappa[1];     mask[OBS_IDX["kappa_2"]] = 1.0
    if len(chi) > 0:
        vals[OBS_IDX["chi_11"]]  = chi[0];       mask[OBS_IDX["chi_11"]]  = 1.0
    if len(chi) > 3:
        vals[OBS_IDX["chi_22"]]  = chi[3];       mask[OBS_IDX["chi_22"]]  = 1.0
    if len(chi) > 1:
        vals[OBS_IDX["chi_12"]]  = chi[1];       mask[OBS_IDX["chi_12"]]  = 1.0
    return vals, mask


def obs_tct(
    freqs_res: list, freqs_qb: list, kappa: list, chi: list,
    Lq1: float = 0.0, Lq2: float = 0.0, **_
) -> tuple[np.ndarray, np.ndarray]:
    """
    Two capacitively-coupled transmons (TCT).

    Dataset storage convention: frequencies are already sorted ascending,
    so freqs_qb[0] = f_low, freqs_qb[1] = f_high; chi11 = self-Kerr of
    the lower-freq mode; chi22 = self-Kerr of the higher-freq mode.

    Active slots: f_1, f_2, chi_11, chi_22, chi_12
    """
    vals, mask = _empty_obs()
    if len(freqs_qb) >= 2 and len(chi) >= 4:
        chi11, chi12, _, chi22 = chi[:4]
        vals[OBS_IDX["f_1"]]    = freqs_qb[0]; mask[OBS_IDX["f_1"]]    = 1.0
        vals[OBS_IDX["f_2"]]    = freqs_qb[1]; mask[OBS_IDX["f_2"]]    = 1.0
        vals[OBS_IDX["chi_11"]] = chi11;        mask[OBS_IDX["chi_11"]] = 1.0
        vals[OBS_IDX["chi_22"]] = chi22;        mask[OBS_IDX["chi_22"]] = 1.0
        vals[OBS_IDX["chi_12"]] = chi12;        mask[OBS_IDX["chi_12"]] = 1.0
    return vals, mask


# ===========================================================================
# Row parsers
# ===========================================================================
#
# HOW TO ADD A NEW PARSER
# ------------------------
# 1. Write a _parse_row_<n>(line: str) function following the pattern below.
# 2. Add it to ROW_PARSERS with the corresponding dataset name as key.
# 3. Register the dataset in DATASETS below.

_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _floats(s: str) -> list[float]:
    return [float(x) for x in _FLOAT_RE.findall(s)]


def _parens_floats(line: str) -> list[list[float]]:
    """Return list of float-lists, one per parenthesised group in the line."""
    groups = re.findall(r"\([^)]*\)", line)
    return [_floats(g) for g in groups]


def _is_header(line: str) -> bool:
    s = line.strip()
    if not s:                        return True
    if s.startswith("Dataset"):      return True
    if set(s) <= set("=- \t"):       return True
    if "[" in s:                     return True  # unit header line
    return False


def _parse_row_qubit(line: str):
    nums = _floats(line)
    if len(nums) < 2:
        return None, None
    Cq, Lq1 = nums[0], nums[1]
    parens   = _parens_floats(line)
    if len(parens) < 2:
        return None, None
    attrs  = {"T1": {"L": Lq1, "C": Cq}}
    obs_kw = dict(
        freqs_res = parens[0],
        freqs_qb  = parens[1],
        kappa     = parens[2] if len(parens) > 2 else [],
        chi       = parens[3] if len(parens) > 3 else [],
    )
    return attrs, obs_kw


def _parse_row_resonator(line: str):
    nums = _floats(line)
    if len(nums) < 1:
        return None, None
    L_res  = nums[0]
    parens = _parens_floats(line)
    if len(parens) < 1:
        return None, None
    attrs  = {"R1": {"length": L_res}}
    obs_kw = dict(
        freqs_res = parens[0],
        freqs_qb  = parens[1] if len(parens) > 1 else [],
        kappa     = parens[2] if len(parens) > 2 else [],
        chi       = parens[3] if len(parens) > 3 else [],
    )
    return attrs, obs_kw


def _parse_row_qrf_cap(line: str):
    nums = _floats(line)
    if len(nums) < 5:
        return None, None
    Cq, Lq1, Cc_qr, Cc_rf, L_res = nums[:5]
    parens = _parens_floats(line)
    if len(parens) < 2:
        return None, None
    attrs = {
        "T":     {"L": Lq1,  "C": Cq},
        "R":     {"length": L_res},
        "Cc_qr": {"Cc": Cc_qr},
        "Cc_rf": {"Cc": Cc_rf},
    }
    obs_kw = dict(
        freqs_res = parens[0],
        freqs_qb  = parens[1],
        kappa     = parens[2] if len(parens) > 2 else [],
        chi       = parens[3] if len(parens) > 3 else [],
    )
    return attrs, obs_kw


def _parse_row_qrf_ind(line: str):
    nums = _floats(line)
    if len(nums) < 6:
        return None, None
    Cq, Lq1, Cc_qr, L_res, L_ind, D_res = nums[:6]
    parens = _parens_floats(line)
    if len(parens) < 2:
        return None, None
    attrs = {
        "T":     {"L": Lq1,  "C": Cq},
        "R":     {"length": L_res},
        "Cc_qr": {"Cc": Cc_qr},
        "Ind":   {"D": D_res, "l": L_ind},
    }
    obs_kw = dict(
        freqs_res = parens[0],
        freqs_qb  = parens[1],
        kappa     = parens[2] if len(parens) > 2 else [],
        chi       = parens[3] if len(parens) > 3 else [],
    )
    return attrs, obs_kw


def _parse_row_tct(line: str):
    nums = _floats(line)
    if len(nums) < 4:
        return None, None
    Cq, Lq1, Lq2, Cc = nums[:4]
    # canonical sort: smaller L → T1
    if Lq1 <= Lq2:
        L1, C1, L2, C2 = Lq1, Cq, Lq2, Cq
    else:
        L1, C1, L2, C2 = Lq2, Cq, Lq1, Cq
    parens = _parens_floats(line)
    if len(parens) < 2:
        return None, None
    attrs = {
        "T1": {"L": L1, "C": C1},
        "T2": {"L": L2, "C": C2},
        "Cc": {"Cc": Cc},
    }
    obs_kw = dict(
        freqs_res = parens[0],
        freqs_qb  = parens[1],
        kappa     = parens[2] if len(parens) > 2 else [],
        chi       = parens[3] if len(parens) > 3 else [],
        Lq1       = Lq1,
        Lq2       = Lq2,
    )
    return attrs, obs_kw


ROW_PARSERS: dict[str, callable] = {
    "Qubit":                                    _parse_row_qubit,
    "Resonator":                                _parse_row_resonator,
    "Qubit_resonator_feedline_capacitive":      _parse_row_qrf_cap,
    "Qubit_resonator_feedline_inductive_nognd": _parse_row_qrf_ind,
    "Two_qubit_with_capacitive_coupling":       _parse_row_tct,
}


# ===========================================================================
# Raw topology templates
# ===========================================================================
#
# HOW TO ADD A NEW TOPOLOGY
# --------------------------
# 1. Write a _build_<name>() function following the pattern below.
# 2. Register it in DATASETS (DatasetDef.topology_fn) below.

def _build_qubit() -> CQEDTopology:
    """Single transmon: T"""
    t = CQEDTopology("Qubit")
    t.add_node(SubgType.TRANSMON, {"L": 0.0, "C": 0.0}, label="T1")
    return t


def _build_resonator() -> CQEDTopology:
    """Single resonator: R"""
    t = CQEDTopology("Resonator")
    t.add_node(SubgType.RESONATOR, {"length": 0.0}, label="R1")
    return t


def _build_qrf_cap() -> CQEDTopology:
    """
    Capacitively-coupled qubit–resonator–feedline.
    Raw: F – Cc_rf – R – Cc_qr – T
    After graphlize: F – RC(Cc_rf, R) – RCT(Cc_qr, R, T)
    """
    t     = CQEDTopology("Qubit_resonator_feedline_capacitive")
    f     = t.add_node(SubgType.FEEDLINE,  {},                   label="F")
    cc_rf = t.add_node(SubgType.C_COUPLER, {"Cc": 0.0},          label="Cc_rf")
    r     = t.add_node(SubgType.RESONATOR, {"length": 0.0},      label="R")
    cc_qr = t.add_node(SubgType.C_COUPLER, {"Cc": 0.0},          label="Cc_qr")
    tr    = t.add_node(SubgType.TRANSMON,  {"L": 0.0, "C": 0.0}, label="T")
    t.add_edge(f, cc_rf)
    t.add_edge(cc_rf, r)
    t.add_edge(r, cc_qr)
    t.add_edge(cc_qr, tr)
    return t


def _build_qrf_ind() -> CQEDTopology:
    """
    Inductively-coupled qubit–resonator–feedline (no ground).
    Raw: F – Ind – R – Cc_qr – T
    After graphlize: F – RIND(Ind, R) – RCT(Cc_qr, R, T)
    """
    t     = CQEDTopology("Qubit_resonator_feedline_inductive_nognd")
    f     = t.add_node(SubgType.FEEDLINE,  {},                    label="F")
    ind   = t.add_node(SubgType.I_COUPLER, {"D": 0.0, "l": 0.0}, label="Ind")
    r     = t.add_node(SubgType.RESONATOR, {"length": 0.0},       label="R")
    cc_qr = t.add_node(SubgType.C_COUPLER, {"Cc": 0.0},           label="Cc_qr")
    tr    = t.add_node(SubgType.TRANSMON,  {"L": 0.0, "C": 0.0},  label="T")
    t.add_edge(f, ind)
    t.add_edge(ind, r)
    t.add_edge(r, cc_qr)
    t.add_edge(cc_qr, tr)
    return t


def _build_tct() -> CQEDTopology:
    """
    Two capacitively-coupled transmons.
    Raw: T1 – Cc – T2
    After graphlize: single TCT block (or kept as 3 separate blocks
    depending on LAZY_PATTERNS configuration).
    """
    t  = CQEDTopology("Two_qubit_with_capacitive_coupling")
    t1 = t.add_node(SubgType.TRANSMON,  {"L": 0.0, "C": 0.0}, label="T1")
    cc = t.add_node(SubgType.C_COUPLER, {"Cc": 0.0},           label="Cc")
    t2 = t.add_node(SubgType.TRANSMON,  {"L": 0.0, "C": 0.0}, label="T2")
    t.add_edge(t1, cc)
    t.add_edge(cc, t2)
    return t


# ===========================================================================
# Dataset registry
# ===========================================================================
#
# HOW TO ADD A NEW DATASET
# --------------------------
# 1. Add the raw data file to the data/ directory.
# 2. Write a topology builder above (_build_<name>).
# 3. Write a row parser above (_parse_row_<name>), add to ROW_PARSERS.
# 4. Write an observable parser above (obs_<name>).
# 5. Add a DatasetDef entry to DATASETS below.
# 6. Add the observable parser to OBS_PARSERS below.
#
# Nothing else needs to change.

@dataclass
class DatasetDef:
    """
    path         : path to the .txt data file (relative to project root)
    topology_fn  : zero-argument callable → fresh raw CQEDTopology
    block_params : ordered list of param-name lists, one per compressed block
                   in the order produced by graphlize().
                   Each inner list must match SUBG_DEFS[type].attrs exactly.
    obs_slots    : which global slots this dataset populates (documentation)
    n_samples    : rows to reservoir-sample per dataset (balanced training)
    """
    path:         str
    topology_fn:  Callable
    block_params: List[List[str]]
    obs_slots:    List[str]
    n_samples:    int = 10_000


DATASETS: dict[str, DatasetDef] = {

    "Qubit": DatasetDef(
        path         = "data/Qubit.txt",
        topology_fn  = _build_qubit,
        block_params = [["L", "C"]],
        obs_slots    = ["f_1", "chi_11"],
        n_samples    = 10_000,
    ),

    "Resonator": DatasetDef(
        path         = "data/Resonator.txt",
        topology_fn  = _build_resonator,
        block_params = [["length"]],
        obs_slots    = ["f_1"],
        n_samples    = 10_000,
    ),

    "Qubit_resonator_feedline_capacitive": DatasetDef(
        path         = "data/Qubit_resonator_feedline_capacitive.txt",
        topology_fn  = _build_qrf_cap,
        # graphlize produces: FEEDLINE – C_COUPLER(Cc_rf) – RCT(length, Cc_qr, L, C)
        block_params = [[], ["Cc"], ["length", "Cc", "L", "C"]],
        obs_slots    = ["f_1", "f_2", "kappa_1", "kappa_2", "chi_11", "chi_22", "chi_12"],
        n_samples    = 10_000,
    ),

    "Qubit_resonator_feedline_inductive_nognd": DatasetDef(
        path         = "data/Qubit_resonator_feedline_inductive_nognd.txt",
        topology_fn  = _build_qrf_ind,
        # graphlize produces: FEEDLINE – RIND(length, D, l) – RCT(length, Cc_qr, L, C)
        block_params = [[], ["D", "l"], ["length", "Cc", "L", "C"]],
        obs_slots    = ["f_1", "f_2", "kappa_1", "kappa_2", "chi_11", "chi_22", "chi_12"],
        n_samples    = 10_000,
    ),

    "Two_qubit_with_capacitive_coupling": DatasetDef(
        path         = "data/Two_qubit_with_capacitive_coupling.txt",
        topology_fn  = _build_tct,
        # graphlize produces: C_COUPLER(Cc) – TRANSMON(L, C) – TRANSMON(L, C)
        # TCT lazy-merge intentionally NOT active; separate blocks give the
        # model more structural signal (which transmon is which).
        block_params = [["Cc"], ["L", "C"], ["L", "C"]],
        obs_slots    = ["f_1", "f_2", "chi_11", "chi_22", "chi_12"],
        n_samples    = 10_000,
    ),
}


OBS_PARSERS: dict[str, Callable] = {
    "Qubit":                                    obs_qubit,
    "Resonator":                                obs_resonator,
    "Qubit_resonator_feedline_capacitive":      obs_qrf,
    "Qubit_resonator_feedline_inductive_nognd": obs_qrf,
    "Two_qubit_with_capacitive_coupling":       obs_tct,
}
