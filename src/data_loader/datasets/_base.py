"""
data_loader.datasets._base
==========================

Base class and shared parsing utilities for dataset modules.

HOW TO ADD A NEW DATASET
-------------------------
1. Create a new file  src/data_loader/datasets/<your_name>.py
2. Subclass DatasetBase and implement the four required methods:
       build_topology()  →  raw CQEDTopology (template with zero attrs)
       parse_row(line)   →  (attrs_dict, obs_kw_dict) | (None, None)
       parse_obs(...)    →  (obs_vals np.ndarray, obs_mask np.ndarray)
3. Set the class attributes (NAME, DATA_PATH, BLOCK_PARAMS, …).
4. That's it.  schema.py auto-discovers any DatasetBase subclass in this
   package and registers it.  No other file needs to change.

The only exception: if your dataset introduces a new observable type
(e.g. f_4, chi_44) that does not yet exist in OBS_SLOTS (schema.py),
append it there first.  That list is intentionally kept centralised
because its indices must be stable across all datasets and checkpoints.
"""

from __future__ import annotations

import re
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

from circuit2graph import CQEDTopology


# ---------------------------------------------------------------------------
# Shared low-level text-parsing helpers (used by every parse_row)
# ---------------------------------------------------------------------------

_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def floats(s: str) -> list[float]:
    """Extract all floats from a string."""
    return [float(x) for x in _FLOAT_RE.findall(s)]


def parens_floats(line: str) -> list[list[float]]:
    """Return one float-list per parenthesised group in the line."""
    return [floats(g) for g in re.findall(r"\([^)]*\)", line)]


def is_header(line: str) -> bool:
    """Return True if the line is a header / separator to be skipped."""
    s = line.strip()
    if not s:                   return True
    if s.startswith("Dataset"): return True
    if set(s) <= set("=- \t"): return True
    if "[" in s:                return True
    return False


# ---------------------------------------------------------------------------
# Observable helpers  (used by parse_obs implementations)
# ---------------------------------------------------------------------------

def _make_obs_arrays(n_slots: int) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.zeros(n_slots, dtype=np.float64),
        np.zeros(n_slots, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class DatasetBase(ABC):
    """
    Abstract base class for a cQED dataset definition.

    Class attributes (set on the subclass, NOT as instance attrs)
    --------------------------------------------------------------
    NAME          : str   — unique key used throughout the codebase
    DATA_PATH     : str   — path to the raw .txt file (relative to repo root)
    BLOCK_PARAMS  : list[list[str]]
                    Ordered list of attr-name lists, one per compressed
                    macro-node produced by graphlize().
                    Must match SUBG_DEFS[type].attrs exactly for each block.
                    Derive this by calling graphlize() on a dummy topology
                    and printing [list(SUBG_DEFS[n.subg_type].attrs)
                                  for n in compressed._nodes].
    OBS_SLOTS_ACTIVE : list[str]
                    Names of the OBS_SLOTS entries this dataset populates.
                    Used for documentation / sanity checks only; the actual
                    mask is set inside parse_obs().
    N_SAMPLES     : int   — rows to reservoir-sample (default 10_000)
    INCLUDE_TRAIN : bool  — if False, excluded from train/val splits but
                            always available for inference.
                            Flip to True to add to training without changing
                            any other file.
    """

    # ── required class attributes ────────────────────────────────────────
    NAME:             ClassVar[str]
    DATA_PATH:        ClassVar[str]
    BLOCK_PARAMS:     ClassVar[list[list[str]]]
    OBS_SLOTS_ACTIVE: ClassVar[list[str]]

    # ── optional class attributes (have defaults) ────────────────────────
    N_SAMPLES:     ClassVar[int]  = 10_000
    INCLUDE_TRAIN: ClassVar[bool] = True

    # ── abstract methods ─────────────────────────────────────────────────

    @classmethod
    @abstractmethod
    def build_topology(cls) -> CQEDTopology:
        """
        Return a fresh raw CQEDTopology with zero-valued attrs.
        This template is deep-copied and filled for each sample.
        """

    @classmethod
    @abstractmethod
    def parse_row(cls, line: str) -> tuple[dict | None, dict | None]:
        """
        Parse one data line.

        Returns
        -------
        attrs  : dict mapping node label → {attr_name: value}
                 Must match the labels used in build_topology().
        obs_kw : dict of keyword arguments passed to parse_obs()
                 Typically: freqs_res, freqs_qb, kappa, chi

        Return (None, None) for malformed / incomplete lines.
        """

    @classmethod
    @abstractmethod
    def parse_obs(
        cls,
        freqs_res: list,
        freqs_qb:  list,
        kappa:     list,
        chi:       list,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build the observable value vector and binary mask.

        Parameters
        ----------
        freqs_res : list[float]  resonator frequencies [GHz]
        freqs_qb  : list[float]  qubit frequencies [GHz]
        kappa     : list[float]  linewidths [MHz]
        chi       : list[float]  chi-matrix entries (row-major) [MHz]
        **kwargs  : any extra fields passed through from parse_row

        Returns
        -------
        obs_vals : np.ndarray [N_OBS_SLOTS]  values (0 where masked)
        obs_mask : np.ndarray [N_OBS_SLOTS]  1 = present, 0 = absent
        """
