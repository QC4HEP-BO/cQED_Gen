"""
data_loader.schema
==================

Auto-discovery registry for cQED dataset definitions.

This file does three things and nothing else:

  1. Defines OBS_SLOTS — the global ordered list of observable names.
     Append new slots here (always at the end) when a new dataset
     introduces an observable type that does not yet exist.

  2. Scans the  data_loader/datasets/  package, imports every module
     whose name does not start with '_', and collects all concrete
     subclasses of DatasetBase into DATASETS, ROW_PARSERS, OBS_PARSERS.

  3. Exposes the DatasetDef dataclass (thin wrapper kept for backward
     compatibility with any code that references it by name) and the
     train_datasets() helper.

HOW TO ADD A NEW DATASET
-------------------------
  1. Create  src/data_loader/datasets/<your_name>.py
     following the template in  datasets/_base.py.
  2. That's it — this file picks it up automatically on the next import.

The only manual step remains: if your dataset introduces a new
observable type (e.g. f_4, chi_44), append it to OBS_SLOTS below.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

import data_loader.datasets as _ds_pkg
from data_loader.datasets._base import DatasetBase


# ===========================================================================
# Observable vocabulary
# ===========================================================================
#
# HOW TO ADD A NEW OBSERVABLE
# ----------------------------
# Append the new name to OBS_SLOTS (always at the end so that existing
# checkpoint indices stay valid).  N_OBS_SLOTS and OBS_IDX are derived
# automatically.

OBS_SLOTS: list[str] = [
    # ── original 7 slots (do not reorder) ──────────────────────────────
    "f_1",
    "f_2",
    "kappa_1",
    "kappa_2",
    "chi_11",
    "chi_22",
    "chi_12",
    # ── added for 3-qubit / 3-mode topologies ──────────────────────────
    "f_3",
    "chi_33",
    "chi_13",
    "chi_23",
]

N_OBS_SLOTS: int            = len(OBS_SLOTS)
OBS_IDX:     dict[str, int] = {name: i for i, name in enumerate(OBS_SLOTS)}


# ===========================================================================
# DatasetDef — thin wrapper kept for backward compatibility
# ===========================================================================

@dataclass
class DatasetDef:
    """
    Mirrors the information stored in a DatasetBase subclass as plain
    attributes so that the rest of the codebase can treat DATASETS as a
    dict[str, DatasetDef] without knowing about the class hierarchy.

    Note: block_params is no longer stored here.  Call
    dataset_class.block_params() to get the auto-derived list, or use
    _extract_params() / SUBG_DEFS directly in the training pipeline.
    """
    path:          str
    topology_fn:   Callable
    obs_slots:     list[str]
    n_samples:     int  = 10_000
    include_train: bool = True


# ===========================================================================
# Auto-discovery
# ===========================================================================

def _discover() -> tuple[
    dict[str, DatasetDef],
    dict[str, Callable],
    dict[str, Callable],
]:
    """
    Walk every module in the datasets package, find DatasetBase subclasses,
    and build the three registry dicts.
    """
    datasets:    dict[str, DatasetDef] = {}
    row_parsers: dict[str, Callable]   = {}
    obs_parsers: dict[str, Callable]   = {}

    for module_info in pkgutil.iter_modules(_ds_pkg.__path__):
        if module_info.name.startswith("_"):
            continue
        mod = importlib.import_module(f"data_loader.datasets.{module_info.name}")
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if (
                issubclass(obj, DatasetBase)
                and obj is not DatasetBase
                and not inspect.isabstract(obj)
            ):
                name = obj.NAME
                defn = DatasetDef(
                    path          = obj.DATA_PATH,
                    topology_fn   = obj.build_topology,
                    obs_slots     = obj.OBS_SLOTS_ACTIVE,
                    n_samples     = obj.N_SAMPLES,
                    include_train = obj.INCLUDE_TRAIN,
                )
                datasets[name]    = defn
                row_parsers[name] = obj.parse_row
                obs_parsers[name] = obj.parse_obs

    return datasets, row_parsers, obs_parsers


DATASETS, ROW_PARSERS, OBS_PARSERS = _discover()


# ===========================================================================
# Helpers
# ===========================================================================

def train_datasets() -> dict[str, DatasetDef]:
    """Return the subset of DATASETS with include_train=True."""
    return {k: v for k, v in DATASETS.items() if v.include_train}


# ---------------------------------------------------------------------------
# Low-level text-parsing helpers re-exported for loader_vae / processing
# (kept here so existing imports from data_loader.schema still work)
# ---------------------------------------------------------------------------

from data_loader.datasets._base import is_header as _is_header  # noqa: E402
