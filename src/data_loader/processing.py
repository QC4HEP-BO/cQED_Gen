"""Processing utilities for the cQED VAE data loader.

This module handles the transformation of raw circuit data into structured
representations suitable for the model (e.g., graphs, tensors, and observable slots).
The public entry point remains data_loader.loader_vae.load_all_datasets_vae.

This module is a critical bridge between raw dataset format and the model,
ensuring consistency, robustness to missing data, and efficient batching.
"""

from __future__ import annotations

import re
import copy
import random
from typing import Callable

import numpy as np
import torch
from collections import defaultdict
from torch_geometric.data import Data

from circuit2graph import CQEDTopology, CQEDNode, SubgType, SUBG_DEFS, ATTR_INDEX, graphlize
from data_loader.schema import (
    DATASETS, DatasetDef, OBS_PARSERS,
    OBS_SLOTS, N_OBS_SLOTS, OBS_IDX,
    ROW_PARSERS, _is_header, train_datasets,
)


# ===========================================================================
# Scalers
# ===========================================================================

class ParamScaler:
    """
    log10 + StandardScaler for physical circuit parameters.

    Two modes are supported:
      1. legacy positional mode: one mean/std per flat column;
      2. global attribute mode: one mean/std per ATTR_INDEX entry, selected by
         attribute name when transforming a topology-specific flat vector.

    Global attribute mode is what allows one scaler to work for unseen
    topologies whose flat target vectors have different lengths/orderings.
    """

    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.std_:  np.ndarray | None = None
        self.attr_index: dict[str, int] | None = None
        self.mode: str = "positional"

    @staticmethod
    def _base_attr(name: str) -> str:
        # Display/evaluation code may suffix duplicates as L_n2, C_n4, ...
        return name.rsplit("_n", 1)[0] if "_n" in name else name

    def fit(self, Y: np.ndarray) -> "ParamScaler":
        """Legacy positional fit on raw parameter matrix Y [N, n_params]."""
        Y_log      = np.log10(np.maximum(np.abs(Y), 1e-30))
        self.mean_ = Y_log.mean(axis=0)
        self.std_  = Y_log.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        self.attr_index = None
        self.mode = "positional"
        return self

    def fit_attr_matrix(self, Y_attr: np.ndarray, mask: np.ndarray) -> "ParamScaler":
        """Fit globally by ATTR_INDEX column, ignoring absent attributes.

        Parameters
        ----------
        Y_attr : [N, len(ATTR_INDEX)] raw values, arbitrary value where mask=0
        mask   : [N, len(ATTR_INDEX)] 1 if that physical attribute is present
        """
        n_attrs = len(ATTR_INDEX)
        self.mean_ = np.zeros(n_attrs, dtype=np.float64)
        self.std_  = np.ones(n_attrs, dtype=np.float64)
        for attr, j in ATTR_INDEX.items():
            present = mask[:, j] > 0.5
            # ``dir`` is discrete and is never part of the continuous target.
            if attr == "dir" or present.sum() == 0:
                continue
            vals = Y_attr[present, j]
            vals_log = np.log10(np.maximum(np.abs(vals), 1e-30))
            self.mean_[j] = vals_log.mean()
            self.std_[j] = max(vals_log.std(), 1e-8)
        self.attr_index = dict(ATTR_INDEX)
        self.mode = "attr"
        return self

    def _indices_for(self, attr_names: list[str] | tuple[str, ...] | None, n_cols: int) -> np.ndarray:
        if self.mode != "attr":
            return np.arange(n_cols, dtype=int)
        if attr_names is None:
            if n_cols == len(self.mean_):
                return np.arange(n_cols, dtype=int)
            raise ValueError(
                "Global ParamScaler needs attr_names for topology-specific flat vectors "
                f"with {n_cols} columns. Pass names in the same order as Y."
            )
        idx = []
        for name in attr_names:
            base = self._base_attr(str(name))
            if base not in self.attr_index:
                raise KeyError(f"Unknown physical attribute {name!r}; add it to ATTR_INDEX.")
            idx.append(self.attr_index[base])
        return np.asarray(idx, dtype=int)

    def transform(self, Y: np.ndarray, attr_names: list[str] | tuple[str, ...] | None = None) -> np.ndarray:
        Y = np.asarray(Y, dtype=np.float64)
        squeeze = Y.ndim == 1
        if squeeze:
            Y = Y.reshape(1, -1)
        idx = self._indices_for(attr_names, Y.shape[1])
        Y_log = np.log10(np.maximum(np.abs(Y), 1e-30))
        out = (Y_log - self.mean_[idx]) / self.std_[idx]
        return out[0] if squeeze else out

    def inverse_transform(self, Y_scaled: np.ndarray, attr_names: list[str] | tuple[str, ...] | None = None) -> np.ndarray:
        Y_scaled = np.asarray(Y_scaled, dtype=np.float64)
        squeeze = Y_scaled.ndim == 1
        if squeeze:
            Y_scaled = Y_scaled.reshape(1, -1)
        idx = self._indices_for(attr_names, Y_scaled.shape[1])
        out = 10.0 ** (Y_scaled * self.std_[idx] + self.mean_[idx])
        return out[0] if squeeze else out


class ObsScaler:
    """
    StandardScaler for observable values.
    Fits only on present (mask == 1) entries per slot; absent slots stay 0.
    """

    def __init__(self):
        self.mean_ = np.zeros(N_OBS_SLOTS, dtype=np.float64)
        self.std_  = np.ones(N_OBS_SLOTS,  dtype=np.float64)

    def fit(self, obs_vals: np.ndarray, obs_masks: np.ndarray) -> "ObsScaler":
        """
        Fit on obs_vals [N, N_OBS_SLOTS] and obs_masks [N, N_OBS_SLOTS].
        """
        for i in range(N_OBS_SLOTS):
            present = obs_masks[:, i] > 0.5
            if present.sum() > 1:
                v = obs_vals[present, i]
                self.mean_[i] = v.mean()
                self.std_[i]  = max(v.std(), 1e-8)
        return self

    def transform_row(self, obs_vals: np.ndarray, obs_mask: np.ndarray) -> np.ndarray:
        """Scale a single row [N_OBS_SLOTS]; absent slots remain 0."""
        out = np.zeros(N_OBS_SLOTS, dtype=np.float64)
        for i in range(N_OBS_SLOTS):
            if obs_mask[i] > 0.5:
                out[i] = (obs_vals[i] - self.mean_[i]) / self.std_[i]
        return out


class DatasetScalers:
    """Container holding both scalers for one dataset."""

    def __init__(self, param_scaler: ParamScaler, obs_scaler: ObsScaler):
        self.param_scaler = param_scaler
        self.obs_scaler   = obs_scaler

    def inverse_targets(self, Y_scaled: np.ndarray) -> np.ndarray:
        return self.param_scaler.inverse_transform(Y_scaled)


# ===========================================================================
# Derived constants
# ===========================================================================

N_SUBTYPES  = len(SubgType)
# N_DATASETS counts only training topologies (include_train=True).
# Inference-only datasets (e.g. Three_qubit_capacitive_line) are excluded
# so that any model component that embeds dataset identity has a stable size.
N_DATASETS  = len(train_datasets())


# ===========================================================================
# Reservoir-sampled file reader
# ===========================================================================

def _read_raw(
    path:     str,
    ds_name:  str,
    n_samples: int,
    rng:      random.Random,
) -> list[tuple[dict, dict]]:
    row_parser = ROW_PARSERS[ds_name]
    obs_parser = OBS_PARSERS[ds_name]
    reservoir: list[tuple] = []
    count = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if _is_header(line):
                continue
            attrs, obs_kw = row_parser(line)
            if attrs is None:
                continue
            count += 1
            entry = (attrs, obs_kw)
            if len(reservoir) < n_samples:
                reservoir.append(entry)
            else:
                j = rng.randint(0, count - 1)
                if j < n_samples:
                    reservoir[j] = entry

    print(f"  {path}: {count} rows found, using {len(reservoir)}")
    return reservoir


# ===========================================================================
# Fill topology attrs from a parsed row
# ===========================================================================

def _fill_attrs(raw_topo: CQEDTopology, attr_dict: dict) -> CQEDTopology:
    topo = copy.deepcopy(raw_topo)
    for node in topo._nodes:
        if node.label in attr_dict:
            node.attrs.update(attr_dict[node.label])
    return topo


# ===========================================================================
# Parameter extraction from a compressed topology
# ===========================================================================


def _extract_param_attr_names(compressed: CQEDTopology) -> list[str]:
    """Return continuous attribute names in the same order as _extract_params()."""
    names: list[str] = []
    for node in compressed._nodes:
        for attr in SUBG_DEFS[node.subg_type].attrs:
            if attr == "dir":
                continue
            names.append(attr)
    return names


def _extract_attr_matrix_row(compressed: CQEDTopology) -> tuple[np.ndarray, np.ndarray]:
    """Dense [len(ATTR_INDEX)] row + mask for global attribute-scaler fitting."""
    vals = np.zeros(len(ATTR_INDEX), dtype=np.float64)
    mask = np.zeros(len(ATTR_INDEX), dtype=np.float64)
    for node in compressed._nodes:
        for attr in SUBG_DEFS[node.subg_type].attrs:
            if attr == "dir":
                continue
            j = ATTR_INDEX[attr]
            vals[j] = float(node.attrs.get(attr, 0.0))
            mask[j] = 1.0
    return vals, mask


def build_global_scaler(
    all_attr_vals: np.ndarray,
    all_attr_masks: np.ndarray,
    all_obs_vals: np.ndarray,
    all_obs_masks: np.ndarray,
) -> DatasetScalers:
    """Fit one global DatasetScalers over all training datasets.

    Circuit parameters are fitted by physical attribute name via ATTR_INDEX,
    not by flat-vector position. Observables are fitted per OBS slot as before.
    """
    ps = ParamScaler().fit_attr_matrix(all_attr_vals, all_attr_masks)
    os_ = ObsScaler().fit(all_obs_vals, all_obs_masks)
    return DatasetScalers(ps, os_)

def _extract_params(compressed: CQEDTopology) -> list[float]:
    """Extract only continuous physical parameters.

    ``dir`` is a discrete topological/compression attribute (+1/-1), so it
    must not be log-scaled together with physical parameters.
    """
    vals = []
    for node in compressed._nodes:
        for attr in SUBG_DEFS[node.subg_type].attrs:
            if attr == "dir":
                continue
            vals.append(node.attrs.get(attr, 0.0))
    return vals


