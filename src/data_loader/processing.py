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

from circuit2graph import CQEDTopology, CQEDNode, SubgType, SUBG_DEFS, graphlize
from data_loader.schema import (
    DATASETS, DatasetDef, OBS_PARSERS,
    OBS_SLOTS, N_OBS_SLOTS, OBS_IDX,
    ROW_PARSERS, _is_header,
)


# ===========================================================================
# Scalers
# ===========================================================================

class ParamScaler:
    """
    log10 + StandardScaler for physical circuit parameters (regression targets).

    Pipeline: Y_scaled = (log10(|Y|) - mean) / std
    Inverse:  Y = 10 ** (Y_scaled * std + mean)
    """

    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.std_:  np.ndarray | None = None

    def fit(self, Y: np.ndarray) -> "ParamScaler":
        """Fit on raw parameter matrix Y [N, n_params]."""
        Y_log      = np.log10(np.maximum(np.abs(Y), 1e-30))
        self.mean_ = Y_log.mean(axis=0)
        self.std_  = Y_log.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        return self

    def transform(self, Y: np.ndarray) -> np.ndarray:
        Y_log = np.log10(np.maximum(np.abs(Y), 1e-30))
        return (Y_log - self.mean_) / self.std_

    def inverse_transform(self, Y_scaled: np.ndarray) -> np.ndarray:
        return 10.0 ** (Y_scaled * self.std_ + self.mean_)


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
N_DATASETS  = len(DATASETS)


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

def _extract_params(compressed: CQEDTopology) -> list[float]:
    vals = []
    for node in compressed._nodes:
        for attr in SUBG_DEFS[node.subg_type].attrs:
            vals.append(node.attrs.get(attr, 0.0))
    return vals


