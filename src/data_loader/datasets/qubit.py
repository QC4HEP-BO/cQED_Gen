"""Dataset: single transmon qubit.

Topology (raw):   T
After graphlize:  TRANSMON(L, C)
"""
from __future__ import annotations

import numpy as np
from circuit2graph import CQEDTopology, SubgType
from data_loader.datasets._base import DatasetBase, floats, parens_floats, is_header


class Qubit(DatasetBase):

    NAME             = "Qubit"
    DATA_PATH        = "data/Qubit.txt"
    OBS_SLOTS_ACTIVE = ["f_1", "chi_11"]
    N_SAMPLES        = 10_000
    INCLUDE_TRAIN    = True

    @classmethod
    def build_topology(cls) -> CQEDTopology:
        t = CQEDTopology("Qubit")
        t.add_node(SubgType.TRANSMON, {"L": 0.0, "C": 0.0}, label="T1")
        return t

    @classmethod
    def parse_row(cls, line: str):
        if is_header(line):
            return None, None
        nums = floats(line)
        if len(nums) < 2:
            return None, None
        Cq, Lq1 = nums[0], nums[1]
        parens   = parens_floats(line)
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

    @classmethod
    def parse_obs(cls, freqs_res, freqs_qb, kappa, chi, **_):
        from data_loader.schema import N_OBS_SLOTS, OBS_IDX
        vals, mask = np.zeros(N_OBS_SLOTS), np.zeros(N_OBS_SLOTS)
        if freqs_qb:
            vals[OBS_IDX["f_1"]]    = freqs_qb[0];  mask[OBS_IDX["f_1"]]    = 1.0
        if chi:
            vals[OBS_IDX["chi_11"]] = chi[0];        mask[OBS_IDX["chi_11"]] = 1.0
        return vals, mask
