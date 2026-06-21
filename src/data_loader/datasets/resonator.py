"""Dataset: single CPW resonator.

Topology (raw):   R
After graphlize:  RESONATOR(length)
"""
from __future__ import annotations

import numpy as np
from circuit2graph import CQEDTopology, SubgType
from data_loader.datasets._base import DatasetBase, floats, parens_floats, is_header


class Resonator(DatasetBase):

    NAME             = "Resonator"
    DATA_PATH        = "data/Resonator.txt"
    OBS_SLOTS_ACTIVE = ["f_1"]
    N_SAMPLES        = 10_000
    INCLUDE_TRAIN    = True

    @classmethod
    def build_topology(cls) -> CQEDTopology:
        t = CQEDTopology("Resonator")
        t.add_node(SubgType.RESONATOR, {"length": 0.0}, label="R1")
        return t

    @classmethod
    def parse_row(cls, line: str):
        if is_header(line):
            return None, None
        nums = floats(line)
        if len(nums) < 1:
            return None, None
        L_res  = nums[0]
        parens = parens_floats(line)
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

    @classmethod
    def parse_obs(cls, freqs_res, freqs_qb, kappa, chi, **_):
        from data_loader.schema import N_OBS_SLOTS, OBS_IDX
        vals, mask = np.zeros(N_OBS_SLOTS), np.zeros(N_OBS_SLOTS)
        if freqs_res:
            vals[OBS_IDX["f_1"]] = freqs_res[0];  mask[OBS_IDX["f_1"]] = 1.0
        return vals, mask
