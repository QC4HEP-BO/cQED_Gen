"""Dataset: two capacitively-coupled transmons.

Topology (raw):   T1 – Cc – T2
After graphlize:  TCT(L, C, Cc, L2, C2, dir)
block_params:     [["L", "C", "Cc", "L2", "C2", "dir"]]

graphlize always lazy-merges T1–Cc–T2 into a single TCT node.
Attrs in TCT preserve traversal/root order: L/C belong to the first transmon
encountered from the root/canonical traversal, L2/C2 to the second.
"""
from __future__ import annotations

import numpy as np
from circuit2graph import CQEDTopology, SubgType
from data_loader.datasets._base import DatasetBase, floats, parens_floats, is_header


class TwoQubitCapacitive(DatasetBase):

    NAME             = "Two_qubit_with_capacitive_coupling"
    DATA_PATH        = "data/Two_qubit_with_capacitive_coupling.txt"
    BLOCK_PARAMS     = [["L", "C", "Cc", "L2", "C2", "dir"]]
    OBS_SLOTS_ACTIVE = ["f_1", "f_2", "chi_11", "chi_22", "chi_12"]
    N_SAMPLES        = 10_000
    INCLUDE_TRAIN    = True

    @classmethod
    def build_topology(cls) -> CQEDTopology:
        t  = CQEDTopology(cls.NAME)
        t1 = t.add_node(SubgType.TRANSMON,  {"L": 0.0, "C": 0.0}, label="T1")
        cc = t.add_node(SubgType.C_COUPLER, {"Cc": 0.0},           label="Cc")
        t2 = t.add_node(SubgType.TRANSMON,  {"L": 0.0, "C": 0.0}, label="T2")
        t.add_edge(t1, cc)
        t.add_edge(cc, t2)
        return t

    @classmethod
    def parse_row(cls, line: str):
        if is_header(line):
            return None, None
        nums = floats(line)
        if len(nums) < 4:
            return None, None
        Cq, Lq1, Lq2, Cc = nums[:4]
        # Preserve primitive order; do not canonical-sort by value.
        L1, C1, L2, C2 = Lq1, Cq, Lq2, Cq
        parens = parens_floats(line)
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
            Lq1=Lq1, Lq2=Lq2,
        )
        return attrs, obs_kw

    @classmethod
    def parse_obs(cls, freqs_res, freqs_qb, kappa, chi, **_):
        from data_loader.schema import N_OBS_SLOTS, OBS_IDX
        vals, mask = np.zeros(N_OBS_SLOTS), np.zeros(N_OBS_SLOTS)
        if len(freqs_qb) >= 2 and len(chi) >= 4:
            chi11, chi12, _, chi22 = chi[:4]
            vals[OBS_IDX["f_1"]]    = freqs_qb[0]; mask[OBS_IDX["f_1"]]    = 1.0
            vals[OBS_IDX["f_2"]]    = freqs_qb[1]; mask[OBS_IDX["f_2"]]    = 1.0
            vals[OBS_IDX["chi_11"]] = chi11;        mask[OBS_IDX["chi_11"]] = 1.0
            vals[OBS_IDX["chi_22"]] = chi22;        mask[OBS_IDX["chi_22"]] = 1.0
            vals[OBS_IDX["chi_12"]] = chi12;        mask[OBS_IDX["chi_12"]] = 1.0
        return vals, mask
