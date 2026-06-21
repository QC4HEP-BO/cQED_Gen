"""Dataset: three transmons in a linear capacitive chain.

Topology (raw):   T1 – Cc12 – T2 – Cc23 – T3
After graphlize:  C_COUPLER(Cc12) – TRANSMON(L,C) – TCT(L,C,Cc23,L2,C2)

Note: TCT absorbs T2 + Cc23 + T3, preserving traversal order T2 -> T3.
      T1 remains as a separate TRANSMON node.
Columns:  Cq, Lq1, Lq2, Lq3, Cc12, Cc23
All three qubits share the same Cq in this dataset.
Chi matrix is 3×3 row-major; order = [Q1, Q2, Q3].

INCLUDE_TRAIN = False
    This topology is held out from training to test generalisation to
    unseen topologies.  To include it in training, flip INCLUDE_TRAIN to
    True — no other file needs to change.
"""
from __future__ import annotations

import numpy as np
from circuit2graph import CQEDTopology, SubgType
from data_loader.datasets._base import DatasetBase, floats, parens_floats, is_header


class ThreeQubitCapacitiveLine(DatasetBase):

    NAME             = "Three_qubit_capacitive_line"
    DATA_PATH        = "data/Three_qubit_capacitive_line.txt"
    OBS_SLOTS_ACTIVE = ["f_1", "f_2", "f_3",
                        "chi_11", "chi_22", "chi_12",
                        "chi_33", "chi_13", "chi_23"]
    N_SAMPLES        = 10_000
    INCLUDE_TRAIN    = False   # ← flip to True to add to training

    @classmethod
    def build_topology(cls) -> CQEDTopology:
        t    = CQEDTopology(cls.NAME)
        t1   = t.add_node(SubgType.TRANSMON,  {"L": 0.0, "C": 0.0}, label="T1")
        cc12 = t.add_node(SubgType.C_COUPLER, {"Cc": 0.0},           label="Cc12")
        t2   = t.add_node(SubgType.TRANSMON,  {"L": 0.0, "C": 0.0}, label="T2")
        cc23 = t.add_node(SubgType.C_COUPLER, {"Cc": 0.0},           label="Cc23")
        t3   = t.add_node(SubgType.TRANSMON,  {"L": 0.0, "C": 0.0}, label="T3")
        t.add_edge(t1, cc12)
        t.add_edge(cc12, t2)
        t.add_edge(t2, cc23)
        t.add_edge(cc23, t3)
        return t

    @classmethod
    def parse_row(cls, line: str):
        if is_header(line):
            return None, None
        nums = floats(line)
        if len(nums) < 6:
            return None, None
        Cq, Lq1, Lq2, Lq3, Cc12, Cc23 = nums[:6]
        parens = parens_floats(line)
        if len(parens) < 2:
            return None, None
        attrs = {
            "T1":   {"L": Lq1, "C": Cq},
            "T2":   {"L": Lq2, "C": Cq},
            "T3":   {"L": Lq3, "C": Cq},
            "Cc12": {"Cc": Cc12},
            "Cc23": {"Cc": Cc23},
        }
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
        if len(freqs_qb) >= 1:
            vals[OBS_IDX["f_1"]]  = freqs_qb[0]; mask[OBS_IDX["f_1"]]  = 1.0
        if len(freqs_qb) >= 2:
            vals[OBS_IDX["f_2"]]  = freqs_qb[1]; mask[OBS_IDX["f_2"]]  = 1.0
        if len(freqs_qb) >= 3:
            vals[OBS_IDX["f_3"]]  = freqs_qb[2]; mask[OBS_IDX["f_3"]]  = 1.0
        if len(chi) >= 9:
            vals[OBS_IDX["chi_11"]] = chi[0]; mask[OBS_IDX["chi_11"]] = 1.0
            vals[OBS_IDX["chi_22"]] = chi[4]; mask[OBS_IDX["chi_22"]] = 1.0
            vals[OBS_IDX["chi_12"]] = chi[1]; mask[OBS_IDX["chi_12"]] = 1.0
            vals[OBS_IDX["chi_33"]] = chi[8]; mask[OBS_IDX["chi_33"]] = 1.0
            vals[OBS_IDX["chi_13"]] = chi[2]; mask[OBS_IDX["chi_13"]] = 1.0
            vals[OBS_IDX["chi_23"]] = chi[5]; mask[OBS_IDX["chi_23"]] = 1.0
        return vals, mask
