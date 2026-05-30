"""Dataset: two qubits coupled via a shared mediator resonator.

Topology (raw):   T1 – Cc_qr1 – R – Cc_qr2 – T2
After graphlize:  C_COUPLER(Cc_qr1) – TRANSMON(L,C) – RCT(length, Cc_qr2, L, C)
block_params:     [["Cc"], ["L", "C"], ["length", "Cc", "L", "C"]]

Note: RCT absorbs R + Cc_qr2 + T2.  T1 is the qubit whose coupler
      becomes the graphlize root (Cc_qr1 has the lower node_id).
Columns:  Cq, Lq1, Lq2, Cc_qr1, Cc_qr2, L_res
Both qubits share the same Cq in this dataset.
Chi matrix is 3×3 row-major; order = [R, Q1, Q2].
"""
from __future__ import annotations

import numpy as np
from circuit2graph import CQEDTopology, SubgType
from data_loader.datasets._base import DatasetBase, floats, parens_floats, is_header


class TwoQubitResonator(DatasetBase):

    NAME             = "Two_qubit_resonator"
    DATA_PATH        = "data/Two_qubit_resonator.txt"
    BLOCK_PARAMS     = [["Cc"], ["L", "C"], ["length", "Cc", "L", "C"]]
    OBS_SLOTS_ACTIVE = ["f_1", "f_2", "f_3",
                        "chi_11", "chi_22", "chi_12",
                        "chi_33", "chi_13", "chi_23"]
    N_SAMPLES        = 10_000
    INCLUDE_TRAIN    = True

    @classmethod
    def build_topology(cls) -> CQEDTopology:
        t      = CQEDTopology(cls.NAME)
        t1     = t.add_node(SubgType.TRANSMON,  {"L": 0.0, "C": 0.0}, label="T1")
        cc_qr1 = t.add_node(SubgType.C_COUPLER, {"Cc": 0.0},           label="Cc_qr1")
        r      = t.add_node(SubgType.RESONATOR, {"length": 0.0},        label="R")
        cc_qr2 = t.add_node(SubgType.C_COUPLER, {"Cc": 0.0},           label="Cc_qr2")
        t2     = t.add_node(SubgType.TRANSMON,  {"L": 0.0, "C": 0.0},  label="T2")
        t.add_edge(t1, cc_qr1)
        t.add_edge(cc_qr1, r)
        t.add_edge(r, cc_qr2)
        t.add_edge(cc_qr2, t2)
        return t

    @classmethod
    def parse_row(cls, line: str):
        if is_header(line):
            return None, None
        nums = floats(line)
        if len(nums) < 6:
            return None, None
        Cq, Lq1, Lq2, Cc_qr1, Cc_qr2, L_res = nums[:6]
        parens = parens_floats(line)
        if len(parens) < 2:
            return None, None
        attrs = {
            "T1":     {"L": Lq1, "C": Cq},
            "T2":     {"L": Lq2, "C": Cq},
            "R":      {"length": L_res},
            "Cc_qr1": {"Cc": Cc_qr1},
            "Cc_qr2": {"Cc": Cc_qr2},
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
        if freqs_res:
            vals[OBS_IDX["f_1"]]  = freqs_res[0]; mask[OBS_IDX["f_1"]]  = 1.0
        if len(freqs_qb) >= 1:
            vals[OBS_IDX["f_2"]]  = freqs_qb[0];  mask[OBS_IDX["f_2"]]  = 1.0
        if len(freqs_qb) >= 2:
            vals[OBS_IDX["f_3"]]  = freqs_qb[1];  mask[OBS_IDX["f_3"]]  = 1.0
        if len(chi) >= 9:
            vals[OBS_IDX["chi_11"]] = chi[0]; mask[OBS_IDX["chi_11"]] = 1.0
            vals[OBS_IDX["chi_22"]] = chi[4]; mask[OBS_IDX["chi_22"]] = 1.0
            vals[OBS_IDX["chi_12"]] = chi[1]; mask[OBS_IDX["chi_12"]] = 1.0
            vals[OBS_IDX["chi_33"]] = chi[8]; mask[OBS_IDX["chi_33"]] = 1.0
            vals[OBS_IDX["chi_13"]] = chi[2]; mask[OBS_IDX["chi_13"]] = 1.0
            vals[OBS_IDX["chi_23"]] = chi[5]; mask[OBS_IDX["chi_23"]] = 1.0
        return vals, mask
