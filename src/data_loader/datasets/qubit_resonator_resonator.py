"""Dataset: qubit coupled to two resonators in series via two capacitors.

Topology (raw):   T – Cc_qr – R1 – Cc_rr – R2
After graphlize:  C_COUPLER(Cc_qr) – TRANSMON(L,C) – RC(length,Cc_rr) – RESONATOR(length)
block_params:     [["Cc"], ["L", "C"], ["length", "Cc"], ["length"]]

Columns:  Cq, Lq1, Cc_qr, Cc_rr, L_res1, L_res2
Chi matrix is 3×3 row-major; order = [R1, R2, Q].
"""
from __future__ import annotations

import numpy as np
from circuit2graph import CQEDTopology, SubgType
from data_loader.datasets._base import DatasetBase, floats, parens_floats, is_header


class QubitResonatorResonator(DatasetBase):

    NAME             = "Qubit_resonator_resonator"
    DATA_PATH        = "data/Qubit_resonator_resonator.txt"
    BLOCK_PARAMS     = [["Cc"], ["L", "C"], ["length", "Cc"], ["length"]]
    OBS_SLOTS_ACTIVE = ["f_1", "f_2", "f_3",
                        "chi_11", "chi_22", "chi_12",
                        "chi_33", "chi_13", "chi_23"]
    N_SAMPLES        = 10_000
    INCLUDE_TRAIN    = True

    @classmethod
    def build_topology(cls) -> CQEDTopology:
        t     = CQEDTopology(cls.NAME)
        tr    = t.add_node(SubgType.TRANSMON,  {"L": 0.0, "C": 0.0}, label="T")
        cc_qr = t.add_node(SubgType.C_COUPLER, {"Cc": 0.0},           label="Cc_qr")
        r1    = t.add_node(SubgType.RESONATOR, {"length": 0.0},        label="R1")
        cc_rr = t.add_node(SubgType.C_COUPLER, {"Cc": 0.0},            label="Cc_rr")
        r2    = t.add_node(SubgType.RESONATOR, {"length": 0.0},        label="R2")
        t.add_edge(tr, cc_qr)
        t.add_edge(cc_qr, r1)
        t.add_edge(r1, cc_rr)
        t.add_edge(cc_rr, r2)
        return t

    @classmethod
    def parse_row(cls, line: str):
        if is_header(line):
            return None, None
        nums = floats(line)
        if len(nums) < 6:
            return None, None
        Cq, Lq1, Cc_qr, Cc_rr, L_res1, L_res2 = nums[:6]
        parens = parens_floats(line)
        if len(parens) < 2:
            return None, None
        attrs = {
            "T":     {"L": Lq1, "C": Cq},
            "R1":    {"length": L_res1},
            "R2":    {"length": L_res2},
            "Cc_qr": {"Cc": Cc_qr},
            "Cc_rr": {"Cc": Cc_rr},
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
        if len(freqs_res) >= 1:
            vals[OBS_IDX["f_1"]]  = freqs_res[0]; mask[OBS_IDX["f_1"]]  = 1.0
        if len(freqs_res) >= 2:
            vals[OBS_IDX["f_2"]]  = freqs_res[1]; mask[OBS_IDX["f_2"]]  = 1.0
        if freqs_qb:
            vals[OBS_IDX["f_3"]]  = freqs_qb[0];  mask[OBS_IDX["f_3"]]  = 1.0
        if len(chi) >= 9:
            vals[OBS_IDX["chi_11"]] = chi[0]; mask[OBS_IDX["chi_11"]] = 1.0
            vals[OBS_IDX["chi_22"]] = chi[4]; mask[OBS_IDX["chi_22"]] = 1.0
            vals[OBS_IDX["chi_12"]] = chi[1]; mask[OBS_IDX["chi_12"]] = 1.0
            vals[OBS_IDX["chi_33"]] = chi[8]; mask[OBS_IDX["chi_33"]] = 1.0
            vals[OBS_IDX["chi_13"]] = chi[2]; mask[OBS_IDX["chi_13"]] = 1.0
            vals[OBS_IDX["chi_23"]] = chi[5]; mask[OBS_IDX["chi_23"]] = 1.0
        return vals, mask
