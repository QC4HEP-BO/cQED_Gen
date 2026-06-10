"""Dataset: qubit–resonator–feedline with inductive coupling (no ground).

Topology (raw):   F – Ind – R – Cc_qr – T
After graphlize:  FEEDLINE – RI(length, D, l) – RCT(length, Cc_qr, L, C)
block_params:     [[], ["length", "D", "l", "dir"], ["length", "Cc", "L", "C", "dir"]]
"""
from __future__ import annotations

import numpy as np
from circuit2graph import CQEDTopology, SubgType
from data_loader.datasets._base import DatasetBase, floats, parens_floats, is_header


class QubitResonatorFeedlineInductiveNognd(DatasetBase):

    NAME             = "Qubit_resonator_feedline_inductive_nognd"
    DATA_PATH        = "data/Qubit_resonator_feedline_inductive_nognd.txt"
    BLOCK_PARAMS     = [[], ["length", "D", "l", "dir"], ["length", "Cc", "L", "C", "dir"]]
    OBS_SLOTS_ACTIVE = ["f_1", "f_2", "kappa_1", "kappa_2",
                        "chi_11", "chi_22", "chi_12"]
    N_SAMPLES        = 10_000
    INCLUDE_TRAIN    = True

    @classmethod
    def build_topology(cls) -> CQEDTopology:
        t     = CQEDTopology(cls.NAME)
        f     = t.add_node(SubgType.FEEDLINE,  {},                    label="F")
        ind   = t.add_node(SubgType.I_COUPLER, {"D": 0.0, "l": 0.0}, label="Ind")
        r     = t.add_node(SubgType.RESONATOR, {"length": 0.0},       label="R")
        cc_qr = t.add_node(SubgType.C_COUPLER, {"Cc": 0.0},           label="Cc_qr")
        tr    = t.add_node(SubgType.TRANSMON,  {"L": 0.0, "C": 0.0},  label="T")
        t.add_edge(f, ind)
        t.add_edge(ind, r)
        t.add_edge(r, cc_qr)
        t.add_edge(cc_qr, tr)
        return t

    @classmethod
    def parse_row(cls, line: str):
        if is_header(line):
            return None, None
        nums = floats(line)
        if len(nums) < 6:
            return None, None
        Cq, Lq1, Cc_qr, L_res, L_ind, D_res = nums[:6]
        parens = parens_floats(line)
        if len(parens) < 2:
            return None, None
        attrs = {
            "T":     {"L": Lq1,  "C": Cq},
            "R":     {"length": L_res},
            "Cc_qr": {"Cc": Cc_qr},
            "Ind":   {"D": D_res, "l": L_ind},
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
            vals[OBS_IDX["f_1"]]     = freqs_res[0]; mask[OBS_IDX["f_1"]]     = 1.0
        if freqs_qb:
            vals[OBS_IDX["f_2"]]     = freqs_qb[0];  mask[OBS_IDX["f_2"]]     = 1.0
        if len(kappa) > 0:
            vals[OBS_IDX["kappa_1"]] = kappa[0];     mask[OBS_IDX["kappa_1"]] = 1.0
        if len(kappa) > 1:
            vals[OBS_IDX["kappa_2"]] = kappa[1];     mask[OBS_IDX["kappa_2"]] = 1.0
        if len(chi) > 0:
            vals[OBS_IDX["chi_11"]]  = chi[0];       mask[OBS_IDX["chi_11"]]  = 1.0
        if len(chi) > 3:
            vals[OBS_IDX["chi_22"]]  = chi[3];       mask[OBS_IDX["chi_22"]]  = 1.0
        if len(chi) > 1:
            vals[OBS_IDX["chi_12"]]  = chi[1];       mask[OBS_IDX["chi_12"]]  = 1.0
        return vals, mask
