"""
TEMPLATE — copy this file and rename it (without the leading underscore).
Files starting with '_' are ignored by the auto-discovery in schema.py.

─────────────────────────────────────────────────────────────────────────────
HOW TO ADD A NEW DATASET  (complete checklist)
─────────────────────────────────────────────────────────────────────────────

Step 1 — (only if needed) add new physical parameters to SUBG_DEFS
    Open  src/circuit2graph/definitions.py  and:
      a) add any new attr name to ATTR_INDEX
      b) add a new SubgType member if introducing a genuinely new element type
      c) add / update the SubgDef entry for that type

    This is NOT needed if your circuit uses only existing element types
    (TRANSMON, RESONATOR, C_COUPLER, FEEDLINE, I_COUPLER, …).

Step 2 — (only if needed) add new observable slots
    Open  src/data_loader/schema.py  and append to OBS_SLOTS.
    Always append at the end so existing indices stay valid.

Step 3 — create this file
    Copy this template to  src/data_loader/datasets/<your_name>.py
    Fill in the four items below:
      a) NAME, DATA_PATH, OBS_SLOTS_ACTIVE   (required class attributes)
      b) build_topology()                    (circuit structure)
      c) parse_row()                         (one line of the .txt file)
      d) parse_obs()                         (observable → slot mapping)

    block_params is derived automatically from build_topology() + SUBG_DEFS.
    You do NOT need to declare or compute it.

Step 4 — done
    schema.py auto-discovers any DatasetBase subclass in this package.
    No other file needs to change.

    To verify:
        >>> from data_loader.schema import DATASETS
        >>> print(list(DATASETS))                 # your new name should appear
        >>> from data_loader.datasets.your_name import YourClass
        >>> print(YourClass.block_params())       # auto-derived from topology
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import numpy as np
from circuit2graph import CQEDTopology, SubgType
from data_loader.datasets._base import DatasetBase, floats, parens_floats, is_header


class _TemplateDataset(DatasetBase):
    """Replace this docstring with a short description of the circuit."""

    # ── Required ──────────────────────────────────────────────────────────
    NAME             = "_template"           # unique key, e.g. "Four_qubit_ring"
    DATA_PATH        = "data/_template.txt"  # path to the raw data file
    OBS_SLOTS_ACTIVE = ["f_1", "chi_11"]     # observables this dataset provides

    # ── Optional ──────────────────────────────────────────────────────────
    N_SAMPLES        = 10_000   # rows to reservoir-sample from the file
    INCLUDE_TRAIN    = True     # set False to use for inference-only evaluation

    # ── Circuit topology ──────────────────────────────────────────────────
    @classmethod
    def build_topology(cls) -> CQEDTopology:
        """
        Build and return a raw CQEDTopology with zero-valued attrs.

        Rules:
        - Every node gets a unique, meaningful label (used in parse_row attrs).
        - Attrs are initialised to 0.0; parse_row fills them per sample.
        - Only the topology (node types + edges) matters here.

        Example — single transmon:
            t = CQEDTopology(cls.NAME)
            t.add_node(SubgType.TRANSMON, {"L": 0.0, "C": 0.0}, label="T1")
            return t

        Example — transmon capacitively coupled to resonator:
            t    = CQEDTopology(cls.NAME)
            t1   = t.add_node(SubgType.TRANSMON,  {"L": 0.0, "C": 0.0}, label="T1")
            cc   = t.add_node(SubgType.C_COUPLER, {"Cc": 0.0},           label="Cc1")
            r    = t.add_node(SubgType.RESONATOR, {"length": 0.0},        label="R1")
            t.add_edge(t1, cc)
            t.add_edge(cc, r)
            return t
        """
        raise NotImplementedError

    # ── Row parser ────────────────────────────────────────────────────────
    @classmethod
    def parse_row(cls, line: str):
        """
        Parse one line of the data file.

        Returns
        -------
        attrs  : dict  { node_label: {attr_name: float, ...}, ... }
                 Labels must match exactly what build_topology() uses.
        obs_kw : dict  keyword arguments for parse_obs()

        Return (None, None) for header lines or malformed rows.

        Example:
            if is_header(line): return None, None
            nums = floats(line)
            if len(nums) < 3: return None, None
            Cq, Lq, L_res = nums[:3]
            parens = parens_floats(line)
            if len(parens) < 2: return None, None
            attrs = {
                "T1": {"L": Lq, "C": Cq},
                "R1": {"length": L_res},
            }
            obs_kw = dict(
                freqs_res = parens[0],
                freqs_qb  = parens[1],
                kappa     = parens[2] if len(parens) > 2 else [],
                chi       = parens[3] if len(parens) > 3 else [],
            )
            return attrs, obs_kw
        """
        raise NotImplementedError

    # ── Observable mapper ─────────────────────────────────────────────────
    @classmethod
    def parse_obs(cls, freqs_res, freqs_qb, kappa, chi, **_):
        """
        Map parsed observables to the global OBS_SLOTS array.

        Use OBS_IDX["f_1"], OBS_IDX["chi_11"], etc. to address slots.
        Set mask[i] = 1.0 for every slot this dataset populates.

        Example:
            from data_loader.schema import N_OBS_SLOTS, OBS_IDX
            vals, mask = np.zeros(N_OBS_SLOTS), np.zeros(N_OBS_SLOTS)
            if freqs_qb:
                vals[OBS_IDX["f_1"]]    = freqs_qb[0]; mask[OBS_IDX["f_1"]]    = 1.0
            if chi:
                vals[OBS_IDX["chi_11"]] = chi[0];       mask[OBS_IDX["chi_11"]] = 1.0
            return vals, mask
        """
        raise NotImplementedError
