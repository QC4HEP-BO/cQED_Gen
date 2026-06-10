"""
cqed.topology.definitions
=========================
The single source of truth for all subgraph types used in the cQED
circuit-graph representation.

EXTENSION GUIDE
---------------
To add a new circuit element (e.g. SQUID):

  1. Add its physical attribute(s) to ATTR_INDEX if not already present.
  2. Add a new member to SubgType (increment enum value).
  3. Add a SubgDef entry to SUBG_DEFS with the descriptor fields filled in.
  4. Implement the merge rule in cqed/graph/merge_rules.py.

Nothing in this file needs to change beyond steps 1–3.
"""

from dataclasses import dataclass, field
from enum import IntEnum


# ---------------------------------------------------------------------------
# 1. Attribute index — add here to expose a new physical parameter globally
# ---------------------------------------------------------------------------

ATTR_INDEX: dict[str, int] = {
    "L":      0,   # inductance         (transmon 1)
    "C":      1,   # capacitance        (transmon 1)
    "length": 2,   # resonator length
    "Cc":     3,   # generic capacitive coupler
    "Cc_qr":  4,   # coupler qubit–resonator
    "Cc_rf":  5,   # coupler resonator–feedline
    "L2":     6,   # inductance         (transmon 2 in a 2-T block)
    "C2":     7,   # capacitance        (transmon 2 in a 2-T block)
    "D":      8,   # wire separation    (inductive coupler, µm)
    "l":      9,   # coupling length    (inductive coupler, m)
    "dir":   10,   # compression direction (+1 canonical, -1 reversed)
}


# ---------------------------------------------------------------------------
# 2. Subgraph type enum — add new members here
# ---------------------------------------------------------------------------

class SubgType(IntEnum):
    FEEDLINE  = 0
    TRANSMON  = 1
    RESONATOR = 2
    C_COUPLER = 3
    TC        = 4
    RC        = 5
    RCT       = 6
    TCT       = 7
    I_COUPLER = 8
    RI        = 9


# ---------------------------------------------------------------------------
# 3. Subgraph definition descriptor
# ---------------------------------------------------------------------------

@dataclass
class SubgDef:
    """
    Everything the rest of the code needs to know about a subgraph type.

    Fields
    ------
    components  : human-readable list of element symbols (e.g. ["T", "C", "T"])
    color       : hex color used when drawing circuit graphs
    legend      : label shown in matplotlib legends
    attrs       : ordered list of ATTR_INDEX keys this subgraph exports as
                  regression targets (must match SUBG_DEFS[type].attrs exactly
                  in dataset_config.py block_params)
    inner_nodes : list of dicts with keys
                    type_id   – int, encodes the circuit-element kind
                    val_attr  – str | None, ATTR_INDEX key for "val" feature
                    val2_attr – str | None, ATTR_INDEX key for "val2" feature
    inner_edges : list of (int, int) pairs among inner_nodes indices
    """
    components:  list
    color:       str
    legend:      str
    attrs:       list
    inner_nodes: list = field(default_factory=list)
    inner_edges: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# 4. THE SUBGRAPH REGISTRY — single place to define / extend subgraph types
# ---------------------------------------------------------------------------

SUBG_DEFS: dict[SubgType, SubgDef] = {

    SubgType.FEEDLINE: SubgDef(
        components  = ["F"],
        color       = "#4C78A8",
        legend      = "Feedline",
        attrs       = [],
        inner_nodes = [{"type_id": 0, "val_attr": None, "val2_attr": None}],
        inner_edges = [],
    ),

    SubgType.TRANSMON: SubgDef(
        components  = ["T"],
        color       = "#F58518",
        legend      = "Transmon",
        attrs       = ["L", "C"],
        inner_nodes = [
            {"type_id": 0, "val_attr": "L",  "val2_attr": None},
            {"type_id": 1, "val_attr": "C",  "val2_attr": None},
        ],
        inner_edges = [(0, 1)],
    ),

    SubgType.RESONATOR: SubgDef(
        components  = ["R"],
        color       = "#54A24B",
        legend      = "Resonator",
        attrs       = ["length"],
        inner_nodes = [{"type_id": 0, "val_attr": "length", "val2_attr": None}],
        inner_edges = [],
    ),

    SubgType.C_COUPLER: SubgDef(
        components  = ["C"],
        color       = "#E45756",
        legend      = "C_COUPLER",
        attrs       = ["Cc"],
        inner_nodes = [{"type_id": 0, "val_attr": "Cc", "val2_attr": None}],
        inner_edges = [],
    ),

    SubgType.TC: SubgDef(
        components  = ["T", "C"],
        color       = "#B279A2",
        legend      = "T+C",
        attrs       = ["L", "C", "Cc", "dir"],
        inner_nodes = [
            {"type_id": 0, "val_attr": "L",  "val2_attr": None},
            {"type_id": 1, "val_attr": "C",  "val2_attr": None},
            {"type_id": 2, "val_attr": "Cc", "val2_attr": None},
        ],
        inner_edges = [(0, 1), (1, 2), (0, 2)],
    ),

    SubgType.RC: SubgDef(
        components  = ["R", "C"],
        color       = "#72B7B2",
        legend      = "R+C",
        attrs       = ["length", "Cc", "dir"],
        inner_nodes = [
            {"type_id": 0, "val_attr": "length", "val2_attr": None},
            {"type_id": 1, "val_attr": "Cc",     "val2_attr": None},
        ],
        inner_edges = [(0, 1)],
    ),

    SubgType.RCT: SubgDef(
        components  = ["R", "C", "T"],
        color       = "#FF9DA6",
        legend      = "R+C+T",
        attrs       = ["length", "Cc", "L", "C", "dir"],
        inner_nodes = [
            {"type_id": 0, "val_attr": "length", "val2_attr": None},
            {"type_id": 1, "val_attr": "Cc",     "val2_attr": None},
            {"type_id": 2, "val_attr": "L",      "val2_attr": None},
            {"type_id": 3, "val_attr": "C",      "val2_attr": None},
        ],
        inner_edges = [(0, 1), (1, 2), (1, 3), (2, 3)],
    ),

    SubgType.TCT: SubgDef(
        components  = ["T", "C", "T"],
        color       = "#FFD700",
        legend      = "T+C+T",
        attrs       = ["L", "C", "Cc", "L2", "C2", "dir"],
        inner_nodes = [
            {"type_id": 0, "val_attr": "L",  "val2_attr": None},
            {"type_id": 1, "val_attr": "C",  "val2_attr": None},
            {"type_id": 2, "val_attr": "Cc", "val2_attr": None},
            {"type_id": 3, "val_attr": "L2", "val2_attr": None},
            {"type_id": 4, "val_attr": "C2", "val2_attr": None},
        ],
        inner_edges = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 2), (2, 4)],
    ),

    SubgType.I_COUPLER: SubgDef(
        components  = ["Ind"],
        color       = "#4E49E0",
        legend      = "I_COUPLER",
        attrs       = ["D", "l"],
        # node 0 = D (wire separation, µm)
        # node 1 = l (coupling length, m)
        # Edge because D and l are geometrically co-defined.
        inner_nodes = [
            {"type_id": 0, "val_attr": "D", "val2_attr": None},
            {"type_id": 1, "val_attr": "l", "val2_attr": None},
        ],
        inner_edges = [(0, 1)],
    ),

    SubgType.RI: SubgDef(
        components  = ["R", "Ind"],
        color       = "#A73AB6",
        legend      = "R+Ind",
        attrs       = ["length", "D", "l", "dir"],
        # node 0 = resonator length, node 1 = D, node 2 = l
        inner_nodes = [
            {"type_id": 0, "val_attr": "length", "val2_attr": None},
            {"type_id": 1, "val_attr": "D",      "val2_attr": None},
            {"type_id": 2, "val_attr": "l",      "val2_attr": None},
        ],
        inner_edges = [(0, 1), (1, 2)],
    ),
}


# ---------------------------------------------------------------------------
# 5. Derived helpers — auto-built from SUBG_DEFS; do not edit manually
# ---------------------------------------------------------------------------

SUBG_NODE:      dict[SubgType, list[str]] = {t: d.components for t, d in SUBG_DEFS.items()}
NODE_COLORS:    dict[SubgType, str]       = {t: d.color      for t, d in SUBG_DEFS.items()}
LEGEND_ENTRIES: list[tuple]               = [(t, d.legend)   for t, d in SUBG_DEFS.items()]
