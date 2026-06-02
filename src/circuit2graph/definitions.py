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
  4. Implement the merge rule in compression.py.

Nothing in this file needs to change beyond steps 1–3.

DIRECTIONAL MACRONODES
----------------------
Asymmetric macronodes (e.g. TC, CT, TCR, RCT) encode the traversal direction
from root to leaf.  The convention is:

  - left  side  = root-facing terminal  (closer to root in the BFS tree)
  - right side  = leaf-facing terminal  (farther from root)

SubgDef.terminal_left  = index into inner_nodes of the root-facing node
SubgDef.terminal_right = index into inner_nodes of the leaf-facing node

Symmetric macronodes (TCT, single-element types) set both to None.
These fields are used by:
  - graphlize() to assign attrs in the correct order
  - expand_macronodes() (step 2) to reconstruct the original topology
"""

from dataclasses import dataclass, field
from enum import IntEnum


# ---------------------------------------------------------------------------
# 1. Attribute index — add here to expose a new physical parameter globally
# ---------------------------------------------------------------------------

ATTR_INDEX: dict[str, int] = {
    "L":      0,   # inductance         (transmon, left slot)
    "C":      1,   # capacitance        (transmon, left slot)
    "length": 2,   # resonator length
    "Cc":     3,   # generic capacitive coupler
    "Cc_qr":  4,   # coupler qubit–resonator
    "Cc_rf":  5,   # coupler resonator–feedline
    "L2":     6,   # inductance         (transmon, right slot in TCT)
    "C2":     7,   # capacitance        (transmon, right slot in TCT)
    "D":      8,   # wire separation    (inductive coupler, µm)
    "l":      9,   # coupling length    (inductive coupler, m)
}


# ---------------------------------------------------------------------------
# 2. Subgraph type enum — add new members here
# ---------------------------------------------------------------------------

class SubgType(IntEnum):
    FEEDLINE  = 0
    TRANSMON  = 1
    RESONATOR = 2
    C_COUPLER = 3
    # --- directional 2-node blocks ---
    TC        = 4   # Transmon(root-side) – C_coupler(leaf-side)
    CT        = 10  # C_coupler(root-side) – Transmon(leaf-side)
    RC        = 5   # Resonator(root-side) – C_coupler(leaf-side)
    CR        = 11  # C_coupler(root-side) – Resonator(leaf-side)
    RI        = 12  # Resonator(root-side) – I_coupler(leaf-side)  (was RIND 2-node)
    IR        = 13  # I_coupler(root-side) – Resonator(leaf-side)
    # --- symmetric / directional 3-node blocks ---
    TCT       = 7   # symmetric: Transmon – C_coupler – Transmon
    TCR       = 14  # Transmon(root) – C_coupler – Resonator(leaf)
    RCT       = 6   # Resonator(root) – C_coupler – Transmon(leaf)
    # --- inductive 3-node blocks ---
    RIND      = 9   # Resonator(root) – I_coupler – ? (legacy, kept for compat)
    TIT       = 15  # Transmon(root) – I_coupler – Transmon(leaf)  [future]
    # --- single-element with coupler kept standalone ---
    I_COUPLER = 8


# ---------------------------------------------------------------------------
# 3. Subgraph definition descriptor
# ---------------------------------------------------------------------------

@dataclass
class SubgDef:
    """
    Everything the rest of the code needs to know about a subgraph type.

    Fields
    ------
    components    : human-readable list of element symbols (e.g. ["T", "C", "T"])
    color         : hex color used when drawing circuit graphs
    legend        : label shown in matplotlib legends
    attrs         : ordered list of ATTR_INDEX keys this subgraph exports as
                    regression targets
    inner_nodes   : list of dicts with keys
                      type_id   – int, encodes the circuit-element kind
                      val_attr  – str | None, ATTR_INDEX key for "val" feature
                      val2_attr – str | None, ATTR_INDEX key for "val2" feature
    inner_edges   : list of (int, int) pairs among inner_nodes indices
    terminal_left : index of the inner_node that faces the root  (None = symmetric)
    terminal_right: index of the inner_node that faces the leaf  (None = symmetric)
    symmetric     : True if the macronode has no preferred orientation
                    (used by graphlize and expand_macronodes)
    """
    components:     list
    color:          str
    legend:         str
    attrs:          list
    inner_nodes:    list = field(default_factory=list)
    inner_edges:    list = field(default_factory=list)
    terminal_left:  int | None = None   # inner_node index, root-facing
    terminal_right: int | None = None   # inner_node index, leaf-facing
    symmetric:      bool = False


# ---------------------------------------------------------------------------
# 4. THE SUBGRAPH REGISTRY — single place to define / extend subgraph types
# ---------------------------------------------------------------------------
#
# inner_node type_id convention (primitive elements):
#   0 = Feedline
#   1 = Transmon (L node — the inductive part)
#   2 = Transmon (C node — the capacitive shunt)
#   3 = Resonator
#   4 = C_coupler
#   5 = I_coupler node D
#   6 = I_coupler node l
#
# terminal_left / terminal_right refer to the outer connection points of the
# macronode, i.e. the inner_nodes that have edges toward other macronodes.

SUBG_DEFS: dict[SubgType, SubgDef] = {

    # ── Primitive single-element types (symmetric, no orientation) ────────

    SubgType.FEEDLINE: SubgDef(
        components     = ["F"],
        color          = "#4C78A8",
        legend         = "Feedline",
        attrs          = [],
        inner_nodes    = [{"type_id": 0, "val_attr": None, "val2_attr": None}],
        inner_edges    = [],
        terminal_left  = 0,
        terminal_right = 0,
        symmetric      = True,
    ),

    SubgType.TRANSMON: SubgDef(
        components     = ["T"],
        color          = "#F58518",
        legend         = "Transmon",
        attrs          = ["L", "C"],
        inner_nodes    = [
            {"type_id": 1, "val_attr": "L", "val2_attr": None},   # 0: L node
            {"type_id": 2, "val_attr": "C", "val2_attr": None},   # 1: C node
        ],
        inner_edges    = [(0, 1)],
        terminal_left  = 0,   # L node is the external connection point
        terminal_right = 0,
        symmetric      = True,
    ),

    SubgType.RESONATOR: SubgDef(
        components     = ["R"],
        color          = "#54A24B",
        legend         = "Resonator",
        attrs          = ["length"],
        inner_nodes    = [{"type_id": 3, "val_attr": "length", "val2_attr": None}],
        inner_edges    = [],
        terminal_left  = 0,
        terminal_right = 0,
        symmetric      = True,
    ),

    SubgType.C_COUPLER: SubgDef(
        components     = ["C"],
        color          = "#E45756",
        legend         = "C_COUPLER",
        attrs          = ["Cc"],
        inner_nodes    = [{"type_id": 4, "val_attr": "Cc", "val2_attr": None}],
        inner_edges    = [],
        terminal_left  = 0,
        terminal_right = 0,
        symmetric      = True,
    ),

    SubgType.I_COUPLER: SubgDef(
        components     = ["Ind"],
        color          = "#4E49E0",
        legend         = "I_COUPLER",
        attrs          = ["D", "l"],
        inner_nodes    = [
            {"type_id": 5, "val_attr": "D", "val2_attr": None},   # 0: D node
            {"type_id": 6, "val_attr": "l", "val2_attr": None},   # 1: l node
        ],
        inner_edges    = [(0, 1)],
        terminal_left  = 0,
        terminal_right = 0,
        symmetric      = True,
    ),

    # ── Directional 2-node blocks ──────────────────────────────────────────
    # Convention: inner_nodes[0] = root-facing, inner_nodes[1] = leaf-facing

    SubgType.TC: SubgDef(
        components     = ["T", "C"],
        color          = "#B279A2",
        legend         = "T→C",
        attrs          = ["L", "C", "Cc"],
        inner_nodes    = [
            {"type_id": 1, "val_attr": "L",  "val2_attr": None},  # 0: T(L) root-side
            {"type_id": 2, "val_attr": "C",  "val2_attr": None},  # 1: T(C)
            {"type_id": 4, "val_attr": "Cc", "val2_attr": None},  # 2: coupler leaf-side
        ],
        inner_edges    = [(0, 1), (0, 2), (1, 2)],
        terminal_left  = 0,   # T(L) connects toward root
        terminal_right = 2,   # coupler connects toward leaf
        symmetric      = False,
    ),

    SubgType.CT: SubgDef(
        components     = ["C", "T"],
        color          = "#C299C2",
        legend         = "C→T",
        attrs          = ["Cc", "L", "C"],
        inner_nodes    = [
            {"type_id": 4, "val_attr": "Cc", "val2_attr": None},  # 0: coupler root-side
            {"type_id": 1, "val_attr": "L",  "val2_attr": None},  # 1: T(L) leaf-side
            {"type_id": 2, "val_attr": "C",  "val2_attr": None},  # 2: T(C)
        ],
        inner_edges    = [(0, 1), (0, 2), (1, 2)],
        terminal_left  = 0,   # coupler connects toward root
        terminal_right = 1,   # T(L) connects toward leaf
        symmetric      = False,
    ),

    SubgType.RC: SubgDef(
        components     = ["R", "C"],
        color          = "#72B7B2",
        legend         = "R→C",
        attrs          = ["length", "Cc"],
        inner_nodes    = [
            {"type_id": 3, "val_attr": "length", "val2_attr": None},  # 0: R root-side
            {"type_id": 4, "val_attr": "Cc",     "val2_attr": None},  # 1: coupler leaf-side
        ],
        inner_edges    = [(0, 1)],
        terminal_left  = 0,   # R connects toward root
        terminal_right = 1,   # coupler connects toward leaf
        symmetric      = False,
    ),

    SubgType.CR: SubgDef(
        components     = ["C", "R"],
        color          = "#92D7D2",
        legend         = "C→R",
        attrs          = ["Cc", "length"],
        inner_nodes    = [
            {"type_id": 4, "val_attr": "Cc",     "val2_attr": None},  # 0: coupler root-side
            {"type_id": 3, "val_attr": "length", "val2_attr": None},  # 1: R leaf-side
        ],
        inner_edges    = [(0, 1)],
        terminal_left  = 0,   # coupler connects toward root
        terminal_right = 1,   # R connects toward leaf
        symmetric      = False,
    ),

    SubgType.RI: SubgDef(
        components     = ["R", "Ind"],
        color          = "#A73AB6",
        legend         = "R→Ind",
        attrs          = ["length", "D", "l"],
        inner_nodes    = [
            {"type_id": 3, "val_attr": "length", "val2_attr": None},  # 0: R root-side
            {"type_id": 5, "val_attr": "D",      "val2_attr": None},  # 1: I(D)
            {"type_id": 6, "val_attr": "l",      "val2_attr": None},  # 2: I(l) leaf-side
        ],
        inner_edges    = [(0, 1), (1, 2)],
        terminal_left  = 0,   # R connects toward root
        terminal_right = 2,   # I(l) connects toward leaf
        symmetric      = False,
    ),

    SubgType.IR: SubgDef(
        components     = ["Ind", "R"],
        color          = "#C75AD6",
        legend         = "Ind→R",
        attrs          = ["D", "l", "length"],
        inner_nodes    = [
            {"type_id": 5, "val_attr": "D",      "val2_attr": None},  # 0: I(D) root-side
            {"type_id": 6, "val_attr": "l",      "val2_attr": None},  # 1: I(l)
            {"type_id": 3, "val_attr": "length", "val2_attr": None},  # 2: R leaf-side
        ],
        inner_edges    = [(0, 1), (1, 2)],
        terminal_left  = 0,   # I(D) connects toward root
        terminal_right = 2,   # R connects toward leaf
        symmetric      = False,
    ),

    # ── 3-node blocks ─────────────────────────────────────────────────────

    SubgType.TCT: SubgDef(
        # Symmetric: both terminals are Transmons; no root/leaf distinction.
        # attrs order: left-T params, coupler, right-T params
        # (left = lower node_id in the raw graph, determined at merge time)
        components     = ["T", "C", "T"],
        color          = "#FFD700",
        legend         = "T–C–T",
        attrs          = ["L", "C", "Cc", "L2", "C2"],
        inner_nodes    = [
            {"type_id": 1, "val_attr": "L",  "val2_attr": None},  # 0: T1(L)
            {"type_id": 2, "val_attr": "C",  "val2_attr": None},  # 1: T1(C)
            {"type_id": 4, "val_attr": "Cc", "val2_attr": None},  # 2: coupler
            {"type_id": 1, "val_attr": "L2", "val2_attr": None},  # 3: T2(L)
            {"type_id": 2, "val_attr": "C2", "val2_attr": None},  # 4: T2(C)
        ],
        inner_edges    = [(0, 1), (0, 2), (2, 3), (3, 4), (1, 2), (2, 4)],
        terminal_left  = 0,   # T1(L) — left terminal
        terminal_right = 3,   # T2(L) — right terminal
        symmetric      = True,
    ),

    SubgType.RCT: SubgDef(
        # Resonator(root) – C_coupler – Transmon(leaf)
        components     = ["R", "C", "T"],
        color          = "#FF9DA6",
        legend         = "R–C–T",
        attrs          = ["length", "Cc", "L", "C"],
        inner_nodes    = [
            {"type_id": 3, "val_attr": "length", "val2_attr": None},  # 0: R root-side
            {"type_id": 4, "val_attr": "Cc",     "val2_attr": None},  # 1: coupler
            {"type_id": 1, "val_attr": "L",      "val2_attr": None},  # 2: T(L) leaf-side
            {"type_id": 2, "val_attr": "C",      "val2_attr": None},  # 3: T(C)
        ],
        inner_edges    = [(0, 1), (1, 2), (1, 3), (2, 3)],
        terminal_left  = 0,   # R connects toward root
        terminal_right = 2,   # T(L) connects toward leaf
        symmetric      = False,
    ),

    SubgType.TCR: SubgDef(
        # Transmon(root) – C_coupler – Resonator(leaf)
        components     = ["T", "C", "R"],
        color          = "#FFBDA6",
        legend         = "T–C–R",
        attrs          = ["L", "C", "Cc", "length"],
        inner_nodes    = [
            {"type_id": 1, "val_attr": "L",      "val2_attr": None},  # 0: T(L) root-side
            {"type_id": 2, "val_attr": "C",      "val2_attr": None},  # 1: T(C)
            {"type_id": 4, "val_attr": "Cc",     "val2_attr": None},  # 2: coupler
            {"type_id": 3, "val_attr": "length", "val2_attr": None},  # 3: R leaf-side
        ],
        inner_edges    = [(0, 1), (0, 2), (1, 2), (2, 3)],
        terminal_left  = 0,   # T(L) connects toward root
        terminal_right = 3,   # R connects toward leaf
        symmetric      = False,
    ),

    # RIND kept as alias for RI for backward compatibility with old checkpoints.
    # New code should use RI / IR.
    SubgType.RIND: SubgDef(
        components     = ["R", "Ind"],
        color          = "#A73AB6",
        legend         = "R+Ind (legacy)",
        attrs          = ["length", "D", "l"],
        inner_nodes    = [
            {"type_id": 3, "val_attr": "length", "val2_attr": None},
            {"type_id": 5, "val_attr": "D",      "val2_attr": None},
            {"type_id": 6, "val_attr": "l",      "val2_attr": None},
        ],
        inner_edges    = [(0, 1), (1, 2)],
        terminal_left  = 0,
        terminal_right = 2,
        symmetric      = False,
    ),

    SubgType.TIT: SubgDef(
        # Transmon(root) – I_coupler – Transmon(leaf)  [reserved for future use]
        components     = ["T", "Ind", "T"],
        color          = "#6E69F0",
        legend         = "T–Ind–T",
        attrs          = ["L", "C", "D", "l", "L2", "C2"],
        inner_nodes    = [
            {"type_id": 1, "val_attr": "L",  "val2_attr": None},  # 0: T1(L) root-side
            {"type_id": 2, "val_attr": "C",  "val2_attr": None},  # 1: T1(C)
            {"type_id": 5, "val_attr": "D",  "val2_attr": None},  # 2: I(D)
            {"type_id": 6, "val_attr": "l",  "val2_attr": None},  # 3: I(l)
            {"type_id": 1, "val_attr": "L2", "val2_attr": None},  # 4: T2(L) leaf-side
            {"type_id": 2, "val_attr": "C2", "val2_attr": None},  # 5: T2(C)
        ],
        inner_edges    = [(0, 1), (0, 2), (2, 3), (3, 4), (4, 5)],
        terminal_left  = 0,
        terminal_right = 4,
        symmetric      = False,
    ),
}


# ---------------------------------------------------------------------------
# 5. Derived helpers — auto-built from SUBG_DEFS; do not edit manually
# ---------------------------------------------------------------------------

SUBG_NODE:      dict[SubgType, list[str]] = {t: d.components for t, d in SUBG_DEFS.items()}
NODE_COLORS:    dict[SubgType, str]       = {t: d.color      for t, d in SUBG_DEFS.items()}
LEGEND_ENTRIES: list[tuple]               = [(t, d.legend)   for t, d in SUBG_DEFS.items()]
