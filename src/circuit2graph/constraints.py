"""
circuit2graph/constraints.py
============================
Physical topology constraints for cQED macro-node graphs.

This module answers one question: given two macro-nodes (each described by a
SubgType and a compression direction), is a direct edge between them physically
valid?

The answer is derived entirely from the existing expansion logic in
expansion.py — no rules are hardcoded here.  Adding a new SubgType to
definitions.py automatically propagates into the compatibility table.

THEORY
------
Every macro-node is a linear chain of primitives.  When two macro-nodes are
connected by a macro edge, the wiring convention (from expansion.py) is:

    max_index(u)  ←—edge—→  min_index(v)

i.e. the rightmost primitive of u connects to the leftmost primitive of v.

The only physical rule at the primitive level is:

    a connection is valid iff exactly one of the two primitives
    involved is a coupler (C_COUPLER or I_COUPLER).

    coupler–coupler  → illegal  (coupler would have degree > 2)
    T/R/F – T/R/F   → illegal  (no coupler mediating the interaction)
    coupler – T/R/F → legal
    T/R/F – coupler → legal

EXPORTED API
------------
outer_ports(subg_type, direction) -> tuple[SubgType, SubgType]
    Returns (left_port, right_port) primitive types for a macro-node.

edge_compatible(type_u, dir_u, type_v, dir_v) -> bool
    True iff a direct macro edge (u → v) is physically valid.

COMPAT : dict[(SubgType, float), dict[(SubgType, float), bool]]
    Pre-computed compatibility table.  Lookup is O(1).
    Keys use direction ∈ {+1.0, -1.0, 0.0}, where 0.0 means "no direction"
    (primitives and TCT).

port_budget(node_types, directions) -> PortBudget
    Counts open coupler and non-coupler external ports for a partial
    sequence of macro-nodes (used by the node-generation mask).

can_terminate(budget) -> bool
    True iff the current port budget can yield at least one connected,
    physically valid graph.

possible_dirs(subg_type) -> list[float]
    Returns the possible compression directions for a macro-node type.

allowed_macro_edges(node_types, directions, oriented=False) -> list[tuple[int, int]]
    Returns the macro-edge pairs that are physically admissible for the
    generated macro-node set.

exists_valid_macro_graph(node_types, directions) -> bool
    True iff the current macro-node set admits at least one connected macro
    graph using only physically valid edges. This is the CCGVAE-style
    compatibility test for partial node generation.

valid_next_macro_types(node_types, directions, candidates=None) -> list[int]
    Returns macro-node types that can be appended while preserving the
    existence of at least one connected physically valid macro graph.

choose_compatible_macro_edge_orientation(u, v, node_types, directions) -> tuple[int, int] | None
    Returns the physically valid orientation for an undirected macro-edge pair.

valid_next_types(budget, candidates) -> list[SubgType]
    Legacy port-budget heuristic. Kept for backwards compatibility, but the
    preferred CCGVAE-style node mask is valid_next_macro_types().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from circuit2graph.definitions import SubgType, SUBG_DEFS
from circuit2graph.topology import CQEDNode
from circuit2graph.expansion import expand_node_to_primitives, PRIMITIVE_TYPES


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_COUPLER_PRIMITIVES: frozenset[SubgType] = frozenset({
    SubgType.C_COUPLER,
    SubgType.I_COUPLER,
})

_NONCOUPLER_PRIMITIVES: frozenset[SubgType] = frozenset({
    SubgType.TRANSMON,
    SubgType.RESONATOR,
    SubgType.FEEDLINE,
})


def _make_dummy_node(subg_type: SubgType, direction: float) -> CQEDNode:
    """
    Construct a minimal CQEDNode for the sole purpose of calling
    expand_node_to_primitives().

    Physical attrs are irrelevant for port computation — only subg_type and
    attrs['dir'] matter.  We use 0.0 for every numerical attribute.
    """
    attrs: dict = {}
    if "dir" in SUBG_DEFS[subg_type].attrs:
        attrs["dir"] = float(direction)
    return CQEDNode(subg_type=subg_type, attrs=attrs, node_id=0, label=None)


def _canonical_dir(subg_type: SubgType, direction: float) -> float:
    """
    Return the canonical direction key used in COMPAT.

    For macro-nodes that have no 'dir' attribute (primitives, TCT which is
    symmetric), direction is irrelevant — normalise to 0.0 so that the COMPAT
    table has a single entry instead of two identical ones.
    """
    if "dir" not in SUBG_DEFS[subg_type].attrs:
        return 0.0
    return 1.0 if float(direction) >= 0.0 else -1.0


# ---------------------------------------------------------------------------
# Public: outer_ports
# ---------------------------------------------------------------------------

def outer_ports(subg_type: SubgType, direction: float) -> tuple[SubgType, SubgType]:
    """
    Return the (left_port, right_port) primitive types of a macro-node.

    The left port is the primitive at index 0 of the expansion chain; the
    right port is the primitive at index -1.  For primitive singletons both
    ports are the node's own type.

    Parameters
    ----------
    subg_type : SubgType
        The macro-node type.
    direction : float
        Compression direction (+1 canonical, -1 reversed, 0 = no direction).

    Returns
    -------
    (left_port, right_port) : tuple[SubgType, SubgType]
        SubgType of the leftmost and rightmost primitive in the chain.

    Examples
    --------
    >>> outer_ports(SubgType.TC, +1)          # T–C  →  left=T, right=C
    (SubgType.TRANSMON, SubgType.C_COUPLER)
    >>> outer_ports(SubgType.TC, -1)          # C–T  →  left=C, right=T
    (SubgType.C_COUPLER, SubgType.TRANSMON)
    >>> outer_ports(SubgType.TCT, +1)         # T–C–T, symmetric
    (SubgType.TRANSMON, SubgType.TRANSMON)
    >>> outer_ports(SubgType.C_COUPLER, 0)    # primitive singleton
    (SubgType.C_COUPLER, SubgType.C_COUPLER)
    >>> outer_ports(SubgType.RCT, +1)         # R–C–T
    (SubgType.RESONATOR, SubgType.TRANSMON)
    >>> outer_ports(SubgType.RCT, -1)         # T–C–R
    (SubgType.TRANSMON, SubgType.RESONATOR)
    """
    node = _make_dummy_node(subg_type, direction)
    chain = expand_node_to_primitives(node)
    return chain[0].subg_type, chain[-1].subg_type


# ---------------------------------------------------------------------------
# Public: edge_compatible
# ---------------------------------------------------------------------------

def edge_compatible(
    type_u: SubgType,
    dir_u:  float,
    type_v: SubgType,
    dir_v:  float,
) -> bool:
    """
    Return True iff a direct macro edge (u → v) is physically valid.

    Convention: the macro edge connects right_port(u) to left_port(v),
    following the expansion wiring rule in expansion.py.

    Physical validity rule (XOR):
        exactly one of {right_port(u), left_port(v)} must be a coupler.

    Parameters
    ----------
    type_u, dir_u : SubgType, float
        Type and direction of the source node u.
    type_v, dir_v : SubgType, float
        Type and direction of the destination node v.

    Returns
    -------
    bool

    Examples
    --------
    >>> edge_compatible(SubgType.TC, +1, SubgType.TC, -1)   # C–C  False
    False
    >>> edge_compatible(SubgType.TC, +1, SubgType.TCT, 0)   # C–T  True
    True
    >>> edge_compatible(SubgType.TCT, 0, SubgType.TCT, 0)   # T–T  False
    False
    >>> edge_compatible(SubgType.C_COUPLER, 0, SubgType.TRANSMON, 0)  # C–T True
    True
    """
    _, right_u = outer_ports(type_u, dir_u)
    left_v, _  = outer_ports(type_v, dir_v)

    u_is_coupler = right_u in _COUPLER_PRIMITIVES
    v_is_coupler = left_v  in _COUPLER_PRIMITIVES

    # XOR: exactly one side must be a coupler
    return u_is_coupler ^ v_is_coupler


# ---------------------------------------------------------------------------
# Public: COMPAT — pre-computed compatibility matrix
# ---------------------------------------------------------------------------

def _build_compat() -> dict[tuple[SubgType, float], dict[tuple[SubgType, float], bool]]:
    """
    Build the full pairwise compatibility table at module import time.

    Keys are (SubgType, canonical_direction) pairs.  Direction is normalised
    by _canonical_dir so that types without a direction attribute have a
    single key (subg_type, 0.0) instead of two identical entries.

    The table is keyed as COMPAT[key_u][key_v] = bool.
    """
    # Collect all distinct (type, canonical_dir) keys
    keys: list[tuple[SubgType, float]] = []
    for st in SubgType:
        has_dir = "dir" in SUBG_DEFS[st].attrs
        if has_dir:
            keys.append((st, 1.0))
            keys.append((st, -1.0))
        else:
            keys.append((st, 0.0))

    compat: dict[tuple[SubgType, float], dict[tuple[SubgType, float], bool]] = {}
    for ku in keys:
        compat[ku] = {}
        for kv in keys:
            compat[ku][kv] = edge_compatible(ku[0], ku[1], kv[0], kv[1])

    return compat


COMPAT: dict[tuple[SubgType, float], dict[tuple[SubgType, float], bool]] = _build_compat()


def is_compatible(
    type_u: SubgType | int,
    dir_u:  float,
    type_v: SubgType | int,
    dir_v:  float,
) -> bool:
    """
    O(1) compatibility lookup via the pre-computed COMPAT table.

    Accepts SubgType or raw int for type arguments (decoder stores ints).
    Direction is normalised internally.

    Parameters
    ----------
    type_u, dir_u : SubgType or int, float
    type_v, dir_v : SubgType or int, float

    Returns
    -------
    bool
    """
    type_u = SubgType(int(type_u))
    type_v = SubgType(int(type_v))
    ku = (type_u, _canonical_dir(type_u, dir_u))
    kv = (type_v, _canonical_dir(type_v, dir_v))
    return COMPAT[ku][kv]


# ---------------------------------------------------------------------------
# CCGVAE-style macro-node feasibility helpers
# ---------------------------------------------------------------------------

def possible_dirs(subg_type: SubgType | int) -> list[float]:
    """
    Return all possible compression directions for a macro-node type.

    Directional macro-nodes are represented by both +1 and -1.  Non-
    directional macro-nodes use the canonical 0.0 direction.  This helper is
    intentionally small and deterministic so it can be used inside the
    autoregressive decoder before a direction has been sampled.
    """
    st = SubgType(int(subg_type))
    if "dir" in SUBG_DEFS[st].attrs:
        return [1.0, -1.0]
    return [0.0]


def allowed_macro_edges(
    node_types: list[int],
    directions: list[float],
    oriented: bool = False,
) -> list[tuple[int, int]]:
    """
    Return all physically admissible macro edges for a macro-node set.

    Parameters
    ----------
    node_types, directions
        Generated macro-node types and compression directions.
    oriented
        If False, an undirected pair (j, i) is returned when at least one of
        the two orientations is physically valid.  This is the right mode for
        testing whether a connected physical graph exists.

        If True, the returned pair orientation is the one that is physically
        valid.  If both orientations are valid, both are returned.

    Returns
    -------
    list[tuple[int, int]]
        Pairs of node indices.  For oriented=False, pairs follow the decoder's
        lower-triangular convention (j, i) with j < i.
    """
    if len(node_types) != len(directions):
        raise ValueError(
            "node_types and directions must have the same length "
            f"(got {len(node_types)} and {len(directions)})."
        )

    N = len(node_types)
    edges: list[tuple[int, int]] = []

    for i in range(N):
        for j in range(i):
            ok_ji = is_compatible(
                node_types[j], directions[j],
                node_types[i], directions[i],
            )
            ok_ij = is_compatible(
                node_types[i], directions[i],
                node_types[j], directions[j],
            )

            if oriented:
                if ok_ji:
                    edges.append((j, i))
                if ok_ij:
                    edges.append((i, j))
            elif ok_ji or ok_ij:
                edges.append((j, i))

    return edges


def _is_connected_from_edges(n_nodes: int, edges: list[tuple[int, int]]) -> bool:
    """Return True iff the undirected graph induced by edges is connected."""
    if n_nodes <= 1:
        return True
    if not edges:
        return False

    adj: list[list[int]] = [[] for _ in range(n_nodes)]
    for u, v in edges:
        u = int(u)
        v = int(v)
        if 0 <= u < n_nodes and 0 <= v < n_nodes and u != v:
            adj[u].append(v)
            adj[v].append(u)

    seen: set[int] = set()
    stack: list[int] = [0]
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        for v in adj[u]:
            if v not in seen:
                stack.append(v)

    return len(seen) == n_nodes


def exists_valid_macro_graph(
    node_types: list[int],
    directions: list[float],
) -> bool:
    """
    Return True iff the macro-node set admits a connected physical graph.

    This is the cQED analogue of the CCGVAE histogram-compatibility check:
    during node generation we do not need to decide the final edge set yet; we
    only need to know whether there exists at least one connected graph that
    can be built with physically valid macro edges.

    Implementation detail: build the graph of all admissible macro-edge pairs
    and test whether that admissibility graph is connected.  If it is
    connected, at least one connected physical subgraph exists, e.g. any
    spanning tree of the admissibility graph.
    """
    if len(node_types) != len(directions):
        raise ValueError(
            "node_types and directions must have the same length "
            f"(got {len(node_types)} and {len(directions)})."
        )

    N = len(node_types)
    if N <= 1:
        return True

    allowed = allowed_macro_edges(node_types, directions, oriented=False)
    return _is_connected_from_edges(N, allowed)


def valid_next_macro_types(
    node_types: list[int],
    directions: list[float],
    candidates: Iterable[int] | None = None,
    allow_empty_fallback: bool = True,
) -> list[int]:
    """
    Return macro-node types that can be appended without killing feasibility.

    A candidate type is accepted if at least one of its possible compression
    directions keeps the partial macro-node set compatible with at least one
    connected physically valid graph.  This is the function that should be
    used by the decoder to mask node-type logits, following the same workflow
    used by CCGVAE for valence-histogram compatibility.

    Parameters
    ----------
    node_types, directions
        Partial macro-node sequence generated so far.
    candidates
        Candidate SubgType integers to test.  Defaults to every SubgType.
    allow_empty_fallback
        If True, return the original candidates when the mask would be empty.
        This prevents autoregressive generation from crashing.  Set False in
        tests/debug scripts when you want strict behavior.
    """
    if len(node_types) != len(directions):
        raise ValueError(
            "node_types and directions must have the same length "
            f"(got {len(node_types)} and {len(directions)})."
        )

    if candidates is None:
        candidates_list = [int(st) for st in SubgType]
    else:
        candidates_list = [int(c) for c in candidates]

    valid: list[int] = []
    for nt in candidates_list:
        for d in possible_dirs(nt):
            trial_types = node_types + [int(nt)]
            trial_dirs = directions + [float(d)]
            if exists_valid_macro_graph(trial_types, trial_dirs):
                valid.append(int(nt))
                break

    if not valid and allow_empty_fallback:
        return candidates_list
    return valid


# ---------------------------------------------------------------------------
# Port budget — legacy heuristic kept for backwards compatibility
# ---------------------------------------------------------------------------

@dataclass
class PortBudget:
    """
    Counts of open external ports in a partial macro-node sequence.

    A "port" is one side of a macro-node that will require an external edge.
    Each macro-node contributes exactly two ports (left and right).  Internal
    ports of compound macro-nodes (e.g. the C inside TCT) are already
    saturated by the internal chain edges and are NOT counted here.

    Attributes
    ----------
    coupler_ports : int
        Number of open coupler (C or I) primitive ports across all nodes.
    noncoupler_ports : int
        Number of open non-coupler (T, R, F) primitive ports across all nodes.
    n_nodes : int
        Total number of macro-nodes added so far.
    """
    coupler_ports:    int = 0
    noncoupler_ports: int = 0
    n_nodes:          int = 0

    def add(self, subg_type: SubgType | int, direction: float) -> "PortBudget":
        """Return a NEW PortBudget with one more node added (immutable update)."""
        subg_type = SubgType(int(subg_type))
        left, right = outer_ports(subg_type, direction)
        dc = sum(1 for p in (left, right) if p in _COUPLER_PRIMITIVES)
        dn = sum(1 for p in (left, right) if p in _NONCOUPLER_PRIMITIVES)
        return PortBudget(
            coupler_ports    = self.coupler_ports    + dc,
            noncoupler_ports = self.noncoupler_ports + dn,
            n_nodes          = self.n_nodes + 1,
        )


def port_budget(
    node_types: list[int],
    directions: list[float],
) -> PortBudget:
    """
    Compute the PortBudget for a (partial) macro-node sequence.

    Parameters
    ----------
    node_types : list[int]
        SubgType int values of the generated nodes so far.
    directions : list[float]
        Compression directions (+1, -1, 0) aligned with node_types.

    Returns
    -------
    PortBudget
    """
    budget = PortBudget()
    for nt, d in zip(node_types, directions):
        budget = budget.add(nt, d)
    return budget


# ---------------------------------------------------------------------------
# Termination feasibility — used to mask END token (Phase 2)
# ---------------------------------------------------------------------------

def can_terminate(budget: PortBudget) -> bool:
    """
    Return True iff the current port budget can yield at least one connected,
    physically valid graph completion.

    Necessary and sufficient conditions (conservative check, O(1)):

    1. At least 2 nodes must exist (a single isolated node has no edges and
       is trivially "connected" but physically meaningless — we require n >= 2
       only when the user has requested multi-component circuits).
       NOTE: single-node graphs are allowed (n==1 is valid for a lone primitive
       like a standalone Transmon in a test topology), but in practice the VAE
       will almost never emit them.

    2. For a connected graph to exist where every edge is coupler–noncoupler:
         every coupler port must be paired with a noncoupler port.
         every noncoupler port must be paired with a coupler port.
       Since each edge consumes exactly one coupler port and one noncoupler port,
       a valid matching exists iff:
           coupler_ports == noncoupler_ports
       or more conservatively (allows some ports to remain unconnected, which is
       fine for non-terminal nodes in a tree topology):
           coupler_ports >= 1  and  noncoupler_ports >= 1
           and coupler_ports <= noncoupler_ports * max_degree_coupler
       where max_degree_coupler = 2 (each coupler has exactly two external ports
       that can connect, one on each side).

       Simplified rule used here:
           noncoupler_ports > 0  →  coupler_ports >= ceil(noncoupler_ports / 2)
           coupler_ports > 0     →  noncoupler_ports >= 1

    3. Edge case: if all ports are coupler (e.g. a sequence of bare C_COUPLERs
       with no T/R/F), no valid graph exists.

    This check is intentionally conservative — it may occasionally block a
    type that would technically allow a valid graph — but false negatives are
    far less harmful than generating physically invalid circuits.

    IMPORTANT NOTE on single-node graphs:
    A single macro-node is always a physically valid circuit.  A TCT, for
    example, expands to the primitive chain T–C–T which is already complete —
    it does not need external edges.  The port budget measures EXTERNAL ports
    that need to be connected to OTHER macro-nodes; for a standalone node
    those ports simply remain unconnected (the circuit is self-contained).

    Therefore: if n_nodes <= 1, always return True.

    IMPORTANT NOTE on degree:
    A single T/R/F node CAN connect to multiple couplers (one per edge),
    because each edge involves a *different* coupler port.  The constraint
    is at the port level, not at the node level.  Example:

            T
            |
            C (standalone C_COUPLER)
            |
        R--I--R--C--T

    Here R has macro-degree 2 (connected to I and C via separate coupler ports).
    This is valid.

    The port-level rule is: each macro edge consumes exactly one coupler port
    and one noncoupler port.  A valid completion requires that every open port
    is eventually covered by a compatible edge.

    Conservative feasibility condition:
        Both coupler_ports >= 1 AND noncoupler_ports >= 1 must hold
        (so at least one valid edge can exist).

        Additionally:
        - If noncoupler_ports > coupler_ports:
              each noncoupler port needs its own coupler port (1:1).
              → BLOCK if noncoupler_ports > coupler_ports.
        - If coupler_ports > noncoupler_ports:
              excess couplers can terminate chains → allowed.

    In summary:
        INVALID iff  n_nodes > 1  and  (
            (c == 0 and n > 0)  or  (n == 0 and c > 0)  or  (n > c)
        )
    """
    # A single macro-node is always a complete, valid circuit
    if budget.n_nodes <= 1:
        return True

    c = budget.coupler_ports
    n = budget.noncoupler_ports

    # Empty ports with multiple nodes: degenerate (isolated nodes), invalid
    if c == 0 and n == 0:
        return False

    # Only noncouplers with no couplers: impossible
    if n > 0 and c == 0:
        return False

    # Only couplers with no noncouplers: floating couplers, invalid for cQED
    if c > 0 and n == 0:
        return False

    # Each noncoupler port needs its own coupler port (1:1 matching)
    if n > c:
        return False

    return True


# ---------------------------------------------------------------------------
# Valid next types — used to mask node-type logits (Phase 2)
# ---------------------------------------------------------------------------

def valid_next_types(
    budget:     PortBudget,
    candidates: Iterable[int],
    direction_hint: float = 1.0,
) -> list[int]:
    """
    Return the subset of candidate SubgType ints whose addition still allows
    a physically valid graph completion.

    Called at each step of autoregressive node generation, before sampling.
    A type is valid iff adding it (with its two ports) results in a PortBudget
    that satisfies can_terminate().

    Note: direction is not yet known at node-type selection time (it is
    predicted separately by head_dir after the type is chosen).  We therefore
    check both directions and accept the type if EITHER direction allows a
    valid completion.  This is slightly optimistic but avoids over-masking.

    Parameters
    ----------
    budget : PortBudget
        Current open-port counts.
    candidates : Iterable[int]
        SubgType int values to filter.
    direction_hint : float
        Direction to try first.  Both +1 and -1 are always tried for
        directional types; this parameter is kept for API clarity.

    Returns
    -------
    list[int]
        Filtered list.  If filtering would leave no candidates (e.g. only
        one legal type and it just got masked), the original list is returned
        as a safety fallback so generation never gets stuck.
    """
    valid: list[int] = []
    for nt in candidates:
        st = SubgType(int(nt))
        has_dir = "dir" in SUBG_DEFS[st].attrs
        dirs = [1.0, -1.0] if has_dir else [0.0]
        for d in dirs:
            candidate_budget = budget.add(st, d)
            if can_terminate(candidate_budget):
                valid.append(nt)
                break   # one valid direction is enough

    # Safety fallback: never leave zero valid types
    if not valid:
        return list(candidates)

    return valid


# ---------------------------------------------------------------------------
# Edge orientation helper
# ---------------------------------------------------------------------------

def choose_compatible_macro_edge_orientation(
    u: int,
    v: int,
    node_types: list[int],
    directions: list[float],
) -> tuple[int, int] | None:
    """Return a physically valid orientation for an undirected macro edge.

    The edge decoder produces one logit for each unordered lower-triangular
    pair. However, the expansion convention is oriented: an edge ``a -> b``
    connects ``right_port(a)`` to ``left_port(b)``. Therefore, when an
    unordered pair is selected, we must store it in the orientation that is
    physically compatible.

    This is the minimal loop fix: a pair such as TCT--TC may be invalid in the
    decoder's default lower-triangular orientation but valid in the reverse
    orientation. Returning the reverse orientation allows loop-closing edges
    like TCT--C--TC--TCT without introducing greedy port-saturation logic.

    If both orientations are valid, the input order is preferred for
    reproducibility. If neither orientation is valid, returns None.
    """
    u = int(u)
    v = int(v)
    if is_compatible(node_types[u], directions[u], node_types[v], directions[v]):
        return (u, v)
    if is_compatible(node_types[v], directions[v], node_types[u], directions[u]):
        return (v, u)
    return None


# ---------------------------------------------------------------------------
# Convenience: build_forbidden_edge_mask
# ---------------------------------------------------------------------------

def build_forbidden_edge_mask(
    node_types: list[int],
    directions: list[float],
) -> list[bool]:
    """
    Build a boolean mask over all N*(N-1)/2 macro-node pairs in lower-
    triangular order (same ordering as _forward_edges in decoder.py).

    mask[k] = True  →  edge pair at position k is physically forbidden in both orientations.
    mask[k] = False →  edge is potentially valid.

    The ordering matches the decoder loop:
        for i in range(N):
            for j in range(i):   # j < i
                ...

    Parameters
    ----------
    node_types : list[int]   SubgType ints (length N)
    directions : list[float] compression directions (length N)

    Returns
    -------
    list[bool]  length = N*(N-1)//2

    Usage in decoder.py
    -------------------
        import torch
        from circuit2graph.constraints import build_forbidden_edge_mask

        mask_list = build_forbidden_edge_mask(node_types, directions)
        forbidden = torch.tensor(mask_list, dtype=torch.bool, device=device)
        edge_logits[0, forbidden, 0] = -1e9
    """
    if len(node_types) != len(directions):
        raise ValueError(
            "node_types and directions must have the same length "
            f"(got {len(node_types)} and {len(directions)})."
        )

    N = len(node_types)
    mask: list[bool] = []
    for i in range(N):
        for j in range(i):
            # The edge decoder scores an unordered pair (j, i), while the
            # primitive expansion is oriented. We therefore forbid the pair
            # only when neither orientation can be materialized physically.
            # The decoder later stores the accepted edge in the compatible
            # orientation using choose_compatible_macro_edge_orientation().
            orient = choose_compatible_macro_edge_orientation(
                j, i, node_types, directions
            )
            mask.append(orient is None)
    return mask


# ---------------------------------------------------------------------------
# Debug / introspection helper
# ---------------------------------------------------------------------------

def print_compat_table(show_ports: bool = True) -> None:
    """
    Pretty-print the compatibility matrix and, optionally, the port table.

    Useful for manual verification.  Example output:

        outer_ports:
          TRANSMON  (dir= 0) : left=TRANSMON   right=TRANSMON
          TC        (dir=+1) : left=TRANSMON   right=C_COUPLER
          TC        (dir=-1) : left=C_COUPLER  right=TRANSMON
          ...

        compatibility (✓ = valid edge, · = forbidden):
                          TRANSMON TC+1 TC-1 ...
          TRANSMON           ·      ✓    ✓  ...
          TC      (+1)       ✓      ·    ·  ...
          ...
    """
    keys = sorted(COMPAT.keys(), key=lambda k: (k[0].value, k[1]))

    if show_ports:
        print("outer_ports:")
        for st in SubgType:
            has_dir = "dir" in SUBG_DEFS[st].attrs
            dirs = [1.0, -1.0] if has_dir else [0.0]
            for d in dirs:
                left, right = outer_ports(st, d)
                dir_str = f"+{d:.0f}" if d >= 0 else f"{d:.0f}"
                if not has_dir:
                    dir_str = " 0"
                print(f"  {st.name:<12} (dir={dir_str}) : "
                      f"left={left.name:<12} right={right.name}")
        print()

    print("compatibility (✓ = valid edge, · = forbidden):")
    header = "  " + " ".join(
        f"{k[0].name[:6]}{'+' if k[1]>0 else ('-' if k[1]<0 else '0'):1}"
        for k in keys
    )
    print(header)
    for ku in keys:
        row = f"  {ku[0].name[:8]:<8}({'+' if ku[1]>0 else ('-' if ku[1]<0 else '0')}1)  "
        row += "  ".join("✓" if COMPAT[ku][kv] else "·" for kv in keys)
        print(row)
