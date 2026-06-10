"""
compression.py
==============

Graph-compression utilities for the cQED representation.

This module transforms a raw CQEDTopology, where each node is a primitive
circuit element, into a compressed topology where nodes correspond to meaningful
subgraph blocks such as RCT, TCT or RI.

The file contains the complete compression pipeline:
- root selection for deterministic traversal
- low-level branch walking utilities
- explicit two-node and three-node merge rules
- a lazy post-pass for remaining coupler-centered patterns
- graphlize(), the public entry point used by the data loader

Keeping this logic in one file makes the compression algorithm readable while
leaving static definitions in definitions.py and data structures in topology.py.
"""

from circuit2graph.definitions import SubgType
from circuit2graph.topology import CQEDNode, CQEDTopology


# ===========================================================================
# Low-level graph traversal utilities
# ===========================================================================

def compute_eccentricity(node_id: int, adj: dict) -> int:
    """
    Node eccentricity = max BFS distance to any other reachable node.
    """
    dist  = {node_id: 0}
    queue = [node_id]
    while queue:
        curr = queue.pop(0)
        for nb in adj[curr]:
            if nb not in dist:
                dist[nb] = dist[curr] + 1
                queue.append(nb)
    return max(dist.values()) if dist else 0


def walk_branch(
    start_id: int,
    from_id:  int,
    adj:      dict,
    consumed: set,
) -> list[int]:
    """
    Walk a branch from start_id until a leaf (degree == 1) or hub (degree > 2).
    Returns the ordered list of node_ids visited (not including from_id).
    """
    path: list[int] = []
    curr = start_id
    prev = from_id

    while True:
        if curr in consumed:
            break
        path.append(curr)
        degree = len(adj[curr])
        if degree == 1:
            break
        if degree > 2 and curr != from_id:
            break
        nexts = [nb for nb in adj[curr] if nb != prev and nb not in consumed]
        if not nexts:
            break
        prev = curr
        curr = nexts[0]

    return path


# ===========================================================================
# Root selection
# ===========================================================================

# Priority for the root node (lower = preferred).
# Extend this dict to change traversal priority for new SubgTypes.
ROOT_PRIORITY: dict[SubgType, int] = {
    SubgType.FEEDLINE:  0,
    SubgType.C_COUPLER: 1,
    SubgType.I_COUPLER: 2,
    SubgType.RESONATOR: 3,
    SubgType.TRANSMON:  4,
}


def find_root(topology: CQEDTopology) -> int:
    """
    Choose the root of the spanning-tree traversal.

    Priority: FEEDLINE > COUPLER > RESONATOR > TRANSMON.
    Ties broken by: highest degree → lowest eccentricity → lowest node_id.
    """
    adj      = topology._adj()
    priority = {n.node_id: ROOT_PRIORITY.get(n.subg_type, 99) for n in topology._nodes}

    candidates: dict[int, list[int]] = {}
    for n in topology._nodes:
        candidates.setdefault(priority[n.node_id], []).append(n.node_id)

    best_group = candidates[min(candidates)]
    if len(best_group) == 1:
        return best_group[0]

    max_deg    = max(len(adj[nid]) for nid in best_group)
    best_group = [nid for nid in best_group if len(adj[nid]) == max_deg]
    best_group.sort(key=lambda nid: (compute_eccentricity(nid, adj), nid))
    return best_group[0]


# ===========================================================================
# Merge rules
# ===========================================================================

# HOW TO ADD A NEW MERGE PATTERN
# --------------------------------
# Adding a new pattern requires touching exactly TWO files:
#     1. circuit2graph/definitions.py  — add SubgType member + SubgDef entry
#     2. THIS file                     — add the merge rule (read below)
#
# WHERE to add the rule depends on when the pattern becomes visible:
#
#   A) Pattern is a linear chain in the RAW graph
#      (e.g. T-C-T, T-C-R already contiguous before any merge)
#        → add an if-block in try_merge_3() or try_merge_2()
#          (3-node or 2-node patterns respectively).
#
#   B) Pattern only emerges AFTER the first compression pass
#      (i.e. two compressed blocks end up flanking a coupler that
#       was a hub in the raw graph)
#        → add a tuple to LAZY_PATTERNS.
#
#   C) Pattern can appear in BOTH situations
#        → add it in both A and B (like RCT currently does).
#
# Nothing else needs to change.


def _make_node(
    subg_type: SubgType,
    attrs:     dict,
    label:     str,
    tmp_id:    int = 0,
) -> CQEDNode:
    """Create a new (not-yet-inserted) CQEDNode with a placeholder id."""
    return CQEDNode(subg_type=subg_type, attrs=attrs, node_id=tmp_id, label=label)


def _ordered_label(*nodes: CQEDNode) -> str:
    """Human-readable primitive order stored inside the macro-node label."""
    return "+".join(str(n.label) for n in nodes)


def try_merge_3(
    n1:      CQEDNode,
    n2:      CQEDNode,
    n3:      CQEDNode,
    adj_raw: dict,
) -> CQEDNode | None:
    """
    Try to merge an ordered triplet (n1, n2-coupler, n3) into a known
    3-node pattern.  The order is the traversal order from the root along the
    branch/cycle.  A ``dir`` attribute is stored so expansion can recover
    whether the internal primitive order is canonical (+1) or reversed (-1).
    """
    # Only merge through degree-2 capacitive couplers.
    if n2.subg_type != SubgType.C_COUPLER or len(adj_raw[n2.node_id]) != 2:
        return None

    types = {n1.subg_type, n3.subg_type}

    # R-C-T (+1) or T-C-R (-1) -> one RCT macro-node.
    if types == {SubgType.TRANSMON, SubgType.RESONATOR}:
        t = n1 if n1.subg_type == SubgType.TRANSMON  else n3
        r = n1 if n1.subg_type == SubgType.RESONATOR else n3
        direction = 1.0 if (
            n1.subg_type == SubgType.RESONATOR and n3.subg_type == SubgType.TRANSMON
        ) else -1.0
        return _make_node(
            SubgType.RCT,
            attrs = {
                "length": r.attrs.get("length", 0.0),
                "Cc":     n2.attrs.get("Cc",     0.0),
                "L":      t.attrs.get("L",      0.0),
                "C":      t.attrs.get("C",      0.0),
                "dir":    direction,
            },
            label = f"RCT({_ordered_label(n1, n2, n3)})",
        )

    # T-C-T.  The two transmons are deliberately kept in traversal order;
    # do not sort by L, otherwise compression is not invertible.
    if types == {SubgType.TRANSMON}:
        return _make_node(
            SubgType.TCT,
            attrs = {
                "L":   n1.attrs.get("L",  0.0),
                "C":   n1.attrs.get("C",  0.0),
                "Cc":  n2.attrs.get("Cc", 0.0),
                "L2":  n3.attrs.get("L",  0.0),
                "C2":  n3.attrs.get("C",  0.0),
                "dir": 1.0,
            },
            label = f"TCT({_ordered_label(n1, n2, n3)})",
        )

    return None

def try_merge_2(
    n1:      CQEDNode,
    n2:      CQEDNode,
    adj_raw: dict,
) -> CQEDNode | None:
    """
    Try to merge an ordered pair into a known 2-node pattern.

    Unlike the old implementation, the coupler may be either first or second
    in the pair.  This makes F-C-R compress to F-CR, while F-R-C compresses to
    F-RC.  The SubgType is still RC in both cases; attrs["dir"] is -1 for CR
    and +1 for RC.
    """
    # Determine whether exactly one side is a coupler.
    if n1.subg_type in {SubgType.C_COUPLER, SubgType.I_COUPLER}:
        coupler, other = n1, n2
        coupler_first = True
    elif n2.subg_type in {SubgType.C_COUPLER, SubgType.I_COUPLER}:
        coupler, other = n2, n1
        coupler_first = False
    else:
        return None

    # Do not absorb hub couplers.  Leaf and through-couplers are allowed:
    # this preserves examples such as F-R-C -> F-RC.
    if len(adj_raw[coupler.node_id]) > 2:
        return None

    # Capacitive patterns.
    if coupler.subg_type == SubgType.C_COUPLER:
        if other.subg_type == SubgType.TRANSMON:  # T-C (+1) or C-T (-1) -> TC
            direction = -1.0 if coupler_first else 1.0
            return _make_node(
                SubgType.TC,
                attrs = {
                    "L":   other.attrs.get("L",  0.0),
                    "C":   other.attrs.get("C",  0.0),
                    "Cc":  coupler.attrs.get("Cc", 0.0),
                    "dir": direction,
                },
                label = f"TC({_ordered_label(n1, n2)})",
            )

        if other.subg_type == SubgType.RESONATOR:  # R-C (+1) or C-R (-1) -> RC
            direction = -1.0 if coupler_first else 1.0
            return _make_node(
                SubgType.RC,
                attrs = {
                    "length": other.attrs.get("length", 0.0),
                    "Cc":     coupler.attrs.get("Cc",     0.0),
                    "dir":    direction,
                },
                label = f"RC({_ordered_label(n1, n2)})",
            )

    # Inductive resonator coupler: R-Ind (+1) or Ind-R (-1) -> RI.
    if coupler.subg_type == SubgType.I_COUPLER and other.subg_type == SubgType.RESONATOR:
        direction = -1.0 if coupler_first else 1.0
        return _make_node(
            SubgType.RI,
            attrs = {
                "length": other.attrs.get("length", 0.0),
                "D":      coupler.attrs.get("D",      0.0),
                "l":      coupler.attrs.get("l",      0.0),
                "dir":    direction,
            },
            label = f"RI({_ordered_label(n1, n2)})",
        )

    return None


# ---------------------------------------------------------------------------
# Lazy merge patterns (post-branch-decomposition pass)
# ---------------------------------------------------------------------------

# Each entry is a 4-tuple:
#   (frozenset of the two outer SubgTypes,
#    merged SubgType,
#    attr_builder(coupler, na, nb) → dict,
#    label_builder(coupler, na, nb) → str)
#
# ADD NEW LAZY PATTERNS HERE — one tuple per pattern.

def _node_rank(node: CQEDNode) -> tuple[int, int]:
    """Traversal rank used as a deterministic proxy for distance from root."""
    return (node.node_id, id(node))


def _ordered_lazy_endpoints(na: CQEDNode, nb: CQEDNode) -> tuple[CQEDNode, CQEDNode]:
    """Return endpoints in deterministic root-to-leaf / canonical order."""
    return (na, nb) if _node_rank(na) <= _node_rank(nb) else (nb, na)


def _build_lazy_rct(coupler: CQEDNode, na: CQEDNode, nb: CQEDNode) -> CQEDNode:
    first, second = _ordered_lazy_endpoints(na, nb)
    t = first if first.subg_type == SubgType.TRANSMON else second
    r = first if first.subg_type == SubgType.RESONATOR else second
    direction = 1.0 if (
        first.subg_type == SubgType.RESONATOR and second.subg_type == SubgType.TRANSMON
    ) else -1.0
    return CQEDNode(
        subg_type = SubgType.RCT,
        attrs = {
            "length": r.attrs.get("length", 0.0),
            "Cc":     coupler.attrs.get("Cc",     0.0),
            "L":      t.attrs.get("L",      0.0),
            "C":      t.attrs.get("C",      0.0),
            "dir":    direction,
        },
        node_id = coupler.node_id,
        label = f"RCT({_ordered_label(first, coupler, second)})",
    )


def _build_lazy_tct(coupler: CQEDNode, na: CQEDNode, nb: CQEDNode) -> CQEDNode:
    first, second = _ordered_lazy_endpoints(na, nb)
    return CQEDNode(
        subg_type = SubgType.TCT,
        attrs = {
            "L":   first.attrs.get("L",  0.0),
            "C":   first.attrs.get("C",  0.0),
            "Cc":  coupler.attrs.get("Cc", 0.0),
            "L2":  second.attrs.get("L",  0.0),
            "C2":  second.attrs.get("C",  0.0),
            "dir": 1.0,
        },
        node_id = coupler.node_id,
        label = f"TCT({_ordered_label(first, coupler, second)})",
    )


# Each entry is a 3-tuple:
#   (frozenset of the two outer SubgTypes,
#    central coupler SubgType,
#    node_builder(coupler, na, nb) -> CQEDNode)
LAZY_PATTERNS: list[tuple] = [
    (frozenset({SubgType.TRANSMON, SubgType.RESONATOR}), SubgType.C_COUPLER, _build_lazy_rct),
    (frozenset({SubgType.TRANSMON}),                    SubgType.C_COUPLER, _build_lazy_tct),
]


# ===========================================================================
# Branch decomposition
# ===========================================================================

def branch_decomposition(
    branch:   list[int],
    node_map: dict[int, CQEDNode],
    adj_raw:  dict[int, list[int]],
) -> list[CQEDNode]:
    """
    Walk a linear branch and apply the longest matching pattern from the
    subgraph basis.  Returns a list of new (compressed) CQEDNodes.
    """
    def _copy(n: CQEDNode) -> CQEDNode:
        return CQEDNode(n.subg_type, dict(n.attrs), n.node_id, n.label)

    result: list[CQEDNode] = []
    i = 0
    while i < len(branch):
        node = node_map[branch[i]]

        # Hub node: pass through unchanged
        if len(adj_raw[node.node_id]) > 2:
            result.append(_copy(node))
            i += 1
            continue

        # Try 3-node patterns first (longest match wins)
        if i + 2 < len(branch):
            n1, n2, n3 = (node_map[branch[i]],
                          node_map[branch[i + 1]],
                          node_map[branch[i + 2]])
            merged = try_merge_3(n1, n2, n3, adj_raw)
            if merged is not None:
                result.append(merged)
                i += 3
                continue

        # Try 2-node patterns
        if i + 1 < len(branch):
            n1, n2 = node_map[branch[i]], node_map[branch[i + 1]]
            merged = try_merge_2(n1, n2, adj_raw)
            if merged is not None:
                result.append(merged)
                i += 2
                continue

        # No pattern matched: keep node as-is
        result.append(_copy(node))
        i += 1

    return result


# ===========================================================================
# Lazy merge (post-branch pass on the compressed graph)
# ===========================================================================

_COUPLER_TYPES = {SubgType.C_COUPLER, SubgType.I_COUPLER}


def _lazy_merge(topology: CQEDTopology) -> CQEDTopology:
    """
    Iterative pass over the compressed graph: merge any remaining degree-2
    coupler that sits between two nodes matching a pattern in LAZY_PATTERNS.

    Iterates until no more merges are possible.
    """

    def _adj_temp():
        a: dict[int, list[int]] = {n.node_id: [] for n in topology._nodes}
        for u, v in topology._edges:
            a[u].append(v)
            a[v].append(u)
        return a

    changed = True
    while changed:
        changed  = False
        adj      = _adj_temp()
        node_map: dict[int, CQEDNode] = {n.node_id: n for n in topology._nodes}

        for node in list(topology._nodes):
            if node.subg_type not in _COUPLER_TYPES:
                continue

            nbrs = adj.get(node.node_id, [])
            if len(nbrs) != 2:
                continue

            na = node_map.get(nbrs[0])
            nb = node_map.get(nbrs[1])
            if na is None or nb is None:
                continue

            pair_types = frozenset({na.subg_type, nb.subg_type})

            for outer_types, coupler_type, node_builder in LAZY_PATTERNS:
                if node.subg_type != coupler_type or pair_types != outer_types:
                    continue

                merged = node_builder(node, na, nb)

                ids_to_remove = {node.node_id, na.node_id, nb.node_id}
                topology._nodes = [
                    n for n in topology._nodes if n.node_id not in ids_to_remove
                ]
                topology._nodes.append(merged)

                remapped: list[tuple[int, int]] = []
                for u, v in topology._edges:
                    if u in ids_to_remove and v in ids_to_remove:
                        continue
                    u2 = node.node_id if u in ids_to_remove else u
                    v2 = node.node_id if v in ids_to_remove else v
                    if u2 != v2:
                        remapped.append((u2, v2))

                seen: set[frozenset] = set()
                topology._edges = []
                for e in remapped:
                    k = frozenset(e)
                    if k not in seen:
                        seen.add(k)
                        topology._edges.append(e)

                changed = True
                break

            if changed:
                break

    return topology


# ===========================================================================
# Main entry point
# ===========================================================================

def graphlize(raw_topology: CQEDTopology) -> CQEDTopology:
    """
    Compress a raw CQEDTopology (one node per circuit element) into a new
    CQEDTopology where each node represents a subgraph pattern.
    """
    if not raw_topology._nodes:
        return CQEDTopology(name=f"{raw_topology.name}__outer")

    adj      = raw_topology._adj()
    node_map = {n.node_id: n for n in raw_topology._nodes}

    # ── Step 1 & 2: hub-based spanning traversal ──────────────────────────
    root_id         = find_root(raw_topology)
    consumed:        set[int]                          = set()
    all_branches:    list[tuple[int, list[list[int]]]] = []
    hub_queue:       list[int]                         = [root_id]
    processed_hubs:  set[int]                          = set()

    while hub_queue:
        hub_id = hub_queue.pop(0)
        if hub_id in processed_hubs:
            continue
        processed_hubs.add(hub_id)
        consumed.add(hub_id)

        hub_branches = []
        for nb_id in sorted(nb for nb in adj[hub_id] if nb not in consumed):
            branch = walk_branch(nb_id, hub_id, adj, consumed)
            if not branch:
                continue

            last     = branch[-1]
            last_deg = len(adj[last])
            for b in branch:
                if b != last or last_deg <= 2:
                    consumed.add(b)
            # If the branch stops because it reached another hub, that terminal
            # hub must be processed exactly once from hub_queue, not materialized
            # as a pass-through node inside the current branch.  Otherwise graphs
            # such as F--C_hub--cycle create two macro nodes with the same hub
            # label: one terminal copy from the incoming branch and one real hub
            # from the hub pass.  Keep the terminal hub unconsumed and remove it
            # from the branch decomposition; edge reconstruction will still wire
            # the previous compressed node to the unique hub via old_to_new.
            if last_deg > 2 and last not in processed_hubs:
                hub_queue.append(last)
                branch = branch[:-1]

            if branch:
                hub_branches.append(branch)
        all_branches.append((hub_id, hub_branches))

    # ── Step 3: decompose branches ────────────────────────────────────────
    new_topology = CQEDTopology(name=f"{raw_topology.name}__outer")
    old_to_new:  dict[int, CQEDNode] = {}
    id_counter   = [0]

    def add_new_node(node: CQEDNode) -> CQEDNode:
        node.node_id = id_counter[0]
        id_counter[0] += 1
        new_topology._nodes.append(node)
        new_topology._next_id = id_counter[0]
        return node

    for hub_id, branches in all_branches:
        hub_node = node_map[hub_id]
        hub_new  = add_new_node(
            CQEDNode(hub_node.subg_type, dict(hub_node.attrs), 0, hub_node.label)
        )
        old_to_new[hub_id] = hub_new

        for branch in branches:
            new_nodes = branch_decomposition(branch, node_map, adj)
            for n in new_nodes:
                new_node = add_new_node(n)
                for old_id in branch:
                    old_n = node_map[old_id]
                    if old_n and n.label and old_n.label in n.label:
                        old_to_new[old_id] = new_node
                    elif old_to_new.get(old_id) is None:
                        old_to_new[old_id] = new_node

    # ── Step 4: reconstruct edges ─────────────────────────────────────────
    seen_edges: set[frozenset] = set()
    for old_u, old_v in raw_topology._edges:
        nu = old_to_new.get(old_u)
        nv = old_to_new.get(old_v)
        if nu is None or nv is None or nu.node_id == nv.node_id:
            continue
        key = frozenset({nu.node_id, nv.node_id})
        if key not in seen_edges:
            seen_edges.add(key)
            new_topology._edges.append((nu.node_id, nv.node_id))

    # ── Step 5: lazy merge ────────────────────────────────────────────────
    new_topology = _lazy_merge(new_topology)

    return new_topology
