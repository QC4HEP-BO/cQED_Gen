"""
compression.py
==============

Graph-compression utilities for the cQED representation.

Transforms a raw CQEDTopology (one node per primitive element) into a
compressed topology where nodes are directional macronode blocks.

DIRECTIONAL CONVENTION
----------------------
The BFS tree rooted at ``find_root()`` defines a canonical orientation for
every branch.  For asymmetric 2- and 3-node blocks the *first* element
encountered walking away from the root goes into the *left* slot (attrs[0]),
and the *last* element goes into the *right* slot.

This makes the mapping raw → compressed → raw invertible without any
heuristic ordering (e.g. the old "sort TCT by L-value" trick is gone).

Symmetric blocks (TCT) still encode left/right by raw node-id order so that
expand_macronodes() can reconstruct them unambiguously.

PUBLIC API
----------
graphlize(raw_topology) -> CQEDTopology
    Main entry point used by the data loader.

EXTENSION GUIDE
---------------
To add a new merge pattern:
  1. Add SubgType + SubgDef in definitions.py
  2. Add an if-block in try_merge_3() or try_merge_2() for linear patterns,
     OR add a tuple to LAZY_PATTERNS for post-branch patterns.
"""

from circuit2graph.definitions import SubgType
from circuit2graph.topology import CQEDNode, CQEDTopology


# ===========================================================================
# Low-level graph traversal utilities
# ===========================================================================

def compute_eccentricity(node_id: int, adj: dict) -> int:
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
    The order is root → leaf, which determines left/right in asymmetric blocks.
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
    Ties: highest degree → lowest eccentricity → lowest node_id.
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

def _make_node(
    subg_type: SubgType,
    attrs:     dict,
    label:     str,
    tmp_id:    int = 0,
) -> CQEDNode:
    return CQEDNode(subg_type=subg_type, attrs=attrs, node_id=tmp_id, label=label)


def try_merge_3(
    n1:      CQEDNode,
    n2:      CQEDNode,
    n3:      CQEDNode,
    adj_raw: dict,
) -> CQEDNode | None:
    """
    Try to merge a triplet (n1 → n2-coupler → n3) into a 3-node macronode.
    n1 is the root-facing node, n3 is the leaf-facing node.
    Returns a new CQEDNode or None if no pattern matched.
    """
    # Only merge through degree-2 capacitive couplers
    if n2.subg_type != SubgType.C_COUPLER or len(adj_raw[n2.node_id]) != 2:
        return None

    t1, t3 = n1.subg_type, n3.subg_type

    # ── TCT  (T – C – T)  symmetric ──────────────────────────────────────
    if t1 == SubgType.TRANSMON and t3 == SubgType.TRANSMON:
        # No ordering by L — left = n1 (root-side), right = n3 (leaf-side)
        return _make_node(
            SubgType.TCT,
            attrs = {
                "L":  n1.attrs.get("L",  0.0),
                "C":  n1.attrs.get("C",  0.0),
                "Cc": n2.attrs.get("Cc", 0.0),
                "L2": n3.attrs.get("L",  0.0),
                "C2": n3.attrs.get("C",  0.0),
            },
            label = f"TCT({n1.label}+{n2.label}+{n3.label})",
        )

    # ── RCT  (R – C – T) ─────────────────────────────────────────────────
    if t1 == SubgType.RESONATOR and t3 == SubgType.TRANSMON:
        return _make_node(
            SubgType.RCT,
            attrs = {
                "length": n1.attrs.get("length", 0.0),
                "Cc":     n2.attrs.get("Cc",     0.0),
                "L":      n3.attrs.get("L",      0.0),
                "C":      n3.attrs.get("C",      0.0),
            },
            label = f"RCT({n1.label}+{n2.label}+{n3.label})",
        )

    # ── TCR  (T – C – R) ─────────────────────────────────────────────────
    if t1 == SubgType.TRANSMON and t3 == SubgType.RESONATOR:
        return _make_node(
            SubgType.TCR,
            attrs = {
                "L":      n1.attrs.get("L",      0.0),
                "C":      n1.attrs.get("C",      0.0),
                "Cc":     n2.attrs.get("Cc",     0.0),
                "length": n3.attrs.get("length", 0.0),
            },
            label = f"TCR({n1.label}+{n2.label}+{n3.label})",
        )

    # ── ADD NEW 3-NODE PATTERNS BELOW ─────────────────────────────────────
    return None


def try_merge_2(
    n1:      CQEDNode,   # root-facing node
    n2:      CQEDNode,   # leaf-facing node (coupler or element)
    adj_raw: dict,
) -> CQEDNode | None:
    """
    Try to merge a pair into a known 2-node directional macronode.
    n1 is the root-facing node, n2 is the leaf-facing node.
    Returns a new CQEDNode or None.
    """
    if len(adj_raw[n2.node_id]) != 2:
        return None

    # ── Capacitive coupler patterns ───────────────────────────────────────
    if n2.subg_type == SubgType.C_COUPLER:

        if n1.subg_type == SubgType.TRANSMON:        # T → C  ⟹  TC
            return _make_node(
                SubgType.TC,
                attrs = {
                    "L":  n1.attrs.get("L",  0.0),
                    "C":  n1.attrs.get("C",  0.0),
                    "Cc": n2.attrs.get("Cc", 0.0),
                },
                label = f"TC({n1.label}+{n2.label})",
            )

        if n1.subg_type == SubgType.RESONATOR:       # R → C  ⟹  RC
            return _make_node(
                SubgType.RC,
                attrs = {
                    "length": n1.attrs.get("length", 0.0),
                    "Cc":     n2.attrs.get("Cc",     0.0),
                },
                label = f"RC({n1.label}+{n2.label})",
            )

    if n1.subg_type == SubgType.C_COUPLER:

        if n2.subg_type == SubgType.TRANSMON:        # C → T  ⟹  CT
            return _make_node(
                SubgType.CT,
                attrs = {
                    "Cc": n1.attrs.get("Cc", 0.0),
                    "L":  n2.attrs.get("L",  0.0),
                    "C":  n2.attrs.get("C",  0.0),
                },
                label = f"CT({n1.label}+{n2.label})",
            )

        if n2.subg_type == SubgType.RESONATOR:       # C → R  ⟹  CR
            return _make_node(
                SubgType.CR,
                attrs = {
                    "Cc":     n1.attrs.get("Cc",     0.0),
                    "length": n2.attrs.get("length", 0.0),
                },
                label = f"CR({n1.label}+{n2.label})",
            )

    # ── Inductive coupler patterns ────────────────────────────────────────
    if n2.subg_type == SubgType.I_COUPLER:

        if n1.subg_type == SubgType.RESONATOR:       # R → Ind  ⟹  RI
            return _make_node(
                SubgType.RI,
                attrs = {
                    "length": n1.attrs.get("length", 0.0),
                    "D":      n2.attrs.get("D",      0.0),
                    "l":      n2.attrs.get("l",      0.0),
                },
                label = f"RI({n1.label}+{n2.label})",
            )

    if n1.subg_type == SubgType.I_COUPLER:

        if n2.subg_type == SubgType.RESONATOR:       # Ind → R  ⟹  IR
            return _make_node(
                SubgType.IR,
                attrs = {
                    "D":      n1.attrs.get("D",      0.0),
                    "l":      n1.attrs.get("l",      0.0),
                    "length": n2.attrs.get("length", 0.0),
                },
                label = f"IR({n1.label}+{n2.label})",
            )

    # ── ADD NEW 2-NODE PATTERNS BELOW ─────────────────────────────────────
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
# IMPORTANT: for directional blocks, na is the root-facing neighbour and nb
# is the leaf-facing neighbour.  The lazy merge pass determines this from
# the compressed graph's BFS-tree structure (see _lazy_merge below).
#
# ADD NEW LAZY PATTERNS HERE — one tuple per pattern.

LAZY_PATTERNS: list[tuple] = [

    # R–C–T  →  RCT  (root=R, leaf=T)
    (
        frozenset({SubgType.RESONATOR, SubgType.TRANSMON}),
        SubgType.RCT,
        lambda coupler, na, nb: {
            "length": (na if na.subg_type == SubgType.RESONATOR else nb).attrs.get("length", 0.0),
            "Cc":     coupler.attrs.get("Cc", 0.0),
            "L":      (na if na.subg_type == SubgType.TRANSMON  else nb).attrs.get("L",      0.0),
            "C":      (na if na.subg_type == SubgType.TRANSMON  else nb).attrs.get("C",      0.0),
        },
        lambda coupler, na, nb: (
            f"RCT({na.label}+{coupler.label}+{nb.label})"
            if na.subg_type == SubgType.RESONATOR
            else f"RCT({nb.label}+{coupler.label}+{na.label})"
        ),
    ),

    # T–C–T  →  TCT  (symmetric; left = lower node_id, as determined at call site)
    (
        frozenset({SubgType.TRANSMON}),
        SubgType.TCT,
        lambda coupler, na, nb: {
            "L":  na.attrs.get("L",  0.0),
            "C":  na.attrs.get("C",  0.0),
            "Cc": coupler.attrs.get("Cc", 0.0),
            "L2": nb.attrs.get("L",  0.0),
            "C2": nb.attrs.get("C",  0.0),
        },
        lambda coupler, na, nb: f"TCT({na.label}+{coupler.label}+{nb.label})",
    ),

    # T–C–R  →  TCR  (root=T, leaf=R)
    (
        frozenset({SubgType.TRANSMON, SubgType.RESONATOR}),
        SubgType.TCR,
        lambda coupler, na, nb: {
            "L":      (na if na.subg_type == SubgType.TRANSMON  else nb).attrs.get("L",      0.0),
            "C":      (na if na.subg_type == SubgType.TRANSMON  else nb).attrs.get("C",      0.0),
            "Cc":     coupler.attrs.get("Cc", 0.0),
            "length": (na if na.subg_type == SubgType.RESONATOR else nb).attrs.get("length", 0.0),
        },
        lambda coupler, na, nb: (
            f"TCR({na.label}+{coupler.label}+{nb.label})"
            if na.subg_type == SubgType.TRANSMON
            else f"TCR({nb.label}+{coupler.label}+{na.label})"
        ),
    ),

    # ── ADD NEW LAZY PATTERNS BELOW ───────────────────────────────────────
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
    Walk a linear branch (root → leaf order) and apply the longest matching
    pattern.  Directional blocks are assigned correctly because n1 is always
    the element closer to the root.
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
    coupler between two nodes that match a LAZY_PATTERNS entry.

    For directional 3-node blocks (RCT, TCR) the orientation is inferred
    from the BFS tree rooted at find_root().  na = root-facing neighbour,
    nb = leaf-facing neighbour.
    """
    def _adj_temp():
        a: dict[int, list[int]] = {n.node_id: [] for n in topology._nodes}
        for u, v in topology._edges:
            a[u].append(v)
            a[v].append(u)
        return a

    def _bfs_depth(root_id: int, adj: dict) -> dict[int, int]:
        depth = {root_id: 0}
        queue = [root_id]
        while queue:
            curr = queue.pop(0)
            for nb in adj[curr]:
                if nb not in depth:
                    depth[nb] = depth[curr] + 1
                    queue.append(nb)
        return depth

    changed = True
    while changed:
        changed  = False
        adj      = _adj_temp()

        # Build BFS depth from the current root to orient directional merges
        if topology._nodes:
            root_id = find_root(topology)
            depth   = _bfs_depth(root_id, adj)
        else:
            depth = {}

        node_map: dict[int, CQEDNode] = {n.node_id: n for n in topology._nodes}

        for node in list(topology._nodes):
            if node.subg_type not in _COUPLER_TYPES:
                continue

            nbrs = adj.get(node.node_id, [])
            if len(nbrs) != 2:
                continue

            na_raw = node_map.get(nbrs[0])
            nb_raw = node_map.get(nbrs[1])
            if na_raw is None or nb_raw is None:
                continue

            # Orient: na = root-facing (smaller BFS depth), nb = leaf-facing
            d0 = depth.get(nbrs[0], 0)
            d1 = depth.get(nbrs[1], 0)
            if d0 <= d1:
                na, nb = na_raw, nb_raw
            else:
                na, nb = nb_raw, na_raw

            pair_types = frozenset({na.subg_type, nb.subg_type})

            for outer_types, merged_type, attr_fn, label_fn in LAZY_PATTERNS:
                if pair_types != outer_types:
                    continue

                merged = CQEDNode(
                    subg_type = merged_type,
                    attrs     = attr_fn(node, na, nb),
                    node_id   = node.node_id,
                    label     = label_fn(node, na, nb),
                )

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
    CQEDTopology where each node is a directional macronode block.

    The BFS root determines orientation: left = root-side, right = leaf-side.
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
            if last_deg > 2 and last not in processed_hubs:
                hub_queue.append(last)

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
