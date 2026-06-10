"""
symmetries.py
=============

Automatic computation of physical symmetries (automorphisms) of cQED circuits.

DESIGN PHILOSOPHY
-----------------
Symmetries are computed on the **primitive expanded graph** — the graph where
every node is a single circuit element (FEEDLINE, TRANSMON, RESONATOR,
C_COUPLER, I_COUPLER).  This is the level at which physics is unambiguous.

The expansion delegates to `expansion.expand_topology()`, which already
implements the deterministic port-assignment backtracking used by the
round-trip tests.  All symmetry machinery here is therefore independent of
the specific macro-compression rules; if a new SubgType is added to
definitions.py and expansion.py, nothing in this file needs to change.

PUBLIC API
----------
expand_to_primitive(topology)  ->  PrimitiveGraph
    Expand a CQEDTopology (raw or compressed) into the primitive node graph.

compute_automorphisms(pg)       ->  list[dict[int, int]]
    Return all graph automorphisms that preserve node subg_type.
    Each automorphism is a dict {prim_node_id -> prim_node_id}.

compute_orbits(automorphisms, node_ids)  ->  list[frozenset[int]]
    Group primitive node ids into orbits under the automorphism group.

param_permutations(pg, automorphisms)  ->  list[dict[tuple[int, str], float]]
    For each automorphism return the permuted parameter assignment suitable
    for a min-over-permutations loss.

symmetry_report(topology)  ->  SymmetryReport
    One-stop convenience: expand -> automorphisms -> orbits -> report.

build_automorphism_cache(topologies)  ->  None
    Pre-compute and cache automorphisms for a list of CQEDTopologies.

TERMINOLOGY
-----------
PrimitiveNode  : one atomic circuit element (subg_type, attrs, label).
PrimitiveGraph : undirected graph of PrimitiveNodes.
Automorphism   : bijection PrimitiveNode -> PrimitiveNode that preserves
                 both edge structure AND subg_type of every node.
Orbit          : equivalence class of nodes under the automorphism group.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher

from circuit2graph.definitions import SUBG_DEFS, SubgType
from circuit2graph.topology import CQEDTopology, CQEDNode
from circuit2graph.expansion import expand_topology, PRIMITIVE_TYPES as _PRIM


# ===========================================================================
# Primitive node / graph structures
# ===========================================================================

@dataclass
class PrimitiveNode:
    """
    One physical primitive node used for symmetry computation.

    A TRANSMON is represented as *one* PrimitiveNode carrying both L and C,
    so both parameters travel together under every automorphism.
    """
    prim_id:   int
    subg_type: SubgType
    attrs:     dict[str, float] = field(default_factory=dict)
    label:     str = ""


@dataclass
class PrimitiveGraph:
    """
    Expanded primitive graph of a cQED circuit.

    nodes : list of PrimitiveNode  (ids are 0..N-1, stable)
    edges : list of (prim_id_a, prim_id_b) undirected pairs
    name  : inherited from the source CQEDTopology
    """
    nodes: list[PrimitiveNode]   = field(default_factory=list)
    edges: list[tuple[int, int]] = field(default_factory=list)
    name:  str                   = ""

    def node_ids(self) -> list[int]:
        return [n.prim_id for n in self.nodes]

    def adj(self) -> dict[int, list[int]]:
        a: dict[int, list[int]] = {n.prim_id: [] for n in self.nodes}
        for u, v in self.edges:
            a[u].append(v)
            a[v].append(u)
        return a

    def to_networkx(self) -> nx.Graph:
        G = nx.Graph(name=self.name)
        for n in self.nodes:
            G.add_node(
                n.prim_id,
                subg_type=int(n.subg_type),
                label=n.label,
                attrs=n.attrs,
                pnode=n,
            )
        G.add_edges_from(self.edges)
        return G


# ===========================================================================
# Expansion: CQEDTopology -> PrimitiveGraph
# ===========================================================================

def expand_to_primitive(topology: CQEDTopology) -> PrimitiveGraph:
    """
    Expand *topology* to the physical primitive level used for symmetries.

    Works for both raw topologies (all nodes already primitive) and
    compressed macro topologies.  In both cases the output is a PrimitiveGraph
    where every node is one of FEEDLINE / TRANSMON / RESONATOR / C_COUPLER /
    I_COUPLER — the same primitive basis used by the datasets.

    Internally delegates to `expansion.expand_topology()`, which performs
    deterministic backtracking port-assignment so that this call always
    reproduces the same primitive graph regardless of node insertion order.
    """
    # If already fully primitive, skip the expand step (avoids an unnecessary
    # validate=True backtracking call which can be slow on large graphs).
    all_primitive = all(n.subg_type in _PRIM for n in topology._nodes)
    if all_primitive:
        physical = topology
    else:
        physical = expand_topology(topology, validate=True)

    pg = PrimitiveGraph(name=physical.name)

    # Sequential prim_ids 0..N-1 mapped from node_ids in the expanded topology.
    node_id_to_prim: dict[int, int] = {}
    for i, n in enumerate(physical._nodes):
        node_id_to_prim[n.node_id] = i
        pg.nodes.append(PrimitiveNode(
            prim_id   = i,
            subg_type = n.subg_type,
            attrs     = dict(n.attrs),
            label     = n.label or n.subg_type.name,
        ))

    seen: set[frozenset[int]] = set()
    for u, v in physical._edges:
        pu, pv = node_id_to_prim[u], node_id_to_prim[v]
        key = frozenset({pu, pv})
        if pu != pv and key not in seen:
            seen.add(key)
            pg.edges.append((pu, pv))

    return pg


# ===========================================================================
# Automorphism computation
# ===========================================================================

def compute_automorphisms(pg: PrimitiveGraph) -> list[dict[int, int]]:
    """
    Compute all automorphisms of the primitive graph that preserve subg_type.

    Numeric parameter values are deliberately ignored: they are moved later by
    `param_permutations()` and compared by the loss.  The identity is always
    first in the returned list.

    Returns
    -------
    list of dicts  {prim_id -> prim_id}
    """
    identity = {n.prim_id: n.prim_id for n in pg.nodes}
    G = pg.to_networkx()

    def node_match(d1: dict, d2: dict) -> bool:
        return d1["subg_type"] == d2["subg_type"]

    gm = GraphMatcher(G, G, node_match=node_match)

    seen = {tuple(sorted(identity.items()))}
    auts: list[dict[int, int]] = [identity]
    for mapping in gm.isomorphisms_iter():
        key = tuple(sorted(mapping.items()))
        if key in seen:
            continue
        seen.add(key)
        auts.append(dict(mapping))

    return auts


# ===========================================================================
# Orbits
# ===========================================================================

def compute_orbits(
    automorphisms: list[dict[int, int]],
    node_ids:      list[int],
) -> list[frozenset[int]]:
    """
    Partition *node_ids* into orbits under the automorphism group.

    Two nodes are in the same orbit iff there exists an automorphism that
    maps one to the other.

    Parameters
    ----------
    automorphisms : output of compute_automorphisms()
    node_ids      : list of prim_ids to partition

    Returns
    -------
    list of frozensets, each frozenset is one orbit
    """
    parent = {nid: nid for nid in node_ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for aut in automorphisms:
        for src, dst in aut.items():
            if src in parent and dst in parent:
                union(src, dst)

    orbits: dict[int, set[int]] = {}
    for nid in node_ids:
        root = find(nid)
        orbits.setdefault(root, set()).add(nid)

    return [frozenset(s) for s in orbits.values()]


# ===========================================================================
# Parameter permutation helpers  (for the min-over-permutations loss)
# ===========================================================================

def param_permutations(
    pg:             PrimitiveGraph,
    automorphisms:  list[dict[int, int]],
    *,
    warn_incomplete: bool = False,
) -> list[dict[tuple[int, str], float]]:
    """
    For each automorphism π, return the permuted parameter assignment.

    Convention:
        π(θ)[src_id, attr] = θ[π(src_id), attr] = θ[dst_id, attr]

    Because the symmetry graph works at primitive-block level, a TRANSMON node
    carries both L and C together.  When an automorphism maps T_src -> T_dst,
    both parameters are moved together:
        π(θ)[T_src, L] = θ[T_dst, L]
        π(θ)[T_src, C] = θ[T_dst, C]

    Parameters
    ----------
    pg              : PrimitiveGraph with attrs populated.
    automorphisms   : output of compute_automorphisms().
    warn_incomplete : if True, emit a warning for any missing attr entry.

    Returns
    -------
    list of dicts  {(prim_id, attr_name) -> float value}
    """
    import warnings

    node_by_id: dict[int, PrimitiveNode] = {n.prim_id: n for n in pg.nodes}
    result: list[dict[tuple[int, str], float]] = []

    for aut_idx, aut in enumerate(automorphisms):
        permuted: dict[tuple[int, str], float] = {}
        missing_entries: list[str] = []

        for src, dst in aut.items():
            src_node = node_by_id.get(src)
            dst_node = node_by_id.get(dst)
            if src_node is None or dst_node is None:
                continue
            if src_node.subg_type != dst_node.subg_type:
                # Should never happen for a valid type-preserving automorphism.
                if warn_incomplete:
                    missing_entries.append(
                        f"type mismatch: prim {src} ({src_node.subg_type.name}) "
                        f"-> prim {dst} ({dst_node.subg_type.name})"
                    )
                continue

            for attr_name in SUBG_DEFS[src_node.subg_type].attrs:
                # The 'dir' attribute is a compression artefact, not a
                # physical parameter — skip it so it never appears in the
                # parameter loss.
                if attr_name == "dir":
                    continue
                if attr_name in dst_node.attrs:
                    if attr_name in src_node.attrs:
                        permuted[(src, attr_name)] = dst_node.attrs[attr_name]
                    else:
                        if warn_incomplete:
                            missing_entries.append(
                                f"prim {src} ({src_node.subg_type.name}) "
                                f"missing attr {attr_name!r} in source node"
                            )
                else:
                    if warn_incomplete:
                        missing_entries.append(
                            f"prim {dst} ({dst_node.subg_type.name}) "
                            f"missing attr {attr_name!r} in destination node"
                        )

        if warn_incomplete and missing_entries:
            warnings.warn(
                f"param_permutations: automorphism #{aut_idx} produced an incomplete "
                f"permutation ({len(missing_entries)} missing entries); "
                f"details:\n  " + "\n  ".join(missing_entries),
                stacklevel=2,
            )

        result.append(permuted)

    return result


# ===========================================================================
# Automorphism cache  (pre-compute once per topology, not per sample)
# ===========================================================================

@dataclass
class _CachedAuts:
    """Pre-computed topology-level symmetry data for one topology name.

    Only the structural automorphism maps are cached, NOT the param_permutations
    values: those are sample-specific (they carry the actual float parameter
    values) and must be rebuilt from the current graph on every call.
    """
    primitive_graph: PrimitiveGraph
    automorphisms:   list[dict[int, int]]


_AUT_CACHE: dict[str, _CachedAuts] = {}


def build_automorphism_cache(topologies: list[CQEDTopology]) -> None:
    """Pre-compute and cache automorphisms for a list of topologies.

    Call once before training/inference.  The topology *name* is the cache key,
    so every dataset that shares the same macro-graph structure reuses the same
    entry.

    Parameters
    ----------
    topologies : one representative CQEDTopology per dataset / circuit class.
        Typically obtained via ``graphlize(dataset.build_topology())`` for each
        dataset.
    """
    global _AUT_CACHE
    for topo in topologies:
        key = topo.name
        if key in _AUT_CACHE:
            continue
        pg   = expand_to_primitive(topo)
        auts = compute_automorphisms(pg)
        _AUT_CACHE[key] = _CachedAuts(primitive_graph=pg, automorphisms=auts)


def get_cached_automorphisms(topology_name: str) -> list[dict[int, int]] | None:
    """Return cached automorphism maps for *topology_name*, or None if absent."""
    entry = _AUT_CACHE.get(topology_name)
    return entry.automorphisms if entry is not None else None


def get_cache_summary() -> str:
    """Return a human-readable summary of the automorphism cache."""
    if not _AUT_CACHE:
        return "Automorphism cache is empty."
    lines = [f"Automorphism cache ({len(_AUT_CACHE)} entries):"]
    for name, entry in _AUT_CACHE.items():
        lines.append(
            f"  {name!r:50s}  |Aut|={len(entry.automorphisms):3d}"
            f"  primitives={len(entry.primitive_graph.nodes)}"
        )
    return "\n".join(lines)


# ===========================================================================
# Convenience: full symmetry report
# ===========================================================================

@dataclass
class SymmetryReport:
    """
    Complete symmetry information for one CQEDTopology.

    Fields
    ------
    topology_name   : name of the source topology
    primitive_graph : expanded primitive graph
    automorphisms   : all automorphisms (identity first)
    orbits          : orbit partition of primitive node ids
    n_automorphisms : |Aut(G, type)| — size of the symmetry group
    """
    topology_name:   str
    primitive_graph: PrimitiveGraph
    automorphisms:   list[dict[int, int]]
    orbits:          list[frozenset[int]]
    n_automorphisms: int


def symmetry_report(topology: CQEDTopology) -> SymmetryReport:
    """
    One-stop function: expand -> automorphisms -> orbits -> report.

    Usage::

        rep = symmetry_report(my_topology)
        print(rep.n_automorphisms)   # e.g. 2 for T-C-T
        print(rep.orbits)            # [{0, 2}, {1}, ...]  (by prim_id)
    """
    pg   = expand_to_primitive(topology)
    auts = compute_automorphisms(pg)
    orbs = compute_orbits(auts, pg.node_ids())
    return SymmetryReport(
        topology_name   = topology.name,
        primitive_graph = pg,
        automorphisms   = auts,
        orbits          = orbs,
        n_automorphisms = len(auts),
    )
