"""
test_symmetries.py
==================

Tests for circuit2graph.symmetries.

Running this file with pytest produces:
  - Rich terminal output for every test case: topology summary, primitive
    nodes, automorphisms, per-permutation parameter assignments.
  - PNG plots (one per named topology) saved to
      test/symmetry_plots/<topology_name>__symmetries.png
    Each plot shows the primitive graph with node labels and a legend that
    lists every equivalence class (orbit) together with the non-trivial
    permutations.

Set the output directory via the env var CQED_SYMMETRY_PLOT_DIR.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Callable

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]
_SRC_DIR   = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import pytest

from circuit2graph.definitions import SubgType, NODE_COLORS
from circuit2graph.topology import CQEDTopology
from circuit2graph.compression import graphlize
from circuit2graph.expansion import expand_topology
from circuit2graph.symmetries import (
    PrimitiveGraph,
    PrimitiveNode,
    expand_to_primitive,
    compute_automorphisms,
    compute_orbits,
    param_permutations,
    symmetry_report,
    build_automorphism_cache,
    get_cached_automorphisms,
    get_cache_summary,
)

from test_graphlize import (          # type: ignore
    _t_attrs, _r_attrs, _cc_attrs,
    build_qubit_resonator_resonator,
    build_resonator_qubit_resonator,
    build_three_qubit_capacitive_line,
    build_three_qubit_capacitive_star,
    build_two_qubit_resonator,
    USER_TOPOLOGY_BUILDERS,
)
from test_expansion import (          # type: ignore
    build_feedline_two_branch_hub_plus_cycle,
    build_inductive_resonator_line_and_hub,
    build_four_transmon_capacitive_ring_with_tail,
)

# ---------------------------------------------------------------------------
# Output directory for plots
# ---------------------------------------------------------------------------
PLOT_DIR = Path(os.environ.get(
    "CQED_SYMMETRY_PLOT_DIR",
    _REPO_ROOT / "test" / "symmetry_plots",
))

# ---------------------------------------------------------------------------
# Type shortcuts
# ---------------------------------------------------------------------------
T = SubgType.TRANSMON
C = SubgType.C_COUPLER
R = SubgType.RESONATOR
F = SubgType.FEEDLINE
I = SubgType.I_COUPLER


def _i_attrs(D: float = 9.0, l: float = 1.3e-4) -> dict:
    return {"D": D, "l": l}


# ===========================================================================
# Terminal printing helpers
# ===========================================================================

_SEP  = "─" * 72
_SEP2 = "═" * 72

def _fmt_val(v: float) -> str:
    """Compact scientific notation for physical values."""
    if v == 0.0:
        return "0"
    abs_v = abs(v)
    if 1e-3 <= abs_v < 1e4:
        return f"{v:.4g}"
    return f"{v:.3e}"


def _print_primitive_graph(pg: PrimitiveGraph) -> None:
    print(f"  Primitive graph '{pg.name}'  ({len(pg.nodes)} nodes, {len(pg.edges)} edges)")
    adj = pg.adj()
    for n in pg.nodes:
        attrs_str = "  ".join(f"{k}={_fmt_val(v)}" for k, v in n.attrs.items())
        nbrs = ", ".join(str(nb) for nb in sorted(adj[n.prim_id]))
        print(f"    [{n.prim_id:2d}] {n.subg_type.name:<12s}  label={n.label:<14s}"
              f"  attrs=({attrs_str})  nbrs=[{nbrs}]")


def _print_automorphisms(
    auts:  list[dict[int, int]],
    pg:    PrimitiveGraph,
    perms: list[dict[tuple[int, str], float]],
) -> None:
    node_by_id = {n.prim_id: n for n in pg.nodes}
    n_nontriv  = sum(1 for a in auts if any(k != v for k, v in a.items()))
    print(f"  |Aut| = {len(auts)}  ({n_nontriv} non-trivial)")

    for i, (aut, perm) in enumerate(zip(auts, perms)):
        is_id = all(k == v for k, v in aut.items())
        tag   = "[identity]" if is_id else f"[perm #{i}  ]"
        moved = {k: v for k, v in aut.items() if k != v}
        cycle_str = _cycles_str(aut, pg)
        print(f"    {tag}  cycles: {cycle_str}")

        if not is_id:
            # Print each mapping that actually moves a node, with its params
            for src, dst in sorted(moved.items()):
                sn, dn = node_by_id[src], node_by_id[dst]
                src_params = "  ".join(
                    f"{a}={_fmt_val(perm.get((src, a), float('nan')))}"
                    for a in sn.attrs if a != "dir"
                )
                dst_params = "  ".join(
                    f"{a}={_fmt_val(dn.attrs.get(a, float('nan')))}"
                    for a in dn.attrs if a != "dir"
                )
                print(f"      [{src}]{sn.subg_type.name}({src_params})"
                      f"  ←  [{dst}]{dn.subg_type.name}({dst_params})")


def _cycles_str(aut: dict[int, int], pg: PrimitiveGraph) -> str:
    """Express an automorphism as disjoint cycles, e.g. (0 2)(1 3)."""
    visited: set[int] = set()
    cycles: list[str] = []
    for start in sorted(aut):
        if start in visited or aut[start] == start:
            visited.add(start)
            continue
        cycle: list[int] = []
        cur = start
        while cur not in visited:
            visited.add(cur)
            cycle.append(cur)
            cur = aut[cur]
        cycles.append("(" + " ".join(str(x) for x in cycle) + ")")
    return "".join(cycles) if cycles else "(id)"


def _print_orbits(orbs: list[frozenset[int]], pg: PrimitiveGraph) -> None:
    node_by_id = {n.prim_id: n for n in pg.nodes}
    print(f"  Orbits ({len(orbs)}):")
    for orb in sorted(orbs, key=lambda o: min(o)):
        ids = sorted(orb)
        labels = [f"[{i}]{node_by_id[i].subg_type.name}" for i in ids]
        print(f"    {{ {', '.join(labels)} }}")


def print_symmetry_summary(
    name:  str,
    raw:   CQEDTopology,
    compressed: CQEDTopology | None = None,
    *,
    verbose: bool = True,
) -> None:
    """Full terminal dump for one topology."""
    if compressed is None:
        compressed = graphlize(raw)
    pg   = expand_to_primitive(compressed)
    auts = compute_automorphisms(pg)
    orbs = compute_orbits(auts, pg.node_ids())
    perms = param_permutations(pg, auts)

    print()
    print(_SEP2)
    print(f"  TOPOLOGY: {name}")
    print(_SEP2)
    _print_primitive_graph(pg)
    print(_SEP)
    _print_automorphisms(auts, pg, perms)
    print(_SEP)
    _print_orbits(orbs, pg)
    print(_SEP)


# ===========================================================================
# Plot helpers
# ===========================================================================

# Palette for coloring orbit groups (cycles through if many orbits)
_ORBIT_PALETTE = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#a65628", "#f781bf", "#999999",
    "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3",
]


def _build_orbit_color_map(
    orbs: list[frozenset[int]],
    pg:   PrimitiveGraph,
) -> dict[int, str]:
    """Map prim_id -> orbit color.  Singleton orbits get a neutral grey."""
    node_by_id = {n.prim_id: n for n in pg.nodes}
    color_map: dict[int, str] = {}
    palette_idx = 0
    for orb in sorted(orbs, key=lambda o: (-len(o), min(o))):
        if len(orb) == 1:
            nid = next(iter(orb))
            color_map[nid] = NODE_COLORS.get(node_by_id[nid].subg_type, "#cccccc")
        else:
            col = _ORBIT_PALETTE[palette_idx % len(_ORBIT_PALETTE)]
            palette_idx += 1
            for nid in orb:
                color_map[nid] = col
    return color_map


def _orbit_legend_text(
    orbs:  list[frozenset[int]],
    auts:  list[dict[int, int]],
    perms: list[dict[tuple[int, str], float]],
    pg:    PrimitiveGraph,
) -> list[str]:
    """
    Build the legend lines:
      - one line per non-singleton orbit: which nodes are equivalent
      - one block per non-trivial automorphism: cycle notation + moved params
    """
    node_by_id = {n.prim_id: n for n in pg.nodes}
    lines: list[str] = []

    # --- Orbits ---
    lines.append("── Equivalence classes (orbits) ──")
    for orb in sorted(orbs, key=lambda o: (-len(o), min(o))):
        ids    = sorted(orb)
        labels = [f"[{i}] {node_by_id[i].subg_type.name}({node_by_id[i].label})" for i in ids]
        marker = "≡" if len(orb) > 1 else "·"
        lines.append(f"  {marker}  {' = '.join(labels)}")

    # --- Non-trivial permutations ---
    nontriv = [(i, a, p) for i, (a, p) in enumerate(zip(auts, perms))
               if any(k != v for k, v in a.items())]
    if nontriv:
        lines.append("")
        lines.append(f"── {len(nontriv)} non-trivial permutation(s) ──")
        for i, aut, perm in nontriv:
            lines.append(f"  perm #{i}: {_cycles_str(aut, pg)}")
            moved = {k: v for k, v in aut.items() if k != v}
            for src, dst in sorted(moved.items()):
                sn, dn = node_by_id[src], node_by_id[dst]
                param_parts = []
                for attr in sn.attrs:
                    if attr == "dir":
                        continue
                    src_val = sn.attrs.get(attr, float("nan"))
                    dst_val = dn.attrs.get(attr, float("nan"))
                    param_parts.append(
                        f"{attr}: {_fmt_val(src_val)}←{_fmt_val(dst_val)}"
                    )
                params_str = ",  ".join(param_parts) if param_parts else "—"
                lines.append(
                    f"    [{src}]{sn.subg_type.name} ← [{dst}]{dn.subg_type.name}"
                    f"   ({params_str})"
                )
    else:
        lines.append("")
        lines.append("── No non-trivial permutations (asymmetric circuit) ──")

    return lines


def plot_symmetries(
    name:        str,
    raw:         CQEDTopology,
    compressed:  CQEDTopology | None = None,
    *,
    outdir:      Path = PLOT_DIR,
) -> Path:
    """
    Draw the primitive graph of *raw* (or of *compressed*) and annotate it
    with orbit colours and a legend listing the equivalence classes and
    all non-trivial permutations.
    """
    if compressed is None:
        compressed = graphlize(raw)
    pg    = expand_to_primitive(compressed)
    auts  = compute_automorphisms(pg)
    orbs  = compute_orbits(auts, pg.node_ids())
    perms = param_permutations(pg, auts)

    # Build networkx graph
    G     = pg.to_networkx()
    pos   = nx.spring_layout(G, seed=42, k=1.8)
    c_map = _build_orbit_color_map(orbs, pg)
    node_colors = [c_map.get(n, "#cccccc") for n in G.nodes]

    node_labels: dict[int, str] = {}
    for n in pg.nodes:
        attrs_short = "  ".join(
            f"{k}={_fmt_val(v)}" for k, v in n.attrs.items() if k != "dir"
        )
        node_labels[n.prim_id] = f"[{n.prim_id}] {n.subg_type.name}\n{n.label}\n{attrs_short}"

    legend_lines = _orbit_legend_text(orbs, auts, perms, pg)
    legend_text  = "\n".join(legend_lines)

    # --- Figure layout: graph on left, legend text on right ---
    fig, (ax_graph, ax_legend) = plt.subplots(
        1, 2,
        figsize=(16, max(7, 0.45 * len(legend_lines) + 2)),
        gridspec_kw={"width_ratios": [1.6, 1]},
    )

    # Draw graph
    nx.draw_networkx_edges(G, pos, ax=ax_graph, alpha=0.55, width=1.8)
    nx.draw_networkx_nodes(
        G, pos, ax=ax_graph,
        node_color=node_colors, node_size=2400, alpha=0.92,
    )
    nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax_graph, font_size=7)

    # Orbit colour legend patches (only for orbits with >1 node)
    legend_patches = []
    for orb in sorted(orbs, key=lambda o: (-len(o), min(o))):
        if len(orb) <= 1:
            continue
        col    = c_map[next(iter(orb))]
        ids    = sorted(orb)
        label  = "orbit {" + ", ".join(str(i) for i in ids) + "}"
        legend_patches.append(mpatches.Patch(color=col, label=label))
    if legend_patches:
        ax_graph.legend(
            handles=legend_patches, loc="lower left",
            fontsize=8, framealpha=0.85,
        )

    ax_graph.set_title(
        f"Primitive graph: {name}\n"
        f"|Aut| = {len(auts)}   orbits = {len(orbs)}   "
        f"nodes = {len(pg.nodes)}   edges = {len(pg.edges)}",
        fontsize=11,
    )
    ax_graph.axis("off")

    # Legend panel
    ax_legend.axis("off")
    ax_legend.text(
        0.04, 0.97, legend_text,
        transform=ax_legend.transAxes,
        va="top", ha="left",
        fontsize=8,
        fontfamily="monospace",
        wrap=True,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f9f9f9", edgecolor="#cccccc"),
    )
    ax_legend.set_title("Symmetry analysis", fontsize=10)

    fig.suptitle(f"Symmetry analysis: {name}", fontsize=13, fontweight="bold")
    fig.tight_layout()

    outdir.mkdir(parents=True, exist_ok=True)
    safe_name = name.replace(" ", "_").replace("/", "_")
    out_path  = outdir / f"{safe_name}__symmetries.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ===========================================================================
# Test helpers
# ===========================================================================

def _mk_raw(name, specs, edges, order=None) -> CQEDTopology:
    t = CQEDTopology(name)
    order = list(range(len(specs))) if order is None else order
    objs  = {}
    for i in order:
        typ, label, attrs = specs[i]
        objs[i] = t.add_node(typ, dict(attrs), label)
    for i, j in edges:
        t.add_edge(objs[i], objs[j])
    return t


def _nx_type_graph(pg: PrimitiveGraph) -> nx.Graph:
    G = nx.Graph()
    for n in pg.nodes:
        G.add_node(n.prim_id, t=int(n.subg_type))
    G.add_edges_from(pg.edges)
    return G


def _iso_by_type(a: PrimitiveGraph, b: PrimitiveGraph) -> bool:
    return nx.is_isomorphic(
        _nx_type_graph(a), _nx_type_graph(b),
        node_match=lambda x, y: x["t"] == y["t"],
    )


def _assert_no_same_type_edges(pg: PrimitiveGraph) -> None:
    type_map = {n.prim_id: n.subg_type for n in pg.nodes}
    bad = [(u, v) for u, v in pg.edges if type_map[u] == type_map[v]]
    assert bad == [], f"Same-type edges in primitive graph: {bad}"


def _all_param_keys(pg: PrimitiveGraph) -> set[tuple[int, str]]:
    from circuit2graph.definitions import SUBG_DEFS
    keys = set()
    for n in pg.nodes:
        for attr in SUBG_DEFS[n.subg_type].attrs:
            if attr == "dir":
                continue
            if attr in n.attrs:
                keys.add((n.prim_id, attr))
    return keys


# ===========================================================================
# ── Section 1: expand_to_primitive structural invariants ──
# ===========================================================================

@pytest.mark.parametrize("name,builder", USER_TOPOLOGY_BUILDERS.items())
def test_expand_to_primitive_from_compressed_is_isomorphic_to_raw(name, builder):
    raw    = builder()
    comp   = graphlize(raw)
    pg_raw = expand_to_primitive(raw)
    pg_comp = expand_to_primitive(comp)

    print_symmetry_summary(name, raw, comp)
    plot_path = plot_symmetries(name, raw, comp)
    print(f"  → plot saved: {plot_path}")

    assert _iso_by_type(pg_raw, pg_comp), (
        f"{name}: expand_to_primitive(compressed) ≠ expand_to_primitive(raw)"
    )


@pytest.mark.parametrize("name,builder", USER_TOPOLOGY_BUILDERS.items())
def test_expand_to_primitive_has_no_same_type_edges(name, builder):
    pg = expand_to_primitive(graphlize(builder()))
    _assert_no_same_type_edges(pg)


# ===========================================================================
# ── Section 2: identity always present ──
# ===========================================================================

@pytest.mark.parametrize("name,builder", USER_TOPOLOGY_BUILDERS.items())
def test_identity_always_present(name, builder):
    pg   = expand_to_primitive(graphlize(builder()))
    auts = compute_automorphisms(pg)
    identity = {n.prim_id: n.prim_id for n in pg.nodes}
    assert identity in auts, f"{name}: identity not in automorphisms"


# ===========================================================================
# ── Section 3: known symmetry counts (with full terminal + plot output) ──
# ===========================================================================

_NAMED_CASES: list[tuple[str, list, list, int]] = [
    (
        "line_T_C_T",
        [(T,"T1",_t_attrs(1e-9)), (C,"C12",_cc_attrs(1e-15)), (T,"T2",_t_attrs(2e-9))],
        [(0,1),(1,2)], 2,
    ),
    (
        "line_T_C_T_same_L",
        [(T,"T1",_t_attrs(1e-9)), (C,"C12",_cc_attrs(1e-15)), (T,"T2",_t_attrs(1e-9))],
        [(0,1),(1,2)], 2,
    ),
    (
        "single_transmon",
        [(T,"T",_t_attrs(10e-9))],
        [], 1,
    ),
    (
        "line_R_C_T",
        [(R,"R",_r_attrs(3e-3)), (C,"C",_cc_attrs(1e-15)), (T,"T",_t_attrs(10e-9))],
        [(0,1),(1,2)], 1,
    ),
    (
        "hub3_T_T_T",
        [
            (F,"F",{}),
            (C,"C1",_cc_attrs(1e-15)), (T,"T1",_t_attrs(1e-9)),
            (C,"C2",_cc_attrs(2e-15)), (T,"T2",_t_attrs(1e-9)),
            (C,"C3",_cc_attrs(3e-15)), (T,"T3",_t_attrs(1e-9)),
        ],
        [(0,1),(1,2),(0,3),(3,4),(0,5),(5,6)], 6,
    ),
    (
        "hub2T_1R",
        [
            (F,"F",{}),
            (C,"C1",_cc_attrs(1e-15)), (T,"T1",_t_attrs(1e-9)),
            (C,"C2",_cc_attrs(2e-15)), (T,"T2",_t_attrs(1e-9)),
            (C,"C3",_cc_attrs(3e-15)), (R,"R3",_r_attrs(3e-3)),
        ],
        [(0,1),(1,2),(0,3),(3,4),(0,5),(5,6)], 2,
    ),
    (
        "cycle3_T_T_T",
        [
            (T,"T1",_t_attrs(1e-9)), (T,"T2",_t_attrs(2e-9)), (T,"T3",_t_attrs(3e-9)),
            (C,"C12",_cc_attrs(12e-15)), (C,"C23",_cc_attrs(23e-15)), (C,"C31",_cc_attrs(31e-15)),
        ],
        [(0,3),(3,1),(1,4),(4,2),(2,5),(5,0)], 6,
    ),
    (
        "cycle4_T_T_T_T",
        [
            (T,"T0",_t_attrs(1e-9)), (T,"T1",_t_attrs(2e-9)),
            (T,"T2",_t_attrs(3e-9)), (T,"T3",_t_attrs(4e-9)),
            (C,"C0",_cc_attrs(20e-15)), (C,"C1",_cc_attrs(21e-15)),
            (C,"C2",_cc_attrs(22e-15)), (C,"C3",_cc_attrs(23e-15)),
        ],
        [(0,4),(4,1),(1,5),(5,2),(2,6),(6,3),(3,7),(7,0)], 8,
    ),
    (
        "asym_cycle_T_R_T",
        [
            (T,"T1",_t_attrs(1e-9)), (R,"R",_r_attrs(9e-3)), (T,"T2",_t_attrs(2e-9)),
            (C,"C1",_cc_attrs(1e-15)), (C,"C2",_cc_attrs(2e-15)), (C,"C3",_cc_attrs(3e-15)),
        ],
        [(0,3),(3,1),(1,4),(4,2),(2,5),(5,0)], 2,
    ),
    (
        "inductive_R_I_R",
        [
            (R,"R1",_r_attrs(3e-3)),
            (I,"I",_i_attrs(9.0, 1e-4)),
            (R,"R2",_r_attrs(3e-3)),
        ],
        [(0,1),(1,2)], 2,
    ),
]


@pytest.mark.parametrize("name,specs,edges,expected_aut", _NAMED_CASES)
def test_known_symmetry_count(name, specs, edges, expected_aut):
    raw  = _mk_raw(name, specs, edges)
    comp = graphlize(raw)

    print_symmetry_summary(name, raw, comp)
    plot_path = plot_symmetries(name, raw, comp)
    print(f"  → plot saved: {plot_path}")

    pg   = expand_to_primitive(comp)
    auts = compute_automorphisms(pg)
    assert len(auts) == expected_aut, (
        f"{name}: expected |Aut|={expected_aut}, got {len(auts)}"
    )


# ===========================================================================
# ── Section 4: param_permutations completeness ──
# ===========================================================================

@pytest.mark.parametrize("name,builder", USER_TOPOLOGY_BUILDERS.items())
def test_param_permutations_cover_all_keys(name, builder):
    raw  = builder()
    pg   = expand_to_primitive(raw)
    auts = compute_automorphisms(pg)
    perms = param_permutations(pg, auts)
    expected_keys = _all_param_keys(pg)

    print()
    print(f"  {name}: |Aut|={len(auts)}  expected param keys={len(expected_keys)}")
    for i, perm in enumerate(perms):
        is_id = i == 0
        tag   = "[id]" if is_id else f"[#{i}]"
        print(f"    perm {tag}  keys={len(perm)}  "
              f"cycles={_cycles_str(auts[i], pg)}")

    for i, perm in enumerate(perms):
        assert set(perm.keys()) == expected_keys, (
            f"{name}: automorphism #{i} covers {set(perm.keys())} "
            f"but expected {expected_keys}"
        )


def test_param_permutations_identity_is_noop():
    """The identity permutation must not change any parameter value."""
    raw   = build_qubit_resonator_resonator()
    pg    = expand_to_primitive(raw)
    auts  = compute_automorphisms(pg)
    id_perm = param_permutations(pg, [auts[0]])[0]

    print()
    print("  Identity permutation check for qubit_resonator_resonator:")
    for (prim_id, attr), val in sorted(id_perm.items()):
        node = pg.nodes[prim_id]
        expected = node.attrs[attr]
        status = "✓" if val == expected else "✗"
        print(f"    {status}  prim[{prim_id}].{attr}: {_fmt_val(expected)} → {_fmt_val(val)}")

    for (prim_id, attr), val in id_perm.items():
        node = pg.nodes[prim_id]
        assert val == node.attrs[attr], (
            f"Identity changed prim {prim_id} attr {attr}: "
            f"expected {node.attrs[attr]}, got {val}"
        )


# ===========================================================================
# ── Section 5: orbits form a valid partition ──
# ===========================================================================

@pytest.mark.parametrize("name,builder", USER_TOPOLOGY_BUILDERS.items())
def test_orbits_partition_node_ids(name, builder):
    pg   = expand_to_primitive(builder())
    auts = compute_automorphisms(pg)
    orbs = compute_orbits(auts, pg.node_ids())

    print()
    node_by_id = {n.prim_id: n for n in pg.nodes}
    print(f"  {name}  ({len(orbs)} orbits):")
    for orb in sorted(orbs, key=lambda o: min(o)):
        ids    = sorted(orb)
        labels = [f"[{i}]{node_by_id[i].subg_type.name}" for i in ids]
        sym    = " ≡ " if len(orb) > 1 else ""
        print(f"    {sym}{' '.join(labels)}")

    all_ids = [nid for orb in orbs for nid in orb]
    assert sorted(all_ids) == sorted(pg.node_ids()), (
        f"{name}: orbits do not partition node ids"
    )


# ===========================================================================
# ── Section 6: insertion-order invariance ──
# ===========================================================================

@pytest.mark.parametrize("name,specs,edges", [
    (
        "order_tct",
        [(T,"T1",_t_attrs(1e-9)), (C,"C12",_cc_attrs(1e-15)), (T,"T2",_t_attrs(2e-9))],
        [(0,1),(1,2)],
    ),
    (
        "order_hub3",
        [
            (F,"F",{}),
            (C,"C1",_cc_attrs(1e-15)), (T,"T1",_t_attrs(1e-9)),
            (C,"C2",_cc_attrs(2e-15)), (T,"T2",_t_attrs(1e-9)),
            (C,"C3",_cc_attrs(3e-15)), (T,"T3",_t_attrs(1e-9)),
        ],
        [(0,1),(1,2),(0,3),(3,4),(0,5),(5,6)],
    ),
    (
        "order_cycle3",
        [
            (T,"T1",_t_attrs(1e-9)), (T,"T2",_t_attrs(2e-9)), (T,"T3",_t_attrs(3e-9)),
            (C,"C12",_cc_attrs(12e-15)), (C,"C23",_cc_attrs(23e-15)), (C,"C31",_cc_attrs(31e-15)),
        ],
        [(0,3),(3,1),(1,4),(4,2),(2,5),(5,0)],
    ),
])
def test_aut_count_independent_of_insertion_order(name, specs, edges):
    rng    = random.Random(42)
    counts = set()
    for trial in range(10):
        order = list(range(len(specs)))
        rng.shuffle(order)
        raw   = _mk_raw(name, specs, edges, order=order)
        mac   = graphlize(raw)
        pg    = expand_to_primitive(mac)
        auts  = compute_automorphisms(pg)
        counts.add(len(auts))

    print(f"\n  {name}: aut counts across 10 random insertion orders: {counts}")
    assert len(counts) == 1, (
        f"{name}: |Aut| varies with insertion order: {counts}"
    )


# ===========================================================================
# ── Section 7: stress topologies (with plot) ──
# ===========================================================================

_STRESS_BUILDERS: list[tuple[str, Callable]] = [
    ("feedline_two_branch_hub_plus_cycle", build_feedline_two_branch_hub_plus_cycle),
    ("inductive_resonator_line_and_hub",   build_inductive_resonator_line_and_hub),
    ("four_transmon_ring_with_tail",       build_four_transmon_capacitive_ring_with_tail),
]


@pytest.mark.parametrize("name,builder", _STRESS_BUILDERS)
def test_stress_topologies_symmetry(name, builder):
    raw  = builder()
    comp = graphlize(raw)

    print_symmetry_summary(name, raw, comp)
    plot_path = plot_symmetries(name, raw, comp)
    print(f"  → plot saved: {plot_path}")

    pg    = expand_to_primitive(comp)
    auts  = compute_automorphisms(pg)
    perms = param_permutations(pg, auts)
    expected_keys = _all_param_keys(pg)

    assert len(auts) >= 1
    assert {n.prim_id: n.prim_id for n in pg.nodes} in auts, "Identity missing"
    for i, perm in enumerate(perms):
        assert set(perm.keys()) == expected_keys, (
            f"{name}: automorphism #{i} missing keys: "
            f"{expected_keys - set(perm.keys())}"
        )


# ===========================================================================
# ── Section 8: symmetry_report convenience wrapper ──
# ===========================================================================

def test_symmetry_report_tct():
    raw = _mk_raw(
        "tct",
        [(T,"T1",_t_attrs(1e-9)), (C,"C",_cc_attrs(1e-15)), (T,"T2",_t_attrs(2e-9))],
        [(0,1),(1,2)],
    )
    rep = symmetry_report(raw)

    print()
    print(f"  symmetry_report(T-C-T):")
    print(f"    |Aut| = {rep.n_automorphisms}")
    print(f"    orbits = {[sorted(o) for o in rep.orbits]}")

    assert rep.topology_name == "tct"
    assert rep.n_automorphisms == 2
    t_ids = [n.prim_id for n in rep.primitive_graph.nodes if n.subg_type == T]
    assert len(t_ids) == 2
    same_orbit = any(frozenset(t_ids) == orb for orb in rep.orbits)
    assert same_orbit, "The two transmons in T-C-T must be in the same orbit"


def test_symmetry_report_rct():
    raw = _mk_raw(
        "rct",
        [(R,"R",_r_attrs(3e-3)), (C,"C",_cc_attrs(1e-15)), (T,"T",_t_attrs(10e-9))],
        [(0,1),(1,2)],
    )
    rep = symmetry_report(raw)

    print()
    print(f"  symmetry_report(R-C-T):")
    print(f"    |Aut| = {rep.n_automorphisms}  (expected 1, no symmetry)")

    assert rep.n_automorphisms == 1, (
        f"R-C-T should have |Aut|=1, got {rep.n_automorphisms}"
    )


# ===========================================================================
# ── Section 9: automorphism cache ──
# ===========================================================================

def test_automorphism_cache_roundtrip():
    builders = list(USER_TOPOLOGY_BUILDERS.values())
    topos    = [graphlize(b()) for b in builders]
    build_automorphism_cache(topos)
    summary  = get_cache_summary()
    print(f"\n{summary}")

    assert "Automorphism cache" in summary
    for topo in topos:
        cached = get_cached_automorphisms(topo.name)
        assert cached is not None, f"Cache miss for {topo.name!r}"
        assert len(cached) >= 1


def test_automorphism_cache_is_idempotent():
    raw  = build_qubit_resonator_resonator()
    topo = graphlize(raw)
    build_automorphism_cache([topo])
    first  = get_cached_automorphisms(topo.name)
    build_automorphism_cache([topo])
    second = get_cached_automorphisms(topo.name)
    assert first == second


def test_cache_miss_returns_none():
    assert get_cached_automorphisms("__nonexistent_topology__") is None
