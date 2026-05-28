"""
src/circuit2graph/test_graphlize.py
====================================

Tests and visual diagnostics for circuit2graph.graphlize().

This file is intentionally self-contained and can be run either from the repo
root with pytest or directly as a script.  When run as a script it writes PNGs
showing each topology before and after graphlize() compression.


Generate plots in a custom directory:
    python src/circuit2graph/test_graphlize.py --plot --outdir graphlize_plots
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable

# ---------------------------------------------------------------------------
# Robust imports
# ---------------------------------------------------------------------------
# The production modules use absolute imports like `circuit2graph.definitions`.
# Therefore the importable path must be `<repo_root>/src`, not the package dir.
# This fixes the common failure that happens when the test is launched directly
# from inside src/circuit2graph.
_THIS_FILE = Path(__file__).resolve()
# Works both from <repo_root>/tests/test_graphlize.py and from the old
# <repo_root>/src/circuit2graph/test_graphlize.py location.
if (_THIS_FILE.parents[1] / "src").exists():
    _REPO_ROOT = _THIS_FILE.parents[1]
elif (_THIS_FILE.parents[2] / "src").exists():
    _REPO_ROOT = _THIS_FILE.parents[2]
else:  # pragma: no cover
    raise RuntimeError("Cannot locate repo root containing src/")
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

try:
    import matplotlib.pyplot as plt
    import networkx as nx
    import pytest
except ImportError as exc:  # pragma: no cover - import guard for direct usage
    raise SystemExit(
        "Missing test/plot dependency. Install pytest, matplotlib and networkx. "
        f"Original error: {exc}"
    ) from exc

from circuit2graph.definitions import NODE_COLORS, SUBG_NODE, SubgType
from circuit2graph.topology import CQEDTopology
from circuit2graph.compression import graphlize


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _t_attrs(L: float) -> dict[str, float]:
    return {"L": L, "C": 70e-15}


def _r_attrs(length: float) -> dict[str, float]:
    return {"length": length}


def _cc_attrs(Cc: float) -> dict[str, float]:
    return {"Cc": Cc}


def topology_to_networkx(topology: CQEDTopology) -> nx.Graph:
    """Convert CQEDTopology to a plain NetworkX graph for plotting/debugging."""
    G = nx.Graph(name=topology.name)
    for node in topology._nodes:
        type_name = node.subg_type.name
        components = "+".join(SUBG_NODE[node.subg_type])
        label = node.label or f"{type_name}_{node.node_id}"
        G.add_node(
            node.node_id,
            label=label,
            subg_type=node.subg_type,
            type_name=type_name,
            components=components,
            attrs=node.attrs,
        )
    G.add_edges_from(topology._edges)
    return G


def plot_before_after(
    raw: CQEDTopology,
    compressed: CQEDTopology | None = None,
    outdir: str | os.PathLike[str] = "graphlize_plots",
    show: bool = False,
) -> Path:
    """Plot raw and compressed topology side by side using NetworkX."""
    if compressed is None:
        compressed = graphlize(raw)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, topo, title in (
        (axes[0], raw, "before graphlize: raw elements"),
        (axes[1], compressed, "after graphlize: compressed blocks"),
    ):
        G = topology_to_networkx(topo)
        pos = nx.spring_layout(G, seed=7)
        colors = [NODE_COLORS.get(G.nodes[n]["subg_type"], "#cccccc") for n in G.nodes]
        labels = {
            n: f"{G.nodes[n]['label']}\n{G.nodes[n]['components']}"
            for n in G.nodes
        }
        nx.draw_networkx_edges(G, pos, ax=ax, width=1.8, alpha=0.65)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors, node_size=1700)
        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=8)
        ax.set_title(f"{topo.name}\n{title}")
        ax.axis("off")

    fig.suptitle(raw.name, fontsize=14)
    fig.tight_layout()
    path = outdir / f"{raw.name}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return path


def compressed_type_names(topology: CQEDTopology) -> list[str]:
    """Return compressed node type names, sorted for stable assertions."""
    return sorted(node.subg_type.name for node in topology._nodes)


# ---------------------------------------------------------------------------
# User-requested topology builders
# ---------------------------------------------------------------------------

def build_qubit_resonator_resonator() -> CQEDTopology:
    """transmon - cc - resonator - cc - resonator"""
    t = CQEDTopology("qubit_resonator_resonator")
    q = t.add_node(SubgType.TRANSMON, _t_attrs(12e-9), label="T1")
    c_qr = t.add_node(SubgType.C_COUPLER, _cc_attrs(10e-15), label="Cc_qr")
    r1 = t.add_node(SubgType.RESONATOR, _r_attrs(3.50e-3), label="R1")
    c_rr = t.add_node(SubgType.C_COUPLER, _cc_attrs(3e-15), label="Cc_rr")
    r2 = t.add_node(SubgType.RESONATOR, _r_attrs(3.86e-3), label="R2")
    t.add_edge(q, c_qr)
    t.add_edge(c_qr, r1)
    t.add_edge(r1, c_rr)
    t.add_edge(c_rr, r2)
    return t


def build_resonator_qubit_resonator() -> CQEDTopology:
    """resonator - cc - transmon - cc - resonator"""
    t = CQEDTopology("resonator_qubit_resonator")
    r1 = t.add_node(SubgType.RESONATOR, _r_attrs(3.50e-3), label="R1")
    c1 = t.add_node(SubgType.C_COUPLER, _cc_attrs(10e-15), label="Cc_qr1")
    q = t.add_node(SubgType.TRANSMON, _t_attrs(12e-9), label="T1")
    c2 = t.add_node(SubgType.C_COUPLER, _cc_attrs(3e-15), label="Cc_qr2")
    r2 = t.add_node(SubgType.RESONATOR, _r_attrs(3.86e-3), label="R2")
    t.add_edge(r1, c1)
    t.add_edge(c1, q)
    t.add_edge(q, c2)
    t.add_edge(c2, r2)
    return t


def build_three_qubit_capacitive_line() -> CQEDTopology:
    """transmon - cc - transmon - cc - transmon"""
    t = CQEDTopology("three_qubit_capacitive_line")
    q1 = t.add_node(SubgType.TRANSMON, _t_attrs(10.9e-9), label="T1")
    c12 = t.add_node(SubgType.C_COUPLER, _cc_attrs(9.22e-15), label="Cc_12")
    q2 = t.add_node(SubgType.TRANSMON, _t_attrs(12.0e-9), label="T2")
    c23 = t.add_node(SubgType.C_COUPLER, _cc_attrs(6.89e-15), label="Cc_23")
    q3 = t.add_node(SubgType.TRANSMON, _t_attrs(12.0e-9), label="T3")
    t.add_edge(q1, c12)
    t.add_edge(c12, q2)
    t.add_edge(q2, c23)
    t.add_edge(c23, q3)
    return t


def build_three_qubit_capacitive_star() -> CQEDTopology:
    """Triangle: T1 - C12 - T2 - C23 - T3 - C31 - T1."""
    t = CQEDTopology("three_qubit_capacitive_star")
    q1 = t.add_node(SubgType.TRANSMON, _t_attrs(10.6e-9), label="T1")
    c12 = t.add_node(SubgType.C_COUPLER, _cc_attrs(3.0e-15), label="Cc_12")
    q2 = t.add_node(SubgType.TRANSMON, _t_attrs(12.0e-9), label="T2")
    c23 = t.add_node(SubgType.C_COUPLER, _cc_attrs(3.0e-15), label="Cc_23")
    q3 = t.add_node(SubgType.TRANSMON, _t_attrs(12.0e-9), label="T3")
    c31 = t.add_node(SubgType.C_COUPLER, _cc_attrs(3.0e-15), label="Cc_31")
    t.add_edge(q1, c12)
    t.add_edge(c12, q2)
    t.add_edge(q2, c23)
    t.add_edge(c23, q3)
    t.add_edge(q3, c31)
    t.add_edge(c31, q1)
    return t


def build_two_qubit_resonator() -> CQEDTopology:
    """transmon - cc - transmon - cc - resonator"""
    t = CQEDTopology("two_qubit_resonator")
    q1 = t.add_node(SubgType.TRANSMON, _t_attrs(12e-9), label="T1")
    c12 = t.add_node(SubgType.C_COUPLER, _cc_attrs(4e-15), label="Cc_12")
    q2 = t.add_node(SubgType.TRANSMON, _t_attrs(12e-9), label="T2")
    c2r = t.add_node(SubgType.C_COUPLER, _cc_attrs(7e-15), label="Cc_2r")
    r = t.add_node(SubgType.RESONATOR, _r_attrs(4.21e-3), label="R")
    t.add_edge(q1, c12)
    t.add_edge(c12, q2)
    t.add_edge(q2, c2r)
    t.add_edge(c2r, r)
    return t


USER_TOPOLOGY_BUILDERS: dict[str, Callable[[], CQEDTopology]] = {
    "qubit_resonator_resonator": build_qubit_resonator_resonator,
    "resonator_qubit_resonator": build_resonator_qubit_resonator,
    "three_qubit_capacitive_line": build_three_qubit_capacitive_line,
    "three_qubit_capacitive_star": build_three_qubit_capacitive_star,
    "two_qubit_resonator": build_two_qubit_resonator,
}


# ---------------------------------------------------------------------------
# Tests for the user-requested cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,builder", USER_TOPOLOGY_BUILDERS.items())
def test_user_topologies_graphlize_without_crashing(name: str, builder: Callable[[], CQEDTopology]) -> None:
    raw = builder()
    compressed = graphlize(raw)
    assert compressed.name == f"{raw.name}__outer"
    assert len(compressed._nodes) > 0
    assert len(compressed._nodes) <= len(raw._nodes)


def test_qubit_resonator_resonator_expected_blocks() -> None:
    # Root selection prefers the highest-priority coupler. Here Cc_qr is kept
    # as the traversal root; the right branch R-C-R compresses only the R-C
    # prefix into RC. There is no R-C-R compressed block in the present basis.
    compressed = graphlize(build_qubit_resonator_resonator())
    assert compressed_type_names(compressed) == ["C_COUPLER", "RC", "RESONATOR", "TRANSMON"]


def test_resonator_qubit_resonator_expected_blocks() -> None:
    # Cc_qr1 is kept as the root; the right branch T-C-R becomes RCT.
    compressed = graphlize(build_resonator_qubit_resonator())
    assert compressed_type_names(compressed) == ["C_COUPLER", "RCT", "RESONATOR"]


def test_three_qubit_capacitive_line_expected_blocks() -> None:
    # Cc_12 is kept as the root; the right branch T-C-T becomes TCT.
    # There is no 3-qubit-line block yet.
    compressed = graphlize(build_three_qubit_capacitive_line())
    assert compressed_type_names(compressed) == ["C_COUPLER", "TCT", "TRANSMON"]


def test_three_qubit_capacitive_star_expected_current_behavior() -> None:
    # Important diagnostic: this graph is a cycle. The current graphlize()
    # implementation is tree/branch-oriented, so the triangle is represented
    # as one TCT block plus the remaining transmon/couplers around the cycle.
    compressed = graphlize(build_three_qubit_capacitive_star())
    assert compressed_type_names(compressed) == ["C_COUPLER", "C_COUPLER", "TCT", "TRANSMON"]


def test_two_qubit_resonator_expected_blocks() -> None:
    # Cc_12 is kept as the root; the right branch T-C-R becomes RCT.
    compressed = graphlize(build_two_qubit_resonator())
    assert compressed_type_names(compressed) == ["C_COUPLER", "RCT", "TRANSMON"]


def test_plot_before_after_writes_png(tmp_path: Path) -> None:
    raw = build_two_qubit_resonator()
    out = plot_before_after(raw, outdir=tmp_path)
    assert out.exists()
    assert out.suffix == ".png"


# ---------------------------------------------------------------------------
# Direct script entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect graphlize() on selected cQED topologies.")
    parser.add_argument("--plot", action="store_true", help="write before/after PNG plots")
    parser.add_argument("--show", action="store_true", help="show matplotlib windows while plotting")
    parser.add_argument("--outdir", default="graphlize_plots", help="directory for generated PNG plots")
    args = parser.parse_args()

    for name, builder in USER_TOPOLOGY_BUILDERS.items():
        raw = builder()
        compressed = graphlize(raw)
        print("=" * 88)
        print(name)
        print("RAW")
        print(raw.summary())
        print("COMPRESSED")
        print(compressed.summary())
        print("compressed types:", compressed_type_names(compressed))
        if args.plot:
            path = plot_before_after(raw, compressed, outdir=args.outdir, show=args.show)
            print(f"plot written to: {path}")


if __name__ == "__main__":
    main()
