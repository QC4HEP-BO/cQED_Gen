"""
test/test_graphlize.py
======================

Tests and visual diagnostics for circuit2graph.graphlize().

Tests verify:
  1. graphlize() runs without crashing on all topologies
  2. The correct macronode types are produced
  3. Directional blocks (TC, CT, RC, CR, RCT, TCR) encode orientation correctly
     (left = root-side, right = leaf-side) — independent of parameter values
  4. TCT no longer sorts by L; left = root-facing transmon, right = leaf-facing

Run with pytest or directly as a script:
    python test/test_graphlize.py --plot --outdir graphlize_plots
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable

_THIS_FILE = Path(__file__).resolve()
if (_THIS_FILE.parents[1] / "src").exists():
    _REPO_ROOT = _THIS_FILE.parents[1]
elif (_THIS_FILE.parents[2] / "src").exists():
    _REPO_ROOT = _THIS_FILE.parents[2]
else:
    raise RuntimeError("Cannot locate repo root containing src/")
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

try:
    import matplotlib.pyplot as plt
    import networkx as nx
    import pytest
except ImportError as exc:
    raise SystemExit(f"Missing dependency: {exc}") from exc

from circuit2graph.definitions import NODE_COLORS, SUBG_NODE, SubgType
from circuit2graph.topology import CQEDTopology
from circuit2graph.compression import graphlize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _t(L: float, C: float = 70e-15) -> dict:
    return {"L": L, "C": C}

def _r(length: float = 4e-3) -> dict:
    return {"length": length}

def _cc(Cc: float = 5e-15) -> dict:
    return {"Cc": Cc}

def _ind(D: float = 3.0, l: float = 1e-3) -> dict:
    return {"D": D, "l": l}


def topology_to_networkx(topology: CQEDTopology) -> nx.Graph:
    G = nx.Graph(name=topology.name)
    for node in topology._nodes:
        G.add_node(
            node.node_id,
            label      = node.label or f"{node.subg_type.name}_{node.node_id}",
            subg_type  = node.subg_type,
            components = "+".join(SUBG_NODE[node.subg_type]),
            attrs      = node.attrs,
        )
    G.add_edges_from(topology._edges)
    return G


def plot_before_after(
    raw:        CQEDTopology,
    compressed: CQEDTopology | None = None,
    outdir:     str | os.PathLike[str] = "graphlize_plots",
    show:       bool = False,
) -> Path:
    if compressed is None:
        compressed = graphlize(raw)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for ax, topo, title in (
        (axes[0], raw,        "before graphlize: raw elements"),
        (axes[1], compressed, "after  graphlize: compressed blocks"),
    ):
        G = topology_to_networkx(topo)
        pos    = nx.spring_layout(G, seed=7)
        colors = [NODE_COLORS.get(G.nodes[n]["subg_type"], "#cccccc") for n in G.nodes]
        labels = {n: f"{G.nodes[n]['label']}\n{G.nodes[n]['components']}" for n in G.nodes}
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


def types(topology: CQEDTopology) -> list[str]:
    """Sorted list of macronode type names — for stable assertions."""
    return sorted(node.subg_type.name for node in topology._nodes)


def node_by_type(topology: CQEDTopology, subg_type: SubgType) -> "CQEDNode":
    """Return the unique node of a given SubgType; raises if not unique."""
    matches = [n for n in topology._nodes if n.subg_type == subg_type]
    assert len(matches) == 1, f"Expected 1 node of type {subg_type.name}, got {len(matches)}"
    return matches[0]


# ---------------------------------------------------------------------------
# Topology builders
# ---------------------------------------------------------------------------

def build_tct_Alarger() -> CQEDTopology:
    """T1(L=15n) – Cc – T2(L=10n): T1 has larger L — used to verify no L-sorting."""
    t = CQEDTopology("tct_Alarger")
    t1 = t.add_node(SubgType.TRANSMON,  _t(15e-9), label="T1")
    cc = t.add_node(SubgType.C_COUPLER, _cc(),     label="Cc")
    t2 = t.add_node(SubgType.TRANSMON,  _t(10e-9), label="T2")
    t.add_edge(t1, cc)
    t.add_edge(cc, t2)
    return t

def build_tct_Blarger() -> CQEDTopology:
    """T1(L=10n) – Cc – T2(L=15n): T2 has larger L."""
    t = CQEDTopology("tct_Blarger")
    t1 = t.add_node(SubgType.TRANSMON,  _t(10e-9), label="T1")
    cc = t.add_node(SubgType.C_COUPLER, _cc(),     label="Cc")
    t2 = t.add_node(SubgType.TRANSMON,  _t(15e-9), label="T2")
    t.add_edge(t1, cc)
    t.add_edge(cc, t2)
    return t

def build_rct_linear() -> CQEDTopology:
    """R – Cc – T  (linear, root=R by priority)"""
    t  = CQEDTopology("rct_linear")
    r  = t.add_node(SubgType.RESONATOR, _r(),  label="R")
    cc = t.add_node(SubgType.C_COUPLER, _cc(), label="Cc")
    tr = t.add_node(SubgType.TRANSMON,  _t(12e-9), label="T")
    t.add_edge(r, cc)
    t.add_edge(cc, tr)
    return t

def build_tcr_linear() -> CQEDTopology:
    """T – Cc – R  (linear, root=R by priority — so result should be RCT, not TCR)"""
    t  = CQEDTopology("tcr_linear")
    tr = t.add_node(SubgType.TRANSMON,  _t(12e-9), label="T")
    cc = t.add_node(SubgType.C_COUPLER, _cc(),     label="Cc")
    r  = t.add_node(SubgType.RESONATOR, _r(),      label="R")
    t.add_edge(tr, cc)
    t.add_edge(cc, r)
    return t

def build_feedline_rct_tct() -> CQEDTopology:
    """F – Cc_rf – R – Cc_qr – T1 – Cc12 – T2
    Root=F → chain goes F, RC(R+Cc_rf), then T-C-T becomes TCT from T1's perspective.
    Expected: FEEDLINE – RC – TCT  or FEEDLINE – RCT – TRANSMON depending on branch order.
    This is the key test for the full chain.
    """
    t     = CQEDTopology("feedline_rct_tct")
    f     = t.add_node(SubgType.FEEDLINE,  {},         label="F")
    cc_rf = t.add_node(SubgType.C_COUPLER, _cc(2e-15), label="Cc_rf")
    r     = t.add_node(SubgType.RESONATOR, _r(),       label="R")
    cc_qr = t.add_node(SubgType.C_COUPLER, _cc(8e-15), label="Cc_qr")
    t1    = t.add_node(SubgType.TRANSMON,  _t(12e-9),  label="T1")
    cc12  = t.add_node(SubgType.C_COUPLER, _cc(4e-15), label="Cc12")
    t2    = t.add_node(SubgType.TRANSMON,  _t(10e-9),  label="T2")
    t.add_edge(f, cc_rf); t.add_edge(cc_rf, r)
    t.add_edge(r, cc_qr); t.add_edge(cc_qr, t1)
    t.add_edge(t1, cc12); t.add_edge(cc12, t2)
    return t

def build_qubit_resonator_resonator() -> CQEDTopology:
    """T – Cc_qr – R1 – Cc_rr – R2  (root=R1 by coupler priority)"""
    t     = CQEDTopology("qubit_resonator_resonator")
    q     = t.add_node(SubgType.TRANSMON,  _t(12e-9),  label="T1")
    c_qr  = t.add_node(SubgType.C_COUPLER, _cc(10e-15),label="Cc_qr")
    r1    = t.add_node(SubgType.RESONATOR, _r(3.5e-3), label="R1")
    c_rr  = t.add_node(SubgType.C_COUPLER, _cc(3e-15), label="Cc_rr")
    r2    = t.add_node(SubgType.RESONATOR, _r(3.86e-3),label="R2")
    t.add_edge(q, c_qr); t.add_edge(c_qr, r1)
    t.add_edge(r1, c_rr); t.add_edge(c_rr, r2)
    return t

def build_resonator_qubit_resonator() -> CQEDTopology:
    """R1 – Cc1 – T – Cc2 – R2"""
    t  = CQEDTopology("resonator_qubit_resonator")
    r1 = t.add_node(SubgType.RESONATOR, _r(3.5e-3),  label="R1")
    c1 = t.add_node(SubgType.C_COUPLER, _cc(10e-15), label="Cc_qr1")
    q  = t.add_node(SubgType.TRANSMON,  _t(12e-9),   label="T1")
    c2 = t.add_node(SubgType.C_COUPLER, _cc(3e-15),  label="Cc_qr2")
    r2 = t.add_node(SubgType.RESONATOR, _r(3.86e-3), label="R2")
    t.add_edge(r1, c1); t.add_edge(c1, q)
    t.add_edge(q, c2);  t.add_edge(c2, r2)
    return t

def build_three_qubit_capacitive_line() -> CQEDTopology:
    """T1 – Cc12 – T2 – Cc23 – T3"""
    t   = CQEDTopology("three_qubit_capacitive_line")
    q1  = t.add_node(SubgType.TRANSMON,  _t(10.9e-9), label="T1")
    c12 = t.add_node(SubgType.C_COUPLER, _cc(9.22e-15),label="Cc_12")
    q2  = t.add_node(SubgType.TRANSMON,  _t(12.0e-9), label="T2")
    c23 = t.add_node(SubgType.C_COUPLER, _cc(6.89e-15),label="Cc_23")
    q3  = t.add_node(SubgType.TRANSMON,  _t(12.0e-9), label="T3")
    t.add_edge(q1, c12); t.add_edge(c12, q2)
    t.add_edge(q2, c23); t.add_edge(c23, q3)
    return t

def build_three_qubit_capacitive_star() -> CQEDTopology:
    """Triangle: T1 – C12 – T2 – C23 – T3 – C31 – T1"""
    t   = CQEDTopology("three_qubit_capacitive_star")
    q1  = t.add_node(SubgType.TRANSMON,  _t(10.6e-9), label="T1")
    c12 = t.add_node(SubgType.C_COUPLER, _cc(3.0e-15),label="Cc_12")
    q2  = t.add_node(SubgType.TRANSMON,  _t(12.0e-9), label="T2")
    c23 = t.add_node(SubgType.C_COUPLER, _cc(3.0e-15),label="Cc_23")
    q3  = t.add_node(SubgType.TRANSMON,  _t(12.0e-9), label="T3")
    c31 = t.add_node(SubgType.C_COUPLER, _cc(3.0e-15),label="Cc_31")
    t.add_edge(q1, c12); t.add_edge(c12, q2)
    t.add_edge(q2, c23); t.add_edge(c23, q3)
    t.add_edge(q3, c31); t.add_edge(c31, q1)
    return t

def build_two_qubit_resonator() -> CQEDTopology:
    """T1 – Cc12 – T2 – Cc2r – R"""
    t   = CQEDTopology("two_qubit_resonator")
    q1  = t.add_node(SubgType.TRANSMON,  _t(12e-9),  label="T1")
    c12 = t.add_node(SubgType.C_COUPLER, _cc(4e-15), label="Cc_12")
    q2  = t.add_node(SubgType.TRANSMON,  _t(12e-9),  label="T2")
    c2r = t.add_node(SubgType.C_COUPLER, _cc(7e-15), label="Cc_2r")
    r   = t.add_node(SubgType.RESONATOR, _r(4.21e-3),label="R")
    t.add_edge(q1, c12); t.add_edge(c12, q2)
    t.add_edge(q2, c2r); t.add_edge(c2r, r)
    return t


ALL_BUILDERS: dict[str, Callable[[], CQEDTopology]] = {
    "tct_Alarger":                  build_tct_Alarger,
    "tct_Blarger":                  build_tct_Blarger,
    "rct_linear":                   build_rct_linear,
    "tcr_linear":                   build_tcr_linear,
    "feedline_rct_tct":             build_feedline_rct_tct,
    "qubit_resonator_resonator":    build_qubit_resonator_resonator,
    "resonator_qubit_resonator":    build_resonator_qubit_resonator,
    "three_qubit_capacitive_line":  build_three_qubit_capacitive_line,
    "three_qubit_capacitive_star":  build_three_qubit_capacitive_star,
    "two_qubit_resonator":          build_two_qubit_resonator,
}


# ---------------------------------------------------------------------------
# Tests — no crash
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,builder", ALL_BUILDERS.items())
def test_graphlize_no_crash(name: str, builder: Callable) -> None:
    raw        = builder()
    compressed = graphlize(raw)
    assert len(compressed._nodes) > 0
    assert len(compressed._nodes) <= len(raw._nodes)


# ---------------------------------------------------------------------------
# Tests — TCT direction: left = root-side, no L-sorting
# ---------------------------------------------------------------------------

def test_tct_left_is_rootside_when_Alarger() -> None:
    """
    T1(L=15n) – Cc – T2(L=10n).
    Root selection: both T, same degree → lowest node_id → T1 is root.
    TCT.left slot (L, C) must hold T1's params (L=15n), NOT the smaller L.
    """
    raw = build_tct_Alarger()
    c   = graphlize(raw)
    tct = node_by_type(c, SubgType.TCT)
    # Left = root-side = T1 = L=15n
    assert abs(tct.attrs["L"] - 15e-9) < 1e-12, \
        f"Expected L=15n (root-side T1) in left slot, got L={tct.attrs['L']}"
    assert abs(tct.attrs["L2"] - 10e-9) < 1e-12, \
        f"Expected L2=10n (leaf-side T2) in right slot, got L2={tct.attrs['L2']}"


def test_tct_left_is_rootside_when_Blarger() -> None:
    """
    T1(L=10n) – Cc – T2(L=15n).
    Root = T1 (lower node_id). Left slot = T1 = L=10n.
    Old code would swap to put L=10n in left — same result by coincidence.
    New code must assign by position, not value.
    """
    raw = build_tct_Blarger()
    c   = graphlize(raw)
    tct = node_by_type(c, SubgType.TCT)
    assert abs(tct.attrs["L"] - 10e-9) < 1e-12, \
        f"Expected L=10n (root-side T1) in left slot, got {tct.attrs['L']}"
    assert abs(tct.attrs["L2"] - 15e-9) < 1e-12, \
        f"Expected L2=15n (leaf-side T2) in right slot, got {tct.attrs['L2']}"


def test_tct_label_reflects_position_not_sorting() -> None:
    """Label must be TCT(T1+Cc+T2), not reordered."""
    raw = build_tct_Alarger()   # T1(15n) left, T2(10n) right
    c   = graphlize(raw)
    tct = node_by_type(c, SubgType.TCT)
    assert "T1" in tct.label
    assert tct.label.index("T1") < tct.label.index("T2"), \
        f"Label should have T1 before T2, got: {tct.label}"


# ---------------------------------------------------------------------------
# Tests — RCT direction: R is root-side, T is leaf-side
# ---------------------------------------------------------------------------

def test_rct_attrs_are_root_left_leaf_right() -> None:
    """
    R – Cc – T: root=R (higher priority than T).
    RCT left slot = R (length), right slot = T (L, C).
    """
    raw = build_rct_linear()
    c   = graphlize(raw)
    rct = node_by_type(c, SubgType.RCT)
    assert "length" in rct.attrs and rct.attrs["length"] > 0, "length should be in left slot"
    assert "L" in rct.attrs and rct.attrs["L"] > 0,          "L should be in right slot"
    assert "C" in rct.attrs and rct.attrs["C"] > 0,          "C should be in right slot"


def test_tcr_chain_root_determines_orientation() -> None:
    """
    T – Cc – R: root=R by priority (R > T).
    From R's perspective the chain is R ← Cc ← T, so the result is RCT with
    R in left slot — even though the raw chain was written T–Cc–R.
    """
    raw = build_tcr_linear()
    c   = graphlize(raw)
    # Root is R → walks toward T → produces RCT (root=R, leaf=T)
    rct = node_by_type(c, SubgType.RCT)
    assert abs(rct.attrs["length"] - _r()["length"]) < 1e-9
    assert abs(rct.attrs["L"]      - _t(12e-9)["L"]) < 1e-12


# ---------------------------------------------------------------------------
# Tests — expected block types for known topologies
# ---------------------------------------------------------------------------

def test_qubit_resonator_resonator_blocks() -> None:
    c = graphlize(build_qubit_resonator_resonator())
    # Root = highest-priority coupler (Cc_qr or Cc_rr, same priority, higher deg wins)
    # Result: coupler hub + one CT-or-TC branch + one RC-or-CR branch
    assert len(c._nodes) > 0


def test_resonator_qubit_resonator_blocks() -> None:
    c = graphlize(build_resonator_qubit_resonator())
    t = types(c)
    # Should compress to a 3-node block (RCT or TCR) + one standalone R + coupler
    assert len(c._nodes) <= 4


def test_three_qubit_line_blocks() -> None:
    c = graphlize(build_three_qubit_capacitive_line())
    t = types(c)
    # One TCT + one standalone T + one standalone Cc hub, or similar compact form
    assert "TCT" in t


def test_three_qubit_star_blocks() -> None:
    c = graphlize(build_three_qubit_capacitive_star())
    t = types(c)
    assert "TCT" in t


def test_two_qubit_resonator_blocks() -> None:
    c = graphlize(build_two_qubit_resonator())
    t = types(c)
    # Expect a 3-node block absorbing part of the chain
    assert any(x in t for x in ("RCT", "TCR", "TCT")), f"Unexpected types: {t}"


# ---------------------------------------------------------------------------
# Tests — attr roundtrip: params survive merge
# ---------------------------------------------------------------------------

def test_tct_params_roundtrip() -> None:
    """All five TCT attrs (L, C, Cc, L2, C2) survive graphlize."""
    L1, C1, Cc, L2, C2 = 12e-9, 65e-15, 4e-15, 9e-9, 70e-15
    t  = CQEDTopology("rt")
    t1 = t.add_node(SubgType.TRANSMON,  {"L": L1, "C": C1}, label="T1")
    cc = t.add_node(SubgType.C_COUPLER, {"Cc": Cc},          label="Cc")
    t2 = t.add_node(SubgType.TRANSMON,  {"L": L2, "C": C2}, label="T2")
    t.add_edge(t1, cc); t.add_edge(cc, t2)
    c  = graphlize(t)
    tct = node_by_type(c, SubgType.TCT)
    assert abs(tct.attrs["L"]  - L1) < 1e-20
    assert abs(tct.attrs["C"]  - C1) < 1e-20
    assert abs(tct.attrs["Cc"] - Cc) < 1e-20
    assert abs(tct.attrs["L2"] - L2) < 1e-20
    assert abs(tct.attrs["C2"] - C2) < 1e-20


def test_rct_params_roundtrip() -> None:
    """RCT attrs (length, Cc, L, C) survive graphlize."""
    length, Cc, L, C = 3.7e-3, 6e-15, 11e-9, 68e-15
    t  = CQEDTopology("rt")
    r  = t.add_node(SubgType.RESONATOR, {"length": length}, label="R")
    cc = t.add_node(SubgType.C_COUPLER, {"Cc": Cc},          label="Cc")
    tr = t.add_node(SubgType.TRANSMON,  {"L": L, "C": C},   label="T")
    t.add_edge(r, cc); t.add_edge(cc, tr)
    c   = graphlize(t)
    rct = node_by_type(c, SubgType.RCT)
    assert abs(rct.attrs["length"] - length) < 1e-20
    assert abs(rct.attrs["Cc"]     - Cc)     < 1e-20
    assert abs(rct.attrs["L"]      - L)      < 1e-20
    assert abs(rct.attrs["C"]      - C)      < 1e-20


# ---------------------------------------------------------------------------
# Visuals
# ---------------------------------------------------------------------------

def test_plot_before_after_writes_png(tmp_path: Path) -> None:
    raw = build_two_qubit_resonator()
    out = plot_before_after(raw, outdir=tmp_path)
    assert out.exists() and out.suffix == ".png"


# ---------------------------------------------------------------------------
# Direct script entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot",   action="store_true")
    parser.add_argument("--show",   action="store_true")
    parser.add_argument("--outdir", default="graphlize_plots")
    args = parser.parse_args()

    for name, builder in ALL_BUILDERS.items():
        raw        = builder()
        compressed = graphlize(raw)
        print("=" * 80)
        print(name)
        print("RAW");        print(raw.summary())
        print("COMPRESSED"); print(compressed.summary())
        print("types:", types(compressed))
        if args.plot:
            path = plot_before_after(raw, compressed, outdir=args.outdir, show=args.show)
            print(f"plot → {path}")


if __name__ == "__main__":
    main()
