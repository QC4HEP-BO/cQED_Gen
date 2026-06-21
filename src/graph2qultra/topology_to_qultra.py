"""
topology_to_qultra_general.py
=============================

Converter generale port-based da CQEDTopology primitiva a net qultra.

Assunzione fisica usata qui:
- TRANSMON    : nodo caldo unico, con C e J verso GND.
- RESONATOR   : risonatore lambda/4, nodo caldo unico + GND.
                Piu coupler possono collegarsi allo stesso nodo caldo.
- C_COUPLER   : capacita fra il nodo caldo dei due vicini.
- I_COUPLER   : CPW_coupler fra il nodo caldo del risonatore e una linea di
                feedline locale. Usa una sola resistenza di load sulla FEEDLINE.
- FEEDLINE    : nodo caldo unico + una resistenza verso GND.

Questo file deve stare nella root della repo, accanto a src/.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if os.path.isdir(SRC) and SRC not in sys.path:
    sys.path.insert(0, SRC)

try:
    from circuit2graph.definitions import SubgType
    from circuit2graph.topology import CQEDTopology
except ImportError as exc:
    raise ImportError(
        "Impossibile importare circuit2graph. Metti questo file nella root "
        "della repo, accanto a src/, oppure aggiungi repo/src al PYTHONPATH."
    ) from exc


GND = 0
DEFAULT_ROUT = 50.0
DEFAULT_L_IND = 100e-6   # m
DEFAULT_D_IND = 9.0      # um

SUPPORTED_TYPES = {
    SubgType.TRANSMON,
    SubgType.RESONATOR,
    SubgType.C_COUPLER,
    SubgType.I_COUPLER,
    SubgType.FEEDLINE,
}


@dataclass
class QElement:
    kind: str
    nodes: list[int]
    params: list[Any]
    comment: str = ""

    def __str__(self) -> str:
        if self.kind == "CPW_coupler":
            return _format_cpw_coupler(self)
        node_str = ", ".join(str(n) for n in self.nodes)
        param_str = ", ".join(str(p) for p in self.params)
        base = f"qu.{self.kind}({node_str}, {param_str})"
        return f"{base:<58}  # {self.comment}" if self.comment else base


@dataclass
class PrimitivePort:
    node_id: int
    qnode: int
    label: str
    subg_type: SubgType


class PortBuilder:
    def __init__(self):
        self.next_qnode = 1
        self.elements: list[QElement] = []
        self.hot: dict[int, PrimitivePort] = {}
        self._emitted_feedline_loads: set[int] = set()

    def new_qnode(self) -> int:
        out = self.next_qnode
        self.next_qnode += 1
        return out

    def add(self, element: QElement) -> None:
        self.elements.append(element)

    def set_hot(self, node, qnode: int) -> None:
        label = node.label or str(node.node_id)
        self.hot[node.node_id] = PrimitivePort(
            node_id=node.node_id,
            qnode=qnode,
            label=label,
            subg_type=node.subg_type,
        )

    def get_hot(self, node_id: int) -> int:
        if node_id not in self.hot:
            raise ValueError(f"Nodo {node_id} non ha un nodo qultra caldo assegnato.")
        return self.hot[node_id].qnode

    def mark_feedline_load_emitted(self, node_id: int) -> bool:
        if node_id in self._emitted_feedline_loads:
            return False
        self._emitted_feedline_loads.add(node_id)
        return True


def _attrs_for(node, attrs_by_label: dict | None) -> dict:
    attrs_by_label = attrs_by_label or {}
    label = node.label or str(node.node_id)
    return {**node.attrs, **attrs_by_label.get(label, {})}


def _label(node) -> str:
    return node.label or str(node.node_id)


def _make_adjacency(topology) -> dict[int, list[int]]:
    adj = {n.node_id: [] for n in topology._nodes}
    for a, b in topology._edges:
        adj[a].append(b)
        adj[b].append(a)
    return adj


def _other_neighbors(adj: dict[int, list[int]], node_id: int) -> list[int]:
    return list(adj[node_id])


def _require_degree(node, adj: dict[int, list[int]], degree: int) -> None:
    got = len(adj[node.node_id])
    if got != degree:
        raise ValueError(
            f"Nodo '{_label(node)}' di tipo {node.subg_type.name} ha {got} vicini; "
            f"attesi {degree}."
        )


def _format_cpw_coupler(el: QElement) -> str:
    nodes_str = str(el.nodes)
    gaps_str = str(el.params[0])
    widths_str = str(el.params[1])
    length_str = str(el.params[2])
    base = (
        "qu.CPW_coupler(\n"
        f"    {nodes_str},\n"
        f"    {gaps_str},\n"
        f"    {widths_str},\n"
        f"    {length_str},\n"
        ")"
    )
    return f"{base}  # {el.comment}" if el.comment else base


def topology_to_net(
    topology,
    attrs: dict | None = None,
    rout: float = DEFAULT_ROUT,
) -> list[QElement]:
    """Converte una CQEDTopology primitiva in elementi qultra."""
    attrs = attrs or {}
    builder = PortBuilder()

    nodes = {n.node_id: n for n in topology._nodes}
    adj = _make_adjacency(topology)

    unsupported = [n for n in topology._nodes if n.subg_type not in SUPPORTED_TYPES]
    if unsupported:
        bad = ", ".join(f"{_label(n)}:{n.subg_type.name}" for n in unsupported)
        raise NotImplementedError(
            "Questo converter generale supporta solo primitive "
            "TRANSMON, RESONATOR, C_COUPLER, I_COUPLER, FEEDLINE. "
            f"Non supportati: {bad}"
        )

    # Passo 1: alloca nodi caldi per componenti che hanno una porta fisica.
    # I coupler sono elementi di connessione e vengono emessi al passo 2.
    for node in topology._nodes:
        st = node.subg_type
        label = _label(node)
        node_attrs = _attrs_for(node, attrs)

        if st == SubgType.TRANSMON:
            q = builder.new_qnode()
            builder.set_hot(node, q)
            Cj = node_attrs.get("C", 100e-15)
            Lj = node_attrs.get("L", 10e-9)
            builder.add(QElement("C", [GND, q], [f"{Cj:.4e}"], comment=f"{label}: Cj"))
            builder.add(QElement("J", [GND, q], [f"{Lj:.4e}", 1], comment=f"{label}: Lj"))

        elif st == SubgType.RESONATOR:
            q = builder.new_qnode()
            builder.set_hot(node, q)
            length = node_attrs.get("length", 1e-2)
            builder.add(QElement("CPW", [q, GND], [f"{length:.4e}"], comment=f"{label}: lambda/4 l={length:.3e}m"))

        elif st == SubgType.FEEDLINE:
            q = builder.new_qnode()
            builder.set_hot(node, q)
            # La resistenza viene emessa ora. Per I_COUPLER si riusa questo stesso
            # nodo come un terminale della linea accoppiata: quindi resta una sola R.
            if builder.mark_feedline_load_emitted(node.node_id):
                builder.add(QElement("R", [q, GND], [f"{rout}"], comment=f"{label}: Rout={rout} Ohm"))

    # Passo 2: emetti i coupler usando i nodi caldi dei vicini.
    for node in topology._nodes:
        st = node.subg_type
        if st not in {SubgType.C_COUPLER, SubgType.I_COUPLER}:
            continue

        label = _label(node)
        node_attrs = _attrs_for(node, attrs)
        neighbors = _other_neighbors(adj, node.node_id)
        _require_degree(node, adj, 2)
        a_id, b_id = neighbors
        a = nodes[a_id]
        b = nodes[b_id]

        if st == SubgType.C_COUPLER:
            n_a = builder.get_hot(a_id)
            n_b = builder.get_hot(b_id)
            Cc = node_attrs.get("Cc", node_attrs.get("Cc_qr", node_attrs.get("Cc_rf", 1e-15)))
            builder.add(QElement("C", [n_a, n_b], [f"{Cc:.4e}"], comment=f"{label}: Cc"))

        elif st == SubgType.I_COUPLER:
            types = {a.subg_type, b.subg_type}
            if SubgType.RESONATOR not in types or SubgType.FEEDLINE not in types:
                raise ValueError(
                    f"I_COUPLER '{label}' deve collegare un RESONATOR e una FEEDLINE; "
                    f"trovati {a.subg_type.name} e {b.subg_type.name}."
                )

            r_node = a if a.subg_type == SubgType.RESONATOR else b
            f_node = a if a.subg_type == SubgType.FEEDLINE else b

            n_res = builder.get_hot(r_node.node_id)
            n_feed_load = builder.get_hot(f_node.node_id)
            n_feed_aux = builder.new_qnode()

            D = node_attrs.get("D", DEFAULT_D_IND)
            l = node_attrs.get("l", DEFAULT_L_IND)
            gaps = [9, D, 9]
            widths = [15, 15]
            builder.add(QElement(
                "CPW_coupler",
                [n_res, GND, n_feed_aux, n_feed_load],
                [gaps, widths, f"{l:.4e}"],
                comment=f"{label}: inductive D={D}um l={l:.3e}m",
            ))

    return builder.elements


def print_net(elements: list[QElement], topology_name: str = "") -> None:
    header = f"# qultra net - {topology_name}" if topology_name else "# qultra net"
    print(header)
    print("net = [")
    for el in elements:
        rendered = str(el)
        if el.kind == "CPW_coupler":
            for line in rendered.splitlines():
                print(f"    {line}")
            print(",")
        else:
            print(f"    {rendered},")
    print("]")


def net_to_string(elements: list[QElement], topology_name: str = "") -> str:
    lines = []
    header = f"# qultra net - {topology_name}" if topology_name else "# qultra net"
    lines.append(header)
    lines.append("net = [")
    for el in elements:
        rendered = str(el)
        if el.kind == "CPW_coupler":
            for line in rendered.splitlines():
                lines.append(f"    {line}")
            lines.append(",")
        else:
            lines.append(f"    {rendered},")
    lines.append("]")
    return "\n".join(lines)




def describe_topology(topology) -> None:
    """Stampa una descrizione leggibile della CQEDTopology prima della conversione."""
    node_by_id = {n.node_id: n for n in topology._nodes}
    adj = _make_adjacency(topology)

    print(f"# CQEDTopology - {topology.name}")
    print("nodes = [")
    for node in topology._nodes:
        label = _label(node)
        neigh = ", ".join(
            f"{node_by_id[nid].label or nid}:{node_by_id[nid].subg_type.name}"
            for nid in adj[node.node_id]
        )
        print(
            f"    id={node.node_id:<2} label={label:<12} "
            f"type={node.subg_type.name:<10} attrs={node.attrs} "
            f"neighbors=[{neigh}]"
        )
    print("]")
    print("edges = [")
    for a_id, b_id in topology._edges:
        a = node_by_id[a_id]
        b = node_by_id[b_id]
        print(
            f"    ({a_id}:{_label(a)}:{a.subg_type.name}) "
            f"-- ({b_id}:{_label(b)}:{b.subg_type.name})"
        )
    print("]")

# ---------------------------------------------------------------------------
# Test/demo topologie primitive
# ---------------------------------------------------------------------------

def _make_topology_qubit():
    t = CQEDTopology("Qubit")
    t.add_node(SubgType.TRANSMON, {"L": 10e-9, "C": 100e-15}, label="T")
    return t, {}


def _make_topology_two_qubit_capacitive():
    t = CQEDTopology("Two_qubit_capacitive")
    t1 = t.add_node(SubgType.TRANSMON, {"L": 10e-9, "C": 100e-15}, label="T1")
    cc = t.add_node(SubgType.C_COUPLER, {"Cc": 3e-15}, label="Cc")
    t2 = t.add_node(SubgType.TRANSMON, {"L": 12e-9, "C": 100e-15}, label="T2")
    t.add_edge(t1, cc)
    t.add_edge(cc, t2)
    return t, {}


def _make_topology_qubit_resonator_feedline_cap():
    t = CQEDTopology("QRF_cap")
    f = t.add_node(SubgType.FEEDLINE, {}, label="F")
    cc_rf = t.add_node(SubgType.C_COUPLER, {"Cc": 5e-15}, label="Cc_rf")
    r = t.add_node(SubgType.RESONATOR, {"length": 5e-3}, label="R")
    cc_qr = t.add_node(SubgType.C_COUPLER, {"Cc": 2e-15}, label="Cc_qr")
    tr = t.add_node(SubgType.TRANSMON, {"L": 10e-9, "C": 100e-15}, label="T")
    t.add_edge(f, cc_rf)
    t.add_edge(cc_rf, r)
    t.add_edge(r, cc_qr)
    t.add_edge(cc_qr, tr)
    return t, {}


def _make_topology_qubit_resonator_resonator():
    t = CQEDTopology("QRR")
    tr = t.add_node(SubgType.TRANSMON, {"L": 10e-9, "C": 100e-15}, label="T")
    cc_qr = t.add_node(SubgType.C_COUPLER, {"Cc": 2e-15}, label="Cc_qr")
    r1 = t.add_node(SubgType.RESONATOR, {"length": 4e-3}, label="R1")
    cc_rr = t.add_node(SubgType.C_COUPLER, {"Cc": 1e-15}, label="Cc_rr")
    r2 = t.add_node(SubgType.RESONATOR, {"length": 6e-3}, label="R2")
    t.add_edge(tr, cc_qr)
    t.add_edge(cc_qr, r1)
    t.add_edge(r1, cc_rr)
    t.add_edge(cc_rr, r2)
    return t, {}


def _make_topology_two_qubit_resonator():
    t = CQEDTopology("TwoQubit_Resonator")
    t1 = t.add_node(SubgType.TRANSMON, {"L": 10e-9, "C": 100e-15}, label="T1")
    cc_qr1 = t.add_node(SubgType.C_COUPLER, {"Cc": 2e-15}, label="Cc_qr1")
    r = t.add_node(SubgType.RESONATOR, {"length": 5e-3}, label="R")
    cc_qr2 = t.add_node(SubgType.C_COUPLER, {"Cc": 2e-15}, label="Cc_qr2")
    t2 = t.add_node(SubgType.TRANSMON, {"L": 12e-9, "C": 100e-15}, label="T2")
    t.add_edge(t1, cc_qr1)
    t.add_edge(cc_qr1, r)
    t.add_edge(r, cc_qr2)
    t.add_edge(cc_qr2, t2)
    return t, {}


def _make_topology_three_qubit_capacitive_star():
    t = CQEDTopology("ThreeQubit_Star")
    t1 = t.add_node(SubgType.TRANSMON, {"L": 10e-9, "C": 100e-15}, label="T1")
    t2 = t.add_node(SubgType.TRANSMON, {"L": 12e-9, "C": 100e-15}, label="T2")
    t3 = t.add_node(SubgType.TRANSMON, {"L": 14e-9, "C": 100e-15}, label="T3")
    cc12 = t.add_node(SubgType.C_COUPLER, {"Cc": 3e-15}, label="Cc12")
    cc13 = t.add_node(SubgType.C_COUPLER, {"Cc": 2e-15}, label="Cc13")
    cc23 = t.add_node(SubgType.C_COUPLER, {"Cc": 4e-15}, label="Cc23")
    t.add_edge(t1, cc12)
    t.add_edge(cc12, t2)
    t.add_edge(t1, cc13)
    t.add_edge(cc13, t3)
    t.add_edge(t2, cc23)
    t.add_edge(cc23, t3)
    return t, {}


def _make_topology_three_qubit_capacitive_line():
    t = CQEDTopology("ThreeQubit_Line")
    t1 = t.add_node(SubgType.TRANSMON, {"L": 10e-9, "C": 100e-15}, label="T1")
    cc12 = t.add_node(SubgType.C_COUPLER, {"Cc": 3e-15}, label="Cc12")
    t2 = t.add_node(SubgType.TRANSMON, {"L": 12e-9, "C": 100e-15}, label="T2")
    cc23 = t.add_node(SubgType.C_COUPLER, {"Cc": 2e-15}, label="Cc23")
    t3 = t.add_node(SubgType.TRANSMON, {"L": 14e-9, "C": 100e-15}, label="T3")
    t.add_edge(t1, cc12)
    t.add_edge(cc12, t2)
    t.add_edge(t2, cc23)
    t.add_edge(cc23, t3)
    return t, {}


def _make_topology_resonator_qubit_resonator():
    t = CQEDTopology("RQR")
    r1 = t.add_node(SubgType.RESONATOR, {"length": 4e-3}, label="R1")
    cc_qr1 = t.add_node(SubgType.C_COUPLER, {"Cc": 2e-15}, label="Cc_qr1")
    tr = t.add_node(SubgType.TRANSMON, {"L": 10e-9, "C": 100e-15}, label="T")
    cc_qr2 = t.add_node(SubgType.C_COUPLER, {"Cc": 2e-15}, label="Cc_qr2")
    r2 = t.add_node(SubgType.RESONATOR, {"length": 6e-3}, label="R2")
    t.add_edge(r1, cc_qr1)
    t.add_edge(cc_qr1, tr)
    t.add_edge(tr, cc_qr2)
    t.add_edge(cc_qr2, r2)
    return t, {}


def _make_topology_qubit_resonator_feedline_inductive():
    t = CQEDTopology("QRF_inductive")
    tr = t.add_node(SubgType.TRANSMON, {"L": 10e-9, "C": 100e-15}, label="T")
    cc_qr = t.add_node(SubgType.C_COUPLER, {"Cc": 2e-15}, label="Cc_qr")
    r = t.add_node(SubgType.RESONATOR, {"length": 5e-3}, label="R")
    ic = t.add_node(SubgType.I_COUPLER, {"D": 9.0, "l": 100e-6}, label="I_rf")
    f = t.add_node(SubgType.FEEDLINE, {}, label="F")
    t.add_edge(tr, cc_qr)
    t.add_edge(cc_qr, r)
    t.add_edge(r, ic)
    t.add_edge(ic, f)
    return t, {}


def _make_topology_four_qubit_common_bus():
    """Quattro transmon accoppiati capacitivamente allo stesso risonatore bus."""
    t = CQEDTopology("FourQubit_CommonBus")
    bus = t.add_node(SubgType.RESONATOR, {"length": 7.5e-3}, label="R_bus")
    for i, (L, Cc) in enumerate([
        (10e-9, 1.8e-15),
        (11e-9, 2.0e-15),
        (12e-9, 2.2e-15),
        (13e-9, 2.4e-15),
    ], start=1):
        q = t.add_node(SubgType.TRANSMON, {"L": L, "C": 100e-15}, label=f"T{i}")
        cc = t.add_node(SubgType.C_COUPLER, {"Cc": Cc}, label=f"Cc_q{i}")
        t.add_edge(q, cc)
        t.add_edge(cc, bus)
    return t, {}


def _make_topology_two_resonator_bus_four_qubits():
    """Due bus capacitivamente accoppiati; due qubit su ciascun bus."""
    t = CQEDTopology("TwoResonatorBus_FourQubits")
    r1 = t.add_node(SubgType.RESONATOR, {"length": 5.5e-3}, label="R1")
    r2 = t.add_node(SubgType.RESONATOR, {"length": 6.5e-3}, label="R2")
    cc_rr = t.add_node(SubgType.C_COUPLER, {"Cc": 0.8e-15}, label="Cc_R1R2")
    t.add_edge(r1, cc_rr)
    t.add_edge(cc_rr, r2)

    specs = [
        ("T1", 10e-9, 1.7e-15, r1),
        ("T2", 11e-9, 1.9e-15, r1),
        ("T3", 12e-9, 2.1e-15, r2),
        ("T4", 13e-9, 2.3e-15, r2),
    ]
    for label, L, Cc, resonator in specs:
        q = t.add_node(SubgType.TRANSMON, {"L": L, "C": 100e-15}, label=label)
        cc = t.add_node(SubgType.C_COUPLER, {"Cc": Cc}, label=f"Cc_{label}")
        t.add_edge(q, cc)
        t.add_edge(cc, resonator)
    return t, {}


def _make_topology_readout_chain_with_filter():
    """Qubit su risonatore readout, Purcell/filter resonator e feedline capacitiva."""
    t = CQEDTopology("ReadoutChain_WithFilter")
    q = t.add_node(SubgType.TRANSMON, {"L": 10e-9, "C": 95e-15}, label="T")
    c_qr = t.add_node(SubgType.C_COUPLER, {"Cc": 1.8e-15}, label="Cc_qr")
    r_read = t.add_node(SubgType.RESONATOR, {"length": 5.0e-3}, label="R_read")
    c_rr = t.add_node(SubgType.C_COUPLER, {"Cc": 0.7e-15}, label="Cc_read_filter")
    r_filter = t.add_node(SubgType.RESONATOR, {"length": 4.2e-3}, label="R_filter")
    c_rf = t.add_node(SubgType.C_COUPLER, {"Cc": 4.5e-15}, label="Cc_filter_feed")
    f = t.add_node(SubgType.FEEDLINE, {}, label="F")
    for a, b in [(q, c_qr), (c_qr, r_read), (r_read, c_rr), (c_rr, r_filter), (r_filter, c_rf), (c_rf, f)]:
        t.add_edge(a, b)
    return t, {}


def _make_topology_hybrid_cap_inductive_readout():
    """Due qubit sullo stesso risonatore, readout induttivo verso feedline."""
    t = CQEDTopology("HybridCapInductive_Readout")
    r = t.add_node(SubgType.RESONATOR, {"length": 5.8e-3}, label="R")
    for label, L, Cc in [("T1", 10e-9, 1.7e-15), ("T2", 12e-9, 2.1e-15)]:
        q = t.add_node(SubgType.TRANSMON, {"L": L, "C": 100e-15}, label=label)
        cc = t.add_node(SubgType.C_COUPLER, {"Cc": Cc}, label=f"Cc_{label}")
        t.add_edge(q, cc)
        t.add_edge(cc, r)
    ic = t.add_node(SubgType.I_COUPLER, {"D": 7.5, "l": 140e-6}, label="I_readout")
    f = t.add_node(SubgType.FEEDLINE, {}, label="F")
    t.add_edge(r, ic)
    t.add_edge(ic, f)
    return t, {}


def _make_topology_resonator_ring_three_qubits():
    """Tre risonatori in ring, ciascuno con un transmon agganciato."""
    t = CQEDTopology("ResonatorRing_ThreeQubits")
    r1 = t.add_node(SubgType.RESONATOR, {"length": 4.8e-3}, label="R1")
    r2 = t.add_node(SubgType.RESONATOR, {"length": 5.2e-3}, label="R2")
    r3 = t.add_node(SubgType.RESONATOR, {"length": 5.6e-3}, label="R3")
    for a, b, label, Cc in [
        (r1, r2, "Cc_R12", 0.9e-15),
        (r2, r3, "Cc_R23", 1.0e-15),
        (r3, r1, "Cc_R31", 1.1e-15),
    ]:
        cc = t.add_node(SubgType.C_COUPLER, {"Cc": Cc}, label=label)
        t.add_edge(a, cc)
        t.add_edge(cc, b)
    for i, (res, L, Cc) in enumerate([(r1, 10e-9, 1.6e-15), (r2, 11e-9, 1.8e-15), (r3, 12e-9, 2.0e-15)], start=1):
        q = t.add_node(SubgType.TRANSMON, {"L": L, "C": 100e-15}, label=f"T{i}")
        cc = t.add_node(SubgType.C_COUPLER, {"Cc": Cc}, label=f"Cc_T{i}")
        t.add_edge(q, cc)
        t.add_edge(cc, res)
    return t, {}


def _make_topology_ladder_six_qubits_two_buses():
    """Ladder: due bus resonator accoppiati e sei qubit distribuiti."""
    t = CQEDTopology("Ladder_SixQubits_TwoBuses")
    r_top = t.add_node(SubgType.RESONATOR, {"length": 6.0e-3}, label="R_top")
    r_bot = t.add_node(SubgType.RESONATOR, {"length": 6.3e-3}, label="R_bot")
    c_bus = t.add_node(SubgType.C_COUPLER, {"Cc": 0.6e-15}, label="Cc_bus")
    t.add_edge(r_top, c_bus)
    t.add_edge(c_bus, r_bot)

    for i in range(1, 7):
        res = r_top if i <= 3 else r_bot
        q = t.add_node(SubgType.TRANSMON, {"L": (9 + i) * 1e-9, "C": 100e-15}, label=f"T{i}")
        cc = t.add_node(SubgType.C_COUPLER, {"Cc": (1.4 + 0.15 * i) * 1e-15}, label=f"Cc_T{i}")
        t.add_edge(q, cc)
        t.add_edge(cc, res)
    return t, {}



def _make_topology_multiplexed_readout_three_qubits_one_feedline():
    """Tre qubit, tre resonator readout, una feedline capacitiva condivisa."""
    t = CQEDTopology("MultiplexedReadout_3Qubits_1Feedline")
    f = t.add_node(SubgType.FEEDLINE, {}, label="F_shared")
    for i, (L, r_len, Cqr, Crf) in enumerate([
        (10e-9, 4.8e-3, 1.5e-15, 4.0e-15),
        (11e-9, 5.2e-3, 1.7e-15, 4.3e-15),
        (12e-9, 5.6e-3, 1.9e-15, 4.6e-15),
    ], start=1):
        q = t.add_node(SubgType.TRANSMON, {"L": L, "C": 100e-15}, label=f"T{i}")
        c_qr = t.add_node(SubgType.C_COUPLER, {"Cc": Cqr}, label=f"Cc_T{i}_R{i}")
        r = t.add_node(SubgType.RESONATOR, {"length": r_len}, label=f"R_read{i}")
        c_rf = t.add_node(SubgType.C_COUPLER, {"Cc": Crf}, label=f"Cc_R{i}_F")
        t.add_edge(q, c_qr)
        t.add_edge(c_qr, r)
        t.add_edge(r, c_rf)
        t.add_edge(c_rf, f)
    return t, {}


def _make_topology_two_inductive_readouts_shared_feedline():
    """Due qubit con due resonator distinti, entrambi accoppiati induttivamente alla stessa feedline."""
    t = CQEDTopology("TwoInductiveReadouts_SharedFeedline")
    f = t.add_node(SubgType.FEEDLINE, {}, label="F")
    for i, (L, r_len, Cqr, D, lc) in enumerate([
        (10e-9, 5.0e-3, 1.6e-15, 7.0, 120e-6),
        (12e-9, 5.7e-3, 2.0e-15, 8.5, 150e-6),
    ], start=1):
        q = t.add_node(SubgType.TRANSMON, {"L": L, "C": 100e-15}, label=f"T{i}")
        c_qr = t.add_node(SubgType.C_COUPLER, {"Cc": Cqr}, label=f"Cc_T{i}_R{i}")
        r = t.add_node(SubgType.RESONATOR, {"length": r_len}, label=f"R{i}")
        i_coup = t.add_node(SubgType.I_COUPLER, {"D": D, "l": lc}, label=f"I_R{i}_F")
        t.add_edge(q, c_qr)
        t.add_edge(c_qr, r)
        t.add_edge(r, i_coup)
        t.add_edge(i_coup, f)
    return t, {}


def _make_topology_heavy_hex_like_seven_qubits():
    """Grafo solo transmon/coupler: sette qubit con connettivita tipo heavy-hex semplificata."""
    t = CQEDTopology("HeavyHexLike_SevenQubits")
    qubits = []
    for i in range(1, 8):
        qubits.append(t.add_node(
            SubgType.TRANSMON,
            {"L": (9.5 + 0.5 * i) * 1e-9, "C": 100e-15},
            label=f"T{i}",
        ))

    edges = [
        (1, 2, 2.0e-15),
        (2, 3, 2.1e-15),
        (2, 4, 1.8e-15),
        (4, 5, 2.2e-15),
        (4, 6, 1.9e-15),
        (6, 7, 2.3e-15),
    ]
    for a, b, Cc in edges:
        cc = t.add_node(SubgType.C_COUPLER, {"Cc": Cc}, label=f"Cc_T{a}_T{b}")
        t.add_edge(qubits[a - 1], cc)
        t.add_edge(cc, qubits[b - 1])
    return t, {}


def _make_topology_mixed_large_processor_block():
    """Blocco misto: bus centrale, due resonator readout capacitivi, un readout induttivo e feedline condivisa."""
    t = CQEDTopology("MixedLargeProcessorBlock")
    bus = t.add_node(SubgType.RESONATOR, {"length": 7.0e-3}, label="R_bus")
    f = t.add_node(SubgType.FEEDLINE, {}, label="F")

    # Quattro qubit sul bus centrale.
    for i, (L, Cc) in enumerate([(10e-9, 1.5e-15), (11e-9, 1.7e-15), (12e-9, 1.9e-15), (13e-9, 2.1e-15)], start=1):
        q = t.add_node(SubgType.TRANSMON, {"L": L, "C": 100e-15}, label=f"T_bus{i}")
        cc = t.add_node(SubgType.C_COUPLER, {"Cc": Cc}, label=f"Cc_bus_T{i}")
        t.add_edge(q, cc)
        t.add_edge(cc, bus)

    # Due readout capacitivi dalla stessa feedline.
    for i, (r_len, Cbr, Crf) in enumerate([(4.9e-3, 0.8e-15, 4.0e-15), (5.4e-3, 0.9e-15, 4.5e-15)], start=1):
        r = t.add_node(SubgType.RESONATOR, {"length": r_len}, label=f"R_read_cap{i}")
        c_bus_r = t.add_node(SubgType.C_COUPLER, {"Cc": Cbr}, label=f"Cc_bus_read{i}")
        c_rf = t.add_node(SubgType.C_COUPLER, {"Cc": Crf}, label=f"Cc_read{i}_F")
        t.add_edge(bus, c_bus_r)
        t.add_edge(c_bus_r, r)
        t.add_edge(r, c_rf)
        t.add_edge(c_rf, f)

    # Un ramo readout induttivo.
    r_ind = t.add_node(SubgType.RESONATOR, {"length": 5.9e-3}, label="R_read_ind")
    c_bus_ind = t.add_node(SubgType.C_COUPLER, {"Cc": 0.75e-15}, label="Cc_bus_ind")
    i_coup = t.add_node(SubgType.I_COUPLER, {"D": 8.0, "l": 130e-6}, label="I_ind_F")
    t.add_edge(bus, c_bus_ind)
    t.add_edge(c_bus_ind, r_ind)
    t.add_edge(r_ind, i_coup)
    t.add_edge(i_coup, f)
    return t, {}



TESTS = [
    ("Qubit (singolo transmon)", _make_topology_qubit),
    ("Two qubit capacitive (T1-Cc-T2)", _make_topology_two_qubit_capacitive),
    ("Qubit-Resonator-Feedline capacitive", _make_topology_qubit_resonator_feedline_cap),
    ("Qubit-Resonator-Resonator", _make_topology_qubit_resonator_resonator),
    ("Two qubit via resonatore mediatore", _make_topology_two_qubit_resonator),
    ("Three qubit star (all-to-all)", _make_topology_three_qubit_capacitive_star),
    ("Three qubit line (T1-T2-T3)", _make_topology_three_qubit_capacitive_line),
    ("Resonator-Qubit-Resonator", _make_topology_resonator_qubit_resonator),
    ("Qubit-Resonator-Feedline inductive", _make_topology_qubit_resonator_feedline_inductive),
    ("COMPLEX: four qubit common bus", _make_topology_four_qubit_common_bus),
    ("COMPLEX: two resonator bus with four qubits", _make_topology_two_resonator_bus_four_qubits),
    ("COMPLEX: readout chain with filter", _make_topology_readout_chain_with_filter),
    ("COMPLEX: hybrid capacitive + inductive readout", _make_topology_hybrid_cap_inductive_readout),
    ("COMPLEX: resonator ring with three qubits", _make_topology_resonator_ring_three_qubits),
    ("COMPLEX: ladder six qubits two buses", _make_topology_ladder_six_qubits_two_buses),
    ("COMPLEX: multiplexed readout 3 qubits 1 feedline", _make_topology_multiplexed_readout_three_qubits_one_feedline),
    ("COMPLEX: two inductive readouts shared feedline", _make_topology_two_inductive_readouts_shared_feedline),
    ("COMPLEX: heavy-hex-like seven qubits", _make_topology_heavy_hex_like_seven_qubits),
    ("COMPLEX: mixed large processor block", _make_topology_mixed_large_processor_block),
]


if __name__ == "__main__":
    sep = "=" * 60
    for name, factory in TESTS:
        print(f"\n{sep}")
        print(f"TOPOLOGIA: {name}")
        print(sep)
        try:
            topology, attrs = factory()
            describe_topology(topology)
            print()
            elements = topology_to_net(topology, attrs)
            print_net(elements, topology.name)
            kinds = {}
            for el in elements:
                kinds[el.kind] = kinds.get(el.kind, 0) + 1
            print(f"\n  -> Totale elementi: {len(elements)}  |  {kinds}")
        except Exception as exc:
            print(f"  ERRORE: {exc}")
