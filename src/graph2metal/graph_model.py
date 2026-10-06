"""Graph model, physics initial guesses, and synthesis configuration.

This is the input layer: it turns a NetworkX graph (or an equivalent JSON/
mapping description) into a normalized :class:`CircuitGraph`, and provides
the deliberately-approximate, explicitly-labelled physics functions that turn
lumped-element attributes (capacitance, inductance, target frequency) into
layout-zero geometric quantities (arm length, coupling gap, resonator
length). None of this is EM-calibrated; every function here is a replaceable
initial guess, and every value derived from it is recorded in the output
manifest so it can never be mistaken for a solved value.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping

import networkx as nx

C0 = 299_792_458.0


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class SynthesisConfig:
    """All geometry is expressed in micrometres.

    The defaults are deliberately conservative. They are layout-zero values,
    not fabrication-ready process rules and not a replacement for an EM solve.
    """

    chip_width_um: float = 12000.0
    chip_height_um: float = 10000.0
    chip_margin_um: float = 350.0
    max_chip_expansions: int = 4
    chip_expansion_factor: float = 1.20

    trace_width_um: float = 10.0
    trace_gap_um: float = 6.0
    component_clearance_um: float = 55.0
    route_clearance_um: float = 45.0
    coupling_window_margin_um: float = 45.0
    grid_step_um: float = 100.0
    fillet_um: float = 60.0

    transmon_reference_capacitance_fF: float = 82.0
    transmon_reference_arm_um: float = 150.0
    transmon_cross_width_um: float = 30.0
    transmon_cross_gap_um: float = 20.0
    transmon_min_arm_um: float = 115.0
    transmon_max_arm_um: float = 230.0

    qubit_pitch_um: float = 1350.0
    bus_row_offset_um: float = 420.0
    readout_row_pitch_um: float = 1450.0
    rigid_group_spacing_um: float = 180.0

    max_bus_qubits: int = 6
    max_bus_coupling_region_fraction: float = 0.40
    bus_open_lead_um: float = 180.0
    bus_post_coupling_lead_um: float = 650.0
    bus_meander_width_um: float = 2200.0
    bus_meander_height_um: float = 2600.0

    readout_open_segment_um: float = 280.0
    readout_open_lead_um: float = 1050.0
    readout_meander_width_um: float = 480.0
    readout_meander_height_um: float = 1100.0
    resonator_turn_pitch_um: float = 115.0

    feedline_coupling_min_um: float = 120.0
    # Native LaunchpadWirebond footprint (0.7.x defaults) is included in
    # planner collision checks. The inset leaves the full pad, pocket, and
    # route-clearance buffer inside the usable chip.
    feedline_launch_inset_um: float = 400.0
    launchpad_lead_length_um: float = 25.0
    launchpad_pad_width_um: float = 80.0
    launchpad_pad_height_um: float = 80.0
    launchpad_pad_gap_um: float = 58.0
    launchpad_taper_height_um: float = 122.0

    default_effective_epsilon: float = 6.0
    default_readout_frequency_GHz: float = 7.0
    default_bus_frequency_GHz: float = 6.0
    length_tolerance_fraction: float = 0.015
    coupling_gap_tolerance_um: float = 1.0
    geometry_area_tolerance_um2: float = 1e-3

    random_seed: int = 11
    strict_validation: bool = True

    def scaled(self, factor: float) -> "SynthesisConfig":
        data = asdict(self)
        data["chip_width_um"] *= factor
        data["chip_height_um"] *= factor
        data["max_chip_expansions"] = self.max_chip_expansions
        return SynthesisConfig(**data)


# --------------------------------------------------------------------------- #
# Graph model
# --------------------------------------------------------------------------- #


class NodeKind(str, Enum):
    TRANSMON = "TRANSMON"
    RESONATOR = "RESONATOR"
    FEEDLINE = "FEEDLINE"
    C_COUPLER = "C_COUPLER"
    I_COUPLER = "I_COUPLER"


class ResonatorRole(str, Enum):
    READOUT = "READOUT"
    COMMON_BUS = "COMMON_BUS"
    ISOLATED = "ISOLATED"


@dataclass(slots=True)
class CircuitNode:
    node_id: int
    kind: NodeKind
    label: str
    attrs: dict[str, Any] = field(default_factory=dict)
    role: ResonatorRole | None = None


@dataclass(slots=True)
class CircuitEdge:
    a: int
    b: int
    kind: str = "connection"
    coupling_fF: float | None = None
    absorbed_node_ids: tuple[int, ...] = ()
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[int, int]:
        return tuple(sorted((self.a, self.b)))


@dataclass
class CircuitGraph:
    name: str
    nodes: dict[int, CircuitNode]
    edges: list[CircuitEdge]
    skipped_nodes: list[CircuitNode] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def nx_graph(self) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(self.nodes)
        for edge in self.edges:
            graph.add_edge(edge.a, edge.b, edge=edge)
        return graph

    def neighbours(self, node_id: int) -> list[int]:
        return list(self.nx_graph().neighbors(node_id))

    def edge_between(self, a: int, b: int) -> CircuitEdge | None:
        key = tuple(sorted((a, b)))
        return next((e for e in self.edges if e.key == key), None)

    def nodes_of_kind(self, kind: NodeKind) -> list[CircuitNode]:
        return [node for node in self.nodes.values() if node.kind == kind]


_TYPE_ALIASES = {
    "Q": NodeKind.TRANSMON,
    "QUBIT": NodeKind.TRANSMON,
    "TRANSMON": NodeKind.TRANSMON,
    "R": NodeKind.RESONATOR,
    "RESONATOR": NodeKind.RESONATOR,
    "CPW": NodeKind.RESONATOR,
    "F": NodeKind.FEEDLINE,
    "FEED": NodeKind.FEEDLINE,
    "FEEDLINE": NodeKind.FEEDLINE,
    "C": NodeKind.C_COUPLER,
    "CC": NodeKind.C_COUPLER,
    "C_COUPLER": NodeKind.C_COUPLER,
    "CAPACITIVE_COUPLER": NodeKind.C_COUPLER,
    "I": NodeKind.I_COUPLER,
    "I_COUPLER": NodeKind.I_COUPLER,
    "INDUCTIVE_COUPLER": NodeKind.I_COUPLER,
}


def normalize_kind(value: Any) -> NodeKind:
    if isinstance(value, NodeKind):
        return value
    name = getattr(value, "name", value)
    key = str(name).strip().upper()
    if key in _TYPE_ALIASES:
        return _TYPE_ALIASES[key]
    raise ValueError(f"Unsupported circuit node type: {value!r}")


def _attrs(record: Mapping[str, Any] | Any) -> dict[str, Any]:
    if isinstance(record, Mapping):
        attrs = dict(record.get("attrs", {}))
        for key, value in record.items():
            if key not in {"id", "node_id", "type", "kind", "subg_type", "label", "attrs"}:
                attrs.setdefault(key, value)
        return attrs
    return dict(getattr(record, "attrs", {}) or {})


def _records_from_topology(data: Any) -> tuple[str, list[Any], list[tuple[int, int]]]:
    if isinstance(data, CircuitGraph):
        raise TypeError("CircuitGraph is already parsed")
    if isinstance(data, Mapping):
        name = str(data.get("name", "graph2metal_design"))
        return name, list(data.get("nodes", [])), [tuple(map(int, e[:2])) for e in data.get("edges", [])]
    if isinstance(data, nx.Graph):
        records = []
        for node_id, attrs in data.nodes(data=True):
            record = dict(attrs)
            record.setdefault("id", int(node_id))
            records.append(record)
        return str(data.graph.get("name", "networkx_design")), records, [(int(a), int(b)) for a, b in data.edges]
    nodes = list(getattr(data, "_nodes", getattr(data, "nodes", [])))
    edges = list(getattr(data, "_edges", getattr(data, "edges", [])))
    return str(getattr(data, "name", "topology_design")), nodes, [(int(a), int(b)) for a, b in edges]


def parse_graph(data: Any, *, skip_inductive: bool = True) -> CircuitGraph:
    """Parse mappings, NetworkX graphs, or the earlier CQEDTopology objects.

    Degree-two capacitive coupler nodes are absorbed into edge metadata. An
    inductive coupler is intentionally not translated to metal geometry; when
    ``skip_inductive`` is true it and its incident edges are recorded as
    skipped instead of being silently discarded.
    """
    if isinstance(data, CircuitGraph):
        return data
    name, raw_nodes, raw_edges = _records_from_topology(data)
    nodes: dict[int, CircuitNode] = {}
    for index, record in enumerate(raw_nodes):
        if isinstance(record, Mapping):
            node_id = int(record.get("id", record.get("node_id", index)))
            kind_raw = record.get("type", record.get("kind", record.get("subg_type")))
            label = str(record.get("label", f"N{node_id}"))
        else:
            node_id = int(getattr(record, "node_id", index))
            kind_raw = getattr(record, "subg_type", getattr(record, "kind", None))
            label = str(getattr(record, "label", f"N{node_id}"))
        nodes[node_id] = CircuitNode(node_id, normalize_kind(kind_raw), label, _attrs(record))

    adjacency: dict[int, list[int]] = {node_id: [] for node_id in nodes}
    for a, b in raw_edges:
        if a not in nodes or b not in nodes:
            raise ValueError(f"Edge ({a}, {b}) references a missing node")
        adjacency[a].append(b)
        adjacency[b].append(a)

    skipped: list[CircuitNode] = []
    notes: list[str] = []
    removed: set[int] = set()
    edges: list[CircuitEdge] = []

    for node in nodes.values():
        if node.kind == NodeKind.I_COUPLER:
            if not skip_inductive:
                raise NotImplementedError(f"Inductive coupler node {node.node_id} is not implemented")
            removed.add(node.node_id)
            skipped.append(node)
            notes.append(f"Skipped inductive coupler node {node.node_id}; its incident edges are not realized.")

    absorbed_edge_keys: set[tuple[int, int]] = set()
    for node in nodes.values():
        if node.kind != NodeKind.C_COUPLER or node.node_id in removed:
            continue
        neighbours = [n for n in adjacency[node.node_id] if n not in removed]
        if len(neighbours) != 2:
            notes.append(
                f"Capacitive coupler node {node.node_id} has degree {len(neighbours)} and cannot be safely absorbed."
            )
            continue
        a, b = neighbours
        removed.add(node.node_id)
        cc_fF = float(node.attrs.get("Cc", node.attrs.get("coupling_capacitance", 0.0))) * 1e15
        kinds = {nodes[a].kind, nodes[b].kind}
        if kinds == {NodeKind.TRANSMON}:
            edge_kind = "transmon_transmon_capacitive"
        elif kinds == {NodeKind.TRANSMON, NodeKind.RESONATOR}:
            edge_kind = "transmon_resonator_capacitive"
        elif kinds == {NodeKind.RESONATOR, NodeKind.FEEDLINE}:
            edge_kind = "resonator_feedline_capacitive"
        elif kinds == {NodeKind.RESONATOR}:
            edge_kind = "resonator_resonator_capacitive"
        else:
            edge_kind = "capacitive"
        edges.append(CircuitEdge(a, b, edge_kind, cc_fF, (node.node_id,), dict(node.attrs)))
        absorbed_edge_keys.add(tuple(sorted((a, node.node_id))))
        absorbed_edge_keys.add(tuple(sorted((b, node.node_id))))

    for a, b in raw_edges:
        key = tuple(sorted((a, b)))
        if key in absorbed_edge_keys or a in removed or b in removed:
            continue
        edges.append(CircuitEdge(a, b, "connection", None))

    parsed_nodes = {node_id: node for node_id, node in nodes.items() if node_id not in removed}
    graph = CircuitGraph(name, parsed_nodes, edges, skipped, notes)
    classify_resonators(graph)
    return graph


def classify_resonators(graph: CircuitGraph) -> None:
    nxg = graph.nx_graph()
    for resonator in graph.nodes_of_kind(NodeKind.RESONATOR):
        neighbours = [graph.nodes[n] for n in nxg.neighbors(resonator.node_id)]
        n_transmons = sum(n.kind == NodeKind.TRANSMON for n in neighbours)
        n_feedlines = sum(n.kind == NodeKind.FEEDLINE for n in neighbours)
        explicit_role = str(resonator.attrs.get("role", resonator.attrs.get("physical_role", ""))).upper()
        labelled_bus = "BUS" in resonator.label.upper()
        if n_transmons >= 2 or explicit_role in {"BUS", "COMMON_BUS"} or labelled_bus:
            resonator.role = ResonatorRole.COMMON_BUS
        elif n_transmons == 1 and n_feedlines >= 1:
            resonator.role = ResonatorRole.READOUT
        elif n_transmons == 1:
            # A single-qubit lambda/4 resonator is treated as readout even if
            # the feedline is absent from an early-stage graph.
            resonator.role = ResonatorRole.READOUT
        else:
            resonator.role = ResonatorRole.ISOLATED


def connected_edges(graph: CircuitGraph, node_id: int, *, kind: str | None = None) -> list[CircuitEdge]:
    result = [edge for edge in graph.edges if node_id in (edge.a, edge.b)]
    if kind is not None:
        result = [edge for edge in result if edge.kind == kind]
    return result


def other(edge: CircuitEdge, node_id: int) -> int:
    if edge.a == node_id:
        return edge.b
    if edge.b == node_id:
        return edge.a
    raise ValueError(f"Node {node_id} is not incident to edge {edge}")


def edge_map(edges: Iterable[CircuitEdge]) -> dict[tuple[int, int], CircuitEdge]:
    return {edge.key: edge for edge in edges}


# --------------------------------------------------------------------------- #
# Physics initial guesses
# --------------------------------------------------------------------------- #


def value_in_um(value: Any, *, assume_si_metres_below: float = 0.1) -> float | None:
    """Convert common numeric/string length representations to micrometres."""
    if value is None:
        return None
    if isinstance(value, str):
        raw = value.strip().lower().replace("µ", "u")
        units = {"nm": 1e-3, "um": 1.0, "mm": 1e3, "cm": 1e4, "m": 1e6}
        for unit, factor in units.items():
            if raw.endswith(unit):
                return float(raw[: -len(unit)].strip()) * factor
        return float(raw)
    number = float(value)
    if abs(number) < assume_si_metres_below:
        return number * 1e6
    return number


def capacitance_fF(attrs: dict[str, Any]) -> float | None:
    for key in ("C", "capacitance", "capacitance_F"):
        if key in attrs and attrs[key] is not None:
            value = float(attrs[key])
            return value * 1e15 if abs(value) < 1e-6 else value
    for key in ("C_fF", "capacitance_fF"):
        if key in attrs and attrs[key] is not None:
            return float(attrs[key])
    return None


def inductance_nH(attrs: dict[str, Any]) -> float | None:
    for key in ("L", "Lj", "inductance", "inductance_H"):
        if key in attrs and attrs[key] is not None:
            value = float(attrs[key])
            return value * 1e9 if abs(value) < 1e-3 else value
    for key in ("L_nH", "Lj_nH", "inductance_nH"):
        if key in attrs and attrs[key] is not None:
            return float(attrs[key])
    return None


def target_frequency_GHz(attrs: dict[str, Any], default: float) -> float:
    for key in ("target_frequency_GHz", "frequency_GHz", "_f_r_GHz", "f_GHz"):
        if key in attrs and attrs[key] is not None:
            return float(attrs[key])
    return default


def quarter_wave_length_um(frequency_GHz: float, epsilon_eff: float) -> float:
    return C0 / (4.0 * frequency_GHz * 1e9 * math.sqrt(epsilon_eff)) * 1e6


def resonator_length_um(attrs: dict[str, Any], *, default_frequency_GHz: float, config: SynthesisConfig) -> float:
    for key in ("length", "total_length", "length_um", "target_length"):
        if key in attrs and attrs[key] not in (None, 0, 0.0):
            converted = value_in_um(attrs[key])
            if converted and converted > 0:
                return converted
    frequency = target_frequency_GHz(attrs, default_frequency_GHz)
    epsilon_eff = float(attrs.get("epsilon_eff", config.default_effective_epsilon))
    return quarter_wave_length_um(frequency, epsilon_eff)


@dataclass(frozen=True, slots=True)
class TransmonDimensions:
    arm_length_um: float
    cross_width_um: float
    capacitance_fF: float
    inductance_nH: float | None


def transmon_dimensions(attrs: dict[str, Any], config: SynthesisConfig) -> TransmonDimensions:
    cap = capacitance_fF(attrs) or config.transmon_reference_capacitance_fF
    explicit_arm = value_in_um(attrs.get("cross_arm_length_um"))
    if explicit_arm is not None:
        arm = float(explicit_arm)
    else:
        # Layout-zero proxy: area scales with C, so a linear dimension scales
        # approximately with sqrt(C). An explicit graph attribute may override the arm length.
        scale = math.sqrt(max(cap, 5.0) / config.transmon_reference_capacitance_fF)
        arm = config.transmon_reference_arm_um * scale
    arm = min(config.transmon_max_arm_um, max(config.transmon_min_arm_um, arm))
    return TransmonDimensions(
        arm_length_um=arm,
        cross_width_um=config.transmon_cross_width_um,
        capacitance_fF=cap,
        inductance_nH=inductance_nH(attrs),
    )


def coupling_gap_um(
    coupling_fF: float | None,
    interface: str,
    explicit_gap_um: float | None = None,
) -> float:
    """Return the capacitive gap used by the physical synthesizer.

    ``explicit_gap_um`` is an optional graph-level override. When absent, graph2metal uses
    its bounded monotonic layout-zero estimate from the lumped coupling.
    """
    if explicit_gap_um is not None:
        return max(2.0, float(explicit_gap_um))
    cc = 5.0 if coupling_fF is None else max(0.2, float(coupling_fF))
    if interface == "transmon_transmon_capacitive":
        return min(28.0, max(4.0, 24.0 - 1.25 * cc))
    if interface == "resonator_feedline_capacitive":
        return min(24.0, max(3.0, 18.0 - 2.2 * cc))
    return min(30.0, max(4.0, 26.0 - 1.8 * cc))


def direct_pad_length_um(coupling_fF: float | None) -> float:
    cc = 4.0 if coupling_fF is None else max(0.2, min(25.0, float(coupling_fF)))
    return 55.0 + 7.0 * cc


def direct_pad_width_um(coupling_fF: float | None) -> float:
    cc = 4.0 if coupling_fF is None else max(0.2, min(25.0, float(coupling_fF)))
    return 24.0 + 0.8 * cc


def feedline_parallel_length_um(
    coupling_fF: float | None,
    config: SynthesisConfig,
    explicit_length_um: float | None = None,
) -> float:
    if explicit_length_um is not None:
        return max(config.feedline_coupling_min_um, float(explicit_length_um))
    cc = 2.5 if coupling_fF is None else max(0.2, min(12.0, float(coupling_fF)))
    return max(config.feedline_coupling_min_um, 95.0 + 22.0 * cc)


def edge_gap_um(edge: CircuitEdge) -> float:
    """Gap helper that honors explicit overrides carried by a coupler node."""
    value = edge.attrs.get("gap_um", edge.attrs.get("target_gap_um"))
    return coupling_gap_um(edge.coupling_fF, edge.kind, value)


def edge_parallel_length_um(edge: CircuitEdge, config: SynthesisConfig) -> float:
    """Parallel-coupling helper that honors explicit graph overrides."""
    value = edge.attrs.get("parallel_length_um")
    return feedline_parallel_length_um(edge.coupling_fF, config, value)


# ---- Stable graph I/O -----------------------------------------------------

import importlib
import json
import pickle
from pathlib import Path

class GraphInputError(ValueError):
    """Raised when a graph input cannot be interpreted."""


def ensure_networkx_graph(graph_like: Any) -> nx.Graph:
    if isinstance(graph_like, nx.Graph):
        return graph_like
    if isinstance(graph_like, Mapping):
        return payload_to_graph(graph_like)
    raise GraphInputError(f"Expected NetworkX Graph or graph mapping, got {type(graph_like).__name__}.")


def _json_compatible(value: Any) -> Any:
    """Convert enums/numpy-like scalars and containers to JSON-safe values."""
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_compatible(item) for item in value]
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return _json_compatible(value.item())
        except Exception:
            pass
    name = getattr(value, "name", None)
    if name is not None and value.__class__.__module__ != "builtins":
        return str(name)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _kind_name(value: Any) -> str:
    """Return a graph2metal-compatible semantic type name."""
    if isinstance(value, NodeKind):
        return value.value
    name = getattr(value, "name", value)
    return str(name).strip().upper()


def graph_to_payload(graph: Any) -> dict[str, Any]:
    """Serialize NetworkX, mappings, CircuitGraph, or cQED_Gen CQEDTopology.

    Supporting ``CQEDTopology`` directly is important for the integrated
    Hamiltonian -> G-VAE -> graph2metal pipeline: no intermediate JSON is
    required to pass the best expanded primitive graph into layout synthesis.
    """
    if isinstance(graph, CircuitGraph):
        nodes = [
            {
                "id": int(node.node_id),
                "type": node.kind.value,
                "label": node.label,
                "attrs": _json_compatible(node.attrs),
                **({"role": node.role.value} if node.role is not None else {}),
            }
            for node in graph.nodes.values()
        ]
        edges = [
            [
                int(edge.a),
                int(edge.b),
                {
                    "kind": edge.kind,
                    "coupling_fF": edge.coupling_fF,
                    "absorbed_node_ids": list(edge.absorbed_node_ids),
                    **_json_compatible(edge.attrs),
                },
            ]
            for edge in graph.edges
        ]
        return {
            "schema_version": "graph2metal-source-graph-1.1",
            "name": graph.name,
            "nodes": nodes,
            "edges": edges,
        }

    if isinstance(graph, Mapping):
        graph = payload_to_graph(graph)

    if isinstance(graph, nx.Graph):
        nodes = []
        for node_id, attrs in graph.nodes(data=True):
            raw = dict(attrs)
            kind_raw = raw.pop("type", raw.pop("kind", raw.pop("subg_type", None)))
            record: dict[str, Any] = {"id": int(node_id)}
            if kind_raw is not None:
                record["type"] = _kind_name(kind_raw)
            record.update(_json_compatible(raw))
            nodes.append(record)
        edges = []
        for a, b, attrs in graph.edges(data=True):
            record: list[Any] = [int(a), int(b)]
            if attrs:
                record.append(_json_compatible(dict(attrs)))
            edges.append(record)
        return {
            "schema_version": "graph2metal-source-graph-1.1",
            "name": str(graph.graph.get("name", "graph2metal_design")),
            "nodes": nodes,
            "edges": edges,
        }

    # cQED_Gen's CQEDTopology and compatible topology objects expose
    # ``_nodes`` and ``_edges``.  Serialize the physical primitive attributes
    # and semantic SubgType names explicitly so the JSON is reloadable.
    name, raw_nodes, raw_edges = _records_from_topology(graph)
    if not raw_nodes and not hasattr(graph, "_nodes") and not hasattr(graph, "nodes"):
        raise GraphInputError(
            f"Expected NetworkX Graph, graph mapping, CircuitGraph, or topology object; "
            f"got {type(graph).__name__}."
        )
    nodes = []
    for index, raw in enumerate(raw_nodes):
        if isinstance(raw, Mapping):
            node_id = int(raw.get("id", raw.get("node_id", index)))
            kind_raw = raw.get("type", raw.get("kind", raw.get("subg_type")))
            label = raw.get("label", f"N{node_id}")
            attrs = _attrs(raw)
        else:
            node_id = int(getattr(raw, "node_id", index))
            kind_raw = getattr(raw, "subg_type", getattr(raw, "kind", None))
            label = getattr(raw, "label", f"N{node_id}")
            attrs = _attrs(raw)
        nodes.append({
            "id": node_id,
            "type": _kind_name(kind_raw),
            "label": None if label is None else str(label),
            "attrs": _json_compatible(attrs),
        })
    return {
        "schema_version": "graph2metal-source-graph-1.1",
        "name": name,
        "nodes": nodes,
        "edges": [[int(a), int(b)] for a, b in raw_edges],
    }


def payload_to_graph(payload: Mapping[str, Any]) -> nx.Graph:
    graph = nx.Graph(name=str(payload.get("name", "graph2metal_design")))
    for index, raw in enumerate(payload.get("nodes", [])):
        if not isinstance(raw, Mapping):
            raise GraphInputError(f"Node record {index} is not an object.")
        attrs = dict(raw)
        node_id = int(attrs.pop("id", attrs.pop("node_id", index)))
        graph.add_node(node_id, **attrs)
    for index, raw in enumerate(payload.get("edges", [])):
        if not isinstance(raw, (list, tuple)) or len(raw) < 2:
            raise GraphInputError(f"Edge record {index} must contain at least two node IDs.")
        attrs = dict(raw[2]) if len(raw) >= 3 and isinstance(raw[2], Mapping) else {}
        graph.add_edge(int(raw[0]), int(raw[1]), **attrs)
    return graph


def save_graph(graph: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(graph_to_payload(graph), indent=2), encoding="utf-8")
    return path


def _load_factory(spec: str, kwargs: Mapping[str, Any] | None = None) -> nx.Graph:
    try:
        module_name, function_name = spec.split(":", 1)
    except ValueError as exc:
        raise GraphInputError("Graph factory must use module:function syntax.") from exc
    module = importlib.import_module(module_name)
    function = getattr(module, function_name, None)
    if not callable(function):
        raise GraphInputError(f"Graph factory {spec!r} is not callable.")
    graph = function(**dict(kwargs or {}))
    try:
        return ensure_networkx_graph(graph)
    except GraphInputError as exc:
        raise GraphInputError(f"Graph factory {spec!r} returned an unsupported object: {exc}") from exc


def load_graph(
    path: str | Path | None = None,
    *,
    factory: str | None = None,
    factory_kwargs: Mapping[str, Any] | None = None,
) -> nx.Graph:
    """Load JSON/pickle graph input or call a Python graph factory.

    The Python-factory path is the preferred API for direct cQED_Gen use.
    Pickle is accepted only for trusted local files.
    """
    if factory:
        return _load_factory(factory, factory_kwargs)
    if path is None:
        raise GraphInputError("Provide either a graph path or a graph factory.")
    path = Path(path)
    if not path.exists():
        raise GraphInputError(f"Graph file not found: {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise GraphInputError("Graph JSON top level must be an object.")
        return payload_to_graph(payload)
    if path.suffix.lower() in {".gpickle", ".pickle", ".pkl"}:
        with path.open("rb") as handle:
            graph = pickle.load(handle)  # noqa: S301 - explicitly documented trusted-local input.
        if not isinstance(graph, nx.Graph):
            raise GraphInputError(f"Pickle contains {type(graph).__name__}, not NetworkX Graph.")
        return graph
    raise GraphInputError(f"Unsupported graph format {path.suffix!r}; use .json, .gpickle, .pickle or .pkl.")
