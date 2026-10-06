"""Translate a normalized circuit graph into a validated physical layout plan."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Iterable, Mapping

import networkx as nx

from .graph_model import (
    CircuitEdge, CircuitGraph, NodeKind, ResonatorRole, SynthesisConfig,
    coupling_gap_um, direct_pad_length_um, edge_gap_um, parse_graph,
    transmon_dimensions,
)
from .elements import (
    LayoutPlan, Violation, RoutingError, LengthAllocationError,
    SynthesisError, build_elements, chip_polygon,
)

@dataclass(slots=True)
class RigidGroup:
    group_id: str
    members: tuple[int, ...]
    local_positions_um: dict[int, tuple[float, float]]
    direct_edges: tuple[CircuitEdge, ...] = ()
    width_um: float = 0.0
    height_um: float = 0.0


@dataclass(slots=True)
class PlacementSolution:
    transmon_positions: dict[int, tuple[float, float]]
    transmon_rotations_deg: dict[int, float]
    groups: dict[str, RigidGroup]
    node_to_group: dict[int, str]
    bus_id: int | None
    bus_order: list[int]
    bus_side: dict[int, int]
    readout_by_transmon: dict[int, int]
    feedline_by_readout: dict[int, int]
    port_assignments: dict[int, dict[str, str]] = field(default_factory=dict)


def direct_transmon_edges(graph: CircuitGraph) -> list[CircuitEdge]:
    return [
        edge
        for edge in graph.edges
        if edge.kind == "transmon_transmon_capacitive"
        and graph.nodes[edge.a].kind == NodeKind.TRANSMON
        and graph.nodes[edge.b].kind == NodeKind.TRANSMON
    ]


def _edge_separation(graph: CircuitGraph, edge: CircuitEdge, config: SynthesisConfig) -> float:
    a_dim = transmon_dimensions(graph.nodes[edge.a].attrs, config)
    b_dim = transmon_dimensions(graph.nodes[edge.b].attrs, config)
    return (
        a_dim.arm_length_um
        + direct_pad_length_um(edge.coupling_fF)
        + edge_gap_um(edge)
        + direct_pad_length_um(edge.coupling_fF)
        + b_dim.arm_length_um
    )


def build_rigid_groups(graph: CircuitGraph, config: SynthesisConfig) -> tuple[dict[str, RigidGroup], dict[int, str]]:
    transmons = sorted(node.node_id for node in graph.nodes_of_kind(NodeKind.TRANSMON))
    coupling_graph = nx.Graph()
    coupling_graph.add_nodes_from(transmons)
    edges = direct_transmon_edges(graph)
    for edge in edges:
        coupling_graph.add_edge(edge.a, edge.b, edge=edge)

    groups: dict[str, RigidGroup] = {}
    node_to_group: dict[int, str] = {}
    for index, component in enumerate(nx.connected_components(coupling_graph)):
        members = tuple(sorted(component))
        group_id = f"rigid_qgroup_{index}"
        subgraph = coupling_graph.subgraph(members)
        direct = tuple(data["edge"] for _, _, data in subgraph.edges(data=True))
        local: dict[int, tuple[float, float]] = {}

        if len(members) == 1:
            local[members[0]] = (0.0, 0.0)
        elif nx.is_connected(subgraph) and subgraph.number_of_edges() == len(members) - 1 and max(dict(subgraph.degree()).values()) <= 2:
            endpoints = sorted([node for node in members if subgraph.degree(node) == 1])
            order = list(nx.shortest_path(subgraph, endpoints[0], endpoints[-1]))
            cursor = 0.0
            local[order[0]] = (0.0, 0.0)
            for previous, current in zip(order, order[1:]):
                edge = subgraph.edges[previous, current]["edge"]
                cursor += _edge_separation(graph, edge, config)
                local[current] = (cursor, 0.0)
            center = (min(x for x, _ in local.values()) + max(x for x, _ in local.values())) / 2.0
            local = {node: (x - center, y) for node, (x, y) in local.items()}
        else:
            # General direct-coupling clusters are uncommon in the first
            # scope. A deterministic spring embedding is frozen into a rigid
            # group, then scaled so every directly coupled pair has at least
            # its requested separation.
            raw = nx.spring_layout(subgraph, seed=config.random_seed, scale=1.0)
            minimum_scale = config.qubit_pitch_um
            for edge in direct:
                dx = raw[edge.a][0] - raw[edge.b][0]
                dy = raw[edge.a][1] - raw[edge.b][1]
                distance = max((dx * dx + dy * dy) ** 0.5, 1e-6)
                minimum_scale = max(minimum_scale, _edge_separation(graph, edge, config) / distance)
            local = {node: (float(raw[node][0] * minimum_scale), float(raw[node][1] * minimum_scale)) for node in members}

        half_widths = []
        half_heights = []
        direct_members = {node for edge in direct for node in (edge.a, edge.b)}
        for node_id, (x, y) in local.items():
            dims = transmon_dimensions(graph.nodes[node_id].attrs, config)
            extra = direct_pad_length_um(8.0) if node_id in direct_members else 0.0
            reach = dims.arm_length_um + extra + config.component_clearance_um
            half_widths.append(abs(x) + reach)
            half_heights.append(abs(y) + reach)
        group = RigidGroup(
            group_id=group_id,
            members=members,
            local_positions_um=local,
            direct_edges=direct,
            width_um=2.0 * max(half_widths, default=config.qubit_pitch_um / 2.0),
            height_um=2.0 * max(half_heights, default=config.qubit_pitch_um / 2.0),
        )
        groups[group_id] = group
        for node_id in members:
            node_to_group[node_id] = group_id
    return groups, node_to_group


def topology_roles(graph: CircuitGraph) -> tuple[int | None, dict[int, int], dict[int, int]]:
    buses = [node.node_id for node in graph.nodes_of_kind(NodeKind.RESONATOR) if node.role == ResonatorRole.COMMON_BUS]
    if len(buses) > 1:
        raise NotImplementedError("The first implementation supports one shared lambda/4 bus per connected design")
    bus_id = buses[0] if buses else None
    nxg = graph.nx_graph()
    readout_by_transmon: dict[int, int] = {}
    feedline_by_readout: dict[int, int] = {}
    for resonator in graph.nodes_of_kind(NodeKind.RESONATOR):
        if resonator.role != ResonatorRole.READOUT:
            continue
        neighbours = list(nxg.neighbors(resonator.node_id))
        transmons = [n for n in neighbours if graph.nodes[n].kind == NodeKind.TRANSMON]
        feedlines = [n for n in neighbours if graph.nodes[n].kind == NodeKind.FEEDLINE]
        if transmons:
            readout_by_transmon[transmons[0]] = resonator.node_id
        if feedlines:
            feedline_by_readout[resonator.node_id] = feedlines[0]
    return bus_id, readout_by_transmon, feedline_by_readout


def _group_sequence(groups: dict[str, RigidGroup], preferred_nodes: Iterable[int], node_to_group: dict[int, str]) -> list[str]:
    sequence: list[str] = []
    for node_id in preferred_nodes:
        group_id = node_to_group[node_id]
        if group_id not in sequence:
            sequence.append(group_id)
    for group_id in sorted(groups):
        if group_id not in sequence:
            sequence.append(group_id)
    return sequence


def solve_initial_placement(graph: CircuitGraph, config: SynthesisConfig) -> PlacementSolution:
    groups, node_to_group = build_rigid_groups(graph, config)
    bus_id, readout_by_transmon, feedline_by_readout = topology_roles(graph)
    transmons = sorted(node.node_id for node in graph.nodes_of_kind(NodeKind.TRANSMON))

    if bus_id is not None:
        bus_neighbours = [
            node
            for node in graph.neighbours(bus_id)
            if graph.nodes[node].kind == NodeKind.TRANSMON
        ]
        if len(bus_neighbours) > config.max_bus_qubits:
            raise ValueError(
                f"Shared bus has {len(bus_neighbours)} qubits; configured maximum is {config.max_bus_qubits}."
            )
        preferred = sorted(bus_neighbours)
    else:
        preferred = transmons

    sequence = _group_sequence(groups, preferred, node_to_group)
    group_centers: dict[str, tuple[float, float]] = {}

    if bus_id is not None:
        # First implementation: all bus-coupled transmons sit above one
        # horizontal open-end coupling corridor. Their south arms face the
        # bus; readout resonators use the opposite (north) arms. This keeps
        # the bus/meander below the qubits and all readout/feedline geometry
        # above them, greatly reducing routing ambiguity for N <= 6.
        widths = [max(groups[group_id].width_um, config.qubit_pitch_um * 0.55) for group_id in sequence]
        total_width = sum(widths) + config.rigid_group_spacing_um * max(0, len(widths) - 1)
        cursor = -total_width / 2.0
        for group_id, width in zip(sequence, widths):
            center_x = cursor + width / 2.0
            group = groups[group_id]
            required_y = 0.0
            for node_id in group.members:
                edge = graph.edge_between(node_id, bus_id)
                cc = edge.coupling_fF if edge is not None else None
                dims = transmon_dimensions(graph.nodes[node_id].attrs, config)
                local_y = group.local_positions_um[node_id][1]
                required_y = max(
                    required_y,
                    -local_y
                    + dims.arm_length_um
                    + coupling_gap_um(cc, "transmon_resonator_capacitive")
                    + config.trace_width_um / 2.0,
                )
            group_centers[group_id] = (center_x, max(required_y, config.bus_row_offset_um))
            cursor += width + config.rigid_group_spacing_um
    else:
        # Without a common bus, put all transmons on one row below a possible
        # shared readout feedline. This yields a clean multiplexed-readout
        # template and still keeps direct-coupled groups rigid.
        widths = [max(groups[group_id].width_um, config.qubit_pitch_um * 0.80) for group_id in sequence]
        total_width = sum(widths) + config.rigid_group_spacing_um * max(0, len(widths) - 1)
        cursor = -total_width / 2.0
        for group_id, width in zip(sequence, widths):
            group_centers[group_id] = (cursor + width / 2.0, -900.0)
            cursor += width + config.rigid_group_spacing_um

    positions: dict[int, tuple[float, float]] = {}
    rotations: dict[int, float] = {}
    bus_side: dict[int, int] = {}
    port_assignments: dict[int, dict[str, str]] = {}
    for group_id, group in groups.items():
        cx, cy = group_centers[group_id]
        for node_id, (lx, ly) in group.local_positions_um.items():
            positions[node_id] = (cx + lx, cy + ly)
            side = 1 if positions[node_id][1] >= 0 else -1
            bus_side[node_id] = side
            # Rotate top-row qubits by 180 degrees so the junction-bearing
            # side of a standard TransmonCross can be oriented away from the
            # central bus. The internal geometry exposes all four arms.
            rotations[node_id] = 0.0
            assignments: dict[str, str] = {}
            if bus_id is not None and node_id in graph.neighbours(bus_id):
                assignments["bus"] = "south" if side > 0 else "north"
            if node_id in readout_by_transmon:
                assignments["readout"] = "north" if side > 0 else "south"
            port_assignments[node_id] = assignments

    # Direct pair arms follow frozen local x ordering.
    for group in groups.values():
        for edge in group.direct_edges:
            ax = positions[edge.a][0]
            bx = positions[edge.b][0]
            port_assignments[edge.a][f"direct:{edge.b}"] = "east" if bx > ax else "west"
            port_assignments[edge.b][f"direct:{edge.a}"] = "west" if bx > ax else "east"

    if bus_id is not None:
        bus_order = sorted(
            [node for node in graph.neighbours(bus_id) if graph.nodes[node].kind == NodeKind.TRANSMON],
            key=lambda node: positions[node][0],
        )
    else:
        bus_order = []

    return PlacementSolution(
        transmon_positions=positions,
        transmon_rotations_deg=rotations,
        groups=groups,
        node_to_group=node_to_group,
        bus_id=bus_id,
        bus_order=bus_order,
        bus_side=bus_side,
        readout_by_transmon=readout_by_transmon,
        feedline_by_readout=feedline_by_readout,
        port_assignments=port_assignments,
    )


def synthesize_layout(
    graph: CircuitGraph,
    config: SynthesisConfig,
    *,
    attempt: int = 0,
) -> LayoutPlan:
    """Create a complete physical plan from the semantic circuit graph."""
    placement = solve_initial_placement(graph, config)
    plan = LayoutPlan(
        name=graph.name,
        chip_width_um=config.chip_width_um,
        chip_height_um=config.chip_height_um,
        chip=chip_polygon(config.chip_width_um, config.chip_height_um, config.chip_margin_um),
        skipped_nodes=[
            {"node_id": node.node_id, "label": node.label, "kind": node.kind.value, "attrs": node.attrs}
            for node in graph.skipped_nodes
        ],
        notes=list(graph.notes),
        synthesis_attempt=attempt,
    )
    build_elements(graph, placement, plan, config)

    realized = {item.node_id for item in plan.components.values()} | {
        item.node_id for item in plan.routes.values()
    }
    missing = [node for node_id, node in graph.nodes.items() if node_id not in realized]
    if missing:
        details = ", ".join(f"{node.node_id}:{node.kind.value}:{node.label}" for node in missing)
        raise SynthesisError(
            "The semantic graph contains nodes outside the implemented physical scope: " + details
        )
    return plan


# ---- Plan validation ------------------------------------------------------

def validate_layout(plan: LayoutPlan, config: SynthesisConfig) -> list[Violation]:
    violations: list[Violation] = []
    objects = plan.all_objects()
    tolerance = config.geometry_area_tolerance_um2

    for owner, obj in objects.items():
        outside = obj.keepout.difference(plan.chip)
        if not outside.is_empty and outside.area > tolerance:
            violations.append(Violation(
                "OUTSIDE_CHIP",
                f"{owner} leaves the chip by {outside.area:.3f} um^2.",
                (owner,), float(outside.area),
            ))

    for (owner_a, obj_a), (owner_b, obj_b) in combinations(objects.items(), 2):
        pair = tuple(sorted((owner_a, owner_b)))
        metal = obj_a.metal.intersection(obj_b.metal)
        if not metal.is_empty and metal.area > tolerance and pair not in plan.expected_galvanic_pairs:
            violations.append(Violation(
                "METAL_INTERSECTION",
                f"Unintended metal overlap: {owner_a} / {owner_b}.",
                pair, float(metal.area),
            ))

        overlap = obj_a.keepout.intersection(obj_b.keepout)
        if overlap.is_empty or overlap.area <= tolerance:
            continue
        allowed = plan.windows_for_pair(owner_a, owner_b)
        illegal = overlap if allowed.is_empty else overlap.difference(allowed)
        if not illegal.is_empty and illegal.area > tolerance:
            violations.append(Violation(
                "KEEPOUT_VIOLATION",
                f"Keep-out overlap outside a coupling window: {owner_a} / {owner_b}.",
                pair, float(illegal.area),
            ))

    for route_id, route in plan.routes.items():
        if not route.centerline.is_simple:
            violations.append(Violation("ROUTE_SELF_INTERSECTION", f"{route_id} crosses itself.", (route_id,)))
        if route.target_length_um:
            error = abs(route.actual_length_um - route.target_length_um) / route.target_length_um
            if error > config.length_tolerance_fraction:
                violations.append(Violation(
                    "LENGTH_MISMATCH",
                    f"{route_id}: target {route.target_length_um:.3f} um, actual {route.actual_length_um:.3f} um.",
                    (route_id,),
                ))
        if route.role in {"READOUT", "COMMON_BUS"} and route.grounded_end is None:
            violations.append(Violation("MISSING_GROUND", f"{route_id} has no grounded endpoint.", (route_id,)))

    for window in plan.coupling_windows:
        owner_a, owner_b = objects.get(window.owner_a), objects.get(window.owner_b)
        if owner_a is None or owner_b is None:
            violations.append(Violation(
                "MISSING_COUPLING_OWNER",
                f"{window.window_id} references a missing owner.",
                (window.owner_a, window.owner_b),
            ))
            continue
        actual = float(owner_a.metal.distance(owner_b.metal))
        window.actual_gap_um = actual
        if abs(actual - window.target_gap_um) > config.coupling_gap_tolerance_um:
            violations.append(Violation(
                "COUPLING_GAP_MISMATCH",
                f"{window.window_id}: target {window.target_gap_um:.3f} um, actual {actual:.3f} um.",
                (window.owner_a, window.owner_b),
            ))

    plan.violations = violations
    return violations


def summarize_violations(violations: list[Violation], limit: int = 12) -> str:
    if not violations:
        return "no violations"
    lines = [f"{len(violations)} violation(s):"]
    lines.extend(f"- {item.code}: {item.message}" for item in violations[:limit])
    if len(violations) > limit:
        lines.append(f"- ... {len(violations) - limit} more")
    return "\n".join(lines)



class LayoutSynthesisFailed(RuntimeError):
    """No valid physical layout was found within the configured retries."""


@dataclass(slots=True)
class TranslationResult:
    graph: CircuitGraph
    plan: LayoutPlan
    config: SynthesisConfig
    attempt_errors: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.plan.violations


def translate_graph(
    graph_like: Any,
    *,
    config: SynthesisConfig | None = None,
    strict: bool | None = None,
) -> TranslationResult:
    """Parse cQED_Gen output, synthesize layout zero, and validate geometry."""
    base = config or SynthesisConfig()
    strict = base.strict_validation if strict is None else bool(strict)
    graph = parse_graph(graph_like, skip_inductive=True)
    errors: list[str] = []
    final_plan: LayoutPlan | None = None
    used_config = base

    for attempt in range(base.max_chip_expansions + 1):
        factor = base.chip_expansion_factor ** attempt
        used_config = base.scaled(factor) if attempt else base
        try:
            candidate = synthesize_layout(graph, used_config, attempt=attempt)
            validate_layout(candidate, used_config)
            final_plan = candidate
            if not candidate.violations:
                break
            errors.append(f"attempt {attempt}: {summarize_violations(candidate.violations)}")
        except (RoutingError, LengthAllocationError, SynthesisError, ValueError) as exc:
            errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")

    if final_plan is None:
        raise LayoutSynthesisFailed("No layout candidate was produced.\n" + "\n".join(errors))
    if final_plan.violations and strict:
        raise LayoutSynthesisFailed(
            "No globally valid layout was found.\n"
            + "\n".join(errors)
            + "\nFinal candidate:\n"
            + summarize_violations(final_plan.violations)
        )
    return TranslationResult(graph=graph, plan=final_plan, config=used_config, attempt_errors=errors)
