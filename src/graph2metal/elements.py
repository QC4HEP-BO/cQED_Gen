"""All physical layout elements and geometry primitives.

This is the extension point of the compact package.  To add a new physical
primitive, implement its builder in this file and include it in
``build_elements`` at the bottom.  The other four modules do not need changes.
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

from shapely import affinity
from shapely.geometry import (
    GeometryCollection, LineString, MultiPolygon, Point, Polygon, box, mapping,
)
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.prepared import prep

from .graph_model import (
    CircuitEdge, CircuitGraph, NodeKind, ResonatorRole, SynthesisConfig,
    coupling_gap_um, direct_pad_length_um, direct_pad_width_um,
    edge_gap_um, edge_parallel_length_um, other, resonator_length_um,
    transmon_dimensions,
)

# --------------------------------------------------------------------------- #
# Poses, ports, and the physical objects that make up a layout
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Pose:
    x_um: float
    y_um: float
    rotation_deg: float = 0.0


@dataclass(frozen=True, slots=True)
class Port:
    name: str
    point_um: tuple[float, float]
    normal: tuple[float, float]
    tangent: tuple[float, float]
    owner_id: int
    role: str | None = None


@dataclass(slots=True)
class PhysicalComponent:
    component_id: str
    node_id: int
    label: str
    kind: str
    role: str
    pose: Pose
    metal: BaseGeometry
    keepout: BaseGeometry
    ports: dict[str, Port] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    group_id: str | None = None


@dataclass(slots=True)
class PhysicalRoute:
    route_id: str
    node_id: int
    label: str
    role: str
    centerline: LineString
    metal: BaseGeometry
    keepout: BaseGeometry
    trace_width_um: float
    trace_gap_um: float
    target_length_um: float | None = None
    grounded_end: tuple[float, float] | None = None
    open_end: tuple[float, float] | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    @property
    def actual_length_um(self) -> float:
        return float(self.centerline.length)


@dataclass(slots=True)
class CouplingWindow:
    window_id: str
    owner_a: str
    owner_b: str
    interface: str
    polygon: BaseGeometry
    target_gap_um: float
    coupling_fF: float | None
    actual_gap_um: float | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    @property
    def pair_key(self) -> tuple[str, str]:
        return tuple(sorted((self.owner_a, self.owner_b)))


@dataclass(slots=True)
class Violation:
    code: str
    message: str
    owners: tuple[str, ...] = ()
    area_um2: float | None = None


@dataclass
class LayoutPlan:
    name: str
    chip_width_um: float
    chip_height_um: float
    chip: Polygon
    components: dict[str, PhysicalComponent] = field(default_factory=dict)
    routes: dict[str, PhysicalRoute] = field(default_factory=dict)
    coupling_windows: list[CouplingWindow] = field(default_factory=list)
    expected_galvanic_pairs: set[tuple[str, str]] = field(default_factory=set)
    skipped_nodes: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    violations: list[Violation] = field(default_factory=list)
    synthesis_attempt: int = 0

    def all_objects(self) -> dict[str, "PhysicalComponent | PhysicalRoute"]:
        return {**self.components, **self.routes}

    def obstacle_union(self, *, exclude: set[str] | None = None) -> BaseGeometry:
        exclude = exclude or set()
        geometries = [obj.keepout for key, obj in self.all_objects().items() if key not in exclude]
        return unary_union(geometries) if geometries else Polygon()

    def windows_for_pair(self, a: str, b: str) -> BaseGeometry:
        pair = tuple(sorted((a, b)))
        windows = [window.polygon for window in self.coupling_windows if window.pair_key == pair]
        return unary_union(windows) if windows else Polygon()

    def to_dict(self) -> dict[str, Any]:
        def geom_payload(geom: BaseGeometry) -> dict[str, Any]:
            return mapping(geom)

        components = []
        for component in self.components.values():
            components.append(
                {
                    "component_id": component.component_id,
                    "node_id": component.node_id,
                    "label": component.label,
                    "kind": component.kind,
                    "role": component.role,
                    "pose": asdict(component.pose),
                    "metal": geom_payload(component.metal),
                    "keepout": geom_payload(component.keepout),
                    "ports": {name: asdict(port) for name, port in component.ports.items()},
                    "parameters": component.parameters,
                    "group_id": component.group_id,
                }
            )
        routes = []
        for route in self.routes.values():
            routes.append(
                {
                    "route_id": route.route_id,
                    "node_id": route.node_id,
                    "label": route.label,
                    "role": route.role,
                    "centerline": geom_payload(route.centerline),
                    "metal": geom_payload(route.metal),
                    "keepout": geom_payload(route.keepout),
                    "trace_width_um": route.trace_width_um,
                    "trace_gap_um": route.trace_gap_um,
                    "target_length_um": route.target_length_um,
                    "actual_length_um": route.actual_length_um,
                    "grounded_end": route.grounded_end,
                    "open_end": route.open_end,
                    "parameters": route.parameters,
                }
            )
        windows = [
            {
                "window_id": window.window_id,
                "owner_a": window.owner_a,
                "owner_b": window.owner_b,
                "interface": window.interface,
                "polygon": geom_payload(window.polygon),
                "target_gap_um": window.target_gap_um,
                "actual_gap_um": window.actual_gap_um,
                "coupling_fF": window.coupling_fF,
                "parameters": window.parameters,
            }
            for window in self.coupling_windows
        ]
        return {
            "schema_version": "graph2metal-layout-plan-0.29",
            "name": self.name,
            "chip": {
                "width_um": self.chip_width_um,
                "height_um": self.chip_height_um,
                "polygon": geom_payload(self.chip),
            },
            "components": components,
            "routes": routes,
            "coupling_windows": windows,
            "expected_galvanic_pairs": sorted([list(pair) for pair in self.expected_galvanic_pairs]),
            "skipped_nodes": self.skipped_nodes,
            "notes": self.notes,
            "violations": [asdict(v) for v in self.violations],
            "synthesis_attempt": self.synthesis_attempt,
        }


def chip_polygon(width_um: float, height_um: float, margin_um: float = 0.0) -> Polygon:
    half_w = width_um / 2.0 - margin_um
    half_h = height_um / 2.0 - margin_um
    return box(-half_w, -half_h, half_w, half_h)


def cross_polygon(arm_length_um: float, width_um: float, pose: Pose) -> BaseGeometry:
    half_w = width_um / 2.0
    horizontal = box(-arm_length_um, -half_w, arm_length_um, half_w)
    vertical = box(-half_w, -arm_length_um, half_w, arm_length_um)
    geom = unary_union([horizontal, vertical])
    geom = affinity.rotate(geom, pose.rotation_deg, origin=(0.0, 0.0), use_radians=False)
    return affinity.translate(geom, xoff=pose.x_um, yoff=pose.y_um)


def local_to_world(point: tuple[float, float], pose: Pose) -> tuple[float, float]:
    p = affinity.rotate(Point(point), pose.rotation_deg, origin=(0.0, 0.0), use_radians=False)
    return (p.x + pose.x_um, p.y + pose.y_um)


def rotate_vector(vector: tuple[float, float], angle_deg: float) -> tuple[float, float]:
    p = affinity.rotate(Point(vector), angle_deg, origin=(0.0, 0.0), use_radians=False)
    return (float(p.x), float(p.y))


def cross_ports(node_id: int, arm_length_um: float, pose: Pose) -> dict[str, Port]:
    definitions = {
        "east": ((arm_length_um, 0.0), (1.0, 0.0), (0.0, 1.0)),
        "west": ((-arm_length_um, 0.0), (-1.0, 0.0), (0.0, 1.0)),
        "north": ((0.0, arm_length_um), (0.0, 1.0), (1.0, 0.0)),
        "south": ((0.0, -arm_length_um), (0.0, -1.0), (1.0, 0.0)),
    }
    ports = {}
    for name, (point, normal, tangent) in definitions.items():
        ports[name] = Port(
            name=name,
            point_um=local_to_world(point, pose),
            normal=rotate_vector(normal, pose.rotation_deg),
            tangent=rotate_vector(tangent, pose.rotation_deg),
            owner_id=node_id,
        )
    return ports


def route_geometry(
    points: list[tuple[float, float]],
    trace_width_um: float,
    trace_gap_um: float,
    clearance_um: float,
) -> tuple[LineString, BaseGeometry, BaseGeometry]:
    line = LineString(points)
    metal = line.buffer(trace_width_um / 2.0, cap_style=2, join_style=1)
    occupied_radius = trace_width_um / 2.0 + trace_gap_um + clearance_um
    keepout = line.buffer(occupied_radius, cap_style=2, join_style=1)
    return line, metal, keepout


def rectangle_along_segment(
    start: tuple[float, float], end: tuple[float, float], half_width_um: float
) -> BaseGeometry:
    return LineString([start, end]).buffer(half_width_um, cap_style=2, join_style=2)


def geometry_bounds_dict(geom: BaseGeometry) -> dict[str, float]:
    minx, miny, maxx, maxy = geom.bounds
    return {"min_x_um": minx, "min_y_um": miny, "max_x_um": maxx, "max_y_um": maxy}


def normalize_polygon(geom: BaseGeometry) -> BaseGeometry:
    if geom.is_empty:
        return Polygon()
    fixed = geom.buffer(0)
    if isinstance(fixed, (Polygon, MultiPolygon)):
        return fixed
    return fixed


def launchpad_wirebond_geometry(
    tie_point_um: tuple[float, float],
    interior_vector: tuple[float, float],
    *,
    trace_width_um: float,
    trace_gap_um: float,
    lead_length_um: float,
    pad_width_um: float,
    pad_height_um: float,
    pad_gap_um: float,
    taper_height_um: float,
    clearance_um: float,
) -> tuple[BaseGeometry, BaseGeometry]:
    """Return the native LaunchpadWirebond metal and routing keep-out.

    The local polygon follows Quantum Metal's 0.7.x ``LaunchpadWirebond``
    construction. ``tie_point_um`` is the component's ``tie`` pin and the
    interior vector points from the chip edge toward the feedline body.
    """

    dx, dy = interior_vector
    norm = (dx * dx + dy * dy) ** 0.5
    if norm <= 1e-12:
        raise ValueError("Launchpad interior vector cannot be zero")
    dx /= norm
    dy /= norm
    angle_deg = math.degrees(math.atan2(dy, dx))
    origin_x = tie_point_um[0] - lead_length_um * dx
    origin_y = tie_point_um[1] - lead_length_um * dy

    trace_half = trace_width_um / 2.0
    pad_half = pad_width_um / 2.0
    metal = Polygon(
        [
            (0.0, trace_half),
            (-taper_height_um, pad_half),
            (-(pad_height_um + taper_height_um), pad_half),
            (-(pad_height_um + taper_height_um), -pad_half),
            (-taper_height_um, -pad_half),
            (0.0, -trace_half),
            (lead_length_um, -trace_half),
            (lead_length_um, trace_half),
        ]
    )
    pocket = Polygon(
        [
            (0.0, trace_half + trace_gap_um),
            (-taper_height_um, pad_half + pad_gap_um),
            (
                -(pad_height_um + taper_height_um + pad_gap_um),
                pad_half + pad_gap_um,
            ),
            (
                -(pad_height_um + taper_height_um + pad_gap_um),
                -(pad_half + pad_gap_um),
            ),
            (-taper_height_um, -(pad_half + pad_gap_um)),
            (0.0, -(trace_half + trace_gap_um)),
            (lead_length_um, -(trace_half + trace_gap_um)),
            (lead_length_um, trace_half + trace_gap_um),
        ]
    )
    metal = affinity.rotate(metal, angle_deg, origin=(0.0, 0.0), use_radians=False)
    pocket = affinity.rotate(pocket, angle_deg, origin=(0.0, 0.0), use_radians=False)
    metal = affinity.translate(metal, xoff=origin_x, yoff=origin_y)
    pocket = affinity.translate(pocket, xoff=origin_x, yoff=origin_y)
    keepout = pocket.buffer(clearance_um, cap_style=2, join_style=2)
    return metal, keepout



# ---- Routing --------------------------------------------------------------

class RoutingError(RuntimeError):
    pass


class LengthAllocationError(RoutingError):
    pass


def polyline_length(points: Iterable[tuple[float, float]]) -> float:
    pts = list(points)
    return sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))


def simplify_orthogonal(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(points) <= 2:
        return points
    simplified = [points[0]]
    for point in points[1:]:
        if math.dist(simplified[-1], point) < 1e-9:
            continue
        simplified.append(point)
    result = [simplified[0]]
    for index in range(1, len(simplified) - 1):
        a, b, c = result[-1], simplified[index], simplified[index + 1]
        if (abs(a[0] - b[0]) < 1e-9 and abs(b[0] - c[0]) < 1e-9) or (
            abs(a[1] - b[1]) < 1e-9 and abs(b[1] - c[1]) < 1e-9
        ):
            continue
        result.append(b)
    result.append(simplified[-1])
    return result


def _nearest_grid_point(point: tuple[float, float], step: float) -> tuple[int, int]:
    return (round(point[0] / step), round(point[1] / step))


def _world(cell: tuple[int, int], step: float) -> tuple[float, float]:
    return (cell[0] * step, cell[1] * step)


def route_orthogonal_astar(
    start_um: tuple[float, float],
    end_um: tuple[float, float],
    *,
    chip: Polygon,
    obstacles: BaseGeometry | None,
    grid_step_um: float,
    corridor_radius_um: float,
    allowed_regions: BaseGeometry | None = None,
    bend_penalty: float = 0.35,
) -> list[tuple[float, float]]:
    """Route an orthogonal finite-width corridor with A*.

    ``obstacles`` should represent already reserved regions. The router adds a
    small finite-width margin and subtracts only explicit allowed coupling
    windows. It never accepts a crossing when no path exists.
    """
    if grid_step_um <= 0:
        raise ValueError("grid_step_um must be positive")
    forbidden = obstacles if obstacles is not None else Polygon()
    if not forbidden.is_empty:
        forbidden = forbidden.buffer(max(1.0, corridor_radius_um), join_style=2)
    if allowed_regions is not None and not allowed_regions.is_empty:
        forbidden = forbidden.difference(allowed_regions)
    valid_area = chip.buffer(-corridor_radius_um)
    if valid_area.is_empty:
        raise RoutingError("Chip has no usable routing area after corridor inset")
    prepared_forbidden = prep(forbidden)
    prepared_valid = prep(valid_area)

    start_cell = _nearest_grid_point(start_um, grid_step_um)
    end_cell = _nearest_grid_point(end_um, grid_step_um)
    minx, miny, maxx, maxy = valid_area.bounds
    ix_min = math.floor(minx / grid_step_um)
    ix_max = math.ceil(maxx / grid_step_um)
    iy_min = math.floor(miny / grid_step_um)
    iy_max = math.ceil(maxy / grid_step_um)

    def cell_valid(cell: tuple[int, int]) -> bool:
        if not (ix_min <= cell[0] <= ix_max and iy_min <= cell[1] <= iy_max):
            return False
        point = Point(_world(cell, grid_step_um))
        return prepared_valid.covers(point) and not prepared_forbidden.intersects(point)

    # Search for nearby free snapped cells if the exact rounded point sits on
    # the edge of a keep-out. The exact endpoint is restored after routing.
    def nearest_valid(seed: tuple[int, int]) -> tuple[int, int]:
        if cell_valid(seed):
            return seed
        for radius in range(1, 9):
            candidates = []
            for dx in range(-radius, radius + 1):
                candidates.append((seed[0] + dx, seed[1] - radius))
                candidates.append((seed[0] + dx, seed[1] + radius))
            for dy in range(-radius + 1, radius):
                candidates.append((seed[0] - radius, seed[1] + dy))
                candidates.append((seed[0] + radius, seed[1] + dy))
            candidates.sort(key=lambda cell: abs(cell[0] - seed[0]) + abs(cell[1] - seed[1]))
            for candidate in candidates:
                if cell_valid(candidate):
                    return candidate
        raise RoutingError(f"No free grid point near endpoint {_world(seed, grid_step_um)}")

    start_cell = nearest_valid(start_cell)
    end_cell = nearest_valid(end_cell)

    directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
    queue: list[tuple[float, float, tuple[int, int], tuple[int, int] | None]] = []
    heapq.heappush(queue, (0.0, 0.0, start_cell, None))
    parent: dict[tuple[tuple[int, int], tuple[int, int] | None], tuple[tuple[int, int], tuple[int, int] | None] | None] = {
        (start_cell, None): None
    }
    best: dict[tuple[tuple[int, int], tuple[int, int] | None], float] = {(start_cell, None): 0.0}
    goal_state: tuple[tuple[int, int], tuple[int, int] | None] | None = None

    while queue:
        _, cost, cell, previous_direction = heapq.heappop(queue)
        state = (cell, previous_direction)
        if cost > best.get(state, float("inf")) + 1e-9:
            continue
        if cell == end_cell:
            goal_state = state
            break
        for direction in directions:
            neighbour = (cell[0] + direction[0], cell[1] + direction[1])
            if not cell_valid(neighbour):
                continue
            a = _world(cell, grid_step_um)
            b = _world(neighbour, grid_step_um)
            segment = LineString([a, b])
            if prepared_forbidden.intersects(segment):
                continue
            turn = bend_penalty * grid_step_um if previous_direction and direction != previous_direction else 0.0
            next_cost = cost + grid_step_um + turn
            next_state = (neighbour, direction)
            if next_cost + 1e-9 >= best.get(next_state, float("inf")):
                continue
            best[next_state] = next_cost
            parent[next_state] = state
            heuristic = (abs(neighbour[0] - end_cell[0]) + abs(neighbour[1] - end_cell[1])) * grid_step_um
            heapq.heappush(queue, (next_cost + heuristic, next_cost, neighbour, direction))

    if goal_state is None:
        raise RoutingError(f"No collision-free orthogonal route from {start_um} to {end_um}")

    cells: list[tuple[int, int]] = []
    state: tuple[tuple[int, int], tuple[int, int] | None] | None = goal_state
    while state is not None:
        cells.append(state[0])
        state = parent[state]
    cells.reverse()
    points = [_world(cell, grid_step_um) for cell in cells]
    points[0] = start_um
    points[-1] = end_um
    return simplify_orthogonal(points)


def connect_waypoints(
    waypoints: list[tuple[float, float]],
    *,
    chip: Polygon,
    obstacles: BaseGeometry | None,
    grid_step_um: float,
    corridor_radius_um: float,
    allowed_regions: BaseGeometry | None = None,
) -> list[tuple[float, float]]:
    if len(waypoints) < 2:
        return list(waypoints)
    result = [waypoints[0]]
    for start, end in zip(waypoints, waypoints[1:]):
        segment = route_orthogonal_astar(
            start,
            end,
            chip=chip,
            obstacles=obstacles,
            grid_step_um=grid_step_um,
            corridor_radius_um=corridor_radius_um,
            allowed_regions=allowed_regions,
        )
        result.extend(segment[1:])
    return simplify_orthogonal(result)


def make_serpentine_exact(
    start_um: tuple[float, float],
    target_length_um: float,
    *,
    bounds_um: tuple[float, float, float, float],
    pitch_um: float,
    first_horizontal: int = 1,
    vertical_direction: int = -1,
) -> list[tuple[float, float]]:
    """Create an exact-length Manhattan serpentine beginning at ``start_um``.

    The start must lie inside the supplied box. The path alternates full-width
    horizontal runs and one-pitch vertical steps. The final segment is trimmed
    so its polyline length exactly matches ``target_length_um``.
    """
    if target_length_um < -1e-9:
        raise LengthAllocationError("Requested negative serpentine length")
    if target_length_um <= 1e-9:
        return [start_um]
    minx, miny, maxx, maxy = bounds_um
    if not (minx - 1e-6 <= start_um[0] <= maxx + 1e-6 and miny - 1e-6 <= start_um[1] <= maxy + 1e-6):
        raise LengthAllocationError(f"Serpentine start {start_um} is outside bounds {bounds_um}")
    if pitch_um <= 0:
        raise ValueError("pitch_um must be positive")

    points = [start_um]
    remaining = target_length_um
    horizontal_direction = 1 if first_horizontal >= 0 else -1
    x, y = start_um
    max_steps = int((maxy - miny) / pitch_um) + 3

    for _ in range(max_steps):
        target_x = maxx if horizontal_direction > 0 else minx
        horizontal = abs(target_x - x)
        if remaining <= horizontal + 1e-9:
            x = x + horizontal_direction * remaining
            points.append((x, y))
            return simplify_orthogonal(points)
        if horizontal > 1e-9:
            x = target_x
            points.append((x, y))
            remaining -= horizontal

        target_y = y + vertical_direction * pitch_um
        if target_y < miny - 1e-9 or target_y > maxy + 1e-9:
            break
        vertical = abs(target_y - y)
        if remaining <= vertical + 1e-9:
            y = y + vertical_direction * remaining
            points.append((x, y))
            return simplify_orthogonal(points)
        y = target_y
        points.append((x, y))
        remaining -= vertical
        horizontal_direction *= -1

    capacity = target_length_um - remaining
    raise LengthAllocationError(
        f"Meander box capacity {capacity:.1f} um is smaller than requested {target_length_um:.1f} um"
    )


def trim_polyline_to_length(points: list[tuple[float, float]], target_length_um: float) -> list[tuple[float, float]]:
    if target_length_um < 0:
        raise ValueError("target_length_um must be non-negative")
    if not points:
        return []
    result = [points[0]]
    remaining = target_length_um
    for a, b in zip(points, points[1:]):
        segment_length = math.dist(a, b)
        if remaining >= segment_length - 1e-9:
            result.append(b)
            remaining -= segment_length
            if remaining <= 1e-9:
                return simplify_orthogonal(result)
            continue
        if segment_length <= 1e-12:
            continue
        ratio = remaining / segment_length
        result.append((a[0] + ratio * (b[0] - a[0]), a[1] + ratio * (b[1] - a[1])))
        return simplify_orthogonal(result)
    if remaining > 1e-6:
        raise LengthAllocationError(
            f"Polyline has length {polyline_length(points):.1f} um, below target {target_length_um:.1f} um"
        )
    return simplify_orthogonal(result)


def window_union(windows: Iterable[BaseGeometry]) -> BaseGeometry:
    items = [window for window in windows if window is not None and not window.is_empty]
    return unary_union(items) if items else Polygon()


# ---- Shared element helpers ----------------------------------------------

class SynthesisError(RuntimeError):
    """The semantic circuit cannot be synthesized by the implemented primitives."""


@dataclass(slots=True)
class FeedlineRequirement:
    feedline_id: int
    resonator_route_id: str
    resonator_node_id: int
    segment_start_um: tuple[float, float]
    segment_end_um: tuple[float, float]
    outward_normal: tuple[float, float]
    target_gap_um: float
    coupling_fF: float | None
    parallel_length_um: float


def _unit(vector: tuple[float, float]) -> tuple[float, float]:
    norm = math.hypot(*vector)
    if norm <= 1e-12:
        raise ValueError("Zero-length vector")
    return vector[0] / norm, vector[1] / norm


def _add(a: tuple[float, float], b: tuple[float, float], scale: float = 1.0) -> tuple[float, float]:
    return a[0] + b[0] * scale, a[1] + b[1] * scale


def _owner_component(node_id: int) -> str:
    return f"component_{node_id}"


def _owner_route(node_id: int) -> str:
    return f"route_{node_id}"


def _ground_pad(point: tuple[float, float], trace_width_um: float, size_um: float = 70.0) -> BaseGeometry:
    return box(
        point[0] - size_um / 2.0,
        point[1] - size_um / 2.0,
        point[0] + size_um / 2.0,
        point[1] + size_um / 2.0,
    )


def _routing_obstacles(
    plan: LayoutPlan,
    *,
    allowed_by_owner: dict[str, BaseGeometry] | None = None,
    exclude: set[str] | None = None,
) -> BaseGeometry:
    allowed_by_owner = allowed_by_owner or {}
    exclude = exclude or set()
    geometries = []
    for owner, obj in plan.all_objects().items():
        if owner in exclude:
            continue
        geometry = obj.keepout
        allowed = allowed_by_owner.get(owner)
        if allowed is not None and not allowed.is_empty:
            geometry = geometry.difference(allowed)
        if not geometry.is_empty:
            geometries.append(geometry)
    return unary_union(geometries) if geometries else Polygon()


def _make_coupling_window(
    window_id: str,
    owner_a: str,
    owner_b: str,
    interface: str,
    geometries: list[BaseGeometry],
    *,
    target_gap_um: float,
    coupling_fF: float | None,
    margin_um: float,
    parameters: dict[str, Any] | None = None,
) -> CouplingWindow:
    polygon = unary_union(geometries).convex_hull.buffer(margin_um, cap_style=2, join_style=2)
    return CouplingWindow(
        window_id=window_id,
        owner_a=owner_a,
        owner_b=owner_b,
        interface=interface,
        polygon=polygon,
        target_gap_um=target_gap_um,
        actual_gap_um=None,
        coupling_fF=coupling_fF,
        parameters=parameters or {},
    )


def _coupling_edge(graph: CircuitGraph, a: int, b: int) -> CircuitEdge | None:
    return graph.edge_between(a, b)



# ---- Physical element builders ------------------------------------------

def _attach_direct_pad(
    metal: BaseGeometry,
    port: Port,
    length_um: float,
    width_um: float,
) -> tuple[BaseGeometry, tuple[float, float], BaseGeometry]:
    start = _add(port.point_um, port.normal, -1.0)
    end = _add(port.point_um, port.normal, length_um)
    pad = rectangle_along_segment(start, end, width_um / 2.0)
    return unary_union([metal, pad]), end, pad


_CARDINAL_ARMS = ("north", "south", "east", "west")


def choose_junction_arm(
    port_assignments: dict[str, str],
    x_um: float,
) -> str:
    """Choose an unused TransmonCross arm for the Josephson junction.

    ``TransmonCross`` always draws its junction on its local south arm.  The
    graph2metal plan, however, can reserve south/north/east/west for capacitive
    bus, readout, or direct-transmon interfaces.  This helper chooses a free
    local arm and prefers the outer horizontal side of the qubit array so that
    the junction is kept away from the common bus and readout resonator.
    """
    occupied = {str(value).lower() for value in port_assignments.values()}
    invalid = occupied.difference(_CARDINAL_ARMS)
    if invalid:
        raise SynthesisError(
            "Unsupported transmon port arm(s): " + ", ".join(sorted(invalid))
        )

    horizontal_preference = ("west", "east") if x_um < 0.0 else ("east", "west")
    for arm in (*horizontal_preference, "south", "north"):
        if arm not in occupied:
            return arm

    raise SynthesisError(
        "No free TransmonCross arm remains for the Josephson junction; "
        f"occupied arms are {sorted(occupied)}."
    )


def build_transmons(
    graph: CircuitGraph,
    placement: PlacementSolution,
    plan: LayoutPlan,
    config: SynthesisConfig,
) -> dict[tuple[int, int], tuple[tuple[float, float], tuple[float, float], BaseGeometry, BaseGeometry]]:
    """Create crosses and direct-coupling pads.

    Returns metadata for every direct edge: pad tip A, pad tip B, pad A, pad B.
    """
    direct_metadata: dict[
        tuple[int, int], tuple[tuple[float, float], tuple[float, float], BaseGeometry, BaseGeometry]
    ] = {}
    components: dict[int, PhysicalComponent] = {}

    for node in graph.nodes_of_kind(NodeKind.TRANSMON):
        dims = transmon_dimensions(node.attrs, config)
        x, y = placement.transmon_positions[node.node_id]
        pose = Pose(x, y, placement.transmon_rotations_deg[node.node_id])
        metal = cross_polygon(dims.arm_length_um, dims.cross_width_um, pose)
        ports = cross_ports(node.node_id, dims.arm_length_um, pose)
        assignments = dict(placement.port_assignments.get(node.node_id, {}))
        junction_arm = choose_junction_arm(assignments, x)
        components[node.node_id] = PhysicalComponent(
            component_id=_owner_component(node.node_id),
            node_id=node.node_id,
            label=node.label,
            kind=NodeKind.TRANSMON.value,
            role="TRANSMON",
            pose=pose,
            metal=metal,
            keepout=Polygon(),
            ports=ports,
            parameters={
                "capacitance_fF": dims.capacitance_fF,
                "Lj_nH": dims.inductance_nH,
                "cross_arm_length_um": dims.arm_length_um,
                "cross_width_um": dims.cross_width_um,
                "cross_gap_um": config.transmon_cross_gap_um,
                "sizing_model": "cross linear dimension proportional to sqrt(C/C_ref)",
                "port_assignments": assignments,
                "junction_arm": junction_arm,
            },
            group_id=placement.node_to_group.get(node.node_id),
        )

    for edge in [edge for edge in graph.edges if edge.kind == "transmon_transmon_capacitive"]:
        a, b = edge.a, edge.b
        component_a = components[a]
        component_b = components[b]
        name_a = placement.port_assignments[a][f"direct:{b}"]
        name_b = placement.port_assignments[b][f"direct:{a}"]
        port_a = component_a.ports[name_a]
        port_b = component_b.ports[name_b]
        pad_length = direct_pad_length_um(edge.coupling_fF)
        pad_width = direct_pad_width_um(edge.coupling_fF)
        component_a.metal, tip_a, pad_a = _attach_direct_pad(component_a.metal, port_a, pad_length, pad_width)
        component_b.metal, tip_b, pad_b = _attach_direct_pad(component_b.metal, port_b, pad_length, pad_width)
        direct_metadata[edge.key] = (tip_a, tip_b, pad_a, pad_b)

    for component in components.values():
        component.keepout = component.metal.buffer(config.component_clearance_um, join_style=2)
        plan.components[component.component_id] = component

    for edge in [edge for edge in graph.edges if edge.kind == "transmon_transmon_capacitive"]:
        tip_a, tip_b, pad_a, pad_b = direct_metadata[edge.key]
        gap = edge_gap_um(edge)
        window = _make_coupling_window(
            f"window_direct_{edge.a}_{edge.b}",
            _owner_component(edge.a),
            _owner_component(edge.b),
            edge.kind,
            [pad_a, pad_b, LineString([tip_a, tip_b])],
            target_gap_um=gap,
            coupling_fF=edge.coupling_fF,
            margin_um=config.component_clearance_um + config.coupling_window_margin_um,
            parameters={"rigid_group": placement.node_to_group.get(edge.a), "pad_length_um": direct_pad_length_um(edge.coupling_fF)},
        )
        window.actual_gap_um = float(Point(tip_a).distance(Point(tip_b)))
        plan.coupling_windows.append(window)
    return direct_metadata


def _route_with_ground(
    route_id: str,
    node_id: int,
    label: str,
    role: str,
    points: list[tuple[float, float]],
    config: SynthesisConfig,
    *,
    target_length_um: float | None,
    open_end: tuple[float, float] | None,
    parameters: dict[str, Any],
) -> PhysicalRoute:
    line, metal, keepout = route_geometry(
        points,
        config.trace_width_um,
        config.trace_gap_um,
        config.route_clearance_um,
    )
    ground = _ground_pad(points[-1], config.trace_width_um)
    metal = unary_union([metal, ground])
    keepout = unary_union([keepout, ground.buffer(config.route_clearance_um, join_style=2)])
    parameters = dict(parameters)
    parameters["ground_termination_polygon_bounds_um"] = list(ground.bounds)
    return PhysicalRoute(
        route_id=route_id,
        node_id=node_id,
        label=label,
        role=role,
        centerline=line,
        metal=metal,
        keepout=keepout,
        trace_width_um=config.trace_width_um,
        trace_gap_um=config.trace_gap_um,
        target_length_um=target_length_um,
        grounded_end=points[-1],
        open_end=open_end,
        parameters=parameters,
    )


def build_bus(
    graph: CircuitGraph,
    placement: PlacementSolution,
    plan: LayoutPlan,
    config: SynthesisConfig,
) -> None:
    if placement.bus_id is None:
        return
    bus = graph.nodes[placement.bus_id]
    target_length = resonator_length_um(
        bus.attrs,
        default_frequency_GHz=config.default_bus_frequency_GHz,
        config=config,
    )
    route_id = _owner_route(bus.node_id)
    waypoints: list[tuple[float, float]] = []
    windows: list[CouplingWindow] = []

    for transmon_id in placement.bus_order:
        component = plan.components[_owner_component(transmon_id)]
        port_name = placement.port_assignments[transmon_id]["bus"]
        port = component.ports[port_name]
        edge = _coupling_edge(graph, transmon_id, bus.node_id)
        coupling = edge.coupling_fF if edge else None
        gap = edge_gap_um(edge) if edge else coupling_gap_um(coupling, "transmon_resonator_capacitive")
        waypoint = _add(port.point_um, port.normal, gap + config.trace_width_um / 2.0)
        waypoints.append(waypoint)
        local_line = LineString([
            _add(waypoint, port.tangent, -140.0),
            _add(waypoint, port.tangent, 140.0),
        ])
        window = _make_coupling_window(
            f"window_bus_{transmon_id}_{bus.node_id}",
            component.component_id,
            route_id,
            "transmon_bus_capacitive",
            [component.metal.intersection(Point(port.point_um).buffer(100.0)), local_line],
            target_gap_um=gap,
            coupling_fF=coupling,
            margin_um=config.component_clearance_um + config.coupling_window_margin_um,
            parameters={"bus_endpoint_region": "open", "waypoint_um": waypoint},
        )
        window.actual_gap_um = gap
        windows.append(window)

    if not waypoints:
        raise SynthesisError("A common bus was classified but has no connected transmons")
    waypoints = sorted(waypoints, key=lambda point: point[0])
    first = waypoints[0]
    last = waypoints[-1]
    points = [(first[0] - config.bus_open_lead_um, first[1])]
    current = points[0]
    for waypoint in waypoints:
        if abs(current[1] - waypoint[1]) > 1e-9:
            # Move the vertical jog away from the coupling point. Explicit graph
            # updates can change individual cross sizes and therefore the bus
            # waypoint heights; a jog directly at the previous transmon would
            # reduce its true metal-to-metal gap.
            jog_x = 0.5 * (current[0] + waypoint[0])
            if abs(jog_x - current[0]) > 1e-9:
                points.append((jog_x, current[1]))
            points.append((jog_x, waypoint[1]))
        points.append(waypoint)
        current = waypoint
    points = simplify_orthogonal(points)
    coupling_region_length = polyline_length(points)

    chip_minx, chip_miny, chip_maxx, chip_maxy = plan.chip.bounds
    meander_width = config.bus_meander_width_um
    meander_height = config.bus_meander_height_um
    right_min = last[0] + config.bus_post_coupling_lead_um
    if right_min + meander_width <= chip_maxx:
        minx = right_min
        maxx = right_min + meander_width
    else:
        maxx = first[0] - config.bus_open_lead_um - config.bus_post_coupling_lead_um
        minx = maxx - meander_width
    # All bus-coupled qubits/readouts are above the coupling corridor, so the
    # bus length-compensation box is reserved below it. The bus enters at the
    # upper corner only after the last qubit and terminates at ground inside
    # this lower, otherwise empty region.
    maxy = min(-220.0, chip_maxy - 120.0)
    miny = max(chip_miny + 120.0, maxy - meander_height)
    entry = (minx if minx > last[0] else maxx, maxy)

    provisional_windows = window_union(window.polygon for window in windows)
    allowed_by_owner = {window.owner_a: window.polygon for window in windows}
    obstacles = _routing_obstacles(plan, allowed_by_owner=allowed_by_owner)
    corridor_radius = config.trace_width_um / 2.0 + config.trace_gap_um + config.route_clearance_um

    # Approach the meander from outside its allocated box. This prevents the
    # connector from cutting across later serpentine turns. The construction
    # is deterministic and stays beyond the rightmost/leftmost qubit after the
    # final coupling point; a finite-width collision check is still applied.
    if entry[0] == minx:
        stage_x = minx - 2.2 * corridor_radius
    else:
        stage_x = maxx + 2.2 * corridor_radius
    connector = simplify_orthogonal([
        points[-1],
        (stage_x, points[-1][1]),
        (stage_x, entry[1]),
        entry,
    ])
    connector_corridor = LineString(connector).buffer(corridor_radius, cap_style=2, join_style=1)
    illegal_connector = connector_corridor.intersection(obstacles)
    if not illegal_connector.is_empty and illegal_connector.area > config.geometry_area_tolerance_um2:
        meander_reserved = box(minx, miny, maxx, maxy).buffer(corridor_radius, join_style=2)
        entry_access = Point(entry).buffer(2.2 * corridor_radius, cap_style=3)
        fallback_obstacles = unary_union([obstacles, meander_reserved.difference(entry_access)])
        stage = (stage_x, entry[1])
        first_leg = route_orthogonal_astar(
            points[-1],
            stage,
            chip=plan.chip,
            obstacles=fallback_obstacles,
            grid_step_um=config.grid_step_um,
            corridor_radius_um=corridor_radius,
            allowed_regions=None,
        )
        connector = simplify_orthogonal(first_leg + [entry])
    points.extend(connector[1:])
    base_length = polyline_length(points)
    remaining = target_length - base_length
    if remaining < -config.length_tolerance_fraction * target_length:
        raise LengthAllocationError(
            f"Bus mandatory route {base_length:.1f} um exceeds target {target_length:.1f} um"
        )
    remaining = max(0.0, remaining)
    first_horizontal = 1 if entry[0] == minx else -1
    serpentine = make_serpentine_exact(
        entry,
        remaining,
        bounds_um=(minx, miny, maxx, maxy),
        pitch_um=config.resonator_turn_pitch_um,
        first_horizontal=first_horizontal,
        vertical_direction=-1,
    )
    points.extend(serpentine[1:])
    points = simplify_orthogonal(points)
    route = _route_with_ground(
        route_id,
        bus.node_id,
        bus.label,
        ResonatorRole.COMMON_BUS.value,
        points,
        config,
        target_length_um=target_length,
        open_end=points[0],
        parameters={
            "lambda_fraction": 0.25,
            "coupling_waypoints_um": waypoints,
            "coupling_region_length_um": coupling_region_length,
            "coupling_region_fraction": coupling_region_length / target_length,
            "meander_after_last_coupling": True,
            "short_end": "grounded",
            "open_end": "first route point",
            "meander_bounds_um": [minx, miny, maxx, maxy],
        },
    )
    plan.routes[route_id] = route
    plan.coupling_windows.extend(windows)


def _readout_edges(graph: CircuitGraph, resonator_id: int) -> tuple[int, CircuitEdge | None, int | None, CircuitEdge | None]:
    transmon_id = None
    transmon_edge = None
    feedline_id = None
    feedline_edge = None
    for edge in graph.edges:
        if resonator_id not in (edge.a, edge.b):
            continue
        neighbour = other(edge, resonator_id)
        kind = graph.nodes[neighbour].kind
        if kind == NodeKind.TRANSMON:
            transmon_id = neighbour
            transmon_edge = edge
        elif kind == NodeKind.FEEDLINE:
            feedline_id = neighbour
            feedline_edge = edge
    if transmon_id is None:
        raise SynthesisError(f"Readout resonator {resonator_id} has no transmon neighbour")
    return transmon_id, transmon_edge, feedline_id, feedline_edge


def build_readouts(
    graph: CircuitGraph,
    placement: PlacementSolution,
    plan: LayoutPlan,
    config: SynthesisConfig,
) -> list[FeedlineRequirement]:
    requirements: list[FeedlineRequirement] = []
    readouts = [node for node in graph.nodes_of_kind(NodeKind.RESONATOR) if node.role == ResonatorRole.READOUT]
    for index, resonator in enumerate(sorted(readouts, key=lambda node: node.node_id)):
        transmon_id, transmon_edge, feedline_id, feedline_edge = _readout_edges(graph, resonator.node_id)
        component = plan.components[_owner_component(transmon_id)]
        assignment = placement.port_assignments.get(transmon_id, {})
        port_name = assignment.get("readout", "north")
        port = component.ports[port_name]
        outward = _unit(port.normal)
        tangent = _unit(port.tangent)
        tangent_sign = -1.0 if component.pose.x_um <= 0 else 1.0
        tangent = (tangent[0] * tangent_sign, tangent[1] * tangent_sign)

        coupling = transmon_edge.coupling_fF if transmon_edge else None
        tr_gap = edge_gap_um(transmon_edge) if transmon_edge else coupling_gap_um(coupling, "transmon_resonator_capacitive")
        open_start = _add(port.point_um, outward, tr_gap)

        rf_coupling = feedline_edge.coupling_fF if feedline_edge else None
        requested_parallel = edge_parallel_length_um(feedline_edge, config) if feedline_edge is not None else 0.0
        open_length = max(config.readout_open_segment_um, requested_parallel)
        coupling_origin = _add(open_start, outward, config.readout_open_lead_um)
        coupling_end = _add(coupling_origin, tangent, open_length)
        inward = (-outward[0], -outward[1])
        meander_entry = _add(coupling_end, inward, 190.0)

        target_length = resonator_length_um(
            resonator.attrs,
            default_frequency_GHz=config.default_readout_frequency_GHz,
            config=config,
        )
        route_id = _owner_route(resonator.node_id)
        window_tr = _make_coupling_window(
            f"window_readout_{transmon_id}_{resonator.node_id}",
            component.component_id,
            route_id,
            "transmon_readout_capacitive",
            [
                component.metal.intersection(Point(port.point_um).buffer(120.0)),
                LineString([open_start, _add(open_start, outward, min(130.0, config.readout_open_lead_um))]),
            ],
            target_gap_um=tr_gap,
            coupling_fF=coupling,
            margin_um=config.component_clearance_um + config.coupling_window_margin_um,
            parameters={"transmon_port": port_name, "resonator_end": "open"},
        )
        window_tr.actual_gap_um = tr_gap

        # The feedline coupling section is moved outward from the transmon.
        # At its far end the resonator turns back to the transmon side of the
        # feedline before entering a compact, laterally displaced meander.
        if tangent[0] > 0:
            minx = meander_entry[0]
            maxx = minx + config.readout_meander_width_um
            first_horizontal = 1
        else:
            maxx = meander_entry[0]
            minx = maxx - config.readout_meander_width_um
            first_horizontal = -1
        if inward[1] > 0:
            miny = meander_entry[1]
            maxy = miny + config.readout_meander_height_um
            vertical_direction = 1
        else:
            maxy = meander_entry[1]
            miny = maxy - config.readout_meander_height_um
            vertical_direction = -1

        base_points = [open_start, coupling_origin, coupling_end, meander_entry]
        remaining = target_length - polyline_length(base_points)
        serpentine = make_serpentine_exact(
            meander_entry,
            remaining,
            bounds_um=(minx, miny, maxx, maxy),
            pitch_um=config.resonator_turn_pitch_um,
            first_horizontal=first_horizontal,
            vertical_direction=vertical_direction,
        )
        points = simplify_orthogonal(base_points + serpentine[1:])
        route = _route_with_ground(
            route_id,
            resonator.node_id,
            resonator.label,
            ResonatorRole.READOUT.value,
            points,
            config,
            target_length_um=target_length,
            open_end=open_start,
            parameters={
                "lambda_fraction": 0.25,
                "transmon_id": transmon_id,
                "transmon_port": port_name,
                "open_segment_um": [coupling_origin, coupling_end],
                "transmon_open_lead_um": [open_start, coupling_origin],
                "meander_bounds_um": [minx, miny, maxx, maxy],
                "short_end": "grounded",
            },
        )
        plan.routes[route_id] = route
        plan.coupling_windows.append(window_tr)

        if feedline_id is not None:
            rf_gap = edge_gap_um(feedline_edge) if feedline_edge else coupling_gap_um(rf_coupling, "resonator_feedline_capacitive")
            parallel_length = open_length
            segment_start = coupling_origin
            segment_end = coupling_end
            requirements.append(
                FeedlineRequirement(
                    feedline_id=feedline_id,
                    resonator_route_id=route_id,
                    resonator_node_id=resonator.node_id,
                    segment_start_um=segment_start,
                    segment_end_um=segment_end,
                    outward_normal=outward,
                    target_gap_um=rf_gap,
                    coupling_fF=rf_coupling,
                    parallel_length_um=parallel_length,
                )
            )
    return requirements


def _mandatory_feedline_segment(
    requirement: FeedlineRequirement,
    config: SynthesisConfig,
) -> tuple[tuple[float, float], tuple[float, float], BaseGeometry]:
    center_offset = config.trace_width_um + requirement.target_gap_um
    start = _add(requirement.segment_start_um, requirement.outward_normal, center_offset)
    end = _add(requirement.segment_end_um, requirement.outward_normal, center_offset)
    tangent = _unit((end[0] - start[0], end[1] - start[1]))
    overhang_um = 45.0
    start = _add(start, tangent, -overhang_um)
    end = _add(end, tangent, overhang_um)
    if start[0] > end[0]:
        start, end = end, start
    resonator_segment = LineString([requirement.segment_start_um, requirement.segment_end_um])
    feed_segment = LineString([start, end])
    provisional = unary_union([resonator_segment, feed_segment]).convex_hull.buffer(
        config.route_clearance_um + config.coupling_window_margin_um,
        cap_style=2,
        join_style=2,
    )
    return start, end, provisional


def _add_native_launchpad_footprints(
    points: list[tuple[float, float]],
    metal: BaseGeometry,
    keepout: BaseGeometry,
    config: SynthesisConfig,
) -> tuple[BaseGeometry, BaseGeometry, dict[str, float]]:
    """Include the exact native launchpad footprints in planner validation."""

    if len(points) < 2:
        raise SynthesisError("A feedline requires at least two points")
    endpoint_neighbours = ((points[0], points[1]), (points[-1], points[-2]))
    metals = [metal]
    keepouts = [keepout]
    for endpoint, neighbour in endpoint_neighbours:
        interior = _unit((neighbour[0] - endpoint[0], neighbour[1] - endpoint[1]))
        launch_metal, launch_keepout = launchpad_wirebond_geometry(
            endpoint,
            interior,
            trace_width_um=config.trace_width_um,
            trace_gap_um=config.trace_gap_um,
            lead_length_um=config.launchpad_lead_length_um,
            pad_width_um=config.launchpad_pad_width_um,
            pad_height_um=config.launchpad_pad_height_um,
            pad_gap_um=config.launchpad_pad_gap_um,
            taper_height_um=config.launchpad_taper_height_um,
            clearance_um=config.route_clearance_um,
        )
        metals.append(launch_metal)
        keepouts.append(launch_keepout)
    options = {
        "lead_length_um": config.launchpad_lead_length_um,
        "pad_width_um": config.launchpad_pad_width_um,
        "pad_height_um": config.launchpad_pad_height_um,
        "pad_gap_um": config.launchpad_pad_gap_um,
        "taper_height_um": config.launchpad_taper_height_um,
    }
    return unary_union(metals), unary_union(keepouts), options


def build_feedlines(
    graph: CircuitGraph,
    requirements: list[FeedlineRequirement],
    plan: LayoutPlan,
    config: SynthesisConfig,
) -> None:
    grouped: dict[int, list[FeedlineRequirement]] = defaultdict(list)
    for requirement in requirements:
        grouped[requirement.feedline_id].append(requirement)

    for feedline_id, items in sorted(grouped.items()):
        feedline = graph.nodes[feedline_id]
        route_id = _owner_route(feedline_id)
        segments = []
        provisional_windows = []
        for requirement in items:
            start, end, provisional = _mandatory_feedline_segment(requirement, config)
            segments.append((requirement, start, end, provisional))
            provisional_windows.append(provisional)
        segments.sort(key=lambda item: (item[1][0], item[1][1]))
        allowed = window_union(provisional_windows)

        # A feedline connected to a single readout is synthesized as a local
        # two-launch U-shaped through line. Both legs leave the capacitive
        # coupling section in the resonator's outward direction and terminate
        # at the nearest chip edge. This avoids dragging every independent
        # feedline across the shared bus and all other qubits.
        if len(segments) == 1:
            requirement, start, end, provisional = segments[0]
            corridor_radius = config.trace_width_um / 2.0 + config.trace_gap_um + config.route_clearance_um
            chip_minx, chip_miny, chip_maxx, chip_maxy = plan.chip.bounds
            if abs(requirement.outward_normal[1]) >= abs(requirement.outward_normal[0]):
                boundary = chip_maxy - config.feedline_launch_inset_um if requirement.outward_normal[1] > 0 else chip_miny + config.feedline_launch_inset_um
                launch_a = (start[0], boundary)
                launch_b = (end[0], boundary)
            else:
                boundary = chip_maxx - config.feedline_launch_inset_um if requirement.outward_normal[0] > 0 else chip_minx + config.feedline_launch_inset_um
                launch_a = (boundary, start[1])
                launch_b = (boundary, end[1])

            routing_opening = provisional.buffer(corridor_radius, cap_style=2, join_style=2)
            obstacles = _routing_obstacles(plan, allowed_by_owner={requirement.resonator_route_id: routing_opening})
            direct_points = simplify_orthogonal([launch_a, start, end, launch_b])
            direct_corridor = LineString(direct_points).buffer(corridor_radius, cap_style=2, join_style=1)
            illegal = direct_corridor.intersection(obstacles)
            if illegal.is_empty or illegal.area <= config.geometry_area_tolerance_um2:
                points = direct_points
            else:
                leg_a = route_orthogonal_astar(
                    launch_a,
                    start,
                    chip=plan.chip,
                    obstacles=obstacles,
                    grid_step_um=config.grid_step_um,
                    corridor_radius_um=corridor_radius,
                    allowed_regions=None,
                )
                leg_b = route_orthogonal_astar(
                    end,
                    launch_b,
                    chip=plan.chip,
                    obstacles=obstacles,
                    grid_step_um=config.grid_step_um,
                    corridor_radius_um=corridor_radius,
                    allowed_regions=None,
                )
                points = simplify_orthogonal(leg_a + [end] + leg_b[1:])

            line, metal, keepout = route_geometry(
                points,
                config.trace_width_um,
                config.trace_gap_um,
                config.route_clearance_um,
            )
            metal, keepout, launchpad_options = _add_native_launchpad_footprints(
                points, metal, keepout, config
            )
            route = PhysicalRoute(
                route_id=route_id,
                node_id=feedline_id,
                label=feedline.label,
                role=NodeKind.FEEDLINE.value,
                centerline=line,
                metal=metal,
                keepout=keepout,
                trace_width_um=config.trace_width_um,
                trace_gap_um=config.trace_gap_um,
                target_length_um=None,
                grounded_end=None,
                open_end=launch_a,
                parameters={
                    "through_line": True,
                    "local_two_launch": True,
                    "launch_points_um": [launch_a, launch_b],
                    "launchpad_options_um": launchpad_options,
                    "coupled_resonators": [requirement.resonator_node_id],
                },
            )
            plan.routes[route_id] = route
            window = CouplingWindow(
                window_id=f"window_feedline_{requirement.resonator_node_id}_{feedline_id}",
                owner_a=requirement.resonator_route_id,
                owner_b=route_id,
                interface="resonator_feedline_capacitive",
                polygon=provisional,
                target_gap_um=requirement.target_gap_um,
                actual_gap_um=requirement.target_gap_um,
                coupling_fF=requirement.coupling_fF,
                parameters={
                    "parallel_length_um": requirement.parallel_length_um,
                    "feedline_segment_um": [start, end],
                    "resonator_segment_um": [requirement.segment_start_um, requirement.segment_end_um],
                },
            )
            plan.coupling_windows.append(window)
            continue

        chip_minx, _, chip_maxx, _ = plan.chip.bounds
        left = (chip_minx + config.feedline_launch_inset_um, segments[0][1][1])
        right = (chip_maxx - config.feedline_launch_inset_um, segments[-1][2][1])
        corridor_radius = config.trace_width_um / 2.0 + config.trace_gap_um + config.route_clearance_um
        allowed_by_owner: dict[str, BaseGeometry] = {}
        for requirement, _start, _end, provisional in segments:
            # The router subsequently buffers all obstacles by its own finite
            # corridor radius. Enlarge only the paired resonator's temporary
            # opening by the same amount so that this global buffer does not
            # close the intended capacitive-coupling channel.
            routing_opening = provisional.buffer(corridor_radius, cap_style=2, join_style=2)
            previous = allowed_by_owner.get(requirement.resonator_route_id)
            allowed_by_owner[requirement.resonator_route_id] = routing_opening if previous is None else unary_union([previous, routing_opening])
        obstacles = _routing_obstacles(plan, allowed_by_owner=allowed_by_owner)

        points: list[tuple[float, float]] = []
        connector = route_orthogonal_astar(
            left,
            segments[0][1],
            chip=plan.chip,
            obstacles=obstacles,
            grid_step_um=config.grid_step_um,
            corridor_radius_um=corridor_radius,
            allowed_regions=None,
        )
        points.extend(connector)
        current = segments[0][1]
        for index, (_requirement, start, end, _provisional) in enumerate(segments):
            if math.dist(current, start) > 1e-6:
                connector = route_orthogonal_astar(
                    current,
                    start,
                    chip=plan.chip,
                    obstacles=obstacles,
                    grid_step_um=config.grid_step_um,
                    corridor_radius_um=corridor_radius,
                    allowed_regions=None,
                )
                points.extend(connector[1:])
            if not points or math.dist(points[-1], start) > 1e-6:
                points.append(start)
            points.append(end)
            current = end
            if index + 1 < len(segments):
                next_start = segments[index + 1][1]
                connector = route_orthogonal_astar(
                    current,
                    next_start,
                    chip=plan.chip,
                    obstacles=obstacles,
                    grid_step_um=config.grid_step_um,
                    corridor_radius_um=corridor_radius,
                    allowed_regions=None,
                )
                points.extend(connector[1:])
                current = next_start
        connector = route_orthogonal_astar(
            current,
            right,
            chip=plan.chip,
            obstacles=obstacles,
            grid_step_um=config.grid_step_um,
            corridor_radius_um=corridor_radius,
            allowed_regions=None,
        )
        points.extend(connector[1:])
        points = simplify_orthogonal(points)
        line, metal, keepout = route_geometry(
            points,
            config.trace_width_um,
            config.trace_gap_um,
            config.route_clearance_um,
        )
        metal, keepout, launchpad_options = _add_native_launchpad_footprints(
            points, metal, keepout, config
        )
        route = PhysicalRoute(
            route_id=route_id,
            node_id=feedline_id,
            label=feedline.label,
            role=NodeKind.FEEDLINE.value,
            centerline=line,
            metal=metal,
            keepout=keepout,
            trace_width_um=config.trace_width_um,
            trace_gap_um=config.trace_gap_um,
            target_length_um=None,
            grounded_end=None,
            open_end=left,
            parameters={
                "through_line": True,
                "launch_points_um": [left, right],
                "launchpad_options_um": launchpad_options,
                "coupled_resonators": [item.resonator_node_id for item in items],
            },
        )
        plan.routes[route_id] = route

        for requirement, start, end, provisional in segments:
            window = CouplingWindow(
                window_id=f"window_feedline_{requirement.resonator_node_id}_{feedline_id}",
                owner_a=requirement.resonator_route_id,
                owner_b=route_id,
                interface="resonator_feedline_capacitive",
                polygon=provisional,
                target_gap_um=requirement.target_gap_um,
                actual_gap_um=requirement.target_gap_um,
                coupling_fF=requirement.coupling_fF,
                parameters={
                    "parallel_length_um": requirement.parallel_length_um,
                    "feedline_segment_um": [start, end],
                    "resonator_segment_um": [requirement.segment_start_um, requirement.segment_end_um],
                },
            )
            plan.coupling_windows.append(window)


def build_elements(
    graph: CircuitGraph,
    placement: PlacementSolution,
    plan: LayoutPlan,
    config: SynthesisConfig,
) -> None:
    """Build every supported physical element in dependency order.

    A new element type should be implemented as a builder in this file and added
    to this short composition function.
    """
    build_transmons(graph, placement, plan, config)
    build_bus(graph, placement, plan, config)
    feedline_requirements = build_readouts(graph, placement, plan, config)
    build_feedlines(graph, feedline_requirements, plan, config)
