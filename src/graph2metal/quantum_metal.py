"""Assemble a validated layout plan into a live Quantum Metal DesignPlanar."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from shapely import affinity, wkt
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Polygon, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from .elements import LayoutPlan, Pose, cross_polygon

try:
    from qiskit_metal import Dict as QMDict
    from qiskit_metal.qlibrary.core import QComponent
    QISKIT_METAL_IMPORTABLE = True
except Exception:  # Quantum Metal is an optional runtime dependency.
    QMDict = dict
    QComponent = object
    QISKIT_METAL_IMPORTABLE = False


class QuantumMetalUnavailable(RuntimeError):
    """Quantum Metal is not installed or incomplete."""


class QuantumMetalBuildError(RuntimeError):
    """The validated plan could not be represented safely in Quantum Metal."""


@dataclass(slots=True)
class QuantumMetalBuild:
    design: Any
    qm_module: Any
    plan_payload: dict[str, Any]
    object_map: dict[str, list[str]] = field(default_factory=dict)
    qgeometry_counts: dict[str, int] = field(default_factory=dict)
    coupling_windows: list[dict[str, Any]] = field(default_factory=list)


def quantum_metal_available() -> bool:
    try:
        import qiskit_metal  # noqa: F401
    except Exception:
        return False
    return True


def import_quantum_metal() -> dict[str, Any]:
    if not QISKIT_METAL_IMPORTABLE:
        raise QuantumMetalUnavailable(
            "Quantum Metal is required for PDF generation. Install this project "
            "with `pip install -e .[metal]`."
        )
    try:
        import qiskit_metal as qm
        from qiskit_metal import designs
        from qiskit_metal.qlibrary.qubits.transmon_cross import TransmonCross
        from qiskit_metal.qlibrary.terminations.launchpad_wb import LaunchpadWirebond
        from qiskit_metal.qlibrary.terminations.open_to_ground import OpenToGround
        from qiskit_metal.qlibrary.terminations.short_to_ground import ShortToGround
    except Exception as exc:
        raise QuantumMetalUnavailable(
            "Quantum Metal is installed, but a required component failed to import."
        ) from exc
    return {
        "qm": qm,
        "designs": designs,
        "TransmonCross": TransmonCross,
        "LaunchpadWirebond": LaunchpadWirebond,
        "OpenToGround": OpenToGround,
        "ShortToGround": ShortToGround,
        "PlannedCPW": PlannedCPW,
        "PlannedMetalPolygon": PlannedMetalPolygon,
    }


def mm(value_um: float) -> float:
    return float(value_um) * 1e-3


def as_um(value_um: float) -> str:
    return f"{float(value_um):.12g}um"


def as_mm(value_um: float) -> str:
    return f"{mm(value_um):.12g}mm"


def geometry_mm(geometry: BaseGeometry) -> BaseGeometry:
    return affinity.scale(geometry, xfact=1e-3, yfact=1e-3, origin=(0.0, 0.0))


def angle_deg(vector: tuple[float, float]) -> float:
    return math.degrees(math.atan2(vector[1], vector[0]))


def unit_vector(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    if length <= 1e-15:
        raise QuantumMetalBuildError(f"Degenerate route segment at {a}")
    return dx / length, dy / length


_JUNCTION_ARM_ROTATION_DEG = {
    "south": 0.0,
    "east": 90.0,
    "north": 180.0,
    "west": 270.0,
}


def transmon_qmetal_orientation_deg(
    pose_rotation_deg: float,
    junction_arm: str,
) -> float:
    """Return the Qiskit-Metal orientation that puts the JJ on ``junction_arm``.

    Quantum Metal's native ``TransmonCross`` draws ``rect_jj`` on the local
    south arm before applying ``orientation``.  The cross is four-fold
    symmetric, so adding a multiple of 90 degrees relocates only the semantic
    junction arm while preserving the validated cross footprint.
    """
    arm = str(junction_arm).lower()
    try:
        offset = _JUNCTION_ARM_ROTATION_DEG[arm]
    except KeyError as exc:
        raise QuantumMetalBuildError(
            f"Invalid junction_arm {junction_arm!r}; expected one of "
            f"{sorted(_JUNCTION_ARM_ROTATION_DEG)}."
        ) from exc
    return (float(pose_rotation_deg) + offset) % 360.0


def _validate_transmon_arm_roles(params: Mapping[str, Any], component_id: str) -> str:
    assignments = {
        str(role): str(arm).lower()
        for role, arm in dict(params.get("port_assignments", {})).items()
    }
    junction_arm = str(params.get("junction_arm", "south")).lower()
    occupied = set(assignments.values())
    if junction_arm in occupied:
        conflicting = sorted(role for role, arm in assignments.items() if arm == junction_arm)
        raise QuantumMetalBuildError(
            f"{component_id}: junction arm {junction_arm!r} is also assigned to "
            f"capacitive/direct role(s) {conflicting}."
        )
    return junction_arm


_ARM_VECTOR = {
    "north": (0.0, 1.0),
    "south": (0.0, -1.0),
    "east": (1.0, 0.0),
    "west": (-1.0, 0.0),
}


def _component_by_name(design: Any, name: str) -> Any | None:
    """Return a live QComponent without treating ``Components`` as a dict.

    ``design.components`` is Quantum Metal's ``Components`` interface.  It
    implements ``__getitem__`` but intentionally has no ``dict.get`` method.
    Accessing ``components.get`` is therefore interpreted as a request for a
    component literally named ``get`` and returns ``None``; calling that value
    causes ``TypeError: 'NoneType' object is not callable``.
    """
    interface = getattr(design, "components", None)
    if interface is not None:
        try:
            component = interface[name]
        except (KeyError, TypeError, AttributeError):
            component = None
        if component is not None:
            return component

    # Conservative fallback for lightweight test doubles and older Metal builds.
    name_to_id = getattr(design, "name_to_id", {})
    try:
        component_id = name_to_id[name]
    except (KeyError, TypeError):
        component_id = None
    if component_id is None:
        return None

    components = getattr(design, "_components", {})
    try:
        return components[component_id]
    except (KeyError, TypeError):
        return None


def _validate_live_junction_arms(design: Any, data: Mapping[str, Any]) -> None:
    """Verify that every live ``rect_jj`` is on its planned free arm."""
    tables = getattr(getattr(design, "qgeometry", None), "tables", {})
    try:
        junctions = tables["junction"]
    except (KeyError, TypeError):
        junctions = None
    if junctions is None:
        raise QuantumMetalBuildError("Quantum Metal did not create a junction QGeometry table.")

    for component in data.get("components", []):
        if component.get("kind") != "TRANSMON":
            continue
        name = str(component["component_id"])
        qcomponent = _component_by_name(design, name)
        if qcomponent is None:
            raise QuantumMetalBuildError(f"Missing live Quantum Metal component {name!r}.")
        numeric_id = getattr(qcomponent, "id", None)
        rows = junctions[junctions["component"] == numeric_id]
        if len(rows) != 1:
            raise QuantumMetalBuildError(
                f"{name}: expected exactly one live junction row, found {len(rows)}."
            )

        line = rows.iloc[0]["geometry"]
        midpoint = line.interpolate(0.5, normalized=True)
        pose = component["pose"]
        dx = float(midpoint.x) - mm(float(pose["x_um"]))
        dy = float(midpoint.y) - mm(float(pose["y_um"]))
        norm = math.hypot(dx, dy)
        if norm <= 1e-15:
            raise QuantumMetalBuildError(f"{name}: junction midpoint coincides with qubit centre.")

        junction_arm = str(component.get("parameters", {}).get("junction_arm", "south")).lower()
        ex, ey = _ARM_VECTOR[junction_arm]
        theta = math.radians(float(pose.get("rotation_deg", 0.0)))
        expected_x = ex * math.cos(theta) - ey * math.sin(theta)
        expected_y = ex * math.sin(theta) + ey * math.cos(theta)
        alignment = (dx * expected_x + dy * expected_y) / norm
        if alignment < 0.999:
            raise QuantumMetalBuildError(
                f"{name}: live junction is not on planned arm {junction_arm!r} "
                f"(direction alignment={alignment:.6f})."
            )

def _polygons(geometry: BaseGeometry):
    if geometry.is_empty:
        return
    if isinstance(geometry, Polygon):
        yield geometry
    elif isinstance(geometry, MultiPolygon):
        yield from geometry.geoms
    elif isinstance(geometry, GeometryCollection):
        for child in geometry.geoms:
            yield from _polygons(child)


class PlannedCPW(QComponent):
    """CPW whose validated centerline is supplied by graph2metal."""

    default_options = QMDict(
        path_wkt="LINESTRING EMPTY",
        trace_width="10um",
        trace_gap="6um",
        chip="main",
        layer="1",
    )
    component_metadata = QMDict(short_name="g2m_cpw", _qgeometry_table_path="True")

    @staticmethod
    def _pin(endpoint, neighbour, width):
        dx, dy = neighbour[0] - endpoint[0], neighbour[1] - endpoint[1]
        length = math.hypot(dx, dy)
        if length <= 1e-15:
            raise ValueError("Cannot make a pin from a zero-length segment")
        nx, ny, half = -dy / length, dx / length, width / 2.0
        return [
            (endpoint[0] - nx * half, endpoint[1] - ny * half),
            (endpoint[0] + nx * half, endpoint[1] + ny * half),
        ]

    def make(self):
        path = wkt.loads(str(self.options.path_wkt))
        if not isinstance(path, LineString) or len(path.coords) < 2:
            raise ValueError("PlannedCPW requires a non-degenerate LineString")
        p, chip, layer = self.p, str(self.options.chip), int(float(self.options.layer))
        self.add_qgeometry("path", {"trace": path}, width=p.trace_width, chip=chip, layer=layer)
        self.add_qgeometry(
            "path",
            {"clearance": path},
            width=p.trace_width + 2.0 * p.trace_gap,
            subtract=True,
            chip=chip,
            layer=layer,
        )
        coords = list(path.coords)
        self.add_pin("start", self._pin(coords[0], coords[1], p.trace_width), p.trace_width)
        self.add_pin("end", self._pin(coords[-1], coords[-2], p.trace_width), p.trace_width)


class PlannedMetalPolygon(QComponent):
    """Supplemental transmon metal and its local ground-plane etch."""

    default_options = QMDict(
        geometry_wkt="POLYGON EMPTY",
        etch_wkt="POLYGON EMPTY",
        chip="main",
        layer="1",
    )
    component_metadata = QMDict(short_name="g2m_poly", _qgeometry_table_poly="True")

    def make(self):
        chip, layer = str(self.options.chip), int(float(self.options.layer))
        metal = list(_polygons(wkt.loads(str(self.options.geometry_wkt))))
        etch = list(_polygons(wkt.loads(str(self.options.etch_wkt))))
        if metal:
            self.add_qgeometry(
                "poly", {f"metal_{i}": polygon for i, polygon in enumerate(metal)},
                chip=chip, layer=layer,
            )
        if etch:
            self.add_qgeometry(
                "poly", {f"etch_{i}": polygon for i, polygon in enumerate(etch)},
                subtract=True, chip=chip, layer=layer,
            )

def _payload(plan_or_payload: LayoutPlan | Mapping[str, Any]) -> dict[str, Any]:
    return plan_or_payload.to_dict() if isinstance(plan_or_payload, LayoutPlan) else dict(plan_or_payload)


def _qgeometry_counts(design: Any) -> dict[str, int]:
    tables = getattr(getattr(design, "qgeometry", None), "tables", {})
    counts: dict[str, int] = {}
    for name in ("poly", "path", "junction"):
        try:
            counts[name] = int(len(tables[name]))
        except Exception:
            pass
    return counts


def _instantiate_transmons(design, data, imports, object_map):
    TransmonCross = imports["TransmonCross"]
    PlannedMetalPolygon = imports["PlannedMetalPolygon"]

    for component in data.get("components", []):
        if component.get("kind") != "TRANSMON":
            continue
        component_id = str(component["component_id"])
        pose = component["pose"]
        params = component.get("parameters", {})
        arm_um = float(params.get("cross_arm_length_um", 150.0))
        width_um = float(params.get("cross_width_um", 30.0))
        gap_um = float(params.get("cross_gap_um", 20.0))
        junction_arm = _validate_transmon_arm_roles(params, component_id)
        qmetal_orientation = transmon_qmetal_orientation_deg(
            float(pose.get("rotation_deg", 0.0)), junction_arm
        )
        options = {
            "pos_x": as_mm(float(pose["x_um"])),
            "pos_y": as_mm(float(pose["y_um"])),
            "orientation": str(qmetal_orientation),
            "cross_length": as_um(arm_um),
            "cross_width": as_um(width_um),
            "cross_gap": as_um(gap_um),
            "connection_pads": {},
            "chip": "main",
        }
        lj_nh = params.get("Lj_nH")
        if lj_nh is not None:
            options["hfss_inductance"] = f"{float(lj_nh):.12g}nH"
            options["q3d_inductance"] = f"{float(lj_nh):.12g}nH"

        transmon = TransmonCross(design, component_id, options=options)
        try:
            transmon.metadata["graph2metal"] = {
                "node_id": component.get("node_id"),
                "role": component.get("role"),
                "capacitance_fF": params.get("capacitance_fF"),
                "Lj_nH": lj_nh,
                "rigid_group": component.get("group_id"),
                "junction_arm": junction_arm,
                "qmetal_orientation_deg": qmetal_orientation,
            }
        except Exception:
            pass
        object_map.setdefault(component_id, []).append(component_id)

        planned = shape(component["metal"])
        bare = cross_polygon(
            arm_um,
            width_um,
            Pose(float(pose["x_um"]), float(pose["y_um"]), float(pose.get("rotation_deg", 0.0))),
        )
        extra = planned.difference(bare).buffer(0)
        if extra.is_empty or extra.area <= 1e-6:
            continue
        extra_name = f"{component_id}__direct_coupling_metal"
        PlannedMetalPolygon(
            design,
            extra_name,
            options={
                "geometry_wkt": wkt.dumps(geometry_mm(extra), rounding_precision=12),
                "etch_wkt": wkt.dumps(
                    geometry_mm(extra.buffer(gap_um, cap_style=2, join_style=2)),
                    rounding_precision=12,
                ),
                "chip": "main",
                "layer": "1",
            },
        )
        object_map[component_id].append(extra_name)


def _instantiate_routes(design, data, imports, object_map):
    PlannedCPW = imports["PlannedCPW"]
    OpenToGround = imports["OpenToGround"]
    ShortToGround = imports["ShortToGround"]
    LaunchpadWirebond = imports["LaunchpadWirebond"]

    for route in data.get("routes", []):
        route_id = str(route["route_id"])
        centerline_um = shape(route["centerline"])
        if not isinstance(centerline_um, LineString) or len(centerline_um.coords) < 2:
            raise QuantumMetalBuildError(f"Route {route_id} has no valid centerline")
        trace_width_um = float(route["trace_width_um"])
        trace_gap_um = float(route["trace_gap_um"])
        cpw = PlannedCPW(
            design,
            route_id,
            options={
                "path_wkt": wkt.dumps(geometry_mm(centerline_um), rounding_precision=12),
                "trace_width": as_um(trace_width_um),
                "trace_gap": as_um(trace_gap_um),
                "chip": "main",
                "layer": "1",
            },
        )
        try:
            cpw.metadata["graph2metal"] = {
                "node_id": route.get("node_id"),
                "role": route.get("role"),
                "target_length_um": route.get("target_length_um"),
                "actual_length_um": route.get("actual_length_um"),
                "planner_parameters": route.get("parameters", {}),
            }
        except Exception:
            pass
        object_map.setdefault(route_id, []).append(route_id)

        coords = [(float(x), float(y)) for x, y in centerline_um.coords]
        role = str(route.get("role", ""))
        if route.get("grounded_end") is not None:
            endpoint = coords[-1]
            short_name = f"{route_id}__ground_short"
            short = ShortToGround(
                design,
                short_name,
                options={
                    "pos_x": as_mm(endpoint[0]),
                    "pos_y": as_mm(endpoint[1]),
                    "orientation": str(angle_deg(unit_vector(coords[-2], coords[-1]))),
                    "width": as_um(trace_width_um),
                    "chip": "main",
                    "layer": "1",
                },
            )
            object_map[route_id].append(short_name)
            try:
                design.connect_pins(cpw.id, "end", short.id, "short")
            except Exception:
                pass

        if route.get("open_end") is not None and role in {"COMMON_BUS", "READOUT"}:
            endpoint = coords[0]
            interior = unit_vector(coords[0], coords[1])
            open_name = f"{route_id}__open_end"
            opening = OpenToGround(
                design,
                open_name,
                options={
                    "pos_x": as_mm(endpoint[0]),
                    "pos_y": as_mm(endpoint[1]),
                    "orientation": str(angle_deg((-interior[0], -interior[1]))),
                    "width": as_um(trace_width_um),
                    "gap": as_um(trace_gap_um),
                    "termination_gap": as_um(trace_gap_um),
                    "chip": "main",
                    "layer": "1",
                },
            )
            object_map[route_id].append(open_name)
            try:
                design.connect_pins(cpw.id, "start", opening.id, "open")
            except Exception:
                pass

        if role == "FEEDLINE":
            params = route.get("parameters", {})
            options = params.get("launchpad_options_um", {})
            lead_um = float(options.get("lead_length_um", 25.0))
            endpoints, neighbours, pins = [coords[0], coords[-1]], [coords[1], coords[-2]], ["start", "end"]
            for index, (endpoint, neighbour, pin_name) in enumerate(zip(endpoints, neighbours, pins), 1):
                interior = unit_vector(endpoint, neighbour)
                origin = (endpoint[0] - lead_um * interior[0], endpoint[1] - lead_um * interior[1])
                launch_name = f"{route_id}__launch_{index}"
                launch = LaunchpadWirebond(
                    design,
                    launch_name,
                    options={
                        "pos_x": as_mm(origin[0]),
                        "pos_y": as_mm(origin[1]),
                        "orientation": str(angle_deg(interior)),
                        "trace_width": as_um(trace_width_um),
                        "trace_gap": as_um(trace_gap_um),
                        "lead_length": as_um(lead_um),
                        "pad_width": as_um(float(options.get("pad_width_um", 80.0))),
                        "pad_height": as_um(float(options.get("pad_height_um", 80.0))),
                        "pad_gap": as_um(float(options.get("pad_gap_um", 58.0))),
                        "taper_height": as_um(float(options.get("taper_height_um", 122.0))),
                        "chip": "main",
                        "layer": "1",
                    },
                )
                object_map[route_id].append(launch_name)
                try:
                    design.connect_pins(cpw.id, pin_name, launch.id, "tie")
                except Exception:
                    pass


def build_quantum_metal_design(
    plan_or_payload: LayoutPlan | Mapping[str, Any], *, enable_renderers: bool = True
) -> QuantumMetalBuild:
    """Instantiate native/custom QComponents and rebuild the DesignPlanar."""
    imports = import_quantum_metal()
    data = _payload(plan_or_payload)
    violations = data.get("violations", [])
    if violations:
        raise QuantumMetalBuildError(
            f"Refusing to build an invalid plan ({len(violations)} violation(s))."
        )

    try:
        design = imports["designs"].DesignPlanar(
            metadata={
                "design_name": data.get("name", "graph2metal_design"),
                "graph2metal_schema": data.get("schema_version"),
            },
            overwrite_enabled=True,
            enable_renderers=enable_renderers,
        )
    except TypeError:
        design = imports["designs"].DesignPlanar()
        design.overwrite_enabled = True

    chip = data["chip"]
    design.chips.main.size.size_x = as_mm(float(chip["width_um"]))
    design.chips.main.size.size_y = as_mm(float(chip["height_um"]))
    routes = data.get("routes", [])
    if routes:
        try:
            design.variables["cpw_width"] = as_um(float(routes[0].get("trace_width_um", 10.0)))
            design.variables["cpw_gap"] = as_um(float(routes[0].get("trace_gap_um", 6.0)))
        except Exception:
            pass

    object_map: dict[str, list[str]] = {}
    _instantiate_transmons(design, data, imports, object_map)
    _instantiate_routes(design, data, imports, object_map)
    rebuild = getattr(design, "rebuild", None)
    if callable(rebuild):
        rebuild()
    _validate_live_junction_arms(design, data)
    counts = _qgeometry_counts(design)
    if counts and counts.get("path", 0) < len(routes):
        raise QuantumMetalBuildError("Quantum Metal contains fewer route paths than the plan.")
    return QuantumMetalBuild(
        design=design,
        qm_module=imports["qm"],
        plan_payload=data,
        object_map=object_map,
        qgeometry_counts=counts,
        coupling_windows=[dict(window) for window in data.get("coupling_windows", [])],
    )


# ---- Post-build copper-overlap validation --------------------------------

def rendered_geometry_by_object(
    design: Any, object_map: Mapping[str, list[str]]
) -> dict[str, BaseGeometry]:
    native_to_logical = {
        native: logical for logical, names in object_map.items() for native in names
    }
    id_to_logical = {
        component.id: native_to_logical[name]
        for name, component in getattr(design, "components", {}).items()
        if name in native_to_logical
    }
    grouped: dict[str, list[BaseGeometry]] = {}
    tables = getattr(getattr(design, "qgeometry", None), "tables", {})

    poly_table = tables.get("poly")
    if poly_table is not None:
        for _, row in poly_table.iterrows():
            if row.get("subtract") or row.get("helper"):
                continue
            logical = id_to_logical.get(row.get("component"))
            if logical is not None:
                grouped.setdefault(logical, []).append(row["geometry"])

    path_table = tables.get("path")
    if path_table is not None:
        for _, row in path_table.iterrows():
            if row.get("subtract") or row.get("helper"):
                continue
            logical = id_to_logical.get(row.get("component"))
            if logical is None:
                continue
            geometry = row["geometry"]
            width = row.get("width")
            if width:
                geometry = geometry.buffer(float(width) / 2.0, cap_style=2)
            grouped.setdefault(logical, []).append(geometry)

    return {logical: unary_union(parts) for logical, parts in grouped.items()}


def geometry_conflicts(
    build: "QuantumMetalBuild", *, area_tolerance_um2: float = 1e-3
) -> list[tuple[str, str, float]]:
    """Return copper overlaps between different logical objects."""
    objects = rendered_geometry_by_object(build.design, build.object_map)
    ids = list(objects)
    conflicts: list[tuple[str, str, float]] = []
    for index, object_a in enumerate(ids):
        for object_b in ids[index + 1 :]:
            area_um2 = objects[object_a].intersection(objects[object_b]).area * 1e6
            if area_um2 > area_tolerance_um2:
                conflicts.append((object_a, object_b, area_um2))
    return conflicts
