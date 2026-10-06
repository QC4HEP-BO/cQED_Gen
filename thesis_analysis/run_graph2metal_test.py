#!/usr/bin/env python3
"""Smoke test for the compact graph2metal package with real chip/PDF auto-fit.

This version fixes two separate issues:

1. Chip auto-fit is computed from the *planned circuit geometry only*.
   The previous auto-fit accidentally included the chip polygon itself, so a
   12 x 10 mm probe chip inevitably produced a ~12 x 10 mm fitted chip.

2. The final Quantum Metal PDF is explicitly zoomed to the rendered circuit
   geometry. qiskit_metal.view() otherwise tends to keep the full chip in the
   axes, which can leave a very large amount of empty space even when the
   physical chip dimensions have been reduced.

Run from the repository root, where src/graph2metal/ is available.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Callable

import networkx as nx


REQUIRED_MODULES = (
    "graph_model.py",
    "elements.py",
    "graph_to_design.py",
    "quantum_metal.py",
    "plot_layout.py",
)


def configure_import_path(repo_root: Path) -> Path:
    """Make graph2metal importable from <repo>/src."""
    src_dir = (repo_root / "src").resolve()
    module_dir = src_dir / "graph2metal"

    missing = [name for name in REQUIRED_MODULES if not (module_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing compact graph2metal file(s) in {module_dir}: {', '.join(missing)}"
        )

    src_text = str(src_dir)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)
    return module_dir


def add_transmon(graph: nx.Graph, node_id: int, *, capacitance_fF: float = 80.0) -> None:
    graph.add_node(
        node_id,
        type="TRANSMON",
        label=f"Q{node_id}",
        C=capacitance_fF * 1e-15,
        L=10.0e-9,
    )


def add_resonator(
    graph: nx.Graph,
    node_id: int,
    *,
    label: str,
    frequency_GHz: float,
    role: str,
) -> None:
    graph.add_node(
        node_id,
        type="RESONATOR",
        label=label,
        role=role,
        target_frequency_GHz=frequency_GHz,
        epsilon_eff=6.0,
    )


def add_feedline(graph: nx.Graph, node_id: int, *, label: str) -> None:
    graph.add_node(node_id, type="FEEDLINE", label=label)


def add_capacitive_coupler(
    graph: nx.Graph,
    coupler_id: int,
    endpoint_a: int,
    endpoint_b: int,
    *,
    coupling_fF: float,
    gap_um: float | None = None,
    parallel_length_um: float | None = None,
) -> None:
    attrs: dict[str, float | str] = {
        "type": "C_COUPLER",
        "label": f"C{coupler_id}",
        "Cc": coupling_fF * 1e-15,
    }
    if gap_um is not None:
        attrs["gap_um"] = gap_um
    if parallel_length_um is not None:
        attrs["parallel_length_um"] = parallel_length_um

    graph.add_node(coupler_id, **attrs)
    graph.add_edge(endpoint_a, coupler_id)
    graph.add_edge(coupler_id, endpoint_b)


def make_single() -> nx.Graph:
    graph = nx.Graph(name="graph2metal_test_single")
    add_transmon(graph, 0)
    return graph


def make_qbus() -> nx.Graph:
    graph = nx.Graph(name="graph2metal_test_qbus")
    add_transmon(graph, 0)
    add_resonator(graph, 10, label="BUS", frequency_GHz=6.0, role="COMMON_BUS")
    add_capacitive_coupler(graph, 100, 0, 10, coupling_fF=5.5)
    return graph


def make_four_bus() -> nx.Graph:
    graph = nx.Graph(name="graph2metal_test_four_bus")
    bus_id = 10
    add_resonator(graph, bus_id, label="BUS", frequency_GHz=6.0, role="COMMON_BUS")

    for qubit_id in range(4):
        add_transmon(graph, qubit_id, capacitance_fF=80.0 + 2.0 * qubit_id)
        add_capacitive_coupler(
            graph,
            100 + qubit_id,
            qubit_id,
            bus_id,
            coupling_fF=5.0 + 0.25 * qubit_id,
        )
    return graph


def make_four_full() -> nx.Graph:
    graph = make_four_bus()
    graph.graph["name"] = "graph2metal_test_four_full"

    next_coupler_id = 200
    for qubit_id in range(4):
        readout_id = 20 + qubit_id
        feedline_id = 30 + qubit_id

        add_resonator(
            graph,
            readout_id,
            label=f"R{qubit_id}",
            frequency_GHz=6.7 + 0.1 * qubit_id,
            role="READOUT",
        )
        add_feedline(graph, feedline_id, label=f"F{qubit_id}")

        add_capacitive_coupler(
            graph,
            next_coupler_id,
            qubit_id,
            readout_id,
            coupling_fF=4.0,
        )
        next_coupler_id += 1

        add_capacitive_coupler(
            graph,
            next_coupler_id,
            readout_id,
            feedline_id,
            coupling_fF=2.5,
            parallel_length_um=180.0,
        )
        next_coupler_id += 1

    return graph


CASE_FACTORIES: dict[str, Callable[[], nx.Graph]] = {
    "single": make_single,
    "qbus": make_qbus,
    "four_bus": make_four_bus,
    "four_full": make_four_full,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an in-memory graph2metal test with automatic chip and PDF fitting."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Repository root containing src/graph2metal/ (default: script directory).",
    )
    parser.add_argument(
        "--case",
        choices=tuple(CASE_FACTORIES),
        default="four_bus",
        help="Test topology (default: four_bus).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: <repo>/runs/graph2metal_test_<case>).",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Create only JSON plan files; do not render Quantum Metal PDF.",
    )
    parser.add_argument(
        "--non-strict",
        action="store_true",
        help="Keep the best layout even if planner validation reports violations.",
    )
    parser.add_argument(
        "--allow-qgeometry-conflicts",
        action="store_true",
        help="Do not abort if rendered Quantum Metal geometry contains overlaps.",
    )

    # None = auto-fit that axis.
    parser.add_argument("--chip-width-um", type=float, default=None)
    parser.add_argument("--chip-height-um", type=float, default=None)

    parser.add_argument(
        "--chip-margin-um",
        type=float,
        default=250.0,
        help=(
            "Physical free margin between circuit keepout and chip edge. "
            "It is also passed to the planner as chip_margin_um (default: 250 um)."
        ),
    )
    parser.add_argument(
        "--plot-margin-um",
        type=float,
        default=None,
        help=(
            "Margin around the circuit in layout_zero.pdf. "
            "Default: same as --chip-margin-um."
        ),
    )
    parser.add_argument(
        "--min-chip-width-um",
        type=float,
        default=1200.0,
        help="Minimum auto-fitted chip width (default: 1200 um).",
    )
    parser.add_argument(
        "--min-chip-height-um",
        type=float,
        default=900.0,
        help="Minimum auto-fitted chip height (default: 900 um).",
    )
    parser.add_argument(
        "--fit-step-um",
        type=float,
        default=50.0,
        help="Round auto-fitted chip dimensions up to this step (default: 50 um).",
    )
    parser.add_argument(
        "--no-tight-pdf",
        action="store_true",
        help="Keep Quantum Metal's default full-chip PDF view.",
    )
    return parser.parse_args()


def _round_up(value: float, step: float) -> float:
    if step <= 0:
        return float(value)
    return math.ceil(float(value) / float(step)) * float(step)


def _geometry_bounds(plan, *, use_keepout: bool) -> tuple[float, float, float, float]:
    """Return bounds of actual circuit objects only, never the chip polygon.

    Bounds are in micrometres.  This is the important fix: the old auto-fit
    recursively read layout_plan.json and therefore also consumed
    chip.polygon.coordinates.  The probe chip itself then dominated the bbox.
    """
    bounds: list[tuple[float, float, float, float]] = []

    for component in plan.components.values():
        geom = component.keepout if use_keepout else component.metal
        if geom is not None and not geom.is_empty:
            bounds.append(tuple(map(float, geom.bounds)))

    for route in plan.routes.values():
        geom = route.keepout if use_keepout else route.metal
        if geom is not None and not geom.is_empty:
            bounds.append(tuple(map(float, geom.bounds)))

    # Coupling windows can extend a little beyond the metal itself. Include
    # them for chip sizing, but not for the visual crop unless explicitly
    # requested through use_keepout=True.
    if use_keepout:
        for window in plan.coupling_windows:
            geom = window.polygon
            if geom is not None and not geom.is_empty:
                bounds.append(tuple(map(float, geom.bounds)))

    if not bounds:
        raise RuntimeError("No physical geometry was found in the synthesized plan.")

    xmin = min(item[0] for item in bounds)
    ymin = min(item[1] for item in bounds)
    xmax = max(item[2] for item in bounds)
    ymax = max(item[3] for item in bounds)
    return xmin, ymin, xmax, ymax


def _fitted_chip_size(
    bounds: tuple[float, float, float, float],
    *,
    margin_um: float,
    min_width_um: float,
    min_height_um: float,
    step_um: float,
) -> tuple[float, float]:
    """Fit a chip centred at (0,0) around the circuit keepout geometry."""
    xmin, ymin, xmax, ymax = bounds

    # LayoutPlan chips are centred at the origin. Therefore width/height must
    # cover the largest absolute coordinate, not merely xmax-xmin/ymax-ymin.
    half_w = max(abs(xmin), abs(xmax)) + margin_um
    half_h = max(abs(ymin), abs(ymax)) + margin_um

    # Tiny numerical safety allowance so geometry does not land exactly on the
    # planner's usable-area boundary.
    safety_um = 25.0
    width = max(min_width_um, 2.0 * (half_w + safety_um))
    height = max(min_height_um, 2.0 * (half_h + safety_um))
    return _round_up(width, step_um), _round_up(height, step_um)


def _tight_render_layout_pdf(plan, target: Path, *, margin_um: float) -> None:
    """Re-render layout_zero.pdf with axes fitted to the actual metal geometry."""
    import matplotlib.pyplot as plt

    from graph2metal.quantum_metal import build_quantum_metal_design
    from graph2metal.plot_layout import _main_axis, remove_dashed_outlines

    build = build_quantum_metal_design(plan)
    figure = build.qm_module.view(build.design)
    if figure is None:
        raise RuntimeError("qiskit_metal.view(design) returned no Figure")

    remove_dashed_outlines(figure)
    axis = _main_axis(figure)

    xmin, ymin, xmax, ymax = _geometry_bounds(plan, use_keepout=False)
    xmin -= margin_um
    xmax += margin_um
    ymin -= margin_um
    ymax += margin_um

    # Quantum Metal's matplotlib view uses millimetres.
    scale = 1e-3
    axis.set_xlim(xmin * scale, xmax * scale)
    axis.set_ylim(ymin * scale, ymax * scale)
    axis.set_aspect("equal", adjustable="box")

    # Match the PDF page aspect ratio to the geometry. This removes the second
    # source of whitespace: a square/default Figure around a very wide layout.
    span_x = max(xmax - xmin, 1.0)
    span_y = max(ymax - ymin, 1.0)
    aspect = span_y / span_x
    fig_w = 9.0
    fig_h = max(2.4, min(9.0, fig_w * aspect + 0.55))
    figure.set_size_inches(fig_w, fig_h, forward=True)

    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    module_dir = configure_import_path(repo_root)
    os.environ.setdefault("MPLBACKEND", "Agg")

    from graph2metal.graph_model import SynthesisConfig
    from graph2metal.graph_to_design import translate_graph
    from graph2metal.plot_layout import generate_layout_zero

    graph = CASE_FACTORIES[args.case]()
    output_dir = (
        args.output.resolve()
        if args.output is not None
        else repo_root / "runs" / f"graph2metal_test_{args.case}"
    )

    print("=== graph2metal compact test: REAL AUTO-FIT ===")
    print(f"modules      : {module_dir}")
    print(f"case         : {args.case}")
    print(f"nodes/edges  : {graph.number_of_nodes()}/{graph.number_of_edges()}")
    print(f"output       : {output_dir}")

    # First synthesize on a deliberately generous chip. We inspect the plan
    # object directly, so the chip polygon itself can never contaminate bbox.
    probe_config = SynthesisConfig(
        chip_width_um=12000.0,
        chip_height_um=10000.0,
        chip_margin_um=args.chip_margin_um,
    )
    probe = translate_graph(graph, config=probe_config, strict=not args.non_strict)
    probe_bounds = _geometry_bounds(probe.plan, use_keepout=True)

    fit_w, fit_h = _fitted_chip_size(
        probe_bounds,
        margin_um=args.chip_margin_um,
        min_width_um=args.min_chip_width_um,
        min_height_um=args.min_chip_height_um,
        step_um=args.fit_step_um,
    )

    chip_w = float(args.chip_width_um if args.chip_width_um is not None else fit_w)
    chip_h = float(args.chip_height_um if args.chip_height_um is not None else fit_h)

    # One refinement pass: a smaller chip can slightly change route placement.
    # Re-fit once from that new plan, then use the resulting dimensions.
    trial_config = SynthesisConfig(
        chip_width_um=chip_w,
        chip_height_um=chip_h,
        chip_margin_um=args.chip_margin_um,
    )
    trial = translate_graph(graph, config=trial_config, strict=not args.non_strict)
    trial_bounds = _geometry_bounds(trial.plan, use_keepout=True)
    fit_w2, fit_h2 = _fitted_chip_size(
        trial_bounds,
        margin_um=args.chip_margin_um,
        min_width_um=args.min_chip_width_um,
        min_height_um=args.min_chip_height_um,
        step_um=args.fit_step_um,
    )

    if args.chip_width_um is None:
        chip_w = fit_w2
    if args.chip_height_um is None:
        chip_h = fit_h2

    print(
        "geometry bbox : "
        f"x=[{trial_bounds[0]:.0f}, {trial_bounds[2]:.0f}] um, "
        f"y=[{trial_bounds[1]:.0f}, {trial_bounds[3]:.0f}] um"
    )
    print(f"requested chip : {chip_w:.0f} x {chip_h:.0f} um")
    print(f"chip margin    : {args.chip_margin_um:.0f} um")

    final_config = SynthesisConfig(
        chip_width_um=chip_w,
        chip_height_um=chip_h,
        chip_margin_um=args.chip_margin_um,
    )

    result = generate_layout_zero(
        graph,
        output_dir,
        config=final_config,
        strict=not args.non_strict,
        strict_geometry_check=not args.allow_qgeometry_conflicts,
        plan_only=args.plan_only,
    )

    plan = result.translation.plan
    print("\n=== result ===")
    print(f"valid              : {result.translation.valid}")
    print(f"synthesis_attempt  : {plan.synthesis_attempt}")
    print(f"actual chip        : {plan.chip_width_um:.0f} x {plan.chip_height_um:.0f} um")
    print(f"components         : {len(plan.components)}")
    print(f"routes             : {len(plan.routes)}")
    print(f"coupling_windows   : {len(plan.coupling_windows)}")
    print(f"violations         : {len(plan.violations)}")
    print(f"source_graph_json  : {result.source_graph_json}")
    print(f"layout_plan_json   : {result.layout_plan_json}")

    if result.translation.attempt_errors:
        print("\nSynthesis retries/diagnostics:")
        for message in result.translation.attempt_errors:
            print(f"  - {message}")

    if plan.violations:
        print("\nPlanner violations:")
        for violation in plan.violations:
            print(f"  - {violation.code}: {violation.message}")

    if result.layout_pdf is not None:
        if not args.no_tight_pdf:
            plot_margin_um = (
                args.chip_margin_um if args.plot_margin_um is None else args.plot_margin_um
            )
            _tight_render_layout_pdf(
                plan,
                result.layout_pdf,
                margin_um=float(plot_margin_um),
            )
            print(f"tight PDF margin   : {plot_margin_um:.0f} um")

        print(f"layout_pdf         : {result.layout_pdf}")
        print(f"render_json        : {result.render_json}")
        print(f"coupler_zoom_pdfs  : {len(result.artifacts.coupler_zoom_pdfs)}")

    print("\nPASS" if result.translation.valid else "\nCOMPLETED WITH VIOLATIONS")
    return 0 if result.translation.valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
