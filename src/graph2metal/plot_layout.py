"""Render layout zero and expose the compact end-to-end command line interface."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from shapely.geometry import shape

from .graph_model import SynthesisConfig, graph_to_payload, load_graph
from .graph_to_design import TranslationResult, translate_graph
from .quantum_metal import (
    QuantumMetalBuild, QuantumMetalBuildError, build_quantum_metal_design,
    geometry_conflicts,
)


@dataclass(slots=True)
class QuantumMetalArtifacts:
    layout_pdf: Path
    render_json: Path
    coupler_zoom_pdfs: dict[str, Path] = field(default_factory=dict)
    rendered_geometry_conflicts: list[tuple[str, str, float]] = field(default_factory=list)


@dataclass(slots=True)
class GenerationResult:
    translation: TranslationResult
    output_dir: Path
    layout_plan_json: Path
    source_graph_json: Path
    artifacts: QuantumMetalArtifacts | None = None

    @property
    def layout_pdf(self) -> Path | None:
        return None if self.artifacts is None else self.artifacts.layout_pdf

    @property
    def render_json(self) -> Path | None:
        return None if self.artifacts is None else self.artifacts.render_json


def _has_dashes(style: Any) -> bool:
    if isinstance(style, str):
        return style.strip().lower() in {"--", ":", "-.", "dashed", "dotted", "dashdot"}
    if isinstance(style, tuple) and len(style) == 2:
        sequence = style[1]
        try:
            return sequence is not None and len(sequence) > 0
        except TypeError:
            return bool(sequence)
    return False


def remove_dashed_outlines(figure: Any) -> int:
    """Hide dashed polygon boundaries without changing Quantum Metal geometry."""
    removed = 0
    for axis in getattr(figure, "axes", []):
        for line in getattr(axis, "lines", []):
            try:
                x, y = line.get_xdata(), line.get_ydata()
                closed = len(x) > 2 and len(y) > 2 and x[0] == x[-1] and y[0] == y[-1]
                if closed and _has_dashes(line.get_linestyle()):
                    line.set_visible(False)
                    removed += 1
            except Exception:
                pass
        for patch in getattr(axis, "patches", []):
            try:
                if _has_dashes(patch.get_linestyle()):
                    patch.set_linestyle("-")
                    patch.set_edgecolor("none")
                    patch.set_linewidth(0.0)
                    removed += 1
            except Exception:
                pass
        for collection in getattr(axis, "collections", []):
            try:
                if any(_has_dashes(style) for style in collection.get_linestyles()):
                    collection.set_linestyles("solid")
                    collection.set_edgecolor("none")
                    collection.set_linewidths(0.0)
                    removed += 1
            except Exception:
                pass
    return removed


def quantum_metal_figure(qm_module: Any, design: Any) -> tuple[Any, int]:
    figure = qm_module.view(design)
    if figure is None:
        raise RuntimeError("qiskit_metal.view(design) returned no Figure")
    return figure, remove_dashed_outlines(figure)


def close_figure(figure: Any) -> None:
    try:
        import matplotlib.pyplot as plt
        plt.close(figure)
    except Exception:
        pass


def save_layout_pdf(qm_module: Any, design: Any, target: Path) -> int:
    figure, removed = quantum_metal_figure(qm_module, design)
    figure.savefig(target, bbox_inches="tight")
    close_figure(figure)
    return removed


def _main_axis(figure: Any) -> Any:
    axes = [axis for axis in getattr(figure, "axes", []) if axis.has_data()]
    axes = axes or list(getattr(figure, "axes", []))
    if not axes:
        raise RuntimeError("Quantum Metal view has no axes")
    return max(
        axes,
        key=lambda axis: abs(
            (axis.get_xlim()[1] - axis.get_xlim()[0])
            * (axis.get_ylim()[1] - axis.get_ylim()[0])
        ),
    )


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "coupler"


def _zoom_limits(window: Mapping[str, Any], margin_fraction: float, min_margin_um: float):
    polygon = shape(window["polygon"])
    if polygon.is_empty:
        raise RuntimeError(f"Empty coupling window: {window.get('window_id')}")
    min_x, min_y, max_x, max_y = polygon.bounds
    width, height = max(max_x - min_x, 1.0), max(max_y - min_y, 1.0)
    margin = max(min_margin_um, margin_fraction * max(width, height))
    center_x, center_y = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0
    half = max(width, height) / 2.0 + margin
    scale = 1e-3
    return (
        ((center_x - half) * scale, (center_x + half) * scale),
        ((center_y - half) * scale, (center_y + half) * scale),
    )


def save_coupler_pdfs(
    qm_module: Any,
    design: Any,
    windows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    margin_fraction: float = 0.20,
    min_margin_um: float = 40.0,
) -> tuple[dict[str, Path], int]:
    if not windows:
        return {}, 0
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    removed_total = 0
    for index, window in enumerate(windows, 1):
        window_id = str(window.get("window_id", f"coupler_{index}"))
        figure, removed = quantum_metal_figure(qm_module, design)
        removed_total += removed
        axis = _main_axis(figure)
        xlim, ylim = _zoom_limits(window, margin_fraction, min_margin_um)
        axis.set_xlim(*xlim)
        axis.set_ylim(*ylim)
        axis.set_aspect("equal", adjustable="box")
        target = output_dir / f"{index:02d}_{_safe_name(window_id)}.pdf"
        figure.savefig(target, bbox_inches="tight")
        close_figure(figure)
        outputs[window_id] = target
    return outputs, removed_total


def _component_summary(component: dict[str, Any], native_names: list[str]) -> dict[str, Any]:
    return {
        key: component.get(key)
        for key in ("component_id", "node_id", "label", "kind", "role", "pose", "group_id", "parameters")
    } | {"quantum_metal_components": native_names}


def _route_summary(route: dict[str, Any], native_names: list[str]) -> dict[str, Any]:
    return {
        key: route.get(key)
        for key in (
            "route_id", "node_id", "label", "role", "centerline",
            "trace_width_um", "trace_gap_um", "target_length_um",
            "actual_length_um", "grounded_end", "open_end", "parameters",
        )
    } | {"quantum_metal_components": native_names}


def _render_report(
    build: QuantumMetalBuild,
    layout_pdf: Path,
    zooms: dict[str, Path],
    removed_outlines: int,
    conflicts: list[tuple[str, str, float]],
) -> dict[str, Any]:
    data = build.plan_payload
    return {
        "schema_version": "graph2metal-compact-render-1.0",
        "design_name": data.get("name"),
        "backend": "Quantum Metal / qiskit_metal",
        "quantum_metal_version": getattr(build.qm_module, "__version__", "unknown"),
        "renderer": "qiskit_metal.view",
        "chip": data.get("chip"),
        "render_style": {
            "dashed_polygon_outlines": "removed",
            "artists_modified": removed_outlines,
        },
        "qgeometry_counts": build.qgeometry_counts,
        "components": [
            _component_summary(item, build.object_map.get(str(item["component_id"]), []))
            for item in data.get("components", [])
        ],
        "routes": [
            _route_summary(item, build.object_map.get(str(item["route_id"]), []))
            for item in data.get("routes", [])
        ],
        "coupling_windows": data.get("coupling_windows", []),
        "post_build_validation": {
            "status": "PASS" if not conflicts else "FAIL",
            "geometry_conflicts": [
                {"object_a": a, "object_b": b, "overlap_area_um2": area}
                for a, b, area in conflicts
            ],
        },
        "artifacts": {
            "layout_pdf": layout_pdf.name,
            "coupler_zoom_pdfs": {
                window_id: str(Path("coupler_zooms") / path.name)
                for window_id, path in zooms.items()
            },
        },
    }


def render_quantum_metal_design(
    build: QuantumMetalBuild,
    output_dir: str | Path,
    *,
    strict_geometry_check: bool = True,
    coupler_zoom_margin_fraction: float = 0.20,
    coupler_zoom_min_margin_um: float = 40.0,
) -> QuantumMetalArtifacts:
    """Write only layout_zero.pdf, coupler PDFs, and quantum_metal_render.json."""
    conflicts = geometry_conflicts(build)
    if conflicts and strict_geometry_check:
        details = "; ".join(f"{a} vs {b} ({area:.2f} um^2)" for a, b, area in conflicts)
        raise QuantumMetalBuildError(
            f"Quantum Metal contains {len(conflicts)} unintended copper overlap(s): {details}"
        )

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="graph2metal_render_") as temporary:
        temp = Path(temporary)
        temp_layout = temp / "layout_zero.pdf"
        removed = save_layout_pdf(build.qm_module, build.design, temp_layout)
        temp_zooms, zoom_removed = save_coupler_pdfs(
            build.qm_module,
            build.design,
            build.coupling_windows,
            temp / "coupler_zooms",
            margin_fraction=coupler_zoom_margin_fraction,
            min_margin_um=coupler_zoom_min_margin_um,
        )
        removed += zoom_removed

        layout_pdf = root / "layout_zero.pdf"
        shutil.copy2(temp_layout, layout_pdf)
        zoom_root = root / "coupler_zooms"
        if zoom_root.exists():
            shutil.rmtree(zoom_root)
        zooms: dict[str, Path] = {}
        if temp_zooms:
            shutil.copytree(temp / "coupler_zooms", zoom_root)
            zooms = {window_id: zoom_root / path.name for window_id, path in temp_zooms.items()}

    render_json = root / "quantum_metal_render.json"
    report = _render_report(build, layout_pdf, zooms, removed, conflicts)
    render_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return QuantumMetalArtifacts(
        layout_pdf=layout_pdf,
        render_json=render_json,
        coupler_zoom_pdfs=zooms,
        rendered_geometry_conflicts=conflicts,
    )


def render_plan_with_quantum_metal(plan: LayoutPlan, output_dir: str | Path, **kwargs):
    """Build and render a plan without using the full graph pipeline."""
    return render_quantum_metal_design(build_quantum_metal_design(plan), output_dir, **kwargs)



def _clean_output_directory(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for child in root.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def generate_layout_zero(
    graph_like: Any,
    output_dir: str | Path,
    *,
    config: SynthesisConfig | None = None,
    strict: bool | None = None,
    strict_geometry_check: bool = True,
    plan_only: bool = False,
) -> GenerationResult:
    """Generate the layout plan and, unless requested otherwise, Quantum Metal PDFs."""
    root = Path(output_dir)
    _clean_output_directory(root)
    translation = translate_graph(graph_like, config=config, strict=strict)

    source_graph_json = root / "source_graph.json"
    source_graph_json.write_text(
        json.dumps(graph_to_payload(graph_like), indent=2),
        encoding="utf-8",
    )
    layout_plan_json = root / "layout_plan.json"
    layout_plan_json.write_text(
        json.dumps(translation.plan.to_dict(), indent=2),
        encoding="utf-8",
    )

    artifacts = None
    if not plan_only:
        build = build_quantum_metal_design(translation.plan)
        artifacts = render_quantum_metal_design(
            build,
            root,
            strict_geometry_check=strict_geometry_check,
        )

    return GenerationResult(
        translation=translation,
        output_dir=root,
        layout_plan_json=layout_plan_json,
        source_graph_json=source_graph_json,
        artifacts=artifacts,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="cQED_Gen graph -> compact graph2metal -> Quantum Metal layout zero"
    )
    parser.add_argument("input", type=Path, help="Graph JSON/pickle produced by cQED_Gen")
    parser.add_argument("-o", "--output", type=Path, default=Path("graph2metal_output"))
    parser.add_argument("--chip-width-um", type=float, default=12000.0)
    parser.add_argument("--chip-height-um", type=float, default=10000.0)
    parser.add_argument("--non-strict", action="store_true")
    parser.add_argument("--allow-qgeometry-conflicts", action="store_true")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Create source_graph.json and layout_plan.json without importing Quantum Metal",
    )
    args = parser.parse_args(argv)

    graph = load_graph(args.input)
    result = generate_layout_zero(
        graph,
        args.output,
        config=SynthesisConfig(
            chip_width_um=args.chip_width_um,
            chip_height_um=args.chip_height_um,
        ),
        strict=not args.non_strict,
        strict_geometry_check=not args.allow_qgeometry_conflicts,
        plan_only=args.plan_only,
    )
    print(f"valid={result.translation.valid}")
    print(f"layout_plan={result.layout_plan_json}")
    if result.layout_pdf is not None:
        print(f"layout_pdf={result.layout_pdf}")
        print(f"render_json={result.render_json}")
        print(f"coupler_zoom_pdfs={len(result.artifacts.coupler_zoom_pdfs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
