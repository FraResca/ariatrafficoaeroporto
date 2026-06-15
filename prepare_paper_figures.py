#!/usr/bin/env python3
"""Prepare paper-ready figures from analysis outputs.

The upwind/downwind figures used in the paper are regenerated directly from the
current plotting code, so the OpenStreetMap basemap and corrected geographic
aspect ratio are preserved whenever the paper figures are refreshed.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import pandas as pd


DEFAULT_UPWIND_DIR = Path("Analysis/slurm_full_upwind")
DEFAULT_CROSS_DIR = Path("Analysis/cross_pollutant")
DEFAULT_PAPER_FIGURES = Path("paper/figures")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare PDF figures used by the paper.")
    parser.add_argument("--upwind-dir", type=Path, default=DEFAULT_UPWIND_DIR)
    parser.add_argument("--cross-dir", type=Path, default=DEFAULT_CROSS_DIR)
    parser.add_argument("--paper-figures", type=Path, default=DEFAULT_PAPER_FIGURES)
    parser.add_argument(
        "--skip-upwind-regeneration",
        action="store_true",
        help="Only convert/copy existing files; do not regenerate upwind figures.",
    )
    return parser.parse_args()


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing source figure: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"Copied {src} -> {dst}")


def convert_svg_to_pdf(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing source figure: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    try:
        import cairosvg  # type: ignore

        cairosvg.svg2pdf(url=str(src), write_to=str(dst))
        print(f"Converted {src} -> {dst} with cairosvg")
        return
    except ImportError:
        pass

    inkscape = shutil.which("inkscape")
    if inkscape is None:
        raise RuntimeError(
            "Cannot convert SVG to PDF: install cairosvg in the conda env or make inkscape available."
        )
    subprocess.run(
        [inkscape, str(src), "--export-type=pdf", f"--export-filename={dst}"],
        check=True,
    )
    print(f"Converted {src} -> {dst} with inkscape")


def regenerate_upwind_paper_figures(upwind_dir: Path, paper_figures: Path) -> None:
    import upwind_downwind_analysis as upwind

    old_format = upwind.PLOT_FORMAT
    try:
        upwind.PLOT_FORMAT = "pdf"
        upwind.save_geometry_map(upwind_dir)

        bootstrap_path = upwind_dir / "upwind_downwind_bootstrap_effects.csv"
        if bootstrap_path.exists():
            upwind.save_bootstrap_effect_plots(pd.read_csv(bootstrap_path), upwind_dir)
        else:
            print(f"Skipping bootstrap regeneration: {bootstrap_path} not found")
    finally:
        upwind.PLOT_FORMAT = old_format

    copy_file(
        upwind_dir / "plots" / "airport_station_geometry_map.pdf",
        paper_figures / "airport_station_geometry_map.pdf",
    )
    bootstrap_pdf = upwind_dir / "plots" / "bootstrap_downwind_minus_upwind.pdf"
    if bootstrap_pdf.exists():
        copy_file(bootstrap_pdf, paper_figures / "bootstrap_downwind_minus_upwind.pdf")


def prepare_cross_pollutant_figures(cross_dir: Path, paper_figures: Path) -> None:
    plots_dir = cross_dir / "plots"
    for name in [
        "cross_pollutant_best_r2_by_horizon",
        "cross_pollutant_ablation_heatmap",
        "cross_pollutant_autoregressive_gain",
        "cross_pollutant_wind_response_heatmap",
    ]:
        pdf_src = plots_dir / f"{name}.pdf"
        svg_src = plots_dir / f"{name}.svg"
        dst = paper_figures / f"{name}.pdf"
        if pdf_src.exists():
            copy_file(pdf_src, dst)
        elif svg_src.exists():
            convert_svg_to_pdf(svg_src, dst)
        else:
            print(f"Skipping {name}: no PDF or SVG found in {plots_dir}")


def main() -> int:
    args = parse_args()
    args.paper_figures.mkdir(parents=True, exist_ok=True)
    if not args.skip_upwind_regeneration:
        regenerate_upwind_paper_figures(args.upwind_dir, args.paper_figures)
    prepare_cross_pollutant_figures(args.cross_dir, args.paper_figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
