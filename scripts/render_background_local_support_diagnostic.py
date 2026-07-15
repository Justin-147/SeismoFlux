"""Render the score-free five-snapshot retained-area diagnostic chart."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from seismoflux.background.local_support_manifest import (
    BackgroundLocalSupportManifest,
    load_background_local_support_manifest,
)

_BAR_COLOR = "#167c80"
_TRACK_COLOR = "#e7ecef"
_GATE_COLOR = "#b33a3a"
_GRID_COLOR = "#c4ccd1"
_TEXT_COLOR = "#24323a"
_MUTED_TEXT_COLOR = "#5d6a72"
_GATE_PERCENT = 95.0


def _snapshot_label(snapshot_id: str) -> str:
    labels = {
        "fold_1": "Fold 1",
        "fold_2": "Fold 2",
        "fold_3": "Fold 3",
        "fold_4": "Fold 4",
        "final_validation": "Final validation",
    }
    return labels.get(snapshot_id, snapshot_id.replace("_", " ").title())


def _configure_axes(axis: Axes) -> None:
    axis.set_xlim(94.75, 100.85)
    axis.set_xlabel("Retained study area (%) — axis zoomed to the 95% gate")
    axis.set_xticks([95.0, 96.0, 97.0, 98.0, 99.0, 100.0])
    axis.grid(axis="x", color=_GRID_COLOR, linewidth=0.7, zorder=0)
    axis.axvline(
        _GATE_PERCENT,
        color=_GATE_COLOR,
        linewidth=1.6,
        linestyle=(0, (4, 3)),
        zorder=3,
    )
    axis.text(
        _GATE_PERCENT + 0.04,
        1.015,
        "95% minimum gate",
        color=_GATE_COLOR,
        fontsize=9,
        ha="left",
        va="bottom",
        transform=axis.get_xaxis_transform(),
    )
    axis.tick_params(axis="y", length=0, colors=_TEXT_COLOR)
    axis.tick_params(axis="x", colors=_MUTED_TEXT_COLOR)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_visible(False)
    axis.spines["bottom"].set_color(_GRID_COLOR)


def render(
    manifest: BackgroundLocalSupportManifest,
    *,
    svg_path: Path,
    png_path: Path,
) -> None:
    """Render a geometry-free chart from the public support manifest."""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.hashsalt": "seismoflux-stage2r-local-support-area-v1",
        }
    )
    snapshots = list(manifest.snapshots)
    retained_percent = [entry.support.retained_area_fraction * 100.0 for entry in snapshots]
    unsupported_counts = [
        sum(cell.status == "unsupported" for cell in entry.support.cells) for entry in snapshots
    ]
    labels = [_snapshot_label(entry.snapshot_id) for entry in snapshots]
    common_mc = [entry.support.common_mc for entry in snapshots]
    positions = list(range(len(snapshots)))

    figure: Figure
    figure, axis = plt.subplots(figsize=(11.5, 5.7), constrained_layout=True)
    axis.barh(
        positions,
        [100.0 - _GATE_PERCENT] * len(snapshots),
        left=_GATE_PERCENT,
        height=0.56,
        color=_TRACK_COLOR,
        edgecolor="none",
        zorder=1,
    )
    axis.barh(
        positions,
        [value - _GATE_PERCENT for value in retained_percent],
        left=_GATE_PERCENT,
        height=0.56,
        color=_BAR_COLOR,
        edgecolor="none",
        zorder=2,
    )

    axis.set_yticks(positions)
    axis.set_yticklabels(labels)
    axis.invert_yaxis()
    _configure_axes(axis)

    for position, value, mc, excluded in zip(
        positions,
        retained_percent,
        common_mc,
        unsupported_counts,
        strict=True,
    ):
        axis.text(
            min(value + 0.08, 100.15),
            position - 0.08,
            f"{value:.6f}%",
            color=_TEXT_COLOR,
            fontsize=10,
            ha="left",
            va="center",
            zorder=4,
        )
        cell_word = "cell" if excluded == 1 else "cells"
        axis.text(
            min(value + 0.08, 100.15),
            position + 0.17,
            f"common Mc {mc:.1f} · {excluded} excluded {cell_word}",
            color=_MUTED_TEXT_COLOR,
            fontsize=8.5,
            ha="left",
            va="center",
            zorder=4,
        )

    axis.set_title(
        "Stage 2R causal local-catalog support",
        loc="left",
        color=_TEXT_COLOR,
        fontsize=15,
        pad=24,
    )
    axis.text(
        0.0,
        1.035,
        "Score-free retained-area diagnostic · fixed 500 km grid · no boundary geometry",
        color=_MUTED_TEXT_COLOR,
        fontsize=9.5,
        ha="left",
        va="bottom",
        transform=axis.transAxes,
    )
    figure.savefig(
        svg_path,
        format="svg",
        metadata={"Date": None, "Creator": "SeismoFlux"},
    )
    svg_text = svg_path.read_text(encoding="utf-8")
    normalized_svg = "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n"
    svg_path.write_text(normalized_svg, encoding="utf-8", newline="\n")
    figure.savefig(
        png_path,
        format="png",
        dpi=180,
        metadata={"Software": "SeismoFlux"},
    )
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/background_local_support_manifest.json"),
    )
    parser.add_argument(
        "--svg",
        type=Path,
        default=Path("docs/background_local_support_diagnostic.svg"),
    )
    parser.add_argument(
        "--png",
        type=Path,
        default=Path("docs/background_local_support_diagnostic.png"),
    )
    args = parser.parse_args()
    render(
        load_background_local_support_manifest(args.manifest),
        svg_path=args.svg,
        png_path=args.png,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
