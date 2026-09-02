# ruff: noqa: E501, RUF001
"""Render the score-frozen S1-C0 screen without opening later data.

Inputs are limited to development scores, sealed development predictions, and
the target-independent study geometry. Outputs are four PNGs, one self-contained
HTML explorer, and a SHA-256 manifest.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import io
import json
import math
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pyproj import Transformer
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from seismoflux.background.grid import EQUAL_AREA_CRS  # noqa: E402
from seismoflux.d1_replay.spatial import (  # noqa: E402
    D1SpatialDomain,
    build_d1_spatial_domain_from_bytes,
    select_alarm_prefixes,
)

SCHEMA_VERSION: Final = 1
SCIENCE_DIAGNOSTIC_SHA256: Final = (
    "86e448dc9c2fef2b054ec58dff9ba50986a670c0f4c87f974d1d928658a744cd"
)
MAIN_HORIZON_DAYS: Final = 30
MAIN_AREA_KM2: Final = 600_000.0
L0_ACTUAL_AREA_KM2: Final = 599_982.099787631
TOTAL_AREA_KM2: Final = 9_415_305.754432771
UNIFORM_RANDOM_AREA_EXPECTATION: Final = L0_ACTUAL_AREA_KM2 / TOTAL_AREA_KM2
EXPECTED_EPISODE_COUNT: Final = 147
L0: Final = "L0_UNIFORM"
L2: Final = "L2_KDE_CAUSAL"
L3: Final = "L3_B0_R30_CAUSAL"
LOCATION_MODELS: Final = (
    L0,
    "L1_REGIONAL_CONSTANT",
    L2,
    "L2_KDE75_LEGACY",
    L3,
)
MODEL_LABELS: Final = {
    L0: "L0 uniform (fixed tie-prefix)",
    "L1_REGIONAL_CONSTANT": "L1 regional",
    L2: "L2 causal KDE",
    "L2_KDE75_LEGACY": "L2 fixed 75 km",
    L3: "L3 long + recent",
}
STATIC_NAMES: Final = (
    "01_main_anchor_recall.png",
    "02_horizon_area_recall.png",
    "03_time_magnitude_joint.png",
    "04_all_episodes_map.png",
    "05_representative_hit_and_major_miss.png",
)
HTML_NAME: Final = "seismoflux_s1c0_explorer.html"
MANIFEST_NAME: Final = "render_manifest.json"


class RenderingError(RuntimeError):
    """Raised when frozen inputs do not match the visualization contract."""


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    episode_id: str
    event_id: str
    longitude: float
    latitude: float
    cell_index: int
    hit: bool
    fold_id: str
    issue_time_utc: str
    global_episode_member_count: int
    l3_log_density: float
    l2_log_density: float
    l0_log_density: float
    l2_hit: bool
    l3_minus_l2_log_density: float
    l3_minus_l0_log_density: float


@dataclass(frozen=True, slots=True)
class SelectedCase:
    role: str
    rule: str
    episode: EpisodeRecord


@dataclass(frozen=True, slots=True)
class SurfaceFrame:
    case: SelectedCase
    mass: np.ndarray[Any, np.dtype[np.float64]]
    alarm_indices: np.ndarray[Any, np.dtype[np.int64]]
    actual_area_km2: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, object]:
    return {"path": path.as_posix(), "sha256": _sha256(path), "size_bytes": path.stat().st_size}


def _status_value(value: Mapping[str, object], *, label: str) -> float:
    raw = value.get("value")
    if (
        value.get("status") != "evaluable"
        or isinstance(raw, bool)
        or not isinstance(raw, int | float)
        or not math.isfinite(raw)
    ):
        raise RenderingError(f"{label} is not an evaluable finite number")
    return float(raw)


def _load_summary(path: Path) -> dict[str, Any]:
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RenderingError("development summary is not valid UTF-8 JSON") from exc
    if not isinstance(summary, dict):
        raise RenderingError("development summary must be an object")
    if summary.get("champion_selection_allowed") is not False:
        raise RenderingError("S1-C0 renderer refuses a champion-selection claim")
    if summary.get("holdout_opening_allowed") is not False:
        raise RenderingError("S1-C0 renderer refuses an opened holdout")
    anchor = summary.get("location", {}).get("main_scientific_anchor", {})
    expected = (MAIN_HORIZON_DAYS, MAIN_AREA_KM2, "M5_6", "fixed_anchor_episode")
    observed = (
        anchor.get("horizon_days"),
        anchor.get("area_budget_km2"),
        anchor.get("magnitude_bin"),
        anchor.get("target_basis"),
    )
    if observed != expected:
        raise RenderingError("main scientific anchor differs from the frozen S1-C0 anchor")
    l0_main_groups = [
        group
        for group in summary["location"]["recall_groups"]
        if group["model_id"] == L0
        and group["basis"] == "anchor"
        and group["magnitude_bin"] == "M5_6"
        and group["horizon_days"] == MAIN_HORIZON_DAYS
        and group["area_budget_km2"] == MAIN_AREA_KM2
    ]
    if len(l0_main_groups) != 4 or any(
        not math.isclose(
            float(group["mean_actual_area_km2"]),
            L0_ACTUAL_AREA_KM2,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
        for group in l0_main_groups
    ):
        raise RenderingError("L0 fixed tie-prefix area differs from the audited value")
    return cast(dict[str, Any], summary)


def _validate_science_diagnostic(diagnostic: Mapping[str, Any]) -> None:
    if diagnostic.get("record_type") != "s1_c0_post_score_science_diagnostic":
        raise RenderingError("science diagnostic has the wrong record type")
    if any(
        diagnostic.get(field) is not False
        for field in ("holdout_opened", "audit_opened", "locked_test_run")
    ):
        raise RenderingError("science diagnostic opened a forbidden evaluation boundary")
    magnitude = diagnostic.get("magnitude_cluster_dependence_diagnostic")
    if not isinstance(magnitude, Mapping):
        raise RenderingError("science diagnostic has no magnitude cluster result")
    bootstrap = magnitude.get("cluster_bootstrap")
    removal = magnitude.get("remove_largest_combined_episode")
    if not isinstance(bootstrap, Mapping) or not isinstance(removal, Mapping):
        raise RenderingError("science diagnostic magnitude result is incomplete")
    expected = (
        magnitude.get("combined_M5_plus_episode_count"),
        bootstrap.get("replicates"),
        bootstrap.get("unit"),
        bootstrap.get("strictly_positive"),
        removal.get("full_catalog_member_count"),
        magnitude.get("wording_gate"),
    )
    if expected != (
        314,
        20_000,
        "combined_M5_plus_fixed_anchor_episode",
        True,
        19,
        "cluster_robust_small_development_signal",
    ):
        raise RenderingError("science diagnostic magnitude contract changed")
    numeric = (
        float(bootstrap.get("lower", math.nan)),
        float(bootstrap.get("upper", math.nan)),
        float(magnitude.get("point_mean_effect_nats_per_event", math.nan)),
        float(removal.get("remaining_mean_effect_nats_per_event", math.nan)),
    )
    if not all(math.isfinite(value) for value in numeric):
        raise RenderingError("science diagnostic magnitude values are not finite")


def _load_science_diagnostic(path: Path) -> dict[str, Any]:
    if _sha256(path) != SCIENCE_DIAGNOSTIC_SHA256:
        raise RenderingError("science diagnostic SHA-256 differs from the unique frozen run")
    try:
        diagnostic = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RenderingError("science diagnostic is not valid UTF-8 JSON") from exc
    if not isinstance(diagnostic, dict):
        raise RenderingError("science diagnostic must be an object")
    _validate_science_diagnostic(diagnostic)
    return cast(dict[str, Any], diagnostic)


def _payloads(raw_scores_path: Path, *, family: str) -> Iterable[dict[str, Any]]:
    try:
        frame = pd.read_parquet(
            raw_scores_path,
            columns=["score_family", "fold_id", "model_id", "payload_json"],
        )
    except (OSError, ValueError) as exc:
        raise RenderingError("raw_scores.parquet could not be read") from exc
    selected = frame.loc[frame["score_family"] == family]
    forbidden = selected["fold_id"].astype(str).str.contains("HOLDOUT|AUDIT|LOCKED", case=False)
    if bool(forbidden.any()):
        raise RenderingError("renderer input contains a forbidden non-development fold")
    for raw in selected["payload_json"]:
        try:
            item = json.loads(str(raw))
        except json.JSONDecodeError as exc:
            raise RenderingError("raw score payload is not valid JSON") from exc
        if not isinstance(item, dict):
            raise RenderingError("raw score payload must be an object")
        yield cast(dict[str, Any], item)


def _is_main_anchor(payload: Mapping[str, Any], *, model_id: str) -> bool:
    return bool(
        payload.get("model_id") == model_id
        and payload.get("metric") == "strict_recall"
        and payload.get("basis") == "anchor"
        and payload.get("is_main_scientific_anchor") is True
        and payload.get("horizon_days") == MAIN_HORIZON_DAYS
        and payload.get("area_budget_km2") == MAIN_AREA_KM2
        and payload.get("magnitude_bin") == "M5_6"
    )


def _extract_episodes(raw_scores_path: Path) -> list[EpisodeRecord]:
    payloads = list(_payloads(raw_scores_path, family="location"))
    baseline_by_event: dict[str, float] = {}
    for payload in payloads:
        if not _is_main_anchor(payload, model_id=L0):
            continue
        for event_id, is_anchor, log_density in zip(
            payload["event_ids"],
            payload["is_episode_anchor"],
            payload["event_log_densities_per_km2"],
            strict=True,
        ):
            if bool(is_anchor):
                baseline_by_event[str(event_id)] = float(log_density)

    l2_by_event: dict[str, tuple[bool, float, int]] = {}
    for payload in payloads:
        if not _is_main_anchor(payload, model_id=L2):
            continue
        for event_id, is_anchor, hit, log_density, member_count in zip(
            payload["event_ids"],
            payload["is_episode_anchor"],
            payload["hit_flags"],
            payload["event_log_densities_per_km2"],
            payload["global_episode_member_counts"],
            strict=True,
        ):
            if bool(is_anchor):
                l2_by_event[str(event_id)] = (bool(hit), float(log_density), int(member_count))

    episodes: dict[str, EpisodeRecord] = {}
    for payload in payloads:
        if not _is_main_anchor(payload, model_id=L3):
            continue
        vectors = zip(
            payload["episode_ids"],
            payload["event_ids"],
            payload["event_longitudes"],
            payload["event_latitudes"],
            payload["event_cell_indices"],
            payload["is_episode_anchor"],
            payload["hit_flags"],
            payload["event_log_densities_per_km2"],
            payload["global_episode_member_counts"],
            strict=True,
        )
        for (
            episode_id,
            event_id,
            lon,
            lat,
            cell,
            is_anchor,
            hit,
            l3_density,
            member_count,
        ) in vectors:
            if not bool(is_anchor):
                continue
            event_key = str(event_id)
            if event_key not in baseline_by_event or event_key not in l2_by_event:
                raise RenderingError("one L3 anchor has no paired L0/L2 result")
            l0_density = baseline_by_event[event_key]
            l2_hit, l2_density, l2_member_count = l2_by_event[event_key]
            if int(member_count) != l2_member_count:
                raise RenderingError("one episode has inconsistent frozen member count")
            record = EpisodeRecord(
                episode_id=str(episode_id),
                event_id=event_key,
                longitude=float(lon),
                latitude=float(lat),
                cell_index=int(cell),
                hit=bool(hit),
                fold_id=str(payload["fold_id"]),
                issue_time_utc=str(payload["issue_time_utc"]),
                global_episode_member_count=int(member_count),
                l3_log_density=float(l3_density),
                l2_log_density=l2_density,
                l0_log_density=l0_density,
                l2_hit=l2_hit,
                l3_minus_l2_log_density=float(l3_density) - l2_density,
                l3_minus_l0_log_density=float(l3_density) - l0_density,
            )
            previous = episodes.setdefault(record.episode_id, record)
            if previous != record:
                raise RenderingError("one episode has inconsistent main-anchor records")
    result = sorted(episodes.values(), key=lambda item: item.episode_id.encode("utf-8"))
    if len(result) != EXPECTED_EPISODE_COUNT:
        raise RenderingError(
            f"expected {EXPECTED_EPISODE_COUNT} main-anchor episodes, found {len(result)}"
        )
    return result


def select_representative_cases(episodes: Sequence[EpisodeRecord]) -> tuple[SelectedCase, ...]:
    """Select one incremental L3 success and the largest physical miss mechanically."""

    hit_rule = (
        "restrict to episodes hit by L3 but missed by L2; sort by L3-minus-L2 event "
        "log-density gain; choose the lower median; break ties by episode_id bytes"
    )
    hit_candidates = [item for item in episodes if item.hit and not item.l2_hit]
    if not hit_candidates or any(
        not math.isfinite(item.l3_minus_l2_log_density) for item in hit_candidates
    ):
        raise RenderingError("cannot select a finite L3-only incremental hit")
    hit_candidates.sort(
        key=lambda item: (
            item.l3_minus_l2_log_density,
            item.episode_id.encode("utf-8"),
        )
    )
    miss_rule = (
        "restrict to L3 misses; choose the largest frozen all-catalog episode member "
        "count; break ties by episode_id bytes"
    )
    miss_candidates = [item for item in episodes if not item.hit]
    if not miss_candidates:
        raise RenderingError("cannot select the largest L3 miss")
    miss_candidates.sort(
        key=lambda item: (-item.global_episode_member_count, item.episode_id.encode("utf-8"))
    )
    return (
        SelectedCase(
            role="representative_incremental_hit",
            rule=hit_rule,
            episode=hit_candidates[(len(hit_candidates) - 1) // 2],
        ),
        SelectedCase(
            role="largest_member_episode_miss", rule=miss_rule, episode=miss_candidates[0]
        ),
    )


def _load_domain(study_area_path: Path) -> D1SpatialDomain:
    try:
        return build_d1_spatial_domain_from_bytes(study_area_path.read_bytes())
    except (OSError, ValueError) as exc:
        raise RenderingError("study-area geometry could not rebuild the frozen grid") from exc


def _issue_microseconds(value: str) -> int:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise RenderingError("episode issue time is not timezone-aware")
    return int(stamp.value // 1_000)


def _surface_for_case(
    case: SelectedCase,
    *,
    prediction_root: Path,
    domain: D1SpatialDomain,
) -> SurfaceFrame:
    npz_path = prediction_root / "folds" / case.episode.fold_id / "predictions.npz"
    if not npz_path.is_file():
        raise RenderingError(f"missing sealed prediction array for {case.episode.fold_id}")
    with np.load(npz_path, allow_pickle=False) as arrays:
        if arrays["location_relative_mass"].shape[2] != domain.operational_grid.cell_count:
            raise RenderingError("prediction grid length differs from rebuilt study grid")
        rows = np.flatnonzero(
            (arrays["primary_issue_time_us"] == _issue_microseconds(case.episode.issue_time_utc))
            & (arrays["primary_horizon_days"] == MAIN_HORIZON_DAYS)
        )
        if rows.size != 1:
            raise RenderingError("case does not map to exactly one sealed primary prediction")
        model_positions = np.flatnonzero(arrays["location_model_index"].astype(np.int64) == 4)
        if model_positions.size != 1:
            raise RenderingError("sealed prediction does not contain exactly one L3 surface")
        mass = np.array(
            arrays["location_relative_mass"][int(rows[0]), int(model_positions[0]), :],
            dtype=np.float64,
            copy=True,
        )
    prefix = next(
        item
        for item in select_alarm_prefixes(mass, domain.operational_grid)
        if item.budget_km2 == MAIN_AREA_KM2
    )
    return SurfaceFrame(
        case=case,
        mass=mass,
        alarm_indices=np.array(prefix.selected_indices, dtype=np.int64, copy=True),
        actual_area_km2=prefix.actual_area_km2,
    )


def _geometry_polygons(geometry: BaseGeometry) -> Iterable[Polygon]:
    if isinstance(geometry, Polygon):
        yield geometry
    elif isinstance(geometry, MultiPolygon):
        yield from geometry.geoms
    else:
        for member in getattr(geometry, "geoms", ()):
            yield from _geometry_polygons(member)


def _add_boundary(ax: Axes, geometry: BaseGeometry, *, color: str = "#34465f") -> None:
    for polygon in _geometry_polygons(geometry):
        x, y = polygon.exterior.xy
        ax.plot(x, y, color=color, linewidth=0.65, zorder=1)
        for ring in polygon.interiors:
            x, y = ring.xy
            ax.plot(x, y, color=color, linewidth=0.35, zorder=1)
    ax.set_xlim(72, 136)
    ax.set_ylim(16, 55)
    ax.set_aspect("equal", adjustable="box")


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.facecolor": "#f7f9fc",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _save_figure(fig: Figure, path: Path) -> None:
    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
        metadata={"Software": "SeismoFlux S1-C0 deterministic renderer"},
    )
    plt.close(fig)


def _wilson_interval(recall: float, count: int) -> tuple[float, float]:
    z = 1.959963984540054
    denominator = 1.0 + z * z / count
    centre = (recall + z * z / (2.0 * count)) / denominator
    radius = (
        z * math.sqrt(recall * (1.0 - recall) / count + z * z / (4.0 * count * count)) / denominator
    )
    return centre - radius, centre + radius


def _main_fold_recalls(summary: Mapping[str, Any], model: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for group in summary["location"]["recall_groups"]:
        if (
            group["model_id"] == model
            and group["basis"] == "anchor"
            and group["magnitude_bin"] == "M5_6"
            and group["horizon_days"] == MAIN_HORIZON_DAYS
            and group["area_budget_km2"] == MAIN_AREA_KM2
        ):
            result[str(group["fold_id"])] = _status_value(
                group["pooled_recall"],
                label=f"{model} fold recall",
            )
    if len(result) != 4:
        raise RenderingError(f"{model} does not have four main-anchor fold recalls")
    return result


def _render_main_anchor(summary: Mapping[str, Any], output_path: Path) -> None:
    anchor = summary["location"]["main_scientific_anchor"]
    comparisons = anchor["comparisons"]
    by_model = {item["candidate_model_id"]: item for item in comparisons}
    baseline = _status_value(comparisons[0]["pooled_baseline_recall"], label="L0 recall")
    recalls = [baseline]
    for model in LOCATION_MODELS[1:]:
        recalls.append(
            _status_value(by_model[model]["pooled_candidate_recall"], label=f"{model} recall")
        )

    intervals = np.asarray([_wilson_interval(value, EXPECTED_EPISODE_COUNT) for value in recalls])
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.8))
    colors = ["#9ca8b8", "#5f8fbd", "#2f77b4", "#67a4cc", "#d95f45"]
    x = np.arange(len(LOCATION_MODELS))
    axes[0].bar(x, np.asarray(recalls) * 100.0, color=colors, width=0.72)
    axes[0].errorbar(
        x,
        np.asarray(recalls) * 100.0,
        yerr=np.vstack(
            (
                (np.asarray(recalls) - intervals[:, 0]) * 100.0,
                (intervals[:, 1] - np.asarray(recalls)) * 100.0,
            )
        ),
        fmt="none",
        ecolor="#2d3b4e",
        capsize=3,
        linewidth=1,
    )
    axes[0].axhline(
        UNIFORM_RANDOM_AREA_EXPECTATION * 100.0,
        color="#5f6977",
        linestyle="--",
        linewidth=1.2,
        label=f"uniform random-area expectation {UNIFORM_RANDOM_AREA_EXPECTATION:.3%}",
    )
    axes[0].set_xticks(
        x,
        [
            "L0 uniform\n(fixed tie-prefix)",
            "L1 regional",
            "L2 causal KDE",
            "L2 fixed 75 km",
            "L3 long + recent",
        ],
        rotation=18,
        ha="right",
    )
    axes[0].set_ylabel("Fixed-anchor episode recall (%)")
    axes[0].set_ylim(0, max(intervals[:, 1]) * 112.0)
    axes[0].set_title("A. Same-area recall + 95% episode Wilson CI")
    axes[0].grid(axis="y", alpha=0.22)
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    for index, value in enumerate(recalls):
        axes[0].text(index, value * 100.0 + 1.0, f"{value:.1%}", ha="center", fontsize=9)

    fold_ids = sorted(_main_fold_recalls(summary, "L1_REGIONAL_CONSTANT"))
    l1_folds = _main_fold_recalls(summary, "L1_REGIONAL_CONSTANT")
    l2_folds = _main_fold_recalls(summary, L2)
    l3_folds = _main_fold_recalls(summary, L3)
    l2_delta = [(l2_folds[fold] - l1_folds[fold]) * 100.0 for fold in fold_ids]
    l3_delta = [(l3_folds[fold] - l1_folds[fold]) * 100.0 for fold in fold_ids]
    l1_recall = _status_value(
        by_model["L1_REGIONAL_CONSTANT"]["pooled_candidate_recall"], label="L1 recall"
    )
    pooled_l2 = recalls[LOCATION_MODELS.index(L2)] - l1_recall
    pooled_l3 = recalls[LOCATION_MODELS.index(L3)] - l1_recall
    x_fold = np.arange(5)
    axes[1].axhline(0, color="#46566a", linewidth=1)
    axes[1].axvline(3.5, color="#aeb9c5", linewidth=1, linestyle=":")
    axes[1].plot(
        x_fold,
        [*l2_delta, pooled_l2 * 100.0],
        "o-",
        color="#2f77b4",
        linewidth=2,
        label="L2 causal KDE - L1",
    )
    axes[1].plot(
        x_fold,
        [*l3_delta, pooled_l3 * 100.0],
        "o-",
        color="#d95f45",
        linewidth=2,
        label="L3 long+recent - L1",
    )
    axes[1].scatter([4, 4], [pooled_l2 * 100.0, pooled_l3 * 100.0], marker="D", s=64)
    axes[1].set_xticks(x_fold, [fold.replace("C_DEV_", "") for fold in fold_ids] + ["pooled"])
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].set_ylabel("Recall gain over L1 (percentage points)")
    axes[1].set_title("B. Primary comparison: L2/L3 versus regional L1")
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].legend(loc="upper right", fontsize=8, frameon=False)
    axes[1].text(
        0.02,
        0.96,
        "Both comparisons positive in 4/4 folds",
        transform=axes[1].transAxes,
        va="top",
        fontsize=9,
        color="#32455e",
    )
    fig.suptitle(
        "S1-C0 main anchor: M5–6, 30 days, 600,000 km², exact-cell hit",
        fontsize=15,
    )
    fig.text(
        0.5,
        -0.01,
        "L0=5/147 is a fixed row/column tie-prefix (interpretation limited); "
        "random-area expectation=9.367/147. C0 is not champion selection.",
        ha="center",
        fontsize=9,
        color="#45556b",
    )
    fig.tight_layout()
    _save_figure(fig, output_path)


def _location_recall_records(summary: Mapping[str, Any]) -> list[dict[str, object]]:
    totals: dict[tuple[int, float, str], list[float]] = defaultdict(lambda: [0.0, 0.0])
    for group in summary["location"]["recall_groups"]:
        if group["basis"] != "anchor" or group["magnitude_bin"] != "M5_6":
            continue
        key = (
            int(group["horizon_days"]),
            float(group["area_budget_km2"]),
            str(group["model_id"]),
        )
        totals[key][0] += float(group["hit_weight"])
        totals[key][1] += float(group["total_weight"])
    records: list[dict[str, object]] = []
    for (horizon, area, model), (hits, total) in sorted(totals.items()):
        if total <= 0.0:
            continue
        records.append(
            {
                "horizon_days": horizon,
                "area_budget_km2": area,
                "model_id": model,
                "model_label": MODEL_LABELS[model],
                "hit_count": round(hits),
                "episode_count": round(total),
                "recall": hits / total,
            }
        )
    return records


def _render_horizon_area(
    summary: Mapping[str, Any],
    output_path: Path,
) -> list[dict[str, object]]:
    records = _location_recall_records(summary)
    horizons = sorted({int(item["horizon_days"]) for item in records})
    areas = sorted({float(item["area_budget_km2"]) for item in records})
    lookup = {
        (
            int(item["horizon_days"]),
            float(item["area_budget_km2"]),
            str(item["model_id"]),
        ): item
        for item in records
    }
    l3 = np.asarray(
        [[float(lookup[(horizon, area, L3)]["recall"]) for area in areas] for horizon in horizons]
    )
    delta = np.asarray(
        [
            [
                float(lookup[(horizon, area, L3)]["recall"])
                - float(lookup[(horizon, area, L2)]["recall"])
                for area in areas
            ]
            for horizon in horizons
        ]
    )
    counts = np.asarray(
        [
            [int(lookup[(horizon, area, L3)]["episode_count"]) for area in areas]
            for horizon in horizons
        ]
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.7))
    maximum_delta = max(5.0, float(abs(delta).max() * 100.0))
    images = [
        axes[0].imshow(
            l3 * 100.0,
            cmap="YlOrRd",
            aspect="auto",
            vmin=0,
            vmax=max(45.0, float(l3.max() * 100.0)),
        ),
        axes[1].imshow(
            delta * 100.0,
            cmap="RdBu_r",
            aspect="auto",
            vmin=-maximum_delta,
            vmax=maximum_delta,
        ),
    ]
    titles = ("A. L3 fixed-anchor recall (%)", "B. L3 − L2 recall (points)")
    for ax, title in zip(axes, titles, strict=True):
        ax.set_xticks(range(len(areas)), [f"{int(area / 1000)}k" for area in areas])
        ax.set_yticks(range(len(horizons)), [str(horizon) for horizon in horizons])
        ax.set_xlabel("Alarm area (km²)")
        ax.set_ylabel("Horizon (days)")
        ax.set_title(title)
        ax.add_patch(
            plt.Rectangle(
                (
                    areas.index(MAIN_AREA_KM2) - 0.5,
                    horizons.index(MAIN_HORIZON_DAYS) - 0.5,
                ),
                1,
                1,
                fill=False,
                edgecolor="#202936",
                linewidth=2.2,
            )
        )
    for row in range(len(horizons)):
        for column in range(len(areas)):
            axes[0].text(
                column,
                row,
                f"{l3[row, column] * 100:.1f}\nn={counts[row, column]}",
                ha="center",
                va="center",
                fontsize=8,
            )
            axes[1].text(
                column,
                row,
                f"{delta[row, column] * 100:+.1f}",
                ha="center",
                va="center",
                fontsize=9,
            )
    fig.colorbar(images[0], ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(images[1], ax=axes[1], fraction=0.046, pad=0.04)
    fig.suptitle(
        "Duration × alarm-area sensitivity (M5–6 fixed-anchor episodes)",
        fontsize=15,
    )
    fig.text(
        0.5,
        -0.01,
        "Every cell is separate; horizons and alarm areas are never pooled to inflate evidence.",
        ha="center",
        fontsize=9,
        color="#45556b",
    )
    fig.tight_layout()
    _save_figure(fig, output_path)
    return records


def _render_time_magnitude_joint(
    summary: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    time_rows = summary["time"]["comparisons"]
    colors = {"M5_6": "#2f77b4", "M6_plus": "#d95f45"}
    for band, offset in (("M5_6", -0.12), ("M6_plus", 0.12)):
        rows = sorted(
            (item for item in time_rows if item["magnitude_band"] == band),
            key=lambda item: item["horizon_days"],
        )
        x = np.arange(len(rows), dtype=float) + offset
        values = np.asarray(
            [
                _status_value(
                    item["pooled_count_log_score_difference_T1_minus_T0"],
                    label="T1-T0",
                )
                for item in rows
            ]
        )
        intervals = np.asarray(
            [item["paired_bootstrap"]["confidence_interval_95"] for item in rows],
            dtype=float,
        )
        axes[0].errorbar(
            x,
            values,
            yerr=np.vstack((values - intervals[:, 0], intervals[:, 1] - values)),
            fmt="o-",
            capsize=3,
            label=band.replace("_", "–"),
            color=colors[band],
        )
    axes[0].axhline(0, color="#46566a", linewidth=1)
    axes[0].set_xticks(range(5), ["7", "30", "90", "180", "365"])
    axes[0].set_xlabel("Horizon (days; separate tasks)")
    axes[0].set_ylabel("T1 − T0 count log score")
    axes[0].set_title("A. Time model")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.22)

    magnitude = diagnostic["magnitude_cluster_dependence_diagnostic"]
    cluster_bootstrap = magnitude["cluster_bootstrap"]
    removal = magnitude["remove_largest_combined_episode"]
    magnitude_value = float(magnitude["point_mean_effect_nats_per_event"])
    magnitude_interval = np.asarray(
        [cluster_bootstrap["lower"], cluster_bootstrap["upper"]],
        dtype=float,
    )
    axes[1].axhline(0, color="#46566a", linewidth=1)
    axes[1].errorbar(
        [0],
        [magnitude_value],
        yerr=[
            [magnitude_value - magnitude_interval[0]],
            [magnitude_interval[1] - magnitude_value],
        ],
        fmt="o",
        color="#6c5aa7",
        capsize=5,
        markersize=8,
    )
    for index, fold in enumerate(magnitude["four_outer_fold_mean_effects"]):
        fold_value = float(fold["mean_effect_nats_per_event"])
        axes[1].scatter(-0.08 + index * 0.052, fold_value, color="#3f315f", s=25)
    axes[1].text(
        0.03,
        0.96,
        "314 combined M5+ fixed-anchor episodes\n"
        "20,000 cluster-bootstrap replicates\n"
        f"remove largest 19-event episode: {float(removal['remaining_mean_effect_nats_per_event']):.6f}",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.75},
    )
    axes[1].set_xlim(-0.45, 0.45)
    axes[1].set_xticks([0], ["M3 long-history − M0\n(445 events, 314 clusters)"])
    axes[1].set_ylabel("Mean log-probability difference (nats/event)")
    axes[1].set_title("B. Magnitude: small cluster-robust signal")
    axes[1].grid(axis="y", alpha=0.22)

    joint_rows = [
        item
        for item in summary["joint"]["comparisons"]
        if item["horizon_days"] == MAIN_HORIZON_DAYS
    ]
    y = np.arange(len(joint_rows))
    joint_values = np.asarray(
        [
            _status_value(item["mean_joint_log_score_difference"], label="joint")
            for item in joint_rows
        ]
    )
    joint_intervals = np.asarray(
        [item["paired_bootstrap"]["confidence_interval_95"] for item in joint_rows],
        dtype=float,
    )
    axes[2].axvline(0, color="#46566a", linewidth=1)
    axes[2].errorbar(
        joint_values,
        y,
        xerr=np.vstack(
            (
                joint_values - joint_intervals[:, 0],
                joint_intervals[:, 1] - joint_values,
            )
        ),
        fmt="o",
        color="#16827a",
        capsize=4,
        markersize=7,
    )
    axes[2].set_yticks(
        y,
        [str(item["candidate_model_id"]).replace("_", " ") for item in joint_rows],
    )
    axes[2].set_xlabel("Joint log-score gain over J0")
    axes[2].set_title("C. Joint model (30 days only)")
    axes[2].grid(axis="x", alpha=0.22)
    fig.suptitle("Separate evidence for when, how large, and joint forecasts", fontsize=15)
    fig.text(
        0.5,
        -0.01,
        "Positive is better. Paired 95% intervals; no horizon is combined with another.",
        ha="center",
        fontsize=9,
        color="#45556b",
    )
    fig.tight_layout()
    _save_figure(fig, output_path)


def _render_episode_map(
    episodes: Sequence[EpisodeRecord],
    geometry: BaseGeometry,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 7.3))
    _add_boundary(ax, geometry)
    misses = [item for item in episodes if not item.hit]
    hits = [item for item in episodes if item.hit]
    ax.scatter(
        [item.longitude for item in misses],
        [item.latitude for item in misses],
        marker="x",
        color="#bd3b45",
        s=34,
        linewidth=1.2,
        label=f"missed ({len(misses)})",
        zorder=3,
    )
    ax.scatter(
        [item.longitude for item in hits],
        [item.latitude for item in hits],
        marker="o",
        facecolor="#1f9d79",
        edgecolor="white",
        s=38,
        linewidth=0.55,
        label=f"hit ({len(hits)})",
        zorder=4,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("All 147 independent M5–6 fixed-anchor episodes: L3 at 30 days / 600,000 km²")
    ax.legend(loc="lower left", frameon=True)
    ax.grid(alpha=0.13)
    fig.text(
        0.5,
        0.02,
        "Both successes and failures are shown. Every case uses the same 600,000 km² budget.",
        ha="center",
        fontsize=9,
        color="#45556b",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save_figure(fig, output_path)


def _grid_lonlat(
    domain: D1SpatialDomain,
) -> tuple[
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
]:
    transformer = Transformer.from_crs(EQUAL_AREA_CRS, "EPSG:4326", always_xy=True)
    xy = domain.operational_grid.query_xy_km * 1_000.0
    lon, lat = transformer.transform(xy[:, 0], xy[:, 1])
    return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)


def _plot_surface(
    ax: Axes,
    frame: SurfaceFrame,
    *,
    domain: D1SpatialDomain,
    lon: np.ndarray[Any, np.dtype[np.float64]],
    lat: np.ndarray[Any, np.dtype[np.float64]],
    title: str,
) -> Any:
    density = frame.mass / domain.operational_grid.clipped_area_km2
    log_density = np.log10(np.maximum(density, np.finfo(np.float64).tiny))
    low, high = np.quantile(log_density, [0.03, 0.995])
    _add_boundary(ax, domain.study_area_wgs84, color="#61728a")
    scatter = ax.scatter(
        lon,
        lat,
        c=log_density,
        s=4.0,
        cmap="magma",
        vmin=float(low),
        vmax=float(high),
        linewidths=0,
        zorder=2,
    )
    alarm = frame.alarm_indices
    ax.scatter(
        lon[alarm],
        lat[alarm],
        facecolors="none",
        edgecolors="#2fd3c2",
        s=8,
        linewidths=0.35,
        zorder=3,
    )
    episode = frame.case.episode
    ax.scatter(
        [episode.longitude],
        [episode.latitude],
        marker="*",
        color="white",
        edgecolor="#101820",
        s=135,
        linewidth=0.8,
        zorder=5,
    )
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    return scatter


def _surface_png_base64(
    frame: SurfaceFrame,
    *,
    domain: D1SpatialDomain,
    lon: np.ndarray[Any, np.dtype[np.float64]],
    lat: np.ndarray[Any, np.dtype[np.float64]],
) -> str:
    fig, ax = plt.subplots(figsize=(8.2, 5.1))
    episode = frame.case.episode
    result = "HIT" if episode.hit else "MISS"
    scatter = _plot_surface(
        ax,
        frame,
        domain=domain,
        lon=lon,
        lat=lat,
        title=(
            f"{frame.case.role.replace('_', ' ').title()} · "
            f"{episode.issue_time_utc[:10]} · {result}"
        ),
    )
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label("log10 relative intensity per km²")
    fig.text(
        0.5,
        0.015,
        "Cyan: complete-cell alarm within the same 600,000 km² budget. Star: observed anchor.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=145,
        bbox_inches="tight",
        metadata={"Software": "SeismoFlux S1-C0 deterministic renderer"},
    )
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _render_case_pair(
    frames: Sequence[SurfaceFrame],
    *,
    domain: D1SpatialDomain,
    lon: np.ndarray[Any, np.dtype[np.float64]],
    lat: np.ndarray[Any, np.dtype[np.float64]],
    output_path: Path,
) -> None:
    if tuple(frame.case.role for frame in frames) != (
        "representative_incremental_hit",
        "largest_member_episode_miss",
    ):
        raise RenderingError("static case pair differs from the mechanical selection")
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.3))
    titles = (
        "A. 增量命中 / Incremental hit\nL3 hit, L2 miss; median L3-L2 gain",
        "B. 重大漏报 / Major miss\nLargest missed episode (19 members)",
    )
    for ax, frame, title in zip(axes, frames, titles, strict=True):
        scatter = _plot_surface(
            ax,
            frame,
            domain=domain,
            lon=lon,
            lat=lat,
            title=title,
        )
        colorbar = fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.02)
        colorbar.set_label("log10 relative intensity per km²")
        episode = frame.case.episode
        ax.text(
            0.01,
            -0.16,
            f"Issue {episode.issue_time_utc[:10]} | {episode.longitude:.2f}°E, "
            f"{episode.latitude:.2f}°N | {'HIT' if episode.hit else 'MISS'} | "
            f"members={episode.global_episode_member_count}",
            transform=ax.transAxes,
            fontsize=8.5,
            color="#34465f",
        )
    fig.suptitle("机械选取案例：成功与失败同时展示 / Mechanical case replay", fontsize=15)
    fig.text(
        0.5,
        0.01,
        "同一 600,000 km² 报警面积预算；颜色为相对强度，并非绝对发震概率。",
        ha="center",
        fontsize=9,
        color="#45556b",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    _save_figure(fig, output_path)


def _svg_boundary_path(geometry: BaseGeometry) -> str:
    simplified = geometry.simplify(0.04, preserve_topology=True)
    parts: list[str] = []
    for polygon in _geometry_polygons(simplified):
        for ring in (polygon.exterior, *polygon.interiors):
            points = list(ring.coords)
            if not points:
                continue
            parts.append(f"M {points[0][0]:.4f} {60.0 - points[0][1]:.4f}")
            parts.extend(f"L {x:.4f} {60.0 - y:.4f}" for x, y in points[1:])
            parts.append("Z")
    return " ".join(parts)


def _json_for_script(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def _render_html(
    *,
    summary: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
    recall_records: Sequence[Mapping[str, object]],
    episodes: Sequence[EpisodeRecord],
    cases: Sequence[SelectedCase],
    case_images: Mapping[str, str],
    geometry: BaseGeometry,
    output_path: Path,
) -> None:
    anchor = summary["location"]["main_scientific_anchor"]
    comparisons = {item["candidate_model_id"]: item for item in anchor["comparisons"]}
    l1_recall = _status_value(
        comparisons["L1_REGIONAL_CONSTANT"]["pooled_candidate_recall"],
        label="L1 recall",
    )
    l2_recall = _status_value(comparisons[L2]["pooled_candidate_recall"], label="L2 recall")
    l3_recall = _status_value(comparisons[L3]["pooled_candidate_recall"], label="L3 recall")
    magnitude = diagnostic["magnitude_cluster_dependence_diagnostic"]
    cluster_bootstrap = magnitude["cluster_bootstrap"]
    removal = magnitude["remove_largest_combined_episode"]
    episode_payload = [asdict(item) for item in episodes]
    case_payload = [
        {
            "role": case.role,
            "rule": case.rule,
            "episode": asdict(case.episode),
            "image": case_images[case.role],
        }
        for case in cases
    ]
    fold_options = "".join(
        f'<option value="{html.escape(fold)}">{html.escape(fold.replace("C_DEV_", ""))}</option>'
        for fold in sorted({item.fold_id for item in episodes})
    )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SeismoFlux S1-C0 科学结果浏览器</title>
<style>
:root{{--ink:#172235;--muted:#607087;--paper:#f3f6fa;--card:#fff;--blue:#2f77b4;--red:#bd3b45;--green:#16876d;--line:#dce3ec}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font-family:"Microsoft YaHei","Noto Sans SC",Arial,sans-serif}}header{{padding:34px max(28px,6vw);background:linear-gradient(120deg,#13243b,#23527b 58%,#187d73);color:white}}header h1{{margin:0 0 10px;font-size:clamp(26px,4vw,44px)}}header p{{max-width:1000px;line-height:1.65;margin:0;color:#dceaf4}}main{{max-width:1240px;margin:auto;padding:24px}}section{{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:22px;margin-bottom:20px;box-shadow:0 8px 24px #17324c10}}h2{{margin-top:0}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px}}.card{{padding:17px;border-radius:12px;background:#f7fafd;border-left:4px solid var(--blue)}}.value{{font-size:30px;font-weight:750;margin:6px 0}}.small{{font-size:13px;color:var(--muted);line-height:1.55}}.controls{{display:flex;flex-wrap:wrap;gap:12px;margin:12px 0}}label{{font-size:13px;color:var(--muted)}}select,input{{display:block;margin-top:4px;padding:8px 10px;border:1px solid #bdc9d8;border-radius:8px;background:white}}table{{width:100%;border-collapse:collapse}}th,td{{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left}}.bar{{height:9px;border-radius:9px;background:#dce6f0;overflow:hidden;min-width:150px}}.bar i{{display:block;height:100%;background:linear-gradient(90deg,#2f77b4,#1b9b83)}}.map-wrap{{overflow:hidden;border:1px solid var(--line);border-radius:12px;background:#edf3f7}}svg{{width:100%;height:auto;display:block}}.episode{{cursor:pointer;transition:.15s}}.episode:hover{{r:.62;stroke-width:.25}}.case-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:18px}}.case img{{width:100%;border-radius:10px;border:1px solid var(--line)}}.badge{{display:inline-block;padding:4px 9px;border-radius:999px;background:#e9f2fa;color:#234e75;font-size:12px;font-weight:700}}.warning{{border-left:5px solid #e0a22f;background:#fff8e8;padding:14px 16px;border-radius:8px;line-height:1.6}}code{{font-size:12px}}@media(max-width:650px){{main{{padding:12px}}section{{padding:15px}}.case-grid{{grid-template-columns:1fr}}}}
</style></head><body>
<header><h1>SeismoFlux S1-C0 科学结果</h1><p>2000–2019 四个开发折上的初步全 M4 目录筛选。所有模型在相同报警面积预算下比较；每个预报时长单独评价。相对强度只表示空间排序，不是绝对发震概率。</p></header>
<main>
<section><h2>一眼看懂</h2><div class="cards">
<div class="card"><div class="small">主锚 L3 召回</div><div class="value">{l3_recall:.1%}</div><div class="small">54 / 147 个独立 M5–6 固定锚点震序</div></div>
<div class="card"><div class="small">相对区域常率 L1</div><div class="value">+{(l3_recall - l1_recall) * 100:.2f}</div><div class="small">百分点；L2 为 +{(l2_recall - l1_recall) * 100:.2f} 个百分点</div></div>
<div class="card"><div class="small">固定科学条件</div><div class="value">30 天</div><div class="small">600,000 km² · 24 h 目录延迟 · exact-cell</div></div>
<div class="card"><div class="small">证据边界</div><div class="value">C0</div><div class="small">不是冠军模型，不允许打开 holdout；S1-C1 仍是必做阶段</div></div>
</div><p class="small">L0 均匀模型的 5/147 来自所有网格等强度时固定 row/column 并列排序形成的前缀，不能解释为“随机猜中 5 个”。精确前缀面积 599,982.100 km²，占冻结总面积 6.372412%，均匀随机面积期望为 9.367/147。</p></section>
<section><h2>时长 × 报警面积（分开看，不混合）</h2><div class="controls"><label>预报时长<select id="horizon"><option>7</option><option selected>30</option><option>90</option><option>180</option><option>365</option></select></label><label>报警面积（km²）<select id="area"><option>300000</option><option>450000</option><option selected>600000</option><option>750000</option><option>960000</option></select></label></div><div id="recall-table"></div></section>
<section><h2>震级模型：小而稳健的开发信号</h2><div class="cards">
<div class="card"><div class="small">M3 − M0 平均提升</div><div class="value">{float(magnitude["point_mean_effect_nats_per_event"]):.6f}</div><div class="small">nats/event；同一批 445 个 M5+ 事件</div></div>
<div class="card"><div class="small">聚类 bootstrap 95% CI</div><div class="value">正</div><div class="small">[{float(cluster_bootstrap["lower"]):.6f}, {float(cluster_bootstrap["upper"]):.6f}]；20,000 次</div></div>
<div class="card"><div class="small">独立聚类单位</div><div class="value">314</div><div class="small">合并 M5+ 固定锚点震序，不把同一震序事件当独立样本</div></div>
<div class="card"><div class="small">移除最大震序后</div><div class="value">{float(removal["remaining_mean_effect_nats_per_event"]):.6f}</div><div class="small">去除 19 个受评事件的最大震序后仍为正</div></div>
</div><p class="small">结论限于开发集：这是一个效应较小、但对震序内相关性和最大震序都稳健的信号（small, cluster-robust development signal），不是最终冠军结论。</p></section>
<section><h2>全部 147 个独立震序（主证据）</h2><div class="controls"><label>开发折<select id="fold"><option value="all">全部</option>{fold_options}</select></label><label>结果<select id="hit"><option value="all">命中与漏报</option><option value="hit">只看命中</option><option value="miss">只看漏报</option></select></label><label>搜索 episode / event<input id="search" placeholder="输入编号"></label></div><div class="small" id="episode-count"></div><div class="map-wrap"><svg viewBox="72 5 64 42" aria-label="全部episode地图"><path d="{_svg_boundary_path(geometry)}" fill="#f8fbfd" stroke="#73869b" stroke-width=".10" fill-rule="evenodd"/><g id="episode-layer"></g></svg></div><div class="small">绿色圆点为命中，红色叉号为漏报。悬停查看 episode、起报日、开发折、成员数和 L3−L0 对数密度差。</div></section>
<section><h2>一个增量命中与一个重大漏报</h2><div class="warning">案例不是人工挑选：增量命中限定为“L3 命中但 L2 漏报”，按 L3−L2 对数密度增益排序取较低中位数；重大漏报在 L3 漏报中按完整目录 episode 成员数最大选取。并列均按 episode_id 字节序。</div><div class="case-grid" id="cases"></div></section>
<section><h2>读图限制</h2><ul><li>C0 只用初步全 M4 地震目录，不能直接宣布最终冠军。</li><li>空间图显示相对强度与同面积报警前缀，不是绝对发震概率。</li><li>7/30/90/180/365 天是不同任务，不能合并来扩大样本量或效果。</li><li>本页同时呈现命中和漏报；没有读取 2020 年以后 holdout、audit 或锁定测试。</li></ul></section>
</main><script>
const recall={_json_for_script(recall_records)};
const episodes={_json_for_script(episode_payload)};
const cases={_json_for_script(case_payload)};
const esc=s=>String(s).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
function renderRecall(){{const h=+document.querySelector('#horizon').value,a=+document.querySelector('#area').value;const rows=recall.filter(x=>x.horizon_days===h&&x.area_budget_km2===a);document.querySelector('#recall-table').innerHTML='<table><thead><tr><th>模型</th><th>召回</th><th>命中/样本</th><th>同面积条形</th></tr></thead><tbody>'+rows.map(x=>`<tr><td>${{esc(x.model_label)}}</td><td><b>${{(x.recall*100).toFixed(1)}}%</b></td><td>${{x.hit_count}} / ${{x.episode_count}}</td><td><div class="bar"><i style="width:${{x.recall*100}}%"></i></div></td></tr>`).join('')+'</tbody></table>';}}
function renderEpisodes(){{const fold=document.querySelector('#fold').value,status=document.querySelector('#hit').value,q=document.querySelector('#search').value.trim().toLowerCase();const shown=episodes.filter(e=>(fold==='all'||e.fold_id===fold)&&(status==='all'||(status==='hit')===e.hit)&&(!q||e.episode_id.toLowerCase().includes(q)||e.event_id.toLowerCase().includes(q)));document.querySelector('#episode-count').textContent=`当前显示 ${{shown.length}} / ${{episodes.length}}（命中 ${{shown.filter(e=>e.hit).length}}，漏报 ${{shown.filter(e=>!e.hit).length}}）`;document.querySelector('#episode-layer').innerHTML=shown.map(e=>{{const y=60-e.latitude,title=`${{e.hit?'命中':'漏报'}} | ${{e.issue_time_utc.slice(0,10)}} | ${{e.fold_id.replace('C_DEV_','')}} | members=${{e.global_episode_member_count}} | Δlog=${{e.l3_minus_l0_log_density.toFixed(3)}} | ${{e.episode_id}}`;return e.hit?`<circle class="episode" cx="${{e.longitude}}" cy="${{y}}" r=".35" fill="#16876d" stroke="white" stroke-width=".12"><title>${{esc(title)}}</title></circle>`:`<g class="episode" stroke="#bd3b45" stroke-width=".18"><line x1="${{e.longitude-.28}}" y1="${{y-.28}}" x2="${{e.longitude+.28}}" y2="${{y+.28}}"/><line x1="${{e.longitude-.28}}" y1="${{y+.28}}" x2="${{e.longitude+.28}}" y2="${{y-.28}}"/><title>${{esc(title)}}</title></g>`;}}).join('');}}
function renderCases(){{document.querySelector('#cases').innerHTML=cases.map(c=>`<article class="case"><h3>${{c.role==='representative_incremental_hit'?'L3 增量命中代表':'最大成员震序漏报'}} <span class="badge">${{c.episode.hit?'HIT':'MISS'}}</span></h3><img alt="${{esc(c.role)}} prediction replay" src="data:image/png;base64,${{c.image}}"><p><b>起报：</b>${{esc(c.episode.issue_time_utc)}}　<b>开发折：</b>${{esc(c.episode.fold_id)}}</p><p><b>位置：</b>${{c.episode.longitude.toFixed(2)}}°, ${{c.episode.latitude.toFixed(2)}}°　<b>震序成员：</b>${{c.episode.global_episode_member_count}}</p><p><b>L3−L2 log密度：</b>${{c.episode.l3_minus_l2_log_density.toFixed(3)}}　<b>L3−L0：</b>${{c.episode.l3_minus_l0_log_density.toFixed(3)}}</p><p class="small"><code>${{esc(c.episode.episode_id)}}</code></p></article>`).join('');}}
['horizon','area'].forEach(id=>document.querySelector('#'+id).addEventListener('change',renderRecall));['fold','hit'].forEach(id=>document.querySelector('#'+id).addEventListener('change',renderEpisodes));document.querySelector('#search').addEventListener('input',renderEpisodes);renderRecall();renderEpisodes();renderCases();
</script></body></html>"""
    output_path.write_text(document, encoding="utf-8", newline="\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render S1-C0 static and offline interactive development results"
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--science-diagnostic", type=Path, required=True)
    parser.add_argument("--raw-scores", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--study-area", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def render(args: argparse.Namespace) -> dict[str, object]:
    summary_path = args.summary.resolve()
    science_diagnostic_path = args.science_diagnostic.resolve()
    raw_scores_path = args.raw_scores.resolve()
    prediction_root = args.prediction_root.resolve()
    study_area_path = args.study_area.resolve()
    output_dir = args.output_dir.resolve()
    required = (summary_path, science_diagnostic_path, raw_scores_path, study_area_path)
    if any(not path.is_file() for path in required) or not prediction_root.is_dir():
        raise RenderingError("one or more required renderer inputs do not exist")
    npz_paths = sorted(prediction_root.glob("folds/C_DEV_*/predictions.npz"))
    if len(npz_paths) != 4:
        raise RenderingError("exactly four sealed development prediction arrays are required")
    output_dir.mkdir(parents=True, exist_ok=True)
    _style()
    summary = _load_summary(summary_path)
    diagnostic = _load_science_diagnostic(science_diagnostic_path)
    episodes = _extract_episodes(raw_scores_path)
    cases = select_representative_cases(episodes)
    domain = _load_domain(study_area_path)
    if domain.operational_grid.cell_count != 15_697:
        raise RenderingError("rebuilt operational grid does not contain 15,697 cells")
    if not math.isclose(
        math.fsum(float(value) for value in domain.operational_grid.clipped_area_km2),
        TOTAL_AREA_KM2,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise RenderingError("rebuilt grid area differs from the frozen audited total")
    surfaces = tuple(
        _surface_for_case(case, prediction_root=prediction_root, domain=domain) for case in cases
    )
    _render_main_anchor(summary, output_dir / STATIC_NAMES[0])
    recall_records = _render_horizon_area(summary, output_dir / STATIC_NAMES[1])
    _render_time_magnitude_joint(summary, diagnostic, output_dir / STATIC_NAMES[2])
    _render_episode_map(episodes, domain.study_area_wgs84, output_dir / STATIC_NAMES[3])
    lon, lat = _grid_lonlat(domain)
    _render_case_pair(
        surfaces,
        domain=domain,
        lon=lon,
        lat=lat,
        output_path=output_dir / STATIC_NAMES[4],
    )
    case_images = {
        frame.case.role: _surface_png_base64(
            frame,
            domain=domain,
            lon=lon,
            lat=lat,
        )
        for frame in surfaces
    }
    _render_html(
        summary=summary,
        diagnostic=diagnostic,
        recall_records=recall_records,
        episodes=episodes,
        cases=cases,
        case_images=case_images,
        geometry=domain.study_area_wgs84,
        output_path=output_dir / HTML_NAME,
    )
    output_paths = [output_dir / name for name in (*STATIC_NAMES, HTML_NAME)]
    inputs = [summary_path, science_diagnostic_path, raw_scores_path, study_area_path, *npz_paths]
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "seismoflux_s1c0_scientific_visualization_manifest",
        "scientific_role": "preliminary_all_M4_screen_not_champion_selection",
        "main_anchor": {
            "magnitude_bin": "M5_6",
            "horizon_days": MAIN_HORIZON_DAYS,
            "area_budget_km2": MAIN_AREA_KM2,
            "catalog_delay_hours": 24,
            "hit_tolerance_km": 0.0,
            "episode_definition": "fixed_anchor_30d_75km",
        },
        "input_identities": [_identity(path) for path in inputs],
        "output_identities": [_identity(path) for path in output_paths],
        "episode_audit": {
            "total": len(episodes),
            "l3_hits": sum(item.hit for item in episodes),
            "l3_misses": sum(not item.hit for item in episodes),
            "l3_only_vs_l2_hits": sum(item.hit and not item.l2_hit for item in episodes),
        },
        "l0_tie_prefix_audit": {
            "observed_hits": 5,
            "actual_area_km2": L0_ACTUAL_AREA_KM2,
            "frozen_total_area_km2": TOTAL_AREA_KM2,
            "uniform_random_area_expectation": UNIFORM_RANDOM_AREA_EXPECTATION,
            "uniform_random_expected_hits_of_147": (
                UNIFORM_RANDOM_AREA_EXPECTATION * EXPECTED_EPISODE_COUNT
            ),
            "interpretation": "fixed_row_column_tie_prefix_not_random_area_draw",
        },
        "magnitude_cluster_diagnostic": diagnostic["magnitude_cluster_dependence_diagnostic"],
        "mechanical_cases": [
            {
                "role": item.role,
                "rule": item.rule,
                "episode_id": item.episode.episode_id,
                "event_id": item.episode.event_id,
                "hit": item.episode.hit,
                "issue_time_utc": item.episode.issue_time_utc,
                "fold_id": item.episode.fold_id,
                "global_episode_member_count": item.episode.global_episode_member_count,
                "l3_minus_l2_log_density": item.episode.l3_minus_l2_log_density,
                "l3_minus_l0_log_density": item.episode.l3_minus_l0_log_density,
            }
            for item in cases
        ],
        "limitations": [
            "S1-C0 is preliminary and cannot select a champion",
            "relative spatial intensity is not absolute earthquake probability",
            "all model recalls use the same alarm-area budget within each comparison",
            "different horizons and alarm areas are displayed separately and never pooled",
            "L0 recall is a fixed row-column tie-prefix result, not random-area expectation",
            "no holdout, audit, locked test, or post-2019 truth was read",
        ],
    }
    (output_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        manifest = render(args)
    except (OSError, RenderingError, TypeError, ValueError, KeyError) as exc:
        print(f"S1-C0 result rendering error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {len(manifest['output_identities'])} visual artifacts and {MANIFEST_NAME}")
    for item in cast(list[dict[str, object]], manifest["output_identities"]):
        print(f"{item['sha256']}  {item['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
