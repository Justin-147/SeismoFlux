"""Generate the deterministic P1-0C real-history adapter rehearsal artifacts.

This command reads only three explicit frozen local inputs.  It cannot accept a
network endpoint, a future-outcome file, or a real prospective ledger.  The
separate mature replay uses canonical synthetic source bytes and is bound to the
unchanged start-time forecast SVG.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from pyproj import CRS, Transformer

for variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(variable, "1")

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from seismoflux.background.grid import EQUAL_AREA_CRS  # noqa: E402
from seismoflux.p1_b0_r30.core import (  # noqa: E402
    AlarmPrefix,
    DualModelForecast,
    GridCell,
    RelativeIntensitySurface,
    SyntheticEvent,
    build_pending_sequential_reviews,
    elapsed_tropical_months,
)
from seismoflux.p1_b0_r30.preflight import (  # noqa: E402
    PREFLIGHT_QUERY_CUTOFF_UTC,
    PREFLIGHT_SCHEDULED_TIME_UTC,
    RealHistoryPreflight,
    build_real_history_preflight,
)
from seismoflux.p1_b0_r30.preflight_rendering import (  # noqa: E402
    build_preflight_forecast_html,
    build_preflight_mature_replay_html,
    render_preflight_forecast_svg,
    render_preflight_mature_replay_svg,
)
from seismoflux.p1_b0_r30.preimage import (  # noqa: E402
    build_catalog_snapshot_bytes,
    recompute_mature_truth_snapshot,
    source_snapshot_sha256,
)

CATALOG_PATH = (
    REPOSITORY_ROOT
    / "data"
    / "processed"
    / "stage1"
    / "debc98054172a4a1"
    / "earthquake_event.parquet"
)
STUDY_AREA_PATH = REPOSITORY_ROOT / "data" / "processed" / "china_mainland.geojson"
SUPPORT_MANIFEST_PATH = (
    REPOSITORY_ROOT / "data" / "manifests" / "background_local_support_manifest.json"
)
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "docs" / "p1_b0_r30_preflight"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_once(path: Path, payload: bytes) -> None:
    """Create an artifact once; an existing byte-identical artifact is reusable."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"refusing to overwrite an existing P1-0C artifact: {path}")
        return
    path.write_bytes(payload)


def _scientific_forecast_for_preflight(
    preflight: RealHistoryPreflight,
) -> DualModelForecast:
    """Adapt the real frozen map to the typed scientific replay grid.

    Only row/column indices are translated to satisfy the synthetic type's
    non-negative index invariant.  Equal-area coordinates, clipped areas,
    cell identities, masses, rankings and alarm masks remain unchanged.
    """

    operational = preflight.domain.operational_grid
    if preflight.recent_source_count != 0:
        raise RuntimeError("the frozen P1-0C historical rehearsal expects an empty R30 window")
    minimum_row = int(min(operational.rows))
    minimum_column = int(min(operational.columns))
    grid = tuple(
        GridCell(
            cell_id=cell_id,
            row=int(row) - minimum_row,
            column=int(column) - minimum_column,
            x_km=float(xy[0]),
            y_km=float(xy[1]),
            area_km2=float(area),
        )
        for cell_id, row, column, xy, area in zip(
            operational.cell_ids,
            operational.rows,
            operational.columns,
            operational.query_xy_km,
            operational.clipped_area_km2,
            strict=True,
        )
    )

    def alarm_prefix(selection: object, *, model_id: str) -> AlarmPrefix:
        if model_id == "B0":
            typed = preflight.b0_alarm
        elif model_id == "B0_R30":
            typed = preflight.challenger_alarm
        else:
            raise AssertionError("unexpected P1 model")
        if selection is not typed:
            raise AssertionError("alarm adapter received the wrong frozen selection")
        return AlarmPrefix(
            model_id=model_id,  # type: ignore[arg-type]
            ranked_cell_ids=tuple(
                operational.cell_ids[int(index)] for index in typed.ranked_indices
            ),
            selected_cell_ids=typed.selected_cell_ids,
            actual_area_km2=typed.actual_area_km2,
            area_cap_km2=typed.area_cap_km2,
            next_complete_cell_area_km2=typed.next_complete_cell_area_km2,
        )

    b0 = RelativeIntensitySurface(
        "B0",
        preflight.b0_mass,
        preflight.b0_source_count,
    )
    recent = RelativeIntensitySurface(
        "R30",
        np.zeros(operational.cell_count, dtype=np.float64),
        preflight.recent_source_count,
    )
    challenger = RelativeIntensitySurface(
        "B0_R30",
        preflight.challenger_mass,
        preflight.b0_source_count,
    )
    return DualModelForecast(
        issue_id="p1-0c-synthetic-raw-replay",
        scheduled_issue_time_utc=PREFLIGHT_SCHEDULED_TIME_UTC,
        query_cutoff_utc=PREFLIGHT_QUERY_CUTOFF_UTC,
        grid=grid,
        B0=b0,
        R30=recent,
        B0_R30=challenger,
        B0_alarm=alarm_prefix(preflight.b0_alarm, model_id="B0"),
        B0_R30_alarm=alarm_prefix(
            preflight.challenger_alarm,
            model_id="B0_R30",
        ),
        recent_fallback_to_B0=preflight.recent_source_count == 0,
    )


def _synthetic_replay_mapping(
    preflight: RealHistoryPreflight,
    *,
    forecast_mapping: dict[str, object],
    forecast_svg_sha256: str,
) -> tuple[dict[str, object], bytes, DualModelForecast, str]:
    """Build a zero-direction known answer from canonical synthetic source bytes."""

    alarm_ids = tuple(preflight.b0_alarm.selected_cell_ids)
    alarm_set = set(alarm_ids)
    operational_grid = preflight.domain.operational_grid
    row_column_by_id = {
        cell_id: (int(row), int(column))
        for cell_id, row, column in zip(
            operational_grid.cell_ids,
            operational_grid.rows,
            operational_grid.columns,
            strict=True,
        )
    }

    def dispersed_ids(
        candidates: tuple[str, ...],
        *,
        count: int,
        occupied: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        remaining = list(candidates)
        selected: list[str] = []
        while len(selected) < count:
            anchors = (*occupied, *selected)
            if not remaining:
                raise RuntimeError("not enough cells for the dispersed synthetic replay")
            if not anchors:
                best_index = 0
            else:
                best_index = max(
                    range(len(remaining)),
                    key=lambda index: (
                        min(
                            (row_column_by_id[remaining[index]][0] - row_column_by_id[anchor][0])
                            ** 2
                            + (row_column_by_id[remaining[index]][1] - row_column_by_id[anchor][1])
                            ** 2
                            for anchor in anchors
                        ),
                        -index,
                    ),
                )
            selected.append(remaining.pop(best_index))
        return tuple(selected)

    outside_ids = tuple(
        cell_id for cell_id in operational_grid.cell_ids if cell_id not in alarm_set
    )
    if len(alarm_ids) < 6 or len(outside_ids) < 4:
        raise RuntimeError("synthetic replay requires six alarmed and four non-alarmed cells")
    chosen_alarm_ids = dispersed_ids(alarm_ids, count=6)
    chosen_outside_ids = dispersed_ids(outside_ids, count=4, occupied=chosen_alarm_ids)
    chosen_ids = (*chosen_alarm_ids, *chosen_outside_ids)
    scientific_forecast = _scientific_forecast_for_preflight(preflight)
    scientific_cell_by_id = {cell.cell_id: cell for cell in scientific_forecast.grid}
    inverse_projection = Transformer.from_crs(
        CRS.from_user_input(EQUAL_AREA_CRS),
        CRS.from_epsg(4326),
        always_xy=True,
    )
    events = tuple(
        SyntheticEvent(
            event_id=f"p1-0c-raw-known-answer-{index:02d}",
            origin_time_utc=PREFLIGHT_SCHEDULED_TIME_UTC + timedelta(days=index + 1),
            available_at_utc=(PREFLIGHT_SCHEDULED_TIME_UTC + timedelta(days=index + 1, minutes=5)),
            x_km=scientific_cell_by_id[cell_id].x_km,
            y_km=scientific_cell_by_id[cell_id].y_km,
            magnitude=5.3,
            source_id="synthetic_ComCat",
            longitude=float(
                inverse_projection.transform(
                    scientific_cell_by_id[cell_id].x_km * 1_000.0,
                    scientific_cell_by_id[cell_id].y_km * 1_000.0,
                )[0]
            ),
            latitude=float(
                inverse_projection.transform(
                    scientific_cell_by_id[cell_id].x_km * 1_000.0,
                    scientific_cell_by_id[cell_id].y_km * 1_000.0,
                )[1]
            ),
        )
        for index, cell_id in enumerate(chosen_ids)
    )
    fetched_at = PREFLIGHT_SCHEDULED_TIME_UTC + timedelta(days=61)
    raw_bytes = build_catalog_snapshot_bytes(
        role="truth",
        issue_id="p1-0c-synthetic-raw-replay",
        scheduled_issue_time_utc=PREFLIGHT_SCHEDULED_TIME_UTC,
        grid=scientific_forecast.grid,
        events=events,
        horizon_days=30,
        truth_fetched_at_utc=fetched_at,
    )
    recomputed = recompute_mature_truth_snapshot(
        raw_bytes,
        scientific_forecast,
        horizon_days=30,
        truth_fetched_at_utc=fetched_at,
    )
    if len(recomputed.clusters) != 10:
        raise RuntimeError("synthetic raw-byte known answer must form ten independent clusters")
    reviews = build_pending_sequential_reviews(
        recomputed.scores,
        elapsed_months=elapsed_tropical_months(PREFLIGHT_SCHEDULED_TIME_UTC, fetched_at),
    )
    if len(reviews) != 1 or reviews[0].review_trigger != "cluster_10":
        raise RuntimeError("ten clusters must produce exactly the frozen cluster_10 review")

    def mechanically_located_cell_id(event: SyntheticEvent) -> str:
        if event.longitude is None or event.latitude is None:
            raise AssertionError("SyntheticEvent always materializes WGS84 coordinates")
        projected_index = preflight.domain.locator.locate_projected(
            event.x_km * 1_000.0,
            event.y_km * 1_000.0,
        )
        geographic_index = preflight.domain.locator.locate_lonlat(
            event.longitude,
            event.latitude,
        )
        if projected_index is None or projected_index != geographic_index:
            raise RuntimeError(
                "synthetic representative coordinates do not locate one frozen 25 km cell"
            )
        return operational_grid.cell_ids[projected_index]

    replay_mapping: dict[str, object] = {
        "replay_id": "p1-0c-synthetic-raw-known-answer-zero",
        "forecast_sha256": forecast_svg_sha256,
        "forecast": forecast_mapping,
        "synthetic_raw_response_sha256": source_snapshot_sha256(raw_bytes),
        "synthetic_raw_response_byte_count": len(raw_bytes),
        "cluster_assignment_sha256": recomputed.cluster_assignment_sha256,
        "ordered_cluster_registry_sha256": recomputed.ordered_cluster_registry_sha256,
        "horizon_days": 30,
        "mature_after_utc": _utc_text(PREFLIGHT_SCHEDULED_TIME_UTC + timedelta(days=60)),
        "replay_created_at_utc": _utc_text(fetched_at),
        "clusters": [
            {
                "cluster_id": cluster.cluster_id,
                "representative_event_id": cluster.representative.event_id,
                "cell_id": mechanically_located_cell_id(cluster.representative),
                "origin_time_utc": _utc_text(cluster.representative.origin_time_utc),
            }
            for cluster in recomputed.clusters
        ],
        "review": reviews[0].as_mapping(),
    }
    return (
        replay_mapping,
        raw_bytes,
        scientific_forecast,
        recomputed.cluster_assignment_sha256,
    )


def _resolve_output_dir(value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = REPOSITORY_ROOT / candidate
    resolved = candidate.resolve()
    repository = REPOSITORY_ROOT.resolve()
    if resolved != repository and repository not in resolved.parents:
        raise ValueError("output directory must stay inside the SeismoFlux repository")
    return resolved


def generate(output_dir: Path) -> dict[str, object]:
    preflight = build_real_history_preflight(
        catalog_bytes=CATALOG_PATH.read_bytes(),
        study_area_bytes=STUDY_AREA_PATH.read_bytes(),
        support_manifest_bytes=SUPPORT_MANIFEST_PATH.read_bytes(),
    )
    forecast_mapping = preflight.as_rendering_mapping()
    forecast_svg = render_preflight_forecast_svg(forecast_mapping)
    forecast_html = build_preflight_forecast_html(forecast_mapping).encode("utf-8")
    forecast_svg_sha = _sha256(forecast_svg)

    forecast_files = {
        "p1_0c_real_history_forecast.svg": forecast_svg,
        "p1_0c_real_history_forecast.html": forecast_html,
    }
    for name, payload in forecast_files.items():
        _write_once(output_dir / name, payload)
    forecast_hashes_before_replay = {
        name: _sha256((output_dir / name).read_bytes()) for name in forecast_files
    }

    (
        replay_mapping,
        raw_truth_bytes,
        scientific_forecast,
        cluster_assignment_sha,
    ) = _synthetic_replay_mapping(
        preflight,
        forecast_mapping=forecast_mapping,
        forecast_svg_sha256=forecast_svg_sha,
    )
    replay_svg = render_preflight_mature_replay_svg(
        replay_mapping,
        raw_truth_bytes=raw_truth_bytes,
        scientific_forecast=scientific_forecast,
        scientific_locator=preflight.domain.locator,
    )
    replay_html = build_preflight_mature_replay_html(
        replay_mapping,
        raw_truth_bytes=raw_truth_bytes,
        scientific_forecast=scientific_forecast,
        scientific_locator=preflight.domain.locator,
    ).encode("utf-8")
    replay_files = {
        "p1_0c_synthetic_truth_snapshot.json": raw_truth_bytes,
        "p1_0c_synthetic_mature_replay.svg": replay_svg,
        "p1_0c_synthetic_mature_replay.html": replay_html,
    }
    for name, payload in replay_files.items():
        _write_once(output_dir / name, payload)

    forecast_hashes_after_replay = {
        name: _sha256((output_dir / name).read_bytes()) for name in forecast_files
    }
    if forecast_hashes_after_replay != forecast_hashes_before_replay:
        raise RuntimeError("mature replay changed a start-time forecast artifact")
    replay_clusters = replay_mapping["clusters"]
    if not isinstance(replay_clusters, list):
        raise AssertionError("synthetic replay clusters must be a list")
    review_mapping = replay_mapping["review"]
    if not isinstance(review_mapping, dict):
        raise AssertionError("synthetic replay review must be a mapping")
    B0_hits = review_mapping.get("B0_hit_clusters")
    challenger_hits = review_mapping.get("B0_R30_hit_clusters")
    if B0_hits != 6 or challenger_hits != 6:
        raise RuntimeError("synthetic raw-byte known answer must produce exactly 6/10 hits")

    result: dict[str, object] = {
        "schema_version": 1,
        "stage_id": "P1-0C",
        "status": "historical_adapter_and_raw_preimage_rehearsal_complete",
        "not_a_real_prospective_issue": True,
        "real_issue_authorized": False,
        "locked_test_run": False,
        "network_accessed": False,
        "catalog_path": CATALOG_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
        "study_area_path": STUDY_AREA_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
        "support_manifest_path": SUPPORT_MANIFEST_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
        "query_cutoff_utc": _utc_text(PREFLIGHT_QUERY_CUTOFF_UTC),
        "real_history_rehearsal": {
            "catalog": {
                "row_count": preflight.catalog.row_count,
                "file_sha256": preflight.catalog.identity.file_sha256,
                "content_sha256": preflight.catalog.identity.content_sha256,
                "schema_sha256": preflight.catalog.identity.schema_sha256,
                "B0_source_count": preflight.b0_source_count,
                "R30_source_count": preflight.recent_source_count,
            },
            "support": preflight.support.as_mapping(),
            "spatial_domain": {
                "study_area_sha256": preflight.study_area_sha256,
                "grid_id": preflight.domain.operational_grid.grid_id,
                "cell_count": preflight.domain.operational_grid.cell_count,
                "cell_size_km": 25.0,
            },
            "method": {
                "kernel": "gaussian_KDE",
                "bandwidth_km": 75.0,
                "challenger_formula": "0.75*B0+0.25*R30",
                "empty_recent_fallback_to_B0": preflight.recent_source_count == 0,
                "value_semantics": "relative_intensity_not_absolute_probability",
            },
            "models": {
                "B0": {
                    "mass_sha256": hashlib.sha256(preflight.b0_mass.tobytes()).hexdigest(),
                    "ranking_sha256": preflight.b0_alarm.ranking_sha256,
                    "alarm_mask_sha256": preflight.b0_alarm.selected_mask_sha256,
                    "selected_cell_count": int(preflight.b0_alarm.selected_indices.size),
                    "actual_alarm_area_km2": preflight.b0_alarm.actual_area_km2,
                    "next_complete_cell_area_km2": (preflight.b0_alarm.next_complete_cell_area_km2),
                },
                "B0_R30": {
                    "mass_sha256": hashlib.sha256(preflight.challenger_mass.tobytes()).hexdigest(),
                    "ranking_sha256": preflight.challenger_alarm.ranking_sha256,
                    "alarm_mask_sha256": preflight.challenger_alarm.selected_mask_sha256,
                    "selected_cell_count": int(preflight.challenger_alarm.selected_indices.size),
                    "actual_alarm_area_km2": preflight.challenger_alarm.actual_area_km2,
                    "next_complete_cell_area_km2": (
                        preflight.challenger_alarm.next_complete_cell_area_km2
                    ),
                },
            },
            "paired_area_difference_km2": preflight.actual_area_difference_km2,
            "empty_recent_bitwise_equal": (
                preflight.b0_mass.tobytes() == preflight.challenger_mass.tobytes()
            ),
        },
        "synthetic_raw_byte_known_answer": {
            "raw_source_sha256": source_snapshot_sha256(raw_truth_bytes),
            "raw_source_byte_count": len(raw_truth_bytes),
            "cluster_assignment_sha256": cluster_assignment_sha,
            "ordered_cluster_registry_sha256": replay_mapping["ordered_cluster_registry_sha256"],
            "independent_cluster_count": len(replay_clusters),
            "B0_hit_clusters": B0_hits,
            "B0_R30_hit_clusters": challenger_hits,
            "observed_direction": "zero",
            "forecast_svg_sha256": forecast_svg_sha,
            "cluster_10_review": review_mapping,
        },
        "start_artifacts_unchanged_after_replay": True,
        "scientific_value": {
            "category": "necessary_enabler",
            "direct_prediction_improvement": "none_new",
            "evidence": (
                "real_frozen_history_recomputes_start_maps_and_synthetic_raw_bytes_recompute_"
                "clusters_without_future_outcomes_in_start_artifacts"
            ),
            "next_decision": "separate_real_issue_authorization_only_after_P1_0C_acceptance",
        },
    }
    result_name = "p1_0c_preflight_result.json"
    _write_once(output_dir / result_name, _json_bytes(result))

    artifact_hashes = {
        name: _sha256((output_dir / name).read_bytes())
        for name in (*forecast_files, *replay_files, result_name)
    }
    manifest = {
        "schema_version": 1,
        "stage_id": "P1-0C",
        "artifact_count": len(artifact_hashes),
        "artifacts": artifact_hashes,
        "forecast_svg_sha256_bound_by_replay": forecast_svg_sha,
        "start_artifacts_unchanged_after_replay": True,
        "scientific_value_category": "necessary_enabler",
        "real_issue_authorized": False,
    }
    _write_once(output_dir / "p1_0c_preflight_manifest.json", _json_bytes(manifest))
    return {
        "output_dir": str(output_dir),
        "catalog_rows": preflight.catalog.row_count,
        "B0_source_count": preflight.b0_source_count,
        "R30_source_count": preflight.recent_source_count,
        "grid_cell_count": preflight.domain.operational_grid.cell_count,
        "B0_actual_area_km2": preflight.b0_alarm.actual_area_km2,
        "B0_R30_actual_area_km2": preflight.challenger_alarm.actual_area_km2,
        "area_difference_km2": preflight.actual_area_difference_km2,
        "forecast_svg_sha256": forecast_svg_sha,
        "artifact_count": len(artifact_hashes) + 1,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="artifact directory inside the repository",
    )
    args = parser.parse_args()
    summary = generate(_resolve_output_dir(args.output_dir))
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
