from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from pyproj import CRS, Transformer

from scripts.run_p1_b0_r30_preflight import _synthetic_replay_mapping
from seismoflux.background.grid import EQUAL_AREA_CRS
from seismoflux.p1_b0_r30.preflight import (
    EXPECTED_SUPPORT_ID,
    EXPECTED_SUPPORT_MANIFEST_SHA256,
    build_real_history_preflight,
    parse_support_water_level,
    reject_future_outcome_fields,
    select_complete_alarm_prefix,
)
from seismoflux.p1_b0_r30.preflight_rendering import (
    P1PreflightRenderingError,
    render_preflight_forecast_svg,
    render_preflight_mature_replay_svg,
)
from seismoflux.stage2s.contracts import SpatialGrid

ROOT = Path(__file__).resolve().parents[2]
SUPPORT_MANIFEST = ROOT / "data" / "manifests" / "background_local_support_manifest.json"


def _small_grid() -> SpatialGrid:
    return SpatialGrid(
        grid_id="p1-preflight-small-grid",
        cell_size_km=25.0,
        cell_ids=("c00", "c01", "c10", "c11"),
        rows=np.asarray([0, 0, 1, 1], dtype=np.int64),
        columns=np.asarray([0, 1, 0, 1], dtype=np.int64),
        query_xy_km=np.asarray(
            [[12.5, 12.5], [37.5, 12.5], [12.5, 37.5], [37.5, 37.5]],
            dtype=np.float64,
        ),
        clipped_area_km2=np.asarray([400.0, 400.0, 400.0, 400.0], dtype=np.float64),
    )


def test_complete_alarm_prefix_is_no_skip_and_records_next_cell() -> None:
    selection = select_complete_alarm_prefix(
        np.asarray([0.4, 0.3, 0.2, 0.1], dtype=np.float64),
        _small_grid(),
        model_id="B0",
        area_cap_km2=1_000.0,
    )

    assert selection.selected_cell_ids == ("c00", "c01")
    assert selection.actual_area_km2 == 800.0
    assert selection.next_complete_cell_area_km2 == 400.0
    assert selection.ranked_indices.tolist() == [0, 1, 2, 3]
    assert len(selection.ranking_sha256) == len(selection.selected_mask_sha256) == 64


def test_complete_alarm_prefix_tie_break_is_row_column_then_cell_id() -> None:
    selection = select_complete_alarm_prefix(
        np.asarray([0.25, 0.25, 0.25, 0.25], dtype=np.float64),
        _small_grid(),
        model_id="B0_R30",
        area_cap_km2=800.0,
    )

    assert selection.selected_cell_ids == ("c00", "c01")
    assert selection.actual_area_km2 == 800.0


@pytest.mark.parametrize(
    "payload",
    [
        {"targets": []},
        {"nested": {"truth": {}}},
        {"items": [{"recall": 0.5}]},
        {"review": "looks_good"},
        {"clusters": [{"hits": 1}]},
        {"target_event_count": 3},
        {"future_targets": []},
        {"outcome_summary": {}},
        {"score_rows": []},
        {"effect_size": 0.1},
        {"hit_rate": 0.5},
        {"review_status": "pending"},
    ],
)
def test_start_payload_rejects_future_outcome_fields(payload: object) -> None:
    with pytest.raises(ValueError, match="future outcome field"):
        reject_future_outcome_fields(payload)


def test_start_payload_accepts_only_pre_issue_scientific_inputs() -> None:
    reject_future_outcome_fields(
        {
            "issue_id": "preflight",
            "query_cutoff_utc": "2026-09-09T15:45:00Z",
            "models": {
                "B0": {"relative_intensity": [0.2, 0.8]},
                "B0_R30": {"relative_intensity": [0.3, 0.7]},
            },
            "future_outcomes_absent": True,
        }
    )


def test_frozen_support_water_level_is_independently_recomputed() -> None:
    result = parse_support_water_level(SUPPORT_MANIFEST.read_bytes())

    assert result.manifest_sha256 == EXPECTED_SUPPORT_MANIFEST_SHA256
    assert result.support_id == EXPECTED_SUPPORT_ID
    assert result.common_mc == 4.0
    assert result.fit_end_utc == "2023-06-30T16:00:00.000000Z"
    assert (result.supported_cell_count, result.indeterminate_cell_count) == (52, 9)
    assert result.unsupported_cell_count == 0
    assert result.retained_area_fraction == 1.0
    assert len(result.status_by_base_row_column()) == 61
    negative_parent_status = result.status_by_base_row_column()[(4, -2)]
    assert result.status_for_25km_cell(row=80, column=-40) == negative_parent_status
    assert result.status_for_25km_cell(row=99, column=-21) == negative_parent_status


def test_support_bytes_fail_closed_after_tampering() -> None:
    payload = SUPPORT_MANIFEST.read_bytes()
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        parse_support_water_level(payload + b"\n")


def test_real_grid_replay_preserves_coordinates_and_mechanically_locates_hits() -> None:
    preflight = build_real_history_preflight(
        catalog_bytes=(
            ROOT / "data" / "processed" / "stage1" / "debc98054172a4a1" / "earthquake_event.parquet"
        ).read_bytes(),
        study_area_bytes=(ROOT / "data" / "processed" / "china_mainland.geojson").read_bytes(),
        support_manifest_bytes=SUPPORT_MANIFEST.read_bytes(),
    )
    forecast_mapping = preflight.as_rendering_mapping()
    forecast_sha = hashlib.sha256(render_preflight_forecast_svg(forecast_mapping)).hexdigest()
    replay, raw, scientific, _ = _synthetic_replay_mapping(
        preflight,
        forecast_mapping=forecast_mapping,
        forecast_svg_sha256=forecast_sha,
    )
    operational = preflight.domain.operational_grid
    minimum_row = int(min(operational.rows))
    minimum_column = int(min(operational.columns))
    for index, cell in enumerate(scientific.grid):
        assert cell.row == int(operational.rows[index]) - minimum_row
        assert cell.column == int(operational.columns[index]) - minimum_column
        assert cell.x_km == float(operational.query_xy_km[index, 0])
        assert cell.y_km == float(operational.query_xy_km[index, 1])

    decoded = json.loads(raw)
    events = {item["event_id"]: item for item in decoded["events"]}
    forward = Transformer.from_crs(
        CRS.from_epsg(4326),
        CRS.from_user_input(EQUAL_AREA_CRS),
        always_xy=True,
    )
    replay_clusters = replay["clusters"]
    assert isinstance(replay_clusters, list)
    for cluster in replay_clusters:
        assert isinstance(cluster, dict)
        event = events[cluster["representative_event_id"]]
        x_m, y_m = forward.transform(event["longitude"], event["latitude"])
        assert x_m / 1_000.0 == pytest.approx(event["x_km"], abs=1e-9)
        assert y_m / 1_000.0 == pytest.approx(event["y_km"], abs=1e-9)
        projected = preflight.domain.locator.locate_projected(x_m, y_m)
        geographic = preflight.domain.locator.locate_lonlat(event["longitude"], event["latitude"])
        assert projected is not None and projected == geographic
        assert operational.cell_ids[projected] == cluster["cell_id"]

    review = replay["review"]
    assert isinstance(review, dict)
    assert review["review_trigger"] == "cluster_10"
    assert review["cumulative_cluster_count"] == 10
    assert (review["B0_hit_clusters"], review["B0_R30_hit_clusters"]) == (6, 6)
    render_preflight_mature_replay_svg(
        replay,
        raw_truth_bytes=raw,
        scientific_forecast=scientific,
        scientific_locator=preflight.domain.locator,
    )

    forged = copy.deepcopy(replay)
    forged_clusters = forged["clusters"]
    assert isinstance(forged_clusters, list)
    first = forged_clusters[0]
    assert isinstance(first, dict)
    first["cell_id"] = next(
        cell_id for cell_id in operational.cell_ids if cell_id != first["cell_id"]
    )
    with pytest.raises(P1PreflightRenderingError, match="display cluster rows differ"):
        render_preflight_mature_replay_svg(
            forged,
            raw_truth_bytes=raw,
            scientific_forecast=scientific,
            scientific_locator=preflight.domain.locator,
        )
