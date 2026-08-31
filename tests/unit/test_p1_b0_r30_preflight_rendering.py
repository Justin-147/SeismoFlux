# ruff: noqa: RUF001
"""Scientific visual checks for the target-blind P1-0C preflight package."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Literal, cast

import numpy as np
import pytest
from pyproj import CRS, Geod, Transformer
from shapely.geometry import box

from seismoflux.background.grid import EQUAL_AREA_CRS
from seismoflux.d1_replay.spatial import Frozen25kmCellLocator
from seismoflux.data.common import canonical_json_bytes
from seismoflux.p1_b0_r30.core import (
    DualModelForecast,
    GridCell,
    SyntheticEvent,
    build_pending_sequential_reviews,
    elapsed_tropical_months,
)
from seismoflux.p1_b0_r30.preflight_rendering import (
    P1PreflightRenderingError,
    build_preflight_forecast_html,
    build_preflight_mature_replay_html,
    render_preflight_forecast_svg,
    render_preflight_mature_replay_svg,
)
from seismoflux.p1_b0_r30.preimage import (
    build_catalog_snapshot_bytes,
    recompute_mature_truth_snapshot,
    source_snapshot_sha256,
)
from seismoflux.p1_b0_r30.synthetic import build_synthetic_scenario
from seismoflux.stage2s.contracts import SpatialGrid

_CELL_COUNT = 1_000
_ALARM_CELL_COUNT = 960


def _cells() -> list[dict[str, object]]:
    return [
        {
            "cell_id": f"c{index:04d}",
            "row": index // 40,
            "column": index % 40,
            "area_km2": 625.0,
            "support_status": "indeterminate" if index % 113 == 0 else "supported",
        }
        for index in range(_CELL_COUNT)
    ]


def _model(weights: Sequence[float]) -> dict[str, object]:
    assert len(weights) == _CELL_COUNT
    total = math.fsum(weights)
    intensity = [value / total for value in weights]
    ranking = sorted(
        range(_CELL_COUNT),
        key=lambda index: (-intensity[index], index // 40, index % 40, f"c{index:04d}"),
    )
    alarm_ids = [f"c{index:04d}" for index in ranking[:_ALARM_CELL_COUNT]]
    return {
        "normalized_cell_mass": intensity,
        "alarm_cell_ids": alarm_ids,
        "actual_alarm_area_km2": 600_000.0,
        "next_complete_cell_area_km2": 625.0,
    }


def _forecast() -> dict[str, object]:
    B0_weights = [float(_CELL_COUNT - index) for index in range(_CELL_COUNT)]
    challenger_weights = [float(index + 1) for index in range(_CELL_COUNT)]
    return {
        "rehearsal_id": "p1-0c-historical-cutoff-rehearsal",
        "scheduled_issue_time_utc": "2026-07-09T04:40:56Z",
        "query_cutoff_utc": "2026-07-09T04:25:56Z",
        "catalog": {
            "path": "data/processed/stage1/debc98054172a4a1/earthquake_event.parquet",
            "sha256": "2" * 64,
            "row_count": 40_898,
            "eligible_B0_event_count": 5_991,
            "R30_event_count": 19,
            "origin_time_max_utc": "2026-07-09T04:25:56Z",
            "available_at_max_utc": "2026-07-09T04:25:56Z",
        },
        "support": {
            "support_id": "local-support-f6816ab6c6581306",
            "manifest_sha256": "6" * 64,
            "fit_end_utc": "2023-06-30T16:00:00Z",
            "common_mc": 4.0,
            "fixed_cell_count": 61,
            "supported_cell_count": 52,
            "indeterminate_cell_count": 9,
            "unsupported_cell_count": 0,
            "retained_area_km2": 9_415_305.754,
        },
        "grid": {"cell_size_km": 25.0, "cells": _cells()},
        "models": {
            "B0": _model(B0_weights),
            "B0_R30": _model(challenger_weights),
        },
    }


def _unequal_area_forecast() -> dict[str, object]:
    forecast = _forecast()
    forecast["grid"] = {
        "cell_size_km": 25.0,
        "cells": [
            {
                "cell_id": "boundary-small",
                "row": 0,
                "column": 0,
                "area_km2": 100.0,
                "support_status": "supported",
            },
            {
                "cell_id": "boundary-medium",
                "row": 0,
                "column": 1,
                "area_km2": 500.0,
                "support_status": "supported",
            },
            {
                "cell_id": "complete",
                "row": 0,
                "column": 2,
                "area_km2": 625.0,
                "support_status": "supported",
            },
        ],
    }
    forecast["models"] = {
        "B0": {
            "normalized_cell_mass": [0.2, 0.5, 0.3],
            "alarm_cell_ids": ["boundary-small", "boundary-medium", "complete"],
            "actual_alarm_area_km2": 1_225.0,
            "next_complete_cell_area_km2": None,
        },
        "B0_R30": {
            "normalized_cell_mass": [0.1, 0.4, 0.5],
            "alarm_cell_ids": ["boundary-small", "boundary-medium", "complete"],
            "actual_alarm_area_km2": 1_225.0,
            "next_complete_cell_area_km2": None,
        },
    }
    return forecast


def _clusters(cell_ids: Sequence[str], *, prefix: str) -> list[dict[str, object]]:
    return [
        {
            "cluster_id": f"{prefix}-cluster-{index:02d}",
            "representative_event_id": f"{prefix}-event-{index:02d}",
            "cell_id": cell_id,
            "origin_time_utc": f"2026-07-{10 + index:02d}T04:40:56Z",
        }
        for index, cell_id in enumerate(cell_ids)
    ]


def _replay(
    forecast: Mapping[str, object], clusters: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    forecast_sha = hashlib.sha256(render_preflight_forecast_svg(forecast)).hexdigest()
    return {
        "replay_id": "p1-0c-synthetic-opposite-outcomes",
        "forecast_sha256": forecast_sha,
        "forecast": copy.deepcopy(forecast),
        "synthetic_raw_response_sha256": "a" * 64,
        "synthetic_raw_response_byte_count": 4_096,
        "horizon_days": 30,
        "mature_after_utc": "2026-09-07T04:40:56Z",
        "replay_created_at_utc": "2026-09-08T04:40:56Z",
        "clusters": [dict(item) for item in clusters],
    }


def _forecast_from_scientific(scientific: DualModelForecast) -> dict[str, object]:
    return {
        "rehearsal_id": "p1-0c-synthetic-renderer-contract",
        "scheduled_issue_time_utc": scientific.scheduled_issue_time_utc.isoformat().replace(
            "+00:00", "Z"
        ),
        "query_cutoff_utc": scientific.query_cutoff_utc.isoformat().replace("+00:00", "Z"),
        "catalog": {
            "path": "synthetic/catalogue.json",
            "sha256": "2" * 64,
            "row_count": scientific.B0.active_event_count,
            "eligible_B0_event_count": scientific.B0.active_event_count,
            "R30_event_count": scientific.R30.active_event_count,
            "origin_time_max_utc": scientific.query_cutoff_utc.isoformat().replace("+00:00", "Z"),
            "available_at_max_utc": scientific.query_cutoff_utc.isoformat().replace("+00:00", "Z"),
        },
        "support": {
            "support_id": "synthetic-support",
            "manifest_sha256": "6" * 64,
            "fit_end_utc": "2023-06-30T16:00:00Z",
            "common_mc": 4.0,
            "fixed_cell_count": 1,
            "supported_cell_count": 1,
            "indeterminate_cell_count": 0,
            "unsupported_cell_count": 0,
            "retained_area_km2": sum(cell.area_km2 for cell in scientific.grid),
        },
        "grid": {
            "cell_size_km": 25.0,
            "cells": [
                {
                    "cell_id": cell.cell_id,
                    "row": cell.row,
                    "column": cell.column,
                    "area_km2": cell.area_km2,
                    "support_status": "supported",
                }
                for cell in scientific.grid
            ],
        },
        "models": {
            "B0": {
                "normalized_cell_mass": scientific.B0.relative_intensity.tolist(),
                "alarm_cell_ids": list(scientific.B0_alarm.selected_cell_ids),
                "actual_alarm_area_km2": scientific.B0_alarm.actual_area_km2,
                "next_complete_cell_area_km2": (scientific.B0_alarm.next_complete_cell_area_km2),
            },
            "B0_R30": {
                "normalized_cell_mass": scientific.B0_R30.relative_intensity.tolist(),
                "alarm_cell_ids": list(scientific.B0_R30_alarm.selected_cell_ids),
                "actual_alarm_area_km2": scientific.B0_R30_alarm.actual_area_km2,
                "next_complete_cell_area_km2": (
                    scientific.B0_R30_alarm.next_complete_cell_area_km2
                ),
            },
        },
    }


def _locator_for_scientific(scientific: DualModelForecast) -> Frozen25kmCellLocator:
    grid = SpatialGrid(
        grid_id="synthetic-renderer-grid",
        cell_size_km=25.0,
        cell_ids=tuple(cell.cell_id for cell in scientific.grid),
        rows=np.asarray([cell.row for cell in scientific.grid], dtype=np.int64),
        columns=np.asarray([cell.column for cell in scientific.grid], dtype=np.int64),
        query_xy_km=np.asarray(
            [[cell.x_km, cell.y_km] for cell in scientific.grid],
            dtype=np.float64,
        ),
        clipped_area_km2=np.asarray(
            [cell.area_km2 for cell in scientific.grid],
            dtype=np.float64,
        ),
    )
    geometries = tuple(
        box(
            cell.x_km * 1_000.0 - 12_500.0,
            cell.y_km * 1_000.0 - 12_500.0,
            cell.x_km * 1_000.0 + 12_500.0,
            cell.y_km * 1_000.0 + 12_500.0,
        )
        for cell in scientific.grid
    )
    return Frozen25kmCellLocator(grid=grid, clipped_geometries=geometries)


@lru_cache(maxsize=3)
def _verified_replay(
    direction: str,
) -> tuple[
    dict[str, object],
    bytes,
    DualModelForecast,
    Frozen25kmCellLocator,
]:
    scenario = build_synthetic_scenario(cast(Literal["positive", "zero", "negative"], direction))
    scientific = scenario.forecast
    locator = _locator_for_scientific(scientific)
    inverse = Transformer.from_crs(
        CRS.from_user_input(EQUAL_AREA_CRS),
        CRS.from_epsg(4326),
        always_xy=True,
    )
    lonlat_by_id: dict[str, tuple[float, float]] = {}
    for cell in scientific.grid:
        longitude, latitude = inverse.transform(cell.x_km * 1_000.0, cell.y_km * 1_000.0)
        lonlat_by_id[cell.cell_id] = (float(longitude), float(latitude))
    geod = Geod(ellps="WGS84")

    def distance_m(left: GridCell, right: GridCell) -> float:
        left_lon, left_lat = lonlat_by_id[left.cell_id]
        right_lon, right_lat = lonlat_by_id[right.cell_id]
        return abs(float(geod.inv(left_lon, left_lat, right_lon, right_lat)[2]))

    def select_spaced(
        candidates: tuple[GridCell, ...],
        *,
        count: int,
        occupied: tuple[GridCell, ...] = (),
    ) -> tuple[GridCell, ...]:
        remaining = list(candidates)
        selected: list[GridCell] = []
        while len(selected) < count:
            anchors = (*occupied, *selected)
            viable = [
                cell
                for cell in remaining
                if all(distance_m(cell, anchor) > 75_000.0 for anchor in anchors)
            ]
            if not viable:
                raise AssertionError("synthetic renderer fixture lacks separated cells")
            chosen = (
                viable[0]
                if not anchors
                else max(
                    viable,
                    key=lambda cell: min(distance_m(cell, anchor) for anchor in anchors),
                )
            )
            selected.append(chosen)
            remaining.remove(chosen)
        return tuple(selected)

    B0 = set(scientific.B0_alarm.selected_cell_ids)
    challenger = set(scientific.B0_R30_alarm.selected_cell_ids)
    by_id = {cell.cell_id: cell for cell in scientific.grid}

    def group(ids: set[str]) -> tuple[GridCell, ...]:
        return tuple(by_id[cell_id] for cell_id in sorted(ids))

    if direction == "positive":
        first = select_spaced(group(challenger - B0), count=1)
        common = select_spaced(group(B0 & challenger), count=5, occupied=first)
        target_cells = (
            *first,
            *common,
            *select_spaced(
                group(set(by_id) - (B0 | challenger)),
                count=4,
                occupied=(*first, *common),
            ),
        )
    elif direction == "negative":
        first = select_spaced(group(B0 - challenger), count=1)
        common = select_spaced(group(B0 & challenger), count=5, occupied=first)
        target_cells = (
            *first,
            *common,
            *select_spaced(
                group(set(by_id) - (B0 | challenger)),
                count=4,
                occupied=(*first, *common),
            ),
        )
    else:
        first = select_spaced(group(set(by_id) - (B0 | challenger)), count=5)
        target_cells = (*first, *select_spaced(group(B0 & challenger), count=5, occupied=first))
    target_events = tuple(
        SyntheticEvent(
            event_id=f"renderer-{direction}-{index:02d}",
            origin_time_utc=scientific.scheduled_issue_time_utc + timedelta(days=1 + 2 * index),
            available_at_utc=(
                scientific.scheduled_issue_time_utc + timedelta(days=1 + 2 * index, minutes=5)
            ),
            x_km=cell.x_km,
            y_km=cell.y_km,
            magnitude=5.3,
            source_id="synthetic_ComCat",
            longitude=lonlat_by_id[cell.cell_id][0],
            latitude=lonlat_by_id[cell.cell_id][1],
        )
        for index, cell in enumerate(target_cells)
    )
    fetched = scientific.scheduled_issue_time_utc + timedelta(days=61)
    raw = build_catalog_snapshot_bytes(
        role="truth",
        issue_id=scientific.issue_id,
        scheduled_issue_time_utc=scientific.scheduled_issue_time_utc,
        grid=scientific.grid,
        events=target_events,
        horizon_days=30,
        truth_fetched_at_utc=fetched,
    )
    recomputed = recompute_mature_truth_snapshot(
        raw,
        scientific,
        horizon_days=30,
        truth_fetched_at_utc=fetched,
    )
    review = build_pending_sequential_reviews(
        recomputed.scores,
        elapsed_months=elapsed_tropical_months(scientific.scheduled_issue_time_utc, fetched),
    )
    assert len(review) == 1 and review[0].review_trigger == "cluster_10"
    forecast = _forecast_from_scientific(scientific)
    forecast_sha = hashlib.sha256(render_preflight_forecast_svg(forecast)).hexdigest()
    replay: dict[str, object] = {
        "replay_id": f"p1-0c-synthetic-{direction}",
        "forecast_sha256": forecast_sha,
        "forecast": forecast,
        "synthetic_raw_response_sha256": source_snapshot_sha256(raw),
        "synthetic_raw_response_byte_count": len(raw),
        "cluster_assignment_sha256": recomputed.cluster_assignment_sha256,
        "ordered_cluster_registry_sha256": recomputed.ordered_cluster_registry_sha256,
        "horizon_days": 30,
        "mature_after_utc": (scientific.scheduled_issue_time_utc + timedelta(days=60))
        .isoformat()
        .replace("+00:00", "Z"),
        "replay_created_at_utc": fetched.isoformat().replace("+00:00", "Z"),
        "clusters": [
            {
                "cluster_id": cluster.cluster_id,
                "representative_event_id": cluster.representative.event_id,
                "cell_id": locator.grid.cell_ids[
                    cast(
                        int,
                        locator.locate_projected(
                            cluster.representative.x_km * 1_000.0,
                            cluster.representative.y_km * 1_000.0,
                        ),
                    )
                ],
                "origin_time_utc": cluster.representative.origin_time_utc.isoformat().replace(
                    "+00:00", "Z"
                ),
            }
            for cluster in recomputed.clusters
        ],
        "review": review[0].as_mapping(),
    }
    return replay, raw, scientific, locator


def _render_verified_svg(
    replay: Mapping[str, object],
    raw: bytes,
    scientific: DualModelForecast,
    locator: Frozen25kmCellLocator,
) -> bytes:
    return render_preflight_mature_replay_svg(
        replay,
        raw_truth_bytes=raw,
        scientific_forecast=scientific,
        scientific_locator=locator,
    )


def _render_verified_html(
    replay: Mapping[str, object],
    raw: bytes,
    scientific: DualModelForecast,
    locator: Frozen25kmCellLocator,
) -> str:
    return build_preflight_mature_replay_html(
        replay,
        raw_truth_bytes=raw,
        scientific_forecast=scientific,
        scientific_locator=locator,
    )


def _embedded_json(document: str, element_id: str) -> object:
    marker = f'<script type="application/json" id="{element_id}">'
    return json.loads(document.split(marker, maxsplit=1)[1].split("</script>", maxsplit=1)[0])


def test_elapsed_months_use_one_frozen_average_tropical_month_formula() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)

    assert elapsed_tropical_months(start, start + timedelta(days=365.2425)) == pytest.approx(
        12.0,
        abs=1e-12,
    )
    with pytest.raises(ValueError, match="must not precede"):
        elapsed_tropical_months(start, start - timedelta(seconds=1))


def _all_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            keys.add(str(key).casefold())
            keys.update(_all_keys(child))
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for child in value:
            keys.update(_all_keys(child))
    return keys


def _svg_cell_fill(document: str, *, model_id: str, cell_id: str) -> str:
    panel = document.split(f'data-model="{model_id}"', maxsplit=1)[1].split("</g>", maxsplit=1)[0]
    match = re.search(rf'<rect [^>]*fill="([^"]+)"[^>]*data-cell-id="{re.escape(cell_id)}"', panel)
    assert match is not None
    return match.group(1)


def test_forecast_svg_and_html_are_deterministic_shared_scale_and_scientifically_plain() -> None:
    forecast = _forecast()
    first_svg = render_preflight_forecast_svg(forecast)
    second_svg = render_preflight_forecast_svg(copy.deepcopy(forecast))
    first_html = build_preflight_forecast_html(forecast)
    second_html = build_preflight_forecast_html(copy.deepcopy(forecast))

    assert first_svg == second_svg
    assert first_html == second_html
    svg_text = first_svg.decode("utf-8")
    for expected in (
        "历史适配演练，不是真实前瞻 issue",
        "2026-07-09T04:40:56Z",
        "2026-07-09T04:25:56Z",
        "B0 / R30 入选 5,991 / 19",
        "support 水位",
        "local-support-f6816ab6c6581306",
        "Mc=4.0",
        "75 km Gaussian KDE",
        "B0_R30 = 0.75 × B0 + 0.25 × R30",
        "颜色 = normalized_cell_mass / 实际格面积",
        "实际报警面积",
        "下一完整格",
        "面积公平",
        "相对强度不是绝对发震概率",
        'data-artifact="p1-0c-preflight-forecast"',
    ):
        assert expected in svg_text
    shared = re.search(r'data-shared-colour-max="([^"]+)"', svg_text)
    assert shared is not None
    assert svg_text.count(f'data-colour-max="{shared.group(1)}"') == 2

    for expected in (
        'id="model-select"',
        'id="intensity-toggle"',
        'id="alarm-toggle"',
        'id="support-toggle"',
        'id="cell-inspector"',
        'id="forecast-data"',
        "shared_colour_maximum",
        "normalized_cell_mass",
        "relative_intensity_per_km2",
        "origin_time_max_utc",
        "available_at_max_utc",
    ):
        assert expected in first_html
    for forbidden in (
        "http://",
        "https://",
        "<script src=",
        "<link rel=",
        "fetch(",
        "XMLHttpRequest",
        "WebSocket",
    ):
        assert forbidden not in first_html

    for artifact in (svg_text, first_html):
        for forbidden_semantic in (
            "targets",
            "truth",
            "cluster_id",
            "hit_count",
            "miss_count",
            "recall",
            "gain",
            "review",
            "data-outcome",
            "命中",
            "漏报",
            "召回",
            "震群",
        ):
            assert forbidden_semantic not in artifact

    payload = _embedded_json(first_html, "forecast-data")
    payload_keys = _all_keys(payload)
    for forbidden_key in (
        "targets",
        "truth",
        "clusters",
        "hit_count",
        "miss_count",
        "recall",
        "gain",
        "review",
    ):
        assert forbidden_key not in payload_keys


@pytest.mark.parametrize(
    "future_key",
    ["truth", "targets", "cluster_registry", "hit_count", "missed_ids", "recall", "gain", "review"],
)
def test_forecast_input_fails_closed_on_every_future_outcome_field(future_key: str) -> None:
    forecast = _forecast()
    forecast[future_key] = {"sentinel": "SHOULD_NOT_ENTER_FORECAST"}
    with pytest.raises(P1PreflightRenderingError, match="forbidden future-outcome field"):
        render_preflight_forecast_svg(forecast)


def test_two_opposite_mature_outcomes_cannot_enter_forecast_input() -> None:
    challenger_only = _clusters(("c0980", "c0990"), prefix="challenger")
    B0_only = _clusters(("c0005", "c0015"), prefix="baseline")
    for clusters in (challenger_only, B0_only):
        contaminated = _forecast()
        contaminated["targets"] = clusters
        with pytest.raises(P1PreflightRenderingError, match="forbidden future-outcome field"):
            build_preflight_forecast_html(contaminated)


def test_nested_future_field_is_rejected_instead_of_silently_ignored() -> None:
    forecast = _forecast()
    catalog = cast(dict[str, object], forecast["catalog"])
    catalog["truth_snapshot_sha256"] = "f" * 64
    with pytest.raises(P1PreflightRenderingError, match="truth_snapshot_sha256"):
        render_preflight_forecast_svg(forecast)


def test_old_relative_intensity_input_key_is_rejected() -> None:
    forecast = _forecast()
    models = cast(dict[str, object], forecast["models"])
    B0 = cast(dict[str, object], models["B0"])
    B0["relative_intensity"] = B0.pop("normalized_cell_mass")

    with pytest.raises(P1PreflightRenderingError, match="keys differ from the frozen schema"):
        render_preflight_forecast_svg(forecast)


def test_legal_issue_time_surface_change_changes_forecast_bytes() -> None:
    forecast = _forecast()
    baseline = render_preflight_forecast_svg(forecast)
    changed = copy.deepcopy(forecast)
    models = cast(dict[str, object], changed["models"])
    B0 = cast(dict[str, object], models["B0"])
    weights = [float(_CELL_COUNT - index) for index in range(_CELL_COUNT)]
    weights[0] += 100.0
    B0.clear()
    B0.update(_model(weights))
    assert render_preflight_forecast_svg(changed) != baseline


def test_forecast_accepts_signed_operational_grid_coordinates() -> None:
    forecast = _forecast()
    cells = cast(dict[str, object], forecast["grid"])["cells"]
    for cell in cast(list[dict[str, object]], cells):
        cell["row"] = cast(int, cell["row"]) - 20
        cell["column"] = cast(int, cell["column"]) - 50

    svg = render_preflight_forecast_svg(forecast)
    html_document = build_preflight_forecast_html(forecast)

    assert svg.startswith(b"<svg")
    embedded = cast(dict[str, object], _embedded_json(html_document, "forecast-data"))
    assert embedded["minimum_row"] == -20
    assert embedded["minimum_column"] == -50


def test_unequal_boundary_cells_use_mass_per_exact_area_for_rank_colour_and_shared_scale() -> None:
    forecast = _unequal_area_forecast()
    svg_text = render_preflight_forecast_svg(forecast).decode("utf-8")
    html_document = build_preflight_forecast_html(forecast)
    payload = cast(dict[str, object], _embedded_json(html_document, "forecast-data"))
    models = cast(dict[str, object], payload["models"])
    B0 = cast(dict[str, object], models["B0"])
    challenger = cast(dict[str, object], models["B0_R30"])

    assert B0["normalized_cell_mass"] == [0.2, 0.5, 0.3]
    assert B0["relative_intensity_per_km2"] == pytest.approx([0.002, 0.001, 0.00048])
    assert challenger["relative_intensity_per_km2"] == pytest.approx([0.001, 0.0008, 0.0008])
    assert cast(dict[str, int], B0["rank_by_cell"])["boundary-small"] == 1
    assert cast(list[str], B0["alarm_cell_ids"])[0] == "boundary-small"
    assert payload["shared_colour_maximum"] == pytest.approx(0.002)

    shared = re.search(r'data-shared-colour-max="([^"]+)"', svg_text)
    assert shared is not None
    assert shared.group(1) == "0.002"
    assert svg_text.count('data-colour-max="0.002"') == 2
    assert _svg_cell_fill(svg_text, model_id="B0", cell_id="boundary-small") == ("rgb(30,92,157)")
    assert _svg_cell_fill(svg_text, model_id="B0", cell_id="boundary-medium") == (
        _svg_cell_fill(svg_text, model_id="B0_R30", cell_id="boundary-small")
    )
    assert 'data-relative-intensity-per-km2="0.002"' in svg_text
    assert "单位面积相对强度 0.002 /km²；排名 1" in svg_text


def test_forecast_rejects_catalog_waterline_after_Q_and_false_area_summary() -> None:
    late = _forecast()
    catalog = cast(dict[str, object], late["catalog"])
    catalog["available_at_max_utc"] = "2026-07-09T04:25:57Z"
    with pytest.raises(P1PreflightRenderingError, match="must not exceed query cutoff"):
        render_preflight_forecast_svg(late)

    false_area = _forecast()
    models = cast(dict[str, object], false_area["models"])
    B0 = cast(dict[str, object], models["B0"])
    B0["actual_alarm_area_km2"] = 599_999.0
    with pytest.raises(P1PreflightRenderingError, match="disagrees with cells"):
        render_preflight_forecast_svg(false_area)


def test_separate_mature_replays_bind_forecast_hash_and_recover_opposite_outcomes() -> None:
    positive, positive_raw, scientific, locator = _verified_replay("positive")
    negative, negative_raw, negative_scientific, negative_locator = _verified_replay("negative")
    forecast = cast(Mapping[str, object], positive["forecast"])
    original_svg = render_preflight_forecast_svg(forecast)
    original_sha = hashlib.sha256(original_svg).hexdigest()
    challenger_svg = _render_verified_svg(positive, positive_raw, scientific, locator)
    baseline_svg = _render_verified_svg(
        negative,
        negative_raw,
        negative_scientific,
        negative_locator,
    )
    challenger_html = _render_verified_html(positive, positive_raw, scientific, locator)
    baseline_html = _render_verified_html(
        negative,
        negative_raw,
        negative_scientific,
        negative_locator,
    )

    assert challenger_svg != baseline_svg
    assert challenger_html != baseline_html
    assert render_preflight_forecast_svg(forecast) == original_svg
    assert hashlib.sha256(render_preflight_forecast_svg(forecast)).hexdigest() == original_sha
    for artifact in (challenger_svg.decode("utf-8"), challenger_html):
        assert original_sha in artifact
        assert "合成原始字节成熟回放" in artifact
        assert "另存回放，不是真实预测效果证据" in artifact
        assert "命中" in artifact
        assert "漏报" in artifact
        assert "召回" in artifact
        assert "相对强度不是绝对发震概率" in artifact
    assert "B0：命中 5、漏报 5、召回 50.0%" in challenger_svg.decode("utf-8")
    assert "B0_R30：命中 6、漏报 4、召回 60.0%" in challenger_svg.decode("utf-8")
    assert "B0：命中 6、漏报 4、召回 60.0%" in baseline_svg.decode("utf-8")
    assert "B0_R30：命中 5、漏报 5、召回 50.0%" in baseline_svg.decode("utf-8")
    assert "冻结 cluster_10 复审" in challenger_svg.decode("utf-8")

    for document in (challenger_html, baseline_html):
        for forbidden in (
            "http://",
            "https://",
            "<script src=",
            "<link rel=",
            "fetch(",
            "XMLHttpRequest",
            "WebSocket",
        ):
            assert forbidden not in document
        payload = _embedded_json(document, "replay-data")
        assert isinstance(payload, dict)
        assert payload["forecast_sha256"] == original_sha
        assert cast(dict[str, object], payload["review"])["review_trigger"] == "cluster_10"


def test_mature_replay_fails_closed_on_wrong_forecast_hash_or_early_creation() -> None:
    original, raw, scientific, locator = _verified_replay("zero")
    replay = copy.deepcopy(original)
    replay["forecast_sha256"] = "f" * 64
    with pytest.raises(P1PreflightRenderingError, match="exact preflight forecast SVG"):
        _render_verified_svg(replay, raw, scientific, locator)

    early = copy.deepcopy(original)
    early["replay_created_at_utc"] = "2026-09-06T04:40:56Z"
    with pytest.raises(P1PreflightRenderingError, match="cannot be created before maturity"):
        _render_verified_html(early, raw, scientific, locator)


def test_mature_replay_rejects_legal_but_forged_cluster_cell() -> None:
    original, raw, scientific, locator = _verified_replay("zero")
    replay = copy.deepcopy(original)
    clusters = cast(list[dict[str, object]], replay["clusters"])
    original_cell = clusters[0]["cell_id"]
    replacement = next(cell.cell_id for cell in scientific.grid if cell.cell_id != original_cell)
    clusters[0]["cell_id"] = replacement
    with pytest.raises(P1PreflightRenderingError, match="display cluster rows differ"):
        _render_verified_svg(replay, raw, scientific, locator)


def test_mature_replay_rejects_raw_hash_assignment_registry_and_review_tampering() -> None:
    original, raw, scientific, locator = _verified_replay("zero")

    with pytest.raises(P1PreflightRenderingError, match="SHA-256 differs"):
        _render_verified_svg(original, raw + b"\n", scientific, locator)

    forged_raw_mapping = json.loads(raw)
    forged_events = cast(list[dict[str, object]], forged_raw_mapping["events"])
    forged_events[0]["magnitude"] = 5.4
    forged_raw = canonical_json_bytes(forged_raw_mapping)
    forged_source_identity = copy.deepcopy(original)
    forged_source_identity["synthetic_raw_response_sha256"] = source_snapshot_sha256(forged_raw)
    forged_source_identity["synthetic_raw_response_byte_count"] = len(forged_raw)
    with pytest.raises(P1PreflightRenderingError, match="cluster assignment differs"):
        _render_verified_svg(forged_source_identity, forged_raw, scientific, locator)

    wrong_assignment = copy.deepcopy(original)
    wrong_assignment["cluster_assignment_sha256"] = "a" * 64
    with pytest.raises(P1PreflightRenderingError, match="cluster assignment differs"):
        _render_verified_svg(wrong_assignment, raw, scientific, locator)

    wrong_registry = copy.deepcopy(original)
    wrong_registry["ordered_cluster_registry_sha256"] = "b" * 64
    with pytest.raises(P1PreflightRenderingError, match="score registry differs"):
        _render_verified_svg(wrong_registry, raw, scientific, locator)

    wrong_review = copy.deepcopy(original)
    review = cast(dict[str, object], wrong_review["review"])
    review["B0_hit_clusters"] = cast(int, review["B0_hit_clusters"]) + 1
    with pytest.raises(P1PreflightRenderingError, match="cluster_10 review differs"):
        _render_verified_svg(wrong_review, raw, scientific, locator)

    created_at = datetime.fromisoformat(
        cast(str, original["replay_created_at_utc"]).replace("Z", "+00:00")
    )
    recomputed = recompute_mature_truth_snapshot(
        raw,
        scientific,
        horizon_days=30,
        truth_fetched_at_utc=created_at,
    )
    forged_reviews = build_pending_sequential_reviews(recomputed.scores, elapsed_months=3.0)
    assert len(forged_reviews) == 1
    coordinated_elapsed_forgery = copy.deepcopy(original)
    coordinated_elapsed_forgery["review"] = forged_reviews[0].as_mapping()
    with pytest.raises(P1PreflightRenderingError, match="cluster_10 review differs"):
        _render_verified_svg(coordinated_elapsed_forgery, raw, scientific, locator)
