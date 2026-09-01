from __future__ import annotations

import copy
import inspect
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from seismoflux.multitask_s1.development_contract import (
    DEVELOPMENT_FOLD_IDS,
    HORIZONS_DAYS,
    load_development_contract,
)
from seismoflux.multitask_s1.runner_inputs import (
    CATALOG_HISTORY_START_UTC,
    EXPECTED_25KM_CELL_COUNT,
    EXPECTED_25KM_GRID_ID,
    EXPECTED_STUDY_AREA_SHA256,
    EXPECTED_TOTAL_AREA_KM2,
    HISTORICAL_M5_START_UTC,
    RunnerInputError,
    adapt_frozen_spatial_grid,
    build_inner_development_exposures,
    catalog_event_table_from_frame,
    causal_catalog_histories,
    load_outer_development_issues,
    load_s1_runner_inputs,
    load_verified_catalog_inputs,
    load_verified_spatial_inputs,
)
from seismoflux.stage2s.contracts import SpatialGrid

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "configs/multitask_s1_development_contract.yaml"
ISSUE_LEDGER_PATH = ROOT / "outputs/multitask_s0/s0_score_blind_20260901/issue_maturity_ledger.csv"
SHANGHAI = timezone(timedelta(hours=8))


def _local_data_root() -> Path | None:
    candidates = [ROOT / "data"]
    if len(ROOT.parents) >= 3:
        candidates.append(ROOT.parents[2])
    for candidate in candidates:
        if (candidate / "processed/china_mainland.geojson").is_file() and (
            candidate / "processed/stage1/debc98054172a4a1/earthquake_event.parquet"
        ).is_file():
            return candidate.resolve()
    return None


REAL_DATA_ROOT = _local_data_root()


def _contract() -> Mapping[str, Any]:
    contract, _ = load_development_contract(CONTRACT_PATH, project_root=ROOT)
    return contract


def _row(
    event_id: str,
    origin: str,
    magnitude: float,
    *,
    available: str | None = None,
    inside: bool = True,
) -> dict[str, object]:
    return {
        "event_id": event_id,
        "origin_time_utc": origin,
        "available_at": available or origin,
        "longitude": 105.0,
        "latitude": 35.0,
        "magnitude": magnitude,
        "inside_study_area": inside,
    }


def test_runner_requires_explicit_keyword_only_project_and_data_roots() -> None:
    signature = inspect.signature(load_s1_runner_inputs)
    for name in ("project_root", "data_root"):
        parameter = signature.parameters[name]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is inspect.Parameter.empty
    with pytest.raises(TypeError):
        cast(Any, load_s1_runner_inputs)()


def test_catalog_and_study_area_hash_fail_closed_before_parsing(tmp_path: Path) -> None:
    catalog_path = tmp_path / "processed/stage1/debc98054172a4a1/earthquake_event.parquet"
    catalog_path.parent.mkdir(parents=True)
    catalog_path.write_bytes(b"not-the-authoritative-catalog")
    with pytest.raises(RunnerInputError, match="catalog verification failed"):
        load_verified_catalog_inputs(tmp_path)

    study_path = tmp_path / "processed/china_mainland.geojson"
    study_path.parent.mkdir(parents=True, exist_ok=True)
    study_path.write_bytes(b'{"type":"FeatureCollection","features":[]}')
    with pytest.raises(RunnerInputError, match="geojson SHA-256 mismatch"):
        load_verified_spatial_inputs(tmp_path)


def test_non_development_fold_request_is_rejected_before_ledger_access(
    tmp_path: Path,
) -> None:
    with pytest.raises(RunnerInputError, match="exactly the four frozen development folds"):
        load_outer_development_issues(
            tmp_path / "missing.csv",
            {},
            requested_fold_ids=("C_HOLDOUT_2020_2022",),
        )


def test_outer_ledger_exposes_only_four_development_folds_and_primary_axis() -> None:
    rows = load_outer_development_issues(ISSUE_LEDGER_PATH, _contract())
    assert len(rows) == 5_215
    assert {row.fold_id for row in rows} == set(DEVELOPMENT_FOLD_IDS)
    assert sum(row.primary_exposure_selected for row in rows) == 396
    assert all(
        row.issue_time_utc.astimezone(SHANGHAI).weekday() == 3
        and row.issue_time_utc.astimezone(SHANGHAI).time() == datetime.min.time()
        for row in rows
    )


def test_inner_thursday_greedy_exposures_match_all_12_by_5_waterlevels() -> None:
    contract = _contract()
    exposures = build_inner_development_exposures(contract)
    assert len(exposures) == 12 * len(HORIZONS_DAYS)
    expected = {
        (str(row["fold"]), str(row["block"]), horizon): int(row["exposures"][index])
        for row in cast(list[dict[str, Any]], contract["inner_block_waterlevels"])
        for index, horizon in enumerate(HORIZONS_DAYS)
    }
    observed = {
        (item.fold_id, item.block_id, item.horizon_days): item.exposure_count for item in exposures
    }
    assert observed == expected
    for item in exposures:
        assert all(
            issue.astimezone(SHANGHAI).weekday() == 3
            and issue.astimezone(SHANGHAI).time() == datetime.min.time()
            for issue in item.issue_times_utc
        )
        assert all(
            later >= earlier + timedelta(days=item.horizon_days + 30)
            for earlier, later in pairwise(item.issue_times_utc)
        )

    changed = copy.deepcopy(dict(contract))
    changed_rows = cast(list[dict[str, Any]], changed["inner_block_waterlevels"])
    changed_rows[0]["exposures"][0] += 1
    with pytest.raises(RunnerInputError, match="frozen water level"):
        build_inner_development_exposures(changed)


def test_causal_histories_use_beijing_1970_start_and_exact_t_minus_24h() -> None:
    issue = datetime(2000, 1, 10, tzinfo=UTC)
    cutoff = issue - timedelta(hours=24)
    frame = pd.DataFrame(
        [
            _row("before_beijing_1900", "1899-12-31T15:59:59Z", 5.3),
            _row("at_beijing_1900", "1899-12-31T16:00:00Z", 5.2),
            _row("before_beijing_1970", "1969-12-31T15:59:59Z", 4.5),
            _row("at_beijing_1970", "1969-12-31T16:00:00Z", 4.0),
            _row("m5", "2000-01-01T00:00:00Z", 5.0),
            _row("m6", "2000-01-02T00:00:00Z", 6.0),
            _row(
                "at_cutoff",
                "2000-01-08T00:00:00Z",
                5.9,
                available=cutoff.isoformat(),
            ),
            _row(
                "late_availability",
                "2000-01-08T00:00:00Z",
                5.5,
                available=(cutoff + timedelta(microseconds=1)).isoformat(),
            ),
            _row(
                "after_cutoff",
                (cutoff + timedelta(microseconds=1)).isoformat(),
                6.5,
            ),
            _row("outside", "2000-01-03T00:00:00Z", 6.2, inside=False),
            _row("below_m4", "2000-01-04T00:00:00Z", 3.9),
        ]
    )
    table = catalog_event_table_from_frame(frame)
    histories = causal_catalog_histories(table, issue)

    assert datetime(1969, 12, 31, 16, tzinfo=UTC) == CATALOG_HISTORY_START_UTC
    assert histories["m4_plus"].data_cutoff_utc == cutoff
    assert histories["m4_plus"].event_ids == (
        "at_beijing_1970",
        "m5",
        "m6",
        "at_cutoff",
    )
    assert histories["m5_6"].event_ids == ("m5", "at_cutoff")
    assert histories["m6_plus"].event_ids == ("m6",)
    assert histories["m5_plus_1970_for_joint"].event_ids == (
        "m5",
        "m6",
        "at_cutoff",
    )
    assert datetime(1899, 12, 31, 16, tzinfo=UTC) == HISTORICAL_M5_START_UTC
    assert histories["m5_plus_1900_for_m3"].event_ids == (
        "at_beijing_1900",
        "m5",
        "m6",
        "at_cutoff",
    )
    assert "at_beijing_1900" not in histories["m4_plus"].event_ids
    assert all(
        int(value) <= int(cutoff.timestamp() * 1_000_000)
        for value in histories["m4_plus"].available_at_us
    )


def test_frozen_spatial_adapter_preserves_coordinates_and_exact_areas() -> None:
    grid = SpatialGrid(
        grid_id="synthetic-25km-grid",
        cell_size_km=25.0,
        cell_ids=("a", "b"),
        rows=np.asarray([0, 0], dtype=np.int64),
        columns=np.asarray([0, 1], dtype=np.int64),
        query_xy_km=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        clipped_area_km2=np.asarray([500.0, 625.0], dtype=np.float64),
    )
    adapted = adapt_frozen_spatial_grid(grid)
    assert adapted.cell_count == 2
    assert adapted.total_area_km2 == 1_125.0
    assert adapted.x_km.tolist() == [1.0, 3.0]
    assert adapted.y_km.tolist() == [2.0, 4.0]


@pytest.mark.skipif(REAL_DATA_ROOT is None, reason="authoritative local data root unavailable")
def test_real_authoritative_catalog_and_rebuilt_grid_match_frozen_identities() -> None:
    assert REAL_DATA_ROOT is not None
    catalog, identity = load_verified_catalog_inputs(REAL_DATA_ROOT)
    domain, frozen, study_hash = load_verified_spatial_inputs(REAL_DATA_ROOT)
    assert catalog.row_count == 40_898
    assert identity["file_sha256"] == (
        "2193514eec2889dbf4ae9598c5d45ef8961a8f3fcd26c7183b233dbe20842347"
    )
    assert study_hash == EXPECTED_STUDY_AREA_SHA256
    assert domain.stage3_grid.grid_id == EXPECTED_25KM_GRID_ID
    assert domain.operational_grid.grid_id == EXPECTED_25KM_GRID_ID
    assert frozen.cell_count == EXPECTED_25KM_CELL_COUNT
    assert frozen.total_area_km2 == pytest.approx(EXPECTED_TOTAL_AREA_KM2, abs=1.0e-9)
