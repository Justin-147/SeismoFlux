from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta, timezone

import pyarrow as pa
import pytest
from shapely.geometry import box

from seismoflux.features.anomaly.dictionary import DEFAULT_FEATURE_DICTIONARY
from seismoflux.features.anomaly.engine import Stage3FeatureEngine
from seismoflux.features.anomaly.grid import build_stage3_query_grid
from seismoflux.features.anomaly.nulls import NullReasonCode
from seismoflux.features.anomaly.snapshot import build_issue_snapshots
from seismoflux.features.anomaly.state import build_anomaly_state_history

LOCAL = timezone(timedelta(hours=8))


def _period(day: date, number: int) -> dict[str, object]:
    issue = datetime.combine(day + timedelta(days=1), time.min, LOCAL).astimezone(UTC)
    return {
        "report_id": f"report-{number}",
        "source_file": f"anomaly/report-{number}.xls",
        "report_year": day.year,
        "report_period": number,
        "report_date": day,
        "available_at": issue,
        "row_count": 1,
        "row_report_date_mismatch_count": 0,
        "row_report_date_before_count": 0,
        "row_report_date_after_count": 0,
        "deformation_row_count": 1,
        "fluid_row_count": 0,
        "electromagnetic_row_count": 0,
        "cross_fault_row_count": 0,
    }


def _observation(period: dict[str, object], number: int) -> dict[str, object]:
    report_date = period["report_date"]
    assert isinstance(report_date, date)
    assert not isinstance(report_date, datetime)
    return {
        "observation_id": f"obs-{number}",
        "anomaly_id": "entity-a",
        "identity_complete": True,
        "report_date": report_date,
        "source_file": period["source_file"],
        "available_at": period["available_at"],
        "station_id": "station-a",
        "longitude": 110.0,
        "latitude": 35.0,
        "discipline": "形变",
        "measurement": "measurement-a",
        "start_time": datetime.combine(report_date - timedelta(days=30), time.min, LOCAL),
        "is_listed": True,
        "report_state": "持续",
        "reported_end_time": None,
        "right_censored": True,
        "reliability_flags": (),
    }


def test_feature_engine_builds_causal_wide_tables_in_issue_order() -> None:
    periods = [_period(date(2024, 1, 1) + timedelta(weeks=index), index + 1) for index in range(4)]
    observations = [_observation(period, index + 1) for index, period in enumerate(periods)]
    states = build_anomaly_state_history(observations, periods)
    snapshots = build_issue_snapshots(states, expected_issue_count=4)
    grid = build_stage3_query_grid(box(109.9, 34.9, 110.1, 35.1))
    engine = Stage3FeatureEngine(snapshots, grid)

    first = engine.build_next_issue()
    second = engine.build_next_issue()
    third = engine.build_next_issue()
    fourth = engine.build_next_issue()

    assert first.table.num_rows == grid.cell_count
    assert fourth.table.schema == engine.schema
    assert engine.next_issue_index == 4
    assert first.table["feature_dictionary_sha256"].unique().to_pylist() == [
        DEFAULT_FEATURE_DICTIONARY.sha256
    ]
    slope_name = "radius_500km__listed_count__slope_4w_per_week"
    reason_name = f"{slope_name}__null_reason_code"
    assert first.table[slope_name].null_count == grid.cell_count
    assert first.table[reason_name].unique().to_pylist() == [
        int(NullReasonCode.INSUFFICIENT_ACTUAL_SNAPSHOTS)
    ]
    assert fourth.table[slope_name].null_count == 0
    assert fourth.table[slope_name].to_pylist() == [0.0] * grid.cell_count
    coverage_name = "radius_500km__current_to_trailing_station_reporting_coverage_proxy"
    assert fourth.table[coverage_name].to_pylist() == [1.0] * grid.cell_count
    assert all(
        table.table["issue_index"].unique().to_pylist() == [index]
        for index, table in enumerate((first, second, third, fourth))
    )
    assert isinstance(fourth.table[slope_name], pa.ChunkedArray)


def test_two_spatial_workers_are_byte_for_byte_deterministic() -> None:
    periods = [_period(date(2024, 1, 1) + timedelta(weeks=index), index + 1) for index in range(4)]
    observations = [_observation(period, index + 1) for index, period in enumerate(periods)]
    snapshots = build_issue_snapshots(
        build_anomaly_state_history(observations, periods),
        expected_issue_count=4,
    )
    grid = build_stage3_query_grid(box(109.9, 34.9, 110.1, 35.1))
    sequential = Stage3FeatureEngine(snapshots, grid, spatial_workers=1)
    parallel = Stage3FeatureEngine(snapshots, grid, spatial_workers=2)

    for _ in snapshots:
        assert sequential.build_next_issue().table.equals(
            parallel.build_next_issue().table,
            check_metadata=True,
        )


@pytest.mark.parametrize("workers", [0, 3, True])
def test_spatial_worker_count_is_bounded(workers: object) -> None:
    periods = [_period(date(2024, 1, 1), 1)]
    observations = [_observation(periods[0], 1)]
    snapshots = build_issue_snapshots(
        build_anomaly_state_history(observations, periods),
        expected_issue_count=1,
    )
    grid = build_stage3_query_grid(box(109.9, 34.9, 110.1, 35.1))

    with pytest.raises(ValueError, match="one or two"):
        Stage3FeatureEngine(snapshots, grid, spatial_workers=workers)  # type: ignore[arg-type]
