"""Synthetic bounded state/mapping reads; never use original anomaly sources."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from seismoflux.features.anomaly.contracts import ANOMALY_STATE_HISTORY_CONTRACT
from seismoflux.features.anomaly.snapshot import Stage3IssueSnapshot
from seismoflux.features.anomaly.state import build_anomaly_state_history
from seismoflux.multitask_s3.null_state_inputs import (
    ENTITY_MAPPING_COLUMNS,
    STATE_COLUMNS,
    load_all_zone_ids,
    load_construction_strata,
    load_issue_snapshots,
)

LOCAL = timezone(timedelta(hours=8))
T1 = datetime(2023, 1, 1, 16, tzinfo=UTC)
T2 = T1 + timedelta(days=7)
END = datetime(2024, 6, 30, 16, tzinfo=UTC)


def _state_table(issue: datetime, *, missing: bool = False, poison: bool = False) -> pa.Table:
    day = (issue.astimezone(LOCAL) - timedelta(days=1)).date()
    period: dict[str, object] = {
        "report_id": f"report-{day}",
        "source_file": f"anomaly/{day}.xls",
        "report_year": day.year,
        "report_period": 1,
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
    observation: dict[str, object] = {
        "observation_id": f"obs-{day}",
        "anomaly_id": "entity-a",
        "identity_complete": True,
        "report_date": day,
        "source_file": period["source_file"],
        "available_at": issue,
        "station_id": "station-a",
        "longitude": 110.0,
        "latitude": 35.0,
        "discipline": "形变",
        "measurement": "measurement-a",
        "start_time": datetime.combine(day - timedelta(days=30), time.min, LOCAL),
        "is_listed": True,
        "report_state": "持续",
        "reported_end_time": None,
        "right_censored": True,
        "reliability_flags": (),
    }
    states = list(build_anomaly_state_history([observation], [period]))
    for index, state in enumerate(states):
        if state.state_row_kind == "entity_state" and missing:
            states[index] = replace(
                state,
                longitude=float("nan"),
                latitude=None,
                spatial_eligible=False,
                spatial_exclusion_reason="missing_coordinate",
                age_days=float("nan"),
            )
    records = [state.to_record() for state in states]
    if poison:
        records[0]["state_row_kind"] = "unread_future_poison"
    return pa.Table.from_pylist(records, schema=ANOMALY_STATE_HISTORY_CONTRACT.schema)


def _write(path: Path, tables: list[pa.Table], **kwargs: Any) -> Path:
    with pq.ParquetWriter(path, tables[0].schema, **kwargs) as writer:
        for table in tables:
            writer.write_table(table)
    return path


def _load(path: Path, **kwargs: Any) -> dict[datetime, Stage3IssueSnapshot]:
    arguments: dict[str, Any] = {"issue_times_utc": [T1], "report_end_exclusive": END}
    arguments.update(kwargs)
    return load_issue_snapshots(path, **arguments)


def test_only_requested_complete_groups_read_multiple_groups_per_issue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _state_table(T1)
    # Move a nested list column before the timestamp: footer leaf indices differ
    # from Arrow field indices, which the bounded loader must handle correctly.
    columns = [
        "current_reporting_station_ids",
        *[name for name in STATE_COLUMNS if name != "current_reporting_station_ids"],
    ]
    tables = [first.slice(0, 1), _state_table(T2), first.slice(1), _state_table(END, poison=True)]
    path = _write(tmp_path / "states.parquet", [table.select(columns) for table in tables])
    real = pq.ParquetFile
    calls = []

    class Spy:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.inner = real(*args, **kwargs)

        def __enter__(self) -> Spy:
            return self

        def __exit__(self, *args: Any) -> None:
            self.inner.close()

        def __getattr__(self, name: str) -> Any:
            if name in {"read", "read_row_groups", "iter_batches"}:
                raise AssertionError("unbounded state value read")
            return getattr(self.inner, name)

        def read_row_group(self, index: int, **kwargs: Any) -> pa.Table:
            calls.append((index, kwargs))
            return self.inner.read_row_group(index, **kwargs)

    monkeypatch.setattr(pq, "ParquetFile", Spy)
    result = _load(path)
    assert list(result) == [T1]
    assert len(result[T1].entities) == 1
    assert calls == [
        (index, {"columns": list(STATE_COLUMNS), "use_threads": False}) for index in (0, 2)
    ]


def test_nan_and_nonspatial_states_are_preserved(tmp_path: Path) -> None:
    result = _load(_write(tmp_path / "missing.parquet", [_state_table(T1, missing=True)]))
    assert len(result[T1].entities) == 1
    entity = result[T1].entities[0]
    assert not entity.spatial_eligible
    assert entity.latitude is None and np.isnan(entity.longitude)
    assert np.isnan(entity.age_days)


def test_requested_order_and_chronological_snapshot_indices(tmp_path: Path) -> None:
    path = _write(tmp_path / "ordered.parquet", [_state_table(T1), _state_table(T2)])
    result = _load(path, issue_times_utc=[T2, T1])
    assert tuple(result) == (T2, T1)
    assert result[T1].issue_index == 0 and result[T2].issue_index == 1


@pytest.mark.parametrize("times", [[END], [T1, T1], [], [T1.replace(tzinfo=None)]])
def test_unauthorized_requests_fail_before_open(tmp_path: Path, times: list[datetime]) -> None:
    with pytest.raises(ValueError):
        _load(tmp_path / "not_opened.parquet", issue_times_utc=times)


def test_out_of_range_cutoff_and_missing_requested_issue(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cutoff"):
        _load(
            tmp_path / "not_opened.parquet", report_end_exclusive=datetime(2026, 1, 1, tzinfo=UTC)
        )
    path = _write(tmp_path / "missing_issue.parquet", [_state_table(T2)])
    with pytest.raises(ValueError, match="omitted"):
        _load(path)


@pytest.mark.parametrize("kind", ["duplicate", "mixed_dates", "no_statistics", "missing_schema"])
def test_malformed_state_sources_fail(tmp_path: Path, kind: str) -> None:
    table = _state_table(T1)
    if kind == "duplicate":
        tables = [table, table]
    elif kind == "mixed_dates":
        tables = [pa.concat_tables([table, _state_table(T2)])]
    elif kind == "missing_schema":
        tables = [table.drop(["age_days"])]
    else:
        tables = [table]
    path = _write(tmp_path / "bad.parquet", tables, write_statistics=kind != "no_statistics")
    with pytest.raises(ValueError):
        _load(path)


def _mapping_table(rows: list[dict[str, object]]) -> pa.Table:
    schema = pa.schema(
        [
            ("state_id", pa.string()),
            ("issue_time_utc", pa.timestamp("us", tz="UTC")),
            ("construction_stratum_id", pa.string()),
            ("future_outcome_do_not_read", pa.string()),
        ]
    )
    return pa.Table.from_pylist(rows, schema=schema)


def _mapping_row(snapshot: Stage3IssueSnapshot, **changes: object) -> dict[str, object]:
    row: dict[str, object] = {
        "state_id": snapshot.entities[0].state_id,
        "issue_time_utc": snapshot.issue_time_utc,
        "construction_stratum_id": "zone-a:inside",
        "future_outcome_do_not_read": "poison",
    }
    row.update(changes)
    return row


def test_mapping_has_arrow_date_predicate_and_exact_three_columns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshots = _load(_write(tmp_path / "s.parquet", [_state_table(T1)]))
    good = _mapping_row(snapshots[T1])
    path = _write(
        tmp_path / "map.parquet",
        [
            _mapping_table(
                [
                    good,
                    {**good, "issue_time_utc": END, "construction_stratum_id": None},
                    {**good, "issue_time_utc": T2, "construction_stratum_id": None},
                ]
            )
        ],
    )
    real = pq.read_table
    calls = []

    def spy(*args: Any, **kwargs: Any) -> pa.Table:
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(pq, "read_table", spy)
    result = load_construction_strata(path, snapshots_by_issue=snapshots, report_end_exclusive=END)
    assert result == {good["state_id"]: "zone-a:inside"}
    assert calls[0]["columns"] == list(ENTITY_MAPPING_COLUMNS)
    assert calls[0]["use_threads"] is False
    assert ("issue_time_utc", "in", [T1]) in calls[0]["filters"]
    assert ("issue_time_utc", "<", END) in calls[0]["filters"]


@pytest.mark.parametrize("kind", ["duplicate", "unknown_id", "wrong_date", "missing", "stratum"])
def test_mapping_state_and_date_mismatches_fail(tmp_path: Path, kind: str) -> None:
    snapshots = _load(
        _write(tmp_path / "s.parquet", [_state_table(T1), _state_table(T2)]),
        issue_times_utc=[T1, T2],
    )
    rows = [_mapping_row(snapshot) for snapshot in snapshots.values()]
    if kind == "duplicate":
        rows.append(rows[0])
    elif kind == "unknown_id":
        rows[0]["state_id"] = "not-in-snapshot"
    elif kind == "wrong_date":
        rows[0]["issue_time_utc"] = T2
    elif kind == "missing":
        rows.pop()
    else:
        rows[0]["construction_stratum_id"] = " "
    path = _write(tmp_path / "m.parquet", [_mapping_table(rows)])
    with pytest.raises(ValueError):
        load_construction_strata(path, snapshots_by_issue=snapshots, report_end_exclusive=END)


def test_empty_spatial_population_is_not_deleted_and_needs_no_mapping(tmp_path: Path) -> None:
    snapshots = _load(_write(tmp_path / "s.parquet", [_state_table(T1, missing=True)]))
    path = _write(tmp_path / "empty.parquet", [_mapping_table([])])
    assert (
        load_construction_strata(path, snapshots_by_issue=snapshots, report_end_exclusive=END) == {}
    )
    assert len(snapshots[T1].entities) == 1
    with pytest.raises(ValueError, match="key"):
        load_construction_strata(
            path, snapshots_by_issue={T2: snapshots[T1]}, report_end_exclusive=END
        )


def test_static_mapping_reads_one_column_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write(
        tmp_path / "cells.parquet",
        [
            pa.table(
                {
                    "construction_zone_id": ["zone-b", "zone-a", "zone-b"],
                    "other_do_not_read": [1, 2, 3],
                }
            )
        ],
    )
    real = pq.read_table
    calls = []

    def spy(*args: Any, **kwargs: Any) -> pa.Table:
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(pq, "read_table", spy)
    assert load_all_zone_ids(path) == ("zone-a", "zone-b")
    assert calls == [{"columns": ["construction_zone_id"], "use_threads": False}]


@pytest.mark.parametrize("values", [[], [None], [""]])
def test_invalid_static_zone_ids_fail(tmp_path: Path, values: list[object]) -> None:
    path = _write(
        tmp_path / "bad_zones.parquet",
        [
            pa.table(
                {
                    "construction_zone_id": pa.array(values, type=pa.string()),
                }
            )
        ],
    )
    with pytest.raises(ValueError):
        load_all_zone_ids(path)
