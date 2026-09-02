"""Synthetic Parquet routing and transformation tests; no real data are opened."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from seismoflux.multitask_s3.features import (
    DESIGN_FEATURE_IDS,
    DESIGN_INDICES,
    FULL_FEATURE_IDS,
    IDENTITY_COLUMNS,
    RAW_FEATURE_COLUMNS,
    RAW_FEATURE_SPECS,
    REPORT_END_UTC,
    REPORT_START_UTC,
    S3IssueFeatures,
    build_feature_matrix,
    load_issue_features,
    read_report_issue_metadata,
)

T1 = datetime(2023, 1, 4, 0, 0, tzinfo=UTC)
T2 = datetime(2024, 1, 3, 0, 0, tzinfo=UTC)
FOLD_END = datetime(2024, 6, 30, 16, 0, tzinfo=UTC)


def _table(
    time: datetime,
    *,
    cell_ids: Sequence[str] = ("b", "a"),
    grid_id: str = "synthetic_grid",
    missing_column: str | None = None,
    poison_values: bool = False,
) -> pa.Table:
    size = len(cell_ids)
    values: dict[str, Any] = {
        # This nested field ensures the time column's Parquet leaf index is not
        # simply its top-level Arrow field index.
        "unselected_nested": pa.array([{"x": 1, "y": 2}] * size),
        "issue_time_utc": pa.array([time] * size, type=pa.timestamp("us", tz="UTC")),
        "grid_id": pa.array([grid_id] * size),
        "cell_id": pa.array(cell_ids),
        "unselected_human_outcome": pa.array(["never selected"] * size),
    }
    for name in RAW_FEATURE_COLUMNS:
        if name != missing_column:
            raw = [-100.0 if poison_values else float(2 * index + 1) for index in range(size)]
            values[name] = pa.array(raw, type=pa.float64())
    return pa.table(values)


def _write_store(path: Path, tables: Sequence[pa.Table], *, statistics: bool = True) -> Path:
    with pq.ParquetWriter(path, tables[0].schema, write_statistics=statistics) as writer:
        for table in tables:
            writer.write_table(table)
    return path


def _track_row_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[int, tuple[str, ...], bool]]:
    actual_class = pq.ParquetFile
    calls: list[tuple[int, tuple[str, ...], bool]] = []

    class TrackingParquet:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.inner = actual_class(*args, **kwargs)

        def __enter__(self) -> TrackingParquet:
            return self

        def __exit__(self, *args: Any) -> None:
            self.inner.close()

        def __getattr__(self, name: str) -> Any:
            return getattr(self.inner, name)

        def read_row_group(self, index: int, *, columns: list[str], use_threads: bool) -> pa.Table:
            calls.append((index, tuple(columns), use_threads))
            return self.inner.read_row_group(index, columns=columns, use_threads=use_threads)

    monkeypatch.setattr(pq, "ParquetFile", TrackingParquet)
    return calls


def _load(path: Path, times: Sequence[datetime], **kwargs: Any) -> dict[datetime, S3IssueFeatures]:
    arguments: dict[str, Any] = {
        "issue_times_utc": times,
        "expected_cell_ids": ("a", "b"),
        "expected_grid_id": "synthetic_grid",
        "report_end_exclusive": FOLD_END,
    }
    arguments.update(kwargs)
    return load_issue_features(path, **arguments)


def test_frozen_feature_contract_exactly_matches_accepted_yaml() -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "multitask_s3_anomaly.yaml"
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw_specs = tuple(
        (entry["id"], entry["column"], entry["transform"]) for entry in protocol["features"]["raw"]
    )
    assert raw_specs == RAW_FEATURE_SPECS
    assert FULL_FEATURE_IDS[16:] == tuple(
        entry["id"] for entry in protocol["features"]["missing_controls"]
    )
    for variant, names in protocol["features"]["designs"].items():
        assert DESIGN_FEATURE_IDS[variant] == tuple(names)
    assert [len(DESIGN_INDICES[name]) for name in ("COV", "SNAP", "DYN")] == [5, 16, 20]
    assert set(DESIGN_INDICES["COV"]) < set(DESIGN_INDICES["SNAP"]) < set(DESIGN_INDICES["DYN"])


def test_metadata_only_returns_authorized_times_and_reads_no_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_store(
        tmp_path / "metadata.parquet",
        [
            _table(REPORT_START_UTC - timedelta(seconds=1)),
            _table(T1),
            _table(T2),
            _table(FOLD_END),
            _table(REPORT_END_UTC),
        ],
    )
    calls = _track_row_reads(monkeypatch)
    result = read_report_issue_metadata(path, report_end_exclusive=FOLD_END)
    assert tuple(item.issue_time_utc for item in result) == (T1, T2)
    assert tuple(item.row_group_index for item in result) == (1, 2)
    assert tuple(item.row_count for item in result) == (2, 2)
    assert calls == []


def test_only_requested_row_groups_and_frozen_columns_are_loaded_and_cells_reordered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_store(
        tmp_path / "routing.parquet",
        [
            _table(T1),
            _table(T2, poison_values=True),
            _table(REPORT_END_UTC, poison_values=True),
        ],
    )
    calls = _track_row_reads(monkeypatch)
    result = _load(path, [T1])
    assert calls == [(0, (*IDENTITY_COLUMNS, *RAW_FEATURE_COLUMNS), False)]
    assert tuple(result) == (T1,)
    issue = result[T1]
    assert issue.cell_ids == ("a", "b")
    assert issue.source_row_group_index == 0
    assert issue.values.shape == (2, 20)
    np.testing.assert_allclose(issue.values[:, 0], np.log1p([3.0, 1.0]))
    assert not issue.values.flags.writeable


def test_requested_issue_order_and_timezone_aware_timestamp_are_preserved(tmp_path: Path) -> None:
    path = _write_store(tmp_path / "order.parquet", [_table(T1), _table(T2)])
    result = _load(path, [pd.Timestamp(T2).tz_convert("Asia/Shanghai"), T1])
    assert tuple(result) == (T2, T1)
    with pytest.raises(ValueError, match="exact report issue"):
        _load(path, [pd.Timestamp(T1) + pd.Timedelta(nanoseconds=1)])


def test_global_report_start_is_inclusive_and_fold_end_is_exclusive(tmp_path: Path) -> None:
    path = _write_store(
        tmp_path / "endpoints.parquet", [_table(REPORT_START_UTC), _table(FOLD_END)]
    )
    result = _load(path, [REPORT_START_UTC])
    assert tuple(result) == (REPORT_START_UTC,)
    metadata = read_report_issue_metadata(path, report_end_exclusive=FOLD_END)
    assert tuple(item.issue_time_utc for item in metadata) == (REPORT_START_UTC,)


@pytest.mark.parametrize(
    "times",
    [
        [],
        [T1, T1],
        [REPORT_START_UTC - timedelta(microseconds=1)],
        [FOLD_END],
        [REPORT_END_UTC],
        [datetime(2023, 1, 4)],
    ],
)
def test_empty_duplicate_naive_or_outside_issues_fail_before_opening_source(
    tmp_path: Path,
    times: Sequence[datetime],
) -> None:
    with pytest.raises(ValueError):
        _load(tmp_path / "does_not_exist.parquet", times)


@pytest.mark.parametrize("cutoff", [REPORT_START_UTC, REPORT_END_UTC + timedelta(microseconds=1)])
def test_caller_cannot_expand_the_global_development_interval(
    tmp_path: Path, cutoff: datetime
) -> None:
    with pytest.raises(ValueError, match="cutoff"):
        _load(tmp_path / "does_not_exist.parquet", [T1], report_end_exclusive=cutoff)


@pytest.mark.parametrize("cells", [("a", "a"), ("a",), ("a", "c")])
def test_duplicate_missing_and_wrong_cells_are_rejected(
    tmp_path: Path, cells: tuple[str, ...]
) -> None:
    path = _write_store(tmp_path / "cells.parquet", [_table(T1, cell_ids=cells)])
    with pytest.raises(ValueError, match="cell"):
        _load(path, [T1])


def test_grid_identity_missing_feature_and_missing_requested_issue_are_rejected(
    tmp_path: Path,
) -> None:
    wrong_grid = _write_store(tmp_path / "grid.parquet", [_table(T1, grid_id="another_grid")])
    with pytest.raises(ValueError, match="grid_id"):
        _load(wrong_grid, [T1])
    missing_feature = _write_store(
        tmp_path / "column.parquet",
        [
            _table(T1, missing_column=RAW_FEATURE_COLUMNS[0]),
        ],
    )
    with pytest.raises(ValueError, match="omitted"):
        _load(missing_feature, [T1])
    valid = _write_store(tmp_path / "valid.parquet", [_table(T1)])
    with pytest.raises(ValueError, match="exact report issue"):
        _load(valid, [T2])


def test_duplicate_or_nonindexed_row_groups_are_rejected_without_value_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    duplicate = _write_store(tmp_path / "duplicate.parquet", [_table(T1), _table(T1)])
    missing_stats = _write_store(tmp_path / "stats.parquet", [_table(T1)], statistics=False)
    calls = _track_row_reads(monkeypatch)
    with pytest.raises(ValueError, match="duplicate"):
        _load(duplicate, [T1])
    with pytest.raises(ValueError, match="one non-null exact issue"):
        _load(missing_stats, [T1])
    assert calls == []


def test_transforms_and_four_masks_are_exact_and_do_not_impute_or_standardize() -> None:
    raw = {name: np.array([0.0, 1.0, 2.0]) for name in RAW_FEATURE_COLUMNS}
    raw[RAW_FEATURE_COLUMNS[7]] = np.array([np.nan, 3.0, 0.0])
    raw[RAW_FEATURE_COLUMNS[8]] = np.array([np.nan, -2.0, 1.0])
    raw[RAW_FEATURE_COLUMNS[9]] = np.array([np.nan, 0.0, np.nan])
    raw[RAW_FEATURE_COLUMNS[10]] = np.array([5.0, np.nan, np.nan])
    raw[RAW_FEATURE_COLUMNS[11]] = np.array([np.nan, 0.5, 1.0])
    raw[RAW_FEATURE_COLUMNS[14]] = np.array([1.0, np.nan, np.nan])
    raw[RAW_FEATURE_COLUMNS[15]] = np.array([1.0, 1.0, np.nan])
    matrix = build_feature_matrix(raw)
    assert matrix.shape == (3, 20)
    np.testing.assert_allclose(matrix[:, 0], np.log1p([0.0, 1.0, 2.0]))
    np.testing.assert_allclose(matrix[:, 8], np.arcsinh([np.nan, -2.0, 1.0]), equal_nan=True)
    np.testing.assert_allclose(matrix[:, 11], [np.nan, 0.5, 1.0], equal_nan=True)
    np.testing.assert_allclose(
        matrix[:, 16:],
        [
            [1.0, 2.0 / 3.0, 1.0, 0.0],
            [0.0, 1.0 / 3.0, 0.0, 1.0],
            [0.0, 2.0 / 3.0, 0.0, 1.0],
        ],
    )
    assert np.isnan(matrix[0, 7])
    issue = S3IssueFeatures(T1, "synthetic_grid", ("a", "b", "c"), matrix, 0)
    for variant, indices in DESIGN_INDICES.items():
        np.testing.assert_array_equal(issue.design(variant), matrix[:, indices])
        assert not issue.design(variant).flags.writeable
    raw[RAW_FEATURE_COLUMNS[0]][:] = 99.0
    np.testing.assert_allclose(issue.values[:, 0], np.log1p([0.0, 1.0, 2.0]))
    with pytest.raises(ValueError, match="read-only"):
        matrix[0, 0] = 999


def test_parquet_nulls_stay_nan_and_missing_masks_are_retained(tmp_path: Path) -> None:
    table = _table(T1)
    index = table.schema.get_field_index(RAW_FEATURE_COLUMNS[7])
    table = table.set_column(
        index, RAW_FEATURE_COLUMNS[7], pa.array([None, 3.0], type=pa.float64())
    )
    path = _write_store(tmp_path / "nulls.parquet", [table])
    issue = _load(path, [T1])[T1]
    np.testing.assert_allclose(issue.values[:, 7], [np.log1p(3.0), np.nan], equal_nan=True)
    np.testing.assert_array_equal(issue.values[:, 16], [0.0, 1.0])


def test_unknown_columns_infinity_and_illegal_count_transform_are_rejected() -> None:
    raw = {name: np.ones(2) for name in RAW_FEATURE_COLUMNS}
    with pytest.raises(ValueError, match="exactly the 16"):
        build_feature_matrix({**raw, "human_forecast": np.ones(2)})
    raw[RAW_FEATURE_COLUMNS[0]][0] = np.inf
    with pytest.raises(ValueError, match="finite or NaN"):
        build_feature_matrix(raw)
    raw[RAW_FEATURE_COLUMNS[0]][0] = -1.0
    with pytest.raises(ValueError, match="non-negative"):
        build_feature_matrix(raw)
