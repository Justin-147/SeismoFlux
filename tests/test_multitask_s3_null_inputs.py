"""Synthetic narrow Parquet reads only, including unread poisoned future rows."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from seismoflux.multitask_s3.null_inputs import RADIUS_BASE_COLUMNS, load_radius_bases

T1 = datetime(2023, 1, 1, tzinfo=UTC)
T2 = datetime(2023, 2, 1, tzinfo=UTC)
END = datetime(2024, 6, 30, 16, tzinfo=UTC)


def _table(
    time: datetime,
    *,
    cells: tuple[str, ...] = ("b", "a"),
    grid: str = "grid",
    values: tuple[float, ...] = (1.0, 3.0),
) -> pa.Table:
    return pa.table(
        {
            "nested_unselected": [{"x": 1, "y": 2}] * len(cells),
            "issue_time_utc": pa.array([time] * len(cells), type=pa.timestamp("ns", tz="UTC")),
            "grid_id": [grid] * len(cells),
            "cell_id": list(cells),
            RADIUS_BASE_COLUMNS[0]: pa.array(values, type=pa.float64()),
            RADIUS_BASE_COLUMNS[1]: pa.array(values, type=pa.float64()),
            "gaussian_200km__source_new_count": [-999.0] * len(cells),
            "human_future_outcome": ["do not read"] * len(cells),
        }
    )


def _write(path: Path, tables: list[pa.Table]) -> Path:
    with pq.ParquetWriter(path, tables[0].schema) as writer:
        for table in tables:
            writer.write_table(table)
    return path


def _load(path: Path, **kwargs: Any) -> dict[datetime, Any]:
    arguments: dict[str, Any] = {
        "issue_times_utc": [T1],
        "expected_cell_ids": ["a", "b"],
        "expected_grid_id": "grid",
        "report_end_exclusive": END,
    }
    arguments.update(kwargs)
    return load_radius_bases(path, **arguments)


def test_exact_requested_groups_and_five_columns_only_future_poison_unread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write(
        tmp_path / "source.parquet",
        [
            _table(T1),
            _table(T2),
            _table(END, values=(-1.0, -2.0)),
        ],
    )
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
            return getattr(self.inner, name)

        def read_row_group(self, index: int, *, columns: list[str], use_threads: bool) -> pa.Table:
            calls.append((index, columns, use_threads))
            return self.inner.read_row_group(index, columns=columns, use_threads=use_threads)

    monkeypatch.setattr(pq, "ParquetFile", Spy)
    result = _load(path, issue_times_utc=[T2, T1])
    assert tuple(result) == (T2, T1)
    selected = ["issue_time_utc", "grid_id", "cell_id", *RADIUS_BASE_COLUMNS]
    assert calls == [(1, selected, False), (0, selected, False)]
    np.testing.assert_array_equal(result[T1], [[3.0, 3.0], [1.0, 1.0]])
    assert not result[T1].flags.writeable


def test_nan_is_preserved_not_imputed(tmp_path: Path) -> None:
    path = _write(tmp_path / "nan.parquet", [_table(T1, values=(np.nan, 2.0))])
    result = _load(path)[T1]
    assert np.isnan(result[1]).all()
    np.testing.assert_array_equal(result[0], [2.0, 2.0])


@pytest.mark.parametrize(
    "table",
    [
        _table(T1, cells=("a", "a")),
        _table(T1, cells=("a", "c")),
        _table(T1, cells=("a",), values=(1.0,)),
        _table(T1, grid="wrong"),
        _table(T1, values=(-1.0, 2.0)),
        _table(T1, values=(np.inf, 2.0)),
    ],
)
def test_invalid_base_values_and_identity_fail(tmp_path: Path, table: pa.Table) -> None:
    with pytest.raises(ValueError):
        _load(_write(tmp_path / "bad.parquet", [table]))


def test_cutoff_and_exact_time_requests_fail_before_reading_values(tmp_path: Path) -> None:
    path = _write(tmp_path / "dates.parquet", [_table(T1), _table(END)])
    for times in (
        [END],
        [T1, T1],
        [T1.replace(tzinfo=None)],
        [pd.Timestamp(T1) + pd.Timedelta(nanoseconds=1)],
    ):
        with pytest.raises(ValueError):
            _load(path, issue_times_utc=times)
    with pytest.raises(ValueError):
        _load(path, report_end_exclusive=datetime(2026, 1, 1, tzinfo=UTC))
    with pytest.raises(ValueError):
        _load(path, issue_times_utc=[T1 - timedelta(days=700)])


def test_missing_column_duplicate_report_and_naive_cutoff_fail(tmp_path: Path) -> None:
    table = _table(T1).drop([RADIUS_BASE_COLUMNS[0]])
    with pytest.raises(ValueError, match="column"):
        _load(_write(tmp_path / "missing.parquet", [table]))
    with pytest.raises(ValueError, match="duplicate"):
        _load(_write(tmp_path / "duplicate.parquet", [_table(T1), _table(T1)]))
    with pytest.raises(ValueError, match="timezone"):
        _load(tmp_path / "not_opened.parquet", report_end_exclusive=END.replace(tzinfo=None))
