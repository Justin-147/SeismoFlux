from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from seismoflux.d1_replay.features import (
    D1_SOURCE_COLUMNS,
    D1FeatureContract,
    D1GroupPreprocessor,
    D1IssueFeatures,
    D1StaticGrid,
    load_d1_feature_contract,
    stream_stage3_issue_features,
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _grid(cell_count: int = 4) -> D1StaticGrid:
    return D1StaticGrid(
        grid_id="d1-grid",
        cell_ids=tuple(f"c{index}" for index in range(cell_count)),
        rows=np.zeros(cell_count, dtype=np.int64),
        columns=np.arange(cell_count, dtype=np.int64),
        query_x_m=np.arange(cell_count, dtype=np.float64) * 25_000.0,
        query_y_m=np.zeros(cell_count, dtype=np.float64),
        clipped_area_km2=np.full(cell_count, 625.0, dtype=np.float64),
    )


def _issue(
    issue_time: datetime,
    *,
    values: np.ndarray | None = None,
    nulls: np.ndarray | None = None,
) -> D1IssueFeatures:
    cell_count = 4
    if values is None:
        row = np.arange(cell_count, dtype=np.float64)[:, None]
        column = np.arange(len(D1_SOURCE_COLUMNS), dtype=np.float64)[None, :]
        values = row + column + 1.0
    if nulls is None:
        nulls = np.zeros((cell_count, len(D1_SOURCE_COLUMNS)), dtype=np.bool_)
    return D1IssueFeatures(
        issue_time_utc=issue_time,
        issue_report_id=f"r-{issue_time.date().isoformat()}",
        grid=_grid(),
        source_columns=D1_SOURCE_COLUMNS,
        values=np.asarray(values, dtype=np.float64),
        null_mask=np.asarray(nulls, dtype=np.bool_),
    )


def _stage3_table(
    issue_times: tuple[datetime, ...],
    *,
    tamper_second_grid: bool = False,
) -> pa.Table:
    cell_count = 3
    total = len(issue_times) * cell_count
    rows = np.tile(np.zeros(cell_count, dtype=np.int32), len(issue_times))
    if tamper_second_grid:
        rows[cell_count] = 1
    data: dict[str, object] = {
        "issue_time_utc": pa.array(
            [value for value in issue_times for _ in range(cell_count)],
            type=pa.timestamp("us", tz="UTC"),
        ),
        "issue_report_id": [
            f"report-{issue_index}"
            for issue_index in range(len(issue_times))
            for _ in range(cell_count)
        ],
        "grid_id": ["d1-grid"] * total,
        "cell_id": [f"c{cell}" for _ in issue_times for cell in range(cell_count)],
        "cell_row": rows,
        "cell_column": np.tile(np.arange(cell_count, dtype=np.int32), len(issue_times)),
        "query_x_m": np.tile(np.arange(cell_count, dtype=np.float64) * 25_000.0, len(issue_times)),
        "query_y_m": np.zeros(total, dtype=np.float64),
        "clipped_area_km2": np.full(total, 625.0, dtype=np.float64),
        "unused_score_like_column": ["must-not-be-read"] * total,
    }
    for source_index, name in enumerate(D1_SOURCE_COLUMNS):
        values = np.arange(total, dtype=np.float64) + source_index
        if source_index == 0:
            data[name] = pa.array(
                [None if index == 1 else float(value) for index, value in enumerate(values)],
                type=pa.float64(),
            )
        else:
            data[name] = values
    return pa.table(data)


def test_stage3_reader_streams_one_issue_row_group_and_preserves_raw_nulls(
    tmp_path: Path,
) -> None:
    issue_times = (
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 2, 1, tzinfo=UTC),
    )
    path = tmp_path / "feature_store.parquet"
    pq.write_table(_stage3_table(issue_times), path, row_group_size=3)

    issues = tuple(
        stream_stage3_issue_features(
            path,
            expected_issue_count=2,
            expected_cell_count=3,
        )
    )

    assert tuple(item.issue_time_utc for item in issues) == issue_times
    assert issues[0].grid is issues[1].grid
    assert issues[0].values.dtype == np.float64
    assert issues[0].null_mask.dtype == np.bool_
    assert issues[0].values.shape == (3, 15)
    assert issues[0].null_mask[1, 0]
    assert np.isnan(issues[0].values[1, 0])
    assert not issues[0].values.flags.writeable
    assert not issues[0].null_mask.flags.writeable


def test_stage3_reader_fails_if_a_later_issue_changes_the_first_static_grid(
    tmp_path: Path,
) -> None:
    issue_times = (
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 2, 1, tzinfo=UTC),
    )
    path = tmp_path / "tampered.parquet"
    pq.write_table(
        _stage3_table(issue_times, tamper_second_grid=True),
        path,
        row_group_size=3,
    )
    with pytest.raises(ValueError, match="static D1 grid column rows"):
        tuple(
            stream_stage3_issue_features(
                path,
                expected_issue_count=2,
                expected_cell_count=3,
            )
        )


def test_group_preprocessor_uses_contract_order_direction_and_raw_null_controls() -> None:
    contract = load_d1_feature_contract(
        _repository_root() / "configs" / "d1_retrospective_development.yaml"
    )
    values = _issue(datetime(2024, 1, 1, tzinfo=UTC)).values.copy()
    nulls = np.zeros(values.shape, dtype=np.bool_)
    nulls[0, 0] = True
    nulls[1, 7] = True
    nulls[2, 11] = True
    training = _issue(
        datetime(2024, 1, 1, tzinfo=UTC),
        values=values,
        nulls=nulls,
    )
    preprocessor = D1GroupPreprocessor.fit(
        contract,
        ("D1", "S5", "C1"),
        (training,),
    )
    design = preprocessor.transform(training)

    assert design.column_names == ("C1", "S5", "D1", "MC1", "MS45", "MD1")
    np.testing.assert_array_equal(design.values[:, 3], [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(design.values[:, 4], [0.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(design.values[:, 5], [0.0, 0.0, 1.0 / 3.0, 0.0])

    source_indices = [9, 10, 11]
    signs = [1.0, 1.0, -1.0]
    components = []
    for source_index, sign in zip(source_indices, signs, strict=True):
        source_name = D1_SOURCE_COLUMNS[source_index]
        fit = preprocessor.source_fits[source_name]
        valid = ~training.null_mask[:, source_index]
        transformed = np.empty(4, dtype=np.float64)
        transformed[~valid] = fit.center
        transformed[valid] = np.clip(
            np.arcsinh(training.values[valid, source_index]),
            fit.clip_low,
            fit.clip_high,
        )
        components.append(sign * (transformed - fit.center) / fit.scale)
    raw_d1 = np.mean(np.column_stack(components), axis=1)
    d1_fit = preprocessor.group_fits["D1"]
    expected_d1 = (
        (raw_d1 - d1_fit.center) / d1_fit.scale if d1_fit.active else np.zeros(4, dtype=np.float64)
    )
    np.testing.assert_allclose(design.values[:, 2], expected_d1, rtol=0.0, atol=1e-15)


def test_future_rows_cannot_change_frozen_training_statistics() -> None:
    contract = load_d1_feature_contract(
        _repository_root() / "configs" / "d1_retrospective_development.yaml"
    )
    start = datetime(2024, 1, 1, tzinfo=UTC)
    training = (_issue(start), _issue(start + timedelta(days=30)))
    preprocessor = D1GroupPreprocessor.fit(contract, ("C1", "C2"), training)
    before_sources = dict(preprocessor.source_fits)
    before_groups = dict(preprocessor.group_fits)
    before_training_design = preprocessor.transform(training[0]).values.copy()

    future_values = np.full((4, len(D1_SOURCE_COLUMNS)), 1.0e9, dtype=np.float64)
    future = _issue(start + timedelta(days=60), values=future_values)
    preprocessor.transform(future)

    assert dict(preprocessor.source_fits) == before_sources
    assert dict(preprocessor.group_fits) == before_groups
    assert preprocessor.fitted_issue_times_utc == tuple(item.issue_time_utc for item in training)
    np.testing.assert_array_equal(
        preprocessor.transform(training[0]).values,
        before_training_design,
    )


def test_constant_group_is_emitted_as_fixed_zero() -> None:
    contract = load_d1_feature_contract(
        _repository_root() / "configs" / "d1_retrospective_development.yaml"
    )
    values = np.ones((4, len(D1_SOURCE_COLUMNS)), dtype=np.float64)
    issue = _issue(datetime(2024, 1, 1, tzinfo=UTC), values=values)
    preprocessor = D1GroupPreprocessor.fit(contract, ("S1",), (issue,))
    design = preprocessor.transform(issue)
    assert design.column_names == ("S1",)
    assert not design.active_coefficients[0]
    np.testing.assert_array_equal(design.values[:, 0], np.zeros(4))


def test_contract_rejects_any_reordering_of_the_fifteen_sources() -> None:
    path = _repository_root() / "configs" / "d1_retrospective_development.yaml"
    contract = load_d1_feature_contract(path)
    assert isinstance(contract, D1FeatureContract)
    assert (
        tuple(source.column for group in contract.groups for source in group.sources)
        == D1_SOURCE_COLUMNS
    )
