"""Verified, target-blind adapters for the accepted stage-3 feature store.

Only anomaly features and local permutation metadata are read here.  The module
contains no earthquake-target path, target parser, score, or locked-test entry
point.  Formal callers must supply the score-blind input receipt already bound
into the stage-4 execution seal.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from seismoflux.anomaly_increment.config import Stage4ProtocolBundle
from seismoflux.anomaly_increment.contracts import FeatureColumnContract, TransformName
from seismoflux.anomaly_increment.grid_features import Stage4IntegrationGrid
from seismoflux.anomaly_increment.qualification import ScoreBlindInputEvidence
from seismoflux.features.anomaly.snapshot import Stage3IssueSnapshot, build_issue_snapshots
from seismoflux.features.anomaly.state import states_from_records

FeatureVariant = Literal["coverage_only", "snapshot", "dynamic"]

FEATURE_IDENTITY_COLUMNS = (
    "issue_time_utc",
    "issue_report_id",
    "grid_id",
    "cell_id",
    "cell_row",
    "cell_column",
    "query_x_m",
    "query_y_m",
    "clipped_area_km2",
)
TIME_PLACEBO_BASE_COLUMNS = (
    "radius_200km__listed_count",
    "radius_200km__first_seen_count",
)


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _strings(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise TypeError(f"{label} must be a sequence")
    output = tuple(value)
    if not all(isinstance(item, str) and item for item in output):
        raise TypeError(f"{label} must contain non-empty strings")
    return cast(tuple[str, ...], output)


def _verified_input_path(
    protocol: Stage4ProtocolBundle,
    score_blind_inputs: ScoreBlindInputEvidence,
    *,
    input_id: str,
) -> Path:
    inputs = _mapping(protocol.protocol.get("inputs"), label="inputs")
    declaration = _mapping(inputs.get(input_id), label=f"inputs.{input_id}")
    relative = declaration.get("path")
    expected = declaration.get("sha256")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"inputs.{input_id}.path must be repository-relative")
    if not isinstance(expected, str):
        raise TypeError(f"inputs.{input_id}.sha256 must be a string")
    observed = dict(score_blind_inputs.observed_project_input_hashes).get(input_id)
    if observed != expected:
        raise ValueError(f"score-blind receipt does not verify inputs.{input_id}")
    root = protocol.repository_root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"inputs.{input_id}.path escapes the repository") from exc
    return path


@dataclass(frozen=True, slots=True)
class FeatureSetContract:
    variant: FeatureVariant
    contracts: tuple[FeatureColumnContract, ...]
    source_columns: tuple[str, ...]
    audit_only_quality_columns: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.variant not in {"coverage_only", "snapshot", "dynamic"}:
            raise ValueError("unknown stage-4 feature variant")
        if not self.contracts or tuple(item.source_column for item in self.contracts) != (
            self.source_columns
        ):
            raise ValueError("feature contracts must align exactly with source columns")
        if set(self.source_columns) & set(self.audit_only_quality_columns):
            raise ValueError("audit-only quality columns cannot enter the design")


def feature_set_contract(
    protocol: Stage4ProtocolBundle,
    variant: FeatureVariant,
) -> FeatureSetContract:
    """Parse the accepted manifest into the executable preprocessor contract."""

    feature_sets = _mapping(protocol.feature_set.document.get("feature_sets"), label="feature_sets")
    definition = _mapping(feature_sets.get(variant), label=f"feature_sets.{variant}")
    source_columns = _strings(
        definition.get("source_value_columns"),
        label=f"feature_sets.{variant}.source_value_columns",
    )
    design_entries = tuple(
        _mapping(item, label=f"feature_sets.{variant}.design_columns")
        for item in cast(Sequence[object], definition.get("design_columns"))
    )
    value_by_source = {
        cast(str, item.get("source_column")): item
        for item in design_entries
        if item.get("role") == "value"
    }
    missing_by_source = {
        cast(str, item.get("source_column")): item
        for item in design_entries
        if item.get("role") == "missing_indicator"
    }
    if set(value_by_source) != set(source_columns) or set(missing_by_source) != set(source_columns):
        raise ValueError("feature manifest value/missing pairs differ from source columns")
    contracts: list[FeatureColumnContract] = []
    allowed_transforms = {
        "identity_finite",
        "identity_binary",
        "log1p_nonnegative",
        "asinh_signed",
    }
    for source in source_columns:
        value = value_by_source[source]
        missing = missing_by_source[source]
        transform = value.get("transform")
        if transform not in allowed_transforms:
            raise ValueError(f"unsupported transform in accepted manifest: {transform}")
        logical = value.get("logical_feature")
        value_output = value.get("output_column")
        missing_output = missing.get("output_column")
        penalty = value.get("penalty_factor")
        if (
            not isinstance(logical, str)
            or not isinstance(value_output, str)
            or not isinstance(missing_output, str)
            or not isinstance(penalty, int | float)
            or isinstance(penalty, bool)
            or missing.get("logical_feature") != logical
            or missing.get("penalty_factor") != penalty
        ):
            raise ValueError("feature manifest value/missing contract is inconsistent")
        contracts.append(
            FeatureColumnContract(
                logical_feature=logical,
                source_column=source,
                value_output_column=value_output,
                missing_output_column=missing_output,
                transform=cast(TransformName, transform),
                penalty_factor=float(penalty),
            )
        )
    return FeatureSetContract(
        variant=variant,
        contracts=tuple(contracts),
        source_columns=source_columns,
        audit_only_quality_columns=_strings(
            definition.get("audit_only_quality_columns"),
            label=f"feature_sets.{variant}.audit_only_quality_columns",
        ),
    )


def _utc(value: object, *, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware")
    return value.astimezone(UTC)


def _row_group_issue_time(file: pq.ParquetFile, row_group_index: int) -> datetime:
    schema_index = file.schema_arrow.get_field_index("issue_time_utc")
    if schema_index < 0:
        raise ValueError("stage-3 feature store omitted issue_time_utc")
    statistics = file.metadata.row_group(row_group_index).column(schema_index).statistics
    if statistics is None or not statistics.has_min_max or statistics.min != statistics.max:
        raise ValueError("each accepted feature-store row group must contain one issue time")
    return _utc(statistics.min, label="feature row-group issue time")


def _numpy(table: pa.Table, name: str, dtype: np.dtype[np.generic]) -> np.ndarray:
    return np.asarray(
        table[name].combine_chunks().to_numpy(zero_copy_only=False),
        dtype=dtype,
    )


def assert_issue_table_matches_frozen_grid(
    table: pa.Table,
    *,
    issue_time_utc: datetime,
    grid: Stage4IntegrationGrid,
) -> None:
    """Require exact row identity before a feature field enters any model."""

    missing = sorted(set(FEATURE_IDENTITY_COLUMNS) - set(table.column_names))
    if missing:
        raise ValueError(f"feature issue table is missing identity columns: {missing}")
    if table.num_rows != grid.cell_count:
        raise ValueError("feature issue row count differs from the frozen 25 km grid")
    observed_times = table["issue_time_utc"].combine_chunks().to_pylist()
    expected_time = _utc(issue_time_utc, label="expected issue time")
    if any(_utc(item, label="feature issue time") != expected_time for item in observed_times):
        raise ValueError("feature table contains another issue time")
    if set(table["grid_id"].combine_chunks().to_pylist()) != {grid.grid_id}:
        raise ValueError("feature table grid_id differs from the frozen primary grid")
    if tuple(table["cell_id"].combine_chunks().to_pylist()) != grid.cell_ids:
        raise ValueError("feature table cell order differs from the frozen primary grid")
    exact_arrays = (
        ("cell_row", grid.rows, np.dtype(np.int64)),
        ("cell_column", grid.columns, np.dtype(np.int64)),
        ("query_x_m", grid.query_xy_m[:, 0], np.dtype(np.float64)),
        ("query_y_m", grid.query_xy_m[:, 1], np.dtype(np.float64)),
        ("clipped_area_km2", grid.clipped_area_km2, np.dtype(np.float64)),
    )
    for name, expected, dtype in exact_arrays:
        if not np.array_equal(_numpy(table, name, dtype), np.asarray(expected, dtype=dtype)):
            raise ValueError(f"feature table {name} differs from the frozen primary grid")


def load_verified_issue_feature_tables(
    protocol: Stage4ProtocolBundle,
    score_blind_inputs: ScoreBlindInputEvidence,
    *,
    issue_times_utc: Sequence[datetime],
    columns: Sequence[str],
    primary_grid: Stage4IntegrationGrid,
) -> dict[datetime, pa.Table]:
    """Read only requested accepted row groups, after score-blind hash verification."""

    expected_times = tuple(_utc(item, label="requested issue time") for item in issue_times_utc)
    if not expected_times or len(set(expected_times)) != len(expected_times):
        raise ValueError("requested issue times must be non-empty and unique")
    selected = tuple(dict.fromkeys((*FEATURE_IDENTITY_COLUMNS, *columns)))
    if len(selected) != len(FEATURE_IDENTITY_COLUMNS) + len(set(columns)):
        raise ValueError("requested feature columns overlap identity columns")
    path = _verified_input_path(
        protocol,
        score_blind_inputs,
        input_id="stage3_feature_store",
    )
    parquet = pq.ParquetFile(path)
    absent = sorted(set(selected) - set(parquet.schema_arrow.names))
    if absent:
        raise ValueError(f"accepted feature store is missing selected columns: {absent}")
    row_group_by_time: dict[datetime, int] = {}
    for index in range(parquet.metadata.num_row_groups):
        issue_time = _row_group_issue_time(parquet, index)
        if issue_time in row_group_by_time:
            raise ValueError("accepted feature store has duplicate issue row groups")
        row_group_by_time[issue_time] = index
    missing_times = sorted(set(expected_times) - set(row_group_by_time))
    if missing_times:
        raise ValueError(f"accepted feature store omitted requested issue times: {missing_times}")
    output: dict[datetime, pa.Table] = {}
    for issue_time in expected_times:
        table = parquet.read_row_group(row_group_by_time[issue_time], columns=list(selected))
        assert_issue_table_matches_frozen_grid(
            table,
            issue_time_utc=issue_time,
            grid=primary_grid,
        )
        output[issue_time] = table
    return output


def concatenate_source_columns(
    tables: Sequence[pa.Table],
    *,
    source_columns: Sequence[str],
) -> dict[str, np.ndarray]:
    """Create preprocessor inputs without using audit-only quality companions."""

    names = tuple(source_columns)
    if not tables or not names or len(set(names)) != len(names):
        raise ValueError("feature table/source selection must be non-empty and unique")
    output: dict[str, np.ndarray] = {}
    for name in names:
        chunks: list[np.ndarray] = []
        for table in tables:
            if name not in table.column_names:
                raise ValueError(f"feature table omitted source column: {name}")
            array = table[name].combine_chunks().cast(pa.float64())
            values = np.asarray(array.to_numpy(zero_copy_only=False), dtype=np.float64)
            missing = np.asarray(array.is_null().to_numpy(zero_copy_only=False), dtype=np.bool_)
            values = values.copy()
            values[missing] = np.nan
            chunks.append(values)
        output[name] = np.concatenate(chunks)
    return output


def load_verified_stage3_snapshots(
    protocol: Stage4ProtocolBundle,
    score_blind_inputs: ScoreBlindInputEvidence,
    *,
    issue_times_utc: Sequence[datetime] | None = None,
) -> tuple[Stage3IssueSnapshot, ...]:
    """Load the accepted anomaly state history; no target catalogue is consulted."""

    path = _verified_input_path(
        protocol,
        score_blind_inputs,
        input_id="stage3_anomaly_state_history",
    )
    parquet = pq.ParquetFile(path)
    if issue_times_utc is None:
        row_group_indices = tuple(range(parquet.metadata.num_row_groups))
    else:
        expected_times = tuple(
            _utc(item, label="requested state-history issue time") for item in issue_times_utc
        )
        if not expected_times or len(set(expected_times)) != len(expected_times):
            raise ValueError("requested state-history issue times must be non-empty and unique")
        row_group_by_time = {
            _row_group_issue_time(parquet, index): index
            for index in range(parquet.metadata.num_row_groups)
        }
        if len(row_group_by_time) != parquet.metadata.num_row_groups:
            raise ValueError("accepted state history has duplicate issue row groups")
        missing = sorted(set(expected_times) - set(row_group_by_time))
        if missing:
            raise ValueError(f"accepted state history omitted requested issue times: {missing}")
        row_group_indices = tuple(row_group_by_time[item] for item in sorted(expected_times))
    tables = tuple(parquet.read_row_group(index) for index in row_group_indices)
    table = pa.concat_tables(tables)
    states = states_from_records(table.to_pylist())
    return build_issue_snapshots(states, expected_issue_count=len(row_group_indices))


__all__ = [
    "FEATURE_IDENTITY_COLUMNS",
    "TIME_PLACEBO_BASE_COLUMNS",
    "FeatureSetContract",
    "FeatureVariant",
    "assert_issue_table_matches_frozen_grid",
    "concatenate_source_columns",
    "feature_set_contract",
    "load_verified_issue_feature_tables",
    "load_verified_stage3_snapshots",
]
