"""Thin S3-A adapter for already verified anomaly files, with no targets or scores.

Only requested report row groups and the frozen 16 value columns are read.  File
identity verification belongs to the caller; this module does not reuse Stage4
authorization or hash the multi-gigabyte source on each call.  Footer inspection
may see all row-group dates, but returned dates and loaded values are restricted
to the authorized development interval and the caller's explicit fold cutoff.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

from seismoflux.multitask_s3.models import nonnegative_log1p, signed_asinh

FloatArray = NDArray[np.float64]
REPORT_START_UTC = datetime.fromisoformat("2022-07-01T00:00:00+08:00").astimezone(UTC)
REPORT_END_UTC = datetime.fromisoformat("2025-07-01T00:00:00+08:00").astimezone(UTC)
IDENTITY_COLUMNS = ("issue_time_utc", "grid_id", "cell_id")
RAW_FEATURE_SPECS = (
    ("deformation", "gaussian_200km__discipline_deformation_count", "log1p"),
    ("fluid", "gaussian_200km__discipline_fluid_count", "log1p"),
    ("electromagnetic", "gaussian_200km__discipline_electromagnetic_count", "log1p"),
    ("cross_fault", "gaussian_200km__discipline_cross_fault_count", "log1p"),
    ("source_new", "gaussian_200km__source_new_count", "log1p"),
    ("first_seen", "gaussian_200km__first_seen_count", "log1p"),
    ("not_continued", "gaussian_200km__not_continued_count", "log1p"),
    ("age", "gaussian_200km__age_mean_days", "log1p"),
    ("listed_slope", "radius_200km__listed_count__slope_13w_per_week", "asinh"),
    (
        "listed_acceleration",
        "radius_200km__listed_count__acceleration_4v13_per_week2",
        "asinh",
    ),
    ("first_seen_slope", "radius_200km__first_seen_count__slope_13w_per_week", "asinh"),
    ("discipline_entropy", "gaussian_200km__discipline_shannon_normalized", "identity"),
    (
        "station_coverage",
        "gaussian_200km__distinct_reporting_station_count_reporting_coverage_proxy",
        "log1p",
    ),
    (
        "measurement_coverage",
        "gaussian_200km__distinct_reporting_measurement_count_reporting_coverage_proxy",
        "log1p",
    ),
    (
        "station_coverage_ratio",
        "gaussian_200km__current_to_trailing_station_reporting_coverage_proxy",
        "identity",
    ),
    (
        "measurement_coverage_ratio",
        "gaussian_200km__current_to_trailing_measurement_reporting_coverage_proxy",
        "identity",
    ),
)
RAW_FEATURE_IDS = tuple(spec[0] for spec in RAW_FEATURE_SPECS)
RAW_FEATURE_COLUMNS = tuple(spec[1] for spec in RAW_FEATURE_SPECS)
MISSING_FEATURE_IDS = ("age_missing", "dynamic_missing", "entropy_missing", "coverage_missing")
FULL_FEATURE_IDS = RAW_FEATURE_IDS + MISSING_FEATURE_IDS
DESIGN_INDICES = MappingProxyType(
    {
        "COV": (12, 13, 14, 15, 19),
        "SNAP": (0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16, 18, 19),
        "DYN": tuple(range(20)),
    }
)
DESIGN_FEATURE_IDS = MappingProxyType(
    {
        name: tuple(FULL_FEATURE_IDS[index] for index in indices)
        for name, indices in DESIGN_INDICES.items()
    }
)


def _readonly(values: FloatArray) -> FloatArray:
    result = np.array(values, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


def _utc(value: datetime, *, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be an exact timezone-aware datetime")
    # Timestamp subclasses retain their precision; never convert via to_pydatetime.
    return value.astimezone(UTC)


def _end(value: datetime) -> datetime:
    end = _utc(value, label="report_end_exclusive")
    if not REPORT_START_UTC < end <= REPORT_END_UTC:
        raise ValueError("fold report cutoff must be inside the authorized S3 development interval")
    return end


@dataclass(frozen=True, slots=True)
class ReportIssueMetadata:
    issue_time_utc: datetime
    row_group_index: int
    row_count: int


def _issue_metadata(
    parquet: pq.ParquetFile, report_end_exclusive: datetime
) -> tuple[ReportIssueMetadata, ...]:
    """Inspect footer statistics only; no row-group value access occurs here."""
    candidates = [
        index
        for index in range(parquet.metadata.num_columns)
        if parquet.schema.column(index).path == "issue_time_utc"
    ]
    if len(candidates) != 1:
        raise ValueError("feature store must have one top-level issue_time_utc field")
    time_column_index = candidates[0]
    issue_entries: dict[datetime, ReportIssueMetadata] = {}
    for index in range(parquet.metadata.num_row_groups):
        group = parquet.metadata.row_group(index)
        stats = group.column(time_column_index).statistics
        if (
            stats is None
            or not stats.has_min_max
            or stats.min != stats.max
            or stats.null_count != 0
            or group.num_rows == 0
        ):
            raise ValueError("each feature row group must contain one non-null exact issue time")
        issue_time = _utc(stats.min, label="row-group issue time")
        if REPORT_START_UTC <= issue_time < report_end_exclusive:
            if issue_time in issue_entries:
                raise ValueError("feature store has duplicate authorized issue row groups")
            issue_entries[issue_time] = ReportIssueMetadata(issue_time, index, group.num_rows)
    return tuple(issue_entries[time] for time in sorted(issue_entries))


def read_report_issue_metadata(
    verified_store_path: str | Path, *, report_end_exclusive: datetime
) -> tuple[ReportIssueMetadata, ...]:
    """Return only authorized report dates/counts; inspect no feature values."""
    end = _end(report_end_exclusive)
    with pq.ParquetFile(verified_store_path) as parquet:
        return _issue_metadata(parquet, end)


def build_feature_matrix(raw_features: Mapping[str, object]) -> FloatArray:
    """Transform 16 frozen values and append four masks, leaving value NaNs intact."""
    if set(raw_features) != set(RAW_FEATURE_COLUMNS):
        raise ValueError("raw feature mapping must contain exactly the 16 frozen source columns")
    raw_columns = [np.asarray(raw_features[name], dtype=np.float64) for name in RAW_FEATURE_COLUMNS]
    if (
        not raw_columns
        or raw_columns[0].ndim != 1
        or raw_columns[0].size == 0
        or any(column.shape != raw_columns[0].shape for column in raw_columns)
        or any(np.isinf(column).any() for column in raw_columns)
    ):
        raise ValueError("raw features must be nonempty aligned vectors, finite or NaN")
    transformed = []
    for column, (_, _, transform) in zip(raw_columns, RAW_FEATURE_SPECS, strict=True):
        if transform == "log1p":
            transformed.append(nonnegative_log1p(column))
        elif transform == "asinh":
            transformed.append(signed_asinh(column))
        else:
            transformed.append(column)
    raw = np.column_stack(raw_columns)
    controls = (
        np.isnan(raw[:, 7]).astype(np.float64),
        np.mean(np.isnan(raw[:, 8:11]), axis=1),
        np.isnan(raw[:, 11]).astype(np.float64),
        np.any(np.isnan(raw[:, 14:16]), axis=1).astype(np.float64),
    )
    return _readonly(np.column_stack((*transformed, *controls)))


@dataclass(frozen=True, slots=True)
class S3IssueFeatures:
    issue_time_utc: datetime
    grid_id: str
    cell_ids: tuple[str, ...]
    values: FloatArray
    source_row_group_index: int

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        if values.shape != (len(self.cell_ids), 20) or np.isinf(values).any():
            raise ValueError(
                "issue feature values must align as (expected cells, 20), finite or NaN"
            )
        object.__setattr__(self, "issue_time_utc", _utc(self.issue_time_utc, label="issue time"))
        object.__setattr__(self, "values", _readonly(values))

    def design(self, variant: str) -> FloatArray:
        """Select the frozen COV5, SNAP16, or DYN20 design, without fitting anything."""
        if variant not in DESIGN_INDICES:
            raise ValueError("variant must be COV, SNAP, or DYN")
        return _readonly(self.values[:, DESIGN_INDICES[variant]])


def load_issue_features(
    verified_store_path: str | Path,
    *,
    issue_times_utc: Sequence[datetime],
    expected_cell_ids: Sequence[str],
    expected_grid_id: str,
    report_end_exclusive: datetime,
) -> dict[datetime, S3IssueFeatures]:
    """Load requested report row groups only, then align to the independent grid.

    The caller supplies grid identity from the authoritative S0 support, not from
    this feature source.  Duplicate/missing/extra cells and out-of-fold dates fail
    explicitly.  The returned dictionary preserves requested issue order.
    """
    end = _end(report_end_exclusive)
    times = tuple(_utc(value, label="requested issue") for value in issue_times_utc)
    if not times or len(set(times)) != len(times):
        raise ValueError("requested issue list must be nonempty and unique")
    if any(not REPORT_START_UTC <= time < end for time in times):
        raise ValueError("requested issue lies outside the authorized fold report interval")
    cell_ids = tuple(expected_cell_ids)
    if (
        not cell_ids
        or any(not isinstance(value, str) or not value for value in cell_ids)
        or len(set(cell_ids)) != len(cell_ids)
        or not isinstance(expected_grid_id, str)
        or not expected_grid_id
    ):
        raise ValueError(
            "independent expected grid and cell identities must be nonempty and unique"
        )
    selected_columns = (*IDENTITY_COLUMNS, *RAW_FEATURE_COLUMNS)
    output: dict[datetime, S3IssueFeatures] = {}
    with pq.ParquetFile(verified_store_path) as parquet:
        schema = parquet.schema_arrow
        if any(schema.get_field_index(name) < 0 for name in selected_columns):
            raise ValueError("feature store omitted or duplicated a selected identity/value column")
        for name in RAW_FEATURE_COLUMNS:
            dtype = schema.field(name).type
            if not (pa.types.is_floating(dtype) or pa.types.is_integer(dtype)):
                raise ValueError(f"frozen feature column must be numeric: {name}")
        by_time = {entry.issue_time_utc: entry for entry in _issue_metadata(parquet, end)}
        if any(time not in by_time for time in times):
            raise ValueError("feature store omitted a requested exact report issue time")
        for time in times:
            entry = by_time[time]
            table = parquet.read_row_group(
                entry.row_group_index, columns=list(selected_columns), use_threads=False
            )
            if table.num_rows != len(cell_ids):
                raise ValueError("feature issue has missing or extra grid cells")
            observed_times = table["issue_time_utc"].to_pylist()
            if any(_utc(value, label="loaded issue time") != time for value in observed_times):
                raise ValueError("loaded feature values contain another issue time")
            if set(table["grid_id"].to_pylist()) != {expected_grid_id}:
                raise ValueError("feature grid_id differs from the independent expected grid")
            observed_ids = tuple(table["cell_id"].to_pylist())
            if len(set(observed_ids)) != len(observed_ids):
                raise ValueError("feature issue contains duplicate cell IDs")
            if set(observed_ids) != set(cell_ids):
                raise ValueError("feature cell IDs differ from the independent expected grid")
            row_by_id = {cell_id: index for index, cell_id in enumerate(observed_ids)}
            order = np.array([row_by_id[cell_id] for cell_id in cell_ids], dtype=np.int64)
            raw = {
                name: np.asarray(
                    table[name].combine_chunks().cast(pa.float64()).to_numpy(zero_copy_only=False),
                    dtype=np.float64,
                )[order]
                for name in RAW_FEATURE_COLUMNS
            }
            output[time] = S3IssueFeatures(
                time, expected_grid_id, cell_ids, build_feature_matrix(raw), entry.row_group_index
            )
    return output


__all__ = [
    "DESIGN_FEATURE_IDS",
    "DESIGN_INDICES",
    "FULL_FEATURE_IDS",
    "IDENTITY_COLUMNS",
    "MISSING_FEATURE_IDS",
    "RAW_FEATURE_COLUMNS",
    "RAW_FEATURE_IDS",
    "RAW_FEATURE_SPECS",
    "REPORT_END_UTC",
    "REPORT_START_UTC",
    "ReportIssueMetadata",
    "S3IssueFeatures",
    "build_feature_matrix",
    "load_issue_features",
    "read_report_issue_metadata",
]
