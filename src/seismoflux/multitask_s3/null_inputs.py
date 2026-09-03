"""Read only requested, authorized radius base series for offline null rebuilding."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

from seismoflux.multitask_s3.features import (
    IDENTITY_COLUMNS,
    REPORT_END_UTC,
    REPORT_START_UTC,
    read_report_issue_metadata,
)

RADIUS_BASE_COLUMNS = ("radius_200km__listed_count", "radius_200km__first_seen_count")


def _utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("issue and cutoff must be exact timezone-aware datetimes")
    return value.astimezone(UTC)


def load_radius_bases(
    verified_store_path: str | Path,
    *,
    issue_times_utc: Sequence[datetime],
    expected_cell_ids: Sequence[str],
    expected_grid_id: str,
    report_end_exclusive: datetime,
) -> dict[datetime, NDArray[np.float64]]:
    """Return readonly (cells, 2) raw bases; no features, scores, or future RG reads.

    The caller verifies source identity once. Footer metadata may describe later
    dates; only explicitly requested dates before the authorized fold cutoff are
    read. NaN remains unknown, never replaced by zero or inferred from outcomes.
    """
    end = _utc(report_end_exclusive)
    if not REPORT_START_UTC < end <= REPORT_END_UTC:
        raise ValueError("fold cutoff is outside authorized development interval")
    times = tuple(_utc(value) for value in issue_times_utc)
    if not times or len(set(times)) != len(times):
        raise ValueError("requested issues must be nonempty and unique")
    if any(not REPORT_START_UTC <= value < end for value in times):
        raise ValueError("requested issue is outside authorized fold interval")
    cells = tuple(expected_cell_ids)
    if (
        not cells
        or any(not isinstance(value, str) or not value for value in cells)
        or len(set(cells)) != len(cells)
        or not isinstance(expected_grid_id, str)
        or not expected_grid_id
    ):
        raise ValueError("independent cell/grid identities must be nonempty and unique")
    metadata = {
        entry.issue_time_utc: entry
        for entry in read_report_issue_metadata(verified_store_path, report_end_exclusive=end)
    }
    if any(time not in metadata for time in times):
        raise ValueError("source omitted a requested exact report issue")
    columns = (*IDENTITY_COLUMNS, *RADIUS_BASE_COLUMNS)
    result: dict[datetime, NDArray[np.float64]] = {}
    with pq.ParquetFile(verified_store_path) as parquet:
        schema = parquet.schema_arrow
        if any(schema.get_field_index(name) < 0 for name in columns):
            raise ValueError("source omitted or duplicated a selected identity/base column")
        for name in RADIUS_BASE_COLUMNS:
            dtype = schema.field(name).type
            if not (pa.types.is_floating(dtype) or pa.types.is_integer(dtype)):
                raise ValueError("radius bases must be numeric")
        for time in times:
            table = parquet.read_row_group(
                metadata[time].row_group_index, columns=list(columns), use_threads=False
            )
            if table.num_rows != len(cells):
                raise ValueError("radius issue has missing or extra grid cells")
            if any(_utc(value) != time for value in table["issue_time_utc"].to_pylist()):
                raise ValueError("loaded row group contains a different exact issue time")
            if set(table["grid_id"].to_pylist()) != {expected_grid_id}:
                raise ValueError("radius grid differs from independent expected grid")
            observed = tuple(table["cell_id"].to_pylist())
            if len(set(observed)) != len(observed) or set(observed) != set(cells):
                raise ValueError("radius cell IDs are duplicate, missing, or extra")
            lookup = {value: index for index, value in enumerate(observed)}
            order = np.array([lookup[cell] for cell in cells], dtype=np.int64)
            values = np.column_stack(
                [
                    np.asarray(
                        table[name]
                        .combine_chunks()
                        .cast(pa.float64())
                        .to_numpy(zero_copy_only=False),
                        dtype=np.float64,
                    )[order]
                    for name in RADIUS_BASE_COLUMNS
                ]
            )
            if np.isinf(values).any() or np.any(values < 0):
                raise ValueError("radius bases must be nonnegative finite values or NaN")
            values.setflags(write=False)
            result[time] = values
    return result
