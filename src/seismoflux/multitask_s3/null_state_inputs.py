"""Bounded state and construction-stratum reads for the S3 spatial null world.

The caller authenticates source hashes once.  No original source is changed, no
full preflight is called, and no future state row group is materialized then
filtered.  Paths and the exact authorized report axis are supplied by the caller.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from seismoflux.features.anomaly.contracts import ANOMALY_STATE_HISTORY_CONTRACT
from seismoflux.features.anomaly.snapshot import Stage3IssueSnapshot, build_issue_snapshots
from seismoflux.features.anomaly.state import states_from_records
from seismoflux.multitask_s3.features import REPORT_END_UTC, REPORT_START_UTC

STATE_COLUMNS = tuple(ANOMALY_STATE_HISTORY_CONTRACT.schema.names)
ENTITY_MAPPING_COLUMNS = ("state_id", "issue_time_utc", "construction_stratum_id")


def _utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("issue and cutoff must be exact timezone-aware datetimes")
    return value.astimezone(UTC)


def _authorized_times(
    issue_times_utc: Sequence[datetime], report_end_exclusive: datetime
) -> tuple[tuple[datetime, ...], datetime]:
    end = _utc(report_end_exclusive)
    if not REPORT_START_UTC < end <= REPORT_END_UTC:
        raise ValueError("fold cutoff is outside authorized S3 development interval")
    times = tuple(_utc(value) for value in issue_times_utc)
    if not times or len(times) != len(set(times)):
        raise ValueError("requested issues must be nonempty and unique")
    if any(not REPORT_START_UTC <= value < end for value in times):
        raise ValueError("requested issue is outside authorized fold interval")
    return times, end


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a nonempty trimmed string")
    return value


def _requested_row_groups(
    parquet: pq.ParquetFile, times: tuple[datetime, ...]
) -> dict[datetime, list[int]]:
    """Inspect only footer dates; an issue may occupy several complete row groups."""
    schema = parquet.schema_arrow
    expected = ANOMALY_STATE_HISTORY_CONTRACT.schema
    if len(schema.names) != len(STATE_COLUMNS) or set(schema.names) != set(STATE_COLUMNS):
        raise ValueError("state history must contain exactly the complete state schema")
    if any(schema.field(name).type != expected.field(name).type for name in STATE_COLUMNS):
        raise ValueError("state history column types differ from the complete state schema")
    time_columns = [
        index
        for index in range(parquet.metadata.num_columns)
        if parquet.schema.column(index).path == "issue_time_utc"
    ]
    if len(time_columns) != 1:
        raise ValueError("state history requires one top-level issue_time_utc column")
    groups: dict[datetime, list[int]] = {time: [] for time in times}
    for index in range(parquet.metadata.num_row_groups):
        group = parquet.metadata.row_group(index)
        stats = group.column(time_columns[0]).statistics
        if (
            stats is None
            or not stats.has_min_max
            or stats.null_count != 0
            or stats.min != stats.max
            or group.num_rows == 0
        ):
            raise ValueError("each state row group must have one non-null exact issue time")
        time = _utc(stats.min)
        if time in groups:
            groups[time].append(index)
    if any(not indices for indices in groups.values()):
        raise ValueError("state history omitted a requested exact report issue")
    return groups


def load_issue_snapshots(
    verified_state_history_path: str | Path,
    *,
    issue_times_utc: Sequence[datetime],
    report_end_exclusive: datetime,
) -> dict[datetime, Stage3IssueSnapshot]:
    """Read complete authorized states, retaining summaries and ineligible entities.

    Multiple row groups of one requested issue are combined before rehydration.
    Returned mapping order follows the caller; snapshot ``issue_index`` follows
    the chronological requested axis.  Missing coordinates/NaNs are not filtered.
    """
    times, _ = _authorized_times(issue_times_utc, report_end_exclusive)
    chronological_index = {time: index for index, time in enumerate(sorted(times))}
    result: dict[datetime, Stage3IssueSnapshot] = {}
    seen_state_ids: set[str] = set()
    with pq.ParquetFile(verified_state_history_path) as parquet:
        groups = _requested_row_groups(parquet, times)
        for time in times:
            tables = [
                parquet.read_row_group(index, columns=list(STATE_COLUMNS), use_threads=False)
                for index in groups[time]
            ]
            table = pa.concat_tables(tables)
            if any(_utc(value) != time for value in table["issue_time_utc"].to_pylist()):
                raise ValueError("state row group changed its exact issue time")
            state_ids = tuple(
                _identifier(value, "state_id") for value in table["state_id"].to_pylist()
            )
            if len(set(state_ids)) != len(state_ids) or seen_state_ids.intersection(state_ids):
                raise ValueError("state history contains duplicate state IDs")
            states = states_from_records(table.to_pylist())
            snapshot = build_issue_snapshots(states, expected_issue_count=1)[0]
            if _utc(snapshot.issue_time_utc) != time:
                raise ValueError("state snapshot changed its exact issue time")
            result[time] = replace(snapshot, issue_index=chronological_index[time])
            seen_state_ids.update(state_ids)
    return result


def load_construction_strata(
    verified_entity_mapping_path: str | Path,
    *,
    snapshots_by_issue: Mapping[datetime, Stage3IssueSnapshot],
    report_end_exclusive: datetime,
) -> dict[str, str]:
    """Read only three mapping columns through an exact authorized-date predicate.

    Mapping IDs must cover exactly the spatially eligible states, with the same
    issue timestamp.  Non-spatial states remain in the supplied snapshots; they
    do not become coordinate-permutation entities and are never silently lost.
    """
    times, end = _authorized_times(tuple(snapshots_by_issue), report_end_exclusive)
    eligible: dict[str, datetime] = {}
    seen: set[str] = set()
    for key, snapshot in snapshots_by_issue.items():
        time = _utc(key)
        if not isinstance(snapshot, Stage3IssueSnapshot) or _utc(snapshot.issue_time_utc) != time:
            raise ValueError("snapshot mapping key differs from its exact issue time")
        for state in (snapshot.summary, *snapshot.entities):
            state_id = _identifier(state.state_id, "state_id")
            if _utc(state.issue_time_utc) != time or state_id in seen:
                raise ValueError("snapshot state date or unique state ID does not match")
            seen.add(state_id)
        for state in snapshot.entities:
            if state.spatial_eligible:
                eligible[state.state_id] = time
    table = pq.read_table(
        verified_entity_mapping_path,
        columns=list(ENTITY_MAPPING_COLUMNS),
        filters=[
            ("issue_time_utc", "in", list(times)),
            ("issue_time_utc", ">=", REPORT_START_UTC),
            ("issue_time_utc", "<", end),
        ],
        use_threads=False,
    )
    result: dict[str, str] = {}
    for row in table.to_pylist():
        state_id = _identifier(row["state_id"], "state_id")
        time = _utc(row["issue_time_utc"])
        stratum = _identifier(row["construction_stratum_id"], "construction_stratum_id")
        if time not in times or state_id not in eligible or eligible[state_id] != time:
            raise ValueError("entity mapping date or state ID does not match its snapshot")
        if state_id in result:
            raise ValueError("entity mapping contains duplicate state IDs")
        result[state_id] = stratum
    if set(result) != set(eligible):
        raise ValueError("entity mapping does not cover exactly the spatially eligible states")
    return result


def load_all_zone_ids(verified_cell_mapping_path: str | Path) -> tuple[str, ...]:
    """Read only static construction-zone IDs, without a target-derived boundary."""
    table = pq.read_table(
        verified_cell_mapping_path, columns=["construction_zone_id"], use_threads=False
    )
    values = [
        _identifier(value, "construction_zone_id")
        for value in table["construction_zone_id"].to_pylist()
    ]
    if not values:
        raise ValueError("static cell mapping must contain construction zones")
    return tuple(sorted(set(values), key=lambda value: value.encode("utf-8")))
