"""Causal source-member inputs for the finite C2B catalog comparison."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray

from seismoflux.multitask_s1.runner_inputs import (
    CatalogEventTable,
    catalog_event_table_from_frame,
)

SOURCE_IDS = ("earthquake_catalog_m3_plus", "earthquake_catalog_m5_plus")
NO_SOURCE = np.iinfo(np.int64).max
MAXIMUM_HISTORY_UTC = datetime(2019, 12, 30, 16, tzinfo=UTC)
EVENT_COLUMNS = (
    "event_id",
    "origin_time_utc",
    "available_at",
    "longitude",
    "latitude",
    "magnitude",
    "inside_study_area",
    "catalog_sources",
)
SOURCE_COLUMNS = ("source_record_id", "source_id", "origin_time_utc", "available_at")


@dataclass(frozen=True, slots=True)
class C2BCatalog:
    table: CatalogEventTable
    source_visible_us: dict[str, NDArray[np.int64]]

    def __post_init__(self) -> None:
        if set(self.source_visible_us) != set(SOURCE_IDS):
            raise ValueError("C2B requires exactly the two declared local source identities")
        result = {}
        for source_id, values in self.source_visible_us.items():
            array = np.array(values, dtype=np.int64, copy=True)
            if array.shape != (self.table.row_count,):
                raise ValueError("source visibility must align with all catalog event IDs")
            array.setflags(write=False)
            result[source_id] = array
        object.__setattr__(self, "source_visible_us", result)


def _timestamp_us(value: datetime | str) -> int:
    instant = pd.Timestamp(value)
    if instant.tzinfo is None or pd.isna(instant):
        raise ValueError("C2B timestamps must be valid and timezone aware")
    utc = instant.tz_convert("UTC")
    nanos = int(utc.as_unit("ns").asm8.astype(np.int64))
    if nanos % 1000:
        raise ValueError("C2B timestamps must be exactly representable in microseconds")
    return int(utc.as_unit("us").asm8.astype(np.int64))


def _column_us(frame: pd.DataFrame, name: str) -> NDArray[np.int64]:
    return np.asarray([_timestamp_us(value) for value in frame[name]], dtype=np.int64)


def c2b_catalog_from_frames(events: pd.DataFrame, sources: pd.DataFrame) -> C2BCatalog:
    """Align any-member visibility to existing canonical values, never source anchors."""

    if not set(EVENT_COLUMNS) <= set(events) or not set(SOURCE_COLUMNS) <= set(sources):
        raise ValueError("C2B event or source input columns are missing")
    if sources["source_record_id"].duplicated().any():
        raise ValueError("source record identifiers must be unique")
    if not set(sources["source_id"]) <= set(SOURCE_IDS):
        raise ValueError("unexpected catalog source outside the two local Ms sources")
    table = catalog_event_table_from_frame(events)
    source_visible = np.maximum(
        _column_us(sources, "origin_time_utc"), _column_us(sources, "available_at")
    )
    records = {
        str(record_id): (str(source_id), int(visible))
        for record_id, source_id, visible in zip(
            sources["source_record_id"], sources["source_id"], source_visible, strict=True
        )
    }
    members_by_event = dict(
        zip(events["event_id"].astype(str), events["catalog_sources"], strict=True)
    )
    visible_by_source = {
        source_id: np.full(table.row_count, NO_SOURCE, dtype=np.int64) for source_id in SOURCE_IDS
    }
    for index, event_id in enumerate(table.event_ids):
        for member in members_by_event[event_id]:
            record = records.get(str(member))
            if record is None:
                continue
            source_id, visible = record
            visible_by_source[source_id][index] = min(visible_by_source[source_id][index], visible)
    return C2BCatalog(table, visible_by_source)


def _verified_path(root: Path, relative: str, expected_sha: str, expected_rows: int) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("C2B catalog input escaped explicit data root")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if (
        digest.hexdigest() != expected_sha
        or pq.ParquetFile(path).metadata.num_rows != expected_rows
    ):
        raise ValueError("C2B catalog file hash or row count differs from protocol")
    return path


def load_c2b_catalog(*, data_root: Path, protocol: dict[str, Any]) -> C2BCatalog:
    """Authenticate inputs and materialize only pre-2020 causal history columns."""

    inputs = protocol["inputs"]
    event_path = _verified_path(
        data_root,
        inputs["canonical_catalog"],
        inputs["canonical_catalog_sha256"],
        inputs["canonical_catalog_rows"],
    )
    source_path = _verified_path(
        data_root,
        inputs["source_records"],
        inputs["source_records_sha256"],
        inputs["source_record_rows"],
    )
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    filters = [
        ("origin_time_utc", "<=", MAXIMUM_HISTORY_UTC),
        ("available_at", "<=", MAXIMUM_HISTORY_UTC),
    ]
    events = pq.read_table(
        event_path, columns=list(EVENT_COLUMNS), filters=filters, use_threads=False
    ).to_pandas()
    sources = pq.read_table(
        source_path, columns=list(SOURCE_COLUMNS), filters=filters, use_threads=False
    ).to_pandas()
    return c2b_catalog_from_frames(events, sources)


def panel_indices(
    catalog: C2BCatalog,
    panel_spec: dict[str, Any],
    issue_time_utc: datetime,
    *,
    recent_days: int | None = None,
) -> NDArray[np.int64]:
    """Select inclusive panel bounds and strictly issue-relative recent history."""

    issue = _timestamp_us(issue_time_utc)
    cutoff = issue - 24 * 3600 * 1_000_000
    table = catalog.table
    keep = (
        table.inside_study_area
        & (table.origin_time_us >= _timestamp_us(panel_spec["start_local"]))
        & (table.origin_time_us <= cutoff)
        & (table.available_at_us <= cutoff)
        & (table.magnitude >= float(panel_spec["magnitude_minimum"]))
    )
    source_id = panel_spec["required_source"]
    if source_id is not None:
        keep &= catalog.source_visible_us[source_id] <= cutoff
    if recent_days is not None:
        if isinstance(recent_days, bool) or not isinstance(recent_days, int) or recent_days <= 0:
            raise ValueError("recent_days must be a positive integer or None")
        lower = issue - recent_days * int(timedelta(days=1).total_seconds()) * 1_000_000
        keep &= table.origin_time_us > lower
    indices = np.flatnonzero(keep).astype(np.int64)
    indices.setflags(write=False)
    return indices
