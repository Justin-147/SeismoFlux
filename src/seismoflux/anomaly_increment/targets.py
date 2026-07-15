"""In-memory parsing and frozen-grid scoring views for authorized stage-4 targets.

This module accepts bytes, never a path.  Consequently it cannot bypass the sole
file entrance in :mod:`seismoflux.anomaly_increment.target_access`.  True event
locations are used only after authorization to map events to already-frozen 25 km
cells; they cannot construct, refine, rank, or bound the grid.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import UTC, datetime, time, timedelta
from typing import Any, Literal, cast
from zoneinfo import ZoneInfo

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray
from pyproj import CRS, Transformer
from shapely import covers, points
from shapely.geometry import Point

from seismoflux.anomaly_increment.grid_features import Stage4IntegrationGrid
from seismoflux.anomaly_increment.runner import ExposurePlan
from seismoflux.background.catalog import StudyArea
from seismoflux.background.grid import (
    GridSpec,
    build_clipped_grid,
    point_cell_index,
)
from seismoflux.data.parquet import schema_sha256, table_content_sha256

MagnitudeBinId = Literal["M5_6", "M6_plus"]
IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

STAGE4_TARGET_COLUMN_ALLOWLIST = (
    "event_id",
    "origin_time_utc",
    "available_at",
    "longitude",
    "latitude",
    "magnitude",
    "inside_study_area",
)


def _readonly(array: NDArray[Any]) -> NDArray[Any]:
    result = np.ascontiguousarray(array)
    result.setflags(write=False)
    return result


def _sha256(value: str, *, label: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _aware_utc(value: object, *, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{label} must be a timezone-aware datetime")
    return value.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class Stage4TargetCatalog:
    """Immutable physical-event target columns allowed by the stage-4 protocol."""

    event_id: NDArray[np.str_]
    origin_time_utc: tuple[datetime, ...]
    available_at_utc: tuple[datetime, ...]
    longitude: FloatArray
    latitude: FloatArray
    x_m: FloatArray
    y_m: FloatArray
    magnitude: FloatArray
    inside_study_area: BoolArray
    source_content_sha256: str
    source_schema_sha256: str

    def __post_init__(self) -> None:
        lengths = {
            len(cast(Any, getattr(self, field.name)))
            for field in fields(self)
            if field.name not in {"source_content_sha256", "source_schema_sha256"}
        }
        if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
            raise ValueError("stage-4 target columns must have one nonzero common length")
        count = next(iter(lengths))
        if len(set(str(value) for value in self.event_id)) != count:
            raise ValueError("stage-4 target event IDs must be unique physical events")
        if any(
            value.tzinfo is None or value.utcoffset() != timedelta(0)
            for value in self.origin_time_utc
        ):
            raise ValueError("stage-4 target origin times must be UTC aware")
        if any(
            value.tzinfo is None or value.utcoffset() != timedelta(0)
            for value in self.available_at_utc
        ):
            raise ValueError("stage-4 target availability times must be UTC aware")
        if any(
            available < origin
            for origin, available in zip(
                self.origin_time_utc,
                self.available_at_utc,
                strict=True,
            )
        ):
            raise ValueError("target availability cannot precede event origin")
        for name in ("longitude", "latitude", "x_m", "y_m", "magnitude"):
            values = np.asarray(getattr(self, name), dtype=np.float64)
            if values.shape != (count,) or not np.all(np.isfinite(values)):
                raise ValueError(f"stage-4 target {name} must be a finite vector")
            object.__setattr__(self, name, cast(FloatArray, _readonly(values)))
        if np.any((self.longitude < -180.0) | (self.longitude > 180.0)) or np.any(
            (self.latitude < -90.0) | (self.latitude > 90.0)
        ):
            raise ValueError("stage-4 target coordinates are outside geographic bounds")
        inside = np.asarray(self.inside_study_area, dtype=np.bool_)
        if inside.shape != (count,):
            raise ValueError("stage-4 target inside flag has the wrong shape")
        object.__setattr__(self, "inside_study_area", cast(BoolArray, _readonly(inside)))
        object.__setattr__(
            self,
            "event_id",
            cast(NDArray[np.str_], _readonly(np.asarray(self.event_id, dtype=np.str_))),
        )
        _sha256(self.source_content_sha256, label="source_content_sha256")
        _sha256(self.source_schema_sha256, label="source_schema_sha256")

    def __len__(self) -> int:
        return len(self.event_id)

    def subset(self, mask: BoolArray) -> Stage4TargetCatalog:
        boolean = np.asarray(mask, dtype=np.bool_)
        if boolean.shape != (len(self),):
            raise ValueError("stage-4 target subset mask has the wrong shape")
        indices = np.flatnonzero(boolean)
        if indices.size == 0:
            raise ValueError("stage-4 target subset is empty")
        return Stage4TargetCatalog(
            event_id=self.event_id[indices],
            origin_time_utc=tuple(self.origin_time_utc[index] for index in indices),
            available_at_utc=tuple(self.available_at_utc[index] for index in indices),
            longitude=self.longitude[indices],
            latitude=self.latitude[indices],
            x_m=self.x_m[indices],
            y_m=self.y_m[indices],
            magnitude=self.magnitude[indices],
            inside_study_area=self.inside_study_area[indices],
            source_content_sha256=self.source_content_sha256,
            source_schema_sha256=self.source_schema_sha256,
        )


@dataclass(frozen=True, slots=True)
class ExposureTargetView:
    exposure: ExposurePlan
    magnitude_bin_id: MagnitudeBinId
    event_indices: IntArray
    event_ids: tuple[str, ...]
    lead_days: FloatArray

    def __post_init__(self) -> None:
        indices = np.asarray(self.event_indices, dtype=np.int64)
        leads = np.asarray(self.lead_days, dtype=np.float64)
        if indices.shape != leads.shape or indices.shape != (len(self.event_ids),):
            raise ValueError("exposure target arrays have inconsistent shapes")
        if (
            np.any(indices < 0)
            or np.any(leads <= 0.0)
            or np.any(leads > self.exposure.horizon_days)
        ):
            raise ValueError("exposure targets must lie strictly inside (T,T+h]")
        if len(set(self.event_ids)) != len(self.event_ids):
            raise ValueError("exposure target view contains duplicate physical events")
        object.__setattr__(self, "event_indices", cast(IntArray, _readonly(indices)))
        object.__setattr__(self, "lead_days", cast(FloatArray, _readonly(leads)))


@dataclass(frozen=True, slots=True)
class TargetCellAssignments:
    """Local-only event-to-frozen-cell mapping used in likelihood event terms."""

    event_ids: tuple[str, ...]
    cell_indices: IntArray
    cell_ids: tuple[str, ...]
    grid_id: str

    def __post_init__(self) -> None:
        indices = np.asarray(self.cell_indices, dtype=np.int64)
        if indices.shape != (len(self.event_ids),) or len(self.cell_ids) != len(self.event_ids):
            raise ValueError("target cell assignment shapes disagree")
        if np.any(indices < 0):
            raise ValueError("target cell assignment index must be nonnegative")
        object.__setattr__(self, "cell_indices", cast(IntArray, _readonly(indices)))


def parse_authorized_stage4_target_bytes(
    payload: bytes,
    *,
    expected_content_sha256: str,
    expected_schema_sha256: str,
    study_area: StudyArea,
) -> Stage4TargetCatalog:
    """Parse bytes supplied by the authorized one-shot entrance and verify identities."""

    _sha256(expected_content_sha256, label="expected_content_sha256")
    _sha256(expected_schema_sha256, label="expected_schema_sha256")
    if not payload:
        raise ValueError("authorized stage-4 target payload is empty")
    full_table = pq.read_table(pa.BufferReader(payload))
    observed_content = table_content_sha256(full_table)
    observed_schema = schema_sha256(full_table.schema)
    if observed_content != expected_content_sha256:
        raise ValueError("authorized target content hash differs from the protocol")
    if observed_schema != expected_schema_sha256:
        raise ValueError("authorized target schema hash differs from the protocol")
    missing = sorted(set(STAGE4_TARGET_COLUMN_ALLOWLIST) - set(full_table.column_names))
    if missing:
        raise ValueError(f"authorized target is missing allowlisted columns: {missing}")
    table = full_table.select(list(STAGE4_TARGET_COLUMN_ALLOWLIST)).combine_chunks()
    if table.num_rows == 0 or any(table[name].null_count for name in table.column_names):
        raise ValueError("stage-4 target allowlist must be nonempty and non-null")

    event_id = np.asarray(table["event_id"].to_numpy(zero_copy_only=False), dtype=np.str_)
    origin = tuple(
        _aware_utc(value, label="origin_time_utc") for value in table["origin_time_utc"].to_pylist()
    )
    available = tuple(
        _aware_utc(value, label="available_at") for value in table["available_at"].to_pylist()
    )
    longitude = np.asarray(table["longitude"].to_numpy(zero_copy_only=False), dtype=np.float64)
    latitude = np.asarray(table["latitude"].to_numpy(zero_copy_only=False), dtype=np.float64)
    magnitude = np.asarray(table["magnitude"].to_numpy(zero_copy_only=False), dtype=np.float64)
    inside = np.asarray(table["inside_study_area"].to_numpy(zero_copy_only=False), dtype=np.bool_)
    transformer = Transformer.from_crs(
        CRS.from_epsg(4326),
        CRS.from_user_input(study_area.equal_area_crs),
        always_xy=True,
    )
    x_raw, y_raw = transformer.transform(longitude, latitude)
    x_m = np.asarray(x_raw, dtype=np.float64)
    y_m = np.asarray(y_raw, dtype=np.float64)
    inside_computed = np.asarray(covers(study_area.projected, points(x_m, y_m)), dtype=np.bool_)
    if not np.array_equal(inside, inside_computed):
        raise ValueError("authorized target inside flags differ from the frozen study area")
    expected_order = sorted(
        range(len(event_id)), key=lambda index: (origin[index], event_id[index])
    )
    if expected_order != list(range(len(event_id))):
        raise ValueError("authorized target must remain sorted by origin time and event ID")
    return Stage4TargetCatalog(
        event_id=event_id,
        origin_time_utc=origin,
        available_at_utc=available,
        longitude=longitude,
        latitude=latitude,
        x_m=x_m,
        y_m=y_m,
        magnitude=magnitude,
        inside_study_area=inside,
        source_content_sha256=observed_content,
        source_schema_sha256=observed_schema,
    )


def exposure_target_view(
    catalog: Stage4TargetCatalog,
    exposure: ExposurePlan,
    *,
    magnitude_bin_id: MagnitudeBinId,
) -> ExposureTargetView:
    """Select physical events in the frozen open-closed target window."""

    timezone = ZoneInfo("Asia/Shanghai")
    issue = datetime.combine(exposure.issue_date_local, time.min, tzinfo=timezone).astimezone(UTC)
    end = issue + timedelta(days=exposure.horizon_days)
    time_mask = np.fromiter(
        (issue < value <= end for value in catalog.origin_time_utc),
        dtype=np.bool_,
        count=len(catalog),
    )
    if magnitude_bin_id == "M5_6":
        magnitude_mask = (catalog.magnitude >= 5.0) & (catalog.magnitude < 6.0)
    elif magnitude_bin_id == "M6_plus":
        magnitude_mask = catalog.magnitude >= 6.0
    else:
        raise ValueError("stage-4 magnitude bin changed")
    indices = np.flatnonzero(time_mask & magnitude_mask & catalog.inside_study_area)
    leads = np.asarray(
        [(catalog.origin_time_utc[index] - issue).total_seconds() / 86_400.0 for index in indices],
        dtype=np.float64,
    )
    return ExposureTargetView(
        exposure=exposure,
        magnitude_bin_id=magnitude_bin_id,
        event_indices=indices,
        event_ids=tuple(str(catalog.event_id[index]) for index in indices),
        lead_days=leads,
    )


def map_targets_to_frozen_primary_grid(
    catalog: Stage4TargetCatalog,
    *,
    study_area: StudyArea,
    primary_grid: Stage4IntegrationGrid,
) -> TargetCellAssignments:
    """Map inside targets to the pre-existing 25 km cells with frozen tie-breaking."""

    if primary_grid.cell_size_km != 25.0:
        raise ValueError("stage-4 event terms must use the frozen 25 km grid")
    rebuilt = build_clipped_grid(study_area.projected, cell_size_km=25.0)
    if rebuilt.cell_ids != primary_grid.cell_ids:
        raise ValueError("target mapping grid differs from the target-independent primary grid")
    cell_by_index = {(cell.row, cell.column): cell for cell in rebuilt.cells}
    output_indices: list[int] = []
    output_ids: list[str] = []
    id_to_index = {cell_id: index for index, cell_id in enumerate(primary_grid.cell_ids)}
    spec = GridSpec(25.0)
    for event_index, (x_m, y_m, inside) in enumerate(
        zip(catalog.x_m, catalog.y_m, catalog.inside_study_area, strict=True)
    ):
        if not inside:
            continue
        row, column = point_cell_index(float(x_m), float(y_m), spec)
        point = Point(float(x_m), float(y_m))
        matches = [
            cell
            for candidate_row in range(row - 1, row + 2)
            for candidate_column in range(column - 1, column + 2)
            if (cell := cell_by_index.get((candidate_row, candidate_column))) is not None
            and cell.clipped_geometry.covers(point)
        ]
        if not matches:
            raise ValueError(
                f"inside target {catalog.event_id[event_index]} does not map to a frozen cell"
            )
        selected = min(matches, key=lambda cell: (cell.row, cell.column, cell.id))
        output_ids.append(selected.id)
        output_indices.append(id_to_index[selected.id])
    inside_ids = tuple(
        str(event_id)
        for event_id, inside in zip(catalog.event_id, catalog.inside_study_area, strict=True)
        if inside
    )
    if len(output_indices) != len(inside_ids):
        raise AssertionError("not every inside target received one frozen-cell assignment")
    return TargetCellAssignments(
        event_ids=inside_ids,
        cell_indices=np.asarray(output_indices, dtype=np.int64),
        cell_ids=tuple(output_ids),
        grid_id=primary_grid.grid_id,
    )


__all__ = [
    "STAGE4_TARGET_COLUMN_ALLOWLIST",
    "ExposureTargetView",
    "MagnitudeBinId",
    "Stage4TargetCatalog",
    "TargetCellAssignments",
    "exposure_target_view",
    "map_targets_to_frozen_primary_grid",
    "parse_authorized_stage4_target_bytes",
]
