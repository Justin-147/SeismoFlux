"""Target-independent multigrid anomaly feature reconstruction for stage 4.

Only the frozen study-area geometry and accepted anomaly snapshots may influence
grid construction or feature values.  Earthquake locations are intentionally not
accepted by any function in this module.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray
from shapely.geometry.base import BaseGeometry

from seismoflux.anomaly_increment.config import Stage4ProtocolBundle
from seismoflux.background.grid import (
    EQUAL_AREA_CRS,
    GRID_CELL_SIZES_KM,
    build_clipped_grid,
    project_study_area_to_equal_area,
)
from seismoflux.data.common import canonical_json_bytes
from seismoflux.features.anomaly.engine import (
    Stage3FeatureEngine,
    Stage3IssueFeatureTable,
)
from seismoflux.features.anomaly.grid import Stage3QueryGrid
from seismoflux.features.anomaly.snapshot import Stage3IssueSnapshot

FeatureVariant = Literal["coverage_only", "snapshot", "dynamic"]
FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _string_sequence(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise TypeError(f"{label} must be a sequence")
    result = tuple(value)
    if not all(isinstance(item, str) and item for item in result):
        raise TypeError(f"{label} must contain non-empty strings")
    return cast(tuple[str, ...], result)


def _readonly(array: NDArray[np.generic]) -> NDArray[np.generic]:
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class Stage4IntegrationGrid:
    """One frozen-origin integration grid derived from the study area only."""

    grid_id: str
    equal_area_crs: str
    cell_size_km: float
    cell_ids: tuple[str, ...]
    rows: IntArray
    columns: IntArray
    query_xy_m: FloatArray
    clipped_area_km2: FloatArray

    def __post_init__(self) -> None:
        if self.equal_area_crs != EQUAL_AREA_CRS:
            raise ValueError("stage-4 integration CRS changed")
        if float(self.cell_size_km) not in GRID_CELL_SIZES_KM:
            raise ValueError("stage-4 grid must be exactly 50, 25, or 12.5 km")
        count = len(self.cell_ids)
        if not self.grid_id or len(set(self.cell_ids)) != count:
            raise ValueError("stage-4 grid identities must be non-empty and unique")
        if (
            self.rows.shape != (count,)
            or self.columns.shape != (count,)
            or self.query_xy_m.shape != (count, 2)
            or self.clipped_area_km2.shape != (count,)
        ):
            raise ValueError("stage-4 grid array shapes disagree")
        if count == 0 or not np.all(np.isfinite(self.query_xy_m)):
            raise ValueError("stage-4 grid must have finite query centers")
        if not np.all(np.isfinite(self.clipped_area_km2)) or np.any(self.clipped_area_km2 <= 0.0):
            raise ValueError("stage-4 clipped grid areas must be finite and positive")
        object.__setattr__(self, "rows", cast(IntArray, _readonly(self.rows)))
        object.__setattr__(self, "columns", cast(IntArray, _readonly(self.columns)))
        object.__setattr__(self, "query_xy_m", cast(FloatArray, _readonly(self.query_xy_m)))
        object.__setattr__(
            self,
            "clipped_area_km2",
            cast(FloatArray, _readonly(self.clipped_area_km2)),
        )

    @property
    def cell_count(self) -> int:
        return len(self.cell_ids)

    @property
    def total_area_km2(self) -> float:
        return float(np.sum(self.clipped_area_km2, dtype=np.float64))

    def as_stage3_query_grid(self) -> Stage3QueryGrid:
        """Adapt the accepted engine to a newly recomputed grid, never interpolate."""

        return Stage3QueryGrid(
            grid_id=self.grid_id,
            equal_area_crs=self.equal_area_crs,
            cell_size_km=self.cell_size_km,
            cell_ids=self.cell_ids,
            rows=self.rows,
            columns=self.columns,
            query_xy_m=self.query_xy_m,
            clipped_area_km2=self.clipped_area_km2,
        )


@dataclass(frozen=True, slots=True)
class Stage4GridFamily:
    coarse_50km: Stage4IntegrationGrid
    primary_25km: Stage4IntegrationGrid
    reference_12_5km: Stage4IntegrationGrid

    def __post_init__(self) -> None:
        if (
            self.coarse_50km.cell_size_km,
            self.primary_25km.cell_size_km,
            self.reference_12_5km.cell_size_km,
        ) != GRID_CELL_SIZES_KM:
            raise ValueError("stage-4 integration grid order changed")
        areas = np.asarray(
            [
                self.coarse_50km.total_area_km2,
                self.primary_25km.total_area_km2,
                self.reference_12_5km.total_area_km2,
            ],
            dtype=np.float64,
        )
        if float(np.max(areas) - np.min(areas)) > max(1e-6, float(np.max(areas)) * 1e-12):
            raise ValueError("stage-4 exact clipped grid areas disagree across resolutions")

    def grids(self) -> tuple[Stage4IntegrationGrid, ...]:
        return (self.coarse_50km, self.primary_25km, self.reference_12_5km)


def _grid_identity_payload(
    *,
    cell_size_km: float,
    cell_ids: tuple[str, ...],
    rows: IntArray,
    columns: IntArray,
    query_xy_m: FloatArray,
    clipped_area_km2: FloatArray,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "role": "stage4_target_independent_integration_grid",
        "equal_area_crs": EQUAL_AREA_CRS,
        "cell_size_km_hex": float(cell_size_km).hex(),
        "cells": [
            {
                "cell_id": cell_id,
                "row": int(row),
                "column": int(column),
                "query_x_m_hex": float(x_m).hex(),
                "query_y_m_hex": float(y_m).hex(),
                "clipped_area_km2_hex": float(area).hex(),
            }
            for cell_id, row, column, (x_m, y_m), area in zip(
                cell_ids,
                rows,
                columns,
                query_xy_m,
                clipped_area_km2,
                strict=True,
            )
        ],
    }


def build_stage4_integration_grid(
    study_area_wgs84: BaseGeometry,
    *,
    cell_size_km: float,
) -> Stage4IntegrationGrid:
    """Build one grid from the study area, with no target-dependent argument."""

    size = float(cell_size_km)
    if size not in GRID_CELL_SIZES_KM:
        raise ValueError("stage-4 grid must be exactly 50, 25, or 12.5 km")
    projected = project_study_area_to_equal_area(study_area_wgs84)
    grid = build_clipped_grid(projected, cell_size_km=size)
    cell_ids = grid.cell_ids
    rows = np.asarray([cell.row for cell in grid.cells], dtype=np.int64)
    columns = np.asarray([cell.column for cell in grid.cells], dtype=np.int64)
    query_xy_m = np.asarray(
        [(cell.representative_point.x, cell.representative_point.y) for cell in grid.cells],
        dtype=np.float64,
    )
    clipped_area_km2 = np.asarray(
        [cell.clipped_area_m2 / 1_000_000.0 for cell in grid.cells],
        dtype=np.float64,
    )
    payload = _grid_identity_payload(
        cell_size_km=size,
        cell_ids=cell_ids,
        rows=rows,
        columns=columns,
        query_xy_m=query_xy_m,
        clipped_area_km2=clipped_area_km2,
    )
    return Stage4IntegrationGrid(
        grid_id=hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
        equal_area_crs=EQUAL_AREA_CRS,
        cell_size_km=size,
        cell_ids=cell_ids,
        rows=rows,
        columns=columns,
        query_xy_m=query_xy_m,
        clipped_area_km2=clipped_area_km2,
    )


def build_stage4_grid_family(study_area_wgs84: BaseGeometry) -> Stage4GridFamily:
    """Build all three frozen grids from the same target-independent geometry."""

    grids = tuple(
        build_stage4_integration_grid(study_area_wgs84, cell_size_km=size)
        for size in GRID_CELL_SIZES_KM
    )
    return Stage4GridFamily(*grids)


def recompute_stage4_features(
    snapshots: tuple[Stage3IssueSnapshot, ...],
    grid: Stage4IntegrationGrid,
    *,
    query_chunk_size: int = 256,
    spatial_workers: int = 1,
) -> tuple[Stage3IssueFeatureTable, ...]:
    """Recompute accepted feature definitions at this grid's cell centers."""

    engine = Stage3FeatureEngine(
        snapshots,
        grid.as_stage3_query_grid(),
        query_chunk_size=query_chunk_size,
        spatial_workers=spatial_workers,
    )
    return tuple(engine.build_next_issue() for _ in snapshots)


def source_columns_for_variant(
    protocol: Stage4ProtocolBundle,
    variant: FeatureVariant,
) -> tuple[str, ...]:
    feature_sets = _mapping(protocol.feature_set.document.get("feature_sets"), label="feature_sets")
    definition = _mapping(feature_sets.get(variant), label=f"feature_sets.{variant}")
    source_columns = _string_sequence(
        definition.get("source_value_columns"),
        label=f"feature_sets.{variant}.source_value_columns",
    )
    audit_only = set(
        _string_sequence(
            definition.get("audit_only_quality_columns"),
            label=f"feature_sets.{variant}.audit_only_quality_columns",
        )
    )
    if set(source_columns) & audit_only:
        raise ValueError("audit-only quality columns escaped into the design sources")
    if len(set(source_columns)) != len(source_columns):
        raise ValueError("stage-4 source columns are duplicated")
    return source_columns


@dataclass(frozen=True, slots=True)
class RawFeatureMatrix:
    """Raw selected feature values and exact null mask before preprocessing."""

    source_columns: tuple[str, ...]
    values: FloatArray
    missing: BoolArray
    positive_area: BoolArray

    def __post_init__(self) -> None:
        rows, columns = self.values.shape
        if columns != len(self.source_columns) or self.missing.shape != (rows, columns):
            raise ValueError("raw stage-4 feature matrix shapes disagree")
        if self.positive_area.shape != (rows,) or not np.all(self.positive_area):
            raise ValueError("stage-4 fitting rows must all have positive clipped area")
        if np.any(self.missing != ~np.isfinite(self.values)):
            raise ValueError("raw stage-4 missing mask differs from nonfinite values")
        object.__setattr__(self, "values", cast(FloatArray, _readonly(self.values)))
        object.__setattr__(self, "missing", cast(BoolArray, _readonly(self.missing)))
        object.__setattr__(
            self,
            "positive_area",
            cast(BoolArray, _readonly(self.positive_area)),
        )


def extract_raw_feature_matrix(
    table: pa.Table,
    *,
    source_columns: Sequence[str],
) -> RawFeatureMatrix:
    """Extract frozen source columns; quality companions cannot be requested here."""

    names = tuple(source_columns)
    if not names or len(set(names)) != len(names):
        raise ValueError("source column selection must be non-empty and unique")
    missing_columns = sorted(set((*names, "clipped_area_km2")) - set(table.column_names))
    if missing_columns:
        raise ValueError(f"stage-4 feature table is missing columns: {missing_columns}")
    columns: list[FloatArray] = []
    masks: list[BoolArray] = []
    for name in names:
        array = table[name].combine_chunks().cast(pa.float64())
        values = np.asarray(array.to_numpy(zero_copy_only=False), dtype=np.float64)
        missing = np.asarray(array.is_null().to_numpy(zero_copy_only=False), dtype=np.bool_)
        values = values.copy()
        values[missing] = np.nan
        columns.append(values)
        masks.append(missing)
    matrix = np.column_stack(columns).astype(np.float64, copy=False)
    missing_matrix = np.column_stack(masks).astype(np.bool_, copy=False)
    areas = np.asarray(
        table["clipped_area_km2"].combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.float64,
    )
    positive_area = np.asarray(np.isfinite(areas) & (areas > 0.0), dtype=np.bool_)
    if not np.all(positive_area):
        raise ValueError("stage-4 feature tables may contain only positive-area grid rows")
    return RawFeatureMatrix(
        source_columns=names,
        values=matrix,
        missing=missing_matrix,
        positive_area=positive_area,
    )


def selected_table_identity_sha256(table: pa.Table, columns: Sequence[str]) -> str:
    """Hash Arrow values and null bitmaps in a stable IPC stream."""

    names = tuple(columns)
    if not names or len(set(names)) != len(names):
        raise ValueError("identity columns must be non-empty and unique")
    if missing := sorted(set(names) - set(table.column_names)):
        raise ValueError(f"identity table is missing columns: {missing}")
    selected = table.select(list(names)).combine_chunks()
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, selected.schema) as writer:
        writer.write_table(selected)
    return hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest()


def assert_selected_columns_exact(
    accepted: pa.Table,
    recomputed: pa.Table,
    *,
    columns: Sequence[str],
) -> str:
    """Fail unless values and Arrow validity bitmaps are exactly identical."""

    accepted_hash = selected_table_identity_sha256(accepted, columns)
    recomputed_hash = selected_table_identity_sha256(recomputed, columns)
    if accepted_hash != recomputed_hash:
        raise ValueError("identity reconstruction differs from the accepted stage-3 columns")
    return accepted_hash


def relative_total_intensity_error(candidate: float, reference: float) -> float:
    """Frozen spatial/time integration convergence denominator."""

    left = float(candidate)
    right = float(reference)
    if not math.isfinite(left) or not math.isfinite(right):
        raise ValueError("integrated intensities must be finite")
    return abs(left - right) / max(abs(right), 1.0e-12)


__all__ = [
    "FeatureVariant",
    "RawFeatureMatrix",
    "Stage4GridFamily",
    "Stage4IntegrationGrid",
    "assert_selected_columns_exact",
    "build_stage4_grid_family",
    "build_stage4_integration_grid",
    "extract_raw_feature_matrix",
    "recompute_stage4_features",
    "relative_total_intensity_error",
    "selected_table_identity_sha256",
    "source_columns_for_variant",
]
