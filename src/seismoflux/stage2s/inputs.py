"""Byte-only Stage 2S input adapters and the non-target spatial preflight."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray
from shapely import normalize as normalize_geometry
from shapely import to_wkb

from seismoflux.background.artifacts import canonical_json_bytes
from seismoflux.background.catalog import StudyArea, load_study_area_bytes
from seismoflux.background.grid import (
    EQUAL_AREA_CRS,
    EqualAreaGrid,
    EqualAreaGridFamily,
    build_equal_area_grid_family,
    cell_id,
)
from seismoflux.features.anomaly.grid import Stage3QueryGrid, build_stage3_query_grid
from seismoflux.stage2s.contracts import SpatialGrid, SpatialQuadratureFamily
from seismoflux.stage2s.protocol import Stage2SProtocolBundle

EXPECTED_MAPPING_FIELDS = (
    ("grid_id", pa.string()),
    ("cell_id", pa.string()),
    ("cell_row", pa.int64()),
    ("cell_column", pa.int64()),
    ("query_x_m", pa.float64()),
    ("query_y_m", pa.float64()),
    ("construction_zone_id", pa.string()),
)
GRID_LAYER_ORDER = ("50", "25", "12.5")
PARENT_RELATION_ORDER = ("25_to_50", "12.5_to_25")
REPRESENTATIVE_POINT_ALGORITHM = "shapely.point_on_surface_of_exact_support_clipped_geometry"


class Stage2SInputError(RuntimeError):
    """Raised when a byte source or target-independent spatial identity is invalid."""


def _readonly(array: object, *, dtype: np.dtype[Any]) -> NDArray[Any]:
    result = np.array(array, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _projected_geometry_sha256(study_area: StudyArea) -> str:
    normalized = normalize_geometry(study_area.projected)
    payload = cast(
        bytes,
        to_wkb(
            normalized,
            hex=False,
            output_dimension=2,
            byte_order=1,
            include_srid=False,
        ),
    )
    return _sha256(payload)


@dataclass(frozen=True, slots=True)
class Stage2SQueryGrid:
    """The complete local-only 25 km grid required by the Stage 2S adapter."""

    grid_id: str
    equal_area_crs: str
    cell_size_km: float
    cell_ids: tuple[str, ...]
    rows: NDArray[np.int64]
    columns: NDArray[np.int64]
    query_xy_m: NDArray[np.float64]
    clipped_area_km2: NDArray[np.float64]

    def __post_init__(self) -> None:
        if not self.grid_id or self.equal_area_crs != EQUAL_AREA_CRS:
            raise Stage2SInputError("query grid identity or CRS is invalid")
        if float(self.cell_size_km) != 25.0:
            raise Stage2SInputError("query grid must use the frozen 25 km cell size")
        cell_ids = tuple(self.cell_ids)
        rows = cast(NDArray[np.int64], _readonly(self.rows, dtype=np.dtype(np.int64)))
        columns = cast(
            NDArray[np.int64],
            _readonly(self.columns, dtype=np.dtype(np.int64)),
        )
        xy = cast(
            NDArray[np.float64],
            _readonly(self.query_xy_m, dtype=np.dtype(np.float64)),
        )
        areas = cast(
            NDArray[np.float64],
            _readonly(self.clipped_area_km2, dtype=np.dtype(np.float64)),
        )
        length = len(cell_ids)
        if not cell_ids or len(set(cell_ids)) != length:
            raise Stage2SInputError("query grid cell IDs must be non-empty and unique")
        if rows.shape != (length,) or columns.shape != (length,):
            raise Stage2SInputError("query grid row/column arrays have the wrong shape")
        if xy.shape != (length, 2) or areas.shape != (length,):
            raise Stage2SInputError("query grid coordinates/areas have the wrong shape")
        if not np.isfinite(xy).all() or not np.isfinite(areas).all() or np.any(areas <= 0.0):
            raise Stage2SInputError("query grid coordinates and areas must be finite")
        order = tuple((int(row), int(column)) for row, column in zip(rows, columns, strict=True))
        if order != tuple(sorted(order)) or len(set(order)) != length:
            raise Stage2SInputError("query grid rows/columns must be unique and ascending")
        object.__setattr__(self, "cell_size_km", 25.0)
        object.__setattr__(self, "cell_ids", cell_ids)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "query_xy_m", xy)
        object.__setattr__(self, "clipped_area_km2", areas)

    @property
    def cell_count(self) -> int:
        return len(self.cell_ids)


@dataclass(frozen=True, slots=True)
class NonTargetSpatialAdapter:
    """The indivisible query-grid plus cell-zone mapping interface."""

    query_grid: Stage2SQueryGrid
    construction_zone_id_by_cell_id: MappingProxyType[str, str]
    grid_family: EqualAreaGridFamily

    def __post_init__(self) -> None:
        mapping = dict(self.construction_zone_id_by_cell_id)
        if set(mapping) != set(self.query_grid.cell_ids):
            raise Stage2SInputError("cell-zone mapping must cover every query cell exactly once")
        if any(not value for value in mapping.values()):
            raise Stage2SInputError("construction zone IDs must be non-empty")
        object.__setattr__(
            self,
            "construction_zone_id_by_cell_id",
            MappingProxyType(mapping),
        )


@dataclass(frozen=True, slots=True)
class NonTargetExpectations:
    study_area_file_sha256: str
    projected_geometry_sha256: str
    projected_area_m2: float
    projected_area_absolute_tolerance_m2: float
    mapping_file_sha256: str
    mapping_row_count: int
    zone_count: int
    expected_grid_id: str | None = None


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


@dataclass(frozen=True, slots=True)
class SpatialGridLayerIdentity:
    """Compact content identity for one target-independent grid layer."""

    layer: str
    cell_size_km: float
    grid_id: str
    operational_cell_identity_sha256: str
    cell_count: int
    total_clipped_area_km2: float
    ordered_clipped_area_sha256: str
    representative_point_count: int
    representative_point_sha256: str

    def __post_init__(self) -> None:
        if self.layer not in GRID_LAYER_ORDER:
            raise Stage2SInputError("spatial grid layer is not one of 50/25/12.5 km")
        expected_size = {"50": 50.0, "25": 25.0, "12.5": 12.5}[self.layer]
        if float(self.cell_size_km) != expected_size:
            raise Stage2SInputError("spatial grid layer size differs from its label")
        if (
            not _valid_sha256(self.grid_id)
            or not _valid_sha256(self.operational_cell_identity_sha256)
            or not _valid_sha256(self.ordered_clipped_area_sha256)
            or not _valid_sha256(self.representative_point_sha256)
        ):
            raise Stage2SInputError("spatial grid identity contains an invalid SHA-256")
        if (
            isinstance(self.cell_count, bool)
            or self.cell_count <= 0
            or self.representative_point_count != self.cell_count
        ):
            raise Stage2SInputError("spatial grid cell/representative-point counts differ")
        area = float(self.total_clipped_area_km2)
        if not math.isfinite(area) or area <= 0.0:
            raise Stage2SInputError("spatial grid total clipped area must be finite and positive")
        object.__setattr__(self, "cell_size_km", expected_size)
        object.__setattr__(self, "total_clipped_area_km2", area)

    def as_mapping(self) -> dict[str, object]:
        return {
            "cell_size_km": self.cell_size_km,
            "grid_id": self.grid_id,
            "operational_cell_identity_sha256": self.operational_cell_identity_sha256,
            "cell_count": self.cell_count,
            "total_clipped_area_km2": self.total_clipped_area_km2,
            "ordered_clipped_area_sha256": self.ordered_clipped_area_sha256,
            "representative_point_algorithm": REPRESENTATIVE_POINT_ALGORITHM,
            "representative_point_count": self.representative_point_count,
            "representative_point_sha256": self.representative_point_sha256,
        }


@dataclass(frozen=True, slots=True)
class SpatialParentRelationIdentity:
    """Exact aligned fine-to-coarse parent and area-closure identity."""

    relation: str
    fine_grid_id: str
    coarse_grid_id: str
    aligned_size_ratio: int
    fine_cell_count: int
    coarse_cell_count: int
    mapped_fine_cell_count: int
    covered_coarse_parent_count: int
    parent_mapping_sha256: str
    parent_area_relation_sha256: str
    maximum_parent_area_absolute_error_km2: float
    total_parent_area_absolute_error_km2: float
    parent_area_absolute_tolerance_km2: float

    def __post_init__(self) -> None:
        if self.relation not in PARENT_RELATION_ORDER:
            raise Stage2SInputError("unknown aligned grid parent relation")
        if (
            not _valid_sha256(self.fine_grid_id)
            or not _valid_sha256(self.coarse_grid_id)
            or not _valid_sha256(self.parent_mapping_sha256)
            or not _valid_sha256(self.parent_area_relation_sha256)
        ):
            raise Stage2SInputError("parent relation contains an invalid SHA-256")
        if self.aligned_size_ratio != 2:
            raise Stage2SInputError("aligned Stage2S parent ratio must be exactly two")
        if (
            any(
                isinstance(value, bool) or value <= 0
                for value in (self.fine_cell_count, self.coarse_cell_count)
            )
            or self.mapped_fine_cell_count != self.fine_cell_count
            or self.covered_coarse_parent_count != self.coarse_cell_count
        ):
            raise Stage2SInputError("aligned parent relation does not cover every grid cell")
        maximum_error = float(self.maximum_parent_area_absolute_error_km2)
        total_error = float(self.total_parent_area_absolute_error_km2)
        tolerance = float(self.parent_area_absolute_tolerance_km2)
        if (
            not all(math.isfinite(value) for value in (maximum_error, total_error, tolerance))
            or min(maximum_error, total_error, tolerance) < 0.0
        ):
            raise Stage2SInputError("parent area relation errors/tolerance are invalid")
        if maximum_error > tolerance:
            raise Stage2SInputError("a parent cell area does not close within tolerance")
        object.__setattr__(
            self,
            "maximum_parent_area_absolute_error_km2",
            maximum_error,
        )
        object.__setattr__(
            self,
            "total_parent_area_absolute_error_km2",
            total_error,
        )
        object.__setattr__(
            self,
            "parent_area_absolute_tolerance_km2",
            tolerance,
        )

    def as_mapping(self) -> dict[str, object]:
        return {
            "fine_grid_id": self.fine_grid_id,
            "coarse_grid_id": self.coarse_grid_id,
            "aligned_size_ratio": self.aligned_size_ratio,
            "fine_cell_count": self.fine_cell_count,
            "coarse_cell_count": self.coarse_cell_count,
            "mapped_fine_cell_count": self.mapped_fine_cell_count,
            "covered_coarse_parent_count": self.covered_coarse_parent_count,
            "parent_mapping_sha256": self.parent_mapping_sha256,
            "parent_area_relation_sha256": self.parent_area_relation_sha256,
            "maximum_parent_area_absolute_error_km2": (self.maximum_parent_area_absolute_error_km2),
            "total_parent_area_absolute_error_km2": (self.total_parent_area_absolute_error_km2),
            "parent_area_absolute_tolerance_km2": (self.parent_area_absolute_tolerance_km2),
        }


@dataclass(frozen=True, slots=True)
class NonTargetSpatialIdentity:
    """Canonical aggregate binding for all three aligned spatial grids."""

    layers: tuple[SpatialGridLayerIdentity, ...]
    parent_relations: tuple[SpatialParentRelationIdentity, ...]
    identity_sha256: str

    def __post_init__(self) -> None:
        if tuple(layer.layer for layer in self.layers) != GRID_LAYER_ORDER:
            raise Stage2SInputError("spatial identity layers must be ordered 50/25/12.5 km")
        if tuple(relation.relation for relation in self.parent_relations) != (
            PARENT_RELATION_ORDER
        ):
            raise Stage2SInputError("spatial parent relations must be ordered 25→50 then 12.5→25")
        expected = _sha256(canonical_json_bytes(self.payload_mapping()))
        if self.identity_sha256 != expected:
            raise Stage2SInputError("aggregate spatial identity SHA-256 differs from its payload")

    def payload_mapping(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "equal_area_crs": EQUAL_AREA_CRS,
            "layer_order": list(GRID_LAYER_ORDER),
            "layers": {layer.layer: layer.as_mapping() for layer in self.layers},
            "parent_relation_order": list(PARENT_RELATION_ORDER),
            "parent_relations": {
                relation.relation: relation.as_mapping() for relation in self.parent_relations
            },
            "representative_point_algorithm": REPRESENTATIVE_POINT_ALGORITHM,
            "target_or_score_input_count": 0,
        }

    def as_mapping(self) -> dict[str, object]:
        return {
            **self.payload_mapping(),
            "identity_sha256": self.identity_sha256,
        }


@dataclass(frozen=True, slots=True)
class NonTargetPreflight:
    adapter: NonTargetSpatialAdapter
    study_area_file_sha256: str
    projected_geometry_sha256: str
    projected_area_m2: float
    mapping_file_sha256: str
    mapping_schema_sha256: str
    zone_ids: tuple[str, ...]
    grid_cell_counts: tuple[int, int, int]
    grid_total_area_km2: tuple[float, float, float]
    spatial_identity: NonTargetSpatialIdentity | None = None

    def __post_init__(self) -> None:
        tolerance = (
            max(1.0e-10, abs(float(self.projected_area_m2)) * 1.0e-18)
            if self.spatial_identity is None
            else self.spatial_identity.parent_relations[0].parent_area_absolute_tolerance_km2
        )
        expected = _build_non_target_spatial_identity(
            self.adapter,
            area_tolerance_km2=tolerance,
        )
        if self.spatial_identity is not None and self.spatial_identity != expected:
            raise Stage2SInputError(
                "preflight spatial identity differs from the reconstructed adapter"
            )
        expected_counts = tuple(layer.cell_count for layer in expected.layers)
        expected_areas = tuple(layer.total_clipped_area_km2 for layer in expected.layers)
        if self.grid_cell_counts != expected_counts or self.grid_total_area_km2 != expected_areas:
            raise Stage2SInputError("preflight grid summaries differ from spatial identity")
        object.__setattr__(self, "spatial_identity", expected)

    def receipt_bindings(self) -> dict[str, object]:
        if self.spatial_identity is None:
            raise Stage2SInputError("preflight spatial identity is unavailable")
        return {
            "study_area_file_sha256": self.study_area_file_sha256,
            "projected_geometry_sha256": self.projected_geometry_sha256,
            "projected_area_m2": self.projected_area_m2,
            "cell_mapping_file_sha256": self.mapping_file_sha256,
            "cell_mapping_schema_sha256": self.mapping_schema_sha256,
            "query_grid_id": self.adapter.query_grid.grid_id,
            "query_grid_cell_count": self.adapter.query_grid.cell_count,
            "construction_zone_ids": list(self.zone_ids),
            "grid_cell_counts_50_25_12_5km": list(self.grid_cell_counts),
            "grid_total_area_km2_50_25_12_5km": list(self.grid_total_area_km2),
            "aligned_grid_identity": self.spatial_identity.as_mapping(),
            "earthquake_catalog_bytes_read": False,
            "assessment_target_view_read": False,
            "score_or_candidate_metric_read": False,
        }


def validated_non_target_preflight_receipt_bindings(
    preflight: NonTargetPreflight,
) -> dict[str, object]:
    """Return receipt bindings only after the frozen field family closes."""

    bindings = preflight.receipt_bindings()
    expected_fields = {
        "study_area_file_sha256",
        "projected_geometry_sha256",
        "projected_area_m2",
        "cell_mapping_file_sha256",
        "cell_mapping_schema_sha256",
        "query_grid_id",
        "query_grid_cell_count",
        "construction_zone_ids",
        "grid_cell_counts_50_25_12_5km",
        "grid_total_area_km2_50_25_12_5km",
        "aligned_grid_identity",
        "earthquake_catalog_bytes_read",
        "assessment_target_view_read",
        "score_or_candidate_metric_read",
    }
    if set(bindings) != expected_fields:
        raise Stage2SInputError("non-target preflight receipt fields are incomplete")
    if (
        bindings["query_grid_id"] != preflight.adapter.query_grid.grid_id
        or bindings["query_grid_cell_count"] != preflight.adapter.query_grid.cell_count
        or bindings["aligned_grid_identity"]
        != cast(NonTargetSpatialIdentity, preflight.spatial_identity).as_mapping()
        or any(
            bindings[key] is not False
            for key in (
                "earthquake_catalog_bytes_read",
                "assessment_target_view_read",
                "score_or_candidate_metric_read",
            )
        )
    ):
        raise Stage2SInputError("non-target preflight receipt binding validation failed")
    return bindings


def _query_grid(value: Stage3QueryGrid) -> Stage2SQueryGrid:
    return Stage2SQueryGrid(
        grid_id=value.grid_id,
        equal_area_crs=value.equal_area_crs,
        cell_size_km=value.cell_size_km,
        cell_ids=value.cell_ids,
        rows=value.rows,
        columns=value.columns,
        query_xy_m=value.query_xy_m,
        clipped_area_km2=value.clipped_area_km2,
    )


def _stage2s_grid_id(
    *,
    cell_size_km: float,
    cell_ids: tuple[str, ...],
    rows: NDArray[np.int64],
    columns: NDArray[np.int64],
    query_xy_m: NDArray[np.float64],
    clipped_area_km2: NDArray[np.float64],
) -> str:
    return _sha256(
        canonical_json_bytes(
            {
                "schema_version": 1,
                "role": "stage2s_target_independent_spatial_quadrature",
                "equal_area_crs": EQUAL_AREA_CRS,
                "cell_size_km": cell_size_km,
                "cells": [
                    {
                        "cell_id": identifier,
                        "row": int(row),
                        "column": int(column),
                        "query_x_m_hex": float(x_m).hex(),
                        "query_y_m_hex": float(y_m).hex(),
                        "clipped_area_km2_hex": float(area).hex(),
                    }
                    for identifier, row, column, (x_m, y_m), area in zip(
                        cell_ids,
                        rows,
                        columns,
                        query_xy_m,
                        clipped_area_km2,
                        strict=True,
                    )
                ],
            }
        )
    )


def _grid_arrays(
    grid: EqualAreaGrid,
) -> tuple[
    tuple[str, ...],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    return (
        grid.cell_ids,
        np.asarray([cell.row for cell in grid.cells], dtype=np.int64),
        np.asarray([cell.column for cell in grid.cells], dtype=np.int64),
        np.asarray(
            [(cell.representative_point.x, cell.representative_point.y) for cell in grid.cells],
            dtype=np.float64,
        ),
        np.asarray(
            [cell.clipped_area_m2 / 1_000_000.0 for cell in grid.cells],
            dtype=np.float64,
        ),
    )


def _grid_layer_identity(
    *,
    layer: str,
    grid: EqualAreaGrid,
    public_grid_id: str | None,
) -> SpatialGridLayerIdentity:
    cell_ids, rows, columns, query_xy_m, areas = _grid_arrays(grid)
    operational_identity = _stage2s_grid_id(
        cell_size_km=grid.spec.cell_size_km,
        cell_ids=cell_ids,
        rows=rows,
        columns=columns,
        query_xy_m=query_xy_m,
        clipped_area_km2=areas,
    )
    area_sha256 = _sha256(
        canonical_json_bytes(
            {
                "cell_size_km": grid.spec.cell_size_km,
                "ordered_cell_area": [
                    {
                        "cell_id": identifier,
                        "clipped_area_km2_hex": float(area).hex(),
                    }
                    for identifier, area in zip(cell_ids, areas, strict=True)
                ],
            }
        )
    )
    representative_point_sha256 = _sha256(
        canonical_json_bytes(
            {
                "algorithm": REPRESENTATIVE_POINT_ALGORITHM,
                "cell_size_km": grid.spec.cell_size_km,
                "ordered_cell_representative_point": [
                    {
                        "cell_id": identifier,
                        "query_x_m_hex": float(x_m).hex(),
                        "query_y_m_hex": float(y_m).hex(),
                    }
                    for identifier, (x_m, y_m) in zip(
                        cell_ids,
                        query_xy_m,
                        strict=True,
                    )
                ],
            }
        )
    )
    return SpatialGridLayerIdentity(
        layer=layer,
        cell_size_km=grid.spec.cell_size_km,
        grid_id=operational_identity if public_grid_id is None else public_grid_id,
        operational_cell_identity_sha256=operational_identity,
        cell_count=len(cell_ids),
        total_clipped_area_km2=math.fsum(float(value) for value in areas),
        ordered_clipped_area_sha256=area_sha256,
        representative_point_count=len(cell_ids),
        representative_point_sha256=representative_point_sha256,
    )


def _parent_relation_identity(
    *,
    relation: str,
    fine: EqualAreaGrid,
    coarse: EqualAreaGrid,
    fine_grid_id: str,
    coarse_grid_id: str,
    area_tolerance_km2: float,
) -> SpatialParentRelationIdentity:
    ratio = coarse.spec.cell_size_mm // fine.spec.cell_size_mm
    if ratio != 2 or coarse.spec.cell_size_mm % fine.spec.cell_size_mm:
        raise Stage2SInputError("grid family is not aligned by the frozen parent ratio")
    coarse_area = {cell.id: cell.clipped_area_m2 / 1_000_000.0 for cell in coarse.cells}
    child_areas: dict[str, list[float]] = {identifier: [] for identifier in coarse.cell_ids}
    mapping_rows: list[dict[str, str]] = []
    for fine_cell in fine.cells:
        parent_id = cell_id(
            coarse.spec,
            row=fine_cell.row // ratio,
            column=fine_cell.column // ratio,
        )
        try:
            child_areas[parent_id].append(fine_cell.clipped_area_m2 / 1_000_000.0)
        except KeyError as exc:
            raise Stage2SInputError("fine grid cell has no positive-area aligned parent") from exc
        mapping_rows.append(
            {
                "fine_cell_id": fine_cell.id,
                "coarse_parent_cell_id": parent_id,
            }
        )
    if any(not values for values in child_areas.values()):
        raise Stage2SInputError("one or more coarse cells have no positive-area fine children")
    area_rows: list[dict[str, str]] = []
    errors: list[float] = []
    for parent_id in coarse.cell_ids:
        fine_area = math.fsum(child_areas[parent_id])
        parent_area = coarse_area[parent_id]
        error = abs(fine_area - parent_area)
        errors.append(error)
        area_rows.append(
            {
                "coarse_parent_cell_id": parent_id,
                "coarse_clipped_area_km2_hex": float(parent_area).hex(),
                "fine_children_area_sum_km2_hex": float(fine_area).hex(),
                "absolute_error_km2_hex": float(error).hex(),
            }
        )
    maximum_error = max(errors)
    if maximum_error > area_tolerance_km2:
        raise Stage2SInputError("aligned fine-to-coarse parent areas do not close")
    return SpatialParentRelationIdentity(
        relation=relation,
        fine_grid_id=fine_grid_id,
        coarse_grid_id=coarse_grid_id,
        aligned_size_ratio=ratio,
        fine_cell_count=len(fine.cells),
        coarse_cell_count=len(coarse.cells),
        mapped_fine_cell_count=len(mapping_rows),
        covered_coarse_parent_count=len(child_areas),
        parent_mapping_sha256=_sha256(canonical_json_bytes(mapping_rows)),
        parent_area_relation_sha256=_sha256(canonical_json_bytes(area_rows)),
        maximum_parent_area_absolute_error_km2=maximum_error,
        total_parent_area_absolute_error_km2=math.fsum(errors),
        parent_area_absolute_tolerance_km2=area_tolerance_km2,
    )


def _build_non_target_spatial_identity(
    adapter: NonTargetSpatialAdapter,
    *,
    area_tolerance_km2: float,
) -> NonTargetSpatialIdentity:
    tolerance = float(area_tolerance_km2)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise Stage2SInputError("spatial parent area tolerance is invalid")
    grids = {
        "50": adapter.grid_family.at(50.0),
        "25": adapter.grid_family.at(25.0),
        "12.5": adapter.grid_family.at(12.5),
    }
    layers = tuple(
        _grid_layer_identity(
            layer=layer,
            grid=grids[layer],
            public_grid_id=(adapter.query_grid.grid_id if layer == "25" else None),
        )
        for layer in GRID_LAYER_ORDER
    )
    layer_by_name = {layer.layer: layer for layer in layers}
    parent_relations = (
        _parent_relation_identity(
            relation="25_to_50",
            fine=grids["25"],
            coarse=grids["50"],
            fine_grid_id=layer_by_name["25"].grid_id,
            coarse_grid_id=layer_by_name["50"].grid_id,
            area_tolerance_km2=tolerance,
        ),
        _parent_relation_identity(
            relation="12.5_to_25",
            fine=grids["12.5"],
            coarse=grids["25"],
            fine_grid_id=layer_by_name["12.5"].grid_id,
            coarse_grid_id=layer_by_name["25"].grid_id,
            area_tolerance_km2=tolerance,
        ),
    )
    payload = {
        "schema_version": 1,
        "equal_area_crs": EQUAL_AREA_CRS,
        "layer_order": list(GRID_LAYER_ORDER),
        "layers": {layer.layer: layer.as_mapping() for layer in layers},
        "parent_relation_order": list(PARENT_RELATION_ORDER),
        "parent_relations": {
            relation.relation: relation.as_mapping() for relation in parent_relations
        },
        "representative_point_algorithm": REPRESENTATIVE_POINT_ALGORITHM,
        "target_or_score_input_count": 0,
    }
    return NonTargetSpatialIdentity(
        layers=cast(
            tuple[
                SpatialGridLayerIdentity,
                SpatialGridLayerIdentity,
                SpatialGridLayerIdentity,
            ],
            layers,
        ),
        parent_relations=parent_relations,
        identity_sha256=_sha256(canonical_json_bytes(payload)),
    )


def to_spatial_quadrature_family(
    adapter: NonTargetSpatialAdapter,
) -> SpatialQuadratureFamily:
    """Convert the verified grid family to immutable Stage 2S numerical contracts."""

    converted: list[SpatialGrid] = []
    for cell_size_km in (50.0, 25.0, 12.5):
        source = adapter.grid_family.at(cell_size_km)
        cell_ids = source.cell_ids
        rows = np.asarray([cell.row for cell in source.cells], dtype=np.int64)
        columns = np.asarray([cell.column for cell in source.cells], dtype=np.int64)
        query_xy_m = np.asarray(
            [(cell.representative_point.x, cell.representative_point.y) for cell in source.cells],
            dtype=np.float64,
        )
        areas = np.asarray(
            [cell.clipped_area_m2 / 1_000_000.0 for cell in source.cells],
            dtype=np.float64,
        )
        grid_id = (
            adapter.query_grid.grid_id
            if cell_size_km == 25.0
            else _stage2s_grid_id(
                cell_size_km=cell_size_km,
                cell_ids=cell_ids,
                rows=rows,
                columns=columns,
                query_xy_m=query_xy_m,
                clipped_area_km2=areas,
            )
        )
        converted.append(
            SpatialGrid(
                grid_id=grid_id,
                cell_size_km=cell_size_km,
                cell_ids=cell_ids,
                rows=rows,
                columns=columns,
                query_xy_km=query_xy_m / 1_000.0,
                clipped_area_km2=areas,
            )
        )
    return SpatialQuadratureFamily(
        grids=cast(tuple[SpatialGrid, SpatialGrid, SpatialGrid], tuple(converted))
    )


def _mapping_schema_sha256(schema: pa.Schema) -> str:
    fields = [
        {
            "name": field.name,
            "type": str(field.type),
            "nullable": field.nullable,
        }
        for field in schema
    ]
    import json

    return _sha256(
        json.dumps(
            fields,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _parse_mapping(
    mapping_bytes: bytes,
    *,
    query_grid: Stage2SQueryGrid,
    expected_rows: int,
    expected_zones: int,
) -> tuple[MappingProxyType[str, str], str, tuple[str, ...]]:
    if not mapping_bytes:
        raise Stage2SInputError("cell mapping bytes must be non-empty")
    try:
        table = pq.read_table(
            pa.BufferReader(mapping_bytes),
            use_threads=False,
        )
    except (pa.ArrowException, OSError) as exc:
        raise Stage2SInputError("cell mapping is not a valid in-memory Parquet table") from exc
    observed_fields = tuple((field.name, field.type) for field in table.schema)
    if observed_fields != EXPECTED_MAPPING_FIELDS:
        raise Stage2SInputError("cell mapping schema fields or order changed")
    if table.num_rows != expected_rows or table.num_rows != query_grid.cell_count:
        raise Stage2SInputError("cell mapping row count does not match the query grid")
    if any(table[column].null_count for column in table.column_names):
        raise Stage2SInputError("cell mapping must not contain nulls")

    grid_ids = tuple(cast(str, value) for value in table["grid_id"].to_pylist())
    cell_ids = tuple(cast(str, value) for value in table["cell_id"].to_pylist())
    rows = table["cell_row"].combine_chunks().to_numpy(zero_copy_only=False)
    columns = table["cell_column"].combine_chunks().to_numpy(zero_copy_only=False)
    x_m = table["query_x_m"].combine_chunks().to_numpy(zero_copy_only=False)
    y_m = table["query_y_m"].combine_chunks().to_numpy(zero_copy_only=False)
    zone_ids = tuple(cast(str, value) for value in table["construction_zone_id"].to_pylist())
    if set(grid_ids) != {query_grid.grid_id}:
        raise Stage2SInputError("cell mapping grid_id differs from the runtime query grid")
    if cell_ids != query_grid.cell_ids:
        raise Stage2SInputError("cell mapping order/IDs differ from the runtime query grid")
    if len(set(cell_ids)) != len(cell_ids):
        raise Stage2SInputError("cell mapping cell IDs are not unique")
    if not np.array_equal(rows, query_grid.rows) or not np.array_equal(
        columns,
        query_grid.columns,
    ):
        raise Stage2SInputError("cell mapping rows/columns differ from the runtime grid")
    if not np.array_equal(x_m, query_grid.query_xy_m[:, 0]) or not np.array_equal(
        y_m,
        query_grid.query_xy_m[:, 1],
    ):
        raise Stage2SInputError("cell mapping coordinates differ bitwise from the runtime grid")
    unique_zones = tuple(sorted(set(zone_ids), key=lambda value: value.encode("utf-8")))
    if len(unique_zones) != expected_zones or any(not value for value in unique_zones):
        raise Stage2SInputError("cell mapping does not contain the frozen non-empty zone count")
    return (
        MappingProxyType(dict(zip(cell_ids, zone_ids, strict=True))),
        _mapping_schema_sha256(table.schema),
        unique_zones,
    )


def build_non_target_adapter_from_bytes(
    *,
    study_area_bytes: bytes,
    mapping_bytes: bytes,
    expectations: NonTargetExpectations,
) -> NonTargetPreflight:
    """Build and verify the complete non-target adapter from two immutable byte sources."""

    observed_study_hash = _sha256(study_area_bytes)
    observed_mapping_hash = _sha256(mapping_bytes)
    if observed_study_hash != expectations.study_area_file_sha256:
        raise Stage2SInputError("study area file hash changed")
    if observed_mapping_hash != expectations.mapping_file_sha256:
        raise Stage2SInputError("cell mapping file hash changed")
    study_area = load_study_area_bytes(study_area_bytes, EQUAL_AREA_CRS)
    geometry_hash = _projected_geometry_sha256(study_area)
    if geometry_hash != expectations.projected_geometry_sha256:
        raise Stage2SInputError("projected study-area geometry identity changed")
    area_m2 = float(study_area.projected.area)
    if not math.isclose(
        area_m2,
        expectations.projected_area_m2,
        rel_tol=0.0,
        abs_tol=expectations.projected_area_absolute_tolerance_m2,
    ):
        raise Stage2SInputError("projected study-area area changed")
    query_grid = _query_grid(build_stage3_query_grid(study_area.geographic))
    if (
        expectations.expected_grid_id is not None
        and query_grid.grid_id != expectations.expected_grid_id
    ):
        raise Stage2SInputError("runtime query grid identity changed")
    grid_family = build_equal_area_grid_family(study_area.projected)
    grid_25 = grid_family.at(25.0)
    if grid_25.cell_ids != query_grid.cell_ids:
        raise Stage2SInputError("operational and query 25 km grid IDs differ")
    grid_xy = np.asarray(
        [(cell.representative_point.x, cell.representative_point.y) for cell in grid_25.cells],
        dtype=np.float64,
    )
    grid_area = np.asarray(
        [cell.clipped_area_m2 / 1_000_000.0 for cell in grid_25.cells],
        dtype=np.float64,
    )
    if not np.array_equal(grid_xy, query_grid.query_xy_m) or not np.array_equal(
        grid_area,
        query_grid.clipped_area_km2,
    ):
        raise Stage2SInputError("query and operational 25 km grids differ bitwise")
    mapping, schema_hash, zone_ids = _parse_mapping(
        mapping_bytes,
        query_grid=query_grid,
        expected_rows=expectations.mapping_row_count,
        expected_zones=expectations.zone_count,
    )
    grid_counts = tuple(len(grid_family.at(cell_size).cells) for cell_size in (50.0, 25.0, 12.5))
    grid_areas = tuple(
        math.fsum(cell.clipped_area_m2 / 1_000_000.0 for cell in grid_family.at(cell_size).cells)
        for cell_size in (50.0, 25.0, 12.5)
    )
    area_tolerance_km2 = expectations.projected_area_absolute_tolerance_m2 / 1_000_000.0
    if any(
        not math.isclose(
            total,
            expectations.projected_area_m2 / 1_000_000.0,
            rel_tol=0.0,
            abs_tol=area_tolerance_km2,
        )
        for total in grid_areas
    ):
        raise Stage2SInputError("one or more grid families do not close to study-area area")
    adapter = NonTargetSpatialAdapter(
        query_grid=query_grid,
        construction_zone_id_by_cell_id=mapping,
        grid_family=grid_family,
    )
    spatial_identity = _build_non_target_spatial_identity(
        adapter,
        area_tolerance_km2=area_tolerance_km2,
    )
    return NonTargetPreflight(
        adapter=adapter,
        study_area_file_sha256=observed_study_hash,
        projected_geometry_sha256=geometry_hash,
        projected_area_m2=area_m2,
        mapping_file_sha256=observed_mapping_hash,
        mapping_schema_sha256=schema_hash,
        zone_ids=zone_ids,
        grid_cell_counts=cast(tuple[int, int, int], grid_counts),
        grid_total_area_km2=cast(tuple[float, float, float], grid_areas),
        spatial_identity=spatial_identity,
    )


def _read_once(path: Path, *, label: str) -> bytes:
    try:
        with path.open("rb") as handle:
            payload = handle.read()
    except OSError as exc:
        raise Stage2SInputError(f"cannot open {label} exactly once") from exc
    if not payload:
        raise Stage2SInputError(f"{label} is empty")
    return payload


def expectations_from_protocol(bundle: Stage2SProtocolBundle) -> NonTargetExpectations:
    sources = cast(dict[str, Any], bundle.config["source_contracts"])
    study = cast(dict[str, Any], sources["study_area"])
    mapping = cast(dict[str, Any], sources["cell_zone_mapping"])
    return NonTargetExpectations(
        study_area_file_sha256=cast(str, study["file_sha256"]),
        projected_geometry_sha256=cast(str, study["projected_geometry_sha256"]),
        projected_area_m2=float(study["projected_total_area_m2"]),
        projected_area_absolute_tolerance_m2=float(study["projected_area_absolute_tolerance_m2"]),
        mapping_file_sha256=cast(str, mapping["sha256"]),
        mapping_row_count=int(mapping["row_count"]),
        zone_count=int(mapping["required_nonempty_zone_count"]),
    )


def run_non_target_spatial_preflight(
    bundle: Stage2SProtocolBundle,
) -> NonTargetPreflight:
    """Open the real study area and mapping once each after remote code-tag verification."""

    sources = cast(dict[str, Any], bundle.config["source_contracts"])
    study = cast(dict[str, Any], sources["study_area"])
    mapping = cast(dict[str, Any], sources["cell_zone_mapping"])
    study_path = bundle.repository_root / cast(str, study["path"])
    mapping_path = bundle.repository_root / cast(str, mapping["path"])
    study_bytes = _read_once(study_path, label="study-area bytes")
    mapping_bytes = _read_once(mapping_path, label="cell-zone mapping bytes")
    return build_non_target_adapter_from_bytes(
        study_area_bytes=study_bytes,
        mapping_bytes=mapping_bytes,
        expectations=expectations_from_protocol(bundle),
    )


__all__ = [
    "EXPECTED_MAPPING_FIELDS",
    "NonTargetExpectations",
    "NonTargetPreflight",
    "NonTargetSpatialAdapter",
    "NonTargetSpatialIdentity",
    "SpatialGridLayerIdentity",
    "SpatialParentRelationIdentity",
    "Stage2SInputError",
    "Stage2SQueryGrid",
    "build_non_target_adapter_from_bytes",
    "expectations_from_protocol",
    "run_non_target_spatial_preflight",
    "to_spatial_quadrature_family",
    "validated_non_target_preflight_receipt_bindings",
]
