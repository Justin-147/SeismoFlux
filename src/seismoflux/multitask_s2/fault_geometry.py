"""Target-independent fault proximity for the frozen S2-A finite experiment.

Only geometry is read. The 2026 snapshot is a retrospective static covariate,
not evidence that these fault maps were available at historical forecast dates.
All surfaces are log *relative spatial cell mass*, not earthquake probability.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray
from pyproj import CRS, Transformer
from scipy.special import logsumexp  # type: ignore[import-untyped]
from shapely import STRtree, from_wkb, points
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform

from seismoflux.background.grid import EQUAL_AREA_CRS
from seismoflux.d1_replay.spatial import D1SpatialDomain
from seismoflux.stage2s.inputs import _parse_mapping, _query_grid

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
_SOURCE_FIELDS = {
    "SIMPLE": ("fault_segment_id", "simplified_geometry_wkb", None),
    "TRACE": ("trace_id", "geometry_wkb", "usable_for_geometry"),
}


@dataclass(frozen=True, slots=True)
class FaultSurfaces:
    """Each source and distance scale maps to an ordered log cell-mass vector."""

    fine: dict[str, dict[float, FloatArray]]
    coarse: dict[str, dict[float, FloatArray]]
    distance_km: dict[str, FloatArray]
    block_index: IntArray
    audit: dict[str, Any]


def _readonly(values: FloatArray) -> FloatArray:
    result = np.array(values, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


def _vector(values: object, *, label: str) -> FloatArray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{label} must be a nonempty finite vector")
    return result


def _areas(values: object, length: int) -> FloatArray:
    result = _vector(values, label="actual cell areas")
    if result.shape != (length,) or np.any(result <= 0.0):
        raise ValueError("actual cell areas must be positive and aligned with the grid")
    return result


def _normalized_log_mass(values: object) -> FloatArray:
    result = _vector(values, label="log cell mass")
    if not math.isclose(float(logsumexp(result)), 0.0, rel_tol=0.0, abs_tol=1.0e-10):
        raise ValueError("input log cell mass must already be nationally normalized")
    return result


def proximity_log_mass(
    distance_km: FloatArray, area_km2: FloatArray, scale_km: float
) -> FloatArray:
    """Return normalized ``area / (1 + (distance / scale)**2)`` in log space."""

    distance = _vector(distance_km, label="projected distance in km")
    area = _areas(area_km2, distance.size)
    scale = float(scale_km)
    if np.any(distance < 0.0) or not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("distances must be nonnegative and scale must be finite and positive")
    # Log space also handles distances whose square or ratio would overflow.
    log_ratio = np.full(distance.shape, -np.inf, dtype=np.float64)
    positive = distance > 0.0
    log_ratio[positive] = np.log(distance[positive]) - math.log(scale)
    unnormalized = np.log(area) - np.logaddexp(0.0, 2.0 * log_ratio)
    result = np.asarray(unnormalized - logsumexp(unnormalized), dtype=np.float64)
    if not np.isfinite(result).all():
        raise FloatingPointError("fault proximity produced a non-finite log mass")
    return _readonly(result)


def coarsen_log_mass(
    log_mass: FloatArray, area_km2: FloatArray, block_index: IntArray
) -> FloatArray:
    """Keep every block's mass but make density constant within each block.

    This is a fixed-block detail-removal control, not a permutation or a
    leave-region-out evaluation. No target locations are involved.
    """

    mass = _normalized_log_mass(log_mass)
    area = _areas(area_km2, mass.size)
    blocks = np.asarray(block_index)
    if blocks.shape != mass.shape or blocks.dtype.kind not in "iu" or np.any(blocks < 0):
        raise ValueError("block indices must be aligned nonnegative integers")
    result = np.empty(mass.shape, dtype=np.float64)
    log_area = np.log(area)
    for block in np.unique(blocks):
        selected = blocks == block
        result[selected] = (
            float(logsumexp(mass[selected]))
            + log_area[selected]
            - float(logsumexp(log_area[selected]))
        )
    # Do not add a global correction: each block's original mass is preserved.
    return _readonly(result)


def blend_log_masses(
    catalog_log_mass: FloatArray, fault_log_mass: FloatArray, alpha: float
) -> FloatArray:
    """Linear mixture of normalized masses, preserving both endpoints exactly."""

    catalog = _normalized_log_mass(catalog_log_mass)
    fault = _normalized_log_mass(fault_log_mass)
    weight = float(alpha)
    if catalog.shape != fault.shape:
        raise ValueError("catalog and fault cell masses must have the same shape")
    if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("fault mixture alpha must be between zero and one")
    if weight == 0.0:
        return _readonly(catalog)
    if weight == 1.0:
        return _readonly(fault)
    mixture = np.logaddexp(catalog + math.log1p(-weight), fault + math.log(weight))
    return _readonly(np.asarray(mixture - logsumexp(mixture), dtype=np.float64))


def _project_lines_to_equal_area(lines: tuple[BaseGeometry, ...]) -> tuple[BaseGeometry, ...]:
    """Project the existing WGS84 interpretation into the original metre CRS."""

    transformer = Transformer.from_crs(
        CRS.from_epsg(4326), CRS.from_user_input(EQUAL_AREA_CRS), always_xy=True
    )
    projected = tuple(transform(transformer.transform, line) for line in lines)
    for line in projected:
        if (
            line.geom_type != "LineString"
            or line.is_empty
            or not line.is_valid
            or not math.isfinite(float(line.length))
            or line.length <= 0.0
            or not np.isfinite(np.asarray(line.coords)).all()
        ):
            raise ValueError("projected fault must be a finite, valid, positive-length LineString")
    return projected


def _minimum_projected_distance_km(
    projected_lines_m: tuple[BaseGeometry, ...], query_xy_km: FloatArray
) -> FloatArray:
    """Nearest-line distance with metre geometries and existing kilometre queries."""

    query = np.asarray(query_xy_km, dtype=np.float64)
    if query.ndim != 2 or query.shape[1] != 2 or not query.shape[0]:
        raise ValueError("query points must have nonempty shape (cells, 2)")
    if not np.isfinite(query).all() or not projected_lines_m:
        raise ValueError("finite query points and at least one fault line are required")
    query_m = query * 1_000.0
    if not np.isfinite(query_m).all():
        raise ValueError("query coordinates overflow after conversion to metres")
    tree = STRtree(projected_lines_m)
    indices, distance_m = tree.query_nearest(
        points(query_m), all_matches=False, return_distance=True
    )
    if not np.array_equal(indices[0], np.arange(query.shape[0], dtype=np.int64)):
        raise ValueError("nearest-line query did not return every ordered grid point")
    result = np.asarray(distance_m, dtype=np.float64) / 1_000.0
    if not np.isfinite(result).all() or np.any(result < 0.0):
        raise ValueError("nearest-line distance must be finite and nonnegative")
    return _readonly(result)


def _verified_payload(data_root: Path, relative_path: str, expected_sha256: str) -> bytes:
    path = (data_root / relative_path).resolve()
    if not path.is_relative_to(data_root):
        raise ValueError("geometry input path must stay within the data root")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f"frozen input SHA-256 mismatch: {relative_path}")
    return payload


def _source_lines(
    payload: bytes, source: str, specification: dict[str, Any]
) -> tuple[tuple[BaseGeometry, ...], dict[str, Any]]:
    id_column, geometry_column, usable_column = _SOURCE_FIELDS[source]
    columns = [id_column, geometry_column]
    if usable_column is not None:
        columns.append(usable_column)
    if (
        specification["allowed_columns"] != columns
        or specification["id_column"] != id_column
        or specification["geometry_column"] != geometry_column
        or specification["usable_column"] != usable_column
    ):
        raise ValueError(f"{source} must use only its frozen geometry field allowlist")
    table = pq.read_table(pa.BufferReader(payload), columns=columns, use_threads=False)
    if table.num_rows != specification["expected_rows"]:
        raise ValueError(f"{source} geometry source row count changed")
    if table[id_column].null_count or table[geometry_column].null_count:
        raise ValueError(f"{source} geometry IDs and WKB must not be null")
    identifiers = table[id_column].to_pylist()
    if any(not isinstance(value, str) or not value for value in identifiers):
        raise ValueError(f"{source} geometry IDs must be nonempty strings")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{source} geometry IDs must be unique")
    if usable_column is None:
        usable = [True] * table.num_rows
    else:
        usable = table[usable_column].to_pylist()
        if any(not isinstance(value, bool) for value in usable):
            raise ValueError(f"{source} geometry usability flags must be explicit booleans")
    selected_wkb = [
        value
        for value, keep in zip(table[geometry_column].to_pylist(), usable, strict=True)
        if keep
    ]
    if len(selected_wkb) != specification["expected_usable_lines"]:
        raise ValueError(f"{source} usable geometry count changed")
    lines = tuple(from_wkb(value) for value in selected_wkb)
    for line in lines:
        if (
            line is None
            or line.geom_type != "LineString"
            or line.is_empty
            or not line.is_valid
            or line.length <= 0.0
            or not np.isfinite(np.asarray(line.coords)).all()
        ):
            raise ValueError(f"{source} usable geometry is not a valid positive-length LineString")
    return lines, {
        "rows": table.num_rows,
        "usable_lines": len(lines),
        "excluded_by_existing_usability_flag": table.num_rows - len(lines),
        "read_columns": columns,
        "attribute_fields_read": False,
        "duplicate_lines_retained": True,
    }


def load_fault_surfaces(
    data_root: Path, protocol: dict[str, Any], domain: D1SpatialDomain
) -> FaultSurfaces:
    """Read authenticated geometry and the existing target-blind block mapping.

    This function does not load catalogs, inspect targets, train any model,
    score forecasts, write files, or alter source eligibility metadata.
    """

    root = Path(data_root).resolve(strict=True)
    inputs = protocol["inputs"]
    geometry_math = protocol["geometry_math"]
    grid = domain.operational_grid
    stage3_grid = domain.stage3_grid
    if (
        grid.grid_id != inputs["grid_id"]
        or grid.cell_count != inputs["grid_cells"]
        or grid.cell_size_km != inputs["grid_cell_size_km"]
        or stage3_grid.grid_id != grid.grid_id
        or stage3_grid.cell_ids != grid.cell_ids
        or not np.array_equal(stage3_grid.query_xy_m / 1_000.0, grid.query_xy_km)
        or not np.array_equal(stage3_grid.clipped_area_km2, grid.clipped_area_km2)
        or not math.isclose(
            math.fsum(float(value) for value in grid.clipped_area_km2),
            float(inputs["grid_area_km2"]),
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
    ):
        raise ValueError(
            "S2-A grid differs from its frozen original representative points or areas"
        )
    if set(inputs["geometry_sources"]) != set(_SOURCE_FIELDS):
        raise ValueError("S2-A requires exactly the SIMPLE and TRACE geometry sources")
    scales = tuple(float(value) for value in geometry_math["scales_km"])
    if scales != (25.0, 75.0, 150.0) or float(geometry_math["power"]) != 2.0:
        raise ValueError("S2-A proximity scales or power differ from the frozen finite experiment")
    mapping_spec = inputs["cell_to_block_mapping"]
    mapping_payload = _verified_payload(root, mapping_spec["path"], mapping_spec["sha256"])
    mapping, schema_hash, zone_ids = _parse_mapping(
        mapping_payload,
        query_grid=_query_grid(stage3_grid),
        expected_rows=grid.cell_count,
        expected_zones=int(mapping_spec["blocks"]),
    )
    zone_index = {zone: index for index, zone in enumerate(zone_ids)}
    block_index = np.asarray([zone_index[mapping[cell]] for cell in grid.cell_ids], dtype=np.int64)
    block_index.setflags(write=False)
    fine: dict[str, dict[float, FloatArray]] = {}
    coarse: dict[str, dict[float, FloatArray]] = {}
    distance: dict[str, FloatArray] = {}
    source_audit: dict[str, Any] = {}
    for source, specification in inputs["geometry_sources"].items():
        payload = _verified_payload(root, specification["path"], specification["sha256"])
        lines, audit = _source_lines(payload, source, specification)
        distance[source] = _minimum_projected_distance_km(
            _project_lines_to_equal_area(lines), grid.query_xy_km
        )
        fine[source] = {
            scale: proximity_log_mass(distance[source], grid.clipped_area_km2, scale)
            for scale in scales
        }
        coarse[source] = {
            scale: coarsen_log_mass(surface, grid.clipped_area_km2, block_index)
            for scale, surface in fine[source].items()
        }
        source_audit[source] = {
            **audit,
            "path": specification["path"],
            "sha256": specification["sha256"],
            "distance_min_km": float(np.min(distance[source])),
            "distance_max_km": float(np.max(distance[source])),
            "distance_median_km": float(np.median(distance[source])),
        }
    return FaultSurfaces(
        fine=fine,
        coarse=coarse,
        distance_km=distance,
        block_index=block_index,
        audit={
            "scientific_role": protocol["scientific_role"],
            "static_snapshot_available_at": inputs["static_snapshot_available_at"],
            "original_historical_model_eligible": inputs["original_historical_model_eligible"],
            "source_eligibility_flags_modified": False,
            "source_crs_assumption": geometry_math["source_crs_assumption"],
            "projected_crs": EQUAL_AREA_CRS,
            "distance_units": "km",
            "grid_id": grid.grid_id,
            "grid_cells": grid.cell_count,
            "cell_to_block_mapping_sha256": mapping_spec["sha256"],
            "mapping_schema_sha256": schema_hash,
            "zone_ids_utf8_order": list(zone_ids),
            "blocks": len(zone_ids),
            "scales_km": list(scales),
            "sources": source_audit,
            "target_coordinates_read": False,
            "coarse_is_permutation_or_transfer_test": False,
        },
    )
