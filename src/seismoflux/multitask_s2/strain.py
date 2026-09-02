"""Frozen GSRM cell-area remapping; no earthquake inputs or output writes."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pyproj import Transformer
from scipy.special import logsumexp
from shapely.geometry import Polygon, box
from shapely.ops import transform
from shapely.strtree import STRtree

from seismoflux.background.grid import EQUAL_AREA_CRS
from seismoflux.d1_replay.spatial import D1SpatialDomain


@dataclass(frozen=True, slots=True)
class StrainSurfaces:
    layers: dict[str, np.ndarray]
    audit: dict[str, Any]


def validate_log_mass(values: Any, expected_cells: int | None = None) -> np.ndarray:
    """Validate normalized log mass without replacing genuine zero support."""
    result = np.array(values, dtype=np.float64, copy=True)
    if result.ndim != 1 or not result.size:
        raise ValueError("log mass must be a nonempty vector")
    if expected_cells is not None and result.size != expected_cells:
        raise ValueError("unexpected cell count")
    if np.isnan(result).any() or np.isposinf(result).any() or np.isneginf(result).all():
        raise ValueError("invalid log mass")
    if not math.isclose(float(logsumexp(result)), 0.0, abs_tol=1e-10):
        raise ValueError("log mass is not nationally normalized")
    result.setflags(write=False)
    return result


def blend_log_masses(catalog: Any, static: Any, alpha: float) -> np.ndarray:
    c = validate_log_mass(catalog)
    s = validate_log_mass(static, c.size)
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    if alpha == 0:
        return c
    if alpha == 1:
        return s
    return validate_log_mass(np.logaddexp(math.log1p(-alpha) + c, math.log(alpha) + s))


def _strain_scalar(exx: Any, eyy: Any, exy: Any) -> np.ndarray:
    components = np.asarray([exx, eyy, exy], dtype=np.float64)
    if not np.isfinite(components).all():
        raise ValueError("nonfinite strain tensor")
    result = np.hypot(np.hypot(components[0], components[1]), math.sqrt(2) * components[2])
    if not np.isfinite(result).all():
        raise ValueError("nonfinite strain invariant")
    return result


def _source_rectangles(lon: np.ndarray, lat: np.ndarray, width: float, height: float) -> list:
    centers = np.column_stack((lon, lat))
    if not np.isfinite(centers).all():
        raise ValueError("nonfinite source coordinates")
    if np.unique(centers, axis=0).shape[0] != len(centers):
        raise ValueError("duplicate source cells")
    if (np.abs(lon) + width / 2 > 180 + 1e-9).any() or (np.abs(lat) + height / 2 > 90 + 1e-9).any():
        raise ValueError("source cells outside geographic bounds")
    rectangles = [
        box(x - width / 2, y - height / 2, x + width / 2, y + height / 2) for x, y in centers
    ]
    tree = STRtree(rectangles)
    # Only shared-edge rounding (< 1e-9 degree), not physical overlap, is ignored.
    for i, rect in enumerate(rectangles):
        for j in tree.query(rect):
            if j <= i:
                continue
            overlap_x = width - abs(lon[i] - lon[j])
            overlap_y = height - abs(lat[i] - lat[j])
            if overlap_x > 1e-9 and overlap_y > 1e-9:
                raise ValueError("overlapping source cells")
    return rectangles


def _densified_rectangle(bounds: Sequence[float], step: float) -> Polygon:
    left, bottom, right, top = bounds
    corners = [(left, bottom), (right, bottom), (right, top), (left, top)]
    coordinates = []
    for start, end in zip(corners, corners[1:] + corners[:1], strict=True):
        count = math.ceil(max(abs(end[0] - start[0]), abs(end[1] - start[1])) / step)
        coordinates.extend(
            (start[0] + (end[0] - start[0]) * k / count, start[1] + (end[1] - start[1]) * k / count)
            for k in range(count)
        )
    return Polygon(coordinates)


def _remap_projected(sources: Sequence, scalars: np.ndarray, targets: Sequence) -> tuple:
    """Accumulate physical intersection areas before any national normalization."""
    if len(sources) != len(scalars) or not np.isfinite(scalars).all() or (scalars < 0).any():
        raise ValueError("invalid source scalar array")
    areas = np.zeros(len(targets), dtype=np.float64)
    weighted = np.zeros(len(targets), dtype=np.float64)
    tree = STRtree(targets)
    for geometry, scalar in zip(sources, scalars, strict=True):
        if not geometry.is_valid:
            raise ValueError("invalid projected source polygon")
        for index in tree.query(geometry, predicate="intersects"):
            area = geometry.intersection(targets[index]).area / 1e6
            areas[index] += area
            weighted[index] += area * scalar
    layers = {}
    for name, mass in (("UNIT", areas), ("STRAIN", weighted)):
        total = math.fsum(mass)
        if not math.isfinite(total) or total <= 0:
            raise ValueError(f"{name} has no positive national mass")
        values = np.full(len(targets), -np.inf, dtype=np.float64)
        positive = mass > 0
        values[positive] = np.log(mass[positive]) - math.log(total)
        layers[name] = validate_log_mass(values, len(targets))
    return layers, areas, weighted


def load_strain_surfaces(
    *, data_root: Path, domain: D1SpatialDomain, protocol: Mapping[str, Any]
) -> StrainSurfaces:
    inputs = protocol["inputs"]
    source = inputs["strain_source"]
    grid = domain.operational_grid
    if (
        grid.grid_id != inputs["grid_id"]
        or grid.cell_count != inputs["grid_cells"]
        or grid.cell_size_km != inputs["grid_cell_size_km"]
    ):
        raise ValueError("frozen operational grid mismatch")
    if not math.isclose(
        math.fsum(grid.clipped_area_km2), inputs["grid_area_km2"], rel_tol=0, abs_tol=1e-9
    ):
        raise ValueError("frozen grid area mismatch")
    targets = domain.locator.clipped_geometries
    if len(targets) != grid.cell_count:
        raise ValueError("clipped geometry count mismatch")
    path = (data_root / source["path"]).resolve()
    if not path.is_relative_to(data_root.resolve()):
        raise ValueError("source path outside data root")
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest.lower() != source["sha256"].lower():
        raise ValueError("GSRM source SHA256 mismatch")
    table = np.loadtxt(path, comments="#", dtype=np.float64, ndmin=2)
    if table.shape != (source["expected_rows"], source["expected_columns"]):
        raise ValueError("GSRM source shape mismatch")
    if not np.isfinite(table).all():
        raise ValueError("nonfinite GSRM source values")
    lat, lon = table[:, 0], table[:, 1]
    scalars = _strain_scalar(table[:, 2], table[:, 3], table[:, 4])
    spatial = protocol["spatial_math"]
    rectangles = _source_rectangles(
        lon, lat, spatial["longitude_width_degrees"], spatial["latitude_height_degrees"]
    )
    # Geographic study bounds prefilter sources, never earthquake locations.
    candidates = STRtree(rectangles).query(box(*domain.study_area_wgs84.bounds))
    projector = Transformer.from_crs("EPSG:4326", EQUAL_AREA_CRS, always_xy=True)
    projected = [
        transform(
            projector.transform,
            _densified_rectangle(
                rectangles[i].bounds, spatial["geodetic_edge_maximum_segment_degrees"]
            ),
        )
        for i in candidates
    ]
    layers, areas, weighted = _remap_projected(projected, scalars[candidates], targets)
    # Principal values are a one-time diagnostic only, never used in layers.
    principal_norm = np.hypot(table[:, 11], table[:, 12])
    audit = {
        "source_path": str(path),
        "source_sha256": digest,
        "source_rows": len(table),
        "candidate_source_cells": len(candidates),
        "grid_id": grid.grid_id,
        "grid_cells": grid.cell_count,
        "source_cell_overlap": "none_except_shared_edge_roundoff",
        "shared_edge_roundoff_tolerance_degrees": 1e-9,
        "strain_scalar": spatial["strain_scalar"],
        "tensor_shear": True,
        "covered_area_km2": math.fsum(areas),
        "strain_area_integral": math.fsum(weighted),
        "unit_zero_cells": int(np.count_nonzero(areas == 0)),
        "strain_zero_cells": int(np.count_nonzero(weighted == 0)),
        "uncovered_domain_role": spatial["uncovered_domain_role"],
        "principal_norm_max_absolute_difference": float(np.max(abs(principal_norm - scalars))),
        "principal_values_used_as_features": False,
        "edge_maximum_segment_degrees": spatial["geodetic_edge_maximum_segment_degrees"],
        "license": source["license"],
        "attribution": source["attribution"],
        "header_release_date": source["header_release_date"],
        "historical_exact_bytes_confirmed": source["historical_exact_bytes_confirmed"],
        "scientific_role": protocol["scientific_role"],
    }
    return StrainSurfaces(layers=layers, audit=audit)
