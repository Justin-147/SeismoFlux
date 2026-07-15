"""Target-independent local query grid for stage-3 anomaly features.

The only scientific input accepted here is the frozen study-area geometry.  In
particular, the builder has no earthquake, target, score, completeness, or local
support-mask argument.  Cell identifiers and projected coordinates are local-only
lineage fields and must be removed from public stage-3 deliverables.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from shapely.geometry.base import BaseGeometry

from seismoflux.background.grid import (
    EQUAL_AREA_CRS,
    build_clipped_grid,
    project_study_area_to_equal_area,
)
from seismoflux.data.common import canonical_json_bytes

STAGE3_QUERY_CELL_SIZE_KM = 25.0


@dataclass(frozen=True, slots=True)
class Stage3QueryGrid:
    """Local-only, row/column ordered 25 km feature query grid."""

    grid_id: str
    equal_area_crs: str
    cell_size_km: float
    cell_ids: tuple[str, ...]
    rows: NDArray[np.int64]
    columns: NDArray[np.int64]
    query_xy_m: NDArray[np.float64]
    clipped_area_km2: NDArray[np.float64]

    @property
    def cell_count(self) -> int:
        return len(self.cell_ids)


def _grid_identity_payload(
    *,
    cell_ids: tuple[str, ...],
    rows: NDArray[np.int64],
    columns: NDArray[np.int64],
    query_xy_m: NDArray[np.float64],
    clipped_area_km2: NDArray[np.float64],
) -> dict[str, object]:
    """Return the exact local grid payload used only for content identity."""

    return {
        "schema_version": 1,
        "role": "target_independent_local_feature_queries",
        "equal_area_crs": EQUAL_AREA_CRS,
        "cell_size_km": STAGE3_QUERY_CELL_SIZE_KM,
        "cells": [
            {
                "cell_id": cell_id,
                "row": int(row),
                "column": int(column),
                "query_x_m_hex": float(x_m).hex(),
                "query_y_m_hex": float(y_m).hex(),
                "clipped_area_km2_hex": float(area_km2).hex(),
            }
            for cell_id, row, column, (x_m, y_m), area_km2 in zip(
                cell_ids,
                rows,
                columns,
                query_xy_m,
                clipped_area_km2,
                strict=True,
            )
        ],
    }


def build_stage3_query_grid(study_area_wgs84: BaseGeometry) -> Stage3QueryGrid:
    """Build the frozen 25 km query grid from the study area and nothing else."""

    projected = project_study_area_to_equal_area(study_area_wgs84)
    grid = build_clipped_grid(projected, cell_size_km=STAGE3_QUERY_CELL_SIZE_KM)
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
        cell_ids=cell_ids,
        rows=rows,
        columns=columns,
        query_xy_m=query_xy_m,
        clipped_area_km2=clipped_area_km2,
    )
    grid_id = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return Stage3QueryGrid(
        grid_id=grid_id,
        equal_area_crs=EQUAL_AREA_CRS,
        cell_size_km=STAGE3_QUERY_CELL_SIZE_KM,
        cell_ids=cell_ids,
        rows=rows,
        columns=columns,
        query_xy_m=query_xy_m,
        clipped_area_km2=clipped_area_km2,
    )


__all__ = [
    "STAGE3_QUERY_CELL_SIZE_KM",
    "Stage3QueryGrid",
    "build_stage3_query_grid",
]
