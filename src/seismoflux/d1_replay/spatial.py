"""Causal spatial foundations for the D1 retrospective replay.

This module deliberately does only four scientific jobs: rebuild the target-
independent study grid, locate events on that frozen grid, construct causal
seismicity backgrounds, and turn one spatial mass vector into complete-cell
alarm prefixes.  It accepts no target score or observed model effect.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Final, cast

import numpy as np
from numpy.typing import NDArray
from pyproj import CRS, Transformer
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

from seismoflux.background.catalog import load_study_area_bytes
from seismoflux.background.grid import (
    EQUAL_AREA_CRS,
    EqualAreaGrid,
    build_equal_area_grid_family,
    project_study_area_to_equal_area,
)
from seismoflux.data.common import canonical_json_bytes
from seismoflux.features.anomaly.grid import Stage3QueryGrid, build_stage3_query_grid
from seismoflux.stage2s.catalog import Stage2SEarthquakeCatalog
from seismoflux.stage2s.contracts import (
    MASS_SUM_ABSOLUTE_TOLERANCE,
    NormalizedSpatialDensity,
    SpatialGrid,
    SpatialQuadratureFamily,
)
from seismoflux.stage2s.spatial import (
    build_normalized_kde,
    build_recent_component,
    mix_density,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

FROZEN_D1_AREA_BUDGETS_KM2: Final[tuple[float, ...]] = (
    300_000.0,
    450_000.0,
    600_000.0,
    750_000.0,
    960_000.0,
)
FROZEN_R30_ALPHA_CANDIDATES: Final[tuple[float, ...]] = (0.0, 0.25, 0.5, 0.75)
_OPERATIONAL_CELL_SIZE_KM: Final = 25.0
_OPERATIONAL_CELL_SIZE_M: Final = 25_000.0
_MICROSECONDS_PER_DAY: Final = 86_400_000_000
_R30_MICROSECONDS: Final = 30 * _MICROSECONDS_PER_DAY
_CATALOG_START_US: Final = 0
_UTC_EPOCH: Final = datetime(1970, 1, 1, tzinfo=UTC)


class D1SpatialError(RuntimeError):
    """Raised when a frozen D1 spatial or causal-input invariant is broken."""


def _readonly_float(values: object) -> FloatArray:
    result = np.array(values, dtype=np.float64, copy=True, order="C")
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError("spatial mass must be a finite one-dimensional array")
    result.setflags(write=False)
    return result


def _readonly_int(values: object) -> IntArray:
    result = np.array(values, dtype=np.int64, copy=True, order="C")
    if result.ndim != 1:
        raise ValueError("selected indices must be one-dimensional")
    result.setflags(write=False)
    return result


def _bitwise_equal(left: NDArray[np.generic], right: NDArray[np.generic]) -> bool:
    """Compare shape, dtype and every stored bit, including signed zero."""

    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.tobytes(order="C") == right.tobytes(order="C")
    )


def _equal_area_grid_arrays(
    grid: EqualAreaGrid,
) -> tuple[tuple[str, ...], IntArray, IntArray, NDArray[np.float64], FloatArray]:
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


def _d1_grid_id(
    *,
    cell_size_km: float,
    cell_ids: tuple[str, ...],
    rows: IntArray,
    columns: IntArray,
    query_xy_m: NDArray[np.float64],
    clipped_area_km2: FloatArray,
) -> str:
    payload = {
        "schema_version": 1,
        "role": "d1_target_independent_spatial_quadrature",
        "equal_area_crs": EQUAL_AREA_CRS,
        "cell_size_km": cell_size_km,
        "cells": [
            {
                "cell_id": cell_identifier,
                "row": int(row),
                "column": int(column),
                "query_x_m_hex": float(x_m).hex(),
                "query_y_m_hex": float(y_m).hex(),
                "clipped_area_km2_hex": float(area).hex(),
            }
            for cell_identifier, row, column, (x_m, y_m), area in zip(
                cell_ids,
                rows,
                columns,
                query_xy_m,
                clipped_area_km2,
                strict=True,
            )
        ],
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class Frozen25kmCellLocator:
    """Map projected or geographic points to exact clipped 25 km cells.

    Exact grid-line points try the high-side cell first, followed by high-row/
    low-column, low-row/high-column, then low-row/low-column.  ``covers`` is
    intentionally used so that points on the target-independent outer boundary
    remain locatable.  No coordinate perturbation or tolerance is applied.
    """

    grid: SpatialGrid
    clipped_geometries: tuple[BaseGeometry, ...]
    _index_by_row_column: MappingProxyType[tuple[int, int], int] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _wgs84_to_equal_area: Transformer = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.grid, SpatialGrid) or self.grid.cell_size_km != 25.0:
            raise D1SpatialError("cell locator requires the frozen 25 km SpatialGrid")
        geometries = tuple(self.clipped_geometries)
        if len(geometries) != self.grid.cell_count:
            raise D1SpatialError("clipped geometries must match every ordered 25 km cell")
        lookup: dict[tuple[int, int], int] = {}
        for index, (row, column, geometry, area, query_xy_km) in enumerate(
            zip(
                self.grid.rows,
                self.grid.columns,
                geometries,
                self.grid.clipped_area_km2,
                self.grid.query_xy_km,
                strict=True,
            )
        ):
            key = (int(row), int(column))
            if key in lookup:
                raise D1SpatialError("25 km locator row/column pairs must be unique")
            if geometry.is_empty or not geometry.is_valid or float(geometry.area) <= 0.0:
                raise D1SpatialError("25 km clipped geometry must be valid and positive-area")
            if float(geometry.area) / 1_000_000.0 != float(area):
                raise D1SpatialError("25 km clipped geometry and SpatialGrid areas differ")
            representative = Point(
                float(query_xy_km[0]) * 1_000.0,
                float(query_xy_km[1]) * 1_000.0,
            )
            if not geometry.covers(representative):
                raise D1SpatialError("25 km representative point is outside its clipped cell")
            lookup[key] = index
        transformer = Transformer.from_crs(
            CRS.from_epsg(4326),
            CRS.from_user_input(EQUAL_AREA_CRS),
            always_xy=True,
        )
        object.__setattr__(self, "clipped_geometries", geometries)
        object.__setattr__(self, "_index_by_row_column", MappingProxyType(lookup))
        object.__setattr__(self, "_wgs84_to_equal_area", transformer)

    @staticmethod
    def candidate_row_columns(x_m: float, y_m: float) -> tuple[tuple[int, int], ...]:
        """Return the exact frozen high-side-first candidate sequence."""

        x_value = float(x_m)
        y_value = float(y_m)
        if not math.isfinite(x_value) or not math.isfinite(y_value):
            raise ValueError("projected point coordinates must be finite")
        high_column = math.floor(x_value / _OPERATIONAL_CELL_SIZE_M)
        high_row = math.floor(y_value / _OPERATIONAL_CELL_SIZE_M)
        x_on_grid_line = x_value == high_column * _OPERATIONAL_CELL_SIZE_M
        y_on_grid_line = y_value == high_row * _OPERATIONAL_CELL_SIZE_M
        candidates = [(high_row, high_column)]
        if x_on_grid_line:
            candidates.append((high_row, high_column - 1))
        if y_on_grid_line:
            candidates.append((high_row - 1, high_column))
        if x_on_grid_line and y_on_grid_line:
            candidates.append((high_row - 1, high_column - 1))
        return tuple(candidates)

    def locate_projected(self, x_m: float, y_m: float) -> int | None:
        """Return the first exact clipped-cell match, or ``None`` outside support."""

        point = Point(float(x_m), float(y_m))
        for key in self.candidate_row_columns(x_m, y_m):
            index = self._index_by_row_column.get(key)
            if index is not None and self.clipped_geometries[index].covers(point):
                return index
        return None

    def locate_lonlat(self, longitude: float, latitude: float) -> int | None:
        """Project one WGS84 point and apply the same exact clipped-cell rule."""

        lon = float(longitude)
        lat = float(latitude)
        if not math.isfinite(lon) or not math.isfinite(lat):
            raise ValueError("event longitude and latitude must be finite")
        x_m, y_m = self._wgs84_to_equal_area.transform(lon, lat)
        return self.locate_projected(float(x_m), float(y_m))


@dataclass(frozen=True, slots=True)
class D1SpatialDomain:
    """One target-independent Stage-3-compatible D1 spatial domain."""

    stage3_grid: Stage3QueryGrid
    quadrature_family: SpatialQuadratureFamily
    study_area_wgs84: BaseGeometry
    study_area_equal_area: BaseGeometry
    locator: Frozen25kmCellLocator

    @property
    def operational_grid(self) -> SpatialGrid:
        return self.quadrature_family.at(_OPERATIONAL_CELL_SIZE_KM)


def _assert_stage3_25km_identity(
    stage3_grid: Stage3QueryGrid,
    grid_25km: EqualAreaGrid,
) -> tuple[tuple[str, ...], IntArray, IntArray, NDArray[np.float64], FloatArray]:
    arrays = _equal_area_grid_arrays(grid_25km)
    cell_ids, rows, columns, query_xy_m, areas = arrays
    if cell_ids != stage3_grid.cell_ids:
        raise D1SpatialError("D1 and Stage-3 25 km cell identifiers differ")
    comparisons = (
        (rows, stage3_grid.rows),
        (columns, stage3_grid.columns),
        (query_xy_m, stage3_grid.query_xy_m),
        (areas, stage3_grid.clipped_area_km2),
    )
    if any(not _bitwise_equal(left, right) for left, right in comparisons):
        raise D1SpatialError("D1 and Stage-3 25 km arrays differ bitwise")
    return arrays


def build_d1_spatial_domain(study_area_wgs84: BaseGeometry) -> D1SpatialDomain:
    """Build the D1 grid solely from the frozen target-independent study area."""

    stage3_grid = build_stage3_query_grid(study_area_wgs84)
    projected = project_study_area_to_equal_area(study_area_wgs84)
    source_family = build_equal_area_grid_family(projected)
    _assert_stage3_25km_identity(stage3_grid, source_family.at(25.0))

    converted: list[SpatialGrid] = []
    for source_grid in source_family.grids:
        cell_ids, rows, columns, query_xy_m, areas = _equal_area_grid_arrays(source_grid)
        grid_id = (
            stage3_grid.grid_id
            if source_grid.spec.cell_size_km == 25.0
            else _d1_grid_id(
                cell_size_km=source_grid.spec.cell_size_km,
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
                cell_size_km=source_grid.spec.cell_size_km,
                cell_ids=cell_ids,
                rows=rows,
                columns=columns,
                query_xy_km=query_xy_m / 1_000.0,
                clipped_area_km2=areas,
            )
        )
    family = SpatialQuadratureFamily(
        grids=cast(tuple[SpatialGrid, SpatialGrid, SpatialGrid], tuple(converted))
    )
    locator = Frozen25kmCellLocator(
        grid=family.at(25.0),
        clipped_geometries=tuple(cell.clipped_geometry for cell in source_family.at(25.0).cells),
    )
    return D1SpatialDomain(
        stage3_grid=stage3_grid,
        quadrature_family=family,
        study_area_wgs84=study_area_wgs84,
        study_area_equal_area=projected,
        locator=locator,
    )


def build_d1_spatial_domain_from_bytes(study_area_geojson_bytes: bytes) -> D1SpatialDomain:
    """Parse one frozen study-area byte payload and build the same D1 domain."""

    study_area = load_study_area_bytes(study_area_geojson_bytes, EQUAL_AREA_CRS)
    return build_d1_spatial_domain(study_area.geographic)


def _issue_time_us(value: int | datetime) -> int:
    if isinstance(value, bool):
        raise TypeError("issue_time must not be bool")
    if isinstance(value, int):
        return int(value)
    if not isinstance(value, datetime):
        raise TypeError("issue_time must be epoch microseconds or a timezone-aware datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("issue_time datetime must be timezone-aware")
    delta = value.astimezone(UTC) - _UTC_EPOCH
    return (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds


@dataclass(frozen=True, slots=True)
class CausalCatalogAudit:
    """Transparent causal-filter water levels for one issue time.

    Rejection counts are independent diagnostics and can overlap.  The two
    selected counts are the exact conjunctions used to build the KDEs.
    """

    issue_time_us: int
    total_catalog_rows: int
    rejected_before_1970: int
    rejected_below_m4: int
    rejected_outside_study_area: int
    rejected_after_issue_origin: int
    rejected_unavailable_at_issue: int
    b0_source_count: int
    recent_30d_source_count: int

    def __post_init__(self) -> None:
        counts = (
            self.total_catalog_rows,
            self.rejected_before_1970,
            self.rejected_below_m4,
            self.rejected_outside_study_area,
            self.rejected_after_issue_origin,
            self.rejected_unavailable_at_issue,
            self.b0_source_count,
            self.recent_30d_source_count,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts
        ):
            raise ValueError("causal catalog audit counts must be non-negative integers")
        if self.b0_source_count > self.total_catalog_rows:
            raise ValueError("B0 source count cannot exceed the catalog row count")
        if self.recent_30d_source_count > self.b0_source_count:
            raise ValueError("recent source count cannot exceed the B0 source count")


@dataclass(frozen=True, slots=True)
class D1CausalBackground:
    """Causal B0 and recent-component masses for one issue time."""

    audit: CausalCatalogAudit
    b0_mass_25km: FloatArray
    recent_mass_25km: FloatArray
    _b0_density: NormalizedSpatialDensity = field(repr=False, compare=False)
    _recent_density: NormalizedSpatialDensity = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._b0_density.model_id != "S0":
            raise D1SpatialError("D1 B0 must reuse the Stage-2S S0 normalized KDE")
        if self._recent_density.model_id not in {"S0", "R"}:
            raise D1SpatialError("D1 recent component must be R or the exact empty-window B0")
        if self._b0_density.grid_family is not self._recent_density.grid_family:
            raise D1SpatialError("D1 background components must share one grid family")
        expected = self._b0_density.grid_family.at(25.0).cell_count
        b0 = _readonly_float(self.b0_mass_25km)
        recent = _readonly_float(self.recent_mass_25km)
        if b0.size != expected or recent.size != expected:
            raise D1SpatialError("D1 background masses must match the 25 km grid")
        for label, values in (("B0", b0), ("recent", recent)):
            if np.any(values < 0.0) or not math.isclose(
                math.fsum(float(item) for item in values),
                1.0,
                rel_tol=0.0,
                abs_tol=MASS_SUM_ABSOLUTE_TOLERANCE,
            ):
                raise D1SpatialError(f"{label} 25 km mass must be normalized and non-negative")
        object.__setattr__(self, "b0_mass_25km", b0)
        object.__setattr__(self, "recent_mass_25km", recent)

    def mass_for_alpha(self, alpha: float) -> FloatArray:
        """Return the frozen R30 mixture for one preregistered alpha candidate."""

        weight = float(alpha)
        if weight not in FROZEN_R30_ALPHA_CANDIDATES:
            raise ValueError("alpha must be one of 0, 0.25, 0.5, or 0.75")
        mixed = mix_density(
            self._b0_density,
            self._recent_density,
            weight,
            model_id="S1",
        )
        return _readonly_float(mixed.mass_25km)


def _project_selected_catalog(
    catalog: Stage2SEarthquakeCatalog,
    mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    transformer = Transformer.from_crs(
        CRS.from_epsg(4326),
        CRS.from_user_input(EQUAL_AREA_CRS),
        always_xy=True,
    )
    x_m, y_m = transformer.transform(catalog.longitude[mask], catalog.latitude[mask])
    result = np.column_stack(
        (
            np.asarray(x_m, dtype=np.float64) / 1_000.0,
            np.asarray(y_m, dtype=np.float64) / 1_000.0,
        )
    )
    if result.ndim != 2 or result.shape[1:] != (2,) or not np.isfinite(result).all():
        raise D1SpatialError("causal catalog projection produced invalid coordinates")
    return result


def build_causal_background_components(
    catalog: Stage2SEarthquakeCatalog,
    issue_time: int | datetime,
    domain: D1SpatialDomain,
) -> D1CausalBackground:
    """Build B0 and ``(T-30d, T]`` recent masses without future information."""

    if not isinstance(catalog, Stage2SEarthquakeCatalog):
        raise TypeError("catalog must be a verified Stage2SEarthquakeCatalog")
    if not isinstance(domain, D1SpatialDomain):
        raise TypeError("domain must be a D1SpatialDomain")
    issue_us = _issue_time_us(issue_time)
    origin = catalog.origin_time_us
    available = catalog.available_at_us
    after_start = origin >= _CATALOG_START_US
    large_enough = catalog.magnitude >= 4.0
    inside = catalog.inside_study_area
    origin_causal = origin <= issue_us
    available_causal = available <= issue_us
    b0_mask = after_start & large_enough & inside & origin_causal & available_causal
    recent_mask = b0_mask & (origin > issue_us - _R30_MICROSECONDS)
    b0_count = int(np.count_nonzero(b0_mask))
    recent_count = int(np.count_nonzero(recent_mask))
    audit = CausalCatalogAudit(
        issue_time_us=issue_us,
        total_catalog_rows=catalog.row_count,
        rejected_before_1970=int(np.count_nonzero(~after_start)),
        rejected_below_m4=int(np.count_nonzero(~large_enough)),
        rejected_outside_study_area=int(np.count_nonzero(~inside)),
        rejected_after_issue_origin=int(np.count_nonzero(~origin_causal)),
        rejected_unavailable_at_issue=int(np.count_nonzero(~available_causal)),
        b0_source_count=b0_count,
        recent_30d_source_count=recent_count,
    )
    if b0_count == 0:
        raise D1SpatialError("B0 has no causal 1970+ M4+ inside-study-area source events")

    b0_xy_km = _project_selected_catalog(catalog, b0_mask)
    unmatched = sum(
        domain.locator.locate_projected(float(x_km) * 1_000.0, float(y_km) * 1_000.0) is None
        for x_km, y_km in b0_xy_km
    )
    if unmatched:
        raise D1SpatialError(
            f"{unmatched} catalog rows flagged inside the study area miss every clipped cell"
        )
    b0 = build_normalized_kde(
        b0_xy_km,
        domain.quadrature_family,
        model_id="S0",
    )
    recent_xy_km = _project_selected_catalog(catalog, recent_mask)
    recent = build_recent_component(
        recent_xy_km,
        domain.quadrature_family,
        component_id="R",
        empty_fallback_s0=b0,
    )
    return D1CausalBackground(
        audit=audit,
        b0_mass_25km=b0.mass_25km,
        recent_mass_25km=recent.mass_25km,
        _b0_density=b0,
        _recent_density=recent,
    )


@dataclass(frozen=True, slots=True)
class D1AlarmPrefix:
    """One complete-cell alarm prefix at a preregistered area budget."""

    budget_km2: float
    actual_area_km2: float
    selected_indices: IntArray
    selected_cell_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        budget = float(self.budget_km2)
        actual = float(self.actual_area_km2)
        if budget not in FROZEN_D1_AREA_BUDGETS_KM2:
            raise ValueError("alarm budget is not one of the five preregistered area levels")
        if not math.isfinite(actual) or actual < 0.0 or actual > budget:
            raise ValueError("actual alarm area must be finite and within budget")
        indices = _readonly_int(self.selected_indices)
        identifiers = tuple(self.selected_cell_ids)
        if indices.size != len(identifiers):
            raise ValueError("selected indices and cell identifiers must have one length")
        if np.any(indices < 0) or len(set(int(value) for value in indices)) != indices.size:
            raise ValueError("selected alarm indices must be unique and non-negative")
        if len(set(identifiers)) != len(identifiers) or any(not value for value in identifiers):
            raise ValueError("selected alarm cell identifiers must be non-empty and unique")
        object.__setattr__(self, "budget_km2", budget)
        object.__setattr__(self, "actual_area_km2", actual)
        object.__setattr__(self, "selected_indices", indices)
        object.__setattr__(self, "selected_cell_ids", identifiers)


def select_alarm_prefixes(
    mass_25km: object,
    grid: SpatialGrid,
    *,
    area_budgets_km2: tuple[float, ...] = FROZEN_D1_AREA_BUDGETS_KM2,
) -> tuple[D1AlarmPrefix, ...]:
    """Rank by mass/area once and return all five no-skip complete prefixes."""

    budgets = tuple(float(value) for value in area_budgets_km2)
    if budgets != FROZEN_D1_AREA_BUDGETS_KM2:
        raise ValueError("D1 requires exactly the five preregistered area budgets")
    if not isinstance(grid, SpatialGrid) or grid.cell_size_km != 25.0:
        raise TypeError("alarm grid must be the frozen 25 km SpatialGrid")
    mass = _readonly_float(mass_25km)
    if mass.size != grid.cell_count or np.any(mass < 0.0):
        raise ValueError("alarm mass must be non-negative and match the 25 km grid")
    if not math.isclose(
        math.fsum(float(value) for value in mass),
        1.0,
        rel_tol=0.0,
        abs_tol=MASS_SUM_ABSOLUTE_TOLERANCE,
    ):
        raise ValueError("alarm mass must sum to one")
    intensity = np.asarray(mass / grid.clipped_area_km2, dtype=np.float64)
    if not np.isfinite(intensity).all() or np.any(intensity < 0.0):
        raise ValueError("alarm mass/area intensity must be finite and non-negative")
    ranking = tuple(
        sorted(
            range(grid.cell_count),
            key=lambda index: (
                -float(intensity[index]),
                int(grid.rows[index]),
                int(grid.columns[index]),
                grid.cell_ids[index].encode("utf-8"),
            ),
        )
    )

    prefixes: list[D1AlarmPrefix] = []
    for budget in budgets:
        selected: list[int] = []
        selected_areas: list[float] = []
        for index in ranking:
            candidate_areas = (*selected_areas, float(grid.clipped_area_km2[index]))
            if math.fsum(candidate_areas) > budget:
                break
            selected.append(index)
            selected_areas.append(float(grid.clipped_area_km2[index]))
        prefixes.append(
            D1AlarmPrefix(
                budget_km2=budget,
                actual_area_km2=math.fsum(selected_areas),
                selected_indices=np.asarray(selected, dtype=np.int64),
                selected_cell_ids=tuple(grid.cell_ids[index] for index in selected),
            )
        )
    return tuple(prefixes)


__all__ = [
    "FROZEN_D1_AREA_BUDGETS_KM2",
    "FROZEN_R30_ALPHA_CANDIDATES",
    "CausalCatalogAudit",
    "D1AlarmPrefix",
    "D1CausalBackground",
    "D1SpatialDomain",
    "D1SpatialError",
    "Frozen25kmCellLocator",
    "build_causal_background_components",
    "build_d1_spatial_domain",
    "build_d1_spatial_domain_from_bytes",
    "select_alarm_prefixes",
]
