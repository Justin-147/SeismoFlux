"""Deterministic, target-independent equal-area grids for stage 2.

The study-area geometry is the only spatial input accepted by the grid builders.
Earthquake locations, scores, and model intensities therefore cannot influence cell
placement or refinement.  All geometry operations use the frozen China Albers
equal-area projection and the three preregistered cell sizes.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import numpy as np
from pyproj import CRS, Transformer
from shapely import (
    area as vector_area,
)
from shapely import (
    box as vector_box,
)
from shapely import (
    covers as vector_covers,
)
from shapely import (
    from_wkb,
    get_x,
    get_y,
    point_on_surface,
    prepare,
    to_wkb,
)
from shapely import (
    intersection as vector_intersection,
)
from shapely import (
    intersects as vector_intersects,
)
from shapely import (
    is_empty as vector_is_empty,
)
from shapely import (
    is_valid as vector_is_valid,
)
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform

EQUAL_AREA_CRS = (
    "+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 +datum=WGS84 +units=m +no_defs +type=crs"
)
GRID_CELL_SIZES_KM = (50.0, 25.0, 12.5)
GRID_ORIGIN_M = (0.0, 0.0)
RELATIVE_EXPECTED_COUNT_TOLERANCE = 0.02
DENSITY_L1_TOLERANCE = 0.05
RELATIVE_DENOMINATOR_FLOOR = 1.0e-12

_CELL_SIZE_MM = {
    50.0: 50_000_000,
    25.0: 25_000_000,
    12.5: 12_500_000,
}
_MAX_SIGNED_INDEX = 9_999_999
_VECTOR_CHUNK_SIZE = 4_096


class GridConvergenceError(RuntimeError):
    """Raised when a preregistered grid-convergence gate does not pass."""


def _validate_polygonal_geometry(geometry: BaseGeometry, *, label: str) -> None:
    if not isinstance(geometry, BaseGeometry):
        raise TypeError(f"{label} must be a Shapely geometry")
    if geometry.geom_type not in {"Polygon", "MultiPolygon"}:
        raise ValueError(f"{label} must be polygonal")
    if geometry.is_empty or not geometry.is_valid:
        raise ValueError(f"{label} must be non-empty and valid")
    if not math.isfinite(float(geometry.area)) or geometry.area <= 0.0:
        raise ValueError(f"{label} must have finite positive area")
    if not all(math.isfinite(float(value)) for value in geometry.bounds):
        raise ValueError(f"{label} bounds must be finite")


def _validate_index(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if abs(value) > _MAX_SIGNED_INDEX:
        raise ValueError(f"{name} exceeds the fixed signed eight-character field")
    return value


@dataclass(frozen=True, slots=True)
class GridSpec:
    """One of the three fixed grids sharing the origin at ``(0, 0)`` metres."""

    cell_size_km: float

    def __post_init__(self) -> None:
        value = float(self.cell_size_km)
        if value not in _CELL_SIZE_MM:
            raise ValueError("cell_size_km must be exactly 50, 25, or 12.5")
        object.__setattr__(self, "cell_size_km", value)

    @property
    def cell_size_m(self) -> float:
        return self.cell_size_km * 1_000.0

    @property
    def cell_size_mm(self) -> int:
        return _CELL_SIZE_MM[self.cell_size_km]

    @property
    def origin_x_m(self) -> float:
        return GRID_ORIGIN_M[0]

    @property
    def origin_y_m(self) -> float:
        return GRID_ORIGIN_M[1]


def cell_id(spec: GridSpec, *, row: int, column: int) -> str:
    """Return the fixed integer-millimetre ID for one signed row/column pair."""

    row_value = _validate_index("row", row)
    column_value = _validate_index("column", column)
    return f"g{spec.cell_size_mm:08d}_r{row_value:+08d}_c{column_value:+08d}"


def point_cell_index(x_m: float, y_m: float, spec: GridSpec) -> tuple[int, int]:
    """Map a projected point to ``(row, column)`` using half-open cells and floor."""

    x_value = float(x_m)
    y_value = float(y_m)
    if not math.isfinite(x_value) or not math.isfinite(y_value):
        raise ValueError("projected point coordinates must be finite")
    column = math.floor((x_value - spec.origin_x_m) / spec.cell_size_m)
    row = math.floor((y_value - spec.origin_y_m) / spec.cell_size_m)
    return _validate_index("row", row), _validate_index("column", column)


def cell_bounds(spec: GridSpec, *, row: int, column: int) -> tuple[float, float, float, float]:
    """Return the left/bottom-closed, right/top-open projected cell bounds."""

    row_value = _validate_index("row", row)
    column_value = _validate_index("column", column)
    minimum_x = spec.origin_x_m + column_value * spec.cell_size_m
    minimum_y = spec.origin_y_m + row_value * spec.cell_size_m
    return (
        minimum_x,
        minimum_y,
        minimum_x + spec.cell_size_m,
        minimum_y + spec.cell_size_m,
    )


@dataclass(frozen=True, slots=True)
class GridCell:
    """One fixed cell clipped exactly to the target-independent study area."""

    spec: GridSpec
    row: int
    column: int
    clipped_geometry: BaseGeometry
    representative_point: Point
    clipped_area_m2: float

    def __post_init__(self) -> None:
        _validate_index("row", self.row)
        _validate_index("column", self.column)
        area = float(self.clipped_area_m2)
        if not math.isfinite(area) or area <= 0.0:
            raise ValueError("clipped cell area must be finite and positive")
        if self.clipped_geometry.is_empty or not self.clipped_geometry.is_valid:
            raise ValueError("clipped cell geometry must be non-empty and valid")
        if float(self.clipped_geometry.area) != area:
            raise ValueError("clipped cell area must equal the exact projected geometry area")
        if self.representative_point.is_empty or not self.clipped_geometry.covers(
            self.representative_point
        ):
            raise ValueError("representative point must be covered by the clipped geometry")
        if not math.isfinite(float(self.representative_point.x)) or not math.isfinite(
            float(self.representative_point.y)
        ):
            raise ValueError("representative point coordinates must be finite")
        object.__setattr__(self, "clipped_area_m2", area)

    @property
    def id(self) -> str:
        return cell_id(self.spec, row=self.row, column=self.column)

    @property
    def bounds_m(self) -> tuple[float, float, float, float]:
        return cell_bounds(self.spec, row=self.row, column=self.column)


def _prevalidated_grid_cell(
    *,
    spec: GridSpec,
    row: int,
    column: int,
    clipped_geometry: BaseGeometry,
    representative_point: Point,
    clipped_area_m2: float,
) -> GridCell:
    """Construct a cell after its whole vector chunk has passed the same checks."""

    cell = object.__new__(GridCell)
    object.__setattr__(cell, "spec", spec)
    object.__setattr__(cell, "row", row)
    object.__setattr__(cell, "column", column)
    object.__setattr__(cell, "clipped_geometry", clipped_geometry)
    object.__setattr__(cell, "representative_point", representative_point)
    object.__setattr__(cell, "clipped_area_m2", clipped_area_m2)
    return cell


@dataclass(frozen=True, slots=True)
class EqualAreaGrid:
    """A deterministic row/column-ordered clipped grid for one resolution."""

    spec: GridSpec
    study_area_equal_area: BaseGeometry
    cells: tuple[GridCell, ...]

    def __post_init__(self) -> None:
        _validate_polygonal_geometry(
            self.study_area_equal_area,
            label="equal-area study area",
        )
        if not self.cells:
            raise ValueError("an equal-area grid must contain at least one positive-area cell")
        order = tuple((cell.row, cell.column) for cell in self.cells)
        if order != tuple(sorted(order)) or len(order) != len(set(order)):
            raise ValueError("grid cells must have unique row/column pairs in sorted order")
        if any(cell.spec != self.spec for cell in self.cells):
            raise ValueError("every cell must use its containing grid specification")

    @property
    def cell_ids(self) -> tuple[str, ...]:
        return tuple(cell.id for cell in self.cells)


@dataclass(frozen=True, slots=True)
class EqualAreaGridFamily:
    """The frozen 50/25/12.5 km grid family built from one study geometry."""

    study_area_equal_area: BaseGeometry
    grids: tuple[EqualAreaGrid, ...]

    def __post_init__(self) -> None:
        if tuple(grid.spec.cell_size_km for grid in self.grids) != GRID_CELL_SIZES_KM:
            raise ValueError("grid family must contain 50, 25, and 12.5 km grids in order")
        if any(
            not grid.study_area_equal_area.equals(self.study_area_equal_area) for grid in self.grids
        ):
            raise ValueError("all grid-family members must use the same study area")

    def at(self, cell_size_km: float) -> EqualAreaGrid:
        requested = float(cell_size_km)
        for grid in self.grids:
            if grid.spec.cell_size_km == requested:
                return grid
        raise KeyError(f"grid family has no {requested:g} km grid")


def project_study_area_to_equal_area(study_area_wgs84: BaseGeometry) -> BaseGeometry:
    """Project a WGS84 study polygon to the frozen China Albers equal-area CRS."""

    _validate_polygonal_geometry(study_area_wgs84, label="WGS84 study area")
    transformer = Transformer.from_crs(
        CRS.from_epsg(4326),
        CRS.from_user_input(EQUAL_AREA_CRS),
        always_xy=True,
    )
    projected = cast(BaseGeometry, transform(transformer.transform, study_area_wgs84))
    _validate_polygonal_geometry(projected, label="projected study area")
    return projected


def build_clipped_grid(
    study_area_equal_area: BaseGeometry,
    *,
    cell_size_km: float,
) -> EqualAreaGrid:
    """Build a full fixed grid using only the projected study-area geometry."""

    _validate_polygonal_geometry(study_area_equal_area, label="equal-area study area")
    spec = GridSpec(cell_size_km)
    minimum_x, minimum_y, maximum_x, maximum_y = study_area_equal_area.bounds
    minimum_row, minimum_column = point_cell_index(minimum_x, minimum_y, spec)
    maximum_row, maximum_column = point_cell_index(maximum_x, maximum_y, spec)

    column_count = maximum_column - minimum_column + 1
    row_count = maximum_row - minimum_row + 1
    candidate_count = row_count * column_count
    predicate_geometry = cast(
        BaseGeometry,
        from_wkb(to_wkb(study_area_equal_area)),
    )
    prepare(predicate_geometry)

    cells: list[GridCell] = []
    for start in range(0, candidate_count, _VECTOR_CHUNK_SIZE):
        stop = min(start + _VECTOR_CHUNK_SIZE, candidate_count)
        flat_indices = np.arange(start, stop, dtype=np.int64)
        rows = minimum_row + flat_indices // column_count
        columns = minimum_column + flat_indices % column_count
        minimum_x_values = spec.origin_x_m + columns.astype(np.float64) * spec.cell_size_m
        minimum_y_values = spec.origin_y_m + rows.astype(np.float64) * spec.cell_size_m
        candidate_boxes = vector_box(
            minimum_x_values,
            minimum_y_values,
            minimum_x_values + spec.cell_size_m,
            minimum_y_values + spec.cell_size_m,
        )

        covered = np.asarray(
            vector_covers(predicate_geometry, candidate_boxes),
            dtype=np.bool_,
        )
        selected = covered.copy()
        boundary_or_outside = np.flatnonzero(~covered)
        if boundary_or_outside.size:
            selected[boundary_or_outside] = np.asarray(
                vector_intersects(
                    predicate_geometry,
                    candidate_boxes[boundary_or_outside],
                ),
                dtype=np.bool_,
            )
        selected_positions = np.flatnonzero(selected)
        if not selected_positions.size:
            continue

        selected_boxes = candidate_boxes[selected_positions]
        selected_covered = covered[selected_positions]
        clipped_geometries = selected_boxes.copy()
        boundary_positions = np.flatnonzero(~selected_covered)
        if boundary_positions.size:
            clipped_geometries[boundary_positions] = vector_intersection(
                study_area_equal_area,
                selected_boxes[boundary_positions],
            )

        areas = np.asarray(vector_area(clipped_geometries), dtype=np.float64)
        positive_area = (
            ~np.asarray(vector_is_empty(clipped_geometries), dtype=np.bool_)
            & np.isfinite(areas)
            & (areas > 0.0)
        )
        if not np.any(positive_area):
            continue
        clipped_geometries = clipped_geometries[positive_area]
        areas = areas[positive_area]
        selected_positions = selected_positions[positive_area]
        points = point_on_surface(clipped_geometries)

        geometry_valid = np.asarray(vector_is_valid(clipped_geometries), dtype=np.bool_)
        points_covered = np.asarray(
            vector_covers(clipped_geometries, points),
            dtype=np.bool_,
        )
        point_x = np.asarray(get_x(points), dtype=np.float64)
        point_y = np.asarray(get_y(points), dtype=np.float64)
        if not (
            np.all(geometry_valid)
            and np.all(points_covered)
            and np.all(np.isfinite(point_x))
            and np.all(np.isfinite(point_y))
        ):
            raise ValueError("vectorized clipped-cell geometry validation failed")

        selected_rows = rows[selected_positions]
        selected_columns = columns[selected_positions]
        for row, column, clipped, point, area_m2 in zip(
            selected_rows,
            selected_columns,
            clipped_geometries,
            points,
            areas,
            strict=True,
        ):
            cells.append(
                _prevalidated_grid_cell(
                    spec=spec,
                    row=int(row),
                    column=int(column),
                    clipped_geometry=cast(BaseGeometry, clipped),
                    representative_point=cast(Point, point),
                    clipped_area_m2=float(area_m2),
                )
            )
    return EqualAreaGrid(
        spec=spec,
        study_area_equal_area=study_area_equal_area,
        cells=tuple(cells),
    )


def build_equal_area_grid_family(
    study_area_equal_area: BaseGeometry,
) -> EqualAreaGridFamily:
    """Build all three fixed grids from an already projected equal-area domain."""

    _validate_polygonal_geometry(study_area_equal_area, label="equal-area study area")
    grids = tuple(
        build_clipped_grid(study_area_equal_area, cell_size_km=cell_size_km)
        for cell_size_km in GRID_CELL_SIZES_KM
    )
    return EqualAreaGridFamily(
        study_area_equal_area=study_area_equal_area,
        grids=grids,
    )


def build_grid_family(study_area_wgs84: BaseGeometry) -> EqualAreaGridFamily:
    """Project one study polygon and build all three target-independent grids."""

    projected = project_study_area_to_equal_area(study_area_wgs84)
    return build_equal_area_grid_family(projected)


def _validated_ordered_masses(
    grid: EqualAreaGrid,
    masses: Mapping[str, float],
    *,
    label: str,
) -> tuple[float, ...]:
    expected_ids = grid.cell_ids
    if set(masses) != set(expected_ids):
        raise ValueError(f"{label} keys must exactly match the grid cell IDs")
    ordered: list[float] = []
    for identifier in expected_ids:
        raw_value = masses[identifier]
        if isinstance(raw_value, bool):
            raise TypeError(f"{label} values must be numeric masses")
        value = float(raw_value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{label} values must be finite and non-negative")
        ordered.append(value)
    return tuple(ordered)


def _aligned_ratio(coarse: EqualAreaGrid, fine: EqualAreaGrid) -> int:
    coarse_mm = coarse.spec.cell_size_mm
    fine_mm = fine.spec.cell_size_mm
    if coarse_mm <= fine_mm or coarse_mm % fine_mm != 0:
        raise ValueError("coarse and fine grids must have an integer aligned size ratio")
    if not coarse.study_area_equal_area.equals(fine.study_area_equal_area):
        raise ValueError("coarse and fine grids must use the same study area")
    return coarse_mm // fine_mm


def aggregate_fine_masses_to_coarse(
    coarse: EqualAreaGrid,
    fine: EqualAreaGrid,
    fine_masses: Mapping[str, float],
) -> dict[str, float]:
    """Aggregate fine-cell masses to their aligned coarse parents deterministically."""

    ratio = _aligned_ratio(coarse, fine)
    ordered_masses = _validated_ordered_masses(fine, fine_masses, label="fine masses")
    grouped: dict[str, list[float]] = {identifier: [] for identifier in coarse.cell_ids}
    for fine_cell, mass in zip(fine.cells, ordered_masses, strict=True):
        parent_id = cell_id(
            coarse.spec,
            row=fine_cell.row // ratio,
            column=fine_cell.column // ratio,
        )
        try:
            grouped[parent_id].append(mass)
        except KeyError as error:
            raise ValueError("fine cell has no positive-area parent in the coarse grid") from error
    return {identifier: math.fsum(grouped[identifier]) for identifier in coarse.cell_ids}


@dataclass(frozen=True, slots=True)
class GridConvergenceDiagnostics:
    """Frozen 2% total-count and 5% normalized-density convergence diagnostics."""

    coarse_cell_size_km: float
    fine_cell_size_km: float
    coarse_total: float
    fine_total: float
    relative_expected_count_difference: float
    density_l1_difference: float

    @property
    def passed(self) -> bool:
        return (
            self.relative_expected_count_difference <= RELATIVE_EXPECTED_COUNT_TOLERANCE
            and self.density_l1_difference <= DENSITY_L1_TOLERANCE
        )


@dataclass(frozen=True, slots=True)
class ThreeGridConvergenceGateEvidence:
    """Complete diagnostic and primary evidence for the frozen three-grid gate."""

    diagnostic_50_to_25: GridConvergenceDiagnostics
    primary_25_to_12_5: GridConvergenceDiagnostics

    def __post_init__(self) -> None:
        observed_pairs = tuple(
            (diagnostics.coarse_cell_size_km, diagnostics.fine_cell_size_km)
            for diagnostics in self.comparisons
        )
        if observed_pairs != ((50.0, 25.0), (25.0, 12.5)):
            raise ValueError(
                "three-grid evidence must contain exactly 50-to-25 diagnostic and "
                "25-to-12.5 primary comparisons"
            )

    @property
    def comparisons(
        self,
    ) -> tuple[GridConvergenceDiagnostics, GridConvergenceDiagnostics]:
        return (self.diagnostic_50_to_25, self.primary_25_to_12_5)

    @property
    def passed(self) -> bool:
        return all(diagnostics.passed for diagnostics in self.comparisons)


def diagnose_grid_convergence(
    coarse: EqualAreaGrid,
    fine: EqualAreaGrid,
    coarse_masses: Mapping[str, float],
    fine_masses: Mapping[str, float],
) -> GridConvergenceDiagnostics:
    """Compare a coarse grid with fine masses aggregated to common parent cells."""

    _aligned_ratio(coarse, fine)
    ordered_coarse = _validated_ordered_masses(coarse, coarse_masses, label="coarse masses")
    aggregated_fine = aggregate_fine_masses_to_coarse(coarse, fine, fine_masses)
    ordered_fine = tuple(aggregated_fine[identifier] for identifier in coarse.cell_ids)
    coarse_total = math.fsum(ordered_coarse)
    fine_total = math.fsum(ordered_fine)

    if coarse_total < RELATIVE_DENOMINATOR_FLOOR and fine_total < RELATIVE_DENOMINATOR_FLOOR:
        relative_difference = 0.0
    else:
        relative_difference = abs(coarse_total - fine_total) / max(
            abs(fine_total),
            RELATIVE_DENOMINATOR_FLOOR,
        )

    density_l1 = math.fsum(
        abs(
            (coarse_mass / coarse_total if coarse_total > 0.0 else 0.0)
            - (fine_mass / fine_total if fine_total > 0.0 else 0.0)
        )
        for coarse_mass, fine_mass in zip(ordered_coarse, ordered_fine, strict=True)
    )
    return GridConvergenceDiagnostics(
        coarse_cell_size_km=coarse.spec.cell_size_km,
        fine_cell_size_km=fine.spec.cell_size_km,
        coarse_total=coarse_total,
        fine_total=fine_total,
        relative_expected_count_difference=relative_difference,
        density_l1_difference=density_l1,
    )


def diagnose_three_grid_convergence(
    grid_family: EqualAreaGridFamily,
    masses_by_cell_size_km: Mapping[float, Mapping[str, float]],
) -> ThreeGridConvergenceGateEvidence:
    """Compute both frozen convergence pairs from one complete three-grid mass set."""

    normalized_masses: dict[float, Mapping[str, float]] = {}
    for raw_cell_size, masses in masses_by_cell_size_km.items():
        cell_size = float(raw_cell_size)
        if cell_size in normalized_masses:
            raise ValueError("grid mass resolutions must be unique")
        normalized_masses[cell_size] = masses
    if tuple(sorted(normalized_masses, reverse=True)) != GRID_CELL_SIZES_KM:
        raise ValueError("grid masses must contain exactly 50, 25, and 12.5 km resolutions")

    grid_50 = grid_family.at(50.0)
    grid_25 = grid_family.at(25.0)
    grid_12_5 = grid_family.at(12.5)
    diagnostic = diagnose_grid_convergence(
        grid_50,
        grid_25,
        normalized_masses[50.0],
        normalized_masses[25.0],
    )
    primary = diagnose_grid_convergence(
        grid_25,
        grid_12_5,
        normalized_masses[25.0],
        normalized_masses[12.5],
    )
    return ThreeGridConvergenceGateEvidence(
        diagnostic_50_to_25=diagnostic,
        primary_25_to_12_5=primary,
    )


def require_grid_convergence(diagnostics: GridConvergenceDiagnostics) -> None:
    """Fail the model gate when either preregistered convergence threshold is exceeded."""

    if not diagnostics.passed:
        raise GridConvergenceError(
            "grid convergence failed: "
            f"relative_expected_count={diagnostics.relative_expected_count_difference:.17g} "
            f"(limit {RELATIVE_EXPECTED_COUNT_TOLERANCE:g}), "
            f"density_l1={diagnostics.density_l1_difference:.17g} "
            f"(limit {DENSITY_L1_TOLERANCE:g})"
        )


def require_three_grid_convergence(evidence: ThreeGridConvergenceGateEvidence) -> None:
    """Fail unless both the diagnostic and primary frozen grid comparisons pass."""

    if evidence.passed:
        return

    pair_messages = []
    for diagnostics in evidence.comparisons:
        pair_messages.append(
            f"{diagnostics.coarse_cell_size_km:g}->{diagnostics.fine_cell_size_km:g}km "
            f"passed={diagnostics.passed} "
            f"coarse_total={diagnostics.coarse_total:.17g} "
            f"fine_total={diagnostics.fine_total:.17g} "
            "relative_expected_count="
            f"{diagnostics.relative_expected_count_difference:.17g} "
            f"density_l1={diagnostics.density_l1_difference:.17g}"
        )
    raise GridConvergenceError("three-grid convergence failed: " + "; ".join(pair_messages))
