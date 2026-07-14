from __future__ import annotations

import math

import pytest
from shapely.geometry import MultiPolygon, Polygon, box

from seismoflux.background.grid import (
    DENSITY_L1_TOLERANCE,
    EQUAL_AREA_CRS,
    GRID_CELL_SIZES_KM,
    RELATIVE_DENOMINATOR_FLOOR,
    RELATIVE_EXPECTED_COUNT_TOLERANCE,
    EqualAreaGrid,
    EqualAreaGridFamily,
    GridConvergenceError,
    GridSpec,
    ThreeGridConvergenceGateEvidence,
    aggregate_fine_masses_to_coarse,
    build_clipped_grid,
    build_grid_family,
    cell_bounds,
    cell_id,
    diagnose_grid_convergence,
    diagnose_three_grid_convergence,
    point_cell_index,
    project_study_area_to_equal_area,
    require_grid_convergence,
    require_three_grid_convergence,
)


def _aligned_grids() -> tuple[EqualAreaGrid, EqualAreaGrid]:
    study_area = box(-25_000.0, -25_000.0, 25_000.0, 25_000.0)
    return (
        build_clipped_grid(study_area, cell_size_km=25.0),
        build_clipped_grid(study_area, cell_size_km=12.5),
    )


def _three_grid_family() -> EqualAreaGridFamily:
    study_area = box(-50_000.0, -50_000.0, 50_000.0, 50_000.0)
    grids = tuple(
        build_clipped_grid(study_area, cell_size_km=cell_size_km)
        for cell_size_km in GRID_CELL_SIZES_KM
    )
    return EqualAreaGridFamily(study_area_equal_area=study_area, grids=grids)


def _passing_three_grid_masses(
    family: EqualAreaGridFamily,
) -> dict[float, dict[str, float]]:
    return {
        50.0: {identifier: 16.0 for identifier in family.at(50.0).cell_ids},
        25.0: {identifier: 4.0 for identifier in family.at(25.0).cell_ids},
        12.5: {identifier: 1.0 for identifier in family.at(12.5).cell_ids},
    }


def _naive_clipped_cells(
    study_area: Polygon | MultiPolygon,
    spec: GridSpec,
) -> list[tuple[int, int, object, object, float]]:
    minimum_x, minimum_y, maximum_x, maximum_y = study_area.bounds
    minimum_row, minimum_column = point_cell_index(minimum_x, minimum_y, spec)
    maximum_row, maximum_column = point_cell_index(maximum_x, maximum_y, spec)
    result: list[tuple[int, int, object, object, float]] = []
    for row in range(minimum_row, maximum_row + 1):
        for column in range(minimum_column, maximum_column + 1):
            clipped = study_area.intersection(box(*cell_bounds(spec, row=row, column=column)))
            if clipped.is_empty or clipped.area <= 0.0:
                continue
            result.append((row, column, clipped, clipped.representative_point(), clipped.area))
    return result


def test_grid_constants_are_the_frozen_protocol_values() -> None:
    assert EQUAL_AREA_CRS == (
        "+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 +datum=WGS84 +units=m +no_defs +type=crs"
    )
    assert GRID_CELL_SIZES_KM == (50.0, 25.0, 12.5)
    assert RELATIVE_EXPECTED_COUNT_TOLERANCE == 0.02
    assert DENSITY_L1_TOLERANCE == 0.05
    assert RELATIVE_DENOMINATOR_FLOOR == 1.0e-12


def test_grid_spec_rejects_non_preregistered_resolution() -> None:
    assert GridSpec(12.5).cell_size_mm == 12_500_000
    assert GridSpec(25).cell_size_m == 25_000.0
    with pytest.raises(ValueError, match="exactly 50, 25, or 12.5"):
        GridSpec(10.0)


def test_floor_indices_half_open_bounds_and_integer_millimetre_ids_are_stable() -> None:
    spec = GridSpec(12.5)
    assert point_cell_index(0.0, 0.0, spec) == (0, 0)
    assert point_cell_index(12_499.999, 12_499.999, spec) == (0, 0)
    assert point_cell_index(12_500.0, 12_500.0, spec) == (1, 1)
    assert point_cell_index(-0.001, -0.001, spec) == (-1, -1)
    assert point_cell_index(-12_500.0, -12_500.0, spec) == (-1, -1)
    assert point_cell_index(-12_500.001, -12_500.001, spec) == (-2, -2)

    assert cell_bounds(spec, row=-2, column=3) == (37_500.0, -25_000.0, 50_000.0, -12_500.0)
    assert cell_id(spec, row=-2, column=3) == "g12500000_r-0000002_c+0000003"


def test_clipped_grid_uses_positive_area_intersections_and_stable_row_column_order() -> None:
    study_area = box(-5_000.0, -5_000.0, 30_000.0, 30_000.0)
    first = build_clipped_grid(study_area, cell_size_km=25.0)
    second = build_clipped_grid(study_area, cell_size_km=25.0)

    assert [(cell.row, cell.column) for cell in first.cells] == [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 0),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]
    assert first.cell_ids == second.cell_ids
    assert [cell.clipped_geometry.wkb for cell in first.cells] == [
        cell.clipped_geometry.wkb for cell in second.cells
    ]
    assert math.fsum(cell.clipped_area_m2 for cell in first.cells) == study_area.area
    for cell in first.cells:
        assert cell.clipped_area_m2 == cell.clipped_geometry.area
        assert cell.clipped_geometry.covers(cell.representative_point)
        assert cell.representative_point.equals(cell.clipped_geometry.representative_point())


def test_exact_outer_boundary_does_not_create_zero_area_neighbor_cells() -> None:
    grid = build_clipped_grid(box(0.0, 0.0, 25_000.0, 25_000.0), cell_size_km=25.0)

    assert [(cell.row, cell.column) for cell in grid.cells] == [(0, 0)]
    assert grid.cells[0].clipped_area_m2 == 625_000_000.0


def test_vectorized_chunking_is_cellwise_equivalent_to_naive_intersection() -> None:
    first = Polygon(
        [
            (-10_000.0, -10_000.0),
            (62_000.0, -10_000.0),
            (62_000.0, 58_000.0),
            (-10_000.0, 58_000.0),
            (-10_000.0, -10_000.0),
        ],
        holes=[
            [
                (15_000.0, 15_000.0),
                (30_000.0, 15_000.0),
                (30_000.0, 30_000.0),
                (15_000.0, 30_000.0),
                (15_000.0, 15_000.0),
            ]
        ],
    )
    second = box(70_000.0, 5_000.0, 82_000.0, 17_000.0)
    study_area = MultiPolygon([first, second])
    spec = GridSpec(25.0)

    vectorized = build_clipped_grid(study_area, cell_size_km=25.0)
    naive = _naive_clipped_cells(study_area, spec)

    assert [(cell.row, cell.column) for cell in vectorized.cells] == [
        (row, column) for row, column, _, _, _ in naive
    ]
    for cell, (_, _, naive_geometry, naive_point, naive_area) in zip(
        vectorized.cells,
        naive,
        strict=True,
    ):
        assert cell.clipped_geometry.equals(naive_geometry)
        assert cell.clipped_area_m2 == naive_area
        assert cell.representative_point.equals(naive_point)


def test_wgs84_projection_and_family_depend_only_on_the_study_polygon() -> None:
    study_area_wgs84 = Polygon(
        [(104.9, 34.9), (105.1, 34.9), (105.1, 35.1), (104.9, 35.1), (104.9, 34.9)]
    )
    projected = project_study_area_to_equal_area(study_area_wgs84)
    family = build_grid_family(study_area_wgs84)

    assert projected.is_valid and projected.area > 0.0
    assert tuple(grid.spec.cell_size_km for grid in family.grids) == GRID_CELL_SIZES_KM
    assert family.at(25.0).study_area_equal_area.equals(family.study_area_equal_area)
    with pytest.raises(KeyError, match="no 10 km grid"):
        family.at(10.0)


def test_fine_mass_aggregation_uses_aligned_parent_indices_including_negative_cells() -> None:
    coarse, fine = _aligned_grids()
    fine_masses = {identifier: 1.0 for identifier in fine.cell_ids}

    aggregated = aggregate_fine_masses_to_coarse(coarse, fine, fine_masses)

    assert list(aggregated) == list(coarse.cell_ids)
    assert aggregated == {identifier: 4.0 for identifier in coarse.cell_ids}


def test_convergence_passes_for_equal_parent_masses_and_both_zero_totals() -> None:
    coarse, fine = _aligned_grids()
    fine_masses = {identifier: 1.0 for identifier in fine.cell_ids}
    coarse_masses = {identifier: 4.0 for identifier in coarse.cell_ids}

    diagnostics = diagnose_grid_convergence(coarse, fine, coarse_masses, fine_masses)
    assert diagnostics.coarse_total == diagnostics.fine_total == 16.0
    assert diagnostics.relative_expected_count_difference == 0.0
    assert diagnostics.density_l1_difference == 0.0
    assert diagnostics.passed is True
    require_grid_convergence(diagnostics)

    zero_diagnostics = diagnose_grid_convergence(
        coarse,
        fine,
        {identifier: 0.0 for identifier in coarse.cell_ids},
        {identifier: 0.0 for identifier in fine.cell_ids},
    )
    assert zero_diagnostics.relative_expected_count_difference == 0.0
    assert zero_diagnostics.density_l1_difference == 0.0
    assert zero_diagnostics.passed is True


def test_two_percent_total_and_five_percent_density_gates_fail_independently() -> None:
    coarse, fine = _aligned_grids()
    fine_masses = {identifier: 1.0 for identifier in fine.cell_ids}
    baseline = {identifier: 4.0 for identifier in coarse.cell_ids}

    total_failure = dict(baseline)
    total_failure[coarse.cell_ids[0]] = 4.4
    total_diagnostics = diagnose_grid_convergence(coarse, fine, total_failure, fine_masses)
    assert math.isclose(total_diagnostics.relative_expected_count_difference, 0.025)
    assert total_diagnostics.density_l1_difference < DENSITY_L1_TOLERANCE
    assert total_diagnostics.passed is False

    density_failure = dict(baseline)
    density_failure[coarse.cell_ids[0]] = 5.0
    density_failure[coarse.cell_ids[1]] = 3.0
    density_diagnostics = diagnose_grid_convergence(coarse, fine, density_failure, fine_masses)
    assert density_diagnostics.relative_expected_count_difference == 0.0
    assert math.isclose(density_diagnostics.density_l1_difference, 0.125)
    assert density_diagnostics.passed is False

    with pytest.raises(GridConvergenceError, match="grid convergence failed"):
        require_grid_convergence(density_diagnostics)


def test_three_grid_gate_preserves_both_frozen_pair_diagnostics() -> None:
    family = _three_grid_family()
    evidence = diagnose_three_grid_convergence(
        family,
        _passing_three_grid_masses(family),
    )

    assert isinstance(evidence, ThreeGridConvergenceGateEvidence)
    assert tuple(
        (item.coarse_cell_size_km, item.fine_cell_size_km) for item in evidence.comparisons
    ) == ((50.0, 25.0), (25.0, 12.5))
    assert evidence.diagnostic_50_to_25.coarse_total == 64.0
    assert evidence.diagnostic_50_to_25.fine_total == 64.0
    assert evidence.primary_25_to_12_5.coarse_total == 64.0
    assert evidence.primary_25_to_12_5.fine_total == 64.0
    assert evidence.passed is True
    require_three_grid_convergence(evidence)


@pytest.mark.parametrize("failed_resolution", [50.0, 12.5])
def test_three_grid_gate_fails_if_either_pair_fails_but_keeps_both_values(
    failed_resolution: float,
) -> None:
    family = _three_grid_family()
    masses = _passing_three_grid_masses(family)
    first_id = next(iter(masses[failed_resolution]))
    masses[failed_resolution][first_id] += 2.0

    evidence = diagnose_three_grid_convergence(family, masses)

    assert len(evidence.comparisons) == 2
    assert evidence.passed is False
    if failed_resolution == 50.0:
        assert evidence.diagnostic_50_to_25.passed is False
        assert evidence.primary_25_to_12_5.passed is True
    else:
        assert evidence.diagnostic_50_to_25.passed is True
        assert evidence.primary_25_to_12_5.passed is False
    with pytest.raises(GridConvergenceError, match="50->25km.*25->12.5km"):
        require_three_grid_convergence(evidence)


def test_three_grid_gate_rejects_incomplete_or_extra_mass_resolutions() -> None:
    family = _three_grid_family()
    masses = _passing_three_grid_masses(family)
    masses.pop(50.0)
    with pytest.raises(ValueError, match="exactly 50, 25, and 12.5"):
        diagnose_three_grid_convergence(family, masses)

    masses = _passing_three_grid_masses(family)
    masses[10.0] = {}
    with pytest.raises(ValueError, match="exactly 50, 25, and 12.5"):
        diagnose_three_grid_convergence(family, masses)


def test_mass_maps_must_be_complete_finite_and_nonnegative() -> None:
    coarse, fine = _aligned_grids()
    fine_masses = {identifier: 1.0 for identifier in fine.cell_ids}
    fine_masses.pop(fine.cell_ids[0])
    with pytest.raises(ValueError, match="exactly match"):
        aggregate_fine_masses_to_coarse(coarse, fine, fine_masses)

    invalid = {identifier: 1.0 for identifier in fine.cell_ids}
    invalid[fine.cell_ids[0]] = -1.0
    with pytest.raises(ValueError, match="finite and non-negative"):
        aggregate_fine_masses_to_coarse(coarse, fine, invalid)


def test_aggregation_rejects_reversed_or_nonmatching_domains() -> None:
    coarse, fine = _aligned_grids()
    coarse_masses = {identifier: 1.0 for identifier in coarse.cell_ids}
    with pytest.raises(ValueError, match="integer aligned size ratio"):
        aggregate_fine_masses_to_coarse(fine, coarse, coarse_masses)

    other_fine = build_clipped_grid(
        box(-25_000.0, -25_000.0, 12_500.0, 25_000.0),
        cell_size_km=12.5,
    )
    with pytest.raises(ValueError, match="same study area"):
        aggregate_fine_masses_to_coarse(
            coarse,
            other_fine,
            {identifier: 1.0 for identifier in other_fine.cell_ids},
        )
