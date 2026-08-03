"""Pure-synthetic tests for the D1 causal spatial foundation."""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pyarrow as pa
import pytest
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

import seismoflux.d1_replay.spatial as spatial
from seismoflux.features.anomaly.grid import Stage3QueryGrid, build_stage3_query_grid
from seismoflux.stage2s.catalog import CatalogIdentity, Stage2SEarthquakeCatalog
from seismoflux.stage2s.contracts import SpatialGrid


def _epoch_us(value: datetime) -> int:
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    delta = value.astimezone(UTC) - epoch
    return (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds


@pytest.fixture(scope="module")
def study_area_wgs84() -> BaseGeometry:
    return box(109.5, 34.5, 110.5, 35.5)


@pytest.fixture(scope="module")
def domain(study_area_wgs84: BaseGeometry) -> spatial.D1SpatialDomain:
    return spatial.build_d1_spatial_domain(study_area_wgs84)


def test_d1_domain_is_bitwise_identical_to_stage3_25km_grid(
    domain: spatial.D1SpatialDomain,
    study_area_wgs84: BaseGeometry,
) -> None:
    expected = build_stage3_query_grid(study_area_wgs84)
    observed = domain.operational_grid
    assert observed.grid_id == expected.grid_id
    assert observed.cell_ids == expected.cell_ids
    assert observed.rows.dtype == expected.rows.dtype
    assert observed.columns.dtype == expected.columns.dtype
    assert observed.rows.tobytes() == expected.rows.tobytes()
    assert observed.columns.tobytes() == expected.columns.tobytes()
    assert observed.query_xy_km.tobytes() == (expected.query_xy_m / 1_000.0).tobytes()
    assert observed.clipped_area_km2.tobytes() == expected.clipped_area_km2.tobytes()
    assert tuple(grid.cell_size_km for grid in domain.quadrature_family.grids) == (
        50.0,
        25.0,
        12.5,
    )
    assert domain.locator.grid is observed


def test_d1_domain_rejects_even_one_bit_of_stage3_grid_drift(
    monkeypatch: pytest.MonkeyPatch,
    study_area_wgs84: BaseGeometry,
) -> None:
    original_builder = build_stage3_query_grid

    def mismatched_builder(geometry: BaseGeometry) -> Stage3QueryGrid:
        original = original_builder(geometry)
        changed = np.array(original.query_xy_m, dtype=np.float64, copy=True)
        changed[0, 0] = np.nextafter(changed[0, 0], math.inf)
        return replace(original, query_xy_m=changed)

    monkeypatch.setattr(
        "seismoflux.d1_replay.spatial.build_stage3_query_grid",
        mismatched_builder,
    )
    with pytest.raises(spatial.D1SpatialError, match="bitwise"):
        spatial.build_d1_spatial_domain(study_area_wgs84)


def _locator_grid_and_geometries() -> tuple[SpatialGrid, tuple[BaseGeometry, ...]]:
    geometries = (
        box(0.0, 0.0, 25_000.0, 25_000.0),
        box(25_000.0, 0.0, 50_000.0, 25_000.0),
        box(0.0, 25_000.0, 25_000.0, 50_000.0),
        box(26_000.0, 26_000.0, 50_000.0, 50_000.0),
    )
    query_xy_km = np.asarray(
        [(12.5, 12.5), (37.5, 12.5), (12.5, 37.5), (38.0, 38.0)],
        dtype=np.float64,
    )
    grid = SpatialGrid(
        grid_id="synthetic-locator-25km",
        cell_size_km=25.0,
        cell_ids=("r0c0", "r0c1", "r1c0", "r1c1"),
        rows=np.asarray([0, 0, 1, 1], dtype=np.int64),
        columns=np.asarray([0, 1, 0, 1], dtype=np.int64),
        query_xy_km=query_xy_km,
        clipped_area_km2=np.asarray(
            [float(geometry.area) / 1_000_000.0 for geometry in geometries],
            dtype=np.float64,
        ),
    )
    return grid, geometries


def test_frozen_locator_uses_high_side_order_covers_and_no_coordinate_nudge() -> None:
    grid, geometries = _locator_grid_and_geometries()
    locator = spatial.Frozen25kmCellLocator(grid=grid, clipped_geometries=geometries)

    assert locator.candidate_row_columns(25_000.0, 25_000.0) == (
        (1, 1),
        (1, 0),
        (0, 1),
        (0, 0),
    )
    # The high/high clipped fragment does not cover the intersection.  The
    # frozen second candidate, high-row/low-column, must win before low/high.
    assert locator.locate_projected(25_000.0, 25_000.0) == 2
    # ``covers`` (not ``contains``) keeps an exact outer-boundary point.
    assert locator.locate_projected(0.0, 12_500.0) == 0
    # One representable float below the line remains on the low side; the exact
    # line itself goes to the high side.  There is no epsilon-based movement.
    below = np.nextafter(25_000.0, -math.inf)
    assert locator.locate_projected(float(below), 12_500.0) == 0
    assert locator.locate_projected(25_000.0, 12_500.0) == 1
    assert locator.locate_projected(-1.0, 12_500.0) is None


def _synthetic_catalog(issue_time: datetime) -> Stage2SEarthquakeCatalog:
    lower = issue_time - timedelta(days=30)
    rows = (
        ("pre1970", datetime(1969, 12, 31, tzinfo=UTC), 109.7, 34.7, 5.0, True),
        ("old", datetime(2023, 1, 1, tzinfo=UTC), 109.7, 34.7, 4.5, True),
        ("lowmag", datetime(2023, 2, 1, tzinfo=UTC), 109.8, 34.8, 3.9, True),
        ("outside", datetime(2023, 3, 1, tzinfo=UTC), 109.9, 34.9, 4.2, False),
        ("lower", lower, 109.8, 34.8, 4.1, True),
        ("after_lower", lower + timedelta(microseconds=1), 109.8, 34.8, 4.2, True),
        ("delayed", issue_time - timedelta(days=1), 110.0, 35.0, 4.3, True),
        ("at_issue", issue_time, 110.2, 34.8, 5.0, True),
        ("future", issue_time + timedelta(days=1), 110.1, 35.1, 4.4, True),
    )
    event_ids = tuple(row[0] for row in rows)
    origins = np.asarray([_epoch_us(row[1]) for row in rows], dtype=np.int64)
    available = np.array(origins, dtype=np.int64, copy=True)
    available[6] = _epoch_us(issue_time + timedelta(days=1))
    longitudes = np.asarray([row[2] for row in rows], dtype=np.float64)
    latitudes = np.asarray([row[3] for row in rows], dtype=np.float64)
    magnitudes = np.asarray([row[4] for row in rows], dtype=np.float64)
    inside = np.asarray([row[5] for row in rows], dtype=np.bool_)
    table = pa.table({"event_id": pa.array(event_ids, type=pa.string())})
    return Stage2SEarthquakeCatalog(
        identity=CatalogIdentity(
            row_count=len(rows),
            file_sha256="1" * 64,
            content_sha256="2" * 64,
            schema_sha256="3" * 64,
        ),
        event_ids=event_ids,
        origin_time_us=origins,
        available_at_us=available,
        longitude=longitudes,
        latitude=latitudes,
        magnitude=magnitudes,
        inside_study_area=inside,
        _table=table,
    )


def test_causal_background_uses_exact_1970_m4_inside_available_and_r30_rules(
    domain: spatial.D1SpatialDomain,
) -> None:
    issue_time = datetime(2024, 4, 1, tzinfo=UTC)
    catalog = _synthetic_catalog(issue_time)
    background = spatial.build_causal_background_components(catalog, issue_time, domain)

    assert background.audit == spatial.CausalCatalogAudit(
        issue_time_us=_epoch_us(issue_time),
        total_catalog_rows=9,
        rejected_before_1970=1,
        rejected_below_m4=1,
        rejected_outside_study_area=1,
        rejected_after_issue_origin=1,
        rejected_unavailable_at_issue=2,
        b0_source_count=4,
        recent_30d_source_count=2,
    )
    assert math.fsum(float(value) for value in background.b0_mass_25km) == pytest.approx(
        1.0,
        rel=0.0,
        abs=1.0e-15,
    )
    assert math.fsum(float(value) for value in background.recent_mass_25km) == pytest.approx(
        1.0,
        rel=0.0,
        abs=1.0e-15,
    )
    assert not background.b0_mass_25km.flags.writeable
    assert not background.recent_mass_25km.flags.writeable
    mixed = background.mass_for_alpha(0.75)
    assert not mixed.flags.writeable
    assert math.fsum(float(value) for value in mixed) == pytest.approx(
        1.0,
        rel=0.0,
        abs=1.0e-15,
    )
    assert not np.array_equal(mixed, background.b0_mass_25km)
    with pytest.raises(ValueError, match="alpha"):
        background.mass_for_alpha(0.6)


def test_empty_recent_window_is_the_exact_b0_spatial_mass(
    domain: spatial.D1SpatialDomain,
) -> None:
    catalog_issue = datetime(2024, 4, 1, tzinfo=UTC)
    replay_issue = datetime(2023, 12, 31, tzinfo=UTC)
    background = spatial.build_causal_background_components(
        _synthetic_catalog(catalog_issue),
        replay_issue,
        domain,
    )
    assert background.audit.b0_source_count == 1
    assert background.audit.recent_30d_source_count == 0
    assert np.array_equal(background.recent_mass_25km, background.b0_mass_25km)
    assert np.array_equal(background.mass_for_alpha(0.75), background.b0_mass_25km)


def _alarm_grid() -> SpatialGrid:
    areas = np.asarray([200_000.0, 250_000.0, 100_000.0, 150_000.0, 200_000.0])
    return SpatialGrid(
        grid_id="synthetic-alarm-25km",
        cell_size_km=25.0,
        cell_ids=("c0", "c1", "c2", "c3", "c4"),
        rows=np.zeros(5, dtype=np.int64),
        columns=np.arange(5, dtype=np.int64),
        query_xy_km=np.asarray(
            [(12.5 + 25.0 * column, 12.5) for column in range(5)],
            dtype=np.float64,
        ),
        clipped_area_km2=areas,
    )


def test_five_alarm_budgets_use_one_ranking_and_stop_at_first_overflow() -> None:
    grid = _alarm_grid()
    # c0 and c1 have equal leading intensity, so row/column puts c0 first.
    # Construct an already normalized mass whose first two mass/area values
    # are bitwise equal.  Normalizing a proportional raw vector can introduce
    # a one-ULP difference and would no longer exercise the registered tie.
    mass = grid.clipped_area_km2 * np.asarray([1.4e-6, 1.4e-6, 1.2e-6, 1.0e-6, 0.5e-6])
    prefixes = spatial.select_alarm_prefixes(mass, grid)

    assert tuple(prefix.budget_km2 for prefix in prefixes) == (
        300_000.0,
        450_000.0,
        600_000.0,
        750_000.0,
        960_000.0,
    )
    assert tuple(prefix.selected_indices.tolist() for prefix in prefixes) == (
        [0],
        [0, 1],
        [0, 1, 2],
        [0, 1, 2, 3],
        [0, 1, 2, 3, 4],
    )
    assert tuple(prefix.actual_area_km2 for prefix in prefixes) == (
        200_000.0,
        450_000.0,
        550_000.0,
        700_000.0,
        900_000.0,
    )
    assert all(not prefix.selected_indices.flags.writeable for prefix in prefixes)
    # At 300k, c2 would fit after c1 overflows.  It is correctly not skipped to.
    assert prefixes[0].selected_cell_ids == ("c0",)
