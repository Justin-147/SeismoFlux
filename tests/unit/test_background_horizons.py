from __future__ import annotations

import math

import numpy as np
from shapely.geometry import box

from seismoflux.background.adapters import etas_variant_id
from seismoflux.background.catalog import EarthquakeCatalog, utc_timestamp_to_day
from seismoflux.background.config import load_background_protocol
from seismoflux.background.etas_fit import ETASModelSpec, ETASParameters
from seismoflux.background.grid import (
    GRID_CELL_SIZES_KM,
    EqualAreaGridFamily,
    build_clipped_grid,
)
from seismoflux.background.horizons import (
    EXPECTED_HORIZONS,
    EXPECTED_PUBLICATION_DELAYS,
    BackgroundHorizonBacktests,
    run_issue_horizon_backtests,
)
from seismoflux.background.issues import (
    FrozenIssueCalendar,
    IssueExposure,
    IssuePartition,
)
from seismoflux.background.poisson import (
    SpatialPoissonModel,
    SpatialQuadrature,
    UniformPoissonModel,
    fit_spatial_poisson_family,
    fit_uniform_poisson,
)


def _grid_family() -> EqualAreaGridFamily:
    study_area = box(1_000.0, 1_000.0, 2_000.0, 2_000.0)
    grids = tuple(
        build_clipped_grid(study_area, cell_size_km=cell_size) for cell_size in GRID_CELL_SIZES_KM
    )
    return EqualAreaGridFamily(study_area_equal_area=study_area, grids=grids)


def _calendar() -> FrozenIssueCalendar:
    issue_date = "2024-07-01"
    issue_time = "2024-06-30T16:00:00Z"
    issue_day = utc_timestamp_to_day(issue_time)
    exposures = tuple(
        (
            horizon,
            (
                IssueExposure(
                    issue_date_local=issue_date,
                    issue_time_utc=issue_time,
                    issue_day=issue_day,
                    horizon_days=horizon,
                    end_day=issue_day + horizon,
                ),
            ),
        )
        for horizon in EXPECTED_HORIZONS
    )
    development = IssuePartition(
        partition_id="development",
        start_local="2023-01-01",
        end_local="2023-12-31",
        actual_issue_dates_local=("2023-01-01",),
        actual_issue_days=(utc_timestamp_to_day("2022-12-31T16:00:00Z"),),
        exposures_by_horizon=exposures,
    )
    validation = IssuePartition(
        partition_id="validation",
        start_local="2024-07-01",
        end_local="2025-07-01",
        actual_issue_dates_local=(issue_date,),
        actual_issue_days=(issue_day,),
        exposures_by_horizon=exposures,
    )
    return FrozenIssueCalendar(
        schema_version="1.0.0",
        frozen_on="2026-07-13",
        freeze_tag="v0.2.0-background-protocol",
        development=development,
        validation=validation,
    )


def _catalog(*, with_targets: bool = True) -> EarthquakeCatalog:
    issue_day = utc_timestamp_to_day("2024-06-30T16:00:00Z")
    rows: list[tuple[str, float, float, float, float]] = [
        ("history", issue_day - 1.0, issue_day - 1.0, 1.5, 1.5),
    ]
    if with_targets:
        rows.extend(
            [
                ("late-target", issue_day + 1.0, issue_day + 5.0, 1.4, 1.5),
                ("later-target", issue_day + 6.0, issue_day + 6.0, 1.6, 1.5),
            ]
        )
    count = len(rows)
    return EarthquakeCatalog(
        event_id=np.asarray([row[0] for row in rows], dtype=np.str_),
        origin_day=np.asarray([row[1] for row in rows], dtype=np.float64),
        available_day=np.asarray([row[2] for row in rows], dtype=np.float64),
        longitude=np.full(count, 105.0),
        latitude=np.full(count, 35.0),
        x_km=np.asarray([row[3] for row in rows], dtype=np.float64),
        y_km=np.asarray([row[4] for row in rows], dtype=np.float64),
        magnitude=np.full(count, 3.2),
        inside_study_area=np.ones(count, dtype=np.bool_),
        inside_external_buffer=np.ones(count, dtype=np.bool_),
    )


def _models(
    family: EqualAreaGridFamily,
) -> tuple[UniformPoissonModel, SpatialPoissonModel]:
    duration_days = 100.0
    uniform = fit_uniform_poisson(
        training_event_count=1,
        training_duration_days=duration_days,
        study_area_km2=1.0,
    )
    quadrature = SpatialQuadrature.from_grid(family.at(12.5))
    spatial = fit_spatial_poisson_family(
        np.asarray([1.5]),
        np.asarray([1.5]),
        training_duration_days=duration_days,
        normalization_quadrature=quadrature,
        bandwidths_km=(75.0,),
    )[75.0]
    return uniform, spatial


def _run(*, with_targets: bool = True) -> BackgroundHorizonBacktests:
    config = load_background_protocol("configs/background.yaml")
    family = _grid_family()
    uniform, spatial = _models(family)
    spec = ETASModelSpec(mc=3.0, beta=math.log(10.0))
    parameters = ETASParameters(
        background_rate_per_day=0.01,
        productivity_k=0.05,
        alpha=0.5,
        c_days=1.0,
        p=1.2,
    )
    result = run_issue_horizon_backtests(
        config,
        _catalog(with_targets=with_targets),
        _calendar(),
        family,
        selected_mc=3.0,
        uniform_model=uniform,
        spatial_model=spatial,
        etas_parameters=parameters,
        etas_spec=spec,
        etas_parameter_snapshot_id="a" * 64,
        etas_model_variant_id=etas_variant_id(config, selected_bandwidth_km=75.0),
    )
    return result


def test_complete_model_delay_horizon_grid_and_physical_targets() -> None:
    result = _run()

    assert len(result.comparisons) == 30
    assert tuple(
        (item.candidate_model_id, item.publication_delay_days, item.horizon_days)
        for item in result.comparisons
    ) == tuple(
        (model_id, delay, horizon)
        for delay in EXPECTED_PUBLICATION_DELAYS
        for model_id in ("spatial_poisson", "etas")
        for horizon in EXPECTED_HORIZONS
    )
    for delay in EXPECTED_PUBLICATION_DELAYS:
        for model_id in ("spatial_poisson", "etas"):
            evidence = result.at(model_id, delay, 7)
            assert evidence.exposure_count == 1
            assert evidence.target_event_ids == ("late-target", "later-target")
            assert evidence.information_gain_per_event is not None
            assert not evidence.event_log_intensity_differences.flags.writeable


def test_publication_delay_changes_only_causal_etas_parent_availability() -> None:
    result = _run()

    etas_zero = result.at("etas", 0, 7).exposure_pairs[0][0]
    etas_seven = result.at("etas", 7, 7).exposure_pairs[0][0]
    spatial_zero = result.at("spatial_poisson", 0, 7).exposure_pairs[0][0]
    spatial_seven = result.at("spatial_poisson", 7, 7).exposure_pairs[0][0]

    assert etas_zero.target_event_ids == etas_seven.target_event_ids
    assert etas_zero.target_event_ids == ("late-target", "later-target")
    assert etas_zero.event_log_intensities[0] > etas_seven.event_log_intensities[0]
    assert etas_zero.event_log_intensities[1] > etas_seven.event_log_intensities[1]
    assert etas_zero.compensator > etas_seven.compensator
    np.testing.assert_array_equal(
        spatial_zero.event_log_intensities,
        spatial_seven.event_log_intensities,
    )
    assert spatial_zero.compensator == spatial_seven.compensator


def test_zero_event_exposures_are_retained_and_report_undefined_information_gain() -> None:
    result = _run(with_targets=False)

    for item in result.comparisons:
        assert item.exposure_count == 1
        assert item.target_event_ids == ()
        assert item.information_gain_per_event is None
        candidate, uniform = item.exposure_pairs[0]
        assert candidate.event_log_intensities.size == 0
        assert uniform.event_log_intensities.size == 0
        assert math.isfinite(candidate.compensator)
        assert math.isfinite(uniform.compensator)
