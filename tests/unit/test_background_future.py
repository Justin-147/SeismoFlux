from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pytest
from shapely.geometry import box

import seismoflux.background.future as future_module
from seismoflux.background.catalog import StudyArea
from seismoflux.background.etas_fit import ETASEvent, ETASModelSpec, ETASParameters
from seismoflux.background.etas_simulation import (
    EventCapExceeded,
    SimulationDomain,
    SimulationResult,
)
from seismoflux.background.etas_simulation import (
    simulate_future_catalog as raw_simulate_future_catalog,
)
from seismoflux.background.future import (
    FUTURE_HORIZONS_DAYS,
    FUTURE_MAXIMUM_EVENTS,
    FUTURE_QUANTILE_METHOD,
    FUTURE_QUANTILE_PROBABILITIES,
    FUTURE_REPLICATE_COUNT,
    BackgroundRejectionSamplingError,
    build_kde_study_area_simulation_domain,
    simulate_all_validation_issue_ensembles,
    simulate_validation_issue_ensemble,
)
from seismoflux.background.grid import (
    EQUAL_AREA_CRS,
    GRID_CELL_SIZES_KM,
    EqualAreaGridFamily,
    build_clipped_grid,
)
from seismoflux.background.issues import (
    FrozenIssueCalendar,
    IssuePartition,
)
from seismoflux.background.poisson import GaussianMixtureFamily, SpatialPoissonModel
from seismoflux.background.randomness import SeedContext


def _study_and_grids() -> tuple[StudyArea, EqualAreaGridFamily]:
    projected = box(0.0, 0.0, 100_000.0, 100_000.0)
    study_area = StudyArea(
        geographic=box(100.0, 20.0, 101.0, 21.0),
        projected=projected,
        equal_area_crs=EQUAL_AREA_CRS,
        area_km2=10_000.0,
    )
    grids = tuple(
        build_clipped_grid(projected, cell_size_km=cell_size) for cell_size in GRID_CELL_SIZES_KM
    )
    return study_area, EqualAreaGridFamily(projected, grids)


def _spatial_model(
    *,
    training_x_km: tuple[float, ...] = (25.0, 75.0),
    training_y_km: tuple[float, ...] = (25.0, 75.0),
    bandwidth_km: float = 2.0,
) -> SpatialPoissonModel:
    return SpatialPoissonModel(
        mixture=GaussianMixtureFamily(
            np.asarray(training_x_km, dtype=np.float64),
            np.asarray(training_y_km, dtype=np.float64),
        ),
        bandwidth_km=bandwidth_km,
        normalization_mass=1.0,
        rate_per_day=0.01,
    )


def _parameters(*, background_rate_per_day: float = 0.01) -> ETASParameters:
    return ETASParameters(
        background_rate_per_day=background_rate_per_day,
        productivity_k=0.0,
        alpha=0.0,
        c_days=0.05,
        p=1.2,
    )


def _spec() -> ETASModelSpec:
    return ETASModelSpec(mc=3.0, beta=2.0)


def _partition(partition_id: str, dates: tuple[str, ...]) -> IssuePartition:
    return IssuePartition(
        partition_id=partition_id,
        start_local=dates[0],
        end_local=dates[-1],
        actual_issue_dates_local=dates,
        actual_issue_days=tuple(float(index) for index in range(len(dates))),
        exposures_by_horizon=tuple((horizon, ()) for horizon in FUTURE_HORIZONS_DAYS),
    )


def _calendar(
    validation_dates: tuple[str, ...] = ("2025-06-26",),
) -> FrozenIssueCalendar:
    return FrozenIssueCalendar(
        schema_version="1.0.0",
        frozen_on="2026-07-13",
        freeze_tag="v0.2.0-background-baselines",
        development=_partition("development", ("2024-06-27",)),
        validation=_partition("validation", validation_dates),
    )


def test_kde_domain_is_conditioned_on_study_and_classifies_exact_buffer() -> None:
    study_area, _ = _study_and_grids()
    domain = build_kde_study_area_simulation_domain(_spatial_model(), study_area)

    assert domain.membership(0.0, 50.0) == "study"
    assert domain.membership(-0.001, 50.0) == "buffer"
    assert domain.membership(-300.0, 50.0) == "buffer"
    assert domain.membership(-300.001, 50.0) == "outside"

    first_generator = np.random.Generator(np.random.PCG64(147))
    second_generator = np.random.Generator(np.random.PCG64(147))
    first = tuple(domain.sample_immigrant_location(first_generator) for _ in range(100))
    second = tuple(domain.sample_immigrant_location(second_generator) for _ in range(100))
    assert first == second
    assert all(domain.membership(x_km, y_km) == "study" for x_km, y_km in first)

    impossible = build_kde_study_area_simulation_domain(
        _spatial_model(
            training_x_km=(1_000.0,),
            training_y_km=(1_000.0,),
            bandwidth_km=0.001,
        ),
        study_area,
        maximum_rejection_attempts=3,
    )
    with pytest.raises(BackgroundRejectionSamplingError, match="rejection-attempt limit"):
        impossible.sample_immigrant_location(np.random.Generator(np.random.PCG64(1)))


def test_one_validation_issue_uses_exactly_128_reusable_catalogs_and_sparse_grids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study_area, grid_family = _study_and_grids()
    calendar = _calendar()
    original = raw_simulate_future_catalog
    calls: list[tuple[float, int, str | None, int]] = []

    def record_simulation(
        parameters: ETASParameters,
        spec: ETASModelSpec,
        history_events: Sequence[ETASEvent],
        domain: SimulationDomain,
        *,
        horizon_days: float,
        seed_context: SeedContext,
        maximum_events: int = FUTURE_MAXIMUM_EVENTS,
    ) -> SimulationResult:
        calls.append(
            (
                horizon_days,
                maximum_events,
                seed_context.issue_id,
                seed_context.replicate_index,
            )
        )
        return original(
            parameters,
            spec,
            history_events,
            domain,
            horizon_days=horizon_days,
            seed_context=seed_context,
            maximum_events=maximum_events,
        )

    monkeypatch.setattr(future_module, "simulate_future_catalog", record_simulation)
    history = (
        ETASEvent(
            "known-at-issue",
            -1.0,
            0.0,
            25.0,
            25.0,
            4.0,
            True,
            True,
        ),
    )
    first = simulate_validation_issue_ensemble(
        _parameters(),
        _spec(),
        history,
        _spatial_model(),
        study_area,
        grid_family,
        calendar,
        issue_date_local="2025-06-26",
    )

    assert len(calls) == FUTURE_REPLICATE_COUNT
    assert calls == [
        (365.0, FUTURE_MAXIMUM_EVENTS, "validation/2025-06-26", replicate_index)
        for replicate_index in range(FUTURE_REPLICATE_COUNT)
    ]
    assert first.issue_id == "validation/2025-06-26"
    assert first.replicate_count == FUTURE_REPLICATE_COUNT
    assert first.quantile_probabilities == FUTURE_QUANTILE_PROBABILITIES
    assert first.quantile_method == FUTURE_QUANTILE_METHOD == "linear"
    assert tuple(summary.horizon_days for summary in first.horizons) == FUTURE_HORIZONS_DAYS

    for replicate_index in range(FUTURE_REPLICATE_COUNT):
        counts = tuple(summary.replicate_counts[replicate_index] for summary in first.horizons)
        assert counts == tuple(sorted(counts))
    for summary in first.horizons:
        assert summary.replicate_indices == tuple(range(FUTURE_REPLICATE_COUNT))
        assert summary.mean_count == pytest.approx(
            math.fsum(summary.replicate_counts) / FUTURE_REPLICATE_COUNT,
            rel=0.0,
            abs=0.0,
        )
        expected_quantiles = np.quantile(
            np.asarray(summary.replicate_counts, dtype=np.float64),
            FUTURE_QUANTILE_PROBABILITIES,
            method="linear",
        )
        np.testing.assert_array_equal(
            np.asarray([item.count for item in summary.quantiles]),
            expected_quantiles,
        )
        assert tuple(grid.cell_size_km for grid in summary.grids) == GRID_CELL_SIZES_KM
        for grid in summary.grids:
            identifiers = tuple(item.cell_id for item in grid.cells)
            assert identifiers == tuple(sorted(identifiers))
            assert all(item.expected_count > 0.0 for item in grid.cells)
            assert grid.expected_total == pytest.approx(summary.mean_count)
            assert grid.expected_count("not-a-real-cell") == 0.0

    second = simulate_validation_issue_ensemble(
        _parameters(),
        _spec(),
        history,
        _spatial_model(),
        study_area,
        grid_family,
        calendar,
        issue_date_local="2025-06-26",
    )
    assert second == first
    assert len(calls) == FUTURE_REPLICATE_COUNT * 2


def test_issue_membership_history_availability_and_event_cap_fail_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study_area, grid_family = _study_and_grids()
    calendar = _calendar()
    common = (
        _parameters(background_rate_per_day=0.0001),
        _spec(),
        _spatial_model(),
        study_area,
        grid_family,
        calendar,
    )

    with pytest.raises(ValueError, match="FrozenIssueCalendar.validation"):
        simulate_validation_issue_ensemble(
            common[0],
            common[1],
            (),
            common[2],
            common[3],
            common[4],
            common[5],
            issue_date_local="2024-06-27",
        )

    future_event = ETASEvent("future", 0.1, 0.1, 25.0, 25.0, 4.0, True, True)
    with pytest.raises(ValueError, match="relative to.*issue"):
        simulate_validation_issue_ensemble(
            common[0],
            common[1],
            (future_event,),
            common[2],
            common[3],
            common[4],
            common[5],
            issue_date_local="2025-06-26",
        )

    delayed = ETASEvent("delayed", -1.0, 0.1, 25.0, 25.0, 4.0, True, True)
    with pytest.raises(ValueError, match="available at or before"):
        simulate_validation_issue_ensemble(
            common[0],
            common[1],
            (delayed,),
            common[2],
            common[3],
            common[4],
            common[5],
            issue_date_local="2025-06-26",
        )

    def hit_cap(
        parameters: ETASParameters,
        spec: ETASModelSpec,
        history_events: Sequence[ETASEvent],
        domain: SimulationDomain,
        *,
        horizon_days: float,
        seed_context: SeedContext,
        maximum_events: int = FUTURE_MAXIMUM_EVENTS,
    ) -> SimulationResult:
        del parameters, spec, history_events, domain, horizon_days, seed_context
        assert maximum_events == FUTURE_MAXIMUM_EVENTS
        raise EventCapExceeded("synthetic hard cap")

    monkeypatch.setattr(future_module, "simulate_future_catalog", hit_cap)
    with pytest.raises(EventCapExceeded, match="synthetic hard cap"):
        simulate_validation_issue_ensemble(
            common[0],
            common[1],
            (),
            common[2],
            common[3],
            common[4],
            common[5],
            issue_date_local="2025-06-26",
        )


def test_all_validation_issues_are_worker_invariant_and_calendar_ordered() -> None:
    study_area, grid_family = _study_and_grids()
    calendar = _calendar(("2025-06-19", "2025-06-26"))
    histories = {
        "2025-06-26": (),
        "2025-06-19": (),
    }
    arguments = (
        _parameters(background_rate_per_day=0.0001),
        _spec(),
        histories,
        _spatial_model(),
        study_area,
        grid_family,
        calendar,
    )

    without_callback = simulate_all_validation_issue_ensembles(
        *arguments,
        max_workers=12,
        physical_core_probe=lambda: None,
    )
    sequential_progress: list[str] = []
    sequential = simulate_all_validation_issue_ensembles(
        *arguments,
        max_workers=12,
        physical_core_probe=lambda: None,
        progress=sequential_progress.append,
    )
    parallel_progress: list[str] = []
    parallel = simulate_all_validation_issue_ensembles(
        *arguments,
        max_workers=3,
        physical_core_probe=lambda: 8,
        progress=parallel_progress.append,
    )

    assert without_callback == sequential
    assert sequential.issues == parallel.issues
    assert tuple(issue.issue_date_local for issue in parallel.issues) == (
        "2025-06-19",
        "2025-06-26",
    )
    assert sequential.resources.effective_workers == 1
    assert parallel.resources.effective_workers == 3
    assert parallel.resources.reserve_physical_cores == 2
    expected_progress = [
        "future:2025-06-19:start",
        "future:2025-06-19:done",
        "future:2025-06-26:start",
        "future:2025-06-26:done",
    ]
    assert sequential_progress == expected_progress
    assert parallel_progress == [
        "future:2025-06-19:start",
        "future:2025-06-26:start",
        "future:2025-06-19:done",
        "future:2025-06-26:done",
    ]

    with pytest.raises(ValueError, match="exactly match"):
        simulate_all_validation_issue_ensembles(
            arguments[0],
            arguments[1],
            {"2025-06-19": ()},
            arguments[3],
            arguments[4],
            arguments[5],
            arguments[6],
            physical_core_probe=lambda: 8,
        )
    with pytest.raises(ValueError, match="reserve at least two"):
        simulate_all_validation_issue_ensembles(
            *arguments,
            reserve_physical_cores=1,
            physical_core_probe=lambda: 8,
        )
