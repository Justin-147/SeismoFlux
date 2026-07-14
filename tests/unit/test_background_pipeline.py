from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import box

import seismoflux.background.pipeline as pipeline_module
from seismoflux.background.catalog import EarthquakeCatalog, StudyArea, utc_timestamp_to_day
from seismoflux.background.completeness import (
    CompletenessAnalysis,
    CompletenessAudit,
    CompletenessError,
    CompletenessScientificInability,
)
from seismoflux.background.config import (
    BackgroundConfig,
    load_background_protocol,
)
from seismoflux.background.etas_fit import (
    ETASEvent,
    ETASFitResult,
    ETASLikelihoodProblem,
    ETASModelSpec,
    ETASParameters,
    HessianAudit,
    StabilityAudit,
    observed_hessian_delta_uncertainty,
)
from seismoflux.background.evidence import (
    AuditedBackgroundModelEvidence,
    AuditedModelSelection,
)
from seismoflux.background.future import (
    FUTURE_HORIZONS_DAYS,
    FUTURE_QUANTILE_METHOD,
    FUTURE_QUANTILE_PROBABILITIES,
    FUTURE_REPLICATE_COUNT,
    FutureCountQuantile,
    FutureHorizonSummary,
    FutureIssueEnsemble,
    FutureWorkerResources,
    SparseGridExpectedCounts,
    ValidationFutureEnsembles,
)
from seismoflux.background.grid import (
    EQUAL_AREA_CRS,
    GRID_CELL_SIZES_KM,
    EqualAreaGridFamily,
    build_clipped_grid,
)
from seismoflux.background.issues import (
    FrozenIssueCalendar,
    IssueExposure,
    IssuePartition,
)
from seismoflux.background.pipeline import (
    BackgroundPipelineFailure,
    BackgroundPipelineGateError,
    BackgroundPipelineResult,
    run_background_pipeline,
)
from seismoflux.background.pipeline_etas import ETASPipelineResult
from seismoflux.background.pipeline_etas import run_etas_pipeline as raw_run_etas_pipeline
from seismoflux.background.pipeline_poisson import (
    PoissonKDEInvariantError,
    PoissonKDEPipelineResult,
    PoissonKDEScientificInability,
)
from seismoflux.background.poisson import SpatialPoissonModel
from seismoflux.background.regression import (
    AnalyticSimulationRegression,
    ProductionFixtureRegression,
)
from seismoflux.background.visualization import ConditionalIntensityRender
from seismoflux.background.workflow import (
    CompletenessSnapshot,
    SnapshotDefinition,
    build_snapshot_definitions,
)


def _study_and_grids() -> tuple[StudyArea, EqualAreaGridFamily]:
    projected = box(1_000.0, 1_000.0, 2_000.0, 2_000.0)
    study_area = StudyArea(
        geographic=box(100.0, 20.0, 101.0, 21.0),
        projected=projected,
        equal_area_crs=EQUAL_AREA_CRS,
        area_km2=1.0,
    )
    grids = tuple(
        build_clipped_grid(projected, cell_size_km=cell_size) for cell_size in GRID_CELL_SIZES_KM
    )
    return study_area, EqualAreaGridFamily(projected, grids)


def _catalog(*, delayed_final_x_km: float = 1.4) -> EarthquakeCatalog:
    rows = (
        ("training", "2000-01-01T00:00:00Z", "2000-01-01T00:00:00Z", 1.5, 1.5),
        ("target-fold-1", "2006-06-01T00:00:00Z", "2006-06-01T00:00:00Z", 1.2, 1.2),
        ("target-fold-2", "2011-06-01T00:00:00Z", "2011-06-01T00:00:00Z", 1.8, 1.2),
        ("target-fold-3", "2016-06-01T00:00:00Z", "2016-06-01T00:00:00Z", 1.2, 1.8),
        ("target-fold-4", "2021-06-01T00:00:00Z", "2021-06-01T00:00:00Z", 1.8, 1.8),
        (
            "delayed-final",
            "2025-01-01T00:00:00Z",
            "2026-01-01T00:00:00Z",
            delayed_final_x_km,
            1.5,
        ),
        ("later-final", "2025-06-01T00:00:00Z", "2025-06-01T00:00:00Z", 1.6, 1.5),
    )
    count = len(rows)
    return EarthquakeCatalog(
        event_id=np.asarray([row[0] for row in rows], dtype=np.str_),
        origin_day=np.asarray([utc_timestamp_to_day(row[1]) for row in rows]),
        available_day=np.asarray([utc_timestamp_to_day(row[2]) for row in rows]),
        longitude=np.full(count, 105.0),
        latitude=np.full(count, 35.0),
        x_km=np.asarray([row[3] for row in rows]),
        y_km=np.asarray([row[4] for row in rows]),
        magnitude=np.full(count, 3.5),
        inside_study_area=np.ones(count, dtype=np.bool_),
        inside_external_buffer=np.ones(count, dtype=np.bool_),
    )


def _analysis(definition: SnapshotDefinition) -> CompletenessAnalysis:
    return CompletenessAnalysis(
        cutoff_utc=datetime.fromisoformat(definition.fit_end_utc.replace("Z", "+00:00")),
        audit=CompletenessAudit(
            input_event_count=0,
            included_historical_inside_count=0,
            excluded_outside_count=0,
            excluded_pre_1970_count=0,
            excluded_future_origin_count=0,
            excluded_unavailable_count=0,
        ),
        temporal_blocks=(),
        spatial_strata=(),
        sparse_cell_resolutions=(),
        regime_changes=(),
        maximum_eligible_estimate=3.0,
        selected_mc=3.0,
        selected_event_count=1,
        selected_aki_b_value=1.0,
        sensitivities=(),
    )


def _completeness(config: BackgroundConfig) -> tuple[CompletenessSnapshot, ...]:
    return tuple(
        CompletenessSnapshot(definition=definition, analysis=_analysis(definition))
        for definition in build_snapshot_definitions(config)
    )


def _calendar() -> FrozenIssueCalendar:
    issue_date = "2025-06-26"
    issue_time = "2025-06-25T16:00:00Z"
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
        for horizon in FUTURE_HORIZONS_DAYS
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


def _stable_fit(
    problem: ETASLikelihoodProblem,
    spec: ETASModelSpec,
    **_: object,
) -> ETASFitResult:
    del problem
    parameters = ETASParameters(0.02, 0.005, 0.5, 1.0, 1.2)
    identity = tuple(
        tuple(1.0 if row == column else 0.0 for column in range(5)) for row in range(5)
    )
    stability = StabilityAudit(
        stable=True,
        converged_start_count=5,
        best_three_relative_objective_range=0.0,
        best_three_transformed_parameter_range=0.0,
        hessian=HessianAudit(True, 1.0, 1.0, identity, None),
        failure_reasons=(),
    )
    return ETASFitResult(
        best_parameters=parameters,
        best_objective=10.0,
        start_results=(),
        stability=stability,
        uncertainty=observed_hessian_delta_uncertainty(parameters, stability, spec),
    )


def _unstable_fit(reason: str) -> ETASFitResult:
    return ETASFitResult(
        best_parameters=None,
        best_objective=None,
        start_results=(),
        stability=StabilityAudit(
            stable=False,
            converged_start_count=0,
            best_three_relative_objective_range=None,
            best_three_transformed_parameter_range=None,
            hessian=HessianAudit(False, None, None, None, "synthetic no Hessian"),
            failure_reasons=(reason,),
        ),
        uncertainty=None,
    )


def _production_regression(*, passed: bool = True) -> ProductionFixtureRegression:
    return ProductionFixtureRegression(
        fixture_id="etas_micro_v1",
        observed_values=(),
        checks=(("synthetic_production", passed),),
    )


def _analytic_regression() -> AnalyticSimulationRegression:
    return AnalyticSimulationRegression(
        replicate_count=8192,
        generic_branching_ratio=0.1,
        root_direct_offspring_mean=0.1,
        root_total_descendants_mean=0.1,
        root_total_descendants_variance=0.1,
        root_zero_descendant_probability=0.9,
        direct_offspring_pit_ks=0.01,
        pooled_direct_offspring_count=8192,
        event_cap_hits=0,
        nonfinite_values=0,
        checks=(("synthetic_analytic", True),),
    )


def _zero_future_issue(issue_date: str) -> FutureIssueEnsemble:
    counts = (0,) * FUTURE_REPLICATE_COUNT
    horizons = tuple(
        FutureHorizonSummary(
            horizon_days=horizon,
            replicate_counts=counts,
            mean_count=0.0,
            quantiles=tuple(
                FutureCountQuantile(probability=probability, count=0.0)
                for probability in FUTURE_QUANTILE_PROBABILITIES
            ),
            grids=tuple(
                SparseGridExpectedCounts(cell_size_km=cell_size, cells=(), expected_total=0.0)
                for cell_size in GRID_CELL_SIZES_KM
            ),
        )
        for horizon in FUTURE_HORIZONS_DAYS
    )
    return FutureIssueEnsemble(
        issue_date_local=issue_date,
        issue_id=f"validation/{issue_date}",
        replicate_count=FUTURE_REPLICATE_COUNT,
        quantile_probabilities=FUTURE_QUANTILE_PROBABILITIES,
        quantile_method=FUTURE_QUANTILE_METHOD,
        horizons=horizons,
    )


def _render() -> ConditionalIntensityRender:
    return ConditionalIntensityRender(
        png_bytes=b"synthetic-png",
        panel_titles=("background", "trigger", "total"),
        colorbar_label="conditional intensity",
        footer_label="synthetic",
        png_metadata=(),
        width_px=1,
        height_px=1,
        dpi=150,
        font_family="test",
        colormap_name="test",
    )


def _install_lightweight_common(
    monkeypatch: pytest.MonkeyPatch,
    config: BackgroundConfig,
) -> None:
    completeness = _completeness(config)
    monkeypatch.setattr(
        pipeline_module,
        "run_production_fixture_regression",
        lambda path: _production_regression(),
    )
    monkeypatch.setattr(
        pipeline_module,
        "run_analytic_simulation_regression",
        lambda **kwargs: _analytic_regression(),
    )

    def analyze(
        catalog: EarthquakeCatalog,
        snapshots: tuple[SnapshotDefinition, ...],
        *,
        progress: Callable[[str], None] | None = None,
    ) -> tuple[CompletenessSnapshot, ...]:
        del catalog
        assert snapshots == build_snapshot_definitions(config)
        if progress is not None:
            for snapshot in snapshots:
                progress(f"completeness:{snapshot.snapshot_id}:synthetic")
        return completeness

    monkeypatch.setattr(pipeline_module, "analyze_snapshot_completeness", analyze)
    monkeypatch.setattr(pipeline_module, "detect_physical_core_count", lambda: 8)
    monkeypatch.setattr(
        pipeline_module,
        "render_conditional_intensity_figure",
        lambda *args, **kwargs: _render(),
    )


def test_successful_production_pipeline_uses_absolute_fixture_progress_and_exact_histories(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = load_background_protocol("configs/background.yaml")
    study_area, family = _study_and_grids()
    catalog = _catalog()
    calendar = _calendar()
    _install_lightweight_common(monkeypatch, config)
    captured: dict[str, object] = {}
    fixture_path = (tmp_path / "external" / "fixture.json").resolve()
    fixture_path.parent.mkdir()
    outside_cwd = tmp_path / "outside-cwd"
    outside_cwd.mkdir()

    def production(path: Path) -> ProductionFixtureRegression:
        captured["fixture_path"] = path
        return _production_regression()

    monkeypatch.setattr(pipeline_module, "run_production_fixture_regression", production)

    def etas_success(
        background: BackgroundConfig,
        earthquake_catalog: EarthquakeCatalog,
        grid_family: EqualAreaGridFamily,
        poisson_result: PoissonKDEPipelineResult,
        completeness_snapshots: tuple[CompletenessSnapshot, ...],
        *,
        progress: Callable[[str], None] | None = None,
    ) -> ETASPipelineResult:
        return raw_run_etas_pipeline(
            background,
            earthquake_catalog,
            grid_family,
            poisson_result,
            completeness_snapshots,
            fit_function=_stable_fit,
            progress=progress,
        )

    monkeypatch.setattr(pipeline_module, "run_etas_pipeline", etas_success)

    def select_etas(
        evidence: tuple[AuditedBackgroundModelEvidence, ...],
    ) -> AuditedModelSelection:
        etas = next(item for item in evidence if item.model_id == "etas")
        return AuditedModelSelection(
            validation_best_model_variant_id=etas.model_variant_id,
            selected_model_variant_id=etas.model_variant_id,
            paired_standard_errors=tuple((item.model_variant_id, 0.0) for item in evidence),
            excluded_model_variants=(),
        )

    monkeypatch.setattr(pipeline_module, "select_audited_background_model", select_etas)

    def future(
        parameters: ETASParameters,
        spec: ETASModelSpec,
        histories_by_issue_date: dict[str, Sequence[ETASEvent]],
        spatial_model: SpatialPoissonModel,
        supplied_study_area: StudyArea,
        supplied_grid_family: EqualAreaGridFamily,
        supplied_calendar: FrozenIssueCalendar,
        *,
        max_workers: int,
        reserve_physical_cores: int,
        physical_core_probe: Callable[[], int | None],
        progress: Callable[[str], None] | None = None,
    ) -> ValidationFutureEnsembles:
        del parameters, spec, spatial_model
        assert supplied_study_area is study_area
        assert supplied_grid_family is family
        assert supplied_calendar is calendar
        captured["histories"] = histories_by_issue_date
        if progress is not None:
            progress("future:2025-06-26:start")
            progress("future:2025-06-26:done")
        detected = physical_core_probe()
        resources = FutureWorkerResources(
            detected_physical_cores=detected,
            reserve_physical_cores=reserve_physical_cores,
            configured_max_workers=max_workers,
            effective_workers=min(max_workers, max(1, (detected or 0) - reserve_physical_cores)),
        )
        return ValidationFutureEnsembles(
            issues=(_zero_future_issue("2025-06-26"),),
            resources=resources,
        )

    monkeypatch.setattr(pipeline_module, "simulate_all_validation_issue_ensembles", future)
    progress: list[str] = []
    monkeypatch.chdir(outside_cwd)
    result = run_background_pipeline(
        config,
        catalog,
        study_area,
        family,
        calendar,
        production_fixture_path=fixture_path,
        max_workers=4,
        reserve_physical_cores=2,
        progress=progress.append,
    )

    assert isinstance(result, BackgroundPipelineResult)
    assert captured["fixture_path"] == fixture_path
    assert result.resources.effective_workers == 4
    assert result.future.ensembles is not None
    assert result.horizons.etas.comparisons is not None
    assert result.horizons.complete_backtests is not None
    assert result.representative.selected_model_id == "etas"
    assert result.representative.eligible_parent_event_ids
    assert np.any(result.representative.triggering_intensity > 0.0)
    assert tuple(item.cell_size_km for item in result.integration_grids.resolutions) == (
        50.0,
        25.0,
        12.5,
    )
    assert all(
        item.total_clipped_area_km2 == pytest.approx(1.0)
        for item in result.integration_grids.resolutions
    )
    primary = result.integration_grids.at(25.0)
    assert result.representative.cell_ids == tuple(cell.cell_id for cell in primary.cells)
    assert result.representative.rows == tuple(cell.row for cell in primary.cells)
    assert result.representative.columns == tuple(cell.column for cell in primary.cells)
    assert np.array_equal(
        result.representative.clipped_area_km2,
        np.asarray([cell.clipped_area_km2 for cell in primary.cells]),
    )
    histories = captured["histories"]
    assert isinstance(histories, dict)
    representative_history = histories["2025-06-26"]
    assert tuple(event.event_id for event in representative_history) == (
        "target-fold-3",
        "target-fold-4",
        "later-final",
    )
    assert all(
        event.time_days <= 0.0 and event.available_time_days <= 0.0 and event.time_days >= -3650.0
        for event in representative_history
    )
    assert progress[0] == "regression:production_fixture:start"
    assert "regression:analytic_simulation_8192:done" in progress
    assert "completeness:final_validation:synthetic" in progress
    assert "future:2025-06-26:start" in progress
    assert "future:2025-06-26:done" in progress
    assert progress[-1] == "map:done"


def test_final_etas_failure_keeps_identical_spatial_horizons_and_explicit_skips(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = load_background_protocol("configs/background.yaml")
    study_area, family = _study_and_grids()
    calendar = _calendar()
    fixture_path = (tmp_path / "fixture.json").resolve()

    def run_pipeline(*, fail_final: bool) -> BackgroundPipelineResult:
        with monkeypatch.context() as patch:
            _install_lightweight_common(patch, config)
            catalog = _catalog(delayed_final_x_km=1.9 if fail_final else 1.1)

            def fit(
                problem: ETASLikelihoodProblem,
                spec: ETASModelSpec,
                **kwargs: object,
            ) -> ETASFitResult:
                if fail_final and kwargs.get("model_id") == "etas/final_validation":
                    return _unstable_fit("synthetic final ETAS failure")
                return _stable_fit(problem, spec)

            def etas(
                background: BackgroundConfig,
                earthquake_catalog: EarthquakeCatalog,
                grid_family: EqualAreaGridFamily,
                poisson_result: PoissonKDEPipelineResult,
                completeness_snapshots: tuple[CompletenessSnapshot, ...],
                *,
                progress: Callable[[str], None] | None = None,
            ) -> ETASPipelineResult:
                return raw_run_etas_pipeline(
                    background,
                    earthquake_catalog,
                    grid_family,
                    poisson_result,
                    completeness_snapshots,
                    fit_function=fit,
                    progress=progress,
                )

            patch.setattr(pipeline_module, "run_etas_pipeline", etas)

            def future(*args: object, **kwargs: object) -> ValidationFutureEnsembles:
                del args, kwargs
                if fail_final:
                    raise AssertionError("future simulation must be skipped after final ETAS fail")
                return ValidationFutureEnsembles(
                    issues=(_zero_future_issue("2025-06-26"),),
                    resources=FutureWorkerResources(8, 2, 4, 4),
                )

            patch.setattr(pipeline_module, "simulate_all_validation_issue_ensembles", future)
            if fail_final:
                patch.setattr(
                    pipeline_module,
                    "evaluate_etas_issue_intensity_fields",
                    lambda *args, **kwargs: (_ for _ in ()).throw(
                        AssertionError("failed ETAS cannot drive representative intensity")
                    ),
                )
            progress: list[str] = []
            result = run_background_pipeline(
                config,
                catalog,
                study_area,
                family,
                calendar,
                production_fixture_path=fixture_path,
                max_workers=4,
                reserve_physical_cores=2,
                progress=progress.append,
            )
            assert isinstance(result, BackgroundPipelineResult)
            captured_progress[fail_final] = progress
            return result

    captured_progress: dict[bool, list[str]] = {}
    success = run_pipeline(fail_final=False)
    failed = run_pipeline(fail_final=True)

    assert failed.etas.attempt("final_validation").failure_reasons
    assert failed.bootstrap.spatial_poisson.interval is not None
    assert failed.bootstrap.etas.interval is None
    assert "final validation evidence unavailable" in (failed.bootstrap.etas.not_run_reason or "")
    failed_spatial_signature = tuple(
        (
            comparison.candidate_model_id,
            comparison.publication_delay_days,
            comparison.horizon_days,
            tuple(
                (candidate.score_id, uniform.score_id)
                for candidate, uniform in comparison.exposure_pairs
            ),
        )
        for comparison in failed.horizons.spatial_poisson.comparisons or ()
    )
    success_spatial_signature = tuple(
        (
            comparison.candidate_model_id,
            comparison.publication_delay_days,
            comparison.horizon_days,
            tuple(
                (candidate.score_id, uniform.score_id)
                for candidate, uniform in comparison.exposure_pairs
            ),
        )
        for comparison in success.horizons.spatial_poisson.comparisons or ()
    )
    assert failed_spatial_signature == success_spatial_signature
    assert failed.integration_grids == success.integration_grids
    assert failed.horizons.etas.comparisons is None
    assert "synthetic final ETAS failure" in (failed.horizons.etas.not_run_reason or "")
    assert failed.future.ensembles is None
    assert "synthetic final ETAS failure" in (failed.future.not_run_reason or "")
    assert failed.representative.selected_model_id != "etas"
    assert not np.any(failed.representative.triggering_intensity)
    assert any(item.startswith("horizons:etas:skipped:") for item in captured_progress[True])
    assert any(item.startswith("future:etas:skipped:") for item in captured_progress[True])


def test_failed_production_regression_hard_stops_before_analytic_or_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = load_background_protocol("configs/background.yaml")
    study_area, family = _study_and_grids()
    calls: list[str] = []
    monkeypatch.setattr(
        pipeline_module,
        "run_production_fixture_regression",
        lambda path: _production_regression(passed=False),
    )

    def analytic(**kwargs: object) -> AnalyticSimulationRegression:
        del kwargs
        calls.append("analytic")
        return _analytic_regression()

    monkeypatch.setattr(pipeline_module, "run_analytic_simulation_regression", analytic)
    monkeypatch.setattr(
        pipeline_module,
        "analyze_snapshot_completeness",
        lambda *args, **kwargs: calls.append("completeness"),
    )
    progress: list[str] = []

    with pytest.raises(BackgroundPipelineGateError, match="synthetic_production"):
        run_background_pipeline(
            config,
            _catalog(),
            study_area,
            family,
            _calendar(),
            production_fixture_path=(tmp_path / "fixture.json").resolve(),
            progress=progress.append,
        )

    assert calls == []
    assert progress == (
        [
            "regression:production_fixture:start",
            "regression:production_fixture:failed",
        ]
    )


def test_completeness_inability_returns_auditable_scientific_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = load_background_protocol("configs/background.yaml")
    study_area, family = _study_and_grids()
    monkeypatch.setattr(
        pipeline_module,
        "run_production_fixture_regression",
        lambda _path: _production_regression(),
    )
    monkeypatch.setattr(
        pipeline_module,
        "run_analytic_simulation_regression",
        lambda **_kwargs: _analytic_regression(),
    )
    monkeypatch.setattr(pipeline_module, "detect_physical_core_count", lambda: 8)

    def fail_completeness(*_args: object, **_kwargs: object) -> object:
        raise CompletenessScientificInability(
            "no_eligible_temporal_stratum",
            "no stable synthetic completeness stratum",
        )

    monkeypatch.setattr(
        pipeline_module,
        "analyze_snapshot_completeness",
        fail_completeness,
    )
    monkeypatch.setattr(
        pipeline_module,
        "run_poisson_kde_pipeline",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Poisson must not run after completeness failure")
        ),
    )

    outcome = run_background_pipeline(
        config,
        _catalog(),
        study_area,
        family,
        _calendar(),
        production_fixture_path=(tmp_path / "fixture.json").resolve(),
    )

    assert isinstance(outcome, BackgroundPipelineFailure)
    assert outcome.failure_stage == "completeness"
    assert outcome.failure_reason_code == "no_eligible_temporal_stratum"
    assert outcome.failure_reasons == ("no stable synthetic completeness stratum",)
    assert outcome.completeness == ()
    assert outcome.pre_score_gate_evidence is None

    def fail_protocol(*_args: object, **_kwargs: object) -> object:
        raise CompletenessError("synthetic completeness protocol defect")

    monkeypatch.setattr(pipeline_module, "analyze_snapshot_completeness", fail_protocol)
    with pytest.raises(CompletenessError, match="protocol defect"):
        run_background_pipeline(
            config,
            _catalog(),
            study_area,
            family,
            _calendar(),
            production_fixture_path=(tmp_path / "fixture.json").resolve(),
        )


def test_poisson_inability_retains_completed_upstream_scientific_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = load_background_protocol("configs/background.yaml")
    study_area, family = _study_and_grids()
    completeness = _completeness(config)
    monkeypatch.setattr(
        pipeline_module,
        "run_production_fixture_regression",
        lambda _path: _production_regression(),
    )
    monkeypatch.setattr(
        pipeline_module,
        "run_analytic_simulation_regression",
        lambda **_kwargs: _analytic_regression(),
    )
    monkeypatch.setattr(pipeline_module, "detect_physical_core_count", lambda: 8)
    monkeypatch.setattr(
        pipeline_module,
        "analyze_snapshot_completeness",
        lambda *_args, **_kwargs: completeness,
    )

    def fail_poisson(*_args: object, **_kwargs: object) -> object:
        raise PoissonKDEScientificInability(
            "zero_training_events",
            "synthetic KDE family cannot be fit",
        )

    monkeypatch.setattr(pipeline_module, "run_poisson_kde_pipeline", fail_poisson)

    outcome = run_background_pipeline(
        config,
        _catalog(),
        study_area,
        family,
        _calendar(),
        production_fixture_path=(tmp_path / "fixture.json").resolve(),
    )

    assert isinstance(outcome, BackgroundPipelineFailure)
    assert outcome.failure_stage == "poisson_kde"
    assert outcome.failure_reason_code == "zero_training_events"
    assert outcome.failure_reasons == ("synthetic KDE family cannot be fit",)
    assert outcome.completeness == completeness

    def fail_invariant(*_args: object, **_kwargs: object) -> object:
        raise PoissonKDEInvariantError("synthetic fitted-state contract defect")

    monkeypatch.setattr(pipeline_module, "run_poisson_kde_pipeline", fail_invariant)
    with pytest.raises(PoissonKDEInvariantError, match="contract defect"):
        run_background_pipeline(
            config,
            _catalog(),
            study_area,
            family,
            _calendar(),
            production_fixture_path=(tmp_path / "fixture.json").resolve(),
        )
