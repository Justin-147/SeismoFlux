from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime

import numpy as np
import pytest
from shapely.geometry import box

import seismoflux.background.pipeline_etas as pipeline_etas
from seismoflux.background.catalog import EarthquakeCatalog, utc_timestamp_to_day
from seismoflux.background.completeness import CompletenessAnalysis, CompletenessAudit
from seismoflux.background.config import BackgroundConfig, load_background_config
from seismoflux.background.etas_fit import (
    ETASFitResult,
    ETASLikelihoodProblem,
    ETASModelSpec,
    ETASParameters,
    HessianAudit,
    StabilityAudit,
    observed_hessian_delta_uncertainty,
)
from seismoflux.background.evidence import EXPECTED_SNAPSHOTS
from seismoflux.background.grid import (
    GRID_CELL_SIZES_KM,
    EqualAreaGridFamily,
    ThreeGridConvergenceGateEvidence,
    build_clipped_grid,
    diagnose_three_grid_convergence,
)
from seismoflux.background.pipeline_etas import run_etas_pipeline
from seismoflux.background.pipeline_poisson import (
    PoissonKDEPipelineResult,
    run_poisson_kde_pipeline,
)
from seismoflux.background.workflow import (
    CompletenessSnapshot,
    SnapshotDefinition,
    build_snapshot_definitions,
)


def _grid_family() -> EqualAreaGridFamily:
    study_area = box(1_000.0, 1_000.0, 2_000.0, 2_000.0)
    grids = tuple(
        build_clipped_grid(study_area, cell_size_km=cell_size_km)
        for cell_size_km in GRID_CELL_SIZES_KM
    )
    return EqualAreaGridFamily(study_area_equal_area=study_area, grids=grids)


def _catalog(*, delayed_available_utc: str = "2026-01-01T00:00:00Z") -> EarthquakeCatalog:
    rows = (
        ("training", "2000-01-01T00:00:00Z", "2000-01-01T00:00:00Z", 1.5, 1.5),
        ("target-fold-1", "2006-06-01T00:00:00Z", "2006-06-01T00:00:00Z", 1.2, 1.2),
        ("target-fold-2", "2011-06-01T00:00:00Z", "2011-06-01T00:00:00Z", 1.8, 1.2),
        ("target-fold-3", "2016-06-01T00:00:00Z", "2016-06-01T00:00:00Z", 1.2, 1.8),
        ("target-fold-4", "2021-06-01T00:00:00Z", "2021-06-01T00:00:00Z", 1.8, 1.8),
        ("delayed-final", "2025-01-01T00:00:00Z", delayed_available_utc, 1.4, 1.5),
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
    cutoff = datetime.fromisoformat(definition.fit_end_utc.replace("Z", "+00:00"))
    return CompletenessAnalysis(
        cutoff_utc=cutoff,
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


def _stable_fit(
    problem: ETASLikelihoodProblem,
    spec: ETASModelSpec,
    **_: object,
) -> ETASFitResult:
    del problem
    parameters = ETASParameters(
        background_rate_per_day=0.02,
        productivity_k=0.005,
        alpha=0.5,
        c_days=1.0,
        p=1.2,
    )
    identity = tuple(
        tuple(1.0 if row == column else 0.0 for column in range(5)) for row in range(5)
    )
    stability = StabilityAudit(
        stable=True,
        converged_start_count=5,
        best_three_relative_objective_range=0.0,
        best_three_transformed_parameter_range=0.0,
        hessian=HessianAudit(
            success=True,
            minimum_eigenvalue=1.0,
            condition_number=1.0,
            matrix=identity,
            failure_reason=None,
        ),
        failure_reasons=(),
    )
    return ETASFitResult(
        best_parameters=parameters,
        best_objective=10.0,
        start_results=(),
        stability=stability,
        uncertainty=observed_hessian_delta_uncertainty(parameters, stability, spec),
    )


def _unstable_fit(reason: str = "synthetic instability") -> ETASFitResult:
    return ETASFitResult(
        best_parameters=None,
        best_objective=None,
        start_results=(),
        stability=StabilityAudit(
            stable=False,
            converged_start_count=0,
            best_three_relative_objective_range=None,
            best_three_transformed_parameter_range=None,
            hessian=HessianAudit(
                success=False,
                minimum_eigenvalue=None,
                condition_number=None,
                matrix=None,
                failure_reason="synthetic no Hessian",
            ),
            failure_reasons=(reason,),
        ),
        uncertainty=None,
    )


def _poisson_inputs(
    config: BackgroundConfig,
    catalog: EarthquakeCatalog,
    family: EqualAreaGridFamily,
    completeness: tuple[CompletenessSnapshot, ...],
) -> PoissonKDEPipelineResult:
    return run_poisson_kde_pipeline(
        config,
        catalog,
        family,
        completeness,
        chunk_size=2,
    )


def test_five_snapshot_etas_keeps_physical_late_report_and_gates_parent_availability() -> None:
    config = load_background_config("configs/background.yaml")
    family = _grid_family()
    completeness = _completeness(config)
    delayed_catalog = _catalog()
    available_catalog = _catalog(delayed_available_utc="2025-01-01T00:00:00Z")
    delayed = run_etas_pipeline(
        config,
        delayed_catalog,
        family,
        _poisson_inputs(config, delayed_catalog, family, completeness),
        completeness,
        fit_function=_stable_fit,
    )
    available = run_etas_pipeline(
        config,
        available_catalog,
        family,
        _poisson_inputs(config, available_catalog, family, completeness),
        completeness,
        fit_function=_stable_fit,
    )

    assert tuple(item.definition.snapshot_id for item in delayed.attempts) == EXPECTED_SNAPSHOTS
    assert not delayed.failed_snapshot_reasons
    assert delayed.etas_evidence.eligible_for_selection
    assert delayed.model_variant_id.startswith("etas/d25_")
    assert "kde_bw" in delayed.model_variant_id
    assert all(item.succeeded for item in delayed.attempts)

    final_attempt = delayed.attempt("final_validation")
    assert final_attempt.score_selection.target_event_ids == ("delayed-final", "later-final")
    origin, available_at = final_attempt.score_selection.parent_times("delayed-final")
    assert available_at > final_attempt.definition.assessment_end_day > origin
    final_pair = final_attempt.paired_evidence
    available_pair = available.attempt("final_validation").paired_evidence
    assert final_pair is not None and available_pair is not None
    assert final_pair.candidate.target_event_ids == final_pair.uniform.target_event_ids
    assert final_pair.candidate.target_event_ids == ("delayed-final", "later-final")
    later_index = final_pair.candidate.target_event_ids.index("later-final")
    assert (
        final_pair.candidate.event_log_intensities[later_index]
        < available_pair.candidate.event_log_intensities[later_index]
    )
    assert (
        final_attempt.parameter_snapshot_id
        == available.attempt("final_validation").parameter_snapshot_id
    )
    assert re.fullmatch(r"[0-9a-f]{64}", final_attempt.parameter_snapshot_id or "")

    for attempt in delayed.attempts:
        gate = attempt.grid_gate_evidence
        pair = attempt.paired_evidence
        assert gate is not None and pair is not None and gate.passed
        assert tuple(item.cell_size_km for item in gate.resolutions) == (50.0, 25.0, 12.5)
        assert tuple(
            (item.coarse_cell_size_km, item.fine_cell_size_km)
            for item in gate.convergence.comparisons
        ) == ((50.0, 25.0), (25.0, 12.5))
        assert all(
            re.fullmatch(r"[0-9a-f]{64}", item.ordered_cell_masses_sha256)
            for item in gate.resolutions
        )
        assert len(pair.candidate.numerical_gate_evidence_ids) == 3


def test_unstable_fits_are_audited_without_stopping_attempts() -> None:
    config = load_background_config("configs/background.yaml")
    family = _grid_family()
    catalog = _catalog()
    completeness = _completeness(config)
    calls: list[str] = []

    def selective_fit(
        problem: ETASLikelihoodProblem,
        spec: ETASModelSpec,
        **kwargs: object,
    ) -> ETASFitResult:
        del kwargs
        model_id = str(problem.assessment_end_days)
        calls.append(model_id)
        if len(calls) == 2:
            return _unstable_fit()
        if len(calls) == 4:
            return _unstable_fit("synthetic second instability")
        return _stable_fit(problem, spec)

    result = run_etas_pipeline(
        config,
        catalog,
        family,
        _poisson_inputs(config, catalog, family, completeness),
        completeness,
        fit_function=selective_fit,
    )

    assert len(calls) == 5
    assert tuple(item[0] for item in result.failed_snapshot_reasons) == ("fold_2", "fold_4")
    assert "synthetic instability" in result.failed_snapshot_reasons[0][1][0]
    assert "synthetic second instability" in result.failed_snapshot_reasons[1][1][0]
    assert tuple(item.candidate.snapshot_id for item in result.etas_evidence.development_folds) == (
        "fold_1",
        "fold_3",
    )
    assert result.etas_evidence.validation is not None
    assert not result.etas_evidence.eligible_for_selection
    assert result.attempt("fold_2").parameter_snapshot_id is not None
    assert result.attempt("fold_4").parameter_snapshot_id is not None
    assert result.attempt("final_validation").succeeded


@pytest.mark.parametrize(
    "exception_type",
    [ValueError, ArithmeticError, np.linalg.LinAlgError, TypeError, AssertionError],
)
def test_programming_exceptions_propagate_without_becoming_scientific_failures(
    exception_type: type[Exception],
) -> None:
    config = load_background_config("configs/background.yaml")
    family = _grid_family()
    catalog = _catalog()
    completeness = _completeness(config)
    calls: list[str] = []

    def broken_fit(
        problem: ETASLikelihoodProblem,
        spec: ETASModelSpec,
        **kwargs: object,
    ) -> ETASFitResult:
        del problem, spec
        calls.append(str(kwargs["model_id"]))
        raise exception_type("synthetic programming defect")

    with pytest.raises(exception_type, match="synthetic programming defect"):
        run_etas_pipeline(
            config,
            catalog,
            family,
            _poisson_inputs(config, catalog, family, completeness),
            completeness,
            fit_function=broken_fit,
        )

    assert calls == ["etas/fold_1"]


def test_internal_contract_value_error_propagates_instead_of_becoming_etas_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_background_config("configs/background.yaml")
    family = _grid_family()
    catalog = _catalog()
    completeness = _completeness(config)

    def broken_parameter_snapshot(**_: object) -> str:
        raise ValueError("synthetic internal contract defect")

    monkeypatch.setattr(pipeline_etas, "_parameter_snapshot_id", broken_parameter_snapshot)

    with pytest.raises(ValueError, match="synthetic internal contract defect"):
        run_etas_pipeline(
            config,
            catalog,
            family,
            _poisson_inputs(config, catalog, family, completeness),
            completeness,
            fit_function=_stable_fit,
        )


@pytest.mark.parametrize("helper_name", ("_grid_gate_evidence", "_candidate_score"))
def test_grid_and_score_contract_errors_propagate(
    helper_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_background_config("configs/background.yaml")
    family = _grid_family()
    catalog = _catalog()
    completeness = _completeness(config)

    def broken_helper(**_: object) -> object:
        raise ValueError(f"synthetic {helper_name} contract defect")

    monkeypatch.setattr(pipeline_etas, helper_name, broken_helper)

    with pytest.raises(ValueError, match=f"synthetic {helper_name} contract defect"):
        run_etas_pipeline(
            config,
            catalog,
            family,
            _poisson_inputs(config, catalog, family, completeness),
            completeness,
            fit_function=_stable_fit,
        )


def test_progress_callback_is_observational_and_reports_every_snapshot() -> None:
    config = load_background_config("configs/background.yaml")
    family = _grid_family()
    catalog = _catalog()
    completeness = _completeness(config)
    poisson = _poisson_inputs(config, catalog, family, completeness)

    without_callback = run_etas_pipeline(
        config,
        catalog,
        family,
        poisson,
        completeness,
        fit_function=_stable_fit,
    )
    messages: list[str] = []
    with_callback = run_etas_pipeline(
        config,
        catalog,
        family,
        poisson,
        completeness,
        fit_function=_stable_fit,
        progress=messages.append,
    )

    assert messages == [
        message
        for snapshot_id in EXPECTED_SNAPSHOTS
        for message in (f"etas:{snapshot_id}:start", f"etas:{snapshot_id}:done")
    ]
    assert with_callback.protocol_sha256 == without_callback.protocol_sha256
    assert with_callback.model_variant_id == without_callback.model_variant_id
    assert with_callback.failed_snapshot_reasons == without_callback.failed_snapshot_reasons
    for observed, expected in zip(
        with_callback.attempts,
        without_callback.attempts,
        strict=True,
    ):
        assert observed.parameter_snapshot_id == expected.parameter_snapshot_id
        assert observed.failure_reasons == expected.failure_reasons
        assert observed.paired_evidence is not None
        assert expected.paired_evidence is not None
        assert observed.paired_evidence.candidate.score_id == (
            expected.paired_evidence.candidate.score_id
        )
        assert observed.paired_evidence.uniform.score_id == (
            expected.paired_evidence.uniform.score_id
        )


def test_three_grid_failure_retains_complete_gate_evidence_and_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_background_config("configs/background.yaml")
    family = _grid_family()
    catalog = _catalog()
    completeness = _completeness(config)
    original = diagnose_three_grid_convergence
    calls = 0

    def fail_second_gate(
        grid_family: EqualAreaGridFamily,
        masses_by_cell_size_km: Mapping[float, Mapping[str, float]],
    ) -> ThreeGridConvergenceGateEvidence:
        nonlocal calls
        calls += 1
        evidence = original(grid_family, masses_by_cell_size_km)
        if calls != 2:
            return evidence
        failed_primary = replace(
            evidence.primary_25_to_12_5,
            relative_expected_count_difference=0.5,
        )
        return ThreeGridConvergenceGateEvidence(
            diagnostic_50_to_25=evidence.diagnostic_50_to_25,
            primary_25_to_12_5=failed_primary,
        )

    monkeypatch.setattr(pipeline_etas, "diagnose_three_grid_convergence", fail_second_gate)
    result = run_etas_pipeline(
        config,
        catalog,
        family,
        _poisson_inputs(config, catalog, family, completeness),
        completeness,
        fit_function=_stable_fit,
    )

    assert calls == 5
    assert tuple(item[0] for item in result.failed_snapshot_reasons) == ("fold_2",)
    failed = result.attempt("fold_2")
    assert failed.paired_evidence is None
    assert failed.grid_gate_evidence is not None
    assert not failed.grid_gate_evidence.passed
    assert tuple(item.cell_size_km for item in failed.grid_gate_evidence.resolutions) == (
        50.0,
        25.0,
        12.5,
    )
    assert len(failed.grid_gate_evidence.convergence.comparisons) == 2
    assert "25->12.5km convergence failed" in failed.failure_reasons[0]
    assert result.attempt("fold_3").succeeded
    assert result.attempt("final_validation").succeeded
