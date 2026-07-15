from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect
from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

import numpy as np
import pytest

from seismoflux.anomaly_increment.compute import Stage4WorkerPlan
from seismoflux.anomaly_increment.contracts import (
    FeatureColumnContract,
    FrozenTargetRateHead,
)
from seismoflux.anomaly_increment.deliverables import (
    assert_prospective_payload_is_target_blind,
)
from seismoflux.anomaly_increment.evaluation import (
    AdoptionDecision,
    BootstrapSample,
    EventHorizonMembership,
    GateOutcome,
    frozen_same_recall_budget_grid,
    stratified_five_horizon_bootstrap_indices,
)
from seismoflux.anomaly_increment.integration import (
    composite_midpoint_quadrature,
    lead_decay,
)
from seismoflux.anomaly_increment.model import (
    RidgePoissonFitResult,
    SharedObjectiveProtocol,
    fit_frozen_target_rate_head,
    fit_shared_ridge_poisson,
)
from seismoflux.anomaly_increment.placebo import (
    InfrastructureInterruption,
    PermutationTestResult,
)
from seismoflux.anomaly_increment.preregistration import Stage4SeedContext
from seismoflux.anomaly_increment.scoring_pipeline import (
    AssembledEvaluationExposure,
    AssembledFitScope,
    AssembledProspectiveIssue,
    CompletedPlaceboReplication,
    EvaluationRegionBinding,
    EvaluationScope,
    ExposureVariantScore,
    FeatureLayout,
    PlaceboExecution,
    PlaceboInjection,
    PlaceboReplicateInput,
    PlaceboRequest,
    PlaceboScientificFailure,
    PlaceboSource,
    RefitPrimaryMacroResult,
    Stage4InMemoryPlan,
    fit_and_primary_macro_statistic,
    run_stage4_in_memory_pipeline,
)

HORIZONS = (7, 30, 90, 180, 365)
CELL_IDS = ("z-cell", "a-cell", "m-cell")
CELL_ROWS = (-1, -1, 0)
CELL_COLUMNS = (-1, 0, -1)
CELL_AREA = np.asarray([300_000.0, 300_000.0, 300_000.0], dtype=np.float64)
BACKGROUND_MASS = np.asarray([1.0 / 3.0] * 3, dtype=np.float64)
FEATURES = {
    "coverage": np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
    "snapshot": np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
    "trend": np.asarray([-1.0, 0.0, 1.0], dtype=np.float64),
}


def _contracts() -> tuple[FeatureColumnContract, ...]:
    return tuple(
        FeatureColumnContract(
            source_column=name,
            logical_feature=name,
            value_output_column=f"{name}_value",
            missing_output_column=f"{name}_missing",
            transform="identity_finite",
        )
        for name in ("coverage", "snapshot", "trend")
    )


def _layout() -> FeatureLayout:
    return FeatureLayout(
        coverage_sources=("coverage",),
        snapshot_sources=("coverage", "snapshot"),
        dynamic_sources=("coverage", "snapshot", "trend"),
    )


def _fit_scope(
    evaluation_id: str,
    *,
    null_dynamic: bool = False,
) -> AssembledFitScope:
    quadrature = composite_midpoint_quadrature(7.0)
    trend = np.zeros(3, dtype=np.float64) if null_dynamic else FEATURES["trend"]
    columns = {
        "coverage": FEATURES["coverage"],
        "snapshot": FEATURES["snapshot"],
        "trend": trend,
    }
    event_trend = 0.0 if null_dynamic else 1.0
    return AssembledFitScope(
        evaluation_id=evaluation_id,
        preprocessing_fit_columns=columns,
        issue_cell_feature_columns=columns,
        event_feature_columns={
            "coverage": np.asarray([0.0], dtype=np.float64),
            "snapshot": np.asarray([0.0], dtype=np.float64),
            "trend": np.asarray([event_trend], dtype=np.float64),
        },
        background_spatial_mass_by_row_and_bin=np.asarray(
            [[1.0 / 3.0, 1.0 / 3.0]] * 3,
            dtype=np.float64,
        ),
        midpoint_widths_days=quadrature.widths_days,
        midpoint_decays=np.asarray(
            lead_decay(quadrature.lead_midpoints_days),
            dtype=np.float64,
        ),
        event_background_intensity=np.asarray(
            [(1.0 / 3.0) / 300_000.0],
            dtype=np.float64,
        ),
        event_decay=np.asarray([lead_decay(1.0)], dtype=np.float64),
        event_magnitude_bin_ids=("M5_6",),
        training_event_counts={"M5_6": 1, "M6_plus": 0},
        background_exposures={"M5_6": 7.0, "M6_plus": 7.0},
    )


def _exposures(
    evaluation_id: str,
    scenario: str,
) -> tuple[AssembledEvaluationExposure, ...]:
    if scenario not in {"positive", "negative", "insufficient"}:
        raise ValueError(scenario)
    event_ids = tuple(f"event-{index:02d}" for index in range(20))
    event_cell = 2 if scenario != "negative" else 0
    output: list[AssembledEvaluationExposure] = []
    for sequence, horizon in enumerate(HORIZONS, start=1):
        present = not (scenario == "insufficient" and horizon == 7)
        supported = event_ids if present else ()
        output.append(
            AssembledEvaluationExposure(
                evaluation_id=evaluation_id,
                issue_date=f"2023-01-{sequence:02d}",
                horizon_days=horizon,
                magnitude_bin="M5_6",
                support_id="synthetic-support-v1",
                compensator_domain_id="synthetic-domain-v1",
                cell_order_ids=CELL_IDS,
                cell_rows=CELL_ROWS,
                cell_columns=CELL_COLUMNS,
                cell_area_km2=CELL_AREA,
                background_spatial_mass=BACKGROUND_MASS,
                cell_feature_columns=FEATURES,
                supported_event_ids=supported,
                all_study_area_event_ids=supported,
                event_cell_indices=(event_cell,) * len(supported),
                event_lead_days=np.ones(len(supported), dtype=np.float64),
            )
        )
        output.append(
            AssembledEvaluationExposure(
                evaluation_id=evaluation_id,
                issue_date=f"2023-01-{sequence:02d}",
                horizon_days=horizon,
                magnitude_bin="M6_plus",
                support_id="synthetic-support-v1",
                compensator_domain_id="synthetic-domain-v1",
                cell_order_ids=CELL_IDS,
                cell_rows=CELL_ROWS,
                cell_columns=CELL_COLUMNS,
                cell_area_km2=CELL_AREA,
                background_spatial_mass=BACKGROUND_MASS,
                cell_feature_columns=FEATURES,
                supported_event_ids=(),
                all_study_area_event_ids=(),
                event_cell_indices=(),
                event_lead_days=np.asarray([], dtype=np.float64),
            )
        )
    return tuple(output)


def _prospective_issues() -> tuple[AssembledProspectiveIssue, ...]:
    magnitude_bins: tuple[Literal["M5_6", "M6_plus"], ...] = ("M5_6", "M6_plus")
    return tuple(
        AssembledProspectiveIssue(
            issue_date="2024-01-01",
            horizon_days=horizon,
            magnitude_bin=magnitude_bin,
            cell_order_ids=CELL_IDS,
            cell_area_km2=CELL_AREA,
            background_spatial_mass=BACKGROUND_MASS,
            cell_feature_columns=FEATURES,
        )
        for magnitude_bin in magnitude_bins
        for horizon in HORIZONS
    )


def _plan(scenario: str) -> Stage4InMemoryPlan:
    development = tuple(
        EvaluationScope(
            fit=_fit_scope(f"development-fold-{index}"),
            exposures=_exposures(f"development-fold-{index}", scenario),
        )
        for index in (1, 2, 3)
    )
    return Stage4InMemoryPlan(
        feature_contracts=_contracts(),
        feature_layout=_layout(),
        development_scopes=development,
        formal_scope=EvaluationScope(
            fit=_fit_scope("formal-validation"),
            exposures=_exposures("formal-validation", scenario),
        ),
        prospective_issues=_prospective_issues(),
        evaluation_region_binding=EvaluationRegionBinding(
            all_construction_zone_ids=tuple(
                f"construction-zone-{index:02d}" for index in range(1, 40)
            ),
            cell_ids=CELL_IDS,
            cell_construction_zone_ids=(
                "construction-zone-01",
                "construction-zone-02",
                "construction-zone-03",
            ),
            event_ids=tuple(f"event-{index:02d}" for index in range(20)),
            event_construction_zone_ids=(
                ("construction-zone-01",) * 20
                if scenario == "negative"
                else ("construction-zone-03",) * 20
            ),
            cell_mapping_sha256="b" * 64,
            event_mapping_sha256="c" * 64,
        ),
        frozen_input_seal_sha256="a" * 64,
        model_version="synthetic-stage4-v1",
    )


def _null_placebo_injection(plan: Stage4InMemoryPlan) -> PlaceboInjection:
    null_scope = _fit_scope("formal-validation", null_dynamic=True)
    exposures = plan.formal_scope.exposures

    def inject(request: PlaceboRequest) -> PlaceboExecution:
        def build(index: int) -> PlaceboReplicateInput:
            return PlaceboReplicateInput(
                replication_index=index,
                mapping_sha256=hashlib.sha256(
                    f"{request.kind}:{request.model_variant}:{index}".encode()
                ).hexdigest(),
                fit_scope=null_scope,
                exposures=exposures,
            )

        return PlaceboExecution(
            source=PlaceboSource(
                source_id_sha256=hashlib.sha256(
                    f"lazy-null:{request.kind}:{request.model_variant}".encode()
                ).hexdigest(),
                frozen_rate_head_sha256=request.frozen_rate_head_sha256,
                replicate_factory=build,
                mapping_sha256_factory=lambda index: hashlib.sha256(
                    f"{request.kind}:{request.model_variant}:{index}".encode()
                ).hexdigest(),
            ),
        )

    return inject


def test_positive_synthetic_e2e_passes_gates_and_builds_separate_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from seismoflux.anomaly_increment import scoring_pipeline

    original_fit = fit_shared_ridge_poisson
    original_rate_head_fit = fit_frozen_target_rate_head
    objective_shapes: list[tuple[str, int, int]] = []
    rate_head_fit_count = 0

    def capture_grouped_objective(objective: Any) -> RidgePoissonFitResult:
        objective_shapes.append(
            (
                type(objective).__name__,
                objective.issue_design.row_count,
                objective.midpoint_widths_days.size,
            )
        )
        return original_fit(cast(SharedObjectiveProtocol, objective))

    def capture_rate_head(
        *,
        training_event_counts: Mapping[str, int],
        background_exposures: Mapping[str, float],
    ) -> FrozenTargetRateHead:
        nonlocal rate_head_fit_count
        rate_head_fit_count += 1
        return original_rate_head_fit(
            training_event_counts=training_event_counts,
            background_exposures=background_exposures,
        )

    monkeypatch.setattr(
        scoring_pipeline,
        "fit_shared_ridge_poisson",
        capture_grouped_objective,
    )
    monkeypatch.setattr(
        scoring_pipeline,
        "fit_frozen_target_rate_head",
        capture_rate_head,
    )
    plan = _plan("positive")
    result = run_stage4_in_memory_pipeline(
        plan,
        placebo_injection=_null_placebo_injection(plan),
    )

    assert result.dynamic_g2.status == "passed"
    assert result.dynamic_g3.status == "passed"
    assert result.adoption.choice == "dynamic"
    assert result.adoption.status == "adopted"
    assert result.dynamic_time_permutation is not None
    assert result.dynamic_time_permutation.monte_carlo_p_value == pytest.approx(1.0 / 1_001.0)
    assert result.dynamic_space_permutation is not None
    assert result.static_svg.startswith("<svg")
    diagnostics = result.publication_diagnostics
    assert f'data-diagnostics-sha256="{diagnostics.content_sha256}"' in result.static_svg
    assert len(diagnostics.information_gain_intervals) == 10
    assert all(
        item.evidence_status == "evidence_insufficient_zero_events"
        and item.independent_event_count == 0
        and item.point_estimate is None
        and item.lower_95 is None
        and item.upper_95 is None
        for item in diagnostics.information_gain_intervals
        if item.magnitude_bin == "M6_plus"
    )
    assert all(
        item.point_estimate is not None
        and item.lower_95 is not None
        and item.upper_95 is not None
        and item.evidence_status == "evidence_insufficient_no_random_split"
        for item in diagnostics.information_gain_intervals
        if item.magnitude_bin == "M5_6" and item.horizon_days in {180, 365}
    )
    assert len(diagnostics.region_ids) == 39
    assert len(diagnostics.region_horizon_metrics) == 39 * 5
    zone_03 = next(
        item
        for item in diagnostics.region_horizon_metrics
        if item.region_id == "zone-03" and item.horizon_days == 7
    )
    assert zone_03.supported_event_count == 20
    assert zone_03.all_study_area_event_count == 20
    assert zone_03.strict_recall == 1.0
    formal_h7 = {
        item.variant: item
        for item in result.exposure_scores
        if item.evaluation_id == "formal-validation"
        and item.magnitude_bin == "M5_6"
        and item.horizon_days == 7
    }
    dynamic_h7 = formal_h7["dynamic"]
    background_h7 = formal_h7["background_no_increment"]
    expected_zone_03_gain = (
        sum(
            dynamic_value - background_value
            for dynamic_value, background_value in zip(
                dynamic_h7.event_log_intensities,
                background_h7.event_log_intensities,
                strict=True,
            )
        )
        - (dynamic_h7.cell_integrated_intensities[2] - background_h7.cell_integrated_intensities[2])
    ) / 20
    assert zone_03.information_gain_nats_per_event == pytest.approx(expected_zone_03_gain)
    assert len(diagnostics.alarm_budget_curves) == 6
    assert all(len(item.points) == 5 for item in diagnostics.alarm_budget_curves)
    assert len(diagnostics.same_recall_area_reductions) == 3
    assert diagnostics.permutation_distributions[0].null_statistics == (
        result.dynamic_time_permutation.null_statistics
    )
    assert "construction-zone-03" not in result.static_svg
    assert "construction-zone-03" not in result.interactive_html
    assert 'id="retrospectivePanel"' in result.interactive_html
    assert 'id="prospectivePanel"' in result.interactive_html
    assert result.prospective.forecast_status == ("retrospective_generated_target_blind_shadow")
    assert "回溯生成的目标盲影子预测" in result.interactive_html
    assert "真正前瞻预测" not in result.interactive_html
    assert_prospective_payload_is_target_blind(result.prospective)
    assert all(scope.optimizer_run_count == 3 for scope in result.fitted_scopes)
    assert all(
        scope.rate_head.by_id("M6_plus").rate_multiplier == 0.0 for scope in result.fitted_scopes
    )
    assert objective_shapes
    assert all(name == "GroupedMidpointSharedPoissonObjective" for name, _, _ in objective_shapes)
    assert all(
        row_count == 3 and midpoint_count == 7 for _, row_count, midpoint_count in objective_shapes
    )
    assert rate_head_fit_count == 4


def test_dynamic_g2_hard_stops_when_space_placebo_failures_exceed_one_percent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from seismoflux.anomaly_increment import scoring_pipeline

    def controlled_permutation(
        _injection: object,
        *,
        kind: str,
        variant: str,
        observed: float | None,
        **_kwargs: object,
    ) -> PermutationTestResult | None:
        if observed is None:
            return None
        failure_count = 11 if (kind, variant) == ("space", "dynamic") else 0
        nulls = (float("inf"),) * failure_count + (0.0,) * (1_000 - failure_count)
        greater_or_equal = sum(value >= observed for value in nulls)
        return PermutationTestResult(
            observed_statistic=observed,
            null_statistics=nulls,
            null_greater_or_equal_count=greater_or_equal,
            scientific_failure_count=failure_count,
            scientific_failure_fraction=failure_count / 1_000.0,
            monte_carlo_p_value=(1 + greater_or_equal) / 1_001.0,
            denominator=1_001,
            status="evidence_insufficient" if failure_count else "passed",
        )

    monkeypatch.setattr(scoring_pipeline, "BOOTSTRAP_REPLICATIONS", 64)
    monkeypatch.setattr(
        scoring_pipeline,
        "_permutation_result",
        controlled_permutation,
    )
    result = run_stage4_in_memory_pipeline(
        _plan("positive"),
        placebo_injection=cast(PlaceboInjection, object()),
    )

    assert result.dynamic_g2.status == "evidence_insufficient"
    assert result.dynamic_g2.reasons == (
        "space_permutation_scientific_failure_fraction_above_0_01",
    )
    spatial = next(
        item
        for item in result.publication_diagnostics.permutation_distributions
        if item.kind == "space" and item.variant == "dynamic"
    )
    assert spatial.evidence_status == ("evidence_insufficient_scientific_failure_fraction")


def test_pipeline_is_deterministic_and_no_anomaly_variants_degenerate_to_background() -> None:
    plan = _plan("positive")
    first = run_stage4_in_memory_pipeline(plan, placebo_injection=None)
    second = run_stage4_in_memory_pipeline(plan, placebo_injection=None)

    assert first.result_fingerprint_sha256 == second.result_fingerprint_sha256
    assert first.static_svg == second.static_svg
    assert first.interactive_html == second.interactive_html
    assert first.prospective == second.prospective
    formal = [item for item in first.exposure_scores if item.evaluation_id == "formal-validation"]
    for issue_date in {item.issue_date for item in formal}:
        for magnitude_bin in ("M5_6", "M6_plus"):
            selected = [
                item
                for item in formal
                if item.issue_date == issue_date and item.magnitude_bin == magnitude_bin
            ]
            by_variant = {item.variant: item for item in selected}
            background = by_variant["background_no_increment"]
            assert background.selected_cell_ids_at_600000_km2 == (
                "z-cell",
                "a-cell",
            )
            for variant in ("coverage_only", "snapshot"):
                candidate = by_variant[variant]
                assert candidate.event_log_intensities == background.event_log_intensities
                assert candidate.integrated_intensity == background.integrated_intensity
                assert (
                    candidate.cell_integrated_intensities == background.cell_integrated_intensities
                )
                assert (
                    candidate.selected_cell_ids_at_600000_km2
                    == background.selected_cell_ids_at_600000_km2
                )


@pytest.mark.parametrize(
    ("scenario", "g2_status", "g3_status", "adoption_status"),
    (
        ("negative", "failed", "failed", "credible_negative"),
        (
            "insufficient",
            "evidence_insufficient",
            "evidence_insufficient",
            "evidence_insufficient",
        ),
    ),
)
def test_negative_and_insufficient_synthetic_e2e_stop_conservatively(
    scenario: str,
    g2_status: str,
    g3_status: str,
    adoption_status: str,
) -> None:
    plan = _plan(scenario)
    injection = _null_placebo_injection(plan) if scenario == "negative" else None
    result = run_stage4_in_memory_pipeline(plan, placebo_injection=injection)

    assert result.dynamic_g2.status == g2_status
    assert result.dynamic_g3.status == g3_status
    assert result.adoption.choice == "background_only"
    assert result.adoption.status == adoption_status
    assert all(
        item.result_status == "evidence_insufficient"
        for item in result.retrospective.horizon_results
        if item.horizon_days in {180, 365}
    )


def test_placebo_refit_reuses_rate_head_and_prospective_type_is_physically_target_blind() -> None:
    plan = _plan("positive")
    formal_rate_head = (
        run_stage4_in_memory_pipeline(
            plan,
            placebo_injection=None,
        )
        .fitted_scopes[-1]
        .rate_head
    )
    before = formal_rate_head.sha256
    refit = fit_and_primary_macro_statistic(
        _fit_scope("formal-validation", null_dynamic=True),
        plan.formal_scope.exposures,
        plan.feature_contracts,
        plan.feature_layout,
        "dynamic",
        formal_rate_head,
    )

    assert refit.statistic is not None
    assert abs(float(refit.statistic)) <= 1.0e-14
    assert refit.frozen_rate_head_sha256 == before == formal_rate_head.sha256
    assert refit.anomaly_fit_sha256 is not None
    prospective_fields = {field.name for field in dataclasses.fields(AssembledProspectiveIssue)}
    assert not prospective_fields & {
        "supported_event_ids",
        "all_study_area_event_ids",
        "event_cell_indices",
        "event_lead_days",
    }
    with pytest.raises(TypeError):
        AssembledProspectiveIssue(
            issue_date="2024-01-01",
            horizon_days=7,
            magnitude_bin="M5_6",
            cell_order_ids=CELL_IDS,
            cell_area_km2=CELL_AREA,
            background_spatial_mass=BACKGROUND_MASS,
            cell_feature_columns=FEATURES,
            supported_event_ids=("forbidden",),  # type: ignore[call-arg]
        )


def test_pipeline_module_has_no_path_io_or_target_loader() -> None:
    from seismoflux.anomaly_increment import scoring_pipeline

    tree = ast.parse(inspect.getsource(scoring_pipeline))
    forbidden_import_roots = {"pathlib", "pandas", "polars", "pyarrow"}
    forbidden_calls = {
        "open",
        "Path",
        "read_csv",
        "read_json",
        "read_parquet",
        "to_csv",
        "to_json",
        "to_parquet",
    }
    imported_roots = {
        alias.name.split(".", maxsplit=1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".", maxsplit=1)[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    call_names = {
        node.func.id
        if isinstance(node.func, ast.Name)
        else node.func.attr
        if isinstance(node.func, ast.Attribute)
        else ""
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }

    assert not imported_roots & forbidden_import_roots
    assert not call_names & forbidden_calls


def test_evaluation_region_receipts_fail_closed_without_full_formal_coverage() -> None:
    plan = _plan("positive")
    binding = plan.evaluation_region_binding
    missing_event = dataclasses.replace(
        binding,
        event_ids=binding.event_ids[:-1],
        event_construction_zone_ids=binding.event_construction_zone_ids[:-1],
    )
    with pytest.raises(ValueError, match="does not cover every formal target event"):
        dataclasses.replace(plan, evaluation_region_binding=missing_event)

    missing_cell = dataclasses.replace(
        binding,
        cell_ids=binding.cell_ids[:-1],
        cell_construction_zone_ids=binding.cell_construction_zone_ids[:-1],
    )
    with pytest.raises(ValueError, match="does not cover every formal exposure cell"):
        dataclasses.replace(plan, evaluation_region_binding=missing_cell)

    prospective_fields = {field.name for field in dataclasses.fields(AssembledProspectiveIssue)}
    assert not prospective_fields & {
        "cell_construction_zone_ids",
        "event_construction_zone_ids",
        "evaluation_region_binding",
    }


def test_fixed_origin_grid_preserves_signed_row_and_column_indices() -> None:
    exposure = _exposures("formal-validation", "positive")[0]
    assert exposure.cell_rows == CELL_ROWS
    assert exposure.cell_columns == CELL_COLUMNS
    assert min(exposure.cell_rows) < 0 and min(exposure.cell_columns) < 0
    with pytest.raises(ValueError, match="signed integers"):
        dataclasses.replace(exposure, cell_rows=(False, -1, 0))


def _placebo_worker_plan(workers: int) -> Stage4WorkerPlan:
    return Stage4WorkerPlan(
        physical_cores=workers + 2,
        logical_processors=workers + 2,
        reserve_physical_cores=2,
        configured_max_workers=workers,
        effective_workers=workers,
        blas_threads_per_worker=1,
        nested_parallelism=False,
    )


def test_lazy_placebo_source_builds_only_the_requested_replication() -> None:
    plan = _plan("positive")
    built: list[int] = []
    request = PlaceboRequest(
        kind="time",
        evaluation_id="formal-validation",
        model_variant="dynamic",
        observed_statistic=0.5,
        frozen_rate_head_sha256="a" * 64,
    )

    def build(index: int) -> PlaceboReplicateInput:
        built.append(index)
        return PlaceboReplicateInput(
            replication_index=index,
            mapping_sha256=hashlib.sha256(str(index).encode()).hexdigest(),
            fit_scope=plan.formal_scope.fit,
            exposures=plan.formal_scope.exposures,
        )

    source = PlaceboSource(
        source_id_sha256="b" * 64,
        frozen_rate_head_sha256=request.frozen_rate_head_sha256,
        replicate_factory=build,
        mapping_sha256_factory=lambda index: hashlib.sha256(str(index).encode()).hexdigest(),
    )
    execution = PlaceboExecution(source=source)

    execution.validate_for(request)
    assert built == []
    assert not hasattr(source, "replicate_inputs")
    assert source.mapping_sha256(request, 731) == hashlib.sha256(b"731").hexdigest()
    assert built == []
    item = source.build(request, 731)
    assert item.replication_index == 731
    assert built == [731]


def test_streaming_placebo_results_are_identical_for_one_two_and_four_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from seismoflux.anomaly_increment import scoring_pipeline

    plan = _plan("positive")
    fingerprints: list[str] = []
    callback_digests: list[str] = []

    def cheap_refit(
        scope: AssembledFitScope,
        exposures: object,
        contracts: object,
        layout: object,
        variant: object,
        frozen_rate_head: object,
    ) -> RefitPrimaryMacroResult:
        del scope, exposures, contracts, layout, variant
        rate_head = frozen_rate_head  # keep the pure callback independent of worker order
        return RefitPrimaryMacroResult(
            statistic=0.0,
            frozen_rate_head_sha256=rate_head.sha256,  # type: ignore[attr-defined]
            fitted_preprocessor_sha256="c" * 64,
            anomaly_fit_sha256="d" * 64,
        )

    monkeypatch.setattr(scoring_pipeline, "fit_and_primary_macro_statistic", cheap_refit)

    def make_injection(
        worker_count: int,
        callback_rows: list[tuple[str, str, int, str]],
    ) -> PlaceboInjection:
        def inject(request: PlaceboRequest) -> PlaceboExecution:
            def build(index: int) -> PlaceboReplicateInput:
                return PlaceboReplicateInput(
                    replication_index=index,
                    mapping_sha256=hashlib.sha256(
                        f"{request.kind}:{request.model_variant}:{index}".encode()
                    ).hexdigest(),
                    fit_scope=plan.formal_scope.fit,
                    exposures=plan.formal_scope.exposures,
                )

            def record(
                observed_request: PlaceboRequest,
                result: CompletedPlaceboReplication,
            ) -> None:
                callback_rows.append(
                    (
                        observed_request.kind,
                        observed_request.model_variant,
                        result.replication_index,
                        result.mapping_sha256,
                    )
                )

            return PlaceboExecution(
                source=PlaceboSource(
                    source_id_sha256=hashlib.sha256(
                        f"worker-invariance:{request.kind}:{request.model_variant}".encode()
                    ).hexdigest(),
                    frozen_rate_head_sha256=request.frozen_rate_head_sha256,
                    replicate_factory=build,
                    mapping_sha256_factory=lambda index: hashlib.sha256(
                        f"{request.kind}:{request.model_variant}:{index}".encode()
                    ).hexdigest(),
                ),
                worker_plan=_placebo_worker_plan(worker_count),
                max_in_flight=worker_count,
                on_result=record,
            )

        return inject

    for workers in (1, 2, 4):
        callback_rows: list[tuple[str, str, int, str]] = []
        result = run_stage4_in_memory_pipeline(
            plan,
            placebo_injection=make_injection(workers, callback_rows),
        )
        fingerprints.append(result.result_fingerprint_sha256)
        assert len(callback_rows) == 3_000
        for kind, variant in (("time", "dynamic"), ("space", "dynamic"), ("time", "snapshot")):
            indices = [
                index
                for row_kind, row_variant, index, _ in callback_rows
                if (row_kind, row_variant) == (kind, variant)
            ]
            assert indices == list(range(1_000))
        callback_digests.append(hashlib.sha256(repr(callback_rows).encode()).hexdigest())

    assert len(set(fingerprints)) == 1
    assert len(set(callback_digests)) == 1


def test_scientific_failure_is_checkpointable_but_infrastructure_error_stops_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from seismoflux.anomaly_increment import scoring_pipeline

    plan = _plan("positive")
    calls = 0
    completed: list[CompletedPlaceboReplication] = []
    interruptions: list[type[BaseException]] = []

    def refit(*_: object) -> RefitPrimaryMacroResult:
        nonlocal calls
        current = calls
        calls += 1
        if current == 0:
            raise PlaceboScientificFailure(
                "optimizer_nonconvergence",
                "synthetic scientific nonconvergence",
            )
        if current == 1:
            raise FloatingPointError("synthetic scientific numerical instability")
        raise OSError("synthetic checkpoint infrastructure interruption")

    monkeypatch.setattr(scoring_pipeline, "fit_and_primary_macro_statistic", refit)

    def inject(request: PlaceboRequest) -> PlaceboExecution:
        def build(index: int) -> PlaceboReplicateInput:
            return PlaceboReplicateInput(
                replication_index=index,
                mapping_sha256=hashlib.sha256(str(index).encode()).hexdigest(),
                fit_scope=plan.formal_scope.fit,
                exposures=plan.formal_scope.exposures,
            )

        return PlaceboExecution(
            source=PlaceboSource(
                source_id_sha256="e" * 64,
                frozen_rate_head_sha256=request.frozen_rate_head_sha256,
                replicate_factory=build,
                mapping_sha256_factory=lambda index: hashlib.sha256(
                    str(index).encode()
                ).hexdigest(),
            ),
            on_result=lambda _, result: completed.append(result),
            on_interruption=lambda _, error: interruptions.append(type(error)),
        )

    with pytest.raises(OSError, match="infrastructure interruption"):
        run_stage4_in_memory_pipeline(plan, placebo_injection=inject)

    assert len(completed) == 2
    assert completed[0].converged is False
    assert completed[0].scientific_failure_code == "optimizer_nonconvergence"
    assert completed[1].scientific_failure_code == "numerical_instability"
    assert interruptions == [OSError]


def test_recovered_mapping_identity_mismatch_fails_before_any_refit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from seismoflux.anomaly_increment import scoring_pipeline

    plan = _plan("positive")
    monkeypatch.setattr(
        scoring_pipeline,
        "fit_and_primary_macro_statistic",
        lambda *_: pytest.fail("mapping mismatch must fail before refitting"),
    )

    def inject(request: PlaceboRequest) -> PlaceboExecution:
        def build(index: int) -> PlaceboReplicateInput:
            del index
            pytest.fail("recovered mapping verification must not rebuild placebo inputs")

        return PlaceboExecution(
            source=PlaceboSource(
                source_id_sha256="f" * 64,
                frozen_rate_head_sha256=request.frozen_rate_head_sha256,
                replicate_factory=build,
                mapping_sha256_factory=lambda index: hashlib.sha256(
                    str(index).encode()
                ).hexdigest(),
            ),
            completed_results=(
                CompletedPlaceboReplication(
                    replication_index=0,
                    mapping_sha256="0" * 64,
                    statistic=0.0,
                    converged=True,
                ),
            ),
        )

    with pytest.raises(InfrastructureInterruption, match="mapping differs"):
        run_stage4_in_memory_pipeline(plan, placebo_injection=inject)


def _bootstrap_scores() -> tuple[ExposureVariantScore, ...]:
    budgets = np.asarray(frozen_same_recall_budget_grid(), dtype=np.float64)
    budgets.setflags(write=False)
    prefix_counts = (0, *(1 for _ in range(1, budgets.size)))
    variant_offsets = {
        "background_no_increment": 0.0,
        "coverage_only": 0.1,
        "snapshot": 0.2,
        "dynamic": 0.3,
    }
    output: list[ExposureVariantScore] = []
    for magnitude_bin, count in (("M5_6", 20), ("M6_plus", 4)):
        event_ids = tuple(f"{magnitude_bin.casefold()}-event-{index}" for index in range(count))
        for horizon in HORIZONS:
            for variant, offset in variant_offsets.items():
                output.append(
                    ExposureVariantScore(
                        evaluation_id="formal-validation",
                        issue_date=f"2023-{horizon:03d}",
                        horizon_days=horizon,
                        magnitude_bin=cast(Literal["M5_6", "M6_plus"], magnitude_bin),
                        variant=cast(
                            Literal[
                                "background_no_increment",
                                "coverage_only",
                                "snapshot",
                                "dynamic",
                            ],
                            variant,
                        ),
                        supported_event_ids=event_ids,
                        all_study_area_event_ids=event_ids,
                        event_log_intensities=(offset,) * count,
                        integrated_intensity=10.0 - offset,
                        cell_integrated_intensities=(10.0 - offset,),
                        alarm_exact_selected_area_km2=budgets,
                        alarm_prefix_cell_counts=prefix_counts,
                        supported_event_alarm_ranks=(0,) * count,
                        selected_cell_ids_at_600000_km2=("aggregate-cell",),
                        strict_hit_event_ids_at_600000_km2=event_ids,
                    )
                )
    return tuple(output)


def test_m6_bootstrap_uses_the_joint_physical_event_draw_without_entering_primary_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from seismoflux.anomaly_increment import scoring_pipeline

    scores = _bootstrap_scores()
    original_sampler = stratified_five_horizon_bootstrap_indices
    sampled_event_sets: list[tuple[tuple[str, str], ...]] = []

    def capture_sampler(
        events: Sequence[EventHorizonMembership],
        *,
        context: Stage4SeedContext,
    ) -> BootstrapSample:
        typed_events = tuple(events)
        sampled_event_sets.append(
            tuple((item.event_id, item.magnitude_bin) for item in typed_events)
        )
        return original_sampler(typed_events, context=context)

    monkeypatch.setattr(scoring_pipeline, "BOOTSTRAP_REPLICATIONS", 32)
    monkeypatch.setattr(
        scoring_pipeline,
        "stratified_five_horizon_bootstrap_indices",
        capture_sampler,
    )
    with_m6 = scoring_pipeline._bootstrap_summary(
        scores,
        frozen_input_seal_sha256="a" * 64,
    )

    assert len(sampled_event_sets) == 32
    assert all(
        {magnitude_bin for _, magnitude_bin in event_set} == {"M5_6", "M6_plus"}
        for event_set in sampled_event_sets
    )
    assert all(
        ("dynamic", "M6_plus", horizon) in with_m6.horizon_information_gain_intervals
        for horizon in HORIZONS
    )
    m6_publication = tuple(
        item
        for item in scoring_pipeline._publication_information_gain(scores, with_m6)
        if item.magnitude_bin == "M6_plus"
    )
    assert all(item.independent_event_count == 4 for item in m6_publication)
    assert all(
        item.evidence_status == "exploratory_low_sample"
        for item in m6_publication
        if item.horizon_days in {7, 30, 90}
    )
    assert all(
        item.evidence_status == "exploratory_low_sample_no_random_split"
        for item in m6_publication
        if item.horizon_days in {180, 365}
    )
    retrospective = scoring_pipeline._retrospective_payload(
        scores,
        with_m6,
        dynamic_g2=GateOutcome("G2", "passed", (), ()),
        dynamic_g3=GateOutcome("G3", "passed", (), ()),
        adoption=AdoptionDecision("dynamic", "adopted", "synthetic_regression"),
        time_permutation=None,
        space_permutation=None,
        model_version="synthetic-stage4-v1",
    )
    populated_m6 = tuple(
        item for item in retrospective.horizon_results if item.magnitude_bin == "M6_plus"
    )
    assert len(populated_m6) == len(HORIZONS)
    assert all(item.result_status == "evidence_insufficient" for item in populated_m6)
    assert all(item.independent_event_count == 4 for item in populated_m6)
    assert all(
        item.information_gain_nats_per_event is not None
        and item.information_gain_lower_95 is not None
        and item.information_gain_upper_95 is not None
        for item in populated_m6
    )

    monkeypatch.setattr(
        scoring_pipeline,
        "stratified_five_horizon_bootstrap_indices",
        original_sampler,
    )
    without_m6 = scoring_pipeline._bootstrap_summary(
        tuple(item for item in scores if item.magnitude_bin == "M5_6"),
        frozen_input_seal_sha256="a" * 64,
    )
    assert with_m6.macro_information_gain_intervals == (without_m6.macro_information_gain_intervals)
    assert with_m6.dynamic_minus_coverage_interval == (without_m6.dynamic_minus_coverage_interval)
    assert with_m6.snapshot_minus_coverage_interval == (without_m6.snapshot_minus_coverage_interval)
    assert with_m6.same_area_recall_intervals == without_m6.same_area_recall_intervals
    assert with_m6.same_recall_area_intervals == without_m6.same_recall_area_intervals
