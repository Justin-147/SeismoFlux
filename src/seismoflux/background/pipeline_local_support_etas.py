"""G1-LS ETAS orchestration on snapshot-specific frozen support domains."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np

from seismoflux.background.adapters import (
    build_etas_model_spec,
    build_etas_parameter_bounds,
    build_optimizer_options,
    build_stability_thresholds,
    etas_variant_id,
    point_area_quadrature_from_grid,
)
from seismoflux.background.catalog import EarthquakeCatalog, utc_timestamp_to_day
from seismoflux.background.config import BackgroundConfig
from seismoflux.background.etas_fit import (
    ETASEvent,
    ETASFitResult,
    ETASLikelihoodProblem,
    ETASModelSpec,
    ETASParameterBounds,
    ETASParameters,
    OptimizerOptions,
    StabilityThresholds,
    etas_log_likelihood,
    fit_etas,
)
from seismoflux.background.evidence import (
    EXPECTED_SNAPSHOTS,
    AuditedBackgroundModelEvidence,
    PairedInformationGainEvidence,
    PointProcessScoreEvidence,
)
from seismoflux.background.local_support_runtime import (
    LocalSupportRuntime,
    LocalSupportRuntimeSnapshot,
)
from seismoflux.background.pipeline_etas import (
    ETASEventSelectionAudit,
    ETASFitCallable,
    ETASGridGateEvidence,
    ETASPipelineResult,
    ETASSnapshotAttempt,
    _canonical_sha256,
    _event_payload,
    _grid_gate_evidence,
    _KDEBackgroundDensity,
    _selection_audit,
)
from seismoflux.background.pipeline_poisson import (
    LocalSupportPoissonKDEPipelineResult,
    LocalSupportPoissonSnapshotFit,
)
from seismoflux.background.poisson import SpatialPoissonModel
from seismoflux.background.scoring_authorization import (
    AuthorizedExecution,
    require_background_scoring_authorized,
)
from seismoflux.background.workflow import (
    ETASParentRoleMasks,
    SnapshotDefinition,
    build_local_support_etas_parent_roles,
    build_snapshot_definitions,
    catalog_etas_events,
)

_SENSITIVITY_SUFFIX = "/exclude_unsupported_conditional_parents"


@dataclass(frozen=True, slots=True)
class LocalSupportETASPipelineResult:
    """Primary G1-LS ETAS evidence and the preregistered parent sensitivity."""

    protocol_sha256: str
    authorization_id: str
    primary: ETASPipelineResult
    sensitivity_attempts: tuple[ETASSnapshotAttempt, ...]
    sensitivity_not_applicable: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if self.primary.protocol_sha256 != self.protocol_sha256:
            raise ValueError("local ETAS primary result uses another protocol")
        observed_sensitivity = tuple(
            attempt.definition.snapshot_id for attempt in self.sensitivity_attempts
        )
        if observed_sensitivity != tuple(
            snapshot_id
            for snapshot_id in EXPECTED_SNAPSHOTS
            if snapshot_id in set(observed_sensitivity)
        ):
            raise ValueError("local ETAS sensitivity attempts are out of frozen order")
        not_applicable_ids = tuple(item[0] for item in self.sensitivity_not_applicable)
        if (
            len(set(observed_sensitivity)) != len(observed_sensitivity)
            or len(set(not_applicable_ids)) != len(not_applicable_ids)
            or set(observed_sensitivity).intersection(not_applicable_ids)
            or set(observed_sensitivity).union(not_applicable_ids) != set(EXPECTED_SNAPSHOTS)
        ):
            raise ValueError("local ETAS sensitivity must account for all five snapshots")
        if any(not reason for _, reason in self.sensitivity_not_applicable):
            raise ValueError("local ETAS sensitivity not-applicable reasons must not be empty")
        for attempt in (*self.primary.attempts, *self.sensitivity_attempts):
            if attempt.paired_evidence is None:
                continue
            candidate = attempt.paired_evidence.candidate
            if candidate.authorization_id != self.authorization_id:
                raise ValueError("local ETAS score uses another scoring authorization")
            if candidate.support_id is None:
                raise ValueError("local ETAS score omitted its support identity")

    @property
    def evidence(self) -> AuditedBackgroundModelEvidence:
        return self.primary.evidence

    @property
    def model_variant_id(self) -> str:
        return self.primary.model_variant_id


def _uniform_scores(
    poisson_result: LocalSupportPoissonKDEPipelineResult,
) -> dict[str, PointProcessScoreEvidence]:
    evidence = poisson_result.uniform_evidence
    if evidence.failed_snapshot_reasons or len(evidence.development_folds) != 4:
        raise ValueError("local Poisson input lacks complete uniform evidence")
    if evidence.validation is None:
        raise ValueError("local Poisson input lacks final uniform validation evidence")
    pairs = (*evidence.development_folds, evidence.validation)
    if tuple(item.uniform.snapshot_id for item in pairs) != EXPECTED_SNAPSHOTS:
        raise ValueError("local uniform evidence is not in frozen snapshot order")
    return {item.uniform.snapshot_id: item.uniform for item in pairs}


def _validate_inputs(
    config: BackgroundConfig,
    catalog: EarthquakeCatalog,
    runtime: LocalSupportRuntime,
    poisson_result: LocalSupportPoissonKDEPipelineResult,
    authorization: AuthorizedExecution,
) -> None:
    require_background_scoring_authorized(config, authorization)
    protocol_sha256 = _canonical_sha256(config.model_dump(mode="python"))
    if poisson_result.protocol_sha256 != protocol_sha256:
        raise ValueError("local Poisson and ETAS protocol fingerprints differ")
    if tuple(str(value) for value in catalog.event_id) != runtime.event_ids:
        raise ValueError("local support runtime event order differs from the ETAS catalog")
    expected_definitions = build_snapshot_definitions(config)
    if tuple(item.definition for item in poisson_result.snapshots) != expected_definitions:
        raise ValueError("local Poisson snapshots differ from frozen ETAS definitions")
    for definition, poisson_snapshot, runtime_snapshot in zip(
        expected_definitions,
        poisson_result.snapshots,
        runtime.snapshots,
        strict=True,
    ):
        support = runtime_snapshot.support
        expected = (
            definition.snapshot_id,
            support.support_id,
            support.common_mc,
            support.retained_area_m2 / 1_000_000.0,
            runtime_snapshot.compensator_domain_id,
            authorization.authorization_id,
        )
        observed = (
            poisson_snapshot.definition.snapshot_id,
            poisson_snapshot.support_id,
            poisson_snapshot.selected_mc,
            poisson_snapshot.supported_area_km2,
            poisson_snapshot.compensator_domain_id,
            poisson_snapshot.authorization_id,
        )
        if observed != expected:
            raise ValueError("local Poisson and ETAS support-domain bindings differ")
        if not math.isfinite(support.retained_selected_aki_b_value) or (
            support.retained_selected_aki_b_value <= 0.0
        ):
            raise ValueError("local ETAS requires a finite positive supported-domain b-value")
    if config.etas.spatial_kernel.d_km2 != 25.0:
        raise ValueError("primary local ETAS must keep d fixed at 25 km^2")


def _role_event_ids(
    catalog: EarthquakeCatalog,
    roles: ETASParentRoleMasks,
    causal_parent_mask: np.ndarray[tuple[int], np.dtype[np.bool_]],
) -> dict[str, tuple[str, ...]]:
    return {
        name: tuple(
            str(catalog.event_id[index]) for index in np.flatnonzero(mask & causal_parent_mask)
        )
        for name, mask in (
            ("supported", roles.supported_parent),
            ("true_external_buffer", roles.true_external_buffer_parent),
            ("unsupported_conditional", roles.unsupported_conditional_parent),
        )
    }


def _parameter_snapshot_id(
    *,
    protocol_sha256: str,
    authorization_id: str,
    model_variant_id: str,
    definition: SnapshotDefinition,
    runtime_snapshot: LocalSupportRuntimeSnapshot,
    fit_start_utc: str,
    spec: ETASModelSpec,
    bounds: ETASParameterBounds,
    options: OptimizerOptions,
    thresholds: StabilityThresholds,
    fit_targets: tuple[ETASEvent, ...],
    fit_parents: tuple[ETASEvent, ...],
    parent_role_event_ids: dict[str, tuple[str, ...]],
    fit_result: ETASFitResult,
    poisson_snapshot: LocalSupportPoissonSnapshotFit,
    background_model: SpatialPoissonModel,
) -> str:
    return _canonical_sha256(
        {
            "protocol_sha256": protocol_sha256,
            "authorization_id": authorization_id,
            "model_id": "etas",
            "model_variant_id": model_variant_id,
            "snapshot_id": definition.snapshot_id,
            "support_id": runtime_snapshot.support.support_id,
            "common_mc": runtime_snapshot.support.common_mc,
            "supported_area_km2": runtime_snapshot.support.retained_area_m2 / 1_000_000.0,
            "compensator_domain_id": runtime_snapshot.compensator_domain_id,
            "fit_interval": {
                "start_utc": fit_start_utc,
                "end_utc": definition.fit_end_utc,
            },
            "spec": asdict(spec),
            "bounds": asdict(bounds),
            "optimizer_options": asdict(options),
            "stability_thresholds": asdict(thresholds),
            "fit_target_events": _event_payload(fit_targets),
            "fit_parent_events": _event_payload(fit_parents),
            "fit_parent_roles": parent_role_event_ids,
            "fit_result": asdict(fit_result),
            "selected_background_kde": {
                "bandwidth_km": background_model.bandwidth_km,
                "normalization_mass": background_model.normalization_mass,
                "rate_per_day": background_model.rate_per_day,
                "training_event_ids": poisson_snapshot.training_event_ids,
                "training_evidence_id": poisson_snapshot.training_evidence_id,
            },
        }
    )


def _candidate_score(
    *,
    protocol_sha256: str,
    authorization_id: str,
    model_variant_id: str,
    definition: SnapshotDefinition,
    runtime_snapshot: LocalSupportRuntimeSnapshot,
    parameter_snapshot_id: str,
    problem: ETASLikelihoodProblem,
    parameters: ETASParameters,
    spec: ETASModelSpec,
    uniform_score: PointProcessScoreEvidence,
    numerical_gate_evidence_ids: tuple[str, ...],
) -> PointProcessScoreEvidence:
    likelihood = etas_log_likelihood(problem, parameters, spec)
    if set(likelihood.target_event_ids) != set(uniform_score.target_event_ids):
        raise ValueError("local ETAS targets differ from their paired uniform targets")
    logs_by_id = {
        event_id: math.log(intensity)
        for event_id, intensity in zip(
            likelihood.target_event_ids,
            likelihood.event_intensities,
            strict=True,
        )
    }
    ordered_logs = np.asarray(
        [logs_by_id[event_id] for event_id in uniform_score.target_event_ids],
        dtype=np.float64,
    )
    return PointProcessScoreEvidence(
        protocol_sha256=protocol_sha256,
        model_id="etas",
        model_variant_id=model_variant_id,
        parameter_snapshot_id=parameter_snapshot_id,
        snapshot_id=definition.snapshot_id,
        fit_end_utc=definition.fit_end_utc,
        assessment_start_utc=definition.assessment_start_utc,
        assessment_end_utc=definition.assessment_end_utc,
        selected_mc=runtime_snapshot.support.common_mc,
        target_event_ids=uniform_score.target_event_ids,
        event_log_intensities=ordered_logs,
        compensator=likelihood.total_compensator,
        numerical_gate_evidence_ids=numerical_gate_evidence_ids,
        support_id=runtime_snapshot.support.support_id,
        supported_area_km2=runtime_snapshot.support.retained_area_m2 / 1_000_000.0,
        compensator_domain_id=runtime_snapshot.compensator_domain_id,
        authorization_id=authorization_id,
    )


def _selection_inputs(
    catalog: EarthquakeCatalog,
    definition: SnapshotDefinition,
    runtime_snapshot: LocalSupportRuntimeSnapshot,
    roles: ETASParentRoleMasks,
    spec: ETASModelSpec,
    *,
    fit_start_day: float,
    history_start_day: float,
) -> tuple[
    ETASEventSelectionAudit,
    ETASEventSelectionAudit,
    tuple[ETASEvent, ...],
    tuple[ETASEvent, ...],
    tuple[ETASEvent, ...],
    tuple[ETASEvent, ...],
    np.ndarray[tuple[int], np.dtype[np.bool_]],
]:
    supported = runtime_snapshot.supported_mask
    minimum = runtime_snapshot.support.common_mc
    fit_parent_start = max(history_start_day, fit_start_day - spec.history_parent_cutoff_days)
    score_parent_start = max(
        history_start_day,
        definition.assessment_start_day - spec.history_parent_cutoff_days,
    )
    fit_targets_mask = np.asarray(
        supported
        & (catalog.magnitude >= minimum)
        & (catalog.origin_day > fit_start_day)
        & (catalog.origin_day <= definition.fit_end_day)
        & (catalog.available_day <= definition.fit_end_day),
        dtype=np.bool_,
    )
    fit_parents_mask = np.asarray(
        roles.parent_mask
        & (catalog.origin_day >= fit_parent_start)
        & (catalog.origin_day <= definition.fit_end_day)
        & (catalog.available_day <= definition.fit_end_day),
        dtype=np.bool_,
    )
    score_targets_mask = np.asarray(
        supported
        & (catalog.magnitude >= minimum)
        & (catalog.origin_day > definition.assessment_start_day)
        & (catalog.origin_day <= definition.assessment_end_day),
        dtype=np.bool_,
    )
    score_parents_mask = np.asarray(
        roles.parent_mask
        & (catalog.origin_day >= score_parent_start)
        & (catalog.origin_day <= definition.assessment_end_day)
        & (catalog.available_day <= definition.assessment_end_day),
        dtype=np.bool_,
    )
    # Physical scoring targets are retained even when their catalog publication
    # occurs after the assessment window.  Their true availability timestamp is
    # carried into ETASEvent, so they cannot trigger any event inside the window;
    # including them here only satisfies the likelihood's target-is-a-parent-row
    # identity invariant without leaking their occurrence to earlier targets.
    score_parents_mask |= score_targets_mask
    target_domain = np.asarray(supported, dtype=np.bool_)
    parent_domain = roles.parent_mask
    fit_targets = catalog_etas_events(
        catalog,
        fit_targets_mask,
        inside_target_domain_mask=target_domain,
        inside_parent_domain_mask=parent_domain,
    )
    fit_parents = catalog_etas_events(
        catalog,
        fit_parents_mask,
        inside_target_domain_mask=target_domain,
        inside_parent_domain_mask=parent_domain,
    )
    score_targets = catalog_etas_events(
        catalog,
        score_targets_mask,
        inside_target_domain_mask=target_domain,
        inside_parent_domain_mask=parent_domain,
    )
    score_parents = catalog_etas_events(
        catalog,
        score_parents_mask,
        inside_target_domain_mask=target_domain,
        inside_parent_domain_mask=parent_domain,
    )
    fit_audit = _selection_audit(
        interval_start_days=fit_start_day,
        interval_end_days=definition.fit_end_day,
        parent_history_start_days=fit_parent_start,
        targets=fit_targets,
        parents=fit_parents,
    )
    score_audit = _selection_audit(
        interval_start_days=definition.assessment_start_day,
        interval_end_days=definition.assessment_end_day,
        parent_history_start_days=score_parent_start,
        targets=score_targets,
        parents=score_parents,
    )
    return (
        fit_audit,
        score_audit,
        fit_targets,
        fit_parents,
        score_targets,
        score_parents,
        fit_parents_mask,
    )


def _attempt(
    *,
    config: BackgroundConfig,
    catalog: EarthquakeCatalog,
    runtime_snapshot: LocalSupportRuntimeSnapshot,
    poisson_snapshot: LocalSupportPoissonSnapshotFit,
    uniform_score: PointProcessScoreEvidence,
    roles: ETASParentRoleMasks,
    model_variant_id: str,
    selected_bandwidth_km: float,
    authorization_id: str,
    protocol_sha256: str,
    bounds: ETASParameterBounds,
    options: OptimizerOptions,
    thresholds: StabilityThresholds,
    fitter: ETASFitCallable,
) -> ETASSnapshotAttempt:
    definition = poisson_snapshot.definition
    support = runtime_snapshot.support
    spec = build_etas_model_spec(
        config,
        selected_mc=support.common_mc,
        aki_b_value=support.retained_selected_aki_b_value,
    )
    fit_start_utc = (
        config.etas.final_fit_start_utc
        if definition.is_validation
        else config.etas.historical_fold_fit_start_utc
    )
    fit_start_day = utc_timestamp_to_day(fit_start_utc)
    history_start_day = utc_timestamp_to_day(config.etas.history_start_utc)
    (
        fit_selection,
        score_selection,
        fit_targets,
        fit_parents,
        score_targets,
        score_parents,
        fit_parent_mask,
    ) = _selection_inputs(
        catalog,
        definition,
        runtime_snapshot,
        roles,
        spec,
        fit_start_day=fit_start_day,
        history_start_day=history_start_day,
    )
    background_model = poisson_snapshot.kde_model(selected_bandwidth_km)
    background_density = _KDEBackgroundDensity(background_model)
    quadrature = point_area_quadrature_from_grid(runtime_snapshot.grid_family.at(12.5))
    fit_problem = ETASLikelihoodProblem(
        assessment_start_days=fit_start_day,
        assessment_end_days=definition.fit_end_day,
        target_events=fit_targets,
        parent_events=fit_parents,
        background_density=background_density,
        spatial_integrator=quadrature,
    )
    score_problem = ETASLikelihoodProblem(
        assessment_start_days=definition.assessment_start_day,
        assessment_end_days=definition.assessment_end_day,
        target_events=score_targets,
        parent_events=score_parents,
        background_density=background_density,
        spatial_integrator=quadrature,
    )
    fit_result = fitter(
        fit_problem,
        spec,
        root_seed=config.randomness.root_seed,
        protocol_version=config.protocol_version,
        model_id=definition.optimizer_model_id,
        bounds=bounds,
        options=options,
        thresholds=thresholds,
    )
    if not isinstance(fit_result, ETASFitResult):
        raise TypeError("fit_function must return ETASFitResult")
    parameter_id = _parameter_snapshot_id(
        protocol_sha256=protocol_sha256,
        authorization_id=authorization_id,
        model_variant_id=model_variant_id,
        definition=definition,
        runtime_snapshot=runtime_snapshot,
        fit_start_utc=fit_start_utc,
        spec=spec,
        bounds=bounds,
        options=options,
        thresholds=thresholds,
        fit_targets=fit_targets,
        fit_parents=fit_parents,
        parent_role_event_ids=_role_event_ids(catalog, roles, fit_parent_mask),
        fit_result=fit_result,
        poisson_snapshot=poisson_snapshot,
        background_model=background_model,
    )
    grid_gate: ETASGridGateEvidence | None = None
    paired: PairedInformationGainEvidence | None = None
    failure_reasons: tuple[str, ...] = ()
    if not fit_result.stability.stable:
        reasons = fit_result.stability.failure_reasons or (
            "ETAS fit failed the frozen numerical-stability gate",
        )
        failure_reasons = tuple(f"numerical stability: {reason}" for reason in reasons)
    else:
        parameters = fit_result.best_parameters
        if parameters is None:
            raise ValueError("stable local ETAS fit omitted best parameters")
        grid_gate = _grid_gate_evidence(
            protocol_sha256=protocol_sha256,
            snapshot_id=definition.snapshot_id,
            parameter_snapshot_id=parameter_id,
            problem=score_problem,
            parameters=parameters,
            spec=spec,
            grid_family=runtime_snapshot.grid_family,
        )
        if not grid_gate.passed:
            failure_reasons = grid_gate.failure_reasons or (
                "local ETAS failed the frozen three-grid convergence gate",
            )
        else:
            kde_gate_id = poisson_snapshot.gate_for(selected_bandwidth_km).numerical_evidence_id
            candidate = _candidate_score(
                protocol_sha256=protocol_sha256,
                authorization_id=authorization_id,
                model_variant_id=model_variant_id,
                definition=definition,
                runtime_snapshot=runtime_snapshot,
                parameter_snapshot_id=parameter_id,
                problem=score_problem,
                parameters=parameters,
                spec=spec,
                uniform_score=uniform_score,
                numerical_gate_evidence_ids=(
                    parameter_id,
                    kde_gate_id,
                    grid_gate.numerical_evidence_id,
                ),
            )
            paired = PairedInformationGainEvidence.build(
                candidate=candidate,
                uniform=uniform_score,
            )
    return ETASSnapshotAttempt(
        definition=definition,
        selected_mc=support.common_mc,
        model_variant_id=model_variant_id,
        fit_selection=fit_selection,
        score_selection=score_selection,
        fit_result=fit_result,
        parameter_snapshot_id=parameter_id,
        grid_gate_evidence=grid_gate,
        paired_evidence=paired,
        failure_reasons=failure_reasons,
    )


def run_local_support_etas_pipeline(
    config: BackgroundConfig,
    catalog: EarthquakeCatalog,
    runtime: LocalSupportRuntime,
    poisson_result: LocalSupportPoissonKDEPipelineResult,
    authorized_execution: AuthorizedExecution,
    *,
    fit_function: ETASFitCallable | None = None,
    progress: Callable[[str], None] | None = None,
) -> LocalSupportETASPipelineResult:
    """Fit and score primary local ETAS plus the unsupported-parent sensitivity."""

    _validate_inputs(config, catalog, runtime, poisson_result, authorized_execution)
    protocol_sha256 = _canonical_sha256(config.model_dump(mode="python"))
    authorization_id = authorized_execution.authorization_id
    uniforms = _uniform_scores(poisson_result)
    selected_bandwidth = poisson_result.selected_bandwidth_km
    primary_variant = etas_variant_id(config, selected_bandwidth_km=selected_bandwidth)
    sensitivity_variant = primary_variant + _SENSITIVITY_SUFFIX
    bounds = build_etas_parameter_bounds(config)
    options = build_optimizer_options(config)
    thresholds = build_stability_thresholds(config)
    fitter = fit_etas if fit_function is None else fit_function

    primary_attempts: list[ETASSnapshotAttempt] = []
    sensitivity_attempts: list[ETASSnapshotAttempt] = []
    not_applicable: list[tuple[str, str]] = []
    for runtime_snapshot, poisson_snapshot in zip(
        runtime.snapshots,
        poisson_result.snapshots,
        strict=True,
    ):
        snapshot_id = runtime_snapshot.snapshot_id
        if progress is not None:
            progress(f"etas_local_support:{snapshot_id}:primary:start")
        supported = runtime_snapshot.supported_mask
        unsupported = np.asarray(
            catalog.inside_study_area & ~supported,
            dtype=np.bool_,
        )
        prevalidated_unsupported = np.asarray(
            runtime_snapshot.etas_primary_parent_role_mask & unsupported,
            dtype=np.bool_,
        )
        roles = build_local_support_etas_parent_roles(
            catalog,
            supported_domain_mask=supported,
            unsupported_domain_mask=unsupported,
            common_mc=runtime_snapshot.support.common_mc,
            prevalidated_unsupported_parent_mask=prevalidated_unsupported,
        )
        primary_attempt = _attempt(
            config=config,
            catalog=catalog,
            runtime_snapshot=runtime_snapshot,
            poisson_snapshot=poisson_snapshot,
            uniform_score=uniforms[snapshot_id],
            roles=roles,
            model_variant_id=primary_variant,
            selected_bandwidth_km=selected_bandwidth,
            authorization_id=authorization_id,
            protocol_sha256=protocol_sha256,
            bounds=bounds,
            options=options,
            thresholds=thresholds,
            fitter=fitter,
        )
        primary_attempts.append(primary_attempt)
        if progress is not None:
            status = "done" if primary_attempt.succeeded else "failed"
            progress(f"etas_local_support:{snapshot_id}:primary:{status}")

        has_unsupported_cells = any(
            cell.status == "unsupported" for cell in runtime_snapshot.support.cells
        )
        if not has_unsupported_cells:
            not_applicable.append((snapshot_id, "snapshot has no unsupported fixed cell"))
            continue
        if progress is not None:
            progress(f"etas_local_support:{snapshot_id}:parent_sensitivity:start")
        sensitivity = _attempt(
            config=config,
            catalog=catalog,
            runtime_snapshot=runtime_snapshot,
            poisson_snapshot=poisson_snapshot,
            uniform_score=uniforms[snapshot_id],
            roles=roles.excluding_unsupported_parents(),
            model_variant_id=sensitivity_variant,
            selected_bandwidth_km=selected_bandwidth,
            authorization_id=authorization_id,
            protocol_sha256=protocol_sha256,
            bounds=bounds,
            options=options,
            thresholds=thresholds,
            fitter=fitter,
        )
        sensitivity_attempts.append(sensitivity)
        if progress is not None:
            status = "done" if sensitivity.succeeded else "failed"
            progress(f"etas_local_support:{snapshot_id}:parent_sensitivity:{status}")

    development = tuple(
        attempt.paired_evidence
        for attempt in primary_attempts[:4]
        if attempt.paired_evidence is not None
    )
    validation = primary_attempts[4].paired_evidence
    failures = tuple(
        (attempt.definition.snapshot_id, attempt.failure_reasons)
        for attempt in primary_attempts
        if attempt.failure_reasons
    )
    audited = AuditedBackgroundModelEvidence(
        model_id="etas",
        model_variant_id=primary_variant,
        protocol_sha256=protocol_sha256,
        development_folds=development,
        validation=validation,
        failed_snapshot_reasons=failures,
    )
    primary = ETASPipelineResult(
        protocol_sha256=protocol_sha256,
        model_variant_id=primary_variant,
        attempts=tuple(primary_attempts),
        etas_evidence=audited,
    )
    return LocalSupportETASPipelineResult(
        protocol_sha256=protocol_sha256,
        authorization_id=authorization_id,
        primary=primary,
        sensitivity_attempts=tuple(sensitivity_attempts),
        sensitivity_not_applicable=tuple(not_applicable),
    )


__all__ = [
    "LocalSupportETASPipelineResult",
    "run_local_support_etas_pipeline",
]
