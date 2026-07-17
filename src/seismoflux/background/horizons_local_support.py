"""Validation-only issue-horizon diagnostics on the final G1-LS support domain."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from seismoflux.background.adapters import (
    build_etas_model_spec,
    point_area_quadrature_from_grid,
)
from seismoflux.background.catalog import EarthquakeCatalog
from seismoflux.background.config import BackgroundConfig
from seismoflux.background.etas_fit import (
    ETASLikelihoodProblem,
    ETASParameters,
    etas_log_likelihood,
)
from seismoflux.background.evidence import (
    PairedInformationGainEvidence,
    PointProcessScoreEvidence,
)
from seismoflux.background.horizons import (
    EXPECTED_HORIZONS,
    EXPECTED_PUBLICATION_DELAYS,
    _CachedPointAreaQuadrature,
    _SpatialBackgroundDensity,
)
from seismoflux.background.issues import FrozenIssueCalendar, IssueExposure
from seismoflux.background.local_support_runtime import LocalSupportRuntime
from seismoflux.background.pipeline_local_support import LocalSupportPrimaryPipelineResult
from seismoflux.background.score_ledger import (
    ScoreLedgerEntry,
    secondary_evaluation_context_id,
)
from seismoflux.background.scoring_authorization import (
    AuthorizedExecution,
    require_background_scoring_authorized,
)
from seismoflux.background.workflow import (
    build_local_support_etas_parent_roles,
    catalog_etas_events,
)

CandidateModelId = Literal["spatial_poisson", "etas"]


def _assessment_end_utc(exposure: IssueExposure) -> str:
    start = datetime.fromisoformat(exposure.issue_time_utc.replace("Z", "+00:00"))
    return (
        (start + timedelta(days=exposure.horizon_days))
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


@dataclass(frozen=True, slots=True)
class LocalSupportHorizonExposure:
    """One exact frozen issue exposure and its candidate-minus-uniform pair."""

    issue_date_local: str
    issue_time_utc: str
    horizon_days: int
    publication_delay_days: int
    paired_evidence: PairedInformationGainEvidence

    def __post_init__(self) -> None:
        candidate = self.paired_evidence.candidate
        if candidate.snapshot_id != "final_validation":
            raise ValueError("local horizon score must use final_validation parameters")
        if candidate.assessment_start_utc != self.issue_time_utc:
            raise ValueError("local horizon score starts at another issue time")
        expected_end = (
            (
                datetime.fromisoformat(self.issue_time_utc.replace("Z", "+00:00"))
                + timedelta(days=self.horizon_days)
            )
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )
        if candidate.assessment_end_utc != expected_end:
            raise ValueError("local horizon score has the wrong exact exposure end")
        context_id = secondary_evaluation_context_id(
            protocol_sha256=candidate.protocol_sha256,
            authorization_id=candidate.authorization_id or "",
            partition_id="validation",
            issue_date_local=self.issue_date_local,
            issue_time_utc=self.issue_time_utc,
            horizon_days=self.horizon_days,
            publication_delay_days=self.publication_delay_days,
        )
        if candidate.evaluation_context_id != context_id:
            raise ValueError("local horizon score uses another evaluation context")


@dataclass(frozen=True, slots=True)
class LocalSupportPairedHorizonBacktest:
    candidate_model_id: CandidateModelId
    publication_delay_days: int
    horizon_days: int
    exposures: tuple[LocalSupportHorizonExposure, ...]

    def __post_init__(self) -> None:
        if self.candidate_model_id not in {"spatial_poisson", "etas"}:
            raise ValueError("local horizon comparison has an unknown candidate")
        if self.publication_delay_days not in EXPECTED_PUBLICATION_DELAYS:
            raise ValueError("local horizon comparison has an unknown publication delay")
        if self.horizon_days not in EXPECTED_HORIZONS:
            raise ValueError("local horizon comparison has an unknown horizon")
        if not self.exposures:
            raise ValueError("local horizon comparison must retain frozen exposures")
        previous = ""
        seen: set[str] = set()
        for exposure in self.exposures:
            candidate = exposure.paired_evidence.candidate
            if candidate.model_id != self.candidate_model_id:
                raise ValueError("local horizon exposure uses another candidate model")
            if (
                exposure.publication_delay_days != self.publication_delay_days
                or exposure.horizon_days != self.horizon_days
                or exposure.issue_date_local <= previous
            ):
                raise ValueError("local horizon exposures are not in frozen order")
            previous = exposure.issue_date_local
            overlap = seen.intersection(candidate.target_event_ids)
            if overlap:
                raise ValueError("one physical target appears in overlapping exposures")
            seen.update(candidate.target_event_ids)

    @property
    def information_gain_per_event(self) -> float | None:
        target_count = sum(
            len(exposure.paired_evidence.candidate.target_event_ids) for exposure in self.exposures
        )
        if target_count == 0:
            return None
        numerator = math.fsum(
            (
                float(
                    np.sum(
                        exposure.paired_evidence.event_log_intensity_differences,
                        dtype=np.float64,
                    )
                )
                - exposure.paired_evidence.compensator_difference
            )
            for exposure in self.exposures
        )
        return numerator / target_count


@dataclass(frozen=True, slots=True)
class LocalSupportHorizonFailure:
    candidate_model_id: CandidateModelId
    publication_delay_days: int
    horizon_days: int
    reason: str

    def __post_init__(self) -> None:
        if self.candidate_model_id not in {"spatial_poisson", "etas"}:
            raise ValueError("local horizon failure has an unknown model")
        if self.publication_delay_days not in EXPECTED_PUBLICATION_DELAYS:
            raise ValueError("local horizon failure has an unknown delay")
        if self.horizon_days not in EXPECTED_HORIZONS:
            raise ValueError("local horizon failure has an unknown horizon")
        if not self.reason:
            raise ValueError("local horizon failure reason must not be empty")


@dataclass(frozen=True, slots=True)
class LocalSupportHorizonBacktests:
    comparisons: tuple[LocalSupportPairedHorizonBacktest, ...]
    failed_comparisons: tuple[LocalSupportHorizonFailure, ...]
    score_entries: tuple[ScoreLedgerEntry, ...]
    generated_scores: tuple[PointProcessScoreEvidence, ...]

    def __post_init__(self) -> None:
        expected = tuple(
            (model, delay, horizon)
            for delay in EXPECTED_PUBLICATION_DELAYS
            for model in ("spatial_poisson", "etas")
            for horizon in EXPECTED_HORIZONS
        )
        observed = tuple(
            (
                item.candidate_model_id,
                item.publication_delay_days,
                item.horizon_days,
            )
            for item in self.comparisons
        ) + tuple(
            (
                item.candidate_model_id,
                item.publication_delay_days,
                item.horizon_days,
            )
            for item in self.failed_comparisons
        )
        if tuple(sorted(observed)) != tuple(sorted(expected)):
            raise ValueError("local horizon outcomes do not cover the frozen comparison grid")
        score_ids = tuple(score.score_id for score in self.generated_scores)
        if len(set(score_ids)) != len(score_ids):
            raise ValueError("local horizon generated scores must be unique")
        ledger_ids = tuple(
            entry.score_id for entry in self.score_entries if entry.score_id is not None
        )
        if set(ledger_ids) != set(score_ids):
            raise ValueError("local horizon ledger entries omit generated scores")


def _target_indices(
    catalog: EarthquakeCatalog,
    exposure: IssueExposure,
    *,
    supported_mask: NDArray[np.bool_],
    selected_mc: float,
) -> NDArray[np.int64]:
    mask = np.asarray(
        supported_mask
        & (catalog.magnitude >= selected_mc)
        & (catalog.origin_day > exposure.issue_day)
        & (catalog.origin_day <= exposure.end_day),
        dtype=np.bool_,
    )
    indices = [int(value) for value in np.flatnonzero(mask)]
    indices.sort(key=lambda index: (float(catalog.origin_day[index]), str(catalog.event_id[index])))
    return np.asarray(indices, dtype=np.int64)


def _score(
    *,
    source: PointProcessScoreEvidence,
    exposure: IssueExposure,
    publication_delay_days: int,
    event_ids: tuple[str, ...],
    event_logs: NDArray[np.float64],
    compensator: float,
) -> PointProcessScoreEvidence:
    if source.authorization_id is None:
        raise ValueError("local horizon source score omitted its authorization")
    context = secondary_evaluation_context_id(
        protocol_sha256=source.protocol_sha256,
        authorization_id=source.authorization_id,
        partition_id="validation",
        issue_date_local=exposure.issue_date_local,
        issue_time_utc=exposure.issue_time_utc,
        horizon_days=exposure.horizon_days,
        publication_delay_days=publication_delay_days,
    )
    return PointProcessScoreEvidence(
        protocol_sha256=source.protocol_sha256,
        model_id=source.model_id,
        model_variant_id=source.model_variant_id,
        parameter_snapshot_id=source.parameter_snapshot_id,
        snapshot_id="final_validation",
        fit_end_utc=source.fit_end_utc,
        assessment_start_utc=exposure.issue_time_utc,
        assessment_end_utc=_assessment_end_utc(exposure),
        selected_mc=source.selected_mc,
        target_event_ids=event_ids,
        event_log_intensities=event_logs,
        compensator=compensator,
        numerical_gate_evidence_ids=source.numerical_gate_evidence_ids,
        support_id=source.support_id,
        supported_area_km2=source.supported_area_km2,
        compensator_domain_id=source.compensator_domain_id,
        authorization_id=source.authorization_id,
        evaluation_context_id=context,
    )


def _entry(
    score: PointProcessScoreEvidence,
    exposure: IssueExposure,
    *,
    publication_delay_days: int,
) -> ScoreLedgerEntry:
    if (
        score.authorization_id is None
        or score.support_id is None
        or score.supported_area_km2 is None
        or score.compensator_domain_id is None
    ):
        raise ValueError("local horizon score omitted its support binding")
    return ScoreLedgerEntry(
        scope="secondary_validation_horizon",
        status="succeeded",
        protocol_sha256=score.protocol_sha256,
        authorization_id=score.authorization_id,
        support_id=score.support_id,
        supported_area_km2=score.supported_area_km2,
        compensator_domain_id=score.compensator_domain_id,
        model_id=score.model_id,
        model_variant_id=score.model_variant_id,
        parameter_snapshot_id=score.parameter_snapshot_id,
        snapshot_id="final_validation",
        fit_end_utc=score.fit_end_utc,
        assessment_start_utc=score.assessment_start_utc,
        assessment_end_utc=score.assessment_end_utc,
        selected_mc=score.selected_mc,
        score=score,
        partition_id="validation",
        issue_date_local=exposure.issue_date_local,
        horizon_days=exposure.horizon_days,
        publication_delay_days=publication_delay_days,
    )


def _final_primary_scores(
    primary: LocalSupportPrimaryPipelineResult,
) -> tuple[PointProcessScoreEvidence, PointProcessScoreEvidence, PointProcessScoreEvidence | None]:
    uniform_validation = primary.poisson.uniform_evidence.validation
    spatial_validation = primary.poisson.spatial_evidence.validation
    if uniform_validation is None or spatial_validation is None:
        raise ValueError("local horizon diagnostics require final Poisson scores")
    etas_validation = primary.etas.evidence.validation
    return (
        uniform_validation.uniform,
        spatial_validation.candidate,
        etas_validation.candidate if etas_validation is not None else None,
    )


def run_local_support_issue_horizon_backtests(
    config: BackgroundConfig,
    catalog: EarthquakeCatalog,
    calendar: FrozenIssueCalendar,
    runtime: LocalSupportRuntime,
    primary: LocalSupportPrimaryPipelineResult,
    authorized_execution: AuthorizedExecution,
) -> LocalSupportHorizonBacktests:
    """Run all frozen validation issue diagnostics without using them for G1-LS."""

    require_background_scoring_authorized(config, authorized_execution)
    if primary.authorization_id != authorized_execution.authorization_id:
        raise ValueError("local horizon input uses another scoring authorization")
    if tuple(str(value) for value in catalog.event_id) != runtime.event_ids:
        raise ValueError("local horizon catalog differs from the support runtime")
    if tuple(config.time.horizons_days) != EXPECTED_HORIZONS:
        raise ValueError("configured horizons differ from the frozen diagnostics")
    if tuple(config.time.publication_delay_sensitivity_days) != EXPECTED_PUBLICATION_DELAYS:
        raise ValueError("configured publication delays differ from the frozen diagnostics")
    if calendar.validation.partition_id != "validation":
        raise ValueError("local horizon diagnostics may use only validation exposures")

    final_runtime = runtime.snapshot("final_validation")
    final_poisson = primary.poisson.snapshot("final_validation")
    selected_mc = final_runtime.support.common_mc
    if final_poisson.selected_mc != selected_mc:
        raise ValueError("local horizon Poisson model uses another common Mc")
    uniform_model = final_poisson.uniform_model
    spatial_model = primary.poisson.selected_kde_model("final_validation")
    if uniform_model.rate_per_day != spatial_model.rate_per_day:
        raise ValueError("local horizon Poisson rates are not paired")
    uniform_source, spatial_source, etas_source = _final_primary_scores(primary)

    etas_attempt = primary.etas.primary.attempt("final_validation")
    etas_parameters: ETASParameters | None = None
    if etas_attempt.fit_result is not None and etas_attempt.fit_result.stability.stable:
        etas_parameters = etas_attempt.fit_result.best_parameters
    etas_reason = (
        "; ".join(etas_attempt.failure_reasons)
        if etas_attempt.failure_reasons
        else "final ETAS score is unavailable"
    )
    etas_spec = build_etas_model_spec(
        config,
        selected_mc=selected_mc,
        aki_b_value=final_runtime.support.retained_selected_aki_b_value,
    )
    if etas_parameters is not None:
        etas_spec.validate_parameters(etas_parameters)

    supported = final_runtime.supported_mask
    unsupported = np.asarray(catalog.inside_study_area & ~supported, dtype=np.bool_)
    eligible_unsupported = np.asarray(
        final_runtime.etas_primary_parent_role_mask & unsupported,
        dtype=np.bool_,
    )
    roles = build_local_support_etas_parent_roles(
        catalog,
        supported_domain_mask=supported,
        unsupported_domain_mask=unsupported,
        common_mc=selected_mc,
        prevalidated_unsupported_parent_mask=eligible_unsupported,
    )
    exposures = tuple(
        exposure
        for horizon in EXPECTED_HORIZONS
        for exposure in calendar.validation.exposures(horizon)
    )
    minimum_issue_day = min(exposure.issue_day for exposure in exposures)
    maximum_end_day = max(exposure.end_day for exposure in exposures)
    cache_mask = np.asarray(
        roles.parent_mask
        & (catalog.origin_day > minimum_issue_day - etas_spec.history_parent_cutoff_days)
        & (catalog.origin_day <= maximum_end_day),
        dtype=np.bool_,
    )
    cache_indices = np.flatnonzero(cache_mask)
    base_quadrature = point_area_quadrature_from_grid(final_runtime.grid_family.at(12.5))
    quadrature = _CachedPointAreaQuadrature(
        base_quadrature,
        parent_x_km=catalog.x_km[cache_indices],
        parent_y_km=catalog.y_km[cache_indices],
        parent_magnitudes=catalog.magnitude[cache_indices],
        spec=etas_spec,
        background_model=spatial_model,
    )

    comparisons: list[LocalSupportPairedHorizonBacktest] = []
    failures: list[LocalSupportHorizonFailure] = []
    entries: list[ScoreLedgerEntry] = []
    generated: dict[str, PointProcessScoreEvidence] = {}
    for delay in EXPECTED_PUBLICATION_DELAYS:
        for horizon in EXPECTED_HORIZONS:
            spatial_exposures: list[LocalSupportHorizonExposure] = []
            etas_exposures: list[LocalSupportHorizonExposure] = []
            for exposure in calendar.validation.exposures(horizon):
                indices = _target_indices(
                    catalog,
                    exposure,
                    supported_mask=supported,
                    selected_mc=selected_mc,
                )
                identifiers = tuple(str(catalog.event_id[index]) for index in indices)
                uniform_log = math.log(uniform_model.rate_per_day) - math.log(
                    uniform_model.study_area_km2
                )
                uniform_score = _score(
                    source=uniform_source,
                    exposure=exposure,
                    publication_delay_days=delay,
                    event_ids=identifiers,
                    event_logs=np.full(len(indices), uniform_log, dtype=np.float64),
                    compensator=uniform_model.rate_per_day * horizon,
                )
                spatial_score = _score(
                    source=spatial_source,
                    exposure=exposure,
                    publication_delay_days=delay,
                    event_ids=identifiers,
                    event_logs=(
                        spatial_model.log_density(catalog.x_km[indices], catalog.y_km[indices])
                        + math.log(spatial_model.rate_per_day)
                    ),
                    compensator=spatial_model.rate_per_day * horizon,
                )
                spatial_pair = PairedInformationGainEvidence.build(
                    candidate=spatial_score,
                    uniform=uniform_score,
                )
                spatial_exposures.append(
                    LocalSupportHorizonExposure(
                        issue_date_local=exposure.issue_date_local,
                        issue_time_utc=exposure.issue_time_utc,
                        horizon_days=horizon,
                        publication_delay_days=delay,
                        paired_evidence=spatial_pair,
                    )
                )
                for score in (uniform_score, spatial_score):
                    if score.score_id not in generated:
                        generated[score.score_id] = score
                        entries.append(_entry(score, exposure, publication_delay_days=delay))

                if etas_parameters is None or etas_source is None:
                    support = final_runtime.support
                    context = secondary_evaluation_context_id(
                        protocol_sha256=primary.protocol_sha256,
                        authorization_id=primary.authorization_id,
                        partition_id="validation",
                        issue_date_local=exposure.issue_date_local,
                        issue_time_utc=exposure.issue_time_utc,
                        horizon_days=horizon,
                        publication_delay_days=delay,
                    )
                    entries.append(
                        ScoreLedgerEntry(
                            scope="secondary_validation_horizon",
                            status="not_run",
                            protocol_sha256=primary.protocol_sha256,
                            authorization_id=primary.authorization_id,
                            support_id=support.support_id,
                            supported_area_km2=support.retained_area_m2 / 1_000_000.0,
                            compensator_domain_id=final_runtime.compensator_domain_id,
                            model_id="etas",
                            model_variant_id=primary.etas.model_variant_id,
                            parameter_snapshot_id=(f"not-run/etas/final_validation/{context[:16]}"),
                            snapshot_id="final_validation",
                            fit_end_utc=uniform_source.fit_end_utc,
                            assessment_start_utc=exposure.issue_time_utc,
                            assessment_end_utc=_assessment_end_utc(exposure),
                            selected_mc=selected_mc,
                            score=None,
                            failure_reasons=(etas_reason,),
                            partition_id="validation",
                            issue_date_local=exposure.issue_date_local,
                            horizon_days=horizon,
                            publication_delay_days=delay,
                        )
                    )
                    continue

                parent_mask = np.asarray(
                    roles.parent_mask
                    & (
                        catalog.origin_day
                        > exposure.issue_day - etas_spec.history_parent_cutoff_days
                    )
                    & (catalog.origin_day <= exposure.end_day),
                    dtype=np.bool_,
                )
                parents = catalog_etas_events(
                    catalog,
                    parent_mask,
                    time_origin_day=exposure.issue_day,
                    publication_delay_days=float(delay),
                    inside_target_domain_mask=supported,
                    inside_parent_domain_mask=roles.parent_mask,
                )
                target_id_set = set(identifiers)
                targets = tuple(event for event in parents if event.event_id in target_id_set)
                likelihood = etas_log_likelihood(
                    ETASLikelihoodProblem(
                        assessment_start_days=0.0,
                        assessment_end_days=float(horizon),
                        target_events=targets,
                        parent_events=parents,
                        background_density=_SpatialBackgroundDensity(spatial_model),
                        spatial_integrator=quadrature,
                    ),
                    etas_parameters,
                    etas_spec,
                )
                logs_by_id = {
                    event_id: math.log(value)
                    for event_id, value in zip(
                        likelihood.target_event_ids,
                        likelihood.event_intensities,
                        strict=True,
                    )
                }
                etas_score = _score(
                    source=etas_source,
                    exposure=exposure,
                    publication_delay_days=delay,
                    event_ids=identifiers,
                    event_logs=np.asarray([logs_by_id[event_id] for event_id in identifiers]),
                    compensator=likelihood.total_compensator,
                )
                etas_pair = PairedInformationGainEvidence.build(
                    candidate=etas_score,
                    uniform=uniform_score,
                )
                etas_exposures.append(
                    LocalSupportHorizonExposure(
                        issue_date_local=exposure.issue_date_local,
                        issue_time_utc=exposure.issue_time_utc,
                        horizon_days=horizon,
                        publication_delay_days=delay,
                        paired_evidence=etas_pair,
                    )
                )
                generated[etas_score.score_id] = etas_score
                entries.append(_entry(etas_score, exposure, publication_delay_days=delay))

            comparisons.append(
                LocalSupportPairedHorizonBacktest(
                    candidate_model_id="spatial_poisson",
                    publication_delay_days=delay,
                    horizon_days=horizon,
                    exposures=tuple(spatial_exposures),
                )
            )
            if etas_parameters is None or etas_source is None:
                failures.append(
                    LocalSupportHorizonFailure(
                        candidate_model_id="etas",
                        publication_delay_days=delay,
                        horizon_days=horizon,
                        reason=etas_reason,
                    )
                )
            else:
                comparisons.append(
                    LocalSupportPairedHorizonBacktest(
                        candidate_model_id="etas",
                        publication_delay_days=delay,
                        horizon_days=horizon,
                        exposures=tuple(etas_exposures),
                    )
                )
    return LocalSupportHorizonBacktests(
        comparisons=tuple(comparisons),
        failed_comparisons=tuple(failures),
        score_entries=tuple(entries),
        generated_scores=tuple(generated.values()),
    )


__all__ = [
    "LocalSupportHorizonBacktests",
    "LocalSupportHorizonExposure",
    "LocalSupportHorizonFailure",
    "LocalSupportPairedHorizonBacktest",
    "run_local_support_issue_horizon_backtests",
]
