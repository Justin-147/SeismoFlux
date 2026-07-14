"""Primary G1-LS orchestration and complete pre-publication score accounting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from seismoflux.background.catalog import EarthquakeCatalog
from seismoflux.background.config import BackgroundConfig
from seismoflux.background.evaluation import (
    LOCAL_SUPPORT_BOOTSTRAP_ISSUE_ID,
    BootstrapInterval,
    InformationGainContributions,
    bootstrap_information_gain,
)
from seismoflux.background.evidence import (
    AuditedG1Assessment,
    AuditedModelSelection,
    PairedInformationGainEvidence,
    PointProcessScoreEvidence,
    assess_audited_g1,
    select_audited_background_model,
)
from seismoflux.background.local_support_runtime import LocalSupportRuntime
from seismoflux.background.pipeline_local_support_etas import (
    LocalSupportETASPipelineResult,
    run_local_support_etas_pipeline,
)
from seismoflux.background.pipeline_poisson import (
    LocalSupportPoissonKDEPipelineResult,
    PoissonKDEScientificInability,
    run_local_support_poisson_kde_pipeline,
)
from seismoflux.background.poisson import FROZEN_BANDWIDTHS_KM
from seismoflux.background.score_ledger import ScoreLedgerEntry
from seismoflux.background.scoring_authorization import AuthorizedExecution
from seismoflux.background.workflow import build_snapshot_definitions


@dataclass(frozen=True, slots=True)
class LocalSupportBootstrapOutcome:
    """Validation uncertainty for one successful nonuniform primary score."""

    model_id: str
    score_id: str | None
    interval: BootstrapInterval | None
    failure_reason: str | None

    def __post_init__(self) -> None:
        if self.model_id not in {"spatial_poisson", "etas"}:
            raise ValueError("local-support bootstrap uses an unknown model")
        if (self.interval is None) == (self.failure_reason is None):
            raise ValueError("bootstrap outcome must be successful or explicitly unavailable")
        if self.interval is not None and self.score_id is None:
            raise ValueError("successful bootstrap must identify its source score")


@dataclass(frozen=True, slots=True)
class LocalSupportPrimaryPipelineResult:
    """Uniform/KDE/ETAS G1-LS result before secondary issue diagnostics."""

    protocol_sha256: str
    authorization_id: str
    poisson: LocalSupportPoissonKDEPipelineResult
    etas: LocalSupportETASPipelineResult
    g1: AuditedG1Assessment
    selection: AuditedModelSelection
    validation_bootstrap: tuple[LocalSupportBootstrapOutcome, ...]
    score_entries: tuple[ScoreLedgerEntry, ...]
    generated_scores: tuple[PointProcessScoreEvidence, ...]

    def __post_init__(self) -> None:
        if self.poisson.protocol_sha256 != self.protocol_sha256:
            raise ValueError("local primary Poisson result uses another protocol")
        if self.etas.protocol_sha256 != self.protocol_sha256:
            raise ValueError("local primary ETAS result uses another protocol")
        if self.etas.authorization_id != self.authorization_id:
            raise ValueError("local primary ETAS result uses another authorization")
        score_ids = tuple(score.score_id for score in self.generated_scores)
        if len(set(score_ids)) != len(score_ids):
            raise ValueError("local primary generated scores must be unique")
        ledger_score_ids = tuple(
            entry.score_id for entry in self.score_entries if entry.score_id is not None
        )
        if set(ledger_score_ids) != set(score_ids):
            raise ValueError("local primary entries do not account for every generated score")

    @property
    def stage3_allowed(self) -> bool:
        return self.g1.passed


@dataclass(frozen=True, slots=True)
class LocalSupportPoissonFailureAccounting:
    """Every Score created before a preregistered Poisson/KDE stop."""

    score_entries: tuple[ScoreLedgerEntry, ...]
    generated_scores: tuple[PointProcessScoreEvidence, ...]

    def __post_init__(self) -> None:
        score_ids = tuple(score.score_id for score in self.generated_scores)
        if len(set(score_ids)) != len(score_ids):
            raise ValueError("partial Poisson generated Score IDs must be unique")
        ledger_ids = tuple(
            entry.score_id for entry in self.score_entries if entry.score_id is not None
        )
        if len(set(ledger_ids)) != len(ledger_ids) or set(ledger_ids) != set(score_ids):
            raise ValueError("partial Poisson ledger does not exactly account for its Scores")


def _score_entry(
    score: PointProcessScoreEvidence,
    *,
    scope: str,
    kde_bandwidth_km: int | None = None,
    etas_parent_variant: str | None = None,
) -> ScoreLedgerEntry:
    if (
        score.authorization_id is None
        or score.support_id is None
        or score.supported_area_km2 is None
        or score.compensator_domain_id is None
    ):
        raise ValueError("G1-LS scores must carry a complete support/authorization identity")
    return ScoreLedgerEntry(
        scope=scope,  # type: ignore[arg-type]
        status="succeeded",
        protocol_sha256=score.protocol_sha256,
        authorization_id=score.authorization_id,
        support_id=score.support_id,
        supported_area_km2=score.supported_area_km2,
        compensator_domain_id=score.compensator_domain_id,
        model_id=score.model_id,
        model_variant_id=score.model_variant_id,
        parameter_snapshot_id=score.parameter_snapshot_id,
        snapshot_id=score.snapshot_id,
        fit_end_utc=score.fit_end_utc,
        assessment_start_utc=score.assessment_start_utc,
        assessment_end_utc=score.assessment_end_utc,
        selected_mc=score.selected_mc,
        score=score,
        kde_bandwidth_km=kde_bandwidth_km,
        etas_parent_variant=etas_parent_variant,  # type: ignore[arg-type]
    )


def build_local_support_poisson_failure_accounting(
    config: BackgroundConfig,
    runtime: LocalSupportRuntime,
    authorized_execution: AuthorizedExecution,
    error: PoissonKDEScientificInability,
) -> LocalSupportPoissonFailureAccounting:
    """Retain partial Scores and gated candidate attempts without opening new targets."""

    if not isinstance(error, PoissonKDEScientificInability):
        raise TypeError("Poisson failure accounting requires a scientific inability")
    protocol_sha256 = authorized_execution.scoring_authorization.protocol_sha256
    authorization_id = authorized_execution.authorization_id
    scoreability = error.scoreability_gate_evidence
    if scoreability is None:
        raise ValueError("local Poisson scientific inability omitted scoreability evidence")
    if any(
        item.protocol_sha256 != protocol_sha256 or item.authorization_id != authorization_id
        for item in scoreability.snapshots
    ):
        raise ValueError("Poisson failure evidence uses another protocol/authorization")
    definitions = build_snapshot_definitions(config)
    if tuple(item.snapshot_id for item in runtime.snapshots) != tuple(
        definition.snapshot_id for definition in definitions
    ):
        raise ValueError("Poisson failure runtime differs from the frozen snapshots")

    partial = error.partial_failure_evidence
    generated: tuple[PointProcessScoreEvidence, ...] = (
        () if partial is None else partial.generated_scores
    )
    entries: list[ScoreLedgerEntry] = []
    successful_uniform = {
        score.snapshot_id: score for score in generated if score.model_id == "uniform_poisson"
    }
    successful_kde = {
        (
            score.snapshot_id,
            int(score.model_variant_id.rsplit("bw", 1)[1].removesuffix("km")),
        ): score
        for score in generated
        if score.model_id == "spatial_poisson"
    }
    for score in successful_uniform.values():
        entries.append(_score_entry(score, scope="primary_snapshot"))

    gate_items = {
        int(item.bandwidth_km): item
        for item in (() if error.gate_evidence is None else error.gate_evidence.candidates)
    }
    if error.fitted_snapshots is not None:
        fitted_ids = tuple(item.definition.snapshot_id for item in error.fitted_snapshots)
        if fitted_ids != tuple(definition.snapshot_id for definition in definitions):
            raise ValueError("Poisson failure fitted snapshots differ from the frozen order")
        for definition, runtime_snapshot in zip(
            definitions[:4], runtime.snapshots[:4], strict=True
        ):
            for bandwidth in FROZEN_BANDWIDTHS_KM:
                key = (definition.snapshot_id, int(bandwidth))
                candidate_score = successful_kde.get(key)
                if candidate_score is not None:
                    entries.append(
                        _score_entry(
                            candidate_score,
                            scope="kde_development_candidate",
                            kde_bandwidth_km=int(bandwidth),
                        )
                    )
                    continue
                gate = gate_items.get(int(bandwidth))
                if gate is not None and not gate.passed:
                    reason = gate.failure_reason or str(error)
                else:
                    reason = str(error)
                support = runtime_snapshot.support
                entries.append(
                    ScoreLedgerEntry(
                        scope="kde_development_candidate",
                        status="not_run",
                        protocol_sha256=protocol_sha256,
                        authorization_id=authorization_id,
                        support_id=support.support_id,
                        supported_area_km2=support.retained_area_m2 / 1_000_000.0,
                        compensator_domain_id=runtime_snapshot.compensator_domain_id,
                        model_id="spatial_poisson",
                        model_variant_id=(f"spatial_poisson/gaussian_kde_bw{bandwidth:g}km"),
                        parameter_snapshot_id=(
                            f"not-run/kde/{definition.snapshot_id}/bw{bandwidth:g}"
                        ),
                        snapshot_id=definition.snapshot_id,
                        fit_end_utc=definition.fit_end_utc,
                        assessment_start_utc=definition.assessment_start_utc,
                        assessment_end_utc=definition.assessment_end_utc,
                        selected_mc=support.common_mc,
                        score=None,
                        failure_reasons=(reason,),
                        kde_bandwidth_km=int(bandwidth),
                    )
                )
    return LocalSupportPoissonFailureAccounting(
        score_entries=tuple(entries),
        generated_scores=generated,
    )


def _failed_etas_entry(
    *,
    attempt: object,
    runtime: LocalSupportRuntime,
    authorization_id: str,
    protocol_sha256: str,
    sensitivity: bool,
) -> ScoreLedgerEntry:
    from seismoflux.background.pipeline_etas import ETASSnapshotAttempt

    item = cast(ETASSnapshotAttempt, attempt)
    definition = item.definition
    support = runtime.snapshot(definition.snapshot_id)
    if not item.failure_reasons:
        raise ValueError("failed ETAS ledger entry requires a failure reason")
    attempt_role = "sensitivity" if sensitivity else "primary"
    return ScoreLedgerEntry(
        scope=("etas_unsupported_parent_sensitivity" if sensitivity else "primary_snapshot"),
        status="failed",
        protocol_sha256=protocol_sha256,
        authorization_id=authorization_id,
        support_id=support.support.support_id,
        supported_area_km2=support.support.retained_area_m2 / 1_000_000.0,
        compensator_domain_id=support.compensator_domain_id,
        model_id="etas",
        model_variant_id=item.model_variant_id,
        parameter_snapshot_id=(
            item.parameter_snapshot_id or f"not-run/etas/{definition.snapshot_id}/{attempt_role}"
        ),
        snapshot_id=definition.snapshot_id,
        fit_end_utc=definition.fit_end_utc,
        assessment_start_utc=definition.assessment_start_utc,
        assessment_end_utc=definition.assessment_end_utc,
        selected_mc=item.selected_mc,
        score=None,
        failure_reasons=tuple(sorted(set(item.failure_reasons))),
        etas_parent_variant=(
            "exclude_all_unsupported"
            if sensitivity
            else (
                "primary_include_eligible_unsupported"
                if definition.snapshot_id in {"fold_1", "fold_3"}
                else None
            )
        ),
    )


def _uniform_scores(
    result: LocalSupportPoissonKDEPipelineResult,
) -> tuple[PointProcessScoreEvidence, ...]:
    validation = result.uniform_evidence.validation
    if validation is None:
        raise ValueError("local Poisson result omitted uniform validation evidence")
    return tuple(pair.uniform for pair in (*result.uniform_evidence.development_folds, validation))


def _primary_score_entries(
    runtime: LocalSupportRuntime,
    poisson: LocalSupportPoissonKDEPipelineResult,
    etas: LocalSupportETASPipelineResult,
    *,
    authorization_id: str,
    protocol_sha256: str,
) -> tuple[tuple[ScoreLedgerEntry, ...], tuple[PointProcessScoreEvidence, ...]]:
    entries: list[ScoreLedgerEntry] = []
    generated: dict[str, PointProcessScoreEvidence] = {}

    for score in _uniform_scores(poisson):
        entry = _score_entry(score, scope="primary_snapshot")
        entries.append(entry)
        generated[score.score_id] = score

    audited_bandwidths: set[float] = set()
    for audit in poisson.bandwidth_fold_audits:
        audited_bandwidths.add(audit.bandwidth_km)
        for pair in audit.development_folds:
            score = pair.candidate
            entry = _score_entry(
                score,
                scope="kde_development_candidate",
                kde_bandwidth_km=int(audit.bandwidth_km),
            )
            entries.append(entry)
            generated[score.score_id] = score

    failed_bandwidths = tuple(
        item
        for item in poisson.pre_score_gate_evidence.candidates
        if item.bandwidth_km not in audited_bandwidths
    )
    for item in failed_bandwidths:
        reason = item.failure_reason or "KDE candidate failed its frozen pre-score gate"
        for snapshot in poisson.snapshots[:4]:
            entries.append(
                ScoreLedgerEntry(
                    scope="kde_development_candidate",
                    status="not_run",
                    protocol_sha256=protocol_sha256,
                    authorization_id=authorization_id,
                    support_id=snapshot.support_id,
                    supported_area_km2=snapshot.supported_area_km2,
                    compensator_domain_id=snapshot.compensator_domain_id,
                    model_id="spatial_poisson",
                    model_variant_id=(f"spatial_poisson/gaussian_kde_bw{item.bandwidth_km:g}km"),
                    parameter_snapshot_id=(
                        f"not-run/kde/{snapshot.definition.snapshot_id}/bw{item.bandwidth_km:g}"
                    ),
                    snapshot_id=snapshot.definition.snapshot_id,
                    fit_end_utc=snapshot.definition.fit_end_utc,
                    assessment_start_utc=snapshot.definition.assessment_start_utc,
                    assessment_end_utc=snapshot.definition.assessment_end_utc,
                    selected_mc=snapshot.selected_mc,
                    score=None,
                    failure_reasons=(reason,),
                    kde_bandwidth_km=int(item.bandwidth_km),
                )
            )

    spatial_validation = poisson.spatial_evidence.validation
    if spatial_validation is None:
        raise ValueError("successful local Poisson pipeline omitted spatial validation")
    spatial_validation_score = spatial_validation.candidate
    entries.append(_score_entry(spatial_validation_score, scope="primary_snapshot"))
    generated[spatial_validation_score.score_id] = spatial_validation_score

    for attempt in etas.primary.attempts:
        parent_variant = (
            "primary_include_eligible_unsupported"
            if attempt.definition.snapshot_id in {"fold_1", "fold_3"}
            else None
        )
        if attempt.paired_evidence is not None:
            score = attempt.paired_evidence.candidate
            entries.append(
                _score_entry(
                    score,
                    scope="primary_snapshot",
                    etas_parent_variant=parent_variant,
                )
            )
            generated[score.score_id] = score
        else:
            entries.append(
                _failed_etas_entry(
                    attempt=attempt,
                    runtime=runtime,
                    authorization_id=authorization_id,
                    protocol_sha256=protocol_sha256,
                    sensitivity=False,
                )
            )

    for attempt in etas.sensitivity_attempts:
        if attempt.paired_evidence is not None:
            score = attempt.paired_evidence.candidate
            entries.append(
                _score_entry(
                    score,
                    scope="etas_unsupported_parent_sensitivity",
                    etas_parent_variant="exclude_all_unsupported",
                )
            )
            generated[score.score_id] = score
        else:
            entries.append(
                _failed_etas_entry(
                    attempt=attempt,
                    runtime=runtime,
                    authorization_id=authorization_id,
                    protocol_sha256=protocol_sha256,
                    sensitivity=True,
                )
            )

    return tuple(entries), tuple(generated.values())


def _bootstrap_pair(
    pair: PairedInformationGainEvidence | None,
    *,
    model_id: str,
) -> LocalSupportBootstrapOutcome:
    if pair is None or pair.information_gain_per_event is None:
        return LocalSupportBootstrapOutcome(
            model_id=model_id,
            score_id=None,
            interval=None,
            failure_reason="validation score is unavailable",
        )
    contributions = InformationGainContributions(
        physical_event_ids=pair.candidate.target_event_ids,
        event_log_intensity_differences=pair.event_log_intensity_differences,
        compensator_difference=pair.compensator_difference,
    )
    model_seed_id = (
        "spatial_poisson_vs_uniform_poisson"
        if model_id == "spatial_poisson"
        else "etas_vs_uniform_poisson"
    )
    interval = bootstrap_information_gain(
        contributions,
        model_seed_id=model_seed_id,  # type: ignore[arg-type]
        protocol_version="0.2.1",
        issue_id=LOCAL_SUPPORT_BOOTSTRAP_ISSUE_ID,
    )
    return LocalSupportBootstrapOutcome(
        model_id=model_id,
        score_id=pair.candidate.score_id,
        interval=interval,
        failure_reason=None,
    )


def run_local_support_primary_pipeline(
    config: BackgroundConfig,
    catalog: EarthquakeCatalog,
    runtime: LocalSupportRuntime,
    authorized_execution: AuthorizedExecution,
    *,
    poisson_chunk_size: int = 256,
    progress: object | None = None,
) -> LocalSupportPrimaryPipelineResult:
    """Run the five-snapshot primary G1-LS comparison without locked-test access."""

    from seismoflux.background.workflow import ProgressCallback

    poisson = run_local_support_poisson_kde_pipeline(
        config,
        catalog,
        runtime,
        authorized_execution,
        chunk_size=poisson_chunk_size,
        progress=cast(ProgressCallback | None, progress),
    )
    etas = run_local_support_etas_pipeline(
        config,
        catalog,
        runtime,
        poisson,
        authorized_execution,
        progress=cast(ProgressCallback | None, progress),
    )
    evidence = (poisson.uniform_evidence, poisson.spatial_evidence, etas.evidence)
    g1 = assess_audited_g1(evidence)
    selection = select_audited_background_model(evidence)
    entries, generated = _primary_score_entries(
        runtime,
        poisson,
        etas,
        authorization_id=authorized_execution.authorization_id,
        protocol_sha256=poisson.protocol_sha256,
    )
    bootstrap = (
        _bootstrap_pair(poisson.spatial_evidence.validation, model_id="spatial_poisson"),
        _bootstrap_pair(etas.evidence.validation, model_id="etas"),
    )
    return LocalSupportPrimaryPipelineResult(
        protocol_sha256=poisson.protocol_sha256,
        authorization_id=authorized_execution.authorization_id,
        poisson=poisson,
        etas=etas,
        g1=g1,
        selection=selection,
        validation_bootstrap=bootstrap,
        score_entries=entries,
        generated_scores=generated,
    )


__all__ = [
    "LocalSupportBootstrapOutcome",
    "LocalSupportPoissonFailureAccounting",
    "LocalSupportPrimaryPipelineResult",
    "build_local_support_poisson_failure_accounting",
    "run_local_support_primary_pipeline",
]
