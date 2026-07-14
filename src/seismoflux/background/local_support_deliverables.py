# ruff: noqa: RUF001
"""Independent, geometry-free publication path for stage-2R-1 results."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, cast

from seismoflux.background.artifacts import canonical_json_bytes
from seismoflux.background.config import BackgroundConfig
from seismoflux.background.evidence import (
    AuditedBackgroundModelEvidence,
    PairedInformationGainEvidence,
    PointProcessScoreEvidence,
)
from seismoflux.background.execution import (
    ExecutionSeal,
    GitCommandRunner,
    require_execution_seal_unchanged,
    subprocess_git_runner,
)
from seismoflux.background.issues import FrozenIssueCalendar
from seismoflux.background.local_support_runtime import LocalSupportRuntime
from seismoflux.background.pipeline_etas import ETASSnapshotAttempt
from seismoflux.background.pipeline_local_support import (
    LocalSupportBootstrapOutcome,
    build_local_support_poisson_failure_accounting,
)
from seismoflux.background.pipeline_poisson import (
    LocalSupportPoissonPartialFailureEvidence,
    LocalSupportPoissonSnapshotFit,
    LocalSupportScoreabilityGateEvidence,
    PoissonKDEInabilityCode,
    PoissonKDEScientificInability,
)
from seismoflux.background.poisson import BandwidthPreScoreGateEvidence
from seismoflux.background.publication import (
    BacktestBundle,
    BundleBinary,
    BundleDocument,
    BundlePublication,
    ExperimentBundle,
    FixedFilePublication,
    ModelBundle,
    ProcessedBundle,
    publish_backtest_bundle,
    publish_experiment_bundle,
    publish_fixed_project_file,
    publish_model_bundle,
    publish_processed_bundle,
)
from seismoflux.background.scientific import scientific_json, scientific_mapping
from seismoflux.background.score_ledger import (
    ScoreLedger,
    ScoreLedgerEntry,
    validate_generated_score_collection,
    validate_registry_score_references,
)
from seismoflux.background.scoring_authorization import (
    AuthorizedExecution,
    require_background_scoring_authorized,
)
from seismoflux.background.stage2r1 import (
    LocalSupportStage2R1Result,
    LocalSupportStageGateError,
)
from seismoflux.background.visualization_local_support import (
    LocalSupportVisualBootstrap,
    LocalSupportVisualHorizon,
    LocalSupportVisualSensitivity,
    LocalSupportVisualSnapshot,
    render_local_support_results_svg,
)

OutcomeStatus: TypeAlias = Literal["completed", "credible_negative"]
FailureStage: TypeAlias = Literal["numerical_regression", "poisson_kde"]

_BUNDLE_ORDER = ("processed", "model", "backtest", "experiment")
_SNAPSHOT_ORDER = ("fold_1", "fold_2", "fold_3", "fold_4", "final_validation")
_MODEL_ORDER = ("uniform_poisson", "spatial_poisson", "etas")
_PUBLIC_RESULTS_FIGURE = "docs/background_local_support_results.svg"
_SCORING_CODE_TAG = "v0.2.1-background-local-support-scoring-code"
_RESULTS_TAG = "v0.2.1-background-local-support-baselines"
_FORBIDDEN_PUBLIC_KEYS = {
    "assessment_target_locations",
    "cell_bounds",
    "clipped_geometry",
    "coordinates",
    "event_log_intensities",
    "geometry",
    "latitude",
    "longitude",
    "parent_event_ids",
    "representative_x_km",
    "representative_y_km",
    "target_event_ids",
    "training_event_ids",
    "validation_target_locations",
    "wkb",
    "x_km",
    "x_m",
    "y_km",
    "y_m",
}
_SAFE_FALSE_SPATIAL_FLAGS = {
    "contains_coordinates",
    "contains_geometry",
    "geometry_included",
    "public_coordinates_included",
    "public_geometry_included",
}
_FORBIDDEN_SPATIAL_KEY_TOKENS = {
    "bbox",
    "boundary",
    "bounds",
    "center",
    "centre",
    "centroid",
    "coordinate",
    "coordinates",
    "crs",
    "envelope",
    "ewkb",
    "ewkt",
    "extent",
    "geographic",
    "geography",
    "geohash",
    "geojson",
    "geom",
    "geometry",
    "lat",
    "latitude",
    "lng",
    "location",
    "locations",
    "lon",
    "longitude",
    "position",
    "srid",
    "wkb",
    "wkt",
}
_WKT_PREFIX = re.compile(
    r"^(?:SRID=\d+;)?(?:POINT|MULTIPOINT|LINESTRING|MULTILINESTRING|POLYGON|"
    r"MULTIPOLYGON|GEOMETRYCOLLECTION|CIRCULARSTRING|COMPOUNDCURVE|CURVEPOLYGON|"
    r"MULTICURVE|MULTISURFACE|POLYHEDRALSURFACE|TIN|TRIANGLE)"
    r"(?:\s+(?:Z|M|ZM))?\s*(?:\(|EMPTY\b)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class LocalSupportPoissonFailureEvidence:
    """Immutable scientific evidence retained at every local Poisson stop point."""

    reason_code: PoissonKDEInabilityCode
    message: str
    scoreability_gate_evidence: LocalSupportScoreabilityGateEvidence
    pre_score_gate_evidence: BandwidthPreScoreGateEvidence | None
    fitted_snapshots: tuple[LocalSupportPoissonSnapshotFit, ...] | None
    partial_failure_evidence: LocalSupportPoissonPartialFailureEvidence | None

    def __post_init__(self) -> None:
        if not self.message:
            raise ValueError("Poisson failure message must not be empty")
        if self.partial_failure_evidence is not None:
            if self.reason_code != "zero_target_snapshot":
                raise ValueError("partial Poisson Scores are valid only for final zero targets")
            if self.fitted_snapshots != self.partial_failure_evidence.snapshots:
                raise ValueError("partial Poisson evidence lost its fitted snapshots")
        if self.pre_score_gate_evidence is not None and self.fitted_snapshots is None:
            raise ValueError("Poisson numerical gate evidence requires fitted snapshots")
        if self.reason_code == "all_bandwidths_failed_numerical_gate" and (
            self.pre_score_gate_evidence is None or self.fitted_snapshots is None
        ):
            raise ValueError("all-bandwidth failure omitted target-free fit/gate evidence")

    @property
    def scores_started(self) -> bool:
        return self.partial_failure_evidence is not None

    @classmethod
    def from_error(
        cls,
        error: PoissonKDEScientificInability,
    ) -> LocalSupportPoissonFailureEvidence:
        scoreability = error.scoreability_gate_evidence
        if scoreability is None:
            raise ValueError("local Poisson scientific failure omitted scoreability evidence")
        return cls(
            reason_code=error.reason_code,
            message=str(error),
            scoreability_gate_evidence=scoreability,
            pre_score_gate_evidence=error.gate_evidence,
            fitted_snapshots=error.fitted_snapshots,
            partial_failure_evidence=error.partial_failure_evidence,
        )


@dataclass(frozen=True, slots=True)
class LocalSupportCredibleNegative:
    """One preregistered scientific inability, permanently separated from code errors."""

    protocol_sha256: str
    authorization_id: str
    failure_stage: FailureStage
    failure_code: str
    failure_reasons: tuple[str, ...]
    score_ledger: ScoreLedger
    poisson_failure_evidence: LocalSupportPoissonFailureEvidence | None
    locked_test_run: bool = False
    locked_test_score_ids: tuple[str, ...] = ()
    locked_test_artifact_ids: tuple[str, ...] = ()
    locked_test_target_count: None = None
    locked_test_result: None = None

    def __post_init__(self) -> None:
        if len(self.protocol_sha256) != 64 or len(self.authorization_id) != 64:
            raise ValueError("credible-negative protocol/authorization IDs must be SHA-256")
        if self.failure_stage not in {"numerical_regression", "poisson_kde"}:
            raise ValueError("credible-negative failure stage is not preregistered")
        if (
            not self.failure_code
            or not self.failure_reasons
            or any(not reason for reason in self.failure_reasons)
        ):
            raise ValueError("credible-negative failure evidence must not be empty")
        if (
            self.locked_test_run is not False
            or self.locked_test_score_ids
            or self.locked_test_artifact_ids
            or self.locked_test_target_count is not None
            or self.locked_test_result is not None
        ):
            raise ValueError("credible-negative locked-test proof must be false/empty/null")
        if self.score_ledger.protocol_sha256 != self.protocol_sha256:
            raise ValueError("credible-negative ledger uses another protocol")
        if self.score_ledger.authorization_id != self.authorization_id:
            raise ValueError("credible-negative ledger uses another authorization")
        if self.score_ledger.coverage != "fragment":
            raise ValueError("credible-negative score ledger must be an honest fragment")
        self.score_ledger.assert_locked_test_not_run()
        if (self.failure_stage == "poisson_kde") != (self.poisson_failure_evidence is not None):
            raise ValueError("credible-negative Poisson evidence/stage binding differs")

    @property
    def g1_ls_passed(self) -> bool:
        return False

    @property
    def stage3_allowed(self) -> bool:
        return False


LocalSupportStage2R1Outcome: TypeAlias = LocalSupportStage2R1Result | LocalSupportCredibleNegative


def credible_negative_from_error(
    config: BackgroundConfig,
    calendar: FrozenIssueCalendar,
    runtime: LocalSupportRuntime,
    authorized_execution: AuthorizedExecution,
    error: LocalSupportStageGateError | PoissonKDEScientificInability,
) -> LocalSupportCredibleNegative:
    """Translate only known scientific gate failures; programming errors still propagate."""

    entries: tuple[ScoreLedgerEntry, ...]
    generated: tuple[PointProcessScoreEvidence, ...]
    if isinstance(error, LocalSupportStageGateError):
        stage: FailureStage = "numerical_regression"
        code = "numerical_regression_failed"
        entries = ()
        generated = ()
        poisson_failure = None
    elif isinstance(error, PoissonKDEScientificInability):
        stage = "poisson_kde"
        code = error.reason_code
        accounting = build_local_support_poisson_failure_accounting(
            config,
            runtime,
            authorized_execution,
            error,
        )
        entries = accounting.score_entries
        generated = accounting.generated_scores
        poisson_failure = LocalSupportPoissonFailureEvidence.from_error(error)
    else:  # pragma: no cover - narrowed by the public type and kept fail-closed
        raise TypeError("unsupported stage-2R-1 failure type")
    ledger = ScoreLedger(
        protocol_sha256=authorized_execution.scoring_authorization.protocol_sha256,
        authorization_id=authorized_execution.authorization_id,
        issue_manifest_sha256=config.inputs.issue_manifest_sha256,
        calendar=calendar,
        entries=entries,
        coverage="fragment",
    )
    validate_generated_score_collection(ledger, generated)
    return LocalSupportCredibleNegative(
        protocol_sha256=authorized_execution.scoring_authorization.protocol_sha256,
        authorization_id=authorized_execution.authorization_id,
        failure_stage=stage,
        failure_code=code,
        failure_reasons=(str(error),),
        score_ledger=ledger,
        poisson_failure_evidence=poisson_failure,
    )


def _assert_public_payload(value: object, *, location: str = "$") -> None:
    """Reject geometry, coordinates, event-level identities, and spatial reconstruction data."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"public payload key at {location} must be a string")
            camel_split = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key)
            camel_split = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", camel_split)
            normalized = re.sub(r"[^a-z0-9]+", "_", camel_split.casefold()).strip("_")
            tokens = frozenset(normalized.split("_"))
            safe_false_flag = normalized in _SAFE_FALSE_SPATIAL_FLAGS and item is False
            identity_subject = tokens.intersection(
                {"event", "events", "earthquake", "earthquakes", "quake", "quakes"}
            )
            identity_marker = tokens.intersection({"id", "ids", "identifier", "identifiers"})
            event_identity = bool(identity_subject and identity_marker)
            bare_event_identity = normalized in {
                "event",
                "events",
                "earthquake",
                "earthquakes",
                "quake",
                "quakes",
            }
            spatial_key = bool(tokens.intersection(_FORBIDDEN_SPATIAL_KEY_TOKENS))
            xy_key = normalized in {"x", "y"} or normalized.endswith(
                ("_x", "_y", "_x_km", "_y_km", "_x_m", "_y_m")
            )
            if not safe_false_flag and (
                normalized in _FORBIDDEN_PUBLIC_KEYS
                or event_identity
                or bare_event_identity
                or spatial_key
                or xy_key
            ):
                raise ValueError(f"forbidden public spatial/event field at {location}.{key}")
            _assert_public_payload(item, location=f"{location}.{key}")
        return
    if isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _assert_public_payload(item, location=f"{location}[{index}]")
        return
    if isinstance(value, str) and _WKT_PREFIX.match(value.lstrip()) is not None:
        raise ValueError(f"public payload contains WKT geometry at {location}")


def _document(relative_path: str, value: object) -> BundleDocument:
    mapped = scientific_mapping(value, location=relative_path)
    _assert_public_payload(mapped, location=relative_path)
    return BundleDocument(relative_path, cast(dict[str, object], mapped))


def _support_summary(runtime: LocalSupportRuntime) -> dict[str, object]:
    snapshots: list[dict[str, object]] = []
    for item in runtime.snapshots:
        support = item.support
        status_counts = {
            status: sum(cell.status == status for cell in support.cells)
            for status in ("supported", "unsupported", "indeterminate")
        }
        snapshots.append(
            {
                "snapshot_id": item.snapshot_id,
                "support_id": support.support_id,
                "fit_end_utc": support.fit_end_utc,
                "common_mc": support.common_mc,
                "supported_area_km2": support.retained_area_m2 / 1_000_000.0,
                "total_area_km2": support.total_area_m2 / 1_000_000.0,
                "supported_area_fraction": support.retained_area_fraction,
                "fixed_cell_status_counts": status_counts,
                "integration_cell_counts": {
                    f"{grid.spec.cell_size_km:g}km": len(grid.cells)
                    for grid in item.grid_family.grids
                },
                "compensator_domain_id": item.compensator_domain_id,
            }
        )
    return {
        "manifest_id": runtime.manifest_id,
        "snapshot_count": len(snapshots),
        "snapshots": snapshots,
        "public_geometry_included": False,
        "public_coordinates_included": False,
    }


def _score_pair_summary(pair: PairedInformationGainEvidence) -> dict[str, object]:
    candidate = pair.candidate
    uniform = pair.uniform
    return {
        "status": "succeeded",
        "snapshot_id": candidate.snapshot_id,
        "model_id": candidate.model_id,
        "model_variant_id": candidate.model_variant_id,
        "parameter_snapshot_id": candidate.parameter_snapshot_id,
        "support_id": candidate.support_id,
        "supported_area_km2": candidate.supported_area_km2,
        "compensator_domain_id": candidate.compensator_domain_id,
        "selected_mc": candidate.selected_mc,
        "target_event_count": len(candidate.target_event_ids),
        "candidate_score_id": candidate.score_id,
        "uniform_score_id": uniform.score_id,
        "candidate_log_likelihood": candidate.log_likelihood,
        "uniform_log_likelihood": uniform.log_likelihood,
        "information_gain_nats_per_event": pair.information_gain_per_event,
    }


def _model_evidence_summary(evidence: AuditedBackgroundModelEvidence) -> dict[str, object]:
    scored = {
        item.candidate.snapshot_id: _score_pair_summary(item)
        for item in (
            *evidence.development_folds,
            *((evidence.validation,) if evidence.validation is not None else ()),
        )
    }
    failures = dict(evidence.failed_snapshot_reasons)
    snapshots: list[dict[str, object]] = []
    for snapshot_id in _SNAPSHOT_ORDER:
        if snapshot_id in scored:
            snapshots.append(scored[snapshot_id])
        else:
            snapshots.append(
                {
                    "status": "failed",
                    "snapshot_id": snapshot_id,
                    "model_id": evidence.model_id,
                    "model_variant_id": evidence.model_variant_id,
                    "failure_reasons": failures[snapshot_id],
                }
            )
    return {
        "model_id": evidence.model_id,
        "model_variant_id": evidence.model_variant_id,
        "eligible_for_selection": evidence.eligible_for_selection,
        "positive_development_fold_count": evidence.positive_development_fold_count,
        "passes_g1_ls_as_same_model": evidence.passes_g1_as_same_model,
        "snapshots": snapshots,
    }


def _poisson_snapshot_public_summary(
    item: LocalSupportPoissonSnapshotFit,
) -> dict[str, object]:
    return {
        "snapshot_id": item.definition.snapshot_id,
        "support_id": item.support_id,
        "selected_mc": item.selected_mc,
        "supported_area_km2": item.supported_area_km2,
        "compensator_domain_id": item.compensator_domain_id,
        "training_event_count": item.training_event_count,
        "training_duration_days": item.training_duration_days,
        "rate_per_day": item.rate_per_day,
        "training_evidence_id": item.training_evidence_id,
        "grid_gates": scientific_json(item.grid_gate_evidence),
    }


def _poisson_failure_public_summary(
    evidence: LocalSupportPoissonFailureEvidence,
) -> dict[str, object]:
    partial = evidence.partial_failure_evidence
    return {
        "status": "failed_after_development_scores" if evidence.scores_started else "failed",
        "reason_code": evidence.reason_code,
        "message": evidence.message,
        "scores_started": evidence.scores_started,
        "scoreability_gate": scientific_json(evidence.scoreability_gate_evidence),
        "pre_score_gates": scientific_json(evidence.pre_score_gate_evidence),
        "snapshots": (
            ()
            if evidence.fitted_snapshots is None
            else tuple(_poisson_snapshot_public_summary(item) for item in evidence.fitted_snapshots)
        ),
        "bandwidth_selection": (
            None if partial is None else scientific_json(partial.bandwidth_selection)
        ),
        "development_uniform_scores": (
            ()
            if partial is None
            else tuple(
                {
                    "snapshot_id": score.snapshot_id,
                    "score_id": score.score_id,
                    "target_event_count": len(score.target_event_ids),
                    "log_likelihood": score.log_likelihood,
                }
                for score in partial.development_uniform_scores
            )
        ),
        "development_kde_attempts": (
            ()
            if partial is None
            else tuple(
                {
                    "bandwidth_km": audit.bandwidth_km,
                    "fold_scores": tuple(
                        _score_pair_summary(pair) for pair in audit.development_folds
                    ),
                }
                for audit in partial.bandwidth_fold_audits
            )
        ),
        "failed_bandwidth_gates": (
            () if partial is None else scientific_json(partial.failed_bandwidth_gate_items)
        ),
        "final_validation": "not_run_zero_target" if partial is not None else "not_run",
        "etas": "not_run",
        "secondary_horizons": "not_run",
        "future_ensembles": "not_run",
    }


def _etas_attempt_summary(attempt: ETASSnapshotAttempt) -> dict[str, object]:
    fit = attempt.fit_result
    stability = fit.stability if fit is not None else None
    parameters = fit.best_parameters if fit is not None else None
    grid = attempt.grid_gate_evidence
    return {
        "snapshot_id": attempt.definition.snapshot_id,
        "status": "succeeded" if attempt.succeeded else "failed",
        "model_variant_id": attempt.model_variant_id,
        "parameter_snapshot_id": attempt.parameter_snapshot_id,
        "selected_mc": attempt.selected_mc,
        "fit_target_count": len(attempt.fit_selection.target_event_ids),
        "fit_parent_count": len(attempt.fit_selection.parent_event_ids),
        "score_target_count": len(attempt.score_selection.target_event_ids),
        "score_parent_count": len(attempt.score_selection.parent_event_ids),
        "parameters": scientific_json(parameters),
        "stability": (
            None
            if stability is None
            else {
                "stable": stability.stable,
                "converged_start_count": stability.converged_start_count,
                "best_three_relative_objective_range": (
                    stability.best_three_relative_objective_range
                ),
                "best_three_transformed_parameter_range": (
                    stability.best_three_transformed_parameter_range
                ),
                "hessian_success": stability.hessian.success,
                "hessian_minimum_eigenvalue": stability.hessian.minimum_eigenvalue,
                "hessian_condition_number": stability.hessian.condition_number,
                "failure_reasons": stability.failure_reasons,
            }
        ),
        "grid_gate": (
            None
            if grid is None
            else {
                "passed": grid.passed,
                "numerical_evidence_id": grid.numerical_evidence_id,
                "failure_reasons": grid.failure_reasons,
            }
        ),
        "score": (
            None
            if attempt.paired_evidence is None
            else _score_pair_summary(attempt.paired_evidence)
        ),
        "failure_reasons": attempt.failure_reasons,
    }


def _etas_parent_sensitivity_summary(
    outcome: LocalSupportStage2R1Result,
) -> tuple[dict[str, object], ...]:
    primary = {item.definition.snapshot_id: item for item in outcome.primary.etas.primary.attempts}
    excluded = {
        item.definition.snapshot_id: item for item in outcome.primary.etas.sensitivity_attempts
    }
    rows: list[dict[str, object]] = []
    for snapshot_id in ("fold_1", "fold_3"):
        primary_attempt = primary[snapshot_id]
        excluded_attempt = excluded[snapshot_id]
        primary_pair = primary_attempt.paired_evidence
        excluded_pair = excluded_attempt.paired_evidence
        primary_ig = primary_pair.information_gain_per_event if primary_pair is not None else None
        excluded_ig = (
            excluded_pair.information_gain_per_event if excluded_pair is not None else None
        )
        difference = (
            excluded_ig - primary_ig if primary_ig is not None and excluded_ig is not None else None
        )
        rows.append(
            {
                "snapshot_id": snapshot_id,
                "status": (
                    "completed"
                    if primary_ig is not None and excluded_ig is not None
                    else "not_evaluable"
                ),
                "primary_includes_eligible_unsupported_parents": {
                    "model_variant_id": primary_attempt.model_variant_id,
                    "score_id": (
                        primary_pair.candidate.score_id if primary_pair is not None else None
                    ),
                    "information_gain_nats_per_event": primary_ig,
                    "failure_reasons": primary_attempt.failure_reasons,
                },
                "exclude_all_unsupported_parents": {
                    "model_variant_id": excluded_attempt.model_variant_id,
                    "score_id": (
                        excluded_pair.candidate.score_id if excluded_pair is not None else None
                    ),
                    "information_gain_nats_per_event": excluded_ig,
                    "failure_reasons": excluded_attempt.failure_reasons,
                },
                "information_gain_difference_exclude_minus_primary": difference,
                "role": "conditional_parent_history_sensitivity_not_spatial_support_spillover",
            }
        )
    return tuple(rows)


def _bootstrap_summary(item: LocalSupportBootstrapOutcome) -> dict[str, object]:
    interval = item.interval
    return {
        "model_id": item.model_id,
        "status": "completed" if interval is not None else "not_run",
        "score_id": item.score_id,
        "point_estimate": interval.point_estimate if interval is not None else None,
        "lower": interval.lower if interval is not None else None,
        "upper": interval.upper if interval is not None else None,
        "replications": interval.replications if interval is not None else None,
        "confidence_level": interval.confidence_level if interval is not None else None,
        "failure_reason": item.failure_reason,
    }


def _horizon_summary(result: LocalSupportStage2R1Result) -> dict[str, object]:
    comparisons = []
    for item in result.horizons.comparisons:
        comparisons.append(
            {
                "model_id": item.candidate_model_id,
                "publication_delay_days": item.publication_delay_days,
                "horizon_days": item.horizon_days,
                "exposure_count": len(item.exposures),
                "target_event_count": sum(
                    len(exposure.paired_evidence.candidate.target_event_ids)
                    for exposure in item.exposures
                ),
                "information_gain_nats_per_event": item.information_gain_per_event,
            }
        )
    failures = [
        {
            "model_id": item.candidate_model_id,
            "publication_delay_days": item.publication_delay_days,
            "horizon_days": item.horizon_days,
            "reason": item.reason,
        }
        for item in result.horizons.failed_comparisons
    ]
    return {
        "role": "secondary_supported_domain_integration_diagnostic_not_used_by_G1_LS",
        "comparisons": comparisons,
        "failed_comparisons": failures,
    }


def _future_seed_contract(config: BackgroundConfig) -> tuple[dict[str, object], int]:
    derivation = config.randomness.seed_derivation
    context = derivation.namespace_contexts.future_simulation
    namespace = next(
        (value for value in derivation.namespaces if value == "future_simulation"),
        None,
    )
    if namespace is None:
        raise ValueError("frozen randomness config has no future_simulation namespace")
    replicate_count = context.replicate_index_last_inclusive - context.replicate_index_first + 1
    seed_contract: dict[str, object] = {
        "root_seed": config.randomness.root_seed,
        "protocol_version": config.protocol_version,
        "namespace": namespace,
        "model_id": context.model_id,
        "issue_id_rule": context.issue_id_rule,
        "replicate_index_role": context.replicate_index_role,
        "replicate_index_first": context.replicate_index_first,
        "replicate_index_last_inclusive": context.replicate_index_last_inclusive,
        "ordered_fields": derivation.ordered_fields,
        "encoding": derivation.encoding,
        "separator_hex": derivation.separator_hex,
        "digest": derivation.digest,
        "entropy": derivation.entropy,
        "generator": derivation.generator,
        "worker_count_invariant": derivation.worker_count_invariant,
        "gather_order": derivation.gather_order,
    }
    seed_contract["seed_context_identity"] = hashlib.sha256(
        canonical_json_bytes(seed_contract)
    ).hexdigest()
    return seed_contract, replicate_count


def _future_summary(
    config: BackgroundConfig,
    result: LocalSupportStage2R1Result,
) -> dict[str, object]:
    future = result.future
    seed_contract, configured_replicate_count = _future_seed_contract(config)
    future_config = config.randomness.future_simulation
    maximum_events = future_config.maximum_events_per_replicate
    if future.ensembles is None:
        return {
            "status": "not_run",
            "failure_reason": future.failure_reason,
            "support_id": future.support_id,
            "compensator_domain_id": future.compensator_domain_id,
            "parameter_snapshot_id": future.parameter_snapshot_id,
            "issue_count": 0,
            "root_seed": config.randomness.root_seed,
            "seed_derivation": seed_contract,
            "replicate_count": 0,
            "configured_replicate_count": configured_replicate_count,
            "maximum_events_per_replicate": maximum_events,
            "event_cap_hit_is_failure": future_config.event_cap_hit_is_failure,
            "event_cap_exceeded": None,
        }
    ensembles = future.ensembles
    if future.status != "succeeded":
        raise ValueError("completed future ensembles require succeeded outcome status")
    observed_replicate_counts = {issue.replicate_count for issue in ensembles.issues}
    if observed_replicate_counts != {configured_replicate_count}:
        raise ValueError("completed future replicate count differs from the frozen seed context")
    return {
        "status": "completed",
        "failure_reason": None,
        "support_id": future.support_id,
        "compensator_domain_id": future.compensator_domain_id,
        "parameter_snapshot_id": future.parameter_snapshot_id,
        "issue_count": len(ensembles.issues),
        "root_seed": config.randomness.root_seed,
        "seed_derivation": seed_contract,
        "replicate_count": next(iter(observed_replicate_counts)),
        "maximum_events_per_replicate": maximum_events,
        "event_cap_hit_is_failure": future_config.event_cap_hit_is_failure,
        "event_cap_exceeded": future.status != "succeeded",
        "resources": scientific_json(ensembles.resources),
        "issues": [
            {
                "issue_date_local": issue.issue_date_local,
                "issue_id": issue.issue_id,
                "replicate_count": issue.replicate_count,
                "horizons": [
                    {
                        "horizon_days": horizon.horizon_days,
                        "mean_count": horizon.mean_count,
                        "quantiles": scientific_json(horizon.quantiles),
                    }
                    for horizon in issue.horizons
                ],
            }
            for issue in ensembles.issues
        ],
        "spatial_grid_values_public": False,
    }


def _visual_inputs(
    runtime: LocalSupportRuntime,
    outcome: LocalSupportStage2R1Outcome,
) -> tuple[
    tuple[LocalSupportVisualSnapshot, ...],
    tuple[LocalSupportVisualHorizon, ...],
    tuple[LocalSupportVisualBootstrap, ...],
    tuple[LocalSupportVisualSensitivity, ...],
    str,
]:
    if isinstance(outcome, LocalSupportStage2R1Result):
        spatial = outcome.primary.poisson.spatial_evidence
        etas = outcome.primary.etas.evidence
        spatial_pairs = (
            *spatial.development_folds,
            *((spatial.validation,) if spatial.validation is not None else ()),
        )
        spatial_values = {
            item.candidate.snapshot_id: item.information_gain_per_event for item in spatial_pairs
        }
        etas_values = {
            item.candidate.snapshot_id: item.information_gain_per_event
            for item in (*etas.development_folds, *((etas.validation,) if etas.validation else ()))
        }
        primary_attempts = {
            item.definition.snapshot_id: item for item in outcome.primary.etas.primary.attempts
        }
        excluded_attempts = {
            item.definition.snapshot_id: item for item in outcome.primary.etas.sensitivity_attempts
        }
        selected = outcome.primary.selection.selected_model_variant_id
    else:
        spatial_values = {}
        etas_values = {}
        primary_attempts = {}
        excluded_attempts = {}
        selected = "none_scientific_gate_failed"
    snapshots = tuple(
        LocalSupportVisualSnapshot(
            snapshot_id=item.snapshot_id,
            supported_area_fraction=item.support.retained_area_fraction,
            spatial_information_gain=cast(float | None, spatial_values.get(item.snapshot_id)),
            etas_information_gain=cast(float | None, etas_values.get(item.snapshot_id)),
        )
        for item in runtime.snapshots
    )

    if isinstance(outcome, LocalSupportStage2R1Result):
        horizon_lookup = {
            (item.candidate_model_id, item.horizon_days): item.information_gain_per_event
            for item in outcome.horizons.comparisons
            if item.publication_delay_days == 0
        }
        bootstrap_lookup = {item.model_id: item for item in outcome.primary.validation_bootstrap}
    else:
        horizon_lookup = {}
        bootstrap_lookup = {}
    horizons = tuple(
        LocalSupportVisualHorizon(
            horizon_days=horizon,
            spatial_information_gain=horizon_lookup.get(("spatial_poisson", horizon)),
            etas_information_gain=horizon_lookup.get(("etas", horizon)),
        )
        for horizon in (7, 30, 90, 180, 365)
    )
    bootstrap_items: list[LocalSupportVisualBootstrap] = []
    for model_id in ("spatial_poisson", "etas"):
        item = bootstrap_lookup.get(model_id)
        interval = item.interval if item is not None else None
        bootstrap_items.append(
            LocalSupportVisualBootstrap(
                model_id=model_id,
                point_estimate=interval.point_estimate if interval is not None else None,
                lower=interval.lower if interval is not None else None,
                upper=interval.upper if interval is not None else None,
            )
        )
    sensitivity_items: list[LocalSupportVisualSensitivity] = []
    for snapshot_id in ("fold_1", "fold_3"):
        primary_attempt = primary_attempts.get(snapshot_id)
        excluded_attempt = excluded_attempts.get(snapshot_id)
        primary_pair = primary_attempt.paired_evidence if primary_attempt is not None else None
        excluded_pair = excluded_attempt.paired_evidence if excluded_attempt is not None else None
        primary_ig = primary_pair.information_gain_per_event if primary_pair else None
        excluded_ig = excluded_pair.information_gain_per_event if excluded_pair else None
        sensitivity_items.append(
            LocalSupportVisualSensitivity(
                snapshot_id=snapshot_id,
                primary_information_gain=primary_ig,
                excluded_information_gain=excluded_ig,
            )
        )
    return snapshots, horizons, tuple(bootstrap_items), tuple(sensitivity_items), selected


@dataclass(frozen=True, slots=True)
class LocalSupportDeliverables:
    processed: ProcessedBundle
    model: ModelBundle
    backtest: BacktestBundle
    experiment: ExperimentBundle
    registry_science: dict[str, object]
    results_svg: bytes


def build_local_support_deliverables(
    config: BackgroundConfig,
    runtime: LocalSupportRuntime,
    outcome: LocalSupportStage2R1Outcome,
    authorized_execution: AuthorizedExecution,
) -> LocalSupportDeliverables:
    """Adapt a completed comparison or a known credible negative without recomputing."""

    require_background_scoring_authorized(config, authorized_execution)
    if outcome.protocol_sha256 != authorized_execution.scoring_authorization.protocol_sha256:
        raise ValueError("local-support outcome uses another protocol")
    if outcome.authorization_id != authorized_execution.authorization_id:
        raise ValueError("local-support outcome uses another scoring authorization")
    if isinstance(outcome, LocalSupportStage2R1Result) and outcome.score_ledger.coverage != (
        "complete"
    ):
        raise ValueError("successful local-support publication requires a complete score ledger")
    if isinstance(outcome, LocalSupportCredibleNegative) and (
        outcome.score_ledger.coverage != "fragment"
    ):
        raise ValueError("credible-negative publication requires a fragment score ledger")

    support_summary = _support_summary(runtime)
    status: OutcomeStatus = (
        "completed" if isinstance(outcome, LocalSupportStage2R1Result) else "credible_negative"
    )
    base_identity = {
        "protocol_version": "0.2.1",
        "protocol_sha256": outcome.protocol_sha256,
        "authorization_id": outcome.authorization_id,
        "support_manifest_id": runtime.manifest_id,
        "outcome_status": status,
    }

    if isinstance(outcome, LocalSupportStage2R1Result):
        numerical = scientific_json(outcome.numerical_regression)
        resources = scientific_json(outcome.resources)
        model_evidence = tuple(
            _model_evidence_summary(item)
            for item in (
                outcome.primary.poisson.uniform_evidence,
                outcome.primary.poisson.spatial_evidence,
                outcome.primary.etas.evidence,
            )
        )
        poisson_model: dict[str, object] = {
            "selected_bandwidth_km": outcome.primary.poisson.selected_bandwidth_km,
            "best_mean_bandwidth_km": (
                outcome.primary.poisson.bandwidth_selection.best_mean_bandwidth_km
            ),
            "bandwidth_candidates": scientific_json(
                outcome.primary.poisson.bandwidth_selection.candidates
            ),
            "scoreability_gate": scientific_json(
                outcome.primary.poisson.scoreability_gate_evidence
            ),
            "pre_score_gates": scientific_json(outcome.primary.poisson.pre_score_gate_evidence),
            "snapshots": [
                _poisson_snapshot_public_summary(item) for item in outcome.primary.poisson.snapshots
            ],
        }
        etas_model: dict[str, object] = {
            "primary_attempts": [
                _etas_attempt_summary(item) for item in outcome.primary.etas.primary.attempts
            ],
            "unsupported_parent_sensitivity": [
                _etas_attempt_summary(item) for item in outcome.primary.etas.sensitivity_attempts
            ],
            "sensitivity_not_applicable": outcome.primary.etas.sensitivity_not_applicable,
        }
        etas_parent_sensitivity = _etas_parent_sensitivity_summary(outcome)
        bootstrap = tuple(_bootstrap_summary(item) for item in outcome.primary.validation_bootstrap)
        horizons = _horizon_summary(outcome)
        future = _future_summary(config, outcome)
        g1_payload = cast(dict[str, object], scientific_json(outcome.primary.g1))
        g1_payload["status"] = "passed" if outcome.primary.g1.passed else "failed"
        g1_selection: dict[str, object] = {
            "g1_ls": g1_payload,
            "selection": scientific_json(outcome.primary.selection),
            "stage3_allowed": outcome.stage3_allowed,
        }
        ledger = outcome.score_ledger.semantic_payload()
        score_ids = outcome.score_ledger.score_ids
        ledger_id = outcome.score_ledger.ledger_id
        ledger_sha256 = outcome.score_ledger.ledger_sha256
        classification_counts: object = scientific_json(outcome.score_ledger.classification_counts)
        failure = None
    else:
        numerical = None
        resources = None
        model_evidence = ()
        poisson_model = (
            {"status": "not_started_numerical_regression_failed"}
            if outcome.poisson_failure_evidence is None
            else _poisson_failure_public_summary(outcome.poisson_failure_evidence)
        )
        etas_model = {"status": "not_run"}
        etas_parent_sensitivity = tuple(
            {
                "snapshot_id": snapshot_id,
                "status": "not_run",
                "role": ("conditional_parent_history_sensitivity_not_spatial_support_spillover"),
            }
            for snapshot_id in ("fold_1", "fold_3")
        )
        bootstrap = ()
        horizons = {"comparisons": (), "failed_comparisons": ()}
        future = {"status": "not_run", "issue_count": 0}
        g1_selection = {
            "g1_ls": {
                "status": "not_evaluable",
                "passed": False,
                "passing_model_variants": (),
            },
            "selection": {"status": "not_evaluable"},
            "stage3_allowed": False,
        }
        ledger = outcome.score_ledger.semantic_payload()
        ledger_sha256 = outcome.score_ledger.ledger_sha256
        ledger_id = outcome.score_ledger.ledger_id
        score_ids = outcome.score_ledger.score_ids
        classification_counts = scientific_json(outcome.score_ledger.classification_counts)
        failure = {
            "stage": outcome.failure_stage,
            "code": outcome.failure_code,
            "reasons": outcome.failure_reasons,
        }

    processed = ProcessedBundle(
        {**base_identity, "artifact_role": "local_support_processed"},
        (
            _document("support_summary.json", support_summary),
            _document(
                "execution_evidence.json",
                {"numerical_regression": numerical, "resources": resources},
            ),
        ),
    )
    model = ModelBundle(
        {**base_identity, "artifact_role": "local_support_model"},
        (
            _document("model_evidence.json", {"models": model_evidence}),
            _document("poisson_kde.json", poisson_model),
            _document("etas.json", etas_model),
        ),
    )
    backtest = BacktestBundle(
        {
            **base_identity,
            "artifact_role": "local_support_backtest",
            "score_ledger_sha256": ledger_sha256,
        },
        (
            _document("score_ledger.json", ledger),
            _document("bootstrap.json", {"outcomes": bootstrap}),
            _document("g1_ls_and_selection.json", g1_selection),
            _document("secondary_horizons.json", horizons),
        ),
    )

    (
        visual_snapshots,
        visual_horizons,
        visual_bootstrap,
        visual_sensitivity,
        selected,
    ) = _visual_inputs(runtime, outcome)
    results_svg = render_local_support_results_svg(
        snapshots=visual_snapshots,
        horizons=visual_horizons,
        bootstrap=visual_bootstrap,
        sensitivity=visual_sensitivity,
        g1_ls_status=(
            "not_evaluable"
            if isinstance(outcome, LocalSupportCredibleNegative)
            else ("passed" if outcome.g1_ls_passed else "failed")
        ),
        selected_model_variant_id=selected,
    )
    experiment = ExperimentBundle(
        {**base_identity, "artifact_role": "local_support_experiment"},
        (
            _document("future_summary.json", future),
            _document(
                "visualization_source.json",
                {
                    "snapshots": visual_snapshots,
                    "horizons_delay_0_days": visual_horizons,
                    "validation_bootstrap": visual_bootstrap,
                    "etas_unsupported_parent_sensitivity": visual_sensitivity,
                    "selected_model_variant_id": selected,
                    "geometry_included": False,
                },
            ),
            BundleBinary(
                "results_overview.svg",
                results_svg,
                media_type="image/svg+xml",
            ),
        ),
    )

    ledger_entries = cast(list[object] | tuple[object, ...], ledger["entries"])
    registry_science: dict[str, object] = {
        "outcome_status": status,
        "failure": failure,
        "support": support_summary,
        "model_evidence": model_evidence,
        "poisson_kde": poisson_model,
        "etas_parent_sensitivity": etas_parent_sensitivity,
        "bootstrap": bootstrap,
        "g1_ls_and_selection": g1_selection,
        "secondary_horizons": horizons,
        "future": future,
        "score_ledger": {
            "ledger_id": ledger_id,
            "ledger_sha256": ledger_sha256,
            "coverage": ledger["coverage"],
            "entry_count": len(ledger_entries),
            "score_count": len(score_ids),
            "score_ids": score_ids,
            "classification_counts": classification_counts,
        },
    }
    _assert_public_payload(registry_science, location="registry_science")
    return LocalSupportDeliverables(
        processed=processed,
        model=model,
        backtest=backtest,
        experiment=experiment,
        registry_science=registry_science,
        results_svg=results_svg,
    )


@dataclass(frozen=True, slots=True)
class PublishedLocalSupportDeliverables:
    processed: BundlePublication
    model: BundlePublication
    backtest: BundlePublication
    experiment: BundlePublication
    registry: FixedFilePublication
    report: FixedFilePublication
    results_figure: FixedFilePublication
    registry_payload: dict[str, object]

    @property
    def bundle_publications(self) -> tuple[BundlePublication, ...]:
        return (self.processed, self.model, self.backtest, self.experiment)


def _bundle_reference(item: BundlePublication) -> dict[str, object]:
    return {
        "bundle_kind": item.bundle_kind,
        "artifact_id": item.artifact.artifact_id,
        "manifest_sha256": item.artifact.manifest_sha256,
    }


def _registry_bytes(registry: Mapping[str, object]) -> bytes:
    _assert_public_payload(registry, location="registry")
    return (
        json.dumps(
            registry,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _format_float(value: object) -> str:
    if value is None:
        return "—"
    number = float(cast(float, value))
    return f"{number:+.5f}" if math.isfinite(number) else "—"


def _report_bytes(registry: Mapping[str, object]) -> bytes:
    science = cast(dict[str, object], registry["science"])
    support = cast(dict[str, object], science["support"])
    snapshots = cast(list[dict[str, object]], support["snapshots"])
    g1_container = cast(dict[str, object], science["g1_ls_and_selection"])
    g1 = cast(dict[str, object], g1_container["g1_ls"])
    ledger = cast(dict[str, object], science["score_ledger"])
    outcome = str(science["outcome_status"])
    passed = bool(g1.get("passed", False))
    g1_status = str(g1.get("status", "passed" if passed else "failed"))
    g1_label = {
        "passed": "通过",
        "failed": "未通过",
        "not_evaluable": "未评估（门前停止）",
    }.get(g1_status)
    if g1_label is None:
        raise ValueError("registry contains an unknown G1-LS status")
    lines = [
        "# SeismoFlux 背景局部支持域报告（G1-LS）",
        "",
        f"- 协议版本：`{registry['protocol_version']}`",
        f"- 结果状态：`{outcome}`",
        f"- G1-LS：`{g1_label}`",
        f"- 阶段3允许：`{str(bool(registry['stage3_allowed'])).lower()}`",
        f"- Score Ledger：`{ledger['ledger_id']}`（{ledger['score_count']} 个 Score ID）",
        "- 锁定测试：未运行；锁定测试的 Score ID、产物 ID 为空，目标数为 null。",
        "",
        "> 本报告中的数值是固定局部支持域内的条件强度和相对信息增益，不是绝对发震概率。",
        "",
        "![G1-LS 结果总览](background_local_support_results.svg)",
        "",
        "## 数据与支持域",
        "",
        "| 快照 | 公共 Mc | 支持面积比例 | 支持面积（km²） | support_id |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for item in snapshots:
        lines.append(
            (
                "| {snapshot_id} | {common_mc:.1f} | {fraction:.3%} | {area:.1f} | `{support_id}` |"
            ).format(
                snapshot_id=item["snapshot_id"],
                common_mc=float(cast(float, item["common_mc"])),
                fraction=float(cast(float, item["supported_area_fraction"])),
                area=float(cast(float, item["supported_area_km2"])),
                support_id=item["support_id"],
            )
        )
    lines.extend(
        (
            "",
            (
                "局部 `raw Mc > 4.0` 只把对应既有固定格标为 unsupported，"
                "不提高其他格的排除状态。达到冻结局部 Mc 的 unsupported 事件仅可作为 "
                "ETAS 条件父历史，并另报完全排除这些父事件的敏感性。"
            ),
            "",
            "## 模型与判定",
            "",
            (
                "比较顺序为均匀 Poisson、支持域归一 KDE 和 ETAS；三者在每个快照"
                "共享目标事件、有效面积与补偿积分域。KDE 带宽与最终模型严格按冻结 "
                "one-SE 规则选择。"
            ),
            "",
        )
    )
    failure = science.get("failure")
    if failure is not None:
        failure_mapping = cast(dict[str, object], failure)
        lines.extend(
            (
                "## 可信负结果",
                "",
                f"- 失败阶段：`{failure_mapping['stage']}`",
                f"- 原因代码：`{failure_mapping['code']}`",
                f"- 说明：{'；'.join(cast(tuple[str, ...], failure_mapping['reasons']))}",
                "",
                "该结果按预登记停止，没有进入阶段3，也没有针对锁定测试调参。",
                "",
            )
        )
    else:
        models = cast(tuple[dict[str, object], ...], science["model_evidence"])
        lines.extend(
            (
                "| 模型 | 最终验证 IG（nats/事件） | 正增益开发折 | G1-LS 同模型通过 |",
                "| --- | ---: | ---: | --- |",
            )
        )
        for model in models:
            final = cast(list[dict[str, object]], model["snapshots"])[-1]
            lines.append(
                "| {model} | {ig} | {folds} | {passed} |".format(
                    model=model["model_id"],
                    ig=_format_float(final.get("information_gain_nats_per_event")),
                    folds=model["positive_development_fold_count"],
                    passed=str(bool(model["passes_g1_ls_as_same_model"])).lower(),
                )
            )
        lines.extend(
            (
                "",
                "## 次级诊断与限制",
                "",
                (
                    "7/30/90/180/365 天 × 0/1/7 天报告延迟只作支持域积分诊断，"
                    "不参与 G1-LS。后续任何全研究区严格召回、Molchan 或报警面积主表"
                    "必须仍以原研究区全部目标为分母，unsupported 区目标计为未覆盖。"
                ),
                "",
            )
        )
    sensitivity_rows = cast(tuple[dict[str, object], ...], science["etas_parent_sensitivity"])
    lines.extend(
        (
            "## ETAS unsupported 父历史敏感性",
            "",
            (
                "这是条件父历史敏感性，不是把局部高 Mc 的空间排除传播到其他格。"
                "两列均相对同一均匀基线计算。"
            ),
            "",
            "| 快照 | 状态 | 纳入合格 unsupported 父历史 IG | 完全排除 IG | 差值（排除−纳入） |",
            "| --- | --- | ---: | ---: | ---: |",
        )
    )
    for row in sensitivity_rows:
        primary = row.get("primary_includes_eligible_unsupported_parents")
        excluded = row.get("exclude_all_unsupported_parents")
        primary_ig = (
            cast(dict[str, object], primary).get("information_gain_nats_per_event")
            if isinstance(primary, dict)
            else None
        )
        excluded_ig = (
            cast(dict[str, object], excluded).get("information_gain_nats_per_event")
            if isinstance(excluded, dict)
            else None
        )
        lines.append(
            "| {snapshot} | {status} | {primary} | {excluded} | {difference} |".format(
                snapshot=row["snapshot_id"],
                status=row["status"],
                primary=_format_float(primary_ig),
                excluded=_format_float(excluded_ig),
                difference=_format_float(
                    row.get("information_gain_difference_exclude_minus_primary")
                ),
            )
        )
    lines.append("")
    lines.extend(
        (
            "## 审计身份",
            "",
            f"- 授权 ID：`{registry['authorization_id']}`",
            f"- 执行封印：`{registry['execution_seal_id']}`",
            f"- 代码提交：`{registry['code_commit']}`",
            f"- 评分代码标签：`{registry['scoring_code_tag']}`",
            f"- 预留物理核心：至少 {registry['reserve_physical_cores']} 个",
            f"- 结果标签（提交结果后创建）：`{_RESULTS_TAG}`",
            "",
        )
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def publish_local_support_deliverables(
    project_root: Path,
    config: BackgroundConfig,
    runtime: LocalSupportRuntime,
    outcome: LocalSupportStage2R1Outcome,
    authorized_execution: AuthorizedExecution,
    *,
    runner: GitCommandRunner = subprocess_git_runner,
) -> PublishedLocalSupportDeliverables:
    """Deterministically rebuild, verify, and publish one sealed 2R-1 outcome."""

    require_background_scoring_authorized(config, authorized_execution)
    seal: ExecutionSeal = authorized_execution.execution_seal
    require_execution_seal_unchanged(project_root, config, seal, runner=runner)
    # Publication deliberately accepts no caller-provided bundle or registry payload.
    # Rebuilding here prevents an authorized caller from publishing a truncated ledger
    # or a figure assembled from a different runtime/outcome pair.
    deliverables = build_local_support_deliverables(
        config,
        runtime,
        outcome,
        authorized_execution,
    )
    ledger_summary = cast(dict[str, object], deliverables.registry_science["score_ledger"])
    raw_score_ids = ledger_summary["score_ids"]
    if not isinstance(raw_score_ids, list | tuple) or not all(
        isinstance(item, str) for item in raw_score_ids
    ):
        raise TypeError("local-support registry Score IDs must be a string sequence")
    validate_registry_score_references(outcome.score_ledger, raw_score_ids)
    identity = seal.repository
    uv_hash = config.inputs.environment_lock_sha256
    processed = publish_processed_bundle(
        project_root,
        config,
        identity,
        deliverables.processed,
        uv_lock_sha256=uv_hash,
    )
    model = publish_model_bundle(
        project_root,
        config,
        identity,
        deliverables.model,
        uv_lock_sha256=uv_hash,
        authorized_execution=authorized_execution,
    )
    backtest = publish_backtest_bundle(
        project_root,
        config,
        identity,
        deliverables.backtest,
        uv_lock_sha256=uv_hash,
        authorized_execution=authorized_execution,
    )
    experiment = publish_experiment_bundle(
        project_root,
        config,
        identity,
        deliverables.experiment,
        uv_lock_sha256=uv_hash,
        authorized_execution=authorized_execution,
    )
    require_execution_seal_unchanged(project_root, config, seal, runner=runner)

    publications = (processed, model, backtest, experiment)
    if tuple(item.bundle_kind for item in publications) != _BUNDLE_ORDER:
        raise ValueError("local-support bundle order changed")
    authorization = authorized_execution.scoring_authorization
    figure_sha256 = hashlib.sha256(deliverables.results_svg).hexdigest()
    support_manifest_sha256 = getattr(config.inputs, "support_manifest_sha256", None)
    if not isinstance(support_manifest_sha256, str):
        raise ValueError("local-support publication requires a support manifest SHA-256")
    registry: dict[str, object] = {
        "schema_version": "1.0.0",
        "protocol_version": "0.2.1",
        "gate_name": "G1-LS",
        "protocol_sha256": outcome.protocol_sha256,
        "protocol_freeze_tag": config.freeze_tag,
        "scoring_code_tag": _SCORING_CODE_TAG,
        "scoring_code_tag_object": authorization.scoring_code_tag_object,
        "code_commit": seal.repository.code_commit,
        "execution_seal_id": seal.seal_id,
        "authorization_id": authorized_execution.authorization_id,
        "authorized_public_repository": authorization.remote_repository,
        "input_hashes": seal.input_hash_mapping(),
        "support_manifest_id": runtime.manifest_id,
        "support_manifest_sha256": support_manifest_sha256,
        "bundles": tuple(_bundle_reference(item) for item in publications),
        "science": deliverables.registry_science,
        "results_figure": {
            "path": _PUBLIC_RESULTS_FIGURE,
            "sha256": figure_sha256,
            "contains_geometry": False,
            "contains_coordinates": False,
        },
        "locked_test": {
            "run": False,
            "score_ids": (),
            "artifact_ids": (),
            "target_count": None,
            "result": None,
        },
        "stage3_allowed": outcome.stage3_allowed,
        "reserve_physical_cores": 2,
    }
    g1 = cast(dict[str, object], deliverables.registry_science["g1_ls_and_selection"])["g1_ls"]
    g1_mapping = cast(dict[str, object], g1)
    if bool(g1_mapping.get("passed", False)) != outcome.g1_ls_passed:
        raise ValueError("registry G1-LS conclusion differs from the immutable outcome")
    expected_g1_status = (
        "not_evaluable"
        if isinstance(outcome, LocalSupportCredibleNegative)
        else ("passed" if outcome.g1_ls_passed else "failed")
    )
    if g1_mapping.get("status") != expected_g1_status:
        raise ValueError("registry G1-LS status differs from the immutable outcome")
    if bool(registry["stage3_allowed"]) != outcome.g1_ls_passed:
        raise ValueError("registry stage3 allowance must equal G1-LS")
    _assert_public_payload(registry, location="registry")
    registry_bytes = _registry_bytes(registry)
    report_bytes = _report_bytes(registry)

    require_execution_seal_unchanged(project_root, config, seal, runner=runner)
    figure_publication = publish_fixed_project_file(
        project_root,
        _PUBLIC_RESULTS_FIGURE,
        deliverables.results_svg,
    )
    registry_publication = publish_fixed_project_file(
        project_root,
        config.outputs.registry,
        registry_bytes,
    )
    report_publication = publish_fixed_project_file(
        project_root,
        config.outputs.report,
        report_bytes,
    )
    return PublishedLocalSupportDeliverables(
        processed=processed,
        model=model,
        backtest=backtest,
        experiment=experiment,
        registry=registry_publication,
        report=report_publication,
        results_figure=figure_publication,
        registry_payload=registry,
    )


__all__ = [
    "LocalSupportCredibleNegative",
    "LocalSupportDeliverables",
    "LocalSupportStage2R1Outcome",
    "PublishedLocalSupportDeliverables",
    "build_local_support_deliverables",
    "credible_negative_from_error",
    "publish_local_support_deliverables",
]
