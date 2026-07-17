"""Target-blind stage-4 scoring plan and deterministic orchestration contracts.

Building a plan is always legal before the scoring-code freeze because it reads
only the accepted protocol and public manifests.  The formal backtest entry point
is added behind the one-shot authorization capability; no target loader exists in
this module.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import PurePosixPath
from typing import Literal, cast

from seismoflux.anomaly_increment.compute import Stage4ComputePlan, build_compute_plan
from seismoflux.anomaly_increment.config import Stage4ProtocolBundle
from seismoflux.data.common import canonical_json_bytes

Partition = Literal["development", "validation"]
ScopeRole = Literal["development_joint_macro", "formal_validation_once"]
ModelVariant = Literal[
    "background_no_increment",
    "coverage_only",
    "snapshot",
    "dynamic",
]
MagnitudeBin = Literal["M5_6", "M6_plus"]

_EXPOSURE_RE = re.compile(
    r"^(?P<partition>development|validation)-h(?P<horizon>[0-9]{3})-"
    r"(?P<issue>[0-9]{4}-[0-9]{2}-[0-9]{2})$"
)


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _sequence(value: object, *, label: str) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise TypeError(f"{label} must be a sequence")
    return tuple(value)


def _strings(value: object, *, label: str) -> tuple[str, ...]:
    result = _sequence(value, label=label)
    if not all(isinstance(item, str) and item for item in result):
        raise TypeError(f"{label} must contain non-empty strings")
    return cast(tuple[str, ...], result)


def _positive_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True, order=True)
class ExposurePlan:
    """One frozen ``(T,T+h]`` exposure identity."""

    partition: Partition
    horizon_days: int
    issue_date_local: date

    def __post_init__(self) -> None:
        if self.partition not in {"development", "validation"}:
            raise ValueError("stage-4 exposure partition changed")
        if self.horizon_days not in {7, 30, 90, 180, 365}:
            raise ValueError("stage-4 exposure horizon changed")

    @classmethod
    def parse(cls, value: str) -> ExposurePlan:
        match = _EXPOSURE_RE.fullmatch(value)
        if match is None:
            raise ValueError(f"invalid stage-4 exposure identity: {value}")
        return cls(
            partition=cast(Partition, match.group("partition")),
            horizon_days=int(match.group("horizon")),
            issue_date_local=date.fromisoformat(match.group("issue")),
        )

    @property
    def identifier(self) -> str:
        return f"{self.partition}-h{self.horizon_days:03d}-{self.issue_date_local.isoformat()}"

    @property
    def target_start_exclusive_local(self) -> date:
        return self.issue_date_local

    @property
    def target_end_inclusive_local(self) -> date:
        return self.issue_date_local + timedelta(days=self.horizon_days)


@dataclass(frozen=True, slots=True)
class HorizonAssessmentPlan:
    horizon_days: int
    exposures: tuple[ExposurePlan, ...]

    def __post_init__(self) -> None:
        if self.horizon_days not in {7, 30, 90, 180, 365}:
            raise ValueError("assessment horizon changed")
        if not self.exposures:
            raise ValueError("assessment horizon must contain exposures")
        if any(item.horizon_days != self.horizon_days for item in self.exposures):
            raise ValueError("assessment exposure has the wrong horizon")
        if len({item.identifier for item in self.exposures}) != len(self.exposures):
            raise ValueError("assessment exposures are duplicated")


@dataclass(frozen=True, slots=True)
class TimePermutationPools:
    fit_issue_ids: tuple[str, ...]
    assessment_issue_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.fit_issue_ids or not self.assessment_issue_ids:
            raise ValueError("time-placebo pools must be non-empty")
        if set(self.fit_issue_ids) & set(self.assessment_issue_ids):
            raise ValueError("time-placebo fit and assessment pools overlap")
        if len(set(self.fit_issue_ids)) != len(self.fit_issue_ids) or len(
            set(self.assessment_issue_ids)
        ) != len(self.assessment_issue_ids):
            raise ValueError("time-placebo pools contain duplicate issues")


@dataclass(frozen=True, slots=True)
class FitScopePlan:
    """One shared 7-day fit and all assessment horizons using that fit."""

    fit_scope_id: str
    role: ScopeRole
    fit_exposures_7d: tuple[ExposurePlan, ...]
    assessments: tuple[HorizonAssessmentPlan, ...]
    time_permutation_pools: TimePermutationPools
    validation_refit_forbidden: bool

    def __post_init__(self) -> None:
        if not self.fit_scope_id:
            raise ValueError("stage-4 fit scope id must not be empty")
        if not self.fit_exposures_7d or any(
            item.partition != "development" or item.horizon_days != 7
            for item in self.fit_exposures_7d
        ):
            raise ValueError("stage-4 model fitting is restricted to development 7-day exposures")
        if len({item.identifier for item in self.fit_exposures_7d}) != len(self.fit_exposures_7d):
            raise ValueError("fit exposures are duplicated")
        if self.role == "development_joint_macro":
            if tuple(item.horizon_days for item in self.assessments) != (7, 30, 90):
                raise ValueError("development joint macro scope must use 7/30/90 days")
            if self.validation_refit_forbidden:
                raise ValueError("development scope cannot claim validation-refit status")
        elif self.role == "formal_validation_once":
            if tuple(item.horizon_days for item in self.assessments) != (7, 30, 90, 180, 365):
                raise ValueError("formal validation scope must contain all five horizons")
            if not self.validation_refit_forbidden:
                raise ValueError("formal validation refitting must remain forbidden")
        else:
            raise ValueError("unknown stage-4 fit-scope role")
        latest_fit_end = max(item.target_end_inclusive_local for item in self.fit_exposures_7d)
        earliest_assessment_start = min(
            item.target_start_exclusive_local
            for assessment in self.assessments
            for item in assessment.exposures
        )
        if latest_fit_end >= earliest_assessment_start:
            raise ValueError("fit target time overlaps the assessment target band")


@dataclass(frozen=True, slots=True)
class Stage4PublicationPlan:
    public_registry: str
    public_report: str
    public_static_svg: str
    local_interactive_html: str
    public_model_card: str
    bundle_root: str
    local_convergence_audit: str
    local_spatial_static: str
    local_spatial_interactive: str

    def __post_init__(self) -> None:
        public_paths = (
            self.public_registry,
            self.public_report,
            self.public_static_svg,
            self.public_model_card,
        )
        all_paths = (
            *public_paths,
            self.bundle_root,
            self.local_interactive_html,
            self.local_convergence_audit,
            self.local_spatial_static,
            self.local_spatial_interactive,
        )
        if any(
            not value
            or value.startswith(("/", "\\"))
            or "\\" in value
            or ".." in PurePosixPath(value).parts
            for value in all_paths
        ):
            raise ValueError("stage-4 publication paths must be non-empty and relative")
        if not self.bundle_root.startswith("models/registry/"):
            raise ValueError("stage-4 bundle root must remain in the model registry")
        local_paths = (
            self.local_interactive_html,
            self.local_convergence_audit,
            self.local_spatial_static,
            self.local_spatial_interactive,
        )
        if any(not value.startswith("outputs/visualizations/") for value in local_paths):
            raise ValueError("stage-4 local outputs must remain local visualizations")
        if len(set(all_paths)) != len(all_paths):
            raise ValueError("stage-4 publication paths must be unique")


@dataclass(frozen=True, slots=True)
class Stage4ScoringPlan:
    """Complete score-free execution structure frozen before target access."""

    protocol_design_sha256: str
    random_input_seal_sha256: str
    model_variants: tuple[ModelVariant, ...]
    magnitude_bins: tuple[MagnitudeBin, ...]
    horizons_days: tuple[int, ...]
    primary_horizons_days: tuple[int, ...]
    evidence_insufficient_horizons_days: tuple[int, ...]
    fit_scopes: tuple[FitScopePlan, ...]
    time_permutation_replications: int
    space_permutation_replications: int
    bootstrap_replications: int
    checkpoint_every_replications: int
    heartbeat_seconds: int
    compute: Stage4ComputePlan
    publication: Stage4PublicationPlan
    formal_attempt_count: int = 0
    target_read_count: int = 0
    locked_test_run: bool = False

    def __post_init__(self) -> None:
        if self.model_variants != (
            "background_no_increment",
            "coverage_only",
            "snapshot",
            "dynamic",
        ):
            raise ValueError("stage-4 model variants changed")
        if self.magnitude_bins != ("M5_6", "M6_plus"):
            raise ValueError("stage-4 magnitude bins changed")
        if self.horizons_days != (7, 30, 90, 180, 365):
            raise ValueError("stage-4 horizons changed")
        if self.primary_horizons_days != (7, 30, 90):
            raise ValueError("stage-4 primary horizons changed")
        if self.evidence_insufficient_horizons_days != (180, 365):
            raise ValueError("stage-4 long-horizon evidence status changed")
        if len(self.fit_scopes) != 4:
            raise ValueError("stage-4 requires three development folds and one validation fit")
        if tuple(scope.role for scope in self.fit_scopes) != (
            "development_joint_macro",
            "development_joint_macro",
            "development_joint_macro",
            "formal_validation_once",
        ):
            raise ValueError("stage-4 fit-scope order changed")
        if self.time_permutation_replications != 1000:
            raise ValueError("stage-4 time permutation count changed")
        if self.space_permutation_replications != 1000:
            raise ValueError("stage-4 space permutation count changed")
        if self.bootstrap_replications != 2000:
            raise ValueError("stage-4 bootstrap count changed")
        if self.checkpoint_every_replications != 25 or self.heartbeat_seconds != 30:
            raise ValueError("stage-4 checkpoint or heartbeat policy changed")
        if self.formal_attempt_count != 0 or self.target_read_count != 0:
            raise ValueError("a score-free stage-4 plan cannot contain an attempt or target read")
        if self.locked_test_run:
            raise ValueError("locked testing is forbidden in stage 4")

    def as_dict(self) -> dict[str, object]:
        return {
            "bootstrap_replications": self.bootstrap_replications,
            "checkpoint_every_replications": self.checkpoint_every_replications,
            "compute": {
                "backend": self.compute.backend,
                "effective_workers": self.compute.workers.effective_workers,
                "gpu_equivalence_sha256": self.compute.gpu_equivalence_sha256,
                "gpu_fallback_reason": self.compute.gpu_fallback_reason,
                "reserve_physical_cores": self.compute.workers.reserve_physical_cores,
            },
            "evidence_insufficient_horizons_days": list(self.evidence_insufficient_horizons_days),
            "fit_scopes": [
                {
                    "assessments": {
                        str(item.horizon_days): [exposure.identifier for exposure in item.exposures]
                        for item in scope.assessments
                    },
                    "fit_exposures_7d": [item.identifier for item in scope.fit_exposures_7d],
                    "fit_scope_id": scope.fit_scope_id,
                    "role": scope.role,
                    "time_assessment_pool": list(scope.time_permutation_pools.assessment_issue_ids),
                    "time_fit_pool": list(scope.time_permutation_pools.fit_issue_ids),
                    "validation_refit_forbidden": scope.validation_refit_forbidden,
                }
                for scope in self.fit_scopes
            ],
            "formal_attempt_count": self.formal_attempt_count,
            "heartbeat_seconds": self.heartbeat_seconds,
            "horizons_days": list(self.horizons_days),
            "locked_test_run": self.locked_test_run,
            "magnitude_bins": list(self.magnitude_bins),
            "model_variants": list(self.model_variants),
            "primary_horizons_days": list(self.primary_horizons_days),
            "protocol_design_sha256": self.protocol_design_sha256,
            "publication": {
                "bundle_root": self.publication.bundle_root,
                "local_convergence_audit": self.publication.local_convergence_audit,
                "local_interactive_html": self.publication.local_interactive_html,
                "local_spatial_interactive": self.publication.local_spatial_interactive,
                "local_spatial_static": self.publication.local_spatial_static,
                "public_model_card": self.publication.public_model_card,
                "public_registry": self.publication.public_registry,
                "public_report": self.publication.public_report,
                "public_static_svg": self.publication.public_static_svg,
            },
            "random_input_seal_sha256": self.random_input_seal_sha256,
            "space_permutation_replications": self.space_permutation_replications,
            "target_read_count": self.target_read_count,
            "time_permutation_replications": self.time_permutation_replications,
        }

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.as_dict())).hexdigest()


def _assessments(raw: object, *, role: ScopeRole) -> tuple[HorizonAssessmentPlan, ...]:
    mapping = _mapping(raw, label="assessment_exposure_ids_by_horizon")
    expected = (7, 30, 90) if role == "development_joint_macro" else (7, 30, 90, 180, 365)
    if set(mapping) != {str(value) for value in expected}:
        raise ValueError("assessment horizon keys differ from the frozen scope")
    return tuple(
        HorizonAssessmentPlan(
            horizon_days=horizon,
            exposures=tuple(
                ExposurePlan.parse(value)
                for value in _strings(
                    mapping[str(horizon)],
                    label=f"assessment horizon {horizon}",
                )
            ),
        )
        for horizon in expected
    )


def _fit_scope(raw: object, *, role: ScopeRole) -> FitScopePlan:
    scope = _mapping(raw, label="fit_scope")
    pools = _mapping(
        scope.get("time_permutation_feature_pools"), label="time_permutation_feature_pools"
    )
    if pools.get("pool_crossing_forbidden") is not True:
        raise ValueError("time-placebo pool crossing must remain forbidden")
    return FitScopePlan(
        fit_scope_id=cast(str, scope["fit_scope_id"]),
        role=role,
        fit_exposures_7d=tuple(
            ExposurePlan.parse(value)
            for value in _strings(scope.get("fit_exposure_ids_7d"), label="fit exposures")
        ),
        assessments=_assessments(scope.get("assessment_exposure_ids_by_horizon"), role=role),
        time_permutation_pools=TimePermutationPools(
            fit_issue_ids=_strings(pools.get("fit"), label="time fit pool"),
            assessment_issue_ids=_strings(pools.get("assessment"), label="time assessment pool"),
        ),
        validation_refit_forbidden=bool(scope.get("validation_refit_forbidden", False)),
    )


def build_stage4_scoring_plan(
    protocol: Stage4ProtocolBundle,
    *,
    compute: Stage4ComputePlan | None = None,
) -> Stage4ScoringPlan:
    """Build the complete formal plan without opening the target catalog."""

    raw_folds = _sequence(
        protocol.fold.document.get("joint_macro_rolling_folds"),
        label="joint_macro_rolling_folds",
    )
    if len(raw_folds) != 3:
        raise ValueError("stage-4 requires exactly three joint macro development folds")
    scopes = (
        *(_fit_scope(item, role="development_joint_macro") for item in raw_folds),
        _fit_scope(
            protocol.fold.document.get("formal_validation_fit"),
            role="formal_validation_once",
        ),
    )
    evaluation = _mapping(protocol.protocol.get("evaluation"), label="evaluation")
    permutations = _mapping(evaluation.get("permutations"), label="permutations")
    time_permutation = _mapping(permutations.get("time"), label="permutations.time")
    space_permutation = _mapping(permutations.get("space"), label="permutations.space")
    bootstrap = _mapping(evaluation.get("bootstrap"), label="bootstrap")
    compute_config = _mapping(protocol.protocol.get("compute"), label="compute")
    publication = _mapping(protocol.protocol.get("publication"), label="publication")
    return Stage4ScoringPlan(
        protocol_design_sha256=protocol.design_sha256,
        random_input_seal_sha256=protocol.random_input_seal_sha256,
        model_variants=(
            "background_no_increment",
            "coverage_only",
            "snapshot",
            "dynamic",
        ),
        magnitude_bins=("M5_6", "M6_plus"),
        horizons_days=(7, 30, 90, 180, 365),
        primary_horizons_days=(7, 30, 90),
        evidence_insufficient_horizons_days=(180, 365),
        fit_scopes=scopes,
        time_permutation_replications=_positive_integer(
            time_permutation.get("replications"), label="time permutation replications"
        ),
        space_permutation_replications=_positive_integer(
            space_permutation.get("replications"), label="space permutation replications"
        ),
        bootstrap_replications=_positive_integer(
            bootstrap.get("replications"), label="bootstrap replications"
        ),
        checkpoint_every_replications=_positive_integer(
            compute_config.get("checkpoint_every_permutation_replications"),
            label="checkpoint_every_permutation_replications",
        ),
        heartbeat_seconds=_positive_integer(
            compute_config.get("heartbeat_seconds"), label="heartbeat_seconds"
        ),
        compute=build_compute_plan(protocol) if compute is None else compute,
        publication=Stage4PublicationPlan(
            public_registry=cast(str, publication["required_registry"]),
            public_report=cast(str, publication["required_report"]),
            public_static_svg=cast(str, publication["required_static_svg"]),
            local_interactive_html=cast(str, publication["required_interactive_local"]),
            public_model_card=cast(str, publication["required_model_card"]),
            bundle_root=cast(str, publication["bundle_root"]),
            local_convergence_audit=cast(str, publication["local_convergence_audit"]),
            local_spatial_static=cast(str, publication["local_spatial_static"]),
            local_spatial_interactive=cast(str, publication["local_spatial_interactive"]),
        ),
    )


__all__ = [
    "ExposurePlan",
    "FitScopePlan",
    "HorizonAssessmentPlan",
    "MagnitudeBin",
    "ModelVariant",
    "Stage4PublicationPlan",
    "Stage4ScoringPlan",
    "TimePermutationPools",
    "build_stage4_scoring_plan",
]
