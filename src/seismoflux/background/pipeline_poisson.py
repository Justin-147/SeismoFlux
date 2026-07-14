"""Pure five-snapshot orchestration for uniform and spatial Poisson baselines."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np

from seismoflux.background.artifacts import canonical_json_bytes
from seismoflux.background.catalog import EarthquakeCatalog
from seismoflux.background.completeness import CATALOG_ANCHOR_UTC
from seismoflux.background.config import BackgroundConfig
from seismoflux.background.evidence import (
    EXPECTED_SNAPSHOTS,
    AuditedBackgroundModelEvidence,
    PairedInformationGainEvidence,
    PointProcessScoreEvidence,
)
from seismoflux.background.grid import (
    EqualAreaGridFamily,
    ThreeGridConvergenceGateEvidence,
    diagnose_three_grid_convergence,
)
from seismoflux.background.local_support_runtime import (
    LocalSupportRuntime,
    LocalSupportRuntimeSnapshot,
)
from seismoflux.background.poisson import (
    FROZEN_BANDWIDTHS_KM,
    BandwidthPreScoreGateEvidence,
    BandwidthPreScoreGateItem,
    BandwidthSelection,
    SpatialPoissonModel,
    SpatialQuadrature,
    UniformPoissonModel,
    evaluate_spatial_poisson_family_cell_masses,
    evaluate_spatial_poisson_family_log_densities,
    fit_spatial_poisson_family,
    fit_uniform_poisson,
    select_kde_bandwidth,
)
from seismoflux.background.scoring_authorization import (
    AuthorizedExecution,
    require_background_scoring_authorized,
)
from seismoflux.background.workflow import (
    CompletenessSnapshot,
    ProgressCallback,
    SnapshotDefinition,
    build_snapshot_definitions,
    historical_training_mask,
    physical_target_mask,
)

_UNIFORM_VARIANT = "uniform_poisson/spatial_uniform_v1"
_NORMALIZATION_SUM_TOLERANCE = 1.0e-12
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_LOCAL_SUPPORT_ID_RE = re.compile(r"local-support-[0-9a-f]{16}")

TargetAccessObserver = Callable[[str], None]


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _spatial_variant(bandwidth_km: float) -> str:
    return f"spatial_poisson/gaussian_kde_bw{bandwidth_km:g}km"


class PoissonKDEPipelineError(RuntimeError):
    """Base error for Poisson/KDE orchestration failures."""

    def __init__(
        self,
        message: str,
        *,
        gate_evidence: BandwidthPreScoreGateEvidence | None = None,
        scoreability_gate_evidence: LocalSupportScoreabilityGateEvidence | None = None,
        partial_failure_evidence: LocalSupportPoissonPartialFailureEvidence | None = None,
        fitted_snapshots: tuple[LocalSupportPoissonSnapshotFit, ...] | None = None,
    ) -> None:
        super().__init__(message)
        if partial_failure_evidence is not None and fitted_snapshots is None:
            fitted_snapshots = partial_failure_evidence.snapshots
        if (
            fitted_snapshots is not None
            and tuple(item.definition.snapshot_id for item in fitted_snapshots)
            != EXPECTED_SNAPSHOTS
        ):
            raise ValueError("fitted local snapshots must contain all five frozen snapshots")
        if (
            partial_failure_evidence is not None
            and fitted_snapshots != partial_failure_evidence.snapshots
        ):
            raise ValueError("partial failure and exception fitted snapshots differ")
        self.gate_evidence = gate_evidence
        self.scoreability_gate_evidence = scoreability_gate_evidence
        self.partial_failure_evidence = partial_failure_evidence
        self.fitted_snapshots = fitted_snapshots
        self.scores_started = partial_failure_evidence is not None


PoissonKDEInabilityCode = Literal[
    "all_bandwidths_failed_numerical_gate",
    "zero_target_snapshot",
    "zero_training_events",
]


class PoissonKDEScientificInability(PoissonKDEPipelineError):
    """Expected data-supported inability to complete the frozen Poisson/KDE family."""

    def __init__(
        self,
        reason_code: PoissonKDEInabilityCode,
        message: str,
        *,
        gate_evidence: BandwidthPreScoreGateEvidence | None = None,
        scoreability_gate_evidence: LocalSupportScoreabilityGateEvidence | None = None,
        partial_failure_evidence: LocalSupportPoissonPartialFailureEvidence | None = None,
        fitted_snapshots: tuple[LocalSupportPoissonSnapshotFit, ...] | None = None,
    ) -> None:
        super().__init__(
            message,
            gate_evidence=gate_evidence,
            scoreability_gate_evidence=scoreability_gate_evidence,
            partial_failure_evidence=partial_failure_evidence,
            fitted_snapshots=fitted_snapshots,
        )
        self.reason_code = reason_code


class PoissonKDEInvariantError(PoissonKDEPipelineError):
    """Internal implementation contract failure that must never become a scientific result."""


@dataclass(frozen=True, slots=True)
class SnapshotKDEGateEvidence:
    """Normalization and complete three-grid evidence for one snapshot/candidate."""

    protocol_sha256: str
    snapshot_id: str
    bandwidth_km: float
    normalization_mass: float
    normalization_cell_mass_sum: float
    convergence: ThreeGridConvergenceGateEvidence

    def __post_init__(self) -> None:
        if self.snapshot_id not in EXPECTED_SNAPSHOTS:
            raise ValueError("KDE gate snapshot must be one of the five frozen snapshots")
        if self.bandwidth_km not in FROZEN_BANDWIDTHS_KM:
            raise ValueError("KDE gate bandwidth must be a frozen candidate")
        if not math.isfinite(self.normalization_mass) or self.normalization_mass <= 0.0:
            raise ValueError("KDE normalization mass evidence must be finite and positive")
        if not math.isfinite(self.normalization_cell_mass_sum):
            raise ValueError("KDE normalization cell-mass sum must be finite")

    @property
    def normalization_passed(self) -> bool:
        return math.isclose(
            self.normalization_cell_mass_sum,
            1.0,
            rel_tol=0.0,
            abs_tol=_NORMALIZATION_SUM_TOLERANCE,
        )

    @property
    def passed(self) -> bool:
        return self.normalization_passed and self.convergence.passed

    @property
    def failure_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if not self.normalization_passed:
            reasons.append(
                "12.5km normalization cell mass sum differs from one: "
                f"{self.normalization_cell_mass_sum:.17g}"
            )
        for diagnostics in self.convergence.comparisons:
            if not diagnostics.passed:
                reasons.append(
                    f"{diagnostics.coarse_cell_size_km:g}->"
                    f"{diagnostics.fine_cell_size_km:g}km convergence failed: "
                    "relative_expected_count="
                    f"{diagnostics.relative_expected_count_difference:.17g}, "
                    f"density_l1={diagnostics.density_l1_difference:.17g}"
                )
        return tuple(reasons)

    @property
    def numerical_evidence_id(self) -> str:
        return _canonical_sha256(
            {
                "protocol_sha256": self.protocol_sha256,
                "snapshot_id": self.snapshot_id,
                "bandwidth_km": self.bandwidth_km,
                "normalization_mass": self.normalization_mass,
                "normalization_cell_mass_sum": self.normalization_cell_mass_sum,
                "comparisons": tuple(
                    {
                        "coarse_cell_size_km": item.coarse_cell_size_km,
                        "fine_cell_size_km": item.fine_cell_size_km,
                        "coarse_total": item.coarse_total,
                        "fine_total": item.fine_total,
                        "relative_expected_count_difference": (
                            item.relative_expected_count_difference
                        ),
                        "density_l1_difference": item.density_l1_difference,
                        "passed": item.passed,
                    }
                    for item in self.convergence.comparisons
                ),
            }
        )


@dataclass(frozen=True, slots=True)
class PoissonSnapshotFit:
    """One immutable snapshot fit retaining the complete KDE family for ETAS."""

    definition: SnapshotDefinition
    selected_mc: float
    training_event_ids: tuple[str, ...]
    training_event_count: int
    training_duration_days: float
    rate_per_day: float
    training_evidence_id: str
    target_event_ids: tuple[str, ...]
    uniform_model: UniformPoissonModel
    kde_family: tuple[tuple[float, SpatialPoissonModel], ...]
    grid_gate_evidence: tuple[SnapshotKDEGateEvidence, ...]

    def __post_init__(self) -> None:
        if self.definition.snapshot_id not in EXPECTED_SNAPSHOTS:
            raise ValueError("snapshot fit uses an unknown frozen snapshot")
        if self.training_event_count <= 0 or self.training_event_count != len(
            self.training_event_ids
        ):
            raise ValueError("snapshot training count must be positive and match event IDs")
        if not math.isfinite(self.training_duration_days) or self.training_duration_days <= 0.0:
            raise ValueError("snapshot training duration must be finite and positive")
        if not math.isfinite(self.rate_per_day) or self.rate_per_day <= 0.0:
            raise ValueError("snapshot rate must be finite and positive")
        if self.uniform_model.rate_per_day != self.rate_per_day:
            raise ValueError("uniform and retained snapshot rates differ")
        if tuple(value for value, _ in self.kde_family) != FROZEN_BANDWIDTHS_KM:
            raise ValueError("snapshot KDE family must contain all five frozen bandwidths")
        if tuple(item.bandwidth_km for item in self.grid_gate_evidence) != (FROZEN_BANDWIDTHS_KM):
            raise ValueError("snapshot grid evidence must contain all five bandwidths")
        if any(item.snapshot_id != self.definition.snapshot_id for item in self.grid_gate_evidence):
            raise ValueError("snapshot grid evidence IDs do not match their fit")
        if any(model.rate_per_day != self.rate_per_day for _, model in self.kde_family):
            raise ValueError("KDE family rates differ from the retained snapshot rate")

    def kde_model(self, bandwidth_km: float) -> SpatialPoissonModel:
        requested = float(bandwidth_km)
        for bandwidth, model in self.kde_family:
            if bandwidth == requested:
                return model
        raise KeyError(f"snapshot has no {requested:g} km KDE model")

    def gate_for(self, bandwidth_km: float) -> SnapshotKDEGateEvidence:
        requested = float(bandwidth_km)
        for evidence in self.grid_gate_evidence:
            if evidence.bandwidth_km == requested:
                return evidence
        raise KeyError(f"snapshot has no {requested:g} km KDE gate evidence")


@dataclass(frozen=True, slots=True)
class BandwidthFoldInformationGainAudit:
    """Four paired continuous-time scores for one gate-passing bandwidth."""

    bandwidth_km: float
    development_folds: tuple[PairedInformationGainEvidence, ...]

    def __post_init__(self) -> None:
        if self.bandwidth_km not in FROZEN_BANDWIDTHS_KM:
            raise ValueError("bandwidth fold audit must use a frozen candidate")
        if (
            tuple(item.candidate.snapshot_id for item in self.development_folds)
            != (EXPECTED_SNAPSHOTS[:4])
        ):
            raise ValueError("bandwidth fold audit must contain the four development folds")
        if any(item.information_gain_per_event is None for item in self.development_folds):
            raise ValueError("zero-target fold information gain is undefined")


@dataclass(frozen=True, slots=True)
class PoissonKDEPipelineResult:
    """Read-only scientific result for the complete uniform/KDE snapshot workflow."""

    protocol_sha256: str
    snapshots: tuple[PoissonSnapshotFit, ...]
    pre_score_gate_evidence: BandwidthPreScoreGateEvidence
    bandwidth_fold_audits: tuple[BandwidthFoldInformationGainAudit, ...]
    bandwidth_selection: BandwidthSelection
    uniform_evidence: AuditedBackgroundModelEvidence
    spatial_evidence: AuditedBackgroundModelEvidence

    def __post_init__(self) -> None:
        if tuple(item.definition.snapshot_id for item in self.snapshots) != EXPECTED_SNAPSHOTS:
            raise ValueError("pipeline result must retain all five snapshots in frozen order")
        if self.bandwidth_selection.pre_score_gate_evidence != self.pre_score_gate_evidence:
            raise ValueError("pipeline selection does not preserve its pre-score gate evidence")
        if self.uniform_evidence.model_id != "uniform_poisson":
            raise ValueError("pipeline uniform evidence has the wrong model family")
        if self.spatial_evidence.model_id != "spatial_poisson":
            raise ValueError("pipeline spatial evidence has the wrong model family")

    @property
    def selected_bandwidth_km(self) -> float:
        return self.bandwidth_selection.selected_bandwidth_km

    def snapshot(self, snapshot_id: str) -> PoissonSnapshotFit:
        for snapshot in self.snapshots:
            if snapshot.definition.snapshot_id == snapshot_id:
                return snapshot
        raise KeyError(f"pipeline has no snapshot {snapshot_id!r}")

    def selected_kde_model(self, snapshot_id: str) -> SpatialPoissonModel:
        return self.snapshot(snapshot_id).kde_model(self.selected_bandwidth_km)


@dataclass(frozen=True, slots=True)
class LocalSupportSnapshotKDEGateEvidence:
    """One local-support KDE candidate's complete pre-score numerical evidence."""

    protocol_sha256: str
    snapshot_id: str
    bandwidth_km: float
    support_id: str
    supported_area_km2: float
    compensator_domain_id: str
    authorization_id: str
    normalization_mass: float
    normalization_cell_mass_sum: float
    convergence: ThreeGridConvergenceGateEvidence

    def __post_init__(self) -> None:
        if _SHA256_RE.fullmatch(self.protocol_sha256) is None:
            raise ValueError("protocol_sha256 must be a lowercase SHA-256")
        if self.snapshot_id not in EXPECTED_SNAPSHOTS:
            raise ValueError("local KDE gate snapshot must be one of the frozen snapshots")
        if self.bandwidth_km not in FROZEN_BANDWIDTHS_KM:
            raise ValueError("local KDE gate bandwidth must be a frozen candidate")
        if _LOCAL_SUPPORT_ID_RE.fullmatch(self.support_id) is None:
            raise ValueError("local KDE gate support_id is malformed")
        if not math.isfinite(self.supported_area_km2) or self.supported_area_km2 <= 0.0:
            raise ValueError("local KDE gate supported area must be finite and positive")
        for name, value in (
            ("compensator_domain_id", self.compensator_domain_id),
            ("authorization_id", self.authorization_id),
        ):
            if _SHA256_RE.fullmatch(value) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256")
        if not math.isfinite(self.normalization_mass) or self.normalization_mass <= 0.0:
            raise ValueError("KDE normalization mass evidence must be finite and positive")
        if not math.isfinite(self.normalization_cell_mass_sum):
            raise ValueError("KDE normalization cell-mass sum must be finite")

    @property
    def normalization_passed(self) -> bool:
        return math.isclose(
            self.normalization_cell_mass_sum,
            1.0,
            rel_tol=0.0,
            abs_tol=_NORMALIZATION_SUM_TOLERANCE,
        )

    @property
    def passed(self) -> bool:
        return self.normalization_passed and self.convergence.passed

    @property
    def failure_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if not self.normalization_passed:
            reasons.append(
                "12.5km normalization cell mass sum differs from one: "
                f"{self.normalization_cell_mass_sum:.17g}"
            )
        for diagnostics in self.convergence.comparisons:
            if not diagnostics.passed:
                reasons.append(
                    f"{diagnostics.coarse_cell_size_km:g}->"
                    f"{diagnostics.fine_cell_size_km:g}km convergence failed: "
                    "relative_expected_count="
                    f"{diagnostics.relative_expected_count_difference:.17g}, "
                    f"density_l1={diagnostics.density_l1_difference:.17g}"
                )
        return tuple(reasons)

    @property
    def numerical_evidence_id(self) -> str:
        return _canonical_sha256(
            {
                "protocol_sha256": self.protocol_sha256,
                "snapshot_id": self.snapshot_id,
                "bandwidth_km": self.bandwidth_km,
                "support_id": self.support_id,
                "supported_area_km2": self.supported_area_km2,
                "compensator_domain_id": self.compensator_domain_id,
                "authorization_id": self.authorization_id,
                "normalization_mass": self.normalization_mass,
                "normalization_cell_mass_sum": self.normalization_cell_mass_sum,
                "comparisons": tuple(
                    {
                        "coarse_cell_size_km": item.coarse_cell_size_km,
                        "fine_cell_size_km": item.fine_cell_size_km,
                        "coarse_total": item.coarse_total,
                        "fine_total": item.fine_total,
                        "relative_expected_count_difference": (
                            item.relative_expected_count_difference
                        ),
                        "density_l1_difference": item.density_l1_difference,
                        "passed": item.passed,
                    }
                    for item in self.convergence.comparisons
                ),
            }
        )


@dataclass(frozen=True, slots=True)
class LocalSupportSnapshotScoreabilityEvidence:
    """Identity-free target/training counts for one frozen support snapshot."""

    protocol_sha256: str
    snapshot_id: str
    support_id: str
    selected_mc: float
    supported_area_km2: float
    compensator_domain_id: str
    authorization_id: str
    training_duration_days: float
    training_event_count: int
    target_event_count: int | None

    def __post_init__(self) -> None:
        if _SHA256_RE.fullmatch(self.protocol_sha256) is None:
            raise ValueError("protocol_sha256 must be a lowercase SHA-256")
        if self.snapshot_id not in EXPECTED_SNAPSHOTS:
            raise ValueError("scoreability snapshot must be one of the frozen snapshots")
        if _LOCAL_SUPPORT_ID_RE.fullmatch(self.support_id) is None:
            raise ValueError("scoreability support_id is malformed")
        if not math.isfinite(self.selected_mc) or self.selected_mc <= 0.0:
            raise ValueError("scoreability common Mc must be finite and positive")
        if not math.isfinite(self.supported_area_km2) or self.supported_area_km2 <= 0.0:
            raise ValueError("scoreability supported area must be finite and positive")
        for name, value in (
            ("compensator_domain_id", self.compensator_domain_id),
            ("authorization_id", self.authorization_id),
        ):
            if _SHA256_RE.fullmatch(value) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256")
        if not math.isfinite(self.training_duration_days) or self.training_duration_days <= 0.0:
            raise ValueError("scoreability training duration must be finite and positive")
        if self.training_event_count < 0 or (
            self.target_event_count is not None and self.target_event_count < 0
        ):
            raise ValueError("scoreability event counts cannot be negative")
        if self.snapshot_id == "final_validation" and self.target_event_count is not None:
            raise ValueError("pre-selection scoreability cannot inspect final targets")
        if self.snapshot_id != "final_validation" and self.target_event_count is None:
            raise ValueError("development scoreability must count assessment targets")

    @property
    def passed(self) -> bool:
        return self.training_event_count > 0 and (
            self.target_event_count is None or self.target_event_count > 0
        )

    @property
    def failure_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if self.training_event_count == 0:
            reasons.append("zero eligible supported training events")
        if self.target_event_count == 0:
            reasons.append("zero eligible supported assessment targets")
        return tuple(reasons)

    @property
    def evidence_id(self) -> str:
        return _canonical_sha256(
            {
                "protocol_sha256": self.protocol_sha256,
                "snapshot_id": self.snapshot_id,
                "support_id": self.support_id,
                "selected_mc": self.selected_mc,
                "supported_area_km2": self.supported_area_km2,
                "compensator_domain_id": self.compensator_domain_id,
                "authorization_id": self.authorization_id,
                "training_duration_days": self.training_duration_days,
                "training_event_count": self.training_event_count,
                "target_event_count": self.target_event_count,
            }
        )


@dataclass(frozen=True, slots=True)
class LocalSupportScoreabilityGateEvidence:
    """Five training counts and four identity-free development target counts."""

    snapshots: tuple[LocalSupportSnapshotScoreabilityEvidence, ...]

    def __post_init__(self) -> None:
        if tuple(item.snapshot_id for item in self.snapshots) != EXPECTED_SNAPSHOTS:
            raise ValueError("scoreability gate must contain all frozen snapshots in order")
        if len({item.protocol_sha256 for item in self.snapshots}) != 1:
            raise ValueError("scoreability snapshots must use one protocol")
        if len({item.authorization_id for item in self.snapshots}) != 1:
            raise ValueError("scoreability snapshots must use one authorization")

    @property
    def passed(self) -> bool:
        return all(item.passed for item in self.snapshots)

    @property
    def zero_training_snapshots(self) -> tuple[str, ...]:
        return tuple(item.snapshot_id for item in self.snapshots if item.training_event_count == 0)

    @property
    def zero_target_snapshots(self) -> tuple[str, ...]:
        return tuple(item.snapshot_id for item in self.snapshots if item.target_event_count == 0)

    @property
    def evidence_id(self) -> str:
        return _canonical_sha256(
            {
                "snapshot_evidence_ids": tuple(item.evidence_id for item in self.snapshots),
            }
        )

    def snapshot(self, snapshot_id: str) -> LocalSupportSnapshotScoreabilityEvidence:
        for item in self.snapshots:
            if item.snapshot_id == snapshot_id:
                return item
        raise KeyError(f"scoreability gate has no snapshot {snapshot_id!r}")


@dataclass(frozen=True, slots=True)
class LocalSupportPoissonSnapshotFit:
    """A target-free local-support fit retaining every frozen KDE candidate."""

    definition: SnapshotDefinition
    support_id: str
    selected_mc: float
    supported_area_km2: float
    compensator_domain_id: str
    authorization_id: str
    training_event_ids: tuple[str, ...]
    training_event_count: int
    training_duration_days: float
    rate_per_day: float
    training_evidence_id: str
    uniform_model: UniformPoissonModel
    kde_family: tuple[tuple[float, SpatialPoissonModel], ...]
    grid_gate_evidence: tuple[LocalSupportSnapshotKDEGateEvidence, ...]

    def __post_init__(self) -> None:
        if self.definition.snapshot_id not in EXPECTED_SNAPSHOTS:
            raise ValueError("local snapshot fit uses an unknown frozen snapshot")
        if _LOCAL_SUPPORT_ID_RE.fullmatch(self.support_id) is None:
            raise ValueError("local snapshot fit support_id is malformed")
        if not math.isfinite(self.selected_mc) or self.selected_mc <= 0.0:
            raise ValueError("local snapshot common Mc must be finite and positive")
        if not math.isfinite(self.supported_area_km2) or self.supported_area_km2 <= 0.0:
            raise ValueError("local snapshot supported area must be finite and positive")
        for name, value in (
            ("compensator_domain_id", self.compensator_domain_id),
            ("authorization_id", self.authorization_id),
            ("training_evidence_id", self.training_evidence_id),
        ):
            if _SHA256_RE.fullmatch(value) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256")
        if self.training_event_count <= 0 or self.training_event_count != len(
            self.training_event_ids
        ):
            raise ValueError("local snapshot training count must be positive and match event IDs")
        if not math.isfinite(self.training_duration_days) or self.training_duration_days <= 0.0:
            raise ValueError("local snapshot training duration must be finite and positive")
        if not math.isfinite(self.rate_per_day) or self.rate_per_day <= 0.0:
            raise ValueError("local snapshot rate must be finite and positive")
        if self.uniform_model.rate_per_day != self.rate_per_day:
            raise ValueError("uniform and retained local snapshot rates differ")
        if self.uniform_model.study_area_km2 != self.supported_area_km2:
            raise ValueError("uniform model area differs from the retained local support")
        if tuple(value for value, _ in self.kde_family) != FROZEN_BANDWIDTHS_KM:
            raise ValueError("local snapshot KDE family must contain all frozen bandwidths")
        if tuple(item.bandwidth_km for item in self.grid_gate_evidence) != FROZEN_BANDWIDTHS_KM:
            raise ValueError("local snapshot grid evidence must contain all frozen bandwidths")
        expected_binding = (
            self.definition.snapshot_id,
            self.support_id,
            self.supported_area_km2,
            self.compensator_domain_id,
            self.authorization_id,
        )
        if any(
            (
                item.snapshot_id,
                item.support_id,
                item.supported_area_km2,
                item.compensator_domain_id,
                item.authorization_id,
            )
            != expected_binding
            for item in self.grid_gate_evidence
        ):
            raise ValueError("local snapshot grid evidence uses another support/domain binding")
        if any(model.rate_per_day != self.rate_per_day for _, model in self.kde_family):
            raise ValueError("local KDE family rates differ from the retained snapshot rate")

    def kde_model(self, bandwidth_km: float) -> SpatialPoissonModel:
        requested = float(bandwidth_km)
        for bandwidth, model in self.kde_family:
            if bandwidth == requested:
                return model
        raise KeyError(f"local snapshot has no {requested:g} km KDE model")

    def gate_for(self, bandwidth_km: float) -> LocalSupportSnapshotKDEGateEvidence:
        requested = float(bandwidth_km)
        for evidence in self.grid_gate_evidence:
            if evidence.bandwidth_km == requested:
                return evidence
        raise KeyError(f"local snapshot has no {requested:g} km KDE gate evidence")


@dataclass(frozen=True, slots=True)
class LocalSupportPoissonKDEPipelineResult:
    """Audited G1-LS Uniform/KDE result including every development candidate score."""

    protocol_sha256: str
    snapshots: tuple[LocalSupportPoissonSnapshotFit, ...]
    scoreability_gate_evidence: LocalSupportScoreabilityGateEvidence
    pre_score_gate_evidence: BandwidthPreScoreGateEvidence
    bandwidth_fold_audits: tuple[BandwidthFoldInformationGainAudit, ...]
    bandwidth_selection: BandwidthSelection
    uniform_evidence: AuditedBackgroundModelEvidence
    spatial_evidence: AuditedBackgroundModelEvidence

    def __post_init__(self) -> None:
        if tuple(item.definition.snapshot_id for item in self.snapshots) != EXPECTED_SNAPSHOTS:
            raise ValueError("local pipeline result must retain five target-free snapshot fits")
        if not self.scoreability_gate_evidence.passed:
            raise ValueError("successful local pipeline retained a failed scoreability gate")
        if tuple(
            (item.snapshot_id, item.support_id)
            for item in self.scoreability_gate_evidence.snapshots
        ) != tuple((item.definition.snapshot_id, item.support_id) for item in self.snapshots):
            raise ValueError("local scoreability gate uses another snapshot/support binding")
        if self.bandwidth_selection.pre_score_gate_evidence != self.pre_score_gate_evidence:
            raise ValueError("local pipeline selection lost its pre-score gate evidence")
        if self.uniform_evidence.model_id != "uniform_poisson":
            raise ValueError("local pipeline uniform evidence has the wrong model family")
        if self.spatial_evidence.model_id != "spatial_poisson":
            raise ValueError("local pipeline spatial evidence has the wrong model family")
        passed = self.pre_score_gate_evidence.passed_bandwidths_km
        if tuple(item.bandwidth_km for item in self.bandwidth_fold_audits) != passed:
            raise ValueError("local pipeline must audit every gate-passing development candidate")

    @property
    def selected_bandwidth_km(self) -> float:
        return self.bandwidth_selection.selected_bandwidth_km

    def snapshot(self, snapshot_id: str) -> LocalSupportPoissonSnapshotFit:
        for snapshot in self.snapshots:
            if snapshot.definition.snapshot_id == snapshot_id:
                return snapshot
        raise KeyError(f"local pipeline has no snapshot {snapshot_id!r}")

    def selected_kde_model(self, snapshot_id: str) -> SpatialPoissonModel:
        return self.snapshot(snapshot_id).kde_model(self.selected_bandwidth_km)


@dataclass(frozen=True, slots=True)
class LocalSupportPoissonPartialFailureEvidence:
    """Completed development attempts retained when final validation cannot score."""

    protocol_sha256: str
    failed_snapshot_id: Literal["final_validation"]
    snapshots: tuple[LocalSupportPoissonSnapshotFit, ...]
    scoreability_gate_evidence: LocalSupportScoreabilityGateEvidence
    pre_score_gate_evidence: BandwidthPreScoreGateEvidence
    bandwidth_fold_audits: tuple[BandwidthFoldInformationGainAudit, ...]
    bandwidth_selection: BandwidthSelection
    development_uniform_scores: tuple[PointProcessScoreEvidence, ...]

    def __post_init__(self) -> None:
        if _SHA256_RE.fullmatch(self.protocol_sha256) is None:
            raise ValueError("partial failure protocol_sha256 must be a lowercase SHA-256")
        if self.failed_snapshot_id != "final_validation":
            raise ValueError("local Poisson partial failure must concern final validation")
        if tuple(item.definition.snapshot_id for item in self.snapshots) != EXPECTED_SNAPSHOTS:
            raise ValueError("partial failure must retain all five target-free fits")
        if not self.scoreability_gate_evidence.passed:
            raise ValueError("partial failure retained a failed pre-selection scoreability gate")
        if tuple(
            (item.snapshot_id, item.support_id)
            for item in self.scoreability_gate_evidence.snapshots
        ) != tuple((item.definition.snapshot_id, item.support_id) for item in self.snapshots):
            raise ValueError("partial scoreability evidence uses another support binding")
        if self.bandwidth_selection.pre_score_gate_evidence != self.pre_score_gate_evidence:
            raise ValueError("partial failure selection lost its numerical gate evidence")
        if tuple(item.bandwidth_km for item in self.bandwidth_fold_audits) != (
            self.pre_score_gate_evidence.passed_bandwidths_km
        ):
            raise ValueError("partial failure must retain every passing KDE candidate")
        if (
            tuple(score.snapshot_id for score in self.development_uniform_scores)
            != (EXPECTED_SNAPSHOTS[:4])
        ):
            raise ValueError("partial failure must retain four development uniform scores")
        if any(
            score.protocol_sha256 != self.protocol_sha256 or score.model_id != "uniform_poisson"
            for score in self.development_uniform_scores
        ):
            raise ValueError("partial failure contains an invalid uniform score")
        for fold_index, uniform_score in enumerate(self.development_uniform_scores):
            if any(
                audit.development_folds[fold_index].uniform != uniform_score
                for audit in self.bandwidth_fold_audits
            ):
                raise ValueError("partial KDE attempts do not share the retained uniform score")
        generated = self.generated_scores
        if len({score.score_id for score in generated}) != len(generated):
            raise ValueError("partial failure contains duplicate generated Score IDs")

    @property
    def generated_scores(self) -> tuple[PointProcessScoreEvidence, ...]:
        return (
            *self.development_uniform_scores,
            *(
                pair.candidate
                for audit in self.bandwidth_fold_audits
                for pair in audit.development_folds
            ),
        )

    @property
    def failed_bandwidth_gate_items(self) -> tuple[BandwidthPreScoreGateItem, ...]:
        return tuple(item for item in self.pre_score_gate_evidence.candidates if not item.passed)


def _validate_completeness_snapshots(
    config: BackgroundConfig,
    snapshots: tuple[CompletenessSnapshot, ...],
) -> tuple[CompletenessSnapshot, ...]:
    expected_definitions = build_snapshot_definitions(config)
    if len(snapshots) != len(EXPECTED_SNAPSHOTS):
        raise ValueError("completeness input must contain exactly four folds plus final")
    if tuple(item.definition for item in snapshots) != expected_definitions:
        raise ValueError("completeness snapshots do not match the frozen snapshot definitions")
    for snapshot in snapshots:
        cutoff = snapshot.analysis.cutoff_utc
        offset = cutoff.utcoffset()
        if offset is None or offset.total_seconds() != 0.0:
            raise ValueError("completeness cutoff must be UTC")
        cutoff_day = cutoff.timestamp() / 86_400.0
        if not math.isclose(
            cutoff_day,
            snapshot.definition.fit_end_day,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError("completeness analysis cutoff does not match snapshot fit_end")
        if snapshot.analysis.selected_mc not in config.completeness.candidate_magnitudes:
            raise ValueError("snapshot completeness magnitude is not a frozen candidate")
    return snapshots


def _cell_mass_mapping(
    quadrature: SpatialQuadrature,
    values: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> dict[str, float]:
    return {
        identifier: float(value)
        for identifier, value in zip(quadrature.cell_ids, values, strict=True)
    }


def _global_gate_evidence(
    protocol_sha256: str,
    snapshots: tuple[PoissonSnapshotFit, ...],
) -> BandwidthPreScoreGateEvidence:
    items: list[BandwidthPreScoreGateItem] = []
    for bandwidth in FROZEN_BANDWIDTHS_KM:
        per_snapshot = tuple(snapshot.gate_for(bandwidth) for snapshot in snapshots)
        passed = all(item.passed for item in per_snapshot)
        failure_parts = tuple(
            f"{item.snapshot_id}: {', '.join(item.failure_reasons)}"
            for item in per_snapshot
            if not item.passed
        )
        evidence_id = _canonical_sha256(
            {
                "protocol_sha256": protocol_sha256,
                "bandwidth_km": bandwidth,
                "snapshot_numerical_evidence_ids": tuple(
                    item.numerical_evidence_id for item in per_snapshot
                ),
            }
        )
        items.append(
            BandwidthPreScoreGateItem(
                bandwidth_km=bandwidth,
                passed=passed,
                numerical_evidence_id=evidence_id,
                failure_reason=None if passed else "; ".join(failure_parts),
            )
        )
    return BandwidthPreScoreGateEvidence(candidates=tuple(items))


def _uniform_parameter_snapshot_id(
    protocol_sha256: str,
    snapshot: PoissonSnapshotFit,
) -> str:
    return _canonical_sha256(
        {
            "protocol_sha256": protocol_sha256,
            "model_id": "uniform_poisson",
            "model_variant_id": _UNIFORM_VARIANT,
            "snapshot_id": snapshot.definition.snapshot_id,
            "selected_mc": snapshot.selected_mc,
            "training_event_ids": snapshot.training_event_ids,
            "training_duration_days": snapshot.training_duration_days,
            "rate_per_day": snapshot.rate_per_day,
            "study_area_km2": snapshot.uniform_model.study_area_km2,
        }
    )


def _spatial_parameter_snapshot_id(
    protocol_sha256: str,
    snapshot: PoissonSnapshotFit,
    model: SpatialPoissonModel,
) -> str:
    return _canonical_sha256(
        {
            "protocol_sha256": protocol_sha256,
            "model_id": "spatial_poisson",
            "model_variant_id": _spatial_variant(model.bandwidth_km),
            "snapshot_id": snapshot.definition.snapshot_id,
            "selected_mc": snapshot.selected_mc,
            "training_event_ids": snapshot.training_event_ids,
            "training_x_km": tuple(float(value) for value in model.mixture.training_x_km),
            "training_y_km": tuple(float(value) for value in model.mixture.training_y_km),
            "training_duration_days": snapshot.training_duration_days,
            "rate_per_day": model.rate_per_day,
            "bandwidth_km": model.bandwidth_km,
            "normalization_mass": model.normalization_mass,
        }
    )


def _uniform_score_evidence(
    protocol_sha256: str,
    snapshot: PoissonSnapshotFit,
) -> PointProcessScoreEvidence:
    log_intensity = math.log(snapshot.rate_per_day) - math.log(
        snapshot.uniform_model.study_area_km2
    )
    return PointProcessScoreEvidence(
        protocol_sha256=protocol_sha256,
        model_id="uniform_poisson",
        model_variant_id=_UNIFORM_VARIANT,
        parameter_snapshot_id=_uniform_parameter_snapshot_id(protocol_sha256, snapshot),
        snapshot_id=snapshot.definition.snapshot_id,
        fit_end_utc=snapshot.definition.fit_end_utc,
        assessment_start_utc=snapshot.definition.assessment_start_utc,
        assessment_end_utc=snapshot.definition.assessment_end_utc,
        selected_mc=snapshot.selected_mc,
        target_event_ids=snapshot.target_event_ids,
        event_log_intensities=np.full(len(snapshot.target_event_ids), log_intensity),
        compensator=(snapshot.rate_per_day * snapshot.definition.assessment_duration_days),
        numerical_gate_evidence_ids=(snapshot.training_evidence_id,),
    )


def _spatial_score_evidence(
    protocol_sha256: str,
    snapshot: PoissonSnapshotFit,
    model: SpatialPoissonModel,
    log_densities: np.ndarray[tuple[int], np.dtype[np.float64]],
    global_gate_item: BandwidthPreScoreGateItem,
) -> PointProcessScoreEvidence:
    event_log_intensities = np.asarray(
        log_densities + math.log(model.rate_per_day),
        dtype=np.float64,
    )
    return PointProcessScoreEvidence(
        protocol_sha256=protocol_sha256,
        model_id="spatial_poisson",
        model_variant_id=_spatial_variant(model.bandwidth_km),
        parameter_snapshot_id=_spatial_parameter_snapshot_id(
            protocol_sha256,
            snapshot,
            model,
        ),
        snapshot_id=snapshot.definition.snapshot_id,
        fit_end_utc=snapshot.definition.fit_end_utc,
        assessment_start_utc=snapshot.definition.assessment_start_utc,
        assessment_end_utc=snapshot.definition.assessment_end_utc,
        selected_mc=snapshot.selected_mc,
        target_event_ids=snapshot.target_event_ids,
        event_log_intensities=event_log_intensities,
        compensator=model.rate_per_day * snapshot.definition.assessment_duration_days,
        numerical_gate_evidence_ids=(
            snapshot.gate_for(model.bandwidth_km).numerical_evidence_id,
            global_gate_item.numerical_evidence_id,
        ),
    )


def run_poisson_kde_pipeline(
    config: BackgroundConfig,
    catalog: EarthquakeCatalog,
    grid_family: EqualAreaGridFamily,
    completeness_snapshots: tuple[CompletenessSnapshot, ...],
    *,
    chunk_size: int = 256,
    progress: ProgressCallback | None = None,
) -> PoissonKDEPipelineResult:
    """Fit, gate, select, and score the frozen five-snapshot Poisson/KDE family."""

    require_background_scoring_authorized(config)
    snapshots_input = _validate_completeness_snapshots(config, completeness_snapshots)
    bandwidths = tuple(float(value) for value in config.spatial_poisson.bandwidth_candidates_km)
    if bandwidths != FROZEN_BANDWIDTHS_KM:
        raise ValueError("configuration bandwidths differ from the frozen KDE candidates")
    protocol_sha256 = _canonical_sha256(config.model_dump(mode="python"))
    quadratures = {
        cell_size: SpatialQuadrature.from_grid(grid_family.at(cell_size))
        for cell_size in (50.0, 25.0, 12.5)
    }
    study_area_km2 = float(grid_family.study_area_equal_area.area) / 1_000_000.0
    anchor_day = CATALOG_ANCHOR_UTC.timestamp() / 86_400.0
    snapshot_fits: list[PoissonSnapshotFit] = []
    target_indices: dict[str, np.ndarray[tuple[int], np.dtype[np.int64]]] = {}

    for completeness in snapshots_input:
        definition = completeness.definition
        if progress is not None:
            progress(f"poisson_kde:{definition.snapshot_id}:start")
        selected_mc = float(completeness.analysis.selected_mc)
        training_mask = historical_training_mask(
            catalog,
            minimum_magnitude=selected_mc,
            fit_end_day=definition.fit_end_day,
        )
        training_indices = np.flatnonzero(training_mask)
        if training_indices.size == 0:
            raise PoissonKDEScientificInability(
                "zero_training_events",
                f"zero eligible training events in snapshot {definition.snapshot_id}",
            )
        training_duration_days = definition.fit_end_day - anchor_day
        uniform_model = fit_uniform_poisson(
            training_event_count=int(training_indices.size),
            training_duration_days=training_duration_days,
            study_area_km2=study_area_km2,
        )
        kde_models = fit_spatial_poisson_family(
            catalog.x_km[training_indices],
            catalog.y_km[training_indices],
            training_duration_days=training_duration_days,
            normalization_quadrature=quadratures[12.5],
            bandwidths_km=bandwidths,
            chunk_size=chunk_size,
        )
        masses_by_grid = {
            cell_size: evaluate_spatial_poisson_family_cell_masses(
                kde_models,
                quadrature,
            )
            for cell_size, quadrature in quadratures.items()
        }
        gate_items: list[SnapshotKDEGateEvidence] = []
        for bandwidth in bandwidths:
            model = kde_models[bandwidth]
            cached_masses = model.normalization_cell_masses
            if cached_masses is None:
                raise PoissonKDEInvariantError("fitted KDE omitted normalization cell masses")
            convergence = diagnose_three_grid_convergence(
                grid_family,
                {
                    cell_size: _cell_mass_mapping(
                        quadratures[cell_size],
                        masses_by_grid[cell_size][bandwidth],
                    )
                    for cell_size in (50.0, 25.0, 12.5)
                },
            )
            gate_items.append(
                SnapshotKDEGateEvidence(
                    protocol_sha256=protocol_sha256,
                    snapshot_id=definition.snapshot_id,
                    bandwidth_km=bandwidth,
                    normalization_mass=model.normalization_mass,
                    normalization_cell_mass_sum=math.fsum(float(value) for value in cached_masses),
                    convergence=convergence,
                )
            )

        target_mask = physical_target_mask(
            catalog,
            minimum_magnitude=selected_mc,
            origin_after_day=definition.assessment_start_day,
            origin_through_day=definition.assessment_end_day,
        )
        current_target_indices = np.flatnonzero(target_mask)
        target_indices[definition.snapshot_id] = current_target_indices
        training_event_ids = tuple(str(catalog.event_id[index]) for index in training_indices)
        training_evidence_id = _canonical_sha256(
            {
                "protocol_sha256": protocol_sha256,
                "snapshot_id": definition.snapshot_id,
                "selected_mc": selected_mc,
                "training_event_ids": training_event_ids,
                "training_origin_days": tuple(
                    float(catalog.origin_day[index]) for index in training_indices
                ),
                "training_available_days": tuple(
                    float(catalog.available_day[index]) for index in training_indices
                ),
                "training_duration_days": training_duration_days,
                "rate_per_day": uniform_model.rate_per_day,
            }
        )
        snapshot_fits.append(
            PoissonSnapshotFit(
                definition=definition,
                selected_mc=selected_mc,
                training_event_ids=training_event_ids,
                training_event_count=int(training_indices.size),
                training_duration_days=training_duration_days,
                rate_per_day=uniform_model.rate_per_day,
                training_evidence_id=training_evidence_id,
                target_event_ids=tuple(
                    str(catalog.event_id[index]) for index in current_target_indices
                ),
                uniform_model=uniform_model,
                kde_family=tuple((bandwidth, kde_models[bandwidth]) for bandwidth in bandwidths),
                grid_gate_evidence=tuple(gate_items),
            )
        )
        if progress is not None:
            progress(f"poisson_kde:{definition.snapshot_id}:done")

    snapshots = tuple(snapshot_fits)
    pre_score_gate = _global_gate_evidence(protocol_sha256, snapshots)
    passed_bandwidths = pre_score_gate.passed_bandwidths_km
    if not passed_bandwidths:
        raise PoissonKDEScientificInability(
            "all_bandwidths_failed_numerical_gate",
            "all KDE candidates failed normalization or three-grid convergence",
            gate_evidence=pre_score_gate,
        )
    zero_target_snapshots = tuple(
        snapshot.definition.snapshot_id for snapshot in snapshots if not snapshot.target_event_ids
    )
    if zero_target_snapshots:
        raise PoissonKDEScientificInability(
            "zero_target_snapshot",
            "KDE information gain is undefined for zero-target snapshots: "
            + ", ".join(zero_target_snapshots),
            gate_evidence=pre_score_gate,
        )

    uniform_scores = tuple(
        _uniform_score_evidence(protocol_sha256, snapshot) for snapshot in snapshots
    )
    global_gate_by_bandwidth = {item.bandwidth_km: item for item in pre_score_gate.candidates}
    fold_pairs_by_bandwidth: dict[float, list[PairedInformationGainEvidence]] = {
        bandwidth: [] for bandwidth in passed_bandwidths
    }
    for snapshot, uniform_score in zip(snapshots[:4], uniform_scores[:4], strict=True):
        indices = target_indices[snapshot.definition.snapshot_id]
        models = {bandwidth: snapshot.kde_model(bandwidth) for bandwidth in passed_bandwidths}
        log_densities = evaluate_spatial_poisson_family_log_densities(
            models,
            catalog.x_km[indices],
            catalog.y_km[indices],
        )
        for bandwidth in passed_bandwidths:
            spatial_score = _spatial_score_evidence(
                protocol_sha256,
                snapshot,
                models[bandwidth],
                log_densities[bandwidth],
                global_gate_by_bandwidth[bandwidth],
            )
            fold_pairs_by_bandwidth[bandwidth].append(
                PairedInformationGainEvidence.build(
                    candidate=spatial_score,
                    uniform=uniform_score,
                )
            )

    bandwidth_fold_audits = tuple(
        BandwidthFoldInformationGainAudit(
            bandwidth_km=bandwidth,
            development_folds=tuple(fold_pairs_by_bandwidth[bandwidth]),
        )
        for bandwidth in passed_bandwidths
    )
    fold_scores = {
        audit.bandwidth_km: cast(
            tuple[float, float, float, float],
            tuple(cast(float, item.information_gain_per_event) for item in audit.development_folds),
        )
        for audit in bandwidth_fold_audits
    }
    selection = select_kde_bandwidth(
        fold_scores,
        pre_score_gate_evidence=pre_score_gate,
    )
    selected_bandwidth = selection.selected_bandwidth_km
    selected_development = next(
        audit.development_folds
        for audit in bandwidth_fold_audits
        if audit.bandwidth_km == selected_bandwidth
    )

    validation_snapshot = snapshots[4]
    validation_indices = target_indices[validation_snapshot.definition.snapshot_id]
    validation_model = validation_snapshot.kde_model(selected_bandwidth)
    validation_log_density = validation_model.log_density(
        catalog.x_km[validation_indices],
        catalog.y_km[validation_indices],
    )
    validation_spatial_score = _spatial_score_evidence(
        protocol_sha256,
        validation_snapshot,
        validation_model,
        validation_log_density,
        global_gate_by_bandwidth[selected_bandwidth],
    )
    validation_pair = PairedInformationGainEvidence.build(
        candidate=validation_spatial_score,
        uniform=uniform_scores[4],
    )
    uniform_pairs = tuple(
        PairedInformationGainEvidence.build(candidate=score, uniform=score)
        for score in uniform_scores
    )
    uniform_evidence = AuditedBackgroundModelEvidence(
        model_id="uniform_poisson",
        model_variant_id=_UNIFORM_VARIANT,
        protocol_sha256=protocol_sha256,
        development_folds=uniform_pairs[:4],
        validation=uniform_pairs[4],
        failed_snapshot_reasons=(),
    )
    spatial_variant = _spatial_variant(selected_bandwidth)
    spatial_evidence = AuditedBackgroundModelEvidence(
        model_id="spatial_poisson",
        model_variant_id=spatial_variant,
        protocol_sha256=protocol_sha256,
        development_folds=selected_development,
        validation=validation_pair,
        failed_snapshot_reasons=(),
    )
    return PoissonKDEPipelineResult(
        protocol_sha256=protocol_sha256,
        snapshots=snapshots,
        pre_score_gate_evidence=pre_score_gate,
        bandwidth_fold_audits=bandwidth_fold_audits,
        bandwidth_selection=selection,
        uniform_evidence=uniform_evidence,
        spatial_evidence=spatial_evidence,
    )


def _validate_local_support_runtime(
    config: BackgroundConfig,
    runtime: LocalSupportRuntime,
) -> tuple[tuple[SnapshotDefinition, LocalSupportRuntimeSnapshot], ...]:
    if not isinstance(runtime, LocalSupportRuntime):
        raise TypeError("local_support_runtime must be a LocalSupportRuntime")
    definitions = build_snapshot_definitions(config)
    if tuple(item.snapshot_id for item in runtime.snapshots) != EXPECTED_SNAPSHOTS:
        raise ValueError("local support runtime does not use the frozen snapshot order")
    aligned: list[tuple[SnapshotDefinition, LocalSupportRuntimeSnapshot]] = []
    for definition, runtime_snapshot in zip(definitions, runtime.snapshots, strict=True):
        if runtime_snapshot.snapshot_id != definition.snapshot_id:
            raise ValueError("local support runtime snapshot IDs differ from the protocol")
        cutoff_day = runtime_snapshot.support.fit_end_utc.timestamp() / 86_400.0
        if not math.isclose(cutoff_day, definition.fit_end_day, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError("local support fit cutoff differs from the frozen snapshot")
        if not runtime_snapshot.grid_family.study_area_equal_area.equals(
            runtime_snapshot.support.retained_geometry
        ):
            raise ValueError("local support integration grid uses another geometry")
        supported_area_km2 = runtime_snapshot.support.retained_area_m2 / 1_000_000.0
        grid_area_km2 = float(runtime_snapshot.grid_family.study_area_equal_area.area) / 1_000_000.0
        if not math.isclose(
            grid_area_km2,
            supported_area_km2,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError("local support integration grid has another effective area")
        aligned.append((definition, runtime_snapshot))
    return tuple(aligned)


def _local_scoreability_gate_evidence(
    catalog: EarthquakeCatalog,
    aligned_snapshots: tuple[
        tuple[SnapshotDefinition, LocalSupportRuntimeSnapshot],
        ...,
    ],
    *,
    protocol_sha256: str,
    authorization_id: str,
) -> LocalSupportScoreabilityGateEvidence:
    """Count all training and development eligibility without opening final targets."""

    anchor_day = CATALOG_ANCHOR_UTC.timestamp() / 86_400.0
    evidence: list[LocalSupportSnapshotScoreabilityEvidence] = []
    for definition, runtime_snapshot in aligned_snapshots:
        selected_mc = float(runtime_snapshot.support.common_mc)
        training_mask = historical_training_mask(
            catalog,
            minimum_magnitude=selected_mc,
            fit_end_day=definition.fit_end_day,
        )
        training_mask &= runtime_snapshot.supported_mask
        target_event_count: int | None = None
        if definition.snapshot_id != "final_validation":
            target_mask = physical_target_mask(
                catalog,
                minimum_magnitude=selected_mc,
                origin_after_day=definition.assessment_start_day,
                origin_through_day=definition.assessment_end_day,
            )
            target_mask &= runtime_snapshot.supported_mask
            target_event_count = int(np.count_nonzero(target_mask))
        evidence.append(
            LocalSupportSnapshotScoreabilityEvidence(
                protocol_sha256=protocol_sha256,
                snapshot_id=definition.snapshot_id,
                support_id=runtime_snapshot.support.support_id,
                selected_mc=selected_mc,
                supported_area_km2=(runtime_snapshot.support.retained_area_m2 / 1_000_000.0),
                compensator_domain_id=runtime_snapshot.compensator_domain_id,
                authorization_id=authorization_id,
                training_duration_days=definition.fit_end_day - anchor_day,
                training_event_count=int(np.count_nonzero(training_mask)),
                target_event_count=target_event_count,
            )
        )
    return LocalSupportScoreabilityGateEvidence(snapshots=tuple(evidence))


def _local_global_gate_evidence(
    protocol_sha256: str,
    snapshots: tuple[LocalSupportPoissonSnapshotFit, ...],
) -> BandwidthPreScoreGateEvidence:
    items: list[BandwidthPreScoreGateItem] = []
    for bandwidth in FROZEN_BANDWIDTHS_KM:
        per_snapshot = tuple(snapshot.gate_for(bandwidth) for snapshot in snapshots)
        passed = all(item.passed for item in per_snapshot)
        failure_parts = tuple(
            f"{item.snapshot_id}: {', '.join(item.failure_reasons)}"
            for item in per_snapshot
            if not item.passed
        )
        evidence_id = _canonical_sha256(
            {
                "protocol_sha256": protocol_sha256,
                "bandwidth_km": bandwidth,
                "local_support_bindings": tuple(
                    {
                        "snapshot_id": item.snapshot_id,
                        "support_id": item.support_id,
                        "supported_area_km2": item.supported_area_km2,
                        "compensator_domain_id": item.compensator_domain_id,
                        "authorization_id": item.authorization_id,
                        "numerical_evidence_id": item.numerical_evidence_id,
                    }
                    for item in per_snapshot
                ),
            }
        )
        items.append(
            BandwidthPreScoreGateItem(
                bandwidth_km=bandwidth,
                passed=passed,
                numerical_evidence_id=evidence_id,
                failure_reason=None if passed else "; ".join(failure_parts),
            )
        )
    return BandwidthPreScoreGateEvidence(candidates=tuple(items))


def _local_uniform_parameter_snapshot_id(
    protocol_sha256: str,
    snapshot: LocalSupportPoissonSnapshotFit,
) -> str:
    return _canonical_sha256(
        {
            "protocol_sha256": protocol_sha256,
            "model_id": "uniform_poisson",
            "model_variant_id": _UNIFORM_VARIANT,
            "snapshot_id": snapshot.definition.snapshot_id,
            "support_id": snapshot.support_id,
            "selected_mc": snapshot.selected_mc,
            "supported_area_km2": snapshot.supported_area_km2,
            "compensator_domain_id": snapshot.compensator_domain_id,
            "authorization_id": snapshot.authorization_id,
            "training_event_ids": snapshot.training_event_ids,
            "training_duration_days": snapshot.training_duration_days,
            "rate_per_day": snapshot.rate_per_day,
        }
    )


def _local_spatial_parameter_snapshot_id(
    protocol_sha256: str,
    snapshot: LocalSupportPoissonSnapshotFit,
    model: SpatialPoissonModel,
) -> str:
    return _canonical_sha256(
        {
            "protocol_sha256": protocol_sha256,
            "model_id": "spatial_poisson",
            "model_variant_id": _spatial_variant(model.bandwidth_km),
            "snapshot_id": snapshot.definition.snapshot_id,
            "support_id": snapshot.support_id,
            "selected_mc": snapshot.selected_mc,
            "supported_area_km2": snapshot.supported_area_km2,
            "compensator_domain_id": snapshot.compensator_domain_id,
            "authorization_id": snapshot.authorization_id,
            "training_event_ids": snapshot.training_event_ids,
            "training_x_km": tuple(float(value) for value in model.mixture.training_x_km),
            "training_y_km": tuple(float(value) for value in model.mixture.training_y_km),
            "training_duration_days": snapshot.training_duration_days,
            "rate_per_day": model.rate_per_day,
            "bandwidth_km": model.bandwidth_km,
            "normalization_mass": model.normalization_mass,
        }
    )


def _local_target_indices(
    catalog: EarthquakeCatalog,
    runtime_snapshot: LocalSupportRuntimeSnapshot,
    definition: SnapshotDefinition,
    *,
    minimum_magnitude: float,
    observer: TargetAccessObserver | None,
) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
    if observer is not None:
        observer(definition.snapshot_id)
    target_mask = physical_target_mask(
        catalog,
        minimum_magnitude=minimum_magnitude,
        origin_after_day=definition.assessment_start_day,
        origin_through_day=definition.assessment_end_day,
    )
    target_mask &= runtime_snapshot.supported_mask
    return np.flatnonzero(target_mask)


def _local_uniform_score_evidence(
    protocol_sha256: str,
    snapshot: LocalSupportPoissonSnapshotFit,
    target_event_ids: tuple[str, ...],
) -> PointProcessScoreEvidence:
    log_intensity = math.log(snapshot.rate_per_day) - math.log(snapshot.supported_area_km2)
    return PointProcessScoreEvidence(
        protocol_sha256=protocol_sha256,
        model_id="uniform_poisson",
        model_variant_id=_UNIFORM_VARIANT,
        parameter_snapshot_id=_local_uniform_parameter_snapshot_id(
            protocol_sha256,
            snapshot,
        ),
        snapshot_id=snapshot.definition.snapshot_id,
        fit_end_utc=snapshot.definition.fit_end_utc,
        assessment_start_utc=snapshot.definition.assessment_start_utc,
        assessment_end_utc=snapshot.definition.assessment_end_utc,
        selected_mc=snapshot.selected_mc,
        target_event_ids=target_event_ids,
        event_log_intensities=np.full(len(target_event_ids), log_intensity),
        compensator=snapshot.rate_per_day * snapshot.definition.assessment_duration_days,
        numerical_gate_evidence_ids=(snapshot.training_evidence_id,),
        support_id=snapshot.support_id,
        supported_area_km2=snapshot.supported_area_km2,
        compensator_domain_id=snapshot.compensator_domain_id,
        authorization_id=snapshot.authorization_id,
    )


def _local_spatial_score_evidence(
    protocol_sha256: str,
    snapshot: LocalSupportPoissonSnapshotFit,
    model: SpatialPoissonModel,
    target_event_ids: tuple[str, ...],
    log_densities: np.ndarray[tuple[int], np.dtype[np.float64]],
    global_gate_item: BandwidthPreScoreGateItem,
) -> PointProcessScoreEvidence:
    return PointProcessScoreEvidence(
        protocol_sha256=protocol_sha256,
        model_id="spatial_poisson",
        model_variant_id=_spatial_variant(model.bandwidth_km),
        parameter_snapshot_id=_local_spatial_parameter_snapshot_id(
            protocol_sha256,
            snapshot,
            model,
        ),
        snapshot_id=snapshot.definition.snapshot_id,
        fit_end_utc=snapshot.definition.fit_end_utc,
        assessment_start_utc=snapshot.definition.assessment_start_utc,
        assessment_end_utc=snapshot.definition.assessment_end_utc,
        selected_mc=snapshot.selected_mc,
        target_event_ids=target_event_ids,
        event_log_intensities=np.asarray(
            log_densities + math.log(model.rate_per_day),
            dtype=np.float64,
        ),
        compensator=model.rate_per_day * snapshot.definition.assessment_duration_days,
        numerical_gate_evidence_ids=(
            snapshot.gate_for(model.bandwidth_km).numerical_evidence_id,
            global_gate_item.numerical_evidence_id,
        ),
        support_id=snapshot.support_id,
        supported_area_km2=snapshot.supported_area_km2,
        compensator_domain_id=snapshot.compensator_domain_id,
        authorization_id=snapshot.authorization_id,
    )


def run_local_support_poisson_kde_pipeline(
    config: BackgroundConfig,
    catalog: EarthquakeCatalog,
    local_support_runtime: LocalSupportRuntime,
    authorized_execution: AuthorizedExecution,
    *,
    chunk_size: int = 256,
    progress: ProgressCallback | None = None,
    target_access_observer: TargetAccessObserver | None = None,
) -> LocalSupportPoissonKDEPipelineResult:
    """Fit and score the authorized G1-LS Uniform/KDE family.

    Five training counts and four identity-free development target counts run
    before any fit or score.  All target-free fits and numerical gates then
    complete before development target identities open.  No final-validation
    target information is read until frozen one-standard-error selection ends.
    """

    require_background_scoring_authorized(config, authorized_execution)
    catalog_event_ids = tuple(str(value) for value in catalog.event_id)
    if catalog_event_ids != local_support_runtime.event_ids:
        raise ValueError("catalog event IDs/order differ from the local support runtime")
    aligned_snapshots = _validate_local_support_runtime(config, local_support_runtime)
    bandwidths = tuple(float(value) for value in config.spatial_poisson.bandwidth_candidates_km)
    if bandwidths != FROZEN_BANDWIDTHS_KM:
        raise ValueError("configuration bandwidths differ from the frozen KDE candidates")

    protocol_sha256 = _canonical_sha256(config.model_dump(mode="python"))
    authorization_id = authorized_execution.authorization_id
    anchor_day = CATALOG_ANCHOR_UTC.timestamp() / 86_400.0
    scoreability_gate = _local_scoreability_gate_evidence(
        catalog,
        aligned_snapshots,
        protocol_sha256=protocol_sha256,
        authorization_id=authorization_id,
    )
    if scoreability_gate.zero_training_snapshots:
        raise PoissonKDEScientificInability(
            "zero_training_events",
            "zero eligible local-support training events in snapshots: "
            + ", ".join(scoreability_gate.zero_training_snapshots),
            scoreability_gate_evidence=scoreability_gate,
        )
    if scoreability_gate.zero_target_snapshots:
        raise PoissonKDEScientificInability(
            "zero_target_snapshot",
            "KDE information gain is undefined for zero-target snapshots: "
            + ", ".join(scoreability_gate.zero_target_snapshots),
            scoreability_gate_evidence=scoreability_gate,
        )
    if progress is not None:
        progress("local_poisson_kde:scoreability_preflight:done")
    snapshot_fits: list[LocalSupportPoissonSnapshotFit] = []

    for definition, runtime_snapshot in aligned_snapshots:
        if progress is not None:
            progress(f"local_poisson_kde:{definition.snapshot_id}:start")
        selected_mc = float(runtime_snapshot.support.common_mc)
        training_mask = historical_training_mask(
            catalog,
            minimum_magnitude=selected_mc,
            fit_end_day=definition.fit_end_day,
        )
        training_mask &= runtime_snapshot.supported_mask
        training_indices = np.flatnonzero(training_mask)
        scoreability_snapshot = scoreability_gate.snapshot(definition.snapshot_id)
        if int(training_indices.size) != scoreability_snapshot.training_event_count:
            raise PoissonKDEInvariantError(
                "local-support training eligibility changed after scoreability preflight"
            )

        training_duration_days = definition.fit_end_day - anchor_day
        supported_area_km2 = runtime_snapshot.support.retained_area_m2 / 1_000_000.0
        uniform_model = fit_uniform_poisson(
            training_event_count=int(training_indices.size),
            training_duration_days=training_duration_days,
            study_area_km2=supported_area_km2,
        )
        quadratures = {
            cell_size: SpatialQuadrature.from_grid(runtime_snapshot.grid_family.at(cell_size))
            for cell_size in (50.0, 25.0, 12.5)
        }
        kde_models = fit_spatial_poisson_family(
            catalog.x_km[training_indices],
            catalog.y_km[training_indices],
            training_duration_days=training_duration_days,
            normalization_quadrature=quadratures[12.5],
            bandwidths_km=bandwidths,
            chunk_size=chunk_size,
        )
        masses_by_grid = {
            cell_size: evaluate_spatial_poisson_family_cell_masses(kde_models, quadrature)
            for cell_size, quadrature in quadratures.items()
        }
        gate_items: list[LocalSupportSnapshotKDEGateEvidence] = []
        for bandwidth in bandwidths:
            model = kde_models[bandwidth]
            cached_masses = model.normalization_cell_masses
            if cached_masses is None:
                raise PoissonKDEInvariantError("fitted local KDE omitted normalization masses")
            convergence = diagnose_three_grid_convergence(
                runtime_snapshot.grid_family,
                {
                    cell_size: _cell_mass_mapping(
                        quadratures[cell_size],
                        masses_by_grid[cell_size][bandwidth],
                    )
                    for cell_size in (50.0, 25.0, 12.5)
                },
            )
            gate_items.append(
                LocalSupportSnapshotKDEGateEvidence(
                    protocol_sha256=protocol_sha256,
                    snapshot_id=definition.snapshot_id,
                    bandwidth_km=bandwidth,
                    support_id=runtime_snapshot.support.support_id,
                    supported_area_km2=supported_area_km2,
                    compensator_domain_id=runtime_snapshot.compensator_domain_id,
                    authorization_id=authorization_id,
                    normalization_mass=model.normalization_mass,
                    normalization_cell_mass_sum=math.fsum(float(value) for value in cached_masses),
                    convergence=convergence,
                )
            )

        training_event_ids = tuple(str(catalog.event_id[index]) for index in training_indices)
        training_evidence_id = _canonical_sha256(
            {
                "protocol_sha256": protocol_sha256,
                "snapshot_id": definition.snapshot_id,
                "support_id": runtime_snapshot.support.support_id,
                "selected_mc": selected_mc,
                "supported_area_km2": supported_area_km2,
                "compensator_domain_id": runtime_snapshot.compensator_domain_id,
                "authorization_id": authorization_id,
                "training_event_ids": training_event_ids,
                "training_origin_days": tuple(
                    float(catalog.origin_day[index]) for index in training_indices
                ),
                "training_available_days": tuple(
                    float(catalog.available_day[index]) for index in training_indices
                ),
                "training_x_km": tuple(float(catalog.x_km[index]) for index in training_indices),
                "training_y_km": tuple(float(catalog.y_km[index]) for index in training_indices),
                "training_duration_days": training_duration_days,
                "rate_per_day": uniform_model.rate_per_day,
            }
        )
        snapshot_fits.append(
            LocalSupportPoissonSnapshotFit(
                definition=definition,
                support_id=runtime_snapshot.support.support_id,
                selected_mc=selected_mc,
                supported_area_km2=supported_area_km2,
                compensator_domain_id=runtime_snapshot.compensator_domain_id,
                authorization_id=authorization_id,
                training_event_ids=training_event_ids,
                training_event_count=int(training_indices.size),
                training_duration_days=training_duration_days,
                rate_per_day=uniform_model.rate_per_day,
                training_evidence_id=training_evidence_id,
                uniform_model=uniform_model,
                kde_family=tuple((bandwidth, kde_models[bandwidth]) for bandwidth in bandwidths),
                grid_gate_evidence=tuple(gate_items),
            )
        )
        if progress is not None:
            progress(f"local_poisson_kde:{definition.snapshot_id}:done")

    snapshots = tuple(snapshot_fits)
    pre_score_gate = _local_global_gate_evidence(protocol_sha256, snapshots)
    if progress is not None:
        progress("local_poisson_kde:numerical_gates:done")
    passed_bandwidths = pre_score_gate.passed_bandwidths_km
    if not passed_bandwidths:
        raise PoissonKDEScientificInability(
            "all_bandwidths_failed_numerical_gate",
            "all local-support KDE candidates failed normalization or three-grid convergence",
            gate_evidence=pre_score_gate,
            scoreability_gate_evidence=scoreability_gate,
            fitted_snapshots=snapshots,
        )

    global_gate_by_bandwidth = {item.bandwidth_km: item for item in pre_score_gate.candidates}
    fold_pairs_by_bandwidth: dict[float, list[PairedInformationGainEvidence]] = {
        bandwidth: [] for bandwidth in passed_bandwidths
    }
    development_uniform_scores: list[PointProcessScoreEvidence] = []
    if progress is not None:
        progress("local_poisson_kde:development_targets:start")
    for snapshot, (_, runtime_snapshot) in zip(
        snapshots[:4],
        aligned_snapshots[:4],
        strict=True,
    ):
        indices = _local_target_indices(
            catalog,
            runtime_snapshot,
            snapshot.definition,
            minimum_magnitude=snapshot.selected_mc,
            observer=target_access_observer,
        )
        if (
            int(indices.size)
            != scoreability_gate.snapshot(snapshot.definition.snapshot_id).target_event_count
        ):
            raise PoissonKDEInvariantError(
                "development target eligibility changed after scoreability preflight"
            )
        target_event_ids = tuple(str(catalog.event_id[index]) for index in indices)
        uniform_score = _local_uniform_score_evidence(
            protocol_sha256,
            snapshot,
            target_event_ids,
        )
        development_uniform_scores.append(uniform_score)
        models = {bandwidth: snapshot.kde_model(bandwidth) for bandwidth in passed_bandwidths}
        log_densities = evaluate_spatial_poisson_family_log_densities(
            models,
            catalog.x_km[indices],
            catalog.y_km[indices],
        )
        for bandwidth in passed_bandwidths:
            spatial_score = _local_spatial_score_evidence(
                protocol_sha256,
                snapshot,
                models[bandwidth],
                target_event_ids,
                log_densities[bandwidth],
                global_gate_by_bandwidth[bandwidth],
            )
            fold_pairs_by_bandwidth[bandwidth].append(
                PairedInformationGainEvidence.build(
                    candidate=spatial_score,
                    uniform=uniform_score,
                )
            )
    if progress is not None:
        progress("local_poisson_kde:development_targets:done")

    bandwidth_fold_audits = tuple(
        BandwidthFoldInformationGainAudit(
            bandwidth_km=bandwidth,
            development_folds=tuple(fold_pairs_by_bandwidth[bandwidth]),
        )
        for bandwidth in passed_bandwidths
    )
    fold_scores = {
        audit.bandwidth_km: cast(
            tuple[float, float, float, float],
            tuple(cast(float, item.information_gain_per_event) for item in audit.development_folds),
        )
        for audit in bandwidth_fold_audits
    }
    selection = select_kde_bandwidth(
        fold_scores,
        pre_score_gate_evidence=pre_score_gate,
    )
    if progress is not None:
        progress("local_poisson_kde:bandwidth_selection:done")

    selected_bandwidth = selection.selected_bandwidth_km
    selected_development = next(
        audit.development_folds
        for audit in bandwidth_fold_audits
        if audit.bandwidth_km == selected_bandwidth
    )
    validation_snapshot = snapshots[4]
    validation_runtime = aligned_snapshots[4][1]
    if progress is not None:
        progress("local_poisson_kde:final_validation_target:start")
    validation_indices = _local_target_indices(
        catalog,
        validation_runtime,
        validation_snapshot.definition,
        minimum_magnitude=validation_snapshot.selected_mc,
        observer=target_access_observer,
    )
    if validation_indices.size == 0:
        partial_failure = LocalSupportPoissonPartialFailureEvidence(
            protocol_sha256=protocol_sha256,
            failed_snapshot_id="final_validation",
            snapshots=snapshots,
            scoreability_gate_evidence=scoreability_gate,
            pre_score_gate_evidence=pre_score_gate,
            bandwidth_fold_audits=bandwidth_fold_audits,
            bandwidth_selection=selection,
            development_uniform_scores=tuple(development_uniform_scores),
        )
        raise PoissonKDEScientificInability(
            "zero_target_snapshot",
            "KDE information gain is undefined for zero-target snapshot: "
            "final_validation; development Scores and one-SE selection already completed",
            gate_evidence=pre_score_gate,
            scoreability_gate_evidence=scoreability_gate,
            partial_failure_evidence=partial_failure,
        )
    validation_target_ids = tuple(str(catalog.event_id[index]) for index in validation_indices)
    validation_uniform_score = _local_uniform_score_evidence(
        protocol_sha256,
        validation_snapshot,
        validation_target_ids,
    )
    validation_model = validation_snapshot.kde_model(selected_bandwidth)
    validation_log_density = validation_model.log_density(
        catalog.x_km[validation_indices],
        catalog.y_km[validation_indices],
    )
    validation_spatial_score = _local_spatial_score_evidence(
        protocol_sha256,
        validation_snapshot,
        validation_model,
        validation_target_ids,
        validation_log_density,
        global_gate_by_bandwidth[selected_bandwidth],
    )
    validation_pair = PairedInformationGainEvidence.build(
        candidate=validation_spatial_score,
        uniform=validation_uniform_score,
    )
    if progress is not None:
        progress("local_poisson_kde:final_validation_target:done")

    uniform_scores = (*development_uniform_scores, validation_uniform_score)
    uniform_pairs = tuple(
        PairedInformationGainEvidence.build(candidate=score, uniform=score)
        for score in uniform_scores
    )
    uniform_evidence = AuditedBackgroundModelEvidence(
        model_id="uniform_poisson",
        model_variant_id=_UNIFORM_VARIANT,
        protocol_sha256=protocol_sha256,
        development_folds=uniform_pairs[:4],
        validation=uniform_pairs[4],
        failed_snapshot_reasons=(),
    )
    spatial_evidence = AuditedBackgroundModelEvidence(
        model_id="spatial_poisson",
        model_variant_id=_spatial_variant(selected_bandwidth),
        protocol_sha256=protocol_sha256,
        development_folds=selected_development,
        validation=validation_pair,
        failed_snapshot_reasons=(),
    )
    return LocalSupportPoissonKDEPipelineResult(
        protocol_sha256=protocol_sha256,
        snapshots=snapshots,
        scoreability_gate_evidence=scoreability_gate,
        pre_score_gate_evidence=pre_score_gate,
        bandwidth_fold_audits=bandwidth_fold_audits,
        bandwidth_selection=selection,
        uniform_evidence=uniform_evidence,
        spatial_evidence=spatial_evidence,
    )


__all__ = [
    "BandwidthFoldInformationGainAudit",
    "LocalSupportPoissonKDEPipelineResult",
    "LocalSupportPoissonPartialFailureEvidence",
    "LocalSupportPoissonSnapshotFit",
    "LocalSupportScoreabilityGateEvidence",
    "LocalSupportSnapshotKDEGateEvidence",
    "LocalSupportSnapshotScoreabilityEvidence",
    "PoissonKDEInabilityCode",
    "PoissonKDEInvariantError",
    "PoissonKDEPipelineError",
    "PoissonKDEPipelineResult",
    "PoissonKDEScientificInability",
    "PoissonSnapshotFit",
    "SnapshotKDEGateEvidence",
    "TargetAccessObserver",
    "run_local_support_poisson_kde_pipeline",
    "run_poisson_kde_pipeline",
]
