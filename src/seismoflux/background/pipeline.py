"""Pure production orchestration for the complete frozen stage-2 science workflow."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from seismoflux.background.adapters import (
    build_analytic_simulation_expectations,
    build_etas_model_spec,
    point_area_quadrature_from_grid,
)
from seismoflux.background.artifacts import canonical_json_bytes
from seismoflux.background.catalog import EarthquakeCatalog, StudyArea
from seismoflux.background.completeness import CompletenessScientificInability
from seismoflux.background.config import BackgroundConfig
from seismoflux.background.etas_fit import (
    ETASEvent,
    ETASIssueIntensityFields,
    ETASModelSpec,
    ETASParameters,
    evaluate_etas_issue_intensity_fields,
)
from seismoflux.background.evaluation import (
    BootstrapInterval,
    InformationGainContributions,
    bootstrap_information_gain,
)
from seismoflux.background.evidence import (
    EXPECTED_SNAPSHOTS,
    AuditedBackgroundModelEvidence,
    AuditedG1Assessment,
    AuditedModelSelection,
    ModelId,
    assess_audited_g1,
    select_audited_background_model,
)
from seismoflux.background.execution import detect_physical_core_count
from seismoflux.background.future import (
    DEFAULT_MAX_WORKERS,
    MINIMUM_RESERVED_PHYSICAL_CORES,
    FutureWorkerResources,
    ValidationFutureEnsembles,
    simulate_all_validation_issue_ensembles,
)
from seismoflux.background.grid import EqualAreaGridFamily
from seismoflux.background.horizons import (
    EXPECTED_HORIZONS,
    EXPECTED_PUBLICATION_DELAYS,
    BackgroundHorizonBacktests,
    IssuePointProcessScore,
    PairedHorizonBacktest,
    run_issue_horizon_backtests,
)
from seismoflux.background.issues import FrozenIssueCalendar, IssueExposure
from seismoflux.background.pipeline_etas import ETASPipelineResult, run_etas_pipeline
from seismoflux.background.pipeline_poisson import (
    PoissonKDEPipelineResult,
    PoissonKDEScientificInability,
    run_poisson_kde_pipeline,
)
from seismoflux.background.poisson import BandwidthPreScoreGateEvidence, SpatialPoissonModel
from seismoflux.background.regression import (
    AnalyticSimulationRegression,
    ProductionFixtureRegression,
    run_analytic_simulation_regression,
    run_production_fixture_regression,
)
from seismoflux.background.scoring_authorization import require_background_scoring_authorized
from seismoflux.background.visualization import (
    ConditionalIntensityRender,
    render_conditional_intensity_figure,
)
from seismoflux.background.workflow import (
    CompletenessSnapshot,
    analyze_snapshot_completeness,
    build_snapshot_definitions,
    catalog_etas_events,
    physical_target_mask,
)

CandidateModelId = Literal["spatial_poisson", "etas"]
ScientificFailureReasonCode = Literal[
    "all_bandwidths_failed_numerical_gate",
    "estimate_above_frozen_maximum",
    "no_eligible_spatial_stratum",
    "no_eligible_temporal_stratum",
    "selected_mc_has_no_events",
    "zero_target_snapshot",
    "zero_training_events",
]
ProgressCallback = Callable[[str], None]
_REPRESENTATIVE_ISSUE_DATE = "2025-06-26"
_PRIMARY_GRID_SIZE_KM = 25.0


class BackgroundPipelineGateError(RuntimeError):
    """Raised when a mandatory production scientific gate fails closed."""


@dataclass(frozen=True, slots=True)
class NumericalRegressionEvidence:
    """Both mandatory pre-data numerical regression gates."""

    production_fixture: ProductionFixtureRegression
    analytic_simulation: AnalyticSimulationRegression

    def __post_init__(self) -> None:
        if not self.production_fixture.passed or not self.analytic_simulation.passed:
            raise ValueError("pipeline regression evidence must contain two passing gates")


@dataclass(frozen=True, slots=True)
class PipelineResourceEvidence:
    """One reserved-core worker plan shared by all future issue ensembles."""

    detected_physical_cores: int | None
    reserve_physical_cores: int
    configured_max_workers: int
    effective_workers: int

    def __post_init__(self) -> None:
        detected = self.detected_physical_cores
        if detected is not None and (
            not isinstance(detected, int) or isinstance(detected, bool) or detected <= 0
        ):
            raise ValueError("detected physical cores must be positive when known")
        if (
            not isinstance(self.reserve_physical_cores, int)
            or isinstance(self.reserve_physical_cores, bool)
            or self.reserve_physical_cores < MINIMUM_RESERVED_PHYSICAL_CORES
        ):
            raise ValueError("pipeline must reserve at least two physical cores")
        if (
            not isinstance(self.configured_max_workers, int)
            or isinstance(self.configured_max_workers, bool)
            or self.configured_max_workers <= 0
        ):
            raise ValueError("configured max_workers must be a positive integer")
        expected = (
            1
            if detected is None
            else min(
                self.configured_max_workers,
                max(1, detected - self.reserve_physical_cores),
            )
        )
        if self.effective_workers != expected:
            raise ValueError("pipeline effective workers disagree with the reserve-core rule")


@dataclass(frozen=True, slots=True)
class IntegrationGridCellEvidence:
    """One geometry-free cell row in a frozen equal-area integration grid."""

    cell_id: str
    row: int
    column: int
    representative_x_km: float
    representative_y_km: float
    clipped_area_km2: float

    def __post_init__(self) -> None:
        if not self.cell_id:
            raise ValueError("integration grid cell_id must not be empty")
        if not isinstance(self.row, int) or isinstance(self.row, bool):
            raise TypeError("integration grid row must be an integer")
        if not isinstance(self.column, int) or isinstance(self.column, bool):
            raise TypeError("integration grid column must be an integer")
        values = (
            self.representative_x_km,
            self.representative_y_km,
            self.clipped_area_km2,
        )
        if any(not math.isfinite(value) for value in values) or self.clipped_area_km2 <= 0.0:
            raise ValueError("integration grid coordinates/area must be finite and positive-area")


@dataclass(frozen=True, slots=True)
class IntegrationGridResolutionEvidence:
    """All cells for one frozen resolution in deterministic row/column order."""

    cell_size_km: float
    cells: tuple[IntegrationGridCellEvidence, ...]
    total_clipped_area_km2: float

    def __post_init__(self) -> None:
        if self.cell_size_km not in {50.0, 25.0, 12.5}:
            raise ValueError("integration grid resolution must be 50, 25, or 12.5 km")
        if not self.cells:
            raise ValueError("integration grid resolution must contain positive-area cells")
        identifiers = tuple(cell.cell_id for cell in self.cells)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("integration grid cell IDs must be unique")
        ordering = tuple((cell.row, cell.column) for cell in self.cells)
        if ordering != tuple(sorted(ordering)):
            raise ValueError("integration grid cells must remain ordered by row then column")
        calculated = math.fsum(cell.clipped_area_km2 for cell in self.cells)
        if not math.isclose(
            self.total_clipped_area_km2,
            calculated,
            rel_tol=5.0e-15,
            abs_tol=1.0e-12,
        ):
            raise ValueError("integration grid total area must equal its exact clipped cell sum")


@dataclass(frozen=True, slots=True)
class IntegrationGridEvidence:
    """Geometry-free 50/25/12.5-km integration support retained for publication."""

    study_area_km2: float
    resolutions: tuple[IntegrationGridResolutionEvidence, ...]

    def __post_init__(self) -> None:
        if not math.isfinite(self.study_area_km2) or self.study_area_km2 <= 0.0:
            raise ValueError("integration grid study area must be finite and positive")
        if tuple(item.cell_size_km for item in self.resolutions) != (50.0, 25.0, 12.5):
            raise ValueError("integration grid evidence must retain 50/25/12.5 km in order")
        if any(
            not math.isclose(
                item.total_clipped_area_km2,
                self.study_area_km2,
                rel_tol=1.0e-12,
                abs_tol=1.0e-9,
            )
            for item in self.resolutions
        ):
            raise ValueError("every integration grid must exactly cover the study area")

    def at(self, cell_size_km: float) -> IntegrationGridResolutionEvidence:
        requested = float(cell_size_km)
        for resolution in self.resolutions:
            if resolution.cell_size_km == requested:
                return resolution
        raise KeyError(f"integration evidence has no {requested:g} km grid")


@dataclass(frozen=True, slots=True)
class BootstrapModelOutcome:
    """One candidate's validation bootstrap, or an explicit scientific skip."""

    model_id: CandidateModelId
    interval: BootstrapInterval | None
    not_run_reason: str | None

    def __post_init__(self) -> None:
        if self.model_id not in {"spatial_poisson", "etas"}:
            raise ValueError("bootstrap outcome has an unknown candidate model")
        if (self.interval is None) == (self.not_run_reason is None):
            raise ValueError("bootstrap outcome must be either completed or explicitly skipped")
        if self.interval is not None and (
            self.interval.replications != 2000 or self.interval.confidence_level != 0.95
        ):
            raise ValueError("bootstrap outcome differs from the frozen 2000-by-95% protocol")
        if self.not_run_reason is not None and not self.not_run_reason:
            raise ValueError("bootstrap skip reason must not be empty")


@dataclass(frozen=True, slots=True)
class ValidationBootstrapEvidence:
    spatial_poisson: BootstrapModelOutcome
    etas: BootstrapModelOutcome

    def __post_init__(self) -> None:
        if self.spatial_poisson.model_id != "spatial_poisson" or self.etas.model_id != "etas":
            raise ValueError("validation bootstrap outcomes have the wrong model families")


@dataclass(frozen=True, slots=True)
class HorizonModelOutcome:
    """All 0/1/7-day non-overlapping horizon diagnostics for one candidate."""

    model_id: CandidateModelId
    comparisons: tuple[PairedHorizonBacktest, ...] | None
    not_run_reason: str | None

    def __post_init__(self) -> None:
        if self.model_id not in {"spatial_poisson", "etas"}:
            raise ValueError("horizon outcome has an unknown candidate model")
        if (self.comparisons is None) == (self.not_run_reason is None):
            raise ValueError("horizon outcome must be completed or explicitly skipped")
        if self.comparisons is not None:
            expected = tuple(
                (delay, horizon)
                for delay in EXPECTED_PUBLICATION_DELAYS
                for horizon in EXPECTED_HORIZONS
            )
            observed = tuple(
                (item.publication_delay_days, item.horizon_days) for item in self.comparisons
            )
            if observed != expected or any(
                item.candidate_model_id != self.model_id for item in self.comparisons
            ):
                raise ValueError("horizon outcome does not contain the frozen delay/horizon grid")
        if self.not_run_reason is not None and not self.not_run_reason:
            raise ValueError("horizon skip reason must not be empty")


@dataclass(frozen=True, slots=True)
class ValidationHorizonEvidence:
    """Spatial diagnostics always run; ETAS may be explicitly unavailable."""

    spatial_poisson: HorizonModelOutcome
    etas: HorizonModelOutcome
    complete_backtests: BackgroundHorizonBacktests | None

    def __post_init__(self) -> None:
        if self.spatial_poisson.model_id != "spatial_poisson" or self.etas.model_id != "etas":
            raise ValueError("horizon evidence has the wrong model families")
        if self.spatial_poisson.comparisons is None:
            raise ValueError("spatial-Poisson horizon diagnostics must always run")
        if (self.complete_backtests is None) != (self.etas.comparisons is None):
            raise ValueError("complete horizon evidence must exist exactly when ETAS ran")


@dataclass(frozen=True, slots=True)
class ETASFutureOutcome:
    """All validation issue ensembles, or an explicit final-ETAS skip."""

    ensembles: ValidationFutureEnsembles | None
    not_run_reason: str | None

    def __post_init__(self) -> None:
        if (self.ensembles is None) == (self.not_run_reason is None):
            raise ValueError("future outcome must be completed or explicitly skipped")
        if self.not_run_reason is not None and not self.not_run_reason:
            raise ValueError("future skip reason must not be empty")


def _readonly(values: object) -> NDArray[np.float64]:
    result = np.ascontiguousarray(values, dtype=np.float64)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class RepresentativeConditionalIntensity:
    """Exact 25-km data and deterministic render for the selected model."""

    issue_date_local: str
    issue_time_utc: str
    selected_model_id: ModelId
    selected_model_variant_id: str
    primary_grid_cell_size_km: float
    cell_ids: tuple[str, ...]
    rows: tuple[int, ...]
    columns: tuple[int, ...]
    representative_x_km: NDArray[np.float64]
    representative_y_km: NDArray[np.float64]
    clipped_area_km2: NDArray[np.float64]
    eligible_parent_event_ids: tuple[str, ...]
    background_intensity: NDArray[np.float64]
    triggering_intensity: NDArray[np.float64]
    total_intensity: NDArray[np.float64]
    render: ConditionalIntensityRender

    def __post_init__(self) -> None:
        if self.issue_date_local != _REPRESENTATIVE_ISSUE_DATE:
            raise ValueError("representative intensity must use the frozen 2025-06-26 issue")
        if self.selected_model_id not in {"uniform_poisson", "spatial_poisson", "etas"}:
            raise ValueError("representative intensity uses an unknown model family")
        if not self.selected_model_variant_id:
            raise ValueError("representative model variant must not be empty")
        if self.primary_grid_cell_size_km != _PRIMARY_GRID_SIZE_KM:
            raise ValueError("representative intensity must use the 25-km primary grid")
        cell_ids = tuple(self.cell_ids)
        rows = tuple(self.rows)
        columns = tuple(self.columns)
        if (
            not cell_ids
            or any(not value for value in cell_ids)
            or len(set(cell_ids)) != len(cell_ids)
            or len(cell_ids) != len(rows)
            or len(cell_ids) != len(columns)
        ):
            raise ValueError("representative cell identifiers/indices must align and be unique")
        if tuple(zip(rows, columns, strict=True)) != tuple(sorted(zip(rows, columns, strict=True))):
            raise ValueError("representative cells must remain ordered by row then column")
        representative_x = _readonly(self.representative_x_km)
        representative_y = _readonly(self.representative_y_km)
        clipped_area = _readonly(self.clipped_area_km2)
        parent_ids = tuple(self.eligible_parent_event_ids)
        if any(not value for value in parent_ids) or len(set(parent_ids)) != len(parent_ids):
            raise ValueError("representative eligible parent IDs must be unique")
        background = _readonly(self.background_intensity)
        triggering = _readonly(self.triggering_intensity)
        total = _readonly(self.total_intensity)
        if (
            representative_x.shape != (len(cell_ids),)
            or representative_y.shape != (len(cell_ids),)
            or clipped_area.shape != (len(cell_ids),)
            or not np.all(np.isfinite(representative_x))
            or not np.all(np.isfinite(representative_y))
            or not np.all(np.isfinite(clipped_area))
            or np.any(clipped_area <= 0.0)
            or background.ndim != 1
            or background.shape != (len(cell_ids),)
            or background.shape != triggering.shape
            or background.shape != total.shape
            or not np.all(np.isfinite(background))
            or not np.all(np.isfinite(triggering))
            or not np.all(np.isfinite(total))
            or np.any(background < 0.0)
            or np.any(triggering < 0.0)
            or np.any(total < 0.0)
        ):
            raise ValueError("representative intensities must be aligned finite vectors")
        if not np.allclose(total, background + triggering, rtol=5.0e-15, atol=1.0e-15):
            raise ValueError("representative total must equal background plus triggering")
        if self.selected_model_id != "etas" and (parent_ids or bool(np.any(triggering != 0.0))):
            raise ValueError("uniform/spatial representative trigger intensity must be zero")
        object.__setattr__(self, "eligible_parent_event_ids", parent_ids)
        object.__setattr__(self, "cell_ids", cell_ids)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "representative_x_km", representative_x)
        object.__setattr__(self, "representative_y_km", representative_y)
        object.__setattr__(self, "clipped_area_km2", clipped_area)
        object.__setattr__(self, "background_intensity", background)
        object.__setattr__(self, "triggering_intensity", triggering)
        object.__setattr__(self, "total_intensity", total)


@dataclass(frozen=True, slots=True)
class BackgroundPipelineFailure:
    """Expected scientific inability to complete the frozen background baselines."""

    protocol_sha256: str
    failure_stage: Literal["completeness", "poisson_kde"]
    failure_reason_code: ScientificFailureReasonCode
    failure_reasons: tuple[str, ...]
    regressions: NumericalRegressionEvidence
    completeness: tuple[CompletenessSnapshot, ...]
    pre_score_gate_evidence: BandwidthPreScoreGateEvidence | None
    resources: PipelineResourceEvidence
    integration_grids: IntegrationGridEvidence

    def __post_init__(self) -> None:
        if len(self.protocol_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.protocol_sha256
        ):
            raise ValueError("pipeline failure protocol fingerprint must be SHA-256")
        normalized_reasons = tuple(sorted(set(self.failure_reasons)))
        if not normalized_reasons or any(not reason.strip() for reason in normalized_reasons):
            raise ValueError("pipeline failure must retain non-empty unique reasons")
        if normalized_reasons != self.failure_reasons:
            raise ValueError("pipeline failure reasons must be sorted and unique")
        if self.failure_stage == "completeness" and self.failure_reason_code not in {
            "estimate_above_frozen_maximum",
            "no_eligible_spatial_stratum",
            "no_eligible_temporal_stratum",
            "selected_mc_has_no_events",
        }:
            raise ValueError("completeness failure has a Poisson reason code")
        if self.failure_stage == "poisson_kde" and self.failure_reason_code not in {
            "all_bandwidths_failed_numerical_gate",
            "zero_target_snapshot",
            "zero_training_events",
        }:
            raise ValueError("Poisson failure has a completeness reason code")
        snapshots = tuple(item.definition.snapshot_id for item in self.completeness)
        if self.failure_stage == "completeness":
            if snapshots or self.pre_score_gate_evidence is not None:
                raise ValueError("completeness failure cannot retain downstream evidence")
        elif snapshots != EXPECTED_SNAPSHOTS:
            raise ValueError("Poisson failure must retain all completeness snapshots")
        gate_evidence = self.pre_score_gate_evidence
        if self.failure_reason_code == "zero_training_events" and gate_evidence is not None:
            raise ValueError("zero-training failure occurs before the pre-score gate")
        if self.failure_reason_code == "all_bandwidths_failed_numerical_gate" and (
            gate_evidence is None or gate_evidence.passed_bandwidths_km
        ):
            raise ValueError("all-bandwidth failure requires a fully failed pre-score gate")
        if self.failure_reason_code == "zero_target_snapshot" and (
            gate_evidence is None or not gate_evidence.passed_bandwidths_km
        ):
            raise ValueError("zero-target failure requires a passing pre-score candidate")


@dataclass(frozen=True, slots=True)
class BackgroundPipelineResult:
    """Complete immutable stage-2 scientific result without source-directory state."""

    protocol_sha256: str
    regressions: NumericalRegressionEvidence
    completeness: tuple[CompletenessSnapshot, ...]
    poisson: PoissonKDEPipelineResult
    etas: ETASPipelineResult
    g1: AuditedG1Assessment
    selection: AuditedModelSelection
    bootstrap: ValidationBootstrapEvidence
    horizons: ValidationHorizonEvidence
    future: ETASFutureOutcome
    representative: RepresentativeConditionalIntensity
    resources: PipelineResourceEvidence
    integration_grids: IntegrationGridEvidence

    def __post_init__(self) -> None:
        if tuple(item.definition.snapshot_id for item in self.completeness) != (EXPECTED_SNAPSHOTS):
            raise ValueError("pipeline completeness must retain the five frozen snapshots")
        if self.poisson.protocol_sha256 != self.protocol_sha256 or (
            self.etas.protocol_sha256 != self.protocol_sha256
        ):
            raise ValueError("pipeline model results do not share one protocol fingerprint")
        if (
            self.selection.selected_model_variant_id
            != self.representative.selected_model_variant_id
        ):
            raise ValueError("representative render does not use the selected model variant")
        if self.future.ensembles is not None:
            future_resources = self.future.ensembles.resources
            expected_resources = FutureWorkerResources(
                detected_physical_cores=self.resources.detected_physical_cores,
                reserve_physical_cores=self.resources.reserve_physical_cores,
                configured_max_workers=self.resources.configured_max_workers,
                effective_workers=self.resources.effective_workers,
            )
            if future_resources != expected_resources:
                raise ValueError("future ensemble resources differ from the pipeline plan")
        primary = self.integration_grids.at(_PRIMARY_GRID_SIZE_KM)
        if (
            self.representative.cell_ids != tuple(cell.cell_id for cell in primary.cells)
            or self.representative.rows != tuple(cell.row for cell in primary.cells)
            or self.representative.columns != tuple(cell.column for cell in primary.cells)
            or not np.array_equal(
                self.representative.representative_x_km,
                np.asarray([cell.representative_x_km for cell in primary.cells]),
            )
            or not np.array_equal(
                self.representative.representative_y_km,
                np.asarray([cell.representative_y_km for cell in primary.cells]),
            )
            or not np.array_equal(
                self.representative.clipped_area_km2,
                np.asarray([cell.clipped_area_km2 for cell in primary.cells]),
            )
        ):
            raise ValueError("representative 25-km rows must align with integration grid evidence")


BackgroundPipelineOutcome = BackgroundPipelineResult | BackgroundPipelineFailure


@dataclass(frozen=True, slots=True)
class _SpatialBackgroundDensity:
    model: SpatialPoissonModel

    def __call__(self, x_km: float, y_km: float) -> float:
        return self.model.density_scalar(x_km, y_km)

    def density_many(self, x_km: object, y_km: object) -> NDArray[np.float64]:
        return self.model.density(x_km, y_km)


@dataclass(frozen=True, slots=True)
class _FinalETASContext:
    parameters: ETASParameters
    spec: ETASModelSpec
    parameter_snapshot_id: str
    model_variant_id: str


def _failed_checks(checks: tuple[tuple[str, bool], ...]) -> str:
    return ", ".join(name for name, passed in checks if not passed)


def _notify(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _run_regressions(
    config: BackgroundConfig,
    production_fixture_path: Path,
    progress: ProgressCallback | None,
) -> NumericalRegressionEvidence:
    if not isinstance(production_fixture_path, Path):
        raise TypeError("production_fixture_path must be pathlib.Path")
    if not production_fixture_path.is_absolute():
        raise ValueError("production_fixture_path must be absolute")
    _notify(progress, "regression:production_fixture:start")
    production = run_production_fixture_regression(production_fixture_path)
    if not production.passed:
        _notify(progress, "regression:production_fixture:failed")
        raise BackgroundPipelineGateError(
            "production ETAS fixture regression failed: " + _failed_checks(production.checks)
        )
    _notify(progress, "regression:production_fixture:done")
    analytic_config = config.numerical_regression.analytic_simulation
    _notify(progress, "regression:analytic_simulation_8192:start")
    analytic = run_analytic_simulation_regression(
        replicate_count=analytic_config.replicate_count,
        maximum_events_per_replicate=(
            config.randomness.future_simulation.maximum_events_per_replicate
        ),
        expectations=build_analytic_simulation_expectations(config),
    )
    if not analytic.passed:
        _notify(progress, "regression:analytic_simulation_8192:failed")
        raise BackgroundPipelineGateError(
            "8192-replicate analytic simulation regression failed: "
            + _failed_checks(analytic.checks)
        )
    _notify(progress, "regression:analytic_simulation_8192:done")
    return NumericalRegressionEvidence(
        production_fixture=production,
        analytic_simulation=analytic,
    )


def _resource_evidence(
    *,
    max_workers: int,
    reserve_physical_cores: int,
) -> PipelineResourceEvidence:
    if not isinstance(max_workers, int) or isinstance(max_workers, bool) or max_workers <= 0:
        raise ValueError("max_workers must be a positive integer")
    if (
        not isinstance(reserve_physical_cores, int)
        or isinstance(reserve_physical_cores, bool)
        or reserve_physical_cores < MINIMUM_RESERVED_PHYSICAL_CORES
    ):
        raise ValueError("reserve_physical_cores must be at least two")
    detected = detect_physical_core_count()
    effective = (
        1 if detected is None else min(max_workers, max(1, detected - reserve_physical_cores))
    )
    return PipelineResourceEvidence(
        detected_physical_cores=detected,
        reserve_physical_cores=reserve_physical_cores,
        configured_max_workers=max_workers,
        effective_workers=effective,
    )


def _validate_spatial_inputs(
    config: BackgroundConfig,
    study_area: StudyArea,
    grid_family: EqualAreaGridFamily,
) -> None:
    if not grid_family.study_area_equal_area.equals(study_area.projected):
        raise ValueError("pipeline grids must use the supplied target-independent study area")
    if tuple(grid.spec.cell_size_km for grid in grid_family.grids) != (50.0, 25.0, 12.5):
        raise ValueError("pipeline requires exactly the frozen 50/25/12.5-km grids")
    if config.integration.primary_grid_cell_km != _PRIMARY_GRID_SIZE_KM:
        raise ValueError("pipeline primary grid must remain frozen at 25 km")
    if not math.isclose(
        study_area.area_km2,
        float(study_area.projected.area) / 1_000_000.0,
        rel_tol=1.0e-12,
        abs_tol=1.0e-9,
    ):
        raise ValueError("study-area reported and projected areas disagree")


def _integration_grid_evidence(
    study_area: StudyArea,
    grid_family: EqualAreaGridFamily,
) -> IntegrationGridEvidence:
    """Freeze all integration supports as geometry-free numeric publication evidence."""

    resolutions = tuple(
        IntegrationGridResolutionEvidence(
            cell_size_km=grid.spec.cell_size_km,
            cells=tuple(
                IntegrationGridCellEvidence(
                    cell_id=cell.id,
                    row=cell.row,
                    column=cell.column,
                    representative_x_km=float(cell.representative_point.x) / 1_000.0,
                    representative_y_km=float(cell.representative_point.y) / 1_000.0,
                    clipped_area_km2=cell.clipped_area_m2 / 1_000_000.0,
                )
                for cell in grid.cells
            ),
            total_clipped_area_km2=math.fsum(
                cell.clipped_area_m2 / 1_000_000.0 for cell in grid.cells
            ),
        )
        for grid in grid_family.grids
    )
    return IntegrationGridEvidence(
        study_area_km2=study_area.area_km2,
        resolutions=resolutions,
    )


def _bootstrap_outcome(
    model_id: CandidateModelId,
    evidence: AuditedBackgroundModelEvidence,
    config: BackgroundConfig,
) -> BootstrapModelOutcome:
    validation = evidence.validation
    if validation is None:
        reasons = "; ".join(
            reason
            for snapshot_id, snapshot_reasons in evidence.failed_snapshot_reasons
            if snapshot_id == "final_validation"
            for reason in snapshot_reasons
        )
        return BootstrapModelOutcome(
            model_id=model_id,
            interval=None,
            not_run_reason=(
                "final validation evidence unavailable" + (f": {reasons}" if reasons else "")
            ),
        )
    if not validation.candidate.target_event_ids:
        return BootstrapModelOutcome(
            model_id=model_id,
            interval=None,
            not_run_reason="validation physical target set is empty",
        )
    seed_id: Literal[
        "spatial_poisson_vs_uniform_poisson",
        "etas_vs_uniform_poisson",
    ] = (
        "spatial_poisson_vs_uniform_poisson"
        if model_id == "spatial_poisson"
        else "etas_vs_uniform_poisson"
    )
    interval = bootstrap_information_gain(
        InformationGainContributions(
            physical_event_ids=validation.candidate.target_event_ids,
            event_log_intensity_differences=validation.event_log_intensity_differences,
            compensator_difference=validation.compensator_difference,
        ),
        model_seed_id=seed_id,
        replications=config.evaluation.bootstrap_replications,
        confidence_level=config.evaluation.confidence_level,
    )
    return BootstrapModelOutcome(model_id=model_id, interval=interval, not_run_reason=None)


def _final_etas_context(
    config: BackgroundConfig,
    completeness: tuple[CompletenessSnapshot, ...],
    etas: ETASPipelineResult,
) -> tuple[_FinalETASContext | None, str | None]:
    attempt = etas.attempt("final_validation")
    if not attempt.succeeded:
        reasons = "; ".join(attempt.failure_reasons) or "final ETAS attempt was not scored"
        return None, f"final_validation ETAS unavailable: {reasons}"
    fit_result = attempt.fit_result
    if (
        fit_result is None
        or not fit_result.stability.stable
        or fit_result.best_parameters is None
        or attempt.parameter_snapshot_id is None
    ):
        return None, "final_validation ETAS omitted a stable fitted parameter snapshot"
    final_completeness = completeness[-1]
    spec = build_etas_model_spec(
        config,
        selected_mc=float(final_completeness.analysis.selected_mc),
        aki_b_value=float(final_completeness.analysis.selected_aki_b_value),
    )
    spec.validate_parameters(fit_result.best_parameters)
    return (
        _FinalETASContext(
            parameters=fit_result.best_parameters,
            spec=spec,
            parameter_snapshot_id=attempt.parameter_snapshot_id,
            model_variant_id=attempt.model_variant_id,
        ),
        None,
    )


def _calendar_issue(
    calendar: FrozenIssueCalendar,
    issue_date_local: str,
) -> tuple[float, str]:
    dates = calendar.validation.actual_issue_dates_local
    if issue_date_local not in dates:
        raise ValueError(
            f"representative issue {issue_date_local} is absent from the frozen validation calendar"
        )
    index = dates.index(issue_date_local)
    if len(calendar.validation.actual_issue_days) != len(dates):
        raise ValueError("validation issue dates and days do not align")
    issue_day = float(calendar.validation.actual_issue_days[index])
    if not math.isfinite(issue_day):
        raise ValueError("validation issue day must be finite")
    issue_time = datetime.fromtimestamp(issue_day * 86_400.0, tz=UTC)
    return issue_day, issue_time.isoformat(timespec="seconds").replace("+00:00", "Z")


def _issue_history(
    catalog: EarthquakeCatalog,
    *,
    issue_day: float,
    spec: ETASModelSpec,
) -> tuple[ETASEvent, ...]:
    mask = np.asarray(
        catalog.inside_external_buffer
        & (catalog.magnitude >= spec.mc)
        & (catalog.origin_day >= issue_day - spec.history_parent_cutoff_days)
        & (catalog.origin_day <= issue_day)
        & (catalog.available_day <= issue_day),
        dtype=np.bool_,
    )
    return catalog_etas_events(catalog, mask, time_origin_day=issue_day)


def _histories_by_validation_issue(
    catalog: EarthquakeCatalog,
    calendar: FrozenIssueCalendar,
    spec: ETASModelSpec,
) -> dict[str, tuple[ETASEvent, ...]]:
    dates = calendar.validation.actual_issue_dates_local
    if not dates or dates != tuple(sorted(set(dates))):
        raise ValueError("validation issue dates must be non-empty, unique, and sorted")
    return {
        issue_date: _issue_history(
            catalog,
            issue_day=_calendar_issue(calendar, issue_date)[0],
            spec=spec,
        )
        for issue_date in dates
    }


def _target_indices(
    catalog: EarthquakeCatalog,
    exposure: IssueExposure,
    *,
    selected_mc: float,
) -> NDArray[np.int64]:
    mask = physical_target_mask(
        catalog,
        minimum_magnitude=selected_mc,
        origin_after_day=exposure.issue_day,
        origin_through_day=exposure.end_day,
    )
    indices = [int(value) for value in np.flatnonzero(mask)]
    indices.sort(key=lambda index: (float(catalog.origin_day[index]), str(catalog.event_id[index])))
    return np.asarray(indices, dtype=np.int64)


def _spatial_horizon_scores(
    *,
    exposure: IssueExposure,
    delay_days: int,
    catalog: EarthquakeCatalog,
    indices: NDArray[np.int64],
    poisson: PoissonKDEPipelineResult,
) -> tuple[IssuePointProcessScore, IssuePointProcessScore]:
    final_snapshot = poisson.snapshot("final_validation")
    uniform_model = final_snapshot.uniform_model
    spatial_model = poisson.selected_kde_model("final_validation")
    uniform_validation = poisson.uniform_evidence.validation
    spatial_validation = poisson.spatial_evidence.validation
    if uniform_validation is None or spatial_validation is None:
        raise ValueError("Poisson validation evidence is incomplete")
    identifiers = tuple(str(catalog.event_id[index]) for index in indices)
    uniform_score = IssuePointProcessScore(
        protocol_sha256=poisson.protocol_sha256,
        model_id="uniform_poisson",
        model_variant_id=uniform_validation.candidate.model_variant_id,
        parameter_snapshot_id=uniform_validation.candidate.parameter_snapshot_id,
        publication_delay_days=delay_days,
        issue_date_local=exposure.issue_date_local,
        issue_time_utc=exposure.issue_time_utc,
        horizon_days=exposure.horizon_days,
        target_event_ids=identifiers,
        event_log_intensities=np.full(
            len(identifiers),
            math.log(uniform_model.rate_per_day) - math.log(uniform_model.study_area_km2),
        ),
        compensator=uniform_model.rate_per_day * exposure.horizon_days,
    )
    spatial_score = IssuePointProcessScore(
        protocol_sha256=poisson.protocol_sha256,
        model_id="spatial_poisson",
        model_variant_id=spatial_validation.candidate.model_variant_id,
        parameter_snapshot_id=spatial_validation.candidate.parameter_snapshot_id,
        publication_delay_days=delay_days,
        issue_date_local=exposure.issue_date_local,
        issue_time_utc=exposure.issue_time_utc,
        horizon_days=exposure.horizon_days,
        target_event_ids=identifiers,
        event_log_intensities=(
            spatial_model.log_density(catalog.x_km[indices], catalog.y_km[indices])
            + math.log(spatial_model.rate_per_day)
        ),
        compensator=spatial_model.rate_per_day * exposure.horizon_days,
    )
    return spatial_score, uniform_score


def _run_spatial_horizons_without_etas(
    config: BackgroundConfig,
    catalog: EarthquakeCatalog,
    calendar: FrozenIssueCalendar,
    poisson: PoissonKDEPipelineResult,
) -> tuple[PairedHorizonBacktest, ...]:
    if (
        tuple(config.time.horizons_days) != EXPECTED_HORIZONS
        or tuple(config.time.publication_delay_sensitivity_days) != EXPECTED_PUBLICATION_DELAYS
    ):
        raise ValueError("pipeline horizon/delay protocol differs from the frozen grid")
    selected_mc = poisson.snapshot("final_validation").selected_mc
    comparisons: list[PairedHorizonBacktest] = []
    for delay_days in EXPECTED_PUBLICATION_DELAYS:
        for horizon_days in EXPECTED_HORIZONS:
            pairs = []
            for exposure in calendar.validation.exposures(horizon_days):
                indices = _target_indices(catalog, exposure, selected_mc=selected_mc)
                pairs.append(
                    _spatial_horizon_scores(
                        exposure=exposure,
                        delay_days=delay_days,
                        catalog=catalog,
                        indices=indices,
                        poisson=poisson,
                    )
                )
            comparisons.append(
                PairedHorizonBacktest(
                    candidate_model_id="spatial_poisson",
                    publication_delay_days=delay_days,
                    horizon_days=horizon_days,
                    exposure_pairs=tuple(pairs),
                )
            )
    return tuple(comparisons)


def _horizon_evidence(
    config: BackgroundConfig,
    catalog: EarthquakeCatalog,
    calendar: FrozenIssueCalendar,
    grid_family: EqualAreaGridFamily,
    poisson: PoissonKDEPipelineResult,
    etas_context: _FinalETASContext | None,
    etas_skip_reason: str | None,
) -> ValidationHorizonEvidence:
    spatial = _run_spatial_horizons_without_etas(config, catalog, calendar, poisson)
    if etas_context is None:
        return ValidationHorizonEvidence(
            spatial_poisson=HorizonModelOutcome("spatial_poisson", spatial, None),
            etas=HorizonModelOutcome(
                "etas",
                None,
                etas_skip_reason or "final_validation ETAS unavailable",
            ),
            complete_backtests=None,
        )
    final_snapshot = poisson.snapshot("final_validation")
    etas_complete = run_issue_horizon_backtests(
        config,
        catalog,
        calendar,
        grid_family,
        selected_mc=final_snapshot.selected_mc,
        uniform_model=final_snapshot.uniform_model,
        spatial_model=poisson.selected_kde_model("final_validation"),
        etas_parameters=etas_context.parameters,
        etas_spec=etas_context.spec,
        etas_parameter_snapshot_id=etas_context.parameter_snapshot_id,
        etas_model_variant_id=etas_context.model_variant_id,
    )
    etas = tuple(item for item in etas_complete.comparisons if item.candidate_model_id == "etas")
    spatial_by_key = {(item.publication_delay_days, item.horizon_days): item for item in spatial}
    etas_by_key = {(item.publication_delay_days, item.horizon_days): item for item in etas}
    combined = BackgroundHorizonBacktests(
        comparisons=tuple(
            item
            for delay in EXPECTED_PUBLICATION_DELAYS
            for model_id in ("spatial_poisson", "etas")
            for horizon in EXPECTED_HORIZONS
            for item in (
                (
                    spatial_by_key[(delay, horizon)]
                    if model_id == "spatial_poisson"
                    else etas_by_key[(delay, horizon)]
                ),
            )
        )
    )
    return ValidationHorizonEvidence(
        spatial_poisson=HorizonModelOutcome("spatial_poisson", spatial, None),
        etas=HorizonModelOutcome("etas", etas, None),
        complete_backtests=combined,
    )


def _selected_evidence(
    evidence: tuple[AuditedBackgroundModelEvidence, ...],
    selection: AuditedModelSelection,
) -> AuditedBackgroundModelEvidence:
    try:
        return next(
            item
            for item in evidence
            if item.model_variant_id == selection.selected_model_variant_id
        )
    except StopIteration as error:
        raise ValueError("selected model variant is absent from audited evidence") from error


def _representative_intensity(
    config: BackgroundConfig,
    catalog: EarthquakeCatalog,
    calendar: FrozenIssueCalendar,
    grid_family: EqualAreaGridFamily,
    integration_grids: IntegrationGridEvidence,
    poisson: PoissonKDEPipelineResult,
    selected: AuditedBackgroundModelEvidence,
    etas_context: _FinalETASContext | None,
) -> RepresentativeConditionalIntensity:
    if config.time.representative_issue_date_local != _REPRESENTATIVE_ISSUE_DATE:
        raise ValueError("representative issue date differs from frozen 2025-06-26")
    issue_day, issue_time_utc = _calendar_issue(calendar, _REPRESENTATIVE_ISSUE_DATE)
    grid = grid_family.at(_PRIMARY_GRID_SIZE_KM)
    primary_grid_evidence = integration_grids.at(_PRIMARY_GRID_SIZE_KM)
    x_km = np.asarray(
        [cell.representative_x_km for cell in primary_grid_evidence.cells],
        dtype=np.float64,
    )
    y_km = np.asarray(
        [cell.representative_y_km for cell in primary_grid_evidence.cells],
        dtype=np.float64,
    )
    final_snapshot = poisson.snapshot("final_validation")
    eligible_parent_ids: tuple[str, ...] = ()
    if selected.model_id == "uniform_poisson":
        uniform = final_snapshot.uniform_model
        background = np.full(
            len(grid.cells),
            uniform.rate_per_day / uniform.study_area_km2,
            dtype=np.float64,
        )
        triggering = np.zeros(len(grid.cells), dtype=np.float64)
    elif selected.model_id == "spatial_poisson":
        spatial = poisson.selected_kde_model("final_validation")
        background = spatial.rate_per_day * spatial.density(x_km, y_km)
        triggering = np.zeros(len(grid.cells), dtype=np.float64)
    else:
        if etas_context is None:
            raise ValueError("selected ETAS model lacks a successful final parameter snapshot")
        spatial = poisson.selected_kde_model("final_validation")
        history = _issue_history(catalog, issue_day=issue_day, spec=etas_context.spec)
        fields: ETASIssueIntensityFields = evaluate_etas_issue_intensity_fields(
            etas_context.parameters,
            etas_context.spec,
            history,
            _SpatialBackgroundDensity(spatial),
            {
                cell_size: point_area_quadrature_from_grid(grid_family.at(cell_size))
                for cell_size in (50.0, 25.0, 12.5)
            },
            issue_time_days=0.0,
        )
        primary = fields.at(_PRIMARY_GRID_SIZE_KM)
        background = primary.background_intensity
        triggering = primary.triggering_intensity
        eligible_parent_ids = fields.eligible_parent_event_ids
    total = np.asarray(background + triggering, dtype=np.float64)
    render = render_conditional_intensity_figure(
        grid,
        background,
        triggering,
        issue_date=_REPRESENTATIVE_ISSUE_DATE,
        model_variant=selected.model_variant_id,
        data_cutoff=issue_time_utc,
    )
    return RepresentativeConditionalIntensity(
        issue_date_local=_REPRESENTATIVE_ISSUE_DATE,
        issue_time_utc=issue_time_utc,
        selected_model_id=selected.model_id,
        selected_model_variant_id=selected.model_variant_id,
        primary_grid_cell_size_km=_PRIMARY_GRID_SIZE_KM,
        cell_ids=tuple(cell.cell_id for cell in primary_grid_evidence.cells),
        rows=tuple(cell.row for cell in primary_grid_evidence.cells),
        columns=tuple(cell.column for cell in primary_grid_evidence.cells),
        representative_x_km=x_km,
        representative_y_km=y_km,
        clipped_area_km2=np.asarray(
            [cell.clipped_area_km2 for cell in primary_grid_evidence.cells],
            dtype=np.float64,
        ),
        eligible_parent_event_ids=eligible_parent_ids,
        background_intensity=background,
        triggering_intensity=triggering,
        total_intensity=total,
        render=render,
    )


def run_background_pipeline(
    config: BackgroundConfig,
    catalog: EarthquakeCatalog,
    study_area: StudyArea,
    grid_family: EqualAreaGridFamily,
    issue_calendar: FrozenIssueCalendar,
    *,
    production_fixture_path: Path,
    max_workers: int = DEFAULT_MAX_WORKERS,
    reserve_physical_cores: int = MINIMUM_RESERVED_PHYSICAL_CORES,
    progress: ProgressCallback | None = None,
) -> BackgroundPipelineOutcome:
    """Run the complete stage-2 production science workflow in frozen order."""

    require_background_scoring_authorized(config)
    regressions = _run_regressions(config, production_fixture_path, progress)
    _validate_spatial_inputs(config, study_area, grid_family)
    integration_grids = _integration_grid_evidence(study_area, grid_family)
    resources = _resource_evidence(
        max_workers=max_workers,
        reserve_physical_cores=reserve_physical_cores,
    )
    protocol_sha256 = hashlib.sha256(
        canonical_json_bytes(config.model_dump(mode="python"))
    ).hexdigest()
    definitions = build_snapshot_definitions(config)
    _notify(progress, "completeness:start")
    try:
        completeness = analyze_snapshot_completeness(catalog, definitions, progress=progress)
    except CompletenessScientificInability as error:
        reason = str(error)
        _notify(progress, f"completeness:failed:{reason}")
        return BackgroundPipelineFailure(
            protocol_sha256=protocol_sha256,
            failure_stage="completeness",
            failure_reason_code=error.reason_code,
            failure_reasons=(reason,),
            regressions=regressions,
            completeness=(),
            pre_score_gate_evidence=None,
            resources=resources,
            integration_grids=integration_grids,
        )
    _notify(progress, "completeness:done")
    _notify(progress, "poisson_kde:start")
    try:
        poisson = run_poisson_kde_pipeline(
            config,
            catalog,
            grid_family,
            completeness,
            progress=progress,
        )
    except PoissonKDEScientificInability as error:
        reason = str(error)
        _notify(progress, f"poisson_kde:failed:{reason}")
        return BackgroundPipelineFailure(
            protocol_sha256=protocol_sha256,
            failure_stage="poisson_kde",
            failure_reason_code=error.reason_code,
            failure_reasons=(reason,),
            regressions=regressions,
            completeness=completeness,
            pre_score_gate_evidence=error.gate_evidence,
            resources=resources,
            integration_grids=integration_grids,
        )
    _notify(progress, "poisson_kde:done")
    _notify(progress, "etas:start")
    etas = run_etas_pipeline(
        config,
        catalog,
        grid_family,
        poisson,
        completeness,
        progress=progress,
    )
    _notify(progress, "etas:done")

    audited_evidence = (
        poisson.uniform_evidence,
        poisson.spatial_evidence,
        etas.etas_evidence,
    )
    _notify(progress, "g1:start")
    g1 = assess_audited_g1(audited_evidence)
    _notify(progress, "g1:done")
    _notify(progress, "selection:start")
    selection = select_audited_background_model(audited_evidence)
    selected = _selected_evidence(audited_evidence, selection)
    _notify(progress, "selection:done")
    _notify(progress, "bootstrap:start")
    bootstrap = ValidationBootstrapEvidence(
        spatial_poisson=_bootstrap_outcome(
            "spatial_poisson",
            poisson.spatial_evidence,
            config,
        ),
        etas=_bootstrap_outcome("etas", etas.etas_evidence, config),
    )
    _notify(progress, "bootstrap:done")

    etas_context, etas_skip_reason = _final_etas_context(config, completeness, etas)
    _notify(progress, "horizons:start")
    horizons = _horizon_evidence(
        config,
        catalog,
        issue_calendar,
        grid_family,
        poisson,
        etas_context,
        etas_skip_reason,
    )
    if etas_context is None:
        _notify(progress, f"horizons:etas:skipped:{etas_skip_reason}")
    _notify(progress, "horizons:done")
    _notify(progress, "future:start")
    if etas_context is None:
        future = ETASFutureOutcome(
            ensembles=None,
            not_run_reason=etas_skip_reason or "final_validation ETAS unavailable",
        )
        _notify(progress, f"future:etas:skipped:{future.not_run_reason}")
    else:
        histories = _histories_by_validation_issue(
            catalog,
            issue_calendar,
            etas_context.spec,
        )
        future_ensembles = simulate_all_validation_issue_ensembles(
            etas_context.parameters,
            etas_context.spec,
            histories,
            poisson.selected_kde_model("final_validation"),
            study_area,
            grid_family,
            issue_calendar,
            max_workers=max_workers,
            reserve_physical_cores=reserve_physical_cores,
            physical_core_probe=lambda: resources.detected_physical_cores,
            progress=progress,
        )
        future = ETASFutureOutcome(ensembles=future_ensembles, not_run_reason=None)
    _notify(progress, "future:done")
    _notify(progress, "map:start")
    representative = _representative_intensity(
        config,
        catalog,
        issue_calendar,
        grid_family,
        integration_grids,
        poisson,
        selected,
        etas_context,
    )
    _notify(progress, "map:done")
    return BackgroundPipelineResult(
        protocol_sha256=poisson.protocol_sha256,
        regressions=regressions,
        completeness=completeness,
        poisson=poisson,
        etas=etas,
        g1=g1,
        selection=selection,
        bootstrap=bootstrap,
        horizons=horizons,
        future=future,
        representative=representative,
        resources=resources,
        integration_grids=integration_grids,
    )


__all__ = [
    "BackgroundPipelineFailure",
    "BackgroundPipelineGateError",
    "BackgroundPipelineOutcome",
    "BackgroundPipelineResult",
    "BootstrapModelOutcome",
    "ETASFutureOutcome",
    "HorizonModelOutcome",
    "IntegrationGridCellEvidence",
    "IntegrationGridEvidence",
    "IntegrationGridResolutionEvidence",
    "NumericalRegressionEvidence",
    "PipelineResourceEvidence",
    "ProgressCallback",
    "RepresentativeConditionalIntensity",
    "ScientificFailureReasonCode",
    "ValidationBootstrapEvidence",
    "ValidationHorizonEvidence",
    "run_background_pipeline",
]
