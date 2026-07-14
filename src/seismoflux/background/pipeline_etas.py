"""Pure five-snapshot orchestration for the frozen stage-2 ETAS baseline."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np

from seismoflux.background.adapters import (
    build_etas_model_spec,
    build_etas_parameter_bounds,
    build_optimizer_options,
    build_stability_thresholds,
    etas_variant_id,
    point_area_quadrature_from_grid,
)
from seismoflux.background.artifacts import canonical_json_bytes
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
    evaluate_etas_cell_expected_masses,
    fit_etas,
)
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
from seismoflux.background.pipeline_poisson import (
    PoissonKDEPipelineResult,
    PoissonSnapshotFit,
)
from seismoflux.background.poisson import SpatialPoissonModel
from seismoflux.background.workflow import (
    CompletenessSnapshot,
    SnapshotDefinition,
    build_snapshot_definitions,
    catalog_etas_events,
    physical_target_mask,
)

_GRID_SIZES_KM = (50.0, 25.0, 12.5)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _canonical_safe(value: object) -> object:
    """Represent optimizer infinities explicitly before frozen canonical hashing."""

    if isinstance(value, float):
        if math.isnan(value):
            return {"__nonfinite_float__": "nan"}
        if value == math.inf:
            return {"__nonfinite_float__": "positive_infinity"}
        if value == -math.inf:
            return {"__nonfinite_float__": "negative_infinity"}
        return value
    if isinstance(value, Mapping):
        return {str(key): _canonical_safe(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return tuple(_canonical_safe(item) for item in value)
    return value


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(canonical_json_bytes(_canonical_safe(payload))).hexdigest()


class ETASFitCallable(Protocol):
    """Injectable ETAS fitter with the exact production call contract."""

    def __call__(
        self,
        problem: ETASLikelihoodProblem,
        spec: ETASModelSpec,
        *,
        root_seed: int,
        protocol_version: str,
        model_id: str,
        bounds: ETASParameterBounds,
        options: OptimizerOptions,
        thresholds: StabilityThresholds,
    ) -> ETASFitResult: ...


@dataclass(frozen=True, slots=True)
class _KDEBackgroundDensity:
    """Scalar and batched ETAS adapter around the selected normalized KDE."""

    model: SpatialPoissonModel

    def __call__(self, x_km: float, y_km: float) -> float:
        return self.model.density_scalar(x_km, y_km)

    def density_many(self, x_km: object, y_km: object) -> object:
        return self.model.density(x_km, y_km)


@dataclass(frozen=True, slots=True)
class ETASEventSelectionAudit:
    """Causal target/history selection retained for one likelihood interval."""

    interval_start_days: float
    interval_end_days: float
    parent_history_start_days: float
    target_event_ids: tuple[str, ...]
    parent_event_ids: tuple[str, ...]
    parent_origin_times_days: tuple[float, ...]
    parent_available_times_days: tuple[float, ...]

    def __post_init__(self) -> None:
        if not (
            math.isfinite(self.parent_history_start_days)
            and math.isfinite(self.interval_start_days)
            and math.isfinite(self.interval_end_days)
            and self.parent_history_start_days <= self.interval_start_days < self.interval_end_days
        ):
            raise ValueError("ETAS selection audit has invalid interval bounds")
        target_ids = tuple(self.target_event_ids)
        parent_ids = tuple(self.parent_event_ids)
        if (
            any(not value for value in (*target_ids, *parent_ids))
            or len(set(target_ids)) != len(target_ids)
            or len(set(parent_ids)) != len(parent_ids)
        ):
            raise ValueError("ETAS selection event IDs must be non-empty and unique")
        if not set(target_ids).issubset(parent_ids):
            raise ValueError("every ETAS target must be retained in parent history")
        if not (
            len(parent_ids)
            == len(self.parent_origin_times_days)
            == len(self.parent_available_times_days)
        ):
            raise ValueError("ETAS parent selection audit vectors must align")
        for origin, available in zip(
            self.parent_origin_times_days,
            self.parent_available_times_days,
            strict=True,
        ):
            if not math.isfinite(origin) or not math.isfinite(available) or available < origin:
                raise ValueError("ETAS parent origin/availability audit is invalid")

    def parent_times(self, event_id: str) -> tuple[float, float]:
        try:
            index = self.parent_event_ids.index(event_id)
        except ValueError as error:
            raise KeyError(f"selection has no parent event {event_id!r}") from error
        return (
            self.parent_origin_times_days[index],
            self.parent_available_times_days[index],
        )


@dataclass(frozen=True, slots=True)
class ETASGridResolutionEvidence:
    """Content-addressed expected-mass evidence for one frozen grid."""

    cell_size_km: float
    cell_count: int
    background_total: float
    triggering_total: float
    total: float
    ordered_cell_masses_sha256: str

    def __post_init__(self) -> None:
        if self.cell_size_km not in _GRID_SIZES_KM:
            raise ValueError("ETAS grid evidence must use 50, 25, or 12.5 km")
        if not isinstance(self.cell_count, int) or isinstance(self.cell_count, bool):
            raise TypeError("ETAS grid cell_count must be an integer")
        if self.cell_count <= 0:
            raise ValueError("ETAS grid evidence must contain at least one cell")
        totals = (self.background_total, self.triggering_total, self.total)
        if any(not math.isfinite(value) or value < 0.0 for value in totals):
            raise ValueError("ETAS grid expected masses must be finite and non-negative")
        if not math.isclose(
            self.total,
            self.background_total + self.triggering_total,
            rel_tol=5.0e-15,
            abs_tol=1.0e-15,
        ):
            raise ValueError("ETAS grid total must equal background plus triggering")
        if _SHA256_PATTERN.fullmatch(self.ordered_cell_masses_sha256) is None:
            raise ValueError("ETAS ordered cell masses require a lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class ETASGridGateEvidence:
    """Complete 50/25/12.5 km expected-mass convergence evidence."""

    protocol_sha256: str
    snapshot_id: str
    parameter_snapshot_id: str
    resolutions: tuple[ETASGridResolutionEvidence, ...]
    convergence: ThreeGridConvergenceGateEvidence

    def __post_init__(self) -> None:
        if _SHA256_PATTERN.fullmatch(self.protocol_sha256) is None:
            raise ValueError("ETAS grid protocol fingerprint must be SHA-256")
        if self.snapshot_id not in EXPECTED_SNAPSHOTS:
            raise ValueError("ETAS grid evidence uses an unknown snapshot")
        if _SHA256_PATTERN.fullmatch(self.parameter_snapshot_id) is None:
            raise ValueError("ETAS grid evidence requires a parameter snapshot SHA-256")
        if tuple(item.cell_size_km for item in self.resolutions) != _GRID_SIZES_KM:
            raise ValueError("ETAS grid evidence must retain 50, 25, and 12.5 km in order")
        totals = {item.cell_size_km: item.total for item in self.resolutions}
        for comparison in self.convergence.comparisons:
            if not math.isclose(
                comparison.coarse_total,
                totals[comparison.coarse_cell_size_km],
                rel_tol=1.0e-14,
                abs_tol=1.0e-14,
            ) or not math.isclose(
                comparison.fine_total,
                totals[comparison.fine_cell_size_km],
                rel_tol=1.0e-14,
                abs_tol=1.0e-14,
            ):
                raise ValueError("ETAS grid summaries and convergence totals disagree")

    @property
    def passed(self) -> bool:
        return self.convergence.passed

    @property
    def failure_reasons(self) -> tuple[str, ...]:
        return tuple(
            (
                f"{item.coarse_cell_size_km:g}->{item.fine_cell_size_km:g}km "
                "convergence failed: relative_expected_count="
                f"{item.relative_expected_count_difference:.17g}, "
                f"density_l1={item.density_l1_difference:.17g}"
            )
            for item in self.convergence.comparisons
            if not item.passed
        )

    @property
    def numerical_evidence_id(self) -> str:
        return _canonical_sha256(
            {
                "protocol_sha256": self.protocol_sha256,
                "snapshot_id": self.snapshot_id,
                "parameter_snapshot_id": self.parameter_snapshot_id,
                "resolutions": tuple(asdict(item) for item in self.resolutions),
                "comparisons": tuple(asdict(item) for item in self.convergence.comparisons),
            }
        )


@dataclass(frozen=True, slots=True)
class ETASSnapshotAttempt:
    """One of the five mandatory ETAS attempts, successful or explicitly failed."""

    definition: SnapshotDefinition
    selected_mc: float
    model_variant_id: str
    fit_selection: ETASEventSelectionAudit
    score_selection: ETASEventSelectionAudit
    fit_result: ETASFitResult | None
    parameter_snapshot_id: str | None
    grid_gate_evidence: ETASGridGateEvidence | None
    paired_evidence: PairedInformationGainEvidence | None
    failure_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.definition.snapshot_id not in EXPECTED_SNAPSHOTS:
            raise ValueError("ETAS attempt uses an unknown frozen snapshot")
        if not math.isfinite(self.selected_mc) or self.selected_mc <= 0.0:
            raise ValueError("ETAS attempt selected Mc must be finite and positive")
        if not self.model_variant_id:
            raise ValueError("ETAS attempt model variant must not be empty")
        if not (
            math.isclose(
                self.score_selection.interval_start_days,
                self.definition.assessment_start_day,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
            and math.isclose(
                self.score_selection.interval_end_days,
                self.definition.assessment_end_day,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
            and math.isclose(
                self.fit_selection.interval_end_days,
                self.definition.fit_end_day,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
        ):
            raise ValueError("ETAS selection audits do not match their frozen snapshot")
        if self.parameter_snapshot_id is not None and (
            _SHA256_PATTERN.fullmatch(self.parameter_snapshot_id) is None
        ):
            raise ValueError("ETAS parameter snapshot ID must be SHA-256")
        if self.grid_gate_evidence is not None and (
            self.grid_gate_evidence.snapshot_id != self.definition.snapshot_id
            or self.grid_gate_evidence.parameter_snapshot_id != self.parameter_snapshot_id
        ):
            raise ValueError("ETAS grid evidence does not match its snapshot attempt")
        reasons = tuple(self.failure_reasons)
        if any(not reason for reason in reasons):
            raise ValueError("ETAS failure reasons must not be empty")
        if (self.paired_evidence is None) == (not reasons):
            raise ValueError("ETAS attempt must be either scored or explicitly failed")
        if self.paired_evidence is not None:
            candidate = self.paired_evidence.candidate
            if (
                candidate.snapshot_id != self.definition.snapshot_id
                or candidate.parameter_snapshot_id != self.parameter_snapshot_id
                or candidate.model_variant_id != self.model_variant_id
                or candidate.selected_mc != self.selected_mc
            ):
                raise ValueError("ETAS paired evidence does not match its attempt")
            if self.grid_gate_evidence is None or not self.grid_gate_evidence.passed:
                raise ValueError("scored ETAS attempt must have passed its complete grid gate")
        object.__setattr__(self, "failure_reasons", reasons)

    @property
    def succeeded(self) -> bool:
        return self.paired_evidence is not None


@dataclass(frozen=True, slots=True)
class ETASPipelineResult:
    """Read-only result of all five frozen ETAS fit/score attempts."""

    protocol_sha256: str
    model_variant_id: str
    attempts: tuple[ETASSnapshotAttempt, ...]
    etas_evidence: AuditedBackgroundModelEvidence

    def __post_init__(self) -> None:
        if _SHA256_PATTERN.fullmatch(self.protocol_sha256) is None:
            raise ValueError("ETAS pipeline protocol fingerprint must be SHA-256")
        if tuple(item.definition.snapshot_id for item in self.attempts) != EXPECTED_SNAPSHOTS:
            raise ValueError("ETAS pipeline must retain exactly five attempts in frozen order")
        if self.etas_evidence.model_id != "etas":
            raise ValueError("ETAS pipeline evidence has the wrong model family")
        if (
            self.etas_evidence.protocol_sha256 != self.protocol_sha256
            or self.etas_evidence.model_variant_id != self.model_variant_id
        ):
            raise ValueError("ETAS pipeline evidence does not match its protocol/model variant")
        attempted_failures = tuple(
            (item.definition.snapshot_id, item.failure_reasons)
            for item in self.attempts
            if item.failure_reasons
        )
        if attempted_failures != self.etas_evidence.failed_snapshot_reasons:
            raise ValueError("ETAS attempt failures and audited model evidence disagree")

    @property
    def evidence(self) -> AuditedBackgroundModelEvidence:
        return self.etas_evidence

    @property
    def failed_snapshot_reasons(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        return self.etas_evidence.failed_snapshot_reasons

    def attempt(self, snapshot_id: str) -> ETASSnapshotAttempt:
        for item in self.attempts:
            if item.definition.snapshot_id == snapshot_id:
                return item
        raise KeyError(f"ETAS pipeline has no snapshot {snapshot_id!r}")


def _selection_audit(
    *,
    interval_start_days: float,
    interval_end_days: float,
    parent_history_start_days: float,
    targets: tuple[ETASEvent, ...],
    parents: tuple[ETASEvent, ...],
) -> ETASEventSelectionAudit:
    return ETASEventSelectionAudit(
        interval_start_days=interval_start_days,
        interval_end_days=interval_end_days,
        parent_history_start_days=parent_history_start_days,
        target_event_ids=tuple(event.event_id for event in targets),
        parent_event_ids=tuple(event.event_id for event in parents),
        parent_origin_times_days=tuple(event.time_days for event in parents),
        parent_available_times_days=tuple(event.available_time_days for event in parents),
    )


def _uniform_scores(
    poisson_result: PoissonKDEPipelineResult,
) -> dict[str, PointProcessScoreEvidence]:
    evidence = poisson_result.uniform_evidence
    if evidence.failed_snapshot_reasons or len(evidence.development_folds) != 4:
        raise ValueError("Poisson input lacks complete five-snapshot uniform evidence")
    if evidence.validation is None:
        raise ValueError("Poisson input lacks final uniform validation evidence")
    pairs = (*evidence.development_folds, evidence.validation)
    if tuple(item.uniform.snapshot_id for item in pairs) != EXPECTED_SNAPSHOTS:
        raise ValueError("Poisson uniform evidence is not in frozen snapshot order")
    return {item.uniform.snapshot_id: item.uniform for item in pairs}


def _validate_inputs(
    config: BackgroundConfig,
    poisson_result: PoissonKDEPipelineResult,
    completeness_snapshots: tuple[CompletenessSnapshot, ...],
) -> tuple[CompletenessSnapshot, ...]:
    protocol_sha256 = _canonical_sha256(config.model_dump(mode="python"))
    if poisson_result.protocol_sha256 != protocol_sha256:
        raise ValueError("Poisson result and ETAS configuration protocol fingerprints differ")
    expected_definitions = build_snapshot_definitions(config)
    if tuple(item.definition for item in poisson_result.snapshots) != expected_definitions:
        raise ValueError("Poisson snapshots do not match the frozen ETAS definitions")
    if len(completeness_snapshots) != len(EXPECTED_SNAPSHOTS):
        raise ValueError("ETAS completeness input must contain exactly five snapshots")
    if tuple(item.definition for item in completeness_snapshots) != expected_definitions:
        raise ValueError("ETAS completeness snapshots do not match frozen definitions")
    for completeness, poisson_snapshot in zip(
        completeness_snapshots,
        poisson_result.snapshots,
        strict=True,
    ):
        analysis = completeness.analysis
        cutoff_offset = analysis.cutoff_utc.utcoffset()
        if cutoff_offset is None or cutoff_offset.total_seconds() != 0.0:
            raise ValueError("ETAS completeness cutoff must be UTC")
        if not math.isclose(
            analysis.cutoff_utc.timestamp() / 86_400.0,
            completeness.definition.fit_end_day,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError("ETAS completeness cutoff does not match snapshot fit_end")
        if analysis.selected_mc != poisson_snapshot.selected_mc:
            raise ValueError("ETAS and Poisson selected completeness magnitudes differ")
        if not math.isfinite(analysis.selected_aki_b_value) or (
            analysis.selected_aki_b_value <= 0.0
        ):
            raise ValueError("ETAS requires a finite positive selected Aki b-value")
    if config.etas.spatial_kernel.d_km2 != 25.0:
        raise ValueError("primary ETAS orchestration must keep d fixed at 25 km^2")
    return completeness_snapshots


def _event_payload(events: tuple[ETASEvent, ...]) -> tuple[dict[str, object], ...]:
    return tuple(asdict(event) for event in events)


def _parameter_snapshot_id(
    *,
    protocol_sha256: str,
    model_variant_id: str,
    definition: SnapshotDefinition,
    fit_start_utc: str,
    spec: ETASModelSpec,
    bounds: ETASParameterBounds,
    options: OptimizerOptions,
    thresholds: StabilityThresholds,
    fit_targets: tuple[ETASEvent, ...],
    fit_parents: tuple[ETASEvent, ...],
    fit_result: ETASFitResult,
    poisson_snapshot: PoissonSnapshotFit,
    background_model: SpatialPoissonModel,
) -> str:
    """Hash every fitted parameter, training event, and numerical audit."""

    return _canonical_sha256(
        {
            "protocol_sha256": protocol_sha256,
            "model_id": "etas",
            "model_variant_id": model_variant_id,
            "snapshot_id": definition.snapshot_id,
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
            "fit_result": asdict(fit_result),
            "selected_background_kde": {
                "bandwidth_km": background_model.bandwidth_km,
                "normalization_mass": background_model.normalization_mass,
                "rate_per_day": background_model.rate_per_day,
                "training_event_ids": poisson_snapshot.training_event_ids,
                "training_x_km": tuple(
                    float(value) for value in background_model.mixture.training_x_km
                ),
                "training_y_km": tuple(
                    float(value) for value in background_model.mixture.training_y_km
                ),
                "training_evidence_id": poisson_snapshot.training_evidence_id,
            },
        }
    )


def _grid_gate_evidence(
    *,
    protocol_sha256: str,
    snapshot_id: str,
    parameter_snapshot_id: str,
    problem: ETASLikelihoodProblem,
    parameters: ETASParameters,
    spec: ETASModelSpec,
    grid_family: EqualAreaGridFamily,
) -> ETASGridGateEvidence:
    masses_by_size: dict[float, dict[str, float]] = {}
    resolutions: list[ETASGridResolutionEvidence] = []
    for cell_size in _GRID_SIZES_KM:
        grid = grid_family.at(cell_size)
        quadrature = point_area_quadrature_from_grid(grid)
        masses = evaluate_etas_cell_expected_masses(problem, parameters, spec, quadrature)
        if masses.total_mass.shape != (len(grid.cells),):
            raise ValueError("ETAS expected masses do not align with their grid")
        masses_by_size[cell_size] = {
            identifier: float(value)
            for identifier, value in zip(grid.cell_ids, masses.total_mass, strict=True)
        }
        mass_sha256 = _canonical_sha256(
            {
                "cell_ids": grid.cell_ids,
                "background_mass": tuple(float(value) for value in masses.background_mass),
                "triggering_mass": tuple(float(value) for value in masses.triggering_mass),
                "total_mass": tuple(float(value) for value in masses.total_mass),
            }
        )
        resolutions.append(
            ETASGridResolutionEvidence(
                cell_size_km=cell_size,
                cell_count=len(grid.cells),
                background_total=masses.background_total,
                triggering_total=masses.triggering_total,
                total=masses.total,
                ordered_cell_masses_sha256=mass_sha256,
            )
        )
    convergence = diagnose_three_grid_convergence(grid_family, masses_by_size)
    return ETASGridGateEvidence(
        protocol_sha256=protocol_sha256,
        snapshot_id=snapshot_id,
        parameter_snapshot_id=parameter_snapshot_id,
        resolutions=tuple(resolutions),
        convergence=convergence,
    )


def _candidate_score(
    *,
    protocol_sha256: str,
    model_variant_id: str,
    definition: SnapshotDefinition,
    selected_mc: float,
    parameter_snapshot_id: str,
    problem: ETASLikelihoodProblem,
    parameters: ETASParameters,
    spec: ETASModelSpec,
    uniform_score: PointProcessScoreEvidence,
    numerical_gate_evidence_ids: tuple[str, ...],
) -> PointProcessScoreEvidence:
    likelihood = etas_log_likelihood(problem, parameters, spec)
    if set(likelihood.target_event_ids) != set(uniform_score.target_event_ids):
        raise ValueError("ETAS physical targets differ from their paired uniform targets")
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
        selected_mc=selected_mc,
        target_event_ids=uniform_score.target_event_ids,
        event_log_intensities=ordered_logs,
        compensator=likelihood.total_compensator,
        numerical_gate_evidence_ids=numerical_gate_evidence_ids,
    )


def run_etas_pipeline(
    config: BackgroundConfig,
    catalog: EarthquakeCatalog,
    grid_family: EqualAreaGridFamily,
    poisson_result: PoissonKDEPipelineResult,
    completeness_snapshots: tuple[CompletenessSnapshot, ...],
    *,
    fit_function: ETASFitCallable | None = None,
    progress: Callable[[str], None] | None = None,
) -> ETASPipelineResult:
    """Attempt, gate, and pair all five frozen ETAS snapshots without fail-fast drift."""

    snapshots_input = _validate_inputs(config, poisson_result, completeness_snapshots)
    protocol_sha256 = _canonical_sha256(config.model_dump(mode="python"))
    uniform_scores = _uniform_scores(poisson_result)
    selected_bandwidth = poisson_result.selected_bandwidth_km
    model_variant_id = etas_variant_id(
        config,
        selected_bandwidth_km=selected_bandwidth,
    )
    bounds = build_etas_parameter_bounds(config)
    options = build_optimizer_options(config)
    thresholds = build_stability_thresholds(config)
    effective_fitter = fit_etas if fit_function is None else fit_function
    score_quadrature = point_area_quadrature_from_grid(grid_family.at(12.5))
    history_start_day = utc_timestamp_to_day(config.etas.history_start_utc)

    attempts: list[ETASSnapshotAttempt] = []
    for completeness, poisson_snapshot in zip(
        snapshots_input,
        poisson_result.snapshots,
        strict=True,
    ):
        definition = completeness.definition
        if progress is not None:
            progress(f"etas:{definition.snapshot_id}:start")
        selected_mc = float(completeness.analysis.selected_mc)
        spec = build_etas_model_spec(
            config,
            selected_mc=selected_mc,
            aki_b_value=float(completeness.analysis.selected_aki_b_value),
        )
        fit_start_utc = (
            config.etas.final_fit_start_utc
            if definition.is_validation
            else config.etas.historical_fold_fit_start_utc
        )
        fit_start_day = utc_timestamp_to_day(fit_start_utc)
        fit_parent_start_day = max(
            history_start_day,
            fit_start_day - spec.history_parent_cutoff_days,
        )
        score_parent_start_day = max(
            history_start_day,
            definition.assessment_start_day - spec.history_parent_cutoff_days,
        )

        fit_target_mask = np.asarray(
            catalog.inside_study_area
            & (catalog.magnitude >= selected_mc)
            & (catalog.origin_day > fit_start_day)
            & (catalog.origin_day <= definition.fit_end_day)
            & (catalog.available_day <= definition.fit_end_day),
            dtype=np.bool_,
        )
        fit_parent_mask = np.asarray(
            catalog.inside_external_buffer
            & (catalog.magnitude >= selected_mc)
            & (catalog.origin_day >= fit_parent_start_day)
            & (catalog.origin_day <= definition.fit_end_day)
            & (catalog.available_day <= definition.fit_end_day),
            dtype=np.bool_,
        )
        score_target_mask = physical_target_mask(
            catalog,
            minimum_magnitude=selected_mc,
            origin_after_day=definition.assessment_start_day,
            origin_through_day=definition.assessment_end_day,
        )
        score_parent_mask = np.asarray(
            catalog.inside_external_buffer
            & (catalog.magnitude >= selected_mc)
            & (catalog.origin_day >= score_parent_start_day)
            & (catalog.origin_day <= definition.assessment_end_day),
            dtype=np.bool_,
        )
        fit_targets = catalog_etas_events(catalog, fit_target_mask)
        fit_parents = catalog_etas_events(catalog, fit_parent_mask)
        score_targets = catalog_etas_events(catalog, score_target_mask)
        score_parents = catalog_etas_events(catalog, score_parent_mask)
        fit_selection = _selection_audit(
            interval_start_days=fit_start_day,
            interval_end_days=definition.fit_end_day,
            parent_history_start_days=fit_parent_start_day,
            targets=fit_targets,
            parents=fit_parents,
        )
        score_selection = _selection_audit(
            interval_start_days=definition.assessment_start_day,
            interval_end_days=definition.assessment_end_day,
            parent_history_start_days=score_parent_start_day,
            targets=score_targets,
            parents=score_parents,
        )

        fit_result: ETASFitResult | None = None
        parameter_snapshot_id: str | None = None
        grid_gate: ETASGridGateEvidence | None = None
        paired: PairedInformationGainEvidence | None = None
        failure_reasons: tuple[str, ...] = ()
        background_model = poisson_result.selected_kde_model(definition.snapshot_id)
        background_density = _KDEBackgroundDensity(background_model)
        fit_problem = ETASLikelihoodProblem(
            assessment_start_days=fit_start_day,
            assessment_end_days=definition.fit_end_day,
            target_events=fit_targets,
            parent_events=fit_parents,
            background_density=background_density,
            spatial_integrator=score_quadrature,
        )
        score_problem = ETASLikelihoodProblem(
            assessment_start_days=definition.assessment_start_day,
            assessment_end_days=definition.assessment_end_day,
            target_events=score_targets,
            parent_events=score_parents,
            background_density=background_density,
            spatial_integrator=score_quadrature,
        )
        fit_result = effective_fitter(
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
        parameter_snapshot_id = _parameter_snapshot_id(
            protocol_sha256=protocol_sha256,
            model_variant_id=model_variant_id,
            definition=definition,
            fit_start_utc=fit_start_utc,
            spec=spec,
            bounds=bounds,
            options=options,
            thresholds=thresholds,
            fit_targets=fit_targets,
            fit_parents=fit_parents,
            fit_result=fit_result,
            poisson_snapshot=poisson_snapshot,
            background_model=background_model,
        )
        if not fit_result.stability.stable:
            stability_reasons = fit_result.stability.failure_reasons or (
                "ETAS fit failed the frozen numerical-stability gate",
            )
            failure_reasons = tuple(
                f"numerical stability: {reason}" for reason in stability_reasons
            )
        else:
            parameters = fit_result.best_parameters
            if parameters is None:
                raise ValueError("stable ETAS fit omitted best parameters")
            grid_gate = _grid_gate_evidence(
                protocol_sha256=protocol_sha256,
                snapshot_id=definition.snapshot_id,
                parameter_snapshot_id=parameter_snapshot_id,
                problem=score_problem,
                parameters=parameters,
                spec=spec,
                grid_family=grid_family,
            )
            if not grid_gate.passed:
                failure_reasons = grid_gate.failure_reasons or (
                    "ETAS failed the frozen three-grid convergence gate",
                )
            else:
                kde_gate_id = poisson_snapshot.gate_for(selected_bandwidth).numerical_evidence_id
                candidate = _candidate_score(
                    protocol_sha256=protocol_sha256,
                    model_variant_id=model_variant_id,
                    definition=definition,
                    selected_mc=selected_mc,
                    parameter_snapshot_id=parameter_snapshot_id,
                    problem=score_problem,
                    parameters=parameters,
                    spec=spec,
                    uniform_score=uniform_scores[definition.snapshot_id],
                    numerical_gate_evidence_ids=(
                        parameter_snapshot_id,
                        kde_gate_id,
                        grid_gate.numerical_evidence_id,
                    ),
                )
                paired = PairedInformationGainEvidence.build(
                    candidate=candidate,
                    uniform=uniform_scores[definition.snapshot_id],
                )

        attempts.append(
            ETASSnapshotAttempt(
                definition=definition,
                selected_mc=selected_mc,
                model_variant_id=model_variant_id,
                fit_selection=fit_selection,
                score_selection=score_selection,
                fit_result=fit_result,
                parameter_snapshot_id=parameter_snapshot_id,
                grid_gate_evidence=grid_gate,
                paired_evidence=paired,
                failure_reasons=failure_reasons,
            )
        )
        if progress is not None:
            if failure_reasons:
                progress(f"etas:{definition.snapshot_id}:failed:{'; '.join(failure_reasons)}")
            else:
                progress(f"etas:{definition.snapshot_id}:done")

    development = tuple(
        item.paired_evidence for item in attempts[:4] if item.paired_evidence is not None
    )
    validation = attempts[4].paired_evidence
    failed = tuple(
        (item.definition.snapshot_id, item.failure_reasons)
        for item in attempts
        if item.failure_reasons
    )
    audited = AuditedBackgroundModelEvidence(
        model_id="etas",
        model_variant_id=model_variant_id,
        protocol_sha256=protocol_sha256,
        development_folds=development,
        validation=validation,
        failed_snapshot_reasons=failed,
    )
    return ETASPipelineResult(
        protocol_sha256=protocol_sha256,
        model_variant_id=model_variant_id,
        attempts=tuple(attempts),
        etas_evidence=audited,
    )


__all__ = [
    "ETASEventSelectionAudit",
    "ETASFitCallable",
    "ETASGridGateEvidence",
    "ETASGridResolutionEvidence",
    "ETASPipelineResult",
    "ETASSnapshotAttempt",
    "run_etas_pipeline",
]
