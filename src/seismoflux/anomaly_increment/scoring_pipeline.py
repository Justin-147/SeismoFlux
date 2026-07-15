"""Pure in-memory orchestration for the frozen stage-4 scoring design.

This module intentionally has no path, dataframe, Arrow, catalogue, or locked-test
loader.  Callers must assemble fit rows, fixed-grid assessment rows, and event-to-cell
indices before entering this boundary.  Prospective inputs are a physically separate
type with no event fields.
"""

from __future__ import annotations

import dataclasses
import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, TypeAlias, cast

import numpy as np

from seismoflux.anomaly_increment.compute import Stage4WorkerPlan, stable_parallel_map
from seismoflux.anomaly_increment.contracts import (
    DesignMatrix,
    FeatureColumnContract,
    FloatArray,
    FrozenTargetRateHead,
    canonical_mapping_sha256,
    readonly_float_matrix,
    readonly_float_vector,
)
from seismoflux.anomaly_increment.deliverables import (
    ProspectivePayload,
    ProspectiveRelativeRank,
    RetrospectiveHorizonResult,
    RetrospectivePayload,
    RetrospectiveTargetCoverage,
    assert_prospective_payload_is_target_blind,
    build_interactive_results_html,
    build_static_results_svg,
    default_protocol_display,
)
from seismoflux.anomaly_increment.evaluation import (
    AdoptionDecision,
    AlarmAreaPoint,
    ConfidenceInterval,
    EventHorizonMembership,
    G2Evidence,
    G3FoldEvidence,
    GateCheck,
    GateOutcome,
    apply_preregistered_adoption_matrix,
    evaluate_g2,
    evaluate_g3,
    frozen_same_recall_budget_grid,
    information_gain_per_physical_event,
    percentile_interval,
    same_recall_union_area_relative_reduction,
    stratified_five_horizon_bootstrap_indices,
)
from seismoflux.anomaly_increment.integration import (
    composite_midpoint_quadrature,
    lead_decay,
)
from seismoflux.anomaly_increment.model import (
    PrimaryRateHeadEvidenceInsufficient,
    RidgePoissonFitResult,
    SharedObjectiveProtocol,
    conditional_intensity,
    fit_frozen_target_rate_head,
    fit_shared_ridge_poisson,
)
from seismoflux.anomaly_increment.placebo import (
    PERMUTATION_REPLICATIONS,
    InfrastructureInterruption,
    PermutationReplication,
    PermutationTestResult,
    reduce_permutation_test,
)
from seismoflux.anomaly_increment.preprocessing import (
    FrozenPreprocessor,
    fit_frozen_preprocessor,
)
from seismoflux.anomaly_increment.preregistration import (
    PRIMARY_MACRO_HORIZONS_DAYS,
    STAGE4_HORIZONS_DAYS,
    Stage4SeedContext,
)
from seismoflux.anomaly_increment.publication_diagnostics import (
    PUBLICATION_ALARM_BUDGETS_KM2,
    AlarmBudgetRecallCurve,
    AlarmBudgetRecallPoint,
    CoefficientEffectCurve,
    DataMethodFlowDiagnostics,
    DistanceLeadDecayDiagnostics,
    FoldMacroValue,
    InformationGainEvidenceStatus,
    InformationGainInterval,
    PermutationDistribution,
    PublicationDiagnostics,
    RegionHorizonMetric,
    SameRecallAreaReduction,
)
from seismoflux.anomaly_increment.scalable_model import (
    GroupedMidpointSharedPoissonObjective,
)

VariantId: TypeAlias = Literal[
    "background_no_increment",
    "coverage_only",
    "snapshot",
    "dynamic",
]
PlaceboKind: TypeAlias = Literal["time", "space"]
MagnitudeBinId: TypeAlias = Literal["M5_6", "M6_plus"]

VARIANT_ORDER: tuple[VariantId, ...] = (
    "background_no_increment",
    "coverage_only",
    "snapshot",
    "dynamic",
)
BOOTSTRAP_REPLICATIONS = 2_000
PRIMARY_ALARM_AREA_KM2 = 600_000.0


class PlaceboScientificFailure(RuntimeError):
    """One expected permutation-fit failure that contributes ``+inf`` to the null."""

    def __init__(self, failure_code: str, message: str) -> None:
        if (
            not failure_code
            or not failure_code[0].isalpha()
            or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_" for char in failure_code)
        ):
            raise ValueError("scientific failure_code must be a normalized identifier")
        super().__init__(message)
        self.failure_code = failure_code


def _default_placebo_worker_plan() -> Stage4WorkerPlan:
    return Stage4WorkerPlan(
        physical_cores=3,
        logical_processors=3,
        reserve_physical_cores=2,
        configured_max_workers=1,
        effective_workers=1,
        blas_threads_per_worker=1,
        nested_parallelism=False,
    )


def _identifier(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed identifier")
    return value


def _unique_ids(values: Sequence[str], *, label: str, allow_empty: bool = False) -> tuple[str, ...]:
    result = tuple(_identifier(value, label=label) for value in values)
    if not allow_empty and not result:
        raise ValueError(f"{label} must not be empty")
    if len(result) != len(set(result)):
        raise ValueError(f"{label} must be unique")
    return result


def _frozen_columns(
    columns: Mapping[str, object],
    *,
    label: str,
    allow_empty_rows: bool,
) -> Mapping[str, FloatArray]:
    if not isinstance(columns, Mapping) or not columns:
        raise ValueError(f"{label} must be a non-empty mapping")
    output: dict[str, FloatArray] = {}
    row_count: int | None = None
    for name, raw in columns.items():
        key = _identifier(name, label=f"{label} column")
        values = np.array(raw, dtype=np.float64, copy=True, order="C")
        if values.ndim != 1:
            raise ValueError(f"{label}.{key} must be one-dimensional")
        if not allow_empty_rows and values.size == 0:
            raise ValueError(f"{label}.{key} must not be empty")
        if row_count is None:
            row_count = int(values.size)
        elif values.size != row_count:
            raise ValueError(f"{label} columns must have one row count")
        values.setflags(write=False)
        output[key] = values
    return MappingProxyType(output)


def _column_row_count(columns: Mapping[str, FloatArray]) -> int:
    return int(next(iter(columns.values())).size)


def _frozen_counts(values: Mapping[str, int], *, label: str) -> Mapping[str, int]:
    expected = {"M5_6", "M6_plus"}
    if set(values) != expected:
        raise ValueError(f"{label} must contain exactly M5_6 and M6_plus")
    output: dict[str, int] = {}
    for key in ("M5_6", "M6_plus"):
        value = values[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{label}.{key} must be a non-negative integer")
        output[key] = value
    return MappingProxyType(output)


def _frozen_exposures(values: Mapping[str, float], *, label: str) -> Mapping[str, float]:
    if set(values) != {"M5_6", "M6_plus"}:
        raise ValueError(f"{label} must contain exactly M5_6 and M6_plus")
    output: dict[str, float] = {}
    for key in ("M5_6", "M6_plus"):
        value = float(values[key])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{label}.{key} must be finite and positive")
        output[key] = value
    return MappingProxyType(output)


@dataclass(frozen=True, slots=True)
class FeatureLayout:
    """Nested source-column sets for the three fitted increment variants."""

    coverage_sources: tuple[str, ...]
    snapshot_sources: tuple[str, ...]
    dynamic_sources: tuple[str, ...]

    def __post_init__(self) -> None:
        coverage = _unique_ids(self.coverage_sources, label="coverage_sources")
        snapshot = _unique_ids(self.snapshot_sources, label="snapshot_sources")
        dynamic = _unique_ids(self.dynamic_sources, label="dynamic_sources")
        if not set(coverage) < set(snapshot) or not set(snapshot) < set(dynamic):
            raise ValueError("feature layout must be strictly nested coverage < snapshot < dynamic")
        object.__setattr__(self, "coverage_sources", coverage)
        object.__setattr__(self, "snapshot_sources", snapshot)
        object.__setattr__(self, "dynamic_sources", dynamic)

    def sources(self, variant: VariantId) -> tuple[str, ...]:
        if variant == "coverage_only":
            return self.coverage_sources
        if variant == "snapshot":
            return self.snapshot_sources
        if variant == "dynamic":
            return self.dynamic_sources
        return ()


@dataclass(frozen=True, slots=True)
class AssembledFitScope:
    """Memory-bounded fit terms with one design row per issue-by-cell row."""

    evaluation_id: str
    preprocessing_fit_columns: Mapping[str, object]
    issue_cell_feature_columns: Mapping[str, object]
    event_feature_columns: Mapping[str, object]
    background_spatial_mass_by_row_and_bin: FloatArray
    midpoint_widths_days: FloatArray
    midpoint_decays: FloatArray
    event_background_intensity: FloatArray
    event_decay: FloatArray
    event_magnitude_bin_ids: tuple[MagnitudeBinId, ...]
    training_event_counts: Mapping[str, int]
    background_exposures: Mapping[str, float]

    def __post_init__(self) -> None:
        evaluation_id = _identifier(self.evaluation_id, label="evaluation_id")
        if evaluation_id not in {
            "development-fold-1",
            "development-fold-2",
            "development-fold-3",
            "formal-validation",
        }:
            raise ValueError("fit scope is outside the frozen stage-4 scopes")
        fit_columns = _frozen_columns(
            self.preprocessing_fit_columns,
            label="preprocessing_fit_columns",
            allow_empty_rows=False,
        )
        issue_columns = _frozen_columns(
            self.issue_cell_feature_columns,
            label="issue_cell_feature_columns",
            allow_empty_rows=False,
        )
        event_columns = _frozen_columns(
            self.event_feature_columns,
            label="event_feature_columns",
            allow_empty_rows=True,
        )
        expected_names = set(fit_columns)
        if set(issue_columns) != expected_names or set(event_columns) != expected_names:
            raise ValueError("fit, issue-cell, and event columns must share the full feature set")
        spatial_mass = readonly_float_matrix(
            "background_spatial_mass_by_row_and_bin",
            self.background_spatial_mass_by_row_and_bin,
            allow_empty_rows=False,
            allow_empty_columns=False,
        )
        midpoint_widths = readonly_float_vector(
            "midpoint_widths_days",
            self.midpoint_widths_days,
        )
        midpoint_decays = readonly_float_vector(
            "midpoint_decays",
            self.midpoint_decays,
        )
        event_background = readonly_float_vector(
            "event_background_intensity",
            self.event_background_intensity,
            allow_empty=True,
        )
        event_decay = readonly_float_vector("event_decay", self.event_decay, allow_empty=True)
        event_bins = tuple(self.event_magnitude_bin_ids)
        if spatial_mass.shape != (_column_row_count(issue_columns), 2):
            raise ValueError("fit spatial mass must have one issue-cell row and two bins")
        if np.any(spatial_mass < 0.0) or not np.any(spatial_mass > 0.0):
            raise ValueError("fit spatial mass must be non-negative with positive mass")
        if midpoint_widths.shape != midpoint_decays.shape or np.any(midpoint_widths <= 0.0):
            raise ValueError("fit midpoint widths and decays must align and be positive")
        if np.any(midpoint_decays <= 0.0) or np.any(midpoint_decays > 1.0):
            raise ValueError("fit midpoint decays must lie in (0, 1]")
        fit_horizon_days = math.fsum(float(item) for item in midpoint_widths)
        if not math.isclose(fit_horizon_days, 7.0, rel_tol=0.0, abs_tol=2.0e-13):
            raise ValueError("fit midpoint widths must partition the frozen 7-day horizon")
        event_count = _column_row_count(event_columns)
        if event_background.size != event_count or event_decay.size != event_count:
            raise ValueError("fit event numeric terms must align with event feature rows")
        if len(event_bins) != event_count or any(
            item not in {"M5_6", "M6_plus"} for item in event_bins
        ):
            raise ValueError("fit event magnitude bins must align with event rows")
        if np.any(event_background <= 0.0):
            raise ValueError("fit event background intensity must be positive")
        if np.any(event_decay <= 0.0) or np.any(event_decay > 1.0):
            raise ValueError("fit event decay must lie in (0, 1]")
        counts = _frozen_counts(self.training_event_counts, label="training_event_counts")
        if counts["M5_6"] != event_bins.count("M5_6") or counts["M6_plus"] != (
            event_bins.count("M6_plus")
        ):
            raise ValueError("training event counts must equal the assembled event rows")
        background_exposures = _frozen_exposures(
            self.background_exposures,
            label="background_exposures",
        )
        summed = np.asarray(
            np.sum(spatial_mass, axis=0, dtype=np.float64) * fit_horizon_days,
            dtype=np.float64,
        )
        for index, bin_id in enumerate(("M5_6", "M6_plus")):
            if not math.isclose(
                float(summed[index]),
                background_exposures[bin_id],
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ):
                raise ValueError("rate-head exposure must equal grouped background mass-time")
        object.__setattr__(self, "evaluation_id", evaluation_id)
        object.__setattr__(self, "preprocessing_fit_columns", fit_columns)
        object.__setattr__(self, "issue_cell_feature_columns", issue_columns)
        object.__setattr__(self, "event_feature_columns", event_columns)
        object.__setattr__(self, "background_spatial_mass_by_row_and_bin", spatial_mass)
        object.__setattr__(self, "midpoint_widths_days", midpoint_widths)
        object.__setattr__(self, "midpoint_decays", midpoint_decays)
        object.__setattr__(self, "event_background_intensity", event_background)
        object.__setattr__(self, "event_decay", event_decay)
        object.__setattr__(self, "event_magnitude_bin_ids", event_bins)
        object.__setattr__(self, "training_event_counts", counts)
        object.__setattr__(self, "background_exposures", background_exposures)


@dataclass(frozen=True, slots=True)
class AssembledEvaluationExposure:
    """Fixed-grid retrospective exposure with pre-mapped supported event cells."""

    evaluation_id: str
    issue_date: str
    horizon_days: int
    magnitude_bin: MagnitudeBinId
    support_id: str
    compensator_domain_id: str
    cell_order_ids: tuple[str, ...]
    cell_rows: tuple[int, ...]
    cell_columns: tuple[int, ...]
    cell_area_km2: FloatArray
    background_spatial_mass: FloatArray
    cell_feature_columns: Mapping[str, object]
    supported_event_ids: tuple[str, ...]
    all_study_area_event_ids: tuple[str, ...]
    event_cell_indices: tuple[int, ...]
    event_lead_days: FloatArray

    def __post_init__(self) -> None:
        evaluation_id = _identifier(self.evaluation_id, label="evaluation_id")
        issue_date = _identifier(self.issue_date, label="issue_date")
        support_id = _identifier(self.support_id, label="support_id")
        domain_id = _identifier(self.compensator_domain_id, label="compensator_domain_id")
        if self.horizon_days not in STAGE4_HORIZONS_DAYS:
            raise ValueError("evaluation horizon is outside the frozen five horizons")
        if self.magnitude_bin not in {"M5_6", "M6_plus"}:
            raise ValueError("evaluation magnitude bin is not frozen")
        cell_ids = _unique_ids(self.cell_order_ids, label="cell_order_ids")
        cell_rows = tuple(self.cell_rows)
        cell_columns = tuple(self.cell_columns)
        if len(cell_rows) != len(cell_ids) or len(cell_columns) != len(cell_ids):
            raise ValueError("evaluation cell row/column indices must align with cell IDs")
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in (*cell_rows, *cell_columns)
        ):
            raise ValueError("evaluation cell row/column indices must be signed integers")
        if len(set(zip(cell_rows, cell_columns, strict=True))) != len(cell_ids):
            raise ValueError("evaluation grid row/column pairs must be unique")
        area = readonly_float_vector("cell_area_km2", self.cell_area_km2)
        mass = readonly_float_vector("background_spatial_mass", self.background_spatial_mass)
        columns = _frozen_columns(
            self.cell_feature_columns,
            label="cell_feature_columns",
            allow_empty_rows=False,
        )
        if (
            len(cell_ids) != area.size
            or mass.size != area.size
            or _column_row_count(columns) != (area.size)
        ):
            raise ValueError("evaluation cell arrays and feature rows must align")
        if np.any(area <= 0.0):
            raise ValueError("evaluation cell areas must be positive")
        if np.any(mass < 0.0) or not np.any(mass > 0.0):
            raise ValueError("evaluation background mass must be non-negative with positive total")
        supported = _unique_ids(
            self.supported_event_ids,
            label="supported_event_ids",
            allow_empty=True,
        )
        all_events = _unique_ids(
            self.all_study_area_event_ids,
            label="all_study_area_event_ids",
            allow_empty=True,
        )
        if not set(supported) <= set(all_events):
            raise ValueError("supported events must be a subset of all study-area events")
        indices = tuple(self.event_cell_indices)
        if len(indices) != len(supported) or any(
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or index >= len(cell_ids)
            for index in indices
        ):
            raise ValueError("one valid frozen cell index is required per supported event")
        leads = readonly_float_vector("event_lead_days", self.event_lead_days, allow_empty=True)
        if (
            leads.size != len(supported)
            or np.any(leads <= 0.0)
            or np.any(leads > self.horizon_days)
        ):
            raise ValueError("event leads must lie inside the open-closed exposure window")
        event_density = (
            mass[np.asarray(indices, dtype=np.int64)] / area[np.asarray(indices, dtype=np.int64)]
        )
        if np.any(event_density <= 0.0):
            raise ValueError("supported events require positive frozen background density")
        object.__setattr__(self, "evaluation_id", evaluation_id)
        object.__setattr__(self, "issue_date", issue_date)
        object.__setattr__(self, "support_id", support_id)
        object.__setattr__(self, "compensator_domain_id", domain_id)
        object.__setattr__(self, "cell_order_ids", cell_ids)
        object.__setattr__(self, "cell_rows", cell_rows)
        object.__setattr__(self, "cell_columns", cell_columns)
        object.__setattr__(self, "cell_area_km2", area)
        object.__setattr__(self, "background_spatial_mass", mass)
        object.__setattr__(self, "cell_feature_columns", columns)
        object.__setattr__(self, "supported_event_ids", supported)
        object.__setattr__(self, "all_study_area_event_ids", all_events)
        object.__setattr__(self, "event_cell_indices", indices)
        object.__setattr__(self, "event_lead_days", leads)


@dataclass(frozen=True, slots=True)
class AssembledProspectiveIssue:
    """Physically target-free fixed-grid input for one true forecast issue."""

    issue_date: str
    horizon_days: int
    magnitude_bin: MagnitudeBinId
    cell_order_ids: tuple[str, ...]
    cell_area_km2: FloatArray
    background_spatial_mass: FloatArray
    cell_feature_columns: Mapping[str, object]

    def __post_init__(self) -> None:
        issue_date = _identifier(self.issue_date, label="prospective issue_date")
        if self.horizon_days not in STAGE4_HORIZONS_DAYS:
            raise ValueError("prospective horizon is outside the frozen five horizons")
        if self.magnitude_bin not in {"M5_6", "M6_plus"}:
            raise ValueError("prospective magnitude bin is not frozen")
        cell_ids = _unique_ids(self.cell_order_ids, label="prospective cell_order_ids")
        area = readonly_float_vector("prospective cell_area_km2", self.cell_area_km2)
        mass = readonly_float_vector(
            "prospective background_spatial_mass",
            self.background_spatial_mass,
        )
        columns = _frozen_columns(
            self.cell_feature_columns,
            label="prospective cell_feature_columns",
            allow_empty_rows=False,
        )
        if (
            len(cell_ids) != area.size
            or mass.size != area.size
            or _column_row_count(columns) != (area.size)
        ):
            raise ValueError("prospective cell inputs must align")
        if np.any(area <= 0.0) or np.any(mass < 0.0) or not np.any(mass > 0.0):
            raise ValueError("prospective areas/masses are outside the frozen domain")
        object.__setattr__(self, "issue_date", issue_date)
        object.__setattr__(self, "cell_order_ids", cell_ids)
        object.__setattr__(self, "cell_area_km2", area)
        object.__setattr__(self, "background_spatial_mass", mass)
        object.__setattr__(self, "cell_feature_columns", columns)


@dataclass(frozen=True, slots=True)
class EvaluationScope:
    fit: AssembledFitScope
    exposures: tuple[AssembledEvaluationExposure, ...]

    def __post_init__(self) -> None:
        exposures = tuple(self.exposures)
        if not exposures:
            raise ValueError("each evaluation scope requires assembled exposures")
        if any(item.evaluation_id != self.fit.evaluation_id for item in exposures):
            raise ValueError("assessment exposures may not cross their fit scope")
        keys = tuple((item.issue_date, item.horizon_days, item.magnitude_bin) for item in exposures)
        if len(keys) != len(set(keys)):
            raise ValueError("assessment exposure identities must be unique")
        object.__setattr__(self, "exposures", exposures)


@dataclass(frozen=True, slots=True)
class PlaceboRequest:
    kind: PlaceboKind
    evaluation_id: str
    model_variant: Literal["snapshot", "dynamic"]
    observed_statistic: float
    frozen_rate_head_sha256: str

    def __post_init__(self) -> None:
        if self.kind not in {"time", "space"}:
            raise ValueError("placebo kind is not frozen")
        _identifier(self.evaluation_id, label="placebo evaluation_id")
        if self.model_variant not in {"snapshot", "dynamic"}:
            raise ValueError("placebo model variant is not frozen")
        if not math.isfinite(self.observed_statistic):
            raise ValueError("placebo observed_statistic must be finite")
        if len(self.frozen_rate_head_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.frozen_rate_head_sha256
        ):
            raise ValueError("placebo frozen rate-head identity must be a lowercase SHA-256")

    def as_mapping(self) -> dict[str, object]:
        return {
            "evaluation_id": self.evaluation_id,
            "frozen_rate_head_sha256": self.frozen_rate_head_sha256,
            "kind": self.kind,
            "model_variant": self.model_variant,
            "observed_statistic_hex": self.observed_statistic.hex(),
        }

    @property
    def content_sha256(self) -> str:
        return canonical_mapping_sha256(self.as_mapping())


@dataclass(frozen=True, slots=True)
class PlaceboReplicateInput:
    """One target permutation assembled by the caller, never a supplied statistic."""

    replication_index: int
    mapping_sha256: str
    fit_scope: AssembledFitScope
    exposures: tuple[AssembledEvaluationExposure, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.replication_index, int)
            or isinstance(self.replication_index, bool)
            or self.replication_index < 0
        ):
            raise ValueError("placebo replication_index must be a non-negative integer")
        if len(self.mapping_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.mapping_sha256
        ):
            raise ValueError("placebo mapping_sha256 must be a lowercase SHA-256")
        exposures = tuple(self.exposures)
        if not exposures or any(
            item.evaluation_id != self.fit_scope.evaluation_id for item in exposures
        ):
            raise ValueError("placebo exposures must be non-empty and match the fit scope")
        keys = tuple((item.issue_date, item.horizon_days, item.magnitude_bin) for item in exposures)
        if len(keys) != len(set(keys)):
            raise ValueError("placebo exposure identities must be unique")
        object.__setattr__(self, "exposures", exposures)


PlaceboReplicateFactory: TypeAlias = Callable[[int], PlaceboReplicateInput]
PlaceboMappingSha256Factory: TypeAlias = Callable[[int], str]


@dataclass(frozen=True, slots=True)
class PlaceboSource:
    """A request-bound source that constructs exactly one replication on demand."""

    source_id_sha256: str
    frozen_rate_head_sha256: str
    replicate_factory: PlaceboReplicateFactory = field(repr=False, compare=False)
    mapping_sha256_factory: PlaceboMappingSha256Factory = field(repr=False, compare=False)
    replication_count: int = PERMUTATION_REPLICATIONS

    def validate_for(self, request: PlaceboRequest) -> None:
        for label, value in (
            ("source_id_sha256", self.source_id_sha256),
            ("frozen_rate_head_sha256", self.frozen_rate_head_sha256),
        ):
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"placebo {label} must be a lowercase SHA-256")
        if self.replication_count != PERMUTATION_REPLICATIONS:
            raise ValueError("placebo source must expose exactly replications 0..999")
        if not callable(self.replicate_factory):
            raise TypeError("placebo replicate_factory must be callable")
        if not callable(self.mapping_sha256_factory):
            raise TypeError("placebo mapping_sha256_factory must be callable")
        if self.frozen_rate_head_sha256 != request.frozen_rate_head_sha256:
            raise ValueError("placebo source changed the frozen target-rate head")

    def _validate_index(self, replication_index: int) -> None:
        if (
            not isinstance(replication_index, int)
            or isinstance(replication_index, bool)
            or not 0 <= replication_index < self.replication_count
        ):
            raise ValueError("placebo source index is outside 0..999")

    def mapping_sha256(self, request: PlaceboRequest, replication_index: int) -> str:
        self.validate_for(request)
        self._validate_index(replication_index)
        value = self.mapping_sha256_factory(replication_index)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)
        ):
            raise InfrastructureInterruption(
                "placebo mapping identity factory returned a malformed SHA-256"
            )
        return value

    def build(self, request: PlaceboRequest, replication_index: int) -> PlaceboReplicateInput:
        self._validate_index(replication_index)
        expected_mapping_sha256 = self.mapping_sha256(request, replication_index)
        item = self.replicate_factory(replication_index)
        if not isinstance(item, PlaceboReplicateInput):
            raise TypeError("placebo source must build PlaceboReplicateInput")
        if item.replication_index != replication_index:
            raise InfrastructureInterruption(
                "placebo source returned a different replication index"
            )
        if item.fit_scope.evaluation_id != request.evaluation_id:
            raise InfrastructureInterruption(
                "placebo replication crossed its frozen evaluation scope"
            )
        if item.mapping_sha256 != expected_mapping_sha256:
            raise InfrastructureInterruption(
                "placebo replicate mapping differs from its lightweight identity"
            )
        return item


@dataclass(frozen=True, slots=True)
class CompletedPlaceboReplication:
    """One ordered, checkpoint-safe permutation result with its mapping identity."""

    replication_index: int
    mapping_sha256: str
    statistic: float | None
    converged: bool
    scientific_failure_code: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.replication_index, int)
            or isinstance(self.replication_index, bool)
            or not 0 <= self.replication_index < PERMUTATION_REPLICATIONS
        ):
            raise ValueError("completed placebo index is outside 0..999")
        if len(self.mapping_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.mapping_sha256
        ):
            raise ValueError("completed placebo mapping identity must be a lowercase SHA-256")
        if self.converged:
            if self.statistic is None or not math.isfinite(self.statistic):
                raise ValueError("converged placebo result requires one finite statistic")
            if self.scientific_failure_code is not None:
                raise ValueError("converged placebo result cannot have a failure code")
        else:
            if self.statistic is not None:
                raise ValueError("failed placebo result must not retain a statistic")
            code = self.scientific_failure_code
            if (
                code is None
                or not code
                or not code[0].isalpha()
                or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_" for char in code)
            ):
                raise ValueError("failed placebo result requires a normalized failure code")

    def as_mapping(self) -> dict[str, object]:
        return {
            "converged": self.converged,
            "mapping_sha256": self.mapping_sha256,
            "replication_index": self.replication_index,
            "scientific_failure_code": self.scientific_failure_code,
            "statistic_hex": None if self.statistic is None else self.statistic.hex(),
        }

    @property
    def content_sha256(self) -> str:
        return canonical_mapping_sha256(self.as_mapping())


PlaceboResultCallback: TypeAlias = Callable[[PlaceboRequest, CompletedPlaceboReplication], None]
PlaceboCompleteCallback: TypeAlias = Callable[
    [PlaceboRequest, tuple[CompletedPlaceboReplication, ...]], None
]
PlaceboInterruptionCallback: TypeAlias = Callable[[PlaceboRequest, BaseException], None]


@dataclass(frozen=True, slots=True)
class PlaceboExecution:
    """Lazy source, recovered prefix, bounded workers, and runtime-only callbacks."""

    source: PlaceboSource
    completed_results: tuple[CompletedPlaceboReplication, ...] = ()
    worker_plan: Stage4WorkerPlan = field(default_factory=_default_placebo_worker_plan)
    max_in_flight: int = 1
    on_result: PlaceboResultCallback | None = field(default=None, repr=False, compare=False)
    on_complete: PlaceboCompleteCallback | None = field(default=None, repr=False, compare=False)
    on_interruption: PlaceboInterruptionCallback | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def validate_for(self, request: PlaceboRequest) -> None:
        if not isinstance(self.source, PlaceboSource):
            raise TypeError("placebo execution requires a PlaceboSource")
        self.source.validate_for(request)
        results = tuple(self.completed_results)
        if tuple(item.replication_index for item in results) != tuple(range(len(results))):
            raise InfrastructureInterruption(
                "recovered placebo results must be one contiguous prefix"
            )
        if len(results) > PERMUTATION_REPLICATIONS:
            raise InfrastructureInterruption("recovered placebo prefix is too long")
        workers = self.worker_plan
        if (
            workers.reserve_physical_cores < 2
            or workers.configured_max_workers > 12
            or workers.blas_threads_per_worker != 1
            or workers.nested_parallelism
        ):
            raise ValueError("placebo execution violates the frozen worker policy")
        if (
            not isinstance(self.max_in_flight, int)
            or isinstance(self.max_in_flight, bool)
            or not 1 <= self.max_in_flight <= workers.effective_workers
        ):
            raise ValueError("placebo max_in_flight must be within the worker count")
        for callback in (self.on_result, self.on_complete, self.on_interruption):
            if callback is not None and not callable(callback):
                raise TypeError("placebo execution callback must be callable")
        object.__setattr__(self, "completed_results", results)


PlaceboInjection: TypeAlias = Callable[[PlaceboRequest], PlaceboExecution]


@dataclass(frozen=True, slots=True)
class FittedVariant:
    variant: VariantId
    design_column_indices: tuple[int, ...]
    beta: FloatArray
    fit_result: RidgePoissonFitResult | None

    def __post_init__(self) -> None:
        indices = tuple(self.design_column_indices)
        if any(index < 0 for index in indices) or len(indices) != len(set(indices)):
            raise ValueError("fitted design column indices must be unique and non-negative")
        beta = readonly_float_vector("fitted beta", self.beta, allow_empty=True)
        if beta.size != len(indices):
            raise ValueError("fitted beta must align with selected design columns")
        if self.variant == "background_no_increment":
            if indices or beta.size or self.fit_result is not None:
                raise ValueError("background variant must bypass every increment term")
        elif not indices:
            raise ValueError("fitted increment variant must retain design columns")
        object.__setattr__(self, "design_column_indices", indices)
        object.__setattr__(self, "beta", beta)


@dataclass(frozen=True, slots=True)
class FittedScope:
    evaluation_id: str
    preprocessor: FrozenPreprocessor
    rate_head: FrozenTargetRateHead
    variants: tuple[FittedVariant, ...]
    primary_fit_evidence_insufficient: bool
    optimizer_run_count: int

    def __post_init__(self) -> None:
        variants = tuple(self.variants)
        if tuple(item.variant for item in variants) != VARIANT_ORDER:
            raise ValueError("fitted scope must retain all four variants in frozen order")
        expected_runs = sum(item.fit_result is not None for item in variants)
        if self.optimizer_run_count != expected_runs:
            raise ValueError("optimizer_run_count disagrees with fitted variants")
        if self.primary_fit_evidence_insufficient and expected_runs != 0:
            raise ValueError("zero-event M5_6 scope must not run any optimizer")
        object.__setattr__(self, "variants", variants)

    def variant(self, variant: VariantId) -> FittedVariant:
        return next(item for item in self.variants if item.variant == variant)


@dataclass(frozen=True, slots=True)
class ExposureVariantScore:
    evaluation_id: str
    issue_date: str
    horizon_days: int
    magnitude_bin: MagnitudeBinId
    variant: VariantId
    supported_event_ids: tuple[str, ...]
    all_study_area_event_ids: tuple[str, ...]
    event_log_intensities: tuple[float, ...]
    integrated_intensity: float
    cell_integrated_intensities: tuple[float, ...]
    alarm_exact_selected_area_km2: FloatArray
    alarm_prefix_cell_counts: tuple[int, ...]
    supported_event_alarm_ranks: tuple[int, ...]
    selected_cell_ids_at_600000_km2: tuple[str, ...]
    strict_hit_event_ids_at_600000_km2: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RefitPrimaryMacroResult:
    statistic: float | None
    frozen_rate_head_sha256: str
    fitted_preprocessor_sha256: str
    anomaly_fit_sha256: str | None

    def __post_init__(self) -> None:
        if self.statistic is not None and not math.isfinite(self.statistic):
            raise ValueError("placebo primary macro statistic must be finite when available")
        for label, value in (
            ("frozen_rate_head_sha256", self.frozen_rate_head_sha256),
            ("fitted_preprocessor_sha256", self.fitted_preprocessor_sha256),
        ):
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"{label} must be a lowercase SHA-256")
        if self.anomaly_fit_sha256 is not None and (
            len(self.anomaly_fit_sha256) != 64
            or any(char not in "0123456789abcdef" for char in self.anomaly_fit_sha256)
        ):
            raise ValueError("anomaly_fit_sha256 must be a lowercase SHA-256 when available")


@dataclass(frozen=True, slots=True)
class PipelineResult:
    fitted_scopes: tuple[FittedScope, ...]
    exposure_scores: tuple[ExposureVariantScore, ...]
    dynamic_g2: GateOutcome
    dynamic_g3: GateOutcome
    snapshot_equivalent_g2: GateOutcome
    adoption: AdoptionDecision
    dynamic_time_permutation: PermutationTestResult | None
    dynamic_space_permutation: PermutationTestResult | None
    retrospective: RetrospectivePayload
    prospective: ProspectivePayload
    publication_diagnostics: PublicationDiagnostics
    static_svg: str
    interactive_html: str
    result_fingerprint_sha256: str


@dataclass(frozen=True, slots=True)
class EvaluationRegionBinding:
    """Local-only evaluation strata; forbidden from every model and forecast input."""

    all_construction_zone_ids: tuple[str, ...]
    cell_ids: tuple[str, ...]
    cell_construction_zone_ids: tuple[str, ...]
    event_ids: tuple[str, ...]
    event_construction_zone_ids: tuple[str, ...]
    cell_mapping_sha256: str
    event_mapping_sha256: str
    diagnostic_role: Literal["evaluation-only-no-fit-feature-order-or-candidate"] = (
        "evaluation-only-no-fit-feature-order-or-candidate"
    )
    public_export_forbidden: Literal[True] = True

    def __post_init__(self) -> None:
        zones = _unique_ids(
            self.all_construction_zone_ids,
            label="all_construction_zone_ids",
        )
        if len(zones) != 39:
            raise ValueError("stage-4 regional diagnostics require exactly 39 frozen zones")
        cells = _unique_ids(self.cell_ids, label="evaluation region cell_ids")
        cell_zones = tuple(self.cell_construction_zone_ids)
        events = _unique_ids(
            self.event_ids,
            label="evaluation region event_ids",
            allow_empty=True,
        )
        event_zones = tuple(self.event_construction_zone_ids)
        if len(cell_zones) != len(cells) or len(event_zones) != len(events):
            raise ValueError("evaluation region bindings require one zone per cell and event")
        zone_set = set(zones)
        if any(zone_id not in zone_set for zone_id in (*cell_zones, *event_zones)):
            raise ValueError("evaluation region binding referenced an unknown construction zone")
        for label, value in (
            ("cell_mapping_sha256", self.cell_mapping_sha256),
            ("event_mapping_sha256", self.event_mapping_sha256),
        ):
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"{label} must be a lowercase SHA-256")
        if self.diagnostic_role != "evaluation-only-no-fit-feature-order-or-candidate":
            raise ValueError("regional binding escaped its evaluation-only role")
        if self.public_export_forbidden is not True:
            raise ValueError("raw construction-zone bindings must never be public")
        object.__setattr__(self, "all_construction_zone_ids", zones)
        object.__setattr__(self, "cell_ids", cells)
        object.__setattr__(self, "cell_construction_zone_ids", cell_zones)
        object.__setattr__(self, "event_ids", events)
        object.__setattr__(self, "event_construction_zone_ids", event_zones)

    @property
    def cell_zone_by_id(self) -> Mapping[str, str]:
        return MappingProxyType(
            dict(zip(self.cell_ids, self.cell_construction_zone_ids, strict=True))
        )

    @property
    def event_zone_by_id(self) -> Mapping[str, str]:
        return MappingProxyType(
            dict(zip(self.event_ids, self.event_construction_zone_ids, strict=True))
        )


@dataclass(frozen=True, slots=True)
class Stage4InMemoryPlan:
    feature_contracts: tuple[FeatureColumnContract, ...]
    feature_layout: FeatureLayout
    development_scopes: tuple[EvaluationScope, ...]
    formal_scope: EvaluationScope
    prospective_issues: tuple[AssembledProspectiveIssue, ...]
    evaluation_region_binding: EvaluationRegionBinding
    frozen_input_seal_sha256: str
    model_version: str

    def __post_init__(self) -> None:
        contracts = tuple(self.feature_contracts)
        if not contracts:
            raise ValueError("pipeline requires frozen feature contracts")
        sources = tuple(item.source_column for item in contracts)
        if (
            len(sources) != len(set(sources))
            or tuple(self.feature_layout.dynamic_sources) != sources
        ):
            raise ValueError("dynamic feature layout must exactly match contract source order")
        development = tuple(self.development_scopes)
        if tuple(item.fit.evaluation_id for item in development) != (
            "development-fold-1",
            "development-fold-2",
            "development-fold-3",
        ):
            raise ValueError("pipeline requires the three development folds in frozen order")
        if self.formal_scope.fit.evaluation_id != "formal-validation":
            raise ValueError("formal scope identity must be formal-validation")
        prospective = tuple(self.prospective_issues)
        if not prospective:
            raise ValueError("pipeline requires physically separate prospective inputs")
        binding = self.evaluation_region_binding
        if not isinstance(binding, EvaluationRegionBinding):
            raise TypeError("pipeline requires a frozen evaluation-only region binding")
        formal_cells = {
            cell_id
            for exposure in self.formal_scope.exposures
            for cell_id in exposure.cell_order_ids
        }
        missing_cells = formal_cells.difference(binding.cell_ids)
        if missing_cells:
            raise ValueError("regional cell receipt does not cover every formal exposure cell")
        formal_events = {
            event_id
            for exposure in self.formal_scope.exposures
            for event_id in exposure.all_study_area_event_ids
        }
        missing_events = formal_events.difference(binding.event_ids)
        if missing_events:
            raise ValueError("regional event receipt does not cover every formal target event")
        if len(self.frozen_input_seal_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.frozen_input_seal_sha256
        ):
            raise ValueError("frozen_input_seal_sha256 must be a lowercase SHA-256")
        _identifier(self.model_version, label="model_version")
        object.__setattr__(self, "feature_contracts", contracts)
        object.__setattr__(self, "development_scopes", development)
        object.__setattr__(self, "prospective_issues", prospective)


def _select_design(full: DesignMatrix, indices: tuple[int, ...]) -> DesignMatrix:
    index = np.asarray(indices, dtype=np.int64)
    return DesignMatrix(
        values=full.values[:, index],
        column_names=tuple(full.column_names[item] for item in indices),
        penalty_factors=full.penalty_factors[index],
        active_coefficients=full.active_coefficients[index],
    )


def _variant_indices(
    preprocessor: FrozenPreprocessor,
    layout: FeatureLayout,
    variant: VariantId,
) -> tuple[int, ...]:
    sources = set(layout.sources(variant))
    return tuple(
        index
        for feature_index, contract in enumerate(preprocessor.contracts)
        if contract.source_column in sources
        for index in (feature_index * 2, feature_index * 2 + 1)
    )


def _fit_scope(
    scope: AssembledFitScope,
    contracts: tuple[FeatureColumnContract, ...],
    layout: FeatureLayout,
) -> FittedScope:
    preprocessor = fit_frozen_preprocessor(contracts, scope.preprocessing_fit_columns)
    issue_full = preprocessor.transform(scope.issue_cell_feature_columns)
    event_full = preprocessor.transform(scope.event_feature_columns)
    rate_head = fit_frozen_target_rate_head(
        training_event_counts=scope.training_event_counts,
        background_exposures=scope.background_exposures,
    )
    variants = [
        FittedVariant(
            variant="background_no_increment",
            design_column_indices=(),
            beta=np.asarray([], dtype=np.float64),
            fit_result=None,
        )
    ]
    increment_variants: tuple[Literal["coverage_only", "snapshot", "dynamic"], ...] = (
        "coverage_only",
        "snapshot",
        "dynamic",
    )
    for variant in increment_variants:
        variants.append(
            _fit_increment_variant(
                scope=scope,
                preprocessor=preprocessor,
                issue_full=issue_full,
                event_full=event_full,
                rate_head=rate_head,
                layout=layout,
                variant=variant,
            )
        )
    return FittedScope(
        evaluation_id=scope.evaluation_id,
        preprocessor=preprocessor,
        rate_head=rate_head,
        variants=tuple(variants),
        primary_fit_evidence_insufficient=rate_head.primary_evidence_insufficient,
        optimizer_run_count=sum(item.fit_result is not None for item in variants),
    )


def _fit_increment_variant(
    *,
    scope: AssembledFitScope,
    preprocessor: FrozenPreprocessor,
    issue_full: DesignMatrix,
    event_full: DesignMatrix,
    rate_head: FrozenTargetRateHead,
    layout: FeatureLayout,
    variant: Literal["coverage_only", "snapshot", "dynamic"],
) -> FittedVariant:
    indices = _variant_indices(preprocessor, layout, variant)
    if rate_head.primary_evidence_insufficient:
        return FittedVariant(
            variant=variant,
            design_column_indices=indices,
            beta=np.zeros(len(indices), dtype=np.float64),
            fit_result=None,
        )
    objective = GroupedMidpointSharedPoissonObjective(
        issue_design=_select_design(issue_full, indices),
        background_spatial_mass_by_row_and_bin=scope.background_spatial_mass_by_row_and_bin,
        midpoint_widths_days=scope.midpoint_widths_days,
        midpoint_decays=scope.midpoint_decays,
        event_design=_select_design(event_full, indices),
        event_background_intensity=scope.event_background_intensity,
        event_decay=scope.event_decay,
        event_magnitude_bin_ids=scope.event_magnitude_bin_ids,
        rate_head=rate_head,
    )
    try:
        fit_result = fit_shared_ridge_poisson(cast(SharedObjectiveProtocol, objective))
    except PrimaryRateHeadEvidenceInsufficient as error:  # pragma: no cover - prechecked
        raise AssertionError("primary rate-head state changed during variant fitting") from error
    if not fit_result.converged:
        raise PlaceboScientificFailure(
            "optimizer_nonconvergence",
            f"frozen anomaly model did not converge: {scope.evaluation_id}/{variant}",
        )
    return FittedVariant(
        variant=variant,
        design_column_indices=indices,
        beta=fit_result.beta,
        fit_result=fit_result,
    )


def _cell_integrated_intensities(
    *,
    background_mass: FloatArray,
    linear_predictor: FloatArray,
    rate_multiplier: float,
    horizon_days: int,
) -> FloatArray:
    if rate_multiplier == 0.0:
        result = np.zeros(background_mass.size, dtype=np.float64)
        result.setflags(write=False)
        return result
    if not np.any(linear_predictor):
        result = np.asarray(background_mass * rate_multiplier * horizon_days, dtype=np.float64)
        result.setflags(write=False)
        return result
    quadrature = composite_midpoint_quadrature(float(horizon_days))
    temporal_multiplier = np.zeros(linear_predictor.size, dtype=np.float64)
    for midpoint, width in zip(
        quadrature.lead_midpoints_days,
        quadrature.widths_days,
        strict=True,
    ):
        decay = float(lead_decay(float(midpoint)))
        with np.errstate(over="raise", invalid="raise"):
            temporal_multiplier += np.exp(decay * linear_predictor) * float(width)
    result = np.asarray(background_mass * rate_multiplier * temporal_multiplier, dtype=np.float64)
    if not np.isfinite(result).all() or np.any(result < 0.0):
        raise FloatingPointError("cell integrated intensity is non-finite")
    result.setflags(write=False)
    return result


def _ordered_cell_indices(
    intensity: FloatArray,
    cell_ids: tuple[str, ...],
    cell_rows: tuple[int, ...],
    cell_columns: tuple[int, ...],
) -> tuple[int, ...]:
    return tuple(
        sorted(
            range(intensity.size),
            key=lambda item: (
                -float(intensity[item]),
                cell_rows[item],
                cell_columns[item],
                cell_ids[item],
            ),
        )
    )


def _alarm_budget_profile(
    intensity: FloatArray,
    area: FloatArray,
    cell_ids: tuple[str, ...],
    cell_rows: tuple[int, ...],
    cell_columns: tuple[int, ...],
    event_cell_indices: tuple[int, ...],
) -> tuple[tuple[int, ...], FloatArray, tuple[int, ...], tuple[int, ...]]:
    """Build every frozen complete-prefix alarm area without consulting targets for order."""

    order = _ordered_cell_indices(intensity, cell_ids, cell_rows, cell_columns)
    ordered = np.asarray(order, dtype=np.int64)
    cumulative = np.cumsum(area[ordered], dtype=np.float64)
    budgets = np.asarray(frozen_same_recall_budget_grid(), dtype=np.float64)
    prefix_counts = np.searchsorted(cumulative, budgets + 1.0e-9, side="right")
    exact_area = np.zeros(budgets.size, dtype=np.float64)
    positive = prefix_counts > 0
    exact_area[positive] = cumulative[prefix_counts[positive] - 1]
    if np.any(exact_area > budgets + 1.0e-9) or np.any(np.diff(exact_area) < 0.0):
        raise AssertionError("alarm-area complete-prefix profile is invalid")
    exact_area.setflags(write=False)

    rank_by_cell = np.empty(intensity.size, dtype=np.int64)
    rank_by_cell[ordered] = np.arange(intensity.size, dtype=np.int64)
    event_ranks = tuple(int(rank_by_cell[index]) for index in event_cell_indices)
    primary_index = frozen_same_recall_budget_grid().index(int(PRIMARY_ALARM_AREA_KM2))
    primary_count = int(prefix_counts[primary_index])
    return (
        order[:primary_count],
        exact_area,
        tuple(int(value) for value in prefix_counts),
        event_ranks,
    )


def _score_exposure(
    exposure: AssembledEvaluationExposure,
    fitted: FittedScope,
) -> tuple[ExposureVariantScore, ...]:
    if exposure.evaluation_id != fitted.evaluation_id:
        raise ValueError("exposure cannot be scored by another fitted scope")
    full_design = fitted.preprocessor.transform(exposure.cell_feature_columns)
    return tuple(
        _score_exposure_variant(
            exposure,
            full_design=full_design,
            rate_head=fitted.rate_head,
            variant=variant,
        )
        for variant in fitted.variants
    )


def _score_exposure_variant(
    exposure: AssembledEvaluationExposure,
    *,
    full_design: DesignMatrix,
    rate_head: FrozenTargetRateHead,
    variant: FittedVariant,
) -> ExposureVariantScore:
    event_index = np.asarray(exposure.event_cell_indices, dtype=np.int64)
    background_density = exposure.background_spatial_mass / exposure.cell_area_km2
    if variant.variant == "background_no_increment":
        linear = np.zeros(full_design.row_count, dtype=np.float64)
        event_design = np.zeros((event_index.size, 1), dtype=np.float64)
        beta = np.zeros(1, dtype=np.float64)
    else:
        design = _select_design(full_design, variant.design_column_indices)
        linear = np.asarray(design.values @ variant.beta, dtype=np.float64)
        event_design = design.values[event_index, :]
        beta = variant.beta
    rate = rate_head.by_id(exposure.magnitude_bin).rate_multiplier
    cell_integrated = _cell_integrated_intensities(
        background_mass=exposure.background_spatial_mass,
        linear_predictor=linear,
        rate_multiplier=rate,
        horizon_days=exposure.horizon_days,
    )
    if event_index.size:
        event_intensity = conditional_intensity(
            background_intensity=background_density[event_index],
            design_values=event_design,
            beta=beta,
            lead_days=exposure.event_lead_days,
            magnitude_bin_id=exposure.magnitude_bin,
            rate_head=rate_head,
            increment_enabled=variant.variant != "background_no_increment",
        )
        if np.any(event_intensity <= 0.0):
            raise ValueError("supported event intensity must be positive")
        event_logs = tuple(float(math.log(value)) for value in event_intensity)
    else:
        event_logs = ()
    selected_indices, exact_alarm_areas, alarm_prefix_counts, event_alarm_ranks = (
        _alarm_budget_profile(
            cell_integrated,
            exposure.cell_area_km2,
            exposure.cell_order_ids,
            exposure.cell_rows,
            exposure.cell_columns,
            exposure.event_cell_indices,
        )
    )
    selected_set = set(selected_indices)
    hits = tuple(
        event_id
        for event_id, index in zip(
            exposure.supported_event_ids,
            exposure.event_cell_indices,
            strict=True,
        )
        if index in selected_set
    )
    return ExposureVariantScore(
        evaluation_id=exposure.evaluation_id,
        issue_date=exposure.issue_date,
        horizon_days=exposure.horizon_days,
        magnitude_bin=exposure.magnitude_bin,
        variant=variant.variant,
        supported_event_ids=exposure.supported_event_ids,
        all_study_area_event_ids=exposure.all_study_area_event_ids,
        event_log_intensities=event_logs,
        integrated_intensity=math.fsum(float(value) for value in cell_integrated),
        cell_integrated_intensities=tuple(float(value) for value in cell_integrated),
        alarm_exact_selected_area_km2=exact_alarm_areas,
        alarm_prefix_cell_counts=alarm_prefix_counts,
        supported_event_alarm_ranks=event_alarm_ranks,
        selected_cell_ids_at_600000_km2=tuple(
            exposure.cell_order_ids[index] for index in selected_indices
        ),
        strict_hit_event_ids_at_600000_km2=hits,
    )


@dataclass(frozen=True, slots=True)
class _Aggregate:
    event_ids: tuple[str, ...]
    all_event_ids: tuple[str, ...]
    event_log_by_id: Mapping[str, float]
    integrated_intensity: float
    hit_ids: frozenset[str]


def _aggregate(
    scores: Sequence[ExposureVariantScore],
    *,
    variant: VariantId,
    horizon: int,
    magnitude_bin: MagnitudeBinId,
) -> _Aggregate:
    selected = tuple(
        item
        for item in scores
        if item.variant == variant
        and item.horizon_days == horizon
        and item.magnitude_bin == magnitude_bin
    )
    event_ids = tuple(event for item in selected for event in item.supported_event_ids)
    all_ids = tuple(event for item in selected for event in item.all_study_area_event_ids)
    if len(event_ids) != len(set(event_ids)) or len(all_ids) != len(set(all_ids)):
        raise ValueError("nonoverlapping assessment exposures duplicated a physical event")
    log_by_id = {
        event_id: value
        for item in selected
        for event_id, value in zip(
            item.supported_event_ids,
            item.event_log_intensities,
            strict=True,
        )
    }
    return _Aggregate(
        event_ids=event_ids,
        all_event_ids=all_ids,
        event_log_by_id=MappingProxyType(log_by_id),
        integrated_intensity=math.fsum(item.integrated_intensity for item in selected),
        hit_ids=frozenset(
            event for item in selected for event in item.strict_hit_event_ids_at_600000_km2
        ),
    )


def _gain(candidate: _Aggregate, comparator: _Aggregate) -> float | None:
    if candidate.event_ids != comparator.event_ids:
        raise ValueError("information-gain variants must share physical events")
    result: float | None = information_gain_per_physical_event(
        event_ids=candidate.event_ids,
        candidate_event_log_intensities=tuple(
            candidate.event_log_by_id[item] for item in candidate.event_ids
        ),
        comparator_event_log_intensities=tuple(
            comparator.event_log_by_id[item] for item in comparator.event_ids
        ),
        candidate_integrated_intensity=candidate.integrated_intensity,
        comparator_integrated_intensity=comparator.integrated_intensity,
    )
    return result


@dataclass(frozen=True, slots=True)
class _SameRecallMacro:
    value: float | None
    bootstrap_numeric_value: float
    evaluable: bool


@dataclass(frozen=True, slots=True)
class _SameRecallHorizon:
    target_recall: float | None
    comparator_area_km2: float | None
    candidate_area_km2: float | None
    area_reduction_fraction: float | None
    evaluable: bool


def _alarm_hit_counts_by_budget(
    score: ExposureVariantScore,
    *,
    event_weights: Mapping[str, int] | None,
) -> np.ndarray:
    ranks = np.asarray(score.supported_event_alarm_ranks, dtype=np.int64)
    weights = np.fromiter(
        (
            1 if event_weights is None else event_weights.get(event_id, 0)
            for event_id in score.supported_event_ids
        ),
        dtype=np.int64,
        count=len(score.supported_event_ids),
    )
    prefix_counts = np.asarray(score.alarm_prefix_cell_counts, dtype=np.int64)
    result = np.zeros(prefix_counts.size, dtype=np.int64)
    if ranks.size == 0 or not np.any(weights):
        return result
    cumulative = np.cumsum(
        np.bincount(ranks, weights=weights, minlength=int(prefix_counts[-1])),
        dtype=np.float64,
    )
    positive = prefix_counts > 0
    result[positive] = np.rint(cumulative[prefix_counts[positive] - 1]).astype(np.int64)
    return result


def _same_recall_horizon(
    scores: Sequence[ExposureVariantScore],
    *,
    variant: Literal["snapshot", "dynamic"],
    horizon: int,
) -> _SameRecallHorizon:
    budgets = frozen_same_recall_budget_grid()
    primary_index = budgets.index(int(PRIMARY_ALARM_AREA_KM2))
    background_scores = tuple(
        item
        for item in scores
        if item.variant == "background_no_increment"
        and item.horizon_days == horizon
        and item.magnitude_bin == "M5_6"
    )
    candidate_scores = tuple(
        item
        for item in scores
        if item.variant == variant and item.horizon_days == horizon and item.magnitude_bin == "M5_6"
    )
    if not background_scores or tuple(item.issue_date for item in background_scores) != tuple(
        item.issue_date for item in candidate_scores
    ):
        raise ValueError("same-recall variants must share ordered nonoverlapping exposures")
    background_all_ids = tuple(
        event_id for item in background_scores for event_id in item.all_study_area_event_ids
    )
    candidate_all_ids = tuple(
        event_id for item in candidate_scores for event_id in item.all_study_area_event_ids
    )
    if background_all_ids != candidate_all_ids or len(background_all_ids) != len(
        set(background_all_ids)
    ):
        raise ValueError("same-recall variants must share unique all-study-area events")
    background_hits = np.asarray(
        np.sum(
            np.stack(
                [
                    _alarm_hit_counts_by_budget(item, event_weights=None)
                    for item in background_scores
                ]
            ),
            axis=0,
            dtype=np.int64,
        ),
        dtype=np.int64,
    )
    candidate_hits = np.asarray(
        np.sum(
            np.stack(
                [_alarm_hit_counts_by_budget(item, event_weights=None) for item in candidate_scores]
            ),
            axis=0,
            dtype=np.int64,
        ),
        dtype=np.int64,
    )
    candidate_areas = np.asarray(
        np.mean(
            np.stack([item.alarm_exact_selected_area_km2 for item in candidate_scores]),
            axis=0,
            dtype=np.float64,
        ),
        dtype=np.float64,
    )
    background_primary = AlarmAreaPoint(
        budget_km2=int(PRIMARY_ALARM_AREA_KM2),
        mean_exact_selected_area_km2=statistics.fmean(
            float(item.alarm_exact_selected_area_km2[primary_index]) for item in background_scores
        ),
        strict_hit_count=int(background_hits[primary_index]),
    )
    reference_hits = background_primary.strict_hit_count
    target_recall = None if not background_all_ids else reference_hits / len(background_all_ids)
    reached = np.flatnonzero(candidate_hits >= reference_hits)
    if reference_hits == 0 or background_primary.mean_exact_selected_area_km2 <= 0.0:
        return _SameRecallHorizon(
            target_recall=target_recall,
            comparator_area_km2=None,
            candidate_area_km2=None,
            area_reduction_fraction=None,
            evaluable=False,
        )
    comparator_area = float(background_primary.mean_exact_selected_area_km2)
    if reached.size == 0:
        return _SameRecallHorizon(
            target_recall=target_recall,
            comparator_area_km2=comparator_area,
            candidate_area_km2=None,
            area_reduction_fraction=None,
            evaluable=False,
        )
    selected_index = int(reached[0])
    candidate_area = float(candidate_areas[selected_index])
    reduction = 1.0 - candidate_area / comparator_area
    profile = tuple(
        AlarmAreaPoint(
            budget_km2=budget,
            mean_exact_selected_area_km2=float(candidate_areas[index]),
            strict_hit_count=int(candidate_hits[index]),
        )
        for index, budget in enumerate(budgets)
    )
    public_result = same_recall_union_area_relative_reduction(
        background_primary=background_primary,
        candidate_budget_profile=profile,
    )
    if not math.isclose(
        public_result.bootstrap_numeric_value,
        reduction,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise AssertionError("same-recall vector path differs from frozen evaluator")
    return _SameRecallHorizon(
        target_recall=target_recall,
        comparator_area_km2=comparator_area,
        candidate_area_km2=candidate_area,
        area_reduction_fraction=reduction,
        evaluable=True,
    )


def _same_recall_macro(
    scores: Sequence[ExposureVariantScore],
    *,
    variant: Literal["snapshot", "dynamic"],
) -> _SameRecallMacro:
    by_horizon = tuple(
        _same_recall_horizon(scores, variant=variant, horizon=horizon)
        for horizon in PRIMARY_MACRO_HORIZONS_DAYS
    )
    values = tuple(
        0.0 if item.area_reduction_fraction is None else item.area_reduction_fraction
        for item in by_horizon
    )
    numeric = statistics.fmean(values)
    all_evaluable = all(item.evaluable for item in by_horizon)
    return _SameRecallMacro(
        value=numeric if all_evaluable else None,
        bootstrap_numeric_value=numeric,
        evaluable=all_evaluable,
    )


def _first_alarm_budget_by_event(
    scores: Sequence[ExposureVariantScore],
    *,
    variant: VariantId,
    horizon: int,
    event_index: Mapping[str, int],
) -> tuple[np.ndarray, FloatArray]:
    selected = tuple(
        item
        for item in scores
        if item.variant == variant and item.horizon_days == horizon and item.magnitude_bin == "M5_6"
    )
    if not selected:
        raise ValueError("same-recall bootstrap requires every frozen horizon exposure")
    budget_count = len(frozen_same_recall_budget_grid())
    first_budget = np.full(len(event_index), -1, dtype=np.int64)
    for score in selected:
        prefix_counts = np.asarray(score.alarm_prefix_cell_counts, dtype=np.int64)
        for event_id, rank in zip(
            score.supported_event_ids,
            score.supported_event_alarm_ranks,
            strict=True,
        ):
            index = int(np.searchsorted(prefix_counts, rank + 1, side="left"))
            if index < budget_count:
                target_index = event_index[event_id]
                if first_budget[target_index] >= 0:
                    raise ValueError("same-recall exposure duplicated a physical event")
                first_budget[target_index] = index
    mean_area = np.asarray(
        np.mean(
            np.stack([item.alarm_exact_selected_area_km2 for item in selected]),
            axis=0,
            dtype=np.float64,
        ),
        dtype=np.float64,
    )
    mean_area.setflags(write=False)
    return first_budget, mean_area


def _weighted_hit_profiles(
    first_budget_by_event: np.ndarray,
    multiplicity_samples: np.ndarray,
) -> np.ndarray:
    budget_count = len(frozen_same_recall_budget_grid())
    increments = np.zeros((multiplicity_samples.shape[0], budget_count), dtype=np.int64)
    for budget_index in np.unique(first_budget_by_event[first_budget_by_event >= 0]):
        event_columns = np.flatnonzero(first_budget_by_event == budget_index)
        increments[:, int(budget_index)] = np.sum(
            multiplicity_samples[:, event_columns],
            axis=1,
            dtype=np.int64,
        )
    return np.cumsum(increments, axis=1, dtype=np.int64)


@dataclass(frozen=True, slots=True)
class _SameRecallBootstrapSamples:
    by_horizon: Mapping[tuple[VariantId, int], tuple[float, ...]]
    macro: Mapping[VariantId, tuple[float, ...]]


def _bootstrap_same_recall_samples(
    scores: Sequence[ExposureVariantScore],
    *,
    event_index: Mapping[str, int],
    multiplicity_samples: np.ndarray,
) -> _SameRecallBootstrapSamples:
    primary_index = frozen_same_recall_budget_grid().index(int(PRIMARY_ALARM_AREA_KM2))
    horizon_output: dict[tuple[VariantId, int], tuple[float, ...]] = {}
    macro_output: dict[VariantId, tuple[float, ...]] = {}
    for variant in ("dynamic", "snapshot"):
        macro = np.zeros(multiplicity_samples.shape[0], dtype=np.float64)
        for horizon in PRIMARY_MACRO_HORIZONS_DAYS:
            background_first, background_area = _first_alarm_budget_by_event(
                scores,
                variant="background_no_increment",
                horizon=horizon,
                event_index=event_index,
            )
            candidate_first, candidate_area = _first_alarm_budget_by_event(
                scores,
                variant=variant,
                horizon=horizon,
                event_index=event_index,
            )
            background_hits = _weighted_hit_profiles(
                background_first,
                multiplicity_samples,
            )[:, primary_index]
            candidate_hits = _weighted_hit_profiles(candidate_first, multiplicity_samples)
            reached = candidate_hits >= background_hits[:, None]
            valid = (background_hits > 0) & np.any(reached, axis=1)
            first_reached = np.argmax(reached, axis=1)
            horizon_values = np.zeros(multiplicity_samples.shape[0], dtype=np.float64)
            background_primary_area = float(background_area[primary_index])
            if background_primary_area <= 0.0 and np.any(background_hits > 0):
                raise ValueError("positive same-recall reference hits require positive area")
            horizon_values[valid] = 1.0 - (
                candidate_area[first_reached[valid]] / background_primary_area
            )
            horizon_output[(variant, horizon)] = tuple(float(value) for value in horizon_values)
            macro += horizon_values / len(PRIMARY_MACRO_HORIZONS_DAYS)
        macro_output[variant] = tuple(float(value) for value in macro)
    return _SameRecallBootstrapSamples(
        by_horizon=MappingProxyType(horizon_output),
        macro=MappingProxyType(macro_output),
    )


@dataclass(frozen=True, slots=True)
class _BootstrapSummary:
    horizon_information_gain_intervals: Mapping[
        tuple[VariantId, MagnitudeBinId, int], ConfidenceInterval
    ]
    macro_information_gain_intervals: Mapping[VariantId, ConfidenceInterval]
    dynamic_minus_coverage_interval: ConfidenceInterval | None
    snapshot_minus_coverage_interval: ConfidenceInterval | None
    same_area_recall_intervals: Mapping[VariantId, ConfidenceInterval]
    same_recall_area_intervals: Mapping[VariantId, ConfidenceInterval]
    same_recall_area_horizon_intervals: Mapping[tuple[VariantId, int], ConfidenceInterval]


def _event_memberships(
    scores: Sequence[ExposureVariantScore],
) -> tuple[EventHorizonMembership, ...]:
    identities: dict[str, tuple[MagnitudeBinId, set[int]]] = {}
    for magnitude_bin in ("M5_6", "M6_plus"):
        for horizon in STAGE4_HORIZONS_DAYS:
            aggregate = _aggregate(
                scores,
                variant="background_no_increment",
                horizon=horizon,
                magnitude_bin=magnitude_bin,
            )
            for event_id in aggregate.all_event_ids:
                existing = identities.get(event_id)
                if existing is None:
                    identities[event_id] = (magnitude_bin, {horizon})
                elif existing[0] != magnitude_bin:
                    raise ValueError("one physical event appeared in both magnitude bins")
                else:
                    existing[1].add(horizon)
    return tuple(
        EventHorizonMembership(
            event_id=event_id,
            magnitude_bin=magnitude_bin,
            membership=(
                7 in horizons,
                30 in horizons,
                90 in horizons,
                180 in horizons,
                365 in horizons,
            ),
        )
        for event_id, (magnitude_bin, horizons) in sorted(identities.items())
    )


def _bootstrap_summary(
    scores: Sequence[ExposureVariantScore],
    *,
    frozen_input_seal_sha256: str,
) -> _BootstrapSummary:
    events = _event_memberships(scores)
    if not events:
        return _BootstrapSummary(
            horizon_information_gain_intervals=MappingProxyType({}),
            macro_information_gain_intervals=MappingProxyType({}),
            dynamic_minus_coverage_interval=None,
            snapshot_minus_coverage_interval=None,
            same_area_recall_intervals=MappingProxyType({}),
            same_recall_area_intervals=MappingProxyType({}),
            same_recall_area_horizon_intervals=MappingProxyType({}),
        )
    event_order = tuple(item.event_id for item in events)
    event_index = {event_id: index for index, event_id in enumerate(event_order)}
    pairs: tuple[tuple[VariantId, VariantId], ...] = (
        ("dynamic", "background_no_increment"),
        ("coverage_only", "background_no_increment"),
        ("snapshot", "background_no_increment"),
        ("dynamic", "coverage_only"),
        ("snapshot", "coverage_only"),
    )
    magnitude_bins: tuple[MagnitudeBinId, MagnitudeBinId] = ("M5_6", "M6_plus")
    aggregates = {
        (variant, magnitude_bin, horizon): _aggregate(
            scores,
            variant=variant,
            horizon=horizon,
            magnitude_bin=magnitude_bin,
        )
        for variant in VARIANT_ORDER
        for magnitude_bin in magnitude_bins
        for horizon in STAGE4_HORIZONS_DAYS
    }
    contributions: dict[tuple[VariantId, VariantId, MagnitudeBinId, int], FloatArray] = {}
    hit_differences: dict[tuple[VariantId, int], FloatArray] = {}
    for candidate, comparator in pairs:
        for magnitude_bin in magnitude_bins:
            for horizon in STAGE4_HORIZONS_DAYS:
                left = aggregates[(candidate, magnitude_bin, horizon)]
                right = aggregates[(comparator, magnitude_bin, horizon)]
                values = np.zeros(len(events), dtype=np.float64)
                for event_id in left.event_ids:
                    values[event_index[event_id]] = (
                        left.event_log_by_id[event_id] - right.event_log_by_id[event_id]
                    )
                contributions[(candidate, comparator, magnitude_bin, horizon)] = values
    for candidate in ("dynamic", "snapshot"):
        for horizon in STAGE4_HORIZONS_DAYS:
            left = aggregates[(candidate, "M5_6", horizon)]
            background = aggregates[("background_no_increment", "M5_6", horizon)]
            values = np.zeros(len(events), dtype=np.float64)
            for event_id in left.all_event_ids:
                values[event_index[event_id]] = float(
                    int(event_id in left.hit_ids) - int(event_id in background.hit_ids)
                )
            hit_differences[(candidate, horizon)] = values

    bootstrap_variants: tuple[VariantId, VariantId] = ("dynamic", "snapshot")
    horizon_samples: dict[tuple[VariantId, MagnitudeBinId, int], list[float]] = {
        (variant, magnitude_bin, horizon): []
        for variant in bootstrap_variants
        for magnitude_bin in magnitude_bins
        for horizon in STAGE4_HORIZONS_DAYS
    }
    macro_samples: dict[VariantId, list[float]] = {
        "dynamic": [],
        "snapshot": [],
    }
    dynamic_coverage_samples: list[float] = []
    snapshot_coverage_samples: list[float] = []
    recall_samples: dict[VariantId, list[float]] = {"dynamic": [], "snapshot": []}
    multiplicity_samples = np.empty(
        (BOOTSTRAP_REPLICATIONS, len(event_order)),
        dtype=np.int64,
    )
    primary_supported = {
        horizon: len(aggregates[("background_no_increment", "M5_6", horizon)].event_ids)
        for horizon in PRIMARY_MACRO_HORIZONS_DAYS
    }
    primary_all = {
        horizon: len(aggregates[("background_no_increment", "M5_6", horizon)].all_event_ids)
        for horizon in PRIMARY_MACRO_HORIZONS_DAYS
    }
    for replicate_index in range(BOOTSTRAP_REPLICATIONS):
        sample = stratified_five_horizon_bootstrap_indices(
            events,
            context=Stage4SeedContext(
                purpose="bootstrap",
                evaluation_id="formal-validation",
                partition_role="joint",
                replicate_index=replicate_index,
                frozen_input_seal_sha256=frozen_input_seal_sha256,
            ),
        )
        multiplicity_samples[replicate_index, :] = np.asarray(
            sample.multiplicities,
            dtype=np.int64,
        )
        multiplicities = np.asarray(sample.multiplicities, dtype=np.float64)
        for variant in bootstrap_variants:
            horizon_values: dict[int, float] = {}
            for magnitude_bin in magnitude_bins:
                for horizon in STAGE4_HORIZONS_DAYS:
                    candidate_aggregate = aggregates[(variant, magnitude_bin, horizon)]
                    comparator_aggregate = aggregates[
                        ("background_no_increment", magnitude_bin, horizon)
                    ]
                    denominator = len(comparator_aggregate.event_ids)
                    if denominator == 0:
                        continue
                    value = (
                        float(
                            np.dot(
                                multiplicities,
                                contributions[
                                    (
                                        variant,
                                        "background_no_increment",
                                        magnitude_bin,
                                        horizon,
                                    )
                                ],
                            )
                        )
                        - (
                            candidate_aggregate.integrated_intensity
                            - comparator_aggregate.integrated_intensity
                        )
                    ) / denominator
                    horizon_samples[(variant, magnitude_bin, horizon)].append(value)
                    if magnitude_bin == "M5_6":
                        horizon_values[horizon] = value
            if all(horizon in horizon_values for horizon in PRIMARY_MACRO_HORIZONS_DAYS):
                macro_samples[variant].append(
                    statistics.fmean(horizon_values[h] for h in PRIMARY_MACRO_HORIZONS_DAYS)
                )
            if all(primary_all[horizon] > 0 for horizon in PRIMARY_MACRO_HORIZONS_DAYS):
                recall_samples[variant].append(
                    statistics.fmean(
                        100.0
                        * float(np.dot(multiplicities, hit_differences[(variant, horizon)]))
                        / primary_all[horizon]
                        for horizon in PRIMARY_MACRO_HORIZONS_DAYS
                    )
                )
        for candidate, samples in (
            ("dynamic", dynamic_coverage_samples),
            ("snapshot", snapshot_coverage_samples),
        ):
            contrast_values: list[float] = []
            for horizon in PRIMARY_MACRO_HORIZONS_DAYS:
                denominator = primary_supported[horizon]
                if denominator == 0:
                    break
                left = aggregates[(candidate, "M5_6", horizon)]
                right = aggregates[("coverage_only", "M5_6", horizon)]
                contrast_values.append(
                    (
                        float(
                            np.dot(
                                multiplicities,
                                contributions[(candidate, "coverage_only", "M5_6", horizon)],
                            )
                        )
                        - (left.integrated_intensity - right.integrated_intensity)
                    )
                    / denominator
                )
            if len(contrast_values) == len(PRIMARY_MACRO_HORIZONS_DAYS):
                samples.append(statistics.fmean(contrast_values))

    horizon_intervals = {
        key: percentile_interval(values) for key, values in horizon_samples.items() if values
    }
    macro_intervals = {
        variant: percentile_interval(values) for variant, values in macro_samples.items() if values
    }
    recall_intervals = {
        variant: percentile_interval(values) for variant, values in recall_samples.items() if values
    }
    same_recall_samples = _bootstrap_same_recall_samples(
        scores,
        event_index=event_index,
        multiplicity_samples=multiplicity_samples,
    )
    same_recall_intervals = {
        variant: percentile_interval(values)
        for variant, values in same_recall_samples.macro.items()
    }
    same_recall_horizon_intervals = {
        key: percentile_interval(values) for key, values in same_recall_samples.by_horizon.items()
    }
    return _BootstrapSummary(
        horizon_information_gain_intervals=MappingProxyType(horizon_intervals),
        macro_information_gain_intervals=MappingProxyType(macro_intervals),
        dynamic_minus_coverage_interval=(
            percentile_interval(dynamic_coverage_samples) if dynamic_coverage_samples else None
        ),
        snapshot_minus_coverage_interval=(
            percentile_interval(snapshot_coverage_samples) if snapshot_coverage_samples else None
        ),
        same_area_recall_intervals=MappingProxyType(recall_intervals),
        same_recall_area_intervals=MappingProxyType(same_recall_intervals),
        same_recall_area_horizon_intervals=MappingProxyType(same_recall_horizon_intervals),
    )


def _primary_metrics(
    scores: Sequence[ExposureVariantScore],
    *,
    variant: Literal["snapshot", "dynamic"],
) -> tuple[dict[int, int], dict[int, float], int, float | None]:
    counts: dict[int, int] = {}
    gains: dict[int, float] = {}
    union: set[str] = set()
    recall_gains: list[float] = []
    for horizon in PRIMARY_MACRO_HORIZONS_DAYS:
        candidate = _aggregate(
            scores,
            variant=variant,
            horizon=horizon,
            magnitude_bin="M5_6",
        )
        background = _aggregate(
            scores,
            variant="background_no_increment",
            horizon=horizon,
            magnitude_bin="M5_6",
        )
        counts[horizon] = len(background.event_ids)
        union.update(background.event_ids)
        gain = _gain(candidate, background)
        gains[horizon] = 0.0 if gain is None else gain
        if background.all_event_ids:
            recall_gains.append(
                100.0
                * (len(candidate.hit_ids) - len(background.hit_ids))
                / len(background.all_event_ids)
            )
    recall = (
        statistics.fmean(recall_gains)
        if len(recall_gains) == len(PRIMARY_MACRO_HORIZONS_DAYS)
        else None
    )
    return counts, gains, len(union), recall


def _validate_reused_rate_head(
    scope: AssembledFitScope,
    frozen_rate_head: FrozenTargetRateHead,
) -> None:
    """Prove a permutation kept the original two target-rate heads unchanged."""

    for magnitude_bin in ("M5_6", "M6_plus"):
        head = frozen_rate_head.by_id(magnitude_bin)
        if head.training_event_count != scope.training_event_counts[magnitude_bin]:
            raise ValueError("placebo fit changed a frozen rate-head event count")
        if not math.isclose(
            head.background_exposure,
            scope.background_exposures[magnitude_bin],
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("placebo fit changed a frozen rate-head exposure")


def fit_and_primary_macro_statistic(
    scope: AssembledFitScope,
    exposures: Sequence[AssembledEvaluationExposure],
    contracts: Sequence[FeatureColumnContract],
    layout: FeatureLayout,
    variant: Literal["snapshot", "dynamic"],
    frozen_rate_head: FrozenTargetRateHead,
) -> RefitPrimaryMacroResult:
    """Refit one permuted anomaly field while reusing exactly one frozen rate head.

    The caller may assemble a time- or space-permuted scope, but cannot inject a
    statistic.  This function refits fit-scope preprocessing and the selected
    anomaly coefficient vector, scores the frozen primary horizons, and computes
    the equal-weight 7/30/90-day information-gain macro internally.
    """

    contract_tuple = tuple(contracts)
    sources = tuple(item.source_column for item in contract_tuple)
    if not contract_tuple or sources != layout.dynamic_sources:
        raise ValueError("placebo refit contracts must match the frozen dynamic layout")
    exposure_tuple = tuple(exposures)
    if not exposure_tuple or any(
        item.evaluation_id != scope.evaluation_id for item in exposure_tuple
    ):
        raise ValueError("placebo refit exposures must remain inside their fit scope")
    _validate_reused_rate_head(scope, frozen_rate_head)
    preprocessor = fit_frozen_preprocessor(
        contract_tuple,
        scope.preprocessing_fit_columns,
    )
    if frozen_rate_head.primary_evidence_insufficient:
        return RefitPrimaryMacroResult(
            statistic=None,
            frozen_rate_head_sha256=frozen_rate_head.sha256,
            fitted_preprocessor_sha256=preprocessor.sha256,
            anomaly_fit_sha256=None,
        )
    issue_full = preprocessor.transform(scope.issue_cell_feature_columns)
    event_full = preprocessor.transform(scope.event_feature_columns)
    fitted_variant = _fit_increment_variant(
        scope=scope,
        preprocessor=preprocessor,
        issue_full=issue_full,
        event_full=event_full,
        rate_head=frozen_rate_head,
        layout=layout,
        variant=variant,
    )
    background = FittedVariant(
        variant="background_no_increment",
        design_column_indices=(),
        beta=np.asarray([], dtype=np.float64),
        fit_result=None,
    )
    scores: list[ExposureVariantScore] = []
    for exposure in exposure_tuple:
        full_design = preprocessor.transform(exposure.cell_feature_columns)
        scores.extend(
            (
                _score_exposure_variant(
                    exposure,
                    full_design=full_design,
                    rate_head=frozen_rate_head,
                    variant=background,
                ),
                _score_exposure_variant(
                    exposure,
                    full_design=full_design,
                    rate_head=frozen_rate_head,
                    variant=fitted_variant,
                ),
            )
        )
    counts, gains, _, _ = _primary_metrics(scores, variant=variant)
    statistic = (
        statistics.fmean(gains[horizon] for horizon in PRIMARY_MACRO_HORIZONS_DAYS)
        if all(counts[horizon] > 0 for horizon in PRIMARY_MACRO_HORIZONS_DAYS)
        else None
    )
    return RefitPrimaryMacroResult(
        statistic=statistic,
        frozen_rate_head_sha256=frozen_rate_head.sha256,
        fitted_preprocessor_sha256=preprocessor.sha256,
        anomaly_fit_sha256=(
            None if fitted_variant.fit_result is None else fitted_variant.fit_result.sha256
        ),
    )


def _permutation_result(
    injection: PlaceboInjection | None,
    *,
    kind: PlaceboKind,
    variant: Literal["snapshot", "dynamic"],
    observed: float | None,
    fitted: FittedScope,
    contracts: tuple[FeatureColumnContract, ...],
    layout: FeatureLayout,
) -> PermutationTestResult | None:
    if injection is None or observed is None or not math.isfinite(observed):
        return None
    request = PlaceboRequest(
        kind=kind,
        evaluation_id="formal-validation",
        model_variant=variant,
        observed_statistic=observed,
        frozen_rate_head_sha256=fitted.rate_head.sha256,
    )
    execution = injection(request)
    if not isinstance(execution, PlaceboExecution):
        raise TypeError("placebo injection must return PlaceboExecution")
    execution.validate_for(request)
    completed = list(execution.completed_results)

    def validate_recovered(item: CompletedPlaceboReplication) -> None:
        mapping_sha256 = execution.source.mapping_sha256(
            request,
            item.replication_index,
        )
        if mapping_sha256 != item.mapping_sha256:
            raise InfrastructureInterruption(
                "recovered placebo mapping differs from the frozen source"
            )

    def evaluate(replication_index: int) -> CompletedPlaceboReplication:
        item = execution.source.build(request, replication_index)
        try:
            result = fit_and_primary_macro_statistic(
                item.fit_scope,
                item.exposures,
                contracts,
                layout,
                variant,
                fitted.rate_head,
            )
        except PlaceboScientificFailure as exc:
            return CompletedPlaceboReplication(
                replication_index=replication_index,
                mapping_sha256=item.mapping_sha256,
                statistic=None,
                converged=False,
                scientific_failure_code=exc.failure_code,
            )
        except FloatingPointError:
            return CompletedPlaceboReplication(
                replication_index=replication_index,
                mapping_sha256=item.mapping_sha256,
                statistic=None,
                converged=False,
                scientific_failure_code="numerical_instability",
            )
        if result.frozen_rate_head_sha256 != fitted.rate_head.sha256:
            raise InfrastructureInterruption("placebo refit returned a changed rate-head identity")
        if result.statistic is None:
            return CompletedPlaceboReplication(
                replication_index=replication_index,
                mapping_sha256=item.mapping_sha256,
                statistic=None,
                converged=False,
                scientific_failure_code="primary_macro_not_evaluable",
            )
        return CompletedPlaceboReplication(
            replication_index=replication_index,
            mapping_sha256=item.mapping_sha256,
            statistic=result.statistic,
            converged=True,
        )

    try:
        for item in completed:
            validate_recovered(item)
        start = len(completed)
        for batch_start in range(start, PERMUTATION_REPLICATIONS, execution.max_in_flight):
            indices = tuple(
                range(
                    batch_start,
                    min(batch_start + execution.max_in_flight, PERMUTATION_REPLICATIONS),
                )
            )
            batch_results = stable_parallel_map(
                evaluate,
                indices,
                workers=execution.worker_plan.effective_workers,
            )
            if tuple(item.replication_index for item in batch_results) != indices:
                raise InfrastructureInterruption(
                    "placebo worker results were not returned in replication order"
                )
            for item in batch_results:
                completed.append(item)
                if execution.on_result is not None:
                    execution.on_result(request, item)
        replications = tuple(
            PermutationReplication(
                replication_index=item.replication_index,
                statistic=item.statistic,
                converged=item.converged,
            )
            for item in completed
        )
        reduced = reduce_permutation_test(observed, replications)
        if execution.on_complete is not None:
            execution.on_complete(request, tuple(completed))
        return reduced
    except BaseException as exc:
        if execution.on_interruption is not None:
            try:
                execution.on_interruption(request, exc)
            except BaseException as callback_error:
                raise InfrastructureInterruption(
                    "placebo interruption checkpoint callback failed"
                ) from callback_error
        raise


def _g2(
    scores: Sequence[ExposureVariantScore],
    bootstrap: _BootstrapSummary,
    *,
    variant: Literal["snapshot", "dynamic"],
    time_permutation: PermutationTestResult | None,
    space_permutation: PermutationTestResult | None = None,
) -> GateOutcome:
    counts, gains, unique_union, recall = _primary_metrics(scores, variant=variant)
    same_recall = _same_recall_macro(scores, variant=variant)
    return evaluate_g2(
        G2Evidence(
            unique_union_event_count=unique_union,
            event_count_by_horizon=counts,
            information_gain_by_horizon=gains,
            macro_information_gain_interval=bootstrap.macro_information_gain_intervals.get(variant),
            time_permutation_p_value=(
                None if time_permutation is None else time_permutation.monte_carlo_p_value
            ),
            dynamic_minus_coverage_interval=(
                bootstrap.dynamic_minus_coverage_interval
                if variant == "dynamic"
                else bootstrap.snapshot_minus_coverage_interval
            ),
            same_area_recall_gain_percentage_points=recall,
            same_area_recall_gain_interval=bootstrap.same_area_recall_intervals.get(variant),
            same_recall_area_relative_reduction=same_recall.value,
            same_recall_area_interval=bootstrap.same_recall_area_intervals.get(variant),
            same_recall_branch_evaluable=same_recall.evaluable,
            permutation_scientific_failure_fraction=(
                0.0 if time_permutation is None else time_permutation.scientific_failure_fraction
            ),
            space_permutation_scientific_failure_fraction=(
                0.0 if space_permutation is None else space_permutation.scientific_failure_fraction
            ),
        )
    )


def _g3(
    development_scores: Mapping[str, Sequence[ExposureVariantScore]],
    fitted_scopes: Sequence[FittedScope],
) -> GateOutcome:
    if any(item.primary_fit_evidence_insufficient for item in fitted_scopes):
        return GateOutcome(
            "G3",
            "evidence_insufficient",
            (GateCheck("primary_fit_has_events", None, "zero_training_events", ">0"),),
            ("primary_fit_zero_training_events",),
        )
    folds: list[G3FoldEvidence] = []
    for evaluation_id in (
        "development-fold-1",
        "development-fold-2",
        "development-fold-3",
    ):
        scores = development_scores[evaluation_id]
        counts: dict[int, int] = {}
        gains: dict[int, float] = {}
        for horizon in PRIMARY_MACRO_HORIZONS_DAYS:
            dynamic = _aggregate(
                scores,
                variant="dynamic",
                horizon=horizon,
                magnitude_bin="M5_6",
            )
            snapshot = _aggregate(
                scores,
                variant="snapshot",
                horizon=horizon,
                magnitude_bin="M5_6",
            )
            counts[horizon] = len(dynamic.event_ids)
            gain = _gain(dynamic, snapshot)
            gains[horizon] = 0.0 if gain is None else gain
        folds.append(G3FoldEvidence(evaluation_id, counts, gains))
    return evaluate_g3(folds)


def _retrospective_payload(
    scores: Sequence[ExposureVariantScore],
    bootstrap: _BootstrapSummary,
    *,
    dynamic_g2: GateOutcome,
    dynamic_g3: GateOutcome,
    adoption: AdoptionDecision,
    time_permutation: PermutationTestResult | None,
    space_permutation: PermutationTestResult | None,
    model_version: str,
) -> RetrospectivePayload:
    horizon_results: list[RetrospectiveHorizonResult] = []
    for magnitude_bin in ("M5_6", "M6_plus"):
        for horizon in STAGE4_HORIZONS_DAYS:
            dynamic = _aggregate(
                scores,
                variant="dynamic",
                horizon=horizon,
                magnitude_bin=magnitude_bin,
            )
            background = _aggregate(
                scores,
                variant="background_no_increment",
                horizon=horizon,
                magnitude_bin=magnitude_bin,
            )
            gain = _gain(dynamic, background)
            interval = bootstrap.horizon_information_gain_intervals.get(
                ("dynamic", magnitude_bin, horizon)
            )
            if horizon in {180, 365} or magnitude_bin == "M6_plus":
                status: Literal["passed", "failed", "evidence_insufficient"] = (
                    "evidence_insufficient"
                )
            else:
                status = dynamic_g2.status
            recall = (
                None
                if not background.all_event_ids
                else len(dynamic.hit_ids) / len(background.all_event_ids)
            )
            horizon_results.append(
                RetrospectiveHorizonResult(
                    horizon_days=horizon,
                    magnitude_bin=magnitude_bin,
                    result_status=status,
                    independent_event_count=len(dynamic.event_ids),
                    information_gain_nats_per_event=gain,
                    information_gain_lower_95=None if interval is None else interval.lower,
                    information_gain_upper_95=None if interval is None else interval.upper,
                    strict_recall_at_600000_km2=recall,
                )
            )
    target_coverage = tuple(
        RetrospectiveTargetCoverage(
            horizon_days=horizon,
            all_study_area_event_count=len(
                _aggregate(
                    scores,
                    variant="dynamic",
                    horizon=horizon,
                    magnitude_bin="M5_6",
                ).all_event_ids
            ),
            strict_hit_count_at_600000_km2=len(
                _aggregate(
                    scores,
                    variant="dynamic",
                    horizon=horizon,
                    magnitude_bin="M5_6",
                ).hit_ids
            ),
        )
        for horizon in STAGE4_HORIZONS_DAYS
    )
    outcome_status: Literal["passed", "failed", "evidence_insufficient"] = (
        "passed"
        if adoption.status == "adopted"
        else "failed"
        if adoption.status == "credible_negative"
        else "evidence_insufficient"
    )
    return RetrospectivePayload(
        protocol=default_protocol_display(model_version=model_version),
        outcome_status=outcome_status,
        g2_status=dynamic_g2.status,
        g3_status=dynamic_g3.status,
        adoption=adoption.choice,
        issue_dates=tuple(sorted({item.issue_date for item in scores})),
        horizon_results=tuple(horizon_results),
        target_coverage=target_coverage,
        time_permutation_p_value=(
            None if time_permutation is None else time_permutation.monte_carlo_p_value
        ),
        space_permutation_p_value=(
            None if space_permutation is None else space_permutation.monte_carlo_p_value
        ),
    )


def _prospective_payload(
    issues: Sequence[AssembledProspectiveIssue],
    fitted: FittedScope,
    *,
    model_version: str,
) -> ProspectivePayload:
    rows: list[ProspectiveRelativeRank] = []
    for issue in issues:
        full_design = fitted.preprocessor.transform(issue.cell_feature_columns)
        for variant in fitted.variants:
            if variant.variant == "background_no_increment":
                linear = np.zeros(full_design.row_count, dtype=np.float64)
            else:
                design = _select_design(full_design, variant.design_column_indices)
                linear = np.asarray(design.values @ variant.beta, dtype=np.float64)
            rate = fitted.rate_head.by_id(issue.magnitude_bin).rate_multiplier
            intensities = _cell_integrated_intensities(
                background_mass=issue.background_spatial_mass,
                linear_predictor=linear,
                rate_multiplier=rate,
                horizon_days=issue.horizon_days,
            )
            total = math.fsum(float(value) for value in intensities)
            if total == 0.0:
                strength = 0.0
                percentile = 0.0
                band = "inactive_zero_training_events"
            else:
                positive_mean = total / intensities.size
                strength = float(np.max(intensities)) / positive_mean
                percentile = 100.0
                band = "highest_conditional_relative_intensity_band"
            rows.append(
                ProspectiveRelativeRank(
                    issue_date=issue.issue_date,
                    magnitude_bin=issue.magnitude_bin,
                    horizon_days=issue.horizon_days,
                    model_variant=variant.variant,
                    rank_band=band,
                    relative_strength_index=strength,
                    rank_percentile=percentile,
                )
            )
    payload = ProspectivePayload(
        protocol=default_protocol_display(model_version=model_version),
        forecast_status="retrospective_generated_target_blind_shadow",
        issue_dates=tuple(sorted({item.issue_date for item in issues})),
        relative_ranks=tuple(rows),
    )
    assert_prospective_payload_is_target_blind(payload)
    return payload


def _publication_data_flow(
    plan: Stage4InMemoryPlan,
    fitted: FittedScope,
    scores: Sequence[ExposureVariantScore],
) -> DataMethodFlowDiagnostics:
    exposures = plan.formal_scope.exposures
    reference = exposures[0]
    cell_ids = reference.cell_order_ids
    if any(
        item.cell_order_ids != cell_ids
        or not np.array_equal(item.cell_area_km2, reference.cell_area_km2)
        for item in exposures[1:]
    ):
        raise ValueError("publication data-flow counts require one frozen formal grid")
    cell_count = len(cell_ids)
    fit_rows = fitted.preprocessor.fit_row_count
    if fit_rows % cell_count != 0:
        raise ValueError("formal fit rows are not an exact issue-by-cell matrix")
    return DataMethodFlowDiagnostics(
        training_issue_count=fit_rows // cell_count,
        training_cell_count=cell_count,
        fitted_feature_count=len(fitted.variant("dynamic").design_column_indices),
        independent_event_count=len(_event_memberships(scores)),
        study_area_km2=math.fsum(float(value) for value in reference.cell_area_km2),
    )


def _publication_coefficient_effects(
    fitted: FittedScope,
) -> tuple[CoefficientEffectCurve, ...]:
    names = fitted.preprocessor.design_column_names
    curves: list[CoefficientEffectCurve] = []
    for variant_id in ("coverage_only", "snapshot", "dynamic"):
        variant = fitted.variant(variant_id)
        if not variant.design_column_indices:
            raise ValueError("publication coefficient effects require increment coefficients")
        selected = min(
            range(len(variant.design_column_indices)),
            key=lambda index: (-abs(float(variant.beta[index])), index),
        )
        coefficient = float(variant.beta[selected])
        inputs = (-2.0, -1.0, 0.0, 1.0, 2.0)
        curves.append(
            CoefficientEffectCurve(
                variant=variant_id,
                coefficient_name=names[variant.design_column_indices[selected]],
                coefficient_estimate=coefficient,
                input_values=inputs,
                effect_values=tuple(coefficient * value for value in inputs),
            )
        )
    return tuple(curves)


def _publication_distance_lead_decay() -> DistanceLeadDecayDiagnostics:
    distances = (0.0, 50.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0)
    leads = (0.0, 7.0, 30.0, 90.0, 180.0, 365.0)
    return DistanceLeadDecayDiagnostics(
        distance_km=distances,
        spatial_relative_weight=tuple(
            math.exp(-(distance * distance) / (2.0 * 200.0 * 200.0)) for distance in distances
        ),
        lead_days=leads,
        temporal_relative_weight=tuple(float(lead_decay(value)) for value in leads),
    )


def _publication_fold_macros(
    scores_by_id: Mapping[str, Sequence[ExposureVariantScore]],
) -> tuple[FoldMacroValue, ...]:
    output: list[FoldMacroValue] = []
    for fold_index in (1, 2, 3):
        scores = scores_by_id[f"development-fold-{fold_index}"]
        counts: dict[int, int] = {}
        dynamic_gains: list[float] = []
        snapshot_gains: list[float] = []
        for horizon in PRIMARY_MACRO_HORIZONS_DAYS:
            background = _aggregate(
                scores,
                variant="background_no_increment",
                horizon=horizon,
                magnitude_bin="M5_6",
            )
            dynamic = _aggregate(
                scores,
                variant="dynamic",
                horizon=horizon,
                magnitude_bin="M5_6",
            )
            snapshot = _aggregate(
                scores,
                variant="snapshot",
                horizon=horizon,
                magnitude_bin="M5_6",
            )
            counts[horizon] = len(background.event_ids)
            if background.event_ids:
                dynamic_gain = _gain(dynamic, background)
                snapshot_gain = _gain(snapshot, background)
                if dynamic_gain is None or snapshot_gain is None:
                    raise AssertionError("populated fold information gain became unavailable")
                dynamic_gains.append(dynamic_gain)
                snapshot_gains.append(snapshot_gain)
        count_tuple = (counts[7], counts[30], counts[90])
        evaluated = all(value > 0 for value in count_tuple)
        output.append(
            FoldMacroValue(
                fold_index=fold_index,
                primary_horizon_event_counts=count_tuple,
                evidence_status=("evaluated" if evaluated else "evidence_insufficient_zero_events"),
                dynamic_macro_information_gain=(
                    statistics.fmean(dynamic_gains) if evaluated else None
                ),
                snapshot_macro_information_gain=(
                    statistics.fmean(snapshot_gains) if evaluated else None
                ),
            )
        )
    return tuple(output)


def _publication_permutation_distribution(
    *,
    variant: Literal["snapshot", "dynamic"],
    kind: PlaceboKind,
    observed: float | None,
    result: PermutationTestResult | None,
    placebo_injection_supplied: bool,
) -> PermutationDistribution:
    if result is not None:
        return PermutationDistribution(
            variant=variant,
            kind=kind,
            observed_statistic=result.observed_statistic,
            null_statistics=result.null_statistics,
            evidence_status=(
                "evidence_insufficient_scientific_failure_fraction"
                if result.scientific_failure_fraction > 0.01
                else "evaluated"
            ),
        )
    return PermutationDistribution(
        variant=variant,
        kind=kind,
        observed_statistic=None,
        null_statistics=(),
        evidence_status=(
            "evidence_insufficient_zero_events"
            if observed is None
            else "evidence_insufficient_no_placebo_injection"
            if not placebo_injection_supplied
            else "evidence_insufficient_zero_events"
        ),
    )


def _publication_information_gain(
    scores: Sequence[ExposureVariantScore],
    bootstrap: _BootstrapSummary,
) -> tuple[InformationGainInterval, ...]:
    output: list[InformationGainInterval] = []
    for magnitude_bin in ("M5_6", "M6_plus"):
        for horizon in STAGE4_HORIZONS_DAYS:
            dynamic = _aggregate(
                scores,
                variant="dynamic",
                horizon=horizon,
                magnitude_bin=magnitude_bin,
            )
            background = _aggregate(
                scores,
                variant="background_no_increment",
                horizon=horizon,
                magnitude_bin=magnitude_bin,
            )
            count = len(background.event_ids)
            if count == 0:
                output.append(
                    InformationGainInterval(
                        magnitude_bin=magnitude_bin,
                        horizon_days=horizon,
                        independent_event_count=0,
                        evidence_status="evidence_insufficient_zero_events",
                        point_estimate=None,
                        lower_95=None,
                        upper_95=None,
                    )
                )
                continue
            point = _gain(dynamic, background)
            interval = bootstrap.horizon_information_gain_intervals.get(
                ("dynamic", magnitude_bin, horizon)
            )
            if point is None or interval is None:
                raise ValueError("populated information gain lacks its real bootstrap interval")
            evidence_status: InformationGainEvidenceStatus
            if magnitude_bin == "M6_plus" and count < 10:
                evidence_status = (
                    "exploratory_low_sample"
                    if horizon in PRIMARY_MACRO_HORIZONS_DAYS
                    else "exploratory_low_sample_no_random_split"
                )
            else:
                evidence_status = (
                    "evaluated"
                    if horizon in PRIMARY_MACRO_HORIZONS_DAYS
                    else "evidence_insufficient_no_random_split"
                )
            output.append(
                InformationGainInterval(
                    magnitude_bin=magnitude_bin,
                    horizon_days=horizon,
                    independent_event_count=count,
                    evidence_status=evidence_status,
                    point_estimate=point,
                    lower_95=interval.lower,
                    upper_95=interval.upper,
                )
            )
    return tuple(output)


@dataclass(frozen=True, slots=True)
class _RegionalHorizonValues:
    information_gain_by_zone: Mapping[str, float | None]
    supported_count_by_zone: Mapping[str, int]
    all_count_by_zone: Mapping[str, int]
    strict_recall_by_zone: Mapping[str, float | None]


def _regional_horizon_values(
    scores: Sequence[ExposureVariantScore],
    exposures: Sequence[AssembledEvaluationExposure],
    binding: EvaluationRegionBinding,
    *,
    horizon: int,
) -> _RegionalHorizonValues:
    zones = binding.all_construction_zone_ids
    cell_zone = binding.cell_zone_by_id
    event_zone = binding.event_zone_by_id
    dynamic_scores = tuple(
        item
        for item in scores
        if item.variant == "dynamic"
        and item.magnitude_bin == "M5_6"
        and item.horizon_days == horizon
    )
    background_scores = tuple(
        item
        for item in scores
        if item.variant == "background_no_increment"
        and item.magnitude_bin == "M5_6"
        and item.horizon_days == horizon
    )
    exposure_by_issue = {
        item.issue_date: item
        for item in exposures
        if item.magnitude_bin == "M5_6" and item.horizon_days == horizon
    }
    if (
        not dynamic_scores
        or tuple(item.issue_date for item in dynamic_scores)
        != tuple(item.issue_date for item in background_scores)
        or len(exposure_by_issue) != len(dynamic_scores)
    ):
        raise ValueError("regional diagnostics require aligned dynamic/background exposures")
    log_difference = dict.fromkeys(zones, 0.0)
    compensator_difference = dict.fromkeys(zones, 0.0)
    supported_count = dict.fromkeys(zones, 0)
    all_count = dict.fromkeys(zones, 0)
    hit_count = dict.fromkeys(zones, 0)
    seen_supported: set[str] = set()
    seen_all: set[str] = set()
    seen_hits: set[str] = set()
    for dynamic, background in zip(dynamic_scores, background_scores, strict=True):
        exposure = exposure_by_issue[dynamic.issue_date]
        if (
            dynamic.supported_event_ids != background.supported_event_ids
            or dynamic.all_study_area_event_ids != background.all_study_area_event_ids
            or len(dynamic.cell_integrated_intensities) != len(exposure.cell_order_ids)
            or len(background.cell_integrated_intensities) != len(exposure.cell_order_ids)
        ):
            raise ValueError("regional diagnostics found a score/exposure alignment mismatch")
        for cell_id, dynamic_value, background_value in zip(
            exposure.cell_order_ids,
            dynamic.cell_integrated_intensities,
            background.cell_integrated_intensities,
            strict=True,
        ):
            try:
                zone_id = cell_zone[cell_id]
            except KeyError as exc:
                raise ValueError("regional compensator cell lacks a frozen zone") from exc
            compensator_difference[zone_id] += dynamic_value - background_value
        for event_id, dynamic_log, background_log in zip(
            dynamic.supported_event_ids,
            dynamic.event_log_intensities,
            background.event_log_intensities,
            strict=True,
        ):
            if event_id in seen_supported:
                raise ValueError("regional supported event was duplicated across exposures")
            seen_supported.add(event_id)
            try:
                zone_id = event_zone[event_id]
            except KeyError as exc:
                raise ValueError("regional supported event lacks a frozen zone") from exc
            supported_count[zone_id] += 1
            log_difference[zone_id] += dynamic_log - background_log
        for event_id in dynamic.all_study_area_event_ids:
            if event_id in seen_all:
                raise ValueError("regional all-study-area event was duplicated across exposures")
            seen_all.add(event_id)
            try:
                zone_id = event_zone[event_id]
            except KeyError as exc:
                raise ValueError("regional study-area event lacks a frozen zone") from exc
            all_count[zone_id] += 1
        for event_id in dynamic.strict_hit_event_ids_at_600000_km2:
            if event_id in seen_hits or event_id not in seen_all:
                raise ValueError(
                    "regional strict-hit event is duplicated or outside the denominator"
                )
            seen_hits.add(event_id)
            hit_count[event_zone[event_id]] += 1
    information_gain = {
        zone_id: (
            None
            if supported_count[zone_id] == 0
            else (log_difference[zone_id] - compensator_difference[zone_id])
            / supported_count[zone_id]
        )
        for zone_id in zones
    }
    recall = {
        zone_id: (None if all_count[zone_id] == 0 else hit_count[zone_id] / all_count[zone_id])
        for zone_id in zones
    }
    return _RegionalHorizonValues(
        information_gain_by_zone=MappingProxyType(information_gain),
        supported_count_by_zone=MappingProxyType(supported_count),
        all_count_by_zone=MappingProxyType(all_count),
        strict_recall_by_zone=MappingProxyType(recall),
    )


def _publication_regional_metrics(
    scores: Sequence[ExposureVariantScore],
    plan: Stage4InMemoryPlan,
) -> tuple[tuple[str, ...], tuple[RegionHorizonMetric, ...]]:
    binding = plan.evaluation_region_binding
    aliases = tuple(
        f"zone-{index:02d}" for index in range(1, len(binding.all_construction_zone_ids) + 1)
    )
    alias_by_zone = dict(zip(binding.all_construction_zone_ids, aliases, strict=True))
    by_horizon = {
        horizon: _regional_horizon_values(
            scores,
            plan.formal_scope.exposures,
            binding,
            horizon=horizon,
        )
        for horizon in STAGE4_HORIZONS_DAYS
    }
    output: list[RegionHorizonMetric] = []
    for zone_id in binding.all_construction_zone_ids:
        alias = alias_by_zone[zone_id]
        for horizon in STAGE4_HORIZONS_DAYS:
            values = by_horizon[horizon]
            supported = values.supported_count_by_zone[zone_id]
            all_events = values.all_count_by_zone[zone_id]
            output.append(
                RegionHorizonMetric(
                    region_id=alias,
                    horizon_days=horizon,
                    information_gain_nats_per_event=values.information_gain_by_zone[zone_id],
                    supported_event_count=supported,
                    all_study_area_event_count=all_events,
                    strict_recall=values.strict_recall_by_zone[zone_id],
                    information_gain_evidence_status=(
                        "evaluated"
                        if supported > 0
                        else "evidence_insufficient_zero_supported_events"
                    ),
                    strict_recall_evidence_status=(
                        "evaluated" if all_events > 0 else "evidence_insufficient_zero_all_events"
                    ),
                )
            )
    return aliases, tuple(output)


def _publication_alarm_budget_curves(
    scores: Sequence[ExposureVariantScore],
) -> tuple[AlarmBudgetRecallCurve, ...]:
    frozen_budgets = frozen_same_recall_budget_grid()
    indices = tuple(frozen_budgets.index(int(budget)) for budget in PUBLICATION_ALARM_BUDGETS_KM2)
    output: list[AlarmBudgetRecallCurve] = []
    for variant in ("background_no_increment", "dynamic"):
        for horizon in PRIMARY_MACRO_HORIZONS_DAYS:
            selected = tuple(
                item
                for item in scores
                if item.variant == variant
                and item.magnitude_bin == "M5_6"
                and item.horizon_days == horizon
            )
            if not selected:
                raise ValueError("publication alarm curve lacks its formal exposures")
            all_ids = tuple(
                event_id for item in selected for event_id in item.all_study_area_event_ids
            )
            if len(all_ids) != len(set(all_ids)):
                raise ValueError("publication alarm curve duplicated a physical event")
            count = len(all_ids)
            hit_counts = np.asarray(
                np.sum(
                    np.stack(
                        [_alarm_hit_counts_by_budget(item, event_weights=None) for item in selected]
                    ),
                    axis=0,
                    dtype=np.int64,
                ),
                dtype=np.int64,
            )
            mean_areas = np.asarray(
                np.mean(
                    np.stack([item.alarm_exact_selected_area_km2 for item in selected]),
                    axis=0,
                    dtype=np.float64,
                ),
                dtype=np.float64,
            )
            output.append(
                AlarmBudgetRecallCurve(
                    variant=variant,
                    magnitude_bin="M5_6",
                    horizon_days=horizon,
                    points=tuple(
                        AlarmBudgetRecallPoint(
                            budget_km2=budget,
                            selected_alarm_area_km2=float(mean_areas[index]),
                            strict_recall=(
                                None if count == 0 else float(hit_counts[index]) / count
                            ),
                            all_study_area_event_count=count,
                            evidence_status=(
                                "evaluated"
                                if count > 0
                                else "evidence_insufficient_zero_all_events"
                            ),
                        )
                        for budget, index in zip(
                            PUBLICATION_ALARM_BUDGETS_KM2,
                            indices,
                            strict=True,
                        )
                    ),
                )
            )
    return tuple(output)


def _publication_same_recall(
    scores: Sequence[ExposureVariantScore],
    bootstrap: _BootstrapSummary,
) -> tuple[SameRecallAreaReduction, ...]:
    output: list[SameRecallAreaReduction] = []
    for horizon in PRIMARY_MACRO_HORIZONS_DAYS:
        result = _same_recall_horizon(scores, variant="dynamic", horizon=horizon)
        interval = bootstrap.same_recall_area_horizon_intervals.get(("dynamic", horizon))
        if result.evaluable and interval is None:
            raise ValueError("evaluable same-recall result lacks its bootstrap interval")
        output.append(
            SameRecallAreaReduction(
                magnitude_bin="M5_6",
                horizon_days=horizon,
                target_recall=result.target_recall,
                comparator_variant="background_no_increment",
                candidate_variant="dynamic",
                comparator_area_km2=result.comparator_area_km2,
                candidate_area_km2=result.candidate_area_km2,
                area_reduction_lower_95=(
                    interval.lower if result.evaluable and interval is not None else None
                ),
                area_reduction_upper_95=(
                    interval.upper if result.evaluable and interval is not None else None
                ),
                evidence_status=(
                    "evaluated"
                    if result.evaluable
                    else "evidence_insufficient_zero_comparator_recall"
                    if result.target_recall in {None, 0.0}
                    else "evidence_insufficient_target_recall_not_reached"
                ),
            )
        )
    return tuple(output)


def _build_publication_diagnostics(
    plan: Stage4InMemoryPlan,
    fitted: FittedScope,
    scores_by_id: Mapping[str, Sequence[ExposureVariantScore]],
    bootstrap: _BootstrapSummary,
    *,
    dynamic_observed: float | None,
    snapshot_observed: float | None,
    dynamic_time: PermutationTestResult | None,
    dynamic_space: PermutationTestResult | None,
    snapshot_time: PermutationTestResult | None,
    placebo_injection_supplied: bool,
) -> PublicationDiagnostics:
    formal_scores = scores_by_id["formal-validation"]
    region_ids, regional_metrics = _publication_regional_metrics(formal_scores, plan)
    return PublicationDiagnostics(
        data_flow=_publication_data_flow(plan, fitted, formal_scores),
        coefficient_effects=_publication_coefficient_effects(fitted),
        distance_lead_decay=_publication_distance_lead_decay(),
        fold_macro_values=_publication_fold_macros(scores_by_id),
        permutation_distributions=(
            _publication_permutation_distribution(
                variant="dynamic",
                kind="time",
                observed=dynamic_observed,
                result=dynamic_time,
                placebo_injection_supplied=placebo_injection_supplied,
            ),
            _publication_permutation_distribution(
                variant="dynamic",
                kind="space",
                observed=dynamic_observed,
                result=dynamic_space,
                placebo_injection_supplied=placebo_injection_supplied,
            ),
            _publication_permutation_distribution(
                variant="snapshot",
                kind="time",
                observed=snapshot_observed,
                result=snapshot_time,
                placebo_injection_supplied=placebo_injection_supplied,
            ),
        ),
        information_gain_intervals=_publication_information_gain(
            formal_scores,
            bootstrap,
        ),
        region_ids=region_ids,
        region_horizon_metrics=regional_metrics,
        alarm_budget_curves=_publication_alarm_budget_curves(formal_scores),
        same_recall_area_reductions=_publication_same_recall(formal_scores, bootstrap),
    )


def run_stage4_in_memory_pipeline(
    plan: Stage4InMemoryPlan,
    *,
    placebo_injection: PlaceboInjection | None,
) -> PipelineResult:
    """Fit each scope once, score five windows, gate, and render target-safe outputs."""

    scope_inputs = (*plan.development_scopes, plan.formal_scope)
    fitted_scopes = tuple(
        _fit_scope(item.fit, plan.feature_contracts, plan.feature_layout) for item in scope_inputs
    )
    fitted_by_id = {item.evaluation_id: item for item in fitted_scopes}
    scores_by_id: dict[str, tuple[ExposureVariantScore, ...]] = {}
    all_scores: list[ExposureVariantScore] = []
    for scope in scope_inputs:
        fitted = fitted_by_id[scope.fit.evaluation_id]
        scores = tuple(
            score for exposure in scope.exposures for score in _score_exposure(exposure, fitted)
        )
        scores_by_id[scope.fit.evaluation_id] = scores
        all_scores.extend(scores)
    formal_scores = scores_by_id["formal-validation"]
    bootstrap = _bootstrap_summary(
        formal_scores,
        frozen_input_seal_sha256=plan.frozen_input_seal_sha256,
    )
    dynamic_counts, dynamic_gains, _, _ = _primary_metrics(formal_scores, variant="dynamic")
    dynamic_observed = (
        statistics.fmean(dynamic_gains.values())
        if all(dynamic_counts[horizon] > 0 for horizon in PRIMARY_MACRO_HORIZONS_DAYS)
        else None
    )
    snapshot_counts, snapshot_gains, _, _ = _primary_metrics(
        formal_scores,
        variant="snapshot",
    )
    snapshot_observed = (
        statistics.fmean(snapshot_gains.values())
        if all(snapshot_counts[horizon] > 0 for horizon in PRIMARY_MACRO_HORIZONS_DAYS)
        else None
    )
    formal_fit = fitted_by_id["formal-validation"]
    dynamic_time = _permutation_result(
        placebo_injection,
        kind="time",
        variant="dynamic",
        observed=dynamic_observed,
        fitted=formal_fit,
        contracts=plan.feature_contracts,
        layout=plan.feature_layout,
    )
    dynamic_space = _permutation_result(
        placebo_injection,
        kind="space",
        variant="dynamic",
        observed=dynamic_observed,
        fitted=formal_fit,
        contracts=plan.feature_contracts,
        layout=plan.feature_layout,
    )
    snapshot_time = _permutation_result(
        placebo_injection,
        kind="time",
        variant="snapshot",
        observed=snapshot_observed,
        fitted=formal_fit,
        contracts=plan.feature_contracts,
        layout=plan.feature_layout,
    )
    dynamic_g2 = _g2(
        formal_scores,
        bootstrap,
        variant="dynamic",
        time_permutation=dynamic_time,
        space_permutation=dynamic_space,
    )
    snapshot_g2 = _g2(
        formal_scores,
        bootstrap,
        variant="snapshot",
        time_permutation=snapshot_time,
    )
    if formal_fit.primary_fit_evidence_insufficient:
        dynamic_g2 = GateOutcome(
            "G2",
            "evidence_insufficient",
            (GateCheck("primary_fit_has_events", None, "zero_training_events", ">0"),),
            ("primary_fit_zero_training_events",),
        )
        snapshot_g2 = dynamic_g2
    dynamic_g3 = _g3(
        {key: scores_by_id[key] for key in scores_by_id if key.startswith("development-fold-")},
        fitted_scopes[:3],
    )
    adoption = apply_preregistered_adoption_matrix(
        dynamic_g2=dynamic_g2,
        dynamic_g3=dynamic_g3,
        snapshot_equivalent_g2=snapshot_g2,
    )
    retrospective = _retrospective_payload(
        formal_scores,
        bootstrap,
        dynamic_g2=dynamic_g2,
        dynamic_g3=dynamic_g3,
        adoption=adoption,
        time_permutation=dynamic_time,
        space_permutation=dynamic_space,
        model_version=plan.model_version,
    )
    prospective = _prospective_payload(
        plan.prospective_issues,
        formal_fit,
        model_version=plan.model_version,
    )
    publication_diagnostics = _build_publication_diagnostics(
        plan,
        formal_fit,
        scores_by_id,
        bootstrap,
        dynamic_observed=dynamic_observed,
        snapshot_observed=snapshot_observed,
        dynamic_time=dynamic_time,
        dynamic_space=dynamic_space,
        snapshot_time=snapshot_time,
        placebo_injection_supplied=placebo_injection is not None,
    )
    static_svg = build_static_results_svg(retrospective, publication_diagnostics)
    interactive_html = build_interactive_results_html(
        retrospective,
        prospective,
        publication_diagnostics,
    )
    fingerprint_payload = {
        "adoption": dataclasses.asdict(adoption),
        "dynamic_g2": dataclasses.asdict(dynamic_g2),
        "dynamic_g3": dataclasses.asdict(dynamic_g3),
        "fits": [
            {
                "evaluation_id": item.evaluation_id,
                "preprocessor_sha256": item.preprocessor.sha256,
                "rate_head_sha256": item.rate_head.sha256,
                "variant_fit_sha256": {
                    variant.variant: (
                        None if variant.fit_result is None else variant.fit_result.sha256
                    )
                    for variant in item.variants
                },
            }
            for item in fitted_scopes
        ],
        "prospective": dataclasses.asdict(prospective),
        "publication_diagnostics_sha256": publication_diagnostics.content_sha256,
        "retrospective": dataclasses.asdict(retrospective),
        "space_permutation_p": (
            None if dynamic_space is None else dynamic_space.monte_carlo_p_value
        ),
        "time_permutation_p": (None if dynamic_time is None else dynamic_time.monte_carlo_p_value),
    }
    return PipelineResult(
        fitted_scopes=fitted_scopes,
        exposure_scores=tuple(all_scores),
        dynamic_g2=dynamic_g2,
        dynamic_g3=dynamic_g3,
        snapshot_equivalent_g2=snapshot_g2,
        adoption=adoption,
        dynamic_time_permutation=dynamic_time,
        dynamic_space_permutation=dynamic_space,
        retrospective=retrospective,
        prospective=prospective,
        publication_diagnostics=publication_diagnostics,
        static_svg=static_svg,
        interactive_html=interactive_html,
        result_fingerprint_sha256=canonical_mapping_sha256(fingerprint_payload),
    )


__all__ = [
    "BOOTSTRAP_REPLICATIONS",
    "PRIMARY_ALARM_AREA_KM2",
    "VARIANT_ORDER",
    "AssembledEvaluationExposure",
    "AssembledFitScope",
    "AssembledProspectiveIssue",
    "CompletedPlaceboReplication",
    "EvaluationRegionBinding",
    "EvaluationScope",
    "ExposureVariantScore",
    "FeatureLayout",
    "FittedScope",
    "FittedVariant",
    "MagnitudeBinId",
    "PipelineResult",
    "PlaceboCompleteCallback",
    "PlaceboExecution",
    "PlaceboInjection",
    "PlaceboInterruptionCallback",
    "PlaceboKind",
    "PlaceboReplicateFactory",
    "PlaceboReplicateInput",
    "PlaceboRequest",
    "PlaceboResultCallback",
    "PlaceboScientificFailure",
    "PlaceboSource",
    "RefitPrimaryMacroResult",
    "Stage4InMemoryPlan",
    "VariantId",
    "fit_and_primary_macro_statistic",
    "run_stage4_in_memory_pipeline",
]
