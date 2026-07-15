"""Pure-memory assembly of frozen stage-4 fit, assessment, and forecast inputs.

The sole authorized target entrance lives elsewhere.  This module accepts only an
already parsed target catalogue and an assignment receipt produced against the
pre-existing 25 km grid.  It performs no filesystem access and the prospective
assembler is physically target-free.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from itertools import pairwise
from typing import Final, Literal, TypeAlias, cast
from zoneinfo import ZoneInfo

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from seismoflux.anomaly_increment.background_adapter import (
    BackgroundGridDensity,
    FrozenBackgroundSnapshot,
    Stage4BackgroundFit,
)
from seismoflux.anomaly_increment.contracts import (
    BoolArray,
    FloatArray,
    canonical_mapping_sha256,
    readonly_bool_vector,
)
from seismoflux.anomaly_increment.feature_adapter import (
    FEATURE_IDENTITY_COLUMNS,
    assert_issue_table_matches_frozen_grid,
    concatenate_source_columns,
)
from seismoflux.anomaly_increment.grid_features import (
    Stage4IntegrationGrid,
    selected_table_identity_sha256,
)
from seismoflux.anomaly_increment.integration import (
    composite_midpoint_quadrature,
    lead_decay,
)
from seismoflux.anomaly_increment.runner import (
    ExposurePlan,
    FitScopePlan,
    Stage4ScoringPlan,
)
from seismoflux.anomaly_increment.scoring_pipeline import (
    AssembledEvaluationExposure,
    AssembledFitScope,
    AssembledProspectiveIssue,
    EvaluationScope,
    FeatureLayout,
)
from seismoflux.anomaly_increment.targets import (
    Stage4TargetCatalog,
    TargetCellAssignments,
    exposure_target_view,
)
from seismoflux.features.anomaly.snapshot import Stage3IssueSnapshot

EvaluationId: TypeAlias = Literal[
    "development-fold-1",
    "development-fold-2",
    "development-fold-3",
    "formal-validation",
]
MagnitudeBinId: TypeAlias = Literal["M5_6", "M6_plus"]
ProspectiveIssueInput: TypeAlias = AssembledProspectiveIssue

_EVALUATION_IDS: tuple[EvaluationId, ...] = (
    "development-fold-1",
    "development-fold-2",
    "development-fold-3",
    "formal-validation",
)
_MAGNITUDE_BINS: tuple[MagnitudeBinId, ...] = ("M5_6", "M6_plus")
_SHANGHAI = ZoneInfo("Asia/Shanghai")
ZERO_EVENT_POLICY: Final[Literal["exact-zero-rate-no-pseudocount-no-optimizer"]] = (
    "exact-zero-rate-no-pseudocount-no-optimizer"
)


def _identifier(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed identifier")
    return value


def _sha256(value: str, *, label: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _utc(value: datetime, *, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{label} must be a timezone-aware datetime")
    result = value.astimezone(UTC)
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{label} must already be UTC")
    return result


def _issue_time_utc(issue_date_local: date) -> datetime:
    return datetime.combine(issue_date_local, time.min, tzinfo=_SHANGHAI).astimezone(UTC)


def _evaluation_id(scope: FitScopePlan) -> EvaluationId:
    if scope.role == "development_joint_macro" and scope.fit_scope_id in _EVALUATION_IDS[:3]:
        return cast(EvaluationId, scope.fit_scope_id)
    if (
        scope.role == "formal_validation_once"
        and scope.fit_scope_id == "full-development-before-validation"
    ):
        return "formal-validation"
    raise ValueError("fit scope identity differs from the frozen four-scope mapping")


def _catalog_event_order_sha256(catalog: Stage4TargetCatalog) -> str:
    return canonical_mapping_sha256(
        {
            "schema_version": 1,
            "source_content_sha256": catalog.source_content_sha256,
            "source_schema_sha256": catalog.source_schema_sha256,
            "ordered_event_ids": [str(value) for value in catalog.event_id],
            "inside_study_area": [bool(value) for value in catalog.inside_study_area],
        }
    )


def _assignment_sha256(
    assignments: TargetCellAssignments,
    *,
    catalog_event_order_sha256: str,
) -> str:
    return canonical_mapping_sha256(
        {
            "schema_version": 1,
            "catalog_event_order_sha256": catalog_event_order_sha256,
            "grid_id": assignments.grid_id,
            "event_ids": list(assignments.event_ids),
            "cell_indices": [int(value) for value in assignments.cell_indices],
            "cell_ids": list(assignments.cell_ids),
        }
    )


@dataclass(frozen=True, slots=True)
class BoundTargetCellAssignments:
    """Assignment receipt bound to exact target bytes, schema, order, and grid."""

    assignments: TargetCellAssignments
    catalog_source_content_sha256: str
    catalog_source_schema_sha256: str
    catalog_event_order_sha256: str
    assignment_sha256: str

    def __post_init__(self) -> None:
        _sha256(self.catalog_source_content_sha256, label="catalog_source_content_sha256")
        _sha256(self.catalog_source_schema_sha256, label="catalog_source_schema_sha256")
        _sha256(self.catalog_event_order_sha256, label="catalog_event_order_sha256")
        _sha256(self.assignment_sha256, label="assignment_sha256")

    @classmethod
    def bind(
        cls,
        catalog: Stage4TargetCatalog,
        assignments: TargetCellAssignments,
        *,
        primary_grid: Stage4IntegrationGrid,
    ) -> BoundTargetCellAssignments:
        receipt = cls(
            assignments=assignments,
            catalog_source_content_sha256=catalog.source_content_sha256,
            catalog_source_schema_sha256=catalog.source_schema_sha256,
            catalog_event_order_sha256=_catalog_event_order_sha256(catalog),
            assignment_sha256=_assignment_sha256(
                assignments,
                catalog_event_order_sha256=_catalog_event_order_sha256(catalog),
            ),
        )
        receipt.verify(catalog, primary_grid=primary_grid)
        return receipt

    def verify(
        self,
        catalog: Stage4TargetCatalog,
        *,
        primary_grid: Stage4IntegrationGrid,
    ) -> None:
        if primary_grid.cell_size_km != 25.0:
            raise ValueError("target assignments require the frozen 25 km primary grid")
        expected_order = tuple(
            sorted(
                range(len(catalog)),
                key=lambda index: (catalog.origin_time_utc[index], str(catalog.event_id[index])),
            )
        )
        if expected_order != tuple(range(len(catalog))):
            raise ValueError("target catalogue order differs from origin-time/event-ID order")
        if self.catalog_source_content_sha256 != catalog.source_content_sha256:
            raise ValueError("target assignment receipt uses another catalogue file identity")
        if self.catalog_source_schema_sha256 != catalog.source_schema_sha256:
            raise ValueError("target assignment receipt uses another catalogue schema identity")
        event_order_sha256 = _catalog_event_order_sha256(catalog)
        if self.catalog_event_order_sha256 != event_order_sha256:
            raise ValueError("target assignment receipt uses another catalogue event order")
        assignments = self.assignments
        if assignments.grid_id != primary_grid.grid_id:
            raise ValueError("target assignments use another frozen grid")
        inside_ids = tuple(
            str(event_id)
            for event_id, inside in zip(
                catalog.event_id,
                catalog.inside_study_area,
                strict=True,
            )
            if inside
        )
        if assignments.event_ids != inside_ids:
            raise ValueError("target assignments differ from the inside-event order")
        for index, cell_id in zip(
            assignments.cell_indices,
            assignments.cell_ids,
            strict=True,
        ):
            position = int(index)
            if position >= primary_grid.cell_count:
                raise ValueError("target assignment cell index lies outside the frozen grid")
            if primary_grid.cell_ids[position] != cell_id:
                raise ValueError("target assignment row/column/cell identity is inconsistent")
        if self.assignment_sha256 != _assignment_sha256(
            assignments,
            catalog_event_order_sha256=event_order_sha256,
        ):
            raise ValueError("target assignment content differs from its bound receipt")


@dataclass(frozen=True, slots=True)
class FrozenBackgroundField:
    """Validated in-memory projection of one frozen background and its 25 km field."""

    snapshot: FrozenBackgroundSnapshot
    primary_grid_density: BackgroundGridDensity

    def __post_init__(self) -> None:
        if self.primary_grid_density.cell_size_km != 25.0:
            raise ValueError("formal assembly requires the frozen 25 km background field")

    @classmethod
    def from_background_fit(cls, fit: Stage4BackgroundFit) -> FrozenBackgroundField:
        return cls(snapshot=fit.snapshot, primary_grid_density=fit.grid(25.0))

    @property
    def evaluation_id(self) -> EvaluationId:
        return self.snapshot.evaluation_id

    @property
    def grid_id(self) -> str:
        return self.primary_grid_density.grid_id

    @property
    def cell_ids(self) -> tuple[str, ...]:
        return self.primary_grid_density.cell_ids

    @property
    def spatial_density_per_km2(self) -> FloatArray:
        return self.primary_grid_density.spatial_density_per_km2

    @property
    def spatial_cell_mass(self) -> FloatArray:
        return self.primary_grid_density.spatial_cell_mass


@dataclass(frozen=True, slots=True)
class FrozenCellSupport:
    """One fixed per-cell support mask; an exclusion cannot spread to a neighbour."""

    evaluation_id: EvaluationId
    grid_id: str
    cell_ids: tuple[str, ...]
    support_id: str
    compensator_domain_id: str
    supported_cell_mask: BoolArray

    def __post_init__(self) -> None:
        if self.evaluation_id not in _EVALUATION_IDS:
            raise ValueError("cell support is outside the frozen stage-4 scopes")
        _identifier(self.grid_id, label="grid_id")
        cells = tuple(self.cell_ids)
        if not cells or len(cells) != len(set(cells)):
            raise ValueError("cell support requires unique ordered cell IDs")
        _identifier(self.support_id, label="support_id")
        _identifier(self.compensator_domain_id, label="compensator_domain_id")
        mask = readonly_bool_vector("supported_cell_mask", self.supported_cell_mask)
        if mask.shape != (len(cells),):
            raise ValueError("cell support mask must align exactly with the frozen grid")
        if not np.any(mask):
            raise ValueError("cell support cannot exclude the complete study area")
        object.__setattr__(self, "cell_ids", cells)
        object.__setattr__(self, "supported_cell_mask", mask)

    @property
    def supported_indices(self) -> NDArray[np.int64]:
        result = np.flatnonzero(self.supported_cell_mask).astype(np.int64, copy=False)
        result.setflags(write=False)
        return result


def _issue_binding_sha256(
    *,
    issue_time_utc: datetime,
    issue_report_id: str,
    state_snapshot_id: str,
    lineage_digest: str,
    grid_id: str,
    source_columns: tuple[str, ...],
    table_identity_sha256: str,
) -> str:
    return canonical_mapping_sha256(
        {
            "schema_version": 1,
            "issue_time_utc": issue_time_utc.isoformat(),
            "issue_report_id": issue_report_id,
            "state_snapshot_id": state_snapshot_id,
            "lineage_digest": lineage_digest,
            "grid_id": grid_id,
            "source_columns": list(source_columns),
            "table_identity_sha256": table_identity_sha256,
        }
    )


@dataclass(frozen=True, slots=True)
class VerifiedStage3Issue:
    """One feature table bound to its verified stage-3 state snapshot and lineage."""

    issue_time_utc: datetime
    issue_report_id: str
    state_snapshot_id: str
    lineage_digest: str
    grid_id: str
    source_columns: tuple[str, ...]
    table_identity_sha256: str
    binding_sha256: str
    table: pa.Table

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "issue_time_utc",
            _utc(self.issue_time_utc, label="verified issue_time_utc"),
        )
        _identifier(self.issue_report_id, label="issue_report_id")
        _sha256(self.state_snapshot_id, label="state_snapshot_id")
        _sha256(self.lineage_digest, label="lineage_digest")
        _identifier(self.grid_id, label="grid_id")
        columns = tuple(self.source_columns)
        if not columns or len(columns) != len(set(columns)):
            raise ValueError("verified issue source columns must be non-empty and unique")
        if set(columns) & set(FEATURE_IDENTITY_COLUMNS):
            raise ValueError("feature sources may not overlap audit identity columns")
        _sha256(self.table_identity_sha256, label="table_identity_sha256")
        _sha256(self.binding_sha256, label="binding_sha256")
        if not isinstance(self.table, pa.Table):
            raise TypeError("verified issue table must be an in-memory Arrow table")
        object.__setattr__(self, "source_columns", columns)

    @classmethod
    def bind(
        cls,
        table: pa.Table,
        snapshot: Stage3IssueSnapshot,
        *,
        primary_grid: Stage4IntegrationGrid,
        source_columns: Sequence[str],
    ) -> VerifiedStage3Issue:
        issue_time = _utc(snapshot.issue_time_utc, label="stage-3 snapshot issue_time_utc")
        assert_issue_table_matches_frozen_grid(
            table,
            issue_time_utc=issue_time,
            grid=primary_grid,
        )
        report_id = _identifier(snapshot.summary.issue_report_id, label="snapshot issue_report_id")
        if snapshot.summary.issue_time_utc != snapshot.issue_time_utc:
            raise ValueError("stage-3 summary and snapshot issue times differ")
        report_ids = tuple(table["issue_report_id"].combine_chunks().to_pylist())
        if not report_ids or set(report_ids) != {report_id}:
            raise ValueError("feature table report identity differs from the state snapshot")
        names = tuple(source_columns)
        if not names or len(names) != len(set(names)):
            raise ValueError("source_columns must be non-empty and unique")
        if set(names) & set(FEATURE_IDENTITY_COLUMNS):
            raise ValueError("source_columns overlap frozen feature identity columns")
        selected = (*FEATURE_IDENTITY_COLUMNS, *names)
        table_sha256 = selected_table_identity_sha256(table, selected)
        binding_sha256 = _issue_binding_sha256(
            issue_time_utc=issue_time,
            issue_report_id=report_id,
            state_snapshot_id=snapshot.state_snapshot_id,
            lineage_digest=snapshot.lineage_digest,
            grid_id=primary_grid.grid_id,
            source_columns=names,
            table_identity_sha256=table_sha256,
        )
        result = cls(
            issue_time_utc=issue_time,
            issue_report_id=report_id,
            state_snapshot_id=snapshot.state_snapshot_id,
            lineage_digest=snapshot.lineage_digest,
            grid_id=primary_grid.grid_id,
            source_columns=names,
            table_identity_sha256=table_sha256,
            binding_sha256=binding_sha256,
            table=table,
        )
        result.verify(primary_grid=primary_grid, source_columns=names)
        return result

    def verify(
        self,
        *,
        primary_grid: Stage4IntegrationGrid,
        source_columns: Sequence[str],
    ) -> None:
        names = tuple(source_columns)
        if self.grid_id != primary_grid.grid_id or self.source_columns != names:
            raise ValueError("verified issue uses another grid or source-column contract")
        assert_issue_table_matches_frozen_grid(
            self.table,
            issue_time_utc=self.issue_time_utc,
            grid=primary_grid,
        )
        report_ids = tuple(self.table["issue_report_id"].combine_chunks().to_pylist())
        if not report_ids or set(report_ids) != {self.issue_report_id}:
            raise ValueError("verified issue report identity changed")
        table_sha256 = selected_table_identity_sha256(
            self.table,
            (*FEATURE_IDENTITY_COLUMNS, *names),
        )
        if table_sha256 != self.table_identity_sha256:
            raise ValueError("verified feature table content changed")
        expected_binding = _issue_binding_sha256(
            issue_time_utc=self.issue_time_utc,
            issue_report_id=self.issue_report_id,
            state_snapshot_id=self.state_snapshot_id,
            lineage_digest=self.lineage_digest,
            grid_id=self.grid_id,
            source_columns=self.source_columns,
            table_identity_sha256=self.table_identity_sha256,
        )
        if self.binding_sha256 != expected_binding:
            raise ValueError("verified stage-3 issue binding changed")

    def rebind_recipient_table(
        self,
        table: pa.Table,
        *,
        primary_grid: Stage4IntegrationGrid,
        source_columns: Sequence[str],
    ) -> VerifiedStage3Issue:
        """Bind a rebuilt placebo table to this recipient's frozen identity.

        Time and space placebos may replace feature values, but the recipient
        issue time, report identity, state snapshot, and lineage remain fixed.
        The returned binding therefore authenticates the rebuilt in-memory
        table without pretending it is the originally observed table.
        """

        names = tuple(source_columns)
        if self.grid_id != primary_grid.grid_id or self.source_columns != names:
            raise ValueError("placebo recipient uses another grid or source-column contract")
        assert_issue_table_matches_frozen_grid(
            table,
            issue_time_utc=self.issue_time_utc,
            grid=primary_grid,
        )
        report_ids = tuple(table["issue_report_id"].combine_chunks().to_pylist())
        if not report_ids or set(report_ids) != {self.issue_report_id}:
            raise ValueError("rebuilt feature table changed its recipient report identity")
        table_sha256 = selected_table_identity_sha256(
            table,
            (*FEATURE_IDENTITY_COLUMNS, *names),
        )
        rebound = VerifiedStage3Issue(
            issue_time_utc=self.issue_time_utc,
            issue_report_id=self.issue_report_id,
            state_snapshot_id=self.state_snapshot_id,
            lineage_digest=self.lineage_digest,
            grid_id=self.grid_id,
            source_columns=names,
            table_identity_sha256=table_sha256,
            binding_sha256=_issue_binding_sha256(
                issue_time_utc=self.issue_time_utc,
                issue_report_id=self.issue_report_id,
                state_snapshot_id=self.state_snapshot_id,
                lineage_digest=self.lineage_digest,
                grid_id=self.grid_id,
                source_columns=names,
                table_identity_sha256=table_sha256,
            ),
            table=table,
        )
        rebound.verify(primary_grid=primary_grid, source_columns=names)
        return rebound


@dataclass(frozen=True, slots=True)
class ProspectiveIssuePlan:
    """Target-free request for one issue, horizon, and magnitude-bin intensity field."""

    issue_date_local: date
    horizon_days: int
    magnitude_bin: MagnitudeBinId

    def __post_init__(self) -> None:
        if not isinstance(self.issue_date_local, date) or isinstance(
            self.issue_date_local, datetime
        ):
            raise TypeError("prospective issue_date_local must be a date")
        if self.horizon_days not in {7, 30, 90, 180, 365}:
            raise ValueError("prospective horizon differs from the frozen five horizons")
        if self.magnitude_bin not in _MAGNITUDE_BINS:
            raise ValueError("prospective magnitude bin changed")


@dataclass(frozen=True, slots=True)
class ScopeAssemblyAudit:
    evaluation_id: EvaluationId
    fit_supported_event_ids: tuple[str, ...]
    fit_all_study_area_event_ids: tuple[str, ...]
    fit_unavailable_event_ids: tuple[str, ...]
    assessment_all_study_area_event_ids: tuple[str, ...]
    fit_target_end_inclusive_utc: datetime
    assessment_target_start_exclusive_utc: datetime
    zero_event_magnitude_bins: tuple[MagnitudeBinId, ...]
    zero_event_policy: Literal["exact-zero-rate-no-pseudocount-no-optimizer"] = ZERO_EVENT_POLICY

    def __post_init__(self) -> None:
        if self.evaluation_id not in _EVALUATION_IDS:
            raise ValueError("scope audit evaluation_id changed")
        fit_end = _utc(
            self.fit_target_end_inclusive_utc,
            label="fit_target_end_inclusive_utc",
        )
        assessment_start = _utc(
            self.assessment_target_start_exclusive_utc,
            label="assessment_target_start_exclusive_utc",
        )
        if fit_end >= assessment_start:
            raise ValueError("fit targets are not cleared before assessment targets")
        if set(self.fit_all_study_area_event_ids) & set(self.assessment_all_study_area_event_ids):
            raise ValueError("a physical target crossed the fit/assessment boundary")
        if not set(self.fit_supported_event_ids) <= set(self.fit_all_study_area_event_ids):
            raise ValueError("supported fit targets must be study-area fit targets")
        if not set(self.fit_unavailable_event_ids) <= set(self.fit_all_study_area_event_ids):
            raise ValueError("unavailable fit targets must be study-area fit targets")
        if self.zero_event_policy != ZERO_EVENT_POLICY:
            raise ValueError("zero-event policy changed")
        if len(set(self.zero_event_magnitude_bins)) != len(self.zero_event_magnitude_bins) or any(
            item not in _MAGNITUDE_BINS for item in self.zero_event_magnitude_bins
        ):
            raise ValueError("zero-event magnitude-bin audit is invalid")
        object.__setattr__(self, "fit_target_end_inclusive_utc", fit_end)
        object.__setattr__(self, "assessment_target_start_exclusive_utc", assessment_start)


@dataclass(frozen=True, slots=True)
class AssembledScopeBundle:
    scope: EvaluationScope
    audit: ScopeAssemblyAudit

    def __post_init__(self) -> None:
        if self.scope.fit.evaluation_id != self.audit.evaluation_id:
            raise ValueError("scope assembly and audit identities differ")


@dataclass(frozen=True, slots=True)
class Stage4FormalAssembly:
    development_scopes: tuple[EvaluationScope, ...]
    formal_scope: EvaluationScope
    prospective_issues: tuple[ProspectiveIssueInput, ...]
    audits: tuple[ScopeAssemblyAudit, ...]

    def __post_init__(self) -> None:
        development = tuple(self.development_scopes)
        if tuple(item.fit.evaluation_id for item in development) != _EVALUATION_IDS[:3]:
            raise ValueError("formal assembly requires three development folds in frozen order")
        if self.formal_scope.fit.evaluation_id != "formal-validation":
            raise ValueError("formal validation scope identity changed")
        prospective = tuple(self.prospective_issues)
        if not prospective:
            raise ValueError("formal assembly requires target-free prospective inputs")
        audits = tuple(self.audits)
        if tuple(item.evaluation_id for item in audits) != _EVALUATION_IDS:
            raise ValueError("formal assembly audits differ from the four frozen scopes")
        object.__setattr__(self, "development_scopes", development)
        object.__setattr__(self, "prospective_issues", prospective)
        object.__setattr__(self, "audits", audits)

    @property
    def fit_scopes(self) -> tuple[AssembledFitScope, ...]:
        return tuple(scope.fit for scope in (*self.development_scopes, self.formal_scope))

    @property
    def evaluation_exposures(self) -> tuple[AssembledEvaluationExposure, ...]:
        return tuple(
            exposure
            for scope in (*self.development_scopes, self.formal_scope)
            for exposure in scope.exposures
        )


def _verify_background_and_support(
    *,
    evaluation_id: EvaluationId,
    background: FrozenBackgroundField,
    support: FrozenCellSupport,
    primary_grid: Stage4IntegrationGrid,
) -> NDArray[np.int64]:
    if primary_grid.cell_size_km != 25.0:
        raise ValueError("formal model event terms require the frozen 25 km grid")
    if background.evaluation_id != evaluation_id or support.evaluation_id != evaluation_id:
        raise ValueError("background/support crossed an evaluation fit scope")
    if background.grid_id != primary_grid.grid_id or support.grid_id != primary_grid.grid_id:
        raise ValueError("background/support uses another frozen primary grid")
    if background.cell_ids != primary_grid.cell_ids or support.cell_ids != primary_grid.cell_ids:
        raise ValueError("background/support cell order differs from the primary grid")
    if background.snapshot.support_id != support.support_id:
        raise ValueError("background and local support IDs differ")
    if background.snapshot.compensator_domain_id != support.compensator_domain_id:
        raise ValueError("background and support compensator domains differ")
    expected_mass = background.spatial_density_per_km2 * primary_grid.clipped_area_km2
    if not np.allclose(
        expected_mass,
        background.spatial_cell_mass,
        rtol=1.0e-12,
        atol=1.0e-15,
    ):
        raise ValueError("background density and fixed-cell mass are inconsistent")
    return support.supported_indices


def _issue_map(
    verified_issues: Sequence[VerifiedStage3Issue],
    *,
    primary_grid: Stage4IntegrationGrid,
    source_columns: tuple[str, ...],
) -> dict[datetime, VerifiedStage3Issue]:
    issues = tuple(verified_issues)
    if not issues:
        raise ValueError("formal assembly requires verified stage-3 issues")
    result: dict[datetime, VerifiedStage3Issue] = {}
    for issue in issues:
        issue.verify(primary_grid=primary_grid, source_columns=source_columns)
        if issue.issue_time_utc in result:
            raise ValueError("verified stage-3 issue times are duplicated")
        result[issue.issue_time_utc] = issue
    return result


def _required_issue(
    issues: dict[datetime, VerifiedStage3Issue],
    exposure: ExposurePlan,
) -> VerifiedStage3Issue:
    issue_time = _issue_time_utc(exposure.issue_date_local)
    try:
        return issues[issue_time]
    except KeyError as exc:
        raise ValueError(f"verified stage-3 issue is missing: {exposure.identifier}") from exc


def _assert_nonoverlapping(exposures: Sequence[ExposurePlan], *, label: str) -> None:
    supplied = tuple(exposures)
    ordered = tuple(sorted(supplied, key=lambda item: item.issue_date_local))
    if supplied != ordered:
        raise ValueError(f"{label} must remain in chronological order")
    if len(ordered) != len({item.identifier for item in ordered}):
        raise ValueError(f"{label} contains duplicate exposures")
    for previous, current in pairwise(ordered):
        if current.issue_date_local < previous.target_end_inclusive_local:
            raise ValueError(f"{label} target windows overlap")


def _source_values(
    issue: VerifiedStage3Issue,
    *,
    source_columns: tuple[str, ...],
) -> dict[str, FloatArray]:
    raw = concatenate_source_columns((issue.table,), source_columns=source_columns)
    return {name: np.asarray(raw[name], dtype=np.float64) for name in source_columns}


def _target_cell_by_event(
    target_assignments: BoundTargetCellAssignments,
) -> dict[str, int]:
    return {
        event_id: int(cell_index)
        for event_id, cell_index in zip(
            target_assignments.assignments.event_ids,
            target_assignments.assignments.cell_indices,
            strict=True,
        )
    }


def _fit_scope_input(
    scope_plan: FitScopePlan,
    *,
    evaluation_id: EvaluationId,
    catalog: Stage4TargetCatalog,
    target_assignments: BoundTargetCellAssignments,
    issues: dict[datetime, VerifiedStage3Issue],
    primary_grid: Stage4IntegrationGrid,
    background: FrozenBackgroundField,
    support: FrozenCellSupport,
    supported_indices: NDArray[np.int64],
    source_columns: tuple[str, ...],
) -> tuple[AssembledFitScope, ScopeAssemblyAudit]:
    _assert_nonoverlapping(scope_plan.fit_exposures_7d, label="fit exposures")
    fit_issues = tuple(
        _required_issue(issues, exposure) for exposure in scope_plan.fit_exposures_7d
    )
    issue_columns: dict[str, list[FloatArray]] = {name: [] for name in source_columns}
    source_by_time: dict[datetime, dict[str, FloatArray]] = {}
    for issue in fit_issues:
        values = _source_values(issue, source_columns=source_columns)
        source_by_time[issue.issue_time_utc] = values
        for name in source_columns:
            issue_columns[name].append(values[name][supported_indices])
    assembled_issue_columns = {
        name: np.concatenate(chunks).astype(np.float64, copy=False)
        for name, chunks in issue_columns.items()
    }
    supported_mass = background.spatial_cell_mass[supported_indices]
    row_mass = np.tile(supported_mass, len(fit_issues))
    spatial_mass_by_bin = np.column_stack((row_mass, row_mass))
    event_columns: dict[str, list[float]] = {name: [] for name in source_columns}
    event_background: list[float] = []
    event_decays: list[float] = []
    event_bins: list[MagnitudeBinId] = []
    supported_event_ids: list[str] = []
    all_fit_ids: list[str] = []
    unavailable_ids: list[str] = []
    counts: dict[MagnitudeBinId, int] = {"M5_6": 0, "M6_plus": 0}
    event_to_cell = _target_cell_by_event(target_assignments)
    supported_set = {int(value) for value in supported_indices}
    assessment_start = min(
        _issue_time_utc(exposure.issue_date_local)
        for assessment in scope_plan.assessments
        for exposure in assessment.exposures
    )
    for exposure, issue in zip(scope_plan.fit_exposures_7d, fit_issues, strict=True):
        values = source_by_time[issue.issue_time_utc]
        for magnitude_bin in _MAGNITUDE_BINS:
            view = exposure_target_view(catalog, exposure, magnitude_bin_id=magnitude_bin)
            all_fit_ids.extend(view.event_ids)
            for catalog_index, event_id, exact_lead in zip(
                view.event_indices,
                view.event_ids,
                view.lead_days,
                strict=True,
            ):
                index = int(catalog_index)
                if catalog.available_at_utc[index] > assessment_start:
                    unavailable_ids.append(event_id)
                    continue
                cell_index = event_to_cell[event_id]
                if cell_index not in supported_set:
                    continue
                for name in source_columns:
                    event_columns[name].append(float(values[name][cell_index]))
                event_background.append(float(background.spatial_density_per_km2[cell_index]))
                event_decays.append(float(lead_decay(float(exact_lead))))
                event_bins.append(magnitude_bin)
                supported_event_ids.append(event_id)
                counts[magnitude_bin] += 1
    if len(all_fit_ids) != len(set(all_fit_ids)):
        raise ValueError("a physical target appears in more than one fit exposure")
    if len(supported_event_ids) != len(set(supported_event_ids)):
        raise ValueError("a supported physical target was duplicated during fitting")
    assessment_ids = {
        event_id
        for assessment in scope_plan.assessments
        for exposure in assessment.exposures
        for magnitude_bin in _MAGNITUDE_BINS
        for event_id in exposure_target_view(
            catalog,
            exposure,
            magnitude_bin_id=magnitude_bin,
        ).event_ids
    }
    if set(all_fit_ids) & assessment_ids:
        raise ValueError("fit and assessment target event identities overlap")
    quadrature = composite_midpoint_quadrature(7.0)
    midpoint_decays = cast(FloatArray, lead_decay(quadrature.lead_midpoints_days))
    exposure_by_bin = np.asarray(
        np.sum(spatial_mass_by_bin, axis=0, dtype=np.float64) * 7.0,
        dtype=np.float64,
    )
    fit = AssembledFitScope(
        evaluation_id=evaluation_id,
        preprocessing_fit_columns=assembled_issue_columns,
        issue_cell_feature_columns=assembled_issue_columns,
        event_feature_columns={
            name: np.asarray(values, dtype=np.float64) for name, values in event_columns.items()
        },
        background_spatial_mass_by_row_and_bin=spatial_mass_by_bin,
        midpoint_widths_days=quadrature.widths_days,
        midpoint_decays=midpoint_decays,
        event_background_intensity=np.asarray(event_background, dtype=np.float64),
        event_decay=np.asarray(event_decays, dtype=np.float64),
        event_magnitude_bin_ids=tuple(event_bins),
        training_event_counts={name: counts[name] for name in _MAGNITUDE_BINS},
        background_exposures={
            name: float(exposure_by_bin[index]) for index, name in enumerate(_MAGNITUDE_BINS)
        },
    )
    fit_end = max(
        _issue_time_utc(exposure.issue_date_local) + timedelta(days=7)
        for exposure in scope_plan.fit_exposures_7d
    )
    audit = ScopeAssemblyAudit(
        evaluation_id=evaluation_id,
        fit_supported_event_ids=tuple(supported_event_ids),
        fit_all_study_area_event_ids=tuple(all_fit_ids),
        fit_unavailable_event_ids=tuple(dict.fromkeys(unavailable_ids)),
        assessment_all_study_area_event_ids=tuple(sorted(assessment_ids)),
        fit_target_end_inclusive_utc=fit_end,
        assessment_target_start_exclusive_utc=assessment_start,
        zero_event_magnitude_bins=tuple(name for name in _MAGNITUDE_BINS if counts[name] == 0),
    )
    return fit, audit


def _evaluation_exposure(
    exposure: ExposurePlan,
    *,
    magnitude_bin: MagnitudeBinId,
    evaluation_id: EvaluationId,
    catalog: Stage4TargetCatalog,
    target_assignments: BoundTargetCellAssignments,
    issue: VerifiedStage3Issue,
    primary_grid: Stage4IntegrationGrid,
    background: FrozenBackgroundField,
    support: FrozenCellSupport,
    supported_indices: NDArray[np.int64],
    source_columns: tuple[str, ...],
) -> AssembledEvaluationExposure:
    values = _source_values(issue, source_columns=source_columns)
    view = exposure_target_view(catalog, exposure, magnitude_bin_id=magnitude_bin)
    event_to_cell = _target_cell_by_event(target_assignments)
    local_by_full = {
        int(full_index): local_index for local_index, full_index in enumerate(supported_indices)
    }
    supported_ids: list[str] = []
    supported_cells: list[int] = []
    supported_leads: list[float] = []
    for event_id, exact_lead in zip(view.event_ids, view.lead_days, strict=True):
        full_index = event_to_cell[event_id]
        local_index = local_by_full.get(full_index)
        if local_index is None:
            continue
        supported_ids.append(event_id)
        supported_cells.append(local_index)
        supported_leads.append(float(exact_lead))
    return AssembledEvaluationExposure(
        evaluation_id=evaluation_id,
        issue_date=exposure.issue_date_local.isoformat(),
        horizon_days=exposure.horizon_days,
        magnitude_bin=magnitude_bin,
        support_id=support.support_id,
        compensator_domain_id=support.compensator_domain_id,
        cell_order_ids=tuple(primary_grid.cell_ids[int(index)] for index in supported_indices),
        cell_rows=tuple(int(primary_grid.rows[int(index)]) for index in supported_indices),
        cell_columns=tuple(int(primary_grid.columns[int(index)]) for index in supported_indices),
        cell_area_km2=primary_grid.clipped_area_km2[supported_indices],
        background_spatial_mass=background.spatial_cell_mass[supported_indices],
        cell_feature_columns={name: values[name][supported_indices] for name in source_columns},
        supported_event_ids=tuple(supported_ids),
        all_study_area_event_ids=view.event_ids,
        event_cell_indices=tuple(supported_cells),
        event_lead_days=np.asarray(supported_leads, dtype=np.float64),
    )


def assemble_evaluation_scope(
    scope_plan: FitScopePlan,
    *,
    catalog: Stage4TargetCatalog,
    target_assignments: BoundTargetCellAssignments,
    verified_issues: Sequence[VerifiedStage3Issue],
    primary_grid: Stage4IntegrationGrid,
    background: FrozenBackgroundField,
    support: FrozenCellSupport,
    feature_layout: FeatureLayout,
) -> AssembledScopeBundle:
    """Assemble one development/formal fit plus all frozen assessment exposures."""

    evaluation_id = _evaluation_id(scope_plan)
    expected_assessment_partition = (
        "validation" if evaluation_id == "formal-validation" else "development"
    )
    if any(
        exposure.partition != expected_assessment_partition
        for assessment in scope_plan.assessments
        for exposure in assessment.exposures
    ):
        raise ValueError("assessment exposure partition differs from its frozen fit scope")
    source_columns = tuple(feature_layout.dynamic_sources)
    target_assignments.verify(catalog, primary_grid=primary_grid)
    supported_indices = _verify_background_and_support(
        evaluation_id=evaluation_id,
        background=background,
        support=support,
        primary_grid=primary_grid,
    )
    issues = _issue_map(
        verified_issues,
        primary_grid=primary_grid,
        source_columns=source_columns,
    )
    fit, audit = _fit_scope_input(
        scope_plan,
        evaluation_id=evaluation_id,
        catalog=catalog,
        target_assignments=target_assignments,
        issues=issues,
        primary_grid=primary_grid,
        background=background,
        support=support,
        supported_indices=supported_indices,
        source_columns=source_columns,
    )
    assembled_exposures: list[AssembledEvaluationExposure] = []
    for assessment in scope_plan.assessments:
        _assert_nonoverlapping(
            assessment.exposures,
            label=f"assessment horizon {assessment.horizon_days}",
        )
        for exposure in assessment.exposures:
            issue = _required_issue(issues, exposure)
            for magnitude_bin in _MAGNITUDE_BINS:
                assembled_exposures.append(
                    _evaluation_exposure(
                        exposure,
                        magnitude_bin=magnitude_bin,
                        evaluation_id=evaluation_id,
                        catalog=catalog,
                        target_assignments=target_assignments,
                        issue=issue,
                        primary_grid=primary_grid,
                        background=background,
                        support=support,
                        supported_indices=supported_indices,
                        source_columns=source_columns,
                    )
                )
    return AssembledScopeBundle(
        scope=EvaluationScope(fit=fit, exposures=tuple(assembled_exposures)),
        audit=audit,
    )


def assemble_prospective_issue_inputs(
    prospective_plans: Sequence[ProspectiveIssuePlan],
    *,
    verified_issues: Sequence[VerifiedStage3Issue],
    primary_grid: Stage4IntegrationGrid,
    background: FrozenBackgroundField,
    support: FrozenCellSupport,
    feature_layout: FeatureLayout,
) -> tuple[ProspectiveIssueInput, ...]:
    """Assemble true forecast inputs without accepting any target object or event field."""

    plans = tuple(prospective_plans)
    if not plans:
        raise ValueError("at least one prospective issue input is required")
    keys = tuple((item.issue_date_local, item.horizon_days, item.magnitude_bin) for item in plans)
    if len(keys) != len(set(keys)):
        raise ValueError("prospective issue requests are duplicated")
    source_columns = tuple(feature_layout.dynamic_sources)
    supported_indices = _verify_background_and_support(
        evaluation_id="formal-validation",
        background=background,
        support=support,
        primary_grid=primary_grid,
    )
    issues = _issue_map(
        verified_issues,
        primary_grid=primary_grid,
        source_columns=source_columns,
    )
    output: list[ProspectiveIssueInput] = []
    for plan in plans:
        issue_time = _issue_time_utc(plan.issue_date_local)
        try:
            issue = issues[issue_time]
        except KeyError as exc:
            raise ValueError(
                f"verified stage-3 prospective issue is missing: {plan.issue_date_local}"
            ) from exc
        values = _source_values(issue, source_columns=source_columns)
        output.append(
            AssembledProspectiveIssue(
                issue_date=plan.issue_date_local.isoformat(),
                horizon_days=plan.horizon_days,
                magnitude_bin=plan.magnitude_bin,
                cell_order_ids=tuple(
                    primary_grid.cell_ids[int(index)] for index in supported_indices
                ),
                cell_area_km2=primary_grid.clipped_area_km2[supported_indices],
                background_spatial_mass=background.spatial_cell_mass[supported_indices],
                cell_feature_columns={
                    name: values[name][supported_indices] for name in source_columns
                },
            )
        )
    return tuple(output)


def assemble_stage4_formal_inputs(
    scoring_plan: Stage4ScoringPlan,
    *,
    catalog: Stage4TargetCatalog,
    target_assignments: BoundTargetCellAssignments,
    verified_issues: Sequence[VerifiedStage3Issue],
    primary_grid: Stage4IntegrationGrid,
    backgrounds: Sequence[FrozenBackgroundField],
    cell_supports: Sequence[FrozenCellSupport],
    feature_layout: FeatureLayout,
    prospective_plans: Sequence[ProspectiveIssuePlan],
) -> Stage4FormalAssembly:
    """Assemble the three development folds, formal validation, and forecasts."""

    target_assignments.verify(catalog, primary_grid=primary_grid)
    scope_plans = tuple(scoring_plan.fit_scopes)
    if tuple(_evaluation_id(item) for item in scope_plans) != _EVALUATION_IDS:
        raise ValueError("scoring plan differs from the four frozen fit scopes")
    background_values = tuple(backgrounds)
    support_values = tuple(cell_supports)
    if len(background_values) != 4 or len(support_values) != 4:
        raise ValueError("formal assembly requires exactly four background/support inputs")
    background_by_id = {item.evaluation_id: item for item in background_values}
    support_by_id = {item.evaluation_id: item for item in support_values}
    if tuple(background_by_id) != _EVALUATION_IDS or tuple(support_by_id) != _EVALUATION_IDS:
        raise ValueError("background/support inputs must follow the four frozen scope identities")
    bundles: list[AssembledScopeBundle] = []
    for scope_plan in scope_plans:
        evaluation_id = _evaluation_id(scope_plan)
        bundles.append(
            assemble_evaluation_scope(
                scope_plan,
                catalog=catalog,
                target_assignments=target_assignments,
                verified_issues=verified_issues,
                primary_grid=primary_grid,
                background=background_by_id[evaluation_id],
                support=support_by_id[evaluation_id],
                feature_layout=feature_layout,
            )
        )
    prospective = assemble_prospective_issue_inputs(
        prospective_plans,
        verified_issues=verified_issues,
        primary_grid=primary_grid,
        background=background_by_id["formal-validation"],
        support=support_by_id["formal-validation"],
        feature_layout=feature_layout,
    )
    return Stage4FormalAssembly(
        development_scopes=tuple(item.scope for item in bundles[:3]),
        formal_scope=bundles[3].scope,
        prospective_issues=prospective,
        audits=tuple(item.audit for item in bundles),
    )


__all__ = [
    "ZERO_EVENT_POLICY",
    "AssembledScopeBundle",
    "BoundTargetCellAssignments",
    "FrozenBackgroundField",
    "FrozenCellSupport",
    "ProspectiveIssueInput",
    "ProspectiveIssuePlan",
    "ScopeAssemblyAudit",
    "Stage4FormalAssembly",
    "VerifiedStage3Issue",
    "assemble_evaluation_scope",
    "assemble_prospective_issue_inputs",
    "assemble_stage4_formal_inputs",
]
