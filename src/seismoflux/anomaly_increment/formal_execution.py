"""Authorized in-memory bridge from score-blind context to formal stage-4 inputs.

The context type deliberately cannot hold an earthquake catalogue, an event-cell
assignment, a filesystem location, or an I/O capability.  Only
``materialize_after_authorized_target`` accepts the already authorized in-memory
catalogue only together with the full execution protocol and its sentinel-protected
authorization capability.  Construction-zone strata are joined after authorization
and remain a separate, local-only evaluation receipt; they never enter fitting,
ranking, or prospective inputs.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, time
from typing import Literal, TypeAlias, cast
from zoneinfo import ZoneInfo

import pyarrow as pa

from seismoflux.anomaly_increment.authorization import (
    Stage4TargetAuthorization,
    require_stage4_target_authorization,
)
from seismoflux.anomaly_increment.background_adapter import (
    FROZEN_BACKGROUND_VARIANT_ID,
    FROZEN_KDE_BANDWIDTH_KM,
    BackgroundDomainBinding,
    FrozenBackgroundSnapshot,
    Stage4BackgroundFit,
    rebuild_stage4_background,
    resolve_frozen_background_snapshot,
)
from seismoflux.anomaly_increment.config import (
    Stage4ProtocolBundle,
    require_stage4_r2_execution_action,
)
from seismoflux.anomaly_increment.contracts import (
    FeatureColumnContract,
    canonical_mapping_sha256,
)
from seismoflux.anomaly_increment.formal_assembly import (
    BoundTargetCellAssignments,
    FrozenBackgroundField,
    FrozenCellSupport,
    ProspectiveIssuePlan,
    Stage4FormalAssembly,
    VerifiedStage3Issue,
    assemble_evaluation_scope,
    assemble_stage4_formal_inputs,
)
from seismoflux.anomaly_increment.grid_features import (
    Stage4GridFamily,
    Stage4IntegrationGrid,
)
from seismoflux.anomaly_increment.preregistration import protocol_design_sha256
from seismoflux.anomaly_increment.runner import FitScopePlan, Stage4ScoringPlan
from seismoflux.anomaly_increment.scoring_pipeline import (
    EvaluationRegionBinding,
    EvaluationScope,
    FeatureLayout,
    Stage4InMemoryPlan,
)
from seismoflux.anomaly_increment.targets import (
    Stage4TargetCatalog,
    map_targets_to_frozen_primary_grid,
)
from seismoflux.background.catalog import StudyArea
from seismoflux.background.grid import EQUAL_AREA_CRS

EvaluationId: TypeAlias = Literal[
    "development-fold-1",
    "development-fold-2",
    "development-fold-3",
    "formal-validation",
]

_EVALUATION_IDS: tuple[EvaluationId, ...] = (
    "development-fold-1",
    "development-fold-2",
    "development-fold-3",
    "formal-validation",
)
_SHANGHAI = ZoneInfo("Asia/Shanghai")


def _identifier(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed identifier")
    return value


def _sha256(value: str, *, label: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _scope_evaluation_id(scope: FitScopePlan) -> EvaluationId:
    if scope.role == "development_joint_macro" and scope.fit_scope_id in _EVALUATION_IDS[:3]:
        return cast(EvaluationId, scope.fit_scope_id)
    if (
        scope.role == "formal_validation_once"
        and scope.fit_scope_id == "full-development-before-validation"
    ):
        return "formal-validation"
    raise ValueError("formal execution fit scopes differ from the frozen four-scope mapping")


def _snapshot_declaration(snapshot: FrozenBackgroundSnapshot) -> dict[str, object]:
    return {
        "snapshot_id": snapshot.snapshot_id,
        "parameter_snapshot_id": snapshot.parameter_snapshot_id,
        "fit_end_utc": snapshot.fit_end_utc.isoformat().replace("+00:00", "Z"),
        "support_id": snapshot.support_id,
        "compensator_domain_id": snapshot.compensator_domain_id,
        "common_mc": snapshot.common_mc,
        "supported_area_fraction": snapshot.supported_area_fraction,
    }


@dataclass(frozen=True, slots=True)
class VerifiedFormalProtocol:
    """Path-free projection of the already validated protocol needed by the bridge."""

    protocol_design_sha256: str
    random_input_seal_sha256: str
    development_snapshot: FrozenBackgroundSnapshot
    formal_validation_snapshot: FrozenBackgroundSnapshot

    def __post_init__(self) -> None:
        _sha256(self.protocol_design_sha256, label="protocol_design_sha256")
        _sha256(self.random_input_seal_sha256, label="random_input_seal_sha256")
        development = self.development_snapshot
        validation = self.formal_validation_snapshot
        if development.evaluation_id != "development-fold-1" or development.role != "development":
            raise ValueError("protocol projection requires the frozen development snapshot")
        if validation.evaluation_id != "formal-validation" or validation.role != "validation":
            raise ValueError("protocol projection requires the frozen validation snapshot")
        if development.fit_end_utc >= validation.fit_end_utc:
            raise ValueError("development background cutoff must precede validation cutoff")
        if development.parameter_snapshot_id == validation.parameter_snapshot_id:
            raise ValueError(
                "development background cannot reuse the validation parameter snapshot"
            )
        if development.study_area_sha256 != validation.study_area_sha256:
            raise ValueError("development and validation backgrounds use different study areas")
        if development.compensator_domain_id != validation.compensator_domain_id:
            raise ValueError("development and validation backgrounds use different domains")
        protocol = self.background_protocol()
        for evaluation_id in _EVALUATION_IDS:
            resolve_frozen_background_snapshot(protocol, evaluation_id=evaluation_id)

    @classmethod
    def from_verified_protocol(cls, protocol: Stage4ProtocolBundle) -> VerifiedFormalProtocol:
        """Project an upstream-validated bundle without retaining its filesystem fields."""

        return cls(
            protocol_design_sha256=protocol.design_sha256,
            random_input_seal_sha256=protocol.random_input_seal_sha256,
            development_snapshot=resolve_frozen_background_snapshot(
                protocol.protocol,
                evaluation_id="development-fold-1",
            ),
            formal_validation_snapshot=resolve_frozen_background_snapshot(
                protocol.protocol,
                evaluation_id="formal-validation",
            ),
        )

    def background_protocol(self) -> Mapping[str, object]:
        """Reconstruct only the verified background declaration needed in memory."""

        return {
            "background": {
                "background_variant_id": FROZEN_BACKGROUND_VARIANT_ID,
                "family": "spatial_poisson",
                "bandwidth_km": FROZEN_KDE_BANDWIDTH_KM,
                "model_reselection_forbidden": True,
                "development": _snapshot_declaration(self.development_snapshot),
                "validation": _snapshot_declaration(self.formal_validation_snapshot),
            },
            "inputs": {"study_area": {"sha256": self.development_snapshot.study_area_sha256}},
        }

    def snapshot_for(self, evaluation_id: EvaluationId) -> FrozenBackgroundSnapshot:
        return resolve_frozen_background_snapshot(
            self.background_protocol(),
            evaluation_id=evaluation_id,
        )


def _cell_zone_mapping_sha256(
    *,
    grid_id: str,
    cell_ids: tuple[str, ...],
    construction_zone_ids: tuple[str, ...],
    all_construction_zone_ids: tuple[str, ...],
    manifest_sha256: str,
) -> str:
    return canonical_mapping_sha256(
        {
            "schema_version": 1,
            "role": "score_blind_fixed_cell_construction_zone_mapping",
            "grid_id": grid_id,
            "manifest_sha256": manifest_sha256,
            "all_construction_zone_ids": list(all_construction_zone_ids),
            "diagnostic_role": ("evaluation-only-compensator-no-fit-order-candidate-prospective"),
            "public_export_forbidden": True,
            "cells": [
                {"cell_id": cell_id, "construction_zone_id": zone_id}
                for cell_id, zone_id in zip(cell_ids, construction_zone_ids, strict=True)
            ],
        }
    )


@dataclass(frozen=True, slots=True)
class FrozenCellZoneMapping:
    """Local score-blind cell-to-zone mapping; never a public publication payload."""

    grid_id: str
    cell_ids: tuple[str, ...]
    construction_zone_ids: tuple[str, ...]
    all_construction_zone_ids: tuple[str, ...]
    manifest_sha256: str
    mapping_sha256: str
    diagnostic_role: Literal["evaluation-only-compensator-no-fit-order-candidate-prospective"] = (
        "evaluation-only-compensator-no-fit-order-candidate-prospective"
    )
    public_export_forbidden: Literal[True] = True

    def __post_init__(self) -> None:
        _identifier(self.grid_id, label="cell-zone grid_id")
        cell_ids = tuple(self.cell_ids)
        zones = tuple(self.construction_zone_ids)
        all_zones = tuple(self.all_construction_zone_ids)
        if not cell_ids or len(cell_ids) != len(set(cell_ids)):
            raise ValueError("cell-zone mapping requires unique ordered cell IDs")
        if len(zones) != len(cell_ids):
            raise ValueError("one construction zone is required per frozen cell")
        for zone_id in zones:
            _identifier(zone_id, label="construction_zone_id")
        if len(all_zones) != 39 or len(all_zones) != len(set(all_zones)):
            raise ValueError("formal cell-zone receipt requires exactly 39 unique zones")
        for zone_id in all_zones:
            _identifier(zone_id, label="all_construction_zone_ids")
        if set(zones) != set(all_zones):
            raise ValueError("every one of the 39 construction zones requires a frozen cell")
        if self.diagnostic_role != (
            "evaluation-only-compensator-no-fit-order-candidate-prospective"
        ):
            raise ValueError("cell-zone receipt escaped its evaluation-only role")
        if self.public_export_forbidden is not True:
            raise ValueError("cell-zone receipt must remain local")
        _sha256(self.manifest_sha256, label="cell-zone manifest_sha256")
        _sha256(self.mapping_sha256, label="cell-zone mapping_sha256")
        expected = _cell_zone_mapping_sha256(
            grid_id=self.grid_id,
            cell_ids=cell_ids,
            construction_zone_ids=zones,
            all_construction_zone_ids=all_zones,
            manifest_sha256=self.manifest_sha256,
        )
        if self.mapping_sha256 != expected:
            raise ValueError("cell-zone mapping differs from its score-blind receipt")
        object.__setattr__(self, "cell_ids", cell_ids)
        object.__setattr__(self, "construction_zone_ids", zones)
        object.__setattr__(self, "all_construction_zone_ids", all_zones)

    @classmethod
    def bind(
        cls,
        *,
        grid_id: str,
        cell_ids: Sequence[str],
        construction_zone_ids: Sequence[str],
        all_construction_zone_ids: Sequence[str] | None = None,
        manifest_sha256: str,
    ) -> FrozenCellZoneMapping:
        cells = tuple(cell_ids)
        zones = tuple(construction_zone_ids)
        all_zones = (
            tuple(sorted(set(zones)))
            if all_construction_zone_ids is None
            else tuple(all_construction_zone_ids)
        )
        return cls(
            grid_id=grid_id,
            cell_ids=cells,
            construction_zone_ids=zones,
            all_construction_zone_ids=all_zones,
            manifest_sha256=manifest_sha256,
            mapping_sha256=_cell_zone_mapping_sha256(
                grid_id=grid_id,
                cell_ids=cells,
                construction_zone_ids=zones,
                all_construction_zone_ids=all_zones,
                manifest_sha256=manifest_sha256,
            ),
        )

    def verify_grid(self, grid_family: Stage4GridFamily) -> None:
        primary = grid_family.primary_25km
        if self.grid_id != primary.grid_id or self.cell_ids != primary.cell_ids:
            raise ValueError("cell-zone mapping differs from the frozen primary grid")
        if self.mapping_sha256 != _cell_zone_mapping_sha256(
            grid_id=self.grid_id,
            cell_ids=self.cell_ids,
            construction_zone_ids=self.construction_zone_ids,
            all_construction_zone_ids=self.all_construction_zone_ids,
            manifest_sha256=self.manifest_sha256,
        ):
            raise ValueError("cell-zone mapping content changed")


EvaluationCellZoneReceipt: TypeAlias = FrozenCellZoneMapping


@dataclass(frozen=True, slots=True)
class TargetBlindFormalContext:
    """Complete pre-authorization context with no catalogue or event assignment field."""

    protocol: VerifiedFormalProtocol
    scoring_plan: Stage4ScoringPlan
    study_area: StudyArea
    grid_family: Stage4GridFamily
    background_domain: BackgroundDomainBinding
    verified_issues: tuple[VerifiedStage3Issue, ...]
    feature_layout: FeatureLayout
    cell_supports: tuple[FrozenCellSupport, ...]
    prospective_plans: tuple[ProspectiveIssuePlan, ...]
    cell_zone_mapping: FrozenCellZoneMapping

    def __post_init__(self) -> None:
        object.__setattr__(self, "verified_issues", tuple(self.verified_issues))
        object.__setattr__(self, "cell_supports", tuple(self.cell_supports))
        object.__setattr__(self, "prospective_plans", tuple(self.prospective_plans))
        self.verify()

    def verify(self) -> None:
        if self.scoring_plan.protocol_design_sha256 != self.protocol.protocol_design_sha256:
            raise ValueError("scoring plan and verified protocol design hashes differ")
        if self.scoring_plan.random_input_seal_sha256 != self.protocol.random_input_seal_sha256:
            raise ValueError("scoring plan and verified random-input seals differ")
        scopes = tuple(self.scoring_plan.fit_scopes)
        if tuple(_scope_evaluation_id(scope) for scope in scopes) != _EVALUATION_IDS:
            raise ValueError("score-blind context requires the four frozen scopes in order")
        if self.study_area.equal_area_crs != EQUAL_AREA_CRS:
            raise ValueError("score-blind context study-area CRS changed")
        grids = self.grid_family.grids()
        if tuple(grid.grid_id for grid in grids) != self.background_domain.grid_ids:
            raise ValueError("background domain differs from the frozen grid family")
        if self.background_domain.study_area_sha256 != (
            self.protocol.development_snapshot.study_area_sha256
        ):
            raise ValueError("background domain differs from the verified study-area identity")
        if self.background_domain.compensator_domain_id != (
            self.protocol.development_snapshot.compensator_domain_id
        ):
            raise ValueError("background domain differs from the verified compensator domain")
        reference_area = self.grid_family.reference_12_5km.total_area_km2
        if not math.isclose(
            self.background_domain.supported_area_km2,
            reference_area,
            rel_tol=1.0e-12,
            abs_tol=1.0e-6,
        ):
            raise ValueError("background domain area differs from the frozen grid family")
        self.cell_zone_mapping.verify_grid(self.grid_family)
        primary = self.grid_family.primary_25km
        sources = tuple(self.feature_layout.dynamic_sources)
        if not self.verified_issues:
            raise ValueError("score-blind context requires verified stage-3 issues")
        for issue in self.verified_issues:
            issue.verify(primary_grid=primary, source_columns=sources)
        if not self.prospective_plans:
            raise ValueError("score-blind context requires prospective issue plans")
        prospective_issue_times = {
            datetime.combine(plan.issue_date_local, time.min, tzinfo=_SHANGHAI).astimezone(UTC)
            for plan in self.prospective_plans
        }
        if not prospective_issue_times <= {issue.issue_time_utc for issue in self.verified_issues}:
            raise ValueError("prospective plans lack verified stage-3 issue inputs")
        supports = tuple(self.cell_supports)
        if tuple(item.evaluation_id for item in supports) != _EVALUATION_IDS:
            raise ValueError("cell supports must follow the four frozen scope identities")
        for support in supports:
            expected = self.protocol.snapshot_for(support.evaluation_id)
            if support.grid_id != primary.grid_id or support.cell_ids != primary.cell_ids:
                raise ValueError("cell support differs from the frozen primary grid")
            if support.support_id != expected.support_id:
                raise ValueError("cell support differs from the frozen background support")
            if support.compensator_domain_id != expected.compensator_domain_id:
                raise ValueError("cell support differs from the frozen compensator domain")


def _zone_receipt_sha256(
    *,
    assignment_sha256: str,
    mapping_sha256: str,
    catalog_source_content_sha256: str,
    catalog_source_schema_sha256: str,
    event_ids: tuple[str, ...],
    construction_zone_ids: tuple[str, ...],
    all_construction_zone_ids: tuple[str, ...],
) -> str:
    return canonical_mapping_sha256(
        {
            "schema_version": 1,
            "role": "local_evaluation_construction_zone_strata",
            "assignment_sha256": assignment_sha256,
            "mapping_sha256": mapping_sha256,
            "catalog_source_content_sha256": catalog_source_content_sha256,
            "catalog_source_schema_sha256": catalog_source_schema_sha256,
            "all_construction_zone_ids": list(all_construction_zone_ids),
            "event_zone_pairs": [
                {"event_id": event_id, "construction_zone_id": zone_id}
                for event_id, zone_id in zip(event_ids, construction_zone_ids, strict=True)
            ],
            "public_export_forbidden": True,
        }
    )


@dataclass(frozen=True, slots=True)
class EvaluationZoneReceipt:
    """Local event-to-zone strata for diagnostics only, never model input or public data."""

    assignment_sha256: str
    mapping_sha256: str
    catalog_source_content_sha256: str
    catalog_source_schema_sha256: str
    event_ids: tuple[str, ...]
    construction_zone_ids: tuple[str, ...]
    all_construction_zone_ids: tuple[str, ...]
    receipt_sha256: str
    diagnostic_role: Literal["evaluation-only-no-fit-feature-order-or-candidate"] = (
        "evaluation-only-no-fit-feature-order-or-candidate"
    )
    public_export_forbidden: Literal[True] = True

    def __post_init__(self) -> None:
        for label, value in (
            ("assignment_sha256", self.assignment_sha256),
            ("mapping_sha256", self.mapping_sha256),
            ("catalog_source_content_sha256", self.catalog_source_content_sha256),
            ("catalog_source_schema_sha256", self.catalog_source_schema_sha256),
            ("receipt_sha256", self.receipt_sha256),
        ):
            _sha256(value, label=label)
        events = tuple(self.event_ids)
        zones = tuple(self.construction_zone_ids)
        all_zones = tuple(self.all_construction_zone_ids)
        if len(events) != len(zones) or len(events) != len(set(events)):
            raise ValueError("evaluation zone receipt requires one zone per physical event")
        for event_id in events:
            _identifier(event_id, label="evaluation event_id")
        for zone_id in zones:
            _identifier(zone_id, label="evaluation construction_zone_id")
        if len(all_zones) != 39 or len(all_zones) != len(set(all_zones)):
            raise ValueError("evaluation receipt requires all 39 score-blind zone identities")
        for zone_id in all_zones:
            _identifier(zone_id, label="evaluation all_construction_zone_ids")
        if not set(zones) <= set(all_zones):
            raise ValueError("an event zone lies outside the score-blind zone list")
        if self.diagnostic_role != "evaluation-only-no-fit-feature-order-or-candidate":
            raise ValueError("construction-zone receipt escaped its evaluation-only role")
        if self.public_export_forbidden is not True:
            raise ValueError("construction-zone event strata must remain local")
        expected = _zone_receipt_sha256(
            assignment_sha256=self.assignment_sha256,
            mapping_sha256=self.mapping_sha256,
            catalog_source_content_sha256=self.catalog_source_content_sha256,
            catalog_source_schema_sha256=self.catalog_source_schema_sha256,
            event_ids=events,
            construction_zone_ids=zones,
            all_construction_zone_ids=all_zones,
        )
        if self.receipt_sha256 != expected:
            raise ValueError("evaluation zone receipt content changed")
        object.__setattr__(self, "event_ids", events)
        object.__setattr__(self, "construction_zone_ids", zones)
        object.__setattr__(self, "all_construction_zone_ids", all_zones)


def _evaluation_zone_receipt(
    assignments: BoundTargetCellAssignments,
    mapping: FrozenCellZoneMapping,
) -> EvaluationZoneReceipt:
    mapping_by_cell = dict(zip(mapping.cell_ids, mapping.construction_zone_ids, strict=True))
    event_ids = assignments.assignments.event_ids
    zones = tuple(mapping_by_cell[cell_id] for cell_id in assignments.assignments.cell_ids)
    receipt_sha256 = _zone_receipt_sha256(
        assignment_sha256=assignments.assignment_sha256,
        mapping_sha256=mapping.mapping_sha256,
        catalog_source_content_sha256=assignments.catalog_source_content_sha256,
        catalog_source_schema_sha256=assignments.catalog_source_schema_sha256,
        event_ids=event_ids,
        construction_zone_ids=zones,
        all_construction_zone_ids=mapping.all_construction_zone_ids,
    )
    return EvaluationZoneReceipt(
        assignment_sha256=assignments.assignment_sha256,
        mapping_sha256=mapping.mapping_sha256,
        catalog_source_content_sha256=assignments.catalog_source_content_sha256,
        catalog_source_schema_sha256=assignments.catalog_source_schema_sha256,
        event_ids=event_ids,
        construction_zone_ids=zones,
        all_construction_zone_ids=mapping.all_construction_zone_ids,
        receipt_sha256=receipt_sha256,
    )


def _verify_rebuilt_backgrounds(
    backgrounds: tuple[Stage4BackgroundFit, ...],
    *,
    context: TargetBlindFormalContext,
    catalog: Stage4TargetCatalog,
) -> None:
    if (
        len(backgrounds) != 4
        or tuple(fit.snapshot.evaluation_id for fit in backgrounds) != _EVALUATION_IDS
    ):
        raise ValueError("rebuilt backgrounds differ from the four frozen scopes")
    catalog_index = {str(event_id): index for index, event_id in enumerate(catalog.event_id)}
    for evaluation_id, fit in zip(_EVALUATION_IDS, backgrounds, strict=True):
        expected = context.protocol.snapshot_for(evaluation_id)
        if fit.snapshot != expected:
            raise ValueError("rebuilt background snapshot differs from the verified protocol")
        if fit.causal_audit.post_cutoff_training_event_count != 0:
            raise ValueError("rebuilt background contains post-cutoff training events")
        if (
            fit.causal_audit.latest_training_origin_utc > fit.snapshot.fit_end_utc
            or fit.causal_audit.latest_training_available_at_utc > fit.snapshot.fit_end_utc
        ):
            raise ValueError("rebuilt background causal audit exceeds its snapshot cutoff")
        for event_id in fit.training_event_ids:
            try:
                index = catalog_index[event_id]
            except KeyError as exc:
                raise ValueError(
                    "rebuilt background references an unknown catalogue event"
                ) from exc
            if (
                catalog.origin_time_utc[index] > fit.snapshot.fit_end_utc
                or catalog.available_at_utc[index] > fit.snapshot.fit_end_utc
            ):
                raise ValueError("validation-era catalogue data backfilled an earlier background")
    development = backgrounds[:3]
    first = development[0]
    if any(
        fit.snapshot.parameter_snapshot_id != first.snapshot.parameter_snapshot_id
        or fit.snapshot.fit_end_utc != first.snapshot.fit_end_utc
        or fit.training_event_ids != first.training_event_ids
        or fit.scientific_identity_sha256 != first.scientific_identity_sha256
        for fit in development[1:]
    ):
        raise ValueError("development folds did not retain one causal background snapshot")
    validation = backgrounds[3]
    if first.snapshot.fit_end_utc >= validation.snapshot.fit_end_utc:
        raise ValueError("development background was backfilled from validation")
    if first.snapshot.parameter_snapshot_id == validation.snapshot.parameter_snapshot_id:
        raise ValueError("development background reused validation parameters")


def _formal_pool_issue_ids(scope_plan: FitScopePlan) -> tuple[str, ...]:
    pools = scope_plan.time_permutation_pools
    return (*pools.fit_issue_ids, *pools.assessment_issue_ids)


def _recipient_issue_id(issue: VerifiedStage3Issue) -> str:
    issue_date = issue.issue_time_utc.astimezone(_SHANGHAI).date()
    return f"anomaly-issue-{issue_date.isoformat()}"


def _formal_recipient_issues(
    scope_plan: FitScopePlan,
    verified_issues: Sequence[VerifiedStage3Issue],
) -> tuple[tuple[str, VerifiedStage3Issue], ...]:
    pool_ids = _formal_pool_issue_ids(scope_plan)
    issues = tuple(verified_issues)
    output: list[tuple[str, VerifiedStage3Issue]] = []
    for issue_id in pool_ids:
        matches = tuple(
            issue
            for issue in issues
            if issue_id in {issue.issue_report_id, _recipient_issue_id(issue)}
        )
        if len(matches) != 1:
            raise ValueError(
                f"formal placebo pool issue must resolve to one verified recipient: {issue_id}"
            )
        output.append((issue_id, matches[0]))
    recipients = tuple(item.binding_sha256 for _, item in output)
    if len(recipients) != len(set(recipients)):
        raise ValueError("one verified recipient was assigned to multiple placebo issue IDs")
    return tuple(output)


def _formal_scope_plan_payload(scope_plan: FitScopePlan) -> dict[str, object]:
    return {
        "fit_scope_id": scope_plan.fit_scope_id,
        "role": scope_plan.role,
        "fit_exposures_7d": [item.identifier for item in scope_plan.fit_exposures_7d],
        "assessments": [
            {
                "horizon_days": item.horizon_days,
                "exposures": [exposure.identifier for exposure in item.exposures],
            }
            for item in scope_plan.assessments
        ],
        "time_fit_issue_ids": list(scope_plan.time_permutation_pools.fit_issue_ids),
        "time_assessment_issue_ids": list(scope_plan.time_permutation_pools.assessment_issue_ids),
        "validation_refit_forbidden": scope_plan.validation_refit_forbidden,
    }


def _placebo_assembly_context_sha256(
    *,
    protocol_design_sha256: str,
    random_input_seal_sha256: str,
    scope_plan: FitScopePlan,
    target_assignments: BoundTargetCellAssignments,
    primary_grid: Stage4IntegrationGrid,
    background: FrozenBackgroundField,
    background_scientific_identity_sha256: str,
    support: FrozenCellSupport,
    feature_layout: FeatureLayout,
    recipient_issues: tuple[tuple[str, VerifiedStage3Issue], ...],
) -> str:
    return canonical_mapping_sha256(
        {
            "schema_version": 1,
            "role": "authorized_formal_placebo_scope_assembler",
            "protocol_design_sha256": protocol_design_sha256,
            "random_input_seal_sha256": random_input_seal_sha256,
            "scope_plan": _formal_scope_plan_payload(scope_plan),
            "catalog_source_content_sha256": (target_assignments.catalog_source_content_sha256),
            "catalog_source_schema_sha256": target_assignments.catalog_source_schema_sha256,
            "catalog_event_order_sha256": target_assignments.catalog_event_order_sha256,
            "target_assignment_sha256": target_assignments.assignment_sha256,
            "primary_grid_id": primary_grid.grid_id,
            "primary_cell_ids": list(primary_grid.cell_ids),
            "background_snapshot_id": background.snapshot.snapshot_id,
            "background_parameter_snapshot_id": background.snapshot.parameter_snapshot_id,
            "background_scientific_identity_sha256": (background_scientific_identity_sha256),
            "support_id": support.support_id,
            "compensator_domain_id": support.compensator_domain_id,
            "supported_cell_mask": support.supported_cell_mask.tolist(),
            "feature_layout": {
                "coverage_sources": list(feature_layout.coverage_sources),
                "snapshot_sources": list(feature_layout.snapshot_sources),
                "dynamic_sources": list(feature_layout.dynamic_sources),
            },
            "recipient_issues": [
                {
                    "issue_id": issue_id,
                    "issue_time_utc": issue.issue_time_utc.isoformat(),
                    "issue_report_id": issue.issue_report_id,
                    "state_snapshot_id": issue.state_snapshot_id,
                    "lineage_digest": issue.lineage_digest,
                    "observed_binding_sha256": issue.binding_sha256,
                }
                for issue_id, issue in recipient_issues
            ],
        }
    )


@dataclass(frozen=True, slots=True)
class AuthorizedPlaceboScopeAssembler:
    """Pure in-memory callable that reassembles one formal placebo replicate."""

    assembly_context_sha256: str
    protocol_design_sha256: str = field(repr=False)
    random_input_seal_sha256: str = field(repr=False)
    scope_plan: FitScopePlan = field(repr=False)
    catalog: Stage4TargetCatalog = field(repr=False, compare=False)
    target_assignments: BoundTargetCellAssignments = field(repr=False)
    primary_grid: Stage4IntegrationGrid = field(repr=False)
    background: FrozenBackgroundField = field(repr=False)
    background_scientific_identity_sha256: str = field(repr=False)
    support: FrozenCellSupport = field(repr=False)
    feature_layout: FeatureLayout = field(repr=False)
    recipient_issues: tuple[tuple[str, VerifiedStage3Issue], ...] = field(repr=False)

    def __post_init__(self) -> None:
        _sha256(self.assembly_context_sha256, label="assembly_context_sha256")
        _sha256(self.protocol_design_sha256, label="protocol_design_sha256")
        _sha256(self.random_input_seal_sha256, label="random_input_seal_sha256")
        _sha256(
            self.background_scientific_identity_sha256,
            label="background_scientific_identity_sha256",
        )
        if _scope_evaluation_id(self.scope_plan) != "formal-validation":
            raise ValueError("placebo assembler is restricted to formal-validation")
        recipients = tuple(self.recipient_issues)
        if tuple(item[0] for item in recipients) != _formal_pool_issue_ids(self.scope_plan):
            raise ValueError("placebo recipients differ from the frozen formal issue pools")
        if self.background.evaluation_id != "formal-validation":
            raise ValueError("placebo assembler requires the formal validation background")
        if self.support.evaluation_id != "formal-validation":
            raise ValueError("placebo assembler requires the formal validation support")
        if (
            self.background.grid_id != self.primary_grid.grid_id
            or self.support.grid_id != self.primary_grid.grid_id
            or self.background.cell_ids != self.primary_grid.cell_ids
            or self.support.cell_ids != self.primary_grid.cell_ids
        ):
            raise ValueError("placebo assembler grid/background/support identities differ")
        self.target_assignments.verify(self.catalog, primary_grid=self.primary_grid)
        sources = tuple(self.feature_layout.dynamic_sources)
        for _, recipient in recipients:
            recipient.verify(primary_grid=self.primary_grid, source_columns=sources)
        expected = _placebo_assembly_context_sha256(
            protocol_design_sha256=self.protocol_design_sha256,
            random_input_seal_sha256=self.random_input_seal_sha256,
            scope_plan=self.scope_plan,
            target_assignments=self.target_assignments,
            primary_grid=self.primary_grid,
            background=self.background,
            background_scientific_identity_sha256=(self.background_scientific_identity_sha256),
            support=self.support,
            feature_layout=self.feature_layout,
            recipient_issues=recipients,
        )
        if self.assembly_context_sha256 != expected:
            raise ValueError("placebo assembly context differs from its frozen identity")
        object.__setattr__(self, "recipient_issues", recipients)

    @property
    def issue_ids(self) -> tuple[str, ...]:
        return tuple(item[0] for item in self.recipient_issues)

    def bind_recipient_issues(
        self,
        issue_tables: Mapping[str, pa.Table],
    ) -> tuple[VerifiedStage3Issue, ...]:
        """Rebind rebuilt values to the original recipient identities and lineage."""

        tables = dict(issue_tables)
        if set(tables) != set(self.issue_ids) or len(tables) != len(self.issue_ids):
            raise ValueError("placebo tables must exactly cover the formal recipient issue pool")
        sources = tuple(self.feature_layout.dynamic_sources)
        return tuple(
            recipient.rebind_recipient_table(
                tables[issue_id],
                primary_grid=self.primary_grid,
                source_columns=sources,
            )
            for issue_id, recipient in self.recipient_issues
        )

    def __call__(self, issue_tables: Mapping[str, pa.Table]) -> EvaluationScope:
        rebound = self.bind_recipient_issues(issue_tables)
        bundle = assemble_evaluation_scope(
            self.scope_plan,
            catalog=self.catalog,
            target_assignments=self.target_assignments,
            verified_issues=rebound,
            primary_grid=self.primary_grid,
            background=self.background,
            support=self.support,
            feature_layout=self.feature_layout,
        )
        if bundle.scope.fit.evaluation_id != "formal-validation":
            raise ValueError("placebo assembler returned another evaluation scope")
        return bundle.scope


def _authorized_placebo_scope_assembler(
    *,
    context: TargetBlindFormalContext,
    catalog: Stage4TargetCatalog,
    target_assignments: BoundTargetCellAssignments,
    background_fit: Stage4BackgroundFit,
) -> AuthorizedPlaceboScopeAssembler:
    scope_plan = tuple(context.scoring_plan.fit_scopes)[3]
    recipients = _formal_recipient_issues(scope_plan, context.verified_issues)
    primary_grid = context.grid_family.primary_25km
    background = FrozenBackgroundField.from_background_fit(background_fit)
    support = context.cell_supports[3]
    identity = _placebo_assembly_context_sha256(
        protocol_design_sha256=context.protocol.protocol_design_sha256,
        random_input_seal_sha256=context.protocol.random_input_seal_sha256,
        scope_plan=scope_plan,
        target_assignments=target_assignments,
        primary_grid=primary_grid,
        background=background,
        background_scientific_identity_sha256=background_fit.scientific_identity_sha256,
        support=support,
        feature_layout=context.feature_layout,
        recipient_issues=recipients,
    )
    return AuthorizedPlaceboScopeAssembler(
        assembly_context_sha256=identity,
        protocol_design_sha256=context.protocol.protocol_design_sha256,
        random_input_seal_sha256=context.protocol.random_input_seal_sha256,
        scope_plan=scope_plan,
        catalog=catalog,
        target_assignments=target_assignments,
        primary_grid=primary_grid,
        background=background,
        background_scientific_identity_sha256=background_fit.scientific_identity_sha256,
        support=support,
        feature_layout=context.feature_layout,
        recipient_issues=recipients,
    )


@dataclass(frozen=True, slots=True)
class AuthorizedFormalMaterialization:
    """Local result of the one authorized in-memory materialization boundary."""

    assembly: Stage4FormalAssembly
    background_fits: tuple[Stage4BackgroundFit, ...]
    cell_assignments: BoundTargetCellAssignments
    evaluation_cell_zone_mapping: EvaluationCellZoneReceipt
    evaluation_zone_receipt: EvaluationZoneReceipt
    evaluation_region_binding: EvaluationRegionBinding
    placebo_scope_assembler: AuthorizedPlaceboScopeAssembler = field(
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        backgrounds = tuple(self.background_fits)
        if tuple(item.snapshot.evaluation_id for item in backgrounds) != _EVALUATION_IDS:
            raise ValueError("materialized backgrounds differ from the four frozen scopes")
        if tuple(item.evaluation_id for item in self.assembly.audits) != _EVALUATION_IDS:
            raise ValueError("materialized assembly differs from the four frozen scopes")
        if self.evaluation_zone_receipt.assignment_sha256 != (
            self.cell_assignments.assignment_sha256
        ):
            raise ValueError("evaluation strata and event-cell assignments differ")
        if self.evaluation_zone_receipt.mapping_sha256 != (
            self.evaluation_cell_zone_mapping.mapping_sha256
        ):
            raise ValueError("cell and event construction-zone receipts differ")
        if self.evaluation_zone_receipt.all_construction_zone_ids != (
            self.evaluation_cell_zone_mapping.all_construction_zone_ids
        ):
            raise ValueError("cell and event receipts use different complete zone lists")
        binding = self.evaluation_region_binding
        if binding.all_construction_zone_ids != (
            self.evaluation_cell_zone_mapping.all_construction_zone_ids
        ):
            raise ValueError("pipeline region binding and local receipts use different zones")
        if (
            binding.cell_ids != self.evaluation_cell_zone_mapping.cell_ids
            or binding.cell_construction_zone_ids
            != self.evaluation_cell_zone_mapping.construction_zone_ids
            or binding.cell_mapping_sha256 != self.evaluation_cell_zone_mapping.mapping_sha256
        ):
            raise ValueError("pipeline region binding differs from the frozen cell receipt")
        if (
            binding.event_ids != self.evaluation_zone_receipt.event_ids
            or binding.event_construction_zone_ids
            != self.evaluation_zone_receipt.construction_zone_ids
            or binding.event_mapping_sha256 != self.evaluation_zone_receipt.receipt_sha256
        ):
            raise ValueError("pipeline region binding differs from the event receipt")
        assembler = self.placebo_scope_assembler
        if (
            assembler.target_assignments.assignment_sha256
            != self.cell_assignments.assignment_sha256
            or assembler.background_scientific_identity_sha256
            != backgrounds[3].scientific_identity_sha256
        ):
            raise ValueError("placebo assembler differs from the authorized formal materialization")
        object.__setattr__(self, "background_fits", backgrounds)

    @property
    def assembly_context_sha256(self) -> str:
        return self.placebo_scope_assembler.assembly_context_sha256


@dataclass(frozen=True, slots=True)
class FormalPlaceboSourceWiring:
    """Exact assembler fields to pass into a post-authorization placebo universe."""

    fit_issue_ids: tuple[str, ...]
    assessment_issue_ids: tuple[str, ...]
    assembly_context_sha256: str
    scope_assembler: AuthorizedPlaceboScopeAssembler = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        fit = tuple(self.fit_issue_ids)
        assessment = tuple(self.assessment_issue_ids)
        pools = self.scope_assembler.scope_plan.time_permutation_pools
        if fit != pools.fit_issue_ids or assessment != pools.assessment_issue_ids:
            raise ValueError("placebo source wiring differs from the formal issue pools")
        if set(fit) & set(assessment):
            raise ValueError("placebo source wiring crossed fit and assessment pools")
        if self.assembly_context_sha256 != (self.scope_assembler.assembly_context_sha256):
            raise ValueError("placebo source wiring uses another assembly context")
        object.__setattr__(self, "fit_issue_ids", fit)
        object.__setattr__(self, "assessment_issue_ids", assessment)


def build_formal_placebo_source_wiring(
    materialization: AuthorizedFormalMaterialization,
) -> FormalPlaceboSourceWiring:
    """Project only the four fields consumed by ``PlaceboSourceUniverse``."""

    if not isinstance(materialization, AuthorizedFormalMaterialization):
        raise TypeError("materialization must be an authorized formal materialization")
    assembler = materialization.placebo_scope_assembler
    pools = assembler.scope_plan.time_permutation_pools
    return FormalPlaceboSourceWiring(
        fit_issue_ids=pools.fit_issue_ids,
        assessment_issue_ids=pools.assessment_issue_ids,
        assembly_context_sha256=assembler.assembly_context_sha256,
        scope_assembler=assembler,
    )


def _required_magnitude_horizon_pairs(
    horizons: tuple[int, ...],
) -> set[tuple[int, str]]:
    return {(horizon, magnitude) for horizon in horizons for magnitude in ("M5_6", "M6_plus")}


def build_stage4_in_memory_plan(
    materialization: AuthorizedFormalMaterialization,
    *,
    feature_contracts: Sequence[FeatureColumnContract],
    feature_layout: FeatureLayout,
    frozen_input_seal_sha256: str,
    model_version: str,
) -> Stage4InMemoryPlan:
    """Bridge authorized formal assembly into the target-agnostic scoring pipeline."""

    if not isinstance(materialization, AuthorizedFormalMaterialization):
        raise TypeError("materialization must be an authorized formal materialization")
    if feature_layout != materialization.placebo_scope_assembler.feature_layout:
        raise ValueError("pipeline feature layout differs from the authorized assembly")
    if frozen_input_seal_sha256 != (
        materialization.placebo_scope_assembler.random_input_seal_sha256
    ):
        raise ValueError("pipeline frozen input seal differs from the authorized assembly")
    assembly = materialization.assembly
    expected_development = _required_magnitude_horizon_pairs((7, 30, 90))
    for scope in assembly.development_scopes:
        observed = {(item.horizon_days, item.magnitude_bin) for item in scope.exposures}
        if observed != expected_development:
            raise ValueError("development assembly lost a frozen magnitude/horizon window")
    expected_formal = _required_magnitude_horizon_pairs((7, 30, 90, 180, 365))
    observed_formal = {
        (item.horizon_days, item.magnitude_bin) for item in assembly.formal_scope.exposures
    }
    if observed_formal != expected_formal:
        raise ValueError("formal assembly lost a frozen M5/M6 magnitude/horizon window")
    if not assembly.prospective_issues:
        raise ValueError("pipeline requires physically separate prospective inputs")
    return Stage4InMemoryPlan(
        feature_contracts=tuple(feature_contracts),
        feature_layout=feature_layout,
        development_scopes=assembly.development_scopes,
        formal_scope=assembly.formal_scope,
        prospective_issues=assembly.prospective_issues,
        evaluation_region_binding=materialization.evaluation_region_binding,
        frozen_input_seal_sha256=frozen_input_seal_sha256,
        model_version=model_version,
    )


def _materialize_in_memory_core(
    context: TargetBlindFormalContext,
    catalog: Stage4TargetCatalog,
) -> AuthorizedFormalMaterialization:
    """Synthetic-capable, I/O-free core used only behind a guarded formal boundary."""

    context.verify()
    protocol = context.protocol.background_protocol()
    backgrounds = tuple(
        rebuild_stage4_background(
            catalog,
            context.grid_family,
            protocol=protocol,
            evaluation_id=evaluation_id,
            domain=context.background_domain,
        )
        for evaluation_id in _EVALUATION_IDS
    )
    _verify_rebuilt_backgrounds(backgrounds, context=context, catalog=catalog)
    primary = context.grid_family.primary_25km
    assignments = BoundTargetCellAssignments.bind(
        catalog,
        map_targets_to_frozen_primary_grid(
            catalog,
            study_area=context.study_area,
            primary_grid=primary,
        ),
        primary_grid=primary,
    )
    zone_receipt = _evaluation_zone_receipt(assignments, context.cell_zone_mapping)
    region_binding = EvaluationRegionBinding(
        all_construction_zone_ids=context.cell_zone_mapping.all_construction_zone_ids,
        cell_ids=context.cell_zone_mapping.cell_ids,
        cell_construction_zone_ids=context.cell_zone_mapping.construction_zone_ids,
        event_ids=zone_receipt.event_ids,
        event_construction_zone_ids=zone_receipt.construction_zone_ids,
        cell_mapping_sha256=context.cell_zone_mapping.mapping_sha256,
        event_mapping_sha256=zone_receipt.receipt_sha256,
    )
    assembly = assemble_stage4_formal_inputs(
        context.scoring_plan,
        catalog=catalog,
        target_assignments=assignments,
        verified_issues=context.verified_issues,
        primary_grid=primary,
        backgrounds=tuple(FrozenBackgroundField.from_background_fit(item) for item in backgrounds),
        cell_supports=context.cell_supports,
        feature_layout=context.feature_layout,
        prospective_plans=context.prospective_plans,
    )
    placebo_assembler = _authorized_placebo_scope_assembler(
        context=context,
        catalog=catalog,
        target_assignments=assignments,
        background_fit=backgrounds[3],
    )
    return AuthorizedFormalMaterialization(
        assembly=assembly,
        background_fits=backgrounds,
        cell_assignments=assignments,
        evaluation_cell_zone_mapping=context.cell_zone_mapping,
        evaluation_zone_receipt=zone_receipt,
        evaluation_region_binding=region_binding,
        placebo_scope_assembler=placebo_assembler,
    )


def materialize_after_authorized_target(
    context: TargetBlindFormalContext,
    catalog: Stage4TargetCatalog,
    *,
    execution_protocol: Mapping[str, object],
    authorization: Stage4TargetAuthorization,
) -> AuthorizedFormalMaterialization:
    """Materialize formal inputs only with the current protocol and real capability.

    The R2 execution guard deliberately runs before either caller-supplied object is
    inspected.  Consequently the currently blocked protocol cannot trigger catalogue
    mapping, background rebuilding, or any other target-derived work.  Unit tests for
    the scientific in-memory transformation use the private core above; it is not a
    formal execution entrance.
    """

    require_stage4_r2_execution_action(
        execution_protocol,
        action="formal_scoring",
    )
    require_stage4_target_authorization(authorization)
    if protocol_design_sha256(execution_protocol) != (context.protocol.protocol_design_sha256):
        raise ValueError("formal materialization protocol differs from its verified context")
    return _materialize_in_memory_core(context, catalog)


__all__ = [
    "AuthorizedFormalMaterialization",
    "AuthorizedPlaceboScopeAssembler",
    "EvaluationCellZoneReceipt",
    "EvaluationZoneReceipt",
    "FormalPlaceboSourceWiring",
    "FrozenCellZoneMapping",
    "TargetBlindFormalContext",
    "VerifiedFormalProtocol",
    "build_formal_placebo_source_wiring",
    "build_stage4_in_memory_plan",
    "materialize_after_authorized_target",
]
