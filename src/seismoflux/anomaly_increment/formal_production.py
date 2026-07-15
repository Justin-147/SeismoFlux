"""Target-blind production assembly for the sole stage-4 formal entry point.

This module may authenticate and assemble frozen inputs, but it deliberately
contains no target reader and no scoring call.  The one executable entry point
is ``scripts/run_stage4_formal.py``; only its explicit ``run`` branch invokes
``run_formal_stage4``.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Final, cast

import numpy as np
import pyarrow.parquet as pq

from seismoflux.anomaly_increment.authorization import (
    Stage4ScoringSeal,
    Stage4TargetReadinessEvidence,
    authorize_stage4_target_access,
    load_stage4_scoring_seal,
    verify_stage4_target_readiness,
)
from seismoflux.anomaly_increment.background_adapter import BackgroundDomainBinding
from seismoflux.anomaly_increment.compute import Stage4ComputePlan, build_compute_plan
from seismoflux.anomaly_increment.config import (
    STAGE4_CHECKPOINT_ROOT_RELATIVE_PATH,
    Stage4ProtocolBundle,
    load_stage4_protocol_bundle,
    stage4_scoring_freeze_relative_path,
)
from seismoflux.anomaly_increment.convergence import (
    build_target_blind_convergence_inputs,
)
from seismoflux.anomaly_increment.feature_adapter import feature_set_contract
from seismoflux.anomaly_increment.formal_assembly import (
    EvaluationId,
    FrozenCellSupport,
)
from seismoflux.anomaly_increment.formal_execution import (
    FrozenCellZoneMapping,
    TargetBlindFormalContext,
    VerifiedFormalProtocol,
)
from seismoflux.anomaly_increment.formal_preflight import (
    FormalPreflightBundle,
    FormalPreflightReceipt,
    load_formal_preflight,
    load_formal_preflight_receipt,
)
from seismoflux.anomaly_increment.formal_run import (
    FormalPreflightArtifacts,
    FormalRunInputs,
    FrozenPlaceboInputs,
    PlaceboConcurrencyPlan,
    Stage4SpatialArtifactHook,
)
from seismoflux.anomaly_increment.immutable_file import (
    UnsafeImmutableFileError,
    open_existing_immutable_file,
)
from seismoflux.anomaly_increment.qualification import (
    ScoreBlindInputEvidence,
    observe_score_blind_inputs,
    validate_stage4_qualification_against_formal_preflight,
    validate_stage4_qualification_against_protocol,
)
from seismoflux.anomaly_increment.runner import build_stage4_scoring_plan
from seismoflux.anomaly_increment.score_blind_path import (
    require_score_blind_project_path,
)
from seismoflux.background.execution import detect_physical_core_count

FORMAL_CHECKPOINT_RELATIVE_PATH: Final[PurePosixPath] = PurePosixPath(
    STAGE4_CHECKPOINT_ROOT_RELATIVE_PATH
)
_CELL_MAPPING_COLUMNS: Final[tuple[str, ...]] = (
    "grid_id",
    "cell_id",
    "cell_row",
    "cell_column",
    "query_x_m",
    "query_y_m",
    "construction_zone_id",
)
_EVALUATION_IDS: Final[tuple[EvaluationId, ...]] = (
    "development-fold-1",
    "development-fold-2",
    "development-fold-3",
    "formal-validation",
)


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _project_path(root: Path, value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} must be a non-empty POSIX project-relative path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must stay inside the project root")
    path = Path(os.path.abspath(os.fspath(root.joinpath(*relative.parts))))
    if not path.is_relative_to(root):
        raise ValueError(f"{label} escapes the project root")
    return path


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(os.path.abspath(os.fspath(left))) == os.path.normcase(
        os.path.abspath(os.fspath(right))
    )


def _scoring_seal_path(protocol: Stage4ProtocolBundle) -> Path:
    return _frozen_scoring_path(
        protocol,
        key="required_seal_path",
        label="canonical stage-4 scoring seal",
    )


def _frozen_scoring_path(
    protocol: Stage4ProtocolBundle,
    *,
    key: str,
    label: str,
) -> Path:
    relative = stage4_scoring_freeze_relative_path(protocol.protocol, key)
    path = protocol.repository_root.resolve().joinpath(*relative.parts)
    return require_score_blind_project_path(
        protocol.repository_root,
        protocol.protocol,
        path,
        label=label,
    )


def _cell_mapping_path(protocol: Stage4ProtocolBundle) -> Path:
    topology = _mapping(
        protocol.protocol.get("spatial_permutation_topology"),
        label="spatial_permutation_topology",
    )
    local = _mapping(
        topology.get("local_restricted_artifacts"),
        label="spatial_permutation_topology.local_restricted_artifacts",
    )
    path = _project_path(
        protocol.repository_root.resolve(),
        local.get("cell_mapping"),
        label="local_restricted_artifacts.cell_mapping",
    )
    return require_score_blind_project_path(
        protocol.repository_root,
        protocol.protocol,
        path,
        label="canonical stage-4 cell mapping",
    )


def _formal_preflight_receipt_path(protocol: Stage4ProtocolBundle) -> Path:
    return _frozen_scoring_path(
        protocol,
        key="formal_preflight_receipt_path",
        label="canonical stage-4 formal preflight receipt",
    )


def _load_frozen_cell_zone_mapping(
    protocol: Stage4ProtocolBundle,
    score_blind_inputs: ScoreBlindInputEvidence,
    preflight: FormalPreflightBundle,
) -> FrozenCellZoneMapping:
    """Authenticate the local score-blind mapping and bind its exact grid order."""

    path = _cell_mapping_path(protocol)
    expected_hashes = dict(score_blind_inputs.restricted_spatial_artifact_hashes)
    expected_sha256 = expected_hashes.get("cell_mapping")
    try:
        with open_existing_immutable_file(
            path,
            label="local construction-zone cell mapping",
        ) as handle:
            digest = hashlib.sha256()
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
            if expected_sha256 is None or digest.hexdigest() != expected_sha256:
                raise ValueError("local construction-zone cell mapping hash changed")
            handle.seek(0)
            table = pq.read_table(handle, columns=list(_CELL_MAPPING_COLUMNS))
    except UnsafeImmutableFileError as exc:
        raise ValueError("local construction-zone cell mapping is an unsafe alias") from exc
    if tuple(table.column_names) != _CELL_MAPPING_COLUMNS:
        raise ValueError("construction-zone cell mapping schema/order changed")
    primary = preflight.grid_family.primary_25km
    bridge = preflight.receipt.bridge
    if bridge.stage4_grid_id != primary.grid_id or bridge.cell_count != primary.cell_count:
        raise ValueError("formal grid-identity bridge differs from the rebuilt primary grid")
    if table.num_rows != primary.cell_count:
        raise ValueError("construction-zone cell mapping row count changed")

    grid_ids = tuple(table["grid_id"].combine_chunks().to_pylist())
    cell_ids = tuple(table["cell_id"].combine_chunks().to_pylist())
    zones = tuple(table["construction_zone_id"].combine_chunks().to_pylist())
    if (
        any(not isinstance(item, str) for item in (*grid_ids, *cell_ids, *zones))
        or set(cast(tuple[str, ...], grid_ids)) != {bridge.stage3_grid_id}
        or cast(tuple[str, ...], cell_ids) != primary.cell_ids
    ):
        raise ValueError("construction-zone cell mapping grid identity changed")
    rows = table["cell_row"].combine_chunks().to_numpy(zero_copy_only=False)
    columns = table["cell_column"].combine_chunks().to_numpy(zero_copy_only=False)
    query_x = table["query_x_m"].combine_chunks().to_numpy(zero_copy_only=False)
    query_y = table["query_y_m"].combine_chunks().to_numpy(zero_copy_only=False)
    if (
        not np.array_equal(rows, primary.rows)
        or not np.array_equal(columns, primary.columns)
        or not np.array_equal(query_x, primary.query_xy_m[:, 0])
        or not np.array_equal(query_y, primary.query_xy_m[:, 1])
    ):
        raise ValueError("construction-zone cell coordinates differ from the frozen grid")
    typed_zones = cast(tuple[str, ...], zones)
    all_zones = tuple(sorted(set(typed_zones)))
    if len(all_zones) != 39:
        raise ValueError("formal construction-zone mapping must contain 39 non-empty zones")
    return FrozenCellZoneMapping.bind(
        # The accepted local mapping carries the authenticated stage-3 role ID.
        # Its rows/columns/query coordinates were checked above and the typed
        # preflight bridge authorizes replacing only that ID in memory.
        grid_id=primary.grid_id,
        cell_ids=primary.cell_ids,
        construction_zone_ids=typed_zones,
        all_construction_zone_ids=all_zones,
        manifest_sha256=protocol.spatial_strata.content_sha256,
    )


def _verify_receipt_rebuild(
    rebuilt: FormalPreflightReceipt,
    stored: FormalPreflightReceipt,
) -> None:
    if rebuilt.content_sha256 != stored.content_sha256:
        raise ValueError("stored formal preflight differs from fresh deterministic inputs")
    if rebuilt.space_placebo_feature_identity != stored.space_placebo_feature_identity:
        raise ValueError("stored space-placebo feature identity differs from fresh inputs")
    resource = stored.space_placebo_resource_observation
    if (
        resource.feature_identity_sha256 != rebuilt.space_placebo_feature_identity.content_sha256
        or resource.output_identity_sha256
        != rebuilt.space_placebo_feature_identity.output_identity_sha256
    ):
        raise ValueError("stored space-placebo resource observation uses another feature build")
    fresh_resource = rebuilt.space_placebo_resource_observation
    if fresh_resource.recommended_max_in_flight < resource.recommended_max_in_flight:
        raise ValueError(
            "fresh target-blind memory observation cannot safely support the sealed "
            "space-placebo concurrency"
        )


def assemble_stage4_formal_preflight_artifacts(
    protocol: Stage4ProtocolBundle,
    score_blind_inputs: ScoreBlindInputEvidence,
    compute: Stage4ComputePlan,
    bundle: FormalPreflightBundle,
    stored_receipt: FormalPreflightReceipt,
    *,
    scoring_code_commit: str,
) -> FormalPreflightArtifacts:
    """Build the complete target-blind adapter consumed by ``FormalRunInputs``."""

    _verify_receipt_rebuild(bundle.receipt, stored_receipt)
    verified_protocol = VerifiedFormalProtocol.from_verified_protocol(protocol)
    scoring_plan = build_stage4_scoring_plan(protocol, compute=compute)
    primary = bundle.grid_family.primary_25km
    domain = BackgroundDomainBinding.from_verified_grid_family(
        bundle.grid_family,
        study_area_sha256=verified_protocol.development_snapshot.study_area_sha256,
        compensator_domain_id=(verified_protocol.development_snapshot.compensator_domain_id),
    )
    support_mask = np.ones(primary.cell_count, dtype=np.bool_)
    supports: list[FrozenCellSupport] = []
    for evaluation_id in _EVALUATION_IDS:
        snapshot = verified_protocol.snapshot_for(evaluation_id)
        if snapshot.supported_area_fraction != 1.0:
            raise ValueError(
                "production assembly requires the frozen full-area support declaration"
            )
        supports.append(
            FrozenCellSupport(
                evaluation_id=evaluation_id,
                grid_id=primary.grid_id,
                cell_ids=primary.cell_ids,
                support_id=snapshot.support_id,
                compensator_domain_id=snapshot.compensator_domain_id,
                supported_cell_mask=support_mask,
            )
        )
    feature_contract = feature_set_contract(protocol, "dynamic")
    if feature_contract.source_columns != bundle.feature_layout.dynamic_sources:
        raise ValueError("formal feature contract differs from the preflight feature layout")
    verified_issues = (
        *bundle.verified_issues_by_issue_id.values(),
        bundle.shadow_issue,
    )
    context = TargetBlindFormalContext(
        protocol=verified_protocol,
        scoring_plan=scoring_plan,
        study_area=bundle.study_area,
        grid_family=bundle.grid_family,
        background_domain=domain,
        verified_issues=tuple(verified_issues),
        feature_layout=bundle.feature_layout,
        cell_supports=tuple(supports),
        prospective_plans=bundle.shadow_plans,
        cell_zone_mapping=_load_frozen_cell_zone_mapping(
            protocol,
            score_blind_inputs,
            bundle,
        ),
    )
    placebo_inputs = FrozenPlaceboInputs(
        issue_tables=bundle.issue_tables,
        snapshots_by_issue_id=bundle.snapshots_by_issue_id,
        query_grid=primary.as_stage3_query_grid(),
        construction_stratum_by_state_id=(bundle.construction_stratum_by_state_id),
        source_input_sha256=stored_receipt.content_sha256,
    )
    formal_scope = scoring_plan.fit_scopes[3]
    formal_issue_ids = (
        *formal_scope.time_permutation_pools.fit_issue_ids,
        *formal_scope.time_permutation_pools.assessment_issue_ids,
    )
    convergence_inputs = build_target_blind_convergence_inputs(
        issue_ids=formal_issue_ids,
        snapshots=tuple(
            placebo_inputs.snapshots_by_issue_id[issue_id] for issue_id in formal_issue_ids
        ),
        grid_family=bundle.grid_family,
        accepted_primary_issue_tables={
            issue_id: placebo_inputs.issue_tables[issue_id] for issue_id in formal_issue_ids
        },
        source_columns=bundle.feature_layout.dynamic_sources,
        source_input_sha256=stored_receipt.content_sha256,
        query_chunk_size=placebo_inputs.query_chunk_size,
        spatial_workers=(2 if compute.workers.effective_workers >= 2 else 1),
    )
    return FormalPreflightArtifacts(
        context=context,
        receipt=stored_receipt,
        feature_contracts=feature_contract.contracts,
        placebo_inputs=placebo_inputs,
        convergence_inputs=convergence_inputs,
        model_version=f"stage4-anomaly-increment-{scoring_code_commit[:12]}",
    )


@dataclass(frozen=True, slots=True)
class FormalProductionReadiness:
    """Authenticated target-unread state immediately before authorization."""

    project_root: Path
    protocol: Stage4ProtocolBundle = field(repr=False, compare=False)
    scoring_seal: Stage4ScoringSeal = field(repr=False)
    scoring_seal_path: Path
    preflight_receipt_path: Path
    preflight: FormalPreflightArtifacts = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        root = Path(self.project_root).resolve()
        if root != self.protocol.repository_root.resolve():
            raise ValueError("production readiness protocol belongs to another project root")
        scoring_seal_path = require_score_blind_project_path(
            root,
            self.protocol.protocol,
            self.scoring_seal_path,
            label="production readiness scoring seal",
        )
        if not _same_path(scoring_seal_path, _scoring_seal_path(self.protocol)):
            raise ValueError("production readiness uses a non-canonical scoring seal")
        preflight_receipt_path = require_score_blind_project_path(
            root,
            self.protocol.protocol,
            self.preflight_receipt_path,
            label="production readiness formal preflight receipt",
        )
        if not _same_path(
            preflight_receipt_path,
            _formal_preflight_receipt_path(self.protocol),
        ):
            raise ValueError("production readiness uses a non-canonical preflight receipt")
        if self.scoring_seal.qualification.formal_preflight_receipt_sha256 != (
            self.preflight.receipt.content_sha256
        ):
            raise ValueError("production readiness seal and preflight receipt differ")
        object.__setattr__(self, "project_root", root)
        object.__setattr__(self, "scoring_seal_path", scoring_seal_path)
        object.__setattr__(
            self,
            "preflight_receipt_path",
            preflight_receipt_path,
        )

    def as_mapping(self) -> dict[str, object]:
        resource = self.preflight.receipt.space_placebo_resource_observation
        convergence = self.preflight.convergence_inputs
        publication = self.preflight.context.scoring_plan.publication
        return {
            "formal_backend": self.scoring_seal.qualification.formal_backend,
            "formal_preflight_receipt_sha256": self.preflight.receipt.content_sha256,
            "gpu_requested": self.scoring_seal.qualification.gpu_requested,
            "gpu_status": self.scoring_seal.qualification.gpu_status,
            "interactive_output": publication.local_spatial_interactive,
            "locked_test_run": False,
            "protocol_design_sha256": self.protocol.design_sha256,
            "scoring_code_commit": self.scoring_seal.qualification.scoring_code_commit,
            "space_placebo_recommended_max_in_flight": (resource.recommended_max_in_flight),
            "static_output": publication.local_spatial_static,
            "target_blind_convergence_grid_sizes_km": [
                item.cell_size_km for item in convergence.grids
            ],
            "target_blind_convergence_inputs_sha256": convergence.content_sha256,
            "target_blind_convergence_issue_count": len(convergence.primary_reproduction_receipts),
            "target_bytes_read": False,
            "target_path_observed": False,
        }


def load_stage4_formal_readiness(project_root: Path) -> FormalProductionReadiness:
    """Rebuild every score-blind input and authenticate the stored formal seal."""

    root = Path(project_root).resolve()
    protocol = load_stage4_protocol_bundle(root)
    seal_path = _scoring_seal_path(protocol)
    scoring_seal = load_stage4_scoring_seal(seal_path)
    score_blind_inputs = observe_score_blind_inputs(root, protocol.protocol)
    if score_blind_inputs != scoring_seal.score_blind_inputs:
        raise ValueError("fresh score-blind inputs differ from the scoring seal")
    compute = build_compute_plan(
        protocol,
        detected_physical_cores=detect_physical_core_count(),
        detected_logical_processors=os.cpu_count(),
    )
    rebuilt_bundle = load_formal_preflight(
        protocol,
        score_blind_inputs,
        compute,
        scoring_code_commit=scoring_seal.qualification.scoring_code_commit,
    )
    receipt_path = _formal_preflight_receipt_path(protocol)
    stored_receipt = load_formal_preflight_receipt(receipt_path)
    validate_stage4_qualification_against_protocol(
        protocol.protocol,
        scoring_seal.qualification,
    )
    validate_stage4_qualification_against_formal_preflight(
        scoring_seal.qualification,
        stored_receipt,
    )
    preflight = assemble_stage4_formal_preflight_artifacts(
        protocol,
        score_blind_inputs,
        compute,
        rebuilt_bundle,
        stored_receipt,
        scoring_code_commit=scoring_seal.qualification.scoring_code_commit,
    )
    return FormalProductionReadiness(
        project_root=root,
        protocol=protocol,
        scoring_seal=scoring_seal,
        scoring_seal_path=seal_path,
        preflight_receipt_path=receipt_path,
        preflight=preflight,
    )


def authorize_stage4_formal_readiness(
    readiness: FormalProductionReadiness,
) -> FormalRunInputs:
    """Grant the target capability without reading it and freeze official outputs."""

    if not isinstance(readiness, FormalProductionReadiness):
        raise TypeError("readiness must be FormalProductionReadiness")
    root = readiness.project_root
    authorization = authorize_stage4_target_access(
        root,
        readiness.protocol.protocol,
        scoring_seal_path=readiness.scoring_seal_path,
        attempt_ledger_path=_frozen_scoring_path(
            readiness.protocol,
            key="formal_attempt_ledger_path",
            label="canonical stage-4 R1 formal-attempt ledger",
        ),
        target_read_ledger_path=_frozen_scoring_path(
            readiness.protocol,
            key="target_read_ledger_path",
            label="canonical stage-4 R1 target-read ledger",
        ),
    )
    preflight = readiness.preflight
    publication = preflight.context.scoring_plan.publication
    return FormalRunInputs(
        project_root=root,
        authorization=authorization,
        preflight=preflight,
        checkpoint_directory=_frozen_scoring_path(
            readiness.protocol,
            key="checkpoint_root",
            label="canonical stage-4 R1 checkpoint root",
        ),
        concurrency=PlaceboConcurrencyPlan.from_preflight_receipt(
            preflight.context.scoring_plan.compute.workers,
            preflight.receipt,
        ),
        same_process_resume_limit=1,
        local_artifact_hook=Stage4SpatialArtifactHook(
            static_relative_path=publication.local_spatial_static,
            interactive_relative_path=publication.local_spatial_interactive,
        ),
    )


def verify_stage4_formal_readiness(
    readiness: FormalProductionReadiness,
) -> Stage4TargetReadinessEvidence:
    """Reprove run gates for ``check`` without constructing target authorization."""

    if not isinstance(readiness, FormalProductionReadiness):
        raise TypeError("readiness must be FormalProductionReadiness")
    root = readiness.project_root
    evidence = verify_stage4_target_readiness(
        root,
        readiness.protocol.protocol,
        scoring_seal_path=readiness.scoring_seal_path,
        attempt_ledger_path=_frozen_scoring_path(
            readiness.protocol,
            key="formal_attempt_ledger_path",
            label="canonical stage-4 R1 formal-attempt ledger",
        ),
        target_read_ledger_path=_frozen_scoring_path(
            readiness.protocol,
            key="target_read_ledger_path",
            label="canonical stage-4 R1 target-read ledger",
        ),
    )
    if evidence.scoring_seal.as_mapping() != readiness.scoring_seal.as_mapping():
        raise ValueError("read-only readiness proof uses another scoring seal")
    if evidence.preflight_receipt.content_sha256 != readiness.preflight.receipt.content_sha256:
        raise ValueError("read-only readiness proof uses another deterministic preflight")
    if evidence.preflight_receipt.space_placebo_resource_observation.content_sha256 != (
        readiness.preflight.receipt.space_placebo_resource_observation.content_sha256
    ):
        raise ValueError("read-only readiness proof uses another resource observation")
    return evidence


__all__: Sequence[str] = (
    "FORMAL_CHECKPOINT_RELATIVE_PATH",
    "FormalProductionReadiness",
    "assemble_stage4_formal_preflight_artifacts",
    "authorize_stage4_formal_readiness",
    "load_stage4_formal_readiness",
    "verify_stage4_formal_readiness",
)
