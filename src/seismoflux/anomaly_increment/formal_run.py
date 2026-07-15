"""Sole production orchestration boundary for the stage-4 formal run.

All four scientific scopes are reserved in one durable ledger write before the
only target ingress.  ``FormalRunSession`` then retains the authorized in-memory
catalogue and materialization for its complete lifetime.  A normal infrastructure
interruption may resume the same three checkpoint identities in the same process;
the target is never reopened.  A hard process crash deliberately leaves ``started``
attempt records.  Cross-process recovery is forbidden until an owner/lease protocol
is separately frozen, so an operator must never clear or counterfeit those records.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Final, Literal, Protocol, TypeAlias, cast

import pyarrow as pa
from shapely.geometry import mapping as geometry_mapping

from seismoflux.anomaly_increment.attempt_ledger import (
    STAGE4_ATTEMPT_SCOPES,
    complete_stage4_attempt_scopes,
    read_stage4_ledger,
    reserve_stage4_attempt_scopes,
)
from seismoflux.anomaly_increment.authorization import (
    Stage4TargetAuthorization,
    require_stage4_target_authorization,
)
from seismoflux.anomaly_increment.compute import Stage4WorkerPlan
from seismoflux.anomaly_increment.config import STAGE4_CHECKPOINT_ROOT_RELATIVE_PATH
from seismoflux.anomaly_increment.contracts import (
    FeatureColumnContract,
    canonical_mapping_sha256,
)
from seismoflux.anomaly_increment.convergence import (
    CompensatorConvergenceAudit,
    FrozenConvergenceModel,
    FrozenTargetBlindConvergenceInputs,
    FrozenVariantCoefficients,
    audit_frozen_compensator_convergence,
)
from seismoflux.anomaly_increment.formal_execution import (
    AuthorizedFormalMaterialization,
    TargetBlindFormalContext,
    build_formal_placebo_source_wiring,
    build_stage4_in_memory_plan,
    materialize_after_authorized_target,
)
from seismoflux.anomaly_increment.formal_preflight import FormalPreflightReceipt
from seismoflux.anomaly_increment.formal_publication import (
    AdditionalLocalArtifact,
    FormalPublicationReceipt,
    publish_failed_formal_result,
    publish_successful_formal_result,
)
from seismoflux.anomaly_increment.grid_features import (
    Stage4IntegrationGrid,
)
from seismoflux.anomaly_increment.immutable_file import (
    ImmutableFileSnapshot,
    UnsafeImmutableFileError,
    ensure_real_directory_tree,
    read_existing_immutable_file,
    require_existing_real_directory,
    unlink_existing_immutable_file,
)
from seismoflux.anomaly_increment.placebo import InfrastructureInterruption
from seismoflux.anomaly_increment.placebo_runtime import (
    PlaceboCheckpointError,
    PlaceboRuntime,
)
from seismoflux.anomaly_increment.placebo_source import (
    PlaceboSourceUniverse,
    build_placebo_source,
)
from seismoflux.anomaly_increment.preregistration import (
    verify_content_sha256,
    with_content_sha256,
)
from seismoflux.anomaly_increment.scoring_pipeline import (
    PipelineResult,
    PlaceboExecution,
    PlaceboRequest,
    Stage4InMemoryPlan,
    run_stage4_in_memory_pipeline,
)
from seismoflux.anomaly_increment.spatial_dashboard import (
    DisplayContextLayer,
    DisplayStudyArea,
)
from seismoflux.anomaly_increment.spatial_results import (
    GenuineProspectiveArchive,
    Stage4SpatialResults,
    build_stage4_spatial_results,
)
from seismoflux.anomaly_increment.target_access import (
    consume_authorized_stage4_target,
)
from seismoflux.anomaly_increment.targets import (
    Stage4TargetCatalog,
    parse_authorized_stage4_target_bytes,
)
from seismoflux.data.common import canonical_json_bytes
from seismoflux.features.anomaly.grid import Stage3QueryGrid
from seismoflux.features.anomaly.snapshot import Stage3IssueSnapshot

FORMAL_RUN_SESSION_SCHEMA_VERSION: Final[int] = 3
FORMAL_SESSION_SEAL_FILENAME: Final[str] = "formal_run_session_seal.json"
FORMAL_CONVERGENCE_AUDIT_FILENAME: Final[str] = "formal_convergence_audit.json"
FORMAL_CONVERGENCE_OUTPUT_PATH: Final[str] = (
    "outputs/visualizations/anomaly_increment_r1_convergence_audit.json"
)
FORMAL_TERMINALIZATION_INCIDENT_FILENAME: Final[str] = "formal_run_terminalization_incident.json"
_SESSION_SENTINEL = object()
_SHA256 = re.compile(r"[0-9a-f]{64}")
_FAILURE_CODE = re.compile(r"[a-z][a-z0-9_]{0,95}")

FormalRunStatus: TypeAlias = Literal["succeeded", "failed"]


class FormalRunError(RuntimeError):
    """Raised when a formal session cannot be safely prepared."""


class FormalRunPreparationError(FormalRunError):
    """A normal preparation failure that was published and terminally registered."""

    def __init__(self, outcome: FormalRunOutcome) -> None:
        super().__init__(outcome.failure_code or "formal_run_preparation_failed")
        self.outcome = outcome


class CompensatorConvergenceFailure(FormalRunError):
    """The frozen 25/12.5 km or 1/0.5 day compensator gate failed."""


def _formal_convergence_model(result: PipelineResult) -> FrozenConvergenceModel:
    """Freeze the already-fitted formal model without refitting any parameter."""

    formal = tuple(
        item for item in result.fitted_scopes if item.evaluation_id == "formal-validation"
    )
    if len(formal) != 1:
        raise ValueError("pipeline result must contain exactly one formal fitted scope")
    fitted = formal[0]
    return FrozenConvergenceModel(
        preprocessor=fitted.preprocessor,
        rate_head=fitted.rate_head,
        variants=tuple(
            FrozenVariantCoefficients(
                variant=item.variant,
                design_column_indices=item.design_column_indices,
                beta=item.beta,
            )
            for item in fitted.variants
        ),
    )


class FormalLocalArtifactHook(Protocol):
    """A frozen local-only builder receiving explicit post-target session state."""

    @property
    def content_sha256(self) -> str: ...

    def build(
        self,
        context: FormalLocalArtifactContext,
    ) -> Sequence[AdditionalLocalArtifact]: ...


@dataclass(frozen=True, slots=True)
class FormalLocalArtifactContext:
    """Typed state available only inside the one authorized in-memory session."""

    result: PipelineResult
    plan: Stage4InMemoryPlan
    materialization: AuthorizedFormalMaterialization
    catalog: Stage4TargetCatalog
    primary_grid: Stage4IntegrationGrid
    study_area: DisplayStudyArea

    def __post_init__(self) -> None:
        if not isinstance(self.result, PipelineResult):
            raise TypeError("local artifact context requires PipelineResult")
        if not isinstance(self.plan, Stage4InMemoryPlan):
            raise TypeError("local artifact context requires Stage4InMemoryPlan")
        if not isinstance(self.materialization, AuthorizedFormalMaterialization):
            raise TypeError("local artifact context requires authorized materialization")
        if not isinstance(self.catalog, Stage4TargetCatalog):
            raise TypeError("local artifact context requires the authorized in-memory catalog")
        if not isinstance(self.primary_grid, Stage4IntegrationGrid):
            raise TypeError("local artifact context requires Stage4IntegrationGrid")
        if not isinstance(self.study_area, DisplayStudyArea):
            raise TypeError(
                "local artifact context requires frozen target-independent DisplayStudyArea"
            )


@dataclass(frozen=True, slots=True)
class Stage4SpatialArtifactHook:
    """Immutable official adapter from the formal session to two local spatial views."""

    display_context_layers: tuple[DisplayContextLayer, ...] = ()
    genuine_prospective_archive: GenuineProspectiveArchive | None = None
    static_relative_path: str = "outputs/visualizations/anomaly_increment_r1_spatial.svg"
    interactive_relative_path: str = "outputs/visualizations/anomaly_increment_r1_spatial.html"

    def __post_init__(self) -> None:
        layers = tuple(self.display_context_layers)
        if any(not isinstance(item, DisplayContextLayer) for item in layers):
            raise TypeError("spatial display contexts must be DisplayContextLayer values")
        if self.genuine_prospective_archive is not None and not isinstance(
            self.genuine_prospective_archive,
            GenuineProspectiveArchive,
        ):
            raise TypeError("prospective archive must be GenuineProspectiveArchive")
        for path, suffix in (
            (self.static_relative_path, ".svg"),
            (self.interactive_relative_path, ".html"),
        ):
            if (
                not path.startswith("outputs/visualizations/")
                or "\\" in path
                or ".." in PurePosixPath(path).parts
                or not path.endswith(suffix)
            ):
                raise ValueError("spatial artifact path is not a safe local output")
        if self.static_relative_path == self.interactive_relative_path:
            raise ValueError("spatial static and interactive paths must differ")
        object.__setattr__(self, "display_context_layers", layers)

    @property
    def content_sha256(self) -> str:
        archive = self.genuine_prospective_archive
        return canonical_mapping_sha256(
            {
                "display_context_layers": [
                    {
                        "geometry_geojson": dict(item.geometry_geojson),
                        "label": item.label,
                        "layer_id": item.layer_id,
                        "layer_kind": item.layer_kind,
                        "source_content_sha256": item.source_content_sha256,
                    }
                    for item in self.display_context_layers
                ],
                "genuine_prospective_archive": (
                    None
                    if archive is None
                    else {
                        "archive_content_sha256": archive.archive_content_sha256,
                        "archive_id": archive.archive_id,
                        "archived_at_utc": archive.archived_at_utc.isoformat(),
                        "issue_dates": list(archive.issue_dates),
                        "model_version": archive.model_version,
                        "target_blind_frame_bundle_sha256": (
                            archive.target_blind_frame_bundle_sha256
                        ),
                    }
                ),
                "interactive_relative_path": self.interactive_relative_path,
                "role": "official_post_authorization_spatial_artifact_builder",
                "static_relative_path": self.static_relative_path,
            }
        )

    def build(
        self,
        context: FormalLocalArtifactContext,
    ) -> tuple[AdditionalLocalArtifact, AdditionalLocalArtifact]:
        if not isinstance(context, FormalLocalArtifactContext):
            raise TypeError("spatial artifact hook requires FormalLocalArtifactContext")
        spatial: Stage4SpatialResults = build_stage4_spatial_results(
            context.result,
            context.plan,
            context.materialization,
            context.primary_grid,
            context.catalog,
            study_area=context.study_area,
            display_context_layers=self.display_context_layers,
            genuine_prospective_archive=self.genuine_prospective_archive,
        )
        return (
            AdditionalLocalArtifact(
                relative_path=self.static_relative_path,
                bundle_filename="spatial-results.svg",
                payload=spatial.static_svg.encode("utf-8"),
            ),
            AdditionalLocalArtifact(
                relative_path=self.interactive_relative_path,
                bundle_filename="spatial-dashboard.html",
                payload=spatial.interactive_html.encode("utf-8"),
            ),
        )


def _sha256(value: str, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _normalized_failure_code(exc: Exception) -> str:
    if isinstance(exc, InfrastructureInterruption):
        return "infrastructure_interruption"
    if isinstance(exc, PlaceboCheckpointError):
        return "placebo_checkpoint_failure"
    name = re.sub(r"(?<!^)(?=[A-Z])", "_", type(exc).__name__).casefold()
    value = re.sub(r"[^a-z0-9_]", "_", name)[:96]
    return value if _FAILURE_CODE.fullmatch(value) is not None else "formal_execution_failure"


def _inside_project(root: Path, path: Path, *, label: str) -> Path:
    resolved_root = root.resolve()
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"{label} must stay inside the project root")
    return resolved


@dataclass(frozen=True, slots=True)
class PlaceboConcurrencyPlan:
    """Pretarget-frozen routing: fast time placebos and memory-bounded space placebos."""

    worker_plan: Stage4WorkerPlan
    time_max_in_flight: int
    space_max_in_flight: int
    space_memory_evidence_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.worker_plan, Stage4WorkerPlan):
            raise TypeError("placebo concurrency requires Stage4WorkerPlan")
        expected_time = min(12, self.worker_plan.effective_workers)
        if self.time_max_in_flight != expected_time:
            raise ValueError("time placebo concurrency must use the frozen available worker cap")
        default_space = min(2, self.worker_plan.effective_workers)
        if (
            not isinstance(self.space_max_in_flight, int)
            or isinstance(self.space_max_in_flight, bool)
            or not 1 <= self.space_max_in_flight <= self.worker_plan.effective_workers
        ):
            raise ValueError("space placebo concurrency must be within the worker count")
        if self.space_max_in_flight == default_space:
            if self.space_memory_evidence_sha256 is not None:
                _sha256(
                    self.space_memory_evidence_sha256,
                    label="space_memory_evidence_sha256",
                )
        elif self.space_memory_evidence_sha256 is None:
            raise ValueError("non-default space concurrency requires target-blind memory evidence")
        else:
            _sha256(
                self.space_memory_evidence_sha256,
                label="space_memory_evidence_sha256",
            )

    @classmethod
    def default(cls, worker_plan: Stage4WorkerPlan) -> PlaceboConcurrencyPlan:
        return cls(
            worker_plan=worker_plan,
            time_max_in_flight=min(12, worker_plan.effective_workers),
            space_max_in_flight=min(2, worker_plan.effective_workers),
        )

    @classmethod
    def from_preflight_receipt(
        cls,
        worker_plan: Stage4WorkerPlan,
        receipt: FormalPreflightReceipt,
    ) -> PlaceboConcurrencyPlan:
        if not isinstance(receipt, FormalPreflightReceipt):
            raise TypeError("receipt must be FormalPreflightReceipt")
        resource = receipt.space_placebo_resource_observation
        return cls(
            worker_plan=worker_plan,
            time_max_in_flight=min(12, worker_plan.effective_workers),
            space_max_in_flight=min(
                worker_plan.effective_workers,
                resource.recommended_max_in_flight,
            ),
            space_memory_evidence_sha256=resource.content_sha256,
        )

    def as_mapping(self) -> dict[str, object]:
        return {
            "space_max_in_flight": self.space_max_in_flight,
            "space_memory_evidence_sha256": self.space_memory_evidence_sha256,
            "time_max_in_flight": self.time_max_in_flight,
            "worker_plan": {
                "blas_threads_per_worker": self.worker_plan.blas_threads_per_worker,
                "configured_max_workers": self.worker_plan.configured_max_workers,
                "effective_workers": self.worker_plan.effective_workers,
                "logical_processors": self.worker_plan.logical_processors,
                "nested_parallelism": self.worker_plan.nested_parallelism,
                "physical_cores": self.worker_plan.physical_cores,
                "reserve_physical_cores": self.worker_plan.reserve_physical_cores,
            },
        }

    @property
    def content_sha256(self) -> str:
        return canonical_mapping_sha256(self.as_mapping())


@dataclass(frozen=True, slots=True)
class FrozenPlaceboInputs:
    """Complete target-blind raw universe needed by the three formal placebos."""

    issue_tables: Mapping[str, pa.Table]
    snapshots_by_issue_id: Mapping[str, Stage3IssueSnapshot]
    query_grid: Stage3QueryGrid
    construction_stratum_by_state_id: Mapping[str, str]
    source_input_sha256: str
    query_chunk_size: int = 256

    def __post_init__(self) -> None:
        tables = dict(self.issue_tables)
        snapshots = dict(self.snapshots_by_issue_id)
        strata = dict(self.construction_stratum_by_state_id)
        if not tables or set(tables) != set(snapshots):
            raise ValueError("placebo tables and snapshots must share one non-empty issue set")
        if any(not isinstance(value, pa.Table) or value.num_rows <= 0 for value in tables.values()):
            raise TypeError("placebo issue tables must be non-empty Arrow tables")
        if any(not isinstance(value, Stage3IssueSnapshot) for value in snapshots.values()):
            raise TypeError("placebo snapshots must be Stage3IssueSnapshot values")
        if not isinstance(self.query_grid, Stage3QueryGrid):
            raise TypeError("placebo query_grid must be Stage3QueryGrid")
        if any(
            not isinstance(key, str) or not key or not isinstance(value, str) or not value
            for key, value in strata.items()
        ):
            raise ValueError("placebo construction strata must be non-empty string pairs")
        _sha256(self.source_input_sha256, label="source_input_sha256")
        if (
            not isinstance(self.query_chunk_size, int)
            or isinstance(self.query_chunk_size, bool)
            or self.query_chunk_size <= 0
        ):
            raise ValueError("query_chunk_size must be a positive integer")
        object.__setattr__(self, "issue_tables", MappingProxyType(tables))
        object.__setattr__(self, "snapshots_by_issue_id", MappingProxyType(snapshots))
        object.__setattr__(
            self,
            "construction_stratum_by_state_id",
            MappingProxyType(strata),
        )


@dataclass(frozen=True, slots=True)
class FormalPreflightArtifacts:
    """Strict adapter surface to be constructed by the target-blind preflight module."""

    context: TargetBlindFormalContext
    receipt: FormalPreflightReceipt
    feature_contracts: tuple[FeatureColumnContract, ...]
    placebo_inputs: FrozenPlaceboInputs
    convergence_inputs: FrozenTargetBlindConvergenceInputs = field(
        repr=False,
        compare=False,
    )
    model_version: str

    def __post_init__(self) -> None:
        if not isinstance(self.context, TargetBlindFormalContext):
            raise TypeError("preflight context must be TargetBlindFormalContext")
        self.context.verify()
        if not isinstance(self.receipt, FormalPreflightReceipt):
            raise TypeError("preflight receipt must be FormalPreflightReceipt")
        if self.receipt.protocol_design_sha256 != (self.context.protocol.protocol_design_sha256):
            raise ValueError("preflight receipt belongs to another protocol")
        if self.receipt.random_input_seal_sha256 != (
            self.context.protocol.random_input_seal_sha256
        ):
            raise ValueError("preflight receipt uses another random input seal")
        contracts = tuple(self.feature_contracts)
        if not contracts or any(not isinstance(item, FeatureColumnContract) for item in contracts):
            raise TypeError("preflight feature contracts must be non-empty and typed")
        sources = tuple(item.source_column for item in contracts)
        if sources != self.context.feature_layout.dynamic_sources:
            raise ValueError("preflight contracts differ from the formal dynamic feature layout")
        if not isinstance(self.placebo_inputs, FrozenPlaceboInputs):
            raise TypeError("preflight placebo_inputs must be FrozenPlaceboInputs")
        convergence = self.convergence_inputs
        if not isinstance(convergence, FrozenTargetBlindConvergenceInputs):
            raise TypeError("preflight convergence inputs must be pretarget-frozen")
        if convergence.source_columns != sources:
            raise ValueError("preflight convergence sources differ from the formal contracts")
        if convergence.source_input_sha256 != self.receipt.content_sha256:
            raise ValueError("preflight convergence inputs belong to another receipt")
        if tuple(item.grid_id for item in convergence.grids) != tuple(
            item.grid_id for item in self.context.grid_family.grids()
        ):
            raise ValueError("preflight convergence inputs use another grid family")
        if not self.model_version or self.model_version != self.model_version.strip():
            raise ValueError("model_version must be a non-empty trimmed identifier")
        formal_scope = self.context.scoring_plan.fit_scopes[3]
        ordered_expected = (
            *formal_scope.time_permutation_pools.fit_issue_ids,
            *formal_scope.time_permutation_pools.assessment_issue_ids,
        )
        expected = set(ordered_expected)
        if set(self.placebo_inputs.issue_tables) != expected:
            raise ValueError("preflight placebo inputs differ from the frozen formal issue pools")
        if (
            tuple(item.issue_id for item in convergence.primary_reproduction_receipts)
            != ordered_expected
        ):
            raise ValueError("25 km convergence proof differs from the frozen formal issue pools")
        selected_issue_id = formal_scope.time_permutation_pools.assessment_issue_ids[-1]
        selected_snapshot = self.placebo_inputs.snapshots_by_issue_id[selected_issue_id]
        if (
            convergence.issue_id != selected_issue_id
            or convergence.selected_issue_index != selected_snapshot.issue_index
            or convergence.selected_state_snapshot_id != selected_snapshot.state_snapshot_id
            or convergence.selected_lineage_digest != selected_snapshot.lineage_digest
        ):
            raise ValueError("convergence inputs differ from the last formal assessment issue")
        object.__setattr__(self, "feature_contracts", contracts)


@dataclass(frozen=True, slots=True)
class FormalRunInputs:
    project_root: Path
    authorization: Stage4TargetAuthorization
    preflight: FormalPreflightArtifacts
    checkpoint_directory: Path
    concurrency: PlaceboConcurrencyPlan
    same_process_resume_limit: int = 1
    local_artifact_hook: FormalLocalArtifactHook | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        root = Path(self.project_root).resolve()
        if not root.is_dir():
            raise ValueError("project_root must be an existing directory")
        authorization = require_stage4_target_authorization(
            self.authorization,
            project_root=root,
        )
        if authorization.formal_backend != "cpu_float64":
            raise ValueError("formal stage-4 execution is locked to CPU float64")
        if not authorization.gpu_requested or authorization.gpu_status != (
            "blocked_no_frozen_backend"
        ):
            raise ValueError(
                "formal run must record the requested GPU as blocked by the frozen backend"
            )
        receipt = self.preflight.receipt
        resource = receipt.space_placebo_resource_observation
        if authorization.formal_preflight_receipt_sha256 != receipt.content_sha256:
            raise ValueError("authorization and deterministic preflight receipt differ")
        if authorization.space_placebo_resource_observation_sha256 != (resource.content_sha256):
            raise ValueError("authorization and space resource observation differ")
        if authorization.space_placebo_recommended_max_in_flight != (
            resource.recommended_max_in_flight
        ):
            raise ValueError("authorization and space concurrency recommendation differ")
        compute = self.preflight.context.scoring_plan.compute
        if compute.backend != "cpu_float64" or compute.gpu_equivalence_sha256 is not None:
            raise ValueError("formal scoring plan must remain locked to CPU float64")
        if self.concurrency.worker_plan != compute.workers:
            raise ValueError("placebo concurrency differs from the sealed compute worker plan")
        expected_space = min(
            self.concurrency.worker_plan.effective_workers,
            resource.recommended_max_in_flight,
        )
        if (
            self.concurrency.space_max_in_flight != expected_space
            or self.concurrency.space_memory_evidence_sha256 != resource.content_sha256
        ):
            raise ValueError(
                "space placebo concurrency is not bound to the preflight resource observation"
            )
        checkpoint = _inside_project(
            root,
            self.checkpoint_directory,
            label="checkpoint_directory",
        )
        allowed = root.joinpath(*STAGE4_CHECKPOINT_ROOT_RELATIVE_PATH.parts).resolve()
        if not checkpoint.is_relative_to(allowed):
            raise ValueError("formal checkpoints must stay under the local stage-4 interim root")
        if (
            not isinstance(self.same_process_resume_limit, int)
            or isinstance(self.same_process_resume_limit, bool)
            or not 0 <= self.same_process_resume_limit <= 10
        ):
            raise ValueError("same_process_resume_limit must be an integer in [0, 10]")
        hook = self.local_artifact_hook
        if hook is not None:
            _sha256(hook.content_sha256, label="local artifact hook content_sha256")
            if not callable(hook.build):
                raise TypeError("local artifact hook build must be callable")
        object.__setattr__(self, "project_root", root)
        object.__setattr__(self, "checkpoint_directory", checkpoint)


@dataclass(frozen=True, slots=True)
class FormalRunOutcome:
    status: FormalRunStatus
    publication: FormalPublicationReceipt | None
    result: PipelineResult | None
    failure_code: str | None
    same_process_resume_count: int
    session_seal_sha256: str
    consistency_incident_sha256: str | None = None
    convergence_audit: CompensatorConvergenceAudit | None = None

    def __post_init__(self) -> None:
        _sha256(self.session_seal_sha256, label="session_seal_sha256")
        if self.status == "succeeded":
            if (
                self.publication is None
                or self.result is None
                or self.failure_code is not None
                or self.consistency_incident_sha256 is not None
                or self.convergence_audit is None
            ):
                raise ValueError(
                    "successful formal outcome requires publication, result, and convergence"
                )
        elif self.result is not None or self.failure_code is None:
            raise ValueError("failed formal outcome requires only a failure code")
        if self.convergence_audit is not None and not isinstance(
            self.convergence_audit,
            CompensatorConvergenceAudit,
        ):
            raise TypeError("formal convergence_audit must be typed when present")
        if self.consistency_incident_sha256 is not None:
            _sha256(
                self.consistency_incident_sha256,
                label="consistency_incident_sha256",
            )
            if self.failure_code != "attempt_terminalization_failure":
                raise ValueError("consistency incident requires an attempt terminalization failure")
        if isinstance(self.same_process_resume_count, bool) or self.same_process_resume_count < 0:
            raise ValueError("same_process_resume_count must be non-negative")


def _operation_ids(authorization_id: str) -> dict[str, str]:
    suffix = authorization_id[:20]
    return {scope: f"stage4-{scope}-{suffix}" for scope in STAGE4_ATTEMPT_SCOPES}


def _session_seal_payload(inputs: FormalRunInputs) -> dict[str, object]:
    context = inputs.preflight.context
    publication = context.scoring_plan.publication
    checkpoint_relative = inputs.checkpoint_directory.relative_to(inputs.project_root).as_posix()
    hook = inputs.local_artifact_hook
    return with_content_sha256(
        {
            "authorization_id": inputs.authorization.authorization_id,
            "backend": "cpu_float64",
            "checkpoint_directory": checkpoint_relative,
            "checkpoint_files": [
                "time-dynamic-permutations.json",
                "space-dynamic-permutations.json",
                "time-snapshot-permutations.json",
            ],
            "consistency_incident_file": FORMAL_TERMINALIZATION_INCIDENT_FILENAME,
            "convergence": {
                "audit_file": FORMAL_CONVERGENCE_AUDIT_FILENAME,
                "output_path": publication.local_convergence_audit,
                "policy": "all_variant_bin_horizon_spatial_and_temporal_gate_before_publication",
                "target_blind_inputs_sha256": (inputs.preflight.convergence_inputs.content_sha256),
            },
            "concurrency": inputs.concurrency.as_mapping(),
            "execution_binding_id": inputs.authorization.execution_binding_id,
            "formal_preflight_receipt_sha256": (inputs.preflight.receipt.content_sha256),
            "feature_contracts": [item.as_mapping() for item in inputs.preflight.feature_contracts],
            "hard_crash_policy": "leave_started_and_forbid_cross_process_pseudo_recovery",
            "local_artifact_hook_sha256": (None if hook is None else hook.content_sha256),
            "model_version": inputs.preflight.model_version,
            "gpu": {
                "formal_backend": inputs.authorization.formal_backend,
                "requested": inputs.authorization.gpu_requested,
                "status": inputs.authorization.gpu_status,
                "fallback_reason": context.scoring_plan.compute.gpu_fallback_reason,
            },
            "placebo_source_input_sha256": (inputs.preflight.placebo_inputs.source_input_sha256),
            "protocol_design_sha256": context.protocol.protocol_design_sha256,
            "publication": {
                "bundle_root": publication.bundle_root,
                "local_convergence_audit": publication.local_convergence_audit,
                "local_interactive_html": publication.local_interactive_html,
                "local_spatial_interactive": publication.local_spatial_interactive,
                "local_spatial_static": publication.local_spatial_static,
                "public_model_card": publication.public_model_card,
                "public_registry": publication.public_registry,
                "public_report": publication.public_report,
                "public_static_svg": publication.public_static_svg,
            },
            "random_input_seal_sha256": context.protocol.random_input_seal_sha256,
            "same_process_resume_limit": inputs.same_process_resume_limit,
            "schema_version": FORMAL_RUN_SESSION_SCHEMA_VERSION,
            "scoring_plan_sha256": context.scoring_plan.content_sha256,
            "space_placebo_feature_identity": {
                "content_sha256": (
                    inputs.preflight.receipt.space_placebo_feature_identity.content_sha256
                ),
                "output_identity_sha256": (
                    inputs.preflight.receipt.space_placebo_feature_identity.output_identity_sha256
                ),
            },
            "space_placebo_resource_observation_sha256": (
                inputs.preflight.receipt.space_placebo_resource_observation.content_sha256
            ),
            "target_bytes_observed": False,
            "verified_stage3_issue_bindings": [
                item.binding_sha256 for item in context.verified_issues
            ],
        }
    )


def _directory_identity(value: os.stat_result) -> tuple[int, int]:
    return value.st_dev, value.st_ino


def _safe_unlink_entry(path: Path) -> None:
    """Remove a staging/checkpoint entry itself, never anything it references."""

    try:
        os.unlink(path)
    except FileNotFoundError:
        return


def _create_or_verify_formal_artifact(
    path: Path,
    payload: bytes,
    *,
    filesystem_anchor: Path,
    label: str,
) -> bytes:
    """Create one immutable artifact or verify an identical safe replay.

    The create operation is an atomic hard-link from a private sibling.  The
    sibling is removed before verification so the accepted artifact must be a
    regular, non-reparse, single-link file.  Existing links are rejected before
    their payload can be read.
    """

    destination = Path(path)
    try:
        ensure_real_directory_tree(
            filesystem_anchor,
            destination.parent,
            label=f"{label} parent directory",
        )
        parent_before = require_existing_real_directory(
            destination.parent,
            label=f"{label} parent directory",
        )
    except UnsafeImmutableFileError as exc:
        raise FormalRunError(f"unsafe {label} parent directory") from exc
    temporary = _write_temporary(destination, payload)
    try:
        staged_payload, staged_snapshot = read_existing_immutable_file(
            temporary,
            label=f"staged {label}",
        )
        if staged_payload != payload:
            raise FormalRunError(f"staged {label} changed before creation")
    except Exception:
        _safe_unlink_entry(temporary)
        raise
    created = False
    created_snapshot: ImmutableFileSnapshot | None = None
    try:
        try:
            os.link(temporary, destination)
            created = True
        except FileExistsError:
            pass
    except OSError as exc:
        raise FormalRunError(f"cannot create {label}") from exc
    finally:
        _safe_unlink_entry(temporary)
    try:
        observed, observed_snapshot = read_existing_immutable_file(
            destination,
            label=label,
        )
        if created:
            if (observed_snapshot.device, observed_snapshot.inode) != (
                staged_snapshot.device,
                staged_snapshot.inode,
            ):
                raise UnsafeImmutableFileError(f"{label} changed before post-creation verification")
            created_snapshot = observed_snapshot
        parent_after = require_existing_real_directory(
            destination.parent,
            label=f"{label} parent directory",
        )
        if _directory_identity(parent_after) != _directory_identity(parent_before):
            raise FormalRunError(f"{label} parent directory changed during creation")
        if observed != payload:
            raise FormalRunError(f"existing {label} contains different bytes")
    except (OSError, UnsafeImmutableFileError, FormalRunError):
        if created_snapshot is not None:
            try:
                unlink_existing_immutable_file(
                    destination,
                    expected=created_snapshot,
                    label=label,
                )
            except (OSError, UnsafeImmutableFileError) as rollback_exc:
                raise FormalRunError(
                    f"{label} changed before identity-bound rollback"
                ) from rollback_exc
        raise
    return observed


def _write_or_verify_session_seal(inputs: FormalRunInputs) -> str:
    path = inputs.checkpoint_directory / FORMAL_SESSION_SEAL_FILENAME
    payload = _session_seal_payload(inputs)
    serialized = canonical_json_bytes(payload) + b"\n"
    try:
        observed_bytes = _create_or_verify_formal_artifact(
            path,
            serialized,
            filesystem_anchor=inputs.project_root,
            label="formal session seal",
        )
        observed = json.loads(observed_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise FormalRunError("existing formal session seal is unreadable") from exc
    if not isinstance(observed, dict) or not verify_content_sha256(observed) or observed != payload:
        raise FormalRunError("existing formal session seal changed or uses another context")
    return cast(str, payload["content_sha256"])


def _write_or_verify_convergence_audit(
    inputs: FormalRunInputs,
    audit: CompensatorConvergenceAudit,
    *,
    expected_model_sha256: str,
) -> bytes:
    """Persist the complete hard-gate evidence before any result publication."""

    if not isinstance(audit, CompensatorConvergenceAudit):
        raise TypeError("formal convergence hook returned an untyped audit")
    if audit.model_sha256 != expected_model_sha256:
        raise ValueError("formal convergence audit belongs to another fitted model")
    payload = with_content_sha256(
        {
            "audit": audit.as_mapping(),
            "execution_binding_id": inputs.authorization.execution_binding_id,
            "formal_preflight_receipt_sha256": inputs.preflight.receipt.content_sha256,
            "policy": "hard_stop_and_forbid_spatial_publication_on_any_gate_failure",
            "schema_version": 1,
            "status": audit.status,
            "target_blind_inputs_sha256": (inputs.preflight.convergence_inputs.content_sha256),
        }
    )
    serialized = canonical_json_bytes(payload) + b"\n"
    path = inputs.checkpoint_directory / FORMAL_CONVERGENCE_AUDIT_FILENAME
    _create_or_verify_formal_artifact(
        path,
        serialized,
        filesystem_anchor=inputs.project_root,
        label="formal convergence audit",
    )
    return serialized


@dataclass(frozen=True, slots=True)
class _FormalPlaceboRouter:
    time_runtime: PlaceboRuntime
    space_runtime: PlaceboRuntime

    def __call__(self, request: PlaceboRequest) -> PlaceboExecution:
        allowed = {
            ("time", "dynamic"),
            ("space", "dynamic"),
            ("time", "snapshot"),
        }
        if (request.kind, request.model_variant) not in allowed:
            raise ValueError("pipeline requested a placebo outside the frozen three requests")
        return self.time_runtime(request) if request.kind == "time" else self.space_runtime(request)


def _build_scientific_components(
    inputs: FormalRunInputs,
    materialization: AuthorizedFormalMaterialization,
) -> tuple[Stage4InMemoryPlan, _FormalPlaceboRouter]:
    preflight = inputs.preflight
    context = preflight.context
    plan = build_stage4_in_memory_plan(
        materialization,
        feature_contracts=preflight.feature_contracts,
        feature_layout=context.feature_layout,
        frozen_input_seal_sha256=context.protocol.random_input_seal_sha256,
        model_version=preflight.model_version,
    )
    wiring = build_formal_placebo_source_wiring(materialization)
    raw = preflight.placebo_inputs
    universe = PlaceboSourceUniverse(
        fit_issue_ids=wiring.fit_issue_ids,
        assessment_issue_ids=wiring.assessment_issue_ids,
        issue_tables=raw.issue_tables,
        snapshots_by_issue_id=raw.snapshots_by_issue_id,
        query_grid=raw.query_grid,
        construction_stratum_by_state_id=raw.construction_stratum_by_state_id,
        frozen_input_seal_sha256=context.protocol.random_input_seal_sha256,
        source_input_sha256=raw.source_input_sha256,
        assembly_context_sha256=wiring.assembly_context_sha256,
        scope_assembler=wiring.scope_assembler,
        query_chunk_size=raw.query_chunk_size,
    )
    source_factory = partial(build_placebo_source, universe)
    time_runtime = PlaceboRuntime(
        checkpoint_directory=inputs.checkpoint_directory,
        execution_binding_id=inputs.authorization.execution_binding_id,
        source_factory=source_factory,
        worker_plan=inputs.concurrency.worker_plan,
        max_in_flight=inputs.concurrency.time_max_in_flight,
        backend="cpu_float64",
    )
    space_runtime = PlaceboRuntime(
        checkpoint_directory=inputs.checkpoint_directory,
        execution_binding_id=inputs.authorization.execution_binding_id,
        source_factory=source_factory,
        worker_plan=inputs.concurrency.worker_plan,
        max_in_flight=inputs.concurrency.space_max_in_flight,
        backend="cpu_float64",
    )
    return plan, _FormalPlaceboRouter(time_runtime, space_runtime)


def _terminalize_attempts(
    inputs: FormalRunInputs,
    *,
    status: FormalRunStatus,
    result_sha256: str | None,
    failure_code: str | None,
) -> None:
    complete_stage4_attempt_scopes(
        inputs.authorization.attempt_ledger_path,
        execution_binding_id=inputs.authorization.execution_binding_id,
        operation_ids_by_scope=_operation_ids(inputs.authorization.authorization_id),
        authorization_id=inputs.authorization.authorization_id,
        status=status,
        result_sha256=result_sha256,
        failure_code=failure_code,
    )


def _confirmed_terminal_ledger(
    inputs: FormalRunInputs,
    *,
    status: FormalRunStatus,
    result_sha256: str | None,
    failure_code: str | None,
) -> object | None:
    """Read back only; never retry a terminal mutation after an ambiguous exception."""

    try:
        ledger = read_stage4_ledger(
            inputs.authorization.attempt_ledger_path,
            expected_kind="formal_attempt",
            expected_binding_id=inputs.authorization.execution_binding_id,
        )
    except Exception:
        return None
    operations = _operation_ids(inputs.authorization.authorization_id)
    if tuple(item.scope for item in ledger.records) != STAGE4_ATTEMPT_SCOPES:
        return None
    if all(
        item.operation_id == operations[item.scope]
        and item.authorization_id == inputs.authorization.authorization_id
        and item.status == status
        and item.result_sha256 == result_sha256
        and item.failure_code == failure_code
        for item in ledger.records
    ):
        return ledger
    return None


def _write_temporary(destination: Path, payload: bytes) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(raw_path)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        _safe_unlink_entry(temporary)
        raise
    return temporary


def _write_terminalization_incident(
    inputs: FormalRunInputs,
    *,
    publication: FormalPublicationReceipt | None,
    session_seal_sha256: str,
    expected_status: FormalRunStatus,
    expected_result_sha256: str | None,
    expected_failure_code: str | None,
    cause: Exception,
) -> str:
    """Create a deterministic audit explanation without mutating the attempt ledger."""

    try:
        observed = read_stage4_ledger(
            inputs.authorization.attempt_ledger_path,
            expected_kind="formal_attempt",
            expected_binding_id=inputs.authorization.execution_binding_id,
        )
    except Exception:
        observed_mapping: dict[str, object] = {
            "content_sha256": None,
            "status_by_scope": None,
        }
    else:
        observed_mapping = {
            "content_sha256": observed.content_sha256,
            "status_by_scope": {item.scope: item.status for item in observed.records},
        }
    payload = with_content_sha256(
        {
            "authorization_id": inputs.authorization.authorization_id,
            "cause_code": _normalized_failure_code(cause),
            "cross_process_recovery_forbidden": True,
            "execution_binding_id": inputs.authorization.execution_binding_id,
            "expected_attempt_state": {
                "result_sha256": expected_result_sha256,
                "failure_code": expected_failure_code,
                "operation_ids_by_scope": _operation_ids(inputs.authorization.authorization_id),
                "status": expected_status,
            },
            "observed_attempt_ledger": observed_mapping,
            "publication": (
                None
                if publication is None
                else {
                    "bundle_id": publication.bundle_id,
                    "manifest_sha256": publication.manifest_sha256,
                    "registry_sha256": publication.registry_sha256,
                    "status": publication.status,
                }
            ),
            "resolution_policy": "manual_audit_only_no_reexecution_or_pseudo_recovery",
            "schema_version": 1,
            "session_seal_sha256": session_seal_sha256,
            "status": "formal_attempt_terminalization_unconfirmed",
        }
    )
    serialized = canonical_json_bytes(payload) + b"\n"
    path = inputs.checkpoint_directory / FORMAL_TERMINALIZATION_INCIDENT_FILENAME
    _create_or_verify_formal_artifact(
        path,
        serialized,
        filesystem_anchor=inputs.project_root,
        label="formal terminalization incident",
    )
    return cast(str, payload["content_sha256"])


def _publish_failure_and_terminalize(
    inputs: FormalRunInputs,
    *,
    failure_code: str,
    session_seal_sha256: str,
    resume_count: int,
) -> FormalRunOutcome:
    publication: FormalPublicationReceipt | None = None
    terminal_code = failure_code
    try:
        publication = publish_failed_formal_result(
            inputs.project_root,
            inputs.preflight.context.scoring_plan.publication,
            execution_binding_id=inputs.authorization.execution_binding_id,
            authorization_id=inputs.authorization.authorization_id,
            model_version=inputs.preflight.model_version,
            failure_code=failure_code,
        )
    except Exception:
        terminal_code = "formal_publication_failure"
    incident_sha256: str | None = None
    try:
        _terminalize_attempts(
            inputs,
            status="failed",
            result_sha256=None,
            failure_code=terminal_code,
        )
    except Exception as exc:
        confirmed = _confirmed_terminal_ledger(
            inputs,
            status="failed",
            result_sha256=None,
            failure_code=terminal_code,
        )
        if confirmed is None:
            try:
                incident_sha256 = _write_terminalization_incident(
                    inputs,
                    publication=publication,
                    session_seal_sha256=session_seal_sha256,
                    expected_status="failed",
                    expected_result_sha256=None,
                    expected_failure_code=terminal_code,
                    cause=exc,
                )
            except Exception as incident_exc:
                raise FormalRunError(
                    "failed formal run could not be reconciled with the attempt ledger"
                ) from incident_exc
            terminal_code = "attempt_terminalization_failure"
    return FormalRunOutcome(
        status="failed",
        publication=publication,
        result=None,
        failure_code=terminal_code,
        same_process_resume_count=resume_count,
        session_seal_sha256=session_seal_sha256,
        consistency_incident_sha256=incident_sha256,
    )


@dataclass(slots=True)
class FormalRunSession:
    """One authorized in-memory session; it has no method that can reopen the target."""

    inputs: FormalRunInputs
    catalog: Stage4TargetCatalog = field(repr=False)
    materialization: AuthorizedFormalMaterialization = field(repr=False)
    plan: Stage4InMemoryPlan = field(repr=False)
    placebo_router: _FormalPlaceboRouter = field(repr=False)
    target_sha256: str
    session_seal_sha256: str
    _sentinel: object = field(repr=False)
    _terminal: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self._sentinel is not _SESSION_SENTINEL:
            raise ValueError("FormalRunSession must be created by prepare_formal_run_session")
        if not isinstance(self.catalog, Stage4TargetCatalog):
            raise TypeError("formal session requires Stage4TargetCatalog")
        if not isinstance(self.materialization, AuthorizedFormalMaterialization):
            raise TypeError("formal session requires AuthorizedFormalMaterialization")
        if not isinstance(self.plan, Stage4InMemoryPlan):
            raise TypeError("formal session requires Stage4InMemoryPlan")
        _sha256(self.target_sha256, label="target_sha256")
        _sha256(self.session_seal_sha256, label="session_seal_sha256")

    def execute(self) -> FormalRunOutcome:
        """Run once, resuming only the same in-process session and checkpoint identities."""

        if self._terminal:
            raise FormalRunError("formal session is already terminal")
        resume_count = 0
        while True:
            try:
                result = run_stage4_in_memory_pipeline(
                    self.plan,
                    placebo_injection=self.placebo_router,
                )
                convergence_model = _formal_convergence_model(result)
                convergence_audit = audit_frozen_compensator_convergence(
                    model=convergence_model,
                    inputs=self.inputs.preflight.convergence_inputs,
                    formal_background=self.materialization.background_fits[-1],
                )
                convergence_payload = _write_or_verify_convergence_audit(
                    self.inputs,
                    convergence_audit,
                    expected_model_sha256=convergence_model.content_sha256,
                )
                if not convergence_audit.passed:
                    raise CompensatorConvergenceFailure(
                        "formal compensator convergence failed; spatial publication forbidden"
                    )
                convergence_artifact = AdditionalLocalArtifact(
                    relative_path=(
                        self.inputs.preflight.context.scoring_plan.publication.local_convergence_audit
                    ),
                    bundle_filename="convergence-audit.json",
                    payload=convergence_payload,
                )
                hook = self.inputs.local_artifact_hook
                extensions: tuple[AdditionalLocalArtifact, ...]
                if hook is None:
                    extensions = (convergence_artifact,)
                else:
                    snapshot = self.inputs.preflight.context.protocol.development_snapshot
                    display_study_area = DisplayStudyArea(
                        study_area_id=(
                            "stage4-frozen-study-area-" + snapshot.study_area_sha256[:12]
                        ),
                        geometry_geojson=cast(
                            Mapping[str, object],
                            geometry_mapping(self.inputs.preflight.context.study_area.geographic),
                        ),
                        source_content_sha256=snapshot.study_area_sha256,
                    )
                    extensions = (
                        convergence_artifact,
                        *tuple(
                            hook.build(
                                FormalLocalArtifactContext(
                                    result=result,
                                    plan=self.plan,
                                    materialization=self.materialization,
                                    catalog=self.catalog,
                                    primary_grid=(
                                        self.inputs.preflight.context.grid_family.primary_25km
                                    ),
                                    study_area=display_study_area,
                                )
                            )
                        ),
                    )
                receipt = publish_successful_formal_result(
                    self.inputs.project_root,
                    self.inputs.preflight.context.scoring_plan.publication,
                    execution_binding_id=self.inputs.authorization.execution_binding_id,
                    authorization_id=self.inputs.authorization.authorization_id,
                    result=result,
                    convergence_audit=convergence_audit,
                    additional_local_artifacts=extensions,
                )
            except (InfrastructureInterruption, PlaceboCheckpointError, OSError) as exc:
                if resume_count < self.inputs.same_process_resume_limit:
                    resume_count += 1
                    continue
                outcome = _publish_failure_and_terminalize(
                    self.inputs,
                    failure_code=_normalized_failure_code(exc),
                    session_seal_sha256=self.session_seal_sha256,
                    resume_count=resume_count,
                )
                self._terminal = True
                return outcome
            except Exception as exc:
                outcome = _publish_failure_and_terminalize(
                    self.inputs,
                    failure_code=_normalized_failure_code(exc),
                    session_seal_sha256=self.session_seal_sha256,
                    resume_count=resume_count,
                )
                self._terminal = True
                return outcome
            try:
                _terminalize_attempts(
                    self.inputs,
                    status="succeeded",
                    result_sha256=receipt.bundle_id,
                    failure_code=None,
                )
            except Exception as exc:
                # A durable ledger replace may have completed immediately before
                # its writer surfaced an I/O exception.  A read-only verification
                # can confirm that success; it must never retry or rewrite the
                # one-shot attempt after the publication boundary.
                confirmed = _confirmed_terminal_ledger(
                    self.inputs,
                    status="succeeded",
                    result_sha256=receipt.bundle_id,
                    failure_code=None,
                )
                if confirmed is None:
                    self._terminal = True
                    try:
                        incident_sha256 = _write_terminalization_incident(
                            self.inputs,
                            publication=receipt,
                            session_seal_sha256=self.session_seal_sha256,
                            expected_status="succeeded",
                            expected_result_sha256=receipt.bundle_id,
                            expected_failure_code=None,
                            cause=exc,
                        )
                    except Exception as incident_exc:
                        raise FormalRunError(
                            "published result could not be reconciled with the attempt ledger"
                        ) from incident_exc
                    return FormalRunOutcome(
                        status="failed",
                        publication=receipt,
                        result=None,
                        failure_code="attempt_terminalization_failure",
                        same_process_resume_count=resume_count,
                        session_seal_sha256=self.session_seal_sha256,
                        consistency_incident_sha256=incident_sha256,
                    )
            self._terminal = True
            return FormalRunOutcome(
                status="succeeded",
                publication=receipt,
                result=result,
                failure_code=None,
                same_process_resume_count=resume_count,
                session_seal_sha256=self.session_seal_sha256,
                convergence_audit=convergence_audit,
            )


def prepare_formal_run_session(inputs: FormalRunInputs) -> FormalRunSession:
    """Reserve every scope, seal runtime choices, and consume the target exactly once."""

    if not isinstance(inputs, FormalRunInputs):
        raise TypeError("formal run preparation requires FormalRunInputs")
    operations = _operation_ids(inputs.authorization.authorization_id)
    reserve_stage4_attempt_scopes(
        inputs.authorization.attempt_ledger_path,
        execution_binding_id=inputs.authorization.execution_binding_id,
        operation_ids_by_scope=operations,
        authorization_id=inputs.authorization.authorization_id,
    )
    session_seal_sha256 = cast(str, _session_seal_payload(inputs)["content_sha256"])
    try:
        session_seal_sha256 = _write_or_verify_session_seal(inputs)
        identity = inputs.authorization.expected_target_mapping()

        def consume(payload: bytes) -> tuple[Stage4TargetCatalog, AuthorizedFormalMaterialization]:
            catalog = parse_authorized_stage4_target_bytes(
                payload,
                expected_content_sha256=identity["content_sha256"],
                expected_schema_sha256=identity["schema_sha256"],
                study_area=inputs.preflight.context.study_area,
            )
            return catalog, materialize_after_authorized_target(
                inputs.preflight.context,
                catalog,
            )

        consumed = consume_authorized_stage4_target(
            inputs.project_root,
            inputs.authorization,
            operation_id=f"stage4-target-{inputs.authorization.authorization_id[:20]}",
            consumer=consume,
        )
        catalog, materialization = consumed.value
        plan, router = _build_scientific_components(inputs, materialization)
        return FormalRunSession(
            inputs=inputs,
            catalog=catalog,
            materialization=materialization,
            plan=plan,
            placebo_router=router,
            target_sha256=consumed.target_sha256,
            session_seal_sha256=session_seal_sha256,
            _sentinel=_SESSION_SENTINEL,
        )
    except Exception as exc:
        outcome = _publish_failure_and_terminalize(
            inputs,
            failure_code=_normalized_failure_code(exc),
            session_seal_sha256=session_seal_sha256,
            resume_count=0,
        )
        raise FormalRunPreparationError(outcome) from exc


def run_formal_stage4(inputs: FormalRunInputs) -> FormalRunOutcome:
    """Run the sole stage-4 production chain and always return normal terminal failures."""

    try:
        session = prepare_formal_run_session(inputs)
    except FormalRunPreparationError as exc:
        return exc.outcome
    return session.execute()


__all__ = [
    "FORMAL_RUN_SESSION_SCHEMA_VERSION",
    "FORMAL_SESSION_SEAL_FILENAME",
    "FORMAL_TERMINALIZATION_INCIDENT_FILENAME",
    "FormalLocalArtifactContext",
    "FormalLocalArtifactHook",
    "FormalPreflightArtifacts",
    "FormalRunError",
    "FormalRunInputs",
    "FormalRunOutcome",
    "FormalRunPreparationError",
    "FormalRunSession",
    "FormalRunStatus",
    "FrozenPlaceboInputs",
    "PlaceboConcurrencyPlan",
    "Stage4SpatialArtifactHook",
    "prepare_formal_run_session",
    "run_formal_stage4",
]
