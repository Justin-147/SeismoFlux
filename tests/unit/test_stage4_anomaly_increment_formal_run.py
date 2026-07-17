from __future__ import annotations

import hashlib
import importlib
import json
import os
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, TypeAlias, cast

import numpy as np
import pytest
from stage4_formal_preflight_fixture import make_formal_preflight_receipt
from test_stage4_anomaly_increment_pipeline import (
    _plan,
    run_stage4_synthetic_test_pipeline,
)

import seismoflux.anomaly_increment.attempt_ledger as attempt_ledger_module
import seismoflux.anomaly_increment.authorization as authorization_module
import seismoflux.anomaly_increment.formal_publication as formal_publication_module
import seismoflux.anomaly_increment.formal_run as formal_run
import seismoflux.anomaly_increment.immutable_file as immutable_file_module
import seismoflux.anomaly_increment.target_access as target_access_module
from seismoflux.anomaly_increment.attempt_ledger import (
    _complete_stage4_attempt_scopes_generic as _complete_stage4_attempt_scopes,
)
from seismoflux.anomaly_increment.attempt_ledger import (
    _initialize_stage4_ledger_generic as initialize_stage4_ledger,
)
from seismoflux.anomaly_increment.attempt_ledger import (
    _read_stage4_ledger_generic as read_stage4_ledger,
)
from seismoflux.anomaly_increment.attempt_ledger import (
    _recover_interrupted_stage4_operations_generic as recover_interrupted_stage4_operations,
)
from seismoflux.anomaly_increment.authorization import (
    STAGE4_ATTEMPT_LEDGER_RELATIVE_PATH,
    STAGE4_TARGET_READ_LEDGER_RELATIVE_PATH,
    Stage4TargetAuthorization,
)
from seismoflux.anomaly_increment.compute import Stage4ComputePlan, Stage4WorkerPlan
from seismoflux.anomaly_increment.convergence import (
    CompensatorConvergenceAudit,
    FrozenConvergenceModel,
    FrozenTargetBlindConvergenceInputs,
    PrimaryGridReproductionReceipt,
    RecomputedGridInputs,
    TargetBlindGridFeatures,
    audit_compensator_convergence,
)
from seismoflux.anomaly_increment.formal_execution import (
    AuthorizedFormalMaterialization,
    _materialize_in_memory_core,
    build_stage4_in_memory_plan,
)
from seismoflux.anomaly_increment.formal_run import (
    FORMAL_CONVERGENCE_AUDIT_FILENAME,
    FORMAL_CONVERGENCE_OUTPUT_PATH,
    FORMAL_SESSION_SEAL_FILENAME,
    FORMAL_TERMINALIZATION_INCIDENT_FILENAME,
    FormalPreflightArtifacts,
    FormalRunInputs,
    FrozenPlaceboInputs,
    PlaceboConcurrencyPlan,
    Stage4SpatialArtifactHook,
    prepare_formal_run_session,
    run_formal_stage4,
)
from seismoflux.anomaly_increment.placebo import InfrastructureInterruption
from seismoflux.anomaly_increment.placebo_runtime import PlaceboRuntime
from seismoflux.anomaly_increment.runner import Stage4PublicationPlan
from seismoflux.anomaly_increment.scoring_pipeline import (
    PipelineResult,
    PlaceboExecution,
    PlaceboRequest,
    Stage4InMemoryPlan,
)
from seismoflux.anomaly_increment.spatial_dashboard import DisplayStudyArea
from seismoflux.features.anomaly.snapshot import Stage3IssueSnapshot
from seismoflux.features.anomaly.state import AnomalyState

_PathArgument: TypeAlias = str | bytes | os.PathLike[str] | os.PathLike[bytes]
_FORMAL_EXECUTION_TEST_MODULE: Any = importlib.import_module(
    "test_stage4_anomaly_increment_formal_execution"
)
_context_and_catalog = _FORMAL_EXECUTION_TEST_MODULE._context_and_catalog
_contracts = _FORMAL_EXECUTION_TEST_MODULE._contracts

_WINDOWS_REPARSE_POINT = 0x0400


@pytest.fixture(autouse=True)
def _allow_future_ledger_creation_for_structural_tests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def allow_future_execution(*args: object, **kwargs: object) -> None:
        return None

    monkeypatch.setattr(
        attempt_ledger_module,
        "require_stage4_r2_execution_action",
        allow_future_execution,
    )
    monkeypatch.setattr(
        formal_run,
        "require_stage4_r2_execution_action",
        allow_future_execution,
    )
    monkeypatch.setattr(
        formal_publication_module,
        "require_stage4_r2_execution_action",
        allow_future_execution,
    )
    monkeypatch.setattr(
        formal_run,
        "protocol_design_sha256",
        lambda protocol: cast(str, protocol["_synthetic_protocol_design_sha256"]),
    )
    monkeypatch.setattr(
        target_access_module,
        "require_stage4_r2_execution_action",
        allow_future_execution,
    )


def _install_lstat_fallback(
    monkeypatch: pytest.MonkeyPatch,
    path: Path,
    *,
    reparse: bool,
    link_count: int | None = None,
) -> None:
    """Simulate an unsafe final entry when Windows cannot create real links."""

    original_lstat = os.lstat
    expected = os.path.normcase(os.path.abspath(os.fspath(path)))

    def simulated_lstat(
        candidate: _PathArgument,
        *,
        dir_fd: int | None = None,
    ) -> os.stat_result:
        observed = (
            original_lstat(candidate)
            if dir_fd is None
            else original_lstat(candidate, dir_fd=dir_fd)
        )
        actual = os.path.normcase(os.path.abspath(os.fsdecode(candidate)))
        if actual != expected:
            return observed
        attributes = int(getattr(observed, "st_file_attributes", 0))
        attributes = (
            attributes | _WINDOWS_REPARSE_POINT if reparse else attributes & ~_WINDOWS_REPARSE_POINT
        )
        return cast(
            os.stat_result,
            SimpleNamespace(
                st_mode=observed.st_mode,
                st_file_attributes=attributes,
                st_dev=observed.st_dev,
                st_ino=observed.st_ino,
                st_nlink=observed.st_nlink if link_count is None else link_count,
                st_size=observed.st_size,
                st_mtime_ns=observed.st_mtime_ns,
                st_ctime_ns=observed.st_ctime_ns,
            ),
        )

    monkeypatch.setattr(os, "lstat", simulated_lstat)


def _publication() -> Stage4PublicationPlan:
    return Stage4PublicationPlan(
        public_registry="data/manifests/anomaly_increment_r2_model_registry.json",
        public_report="docs/anomaly_increment_r2_report.md",
        public_static_svg="docs/anomaly_increment_r2_results.svg",
        local_interactive_html="outputs/visualizations/anomaly_increment_r2_dashboard.html",
        public_model_card="docs/model_cards/anomaly_increment_r2.md",
        bundle_root="models/registry/anomaly_increment_r2",
        local_convergence_audit=FORMAL_CONVERGENCE_OUTPUT_PATH,
        local_spatial_static=("outputs/visualizations/anomaly_increment_r2_forecast_spatial.svg"),
        local_spatial_interactive=(
            "outputs/visualizations/anomaly_increment_r2_forecast_spatial.html"
        ),
    )


def _synthetic_convergence_inputs(
    context: Any,
    placebo_inputs: FrozenPlaceboInputs,
    *,
    source_input_sha256: str,
) -> FrozenTargetBlindConvergenceInputs:
    formal_scope = context.scoring_plan.fit_scopes[3]
    issue_id = formal_scope.time_permutation_pools.assessment_issue_ids[-1]
    snapshot = placebo_inputs.snapshots_by_issue_id[issue_id]
    sources = tuple(context.feature_layout.dynamic_sources)
    reproductions = []
    for history_issue_id, history_snapshot in placebo_inputs.snapshots_by_issue_id.items():
        identity = hashlib.sha256(
            f"synthetic-primary-convergence:{history_issue_id}".encode()
        ).hexdigest()
        reproductions.append(
            PrimaryGridReproductionReceipt(
                issue_id=history_issue_id,
                issue_index=history_snapshot.issue_index,
                issue_report_id=history_snapshot.summary.issue_report_id,
                accepted_table_sha256=identity,
                recomputed_table_sha256=identity,
            )
        )
    grids = []
    primary_identity = reproductions[-1].recomputed_table_sha256
    for grid in context.grid_family.grids():
        identity = (
            primary_identity
            if grid.cell_size_km == 25.0
            else hashlib.sha256(f"synthetic-{grid.grid_id}".encode()).hexdigest()
        )
        grids.append(
            TargetBlindGridFeatures(
                grid_id=grid.grid_id,
                cell_size_km=grid.cell_size_km,
                cell_ids=grid.cell_ids,
                feature_columns={
                    source: np.zeros(grid.cell_count, dtype=np.float64) for source in sources
                },
                feature_table_identity_sha256=identity,
            )
        )
    return FrozenTargetBlindConvergenceInputs(
        issue_id=issue_id,
        issue_report_id=snapshot.summary.issue_report_id,
        selected_issue_index=snapshot.issue_index,
        selected_state_snapshot_id=snapshot.state_snapshot_id,
        selected_lineage_digest=snapshot.lineage_digest,
        source_columns=sources,
        source_input_sha256=source_input_sha256,
        grids=cast(Any, tuple(grids)),
        primary_reproduction_receipts=tuple(reproductions),
        query_chunk_size=256,
        spatial_workers=1,
    )


def _synthetic_convergence_audit(
    model: FrozenConvergenceModel,
    *,
    fail_spatial: bool = False,
) -> CompensatorConvergenceAudit:
    sources = tuple(item.source_column for item in model.preprocessor.contracts)
    grids = tuple(
        RecomputedGridInputs(
            grid_id=f"synthetic-{cell_size:g}km",
            cell_size_km=cell_size,
            background_spatial_mass_by_cell_and_bin=(
                np.asarray([[0.5, 0.5], [0.5, 0.5]], dtype=np.float64)
                * (1.02 if fail_spatial and cell_size == 12.5 else 1.0)
            ),
            feature_columns={source: np.zeros(2, dtype=np.float64) for source in sources},
        )
        for cell_size in (50.0, 25.0, 12.5)
    )
    return audit_compensator_convergence(model=model, grids=cast(Any, grids))


@pytest.fixture(scope="module")
def scientific_inputs() -> tuple[
    FormalPreflightArtifacts,
    AuthorizedFormalMaterialization,
    Stage4InMemoryPlan,
]:
    context, catalog = _context_and_catalog()
    worker = Stage4WorkerPlan(
        physical_cores=16,
        logical_processors=32,
        reserve_physical_cores=2,
        configured_max_workers=6,
        effective_workers=6,
        blas_threads_per_worker=1,
        nested_parallelism=False,
    )
    scoring_plan = cast(Any, context.scoring_plan)
    scoring_plan.compute = Stage4ComputePlan(
        backend="cpu_float64",
        workers=worker,
        gpu_equivalence_sha256=None,
        gpu_fallback_reason="project_environment_has_no_frozen_gpu_backend",
    )
    scoring_plan.publication = _publication()
    scoring_plan.content_sha256 = "9" * 64
    formal_scope = context.scoring_plan.fit_scopes[3]
    issue_ids = (
        *formal_scope.time_permutation_pools.fit_issue_ids,
        *formal_scope.time_permutation_pools.assessment_issue_ids,
    )
    verified_by_date = {
        issue.issue_report_id.removeprefix("formal-report-"): issue
        for issue in context.verified_issues
    }
    tables = {}
    snapshots = {}
    for index, issue_id in enumerate(issue_ids):
        issue_date = issue_id.removeprefix("anomaly-issue-")
        verified = verified_by_date[issue_date]
        summary = cast(
            AnomalyState,
            SimpleNamespace(
                issue_report_id=verified.issue_report_id,
                issue_time_utc=verified.issue_time_utc,
            ),
        )
        tables[issue_id] = verified.table
        snapshots[issue_id] = Stage3IssueSnapshot(
            issue_index=index,
            issue_time_utc=verified.issue_time_utc,
            summary=summary,
            entities=(),
            state_snapshot_id=verified.state_snapshot_id,
            lineage_digest=verified.lineage_digest,
        )
    placebo_inputs = FrozenPlaceboInputs(
        issue_tables=tables,
        snapshots_by_issue_id=snapshots,
        query_grid=context.grid_family.primary_25km.as_stage3_query_grid(),
        construction_stratum_by_state_id={},
        source_input_sha256="8" * 64,
    )
    receipt = replace(
        make_formal_preflight_receipt(),
        protocol_design_sha256=context.protocol.protocol_design_sha256,
        random_input_seal_sha256=context.protocol.random_input_seal_sha256,
    )
    preflight = FormalPreflightArtifacts(
        context=context,
        receipt=receipt,
        feature_contracts=_contracts(),
        placebo_inputs=placebo_inputs,
        convergence_inputs=_synthetic_convergence_inputs(
            context,
            placebo_inputs,
            source_input_sha256=receipt.content_sha256,
        ),
        model_version="synthetic-stage4-v1",
    )
    materialization = _materialize_in_memory_core(context, catalog)
    plan = build_stage4_in_memory_plan(
        materialization,
        feature_contracts=preflight.feature_contracts,
        feature_layout=context.feature_layout,
        frozen_input_seal_sha256=context.protocol.random_input_seal_sha256,
        model_version=preflight.model_version,
    )
    return preflight, materialization, plan


@pytest.fixture(scope="module")
def pipeline_result() -> PipelineResult:
    return run_stage4_synthetic_test_pipeline(_plan("positive"), placebo_injection=None)


def _inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    preflight: FormalPreflightArtifacts,
    *,
    payload: bytes = b"synthetic-formal-target",
) -> FormalRunInputs:
    del monkeypatch
    target = tmp_path / "synthetic" / "target.bin"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    binding = "c" * 64
    attempt_path = tmp_path.joinpath(*Path(STAGE4_ATTEMPT_LEDGER_RELATIVE_PATH).parts)
    target_ledger_path = tmp_path.joinpath(*Path(STAGE4_TARGET_READ_LEDGER_RELATIVE_PATH).parts)
    initialize_stage4_ledger(
        attempt_path,
        kind="formal_attempt",
        execution_binding_id=binding,
    )
    initialize_stage4_ledger(
        target_ledger_path,
        kind="target_read",
        execution_binding_id=binding,
    )
    receipt = preflight.receipt
    authorization = Stage4TargetAuthorization(
        seal_id="d" * 64,
        execution_binding_id=binding,
        repository_evidence_sha256="e" * 64,
        expected_target_identity=tuple(
            sorted(
                {
                    "content_sha256": "1" * 64,
                    "contract_path": "synthetic/target_contract.json",
                    "contract_sha256": "2" * 64,
                    "path": "synthetic/target.bin",
                    "physical_event_id_column": "event_id",
                    "schema_sha256": "3" * 64,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }.items()
            )
        ),
        formal_backend="cpu_float64",
        formal_preflight_receipt_sha256=receipt.content_sha256,
        space_placebo_resource_observation_sha256=(
            receipt.space_placebo_resource_observation.content_sha256
        ),
        space_placebo_recommended_max_in_flight=(
            receipt.space_placebo_resource_observation.recommended_max_in_flight
        ),
        gpu_requested=True,
        gpu_status="blocked_no_frozen_backend",
        attempt_ledger_path=attempt_path,
        target_read_ledger_path=target_ledger_path,
        _sentinel=authorization_module._AUTHORIZATION_SENTINEL,
    )
    return FormalRunInputs(
        project_root=tmp_path,
        authorization=authorization,
        preflight=preflight,
        execution_protocol={
            "_synthetic_protocol_design_sha256": (preflight.context.protocol.protocol_design_sha256)
        },
        checkpoint_directory=(
            tmp_path / "data" / "interim" / "stage4" / "anomaly_increment_r2" / "checkpoints"
        ),
        concurrency=PlaceboConcurrencyPlan.from_preflight_receipt(
            preflight.context.scoring_plan.compute.workers,
            receipt,
        ),
        same_process_resume_limit=1,
    )


def _patch_scientific_preparation(
    monkeypatch: pytest.MonkeyPatch,
    scientific_inputs: tuple[
        FormalPreflightArtifacts,
        AuthorizedFormalMaterialization,
        Stage4InMemoryPlan,
    ],
) -> None:
    _, materialization, plan = scientific_inputs
    monkeypatch.setattr(
        formal_run,
        "parse_authorized_stage4_target_bytes",
        lambda *_args, **_kwargs: materialization.placebo_scope_assembler.catalog,
    )
    monkeypatch.setattr(
        formal_run,
        "materialize_after_authorized_target",
        lambda *_args, **_kwargs: materialization,
    )
    monkeypatch.setattr(
        formal_run,
        "_build_scientific_components",
        lambda *_args, **_kwargs: (plan, cast(Any, object())),
    )
    monkeypatch.setattr(
        formal_run,
        "audit_frozen_compensator_convergence",
        lambda *, model, **_kwargs: _synthetic_convergence_audit(model),
    )


def test_same_process_resume_reuses_session_without_second_target_ingress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scientific_inputs: tuple[
        FormalPreflightArtifacts,
        AuthorizedFormalMaterialization,
        Stage4InMemoryPlan,
    ],
    pipeline_result: PipelineResult,
) -> None:
    preflight, _, _ = scientific_inputs
    inputs = _inputs(monkeypatch, tmp_path, preflight)
    _patch_scientific_preparation(monkeypatch, scientific_inputs)
    target_reads = 0
    original_read = target_access_module._read_target_bytes_once

    def counted_read(path: Path) -> bytes:
        nonlocal target_reads
        target_reads += 1
        return original_read(path)

    pipeline_calls = 0

    def interrupted_once(*_args: object, **_kwargs: object) -> PipelineResult:
        nonlocal pipeline_calls
        pipeline_calls += 1
        if pipeline_calls == 1:
            raise InfrastructureInterruption("synthetic resumable interruption")
        return pipeline_result

    monkeypatch.setattr(target_access_module, "_read_target_bytes_once", counted_read)
    monkeypatch.setattr(formal_run, "_run_stage4_in_memory_pipeline_core", interrupted_once)
    outcome = run_formal_stage4(inputs)

    assert outcome.status == "succeeded"
    assert outcome.same_process_resume_count == 1
    assert target_reads == 1
    assert pipeline_calls == 2
    assert outcome.convergence_audit is not None
    assert outcome.convergence_audit.passed
    convergence_checkpoint = json.loads(
        (inputs.checkpoint_directory / FORMAL_CONVERGENCE_AUDIT_FILENAME).read_text("utf-8")
    )
    assert convergence_checkpoint["status"] == "passed"
    assert convergence_checkpoint["target_blind_inputs_sha256"] == (
        preflight.convergence_inputs.content_sha256
    )
    assert os.stat(inputs.checkpoint_directory / FORMAL_CONVERGENCE_AUDIT_FILENAME).st_nlink == 1
    assert (tmp_path / FORMAL_CONVERGENCE_OUTPUT_PATH).is_file()
    attempt = read_stage4_ledger(inputs.authorization.attempt_ledger_path)
    target = read_stage4_ledger(inputs.authorization.target_read_ledger_path)
    assert attempt.succeeded_count == 4
    assert {item.result_sha256 for item in attempt.records} == {
        outcome.publication.bundle_id if outcome.publication is not None else None
    }
    assert target.succeeded_count == 1
    seal = json.loads(
        (inputs.checkpoint_directory / FORMAL_SESSION_SEAL_FILENAME).read_text("utf-8")
    )
    assert os.stat(inputs.checkpoint_directory / FORMAL_SESSION_SEAL_FILENAME).st_nlink == 1
    assert seal["backend"] == "cpu_float64"
    assert seal["formal_preflight_receipt_sha256"] == preflight.receipt.content_sha256
    assert seal["gpu"] == {
        "fallback_reason": "project_environment_has_no_frozen_gpu_backend",
        "formal_backend": "cpu_float64",
        "requested": True,
        "status": "blocked_no_frozen_backend",
    }
    assert seal["concurrency"]["time_max_in_flight"] == 6
    assert seal["concurrency"]["space_max_in_flight"] == 2
    assert seal["concurrency"]["space_memory_evidence_sha256"] == (
        preflight.receipt.space_placebo_resource_observation.content_sha256
    )
    assert seal["space_placebo_resource_observation_sha256"] == (
        preflight.receipt.space_placebo_resource_observation.content_sha256
    )
    assert seal["space_placebo_feature_identity"] == {
        "content_sha256": (preflight.receipt.space_placebo_feature_identity.content_sha256),
        "output_identity_sha256": (
            preflight.receipt.space_placebo_feature_identity.output_identity_sha256
        ),
    }
    assert seal["checkpoint_files"] == [
        "time-dynamic-permutations.json",
        "space-dynamic-permutations.json",
        "time-snapshot-permutations.json",
    ]
    assert seal["convergence"] == {
        "audit_file": FORMAL_CONVERGENCE_AUDIT_FILENAME,
        "output_path": FORMAL_CONVERGENCE_OUTPUT_PATH,
        "policy": "all_variant_bin_horizon_spatial_and_temporal_gate_before_publication",
        "target_blind_inputs_sha256": preflight.convergence_inputs.content_sha256,
    }


def test_official_spatial_hook_receives_only_explicit_authorized_session_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scientific_inputs: tuple[
        FormalPreflightArtifacts,
        AuthorizedFormalMaterialization,
        Stage4InMemoryPlan,
    ],
    pipeline_result: PipelineResult,
) -> None:
    preflight, materialization, plan = scientific_inputs
    hook = Stage4SpatialArtifactHook()
    inputs = replace(
        _inputs(monkeypatch, tmp_path, preflight),
        local_artifact_hook=hook,
    )
    _patch_scientific_preparation(monkeypatch, scientific_inputs)
    monkeypatch.setattr(
        formal_run,
        "_run_stage4_in_memory_pipeline_core",
        lambda *_args, **_kwargs: pipeline_result,
    )
    observed: dict[str, object] = {}

    def build_spatial(
        result: PipelineResult,
        received_plan: Stage4InMemoryPlan,
        received_materialization: AuthorizedFormalMaterialization,
        primary_grid: object,
        catalog: object,
        **kwargs: object,
    ) -> object:
        observed.update(
            {
                "catalog": catalog,
                "materialization": received_materialization,
                "plan": received_plan,
                "primary_grid": primary_grid,
                "result": result,
                "study_area": kwargs["study_area"],
            }
        )
        return SimpleNamespace(
            static_svg=(
                '<svg xmlns="http://www.w3.org/2000/svg"><text>'
                "local longitude latitude</text></svg>"
            ),
            interactive_html=(
                '<!doctype html><html><body data-longitude="100" data-latitude="30"></body></html>'
            ),
        )

    monkeypatch.setattr(formal_run, "build_stage4_spatial_results", build_spatial)
    outcome = run_formal_stage4(inputs)

    assert outcome.status == "succeeded"
    assert observed["result"] is pipeline_result
    assert observed["plan"] is plan
    assert observed["materialization"] is materialization
    assert observed["catalog"] is materialization.placebo_scope_assembler.catalog
    assert observed["primary_grid"] is preflight.context.grid_family.primary_25km
    assert isinstance(observed["study_area"], DisplayStudyArea)
    study_area = observed["study_area"]
    assert study_area.role == "frozen_target_independent_display_clip"
    assert study_area.source_content_sha256 == (
        preflight.context.protocol.development_snapshot.study_area_sha256
    )
    assert (tmp_path / hook.static_relative_path).is_file()
    assert (tmp_path / hook.interactive_relative_path).is_file()
    registry_text = (tmp_path / _publication().public_registry).read_text("utf-8")
    assert "longitude" not in registry_text
    assert "latitude" not in registry_text


def test_convergence_failure_is_a_hard_stop_before_success_or_spatial_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scientific_inputs: tuple[
        FormalPreflightArtifacts,
        AuthorizedFormalMaterialization,
        Stage4InMemoryPlan,
    ],
    pipeline_result: PipelineResult,
) -> None:
    preflight, _, _ = scientific_inputs

    class NeverBuildSpatial:
        content_sha256 = "f" * 64

        def build(self, *_args: object, **_kwargs: object) -> tuple[object, ...]:
            raise AssertionError("spatial hook ran after failed convergence")

    inputs = replace(
        _inputs(monkeypatch, tmp_path, preflight),
        local_artifact_hook=cast(Any, NeverBuildSpatial()),
    )
    _patch_scientific_preparation(monkeypatch, scientific_inputs)
    monkeypatch.setattr(
        formal_run,
        "_run_stage4_in_memory_pipeline_core",
        lambda *_args, **_kwargs: pipeline_result,
    )
    monkeypatch.setattr(
        formal_run,
        "audit_frozen_compensator_convergence",
        lambda *, model, **_kwargs: _synthetic_convergence_audit(
            model,
            fail_spatial=True,
        ),
    )

    def forbidden_success(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("success publication ran after failed convergence")

    monkeypatch.setattr(
        formal_run,
        "publish_successful_formal_result",
        forbidden_success,
    )
    outcome = run_formal_stage4(inputs)

    assert outcome.status == "failed"
    assert outcome.failure_code == "compensator_convergence_failure"
    assert read_stage4_ledger(inputs.authorization.attempt_ledger_path).failed_count == 4
    audit = json.loads(
        (inputs.checkpoint_directory / FORMAL_CONVERGENCE_AUDIT_FILENAME).read_text("utf-8")
    )
    assert audit["status"] == "failed"
    assert not (tmp_path / FORMAL_CONVERGENCE_OUTPUT_PATH).exists()
    assert not (tmp_path / _publication().local_spatial_static).exists()
    registry = json.loads((tmp_path / _publication().public_registry).read_text("utf-8"))
    assert registry["status"] == "failed"
    assert registry["scientific_values_available"] is False
    assert registry["convergence"] is None


def test_normal_scientific_or_software_failure_publishes_value_free_output_and_four_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scientific_inputs: tuple[
        FormalPreflightArtifacts,
        AuthorizedFormalMaterialization,
        Stage4InMemoryPlan,
    ],
) -> None:
    preflight, _, _ = scientific_inputs
    inputs = _inputs(monkeypatch, tmp_path, preflight)
    _patch_scientific_preparation(monkeypatch, scientific_inputs)

    def fail(*_args: object, **_kwargs: object) -> PipelineResult:
        raise ValueError("synthetic formal scientific failure")

    monkeypatch.setattr(formal_run, "_run_stage4_in_memory_pipeline_core", fail)
    outcome = run_formal_stage4(inputs)

    assert outcome.status == "failed"
    assert outcome.failure_code == "value_error"
    attempt = read_stage4_ledger(inputs.authorization.attempt_ledger_path)
    assert attempt.failed_count == 4
    assert {item.failure_code for item in attempt.records} == {"value_error"}
    registry = json.loads((tmp_path / _publication().public_registry).read_text("utf-8"))
    assert registry["scientific_values_available"] is False
    assert registry["failure_code"] == "value_error"


def test_target_consumer_failure_is_published_and_all_four_attempts_are_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scientific_inputs: tuple[
        FormalPreflightArtifacts,
        AuthorizedFormalMaterialization,
        Stage4InMemoryPlan,
    ],
) -> None:
    preflight, _, _ = scientific_inputs
    inputs = _inputs(monkeypatch, tmp_path, preflight)

    def bad_target(*_args: object, **_kwargs: object) -> object:
        raise ValueError("synthetic target parse failure")

    monkeypatch.setattr(formal_run, "parse_authorized_stage4_target_bytes", bad_target)
    outcome = run_formal_stage4(inputs)

    assert outcome.status == "failed"
    assert read_stage4_ledger(inputs.authorization.attempt_ledger_path).failed_count == 4
    assert read_stage4_ledger(inputs.authorization.target_read_ledger_path).failed_count == 1
    assert (tmp_path / _publication().local_interactive_html).is_file()


def test_hard_crash_leaves_all_four_started_and_session_retains_authorized_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scientific_inputs: tuple[
        FormalPreflightArtifacts,
        AuthorizedFormalMaterialization,
        Stage4InMemoryPlan,
    ],
) -> None:
    preflight, materialization, _ = scientific_inputs
    inputs = _inputs(monkeypatch, tmp_path, preflight)
    _patch_scientific_preparation(monkeypatch, scientific_inputs)
    session = prepare_formal_run_session(inputs)
    assert session.materialization is materialization
    assert session.catalog is materialization.placebo_scope_assembler.catalog

    def hard_crash(*_args: object, **_kwargs: object) -> PipelineResult:
        raise KeyboardInterrupt

    monkeypatch.setattr(formal_run, "_run_stage4_in_memory_pipeline_core", hard_crash)
    with pytest.raises(KeyboardInterrupt):
        session.execute()

    attempt = read_stage4_ledger(inputs.authorization.attempt_ledger_path)
    assert attempt.started_count == 4
    with pytest.raises(Exception, match="automatic recovery is forbidden"):
        recover_interrupted_stage4_operations(
            inputs.authorization.attempt_ledger_path,
            kind="formal_attempt",
            execution_binding_id=inputs.authorization.execution_binding_id,
        )


def test_session_seal_rejects_changed_checkpoint_concurrency_before_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scientific_inputs: tuple[
        FormalPreflightArtifacts,
        AuthorizedFormalMaterialization,
        Stage4InMemoryPlan,
    ],
) -> None:
    preflight, _, _ = scientific_inputs
    inputs = _inputs(monkeypatch, tmp_path, preflight)
    first = formal_run._write_or_verify_session_seal(inputs)
    with pytest.raises(ValueError, match="resource observation"):
        replace(
            inputs,
            concurrency=PlaceboConcurrencyPlan(
                worker_plan=inputs.concurrency.worker_plan,
                time_max_in_flight=inputs.concurrency.time_max_in_flight,
                space_max_in_flight=1,
                space_memory_evidence_sha256="7" * 64,
            ),
        )
    assert len(first) == 64
    assert read_stage4_ledger(inputs.authorization.target_read_ledger_path).operation_count == 0


def test_formal_artifact_create_only_replay_conflict_and_race_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "checkpoints" / "artifact.json"
    payload = b'{"status":"sealed"}\n'

    assert (
        formal_run._create_or_verify_formal_artifact(
            path,
            payload,
            filesystem_anchor=tmp_path,
            label="synthetic formal artifact",
        )
        == payload
    )
    assert (
        formal_run._create_or_verify_formal_artifact(
            path,
            payload,
            filesystem_anchor=tmp_path,
            label="synthetic formal artifact",
        )
        == payload
    )
    assert os.stat(path).st_nlink == 1
    with pytest.raises(formal_run.FormalRunError, match="different bytes"):
        formal_run._create_or_verify_formal_artifact(
            path,
            b'{"status":"changed"}\n',
            filesystem_anchor=tmp_path,
            label="synthetic formal artifact",
        )

    raced = tmp_path / "checkpoints" / "raced.json"
    attacker = tmp_path / "attacker.bin"
    attacker.write_bytes(b"restricted-target-bytes")
    original_link = os.link

    def replace_after_link(
        source: _PathArgument,
        destination: _PathArgument,
    ) -> None:
        original_link(source, destination)
        os.unlink(destination)
        original_link(attacker, destination)

    monkeypatch.setattr(os, "link", replace_after_link)
    with pytest.raises(immutable_file_module.UnsafeImmutableFileError, match="single-link"):
        formal_run._create_or_verify_formal_artifact(
            raced,
            payload,
            filesystem_anchor=tmp_path,
            label="raced formal artifact",
        )
    assert attacker.read_bytes() == b"restricted-target-bytes"
    assert raced.read_bytes() == b"restricted-target-bytes"
    assert os.path.samefile(raced, attacker)


def test_formal_artifact_identity_bound_rollback_preserves_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "checkpoints" / "artifact.json"
    attacker = tmp_path / "attacker.bin"
    attacker.write_bytes(b"replacement-must-be-preserved")
    original_require = immutable_file_module.require_existing_real_directory
    parent_checks = 0

    def replace_before_parent_recheck(path: Path, *, label: str) -> os.stat_result:
        nonlocal parent_checks
        parent_checks += 1
        if parent_checks == 2:
            os.replace(attacker, destination)
            raise immutable_file_module.UnsafeImmutableFileError(
                "synthetic post-verification parent race"
            )
        return original_require(path, label=label)

    monkeypatch.setattr(
        formal_run,
        "require_existing_real_directory",
        replace_before_parent_recheck,
    )
    with pytest.raises(formal_run.FormalRunError, match="identity-bound rollback"):
        formal_run._create_or_verify_formal_artifact(
            destination,
            b'{"safe":true}\n',
            filesystem_anchor=tmp_path,
            label="rollback-raced formal artifact",
        )
    assert destination.read_bytes() == b"replacement-must-be-preserved"
    assert os.stat(destination).st_nlink == 1


def test_formal_artifact_rejects_linked_parent_directory_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    try:
        os.symlink(real_parent, linked_parent, target_is_directory=True)
    except OSError:
        linked_parent.mkdir()
        _install_lstat_fallback(
            monkeypatch,
            linked_parent,
            reparse=True,
        )

    with pytest.raises(formal_run.FormalRunError, match="unsafe .* parent directory"):
        formal_run._create_or_verify_formal_artifact(
            linked_parent / "artifact.json",
            b'{"safe":true}\n',
            filesystem_anchor=tmp_path,
            label="linked-parent formal artifact",
        )
    assert not (real_parent / "artifact.json").exists()
    assert not (linked_parent / "artifact.json").exists()


def test_formal_artifact_rejects_windows_reparse_parent_before_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "simulated-junction"
    parent.mkdir()
    parent_stat = os.lstat(parent)
    original = immutable_file_module._is_reparse_point

    def simulated_reparse(value: os.stat_result) -> bool:
        return (value.st_dev, value.st_ino) == (
            parent_stat.st_dev,
            parent_stat.st_ino,
        ) or original(value)

    monkeypatch.setattr(
        immutable_file_module,
        "_is_reparse_point",
        simulated_reparse,
    )
    with pytest.raises(formal_run.FormalRunError, match="unsafe .* parent directory"):
        formal_run._create_or_verify_formal_artifact(
            parent / "artifact.json",
            b'{"safe":true}\n',
            filesystem_anchor=tmp_path,
            label="junction-parent formal artifact",
        )
    assert not (parent / "artifact.json").exists()


@pytest.mark.parametrize("link_kind", ("hardlink", "symlink"))
def test_formal_artifact_rejects_links_before_opening_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    link_kind: str,
) -> None:
    target = tmp_path / "restricted-target.bin"
    target.write_bytes(b"must-not-be-read")
    artifact = tmp_path / "checkpoints" / "artifact.json"
    artifact.parent.mkdir()
    try:
        if link_kind == "hardlink":
            os.link(target, artifact)
        else:
            os.symlink(target, artifact)
    except OSError:
        artifact.write_bytes(b"simulated-unsafe-link-entry")
        _install_lstat_fallback(
            monkeypatch,
            artifact,
            reparse=link_kind == "symlink",
            link_count=2 if link_kind == "hardlink" else None,
        )

    original_open = immutable_file_module._open_no_follow

    def forbidden_target_open(path: Path) -> int:
        if Path(path) == artifact:
            raise AssertionError("linked target payload was opened")
        return original_open(path)

    monkeypatch.setattr(
        immutable_file_module,
        "_open_no_follow",
        forbidden_target_open,
    )
    with pytest.raises(immutable_file_module.UnsafeImmutableFileError):
        formal_run._create_or_verify_formal_artifact(
            artifact,
            b'{"safe":true}\n',
            filesystem_anchor=tmp_path,
            label="linked formal artifact",
        )


@pytest.mark.parametrize("link_kind", ("hardlink", "symlink"))
def test_pretarget_linked_session_seal_causes_zero_target_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scientific_inputs: tuple[
        FormalPreflightArtifacts,
        AuthorizedFormalMaterialization,
        Stage4InMemoryPlan,
    ],
    link_kind: str,
) -> None:
    preflight, _, _ = scientific_inputs
    inputs = _inputs(monkeypatch, tmp_path, preflight)
    target = tmp_path / "synthetic" / "target.bin"
    inputs.checkpoint_directory.mkdir(parents=True, exist_ok=True)
    seal_path = inputs.checkpoint_directory / FORMAL_SESSION_SEAL_FILENAME
    try:
        if link_kind == "hardlink":
            os.link(target, seal_path)
        else:
            os.symlink(target, seal_path)
    except OSError:
        seal_path.write_bytes(b"simulated-unsafe-session-seal")
        _install_lstat_fallback(
            monkeypatch,
            seal_path,
            reparse=link_kind == "symlink",
            link_count=2 if link_kind == "hardlink" else None,
        )
    target_reads = 0

    def forbidden_target_read(_path: Path) -> bytes:
        nonlocal target_reads
        target_reads += 1
        raise AssertionError("target ingress ran after unsafe pretarget seal")

    monkeypatch.setattr(
        target_access_module,
        "_read_target_bytes_once",
        forbidden_target_read,
    )
    outcome = run_formal_stage4(inputs)

    assert outcome.status == "failed"
    assert target_reads == 0
    assert read_stage4_ledger(inputs.authorization.target_read_ledger_path).operation_count == 0


def test_hashed_resource_recommendation_controls_space_concurrency_without_two_worker_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scientific_inputs: tuple[
        FormalPreflightArtifacts,
        AuthorizedFormalMaterialization,
        Stage4InMemoryPlan,
    ],
) -> None:
    preflight, _, _ = scientific_inputs
    resource = replace(
        preflight.receipt.space_placebo_resource_observation,
        recommended_max_in_flight=5,
    )
    receipt = replace(
        preflight.receipt,
        space_placebo_resource_observation=resource,
    )
    resource_preflight = replace(preflight, receipt=receipt)
    inputs = _inputs(monkeypatch, tmp_path, resource_preflight)
    assert inputs.concurrency.space_max_in_flight == 5
    assert inputs.concurrency.space_max_in_flight <= (
        inputs.concurrency.worker_plan.effective_workers
    )
    assert inputs.concurrency.worker_plan.reserve_physical_cores >= 2
    assert inputs.concurrency.space_memory_evidence_sha256 == resource.content_sha256
    assert read_stage4_ledger(inputs.authorization.target_read_ledger_path).operation_count == 0


def test_placebo_router_separates_time_and_space_runtime_identities() -> None:
    time_calls: list[tuple[str, str]] = []
    space_calls: list[tuple[str, str]] = []
    marker = cast(PlaceboExecution, object())

    class FakeRuntime:
        def __init__(self, calls: list[tuple[str, str]]) -> None:
            self.calls = calls

        def __call__(self, request: PlaceboRequest) -> PlaceboExecution:
            self.calls.append((request.kind, request.model_variant))
            return marker

    router = formal_run._FormalPlaceboRouter(
        time_runtime=cast(PlaceboRuntime, FakeRuntime(time_calls)),
        space_runtime=cast(PlaceboRuntime, FakeRuntime(space_calls)),
    )

    def request(
        kind: Literal["time", "space"],
        model_variant: Literal["snapshot", "dynamic"],
    ) -> PlaceboRequest:
        return PlaceboRequest(
            kind=kind,
            evaluation_id="formal-validation",
            model_variant=model_variant,
            observed_statistic=0.1,
            frozen_rate_head_sha256="6" * 64,
        )

    assert router(request("time", "dynamic")) is marker
    assert router(request("time", "snapshot")) is marker
    assert router(request("space", "dynamic")) is marker
    assert time_calls == [("time", "dynamic"), ("time", "snapshot")]
    assert space_calls == [("space", "dynamic")]
    with pytest.raises(ValueError, match="outside the frozen three"):
        router(request("space", "snapshot"))


def test_success_publication_with_unconfirmed_attempt_terminalization_is_explicit_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scientific_inputs: tuple[
        FormalPreflightArtifacts,
        AuthorizedFormalMaterialization,
        Stage4InMemoryPlan,
    ],
    pipeline_result: PipelineResult,
) -> None:
    preflight, _, _ = scientific_inputs
    inputs = _inputs(monkeypatch, tmp_path, preflight)
    _patch_scientific_preparation(monkeypatch, scientific_inputs)
    monkeypatch.setattr(
        formal_run,
        "_run_stage4_in_memory_pipeline_core",
        lambda *_args, **_kwargs: pipeline_result,
    )

    def fail_before_terminal_write(*_args: object, **_kwargs: object) -> None:
        raise OSError("synthetic attempt terminalization failure")

    monkeypatch.setattr(
        formal_run,
        "complete_stage4_attempt_scopes",
        fail_before_terminal_write,
    )
    outcome = run_formal_stage4(inputs)

    assert outcome.status == "failed"
    assert outcome.failure_code == "attempt_terminalization_failure"
    assert outcome.result is None
    assert outcome.publication is not None
    assert outcome.publication.status == "succeeded"
    assert outcome.consistency_incident_sha256 is not None
    assert read_stage4_ledger(inputs.authorization.attempt_ledger_path).started_count == 4
    incident_path = inputs.checkpoint_directory / FORMAL_TERMINALIZATION_INCIDENT_FILENAME
    incident = json.loads(incident_path.read_text("utf-8"))
    assert os.stat(incident_path).st_nlink == 1
    assert incident["content_sha256"] == outcome.consistency_incident_sha256
    assert incident["status"] == "formal_attempt_terminalization_unconfirmed"
    assert incident["publication"]["bundle_id"] == outcome.publication.bundle_id
    assert incident["cross_process_recovery_forbidden"] is True
    registry = json.loads((tmp_path / _publication().public_registry).read_text("utf-8"))
    assert registry["status"] == "succeeded"
    seal = json.loads(
        (inputs.checkpoint_directory / FORMAL_SESSION_SEAL_FILENAME).read_text("utf-8")
    )
    assert seal["consistency_incident_file"] == (FORMAL_TERMINALIZATION_INCIDENT_FILENAME)


def test_post_replace_terminalization_exception_is_read_back_not_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scientific_inputs: tuple[
        FormalPreflightArtifacts,
        AuthorizedFormalMaterialization,
        Stage4InMemoryPlan,
    ],
    pipeline_result: PipelineResult,
) -> None:
    preflight, _, _ = scientific_inputs
    inputs = _inputs(monkeypatch, tmp_path, preflight)
    _patch_scientific_preparation(monkeypatch, scientific_inputs)
    monkeypatch.setattr(
        formal_run,
        "_run_stage4_in_memory_pipeline_core",
        lambda *_args, **_kwargs: pipeline_result,
    )
    calls = 0

    def complete_then_raise(
        project_root: Path,
        *,
        execution_binding_id: str,
        operation_ids_by_scope: Mapping[str, str],
        authorization_id: str,
        status: Literal["succeeded", "failed"],
        result_sha256: str | None = None,
        failure_code: str | None = None,
        protocol: Mapping[str, object],
    ) -> None:
        del project_root, protocol
        nonlocal calls
        calls += 1
        _complete_stage4_attempt_scopes(
            inputs.authorization.attempt_ledger_path,
            execution_binding_id=execution_binding_id,
            operation_ids_by_scope=operation_ids_by_scope,
            authorization_id=authorization_id,
            status=status,
            result_sha256=result_sha256,
            failure_code=failure_code,
        )
        raise OSError("synthetic post-replace ambiguity")

    monkeypatch.setattr(
        formal_run,
        "complete_stage4_attempt_scopes",
        complete_then_raise,
    )
    outcome = run_formal_stage4(inputs)

    assert outcome.status == "succeeded"
    assert calls == 1
    assert outcome.consistency_incident_sha256 is None
    assert read_stage4_ledger(inputs.authorization.attempt_ledger_path).succeeded_count == 4
    assert not (inputs.checkpoint_directory / FORMAL_TERMINALIZATION_INCIDENT_FILENAME).exists()


def test_failed_publication_with_unconfirmed_attempt_terminalization_is_audited(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scientific_inputs: tuple[
        FormalPreflightArtifacts,
        AuthorizedFormalMaterialization,
        Stage4InMemoryPlan,
    ],
) -> None:
    preflight, _, _ = scientific_inputs
    inputs = _inputs(monkeypatch, tmp_path, preflight)
    _patch_scientific_preparation(monkeypatch, scientific_inputs)

    def scientific_failure(*_args: object, **_kwargs: object) -> PipelineResult:
        raise ValueError("synthetic scientific failure")

    def fail_before_terminal_write(*_args: object, **_kwargs: object) -> None:
        raise OSError("synthetic failed-attempt terminalization failure")

    monkeypatch.setattr(
        formal_run,
        "_run_stage4_in_memory_pipeline_core",
        scientific_failure,
    )
    monkeypatch.setattr(
        formal_run,
        "complete_stage4_attempt_scopes",
        fail_before_terminal_write,
    )
    outcome = run_formal_stage4(inputs)

    assert outcome.status == "failed"
    assert outcome.failure_code == "attempt_terminalization_failure"
    assert outcome.publication is not None
    assert outcome.publication.status == "failed"
    assert outcome.consistency_incident_sha256 is not None
    assert read_stage4_ledger(inputs.authorization.attempt_ledger_path).started_count == 4
    incident_path = inputs.checkpoint_directory / FORMAL_TERMINALIZATION_INCIDENT_FILENAME
    incident = json.loads(incident_path.read_text("utf-8"))
    assert os.stat(incident_path).st_nlink == 1
    assert incident["expected_attempt_state"]["status"] == "failed"
    assert incident["expected_attempt_state"]["failure_code"] == "value_error"
    assert incident["publication"]["status"] == "failed"
    registry = json.loads((tmp_path / _publication().public_registry).read_text("utf-8"))
    assert registry["status"] == "failed"
