from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from stage4_formal_preflight_fixture import make_formal_preflight_receipt
from test_stage4_anomaly_increment_formal_execution import (
    _context_and_catalog,
    _contracts,
)

import seismoflux.anomaly_increment.formal_production as formal_production
from seismoflux.anomaly_increment.compute import Stage4ComputePlan, Stage4WorkerPlan
from seismoflux.anomaly_increment.config import load_stage4_protocol_bundle
from seismoflux.anomaly_increment.convergence import (
    FrozenTargetBlindConvergenceInputs,
    PrimaryGridReproductionReceipt,
    TargetBlindGridFeatures,
)
from seismoflux.anomaly_increment.formal_preflight import FORMAL_PREFLIGHT_RECEIPT_PATH
from seismoflux.anomaly_increment.formal_production import (
    FormalProductionReadiness,
    assemble_stage4_formal_preflight_artifacts,
    authorize_stage4_formal_readiness,
)
from seismoflux.anomaly_increment.formal_run import Stage4SpatialArtifactHook
from seismoflux.features.anomaly.snapshot import Stage3IssueSnapshot
from seismoflux.features.anomaly.state import AnomalyState

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_stage4_formal.py"
PRODUCTION_MODULE = ROOT / "src" / "seismoflux" / "anomaly_increment" / "formal_production.py"


def _synthetic_pretarget_convergence(
    context: Any,
    *,
    issue_ids: tuple[str, ...],
    snapshots: dict[str, Stage3IssueSnapshot],
    source_input_sha256: str,
) -> FrozenTargetBlindConvergenceInputs:
    sources = tuple(context.feature_layout.dynamic_sources)
    reproductions = []
    for issue_index, issue_id in enumerate(issue_ids):
        snapshot = snapshots[issue_id]
        identity = hashlib.sha256(
            f"synthetic-primary-reproduction:{issue_index}".encode()
        ).hexdigest()
        reproductions.append(
            PrimaryGridReproductionReceipt(
                issue_id=issue_id,
                issue_index=issue_index,
                issue_report_id=snapshot.summary.issue_report_id,
                accepted_table_sha256=identity,
                recomputed_table_sha256=identity,
            )
        )
    issue_id = issue_ids[-1]
    snapshot = snapshots[issue_id]
    final_identity = reproductions[-1].recomputed_table_sha256
    grids = []
    for grid in context.grid_family.grids():
        identity = (
            final_identity
            if grid.cell_size_km == 25.0
            else hashlib.sha256(f"synthetic-grid:{grid.grid_id}".encode()).hexdigest()
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
        spatial_workers=2,
    )


def _load_entry_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("stage4_formal_entry_test", SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError("cannot load the formal production entry")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_production_entry_has_one_explicit_run_call_and_no_alternate_scoring_chain() -> None:
    entry_tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    run_calls = [
        node
        for node in ast.walk(entry_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_formal_stage4"
    ]
    assert len(run_calls) == 1
    forbidden_calls = {
        "consume_authorized_stage4_target",
        "materialize_after_authorized_target",
        "prepare_formal_run_session",
        "run_stage4_in_memory_pipeline",
    }
    assert (
        not {
            node.func.id
            for node in ast.walk(entry_tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        & forbidden_calls
    )

    production_tree = ast.parse(PRODUCTION_MODULE.read_text(encoding="utf-8"))
    imported_modules = {
        alias.name
        for node in ast.walk(production_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or "" for node in ast.walk(production_tree) if isinstance(node, ast.ImportFrom)
    }
    assert not any("target_access" in name for name in imported_modules)
    assert not any("scoring_pipeline" in name for name in imported_modules)
    assert "run_formal_stage4" not in {
        node.func.id
        for node in ast.walk(production_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


def test_check_mode_uses_read_only_proof_without_granting_target_authorization(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_entry_module()
    readiness = SimpleNamespace(
        as_mapping=lambda: {"target_bytes_read": False, "formal_backend": "cpu_float64"}
    )
    proof = SimpleNamespace(
        as_mapping=lambda: {
            "authorization_granted": False,
            "readiness_id": "a" * 64,
            "repository_and_tags_verified": True,
        }
    )
    monkeypatch.setattr(module, "load_stage4_formal_readiness", lambda root: readiness)
    monkeypatch.setattr(module, "verify_stage4_formal_readiness", lambda value: proof)

    def forbidden_authorize(value: object) -> object:
        del value
        raise AssertionError("check mode must not grant a target authorization")

    monkeypatch.setattr(module, "authorize_stage4_formal_readiness", forbidden_authorize)

    def forbidden_run(value: object) -> object:
        del value
        raise AssertionError("check mode must not enter the formal target action")

    monkeypatch.setattr(module, "run_formal_stage4", forbidden_run)
    assert module.main(["check"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "check"
    assert payload["authorization_granted"] is False
    assert payload["repository_and_tags_verified"] is True
    assert payload["target_bytes_read"] is False


@pytest.mark.parametrize(
    ("status", "expected_code"),
    (("succeeded", 0), ("failed", 2)),
)
def test_run_mode_routes_once_through_the_formal_orchestrator(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    status: str,
    expected_code: int,
) -> None:
    module = _load_entry_module()
    readiness = SimpleNamespace(as_mapping=lambda: {})
    inputs = SimpleNamespace(authorization=SimpleNamespace(authorization_id="a" * 64))
    events: list[str] = []

    def load_readiness(root: object) -> object:
        del root
        events.append("target-blind-readiness")
        return readiness

    def authorize(value: object) -> object:
        assert value is readiness
        assert events == ["target-blind-readiness"]
        events.append("target-authorization")
        return inputs

    monkeypatch.setattr(module, "load_stage4_formal_readiness", load_readiness)
    monkeypatch.setattr(module, "authorize_stage4_formal_readiness", authorize)
    calls: list[object] = []

    def run_once(value: object) -> object:
        events.append("formal-run")
        calls.append(value)
        return SimpleNamespace(
            failure_code=None if status == "succeeded" else "synthetic_failure",
            publication=(SimpleNamespace(bundle_id="bundle-1") if status == "succeeded" else None),
            same_process_resume_count=0,
            session_seal_sha256="b" * 64,
            status=status,
        )

    monkeypatch.setattr(module, "run_formal_stage4", run_once)
    assert module.main(["run"]) == expected_code
    assert calls == [inputs]
    assert events == ["target-blind-readiness", "target-authorization", "formal-run"]
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "run"
    assert payload["status"] == status


def test_production_assembler_builds_full_support_and_keeps_shadow_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, _ = _context_and_catalog()
    formal_scope = context.scoring_plan.fit_scopes[3]
    issue_ids = (
        *formal_scope.time_permutation_pools.fit_issue_ids,
        *formal_scope.time_permutation_pools.assessment_issue_ids,
    )
    verified_by_date = {
        issue.issue_report_id.removeprefix("formal-report-"): issue
        for issue in context.verified_issues
    }
    tables: dict[str, object] = {}
    snapshots: dict[str, Stage3IssueSnapshot] = {}
    verified: dict[str, object] = {}
    for index, issue_id in enumerate(issue_ids):
        issue_date = issue_id.removeprefix("anomaly-issue-")
        issue = verified_by_date[issue_date]
        tables[issue_id] = issue.table
        verified[issue_id] = issue
        summary = cast(
            AnomalyState,
            SimpleNamespace(
                issue_report_id=issue.issue_report_id,
                issue_time_utc=issue.issue_time_utc,
            ),
        )
        snapshots[issue_id] = Stage3IssueSnapshot(
            issue_index=index,
            issue_time_utc=issue.issue_time_utc,
            summary=summary,
            entities=(),
            state_snapshot_id=issue.state_snapshot_id,
            lineage_digest=issue.lineage_digest,
        )
    worker = Stage4WorkerPlan(
        physical_cores=16,
        logical_processors=32,
        reserve_physical_cores=2,
        configured_max_workers=12,
        effective_workers=12,
        blas_threads_per_worker=1,
        nested_parallelism=False,
    )
    compute = Stage4ComputePlan(
        backend="cpu_float64",
        workers=worker,
        gpu_equivalence_sha256=None,
        gpu_fallback_reason="project_environment_has_no_frozen_gpu_backend",
    )
    cast(Any, context.scoring_plan).compute = compute
    receipt = replace(
        make_formal_preflight_receipt(),
        protocol_design_sha256=context.protocol.protocol_design_sha256,
        random_input_seal_sha256=context.protocol.random_input_seal_sha256,
    )
    bundle = SimpleNamespace(
        receipt=receipt,
        grid_family=context.grid_family,
        feature_layout=context.feature_layout,
        study_area=context.study_area,
        verified_issues_by_issue_id=verified,
        shadow_issue=context.verified_issues[-1],
        shadow_plans=context.prospective_plans,
        issue_tables=tables,
        snapshots_by_issue_id=snapshots,
        construction_stratum_by_state_id={},
    )
    monkeypatch.setattr(
        cast(Any, formal_production).VerifiedFormalProtocol,
        "from_verified_protocol",
        classmethod(lambda cls, protocol: context.protocol),
    )
    monkeypatch.setattr(
        formal_production,
        "build_stage4_scoring_plan",
        lambda protocol, compute: context.scoring_plan,
    )
    monkeypatch.setattr(
        formal_production,
        "feature_set_contract",
        lambda protocol, variant: SimpleNamespace(
            source_columns=context.feature_layout.dynamic_sources,
            contracts=_contracts(),
        ),
    )
    monkeypatch.setattr(
        formal_production,
        "_load_frozen_cell_zone_mapping",
        lambda protocol, score_blind_inputs, preflight: context.cell_zone_mapping,
    )
    convergence_calls: list[dict[str, object]] = []

    def build_convergence(**kwargs: object) -> FrozenTargetBlindConvergenceInputs:
        convergence_calls.append(kwargs)
        return _synthetic_pretarget_convergence(
            context,
            issue_ids=issue_ids,
            snapshots=snapshots,
            source_input_sha256=receipt.content_sha256,
        )

    monkeypatch.setattr(
        formal_production,
        "build_target_blind_convergence_inputs",
        build_convergence,
    )
    artifacts = assemble_stage4_formal_preflight_artifacts(
        cast(Any, SimpleNamespace()),
        cast(Any, SimpleNamespace()),
        compute,
        cast(Any, bundle),
        receipt,
        scoring_code_commit="1" * 40,
    )
    assert len(artifacts.context.verified_issues) == len(verified) + 1
    assert artifacts.context.verified_issues[-1] is bundle.shadow_issue
    assert all(support.supported_cell_mask.all() for support in artifacts.context.cell_supports)
    assert set(artifacts.placebo_inputs.issue_tables) == set(issue_ids)
    assert bundle.shadow_issue.table not in artifacts.placebo_inputs.issue_tables.values()
    assert len(convergence_calls) == 1
    assert convergence_calls[0]["issue_ids"] == issue_ids
    accepted_tables = cast(
        dict[str, object],
        convergence_calls[0]["accepted_primary_issue_tables"],
    )
    assert tuple(accepted_tables) == issue_ids
    assert artifacts.convergence_inputs.target_bytes_read is False
    assert artifacts.convergence_inputs.target_path_observed is False
    assert len(artifacts.convergence_inputs.primary_reproduction_receipts) == len(issue_ids)


def test_local_cell_mapping_uses_the_authenticated_stage3_to_stage4_id_bridge(
    tmp_path: Path,
) -> None:
    context, _ = _context_and_catalog()
    primary = context.grid_family.primary_25km
    stage3_grid_id = "3" * 64
    all_zones = tuple(f"zone-{index:02d}" for index in range(39))
    zones = tuple(all_zones[index % 39] for index in range(primary.cell_count))
    relative = Path("data/interim/stage4/anomaly_increment_r1/cells.parquet")
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "grid_id": [stage3_grid_id] * primary.cell_count,
                "cell_id": primary.cell_ids,
                "cell_row": primary.rows,
                "cell_column": primary.columns,
                "query_x_m": primary.query_xy_m[:, 0],
                "query_y_m": primary.query_xy_m[:, 1],
                "construction_zone_id": zones,
            }
        ),
        path,
    )
    observed_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    protocol = cast(
        Any,
        SimpleNamespace(
            repository_root=tmp_path,
            protocol={
                "inputs": {
                    "earthquake_target": {"path": "synthetic/target.bin"},
                },
                "spatial_permutation_topology": {
                    "local_restricted_artifacts": {
                        "cell_mapping": relative.as_posix(),
                    }
                },
            },
            spatial_strata=SimpleNamespace(content_sha256="d" * 64),
        ),
    )
    evidence = cast(
        Any,
        SimpleNamespace(restricted_spatial_artifact_hashes=(("cell_mapping", observed_sha256),)),
    )
    preflight = cast(
        Any,
        SimpleNamespace(
            grid_family=context.grid_family,
            receipt=SimpleNamespace(
                bridge=SimpleNamespace(
                    stage3_grid_id=stage3_grid_id,
                    stage4_grid_id=primary.grid_id,
                    cell_count=primary.cell_count,
                )
            ),
        ),
    )
    mapping = cast(Any, formal_production)._load_frozen_cell_zone_mapping(
        protocol,
        evidence,
        preflight,
    )
    assert mapping.grid_id == primary.grid_id
    assert mapping.cell_ids == primary.cell_ids
    assert mapping.all_construction_zone_ids == all_zones


def test_fresh_resource_observation_must_support_the_sealed_space_concurrency() -> None:
    stored = make_formal_preflight_receipt()
    stored_resource = replace(
        stored.space_placebo_resource_observation,
        recommended_max_in_flight=5,
    )
    stored = replace(stored, space_placebo_resource_observation=stored_resource)
    unsafe_fresh = replace(
        stored,
        space_placebo_resource_observation=replace(
            stored_resource,
            recommended_max_in_flight=4,
        ),
    )
    with pytest.raises(ValueError, match="fresh target-blind memory observation"):
        formal_production._verify_receipt_rebuild(unsafe_fresh, stored)

    safer_fresh = replace(
        stored,
        space_placebo_resource_observation=replace(
            stored_resource,
            recommended_max_in_flight=6,
        ),
    )
    formal_production._verify_receipt_rebuild(safer_fresh, stored)


def test_authorization_freezes_canonical_paths_and_official_spatial_hook(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    receipt = make_formal_preflight_receipt()
    seal_path = tmp_path / "data" / "manifests" / "anomaly_increment_r1_scoring_seal.json"
    receipt_path = tmp_path.joinpath(*FORMAL_PREFLIGHT_RECEIPT_PATH.parts)
    frozen_protocol = load_stage4_protocol_bundle(ROOT).protocol
    protocol = cast(
        Any,
        SimpleNamespace(
            repository_root=tmp_path,
            protocol=frozen_protocol,
            design_sha256="a" * 64,
        ),
    )
    qualification = SimpleNamespace(
        formal_preflight_receipt_sha256=receipt.content_sha256,
        scoring_code_commit="1" * 40,
        formal_backend="cpu_float64",
        gpu_requested=True,
        gpu_status="blocked_no_frozen_backend",
    )
    scoring_seal = cast(Any, SimpleNamespace(qualification=qualification))
    worker = Stage4WorkerPlan(
        physical_cores=8,
        logical_processors=16,
        reserve_physical_cores=2,
        configured_max_workers=12,
        effective_workers=6,
        blas_threads_per_worker=1,
        nested_parallelism=False,
    )
    preflight = cast(
        Any,
        SimpleNamespace(
            receipt=receipt,
            context=SimpleNamespace(
                scoring_plan=SimpleNamespace(
                    compute=Stage4ComputePlan(
                        backend="cpu_float64",
                        workers=worker,
                        gpu_equivalence_sha256=None,
                        gpu_fallback_reason=("project_environment_has_no_frozen_gpu_backend"),
                    ),
                    publication=SimpleNamespace(
                        local_spatial_static=(
                            "outputs/visualizations/anomaly_increment_r1_spatial.svg"
                        ),
                        local_spatial_interactive=(
                            "outputs/visualizations/anomaly_increment_r1_spatial.html"
                        ),
                    ),
                )
            ),
        ),
    )
    readiness = FormalProductionReadiness(
        project_root=tmp_path,
        protocol=protocol,
        scoring_seal=scoring_seal,
        scoring_seal_path=seal_path,
        preflight_receipt_path=receipt_path,
        preflight=preflight,
    )
    captured: dict[str, object] = {}
    authorization = SimpleNamespace(authorization_id="b" * 64)

    def fake_authorize(*args: object, **kwargs: object) -> object:
        captured["authorization_args"] = args
        captured["authorization_kwargs"] = kwargs
        return authorization

    def fake_inputs(**kwargs: object) -> object:
        captured["run_inputs"] = kwargs
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(formal_production, "authorize_stage4_target_access", fake_authorize)
    monkeypatch.setattr(formal_production, "FormalRunInputs", fake_inputs)
    inputs = cast(Any, authorize_stage4_formal_readiness(readiness))
    run_inputs = cast(dict[str, object], captured["run_inputs"])
    assert inputs.authorization is authorization
    hook = run_inputs["local_artifact_hook"]
    assert isinstance(hook, Stage4SpatialArtifactHook)
    assert hook.static_relative_path.endswith("anomaly_increment_r1_spatial.svg")
    assert hook.interactive_relative_path.endswith("anomaly_increment_r1_spatial.html")
    assert run_inputs["same_process_resume_limit"] == 1
    assert cast(Path, run_inputs["checkpoint_directory"]) == (
        tmp_path / "data" / "interim" / "stage4" / "anomaly_increment_r1" / "checkpoints"
    )
    authorization_kwargs = cast(dict[str, object], captured["authorization_kwargs"])
    assert cast(Path, authorization_kwargs["attempt_ledger_path"]).is_relative_to(tmp_path)
    assert cast(Path, authorization_kwargs["target_read_ledger_path"]).is_relative_to(tmp_path)
