from __future__ import annotations

import ast
import inspect
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest
import yaml
from shapely.geometry import box

import seismoflux.anomaly_increment.attempt_ledger as attempt_ledger_module
import seismoflux.anomaly_increment.formal_preflight as formal_preflight_module
import seismoflux.anomaly_increment.formal_production as formal_production_module
import seismoflux.anomaly_increment.formal_run as formal_run_module
import seismoflux.anomaly_increment.qualification as qualification_module
import seismoflux.anomaly_increment.target_access as target_access_module
from scripts import build_stage4_formal_preflight as formal_preflight_script
from scripts import build_stage4_scoring_qualification as scoring_qualification_script
from scripts import build_stage4_scoring_seal as scoring_seal_script
from seismoflux.anomaly_increment.attempt_ledger import (
    Stage4LedgerError,
    complete_stage4_attempt_scopes,
    complete_stage4_operation,
    initialize_stage4_ledger,
    read_stage4_ledger,
    recover_interrupted_stage4_operations,
    registered_stage4_attempt,
    reserve_stage4_attempt_scopes,
    reserve_stage4_operation,
)
from seismoflux.anomaly_increment.authorization import (
    Stage4ScoringNotAuthorizedError,
    authorize_stage4_target_access,
    build_stage4_scoring_seal,
    load_stage4_scoring_seal,
    write_stage4_scoring_seal_atomic,
)
from seismoflux.anomaly_increment.compute import (
    GPU_BACKEND_ID,
    BackendEquivalenceEvidence,
    NumpyFloat64Backend,
    build_compute_plan,
    qualify_worker_invariance,
)
from seismoflux.anomaly_increment.config import (
    STAGE4_ATTEMPT_LEDGER_RELATIVE_PATH,
    STAGE4_PROTOCOL_PATH,
    STAGE4_PROTOCOL_TAG,
    STAGE4_RESULT_TAG,
    STAGE4_SCORING_CODE_TAG,
    STAGE4_TARGET_READ_LEDGER_RELATIVE_PATH,
    Stage4R2ExecutionBlockedError,
    load_stage4_protocol_bundle,
    require_stage4_r2_execution_action,
    validate_stage4_r2_execution_contract,
)
from seismoflux.anomaly_increment.formal_preflight import (
    load_formal_preflight,
    load_formal_preflight_receipt,
    load_space_placebo_resource_observation,
)
from seismoflux.anomaly_increment.grid_features import (
    assert_selected_columns_exact,
    build_stage4_grid_family,
    build_stage4_integration_grid,
    extract_raw_feature_matrix,
    selected_table_identity_sha256,
    source_columns_for_variant,
)
from seismoflux.anomaly_increment.qualification import (
    build_stage4_qualification_evidence,
    load_stage4_qualification_evidence,
    write_stage4_qualification_evidence_atomic,
)
from seismoflux.anomaly_increment.runner import (
    ExposurePlan,
    build_stage4_scoring_plan,
)


def test_protocol_bundle_loading_never_probes_the_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guarded_fragment = "earthquake_event"
    original_open = Path.open
    original_exists = Path.exists
    original_stat = Path.stat
    original_read_bytes = Path.read_bytes

    def assert_not_target(path: Path) -> None:
        assert guarded_fragment not in path.as_posix()

    def guarded_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        assert_not_target(path)
        return original_open(path, *args, **kwargs)

    def guarded_exists(path: Path) -> bool:
        assert_not_target(path)
        return original_exists(path)

    def guarded_stat(path: Path, *args: Any, **kwargs: Any) -> Any:
        assert_not_target(path)
        return original_stat(path, *args, **kwargs)

    def guarded_read_bytes(path: Path) -> bytes:
        assert_not_target(path)
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(Path, "exists", guarded_exists)
    monkeypatch.setattr(Path, "stat", guarded_stat)
    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)

    bundle = load_stage4_protocol_bundle(Path.cwd())
    assert bundle.expected_target.relative_path.endswith("earthquake_event.parquet")
    assert bundle.validation_receipt["target_read_count"] == 0


def test_score_free_bundle_identity_is_complete_and_unobserved() -> None:
    bundle = load_stage4_protocol_bundle(Path.cwd())
    identity = bundle.score_free_identity()

    assert identity["protocol_tag"] == STAGE4_PROTOCOL_TAG
    assert identity["expected_scoring_code_tag"] == STAGE4_SCORING_CODE_TAG
    assert identity["formal_attempt_count"] == 0
    assert identity["target_read_count"] == 0
    assert identity["locked_test_run"] is False
    target = identity["expected_target"]
    assert isinstance(target, dict)
    assert target["observed"] is False
    assert len(identity["manifest_file_sha256"]) == 4  # type: ignore[arg-type]
    assert len(identity["manifest_content_sha256"]) == 4  # type: ignore[arg-type]


def test_runtime_defaults_and_machine_contract_are_locked_to_execution_r2() -> None:
    bundle = load_stage4_protocol_bundle(Path.cwd())
    freeze = bundle.protocol["freeze"]
    assert isinstance(freeze, dict)
    assert STAGE4_PROTOCOL_PATH.as_posix() == "configs/anomaly_increment_r2.yaml"
    assert STAGE4_PROTOCOL_TAG.endswith("-protocol-r2")
    assert STAGE4_SCORING_CODE_TAG.endswith("-scoring-code-r2")
    assert STAGE4_RESULT_TAG.endswith("-increment-r2")
    assert freeze["execution_revision"] == "r2"
    validate_stage4_r2_execution_contract(bundle.protocol)


def test_current_r2_blocks_every_scoring_mutation_before_any_artifact_or_target_access(
    tmp_path: Path,
) -> None:
    bundle = load_stage4_protocol_bundle(Path.cwd())
    actions = (
        "formal_preflight",
        "qualification",
        "scoring_seal",
        "formal_attempt_ledger_creation",
        "target_read_ledger_creation",
        "formal_target_read",
        "formal_scoring",
    )
    for action in actions:
        with pytest.raises(Stage4R2ExecutionBlockedError, match="ETAS comparator"):
            require_stage4_r2_execution_action(bundle.protocol, action=action)

    attempt_path = tmp_path.joinpath(*STAGE4_ATTEMPT_LEDGER_RELATIVE_PATH.parts)
    target_path = tmp_path.joinpath(*STAGE4_TARGET_READ_LEDGER_RELATIVE_PATH.parts)
    for path, kind in (
        (attempt_path, "formal_attempt"),
        (target_path, "target_read"),
    ):
        with pytest.raises(Stage4LedgerError, match="ETAS comparator"):
            initialize_stage4_ledger(
                tmp_path,
                kind=kind,  # type: ignore[arg-type]
                execution_binding_id="a" * 64,
                protocol=bundle.protocol,
            )
        assert not path.exists()

    blocked_ledger_calls = (
        lambda: read_stage4_ledger(
            tmp_path,
            kind="formal_attempt",
            protocol=bundle.protocol,
        ),
        lambda: reserve_stage4_operation(
            tmp_path,
            kind="target_read",
            execution_binding_id="a" * 64,
            operation_id="blocked-target-read",
            scope="blocked-target",
            authorization_id="b" * 64,
            protocol=bundle.protocol,
        ),
        lambda: reserve_stage4_attempt_scopes(
            tmp_path,
            execution_binding_id="a" * 64,
            operation_ids_by_scope={},
            authorization_id="b" * 64,
            protocol=bundle.protocol,
        ),
        lambda: complete_stage4_operation(
            tmp_path,
            kind="target_read",
            execution_binding_id="a" * 64,
            operation_id="blocked-target-read",
            status="failed",
            failure_code="blocked",
            protocol=bundle.protocol,
        ),
        lambda: complete_stage4_attempt_scopes(
            tmp_path,
            execution_binding_id="a" * 64,
            operation_ids_by_scope={},
            authorization_id="b" * 64,
            status="failed",
            failure_code="blocked",
            protocol=bundle.protocol,
        ),
        lambda: recover_interrupted_stage4_operations(
            tmp_path,
            kind="formal_attempt",
            execution_binding_id="a" * 64,
            protocol=bundle.protocol,
        ),
        lambda: registered_stage4_attempt(
            tmp_path,
            execution_binding_id="a" * 64,
            operation_id="blocked-attempt",
            scope="development-fold-1",
            authorization_id="b" * 64,
            protocol=bundle.protocol,
        ).__enter__(),
    )
    for blocked_call in blocked_ledger_calls:
        with pytest.raises(Stage4LedgerError, match="ETAS comparator"):
            blocked_call()

    for hostile_root, kind in (
        (tmp_path / "CASE-ALIAS", "formal_attempt"),
        (tmp_path / "trailing-dot.", "target_read"),
        (tmp_path / "trailing-space ", "formal_attempt"),
        (tmp_path / "SEISMO~1", "target_read"),
        (tmp_path / "reparse-or-symlink-alias", "formal_attempt"),
        (tmp_path / "hardlink-alias", "target_read"),
    ):
        with pytest.raises(Stage4LedgerError, match="ETAS comparator"):
            initialize_stage4_ledger(
                hostile_root,
                kind=kind,  # type: ignore[arg-type]
                execution_binding_id="a" * 64,
                protocol=bundle.protocol,
            )
        assert not hostile_root.exists()

    with pytest.raises(Stage4R2ExecutionBlockedError, match="ETAS comparator"):
        build_stage4_qualification_evidence(
            bundle.protocol,
            scoring_code_commit="0" * 40,
            score_blind_input_evidence=None,  # type: ignore[arg-type]
            formal_preflight_receipt=None,  # type: ignore[arg-type]
            logical_identity_replay_audit_sha256="0" * 64,
            stage4_pytest=None,  # type: ignore[arg-type]
            full_pytest=None,  # type: ignore[arg-type]
        )
    with pytest.raises(Stage4R2ExecutionBlockedError, match="ETAS comparator"):
        load_formal_preflight(
            bundle,
            None,  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
            scoring_code_commit="0" * 40,
        )
    with pytest.raises(Stage4ScoringNotAuthorizedError, match="ETAS comparator"):
        build_stage4_scoring_seal(
            bundle.protocol,
            repository=None,  # type: ignore[arg-type]
            score_blind_inputs=None,  # type: ignore[arg-type]
            qualification=None,  # type: ignore[arg-type]
            formal_preflight_receipt=None,  # type: ignore[arg-type]
            attempt_ledger=None,  # type: ignore[arg-type]
            target_read_ledger=None,  # type: ignore[arg-type]
        )
    with pytest.raises(Stage4ScoringNotAuthorizedError, match="ETAS comparator"):
        load_stage4_scoring_seal(
            tmp_path / "missing-seal.json",
            protocol=bundle.protocol,
        )
    with pytest.raises(Stage4ScoringNotAuthorizedError, match="ETAS comparator"):
        write_stage4_scoring_seal_atomic(
            tmp_path / "forbidden-seal.json",
            None,  # type: ignore[arg-type]
            protocol=bundle.protocol,
        )
    assert not (tmp_path / "forbidden-seal.json").exists()
    with pytest.raises(Stage4ScoringNotAuthorizedError, match="ETAS comparator"):
        authorize_stage4_target_access(
            tmp_path,
            bundle.protocol,
            scoring_seal_path=tmp_path / "missing-seal.json",
            attempt_ledger_path=attempt_path,
            target_read_ledger_path=target_path,
        )


def test_blocked_public_artifact_io_never_dereferences_hostile_paths_or_creates_files(
    tmp_path: Path,
) -> None:
    """Every public R2 artifact API must stop before even converting its path."""

    bundle = load_stage4_protocol_bundle(Path.cwd())
    path_conversions = 0

    class HostilePath:
        def __fspath__(self) -> str:
            nonlocal path_conversions
            path_conversions += 1
            raise AssertionError("blocked artifact API dereferenced a hostile path")

    hostile = HostilePath()
    before = tuple(tmp_path.iterdir())
    ledger_calls = (
        lambda: initialize_stage4_ledger(
            hostile,  # type: ignore[arg-type]
            kind="formal_attempt",
            execution_binding_id="a" * 64,
            protocol=bundle.protocol,
        ),
        lambda: read_stage4_ledger(
            hostile,  # type: ignore[arg-type]
            kind="formal_attempt",
            protocol=bundle.protocol,
        ),
        lambda: reserve_stage4_operation(
            hostile,  # type: ignore[arg-type]
            kind="target_read",
            execution_binding_id="a" * 64,
            operation_id="blocked-target-read",
            scope="blocked-target",
            authorization_id="b" * 64,
            protocol=bundle.protocol,
        ),
        lambda: reserve_stage4_attempt_scopes(
            hostile,  # type: ignore[arg-type]
            execution_binding_id="a" * 64,
            operation_ids_by_scope={},
            authorization_id="b" * 64,
            protocol=bundle.protocol,
        ),
        lambda: complete_stage4_operation(
            hostile,  # type: ignore[arg-type]
            kind="target_read",
            execution_binding_id="a" * 64,
            operation_id="blocked-target-read",
            status="failed",
            failure_code="blocked",
            protocol=bundle.protocol,
        ),
        lambda: complete_stage4_attempt_scopes(
            hostile,  # type: ignore[arg-type]
            execution_binding_id="a" * 64,
            operation_ids_by_scope={},
            authorization_id="b" * 64,
            status="failed",
            failure_code="blocked",
            protocol=bundle.protocol,
        ),
        lambda: recover_interrupted_stage4_operations(
            hostile,  # type: ignore[arg-type]
            kind="formal_attempt",
            execution_binding_id="a" * 64,
            protocol=bundle.protocol,
        ),
        lambda: registered_stage4_attempt(
            hostile,  # type: ignore[arg-type]
            execution_binding_id="a" * 64,
            operation_id="blocked-attempt",
            scope="development-fold-1",
            authorization_id="b" * 64,
            protocol=bundle.protocol,
        ),
    )
    for call in ledger_calls:
        with pytest.raises(Stage4LedgerError, match="ETAS comparator"):
            call()

    public_artifact_calls = (
        lambda: load_stage4_qualification_evidence(
            hostile,  # type: ignore[arg-type]
            protocol=bundle.protocol,
        ),
        lambda: write_stage4_qualification_evidence_atomic(
            hostile,  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
            protocol=bundle.protocol,
        ),
        lambda: load_formal_preflight_receipt(
            hostile,  # type: ignore[arg-type]
            protocol=bundle.protocol,
        ),
        lambda: load_space_placebo_resource_observation(
            hostile,  # type: ignore[arg-type]
            protocol=bundle.protocol,
        ),
        lambda: formal_preflight_script._write_atomic(
            hostile,  # type: ignore[arg-type]
            {},
            protocol=bundle.protocol,
        ),
    )
    for call in public_artifact_calls:
        with pytest.raises(Stage4R2ExecutionBlockedError, match="ETAS comparator"):
            call()

    assert path_conversions == 0
    assert tuple(tmp_path.iterdir()) == before


def test_public_r2_artifact_io_functions_have_guard_as_first_executable_statement() -> None:
    """Prevent a future resolve/stat/open/mkdir/tempfile/lock/hash prelude."""

    artifact_guards = {
        qualification_module.load_stage4_qualification_evidence: (
            "require_stage4_r2_execution_action",
            "qualification",
        ),
        qualification_module.write_stage4_qualification_evidence_atomic: (
            "require_stage4_r2_execution_action",
            "qualification",
        ),
        formal_preflight_module.load_formal_preflight_receipt: (
            "require_stage4_r2_execution_action",
            "formal_preflight",
        ),
        formal_preflight_module.load_space_placebo_resource_observation: (
            "require_stage4_r2_execution_action",
            "formal_preflight",
        ),
        formal_preflight_script._write_atomic: (
            "require_stage4_r2_execution_action",
            "formal_preflight",
        ),
    }
    ledger_functions = (
        attempt_ledger_module.initialize_stage4_ledger,
        attempt_ledger_module.read_stage4_ledger,
        attempt_ledger_module.reserve_stage4_operation,
        attempt_ledger_module.reserve_stage4_attempt_scopes,
        attempt_ledger_module.complete_stage4_operation,
        attempt_ledger_module.complete_stage4_attempt_scopes,
        attempt_ledger_module.recover_interrupted_stage4_operations,
        attempt_ledger_module.registered_stage4_attempt,
    )

    for function, (guard_name, action) in artifact_guards.items():
        tree = ast.parse(inspect.getsource(function))
        definition = tree.body[0]
        assert isinstance(definition, ast.FunctionDef)
        statements = list(definition.body)
        if (
            statements
            and isinstance(statements[0], ast.Expr)
            and isinstance(statements[0].value, ast.Constant)
            and isinstance(statements[0].value.value, str)
        ):
            statements = statements[1:]
        first = statements[0]
        assert isinstance(first, ast.Expr)
        assert isinstance(first.value, ast.Call)
        assert isinstance(first.value.func, ast.Name)
        assert first.value.func.id == guard_name
        action_keywords = [item for item in first.value.keywords if item.arg == "action"]
        assert len(action_keywords) == 1
        assert isinstance(action_keywords[0].value, ast.Constant)
        assert action_keywords[0].value.value == action

    for function in ledger_functions:
        tree = ast.parse(inspect.getsource(function))
        definition = tree.body[0]
        assert isinstance(definition, ast.FunctionDef)
        statements = list(definition.body)
        if (
            statements
            and isinstance(statements[0], ast.Expr)
            and isinstance(statements[0].value, ast.Constant)
            and isinstance(statements[0].value.value, str)
        ):
            statements = statements[1:]
        first = statements[0]
        assert isinstance(first, ast.Expr)
        assert isinstance(first.value, ast.Call)
        assert isinstance(first.value.func, ast.Name)
        assert first.value.func.id == "_require_r2_ledger_action"


def test_private_internal_cores_have_only_guarded_repository_production_callers() -> None:
    """Keep private mechanics behind the audited public production call graph."""

    repository = Path.cwd()
    attempt_ledger_path = "src/seismoflux/anomaly_increment/attempt_ledger.py"
    allowed_callers = {
        "_initialize_stage4_ledger_generic": {attempt_ledger_path},
        "_read_stage4_ledger_generic": {attempt_ledger_path},
        "_reserve_stage4_operation_generic": {attempt_ledger_path},
        "_reserve_stage4_attempt_scopes_generic": {attempt_ledger_path},
        "_complete_stage4_operation_generic": {attempt_ledger_path},
        "_complete_stage4_attempt_scopes_generic": {attempt_ledger_path},
        "_recover_interrupted_stage4_operations_generic": {attempt_ledger_path},
        "_registered_stage4_attempt_generic": {attempt_ledger_path},
        "_load_stage4_qualification_evidence_generic": {
            "src/seismoflux/anomaly_increment/qualification.py"
        },
        "_write_stage4_qualification_evidence_atomic_generic": {
            "src/seismoflux/anomaly_increment/qualification.py"
        },
        "_load_formal_preflight_receipt_generic": {
            "src/seismoflux/anomaly_increment/formal_preflight.py"
        },
        "_run_stage4_in_memory_pipeline_core": {"src/seismoflux/anomaly_increment/formal_run.py"},
    }
    allowed_imports = {
        "_run_stage4_in_memory_pipeline_core": {"src/seismoflux/anomaly_increment/formal_run.py"}
    }
    observed_calls = {name: set() for name in allowed_callers}
    observed_imports = {name: set() for name in allowed_callers}

    production_files = tuple(repository.joinpath("src").rglob("*.py")) + tuple(
        repository.joinpath("scripts").rglob("*.py")
    )
    for path in production_files:
        relative = path.relative_to(repository).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    name = node.func.attr
                else:
                    continue
                if name in observed_calls:
                    observed_calls[name].add(relative)
            elif isinstance(node, ast.ImportFrom):
                for imported in node.names:
                    if imported.name in observed_imports:
                        observed_imports[imported.name].add(relative)

    assert {
        name: callers - allowed_callers[name]
        for name, callers in observed_calls.items()
        if callers - allowed_callers[name]
    } == {}
    assert {
        name: importers - allowed_imports.get(name, set())
        for name, importers in observed_imports.items()
        if importers - allowed_imports.get(name, set())
    } == {}
    assert observed_calls["_run_stage4_in_memory_pipeline_core"] == {
        "src/seismoflux/anomaly_increment/formal_run.py"
    }


@pytest.mark.parametrize(
    "entrypoint",
    (formal_run_module.prepare_formal_run_session, formal_run_module.run_formal_stage4),
)
def test_blocked_formal_entrypoints_do_not_probe_or_hash_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    entrypoint: Any,
) -> None:
    bundle = load_stage4_protocol_bundle(Path.cwd())
    inputs = object.__new__(formal_run_module.FormalRunInputs)
    object.__setattr__(inputs, "execution_protocol", bundle.protocol)
    target = (tmp_path / "data" / "private" / "earthquake_event.parquet").resolve()
    probes = {"exists": 0, "stat": 0, "open": 0, "hash": 0}
    original_exists = Path.exists
    original_stat = Path.stat
    original_open = Path.open

    def is_target(path: Path) -> bool:
        return Path(path).resolve() == target

    def counted_exists(path: Path) -> bool:
        if is_target(path):
            probes["exists"] += 1
        return original_exists(path)

    def counted_stat(path: Path, *args: Any, **kwargs: Any) -> Any:
        if is_target(path):
            probes["stat"] += 1
        return original_stat(path, *args, **kwargs)

    def counted_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        if is_target(path):
            probes["open"] += 1
        return original_open(path, *args, **kwargs)

    def forbidden_target_consume(*args: Any, **kwargs: Any) -> Any:
        probes["hash"] += 1
        raise AssertionError("blocked formal entry reached the target consume/hash boundary")

    monkeypatch.setattr(Path, "exists", counted_exists)
    monkeypatch.setattr(Path, "stat", counted_stat)
    monkeypatch.setattr(Path, "open", counted_open)
    monkeypatch.setattr(
        formal_run_module,
        "consume_authorized_stage4_target",
        forbidden_target_consume,
    )

    with pytest.raises(Stage4R2ExecutionBlockedError, match="ETAS comparator"):
        entrypoint(inputs)

    assert probes == {"exists": 0, "stat": 0, "open": 0, "hash": 0}


def test_blocked_formal_session_stops_before_scoring_core_or_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = yaml.safe_load((Path.cwd() / STAGE4_PROTOCOL_PATH).read_text(encoding="utf-8"))
    assert isinstance(protocol, dict)
    inputs = object.__new__(formal_run_module.FormalRunInputs)
    object.__setattr__(inputs, "execution_protocol", protocol)
    session = object.__new__(formal_run_module.FormalRunSession)
    object.__setattr__(session, "inputs", inputs)
    touches = {"callback": 0, "scoring_core": 0}

    class ForbiddenCallbackProbe:
        def __getattribute__(self, name: str) -> Any:
            touches["callback"] += 1
            raise AssertionError(f"blocked formal session touched callback state: {name}")

    object.__setattr__(session, "placebo_router", ForbiddenCallbackProbe())

    def forbidden_scoring(*args: Any, **kwargs: Any) -> Any:
        touches["scoring_core"] += 1
        raise AssertionError("blocked formal session reached scientific scoring")

    monkeypatch.setattr(
        formal_run_module,
        "_run_stage4_in_memory_pipeline_core",
        forbidden_scoring,
    )

    with pytest.raises(Stage4R2ExecutionBlockedError, match="ETAS comparator"):
        session.execute()

    assert touches == {"callback": 0, "scoring_core": 0}


def test_direct_target_consumer_hard_stops_before_capability_or_target_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle = load_stage4_protocol_bundle(Path.cwd())
    touched = {"capability": 0, "ledger": 0, "path": 0, "target": 0, "hash": 0}

    def forbidden(label: str) -> Any:
        def fail(*args: Any, **kwargs: Any) -> Any:
            touched[label] += 1
            raise AssertionError(f"blocked direct target consumer touched {label}")

        return fail

    monkeypatch.setattr(
        target_access_module,
        "require_stage4_target_authorization",
        forbidden("capability"),
    )
    monkeypatch.setattr(
        target_access_module,
        "reserve_stage4_operation",
        forbidden("ledger"),
    )
    monkeypatch.setattr(
        target_access_module,
        "_target_path_after_reservation",
        forbidden("path"),
    )
    monkeypatch.setattr(
        target_access_module,
        "_read_target_bytes_once",
        forbidden("target"),
    )
    monkeypatch.setattr(
        target_access_module,
        "hashlib",
        type("ForbiddenHashlib", (), {"sha256": staticmethod(forbidden("hash"))}),
    )

    with pytest.raises(Stage4R2ExecutionBlockedError, match="ETAS comparator"):
        target_access_module.consume_authorized_stage4_target(
            tmp_path,
            object(),  # type: ignore[arg-type]
            protocol=bundle.protocol,
            operation_id="forged-capability-attempt",
            consumer=lambda payload: payload,
        )

    assert touched == {
        "capability": 0,
        "ledger": 0,
        "path": 0,
        "target": 0,
        "hash": 0,
    }


def test_blocked_scoring_seal_check_reads_no_seal_preflight_or_ledger(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle = load_stage4_protocol_bundle(Path.cwd())
    reads = {"seal": 0, "preflight": 0, "ledger": 0}

    def forbidden(label: str) -> Any:
        def fail(*args: Any, **kwargs: Any) -> Any:
            reads[label] += 1
            raise AssertionError(f"blocked scoring-seal check read {label}")

        return fail

    monkeypatch.setattr(
        scoring_seal_script,
        "load_stage4_scoring_seal",
        forbidden("seal"),
    )
    monkeypatch.setattr(
        scoring_seal_script,
        "load_formal_preflight_receipt",
        forbidden("preflight"),
    )
    monkeypatch.setattr(scoring_seal_script, "read_stage4_ledger", forbidden("ledger"))

    with pytest.raises(Stage4R2ExecutionBlockedError, match="ETAS comparator"):
        scoring_seal_script.check(
            tmp_path,
            bundle.protocol,
            qualification_path=tmp_path / "qualification.json",
            preflight_receipt_path=tmp_path / "preflight.json",
            attempt_ledger_path=tmp_path / "attempt.json",
            target_read_ledger_path=tmp_path / "target-read.json",
            repository_adapter=None,  # type: ignore[arg-type]
        )

    assert reads == {"seal": 0, "preflight": 0, "ledger": 0}


def test_blocked_formal_production_reads_no_seal_or_preflight(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle = load_stage4_protocol_bundle(Path.cwd())
    reads = {"seal": 0, "preflight": 0, "receipt": 0}

    def forbidden(label: str) -> Any:
        def fail(*args: Any, **kwargs: Any) -> Any:
            reads[label] += 1
            raise AssertionError(f"blocked formal production read {label}")

        return fail

    monkeypatch.setattr(
        formal_production_module,
        "load_stage4_protocol_bundle",
        lambda _root: bundle,
    )
    monkeypatch.setattr(
        formal_production_module,
        "load_stage4_scoring_seal",
        forbidden("seal"),
    )
    monkeypatch.setattr(
        formal_production_module,
        "load_formal_preflight",
        forbidden("preflight"),
    )
    monkeypatch.setattr(
        formal_production_module,
        "load_formal_preflight_receipt",
        forbidden("receipt"),
    )

    with pytest.raises(Stage4R2ExecutionBlockedError, match="ETAS comparator"):
        formal_production_module.load_stage4_formal_readiness(tmp_path)

    assert reads == {"seal": 0, "preflight": 0, "receipt": 0}


@pytest.mark.parametrize(
    "entrypoint",
    (scoring_qualification_script.generate, scoring_qualification_script.check),
)
def test_blocked_qualification_entrypoints_touch_no_output_or_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    entrypoint: Any,
) -> None:
    bundle = load_stage4_protocol_bundle(Path.cwd())
    touched = {"path": 0, "build": 0, "evidence": 0}

    def forbidden(label: str) -> Any:
        def fail(*args: Any, **kwargs: Any) -> Any:
            touched[label] += 1
            raise AssertionError(f"blocked qualification touched {label}")

        return fail

    monkeypatch.setattr(
        scoring_qualification_script,
        "require_score_blind_project_path",
        forbidden("path"),
    )
    monkeypatch.setattr(scoring_qualification_script, "_build", forbidden("build"))
    monkeypatch.setattr(
        scoring_qualification_script,
        "load_stage4_qualification_evidence",
        forbidden("evidence"),
    )

    with pytest.raises(Stage4R2ExecutionBlockedError, match="ETAS comparator"):
        entrypoint(
            tmp_path,
            bundle.protocol,
            stage4_junit_path=tmp_path / "stage4.xml",
            full_junit_path=tmp_path / "full.xml",
            preflight_receipt_path=tmp_path / "preflight.json",
            qualification_path=tmp_path / "qualification.json",
        )

    assert touched == {"path": 0, "build": 0, "evidence": 0}


@pytest.mark.parametrize(
    ("field", "legacy_value"),
    (
        ("execution_revision", "r1"),
        ("pre_score_tag", "v0.3.0-anomaly-increment-protocol-r1"),
        ("results_tag", "v0.3.0-anomaly-increment-r1"),
    ),
)
def test_runtime_machine_contract_rejects_r1_freeze_identity(
    field: str,
    legacy_value: str,
) -> None:
    protocol = deepcopy(dict(load_stage4_protocol_bundle(Path.cwd()).protocol))
    freeze = protocol["freeze"]
    assert isinstance(freeze, dict)
    freeze[field] = legacy_value

    with pytest.raises(ValueError, match="stage-4"):
        validate_stage4_r2_execution_contract(protocol)


@pytest.mark.parametrize(
    ("path", "drifted_value"),
    (
        (("frozen_on",), "2026-07-16"),
        (("status",), "preregistered_before_any_stage4_target_score"),
        (("compute", "max_workers"), 7),
        (
            ("freeze", "scoring_code_freeze", "test_count_binding", "expected_test_count"),
            1089,
        ),
        (
            (
                "evaluation",
                "machine_status_vocabularies",
                "scientific_gate_status",
                "allowed",
            ),
            ["passed", "fail"],
        ),
        (
            ("evaluation", "permutations", "primary_statistic_comparator_variant"),
            "kde_background_no_increment",
        ),
        (("evaluation", "permutations", "formal_requests"), []),
        (("evaluation", "permutations", "exact_request_set_required"), False),
        (
            (
                "evaluation",
                "multiple_comparisons",
                "exploratory_holm_families",
                "exact_member_count_per_family",
            ),
            9,
        ),
        (
            (
                "evaluation",
                "multiple_comparisons",
                "confirmatory_gatekeeping",
                "future_post_etas_preregistered_design",
                "all_four_observed_raw_statistics_and_full_null_distributions_always_computed",
            ),
            False,
        ),
        (
            (
                "evaluation",
                "multiple_comparisons",
                "confirmatory_gatekeeping",
                "future_post_etas_preregistered_design",
                "comparator_contract",
                "mandatory_primary_scientific_comparator",
                "status_required",
            ),
            "not_evaluable",
        ),
        (
            (
                "evaluation",
                "multiple_comparisons",
                "confirmatory_gatekeeping",
                "future_post_etas_preregistered_design",
                "snapshot_qualification_gate",
                "otherwise",
                "qualification_status",
            ),
            "failed",
        ),
        (
            (
                "evaluation",
                "multiple_comparisons",
                "confirmatory_gatekeeping",
                "future_post_etas_preregistered_design",
                "gate_record_identity_requires",
            ),
            ["gate_reached", "gate_reached_reason"],
        ),
        (("evaluation", "gates", "G2", "primary_space_permutation_p_lte"), 0.10),
        (
            ("evaluation", "gates", "G2", "comparator_variant"),
            "kde_background_no_increment",
        ),
        (
            (
                "evaluation",
                "gates",
                "G2",
                "direct_candidate_minus_etas_G2_result_required_for_each_evaluated_variant",
            ),
            False,
        ),
        (
            ("evaluation", "gates", "G2", "practical_improvement_any_of"),
            [],
        ),
        (
            (
                "evaluation",
                "gates",
                "development_increment_stability",
                "minimum_positive_folds_per_variant_and_comparator",
            ),
            1,
        ),
        (
            (
                "evaluation",
                "gates",
                "development_increment_stability",
                "result_identity",
                "expected_horizon_row_count",
            ),
            53,
        ),
        (
            (
                "evaluation",
                "regional_stability",
                "pass_requirements_per_variant_and_comparator",
                "minimum_positive_event_bearing_regions",
            ),
            1,
        ),
        (
            (
                "evaluation",
                "regional_stability",
                "strict_recall_diagnostic",
                "zero_denominator_is_diagnostic_not_track_evidence_insufficient",
            ),
            False,
        ),
        (
            (
                "evaluation",
                "regional_stability",
                "result_identity",
                "sum_region_supported_unique_physical_event_count_must_equal_global_"
                "supported_unique_physical_event_count_within_horizon",
            ),
            False,
        ),
        (
            (
                "evaluation",
                "adoption_matrix",
                "snapshot_independent_rescue_when_dynamic_G2_not_pass",
            ),
            "allowed",
        ),
        (
            ("evaluation", "adoption_matrix", "G3_not_pass_statuses"),
            ["fail", "evidence_insufficient"],
        ),
        (
            (
                "evaluation",
                "adoption_matrix",
                "direct_etas_track_pass_required_for_any_anomaly_adoption",
            ),
            False,
        ),
        (
            (
                "spatial_permutation_topology",
                "local_restricted_artifacts",
                "access_control",
                "windows",
                "inherited_ace_count_required",
            ),
            1,
        ),
        (
            (
                "inputs",
                "earthquake_target",
                "frozen_catalog_coverage",
                "observed_available_at_max_utc",
            ),
            "2025-07-01T00:00:00Z",
        ),
        (
            ("publication", "result_identity_contract", "future_required_objects"),
            ["dynamic_G2"],
        ),
        (
            (
                "publication",
                "display_semantics",
                "etas_background_primary_comparator_option_required",
            ),
            False,
        ),
        (("publication", "display_semantics", "coverage_only_option_required"), False),
        (("publication", "spatial_output_isolation", "physical_file_count"), 2),
        (
            (
                "publication",
                "spatial_output_isolation",
                "retrospective_access_control",
                "receipt_path",
            ),
            "outputs/visualizations/unbound.json",
        ),
        (
            (
                "publication",
                "spatial_output_isolation",
                "retrospective_access_control",
                "same_verified_handle_held_from_acl_check_through_final_write",
            ),
            False,
        ),
        (
            (
                "publication",
                "spatial_output_isolation",
                "public_forecast_artifact_validator",
                "keyword_scan_or_ui_hiding_sufficient",
            ),
            True,
        ),
        (
            (
                "evaluation",
                "gates",
                "G2",
                "reporting_confound_guard_applies_independently_to_variants",
            ),
            ["dynamic"],
        ),
        (("freeze", "formal_validation_scientific_runs_allowed"), 2),
        (("freeze", "validation_is_not_a_tuning_partition"), False),
        (
            (
                "freeze",
                "r2_protocol_tag_required_before_any_future_execution_revision_scoring_code_changes",
            ),
            False,
        ),
        (("locked_test", "action"), "run"),
        (("locked_test", "run"), True),
        (("evaluation", "gates", "G3", "minimum_positive_folds"), 1),
        (("evaluation", "gates", "G2", "minimum_unique_independent_events", "threshold"), 1),
        (("evaluation", "gates", "G2", "macro_information_gain_lower_95pct_bound_gt"), -1),
        (
            (
                "evaluation",
                "multiple_comparisons",
                "confirmatory_gatekeeping",
                "etas_comparator_prerequisite",
                "required_for_any_G2_gate_reached_or_anomaly_adoption",
            ),
            False,
        ),
        (
            (
                "evaluation",
                "regional_stability",
                "result_identity",
                "expected_horizon_row_count",
            ),
            467,
        ),
    ),
)
def test_runtime_machine_contract_rejects_r2_critical_drift(
    path: tuple[str, ...],
    drifted_value: object,
) -> None:
    protocol = deepcopy(dict(load_stage4_protocol_bundle(Path.cwd()).protocol))
    node: dict[str, Any] = protocol
    for key in path[:-1]:
        child = node[key]
        assert isinstance(child, dict)
        node = child
    node[path[-1]] = drifted_value

    with pytest.raises(ValueError, match="stage-4 R2"):
        validate_stage4_r2_execution_contract(protocol)


def test_target_blind_scoring_plan_freezes_all_fit_scopes_and_counts() -> None:
    bundle = load_stage4_protocol_bundle(Path.cwd())
    plan = build_stage4_scoring_plan(bundle)

    assert [len(scope.fit_exposures_7d) for scope in plan.fit_scopes] == [12, 25, 38, 50]
    assert [[len(item.exposures) for item in scope.assessments] for scope in plan.fit_scopes] == [
        [12, 2, 1],
        [12, 1, 1],
        [11, 2, 1],
        [51, 11, 4, 2, 1],
    ]
    assert plan.formal_attempt_count == 0
    assert plan.target_read_count == 0
    assert plan.locked_test_run is False
    assert len(plan.content_sha256) == 64
    assert plan.content_sha256 == build_stage4_scoring_plan(bundle).content_sha256

    serialized = json.dumps(plan.as_dict(), sort_keys=True)
    assert "earthquake_event" not in serialized
    assert "epicenter" not in serialized


def test_exposure_parser_preserves_open_closed_window() -> None:
    exposure = ExposurePlan.parse("validation-h090-2025-04-03")
    assert exposure.identifier == "validation-h090-2025-04-03"
    assert exposure.target_start_exclusive_local.isoformat() == "2025-04-03"
    assert exposure.target_end_inclusive_local.isoformat() == "2025-07-02"

    with pytest.raises(ValueError, match="exposure horizon changed"):
        ExposurePlan.parse("validation-h091-2025-04-03")


def test_compute_plan_reserves_cores_and_keeps_current_protocol_cpu_only() -> None:
    bundle = load_stage4_protocol_bundle(Path.cwd())
    evidence = BackendEquivalenceEvidence(
        candidate_backend=GPU_BACKEND_ID,
        objective_relative_error=0.0,
        gradient_max_abs_error=0.0,
        coefficient_max_abs_error=0.0,
        integrated_intensity_relative_error=0.0,
        random_mapping_byte_identity=True,
        scientific_decision_identity=True,
        repeated_run_identity=True,
        worker_count_identity=True,
    )
    plan = build_compute_plan(
        bundle,
        requested_backend=GPU_BACKEND_ID,
        gpu_evidence=evidence,
        detected_physical_cores=24,
        detected_logical_processors=48,
    )

    assert plan.backend == "cpu_float64"
    assert plan.gpu_fallback_reason == "project_environment_has_no_frozen_gpu_backend"
    assert plan.workers.effective_workers == 6
    assert plan.workers.reserve_physical_cores == 2
    assert set(plan.workers.blas_environment().values()) == {"1"}


def test_compute_plan_caps_configured_physical_cores_at_available_logical_cores() -> None:
    bundle = load_stage4_protocol_bundle(Path.cwd())
    plan = build_compute_plan(bundle, detected_logical_processors=4)

    assert plan.workers.physical_cores == 4
    assert plan.workers.logical_processors == 4
    assert plan.workers.reserve_physical_cores == 2
    assert plan.workers.effective_workers == 2


def test_compute_plan_rejects_explicitly_inconsistent_detected_core_counts() -> None:
    bundle = load_stage4_protocol_bundle(Path.cwd())

    with pytest.raises(ValueError, match="logical processor count cannot be below physical"):
        build_compute_plan(
            bundle,
            detected_physical_cores=24,
            detected_logical_processors=4,
        )


def test_compute_plan_fails_closed_when_two_cores_cannot_be_reserved() -> None:
    bundle = load_stage4_protocol_bundle(Path.cwd())

    with pytest.raises(ValueError, match="reserved physical cores must be fewer"):
        build_compute_plan(bundle, detected_logical_processors=2)


def test_numpy_float64_backend_rejects_nonfinite_and_shape_drift() -> None:
    backend = NumpyFloat64Backend()
    result = backend.matvec([[1.0, 2.0], [3.0, 4.0]], [0.5, -0.25])
    np.testing.assert_array_equal(result, np.asarray([0.0, 0.5], dtype=np.float64))
    np.testing.assert_allclose(backend.exp([0.0, 1.0]), [1.0, np.e])

    with pytest.raises(ValueError, match="finite"):
        backend.as_float64([np.nan])
    with pytest.raises(ValueError, match="shapes"):
        backend.matvec([[1.0, 2.0]], [1.0])


def test_worker_count_qualification_preserves_result_order_and_bytes() -> None:
    evidence = qualify_worker_invariance(
        lambda value: np.asarray([value, value * value], dtype=np.float64),
        tuple(range(8)),
        worker_counts=(1, 2, 4),
    )

    assert evidence.passed is True
    assert evidence.worker_counts == (1, 2, 4)
    assert len(set(evidence.result_digests)) == 1
    np.testing.assert_array_equal(evidence.reference_results[3], [3.0, 9.0])


def test_multigrid_builder_has_no_target_argument_and_is_deterministic() -> None:
    parameters = set(inspect.signature(build_stage4_integration_grid).parameters)
    assert not any(
        "target" in value or "event" in value or "epicenter" in value for value in parameters
    )

    study_area = box(105.0, 34.0, 105.05, 34.05)
    family = build_stage4_grid_family(study_area)
    repeated = build_stage4_integration_grid(study_area, cell_size_km=25.0)

    assert repeated.grid_id == family.primary_25km.grid_id
    assert repeated.cell_ids == family.primary_25km.cell_ids
    assert repeated.cell_count > 0
    assert family.coarse_50km.total_area_km2 == pytest.approx(
        family.reference_12_5km.total_area_km2,
        rel=1e-12,
        abs=1e-6,
    )


def test_feature_manifest_excludes_quality_companions_from_sources() -> None:
    bundle = load_stage4_protocol_bundle(Path.cwd())
    coverage = source_columns_for_variant(bundle, "coverage_only")
    snapshot = source_columns_for_variant(bundle, "snapshot")
    dynamic = source_columns_for_variant(bundle, "dynamic")

    assert len(coverage) == 9
    assert len(snapshot) == 17
    assert len(dynamic) == 27
    assert set(coverage) < set(snapshot) < set(dynamic)
    assert not any(
        token in name for name in dynamic for token in ("null_reason", "sample_count", "valid")
    )


def test_raw_feature_extraction_preserves_missingness_and_positive_area() -> None:
    table = pa.table(
        {
            "signal": pa.array([1.0, None, 3.0], type=pa.float64()),
            "binary": pa.array([True, False, True], type=pa.bool_()),
            "clipped_area_km2": pa.array([1.0, 2.0, 3.0], type=pa.float64()),
        }
    )
    matrix = extract_raw_feature_matrix(table, source_columns=("signal", "binary"))

    assert matrix.values.shape == (3, 2)
    assert matrix.missing.tolist() == [[False, False], [True, False], [False, False]]
    assert np.isnan(matrix.values[1, 0])
    assert matrix.values.flags.writeable is False


def test_arrow_identity_hash_includes_the_null_bitmap() -> None:
    accepted = pa.table(
        {
            "value": pa.array([1.0, None], type=pa.float64()),
            "valid": pa.array([True, False], type=pa.bool_()),
        }
    )
    same = pa.table(
        {
            "value": pa.array([1.0, None], type=pa.float64()),
            "valid": pa.array([True, False], type=pa.bool_()),
        }
    )
    changed_bitmap = pa.table(
        {
            "value": pa.array([1.0, 0.0], type=pa.float64()),
            "valid": pa.array([True, False], type=pa.bool_()),
        }
    )

    identity = assert_selected_columns_exact(
        accepted,
        same,
        columns=("value", "valid"),
    )
    assert identity == selected_table_identity_sha256(accepted, ("value", "valid"))
    with pytest.raises(ValueError, match="identity reconstruction"):
        assert_selected_columns_exact(
            accepted,
            changed_bitmap,
            columns=("value", "valid"),
        )


def test_grid_builder_rejects_unfrozen_resolution() -> None:
    with pytest.raises(ValueError, match="exactly 50, 25, or 12.5"):
        build_stage4_integration_grid(box(105.0, 34.0, 105.05, 34.05), cell_size_km=10.0)
