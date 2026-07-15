from __future__ import annotations

import ast
import importlib.util
import os
import stat
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest

import seismoflux.anomaly_increment.formal_production as formal_production
import seismoflux.anomaly_increment.immutable_file as immutable_file_module
import seismoflux.anomaly_increment.qualification as qualification_module
from seismoflux.anomaly_increment.authorization import (
    STAGE4_ATTEMPT_LEDGER_RELATIVE_PATH,
    STAGE4_TARGET_READ_LEDGER_RELATIVE_PATH,
    verify_stage4_target_readiness,
)
from seismoflux.anomaly_increment.config import load_stage4_protocol_bundle
from seismoflux.anomaly_increment.formal_preflight import (
    FORMAL_PREFLIGHT_RECEIPT_PATH,
    resolve_score_blind_input_path,
)
from seismoflux.anomaly_increment.immutable_file import UnsafeImmutableFileError
from seismoflux.anomaly_increment.score_blind_path import (
    ScoreBlindPathError,
    require_score_blind_project_path,
)

ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str) -> ModuleType:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(f"score_blind_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _protocol() -> dict[str, object]:
    return {
        "protocol_version": "0.4.0",
        "inputs": {
            "earthquake_target": {
                "path": "synthetic/target.bin",
            }
        },
        "freeze": {
            "execution_revision": "r1",
            "corrects_execution_revision": "r0",
            "execution_revision_document": "docs/anomaly_increment_protocol_r1.md",
            "readiness_incident_document": ("docs/phase4_scoring_readiness_incident_r0.md"),
            "pre_score_tag": "v0.3.0-anomaly-increment-protocol-r1",
            "results_tag": "v0.3.0-anomaly-increment-r1",
            "protocol_tag_authorizes_only_score_free_implementation": True,
            "scoring_code_freeze": {
                "expected_tag": "v0.3.0-anomaly-increment-scoring-code-r1",
                "required_seal_path": ("data/manifests/anomaly_increment_r1_scoring_seal.json"),
                "formal_preflight_receipt_path": (
                    "data/interim/stage4/anomaly_increment_r1/formal_preflight_receipt.json"
                ),
                "qualification_path": (
                    "data/interim/stage4/anomaly_increment_r1/scoring_qualification.json"
                ),
                "stage4_junit_path": (
                    "data/interim/stage4/anomaly_increment_r1/qualification_stage4.junit.xml"
                ),
                "full_non_target_junit_path": (
                    "data/interim/stage4/anomaly_increment_r1/"
                    "qualification_full_non_target.junit.xml"
                ),
                "formal_attempt_ledger_path": (
                    "data/manifests/anomaly_increment_r1_attempt_ledger.json"
                ),
                "target_read_ledger_path": (
                    "data/manifests/anomaly_increment_r1_target_read_ledger.json"
                ),
                "checkpoint_root": ("data/interim/stage4/anomaly_increment_r1/checkpoints"),
                "selected_table_logical_identity": {
                    "method_id": "arrow_ipc_selected_table_logical_identity_r1",
                    "sha256_domain_separator_ascii": (
                        "seismoflux.selected-table-logical-identity.r1"
                    ),
                    "sha256_domain_separator_nul_terminated": True,
                    "top_level_schema_metadata": "excluded",
                    "field_name_order_type_nullability_and_metadata": "preserved_exactly",
                    "null_payload": "canonical_type_zero",
                    "validity_bitmap": "preserved_with_length_padding_zeroed",
                    "boolean_value_padding": "zeroed_outside_logical_length",
                    "chunking_and_slice_offsets": "canonicalized",
                    "field_metadata_key_order": "bytewise_ascending",
                    "supported_types": [
                        "boolean",
                        "signed_integer",
                        "unsigned_integer",
                        "floating_point",
                        "timestamp",
                        "utf8_string",
                    ],
                    "valid_payload_bits": "preserved_exactly",
                    "unsupported_types": "fail_closed",
                },
            },
        },
    }


def test_stage4_pretarget_cli_defaults_use_only_the_r1_execution_namespace() -> None:
    qualification = _load_script("build_stage4_scoring_qualification.py")
    seal = _load_script("build_stage4_scoring_seal.py")
    preflight = _load_script("build_stage4_formal_preflight.py")

    assert qualification.PROTOCOL_PATH == ROOT / "configs" / "anomaly_increment_r1.yaml"
    assert seal.PROTOCOL_PATH == ROOT / "configs" / "anomaly_increment_r1.yaml"
    assert (
        ROOT.joinpath(*FORMAL_PREFLIGHT_RECEIPT_PATH.parts) == qualification.PREFLIGHT_RECEIPT_PATH
    )
    assert ROOT.joinpath(*FORMAL_PREFLIGHT_RECEIPT_PATH.parts) == seal.PREFLIGHT_RECEIPT_PATH
    assert ROOT.joinpath(*FORMAL_PREFLIGHT_RECEIPT_PATH.parts) == preflight.OUTPUT
    for path in (
        qualification.QUALIFICATION_PATH,
        qualification.STAGE4_JUNIT_PATH,
        qualification.FULL_JUNIT_PATH,
        seal.QUALIFICATION_PATH,
        seal.ATTEMPT_LEDGER_PATH,
        seal.TARGET_READ_LEDGER_PATH,
        preflight.OUTPUT,
    ):
        assert "anomaly_increment_r1" in path.as_posix()


def _forbid_target_lstat(
    monkeypatch: pytest.MonkeyPatch,
    target: Path,
) -> None:
    original = os.lstat
    target_text = os.path.normcase(os.path.abspath(os.fspath(target)))

    def tracked(
        path: os.PathLike[str] | str,
        *,
        dir_fd: int | None = None,
    ) -> os.stat_result:
        observed = os.path.normcase(os.path.abspath(os.fspath(path)))
        if observed == target_text:
            raise AssertionError("the forbidden target path was lstat'ed before rejection")
        if dir_fd is None:
            return original(path)
        return original(path, dir_fd=dir_fd)

    monkeypatch.setattr(os, "lstat", tracked)


def test_exact_target_is_rejected_lexically_before_target_filesystem_contact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "synthetic" / "target.bin"
    _forbid_target_lstat(monkeypatch, target)
    with pytest.raises(ScoreBlindPathError, match="frozen earthquake target"):
        require_score_blind_project_path(
            tmp_path,
            _protocol(),
            target,
            label="synthetic qualification input",
        )


def test_symbolic_link_alias_is_rejected_without_following_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path.resolve()
    target = root / "synthetic" / "target.bin"
    target.parent.mkdir()
    target.write_bytes(b"synthetic target")
    alias = root / "safe" / "alias.xml"
    alias.parent.mkdir()
    try:
        alias.symlink_to(target)
    except OSError:
        original = os.lstat

        def symlink_lstat(
            path: os.PathLike[str] | str,
            *,
            dir_fd: int | None = None,
        ) -> Any:
            if os.path.abspath(os.fspath(path)) == os.path.abspath(os.fspath(alias)):
                return SimpleNamespace(st_file_attributes=0, st_mode=stat.S_IFLNK, st_nlink=1)
            if dir_fd is None:
                return original(path)
            return original(path, dir_fd=dir_fd)

        monkeypatch.setattr(os, "lstat", symlink_lstat)
    with pytest.raises(ScoreBlindPathError, match="symbolic link or junction"):
        require_score_blind_project_path(root, _protocol(), alias, label="alias")


def test_reparse_point_is_rejected_without_following_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path.resolve()
    alias = root / "safe" / "junction" / "input.xml"
    original = os.lstat
    junction = root / "safe" / "junction"
    (root / "safe").mkdir()

    def reparse_lstat(
        path: os.PathLike[str] | str,
        *,
        dir_fd: int | None = None,
    ) -> Any:
        if os.path.abspath(os.fspath(path)) == os.path.abspath(os.fspath(junction)):
            return SimpleNamespace(
                st_file_attributes=0x400,
                st_mode=stat.S_IFDIR,
                st_nlink=1,
            )
        if dir_fd is None:
            return original(path)
        return original(path, dir_fd=dir_fd)

    monkeypatch.setattr(os, "lstat", reparse_lstat)
    with pytest.raises(ScoreBlindPathError, match="symbolic link or junction"):
        require_score_blind_project_path(root, _protocol(), alias, label="junction alias")


def test_hard_link_alias_and_project_escape_are_rejected(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    target = root / "synthetic" / "target.bin"
    target.parent.mkdir()
    target.write_bytes(b"synthetic target")
    alias = root / "safe" / "alias.xml"
    alias.parent.mkdir()
    os.link(target, alias)
    with pytest.raises(ScoreBlindPathError, match="multiply-linked"):
        require_score_blind_project_path(root, _protocol(), alias, label="hard-link alias")
    with pytest.raises(ScoreBlindPathError, match="inside the project"):
        require_score_blind_project_path(
            root,
            _protocol(),
            root.parent / "outside.xml",
            label="outside",
        )


def test_qualification_internal_path_rejects_hardlink_before_hash_or_open(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    target = root / "synthetic" / "target.bin"
    target.parent.mkdir()
    target.write_bytes(b"synthetic target")
    alias = root / "data" / "input.bin"
    alias.parent.mkdir()
    os.link(target, alias)
    with pytest.raises(Exception, match="unsafe score-blind path"):
        qualification_module._project_path(
            root,
            _protocol(),
            "data/input.bin",
            label="synthetic internal input",
        )


def test_protocol_loader_rejects_hardlink_before_parsing_any_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path.resolve()
    target = root / "synthetic-target.bin"
    payload = b"target bytes are not a stage-4 protocol"
    target.write_bytes(payload)
    protocol_path = root / "configs" / "anomaly_increment_r1.yaml"
    protocol_path.parent.mkdir()
    os.link(target, protocol_path)

    def forbidden_open(path: Path) -> int:
        raise AssertionError(f"unsafe protocol alias reached open: {path}")

    monkeypatch.setattr(immutable_file_module, "_open_no_follow", forbidden_open)
    with pytest.raises(UnsafeImmutableFileError, match="single-link"):
        load_stage4_protocol_bundle(root)
    assert target.read_bytes() == payload


def test_scoring_seal_cli_protocol_loader_rejects_hardlink_before_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script("build_stage4_scoring_seal.py")
    target = tmp_path / "synthetic-target.bin"
    payload = b"target bytes must not enter the seal protocol loader"
    target.write_bytes(payload)
    protocol_path = tmp_path / "configs" / "anomaly_increment_r1.yaml"
    protocol_path.parent.mkdir()
    os.link(target, protocol_path)

    def forbidden_open(path: Path) -> int:
        raise AssertionError(f"unsafe seal protocol alias reached open: {path}")

    monkeypatch.setattr(immutable_file_module, "_open_no_follow", forbidden_open)
    with pytest.raises(ValueError, match="protocol cannot be read safely"):
        module._load_protocol(protocol_path)
    assert target.read_bytes() == payload


def test_scoring_qualification_protocol_loader_rejects_hardlink_before_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script("build_stage4_scoring_qualification.py")
    target = tmp_path / "synthetic-target.bin"
    payload = b"target bytes must not enter the qualification protocol loader"
    target.write_bytes(payload)
    protocol_path = tmp_path / "configs" / "anomaly_increment_r1.yaml"
    protocol_path.parent.mkdir()
    os.link(target, protocol_path)

    def forbidden_open(path: Path) -> int:
        raise AssertionError(f"unsafe qualification protocol alias reached open: {path}")

    monkeypatch.setattr(immutable_file_module, "_open_no_follow", forbidden_open)
    with pytest.raises(ValueError, match="protocol cannot be read safely"):
        module._load_protocol(protocol_path)
    assert target.read_bytes() == payload


@pytest.mark.parametrize("junit_label", ("stage4 JUnit", "full JUnit"))
@pytest.mark.parametrize("alias_kind", ("hardlink", "reparse"))
def test_qualification_junit_swap_after_guard_fails_before_parse_or_target_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    junit_label: str,
    alias_kind: str,
) -> None:
    module = _load_script("build_stage4_scoring_qualification.py")
    protocol = _protocol()
    r1_root = tmp_path / "data" / "interim" / "stage4" / "anomaly_increment_r1"
    stage4_junit = r1_root / "qualification_stage4.junit.xml"
    full_junit = r1_root / "qualification_full_non_target.junit.xml"
    preflight = r1_root / "formal_preflight_receipt.json"
    logical_replay = r1_root / "logical_identity_worker_replay.json"
    stage4_junit.parent.mkdir(parents=True)
    passing_xml = (
        b'<testsuite name="pytest" errors="0" failures="0" skipped="0" tests="1">'
        b'<testcase classname="tests.unit.safe" name="test_safe" />'
        b"</testsuite>"
    )
    stage4_junit.write_bytes(passing_xml)
    full_junit.write_bytes(passing_xml)
    preflight.write_bytes(b"{}")
    selected = stage4_junit if junit_label == "stage4 JUnit" else full_junit
    target = tmp_path / "synthetic" / "target.bin"
    target.parent.mkdir()
    target_payload = b"target bytes must never enter the JUnit parser"
    target.write_bytes(target_payload)

    real_guard = module.require_score_blind_project_path
    real_lstat = os.lstat
    selected_text = os.path.normcase(os.path.abspath(os.fspath(selected)))
    reparse_armed = False

    def swapping_guard(
        root: Path,
        protocol_value: dict[str, object],
        candidate: Path,
        *,
        label: str,
    ) -> Path:
        nonlocal reparse_armed
        safe = cast(Path, real_guard(root, protocol_value, candidate, label=label))
        if label == junit_label:
            if alias_kind == "hardlink":
                safe.unlink()
                os.link(target, safe)
            else:
                reparse_armed = True
        return safe

    def simulated_reparse_lstat(
        path: os.PathLike[str] | str,
        *,
        dir_fd: int | None = None,
    ) -> Any:
        observed = os.path.normcase(os.path.abspath(os.fspath(path)))
        if reparse_armed and observed == selected_text:
            return SimpleNamespace(
                st_file_attributes=0x0400,
                st_mode=stat.S_IFLNK,
                st_nlink=1,
            )
        if dir_fd is None:
            return real_lstat(path)
        return real_lstat(path, dir_fd=dir_fd)

    monkeypatch.setattr(module, "require_score_blind_project_path", swapping_guard)
    if alias_kind == "reparse":
        monkeypatch.setattr(os, "lstat", simulated_reparse_lstat)

    real_parse = module.parse_pytest_junit_evidence
    parsed_payloads: list[bytes] = []

    def guarded_parse(payload: bytes) -> Any:
        assert payload != target_payload
        parsed_payloads.append(payload)
        return real_parse(payload)

    monkeypatch.setattr(module, "parse_pytest_junit_evidence", guarded_parse)
    original_open = immutable_file_module._open_no_follow

    def tracked_open(path: Path) -> int:
        if os.path.normcase(os.path.abspath(os.fspath(path))) == selected_text:
            raise AssertionError("swapped JUnit target alias reached open")
        return original_open(path)

    monkeypatch.setattr(immutable_file_module, "_open_no_follow", tracked_open)
    with pytest.raises(ValueError, match="JUnit evidence cannot be read safely"):
        module._build(
            tmp_path,
            protocol,
            stage4_junit_path=stage4_junit,
            full_junit_path=full_junit,
            preflight_receipt_path=preflight,
            logical_replay_audit_path=logical_replay,
            git_runner=cast(Any, None),
        )
    assert target.read_bytes() == target_payload
    assert parsed_payloads == ([] if junit_label == "stage4 JUnit" else [passing_xml])


def test_frozen_formal_ingress_has_no_unguarded_path_byte_readers() -> None:
    relative_sources = (
        "src/seismoflux/anomaly_increment/config.py",
        "src/seismoflux/anomaly_increment/qualification.py",
        "src/seismoflux/anomaly_increment/authorization.py",
        "src/seismoflux/anomaly_increment/attempt_ledger.py",
        "src/seismoflux/anomaly_increment/formal_preflight.py",
        "src/seismoflux/anomaly_increment/formal_production.py",
        "src/seismoflux/anomaly_increment/formal_run.py",
        "src/seismoflux/anomaly_increment/placebo_runtime.py",
        "src/seismoflux/anomaly_increment/formal_publication.py",
        "scripts/build_stage4_scoring_seal.py",
        "scripts/run_stage4_formal.py",
    )
    violations: list[str] = []
    for relative in relative_sources:
        source_path = ROOT / relative
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=relative)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr in {"read_text", "read_bytes"}:
                violations.append(f"{relative}:{node.lineno}:{node.func.attr}")
            if node.func.attr == "open" and not (
                isinstance(node.func.value, ast.Name) and node.func.value.id == "os"
            ):
                violations.append(f"{relative}:{node.lineno}:open")
    assert violations == []

    target_access = ROOT / "src/seismoflux/anomaly_increment/target_access.py"
    target_tree = ast.parse(
        target_access.read_text(encoding="utf-8"),
        filename="target_access.py",
    )
    target_opens = [
        node
        for node in ast.walk(target_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "open"
    ]
    assert len(target_opens) == 1


def test_formal_preflight_allowlisted_input_rejects_target_hardlink_alias(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    target = root / "synthetic" / "target.bin"
    target.parent.mkdir()
    target.write_bytes(b"synthetic target")
    alias = root / "data" / "stage3.parquet"
    alias.parent.mkdir()
    os.link(target, alias)
    protocol = cast(
        Any,
        SimpleNamespace(
            repository_root=root,
            design_sha256="a" * 64,
            random_input_seal_sha256="b" * 64,
            protocol={
                "inputs": {
                    "earthquake_target": {"path": "synthetic/target.bin"},
                    "stage3_feature_store": {
                        "path": "data/stage3.parquet",
                        "sha256": "c" * 64,
                    },
                }
            },
        ),
    )
    evidence = cast(
        Any,
        SimpleNamespace(
            observed_project_input_hashes=(("stage3_feature_store", "c" * 64),),
            protocol_design_sha256="a" * 64,
            random_input_seal_sha256="b" * 64,
        ),
    )
    with pytest.raises(ScoreBlindPathError, match="multiply-linked"):
        resolve_score_blind_input_path(
            protocol,
            evidence,
            input_id="stage3_feature_store",
        )


@pytest.mark.parametrize("artifact", ("scoring_seal", "preflight_receipt"))
def test_production_readiness_rejects_internal_hard_link_alias_before_loading(
    tmp_path: Path,
    artifact: str,
) -> None:
    root = tmp_path.resolve()
    target = root / "synthetic" / "target.bin"
    target.parent.mkdir()
    target.write_bytes(b"synthetic target that must not be loaded")
    if artifact == "scoring_seal":
        alias = root / "data" / "manifests" / "anomaly_increment_r1_scoring_seal.json"
    else:
        alias = root.joinpath(*FORMAL_PREFLIGHT_RECEIPT_PATH.parts)
    alias.parent.mkdir(parents=True, exist_ok=True)
    os.link(target, alias)
    protocol = cast(
        Any,
        SimpleNamespace(
            repository_root=root,
            protocol=_protocol(),
        ),
    )
    with pytest.raises(ScoreBlindPathError, match="multiply-linked"):
        if artifact == "scoring_seal":
            formal_production._scoring_seal_path(protocol)
        else:
            formal_production._formal_preflight_receipt_path(protocol)


@pytest.mark.parametrize(
    ("script_name", "call_kind"),
    (
        ("build_stage4_scoring_qualification.py", "qualification_input"),
        ("build_stage4_scoring_qualification.py", "qualification_output"),
        ("build_stage4_scoring_seal.py", "seal_qualification_input"),
        ("build_stage4_scoring_seal.py", "seal_preflight_input"),
        ("build_stage4_scoring_seal.py", "seal_attempt_ledger"),
        ("build_stage4_scoring_seal.py", "seal_target_read_ledger"),
        ("build_stage4_scoring_seal.py", "seal_output"),
    ),
)
def test_target_path_override_is_rejected_by_every_pretarget_cli_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    script_name: str,
    call_kind: str,
) -> None:
    module = _load_script(script_name)
    protocol = _protocol()
    target = tmp_path / "synthetic" / "target.bin"
    if call_kind == "seal_output":
        protocol["freeze"] = {
            "scoring_code_freeze": {
                "required_seal_path": "synthetic/target.bin",
            }
        }
    _forbid_target_lstat(monkeypatch, target)
    safe_stage4 = tmp_path / "data" / "interim" / "stage4.xml"
    safe_full = tmp_path / "data" / "interim" / "full.xml"
    safe_preflight = tmp_path.joinpath(*FORMAL_PREFLIGHT_RECEIPT_PATH.parts)
    safe_qualification = tmp_path / "data" / "interim" / "qualification.json"
    safe_logical_replay = tmp_path / "data" / "interim" / "logical-replay.json"

    with pytest.raises(ScoreBlindPathError, match="frozen earthquake target"):
        if call_kind == "qualification_input":
            module._build(
                tmp_path,
                protocol,
                stage4_junit_path=target,
                full_junit_path=safe_full,
                preflight_receipt_path=safe_preflight,
                logical_replay_audit_path=safe_logical_replay,
                git_runner=cast(Any, None),
            )
        elif call_kind == "qualification_output":
            module.generate(
                tmp_path,
                protocol,
                stage4_junit_path=safe_stage4,
                full_junit_path=safe_full,
                preflight_receipt_path=safe_preflight,
                qualification_path=target,
                logical_replay_audit_path=safe_logical_replay,
                git_runner=cast(Any, None),
            )
        else:
            module.generate(
                tmp_path,
                protocol,
                qualification_path=(
                    target if call_kind == "seal_qualification_input" else safe_qualification
                ),
                preflight_receipt_path=(
                    target if call_kind == "seal_preflight_input" else safe_preflight
                ),
                attempt_ledger_path=(
                    target
                    if call_kind == "seal_attempt_ledger"
                    else tmp_path.joinpath(*Path(STAGE4_ATTEMPT_LEDGER_RELATIVE_PATH).parts)
                ),
                target_read_ledger_path=(
                    target
                    if call_kind == "seal_target_read_ledger"
                    else tmp_path.joinpath(*Path(STAGE4_TARGET_READ_LEDGER_RELATIVE_PATH).parts)
                ),
                repository_adapter=cast(Any, None),
            )


@pytest.mark.parametrize(
    "override",
    ("scoring_seal", "attempt_ledger", "target_read_ledger"),
)
def test_read_only_formal_proof_rejects_target_path_arguments_before_lstat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    override: str,
) -> None:
    target = tmp_path / "synthetic" / "target.bin"
    _forbid_target_lstat(monkeypatch, target)
    safe_seal = tmp_path / "data" / "manifests" / "anomaly_increment_r1_scoring_seal.json"
    safe_attempt = tmp_path.joinpath(*Path(STAGE4_ATTEMPT_LEDGER_RELATIVE_PATH).parts)
    safe_target_ledger = tmp_path.joinpath(*Path(STAGE4_TARGET_READ_LEDGER_RELATIVE_PATH).parts)
    with pytest.raises(ScoreBlindPathError, match="frozen earthquake target"):
        verify_stage4_target_readiness(
            tmp_path,
            _protocol(),
            scoring_seal_path=target if override == "scoring_seal" else safe_seal,
            attempt_ledger_path=(target if override == "attempt_ledger" else safe_attempt),
            target_read_ledger_path=(
                target if override == "target_read_ledger" else safe_target_ledger
            ),
        )
