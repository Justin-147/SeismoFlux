from __future__ import annotations

import copy
import json
from collections import deque
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pandas as pd
import pytest
import yaml

import seismoflux.background.execution as execution_module
from seismoflux.background.artifacts import content_address_id
from seismoflux.background.config import (
    BackgroundConfig,
    BackgroundLocalSupportConfig,
    load_background_protocol,
)
from seismoflux.background.execution import (
    CommandResult,
    ExecutionSeal,
    ExecutionSealError,
    RepositoryIdentity,
    RepositoryIdentityError,
    build_address_inputs,
    build_background_plan,
    create_execution_seal,
    require_execution_seal_unchanged,
    require_repository_identity,
)
from seismoflux.config import SeismoFluxConfig

HEAD = "a" * 40
TAG_COMMIT = "1" * 40
UPSTREAM = "origin/main"
FREEZE_TAG = "v0.2.0-background-protocol"
ROOT = Path("synthetic-project")

HEAD_COMMAND = ("git", "rev-parse", "--verify", "HEAD^{commit}")
STATUS_COMMAND = ("git", "status", "--porcelain=v1", "--untracked-files=all")
TAG_COMMAND = (
    "git",
    "rev-parse",
    "--verify",
    f"refs/tags/{FREEZE_TAG}^{{commit}}",
)
ANCESTOR_COMMAND = ("git", "merge-base", "--is-ancestor", TAG_COMMIT, HEAD)
BRANCH_COMMAND = ("git", "symbolic-ref", "--quiet", "--short", "HEAD")
UPSTREAM_NAME_COMMAND = (
    "git",
    "rev-parse",
    "--abbrev-ref",
    "--symbolic-full-name",
    "@{upstream}",
)
UPSTREAM_COMMIT_COMMAND = (
    "git",
    "rev-parse",
    "--verify",
    "@{upstream}^{commit}",
)


@pytest.fixture(scope="module")
def background() -> BackgroundConfig:
    raw = yaml.safe_load(Path("configs/background.yaml").read_text(encoding="utf-8"))
    return BackgroundConfig.model_validate(raw)


@pytest.fixture(scope="module")
def local_support_background() -> BackgroundLocalSupportConfig:
    config = load_background_protocol(Path("configs/background_local_support.yaml"))
    assert isinstance(config, BackgroundLocalSupportConfig)
    return config


@pytest.fixture(scope="module")
def project() -> SeismoFluxConfig:
    raw = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))
    return SeismoFluxConfig.model_validate(raw)


class FakeRunner:
    def __init__(
        self,
        responses: dict[tuple[str, ...], CommandResult | list[CommandResult]],
    ) -> None:
        self._responses: dict[tuple[str, ...], CommandResult | deque[CommandResult]] = {
            command: deque(value) if isinstance(value, list) else value
            for command, value in responses.items()
        }
        self.calls: list[tuple[tuple[str, ...], Path]] = []

    def __call__(self, command: tuple[str, ...], cwd: Path) -> CommandResult:
        self.calls.append((command, cwd))
        response = self._responses[command]
        if isinstance(response, deque):
            if len(response) > 1:
                return response.popleft()
            return response[0]
        return response


def _responses() -> dict[tuple[str, ...], CommandResult | list[CommandResult]]:
    return {
        ("git", "rev-parse", "--is-inside-work-tree"): CommandResult(0, "true\n"),
        HEAD_COMMAND: CommandResult(0, f"{HEAD}\n"),
        STATUS_COMMAND: CommandResult(0, ""),
        TAG_COMMAND: CommandResult(0, f"{TAG_COMMIT}\n"),
        ANCESTOR_COMMAND: CommandResult(0),
        BRANCH_COMMAND: CommandResult(0, "main\n"),
        UPSTREAM_NAME_COMMAND: CommandResult(0, f"{UPSTREAM}\n"),
        UPSTREAM_COMMIT_COMMAND: CommandResult(0, f"{HEAD}\n"),
    }


def _responses_for_freeze_tag(
    freeze_tag: str,
) -> dict[tuple[str, ...], CommandResult | list[CommandResult]]:
    responses = _responses()
    if freeze_tag != FREEZE_TAG:
        tag_result = responses.pop(TAG_COMMAND)
        responses[("git", "rev-parse", "--verify", f"refs/tags/{freeze_tag}^{{commit}}")] = (
            tag_result
        )
    return responses


def _ready_identity(
    *,
    commit: str = HEAD,
    freeze_tag: str = FREEZE_TAG,
) -> RepositoryIdentity:
    return RepositoryIdentity(
        code_commit=commit,
        branch="main",
        upstream=UPSTREAM,
        upstream_commit=commit,
        freeze_tag=freeze_tag,
        freeze_tag_commit=TAG_COMMIT,
        git_available=True,
        worktree_clean=True,
        tag_is_ancestor=True,
        upstream_matches_head=True,
    )


def _sealed_input_hashes(
    tmp_path: Path,
    background: BackgroundConfig,
) -> dict[Path, str]:
    hashes: dict[Path, str] = {}
    for reference, expected in execution_module._execution_input_references(background).values():
        path = tmp_path.joinpath(*reference.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(reference.encode("utf-8"))
        hashes[path.resolve()] = expected
    return hashes


def test_repository_identity_requires_clean_pushed_tagged_commit(tmp_path: Path) -> None:
    runner = FakeRunner(_responses())

    identity = require_repository_identity(tmp_path, freeze_tag=FREEZE_TAG, runner=runner)

    assert identity.ready is True
    assert identity.code_commit == HEAD
    assert identity.freeze_tag_commit == TAG_COMMIT
    assert identity.branch == "main"
    assert identity.upstream == UPSTREAM
    assert [command for command, _ in runner.calls].count(HEAD_COMMAND) == 2
    assert [command for command, _ in runner.calls].count(STATUS_COMMAND) == 2
    assert all(cwd == tmp_path.resolve() for _, cwd in runner.calls)


def test_repository_identity_rejects_unavailable_git(tmp_path: Path) -> None:
    def unavailable(_: tuple[str, ...], __: Path) -> CommandResult:
        raise OSError("git missing")

    with pytest.raises(RepositoryIdentityError, match="Git is unavailable"):
        require_repository_identity(tmp_path, freeze_tag=FREEZE_TAG, runner=unavailable)


@pytest.mark.parametrize("dirty", [" M tracked.py\n", "?? untracked.py\n"])
def test_repository_identity_rejects_tracked_and_untracked_changes(
    tmp_path: Path,
    dirty: str,
) -> None:
    responses = _responses()
    responses[STATUS_COMMAND] = CommandResult(0, dirty)

    with pytest.raises(RepositoryIdentityError, match="must be clean"):
        require_repository_identity(
            tmp_path,
            freeze_tag=FREEZE_TAG,
            runner=FakeRunner(responses),
        )


def test_repository_identity_rejects_non_lowercase_head(tmp_path: Path) -> None:
    responses = _responses()
    responses[HEAD_COMMAND] = CommandResult(0, f"{HEAD.upper()}\n")

    with pytest.raises(RepositoryIdentityError, match="lowercase Git object ID"):
        require_repository_identity(
            tmp_path,
            freeze_tag=FREEZE_TAG,
            runner=FakeRunner(responses),
        )


@pytest.mark.parametrize(
    ("command", "result", "message"),
    (
        (TAG_COMMAND, CommandResult(128, stderr="missing tag"), "resolve protocol tag"),
        (ANCESTOR_COMMAND, CommandResult(1), "ancestor of HEAD"),
        (BRANCH_COMMAND, CommandResult(1), "current branch"),
        (UPSTREAM_NAME_COMMAND, CommandResult(128), "branch upstream"),
    ),
)
def test_repository_identity_rejects_missing_tag_ancestry_branch_or_upstream(
    tmp_path: Path,
    command: tuple[str, ...],
    result: CommandResult,
    message: str,
) -> None:
    responses = _responses()
    responses[command] = result

    with pytest.raises(RepositoryIdentityError, match=message):
        require_repository_identity(
            tmp_path,
            freeze_tag=FREEZE_TAG,
            runner=FakeRunner(responses),
        )


def test_repository_identity_rejects_unpushed_head(tmp_path: Path) -> None:
    responses = _responses()
    responses[UPSTREAM_COMMIT_COMMAND] = CommandResult(0, f"{'b' * 40}\n")

    with pytest.raises(RepositoryIdentityError, match="upstream commit must exactly equal HEAD"):
        require_repository_identity(
            tmp_path,
            freeze_tag=FREEZE_TAG,
            runner=FakeRunner(responses),
        )


def test_repository_identity_detects_head_race(tmp_path: Path) -> None:
    responses = _responses()
    responses[HEAD_COMMAND] = [
        CommandResult(0, f"{HEAD}\n"),
        CommandResult(0, f"{'b' * 40}\n"),
    ]

    with pytest.raises(RepositoryIdentityError, match="HEAD changed"):
        require_repository_identity(
            tmp_path,
            freeze_tag=FREEZE_TAG,
            runner=FakeRunner(responses),
        )


def test_execution_seal_binds_repository_protocol_and_all_seven_inputs(
    tmp_path: Path,
    background: BackgroundConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = _sealed_input_hashes(tmp_path, background)
    monkeypatch.setattr(execution_module, "sha256_file", lambda path: observed[path.resolve()])

    seal = create_execution_seal(
        tmp_path,
        background,
        runner=FakeRunner(_responses()),
    )
    rechecked = require_execution_seal_unchanged(
        tmp_path,
        background,
        seal,
        runner=FakeRunner(_responses()),
    )

    assert rechecked == seal
    assert rechecked.seal_id == seal.seal_id
    assert tuple(rechecked.input_hash_mapping()) == (
        "data_catalog",
        "earthquake_dataset",
        "environment_lock",
        "issue_manifest",
        "oracle_metadata",
        "production_fixture",
        "study_area",
    )


def test_v020_execution_seal_id_is_unchanged() -> None:
    seal = ExecutionSeal(
        repository=_ready_identity(),
        protocol_sha256="9" * 64,
        input_hashes=tuple(
            (key, f"{index:064x}")
            for index, key in enumerate(execution_module._EXECUTION_INPUT_KEYS, start=1)
        ),
    )

    assert seal.seal_id == ("97cbade78428312c682bb8acb8eedfcae25de27bc1cfb26806957c5924d02b79")


def test_execution_seal_binds_sorted_eighth_support_manifest_input(
    tmp_path: Path,
    local_support_background: BackgroundLocalSupportConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = _sealed_input_hashes(tmp_path, local_support_background)
    monkeypatch.setattr(execution_module, "sha256_file", lambda path: observed[path.resolve()])

    seal = create_execution_seal(
        tmp_path,
        local_support_background,
        runner=FakeRunner(_responses_for_freeze_tag(local_support_background.freeze_tag)),
    )

    assert tuple(seal.input_hash_mapping()) == (
        "data_catalog",
        "earthquake_dataset",
        "environment_lock",
        "issue_manifest",
        "oracle_metadata",
        "production_fixture",
        "study_area",
        "support_manifest",
    )

    support_path = tmp_path.joinpath(*local_support_background.inputs.support_manifest.split("/"))
    observed[support_path.resolve()] = "7" * 64
    with pytest.raises(ExecutionSealError, match="support_manifest"):
        require_execution_seal_unchanged(
            tmp_path,
            local_support_background,
            seal,
            runner=FakeRunner(_responses_for_freeze_tag(local_support_background.freeze_tag)),
        )


def test_execution_seal_rejects_multiple_missing_versioned_input_keys(
    tmp_path: Path,
    local_support_background: BackgroundLocalSupportConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = _sealed_input_hashes(tmp_path, local_support_background)
    monkeypatch.setattr(execution_module, "sha256_file", lambda path: observed[path.resolve()])
    seal = create_execution_seal(
        tmp_path,
        local_support_background,
        runner=FakeRunner(_responses_for_freeze_tag(local_support_background.freeze_tag)),
    )
    incomplete = tuple(
        item for item in seal.input_hashes if item[0] not in {"study_area", "support_manifest"}
    )

    with pytest.raises(ValueError, match="complete sorted frozen input-hash schema"):
        replace(seal, input_hashes=incomplete)


@pytest.mark.parametrize(
    "missing_fields",
    (("support_manifest_sha256",), ("support_manifest", "support_manifest_sha256")),
)
def test_execution_rejects_missing_support_manifest_config_for_v021(
    missing_fields: tuple[str, ...],
) -> None:
    input_values = {
        "support_manifest": "data/manifests/background_local_support_manifest.json",
        "support_manifest_sha256": "8" * 64,
    }
    for field in missing_fields:
        del input_values[field]
    supported = SimpleNamespace(
        protocol_version="0.2.1",
        inputs=SimpleNamespace(**input_values),
    )

    with pytest.raises(ValueError, match="0.2.1 requires support_manifest"):
        execution_module._support_manifest_reference(supported)


def test_execution_dispatches_support_manifest_strictly_by_protocol_version() -> None:
    injected_inputs = SimpleNamespace(
        support_manifest="data/manifests/background_local_support_manifest.json",
        support_manifest_sha256="8" * 64,
    )
    v020_with_support = SimpleNamespace(protocol_version="0.2.0", inputs=injected_inputs)
    with pytest.raises(ValueError, match="0.2.0 forbids support_manifest"):
        execution_module._support_manifest_reference(v020_with_support)

    unknown = SimpleNamespace(protocol_version="0.2.2", inputs=injected_inputs)
    with pytest.raises(ValueError, match="unsupported background protocol_version"):
        execution_module._support_manifest_reference(unknown)


def test_execution_seal_recheck_rejects_any_changed_input_bytes(
    tmp_path: Path,
    background: BackgroundConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = _sealed_input_hashes(tmp_path, background)
    monkeypatch.setattr(execution_module, "sha256_file", lambda path: observed[path.resolve()])
    seal = create_execution_seal(
        tmp_path,
        background,
        runner=FakeRunner(_responses()),
    )
    earthquake_path = tmp_path.joinpath(*background.inputs.earthquake_dataset_path.split("/"))
    observed[earthquake_path.resolve()] = "f" * 64

    with pytest.raises(ExecutionSealError, match="earthquake_dataset"):
        require_execution_seal_unchanged(
            tmp_path,
            background,
            seal,
            runner=FakeRunner(_responses()),
        )


def test_background_plan_is_complete_score_free_and_machine_readable(
    background: BackgroundConfig,
    project: SeismoFluxConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_catalog_read(*_: object, **__: object) -> object:
        raise AssertionError("plan construction must not read earthquake rows")

    monkeypatch.setattr(pd, "read_parquet", forbidden_catalog_read)
    plan = build_background_plan(background, project, physical_core_probe=lambda: 8)

    assert tuple(item.snapshot_id for item in plan.snapshots) == (
        "fold_1",
        "fold_2",
        "fold_3",
        "fold_4",
        "final_validation",
    )
    assert tuple(item.optimizer_model_id for item in plan.snapshots) == (
        "etas/fold_1",
        "etas/fold_2",
        "etas/fold_3",
        "etas/fold_4",
        "etas/final_validation",
    )
    assert plan.models == ("uniform_poisson", "spatial_poisson", "etas")
    assert plan.bandwidth_candidates_km == (75.0, 100.0, 150.0, 200.0, 300.0)
    assert plan.grid_cells_km == (50.0, 25.0, 12.5)
    assert plan.horizons_days == (7, 30, 90, 180, 365)
    assert dict(plan.output_paths) == {
        "processed_root": "data/processed/stage2",
        "backtest_root": "outputs/backtests/background",
        "experiment_root": "outputs/experiments/background",
        "model_root": "models/registry/background",
        "registry": "data/manifests/background_model_registry.json",
        "report": "docs/background_baseline_report.md",
    }
    assert plan.resources.detected_physical_cores == 8
    assert plan.resources.reserve_physical_cores == 2
    assert plan.resources.effective_workers == 6
    assert plan.resources.nested_parallelism is False
    json.dumps(plan.to_manifest_details(), allow_nan=False)


def test_background_plan_uses_one_worker_when_physical_cores_are_unknown(
    background: BackgroundConfig,
    project: SeismoFluxConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def logical_core_count_is_forbidden() -> int:
        raise AssertionError("logical os.cpu_count must not be used")

    monkeypatch.setattr("os.cpu_count", logical_core_count_is_forbidden)
    plan = build_background_plan(background, project, physical_core_probe=lambda: None)

    assert plan.resources.detected_physical_cores is None
    assert plan.resources.effective_workers == 1


@pytest.mark.parametrize("detected", [True, 0, -1])
def test_background_plan_rejects_invalid_physical_core_probe(
    background: BackgroundConfig,
    project: SeismoFluxConfig,
    detected: int,
) -> None:
    def probe() -> int | None:
        return detected

    with pytest.raises(ValueError, match="positive integer or None"):
        build_background_plan(background, project, physical_core_probe=probe)


def test_background_plan_rejects_project_protocol_drift(
    background: BackgroundConfig,
    project: SeismoFluxConfig,
) -> None:
    changed_project = project.model_copy(
        update={"project": project.project.model_copy(update={"random_seed": 148})}
    )

    with pytest.raises(ValueError, match="root seed"):
        build_background_plan(background, changed_project, physical_core_probe=lambda: None)


def test_address_factory_uses_complete_frozen_semantic_identity(
    background: BackgroundConfig,
) -> None:
    parameters: dict[str, object] = {
        "model_id": "spatial_poisson",
        "bandwidth_km": 75.0,
        "snapshot_id": "final_validation",
    }

    address = build_address_inputs(
        background,
        _ready_identity(),
        parameters,
        uv_lock_sha256=background.inputs.environment_lock_sha256,
    )

    assert set(address) == {
        "protocol",
        "input_hashes",
        "model_parameters",
        "code_commit",
        "uv_lock_sha256",
    }
    assert address["protocol"] == background.model_dump(mode="python")
    assert address["input_hashes"] == {
        "environment_lock": background.inputs.environment_lock_sha256,
        "data_catalog": background.inputs.data_catalog_sha256,
        "earthquake_dataset": background.inputs.earthquake_dataset_sha256,
        "study_area": background.inputs.study_area_sha256,
        "issue_manifest": background.inputs.issue_manifest_sha256,
        "production_fixture": background.numerical_regression.production_fixture_sha256,
        "oracle_metadata": background.numerical_regression.oracle_metadata_sha256,
    }
    assert address["code_commit"] == HEAD
    assert content_address_id(address) == "790ad470df5eb797"

    parameters["bandwidth_km"] = 300.0
    assert cast(dict[str, object], address["model_parameters"])["bandwidth_km"] == 75.0


def test_address_factory_includes_versioned_support_manifest_hash(
    local_support_background: BackgroundLocalSupportConfig,
) -> None:
    address = build_address_inputs(
        local_support_background,
        _ready_identity(freeze_tag=local_support_background.freeze_tag),
        {"model_id": "spatial_poisson", "snapshot_id": "fold_1"},
        uv_lock_sha256=local_support_background.inputs.environment_lock_sha256,
    )

    assert address["protocol"] == local_support_background.model_dump(mode="python")
    input_hashes = cast(dict[str, str], address["input_hashes"])
    assert tuple(sorted(input_hashes)) == (
        "data_catalog",
        "earthquake_dataset",
        "environment_lock",
        "issue_manifest",
        "oracle_metadata",
        "production_fixture",
        "study_area",
        "support_manifest",
    )
    assert (
        input_hashes["support_manifest"] == local_support_background.inputs.support_manifest_sha256
    )

    changed_address = copy.deepcopy(address)
    cast(dict[str, str], changed_address["input_hashes"])["support_manifest"] = "7" * 64
    changed_protocol = cast(dict[str, object], changed_address["protocol"])
    cast(dict[str, object], changed_protocol["inputs"])["support_manifest_sha256"] = "7" * 64
    assert content_address_id(changed_address) != content_address_id(address)


def test_address_changes_for_parameters_every_input_hash_and_commit(
    background: BackgroundConfig,
) -> None:
    lock_hash = background.inputs.environment_lock_sha256
    base = build_address_inputs(
        background,
        _ready_identity(),
        {"model_id": "spatial_poisson", "bandwidth_km": 75.0},
        uv_lock_sha256=lock_hash,
    )
    base_id = content_address_id(base)

    changed_parameters = build_address_inputs(
        background,
        _ready_identity(),
        {"model_id": "spatial_poisson", "bandwidth_km": 100.0},
        uv_lock_sha256=lock_hash,
    )
    changed_commit = build_address_inputs(
        background,
        _ready_identity(commit="b" * 40),
        {"model_id": "spatial_poisson", "bandwidth_km": 75.0},
        uv_lock_sha256=lock_hash,
    )
    assert content_address_id(changed_parameters) != base_id
    assert content_address_id(changed_commit) != base_id

    input_hashes = cast(dict[str, str], base["input_hashes"])
    for key in tuple(input_hashes):
        changed_input = copy.deepcopy(base)
        changed_hashes = cast(dict[str, str], changed_input["input_hashes"])
        changed_hashes[key] = "f" * 64
        if key == "environment_lock":
            changed_input["uv_lock_sha256"] = "f" * 64
        assert content_address_id(changed_input) != base_id


def test_address_factory_rejects_nonready_identity_and_uv_lock_drift(
    background: BackgroundConfig,
) -> None:
    dirty = replace(_ready_identity(), worktree_clean=False)
    with pytest.raises(ValueError, match="not ready"):
        build_address_inputs(
            background,
            dirty,
            {"model_id": "etas"},
            uv_lock_sha256=background.inputs.environment_lock_sha256,
        )

    with pytest.raises(ValueError, match="differs from the frozen environment lock"):
        build_address_inputs(
            background,
            _ready_identity(),
            {"model_id": "etas"},
            uv_lock_sha256="f" * 64,
        )


def test_address_factory_rejects_wrong_tag_empty_parameters_and_bad_lock(
    background: BackgroundConfig,
) -> None:
    with pytest.raises(ValueError, match="freeze tag differs"):
        build_address_inputs(
            background,
            replace(_ready_identity(), freeze_tag="v0.2.0-other"),
            {"model_id": "etas"},
            uv_lock_sha256=background.inputs.environment_lock_sha256,
        )
    with pytest.raises(ValueError, match="non-empty mapping"):
        build_address_inputs(
            background,
            _ready_identity(),
            {},
            uv_lock_sha256=background.inputs.environment_lock_sha256,
        )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        build_address_inputs(
            background,
            _ready_identity(),
            {"model_id": "etas"},
            uv_lock_sha256=background.inputs.environment_lock_sha256.upper(),
        )
