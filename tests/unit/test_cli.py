from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
import yaml

import seismoflux.cli as cli_module
from seismoflux.background.config import load_background_protocol
from seismoflux.cli import COMMAND_SPECS, main

BACKGROUND_CONFIG = Path("configs/background.yaml")


@pytest.fixture
def _background_protocol_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    protocol = load_background_protocol(BACKGROUND_CONFIG)
    monkeypatch.setattr(
        cli_module,
        "load_project_background_config",
        lambda _path: protocol,
    )


def _required_arguments(command: str) -> list[str]:
    if command in {"train", "backtest", "optimize-regions", "freeze"}:
        return ["--experiment", "synthetic-v1"]
    if command == "forecast":
        return ["--issue-date", "2026-07-13", "--model", "frozen/model-id"]
    if command == "mature":
        return ["--as-of", "2026-07-13"]
    if command in {"render", "validate-release"}:
        return ["--issue-id", "ISSUE_ID"]
    return []


@pytest.mark.parametrize("command", sorted(COMMAND_SPECS))
def test_every_command_supports_machine_readable_dry_run(
    command: str,
    capsys: pytest.CaptureFixture[str],
    _background_protocol_loader: None,
) -> None:
    exit_code = main([command, *_required_arguments(command), "--dry-run"])
    captured = capsys.readouterr()
    manifest = cast(dict[str, Any], json.loads(captured.out))

    assert exit_code == 0
    assert captured.err == ""
    assert manifest["command"] == command
    assert manifest["mode"] == "dry_run"
    assert manifest["status"] == "planned"


def test_deferred_command_fails_instead_of_making_placeholder_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["build-anomaly-history"])
    captured = capsys.readouterr()
    manifest = cast(dict[str, Any], json.loads(captured.out))

    assert exit_code == 2
    assert manifest["status"] == "blocked"
    assert "deferred to stage 3" in captured.err


def test_background_dry_run_is_score_free_and_machine_readable(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    _background_protocol_loader: None,
) -> None:
    def forbidden(*_: object, **__: object) -> object:
        raise AssertionError("background dry-run must not load rows, score, or publish")

    monkeypatch.setattr(cli_module, "run_background_stage2", forbidden)
    monkeypatch.setattr(pd, "read_parquet", forbidden)

    exit_code = main(["build-background", "--dry-run"])
    captured = capsys.readouterr()
    manifest = cast(dict[str, Any], json.loads(captured.out))

    assert exit_code == 0
    assert captured.err == ""
    assert manifest["implementation_stage"] == 2
    assert manifest["implementation_status"] == "implemented"
    assert manifest["status"] == "planned"
    assert manifest["details"]["models"] == [
        "uniform_poisson",
        "spatial_poisson",
        "etas",
    ]
    assert len(manifest["details"]["snapshots"]) == 5


def test_background_dry_run_without_manifest_never_calls_filesystem_writers(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    _background_protocol_loader: None,
) -> None:
    def forbidden(*_: object, **__: object) -> object:
        raise AssertionError("background dry-run attempted a filesystem write")

    monkeypatch.setattr(Path, "write_text", forbidden)
    monkeypatch.setattr(Path, "write_bytes", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    monkeypatch.setattr(tempfile, "NamedTemporaryFile", forbidden)
    monkeypatch.setattr(os, "replace", forbidden)

    exit_code = main(["build-background", "--dry-run"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert json.loads(captured.out)["status"] == "planned"
    assert captured.err == ""


class _SyntheticBackgroundRun:
    def to_manifest_details(self) -> dict[str, object]:
        return {
            "execution_seal_id": "a" * 64,
            "fixed_delivery_confirmed": True,
            "g1_passed": False,
            "selected_model_id": "uniform_poisson",
            "stage3_allowed": False,
        }


def test_background_execute_calls_single_production_entry_and_emits_result_manifest(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    _background_protocol_loader: None,
) -> None:
    calls: list[tuple[Path, object]] = []

    def execute(
        config_path: Path,
        *,
        progress: object,
    ) -> _SyntheticBackgroundRun:
        calls.append((config_path, progress))
        return _SyntheticBackgroundRun()

    monkeypatch.setattr(cli_module, "run_background_stage2", execute)
    exit_code = main(["build-background"])
    captured = capsys.readouterr()
    manifest = cast(dict[str, Any], json.loads(captured.out))

    assert exit_code == 0
    assert captured.err == ""
    assert calls == [(Path("configs/base.yaml"), cli_module._background_progress)]
    assert manifest["mode"] == "execute"
    assert manifest["status"] == "completed"
    assert manifest["details"]["execution_seal_id"] == "a" * 64
    assert manifest["details"]["fixed_delivery_confirmed"] is True
    assert manifest["details"]["stage3_allowed"] is False


def test_background_failure_emits_machine_readable_negative_run_manifest(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    _background_protocol_loader: None,
) -> None:
    def fail(*_: object, **__: object) -> object:
        raise RuntimeError("synthetic background failure")

    monkeypatch.setattr(cli_module, "run_background_stage2", fail)
    exit_code = main(["build-background"])
    captured = capsys.readouterr()
    manifest = cast(dict[str, Any], json.loads(captured.out))

    assert exit_code == 2
    assert manifest["status"] == "failed"
    assert manifest["details"]["failure"]["error_type"] == "RuntimeError"
    assert manifest["details"]["failure"]["fixed_delivery_confirmed"] is False
    assert "synthetic background failure" in captured.err


@pytest.mark.parametrize("failure_point", ("details", "build", "emit"))
def test_background_post_delivery_reporting_failure_preserves_success(
    failure_point: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    _background_protocol_loader: None,
) -> None:
    calls = 0

    class _Result:
        def to_manifest_details(self) -> dict[str, object]:
            if failure_point == "details":
                raise OSError("synthetic details failure")
            return {"fixed_delivery_confirmed": True}

    def execute(*_: object, **__: object) -> _Result:
        nonlocal calls
        calls += 1
        return _Result()

    def fail(*_: object, **__: object) -> object:
        raise OSError(f"synthetic {failure_point} failure")

    monkeypatch.setattr(cli_module, "run_background_stage2", execute)
    if failure_point == "build":
        monkeypatch.setattr(cli_module, "build_run_manifest", fail)
    if failure_point == "emit":
        monkeypatch.setattr(cli_module, "emit_run_manifest", fail)

    exit_code = main(["build-background"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert calls == 1
    assert "阶段2固定交付已确认" in captured.err
    assert '"status": "failed"' not in captured.out


def test_background_failure_manifest_error_does_not_mask_runner_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    _background_protocol_loader: None,
) -> None:
    def fail_run(*_: object, **__: object) -> object:
        raise RuntimeError("synthetic pre-delivery failure")

    def fail_manifest(*_: object, **__: object) -> object:
        raise OSError("synthetic failure-manifest error")

    monkeypatch.setattr(cli_module, "run_background_stage2", fail_run)
    monkeypatch.setattr(cli_module, "emit_run_manifest", fail_manifest)

    exit_code = main(["build-background"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "synthetic pre-delivery failure" in captured.err
    assert "失败运行清单报告也失败" in captured.err
    assert "固定交付未确认" in captured.err


def test_background_dry_run_manifest_failure_remains_a_cli_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    _background_protocol_loader: None,
) -> None:
    def fail(*_: object, **__: object) -> object:
        raise OSError("synthetic dry-run manifest failure")

    monkeypatch.setattr(cli_module, "emit_run_manifest", fail)

    exit_code = main(["build-background", "--dry-run"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "synthetic dry-run manifest failure" in captured.err
    assert "固定交付已确认" not in captured.err


@pytest.mark.parametrize("runner_succeeds", (True, False))
def test_background_reporting_survives_a_broken_stderr(
    runner_succeeds: bool,
    monkeypatch: pytest.MonkeyPatch,
    _background_protocol_loader: None,
) -> None:
    class _BrokenStderr:
        def write(self, _: str) -> int:
            raise ValueError("synthetic closed stderr")

    def execute(*_: object, **__: object) -> _SyntheticBackgroundRun:
        if not runner_succeeds:
            raise RuntimeError("synthetic pre-delivery failure")
        return _SyntheticBackgroundRun()

    def fail_manifest(*_: object, **__: object) -> object:
        raise OSError("synthetic manifest failure")

    monkeypatch.setattr(cli_module, "run_background_stage2", execute)
    monkeypatch.setattr(cli_module, "emit_run_manifest", fail_manifest)
    monkeypatch.setattr(sys, "stderr", _BrokenStderr())

    assert main(["build-background"]) == (0 if runner_succeeds else 2)


@pytest.mark.parametrize(
    "destination",
    (
        "data/manifests/background_model_registry.json",
        "docs/background_baseline_report.md",
        "data/processed/stage2/run-manifest.json",
        "models/registry/background/run-manifest.json",
        "outputs/backtests/background/run-manifest.json",
        "outputs/experiments/background/run-manifest.json",
    ),
)
def test_background_manifest_cannot_collide_with_inputs_or_outputs(
    destination: str,
    capsys: pytest.CaptureFixture[str],
    _background_protocol_loader: None,
) -> None:
    exit_code = main(["build-background", "--dry-run", "--manifest", destination])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert captured.out == ""
    if destination.endswith(".json"):
        assert "collides with a protected project artifact" in captured.err
    else:
        assert "must use a .json extension" in captured.err


@pytest.mark.parametrize(
    ("command", "pipeline_name"),
    (("ingest", "ingest_stage1"), ("validate-data", "validate_stage1_data")),
)
def test_stage1_dry_run_does_not_execute_pipeline(
    command: str,
    pipeline_name: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def unexpected_pipeline_call(**_: Any) -> dict[str, Any]:
        raise AssertionError("dry-run must not execute the stage-1 pipeline")

    monkeypatch.setattr(cli_module, pipeline_name, unexpected_pipeline_call)

    exit_code = main([command, "--dry-run"])
    captured = capsys.readouterr()
    manifest = cast(dict[str, Any], json.loads(captured.out))

    assert exit_code == 0
    assert captured.err == ""
    assert manifest["implementation_status"] == "implemented"
    assert manifest["status"] == "planned"
    assert manifest["details"]["source_count"] == 7


@pytest.mark.parametrize(
    ("command", "pipeline_name"),
    (("ingest", "ingest_stage1"), ("validate-data", "validate_stage1_data")),
)
def test_stage1_execute_calls_pipeline_and_emits_completed_manifest(
    command: str,
    pipeline_name: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[dict[str, Any]] = []

    def pipeline(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"pipeline_result": "passed"}

    monkeypatch.setattr(cli_module, pipeline_name, pipeline)

    exit_code = main([command])
    captured = capsys.readouterr()
    manifest = cast(dict[str, Any], json.loads(captured.out))

    assert exit_code == 0
    assert captured.err == ""
    assert len(calls) == 1
    assert calls[0]["config_path"] == Path("configs/base.yaml")
    assert manifest["mode"] == "execute"
    assert manifest["implementation_status"] == "implemented"
    assert manifest["status"] == "completed"
    assert manifest["details"]["pipeline_result"] == "passed"


@pytest.mark.parametrize(
    "destination",
    (
        "data/manifests/data_catalog.json",
        "data/manifests/data_quality_report.json",
        "data/contracts/run-manifest.json",
        "data/processed/stage1/run-manifest.json",
    ),
)
def test_stage1_manifest_cannot_overwrite_protected_artifacts(
    destination: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["ingest", "--dry-run", "--manifest", destination])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert captured.out == ""
    assert "collides with a protected project artifact" in captured.err


def test_dry_run_writes_only_an_explicit_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "运行清单.json"

    exit_code = main(["inventory", "--dry-run", "--manifest", str(output)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8")) == json.loads(captured.out)


def test_background_dry_run_writes_only_an_explicit_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    _background_protocol_loader: None,
) -> None:
    output = tmp_path / "阶段2运行清单.json"

    exit_code = main(["build-background", "--dry-run", "--manifest", str(output)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8")) == json.loads(captured.out)
    assert tuple(tmp_path.iterdir()) == (output,)


def test_manifest_cannot_overwrite_inventory(
    capsys: pytest.CaptureFixture[str],
) -> None:
    inventory = Path("data/manifests/source_inventory.csv")
    before = inventory.read_bytes()

    exit_code = main(
        ["inventory", "--dry-run", "--manifest", "data/manifests/source_inventory.csv"]
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "must use a .json extension" in captured.err
    assert inventory.read_bytes() == before


def test_manifest_cannot_collide_with_json_inventory_override(
    capsys: pytest.CaptureFixture[str],
) -> None:
    collision = "outputs/run_manifests/collision.json"

    exit_code = main(["inventory", "--dry-run", "--output", collision, "--manifest", collision])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "collides with a protected project artifact" in captured.err
    assert not Path(collision).exists()


def test_cli_rejects_invalid_iso_date(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(["forecast", "--issue-date", "13-07-2026", "--model", "frozen/model-id"])
    capsys.readouterr()

    assert error.value.code == 2


def test_cli_rejects_basic_iso_date_without_separators(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        main(["forecast", "--issue-date", "20260713", "--model", "frozen/model-id"])
    capsys.readouterr()

    assert error.value.code == 2


def test_inventory_output_must_remain_inside_project(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    outside = tmp_path / "outside.csv"

    exit_code = main(["inventory", "--dry-run", "--output", str(outside)])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "project-relative" in captured.err
    assert not outside.exists()


def test_cli_executes_inventory_with_synthetic_unicode_source(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    project_root = tmp_path / "project"
    config_dir = project_root / "configs"
    source_root = tmp_path / "原始输入"
    config_dir.mkdir(parents=True)
    source_root.mkdir()
    (source_root / "目录.eqt").write_bytes(b"catalog")

    base = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))
    base["config_files"]["data_sources"] = "configs/data_sources.yaml"
    (config_dir / "base.yaml").write_text(
        yaml.safe_dump(base, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    data_sources = {
        "schema_version": 1,
        "source_root_env": "SEISMOFLUX_UNUSED_SOURCE_ROOT",
        "source_root": str(source_root),
        "inventory_output": "data/manifests/source_inventory.csv",
        "sources": [
            {
                "id": "catalog",
                "category": "synthetic",
                "kind": "file",
                "path": "目录.eqt",
                "recursive": False,
                "license_status": "unknown_no_redistribution",
            }
        ],
    }
    (config_dir / "data_sources.yaml").write_text(
        yaml.safe_dump(data_sources, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    exit_code = main(["inventory", "--config", str(config_dir / "base.yaml")])
    manifest = cast(dict[str, Any], json.loads(capsys.readouterr().out))

    output = project_root / "data" / "manifests" / "source_inventory.csv"
    assert exit_code == 0
    assert output.is_file()
    assert manifest["details"]["file_count"] == 1
    assert manifest["config"]["path"] == "configs/base.yaml"


def test_manifest_uses_configuration_repository_when_called_elsewhere(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = Path.cwd().resolve()
    config_path = repository / "configs" / "base.yaml"
    expected_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.chdir(tmp_path)

    exit_code = main(["inventory", "--config", str(config_path), "--dry-run"])
    manifest = cast(dict[str, Any], json.loads(capsys.readouterr().out))

    assert exit_code == 0
    assert manifest["environment"]["git_commit"] == expected_commit
    assert manifest["config"]["path"] == "configs/base.yaml"
    assert str(repository) not in json.dumps(manifest)


def test_inventory_runtime_safety_error_returns_standard_exit_code(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def changed_source(_: object) -> list[object]:
        raise RuntimeError("source changed")

    monkeypatch.setattr(cli_module, "build_inventory", changed_source)

    exit_code = main(["inventory"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert captured.out == ""
    assert "source changed" in captured.err
