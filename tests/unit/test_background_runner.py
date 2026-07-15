from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import yaml

import seismoflux.background.runner as runner_module
from seismoflux.background.artifacts import canonical_json_bytes
from seismoflux.background.config import BackgroundConfig
from seismoflux.background.deliverables import (
    BackgroundDeliverables,
    PublishedBackgroundDeliverables,
)
from seismoflux.background.execution import ExecutionSeal, RepositoryIdentity
from seismoflux.background.pipeline import BackgroundPipelineResult
from seismoflux.background.publication import BackgroundRegistry, RegistryReportPublication
from seismoflux.config import SeismoFluxConfig


@pytest.fixture(scope="module")
def project() -> SeismoFluxConfig:
    raw = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))
    return SeismoFluxConfig.model_validate(raw)


@pytest.fixture(scope="module")
def background() -> BackgroundConfig:
    raw = yaml.safe_load(Path("configs/background.yaml").read_text(encoding="utf-8"))
    return BackgroundConfig.model_validate(raw)


def _seal(background: BackgroundConfig) -> ExecutionSeal:
    identity = RepositoryIdentity(
        code_commit="a" * 40,
        branch="codex/phase-2",
        upstream="origin/codex/phase-2",
        upstream_commit="a" * 40,
        freeze_tag=background.freeze_tag,
        freeze_tag_commit="b" * 40,
        git_available=True,
        worktree_clean=True,
        tag_is_ancestor=True,
        upstream_matches_head=True,
    )
    return ExecutionSeal(
        repository=identity,
        protocol_sha256=hashlib.sha256(
            canonical_json_bytes(background.model_dump(mode="python"))
        ).hexdigest(),
        input_hashes=tuple(
            sorted(
                {
                    "environment_lock": background.inputs.environment_lock_sha256,
                    "data_catalog": background.inputs.data_catalog_sha256,
                    "earthquake_dataset": background.inputs.earthquake_dataset_sha256,
                    "study_area": background.inputs.study_area_sha256,
                    "issue_manifest": background.inputs.issue_manifest_sha256,
                    "production_fixture": (
                        background.numerical_regression.production_fixture_sha256
                    ),
                    "oracle_metadata": background.numerical_regression.oracle_metadata_sha256,
                }.items()
            )
        ),
    )


@dataclass(frozen=True)
class _Published:
    registry: object


@dataclass(frozen=True)
class _Study:
    geographic: object


def test_runner_seals_before_catalog_and_reseals_before_each_publication_surface(
    tmp_path: Path,
    project: SeismoFluxConfig,
    background: BackgroundConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "configs" / "base.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.touch()
    order: list[str] = []
    seal = _seal(background)
    science = cast(BackgroundPipelineResult, object())
    bundle_inputs = cast(BackgroundDeliverables, object())
    registry = cast(BackgroundRegistry, object())
    published = cast(PublishedBackgroundDeliverables, _Published(registry))
    fixed = cast(RegistryReportPublication, object())

    monkeypatch.setattr(runner_module, "load_config", lambda _: project)

    def load_protocol(_: Path) -> BackgroundConfig:
        order.append("protocol")
        return background

    def load_bound_config(_: Path) -> BackgroundConfig:
        order.append("bound_config")
        return background

    monkeypatch.setattr(runner_module, "load_background_protocol", load_protocol)
    monkeypatch.setattr(
        runner_module,
        "load_project_background_config",
        load_bound_config,
    )

    def create_seal(*_: object, **__: object) -> ExecutionSeal:
        order.append("seal")
        return seal

    def study_loader(*_: object, **__: object) -> object:
        order.append("study")
        return _Study(geographic=object())

    def catalog_loader(*_: object, **__: object) -> object:
        order.append("catalog")

        class _Catalog:
            def __len__(self) -> int:
                return 7

        return _Catalog()

    def calendar_loader(*_: object, **__: object) -> object:
        order.append("calendar")
        return object()

    def grid_builder(_: object) -> object:
        order.append("grid")
        return object()

    def pipeline(*_: object, **kwargs: object) -> BackgroundPipelineResult:
        order.append("science")
        fixture_path = cast(Path, kwargs["production_fixture_path"])
        assert fixture_path.is_absolute()
        return science

    def adapt(*_: object) -> BackgroundDeliverables:
        order.append("adapt")
        return bundle_inputs

    def reseal(*_: object, **__: object) -> ExecutionSeal:
        order.append("reseal_before_bundles")
        return seal

    def publish_bundles(*_: object, **__: object) -> PublishedBackgroundDeliverables:
        order.append("publish_bundles")
        return published

    def publish_fixed(*_: object, **__: object) -> RegistryReportPublication:
        order.append("reseal_and_publish_fixed")
        return fixed

    def memory_probe() -> int:
        order.append("memory_probe")
        return 123_456

    monkeypatch.setattr(runner_module, "create_execution_seal", create_seal)
    monkeypatch.setattr(runner_module, "load_study_area", study_loader)
    monkeypatch.setattr(runner_module, "load_earthquake_catalog", catalog_loader)
    monkeypatch.setattr(runner_module, "load_frozen_issue_calendar", calendar_loader)
    monkeypatch.setattr(runner_module, "build_grid_family", grid_builder)
    monkeypatch.setattr(runner_module, "run_background_pipeline", pipeline)
    monkeypatch.setattr(runner_module, "build_background_deliverables", adapt)
    monkeypatch.setattr(runner_module, "require_execution_seal_unchanged", reseal)
    monkeypatch.setattr(runner_module, "publish_background_deliverables", publish_bundles)
    monkeypatch.setattr(runner_module, "publish_registry_and_report_sealed", publish_fixed)

    progress: list[str] = []

    def report_progress(message: str) -> None:
        progress.append(message)
        if message.startswith("execution_seal:ready:"):
            os.chdir(tmp_path.parent)

    monkeypatch.chdir(tmp_path)
    result = runner_module.run_background_stage2(
        Path("configs/base.yaml"),
        progress=report_progress,
        memory_probe=memory_probe,
    )

    assert order == [
        "protocol",
        "bound_config",
        "seal",
        "study",
        "catalog",
        "calendar",
        "grid",
        "science",
        "adapt",
        "reseal_before_bundles",
        "publish_bundles",
        "memory_probe",
        "reseal_and_publish_fixed",
    ]
    assert result.execution_seal == seal
    assert result.science is science
    assert result.telemetry.process_peak_working_set_bytes == 123_456
    assert result.telemetry.elapsed_seconds >= 0.0
    assert result.telemetry.cpu_seconds >= 0.0
    assert progress[0] == "execution_seal:start"
    assert progress[-1] == "deliverables:fixed_publish:start"


def test_runner_unclassified_execution_error_publishes_nothing(
    tmp_path: Path,
    project: SeismoFluxConfig,
    background: BackgroundConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "configs" / "base.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.touch()
    monkeypatch.setattr(runner_module, "load_config", lambda _: project)
    monkeypatch.setattr(runner_module, "load_background_protocol", lambda _: background)
    monkeypatch.setattr(
        runner_module,
        "load_project_background_config",
        lambda _: background,
    )
    monkeypatch.setattr(runner_module, "create_execution_seal", lambda *_, **__: _seal(background))
    monkeypatch.setattr(
        runner_module,
        "load_study_area",
        lambda *_, **__: _Study(geographic=object()),
    )

    class _Catalog:
        def __len__(self) -> int:
            return 1

    monkeypatch.setattr(runner_module, "load_earthquake_catalog", lambda *_, **__: _Catalog())
    monkeypatch.setattr(runner_module, "load_frozen_issue_calendar", lambda *_, **__: object())
    monkeypatch.setattr(runner_module, "build_grid_family", lambda _: object())

    def fail_science(*_: object, **__: object) -> BackgroundPipelineResult:
        raise RuntimeError("synthetic unclassified execution failure")

    monkeypatch.setattr(runner_module, "run_background_pipeline", fail_science)

    def forbidden(*_: object, **__: object) -> object:
        raise AssertionError("failure path must not publish")

    monkeypatch.setattr(runner_module, "build_background_deliverables", forbidden)
    monkeypatch.setattr(runner_module, "publish_background_deliverables", forbidden)
    monkeypatch.setattr(runner_module, "publish_registry_and_report_sealed", forbidden)

    with pytest.raises(RuntimeError, match="synthetic unclassified execution failure"):
        runner_module.run_background_stage2(config_path)


def test_runner_reseal_failure_prevents_all_publication(
    tmp_path: Path,
    project: SeismoFluxConfig,
    background: BackgroundConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "configs" / "base.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.touch()
    monkeypatch.setattr(runner_module, "load_config", lambda _: project)
    monkeypatch.setattr(runner_module, "load_background_protocol", lambda _: background)
    monkeypatch.setattr(
        runner_module,
        "load_project_background_config",
        lambda _: background,
    )
    monkeypatch.setattr(runner_module, "create_execution_seal", lambda *_, **__: _seal(background))
    monkeypatch.setattr(
        runner_module,
        "load_study_area",
        lambda *_, **__: _Study(geographic=object()),
    )

    class _Catalog:
        def __len__(self) -> int:
            return 1

    monkeypatch.setattr(runner_module, "load_earthquake_catalog", lambda *_, **__: _Catalog())
    monkeypatch.setattr(runner_module, "load_frozen_issue_calendar", lambda *_, **__: object())
    monkeypatch.setattr(runner_module, "build_grid_family", lambda _: object())
    monkeypatch.setattr(
        runner_module,
        "run_background_pipeline",
        lambda *_, **__: cast(BackgroundPipelineResult, object()),
    )
    monkeypatch.setattr(
        runner_module,
        "build_background_deliverables",
        lambda *_: cast(BackgroundDeliverables, object()),
    )

    def drift(*_: object, **__: object) -> ExecutionSeal:
        raise RuntimeError("synthetic seal drift")

    monkeypatch.setattr(runner_module, "require_execution_seal_unchanged", drift)

    def forbidden(*_: object, **__: object) -> object:
        raise AssertionError("seal drift must prevent publication")

    monkeypatch.setattr(runner_module, "publish_background_deliverables", forbidden)
    monkeypatch.setattr(runner_module, "publish_registry_and_report_sealed", forbidden)

    with pytest.raises(RuntimeError, match="synthetic seal drift"):
        runner_module.run_background_stage2(config_path)


def test_process_peak_memory_probe_is_portable_on_unknown_os(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(os, "name", "unknown")
    assert runner_module.process_peak_working_set_bytes() is None


def test_process_peak_memory_probe_reports_supported_platform() -> None:
    observed = runner_module.process_peak_working_set_bytes()
    if os.name in {"nt", "posix"}:
        assert observed is not None and observed > 0


def test_runner_module_imports_in_a_fresh_platform_process() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", "import seismoflux.background.runner"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr


def test_archived_scientific_failure_manifest_confirms_fixed_negative_delivery(
    background: BackgroundConfig,
) -> None:
    seal = _seal(background)
    attempts = tuple(SimpleNamespace(status="not_run") for _ in range(15))
    failure = SimpleNamespace(
        model_dump=lambda mode: {
            "failure_stage": "completeness",
            "failure_reason_code": "no_eligible_temporal_stratum",
            "failure_reasons": ["no eligible temporal completeness stratum"],
            "evidence_id": "f" * 64,
        }
    )
    registry = SimpleNamespace(
        model_attempts=attempts,
        g1=SimpleNamespace(passed=False, status="not_evaluable", passing_models=()),
        selection=SimpleNamespace(selected_model_id=None),
        stage3_allowed=False,
        scientific_summary=SimpleNamespace(
            outcome_status="scientific_gate_failed",
            failure=failure,
        ),
    )
    bundle_publications = tuple(
        SimpleNamespace(
            bundle_kind=bundle_kind,
            artifact=SimpleNamespace(
                artifact_id=f"{index:016x}",
                manifest_sha256=f"{index + 10:064x}",
            ),
        )
        for index, bundle_kind in enumerate(
            ("processed", "model", "backtest", "experiment"),
            start=1,
        )
    )
    resources = SimpleNamespace(
        detected_physical_cores=8,
        reserve_physical_cores=2,
        configured_max_workers=12,
        effective_workers=6,
    )
    run = runner_module.BackgroundRunResult(
        execution_seal=seal,
        science=cast(BackgroundPipelineResult, SimpleNamespace(resources=resources)),
        deliverables=cast(
            PublishedBackgroundDeliverables,
            SimpleNamespace(
                registry=registry,
                bundle_publications=bundle_publications,
            ),
        ),
        fixed_publication=cast(
            RegistryReportPublication,
            SimpleNamespace(
                registry=SimpleNamespace(sha256="1" * 64),
                report=SimpleNamespace(sha256="2" * 64),
            ),
        ),
        telemetry=runner_module.BackgroundRunTelemetry(1.0, 0.5, 1024),
    )

    details = run.to_manifest_details()

    assert details["fixed_delivery_confirmed"] is True
    assert details["scientific_outcome_status"] == "scientific_gate_failed"
    assert details["g1_status"] == "not_evaluable"
    assert details["stage3_allowed"] is False
    assert details["model_attempts"] == {
        "total": 15,
        "succeeded": 0,
        "failed": 0,
        "not_run": 15,
    }
