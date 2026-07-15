from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import seismoflux.background.runner_local_support as runner_module
from seismoflux.background.config import load_background_protocol
from seismoflux.background.execution import ExecutionSeal
from seismoflux.background.local_support_deliverables import (
    LocalSupportStage2R1Outcome,
    PublishedLocalSupportDeliverables,
)
from seismoflux.background.local_support_runtime import LocalSupportRuntime
from seismoflux.background.scoring_authorization import (
    AuthorizedExecution,
    BackgroundScoringNotAuthorizedError,
)

CONFIG = Path("configs/base_local_support.yaml").resolve()
PROTOCOL = load_background_protocol(Path("configs/background_local_support.yaml"))


def _forbidden_input(*_: object, **__: object) -> object:
    pytest.fail("data inputs must remain unopened before scoring authorization succeeds")


def test_authorization_failure_occurs_before_any_data_input_is_opened(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seal = cast(ExecutionSeal, SimpleNamespace(seal_id="seal"))
    monkeypatch.setattr(runner_module, "load_project_background_config", lambda *_: PROTOCOL)
    monkeypatch.setattr(runner_module, "create_execution_seal", lambda *_, **__: seal)

    def reject(*_: object, **__: object) -> AuthorizedExecution:
        raise BackgroundScoringNotAuthorizedError("synthetic authorization rejection")

    monkeypatch.setattr(runner_module, "create_authorized_execution", reject)
    monkeypatch.setattr(runner_module, "load_study_area", _forbidden_input)
    monkeypatch.setattr(runner_module, "load_earthquake_catalog", _forbidden_input)
    monkeypatch.setattr(runner_module, "load_frozen_issue_calendar", _forbidden_input)
    monkeypatch.setattr(
        runner_module,
        "load_background_local_support_manifest",
        _forbidden_input,
    )

    with pytest.raises(
        BackgroundScoringNotAuthorizedError,
        match="synthetic authorization rejection",
    ):
        runner_module.run_background_stage2_local_support(CONFIG)


def test_seal_is_rechecked_immediately_before_scoring_and_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace: list[str] = []
    seal = cast(ExecutionSeal, SimpleNamespace(seal_id="seal"))
    authorized = cast(AuthorizedExecution, SimpleNamespace(authorization_id="auth"))
    runtime = cast(LocalSupportRuntime, SimpleNamespace(manifest_id="support"))
    science = cast(LocalSupportStage2R1Outcome, object())
    published = cast(PublishedLocalSupportDeliverables, object())
    monkeypatch.setattr(runner_module, "load_project_background_config", lambda *_: PROTOCOL)

    def create_seal(*_: object, **__: object) -> ExecutionSeal:
        trace.append("seal")
        return seal

    def authorize(*_: object, **__: object) -> AuthorizedExecution:
        trace.append("authorize")
        return authorized

    def reseal(*_: object, **__: object) -> ExecutionSeal:
        trace.append("reseal")
        return seal

    def study(*_: object, **__: object) -> Any:
        trace.append("study")
        return SimpleNamespace(projected=object())

    class Catalog:
        def __len__(self) -> int:
            return 1

    def catalog(*_: object, **__: object) -> Any:
        trace.append("catalog")
        return Catalog()

    def calendar(*_: object, **__: object) -> object:
        trace.append("calendar")
        return object()

    def manifest(*_: object, **__: object) -> object:
        trace.append("manifest")
        return object()

    def reconstruct(*_: object, **__: object) -> LocalSupportRuntime:
        trace.append("runtime")
        return runtime

    def score(*_: object, **__: object) -> LocalSupportStage2R1Outcome:
        trace.append("score")
        return science

    def publish(*_: object, **__: object) -> PublishedLocalSupportDeliverables:
        trace.append("publish")
        return published

    monkeypatch.setattr(runner_module, "create_execution_seal", create_seal)
    monkeypatch.setattr(runner_module, "create_authorized_execution", authorize)
    monkeypatch.setattr(runner_module, "require_execution_seal_unchanged", reseal)
    monkeypatch.setattr(runner_module, "load_study_area", study)
    monkeypatch.setattr(runner_module, "load_earthquake_catalog", catalog)
    monkeypatch.setattr(runner_module, "load_frozen_issue_calendar", calendar)
    monkeypatch.setattr(runner_module, "load_background_local_support_manifest", manifest)
    monkeypatch.setattr(runner_module, "catalog_completeness_events", lambda _: ())
    monkeypatch.setattr(runner_module, "build_local_support_runtime", reconstruct)
    monkeypatch.setattr(runner_module, "run_local_support_stage2r1", score)
    monkeypatch.setattr(runner_module, "publish_local_support_deliverables", publish)

    result = runner_module.run_background_stage2_local_support(
        CONFIG,
        memory_probe=lambda: 123,
    )

    assert trace == [
        "seal",
        "authorize",
        "reseal",
        "study",
        "catalog",
        "calendar",
        "manifest",
        "runtime",
        "reseal",
        "score",
        "reseal",
        "publish",
    ]
    assert result.science is science
    assert result.deliverables is published
    assert result.telemetry.process_peak_working_set_bytes == 123
