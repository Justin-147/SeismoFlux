from __future__ import annotations

import hashlib
import importlib
import inspect
import json
from collections.abc import Callable
from dataclasses import replace
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path
from typing import cast
from xml.etree import ElementTree

import pytest
from test_stage4_anomaly_increment_pipeline import (
    _plan,
    run_stage4_synthetic_test_pipeline,
)

import seismoflux.anomaly_increment.formal_publication as formal_publication
import seismoflux.anomaly_increment.immutable_file as immutable_file_module
from seismoflux.anomaly_increment.authorization import Stage4TargetAuthorization
from seismoflux.anomaly_increment.config import Stage4R2ExecutionBlockedError
from seismoflux.anomaly_increment.convergence import (
    CompensatorConvergenceAudit,
    FrozenConvergenceModel,
    RecomputedGridInputs,
    audit_compensator_convergence,
)
from seismoflux.anomaly_increment.formal_publication import (
    AdditionalLocalArtifact,
    FormalPublicationError,
    FormalPublicationReceipt,
    publish_failed_formal_result,
    publish_successful_formal_result,
)
from seismoflux.anomaly_increment.runner import Stage4PublicationPlan
from seismoflux.anomaly_increment.scoring_pipeline import PipelineResult

BINDING = "a" * 64
AUTHORIZATION = "b" * 64
CONVERGENCE_PATH = "outputs/visualizations/anomaly_increment_r2_convergence_audit.json"
_CONVERGENCE_FIXTURES = importlib.import_module("test_stage4_anomaly_increment_convergence")
_model = cast(
    Callable[..., FrozenConvergenceModel],
    _CONVERGENCE_FIXTURES._model,
)
_grids = cast(
    Callable[..., tuple[RecomputedGridInputs, ...]],
    _CONVERGENCE_FIXTURES._grids,
)


@lru_cache(maxsize=1)
def _convergence_evidence() -> tuple[
    CompensatorConvergenceAudit,
    AdditionalLocalArtifact,
]:
    audit = audit_compensator_convergence(model=_model(), grids=_grids())
    artifact = AdditionalLocalArtifact(
        relative_path=CONVERGENCE_PATH,
        bundle_filename="convergence-audit.json",
        payload=json.dumps(audit.as_mapping(), sort_keys=True).encode(),
    )
    return audit, artifact


def _publish_successful(
    project_root: Path,
    publication: Stage4PublicationPlan,
    *,
    execution_binding_id: str,
    authorization_id: str,
    result: PipelineResult,
    additional_local_artifacts: tuple[AdditionalLocalArtifact, ...] = (),
    create_file: Callable[[Path, Path], bool] = formal_publication._atomic_create_only,
) -> FormalPublicationReceipt:
    audit, artifact = _convergence_evidence()
    return formal_publication._publish(
        project_root,
        publication,
        execution_binding_id=execution_binding_id,
        authorization_id=authorization_id,
        model_version=result.retrospective.protocol.model_version,
        status="succeeded",
        result=result,
        failure_code=None,
        convergence_audit=audit,
        additional_local_artifacts=(artifact, *additional_local_artifacts),
        create_file=create_file,
    )


def _publish_failed(
    project_root: Path,
    publication: Stage4PublicationPlan,
    *,
    execution_binding_id: str,
    authorization_id: str,
    model_version: str,
    failure_code: str,
    create_file: Callable[[Path, Path], bool] = formal_publication._atomic_create_only,
) -> FormalPublicationReceipt:
    """Synthetic publication helper; deliberately not a formal execution entrance."""

    return formal_publication._publish(
        project_root,
        publication,
        execution_binding_id=execution_binding_id,
        authorization_id=authorization_id,
        model_version=model_version,
        status="failed",
        result=None,
        failure_code=failure_code,
        convergence_audit=None,
        additional_local_artifacts=(),
        create_file=create_file,
    )


def test_formal_publication_apis_require_protocol_and_unforgeable_provenance_guard_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert tuple(inspect.signature(publish_successful_formal_result).parameters) == (
        "project_root",
        "publication",
        "execution_protocol",
        "authorization",
        "result",
        "convergence_audit",
        "additional_local_artifacts",
        "create_file",
    )
    assert tuple(inspect.signature(publish_failed_formal_result).parameters) == (
        "project_root",
        "publication",
        "execution_protocol",
        "authorization",
        "model_version",
        "failure_code",
        "create_file",
    )

    calls: list[tuple[object, str]] = []
    protocol = {"protocol": "blocked-r2"}

    def block(value: object, *, action: str) -> None:
        calls.append((value, action))
        raise Stage4R2ExecutionBlockedError("synthetic blocked R2")

    def unexpected(*args: object, **kwargs: object) -> object:
        raise AssertionError("formal publication performed work before its R2 guard")

    class ExplodesOnAccess:
        def __getattribute__(self, name: str) -> object:
            raise AssertionError(f"caller object inspected before guard: {name}")

    bomb = ExplodesOnAccess()
    monkeypatch.setattr(formal_publication, "require_stage4_r2_execution_action", block)
    monkeypatch.setattr(formal_publication, "require_stage4_target_authorization", unexpected)
    monkeypatch.setattr(formal_publication, "_publish", unexpected)

    with pytest.raises(Stage4R2ExecutionBlockedError, match="blocked R2"):
        publish_successful_formal_result(
            cast(Path, bomb),
            cast(Stage4PublicationPlan, bomb),
            execution_protocol=protocol,
            authorization=cast(Stage4TargetAuthorization, bomb),
            result=cast(PipelineResult, bomb),
            convergence_audit=cast(CompensatorConvergenceAudit, bomb),
            create_file=cast(Callable[[Path, Path], bool], unexpected),
        )
    with pytest.raises(Stage4R2ExecutionBlockedError, match="blocked R2"):
        publish_failed_formal_result(
            cast(Path, bomb),
            cast(Stage4PublicationPlan, bomb),
            execution_protocol=protocol,
            authorization=cast(Stage4TargetAuthorization, bomb),
            model_version="blocked-r2",
            failure_code="blocked_r2",
            create_file=cast(Callable[[Path, Path], bool], unexpected),
        )
    assert calls == [
        (protocol, "formal_scoring"),
        (protocol, "formal_scoring"),
    ]


def _publication() -> Stage4PublicationPlan:
    return Stage4PublicationPlan(
        public_registry="data/manifests/anomaly_increment_r2_model_registry.json",
        public_report="docs/anomaly_increment_r2_report.md",
        public_static_svg="docs/anomaly_increment_r2_results.svg",
        local_interactive_html="outputs/visualizations/anomaly_increment_r2_dashboard.html",
        public_model_card="docs/model_cards/anomaly_increment_r2.md",
        bundle_root="models/registry/anomaly_increment_r2",
        local_convergence_audit=CONVERGENCE_PATH,
        local_spatial_static=("outputs/visualizations/anomaly_increment_r2_forecast_spatial.svg"),
        local_spatial_interactive=(
            "outputs/visualizations/anomaly_increment_r2_forecast_spatial.html"
        ),
    )


@pytest.fixture(scope="module")
def pipeline_result() -> PipelineResult:
    return run_stage4_synthetic_test_pipeline(_plan("positive"), placebo_injection=None)


def _fixed_paths(root: Path) -> tuple[Path, ...]:
    publication = _publication()
    return tuple(
        root / value
        for value in (
            publication.public_report,
            publication.public_static_svg,
            publication.public_model_card,
            publication.local_interactive_html,
            publication.public_registry,
        )
    )


def test_success_bundle_is_immutable_content_addressed_and_registry_is_last(
    tmp_path: Path,
    pipeline_result: PipelineResult,
) -> None:
    replacements: list[Path] = []

    def capture(source: Path, destination: Path) -> bool:
        replacements.append(destination)
        return formal_publication._atomic_create_only(source, destination)

    receipt = _publish_successful(
        tmp_path,
        _publication(),
        execution_binding_id=BINDING,
        authorization_id=AUTHORIZATION,
        result=pipeline_result,
        create_file=capture,
    )

    assert receipt.status == "succeeded"
    assert receipt.bundle_directory.name == receipt.bundle_id
    assert receipt.bundle_directory.is_dir()
    assert replacements[-1] == tmp_path / _publication().public_registry
    assert all(path.is_file() for path in _fixed_paths(tmp_path))
    assert all(path.stat().st_nlink == 1 for path in _fixed_paths(tmp_path))
    assert all(path.stat().st_nlink == 1 for path in receipt.bundle_directory.iterdir())
    registry = json.loads((tmp_path / _publication().public_registry).read_text("utf-8"))
    assert registry["status"] == "succeeded"
    assert registry["scientific_values_available"] is True
    assert registry["result_fingerprint_sha256"] == (pipeline_result.result_fingerprint_sha256)
    assert registry["forecast_status"] == ("retrospective_generated_target_blind_shadow")
    assert registry["convergence"] == {
        "audit_content_sha256": _convergence_evidence()[0].content_sha256,
        "coarse_50km_vs_25km": "diagnostic_only_not_a_gate",
        "local_artifact_path": CONVERGENCE_PATH,
        "local_artifact_sha256": hashlib.sha256(_convergence_evidence()[1].payload).hexdigest(),
        "near_zero_absolute_tolerance": 1.0e-10,
        "passed_check_count": 40,
        "relative_tolerance": 0.005,
        "spatial_gate": "25km_vs_12.5km",
        "status": "passed",
        "temporal_gate": "1day_vs_0.5day",
        "total_check_count": 40,
    }
    report = (tmp_path / _publication().public_report).read_text("utf-8")
    assert "Gating checks passed: `40/40`" in report
    assert "coarse diagnostic only, not a gate" in report
    model_card = (tmp_path / _publication().public_model_card).read_text("utf-8")
    assert "Passed checks: `40/40`" in model_card
    assert "near-zero absolute tolerance: `1e-10`" in model_card
    assert "![阶段4方法效果](anomaly_increment_r2_results.svg)" in report
    assert f"`{_publication().local_interactive_html}`" in report
    assert "ETAS 数值拟合当前不可用" in report
    assert "M6+ 与 180/365 天窗口始终按证据不足解释" in report
    assert "局部 Mc 偏高只降低对应区域" in report
    assert "目标盲影子预测" in report
    assert "不是真正前瞻预测" in report
    assert "下一步" in report and "G2/G3" in report

    again = _publish_successful(
        tmp_path,
        _publication(),
        execution_binding_id=BINDING,
        authorization_id=AUTHORIZATION,
        result=pipeline_result,
    )
    assert again.bundle_id == receipt.bundle_id
    assert again.manifest_sha256 == receipt.manifest_sha256


def test_partial_fixed_path_publication_removes_only_new_files(
    tmp_path: Path,
    pipeline_result: PipelineResult,
) -> None:
    calls = 0

    def fail_third(source: Path, destination: Path) -> bool:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("synthetic fixed publication failure")
        return formal_publication._atomic_create_only(source, destination)

    with pytest.raises(FormalPublicationError, match="rolled back"):
        _publish_successful(
            tmp_path,
            _publication(),
            execution_binding_id=BINDING,
            authorization_id=AUTHORIZATION,
            result=pipeline_result,
            create_file=fail_third,
        )

    assert not any(path.exists() for path in _fixed_paths(tmp_path))


def test_fixed_formal_paths_are_create_only_and_never_overwritten(
    tmp_path: Path,
    pipeline_result: PipelineResult,
) -> None:
    _publish_successful(
        tmp_path,
        _publication(),
        execution_binding_id=BINDING,
        authorization_id=AUTHORIZATION,
        result=pipeline_result,
    )
    originals = {path: path.read_bytes() for path in _fixed_paths(tmp_path)}

    with pytest.raises(FormalPublicationError, match="different bytes"):
        _publish_failed(
            tmp_path,
            _publication(),
            execution_binding_id=BINDING,
            authorization_id=AUTHORIZATION,
            model_version="stage4-formal-v1",
            failure_code="infrastructure_interruption",
        )

    assert {path: path.read_bytes() for path in _fixed_paths(tmp_path)} == originals


def test_prepositioned_fixed_hardlink_is_rejected_without_opening_target_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "synthetic-sensitive-target.bin"
    target_payload = b"must-never-be-read-through-publication-alias"
    target.write_bytes(target_payload)
    aliased_output = tmp_path / _publication().public_report
    aliased_output.parent.mkdir(parents=True, exist_ok=True)
    try:
        aliased_output.hardlink_to(target)
    except OSError as exc:
        pytest.skip(f"hard links are unavailable on this test filesystem: {exc}")
    opened: list[Path] = []
    real_open = immutable_file_module._open_no_follow

    def record_open(path: Path) -> int:
        opened.append(Path(path))
        return real_open(path)

    monkeypatch.setattr(immutable_file_module, "_open_no_follow", record_open)
    with pytest.raises(FormalPublicationError, match="unsafe immutable publication file"):
        _publish_failed(
            tmp_path,
            _publication(),
            execution_binding_id=BINDING,
            authorization_id=AUTHORIZATION,
            model_version="stage4-formal-v1",
            failure_code="infrastructure_interruption",
        )

    assert aliased_output not in opened
    assert target.read_bytes() == target_payload
    assert aliased_output.samefile(target)


def test_existing_bundle_hardlink_is_rejected_without_opening_target_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _publish_failed(
        tmp_path,
        _publication(),
        execution_binding_id=BINDING,
        authorization_id=AUTHORIZATION,
        model_version="stage4-formal-v1",
        failure_code="infrastructure_interruption",
    )
    artifact = receipt.bundle_directory / "anomaly_increment_report.md"
    target_payload = artifact.read_bytes()
    target = tmp_path / "synthetic-bundle-target.bin"
    target.write_bytes(target_payload)
    artifact.unlink()
    try:
        artifact.hardlink_to(target)
    except OSError as exc:
        pytest.skip(f"hard links are unavailable on this test filesystem: {exc}")
    opened: list[Path] = []
    real_open = immutable_file_module._open_no_follow

    def record_open(path: Path) -> int:
        opened.append(Path(path))
        return real_open(path)

    monkeypatch.setattr(immutable_file_module, "_open_no_follow", record_open)
    with pytest.raises(FormalPublicationError, match="unsafe immutable bundle artifact"):
        _publish_failed(
            tmp_path,
            _publication(),
            execution_binding_id=BINDING,
            authorization_id=AUTHORIZATION,
            model_version="stage4-formal-v1",
            failure_code="infrastructure_interruption",
        )

    assert artifact not in opened
    assert target.read_bytes() == target_payload
    assert artifact.samefile(target)


def test_rollback_refuses_replaced_hardlink_without_opening_or_unlinking_target(
    tmp_path: Path,
    pipeline_result: PipelineResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "synthetic-rollback-target.bin"
    target_payload = b"rollback-must-not-read-or-unlink-this-target"
    target.write_bytes(target_payload)
    created: list[Path] = []
    replacement_active = False
    unsafe_opens: list[Path] = []
    real_open = immutable_file_module._open_no_follow

    def record_open(path: Path) -> int:
        if replacement_active and created and Path(path) == created[0]:
            unsafe_opens.append(Path(path))
        return real_open(path)

    def replace_first_then_fail(source: Path, destination: Path) -> bool:
        nonlocal replacement_active
        if len(created) == 2:
            created[0].unlink()
            created[0].hardlink_to(target)
            replacement_active = True
            raise OSError("synthetic fixed publication failure after replacement")
        was_created = formal_publication._atomic_create_only(source, destination)
        if was_created:
            created.append(destination)
        return was_created

    monkeypatch.setattr(immutable_file_module, "_open_no_follow", record_open)
    with pytest.raises(FormalPublicationError, match="rollback was incomplete"):
        _publish_successful(
            tmp_path,
            _publication(),
            execution_binding_id=BINDING,
            authorization_id=AUTHORIZATION,
            result=pipeline_result,
            create_file=replace_first_then_fail,
        )

    assert unsafe_opens == []
    assert target.read_bytes() == target_payload
    assert created[0].samefile(target)
    assert not created[1].exists()


def test_failed_run_publishes_value_free_pages_without_fake_curves(tmp_path: Path) -> None:
    receipt = _publish_failed(
        tmp_path,
        _publication(),
        execution_binding_id=BINDING,
        authorization_id=AUTHORIZATION,
        model_version="stage4-formal-v1",
        failure_code="infrastructure_interruption",
    )

    assert receipt.status == "failed"
    registry = json.loads((tmp_path / _publication().public_registry).read_text("utf-8"))
    assert registry["status"] == "failed"
    assert registry["scientific_values_available"] is False
    assert registry["result_fingerprint_sha256"] is None
    assert registry["diagnostics_sha256"] is None
    html = (tmp_path / _publication().local_interactive_html).read_text("utf-8")
    svg = (tmp_path / _publication().public_static_svg).read_text("utf-8")
    assert 'data-scientific-values="absent"' in html
    assert 'data-scientific-values="absent"' in svg
    assert "没有生成科学数值、曲线或替代结论" in html
    assert "information_gain" not in html
    assert "rankRows" not in html


def test_publication_rejects_forbidden_absolute_wording(
    tmp_path: Path,
    pipeline_result: PipelineResult,
) -> None:
    changed = replace(
        pipeline_result,
        static_svg=pipeline_result.static_svg.replace("</svg>", "probability</svg>"),
    )
    with pytest.raises(FormalPublicationError, match="forbidden"):
        _publish_successful(
            tmp_path,
            _publication(),
            execution_binding_id=BINDING,
            authorization_id=AUTHORIZATION,
            result=changed,
        )


@pytest.mark.parametrize(
    "payload",
    (
        b'{"coordinates":[[120.1234,30.5678]]}',
        b'{"event_id":"real-target-id"}',
        b'{"target_locations":["restricted"]}',
        b'local_path="D:\\\\AIPred\\\\SeismoFlux\\\\secret.json"',
        b'local_path="/home/user/SeismoFlux/secret.json"',
        b'local_path="file:///tmp/secret.json"',
        b'local_path="/opt/seismoflux/private/receipt.json"',
        b'{"geometry":"POINT (120.1 30.2)"}',
        '{"震中":"受限目标明细"}'.encode(),
        b'{"target_id":"held-out-event"}',
    ),
)
def test_public_payload_rejects_spatial_target_metadata_and_absolute_paths(
    payload: bytes,
) -> None:
    with pytest.raises(
        FormalPublicationError,
        match="restricted geometry token|absolute local path",
    ):
        formal_publication._validate_public_payload("public.json", payload)


def test_local_spatial_artifact_may_contain_coordinates_but_registry_contains_only_hash(
    tmp_path: Path,
    pipeline_result: PipelineResult,
) -> None:
    payload = (
        b'<!doctype html><main data-longitude="120.1234" data-latitude="30.5678">'
        b"<script>const point={longitude:120.1234,latitude:30.5678};</script></main>"
    )
    local_path = "outputs/visualizations/anomaly_increment_spatial.html"
    extension = AdditionalLocalArtifact(
        relative_path=local_path,
        bundle_filename="spatial.html",
        payload=payload,
    )

    _publish_successful(
        tmp_path,
        _publication(),
        execution_binding_id=BINDING,
        authorization_id=AUTHORIZATION,
        result=pipeline_result,
        additional_local_artifacts=(extension,),
    )

    assert (tmp_path / local_path).read_bytes() == payload
    registry_text = (tmp_path / _publication().public_registry).read_text("utf-8")
    registry = json.loads(registry_text)
    assert (
        registry["additional_local_artifact_sha256"][local_path]
        == hashlib.sha256(payload).hexdigest()
    )
    assert (
        registry["additional_local_artifact_sha256"][CONVERGENCE_PATH]
        == (registry["convergence"]["local_artifact_sha256"])
    )
    assert "120.1234" not in registry_text
    assert "30.5678" not in registry_text
    assert "longitude" not in registry_text.casefold()
    assert "latitude" not in registry_text.casefold()
    report = (tmp_path / _publication().public_report).read_text("utf-8")
    assert f"`{local_path}`" in report


@pytest.mark.parametrize(
    "payload",
    (
        b"<script>fetch('/target')</script>",
        b'<script src="local-secret.js"></script>',
    ),
)
def test_local_artifact_rejects_remote_or_external_script_capabilities(
    payload: bytes,
) -> None:
    with pytest.raises(FormalPublicationError, match="dangerous capability"):
        AdditionalLocalArtifact(
            relative_path="outputs/visualizations/spatial.html",
            bundle_filename="spatial.html",
            payload=payload,
        )


class _TagRecorder(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: set[str] = set()

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        del attrs
        self.tags.add(tag)


def _assert_utf8_xml_html_and_no_mojibake(root: Path) -> None:
    publication = _publication()
    for path in _fixed_paths(root):
        payload = path.read_bytes()
        text = payload.decode("utf-8")
        assert text.encode("utf-8") == payload
        assert not any(fragment in text for fragment in formal_publication._MOJIBAKE_FRAGMENTS)
    svg_root = ElementTree.fromstring((root / publication.public_static_svg).read_bytes())
    assert svg_root.tag.endswith("svg")
    parser = _TagRecorder()
    parser.feed((root / publication.local_interactive_html).read_text("utf-8"))
    parser.close()
    assert {"html", "body"}.issubset(parser.tags)


def test_success_and_failure_outputs_are_utf8_parseable_and_not_mojibake(
    tmp_path: Path,
    pipeline_result: PipelineResult,
) -> None:
    success_root = tmp_path / "success"
    failure_root = tmp_path / "failure"
    success_root.mkdir()
    failure_root.mkdir()
    _publish_successful(
        success_root,
        _publication(),
        execution_binding_id=BINDING,
        authorization_id=AUTHORIZATION,
        result=pipeline_result,
    )
    _publish_failed(
        failure_root,
        _publication(),
        execution_binding_id=BINDING,
        authorization_id=AUTHORIZATION,
        model_version="stage4-formal-v1",
        failure_code="infrastructure_interruption",
    )

    _assert_utf8_xml_html_and_no_mojibake(success_root)
    _assert_utf8_xml_html_and_no_mojibake(failure_root)
