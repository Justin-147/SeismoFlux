# ruff: noqa: RUF001
from __future__ import annotations

import inspect
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import seismoflux.background.local_support_deliverables as publication_module
from seismoflux.background.config import BackgroundConfig
from seismoflux.background.evidence import (
    AuditedBackgroundModelEvidence,
    PairedInformationGainEvidence,
    PointProcessScoreEvidence,
)
from seismoflux.background.local_support_deliverables import LocalSupportDeliverables
from seismoflux.background.local_support_runtime import LocalSupportRuntime
from seismoflux.background.publication import (
    BacktestBundle,
    ExperimentBundle,
    ModelBundle,
    ProcessedBundle,
)
from seismoflux.background.score_ledger import FROZEN_PRIMARY_INTERVALS
from seismoflux.background.scoring_authorization import AuthorizedExecution
from seismoflux.background.stage2r1 import LocalSupportStage2R1Result
from seismoflux.background.visualization_local_support import (
    LocalSupportVisualBootstrap,
    LocalSupportVisualHorizon,
    LocalSupportVisualSensitivity,
    LocalSupportVisualSnapshot,
    render_local_support_results_svg,
)


def test_publication_api_does_not_accept_caller_supplied_deliverables() -> None:
    parameters = inspect.signature(publication_module.publish_local_support_deliverables).parameters
    assert "deliverables" not in parameters


@pytest.mark.parametrize(
    "payload",
    [
        {"event_id": "hidden"},
        {"event_ids": ("hidden",)},
        {"eventId": "hidden"},
        {"targetEventIds": ("hidden",)},
        {"event_identifier": "hidden"},
        {"earthquake_id": "hidden"},
        {"physical_event_ids": ("hidden",)},
        {"lat": 35.0},
        {"lon": 105.0},
        {"lng": 105.0},
        {"location": [105.0, 35.0]},
        {"centroid": [105.0, 35.0]},
        {"geojson": {"type": "Point"}},
        {"geometry_wkb_hex": "010100000000000000000000000000000000000000"},
        {"geometryWkbHex": "010100000000000000000000000000000000000000"},
        {"crs": "EPSG:4326"},
        {"srid": 4326},
        {"payload": "POINT (105 35)"},
        {"payload": "LINESTRING Z (105 35 0, 106 36 0)"},
        {"payload": "SRID=4326;MULTIPOINT EMPTY"},
    ],
)
def test_public_payload_guard_rejects_event_identity_and_common_geometry_forms(
    payload: object,
) -> None:
    with pytest.raises(ValueError, match="spatial/event field|WKT geometry"):
        publication_module._assert_public_payload(payload)


def test_public_payload_guard_allows_counts_hashes_and_explicit_false_flags() -> None:
    publication_module._assert_public_payload(
        {
            "target_event_count": 3,
            "maximum_events_per_replicate": 100_000,
            "score_id": "a" * 64,
            "support_id": "local-support-0123456789abcdef",
            "earthquake_dataset": "b" * 64,
            "earthquake_event_count": 17,
            "contains_geometry": False,
            "contains_coordinates": False,
            "public_geometry_included": False,
        }
    )


def test_registry_serialization_normalizes_aware_datetime_and_rejects_naive() -> None:
    decoded = json.loads(
        publication_module._registry_bytes(
            {"science": {"support": {"fit_end_utc": datetime(2024, 1, 2, tzinfo=UTC)}}}
        )
    )
    assert decoded["science"]["support"]["fit_end_utc"] == "2024-01-02T00:00:00.000000Z"

    with pytest.raises(ValueError, match="timezone-aware"):
        publication_module._registry_bytes(
            {"science": {"support": {"fit_end_utc": datetime(2024, 1, 2)}}}
        )


def test_visual_preserves_one_sided_sensitivity_and_not_evaluable_gate() -> None:
    snapshots = tuple(
        LocalSupportVisualSnapshot(snapshot_id, 0.97, None, None)
        for snapshot_id in FROZEN_PRIMARY_INTERVALS
    )
    horizons = tuple(
        LocalSupportVisualHorizon(horizon, None, None) for horizon in (7, 30, 90, 180, 365)
    )
    bootstrap = tuple(
        LocalSupportVisualBootstrap(model_id, None, None, None)
        for model_id in ("spatial_poisson", "etas")
    )
    sensitivity = (
        LocalSupportVisualSensitivity("fold_1", 0.125, None),
        LocalSupportVisualSensitivity("fold_3", None, -0.25),
    )

    svg = render_local_support_results_svg(
        snapshots=snapshots,
        horizons=horizons,
        bootstrap=bootstrap,
        sensitivity=sensitivity,
        g1_ls_status="not_evaluable",
        selected_model_variant_id="not_selected",
    ).decode("utf-8")

    assert "G1-LS 未评估" in svg
    assert "G1-LS 未通过" not in svg
    assert "+0.1250" in svg
    assert "-0.2500" in svg
    assert svg.count("未评分") >= 2


def test_publication_rebuilds_inside_seal_and_validates_exact_score_references(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    score_ids = ("score-a", "score-b")
    ledger = SimpleNamespace(score_ids=score_ids)

    class CompletedOutcome:
        protocol_sha256 = "a" * 64
        g1_ls_passed = True
        stage3_allowed = True
        score_ledger = ledger

    outcome = cast(LocalSupportStage2R1Result, CompletedOutcome())
    repository = SimpleNamespace(code_commit="b" * 40)
    seal = SimpleNamespace(
        seal_id="4" * 64,
        repository=repository,
        input_hash_mapping=lambda: {"support_manifest": "c" * 64},
    )
    authorization = SimpleNamespace(
        scoring_code_tag="v0.2.1-background-local-support-scoring-code-r1",
        scoring_code_tag_object="d" * 40,
        remote_repository="github.com/Justin-147/SeismoFlux",
    )
    authorized = cast(
        AuthorizedExecution,
        SimpleNamespace(
            execution_seal=seal,
            scoring_authorization=authorization,
            authorization_id="e" * 64,
        ),
    )
    config = cast(
        BackgroundConfig,
        SimpleNamespace(
            inputs=SimpleNamespace(
                environment_lock_sha256="f" * 64,
                support_manifest_sha256="1" * 64,
            ),
            freeze_tag="v0.2.1-background-local-support-protocol",
            outputs=SimpleNamespace(
                registry="reports/registry.json",
                report="reports/report.md",
            ),
        ),
    )
    runtime = cast(LocalSupportRuntime, SimpleNamespace(manifest_id="support-id"))
    adapted = LocalSupportDeliverables(
        processed=cast(ProcessedBundle, object()),
        model=cast(ModelBundle, object()),
        backtest=cast(BacktestBundle, object()),
        experiment=cast(ExperimentBundle, object()),
        registry_science={
            "g1_ls_and_selection": {"g1_ls": {"passed": True, "status": "passed"}},
            "score_ledger": {"score_ids": score_ids},
        },
        results_svg=b"<svg/>",
    )
    trace: list[str] = []

    monkeypatch.setattr(
        publication_module,
        "LocalSupportStage2R1Result",
        CompletedOutcome,
    )
    monkeypatch.setattr(
        publication_module,
        "require_background_scoring_authorized",
        lambda *_: trace.append("authorized"),
    )
    monkeypatch.setattr(
        publication_module,
        "require_execution_seal_unchanged",
        lambda *_, **__: trace.append("reseal"),
    )

    def rebuild(*values: object) -> LocalSupportDeliverables:
        assert values == (config, runtime, outcome, authorized)
        trace.append("rebuild")
        return adapted

    monkeypatch.setattr(publication_module, "build_local_support_deliverables", rebuild)

    def validate(actual: object, references: object) -> None:
        assert actual is ledger
        assert tuple(cast(Any, references)) == score_ids
        trace.append("validate_scores")

    monkeypatch.setattr(
        publication_module,
        "validate_registry_score_references",
        validate,
    )

    def bundle(kind: str) -> Any:
        return SimpleNamespace(
            bundle_kind=kind,
            artifact=SimpleNamespace(
                artifact_id=f"artifact-{kind}",
                manifest_sha256="2" * 64,
            ),
        )

    monkeypatch.setattr(
        publication_module,
        "publish_processed_bundle",
        lambda *_, **__: bundle("processed"),
    )
    monkeypatch.setattr(
        publication_module,
        "publish_model_bundle",
        lambda *_, **__: bundle("model"),
    )
    monkeypatch.setattr(
        publication_module,
        "publish_backtest_bundle",
        lambda *_, **__: bundle("backtest"),
    )
    monkeypatch.setattr(
        publication_module,
        "publish_experiment_bundle",
        lambda *_, **__: bundle("experiment"),
    )
    monkeypatch.setattr(publication_module, "_assert_public_payload", lambda *_, **__: None)
    monkeypatch.setattr(publication_module, "_registry_bytes", lambda _: b"registry")
    monkeypatch.setattr(publication_module, "_report_bytes", lambda _: b"report")

    def fixed(_: Path, path: str, payload: bytes) -> Any:
        trace.append(f"fixed:{path}:{payload.decode('utf-8') if payload != b'<svg/>' else 'svg'}")
        return SimpleNamespace(path=tmp_path / path, sha256="3" * 64, created=True)

    monkeypatch.setattr(publication_module, "publish_fixed_project_file", fixed)

    published = publication_module.publish_local_support_deliverables(
        tmp_path,
        config,
        runtime,
        outcome,
        authorized,
    )

    assert trace[:5] == [
        "authorized",
        "reseal",
        "rebuild",
        "validate_scores",
        "reseal",
    ]
    assert trace.count("reseal") == 3
    assert trace[-3:] == [
        "fixed:docs/background_local_support_results.svg:svg",
        "fixed:reports/registry.json:registry",
        "fixed:reports/report.md:report",
    ]
    assert published.registry_payload["stage3_allowed"] is True
    assert published.registry_payload["scoring_code_tag"] == authorization.scoring_code_tag
    assert published.registry_payload["science"] is adapted.registry_science


def test_real_model_summary_and_human_report_render_all_five_snapshots_and_sensitivity() -> None:
    pairs: list[PairedInformationGainEvidence] = []
    for snapshot_id, (
        fit_end,
        assessment_start,
        assessment_end,
    ) in FROZEN_PRIMARY_INTERVALS.items():
        common = {
            "protocol_sha256": "a" * 64,
            "parameter_snapshot_id": f"parameters/{snapshot_id}",
            "snapshot_id": snapshot_id,
            "fit_end_utc": fit_end,
            "assessment_start_utc": assessment_start,
            "assessment_end_utc": assessment_end,
            "selected_mc": 4.0,
            "target_event_ids": (f"event-{snapshot_id}",),
            "compensator": 1.0,
            "numerical_gate_evidence_ids": ("b" * 64,),
            "support_id": "local-support-0123456789abcdef",
            "supported_area_km2": 9_500_000.0,
            "compensator_domain_id": "c" * 64,
            "authorization_id": "d" * 64,
        }
        uniform = PointProcessScoreEvidence(
            model_id="uniform_poisson",
            model_variant_id="uniform_poisson/spatial_uniform_v1",
            event_log_intensities=np.asarray([-4.0]),
            **common,  # type: ignore[arg-type]
        )
        candidate = PointProcessScoreEvidence(
            model_id="spatial_poisson",
            model_variant_id="spatial_poisson/gaussian_kde_bw75km",
            event_log_intensities=np.asarray([-3.0]),
            **common,  # type: ignore[arg-type]
        )
        pairs.append(PairedInformationGainEvidence.build(candidate=candidate, uniform=uniform))
    evidence = AuditedBackgroundModelEvidence(
        model_id="spatial_poisson",
        model_variant_id="spatial_poisson/gaussian_kde_bw75km",
        protocol_sha256="a" * 64,
        development_folds=tuple(pairs[:4]),
        validation=pairs[4],
        failed_snapshot_reasons=(),
    )
    summary = publication_module._model_evidence_summary(evidence)
    assert len(cast(list[object], summary["snapshots"])) == 5

    support_snapshots = [
        {
            "snapshot_id": snapshot_id,
            "common_mc": 4.0,
            "supported_area_fraction": 0.97,
            "supported_area_km2": 9_500_000.0,
            "support_id": "local-support-0123456789abcdef",
        }
        for snapshot_id in FROZEN_PRIMARY_INTERVALS
    ]
    registry: dict[str, object] = {
        "protocol_version": "0.2.1",
        "stage3_allowed": False,
        "authorization_id": "d" * 64,
        "execution_seal_id": "e" * 64,
        "code_commit": "f" * 40,
        "scoring_code_tag": "v0.2.1-background-local-support-scoring-code-r1",
        "reserve_physical_cores": 2,
        "science": {
            "outcome_status": "completed",
            "failure": None,
            "support": {"snapshots": support_snapshots},
            "g1_ls_and_selection": {"g1_ls": {"passed": False, "status": "failed"}},
            "score_ledger": {"ledger_id": "ledger", "score_count": 10},
            "model_evidence": (summary,),
            "etas_parent_sensitivity": (
                {
                    "snapshot_id": "fold_1",
                    "status": "completed",
                    "primary_includes_eligible_unsupported_parents": {
                        "information_gain_nats_per_event": 0.1
                    },
                    "exclude_all_unsupported_parents": {"information_gain_nats_per_event": 0.08},
                    "information_gain_difference_exclude_minus_primary": -0.02,
                },
                {
                    "snapshot_id": "fold_3",
                    "status": "completed",
                    "primary_includes_eligible_unsupported_parents": {
                        "information_gain_nats_per_event": 0.03
                    },
                    "exclude_all_unsupported_parents": {"information_gain_nats_per_event": 0.04},
                    "information_gain_difference_exclude_minus_primary": 0.01,
                },
            ),
        },
    }
    report = publication_module._report_bytes(registry).decode("utf-8")
    assert "spatial_poisson" in report
    assert "ETAS unsupported 父历史敏感性" in report
    assert "fold_1" in report and "fold_3" in report
    assert "-0.02000" in report and "+0.01000" in report

    g1 = cast(
        dict[str, object], cast(dict[str, object], registry["science"])["g1_ls_and_selection"]
    )
    cast(dict[str, object], g1["g1_ls"])["status"] = "not_evaluable"
    negative_report = publication_module._report_bytes(registry).decode("utf-8")
    assert "G1-LS：`未评估（门前停止）`" in negative_report
    assert "G1-LS：`未通过`" not in negative_report
