from __future__ import annotations

import dataclasses
import inspect
import json
import re
from datetime import UTC, datetime, timedelta
from typing import cast

import numpy as np
import pytest

from seismoflux.anomaly_increment import spatial_dashboard
from seismoflux.anomaly_increment.spatial_dashboard import (
    DisplayContextLayer,
    DisplayRole,
    DisplayStudyArea,
    ForecastGridFrame,
    ForecastStatus,
    RetrospectiveTargetFrame,
    build_local_spatial_dashboard_html,
)


def _study_area() -> DisplayStudyArea:
    return DisplayStudyArea(
        study_area_id="frozen-study-area",
        geometry_geojson={
            "type": "Polygon",
            "coordinates": [[[99.0, 29.0], [102.0, 29.0], [102.0, 32.0], [99.0, 29.0]]],
        },
        source_content_sha256="9" * 64,
    )


def _field(
    issue: str,
    *,
    model_version: str = "v0.3.0",
    grid_id: str = "synthetic-grid-25km",
    status: ForecastStatus = "retrospective_evaluation",
    display_role: DisplayRole = "primary",
) -> ForecastGridFrame:
    target_blind = status != "retrospective_evaluation"
    return ForecastGridFrame(
        issue_date=issue,
        magnitude_bin="M5_6",
        horizon_days=7,
        model_variant="dynamic",
        model_version=model_version,
        training_sample_size=24,
        grid_id=grid_id,
        cell_size_km=25.0,
        alarm_threshold_rank_percentile=90.0,
        selected_alarm_area_km2=625.0,
        displayed_domain_area_km2=1_250.0,
        cell_ids=("a", "b"),
        longitude=np.asarray([100.0, 101.0]),
        latitude=np.asarray([30.0, 31.0]),
        relative_strength=np.asarray([0.3, 1.7]),
        rank_percentile=np.asarray([25.0, 95.0]),
        alarm_selected=np.asarray([False, True], dtype=np.bool_),
        forecast_status=status,
        evidence_status=("not_evaluated_target_blind" if target_blind else "passed"),
        display_role=display_role,
        knowledge_cutoff_utc=datetime.fromisoformat(f"{issue}T00:00:00+08:00").astimezone(UTC),
        source_model_fingerprint_sha256="8" * 64,
        source_result_fingerprint_sha256=None if target_blind else "a" * 64,
    )


def _target(issue: str = "2025-01-01") -> RetrospectiveTargetFrame:
    return RetrospectiveTargetFrame(
        issue_date=issue,
        magnitude_bin="M5_6",
        horizon_days=7,
        event_ids=("target-1",),
        longitude=np.asarray([100.5]),
        latitude=np.asarray([30.5]),
        covered_by_600000_km2=(True,),
        hit_event_ids_at_600000_km2=("target-1",),
        source_catalog_content_sha256="b" * 64,
        source_result_fingerprint_sha256="a" * 64,
    )


def _context_layer() -> DisplayContextLayer:
    return DisplayContextLayer(
        layer_id="fault-1",
        label="登记断层",
        layer_kind="fault",
        geometry_geojson={
            "type": "LineString",
            "coordinates": [[99.5, 29.5], [101.5, 31.5]],
        },
        source_content_sha256="c" * 64,
    )


def _payload(document: str, payload_id: str) -> dict[str, object]:
    matched = re.search(
        rf'<script type="application/json" id="{re.escape(payload_id)}">(.*?)</script>',
        document,
        flags=re.DOTALL,
    )
    assert matched is not None
    decoded = json.loads(matched.group(1))
    if not isinstance(decoded, dict):
        raise AssertionError(f"{payload_id} must contain a JSON object")
    return cast(dict[str, object], decoded)


def _dashboard(
    *,
    prospective: ForecastGridFrame | None = None,
    retrospective: ForecastGridFrame | None = None,
    targets: tuple[RetrospectiveTargetFrame, ...] | None = None,
    context_layers: tuple[DisplayContextLayer, ...] = (),
) -> str:
    return build_local_spatial_dashboard_html(
        study_area=_study_area(),
        retrospective_fields=(retrospective or _field("2025-01-01"),),
        prospective_fields=(
            prospective
            or _field(
                "2026-01-01",
                status="retrospective_generated_target_blind_shadow",
            ),
        ),
        retrospective_targets=targets if targets is not None else (_target(),),
        display_context_layers=context_layers,
    )


def test_spatial_dashboard_physically_separates_forecast_and_target_payloads() -> None:
    document = _dashboard(context_layers=(_context_layer(),))
    prospective_payload = _payload(document, "prospective-spatial-payload")
    retrospective_fields = _payload(document, "retrospective-spatial-fields-payload")
    target_overlay = _payload(document, "retrospective-target-overlay-payload")
    display_context = _payload(document, "display-context-payload")

    prospective_serialized = json.dumps(prospective_payload, sort_keys=True).casefold()
    retrospective_serialized = json.dumps(retrospective_fields, sort_keys=True).casefold()
    assert '"targets"' not in prospective_serialized
    assert "covered_by" not in prospective_serialized
    assert "event_id" not in prospective_serialized
    assert "geometry" not in prospective_serialized
    assert '"targets"' not in retrospective_serialized
    assert "covered_by" not in retrospective_serialized
    assert set(prospective_payload) == {"fields"}
    assert set(retrospective_fields) == {"fields"}
    assert set(target_overlay) == {"targets"}
    assert set(display_context) == {
        "layers",
        "status_label",
        "study_area",
        "study_area_geometry_sha256",
        "study_area_id",
        "study_area_role",
        "study_area_source_content_sha256",
    }
    targets = cast(list[dict[str, object]], target_overlay["targets"])
    assert targets[0]["covered_by_600000_km2"] == [True]
    assert targets[0]["event_ids"] == ["target-1"]
    assert targets[0]["hit_event_ids_at_600000_km2"] == ["target-1"]
    layers = cast(list[dict[str, object]], display_context["layers"])
    assert layers[0]["role"] == "display_only_forbidden_from_model_or_candidate_generation"

    fields = cast(list[dict[str, object]], prospective_payload["fields"])
    assert fields[0]["grid_id"] == "synthetic-grid-25km"
    assert fields[0]["cell_size_km"] == 25.0
    assert fields[0]["selected_alarm_area_km2"] == 625.0
    assert fields[0]["displayed_domain_area_km2"] == 1_250.0
    assert fields[0]["evidence_status"] == "not_evaluated_target_blind"
    assert fields[0]["evidence_status_label"] == "目标盲帧不携带回溯证据"
    assert fields[0]["source_model_fingerprint_sha256"] == "8" * 64
    assert "source_result_fingerprint_sha256" not in fields[0]
    assert fields[0]["alarm_threshold_rank_percentile"] == 90.0
    assert fields[0]["alarm_selected"] == [False, True]


def test_spatial_dashboard_defaults_to_shadow_and_loads_targets_lazily() -> None:
    document = _dashboard()
    assert '<option value="prospective" selected>' in document
    assert '<span id="targetLegend" hidden>' in document
    assert 'controls.mode.value==="retrospective"' in document
    assert (
        'const pros=JSON.parse(document.getElementById("prospective-spatial-payload")' in document
    )
    assert "let retroFieldsCache=null,targetCache=null" in document
    assert "function retrospectiveFields()" in document
    assert "function retrospectiveTargets()" in document
    assert document.count('getElementById("retrospective-target-overlay-payload")') == 1
    assert "targetCache=JSON.parse" in document
    assert "回溯生成的目标盲影子预测" in document
    assert "真正前瞻归档" not in document
    assert "滚轮缩放" in document
    assert "拖动平移" in document
    assert "不能解释为绝对发生率估计" in document
    assert "概率" not in document
    assert "上下文层未提供" in document


def test_spatial_dashboard_exposes_controls_metadata_and_offline_assets() -> None:
    document = _dashboard(context_layers=(_context_layer(),))
    for control_id in ("mode", "issue", "magnitude", "horizon", "variant", "version"):
        assert f'id="{control_id}"' in document
    for label in (
        "报警面积",
        "展示域",
        "顺位阈值",
        "相对强度峰值",
        "证据等级",
        "网格",
        "样本",
        "模型",
        "知识截止",
    ):
        assert label in document
    for interaction in (
        "wheel",
        "pointerdown",
        "pointermove",
        'ctx.clip("evenodd")',
        "contextPaths",
        "previousFrame",
        "ctx.setLineDash([4,3])",
    ):
        assert interaction in document
    assert "显示专用上下文" in document
    assert "登记断层" in document
    assert "--map-outside:#fff" in document
    assert 'data-classification="local_restricted_target_bearing_spatial_visualization"' in document
    assert "fetch(" not in document
    assert "xmlhttprequest" not in document.casefold()
    assert "websocket" not in document.casefold()
    assert "<script src=" not in document.casefold()
    assert "http://" not in document.casefold()
    assert "https://" not in document.casefold()


def test_spatial_dashboard_embedded_json_is_script_safe_and_reversible() -> None:
    unsafe_version = "safe</script><script>alert(1)</script>&\u2028end"
    document = _dashboard(
        prospective=_field(
            "2026-01-01",
            model_version=unsafe_version,
            status="retrospective_generated_target_blind_shadow",
        )
    )
    assert unsafe_version not in document
    assert "\\u003c/script\\u003e\\u003cscript\\u003ealert(1)" in document
    assert "\\u0026\\u2028end" in document
    assert document.count("</script>") == 5
    payload = _payload(document, "prospective-spatial-payload")
    fields = cast(list[dict[str, object]], payload["fields"])
    assert fields[0]["model_version"] == unsafe_version


def test_forecast_grid_frame_rejects_shape_rank_and_alarm_drift() -> None:
    field = _field("2026-01-01")
    with pytest.raises(ValueError, match="one value per cell"):
        dataclasses.replace(
            field,
            longitude=np.asarray([100.0]),
            latitude=np.asarray([30.0]),
        )
    with pytest.raises(ValueError, match="rank percentiles"):
        dataclasses.replace(field, rank_percentile=np.asarray([25.0, 101.0]))
    with pytest.raises(ValueError, match="below the declared rank threshold"):
        dataclasses.replace(
            field,
            alarm_selected=np.asarray([True, False], dtype=np.bool_),
        )
    with pytest.raises(ValueError, match="empty alarm prefix requires zero"):
        dataclasses.replace(
            field,
            alarm_selected=np.asarray([False, False], dtype=np.bool_),
        )
    with pytest.raises(TypeError, match="boolean vector"):
        dataclasses.replace(field, alarm_selected=np.asarray([0, 1]))
    issue_time = datetime.fromisoformat("2026-01-01T00:00:00+08:00").astimezone(UTC)
    with pytest.raises(ValueError, match="cannot pass"):
        dataclasses.replace(
            field,
            knowledge_cutoff_utc=issue_time + timedelta(seconds=1),
        )

    target_blind = _field(
        "2026-01-01",
        status="retrospective_generated_target_blind_shadow",
    )
    with pytest.raises(ValueError, match="target-derived evidence"):
        dataclasses.replace(target_blind, evidence_status="passed")
    with pytest.raises(ValueError, match="target-derived result fingerprint"):
        dataclasses.replace(
            target_blind,
            source_result_fingerprint_sha256="a" * 64,
        )


def test_spatial_dashboard_rejects_invalid_area_unmatched_targets_and_status_mix() -> None:
    with pytest.raises(ValueError, match="rings must be closed"):
        DisplayStudyArea(
            study_area_id="invalid-study-area",
            geometry_geojson={
                "type": "Polygon",
                "coordinates": [[[99.0, 29.0], [102.0, 29.0], [102.0, 32.0], [99.0, 30.0]]],
            },
            source_content_sha256="9" * 64,
        )
    with pytest.raises(ValueError, match="valid and positive-area"):
        DisplayStudyArea(
            study_area_id="degenerate-study-area",
            geometry_geojson={
                "type": "Polygon",
                "coordinates": [[[99.0, 29.0], [100.0, 30.0], [101.0, 31.0], [99.0, 29.0]]],
            },
            source_content_sha256="9" * 64,
        )
    with pytest.raises(ValueError, match="no matching forecast frame"):
        _dashboard(targets=(_target("2025-02-01"),))
    with pytest.raises(ValueError, match="forecast status"):
        build_local_spatial_dashboard_html(
            study_area=_study_area(),
            retrospective_fields=(_field("2025-01-01"),),
            prospective_fields=(_field("2026-01-01"),),
            retrospective_targets=(_target(),),
        )
    with pytest.raises(ValueError, match="different fitted models"):
        _dashboard(
            prospective=dataclasses.replace(
                _field(
                    "2026-01-01",
                    status="retrospective_generated_target_blind_shadow",
                ),
                source_model_fingerprint_sha256="d" * 64,
            )
        )


def test_spatial_dashboard_only_labels_true_archive_when_explicit_and_has_no_mojibake() -> None:
    genuine = _dashboard(
        prospective=_field(
            "2026-01-01",
            status="genuine_prospective_archive",
        )
    )
    assert "真正前瞻归档" in genuine
    assert "回溯生成的目标盲影子预测" not in genuine

    document = _dashboard()
    source = inspect.getsource(spatial_dashboard)
    for mojibake in ("婊氳", "鎷栧", "鍥炴函", "闈㈢Н", "椤轰綅"):
        assert mojibake not in document
        assert mojibake not in source
    assert source.encode("utf-8").decode("utf-8") == source
    assert '<meta charset="utf-8"/>' in document


def test_context_and_forecast_marks_share_study_area_clip_before_outline() -> None:
    document = _dashboard(context_layers=(_context_layer(),))
    draw = document[document.index("function draw()") : document.index("Object.values(controls)")]
    assert draw.index('ctx.clip("evenodd")') < draw.index("drawContextLayers()")
    assert draw.index("drawContextLayers()") < draw.index("for(let i=0;i<f.longitude.length")
    assert draw.index("ctx.restore()") < draw.index("drawOutline()")
    assert "上一期" in document
    assert "无上一期" in document
