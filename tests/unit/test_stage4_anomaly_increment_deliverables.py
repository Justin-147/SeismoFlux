from __future__ import annotations

import dataclasses
import inspect
import json
import math
import re
import xml.etree.ElementTree as ET
from typing import cast

import pytest

import seismoflux.anomaly_increment.deliverables as deliverables_module
from seismoflux.anomaly_increment.deliverables import (
    ForecastStatus,
    ModelVariant,
    ProspectivePayload,
    ProspectiveRelativeRank,
    ResultStatus,
    RetrospectiveHorizonResult,
    RetrospectivePayload,
    RetrospectiveTargetCoverage,
    assert_prospective_payload_is_target_blind,
    build_interactive_results_html,
    build_static_results_svg,
    default_protocol_display,
)
from seismoflux.anomaly_increment.evaluation import GateStatus
from seismoflux.anomaly_increment.publication_diagnostics import (
    PUBLICATION_ALARM_BUDGETS_KM2,
    AlarmBudgetRecallCurve,
    AlarmBudgetRecallPoint,
    CoefficientEffectCurve,
    DataMethodFlowDiagnostics,
    DistanceLeadDecayDiagnostics,
    FoldMacroValue,
    IncrementVariant,
    InformationGainInterval,
    PermutationDistribution,
    PublicationDiagnostics,
    RegionHorizonMetric,
    SameRecallAreaReduction,
)
from seismoflux.anomaly_increment.publication_diagnostics import (
    MagnitudeBin as PublicationMagnitudeBin,
)
from seismoflux.anomaly_increment.publication_diagnostics import (
    ModelVariant as PublicationModelVariant,
)


def _retrospective(status: ResultStatus) -> RetrospectivePayload:
    gate_status: GateStatus = status
    primary_value = {"passed": 0.2, "failed": -0.1, "evidence_insufficient": None}[status]
    primary_result_status = status
    return RetrospectivePayload(
        protocol=default_protocol_display(model_version="synthetic-stage4-code-freeze"),
        outcome_status=status,
        g2_status=gate_status,
        g3_status=gate_status,
        adoption="dynamic" if status == "passed" else "background_only",
        issue_dates=("2025-01-02", "2025-01-09"),
        horizon_results=tuple(
            RetrospectiveHorizonResult(
                horizon,
                "M5_6",
                primary_result_status if horizon in {7, 30, 90} else "evidence_insufficient",
                8 if horizon in {7, 30, 90} else None,
                primary_value if horizon in {7, 30, 90} else None,
                (primary_value - 0.05)
                if primary_value is not None and horizon in {7, 30, 90}
                else None,
                (primary_value + 0.05)
                if primary_value is not None and horizon in {7, 30, 90}
                else None,
                0.5 if horizon in {7, 30, 90} else None,
            )
            for horizon in (7, 30, 90, 180, 365)
        ),
        target_coverage=tuple(
            RetrospectiveTargetCoverage(horizon, 8, 4) for horizon in (7, 30, 90, 180, 365)
        ),
        time_permutation_p_value=0.01 if status == "passed" else None,
        space_permutation_p_value=0.03 if status == "passed" else None,
    )


def _prospective(
    *,
    rank_band: str = "top-10-percent",
    forecast_status: ForecastStatus = "retrospective_generated_target_blind_shadow",
) -> ProspectivePayload:
    protocol = default_protocol_display(model_version="synthetic-stage4-code-freeze")
    variants: tuple[ModelVariant, ...] = (
        "background_no_increment",
        "snapshot",
        "dynamic",
    )
    return ProspectivePayload(
        protocol=protocol,
        forecast_status=forecast_status,
        issue_dates=("2026-07-16",),
        relative_ranks=tuple(
            ProspectiveRelativeRank(
                "2026-07-16",
                "M5_6",
                horizon,
                variant,
                rank_band,
                relative_strength_index=1.35,
                rank_percentile=92.5,
            )
            for horizon in (7, 30, 90, 180, 365)
            for variant in variants
        ),
    )


def _diagnostics() -> PublicationDiagnostics:
    horizons = (7, 30, 90, 180, 365)
    regions = tuple(f"construction-zone-{index:02d}" for index in range(1, 40))
    magnitude_bins: tuple[PublicationMagnitudeBin, ...] = ("M5_6", "M6_plus")
    curve_variants: tuple[PublicationModelVariant, ...] = (
        "background_no_increment",
        "dynamic",
    )
    effect_specs: tuple[tuple[IncrementVariant, float], ...] = (
        ("coverage_only", 0.12),
        ("snapshot", 0.21),
        ("dynamic", 0.33),
    )
    intervals = tuple(
        InformationGainInterval(
            magnitude_bin=magnitude_bin,
            horizon_days=horizon,
            independent_event_count=12 if magnitude_bin == "M5_6" else 3,
            evidence_status=(
                (
                    "exploratory_low_sample"
                    if horizon in {7, 30, 90}
                    else "exploratory_low_sample_no_random_split"
                )
                if magnitude_bin == "M6_plus"
                else (
                    "evaluated"
                    if horizon in {7, 30, 90}
                    else "evidence_insufficient_no_random_split"
                )
            ),
            point_estimate=0.22 - 0.0003 * horizon - magnitude_index * 0.04,
            lower_95=0.12 - 0.0003 * horizon - magnitude_index * 0.04,
            upper_95=0.31 - 0.0003 * horizon - magnitude_index * 0.04,
        )
        for magnitude_index, magnitude_bin in enumerate(magnitude_bins)
        for horizon in horizons
    )
    curves = tuple(
        AlarmBudgetRecallCurve(
            variant=variant,
            magnitude_bin="M5_6",
            horizon_days=horizon,
            points=tuple(
                AlarmBudgetRecallPoint(
                    budget_km2=budget,
                    selected_alarm_area_km2=budget - 5_000.0,
                    strict_recall=min(
                        0.98,
                        base + (0.08 if variant == "dynamic" else 0.0),
                    ),
                    all_study_area_event_count=12,
                    evidence_status="evaluated",
                )
                for budget, base in zip(
                    PUBLICATION_ALARM_BUDGETS_KM2,
                    (0.12, 0.24, 0.36, 0.5, 0.64),
                    strict=True,
                )
            ),
        )
        for variant in curve_variants
        for horizon in (7, 30, 90)
    )
    return PublicationDiagnostics(
        data_flow=DataMethodFlowDiagnostics(84, 12_480, 18, 37, 9_600_000.0),
        coefficient_effects=tuple(
            CoefficientEffectCurve(
                variant,
                f"{variant}_signal",
                coefficient,
                (-1.0, 0.0, 1.0),
                (-coefficient, 0.0, coefficient),
            )
            for variant, coefficient in effect_specs
        ),
        distance_lead_decay=DistanceLeadDecayDiagnostics(
            (0.0, 100.0, 200.0, 400.0),
            (1.0, 0.78, 0.37, 0.02),
            (0.0, 30.0, 90.0, 180.0),
            (1.0, 0.79, 0.5, 0.25),
        ),
        fold_macro_values=(
            FoldMacroValue(1, (4, 5, 6), "evaluated", 0.08, 0.03),
            FoldMacroValue(2, (5, 5, 7), "evaluated", 0.11, 0.05),
            FoldMacroValue(3, (3, 4, 5), "evaluated", 0.06, 0.02),
        ),
        permutation_distributions=(
            PermutationDistribution(
                "dynamic",
                "time",
                0.15,
                tuple(-0.2 + index * 0.00035 for index in range(1_000)),
            ),
            PermutationDistribution(
                "dynamic",
                "space",
                0.14,
                tuple(
                    math.inf if index == 999 else -0.24 + index * 0.00035 for index in range(1_000)
                ),
            ),
            PermutationDistribution(
                "snapshot",
                "time",
                0.09,
                tuple(-0.18 + index * 0.00035 for index in range(1_000)),
            ),
        ),
        information_gain_intervals=intervals,
        region_ids=regions,
        region_horizon_metrics=tuple(
            RegionHorizonMetric(
                region_id=region,
                horizon_days=horizon,
                information_gain_nats_per_event=(index % 7 - 3) * 0.015,
                supported_event_count=2,
                all_study_area_event_count=3,
                strict_recall=min(1.0, 0.2 + index * 0.01),
                information_gain_evidence_status="evaluated",
                strict_recall_evidence_status="evaluated",
            )
            for index, region in enumerate(regions)
            for horizon in horizons
        ),
        alarm_budget_curves=curves,
        same_recall_area_reductions=tuple(
            SameRecallAreaReduction(
                "M5_6",
                horizon,
                0.5,
                "background_no_increment",
                "dynamic",
                750_000.0,
                600_000.0 - index * 15_000.0,
                0.1,
                0.35,
            )
            for index, horizon in enumerate((7, 30, 90))
        ),
    )


def _json_payload(document: str, payload_id: str) -> dict[str, object]:
    matched = re.search(
        rf'<script type="application/json" id="{re.escape(payload_id)}">(.*?)</script>',
        document,
        flags=re.DOTALL,
    )
    assert matched is not None
    result = json.loads(matched.group(1))
    if not isinstance(result, dict):
        raise AssertionError(f"{payload_id} JSON must be an object")
    return cast(dict[str, object], result)


def _prospective_json(document: str) -> dict[str, object]:
    return _json_payload(document, "prospective-payload")


def test_static_svg_is_valid_coordinate_free_and_displays_required_boundaries() -> None:
    diagnostics = _diagnostics()
    svg = build_static_results_svg(_retrospective("passed"), diagnostics)
    root = ET.fromstring(svg)
    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    assert root.attrib["viewBox"] == "0 0 1200 1040"
    assert root.attrib["data-template"] == "stage4-static-nine-panel-v2"
    assert root.attrib["data-diagnostics-sha256"] == diagnostics.content_sha256
    expected_panels = (
        "data_and_model_flow",
        "coefficient_and_effect_curves",
        "distance_and_lead_decay",
        "foldwise_dynamic_vs_snapshot",
        "observed_vs_time_and_space_permutation",
        "information_gain_with_bootstrap_intervals",
        "region_by_horizon_increment_heatmap",
        "molchan_curve",
        "fixed_area_recall_curve",
    )
    actual_panels = tuple(
        element.attrib["data-panel"] for element in root.iter() if "data-panel" in element.attrib
    )
    assert actual_panels == expected_panels
    point_counts = {
        element.attrib["data-panel"]: int(element.attrib["data-point-count"])
        for element in root.iter()
        if "data-panel" in element.attrib
    }
    assert point_counts == {
        "data_and_model_flow": 5,
        "coefficient_and_effect_curves": 9,
        "distance_and_lead_decay": 8,
        "foldwise_dynamic_vs_snapshot": 6,
        "observed_vs_time_and_space_permutation": 3_003,
        "information_gain_with_bootstrap_intervals": 10,
        "region_by_horizon_increment_heatmap": 195,
        "molchan_curve": 30,
        "fixed_area_recall_curve": 33,
    }
    assert svg.count('class="observed"') == 3
    assert svg.count('data-region-alias="R') == 195
    assert svg.count('data-null-count="1000"') == 0
    assert 'data-dynamic-time-null-count="1000"' in svg
    assert 'data-dynamic-space-null-count="1000"' in svg
    assert 'data-snapshot-time-null-count="1000"' in svg
    assert 'data-region-count="39"' in svg
    assert 'data-budget-count="5"' in svg
    assert 'data-same-recall-count="3"' in svg
    assert svg.count('class="mark-exploratory"') == 5
    assert "M6+探索性 n&lt;10" in svg
    assert "条件相对强度与顺位" in svg
    assert "probability" not in svg.casefold()
    assert "概率" not in svg
    for forbidden in ("longitude", "latitude", "geojson", "cell_id", "query_x_m"):
        assert forbidden not in svg.casefold()
    for placeholder in ("registered", "未发布", "缺少点", "缺失点", "占位", "不伪造"):
        assert placeholder not in svg


def test_deliverables_source_has_one_renderer_and_no_legacy_placeholder_path() -> None:
    source = inspect.getsource(deliverables_module)

    assert source.count("def build_static_results_svg(") == 1
    assert "_legacy_static_results_svg" not in source
    assert "_information_gain_panel_body" not in source
    assert "_recall_panel_body" not in source
    assert "stage4-static-nine-panel-v1" not in source
    assert "registered" not in source.casefold()
    assert "已登记" not in source
    assert "完整曲线需" not in source
    assert "不伪造缺失点" not in source
    assert "\ufffd" not in source


def test_formal_renderers_fail_closed_without_publication_diagnostics() -> None:
    with pytest.raises(TypeError):
        build_static_results_svg(_retrospective("passed"))  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        build_interactive_results_html(  # type: ignore[call-arg]
            _retrospective("passed"),
            _prospective(),
        )


def test_static_svg_renders_real_evidence_insufficient_branches_without_fake_zeroes() -> None:
    diagnostics = _diagnostics()
    zero_curves = tuple(
        dataclasses.replace(
            curve,
            points=tuple(
                dataclasses.replace(
                    point,
                    strict_recall=None,
                    all_study_area_event_count=0,
                    evidence_status="evidence_insufficient_zero_all_events",
                )
                for point in curve.points
            ),
        )
        for curve in diagnostics.alarm_budget_curves
    )
    insufficient = dataclasses.replace(
        diagnostics,
        fold_macro_values=(
            FoldMacroValue(
                1,
                (2, 0, 3),
                "evidence_insufficient_zero_events",
                None,
                None,
            ),
            *diagnostics.fold_macro_values[1:],
        ),
        permutation_distributions=(
            PermutationDistribution(
                "dynamic",
                "time",
                None,
                (),
                "evidence_insufficient_zero_events",
            ),
            *diagnostics.permutation_distributions[1:],
        ),
        alarm_budget_curves=zero_curves,
        same_recall_area_reductions=tuple(
            dataclasses.replace(
                item,
                target_recall=0.0,
                candidate_area_km2=None,
                area_reduction_lower_95=None,
                area_reduction_upper_95=None,
                evidence_status="evidence_insufficient_zero_comparator_recall",
            )
            for item in diagnostics.same_recall_area_reductions
        ),
    )

    svg = build_static_results_svg(_retrospective("evidence_insufficient"), insufficient)

    assert "evidence_insufficient_zero_events" in svg
    assert "evidence_insufficient_zero_all_events" in svg
    assert "evidence_insufficient_zero_comparator_recall" in svg
    assert 'data-all-study-area-event-count="0"' in svg
    assert "证据不足" in svg


@pytest.mark.parametrize("status", ["passed", "failed", "evidence_insufficient"])
def test_positive_negative_and_insufficient_results_share_one_interactive_template(
    status: ResultStatus,
) -> None:
    document = build_interactive_results_html(
        _retrospective(status),
        _prospective(),
        _diagnostics(),
    )
    assert document.startswith("<!doctype html>")
    assert document.count('type="application/json"') == 2
    assert '<section id="retrospectivePanel" hidden' in document
    assert '<section id="prospectivePanel" aria-labelledby=' in document
    assert '<section id="prospectivePanel" hidden' not in document
    assert '<option value="prospective" selected>' in document
    assert 'id="retrospective-target-coverage"' in document
    assert 'id="mode"' in document
    assert 'id="horizon"' in document
    assert 'id="variant"' in document
    assert 'id="version"' in document
    assert 'id="retrospectiveDiagnostics"' in document
    assert "not_evaluable" in document
    assert document.count("evidence_insufficient_no_random_split") >= 2
    assert status in document
    assert "fetch(" not in document
    assert "xmlhttprequest" not in document.casefold()
    assert "websocket" not in document.casefold()
    assert "<script src=" not in document.casefold()
    assert document.casefold().count("http://") == 1
    assert 'xmlns="http://www.w3.org/2000/svg"' in document
    assert "https://" not in document.casefold()
    assert "probability" not in document.casefold()
    assert "概率" not in document


def test_interactive_template_structure_does_not_change_with_scientific_outcome() -> None:
    prospective = _prospective()
    id_sets = []
    for status in ("passed", "failed", "evidence_insufficient"):
        document = build_interactive_results_html(
            _retrospective(status),
            prospective,
            _diagnostics(),
        )
        id_sets.append(tuple(sorted(set(re.findall(r'\bid="([^"]+)"', document)))))
    assert id_sets[0] == id_sets[1] == id_sets[2]


def test_prospective_type_and_serialized_payload_physically_exclude_evaluation_fields() -> None:
    prospective = _prospective()
    assert_prospective_payload_is_target_blind(prospective)
    field_names = {field.name for field in dataclasses.fields(ProspectivePayload)}
    forbidden_parts = (
        "target",
        "hit",
        "recall",
        "information_gain",
        "p_value",
        "score",
        "event_count",
    )
    assert not any(part in field for field in field_names for part in forbidden_parts)

    document = build_interactive_results_html(
        _retrospective("passed"),
        prospective,
        _diagnostics(),
    )
    payload = _prospective_json(document)
    retrospective_payload = _json_payload(document, "retrospective-payload")
    serialized = json.dumps(payload, sort_keys=True).casefold()
    assert not any(f'"{part}' in serialized for part in forbidden_parts)
    assert "relative_strength_index" in serialized
    assert "rank_percentile" in serialized
    assert "target_coverage" not in payload
    retrospective_summary = cast(dict[str, object], retrospective_payload["summary"])
    assert "target_coverage" in retrospective_summary
    assert "diagnostics" in retrospective_payload
    assert '<script type="application/json" id="prospective-payload">' in document
    assert '<script type="application/json" id="retrospective-payload">' in document
    assert 'const pros = JSON.parse(document.getElementById("prospective-payload")' in document
    assert "let retroCache = null" in document
    assert "function retrospectivePayload()" in document
    assert 'const retro=JSON.parse(document.getElementById("retrospective-payload")' not in document


def test_target_blind_shadow_is_not_presented_as_a_genuine_prospective_forecast() -> None:
    shadow = build_interactive_results_html(
        _retrospective("passed"),
        _prospective(),
        _diagnostics(),
    )
    assert "回溯生成的目标盲影子预测" in shadow
    assert "真正前瞻预测" not in shadow
    payload = _prospective_json(shadow)
    assert payload["forecast_status"] == "retrospective_generated_target_blind_shadow"

    genuine = build_interactive_results_html(
        _retrospective("passed"),
        _prospective(forecast_status="genuine_prospective"),
        _diagnostics(),
    )
    assert "真正前瞻预测" in genuine
    assert "回溯生成的目标盲影子预测" not in genuine


def test_embedded_payload_escapes_script_termination_and_contains_no_restricted_geometry() -> None:
    unsafe_rank_band = "safe</script><script>unsafe&\u2028end"
    prospective = _prospective(rank_band=unsafe_rank_band)
    document = build_interactive_results_html(
        _retrospective("passed"),
        prospective,
        _diagnostics(),
    )
    assert unsafe_rank_band not in document
    assert "safe\\u003c/script\\u003e\\u003cscript\\u003eunsafe\\u0026\\u2028end" in document
    assert document.count("</script>") == 3
    payload = _prospective_json(document)
    rows = cast(list[dict[str, object]], payload["relative_ranks"])
    assert rows[0]["rank_band"] == unsafe_rank_band
    for forbidden in (
        "longitude",
        "latitude",
        "geojson",
        "linestring",
        "polygon_wkb",
        "query_x_m",
        "query_y_m",
        "cell_id",
    ):
        assert forbidden not in document.casefold()


def test_long_horizons_cannot_be_mislabeled_as_confirmatory_results() -> None:
    with pytest.raises(ValueError, match="180/365"):
        RetrospectiveHorizonResult(
            180,
            "M5_6",
            "passed",
            3,
            0.1,
            0.01,
            0.2,
            0.5,
        )
