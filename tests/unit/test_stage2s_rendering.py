# ruff: noqa: RUF001
from __future__ import annotations

import json
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import replace
from html.parser import HTMLParser
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from numpy.typing import NDArray

from seismoflux.stage2s import rendering
from seismoflux.stage2s.records import Stage2SWholeRunRecord
from seismoflux.stage2s.rendering import (
    ALL_ARTIFACT_NAMES,
    COMPANION_PNG_NAME,
    DISPLAY_ALARM_BUDGETS_KM2,
    FORMAL_ALARM_BUDGET_KM2,
    PROTOCOL_ARTIFACT_NAMES,
    Stage2SMapFrame,
    Stage2SRenderingError,
    Stage2SRenderPayload,
    build_rank_map_frame,
    parse_stage2s_render_payload,
    render_stage2s_bundle,
    verify_stage2s_bundle_against_record,
)


class _DocumentParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.start_tags: list[str] = []
        self.end_tags: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        del attrs
        self.start_tags.append(tag)

    def handle_endtag(self, tag: str) -> None:
        self.end_tags.append(tag)


def _record(
    *,
    mode: str = "synthetic_acceptance",
    artifact_sha256_by_name: Mapping[str, object] | None = None,
    claim_limited: bool = False,
) -> Stage2SWholeRunRecord:
    metric_keys = (
        "S1_minus_S0:IG",
        "S1_minus_S0:recall",
        "S1_minus_SP:IG",
        "S1_minus_SP:recall",
    )
    overall_macros = {
        "S1_minus_S0:IG": 0.5,
        "S1_minus_S0:recall": 0.5,
        "S1_minus_SP:IG": 0.25,
        "S1_minus_SP:recall": 0.25,
    }
    cells = []
    for fold_index in (1, 2, 3):
        for horizon_days in (7, 30, 90):
            base = fold_index * 0.1 + horizon_days * 0.001
            cells.append(
                {
                    "fold_index": fold_index,
                    "horizon_days": horizon_days,
                    "issue_count": 2,
                    "event_ids": [
                        f"private-event-{fold_index}-{horizon_days}-a",
                        f"private-event-{fold_index}-{horizon_days}-b",
                    ],
                    "hit_by_model": {
                        "S0": [False, True],
                        "S1": [True, True],
                        "SP": [False, True],
                    },
                    "supported_ig": [True, True],
                    "ig_event_log_ratios": {
                        "S1_minus_S0": [base, base],
                        "S1_minus_SP": [base / 2.0, base / 2.0],
                    },
                    "recall_hit_differences": {
                        "S1_minus_S0": [1.0, 0.0],
                        "S1_minus_SP": [0.0, 0.0],
                    },
                    "compensator_differences": {
                        "S1_minus_S0": 0.0,
                        "S1_minus_SP": 0.0,
                    },
                    "information_gain": {
                        "S1_minus_S0": base,
                        "S1_minus_SP": base / 2.0,
                    },
                    "recall_gain": {
                        "S1_minus_S0": 0.5,
                        "S1_minus_SP": 0.25,
                    },
                }
            )
    first_contributions = {
        "S1_minus_S0:IG": 0.5 if claim_limited else 0.25,
        "S1_minus_S0:recall": 0.25,
        "S1_minus_SP:IG": 0.125,
        "S1_minus_SP:recall": 0.125,
    }
    second_contributions = {
        key: overall_macros[key] - first_contributions[key] for key in metric_keys
    }

    def sequence_component(
        component_id: str,
        contributions: dict[str, float],
    ) -> dict[str, object]:
        return {
            "component_id": component_id,
            "event_ids": [component_id],
            "event_count": 1,
            "event_fraction": 0.5,
            "origin_time_span_days": 0.0,
            "max_pairwise_geodesic_distance_km": 0.0,
            "contributions": contributions,
            "model_hits": {
                "S0": {"raw": 0.1, "fraction": 0.5},
                "S1": {"raw": 0.35, "fraction": 0.5},
                "SP": {"raw": 0.225, "fraction": 0.5},
            },
            "information_gain": {
                "S1_minus_S0": {
                    "raw": contributions["S1_minus_S0:IG"],
                    "fraction": (
                        contributions["S1_minus_S0:IG"] / overall_macros["S1_minus_S0:IG"]
                    ),
                },
                "S1_minus_SP": {
                    "raw": contributions["S1_minus_SP:IG"],
                    "fraction": (
                        contributions["S1_minus_SP:IG"] / overall_macros["S1_minus_SP:IG"]
                    ),
                },
            },
        }

    component_contributions_by_id = {
        "sequence-01": first_contributions,
        "sequence-02": second_contributions,
    }
    components = [
        sequence_component(component_id, contributions)
        for component_id, contributions in component_contributions_by_id.items()
    ]
    strongest_component_id_by_metric = {
        key: min(
            component_contributions_by_id,
            key=lambda component_id: (
                -component_contributions_by_id[component_id][key],
                component_id.encode("utf-8"),
            ),
        )
        for key in metric_keys
    }
    leave_largest_count_out = {
        key: overall_macros[key] - first_contributions[key] for key in metric_keys
    }
    leave_largest_gain_out = {
        key: overall_macros[key]
        - component_contributions_by_id[strongest_component_id_by_metric[key]][key]
        for key in metric_keys
    }
    interpretation_limit = (
        "claim_limited_to_sequence_associated_continuation_not_broad_regional_gain"
        if claim_limited
        else "no_sequence_interpretation_limit"
    )
    return Stage2SWholeRunRecord(
        mode=mode,  # type: ignore[arg-type]
        identity={
            "experiment_id": "stage2s-render-synthetic",
            "unstable_run_clock": "must-not-be-rendered",
        },
        input_receipts={"synthetic_fixture_sha256": "a" * 64},
        fold_fit_summaries=[
            {
                "fold_index": fold_index,
                "fit_receipt_sha256": str(fold_index) * 64,
                "fold_prediction_seal_sha256": str(fold_index + 6) * 64,
                "alpha_R_by_delay": {"0": 1.0, "1": 0.8, "7": 0.4},
                "alpha_P_by_delay": {"0": 0.0, "1": 0.1, "7": 0.2},
                "shared_rate_per_day": 0.05,
            }
            for fold_index in (1, 2, 3)
        ],
        issue_prediction_summaries=[
            {
                "fold_index": 1,
                "issue_date": "2022-01-06",
                "issue_time_utc": "2022-01-06T16:00:00Z",
                "horizons_days": [7],
                "issue_prediction_seal_sha256": "4" * 64,
                "actual_alarm_area_km2": {
                    "delay0:S0": 300.0,
                    "delay0:S1": 300.0,
                    "delay0:SP": 300.0,
                },
            }
        ],
        seal_chain={
            "fold_fit_receipt_sha256": ["1" * 64, "2" * 64, "3" * 64],
            "issue_prediction_seal_sha256": ["4" * 64],
            "fold_prediction_seal_sha256": ["7" * 64, "8" * 64, "9" * 64],
            "master_prediction_seal_sha256": "f" * 64,
        },
        cell_scores=cells,
        bootstrap_summary={
            "replications": 2000,
            "intervals": {
                "S1_minus_S0:IG": {"point": 0.5, "lower": 0.2, "upper": 0.8},
                "S1_minus_S0:recall": {"point": 0.5, "lower": 0.1, "upper": 0.7},
                "S1_minus_SP:IG": {"point": 0.25, "lower": 0.05, "upper": 0.4},
                "S1_minus_SP:recall": {"point": 0.25, "lower": 0.05, "upper": 0.4},
            },
        },
        bootstrap_rows=[[0.5, 0.5, 0.25, 0.25] for _ in range(2000)],
        regional_evidence={
            "regions": [
                {
                    "zone_id": f"zone-{zone_index:02d}",
                    "ig_event_count": zone_index % 4,
                    "recall_event_count": zone_index % 5,
                    "contributions": {
                        "S1_minus_S0:IG": zone_index / 1_000.0,
                        "S1_minus_S0:recall": zone_index / 2_000.0,
                        "S1_minus_SP:IG": zone_index / 3_000.0,
                        "S1_minus_SP:recall": zone_index / 4_000.0,
                    },
                }
                for zone_index in range(1, 40)
            ],
            "results": {
                "S1_minus_S0:IG": {
                    "event_bearing_zone_count": 4,
                    "positive_event_bearing_zone_count": 3,
                    "strongest_positive_zone_id": "zone-39",
                    "leave_strongest_out_residual": 0.2,
                    "passed": True,
                },
                "S1_minus_S0:recall": {
                    "event_bearing_zone_count": 4,
                    "positive_event_bearing_zone_count": 3,
                    "strongest_positive_zone_id": "zone-39",
                    "leave_strongest_out_residual": 0.1,
                    "passed": True,
                },
                "S1_minus_SP:IG": {
                    "event_bearing_zone_count": 4,
                    "positive_event_bearing_zone_count": 2,
                    "strongest_positive_zone_id": "zone-38",
                    "leave_strongest_out_residual": 0.05,
                    "passed": True,
                },
                "S1_minus_SP:recall": {
                    "event_bearing_zone_count": 4,
                    "positive_event_bearing_zone_count": 2,
                    "strongest_positive_zone_id": "zone-38",
                    "leave_strongest_out_residual": 0.02,
                    "passed": True,
                },
            },
            "failures": [],
            "passed": True,
        },
        sequence_evidence={
            "component_count": 2,
            "event_resampling_unit_count": 2,
            "global_residual": {key: 0.0 for key in metric_keys},
            "primary_model_recall": {"S0": 0.2, "S1": 0.7, "SP": 0.45},
            "components": components,
            "largest_count_component_id": "sequence-01",
            "largest_count_component": {
                **components[0],
                "leave_out": leave_largest_count_out,
            },
            "largest_gain_component_id": {
                key: strongest_component_id_by_metric[key] for key in metric_keys
            },
            "largest_gain_component": {
                key: {
                    "component_id": strongest_component_id_by_metric[key],
                    "raw_contribution": component_contributions_by_id[
                        strongest_component_id_by_metric[key]
                    ][key],
                    "leave_out": leave_largest_gain_out[key],
                }
                for key in metric_keys
            },
            "leave_out_residual": leave_largest_gain_out,
            "leave_largest_count_out": leave_largest_count_out,
            "leave_largest_gain_out": leave_largest_gain_out,
            "claim_limited": claim_limited,
            "interpretation_limit": interpretation_limit,
        },
        descriptive_point_estimates={
            "SP_minus_S0": {
                "information_gain": 0.25,
                "recall_gain": 0.25,
                "derivation": "S1_minus_S0_minus_S1_minus_SP",
                "inferential_status": "descriptive_point_estimate_only",
                "included_in_bootstrap_ci": False,
                "included_in_gate": False,
            }
        },
        latency_evidence=[
            {
                "delay_days": 1,
                "metrics": {
                    "S1_minus_S0:IG": 0.4,
                    "S1_minus_S0:recall": 0.3,
                    "S1_minus_SP:IG": 0.2,
                    "S1_minus_SP:recall": 0.1,
                },
            },
            {
                "delay_days": 7,
                "metrics": {
                    "S1_minus_S0:IG": 0.2,
                    "S1_minus_S0:recall": 0.1,
                    "S1_minus_SP:IG": 0.08,
                    "S1_minus_SP:recall": 0.03,
                },
            },
        ],
        gate_evidence={
            "status": "passed_development_signal",
            "reasons": [],
            "overall_macros": overall_macros,
            "claim_limited": claim_limited,
            "interpretation_limit": interpretation_limit,
            "interpretation_scope": (
                "sequence_associated_continuation_only"
                if claim_limited
                else "broad_regional_gain_not_sequence_limited"
            ),
        },
        artifact_sha256_by_name=artifact_sha256_by_name or {},
    )


def _frames() -> tuple[Stage2SMapFrame, ...]:
    x = np.tile(np.arange(4, dtype=np.float64), 4)
    y = np.repeat(np.arange(4, dtype=np.float64), 4)
    area = np.full(16, 100.0, dtype=np.float64)
    alarm = np.zeros(16, dtype=np.bool_)
    alarm[-3:] = True
    model_ids: tuple[Literal["S0", "S1", "SP"], ...] = ("S0", "S1", "SP")
    return tuple(
        build_rank_map_frame(
            issue_time_utc="2022-01-06T16:00:00Z",
            data_cutoff_utc="2022-01-06T16:00:00Z",
            fold_index=1,
            horizon_days=7,
            model_id=model_id,
            projected_x_m=x,
            projected_y_m=y,
            relative_mass=area
            * np.linspace(0.1 + model_index * 0.01, 1.6 + model_index * 0.01, 16),
            clipped_area_km2=area,
            alarm=alarm,
            actual_alarm_area_km2=300.0,
            raster_width=4,
            raster_height=4,
        )
        for model_index, model_id in enumerate(model_ids)
    )


def _payload(
    *,
    mode: str = "synthetic_acceptance",
    artifact_sha256_by_name: Mapping[str, object] | None = None,
    claim_limited: bool = False,
) -> Stage2SRenderPayload:
    return Stage2SRenderPayload(
        record=_record(
            mode=mode,
            artifact_sha256_by_name=artifact_sha256_by_name,
            claim_limited=claim_limited,
        ),
        s0_training_cutoff_utc="2019-12-31T16:00:00Z",
        recent_origin_window="(T-30d, T]",
        preceding_origin_window="(T-60d, T-30d]",
        available_at_cutoff="available_at <= T；延迟敏感性另用 T-1d / T-7d",
        map_frames=_frames(),
    )


def _payload_mapping(payload: Stage2SRenderPayload) -> dict[str, object]:
    return {
        "record": json.loads(payload.record.to_canonical_bytes()),
        "s0_training_cutoff_utc": payload.s0_training_cutoff_utc,
        "recent_origin_window": payload.recent_origin_window,
        "preceding_origin_window": payload.preceding_origin_window,
        "available_at_cutoff": payload.available_at_cutoff,
        "map_frames": [
            {
                "issue_time_utc": frame.issue_time_utc,
                "data_cutoff_utc": frame.data_cutoff_utc,
                "fold_index": frame.fold_index,
                "horizon_days": frame.horizon_days,
                "model_id": frame.model_id,
                "relative_intensity_rank": frame.relative_intensity_rank,
                "study_area_km2": frame.study_area_km2,
                "alarm_area_fraction_by_budget_km2": dict(frame.alarm_area_fraction_by_budget_km2),
                "actual_alarm_area_km2_by_budget": dict(frame.actual_alarm_area_km2_by_budget),
            }
            for frame in payload.map_frames
        ],
    }


def _artifact_bound_payload() -> Stage2SRenderPayload:
    provisional = _payload()
    artifact_sha256_by_name = dict(render_stage2s_bundle(provisional).artifact_sha256_by_name)
    return _payload(artifact_sha256_by_name=artifact_sha256_by_name)


def test_bundle_contains_all_preregistered_files_and_png_deterministically() -> None:
    first = render_stage2s_bundle(_payload())
    second = render_stage2s_bundle(_payload())

    assert tuple(item.name for item in first.artifacts) == ALL_ARTIFACT_NAMES
    assert set(PROTOCOL_ARTIFACT_NAMES).issubset(first.artifact_sha256_by_name)
    assert first.artifact_sha256_by_name == second.artifact_sha256_by_name
    assert tuple(item.payload for item in first.artifacts) == tuple(
        item.payload for item in second.artifacts
    )
    png = first.artifact(COMPANION_PNG_NAME).payload
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert struct.unpack(">II", png[16:24]) == (1600, 1000)
    assert b"tIME" not in png


def _axis_summary(*, recall_scale: float = 0.5) -> rendering._Summary:
    return rendering._Summary(
        mode_code="SYNTHETIC",
        status_banner="synthetic",
        scope_note="synthetic",
        experiment_id="axis-regression",
        gate_status="passed_development_signal",
        gate_label="passed",
        gate_reasons=(),
        metrics={
            "S1_minus_S0:IG": rendering._Interval(point=0.5, lower=0.2, upper=0.8),
            "S1_minus_SP:IG": rendering._Interval(point=0.25, lower=0.05, upper=0.4),
            "S1_minus_S0:recall": rendering._Interval(
                point=recall_scale,
                lower=-recall_scale,
                upper=recall_scale,
            ),
            "S1_minus_SP:recall": rendering._Interval(
                point=-recall_scale,
                lower=-recall_scale,
                upper=recall_scale,
            ),
        },
        cells=(),
        regional={},
        sequence={},
        latency=(),
    )


def test_png_metric_panel_boundary_keeps_titles_and_tick_labels_apart() -> None:
    figure = Figure(figsize=(16.0, 10.0), dpi=100)
    canvas = FigureCanvasAgg(figure)
    summary = _axis_summary()
    ig_axis = figure.add_axes(rendering._IG_AXIS_BOUNDS)
    recall_axis = figure.add_axes(rendering._RECALL_AXIS_BOUNDS)

    rendering._configure_metric_axis(ig_axis, summary=summary, metric="IG")
    rendering._configure_metric_axis(recall_axis, summary=summary, metric="recall")
    canvas.draw()  # type: ignore[no-untyped-call]

    renderer = canvas.get_renderer()  # type: ignore[no-untyped-call]
    ig_tick_bottom = min(
        label.get_window_extent(renderer).y0
        for label in ig_axis.get_xticklabels()
        if label.get_visible()
    )
    recall_title_top = recall_axis.title.get_window_extent(renderer).y1
    assert ig_tick_bottom - recall_title_top >= 12.0


def test_near_zero_recall_axis_has_readable_plain_ticks_without_offset() -> None:
    figure = Figure(figsize=(8.0, 4.0), dpi=100)
    canvas = FigureCanvasAgg(figure)
    axis = figure.add_axes((0.1, 0.2, 0.8, 0.6))

    rendering._configure_metric_axis(
        axis,
        summary=_axis_summary(recall_scale=1.0e-12),
        metric="recall",
    )
    canvas.draw()  # type: ignore[no-untyped-call]

    assert max(abs(limit) for limit in axis.get_xlim()) >= 0.1
    assert axis.xaxis.get_offset_text().get_text() == ""
    tick_labels = [label.get_text().casefold() for label in axis.get_xticklabels()]
    assert all("e" not in label and "×" not in label for label in tick_labels)


def test_svg_and_html_are_parseable_offline_and_do_not_publish_target_rows() -> None:
    bundle = render_stage2s_bundle(_payload())

    for name in PROTOCOL_ARTIFACT_NAMES[:3]:
        root = ET.fromstring(bundle.artifact(name).payload)
        assert root.tag.endswith("svg")
    for name in PROTOCOL_ARTIFACT_NAMES[3:]:
        document = bundle.artifact(name).payload.decode("utf-8")
        parser = _DocumentParser()
        parser.feed(document)
        parser.close()
        assert parser.start_tags.count("html") == 1
        assert parser.end_tags.count("html") == 1
        assert "http://" not in document and "https://" not in document
        assert "fetch(" not in document and "XMLHttpRequest" not in document
        assert "private-event" not in document
        for field in (
            "S0_training_cutoff",
            "R_and_RP_origin_windows",
            "available_at_cutoff",
            "fold_alpha_R_alpha_P_and_shared_rate",
            "fold_and_horizon",
            "issue_fold_and_master_seal_sha256",
            "actual_alarm_area_km2",
        ):
            assert field in document


def test_synthetic_and_formal_modes_are_unmistakable_and_never_claim_probability() -> None:
    synthetic = render_stage2s_bundle(_payload())
    formal = render_stage2s_bundle(_payload(mode="formal_development"))

    synthetic_text = b"\n".join(item.payload for item in synthetic.artifacts[:5]).decode("utf-8")
    formal_text = b"\n".join(item.payload for item in formal.artifacts[:5]).decode("utf-8")
    assert "SYNTHETIC" in synthetic_text
    assert "REAL DATA 未读取" in synthetic_text
    assert "REAL DATA" in formal_text and "DEVELOPMENT" in formal_text
    assert "历史回溯" in formal_text
    assert "不是绝对发震概率" in synthetic_text
    assert "当前或前瞻预测" in formal_text


def test_artifact_bytes_ignore_record_hash_and_artifact_hash_map_to_break_cycle() -> None:
    empty_record = _record()
    bound_record = _record(
        artifact_sha256_by_name={
            "some_prior_artifact.svg": "a" * 64,
            "another_prior_artifact.html": "b" * 64,
        }
    )
    assert empty_record.run_record_sha256 != bound_record.run_record_sha256
    empty_bundle = render_stage2s_bundle(_payload())
    bound_bundle = render_stage2s_bundle(
        _payload(
            artifact_sha256_by_name={
                "some_prior_artifact.svg": "a" * 64,
                "another_prior_artifact.html": "b" * 64,
            }
        )
    )
    assert tuple(item.payload for item in empty_bundle.artifacts) == tuple(
        item.payload for item in bound_bundle.artifacts
    )


def test_map_frame_factory_removes_coordinates_and_returns_rank_raster() -> None:
    clipped_area = np.array([100.0, 200.0, 300.0, 400.0])
    frame = build_rank_map_frame(
        issue_time_utc="2022-01-06T16:00:00Z",
        data_cutoff_utc="2022-01-06T16:00:00Z",
        fold_index=1,
        horizon_days=7,
        model_id="S1",
        projected_x_m=np.array([0.0, 1.0, 0.0, 1.0]),
        projected_y_m=np.array([0.0, 0.0, 1.0, 1.0]),
        relative_mass=clipped_area * np.array([0.1, 0.2, 0.3, 0.4]),
        clipped_area_km2=clipped_area,
        alarm=np.array([False, False, True, True]),
        actual_alarm_area_km2=700.0,
        raster_width=4,
        raster_height=4,
    )

    assert len(frame.relative_intensity_rank) == 4
    assert len(frame.relative_intensity_rank[0]) == 4
    assert (
        max(value for row in frame.relative_intensity_rank for value in row if value is not None)
        == 1.0
    )
    assert not hasattr(frame, "projected_x_m")
    assert not hasattr(frame, "longitude")


def test_map_frames_require_complete_model_triplets_and_valid_rank() -> None:
    with pytest.raises(Stage2SRenderingError, match="S0, S1, and SP"):
        Stage2SRenderPayload(
            record=_record(),
            s0_training_cutoff_utc="2019-12-31T16:00:00Z",
            recent_origin_window="(T-30d,T]",
            preceding_origin_window="(T-60d,T-30d]",
            available_at_cutoff="available_at <= T",
            map_frames=_frames()[:2],
        )
    with pytest.raises(Stage2SRenderingError, match=r"\[0, 1\]"):
        source = _frames()[0]
        invalid_rank = [list(row) for row in source.relative_intensity_rank]
        first_populated = next(
            (row_index, column_index)
            for row_index, row in enumerate(invalid_rank)
            for column_index, value in enumerate(row)
            if value is not None
        )
        invalid_rank[first_populated[0]][first_populated[1]] = 1.1
        replace(source, relative_intensity_rank=invalid_rank)


def test_json_parser_verifies_record_hash_and_reconstructs_payload() -> None:
    original = _payload()
    mapping = _payload_mapping(original)
    parsed = parse_stage2s_render_payload(json.dumps(mapping, ensure_ascii=False).encode("utf-8"))
    assert parsed.record.run_record_sha256 == original.record.run_record_sha256
    assert render_stage2s_bundle(parsed).artifact_sha256_by_name == (
        render_stage2s_bundle(original).artifact_sha256_by_name
    )

    missing_hash_mapping = _payload_mapping(original)
    missing_hash_record = cast(dict[str, object], missing_hash_mapping["record"])
    missing_hash_record.pop("run_record_sha256")
    with pytest.raises(Stage2SRenderingError, match="missing run_record_sha256"):
        parse_stage2s_render_payload(
            json.dumps(missing_hash_mapping, ensure_ascii=False).encode("utf-8")
        )

    mismatch_mapping = _payload_mapping(original)
    mismatch_record = cast(dict[str, object], mismatch_mapping["record"])
    mismatch_record["run_record_sha256"] = "0" * 64
    with pytest.raises(Stage2SRenderingError, match="SHA-256"):
        parse_stage2s_render_payload(
            json.dumps(mismatch_mapping, ensure_ascii=False).encode("utf-8")
        )


def test_json_parser_rejects_forged_map_issue_identity() -> None:
    mapping = _payload_mapping(_payload())
    frames = cast(list[dict[str, object]], mapping["map_frames"])
    for frame in frames:
        frame["issue_time_utc"] = "2099-01-01T00:00:00Z"
        frame["data_cutoff_utc"] = "2099-01-01T00:00:00Z"

    with pytest.raises(Stage2SRenderingError, match="map-frame issue groups differ"):
        parse_stage2s_render_payload(json.dumps(mapping, ensure_ascii=False).encode("utf-8"))


def test_render_rejects_record_seal_chain_not_bound_to_fold_summary() -> None:
    original = _payload()
    forged_record = replace(
        original.record,
        seal_chain={
            **dict(original.record.seal_chain),
            "fold_prediction_seal_sha256": ["a" * 64, "8" * 64, "9" * 64],
        },
    )
    forged_payload = replace(original, record=forged_record)

    with pytest.raises(Stage2SRenderingError, match="fold prediction seals"):
        render_stage2s_bundle(forged_payload)


def test_final_record_must_bind_exactly_all_six_render_hashes() -> None:
    provisional = _payload()
    bundle = render_stage2s_bundle(provisional)
    exact_hashes = dict(bundle.artifact_sha256_by_name)
    verify_stage2s_bundle_against_record(
        _payload(artifact_sha256_by_name=exact_hashes),
        bundle,
    )

    missing = dict(exact_hashes)
    missing.pop(ALL_ARTIFACT_NAMES[-1])
    with pytest.raises(Stage2SRenderingError, match="missing="):
        verify_stage2s_bundle_against_record(
            _payload(artifact_sha256_by_name=missing),
            bundle,
        )

    extra = {**exact_hashes, "unexpected.svg": "a" * 64}
    with pytest.raises(Stage2SRenderingError, match="extra="):
        verify_stage2s_bundle_against_record(
            _payload(artifact_sha256_by_name=extra),
            bundle,
        )

    mismatch = {**exact_hashes, ALL_ARTIFACT_NAMES[0]: "0" * 64}
    with pytest.raises(Stage2SRenderingError, match="differs from whole-run record"):
        verify_stage2s_bundle_against_record(
            _payload(artifact_sha256_by_name=mismatch),
            bundle,
        )


def test_rank_frame_uses_area_weighted_density_ties_and_is_order_invariant() -> None:
    x = np.array([0.0, 0.1, 3.0, 3.1, 0.0, 0.1, 3.0, 3.1])
    y = np.array([0.0, 0.1, 0.0, 0.1, 3.0, 3.1, 3.0, 3.1])
    area = np.array([100.0, 300.0, 200.0, 400.0, 125.0, 375.0, 250.0, 500.0])
    mass = area * 2.0
    alarm = np.array([True, False, False, False, False, False, False, False])

    def build(order: NDArray[np.int64]) -> Stage2SMapFrame:
        return build_rank_map_frame(
            issue_time_utc="2022-01-06T16:00:00Z",
            data_cutoff_utc="2022-01-06T16:00:00Z",
            fold_index=1,
            horizon_days=7,
            model_id="S1",
            projected_x_m=x[order],
            projected_y_m=y[order],
            relative_mass=mass[order],
            clipped_area_km2=area[order],
            alarm=alarm[order],
            actual_alarm_area_km2=100.0,
            raster_width=4,
            raster_height=4,
        )

    natural = build(np.arange(x.size))
    shuffled = build(np.array([6, 1, 4, 0, 7, 3, 5, 2]))
    populated_ranks = [
        value for row in natural.relative_intensity_rank for value in row if value is not None
    ]

    assert populated_ranks == [0.5, 0.5, 0.5, 0.5]
    assert natural.relative_intensity_rank == shuffled.relative_intensity_rank
    assert natural.study_area_km2 == shuffled.study_area_km2
    assert dict(natural.alarm_area_fraction_by_budget_km2) == dict(
        shuffled.alarm_area_fraction_by_budget_km2
    )
    assert dict(natural.actual_alarm_area_km2_by_budget) == dict(
        shuffled.actual_alarm_area_km2_by_budget
    )


def test_alarm_fraction_rasters_close_to_exact_clipped_area() -> None:
    frame = build_rank_map_frame(
        issue_time_utc="2022-01-06T16:00:00Z",
        data_cutoff_utc="2022-01-06T16:00:00Z",
        fold_index=1,
        horizon_days=7,
        model_id="S1",
        projected_x_m=np.array([0.0, 0.1, 3.0, 3.1]),
        projected_y_m=np.array([0.0, 0.1, 3.0, 3.1]),
        relative_mass=np.array([400.0, 300.0, 400.0, 400.0]),
        clipped_area_km2=np.array([100.0, 300.0, 200.0, 400.0]),
        alarm=np.array([True, False, False, False]),
        actual_alarm_area_km2=100.0,
        raster_width=4,
        raster_height=4,
    )

    formal_fractions = [
        value
        for row in frame.alarm_area_fraction(FORMAL_ALARM_BUDGET_KM2)
        for value in row
        if value is not None
    ]
    assert 0.25 in formal_fractions
    assert frame.actual_alarm_area_km2 == 100.0
    for budget in DISPLAY_ALARM_BUDGETS_KM2:
        exact_area = sum(
            float(area) * float(fraction)
            for area_row, fraction_row in zip(
                frame.study_area_km2,
                frame.alarm_area_fraction(budget),
                strict=True,
            )
            for area, fraction in zip(area_row, fraction_row, strict=True)
            if area is not None and fraction is not None
        )
        assert exact_area == pytest.approx(
            frame.actual_alarm_area_km2_by_budget[str(budget)],
            abs=1.0e-10,
        )


def test_explorers_expose_all_controls_labels_regions_and_no_raw_geometry() -> None:
    bundle = render_stage2s_bundle(_payload())
    diagnostics = bundle.artifact(PROTOCOL_ARTIFACT_NAMES[1]).payload.decode("utf-8")
    backtest = bundle.artifact(PROTOCOL_ARTIFACT_NAMES[3]).payload.decode("utf-8")
    map_explorer = bundle.artifact(PROTOCOL_ARTIFACT_NAMES[4]).payload.decode("utf-8")

    assert 'id="region-select"' in backtest
    assert "39 个冻结区域贡献与 LORO" in backtest
    assert "S1_minus_S0:IG" in backtest
    assert "S1_minus_SP:IG" in backtest
    assert "S1−S0：" in backtest and "S1−SP：" in backtest
    for zone_index in range(1, 40):
        assert f"zone-{zone_index:02d}" in backtest
    for field in (
        "event_resampling_unit_count",
        "event_fraction",
        "model_hit_fractions",
        "information_gain_fractions",
        "origin_time_span_days",
        "max_pairwise_geodesic_distance_km",
        "leave_largest_count_out",
        "largest_gain_component_id",
        "leave_largest_gain_out",
    ):
        assert field in backtest
    for label in (
        "事件块重采样单位",
        "最大震群",
        "命中占比",
        "IG 占比",
        "去最大事件数震群",
        "去最大增益",
    ):
        assert label in diagnostics
    assert "map_frame_bindings" in backtest

    assert 'id="issue-select"' in map_explorer
    assert 'id="area-select"' in map_explorer
    for model_id in ("S0", "S1", "SP"):
        assert f'data-model="{model_id}"' in map_explorer
    for budget in DISPLAY_ALARM_BUDGETS_KM2:
        assert str(budget) in map_explorer
    assert "仅展示性派生" in map_explorer
    assert "正式门固定 600,000 km²" in map_explorer
    assert "map_frame_bindings" in map_explorer
    for digest in ("1" * 64, "4" * 64, "7" * 64, "f" * 64):
        assert digest in backtest
        assert digest in map_explorer
    for forbidden in (
        "projected_x_m",
        "projected_y_m",
        "longitude",
        "latitude",
        '"geometry"',
        "http://",
        "https://",
        "fetch(",
        "XMLHttpRequest",
    ):
        assert forbidden not in map_explorer


def test_claim_limited_is_prominent_and_broad_claim_is_forbidden() -> None:
    bundle = render_stage2s_bundle(_payload(claim_limited=True))
    published_text = b"\n".join(artifact.payload for artifact in bundle.artifacts[:5]).decode(
        "utf-8"
    )

    assert "仅支持震群相关续发" in published_text
    assert "不支持广泛区域提升" in published_text


def test_bundle_writer_is_create_once_and_check_only(tmp_path: Path) -> None:
    bundle = render_stage2s_bundle(_payload())
    output_dir = (tmp_path / "rendered").resolve()
    output_dir.mkdir()
    existing = output_dir / ALL_ARTIFACT_NAMES[2]
    existing.write_bytes(b"pre-existing")

    with pytest.raises(Stage2SRenderingError, match="already exists"):
        bundle.write_to(output_dir)
    assert existing.read_bytes() == b"pre-existing"
    assert tuple(path.name for path in output_dir.iterdir()) == (existing.name,)

    clean_output = (tmp_path / "clean-rendered").resolve()
    bundle.write_to(clean_output)
    bundle.write_to(clean_output, check=True)
    with pytest.raises(Stage2SRenderingError, match="already exists"):
        bundle.write_to(clean_output)
    stale = clean_output / ALL_ARTIFACT_NAMES[0]
    stale.write_bytes(b"stale")
    with pytest.raises(Stage2SRenderingError, match="stale"):
        bundle.write_to(clean_output, check=True)


def test_cli_forbids_noncheck_write_to_canonical_formal_output() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            str(repository_root / "scripts/render_stage2s_results.py"),
            "--input",
            str(repository_root / "deliberately-missing-stage2s-render-payload.json"),
            "--output-dir",
            str(repository_root / "outputs/stage2s/causal_seismicity_screen"),
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )

    assert completed.returncode == 1
    assert "canonical formal Stage2S output is immutable" in completed.stderr


def test_standalone_cli_requires_final_hash_binding_for_write_and_check(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    script = repository_root / "scripts/render_stage2s_results.py"
    input_path = tmp_path / "render-payload.json"
    output_dir = (tmp_path / "rendered").resolve()

    input_path.write_text(
        json.dumps(_payload_mapping(_payload()), ensure_ascii=False),
        encoding="utf-8",
    )
    missing_hashes = subprocess.run(
        [
            sys.executable,
            str(script),
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert missing_hashes.returncode == 1
    assert "whole-run artifact hashes changed" in missing_hashes.stderr
    assert not output_dir.exists()

    input_path.write_text(
        json.dumps(_payload_mapping(_artifact_bound_payload()), ensure_ascii=False),
        encoding="utf-8",
    )
    write_result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert write_result.returncode == 0, write_result.stderr
    assert {path.name for path in output_dir.iterdir()} == set(ALL_ARTIFACT_NAMES)

    check_result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--check",
        ],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )
    assert check_result.returncode == 0, check_result.stderr
    assert "verified 6 deterministic Stage2S result files" in check_result.stdout
