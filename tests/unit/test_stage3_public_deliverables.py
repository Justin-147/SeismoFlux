from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date
from pathlib import Path
from xml.etree import ElementTree

import pytest

from seismoflux.background.publication import FixedFileConflictError
from seismoflux.features.anomaly.audit import AuditResult
from seismoflux.features.anomaly.dictionary import DEFAULT_FEATURE_DICTIONARY
from seismoflux.features.anomaly.public_deliverables import (
    PUBLIC_AUDIT_SVG_PATH,
    PUBLIC_DICTIONARY_PATH,
    PUBLIC_REGISTRY_PATH,
    PUBLIC_REPORT_PATH,
    Stage3AggregateSummary,
    Stage3PublicDeliverables,
    build_stage3_public_deliverables,
    publish_stage3_public_deliverables,
)
from seismoflux.features.anomaly.publication import stage3_identity_sha256
from seismoflux.features.anomaly.synthetic_audit import SyntheticPrefixAuditResult

SHA = "a" * 64


def _aggregate() -> Stage3AggregateSummary:
    return Stage3AggregateSummary(
        snapshot_count=205,
        first_report_date=date(2022, 7, 20),
        last_report_date=date(2026, 7, 1),
        known_missing_period_count=2,
        state_row_count=1_205,
        entity_state_row_count=1_000,
        report_summary_row_count=205,
        complete_entity_state_row_count=990,
        temporary_entity_state_row_count=10,
        reliability_high_state_row_count=700,
        reliability_cautious_state_row_count=290,
        reliability_excluded_state_row_count=10,
        query_cell_count=3,
        feature_row_count=615,
        nullable_value_count=10_000,
        null_value_count=400,
        future_source_reference_count=0,
        missing_period_zero_imputation_count=0,
        source_duration_feature_use_count=0,
        forbidden_source_field_use_count=0,
        target_or_earthquake_label_read_count=0,
        synthetic_prefix_seed_count=32,
        synthetic_prefix_property_failures=0,
    )


def _audit() -> AuditResult:
    return AuditResult(
        passed=True,
        selected_issue_count=12,
        selected_feature_row_count=12,
        unique_selected_cell_count=11,
        state_row_count_checked=80,
        observation_reference_count_checked=60,
        report_reference_count_checked=30,
        dictionary_definition_count=72,
        dictionary_value_column_count=783,
        feature_field_count=1_637,
        feature_scalar_count_compared=19_644,
        nullable_value_count_checked=8_000,
        null_value_count_checked=400,
        trajectory_value_count_checked=2_700,
    )


def _synthetic() -> SyntheticPrefixAuditResult:
    return SyntheticPrefixAuditResult(
        passed=True,
        seed_count=32,
        invariant_count=7,
        check_count=224,
        failure_count=0,
    )


def _dataset(row_count: int, filename: str, *, field_count: int) -> dict[str, object]:
    return {
        "content_sha256": SHA,
        "fields": [
            {"name": f"field_{index}", "nullable": False, "type": "int64"}
            for index in range(field_count)
        ],
        "file_sha256": SHA,
        "file_size_bytes": 100,
        "path": f"local/hidden/{filename}",
        "row_count": row_count,
        "row_group_count": 2,
        "schema_sha256": SHA,
        "sort_keys": ["field"],
    }


def _manifest() -> dict[str, object]:
    identity = {
        "schema_version": 1,
        "protocol_version": "0.3.0",
        "execution_mode": "feature_only_no_target_scoring",
        "protocol_freeze_tag": "v0.3.0-anomaly-feature-protocol",
        "code_commit": "c" * 40,
        "feature_dictionary_sha256": DEFAULT_FEATURE_DICTIONARY.sha256,
        "grid": {"grid_id": "d" * 64, "cell_count": 3, "cell_size_km": 25.0},
        "input_hashes": _input_hashes(),
    }
    identity_sha256 = stage3_identity_sha256(identity)
    aggregate = _aggregate()
    return {
        "schema_version": 1,
        "protocol_version": "0.3.0",
        "bundle_id": "anomaly-feature-bundle-" + identity_sha256[:16],
        "identity": identity,
        "identity_sha256": identity_sha256,
        "datasets": {
            "anomaly_state_history": _dataset(
                1_205,
                "anomaly_state_history.parquet",
                field_count=1,
            ),
            "anomaly_feature_store": _dataset(
                615,
                "anomaly_feature_store.parquet",
                field_count=1_637,
            ),
        },
        "files": [
            {
                "relative_path": "anomaly_state_history.parquet",
                "byte_count": 100,
                "sha256": SHA,
            },
            {
                "relative_path": "anomaly_feature_store.parquet",
                "byte_count": 100,
                "sha256": SHA,
            },
        ],
        "audit": {
            "actual_snapshot_count": aggregate.snapshot_count,
            "first_report_date": aggregate.first_report_date.isoformat(),
            "last_report_date": aggregate.last_report_date.isoformat(),
            "report_summary_row_count": aggregate.report_summary_row_count,
            "state_row_count": aggregate.state_row_count,
            "complete_entity_state_row_count": aggregate.complete_entity_state_row_count,
            "temporary_entity_state_row_count": aggregate.temporary_entity_state_row_count,
            "reliability_entity_state_counts": {
                "high": aggregate.reliability_high_state_row_count,
                "cautious": aggregate.reliability_cautious_state_row_count,
                "excluded": aggregate.reliability_excluded_state_row_count,
            },
            "query_cell_count": aggregate.query_cell_count,
            "feature_row_count": aggregate.feature_row_count,
            "entity_state_count": aggregate.entity_state_row_count,
            "nullable_value_count": aggregate.nullable_value_count,
            "null_value_count": aggregate.null_value_count,
            "future_source_reference_count": aggregate.future_source_reference_count,
            "known_missing_period_count": aggregate.known_missing_period_count,
            "missing_period_zero_imputation_count": (
                aggregate.missing_period_zero_imputation_count
            ),
            "source_duration_feature_use_count": aggregate.source_duration_feature_use_count,
            "forbidden_source_field_use_count": aggregate.forbidden_source_field_use_count,
            "target_or_earthquake_label_read_count": (
                aggregate.target_or_earthquake_label_read_count
            ),
            "lineage_replay": asdict(_audit()),
            "synthetic_prefix": asdict(_synthetic()),
        },
        "locked_test": {
            "artifact_ids": [],
            "result": None,
            "run": False,
            "score_ids": [],
            "target_count": None,
            "target_ids": [],
        },
    }


def _input_hashes() -> dict[str, str]:
    return {
        "protocol_bytes_sha256": SHA,
        "anomaly_history_config_bytes_sha256": SHA,
        "environment_lock_sha256": SHA,
        "data_catalog_sha256": SHA,
        "anomaly_observation_file_sha256": SHA,
        "anomaly_observation_content_sha256": SHA,
        "anomaly_observation_schema_sha256": SHA,
        "anomaly_report_period_file_sha256": SHA,
        "anomaly_report_period_content_sha256": SHA,
        "anomaly_report_period_schema_sha256": SHA,
        "study_area_sha256": SHA,
    }


def _deliverables() -> Stage3PublicDeliverables:
    return build_stage3_public_deliverables(
        manifest=_manifest(),
        dictionary=DEFAULT_FEATURE_DICTIONARY,
        audit=_audit(),
        synthetic=_synthetic(),
        code_commit="c" * 40,
        input_hashes=_input_hashes(),
    )


def test_public_deliverables_are_aggregate_only_and_explain_stage_boundary() -> None:
    deliverables = _deliverables()
    registry = json.loads(deliverables.registry_bytes)
    serialized = deliverables.registry_bytes.decode("utf-8").casefold()

    assert registry["stage4_allowed"] is True
    assert registry["locked_test"]["run"] is False
    assert "local/hidden" not in serialized
    for forbidden in ("cell_id", "longitude", "latitude", "source_row", "station_id"):
        assert forbidden not in serialized
    report = deliverables.report_bytes.decode("utf-8")
    assert "不是模型预测好坏" in report
    assert "局部缺失不会屏蔽其它区域" in report
    assert DEFAULT_FEATURE_DICTIONARY.sha256 in deliverables.dictionary_bytes.decode("utf-8") or (
        DEFAULT_FEATURE_DICTIONARY.sha256 == registry["feature_dictionary"]["sha256"]
    )


def test_audit_svg_is_valid_and_has_no_spatial_detail() -> None:
    svg = _deliverables().audit_svg_bytes
    root = ElementTree.fromstring(svg)

    assert root.tag.endswith("svg")
    text = svg.decode("utf-8")
    assert "205" in text
    assert "锁定测试未运行" in text
    for forbidden in ("cell_id", "longitude", "latitude", "POLYGON", "query_x_m"):
        assert forbidden not in text


def test_audit_svg_places_missing_periods_on_the_date_axis() -> None:
    svg = _deliverables().audit_svg_bytes
    root = ElementTree.fromstring(svg)
    start = date(2022, 7, 20)
    end = date(2026, 7, 1)
    axis_start_x = 96.0
    axis_width = 1104.0 - axis_start_x
    expected = {
        "2024 第36期缺报": axis_start_x
        + axis_width * (date(2024, 9, 4) - start).days / (end - start).days,
        "2025 第44期缺报": axis_start_x
        + axis_width * (date(2025, 10, 29) - start).days / (end - start).days,
    }
    marker_path = next(
        element.attrib["d"]
        for element in root.iter()
        if element.tag.endswith("path") and "#d97706" in element.attrib.get("stroke", "")
    )

    for label, expected_x in expected.items():
        label_node = next(element for element in root.iter() if element.text == label)
        assert float(label_node.attrib["x"]) == pytest.approx(expected_x, abs=0.001)
        assert f"M{expected_x:.3f} 254 v32" in marker_path


def test_fixed_publication_is_idempotent_and_refuses_conflict(tmp_path: Path) -> None:
    deliverables = _deliverables()
    first = publish_stage3_public_deliverables(tmp_path, deliverables)
    second = publish_stage3_public_deliverables(tmp_path, deliverables)

    assert first.registry.created is True
    assert second.registry.created is False
    for relative in (
        PUBLIC_REGISTRY_PATH,
        PUBLIC_REPORT_PATH,
        PUBLIC_AUDIT_SVG_PATH,
        PUBLIC_DICTIONARY_PATH,
    ):
        assert (tmp_path / relative).is_file()

    (tmp_path / PUBLIC_REPORT_PATH).write_text("conflict", encoding="utf-8")
    with pytest.raises(FixedFileConflictError):
        publish_stage3_public_deliverables(tmp_path, deliverables)


def test_fixed_publication_rolls_back_only_files_created_by_failed_attempt(
    tmp_path: Path,
) -> None:
    deliverables = _deliverables()
    conflicting_report = tmp_path / PUBLIC_REPORT_PATH
    conflicting_report.parent.mkdir(parents=True)
    conflicting_report.write_text("historical conflict", encoding="utf-8")

    with pytest.raises(FixedFileConflictError):
        publish_stage3_public_deliverables(tmp_path, deliverables)

    assert conflicting_report.read_text(encoding="utf-8") == "historical conflict"
    for relative in (
        PUBLIC_REGISTRY_PATH,
        PUBLIC_AUDIT_SVG_PATH,
        PUBLIC_DICTIONARY_PATH,
    ):
        assert not (tmp_path / relative).exists()


def test_publication_rejects_failed_acceptance() -> None:
    manifest = _manifest()
    manifest["audit"]["future_source_reference_count"] = 1  # type: ignore[index]

    with pytest.raises(ValueError, match="acceptance gate"):
        build_stage3_public_deliverables(
            manifest=manifest,
            dictionary=DEFAULT_FEATURE_DICTIONARY,
            audit=_audit(),
            synthetic=_synthetic(),
            code_commit="c" * 40,
            input_hashes=_input_hashes(),
        )
