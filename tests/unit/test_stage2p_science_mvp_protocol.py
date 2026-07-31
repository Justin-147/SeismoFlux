from __future__ import annotations

import ast
import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pytest
import yaml
from jsonschema import Draft202012Validator, FormatChecker  # type: ignore[import-untyped]
from jsonschema.exceptions import ValidationError  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "prospective_recent_seismicity_science_v2.yaml"
IMPLEMENTATION_CONFIG_PATH = ROOT / "configs" / "stage2p_science_mvp_implementation.yaml"
SCHEMA_PATH = ROOT / "data" / "contracts" / "stage2p_science_records_v1.json"
OLD_CONFIG_PATH = ROOT / "configs" / "prospective_recent_seismicity.yaml"
OLD_SCHEMA_PATH = ROOT / "data" / "contracts" / "stage2p_prospective_records.json"
PREREGISTRATION_PATH = (
    ROOT / "docs" / "phase2p1a2_minimal_scientific_preregistration.md"
)
HANDOFF_PATH = ROOT / "docs" / "restart_handoff_2026-07-31_stage2p_science_mvp.md"
BLUEPRINT_PATH = ROOT / "SEISMOFLUX_IMPLEMENTATION_HANDOFF.md"
RESEARCH_PROTOCOL_PATH = ROOT / "docs" / "research_protocol.md"
SCIENCE_REVIEW_PATH = (
    ROOT / "docs" / "scientific_value_review_and_model_composition.md"
)


class UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(
    loader: UniqueKeyLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)  # type: ignore[no-untyped-call]
        if key in result:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        result[cast(str, key)] = loader.construct_object(  # type: ignore[no-untyped-call]
            value_node,
            deep=deep,
        )
    return result


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


def _reject_duplicate_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def _load_schema() -> dict[str, Any]:
    value = json.loads(
        SCHEMA_PATH.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_json_pairs,
    )
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def _canonical_content_bytes(record: Mapping[str, Any]) -> bytes:
    preimage = {key: value for key, value in record.items() if key != "content_sha256"}
    return json.dumps(
        preimage,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _with_content_sha256(record: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(record))
    result["content_sha256"] = hashlib.sha256(
        _canonical_content_bytes(result)
    ).hexdigest()
    return result


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _git_sha(label: str) -> str:
    return hashlib.sha1(label.encode("utf-8")).hexdigest()


def _forecast(model_id: str) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "relative_intensity_grid_sha256": _sha(f"{model_id}-grid"),
        "alarm_mask_sha256": _sha(f"{model_id}-alarm"),
        "alarm_area_km2": 599375,
    }


def _comparison(gain: float) -> dict[str, float]:
    return {
        "recall_gain_percentage_points": gain,
        "recall_ci_lower": 0.5,
        "recall_ci_upper": 9.5,
        "information_gain_nats_per_event": 0.12,
        "information_gain_ci_lower": 0.01,
        "information_gain_ci_upper": 0.23,
    }


def _positive_records() -> dict[str, dict[str, Any]]:
    common = {
        "schema_version": 1,
        "recorded_at_utc": "2026-09-09T15:58:00Z",
        "previous_record_sha256": _sha("previous-record"),
    }
    cohort = _with_content_sha256(
        {
            **common,
            "record_type": "CohortRecord",
            "previous_record_sha256": None,
            "experiment_id": "stage2p-prospective-recent-seismicity-science-mvp-v2",
            "protocol_version": "0.2.5",
            "protocol_tag": "v0.2.5-prospective-science-mvp-protocol",
            "code_tag": "v0.2.5-prospective-science-mvp-code",
            "code_commit": _git_sha("code"),
            "valid_from_utc": "2026-09-09T16:00:00Z",
            "historical_catalog_cutoff_utc": "2026-07-09T04:25:56Z",
            "model_ids": ["P0", "P1", "PP"],
            "horizons_days": [7, 30, 90],
            "maximum_alarm_area_km2": 600000,
            "target_magnitude_minimum": 5.0,
            "target_magnitude_maximum_exclusive": 6.0,
        }
    )
    issue = _with_content_sha256(
        {
            **common,
            "record_type": "ForecastIssueRecord",
            "issue_id": "stage2p-20260909T160000Z",
            "scheduled_issue_sequence": 1,
            "status": "on_time",
            "issue_time_utc": "2026-09-09T16:00:00Z",
            "query_cutoff_utc": "2026-09-09T15:45:00Z",
            "forecast_created_at_utc": "2026-09-09T15:55:00Z",
            "code_commit": _git_sha("code"),
            "source_snapshot_sha256": _sha("source"),
            "P0_event_set_sha256": _sha("P0-events"),
            "R30_event_set_sha256": _sha("R30-events"),
            "RP30_event_set_sha256": _sha("RP30-events"),
            "forecast_bundle": {
                "P0": _forecast("P0"),
                "P1": _forecast("P1"),
                "PP": _forecast("PP"),
                "static_svg_sha256": _sha("forecast.svg"),
                "interactive_html_sha256": _sha("forecast.html"),
            },
        }
    )
    receipt = _with_content_sha256(
        {
            **common,
            "record_type": "GitHubPublicationReceipt",
            "subject_record_type": "ForecastIssueRecord",
            "subject_path": "records/issues/stage2p-20260909T160000Z.json",
            "subject_content_sha256": issue["content_sha256"],
            "repository_url": "https://github.com/Justin-147/SeismoFlux.git",
            "commit_sha": _git_sha("publication"),
            "git_blob_sha1": _git_sha("record-blob"),
            "remote_ref": "refs/heads/codex/stage2-etas-science-first",
            "remote_verified_at_utc": "2026-09-09T15:59:00Z",
            "verification_response_sha256": _sha("github-response"),
            "issue_time_utc": "2026-09-09T16:00:00Z",
        }
    )
    truth = _with_content_sha256(
        {
            **common,
            "record_type": "TruthSnapshotRecord",
            "recorded_at_utc": "2026-10-16T00:00:00Z",
            "issue_id": "stage2p-20260909T160000Z",
            "horizon_days": 7,
            "revision_sequence": 0,
            "status": "mature_truth_sealed",
            "mature_after_utc": "2026-10-16T00:00:00Z",
            "fetched_at_utc": "2026-10-16T00:00:00Z",
            "target_snapshot_sha256": _sha("truth-snapshot"),
            "target_count": 2,
        }
    )
    evaluation = _with_content_sha256(
        {
            **common,
            "record_type": "EvaluationRecord",
            "recorded_at_utc": "2027-09-09T00:00:00Z",
            "phase": "result_seal",
            "checkpoint_on_time_issue_count": 52,
            "selected_exposure_manifest_sha256": _sha("exposures"),
            "truth_manifest_sha256": _sha("truth-manifest"),
            "bootstrap_indices_sha256": _sha("bootstrap"),
            "unique_target_event_count": 20,
            "independent_cluster_count": 10,
            "input_freeze_content_sha256": _sha("input-freeze"),
            "effect_rows_opened_at_utc": "2027-09-08T23:59:00Z",
            "results": {
                "P1_minus_P0": _comparison(5.5),
                "P1_minus_PP": _comparison(3.0),
                "largest_positive_region_removed_still_positive": True,
                "largest_positive_cluster_removed_still_positive": True,
                "decision": "pass_direct_improvement_candidate",
            },
        }
    )
    return {
        "CohortRecord": cohort,
        "ForecastIssueRecord": issue,
        "GitHubPublicationReceipt": receipt,
        "TruthSnapshotRecord": truth,
        "EvaluationRecord": evaluation,
    }


def _validator() -> Draft202012Validator:
    return Draft202012Validator(_load_schema(), format_checker=FormatChecker())


def _assert_invalid(record: Mapping[str, Any]) -> None:
    with pytest.raises(ValidationError):
        _validator().validate(record)


def _walk_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return {
            str(key)
            for key in value
        } | {
            nested
            for child in value.values()
            for nested in _walk_keys(child)
        }
    if isinstance(value, list):
        return {nested for child in value for nested in _walk_keys(child)}
    return set()


def test_protocol_and_schema_are_strict_and_draft_2020_12_valid() -> None:
    config = _load_yaml(CONFIG_PATH)
    schema = _load_schema()

    assert config["schema_version"] == 1
    Draft202012Validator.check_schema(schema)
    assert set(schema["$defs"]) >= {
        "CohortRecord",
        "ForecastIssueRecord",
        "GitHubPublicationReceipt",
        "TruthSnapshotRecord",
        "EvaluationRecord",
    }
    scheduled_sequence = schema["$defs"]["ForecastIssueRecord"]["properties"][
        "scheduled_issue_sequence"
    ]
    assert scheduled_sequence["minimum"] == 1
    assert "maximum" not in scheduled_sequence


def test_scientific_question_models_and_effect_gate_are_frozen() -> None:
    config = _load_yaml(CONFIG_PATH)

    assert config["protocol"]["protocol_version"] == "0.2.5"
    assert config["protocol"]["stage_id"] == "Stage2P-1A2"
    assert config["protocol"]["status"] in {"candidate", "accepted"}
    assert config["models"]["bandwidth_km"] == 75
    assert config["models"]["mixture_weight"] == 0.5
    assert config["models"]["P1"]["definition"] == "0.5_P0_plus_0.5_R30"
    assert config["models"]["PP"]["definition"] == "0.5_P0_plus_0.5_RP30"
    assert config["models"]["same_projection_grid_support_and_normalization_for_all_models"]
    assert config["causality"]["local_Mc_only_affects_its_own_spatial_unit"]
    assert config["data"]["P0_composite_rule"] == {
        "historical_component": "local_catalog_origin_lte_historical_cutoff",
        "prospective_component": "ComCat_origin_gt_historical_cutoff_and_lte_Q",
        "same_local_Mc_filter_applies_to_both_components": True,
        "rebuilt_from_the_issue_snapshot": True,
    }
    support = config["models"]["frozen_spatial_support"]
    assert support["support_id"] == "local-support-f6816ab6c6581306"
    assert support["local_Mc"] == 4.0
    assert support["local_Mc_unit_km"] == 500
    assert support["event_below_local_Mc_excluded_only_from_that_unit"]
    assert config["calendar"]["horizons_days"] == [7, 30, 90]
    alarm = config["alarm"]
    assert alarm["maximum_alarm_area_km2"] == 600000
    assert alarm["rank_value"] == (
        "normalized_25km_cell_mass_divided_by_exact_clipped_cell_area"
    )
    assert alarm["tie_break"] == [
        "cell_row_ascending",
        "cell_column_ascending",
        "cell_id_ascending",
    ]
    assert alarm["maximum_pairwise_actual_alarm_area_difference_km2"] == 625
    assert alarm["same_rule_for_P0_P1_PP"]

    evaluation = config["evaluation"]
    assert evaluation["comparisons"] == ["P1_minus_P0", "P1_minus_PP"]
    assert evaluation["minimum_unique_M5_to_M6_events"] == 20
    assert evaluation["minimum_independent_cluster_blocks"] == 10
    regional = evaluation["regional_robustness"]
    assert regional["fixed_region_partition_count"] == 39
    assert regional["region_manifest_path"] == (
        "data/manifests/anomaly_increment_r2_spatial_strata.json"
    )
    assert regional["region_manifest_sha256"] == (
        "283a6790f6e7c16bc31d9498b2cc3cd043e19c8f141046afb898e988f25dcc83"
    )
    assert (
        regional["largest_positive_region_tie_break"]
        == "unsigned_UTF8_region_id_ascending"
    )
    assert (
        regional["largest_positive_cluster_tie_break"]
        == "unsigned_UTF8_component_id_ascending"
    )
    bootstrap = evaluation["paired_cluster_bootstrap"]
    assert bootstrap["replicates"] == 2000
    assert bootstrap["seed"] == 147
    assert bootstrap["paired_resample_shared_by_all_models_horizons_and_endpoints"]
    assert bootstrap["simultaneous_family"]["familywise_confidence_level"] == 0.95
    assert bootstrap["simultaneous_family"]["correction"] == "Bonferroni"
    assert bootstrap["simultaneous_family"]["endpoints"] == [
        "P1_minus_P0_information_gain",
        "P1_minus_PP_information_gain",
        "P1_minus_P0_strict_recall",
        "P1_minus_PP_strict_recall",
    ]
    assert evaluation["maximum_effect_looks"] == 1
    assert evaluation["formal_look_logic"] == {
        "issue_52_first_checks_sample_gate_without_opening_effect_rows": True,
        "effect_rows_open_at_52_only_if_all_sample_gates_pass": True,
        "otherwise_continue_blindly_to_104": True,
        "issue_104_is_final_sample_gate_and_only_remaining_effect_look": True,
        "inadequate_sample_at_104": "evidence_insufficient_stop_P1_retain_P0",
    }
    assert (
        evaluation["pass_requires"][
            "P1_minus_P0_recall_gain_percentage_points_minimum"
        ]
        == 5.0
    )
    assert (
        evaluation["failure_action"]
        == "stop_P1_retain_P0_and_report_negative_or_insufficient_evidence"
    )


def test_no_real_data_or_locked_test_is_authorized_by_protocol_stage() -> None:
    protocol = _load_yaml(CONFIG_PATH)["protocol"]

    assert protocol["execution_authorized"] is False
    assert protocol["real_issue_authorized"] is False
    assert protocol["real_catalog_read_authorized"] is False
    assert protocol["real_network_fetch_authorized"] is False
    assert protocol["new_target_read_count"] == 0
    assert protocol["locked_test_read_count"] == 0
    assert protocol["locked_test_run_count"] == 0


def test_next_stage_requires_three_synthetic_scenarios_and_visible_results() -> None:
    config = _load_yaml(CONFIG_PATH)
    acceptance = config["stage2p1b_mvp_synthetic_acceptance"]

    assert acceptance["uses_only_synthetic_events"]
    assert acceptance["reads_real_catalog"] is False
    assert acceptance["uses_network"] is False
    assert acceptance["required_scenarios"] == {
        "recent_activity_predictive": "P1_outperforms_P0_and_PP",
        "no_recent_signal": "P1_has_no_material_gain",
        "recent_activity_misleading": "P1_underperforms_at_least_one_control",
    }
    assert "static_and_offline_interactive_figures" in acceptance["must_verify"]
    assert acceptance["synthetic_results_are_real_prediction_evidence"] is False
    assert config["visualization"]["every_synthetic_scenario"] == [
        "three_panel_P0_P1_PP_relative_intensity_map",
        "alarm_area_and_hit_miss_overlay",
        "comparison_metric_plot",
    ]
    assert config["visualization"]["every_real_issue"] == [
        "static_SVG",
        "offline_interactive_HTML",
    ]
    assert (
        config["science_priority"]["next_stage_failure_to_produce_these_outputs"]
        == "stop_and_simplify"
    )


def test_resources_leave_cpu_headroom() -> None:
    limits = _load_yaml(CONFIG_PATH)["resource_limits"]

    assert limits["reserve_physical_cpu_cores"] >= 2
    assert limits["maximum_workers"] <= 8
    assert limits["inner_numeric_threads"] == 1


def test_science_mvp_imports_use_only_the_stage2p1b_symbol_allowlist() -> None:
    manifest = _load_yaml(IMPLEMENTATION_CONFIG_PATH)
    implementation = manifest["implementation"]
    isolation = manifest["implementation_isolation"]
    allowed = set(cast(Sequence[str], isolation["allowed_pure_reuse"]))
    forbidden = set(cast(Sequence[str], isolation["forbidden_reuse"]))
    expected_allowed = {
        "seismoflux.data.common.canonical_json_bytes",
        "seismoflux.stage2s.contracts.AlarmMask",
        "seismoflux.stage2s.contracts.NormalizedSpatialDensity",
        "seismoflux.stage2s.contracts.SpatialGrid",
        "seismoflux.stage2s.contracts.SpatialQuadratureFamily",
        "seismoflux.stage2s.spatial.build_normalized_kde",
        "seismoflux.stage2s.spatial.build_recent_component",
        "seismoflux.stage2s.spatial.event_cell_index_25km",
        "seismoflux.stage2s.spatial.mix_density",
        "seismoflux.stage2s.spatial.select_alarm_prefix",
    }

    assert implementation == {
        "stage_id": "Stage2P-1B",
        "status": "accepted",
        "implementation_frozen": True,
        "protocol_version": "0.2.5",
        "protocol_tag": "v0.2.5-prospective-science-mvp-protocol",
        "planned_code_tag": "v0.2.5-prospective-science-mvp-code",
        "synthetic_only": True,
        "real_catalog_read_authorized": False,
        "real_network_fetch_authorized": False,
        "science_value_category": "necessary_enabler",
    }
    assert allowed == expected_allowed
    assert allowed.isdisjoint(forbidden)

    acceptance = manifest["acceptance"]
    assert acceptance["status"] == "accepted"
    assert acceptance["exact_regression_test_count"] == 282
    assert acceptance["independent_audit_P0_count"] == 0
    assert acceptance["independent_audit_P1_count"] == 0
    output_directory = ROOT / cast(str, acceptance["output_directory"])
    output_hashes = cast(Mapping[str, str], acceptance["output_sha256"])
    assert set(output_hashes) == {
        "recent_activity_predictive.svg",
        "no_recent_signal.svg",
        "recent_activity_misleading.svg",
        "scenario_comparison.svg",
        "stage2p_science_mvp_explorer.html",
        "metrics.json",
    }
    for name, expected_sha256 in output_hashes.items():
        assert hashlib.sha256((output_directory / name).read_bytes()).hexdigest() == (
            expected_sha256
        )

    controlled_prefixes = (
        "seismoflux.background",
        "seismoflux.stage2s",
        "seismoflux.anomaly_increment",
        "seismoflux.data",
    )
    expected_sources = {
        ROOT / relative_path
        for relative_path in cast(Sequence[str], manifest["source_scope"])
    }
    assert expected_sources
    for source_path in sorted(expected_sources):
        assert source_path.is_file()
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(controlled_prefixes):
                        pytest.fail(
                            f"{source_path}: module import {alias.name!r} bypasses "
                            "the exact symbol allowlist"
                        )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.startswith("seismoflux.stage2p"):
                    continue
                if not module.startswith(controlled_prefixes):
                    continue
                for alias in node.names:
                    assert alias.name != "*", (
                        f"{source_path}: wildcard import from {module!r} is forbidden"
                    )
                    symbol = f"{module}.{alias.name}"
                    assert symbol in allowed, (
                        f"{source_path}: controlled reuse {symbol!r} is not in the "
                        "Stage2P-1B exact allowlist"
                    )
                    assert not any(
                        symbol == denied or symbol.startswith(f"{denied}.")
                        for denied in forbidden
                    ), f"{source_path}: forbidden reuse {symbol!r}"


def test_all_five_record_examples_have_valid_canonical_hash_and_schema() -> None:
    records = _positive_records()
    validator = _validator()

    assert set(records) == {
        "CohortRecord",
        "ForecastIssueRecord",
        "GitHubPublicationReceipt",
        "TruthSnapshotRecord",
        "EvaluationRecord",
    }
    for record_type, record in records.items():
        assert record["record_type"] == record_type
        assert record["content_sha256"] == hashlib.sha256(
            _canonical_content_bytes(record)
        ).hexdigest()
        validator.validate(record)


def test_record_contract_rejects_extra_fields_bad_sha_and_invalid_states() -> None:
    records = _positive_records()

    extra = copy.deepcopy(records["CohortRecord"])
    extra["unregistered_field"] = "forbidden"
    _assert_invalid(extra)

    bad_sha = copy.deepcopy(records["ForecastIssueRecord"])
    bad_sha["source_snapshot_sha256"] = "A" * 64
    _assert_invalid(bad_sha)

    missed_with_forecast = copy.deepcopy(records["ForecastIssueRecord"])
    missed_with_forecast["status"] = "missed_issue"
    _assert_invalid(missed_with_forecast)

    unavailable_with_count = copy.deepcopy(records["TruthSnapshotRecord"])
    unavailable_with_count["status"] = "truth_snapshot_unavailable"
    _assert_invalid(unavailable_with_count)

    freeze_with_results = copy.deepcopy(records["EvaluationRecord"])
    freeze_with_results["phase"] = "input_freeze"
    _assert_invalid(freeze_with_results)


def test_old_v024_contract_remains_distinct_and_unchanged_in_role() -> None:
    old_config = _load_yaml(OLD_CONFIG_PATH)
    new_config = _load_yaml(CONFIG_PATH)
    old_schema = json.loads(OLD_SCHEMA_PATH.read_text(encoding="utf-8"))
    new_schema = _load_schema()

    assert OLD_CONFIG_PATH != CONFIG_PATH
    assert OLD_SCHEMA_PATH != SCHEMA_PATH
    assert old_config["protocol"]["protocol_version"] == "0.2.4"
    assert old_config["protocol"]["protocol_tag"] == (
        "v0.2.4-prospective-seismicity-protocol"
    )
    assert new_config["protocol"]["supersedes_for_future_execution"] == {
        "protocol_version": "0.2.4",
        "protocol_tag": "v0.2.4-prospective-seismicity-protocol",
        "scope": "engineering_and_record_contract_only",
        "historical_bytes_remain_unchanged": True,
        "historical_results_remain_none": True,
    }
    assert old_schema["$id"] != new_schema["$id"]
    assert old_schema["title"] != new_schema["title"]


def test_new_contract_has_no_old_timestamp_or_certificate_dependency() -> None:
    config = _load_yaml(CONFIG_PATH)
    schema = _load_schema()
    all_keys = _walk_keys(config) | _walk_keys(schema)
    banned_keys = {
        "timestamp_attempt_evidence",
        "remote_timestamp",
        "trusted_registry",
        "rfc3161",
        "tsa_url",
        "tsa_request",
        "tsa_response",
        "certificate_chain",
    }
    combined = (
        CONFIG_PATH.read_text(encoding="utf-8")
        + SCHEMA_PATH.read_text(encoding="utf-8")
    ).lower()

    assert all_keys.isdisjoint(banned_keys)
    for token in ("rfc3161", ".tsq", ".tsr", "timestamp_attempt_evidence"):
        assert token not in combined
    assert config["science_priority"]["excluded_as_mandatory_gates"] == [
        "external_cryptographic_timestamping",
        "public_key_certificate_validation",
        "per_artifact_typed_manifest_registry",
        "full_byte_reconstruction_of_every_intermediate_table",
        "hardware_identity_receipts",
    ]


def test_preregistration_handoff_blueprint_and_science_review_are_current() -> None:
    paths = [
        PREREGISTRATION_PATH,
        HANDOFF_PATH,
        BLUEPRINT_PATH,
        RESEARCH_PROTOCOL_PATH,
        SCIENCE_REVIEW_PATH,
    ]

    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "Stage2P-1A2" in text, path
        assert "0.2.5" in text, path
        assert all(model_id in text for model_id in ("P0", "P1", "PP")), path

    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert "necessary_enabler" in combined
    assert "synthetic" in combined.lower()
    assert "static" in combined.lower()
    assert "interactive" in combined.lower()
