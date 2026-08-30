from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import yaml
from jsonschema import Draft202012Validator  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "p1_b0_r30_prospective.yaml"
SCHEMA_PATH = ROOT / "data" / "contracts" / "p1_prospective_records_v1.json"
SOURCE_PATH = ROOT / "data" / "manifests" / "p1_source_boundary_manifest.json"
MODEL_PATH = ROOT / "data" / "manifests" / "p1_model_manifest.json"
RESEARCH_PROTOCOL_PATH = ROOT / "configs" / "research_protocol.yaml"


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


def _load_yaml() -> dict[str, Any]:
    value = yaml.load(CONFIG_PATH.read_text(encoding="utf-8"), Loader=UniqueKeyLoader)
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def _load_research_protocol() -> dict[str, Any]:
    value = yaml.load(
        RESEARCH_PROTOCOL_PATH.read_text(encoding="utf-8"),
        Loader=UniqueKeyLoader,
    )
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def _reject_duplicate_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_json_pairs,
    )
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _git_sha(label: str) -> str:
    return hashlib.sha1(label.encode("utf-8")).hexdigest()


def _seal(record: dict[str, Any]) -> dict[str, Any]:
    sealed = dict(record)
    preimage = {key: value for key, value in sealed.items() if key != "content_sha256"}
    sealed["content_sha256"] = hashlib.sha256(
        json.dumps(
            preimage,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return sealed


def _header(
    record_type: str,
    sequence: int,
    previous: dict[str, Any] | None,
    recorded_at_utc: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "record_type": record_type,
        "recorded_at_utc": recorded_at_utc,
        "chain_sequence": sequence,
        "previous_record_type": None if previous is None else previous["record_type"],
        "previous_record_sha256": None if previous is None else previous["content_sha256"],
    }


def _record_chain_examples() -> list[dict[str, Any]]:
    protocol = _seal(
        {
            **_header("ProtocolDefinition", 0, None, "2026-08-30T00:00:00Z"),
            "protocol_id": "p1-b0-r30-prospective-v1",
            "protocol_tag": "v0.2.6-p1-b0-r30-protocol",
            "code_tag": "v0.2.6-p1-b0-r30-code",
            "valid_from_utc": "2026-09-09T16:00:00Z",
            "historical_catalog_cutoff_utc": "2026-07-09T04:25:56Z",
            "source_boundary_manifest_sha256": _sha("source-boundary"),
            "model_manifest_sha256": _sha("model-manifest"),
            "protocol_commit": _git_sha("protocol-commit"),
            "real_issue_authorized": False,
        }
    )
    missed = _seal(
        {
            **_header("MissedIssueRecord", 1, protocol, "2026-09-09T16:05:00Z"),
            "issue_id": "p1-20260909T160000Z",
            "status": "missed_issue",
            "scheduled_issue_time_utc": "2026-09-09T16:00:00Z",
            "authorization_state": "not_authorized",
            "authorization_record_sha256": None,
            "reason": "real_issue_not_authorized_before_T",
            "prediction_generated": False,
            "backfill_forbidden": True,
            "valid_from_remains_fixed": True,
        }
    )
    authorization = _seal(
        {
            **_header(
                "RealIssueAuthorizationRecord",
                2,
                missed,
                "2026-09-10T08:00:00Z",
            ),
            "protocol_definition_sha256": protocol["content_sha256"],
            "authorization_commit": _git_sha("authorization-commit"),
            "code_commit": _git_sha("code-commit"),
            "remote_verified_at_utc": "2026-09-10T07:59:00Z",
            "authorized_from_scheduled_issue_utc": "2026-09-16T16:00:00Z",
            "real_issue_authorized": True,
        }
    )
    forecasts = [
        {
            "model_id": model_id,
            "relative_intensity_grid_sha256": _sha(f"{model_id}-grid"),
            "alarm_mask_sha256": _sha(f"{model_id}-mask"),
            "alarm_ranking_sha256": _sha(f"{model_id}-ranking"),
            "actual_alarm_area_km2": area,
        }
        for model_id, area in (("B0", 599900), ("B0_R30", 599500))
    ]
    forecast = _seal(
        {
            **_header("ForecastIssueRecord", 3, authorization, "2026-09-16T15:59:30Z"),
            "issue_id": "p1-20260916T160000Z",
            "status": "on_time",
            "scheduled_issue_time_utc": "2026-09-16T16:00:00Z",
            "query_cutoff_utc": "2026-09-16T15:45:00Z",
            "forecast_created_at_utc": "2026-09-16T15:55:00Z",
            "publication_completed_at_utc": "2026-09-16T15:59:00Z",
            "protocol_definition_sha256": protocol["content_sha256"],
            "authorization_record_sha256": authorization["content_sha256"],
            "model_manifest_sha256": _sha("model-manifest"),
            "source_boundary_manifest_sha256": _sha("source-boundary"),
            "source_snapshot_sha256": _sha("source-snapshot"),
            "code_commit": _git_sha("code-commit"),
            "forecasts": forecasts,
            "static_svg_sha256": _sha("static-svg"),
            "offline_interactive_html_sha256": _sha("interactive-html"),
            "B0_reference_area_km2": 599900,
            "B0_R30_next_complete_cell_area_km2": 500,
            "actual_area_difference_km2": 400,
            "area_fairness_status": "passed",
            "original_artifacts_immutable": True,
        }
    )
    truth = _seal(
        {
            **_header("TruthSnapshotRecord", 4, forecast, "2026-11-16T16:00:00Z"),
            "issue_id": "p1-20260916T160000Z",
            "horizon_days": 30,
            "protocol_definition_sha256": protocol["content_sha256"],
            "authorization_record_sha256": authorization["content_sha256"],
            "source_snapshot_sha256": _sha("truth-source"),
            "status": "mature_truth",
            "mature_after_utc": "2026-11-15T16:00:00Z",
            "truth_fetched_at_utc": "2026-11-16T16:00:00Z",
            "target_event_count": 12,
            "independent_cluster_count": 10,
            "cluster_assignment_sha256": _sha("cluster-assignment"),
            "exposure_cluster_registry_sha256": _sha("exposure-cluster-registry"),
            "magnitude_minimum": 5.0,
            "magnitude_maximum_exclusive": 6.0,
        }
    )
    review = _seal(
        {
            **_header("SequentialReviewRecord", 5, truth, "2026-11-17T00:00:00Z"),
            "protocol_definition_sha256": protocol["content_sha256"],
            "authorization_record_sha256": authorization["content_sha256"],
            "horizon_days": 30,
            "review_trigger": "cluster_10",
            "look_sequence": 1,
            "prior_completed_look_count": 0,
            "cumulative_cluster_count": 10,
            "ordered_cluster_registry_sha256": _sha("ordered-clusters"),
            "selected_cluster_prefix_sha256": _sha("first-10-clusters"),
            "elapsed_months": 2.2,
            "B0_hit_clusters": 3,
            "B0_R30_hit_clusters": 4,
            "recall_gain_percentage_points": 10.0,
            "sequentially_adjusted_interval_lower": -5.0,
            "sequentially_adjusted_interval_upper": 25.0,
            "decision": "continue_accumulation",
        }
    )
    return [protocol, missed, authorization, forecast, truth, review]


def _assert_chain_semantics(records: list[dict[str, Any]]) -> None:
    if not records or records[0]["record_type"] != "ProtocolDefinition":
        raise ValueError("chain must start with ProtocolDefinition")
    authorization_sha: str | None = None
    for index, record in enumerate(records):
        if record["chain_sequence"] != index:
            raise ValueError("chain sequence is not contiguous")
        expected_hash = _seal(record)["content_sha256"]
        if record["content_sha256"] != expected_hash:
            raise ValueError("record content hash is invalid")
        if index == 0:
            if (
                record["previous_record_type"] is not None
                or record["previous_record_sha256"] is not None
            ):
                raise ValueError("genesis predecessor must be null")
        else:
            previous = records[index - 1]
            if record["previous_record_type"] != previous["record_type"]:
                raise ValueError("previous record type mismatch")
            if record["previous_record_sha256"] != previous["content_sha256"]:
                raise ValueError("previous record hash mismatch")
        if record["record_type"] == "RealIssueAuthorizationRecord":
            authorization_sha = cast(str, record["content_sha256"])
        if record["record_type"] in {
            "ForecastIssueRecord",
            "TruthSnapshotRecord",
            "SequentialReviewRecord",
        }:
            if authorization_sha is None:
                raise ValueError("real record precedes authorization")
            if record["authorization_record_sha256"] != authorization_sha:
                raise ValueError("authorization binding mismatch")
        if record["record_type"] == "MissedIssueRecord":
            expected_state = "authorized" if authorization_sha else "not_authorized"
            if record["authorization_state"] != expected_state:
                raise ValueError("missed issue authorization state mismatch")
            expected_authorization_sha = authorization_sha if authorization_sha else None
            if record["authorization_record_sha256"] != expected_authorization_sha:
                raise ValueError("missed issue authorization hash mismatch")


def _select_guarded_issues(
    issue_times: list[datetime],
    horizon_days: int,
) -> list[datetime]:
    selected: list[datetime] = []
    for issue_time in sorted(issue_times):
        if not selected or issue_time >= selected[-1] + timedelta(days=horizon_days + 30):
            selected.append(issue_time)
    return selected


def _crossed_cluster_looks(previous_count: int, current_count: int) -> list[int]:
    return [look for look in (10, 20, 30) if previous_count < look <= current_count]


def _area_fairness_is_valid(
    B0_area: float,
    B0_R30_area: float,
    B0_R30_next_cell_area: float,
) -> bool:
    difference = B0_area - B0_R30_area
    return (
        0 <= B0_R30_area <= B0_area <= 600000
        and 0 <= difference < B0_R30_next_cell_area
        and difference < 625
    )


def test_protocol_is_frozen_but_only_synthetic_acceptance_is_authorized() -> None:
    config = _load_yaml()
    protocol = config["protocol"]

    assert protocol["protocol_frozen"] is True
    assert protocol["protocol_tag"] == "v0.2.6-p1-b0-r30-protocol"
    assert protocol["planned_code_tag"] == "v0.2.6-p1-b0-r30-code"
    assert protocol["real_issue_authorized"] is False
    assert protocol["real_catalog_read_authorized"] is False
    assert protocol["real_network_fetch_authorized"] is False
    assert protocol["next_authorized_action"] == ("P1-0B_synthetic_dual_model_acceptance_only")
    assert protocol["locked_test_authorized"] is False
    assert protocol["actual_record_count"] == 0
    assert protocol["actual_record_creation_authorized"] is False
    assert protocol["record_chain"] == {
        "genesis_record_type": "ProtocolDefinition",
        "authorization_record_type": "RealIssueAuthorizationRecord",
        "append_only_single_chain": True,
        "content_sha256_excludes_only_content_sha256": True,
        "every_non_genesis_record_requires_exact_previous_record_sha256": True,
        "no_actual_record_is_created_in_P1_0A": True,
        "preauthorization_missed_issue_records_allowed": True,
        "forecast_truth_and_review_require_prior_authorization_record": True,
    }


def test_first_issue_causality_and_missed_issue_semantics_are_frozen() -> None:
    config = _load_yaml()
    calendar = config["calendar"]

    assert calendar["timezone"] == "Asia/Shanghai"
    assert calendar["fixed_valid_from_local"] == "2026-09-10T00:00:00+08:00"
    assert calendar["fixed_valid_from_utc"] == "2026-09-09T16:00:00Z"
    assert calendar["first_query_cutoff_utc"] == "2026-09-09T15:45:00Z"
    assert calendar["valid_from_is_never_shifted_or_rewritten"] is True
    assert calendar["real_source_fetch_forbidden_until_all_prerequisites_closed"] is True
    assert calendar["first_on_time_issue_may_be_later_than_valid_from"] is True
    assert calendar["every_missed_rule_time_requires_one_MissedIssueRecord"] is True
    assert calendar["missed_issue_record_may_be_appended_after_T"] is True
    assert calendar["on_time_forecast_created_and_published_strictly_before_T"] is True
    assert calendar["prerequisite_failure_at_T"] == {
        "issue_status": "missed_issue",
        "advance_to_next_scheduled_Thursday": True,
        "backfill_forbidden": True,
        "valid_from_remains_fixed": True,
    }


def test_source_boundary_freezes_catalogs_washout_and_shared_snapshot() -> None:
    config = _load_yaml()
    source = _load_json(SOURCE_PATH)
    local = source["historical_local_catalog"]
    prospective = source["prospective_source"]

    assert local["cutoff_utc"] == "2026-07-09T04:25:56Z"
    assert local["sha256"] == ("2193514eec2889dbf4ae9598c5d45ef8961a8f3fcd26c7183b233dbe20842347")
    assert prospective["source_id"] == "usgs_anss_comcat_fdsn_event_api_v1"
    assert prospective["query_endpoint"] == ("https://earthquake.usgs.gov/fdsnws/event/1/query")
    assert prospective["count_endpoint"] == ("https://earthquake.usgs.gov/fdsnws/event/1/count")
    assert prospective["query_minimum_magnitude"] == 3.9
    assert prospective["local_filter_minimum_magnitude"] == 4.0
    assert prospective["bbox"] == {
        "maxlatitude": 53.557926,
        "maxlongitude": 135.08583,
        "minlatitude": 20.22909,
        "minlongitude": 73.446961,
    }
    assert source["same_source_washout"] == {
        "duration_days": 60,
        "first_query_cutoff_utc": "2026-09-09T15:45:00Z",
        "purpose": "ensure_every_R30_event_is_from_ComCat",
        "proves_catalog_completeness_or_source_equivalence": False,
        "required_before_first_on_time_forecast": True,
    }
    assert source["causal_snapshot_rules"]["one_snapshot_for_B0_and_B0_R30"] is True
    assert source["causal_snapshot_rules"]["local_Mc_affects_only_its_own_spatial_unit"] is True
    assert source["acquisition_evidence"] == {
        "fetch_must_complete_before_T": True,
        "local_origin_time_must_be_lte_Q": True,
        "raw_response_bytes_saved": True,
        "raw_response_sha256_required": True,
        "response_headers_saved": True,
        "response_headers_sha256_required": True,
    }
    assert source["source_boundary_deduplication"] == {
        "anchor_priority": "local_catalog",
        "candidate_order": [
            "absolute_time_difference_ascending",
            "WGS84_distance_ascending",
            "absolute_magnitude_difference_ascending",
            "stable_source_event_id_ascending",
        ],
        "manual_merge_forbidden": True,
        "matching": "deterministic_one_to_one",
        "maximum_WGS84_distance_km": 50,
        "maximum_absolute_magnitude_difference": 0.5,
        "maximum_origin_time_difference_seconds": 300,
    }
    assert source["truth"] == {
        "fetched_separately_from_issue_input": True,
        "maturity_rule": "T_plus_h_plus_30_days",
        "source_id": "usgs_anss_comcat_fdsn_event_api_v1",
        "target_window": "(T,T+h]",
        "unavailable_exposure_may_not_be_replaced": True,
        "unavailable_is_not_zero_events": True,
    }
    assert source["cross_exposure_cluster_control"] == {
        "cross_exposure_duplicate_count_forbidden": True,
        "distance_method": "WGS84_geodesic",
        "edge_distance_km_inclusive": 75,
        "edge_time_days_inclusive": 30,
        "exposure_selection": {
            "first": "earliest_on_time_issue",
            "guard_gap_days": 30,
            "next_rule": "T_next_gte_T_previous_plus_horizon_plus_30_days",
            "scope": "separately_within_each_horizon",
        },
        "graph_rule": "undirected_connected_components",
        "per_exposure_clustering_only": True,
        "representative_rule": ("earliest_origin_time_then_unsigned_UTF8_stable_event_id"),
        "selected_window_target_events_are_strictly_more_than_30_days_apart": True,
    }
    assert config["data"]["excluded_inputs"] == [
        "anomaly_reports",
        "fault_data",
        "manually_entered_prediction_fields",
        "locked_test_targets",
    ]


def test_model_alarm_target_and_sequential_decision_are_frozen() -> None:
    config = _load_yaml()
    model = _load_json(MODEL_PATH)

    assert set(model["models"]) == {"B0", "B0_R30"}
    assert model["shared_construction"]["bandwidth_km"] == 75
    assert model["models"]["B0_R30"]["alpha"] == 0.25
    assert model["models"]["B0_R30"]["definition"] == "(1-0.25)*B0+0.25*R30"
    assert model["models"]["B0"] == {
        "definition": "long_term_75km_KDE",
        "event_weight": "equal_per_eligible_event",
        "event_window": "[1970-01-01T00:00:00Z,Q]",
        "local_Mc_rule": (
            "frozen_local_unit_support_marker_only_no_recomputation_and_no_threshold_raise_elsewhere"
        ),
        "normalization_domain": "frozen_full_study_area",
        "source_available_at_lte_Q": True,
        "source_inside_frozen_support": True,
        "source_inside_study_area": True,
        "source_magnitude": "M_gte_4",
        "support_manifest_common_Mc": 4.0,
    }
    assert model["alarm_rule"]["grid_km"] == 25
    assert model["alarm_rule"]["maximum_complete_cell_prefix_area_km2"] == 600000
    assert model["alarm_rule"]["rank_value"] == (
        "normalized_cell_mass_divided_by_exact_clipped_cell_area"
    )
    assert model["alarm_rule"]["rank_direction"] == "descending"
    assert model["alarm_rule"]["tie_break"] == [
        "row_ascending",
        "column_ascending",
        "cell_id_ascending",
    ]
    assert model["alarm_rule"]["paired_primary_fairness"] == {
        "B0_R30_may_never_use_more_area_than_B0": True,
        "absolute_area_difference_must_be_less_than_km2": 625,
        "actual_area_difference_must_be_less_than_B0_R30_next_complete_cell_area": True,
        "challenger_rule": "B0_R30_complete_cell_prefix_not_exceeding_B0_reference_area",
        "difference_failure_action": "scientific_integrity_failure_not_evaluable",
        "initial_prefix_cap_km2": 600000,
        "reference_area_rule": "B0_actual_complete_cell_prefix_area_at_initial_cap",
    }
    evaluation = model["evaluation"]
    assert evaluation["primary_horizon_days"] == 30
    assert evaluation["secondary_horizon_days"] == 90
    assert evaluation["magnitude_minimum"] == 5.0
    assert evaluation["magnitude_maximum_exclusive"] == 6.0
    assert evaluation["independent_cluster_rule"] == {
        "connected_components": True,
        "construction_scope": "separately_within_each_selected_exposure",
        "cross_exposure_duplicate_count_forbidden": True,
        "maximum_distance_km": 75,
        "maximum_origin_time_difference_days": 30,
        "representative_rule": ("earliest_origin_time_then_unsigned_UTF8_stable_event_id"),
    }
    assert evaluation["sequential_review_cluster_counts"] == [10, 20, 30]
    assert evaluation["maximum_followup_months"] == 36
    assert evaluation["terminal_local"] == "2029-09-10T00:00:00+08:00"
    assert evaluation["terminal_utc"] == "2029-09-09T16:00:00Z"
    assert evaluation["exposure_selection"] == {
        "first": "earliest_on_time_issue",
        "guard_gap_days": 30,
        "next_rule": "T_next_gte_T_previous_plus_horizon_plus_30_days",
        "scope": "separately_within_each_horizon",
    }
    assert evaluation["sequential_interval"] == {
        "bootstrap_replicates": 2000,
        "correction": "Bonferroni",
        "generator": "numpy_PCG64",
        "looks": 3,
        "percentile_lower": 0.0083333333333333,
        "percentile_upper": 0.9916666666666667,
        "seed": 147,
        "terminal_time_replaces_next_unreached_cluster_look": True,
        "two_sided_confidence_level": 0.9833333333333333,
        "unit": "paired_independent_cluster",
    }
    assert evaluation["strong_evidence"] == {
        "recall_gain_percentage_points_minimum": 5.0,
        "sequentially_adjusted_interval_lower_bound_above_zero": True,
    }
    assert evaluation["interim_review_rule"] == {
        "cluster_counts": [10, 20],
        "effect_result_may_not_trigger_model_change": True,
        "scientifically_valid_action": "report_and_continue",
    }
    assert evaluation["substantive_progress"] == {"stable_additional_hit_clusters_minimum": 1}
    assert evaluation["secondary_practical_success"] == {
        "same_recall_alarm_area_reduction_fraction_minimum": 0.08,
        "same_recall_alarm_area_reduction_km2_at_primary_budget": 48000,
    }
    assert evaluation["terminal_no_positive_direction_action"] == ("stop_B0_R30_retain_B0")
    assert evaluation["terminal_trigger"] == ("first_of_30_clusters_or_2029-09-10T00:00:00+08:00")
    assert model["forbidden_model_families"] == [
        "decision_tree_ensembles",
        "neural_networks",
    ]
    assert model["development_evidence"] == {
        "d1_close_commit": "0dab57fd1491b5f4924cbae87c0b2001c6fc6b24",
        "d1_final_attribution_result_sha256": (
            "f9fd81887863ac8f6ac174e346a015e7e18cb0f3b9a906b1deea0c4013ec3ec9"
        ),
        "d1_model_commit": "078e950a2b4a837f2ebaaed0d62708012c6e6e23",
        "d1_observed_result_sha256": (
            "6cce1e809b4de4afce37faeade5ecaa9232a5026d777b41dfeec843322b9e804"
        ),
        "role": "retrospective_development_basis_not_prospective_confirmation",
    }
    assert config["targets"]["magnitude_interval"] == "[5.0,6.0)"


def test_record_schema_is_strict_minimal_and_dual_model_only() -> None:
    schema = _load_json(SCHEMA_PATH)
    Draft202012Validator.check_schema(schema)

    assert set(schema["$defs"]) == {
        "Sha256",
        "GitSha1",
        "UtcTimestamp",
        "RecordHeader",
        "ModelForecast",
        "ProtocolDefinition",
        "RealIssueAuthorizationRecord",
        "ForecastIssueRecord",
        "MissedIssueRecord",
        "TruthSnapshotRecord",
        "SequentialReviewRecord",
    }
    model_ids = schema["$defs"]["ModelForecast"]["properties"]["model_id"]["enum"]
    assert model_ids == ["B0", "B0_R30"]
    assert "alarm_ranking_sha256" in schema["$defs"]["ModelForecast"]["required"]
    assert schema["$defs"]["TruthSnapshotRecord"]["allOf"][1]["properties"]["horizon_days"][
        "enum"
    ] == [30, 90]
    triggers = schema["$defs"]["SequentialReviewRecord"]["allOf"][1]["properties"][
        "review_trigger"
    ]["enum"]
    assert triggers == ["cluster_10", "cluster_20", "cluster_30", "time_36_months"]
    issue = schema["$defs"]["ForecastIssueRecord"]["allOf"][1]
    assert issue["properties"]["status"] == {"const": "on_time"}
    assert "publication_completed_at_utc" in issue["required"]
    assert set(issue["required"]) >= {
        "protocol_definition_sha256",
        "authorization_record_sha256",
        "model_manifest_sha256",
        "source_boundary_manifest_sha256",
        "B0_reference_area_km2",
        "actual_area_difference_km2",
    }
    forecast_prefix = issue["properties"]["forecasts"]["prefixItems"]
    assert forecast_prefix[0]["allOf"][1]["properties"]["model_id"] == {"const": "B0"}
    assert forecast_prefix[1]["allOf"][1]["properties"]["model_id"] == {"const": "B0_R30"}
    missed = schema["$defs"]["MissedIssueRecord"]["allOf"][1]
    assert missed["properties"]["status"] == {"const": "missed_issue"}
    assert missed["properties"]["prediction_generated"] == {"const": False}
    assert "forecasts" not in missed["properties"]
    assert missed["additionalProperties"] is False
    protocol = schema["$defs"]["ProtocolDefinition"]["allOf"][1]
    assert protocol["properties"]["chain_sequence"] == {"const": 0}
    assert protocol["properties"]["previous_record_sha256"] == {"type": "null"}
    assert protocol["properties"]["real_issue_authorized"] == {"const": False}
    authorization = schema["$defs"]["RealIssueAuthorizationRecord"]["allOf"][1]
    assert authorization["properties"]["real_issue_authorized"] == {"const": True}
    assert set(authorization["required"]) >= {
        "authorization_commit",
        "remote_verified_at_utc",
        "protocol_definition_sha256",
    }


def test_unavailable_truth_is_null_not_a_false_zero() -> None:
    schema = _load_json(SCHEMA_PATH)
    unavailable = {
        "schema_version": 1,
        "record_type": "TruthSnapshotRecord",
        "recorded_at_utc": "2026-11-09T16:00:00Z",
        "chain_sequence": 4,
        "previous_record_type": "ForecastIssueRecord",
        "previous_record_sha256": "0" * 64,
        "content_sha256": "1" * 64,
        "issue_id": "p1-20260909T160000Z",
        "horizon_days": 30,
        "protocol_definition_sha256": "2" * 64,
        "authorization_record_sha256": "3" * 64,
        "source_snapshot_sha256": None,
        "status": "truth_snapshot_unavailable",
        "mature_after_utc": "2026-11-08T16:00:00Z",
        "truth_fetched_at_utc": "2026-11-09T16:00:00Z",
        "target_event_count": None,
        "independent_cluster_count": None,
        "cluster_assignment_sha256": None,
        "exposure_cluster_registry_sha256": None,
        "magnitude_minimum": 5.0,
        "magnitude_maximum_exclusive": 6.0,
    }
    validator = Draft202012Validator(schema)
    assert not list(validator.iter_errors(unavailable))

    false_zero = {**unavailable, "target_event_count": 0}
    assert list(validator.iter_errors(false_zero))


def test_all_six_record_types_validate_and_form_one_hash_linked_tamper_evident_chain() -> None:
    schema = _load_json(SCHEMA_PATH)
    validator = Draft202012Validator(schema)
    records = _record_chain_examples()

    assert [record["record_type"] for record in records] == [
        "ProtocolDefinition",
        "MissedIssueRecord",
        "RealIssueAuthorizationRecord",
        "ForecastIssueRecord",
        "TruthSnapshotRecord",
        "SequentialReviewRecord",
    ]
    for record in records:
        assert not list(validator.iter_errors(record))
    _assert_chain_semantics(records)

    forged = [dict(record) for record in records]
    forged[4]["previous_record_sha256"] = "f" * 64
    forged[4] = _seal(forged[4])
    try:
        _assert_chain_semantics(forged)
    except ValueError as error:
        assert "hash" in str(error)
    else:
        raise AssertionError("forged predecessor was accepted")


def test_forecast_before_authorization_and_inconsistent_missed_status_are_rejected() -> None:
    schema = _load_json(SCHEMA_PATH)
    validator = Draft202012Validator(schema)
    records = _record_chain_examples()

    forecast_before_authorization = dict(records[3])
    forecast_before_authorization.update(
        {
            "chain_sequence": 1,
            "previous_record_type": "ProtocolDefinition",
            "previous_record_sha256": records[0]["content_sha256"],
        }
    )
    forecast_before_authorization = _seal(forecast_before_authorization)
    assert list(validator.iter_errors(forecast_before_authorization))

    inconsistent_missed = {**records[1], "status": "on_time"}
    inconsistent_missed = _seal(inconsistent_missed)
    assert list(validator.iter_errors(inconsistent_missed))


def test_guard_gap_prevents_the_same_30_day_cluster_crossing_exposures() -> None:
    origin = datetime(2026, 9, 9, 16, tzinfo=UTC)
    issues = [origin + timedelta(days=days) for days in (0, 30, 59, 60, 61, 120)]

    selected = _select_guarded_issues(issues, horizon_days=30)
    assert selected == [origin, origin + timedelta(days=60), origin + timedelta(days=120)]
    for index in range(len(selected) - 1):
        left = selected[index]
        right = selected[index + 1]
        left_window_end = left + timedelta(days=30)
        assert right - left_window_end >= timedelta(days=30)


def test_batch_cluster_arrival_emits_every_crossed_look_in_order() -> None:
    assert _crossed_cluster_looks(9, 11) == [10]
    assert _crossed_cluster_looks(19, 21) == [20]
    assert _crossed_cluster_looks(29, 31) == [30]
    assert _crossed_cluster_looks(8, 25) == [10, 20]
    assert _crossed_cluster_looks(9, 31) == [10, 20, 30]
    assert _crossed_cluster_looks(20, 29) == []

    schema = _load_json(SCHEMA_PATH)
    validator = Draft202012Validator(schema)
    review = _record_chain_examples()[-1]
    missing_horizon = dict(review)
    missing_horizon.pop("horizon_days")
    missing_horizon = _seal(missing_horizon)
    assert list(validator.iter_errors(missing_horizon))

    wrong_count = _seal({**review, "cumulative_cluster_count": 11})
    assert list(validator.iter_errors(wrong_count))
    wrong_look_sequence = _seal({**review, "look_sequence": 2})
    assert list(validator.iter_errors(wrong_look_sequence))
    cluster_10_at_terminal = _seal({**review, "elapsed_months": 36})
    assert list(validator.iter_errors(cluster_10_at_terminal))

    cluster_20 = _seal(
        {
            **review,
            "review_trigger": "cluster_20",
            "prior_completed_look_count": 1,
            "look_sequence": 2,
            "cumulative_cluster_count": 20,
        }
    )
    assert not list(validator.iter_errors(cluster_20))
    cluster_30_tie = _seal(
        {
            **review,
            "review_trigger": "cluster_30",
            "prior_completed_look_count": 2,
            "look_sequence": 3,
            "cumulative_cluster_count": 30,
            "elapsed_months": 36,
            "decision": "report_uncertain_at_final_review",
        }
    )
    assert not list(validator.iter_errors(cluster_30_tie))
    cluster_30_continue = _seal({**cluster_30_tie, "decision": "continue_accumulation"})
    assert list(validator.iter_errors(cluster_30_continue))

    terminal = _seal(
        {
            **review,
            "review_trigger": "time_36_months",
            "prior_completed_look_count": 1,
            "look_sequence": 2,
            "cumulative_cluster_count": 17,
            "elapsed_months": 36,
            "decision": "report_uncertain_at_final_review",
        }
    )
    assert not list(validator.iter_errors(terminal))
    terminal_before_36 = _seal({**terminal, "elapsed_months": 35.9})
    assert list(validator.iter_errors(terminal_before_36))
    wrong_terminal_sequence = _seal({**terminal, "look_sequence": 3})
    assert list(validator.iter_errors(wrong_terminal_sequence))
    wrong_prior_zero_range = _seal(
        {
            **terminal,
            "prior_completed_look_count": 0,
            "look_sequence": 1,
            "cumulative_cluster_count": 10,
        }
    )
    assert list(validator.iter_errors(wrong_prior_zero_range))
    wrong_prior_two_range = _seal(
        {
            **terminal,
            "prior_completed_look_count": 2,
            "look_sequence": 3,
            "cumulative_cluster_count": 19,
        }
    )
    assert list(validator.iter_errors(wrong_prior_two_range))
    terminal_count_30 = _seal({**terminal, "cumulative_cluster_count": 30})
    assert list(validator.iter_errors(terminal_count_30))
    terminal_continue = _seal({**terminal, "decision": "continue_accumulation"})
    assert list(validator.iter_errors(terminal_continue))


def test_zero_cluster_terminal_review_is_evidence_insufficient_not_zero_gain() -> None:
    schema = _load_json(SCHEMA_PATH)
    validator = Draft202012Validator(schema)
    review = _record_chain_examples()[-1]
    zero_terminal = _seal(
        {
            **review,
            "review_trigger": "time_36_months",
            "prior_completed_look_count": 0,
            "look_sequence": 1,
            "cumulative_cluster_count": 0,
            "elapsed_months": 36,
            "B0_hit_clusters": 0,
            "B0_R30_hit_clusters": 0,
            "recall_gain_percentage_points": None,
            "sequentially_adjusted_interval_lower": None,
            "sequentially_adjusted_interval_upper": None,
            "decision": "report_evidence_insufficient_at_final_review",
        }
    )
    assert not list(validator.iter_errors(zero_terminal))

    false_zero_gain = _seal({**zero_terminal, "recall_gain_percentage_points": 0.0})
    assert list(validator.iter_errors(false_zero_gain))
    false_zero_hit = _seal({**zero_terminal, "B0_R30_hit_clusters": 1})
    assert list(validator.iter_errors(false_zero_hit))
    false_zero_decision = _seal({**zero_terminal, "decision": "report_uncertain_at_final_review"})
    assert list(validator.iter_errors(false_zero_decision))

    positive_count_null_effect = _seal(
        {
            **zero_terminal,
            "prior_completed_look_count": 1,
            "look_sequence": 2,
            "cumulative_cluster_count": 15,
            "recall_gain_percentage_points": None,
            "sequentially_adjusted_interval_lower": -10.0,
            "sequentially_adjusted_interval_upper": 10.0,
            "decision": "report_uncertain_at_final_review",
        }
    )
    assert list(validator.iter_errors(positive_count_null_effect))
    positive_count_evidence_insufficient = _seal(
        {
            **positive_count_null_effect,
            "recall_gain_percentage_points": 0.0,
            "decision": "report_evidence_insufficient_at_final_review",
        }
    )
    assert list(validator.iter_errors(positive_count_evidence_insufficient))


def test_challenger_cannot_gain_recall_by_using_more_alarm_area() -> None:
    assert _area_fairness_is_valid(599900, 599500, 500)
    assert not _area_fairness_is_valid(599500, 599900, 500)
    assert not _area_fairness_is_valid(599900, 599275, 625)
    assert not _area_fairness_is_valid(599900, 599500, 400)


def test_current_research_authority_points_to_the_same_frozen_contract() -> None:
    config = _load_yaml()
    research = _load_research_protocol()
    selector = research["current_authority_selector"]
    authority = research["p1_b0_r30_current_authority"]

    assert selector == {
        "future_execution_authority_key": "p1_b0_r30_current_authority",
        "selection_scope": "P1_future_issues_only",
        "exactly_one_current_authority": True,
        "legacy_stage2p_sections_are_historical_only": True,
        "current_authority_must_be_used_for_protocol_code_data_model_and_evaluation": True,
    }

    assert authority["model_ids"] == ["B0", "B0_R30"]
    assert authority["B0_R30_formula"] == "0.75_B0_plus_0.25_R30"
    assert authority["valid_from_local"] == config["calendar"]["fixed_valid_from_local"]
    assert authority["real_issue_authorized"] is False
    assert authority["real_catalog_read_authorized"] is False
    assert authority["real_network_fetch_authorized"] is False
    assert authority["next_authorized_action"] == config["protocol"]["next_authorized_action"]


def test_public_contracts_exclude_non_scientific_receipt_requirements() -> None:
    combined = "\n".join(
        path.read_text(encoding="utf-8").lower()
        for path in (CONFIG_PATH, SCHEMA_PATH, SOURCE_PATH, MODEL_PATH)
    )

    assert "rfc3161" not in combined
    assert "certificate" not in combined
    config = _load_yaml()
    assert config["retired_non_scientific_requirements"] == [
        "third_party_timestamping",
        "public_key_trust_chain_receipts",
        "hardware_identity_receipts",
    ]
