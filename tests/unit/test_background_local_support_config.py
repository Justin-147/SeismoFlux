from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from seismoflux.background.config import (
    BackgroundConfig,
    BackgroundLocalSupportConfig,
    LocalSupportPolicyConfig,
    load_background_config,
    load_background_protocol,
)
from seismoflux.config import load_yaml_mapping

BACKGROUND_CONFIG = Path("configs/background.yaml")
LOCAL_SUPPORT_CONFIG = Path("configs/background_local_support.yaml")
FOLD_MANIFEST = Path("data/manifests/background_local_support_fold_manifest.json")
PARENT_FOLD_MANIFEST = Path("data/manifests/background_fold_manifest.json")


def _canonical_sha256(config: BackgroundConfig) -> str:
    payload = json.dumps(
        config.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def test_v020_protocol_bytes_and_canonical_behavior_remain_frozen() -> None:
    assert hashlib.sha256(BACKGROUND_CONFIG.read_bytes()).hexdigest() == (
        "d845f7aea4b8cc50f264560a074e93c91efc5cf026a1425f04f1c160d595e24b"
    )
    config = load_background_protocol(BACKGROUND_CONFIG)
    assert type(config) is BackgroundConfig
    assert _canonical_sha256(config) == (
        "b36ce50f7f4df6d712c743bb28ce4f1fd05dfdbc3d5026b4bf75d00477765c6d"
    )
    assert config.outputs.input_hashes_required_keys == (
        "environment_lock",
        "data_catalog",
        "earthquake_dataset",
        "study_area",
        "issue_manifest",
        "production_fixture",
        "oracle_metadata",
    )


def test_v021_protocol_dispatches_to_independent_local_support_model() -> None:
    config = load_background_protocol(LOCAL_SUPPORT_CONFIG)

    assert type(config) is BackgroundLocalSupportConfig
    assert config.protocol_version == "0.2.1"
    assert config.freeze_tag == "v0.2.1-background-local-support-protocol"
    assert config.background_scores_seen_before_freeze is False
    assert config.completeness_diagnostics_seen_before_freeze is True
    assert config.parent_protocol_result_immutable is True
    assert config.local_support.gate_name == "G1-LS"
    assert config.local_support.base_cell_km == 500.0
    assert config.local_support.sparse_parent_cell_km == 1000.0
    assert config.local_support.minimum_events_per_stratum == 200
    assert config.local_support.maximum_supported_raw_mc == 4.0
    assert config.local_support.minimum_supported_area_fraction == 0.95
    assert config.local_support.temporal_above_maximum_action == ("hard_fail_entire_experiment")
    assert config.local_support.full_region_metrics_unsupported_targets == ("count_as_misses")
    assert config.local_support.locked_test_action == "do_not_run"
    assert config.outputs.input_hashes_required_keys[-1] == "support_manifest"
    assert _canonical_sha256(config) == (
        "5dd31a212f4894625cf9db1bbfb1de3a4861a9bcffaeef9676c06707c03eceb8"
    )


def test_v021_nested_model_domains_are_explicitly_supported_only() -> None:
    config = load_background_protocol(LOCAL_SUPPORT_CONFIG)
    assert isinstance(config, BackgroundLocalSupportConfig)

    assert config.completeness.target_event_domain == "supported_domain_only"
    assert config.completeness.no_eligible_temporal_or_spatial_strata_action == (
        "fail_if_no_eligible_temporal_stratum"
    )
    assert config.uniform_poisson.rate_estimator == (
        "supported_event_count_over_supported_equal_area_days"
    )
    assert config.spatial_poisson.fitting_events == "supported_domain_only"
    assert config.spatial_poisson.spatial_density_integral_over_supported_domain == 1.0
    assert not hasattr(config.spatial_poisson, "spatial_density_integral_over_study_area")
    assert config.etas.background_component.density_domain == "supported_domain"
    assert config.etas.likelihood.compensator_domain == "supported_domain_only"
    assert config.etas.simulation.target_domain == "supported_domain_only"
    assert config.integration.inclusion_rule == (
        "positive_area_intersection_with_snapshot_supported_domain"
    )
    assert config.evaluation.primary_target.endswith("inside_supported_domain_only")
    assert "supported_physical_target_event_count" in (
        config.evaluation.g1_primary_endpoint.score_formula
    )
    assert config.evaluation.g1_pass_rule.gate_name == "G1-LS"
    assert config.evaluation.g1_pass_rule.comparison_domain == ("same_snapshot_supported_domain")
    assert config.evaluation.model_selection.paired_difference == (
        "candidate_minus_validation_best_on_same_supported_fold_events_and_compensator"
    )
    assert config.randomness.seed_derivation.namespace_contexts.bootstrap.issue_id.startswith(
        "g1_ls_primary_supported_validation_"
    )


def test_v021_paths_are_disjoint_and_manifest_digest_is_replaceable() -> None:
    config = load_background_protocol(LOCAL_SUPPORT_CONFIG)
    assert isinstance(config, BackgroundLocalSupportConfig)

    assert config.inputs.issue_manifest == FOLD_MANIFEST.as_posix()
    assert config.inputs.support_manifest == (
        "data/manifests/background_local_support_manifest.json"
    )
    assert len(config.inputs.support_manifest_sha256) == 64
    assert config.outputs.processed_root == "data/processed/stage2R/local_support"
    assert config.outputs.backtest_root == "outputs/backtests/background_local_support"
    assert config.outputs.experiment_root == "outputs/experiments/background_local_support"
    assert config.outputs.model_root == "models/registry/background_local_support"
    assert config.outputs.registry == (
        "data/manifests/background_local_support_model_registry.json"
    )
    assert config.outputs.report == "docs/background_local_support_report.md"


def test_real_support_manifest_is_content_addressed_and_strictly_loaded() -> None:
    config = load_background_protocol(LOCAL_SUPPORT_CONFIG)
    assert isinstance(config, BackgroundLocalSupportConfig)
    manifest_path = Path(config.inputs.support_manifest)
    assert manifest_path.is_file()
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == (
        config.inputs.support_manifest_sha256
    )

    loaded = load_background_config(LOCAL_SUPPORT_CONFIG)
    assert isinstance(loaded, BackgroundLocalSupportConfig)
    assert loaded.inputs.support_manifest_sha256 == (
        "632278416dfc717dbcb9d2eae048a4f13cdf7737a31e6e5e704a9dd17d7cef8d"
    )


def test_v021_fold_manifest_changes_only_freeze_identity() -> None:
    config = load_background_protocol(LOCAL_SUPPORT_CONFIG)
    local = json.loads(FOLD_MANIFEST.read_text(encoding="utf-8"))
    parent = json.loads(PARENT_FOLD_MANIFEST.read_text(encoding="utf-8"))

    assert hashlib.sha256(FOLD_MANIFEST.read_bytes()).hexdigest() == (
        config.inputs.issue_manifest_sha256
    )
    assert local["frozen_on"] == "2026-07-14"
    assert local["freeze_tag"] == config.freeze_tag
    assert local["scores_seen_before_freeze"] is False
    for key in (
        "schema_version",
        "scores_seen_before_freeze",
        "source",
        "semantics",
        "partitions",
        "cross_checks",
    ):
        assert local[key] == parent[key]


def test_version_dispatch_rejects_unknown_protocol(tmp_path: Path) -> None:
    raw = deepcopy(load_yaml_mapping(LOCAL_SUPPORT_CONFIG))
    raw["protocol_version"] = "0.2.999"
    path = tmp_path / "unknown_background.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported background protocol_version"):
        load_background_protocol(path)


def test_v021_mapping_remains_strict_and_cannot_use_v020_model() -> None:
    raw = deepcopy(load_yaml_mapping(LOCAL_SUPPORT_CONFIG))
    raw["unexpected_result_field"] = "forbidden"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        BackgroundLocalSupportConfig.model_validate(raw)

    pristine = load_yaml_mapping(LOCAL_SUPPORT_CONFIG)
    with pytest.raises(ValidationError):
        BackgroundConfig.model_validate(pristine)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gate_name", "G1"),
        ("base_cell_km", 499.0),
        ("sparse_parent_cell_km", 999.0),
        ("grid_origin_m", [1.0, 0.0]),
        ("minimum_events_per_stratum", 199),
        ("maximum_supported_raw_mc", 4.5),
        ("temporal_above_maximum_action", "mask_temporal_failure"),
        ("minimum_supported_area_fraction", 0.94),
        ("minimum_supported_area_boundary_inclusive", False),
        ("later_snapshot_backfill_forbidden", False),
        ("full_region_metrics_unsupported_targets", "drop_from_denominator"),
        ("locked_test_action", "run"),
    ],
)
def test_local_support_scientific_policy_drift_is_rejected(
    field: str,
    value: object,
) -> None:
    policy = deepcopy(load_yaml_mapping(LOCAL_SUPPORT_CONFIG)["local_support"])
    assert isinstance(policy, dict)
    policy[field] = value

    with pytest.raises(ValidationError):
        LocalSupportPolicyConfig.model_validate(policy)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("completeness", "target_event_domain"), "inside_study_area_only"),
        (("uniform_poisson", "rate_estimator"), "inside_event_count_over_equal_area_days"),
        (
            ("spatial_poisson", "mixture_boundary_normalization"),
            "normalize_complete_mixture_once_over_study_area",
        ),
        (("spatial_poisson", "fitting_events"), "inside_study_area_only"),
        (("etas", "background_component", "density_domain"), "study_area"),
        (("etas", "likelihood", "compensator_domain"), "study_area_only"),
        (("etas", "simulation", "target_domain"), "study_area_only"),
        (
            ("integration", "inclusion_rule"),
            "positive_area_intersection_with_study_area",
        ),
        (
            ("evaluation", "primary_target"),
            "unmarked_events_at_or_above_selected_completeness_magnitude_inside_study_area_only",
        ),
        (
            ("evaluation", "issue_based_horizon_backtests", "denominator"),
            "unique_inside_physical_target_events_per_partition_and_horizon",
        ),
        (
            ("evaluation", "model_selection", "paired_difference"),
            "candidate_minus_validation_best_on_same_fold_events_and_compensator",
        ),
    ],
)
def test_v021_nested_supported_domain_semantics_cannot_drift(
    path: tuple[str, ...],
    value: object,
) -> None:
    raw = deepcopy(load_yaml_mapping(LOCAL_SUPPORT_CONFIG))
    node: object = raw
    for component in path[:-1]:
        assert isinstance(node, dict)
        node = node[component]
    assert isinstance(node, dict)
    node[path[-1]] = value

    with pytest.raises(ValidationError):
        BackgroundLocalSupportConfig.model_validate(raw)
