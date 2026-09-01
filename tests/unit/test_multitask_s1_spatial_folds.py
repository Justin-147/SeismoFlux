from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest


def _load_freezer() -> object:
    path = Path(__file__).parents[2] / "scripts" / "freeze_multitask_s1_spatial_folds.py"
    spec = importlib.util.spec_from_file_location("seismoflux_s1_spatial_freezer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


freezer = _load_freezer()
PROJECT_ROOT = Path(__file__).parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "multitask_s1_spatial_folds.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fallback_is_score_blind_and_forces_single_numeric_thread() -> None:
    config = freezer.load_yaml(CONFIG_PATH)
    freezer.validate_fallback_config(config)
    verified = freezer.verify_frozen_inputs(config, PROJECT_ROOT)

    assert config["status"] == "not_materialized_scientific_fallback"
    assert config["frozen_decision"]["selected_k"] is None
    assert config["frozen_decision"]["candidate_k_evaluated"] is False
    assert config["frozen_decision"]["model_selection_axis"] == "nested_time_forward_only"
    assert verified["spatial_manifest"]["nonempty_atomic_block_count"] == 39
    assert all(os.environ[name] == "1" for name in freezer.NUMERIC_THREAD_ENVIRONMENT)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("selected_k", 5, "selected_k must remain null"),
        ("fold_mapping_materialized", True, "fold mapping must not be materialized"),
        ("candidate_k_evaluated", True, "candidate k values must not be evaluated"),
    ],
)
def test_freeze_fails_closed_if_a_candidate_or_mapping_is_reintroduced(
    field: str,
    value: object,
    match: str,
) -> None:
    config = copy.deepcopy(freezer.load_yaml(CONFIG_PATH))
    config["frozen_decision"][field] = value
    with pytest.raises(freezer.SpatialFoldFreezeError, match=match):
        freezer.validate_fallback_config(config)


def test_public_decision_package_contains_no_mapping_or_target_access(tmp_path: Path) -> None:
    first_decision, first_manifest = freezer.write_decision_package(
        config_path=CONFIG_PATH,
        project_root=PROJECT_ROOT,
        output_root=tmp_path,
    )
    first_hashes = (_sha256(first_decision), _sha256(first_manifest))
    second_decision, second_manifest = freezer.write_decision_package(
        config_path=CONFIG_PATH,
        project_root=PROJECT_ROOT,
        output_root=tmp_path,
    )

    assert first_hashes == (_sha256(second_decision), _sha256(second_manifest))
    decision = json.loads(second_decision.read_text(encoding="utf-8"))
    manifest = json.loads(second_manifest.read_text(encoding="utf-8"))
    encoded = json.dumps(decision, ensure_ascii=False).lower()
    assert decision["decision"]["selected_k"] is None
    assert decision["decision"]["fold_mapping_materialized"] is False
    assert decision["execution_receipt"]["freeze_script_catalog_read"] is False
    assert (
        decision["execution_receipt"][
            "pre_decision_preflight_nonspatial_catalog_columns_opened"
        ]
        is True
    )
    assert decision["execution_receipt"]["preflight_failed_before_producing_a_count"] is True
    assert decision["execution_receipt"]["preflight_columns"] == [
        "origin_time_utc",
        "magnitude",
        "inside_study_area",
    ]
    assert decision["execution_receipt"]["preflight_result_used_for_decision"] is False
    assert decision["execution_receipt"]["longitude_or_latitude_read"] is False
    assert decision["execution_receipt"]["evaluation_epicenter_read"] is False
    assert decision["execution_receipt"]["model_score_read"] is False
    assert decision["execution_receipt"]["power_simulation_run"] is False
    assert "construction_zone_id" not in encoded
    assert '"cell_id"' not in encoded
    assert '"longitude":' not in encoded
    assert '"latitude":' not in encoded
    assert manifest["artifact_count"] == 1
    assert manifest["preflight_columns"] == [
        "origin_time_utc",
        "magnitude",
        "inside_study_area",
    ]
    assert manifest["artifacts"][0]["sha256"] == _sha256(second_decision)


def test_public_safety_rejects_positive_restricted_mapping_payload() -> None:
    with pytest.raises(freezer.SpatialFoldFreezeError, match="forbidden public mapping key"):
        freezer._assert_public_safe({"cell_id": "private-cell"})
    freezer._assert_public_safe({"contains_cell_ids": False})
