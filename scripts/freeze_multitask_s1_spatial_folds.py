from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

NUMERIC_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
for _name in NUMERIC_THREAD_ENVIRONMENT:
    os.environ[_name] = "1"


EXPECTED_STATUS = "not_materialized_scientific_fallback"
EXPECTED_UNRESOLVED_FIELDS = {
    "test_statistic_and_estimand",
    "null_and_target_blind_alternative",
    "minimum_scientifically_meaningful_effect",
    "type_I_error_rate",
    "target_power",
    "simulation_exposure_duration",
    "pre_evaluation_background_estimator_and_catalog_completeness_rule",
    "per_fold_vs_pooled_success_criterion",
}
FORBIDDEN_PUBLIC_KEY_FRAGMENTS = (
    "longitude",
    "latitude",
    "coordinate_x",
    "coordinate_y",
    "geometry_wkb",
    "raw_zone_id",
    "construction_zone_id",
    "cell_id",
    "zone_to_fold",
    "atomic_block_to_fold",
)


class SpatialFoldFreezeError(RuntimeError):
    """Raised when the target-blind scientific fallback contract is violated."""


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SpatialFoldFreezeError(f"{label} must be a mapping")
    return value


def _sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise SpatialFoldFreezeError(f"{label} must be a sequence")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )


def _safe_project_path(project_root: Path, relative: object, label: str) -> Path:
    candidate = Path(str(relative))
    if candidate.is_absolute() or ".." in candidate.parts:
        raise SpatialFoldFreezeError(f"{label} must be a project-relative path")
    resolved_root = project_root.resolve()
    resolved = (resolved_root / candidate).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise SpatialFoldFreezeError(f"{label} escapes the project root") from exc
    if not resolved.is_file():
        raise SpatialFoldFreezeError(f"missing frozen input: {candidate.as_posix()}")
    return resolved


def _assert_false_flags(value: Mapping[str, Any], label: str) -> None:
    for key, state in value.items():
        if state is not False:
            raise SpatialFoldFreezeError(f"{label}.{key} must remain false")


def _assert_public_safe(value: object, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            if any(
                fragment in normalized for fragment in FORBIDDEN_PUBLIC_KEY_FRAGMENTS
            ) and child is not False:
                raise SpatialFoldFreezeError(f"forbidden public mapping key at {path}.{key}")
            _assert_public_safe(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_public_safe(child, f"{path}[{index}]")


def load_yaml(path: Path) -> Mapping[str, Any]:
    return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), path.name)


def validate_fallback_config(config: Mapping[str, Any]) -> None:
    if config.get("status") != EXPECTED_STATUS:
        raise SpatialFoldFreezeError("scientific fallback status changed")
    if config.get("score_blind") is not True:
        raise SpatialFoldFreezeError("score_blind must remain true")

    decision = _mapping(config.get("frozen_decision"), "frozen_decision")
    if decision.get("selected_k") is not None:
        raise SpatialFoldFreezeError("selected_k must remain null")
    if decision.get("fold_mapping_materialized") is not False:
        raise SpatialFoldFreezeError("fold mapping must not be materialized")
    if decision.get("fold_mapping_sha256") is not None:
        raise SpatialFoldFreezeError("fold mapping hash must remain null")
    if decision.get("candidate_k_evaluated") is not False:
        raise SpatialFoldFreezeError("candidate k values must not be evaluated")
    if decision.get("model_selection_axis") != "nested_time_forward_only":
        raise SpatialFoldFreezeError("model selection must remain time-forward only")

    unresolved = set(
        str(item)
        for item in _sequence(
            config.get("unresolved_power_protocol_fields"),
            "unresolved_power_protocol_fields",
        )
    )
    if unresolved != EXPECTED_UNRESOLVED_FIELDS:
        raise SpatialFoldFreezeError("the under-specified power fields changed")

    withdrawn = _mapping(
        config.get("withdrawn_recipe_recorded_but_not_executed"),
        "withdrawn_recipe_recorded_but_not_executed",
    )
    if list(_sequence(withdrawn.get("candidate_k_in_order"), "candidate_k_in_order")) != [
        5,
        4,
        3,
    ]:
        raise SpatialFoldFreezeError("the historical candidate-k record changed")
    if withdrawn.get("execution_status") != "not_run":
        raise SpatialFoldFreezeError("the withdrawn k recipe must not run")

    retained = _mapping(
        config.get("retained_secondary_spatial_track"),
        "retained_secondary_spatial_track",
    )
    if retained.get("retained") is not True or retained.get("atomic_block_count") != 39:
        raise SpatialFoldFreezeError("the frozen 39-block LOBO track changed")
    if retained.get("may_tune_models_or_hyperparameters") is not False:
        raise SpatialFoldFreezeError("LOBO must not tune models or hyperparameters")
    if retained.get("m6_plus_may_select_or_balance_blocks") is not False:
        raise SpatialFoldFreezeError("M6+ must not select or balance spatial blocks")

    allowed = _mapping(config.get("allowed_selection_inputs_used"), "allowed_selection_inputs_used")
    if allowed.get("frozen_public_input_identities_only") is not True:
        raise SpatialFoldFreezeError("only frozen public identities may be used")
    for key in (
        "frozen_geometry_content_read",
        "pre_evaluation_catalog_used_for_selection",
        "target_blind_power_simulation_run",
    ):
        if allowed.get(key) is not False:
            raise SpatialFoldFreezeError(f"{key} must remain false")

    audit = _mapping(config.get("execution_audit"), "execution_audit")
    expected_audit = {
        "freeze_script_catalog_read": False,
        "pre_decision_preflight_nonspatial_catalog_columns_opened": True,
        "preflight_columns": ["origin_time_utc", "magnitude", "inside_study_area"],
        "preflight_failed_before_producing_a_count": True,
        "longitude_or_latitude_read": False,
        "per_event_values_emitted": False,
        "preflight_result_used_for_decision": False,
    }
    if dict(audit) != expected_audit:
        raise SpatialFoldFreezeError(
            "execution audit must preserve the aborted preflight disclosure"
        )

    _assert_false_flags(
        _mapping(config.get("forbidden_actions"), "forbidden_actions"),
        "forbidden_actions",
    )
    publication = _mapping(config.get("public_output_contract"), "public_output_contract")
    for key in (
        "publish_atomic_block_to_fold_table",
        "publish_raw_zone_ids",
        "publish_cell_ids",
        "publish_coordinates_or_geometry",
    ):
        if publication.get(key) is not False:
            raise SpatialFoldFreezeError(f"public_output_contract.{key} must remain false")


def verify_frozen_inputs(config: Mapping[str, Any], project_root: Path) -> dict[str, object]:
    identities = _mapping(config.get("input_identities"), "input_identities")
    verified: dict[str, object] = {}
    for name in ("s0_config", "s0_science_contract", "s0_spatial_review", "s0_acceptance"):
        record = _mapping(identities.get(name), f"input_identities.{name}")
        path = _safe_project_path(project_root, record.get("path"), f"input_identities.{name}.path")
        actual = _sha256_file(path)
        if actual != record.get("sha256"):
            raise SpatialFoldFreezeError(f"frozen input identity changed: {name}")
        verified[name] = {"sha256": actual, "hash_matches": True}

    manifest_record = _mapping(
        identities.get("spatial_manifest"), "input_identities.spatial_manifest"
    )
    manifest_path = _safe_project_path(
        project_root,
        manifest_record.get("path"),
        "input_identities.spatial_manifest.path",
    )
    manifest_hash = _sha256_file(manifest_path)
    if manifest_hash != manifest_record.get("sha256"):
        raise SpatialFoldFreezeError("frozen spatial manifest identity changed")
    manifest = _mapping(json.loads(manifest_path.read_text(encoding="utf-8")), "spatial_manifest")
    aggregate = _mapping(manifest.get("aggregate"), "spatial_manifest.aggregate")
    security = _mapping(manifest.get("security"), "spatial_manifest.security")
    publication = _mapping(
        manifest.get("publication_safety"), "spatial_manifest.publication_safety"
    )
    local = _mapping(manifest.get("local_artifacts"), "spatial_manifest.local_artifacts")
    cell_mapping = _mapping(
        local.get("cell_mapping"), "spatial_manifest.local_artifacts.cell_mapping"
    )
    if security.get("target_or_score_read") is not False:
        raise SpatialFoldFreezeError("spatial manifest is not target/score blind")
    if any(
        publication.get(key) is not False
        for key in ("contains_coordinates", "contains_geometry", "contains_per_cell_mapping")
    ):
        raise SpatialFoldFreezeError("public spatial manifest contains restricted material")
    if (
        aggregate.get("zone_count") != 65
        or aggregate.get("assigned_nonempty_zone_count") != 39
        or aggregate.get("assigned_query_cell_count") != 15_697
    ):
        raise SpatialFoldFreezeError("frozen 65/39/15697 spatial identity changed")
    if aggregate.get("zone_set_sha256") != identities.get("frozen_zone_set_sha256"):
        raise SpatialFoldFreezeError("frozen zone-set identity changed")
    if cell_mapping.get("sha256") != identities.get("frozen_25km_cell_mapping_file_sha256"):
        raise SpatialFoldFreezeError("frozen cell-mapping identity changed")

    s0_path = _safe_project_path(
        project_root,
        _mapping(identities.get("s0_config"), "input_identities.s0_config").get("path"),
        "input_identities.s0_config.path",
    )
    s0 = load_yaml(s0_path)
    selection = _mapping(
        _mapping(s0.get("spatial_extrapolation"), "s0.spatial_extrapolation").get(
            "model_selection_geometry_folds"
        ),
        "s0.model_selection_geometry_folds",
    )
    if selection.get("current_status") != "deferred_target_blind":
        raise SpatialFoldFreezeError("S0 deferred fold status changed")
    if list(_sequence(selection.get("candidate_k_in_order"), "s0.candidate_k_in_order")) != [
        5,
        4,
        3,
    ]:
        raise SpatialFoldFreezeError("S0 candidate-k record changed")
    if selection.get("m6_plus_may_not_balance_or_select_k") is not True:
        raise SpatialFoldFreezeError("S0 M6+ selection prohibition changed")

    verified["spatial_manifest"] = {
        "sha256": manifest_hash,
        "hash_matches": True,
        "zone_count": 65,
        "nonempty_atomic_block_count": 39,
        "query_cell_count": 15_697,
        "target_or_score_read": False,
        "restricted_content_read": False,
    }
    return verified


def build_decision_record(
    config: Mapping[str, Any],
    *,
    config_sha256: str,
    verified_inputs: Mapping[str, object],
) -> dict[str, object]:
    decision = _mapping(config.get("frozen_decision"), "frozen_decision")
    audit = _mapping(config.get("execution_audit"), "execution_audit")
    record: dict[str, object] = {
        "schema_version": 1,
        "decision_id": config.get("decision_id"),
        "stage": config.get("stage"),
        "status": config.get("status"),
        "score_blind": True,
        "science_value_category": config.get("science_value_category"),
        "decision": {
            "selected_k": None,
            "fold_mapping_materialized": False,
            "fold_mapping_sha256": None,
            "candidate_k_evaluated": False,
            "model_selection_axis": decision.get("model_selection_axis"),
            "reason_code": decision.get("reason_code"),
            "reason": decision.get("reason"),
        },
        "unresolved_power_protocol_fields": list(
            _sequence(
                config.get("unresolved_power_protocol_fields"),
                "unresolved_power_protocol_fields",
            )
        ),
        "retained_secondary_spatial_track": dict(
            _mapping(
                config.get("retained_secondary_spatial_track"),
                "retained_secondary_spatial_track",
            )
        ),
        "verified_input_identities": dict(verified_inputs),
        "execution_receipt": {
            "config_sha256": config_sha256,
            "freeze_script_catalog_read": False,
            "pre_decision_preflight_nonspatial_catalog_columns_opened": True,
            "preflight_columns": list(
                _sequence(audit.get("preflight_columns"), "execution_audit.preflight_columns")
            ),
            "preflight_failed_before_producing_a_count": True,
            "preflight_result_used_for_decision": False,
            "longitude_or_latitude_read": False,
            "per_event_values_emitted": False,
            "evaluation_epicenter_read": False,
            "restricted_geometry_or_mapping_content_read": False,
            "power_simulation_run": False,
            "candidate_k_evaluated": False,
            "model_training_run": False,
            "model_score_read": False,
            "locked_test_run": False,
            "network_accessed": False,
            "numeric_library_threads": 1,
        },
        "public_safety": {
            "contains_raw_zone_ids": False,
            "contains_cell_ids": False,
            "contains_coordinates": False,
            "contains_geometry": False,
            "contains_zone_or_block_to_fold_mapping": False,
        },
        "next_scientific_action": config.get("next_scientific_action"),
    }
    _assert_public_safe(record)
    return record


def write_decision_package(
    *,
    config_path: Path,
    project_root: Path,
    output_root: Path,
) -> tuple[Path, Path]:
    config = load_yaml(config_path)
    validate_fallback_config(config)
    verified_inputs = verify_frozen_inputs(config, project_root)
    config_hash = _sha256_file(config_path)
    decision_record = build_decision_record(
        config,
        config_sha256=config_hash,
        verified_inputs=verified_inputs,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    allowed_names = {
        str(
            _mapping(config.get("public_output_contract"), "public_output_contract").get(
                "decision_filename"
            )
        ),
        str(
            _mapping(config.get("public_output_contract"), "public_output_contract").get(
                "manifest_filename"
            )
        ),
    }
    unexpected = sorted(
        path.name for path in output_root.iterdir() if path.name not in allowed_names
    )
    if unexpected:
        raise SpatialFoldFreezeError(f"unexpected file in decision output directory: {unexpected}")

    publication = _mapping(config.get("public_output_contract"), "public_output_contract")
    decision_path = output_root / str(publication.get("decision_filename"))
    manifest_path = output_root / str(publication.get("manifest_filename"))
    decision_path.write_bytes(_canonical_json_bytes(decision_record))
    decision_hash = _sha256_file(decision_path)
    audit = _mapping(config.get("execution_audit"), "execution_audit")
    manifest: dict[str, object] = {
        "schema_version": 1,
        "decision_id": config.get("decision_id"),
        "status": EXPECTED_STATUS,
        "score_blind": True,
        "selected_k": None,
        "fold_mapping_materialized": False,
        "artifact_count": 1,
        "artifacts": [
            {
                "path": decision_path.name,
                "byte_count": decision_path.stat().st_size,
                "sha256": decision_hash,
            }
        ],
        "config_sha256": config_hash,
        "freeze_script_catalog_read": False,
        "pre_decision_preflight_nonspatial_catalog_columns_opened": True,
        "preflight_columns": list(
            _sequence(audit.get("preflight_columns"), "execution_audit.preflight_columns")
        ),
        "preflight_failed_before_producing_a_count": True,
        "preflight_result_used_for_decision": False,
        "evaluation_epicenter_read": False,
        "model_score_read": False,
        "locked_test_run": False,
        "network_accessed": False,
        "public_contains_restricted_spatial_material": False,
    }
    _assert_public_safe(manifest)
    manifest_path.write_bytes(_canonical_json_bytes(manifest))
    return decision_path, manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the S1 target-blind spatial model-selection fallback."
    )
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--project-root",
        type=Path,
        default=project_root,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "configs" / "multitask_s1_spatial_folds.yaml",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "outputs" / "multitask_s1" / "spatial_fold_freeze_v1",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    decision_path, manifest_path = write_decision_package(
        config_path=args.config.resolve(),
        project_root=args.project_root.resolve(),
        output_root=args.output_root.resolve(),
    )
    print(
        json.dumps(
            {
                "status": EXPECTED_STATUS,
                "selected_k": None,
                "decision": decision_path.as_posix(),
                "manifest": manifest_path.as_posix(),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
