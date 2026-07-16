"""Target-blind qualification evidence for the stage-4 scoring-code freeze."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Final, Literal, cast

from seismoflux.anomaly_increment.compute import BackendEquivalenceEvidence
from seismoflux.anomaly_increment.immutable_file import (
    UnsafeImmutableFileError,
    read_existing_immutable_bytes,
    require_existing_real_directory_tree,
    sha256_existing_immutable_file,
)
from seismoflux.anomaly_increment.preregistration import (
    build_random_input_seal,
    protocol_design_sha256,
    validate_stage4_protocol_bundle,
    verify_content_sha256,
    with_content_sha256,
)
from seismoflux.anomaly_increment.score_blind_path import (
    require_score_blind_project_path,
)
from seismoflux.data.common import canonical_json_bytes

if TYPE_CHECKING:
    from seismoflux.anomaly_increment.formal_preflight import FormalPreflightReceipt

STAGE4_PROTOCOL_VERSION: Final[str] = "0.4.1"
STAGE4_QUALIFICATION_SCHEMA_VERSION: Final[int] = 4
REQUIRED_SCORE_BLIND_QUALIFICATIONS: Final[tuple[str, ...]] = (
    "generated_manifest_hashes_verified",
    "topology_gate_passed",
    "all_non_target_tests_passed",
    "synthetic_end_to_end_passed",
    "cpu_float64_numerical_regression_passed",
    "restricted_spatial_artifact_hashes_verified",
    "identity_time_mapping_reproduces_accepted_snapshot_and_trajectory_columns",
    "identity_space_mapping_reproduces_accepted_200km_scientific_columns",
    "placebo_coverage_values_and_null_bitmap_exactly_unchanged",
    "fixed_small_permutation_matches_stage3_low_level_reference",
    "worker_count_invariance_passed",
    "logical_arrow_identity_r1_verified",
)
FROZEN_FULL_NON_TARGET_TEST_COUNT: Final[int] = 1078
REQUIRED_TESTS_BY_QUALIFICATION: Final[dict[str, tuple[str, ...]]] = {
    "all_non_target_tests_passed": (),
    "synthetic_end_to_end_passed": (
        "tests.unit.test_stage4_anomaly_increment_background_adapter::"
        "test_adapter_has_no_file_read_path",
        "tests.unit.test_stage4_anomaly_increment_background_adapter::"
        "test_future_values_cannot_change_development_background_and_rebuild_is_deterministic",
        "tests.unit.test_stage4_anomaly_increment_background_adapter::"
        "test_study_support_and_grid_identities_are_verified_before_fit",
        "tests.unit.test_stage4_anomaly_increment_feature_adapter::"
        "test_issue_table_grid_identity_and_null_bitmap_are_preserved",
        "tests.unit.test_stage4_anomaly_increment_feature_adapter::"
        "test_real_feature_manifest_builds_exact_nested_executable_contracts",
        "tests.unit.test_stage4_anomaly_increment_deliverables::"
        "test_interactive_template_structure_does_not_change_with_scientific_outcome",
        "tests.unit.test_stage4_anomaly_increment_deliverables::"
        "test_prospective_type_and_serialized_payload_physically_exclude_evaluation_fields",
        "tests.unit.test_stage4_anomaly_increment_deliverables::"
        "test_static_svg_is_valid_coordinate_free_and_displays_required_boundaries",
        "tests.unit.test_stage4_anomaly_increment_pipeline::"
        "test_negative_and_insufficient_synthetic_e2e_stop_conservatively"
        "[insufficient-evidence_insufficient-evidence_insufficient-evidence_insufficient]",
        "tests.unit.test_stage4_anomaly_increment_pipeline::"
        "test_negative_and_insufficient_synthetic_e2e_stop_conservatively"
        "[negative-failed-failed-credible_negative]",
        "tests.unit.test_stage4_anomaly_increment_pipeline::"
        "test_pipeline_module_has_no_path_io_or_target_loader",
        "tests.unit.test_stage4_anomaly_increment_pipeline::"
        "test_pipeline_is_deterministic_and_no_anomaly_variants_degenerate_to_background",
        "tests.unit.test_stage4_anomaly_increment_pipeline::"
        "test_placebo_refit_reuses_rate_head_and_prospective_type_is_physically_target_blind",
        "tests.unit.test_stage4_anomaly_increment_pipeline::"
        "test_positive_synthetic_e2e_passes_gates_and_builds_separate_outputs",
        "tests.unit.test_stage4_anomaly_increment_spatial_dashboard::"
        "test_spatial_dashboard_physically_separates_forecast_and_target_payloads",
    ),
    "cpu_float64_numerical_regression_passed": (
        "tests.unit.test_stage4_anomaly_increment_math::"
        "test_ridge_poisson_objective_and_gradient_match_hand_calculation",
        "tests.unit.test_stage4_anomaly_increment_math::"
        "test_analytic_gradient_matches_central_finite_difference",
        "tests.unit.test_stage4_anomaly_increment_runtime::"
        "test_numpy_float64_backend_rejects_nonfinite_and_shape_drift",
        "tests.unit.test_stage4_anomaly_increment_scalable_model::"
        "test_grouped_objective_is_float64_equivalent_to_dense_midpoint_expansion",
        "tests.unit.test_stage4_anomaly_increment_scalable_model::"
        "test_grouped_objective_uses_one_design_copy_and_shared_optimizer",
    ),
    "identity_time_mapping_reproduces_accepted_snapshot_and_trajectory_columns": (
        "tests.unit.test_stage4_anomaly_increment_placebo_features::"
        "test_identity_time_mapping_reproduces_accepted_snapshot_and_trajectory_columns",
    ),
    "identity_space_mapping_reproduces_accepted_200km_scientific_columns": (
        "tests.unit.test_stage4_anomaly_increment_placebo_features::"
        "test_identity_space_mapping_reproduces_accepted_200km_scientific_columns",
    ),
    "placebo_coverage_values_and_null_bitmap_exactly_unchanged": (
        "tests.unit.test_stage4_anomaly_increment_placebo_features::"
        "test_placebo_coverage_values_and_null_bitmap_exactly_unchanged",
    ),
    "fixed_small_permutation_matches_stage3_low_level_reference": (
        "tests.unit.test_stage4_anomaly_increment_placebo_features::"
        "test_fixed_small_permutations_match_stage3_low_level_and_are_deterministic",
        "tests.unit.test_stage4_anomaly_increment_protocol::"
        "test_typed_rng_vectors_cover_zero_one_and_last_for_every_legal_context",
    ),
    "worker_count_invariance_passed": (
        "tests.unit.test_stage4_anomaly_increment_runtime::"
        "test_worker_count_qualification_preserves_result_order_and_bytes",
    ),
    "logical_arrow_identity_r1_verified": (
        "tests.unit.test_stage4_anomaly_increment_logical_identity_r1::"
        "test_selected_table_logical_identity_r1_covers_frozen_canonicalization_contract",
        "tests.unit.test_stage4_anomaly_increment_logical_identity_r1::"
        "test_r1_identity_zeroes_only_null_fixed_width_payload[signed-int8]",
        "tests.unit.test_stage4_anomaly_increment_logical_identity_r1::"
        "test_r1_identity_zeroes_only_null_fixed_width_payload[unsigned-int32]",
        "tests.unit.test_stage4_anomaly_increment_logical_identity_r1::"
        "test_r1_identity_zeroes_only_null_fixed_width_payload[float64]",
        "tests.unit.test_stage4_anomaly_increment_logical_identity_r1::"
        "test_r1_identity_zeroes_only_null_fixed_width_payload[timestamp-ns-utc]",
        "tests.unit.test_stage4_anomaly_increment_logical_identity_r1::"
        "test_r1_identity_canonicalizes_boolean_null_payload_and_padding",
        "tests.unit.test_stage4_anomaly_increment_logical_identity_r1::"
        "test_r1_identity_canonicalizes_utf8_null_payload_but_preserves_valid_bytes",
        "tests.unit.test_stage4_anomaly_increment_logical_identity_r1::"
        "test_r1_identity_preserves_valid_float_payload_bits",
        "tests.unit.test_stage4_anomaly_increment_logical_identity_r1::"
        "test_r1_identity_excludes_top_metadata_but_preserves_fields_exactly",
        "tests.unit.test_stage4_anomaly_increment_logical_identity_r1::"
        "test_r1_identity_fails_closed_for_unsupported_arrow_types[nested-list]",
        "tests.unit.test_stage4_anomaly_increment_logical_identity_r1::"
        "test_r1_identity_fails_closed_for_unsupported_arrow_types[binary]",
        "tests.unit.test_stage4_anomaly_increment_logical_identity_r1::"
        "test_r1_identity_fails_closed_for_unsupported_arrow_types[date32]",
        "tests.unit.test_stage4_anomaly_increment_logical_identity_r1::"
        "test_r1_identity_fails_closed_for_unsupported_arrow_types[dictionary]",
        "tests.unit.test_stage4_anomaly_increment_logical_identity_r1::"
        "test_r1_identity_fails_closed_for_unsupported_arrow_types[extension]",
    ),
}

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_GIT_OID_PATTERN = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_TARGET_IDENTITY_FIELDS: Final[tuple[str, ...]] = (
    "path",
    "sha256",
    "content_sha256",
    "schema_sha256",
    "contract_path",
    "contract_sha256",
    "physical_event_id_column",
)
_PYTEST_EVIDENCE_SENTINEL = object()
_SCORE_BLIND_INPUT_QUALIFICATIONS: Final[frozenset[str]] = frozenset(
    {
        "generated_manifest_hashes_verified",
        "restricted_spatial_artifact_hashes_verified",
        "topology_gate_passed",
    }
)
_REPOSITORY_AND_LEDGER_GATES: Final[frozenset[str]] = frozenset(
    {
        "execution_seal_verified",
        "formal_attempt_count_equals_zero",
        "pre_score_tag_pushed",
        "protocol_commit_pushed",
        "scoring_code_commit_pushed",
        "scoring_code_tag_pushed",
        "target_read_count_equals_zero",
    }
)


class Stage4QualificationError(RuntimeError):
    """Raised when score-blind evidence is absent, altered, or incomplete."""


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise Stage4QualificationError(f"{label} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise Stage4QualificationError(f"{label} must be a lowercase SHA-256 string")
    return value


def _git_oid(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _GIT_OID_PATTERN.fullmatch(value) is None:
        raise Stage4QualificationError(f"{label} must be a lowercase Git object ID")
    return value


def _relative_posix(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise Stage4QualificationError(f"{label} must be a normalized project-relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise Stage4QualificationError(f"{label} must be a normalized project-relative path")
    return value


def _project_path(
    project_root: Path,
    protocol: Mapping[str, object],
    value: object,
    *,
    label: str,
) -> Path:
    relative = _relative_posix(value, label=label)
    root = Path(project_root).resolve()
    path = Path(os.path.abspath(os.fspath(root.joinpath(*PurePosixPath(relative).parts))))
    try:
        return require_score_blind_project_path(
            root,
            protocol,
            path,
            label=label,
        )
    except ValueError as exc:
        raise Stage4QualificationError(f"unsafe score-blind path: {label}") from exc


def expected_target_identity_from_protocol(protocol: Mapping[str, object]) -> dict[str, str]:
    """Copy only the preregistered target identity; never construct or inspect its path."""

    inputs = _mapping(protocol.get("inputs"), label="inputs")
    target = _mapping(inputs.get("earthquake_target"), label="inputs.earthquake_target")
    missing = set(_TARGET_IDENTITY_FIELDS) - set(target)
    if missing:
        raise Stage4QualificationError(
            f"earthquake target identity is incomplete: {sorted(missing)}"
        )
    identity = {field: target[field] for field in _TARGET_IDENTITY_FIELDS}
    _relative_posix(identity["path"], label="earthquake target expected path")
    _relative_posix(identity["contract_path"], label="earthquake target contract path")
    for hash_field in ("sha256", "content_sha256", "schema_sha256", "contract_sha256"):
        _sha256(identity[hash_field], label=f"earthquake target {hash_field}")
    event_id_column = identity["physical_event_id_column"]
    if not isinstance(event_id_column, str) or event_id_column != "event_id":
        raise Stage4QualificationError("earthquake target physical-event identity changed")
    if target.get("unavailable_before_protocol_freeze") is not True:
        raise Stage4QualificationError("earthquake target score-blind declaration changed")
    if target.get("human_prediction_fields_forbidden") is not True:
        raise Stage4QualificationError("human prediction-field prohibition changed")
    return cast(dict[str, str], identity)


@dataclass(frozen=True, slots=True)
class ScoreBlindInputEvidence:
    """Fresh observations of every allowed pre-target input boundary."""

    protocol_design_sha256: str
    random_input_seal_sha256: str
    protocol_validation_sha256: str
    observed_project_input_hashes: tuple[tuple[str, str], ...]
    generated_manifest_hashes: tuple[tuple[str, str], ...]
    restricted_spatial_artifact_hashes: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        for label, value in (
            ("protocol_design_sha256", self.protocol_design_sha256),
            ("random_input_seal_sha256", self.random_input_seal_sha256),
            ("protocol_validation_sha256", self.protocol_validation_sha256),
        ):
            _sha256(value, label=label)
        for group_label, values in (
            ("observed_project_input_hashes", self.observed_project_input_hashes),
            ("generated_manifest_hashes", self.generated_manifest_hashes),
            ("restricted_spatial_artifact_hashes", self.restricted_spatial_artifact_hashes),
        ):
            names = tuple(name for name, _ in values)
            if names != tuple(sorted(set(names))):
                raise ValueError(f"{group_label} names must be sorted and unique")
            for name, digest in values:
                if not name or name != name.strip():
                    raise ValueError(f"{group_label} contains an invalid name")
                _sha256(digest, label=f"{group_label}.{name}")
        if any(name == "earthquake_target" for name, _ in self.observed_project_input_hashes):
            raise ValueError("score-blind evidence must never observe the earthquake target")

    def as_mapping(self) -> dict[str, object]:
        return with_content_sha256(
            {
                "generated_manifest_hashes": dict(self.generated_manifest_hashes),
                "observed_project_input_hashes": dict(self.observed_project_input_hashes),
                "protocol_design_sha256": self.protocol_design_sha256,
                "protocol_validation_sha256": self.protocol_validation_sha256,
                "random_input_seal_sha256": self.random_input_seal_sha256,
                "restricted_spatial_artifact_hashes": dict(self.restricted_spatial_artifact_hashes),
                "schema_version": STAGE4_QUALIFICATION_SCHEMA_VERSION,
                "target_bytes_read": False,
                "target_path_observed": False,
            }
        )

    @property
    def content_sha256(self) -> str:
        return cast(str, self.as_mapping()["content_sha256"])

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ScoreBlindInputEvidence:
        expected = {
            "content_sha256",
            "generated_manifest_hashes",
            "observed_project_input_hashes",
            "protocol_design_sha256",
            "protocol_validation_sha256",
            "random_input_seal_sha256",
            "restricted_spatial_artifact_hashes",
            "schema_version",
            "target_bytes_read",
            "target_path_observed",
        }
        if set(value) != expected or not verify_content_sha256(value):
            raise Stage4QualificationError("score-blind input evidence hash or schema is invalid")
        if (
            value.get("schema_version") != STAGE4_QUALIFICATION_SCHEMA_VERSION
            or value.get("target_bytes_read") is not False
            or value.get("target_path_observed") is not False
        ):
            raise Stage4QualificationError("score-blind input evidence crossed the target boundary")

        def pairs(field: str) -> tuple[tuple[str, str], ...]:
            raw = _mapping(value.get(field), label=field)
            return tuple(
                sorted(
                    (name, _sha256(digest, label=f"{field}.{name}")) for name, digest in raw.items()
                )
            )

        try:
            return cls(
                protocol_design_sha256=cast(str, value["protocol_design_sha256"]),
                random_input_seal_sha256=cast(str, value["random_input_seal_sha256"]),
                protocol_validation_sha256=cast(str, value["protocol_validation_sha256"]),
                observed_project_input_hashes=pairs("observed_project_input_hashes"),
                generated_manifest_hashes=pairs("generated_manifest_hashes"),
                restricted_spatial_artifact_hashes=pairs("restricted_spatial_artifact_hashes"),
            )
        except (TypeError, ValueError) as exc:
            raise Stage4QualificationError("score-blind input evidence invariants failed") from exc


def _verified_payload(
    path: Path,
    expected: object,
    *,
    label: str,
) -> tuple[bytes, str]:
    expected_sha256 = _sha256(expected, label=f"{label} expected SHA-256")
    try:
        payload = read_existing_immutable_bytes(path, label=label)
    except UnsafeImmutableFileError as exc:
        raise Stage4QualificationError(f"cannot read score-blind input {label}") from exc
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise Stage4QualificationError(f"score-blind input changed: {label}")
    return payload, actual


def _verify_hash(path: Path, expected: object, *, label: str) -> str:
    expected_sha256 = _sha256(expected, label=f"{label} expected SHA-256")
    try:
        actual = sha256_existing_immutable_file(path, label=label)
    except UnsafeImmutableFileError as exc:
        raise Stage4QualificationError(f"cannot read score-blind input {label}") from exc
    if actual != expected_sha256:
        raise Stage4QualificationError(f"score-blind input changed: {label}")
    return actual


def observe_score_blind_inputs(
    project_root: Path,
    protocol: Mapping[str, object],
) -> ScoreBlindInputEvidence:
    """Verify all permitted pre-target bytes while structurally excluding the target path."""

    if protocol.get("protocol_version") != STAGE4_PROTOCOL_VERSION:
        raise Stage4QualificationError("stage-4 protocol version changed")
    expected_target_identity_from_protocol(protocol)
    locked = _mapping(protocol.get("locked_test"), label="locked_test")
    expected_locked = {
        "action": "do_not_run",
        "run": False,
        "issue_dates_start": None,
        "issue_dates_end": None,
        "target_count": None,
        "target_ids": [],
        "score_ids": [],
        "artifact_ids": [],
        "result": None,
        "cohort_manifest_path": None,
        "formal_test_execution_forbidden_in_stage4": True,
        "formal_test_belongs_to_stage9": True,
        "outcomes_must_remain_unread": True,
    }
    if dict(locked) != expected_locked:
        raise Stage4QualificationError("locked-test prohibition changed")

    inputs = _mapping(protocol.get("inputs"), label="inputs")
    observed_inputs: dict[str, str] = {}
    for input_id, raw_entry in inputs.items():
        # This exact structural exclusion is the score-blind boundary.  Do not move target
        # verification into a generic input loop: even stat/exists is forbidden here.
        if input_id == "earthquake_target":
            continue
        entry = _mapping(raw_entry, label=f"inputs.{input_id}")
        if entry.get("path_scope") == "external_locationpred_root":
            continue
        if "path" not in entry or "sha256" not in entry:
            raise Stage4QualificationError(f"project input {input_id} lacks path/hash identity")
        path = _project_path(
            project_root,
            protocol,
            entry["path"],
            label=f"inputs.{input_id}.path",
        )
        observed_inputs[input_id] = _verify_hash(
            path,
            entry["sha256"],
            label=f"inputs.{input_id}",
        )

    target_metadata = _mapping(inputs["earthquake_target"], label="earthquake target metadata")
    contract_path = _project_path(
        project_root,
        protocol,
        target_metadata["contract_path"],
        label="inputs.earthquake_target.contract_path",
    )
    observed_inputs["earthquake_target_contract"] = _verify_hash(
        contract_path,
        target_metadata["contract_sha256"],
        label="inputs.earthquake_target.contract",
    )

    generated_config = _mapping(protocol.get("generated_manifests"), label="generated_manifests")
    generated_documents: dict[str, dict[str, object]] = {}
    generated_hashes: dict[str, str] = {}
    for manifest_id, raw_entry in generated_config.items():
        entry = _mapping(raw_entry, label=f"generated_manifests.{manifest_id}")
        path = _project_path(
            project_root,
            protocol,
            entry.get("path"),
            label=f"generated_manifests.{manifest_id}.path",
        )
        manifest_payload, generated_hashes[manifest_id] = _verified_payload(
            path,
            entry.get("sha256"),
            label=f"generated_manifests.{manifest_id}",
        )
        try:
            document = dict(
                _mapping(
                    json.loads(manifest_payload.decode("utf-8")),
                    label=f"generated_manifests.{manifest_id}",
                )
            )
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise Stage4QualificationError(
                f"generated manifest is not valid JSON: {manifest_id}"
            ) from exc
        if not verify_content_sha256(document):
            raise Stage4QualificationError(
                f"generated manifest content hash changed: {manifest_id}"
            )
        generated_documents[manifest_id] = document

    expected_generated = {"feature_set", "fold", "randomness", "spatial_strata"}
    if set(generated_documents) != expected_generated:
        raise Stage4QualificationError("stage-4 generated manifest set changed")
    spatial = generated_documents["spatial_strata"]
    topology_gate = _mapping(spatial.get("topology_gate"), label="spatial.topology_gate")
    input_hash_gate = _mapping(spatial.get("input_hash_gate"), label="spatial.input_hash_gate")
    entity_gate = _mapping(
        spatial.get("entity_stratification_gate"),
        label="spatial.entity_stratification_gate",
    )
    if not all(
        gate.get("passed") is True for gate in (topology_gate, input_hash_gate, entity_gate)
    ):
        raise Stage4QualificationError("frozen construction topology or identity gate failed")
    if spatial.get("spatial_placebo_implementation_authorized") is not True:
        raise Stage4QualificationError("score-blind spatial placebo was not authorized")
    if spatial.get("stage4_target_read_authorized") is not False:
        raise Stage4QualificationError("public spatial manifest improperly authorizes target read")

    local_config = _mapping(
        _mapping(
            protocol.get("spatial_permutation_topology"),
            label="spatial_permutation_topology",
        ).get("local_restricted_artifacts"),
        label="local_restricted_artifacts",
    )
    local_manifest = _mapping(spatial.get("local_artifacts"), label="spatial.local_artifacts")
    config_keys = {
        "cell_mapping": "cell_mapping",
        "connectors": "connectors",
        "entity_mapping": "entity_mapping",
        "zone_geometry": "zone_geometry",
    }
    local_hashes: dict[str, str] = {}
    for artifact_id, config_key in config_keys.items():
        entry = _mapping(local_manifest.get(artifact_id), label=f"local_artifacts.{artifact_id}")
        path = _project_path(
            project_root,
            protocol,
            local_config.get(config_key),
            label=f"local_restricted_artifacts.{config_key}",
        )
        local_hashes[artifact_id] = _verify_hash(
            path,
            entry.get("sha256"),
            label=f"local artifact {artifact_id}",
        )

    validation = validate_stage4_protocol_bundle(
        protocol,
        fold_manifest=generated_documents["fold"],
        feature_manifest=generated_documents["feature_set"],
        randomness_manifest=generated_documents["randomness"],
        spatial_manifest=spatial,
    )
    random_input_seal = build_random_input_seal(
        protocol,
        fold_manifest=generated_documents["fold"],
        feature_manifest=generated_documents["feature_set"],
        spatial_manifest=spatial,
    )
    return ScoreBlindInputEvidence(
        protocol_design_sha256=protocol_design_sha256(protocol),
        random_input_seal_sha256=_sha256(
            random_input_seal.get("content_sha256"),
            label="random input seal",
        ),
        protocol_validation_sha256=_sha256(
            validation.get("content_sha256"),
            label="protocol validation",
        ),
        observed_project_input_hashes=tuple(sorted(observed_inputs.items())),
        generated_manifest_hashes=tuple(sorted(generated_hashes.items())),
        restricted_spatial_artifact_hashes=tuple(sorted(local_hashes.items())),
    )


@dataclass(frozen=True, slots=True)
class PytestRunEvidence:
    """A zero-failure JUnit receipt parsed from an actual pytest XML document."""

    xml_sha256: str
    test_ids: tuple[str, ...]
    test_count: int
    _sentinel: object

    def __post_init__(self) -> None:
        if self._sentinel is not _PYTEST_EVIDENCE_SENTINEL:
            raise ValueError("pytest evidence must be parsed from JUnit XML")
        _sha256(self.xml_sha256, label="pytest XML SHA-256")
        if isinstance(self.test_count, bool) or self.test_count < 1:
            raise ValueError("pytest evidence must contain at least one test")
        if self.test_ids != tuple(sorted(set(self.test_ids))):
            raise ValueError("pytest test IDs must be sorted and unique")
        if len(self.test_ids) != self.test_count:
            raise ValueError("pytest test count differs from its unique test IDs")
        if any("::" not in item or not item.startswith("tests.") for item in self.test_ids):
            raise ValueError("pytest evidence contains an invalid test identity")

    def as_mapping(self) -> dict[str, object]:
        return with_content_sha256(
            {
                "errors": 0,
                "failures": 0,
                "skipped": 0,
                "test_count": self.test_count,
                "test_ids": list(self.test_ids),
                "xml_sha256": self.xml_sha256,
            }
        )

    @property
    def content_sha256(self) -> str:
        return cast(str, self.as_mapping()["content_sha256"])

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> PytestRunEvidence:
        expected = {
            "content_sha256",
            "errors",
            "failures",
            "skipped",
            "test_count",
            "test_ids",
            "xml_sha256",
        }
        if set(value) != expected or not verify_content_sha256(value):
            raise Stage4QualificationError("pytest evidence hash or schema is invalid")
        if any(value.get(name) != 0 for name in ("errors", "failures", "skipped")):
            raise Stage4QualificationError("pytest qualification run was not entirely successful")
        raw_ids = value.get("test_ids")
        if not isinstance(raw_ids, list) or any(not isinstance(item, str) for item in raw_ids):
            raise Stage4QualificationError("pytest evidence test IDs are malformed")
        try:
            return cls(
                xml_sha256=cast(str, value["xml_sha256"]),
                test_ids=tuple(cast(list[str], raw_ids)),
                test_count=cast(int, value["test_count"]),
                _sentinel=_PYTEST_EVIDENCE_SENTINEL,
            )
        except (TypeError, ValueError) as exc:
            raise Stage4QualificationError("pytest evidence invariants failed") from exc


def parse_pytest_junit_evidence(payload: bytes) -> PytestRunEvidence:
    """Parse zero-failure, zero-skip pytest evidence; never accept caller-supplied booleans."""

    if not isinstance(payload, bytes) or not payload:
        raise Stage4QualificationError("pytest JUnit evidence must be non-empty bytes")
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise Stage4QualificationError("pytest JUnit XML is malformed") from exc
    if root.tag not in {"testsuite", "testsuites"}:
        raise Stage4QualificationError("pytest JUnit XML has an unknown root")
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    if not suites:
        raise Stage4QualificationError("pytest JUnit XML contains no test suite")
    test_ids: list[str] = []
    total = failures = errors = skipped = 0
    for suite in suites:
        try:
            total += int(suite.attrib["tests"])
            failures += int(suite.attrib["failures"])
            errors += int(suite.attrib["errors"])
            skipped += int(suite.attrib["skipped"])
        except (KeyError, ValueError) as exc:
            raise Stage4QualificationError("pytest JUnit suite counters are malformed") from exc
        for testcase in suite.findall("testcase"):
            class_name = testcase.attrib.get("classname", "")
            test_name = testcase.attrib.get("name", "")
            if not class_name.startswith("tests.") or not test_name.startswith("test_"):
                raise Stage4QualificationError("pytest JUnit test identity is malformed")
            if list(testcase):
                raise Stage4QualificationError("pytest JUnit contains a non-passing testcase")
            test_ids.append(f"{class_name}::{test_name}")
    if failures or errors or skipped or total != len(test_ids):
        raise Stage4QualificationError(
            "pytest qualification requires every reported test to pass without skips"
        )
    try:
        return PytestRunEvidence(
            xml_sha256=hashlib.sha256(payload).hexdigest(),
            test_ids=tuple(sorted(test_ids)),
            test_count=total,
            _sentinel=_PYTEST_EVIDENCE_SENTINEL,
        )
    except ValueError as exc:
        raise Stage4QualificationError("pytest JUnit evidence is not unique") from exc


def _qualification_check_contract(
    name: str,
) -> tuple[Literal["score_blind_inputs", "stage4_pytest", "full_pytest"], tuple[str, ...]]:
    if name in _SCORE_BLIND_INPUT_QUALIFICATIONS:
        return "score_blind_inputs", ()
    if name == "all_non_target_tests_passed":
        return "full_pytest", ()
    try:
        return "stage4_pytest", tuple(sorted(REQUIRED_TESTS_BY_QUALIFICATION[name]))
    except KeyError as exc:
        raise ValueError("unknown stage-4 qualification check") from exc


@dataclass(frozen=True, slots=True)
class QualificationCheckReceipt:
    """One gate bound to structured source evidence and exact passing tests."""

    name: str
    evidence_kind: Literal["score_blind_inputs", "stage4_pytest", "full_pytest"]
    evidence_sha256: str
    required_test_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.name not in REQUIRED_SCORE_BLIND_QUALIFICATIONS:
            raise ValueError("unknown stage-4 qualification check")
        if self.evidence_kind not in {"score_blind_inputs", "stage4_pytest", "full_pytest"}:
            raise ValueError("unknown stage-4 qualification evidence kind")
        _sha256(self.evidence_sha256, label=f"qualification {self.name} evidence")
        if self.required_test_ids != tuple(sorted(set(self.required_test_ids))):
            raise ValueError("qualification required test IDs must be sorted and unique")
        expected_kind, expected_ids = _qualification_check_contract(self.name)
        if self.evidence_kind != expected_kind or self.required_test_ids != expected_ids:
            raise ValueError("qualification check differs from its frozen evidence contract")

    def as_mapping(self) -> dict[str, object]:
        return {
            "evidence_kind": self.evidence_kind,
            "evidence_sha256": self.evidence_sha256,
            "required_test_ids": list(self.required_test_ids),
        }


GpuQualificationStatus = Literal[
    "not_requested",
    "blocked_no_frozen_backend",
    "not_equivalent",
    "equivalent_and_frozen",
]


def _gpu_evidence_mapping(value: BackendEquivalenceEvidence | None) -> dict[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, BackendEquivalenceEvidence):
        raise TypeError("GPU evidence must be BackendEquivalenceEvidence")
    return {
        "candidate_backend": value.candidate_backend,
        "coefficient_max_abs_error_hex": value.coefficient_max_abs_error.hex(),
        "content_sha256": value.content_sha256(),
        "gradient_max_abs_error_hex": value.gradient_max_abs_error.hex(),
        "integrated_intensity_relative_error_hex": (
            value.integrated_intensity_relative_error.hex()
        ),
        "objective_relative_error_hex": value.objective_relative_error.hex(),
        "random_mapping_byte_identity": value.random_mapping_byte_identity,
        "repeated_run_identity": value.repeated_run_identity,
        "scientific_decision_identity": value.scientific_decision_identity,
        "worker_count_identity": value.worker_count_identity,
    }


def _gpu_evidence_from_mapping(value: object) -> BackendEquivalenceEvidence | None:
    if value is None:
        return None
    raw = _mapping(value, label="GPU equivalence evidence")
    expected = {
        "candidate_backend",
        "coefficient_max_abs_error_hex",
        "content_sha256",
        "gradient_max_abs_error_hex",
        "integrated_intensity_relative_error_hex",
        "objective_relative_error_hex",
        "random_mapping_byte_identity",
        "repeated_run_identity",
        "scientific_decision_identity",
        "worker_count_identity",
    }
    if set(raw) != expected or raw.get("candidate_backend") != "gpu_float64":
        raise Stage4QualificationError("GPU equivalence evidence schema changed")
    try:
        evidence = BackendEquivalenceEvidence(
            candidate_backend="gpu_float64",
            objective_relative_error=float.fromhex(cast(str, raw["objective_relative_error_hex"])),
            gradient_max_abs_error=float.fromhex(cast(str, raw["gradient_max_abs_error_hex"])),
            coefficient_max_abs_error=float.fromhex(
                cast(str, raw["coefficient_max_abs_error_hex"])
            ),
            integrated_intensity_relative_error=float.fromhex(
                cast(str, raw["integrated_intensity_relative_error_hex"])
            ),
            random_mapping_byte_identity=cast(bool, raw["random_mapping_byte_identity"]),
            scientific_decision_identity=cast(bool, raw["scientific_decision_identity"]),
            repeated_run_identity=cast(bool, raw["repeated_run_identity"]),
            worker_count_identity=cast(bool, raw["worker_count_identity"]),
        )
    except (TypeError, ValueError) as exc:
        raise Stage4QualificationError("GPU equivalence evidence values are malformed") from exc
    if (
        any(
            not isinstance(raw[name], bool)
            for name in (
                "random_mapping_byte_identity",
                "scientific_decision_identity",
                "repeated_run_identity",
                "worker_count_identity",
            )
        )
        or raw.get("content_sha256") != evidence.content_sha256()
    ):
        raise Stage4QualificationError("GPU equivalence evidence content hash changed")
    return evidence


def _gpu_passes_protocol(
    protocol: Mapping[str, object],
    evidence: BackendEquivalenceEvidence | None,
) -> bool:
    if evidence is None:
        return False
    compute = _mapping(protocol.get("compute"), label="compute")
    policy = _mapping(compute.get("gpu_equivalence"), label="compute.gpu_equivalence")

    def threshold(name: str) -> float:
        value = policy.get(name)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise Stage4QualificationError(f"GPU threshold {name} is not numeric")
        result = float(value)
        if not math.isfinite(result) or result < 0.0:
            raise Stage4QualificationError(f"GPU threshold {name} is invalid")
        return result

    return (
        evidence.objective_relative_error <= threshold("objective_relative_tolerance")
        and evidence.gradient_max_abs_error <= threshold("gradient_max_abs_tolerance")
        and evidence.coefficient_max_abs_error <= threshold("coefficient_max_abs_tolerance")
        and evidence.integrated_intensity_relative_error
        <= threshold("integrated_intensity_relative_tolerance")
        and evidence.random_mapping_byte_identity
        and evidence.scientific_decision_identity
        and evidence.repeated_run_identity
        and evidence.worker_count_identity
    )


def _gpu_status(
    protocol: Mapping[str, object],
    *,
    requested: bool,
    evidence: BackendEquivalenceEvidence | None,
) -> GpuQualificationStatus:
    freeze = _mapping(protocol.get("freeze"), label="freeze")
    scoring = _mapping(freeze.get("scoring_code_freeze"), label="scoring_code_freeze")
    if scoring.get("gpu_if_not_equivalent_at_code_freeze") != ("lock_formal_run_to_cpu_float64"):
        raise Stage4QualificationError("stage-4 CPU fallback policy changed")
    compute = _mapping(protocol.get("compute"), label="compute")
    reference = _mapping(
        compute.get("gpu_available_reference"),
        label="compute.gpu_available_reference",
    )
    if not requested:
        return "not_requested"
    if reference.get("python_gpu_backend_installed_at_freeze") is not True:
        return "blocked_no_frozen_backend"
    return "equivalent_and_frozen" if _gpu_passes_protocol(protocol, evidence) else "not_equivalent"


@dataclass(frozen=True, slots=True)
class Stage4QualificationEvidence:
    """Receipt derived from parsed tests, fresh inputs, and typed GPU evidence."""

    protocol_design_sha256: str
    scoring_code_commit: str
    checks: tuple[QualificationCheckReceipt, ...]
    score_blind_input_evidence_sha256: str
    formal_preflight_receipt_sha256: str
    logical_identity_replay_audit_sha256: str
    stage4_pytest: PytestRunEvidence
    full_pytest: PytestRunEvidence
    gpu_requested: bool
    gpu_status: GpuQualificationStatus
    space_placebo_resource_observation_sha256: str = field(compare=False)
    gpu_equivalence: BackendEquivalenceEvidence | None = None

    def __post_init__(self) -> None:
        _sha256(self.protocol_design_sha256, label="protocol_design_sha256")
        _git_oid(self.scoring_code_commit, label="scoring_code_commit")
        _sha256(
            self.score_blind_input_evidence_sha256,
            label="score_blind_input_evidence_sha256",
        )
        _sha256(
            self.formal_preflight_receipt_sha256,
            label="formal_preflight_receipt_sha256",
        )
        _sha256(
            self.logical_identity_replay_audit_sha256,
            label="logical_identity_replay_audit_sha256",
        )
        _sha256(
            self.space_placebo_resource_observation_sha256,
            label="space_placebo_resource_observation_sha256",
        )
        if tuple(item.name for item in self.checks) != REQUIRED_SCORE_BLIND_QUALIFICATIONS:
            raise ValueError("qualification checks differ from the frozen gate order")
        if self.full_pytest.test_count != FROZEN_FULL_NON_TARGET_TEST_COUNT:
            raise ValueError("full non-target regression count differs from the scoring freeze")
        if self.gpu_status not in {
            "not_requested",
            "blocked_no_frozen_backend",
            "not_equivalent",
            "equivalent_and_frozen",
        }:
            raise ValueError("unknown GPU qualification status")
        if self.gpu_status == "equivalent_and_frozen" and self.gpu_equivalence is None:
            raise ValueError("accepted GPU backend requires typed equivalence evidence")

    @property
    def formal_backend(self) -> str:
        return "gpu_float64" if self.gpu_status == "equivalent_and_frozen" else "cpu_float64"

    def as_mapping(self) -> dict[str, object]:
        deterministic = with_content_sha256(
            {
                "checks": {item.name: item.as_mapping() for item in self.checks},
                "formal_attempt_count": 0,
                "formal_preflight_receipt_sha256": (self.formal_preflight_receipt_sha256),
                "full_pytest": self.full_pytest.as_mapping(),
                "gpu": {
                    "equivalence": _gpu_evidence_mapping(self.gpu_equivalence),
                    "formal_backend": self.formal_backend,
                    "requested": self.gpu_requested,
                    "status": self.gpu_status,
                },
                "locked_test_contacted": False,
                "logical_identity_replay_audit_sha256": (self.logical_identity_replay_audit_sha256),
                "protocol_design_sha256": self.protocol_design_sha256,
                "protocol_version": STAGE4_PROTOCOL_VERSION,
                "schema_version": STAGE4_QUALIFICATION_SCHEMA_VERSION,
                "score_blind_input_evidence_sha256": self.score_blind_input_evidence_sha256,
                "scoring_code_commit": self.scoring_code_commit,
                "stage4_pytest": self.stage4_pytest.as_mapping(),
                "target_read_count": 0,
            }
        )
        # Resource telemetry is authenticated and carried by the qualification,
        # but deliberately excluded from its deterministic scientific identity.
        deterministic["space_placebo_resource_observation_sha256"] = (
            self.space_placebo_resource_observation_sha256
        )
        return deterministic

    @property
    def content_sha256(self) -> str:
        return cast(str, self.as_mapping()["content_sha256"])

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Stage4QualificationEvidence:
        expected = {
            "checks",
            "content_sha256",
            "formal_attempt_count",
            "formal_preflight_receipt_sha256",
            "full_pytest",
            "gpu",
            "locked_test_contacted",
            "logical_identity_replay_audit_sha256",
            "protocol_design_sha256",
            "protocol_version",
            "schema_version",
            "score_blind_input_evidence_sha256",
            "scoring_code_commit",
            "space_placebo_resource_observation_sha256",
            "stage4_pytest",
            "target_read_count",
        }
        deterministic = dict(value)
        deterministic.pop("space_placebo_resource_observation_sha256", None)
        if set(value) != expected or not verify_content_sha256(deterministic):
            raise Stage4QualificationError("qualification evidence hash or schema is invalid")
        if (
            value.get("schema_version") != STAGE4_QUALIFICATION_SCHEMA_VERSION
            or value.get("protocol_version") != STAGE4_PROTOCOL_VERSION
            or value.get("formal_attempt_count") != 0
            or value.get("target_read_count") != 0
            or value.get("locked_test_contacted") is not False
        ):
            raise Stage4QualificationError("qualification evidence is not target-blind")
        stage4_pytest = PytestRunEvidence.from_mapping(
            _mapping(value.get("stage4_pytest"), label="stage4_pytest")
        )
        full_pytest = PytestRunEvidence.from_mapping(
            _mapping(value.get("full_pytest"), label="full_pytest")
        )
        checks = _mapping(value.get("checks"), label="qualification checks")
        if set(checks) != set(REQUIRED_SCORE_BLIND_QUALIFICATIONS):
            raise Stage4QualificationError("qualification evidence check set changed")
        receipts: list[QualificationCheckReceipt] = []
        for name in REQUIRED_SCORE_BLIND_QUALIFICATIONS:
            entry = _mapping(checks.get(name), label=f"qualification checks.{name}")
            if set(entry) != {"evidence_kind", "evidence_sha256", "required_test_ids"}:
                raise Stage4QualificationError(f"qualification check schema changed: {name}")
            raw_ids = entry.get("required_test_ids")
            if not isinstance(raw_ids, list) or any(not isinstance(item, str) for item in raw_ids):
                raise Stage4QualificationError(f"qualification test IDs changed: {name}")
            try:
                receipts.append(
                    QualificationCheckReceipt(
                        name=name,
                        evidence_kind=cast(Any, entry["evidence_kind"]),
                        evidence_sha256=cast(str, entry["evidence_sha256"]),
                        required_test_ids=tuple(cast(list[str], raw_ids)),
                    )
                )
            except (TypeError, ValueError) as exc:
                raise Stage4QualificationError(f"qualification check invalid: {name}") from exc
        gpu = _mapping(value.get("gpu"), label="qualification gpu")
        if set(gpu) != {"equivalence", "formal_backend", "requested", "status"} or not isinstance(
            gpu.get("requested"), bool
        ):
            raise Stage4QualificationError("GPU qualification fields changed")
        evidence = _gpu_evidence_from_mapping(gpu.get("equivalence"))
        try:
            result = cls(
                protocol_design_sha256=cast(str, value["protocol_design_sha256"]),
                scoring_code_commit=cast(str, value["scoring_code_commit"]),
                checks=tuple(receipts),
                score_blind_input_evidence_sha256=cast(
                    str, value["score_blind_input_evidence_sha256"]
                ),
                formal_preflight_receipt_sha256=cast(str, value["formal_preflight_receipt_sha256"]),
                logical_identity_replay_audit_sha256=cast(
                    str, value["logical_identity_replay_audit_sha256"]
                ),
                stage4_pytest=stage4_pytest,
                full_pytest=full_pytest,
                gpu_requested=cast(bool, gpu["requested"]),
                gpu_status=cast(GpuQualificationStatus, gpu["status"]),
                gpu_equivalence=evidence,
                space_placebo_resource_observation_sha256=cast(
                    str, value["space_placebo_resource_observation_sha256"]
                ),
            )
        except (TypeError, ValueError) as exc:
            raise Stage4QualificationError("qualification evidence invariants failed") from exc
        if gpu.get("formal_backend") != result.formal_backend:
            raise Stage4QualificationError("formal backend is inconsistent with GPU evidence")
        if result.as_mapping() != dict(value):
            raise Stage4QualificationError("qualification evidence did not canonicalize")
        return result


def validate_stage4_qualification_against_protocol(
    protocol: Mapping[str, object],
    evidence: Stage4QualificationEvidence,
) -> None:
    """Recompute all bindings from the current protocol and typed evidence."""

    freeze = _mapping(protocol.get("freeze"), label="freeze")
    scoring = _mapping(freeze.get("scoring_code_freeze"), label="scoring_code_freeze")

    def gate_names(value: object, *, label: str) -> set[str]:
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise Stage4QualificationError(f"{label} must be a string list")
        if len(value) != len(set(value)):
            raise Stage4QualificationError(f"{label} contains duplicate gates")
        return set(value)

    declared_gates = gate_names(
        freeze.get("required_before_any_target_read_or_score"),
        label="required_before_any_target_read_or_score",
    ) | gate_names(
        scoring.get("required_before_target_read"),
        label="scoring_code_freeze.required_before_target_read",
    )
    expected_gates = set(REQUIRED_SCORE_BLIND_QUALIFICATIONS) | set(_REPOSITORY_AND_LEDGER_GATES)
    if declared_gates != expected_gates:
        raise Stage4QualificationError(
            "structured qualification checks differ from the protocol gate set"
        )
    if evidence.protocol_design_sha256 != protocol_design_sha256(protocol):
        raise Stage4QualificationError("qualification uses another protocol design")
    expected_status = _gpu_status(
        protocol,
        requested=evidence.gpu_requested,
        evidence=evidence.gpu_equivalence,
    )
    if evidence.gpu_status != expected_status:
        raise Stage4QualificationError("GPU qualification status does not match the protocol")
    stage4_ids = set(evidence.stage4_pytest.test_ids)
    full_ids = set(evidence.full_pytest.test_ids)
    if not stage4_ids <= full_ids:
        raise Stage4QualificationError("full regression evidence omits stage-4 tests")
    for receipt in evidence.checks:
        if receipt.evidence_kind == "score_blind_inputs":
            if receipt.evidence_sha256 != evidence.score_blind_input_evidence_sha256:
                raise Stage4QualificationError("score-blind check uses another input receipt")
            continue
        run = (
            evidence.full_pytest
            if receipt.evidence_kind == "full_pytest"
            else evidence.stage4_pytest
        )
        if receipt.evidence_sha256 != run.content_sha256:
            raise Stage4QualificationError("qualification check uses another pytest run")
        if not set(receipt.required_test_ids) <= set(run.test_ids):
            raise Stage4QualificationError(
                f"qualification evidence omits required tests: {receipt.name}"
            )


def validate_stage4_qualification_against_formal_preflight(
    evidence: Stage4QualificationEvidence,
    receipt: FormalPreflightReceipt,
) -> None:
    """Bind deterministic preflight science and separate resource telemetry."""

    from seismoflux.anomaly_increment.formal_preflight import FormalPreflightReceipt

    if not isinstance(receipt, FormalPreflightReceipt):
        raise TypeError("receipt must be FormalPreflightReceipt")
    if evidence.formal_preflight_receipt_sha256 != receipt.content_sha256:
        raise Stage4QualificationError("qualification uses another formal preflight receipt")
    if evidence.space_placebo_resource_observation_sha256 != (
        receipt.space_placebo_resource_observation.content_sha256
    ):
        raise Stage4QualificationError("qualification uses another space resource observation")
    if evidence.protocol_design_sha256 != receipt.protocol_design_sha256:
        raise Stage4QualificationError("qualification and formal preflight protocols differ")
    if evidence.scoring_code_commit != receipt.scoring_code_commit:
        raise Stage4QualificationError("qualification and formal preflight commits differ")
    if evidence.score_blind_input_evidence_sha256 != (receipt.score_blind_input_evidence_sha256):
        raise Stage4QualificationError(
            "qualification and formal preflight score-blind receipts differ"
        )


def build_stage4_qualification_evidence(
    protocol: Mapping[str, object],
    *,
    scoring_code_commit: str,
    score_blind_input_evidence: ScoreBlindInputEvidence,
    formal_preflight_receipt: FormalPreflightReceipt,
    logical_identity_replay_audit_sha256: str,
    stage4_pytest: PytestRunEvidence,
    full_pytest: PytestRunEvidence,
    gpu_requested: bool = False,
    gpu_equivalence: BackendEquivalenceEvidence | None = None,
) -> Stage4QualificationEvidence:
    """Derive every check from parsed JUnit, fresh inputs, and typed backend evidence."""

    if not isinstance(score_blind_input_evidence, ScoreBlindInputEvidence):
        raise TypeError("score_blind_input_evidence must be ScoreBlindInputEvidence")
    # Local import avoids a module cycle: formal_preflight itself depends on the
    # score-blind evidence type above.
    from seismoflux.anomaly_increment.formal_preflight import FormalPreflightReceipt

    if not isinstance(formal_preflight_receipt, FormalPreflightReceipt):
        raise TypeError("formal_preflight_receipt must be FormalPreflightReceipt")
    _sha256(
        logical_identity_replay_audit_sha256,
        label="logical_identity_replay_audit_sha256",
    )
    if not isinstance(stage4_pytest, PytestRunEvidence) or not isinstance(
        full_pytest, PytestRunEvidence
    ):
        raise TypeError("qualification tests must be parsed PytestRunEvidence")
    design_sha256 = protocol_design_sha256(protocol)
    if design_sha256 != score_blind_input_evidence.protocol_design_sha256:
        raise Stage4QualificationError("qualification and score-blind inputs use another protocol")
    if formal_preflight_receipt.protocol_design_sha256 != design_sha256:
        raise Stage4QualificationError("formal preflight belongs to another protocol")
    if formal_preflight_receipt.score_blind_input_evidence_sha256 != (
        score_blind_input_evidence.content_sha256
    ):
        raise Stage4QualificationError("formal preflight uses another score-blind input receipt")
    if formal_preflight_receipt.scoring_code_commit != scoring_code_commit:
        raise Stage4QualificationError("formal preflight belongs to another scoring commit")
    if formal_preflight_receipt.random_input_seal_sha256 != (
        score_blind_input_evidence.random_input_seal_sha256
    ):
        raise Stage4QualificationError("formal preflight uses another random input seal")
    receipts: list[QualificationCheckReceipt] = []
    for name in REQUIRED_SCORE_BLIND_QUALIFICATIONS:
        if name in _SCORE_BLIND_INPUT_QUALIFICATIONS:
            receipts.append(
                QualificationCheckReceipt(
                    name=name,
                    evidence_kind="score_blind_inputs",
                    evidence_sha256=score_blind_input_evidence.content_sha256,
                )
            )
            continue
        if name == "all_non_target_tests_passed":
            receipts.append(
                QualificationCheckReceipt(
                    name=name,
                    evidence_kind="full_pytest",
                    evidence_sha256=full_pytest.content_sha256,
                )
            )
            continue
        required = tuple(sorted(REQUIRED_TESTS_BY_QUALIFICATION[name]))
        receipts.append(
            QualificationCheckReceipt(
                name=name,
                evidence_kind="stage4_pytest",
                evidence_sha256=stage4_pytest.content_sha256,
                required_test_ids=required,
            )
        )
    result = Stage4QualificationEvidence(
        protocol_design_sha256=design_sha256,
        scoring_code_commit=scoring_code_commit,
        checks=tuple(receipts),
        score_blind_input_evidence_sha256=score_blind_input_evidence.content_sha256,
        formal_preflight_receipt_sha256=formal_preflight_receipt.content_sha256,
        logical_identity_replay_audit_sha256=logical_identity_replay_audit_sha256,
        stage4_pytest=stage4_pytest,
        full_pytest=full_pytest,
        gpu_requested=gpu_requested,
        gpu_status=_gpu_status(
            protocol,
            requested=gpu_requested,
            evidence=gpu_equivalence,
        ),
        gpu_equivalence=gpu_equivalence,
        space_placebo_resource_observation_sha256=(
            formal_preflight_receipt.space_placebo_resource_observation.content_sha256
        ),
    )
    validate_stage4_qualification_against_protocol(protocol, result)
    validate_stage4_qualification_against_formal_preflight(
        result,
        formal_preflight_receipt,
    )
    return result


def load_stage4_qualification_evidence(path: Path) -> Stage4QualificationEvidence:
    target = Path(os.path.abspath(os.fspath(path)))
    try:
        require_existing_real_directory_tree(
            Path(target.anchor) if target.anchor else Path.cwd(),
            target.parent,
            label="stage-4 qualification evidence directory",
        )
        payload = read_existing_immutable_bytes(
            target,
            label="stage-4 qualification evidence",
        )
        value = json.loads(payload.decode("utf-8"))
    except (UnsafeImmutableFileError, UnicodeError, json.JSONDecodeError) as exc:
        raise Stage4QualificationError("cannot read stage-4 qualification evidence") from exc
    return Stage4QualificationEvidence.from_mapping(_mapping(value, label="qualification evidence"))


def write_stage4_qualification_evidence_atomic(
    path: Path,
    evidence: Stage4QualificationEvidence,
) -> str:
    """Create qualification evidence once; permit only byte-identical replay."""

    if not isinstance(evidence, Stage4QualificationEvidence):
        raise TypeError("evidence must be Stage4QualificationEvidence")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    serialized = canonical_json_bytes(evidence.as_mapping()) + b"\n"
    serialized_sha256 = hashlib.sha256(serialized).hexdigest()
    temporary_name: str | None = None
    created = False
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        staged = Path(temporary_name)
        try:
            os.link(staged, target)
            created = True
        except FileExistsError:
            try:
                existing = read_existing_immutable_bytes(
                    target,
                    label="existing stage-4 qualification evidence",
                )
            except UnsafeImmutableFileError:
                existing = None
            if existing == serialized:
                return serialized_sha256
            raise Stage4QualificationError(
                "immutable stage-4 qualification already contains different bytes "
                "or is not a safe single-link regular file"
            ) from None
        finally:
            staged.unlink(missing_ok=True)
            temporary_name = None
    except Exception:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise
    if not created:
        raise AssertionError("stage-4 qualification publication reached no terminal state")
    try:
        if (
            read_existing_immutable_bytes(
                target,
                label="new stage-4 qualification evidence",
            )
            != serialized
        ):
            raise UnsafeImmutableFileError("new stage-4 qualification bytes changed")
    except UnsafeImmutableFileError as exc:
        raise Stage4QualificationError(
            "new stage-4 qualification is not a safe single-link regular file"
        ) from exc
    return serialized_sha256


__all__ = [
    "FROZEN_FULL_NON_TARGET_TEST_COUNT",
    "REQUIRED_SCORE_BLIND_QUALIFICATIONS",
    "REQUIRED_TESTS_BY_QUALIFICATION",
    "STAGE4_PROTOCOL_VERSION",
    "STAGE4_QUALIFICATION_SCHEMA_VERSION",
    "PytestRunEvidence",
    "QualificationCheckReceipt",
    "ScoreBlindInputEvidence",
    "Stage4QualificationError",
    "Stage4QualificationEvidence",
    "build_stage4_qualification_evidence",
    "expected_target_identity_from_protocol",
    "load_stage4_qualification_evidence",
    "observe_score_blind_inputs",
    "parse_pytest_junit_evidence",
    "validate_stage4_qualification_against_formal_preflight",
    "validate_stage4_qualification_against_protocol",
    "write_stage4_qualification_evidence_atomic",
]
