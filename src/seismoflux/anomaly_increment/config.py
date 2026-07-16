"""Strict, target-blind loading of the frozen stage-4 protocol bundle.

This module deliberately has no target-catalog loader.  The earthquake target is
represented only by the identity strings that were frozen in the protocol.  In
particular, loading or validating a :class:`Stage4ProtocolBundle` must not call
``exists()``, ``stat()``, ``open()`` or a hashing function on the target path.
The only target byte access lives in :mod:`seismoflux.anomaly_increment.target_access`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Final, cast

import yaml

from seismoflux.anomaly_increment.immutable_file import (
    read_existing_immutable_bytes,
    require_existing_real_directory_tree,
    sha256_existing_immutable_file,
)
from seismoflux.anomaly_increment.preregistration import (
    protocol_design_sha256,
    validate_stage4_protocol_bundle,
)

STAGE4_EXECUTION_REVISION: Final = "r2"
STAGE4_PROTOCOL_PATH: Final = Path("configs/anomaly_increment_r2.yaml")
STAGE4_PROTOCOL_TAG: Final = "v0.3.1-anomaly-increment-protocol-r2"
STAGE4_SCORING_CODE_TAG: Final = "v0.3.1-anomaly-increment-scoring-code-r2"
STAGE4_RESULT_TAG: Final = "v0.3.1-anomaly-increment-r2"
STAGE4_SCORING_SEAL_RELATIVE_PATH: Final[PurePosixPath] = PurePosixPath(
    "data/manifests/anomaly_increment_r2_scoring_seal.json"
)
STAGE4_FORMAL_PREFLIGHT_RECEIPT_RELATIVE_PATH: Final[PurePosixPath] = PurePosixPath(
    "data/interim/stage4/anomaly_increment_r2/formal_preflight_receipt.json"
)
STAGE4_QUALIFICATION_RELATIVE_PATH: Final[PurePosixPath] = PurePosixPath(
    "data/interim/stage4/anomaly_increment_r2/scoring_qualification.json"
)
STAGE4_LOGICAL_REPLAY_AUDIT_RELATIVE_PATH: Final[PurePosixPath] = PurePosixPath(
    "data/interim/stage4/anomaly_increment_r2/logical_identity_worker_replay.json"
)
STAGE4_JUNIT_RELATIVE_PATH: Final[PurePosixPath] = PurePosixPath(
    "data/interim/stage4/anomaly_increment_r2/qualification_stage4.junit.xml"
)
STAGE4_FULL_NON_TARGET_JUNIT_RELATIVE_PATH: Final[PurePosixPath] = PurePosixPath(
    "data/interim/stage4/anomaly_increment_r2/qualification_full_non_target.junit.xml"
)
STAGE4_ATTEMPT_LEDGER_RELATIVE_PATH: Final[PurePosixPath] = PurePosixPath(
    "data/manifests/anomaly_increment_r2_attempt_ledger.json"
)
STAGE4_TARGET_READ_LEDGER_RELATIVE_PATH: Final[PurePosixPath] = PurePosixPath(
    "data/manifests/anomaly_increment_r2_target_read_ledger.json"
)
STAGE4_CHECKPOINT_ROOT_RELATIVE_PATH: Final[PurePosixPath] = PurePosixPath(
    "data/interim/stage4/anomaly_increment_r2/checkpoints"
)

_STAGE4_SCORING_FREEZE_PATHS: Final[tuple[tuple[str, PurePosixPath], ...]] = (
    ("required_seal_path", STAGE4_SCORING_SEAL_RELATIVE_PATH),
    ("formal_preflight_receipt_path", STAGE4_FORMAL_PREFLIGHT_RECEIPT_RELATIVE_PATH),
    ("qualification_path", STAGE4_QUALIFICATION_RELATIVE_PATH),
    ("stage4_junit_path", STAGE4_JUNIT_RELATIVE_PATH),
    ("full_non_target_junit_path", STAGE4_FULL_NON_TARGET_JUNIT_RELATIVE_PATH),
    ("formal_attempt_ledger_path", STAGE4_ATTEMPT_LEDGER_RELATIVE_PATH),
    ("target_read_ledger_path", STAGE4_TARGET_READ_LEDGER_RELATIVE_PATH),
    ("checkpoint_root", STAGE4_CHECKPOINT_ROOT_RELATIVE_PATH),
)

_STAGE4_LOGICAL_IDENTITY_CONTRACT: Final[dict[str, object]] = {
    "method_id": "arrow_ipc_selected_table_logical_identity_r1",
    "sha256_domain_separator_ascii": "seismoflux.selected-table-logical-identity.r1",
    "sha256_domain_separator_nul_terminated": True,
    "top_level_schema_metadata": "excluded",
    "field_name_order_type_nullability_and_metadata": "preserved_exactly",
    "null_payload": "canonical_type_zero",
    "validity_bitmap": "preserved_with_length_padding_zeroed",
    "boolean_value_padding": "zeroed_outside_logical_length",
    "chunking_and_slice_offsets": "canonicalized",
    "field_metadata_key_order": "bytewise_ascending",
    "supported_types": [
        "boolean",
        "signed_integer",
        "unsigned_integer",
        "floating_point",
        "timestamp",
        "utf8_string",
    ],
    "valid_payload_bits": "preserved_exactly",
    "unsupported_types": "fail_closed",
}

JsonObject = dict[str, object]


def _mapping(value: object, *, label: str) -> JsonObject:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be a string-keyed mapping")
    return cast(JsonObject, value)


def _string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{label} must be a non-empty string")
    return value


def _sha256(value: object, *, label: str) -> str:
    text = _string(value, label=label)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return text


def _read_yaml_mapping(path: Path) -> JsonObject:
    payload = read_existing_immutable_bytes(path, label=str(path))
    return _mapping(yaml.safe_load(payload.decode("utf-8")), label=str(path))


def _read_json_mapping(path: Path) -> JsonObject:
    payload = read_existing_immutable_bytes(path, label=str(path))
    return _mapping(json.loads(payload.decode("utf-8")), label=str(path))


def _file_sha256(path: Path) -> str:
    return sha256_existing_immutable_file(path, label=str(path))


def validate_stage4_r2_execution_contract(protocol: Mapping[str, object]) -> None:
    """Require the exact R2 execution namespace before any scoring authorization."""

    if protocol.get("protocol_version") != "0.4.1":
        raise ValueError("stage-4 scientific protocol_version must remain 0.4.1")
    freeze = _mapping(protocol.get("freeze"), label="freeze")
    if freeze.get("execution_revision") != STAGE4_EXECUTION_REVISION:
        raise ValueError("stage-4 execution revision must be r2")
    if freeze.get("corrects_execution_revision") != "r1":
        raise ValueError("stage-4 R2 must identify the corrected r1 execution")
    if freeze.get("execution_revision_document") != "docs/anomaly_increment_protocol_r2.md":
        raise ValueError("stage-4 R2 execution-revision document path changed")
    if freeze.get("readiness_incident_document") != (
        "docs/phase4_scoring_readiness_incident_r0.md"
    ):
        raise ValueError("stage-4 R2 readiness-incident document path changed")
    if freeze.get("pre_score_tag") != STAGE4_PROTOCOL_TAG:
        raise ValueError("stage-4 R2 protocol tag changed")
    if freeze.get("results_tag") != STAGE4_RESULT_TAG:
        raise ValueError("stage-4 R2 results tag changed")

    scoring = _mapping(
        freeze.get("scoring_code_freeze"),
        label="freeze.scoring_code_freeze",
    )
    if scoring.get("expected_tag") != STAGE4_SCORING_CODE_TAG:
        raise ValueError("stage-4 R2 scoring-code tag changed")
    for key, expected in _STAGE4_SCORING_FREEZE_PATHS:
        if scoring.get(key) != expected.as_posix():
            raise ValueError(f"stage-4 R2 scoring freeze path changed: {key}")
    logical_identity = _mapping(
        scoring.get("selected_table_logical_identity"),
        label="freeze.scoring_code_freeze.selected_table_logical_identity",
    )
    if logical_identity != _STAGE4_LOGICAL_IDENTITY_CONTRACT:
        raise ValueError("stage-4 R2 inherited logical identity contract changed")

    retirement = _mapping(
        freeze.get("r1_retirement"),
        label="freeze.r1_retirement",
    )
    if retirement.get("protocol_design_sha256") != (
        "c15d3bbca5cef4b363a79e183d715124256a12088873d81cd77de489766b32de"
    ):
        raise ValueError("stage-4 R1 retirement protocol identity changed")
    if retirement.get("scoring_seal_file_sha256") != (
        "a6e8dc9ac283813edb62e301114d4985ae332b9c607584c987a4297efe5978f3"
    ):
        raise ValueError("stage-4 R1 retirement seal identity changed")
    for ledger_name, expected_file, expected_content in (
        (
            "formal_attempt_ledger",
            "9ac5e5e080c1d5425f985cb3091b94c0da69d211469589d26ae8bfc314088142",
            "cadc80e5a0f00ffce241f910409750b01e3f410d910dda5d5aad0ff3033d2448",
        ),
        (
            "target_read_ledger",
            "0a49450cc1006ccd0ced26fba30330417f1ec8667c5cedf0bf04242f158210c8",
            "4c1fb843edfa8f59f37f137d8d68962cbdae5991cc8809302d39d240f0b395b6",
        ),
    ):
        ledger = _mapping(
            retirement.get(ledger_name),
            label=f"freeze.r1_retirement.{ledger_name}",
        )
        if (
            ledger.get("file_sha256") != expected_file
            or ledger.get("content_sha256") != expected_content
            or ledger.get("operation_count") != 0
        ):
            raise ValueError(f"stage-4 R1 retirement ledger changed: {ledger_name}")
    if retirement.get("target_bytes_observed") is not False:
        raise ValueError("stage-4 R1 retirement target-read status changed")
    if retirement.get("reusable_for_r2_authorization") is not False:
        raise ValueError("stage-4 R1 artifacts may not authorize R2")

    inputs = _mapping(protocol.get("inputs"), label="inputs")
    target = _mapping(inputs.get("earthquake_target"), label="inputs.earthquake_target")
    coverage = _mapping(
        target.get("frozen_catalog_coverage"),
        label="inputs.earthquake_target.frozen_catalog_coverage",
    )
    expected_coverage = {
        "basis_document": "docs/data_quality_report.md",
        "basis_document_sha256": (
            "f4bf6633ce433b2c8d85d9d6d36cecd4c6824889f80e4528eeaae4de055ee9de"
        ),
        "observed_origin_time_max_utc": "2026-07-09T04:25:56Z",
        "observed_available_at_max_utc": "2026-07-09T04:25:56Z",
        "frozen_validation_window_end_max_utc": "2025-07-18T16:00:00Z",
        "all_frozen_validation_window_endpoints_must_be_lte_both_catalog_maxima": True,
        "verify_after_authorized_target_open_before_first_score": True,
        "missing_or_short_coverage_action": (
            "fail_closed_register_invalid_attempt_and_do_not_score"
        ),
        "available_at_equals_origin_time_is_optimistic_timeliness_assumption": True,
    }
    if coverage != expected_coverage:
        raise ValueError("stage-4 R2 earthquake-catalog coverage contract changed")

    evaluation = _mapping(protocol.get("evaluation"), label="evaluation")
    permutations = _mapping(evaluation.get("permutations"), label="evaluation.permutations")
    expected_requests = [
        {"kind": "time", "model_variant": "dynamic"},
        {"kind": "space", "model_variant": "dynamic"},
        {"kind": "time", "model_variant": "snapshot"},
        {"kind": "space", "model_variant": "snapshot"},
    ]
    if permutations.get("formal_requests") != expected_requests:
        raise ValueError("stage-4 R2 requires exactly four formal placebo requests")
    if permutations.get("formal_checkpoint_request_identities") != [
        "time-dynamic",
        "space-dynamic",
        "time-snapshot",
        "space-snapshot",
    ]:
        raise ValueError("stage-4 R2 formal checkpoint identities changed")
    if (
        permutations.get("checkpoint_identity_pattern") != "kind-model_variant"
        or permutations.get("exact_request_set_required") is not True
        or permutations.get("mappings_paired_across_dynamic_and_snapshot") is not True
    ):
        raise ValueError("stage-4 R2 paired placebo request contract changed")
    gates = _mapping(evaluation.get("gates"), label="evaluation.gates")
    g2 = _mapping(gates.get("G2"), label="evaluation.gates.G2")
    if (
        g2.get("candidate_variant") != "dynamic"
        or g2.get("snapshot_equivalent_candidate_variant") != "snapshot"
        or g2.get("evaluated_model_variants") != ["dynamic", "snapshot"]
        or g2.get("required_primary_placebos_by_variant")
        != {"dynamic": ["time", "space"], "snapshot": ["time", "space"]}
        or g2.get("primary_time_permutation_p_lte") != 0.05
        or g2.get("primary_space_permutation_p_lte") != 0.05
        or g2.get("both_primary_p_values_required_for_each_evaluated_variant") is not True
        or g2.get("same_practical_improvement_thresholds_apply_per_evaluated_variant") is not True
        or g2.get("reporting_confound_guard_each_evaluated_candidate_gt_coverage_only") is not True
        or g2.get("reporting_confound_guard_applies_independently_to_variants")
        != ["dynamic", "snapshot"]
        or g2.get("candidate_minus_coverage_only_macro_information_gain_lower_95pct_bound_gt") != 0
    ):
        raise ValueError("stage-4 R2 dynamic/snapshot G2 contract changed")
    practical = g2.get("practical_improvement_any_of")
    if not isinstance(practical, list) or len(practical) != 2:
        raise ValueError("stage-4 R2 practical-improvement branches changed")
    for index, branch_value in enumerate(practical):
        branch = _mapping(
            branch_value,
            label=f"evaluation.gates.G2.practical_improvement_any_of[{index}]",
        )
        if branch.get("candidate_variant") != "current_evaluated_candidate_variant":
            raise ValueError(
                "stage-4 R2 practical thresholds must apply to each evaluated candidate"
            )

    compute = _mapping(protocol.get("compute"), label="compute")
    if (
        compute.get("max_workers") != 6
        or compute.get("logical_cpu_affinity_limit") != 6
        or compute.get("process_priority") != "below_normal"
        or compute.get("nested_parallelism") is not False
        or compute.get("blas_threads_per_worker") != 1
        or compute.get("blas_environment_must_be_set_before_numpy_or_scipy_import") is not True
        or compute.get("resource_control_receipt_required_before_target_read") is not True
    ):
        raise ValueError("stage-4 R2 resource-control contract changed")

    publication = _mapping(protocol.get("publication"), label="publication")
    isolation = _mapping(
        publication.get("spatial_output_isolation"),
        label="publication.spatial_output_isolation",
    )
    expected_forecast_files = [
        "outputs/visualizations/anomaly_increment_r2_forecast_spatial.svg",
        "outputs/visualizations/anomaly_increment_r2_forecast_spatial.html",
    ]
    expected_retrospective_files = [
        "outputs/visualizations/anomaly_increment_r2_retrospective_target_local.svg",
        "outputs/visualizations/anomaly_increment_r2_retrospective_target_local.html",
    ]
    if (
        isolation.get("physical_file_count") != 4
        or isolation.get("forecast_target_free_files") != expected_forecast_files
        or isolation.get("retrospective_target_bearing_local_restricted_files")
        != expected_retrospective_files
        or len(set(expected_forecast_files + expected_retrospective_files)) != 4
        or isolation.get("target_payload_in_forecast_files_forbidden") is not True
        or isolation.get("automatic_cross_file_target_loading_forbidden") is not True
    ):
        raise ValueError("stage-4 R2 spatial-output isolation contract changed")
    public_validator = _mapping(
        isolation.get("public_forecast_artifact_validator"),
        label="publication.spatial_output_isolation.public_forecast_artifact_validator",
    )
    if public_validator != {
        "reject_artifact_classifications": ["local_restricted", "target_bearing"],
        "forbidden_payload_fields": [
            "event_id",
            "target_coordinates",
            "target_longitude",
            "target_latitude",
            "epicenter_longitude",
            "epicenter_latitude",
            "hit_status",
            "target_marker",
        ],
        "validation_scope": ("parsed_static_dom_and_recursively_deserialized_interactive_payload"),
        "keyword_scan_or_ui_hiding_sufficient": False,
        "failure_action": "fail_closed_forbid_publication",
    }:
        raise ValueError("stage-4 R2 public forecast artifact validator changed")
    if publication.get("result_identity_requires") != [
        "dynamic_G2",
        "snapshot_equivalent_G2",
        "time_dynamic_placebo_result_distribution",
        "space_dynamic_placebo_result_distribution",
        "time_snapshot_placebo_result_distribution",
        "space_snapshot_placebo_result_distribution",
        "dynamic_G3",
        "adoption_decision",
        "adopted_variant_metrics_table",
    ]:
        raise ValueError("stage-4 R2 result identity contract changed")
    rendering = _mapping(
        publication.get("spatial_rendering_contract"),
        label="publication.spatial_rendering_contract",
    )
    if rendering.get("center_point_fallback_warning_text") != (
        "中心点示意，非面积几何；报警面积以数值为准"  # noqa: RUF001
    ):
        raise ValueError("stage-4 R2 center-point warning changed")
    axes = _mapping(publication.get("plot_axis_contract"), label="publication.plot_axis_contract")
    if (
        axes.get("numeric_ticks_required") is not True
        or axes.get("molchan_x_domain") != [0.0, 1.0]
        or axes.get("fixed_area_x_domain_km2") != [0, 960000]
    ):
        raise ValueError("stage-4 R2 plot-axis contract changed")
    limitations = _mapping(publication.get("limitations"), label="publication.limitations")
    if limitations != {
        "earthquake_available_at_assumption": (
            "available_at_equals_origin_time_is_an_optimistic_timeliness_assumption"
        ),
        "bootstrap_interval_scope": (
            "conditional_on_fixed_fitted_model_and_excludes_refit_uncertainty"
        ),
        "etas_comparator_status": "not_evaluable",
        "allowed_increment_claim": "relative_to_frozen_kde_background_only",
        "incremental_value_over_etas_claim_forbidden": True,
    }:
        raise ValueError("stage-4 R2 publication limitations changed")
    display_semantics = _mapping(
        publication.get("display_semantics"),
        label="publication.display_semantics",
    )
    if display_semantics != {
        "coverage_only_option_required": True,
        "aggregate_retrospective_view": {
            "issue_and_model_controls": "hidden_or_disabled",
            "required_summary_label_template_zh": "全部{N}个起报日汇总",
            "issue_count_source": "frozen_issue_calendar",
        },
        "peak_value_100pct": {
            "required_label_zh": "峰值网格百分位",
            "prediction_accuracy_term_forbidden": True,
        },
        "relative_strength": {
            "formula": ("peak_integrated_grid_intensity/mean_integrated_grid_intensity"),
            "absolute_probability_interpretation_forbidden": True,
        },
        "adoption": {
            "adoption_card_required": True,
            "adopted_variant_required": True,
        },
        "latest_retrospective_landmark": {
            "required_label_zh": "最新冻结日历地标",
            "current_forecast_implication_forbidden": True,
        },
        "forecast_spatial": {
            "rendered_variant": "adopted_variant",
            "unadopted_dynamic_required_label": "research_candidate",
            "unadopted_dynamic_may_not_be_current_forecast": True,
        },
        "placebo_static_panel_layout": {
            "required_panels": [
                "time_dynamic",
                "space_dynamic",
                "time_snapshot",
                "space_snapshot",
            ],
            "all_panels_within_render_bounds_required": True,
            "render_boundary_test_required": True,
        },
    }:
        raise ValueError("stage-4 R2 publication display semantics changed")


def stage4_scoring_freeze_relative_path(
    protocol: Mapping[str, object],
    key: str,
) -> PurePosixPath:
    """Return one validated R2 scoring path from the frozen machine contract."""

    validate_stage4_r2_execution_contract(protocol)
    paths = dict(_STAGE4_SCORING_FREEZE_PATHS)
    try:
        expected = paths[key]
    except KeyError as exc:
        raise ValueError(f"unknown stage-4 R2 scoring freeze path: {key}") from exc
    scoring = _mapping(
        _mapping(protocol.get("freeze"), label="freeze").get("scoring_code_freeze"),
        label="freeze.scoring_code_freeze",
    )
    raw = _string(scoring.get(key), label=f"freeze.scoring_code_freeze.{key}")
    relative = PurePosixPath(raw)
    if relative != expected or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"stage-4 R2 scoring freeze path changed: {key}")
    return relative


@dataclass(frozen=True, slots=True)
class ExpectedTargetIdentity:
    """Unobserved target identity copied verbatim from the frozen protocol.

    The path is intentionally a string rather than a :class:`~pathlib.Path` so
    ordinary protocol inspection cannot accidentally probe the filesystem.
    """

    relative_path: str
    expected_file_sha256: str
    expected_content_sha256: str
    expected_schema_sha256: str
    contract_relative_path: str
    expected_contract_sha256: str
    physical_event_id_column: str

    def __post_init__(self) -> None:
        if (
            Path(self.relative_path).is_absolute()
            or Path(self.contract_relative_path).is_absolute()
        ):
            raise ValueError("stage-4 target identity paths must remain repository-relative")
        for label, value in (
            ("expected_file_sha256", self.expected_file_sha256),
            ("expected_content_sha256", self.expected_content_sha256),
            ("expected_schema_sha256", self.expected_schema_sha256),
            ("expected_contract_sha256", self.expected_contract_sha256),
        ):
            _sha256(value, label=label)
        if self.physical_event_id_column != "event_id":
            raise ValueError("stage-4 physical event identity column changed")

    def as_expected_metadata(self) -> JsonObject:
        """Return score-free metadata; this method performs no filesystem access."""

        return {
            "relative_path": self.relative_path,
            "expected_file_sha256": self.expected_file_sha256,
            "expected_content_sha256": self.expected_content_sha256,
            "expected_schema_sha256": self.expected_schema_sha256,
            "contract_relative_path": self.contract_relative_path,
            "expected_contract_sha256": self.expected_contract_sha256,
            "physical_event_id_column": self.physical_event_id_column,
            "observed": False,
        }


@dataclass(frozen=True, slots=True)
class FrozenManifest:
    """One public, content-addressed score-free manifest."""

    manifest_id: str
    relative_path: str
    expected_file_sha256: str
    document: MappingProxyType[str, object]

    @property
    def content_sha256(self) -> str:
        return _sha256(self.document.get("content_sha256"), label=f"{self.manifest_id}.content")


@dataclass(frozen=True, slots=True)
class Stage4ProtocolBundle:
    """Validated protocol plus four public score-free manifests."""

    repository_root: Path
    protocol_path: Path
    protocol: MappingProxyType[str, object]
    fold: FrozenManifest
    feature_set: FrozenManifest
    randomness: FrozenManifest
    spatial_strata: FrozenManifest
    expected_target: ExpectedTargetIdentity
    validation_receipt: MappingProxyType[str, object]

    @property
    def design_sha256(self) -> str:
        return protocol_design_sha256(self.protocol)

    @property
    def random_input_seal_sha256(self) -> str:
        return _sha256(
            self.validation_receipt.get("random_input_seal_sha256"),
            label="random_input_seal_sha256",
        )

    @property
    def manifests(self) -> tuple[FrozenManifest, ...]:
        return (self.fold, self.feature_set, self.randomness, self.spatial_strata)

    def score_free_identity(self) -> JsonObject:
        """Return the complete identity allowed before the first target read."""

        return {
            "protocol_design_sha256": self.design_sha256,
            "protocol_tag": STAGE4_PROTOCOL_TAG,
            "expected_scoring_code_tag": STAGE4_SCORING_CODE_TAG,
            "expected_result_tag": STAGE4_RESULT_TAG,
            "random_input_seal_sha256": self.random_input_seal_sha256,
            "manifest_file_sha256": {
                item.manifest_id: item.expected_file_sha256 for item in self.manifests
            },
            "manifest_content_sha256": {
                item.manifest_id: item.content_sha256 for item in self.manifests
            },
            "expected_target": self.expected_target.as_expected_metadata(),
            "formal_attempt_count": 0,
            "target_read_count": 0,
            "locked_test_run": False,
        }


def _expected_target_identity(protocol: JsonObject) -> ExpectedTargetIdentity:
    inputs = _mapping(protocol.get("inputs"), label="inputs")
    target = _mapping(inputs.get("earthquake_target"), label="inputs.earthquake_target")
    if target.get("unavailable_before_protocol_freeze") is not True:
        raise ValueError("target must remain unavailable before the stage-4 protocol freeze")
    if target.get("human_prediction_fields_forbidden") is not True:
        raise ValueError("human prediction fields must remain forbidden")
    return ExpectedTargetIdentity(
        relative_path=_string(target.get("path"), label="earthquake_target.path"),
        expected_file_sha256=_sha256(target.get("sha256"), label="earthquake_target.sha256"),
        expected_content_sha256=_sha256(
            target.get("content_sha256"), label="earthquake_target.content_sha256"
        ),
        expected_schema_sha256=_sha256(
            target.get("schema_sha256"), label="earthquake_target.schema_sha256"
        ),
        contract_relative_path=_string(
            target.get("contract_path"), label="earthquake_target.contract_path"
        ),
        expected_contract_sha256=_sha256(
            target.get("contract_sha256"), label="earthquake_target.contract_sha256"
        ),
        physical_event_id_column=_string(
            target.get("physical_event_id_column"),
            label="earthquake_target.physical_event_id_column",
        ),
    )


def _load_manifest(
    repository_root: Path,
    *,
    manifest_id: str,
    declaration: object,
) -> FrozenManifest:
    metadata = _mapping(declaration, label=f"generated_manifests.{manifest_id}")
    relative_path = _string(metadata.get("path"), label=f"{manifest_id}.path")
    if Path(relative_path).is_absolute():
        raise ValueError("generated manifest paths must remain repository-relative")
    expected = _sha256(metadata.get("sha256"), label=f"{manifest_id}.sha256")
    path = repository_root / relative_path
    require_existing_real_directory_tree(
        repository_root,
        path.parent,
        label=f"stage-4 {manifest_id} manifest directory",
    )
    observed = _file_sha256(path)
    if observed != expected:
        raise ValueError(f"{manifest_id} file hash differs from the frozen protocol")
    document = _read_json_mapping(path)
    return FrozenManifest(
        manifest_id=manifest_id,
        relative_path=relative_path,
        expected_file_sha256=expected,
        document=MappingProxyType(document),
    )


def load_stage4_protocol_bundle(
    repository_root: Path | None = None,
    *,
    protocol_relative_path: Path = STAGE4_PROTOCOL_PATH,
) -> Stage4ProtocolBundle:
    """Load and cross-validate the frozen score-free bundle without probing targets."""

    root = (
        Path(__file__).resolve().parents[3]
        if repository_root is None
        else repository_root.resolve()
    )
    if protocol_relative_path.is_absolute():
        raise ValueError("the stage-4 protocol path must be repository-relative")
    if protocol_relative_path != STAGE4_PROTOCOL_PATH:
        raise ValueError("stage-4 execution must use the sole R2 protocol path")
    protocol_path = root / protocol_relative_path
    require_existing_real_directory_tree(
        root,
        protocol_path.parent,
        label="stage-4 protocol directory",
    )
    protocol = _read_yaml_mapping(protocol_path)
    validate_stage4_r2_execution_contract(protocol)

    generated = _mapping(protocol.get("generated_manifests"), label="generated_manifests")
    fold = _load_manifest(root, manifest_id="fold", declaration=generated.get("fold"))
    feature_set = _load_manifest(
        root,
        manifest_id="feature_set",
        declaration=generated.get("feature_set"),
    )
    randomness = _load_manifest(
        root,
        manifest_id="randomness",
        declaration=generated.get("randomness"),
    )
    spatial = _load_manifest(
        root,
        manifest_id="spatial_strata",
        declaration=generated.get("spatial_strata"),
    )
    receipt = validate_stage4_protocol_bundle(
        protocol,
        fold_manifest=fold.document,
        feature_manifest=feature_set.document,
        randomness_manifest=randomness.document,
        spatial_manifest=spatial.document,
    )
    if receipt.get("target_read_count") != 0:
        raise ValueError("score-free bundle unexpectedly reports a target read")
    return Stage4ProtocolBundle(
        repository_root=root,
        protocol_path=protocol_path,
        protocol=MappingProxyType(protocol),
        fold=fold,
        feature_set=feature_set,
        randomness=randomness,
        spatial_strata=spatial,
        expected_target=_expected_target_identity(protocol),
        validation_receipt=MappingProxyType(receipt),
    )


__all__ = [
    "STAGE4_ATTEMPT_LEDGER_RELATIVE_PATH",
    "STAGE4_CHECKPOINT_ROOT_RELATIVE_PATH",
    "STAGE4_EXECUTION_REVISION",
    "STAGE4_FORMAL_PREFLIGHT_RECEIPT_RELATIVE_PATH",
    "STAGE4_FULL_NON_TARGET_JUNIT_RELATIVE_PATH",
    "STAGE4_JUNIT_RELATIVE_PATH",
    "STAGE4_LOGICAL_REPLAY_AUDIT_RELATIVE_PATH",
    "STAGE4_PROTOCOL_PATH",
    "STAGE4_PROTOCOL_TAG",
    "STAGE4_QUALIFICATION_RELATIVE_PATH",
    "STAGE4_RESULT_TAG",
    "STAGE4_SCORING_CODE_TAG",
    "STAGE4_SCORING_SEAL_RELATIVE_PATH",
    "STAGE4_TARGET_READ_LEDGER_RELATIVE_PATH",
    "ExpectedTargetIdentity",
    "FrozenManifest",
    "Stage4ProtocolBundle",
    "load_stage4_protocol_bundle",
    "stage4_scoring_freeze_relative_path",
    "validate_stage4_r2_execution_contract",
]
