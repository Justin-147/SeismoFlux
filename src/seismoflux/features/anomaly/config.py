"""Strict validation of the complete frozen stage-3 machine protocol."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Self

from pydantic import model_validator

from seismoflux.config import StrictModel, load_yaml_mapping
from seismoflux.data.common import canonical_json_bytes

ANOMALY_HISTORY_CONTRACT_VERSION = "0.3.0"
ANOMALY_HISTORY_PROTOCOL_SEMANTIC_SHA256 = (
    "160d9a0e9f581d62bd3a021ac74aba6634172cec3688d9f2e3492754169e01bc"
)
TEMPORAL_WINDOWS_WEEKS = (4, 8, 13, 26, 52)
SPATIAL_RADII_KM = (50.0, 100.0, 200.0, 300.0, 500.0)
GAUSSIAN_BANDWIDTHS_KM = SPATIAL_RADII_KM
ALLOWED_SCIENTIFIC_DATASETS = (
    "anomaly_observation",
    "anomaly_report_period",
)


def _semantic_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(value))).hexdigest()


def _mapping_value(mapping: Mapping[str, object], key: str) -> object:
    value = mapping.get(key)
    if value is None:
        raise ValueError(f"missing frozen stage-3 protocol field: {key}")
    return value


def _nested_mapping(mapping: Mapping[str, object], *keys: str) -> Mapping[str, object]:
    current = mapping
    traversed: list[str] = []
    for key in keys:
        traversed.append(key)
        value = current.get(key)
        if not isinstance(value, Mapping):
            location = ".".join(traversed)
            raise ValueError(f"frozen stage-3 protocol field must be a mapping: {location}")
        current = value
    return current


def _tuple_value(mapping: Mapping[str, object], key: str) -> tuple[object, ...]:
    value = _mapping_value(mapping, key)
    if not isinstance(value, list | tuple):
        raise ValueError(f"frozen stage-3 protocol field must be a sequence: {key}")
    return tuple(value)


class AnomalyHistoryConfig(StrictModel):
    """The exact complete ``configs/anomaly_history.yaml`` protocol.

    Every top-level section is represented, undocumented top-level fields are rejected,
    and a canonical semantic fingerprint validates every nested key and value.  This keeps
    the large preregistration strict without maintaining a second, divergent mini-schema.
    Any scientific protocol change therefore requires an intentional versioned update to
    both the YAML and this validator.
    """

    schema_version: Literal[1]
    protocol_version: Literal["0.3.0"]
    stage: Literal[3]
    execution_mode: Literal["feature_only_no_target_scoring"]
    status: Literal["preregistered_before_stage3_feature_execution"]
    frozen_on: str
    freeze_tag: Literal["v0.3.0-anomaly-feature-protocol"]
    blueprint: Literal["SEISMOFLUX_IMPLEMENTATION_HANDOFF.md"]

    authorization: dict[str, object]
    scientific_inputs: dict[str, object]
    forbidden_source_semantics: dict[str, object]
    time_semantics: dict[str, object]
    missing_periods: dict[str, object]
    state_reconstruction: dict[str, object]
    query_grid: dict[str, object]
    spatial_features: dict[str, object]
    feature_families: dict[str, object]
    reliability: dict[str, object]
    lineage: dict[str, object]
    audit: dict[str, object]
    outputs: dict[str, object]
    locked_test: dict[str, object]
    stage4_boundary: dict[str, object]
    acceptance: dict[str, object]

    @model_validator(mode="before")
    @classmethod
    def validate_complete_protocol_semantics(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            raise ValueError("stage-3 anomaly history protocol must be a mapping")
        semantic_hash = _semantic_sha256(value)
        if semantic_hash != ANOMALY_HISTORY_PROTOCOL_SEMANTIC_SHA256:
            raise ValueError(
                "stage-3 anomaly history protocol differs from the complete frozen "
                "semantic contract: "
                f"expected={ANOMALY_HISTORY_PROTOCOL_SEMANTIC_SHA256}, "
                f"observed={semantic_hash}"
            )
        return value

    @model_validator(mode="after")
    def validate_safety_critical_semantics(self) -> Self:
        scientific_inputs = self.scientific_inputs
        if _tuple_value(scientific_inputs, "exact_dataset_names") != (ALLOWED_SCIENTIFIC_DATASETS):
            raise ValueError("scientific inputs must be exactly the two anomaly datasets")
        if _mapping_value(scientific_inputs, "earthquake_catalog_forbidden") is not True:
            raise ValueError("earthquake catalogs must remain forbidden in stage 3")
        if _mapping_value(scientific_inputs, "earthquake_target_labels_forbidden") is not True:
            raise ValueError("earthquake target labels must remain forbidden in stage 3")

        windows = _nested_mapping(self.time_semantics, "windows")
        if _tuple_value(windows, "lookback_weeks") != TEMPORAL_WINDOWS_WEEKS:
            raise ValueError("temporal windows must be exactly 4, 8, 13, 26, and 52 weeks")
        if _mapping_value(self.time_semantics, "actual_report_period_count") != 205:
            raise ValueError("the full actual anomaly schedule must contain 205 periods")

        closed_balls = _nested_mapping(self.spatial_features, "fixed_closed_balls")
        if _tuple_value(closed_balls, "radii_km") != SPATIAL_RADII_KM:
            raise ValueError("closed-ball radii must be exactly 50, 100, 200, 300, and 500 km")
        gaussian = _nested_mapping(self.spatial_features, "gaussian_kernels")
        if _tuple_value(gaussian, "bandwidths_km") != GAUSSIAN_BANDWIDTHS_KM:
            raise ValueError("Gaussian bandwidths must be exactly 50, 100, 200, 300, and 500 km")
        if _mapping_value(gaussian, "truncate_at_sigma") != 3.0:
            raise ValueError("Gaussian anomaly kernels must be truncated at 3 sigma")
        if (
            _mapping_value(
                self.spatial_features,
                "outside_anomalies_may_influence_inside_query_cells",
            )
            is not True
        ):
            raise ValueError("registered external anomalies must remain eligible by distance")

        reporting_proxy = _nested_mapping(
            self.feature_families,
            "reporting_coverage_proxy",
        )
        if _mapping_value(reporting_proxy, "required_name_suffix") != "reporting_coverage_proxy":
            raise ValueError("coverage features must remain explicitly named as proxies")
        if (
            _mapping_value(reporting_proxy, "absolute_observation_coverage_claim_forbidden")
            is not True
        ):
            raise ValueError("absolute observation-coverage claims must remain forbidden")

        fault_features = _nested_mapping(self.feature_families, "fault_features")
        if _mapping_value(fault_features, "status") != "deferred_to_stage_5":
            raise ValueError("fault fields must remain deferred to stage 5")

        grades = _nested_mapping(self.reliability, "grades")
        expected_weights = {"high": 1.0, "cautious": 0.5, "excluded": 0.0}
        for grade, expected_weight in expected_weights.items():
            grade_config = _nested_mapping(grades, grade)
            if _mapping_value(grade_config, "weight") != expected_weight:
                raise ValueError(f"frozen reliability weight changed for grade {grade}")

        if _mapping_value(self.locked_test, "run") is not False:
            raise ValueError("stage 3 must not run the locked test")
        if _mapping_value(self.locked_test, "action") != "do_not_run":
            raise ValueError("stage-3 locked-test action must remain do_not_run")
        if any(
            _mapping_value(self.locked_test, field)
            for field in ("target_ids", "score_ids", "artifact_ids")
        ):
            raise ValueError("stage-3 locked-test identity lists must remain empty")
        return self

    @property
    def contract_version(self) -> str:
        return self.protocol_version

    @property
    def expected_report_period_count(self) -> int:
        return 205

    @property
    def temporal_windows_weeks(self) -> tuple[int, ...]:
        return TEMPORAL_WINDOWS_WEEKS

    @property
    def spatial_radii_km(self) -> tuple[float, ...]:
        return SPATIAL_RADII_KM

    @property
    def gaussian_bandwidths_km(self) -> tuple[float, ...]:
        return GAUSSIAN_BANDWIDTHS_KM

    @property
    def query_grid_cell_km(self) -> float:
        return 25.0

    @property
    def gaussian_truncate_at_sigma(self) -> float:
        return 3.0

    @property
    def locked_test_run(self) -> bool:
        return False


def load_anomaly_history_config(path: str | Path) -> AnomalyHistoryConfig:
    """Load and strictly validate the one complete frozen stage-3 YAML protocol."""

    config_path = Path(path)
    return AnomalyHistoryConfig.model_validate(load_yaml_mapping(config_path))


__all__ = [
    "ALLOWED_SCIENTIFIC_DATASETS",
    "ANOMALY_HISTORY_CONTRACT_VERSION",
    "ANOMALY_HISTORY_PROTOCOL_SEMANTIC_SHA256",
    "GAUSSIAN_BANDWIDTHS_KM",
    "SPATIAL_RADII_KM",
    "TEMPORAL_WINDOWS_WEEKS",
    "AnomalyHistoryConfig",
    "load_anomaly_history_config",
]
