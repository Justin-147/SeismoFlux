"""Public-safe feature dictionary and Arrow contract for stage 3.

The dictionary describes every numerical field emitted by :mod:`spatial` and
:mod:`trajectory` without publishing query-cell, station, anomaly, or source-row
details.  It is deliberately independent of the stage-1 contract registry so stage 3
can evolve its local feature-store schema without modifying the frozen ingestion
contracts.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

import pyarrow as pa

from seismoflux.background.artifacts import CANONICAL_JSON_VERSION, canonical_json_bytes
from seismoflux.features.anomaly.nulls import NULL_REASON_DEFINITIONS, NullReasonCode
from seismoflux.features.anomaly.spatial import DISCIPLINE_NAMES, SPATIAL_SCALES_KM
from seismoflux.features.anomaly.trajectory import TRAJECTORY_WINDOWS_WEEKS

FEATURE_DICTIONARY_SCHEMA_VERSION: Final[int] = 1
FEATURE_PROTOCOL_VERSION: Final[str] = "0.3.0"
FEATURE_SEMANTICS: Final[str] = "relative_features_not_absolute_probability"
REPORTING_PROXY_INTERPRETATION: Final[str] = (
    "reporting_coverage_proxy_not_absolute_observation_coverage_or_probability"
)
FEATURE_DICTIONARY_ID_PREFIX: Final[str] = "anomaly-feature-dictionary"

SPATIAL_KERNELS: Final[tuple[str, ...]] = ("closed_ball", "gaussian")
TRAJECTORY_KERNELS: Final[tuple[str, ...]] = ("closed_ball",)
TRAJECTORY_BASE_SOURCE_FIELDS: Final[tuple[str, ...]] = (
    "listed_count",
    "source_new_count",
    "first_seen_count",
    "explicit_end_count",
    "not_continued_count",
)
NULL_REASON_VALID_CODE: Final[int] = int(NullReasonCode.VALID)
FEATURE_STORE_SORT_KEYS: Final[tuple[str, ...]] = (
    "issue_index",
    "cell_row",
    "cell_column",
    "cell_id",
)

_LOCAL_FEATURE_STORE_FIELDS: Final[tuple[tuple[str, str, bool], ...]] = (
    ("issue_index", "int16", False),
    ("issue_time_utc", "timestamp_us_utc", False),
    ("issue_report_id", "string", False),
    ("issue_report_date", "date32", False),
    ("issue_report_year", "int16", False),
    ("issue_report_period", "int16", False),
    ("state_snapshot_id", "string", False),
    ("lineage_digest", "string", False),
    ("feature_dictionary_sha256", "string", False),
    ("grid_id", "string", False),
    ("equal_area_crs", "string", False),
    ("cell_size_km", "float64", False),
    ("cell_id", "string", False),
    ("cell_row", "int32", False),
    ("cell_column", "int32", False),
    ("query_x_m", "float64", False),
    ("query_y_m", "float64", False),
    ("clipped_area_km2", "float64", False),
)

_NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]*")
_FORBIDDEN_CONCEPTS: Final[frozenset[str]] = frozenset(
    {
        "completeness",
        "earthquake",
        "epicenter",
        "epicentre",
        "fault",
        "hit",
        "magnitude",
        "mc",
        "recall",
        "score",
        "target",
    }
)
_CROSS_FAULT_DISCIPLINE_NAME: Final[str] = "discipline_cross_fault_count"
_CROSS_FAULT_DISCIPLINE_FORMULA: Final[str] = (
    "sum(kernel_weight * indicator(listed and discipline equals cross_fault))"
)
_PUBLIC_DICTIONARY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "canonical_json_version",
        "feature_count",
        "features",
        "null_reason_definitions",
        "protocol_version",
        "schema_version",
        "semantics",
        "storage_contract",
    }
)
_PUBLIC_FEATURE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "arrow_type",
        "causal_sources",
        "emits_validity_and_sample_count",
        "family",
        "formula",
        "interpretation",
        "kernels",
        "name",
        "null_reason_codes",
        "null_reasons",
        "null_semantics",
        "nullable",
        "producer",
        "quality_companions",
        "scales_km",
        "source_output_field",
        "storage_value_columns",
        "unit",
        "windows_weeks",
    }
)
_FORBIDDEN_PUBLIC_KEYS: Final[frozenset[str]] = frozenset(
    {
        "longitude",
        "latitude",
        "x_m",
        "y_m",
        "xy_m",
        "geometry",
        "wkt",
        "wkb",
        "geojson",
        "cell_id",
        "station_id",
        "anomaly_id",
        "observation_id",
        "source_file",
        "source_sheet",
        "source_row",
    }
)
_ALLOWED_INTERPRETATIONS: Final[frozenset[str]] = frozenset(
    {FEATURE_SEMANTICS, REPORTING_PROXY_INTERPRETATION}
)
_NULL_REASON_CODE_BY_NAME: Final[dict[str, int]] = {
    reason: code for code, reason in NULL_REASON_DEFINITIONS.items()
}

ArrowTypeName: TypeAlias = Literal["float64", "int64", "bool"]
FeatureProducer: TypeAlias = Literal[
    "spatial_v1",
    "trajectory_v1",
    "local_coverage_v1",
    "protocol_v1",
]
FeatureFamily: TypeAlias = Literal[
    "snapshot_state",
    "reliability",
    "reporting_coverage_proxy",
    "multidisciplinary",
    "age_duration",
    "spatial_pattern",
    "temporal_trajectory",
]

_ALLOWED_ARROW_TYPES: Final[frozenset[str]] = frozenset({"float64", "int64", "bool"})
_ALLOWED_PRODUCERS: Final[frozenset[str]] = frozenset(
    {"spatial_v1", "trajectory_v1", "local_coverage_v1", "protocol_v1"}
)
_ALLOWED_FAMILIES: Final[frozenset[str]] = frozenset(
    {
        "snapshot_state",
        "reliability",
        "reporting_coverage_proxy",
        "multidisciplinary",
        "age_duration",
        "spatial_pattern",
        "temporal_trajectory",
    }
)


def _scale_token(scale_km: float) -> str:
    rounded = round(scale_km)
    if not math.isclose(scale_km, rounded, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"storage column scale must be an integer kilometre value: {scale_km}")
    return str(rounded)


def _kernel_storage_prefix(kernel: str, scale_km: float) -> str:
    if kernel == "closed_ball":
        kernel_name = "radius"
    elif kernel == "gaussian":
        kernel_name = "gaussian"
    else:
        raise ValueError(f"unknown storage kernel: {kernel}")
    return f"{kernel_name}_{_scale_token(scale_km)}km"


def _local_key_schema_sha256() -> str:
    payload = [
        {"name": name, "nullable": nullable, "type": type_name}
        for name, type_name, nullable in _LOCAL_FEATURE_STORE_FIELDS
    ]
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _quality_companion_mapping(*, nullable: bool, emits_quality: bool) -> dict[str, str | None]:
    return {
        "null_reason_code_suffix": "__null_reason_code" if nullable else None,
        "null_reason_code_type": "int8" if nullable else None,
        "sample_count_suffix": "__sample_count" if emits_quality else None,
        "sample_count_type": "int64" if emits_quality else None,
        "validity_suffix": "__valid" if emits_quality else None,
        "validity_type": "bool" if emits_quality else None,
    }


def _storage_contract_mapping(value_column_count: int) -> dict[str, object]:
    return {
        "layout": "one_issue_cell_wide_row",
        "local_key_schema_sha256": _local_key_schema_sha256(),
        "local_reporting_coverage_scope": "same_query_cell_kernel_scale_only",
        "local_reporting_coverage_window_days": 364,
        "null_reason_code_type": "int8",
        "null_reason_valid_code": NULL_REASON_VALID_CODE,
        "spatial_column_template": (
            "radius_{scale}km__{spatial_feature} or gaussian_{scale}km__{spatial_feature}"
        ),
        "trajectory_base_source_fields": list(TRAJECTORY_BASE_SOURCE_FIELDS),
        "trajectory_column_template": "radius_{scale}km__{base}__{trajectory_feature}",
        "trajectory_kernel": "closed_ball",
        "trajectory_series_count": (len(TRAJECTORY_BASE_SOURCE_FIELDS) * len(SPATIAL_SCALES_KM)),
        "value_column_count": value_column_count,
    }


def _forbidden_concept(text: str) -> str | None:
    normalized = re.sub(r"\bm[\s_-]*c\b", " mc ", text.casefold())
    tokens: list[str] = re.findall(r"[a-z0-9]+", normalized)
    for token in tokens:
        if token in _FORBIDDEN_CONCEPTS:
            return token
    return None


def _without_approved_cross_fault_discipline(text: str) -> str:
    return text.replace(_CROSS_FAULT_DISCIPLINE_NAME, "").replace(
        _CROSS_FAULT_DISCIPLINE_FORMULA,
        "",
    )


@dataclass(frozen=True, slots=True)
class FeatureDefinition:
    """One frozen stage-3 feature and its storage/publication semantics."""

    name: str
    source_output_field: str | None
    producer: FeatureProducer
    family: FeatureFamily
    kernels: tuple[str, ...]
    scales_km: tuple[float, ...]
    windows_weeks: tuple[int, ...]
    formula: str
    unit: str
    arrow_type: ArrowTypeName
    causal_sources: tuple[str, ...]
    nullable: bool
    null_semantics: str
    null_reasons: tuple[str, ...]
    emits_validity_and_sample_count: bool
    interpretation: str

    def __post_init__(self) -> None:
        if self.producer not in _ALLOWED_PRODUCERS:
            raise ValueError(f"invalid feature producer: {self.name}")
        if self.family not in _ALLOWED_FAMILIES:
            raise ValueError(f"invalid feature family: {self.name}")
        if self.arrow_type not in _ALLOWED_ARROW_TYPES:
            raise ValueError(f"invalid feature Arrow type: {self.name}")
        if _NAME_PATTERN.fullmatch(self.name) is None:
            raise ValueError(f"invalid feature name: {self.name!r}")
        if self.source_output_field is not None and (
            _NAME_PATTERN.fullmatch(self.source_output_field) is None
        ):
            raise ValueError(f"invalid source output field: {self.source_output_field!r}")
        if not self.formula or self.formula != self.formula.strip():
            raise ValueError(f"feature formula must be non-empty and trimmed: {self.name}")
        if not self.unit or self.unit != self.unit.strip():
            raise ValueError(f"feature unit must be non-empty and trimmed: {self.name}")
        if not self.causal_sources or any(
            not source or source != source.strip() for source in self.causal_sources
        ):
            raise ValueError(f"feature causal sources must be non-empty: {self.name}")
        if not self.null_semantics or self.null_semantics != self.null_semantics.strip():
            raise ValueError(f"feature null semantics must be non-empty: {self.name}")
        if self.nullable != bool(self.null_reasons):
            raise ValueError(
                f"nullable feature must declare reasons and non-null feature must not: {self.name}"
            )
        if any(not reason or reason != reason.strip() for reason in self.null_reasons):
            raise ValueError(f"feature null reasons must be non-empty and trimmed: {self.name}")
        if self.interpretation not in _ALLOWED_INTERPRETATIONS:
            raise ValueError(f"invalid feature interpretation: {self.name}")
        if self.family == "reporting_coverage_proxy":
            if not self.name.endswith("reporting_coverage_proxy"):
                raise ValueError(f"reporting proxy feature name lacks required suffix: {self.name}")
            if self.interpretation != REPORTING_PROXY_INTERPRETATION:
                raise ValueError(f"reporting proxy interpretation is not explicit: {self.name}")
        elif self.interpretation != FEATURE_SEMANTICS:
            raise ValueError(f"non-proxy feature must use non-probability semantics: {self.name}")
        if any(kernel not in SPATIAL_KERNELS for kernel in self.kernels):
            raise ValueError(f"unknown spatial kernel in feature: {self.name}")
        if len(self.kernels) != len(set(self.kernels)):
            raise ValueError(f"duplicate spatial kernel in feature: {self.name}")
        if any(
            not math.isfinite(scale) or scale <= 0.0 or scale not in SPATIAL_SCALES_KM
            for scale in self.scales_km
        ):
            raise ValueError(f"feature uses an unfrozen spatial scale: {self.name}")
        if any(window not in TRAJECTORY_WINDOWS_WEEKS for window in self.windows_weeks):
            raise ValueError(f"feature uses an unfrozen trajectory window: {self.name}")
        if self.producer in {"spatial_v1", "local_coverage_v1"} and (
            self.kernels != SPATIAL_KERNELS or self.scales_km != SPATIAL_SCALES_KM
        ):
            raise ValueError(
                f"spatial/local feature must cover both kernels and all scales: {self.name}"
            )
        if self.producer == "trajectory_v1":
            if self.kernels != TRAJECTORY_KERNELS or self.scales_km != SPATIAL_SCALES_KM:
                raise ValueError(
                    "trajectory feature must use closed-ball core series at all scales: "
                    f"{self.name}"
                )
            if not self.emits_validity_and_sample_count:
                raise ValueError(
                    f"trajectory feature must preserve validity and sample count: {self.name}"
                )
        if len(self.null_reasons) > 127:
            raise ValueError(f"int8 null-reason codes exhausted for feature: {self.name}")
        if len(self.null_reasons) != len(set(self.null_reasons)):
            raise ValueError(f"feature null reasons must be unique: {self.name}")
        unknown_null_reasons = set(self.null_reasons) - set(_NULL_REASON_CODE_BY_NAME)
        if unknown_null_reasons:
            raise ValueError(
                f"feature uses unfrozen null reasons {sorted(unknown_null_reasons)}: {self.name}"
            )
        if "valid" in self.null_reasons:
            raise ValueError(f"valid is not a null reason: {self.name}")

        scientific_text = " ".join(
            (
                self.name,
                self.source_output_field or "",
                self.family,
                self.formula,
                self.unit,
                self.null_semantics,
                *self.causal_sources,
                *self.null_reasons,
            )
        )
        if self.name == _CROSS_FAULT_DISCIPLINE_NAME:
            if (
                self.source_output_field != _CROSS_FAULT_DISCIPLINE_NAME
                or self.formula != _CROSS_FAULT_DISCIPLINE_FORMULA
            ):
                raise ValueError("cross_fault is allowed only as the frozen source discipline")
            scientific_text = _without_approved_cross_fault_discipline(scientific_text)
        forbidden = _forbidden_concept(scientific_text)
        if forbidden is not None:
            raise ValueError(f"forbidden concept {forbidden!r} in feature {self.name}")

    @property
    def null_reason_codes(self) -> dict[str, int]:
        if not self.nullable:
            return {}
        return {
            "valid": NULL_REASON_VALID_CODE,
            **{reason: _NULL_REASON_CODE_BY_NAME[reason] for reason in self.null_reasons},
        }

    def storage_value_columns(self) -> tuple[str, ...]:
        """Return frozen wide-table value columns for this logical feature."""

        if self.producer in {"spatial_v1", "local_coverage_v1"}:
            return tuple(
                f"{_kernel_storage_prefix(kernel, scale_km)}__{self.name}"
                for kernel in self.kernels
                for scale_km in self.scales_km
            )
        if self.producer == "trajectory_v1":
            return tuple(
                (
                    f"{_kernel_storage_prefix('closed_ball', scale_km)}__"
                    f"{base_source_field}__{self.name}"
                )
                for base_source_field in TRAJECTORY_BASE_SOURCE_FIELDS
                for scale_km in self.scales_km
            )
        return (self.name,)

    def validity_field(self, value_column: str) -> str | None:
        if not self.emits_validity_and_sample_count:
            return None
        return f"{value_column}__valid"

    def sample_count_field(self, value_column: str) -> str | None:
        if not self.emits_validity_and_sample_count:
            return None
        return f"{value_column}__sample_count"

    def null_reason_code_field(self, value_column: str) -> str | None:
        if not self.nullable:
            return None
        return f"{value_column}__null_reason_code"

    def as_mapping(self) -> dict[str, object]:
        """Return the public-safe machine representation."""

        storage_columns = self.storage_value_columns()
        return {
            "arrow_type": self.arrow_type,
            "causal_sources": list(self.causal_sources),
            "emits_validity_and_sample_count": self.emits_validity_and_sample_count,
            "family": self.family,
            "formula": self.formula,
            "interpretation": self.interpretation,
            "kernels": list(self.kernels),
            "name": self.name,
            "null_reason_codes": self.null_reason_codes,
            "null_reasons": list(self.null_reasons),
            "null_semantics": self.null_semantics,
            "nullable": self.nullable,
            "producer": self.producer,
            "quality_companions": _quality_companion_mapping(
                nullable=self.nullable,
                emits_quality=self.emits_validity_and_sample_count,
            ),
            "scales_km": list(self.scales_km),
            "source_output_field": self.source_output_field,
            "storage_value_columns": list(storage_columns),
            "unit": self.unit,
            "windows_weeks": list(self.windows_weeks),
        }


@dataclass(frozen=True, slots=True)
class FeatureDictionary:
    """Deterministically ordered collection of stage-3 definitions."""

    definitions: tuple[FeatureDefinition, ...]
    schema_version: int = FEATURE_DICTIONARY_SCHEMA_VERSION
    protocol_version: str = FEATURE_PROTOCOL_VERSION
    semantics: str = FEATURE_SEMANTICS

    def __post_init__(self) -> None:
        if self.schema_version != FEATURE_DICTIONARY_SCHEMA_VERSION:
            raise ValueError("unsupported feature dictionary schema version")
        if self.protocol_version != FEATURE_PROTOCOL_VERSION:
            raise ValueError("feature dictionary protocol version must be 0.3.0")
        if self.semantics != FEATURE_SEMANTICS:
            raise ValueError("feature dictionary must explicitly reject probability semantics")
        ordered = tuple(sorted(self.definitions, key=lambda item: item.name))
        object.__setattr__(self, "definitions", ordered)
        names = tuple(item.name for item in ordered)
        if len(names) != len(set(names)):
            raise ValueError("feature dictionary names must be unique")
        source_keys = tuple(
            (item.producer, item.source_output_field)
            for item in ordered
            if item.source_output_field is not None
        )
        if len(source_keys) != len(set(source_keys)):
            raise ValueError("producer source-output fields must be covered exactly once")

        storage_names = {name for name, _, _ in _LOCAL_FEATURE_STORE_FIELDS}
        for definition in ordered:
            for value_column in definition.storage_value_columns():
                if value_column in storage_names:
                    raise ValueError(f"feature storage field collision: {value_column}")
                storage_names.add(value_column)
                for companion in (
                    definition.validity_field(value_column),
                    definition.sample_count_field(value_column),
                    definition.null_reason_code_field(value_column),
                ):
                    if companion is None:
                        continue
                    if companion in storage_names:
                        raise ValueError(f"feature storage field collision: {companion}")
                    storage_names.add(companion)

    def as_mapping(self) -> dict[str, object]:
        storage_value_columns = self.storage_value_columns()
        return {
            "canonical_json_version": CANONICAL_JSON_VERSION,
            "feature_count": len(self.definitions),
            "features": [definition.as_mapping() for definition in self.definitions],
            "null_reason_definitions": {
                str(code): reason for code, reason in sorted(NULL_REASON_DEFINITIONS.items())
            },
            "protocol_version": self.protocol_version,
            "schema_version": self.schema_version,
            "semantics": self.semantics,
            "storage_contract": _storage_contract_mapping(len(storage_value_columns)),
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.as_mapping())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    @property
    def dictionary_id(self) -> str:
        return f"{FEATURE_DICTIONARY_ID_PREFIX}-{self.sha256[:16]}"

    def by_name(self) -> dict[str, FeatureDefinition]:
        return {definition.name: definition for definition in self.definitions}

    def storage_value_columns(self) -> tuple[str, ...]:
        return tuple(
            column
            for definition in self.definitions
            for column in definition.storage_value_columns()
        )

    def storage_column_map(self) -> dict[str, FeatureDefinition]:
        return {
            column: definition
            for definition in self.definitions
            for column in definition.storage_value_columns()
        }

    def source_field_map(self, producer: FeatureProducer) -> dict[str, str]:
        return {
            definition.source_output_field: definition.name
            for definition in self.definitions
            if definition.producer == producer and definition.source_output_field is not None
        }


def _definition(
    *,
    name: str,
    source_output_field: str | None,
    producer: FeatureProducer,
    family: FeatureFamily,
    formula: str,
    unit: str,
    causal_sources: tuple[str, ...],
    nullable: bool = False,
    null_semantics: str = "empty causal support is the numeric value zero",
    null_reasons: tuple[str, ...] = (),
    kernels: tuple[str, ...] = (),
    scales_km: tuple[float, ...] = (),
    windows_weeks: tuple[int, ...] = (),
    arrow_type: ArrowTypeName = "float64",
    emits_validity_and_sample_count: bool = False,
    interpretation: str = FEATURE_SEMANTICS,
) -> FeatureDefinition:
    return FeatureDefinition(
        name=name,
        source_output_field=source_output_field,
        producer=producer,
        family=family,
        kernels=kernels,
        scales_km=scales_km,
        windows_weeks=windows_weeks,
        formula=formula,
        unit=unit,
        arrow_type=arrow_type,
        causal_sources=causal_sources,
        nullable=nullable,
        null_semantics=null_semantics,
        null_reasons=null_reasons,
        emits_validity_and_sample_count=emits_validity_and_sample_count,
        interpretation=interpretation,
    )


def _spatial_definition(
    *,
    name: str,
    source_output_field: str | None = None,
    family: FeatureFamily,
    formula: str,
    unit: str,
    nullable: bool = False,
    null_semantics: str = "empty causal support is the numeric value zero",
    null_reasons: tuple[str, ...] = (),
    interpretation: str = FEATURE_SEMANTICS,
) -> FeatureDefinition:
    return _definition(
        name=name,
        source_output_field=source_output_field or name,
        producer="spatial_v1",
        family=family,
        formula=formula,
        unit=unit,
        causal_sources=(
            "causal anomaly entity state available by the issue time",
            "fixed query-cell distance and frozen spatial kernel",
        ),
        nullable=nullable,
        null_semantics=null_semantics,
        null_reasons=null_reasons,
        kernels=SPATIAL_KERNELS,
        scales_km=SPATIAL_SCALES_KM,
        interpretation=interpretation,
    )


def _spatial_definitions() -> list[FeatureDefinition]:
    definitions: list[FeatureDefinition] = []
    statuses = (
        "listed",
        "source_new",
        "first_seen",
        "explicit_end",
        "not_continued",
        "relisted",
        "right_censored",
        "left_truncated",
        "late_entry",
        "temporary_entity",
    )
    for status in statuses:
        definitions.append(
            _spatial_definition(
                name=f"{status}_count",
                family="snapshot_state",
                formula=f"sum(kernel_weight * indicator({status}))",
                unit="entity_equivalent_count",
            )
        )
        definitions.append(
            _spatial_definition(
                name=f"{status}_weighted_count",
                family="snapshot_state",
                formula=(f"sum(kernel_weight * indicator({status}) * reliability_weight_1_0_5_0)"),
                unit="reliability_weighted_entity_equivalent_count",
            )
        )

    for name, formula in (
        (
            "first_seen_rate",
            "first_seen_count / (listed_count - left_truncated_count)",
        ),
        (
            "first_seen_weighted_rate",
            (
                "first_seen_weighted_count / "
                "(reliability_weighted_listed_count - left_truncated_weighted_count)"
            ),
        ),
        (
            "right_censored_fraction",
            "right_censored_count / listed_count",
        ),
        (
            "right_censored_weighted_fraction",
            "right_censored_weighted_count / reliability_weighted_listed_count",
        ),
    ):
        definitions.append(
            _spatial_definition(
                name=name,
                family="snapshot_state",
                formula=formula,
                unit="ratio_0_to_1",
                nullable=True,
                null_semantics="null means the causal denominator is zero",
                null_reasons=("zero_denominator",),
            )
        )

    reliability_formulas = {
        "high_reliability_listed_count": (
            "sum(kernel_weight * indicator(listed and reliability_grade_high))"
        ),
        "cautious_reliability_listed_count": (
            "sum(kernel_weight * indicator(listed and reliability_grade_cautious))"
        ),
        "excluded_reliability_listed_count": (
            "sum(kernel_weight * indicator(listed and reliability_grade_excluded))"
        ),
        "reliability_weighted_listed_count": (
            "sum(kernel_weight * indicator(listed) * reliability_weight_1_0_5_0)"
        ),
    }
    for name, formula in reliability_formulas.items():
        definitions.append(
            _spatial_definition(
                name=name,
                family="reliability",
                formula=formula,
                unit="entity_equivalent_count",
            )
        )

    definitions.append(
        _spatial_definition(
            name="spatial_weight_mass",
            family="spatial_pattern",
            formula="sum(kernel_weight for every coordinate_valid causal entity in support)",
            unit="kernel_weight_mass",
        )
    )

    reporting_proxy_fields = (
        (
            "distinct_reporting_station_count_reporting_coverage_proxy",
            "unique_station_count",
            "sum(maximum kernel_weight for each distinct reporting station)",
        ),
        (
            "distinct_reporting_measurement_count_reporting_coverage_proxy",
            "unique_measurement_count",
            "sum(maximum kernel_weight for each distinct reporting measurement)",
        ),
        (
            "distinct_reporting_station_measurement_count_reporting_coverage_proxy",
            "unique_station_measurement_count",
            "sum(maximum kernel_weight for each distinct reporting station measurement pair)",
        ),
    )
    for name, source_output_field, formula in reporting_proxy_fields:
        definitions.append(
            _spatial_definition(
                name=name,
                source_output_field=source_output_field,
                family="reporting_coverage_proxy",
                formula=formula,
                unit="kernel_weighted_distinct_reporting_identifier_mass",
                interpretation=REPORTING_PROXY_INTERPRETATION,
            )
        )

    definitions.extend(
        (
            _spatial_definition(
                name="multidisciplinary_entity_count",
                family="multidisciplinary",
                formula="sum(kernel_weight * indicator(listed and multidisciplinary))",
                unit="entity_equivalent_count",
            ),
            _spatial_definition(
                name="multidisciplinary_entity_weighted_count",
                family="multidisciplinary",
                formula=(
                    "sum(kernel_weight * indicator(listed and multidisciplinary) * "
                    "reliability_weight_1_0_5_0)"
                ),
                unit="reliability_weighted_entity_equivalent_count",
            ),
            _spatial_definition(
                name="multidisciplinary_entity_fraction",
                family="multidisciplinary",
                formula="multidisciplinary_entity_count / listed_count",
                unit="ratio_0_to_1",
                nullable=True,
                null_semantics="null means the causal listed denominator is zero",
                null_reasons=("zero_denominator",),
            ),
            _spatial_definition(
                name="multidisciplinary_entity_weighted_fraction",
                family="multidisciplinary",
                formula=(
                    "multidisciplinary_entity_weighted_count / reliability_weighted_listed_count"
                ),
                unit="ratio_0_to_1",
                nullable=True,
                null_semantics="null means the causal weighted denominator is zero",
                null_reasons=("zero_denominator",),
            ),
        )
    )

    definitions.append(
        _spatial_definition(
            name="discipline_count",
            family="multidisciplinary",
            formula="count(discipline categories with positive listed kernel mass)",
            unit="discipline_category_count",
        )
    )
    for discipline in DISCIPLINE_NAMES:
        definitions.append(
            _spatial_definition(
                name=f"discipline_{discipline}_count",
                family="multidisciplinary",
                formula=(
                    f"sum(kernel_weight * indicator(listed and discipline equals {discipline}))"
                ),
                unit="entity_equivalent_count",
            )
        )
    definitions.append(
        _spatial_definition(
            name="discipline_shannon_normalized",
            family="multidisciplinary",
            formula="negative_sum(p_log_p)_over_log_4_from_listed_discipline_kernel_masses",
            unit="ratio_0_to_1",
            nullable=True,
            null_semantics="null means no listed discipline mass is available",
            null_reasons=("no_eligible_entity",),
        )
    )

    definitions.extend(
        (
            _spatial_definition(
                name="age_mean_days",
                family="age_duration",
                formula=(
                    "kernel_weighted_mean(causal age_days among listed entities with known age)"
                ),
                unit="days",
                nullable=True,
                null_semantics="null means no listed entity has a causal known age",
                null_reasons=("no_known_age",),
            ),
            _spatial_definition(
                name="age_known_count",
                family="age_duration",
                formula=(
                    "count(listed entities with finite causal age_days and positive kernel weight)"
                ),
                unit="entity_count",
            ),
            _spatial_definition(
                name="known_duration_mean_days",
                family="age_duration",
                formula=(
                    "kernel_weighted_mean(causal known_duration_days among listed entities "
                    "with known duration)"
                ),
                unit="days",
                nullable=True,
                null_semantics="null means no listed entity has a causal known duration",
                null_reasons=("no_known_duration",),
            ),
            _spatial_definition(
                name="known_duration_count",
                family="age_duration",
                formula=(
                    "count(listed entities with finite causal known_duration_days and positive "
                    "kernel weight)"
                ),
                unit="entity_count",
            ),
        )
    )

    spatial_nullable = (
        (
            "mean_distance_km",
            "sum(distance_km * listed kernel_weight) / sum(listed kernel_weight)",
            "km",
            ("no_eligible_entity",),
        ),
        (
            "diffusion_radius_km",
            "square_root(trace(kernel_weighted listed coordinate covariance)) / 1000",
            "km",
            ("no_eligible_entity",),
        ),
        (
            "concentration",
            "clip(1 - diffusion_radius / kernel_support_radius, 0, 1)",
            "ratio_0_to_1",
            ("no_eligible_entity",),
        ),
        (
            "principal_direction_deg",
            "major_eigenvector_direction_modulo_180_degrees",
            "degrees_0_to_180",
            ("no_eligible_entity", "direction_undefined"),
        ),
        (
            "anisotropy",
            "largest_minus_smallest_covariance_eigenvalue_divided_by_trace",
            "ratio_0_to_1",
            ("no_eligible_entity",),
        ),
    )
    for name, formula, unit, reasons in spatial_nullable:
        definitions.append(
            _spatial_definition(
                name=name,
                family="spatial_pattern",
                formula=formula,
                unit=unit,
                nullable=True,
                null_semantics="null means the causal listed spatial support is insufficient",
                null_reasons=reasons,
            )
        )
    return definitions


def _trajectory_definition(
    *,
    name: str,
    windows_weeks: tuple[int, ...],
    formula: str,
    unit: str,
    null_reasons: tuple[str, ...],
) -> FeatureDefinition:
    return _definition(
        name=name,
        source_output_field=name,
        producer="trajectory_v1",
        family="temporal_trajectory",
        formula=formula,
        unit=unit,
        causal_sources=(
            "actual issue snapshots strictly inside the frozen left-open lookback",
            "one causal feature series at the same fixed query-cell kernel and scale",
        ),
        nullable=True,
        null_semantics="null is retained with validity false sample count and an explicit reason",
        null_reasons=null_reasons,
        kernels=TRAJECTORY_KERNELS,
        scales_km=SPATIAL_SCALES_KM,
        windows_weeks=windows_weeks,
        emits_validity_and_sample_count=True,
    )


def _trajectory_definitions() -> list[FeatureDefinition]:
    definitions = [
        _trajectory_definition(
            name=f"slope_{window}w_per_week",
            windows_weeks=(window,),
            formula=(f"ordinary_least_squares_slope_on_actual_elapsed_weeks_in_(T-{window}w,T]"),
            unit="entity_equivalent_count_per_week",
            null_reasons=("insufficient_actual_snapshots", "zero_elapsed_time_variance"),
        )
        for window in TRAJECTORY_WINDOWS_WEEKS
    ]
    definitions.extend(
        (
            _trajectory_definition(
                name="acceleration_4v13_per_week2",
                windows_weeks=(4, 13),
                formula="(slope_4w_per_week - slope_13w_per_week) / 4.5",
                unit="entity_equivalent_count_per_week_squared",
                null_reasons=("insufficient_actual_snapshots",),
            ),
            _trajectory_definition(
                name="surge_z_13w",
                windows_weeks=(13,),
                formula=(
                    "(current - prior_13w_arithmetic_mean) / "
                    "prior_13w_sample_standard_deviation_ddof_1"
                ),
                unit="sample_standard_deviation",
                null_reasons=(
                    "insufficient_actual_snapshots",
                    "zero_or_undefined_baseline_variance",
                    "current_value_unavailable",
                ),
            ),
            _trajectory_definition(
                name="peak_drop_52w",
                windows_weeks=(52,),
                formula="causal_peak_in_(T-52w,T]_including_current - current",
                unit="entity_equivalent_count",
                null_reasons=("insufficient_actual_snapshots", "current_value_unavailable"),
            ),
            _trajectory_definition(
                name="peak_ratio_52w",
                windows_weeks=(52,),
                formula="current / causal_peak_in_(T-52w,T]_including_current",
                unit="ratio_0_to_1",
                null_reasons=(
                    "insufficient_actual_snapshots",
                    "current_value_unavailable",
                    "zero_causal_peak",
                ),
            ),
        )
    )
    return definitions


def _protocol_proxy_definitions() -> list[FeatureDefinition]:
    proxy_source = ("causal report periods and anomaly observations available by the issue time",)
    definitions = [
        _definition(
            name="report_present_reporting_coverage_proxy",
            source_output_field=None,
            producer="protocol_v1",
            family="reporting_coverage_proxy",
            formula="indicator(an actual report period is available at the issue time)",
            unit="boolean",
            arrow_type="bool",
            causal_sources=proxy_source,
            interpretation=REPORTING_PROXY_INTERPRETATION,
        ),
        _definition(
            name="report_row_count_reporting_coverage_proxy",
            source_output_field=None,
            producer="protocol_v1",
            family="reporting_coverage_proxy",
            formula="row_count from the current actual report period",
            unit="reported_row_count",
            arrow_type="int64",
            causal_sources=proxy_source,
            interpretation=REPORTING_PROXY_INTERPRETATION,
        ),
        _definition(
            name="days_since_previous_actual_report_reporting_coverage_proxy",
            source_output_field=None,
            producer="protocol_v1",
            family="reporting_coverage_proxy",
            formula="elapsed calendar days since the previous actual available report",
            unit="days",
            arrow_type="int64",
            causal_sources=proxy_source,
            nullable=True,
            null_semantics="null at the first actual report because no predecessor exists",
            null_reasons=("insufficient_actual_snapshots",),
            interpretation=REPORTING_PROXY_INTERPRETATION,
        ),
    ]
    for window in TRAJECTORY_WINDOWS_WEEKS:
        definitions.append(
            _definition(
                name=f"missing_expected_period_count_{window}w_reporting_coverage_proxy",
                source_output_field=None,
                producer="protocol_v1",
                family="reporting_coverage_proxy",
                formula=f"count(known missing expected periods in (T-{window}w,T])",
                unit="missing_period_count",
                arrow_type="int64",
                causal_sources=proxy_source,
                windows_weeks=(window,),
                interpretation=REPORTING_PROXY_INTERPRETATION,
            )
        )
    return definitions


def _local_coverage_definitions() -> list[FeatureDefinition]:
    causal_sources = (
        "same fixed query cell kernel and scale as the stored spatial value",
        "current report and actual reports available by T inside causal [T-364 days,T]",
        "only reporting identifiers locally supported at that cell and scale",
    )
    trailing_definitions = (
        (
            "trailing_station_count_reporting_coverage_proxy",
            (
                "sum(maximum kernel weight for each distinct reporting station observed in "
                "causal [T-364 days,T])"
            ),
        ),
        (
            "trailing_measurement_count_reporting_coverage_proxy",
            (
                "sum(maximum kernel weight for each distinct reporting measurement observed in "
                "causal [T-364 days,T])"
            ),
        ),
    )
    definitions = [
        _definition(
            name=name,
            source_output_field=name,
            producer="local_coverage_v1",
            family="reporting_coverage_proxy",
            formula=formula,
            unit="kernel_weighted_distinct_reporting_identifier_mass",
            causal_sources=causal_sources,
            kernels=SPATIAL_KERNELS,
            scales_km=SPATIAL_SCALES_KM,
            windows_weeks=(52,),
            interpretation=REPORTING_PROXY_INTERPRETATION,
        )
        for name, formula in trailing_definitions
    ]
    ratio_definitions = (
        (
            "current_to_trailing_station_reporting_coverage_proxy",
            (
                "distinct_reporting_station_count_reporting_coverage_proxy / "
                "trailing_station_count_reporting_coverage_proxy at the same cell kernel scale"
            ),
        ),
        (
            "current_to_trailing_measurement_reporting_coverage_proxy",
            (
                "distinct_reporting_measurement_count_reporting_coverage_proxy / "
                "trailing_measurement_count_reporting_coverage_proxy at the same cell kernel scale"
            ),
        ),
    )
    for name, formula in ratio_definitions:
        definitions.append(
            _definition(
                name=name,
                source_output_field=name,
                producer="local_coverage_v1",
                family="reporting_coverage_proxy",
                formula=formula,
                unit="ratio_0_to_1",
                causal_sources=causal_sources,
                nullable=True,
                null_semantics=(
                    "null affects only this cell kernel scale and means its causal trailing "
                    "reporting reference is empty"
                ),
                null_reasons=("zero_denominator",),
                kernels=SPATIAL_KERNELS,
                scales_km=SPATIAL_SCALES_KM,
                windows_weeks=(52,),
                interpretation=REPORTING_PROXY_INTERPRETATION,
            )
        )
    return definitions


def _build_frozen_feature_dictionary_unchecked() -> FeatureDictionary:
    return FeatureDictionary(
        definitions=tuple(
            [
                *_spatial_definitions(),
                *_trajectory_definitions(),
                *_local_coverage_definitions(),
                *_protocol_proxy_definitions(),
            ]
        )
    )


def build_feature_dictionary() -> FeatureDictionary:
    """Build and public-safety validate the complete frozen stage-3 dictionary."""

    dictionary = _build_frozen_feature_dictionary_unchecked()
    assert_public_safe_feature_dictionary(dictionary)
    return dictionary


def _walk_public_keys(value: object, *, path: str, errors: list[str]) -> None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            if not isinstance(raw_key, str):
                errors.append(f"{path} contains a non-string key")
                continue
            key = raw_key.casefold()
            item_path = f"{path}.{raw_key}"
            if key in _FORBIDDEN_PUBLIC_KEYS:
                errors.append(f"{item_path} is forbidden in the public dictionary")
            _walk_public_keys(item, path=item_path, errors=errors)
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _walk_public_keys(item, path=f"{path}[{index}]", errors=errors)


def public_feature_dictionary_errors(value: Mapping[str, object]) -> tuple[str, ...]:
    """Return public-safety and semantic errors for a dictionary mapping."""

    errors: list[str] = []
    _walk_public_keys(value, path="$", errors=errors)
    public_keys = {key for key in value if isinstance(key, str)}
    if public_keys != _PUBLIC_DICTIONARY_KEYS:
        errors.append("$ contains unknown or missing public dictionary fields")
    if value.get("canonical_json_version") != CANONICAL_JSON_VERSION:
        errors.append("$.canonical_json_version is invalid")
    if value.get("schema_version") != FEATURE_DICTIONARY_SCHEMA_VERSION:
        errors.append("$.schema_version is invalid")
    if value.get("protocol_version") != FEATURE_PROTOCOL_VERSION:
        errors.append("$.protocol_version must be 0.3.0")
    if value.get("semantics") != FEATURE_SEMANTICS:
        errors.append("$.semantics must explicitly state non-probability feature semantics")
    expected_null_reasons = {
        str(code): reason for code, reason in sorted(NULL_REASON_DEFINITIONS.items())
    }
    if value.get("null_reason_definitions") != expected_null_reasons:
        errors.append("$.null_reason_definitions must match the frozen int8 codebook")
    features = value.get("features")
    if not isinstance(features, list):
        errors.append("$.features must be a list")
        return tuple(errors)

    seen_names: set[str] = set()
    seen_storage_columns: set[str] = set()
    for index, raw_feature in enumerate(features):
        path = f"$.features[{index}]"
        if not isinstance(raw_feature, Mapping):
            errors.append(f"{path} must be a mapping")
            continue
        feature_keys = {key for key in raw_feature if isinstance(key, str)}
        if feature_keys != _PUBLIC_FEATURE_KEYS:
            errors.append(f"{path} contains unknown or missing public feature fields")
        name = raw_feature.get("name")
        if not isinstance(name, str) or _NAME_PATTERN.fullmatch(name) is None:
            errors.append(f"{path}.name is invalid")
            continue
        if name in seen_names:
            errors.append(f"{path}.name is duplicated")
        seen_names.add(name)
        producer = raw_feature.get("producer")
        if not isinstance(producer, str) or producer not in _ALLOWED_PRODUCERS:
            errors.append(f"{path}.producer is invalid")
        family = raw_feature.get("family")
        if not isinstance(family, str) or family not in _ALLOWED_FAMILIES:
            errors.append(f"{path}.family is invalid")
        arrow_type = raw_feature.get("arrow_type")
        if not isinstance(arrow_type, str) or arrow_type not in _ALLOWED_ARROW_TYPES:
            errors.append(f"{path}.arrow_type is invalid")
        emits_quality = raw_feature.get("emits_validity_and_sample_count")
        if not isinstance(emits_quality, bool):
            errors.append(f"{path}.emits_validity_and_sample_count must be boolean")
        elif emits_quality is not (producer == "trajectory_v1"):
            errors.append(f"{path}.emits_validity_and_sample_count is inconsistent")

        kernels = raw_feature.get("kernels")
        if not isinstance(kernels, list) or any(not isinstance(item, str) for item in kernels):
            errors.append(f"{path}.kernels must be a string list")
        else:
            expected_kernels: list[str]
            if producer in {"spatial_v1", "local_coverage_v1"}:
                expected_kernels = list(SPATIAL_KERNELS)
            elif producer == "trajectory_v1":
                expected_kernels = list(TRAJECTORY_KERNELS)
            else:
                expected_kernels = []
            if kernels != expected_kernels:
                errors.append(f"{path}.kernels does not match its producer")

        scales = raw_feature.get("scales_km")
        if not isinstance(scales, list) or any(
            isinstance(item, bool) or not isinstance(item, int | float) for item in scales
        ):
            errors.append(f"{path}.scales_km must be a numeric list")
        else:
            expected_scales = (
                list(SPATIAL_SCALES_KM)
                if producer in {"spatial_v1", "trajectory_v1", "local_coverage_v1"}
                else []
            )
            if scales != expected_scales:
                errors.append(f"{path}.scales_km does not match its producer")

        windows = raw_feature.get("windows_weeks")
        if not isinstance(windows, list) or any(
            isinstance(item, bool) or not isinstance(item, int) for item in windows
        ):
            errors.append(f"{path}.windows_weeks must be an integer list")
        elif len(windows) != len(set(windows)) or any(
            item not in TRAJECTORY_WINDOWS_WEEKS for item in windows
        ):
            errors.append(f"{path}.windows_weeks contains an unfrozen window")

        source_output_field = raw_feature.get("source_output_field")
        if producer == "protocol_v1":
            if source_output_field is not None:
                errors.append(f"{path}.source_output_field must be null for protocol features")
        elif producer in _ALLOWED_PRODUCERS and (
            not isinstance(source_output_field, str)
            or _NAME_PATTERN.fullmatch(source_output_field) is None
        ):
            errors.append(f"{path}.source_output_field is invalid")

        causal_sources = raw_feature.get("causal_sources")
        if (
            not isinstance(causal_sources, list)
            or not causal_sources
            or any(
                not isinstance(item, str) or not item or item != item.strip()
                for item in causal_sources
            )
        ):
            errors.append(f"{path}.causal_sources must be non-empty strings")
        for text_field in ("formula", "unit", "null_semantics"):
            text_value = raw_feature.get(text_field)
            if (
                not isinstance(text_value, str)
                or not text_value
                or text_value != text_value.strip()
            ):
                errors.append(f"{path}.{text_field} must be a non-empty trimmed string")

        if producer in {"local_coverage_v1", "protocol_v1"} and family != (
            "reporting_coverage_proxy"
        ):
            errors.append(f"{path}.family is inconsistent with its producer")
        if producer == "trajectory_v1" and family != "temporal_trajectory":
            errors.append(f"{path}.family is inconsistent with its producer")
        interpretation = raw_feature.get("interpretation")
        if interpretation not in _ALLOWED_INTERPRETATIONS:
            errors.append(f"{path}.interpretation is not an approved non-probability statement")
        if family == "reporting_coverage_proxy":
            if not name.endswith("reporting_coverage_proxy"):
                errors.append(f"{path}.name lacks the reporting proxy suffix")
            if interpretation != REPORTING_PROXY_INTERPRETATION:
                errors.append(f"{path}.interpretation overclaims reporting coverage")
        elif interpretation != FEATURE_SEMANTICS:
            errors.append(f"{path}.interpretation must reject absolute probability")

        text_values: list[str] = []
        for field_name in (
            "name",
            "source_output_field",
            "family",
            "formula",
            "unit",
            "null_semantics",
        ):
            item = raw_feature.get(field_name)
            if isinstance(item, str):
                text_values.append(item)
        for field_name in ("causal_sources", "null_reasons"):
            item = raw_feature.get(field_name)
            if isinstance(item, list):
                text_values.extend(value for value in item if isinstance(value, str))
        scientific_text = " ".join(text_values)
        if name == _CROSS_FAULT_DISCIPLINE_NAME:
            if (
                raw_feature.get("source_output_field") != _CROSS_FAULT_DISCIPLINE_NAME
                or raw_feature.get("formula") != _CROSS_FAULT_DISCIPLINE_FORMULA
            ):
                errors.append(f"{path} misuses the cross_fault source discipline exception")
            scientific_text = _without_approved_cross_fault_discipline(scientific_text)
        forbidden = _forbidden_concept(scientific_text)
        if forbidden is not None:
            errors.append(f"{path} contains forbidden concept {forbidden!r}")
        quality_companions = raw_feature.get("quality_companions")
        nullable = raw_feature.get("nullable")
        if not isinstance(quality_companions, Mapping):
            errors.append(f"{path}.quality_companions has an invalid field set")
        elif not isinstance(nullable, bool) or not isinstance(emits_quality, bool):
            errors.append(f"{path}.quality_companions cannot be validated")
        elif dict(quality_companions) != _quality_companion_mapping(
            nullable=nullable,
            emits_quality=emits_quality,
        ):
            errors.append(f"{path}.quality_companions does not match feature semantics")
        storage_columns = raw_feature.get("storage_value_columns")
        if not isinstance(storage_columns, list) or not storage_columns:
            errors.append(f"{path}.storage_value_columns must be a non-empty list")
        else:
            for column in storage_columns:
                if not isinstance(column, str) or _NAME_PATTERN.fullmatch(column) is None:
                    errors.append(f"{path}.storage_value_columns contains an invalid name")
                    continue
                if column in seen_storage_columns:
                    errors.append(f"{path}.storage_value_columns duplicates {column}")
                seen_storage_columns.add(column)
                audited_column = column
                if name == _CROSS_FAULT_DISCIPLINE_NAME and column.endswith(
                    f"__{_CROSS_FAULT_DISCIPLINE_NAME}"
                ):
                    audited_column = _without_approved_cross_fault_discipline(column)
                column_forbidden = _forbidden_concept(audited_column)
                if column_forbidden is not None:
                    errors.append(
                        f"{path}.storage_value_columns contains forbidden concept "
                        f"{column_forbidden!r}"
                    )
                if family == "reporting_coverage_proxy" and not column.endswith(
                    "reporting_coverage_proxy"
                ):
                    errors.append(f"{path}.storage_value_columns has an unmarked reporting proxy")

            expected_storage_columns: tuple[str, ...] | None = None
            if producer in {"spatial_v1", "local_coverage_v1"}:
                expected_storage_columns = tuple(
                    f"{_kernel_storage_prefix(kernel, scale_km)}__{name}"
                    for kernel in SPATIAL_KERNELS
                    for scale_km in SPATIAL_SCALES_KM
                )
            elif producer == "trajectory_v1":
                expected_storage_columns = tuple(
                    (
                        f"{_kernel_storage_prefix('closed_ball', scale_km)}__"
                        f"{base_source_field}__{name}"
                    )
                    for base_source_field in TRAJECTORY_BASE_SOURCE_FIELDS
                    for scale_km in SPATIAL_SCALES_KM
                )
            elif producer == "protocol_v1":
                expected_storage_columns = (name,)
            if expected_storage_columns is not None and storage_columns != list(
                expected_storage_columns
            ):
                errors.append(f"{path}.storage_value_columns does not match its producer")

        null_reasons = raw_feature.get("null_reasons")
        null_reason_codes = raw_feature.get("null_reason_codes")
        if nullable is True and isinstance(null_reasons, list):
            expected_codes = {"valid": NULL_REASON_VALID_CODE}
            for reason in null_reasons:
                if not isinstance(reason, str) or reason not in _NULL_REASON_CODE_BY_NAME:
                    errors.append(f"{path}.null_reasons contains an unfrozen reason")
                    continue
                expected_codes[reason] = _NULL_REASON_CODE_BY_NAME[reason]
            if null_reason_codes != expected_codes:
                errors.append(f"{path}.null_reason_codes does not match the frozen codebook")
        elif nullable is False:
            if null_reasons != [] or null_reason_codes != {}:
                errors.append(f"{path} is non-nullable but declares null reasons")
        else:
            errors.append(f"{path}.nullable or null_reasons is invalid")
    declared_count = value.get("feature_count")
    if declared_count != len(features):
        errors.append("$.feature_count does not match $.features")
    storage_contract = value.get("storage_contract")
    if not isinstance(storage_contract, Mapping):
        errors.append("$.storage_contract must be a mapping")
    else:
        expected_storage_contract = _storage_contract_mapping(len(seen_storage_columns))
        if dict(storage_contract) != expected_storage_contract:
            errors.append("$.storage_contract does not match the frozen contract")
    try:
        observed_canonical = canonical_json_bytes(value)
    except (TypeError, ValueError) as error:
        errors.append(f"$ cannot be canonicalized: {error}")
    else:
        expected_canonical = _build_frozen_feature_dictionary_unchecked().canonical_bytes
        if observed_canonical != expected_canonical:
            errors.append("$ does not match the frozen canonical feature dictionary")
    return tuple(errors)


def assert_public_safe_feature_dictionary(dictionary: FeatureDictionary) -> None:
    """Raise when the dictionary would expose spatial detail or forbidden concepts."""

    errors = public_feature_dictionary_errors(dictionary.as_mapping())
    if errors:
        raise ValueError("unsafe public feature dictionary: " + "; ".join(errors))


def _arrow_type(name: ArrowTypeName) -> pa.DataType:
    if name == "float64":
        return pa.float64()
    if name == "int64":
        return pa.int64()
    if name == "bool":
        return pa.bool_()
    raise AssertionError(f"unreachable Arrow type: {name}")


def _local_storage_field(name: str, type_name: str, nullable: bool) -> pa.Field:
    types: dict[str, pa.DataType] = {
        "date32": pa.date32(),
        "float64": pa.float64(),
        "int16": pa.int16(),
        "int32": pa.int32(),
        "string": pa.string(),
        "timestamp_us_utc": pa.timestamp("us", tz="UTC"),
    }
    try:
        arrow_type = types[type_name]
    except KeyError as error:
        raise AssertionError(f"unreachable local Arrow type: {type_name}") from error
    return pa.field(name, arrow_type, nullable=nullable)


def build_feature_store_schema(
    dictionary: FeatureDictionary | None = None,
) -> pa.Schema:
    """Build the local wide feature-store schema from the frozen dictionary.

    Query-cell identifiers are local storage keys and never enter the public feature
    dictionary.  Every nullable value receives an adjacent reason field.  Trajectory
    values additionally preserve the exact validity mask and actual sample count
    emitted by :mod:`trajectory`.
    """

    resolved = DEFAULT_FEATURE_DICTIONARY if dictionary is None else dictionary
    assert_public_safe_feature_dictionary(resolved)
    fields = [
        _local_storage_field(name, type_name, nullable)
        for name, type_name, nullable in _LOCAL_FEATURE_STORE_FIELDS
    ]
    for definition in resolved.definitions:
        null_reason_metadata = {
            b"seismoflux_null_reason_codes": canonical_json_bytes(
                {str(code): reason for reason, code in definition.null_reason_codes.items()}
            )
        }
        for value_column in definition.storage_value_columns():
            fields.append(
                pa.field(
                    value_column,
                    _arrow_type(definition.arrow_type),
                    nullable=definition.nullable,
                    metadata={
                        b"seismoflux_logical_feature": definition.name.encode("ascii"),
                        b"seismoflux_producer": definition.producer.encode("ascii"),
                    },
                )
            )
            validity_field = definition.validity_field(value_column)
            if validity_field is not None:
                fields.append(pa.field(validity_field, pa.bool_(), nullable=False))
            sample_count_field = definition.sample_count_field(value_column)
            if sample_count_field is not None:
                fields.append(pa.field(sample_count_field, pa.int64(), nullable=False))
            null_reason_code_field = definition.null_reason_code_field(value_column)
            if null_reason_code_field is not None:
                fields.append(
                    pa.field(
                        null_reason_code_field,
                        pa.int8(),
                        nullable=False,
                        metadata=null_reason_metadata,
                    )
                )

    metadata = {
        b"seismoflux_contract": b"0.3.0-anomaly-feature-store",
        b"seismoflux_feature_dictionary_id": resolved.dictionary_id.encode("ascii"),
        b"seismoflux_feature_dictionary_sha256": resolved.sha256.encode("ascii"),
        b"seismoflux_feature_semantics": FEATURE_SEMANTICS.encode("ascii"),
        b"seismoflux_layout": b"one_issue_cell_wide_row",
        b"seismoflux_local_key_schema_sha256": _local_key_schema_sha256().encode("ascii"),
        b"seismoflux_sort_keys": ",".join(FEATURE_STORE_SORT_KEYS).encode("ascii"),
        b"seismoflux_value_column_count": str(len(resolved.storage_value_columns())).encode(
            "ascii"
        ),
    }
    return pa.schema(fields, metadata=metadata)


DEFAULT_FEATURE_DICTIONARY: Final[FeatureDictionary] = build_feature_dictionary()


__all__ = [
    "DEFAULT_FEATURE_DICTIONARY",
    "FEATURE_DICTIONARY_ID_PREFIX",
    "FEATURE_DICTIONARY_SCHEMA_VERSION",
    "FEATURE_PROTOCOL_VERSION",
    "FEATURE_SEMANTICS",
    "FEATURE_STORE_SORT_KEYS",
    "NULL_REASON_VALID_CODE",
    "REPORTING_PROXY_INTERPRETATION",
    "SPATIAL_KERNELS",
    "TRAJECTORY_BASE_SOURCE_FIELDS",
    "TRAJECTORY_KERNELS",
    "FeatureDefinition",
    "FeatureDictionary",
    "assert_public_safe_feature_dictionary",
    "build_feature_dictionary",
    "build_feature_store_schema",
    "public_feature_dictionary_errors",
]
