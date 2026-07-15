from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from typing import cast

import numpy as np
import pyarrow as pa

from seismoflux.features.anomaly.dictionary import (
    DEFAULT_FEATURE_DICTIONARY,
    FEATURE_SEMANTICS,
    REPORTING_PROXY_INTERPRETATION,
    TRAJECTORY_BASE_SOURCE_FIELDS,
    FeatureDictionary,
    build_feature_dictionary,
    build_feature_store_schema,
    public_feature_dictionary_errors,
)
from seismoflux.features.anomaly.nulls import NULL_REASON_DEFINITIONS
from seismoflux.features.anomaly.spatial import (
    SPATIAL_SCALES_KM,
    SpatialEntityArrays,
    compute_spatial_features,
)
from seismoflux.features.anomaly.trajectory import compute_trajectory_features

_FORBIDDEN_SCIENTIFIC_CONCEPT = re.compile(
    r"\b(?:earthquake|epicenter|magnitude|target|fault|mc|score|hit|recall)\b",
    re.IGNORECASE,
)
_FORBIDDEN_PUBLIC_KEYS = {
    "anomaly_id",
    "cell_id",
    "geometry",
    "latitude",
    "longitude",
    "observation_id",
    "source_file",
    "source_row",
    "source_sheet",
    "station_id",
    "wkb",
    "wkt",
    "x_m",
    "xy_m",
    "y_m",
}


def _public_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str):
                keys.add(key.casefold())
            keys.update(_public_keys(item))
    elif isinstance(value, list | tuple):
        for item in value:
            keys.update(_public_keys(item))
    return keys


def _one_spatial_entity() -> SpatialEntityArrays:
    false = np.asarray([False], dtype=np.bool_)
    return SpatialEntityArrays(
        xy_m=np.asarray([[0.0, 0.0]], dtype=np.float64),
        listed=np.asarray([True], dtype=np.bool_),
        source_new=false,
        first_seen=false,
        explicit_end=false,
        not_continued=false,
        relisted=false,
        right_censored=false,
        reliability_high=np.asarray([True], dtype=np.bool_),
        reliability_cautious=false,
        station_id=np.asarray(["station-a"], dtype=object),
        measurement_id=np.asarray(["measurement-a"], dtype=object),
        discipline_code=np.asarray([0], dtype=np.int64),
        age_days=np.asarray([1.0], dtype=np.float64),
        known_duration_days=np.asarray([np.nan], dtype=np.float64),
    )


def test_dictionary_is_canonical_and_order_independent() -> None:
    first = build_feature_dictionary()
    second = FeatureDictionary(definitions=tuple(reversed(first.definitions)))

    assert first.canonical_bytes == second.canonical_bytes
    assert first.sha256 == second.sha256
    assert first.dictionary_id == second.dictionary_id
    assert first.sha256 == build_feature_dictionary().sha256
    assert tuple(item.name for item in first.definitions) == tuple(
        sorted(item.name for item in first.definitions)
    )


def test_dictionary_covers_every_spatial_kernel_output_exactly_once() -> None:
    result = compute_spatial_features(
        np.asarray([[0.0, 0.0]], dtype=np.float64),
        _one_spatial_entity(),
    )
    source_fields = DEFAULT_FEATURE_DICTIONARY.source_field_map("spatial_v1")

    assert set(source_fields) == set(result.radius_features)
    assert set(source_fields) == set(result.gaussian_features)
    assert len(source_fields) == len(set(source_fields.values()))

    definitions = DEFAULT_FEATURE_DICTIONARY.by_name()
    for public_name in source_fields.values():
        definition = definitions[public_name]
        assert definition.kernels == ("closed_ball", "gaussian")
        assert definition.scales_km == (50.0, 100.0, 200.0, 300.0, 500.0)
        assert definition.windows_weeks == ()
        assert definition.formula
        assert definition.unit
        assert definition.arrow_type == "float64"
        assert definition.causal_sources
        assert definition.null_semantics
        expected_columns = {
            f"{kernel}_{int(scale_km)}km__{public_name}"
            for kernel in ("radius", "gaussian")
            for scale_km in SPATIAL_SCALES_KM
        }
        assert set(definition.storage_value_columns()) == expected_columns


def test_dictionary_covers_every_trajectory_output_and_quality_companion() -> None:
    issue_times = np.asarray(
        ["2022-01-01", "2022-01-08", "2022-01-15", "2022-01-22"],
        dtype="datetime64[ns]",
    )
    result = compute_trajectory_features(
        issue_times,
        np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
    )
    source_fields = DEFAULT_FEATURE_DICTIONARY.source_field_map("trajectory_v1")
    schema = build_feature_store_schema()

    assert set(source_fields) == set(result.features)
    assert set(source_fields) == set(result.valid_masks)
    assert set(source_fields) == set(result.sample_counts)
    trajectory_columns: set[str] = set()
    for public_name in source_fields.values():
        definition = DEFAULT_FEATURE_DICTIONARY.by_name()[public_name]
        assert definition.windows_weeks
        assert definition.formula
        assert definition.unit
        assert definition.causal_sources
        assert definition.null_semantics
        assert definition.null_reasons
        value_columns = definition.storage_value_columns()
        assert len(value_columns) == 25
        trajectory_columns.update(value_columns)
        for value_column in value_columns:
            assert value_column.startswith("radius_")
            assert "gaussian_" not in value_column
            assert schema.field(value_column).type == pa.float64()
            assert schema.field(f"{value_column}__valid").type == pa.bool_()
            assert not schema.field(f"{value_column}__valid").nullable
            assert schema.field(f"{value_column}__sample_count").type == pa.int64()
            assert not schema.field(f"{value_column}__sample_count").nullable
            reason_field = schema.field(f"{value_column}__null_reason_code")
            assert reason_field.type == pa.int8()
            assert not reason_field.nullable

    expected_columns = {
        f"radius_{int(scale_km)}km__{base}__{trajectory_feature}"
        for base in TRAJECTORY_BASE_SOURCE_FIELDS
        for scale_km in SPATIAL_SCALES_KM
        for trajectory_feature in source_fields.values()
    }
    assert trajectory_columns == expected_columns
    assert len(trajectory_columns) == 25 * 9


def test_reporting_proxies_are_explicit_and_never_probability_claims() -> None:
    for definition in DEFAULT_FEATURE_DICTIONARY.definitions:
        if definition.family == "reporting_coverage_proxy":
            assert definition.name.endswith("reporting_coverage_proxy")
            assert definition.interpretation == REPORTING_PROXY_INTERPRETATION
            assert all(
                column.endswith("reporting_coverage_proxy")
                for column in definition.storage_value_columns()
            )
        else:
            assert definition.interpretation == FEATURE_SEMANTICS


def test_trailing_reporting_proxies_are_local_to_each_kernel_scale() -> None:
    storage_contract = cast(
        dict[str, object], DEFAULT_FEATURE_DICTIONARY.as_mapping()["storage_contract"]
    )
    assert storage_contract["trajectory_column_template"] == (
        "radius_{scale}km__{base}__{trajectory_feature}"
    )
    assert storage_contract["local_reporting_coverage_scope"] == (
        "same_query_cell_kernel_scale_only"
    )
    assert storage_contract["local_reporting_coverage_window_days"] == 364
    local_definitions = {
        definition.name: definition
        for definition in DEFAULT_FEATURE_DICTIONARY.definitions
        if definition.producer == "local_coverage_v1"
    }
    assert set(local_definitions) == {
        "current_to_trailing_measurement_reporting_coverage_proxy",
        "current_to_trailing_station_reporting_coverage_proxy",
        "trailing_measurement_count_reporting_coverage_proxy",
        "trailing_station_count_reporting_coverage_proxy",
    }
    for definition in local_definitions.values():
        assert definition.kernels == ("closed_ball", "gaussian")
        assert definition.scales_km == SPATIAL_SCALES_KM
        assert definition.windows_weeks == (52,)
        assert len(definition.storage_value_columns()) == 10
        assert all(
            column.startswith(("radius_", "gaussian_"))
            and column.endswith("reporting_coverage_proxy")
            for column in definition.storage_value_columns()
        )
        assert any(
            "same fixed query cell kernel and scale" in source
            for source in definition.causal_sources
        )

    protocol_names = {
        definition.name
        for definition in DEFAULT_FEATURE_DICTIONARY.definitions
        if definition.producer == "protocol_v1"
    }
    assert not any("trailing_station" in name for name in protocol_names)
    assert not any("trailing_measurement" in name for name in protocol_names)


def test_feature_store_schema_is_dictionary_driven_and_preserves_null_reasons() -> None:
    schema = build_feature_store_schema(DEFAULT_FEATURE_DICTIONARY)
    metadata = schema.metadata

    assert metadata is not None
    assert metadata[b"seismoflux_feature_dictionary_id"].decode() == (
        DEFAULT_FEATURE_DICTIONARY.dictionary_id
    )
    assert metadata[b"seismoflux_feature_dictionary_sha256"].decode() == (
        DEFAULT_FEATURE_DICTIONARY.sha256
    )
    assert metadata[b"seismoflux_feature_semantics"].decode() == FEATURE_SEMANTICS
    assert metadata[b"seismoflux_layout"] == b"one_issue_cell_wide_row"
    assert metadata[b"seismoflux_value_column_count"] == b"783"
    assert len(schema) == 1637
    assert schema.field("issue_time_utc").type == pa.timestamp("us", tz="UTC")
    assert schema.field("issue_report_id").type == pa.string()
    assert schema.field("state_snapshot_id").type == pa.string()
    assert schema.field("lineage_digest").type == pa.string()
    assert schema.field("feature_dictionary_sha256").type == pa.string()
    assert schema.field("equal_area_crs").type == pa.string()
    assert schema.field("cell_size_km").type == pa.float64()
    assert schema.field("cell_id").type == pa.string()
    assert schema.field("cell_row").type == pa.int32()
    assert schema.field("cell_column").type == pa.int32()
    assert schema.field("query_x_m").type == pa.float64()
    assert schema.field("query_y_m").type == pa.float64()
    assert schema.field("clipped_area_km2").type == pa.float64()
    assert schema.field("radius_50km__listed_count").type == pa.float64()
    assert not schema.field("radius_50km__listed_count").nullable
    assert schema.field("report_present_reporting_coverage_proxy").type == pa.bool_()
    assert schema.field("report_row_count_reporting_coverage_proxy").type == pa.int64()

    for definition in DEFAULT_FEATURE_DICTIONARY.definitions:
        for value_column in definition.storage_value_columns():
            field = schema.field(value_column)
            assert field.nullable is definition.nullable
            assert field.metadata is not None
            assert field.metadata[b"seismoflux_logical_feature"].decode() == definition.name
            reason_column = f"{value_column}__null_reason_code"
            if definition.nullable:
                assert definition.null_reasons
                reason_field = schema.field(reason_column)
                assert reason_field.type == pa.int8()
                assert not reason_field.nullable
                assert reason_field.metadata is not None
            else:
                assert definition.null_reasons == ()
                assert reason_column not in schema.names

    spatial_count = len(DEFAULT_FEATURE_DICTIONARY.source_field_map("spatial_v1")) * 2 * 5
    trajectory_count = len(DEFAULT_FEATURE_DICTIONARY.source_field_map("trajectory_v1")) * 25
    protocol_count = sum(
        len(definition.storage_value_columns())
        for definition in DEFAULT_FEATURE_DICTIONARY.definitions
        if definition.producer == "protocol_v1"
    )
    local_coverage_count = sum(
        len(definition.storage_value_columns())
        for definition in DEFAULT_FEATURE_DICTIONARY.definitions
        if definition.producer == "local_coverage_v1"
    )
    assert spatial_count == 510
    assert trajectory_count == 225
    assert protocol_count == 8
    assert local_coverage_count == 40
    assert len(DEFAULT_FEATURE_DICTIONARY.storage_value_columns()) == (
        spatial_count + trajectory_count + protocol_count + local_coverage_count
    )


def test_public_dictionary_excludes_spatial_keys_and_forbidden_scientific_inputs() -> None:
    mapping = DEFAULT_FEATURE_DICTIONARY.as_mapping()

    assert public_feature_dictionary_errors(mapping) == ()
    assert mapping["null_reason_definitions"] == {
        str(code): reason for code, reason in sorted(NULL_REASON_DEFINITIONS.items())
    }
    assert _public_keys(mapping).isdisjoint(_FORBIDDEN_PUBLIC_KEYS)
    for definition in DEFAULT_FEATURE_DICTIONARY.definitions:
        scientific_text = " ".join(
            (
                definition.name,
                definition.source_output_field or "",
                definition.formula,
                *definition.causal_sources,
            )
        )
        assert _FORBIDDEN_SCIENTIFIC_CONCEPT.search(scientific_text) is None


def test_public_safety_validator_rejects_forbidden_keys_and_concepts() -> None:
    mapping = copy.deepcopy(DEFAULT_FEATURE_DICTIONARY.as_mapping())
    features = cast(list[dict[str, object]], mapping["features"])
    features[0]["formula"] = "target score from an earthquake"
    features[0]["station_id"] = "private-location-key"
    features[0]["notes"] = "unreviewed scientific detail"
    storage_columns = cast(list[str], features[0]["storage_value_columns"])
    storage_columns[0] = "radius_50km__target_score"

    errors = public_feature_dictionary_errors(mapping)

    assert any("forbidden concept" in error for error in errors)
    assert any("storage_value_columns contains forbidden concept" in error for error in errors)
    assert any("station_id" in error for error in errors)
    assert any("unknown or missing public feature fields" in error for error in errors)


def test_public_safety_validator_blocks_aliases_and_limits_cross_fault_exception() -> None:
    for unsafe_formula in (
        "m_c completeness threshold input",
        "cross fault geological input",
    ):
        mapping = copy.deepcopy(DEFAULT_FEATURE_DICTIONARY.as_mapping())
        features = cast(list[dict[str, object]], mapping["features"])
        features[0]["formula"] = unsafe_formula

        assert public_feature_dictionary_errors(mapping)

    mapping = copy.deepcopy(DEFAULT_FEATURE_DICTIONARY.as_mapping())
    features = cast(list[dict[str, object]], mapping["features"])
    cross_fault = next(
        feature for feature in features if feature["name"] == "discipline_cross_fault_count"
    )
    cross_fault["formula"] = "cross_fault geological input"

    errors = public_feature_dictionary_errors(mapping)

    assert any("cross_fault source discipline exception" in error for error in errors)
    assert any("forbidden concept 'fault'" in error for error in errors)


def test_public_safety_validator_freezes_producer_types_kernels_and_storage_contract() -> None:
    feature_mutations: tuple[tuple[str, object], ...] = (
        ("producer", "earthquake_target_v1"),
        ("kernels", ["earthquake"]),
        ("arrow_type", "target_score"),
    )
    for field_name, unsafe_value in feature_mutations:
        mapping = copy.deepcopy(DEFAULT_FEATURE_DICTIONARY.as_mapping())
        features = cast(list[dict[str, object]], mapping["features"])
        features[0][field_name] = unsafe_value

        assert public_feature_dictionary_errors(mapping)

    mapping = copy.deepcopy(DEFAULT_FEATURE_DICTIONARY.as_mapping())
    features = cast(list[dict[str, object]], mapping["features"])
    companions = cast(dict[str, object], features[0]["quality_companions"])
    companions["validity_type"] = "earthquake target"

    assert public_feature_dictionary_errors(mapping)

    storage_mutations: tuple[tuple[str, object], ...] = (
        ("spatial_column_template", "earthquake target coordinates"),
        ("trajectory_base_source_fields", ["target_score"]),
        ("trajectory_kernel", "earthquake"),
    )
    for field_name, unsafe_value in storage_mutations:
        mapping = copy.deepcopy(DEFAULT_FEATURE_DICTIONARY.as_mapping())
        storage = cast(dict[str, object], mapping["storage_contract"])
        storage[field_name] = unsafe_value

        assert public_feature_dictionary_errors(mapping)


def test_public_safety_validator_rejects_safe_but_noncanonical_semantic_changes() -> None:
    mapping = copy.deepcopy(DEFAULT_FEATURE_DICTIONARY.as_mapping())
    features = cast(list[dict[str, object]], mapping["features"])
    features[0]["formula"] = "causal count from the same approved anomaly source"

    errors = public_feature_dictionary_errors(mapping)

    assert any("frozen canonical feature dictionary" in error for error in errors)
