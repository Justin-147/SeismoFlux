"""Frozen Arrow schemas and sort orders for the stage-1 data contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa

from seismoflux.data.common import write_json_atomic

STRING_LIST = pa.list_(pa.field("element", pa.string()))
UTC_TIMESTAMP = pa.timestamp("us", tz="UTC")
LOCAL_TIMESTAMP = pa.timestamp("us", tz="+08:00")


@dataclass(frozen=True, slots=True)
class DatasetContract:
    schema: pa.Schema
    sort_keys: tuple[str, ...]
    description: str


def _schema(fields: list[tuple[str, pa.DataType, bool]]) -> pa.Schema:
    return pa.schema(
        [pa.field(name, data_type, nullable=nullable) for name, data_type, nullable in fields]
    )


CONTRACTS: dict[str, DatasetContract] = {
    "anomaly_observation": DatasetContract(
        _schema(
            [
                ("observation_id", pa.string(), False),
                ("anomaly_id", pa.string(), False),
                ("station_id", pa.string(), False),
                ("report_date", pa.date32(), False),
                ("raw_row_report_date", pa.date32(), False),
                ("available_at", UTC_TIMESTAMP, False),
                ("unit", pa.string(), False),
                ("station_name", pa.string(), True),
                ("longitude", pa.float64(), True),
                ("latitude", pa.float64(), True),
                ("discipline", pa.string(), False),
                ("measurement", pa.string(), True),
                ("instrument_or_method", pa.string(), True),
                ("observation_period", pa.string(), True),
                ("forecast_efficacy", pa.string(), True),
                ("trend_time", pa.date32(), True),
                ("start_time", LOCAL_TIMESTAMP, False),
                ("reported_end_time", LOCAL_TIMESTAMP, True),
                ("end_flag", pa.string(), False),
                ("report_state", pa.string(), False),
                ("anomaly_type", pa.string(), True),
                ("anomaly_description", pa.string(), True),
                ("annual_anomaly", pa.string(), True),
                ("r_value", pa.string(), True),
                ("r0_value", pa.string(), True),
                ("term_class", pa.string(), True),
                ("is_listed", pa.bool_(), False),
                ("right_censored", pa.bool_(), False),
                ("identity_complete", pa.bool_(), False),
                ("reliability_flags", STRING_LIST, False),
                ("source_id", pa.string(), False),
                ("source_file", pa.string(), False),
                ("source_sha256", pa.string(), False),
                ("source_sheet", pa.string(), False),
                ("source_row", pa.int64(), False),
                ("source_serial_number", pa.int64(), False),
                ("source_period_year", pa.int64(), False),
                ("source_period", pa.int64(), False),
            ]
        ),
        ("report_date", "source_file", "source_sheet", "source_row", "observation_id"),
        "One immutable observation as listed in one weekly anomaly report.",
    ),
    "anomaly_entity_audit": DatasetContract(
        _schema(
            [
                ("audit_id", pa.string(), False),
                ("audit_type", pa.string(), False),
                ("status", pa.string(), False),
                ("source_file", pa.string(), True),
                ("source_sheet", pa.string(), True),
                ("report_date", pa.date32(), True),
                ("anomaly_ids", STRING_LIST, False),
                ("observation_ids", STRING_LIST, False),
                ("reliability_flags", STRING_LIST, False),
                ("previous_reported_end_time", LOCAL_TIMESTAMP, True),
                ("current_reported_end_time", LOCAL_TIMESTAMP, True),
            ]
        ),
        ("audit_type", "audit_id"),
        "Non-destructive anomaly identity, duplicate, and end-time review records.",
    ),
    "anomaly_report_period": DatasetContract(
        _schema(
            [
                ("report_id", pa.string(), False),
                ("source_id", pa.string(), False),
                ("source_file", pa.string(), False),
                ("source_sha256", pa.string(), False),
                ("report_year", pa.int64(), False),
                ("report_period", pa.int64(), False),
                ("report_date", pa.date32(), False),
                ("available_at", UTC_TIMESTAMP, False),
                ("row_count", pa.int64(), False),
                ("row_report_date_mismatch_count", pa.int64(), False),
                ("row_report_date_before_count", pa.int64(), False),
                ("row_report_date_after_count", pa.int64(), False),
                ("deformation_row_count", pa.int64(), False),
                ("fluid_row_count", pa.int64(), False),
                ("electromagnetic_row_count", pa.int64(), False),
                ("cross_fault_row_count", pa.int64(), False),
            ]
        ),
        ("report_date", "source_file"),
        "Observed weekly report periods; missing periods are never treated as zero anomalies.",
    ),
    "earthquake_event": DatasetContract(
        _schema(
            [
                ("event_id", pa.string(), False),
                ("origin_time_utc", UTC_TIMESTAMP, False),
                ("available_at", UTC_TIMESTAMP, False),
                ("origin_time_local", LOCAL_TIMESTAMP, False),
                ("longitude", pa.float64(), False),
                ("latitude", pa.float64(), False),
                ("depth_km", pa.float64(), True),
                ("magnitude", pa.float64(), False),
                ("magnitude_type", pa.string(), True),
                ("place", pa.string(), True),
                ("catalog_sources", STRING_LIST, False),
                ("inside_study_area", pa.bool_(), False),
                ("dedup_confidence", pa.string(), False),
                ("anchor_source_record_id", pa.string(), False),
                ("quality_flags", STRING_LIST, False),
            ]
        ),
        ("origin_time_utc", "event_id"),
        "Canonical physical events after configured, auditable source deduplication.",
    ),
    "earthquake_source_record": DatasetContract(
        _schema(
            [
                ("source_record_id", pa.string(), False),
                ("duplicate_group_id", pa.string(), False),
                ("source_id", pa.string(), False),
                ("source_file", pa.string(), False),
                ("source_file_sha256", pa.string(), False),
                ("source_row", pa.int64(), False),
                ("raw_record_sha256", pa.string(), False),
                ("origin_time_raw", pa.string(), False),
                ("origin_time_local", LOCAL_TIMESTAMP, False),
                ("origin_time_utc", UTC_TIMESTAMP, False),
                ("available_at", UTC_TIMESTAMP, False),
                ("longitude_raw", pa.string(), False),
                ("latitude_raw", pa.string(), False),
                ("depth_raw", pa.string(), False),
                ("magnitude_raw", pa.string(), False),
                ("longitude", pa.float64(), False),
                ("latitude", pa.float64(), False),
                ("depth_km", pa.float64(), True),
                ("magnitude", pa.float64(), False),
                ("magnitude_type", pa.string(), True),
                ("place_raw", pa.string(), True),
                ("place", pa.string(), True),
                ("fixed_field_raw", pa.string(), True),
                ("inside_study_area", pa.bool_(), False),
                ("normalization_flags", STRING_LIST, False),
            ]
        ),
        ("source_id", "source_row", "source_record_id"),
        "Every raw catalog row retained with normalization and source lineage.",
    ),
    "earthquake_dedup_candidate": DatasetContract(
        _schema(
            [
                ("candidate_id", pa.string(), False),
                ("m3_group_id", pa.string(), False),
                ("m5_group_id", pa.string(), False),
                ("m3_source_record_ids", STRING_LIST, False),
                ("m5_source_record_ids", STRING_LIST, False),
                ("time_delta_seconds", pa.float64(), False),
                ("distance_km", pa.float64(), False),
                ("magnitude_delta", pa.float64(), False),
                ("exact_match", pa.bool_(), False),
                ("auto_eligible", pa.bool_(), False),
                ("m3_auto_degree", pa.int64(), False),
                ("m5_auto_degree", pa.int64(), False),
                ("decision", pa.string(), False),
                ("merged_event_id", pa.string(), True),
            ]
        ),
        ("time_delta_seconds", "distance_km", "magnitude_delta", "candidate_id"),
        "Cross-catalog duplicate candidates and their non-transitive decisions.",
    ),
    "fault_point_raw": DatasetContract(
        _schema(
            [
                ("fault_point_id", pa.string(), False),
                ("fault_id", pa.int64(), False),
                ("segment_number", pa.int64(), False),
                ("point_number", pa.int64(), False),
                ("longitude", pa.float64(), False),
                ("latitude", pa.float64(), False),
                ("is_new_in_source_version", pa.bool_(), False),
                ("fault_name_raw", pa.string(), False),
                ("segment_name_raw", pa.string(), False),
                ("source_file", pa.string(), False),
                ("source_row", pa.int64(), False),
                ("source_available_at", UTC_TIMESTAMP, False),
                ("historical_model_eligible", pa.bool_(), False),
                ("quality_flags", STRING_LIST, False),
            ]
        ),
        ("fault_id", "segment_number", "point_number"),
        "Original simplified-fault points without destructive coordinate cleaning.",
    ),
    "fault_segment": DatasetContract(
        _schema(
            [
                ("fault_segment_id", pa.string(), False),
                ("fault_id", pa.int64(), False),
                ("segment_number", pa.int64(), False),
                ("fault_name_raw", pa.string(), False),
                ("segment_name_raw", pa.string(), False),
                ("simplified_geometry_wkb", pa.binary(), False),
                ("raw_point_count", pa.int64(), False),
                ("geometry_point_count", pa.int64(), False),
                ("is_new_in_source_version", pa.bool_(), False),
                ("strike_deg", pa.float64(), False),
                ("dip_deg", pa.float64(), False),
                ("slip_angle_deg", pa.float64(), False),
                ("geologic_total_rate_mm_per_year", pa.float64(), True),
                ("geologic_strike_slip_rate_mm_per_year", pa.float64(), True),
                ("geologic_dip_slip_rate_mm_per_year", pa.float64(), True),
                ("geodetic_total_rate_mm_per_year", pa.float64(), True),
                ("geodetic_strike_slip_rate_mm_per_year", pa.float64(), True),
                ("geodetic_strike_slip_error_mm_per_year", pa.float64(), True),
                ("geodetic_dip_slip_rate_mm_per_year", pa.float64(), True),
                ("geodetic_dip_slip_error_mm_per_year", pa.float64(), True),
                ("recurrence_period_years", pa.float64(), True),
                ("recurrence_reliability", pa.int64(), True),
                ("last_strong_earthquake_year_raw", pa.int64(), True),
                ("last_strong_earthquake_reliability", pa.int64(), True),
                ("elapsed_ratio_at_snapshot", pa.float64(), True),
                ("rupture_gap_weight", pa.float64(), False),
                ("locking_weight", pa.float64(), False),
                ("microseismic_sparsity_weight", pa.float64(), False),
                ("coulomb_stress_weight", pa.float64(), False),
                ("long_term_hazard_score", pa.float64(), False),
                ("missing_fields", STRING_LIST, False),
                ("geometry_flags", STRING_LIST, False),
                ("attribute_source_file", pa.string(), False),
                ("hazard_source_file", pa.string(), False),
                ("source_available_at", UTC_TIMESTAMP, False),
                ("historical_model_eligible", pa.bool_(), False),
                ("true_trace_id", pa.string(), True),
                ("true_trace_mapping_status", pa.string(), False),
                ("true_trace_match_confidence", pa.string(), False),
            ]
        ),
        ("fault_id", "segment_number"),
        "Fault geometry, keyed attributes, and long-term scores; never row-position joined.",
    ),
    "fault_trace": DatasetContract(
        _schema(
            [
                ("trace_id", pa.string(), False),
                ("source_segment_number", pa.int64(), False),
                ("trace_name_raw", pa.string(), True),
                ("trace_name_normalized", pa.string(), True),
                ("geometry_wkb", pa.binary(), True),
                ("raw_point_count", pa.int64(), False),
                ("geometry_point_count", pa.int64(), False),
                ("is_closed", pa.bool_(), False),
                ("usable_for_geometry", pa.bool_(), False),
                ("geometry_hash", pa.string(), False),
                ("duplicate_group_id", pa.string(), True),
                ("duplicate_geometry", pa.bool_(), False),
                ("duplicate_name_conflict", pa.bool_(), False),
                ("quality_flags", STRING_LIST, False),
                ("source_file", pa.string(), False),
                ("delimiter_source_line", pa.int64(), False),
                ("source_available_at", UTC_TIMESTAMP, False),
                ("historical_model_eligible", pa.bool_(), False),
            ]
        ),
        ("source_segment_number", "trace_id"),
        "True fault-trace segments retained separately from simplified attributed faults.",
    ),
    "basemap_feature": DatasetContract(
        _schema(
            [
                ("basemap_feature_id", pa.string(), False),
                ("role", pa.string(), False),
                ("source_segment_number", pa.int64(), False),
                ("geometry_wkb", pa.binary(), False),
                ("raw_point_count", pa.int64(), False),
                ("geometry_point_count", pa.int64(), False),
                ("is_closed", pa.bool_(), False),
                ("geometry_hash", pa.string(), False),
                ("duplicate_group_id", pa.string(), True),
                ("duplicate_geometry", pa.bool_(), False),
                ("quality_flags", STRING_LIST, False),
                ("source_file", pa.string(), False),
                ("delimiter_source_line", pa.int64(), False),
                ("source_comments", STRING_LIST, False),
                ("source_available_at", UTC_TIMESTAMP, False),
                ("model_feature_eligible", pa.bool_(), False),
            ]
        ),
        ("role", "source_file", "source_segment_number", "basemap_feature_id"),
        "Visualization and boundary linework kept separate from fault attributes.",
    ),
    "fault_trace_crosswalk_audit": DatasetContract(
        _schema(
            [
                ("crosswalk_candidate_id", pa.string(), False),
                ("fault_segment_id", pa.string(), False),
                ("trace_id", pa.string(), False),
                ("fault_name_raw", pa.string(), False),
                ("trace_name_raw", pa.string(), False),
                ("name_match_method", pa.string(), False),
                ("status", pa.string(), False),
                ("match_confidence", pa.string(), False),
                ("trace_duplicate_group_id", pa.string(), True),
                ("trace_duplicate_name_conflict", pa.bool_(), False),
                ("historical_model_eligible", pa.bool_(), False),
            ]
        ),
        ("fault_segment_id", "trace_id", "crosswalk_candidate_id"),
        "Unreviewed stage-1 name candidates; spatial acceptance is deferred to stage 5.",
    ),
}


def contract_document(name: str, contract: DatasetContract) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "contract_version": "0.1.0",
        "dataset": name,
        "description": contract.description,
        "sort_keys": list(contract.sort_keys),
        "fields": [
            {"name": field.name, "type": str(field.type), "nullable": field.nullable}
            for field in contract.schema
        ],
    }


def write_contract_documents(output_root: Path) -> dict[str, str]:
    """Write every contract deterministically and return dataset-to-path names."""

    written: dict[str, str] = {}
    for name, contract in sorted(CONTRACTS.items()):
        path = output_root / f"{name}.json"
        write_json_atomic(path, contract_document(name, contract))
        written[name] = path.name
    return written
