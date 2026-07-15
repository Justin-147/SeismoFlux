"""Independent Arrow contract for stage-3 dynamic-anomaly state history."""

from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa

from seismoflux.features.anomaly.config import ANOMALY_HISTORY_CONTRACT_VERSION

STRING_LIST = pa.list_(pa.field("element", pa.string()))
UTC_TIMESTAMP = pa.timestamp("us", tz="UTC")
LOCAL_TIMESTAMP = pa.timestamp("us", tz="+08:00")


@dataclass(frozen=True, slots=True)
class Stage3DatasetContract:
    """One stage-3 dataset schema without mutating the frozen stage-1 registry."""

    name: str
    version: str
    schema: pa.Schema
    sort_keys: tuple[str, ...]
    description: str


ANOMALY_STATE_HISTORY_CONTRACT = Stage3DatasetContract(
    name="anomaly_state_history",
    version=ANOMALY_HISTORY_CONTRACT_VERSION,
    schema=pa.schema(
        [
            pa.field("state_id", pa.string(), nullable=False),
            pa.field("contract_version", pa.string(), nullable=False),
            pa.field("state_row_kind", pa.string(), nullable=False),
            pa.field("issue_time_utc", UTC_TIMESTAMP, nullable=False),
            pa.field("issue_report_id", pa.string(), nullable=False),
            pa.field("issue_report_date", pa.date32(), nullable=False),
            pa.field("issue_report_year", pa.int64(), nullable=False),
            pa.field("issue_report_period", pa.int64(), nullable=False),
            pa.field("row_count", pa.int64(), nullable=False),
            pa.field("row_report_date_mismatch_count", pa.int64(), nullable=False),
            pa.field("row_report_date_before_count", pa.int64(), nullable=False),
            pa.field("row_report_date_after_count", pa.int64(), nullable=False),
            pa.field("deformation_row_count", pa.int64(), nullable=False),
            pa.field("fluid_row_count", pa.int64(), nullable=False),
            pa.field("electromagnetic_row_count", pa.int64(), nullable=False),
            pa.field("cross_fault_row_count", pa.int64(), nullable=False),
            pa.field("previous_issue_report_id", pa.string(), nullable=True),
            pa.field("previous_period_consecutive", pa.bool_(), nullable=False),
            pa.field("anomaly_id", pa.string(), nullable=False),
            pa.field("identity_complete", pa.bool_(), nullable=False),
            pa.field("entity_scope", pa.string(), nullable=False),
            pa.field("station_id", pa.string(), nullable=True),
            pa.field("station_id_null_reason", pa.string(), nullable=True),
            pa.field("longitude", pa.float64(), nullable=True),
            pa.field("longitude_null_reason", pa.string(), nullable=True),
            pa.field("latitude", pa.float64(), nullable=True),
            pa.field("latitude_null_reason", pa.string(), nullable=True),
            pa.field("discipline", pa.string(), nullable=True),
            pa.field("discipline_null_reason", pa.string(), nullable=True),
            pa.field("measurement", pa.string(), nullable=True),
            pa.field("measurement_null_reason", pa.string(), nullable=True),
            pa.field("start_time", LOCAL_TIMESTAMP, nullable=True),
            pa.field("start_time_null_reason", pa.string(), nullable=True),
            pa.field("spatial_eligible", pa.bool_(), nullable=False),
            pa.field("spatial_exclusion_reason", pa.string(), nullable=True),
            pa.field("current_reporting_station_ids", STRING_LIST, nullable=False),
            pa.field("current_reporting_disciplines", STRING_LIST, nullable=False),
            pa.field("current_reporting_measurements", STRING_LIST, nullable=False),
            pa.field("known_station_ids", STRING_LIST, nullable=False),
            pa.field("known_disciplines", STRING_LIST, nullable=False),
            pa.field("known_measurements", STRING_LIST, nullable=False),
            pa.field("current_report_listed", pa.bool_(), nullable=False),
            pa.field("source_new", pa.bool_(), nullable=False),
            pa.field("source_continued", pa.bool_(), nullable=False),
            pa.field("source_cancelled", pa.bool_(), nullable=False),
            pa.field("current_source_report_states", STRING_LIST, nullable=False),
            pa.field("latest_source_report_states", STRING_LIST, nullable=False),
            pa.field("system_first_seen", pa.bool_(), nullable=False),
            pa.field("system_not_continued", pa.bool_(), nullable=False),
            pa.field("system_relisted", pa.bool_(), nullable=False),
            pa.field("left_truncated", pa.bool_(), nullable=False),
            pa.field("late_entry_or_gap_unknown", pa.bool_(), nullable=False),
            pa.field("explicit_end_known", pa.bool_(), nullable=False),
            pa.field("right_censored", pa.bool_(), nullable=False),
            pa.field("reported_end_time", LOCAL_TIMESTAMP, nullable=True),
            pa.field("reported_end_time_null_reason", pa.string(), nullable=True),
            pa.field("age_days", pa.float64(), nullable=True),
            pa.field("age_days_null_reason", pa.string(), nullable=True),
            pa.field("known_duration_days", pa.float64(), nullable=True),
            pa.field("known_duration_days_null_reason", pa.string(), nullable=True),
            pa.field("reliability_flags", STRING_LIST, nullable=False),
            pa.field("reliability_grade", pa.string(), nullable=False),
            pa.field("reliability_weight", pa.float64(), nullable=False),
            pa.field("first_available_at_utc", UTC_TIMESTAMP, nullable=False),
            pa.field("latest_available_at_utc", UTC_TIMESTAMP, nullable=False),
            pa.field("first_source_report_id", pa.string(), nullable=False),
            pa.field("latest_source_report_id", pa.string(), nullable=False),
            pa.field("latest_source_report_date", pa.date32(), nullable=False),
            pa.field("current_observation_ids", STRING_LIST, nullable=False),
            pa.field("latest_observation_ids", STRING_LIST, nullable=False),
            pa.field("latest_observation_id", pa.string(), nullable=True),
            pa.field("lineage_observation_ids", STRING_LIST, nullable=False),
            pa.field("lineage_source_report_ids", STRING_LIST, nullable=False),
            pa.field("lineage_observation_count", pa.int64(), nullable=False),
            pa.field("lineage_max_available_at_utc", UTC_TIMESTAMP, nullable=False),
            pa.field("lineage_sha256", pa.string(), nullable=False),
            pa.field("source_observation_ids", STRING_LIST, nullable=False),
            pa.field("source_report_ids", STRING_LIST, nullable=False),
            pa.field("max_source_available_at", UTC_TIMESTAMP, nullable=False),
        ]
    ),
    sort_keys=(
        "issue_time_utc",
        "state_row_kind",
        "anomaly_id",
        "entity_scope",
        "state_id",
    ),
    description=(
        "Causal anomaly state at every observed report issue, including one explicit "
        "report-period summary row so zero-entity issues remain replayable; complete "
        "entities persist after first causal availability while incomplete entities remain "
        "report-local."
    ),
)

STAGE3_CONTRACTS: dict[str, Stage3DatasetContract] = {
    ANOMALY_STATE_HISTORY_CONTRACT.name: ANOMALY_STATE_HISTORY_CONTRACT
}


def contract_document(contract: Stage3DatasetContract) -> dict[str, object]:
    """Return the deterministic machine-readable contract document."""

    return {
        "schema_version": 1,
        "contract_version": contract.version,
        "dataset": contract.name,
        "description": contract.description,
        "sort_keys": list(contract.sort_keys),
        "fields": [
            {"name": field.name, "type": str(field.type), "nullable": field.nullable}
            for field in contract.schema
        ],
    }


__all__ = [
    "ANOMALY_STATE_HISTORY_CONTRACT",
    "LOCAL_TIMESTAMP",
    "STAGE3_CONTRACTS",
    "STRING_LIST",
    "UTC_TIMESTAMP",
    "Stage3DatasetContract",
    "contract_document",
]
