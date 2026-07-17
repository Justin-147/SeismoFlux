"""Deterministic, target-blind replay audit for sealed stage-3 feature artifacts."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import cast
from zoneinfo import ZoneInfo

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from seismoflux.data.common import canonical_json_bytes
from seismoflux.features.anomaly.config import (
    ANOMALY_HISTORY_CONTRACT_VERSION,
    AnomalyHistoryConfig,
)
from seismoflux.features.anomaly.contracts import ANOMALY_STATE_HISTORY_CONTRACT
from seismoflux.features.anomaly.dictionary import (
    DEFAULT_FEATURE_DICTIONARY,
    NULL_REASON_VALID_CODE,
    FeatureDictionary,
    build_feature_store_schema,
)
from seismoflux.features.anomaly.engine import Stage3FeatureEngine
from seismoflux.features.anomaly.grid import Stage3QueryGrid
from seismoflux.features.anomaly.snapshot import Stage3IssueSnapshot, build_issue_snapshots
from seismoflux.features.anomaly.state import AnomalyState, states_from_records

_BEIJING = ZoneInfo("Asia/Shanghai")
_EXPECTED_AUDIT_CHECKS = frozenset(
    {
        "every_source_reference_resolves",
        "every_source_available_at_lte_issue_time",
        "every_feature_has_dictionary_definition",
        "every_feature_replays_from_state_history",
        "every_null_has_explicit_reason",
    }
)


@dataclass(frozen=True, slots=True)
class LineageAuditPlan:
    """Frozen target-blind issue and query-cell sampling instructions."""

    algorithm: str
    namespace: str
    seed_text: str
    issue_dates_local: tuple[date, ...]

    def __post_init__(self) -> None:
        if self.algorithm != "sha256_ranked_interior_plus_fixed_partition_boundaries_v1":
            raise ValueError("unsupported stage-3 lineage-audit sampling algorithm")
        if self.namespace != "stage3-lineage-audit" or self.seed_text != "147":
            raise ValueError("stage-3 lineage-audit namespace or seed drifted")
        if len(self.issue_dates_local) != 12 or len(set(self.issue_dates_local)) != 12:
            raise ValueError("stage-3 lineage audit requires 12 unique local issue dates")


@dataclass(frozen=True, slots=True)
class AuditResult:
    """Aggregate public-safe result; local cells and lineage identifiers never escape."""

    passed: bool
    selected_issue_count: int
    selected_feature_row_count: int
    unique_selected_cell_count: int
    state_row_count_checked: int
    observation_reference_count_checked: int
    report_reference_count_checked: int
    dictionary_definition_count: int
    dictionary_value_column_count: int
    feature_field_count: int
    feature_scalar_count_compared: int
    nullable_value_count_checked: int
    null_value_count_checked: int
    trajectory_value_count_checked: int


@dataclass(frozen=True, slots=True)
class _SourceReport:
    report_id: str
    report_date: date
    source_file: str
    available_at: datetime
    report_year: int
    report_period: int
    period_index: int


@dataclass(frozen=True, slots=True)
class _SourceObservation:
    observation_id: str
    anomaly_id: str
    available_at: datetime
    report: _SourceReport


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _text(mapping: Mapping[str, object], key: str, *, label: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label}.{key} must be a non-empty string")
    return value


def _integer(mapping: Mapping[str, object], key: str, *, label: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label}.{key} must be an integer")
    return value


def _string_sequence(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list | tuple) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"{label} must be a non-empty-string sequence")
    return tuple(value)


def parse_lineage_audit_plan(config: AnomalyHistoryConfig) -> LineageAuditPlan:
    """Parse the twelve frozen local audit dates without consulting targets or scores."""

    audit = _mapping(config.audit, label="audit")
    sample = _mapping(audit.get("deterministic_issue_sample"), label="audit sample")
    if sample.get("target_or_score_blind") is not True:
        raise ValueError("lineage-audit sampling must remain target- and score-blind")
    if sample.get("query_cell_sample_rule") != (
        "sha256_ranked_fixed_cell_ids_without_target_information"
    ):
        raise ValueError("lineage-audit query-cell sampling rule drifted")
    if sample.get("check_all_scales_and_features_for_selected_rows") is not True:
        raise ValueError("lineage audit must check every scale and feature")
    required_checks = frozenset(
        _string_sequence(sample.get("required_checks"), label="audit required_checks")
    )
    if required_checks != _EXPECTED_AUDIT_CHECKS:
        raise ValueError("lineage-audit required checks drifted")

    issue_dates = _mapping(sample.get("issue_dates_local"), label="audit issue_dates_local")
    if set(issue_dates) != {"development", "validation"}:
        raise ValueError("lineage-audit issue partitions must be development and validation")
    parsed: list[date] = []
    for partition in ("development", "validation"):
        values = _string_sequence(
            issue_dates.get(partition),
            label=f"audit issue_dates_local.{partition}",
        )
        if len(values) != 6:
            raise ValueError("each lineage-audit partition must contain six issue dates")
        for value in values:
            try:
                parsed.append(date.fromisoformat(value))
            except ValueError as exc:
                raise ValueError("lineage-audit issue date must use ISO YYYY-MM-DD") from exc

    if _integer(sample, "sample_size", label="audit sample") != len(parsed):
        raise ValueError("lineage-audit sample_size differs from frozen issue dates")
    return LineageAuditPlan(
        algorithm=_text(sample, "algorithm", label="audit sample"),
        namespace=_text(sample, "namespace", label="audit sample"),
        seed_text=_text(sample, "seed_text", label="audit sample"),
        issue_dates_local=tuple(parsed),
    )


def _utc(value: object, *, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be a timezone-aware datetime")
    return value.astimezone(UTC)


def _date(value: object, *, label: str) -> date:
    if isinstance(value, datetime) or not isinstance(value, date):
        raise ValueError(f"{label} must be a date")
    return value


def _nonempty_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _schema_without_document_metadata(schema: pa.Schema) -> pa.Schema:
    return schema.remove_metadata()


def _load_state_history(
    path: Path,
) -> tuple[tuple[AnomalyState, ...], tuple[Stage3IssueSnapshot, ...]]:
    if not path.is_file():
        raise ValueError("sealed anomaly_state_history.parquet is missing")
    expected_schema = _schema_without_document_metadata(ANOMALY_STATE_HISTORY_CONTRACT.schema)
    actual_schema = _schema_without_document_metadata(pq.read_schema(path))
    if actual_schema != expected_schema:
        raise ValueError("anomaly state-history Parquet schema differs from the frozen contract")
    table = pq.read_table(path)
    sorted_table = table.sort_by(
        [(key, "ascending") for key in ANOMALY_STATE_HISTORY_CONTRACT.sort_keys]
    )
    if not table.equals(sorted_table, check_metadata=False):
        raise ValueError("anomaly state-history Parquet is not deterministically sorted")
    records = cast(Sequence[Mapping[str, object]], table.to_pylist())
    states = states_from_records(records)
    return states, build_issue_snapshots(states)


def _source_indices(
    observation_table: pa.Table,
    report_period_table: pa.Table,
) -> tuple[dict[str, _SourceObservation], dict[str, _SourceReport]]:
    required_reports = {
        "report_id",
        "report_date",
        "source_file",
        "available_at",
        "report_year",
        "report_period",
    }
    required_observations = {
        "observation_id",
        "anomaly_id",
        "report_date",
        "source_file",
        "available_at",
    }
    if not required_reports <= set(report_period_table.column_names):
        raise ValueError("anomaly_report_period is missing lineage-audit columns")
    if not required_observations <= set(observation_table.column_names):
        raise ValueError("anomaly_observation is missing lineage-audit columns")

    raw_reports: list[tuple[str, date, str, datetime, int, int]] = []
    report_ids: set[str] = set()
    report_keys: set[tuple[date, str]] = set()
    for row in report_period_table.select(sorted(required_reports)).to_pylist():
        report_id = _nonempty_text(row["report_id"], label="report_id")
        report_date = _date(row["report_date"], label="report_date")
        source_file = _nonempty_text(row["source_file"], label="report source_file")
        available_at = _utc(row["available_at"], label="report available_at")
        report_year = row["report_year"]
        report_period = row["report_period"]
        if (
            isinstance(report_year, bool)
            or not isinstance(report_year, int)
            or isinstance(report_period, bool)
            or not isinstance(report_period, int)
        ):
            raise ValueError("source report year and period must be integers")
        if report_id in report_ids or (report_date, source_file) in report_keys:
            raise ValueError("source report identities must be unique for lineage audit")
        report_ids.add(report_id)
        report_keys.add((report_date, source_file))
        raw_reports.append(
            (report_id, report_date, source_file, available_at, report_year, report_period)
        )
    raw_reports.sort(key=lambda item: (item[3], item[1], item[0]))
    reports = {
        report_id: _SourceReport(
            report_id=report_id,
            report_date=report_date,
            source_file=source_file,
            available_at=available_at,
            report_year=report_year,
            report_period=report_period,
            period_index=index,
        )
        for index, (
            report_id,
            report_date,
            source_file,
            available_at,
            report_year,
            report_period,
        ) in enumerate(raw_reports)
    }
    reports_by_key = {(item.report_date, item.source_file): item for item in reports.values()}

    observations: dict[str, _SourceObservation] = {}
    for row in observation_table.select(sorted(required_observations)).to_pylist():
        observation_id = _nonempty_text(row["observation_id"], label="observation_id")
        if observation_id in observations:
            raise ValueError("source observation identities must be unique for lineage audit")
        report_date = _date(row["report_date"], label="observation report_date")
        source_file = _nonempty_text(row["source_file"], label="observation source_file")
        report = reports_by_key.get((report_date, source_file))
        if report is None:
            raise ValueError("source observation has no registered report-period identity")
        observations[observation_id] = _SourceObservation(
            observation_id=observation_id,
            anomaly_id=_nonempty_text(row["anomaly_id"], label="anomaly_id"),
            available_at=_utc(row["available_at"], label="observation available_at"),
            report=report,
        )
    return observations, reports


def _utc_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _expected_entity_lineage_sha256(
    state: AnomalyState,
    observations: Sequence[_SourceObservation],
) -> str:
    payload = {
        "contract_version": ANOMALY_HISTORY_CONTRACT_VERSION,
        "anomaly_id": state.anomaly_id,
        "entity_scope": state.entity_scope,
        "observations": [
            {
                "observation_id": observation.observation_id,
                "report_id": observation.report.report_id,
                "available_at": _utc_text(observation.available_at),
            }
            for observation in observations
        ],
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _expected_summary_lineage_sha256(state: AnomalyState) -> str:
    payload = {
        "contract_version": ANOMALY_HISTORY_CONTRACT_VERSION,
        "state_row_kind": "report_period_summary",
        "report_id": state.issue_report_id,
        "available_at": _utc_text(state.issue_time_utc),
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _audit_selected_state_lineage(
    snapshots: Sequence[Stage3IssueSnapshot],
    observations: Mapping[str, _SourceObservation],
    reports: Mapping[str, _SourceReport],
) -> tuple[int, int, int]:
    referenced_observations: set[str] = set()
    referenced_reports: set[str] = set()
    state_count = 0
    for snapshot in snapshots:
        for state in (snapshot.summary, *snapshot.entities):
            state_count += 1
            if state.contract_version != ANOMALY_HISTORY_CONTRACT_VERSION:
                raise ValueError("selected state row has an unexpected contract version")
            if (
                state.source_observation_ids != state.lineage_observation_ids
                or state.source_report_ids != state.lineage_source_report_ids
                or state.max_source_available_at != state.lineage_max_available_at_utc
            ):
                raise ValueError("selected state lineage aliases disagree")
            if state.lineage_observation_count != len(state.lineage_observation_ids):
                raise ValueError("selected state lineage observation count disagrees")
            if state.latest_observation_id is not None and (
                state.latest_observation_id not in state.latest_observation_ids
            ):
                raise ValueError("selected state latest-observation identity disagrees")

            observation_references = set(
                (
                    *state.current_observation_ids,
                    *state.latest_observation_ids,
                    *state.lineage_observation_ids,
                    *state.source_observation_ids,
                )
            )
            if state.latest_observation_id is not None:
                observation_references.add(state.latest_observation_id)
            report_references = set(
                (
                    state.issue_report_id,
                    state.first_source_report_id,
                    state.latest_source_report_id,
                    *state.lineage_source_report_ids,
                    *state.source_report_ids,
                )
            )
            if state.previous_issue_report_id is not None:
                report_references.add(state.previous_issue_report_id)
            if not observation_references <= set(observations):
                raise ValueError("selected state has a dangling source-observation reference")
            if not report_references <= set(reports):
                raise ValueError("selected state has a dangling source-report reference")
            referenced_observations.update(observation_references)
            referenced_reports.update(report_references)

            issue_report = reports[state.issue_report_id]
            if (
                issue_report.available_at != state.issue_time_utc
                or issue_report.report_date != state.issue_report_date
                or issue_report.report_year != state.issue_report_year
                or issue_report.report_period != state.issue_report_period
            ):
                raise ValueError("selected state issue identity differs from its source report")

            if any(
                observations[identifier].available_at > state.issue_time_utc
                for identifier in observation_references
            ):
                raise ValueError("selected state references an observation later than its issue")
            if any(
                reports[identifier].available_at > state.issue_time_utc
                for identifier in report_references
            ):
                raise ValueError("selected state references a report later than its issue")
            if any(
                value > state.issue_time_utc
                for value in (
                    state.first_available_at_utc,
                    state.latest_available_at_utc,
                    state.lineage_max_available_at_utc,
                    state.max_source_available_at,
                )
            ):
                raise ValueError("selected state max source availability exceeds its issue")

            if state.state_row_kind == "report_period_summary":
                if state.lineage_observation_ids or state.lineage_source_report_ids != (
                    state.issue_report_id,
                ):
                    raise ValueError("selected report-summary lineage differs from its report")
                if state.lineage_max_available_at_utc != state.issue_time_utc:
                    raise ValueError("selected report-summary max availability differs from issue")
                expected_sha256 = _expected_summary_lineage_sha256(state)
            else:
                if not state.lineage_observation_ids:
                    raise ValueError("selected entity state has empty observation lineage")
                ordered = tuple(
                    sorted(
                        (observations[item] for item in state.lineage_observation_ids),
                        key=lambda item: (
                            item.available_at,
                            item.report.period_index,
                            item.observation_id,
                        ),
                    )
                )
                expected_observation_ids = tuple(item.observation_id for item in ordered)
                expected_report_ids = tuple(
                    dict.fromkeys(item.report.report_id for item in ordered)
                )
                if expected_observation_ids != state.lineage_observation_ids:
                    raise ValueError("selected entity observation lineage order differs")
                if expected_report_ids != state.lineage_source_report_ids:
                    raise ValueError("selected entity report lineage order differs")
                if any(item.anomaly_id != state.anomaly_id for item in ordered):
                    raise ValueError("selected entity lineage resolves to another anomaly identity")
                if max(item.available_at for item in ordered) != state.lineage_max_available_at_utc:
                    raise ValueError("selected entity lineage max availability differs")
                expected_sha256 = _expected_entity_lineage_sha256(state, ordered)
            if state.lineage_sha256 != expected_sha256:
                raise ValueError("selected state lineage SHA-256 cannot be reproduced")
    return state_count, len(referenced_observations), len(referenced_reports)


def _cell_rank(
    plan: LineageAuditPlan,
    issue_date_local: date,
    cell_id: str,
) -> bytes:
    payload = {
        "algorithm": "sha256_ranked_fixed_cell_ids_without_target_information_v1",
        "cell_id": cell_id,
        "issue_date_local": issue_date_local.isoformat(),
        "namespace": plan.namespace,
        "seed_text": plan.seed_text,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).digest()


def _selected_cells(
    plan: LineageAuditPlan,
    query_grid: Stage3QueryGrid,
) -> dict[date, str]:
    cell_ids = query_grid.cell_ids
    if not cell_ids or len(cell_ids) != len(set(cell_ids)):
        raise ValueError("fixed stage-3 query grid must contain unique cell identities")
    return {
        issue_date: min(
            cell_ids,
            key=lambda cell_id: (_cell_rank(plan, issue_date, cell_id), cell_id),
        )
        for issue_date in plan.issue_dates_local
    }


def _selected_subgrid(
    query_grid: Stage3QueryGrid,
    selected_cell_ids: set[str],
) -> Stage3QueryGrid:
    size = query_grid.cell_count
    if not (
        query_grid.rows.shape == (size,)
        and query_grid.columns.shape == (size,)
        and query_grid.query_xy_m.shape == (size, 2)
        and query_grid.clipped_area_km2.shape == (size,)
    ):
        raise ValueError("fixed stage-3 query-grid arrays have inconsistent shapes")
    indices = np.asarray(
        [
            index
            for index, cell_id in enumerate(query_grid.cell_ids)
            if cell_id in selected_cell_ids
        ],
        dtype=np.int64,
    )
    if indices.size != len(selected_cell_ids):
        raise ValueError("selected audit cell does not resolve to the fixed query grid")
    return Stage3QueryGrid(
        grid_id=query_grid.grid_id,
        equal_area_crs=query_grid.equal_area_crs,
        cell_size_km=query_grid.cell_size_km,
        cell_ids=tuple(query_grid.cell_ids[index] for index in indices),
        rows=query_grid.rows[indices],
        columns=query_grid.columns[indices],
        query_xy_m=query_grid.query_xy_m[indices, :],
        clipped_area_km2=query_grid.clipped_area_km2[indices],
    )


def _dictionary_columns(dictionary: FeatureDictionary) -> tuple[set[str], int]:
    columns: set[str] = set()
    for definition in dictionary.definitions:
        for value_column in definition.storage_value_columns():
            columns.add(value_column)
            for companion in (
                definition.validity_field(value_column),
                definition.sample_count_field(value_column),
                definition.null_reason_code_field(value_column),
            ):
                if companion is not None:
                    columns.add(companion)
    if len(dictionary.storage_column_map()) != len(dictionary.storage_value_columns()):
        raise ValueError("feature dictionary does not define each value column exactly once")
    return columns, len(dictionary.storage_value_columns())


def _validate_feature_store_schema(
    path: Path,
    dictionary: FeatureDictionary,
) -> tuple[pa.Schema, int]:
    if not path.is_file():
        raise ValueError("sealed anomaly_feature_store.parquet is missing")
    expected_schema = build_feature_store_schema(dictionary)
    actual_schema = pq.read_schema(path)
    if _schema_without_document_metadata(actual_schema) != _schema_without_document_metadata(
        expected_schema
    ):
        raise ValueError("feature-store Parquet schema differs from the dictionary contract")
    dictionary_columns, value_column_count = _dictionary_columns(dictionary)
    if not dictionary_columns <= set(actual_schema.names):
        raise ValueError("feature-store Parquet does not cover every dictionary column")
    return expected_schema, value_column_count


def _one_replay_row(table: pa.Table, cell_id: str) -> pa.Table:
    row = table.filter(pc.equal(table["cell_id"], pa.scalar(cell_id)))
    if row.num_rows != 1:
        raise ValueError("replayed issue table did not resolve exactly one selected cell")
    return row


def _one_persisted_row(
    path: Path,
    *,
    issue_index: int,
    cell_id: str,
) -> pa.Table:
    row = pq.read_table(
        path,
        filters=[("issue_index", "=", issue_index), ("cell_id", "=", cell_id)],
    )
    if row.num_rows != 1:
        raise ValueError("feature store did not resolve exactly one selected issue-cell row")
    return row


def _audit_feature_row_quality(
    row: pa.Table,
    dictionary: FeatureDictionary,
    expected_schema: pa.Schema,
) -> tuple[int, int, int]:
    if row.num_rows != 1:
        raise ValueError("feature quality audit requires exactly one row")
    for field in expected_schema:
        if not field.nullable and not row[field.name][0].is_valid:
            raise ValueError("feature store contains a null in a non-nullable field")

    nullable_values = 0
    null_values = 0
    trajectory_values = 0
    for definition in dictionary.definitions:
        allowed_reason_codes = set(definition.null_reason_codes.values())
        for value_column in definition.storage_value_columns():
            value = row[value_column][0]
            reason_field = definition.null_reason_code_field(value_column)
            if definition.nullable:
                nullable_values += 1
                if reason_field is None:
                    raise ValueError("nullable dictionary value has no reason-code field")
                reason = row[reason_field][0].as_py()
                if isinstance(reason, bool) or not isinstance(reason, int):
                    raise ValueError("feature null-reason code must be an integer")
                if reason not in allowed_reason_codes:
                    raise ValueError("feature null-reason code is absent from its definition")
                if value.is_valid and reason != NULL_REASON_VALID_CODE:
                    raise ValueError("valid feature value must use null-reason code zero")
                if not value.is_valid:
                    null_values += 1
                    if reason == NULL_REASON_VALID_CODE:
                        raise ValueError("null feature value must use a nonzero reason code")

            validity_field = definition.validity_field(value_column)
            sample_count_field = definition.sample_count_field(value_column)
            if definition.producer == "trajectory_v1":
                trajectory_values += 1
                if validity_field is None or sample_count_field is None:
                    raise ValueError("trajectory value is missing validity or sample count")
                valid = row[validity_field][0].as_py()
                sample_count = row[sample_count_field][0].as_py()
                if not isinstance(valid, bool):
                    raise ValueError("trajectory validity companion must be boolean")
                if isinstance(sample_count, bool) or not isinstance(sample_count, int):
                    raise ValueError("trajectory sample-count companion must be an integer")
                if valid != value.is_valid:
                    raise ValueError("trajectory validity companion disagrees with stored value")
                if sample_count < 0 or (valid and sample_count == 0):
                    raise ValueError("trajectory validity and sample count are inconsistent")
    return nullable_values, null_values, trajectory_values


def _first_different_field(expected: pa.Table, observed: pa.Table) -> str | None:
    for name in cast(list[str], expected.column_names):
        if not expected[name].equals(observed[name]):
            return name
    return None


def run_lineage_replay_audit(
    *,
    config: AnomalyHistoryConfig,
    anomaly_state_history_path: Path,
    anomaly_feature_store_path: Path,
    query_grid: Stage3QueryGrid,
    observation_table: pa.Table,
    report_period_table: pa.Table,
    dictionary: FeatureDictionary = DEFAULT_FEATURE_DICTIONARY,
    query_chunk_size: int = 256,
) -> AuditResult:
    """Replay the twelve frozen rows and return only aggregate public-safe diagnostics."""

    plan = parse_lineage_audit_plan(config)
    _, snapshots = _load_state_history(anomaly_state_history_path)
    observations, reports = _source_indices(observation_table, report_period_table)

    snapshots_by_local_date: dict[date, Stage3IssueSnapshot] = {}
    for snapshot in snapshots:
        local_date = snapshot.issue_time_utc.astimezone(_BEIJING).date()
        if local_date in snapshots_by_local_date:
            raise ValueError("multiple state snapshots share one local issue date")
        snapshots_by_local_date[local_date] = snapshot
    missing_dates = set(plan.issue_dates_local) - set(snapshots_by_local_date)
    if missing_dates:
        raise ValueError("one or more frozen lineage-audit issue dates are absent")
    selected_snapshots = tuple(snapshots_by_local_date[item] for item in plan.issue_dates_local)
    state_count, observation_reference_count, report_reference_count = (
        _audit_selected_state_lineage(selected_snapshots, observations, reports)
    )

    expected_schema, dictionary_value_column_count = _validate_feature_store_schema(
        anomaly_feature_store_path,
        dictionary,
    )
    selected_cell_by_date = _selected_cells(plan, query_grid)
    selected_cell_ids = set(selected_cell_by_date.values())
    audit_grid = _selected_subgrid(query_grid, selected_cell_ids)
    selected_by_issue_index = {
        snapshots_by_local_date[issue_date].issue_index: selected_cell_by_date[issue_date]
        for issue_date in plan.issue_dates_local
    }

    engine = Stage3FeatureEngine(
        snapshots,
        audit_grid,
        dictionary=dictionary,
        query_chunk_size=query_chunk_size,
    )
    compared_rows = 0
    nullable_values = 0
    null_values = 0
    trajectory_values = 0
    for issue_index in range(len(snapshots)):
        replay = engine.build_next_issue().table
        cell_id = selected_by_issue_index.get(issue_index)
        if cell_id is None:
            continue
        expected_row = _one_replay_row(replay, cell_id)
        observed_row = _one_persisted_row(
            anomaly_feature_store_path,
            issue_index=issue_index,
            cell_id=cell_id,
        ).select(expected_schema.names)
        row_nullable, row_nulls, row_trajectory = _audit_feature_row_quality(
            observed_row,
            dictionary,
            expected_schema,
        )
        nullable_values += row_nullable
        null_values += row_nulls
        trajectory_values += row_trajectory
        if not expected_row.equals(observed_row, check_metadata=False):
            field = _first_different_field(expected_row, observed_row)
            raise ValueError(f"feature replay differs from persisted row in field: {field}")
        compared_rows += 1

    if compared_rows != len(plan.issue_dates_local):
        raise AssertionError("lineage replay did not compare all twelve frozen issue rows")
    return AuditResult(
        passed=True,
        selected_issue_count=len(plan.issue_dates_local),
        selected_feature_row_count=compared_rows,
        unique_selected_cell_count=len(selected_cell_ids),
        state_row_count_checked=state_count,
        observation_reference_count_checked=observation_reference_count,
        report_reference_count_checked=report_reference_count,
        dictionary_definition_count=len(dictionary.definitions),
        dictionary_value_column_count=dictionary_value_column_count,
        feature_field_count=len(expected_schema),
        feature_scalar_count_compared=compared_rows * len(expected_schema),
        nullable_value_count_checked=nullable_values,
        null_value_count_checked=null_values,
        trajectory_value_count_checked=trajectory_values,
    )


__all__ = [
    "AuditResult",
    "LineageAuditPlan",
    "parse_lineage_audit_plan",
    "run_lineage_replay_audit",
]
