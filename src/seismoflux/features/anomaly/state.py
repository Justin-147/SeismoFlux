"""Pure, causal reconstruction of dynamic-anomaly state history."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, fields
from datetime import UTC, date, datetime, timedelta, timezone
from itertools import pairwise
from typing import Any, Literal, TypeVar

from seismoflux.data.common import canonical_json_bytes, stable_token
from seismoflux.features.anomaly.config import ANOMALY_HISTORY_CONTRACT_VERSION

StateRowKind = Literal["entity_state", "report_period_summary"]
EntityScope = Literal[
    "persistent_complete_entity",
    "report_local_incomplete_entity",
    "report_period_summary",
]
ReliabilityGrade = Literal["high", "cautious", "excluded"]

_REPORT_STATES = ("新增", "持续", "取消")
_REPORT_STATE_ORDER = {value: index for index, value in enumerate(_REPORT_STATES)}
_LOCAL_FIXED_OFFSET = timezone(timedelta(hours=8))
_OBSERVATION_COLUMN_ALLOWLIST = frozenset(
    {
        "observation_id",
        "anomaly_id",
        "station_id",
        "report_date",
        "raw_row_report_date",
        "available_at",
        "longitude",
        "latitude",
        "discipline",
        "measurement",
        "instrument_or_method",
        "observation_period",
        "start_time",
        "reported_end_time",
        "end_flag",
        "report_state",
        "anomaly_type",
        "is_listed",
        "right_censored",
        "identity_complete",
        "reliability_flags",
        "source_id",
        "source_file",
        "source_sha256",
        "source_sheet",
        "source_row",
        "source_serial_number",
        "source_period_year",
        "source_period",
    }
)
_REPORT_PERIOD_COLUMN_ALLOWLIST = frozenset(
    {
        "report_id",
        "source_id",
        "source_file",
        "source_sha256",
        "report_year",
        "report_period",
        "report_date",
        "available_at",
        "row_count",
        "row_report_date_mismatch_count",
        "row_report_date_before_count",
        "row_report_date_after_count",
        "deformation_row_count",
        "fluid_row_count",
        "electromagnetic_row_count",
        "cross_fault_row_count",
    }
)
_CAUTIOUS_RELIABILITY_FLAGS = frozenset(
    {
        "available_at_conservatively_delayed",
        "row_report_date_mismatch_source",
        "row_report_date_before_source",
        "row_report_date_after_source",
        "exact_duplicate_in_report",
        "natural_key_collision",
        "end_time_retracted",
        "end_time_revised",
        "listed_after_reported_end",
        "cancel_without_end_time",
        "continued_with_end_time",
        "new_with_end_time",
        "end_after_raw_row_report_date",
        "future_reported_end_time",
        "end_before_start",
        "end_flag_inconsistent",
        "start_after_report_date",
    }
)
_EXCLUDED_RELIABILITY_FLAGS = frozenset(
    {
        "identity_incomplete",
        "entity_unresolved",
        "missing_station_name",
        "missing_measurement",
    }
)
_RELIABILITY_WEIGHTS: dict[ReliabilityGrade, float] = {
    "high": 1.0,
    "cautious": 0.5,
    "excluded": 0.0,
}
_ValueT = TypeVar("_ValueT", bound=Hashable)


@dataclass(frozen=True, slots=True)
class AnomalyState:
    """One issue-time state with causal lineage and separate source/system transitions."""

    state_id: str
    contract_version: str
    state_row_kind: StateRowKind
    issue_time_utc: datetime
    issue_report_id: str
    issue_report_date: date
    issue_report_year: int
    issue_report_period: int
    row_count: int
    row_report_date_mismatch_count: int
    row_report_date_before_count: int
    row_report_date_after_count: int
    deformation_row_count: int
    fluid_row_count: int
    electromagnetic_row_count: int
    cross_fault_row_count: int
    previous_issue_report_id: str | None
    previous_period_consecutive: bool
    anomaly_id: str
    identity_complete: bool
    entity_scope: EntityScope
    station_id: str | None
    station_id_null_reason: str | None
    longitude: float | None
    longitude_null_reason: str | None
    latitude: float | None
    latitude_null_reason: str | None
    discipline: str | None
    discipline_null_reason: str | None
    measurement: str | None
    measurement_null_reason: str | None
    start_time: datetime | None
    start_time_null_reason: str | None
    spatial_eligible: bool
    spatial_exclusion_reason: str | None
    current_reporting_station_ids: tuple[str, ...]
    current_reporting_disciplines: tuple[str, ...]
    current_reporting_measurements: tuple[str, ...]
    known_station_ids: tuple[str, ...]
    known_disciplines: tuple[str, ...]
    known_measurements: tuple[str, ...]
    current_report_listed: bool
    source_new: bool
    source_continued: bool
    source_cancelled: bool
    current_source_report_states: tuple[str, ...]
    latest_source_report_states: tuple[str, ...]
    system_first_seen: bool
    system_not_continued: bool
    system_relisted: bool
    left_truncated: bool
    late_entry_or_gap_unknown: bool
    explicit_end_known: bool
    right_censored: bool
    reported_end_time: datetime | None
    reported_end_time_null_reason: str | None
    age_days: float | None
    age_days_null_reason: str | None
    known_duration_days: float | None
    known_duration_days_null_reason: str | None
    reliability_flags: tuple[str, ...]
    reliability_grade: ReliabilityGrade
    reliability_weight: float
    first_available_at_utc: datetime
    latest_available_at_utc: datetime
    first_source_report_id: str
    latest_source_report_id: str
    latest_source_report_date: date
    current_observation_ids: tuple[str, ...]
    latest_observation_ids: tuple[str, ...]
    latest_observation_id: str | None
    lineage_observation_ids: tuple[str, ...]
    lineage_source_report_ids: tuple[str, ...]
    lineage_observation_count: int
    lineage_max_available_at_utc: datetime
    lineage_sha256: str
    source_observation_ids: tuple[str, ...]
    source_report_ids: tuple[str, ...]
    max_source_available_at: datetime

    def to_record(self) -> dict[str, object]:
        """Return a record matching the independent stage-3 Arrow schema."""

        return {
            "state_id": self.state_id,
            "contract_version": self.contract_version,
            "state_row_kind": self.state_row_kind,
            "issue_time_utc": self.issue_time_utc,
            "issue_report_id": self.issue_report_id,
            "issue_report_date": self.issue_report_date,
            "issue_report_year": self.issue_report_year,
            "issue_report_period": self.issue_report_period,
            "row_count": self.row_count,
            "row_report_date_mismatch_count": self.row_report_date_mismatch_count,
            "row_report_date_before_count": self.row_report_date_before_count,
            "row_report_date_after_count": self.row_report_date_after_count,
            "deformation_row_count": self.deformation_row_count,
            "fluid_row_count": self.fluid_row_count,
            "electromagnetic_row_count": self.electromagnetic_row_count,
            "cross_fault_row_count": self.cross_fault_row_count,
            "previous_issue_report_id": self.previous_issue_report_id,
            "previous_period_consecutive": self.previous_period_consecutive,
            "anomaly_id": self.anomaly_id,
            "identity_complete": self.identity_complete,
            "entity_scope": self.entity_scope,
            "station_id": self.station_id,
            "station_id_null_reason": self.station_id_null_reason,
            "longitude": self.longitude,
            "longitude_null_reason": self.longitude_null_reason,
            "latitude": self.latitude,
            "latitude_null_reason": self.latitude_null_reason,
            "discipline": self.discipline,
            "discipline_null_reason": self.discipline_null_reason,
            "measurement": self.measurement,
            "measurement_null_reason": self.measurement_null_reason,
            "start_time": self.start_time,
            "start_time_null_reason": self.start_time_null_reason,
            "spatial_eligible": self.spatial_eligible,
            "spatial_exclusion_reason": self.spatial_exclusion_reason,
            "current_reporting_station_ids": list(self.current_reporting_station_ids),
            "current_reporting_disciplines": list(self.current_reporting_disciplines),
            "current_reporting_measurements": list(self.current_reporting_measurements),
            "known_station_ids": list(self.known_station_ids),
            "known_disciplines": list(self.known_disciplines),
            "known_measurements": list(self.known_measurements),
            "current_report_listed": self.current_report_listed,
            "source_new": self.source_new,
            "source_continued": self.source_continued,
            "source_cancelled": self.source_cancelled,
            "current_source_report_states": list(self.current_source_report_states),
            "latest_source_report_states": list(self.latest_source_report_states),
            "system_first_seen": self.system_first_seen,
            "system_not_continued": self.system_not_continued,
            "system_relisted": self.system_relisted,
            "left_truncated": self.left_truncated,
            "late_entry_or_gap_unknown": self.late_entry_or_gap_unknown,
            "explicit_end_known": self.explicit_end_known,
            "right_censored": self.right_censored,
            "reported_end_time": self.reported_end_time,
            "reported_end_time_null_reason": self.reported_end_time_null_reason,
            "age_days": self.age_days,
            "age_days_null_reason": self.age_days_null_reason,
            "known_duration_days": self.known_duration_days,
            "known_duration_days_null_reason": self.known_duration_days_null_reason,
            "reliability_flags": list(self.reliability_flags),
            "reliability_grade": self.reliability_grade,
            "reliability_weight": self.reliability_weight,
            "first_available_at_utc": self.first_available_at_utc,
            "latest_available_at_utc": self.latest_available_at_utc,
            "first_source_report_id": self.first_source_report_id,
            "latest_source_report_id": self.latest_source_report_id,
            "latest_source_report_date": self.latest_source_report_date,
            "current_observation_ids": list(self.current_observation_ids),
            "latest_observation_ids": list(self.latest_observation_ids),
            "latest_observation_id": self.latest_observation_id,
            "lineage_observation_ids": list(self.lineage_observation_ids),
            "lineage_source_report_ids": list(self.lineage_source_report_ids),
            "lineage_observation_count": self.lineage_observation_count,
            "lineage_max_available_at_utc": self.lineage_max_available_at_utc,
            "lineage_sha256": self.lineage_sha256,
            "source_observation_ids": list(self.source_observation_ids),
            "source_report_ids": list(self.source_report_ids),
            "max_source_available_at": self.max_source_available_at,
        }


@dataclass(frozen=True, slots=True)
class _ReportPeriod:
    index: int
    report_id: str
    source_file: str
    report_date: date
    available_at: datetime
    report_year: int
    report_period: int
    row_count: int
    row_report_date_mismatch_count: int
    row_report_date_before_count: int
    row_report_date_after_count: int
    deformation_row_count: int
    fluid_row_count: int
    electromagnetic_row_count: int
    cross_fault_row_count: int
    previous_period_consecutive: bool

    @property
    def key(self) -> tuple[date, str]:
        return self.report_date, self.source_file


@dataclass(frozen=True, slots=True)
class _Observation:
    observation_id: str
    anomaly_id: str
    identity_complete: bool
    available_at: datetime
    period_index: int
    source_report_id: str
    source_report_date: date
    report_state: str
    station_id: str
    longitude: float | None
    latitude: float | None
    discipline: str
    measurement: str | None
    start_time: datetime
    is_listed: bool
    right_censored: bool
    reported_end_time: datetime | None
    reliability_flags: tuple[str, ...]


def _require_string(record: Mapping[str, object], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _optional_string(record: Mapping[str, object], field: str) -> str | None:
    value = record.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be null or a non-empty string")
    return value


def _require_bool(record: Mapping[str, object], field: str) -> bool:
    value = record.get(field)
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _require_integer(record: Mapping[str, object], field: str) -> int:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _require_date(record: Mapping[str, object], field: str) -> date:
    value = record.get(field)
    if isinstance(value, datetime) or not isinstance(value, date):
        raise ValueError(f"{field} must be a date")
    return value


def _aware_datetime(value: object, *, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be a timezone-aware datetime")
    return value


def _require_utc_datetime(record: Mapping[str, object], field: str) -> datetime:
    return _aware_datetime(record.get(field), field=field).astimezone(UTC)


def _optional_local_datetime(record: Mapping[str, object], field: str) -> datetime | None:
    value = record.get(field)
    if value is None:
        return None
    return _aware_datetime(value, field=field).astimezone(_LOCAL_FIXED_OFFSET)


def _require_local_datetime(record: Mapping[str, object], field: str) -> datetime:
    return _aware_datetime(record.get(field), field=field).astimezone(_LOCAL_FIXED_OFFSET)


def _optional_finite_float(
    record: Mapping[str, object],
    field: str,
    *,
    lower: float,
    upper: float,
) -> float | None:
    value = record.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be null or a finite number")
    number = float(value)
    if not math.isfinite(number) or not lower <= number <= upper:
        raise ValueError(f"{field} must be within [{lower}, {upper}]")
    return number


def _require_string_sequence(record: Mapping[str, object], field: str) -> tuple[str, ...]:
    value = record.get(field)
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError(f"{field} must be a sequence of strings")
    parsed: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise ValueError(f"{field} must contain only non-empty strings")
        parsed.append(item)
    return tuple(sorted(set(parsed)))


def _validate_observation_columns(record: Mapping[str, object]) -> None:
    unexpected = sorted(str(field) for field in set(record) - _OBSERVATION_COLUMN_ALLOWLIST)
    if unexpected:
        raise ValueError(
            "anomaly observation contains forbidden or unexpected columns: " + ", ".join(unexpected)
        )


def _validate_report_period_columns(record: Mapping[str, object]) -> None:
    unexpected = sorted(str(field) for field in set(record) - _REPORT_PERIOD_COLUMN_ALLOWLIST)
    if unexpected:
        raise ValueError(
            "anomaly report period contains forbidden or unexpected columns: "
            + ", ".join(unexpected)
        )


def _resolve_duplicate_values(
    values: Sequence[_ValueT | None],
) -> tuple[_ValueT | None, str | None]:
    non_null = tuple(dict.fromkeys(value for value in values if value is not None))
    if not non_null:
        return None, "source_value_missing"
    if len(non_null) > 1:
        return None, "conflicting_duplicate_values"
    if any(value is None for value in values):
        return None, "partial_missing_duplicate_values"
    return non_null[0], None


def _resolve_end_time(
    observations: Sequence[_Observation],
) -> tuple[bool, datetime | None, str | None]:
    end_times = tuple(observation.reported_end_time for observation in observations)
    non_null = tuple(dict.fromkeys(value for value in end_times if value is not None))
    if not non_null:
        return False, None, "no_reported_end_time"
    if any(observation.right_censored for observation in observations):
        return False, None, "end_evidence_right_censored_or_future"
    if len(non_null) > 1:
        return False, None, "conflicting_duplicate_end_evidence"
    if any(value is None for value in end_times):
        return False, None, "partial_missing_duplicate_end_evidence"
    return True, non_null[0], None


def _reliability(
    observations: Sequence[_Observation],
    *,
    identity_complete: bool,
) -> tuple[tuple[str, ...], ReliabilityGrade, float]:
    flags = tuple(
        sorted({flag for observation in observations for flag in observation.reliability_flags})
    )
    flag_set = set(flags)
    if not identity_complete or flag_set & _EXCLUDED_RELIABILITY_FLAGS:
        grade: ReliabilityGrade = "excluded"
    elif len(observations) > 1 or flag_set & _CAUTIOUS_RELIABILITY_FLAGS:
        grade = "cautious"
    else:
        grade = "high"
    return flags, grade, _RELIABILITY_WEIGHTS[grade]


def _days(delta: timedelta) -> float:
    return delta.total_seconds() / 86_400.0


def _utc_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_report_periods(
    records: Sequence[Mapping[str, object]],
    *,
    expected_report_period_count: int | None,
) -> tuple[_ReportPeriod, ...]:
    if not records:
        raise ValueError("anomaly report-period schedule must not be empty")
    if expected_report_period_count is not None:
        if expected_report_period_count <= 0:
            raise ValueError("expected_report_period_count must be positive")
        if len(records) != expected_report_period_count:
            raise ValueError(
                "anomaly report-period count differs from the frozen expected count: "
                f"expected={expected_report_period_count}, observed={len(records)}"
            )

    raw: list[
        tuple[str, str, date, datetime, int, int, int, int, int, int, int, int, int, int]
    ] = []
    report_ids: set[str] = set()
    period_keys: set[tuple[date, str]] = set()
    for record in records:
        _validate_report_period_columns(record)
        report_id = _require_string(record, "report_id")
        source_file = _require_string(record, "source_file")
        report_date = _require_date(record, "report_date")
        available_at = _require_utc_datetime(record, "available_at")
        report_year = _require_integer(record, "report_year")
        report_period = _require_integer(record, "report_period")
        report_counts = (
            _require_integer(record, "row_count"),
            _require_integer(record, "row_report_date_mismatch_count"),
            _require_integer(record, "row_report_date_before_count"),
            _require_integer(record, "row_report_date_after_count"),
            _require_integer(record, "deformation_row_count"),
            _require_integer(record, "fluid_row_count"),
            _require_integer(record, "electromagnetic_row_count"),
            _require_integer(record, "cross_fault_row_count"),
        )
        if report_year <= 0 or report_period <= 0:
            raise ValueError("report_year and report_period must be positive")
        if any(count < 0 for count in report_counts):
            raise ValueError("anomaly report-period aggregate counts must be non-negative")
        if sum(report_counts[4:]) != report_counts[0]:
            raise ValueError("anomaly report-period discipline counts must sum to row_count")
        if report_counts[1] != report_counts[2] + report_counts[3]:
            raise ValueError(
                "row_report_date_mismatch_count must equal before_count plus after_count"
            )
        if report_id in report_ids:
            raise ValueError(f"duplicate anomaly report_id: {report_id}")
        key = (report_date, source_file)
        if key in period_keys:
            raise ValueError(
                "duplicate anomaly report period for report_date/source_file: "
                f"{report_date.isoformat()} {source_file}"
            )
        report_ids.add(report_id)
        period_keys.add(key)
        raw.append(
            (
                report_id,
                source_file,
                report_date,
                available_at,
                report_year,
                report_period,
                *report_counts,
            )
        )

    raw.sort(key=lambda item: (item[3], item[2], item[0]))
    for previous, current in pairwise(raw):
        if current[3] <= previous[3]:
            raise ValueError("report-period available_at values must be strictly increasing")
        if current[2] <= previous[2]:
            raise ValueError("report-period report_date values must be strictly increasing")

    parsed: list[_ReportPeriod] = []
    for index, (
        report_id,
        source_file,
        report_date,
        available_at,
        report_year,
        report_period,
        row_count,
        row_report_date_mismatch_count,
        row_report_date_before_count,
        row_report_date_after_count,
        deformation_row_count,
        fluid_row_count,
        electromagnetic_row_count,
        cross_fault_row_count,
    ) in enumerate(raw):
        previous_period_consecutive = bool(
            index and report_date - raw[index - 1][2] == timedelta(days=7)
        )
        parsed.append(
            _ReportPeriod(
                index=index,
                report_id=report_id,
                source_file=source_file,
                report_date=report_date,
                available_at=available_at,
                report_year=report_year,
                report_period=report_period,
                row_count=row_count,
                row_report_date_mismatch_count=row_report_date_mismatch_count,
                row_report_date_before_count=row_report_date_before_count,
                row_report_date_after_count=row_report_date_after_count,
                deformation_row_count=deformation_row_count,
                fluid_row_count=fluid_row_count,
                electromagnetic_row_count=electromagnetic_row_count,
                cross_fault_row_count=cross_fault_row_count,
                previous_period_consecutive=previous_period_consecutive,
            )
        )
    return tuple(parsed)


def _parse_observations(
    records: Sequence[Mapping[str, object]],
    periods: tuple[_ReportPeriod, ...],
) -> tuple[_Observation, ...]:
    period_by_key = {period.key: period for period in periods}
    observation_ids: set[str] = set()
    parsed: list[_Observation] = []
    for record in records:
        _validate_observation_columns(record)
        observation_id = _require_string(record, "observation_id")
        if observation_id in observation_ids:
            raise ValueError(f"duplicate observation_id: {observation_id}")
        observation_ids.add(observation_id)

        source_file = _require_string(record, "source_file")
        report_date = _require_date(record, "report_date")
        period = period_by_key.get((report_date, source_file))
        if period is None:
            raise ValueError(
                f"observation does not resolve to an anomaly_report_period: {observation_id}"
            )
        available_at = _require_utc_datetime(record, "available_at")
        if available_at < period.available_at:
            raise ValueError(
                "observation available_at cannot precede its report-period available_at: "
                f"{observation_id}"
            )
        report_state = _require_string(record, "report_state")
        if report_state not in _REPORT_STATE_ORDER:
            raise ValueError(f"unsupported anomaly report_state: {report_state}")
        right_censored = _require_bool(record, "right_censored")
        reported_end_time = _optional_local_datetime(record, "reported_end_time")
        if reported_end_time is None and not right_censored:
            raise ValueError(
                "an observation without reported_end_time must remain right-censored: "
                f"{observation_id}"
            )
        parsed.append(
            _Observation(
                observation_id=observation_id,
                anomaly_id=_require_string(record, "anomaly_id"),
                identity_complete=_require_bool(record, "identity_complete"),
                available_at=available_at,
                period_index=period.index,
                source_report_id=period.report_id,
                source_report_date=period.report_date,
                report_state=report_state,
                station_id=_require_string(record, "station_id"),
                longitude=_optional_finite_float(
                    record,
                    "longitude",
                    lower=-180.0,
                    upper=180.0,
                ),
                latitude=_optional_finite_float(
                    record,
                    "latitude",
                    lower=-90.0,
                    upper=90.0,
                ),
                discipline=_require_string(record, "discipline"),
                measurement=_optional_string(record, "measurement"),
                start_time=_require_local_datetime(record, "start_time"),
                is_listed=_require_bool(record, "is_listed"),
                right_censored=right_censored,
                reported_end_time=reported_end_time,
                reliability_flags=_require_string_sequence(record, "reliability_flags"),
            )
        )
    return tuple(
        sorted(
            parsed,
            key=lambda item: (
                item.available_at,
                item.period_index,
                item.anomaly_id,
                item.observation_id,
            ),
        )
    )


def _source_states(observations: Sequence[_Observation]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {observation.report_state for observation in observations},
            key=_REPORT_STATE_ORDER.__getitem__,
        )
    )


def _lineage_sha256(
    *,
    anomaly_id: str,
    entity_scope: EntityScope,
    observations: Sequence[_Observation],
) -> str:
    ordered = sorted(
        observations,
        key=lambda item: (
            item.available_at,
            item.period_index,
            item.observation_id,
        ),
    )
    payload = {
        "contract_version": ANOMALY_HISTORY_CONTRACT_VERSION,
        "anomaly_id": anomaly_id,
        "entity_scope": entity_scope,
        "observations": [
            {
                "observation_id": observation.observation_id,
                "report_id": observation.source_report_id,
                "available_at": _utc_text(observation.available_at),
            }
            for observation in ordered
        ],
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _make_state(
    *,
    period: _ReportPeriod,
    periods: tuple[_ReportPeriod, ...],
    previous_period: _ReportPeriod | None,
    previous_period_consecutive: bool,
    anomaly_id: str,
    identity_complete: bool,
    entity_scope: EntityScope,
    history: Sequence[_Observation],
    current: Sequence[_Observation],
    system_first_seen: bool,
    system_not_continued: bool,
    system_relisted: bool,
) -> AnomalyState:
    if not history:
        raise AssertionError("a state requires at least one causal observation")
    by_period: dict[int, list[_Observation]] = defaultdict(list)
    for observation in history:
        by_period[observation.period_index].append(observation)
    latest_period_index = max(by_period)
    latest_group = tuple(by_period[latest_period_index])
    latest_canonical = max(
        latest_group,
        key=lambda item: (item.available_at, item.observation_id),
    )
    structured_group = tuple(current) if current else latest_group
    first = min(
        history,
        key=lambda item: (item.available_at, item.period_index, item.observation_id),
    )
    station_id, station_id_null_reason = _resolve_duplicate_values(
        tuple(item.station_id for item in structured_group)
    )
    longitude, longitude_null_reason = _resolve_duplicate_values(
        tuple(item.longitude for item in structured_group)
    )
    latitude, latitude_null_reason = _resolve_duplicate_values(
        tuple(item.latitude for item in structured_group)
    )
    discipline, discipline_null_reason = _resolve_duplicate_values(
        tuple(item.discipline for item in structured_group)
    )
    measurement, measurement_null_reason = _resolve_duplicate_values(
        tuple(item.measurement for item in structured_group)
    )
    start_time, start_time_null_reason = _resolve_duplicate_values(
        tuple(item.start_time for item in structured_group)
    )
    if longitude is not None and latitude is not None:
        spatial_eligible = True
        spatial_exclusion_reason = None
    elif longitude is None and latitude is None:
        spatial_eligible = False
        spatial_exclusion_reason = "longitude_and_latitude_unavailable"
    elif longitude is None:
        spatial_eligible = False
        spatial_exclusion_reason = "longitude_unavailable"
    else:
        spatial_eligible = False
        spatial_exclusion_reason = "latitude_unavailable"

    current_reporting_station_ids = tuple(
        sorted({item.station_id for item in current if item.is_listed})
    )
    current_reporting_disciplines = tuple(
        sorted({item.discipline for item in current if item.is_listed})
    )
    current_reporting_measurements = tuple(
        sorted(
            {
                item.measurement
                for item in current
                if item.is_listed and item.measurement is not None
            }
        )
    )
    known_station_ids = tuple(sorted({item.station_id for item in history if item.is_listed}))
    known_disciplines = tuple(sorted({item.discipline for item in history if item.is_listed}))
    known_measurements = tuple(
        sorted(
            {
                item.measurement
                for item in history
                if item.is_listed and item.measurement is not None
            }
        )
    )

    reliability_flags, reliability_grade, reliability_weight = _reliability(
        structured_group,
        identity_complete=identity_complete,
    )
    current_states = _source_states(current)
    latest_states = _source_states(latest_group)
    current_ids = tuple(sorted(item.observation_id for item in current))
    latest_ids = tuple(sorted(item.observation_id for item in latest_group))
    ordered_lineage = tuple(
        sorted(
            history,
            key=lambda item: (
                item.available_at,
                item.period_index,
                item.observation_id,
            ),
        )
    )
    lineage_observation_ids = tuple(item.observation_id for item in ordered_lineage)
    lineage_source_report_ids = tuple(
        dict.fromkeys(item.source_report_id for item in ordered_lineage)
    )
    lineage_max = max(item.available_at for item in history)
    explicit_end_known, reported_end_time, reported_end_time_null_reason = _resolve_end_time(
        structured_group
    )
    if (
        explicit_end_known
        and reported_end_time is not None
        and reported_end_time.astimezone(UTC) > period.available_at
    ):
        explicit_end_known = False
        reported_end_time = None
        reported_end_time_null_reason = "end_after_issue_time"
    if explicit_end_known and start_time is None:
        explicit_end_known = False
        reported_end_time = None
        reported_end_time_null_reason = "start_time_unavailable_for_end_validation"
    if (
        explicit_end_known
        and reported_end_time is not None
        and start_time is not None
        and reported_end_time < start_time
    ):
        explicit_end_known = False
        reported_end_time = None
        reported_end_time_null_reason = "end_before_start_time"

    left_truncated = any(item.period_index == 0 for item in history)
    first_known_group = tuple(
        item
        for item in history
        if item.available_at == first.available_at and item.period_index == first.period_index
    )
    first_known_start, _ = _resolve_duplicate_values(
        tuple(item.start_time for item in first_known_group)
    )
    first_known_period = periods[first.period_index]
    late_entry_or_gap_unknown = bool(
        not left_truncated
        and first_known_period.index > 0
        and first_known_start is not None
        and first_known_start.date() < periods[first_known_period.index - 1].report_date
    )

    if not identity_complete:
        age_days = None
        age_days_null_reason = "temporary_entity_longitudinal_excluded"
    elif start_time is None:
        age_days = None
        age_days_null_reason = "start_time_unavailable"
    else:
        age_delta = period.available_at - start_time.astimezone(UTC)
        if age_delta.total_seconds() < 0.0:
            age_days = None
            age_days_null_reason = "start_time_after_issue_time"
        else:
            age_days = _days(age_delta)
            age_days_null_reason = None

    if not identity_complete:
        known_duration_days = None
        known_duration_days_null_reason = "temporary_entity_longitudinal_excluded"
    elif start_time is None:
        known_duration_days = None
        known_duration_days_null_reason = "start_time_unavailable"
    elif not explicit_end_known or reported_end_time is None:
        known_duration_days = None
        known_duration_days_null_reason = reported_end_time_null_reason or "explicit_end_not_known"
    else:
        duration_delta = reported_end_time - start_time
        if duration_delta.total_seconds() < 0.0:
            known_duration_days = None
            known_duration_days_null_reason = "end_before_start_time"
        else:
            known_duration_days = _days(duration_delta)
            known_duration_days_null_reason = None

    state_id = stable_token(
        "anomaly_state",
        ANOMALY_HISTORY_CONTRACT_VERSION,
        _utc_text(period.available_at),
        period.report_id,
        anomaly_id,
        entity_scope,
        "" if identity_complete else latest_canonical.observation_id,
        length=32,
    )
    return AnomalyState(
        state_id=state_id,
        contract_version=ANOMALY_HISTORY_CONTRACT_VERSION,
        state_row_kind="entity_state",
        issue_time_utc=period.available_at,
        issue_report_id=period.report_id,
        issue_report_date=period.report_date,
        issue_report_year=period.report_year,
        issue_report_period=period.report_period,
        row_count=period.row_count,
        row_report_date_mismatch_count=period.row_report_date_mismatch_count,
        row_report_date_before_count=period.row_report_date_before_count,
        row_report_date_after_count=period.row_report_date_after_count,
        deformation_row_count=period.deformation_row_count,
        fluid_row_count=period.fluid_row_count,
        electromagnetic_row_count=period.electromagnetic_row_count,
        cross_fault_row_count=period.cross_fault_row_count,
        previous_issue_report_id=(None if previous_period is None else previous_period.report_id),
        previous_period_consecutive=previous_period_consecutive,
        anomaly_id=anomaly_id,
        identity_complete=identity_complete,
        entity_scope=entity_scope,
        station_id=station_id,
        station_id_null_reason=station_id_null_reason,
        longitude=longitude,
        longitude_null_reason=longitude_null_reason,
        latitude=latitude,
        latitude_null_reason=latitude_null_reason,
        discipline=discipline,
        discipline_null_reason=discipline_null_reason,
        measurement=measurement,
        measurement_null_reason=measurement_null_reason,
        start_time=start_time,
        start_time_null_reason=start_time_null_reason,
        spatial_eligible=spatial_eligible,
        spatial_exclusion_reason=spatial_exclusion_reason,
        current_reporting_station_ids=current_reporting_station_ids,
        current_reporting_disciplines=current_reporting_disciplines,
        current_reporting_measurements=current_reporting_measurements,
        known_station_ids=known_station_ids,
        known_disciplines=known_disciplines,
        known_measurements=known_measurements,
        current_report_listed=any(item.is_listed for item in current),
        source_new="新增" in current_states,
        source_continued="持续" in current_states,
        source_cancelled="取消" in current_states,
        current_source_report_states=current_states,
        latest_source_report_states=latest_states,
        system_first_seen=system_first_seen,
        system_not_continued=system_not_continued,
        system_relisted=system_relisted,
        left_truncated=left_truncated,
        late_entry_or_gap_unknown=late_entry_or_gap_unknown,
        explicit_end_known=explicit_end_known,
        right_censored=not explicit_end_known,
        reported_end_time=reported_end_time,
        reported_end_time_null_reason=reported_end_time_null_reason,
        age_days=age_days,
        age_days_null_reason=age_days_null_reason,
        known_duration_days=known_duration_days,
        known_duration_days_null_reason=known_duration_days_null_reason,
        reliability_flags=reliability_flags,
        reliability_grade=reliability_grade,
        reliability_weight=reliability_weight,
        first_available_at_utc=first.available_at,
        latest_available_at_utc=latest_canonical.available_at,
        first_source_report_id=first.source_report_id,
        latest_source_report_id=latest_canonical.source_report_id,
        latest_source_report_date=latest_canonical.source_report_date,
        current_observation_ids=current_ids,
        latest_observation_ids=latest_ids,
        latest_observation_id=latest_canonical.observation_id,
        lineage_observation_ids=lineage_observation_ids,
        lineage_source_report_ids=lineage_source_report_ids,
        lineage_observation_count=len(history),
        lineage_max_available_at_utc=lineage_max,
        lineage_sha256=_lineage_sha256(
            anomaly_id=anomaly_id,
            entity_scope=entity_scope,
            observations=history,
        ),
        source_observation_ids=lineage_observation_ids,
        source_report_ids=lineage_source_report_ids,
        max_source_available_at=lineage_max,
    )


def _make_report_period_summary_state(
    period: _ReportPeriod,
    previous_period: _ReportPeriod | None,
) -> AnomalyState:
    not_applicable = "not_applicable_report_period_summary"
    lineage_sha256 = hashlib.sha256(
        canonical_json_bytes(
            {
                "contract_version": ANOMALY_HISTORY_CONTRACT_VERSION,
                "state_row_kind": "report_period_summary",
                "report_id": period.report_id,
                "available_at": _utc_text(period.available_at),
            }
        )
    ).hexdigest()
    return AnomalyState(
        state_id=stable_token(
            "anomaly_report_period_state",
            ANOMALY_HISTORY_CONTRACT_VERSION,
            period.report_id,
            _utc_text(period.available_at),
            length=32,
        ),
        contract_version=ANOMALY_HISTORY_CONTRACT_VERSION,
        state_row_kind="report_period_summary",
        issue_time_utc=period.available_at,
        issue_report_id=period.report_id,
        issue_report_date=period.report_date,
        issue_report_year=period.report_year,
        issue_report_period=period.report_period,
        row_count=period.row_count,
        row_report_date_mismatch_count=period.row_report_date_mismatch_count,
        row_report_date_before_count=period.row_report_date_before_count,
        row_report_date_after_count=period.row_report_date_after_count,
        deformation_row_count=period.deformation_row_count,
        fluid_row_count=period.fluid_row_count,
        electromagnetic_row_count=period.electromagnetic_row_count,
        cross_fault_row_count=period.cross_fault_row_count,
        previous_issue_report_id=(None if previous_period is None else previous_period.report_id),
        previous_period_consecutive=period.previous_period_consecutive,
        anomaly_id="__report_period_summary__",
        identity_complete=False,
        entity_scope="report_period_summary",
        station_id=None,
        station_id_null_reason=not_applicable,
        longitude=None,
        longitude_null_reason=not_applicable,
        latitude=None,
        latitude_null_reason=not_applicable,
        discipline=None,
        discipline_null_reason=not_applicable,
        measurement=None,
        measurement_null_reason=not_applicable,
        start_time=None,
        start_time_null_reason=not_applicable,
        spatial_eligible=False,
        spatial_exclusion_reason=not_applicable,
        current_reporting_station_ids=(),
        current_reporting_disciplines=(),
        current_reporting_measurements=(),
        known_station_ids=(),
        known_disciplines=(),
        known_measurements=(),
        current_report_listed=False,
        source_new=False,
        source_continued=False,
        source_cancelled=False,
        current_source_report_states=(),
        latest_source_report_states=(),
        system_first_seen=False,
        system_not_continued=False,
        system_relisted=False,
        left_truncated=False,
        late_entry_or_gap_unknown=False,
        explicit_end_known=False,
        right_censored=False,
        reported_end_time=None,
        reported_end_time_null_reason=not_applicable,
        age_days=None,
        age_days_null_reason=not_applicable,
        known_duration_days=None,
        known_duration_days_null_reason=not_applicable,
        reliability_flags=(),
        reliability_grade="excluded",
        reliability_weight=0.0,
        first_available_at_utc=period.available_at,
        latest_available_at_utc=period.available_at,
        first_source_report_id=period.report_id,
        latest_source_report_id=period.report_id,
        latest_source_report_date=period.report_date,
        current_observation_ids=(),
        latest_observation_ids=(),
        latest_observation_id=None,
        lineage_observation_ids=(),
        lineage_source_report_ids=(period.report_id,),
        lineage_observation_count=0,
        lineage_max_available_at_utc=period.available_at,
        lineage_sha256=lineage_sha256,
        source_observation_ids=(),
        source_report_ids=(period.report_id,),
        max_source_available_at=period.available_at,
    )


def build_anomaly_state_history(
    observations: Sequence[Mapping[str, object]],
    report_periods: Sequence[Mapping[str, object]],
    *,
    expected_report_period_count: int | None = None,
) -> tuple[AnomalyState, ...]:
    """Build issue-by-entity state using only records causally available at each issue.

    The public API intentionally accepts only anomaly observations and anomaly report
    periods.  Every actual issue receives one ``report_period_summary`` row, including
    zero-entity issues.  Complete entities persist after their first causal observation.
    Incomplete identities are report-local and can never be linked across periods.
    """

    periods = _parse_report_periods(
        report_periods,
        expected_report_period_count=expected_report_period_count,
    )
    parsed_observations = _parse_observations(observations, periods)
    observations_by_period: dict[int, list[_Observation]] = defaultdict(list)
    for observation in parsed_observations:
        observations_by_period[observation.period_index].append(observation)

    complete_history: dict[str, list[_Observation]] = defaultdict(list)
    previous_current_complete: set[str] = set()
    observation_pointer = 0
    output: list[AnomalyState] = []

    for period_index, period in enumerate(periods):
        previous_period = periods[period_index - 1] if period_index else None
        previous_period_consecutive = period.previous_period_consecutive
        known_before_issue = set(complete_history)

        while (
            observation_pointer < len(parsed_observations)
            and parsed_observations[observation_pointer].available_at <= period.available_at
        ):
            observation = parsed_observations[observation_pointer]
            if observation.identity_complete:
                complete_history[observation.anomaly_id].append(observation)
            observation_pointer += 1

        current_available = tuple(
            observation
            for observation in observations_by_period.get(period.index, ())
            if observation.available_at <= period.available_at
        )
        current_complete: dict[str, list[_Observation]] = defaultdict(list)
        current_incomplete: list[_Observation] = []
        for observation in current_available:
            if observation.identity_complete:
                current_complete[observation.anomaly_id].append(observation)
            else:
                current_incomplete.append(observation)

        issue_states: list[AnomalyState] = [
            _make_report_period_summary_state(period, previous_period)
        ]
        for anomaly_id in sorted(complete_history):
            current = tuple(current_complete.get(anomaly_id, ()))
            issue_states.append(
                _make_state(
                    period=period,
                    periods=periods,
                    previous_period=previous_period,
                    previous_period_consecutive=previous_period_consecutive,
                    anomaly_id=anomaly_id,
                    identity_complete=True,
                    entity_scope="persistent_complete_entity",
                    history=tuple(complete_history[anomaly_id]),
                    current=current,
                    system_first_seen=anomaly_id not in known_before_issue,
                    system_not_continued=(
                        previous_period_consecutive
                        and anomaly_id in previous_current_complete
                        and not current
                    ),
                    system_relisted=(
                        previous_period_consecutive
                        and bool(current)
                        and anomaly_id in known_before_issue
                        and anomaly_id not in previous_current_complete
                    ),
                )
            )

        for observation in sorted(
            current_incomplete,
            key=lambda item: (item.anomaly_id, item.observation_id),
        ):
            current = (observation,)
            issue_states.append(
                _make_state(
                    period=period,
                    periods=periods,
                    previous_period=previous_period,
                    previous_period_consecutive=previous_period_consecutive,
                    anomaly_id=observation.anomaly_id,
                    identity_complete=False,
                    entity_scope="report_local_incomplete_entity",
                    history=current,
                    current=current,
                    system_first_seen=True,
                    system_not_continued=False,
                    system_relisted=False,
                )
            )

        issue_states.sort(
            key=lambda state: (
                state.state_row_kind,
                state.anomaly_id,
                state.entity_scope,
                state.state_id,
            )
        )
        output.extend(issue_states)
        previous_current_complete = set(current_complete)

    return tuple(output)


def state_records(states: Sequence[AnomalyState]) -> tuple[dict[str, object], ...]:
    """Convert an in-memory state sequence to Arrow-ready records."""

    return tuple(state.to_record() for state in states)


_STATE_TUPLE_FIELDS = frozenset(
    {
        "current_reporting_station_ids",
        "current_reporting_disciplines",
        "current_reporting_measurements",
        "known_station_ids",
        "known_disciplines",
        "known_measurements",
        "current_source_report_states",
        "latest_source_report_states",
        "reliability_flags",
        "current_observation_ids",
        "latest_observation_ids",
        "lineage_observation_ids",
        "lineage_source_report_ids",
        "source_observation_ids",
        "source_report_ids",
    }
)


def states_from_records(records: Sequence[Mapping[str, object]]) -> tuple[AnomalyState, ...]:
    """Rehydrate state rows read from the sealed local Parquet for replay audits."""

    expected_fields = {field.name for field in fields(AnomalyState)}
    output: list[AnomalyState] = []
    for index, record in enumerate(records):
        if set(record) != expected_fields:
            missing = sorted(expected_fields - set(record))
            extras = sorted(set(record) - expected_fields)
            raise ValueError(
                f"state replay record {index} differs from the contract: "
                f"missing={missing}, extras={extras}"
            )
        payload: dict[str, Any] = dict(record)
        for name in _STATE_TUPLE_FIELDS:
            value = payload[name]
            if not isinstance(value, list | tuple):
                raise ValueError(f"state replay field must be a sequence: {name}")
            payload[name] = tuple(value)
        if payload["state_row_kind"] not in {"entity_state", "report_period_summary"}:
            raise ValueError("invalid state_row_kind in replay record")
        if payload["entity_scope"] not in {
            "persistent_complete_entity",
            "report_local_incomplete_entity",
            "report_period_summary",
        }:
            raise ValueError("invalid entity_scope in replay record")
        if payload["reliability_grade"] not in {"high", "cautious", "excluded"}:
            raise ValueError("invalid reliability_grade in replay record")
        state = AnomalyState(**payload)
        if state.to_record() != dict(record):
            raise ValueError("state replay round-trip changed a contract value")
        output.append(state)
    return tuple(output)


__all__ = [
    "AnomalyState",
    "EntityScope",
    "ReliabilityGrade",
    "StateRowKind",
    "build_anomaly_state_history",
    "state_records",
    "states_from_records",
]
