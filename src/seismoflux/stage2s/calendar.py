"""Strict Stage 2S fold calendars and causal catalog memberships."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta, timezone
from itertools import pairwise
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from seismoflux.background.artifacts import canonical_json_bytes
from seismoflux.stage2s.catalog import Stage2SEarthquakeCatalog
from seismoflux.stage2s.seals import SealedRecord

IntArray = NDArray[np.int64]
ExposureRole = Literal["fit", "assessment"]
RecentComponentId = Literal["R", "RP"]
AdditionalDelayDays = Literal[0, 1, 7]

FROZEN_FOLD_MANIFEST_SHA256 = "c3e2444e8892addd03d4c57526c007e2a861137dac50d5abe2e53bac004456e6"
FOLD_ORDER = (1, 2, 3)
FIT_HORIZON_DAYS = 7
ASSESSMENT_HORIZONS_DAYS = (7, 30, 90)
M4_MINIMUM = 4.0
M5_6_MINIMUM = 5.0
M5_6_MAXIMUM_EXCLUSIVE = 6.0
_BEIJING_FIXED_OFFSET = timezone(timedelta(hours=8), name="UTC+08:00")
_EPOCH_UTC = datetime(1970, 1, 1, tzinfo=UTC)
_MICROSECONDS_PER_DAY = 86_400_000_000


def _exact_keys(value: Mapping[str, object], expected: set[str], *, label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} keys differ; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _mapping(value: object, *, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} keys must be strings")
    return cast(dict[str, object], value)


def _list(value: object, *, label: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a JSON array")
    return value


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not (normalized := value.strip()):
        raise ValueError(f"{label} must be a non-empty string")
    return normalized


def _integer(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    return value


def _required_bool(value: object, expected: bool, *, label: str) -> None:
    if not isinstance(value, bool) or value is not expected:
        raise ValueError(f"{label} must be {expected}")


def _sha256_hex(value: object, *, label: str) -> str:
    normalized = _text(value, label=label)
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")
    return normalized


def _local_date(value: object, *, label: str) -> date:
    text_value = _text(value, label=label)
    try:
        parsed = date.fromisoformat(text_value)
    except ValueError as error:
        raise ValueError(f"{label} must be an ISO-8601 calendar date") from error
    if parsed.isoformat() != text_value:
        raise ValueError(f"{label} must use exact YYYY-MM-DD form")
    return parsed


def _utc_datetime(value: object, *, label: str) -> datetime:
    text_value = _text(value, label=label)
    try:
        parsed = datetime.strptime(text_value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError as error:
        raise ValueError(f"{label} must use exact second-resolution UTC Z form") from error
    return parsed


def _local_midnight_utc(value: date) -> datetime:
    return datetime.combine(value, time.min, _BEIJING_FIXED_OFFSET).astimezone(UTC)


def _datetime_us(value: datetime, *, label: str) -> int:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{label} must be a timezone-aware datetime")
    normalized = value.astimezone(UTC)
    delta = normalized - _EPOCH_UTC
    return delta.days * _MICROSECONDS_PER_DAY + delta.seconds * 1_000_000 + delta.microseconds


def _read_only_indices(value: object, *, catalog_length: int) -> IntArray:
    raw = np.asarray(value)
    if raw.ndim != 1:
        raise ValueError("event_indices must be one-dimensional")
    if raw.size == 0:
        result = np.asarray((), dtype=np.int64)
    else:
        if not np.issubdtype(raw.dtype, np.integer):
            raise ValueError("event_indices must contain only integers")
        result = np.asarray(raw, dtype=np.int64)
    if np.any(result < 0) or np.any(result >= catalog_length):
        raise ValueError("event_indices lie outside the source catalog")
    if result.size and not np.all(result[:-1] < result[1:]):
        raise ValueError("event_indices must be unique and in source-catalog order")
    owned = np.array(result, dtype=np.int64, copy=True, order="C")
    owned.setflags(write=False)
    return owned


def _read_only_issue_times(value: object, *, length: int) -> IntArray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.size != length:
        raise ValueError("assigned_issue_time_us must match event_indices")
    if raw.size == 0:
        result = np.asarray((), dtype=np.int64)
    else:
        if not np.issubdtype(raw.dtype, np.integer):
            raise ValueError("assigned_issue_time_us must contain only integers")
        result = np.asarray(raw, dtype=np.int64)
    owned = np.array(result, dtype=np.int64, copy=True, order="C")
    owned.setflags(write=False)
    return owned


@dataclass(frozen=True, slots=True)
class TargetExposure:
    """One frozen half-open/closed target interval ``(T, T+h]``."""

    fold_index: int
    role: ExposureRole
    issue_date_local: date
    issue_time_utc: datetime
    horizon_days: int
    target_end_inclusive_utc: datetime

    def __post_init__(self) -> None:
        if self.fold_index not in FOLD_ORDER:
            raise ValueError("fold_index must be 1, 2, or 3")
        if self.role not in {"fit", "assessment"}:
            raise ValueError("exposure role must be fit or assessment")
        if not isinstance(self.issue_date_local, date):
            raise TypeError("issue_date_local must be a date")
        expected_issue = _local_midnight_utc(self.issue_date_local)
        issue = self.issue_time_utc
        if not isinstance(issue, datetime) or issue.tzinfo is None:
            raise ValueError("issue_time_utc must be timezone-aware")
        issue = issue.astimezone(UTC)
        if issue != expected_issue:
            raise ValueError("issue_time_utc must equal local UTC+08 midnight")
        if (
            not isinstance(self.horizon_days, int)
            or isinstance(self.horizon_days, bool)
            or (self.role == "fit" and self.horizon_days != FIT_HORIZON_DAYS)
            or (self.role == "assessment" and self.horizon_days not in ASSESSMENT_HORIZONS_DAYS)
        ):
            raise ValueError("horizon_days is not allowed for this exposure role")
        target_end = self.target_end_inclusive_utc
        if not isinstance(target_end, datetime) or target_end.tzinfo is None:
            raise ValueError("target_end_inclusive_utc must be timezone-aware")
        target_end = target_end.astimezone(UTC)
        if target_end != issue + timedelta(days=self.horizon_days):
            raise ValueError("target end must equal issue time plus the exact horizon")
        object.__setattr__(self, "issue_time_utc", issue)
        object.__setattr__(self, "target_end_inclusive_utc", target_end)

    @property
    def issue_time_us(self) -> int:
        return _datetime_us(self.issue_time_utc, label="issue_time_utc")

    @property
    def target_end_inclusive_us(self) -> int:
        return _datetime_us(
            self.target_end_inclusive_utc,
            label="target_end_inclusive_utc",
        )


@dataclass(frozen=True, slots=True)
class AssessmentIssue:
    """One unique issue date with all scheduled horizons processed together."""

    fold_index: int
    issue_date_local: date
    issue_time_utc: datetime
    horizons_days: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.fold_index not in FOLD_ORDER:
            raise ValueError("fold_index must be 1, 2, or 3")
        issue = self.issue_time_utc
        if not isinstance(issue, datetime) or issue.tzinfo is None:
            raise ValueError("issue_time_utc must be timezone-aware")
        issue = issue.astimezone(UTC)
        if issue != _local_midnight_utc(self.issue_date_local):
            raise ValueError("assessment issue must equal local UTC+08 midnight")
        horizons = tuple(self.horizons_days)
        if (
            not horizons
            or len(set(horizons)) != len(horizons)
            or horizons != tuple(value for value in ASSESSMENT_HORIZONS_DAYS if value in horizons)
        ):
            raise ValueError("assessment issue horizons must follow frozen 7/30/90 order")
        object.__setattr__(self, "issue_time_utc", issue)
        object.__setattr__(self, "horizons_days", horizons)


@dataclass(frozen=True, slots=True)
class FoldCalendar:
    """One strictly parsed fold with fit and assessment exposure schedules."""

    fold_index: int
    fit_scope_id: str
    fit_exposures: tuple[TargetExposure, ...]
    fit_target_end_inclusive_utc: datetime
    assessment_start_exclusive_utc: datetime
    assessment_end_inclusive_utc: datetime
    assessment_exposures: tuple[TargetExposure, ...]
    assessment_issues: tuple[AssessmentIssue, ...]

    def __post_init__(self) -> None:
        if self.fold_index not in FOLD_ORDER:
            raise ValueError("fold_index must be 1, 2, or 3")
        if self.fit_scope_id != f"stage2s-development-fold-{self.fold_index}":
            raise ValueError("fit_scope_id does not match the frozen fold index")
        fit = tuple(self.fit_exposures)
        if not fit or any(item.fold_index != self.fold_index or item.role != "fit" for item in fit):
            raise ValueError("fit_exposures must be non-empty fit intervals for this fold")
        fit_dates = tuple(item.issue_date_local for item in fit)
        if fit_dates != tuple(sorted(set(fit_dates))):
            raise ValueError("fit issue dates must be unique and ascending")
        if any(
            later.issue_time_utc != earlier.target_end_inclusive_utc
            for earlier, later in pairwise(fit)
        ):
            raise ValueError("fit h007 exposures must be adjacent and non-overlapping")
        fit_end = self.fit_target_end_inclusive_utc.astimezone(UTC)
        if fit_end != fit[-1].target_end_inclusive_utc:
            raise ValueError("fit target end must equal the last h007 interval end")
        assessment_start = self.assessment_start_exclusive_utc.astimezone(UTC)
        assessment_end = self.assessment_end_inclusive_utc.astimezone(UTC)
        if fit_end >= assessment_start:
            raise ValueError("fit target end must be strictly before assessment start")
        if assessment_end - assessment_start != timedelta(days=90):
            raise ValueError("assessment band must be exactly 90 days")
        assessment = tuple(self.assessment_exposures)
        if not assessment or any(
            item.fold_index != self.fold_index or item.role != "assessment" for item in assessment
        ):
            raise ValueError("assessment_exposures must belong to this fold")
        expected_order = tuple(
            sorted(
                assessment,
                key=lambda item: (
                    item.issue_time_utc,
                    ASSESSMENT_HORIZONS_DAYS.index(item.horizon_days),
                ),
            )
        )
        if assessment != expected_order or len(
            {(item.issue_time_utc, item.horizon_days) for item in assessment}
        ) != len(assessment):
            raise ValueError("assessment issue/horizon pairs must be unique and ordered")
        for horizon in ASSESSMENT_HORIZONS_DAYS:
            exposures = tuple(item for item in assessment if item.horizon_days == horizon)
            if not exposures:
                raise ValueError("every frozen assessment horizon must contain an issue")
            if any(
                later.issue_time_utc < earlier.target_end_inclusive_utc
                for earlier, later in pairwise(exposures)
            ):
                raise ValueError("issues within a fold/horizon must have disjoint targets")
        if any(
            item.issue_time_utc < assessment_start or item.target_end_inclusive_utc > assessment_end
            for item in assessment
        ):
            raise ValueError("assessment exposure lies outside its frozen 90-day band")
        issues = tuple(self.assessment_issues)
        if not issues:
            raise ValueError("assessment_issues must not be empty")
        grouped: dict[datetime, tuple[int, ...]] = {}
        for item in assessment:
            grouped.setdefault(item.issue_time_utc, ())
            grouped[item.issue_time_utc] = (
                *grouped[item.issue_time_utc],
                item.horizon_days,
            )
        expected_issues = tuple(
            AssessmentIssue(
                fold_index=self.fold_index,
                issue_date_local=issue_time.astimezone(_BEIJING_FIXED_OFFSET).date(),
                issue_time_utc=issue_time,
                horizons_days=horizons,
            )
            for issue_time, horizons in sorted(grouped.items())
        )
        if issues != expected_issues:
            raise ValueError("assessment_issues must group all horizons by unique issue date")
        object.__setattr__(self, "fit_exposures", fit)
        object.__setattr__(self, "fit_target_end_inclusive_utc", fit_end)
        object.__setattr__(self, "assessment_start_exclusive_utc", assessment_start)
        object.__setattr__(self, "assessment_end_inclusive_utc", assessment_end)
        object.__setattr__(self, "assessment_exposures", assessment)
        object.__setattr__(self, "assessment_issues", issues)


@dataclass(frozen=True, slots=True)
class Stage2SFoldCalendar:
    """The exact 1→2→3 fold schedule bound to one manifest hash."""

    manifest_sha256: str
    folds: tuple[FoldCalendar, FoldCalendar, FoldCalendar]

    def __post_init__(self) -> None:
        digest = _sha256_hex(self.manifest_sha256, label="manifest_sha256")
        folds = tuple(self.folds)
        if len(folds) != 3 or tuple(item.fold_index for item in folds) != FOLD_ORDER:
            raise ValueError("folds must be ordered exactly as 1, 2, and 3")
        for previous, current in pairwise(folds):
            previous_fit_dates = tuple(item.issue_date_local for item in previous.fit_exposures)
            current_fit_dates = tuple(item.issue_date_local for item in current.fit_exposures)
            if (
                len(current_fit_dates) <= len(previous_fit_dates)
                or current_fit_dates[: len(previous_fit_dates)] != previous_fit_dates
            ):
                raise ValueError("fit h007 exposures must expand by preserving the prior prefix")
            if previous.assessment_end_inclusive_utc >= current.assessment_start_exclusive_utc:
                raise ValueError("fold assessment target bands must be mutually disjoint")
        object.__setattr__(self, "manifest_sha256", digest)
        object.__setattr__(self, "folds", folds)

    def fold(self, fold_index: int) -> FoldCalendar:
        if fold_index not in FOLD_ORDER:
            raise KeyError("fold_index must be 1, 2, or 3")
        return self.folds[fold_index - 1]


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _parse_json(payload: bytes) -> dict[str, object]:
    if type(payload) is not bytes:
        raise TypeError("manifest payload must be one immutable bytes object")
    try:
        decoded = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("fold manifest must be UTF-8") from error
    try:
        value = json.loads(
            decoded,
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value is forbidden: {token}")
            ),
        )
    except json.JSONDecodeError as error:
        raise ValueError("fold manifest is not valid JSON") from error
    return _mapping(value, label="fold manifest")


def _parse_local_dates(value: object, *, label: str) -> tuple[date, ...]:
    dates = tuple(_local_date(item, label=f"{label} item") for item in _list(value, label=label))
    if not dates or dates != tuple(sorted(set(dates))):
        raise ValueError(f"{label} must contain unique ascending dates")
    return dates


def _parse_fold(value: object) -> FoldCalendar:
    raw = _mapping(value, label="fold")
    _exact_keys(
        raw,
        {
            "fold_index",
            "fit_scope_id",
            "fit_issue_dates_local_h007",
            "fit_target_end_inclusive_utc",
            "assessment_band",
            "assessment_issue_dates_local_by_horizon",
        },
        label="fold",
    )
    fold_index = _integer(raw["fold_index"], label="fold_index")
    fit_scope_id = _text(raw["fit_scope_id"], label="fit_scope_id")
    fit_dates = _parse_local_dates(
        raw["fit_issue_dates_local_h007"],
        label="fit_issue_dates_local_h007",
    )
    fit_exposures = tuple(
        TargetExposure(
            fold_index=fold_index,
            role="fit",
            issue_date_local=issue_date,
            issue_time_utc=_local_midnight_utc(issue_date),
            horizon_days=FIT_HORIZON_DAYS,
            target_end_inclusive_utc=_local_midnight_utc(issue_date)
            + timedelta(days=FIT_HORIZON_DAYS),
        )
        for issue_date in fit_dates
    )
    fit_target_end = _utc_datetime(
        raw["fit_target_end_inclusive_utc"],
        label="fit_target_end_inclusive_utc",
    )
    assessment_band = _mapping(raw["assessment_band"], label="assessment_band")
    _exact_keys(
        assessment_band,
        {
            "start_exclusive_local",
            "start_exclusive_utc",
            "end_inclusive_local",
            "end_inclusive_utc",
        },
        label="assessment_band",
    )
    assessment_start_local = _local_date(
        assessment_band["start_exclusive_local"],
        label="assessment start_exclusive_local",
    )
    assessment_end_local = _local_date(
        assessment_band["end_inclusive_local"],
        label="assessment end_inclusive_local",
    )
    assessment_start = _utc_datetime(
        assessment_band["start_exclusive_utc"],
        label="assessment start_exclusive_utc",
    )
    assessment_end = _utc_datetime(
        assessment_band["end_inclusive_utc"],
        label="assessment end_inclusive_utc",
    )
    if assessment_start != _local_midnight_utc(assessment_start_local):
        raise ValueError("assessment start local and UTC fields disagree")
    if assessment_end != _local_midnight_utc(assessment_end_local):
        raise ValueError("assessment end local and UTC fields disagree")
    by_horizon = _mapping(
        raw["assessment_issue_dates_local_by_horizon"],
        label="assessment_issue_dates_local_by_horizon",
    )
    _exact_keys(by_horizon, {"7", "30", "90"}, label="assessment horizons")
    assessment_exposures_unsorted: list[TargetExposure] = []
    for horizon in ASSESSMENT_HORIZONS_DAYS:
        issue_dates = _parse_local_dates(
            by_horizon[str(horizon)],
            label=f"assessment horizon {horizon}",
        )
        assessment_exposures_unsorted.extend(
            TargetExposure(
                fold_index=fold_index,
                role="assessment",
                issue_date_local=issue_date,
                issue_time_utc=_local_midnight_utc(issue_date),
                horizon_days=horizon,
                target_end_inclusive_utc=_local_midnight_utc(issue_date) + timedelta(days=horizon),
            )
            for issue_date in issue_dates
        )
    assessment_exposures = tuple(
        sorted(
            assessment_exposures_unsorted,
            key=lambda item: (
                item.issue_time_utc,
                ASSESSMENT_HORIZONS_DAYS.index(item.horizon_days),
            ),
        )
    )
    grouped: dict[datetime, list[int]] = {}
    for exposure in assessment_exposures:
        grouped.setdefault(exposure.issue_time_utc, []).append(exposure.horizon_days)
    assessment_issues = tuple(
        AssessmentIssue(
            fold_index=fold_index,
            issue_date_local=issue_time.astimezone(_BEIJING_FIXED_OFFSET).date(),
            issue_time_utc=issue_time,
            horizons_days=tuple(horizons),
        )
        for issue_time, horizons in sorted(grouped.items())
    )
    return FoldCalendar(
        fold_index=fold_index,
        fit_scope_id=fit_scope_id,
        fit_exposures=fit_exposures,
        fit_target_end_inclusive_utc=fit_target_end,
        assessment_start_exclusive_utc=assessment_start,
        assessment_end_inclusive_utc=assessment_end,
        assessment_exposures=assessment_exposures,
        assessment_issues=assessment_issues,
    )


def parse_fold_manifest_bytes(
    payload: bytes,
    *,
    expected_sha256: str,
) -> Stage2SFoldCalendar:
    """Strictly parse a caller-supplied, hash-bound Stage 2S fold manifest."""

    expected_digest = _sha256_hex(expected_sha256, label="expected_sha256")
    actual_digest = hashlib.sha256(payload).hexdigest()
    if actual_digest != expected_digest:
        raise ValueError("fold manifest SHA-256 mismatch")
    raw = _parse_json(payload)
    _exact_keys(
        raw,
        {
            "schema_version",
            "protocol_version",
            "experiment_id",
            "status",
            "source_design",
            "issue_semantics",
            "target_bands_mutually_disjoint",
            "rolling_rule",
            "folds",
            "security",
        },
        label="fold manifest",
    )
    if _integer(raw["schema_version"], label="schema_version") != 1:
        raise ValueError("fold manifest schema_version must be 1")
    if _text(raw["protocol_version"], label="protocol_version") != "0.2.3":
        raise ValueError("fold manifest protocol_version must be 0.2.3")
    if (
        _text(raw["experiment_id"], label="experiment_id")
        != "stage2s-causal-seismicity-development-v1"
    ):
        raise ValueError("fold manifest experiment_id mismatch")
    if (
        _text(raw["status"], label="status")
        != "target_blind_calendar_only_no_execution_or_scoring_authority"
    ):
        raise ValueError("fold manifest status mismatch")
    source_design = _mapping(raw["source_design"], label="source_design")
    _exact_keys(
        source_design,
        {
            "path",
            "file_sha256",
            "content_sha256",
            "allowed_pointers",
            "inherited_anomaly_ids_pools_execution_attempt_randomness_or_results",
        },
        label="source_design",
    )
    _text(source_design["path"], label="source_design path")
    _sha256_hex(source_design["file_sha256"], label="source_design file_sha256")
    _sha256_hex(source_design["content_sha256"], label="source_design content_sha256")
    allowed_pointers = tuple(
        _text(item, label="source_design allowed pointer")
        for item in _list(source_design["allowed_pointers"], label="source_design allowed_pointers")
    )
    if allowed_pointers != (
        "/joint_macro_rolling_folds",
        "/target_window_rule",
        "/training_target_end_must_be_strictly_before_assessment_target_start",
    ):
        raise ValueError("source_design allowed pointers differ from the frozen subset")
    _required_bool(
        source_design["inherited_anomaly_ids_pools_execution_attempt_randomness_or_results"],
        False,
        label="source_design inherited execution fields",
    )
    semantics = _mapping(raw["issue_semantics"], label="issue_semantics")
    _exact_keys(
        semantics,
        {
            "timezone",
            "local_time",
            "utc_offset",
            "target_window",
            "fit_horizon_days",
            "assessment_horizons_days",
            "training_target_end_strictly_before_assessment_target_start",
            "random_split",
        },
        label="issue_semantics",
    )
    expected_text = {
        "timezone": "Asia/Shanghai",
        "local_time": "00:00:00",
        "utc_offset": "+08:00",
        "target_window": "(T,T+h]",
    }
    for key, expected in expected_text.items():
        if _text(semantics[key], label=f"issue_semantics {key}") != expected:
            raise ValueError(f"issue_semantics {key} mismatch")
    if _integer(semantics["fit_horizon_days"], label="fit_horizon_days") != FIT_HORIZON_DAYS:
        raise ValueError("fit_horizon_days must be 7")
    horizons = tuple(
        _integer(item, label="assessment horizon")
        for item in _list(
            semantics["assessment_horizons_days"],
            label="assessment_horizons_days",
        )
    )
    if horizons != ASSESSMENT_HORIZONS_DAYS:
        raise ValueError("assessment horizons must be exactly 7, 30, and 90")
    _required_bool(
        semantics["training_target_end_strictly_before_assessment_target_start"],
        True,
        label="strict fit/assessment separation",
    )
    _required_bool(semantics["random_split"], False, label="random_split")
    _required_bool(
        raw["target_bands_mutually_disjoint"],
        True,
        label="target_bands_mutually_disjoint",
    )
    if (
        _text(raw["rolling_rule"], label="rolling_rule")
        != "three_disjoint_90d_target_bands_with_expanding_nonoverlapping_h007_fit"
    ):
        raise ValueError("rolling_rule mismatch")
    folds = tuple(_parse_fold(value) for value in _list(raw["folds"], label="folds"))
    if len(folds) != 3:
        raise ValueError("fold manifest must contain exactly three folds")
    security = _mapping(raw["security"], label="security")
    _exact_keys(
        security,
        {
            "contains_target_ids_coordinates_scores_hits_or_model_results",
            "contains_anomaly_values_features_or_entity_ids",
            "development_target_read_authorized",
            "independent_validation_or_locked_test_authorized",
        },
        label="security",
    )
    for key, value in security.items():
        _required_bool(value, False, label=f"security {key}")
    return Stage2SFoldCalendar(
        manifest_sha256=actual_digest,
        folds=folds,
    )


def parse_frozen_fold_manifest_bytes(payload: bytes) -> Stage2SFoldCalendar:
    """Parse only the preregistered Stage 2S fold-manifest bytes."""

    return parse_fold_manifest_bytes(
        payload,
        expected_sha256=FROZEN_FOLD_MANIFEST_SHA256,
    )


@dataclass(frozen=True, slots=True)
class CausalSourceMembership:
    """One R or RP source view derived from the shared raw catalog object."""

    catalog: Stage2SEarthquakeCatalog = field(repr=False, compare=False)
    component_id: RecentComponentId
    issue_time_utc: datetime
    event_indices: IntArray
    additional_delay_days: AdditionalDelayDays = 0

    def __post_init__(self) -> None:
        if self.component_id not in {"R", "RP"}:
            raise ValueError("component_id must be R or RP")
        if type(self.additional_delay_days) is not int or self.additional_delay_days not in {
            0,
            1,
            7,
        }:
            raise ValueError("additional_delay_days must be exactly 0, 1, or 7")
        issue = self.issue_time_utc
        if not isinstance(issue, datetime) or issue.tzinfo is None:
            raise ValueError("issue_time_utc must be timezone-aware")
        indices = _read_only_indices(self.event_indices, catalog_length=self.catalog.row_count)
        object.__setattr__(self, "issue_time_utc", issue.astimezone(UTC))
        object.__setattr__(self, "event_indices", indices)

    @property
    def event_ids(self) -> tuple[str, ...]:
        return tuple(self.catalog.event_ids[index] for index in self.event_indices)


@dataclass(frozen=True, slots=True)
class FitTargetMembership:
    """Unique supported M5--6 events assigned across one fold's h007 exposures."""

    catalog: Stage2SEarthquakeCatalog = field(repr=False, compare=False)
    fold_index: int
    event_indices: IntArray
    assigned_issue_time_us: IntArray
    exposure_days: float

    def __post_init__(self) -> None:
        if self.fold_index not in FOLD_ORDER:
            raise ValueError("fold_index must be 1, 2, or 3")
        indices = _read_only_indices(self.event_indices, catalog_length=self.catalog.row_count)
        issue_times = _read_only_issue_times(
            self.assigned_issue_time_us,
            length=indices.size,
        )
        exposure = float(self.exposure_days)
        if not math.isfinite(exposure) or exposure <= 0.0:
            raise ValueError("exposure_days must be finite and positive")
        object.__setattr__(self, "event_indices", indices)
        object.__setattr__(self, "assigned_issue_time_us", issue_times)
        object.__setattr__(self, "exposure_days", exposure)

    @property
    def event_ids(self) -> tuple[str, ...]:
        return tuple(self.catalog.event_ids[index] for index in self.event_indices)


@dataclass(frozen=True, slots=True)
class AssessmentTargetMembership:
    """One fold-by-horizon target view authorized by the master prediction seal."""

    catalog: Stage2SEarthquakeCatalog = field(repr=False, compare=False)
    fold_index: int
    horizon_days: int
    event_indices: IntArray
    assigned_issue_time_us: IntArray
    master_seal_file_sha256: str

    def __post_init__(self) -> None:
        if self.fold_index not in FOLD_ORDER:
            raise ValueError("fold_index must be 1, 2, or 3")
        if self.horizon_days not in ASSESSMENT_HORIZONS_DAYS:
            raise ValueError("assessment horizon must be 7, 30, or 90 days")
        indices = _read_only_indices(self.event_indices, catalog_length=self.catalog.row_count)
        issue_times = _read_only_issue_times(
            self.assigned_issue_time_us,
            length=indices.size,
        )
        digest = _sha256_hex(
            self.master_seal_file_sha256,
            label="master_seal_file_sha256",
        )
        object.__setattr__(self, "event_indices", indices)
        object.__setattr__(self, "assigned_issue_time_us", issue_times)
        object.__setattr__(self, "master_seal_file_sha256", digest)

    @property
    def event_ids(self) -> tuple[str, ...]:
        return tuple(self.catalog.event_ids[index] for index in self.event_indices)


def causal_source_membership(
    catalog: Stage2SEarthquakeCatalog,
    *,
    issue_time_utc: datetime,
    component_id: RecentComponentId,
    additional_delay_days: AdditionalDelayDays = 0,
) -> CausalSourceMembership:
    """Return one causal M4+ source window under the frozen 0/1/7-day feed delay.

    The origin-time window remains fixed.  Delay only moves the inclusive
    ``available_at`` cutoff earlier, so an event is never made visible before
    the catalog says it was available.
    """

    if type(additional_delay_days) is not int or additional_delay_days not in {0, 1, 7}:
        raise ValueError("additional_delay_days must be exactly 0, 1, or 7")
    issue_us = _datetime_us(issue_time_utc, label="issue_time_utc")
    if component_id == "R":
        lower_exclusive = issue_us - 30 * _MICROSECONDS_PER_DAY
        upper_inclusive = issue_us
        baseline_availability_inclusive = issue_us
    elif component_id == "RP":
        lower_exclusive = issue_us - 60 * _MICROSECONDS_PER_DAY
        upper_inclusive = issue_us - 30 * _MICROSECONDS_PER_DAY
        baseline_availability_inclusive = upper_inclusive
    else:
        raise ValueError("component_id must be R or RP")
    availability_inclusive = (
        baseline_availability_inclusive - additional_delay_days * _MICROSECONDS_PER_DAY
    )
    mask = (
        catalog.inside_study_area
        & (catalog.magnitude >= M4_MINIMUM)
        & (catalog.origin_time_us > lower_exclusive)
        & (catalog.origin_time_us <= upper_inclusive)
        & (catalog.available_at_us <= availability_inclusive)
    )
    indices = np.asarray(np.flatnonzero(mask), dtype=np.int64)
    indices.setflags(write=False)
    return CausalSourceMembership(
        catalog=catalog,
        component_id=component_id,
        issue_time_utc=issue_time_utc,
        event_indices=indices,
        additional_delay_days=additional_delay_days,
    )


def fit_target_membership(
    catalog: Stage2SEarthquakeCatalog,
    fold: FoldCalendar,
) -> FitTargetMembership:
    """Assign each eligible fit M5--6 event to one unique ``(T,T+7d]`` issue."""

    eligible = (
        catalog.inside_study_area
        & (catalog.magnitude >= M5_6_MINIMUM)
        & (catalog.magnitude < M5_6_MAXIMUM_EXCLUSIVE)
        & (
            catalog.available_at_us
            <= _datetime_us(
                fold.fit_target_end_inclusive_utc,
                label="fit_target_end_inclusive_utc",
            )
        )
    )
    assigned_issue = np.full(catalog.row_count, np.iinfo(np.int64).min, dtype=np.int64)
    assigned = np.zeros(catalog.row_count, dtype=np.bool_)
    for exposure in fold.fit_exposures:
        membership = (
            eligible
            & (catalog.origin_time_us > exposure.issue_time_us)
            & (catalog.origin_time_us <= exposure.target_end_inclusive_us)
        )
        if np.any(assigned & membership):
            raise ValueError("fit event matched more than one h007 issue")
        assigned[membership] = True
        assigned_issue[membership] = exposure.issue_time_us
    indices = np.asarray(np.flatnonzero(assigned), dtype=np.int64)
    issue_times = np.asarray(assigned_issue[indices], dtype=np.int64)
    indices.setflags(write=False)
    issue_times.setflags(write=False)
    return FitTargetMembership(
        catalog=catalog,
        fold_index=fold.fold_index,
        event_indices=indices,
        assigned_issue_time_us=issue_times,
        exposure_days=float(len(fold.fit_exposures) * FIT_HORIZON_DAYS),
    )


def _verified_master_seal(record: SealedRecord) -> tuple[str, str]:
    if not isinstance(record, SealedRecord):
        raise TypeError("master_seal must be a SealedRecord identity")
    if record.record_type != "stage2s_master_prediction_seal":
        raise ValueError("assessment membership requires a master prediction seal")
    content_digest = _sha256_hex(record.content_sha256, label="master content_sha256")
    file_digest = _sha256_hex(record.file_sha256, label="master file_sha256")
    payload = dict(record.payload)
    if payload.get("record_type") != record.record_type:
        raise ValueError("master seal payload record_type mismatch")
    if payload.get("content_sha256") != content_digest:
        raise ValueError("master seal payload content identity mismatch")
    unsigned = dict(payload)
    del unsigned["content_sha256"]
    if hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() != content_digest:
        raise ValueError("master seal content_sha256 is not internally valid")
    serialized = canonical_json_bytes(payload) + b"\n"
    if hashlib.sha256(serialized).hexdigest() != file_digest:
        raise ValueError("master seal file_sha256 is not internally valid")
    bindings = payload.get("bindings")
    if not isinstance(bindings, Mapping) or (
        bindings.get("assessment_target_role_or_score_exposed_before_master_seal") is not False
    ):
        raise ValueError("master seal lacks the no-prior-assessment binding")
    return content_digest, file_digest


def assessment_target_memberships(
    catalog: Stage2SEarthquakeCatalog,
    calendar: Stage2SFoldCalendar,
    *,
    master_seal: SealedRecord,
) -> tuple[AssessmentTargetMembership, ...]:
    """Expose all fold-by-horizon M5--6 memberships only after master sealing."""

    _, master_file_sha256 = _verified_master_seal(master_seal)
    eligible = (
        catalog.inside_study_area
        & (catalog.magnitude >= M5_6_MINIMUM)
        & (catalog.magnitude < M5_6_MAXIMUM_EXCLUSIVE)
    )
    outputs: list[AssessmentTargetMembership] = []
    unique_memberships: set[tuple[str, int, int]] = set()
    for fold in calendar.folds:
        for horizon in ASSESSMENT_HORIZONS_DAYS:
            assigned_issue = np.full(
                catalog.row_count,
                np.iinfo(np.int64).min,
                dtype=np.int64,
            )
            assigned = np.zeros(catalog.row_count, dtype=np.bool_)
            exposures = tuple(
                item for item in fold.assessment_exposures if item.horizon_days == horizon
            )
            for exposure in exposures:
                membership = (
                    eligible
                    & (catalog.origin_time_us > exposure.issue_time_us)
                    & (catalog.origin_time_us <= exposure.target_end_inclusive_us)
                )
                if np.any(assigned & membership):
                    raise ValueError(
                        "assessment event matched more than one issue in a fold/horizon"
                    )
                assigned[membership] = True
                assigned_issue[membership] = exposure.issue_time_us
            indices = np.asarray(np.flatnonzero(assigned), dtype=np.int64)
            for index in indices:
                key = (catalog.event_ids[int(index)], fold.fold_index, horizon)
                if key in unique_memberships:
                    raise ValueError("duplicate cross-issue/horizon assessment membership")
                unique_memberships.add(key)
            issue_times = np.asarray(assigned_issue[indices], dtype=np.int64)
            indices.setflags(write=False)
            issue_times.setflags(write=False)
            outputs.append(
                AssessmentTargetMembership(
                    catalog=catalog,
                    fold_index=fold.fold_index,
                    horizon_days=horizon,
                    event_indices=indices,
                    assigned_issue_time_us=issue_times,
                    master_seal_file_sha256=master_file_sha256,
                )
            )
    return tuple(outputs)


__all__ = [
    "ASSESSMENT_HORIZONS_DAYS",
    "FIT_HORIZON_DAYS",
    "FOLD_ORDER",
    "FROZEN_FOLD_MANIFEST_SHA256",
    "AdditionalDelayDays",
    "AssessmentIssue",
    "AssessmentTargetMembership",
    "CausalSourceMembership",
    "FitTargetMembership",
    "FoldCalendar",
    "Stage2SFoldCalendar",
    "TargetExposure",
    "assessment_target_memberships",
    "causal_source_membership",
    "fit_target_membership",
    "parse_fold_manifest_bytes",
    "parse_frozen_fold_manifest_bytes",
]
