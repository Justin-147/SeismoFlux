"""Strict reader for the frozen stage-2 issue and exposure calendar."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

from seismoflux.background.catalog import utc_timestamp_to_day
from seismoflux.background.config import BackgroundConfig


@dataclass(frozen=True, slots=True)
class IssueExposure:
    """One preregistered issue time and its half-open forecast horizon."""

    issue_date_local: str
    issue_time_utc: str
    issue_day: float
    horizon_days: int
    end_day: float

    def __post_init__(self) -> None:
        if self.horizon_days not in {7, 30, 90, 180, 365}:
            raise ValueError("issue exposure uses an unknown frozen horizon")
        if self.end_day != self.issue_day + float(self.horizon_days):
            raise ValueError("issue exposure end must equal issue plus its horizon")


@dataclass(frozen=True, slots=True)
class IssuePartition:
    """One frozen calendar partition and its non-overlapping exposures."""

    partition_id: str
    start_local: str
    end_local: str
    actual_issue_dates_local: tuple[str, ...]
    actual_issue_days: tuple[float, ...]
    exposures_by_horizon: tuple[tuple[int, tuple[IssueExposure, ...]], ...]

    def exposures(self, horizon_days: int) -> tuple[IssueExposure, ...]:
        for horizon, values in self.exposures_by_horizon:
            if horizon == horizon_days:
                return values
        raise KeyError(f"partition has no {horizon_days}-day exposure calendar")


@dataclass(frozen=True, slots=True)
class FrozenIssueCalendar:
    """The complete development and validation schedule frozen before scoring."""

    schema_version: str
    frozen_on: str
    freeze_tag: str
    development: IssuePartition
    validation: IssuePartition

    def partition(self, partition_id: str) -> IssuePartition:
        if partition_id == "development":
            return self.development
        if partition_id == "validation":
            return self.validation
        raise KeyError(f"unknown issue partition: {partition_id}")


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return cast(dict[str, Any], value)


def _exact_keys(value: dict[str, Any], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} keys differ from the frozen manifest schema")


def _iso_local_date(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO local date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO local date") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"{label} must use canonical YYYY-MM-DD form")
    return value


def _local_issue(value: str) -> tuple[str, float]:
    local = datetime.combine(date.fromisoformat(value), datetime.min.time()).replace(
        tzinfo=ZoneInfo("Asia/Shanghai")
    )
    utc_value = local.astimezone(UTC)
    utc_text = utc_value.isoformat().replace("+00:00", "Z")
    return utc_text, utc_timestamp_to_day(utc_text)


def _expected_greedy_dates(actual: tuple[str, ...], horizon_days: int) -> tuple[str, ...]:
    selected: list[str] = []
    next_eligible: date | None = None
    for raw_value in actual:
        value = date.fromisoformat(raw_value)
        if next_eligible is None or value >= next_eligible:
            selected.append(raw_value)
            next_eligible = value + timedelta(days=horizon_days)
    return tuple(selected)


def _parse_partition(
    partition_id: str,
    raw_value: object,
    *,
    horizons: tuple[int, ...],
) -> IssuePartition:
    raw = _mapping(raw_value, label=f"{partition_id} partition")
    _exact_keys(
        raw,
        {
            "start_local",
            "end_local",
            "actual_issue_date_count",
            "actual_issue_dates_local",
            "non_overlapping_exposures",
        },
        label=f"{partition_id} partition",
    )
    start_local = _iso_local_date(raw["start_local"], label="partition start")
    end_local = _iso_local_date(raw["end_local"], label="partition end")
    if start_local >= end_local:
        raise ValueError("issue partition must have positive duration")
    dates_value = raw["actual_issue_dates_local"]
    if not isinstance(dates_value, list):
        raise ValueError("actual issue dates must be a list")
    actual = tuple(_iso_local_date(item, label="actual issue date") for item in dates_value)
    if not actual or actual != tuple(sorted(set(actual))):
        raise ValueError("actual issue dates must be non-empty, unique, and sorted")
    if any(not (start_local <= value <= end_local) for value in actual):
        raise ValueError("actual issue date lies outside its frozen partition")
    count = raw["actual_issue_date_count"]
    if not isinstance(count, int) or isinstance(count, bool) or count != len(actual):
        raise ValueError("actual issue date count does not match the frozen dates")

    actual_days = tuple(_local_issue(value)[1] for value in actual)
    exposure_root = _mapping(
        raw["non_overlapping_exposures"],
        label="non-overlapping exposures",
    )
    if set(exposure_root) != {str(value) for value in horizons}:
        raise ValueError("exposure horizons differ from the frozen configuration")
    exposures_by_horizon: list[tuple[int, tuple[IssueExposure, ...]]] = []
    for horizon in horizons:
        node = _mapping(exposure_root[str(horizon)], label=f"{horizon}-day exposures")
        _exact_keys(
            node,
            {"count", "issue_dates_local"},
            label=f"{horizon}-day exposures",
        )
        date_values = node["issue_dates_local"]
        if not isinstance(date_values, list):
            raise ValueError("exposure issue dates must be a list")
        selected = tuple(_iso_local_date(item, label="exposure issue date") for item in date_values)
        if selected != _expected_greedy_dates(actual, horizon):
            raise ValueError("non-overlapping exposures violate the frozen greedy rule")
        exposure_count = node["count"]
        if (
            not isinstance(exposure_count, int)
            or isinstance(exposure_count, bool)
            or exposure_count != len(selected)
        ):
            raise ValueError("exposure count does not match its frozen dates")
        exposures: list[IssueExposure] = []
        for issue_date in selected:
            utc_text, issue_day = _local_issue(issue_date)
            exposures.append(
                IssueExposure(
                    issue_date_local=issue_date,
                    issue_time_utc=utc_text,
                    issue_day=issue_day,
                    horizon_days=horizon,
                    end_day=issue_day + float(horizon),
                )
            )
        exposures_by_horizon.append((horizon, tuple(exposures)))
    return IssuePartition(
        partition_id=partition_id,
        start_local=start_local,
        end_local=end_local,
        actual_issue_dates_local=actual,
        actual_issue_days=actual_days,
        exposures_by_horizon=tuple(exposures_by_horizon),
    )


def load_frozen_issue_calendar(
    path: Path,
    *,
    config: BackgroundConfig,
) -> FrozenIssueCalendar:
    """Load and fully cross-check the content-addressed issue metadata only."""

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read frozen issue manifest: {path}") from exc
    root = _mapping(document, label="issue manifest")
    _exact_keys(
        root,
        {
            "schema_version",
            "frozen_on",
            "freeze_tag",
            "scores_seen_before_freeze",
            "source",
            "semantics",
            "partitions",
            "cross_checks",
        },
        label="issue manifest",
    )
    if root["schema_version"] != "1.0.0":
        raise ValueError("issue manifest schema version is not frozen at 1.0.0")
    if root["frozen_on"] != config.frozen_on or root["freeze_tag"] != config.freeze_tag:
        raise ValueError("issue manifest freeze identity differs from the background protocol")
    if root["scores_seen_before_freeze"] is not False:
        raise ValueError("issue manifest was not frozen before background scores")

    semantics = _mapping(root["semantics"], label="issue manifest semantics")
    if semantics.get("issue_timezone") != config.time.timezone:
        raise ValueError("issue manifest timezone differs from the frozen protocol")
    if semantics.get("issue_time_local") != config.time.issue_time_local:
        raise ValueError("issue manifest issue time differs from the frozen protocol")
    if semantics.get("forecast_interval") != config.time.forecast_interval:
        raise ValueError("issue manifest interval differs from the frozen protocol")

    source = _mapping(root["source"], label="issue manifest source")
    if (
        source.get("column_read") != "available_at"
        or source.get("all_other_anomaly_columns_forbidden") is not True
    ):
        raise ValueError("issue manifest violates the available_at-only boundary")
    if source.get("anomaly_report_period_sha256") != config.inputs.issue_manifest_source_sha256:
        raise ValueError("issue manifest source hash differs from the frozen protocol")
    if source.get("data_catalog_sha256") != config.inputs.data_catalog_sha256:
        raise ValueError("issue manifest data-catalog hash differs from the frozen protocol")
    if source.get("study_area_sha256") != config.inputs.study_area_sha256:
        raise ValueError("issue manifest study-area hash differs from the frozen protocol")
    if source.get("stage1_snapshot_id") != config.inputs.expected_stage1_snapshot_id:
        raise ValueError("issue manifest stage-1 snapshot differs from the frozen protocol")

    partitions = _mapping(root["partitions"], label="issue partitions")
    _exact_keys(partitions, {"development", "validation"}, label="issue partitions")
    development = _parse_partition(
        "development",
        partitions["development"],
        horizons=config.time.horizons_days,
    )
    validation = _parse_partition(
        "validation",
        partitions["validation"],
        horizons=config.time.horizons_days,
    )
    if validation.start_local != config.time.validation_start_local:
        raise ValueError("validation calendar start differs from the frozen protocol")
    if validation.end_local != config.time.validation_end_local:
        raise ValueError("validation calendar end differs from the frozen protocol")
    if config.time.representative_issue_date_local not in validation.actual_issue_dates_local:
        raise ValueError("representative issue date is absent from the frozen validation calendar")
    return FrozenIssueCalendar(
        schema_version=cast(str, root["schema_version"]),
        frozen_on=cast(str, root["frozen_on"]),
        freeze_tag=cast(str, root["freeze_tag"]),
        development=development,
        validation=validation,
    )


__all__ = [
    "FrozenIssueCalendar",
    "IssueExposure",
    "IssuePartition",
    "load_frozen_issue_calendar",
]
