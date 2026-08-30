"""Canonical, schema-checked, single-chain helpers for the six P1 record types."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, TypeAlias, cast

from jsonschema import Draft202012Validator, FormatChecker  # type: ignore[import-untyped]

from seismoflux.data.common import canonical_json_bytes
from seismoflux.p1_b0_r30.core import (
    ClusterScore,
    ScoreSummary,
    ordered_cluster_registry_sha256,
    paired_bootstrap_interval,
    selected_cluster_prefix_sha256,
)

RecordType: TypeAlias = Literal[
    "ProtocolDefinition",
    "RealIssueAuthorizationRecord",
    "ForecastIssueRecord",
    "MissedIssueRecord",
    "TruthSnapshotRecord",
    "SequentialReviewRecord",
]
JsonRecord: TypeAlias = dict[str, Any]

RECORD_TYPES: tuple[RecordType, ...] = (
    "ProtocolDefinition",
    "RealIssueAuthorizationRecord",
    "ForecastIssueRecord",
    "MissedIssueRecord",
    "TruthSnapshotRecord",
    "SequentialReviewRecord",
)
AUTHORIZATION_REQUIRED_TYPES = {
    "ForecastIssueRecord",
    "TruthSnapshotRecord",
    "SequentialReviewRecord",
}


def _parse_utc(value: object, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{label} must be an ISO-8601 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{label} must be a valid ISO-8601 UTC timestamp") from exc
    return parsed.astimezone(UTC)


def _record_type(record: Mapping[str, object]) -> RecordType:
    value = record.get("record_type")
    if value not in RECORD_TYPES:
        raise ValueError("record_type is not one of the six frozen P1 record types")
    return cast(RecordType, value)


def _expected_issue_id(scheduled: datetime) -> str:
    return f"p1-{scheduled.strftime('%Y%m%dT%H%M%SZ')}"


def _number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be numeric")
    return float(value)


def _add_calendar_years(value: datetime, years: int) -> datetime:
    """Add whole calendar years without approximating the terminal as day counts."""

    try:
        return value.replace(year=value.year + years)
    except ValueError:
        return value.replace(year=value.year + years, day=28)


def _load_score_registry(
    declared_sha256: object,
    *,
    horizon_days: Literal[30, 90],
    registries_by_sha256: Mapping[str, Sequence[Mapping[str, object]]] | None,
) -> tuple[ClusterScore, ...]:
    if not isinstance(declared_sha256, str):
        raise ValueError("score registry SHA-256 must be a string")
    if registries_by_sha256 is None or declared_sha256 not in registries_by_sha256:
        raise ValueError("score registry preimage is required for scientific validation")
    expected_fields = {
        "issue_id",
        "cluster_id",
        "representative_origin_time_utc",
        "representative_event_id",
        "B0_hit",
        "B0_R30_hit",
    }
    scores: list[ClusterScore] = []
    for raw in registries_by_sha256[declared_sha256]:
        if set(raw) != expected_fields:
            raise ValueError("score registry row fields differ from the frozen contract")
        if type(raw["B0_hit"]) is not bool or type(raw["B0_R30_hit"]) is not bool:
            raise ValueError("score registry hit flags must be booleans")
        issue_id = raw["issue_id"]
        cluster_id = raw["cluster_id"]
        event_id = raw["representative_event_id"]
        if not all(isinstance(value, str) for value in (issue_id, cluster_id, event_id)):
            raise ValueError("score registry identity fields must be strings")
        scores.append(
            ClusterScore(
                issue_id=cast(str, issue_id),
                cluster_id=cast(str, cluster_id),
                representative_origin_time_utc=_parse_utc(
                    raw["representative_origin_time_utc"],
                    label="representative_origin_time_utc",
                ),
                representative_event_id=cast(str, event_id),
                B0_hit=raw["B0_hit"],
                B0_R30_hit=raw["B0_R30_hit"],
            )
        )
    ordered = ScoreSummary(horizon_days=horizon_days, scores=tuple(scores)).scores
    if ordered_cluster_registry_sha256(ordered) != declared_sha256:
        raise ValueError("score registry preimage does not match its declared SHA-256")
    return ordered


def canonical_record_sha256(record: Mapping[str, object]) -> str:
    """Hash canonical JSON after excluding exactly the ``content_sha256`` field."""

    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")
    preimage = {key: value for key, value in record.items() if key != "content_sha256"}
    return hashlib.sha256(canonical_json_bytes(preimage)).hexdigest()


def seal_record(record: Mapping[str, object]) -> JsonRecord:
    """Return a copy carrying its canonical content hash."""

    sealed: JsonRecord = dict(record)
    sealed["content_sha256"] = canonical_record_sha256(sealed)
    return sealed


def build_record(
    record_type: RecordType,
    *,
    recorded_at_utc: str,
    previous_record: Mapping[str, object] | None,
    fields: Mapping[str, object],
) -> JsonRecord:
    """Build one linked record without reading or writing any external state."""

    if record_type not in RECORD_TYPES:
        raise ValueError("unsupported record_type")
    _parse_utc(recorded_at_utc, label="recorded_at_utc")
    if any(
        key in fields
        for key in (
            "schema_version",
            "record_type",
            "recorded_at_utc",
            "chain_sequence",
            "previous_record_type",
            "previous_record_sha256",
            "content_sha256",
        )
    ):
        raise ValueError("record fields may not override the canonical chain header")
    if previous_record is None:
        sequence = 0
        previous_type: RecordType | None = None
        previous_sha: str | None = None
    else:
        sequence_value = previous_record.get("chain_sequence")
        previous_sha_value = previous_record.get("content_sha256")
        if type(sequence_value) is not int or not isinstance(previous_sha_value, str):
            raise ValueError("previous record must be sealed and have an integer sequence")
        if canonical_record_sha256(previous_record) != previous_sha_value:
            raise ValueError("previous record content hash is invalid")
        sequence = sequence_value + 1
        previous_type = _record_type(previous_record)
        previous_sha = previous_sha_value
    return seal_record(
        {
            "schema_version": 1,
            "record_type": record_type,
            "recorded_at_utc": recorded_at_utc,
            "chain_sequence": sequence,
            "previous_record_type": previous_type,
            "previous_record_sha256": previous_sha,
            **dict(fields),
        }
    )


def validate_record_against_schema(
    record: Mapping[str, object], schema: Mapping[str, object]
) -> None:
    """Validate one record structurally and verify its canonical content hash."""

    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(record),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        locations = "; ".join(
            f"{'.'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message}"
            for error in errors[:3]
        )
        raise ValueError(f"record schema validation failed: {locations}")
    declared = record.get("content_sha256")
    if not isinstance(declared, str) or declared != canonical_record_sha256(record):
        raise ValueError("record content_sha256 does not match its canonical content")


def validate_record_chain(
    records: Sequence[Mapping[str, object]],
    schema: Mapping[str, object],
    *,
    score_registries_by_sha256: Mapping[str, Sequence[Mapping[str, object]]] | None = None,
) -> None:
    """Enforce one contiguous append-only chain and pre-authorization boundaries."""

    if not records:
        raise ValueError("record chain must not be empty")
    if _record_type(records[0]) != "ProtocolDefinition":
        raise ValueError("record chain must start with ProtocolDefinition")
    protocol_sha: str | None = None
    protocol_recorded_at: datetime | None = None
    protocol_valid_from: datetime | None = None
    protocol_model_manifest_sha: str | None = None
    protocol_source_manifest_sha: str | None = None
    authorization_sha: str | None = None
    authorization_from: datetime | None = None
    authorization_code_commit: str | None = None
    forecast_issues: dict[str, datetime] = {}
    selected_primary_issue_ids: set[str] = set()
    last_selected_primary_issue: datetime | None = None
    selected_secondary_issue_ids: set[str] = set()
    last_selected_secondary_issue: datetime | None = None
    seen_issue_ids: set[str] = set()
    last_scheduled_issue: datetime | None = None
    truth_keys: set[tuple[str, int]] = set()
    truth_status_by_key: dict[tuple[str, int], object] = {}
    truth_fetched_at_by_key: dict[tuple[str, int], datetime] = {}
    score_registry_by_truth_key: dict[tuple[str, int], tuple[ClusterScore, ...]] = {}
    representative_event_ids_by_horizon: dict[int, set[str]] = {30: set(), 90: set()}
    mature_primary_truth_seen = False
    mature_primary_cluster_count = 0
    terminal_primary_cluster_count = 0
    completed_review_triggers: list[str] = []
    pending_review_batch_recorded_at: datetime | None = None
    final_review_seen = False
    previous_recorded_at: datetime | None = None
    for index, record in enumerate(records):
        validate_record_against_schema(record, schema)
        record_type = _record_type(record)
        chain_closed_before_record = final_review_seen
        if chain_closed_before_record and record_type != "TruthSnapshotRecord":
            raise ValueError(
                "after a final sequential decision only pending truth follow-up may be appended"
            )
        sequence = record.get("chain_sequence")
        if type(sequence) is not int or sequence != index:
            raise ValueError("chain_sequence must be contiguous from zero")
        recorded_at = _parse_utc(record.get("recorded_at_utc"), label="recorded_at_utc")
        if previous_recorded_at is not None and recorded_at < previous_recorded_at:
            raise ValueError("recorded_at_utc must be non-decreasing along the chain")
        previous_recorded_at = recorded_at
        if index == 0:
            if (
                record.get("previous_record_type") is not None
                or record.get("previous_record_sha256") is not None
            ):
                raise ValueError("ProtocolDefinition genesis predecessor must be null")
            protocol_sha = cast(str, record["content_sha256"])
            protocol_recorded_at = recorded_at
            protocol_valid_from = _parse_utc(record.get("valid_from_utc"), label="valid_from_utc")
            if not recorded_at < protocol_valid_from:
                raise ValueError("ProtocolDefinition must be recorded before valid_from")
            protocol_model_manifest_sha = cast(str, record["model_manifest_sha256"])
            protocol_source_manifest_sha = cast(str, record["source_boundary_manifest_sha256"])
        else:
            previous = records[index - 1]
            if record.get("previous_record_type") != previous.get("record_type"):
                raise ValueError("previous_record_type does not match the chain predecessor")
            if record.get("previous_record_sha256") != previous.get("content_sha256"):
                raise ValueError("previous_record_sha256 does not match the chain predecessor")
            if record_type == "ProtocolDefinition":
                raise ValueError("ProtocolDefinition may appear only as genesis")

        due_truth_keys = sorted(
            (
                (issue_id, horizon)
                for horizon, selected_issue_ids in (
                    (30, selected_primary_issue_ids),
                    (90, selected_secondary_issue_ids),
                )
                for issue_id in selected_issue_ids
                if forecast_issues[issue_id] + timedelta(days=horizon + 30) <= recorded_at
                and (issue_id, horizon) not in truth_keys
            ),
            key=lambda key: (
                forecast_issues[key[0]] + timedelta(days=key[1] + 30),
                key[1],
                key[0].encode("utf-8"),
            ),
        )
        current_truth_key = (
            (record.get("issue_id"), record.get("horizon_days"))
            if record_type == "TruthSnapshotRecord"
            else None
        )
        if due_truth_keys and current_truth_key != due_truth_keys[0]:
            raise ValueError("record chain cannot skip a due guard-selected truth snapshot")

        pending_cluster_triggers = [
            trigger
            for threshold, trigger in (
                (10, "cluster_10"),
                (20, "cluster_20"),
                (30, "cluster_30"),
            )
            if mature_primary_cluster_count >= threshold
            and trigger not in completed_review_triggers
        ]
        if (
            not chain_closed_before_record
            and pending_cluster_triggers
            and record_type != "SequentialReviewRecord"
        ):
            same_truth_batch = (
                record_type == "TruthSnapshotRecord"
                and pending_review_batch_recorded_at == recorded_at
            )
            if not same_truth_batch:
                raise ValueError(
                    "crossed cluster review thresholds must be recorded before any later record"
                )

        if protocol_valid_from is not None and not chain_closed_before_record:
            terminal_at_before_record = _add_calendar_years(protocol_valid_from, 3)
            if recorded_at >= terminal_at_before_record and record_type != (
                "SequentialReviewRecord"
            ):
                truth_exactly_at_terminal = (
                    record_type == "TruthSnapshotRecord"
                    and recorded_at == terminal_at_before_record
                )
                if not truth_exactly_at_terminal:
                    raise ValueError(
                        "the 36-month terminal review must precede any post-terminal record"
                    )

        if record_type == "RealIssueAuthorizationRecord":
            if authorization_sha is not None:
                raise ValueError("RealIssueAuthorizationRecord may appear only once")
            if record.get("protocol_definition_sha256") != protocol_sha:
                raise ValueError("authorization is not bound to the genesis protocol")
            remote_verified = _parse_utc(
                record.get("remote_verified_at_utc"), label="remote_verified_at_utc"
            )
            authorization_from = _parse_utc(
                record.get("authorized_from_scheduled_issue_utc"),
                label="authorized_from_scheduled_issue_utc",
            )
            if (
                protocol_valid_from is None
                or authorization_from < protocol_valid_from
                or (authorization_from - protocol_valid_from) % timedelta(days=7) != timedelta(0)
            ):
                raise ValueError(
                    "authorization effective issue is not on the frozen weekly calendar"
                )
            expected_authorization_issue = (
                protocol_valid_from
                if last_scheduled_issue is None
                else last_scheduled_issue + timedelta(days=7)
            )
            if authorization_from != expected_authorization_issue:
                raise ValueError("authorization must begin at the next unrecorded weekly issue")
            if (
                protocol_recorded_at is None
                or remote_verified < protocol_recorded_at
                or remote_verified > recorded_at
                or not recorded_at < authorization_from
            ):
                raise ValueError(
                    "authorization verification must follow the recorded protocol and precede "
                    "its effective issue"
                )
            authorization_sha = cast(str, record["content_sha256"])
            authorization_code_commit = cast(str, record["code_commit"])

        if record_type in AUTHORIZATION_REQUIRED_TYPES:
            if authorization_sha is None:
                raise ValueError(f"{record_type} cannot precede RealIssueAuthorizationRecord")
            if record.get("authorization_record_sha256") != authorization_sha:
                raise ValueError(f"{record_type} does not bind the active authorization record")
            if record.get("protocol_definition_sha256") != protocol_sha:
                raise ValueError(f"{record_type} does not bind the genesis protocol")

        if record_type == "ForecastIssueRecord":
            scheduled = _parse_utc(
                record.get("scheduled_issue_time_utc"), label="scheduled_issue_time_utc"
            )
            cutoff = _parse_utc(record.get("query_cutoff_utc"), label="query_cutoff_utc")
            created = _parse_utc(
                record.get("forecast_created_at_utc"), label="forecast_created_at_utc"
            )
            published = _parse_utc(
                record.get("publication_completed_at_utc"), label="publication_completed_at_utc"
            )
            if authorization_from is None or scheduled < authorization_from:
                raise ValueError("forecast issue precedes the authorization effective issue")
            if cutoff != scheduled - timedelta(minutes=15):
                raise ValueError("forecast query cutoff must equal T minus 15 minutes")
            if not cutoff <= created <= published <= recorded_at < scheduled:
                raise ValueError(
                    "forecast must be created after Q and published/recorded in order before T"
                )
            issue_id = record.get("issue_id")
            if not isinstance(issue_id, str) or issue_id != _expected_issue_id(scheduled):
                raise ValueError("forecast issue_id must exactly encode its scheduled UTC time")
            if protocol_valid_from is None or scheduled < protocol_valid_from:
                raise ValueError("forecast issue precedes valid_from")
            if (scheduled - protocol_valid_from) % timedelta(days=7) != timedelta(0):
                raise ValueError("forecast is not on the frozen weekly calendar")
            expected_scheduled = (
                protocol_valid_from
                if last_scheduled_issue is None
                else last_scheduled_issue + timedelta(days=7)
            )
            if scheduled != expected_scheduled or issue_id in seen_issue_ids:
                raise ValueError("forecast issue must be the next unique weekly issue")
            if record.get("model_manifest_sha256") != protocol_model_manifest_sha:
                raise ValueError("forecast model manifest differs from the genesis protocol")
            if record.get("source_boundary_manifest_sha256") != protocol_source_manifest_sha:
                raise ValueError("forecast source manifest differs from the genesis protocol")
            if record.get("code_commit") != authorization_code_commit:
                raise ValueError("forecast code commit differs from the authorization")
            raw_forecasts = record.get("forecasts")
            if not isinstance(raw_forecasts, list):
                raise ValueError("forecasts must be a two-model list")
            forecasts_by_model = {
                item.get("model_id"): item for item in raw_forecasts if isinstance(item, Mapping)
            }
            if set(forecasts_by_model) != {"B0", "B0_R30"} or len(raw_forecasts) != 2:
                raise ValueError("forecast must contain exactly B0 and B0_R30")
            B0_area = _number(
                forecasts_by_model["B0"].get("actual_alarm_area_km2"),
                label="B0 actual area",
            )
            challenger_area = _number(
                forecasts_by_model["B0_R30"].get("actual_alarm_area_km2"),
                label="B0_R30 actual area",
            )
            reference_area = _number(record.get("B0_reference_area_km2"), label="B0 reference area")
            declared_difference = _number(
                record.get("actual_area_difference_km2"), label="actual area difference"
            )
            next_area = _number(
                record.get("B0_R30_next_complete_cell_area_km2"),
                label="B0_R30 next complete-cell area",
            )
            actual_difference = B0_area - challenger_area
            if not abs(reference_area - B0_area) <= 1e-9:
                raise ValueError("B0 reference area must equal B0 actual alarm area")
            if challenger_area > B0_area + 1e-9:
                raise ValueError("B0_R30 may not use more alarm area than B0")
            if not abs(declared_difference - actual_difference) <= 1e-9:
                raise ValueError("declared alarm-area difference is incorrect")
            if not 0.0 <= actual_difference < next_area or not actual_difference < 625.0:
                raise ValueError("paired alarm-area fairness inequalities failed")
            forecast_issues[issue_id] = scheduled
            if (
                last_selected_primary_issue is None
                or scheduled >= last_selected_primary_issue + timedelta(days=60)
            ):
                selected_primary_issue_ids.add(issue_id)
                last_selected_primary_issue = scheduled
            if (
                last_selected_secondary_issue is None
                or scheduled >= last_selected_secondary_issue + timedelta(days=120)
            ):
                selected_secondary_issue_ids.add(issue_id)
                last_selected_secondary_issue = scheduled
            seen_issue_ids.add(issue_id)
            last_scheduled_issue = scheduled

        if record_type == "TruthSnapshotRecord":
            issue_id = record.get("issue_id")
            if not isinstance(issue_id, str) or issue_id not in forecast_issues:
                raise ValueError("truth snapshot must bind an earlier on-time forecast issue")
            horizon = record.get("horizon_days")
            if type(horizon) is not int or horizon not in {30, 90}:
                raise ValueError("truth horizon must be 30 or 90 days")
            selected_issue_ids = (
                selected_primary_issue_ids if horizon == 30 else selected_secondary_issue_ids
            )
            if issue_id not in selected_issue_ids:
                raise ValueError("truth snapshot issue is not guard-selected for its horizon")
            truth_key = (issue_id, horizon)
            if truth_key in truth_keys:
                raise ValueError("truth snapshot must be unique for each issue and horizon")
            mature_after = _parse_utc(record.get("mature_after_utc"), label="mature_after_utc")
            fetched_at = _parse_utc(
                record.get("truth_fetched_at_utc"), label="truth_fetched_at_utc"
            )
            expected_maturity = forecast_issues[issue_id] + timedelta(days=horizon + 30)
            if mature_after != expected_maturity:
                raise ValueError("truth maturity must equal T plus horizon plus 30 days")
            if fetched_at < mature_after or recorded_at < fetched_at:
                raise ValueError("truth may be recorded only after its frozen maturity time")
            truth_keys.add(truth_key)
            truth_status_by_key[truth_key] = record.get("status")
            truth_fetched_at_by_key[truth_key] = fetched_at
            if record.get("status") == "mature_truth":
                target_count = record.get("target_event_count")
                cluster_count = record.get("independent_cluster_count")
                if type(target_count) is not int or type(cluster_count) is not int:
                    raise ValueError("mature truth counts must be integers")
                if cluster_count > target_count:
                    raise ValueError("independent cluster count cannot exceed target event count")
                registry = _load_score_registry(
                    record.get("exposure_cluster_registry_sha256"),
                    horizon_days=cast(Literal[30, 90], horizon),
                    registries_by_sha256=score_registries_by_sha256,
                )
                if len(registry) != cluster_count:
                    raise ValueError("score registry count differs from mature truth clusters")
                scheduled = forecast_issues[issue_id]
                if any(
                    score.issue_id != issue_id
                    or not scheduled
                    < score.representative_origin_time_utc
                    <= scheduled + timedelta(days=horizon)
                    for score in registry
                ):
                    raise ValueError("score registry does not belong to the truth exposure window")
                representative_event_ids = {score.representative_event_id for score in registry}
                already_counted = representative_event_ids_by_horizon[horizon]
                if len(representative_event_ids) != len(registry) or (
                    representative_event_ids & already_counted
                ):
                    raise ValueError(
                        "representative truth events may not be counted in multiple selected "
                        "exposures of the same horizon"
                    )
                already_counted.update(representative_event_ids)
                score_registry_by_truth_key[truth_key] = registry
                if (
                    horizon == 30
                    and issue_id in selected_primary_issue_ids
                    and not chain_closed_before_record
                ):
                    mature_primary_truth_seen = True
                    mature_primary_cluster_count += cluster_count
                    pending_after_truth = any(
                        mature_primary_cluster_count >= threshold
                        and trigger not in completed_review_triggers
                        for threshold, trigger in (
                            (10, "cluster_10"),
                            (20, "cluster_20"),
                            (30, "cluster_30"),
                        )
                    )
                    if pending_after_truth and pending_review_batch_recorded_at is None:
                        pending_review_batch_recorded_at = recorded_at
                    if protocol_valid_from is None:
                        raise AssertionError(
                            "protocol valid_from must exist after schema validation"
                        )
                    terminal_at = _add_calendar_years(protocol_valid_from, 3)
                    if mature_after <= terminal_at and fetched_at <= terminal_at:
                        terminal_primary_cluster_count += cluster_count

        if record_type == "MissedIssueRecord":
            scheduled = _parse_utc(
                record.get("scheduled_issue_time_utc"), label="scheduled_issue_time_utc"
            )
            issue_id = record.get("issue_id")
            if not isinstance(issue_id, str) or issue_id != _expected_issue_id(scheduled):
                raise ValueError("missed issue_id must exactly encode its scheduled UTC time")
            if protocol_valid_from is None:
                raise AssertionError("protocol valid_from must exist after schema validation")
            expected_scheduled = (
                protocol_valid_from
                if last_scheduled_issue is None
                else last_scheduled_issue + timedelta(days=7)
            )
            if scheduled != expected_scheduled or issue_id in seen_issue_ids:
                raise ValueError("missed issue must be the next unique weekly issue")
            if recorded_at < scheduled:
                raise ValueError("MissedIssueRecord cannot be written before its scheduled T")
            is_authorized = (
                authorization_sha is not None
                and authorization_from is not None
                and scheduled >= authorization_from
            )
            expected_state = "authorized" if is_authorized else "not_authorized"
            expected_sha = authorization_sha if is_authorized else None
            if record.get("authorization_state") != expected_state:
                raise ValueError("missed issue authorization_state is inconsistent with the chain")
            if record.get("authorization_record_sha256") != expected_sha:
                raise ValueError("missed issue authorization hash is inconsistent with the chain")
            reason = record.get("reason")
            allowed_reasons = (
                {
                    "source_snapshot_unavailable_before_T",
                    "forecast_not_frozen_before_T",
                }
                if is_authorized
                else {
                    "protocol_not_remotely_closed_before_T",
                    "code_not_remotely_closed_before_T",
                    "real_issue_not_authorized_before_T",
                }
            )
            if reason not in allowed_reasons:
                raise ValueError("missed issue reason is inconsistent with its authorization state")
            seen_issue_ids.add(issue_id)
            last_scheduled_issue = scheduled

        if record_type == "SequentialReviewRecord":
            if not mature_primary_truth_seen and record.get("review_trigger") != "time_36_months":
                raise ValueError("sequential review requires prior mature 30-day truth")
            trigger = record.get("review_trigger")
            if not isinstance(trigger, str):
                raise ValueError("review_trigger must be a string")
            if protocol_valid_from is None:
                raise AssertionError("protocol valid_from must exist after schema validation")
            terminal_at = _add_calendar_years(protocol_valid_from, 3)
            review_truth_cutoff = min(recorded_at, terminal_at)
            due_primary_truth_keys = {
                (issue_id, 30)
                for issue_id in selected_primary_issue_ids
                if forecast_issues[issue_id] + timedelta(days=60) <= review_truth_cutoff
            }
            primary_truth_available_by_cutoff = {
                key for key in truth_keys if truth_fetched_at_by_key[key] <= review_truth_cutoff
            }
            missing_primary_truth = due_primary_truth_keys - primary_truth_available_by_cutoff
            if missing_primary_truth:
                raise ValueError(
                    "sequential review cannot omit a guard-selected mature 30-day truth"
                )
            expected_by_position = ("cluster_10", "cluster_20", "cluster_30")
            position = len(completed_review_triggers)
            if trigger == "time_36_months":
                if position > 2:
                    raise ValueError("time terminal cannot follow a completed final look")
                if recorded_at < terminal_at:
                    raise ValueError("time_36_months review cannot precede the frozen terminal")
            elif position >= 3 or trigger != expected_by_position[position]:
                raise ValueError("cluster reviews must be unique and ordered 10, 20, 30")
            if record.get("look_sequence") != position + 1:
                raise ValueError("review look_sequence is inconsistent with prior looks")
            if record.get("prior_completed_look_count") != position:
                raise ValueError("review prior_completed_look_count is inconsistent")
            cumulative = record.get("cumulative_cluster_count")
            B0_hits = record.get("B0_hit_clusters")
            challenger_hits = record.get("B0_R30_hit_clusters")
            if (
                type(cumulative) is not int
                or type(B0_hits) is not int
                or type(challenger_hits) is not int
            ):
                raise ValueError("review counts must be integers")
            if cumulative > mature_primary_cluster_count:
                raise ValueError("review cluster count exceeds prior mature 30-day truth support")
            if trigger == "time_36_months" and cumulative != terminal_primary_cluster_count:
                raise ValueError(
                    "time_36_months review must include every primary cluster mature by terminal"
                )
            future_review_triggers: list[object] = []
            for later in records[index + 1 :]:
                if later.get("record_type") != "SequentialReviewRecord":
                    break
                future_review_triggers.append(later.get("review_trigger"))
            is_integrity_pause = record.get("decision") == "pause_scientific_integrity_failure"
            if trigger in {"cluster_10", "cluster_20"} and not is_integrity_pause:
                crossed_triggers = [
                    threshold_trigger
                    for threshold, threshold_trigger in (
                        (10, "cluster_10"),
                        (20, "cluster_20"),
                        (30, "cluster_30"),
                    )
                    if mature_primary_cluster_count >= threshold
                    and threshold_trigger not in completed_review_triggers
                ]
                required_following = crossed_triggers[1:]
                if future_review_triggers[: len(required_following)] != required_following:
                    raise ValueError(
                        "one mature batch must append every crossed cluster look contiguously"
                    )
            if (
                recorded_at >= terminal_at
                and trigger in {"cluster_10", "cluster_20"}
                and not is_integrity_pause
            ):
                expected_final = (
                    "cluster_30" if terminal_primary_cluster_count >= 30 else "time_36_months"
                )
                if expected_final not in future_review_triggers:
                    raise ValueError(
                        "terminal catch-up cluster reviews require a contiguous final review"
                    )
            if (
                recorded_at > terminal_at
                and trigger == "cluster_30"
                and terminal_primary_cluster_count < 30
            ):
                raise ValueError(
                    "cluster_30 after terminal requires 30 clusters frozen by terminal"
                )
            if B0_hits > cumulative or challenger_hits > cumulative:
                raise ValueError("review hit counts cannot exceed the cumulative cluster count")
            gain = record.get("recall_gain_percentage_points")
            lower = record.get("sequentially_adjusted_interval_lower")
            upper = record.get("sequentially_adjusted_interval_upper")
            decision = record.get("decision")
            has_scientific_integrity_gap = not due_primary_truth_keys or any(
                truth_status_by_key[key] != "mature_truth" for key in due_primary_truth_keys
            )
            if (decision == "pause_scientific_integrity_failure") != has_scientific_integrity_gap:
                raise ValueError(
                    "integrity pause must correspond exactly to absent exposure or unavailable "
                    "selected truth"
                )
            review_scores = tuple(
                sorted(
                    (
                        score
                        for key in due_primary_truth_keys
                        if truth_status_by_key[key] == "mature_truth"
                        for score in score_registry_by_truth_key[key]
                    ),
                    key=lambda score: (
                        score.representative_origin_time_utc,
                        score.representative_event_id.encode("utf-8"),
                        score.issue_id.encode("utf-8"),
                        score.cluster_id.encode("utf-8"),
                    ),
                )
            )
            expected_support_count = (
                terminal_primary_cluster_count
                if review_truth_cutoff == terminal_at
                else mature_primary_cluster_count
            )
            if len(review_scores) != expected_support_count:
                raise ValueError("review score rows differ from mature primary truth support")
            if record.get("ordered_cluster_registry_sha256") != (
                ordered_cluster_registry_sha256(review_scores)
            ):
                raise ValueError("review ordered registry differs from mature truth score rows")
            prefix_scores = review_scores[:cumulative]
            if record.get("selected_cluster_prefix_sha256") != (
                selected_cluster_prefix_sha256(prefix_scores)
            ):
                raise ValueError("review selected prefix differs from mature truth score rows")
            expected_B0_hits = sum(score.B0_hit for score in prefix_scores)
            expected_challenger_hits = sum(score.B0_R30_hit for score in prefix_scores)
            if B0_hits != expected_B0_hits or challenger_hits != expected_challenger_hits:
                raise ValueError("review hit counts differ from mature truth score rows")
            if cumulative == 0:
                if any(value is not None for value in (gain, lower, upper)):
                    raise ValueError("zero-cluster review must use null effect and interval")
                expected_decision = "report_evidence_insufficient_at_final_review"
            else:
                expected_gain = 100.0 * (challenger_hits - B0_hits) / cumulative
                declared_gain = _number(gain, label="recall gain")
                lower_value = _number(lower, label="interval lower")
                upper_value = _number(upper, label="interval upper")
                if not abs(declared_gain - expected_gain) <= 1e-9:
                    raise ValueError("review recall gain does not match paired hit counts")
                expected_interval = paired_bootstrap_interval(prefix_scores)
                if expected_interval is None:
                    raise AssertionError("non-empty review prefix must produce an interval")
                if not math.isclose(lower_value, expected_interval[0], abs_tol=1e-12) or not (
                    math.isclose(upper_value, expected_interval[1], abs_tol=1e-12)
                ):
                    raise ValueError("review interval differs from mature truth score rows")
                if trigger in {"cluster_10", "cluster_20"}:
                    expected_decision = "continue_accumulation"
                elif declared_gain <= 0.0:
                    expected_decision = "stop_B0_R30_retain_B0"
                elif declared_gain >= 5.0 and lower_value > 0.0:
                    expected_decision = "confirm_strong_prospective_improvement"
                else:
                    expected_decision = "report_uncertain_at_final_review"
            if decision != "pause_scientific_integrity_failure" and decision != expected_decision:
                raise ValueError("review decision is inconsistent with the frozen rule")
            completed_review_triggers.append(trigger)
            remaining_cluster_trigger = any(
                mature_primary_cluster_count >= threshold
                and threshold_trigger not in completed_review_triggers
                for threshold, threshold_trigger in (
                    (10, "cluster_10"),
                    (20, "cluster_20"),
                    (30, "cluster_30"),
                )
            )
            if not remaining_cluster_trigger:
                pending_review_batch_recorded_at = None
            if trigger in {"cluster_30", "time_36_months"} or (
                decision == "pause_scientific_integrity_failure"
            ):
                final_review_seen = True

        if protocol_valid_from is not None and not chain_closed_before_record:
            next_scheduled_issue = (
                protocol_valid_from
                if last_scheduled_issue is None
                else last_scheduled_issue + timedelta(days=7)
            )
            continuity_cutoff = min(recorded_at, _add_calendar_years(protocol_valid_from, 3))
            if next_scheduled_issue < continuity_cutoff:
                raise ValueError("record chain omitted an already elapsed weekly issue")


__all__ = [
    "RECORD_TYPES",
    "JsonRecord",
    "RecordType",
    "build_record",
    "canonical_record_sha256",
    "seal_record",
    "validate_record_against_schema",
    "validate_record_chain",
]
