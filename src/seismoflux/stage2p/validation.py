"""Fail-closed semantic validation for Stage 2P prospective records.

JSON Schema proves the shape of a record.  This module proves relationships
between fields and between append-only records that JSON Schema cannot express.
It performs no file, network, catalogue, or locked-test access.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta, timezone
from typing import NoReturn
from urllib.parse import quote

from jsonschema import Draft202012Validator, FormatChecker

from seismoflux.data.common import canonical_json_bytes

_ARITHMETIC_ABS_TOLERANCE = 1e-9
_SHANGHAI_OFFSET = timedelta(hours=8)
_THURSDAY = 3
_DEFAULT_CALENDAR: Mapping[str, object] = {
    "issue_weekday": "Thursday",
    "issue_local_time": "00:00:00",
    "issue_utc_offset": "+08:00",
    "first_issue_not_before_local": "2026-09-10T00:00:00+08:00",
    "first_issue_not_before_utc": "2026-09-09T16:00:00Z",
    "query_end_lag_minutes": 15,
}
_REASON_FIELDS = (
    "unevaluable_reason",
    "not_evaluable_reason_codes",
    "not_evaluable_reasons",
    "non_evaluable_reason_codes",
    "evaluability_failure_reasons",
    "evaluable_false_reasons",
    "not_evaluable_reason",
)
_TRUTH_UNAVAILABLE_COUNT_FIELDS = (
    "selected_truth_snapshot_unavailable_count",
    "truth_snapshot_unavailable_count",
    "unavailable_selected_exposure_count",
)
_TRUTH_AVAILABLE_BOOLEAN_FIELDS = (
    "all_selected_truth_snapshots_available",
    "selected_truth_snapshots_all_available",
    "truth_availability_complete",
)
_DENSITY_VALID_BOOLEAN_FIELDS = (
    "all_required_forecast_densities_finite_positive",
    "all_model_target_densities_finite_positive",
    "target_densities_finite_positive",
)
_TRUTH_RETRY_OFFSETS_HOURS = (0, 6, 24, 72, 168)
_REVISION_REASON_ORDER = (
    "current_target_added_prior_not_observed",
    "previous_target_removed_current_not_observed",
    "magnitude_revision",
    "location_revision",
    "origin_time_revision_or_window_boundary_reassignment",
    "identity_revision",
    "dedup_merge",
    "dedup_split",
)
_ISSUE_ID_PATTERN = re.compile(r"^stage2p-issue-(\d{8})T000000\+0800$")
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_EMPTY_ARRAY_SHA256 = hashlib.sha256(b"[]").hexdigest()
_FORMAT_CHECKER = FormatChecker()
LIFECYCLE_IMPLEMENTATION_STATUS = "stage2p1b_required"
_STAGE2P1B_VALIDATOR_NOT_IMPLEMENTED = "stage2p1b_validator_not_implemented"
_FORMAL_TABLE_SCHEMA_SHA256: Mapping[str, str] = {
    "normalized_rows": "edbc68907cfc2b4c321b78c11a254b32b7720c19e34211796df2c29f8ba96dd2",
    "deduplicated_rows": "201be88ebfd8cf21501bf2a195dde9e79d4832a30e8ac33367cbc2700b169342",
    "preferred_field_rows": "6ce072a6511237c69f86c6134e7499e1726a556b5f74cbfa29ff541a2164bae9",
    "window_membership_rows": "fe868712e58949e802a0aaff59a88ab6f3f78b6b600a8b551adf4f14ea513b1f",
    "formal_window_target_bindings": (
        "3ce9c6e921d27c6e99644880965b260116d5effe933504a270a7db4ea647d1b0"
    ),
}
_ARTIFACT_PROFILE_TABLE_CONTRACTS: Mapping[
    str,
    tuple[str, str | tuple[str, ...], str, str],
] = {
    "source_normalized_rows": (
        "#/$defs/SourceNormalizedRowsArtifactIdentity",
        "source_normalized_rows",
        "#/$defs/FormalNormalizedRow",
        "stage2p_source_normalized_rows_sort_v1",
    ),
    "source_deduplicated_rows": (
        "#/$defs/SourceDeduplicatedRowsArtifactIdentity",
        "source_deduplicated_rows",
        "#/$defs/FormalDeduplicatedRow",
        "stage2p_source_deduplicated_rows_sort_v1",
    ),
    "causal_model_view_rows": (
        "#/$defs/CausalModelViewRowsArtifactIdentity",
        "causal_model_view_rows",
        "#/$defs/CausalModelViewRow",
        "stage2p_causal_model_view_rows_sort_v1",
    ),
    "cutover_match_rows": (
        "#/$defs/CutoverMatchRowsArtifactIdentity",
        "cutover_match_rows",
        "#/$defs/CutoverMatchRow",
        "stage2p_cutover_match_rows_sort_v1",
    ),
    "event_set_rows": (
        "#/$defs/EventSetRowsArtifactIdentity",
        ("P0_event_rows", "R30_event_rows", "RP30_event_rows"),
        "#/$defs/EventSetRow",
        "stage2p_event_set_rows_sort_v1",
    ),
    "truth_normalized_rows": (
        "#/$defs/TruthNormalizedRowsArtifactIdentity",
        "truth_normalized_rows",
        "#/$defs/FormalNormalizedRow",
        "stage2p_truth_normalized_rows_sort_v1",
    ),
    "truth_deduplicated_rows": (
        "#/$defs/TruthDeduplicatedRowsArtifactIdentity",
        "truth_deduplicated_rows",
        "#/$defs/FormalDeduplicatedRow",
        "stage2p_truth_deduplicated_rows_sort_v1",
    ),
    "truth_window_membership_rows": (
        "#/$defs/TruthWindowMembershipRowsArtifactIdentity",
        "truth_window_membership_rows",
        "#/$defs/TruthWindowMembershipRow",
        "stage2p_truth_window_membership_rows_sort_v1",
    ),
    "scientific_target_rows": (
        "#/$defs/ScientificTargetRowsArtifactIdentity",
        "scientific_target_rows",
        "#/$defs/FormalScientificTargetRow",
        "stage2p_scientific_target_rows_sort_v1",
    ),
    "grid_cell_geometry_rows": (
        "#/$defs/GridCellGeometryRowsArtifactIdentity",
        "grid_cell_geometry_rows",
        "#/$defs/GridCellGeometryRow",
        "stage2p_grid_cell_geometry_rows_sort_v1",
    ),
    "complete_grid_score_rows": (
        "#/$defs/GridScoreRowsArtifactIdentity",
        "complete_grid_score_rows",
        "#/$defs/GridScoreRow",
        "stage2p_complete_grid_score_rows_sort_v1",
    ),
    "ranked_grid_rows": (
        "#/$defs/RankedGridRowsArtifactIdentity",
        "ranked_grid_rows",
        "#/$defs/RankedGridRow",
        "stage2p_ranked_grid_rows_sort_v1",
    ),
    "alarm_prefix_rows": (
        "#/$defs/AlarmPrefixRowsArtifactIdentity",
        "alarm_prefix_rows",
        "#/$defs/AlarmPrefixRow",
        "stage2p_alarm_prefix_rows_sort_v1",
    ),
    "cluster_membership_rows": (
        "#/$defs/ClusterMembershipRowsArtifactIdentity",
        "cluster_membership_rows",
        "#/$defs/ClusterMembershipRow",
        "stage2p_cluster_membership_rows_sort_v1",
    ),
    "point_contribution_rows": (
        "#/$defs/PointContributionRowsArtifactIdentity",
        "point_contribution_rows",
        "#/$defs/PointContributionRow",
        "stage2p_point_contribution_rows_sort_v1",
    ),
    "region_contribution_rows": (
        "#/$defs/RegionContributionRowsArtifactIdentity",
        "region_contribution_rows",
        "#/$defs/RegionContributionRow",
        "stage2p_region_contribution_rows_sort_v1",
    ),
    "cluster_contribution_rows": (
        "#/$defs/ClusterContributionRowsArtifactIdentity",
        "cluster_contribution_rows",
        "#/$defs/ClusterContributionRow",
        "stage2p_cluster_contribution_rows_sort_v1",
    ),
    "bootstrap_index_rows": (
        "#/$defs/BootstrapIndexRowsArtifactIdentity",
        "bootstrap_index_rows",
        "#/$defs/BootstrapIndexRow",
        "stage2p_bootstrap_index_rows_sort_v1",
    ),
    "bootstrap_distribution_rows": (
        "#/$defs/BootstrapDistributionRowsArtifactIdentity",
        "bootstrap_distribution_rows",
        "#/$defs/BootstrapDistributionRow",
        "stage2p_bootstrap_distribution_rows_sort_v1",
    ),
}
_ARTIFACT_PROFILE_TABLE_NAMES = frozenset(_ARTIFACT_PROFILE_TABLE_CONTRACTS)
_ARTIFACT_PROFILE_MANIFEST_NAMES = frozenset(
    {"forecast_bundle", "effect_rows", "result_bundle"}
)
_ARTIFACT_PROFILE_MANIFEST_CONTRACTS: Mapping[str, tuple[str, str, str]] = {
    "forecast_bundle": (
        "#/$defs/ForecastBundleManifest",
        "stage2p_forecast_bundle_manifest_v1",
        "b032285431dee62bcaa82178ad681781db8ef00e882ce0086e58bebb383e2605",
    ),
    "effect_rows": (
        "#/$defs/EffectRowsManifest",
        "stage2p_effect_rows_manifest_v1",
        "8b5251145e5207f63e4f20f4104768f3c1bfd08f458cdfce913c9757e7816e32",
    ),
    "result_bundle": (
        "#/$defs/ResultBundleManifest",
        "stage2p_result_bundle_manifest_v1",
        "3895e6bf772333deac0069470346a60a5a889560956c1ad686105f3720d9ec5e",
    ),
}
_ARTIFACT_PROFILE_IDENTITY_SCHEMA_NAMES = frozenset(
    {
        "Float64BitsHex",
        "TableArtifactIdentity",
        "RawArrayArtifactIdentity",
        "Float64ArrayArtifactIdentity",
        "MaskArrayArtifactIdentity",
        "GridFamilyIdentity",
        "DensityIdentity",
        "ForecastArtifactSetIdentity",
        "SelectedExposureRow",
        "SelectedExposureManifest",
        "AlarmAreaManifestEntry",
        "AlarmAreaManifest",
        "AlarmAreaComparison",
        "PointContributionArtifactMap",
        "RegionContributionArtifactMap",
        "ClusterContributionArtifactMap",
        "BootstrapDistributionArtifactMap",
    }
)
_FORMAL_FREEZE_UNAVAILABLE_STATUSES = frozenset(
    {
        "not_run_scheduled_issue_cap_terminal",
        "failed_count_preflight",
        "failed_count_limit",
        "failed_query_fetch",
        "failed_query_parse_or_count_mismatch",
        "failed_local_derivation_or_freeze",
    }
)


@_FORMAT_CHECKER.checks("date-time")
def _is_rfc3339_datetime(value: object) -> bool:
    """Provide fail-closed date-time checking even without rfc3339-validator."""

    if not isinstance(value, str):
        return True
    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


class SemanticValidationError(ValueError):
    """A stable semantic error code plus a precise record path."""

    def __init__(self, code: str, message: str, *, path: str = "$") -> None:
        self.code = code
        self.path = path
        self.detail = message
        super().__init__(f"{code} at {path}: {message}")


def _fail(code: str, message: str, *, path: str = "$") -> NoReturn:
    raise SemanticValidationError(code, message, path=path)


def _validate_float64_bits_hex(value: object, *, path: str) -> float:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{16}", value) is None:
        _fail(
            "float64_bits_hex_invalid",
            "float64 bits must be exactly 16 lowercase hexadecimal characters",
            path=path,
        )
    decoded = struct.unpack(">d", bytes.fromhex(value))[0]
    if not math.isfinite(decoded):
        _fail(
            "nonfinite_float64_bits",
            "NaN and positive or negative infinity are forbidden",
            path=path,
        )
    return decoded


def _walk_float64_bits_hex(value: object, *, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            item_path = f"{path}.{key}"
            if key.endswith("_float64_hex") and item is not None:
                _validate_float64_bits_hex(item, path=item_path)
            else:
                _walk_float64_bits_hex(item, path=item_path)
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for index, item in enumerate(value):
            _walk_float64_bits_hex(item, path=f"{path}[{index}]")


def _reject_duplicate_object_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            _fail(
                "duplicate_json_object_key",
                f"duplicate JSON object key is forbidden: {key!r}",
            )
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> NoReturn:
    _fail(
        "nonfinite_json_number",
        f"non-finite JSON numeric constant is forbidden: {value}",
    )


def parse_record_json_bytes(raw: bytes | bytearray | memoryview) -> Mapping[str, object]:
    """Strictly decode one Stage 2P record from its original UTF-8 JSON bytes."""

    if not isinstance(raw, bytes | bytearray | memoryview):
        _fail("record_json_not_bytes", "record JSON input must be bytes")
    try:
        text = bytes(raw).decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise SemanticValidationError(
            "record_json_not_utf8",
            "record JSON bytes must be valid UTF-8",
        ) from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_object_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise SemanticValidationError(
            "record_json_invalid",
            f"record JSON syntax is invalid: {exc.msg}",
        ) from exc
    if not isinstance(value, Mapping):
        _fail(
            "record_json_top_level_not_object",
            "a Stage 2P record JSON document must contain one top-level object",
        )
    return value


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _artifact_bytes(
    artifacts_by_sha256: Mapping[str, object | bytes],
    sha256: object,
    *,
    path: str,
) -> bytes:
    if not isinstance(sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", sha256):
        _fail(
            "artifact_sha256_invalid",
            "artifact identity must be a lowercase SHA-256 digest",
            path=path,
        )
    if sha256 not in artifacts_by_sha256:
        _fail(
            "lifecycle_artifact_missing",
            "every lifecycle artifact must be supplied by its SHA-256 key",
            path=path,
        )
    value = artifacts_by_sha256[sha256]
    if not isinstance(value, bytes | bytearray | memoryview):
        _fail(
            "lifecycle_artifact_not_bytes",
            "lifecycle artifact values must be exact bytes, not decoded objects",
            path=path,
        )
    raw = bytes(value)
    if _sha256_bytes(raw) != sha256:
        _fail(
            "lifecycle_artifact_hash_mismatch",
            "artifact map key must equal SHA-256 of the supplied exact bytes",
            path=path,
        )
    return raw


def _strict_json_value_bytes(raw: bytes, *, path: str) -> object:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise SemanticValidationError(
            "artifact_json_not_utf8",
            "JSON artifact must be strict UTF-8",
            path=path,
        ) from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_object_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise SemanticValidationError(
            "artifact_json_invalid",
            f"JSON artifact syntax is invalid: {exc.msg}",
            path=path,
        ) from exc


def _mapping(value: object, *, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _fail("expected_mapping", "value must be a mapping", path=path)
    return value


def _sequence(value: object, *, path: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        _fail("expected_sequence", "value must be a non-string sequence", path=path)
    return value


def _integer(value: object, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail("expected_integer", "value must be an integer", path=path)
    return value


def _number(value: object, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        _fail("expected_finite_number", "value must be a finite JSON number", path=path)
    result = float(value)
    if not math.isfinite(result):
        _fail("expected_finite_number", "value must be finite", path=path)
    return result


def _timestamp(value: object, *, path: str) -> datetime:
    if not isinstance(value, str):
        _fail("invalid_timestamp", "timestamp must be a string", path=path)
    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise SemanticValidationError(
            "invalid_timestamp",
            f"timestamp is not valid ISO 8601: {value!r}",
            path=path,
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        _fail("invalid_timestamp", "timestamp must include a UTC offset", path=path)
    return parsed


def _utc_timestamp(value: object, *, path: str) -> datetime:
    parsed = _timestamp(value, path=path)
    if parsed.utcoffset() != timedelta(0):
        _fail("timestamp_not_utc", "timestamp must use UTC", path=path)
    return parsed.astimezone(UTC)


def _calendar(protocol: Mapping[str, object]) -> Mapping[str, object]:
    nested = protocol.get("calendar")
    if nested is None:
        return protocol
    return _mapping(nested, path="protocol.calendar")


def _calendar_value(calendar: Mapping[str, object], key: str) -> object:
    return calendar[key] if key in calendar else _DEFAULT_CALENDAR[key]


def _validate_protocol_calendar(calendar: Mapping[str, object]) -> None:
    if _calendar_value(calendar, "issue_weekday") != "Thursday":
        _fail(
            "protocol_issue_weekday_not_thursday",
            "Stage 2P issue_weekday must remain Thursday",
            path="protocol.calendar.issue_weekday",
        )
    if _calendar_value(calendar, "issue_local_time") != "00:00:00":
        _fail(
            "protocol_issue_time_not_midnight",
            "Stage 2P issue_local_time must remain 00:00:00",
            path="protocol.calendar.issue_local_time",
        )
    if _calendar_value(calendar, "issue_utc_offset") != "+08:00":
        _fail(
            "protocol_issue_offset_not_shanghai",
            "Stage 2P issue offset must remain +08:00",
            path="protocol.calendar.issue_utc_offset",
        )
    lag = _integer(
        _calendar_value(calendar, "query_end_lag_minutes"),
        path="protocol.calendar.query_end_lag_minutes",
    )
    if lag != 15:
        _fail(
            "protocol_query_lag_not_15_minutes",
            "Stage 2P query_end_lag_minutes must remain 15",
            path="protocol.calendar.query_end_lag_minutes",
        )


def _first_issue_not_before(calendar: Mapping[str, object]) -> datetime:
    local_raw = _calendar_value(calendar, "first_issue_not_before_local")
    utc_raw = _calendar_value(calendar, "first_issue_not_before_utc")
    local = _timestamp(local_raw, path="protocol.calendar.first_issue_not_before_local")
    utc = _utc_timestamp(utc_raw, path="protocol.calendar.first_issue_not_before_utc")
    if local.utcoffset() != _SHANGHAI_OFFSET:
        _fail(
            "protocol_first_issue_offset_invalid",
            "first_issue_not_before_local must use +08:00",
            path="protocol.calendar.first_issue_not_before_local",
        )
    if local.astimezone(UTC) != utc:
        _fail(
            "protocol_first_issue_times_disagree",
            "local and UTC first-issue thresholds must name the same instant",
            path="protocol.calendar",
        )
    return utc


def _validate_issue_clock(
    record: Mapping[str, object],
    protocol: Mapping[str, object],
) -> tuple[datetime, datetime, datetime]:
    calendar = _calendar(protocol)
    _validate_protocol_calendar(calendar)

    local = _timestamp(record.get("issue_time_local"), path="$.issue_time_local")
    issue_utc = _utc_timestamp(record.get("issue_time_utc"), path="$.issue_time_utc")
    query_end = _utc_timestamp(record.get("query_end_utc"), path="$.query_end_utc")

    if local.utcoffset() != _SHANGHAI_OFFSET:
        _fail(
            "issue_local_offset_not_shanghai",
            "issue_time_local must use the +08:00 Asia/Shanghai offset",
            path="$.issue_time_local",
        )
    if (local.hour, local.minute, local.second, local.microsecond) != (0, 0, 0, 0):
        _fail(
            "issue_not_local_midnight",
            "issue_time_local must be exactly 00:00:00",
            path="$.issue_time_local",
        )
    if local.weekday() != _THURSDAY:
        _fail(
            "issue_not_thursday",
            "issue_time_local must be a Thursday",
            path="$.issue_time_local",
        )
    if local.astimezone(UTC) != issue_utc:
        _fail(
            "issue_local_utc_mismatch",
            "issue_time_local and issue_time_utc must name the same instant",
            path="$.issue_time_utc",
        )
    if query_end != issue_utc - timedelta(minutes=15):
        _fail(
            "query_end_not_t_minus_15_minutes",
            "query_end_utc must equal issue T minus 15 minutes",
            path="$.query_end_utc",
        )
    if issue_utc < _first_issue_not_before(calendar):
        _fail(
            "issue_before_first_issue_not_before",
            "issue T precedes the frozen first-issue threshold",
            path="$.issue_time_utc",
        )
    return local, issue_utc, query_end


def _equal_timestamp(
    value: object,
    expected: datetime,
    *,
    code: str,
    path: str,
    message: str,
) -> None:
    if _utc_timestamp(value, path=path) != expected:
        _fail(code, message, path=path)


def _strictly_before(
    value: object,
    upper: datetime,
    *,
    code: str,
    path: str,
    message: str,
) -> datetime:
    parsed = _utc_timestamp(value, path=path)
    if parsed >= upper:
        _fail(code, message, path=path)
    return parsed


def _at_or_after(
    value: object,
    lower: datetime,
    *,
    code: str,
    path: str,
    message: str,
) -> datetime:
    parsed = _utc_timestamp(value, path=path)
    if parsed < lower:
        _fail(code, message, path=path)
    return parsed


def _validate_fetch_interval(
    fetch: Mapping[str, object],
    *,
    query_end: datetime,
    issue_utc: datetime,
    path: str,
) -> None:
    started = _at_or_after(
        fetch.get("fetch_started_at_utc"),
        query_end,
        code="fetch_started_before_query_end",
        path=f"{path}.fetch_started_at_utc",
        message="on-time issue fetch must start at or after Q",
    )
    completed = _strictly_before(
        fetch.get("fetch_completed_at_utc"),
        issue_utc,
        code="fetch_completed_not_before_issue",
        path=f"{path}.fetch_completed_at_utc",
        message="on-time issue fetch must complete before T",
    )
    if completed < started:
        _fail(
            "fetch_completed_before_fetch_started",
            "fetch completion cannot precede fetch start",
            path=f"{path}.fetch_completed_at_utc",
        )


def _validate_count_preflight_interval(
    fetch: Mapping[str, object],
    *,
    query_end: datetime,
    issue_utc: datetime,
    path: str,
) -> None:
    preflight_value = fetch.get("count_preflight")
    if preflight_value is None:
        return
    preflight = _mapping(preflight_value, path=f"{path}.count_preflight")
    started_value = preflight.get("fetch_started_at_utc")
    completed_value = preflight.get("fetch_completed_at_utc")
    if started_value is None and completed_value is None:
        return
    started = _at_or_after(
        started_value,
        query_end,
        code="count_preflight_started_before_query_end",
        path=f"{path}.count_preflight.fetch_started_at_utc",
        message="count preflight must start at or after Q",
    )
    completed = _strictly_before(
        completed_value,
        issue_utc,
        code="count_preflight_completed_not_before_issue",
        path=f"{path}.count_preflight.fetch_completed_at_utc",
        message="count preflight must complete before T",
    )
    if completed < started:
        _fail(
            "count_preflight_completed_before_start",
            "count preflight completion cannot precede its start",
            path=f"{path}.count_preflight.fetch_completed_at_utc",
        )
    query_started = _utc_timestamp(
        fetch.get("fetch_started_at_utc"),
        path=f"{path}.fetch_started_at_utc",
    )
    if query_started < completed:
        _fail(
            "query_started_before_count_preflight_completed",
            "formal query must not start before count preflight completes",
            path=f"{path}.fetch_started_at_utc",
        )


def _validate_on_time_issue_causality(
    record: Mapping[str, object],
    *,
    issue_utc: datetime,
    query_end: datetime,
) -> None:
    attempts = _sequence(record.get("attempt_evidence"), path="$.attempt_evidence")
    if len(attempts) != 1:
        _fail(
            "on_time_issue_attempt_count_invalid",
            "an on-time issue must contain exactly one issue fetch attempt",
            path="$.attempt_evidence",
        )
    attempt = _mapping(attempts[0], path="$.attempt_evidence[0]")
    _equal_timestamp(
        attempt.get("query_end_utc"),
        query_end,
        code="attempt_query_end_mismatch",
        path="$.attempt_evidence[0].query_end_utc",
        message="attempt query_end_utc must equal top-level Q",
    )
    _validate_fetch_interval(
        attempt,
        query_end=query_end,
        issue_utc=issue_utc,
        path="$.attempt_evidence[0]",
    )
    _validate_count_preflight_interval(
        attempt,
        query_end=query_end,
        issue_utc=issue_utc,
        path="$.attempt_evidence[0]",
    )

    snapshot = _mapping(record.get("source_snapshot"), path="$.source_snapshot")
    _equal_timestamp(
        snapshot.get("issue_time_utc"),
        issue_utc,
        code="source_snapshot_issue_time_mismatch",
        path="$.source_snapshot.issue_time_utc",
        message="source snapshot issue_time_utc must equal top-level T",
    )
    _equal_timestamp(
        snapshot.get("query_end_utc"),
        query_end,
        code="source_snapshot_query_end_mismatch",
        path="$.source_snapshot.query_end_utc",
        message="source snapshot query_end_utc must equal top-level Q",
    )
    acquisition = _mapping(snapshot.get("acquisition"), path="$.source_snapshot.acquisition")
    _equal_timestamp(
        acquisition.get("query_end_utc"),
        query_end,
        code="acquisition_query_end_mismatch",
        path="$.source_snapshot.acquisition.query_end_utc",
        message="source acquisition query_end_utc must equal top-level Q",
    )
    _validate_fetch_interval(
        acquisition,
        query_end=query_end,
        issue_utc=issue_utc,
        path="$.source_snapshot.acquisition",
    )
    _validate_count_preflight_interval(
        acquisition,
        query_end=query_end,
        issue_utc=issue_utc,
        path="$.source_snapshot.acquisition",
    )
    _validate_attempt_acquisition_exchange_binding(
        attempt,
        acquisition,
        path="$.source_snapshot.acquisition",
        subject="issue",
    )
    source_sealed = _strictly_before(
        snapshot.get("seal_completed_at_utc"),
        issue_utc,
        code="source_snapshot_sealed_not_before_issue",
        path="$.source_snapshot.seal_completed_at_utc",
        message="source snapshot seal must complete before T",
    )
    acquisition_completed = _utc_timestamp(
        acquisition.get("fetch_completed_at_utc"),
        path="$.source_snapshot.acquisition.fetch_completed_at_utc",
    )
    if source_sealed < acquisition_completed:
        _fail(
            "source_snapshot_sealed_before_query_completed",
            "source snapshot cannot seal before its selected query completes",
            path="$.source_snapshot.seal_completed_at_utc",
        )

    seal = _mapping(record.get("prediction_seal"), path="$.prediction_seal")
    _equal_timestamp(
        seal.get("issue_time_utc"),
        issue_utc,
        code="prediction_seal_issue_time_mismatch",
        path="$.prediction_seal.issue_time_utc",
        message="prediction seal issue_time_utc must equal top-level T",
    )
    _equal_timestamp(
        seal.get("query_end_utc"),
        query_end,
        code="prediction_seal_query_end_mismatch",
        path="$.prediction_seal.query_end_utc",
        message="prediction seal query_end_utc must equal top-level Q",
    )
    _equal_timestamp(
        seal.get("valid_from_utc"),
        issue_utc,
        code="prediction_valid_from_mismatch",
        path="$.prediction_seal.valid_from_utc",
        message="prediction valid_from_utc must equal T",
    )
    prediction_sealed = _strictly_before(
        seal.get("sealed_at_utc"),
        issue_utc,
        code="prediction_sealed_not_before_issue",
        path="$.prediction_seal.sealed_at_utc",
        message="prediction seal must complete before T",
    )
    if prediction_sealed < source_sealed:
        _fail(
            "prediction_sealed_before_source_snapshot",
            "prediction seal cannot precede the source snapshot seal",
            path="$.prediction_seal.sealed_at_utc",
        )
    core_frozen = _utc_timestamp(
        record.get("record_core_frozen_at_utc"),
        path="$.record_core_frozen_at_utc",
    )
    if core_frozen < prediction_sealed:
        _fail(
            "issue_core_frozen_before_prediction_seal",
            "issue record core cannot freeze before its prediction seal",
            path="$.record_core_frozen_at_utc",
        )


def _validate_forecast_identity(identity: Mapping[str, object], *, path: str) -> None:
    complete_count = _integer(
        identity.get("complete_grid_cell_count"),
        path=f"{path}.complete_grid_cell_count",
    )
    prefix_count = _integer(
        identity.get("alarm_prefix_cell_count"),
        path=f"{path}.alarm_prefix_cell_count",
    )
    if prefix_count > complete_count:
        _fail(
            "alarm_prefix_exceeds_complete_grid",
            "alarm_prefix_cell_count cannot exceed complete_grid_cell_count",
            path=f"{path}.alarm_prefix_cell_count",
        )
    termination_reason = identity.get("alarm_prefix_termination_reason")
    allowed_termination_reasons = {
        "next_complete_cell_would_exceed_budget",
        "domain_exhausted",
    }
    if termination_reason not in allowed_termination_reasons:
        _fail(
            "alarm_prefix_termination_reason_invalid",
            "alarm_prefix_termination_reason must use a frozen deterministic value",
            path=f"{path}.alarm_prefix_termination_reason",
        )
    expected_termination_reason = (
        "domain_exhausted"
        if prefix_count == complete_count
        else "next_complete_cell_would_exceed_budget"
    )
    if termination_reason != expected_termination_reason:
        _fail(
            "alarm_prefix_termination_reason_inconsistent",
            "termination reason must be derived from the complete ranking and selected prefix",
            path=f"{path}.alarm_prefix_termination_reason",
        )

    budget = _number(identity.get("alarm_area_budget_km2"), path=f"{path}.alarm_area_budget_km2")
    actual = _number(identity.get("actual_alarm_area_km2"), path=f"{path}.actual_alarm_area_km2")
    remaining = _number(
        identity.get("remaining_budget_km2"),
        path=f"{path}.remaining_budget_km2",
    )
    if actual > budget:
        _fail(
            "actual_alarm_area_exceeds_budget",
            "actual_alarm_area_km2 cannot exceed alarm_area_budget_km2",
            path=f"{path}.actual_alarm_area_km2",
        )
    if not math.isclose(
        math.fsum((actual, remaining)),
        budget,
        rel_tol=0.0,
        abs_tol=_ARITHMETIC_ABS_TOLERANCE,
    ):
        _fail(
            "alarm_area_budget_arithmetic_mismatch",
            "actual_alarm_area_km2 + remaining_budget_km2 must equal the budget",
            path=path,
        )
    next_fields = (
        "next_unselected_rank_position_1_based",
        "next_unselected_cell_id",
        "next_unselected_complete_cell_area_km2",
        "next_unselected_ranked_row_sha256",
    )
    if termination_reason == "next_complete_cell_would_exceed_budget":
        if any(identity.get(field) is None for field in next_fields):
            _fail(
                "next_unselected_cell_evidence_missing",
                "budget termination requires complete identity and area evidence for the next cell",
                path=path,
            )
        next_position = _integer(
            identity.get("next_unselected_rank_position_1_based"),
            path=f"{path}.next_unselected_rank_position_1_based",
        )
        if next_position != prefix_count + 1:
            _fail(
                "next_unselected_rank_position_mismatch",
                "next unselected rank must immediately follow the selected prefix",
                path=f"{path}.next_unselected_rank_position_1_based",
            )
        next_cell_id = identity.get("next_unselected_cell_id")
        if not isinstance(next_cell_id, str) or not next_cell_id:
            _fail(
                "next_unselected_cell_identity_invalid",
                "next unselected cell_id must be a non-empty string",
                path=f"{path}.next_unselected_cell_id",
            )
        next_area = _number(
            identity.get("next_unselected_complete_cell_area_km2"),
            path=f"{path}.next_unselected_complete_cell_area_km2",
        )
        if not 0 < next_area <= 625:
            _fail(
                "next_unselected_cell_area_invalid",
                "next complete cell area must be in (0, 625] km2",
                path=f"{path}.next_unselected_complete_cell_area_km2",
            )
        if remaining >= next_area:
            _fail(
                "next_unselected_cell_would_fit_budget",
                "termination is valid only when remaining budget is less than next cell area",
                path=f"{path}.next_unselected_complete_cell_area_km2",
            )
        next_row_hash = identity.get("next_unselected_ranked_row_sha256")
        if not isinstance(next_row_hash, str):
            _fail(
                "next_unselected_ranked_row_identity_invalid",
                "next ranked row requires its SHA-256 identity",
                path=f"{path}.next_unselected_ranked_row_sha256",
            )
    elif any(identity.get(field) is not None for field in next_fields):
        _fail(
            "domain_exhausted_contains_next_cell_evidence",
            "domain exhaustion requires all next-unselected fields to be null",
            path=path,
        )


def _canonical_identity(identity: Mapping[str, object], *, path: str) -> bytes:
    try:
        return canonical_json_bytes(dict(identity))
    except (TypeError, ValueError) as exc:
        raise SemanticValidationError(
            "forecast_identity_not_canonicalizable",
            "ForecastIdentity cannot be encoded as canonical JSON",
            path=path,
        ) from exc


def _validate_object_identity(
    value: Mapping[str, object],
    *,
    id_field: str,
    hash_field: str,
    subject: str,
    path: str,
) -> None:
    identity = value.get(id_field)
    declared_hash = value.get(hash_field)
    if not isinstance(identity, str) or not isinstance(declared_hash, str):
        _fail(
            f"{subject}_identity_missing",
            f"{id_field} and {hash_field} must both contain the derived SHA-256",
            path=path,
        )
    if identity != declared_hash:
        _fail(
            f"{subject}_id_hash_mismatch",
            f"{id_field} must exactly equal {hash_field}",
            path=f"{path}.{id_field}",
        )
    body = {
        key: item
        for key, item in value.items()
        if key not in {id_field, hash_field}
    }
    try:
        calculated = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    except (TypeError, ValueError) as exc:
        raise SemanticValidationError(
            f"{subject}_not_canonicalizable",
            f"{subject} cannot be encoded as canonical JSON",
            path=path,
        ) from exc
    if declared_hash != calculated:
        _fail(
            f"{subject}_hash_mismatch",
            f"{hash_field} must be recomputed after excluding exactly {id_field} and {hash_field}",
            path=f"{path}.{hash_field}",
        )


def _validate_canonical_self_hash(
    value: Mapping[str, object],
    *,
    hash_field: str,
    subject: str,
    path: str,
) -> None:
    body = {key: item for key, item in value.items() if key != hash_field}
    try:
        calculated = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    except (TypeError, ValueError) as exc:
        raise SemanticValidationError(
            f"{subject}_not_canonicalizable",
            f"{subject} cannot be encoded as canonical JSON",
            path=path,
        ) from exc
    if value.get(hash_field) != calculated:
        _fail(
            f"{subject}_hash_mismatch",
            f"{hash_field} must hash the canonical object excluding only itself",
            path=f"{path}.{hash_field}",
        )


def _validate_source_acquisition_provenance(
    acquisition: Mapping[str, object],
    *,
    path: str,
) -> None:
    """Recompute the response-header and raw-body provenance held by an acquisition."""

    captured_value = acquisition.get("captured_response_headers")
    raw_value = acquisition.get("raw_response")
    if captured_value is None or raw_value is None:
        return
    captured = _mapping(
        captured_value,
        path=f"{path}.captured_response_headers",
    )
    calculated_headers_sha256 = hashlib.sha256(
        canonical_json_bytes(dict(captured))
    ).hexdigest()
    if acquisition.get("response_headers_sha256") != calculated_headers_sha256:
        _fail(
            "captured_response_headers_hash_mismatch",
            "response_headers_sha256 must hash the canonical captured headers",
            path=f"{path}.response_headers_sha256",
        )
    captured_content_type = captured.get("content_type")
    if (
        captured_content_type is not None
        and acquisition.get("response_content_type") != captured_content_type
    ):
        _fail(
            "captured_response_content_type_mismatch",
            "non-null captured content_type must equal response_content_type",
            path=f"{path}.captured_response_headers.content_type",
        )

    raw = _mapping(raw_value, path=f"{path}.raw_response")
    if raw.get("artifact_id") != raw.get("file_sha256"):
        _fail(
            "raw_artifact_id_mismatch",
            "RawArtifactIdentity artifact_id must equal exact file_sha256",
            path=f"{path}.raw_response.artifact_id",
        )
    body_byte_count = _integer(
        acquisition.get("response_body_byte_count"),
        path=f"{path}.response_body_byte_count",
    )
    raw_byte_count = _integer(
        raw.get("byte_count"),
        path=f"{path}.raw_response.byte_count",
    )
    if raw_byte_count != body_byte_count:
        _fail(
            "raw_response_body_byte_count_mismatch",
            "raw response byte_count must equal response_body_byte_count",
            path=f"{path}.raw_response.byte_count",
        )
    content_length = captured.get("content_length")
    if content_length is not None:
        if (
            not isinstance(content_length, str)
            or re.fullmatch(r"[0-9]+", content_length) is None
        ):
            _fail(
                "captured_content_length_invalid",
                "captured content_length must be null or an unsigned decimal integer",
                path=f"{path}.captured_response_headers.content_length",
            )
        if int(content_length) != body_byte_count:
            _fail(
                "captured_content_length_body_mismatch",
                "captured content_length must equal the frozen response body size",
                path=f"{path}.captured_response_headers.content_length",
            )
    if acquisition.get("http_status") == 204 and (
        raw_byte_count != 0 or raw.get("file_sha256") != _EMPTY_SHA256
    ):
        _fail(
            "http_204_raw_response_not_empty",
            "HTTP 204 raw response must be the exact empty byte sequence",
            path=f"{path}.raw_response",
        )


def _table_artifact_row_count(
    manifest: Mapping[str, object],
    field: str,
    *,
    path: str,
    protocol: Mapping[str, object] | None = None,
) -> int:
    identity = _mapping(manifest.get(field), path=f"{path}.{field}")
    if identity.get("local_restricted") is not True:
        _fail(
            "formal_freeze_table_not_local_restricted",
            "formal freeze table artifacts must remain local restricted",
            path=f"{path}.{field}.local_restricted",
        )
    if identity.get("artifact_id") != identity.get("file_sha256"):
        _fail(
            "formal_freeze_table_artifact_id_mismatch",
            "formal table artifact_id must equal exact file_sha256",
            path=f"{path}.{field}.artifact_id",
        )
    if protocol is None or "target_cohort" not in protocol:
        _fail(
            "formal_freeze_table_registry_required",
            "formal manifests require the complete externally frozen protocol table registry",
            path="protocol.target_cohort.formal_freeze_source_manifest",
        )
    target_cohort = _mapping(
        protocol.get("target_cohort"),
        path="protocol.target_cohort",
    )
    if "formal_freeze_source_manifest" not in target_cohort:
        _fail(
            "formal_freeze_table_registry_required",
            "formal manifests require the complete externally frozen protocol table registry",
            path="protocol.target_cohort.formal_freeze_source_manifest",
        )
    if protocol is not None:
        formal_contract = _mapping(
            target_cohort.get("formal_freeze_source_manifest"),
            path="protocol.target_cohort.formal_freeze_source_manifest",
        )
        if "derived_table_registry" not in formal_contract:
            _fail(
                "formal_freeze_table_registry_required",
                "formal manifests require the complete derived-table registry",
                path=(
                    "protocol.target_cohort.formal_freeze_source_manifest."
                    "derived_table_registry"
                ),
            )
        registry = _mapping(
            formal_contract.get("derived_table_registry"),
            path=(
                "protocol.target_cohort.formal_freeze_source_manifest."
                "derived_table_registry"
            ),
        )
        serialization = _mapping(
            registry.get("serialization"),
            path=(
                "protocol.target_cohort.formal_freeze_source_manifest."
                "derived_table_registry.serialization"
            ),
        )
        tables = _mapping(
            registry.get("tables"),
            path=(
                "protocol.target_cohort.formal_freeze_source_manifest."
                "derived_table_registry.tables"
            ),
        )
        contract = _mapping(
            tables.get(field),
            path=(
                "protocol.target_cohort.formal_freeze_source_manifest."
                f"derived_table_registry.tables.{field}"
            ),
        )
        calculated_sort_sha256 = _sha256_bytes(
            canonical_json_bytes(
                _mapping(
                    contract.get("sort_spec"),
                    path=(
                        "protocol.target_cohort.formal_freeze_source_manifest."
                        f"derived_table_registry.tables.{field}.sort_spec"
                    ),
                )
            )
        )
        if contract.get("expected_row_schema_sha256") != (
            _FORMAL_TABLE_SCHEMA_SHA256[field]
        ):
            _fail(
                "formal_table_registry_expected_schema_hash_invalid",
                "protocol expected_row_schema_sha256 must equal the frozen schema identity",
                path=(
                    "protocol.target_cohort.formal_freeze_source_manifest."
                    f"derived_table_registry.tables.{field}."
                    "expected_row_schema_sha256"
                ),
            )
        if contract.get("expected_sort_order_sha256") != calculated_sort_sha256:
            _fail(
                "formal_table_registry_expected_sort_hash_invalid",
                "protocol expected_sort_order_sha256 must hash its exact sort_spec",
                path=(
                    "protocol.target_cohort.formal_freeze_source_manifest."
                    f"derived_table_registry.tables.{field}."
                    "expected_sort_order_sha256"
                ),
            )
        expected = {
            "table_role": contract.get("table_role"),
            "serialization_profile": serialization.get("profile"),
            "row_schema_ref": contract.get("row_schema_ref"),
            "sort_profile": contract.get("sort_profile"),
            "schema_sha256": contract.get("expected_row_schema_sha256"),
            "sort_order_sha256": contract.get("expected_sort_order_sha256"),
        }
        for identity_field, expected_value in expected.items():
            if identity.get(identity_field) != expected_value:
                _fail(
                    "formal_freeze_table_registry_binding_mismatch",
                    f"{identity_field} must equal the frozen protocol table registry",
                    path=f"{path}.{field}.{identity_field}",
                )
    return _integer(
        identity.get("row_count"),
        path=f"{path}.{field}.row_count",
    )


def _validate_formal_freeze_source_manifest(
    manifest: Mapping[str, object],
    *,
    path: str,
    protocol: Mapping[str, object] | None = None,
) -> tuple[datetime, datetime]:
    _validate_canonical_self_hash(
        manifest,
        hash_field="manifest_sha256",
        subject="formal_freeze_source_manifest",
        path=path,
    )
    if manifest.get("status") != "succeeded_single_full_cohort_response":
        _fail(
            "formal_freeze_status_invalid",
            "formal freeze manifest must represent one successful full-cohort response",
            path=f"{path}.status",
        )
    observed = _utc_timestamp(
        manifest.get("snapshot_observed_at_utc"),
        path=f"{path}.snapshot_observed_at_utc",
    )
    completed = _utc_timestamp(
        manifest.get("freeze_completed_at_utc"),
        path=f"{path}.freeze_completed_at_utc",
    )
    if completed < observed:
        _fail(
            "formal_freeze_completed_before_observed",
            "formal freeze completion cannot precede its unified observation time",
            path=f"{path}.freeze_completed_at_utc",
        )

    batch_count = _integer(
        manifest.get("query_batch_count"),
        path=f"{path}.query_batch_count",
    )
    requests = _sequence(
        manifest.get("ordered_query_request_sha256"),
        path=f"{path}.ordered_query_request_sha256",
    )
    responses = _sequence(
        manifest.get("ordered_raw_response_sha256"),
        path=f"{path}.ordered_raw_response_sha256",
    )
    headers = _sequence(
        manifest.get("ordered_response_headers_sha256"),
        path=f"{path}.ordered_response_headers_sha256",
    )
    if batch_count != 1:
        _fail(
            "formal_freeze_query_batch_count_not_one",
            "formal freeze must use exactly one full-cohort query response",
            path=f"{path}.query_batch_count",
        )
    if (
        batch_count != len(requests)
        or batch_count != len(headers)
        or batch_count != len(responses)
    ):
        _fail(
            "formal_freeze_query_batch_count_mismatch",
            "query_batch_count must equal request, header, and response lengths",
            path=f"{path}.query_batch_count",
        )
    request_hashes: list[str] = []
    for index, value in enumerate(requests):
        if not isinstance(value, str):
            _fail(
                "formal_freeze_request_identity_invalid",
                "every ordered query request identity must be a string",
                path=f"{path}.ordered_query_request_sha256[{index}]",
            )
        request_hashes.append(value)
    if len(set(request_hashes)) != len(request_hashes):
        _fail(
            "formal_freeze_query_request_identity_not_unique",
            "ordered query request identities must be unique",
            path=f"{path}.ordered_query_request_sha256",
        )
    for index, value in enumerate(responses):
        if not isinstance(value, str):
            _fail(
                "formal_freeze_response_identity_invalid",
                "every ordered raw response identity must be a string",
                path=f"{path}.ordered_raw_response_sha256[{index}]",
            )
    acquisition = _mapping(
        manifest.get("source_acquisition"),
        path=f"{path}.source_acquisition",
    )
    global_start = manifest.get("global_target_start_exclusive_utc")
    global_end = manifest.get("global_target_end_inclusive_utc")
    if acquisition.get("query_start_utc") != global_start:
        _fail(
            "formal_freeze_global_query_start_mismatch",
            "formal acquisition query start must equal the global selected-exposure start",
            path=f"{path}.source_acquisition.query_start_utc",
        )
    if acquisition.get("query_end_utc") != global_end:
        _fail(
            "formal_freeze_global_query_end_mismatch",
            "formal acquisition query end must equal the global selected-exposure end",
            path=f"{path}.source_acquisition.query_end_utc",
        )
    global_start_time = _utc_timestamp(
        global_start,
        path=f"{path}.global_target_start_exclusive_utc",
    )
    global_end_time = _utc_timestamp(
        global_end,
        path=f"{path}.global_target_end_inclusive_utc",
    )
    if global_start_time >= global_end_time:
        _fail(
            "formal_freeze_global_interval_not_positive",
            "formal freeze global target interval must have positive duration",
            path=f"{path}.global_target_end_inclusive_utc",
        )
    _validate_source_acquisition_provenance(
        acquisition,
        path=f"{path}.source_acquisition",
    )
    raw_response = _mapping(
        acquisition.get("raw_response"),
        path=f"{path}.source_acquisition.raw_response",
    )
    snapshot_preimage = {
        "profile": "stage2p_formal_freeze_source_snapshot_v1",
        "source_id": acquisition.get("source_id"),
        "canonical_query_request_sha256": acquisition.get(
            "request_identity_sha256"
        ),
        "response_headers_sha256": acquisition.get("response_headers_sha256"),
        "raw_response_sha256": raw_response.get("file_sha256"),
        "response_content_type": acquisition.get("response_content_type"),
        "response_body_byte_count": acquisition.get(
            "response_body_byte_count"
        ),
        "http_status": acquisition.get("http_status"),
        "fetch_started_at_utc": acquisition.get("fetch_started_at_utc"),
        "fetch_completed_at_utc": acquisition.get("fetch_completed_at_utc"),
    }
    snapshot_sha256 = hashlib.sha256(
        canonical_json_bytes(snapshot_preimage)
    ).hexdigest()
    if manifest.get("source_snapshot_sha256") != snapshot_sha256:
        _fail(
            "formal_freeze_source_snapshot_hash_mismatch",
            "source_snapshot_sha256 must bind the frozen single-response preimage",
            path=f"{path}.source_snapshot_sha256",
        )
    direct_bindings = (
        (
            requests[0],
            acquisition.get("request_identity_sha256"),
            "ordered_query_request_sha256",
        ),
        (
            headers[0],
            acquisition.get("response_headers_sha256"),
            "ordered_response_headers_sha256",
        ),
        (
            responses[0],
            raw_response.get("file_sha256"),
            "ordered_raw_response_sha256",
        ),
    )
    for declared, expected, field in direct_bindings:
        if declared != expected:
            _fail(
                "formal_freeze_ordered_exchange_identity_mismatch",
                f"{field}[0] must bind the single source acquisition exchange",
                path=f"{path}.{field}[0]",
            )
    acquisition_completed = _utc_timestamp(
        acquisition.get("fetch_completed_at_utc"),
        path=f"{path}.source_acquisition.fetch_completed_at_utc",
    )
    if observed != acquisition_completed:
        _fail(
            "formal_freeze_observed_not_query_completed",
            "snapshot observation must equal the single query completion time",
            path=f"{path}.snapshot_observed_at_utc",
        )
    feature_count = _integer(
        acquisition.get("feature_count"),
        path=f"{path}.source_acquisition.feature_count",
    )
    normalized_count = _table_artifact_row_count(
        manifest,
        "normalized_rows",
        path=path,
        protocol=protocol,
    )
    deduplicated_count = _table_artifact_row_count(
        manifest,
        "deduplicated_rows",
        path=path,
        protocol=protocol,
    )
    preferred_count = _table_artifact_row_count(
        manifest,
        "preferred_field_rows",
        path=path,
        protocol=protocol,
    )
    _table_artifact_row_count(
        manifest,
        "window_membership_rows",
        path=path,
        protocol=protocol,
    )
    _table_artifact_row_count(
        manifest,
        "formal_window_target_bindings",
        path=path,
        protocol=protocol,
    )
    if normalized_count != feature_count:
        _fail(
            "formal_freeze_normalized_count_mismatch",
            "normalized rows must account for every feature in the single response",
            path=f"{path}.normalized_rows.row_count",
        )
    if not preferred_count <= deduplicated_count <= normalized_count:
        _fail(
            "formal_freeze_row_count_order_invalid",
            "preferred <= deduplicated <= normalized row counts must hold",
            path=path,
        )
    if manifest.get("local_restricted") is not True:
        _fail(
            "formal_freeze_not_local_restricted",
            "formal freeze source artifacts must remain local restricted",
            path=f"{path}.local_restricted",
        )
    return observed, completed


def _validate_formal_freeze_failure_evidence(
    record: Mapping[str, object],
) -> None:
    evidence = _mapping(
        record.get("formal_freeze_failure_evidence"),
        path="$.formal_freeze_failure_evidence",
    )
    _validate_canonical_self_hash(
        evidence,
        hash_field="failure_evidence_sha256",
        subject="formal_freeze_failure_evidence",
        path="$.formal_freeze_failure_evidence",
    )
    evidence_sha256 = evidence.get("failure_evidence_sha256")
    if record.get("formal_freeze_failure_evidence_sha256") != evidence_sha256:
        _fail(
            "formal_freeze_failure_top_level_hash_mismatch",
            "top-level failure evidence hash must equal the nested evidence identity",
            path="$.formal_freeze_failure_evidence_sha256",
        )
    status = evidence.get("status")
    if record.get("formal_freeze_status") != status:
        _fail(
            "formal_freeze_failure_status_mismatch",
            "top-level formal_freeze_status must equal failure evidence status",
            path="$.formal_freeze_status",
        )
    if (
        record.get("formal_freeze_source_manifest") is not None
        or record.get("formal_freeze_source_manifest_sha256") is not None
    ):
        _fail(
            "formal_freeze_failure_contains_success_manifest",
            "formal freeze failure cannot retain a successful source manifest",
            path="$.formal_freeze_source_manifest",
        )
    if evidence.get("contains_target_or_effect_rows") is not False:
        _fail(
            "formal_freeze_failure_contains_target_or_effect_rows",
            "formal freeze failure evidence must certify no target or effect rows",
            path="$.formal_freeze_failure_evidence.contains_target_or_effect_rows",
        )
    fail_closed_pairs = (
        ("effect_rows_opened_at_utc", None),
        ("sample_gate_met", False),
        ("confirmatory_effects_authorized", False),
        ("confirmatory_result", None),
    )
    for field, expected in fail_closed_pairs:
        if field in record and record.get(field) != expected:
            _fail(
                "formal_freeze_failure_not_fail_closed",
                f"{field} must remain {expected!r} after formal freeze failure",
                path=f"$.{field}",
            )
    target_summary_fields = (
        "union_unique_supported_M5_6_event_count",
        "union_unique_full_study_area_M5_6_event_count",
        "unique_target_cluster_count",
    )
    unavailable = status in _FORMAL_FREEZE_UNAVAILABLE_STATUSES
    expected_summary = None if unavailable else 0
    for field in target_summary_fields:
        if field in record and record.get(field) != expected_summary:
            _fail(
                "formal_freeze_failure_contains_target_summary",
                (
                    "unavailable formal freezes require null target summaries; "
                    "only a mechanically empty complete scope may report zero"
                ),
                path=f"$.{field}",
            )
    target_union = record.get("realized_target_union")
    if unavailable and "realized_target_union" in record and target_union is not None:
        _fail(
            "formal_freeze_failure_contains_target_union",
            "unavailable formal freezes require a null realized target union",
            path="$.realized_target_union",
        )
    if (
        not unavailable
        and isinstance(target_union, Mapping)
        and target_union.get("unique_event_count") != 0
    ):
        _fail(
            "formal_freeze_failure_contains_target_union",
            "a mechanically empty complete scope requires an empty target union",
            path="$.realized_target_union.unique_event_count",
        )
    if status == "not_run_no_complete_scope":
        selected_manifest_value = record.get("selected_exposure_manifest")
        if selected_manifest_value is not None:
            selected_manifest = _mapping(
                selected_manifest_value,
                path="$.selected_exposure_manifest",
            )
            if _sequence(
                selected_manifest.get("rows"),
                path="$.selected_exposure_manifest.rows",
            ):
                _fail(
                    "formal_no_complete_scope_contains_selected_exposure",
                    "not_run_no_complete_scope requires an empty selected exposure axis",
                    path="$.selected_exposure_manifest.rows",
                )
        for field in (
            "final_window_membership_sha256",
            "cluster_membership_sha256",
        ):
            if field in record and record.get(field) != _EMPTY_ARRAY_SHA256:
                _fail(
                    "formal_no_complete_scope_empty_identity_mismatch",
                    f"{field} must identify the canonical empty ordered row array",
                    path=f"$.{field}",
                )
        if isinstance(target_union, Mapping):
            _validate_target_set_identity(
                target_union,
                path="$.realized_target_union",
            )
            for field in (
                "ordered_event_ids_sha256",
                "target_rows_content_sha256",
                "region_membership_sha256",
            ):
                if field in target_union and target_union.get(field) != (
                    _EMPTY_ARRAY_SHA256
                ):
                    _fail(
                        "formal_no_complete_scope_empty_target_hash_mismatch",
                        f"{field} must identify the canonical empty array",
                        path=f"$.realized_target_union.{field}",
                    )
            target_rows = target_union.get("target_rows")
            if isinstance(target_rows, Mapping) and (
                target_rows.get("row_count") != 0
                or target_rows.get("byte_count") != 0
                or target_rows.get("file_sha256") != _EMPTY_SHA256
                or target_rows.get("artifact_id") != _EMPTY_SHA256
                or target_rows.get("content_sha256") != _EMPTY_ARRAY_SHA256
            ):
                _fail(
                    "formal_no_complete_scope_target_rows_not_empty",
                    "empty-scope target table must use exact empty file and content identities",
                    path="$.realized_target_union.target_rows",
                )
    if unavailable:
        for field in (
            "final_window_membership_sha256",
            "cluster_membership_sha256",
        ):
            if field in record and record.get(field) is not None:
                _fail(
                    "formal_freeze_failure_contains_derived_identity",
                    f"{field} must be null when the formal freeze is unavailable",
                    path=f"$.{field}",
                )

    scheduled = _utc_timestamp(
        evidence.get("scheduled_at_utc"),
        path="$.formal_freeze_failure_evidence.scheduled_at_utc",
    )
    record_core = record.get("record_core_frozen_at_utc")
    if record_core is not None and scheduled > _utc_timestamp(
        record_core,
        path="$.record_core_frozen_at_utc",
    ):
        _fail(
            "formal_freeze_failure_scheduled_after_record_core",
            "formal freeze failure must be scheduled no later than the record core freeze",
            path="$.formal_freeze_failure_evidence.scheduled_at_utc",
        )

    global_start = evidence.get("global_target_start_exclusive_utc")
    global_end = evidence.get("global_target_end_inclusive_utc")
    count_value = evidence.get("count_preflight")
    failure_artifact = evidence.get("query_failure_artifact_sha256")
    failure_stage = evidence.get("failure_stage")
    scope_not_run = status in {
        "not_run_no_complete_scope",
        "not_run_scheduled_issue_cap_terminal",
    }
    if scope_not_run:
        if (
            global_start is not None
            or global_end is not None
            or count_value is not None
            or failure_artifact is not None
            or failure_stage != "scope_selection"
        ):
            _fail(
                "formal_freeze_scope_not_run_evidence_inconsistent",
                "scope-selection not-run evidence cannot contain query or target artifacts",
                path="$.formal_freeze_failure_evidence",
            )
        return
    if global_start is None or global_end is None:
        _fail(
            "formal_freeze_failure_global_interval_missing",
            "attempted formal freezes require the complete global target interval",
            path="$.formal_freeze_failure_evidence",
        )
    start_time = _utc_timestamp(
        global_start,
        path="$.formal_freeze_failure_evidence.global_target_start_exclusive_utc",
    )
    end_time = _utc_timestamp(
        global_end,
        path="$.formal_freeze_failure_evidence.global_target_end_inclusive_utc",
    )
    if start_time >= end_time:
        _fail(
            "formal_freeze_failure_global_interval_not_positive",
            "attempted formal freeze interval must have positive duration",
            path="$.formal_freeze_failure_evidence.global_target_end_inclusive_utc",
        )
    count = _mapping(
        count_value,
        path="$.formal_freeze_failure_evidence.count_preflight",
    )
    _validate_count_preflight(
        count,
        path="$.formal_freeze_failure_evidence.count_preflight",
        scheduled_at=scheduled,
    )
    count_outcome = count.get("outcome")
    parsed_count = count.get("parsed_count")
    if status == "failed_count_preflight":
        if count_outcome == "succeeded" or failure_stage != "count_preflight":
            _fail(
                "formal_freeze_count_failure_evidence_inconsistent",
                "failed_count_preflight requires a failed count exchange",
                path="$.formal_freeze_failure_evidence",
            )
        if failure_artifact is not None:
            _fail(
                "formal_freeze_count_failure_contains_query_artifact",
                "count failure occurs before any formal query artifact exists",
                path="$.formal_freeze_failure_evidence.query_failure_artifact_sha256",
            )
    elif status == "failed_count_limit":
        if (
            count_outcome != "succeeded"
            or not isinstance(parsed_count, int)
            or isinstance(parsed_count, bool)
            or parsed_count < 20_000
            or failure_stage != "count_preflight"
            or failure_artifact is not None
        ):
            _fail(
                "formal_freeze_count_limit_evidence_inconsistent",
                "failed_count_limit requires a successful count >= 20000 and no query",
                path="$.formal_freeze_failure_evidence",
            )
    else:
        if (
            count_outcome != "succeeded"
            or not isinstance(parsed_count, int)
            or isinstance(parsed_count, bool)
            or parsed_count >= 20_000
        ):
            _fail(
                "formal_freeze_post_count_failure_evidence_inconsistent",
                "post-count failures require a successful count below the hard limit",
                path="$.formal_freeze_failure_evidence.count_preflight",
            )
        expected_stage = {
            "failed_query_fetch": "query_fetch",
            "failed_query_parse_or_count_mismatch": (
                "query_parse_or_count_consistency"
            ),
        }.get(status)
        if expected_stage is not None and failure_stage != expected_stage:
            _fail(
                "formal_freeze_failure_stage_mismatch",
                "formal freeze failure stage must match its status",
                path="$.formal_freeze_failure_evidence.failure_stage",
            )
        if (
            status == "failed_local_derivation_or_freeze"
            and failure_stage
            not in {
                "normalization_deduplication_or_window_binding",
                "durable_freeze",
            }
        ):
            _fail(
                "formal_freeze_failure_stage_mismatch",
                "local derivation failure must occur during derivation or durable freeze",
                path="$.formal_freeze_failure_evidence.failure_stage",
            )
        if (
            status
            in {
                "failed_query_parse_or_count_mismatch",
                "failed_local_derivation_or_freeze",
            }
            and failure_artifact is None
        ):
            _fail(
                "formal_freeze_failure_artifact_missing",
                "captured query/local-derivation failures require an audit artifact",
                path="$.formal_freeze_failure_evidence.query_failure_artifact_sha256",
            )


def _validate_formal_freeze_record_binding(
    record: Mapping[str, object],
    protocol: Mapping[str, object] | None = None,
) -> None:
    manifest_value = record.get("formal_freeze_source_manifest")
    if manifest_value is None:
        if record.get("formal_freeze_failure_evidence") is not None:
            _validate_formal_freeze_failure_evidence(record)
        return
    manifest = _mapping(
        manifest_value,
        path="$.formal_freeze_source_manifest",
    )
    _, completed = _validate_formal_freeze_source_manifest(
        manifest,
        path="$.formal_freeze_source_manifest",
        protocol=protocol,
    )
    if record.get("formal_freeze_source_manifest_sha256") != manifest.get(
        "manifest_sha256"
    ):
        _fail(
            "formal_freeze_manifest_top_level_hash_mismatch",
            "top-level formal freeze manifest hash must equal the nested manifest identity",
            path="$.formal_freeze_source_manifest_sha256",
        )
    window_rows = _mapping(
        manifest.get("window_membership_rows"),
        path="$.formal_freeze_source_manifest.window_membership_rows",
    )
    if (
        "final_window_membership_sha256" in record
        and record.get("final_window_membership_sha256")
        != window_rows.get("content_sha256")
    ):
        _fail(
            "final_window_membership_hash_mismatch",
            "formal success final window hash must equal the frozen table content",
            path="$.final_window_membership_sha256",
        )
    selected_manifest_value = record.get("selected_exposure_manifest")
    if selected_manifest_value is not None:
        selected_manifest = _mapping(
            selected_manifest_value,
            path="$.selected_exposure_manifest",
        )
        selected_rows = _sequence(
            selected_manifest.get("rows"),
            path="$.selected_exposure_manifest.rows",
        )
        binding_rows = _mapping(
            manifest.get("formal_window_target_bindings"),
            path=(
                "$.formal_freeze_source_manifest."
                "formal_window_target_bindings"
            ),
        )
        if binding_rows.get("row_count") != len(selected_rows):
            _fail(
                "formal_window_binding_selected_exposure_count_mismatch",
                "formal success requires one binding row per selected exposure",
                path=(
                    "$.formal_freeze_source_manifest."
                    "formal_window_target_bindings.row_count"
                ),
            )
    if record.get("formal_freeze_status") not in {
        None,
        "succeeded_single_full_cohort_response",
    }:
        _fail(
            "formal_freeze_success_status_mismatch",
            "successful formal manifest requires the successful top-level status",
            path="$.formal_freeze_status",
        )
    if record.get("formal_freeze_failure_evidence") is not None or record.get(
        "formal_freeze_failure_evidence_sha256"
    ) is not None:
        _fail(
            "formal_freeze_success_contains_failure_evidence",
            "successful formal freeze cannot retain failure evidence",
            path="$.formal_freeze_failure_evidence",
        )
    core_frozen = _utc_timestamp(
        record.get("record_core_frozen_at_utc"),
        path="$.record_core_frozen_at_utc",
    )
    if core_frozen < completed:
        _fail(
            "evaluation_core_frozen_before_formal_freeze_completed",
            "evaluation core cannot freeze before the formal source manifest completes",
            path="$.record_core_frozen_at_utc",
        )
    effect_opened = record.get("effect_rows_opened_at_utc")
    if effect_opened is not None and _utc_timestamp(
        effect_opened,
        path="$.effect_rows_opened_at_utc",
    ) < completed:
        _fail(
            "effect_rows_opened_before_formal_freeze_completed",
            "confirmatory effect rows cannot open before formal source freeze completion",
            path="$.effect_rows_opened_at_utc",
        )


def _validate_typed_table_identity(
    value: object,
    *,
    expected_role: str,
    expected_row_schema_ref: str,
    expected_sort_profile: str,
    path: str,
) -> Mapping[str, object]:
    identity = _mapping(value, path=path)
    if identity.get("artifact_id") != identity.get("file_sha256"):
        _fail(
            "typed_table_artifact_id_mismatch",
            "typed table artifact_id must equal its exact file_sha256",
            path=f"{path}.artifact_id",
        )
    expected = {
        "table_role": expected_role,
        "serialization_profile": "seismoflux_canonical_jsonl_v1",
        "row_schema_ref": expected_row_schema_ref,
        "sort_profile": expected_sort_profile,
        "local_restricted": True,
    }
    for field, expected_value in expected.items():
        if identity.get(field) != expected_value:
            _fail(
                "typed_table_contract_mismatch",
                f"{field} must equal the frozen typed-table contract",
                path=f"{path}.{field}",
            )
    return identity


def _validate_event_set_identity(
    value: object,
    *,
    component_role: str,
    path: str,
) -> None:
    event_set = _mapping(value, path=path)
    # Lightweight semantic fixtures predate the typed row artifact. Complete
    # schema-valid Stage 2P records always take this strict branch.
    if "component_role" not in event_set and "event_rows" not in event_set:
        return
    if event_set.get("component_role") != component_role:
        _fail(
            "event_set_component_role_mismatch",
            "event-set component_role must match its enclosing SourceSnapshot field",
            path=f"{path}.component_role",
        )
    rows = _validate_typed_table_identity(
        event_set.get("event_rows"),
        expected_role=f"{component_role}_event_rows",
        expected_row_schema_ref="#/$defs/EventSetRow",
        expected_sort_profile="stage2p_event_set_rows_sort_v1",
        path=f"{path}.event_rows",
    )
    if event_set.get("event_count") != rows.get("row_count"):
        _fail(
            "event_set_row_count_mismatch",
            "event_count must equal the typed event_rows row_count",
            path=f"{path}.event_count",
        )
    if event_set.get("event_rows_content_sha256") != rows.get("content_sha256"):
        _fail(
            "event_set_rows_content_hash_mismatch",
            "event_rows_content_sha256 must equal the typed row content identity",
            path=f"{path}.event_rows_content_sha256",
        )


def _validate_target_set_identity(value: object, *, path: str) -> None:
    target_set = _mapping(value, path=path)
    if "target_rows" not in target_set:
        return
    rows = _validate_typed_table_identity(
        target_set.get("target_rows"),
        expected_role="scientific_target_rows",
        expected_row_schema_ref="#/$defs/FormalScientificTargetRow",
        expected_sort_profile="stage2p_scientific_target_rows_sort_v1",
        path=f"{path}.target_rows",
    )
    if target_set.get("unique_event_count") != rows.get("row_count"):
        _fail(
            "target_set_row_count_mismatch",
            "unique_event_count must equal the typed target_rows row_count",
            path=f"{path}.unique_event_count",
        )
    if target_set.get("target_rows_content_sha256") != rows.get("content_sha256"):
        _fail(
            "target_set_rows_content_hash_mismatch",
            "target_rows_content_sha256 must equal the typed row content identity",
            path=f"{path}.target_rows_content_sha256",
        )


def _validate_source_snapshot_typed_artifacts(
    snapshot: Mapping[str, object],
    *,
    path: str,
) -> None:
    typed_fields = {
        "normalized_rows": (
            "source_normalized_rows",
            "#/$defs/FormalNormalizedRow",
            "stage2p_source_normalized_rows_sort_v1",
        ),
        "deduplicated_rows": (
            "source_deduplicated_rows",
            "#/$defs/FormalDeduplicatedRow",
            "stage2p_source_deduplicated_rows_sort_v1",
        ),
        "causal_model_view": (
            "causal_model_view_rows",
            "#/$defs/CausalModelViewRow",
            "stage2p_causal_model_view_rows_sort_v1",
        ),
    }
    for field, (role, row_schema_ref, sort_profile) in typed_fields.items():
        if field in snapshot:
            _validate_typed_table_identity(
                snapshot.get(field),
                expected_role=role,
                expected_row_schema_ref=row_schema_ref,
                expected_sort_profile=sort_profile,
                path=f"{path}.{field}",
            )
    if "cutover_cross_source_match_rows" in snapshot:
        rows = _validate_typed_table_identity(
            snapshot.get("cutover_cross_source_match_rows"),
            expected_role="cutover_match_rows",
            expected_row_schema_ref="#/$defs/CutoverMatchRow",
            expected_sort_profile="stage2p_cutover_match_rows_sort_v1",
            path=f"{path}.cutover_cross_source_match_rows",
        )
        if snapshot.get("cutover_cross_source_match_count") != rows.get("row_count"):
            _fail(
                "cutover_match_row_count_mismatch",
                "cutover match count must equal its typed artifact row_count",
                path=f"{path}.cutover_cross_source_match_count",
            )
        if snapshot.get("cutover_cross_source_match_sha256") != rows.get(
            "content_sha256"
        ):
            _fail(
                "cutover_match_content_hash_mismatch",
                "cutover match summary must equal its typed artifact content identity",
                path=f"{path}.cutover_cross_source_match_sha256",
            )
    for field, role in (
        ("P0_event_set", "P0"),
        ("R30_event_set", "R30"),
        ("RP30_event_set", "RP30"),
    ):
        if field in snapshot:
            _validate_event_set_identity(
                snapshot.get(field),
                component_role=role,
                path=f"{path}.{field}",
            )


def _validate_truth_snapshot_typed_artifacts(
    snapshot: Mapping[str, object],
    *,
    path: str,
) -> None:
    typed_fields = {
        "normalized_rows": (
            "truth_normalized_rows",
            "#/$defs/FormalNormalizedRow",
            "stage2p_truth_normalized_rows_sort_v1",
        ),
        "deduplicated_rows": (
            "truth_deduplicated_rows",
            "#/$defs/FormalDeduplicatedRow",
            "stage2p_truth_deduplicated_rows_sort_v1",
        ),
    }
    for field, (role, row_schema_ref, sort_profile) in typed_fields.items():
        if field in snapshot:
            _validate_typed_table_identity(
                snapshot.get(field),
                expected_role=role,
                expected_row_schema_ref=row_schema_ref,
                expected_sort_profile=sort_profile,
                path=f"{path}.{field}",
            )
    if "window_membership_rows" in snapshot:
        rows = _validate_typed_table_identity(
            snapshot.get("window_membership_rows"),
            expected_role="truth_window_membership_rows",
            expected_row_schema_ref="#/$defs/TruthWindowMembershipRow",
            expected_sort_profile="stage2p_truth_window_membership_rows_sort_v1",
            path=f"{path}.window_membership_rows",
        )
        if snapshot.get("window_membership_sha256") != rows.get("content_sha256"):
            _fail(
                "truth_window_membership_content_hash_mismatch",
                "window_membership_sha256 must equal its typed artifact content identity",
                path=f"{path}.window_membership_sha256",
            )
    target_set = snapshot.get("realized_target_set")
    if target_set is not None:
        _validate_target_set_identity(
            target_set,
            path=f"{path}.realized_target_set",
        )


def _validate_embedded_object_identities(record: Mapping[str, object]) -> None:
    snapshot_value = record.get("source_snapshot")
    snapshot: Mapping[str, object] | None = None
    if snapshot_value is not None:
        snapshot = _mapping(snapshot_value, path="$.source_snapshot")
        _validate_source_snapshot_typed_artifacts(
            snapshot,
            path="$.source_snapshot",
        )
        _validate_object_identity(
            snapshot,
            id_field="snapshot_id",
            hash_field="snapshot_sha256",
            subject="source_snapshot",
            path="$.source_snapshot",
        )

    prediction_value = record.get("prediction_seal")
    if prediction_value is not None:
        prediction = _mapping(prediction_value, path="$.prediction_seal")
        visualization_value = prediction.get("visualization_evidence")
        if visualization_value is not None:
            _validate_canonical_self_hash(
                _mapping(
                    visualization_value,
                    path="$.prediction_seal.visualization_evidence",
                ),
                hash_field="visualization_evidence_sha256",
                subject="visualization_evidence",
                path="$.prediction_seal.visualization_evidence",
            )
        _validate_object_identity(
            prediction,
            id_field="prediction_seal_id",
            hash_field="prediction_seal_sha256",
            subject="prediction_seal",
            path="$.prediction_seal",
        )
        if (
            snapshot is not None
            and prediction.get("source_snapshot_sha256")
            != snapshot.get("snapshot_sha256")
        ):
            _fail(
                "prediction_source_snapshot_hash_mismatch",
                "PredictionSeal must bind the enclosing SourceSnapshot identity",
                path="$.prediction_seal.source_snapshot_sha256",
            )
        event_bindings = (
            ("P0_event_set", "P0_event_set_sha256"),
            ("R30_event_set", "R30_event_set_sha256"),
            ("RP30_event_set", "RP30_event_set_sha256"),
        )
        if snapshot is not None and any(
            event_set_field in snapshot or seal_field in prediction
            for event_set_field, seal_field in event_bindings
        ):
            for event_set_field, seal_field in event_bindings:
                event_set = _mapping(
                    snapshot.get(event_set_field),
                    path=f"$.source_snapshot.{event_set_field}",
                )
                if prediction.get(seal_field) != event_set.get(
                    "event_rows_content_sha256"
                ):
                    _fail(
                        "prediction_event_set_hash_mismatch",
                        (
                            f"PredictionSeal.{seal_field} must bind the enclosing "
                            f"SourceSnapshot.{event_set_field} content identity"
                        ),
                        path=f"$.prediction_seal.{seal_field}",
                    )

    truth_value = record.get("truth_snapshot")
    if truth_value is not None:
        truth_snapshot = _mapping(truth_value, path="$.truth_snapshot")
        _validate_truth_snapshot_typed_artifacts(
            truth_snapshot,
            path="$.truth_snapshot",
        )
        _validate_object_identity(
            truth_snapshot,
            id_field="truth_snapshot_id",
            hash_field="truth_snapshot_sha256",
            subject="truth_snapshot",
            path="$.truth_snapshot",
        )

    revised_target_set = record.get("revised_target_set")
    if revised_target_set is not None:
        _validate_target_set_identity(
            revised_target_set,
            path="$.revised_target_set",
        )

    replay_value = record.get("replay_visualization")
    if replay_value is not None:
        replay = _mapping(replay_value, path="$.replay_visualization")
        _validate_canonical_self_hash(
            replay,
            hash_field="replay_visualization_sha256",
            subject="replay_visualization",
            path="$.replay_visualization",
        )
        record_type = record.get("record_type")
        if (
            record_type == "MatureTruthSnapshotRecord"
            and replay.get("previous_replay_visualization_sha256") is not None
        ):
            _fail(
                "mature_truth_replay_previous_not_null",
                "first mature truth replay must have a null previous replay hash",
                path="$.replay_visualization.previous_replay_visualization_sha256",
            )
        target_set: Mapping[str, object] | None = None
        if record_type == "MatureTruthSnapshotRecord" and isinstance(truth_value, Mapping):
            candidate = truth_value.get("realized_target_set")
            if isinstance(candidate, Mapping):
                target_set = candidate
        elif record_type == "TruthRevisionRecord":
            candidate = record.get("revised_target_set")
            if isinstance(candidate, Mapping):
                target_set = candidate
        if (
            target_set is not None
            and replay.get("truth_target_set_sha256")
            != target_set.get("target_rows_content_sha256")
        ):
            _fail(
                "replay_truth_target_set_hash_mismatch",
                "replay truth_target_set_sha256 must bind the current target rows",
                path="$.replay_visualization.truth_target_set_sha256",
            )


def _validate_runtime(runtime: Mapping[str, object], *, path: str) -> None:
    physical = _integer(runtime.get("physical_core_count"), path=f"{path}.physical_core_count")
    workers = _integer(runtime.get("worker_count"), path=f"{path}.worker_count")
    expected_workers = min(8, max(1, physical - 2))
    if workers != expected_workers:
        _fail(
            "worker_count_not_frozen_default",
            "worker_count must equal min(8, max(1, physical_core_count - 2)) "
            f"= {expected_workers}",
            path=f"{path}.worker_count",
        )
    total_memory = _integer(
        runtime.get("host_total_physical_memory_bytes"),
        path=f"{path}.host_total_physical_memory_bytes",
    )
    available_memory = _integer(
        runtime.get("host_available_memory_bytes_at_start"),
        path=f"{path}.host_available_memory_bytes_at_start",
    )
    peak_resident = _integer(
        runtime.get("peak_resident_set_bytes"),
        path=f"{path}.peak_resident_set_bytes",
    )
    if available_memory > total_memory:
        _fail(
            "host_available_memory_exceeds_total",
            "host available memory at start cannot exceed total physical memory",
            path=f"{path}.host_available_memory_bytes_at_start",
        )
    if peak_resident > total_memory:
        _fail(
            "peak_resident_set_exceeds_total_memory",
            "peak resident set cannot exceed total physical memory",
            path=f"{path}.peak_resident_set_bytes",
        )

    gpu_fields = (
        "gpu_model",
        "gpu_driver_version",
        "gpu_runtime_version",
    )
    receipt = runtime.get("GPU_equivalence_receipt_sha256")
    execution_device = runtime.get("execution_device")
    if execution_device == "CPU":
        if receipt is not None or any(runtime.get(field) is not None for field in gpu_fields):
            _fail(
                "cpu_runtime_contains_gpu_identity",
                "CPU execution requires null GPU identity fields and null equivalence receipt",
                path=path,
            )
    elif execution_device == "GPU_equivalent":
        if not isinstance(receipt, str) or not receipt:
            _fail(
                "gpu_runtime_equivalence_receipt_missing",
                "GPU-equivalent execution requires a frozen equivalence receipt",
                path=f"{path}.GPU_equivalence_receipt_sha256",
            )
        for field in gpu_fields:
            value = runtime.get(field)
            if not isinstance(value, str) or not value:
                _fail(
                    "gpu_runtime_identity_missing",
                    "GPU-equivalent execution requires model, driver, and runtime identities",
                    path=f"{path}.{field}",
                )
    else:
        _fail(
            "runtime_execution_device_invalid",
            "execution_device must be CPU or GPU_equivalent",
            path=f"{path}.execution_device",
        )


def _validate_prediction_seal(seal: Mapping[str, object], *, path: str) -> None:
    grid_identity = seal.get("grid_identity_sha256")
    forecasts: dict[str, Mapping[str, object]] = {}
    for model in ("P0", "P1", "PP"):
        forecast = _mapping(seal.get(model), path=f"{path}.{model}")
        forecasts[model] = forecast
        _validate_forecast_identity(forecast, path=f"{path}.{model}")
        if forecast.get("grid_identity_sha256") != grid_identity:
            _fail(
                "forecast_grid_identity_mismatch",
                f"{model}.grid_identity_sha256 must equal PredictionSeal.grid_identity_sha256",
                path=f"{path}.{model}.grid_identity_sha256",
            )
    actual_areas = [
        _number(
            forecasts[model].get("actual_alarm_area_km2"),
            path=f"{path}.{model}.actual_alarm_area_km2",
        )
        for model in ("P0", "P1", "PP")
    ]
    calculated_area_difference = max(actual_areas) - min(actual_areas)
    reported_area_difference = _number(
        seal.get("maximum_pairwise_actual_alarm_area_difference_km2"),
        path=f"{path}.maximum_pairwise_actual_alarm_area_difference_km2",
    )
    if not math.isclose(
        reported_area_difference,
        calculated_area_difference,
        rel_tol=0.0,
        abs_tol=_ARITHMETIC_ABS_TOLERANCE,
    ):
        _fail(
            "maximum_pairwise_alarm_area_difference_mismatch",
            "maximum pairwise alarm area difference must be recomputed from P0/P1/PP",
            path=f"{path}.maximum_pairwise_actual_alarm_area_difference_km2",
        )

    if (
        _integer(seal.get("R30_event_count"), path=f"{path}.R30_event_count") == 0
        and _canonical_identity(forecasts["P1"], path=f"{path}.P1")
        != _canonical_identity(
            forecasts["P0"],
            path=f"{path}.P0",
        )
    ):
        _fail(
            "empty_r30_forecast_not_byte_identical_to_p0",
            "R30_event_count=0 requires canonical P1 ForecastIdentity bytes to equal P0",
            path=f"{path}.P1",
        )
    if (
        _integer(seal.get("RP30_event_count"), path=f"{path}.RP30_event_count") == 0
        and _canonical_identity(forecasts["PP"], path=f"{path}.PP")
        != _canonical_identity(
            forecasts["P0"],
            path=f"{path}.P0",
        )
    ):
        _fail(
            "empty_rp30_forecast_not_byte_identical_to_p0",
            "RP30_event_count=0 requires canonical PP ForecastIdentity bytes to equal P0",
            path=f"{path}.PP",
        )

    runtime = seal.get("runtime_evidence")
    if runtime is not None:
        _validate_runtime(
            _mapping(runtime, path=f"{path}.runtime_evidence"),
            path=f"{path}.runtime_evidence",
        )
    visualization_value = seal.get("visualization_evidence")
    if isinstance(visualization_value, Mapping):
        visualized_bundle = visualization_value.get(
            "visualized_forecast_bundle_manifest_sha256"
        )
        if visualized_bundle != seal.get("forecast_bundle_manifest_sha256"):
            _fail(
                "visualization_forecast_bundle_manifest_mismatch",
                "visualization evidence must bind the enclosing forecast bundle manifest",
                path=(
                    f"{path}.visualization_evidence."
                    "visualized_forecast_bundle_manifest_sha256"
                ),
            )


def _optional_boolean(mapping: Mapping[str, object], names: Sequence[str], *, path: str) -> bool:
    present = [name for name in names if name in mapping]
    if not present:
        return True
    values = []
    for name in present:
        value = mapping[name]
        if not isinstance(value, bool):
            _fail("expected_boolean", "value must be boolean", path=f"{path}.{name}")
        values.append(value)
    if len(set(values)) != 1:
        _fail(
            "truth_availability_evidence_inconsistent",
            "truth-availability boolean evidence disagrees",
            path=path,
        )
    return values[0]


def _optional_unavailable_count(mapping: Mapping[str, object], *, path: str) -> int:
    present = [name for name in _TRUTH_UNAVAILABLE_COUNT_FIELDS if name in mapping]
    if not present:
        return 0
    values = [_integer(mapping[name], path=f"{path}.{name}") for name in present]
    if len(set(values)) != 1:
        _fail(
            "truth_unavailable_counts_inconsistent",
            "truth-unavailable count evidence disagrees",
            path=path,
        )
    return values[0]


def _truth_available(mapping: Mapping[str, object], *, complete: int, path: str) -> bool:
    unavailable = _optional_unavailable_count(mapping, path=path)
    count_value = mapping.get("truth_available_complete_exposure_count")
    if count_value is not None:
        available = _integer(
            count_value,
            path=f"{path}.truth_available_complete_exposure_count",
        )
        if available > complete:
            _fail(
                "truth_available_exposures_exceed_complete_exposures",
                "truth-available complete exposure count cannot exceed complete exposure count",
                path=f"{path}.truth_available_complete_exposure_count",
            )
        if available + unavailable != complete:
            _fail(
                "truth_exposure_partition_mismatch",
                "complete exposures must equal truth-available plus selected-unavailable exposures",
                path=path,
            )
        counted_available = available == complete and unavailable == 0
    else:
        counted_available = unavailable == 0

    if any(name in mapping for name in _TRUTH_AVAILABLE_BOOLEAN_FIELDS):
        declared_available = _optional_boolean(
            mapping,
            _TRUTH_AVAILABLE_BOOLEAN_FIELDS,
            path=path,
        )
        if declared_available != counted_available:
            _fail(
                "truth_availability_evidence_inconsistent",
                "truth availability evidence must agree with complete exposure counts",
                path=path,
            )
    return counted_available


def _reason_codes(mapping: Mapping[str, object], *, path: str) -> tuple[str, ...]:
    present = [name for name in _REASON_FIELDS if name in mapping]
    if not present:
        return ()
    if len(present) > 1:
        _fail(
            "multiple_not_evaluable_reason_fields",
            "exactly one structured not-evaluable reason field is allowed",
            path=path,
        )
    value = mapping[present[0]]
    if value is None:
        return ()
    if isinstance(value, str):
        values: Sequence[object] = [value]
    else:
        values = _sequence(value, path=f"{path}.{present[0]}")
    result: list[str] = []
    for index, item in enumerate(values):
        if not isinstance(item, str) or not item:
            _fail(
                "invalid_not_evaluable_reason",
                "not-evaluable reasons must be non-empty strings",
                path=f"{path}.{present[0]}[{index}]",
            )
        result.append(item)
    return tuple(result)


def _reason_categories(reasons: Sequence[str]) -> set[str]:
    categories: set[str] = set()
    for reason in reasons:
        lowered = reason.casefold()
        if "exposure" in lowered and "truth" not in lowered:
            categories.add("exposure")
        if "truth" in lowered or "unavailable" in lowered:
            categories.add("truth")
        if "support" in lowered:
            categories.add("supported")
        if "full" in lowered or "study_area" in lowered or "all_region" in lowered:
            categories.add("full")
        if "cluster" in lowered:
            categories.add("cluster")
        if "density" in lowered or "finite" in lowered:
            categories.add("density")
    return categories


def _validate_horizon_evaluability(horizon: Mapping[str, object], *, path: str) -> None:
    complete = _integer(
        horizon.get("complete_exposure_count"),
        path=f"{path}.complete_exposure_count",
    )
    reasons = _reason_codes(horizon, path=path)
    unavailable_reasons = {
        "formal_freeze_unavailable",
        "scheduled_issue_cap_terminal_before_formal_freeze",
    }
    if len(reasons) == 1 and reasons[0] in unavailable_reasons:
        _truth_available(horizon, complete=complete, path=path)
        for field in (
            "supported_unique_M5_6_event_count",
            "full_study_area_unique_M5_6_event_count",
            "unique_target_cluster_count",
            "all_required_forecast_densities_finite_positive",
        ):
            if horizon.get(field) is not None:
                _fail(
                    "formal_freeze_unavailable_horizon_contains_derived_value",
                    f"{field} must be null when formal target rows are unavailable",
                    path=f"{path}.{field}",
                )
        if horizon.get("evaluable") is not False:
            _fail(
                "formal_freeze_unavailable_horizon_evaluable",
                "a horizon without a formal freeze cannot be evaluable",
                path=f"{path}.evaluable",
            )
        return
    supported = _integer(
        horizon.get("supported_unique_M5_6_event_count"),
        path=f"{path}.supported_unique_M5_6_event_count",
    )
    full = _integer(
        horizon.get("full_study_area_unique_M5_6_event_count"),
        path=f"{path}.full_study_area_unique_M5_6_event_count",
    )
    clusters = _integer(
        horizon.get("unique_target_cluster_count"),
        path=f"{path}.unique_target_cluster_count",
    )
    truth_available = _truth_available(horizon, complete=complete, path=path)
    density_valid = _optional_boolean(horizon, _DENSITY_VALID_BOOLEAN_FIELDS, path=path)
    expected = (
        complete > 0
        and truth_available
        and supported > 0
        and full > 0
        and clusters > 0
        and density_valid
    )
    evaluable = horizon.get("evaluable")
    if not isinstance(evaluable, bool):
        _fail("expected_boolean", "evaluable must be boolean", path=f"{path}.evaluable")
    if evaluable != expected:
        _fail(
            "horizon_evaluable_not_derived_from_evidence",
            "evaluable must be recomputed from exposure, availability, denominators, "
            "clusters, and density evidence",
            path=f"{path}.evaluable",
        )

    if evaluable:
        if reasons:
            _fail(
                "evaluable_horizon_has_failure_reason",
                "an evaluable horizon cannot carry not-evaluable reasons",
                path=path,
            )
        return
    if not reasons:
        _fail(
            "not_evaluable_reason_missing",
            "evaluable=false requires a structured reason",
            path=path,
        )

    failures = {
        name
        for name, failed in (
            ("exposure", complete <= 0),
            ("truth", not truth_available),
            ("supported", supported <= 0),
            ("full", full <= 0),
            ("cluster", clusters <= 0),
            ("density", not density_valid),
        )
        if failed
    }
    categories = _reason_categories(reasons)
    if not categories:
        _fail(
            "not_evaluable_reason_unrecognized",
            "structured reason must identify a frozen evaluability condition",
            path=path,
        )
    if not categories.issubset(failures):
        _fail(
            "not_evaluable_reason_inconsistent",
            "structured reasons claim a failure not supported by evidence",
            path=path,
        )


def _validate_selected_exposure_manifest(
    record: Mapping[str, object],
) -> tuple[Mapping[str, object], tuple[Mapping[str, object], ...]] | None:
    manifest_value = record.get("selected_exposure_manifest")
    binding_fields = (
        "selected_exposure_manifest_union_sha256",
        "truth_availability_manifest_union_sha256",
    )
    if manifest_value is None and not any(field in record for field in binding_fields):
        return None
    manifest = _mapping(
        manifest_value,
        path="$.selected_exposure_manifest",
    )
    frozen_header = {
        "profile": "stage2p_selected_exposure_manifest_v1",
        "selection_rule": (
            "earliest_issue_greedy_nonoverlap_separately_within_each_horizon"
        ),
        "selection_inputs": [
            "on_time_status",
            "issue_time_utc",
            "horizon_days",
        ],
        "horizon_order": [7, 30, 90],
        "zero_event_exposures_retained": True,
        "unavailable_exposure_replacement_forbidden": True,
    }
    for field, expected in frozen_header.items():
        if manifest.get(field) != expected:
            _fail(
                "selected_exposure_manifest_header_mismatch",
                f"{field} must equal the frozen target-blind selection contract",
                path=f"$.selected_exposure_manifest.{field}",
            )
    candidate_axis = _sequence(
        manifest.get("candidate_issue_prediction_seal_sha256"),
        path=(
            "$.selected_exposure_manifest."
            "candidate_issue_prediction_seal_sha256"
        ),
    )
    record_axis = _sequence(
        record.get("ordered_issue_prediction_seal_sha256"),
        path="$.ordered_issue_prediction_seal_sha256",
    )
    if list(candidate_axis) != list(record_axis):
        _fail(
            "selected_exposure_candidate_axis_mismatch",
            "selected exposure candidate axis must equal the frozen issue seal axis",
            path=(
                "$.selected_exposure_manifest."
                "candidate_issue_prediction_seal_sha256"
            ),
        )
    if any(not isinstance(value, str) for value in candidate_axis):
        _fail(
            "selected_exposure_candidate_axis_invalid",
            "every candidate axis value must be a prediction seal SHA-256",
            path=(
                "$.selected_exposure_manifest."
                "candidate_issue_prediction_seal_sha256"
            ),
        )
    candidate_set = set(candidate_axis)
    rows_value = _sequence(
        manifest.get("rows"),
        path="$.selected_exposure_manifest.rows",
    )
    rows = tuple(
        _mapping(value, path=f"$.selected_exposure_manifest.rows[{index}]")
        for index, value in enumerate(rows_value)
    )
    horizon_order = {7: 0, 30: 1, 90: 2}
    actual_sort_keys: list[tuple[int, int, int, bytes]] = []
    rows_by_horizon: dict[int, list[Mapping[str, object]]] = {
        7: [],
        30: [],
        90: [],
    }
    previous_issue_time: dict[int, datetime] = {}
    previous_scheduled_sequence: dict[int, int] = {}
    seen_horizon_seals: set[tuple[int, str]] = set()
    projection_fields = (
        "horizon_days",
        "selection_ordinal_1_based",
        "issue_id",
        "prediction_seal_sha256",
        "selected_truth_record_sha256",
        "selected_truth_revision_sequence",
        "truth_record_status",
        "truth_available",
    )
    truth_projection: list[Mapping[str, object]] = []
    for index, row in enumerate(rows):
        row_path = f"$.selected_exposure_manifest.rows[{index}]"
        horizon_days = _integer(
            row.get("horizon_days"),
            path=f"{row_path}.horizon_days",
        )
        if horizon_days not in horizon_order:
            _fail(
                "selected_exposure_horizon_invalid",
                "selected exposure horizon must be 7, 30, or 90 days",
                path=f"{row_path}.horizon_days",
            )
        ordinal = _integer(
            row.get("selection_ordinal_1_based"),
            path=f"{row_path}.selection_ordinal_1_based",
        )
        scheduled_sequence = _integer(
            row.get("scheduled_issue_sequence"),
            path=f"{row_path}.scheduled_issue_sequence",
        )
        issue_id = row.get("issue_id")
        prediction_seal_sha256 = row.get("prediction_seal_sha256")
        if not isinstance(issue_id, str) or not isinstance(
            prediction_seal_sha256,
            str,
        ):
            _fail(
                "selected_exposure_identity_invalid",
                "issue_id and prediction_seal_sha256 must be strings",
                path=row_path,
            )
        if prediction_seal_sha256 not in candidate_set:
            _fail(
                "selected_exposure_not_from_candidate_axis",
                "every selected prediction seal must come from the frozen candidate axis",
                path=f"{row_path}.prediction_seal_sha256",
            )
        horizon_rows = rows_by_horizon[horizon_days]
        if ordinal != len(horizon_rows) + 1:
            _fail(
                "selected_exposure_ordinal_not_contiguous",
                "selection ordinals must be contiguous from one within each horizon",
                path=f"{row_path}.selection_ordinal_1_based",
            )
        seal_key = (horizon_days, prediction_seal_sha256)
        if seal_key in seen_horizon_seals:
            _fail(
                "selected_exposure_duplicate_within_horizon",
                "a prediction seal may occur at most once within a horizon",
                path=f"{row_path}.prediction_seal_sha256",
            )
        seen_horizon_seals.add(seal_key)
        issue_time = _utc_timestamp(
            row.get("issue_time_utc"),
            path=f"{row_path}.issue_time_utc",
        )
        target_start = _utc_timestamp(
            row.get("target_start_exclusive_utc"),
            path=f"{row_path}.target_start_exclusive_utc",
        )
        target_end = _utc_timestamp(
            row.get("target_end_inclusive_utc"),
            path=f"{row_path}.target_end_inclusive_utc",
        )
        if target_start != issue_time or target_end != (
            issue_time + timedelta(days=horizon_days)
        ):
            _fail(
                "selected_exposure_target_window_mismatch",
                "target window must be (issue time, issue time plus horizon]",
                path=row_path,
            )
        if horizon_days in previous_issue_time and issue_time < (
            previous_issue_time[horizon_days] + timedelta(days=horizon_days)
        ):
            _fail(
                "selected_exposure_windows_overlap",
                "selected exposures must be nonoverlapping within each horizon",
                path=f"{row_path}.issue_time_utc",
            )
        if horizon_days in previous_scheduled_sequence and scheduled_sequence <= (
            previous_scheduled_sequence[horizon_days]
        ):
            _fail(
                "selected_exposure_scheduled_sequence_not_increasing",
                "scheduled issue sequence must increase within each horizon",
                path=f"{row_path}.scheduled_issue_sequence",
            )
        truth_available = row.get("truth_available")
        truth_status = row.get("truth_record_status")
        expected_truth_available = {
            "mature_truth_sealed": True,
            "truth_snapshot_unavailable": False,
        }.get(truth_status)
        if (
            expected_truth_available is None
            or truth_available is not expected_truth_available
        ):
            _fail(
                "selected_exposure_truth_availability_mismatch",
                "truth_available must be derived from truth_record_status",
                path=f"{row_path}.truth_available",
            )
        previous_issue_time[horizon_days] = issue_time
        previous_scheduled_sequence[horizon_days] = scheduled_sequence
        horizon_rows.append(row)
        actual_sort_keys.append(
            (
                horizon_order[horizon_days],
                ordinal,
                scheduled_sequence,
                issue_id.encode("utf-8"),
            )
        )
        truth_projection.append({field: row.get(field) for field in projection_fields})
    if actual_sort_keys != sorted(actual_sort_keys):
        _fail(
            "selected_exposure_rows_not_sorted",
            "selected exposure rows must use the frozen horizon and ordinal order",
            path="$.selected_exposure_manifest.rows",
        )
    if record.get("formal_freeze_status") == "not_run_no_complete_scope" and rows:
        _fail(
            "formal_no_complete_scope_contains_selected_exposure",
            "not_run_no_complete_scope is allowed only when selected rows are empty",
            path="$.selected_exposure_manifest.rows",
        )

    manifest_sha256 = _sha256_bytes(canonical_json_bytes(manifest))
    if record.get("selected_exposure_manifest_union_sha256") != manifest_sha256:
        _fail(
            "selected_exposure_manifest_hash_mismatch",
            "top-level selected exposure hash must identify the canonical manifest",
            path="$.selected_exposure_manifest_union_sha256",
        )
    truth_union_sha256 = _sha256_bytes(canonical_json_bytes(truth_projection))
    if record.get("truth_availability_manifest_union_sha256") != (
        truth_union_sha256
    ):
        _fail(
            "truth_availability_manifest_union_hash_mismatch",
            "truth availability union hash must identify the exact row projection",
            path="$.truth_availability_manifest_union_sha256",
        )
    horizons = _sequence(
        record.get("horizon_evaluability"),
        path="$.horizon_evaluability",
    )
    horizon_map: dict[int, Mapping[str, object]] = {}
    for index, value in enumerate(horizons):
        horizon = _mapping(value, path=f"$.horizon_evaluability[{index}]")
        days = _integer(
            horizon.get("horizon_days"),
            path=f"$.horizon_evaluability[{index}].horizon_days",
        )
        if days in horizon_map:
            _fail(
                "selected_exposure_horizon_duplicate",
                "horizon evaluability rows must be unique by horizon",
                path=f"$.horizon_evaluability[{index}].horizon_days",
            )
        horizon_map[days] = horizon
    if set(horizon_map) != {7, 30, 90}:
        _fail(
            "selected_exposure_horizon_set_mismatch",
            "horizon evaluability must contain exactly 7, 30, and 90 days",
            path="$.horizon_evaluability",
        )
    for days in (7, 30, 90):
        selected_rows = rows_by_horizon[days]
        selected_projection = [
            projection
            for projection in truth_projection
            if projection["horizon_days"] == days
        ]
        horizon = horizon_map[days]
        expected_bindings = {
            "complete_exposure_count": len(selected_rows),
            "truth_available_complete_exposure_count": sum(
                row.get("truth_available") is True for row in selected_rows
            ),
            "selected_truth_snapshot_unavailable_count": sum(
                row.get("truth_available") is False for row in selected_rows
            ),
            "selected_exposure_manifest_sha256": _sha256_bytes(
                canonical_json_bytes(selected_rows)
            ),
            "truth_availability_manifest_sha256": _sha256_bytes(
                canonical_json_bytes(selected_projection)
            ),
        }
        for field, expected in expected_bindings.items():
            if horizon.get(field) != expected:
                _fail(
                    "selected_exposure_horizon_projection_mismatch",
                    f"horizon {field} must be derived from selected exposure rows",
                    path=(
                        "$.horizon_evaluability"
                        f"[horizon_days={days}].{field}"
                    ),
                )
    return manifest, rows


def _validate_alarm_area_manifest(
    record: Mapping[str, object],
    selected: (
        tuple[Mapping[str, object], tuple[Mapping[str, object], ...]] | None
    ),
) -> None:
    manifest_value = record.get("alarm_area_manifest")
    if manifest_value is None and "alarm_area_manifest_sha256" not in record:
        return
    if selected is None:
        _fail(
            "alarm_area_selected_exposure_manifest_missing",
            "alarm area manifest requires the selected exposure manifest",
            path="$.alarm_area_manifest",
        )
    _, selected_rows = selected
    manifest = _mapping(manifest_value, path="$.alarm_area_manifest")
    if manifest.get("profile") != "stage2p_alarm_area_manifest_v1":
        _fail(
            "alarm_area_manifest_profile_mismatch",
            "alarm area manifest profile is not frozen",
            path="$.alarm_area_manifest.profile",
        )
    if manifest.get("selected_exposure_manifest_union_sha256") != record.get(
        "selected_exposure_manifest_union_sha256"
    ):
        _fail(
            "alarm_area_selected_exposure_hash_mismatch",
            "alarm area manifest must bind the enclosing selected exposures",
            path="$.alarm_area_manifest.selected_exposure_manifest_union_sha256",
        )
    threshold_hex = manifest.get(
        "maximum_allowed_pairwise_difference_km2_float64_hex"
    )
    if threshold_hex != "4083880000000000" or _validate_float64_bits_hex(
        threshold_hex,
        path=(
            "$.alarm_area_manifest."
            "maximum_allowed_pairwise_difference_km2_float64_hex"
        ),
    ) != 625.0:
        _fail(
            "alarm_area_threshold_mismatch",
            "alarm area maximum pairwise threshold must be exactly 625 km2",
            path=(
                "$.alarm_area_manifest."
                "maximum_allowed_pairwise_difference_km2_float64_hex"
            ),
        )
    selected_by_seal: dict[str, tuple[int, str]] = {}
    for index, row in enumerate(selected_rows):
        seal = row.get("prediction_seal_sha256")
        scheduled = row.get("scheduled_issue_sequence")
        issue_id = row.get("issue_id")
        if not isinstance(seal, str) or not isinstance(scheduled, int) or not isinstance(
            issue_id,
            str,
        ):
            _fail(
                "alarm_area_selected_exposure_identity_invalid",
                "selected exposure identity is incomplete",
                path=f"$.selected_exposure_manifest.rows[{index}]",
            )
        identity = (scheduled, issue_id)
        if seal in selected_by_seal and selected_by_seal[seal] != identity:
            _fail(
                "alarm_area_selected_seal_identity_inconsistent",
                "one prediction seal must have one scheduled issue identity",
                path=f"$.selected_exposure_manifest.rows[{index}]",
            )
        selected_by_seal[seal] = identity
    expected_seals = [
        seal
        for seal, _ in sorted(
            selected_by_seal.items(),
            key=lambda item: (
                item[1][0],
                item[1][1].encode("utf-8"),
                item[0],
            ),
        )
    ]
    entries_value = _sequence(
        manifest.get("entries"),
        path="$.alarm_area_manifest.entries",
    )
    entries = [
        _mapping(value, path=f"$.alarm_area_manifest.entries[{index}]")
        for index, value in enumerate(entries_value)
    ]
    actual_seals = [entry.get("prediction_seal_sha256") for entry in entries]
    if actual_seals != expected_seals:
        _fail(
            "alarm_area_entry_axis_mismatch",
            "alarm entries must equal distinct selected seals in scheduled order",
            path="$.alarm_area_manifest.entries",
        )
    area_fields = (
        "P0_actual_alarm_area_km2_float64_hex",
        "P1_actual_alarm_area_km2_float64_hex",
        "PP_actual_alarm_area_km2_float64_hex",
    )
    for index, entry in enumerate(entries):
        entry_path = f"$.alarm_area_manifest.entries[{index}]"
        seal = entry.get("prediction_seal_sha256")
        expected_identity = selected_by_seal.get(str(seal))
        if expected_identity is None or (
            entry.get("scheduled_issue_sequence"),
            entry.get("issue_id"),
        ) != expected_identity:
            _fail(
                "alarm_area_entry_identity_mismatch",
                "alarm entry identity must repeat its selected issue",
                path=entry_path,
            )
        areas = [
            _validate_float64_bits_hex(
                entry.get(field),
                path=f"{entry_path}.{field}",
            )
            for field in area_fields
        ]
        if any(area < 0 for area in areas):
            _fail(
                "alarm_area_negative",
                "actual alarm area cannot be negative",
                path=entry_path,
            )
        maximum = max(
            abs(areas[0] - areas[1]),
            abs(areas[0] - areas[2]),
            abs(areas[1] - areas[2]),
        )
        expected_maximum_hex = struct.pack(">d", maximum).hex()
        if entry.get(
            "maximum_pairwise_actual_alarm_area_difference_km2_float64_hex"
        ) != expected_maximum_hex:
            _fail(
                "alarm_area_pairwise_maximum_mismatch",
                "entry maximum must be recomputed from P0, P1, and PP areas",
                path=(
                    f"{entry_path}."
                    "maximum_pairwise_actual_alarm_area_difference_km2_float64_hex"
                ),
            )
        if maximum > 625.0 or entry.get(
            "within_maximum_pairwise_difference"
        ) is not True:
            _fail(
                "alarm_area_pairwise_threshold_exceeded",
                "every selected issue must remain within the exact 625 km2 limit",
                path=entry_path,
            )
    manifest_sha256 = _sha256_bytes(canonical_json_bytes(manifest))
    if record.get("alarm_area_manifest_sha256") != manifest_sha256:
        _fail(
            "alarm_area_manifest_hash_mismatch",
            "top-level alarm area hash must identify the canonical manifest",
            path="$.alarm_area_manifest_sha256",
        )


def _validate_evaluation_policy_hashes(
    record: Mapping[str, object],
    protocol: Mapping[str, object],
) -> None:
    if record.get("record_type") != "EvaluationFreezeRecord":
        return
    fields = (
        "bootstrap_plan_sha256",
        "statistics_policy_sha256",
        "region_map_sha256",
    )
    if not any(field in record for field in fields):
        return
    evaluation = _mapping(
        protocol.get("evaluation"),
        path="protocol.evaluation",
    )
    bootstrap = _mapping(
        evaluation.get("bootstrap"),
        path="protocol.evaluation.bootstrap",
    )
    statistics_policy = {
        "profile": "stage2p_statistics_policy_v1",
        **{
            field: _mapping(
                evaluation.get(field),
                path=f"protocol.evaluation.{field}",
            )
            for field in (
                "information_gain",
                "strict_recall",
                "macro_aggregation",
                "target_clusters",
                "simultaneous_inference",
                "pass_gate",
                "regional_robustness",
            )
        },
    }
    regional = _mapping(
        evaluation.get("regional_robustness"),
        path="protocol.evaluation.regional_robustness",
    )
    expected = {
        "bootstrap_plan_sha256": _sha256_bytes(canonical_json_bytes(bootstrap)),
        "statistics_policy_sha256": _sha256_bytes(
            canonical_json_bytes(statistics_policy)
        ),
        "region_map_sha256": regional.get("region_manifest_file_sha256"),
    }
    for field, expected_value in expected.items():
        if record.get(field) != expected_value:
            _fail(
                "evaluation_policy_hash_mismatch",
                f"{field} must be recomputed from the trusted protocol",
                path=f"$.{field}",
            )


def _validate_evaluation_result_binding(record: Mapping[str, object]) -> None:
    if record.get("phase") != "result_seal":
        return
    top_level = record.get("input_freeze_sha256")
    result = _mapping(record.get("confirmatory_result"), path="$.confirmatory_result")
    nested = result.get("input_freeze_sha256")
    if top_level != nested:
        _fail(
            "confirmatory_result_input_freeze_mismatch",
            "top-level and nested confirmatory input_freeze_sha256 must match",
            path="$.confirmatory_result.input_freeze_sha256",
        )
    for field in ("evaluation_code_commit", "evaluation_code_sha256"):
        if field in record and result.get(field) != record[field]:
            _fail(
                "confirmatory_result_execution_identity_mismatch",
                f"confirmatory_result.{field} must equal the top-level frozen identity",
                path=f"$.confirmatory_result.{field}",
            )


def _validate_evaluation_sample_gate(record: Mapping[str, object]) -> None:
    if record.get("record_type") != "EvaluationFreezeRecord":
        return
    horizons = _sequence(
        record.get("horizon_evaluability"),
        path="$.horizon_evaluability",
    )
    horizon_map: dict[int, Mapping[str, object]] = {}
    for index, value in enumerate(horizons):
        horizon = _mapping(value, path=f"$.horizon_evaluability[{index}]")
        days = _integer(
            horizon.get("horizon_days"),
            path=f"$.horizon_evaluability[{index}].horizon_days",
        )
        horizon_map[days] = horizon
    horizons_ready = (
        set(horizon_map) == {7, 30, 90}
        and all(horizon_map[days].get("evaluable") is True for days in (7, 30, 90))
    )
    formal_status = record.get("formal_freeze_status")
    if formal_status in _FORMAL_FREEZE_UNAVAILABLE_STATUSES:
        if len(horizons) != 3 or set(horizon_map) != {7, 30, 90}:
            _fail(
                "formal_freeze_unavailable_horizon_set_mismatch",
                "unavailable formal freeze must retain exactly 7, 30, and 90 day exposure evidence",
                path="$.horizon_evaluability",
            )
        expected_reason = (
            "scheduled_issue_cap_terminal_before_formal_freeze"
            if formal_status == "not_run_scheduled_issue_cap_terminal"
            else "formal_freeze_unavailable"
        )
        for days, horizon in horizon_map.items():
            if (
                horizon.get("evaluable") is not False
                or _reason_codes(
                    horizon,
                    path=f"$.horizon_evaluability[horizon_days={days}]",
                )
                != (expected_reason,)
            ):
                _fail(
                    "formal_freeze_unavailable_horizon_reason_mismatch",
                    "each horizon must carry the exact formal-freeze unavailable reason",
                    path=f"$.horizon_evaluability[horizon_days={days}]",
                )
        for field in (
            "union_unique_supported_M5_6_event_count",
            "union_unique_full_study_area_M5_6_event_count",
            "unique_target_cluster_count",
            "realized_target_union",
            "final_window_membership_sha256",
            "cluster_membership_sha256",
        ):
            if record.get(field) is not None:
                _fail(
                    "formal_freeze_unavailable_contains_derived_result",
                    f"{field} must be null when the formal freeze is unavailable",
                    path=f"$.{field}",
                )
        bootstrap = _mapping(
            record.get("bootstrap_preflight"),
            path="$.bootstrap_preflight",
        )
        if (
            bootstrap.get("status") != "not_run_formal_freeze_unavailable"
            or bootstrap.get("index_matrix_sha256") is not None
            or bootstrap.get("generated_replication_count") != 0
            or bootstrap.get("zero_denominator_replication_count") is not None
            or bootstrap.get("frozen_before_effect_rows_open") is not False
            or bootstrap.get("redraw_or_discard_performed") is not False
        ):
            _fail(
                "formal_freeze_unavailable_bootstrap_not_fail_closed",
                "formal-freeze failure requires a mechanically unopened bootstrap",
                path="$.bootstrap_preflight",
            )
        for field, expected in (
            ("sample_gate_met", False),
            ("confirmatory_effects_authorized", False),
            ("effect_rows_opened_at_utc", None),
            ("confirmatory_result", None),
            ("input_freeze_sha256", None),
        ):
            if record.get(field) != expected:
                _fail(
                    "formal_freeze_unavailable_not_fail_closed",
                    f"{field} must remain {expected!r}",
                    path=f"$.{field}",
                )
        if record.get("phase") != "input_freeze":
            _fail(
                "formal_freeze_unavailable_result_phase_forbidden",
                "formal-freeze failure cannot enter result_seal",
                path="$.phase",
            )
        expected_status = (
            "continue_blind_to_104"
            if record.get("trigger_reason") == "on_time_checkpoint_52"
            else "evidence_insufficient"
        )
        if record.get("status") != expected_status:
            _fail(
                "evaluation_status_self_report_mismatch",
                "formal-freeze unavailable status must follow the frozen checkpoint rule",
                path="$.status",
            )
        return
    supported = _integer(
        record.get("union_unique_supported_M5_6_event_count"),
        path="$.union_unique_supported_M5_6_event_count",
    )
    clusters = _integer(
        record.get("unique_target_cluster_count"),
        path="$.unique_target_cluster_count",
    )
    full = _integer(
        record.get("union_unique_full_study_area_M5_6_event_count"),
        path="$.union_unique_full_study_area_M5_6_event_count",
    )
    realized_union = _mapping(
        record.get("realized_target_union"),
        path="$.realized_target_union",
    )
    realized_count = _integer(
        realized_union.get("unique_event_count"),
        path="$.realized_target_union.unique_event_count",
    )
    if supported > full:
        _fail(
            "evaluation_supported_count_exceeds_full_count",
            "supported target count cannot exceed full-study-area target count",
            path="$.union_unique_supported_M5_6_event_count",
        )
    if realized_count != full:
        _fail(
            "evaluation_realized_union_count_mismatch",
            "realized target union count must equal the full-study-area union count",
            path="$.realized_target_union.unique_event_count",
        )
    if clusters > full:
        _fail(
            "evaluation_cluster_count_exceeds_full_count",
            "unique target-cluster count cannot exceed the full target count",
            path="$.unique_target_cluster_count",
        )
    for days, horizon in horizon_map.items():
        horizon_supported = _integer(
            horizon.get("supported_unique_M5_6_event_count"),
            path=(
                "$.horizon_evaluability"
                f"[horizon_days={days}].supported_unique_M5_6_event_count"
            ),
        )
        horizon_full = _integer(
            horizon.get("full_study_area_unique_M5_6_event_count"),
            path=(
                "$.horizon_evaluability"
                f"[horizon_days={days}].full_study_area_unique_M5_6_event_count"
            ),
        )
        horizon_clusters = _integer(
            horizon.get("unique_target_cluster_count"),
            path=(
                "$.horizon_evaluability"
                f"[horizon_days={days}].unique_target_cluster_count"
            ),
        )
        if (
            horizon_supported > supported
            or horizon_full > full
            or horizon_clusters > clusters
        ):
            _fail(
                "evaluation_union_does_not_cover_horizon",
                "union N/B counts must cover every horizon-specific count",
                path=f"$.horizon_evaluability[horizon_days={days}]",
            )
    basic_gate = (
        record.get("all_90d_windows_mature") is True
        and supported >= 20
        and clusters >= 10
        and horizons_ready
    )
    bootstrap = _mapping(
        record.get("bootstrap_preflight"),
        path="$.bootstrap_preflight",
    )
    bootstrap_status = bootstrap.get("status")
    if basic_gate and bootstrap_status == "not_run_basic_gate_failed":
        _fail(
            "bootstrap_not_run_despite_satisfied_basic_gate",
            "bootstrap preflight cannot self-report not_run when the basic gate passes",
            path="$.bootstrap_preflight.status",
        )
    if not basic_gate and bootstrap_status != "not_run_basic_gate_failed":
        _fail(
            "bootstrap_run_despite_failed_basic_gate",
            "failed basic sample gate requires not_run_basic_gate_failed",
            path="$.bootstrap_preflight.status",
        )
    gate_met = basic_gate and bootstrap_status == "passed"
    if record.get("sample_gate_met") is not gate_met:
        _fail(
            "sample_gate_self_report_mismatch",
            "sample_gate_met must be mechanically derived from N, B, horizons, and bootstrap",
            path="$.sample_gate_met",
        )

    phase = record.get("phase")
    if phase == "input_freeze":
        if record.get("confirmatory_effects_authorized") is not gate_met:
            _fail(
                "confirmatory_authorization_self_report_mismatch",
                "input confirmatory authorization must exactly equal the derived sample gate",
                path="$.confirmatory_effects_authorized",
            )
        expected_status = (
            "confirmatory_effects_authorized"
            if gate_met
            else (
                "continue_blind_to_104"
                if record.get("trigger_reason") == "on_time_checkpoint_52"
                else "evidence_insufficient"
            )
        )
        if record.get("status") != expected_status:
            _fail(
                "evaluation_status_self_report_mismatch",
                "input evaluation status must be derived from its gate and checkpoint",
                path="$.status",
            )
        if record.get("effect_rows_opened_at_utc") is not None:
            _fail(
                "input_freeze_contains_effect_open_time",
                "effect rows remain unopened at input freeze",
                path="$.effect_rows_opened_at_utc",
            )
    elif phase == "result_seal":
        if not gate_met or record.get("confirmatory_effects_authorized") is not True:
            _fail(
                "result_seal_without_derived_sample_gate",
                "result seal requires a mechanically satisfied sample gate",
                path="$",
            )
        if record.get("status") != "confirmatory_result_sealed":
            _fail(
                "evaluation_status_self_report_mismatch",
                "result phase status must be confirmatory_result_sealed",
                path="$.status",
            )
        _utc_timestamp(
            record.get("effect_rows_opened_at_utc"),
            path="$.effect_rows_opened_at_utc",
        )


def _validate_confirmatory_result_semantics(record: Mapping[str, object]) -> None:
    if (
        record.get("record_type") != "EvaluationFreezeRecord"
        or record.get("phase") != "result_seal"
    ):
        return
    result = _mapping(record.get("confirmatory_result"), path="$.confirmatory_result")
    execution_status = result.get("execution_status")
    if execution_status == "invalid_execution":
        if (
            result.get("decision") != "invalid_execution_stop"
            or result.get("stop_action") != "stop_P1_keep_P0"
            or result.get("additional_confirmatory_looks_authorized") is not False
            or result.get("test_tuning_authorized") is not False
        ):
            _fail(
                "invalid_execution_decision_mismatch",
                "invalid execution must fail closed without another look or test tuning",
                path="$.confirmatory_result",
            )
        return
    if execution_status != "valid":
        _fail(
            "confirmatory_execution_status_invalid",
            "confirmatory execution_status must be valid or invalid_execution",
            path="$.confirmatory_result.execution_status",
        )

    endpoints = _mapping(
        result.get("endpoint_results"),
        path="$.confirmatory_result.endpoint_results",
    )
    robustness = _mapping(
        result.get("robustness_results"),
        path="$.confirmatory_result.robustness_results",
    )
    endpoint_keys = (
        "P1_minus_P0_macro_information_gain",
        "P1_minus_PP_macro_information_gain",
        "P1_minus_P0_macro_recall_gain",
        "P1_minus_PP_macro_recall_gain",
    )
    all_lower_positive = True
    all_region_positive = True
    all_cluster_positive = True
    p1_p0_recall_gain = 0.0
    for key in endpoint_keys:
        endpoint = _mapping(
            endpoints.get(key),
            path=f"$.confirmatory_result.endpoint_results.{key}",
        )
        point = _number(
            endpoint.get("point_estimate"),
            path=f"$.confirmatory_result.endpoint_results.{key}.point_estimate",
        )
        lower = _number(
            endpoint.get("familywise_lower_bound"),
            path=f"$.confirmatory_result.endpoint_results.{key}.familywise_lower_bound",
        )
        upper = _number(
            endpoint.get("familywise_upper_bound"),
            path=f"$.confirmatory_result.endpoint_results.{key}.familywise_upper_bound",
        )
        if not lower <= point <= upper:
            _fail(
                "confirmatory_endpoint_interval_inverted",
                "each endpoint must satisfy lower <= point_estimate <= upper",
                path=f"$.confirmatory_result.endpoint_results.{key}",
            )
        lower_positive = lower > 0
        if endpoint.get("familywise_lower_bound_gt_zero") is not lower_positive:
            _fail(
                "endpoint_lower_bound_boolean_mismatch",
                "familywise_lower_bound_gt_zero must be derived from the numeric lower bound",
                path=(
                    "$.confirmatory_result.endpoint_results."
                    f"{key}.familywise_lower_bound_gt_zero"
                ),
            )
        all_lower_positive &= lower_positive
        if key == "P1_minus_P0_macro_recall_gain":
            p1_p0_recall_gain = point

        robust = _mapping(
            robustness.get(key),
            path=f"$.confirmatory_result.robustness_results.{key}",
        )
        robust_cluster_count = _integer(
            robust.get("target_cluster_count"),
            path=(
                "$.confirmatory_result.robustness_results."
                f"{key}.target_cluster_count"
            ),
        )
        if robust_cluster_count != _integer(
            record.get("unique_target_cluster_count"),
            path="$.unique_target_cluster_count",
        ):
            _fail(
                "robustness_target_cluster_count_mismatch",
                "each robustness endpoint cluster count must equal the frozen top-level B",
                path=(
                    "$.confirmatory_result.robustness_results."
                    f"{key}.target_cluster_count"
                ),
            )
        for dimension in ("region", "cluster"):
            after_field = (
                "point_estimate_after_largest_positive_"
                f"{dimension}_removal"
            )
            remains_field = (
                f"remains_positive_after_largest_positive_{dimension}_removal"
            )
            identity_field = f"largest_positive_{dimension}_identity_sha256"
            after = _number(
                robust.get(after_field),
                path=(
                    "$.confirmatory_result.robustness_results."
                    f"{key}.{after_field}"
                ),
            )
            remains_positive = after > 0
            if robust.get(remains_field) is not remains_positive:
                _fail(
                    "robustness_positive_boolean_mismatch",
                    "robustness positivity booleans must be derived from numeric removals",
                    path=(
                        "$.confirmatory_result.robustness_results."
                        f"{key}.{remains_field}"
                    ),
                )
            positive_contribution_exists = point - after > _ARITHMETIC_ABS_TOLERANCE
            identity = robust.get(identity_field)
            if positive_contribution_exists and not isinstance(identity, str):
                _fail(
                    "largest_positive_contribution_identity_missing",
                    "a positive removal contribution requires its frozen winning identity",
                    path=(
                        "$.confirmatory_result.robustness_results."
                        f"{key}.{identity_field}"
                    ),
                )
            if not positive_contribution_exists and identity is not None:
                _fail(
                    "largest_positive_contribution_identity_spurious",
                    "largest-positive identity must be null when no positive contribution exists",
                    path=(
                        "$.confirmatory_result.robustness_results."
                        f"{key}.{identity_field}"
                    ),
                )
            if dimension == "region":
                all_region_positive &= remains_positive
            else:
                all_cluster_positive &= remains_positive

    recall_gate = p1_p0_recall_gain >= 0.05
    derived_flags = {
        "all_four_familywise_lower_bounds_gt_zero": all_lower_positive,
        "P1_minus_P0_macro_recall_point_gain_gte_0_05": recall_gate,
        "all_four_region_removals_remain_positive": all_region_positive,
        "all_four_cluster_removals_remain_positive": all_cluster_positive,
    }
    for field, expected in derived_flags.items():
        if result.get(field) is not expected:
            _fail(
                "confirmatory_summary_boolean_mismatch",
                f"{field} must be derived from the four endpoint records",
                path=f"$.confirmatory_result.{field}",
            )
    formal_gate = all(derived_flags.values())
    if result.get("formal_gate_passed") is not formal_gate:
        _fail(
            "formal_gate_self_report_mismatch",
            "formal_gate_passed must be the conjunction of the four frozen criteria",
            path="$.confirmatory_result.formal_gate_passed",
        )
    expected_decision = "pass_direct_improvement" if formal_gate else "fail_stop"
    expected_stop = (
        "promote_P1_to_next_independent_validation_without_test_tuning"
        if formal_gate
        else "stop_P1_keep_P0"
    )
    if (
        result.get("decision") != expected_decision
        or result.get("stop_action") != expected_stop
        or result.get("additional_confirmatory_looks_authorized") is not False
        or result.get("test_tuning_authorized") is not False
    ):
        _fail(
            "confirmatory_decision_self_report_mismatch",
            "decision and stop action must be mechanically derived and terminal",
            path="$.confirmatory_result",
        )


def _validate_confirmatory_result_timeline(
    record: Mapping[str, object],
) -> None:
    if (
        record.get("record_type") != "EvaluationFreezeRecord"
        or record.get("phase") != "result_seal"
    ):
        return
    effect_opened = _utc_timestamp(
        record.get("effect_rows_opened_at_utc"),
        path="$.effect_rows_opened_at_utc",
    )
    result = _mapping(record.get("confirmatory_result"), path="$.confirmatory_result")
    result_sealed = _utc_timestamp(
        result.get("sealed_at_utc"),
        path="$.confirmatory_result.sealed_at_utc",
    )
    outer_frozen = _utc_timestamp(
        record.get("frozen_at_utc"),
        path="$.frozen_at_utc",
    )
    core_frozen = _utc_timestamp(
        record.get("record_core_frozen_at_utc"),
        path="$.record_core_frozen_at_utc",
    )
    if outer_frozen != core_frozen:
        _fail(
            "evaluation_core_frozen_at_mismatch",
            "evaluation frozen_at_utc must equal record_core_frozen_at_utc",
            path="$.record_core_frozen_at_utc",
        )
    if not effect_opened <= result_sealed <= outer_frozen:
        _fail(
            "confirmatory_result_seal_timeline_invalid",
            "effect open <= confirmatory result seal <= outer result freeze must hold",
            path="$.confirmatory_result.sealed_at_utc",
        )


def _validate_result_freeze_identity(
    input_freeze: Mapping[str, object],
    result_seal: Mapping[str, object],
    *,
    path: str,
) -> None:
    fields = (
        "evaluation_code_commit",
        "evaluation_code_sha256",
        "environment_lock_file_sha256",
        "pyproject_file_sha256",
    )
    for field in fields:
        if (
            field in input_freeze or field in result_seal
        ) and input_freeze.get(field) != result_seal.get(field):
            _fail(
                "result_seal_execution_identity_changed",
                f"result_seal must preserve input_freeze {field}",
                path=f"{path}.{field}",
            )


_EVALUATION_RESULT_MUTABLE_FIELDS = {
    "evaluation_sequence",
    "phase",
    "previous_evaluation_freeze_sha256",
    "input_freeze_sha256",
    "frozen_at_utc",
    "record_core_frozen_at_utc",
    "timestamp_deadline_utc",
    "effect_rows_opened_at_utc",
    "status",
    "confirmatory_result",
    "timestamp_attempt_evidence",
    "remote_timestamp",
    "content_sha256",
}


def _validate_evaluation_frozen_fields(
    input_freeze: Mapping[str, object],
    result_seal: Mapping[str, object],
    *,
    path: str,
) -> None:
    for field in sorted(set(input_freeze) | set(result_seal)):
        if field in _EVALUATION_RESULT_MUTABLE_FIELDS:
            continue
        if field not in input_freeze or field not in result_seal:
            _fail(
                "result_seal_frozen_input_changed",
                f"result seal cannot add or remove frozen input field {field}",
                path=f"{path}.{field}",
            )
        try:
            same = canonical_json_bytes(input_freeze[field]) == canonical_json_bytes(
                result_seal[field]
            )
        except (TypeError, ValueError) as exc:
            raise SemanticValidationError(
                "evaluation_frozen_field_not_canonicalizable",
                f"frozen evaluation field {field} cannot be canonicalized",
                path=f"{path}.{field}",
            ) from exc
        if not same:
            _fail(
                "result_seal_frozen_input_changed",
                f"result seal changed frozen input field {field}",
                path=f"{path}.{field}",
            )


def _selected_tsa_attempt(record: Mapping[str, object], *, path: str) -> Mapping[str, object]:
    proof = _mapping(record.get("remote_timestamp"), path=f"{path}.remote_timestamp")
    selected_index = _integer(
        proof.get("selected_attempt_index"),
        path=f"{path}.remote_timestamp.selected_attempt_index",
    )
    attempts = _sequence(
        record.get("timestamp_attempt_evidence"),
        path=f"{path}.timestamp_attempt_evidence",
    )
    selected = [
        _mapping(value, path=f"{path}.timestamp_attempt_evidence[{index}]")
        for index, value in enumerate(attempts)
        if isinstance(value, Mapping) and value.get("attempt_index") == selected_index
    ]
    if len(selected) != 1:
        _fail(
            "selected_tsa_attempt_not_unique",
            "selected TSA attempt must be unique",
            path=f"{path}.remote_timestamp.selected_attempt_index",
        )
    return selected[0]


def _validate_evaluation_effect_open_order(
    input_freeze: Mapping[str, object],
    result_seal: Mapping[str, object],
    *,
    input_path: str,
    result_path: str,
) -> None:
    if result_seal.get("effect_rows_opened_at_utc") is None:
        return
    input_attempt = _selected_tsa_attempt(input_freeze, path=input_path)
    result_attempt = _selected_tsa_attempt(result_seal, path=result_path)
    input_receipt = _utc_timestamp(
        input_attempt.get("response_received_at_utc"),
        path=f"{input_path}.timestamp_attempt_evidence.response_received_at_utc",
    )
    effect_open = _utc_timestamp(
        result_seal.get("effect_rows_opened_at_utc"),
        path=f"{result_path}.effect_rows_opened_at_utc",
    )
    result_core = _utc_timestamp(
        result_seal.get("record_core_frozen_at_utc"),
        path=f"{result_path}.record_core_frozen_at_utc",
    )
    result_request = _utc_timestamp(
        result_attempt.get("request_started_at_utc"),
        path=f"{result_path}.timestamp_attempt_evidence.request_started_at_utc",
    )
    if input_receipt > effect_open:
        _fail(
            "effect_rows_opened_before_input_timestamp_receipt",
            "effect rows may open only after the input freeze TSA response is received",
            path=f"{result_path}.effect_rows_opened_at_utc",
        )
    if effect_open > result_core:
        _fail(
            "result_core_frozen_before_effect_rows_opened",
            "result core cannot freeze before effect rows are opened",
            path=f"{result_path}.record_core_frozen_at_utc",
        )
    if result_core > result_request:
        _fail(
            "result_tsa_request_before_core_frozen",
            "result TSA request cannot start before the result core freeze",
            path=f"{result_path}.timestamp_attempt_evidence.request_started_at_utc",
        )


def _tsa_value_present(attempt: Mapping[str, object], field: str) -> bool:
    value = attempt.get(field)
    return value is not None and value != 0 and value is not False


def _validate_tsa_attempt_outcome(
    attempt: Mapping[str, object],
    *,
    path: str,
) -> None:
    outcome = attempt.get("outcome")
    status = attempt.get("http_status")
    content_type = attempt.get("response_content_type")
    body_count = attempt.get("response_byte_count")
    response_received = attempt.get("response_received_at_utc")
    token_fields = (
        "response_sha256",
        "authority_identity_sha256",
        "trust_chain_sha256",
        "genTime_utc",
    )
    if outcome == "selected_valid":
        if (
            status != 200
            or content_type != "application/timestamp-reply"
            or not isinstance(body_count, int)
            or isinstance(body_count, bool)
            or body_count <= 0
            or response_received is None
            or any(not _tsa_value_present(attempt, field) for field in token_fields)
            or attempt.get("offline_trust_path_valid") is not True
            or attempt.get("genTime_before_deadline") is not True
        ):
            _fail(
                "tsa_selected_valid_evidence_incomplete",
                "selected_valid requires a complete HTTP 200 RFC3161 token and trust receipt",
                path=path,
            )
    elif outcome == "network_failure":
        forbidden = (
            "response_received_at_utc",
            "response_content_type",
            "response_sha256",
            "authority_identity_sha256",
            "trust_chain_sha256",
            "genTime_utc",
        )
        if (
            status is not None
            or any(attempt.get(field) is not None for field in forbidden)
            or body_count not in {None, 0}
            or attempt.get("offline_trust_path_valid") is not False
            or attempt.get("genTime_before_deadline") is not False
        ):
            _fail(
                "tsa_network_failure_contains_response_evidence",
                "network_failure cannot contain HTTP, response, token, or trust evidence",
                path=path,
            )
    elif outcome == "http_failure":
        if (
            not isinstance(status, int)
            or isinstance(status, bool)
            or not 300 <= status <= 599
            or response_received is None
            or any(
                attempt.get(field) is not None
                for field in (
                    "authority_identity_sha256",
                    "trust_chain_sha256",
                    "genTime_utc",
                )
            )
            or attempt.get("offline_trust_path_valid") is not False
        ):
            _fail(
                "tsa_http_failure_evidence_inconsistent",
                "http_failure requires HTTP 300-599 and no parsed trust or genTime evidence",
                path=path,
            )
    elif outcome == "invalid_content_type":
        if (
            status != 200
            or response_received is None
            or content_type in {None, "application/timestamp-reply"}
            or not isinstance(body_count, int)
            or isinstance(body_count, bool)
            or body_count <= 0
            or not _tsa_value_present(attempt, "response_sha256")
        ):
            _fail(
                "tsa_invalid_content_type_evidence_inconsistent",
                "invalid_content_type requires a non-RFC3161 HTTP 200 response body",
                path=path,
            )
    elif outcome == "invalid_response":
        if (
            status != 200
            or content_type != "application/timestamp-reply"
            or response_received is None
            or not isinstance(body_count, int)
            or isinstance(body_count, bool)
            or body_count <= 0
            or not _tsa_value_present(attempt, "response_sha256")
            or attempt.get("offline_trust_path_valid") is not False
        ):
            _fail(
                "tsa_invalid_response_evidence_inconsistent",
                "invalid_response requires a captured but unverifiable RFC3161 reply",
                path=path,
            )
    elif outcome == "invalid_trust_chain":
        required = (
            "response_sha256",
            "authority_identity_sha256",
            "trust_chain_sha256",
            "genTime_utc",
        )
        if (
            status != 200
            or content_type != "application/timestamp-reply"
            or response_received is None
            or any(not _tsa_value_present(attempt, field) for field in required)
            or attempt.get("offline_trust_path_valid") is not False
        ):
            _fail(
                "tsa_invalid_trust_chain_evidence_inconsistent",
                "invalid_trust_chain requires a parsed token with a failed "
                "offline trust-path verification",
                path=path,
            )
    elif outcome == "genTime_not_before_deadline":
        required = (
            "response_sha256",
            "authority_identity_sha256",
            "trust_chain_sha256",
            "genTime_utc",
        )
        if (
            status != 200
            or content_type != "application/timestamp-reply"
            or response_received is None
            or any(not _tsa_value_present(attempt, field) for field in required)
            or attempt.get("offline_trust_path_valid") is not True
            or attempt.get("genTime_before_deadline") is not False
        ):
            _fail(
                "tsa_deadline_failure_evidence_inconsistent",
                "genTime deadline failure requires a fully trusted token and false deadline flag",
                path=path,
            )
    elif outcome is not None:
        _fail(
            "tsa_attempt_outcome_invalid",
            "unrecognized TSA attempt outcome",
            path=f"{path}.outcome",
        )


def _validate_top_level_timestamp_window(
    record: Mapping[str, object],
) -> tuple[datetime, datetime]:
    final_core_frozen = _utc_timestamp(
        record.get("record_core_frozen_at_utc"),
        path="$.record_core_frozen_at_utc",
    )
    deadline = _utc_timestamp(
        record.get("timestamp_deadline_utc"),
        path="$.timestamp_deadline_utc",
    )
    record_type = record.get("record_type")
    is_missed_issue = (
        record_type == "IssueInputSnapshotRecord"
        and record.get("status") == "missed_issue"
    )
    failed_candidate_value = record.get("failed_candidate_on_time_core")
    core_frozen = final_core_frozen
    if is_missed_issue and failed_candidate_value is not None:
        candidate = _mapping(
            failed_candidate_value,
            path="$.failed_candidate_on_time_core",
        )
        if candidate.get("profile") != "stage2p_failed_candidate_on_time_core_v1":
            _fail(
                "failed_candidate_profile_invalid",
                "failed TSA candidate must use the frozen on-time core profile",
                path="$.failed_candidate_on_time_core.profile",
            )
        if candidate.get("candidate_core_sha256") != candidate.get(
            "candidate_core_artifact_sha256"
        ):
            _fail(
                "failed_candidate_core_artifact_hash_mismatch",
                "candidate core identity and content-addressed artifact hash must match",
                path=(
                    "$.failed_candidate_on_time_core."
                    "candidate_core_artifact_sha256"
                ),
            )
        if candidate.get("issue_id") != record.get("issue_id"):
            _fail(
                "failed_candidate_issue_id_mismatch",
                "failed candidate issue_id must equal the enclosing issue",
                path="$.failed_candidate_on_time_core.issue_id",
            )
        if candidate.get("timestamp_deadline_utc") != record.get(
            "timestamp_deadline_utc"
        ):
            _fail(
                "failed_candidate_deadline_mismatch",
                "failed candidate deadline must equal the enclosing T-minus-5 deadline",
                path="$.failed_candidate_on_time_core.timestamp_deadline_utc",
            )
        if candidate.get("local_restricted") is not True:
            _fail(
                "failed_candidate_not_local_restricted",
                "failed candidate core artifact must remain local restricted",
                path="$.failed_candidate_on_time_core.local_restricted",
            )
        core_frozen = _utc_timestamp(
            candidate.get("core_frozen_at_utc"),
            path="$.failed_candidate_on_time_core.core_frozen_at_utc",
        )
        if final_core_frozen < core_frozen:
            _fail(
                "missed_issue_final_core_before_failed_candidate",
                "final missed-issue record cannot freeze before its failed candidate",
                path="$.record_core_frozen_at_utc",
            )
    if not is_missed_issue and core_frozen > deadline:
        _fail(
            "record_core_frozen_after_timestamp_deadline",
            "record core must freeze on or before its timestamp deadline",
            path="$.record_core_frozen_at_utc",
        )
    if record_type == "TargetCohortDefinition":
        expected = _utc_timestamp(
            record.get("valid_from_utc"),
            path="$.valid_from_utc",
        )
        if deadline != expected:
            _fail(
                "target_tsa_deadline_not_valid_from",
                "TargetCohortDefinition timestamp deadline must equal valid_from_utc",
                path="$.timestamp_deadline_utc",
            )
    elif record_type == "IssueInputSnapshotRecord":
        issue_time = _utc_timestamp(
            record.get("issue_time_utc"),
            path="$.issue_time_utc",
        )
        expected = issue_time - timedelta(minutes=5)
        if deadline != expected:
            _fail(
                "issue_tsa_deadline_not_t_minus_five_minutes",
                "Issue candidate timestamp deadline must exactly equal T minus 5 minutes",
                path="$.timestamp_deadline_utc",
            )
        if is_missed_issue and final_core_frozen > issue_time - timedelta(
            minutes=4
        ):
            _fail(
                "missed_issue_final_core_after_t_minus_four_minutes",
                "final missed-issue audit core must freeze by T minus 4 minutes",
                path="$.record_core_frozen_at_utc",
            )
        if is_missed_issue:
            completion_fields = {
                "attempt_completed_at_utc",
                "count_completed_at_utc",
                "fetch_completed_at_utc",
                "query_completed_at_utc",
            }

            def walk_failed_evidence(value: object, *, path: str) -> None:
                if isinstance(value, Mapping):
                    for field, item in value.items():
                        if field in {
                            "missed_audit_timestamp_attempt_evidence",
                            "missed_audit_remote_timestamp",
                        }:
                            continue
                        item_path = f"{path}.{field}"
                        if field in completion_fields and item is not None:
                            completed = _utc_timestamp(item, path=item_path)
                            if completed > final_core_frozen:
                                _fail(
                                    "missed_issue_final_core_before_failed_evidence_completed",
                                    (
                                        "final missed core must freeze after every "
                                        "failed fetch, count, query, and on-time TSA "
                                        "attempt has completed"
                                    ),
                                    path="$.record_core_frozen_at_utc",
                                )
                        walk_failed_evidence(item, path=item_path)
                elif isinstance(value, Sequence) and not isinstance(
                    value,
                    str | bytes | bytearray,
                ):
                    for index, item in enumerate(value):
                        walk_failed_evidence(item, path=f"{path}[{index}]")

            walk_failed_evidence(record, path="$")
    elif record_type in {
        "MatureTruthSnapshotRecord",
        "TruthRevisionRecord",
        "EvaluationFreezeRecord",
    } and deadline != core_frozen + timedelta(minutes=5):
        _fail(
            "timestamp_deadline_not_core_plus_five_minutes",
            "truth, revision, and evaluation timestamp deadlines equal "
            "core freeze plus 5 minutes",
            path="$.timestamp_deadline_utc",
        )
    return core_frozen, deadline


def _validate_missed_issue_audit(record: Mapping[str, object]) -> None:
    if record.get("record_type") != "IssueInputSnapshotRecord":
        return
    audit_deadline_value = record.get("missed_audit_timestamp_deadline_utc")
    attempts_value = record.get("missed_audit_timestamp_attempt_evidence")
    proof_value = record.get("missed_audit_remote_timestamp")
    if record.get("status") != "missed_issue":
        if audit_deadline_value is not None or proof_value is not None:
            _fail(
                "on_time_issue_contains_missed_audit_proof",
                "on-time issues cannot contain the distinct missed-issue audit proof",
                path="$.missed_audit_remote_timestamp",
            )
        if attempts_value is not None and _sequence(
            attempts_value,
            path="$.missed_audit_timestamp_attempt_evidence",
        ):
            _fail(
                "on_time_issue_contains_missed_audit_attempt",
                "on-time issues cannot contain missed-issue audit attempts",
                path="$.missed_audit_timestamp_attempt_evidence",
            )
        return

    issue_time = _utc_timestamp(
        record.get("issue_time_utc"),
        path="$.issue_time_utc",
    )
    audit_deadline = _utc_timestamp(
        audit_deadline_value,
        path="$.missed_audit_timestamp_deadline_utc",
    )
    if audit_deadline != issue_time:
        _fail(
            "missed_audit_deadline_not_issue_time",
            "missed-issue audit timestamp deadline must equal issue T",
            path="$.missed_audit_timestamp_deadline_utc",
        )
    final_core = _utc_timestamp(
        record.get("record_core_frozen_at_utc"),
        path="$.record_core_frozen_at_utc",
    )
    if final_core > issue_time - timedelta(minutes=4):
        _fail(
            "missed_issue_final_core_after_t_minus_four_minutes",
            "final missed-issue audit core must freeze by T minus 4 minutes",
            path="$.record_core_frozen_at_utc",
        )
    attempts = _sequence(
        attempts_value,
        path="$.missed_audit_timestamp_attempt_evidence",
    )
    if not attempts:
        _fail(
            "missed_issue_audit_attempt_missing",
            "a formal missed issue requires a successful, distinct audit TSA attempt",
            path="$.missed_audit_timestamp_attempt_evidence",
        )
    previous_completed: datetime | None = None
    for position, attempt_value in enumerate(attempts):
        path = f"$.missed_audit_timestamp_attempt_evidence[{position}]"
        attempt = _mapping(attempt_value, path=path)
        if attempt.get("attempt_index") != position:
            _fail(
                "missed_audit_tsa_attempt_index_not_contiguous",
                "missed audit TSA attempt indexes must be contiguous from zero",
                path=f"{path}.attempt_index",
            )
        _validate_tsa_attempt_outcome(attempt, path=path)
        request_started = _utc_timestamp(
            attempt.get("request_started_at_utc"),
            path=f"{path}.request_started_at_utc",
        )
        completed = _utc_timestamp(
            attempt.get("attempt_completed_at_utc"),
            path=f"{path}.attempt_completed_at_utc",
        )
        if request_started < final_core:
            _fail(
                "missed_audit_tsa_request_before_final_core",
                "missed audit TSA request cannot precede the final missed core",
                path=f"{path}.request_started_at_utc",
            )
        if previous_completed is not None and request_started < previous_completed:
            _fail(
                "missed_audit_tsa_attempt_started_before_previous_completed",
                "missed audit authorities must be attempted serially",
                path=f"{path}.request_started_at_utc",
            )
        if completed < request_started or completed > audit_deadline:
            _fail(
                "missed_audit_tsa_attempt_outside_deadline",
                "missed audit attempt must complete between core freeze and T",
                path=f"{path}.attempt_completed_at_utc",
            )
        response_value = attempt.get("response_received_at_utc")
        if response_value is not None and _utc_timestamp(
            response_value,
            path=f"{path}.response_received_at_utc",
        ) != completed:
            _fail(
                "missed_audit_tsa_response_not_attempt_completion",
                "missed audit response receipt must equal attempt completion",
                path=f"{path}.attempt_completed_at_utc",
            )
        previous_completed = completed

    proof = _mapping(
        proof_value,
        path="$.missed_audit_remote_timestamp",
    )
    if proof.get("proof_field_name") != "missed_audit_remote_timestamp" or proof.get(
        "preimage_profile"
    ) != "stage2p_missed_audit_rfc3161_core_v1":
        _fail(
            "missed_audit_proof_profile_invalid",
            "missed audit proof must use its distinct field and core profile",
            path="$.missed_audit_remote_timestamp",
        )
    if proof.get("deadline_utc") != audit_deadline_value:
        _fail(
            "missed_audit_proof_deadline_mismatch",
            "missed audit proof deadline must equal the top-level audit deadline",
            path="$.missed_audit_remote_timestamp.deadline_utc",
        )
    selected_index = _integer(
        proof.get("selected_attempt_index"),
        path="$.missed_audit_remote_timestamp.selected_attempt_index",
    )
    if selected_index >= len(attempts):
        _fail(
            "missed_audit_selected_attempt_missing",
            "missed audit proof must select one captured attempt",
            path="$.missed_audit_remote_timestamp.selected_attempt_index",
        )
    selected = _mapping(
        attempts[selected_index],
        path=f"$.missed_audit_timestamp_attempt_evidence[{selected_index}]",
    )
    if selected.get("outcome") != "selected_valid":
        _fail(
            "missed_audit_selected_attempt_not_valid",
            "formal missed audit proof must select a valid TSA response",
            path=(
                f"$.missed_audit_timestamp_attempt_evidence[{selected_index}]."
                "outcome"
            ),
        )
    if any(
        isinstance(item, Mapping) and item.get("outcome") == "selected_valid"
        for item in attempts[:selected_index]
    ):
        _fail(
            "missed_audit_selected_attempt_not_first_valid",
            "missed audit proof must select the first valid authority response",
            path="$.missed_audit_remote_timestamp.selected_attempt_index",
        )
    if proof.get("selected_response_sha256") != selected.get("response_sha256"):
        _fail(
            "missed_audit_selected_response_mismatch",
            "missed audit proof must bind the selected response bytes",
            path="$.missed_audit_remote_timestamp.selected_response_sha256",
        )
    request_started = _utc_timestamp(
        selected.get("request_started_at_utc"),
        path=(
            f"$.missed_audit_timestamp_attempt_evidence[{selected_index}]."
            "request_started_at_utc"
        ),
    )
    generated = _utc_timestamp(
        selected.get("genTime_utc"),
        path=(
            f"$.missed_audit_timestamp_attempt_evidence[{selected_index}]."
            "genTime_utc"
        ),
    )
    response_received = _utc_timestamp(
        selected.get("response_received_at_utc"),
        path=(
            f"$.missed_audit_timestamp_attempt_evidence[{selected_index}]."
            "response_received_at_utc"
        ),
    )
    if not final_core <= request_started <= generated <= response_received:
        _fail(
            "missed_audit_selected_timeline_invalid",
            "missed audit selected core/request/genTime/response order is invalid",
            path="$.missed_audit_remote_timestamp",
        )
    if generated >= issue_time or response_received > issue_time:
        _fail(
            "missed_audit_not_completed_before_issue",
            "missed audit genTime must be before T and its response received by T",
            path="$.missed_audit_remote_timestamp",
        )


def _validate_rfc3161(record: Mapping[str, object]) -> None:
    attempts_value = record.get("timestamp_attempt_evidence")
    if attempts_value is None:
        return
    attempts = _sequence(attempts_value, path="$.timestamp_attempt_evidence")
    core_frozen, deadline = _validate_top_level_timestamp_window(record)
    previous_completed: datetime | None = None
    for position, attempt_value in enumerate(attempts):
        attempt = _mapping(
            attempt_value,
            path=f"$.timestamp_attempt_evidence[{position}]",
        )
        if attempt.get("attempt_index") != position:
            _fail(
                "tsa_attempt_index_not_contiguous",
                "TSA attempt_index must equal its ordered evidence position",
                path=f"$.timestamp_attempt_evidence[{position}].attempt_index",
            )
        _validate_tsa_attempt_outcome(
            attempt,
            path=f"$.timestamp_attempt_evidence[{position}]",
        )
        request_started = _utc_timestamp(
            attempt.get("request_started_at_utc"),
            path=f"$.timestamp_attempt_evidence[{position}].request_started_at_utc",
        )
        attempt_completed = _utc_timestamp(
            attempt.get("attempt_completed_at_utc"),
            path=f"$.timestamp_attempt_evidence[{position}].attempt_completed_at_utc",
        )
        if request_started < core_frozen:
            _fail(
                "tsa_request_started_before_record_core_frozen",
                "every TSA request must start at or after the record core freeze",
                path=f"$.timestamp_attempt_evidence[{position}].request_started_at_utc",
            )
        if previous_completed is not None and request_started < previous_completed:
            _fail(
                "tsa_attempt_started_before_previous_completed",
                "the next TSA authority may be tried only after the prior attempt completes",
                path=f"$.timestamp_attempt_evidence[{position}].request_started_at_utc",
            )
        if attempt_completed < request_started:
            _fail(
                "tsa_attempt_completed_before_request",
                "TSA attempt completion cannot precede its request start",
                path=f"$.timestamp_attempt_evidence[{position}].attempt_completed_at_utc",
            )
        if attempt_completed > deadline:
            _fail(
                "tsa_attempt_completed_after_deadline",
                "every TSA attempt must complete on or before the frozen deadline",
                path=f"$.timestamp_attempt_evidence[{position}].attempt_completed_at_utc",
            )
        response_value = attempt.get("response_received_at_utc")
        if response_value is not None:
            response_received = _utc_timestamp(
                response_value,
                path=(
                    f"$.timestamp_attempt_evidence[{position}]."
                    "response_received_at_utc"
                ),
            )
            if not request_started <= response_received:
                _fail(
                    "tsa_response_outside_attempt_window",
                    "TSA response receipt cannot precede its request start",
                    path=(
                        f"$.timestamp_attempt_evidence[{position}]."
                        "response_received_at_utc"
                    ),
                )
            if response_received != attempt_completed:
                _fail(
                    "tsa_response_receipt_not_attempt_completion",
                    "every response outcome must complete at its response receipt instant",
                    path=(
                        f"$.timestamp_attempt_evidence[{position}]."
                        "attempt_completed_at_utc"
                    ),
                )
        previous_completed = attempt_completed

    proof_value = record.get("remote_timestamp")
    if proof_value is None:
        if (
            record.get("record_type") == "IssueInputSnapshotRecord"
            and record.get("status") == "missed_issue"
        ):
            candidate_value = record.get("failed_candidate_on_time_core")
            if record.get("failure_code") == "timestamp_failure":
                candidate = _mapping(
                    candidate_value,
                    path="$.failed_candidate_on_time_core",
                )
                candidate_sha256 = candidate.get("candidate_core_sha256")
                if not attempts:
                    _fail(
                        "timestamp_failure_without_tsa_attempt",
                        "timestamp_failure requires at least one failed TSA attempt",
                        path="$.timestamp_attempt_evidence",
                    )
                for index, attempt in enumerate(attempts):
                    if (
                        not isinstance(attempt, Mapping)
                        or attempt.get("outcome") == "selected_valid"
                    ):
                        _fail(
                            "timestamp_failure_contains_valid_tsa_attempt",
                            "timestamp_failure may contain only failed TSA attempts",
                            path=f"$.timestamp_attempt_evidence[{index}].outcome",
                        )
                    if attempt.get("attempt_preimage_sha256") != candidate_sha256:
                        _fail(
                            "failed_tsa_attempt_candidate_preimage_mismatch",
                            "every failed TSA request must bind the candidate on-time core",
                            path=(
                                f"$.timestamp_attempt_evidence[{index}]."
                                "attempt_preimage_sha256"
                            ),
                        )
            elif candidate_value is not None or attempts:
                _fail(
                    "non_timestamp_missed_issue_contains_candidate_or_tsa_attempt",
                    "non-timestamp missed issues have no failed candidate or TSA attempts",
                    path="$",
                )
        if any(
            isinstance(attempt, Mapping)
            and attempt.get("outcome") == "selected_valid"
            for attempt in attempts
        ):
            _fail(
                "tsa_valid_attempt_without_remote_proof",
                "a valid TSA response requires a bound remote timestamp proof",
                path="$.remote_timestamp",
            )
        _validate_missed_issue_audit(record)
        return
    proof = _mapping(proof_value, path="$.remote_timestamp")
    top_deadline_value = record.get("timestamp_deadline_utc")
    if proof.get("deadline_utc") != top_deadline_value:
        _fail(
            "tsa_proof_deadline_top_level_mismatch",
            "remote_timestamp.deadline_utc must exactly equal timestamp_deadline_utc",
            path="$.remote_timestamp.deadline_utc",
        )
    selected_index = _integer(
        proof.get("selected_attempt_index"),
        path="$.remote_timestamp.selected_attempt_index",
    )
    selected = [
        _mapping(attempt, path=f"$.timestamp_attempt_evidence[{position}]")
        for position, attempt in enumerate(attempts)
        if isinstance(attempt, Mapping) and attempt.get("attempt_index") == selected_index
    ]
    if len(selected) != 1:
        _fail(
            "selected_tsa_attempt_not_unique",
            "selected_attempt_index must identify exactly one TSA attempt",
            path="$.remote_timestamp.selected_attempt_index",
        )
    attempt = selected[0]
    if attempt.get("outcome") != "selected_valid":
        _fail(
            "selected_tsa_attempt_not_valid",
            "remote timestamp proof must select a selected_valid attempt",
            path=f"$.timestamp_attempt_evidence[{selected_index}].outcome",
        )
    earlier_valid = [
        item
        for item in attempts
        if isinstance(item, Mapping)
        and isinstance(item.get("attempt_index"), int)
        and item["attempt_index"] < selected_index
        and item.get("outcome") == "selected_valid"
    ]
    if earlier_valid:
        _fail(
            "selected_tsa_attempt_not_first_valid",
            "proof must select the first valid TSA response in frozen authority order",
            path="$.remote_timestamp.selected_attempt_index",
        )
    if proof.get("selected_response_sha256") != attempt.get("response_sha256"):
        _fail(
            "selected_tsa_response_mismatch",
            "selected_response_sha256 must equal the selected attempt response_sha256",
            path="$.remote_timestamp.selected_response_sha256",
        )
    request_started = _utc_timestamp(
        attempt.get("request_started_at_utc"),
        path=f"$.timestamp_attempt_evidence[{selected_index}].request_started_at_utc",
    )
    generation_time = _utc_timestamp(
        attempt.get("genTime_utc"),
        path=f"$.timestamp_attempt_evidence[{selected_index}].genTime_utc",
    )
    response_received = _utc_timestamp(
        attempt.get("response_received_at_utc"),
        path=f"$.timestamp_attempt_evidence[{selected_index}].response_received_at_utc",
    )
    if request_started < core_frozen:
        _fail(
            "tsa_request_started_before_record_core_frozen",
            "selected TSA request cannot start before the record core is frozen",
            path=f"$.timestamp_attempt_evidence[{selected_index}].request_started_at_utc",
        )
    if generation_time < request_started:
        _fail(
            "tsa_generation_time_before_request",
            "RFC3161 genTime cannot precede the selected request start",
            path=f"$.timestamp_attempt_evidence[{selected_index}].genTime_utc",
        )
    if response_received < generation_time:
        _fail(
            "tsa_response_received_before_generation_time",
            "selected TSA response receipt cannot precede RFC3161 genTime",
            path=f"$.timestamp_attempt_evidence[{selected_index}].response_received_at_utc",
        )
    if generation_time >= deadline:
        _fail(
            "tsa_generation_time_not_before_deadline",
            "RFC3161 genTime must be strictly earlier than the frozen deadline",
            path=f"$.timestamp_attempt_evidence[{selected_index}].genTime_utc",
        )
    if response_received > deadline:
        _fail(
            "tsa_response_received_after_deadline",
            "selected TSA response must be received on or before the frozen deadline",
            path=f"$.timestamp_attempt_evidence[{selected_index}].response_received_at_utc",
        )

    record_type = record.get("record_type")
    if record_type == "TargetCohortDefinition":
        expected_deadline = _utc_timestamp(
            record.get("valid_from_utc"),
            path="$.valid_from_utc",
        )
        if deadline != expected_deadline:
            _fail(
                "target_tsa_deadline_not_valid_from",
                "TargetCohortDefinition timestamp deadline must equal valid_from_utc",
                path="$.timestamp_deadline_utc",
            )
        definition_frozen = _utc_timestamp(
            record.get("definition_frozen_at_utc"),
            path="$.definition_frozen_at_utc",
        )
        if core_frozen < definition_frozen:
            _fail(
                "target_core_frozen_before_definition",
                "target record core cannot freeze before its definition is frozen",
                path="$.record_core_frozen_at_utc",
            )
    elif record_type == "IssueInputSnapshotRecord":
        issue_time = _utc_timestamp(
            record.get("issue_time_utc"),
            path="$.issue_time_utc",
        )
        if deadline != issue_time - timedelta(minutes=5):
            _fail(
                "issue_tsa_deadline_not_t_minus_five_minutes",
                "Issue candidate timestamp deadline must exactly equal T minus 5 minutes",
                path="$.timestamp_deadline_utc",
            )
        source = record.get("source_snapshot")
        prediction = record.get("prediction_seal")
        lower_bounds: list[tuple[object, str]] = []
        if isinstance(source, Mapping):
            lower_bounds.append(
                (source.get("seal_completed_at_utc"), "$.source_snapshot.seal_completed_at_utc")
            )
        if isinstance(prediction, Mapping):
            lower_bounds.append(
                (prediction.get("sealed_at_utc"), "$.prediction_seal.sealed_at_utc")
            )
        for value, path in lower_bounds:
            if value is not None and core_frozen < _utc_timestamp(value, path=path):
                _fail(
                    "issue_core_frozen_before_input_seal",
                    "issue record core must freeze after source and prediction seals",
                    path="$.record_core_frozen_at_utc",
                )
    elif record_type in {
        "MatureTruthSnapshotRecord",
        "TruthRevisionRecord",
        "EvaluationFreezeRecord",
    }:
        if deadline != core_frozen + timedelta(minutes=5):
            _fail(
                "timestamp_deadline_not_core_plus_five_minutes",
                "truth, revision, and evaluation timestamp deadlines equal "
                "core freeze plus 5 minutes",
                path="$.timestamp_deadline_utc",
            )

    if record_type == "MatureTruthSnapshotRecord":
        lower_bounds = []
        attempts_value = record.get("attempt_evidence")
        if isinstance(attempts_value, Sequence) and not isinstance(
            attempts_value, str | bytes | bytearray
        ):
            for index, item in enumerate(attempts_value):
                if isinstance(item, Mapping) and item.get("fetch_completed_at_utc") is not None:
                    lower_bounds.append(
                        (
                            item["fetch_completed_at_utc"],
                            f"$.attempt_evidence[{index}].fetch_completed_at_utc",
                        )
                    )
        truth_snapshot = record.get("truth_snapshot")
        if isinstance(truth_snapshot, Mapping):
            lower_bounds.append(
                (
                    truth_snapshot.get("seal_completed_at_utc"),
                    "$.truth_snapshot.seal_completed_at_utc",
                )
            )
        for value, path in lower_bounds:
            if value is not None and core_frozen < _utc_timestamp(value, path=path):
                _fail(
                    "mature_truth_core_frozen_before_evidence_complete",
                    "mature truth core must freeze after final fetch and truth seal",
                    path="$.record_core_frozen_at_utc",
                )
    elif record_type == "TruthRevisionRecord":
        lower_bounds = [
            (
                record.get("revision_derived_completed_at_utc"),
                "$.revision_derived_completed_at_utc",
            ),
            (
                record.get("seal_completed_at_utc"),
                "$.seal_completed_at_utc",
            )
        ]
        for value, path in lower_bounds:
            if value is not None and core_frozen < _utc_timestamp(value, path=path):
                _fail(
                    "truth_revision_core_frozen_before_evidence_complete",
                    "truth revision core must freeze after local derivation and seal",
                    path="$.record_core_frozen_at_utc",
                )
    elif record_type == "EvaluationFreezeRecord":
        frozen_at = _utc_timestamp(record.get("frozen_at_utc"), path="$.frozen_at_utc")
        if core_frozen != frozen_at:
            _fail(
                "evaluation_core_frozen_at_mismatch",
                "evaluation record_core_frozen_at_utc must equal frozen_at_utc",
                path="$.record_core_frozen_at_utc",
            )
    _validate_missed_issue_audit(record)


def _validate_record_hashes(record: Mapping[str, object]) -> None:
    if record.get("record_type") not in {
        "TargetCohortDefinition",
        "IssueInputSnapshotRecord",
        "MatureTruthSnapshotRecord",
        "TruthRevisionRecord",
        "EvaluationFreezeRecord",
    }:
        return
    proof_value = record.get("remote_timestamp")
    if proof_value is not None:
        proof = _mapping(proof_value, path="$.remote_timestamp")
        core = {
            key: value
            for key, value in record.items()
            if key
            not in {
                "timestamp_attempt_evidence",
                "remote_timestamp",
                "content_sha256",
            }
        }
        try:
            preimage = hashlib.sha256(canonical_json_bytes(core)).hexdigest()
        except (TypeError, ValueError) as exc:
            raise SemanticValidationError(
                "record_core_not_canonicalizable",
                "record core cannot be encoded as canonical JSON",
            ) from exc
        if proof.get("preimage_sha256") != preimage:
            _fail(
                "record_core_preimage_hash_mismatch",
                "remote timestamp preimage must hash the exact frozen record core",
                path="$.remote_timestamp.preimage_sha256",
            )
        attempts_value = record.get("timestamp_attempt_evidence")
        if isinstance(attempts_value, Sequence) and not isinstance(
            attempts_value,
            str | bytes | bytearray,
        ):
            for index, attempt in enumerate(attempts_value):
                if (
                    not isinstance(attempt, Mapping)
                    or attempt.get("attempt_preimage_sha256") != preimage
                ):
                    _fail(
                        "tsa_attempt_preimage_hash_mismatch",
                        "every TSA request must carry the exact recomputed record-core preimage",
                        path=(
                            f"$.timestamp_attempt_evidence[{index}]."
                            "attempt_preimage_sha256"
                        ),
                    )

    audit_proof_value = record.get("missed_audit_remote_timestamp")
    if audit_proof_value is not None:
        audit_proof = _mapping(
            audit_proof_value,
            path="$.missed_audit_remote_timestamp",
        )
        audit_core = {
            key: value
            for key, value in record.items()
            if key
            not in {
                "missed_audit_timestamp_attempt_evidence",
                "missed_audit_remote_timestamp",
                "content_sha256",
            }
        }
        try:
            audit_preimage = hashlib.sha256(
                canonical_json_bytes(audit_core)
            ).hexdigest()
        except (TypeError, ValueError) as exc:
            raise SemanticValidationError(
                "missed_audit_core_not_canonicalizable",
                "missed audit core cannot be encoded as canonical JSON",
            ) from exc
        if audit_proof.get("preimage_sha256") != audit_preimage:
            _fail(
                "missed_audit_core_preimage_hash_mismatch",
                "missed audit proof must hash the exact distinct audit core",
                path="$.missed_audit_remote_timestamp.preimage_sha256",
            )
        audit_attempts_value = record.get(
            "missed_audit_timestamp_attempt_evidence"
        )
        if isinstance(audit_attempts_value, Sequence) and not isinstance(
            audit_attempts_value,
            str | bytes | bytearray,
        ):
            for index, attempt in enumerate(audit_attempts_value):
                if (
                    not isinstance(attempt, Mapping)
                    or attempt.get("attempt_preimage_sha256")
                    != audit_preimage
                ):
                    _fail(
                        "missed_audit_tsa_attempt_preimage_hash_mismatch",
                        "every audit TSA request must carry the exact audit-core preimage",
                        path=(
                            f"$.missed_audit_timestamp_attempt_evidence[{index}]."
                            "attempt_preimage_sha256"
                        ),
                    )

    complete = {key: value for key, value in record.items() if key != "content_sha256"}
    try:
        content = hashlib.sha256(canonical_json_bytes(complete)).hexdigest()
    except (TypeError, ValueError) as exc:
        raise SemanticValidationError(
            "record_content_not_canonicalizable",
            "complete record cannot be encoded as canonical JSON",
        ) from exc
    if record.get("content_sha256") != content:
        _fail(
            "record_content_hash_mismatch",
            "content_sha256 must hash the complete record excluding only itself",
            path="$.content_sha256",
        )


def _extract_tag_receipts(
    record: Mapping[str, object],
) -> tuple[Mapping[str, object], Mapping[str, object]] | None:
    pairs = (
        ("protocol_remote_tag_receipt", "code_remote_tag_receipt"),
        ("protocol_tag_remote_receipt", "code_tag_remote_receipt"),
        ("protocol_tag_receipt", "code_tag_receipt"),
    )
    for protocol_name, code_name in pairs:
        if protocol_name in record or code_name in record:
            return (
                _mapping(record.get(protocol_name), path=f"$.{protocol_name}"),
                _mapping(record.get(code_name), path=f"$.{code_name}"),
            )
    nested = record.get("remote_tag_receipts")
    if nested is None:
        return None
    receipt_map = _mapping(nested, path="$.remote_tag_receipts")
    protocol_receipt = receipt_map.get("protocol") or receipt_map.get("protocol_tag")
    code_receipt = receipt_map.get("code") or receipt_map.get("code_tag")
    return (
        _mapping(protocol_receipt, path="$.remote_tag_receipts.protocol"),
        _mapping(code_receipt, path="$.remote_tag_receipts.code"),
    )


def _receipt_commit(receipt: Mapping[str, object], *, path: str) -> object:
    for name in (
        "remote_peeled_commit",
        "peeled_commit",
        "peeled_commit_sha",
        "peeled_commit_sha1",
    ):
        if name in receipt:
            return receipt[name]
    _fail(
        "remote_tag_receipt_missing_peeled_commit",
        "remote tag receipt must contain its peeled commit",
        path=path,
    )


def _receipt_verified_at(receipt: Mapping[str, object], *, path: str) -> datetime:
    for name in ("verified_at_utc", "verified_at"):
        if name in receipt:
            return _utc_timestamp(receipt[name], path=f"{path}.{name}")
    _fail(
        "remote_tag_receipt_missing_verified_at",
        "remote tag receipt must contain verified_at_utc",
        path=path,
    )


def _validate_receipt_identity(
    receipt: Mapping[str, object],
    *,
    path: str,
) -> None:
    body = {
        key: value
        for key, value in receipt.items()
        if key != "receipt_sha256"
    }
    calculated = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    if receipt.get("receipt_sha256") != calculated:
        _fail(
            "remote_tag_receipt_hash_mismatch",
            "receipt_sha256 must hash the canonical receipt excluding only itself",
            path=f"{path}.receipt_sha256",
        )


def _validate_target_cohort(record: Mapping[str, object]) -> None:
    receipts = _extract_tag_receipts(record)
    if receipts is None:
        return
    protocol_receipt, code_receipt = receipts
    _validate_receipt_identity(
        protocol_receipt,
        path="$.protocol_tag_remote_receipt",
    )
    _validate_receipt_identity(
        code_receipt,
        path="$.code_tag_remote_receipt",
    )
    protocol_commit = record.get("protocol_commit")
    code_commit = record.get("code_commit", protocol_commit)
    if _receipt_commit(protocol_receipt, path="$.protocol_tag_receipt") != protocol_commit:
        _fail(
            "protocol_tag_peeled_commit_mismatch",
            "protocol tag peeled commit must equal protocol_commit",
            path="$.protocol_tag_receipt.peeled_commit",
        )
    if _receipt_commit(code_receipt, path="$.code_tag_receipt") != code_commit:
        _fail(
            "code_tag_peeled_commit_mismatch",
            "code tag peeled commit must equal code_commit",
            path="$.code_tag_receipt.peeled_commit",
        )
    valid_from = _utc_timestamp(record.get("valid_from_utc"), path="$.valid_from_utc")
    for label, receipt in (("protocol", protocol_receipt), ("code", code_receipt)):
        if _receipt_verified_at(receipt, path=f"$.{label}_tag_receipt") >= valid_from:
            _fail(
                "remote_tag_verified_not_before_valid_from",
                "each remote tag must be verified strictly before valid_from_utc",
                path=f"$.{label}_tag_receipt.verified_at_utc",
            )
    definition_frozen = _utc_timestamp(
        record.get("definition_frozen_at_utc"),
        path="$.definition_frozen_at_utc",
    )
    latest_receipt = max(
        _receipt_verified_at(protocol_receipt, path="$.protocol_tag_receipt"),
        _receipt_verified_at(code_receipt, path="$.code_tag_receipt"),
    )
    threshold_value = record.get(
        "first_issue_not_before_utc",
        _DEFAULT_CALENDAR["first_issue_not_before_utc"],
    )
    threshold = _utc_timestamp(
        threshold_value,
        path="$.first_issue_not_before_utc",
    )
    expected_valid_from = max(
        threshold,
        _next_rule_thursday_after(latest_receipt),
    )
    if valid_from != expected_valid_from:
        _fail(
            "target_valid_from_not_mechanically_derived",
            "valid_from_utc must equal the fixed lower bound or the strict next "
            "rule Thursday after the later verified tag receipt",
            path="$.valid_from_utc",
        )
    if latest_receipt > definition_frozen:
        _fail(
            "target_definition_frozen_before_tag_receipt",
            "TargetCohortDefinition cannot freeze before both remote tag receipts",
            path="$.definition_frozen_at_utc",
        )


def _validate_code_manifest_identity(manifest: Mapping[str, object], *, path: str) -> None:
    declared = manifest.get("manifest_sha256")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    calculated = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    if declared != calculated:
        _fail(
            "code_manifest_hash_mismatch",
            "manifest_sha256 must bind the canonical component-hash mapping",
            path=f"{path}.manifest_sha256",
        )


def _code_manifest_from_context(
    record: Mapping[str, object],
    protocol: Mapping[str, object],
) -> tuple[Mapping[str, object] | None, Mapping[str, object] | None]:
    cohort: Mapping[str, object] | None = None
    if record.get("record_type") == "TargetCohortDefinition":
        cohort = record
    else:
        for name in ("cohort_definition", "target_cohort_definition"):
            value = protocol.get(name)
            if value is not None:
                cohort = _mapping(value, path=f"protocol.{name}")
                break
    manifest_value = record.get("code_manifest") if cohort is record else None
    if manifest_value is None and cohort is not None:
        manifest_value = cohort.get("code_manifest")
    if manifest_value is None:
        manifest_value = protocol.get("code_manifest")
    if manifest_value is None:
        return cohort, None
    return cohort, _mapping(manifest_value, path="code_manifest")


def _walk_code_bindings(
    value: object,
    manifest: Mapping[str, object],
    *,
    path: str = "$",
) -> None:
    component_fields = (
        "parser_code_sha256",
        "normalization_config_sha256",
        "deduplication_code_sha256",
        "deduplication_config_sha256",
        "revision_policy_sha256",
        "model_code_sha256",
        "evaluation_code_sha256",
        "visualization_code_sha256",
        "semantic_validator_code_sha256",
        "environment_lock_file_sha256",
        "pyproject_file_sha256",
    )
    if isinstance(value, Mapping):
        for field in component_fields:
            if field in value and field in manifest and value[field] != manifest[field]:
                _fail(
                    "record_code_manifest_binding_mismatch",
                    f"{field} must equal the TargetCohortDefinition code manifest",
                    path=f"{path}.{field}",
                )
        if "code_manifest_sha256" in value and value["code_manifest_sha256"] != manifest.get(
            "manifest_sha256"
        ):
            _fail(
                "record_code_manifest_hash_mismatch",
                "record code_manifest_sha256 must equal the cohort manifest identity",
                path=f"{path}.code_manifest_sha256",
            )
        for key, item in value.items():
            if key != "code_manifest":
                _walk_code_bindings(item, manifest, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for index, item in enumerate(value):
            _walk_code_bindings(item, manifest, path=f"{path}[{index}]")


def _validate_code_bindings(
    record: Mapping[str, object],
    protocol: Mapping[str, object],
) -> None:
    cohort, manifest = _code_manifest_from_context(record, protocol)
    if manifest is None:
        return
    _validate_code_manifest_identity(manifest, path="$.code_manifest")
    _walk_code_bindings(record, manifest)
    if cohort is not None:
        expected_commit = cohort.get("code_commit")
        for field in ("code_commit", "evaluation_code_commit"):
            if field in record and record[field] != expected_commit:
                _fail(
                    "record_code_commit_mismatch",
                    f"{field} must equal the cohort code_commit",
                    path=f"$.{field}",
                )


def _validate_exchange_outcome(mapping: Mapping[str, object], *, path: str) -> None:
    if "exchange_outcome" not in mapping:
        return
    exchange = mapping["exchange_outcome"]
    final_outcome = mapping.get("outcome")
    status = mapping.get("http_status")
    response_sha = mapping.get("raw_response_sha256")
    if response_sha is None and isinstance(mapping.get("raw_response"), Mapping):
        response_sha = mapping["raw_response"].get("file_sha256")  # type: ignore[union-attr]

    if isinstance(exchange, str):
        normalized = exchange.casefold()
        if normalized == "not_attempted_count_preflight_failed_or_limit_reached":
            if final_outcome == "succeeded":
                _fail(
                    "not_attempted_exchange_marked_succeeded",
                    "a query not attempted after preflight cannot have succeeded",
                    path=f"{path}.outcome",
                )
            if (
                mapping.get("fetch_started_at_utc") is not None
                or mapping.get("fetch_completed_at_utc") is not None
                or status is not None
                or response_sha is not None
            ):
                _fail(
                    "not_attempted_exchange_contains_query_evidence",
                    "a query not attempted after preflight cannot contain query exchange evidence",
                    path=path,
                )
            return
        expected = (
            "network_failure"
            if final_outcome == "network_failure"
            else ("http_failure" if final_outcome == "http_failure" else "response_received")
        )
        if normalized != expected:
            _fail(
                "exchange_outcome_inconsistent",
                "exchange_outcome must agree with the final fetch outcome",
                path=f"{path}.exchange_outcome",
            )
        if normalized == "network_failure":
            if status is not None or response_sha is not None:
                _fail(
                    "network_failure_contains_response_evidence",
                    "network_failure cannot contain HTTP or response identity evidence",
                    path=path,
                )
            return
        if not isinstance(status, int) or isinstance(status, bool):
            _fail(
                "response_exchange_missing_http_status",
                "a received HTTP exchange requires an integer http_status",
                path=f"{path}.http_status",
            )
        if normalized == "http_failure" and status < 300:
            _fail(
                "http_failure_status_inconsistent",
                "http_failure requires HTTP 300-599 when redirects are forbidden",
                path=f"{path}.http_status",
            )
        if normalized == "response_received" and final_outcome == "succeeded":
            if status not in {200, 204}:
                _fail(
                    "successful_fetch_http_status_inconsistent",
                    "successful FDSN fetch requires HTTP 200 or 204",
                    path=f"{path}.http_status",
                )
            if response_sha is None:
                _fail(
                    "successful_fetch_response_identity_missing",
                    "succeeded fetch requires response identity evidence",
                    path=path,
                )
        return

    evidence = _mapping(exchange, path=f"{path}.exchange_outcome")
    if "final_outcome" in evidence and evidence["final_outcome"] != final_outcome:
        _fail(
            "exchange_final_outcome_mismatch",
            "exchange final_outcome must equal the enclosing outcome",
            path=f"{path}.exchange_outcome.final_outcome",
        )
    if "http_status" in evidence and evidence["http_status"] != status:
        _fail(
            "exchange_http_status_mismatch",
            "exchange http_status must equal the enclosing http_status",
            path=f"{path}.exchange_outcome.http_status",
        )
    if "response_sha256" in evidence and evidence["response_sha256"] != response_sha:
        _fail(
            "exchange_response_identity_mismatch",
            "exchange response identity must equal the enclosing response identity",
            path=f"{path}.exchange_outcome.response_sha256",
        )
    network_value = evidence.get("network_outcome")
    if network_value is None and "network_succeeded" in evidence:
        network_value = "succeeded" if evidence["network_succeeded"] is True else "failed"
    if final_outcome == "network_failure" and network_value not in {
        "failed",
        "network_failure",
    }:
        _fail(
            "exchange_network_outcome_mismatch",
            "network_failure final outcome requires failed network evidence",
            path=f"{path}.exchange_outcome",
        )
    if final_outcome == "succeeded":
        if network_value in {"failed", "network_failure"}:
            _fail(
                "exchange_network_outcome_mismatch",
                "successful final outcome cannot contain failed network evidence",
                path=f"{path}.exchange_outcome",
            )
        if (
            not isinstance(status, int)
            or isinstance(status, bool)
            or status not in {200, 204}
        ):
            _fail(
                "exchange_http_outcome_mismatch",
                "successful final outcome requires HTTP 200 or 204",
                path=f"{path}.http_status",
            )
        if response_sha is None:
            _fail(
                "exchange_response_identity_missing",
                "successful final outcome requires response identity evidence",
                path=path,
            )


def _url_value(value: object, *, path: str) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float) and math.isfinite(value):
        return str(value)
    _fail(
        "fdsn_parameter_not_canonicalizable",
        "FDSN parameter must be a finite scalar",
        path=path,
    )


def _fdsn_parameter_values(request: Mapping[str, object], *, path: str) -> dict[str, object]:
    bbox = _sequence(request.get("bbox"), path=f"{path}.bbox")
    if len(bbox) != 4:
        _fail(
            "fdsn_bbox_length_invalid",
            "FDSN bbox must contain four values",
            path=f"{path}.bbox",
        )
    return {
        "starttime": request.get("starttime_utc"),
        "endtime": request.get("endtime_utc"),
        "minlongitude": bbox[0],
        "minlatitude": bbox[1],
        "maxlongitude": bbox[2],
        "maxlatitude": bbox[3],
        "minmagnitude": request.get("minmagnitude"),
        "eventtype": request.get("eventtype"),
        "format": request.get("format"),
        "orderby": request.get("orderby"),
        "limit": request.get("limit"),
        "includeallorigins": request.get("includeallorigins"),
        "includeallmagnitudes": request.get("includeallmagnitudes"),
        "includedeleted": request.get("includedeleted"),
        "includesuperseded": request.get("includesuperseded"),
        "reviewstatus": request.get("reviewstatus"),
        "offset": request.get("offset"),
        "jsonerror": request.get("jsonerror"),
        "nodata": request.get("nodata"),
    }


def _canonical_fdsn_url(request: Mapping[str, object], *, path: str) -> str:
    endpoint = request.get("endpoint")
    if not isinstance(endpoint, str):
        _fail("fdsn_endpoint_invalid", "FDSN endpoint must be a string", path=f"{path}.endpoint")
    order = _sequence(
        request.get("canonical_parameter_order"),
        path=f"{path}.canonical_parameter_order",
    )
    values = _fdsn_parameter_values(request, path=path)
    pairs: list[str] = []
    for index, name_value in enumerate(order):
        if not isinstance(name_value, str) or name_value not in values:
            _fail(
                "fdsn_parameter_order_invalid",
                "canonical_parameter_order contains an unknown parameter",
                path=f"{path}.canonical_parameter_order[{index}]",
            )
        if name_value in {"includesuperseded", "reviewstatus"}:
            _fail(
                "fdsn_omitted_parameter_in_canonical_order",
                "interval queries must omit includesuperseded and reviewstatus from the URL",
                path=f"{path}.canonical_parameter_order[{index}]",
            )
        value = values[name_value]
        if value is None:
            _fail(
                "fdsn_parameter_missing",
                f"canonical FDSN parameter {name_value} is missing",
                path=path,
            )
        encoded_name = quote(name_value, safe="-._~")
        encoded_value = quote(
            _url_value(value, path=f"{path}.{name_value}"),
            safe="-._~",
        )
        pairs.append(f"{encoded_name}={encoded_value}")
    return f"{endpoint}?{'&'.join(pairs)}"


def _validate_fdsn_request(request: Mapping[str, object], *, path: str) -> str:
    canonical_url = _canonical_fdsn_url(request, path=path)
    calculated = hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()
    if request.get("canonical_url_utf8_sha256") != calculated:
        _fail(
            "fdsn_canonical_url_hash_mismatch",
            "canonical_url_utf8_sha256 must be recomputed from the frozen request",
            path=f"{path}.canonical_url_utf8_sha256",
        )
    return calculated


def _validate_count_preflight(
    count: Mapping[str, object],
    *,
    path: str,
    scheduled_at: datetime | None = None,
) -> None:
    request_value = count.get("request")
    if request_value is not None:
        _validate_fdsn_request(
            _mapping(request_value, path=f"{path}.request"),
            path=f"{path}.request",
        )
    started_value = count.get("fetch_started_at_utc")
    completed_value = count.get("fetch_completed_at_utc")
    if (started_value is None) != (completed_value is None):
        _fail(
            "count_preflight_partial_timeline",
            "count preflight start and completion must both be present or both null",
            path=path,
        )
    if started_value is not None:
        started = _utc_timestamp(
            started_value,
            path=f"{path}.fetch_started_at_utc",
        )
        completed = _utc_timestamp(
            completed_value,
            path=f"{path}.fetch_completed_at_utc",
        )
        if scheduled_at is not None and started < scheduled_at:
            _fail(
                "count_preflight_started_before_schedule",
                "count preflight cannot start before its frozen schedule",
                path=f"{path}.fetch_started_at_utc",
            )
        if completed < started:
            _fail(
                "count_preflight_completed_before_start",
                "count preflight completion cannot precede its start",
                path=f"{path}.fetch_completed_at_utc",
            )
    outcome = count.get("outcome")
    status = count.get("http_status")
    if outcome == "network_failure":
        for field in (
            "http_status",
            "response_content_type",
            "response_body_byte_count",
            "response_headers_sha256",
            "raw_response_sha256",
            "geojson_parse_verified",
            "parsed_count",
        ):
            if count.get(field) is not None:
                _fail(
                    "count_network_failure_contains_response_evidence",
                    "count network failure cannot contain response evidence",
                    path=f"{path}.{field}",
                )
    if status == 204 and count.get("raw_response_sha256") != _EMPTY_SHA256:
        _fail(
            "count_http_204_raw_response_not_empty",
            "HTTP 204 count response must hash the exact empty byte sequence",
            path=f"{path}.raw_response_sha256",
        )


def _validate_fdsn_fetch(fetch: Mapping[str, object], *, path: str) -> None:
    query_value = fetch.get("query_request")
    count_value = fetch.get("count_preflight")
    if query_value is None or count_value is None:
        return
    query = _mapping(query_value, path=f"{path}.query_request")
    count = _mapping(count_value, path=f"{path}.count_preflight")
    query_hash = _validate_fdsn_request(query, path=f"{path}.query_request")
    count_request = _mapping(count.get("request"), path=f"{path}.count_preflight.request")
    count_hash = _validate_fdsn_request(
        count_request,
        path=f"{path}.count_preflight.request",
    )
    scheduled_at: datetime | None = None
    if fetch.get("scheduled_at_utc") is not None:
        scheduled_at = _utc_timestamp(
            fetch.get("scheduled_at_utc"),
            path=f"{path}.scheduled_at_utc",
        )
    _validate_count_preflight(
        count,
        path=f"{path}.count_preflight",
        scheduled_at=scheduled_at,
    )

    if fetch.get("request_identity_sha256") != query_hash:
        _fail(
            "query_request_identity_mismatch",
            "request_identity_sha256 must equal the canonical query URL hash",
            path=f"{path}.request_identity_sha256",
        )
    if fetch.get("query_count_preflight_request_sha256") != count_hash:
        _fail(
            "count_request_identity_mismatch",
            "query_count_preflight_request_sha256 must equal the canonical count URL hash",
            path=f"{path}.query_count_preflight_request_sha256",
        )
    for field in (
        "starttime_utc",
        "endtime_utc",
        "bbox",
        "minmagnitude",
        "eventtype",
        "format",
        "includedeleted",
        "includesuperseded",
        "reviewstatus",
        "jsonerror",
        "nodata",
    ):
        if query.get(field) != count_request.get(field):
            _fail(
                "count_and_query_selection_mismatch",
                f"count and query requests must use identical {field}",
                path=f"{path}.count_preflight.request.{field}",
            )
    enclosing_start = fetch.get(
        "query_start_utc",
        fetch.get("target_start_exclusive_utc"),
    )
    enclosing_end = fetch.get(
        "query_end_utc",
        fetch.get("target_end_inclusive_utc"),
    )
    if query.get("starttime_utc") != enclosing_start:
        _fail(
            "query_request_start_mismatch",
            "query request starttime must equal enclosing query_start_utc",
            path=f"{path}.query_request.starttime_utc",
        )
    if query.get("endtime_utc") != enclosing_end:
        _fail(
            "query_request_end_mismatch",
            "query request endtime must equal enclosing query_end_utc",
            path=f"{path}.query_request.endtime_utc",
        )

    parsed_count = count.get("parsed_count")
    query_count = fetch.get("query_count")
    if parsed_count is not None and parsed_count != query_count:
        _fail(
            "count_preflight_query_count_mismatch",
            "count preflight parsed_count must equal enclosing query_count",
            path=f"{path}.query_count",
        )
    exchange_outcome = fetch.get("exchange_outcome")
    query_was_attempted = (
        exchange_outcome == "response_received"
        or (
            exchange_outcome is None
            and fetch.get("http_status") in {200, 204}
        )
    )
    if query_was_attempted:
        count_started = _utc_timestamp(
            count.get("fetch_started_at_utc"),
            path=f"{path}.count_preflight.fetch_started_at_utc",
        )
        count_completed = _utc_timestamp(
            count.get("fetch_completed_at_utc"),
            path=f"{path}.count_preflight.fetch_completed_at_utc",
        )
        query_started = _utc_timestamp(
            fetch.get("fetch_started_at_utc"),
            path=f"{path}.fetch_started_at_utc",
        )
        query_completed = _utc_timestamp(
            fetch.get("fetch_completed_at_utc"),
            path=f"{path}.fetch_completed_at_utc",
        )
        if count_completed < count_started:
            _fail(
                "count_preflight_completed_before_start",
                "count preflight completion cannot precede its start",
                path=f"{path}.count_preflight.fetch_completed_at_utc",
            )
        if scheduled_at is not None and count_started < scheduled_at:
            _fail(
                "count_preflight_started_before_schedule",
                "count preflight cannot start before its frozen schedule",
                path=f"{path}.count_preflight.fetch_started_at_utc",
            )
        if query_started < count_completed:
            _fail(
                "query_started_before_count_preflight_completed",
                "formal query cannot start before count preflight completes",
                path=f"{path}.fetch_started_at_utc",
            )
        if query_completed < query_started:
            _fail(
                "query_completed_before_start",
                "formal query completion cannot precede its start",
                path=f"{path}.fetch_completed_at_utc",
            )
    not_attempted = (
        exchange_outcome
        == "not_attempted_count_preflight_failed_or_limit_reached"
    )
    if not_attempted:
        null_fields = (
            "fetch_started_at_utc",
            "fetch_completed_at_utc",
            "http_status",
            "response_content_type",
            "response_body_byte_count",
            "geojson_parse_verified",
            "response_headers_sha256",
            "raw_response_sha256",
        )
        for field in null_fields:
            if fetch.get(field) is not None:
                _fail(
                    "not_attempted_query_contains_exchange_evidence",
                    "a query blocked by count preflight must retain null query/response evidence",
                    path=f"{path}.{field}",
                )
    if (
        isinstance(parsed_count, int)
        and not isinstance(parsed_count, bool)
        and parsed_count >= 20_000
        and exchange_outcome
        != "not_attempted_count_preflight_failed_or_limit_reached"
    ):
        _fail(
            "query_executed_after_count_limit_reached",
            "parsed_count >= 20000 must prevent the query exchange",
            path=f"{path}.exchange_outcome",
        )
    succeeded = fetch.get("outcome") == "succeeded" or (
        exchange_outcome is None
        and fetch.get("http_status") in {200, 204}
        and count.get("outcome") == "succeeded"
    )
    if succeeded and (
        parsed_count != query_count or query_count != fetch.get("feature_count")
    ):
        _fail(
            "successful_fdsn_counts_disagree",
            "successful FDSN fetch requires parsed_count == query_count == feature_count",
            path=path,
        )
    if count.get("outcome") != "succeeded" and fetch.get("exchange_outcome") != (
        "not_attempted_count_preflight_failed_or_limit_reached"
    ):
        _fail(
            "query_executed_after_failed_count_preflight",
            "failed count preflight must prevent the query exchange",
            path=f"{path}.exchange_outcome",
        )
    _validate_source_acquisition_provenance(fetch, path=path)


def _validate_geojson_parse_evidence(
    value: Mapping[str, object],
    *,
    path: str,
) -> None:
    if "geojson_parse_verified" not in value:
        return
    status = value.get("http_status")
    outcome = value.get("outcome")
    verified = value.get("geojson_parse_verified")
    if status == 204 and verified is not False:
        _fail(
            "http_204_marked_geojson_parse_verified",
            "HTTP 204 has no body and therefore cannot have verified GeoJSON parsing",
            path=f"{path}.geojson_parse_verified",
        )
    if status == 200 and outcome == "succeeded" and verified is not True:
        _fail(
            "http_200_success_without_geojson_parse_verification",
            "successful HTTP 200 catalogue evidence requires verified GeoJSON parsing",
            path=f"{path}.geojson_parse_verified",
        )


def _walk_fdsn_fetches(value: object, *, path: str = "$") -> None:
    if isinstance(value, Mapping):
        _validate_geojson_parse_evidence(value, path=path)
        if "query_request" in value and "count_preflight" in value:
            _validate_fdsn_fetch(value, path=path)
        for key, item in value.items():
            _walk_fdsn_fetches(item, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for index, item in enumerate(value):
            _walk_fdsn_fetches(item, path=f"{path}[{index}]")


def _issue_time_from_id(issue_id: object, *, path: str) -> datetime:
    if not isinstance(issue_id, str):
        _fail("issue_id_invalid", "issue_id must be a string", path=path)
    match = _ISSUE_ID_PATTERN.fullmatch(issue_id)
    if match is None:
        _fail(
            "issue_id_invalid",
            "issue_id must encode a Stage 2P Thursday local T",
            path=path,
        )
    try:
        local = datetime.strptime(match.group(1), "%Y%m%d").replace(
            tzinfo=timezone(_SHANGHAI_OFFSET)
        )
    except ValueError as exc:
        raise SemanticValidationError(
            "issue_id_invalid",
            "issue_id contains an invalid local calendar date",
            path=path,
        ) from exc
    if local.weekday() != _THURSDAY:
        _fail(
            "issue_id_not_thursday",
            "issue_id date must be a Thursday in Asia/Shanghai",
            path=path,
        )
    return local.astimezone(UTC)


def _bind_optional_timestamp(
    mapping: Mapping[str, object],
    field: str,
    expected: datetime,
    *,
    path: str,
    code: str,
) -> None:
    if field in mapping:
        _equal_timestamp(
            mapping[field],
            expected,
            code=code,
            path=f"{path}.{field}",
            message=f"{field} must match the enclosing issue/horizon timeline",
        )


def _validate_truth_attempt_timeline(
    attempt: Mapping[str, object],
    *,
    issue_id: object,
    issue_utc: datetime,
    horizon: int,
    target_end: datetime,
    maturity_due: datetime,
    expected_offset: int | None,
    path: str,
) -> None:
    if "issue_id" in attempt and attempt["issue_id"] != issue_id:
        _fail(
            "truth_attempt_issue_id_mismatch",
            "truth attempt issue_id must equal the enclosing record issue_id",
            path=f"{path}.issue_id",
        )
    _bind_optional_timestamp(
        attempt,
        "issue_time_utc",
        issue_utc,
        path=path,
        code="truth_attempt_issue_time_mismatch",
    )
    if "horizon_days" in attempt and attempt["horizon_days"] != horizon:
        _fail(
            "truth_attempt_horizon_mismatch",
            "truth attempt horizon_days must equal the enclosing horizon",
            path=f"{path}.horizon_days",
        )
    _bind_optional_timestamp(
        attempt,
        "target_start_exclusive_utc",
        issue_utc,
        path=path,
        code="truth_attempt_target_start_mismatch",
    )
    _bind_optional_timestamp(
        attempt,
        "target_end_inclusive_utc",
        target_end,
        path=path,
        code="truth_attempt_target_end_mismatch",
    )
    _bind_optional_timestamp(
        attempt,
        "maturity_due_at_utc",
        maturity_due,
        path=path,
        code="truth_attempt_maturity_due_mismatch",
    )

    offset_value = attempt.get("retry_offset_hours")
    if offset_value is not None:
        offset = _integer(offset_value, path=f"{path}.retry_offset_hours")
        if expected_offset is not None and offset != expected_offset:
            _fail(
                "truth_retry_offset_order_mismatch",
                "truth retry offsets must follow the frozen 0/6/24/72/168 hour order",
                path=f"{path}.retry_offset_hours",
            )
        scheduled_expected = maturity_due + timedelta(hours=offset)
        _bind_optional_timestamp(
            attempt,
            "scheduled_at_utc",
            scheduled_expected,
            path=path,
            code="truth_attempt_scheduled_time_mismatch",
        )
    elif expected_offset is not None:
        _fail(
            "truth_retry_offset_missing",
            "mature truth attempt must declare its retry_offset_hours",
            path=f"{path}.retry_offset_hours",
        )

    scheduled_value = attempt.get("scheduled_at_utc")
    started_value = attempt.get("fetch_started_at_utc")
    completed_value = attempt.get("fetch_completed_at_utc")
    count_value = attempt.get("count_preflight")
    if scheduled_value is not None and isinstance(count_value, Mapping):
        count_started_value = count_value.get("fetch_started_at_utc")
        if count_started_value is not None:
            _at_or_after(
                count_started_value,
                _utc_timestamp(
                    scheduled_value,
                    path=f"{path}.scheduled_at_utc",
                ),
                code="truth_count_started_before_schedule",
                path=f"{path}.count_preflight.fetch_started_at_utc",
                message="truth count preflight must start at or after its frozen schedule",
            )
    if started_value is not None and scheduled_value is not None:
        scheduled = _utc_timestamp(scheduled_value, path=f"{path}.scheduled_at_utc")
        started = _at_or_after(
            started_value,
            scheduled,
            code="truth_fetch_started_before_schedule",
            path=f"{path}.fetch_started_at_utc",
            message="truth fetch must start at or after its frozen schedule",
        )
        if completed_value is not None:
            completed = _utc_timestamp(
                completed_value,
                path=f"{path}.fetch_completed_at_utc",
            )
            if completed < started:
                _fail(
                    "truth_fetch_completed_before_start",
                    "truth fetch completion cannot precede its start",
                    path=f"{path}.fetch_completed_at_utc",
                )
    elif completed_value is not None:
        _fail(
            "truth_fetch_completion_without_start",
            "truth fetch completion requires a fetch start",
            path=f"{path}.fetch_completed_at_utc",
        )


def _validate_truth_snapshot_timeline(
    snapshot: Mapping[str, object],
    *,
    issue_id: object,
    issue_utc: datetime,
    horizon: int,
    target_end: datetime,
    maturity_due: datetime,
    selected_offset: int | None,
    path: str,
) -> None:
    if "issue_id" in snapshot and snapshot["issue_id"] != issue_id:
        _fail(
            "truth_snapshot_issue_id_mismatch",
            "TruthSnapshot issue_id must equal the enclosing record issue_id",
            path=f"{path}.issue_id",
        )
    _bind_optional_timestamp(
        snapshot,
        "issue_time_utc",
        issue_utc,
        path=path,
        code="truth_snapshot_issue_time_mismatch",
    )
    if "horizon_days" in snapshot and snapshot["horizon_days"] != horizon:
        _fail(
            "truth_snapshot_horizon_mismatch",
            "TruthSnapshot horizon_days must equal the enclosing horizon",
            path=f"{path}.horizon_days",
        )
    _bind_optional_timestamp(
        snapshot,
        "target_start_exclusive_utc",
        issue_utc,
        path=path,
        code="truth_snapshot_target_start_mismatch",
    )
    _bind_optional_timestamp(
        snapshot,
        "target_end_inclusive_utc",
        target_end,
        path=path,
        code="truth_snapshot_target_end_mismatch",
    )
    _bind_optional_timestamp(
        snapshot,
        "maturity_due_at_utc",
        maturity_due,
        path=path,
        code="truth_snapshot_maturity_due_mismatch",
    )
    if (
        selected_offset is not None
        and snapshot.get("retry_offset_hours") != selected_offset
    ):
        _fail(
            "truth_snapshot_retry_offset_mismatch",
            "TruthSnapshot retry offset must equal the selected successful attempt",
            path=f"{path}.retry_offset_hours",
        )
    acquisition_value = snapshot.get("acquisition")
    if acquisition_value is not None:
        acquisition = _mapping(acquisition_value, path=f"{path}.acquisition")
        _bind_optional_timestamp(
            acquisition,
            "query_start_utc",
            issue_utc,
            path=f"{path}.acquisition",
            code="truth_acquisition_target_start_mismatch",
        )
        _bind_optional_timestamp(
            acquisition,
            "query_end_utc",
            target_end,
            path=f"{path}.acquisition",
            code="truth_acquisition_target_end_mismatch",
        )


def _validate_revision_reasons(
    value: object,
    *,
    path: str,
    allow_empty: bool,
) -> tuple[str, ...]:
    sequence = _sequence(value, path=path)
    reasons: list[str] = []
    for index, reason in enumerate(sequence):
        if not isinstance(reason, str) or reason not in _REVISION_REASON_ORDER:
            _fail(
                "revision_reason_invalid",
                "revision reason must use a frozen formal-change category",
                path=f"{path}[{index}]",
            )
        reasons.append(reason)
    if not allow_empty and not reasons:
        _fail(
            "revision_reasons_empty",
            "changed truth revision requires at least one reason",
            path=path,
        )
    if len(set(reasons)) != len(reasons):
        _fail(
            "revision_reasons_not_unique",
            "revision reasons must be unique",
            path=path,
        )
    order = {reason: index for index, reason in enumerate(_REVISION_REASON_ORDER)}
    if reasons != sorted(reasons, key=order.__getitem__):
        _fail(
            "revision_reasons_not_frozen_order",
            "revision reasons must follow the frozen category order",
            path=path,
        )
    return tuple(reasons)


def _validate_attempt_acquisition_exchange_binding(
    attempt: Mapping[str, object],
    acquisition: Mapping[str, object],
    *,
    path: str,
    subject: str,
) -> None:
    exact_fields = (
        "query_request",
        "count_preflight",
        "request_identity_sha256",
        "query_count_preflight_request_sha256",
        "query_count",
        "feature_count",
        "fetch_started_at_utc",
        "fetch_completed_at_utc",
        "http_status",
        "response_content_type",
        "response_body_byte_count",
        "geojson_parse_verified",
        "response_headers_sha256",
    )
    for field, fallback in (
        ("query_start_utc", "target_start_exclusive_utc"),
        ("query_end_utc", "target_end_inclusive_utc"),
    ):
        expected = (
            attempt.get(field)
            if field in attempt
            else attempt.get(fallback)
        )
        if acquisition.get(field) != expected:
            _fail(
                f"{subject}_attempt_acquisition_mismatch",
                f"selected attempt and acquisition must contain identical {field}",
                path=f"{path}.{field}",
            )
    for field in exact_fields:
        if acquisition.get(field) != attempt.get(field):
            _fail(
                f"{subject}_attempt_acquisition_mismatch",
                f"selected attempt and acquisition must contain identical {field}",
                path=f"{path}.{field}",
            )
    raw_response_value = acquisition.get("raw_response")
    if isinstance(raw_response_value, Mapping):
        acquisition_raw_sha256 = raw_response_value.get("file_sha256")
        raw_path = f"{path}.raw_response.file_sha256"
    else:
        acquisition_raw_sha256 = acquisition.get("raw_response_sha256")
        raw_path = f"{path}.raw_response_sha256"
    if acquisition_raw_sha256 != attempt.get("raw_response_sha256"):
        _fail(
            f"{subject}_attempt_acquisition_raw_response_mismatch",
            "selected attempt raw response must equal the acquisition artifact identity",
            path=raw_path,
        )


def _validate_truth_record_timeline(record: Mapping[str, object]) -> None:
    record_type = record.get("record_type")
    if record_type not in {"MatureTruthSnapshotRecord", "TruthRevisionRecord"}:
        return
    issue_id = record.get("issue_id")
    issue_utc = _issue_time_from_id(issue_id, path="$.issue_id")
    _bind_optional_timestamp(
        record,
        "issue_time_utc",
        issue_utc,
        path="$",
        code="truth_record_issue_time_mismatch",
    )
    horizon = _integer(record.get("horizon_days"), path="$.horizon_days")
    if horizon not in {7, 30, 90}:
        _fail(
            "truth_record_horizon_invalid",
            "truth horizon must be 7, 30, or 90 days",
            path="$.horizon_days",
        )
    target_end = issue_utc + timedelta(days=horizon)
    maturity_due = target_end + timedelta(days=30)
    _bind_optional_timestamp(
        record,
        "target_start_exclusive_utc",
        issue_utc,
        path="$",
        code="truth_record_target_start_mismatch",
    )
    _bind_optional_timestamp(
        record,
        "target_end_inclusive_utc",
        target_end,
        path="$",
        code="truth_record_target_end_mismatch",
    )
    _bind_optional_timestamp(
        record,
        "maturity_due_at_utc",
        maturity_due,
        path="$",
        code="truth_record_maturity_due_mismatch",
    )

    if record_type == "MatureTruthSnapshotRecord":
        attempts = _sequence(record.get("attempt_evidence"), path="$.attempt_evidence")
        selected_offset: int | None = None
        selected_attempt: Mapping[str, object] | None = None
        selected_seen = False
        previous_completed: datetime | None = None
        for index, attempt_value in enumerate(attempts):
            attempt = _mapping(attempt_value, path=f"$.attempt_evidence[{index}]")
            if selected_seen:
                _fail(
                    "truth_attempt_after_selected_success",
                    "mature truth acquisition must stop after its selected success",
                    path=f"$.attempt_evidence[{index}]",
                )
            if attempt.get("attempt_index") != index:
                _fail(
                    "truth_attempt_index_not_contiguous",
                    "mature truth attempt_index must equal its ordered position",
                    path=f"$.attempt_evidence[{index}].attempt_index",
                )
            expected_offset = (
                _TRUTH_RETRY_OFFSETS_HOURS[index]
                if index < len(_TRUTH_RETRY_OFFSETS_HOURS)
                else None
            )
            _validate_truth_attempt_timeline(
                attempt,
                issue_id=issue_id,
                issue_utc=issue_utc,
                horizon=horizon,
                target_end=target_end,
                maturity_due=maturity_due,
                expected_offset=expected_offset,
                path=f"$.attempt_evidence[{index}]",
            )
            if index > 0:
                if previous_completed is None:
                    _fail(
                        "truth_retry_previous_completion_missing",
                        "a subsequent mature-truth retry requires the prior completion time",
                        path=f"$.attempt_evidence[{index - 1}].fetch_completed_at_utc",
                    )
                current_count = _mapping(
                    attempt.get("count_preflight"),
                    path=f"$.attempt_evidence[{index}].count_preflight",
                )
                current_started = _utc_timestamp(
                    current_count.get("fetch_started_at_utc"),
                    path=(
                        f"$.attempt_evidence[{index}].count_preflight."
                        "fetch_started_at_utc"
                    ),
                )
                if current_started < previous_completed:
                    _fail(
                        "truth_retry_started_before_previous_completed",
                        "the next retry count preflight may start only after the prior "
                        "query completes",
                        path=(
                            f"$.attempt_evidence[{index}].count_preflight."
                            "fetch_started_at_utc"
                        ),
                    )
            completed_value = attempt.get("fetch_completed_at_utc")
            previous_completed = (
                None
                if completed_value is None
                else _utc_timestamp(
                    completed_value,
                    path=f"$.attempt_evidence[{index}].fetch_completed_at_utc",
                )
            )
            if attempt.get("selected_as_truth_snapshot") is True:
                selected_offset = _integer(
                    attempt.get("retry_offset_hours"),
                    path=f"$.attempt_evidence[{index}].retry_offset_hours",
                )
                selected_attempt = attempt
                selected_seen = True
        snapshot_value = record.get("truth_snapshot")
        if snapshot_value is not None:
            snapshot = _mapping(snapshot_value, path="$.truth_snapshot")
            _validate_truth_snapshot_timeline(
                snapshot,
                issue_id=issue_id,
                issue_utc=issue_utc,
                horizon=horizon,
                target_end=target_end,
                maturity_due=maturity_due,
                selected_offset=selected_offset,
                path="$.truth_snapshot",
            )
            if selected_attempt is None:
                _fail(
                    "truth_snapshot_without_selected_attempt",
                    "a sealed mature truth snapshot requires one selected successful attempt",
                    path="$.truth_snapshot",
                )
            acquisition = _mapping(
                snapshot.get("acquisition"),
                path="$.truth_snapshot.acquisition",
            )
            _validate_attempt_acquisition_exchange_binding(
                selected_attempt,
                acquisition,
                path="$.truth_snapshot.acquisition",
                subject="mature_truth",
            )
            selected_completed = _utc_timestamp(
                selected_attempt.get("fetch_completed_at_utc"),
                path="$.attempt_evidence[selected].fetch_completed_at_utc",
            )
            snapshot_sealed = _utc_timestamp(
                snapshot.get("seal_completed_at_utc"),
                path="$.truth_snapshot.seal_completed_at_utc",
            )
            if snapshot_sealed < selected_completed:
                _fail(
                    "truth_snapshot_sealed_before_query_completed",
                    "mature truth snapshot cannot seal before its selected query completes",
                    path="$.truth_snapshot.seal_completed_at_utc",
                )
    else:
        _validate_revision_reasons(
            record.get("revision_reasons"),
            path="$.revision_reasons",
            allow_empty=False,
        )
        observed = _utc_timestamp(
            record.get("revision_observed_at_utc"),
            path="$.revision_observed_at_utc",
        )
        derived_started = _at_or_after(
            record.get("revision_derived_started_at_utc"),
            observed,
            code="truth_revision_derived_before_observed",
            path="$.revision_derived_started_at_utc",
            message="local revision derivation cannot start before its source observation",
        )
        derived_completed = _at_or_after(
            record.get("revision_derived_completed_at_utc"),
            derived_started,
            code="truth_revision_derived_completed_before_start",
            path="$.revision_derived_completed_at_utc",
            message="local revision derivation completion cannot precede its start",
        )
        seal_completed = _at_or_after(
            record.get("seal_completed_at_utc"),
            derived_completed,
            code="truth_revision_sealed_before_derivation_completed",
            path="$.seal_completed_at_utc",
            message="truth revision cannot seal before local derivation completes",
        )
        _at_or_after(
            record.get("record_core_frozen_at_utc"),
            seal_completed,
            code="truth_revision_core_frozen_before_seal",
            path="$.record_core_frozen_at_utc",
            message="truth revision core cannot freeze before its seal",
        )


def _walk_exchange_outcomes(value: object, *, path: str = "$") -> None:
    if isinstance(value, Mapping):
        _validate_exchange_outcome(value, path=path)
        for key, item in value.items():
            _walk_exchange_outcomes(item, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for index, item in enumerate(value):
            _walk_exchange_outcomes(item, path=f"{path}[{index}]")


def validate_record_semantics(
    record: Mapping[str, object],
    protocol: Mapping[str, object],
) -> None:
    """Validate all Stage 2P cross-field semantics applicable to one record.

    Success returns ``None``.  The first failure raises
    :class:`SemanticValidationError` with a stable ``code`` and ``path``.
    """

    record = _mapping(record, path="$")
    protocol = _mapping(protocol, path="protocol")
    record_type = record.get("record_type")
    _walk_float64_bits_hex(record)

    if record_type == "IssueInputSnapshotRecord" or "issue_time_local" in record:
        _, issue_utc, query_end = _validate_issue_clock(record, protocol)
        if record.get("status") == "on_time":
            _validate_on_time_issue_causality(
                record,
                issue_utc=issue_utc,
                query_end=query_end,
            )

    seal_value = record.get("prediction_seal")
    if seal_value is not None:
        _validate_prediction_seal(
            _mapping(seal_value, path="$.prediction_seal"),
            path="$.prediction_seal",
        )
    elif all(name in record for name in ("P0", "P1", "PP", "grid_identity_sha256")):
        _validate_prediction_seal(record, path="$")

    selected_exposure = _validate_selected_exposure_manifest(record)
    _validate_alarm_area_manifest(record, selected_exposure)
    _validate_evaluation_policy_hashes(record, protocol)

    horizons_value = record.get("horizon_evaluability")
    if horizons_value is not None:
        horizons = _sequence(horizons_value, path="$.horizon_evaluability")
        for index, horizon in enumerate(horizons):
            _validate_horizon_evaluability(
                _mapping(horizon, path=f"$.horizon_evaluability[{index}]"),
                path=f"$.horizon_evaluability[{index}]",
            )
    elif "horizon_days" in record and "evaluable" in record:
        _validate_horizon_evaluability(record, path="$")

    if record_type == "TargetCohortDefinition":
        _validate_target_cohort(record)
    _validate_formal_freeze_record_binding(record, protocol)
    _validate_code_bindings(record, protocol)
    _validate_truth_record_timeline(record)
    _validate_evaluation_result_binding(record)
    _validate_evaluation_sample_gate(record)
    _validate_confirmatory_result_semantics(record)
    _validate_confirmatory_result_timeline(record)
    _validate_rfc3161(record)
    _walk_fdsn_fetches(record)
    _walk_exchange_outcomes(record)
    _validate_embedded_object_identities(record)
    _validate_record_hashes(record)


def _checkpoint_kind(record: Mapping[str, object], *, path: str) -> str | None:
    signals: set[str] = set()
    if record.get("trigger_reason") == "on_time_checkpoint_52":
        signals.add("52")
    if record.get("trigger_reason") == "on_time_checkpoint_104":
        signals.add("104")
    if record.get("trigger_reason") == "scheduled_issue_cap_130":
        signals.add("130")
    if record.get("trigger_on_time_issue_count") == 52:
        signals.add("52")
    if record.get("trigger_on_time_issue_count") == 104:
        signals.add("104")
    if record.get("checkpoint_number") == 1:
        signals.add("52")
    if record.get("checkpoint_number") == 2:
        signals.add("104")
    if record.get("checkpoint_number") == 3:
        signals.add("130")
    if len(signals) > 1:
        _fail(
            "checkpoint_identity_inconsistent",
            "checkpoint number, trigger, and on-time count disagree",
            path=path,
        )
    return next(iter(signals), None)


def validate_evaluation_chain(records: Sequence[Mapping[str, object]]) -> None:
    """Validate the append-only 52/104 evaluation-freeze state machine."""

    records = _sequence(records, path="$")
    previous: Mapping[str, object] | None = None
    input_by_checkpoint: dict[str, Mapping[str, object]] = {}
    result_count = 0
    result_seen = False

    for index, value in enumerate(records):
        path = f"$[{index}]"
        record = _mapping(value, path=path)
        sequence = _integer(record.get("evaluation_sequence"), path=f"{path}.evaluation_sequence")
        if sequence != index + 1:
            _fail(
                "evaluation_sequence_not_contiguous",
                "evaluation_sequence must be contiguous from 1",
                path=f"{path}.evaluation_sequence",
            )
        expected_previous = None if previous is None else previous.get("content_sha256")
        if record.get("previous_evaluation_freeze_sha256") != expected_previous:
            _fail(
                "previous_evaluation_hash_mismatch",
                "previous_evaluation_freeze_sha256 must link to the prior record",
                path=f"{path}.previous_evaluation_freeze_sha256",
            )

        phase = record.get("phase")
        if phase not in {"input_freeze", "result_seal"}:
            _fail(
                "invalid_evaluation_phase",
                "phase must be input_freeze or result_seal",
                path=f"{path}.phase",
            )
        checkpoint = _checkpoint_kind(record, path=path)

        if phase == "result_seal":
            result_count += 1
            if result_count > 1:
                _fail(
                    "second_result_seal_forbidden",
                    "the complete evaluation chain may contain only one result_seal",
                    path=path,
                )
            if previous is None or previous.get("phase") != "input_freeze":
                _fail(
                    "result_seal_not_immediately_after_input_freeze",
                    "result_seal must immediately follow its input_freeze",
                    path=path,
                )
            if previous.get("sample_gate_met") is not True:
                _fail(
                    "result_seal_without_passed_sample_gate",
                    "result_seal requires the immediately preceding sample gate to pass",
                    path=path,
                )
            previous_hash = previous.get("content_sha256")
            if record.get("input_freeze_sha256") != previous_hash:
                _fail(
                    "result_seal_input_freeze_link_mismatch",
                    "result_seal input_freeze_sha256 must equal the preceding input hash",
                    path=f"{path}.input_freeze_sha256",
                )
            previous_checkpoint = _checkpoint_kind(previous, path=f"$[{index - 1}]")
            if checkpoint != previous_checkpoint:
                _fail(
                    "result_seal_checkpoint_mismatch",
                    "result_seal must close the same checkpoint as its input_freeze",
                    path=path,
                )
            _validate_result_freeze_identity(previous, record, path=path)
            _validate_evaluation_frozen_fields(previous, record, path=path)
            _validate_evaluation_result_binding(record)
            if "frozen_at_utc" in previous or "frozen_at_utc" in record:
                input_frozen = _utc_timestamp(
                    previous.get("frozen_at_utc"),
                    path=f"$[{index - 1}].frozen_at_utc",
                )
                result_frozen = _utc_timestamp(
                    record.get("frozen_at_utc"),
                    path=f"{path}.frozen_at_utc",
                )
                if result_frozen <= input_frozen:
                    _fail(
                        "result_seal_not_after_input_freeze",
                        "result_seal frozen_at_utc must be later than input_freeze frozen_at_utc",
                        path=f"{path}.frozen_at_utc",
                    )
            if record.get("effect_rows_opened_at_utc") is not None:
                _validate_evaluation_effect_open_order(
                    previous,
                    record,
                    input_path=f"$[{index - 1}]",
                    result_path=path,
                )
            result_seen = True
        else:
            if result_seen:
                _fail(
                    "evaluation_record_after_result_seal",
                    "result_seal is terminal for the evaluation chain",
                    path=path,
                )
            if checkpoint is None:
                _fail(
                    "evaluation_checkpoint_unknown",
                    "input_freeze must identify checkpoint 52, 104, or scheduled cap 130",
                    path=path,
                )
            if checkpoint in input_by_checkpoint:
                _fail(
                    "duplicate_evaluation_input_freeze",
                    "each checkpoint may have only one input_freeze",
                    path=path,
                )
            scheduled_count = _integer(
                record.get("scheduled_issue_count"),
                path=f"{path}.scheduled_issue_count",
            )
            on_time_count = _integer(
                record.get("trigger_on_time_issue_count"),
                path=f"{path}.trigger_on_time_issue_count",
            )
            if checkpoint == "52" and scheduled_count > 129:
                _fail(
                    "checkpoint_52_reached_at_scheduled_cap",
                    "checkpoint 52 must occur before scheduled issue cap 130",
                    path=f"{path}.scheduled_issue_count",
                )
            if checkpoint == "104":
                first = input_by_checkpoint.get("52")
                if first is None or first.get("sample_gate_met") is not False:
                    _fail(
                        "checkpoint_104_without_failed_52_gate",
                        "checkpoint 104 is allowed only after an explicit failed 52 gate",
                        path=path,
                    )
            if checkpoint == "130":
                if scheduled_count != 130 or on_time_count >= 104:
                    _fail(
                        "scheduled_cap_counts_invalid",
                        "scheduled cap requires scheduled=130 and on_time<104",
                        path=path,
                    )
                second = input_by_checkpoint.get("104")
                if second is not None and second.get("sample_gate_met") is not False:
                    _fail(
                        "scheduled_cap_without_failed_104_gate",
                        "scheduled cap cannot follow a passed 104 gate",
                        path=path,
                    )
                if (
                    record.get("sample_gate_met") is not False
                    or record.get("status") != "evidence_insufficient"
                    or record.get("confirmatory_effects_authorized") is not False
                ):
                    _fail(
                        "scheduled_cap_not_fail_closed",
                        "scheduled cap must be an evidence_insufficient terminal stop",
                        path=path,
                    )
            input_by_checkpoint[checkpoint] = record

        if (
            previous is not None
            and previous.get("phase") == "input_freeze"
            and previous.get("sample_gate_met") is True
            and phase != "result_seal"
        ):
            _fail(
                "passed_input_freeze_not_immediately_closed",
                "a passed input_freeze may only be followed by its result_seal",
                path=path,
            )
        previous = record


def _expected_issue_id(local: datetime) -> str:
    return f"stage2p-issue-{local:%Y%m%dT000000}+0800"


def _next_rule_thursday_after(value: datetime) -> datetime:
    local = value.astimezone(timezone(_SHANGHAI_OFFSET))
    midnight = local.replace(hour=0, minute=0, second=0, microsecond=0)
    days = (_THURSDAY - midnight.weekday()) % 7
    candidate = midnight + timedelta(days=days)
    if candidate <= local:
        candidate += timedelta(days=7)
    return candidate.astimezone(UTC)


def _expected_first_issue_from_cohort(
    cohort: Mapping[str, object],
    protocol: Mapping[str, object],
) -> datetime | None:
    receipts = _extract_tag_receipts(cohort)
    if receipts is None:
        return None
    valid_from = _utc_timestamp(cohort.get("valid_from_utc"), path="cohort.valid_from_utc")
    verified = max(
        _receipt_verified_at(receipts[0], path="cohort.protocol_tag_receipt"),
        _receipt_verified_at(receipts[1], path="cohort.code_tag_receipt"),
    )
    threshold_value = cohort.get(
        "first_issue_not_before_utc",
        _first_issue_not_before(_calendar(protocol)).isoformat().replace(
            "+00:00",
            "Z",
        ),
    )
    threshold = _utc_timestamp(
        threshold_value,
        path="cohort.first_issue_not_before_utc",
    )
    mechanically_derived = max(threshold, _next_rule_thursday_after(verified))
    if valid_from != mechanically_derived:
        _fail(
            "target_valid_from_not_mechanically_derived",
            "cohort valid_from_utc must equal the mechanically derived first issue T",
            path="cohort.valid_from_utc",
        )
    return valid_from


def validate_issue_chain(
    records: Sequence[Mapping[str, object]],
    protocol: Mapping[str, object] | None = None,
    *,
    cohort_definition: Mapping[str, object] | None = None,
) -> None:
    """Validate weekly issue continuity, hashes, IDs, and first-issue activation."""

    records = _sequence(records, path="$")
    protocol = _DEFAULT_CALENDAR if protocol is None else _mapping(protocol, path="protocol")
    previous: Mapping[str, object] | None = None
    previous_issue_utc: datetime | None = None
    on_time_count = 0

    for index, value in enumerate(records):
        path = f"$[{index}]"
        record = _mapping(value, path=path)
        validate_record_semantics(record, protocol)
        local = _timestamp(record.get("issue_time_local"), path=f"{path}.issue_time_local")
        issue_utc = _utc_timestamp(record.get("issue_time_utc"), path=f"{path}.issue_time_utc")
        if record.get("issue_id") != _expected_issue_id(local):
            _fail(
                "issue_id_not_derived_from_local_time",
                "issue_id must be derived exactly from local T",
                path=f"{path}.issue_id",
            )
        scheduled = _integer(
            record.get("scheduled_issue_sequence"),
            path=f"{path}.scheduled_issue_sequence",
        )
        if scheduled != index + 1:
            _fail(
                "scheduled_issue_sequence_not_contiguous",
                "scheduled_issue_sequence must be contiguous from 1",
                path=f"{path}.scheduled_issue_sequence",
            )
        if record.get("status") == "on_time":
            on_time_count += 1
            if record.get("on_time_issue_sequence") != on_time_count:
                _fail(
                    "on_time_issue_sequence_not_contiguous",
                    "on_time_issue_sequence must count only on-time records from 1",
                    path=f"{path}.on_time_issue_sequence",
                )
        elif record.get("status") == "missed_issue":
            if record.get("on_time_issue_sequence") is not None:
                _fail(
                    "missed_issue_has_on_time_sequence",
                    "missed_issue must have null on_time_issue_sequence",
                    path=f"{path}.on_time_issue_sequence",
                )
        expected_previous = None if previous is None else previous.get("content_sha256")
        if record.get("previous_issue_record_sha256") != expected_previous:
            _fail(
                "previous_issue_hash_mismatch",
                "previous_issue_record_sha256 must link to the prior issue record",
                path=f"{path}.previous_issue_record_sha256",
            )
        if previous_issue_utc is not None and issue_utc != previous_issue_utc + timedelta(days=7):
            _fail(
                "issue_weekly_cadence_broken",
                "each issue T must equal the preceding T plus exactly seven days",
                path=f"{path}.issue_time_utc",
            )
        previous = record
        previous_issue_utc = issue_utc

    if records and cohort_definition is not None:
        cohort = _mapping(cohort_definition, path="cohort")
        _validate_target_cohort(cohort)
        expected_first = _expected_first_issue_from_cohort(cohort, protocol)
        if expected_first is not None:
            first = _utc_timestamp(records[0].get("issue_time_utc"), path="$[0].issue_time_utc")
            if first != expected_first:
                _fail(
                    "first_issue_not_first_rule_after_activation",
                    "first issue must be the first rule Thursday after both tag receipts "
                    "and valid_from",
                    path="$[0].issue_time_utc",
                )


def _truth_target_set(
    record: Mapping[str, object],
    *,
    path: str,
) -> Mapping[str, object] | None:
    if record.get("record_type") == "MatureTruthSnapshotRecord":
        snapshot = record.get("truth_snapshot")
        if snapshot is None:
            return None
        snapshot_mapping = _mapping(snapshot, path=f"{path}.truth_snapshot")
        target = snapshot_mapping.get("realized_target_set")
        return (
            None
            if target is None
            else _mapping(target, path=f"{path}.truth_snapshot.realized_target_set")
        )
    target = record.get("revised_target_set")
    return (
        None
        if target is None
        else _mapping(target, path=f"{path}.revised_target_set")
    )


def _validate_formal_window_binding_against_truth_chain(
    records: Sequence[object],
    binding_row: Mapping[str, object],
    manifest: Mapping[str, object],
    *,
    issue_prediction_seal: Mapping[str, object] | None,
    protocol: Mapping[str, object] | None = None,
) -> None:
    _validate_canonical_self_hash(
        binding_row,
        hash_field="binding_row_sha256",
        subject="formal_window_target_binding_row",
        path="formal_window_target_binding",
    )
    first = _mapping(records[0], path="$[0]")
    final_index = len(records) - 1
    final = _mapping(records[final_index], path=f"$[{final_index}]")
    if (
        binding_row.get("issue_id") != first.get("issue_id")
        or binding_row.get("horizon_days") != first.get("horizon_days")
    ):
        _fail(
            "formal_window_binding_scope_mismatch",
            "formal binding row must identify the same issue and horizon as its truth chain",
            path="formal_window_target_binding",
        )
    if (
        issue_prediction_seal is not None
        and binding_row.get("issue_prediction_seal_sha256")
        != issue_prediction_seal.get("prediction_seal_sha256")
    ):
        _fail(
            "formal_window_binding_prediction_seal_mismatch",
            "formal binding row must bind the supplied issue prediction seal",
            path="formal_window_target_binding.issue_prediction_seal_sha256",
        )
    if binding_row.get("source_snapshot_sha256") != manifest.get(
        "source_snapshot_sha256"
    ):
        _fail(
            "formal_window_binding_source_snapshot_mismatch",
            "formal binding row must use the current formal manifest source snapshot",
            path="formal_window_target_binding.source_snapshot_sha256",
        )
    issue_time = _issue_time_from_id(
        first.get("issue_id"),
        path="$[0].issue_id",
    )
    horizon = _integer(first.get("horizon_days"), path="$[0].horizon_days")
    _equal_timestamp(
        binding_row.get("target_start_exclusive_utc"),
        issue_time,
        code="formal_window_binding_target_start_mismatch",
        path="formal_window_target_binding.target_start_exclusive_utc",
        message="formal binding target start must equal issue T",
    )
    _equal_timestamp(
        binding_row.get("target_end_inclusive_utc"),
        issue_time + timedelta(days=horizon),
        code="formal_window_binding_target_end_mismatch",
        path="formal_window_target_binding.target_end_inclusive_utc",
        message="formal binding target end must equal T plus horizon",
    )

    status = binding_row.get("change_status")
    if status == "truth_snapshot_unavailable_remains_unavailable":
        if (
            len(records) != 1
            or first.get("status") != "truth_snapshot_unavailable"
            or binding_row.get("previous_truth_record_sha256")
            != first.get("content_sha256")
            or binding_row.get("truth_available") is not False
        ):
            _fail(
                "formal_window_binding_unavailable_chain_mismatch",
                "unavailable binding must retain the one terminal unavailable truth record",
                path="formal_window_target_binding",
            )
        return

    final_target = _truth_target_set(final, path=f"$[{final_index}]")
    formal_target = _mapping(
        binding_row.get("formal_target_set"),
        path="formal_window_target_binding.formal_target_set",
    )
    if final_target is None or dict(formal_target) != dict(final_target):
        _fail(
            "formal_window_binding_current_target_mismatch",
            "formal target set must exactly equal the selected chain-tip target",
            path="formal_window_target_binding.formal_target_set",
        )
    if status == "unchanged_reuses_previous_truth_record":
        if binding_row.get("previous_truth_record_sha256") != final.get(
            "content_sha256"
        ):
            _fail(
                "unchanged_binding_previous_truth_mismatch",
                "unchanged binding must reuse the current chain-tip truth record",
                path="formal_window_target_binding.previous_truth_record_sha256",
            )
        if binding_row.get("previous_target_set_sha256") != final_target.get(
            "target_rows_content_sha256"
        ):
            _fail(
                "unchanged_binding_previous_target_mismatch",
                "unchanged binding must prove current and previous target hashes are equal",
                path="formal_window_target_binding.previous_target_set_sha256",
            )
        for field in ("added", "removed", "modified"):
            change_set = _mapping(
                binding_row.get(field),
                path=f"formal_window_target_binding.{field}",
            )
            if change_set.get("row_count") != 0:
                _fail(
                    "unchanged_binding_contains_change_rows",
                    "unchanged formal binding change sets must all be empty",
                    path=f"formal_window_target_binding.{field}.row_count",
                )
        return
    if status != "changed_requires_truth_revision":
        _fail(
            "formal_window_binding_change_status_invalid",
            "formal binding change_status is not recognized",
            path="formal_window_target_binding.change_status",
        )
    if len(records) < 2 or final.get("record_type") != "TruthRevisionRecord":
        _fail(
            "changed_binding_without_new_truth_revision",
            "changed formal binding requires a new chain-tip TruthRevisionRecord",
            path=f"$[{final_index}]",
        )
    prior = _mapping(records[-2], path=f"$[{final_index - 1}]")
    prior_target = _truth_target_set(prior, path=f"$[{final_index - 1}]")
    if prior_target is None:
        _fail(
            "changed_binding_prior_target_missing",
            "changed formal binding requires an available preceding target set",
            path=f"$[{final_index - 1}]",
        )
    if binding_row.get("previous_truth_record_sha256") != prior.get(
        "content_sha256"
    ):
        _fail(
            "changed_binding_previous_truth_mismatch",
            "changed binding must identify the record immediately before its new revision",
            path="formal_window_target_binding.previous_truth_record_sha256",
        )
    if binding_row.get("previous_target_set_sha256") != prior_target.get(
        "target_rows_content_sha256"
    ):
        _fail(
            "changed_binding_previous_target_mismatch",
            "changed binding must bind the target set immediately before revision",
            path="formal_window_target_binding.previous_target_set_sha256",
        )
    exact_bindings = (
        (
            "formal_freeze_source_manifest_sha256",
            manifest.get("manifest_sha256"),
        ),
        (
            "formal_freeze_source_snapshot_sha256",
            manifest.get("source_snapshot_sha256"),
        ),
        (
            "formal_window_binding_row_sha256",
            binding_row.get("binding_row_sha256"),
        ),
        (
            "formal_preferred_rows_subset_sha256",
            binding_row.get("formal_preferred_rows_subset_sha256"),
        ),
        (
            "formal_preferred_row_count",
            binding_row.get("formal_preferred_row_count"),
        ),
        ("revision_reasons", binding_row.get("revision_reasons")),
        ("added", binding_row.get("added")),
        ("removed", binding_row.get("removed")),
        ("modified", binding_row.get("modified")),
    )
    for field, expected in exact_bindings:
        if final.get(field) != expected:
            _fail(
                "changed_revision_formal_binding_mismatch",
                f"new revision must copy {field} from the current formal binding",
                path=f"$[{final_index}].{field}",
            )
    if not any(
        _mapping(
            binding_row.get(field),
            path=f"formal_window_target_binding.{field}",
        ).get("row_count", 0)
        > 0
        for field in ("added", "removed", "modified")
    ):
        _fail(
            "changed_binding_has_no_change_rows",
            "changed formal binding requires at least one added, removed, or modified row",
            path="formal_window_target_binding",
        )
    observed, completed = _validate_formal_freeze_source_manifest(
        manifest,
        path="formal_freeze_source_manifest",
        protocol=protocol,
    )
    if _utc_timestamp(
        final.get("revision_observed_at_utc"),
        path=f"$[{final_index}].revision_observed_at_utc",
    ) != observed:
        _fail(
            "truth_revision_observed_not_formal_snapshot_observed",
            "changed revision observation must equal the current formal snapshot observation",
            path=f"$[{final_index}].revision_observed_at_utc",
        )
    if _utc_timestamp(
        final.get("revision_derived_started_at_utc"),
        path=f"$[{final_index}].revision_derived_started_at_utc",
    ) < completed:
        _fail(
            "truth_revision_derived_before_formal_manifest_completed",
            "changed revision derivation starts only after current formal freeze completion",
            path=f"$[{final_index}].revision_derived_started_at_utc",
        )


def validate_truth_chain(
    records: Sequence[Mapping[str, object]],
    protocol: Mapping[str, object] | None = None,
    *,
    issue_prediction_seal: Mapping[str, object] | None = None,
    formal_freeze_source_manifest: Mapping[str, object] | None = None,
    formal_window_target_binding: Mapping[str, object] | None = None,
) -> None:
    """Validate one issue/horizon mature-truth plus append-only revision chain."""

    records = _sequence(records, path="$")
    if not records:
        _fail("truth_chain_empty", "truth chain must contain its mature truth record")
    protocol = _DEFAULT_CALENDAR if protocol is None else _mapping(protocol, path="protocol")
    previous: Mapping[str, object] | None = None
    first_issue_id: object = None
    first_horizon: object = None
    original_visualization_hash: object = None
    expected_original: object = None
    if issue_prediction_seal is not None:
        prediction = _mapping(issue_prediction_seal, path="issue_prediction_seal")
        visualization = _mapping(
            prediction.get("visualization_evidence"),
            path="issue_prediction_seal.visualization_evidence",
        )
        expected_original = visualization.get("visualization_evidence_sha256")
    if formal_freeze_source_manifest is not None:
        formal_manifest = _mapping(
            formal_freeze_source_manifest,
            path="formal_freeze_source_manifest",
        )
        _validate_formal_freeze_source_manifest(
            formal_manifest,
            path="formal_freeze_source_manifest",
            protocol=protocol,
        )
        if formal_window_target_binding is None:
            _fail(
                "formal_window_target_binding_required",
                "current formal manifest requires its row-level truth binding",
                path="formal_window_target_binding",
            )

    for index, value in enumerate(records):
        path = f"$[{index}]"
        record = _mapping(value, path=path)
        validate_record_semantics(record, protocol)
        record_type = record.get("record_type")
        if index == 0:
            if record_type != "MatureTruthSnapshotRecord":
                _fail(
                    "truth_chain_first_record_not_mature",
                    "truth chain must begin with MatureTruthSnapshotRecord",
                    path=path,
                )
            if record.get("revision_sequence") not in {None, 0}:
                _fail(
                    "mature_truth_revision_sequence_invalid",
                    "mature truth revision_sequence must be zero",
                    path=f"{path}.revision_sequence",
                )
            if record.get("previous_truth_record_sha256") is not None:
                _fail(
                    "mature_truth_previous_record_not_null",
                    "mature truth previous record hash must be null",
                    path=f"{path}.previous_truth_record_sha256",
                )
            first_issue_id = record.get("issue_id")
            first_horizon = record.get("horizon_days")
            if (
                issue_prediction_seal is not None
                and record.get("issue_prediction_seal_sha256")
                != issue_prediction_seal.get("prediction_seal_sha256")
            ):
                _fail(
                    "mature_truth_prediction_seal_hash_mismatch",
                    "mature truth must bind the supplied issue PredictionSeal",
                    path=f"{path}.issue_prediction_seal_sha256",
                )
            if record.get("status") == "truth_snapshot_unavailable":
                if (
                    record.get("truth_snapshot") is not None
                    or record.get("replay_visualization") is not None
                ):
                    _fail(
                        "unavailable_truth_contains_snapshot_or_replay",
                        "unavailable mature truth must contain neither truth snapshot nor replay",
                        path=path,
                    )
                if len(records) != 1:
                    _fail(
                        "truth_revision_after_unavailable_forbidden",
                        "unavailable truth is terminal and cannot be recovered by revision",
                        path="$[1]",
                    )
                if (
                    formal_freeze_source_manifest is not None
                    and formal_window_target_binding is not None
                ):
                    _validate_formal_window_binding_against_truth_chain(
                        records,
                        _mapping(
                            formal_window_target_binding,
                            path="formal_window_target_binding",
                        ),
                        formal_manifest,
                        issue_prediction_seal=issue_prediction_seal,
                        protocol=protocol,
                    )
                return
        else:
            if record_type != "TruthRevisionRecord":
                _fail(
                    "truth_chain_non_revision_after_mature",
                    "only TruthRevisionRecord may follow mature truth",
                    path=path,
                )
            if record.get("revision_sequence") != index:
                _fail(
                    "truth_revision_sequence_not_contiguous",
                    "truth revision_sequence must be contiguous from one",
                    path=f"{path}.revision_sequence",
                )
            if (
                record.get("issue_id") != first_issue_id
                or record.get("horizon_days") != first_horizon
            ):
                _fail(
                    "truth_revision_scope_mismatch",
                    "all revisions must retain the mature issue_id and horizon",
                    path=path,
                )
            if record.get("previous_truth_record_sha256") != previous.get(
                "content_sha256"
            ):
                _fail(
                    "truth_revision_previous_record_hash_mismatch",
                    "revision must link to the immediately preceding truth record",
                    path=f"{path}.previous_truth_record_sha256",
                )
            previous_target = (
                previous.get("truth_snapshot")
                if previous.get("record_type") == "MatureTruthSnapshotRecord"
                else previous.get("revised_target_set")
            )
            if isinstance(previous_target, Mapping) and "realized_target_set" in previous_target:
                previous_target = previous_target["realized_target_set"]
            if (
                isinstance(previous_target, Mapping)
                and record.get("previous_target_set_sha256")
                != previous_target.get("target_rows_content_sha256")
            ):
                _fail(
                    "truth_revision_previous_target_set_hash_mismatch",
                    "revision must bind the immediately preceding target-set rows",
                    path=f"{path}.previous_target_set_sha256",
                )
        replay = _mapping(
            record.get("replay_visualization"),
            path=f"{path}.replay_visualization",
        )
        current_original = replay.get(
            "original_prediction_visualization_evidence_sha256"
        )
        if index == 0:
            original_visualization_hash = current_original
            if (
                expected_original is not None
                and current_original != expected_original
            ):
                _fail(
                    "mature_replay_original_prediction_hash_mismatch",
                    "mature replay must bind the issue PredictionSeal visualization",
                    path=(
                        f"{path}.replay_visualization."
                        "original_prediction_visualization_evidence_sha256"
                    ),
                )
        else:
            previous_replay = _mapping(
                previous.get("replay_visualization"),
                path=f"$[{index - 1}].replay_visualization",
            )
            if replay.get("previous_replay_visualization_sha256") != previous_replay.get(
                "replay_visualization_sha256"
            ):
                _fail(
                    "truth_revision_previous_replay_hash_mismatch",
                    "revision replay must link to the preceding replay hash",
                    path=(
                        f"{path}.replay_visualization."
                        "previous_replay_visualization_sha256"
                    ),
                )
            if current_original != original_visualization_hash:
                _fail(
                    "truth_revision_original_prediction_hash_changed",
                    "all revisions must retain the original prediction visualization hash",
                    path=(
                        f"{path}.replay_visualization."
                        "original_prediction_visualization_evidence_sha256"
                    ),
                )
        previous = record
    if (
        formal_freeze_source_manifest is not None
        and formal_window_target_binding is not None
    ):
        _validate_formal_window_binding_against_truth_chain(
            records,
            _mapping(
                formal_window_target_binding,
                path="formal_window_target_binding",
            ),
            formal_manifest,
            issue_prediction_seal=issue_prediction_seal,
            protocol=protocol,
        )


def _strict_yaml_mapping_bytes(raw: bytes, *, path: str) -> Mapping[str, object]:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise SemanticValidationError(
            "trusted_protocol_not_utf8",
            "trusted protocol config bytes must be strict UTF-8",
            path=path,
        ) from exc
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - locked environment dependency
        raise SemanticValidationError(
            "trusted_protocol_yaml_parser_unavailable",
            "PyYAML is required to inspect trusted protocol bytes",
            path=path,
        ) from exc

    class UniqueKeyLoader(yaml.SafeLoader):
        pass

    def construct_mapping(
        loader: yaml.SafeLoader,
        node: yaml.MappingNode,
        deep: bool = False,
    ) -> dict[object, object]:
        loader.flatten_mapping(node)
        result: dict[object, object] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in result:
                _fail(
                    "trusted_protocol_duplicate_yaml_key",
                    f"duplicate YAML mapping key is forbidden: {key!r}",
                    path=path,
                )
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    UniqueKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping,
    )
    try:
        value = yaml.load(text, Loader=UniqueKeyLoader)
    except SemanticValidationError:
        raise
    except yaml.YAMLError as exc:
        raise SemanticValidationError(
            "trusted_protocol_yaml_invalid",
            "trusted protocol config is not valid strict YAML",
            path=path,
        ) from exc
    return _mapping(value, path=path)


def _git_blob_sha1(raw: bytes) -> str:
    header = f"blob {len(raw)}\0".encode("ascii")
    return hashlib.sha1(header + raw).hexdigest()


def _release_file_bytes(
    files: Mapping[str, object],
    path_value: object,
    *,
    artifacts_by_sha256: Mapping[str, object | bytes],
    path: str,
) -> tuple[str, bytes]:
    if (
        not isinstance(path_value, str)
        or not path_value
        or "\\" in path_value
        or path_value.startswith("/")
        or any(part in {"", ".", ".."} for part in path_value.split("/"))
    ):
        _fail(
            "trusted_release_file_path_invalid",
            "trusted release paths must be normalized relative POSIX paths",
            path=path,
        )
    entry = _mapping(files.get(path_value), path=f"trusted_release_manifest.files.{path_value}")
    file_sha = entry.get("file_sha256")
    raw = _artifact_bytes(
        artifacts_by_sha256,
        file_sha,
        path=f"trusted_release_manifest.files.{path_value}.file_sha256",
    )
    if entry.get("git_blob_sha1") != _git_blob_sha1(raw):
        _fail(
            "trusted_release_git_blob_mismatch",
            "git_blob_sha1 must identify the exact trusted file bytes",
            path=f"trusted_release_manifest.files.{path_value}.git_blob_sha1",
        )
    if entry.get("commit_role") not in {"protocol_commit", "code_commit"}:
        _fail(
            "trusted_release_commit_role_invalid",
            "trusted release file commit_role must be protocol_commit or code_commit",
            path=f"trusted_release_manifest.files.{path_value}.commit_role",
        )
    return str(file_sha), raw


def _release_component_identity(
    path_value: object,
    *,
    files: Mapping[str, object],
    artifacts_by_sha256: Mapping[str, object | bytes],
    path: str,
) -> str:
    if not isinstance(path_value, str):
        _fail(
            "trusted_release_component_path_invalid",
            "each frozen CodeManifest component must map to one tagged-tree path",
            path=path,
        )
    file_sha, _ = _release_file_bytes(
        files,
        path_value,
        artifacts_by_sha256=artifacts_by_sha256,
        path=path,
    )
    return file_sha


def _validate_trusted_timestamp_registry(
    registry_value: object,
    *,
    protocol: Mapping[str, object],
    cohort_definition: Mapping[str, object] | None,
) -> None:
    import base64
    import binascii

    registry = _mapping(
        registry_value,
        path="trusted_release_manifest.timestamp_trust_registry",
    )
    remote_protocol = _mapping(
        protocol.get("remote_timestamp"),
        path="trusted_protocol.remote_timestamp",
    )
    protocol_registry = _mapping(
        remote_protocol.get("trusted_registry"),
        path="trusted_protocol.remote_timestamp.trusted_registry",
    )
    if dict(registry) != dict(protocol_registry):
        _fail(
            "trusted_timestamp_registry_not_from_protocol_release",
            "timestamp trust registry must equal the registry parsed from trusted protocol bytes",
            path="trusted_release_manifest.timestamp_trust_registry",
        )
    body = {key: value for key, value in registry.items() if key != "registry_sha256"}
    registry_sha = _sha256_bytes(canonical_json_bytes(body))
    if registry.get("registry_sha256") != registry_sha:
        _fail(
            "trusted_timestamp_registry_hash_mismatch",
            "registry_sha256 must hash the exact external registry excluding itself",
            path="trusted_release_manifest.timestamp_trust_registry.registry_sha256",
        )
    authorities = _sequence(
        registry.get("authorities"),
        path="trusted_release_manifest.timestamp_trust_registry.authorities",
    )
    anchor_bindings: list[Mapping[str, object]] = []
    policy_bindings: list[Mapping[str, object]] = []
    for index, authority_value in enumerate(authorities):
        authority_path = (
            "trusted_release_manifest.timestamp_trust_registry."
            f"authorities[{index}]"
        )
        authority = _mapping(authority_value, path=authority_path)
        if authority.get("attempt_index") != index:
            _fail(
                "trusted_timestamp_authority_order_mismatch",
                "trusted timestamp authority attempt indexes must be contiguous",
                path=f"{authority_path}.attempt_index",
            )
        anchor = _mapping(
            authority.get("pinned_anchor"),
            path=f"{authority_path}.pinned_anchor",
        )
        encoded = anchor.get("DER_base64")
        if not isinstance(encoded, str):
            _fail(
                "trusted_timestamp_anchor_base64_missing",
                "each protocol-pinned anchor must carry its exact DER bytes",
                path=f"{authority_path}.pinned_anchor.DER_base64",
            )
        try:
            der = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise SemanticValidationError(
                "trusted_timestamp_anchor_base64_invalid",
                "protocol-pinned anchor DER_base64 is invalid",
                path=f"{authority_path}.pinned_anchor.DER_base64",
            ) from exc
        if anchor.get("DER_sha256") != _sha256_bytes(der):
            _fail(
                "trusted_timestamp_anchor_der_hash_mismatch",
                "pinned anchor DER_sha256 must hash decoded DER bytes",
                path=f"{authority_path}.pinned_anchor.DER_sha256",
            )
        anchor_bindings.append(
            {
                "authority_url": authority.get("authority_url"),
                "pinned_anchor": anchor,
            }
        )
        policy_bindings.append(
            {
                "authority_url": authority.get("authority_url"),
                "allowed_TSTInfo_policy_oids": authority.get(
                    "allowed_TSTInfo_policy_oids"
                ),
            }
        )
    identities = _mapping(
        remote_protocol.get("trusted_registry_identity"),
        path="trusted_protocol.remote_timestamp.trusted_registry_identity",
    )
    expected = _mapping(
        identities.get("expected"),
        path="trusted_protocol.remote_timestamp.trusted_registry_identity.expected",
    )
    calculated = {
        "registry_sha256": registry_sha,
        "trust_anchor_bundle_sha256": _sha256_bytes(
            canonical_json_bytes(anchor_bindings)
        ),
        "allowed_policy_oid_set_sha256": _sha256_bytes(
            canonical_json_bytes(policy_bindings)
        ),
        "tsa_authority_identity_manifest_sha256": _sha256_bytes(
            canonical_json_bytes(authorities)
        ),
    }
    for field, value in calculated.items():
        if expected.get(field) != value:
            _fail(
                "trusted_timestamp_derived_identity_mismatch",
                f"{field} must be recomputed from the trusted protocol registry",
                path=(
                    "trusted_protocol.remote_timestamp."
                    f"trusted_registry_identity.expected.{field}"
                ),
            )
    if cohort_definition is not None:
        cohort_policy = _mapping(
            cohort_definition.get("remote_timestamp_policy"),
            path="cohort_definition.remote_timestamp_policy",
        )
        repeated = {
            **calculated,
            "verification_level": registry.get("verification_level"),
            "revocation_evaluated": _mapping(
                registry.get("revocation"),
                path=(
                    "trusted_release_manifest.timestamp_trust_registry.revocation"
                ),
            ).get("evaluated"),
            "full_non_revocation_claim": False,
        }
        for field, value in repeated.items():
            if cohort_policy.get(field) != value:
                _fail(
                    "cohort_timestamp_registry_binding_mismatch",
                    f"cohort remote_timestamp_policy.{field} must repeat the external registry",
                    path=f"cohort_definition.remote_timestamp_policy.{field}",
                )


def _validate_protocol_table_registry_against_schema(
    protocol: Mapping[str, object],
    schema: Mapping[str, object],
) -> None:
    target_cohort = _mapping(
        protocol.get("target_cohort"),
        path="trusted_protocol.target_cohort",
    )
    formal = _mapping(
        target_cohort.get("formal_freeze_source_manifest"),
        path="trusted_protocol.target_cohort.formal_freeze_source_manifest",
    )
    registry = _mapping(
        formal.get("derived_table_registry"),
        path=(
            "trusted_protocol.target_cohort.formal_freeze_source_manifest."
            "derived_table_registry"
        ),
    )
    tables = _mapping(
        registry.get("tables"),
        path=(
            "trusted_protocol.target_cohort.formal_freeze_source_manifest."
            "derived_table_registry.tables"
        ),
    )
    definitions = _mapping(schema.get("$defs"), path="trusted_schema.$defs")
    if set(tables) != set(_FORMAL_TABLE_SCHEMA_SHA256):
        _fail(
            "formal_table_registry_table_set_mismatch",
            "protocol table registry must contain every and only frozen formal table",
            path=(
                "trusted_protocol.target_cohort.formal_freeze_source_manifest."
                "derived_table_registry.tables"
            ),
        )
    for table_name, frozen_schema_sha in _FORMAL_TABLE_SCHEMA_SHA256.items():
        contract = _mapping(
            tables.get(table_name),
            path=(
                "trusted_protocol.target_cohort.formal_freeze_source_manifest."
                f"derived_table_registry.tables.{table_name}"
            ),
        )
        schema_ref = contract.get("row_schema_ref")
        if (
            not isinstance(schema_ref, str)
            or not schema_ref.startswith("#/$defs/")
        ):
            _fail(
                "formal_table_registry_row_schema_ref_invalid",
                "row_schema_ref must point directly into trusted schema $defs",
                path=(
                    "trusted_protocol.target_cohort.formal_freeze_source_manifest."
                    f"derived_table_registry.tables.{table_name}.row_schema_ref"
                ),
            )
        schema_name = schema_ref.removeprefix("#/$defs/")
        row_schema = _mapping(
            definitions.get(schema_name),
            path=f"trusted_schema.$defs.{schema_name}",
        )
        actual_schema_sha = _sha256_bytes(canonical_json_bytes(row_schema))
        if (
            actual_schema_sha != frozen_schema_sha
            or contract.get("expected_row_schema_sha256") != actual_schema_sha
        ):
            _fail(
                "formal_table_registry_schema_identity_mismatch",
                "validator constant, protocol expected hash, and actual row schema must agree",
                path=(
                    "trusted_protocol.target_cohort.formal_freeze_source_manifest."
                    f"derived_table_registry.tables.{table_name}."
                    "expected_row_schema_sha256"
                ),
            )
        sort_spec = _mapping(
            contract.get("sort_spec"),
            path=(
                "trusted_protocol.target_cohort.formal_freeze_source_manifest."
                f"derived_table_registry.tables.{table_name}.sort_spec"
            ),
        )
        actual_sort_sha = _sha256_bytes(canonical_json_bytes(sort_spec))
        if contract.get("expected_sort_order_sha256") != actual_sort_sha:
            _fail(
                "formal_table_registry_sort_identity_mismatch",
                "protocol expected sort hash must bind its actual frozen sort spec",
                path=(
                    "trusted_protocol.target_cohort.formal_freeze_source_manifest."
                    f"derived_table_registry.tables.{table_name}."
                    "expected_sort_order_sha256"
                ),
            )


def _validate_artifact_profile_registry_against_schema(
    protocol: Mapping[str, object],
    schema: Mapping[str, object],
) -> None:
    registry = _mapping(
        protocol.get("artifact_profile_registry"),
        path="trusted_protocol.artifact_profile_registry",
    )
    if registry.get("profile") != "stage2p_artifact_profile_registry_v1":
        _fail(
            "artifact_profile_registry_profile_mismatch",
            "artifact profile registry must use its frozen Stage 2P profile",
            path="trusted_protocol.artifact_profile_registry.profile",
        )
    canonical_jsonl = _mapping(
        registry.get("canonical_jsonl"),
        path="trusted_protocol.artifact_profile_registry.canonical_jsonl",
    )
    serialization_profile = canonical_jsonl.get("profile")
    if serialization_profile != "seismoflux_canonical_jsonl_v1":
        _fail(
            "artifact_profile_registry_serialization_profile_mismatch",
            "typed tables require the frozen canonical JSONL profile",
            path=(
                "trusted_protocol.artifact_profile_registry."
                "canonical_jsonl.profile"
            ),
        )
    definitions = _mapping(schema.get("$defs"), path="trusted_schema.$defs")

    identity_schemas = _mapping(
        registry.get("identity_schemas"),
        path="trusted_protocol.artifact_profile_registry.identity_schemas",
    )
    if set(identity_schemas) != _ARTIFACT_PROFILE_IDENTITY_SCHEMA_NAMES:
        _fail(
            "artifact_profile_registry_identity_schema_set_mismatch",
            "identity_schemas must contain every and only the 17 frozen definitions",
            path="trusted_protocol.artifact_profile_registry.identity_schemas",
        )
    for definition_name in sorted(_ARTIFACT_PROFILE_IDENTITY_SCHEMA_NAMES):
        identity_path = (
            "trusted_protocol.artifact_profile_registry."
            f"identity_schemas.{definition_name}"
        )
        contract = _mapping(
            identity_schemas.get(definition_name),
            path=identity_path,
        )
        expected_ref = f"#/$defs/{definition_name}"
        definition = _mapping(
            definitions.get(definition_name),
            path=f"trusted_schema.$defs.{definition_name}",
        )
        actual_sha256 = _sha256_bytes(canonical_json_bytes(definition))
        if (
            contract.get("schema_ref") != expected_ref
            or contract.get("expected_schema_sha256") != actual_sha256
        ):
            _fail(
                "artifact_profile_registry_identity_schema_hash_mismatch",
                "identity schema ref and hash must identify the exact trusted $defs object",
                path=identity_path,
            )

    embedded_manifests = _mapping(
        registry.get("embedded_manifests"),
        path="trusted_protocol.artifact_profile_registry.embedded_manifests",
    )
    if set(embedded_manifests) != {"selected_exposure", "alarm_area"}:
        _fail(
            "artifact_profile_registry_embedded_manifest_set_mismatch",
            "embedded_manifests must contain selected_exposure and alarm_area",
            path="trusted_protocol.artifact_profile_registry.embedded_manifests",
        )
    selected_contract = _mapping(
        embedded_manifests.get("selected_exposure"),
        path=(
            "trusted_protocol.artifact_profile_registry."
            "embedded_manifests.selected_exposure"
        ),
    )
    selected_schema = _mapping(
        definitions.get("SelectedExposureManifest"),
        path="trusted_schema.$defs.SelectedExposureManifest",
    )
    selected_schema_sha256 = (
        "3dd6742f69fa89463491fa3c73374412b5fb40ca00818b4849e71e568cfba005"
    )
    selected_row_schema_sha256 = (
        "ee2d9b532c534244c576a49a3c243ea73d7fe341856c36898e23bb417b57b09f"
    )
    selected_row_contract = _mapping(
        identity_schemas.get("SelectedExposureRow"),
        path=(
            "trusted_protocol.artifact_profile_registry."
            "identity_schemas.SelectedExposureRow"
        ),
    )
    selected_row_schema = _mapping(
        definitions.get("SelectedExposureRow"),
        path="trusted_schema.$defs.SelectedExposureRow",
    )
    if (
        selected_contract.get("schema_ref")
        != "#/$defs/SelectedExposureManifest"
        or selected_contract.get("expected_schema_sha256")
        != selected_schema_sha256
        or _sha256_bytes(canonical_json_bytes(selected_schema))
        != selected_schema_sha256
        or selected_row_contract.get("expected_schema_sha256")
        != selected_row_schema_sha256
        or _sha256_bytes(canonical_json_bytes(selected_row_schema))
        != selected_row_schema_sha256
        or selected_contract.get("row_sort")
        != [
            ["horizon_days", "frozen_order_7_then_30_then_90"],
            ["selection_ordinal_1_based", "integer_ascending_contiguous_from_1"],
            ["scheduled_issue_sequence", "integer_ascending"],
            ["issue_id", "unsigned_UTF8_ascending"],
        ]
    ):
        _fail(
            "artifact_profile_registry_selected_exposure_contract_mismatch",
            "selected exposure schema and row order must equal the frozen contract",
            path=(
                "trusted_protocol.artifact_profile_registry."
                "embedded_manifests.selected_exposure"
            ),
        )
    alarm_contract = _mapping(
        embedded_manifests.get("alarm_area"),
        path=(
            "trusted_protocol.artifact_profile_registry."
            "embedded_manifests.alarm_area"
        ),
    )
    alarm_schema = _mapping(
        definitions.get("AlarmAreaManifest"),
        path="trusted_schema.$defs.AlarmAreaManifest",
    )
    alarm_entry_schema = _mapping(
        definitions.get("AlarmAreaManifestEntry"),
        path="trusted_schema.$defs.AlarmAreaManifestEntry",
    )
    alarm_comparison_schema = _mapping(
        definitions.get("AlarmAreaComparison"),
        path="trusted_schema.$defs.AlarmAreaComparison",
    )
    alarm_schema_sha256 = (
        "ee499ed45a43be76ef8d79766178ea88ba7cbd2baf2e6b86cd70d0dc6510d0e3"
    )
    alarm_entry_schema_sha256 = (
        "223c759aee311936e96179f75a0d506192715dca47b6aeb05c1029ace8cb1f2a"
    )
    alarm_comparison_schema_sha256 = (
        "6f6d995674c8f1e4a78069dd3cb5e361e076379ffa5d2e49acb59f9bc160f179"
    )
    alarm_contract_matches = (
        alarm_contract.get("schema_ref") == "#/$defs/AlarmAreaManifest"
        and alarm_contract.get("expected_schema_sha256")
        == alarm_schema_sha256
        and _sha256_bytes(canonical_json_bytes(alarm_schema))
        == alarm_schema_sha256
        and alarm_contract.get("entry_schema_ref")
        == "#/$defs/AlarmAreaManifestEntry"
        and alarm_contract.get("entry_expected_schema_sha256")
        == alarm_entry_schema_sha256
        and _sha256_bytes(canonical_json_bytes(alarm_entry_schema))
        == alarm_entry_schema_sha256
        and alarm_contract.get("comparison_schema_ref")
        == "#/$defs/AlarmAreaComparison"
        and alarm_contract.get("comparison_expected_schema_sha256")
        == alarm_comparison_schema_sha256
        and _sha256_bytes(canonical_json_bytes(alarm_comparison_schema))
        == alarm_comparison_schema_sha256
        and alarm_contract.get(
            "maximum_allowed_pairwise_difference_km2_exact_float64"
        )
        == 625.0
        and alarm_contract.get(
            "maximum_allowed_pairwise_difference_km2_float64_hex"
        )
        == "4083880000000000"
    )
    if not alarm_contract_matches:
        _fail(
            "artifact_profile_registry_alarm_area_contract_mismatch",
            "alarm area schemas and 625 km2 threshold must equal the frozen contract",
            path=(
                "trusted_protocol.artifact_profile_registry."
                "embedded_manifests.alarm_area"
            ),
        )

    tables = _mapping(
        registry.get("tables"),
        path="trusted_protocol.artifact_profile_registry.tables",
    )
    if set(tables) != _ARTIFACT_PROFILE_TABLE_NAMES:
        _fail(
            "artifact_profile_registry_table_set_mismatch",
            "artifact profile registry must contain every and only the 19 frozen tables",
            path="trusted_protocol.artifact_profile_registry.tables",
        )
    identity_refs: set[str] = set()
    for table_name in sorted(_ARTIFACT_PROFILE_TABLE_NAMES):
        table_path = (
            f"trusted_protocol.artifact_profile_registry.tables.{table_name}"
        )
        contract = _mapping(tables.get(table_name), path=table_path)
        (
            frozen_identity_schema_ref,
            frozen_roles,
            frozen_row_schema_ref,
            frozen_sort_profile,
        ) = _ARTIFACT_PROFILE_TABLE_CONTRACTS[table_name]
        if (
            contract.get("identity_schema_ref") != frozen_identity_schema_ref
            or contract.get("row_schema_ref") != frozen_row_schema_ref
            or contract.get("sort_profile") != frozen_sort_profile
        ):
            _fail(
                "artifact_profile_registry_table_contract_mismatch",
                "table identity, row schema, and sort profile must equal the frozen contract",
                path=table_path,
            )
        if isinstance(frozen_roles, str):
            registry_roles_match = (
                contract.get("table_role") == frozen_roles
                and contract.get("table_roles") is None
            )
        else:
            registry_role_values = contract.get("table_roles")
            registry_roles_match = (
                contract.get("table_role") is None
                and isinstance(registry_role_values, Sequence)
                and not isinstance(
                    registry_role_values,
                    str | bytes | bytearray,
                )
                and tuple(registry_role_values) == frozen_roles
            )
        if not registry_roles_match:
            _fail(
                "artifact_profile_registry_table_contract_mismatch",
                "table role or roles must equal the frozen scientific contract",
                path=table_path,
            )
        row_schema_ref = contract.get("row_schema_ref")
        if (
            not isinstance(row_schema_ref, str)
            or not row_schema_ref.startswith("#/$defs/")
            or "/" in row_schema_ref.removeprefix("#/$defs/")
        ):
            _fail(
                "artifact_profile_registry_row_schema_ref_invalid",
                "row_schema_ref must point directly into trusted schema $defs",
                path=f"{table_path}.row_schema_ref",
            )
        row_schema_name = row_schema_ref.removeprefix("#/$defs/")
        row_schema = _mapping(
            definitions.get(row_schema_name),
            path=f"trusted_schema.$defs.{row_schema_name}",
        )
        row_schema_sha256 = _sha256_bytes(canonical_json_bytes(row_schema))
        if contract.get("expected_row_schema_sha256") != row_schema_sha256:
            _fail(
                "artifact_profile_registry_schema_identity_mismatch",
                "expected row schema hash must identify the exact trusted $defs object",
                path=f"{table_path}.expected_row_schema_sha256",
            )
        sort_spec = _sequence(
            contract.get("sort_spec"),
            path=f"{table_path}.sort_spec",
        )
        sort_order_sha256 = _sha256_bytes(canonical_json_bytes(sort_spec))
        if contract.get("expected_sort_order_sha256") != sort_order_sha256:
            _fail(
                "artifact_profile_registry_sort_identity_mismatch",
                "expected sort hash must identify the exact frozen sort specification",
                path=f"{table_path}.expected_sort_order_sha256",
            )

        identity_schema_ref = contract.get("identity_schema_ref")
        if (
            not isinstance(identity_schema_ref, str)
            or not identity_schema_ref.startswith("#/$defs/")
            or "/" in identity_schema_ref.removeprefix("#/$defs/")
        ):
            _fail(
                "artifact_profile_registry_identity_schema_ref_invalid",
                "identity_schema_ref must point directly into trusted schema $defs",
                path=f"{table_path}.identity_schema_ref",
            )
        if identity_schema_ref in identity_refs:
            _fail(
                "artifact_profile_registry_identity_schema_ref_reused",
                "each frozen table requires its own typed identity schema",
                path=f"{table_path}.identity_schema_ref",
            )
        identity_refs.add(identity_schema_ref)
        identity_schema_name = identity_schema_ref.removeprefix("#/$defs/")
        identity_schema = _mapping(
            definitions.get(identity_schema_name),
            path=f"trusted_schema.$defs.{identity_schema_name}",
        )
        all_of = _sequence(
            identity_schema.get("allOf"),
            path=f"trusted_schema.$defs.{identity_schema_name}.allOf",
        )
        if len(all_of) != 2:
            _fail(
                "artifact_profile_registry_typed_identity_mismatch",
                "typed identity must combine the base identity with one constraint branch",
                path=f"trusted_schema.$defs.{identity_schema_name}.allOf",
            )
        base = _mapping(
            all_of[0],
            path=f"trusted_schema.$defs.{identity_schema_name}.allOf[0]",
        )
        if base != {"$ref": "#/$defs/TableArtifactIdentity"}:
            _fail(
                "artifact_profile_registry_typed_identity_mismatch",
                "typed identity must inherit the frozen table artifact identity",
                path=f"trusted_schema.$defs.{identity_schema_name}.allOf[0]",
            )
        constraint = _mapping(
            all_of[1],
            path=f"trusted_schema.$defs.{identity_schema_name}.allOf[1]",
        )
        properties = _mapping(
            constraint.get("properties"),
            path=(
                f"trusted_schema.$defs.{identity_schema_name}."
                "allOf[1].properties"
            ),
        )
        expected_property_values = {
            "serialization_profile": ("const", serialization_profile),
            "row_schema_ref": ("const", row_schema_ref),
            "sort_profile": ("const", contract.get("sort_profile")),
            "schema_sha256": ("const", row_schema_sha256),
            "sort_order_sha256": ("const", sort_order_sha256),
        }
        for field, (keyword, expected_value) in expected_property_values.items():
            field_schema = _mapping(
                properties.get(field),
                path=(
                    f"trusted_schema.$defs.{identity_schema_name}."
                    f"allOf[1].properties.{field}"
                ),
            )
            if field_schema.get(keyword) != expected_value:
                _fail(
                    "artifact_profile_registry_typed_identity_mismatch",
                    f"typed identity {field} must repeat the registry constant",
                    path=(
                        f"trusted_schema.$defs.{identity_schema_name}."
                        f"allOf[1].properties.{field}.{keyword}"
                    ),
                )
        role_schema = _mapping(
            properties.get("table_role"),
            path=(
                f"trusted_schema.$defs.{identity_schema_name}."
                "allOf[1].properties.table_role"
            ),
        )
        if isinstance(frozen_roles, str):
            role_matches = role_schema.get("const") == frozen_roles
        else:
            role_values = role_schema.get("enum")
            role_matches = (
                isinstance(role_values, Sequence)
                and not isinstance(role_values, str | bytes | bytearray)
                and tuple(role_values) == frozen_roles
            )
        if not role_matches:
            _fail(
                "artifact_profile_registry_typed_identity_mismatch",
                "typed identity table_role must repeat the registry role contract",
                path=(
                    f"trusted_schema.$defs.{identity_schema_name}."
                    "allOf[1].properties.table_role"
                ),
            )

    manifests = _mapping(
        registry.get("manifests"),
        path="trusted_protocol.artifact_profile_registry.manifests",
    )
    if set(manifests) != _ARTIFACT_PROFILE_MANIFEST_NAMES:
        _fail(
            "artifact_profile_registry_manifest_set_mismatch",
            "artifact profile registry must contain every and only three manifests",
            path="trusted_protocol.artifact_profile_registry.manifests",
        )
    for manifest_name in sorted(_ARTIFACT_PROFILE_MANIFEST_NAMES):
        manifest_path = (
            "trusted_protocol.artifact_profile_registry."
            f"manifests.{manifest_name}"
        )
        contract = _mapping(manifests.get(manifest_name), path=manifest_path)
        frozen_schema_ref, frozen_profile, frozen_schema_sha256 = (
            _ARTIFACT_PROFILE_MANIFEST_CONTRACTS[manifest_name]
        )
        if (
            contract.get("schema_ref") != frozen_schema_ref
            or contract.get("profile") != frozen_profile
            or contract.get("expected_schema_sha256")
            != frozen_schema_sha256
        ):
            _fail(
                "artifact_profile_registry_manifest_contract_mismatch",
                "manifest schema and profile must equal the frozen contract",
                path=manifest_path,
            )
        schema_ref = contract.get("schema_ref")
        if (
            not isinstance(schema_ref, str)
            or not schema_ref.startswith("#/$defs/")
            or "/" in schema_ref.removeprefix("#/$defs/")
        ):
            _fail(
                "artifact_profile_registry_manifest_schema_ref_invalid",
                "manifest schema_ref must point directly into trusted schema $defs",
                path=f"{manifest_path}.schema_ref",
            )
        schema_name = schema_ref.removeprefix("#/$defs/")
        manifest_schema = _mapping(
            definitions.get(schema_name),
            path=f"trusted_schema.$defs.{schema_name}",
        )
        actual_schema_sha256 = _sha256_bytes(
            canonical_json_bytes(manifest_schema)
        )
        if actual_schema_sha256 != frozen_schema_sha256:
            _fail(
                "artifact_profile_registry_manifest_schema_identity_mismatch",
                "manifest schema hash must identify the exact trusted $defs object",
                path=f"{manifest_path}.expected_schema_sha256",
            )
        properties = _mapping(
            manifest_schema.get("properties"),
            path=f"trusted_schema.$defs.{schema_name}.properties",
        )
        profile_schema = _mapping(
            properties.get("profile"),
            path=f"trusted_schema.$defs.{schema_name}.properties.profile",
        )
        if profile_schema.get("const") != contract.get("profile"):
            _fail(
                "artifact_profile_registry_manifest_profile_mismatch",
                "manifest schema profile must repeat the registry profile",
                path=f"trusted_schema.$defs.{schema_name}.properties.profile.const",
            )

    raw_arrays = _mapping(
        registry.get("raw_arrays"),
        path="trusted_protocol.artifact_profile_registry.raw_arrays",
    )
    if raw_arrays.get("identity_schema_ref") != (
        "#/$defs/RawArrayArtifactIdentity"
    ) or "RawArrayArtifactIdentity" not in definitions:
        _fail(
            "artifact_profile_registry_raw_array_schema_mismatch",
            "raw array registry must bind RawArrayArtifactIdentity",
            path=(
                "trusted_protocol.artifact_profile_registry."
                "raw_arrays.identity_schema_ref"
            ),
        )
    density_contract = _mapping(
        registry.get("density_contract"),
        path="trusted_protocol.artifact_profile_registry.density_contract",
    )
    if density_contract.get("identity_schema_ref") != (
        "#/$defs/DensityIdentity"
    ) or "DensityIdentity" not in definitions:
        _fail(
            "artifact_profile_registry_density_schema_mismatch",
            "density registry must bind DensityIdentity",
            path=(
                "trusted_protocol.artifact_profile_registry."
                "density_contract.identity_schema_ref"
            ),
        )
    float64_bits_schema = _mapping(
        definitions.get("Float64BitsHex"),
        path="trusted_schema.$defs.Float64BitsHex",
    )
    if (
        float64_bits_schema.get("type") != "string"
        or float64_bits_schema.get("pattern") != "^[0-9a-f]{16}$"
    ):
        _fail(
            "artifact_profile_registry_float64_bits_schema_mismatch",
            "Float64BitsHex must freeze exactly eight bytes as lowercase hex",
            path="trusted_schema.$defs.Float64BitsHex",
        )

    def require_direct_property_refs(
        definition_name: str,
        expected: Mapping[str, str],
    ) -> None:
        definition = _mapping(
            definitions.get(definition_name),
            path=f"trusted_schema.$defs.{definition_name}",
        )
        properties = _mapping(
            definition.get("properties"),
            path=f"trusted_schema.$defs.{definition_name}.properties",
        )
        for field, expected_ref in expected.items():
            field_schema = _mapping(
                properties.get(field),
                path=(
                    f"trusted_schema.$defs.{definition_name}."
                    f"properties.{field}"
                ),
            )
            if field_schema != {"$ref": expected_ref}:
                _fail(
                    "artifact_profile_registry_schema_structure_mismatch",
                    f"{definition_name}.{field} must use its frozen typed schema",
                    path=(
                        f"trusted_schema.$defs.{definition_name}."
                        f"properties.{field}"
                    ),
                )

    require_direct_property_refs(
        "SourceSnapshot",
        {
            "cutover_cross_source_match_rows": (
                "#/$defs/CutoverMatchRowsArtifactIdentity"
            ),
            "normalized_rows": "#/$defs/SourceNormalizedRowsArtifactIdentity",
            "deduplicated_rows": (
                "#/$defs/SourceDeduplicatedRowsArtifactIdentity"
            ),
            "causal_model_view": "#/$defs/CausalModelViewRowsArtifactIdentity",
            "P0_event_set": "#/$defs/P0EventSetIdentity",
            "R30_event_set": "#/$defs/R30EventSetIdentity",
            "RP30_event_set": "#/$defs/RP30EventSetIdentity",
        },
    )
    require_direct_property_refs(
        "TruthSnapshot",
        {
            "window_membership_rows": (
                "#/$defs/TruthWindowMembershipRowsArtifactIdentity"
            ),
            "normalized_rows": "#/$defs/TruthNormalizedRowsArtifactIdentity",
            "deduplicated_rows": (
                "#/$defs/TruthDeduplicatedRowsArtifactIdentity"
            ),
            "realized_target_set": "#/$defs/TargetSetIdentity",
        },
    )
    require_direct_property_refs(
        "EvaluationFreezeRecord",
        {
            "selected_exposure_manifest": (
                "#/$defs/SelectedExposureManifest"
            ),
            "alarm_area_manifest": "#/$defs/AlarmAreaManifest",
        },
    )
    require_direct_property_refs(
        "EventSetIdentity",
        {"event_rows": "#/$defs/EventSetRowsArtifactIdentity"},
    )
    require_direct_property_refs(
        "TargetSetIdentity",
        {"target_rows": "#/$defs/ScientificTargetRowsArtifactIdentity"},
    )
    require_direct_property_refs(
        "GridFamilyIdentity",
        {
            "grid_geometry_rows": (
                "#/$defs/GridCellGeometryRowsArtifactIdentity"
            )
        },
    )
    require_direct_property_refs(
        "ForecastArtifactSetIdentity",
        {
            "score_rows": "#/$defs/GridScoreRowsArtifactIdentity",
            "ranked_rows": "#/$defs/RankedGridRowsArtifactIdentity",
            "alarm_prefix_rows": "#/$defs/AlarmPrefixRowsArtifactIdentity",
            "alarm_mask": "#/$defs/MaskArrayArtifactIdentity",
        },
    )
    require_direct_property_refs(
        "ForecastBundleManifest",
        {"grid_family": "#/$defs/GridFamilyIdentity"},
    )
    require_direct_property_refs(
        "EffectRowsManifest",
        {
            "cluster_membership_rows": (
                "#/$defs/ClusterMembershipRowsArtifactIdentity"
            ),
            "point_contribution_rows": (
                "#/$defs/PointContributionArtifactMap"
            ),
            "region_contribution_rows": (
                "#/$defs/RegionContributionArtifactMap"
            ),
            "cluster_contribution_rows": (
                "#/$defs/ClusterContributionArtifactMap"
            ),
            "bootstrap_index_rows": (
                "#/$defs/BootstrapIndexRowsArtifactIdentity"
            ),
        },
    )
    require_direct_property_refs(
        "DensityIdentity",
        {
            "mass_12_5km": "#/$defs/Float64ArrayArtifactIdentity",
            "direct_mass_25km": "#/$defs/Float64ArrayArtifactIdentity",
            "mass_25km": "#/$defs/Float64ArrayArtifactIdentity",
            "direct_mass_50km": "#/$defs/Float64ArrayArtifactIdentity",
        },
    )
    require_direct_property_refs(
        "AlarmAreaManifestEntry",
        {
            "P0_actual_alarm_area_km2_float64_hex": (
                "#/$defs/Float64BitsHex"
            ),
            "P1_actual_alarm_area_km2_float64_hex": (
                "#/$defs/Float64BitsHex"
            ),
            "PP_actual_alarm_area_km2_float64_hex": (
                "#/$defs/Float64BitsHex"
            ),
            "maximum_pairwise_actual_alarm_area_difference_km2_float64_hex": (
                "#/$defs/Float64BitsHex"
            ),
        },
    )
    require_direct_property_refs(
        "AlarmAreaManifest",
        {
            "maximum_allowed_pairwise_difference_km2_float64_hex": (
                "#/$defs/Float64BitsHex"
            )
        },
    )
    require_direct_property_refs(
        "AlarmAreaComparison",
        {
            "maximum_pairwise_actual_alarm_area_difference_km2_float64_hex": (
                "#/$defs/Float64BitsHex"
            )
        },
    )

    for definition_name, field, expected_ref in (
        (
            "SelectedExposureManifest",
            "rows",
            "#/$defs/SelectedExposureRow",
        ),
        (
            "AlarmAreaManifest",
            "entries",
            "#/$defs/AlarmAreaManifestEntry",
        ),
    ):
        definition = _mapping(
            definitions.get(definition_name),
            path=f"trusted_schema.$defs.{definition_name}",
        )
        properties = _mapping(
            definition.get("properties"),
            path=f"trusted_schema.$defs.{definition_name}.properties",
        )
        array_schema = _mapping(
            properties.get(field),
            path=(
                f"trusted_schema.$defs.{definition_name}.properties.{field}"
            ),
        )
        items = _mapping(
            array_schema.get("items"),
            path=(
                f"trusted_schema.$defs.{definition_name}."
                f"properties.{field}.items"
            ),
        )
        if items != {"$ref": expected_ref}:
            _fail(
                "artifact_profile_registry_schema_structure_mismatch",
                f"{definition_name}.{field} must use its frozen row schema",
                path=(
                    f"trusted_schema.$defs.{definition_name}."
                    f"properties.{field}.items"
                ),
            )

    forecast_bundle = _mapping(
        definitions.get("ForecastBundleManifest"),
        path="trusted_schema.$defs.ForecastBundleManifest",
    )
    forecast_properties = _mapping(
        forecast_bundle.get("properties"),
        path="trusted_schema.$defs.ForecastBundleManifest.properties",
    )
    for container_field, fields, expected_ref in (
        (
            "densities",
            ("P0", "R30", "RP30", "P1", "PP"),
            "#/$defs/DensityIdentity",
        ),
        (
            "forecasts",
            ("P0", "P1", "PP"),
            "#/$defs/ForecastArtifactSetIdentity",
        ),
    ):
        container = _mapping(
            forecast_properties.get(container_field),
            path=(
                "trusted_schema.$defs.ForecastBundleManifest."
                f"properties.{container_field}"
            ),
        )
        container_properties = _mapping(
            container.get("properties"),
            path=(
                "trusted_schema.$defs.ForecastBundleManifest."
                f"properties.{container_field}.properties"
            ),
        )
        if set(container.get("required", ())) != set(fields):
            _fail(
                "artifact_profile_registry_schema_structure_mismatch",
                f"ForecastBundleManifest.{container_field} key set is not frozen",
                path=(
                    "trusted_schema.$defs.ForecastBundleManifest."
                    f"properties.{container_field}.required"
                ),
            )
        for field in fields:
            field_schema = _mapping(
                container_properties.get(field),
                path=(
                    "trusted_schema.$defs.ForecastBundleManifest."
                    f"properties.{container_field}.properties.{field}"
                ),
            )
            if field_schema != {"$ref": expected_ref}:
                _fail(
                    "artifact_profile_registry_schema_structure_mismatch",
                    f"{container_field}.{field} must use its frozen typed schema",
                    path=(
                        "trusted_schema.$defs.ForecastBundleManifest."
                        f"properties.{container_field}.properties.{field}"
                    ),
                )

    endpoint_ids = (
        "P1_minus_P0_macro_information_gain",
        "P1_minus_PP_macro_information_gain",
        "P1_minus_P0_macro_recall_gain",
        "P1_minus_PP_macro_recall_gain",
    )
    for definition_name, expected_ref in (
        (
            "PointContributionArtifactMap",
            "#/$defs/PointContributionRowsArtifactIdentity",
        ),
        (
            "RegionContributionArtifactMap",
            "#/$defs/RegionContributionRowsArtifactIdentity",
        ),
        (
            "ClusterContributionArtifactMap",
            "#/$defs/ClusterContributionRowsArtifactIdentity",
        ),
        (
            "BootstrapDistributionArtifactMap",
            "#/$defs/BootstrapDistributionRowsArtifactIdentity",
        ),
    ):
        definition = _mapping(
            definitions.get(definition_name),
            path=f"trusted_schema.$defs.{definition_name}",
        )
        properties = _mapping(
            definition.get("properties"),
            path=f"trusted_schema.$defs.{definition_name}.properties",
        )
        if set(definition.get("required", ())) != set(endpoint_ids):
            _fail(
                "artifact_profile_registry_schema_structure_mismatch",
                f"{definition_name} must require the frozen four endpoints",
                path=f"trusted_schema.$defs.{definition_name}.required",
            )
        for endpoint_id in endpoint_ids:
            endpoint_schema = _mapping(
                properties.get(endpoint_id),
                path=(
                    f"trusted_schema.$defs.{definition_name}."
                    f"properties.{endpoint_id}"
                ),
            )
            if endpoint_schema != {"$ref": expected_ref}:
                _fail(
                    "artifact_profile_registry_schema_structure_mismatch",
                    f"{definition_name} endpoint must use its typed table identity",
                    path=(
                        f"trusted_schema.$defs.{definition_name}."
                        f"properties.{endpoint_id}"
                    ),
                )

    def nested_refs(value: object) -> set[str]:
        refs: set[str] = set()
        if isinstance(value, Mapping):
            reference = value.get("$ref")
            if isinstance(reference, str):
                refs.add(reference)
            for nested in value.values():
                refs.update(nested_refs(nested))
        elif isinstance(value, Sequence) and not isinstance(
            value,
            str | bytes | bytearray,
        ):
            for nested in value:
                refs.update(nested_refs(nested))
        return refs

    density = _mapping(
        definitions.get("DensityIdentity"),
        path="trusted_schema.$defs.DensityIdentity",
    )
    density_properties = _mapping(
        density.get("properties"),
        path="trusted_schema.$defs.DensityIdentity.properties",
    )
    projected_xy_schema = _mapping(
        density_properties.get("projected_xy_array"),
        path=(
            "trusted_schema.$defs.DensityIdentity."
            "properties.projected_xy_array"
        ),
    )
    if nested_refs(projected_xy_schema) != {
        "#/$defs/Float64ArrayArtifactIdentity"
    }:
        _fail(
            "artifact_profile_registry_schema_structure_mismatch",
            "projected event coordinates must use the float64 array identity",
            path=(
                "trusted_schema.$defs.DensityIdentity."
                "properties.projected_xy_array"
            ),
        )
    result_bundle = _mapping(
        definitions.get("ResultBundleManifest"),
        path="trusted_schema.$defs.ResultBundleManifest",
    )
    result_properties = _mapping(
        result_bundle.get("properties"),
        path="trusted_schema.$defs.ResultBundleManifest.properties",
    )
    bootstrap_schema = _mapping(
        result_properties.get("bootstrap_distribution_rows"),
        path=(
            "trusted_schema.$defs.ResultBundleManifest."
            "properties.bootstrap_distribution_rows"
        ),
    )
    if nested_refs(bootstrap_schema) != {
        "#/$defs/BootstrapDistributionArtifactMap"
    }:
        _fail(
            "artifact_profile_registry_schema_structure_mismatch",
            "result bundle bootstrap slot must use the frozen endpoint map",
            path=(
                "trusted_schema.$defs.ResultBundleManifest."
                "properties.bootstrap_distribution_rows"
            ),
        )

    for definition_name, dtype in (
        ("Float64ArrayArtifactIdentity", "<f8"),
        ("MaskArrayArtifactIdentity", "|u1"),
    ):
        wrapper = _mapping(
            definitions.get(definition_name),
            path=f"trusted_schema.$defs.{definition_name}",
        )
        all_of = _sequence(
            wrapper.get("allOf"),
            path=f"trusted_schema.$defs.{definition_name}.allOf",
        )
        if len(all_of) != 2 or _mapping(
            all_of[0],
            path=f"trusted_schema.$defs.{definition_name}.allOf[0]",
        ) != {"$ref": "#/$defs/RawArrayArtifactIdentity"}:
            _fail(
                "artifact_profile_registry_schema_structure_mismatch",
                f"{definition_name} must inherit RawArrayArtifactIdentity",
                path=f"trusted_schema.$defs.{definition_name}.allOf",
            )
        constraint = _mapping(
            all_of[1],
            path=f"trusted_schema.$defs.{definition_name}.allOf[1]",
        )
        properties = _mapping(
            constraint.get("properties"),
            path=(
                f"trusted_schema.$defs.{definition_name}."
                "allOf[1].properties"
            ),
        )
        dtype_schema = _mapping(
            properties.get("dtype"),
            path=(
                f"trusted_schema.$defs.{definition_name}."
                "allOf[1].properties.dtype"
            ),
        )
        if dtype_schema != {"const": dtype}:
            _fail(
                "artifact_profile_registry_schema_structure_mismatch",
                f"{definition_name} must freeze dtype {dtype}",
                path=(
                    f"trusted_schema.$defs.{definition_name}."
                    "allOf[1].properties.dtype"
                ),
            )


def _validate_trusted_release_boundary(
    cohort_definition: Mapping[str, object],
    *,
    artifacts_by_sha256: Mapping[str, object | bytes],
    schema: Mapping[str, object] | None,
    protocol: Mapping[str, object] | None,
    trusted_release_manifest: Mapping[str, object],
    trusted_release_manifest_file_sha256: str,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    release = _mapping(
        trusted_release_manifest,
        path="trusted_release_manifest",
    )
    release_bytes = _artifact_bytes(
        artifacts_by_sha256,
        trusted_release_manifest_file_sha256,
        path="trusted_release_manifest_file_sha256",
    )
    parsed_release = parse_record_json_bytes(release_bytes)
    if dict(parsed_release) != dict(release):
        _fail(
            "trusted_release_manifest_bytes_mapping_mismatch",
            "trusted release exact JSON bytes must decode to the supplied mapping",
            path="trusted_release_manifest",
        )
    if release_bytes != canonical_json_bytes(parsed_release):
        _fail(
            "trusted_release_manifest_bytes_not_canonical",
            "trusted release file must use its one exact canonical JSON byte encoding",
            path="trusted_release_manifest_file_sha256",
        )
    if release.get("profile") != "stage2p_trusted_release_manifest_v1":
        _fail(
            "trusted_release_profile_invalid",
            "trusted release must use stage2p_trusted_release_manifest_v1",
            path="trusted_release_manifest.profile",
        )
    implementation_status = release.get("implementation_status")
    if implementation_status not in {
        LIFECYCLE_IMPLEMENTATION_STATUS,
        "stage2p1b_synthetic_accepted",
    }:
        _fail(
            "trusted_release_implementation_status_invalid",
            "trusted release implementation status is not recognized",
            path="trusted_release_manifest.implementation_status",
        )
    _validate_canonical_self_hash(
        release,
        hash_field="manifest_sha256",
        subject="trusted_release_manifest",
        path="trusted_release_manifest",
    )
    files = _mapping(release.get("files"), path="trusted_release_manifest.files")
    if not files:
        _fail(
            "trusted_release_files_empty",
            "trusted release must map exact tagged-tree paths to file and Git blob hashes",
            path="trusted_release_manifest.files",
        )
    protocol_path = release.get("protocol_config_path")
    protocol_sha, protocol_bytes = _release_file_bytes(
        files,
        protocol_path,
        artifacts_by_sha256=artifacts_by_sha256,
        path="trusted_release_manifest.protocol_config_path",
    )
    schema_path = release.get("record_schema_path")
    schema_sha, schema_bytes = _release_file_bytes(
        files,
        schema_path,
        artifacts_by_sha256=artifacts_by_sha256,
        path="trusted_release_manifest.record_schema_path",
    )
    parsed_protocol = _strict_yaml_mapping_bytes(
        protocol_bytes,
        path=f"trusted_release_manifest.files.{protocol_path}",
    )
    parsed_schema_value = _strict_json_value_bytes(
        schema_bytes,
        path=f"trusted_release_manifest.files.{schema_path}",
    )
    parsed_schema = _mapping(
        parsed_schema_value,
        path=f"trusted_release_manifest.files.{schema_path}",
    )
    _validate_protocol_table_registry_against_schema(
        parsed_protocol,
        parsed_schema,
    )
    _validate_artifact_profile_registry_against_schema(
        parsed_protocol,
        parsed_schema,
    )
    release_schema: Mapping[str, object] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$ref": "#/$defs/TrustedReleaseManifest",
        "$defs": parsed_schema.get("$defs"),
    }
    Draft202012Validator(
        release_schema,
        format_checker=_FORMAT_CHECKER,
    ).validate(release)
    if schema is not None and dict(_mapping(schema, path="schema")) != dict(
        parsed_schema
    ):
        _fail(
            "supplied_schema_not_trusted_release_schema",
            "supplied schema must exactly equal the object parsed from trusted bytes",
            path="schema",
        )
    if protocol is not None and dict(_mapping(protocol, path="protocol")) != dict(
        parsed_protocol
    ):
        _fail(
            "supplied_protocol_not_trusted_release_protocol",
            "supplied protocol must exactly equal the mapping parsed from trusted bytes",
            path="protocol",
        )
    protocol_identity = _mapping(
        parsed_protocol.get("protocol"),
        path="trusted_protocol.protocol",
    )
    if release.get("protocol_tag") != protocol_identity.get("protocol_tag"):
        _fail(
            "trusted_release_protocol_tag_not_tagged_config",
            "trusted release protocol_tag must equal the trusted config constant",
            path="trusted_release_manifest.protocol_tag",
        )
    for required_protocol_path in (protocol_path, schema_path):
        entry = _mapping(
            files.get(required_protocol_path),
            path=f"trusted_release_manifest.files.{required_protocol_path}",
        )
        if entry.get("commit_role") != "protocol_commit":
            _fail(
                "trusted_release_protocol_file_role_mismatch",
                "protocol config and record schema must come from protocol_commit",
                path=(
                    f"trusted_release_manifest.files.{required_protocol_path}."
                    "commit_role"
                ),
            )
    component_paths = _mapping(
        release.get("component_paths"),
        path="trusted_release_manifest.component_paths",
    )
    record_contract = _mapping(
        parsed_protocol.get("record_contract"),
        path="trusted_protocol.record_contract",
    )
    release_contract = _mapping(
        record_contract.get("trusted_release_manifest_contract"),
        path="trusted_protocol.record_contract.trusted_release_manifest_contract",
    )
    fixed_component_paths = _mapping(
        release_contract.get("fixed_component_paths"),
        path=(
            "trusted_protocol.record_contract.trusted_release_manifest_contract."
            "fixed_component_paths"
        ),
    )
    if dict(component_paths) != dict(fixed_component_paths):
        _fail(
            "trusted_release_component_paths_not_frozen_mapping",
            "component_paths must exactly equal the mapping frozen in tagged protocol bytes",
            path="trusted_release_manifest.component_paths",
        )
    fixed_path_roles = _mapping(
        release_contract.get("fixed_path_commit_roles"),
        path=(
            "trusted_protocol.record_contract.trusted_release_manifest_contract."
            "fixed_path_commit_roles"
        ),
    )
    release_registry = release.get("timestamp_trust_registry")
    if release.get("timestamp_trust_registry_sha256") != (
        _mapping(
            release_registry,
            path="trusted_release_manifest.timestamp_trust_registry",
        ).get("registry_sha256")
    ):
        _fail(
            "trusted_release_timestamp_registry_top_level_mismatch",
            "release timestamp registry hash must equal its nested registry identity",
            path="trusted_release_manifest.timestamp_trust_registry_sha256",
        )
    mirrored_protocol_files = _mapping(
        release.get("code_commit_mirrored_protocol_files"),
        path="trusted_release_manifest.code_commit_mirrored_protocol_files",
    )
    if implementation_status == LIFECYCLE_IMPLEMENTATION_STATUS:
        if mirrored_protocol_files:
            _fail(
                "stage2p1a_mirrored_protocol_files_not_empty",
                "Stage 2P-1A has no code commit and requires an empty mirror mapping",
                path="trusted_release_manifest.code_commit_mirrored_protocol_files",
            )
    else:
        mirror_paths = {str(protocol_path), str(schema_path)}
        if set(mirrored_protocol_files) != mirror_paths:
            _fail(
                "trusted_release_protocol_mirror_path_set_mismatch",
                "accepted code commit must mirror exactly protocol config and record schema",
                path="trusted_release_manifest.code_commit_mirrored_protocol_files",
            )
        for mirror_path in sorted(mirror_paths):
            mirrored = _mapping(
                mirrored_protocol_files.get(mirror_path),
                path=(
                    "trusted_release_manifest.code_commit_mirrored_protocol_files."
                    f"{mirror_path}"
                ),
            )
            protocol_entry = _mapping(
                files.get(mirror_path),
                path=f"trusted_release_manifest.files.{mirror_path}",
            )
            if mirrored.get("commit_role") != "code_commit" or any(
                mirrored.get(field) != protocol_entry.get(field)
                for field in ("file_sha256", "git_blob_sha1")
            ):
                _fail(
                    "trusted_release_protocol_mirror_identity_mismatch",
                    "code-commit protocol mirror must byte-match the protocol-commit file",
                    path=(
                        "trusted_release_manifest."
                        f"code_commit_mirrored_protocol_files.{mirror_path}"
                    ),
                )
    if implementation_status == LIFECYCLE_IMPLEMENTATION_STATUS:
        stage2p1a_paths = tuple(
            _sequence(
                release_contract.get("Stage2P_1A_required_existing_protocol_files"),
                path=(
                    "trusted_protocol.record_contract."
                    "trusted_release_manifest_contract."
                    "Stage2P_1A_required_existing_protocol_files"
                ),
            )
        )
        expected_file_paths = set(stage2p1a_paths)
        expected_path_roles = _mapping(
            release_contract.get("Stage2P_1A_path_commit_roles"),
            path=(
                "trusted_protocol.record_contract."
                "trusted_release_manifest_contract."
                "Stage2P_1A_path_commit_roles"
            ),
        )
        if set(expected_path_roles) != expected_file_paths:
            _fail(
                "stage2p1a_path_role_set_mismatch",
                "Stage 2P-1A path roles must cover exactly its six protocol files",
                path=(
                    "trusted_protocol.record_contract."
                    "trusted_release_manifest_contract."
                    "Stage2P_1A_path_commit_roles"
                ),
            )
    else:
        expected_file_paths = set(fixed_path_roles)
        expected_path_roles = fixed_path_roles
    if set(files) != expected_file_paths:
        _fail(
            "trusted_release_file_path_set_mismatch",
            "release files must equal the frozen path set for its implementation status",
            path="trusted_release_manifest.files",
        )
    if any(not isinstance(item, str) for item in expected_file_paths):
        _fail(
            "trusted_release_file_path_invalid",
            "every frozen release path must be a string",
            path="trusted_release_manifest.files",
        )
    for release_path in sorted(expected_file_paths):
        _release_file_bytes(
            files,
            release_path,
            artifacts_by_sha256=artifacts_by_sha256,
            path=f"trusted_release_manifest.files.{release_path}",
        )
        entry = _mapping(
            files.get(release_path),
            path=f"trusted_release_manifest.files.{release_path}",
        )
        if entry.get("commit_role") != expected_path_roles.get(release_path):
            _fail(
                "trusted_release_file_commit_role_mismatch",
                "release file commit_role must equal the path role frozen in protocol",
                path=(
                    f"trusted_release_manifest.files.{release_path}."
                    "commit_role"
                ),
            )
    if implementation_status == LIFECYCLE_IMPLEMENTATION_STATUS:
        _validate_trusted_timestamp_registry(
            release_registry,
            protocol=parsed_protocol,
            cohort_definition=None,
        )
        return parsed_schema, parsed_protocol

    if cohort_definition.get("protocol_config_sha256") != protocol_sha:
        _fail(
            "cohort_protocol_config_not_trusted_release",
            "cohort protocol_config_sha256 must hash the trusted protocol bytes",
            path="cohort_definition.protocol_config_sha256",
        )
    if cohort_definition.get("record_schema_sha256") != schema_sha:
        _fail(
            "cohort_record_schema_not_trusted_release",
            "cohort record_schema_sha256 must hash the trusted schema bytes",
            path="cohort_definition.record_schema_sha256",
        )
    for field in ("protocol_tag", "code_tag", "protocol_commit", "code_commit"):
        if release.get(field) != cohort_definition.get(field):
            _fail(
                "cohort_release_tag_or_commit_mismatch",
                f"cohort {field} must equal the caller-trusted peeled release identity",
                path=f"cohort_definition.{field}",
            )
    code_manifest = _mapping(
        cohort_definition.get("code_manifest"),
        path="cohort_definition.code_manifest",
    )
    _validate_code_manifest_identity(
        code_manifest,
        path="cohort_definition.code_manifest",
    )
    if cohort_definition.get("code_manifest_sha256") != code_manifest.get(
        "manifest_sha256"
    ):
        _fail(
            "cohort_code_manifest_top_level_mismatch",
            "cohort code_manifest_sha256 must equal the nested manifest identity",
            path="cohort_definition.code_manifest_sha256",
        )
    component_fields = {
        key for key in code_manifest if key != "manifest_sha256"
    }
    if set(component_paths) != component_fields:
        _fail(
            "trusted_release_component_path_set_mismatch",
            "trusted release must map every and only CodeManifest component field",
            path="trusted_release_manifest.component_paths",
        )
    for field in sorted(component_fields):
        calculated = _release_component_identity(
            component_paths[field],
            files=files,
            artifacts_by_sha256=artifacts_by_sha256,
            path=f"trusted_release_manifest.component_paths.{field}",
        )
        if code_manifest.get(field) != calculated:
            _fail(
                "cohort_code_component_not_trusted_release",
                f"CodeManifest {field} does not bind its tagged-tree file bytes",
                path=f"cohort_definition.code_manifest.{field}",
            )
    _validate_trusted_timestamp_registry(
        release_registry,
        protocol=parsed_protocol,
        cohort_definition=cohort_definition,
    )
    return parsed_schema, parsed_protocol


def validate_prospective_lifecycle(
    cohort_definition: Mapping[str, object],
    issues: Sequence[Mapping[str, object]],
    mature_truth_records: Sequence[Mapping[str, object]],
    truth_revisions: Sequence[Mapping[str, object]],
    evaluations: Sequence[Mapping[str, object]],
    *,
    artifacts_by_sha256: Mapping[str, object | bytes],
    trusted_release_manifest: Mapping[str, object],
    trusted_release_manifest_file_sha256: str,
    schema: Mapping[str, object] | None = None,
    protocol: Mapping[str, object] | None = None,
) -> None:
    """Gate the production lifecycle at the Stage 2P-1A/1B boundary.

    Stage 2P-1A validates the caller-supplied tagged-release trust boundary and
    its exact bytes.  It deliberately cannot return production-ready: RFC 3161
    ASN.1/CMS verification, formal five-table byte reconstruction, forecast
    artifact replay, and contribution recomputation are Stage 2P-1B work.
    """

    cohort = _mapping(cohort_definition, path="cohort_definition")
    _sequence(issues, path="issues")
    _sequence(mature_truth_records, path="mature_truth_records")
    _sequence(truth_revisions, path="truth_revisions")
    _sequence(evaluations, path="evaluations")
    _mapping(artifacts_by_sha256, path="artifacts_by_sha256")
    _validate_trusted_release_boundary(
        cohort,
        artifacts_by_sha256=artifacts_by_sha256,
        schema=schema,
        protocol=protocol,
        trusted_release_manifest=trusted_release_manifest,
        trusted_release_manifest_file_sha256=(
            trusted_release_manifest_file_sha256
        ),
    )
    implementation_status = trusted_release_manifest.get("implementation_status")
    if implementation_status == "stage2p1b_synthetic_accepted":
        _fail(
            _STAGE2P1B_VALIDATOR_NOT_IMPLEMENTED,
            (
                "the trusted release is synthetically accepted, but the Stage "
                "2P-1B byte-reconstruction validator is not implemented"
            ),
            path="trusted_release_manifest.implementation_status",
        )
    _fail(
        LIFECYCLE_IMPLEMENTATION_STATUS,
        (
            "production lifecycle verification remains closed until Stage 2P-1B "
            "implements actual RFC 3161, formal-table, forecast, and evaluation "
            "artifact recomputation"
        ),
        path="trusted_release_manifest.implementation_status",
    )


def validate_record_against_schema(
    schema: Mapping[str, object],
    record: Mapping[str, object],
    protocol: Mapping[str, object] | None = None,
) -> None:
    """Apply Draft 2020-12, format, and Stage 2P semantic validation."""

    schema = _mapping(schema, path="schema")
    record = _mapping(record, path="$")
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema, format_checker=_FORMAT_CHECKER).validate(record)
    validate_record_semantics(
        record,
        _DEFAULT_CALENDAR if protocol is None else protocol,
    )


def validate_record_json_bytes(
    schema: Mapping[str, object],
    raw: bytes | bytearray | memoryview,
    protocol: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    """Strictly parse original bytes, then validate schema and production semantics."""

    record = parse_record_json_bytes(raw)
    if bytes(raw) != canonical_json_bytes(record):
        _fail(
            "record_json_bytes_not_canonical",
            "record bytes must equal the one exact canonical JSON encoding",
        )
    validate_record_against_schema(schema, record, protocol)
    return record
