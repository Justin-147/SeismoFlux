"""Target-blind production input primitives for the frozen P1 protocol.

This module deliberately stops at the scientific input boundary.  It computes the
weekly ``T``/``Q`` calendar, parses caller-supplied ComCat GeoJSON bytes, records
the exact HTTP/body identity, and removes repeated revisions across snapshots.
It contains no network client and cannot silently fetch data on import or call.

Real ComCat events have their own type.  They are never labelled as
``SyntheticEvent`` and therefore cannot accidentally enter the synthetic P1
acceptance path.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, time, timedelta
from typing import Final, Literal, Protocol, TypeAlias, cast
from urllib.parse import quote
from zoneinfo import ZoneInfo

from seismoflux.data.common import canonical_json_bytes
from seismoflux.p1_b0_r30.core import LOCAL_CATALOG_CUTOFF_UTC

COMCAT_SOURCE_ID: Final = "usgs_anss_comcat_fdsn_event_api_v1"
COMCAT_QUERY_HOST: Final = "earthquake.usgs.gov"
COMCAT_QUERY_PATH: Final = "/fdsnws/event/1/query"
COMCAT_QUERY_ENDPOINT: Final = f"https://{COMCAT_QUERY_HOST}{COMCAT_QUERY_PATH}"
COMCAT_COUNT_ENDPOINT: Final = f"https://{COMCAT_QUERY_HOST}/fdsnws/event/1/count"
COMCAT_RESPONSE_LIMIT: Final = 20_000
P1_TIMEZONE_NAME: Final = "Asia/Shanghai"
P1_TIMEZONE: Final = ZoneInfo(P1_TIMEZONE_NAME)
P1_FIRST_ISSUE_UTC: Final = datetime(2026, 9, 9, 16, 0, tzinfo=UTC)
P1_QUERY_LAG: Final = timedelta(minutes=15)
P1_ISSUE_WEEKDAY: Final = 3  # datetime.weekday(): Monday=0, Thursday=3

CapturedHeaders: TypeAlias = dict[str, str | None]

_CAPTURED_HEADER_NAMES: Final[tuple[tuple[str, str], ...]] = (
    ("date", "date"),
    ("etag", "etag"),
    ("last-modified", "last_modified"),
    ("content-type", "content_type"),
    ("content-length", "content_length"),
)

_COMMON_REQUEST_PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("minlongitude", "73.446961"),
    ("minlatitude", "20.22909"),
    ("maxlongitude", "135.08583"),
    ("maxlatitude", "53.557926"),
    ("minmagnitude", "3.9"),
    ("eventtype", "earthquake"),
    ("format", "geojson"),
)
_QUERY_ONLY_REQUEST_PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("orderby", "time-asc"),
    ("limit", "20000"),
    ("includeallorigins", "false"),
    ("includeallmagnitudes", "false"),
)
_COMMON_TRAILING_REQUEST_PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("includedeleted", "false"),
    ("jsonerror", "true"),
    ("nodata", "204"),
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _utc(value: datetime, *, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware")
    return value.astimezone(UTC)


def _utc_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_utc_text(value: str, *, label: str) -> datetime:
    if not value.endswith("Z"):
        raise ValueError(f"{label} must end in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{label} must be a valid ISO-8601 UTC timestamp") from exc
    return parsed.astimezone(UTC)


def _canonical_url(endpoint: str, pairs: Sequence[tuple[str, str]]) -> str:
    encoded = "&".join(
        f"{quote(name, safe='-._~')}={quote(value, safe='-._~')}" for name, value in pairs
    )
    return f"{endpoint}?{encoded}"


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _epoch_milliseconds(value: object, *, label: str) -> datetime:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer Unix millisecond timestamp")
    try:
        return datetime(1970, 1, 1, tzinfo=UTC) + timedelta(milliseconds=value)
    except OverflowError as exc:
        raise ValueError(f"{label} is outside the supported datetime range") from exc


@dataclass(frozen=True, slots=True)
class P1IssueSchedule:
    """One legal weekly issue time and its causally prior catalogue cutoff."""

    issue_id: str
    scheduled_issue_time_utc: datetime
    query_cutoff_utc: datetime

    def __post_init__(self) -> None:
        scheduled = _utc(self.scheduled_issue_time_utc, label="scheduled_issue_time_utc")
        cutoff = _utc(self.query_cutoff_utc, label="query_cutoff_utc")
        expected = issue_schedule(scheduled)
        if self.issue_id != expected.issue_id or cutoff != expected.query_cutoff_utc:
            raise ValueError("issue schedule fields do not match the frozen weekly calendar")
        object.__setattr__(self, "scheduled_issue_time_utc", scheduled)
        object.__setattr__(self, "query_cutoff_utc", cutoff)

    def as_mapping(self) -> dict[str, str]:
        return {
            "issue_id": self.issue_id,
            "scheduled_issue_time_utc": _utc_text(self.scheduled_issue_time_utc),
            "query_cutoff_utc": _utc_text(self.query_cutoff_utc),
        }


def _new_issue_schedule(scheduled: datetime) -> P1IssueSchedule:
    """Construct a schedule after calendar validation without recursive validation."""

    result = object.__new__(P1IssueSchedule)
    object.__setattr__(result, "issue_id", f"p1-{scheduled.strftime('%Y%m%dT%H%M%SZ')}")
    object.__setattr__(result, "scheduled_issue_time_utc", scheduled)
    object.__setattr__(result, "query_cutoff_utc", scheduled - P1_QUERY_LAG)
    return result


def issue_schedule(scheduled_issue_time_utc: datetime) -> P1IssueSchedule:
    """Validate one frozen weekly ``T`` and derive ``Q=T-15 minutes``."""

    scheduled = _utc(scheduled_issue_time_utc, label="scheduled_issue_time_utc")
    local = scheduled.astimezone(P1_TIMEZONE)
    if scheduled < P1_FIRST_ISSUE_UTC:
        raise ValueError("scheduled issue precedes the frozen P1 valid-from time")
    if (
        local.weekday() != P1_ISSUE_WEEKDAY
        or local.hour != 0
        or local.minute != 0
        or local.second != 0
        or local.microsecond != 0
    ):
        raise ValueError("scheduled issue is not Thursday 00:00:00 Asia/Shanghai")
    return _new_issue_schedule(scheduled)


def next_issue_schedule(after_utc: datetime) -> P1IssueSchedule:
    """Return the first rule issue strictly after ``after_utc``; never backfill."""

    after = _utc(after_utc, label="after_utc")
    if after < P1_FIRST_ISSUE_UTC:
        return _new_issue_schedule(P1_FIRST_ISSUE_UTC)

    local_after = after.astimezone(P1_TIMEZONE)
    days_ahead = (P1_ISSUE_WEEKDAY - local_after.weekday()) % 7
    candidate_date = local_after.date() + timedelta(days=days_ahead)
    candidate_local = datetime.combine(candidate_date, time.min, tzinfo=P1_TIMEZONE)
    candidate = candidate_local.astimezone(UTC)
    if candidate <= after:
        candidate = (candidate_local + timedelta(days=7)).astimezone(UTC)
    return issue_schedule(candidate)


@dataclass(frozen=True, slots=True)
class ComCatEvent:
    """One explicitly real ComCat revision observed in a sealed response."""

    event_id: str
    associated_ids: tuple[str, ...]
    origin_time_utc: datetime
    provider_updated_at_utc: datetime
    first_seen_at_utc: datetime
    observed_at_utc: datetime
    longitude: float
    latitude: float
    depth_km: float
    magnitude: float
    feature_canonical_sha256: str
    source_id: Literal["usgs_anss_comcat_fdsn_event_api_v1"] = COMCAT_SOURCE_ID

    def __post_init__(self) -> None:
        if not self.event_id or self.event_id.strip() != self.event_id:
            raise ValueError("event_id must be a non-empty stripped string")
        if self.source_id != COMCAT_SOURCE_ID:
            raise ValueError("production events must retain the real ComCat source identity")
        identifiers = tuple(
            sorted(set(self.associated_ids) | {self.event_id}, key=lambda item: item.encode())
        )
        if any(not item or item.strip() != item for item in identifiers):
            raise ValueError("associated_ids must contain non-empty stripped identifiers")
        object.__setattr__(self, "associated_ids", identifiers)

        origin = _utc(self.origin_time_utc, label="origin_time_utc")
        updated = _utc(self.provider_updated_at_utc, label="provider_updated_at_utc")
        first_seen = _utc(self.first_seen_at_utc, label="first_seen_at_utc")
        observed = _utc(self.observed_at_utc, label="observed_at_utc")
        if updated < origin:
            raise ValueError("provider_updated_at_utc cannot precede origin_time_utc")
        if first_seen < origin:
            raise ValueError("first_seen_at_utc cannot precede origin_time_utc")
        if observed < first_seen:
            raise ValueError("observed_at_utc cannot precede first_seen_at_utc")
        if updated > observed:
            raise ValueError("a response cannot contain a provider revision from the future")
        object.__setattr__(self, "origin_time_utc", origin)
        object.__setattr__(self, "provider_updated_at_utc", updated)
        object.__setattr__(self, "first_seen_at_utc", first_seen)
        object.__setattr__(self, "observed_at_utc", observed)

        longitude = _finite_number(self.longitude, label="longitude")
        latitude = _finite_number(self.latitude, label="latitude")
        depth = _finite_number(self.depth_km, label="depth_km")
        magnitude = _finite_number(self.magnitude, label="magnitude")
        if not -180.0 <= longitude <= 180.0 or not -90.0 <= latitude <= 90.0:
            raise ValueError("longitude/latitude are outside WGS84 bounds")
        object.__setattr__(self, "longitude", longitude)
        object.__setattr__(self, "latitude", latitude)
        object.__setattr__(self, "depth_km", depth)
        object.__setattr__(self, "magnitude", magnitude)
        if not _is_sha256(self.feature_canonical_sha256):
            raise ValueError("feature_canonical_sha256 must be a lowercase SHA-256 digest")

    @property
    def component_sha256(self) -> str:
        """Stable identity of the connected identifier component."""

        return _sha256(canonical_json_bytes(list(self.associated_ids)))

    def as_mapping(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "associated_ids": list(self.associated_ids),
            "origin_time_utc": _utc_text(self.origin_time_utc),
            "provider_updated_at_utc": _utc_text(self.provider_updated_at_utc),
            "first_seen_at_utc": _utc_text(self.first_seen_at_utc),
            "observed_at_utc": _utc_text(self.observed_at_utc),
            "longitude": self.longitude,
            "latitude": self.latitude,
            "depth_km": self.depth_km,
            "magnitude": self.magnitude,
            "feature_canonical_sha256": self.feature_canonical_sha256,
            "component_sha256": self.component_sha256,
            "source_id": self.source_id,
        }


@dataclass(frozen=True, slots=True)
class ParsedComCatCatalogue:
    """Local causal filtering result for exact supplied GeoJSON bytes."""

    feature_count: int
    before_start_excluded_count: int
    after_end_excluded_count: int
    unavailable_at_Q_excluded_count: int
    events: tuple[ComCatEvent, ...]

    def __post_init__(self) -> None:
        if (
            min(
                self.feature_count,
                self.before_start_excluded_count,
                self.after_end_excluded_count,
                self.unavailable_at_Q_excluded_count,
            )
            < 0
        ):
            raise ValueError("catalogue counts must be non-negative")
        if (
            self.before_start_excluded_count
            + self.after_end_excluded_count
            + self.unavailable_at_Q_excluded_count
            + len(self.events)
            != self.feature_count
        ):
            raise ValueError("catalogue filter counts do not reconcile")


def _associated_ids(feature_id: str, properties: Mapping[str, object]) -> tuple[str, ...]:
    value = properties.get("ids")
    if value is None:
        identifiers = {feature_id}
    elif isinstance(value, str):
        identifiers = {feature_id, *(item.strip() for item in value.split(",") if item.strip())}
    else:
        raise ValueError("feature properties.ids must be a comma-separated string or null")
    return tuple(sorted(identifiers, key=lambda item: item.encode()))


def _parse_feature(feature: object, *, observed_at_utc: datetime) -> ComCatEvent:
    if not isinstance(feature, Mapping):
        raise ValueError("every GeoJSON feature must be an object")
    feature_mapping = cast(Mapping[str, object], feature)
    if feature_mapping.get("type") != "Feature":
        raise ValueError("every catalogue row must be a GeoJSON Feature")
    feature_id = feature_mapping.get("id")
    if not isinstance(feature_id, str) or not feature_id or feature_id.strip() != feature_id:
        raise ValueError("every ComCat feature requires a stable non-empty id")
    properties_value = feature_mapping.get("properties")
    geometry_value = feature_mapping.get("geometry")
    if not isinstance(properties_value, Mapping):
        raise ValueError("feature properties must be an object")
    if not isinstance(geometry_value, Mapping) or geometry_value.get("type") != "Point":
        raise ValueError("feature geometry must be a GeoJSON Point")
    properties = cast(Mapping[str, object], properties_value)
    if properties.get("type") != "earthquake":
        raise ValueError("every ComCat catalogue row must have properties.type=earthquake")
    geometry = cast(Mapping[str, object], geometry_value)
    coordinates_value = geometry.get("coordinates")
    if (
        not isinstance(coordinates_value, Sequence)
        or isinstance(coordinates_value, str | bytes | bytearray)
        or len(coordinates_value) < 3
    ):
        raise ValueError("Point coordinates must contain longitude, latitude, and depth")
    coordinates = cast(Sequence[object], coordinates_value)
    observed = _utc(observed_at_utc, label="observed_at_utc")
    return ComCatEvent(
        event_id=feature_id,
        associated_ids=_associated_ids(feature_id, properties),
        origin_time_utc=_epoch_milliseconds(properties.get("time"), label="properties.time"),
        provider_updated_at_utc=_epoch_milliseconds(
            properties.get("updated"), label="properties.updated"
        ),
        first_seen_at_utc=observed,
        observed_at_utc=observed,
        longitude=_finite_number(coordinates[0], label="coordinates[0]"),
        latitude=_finite_number(coordinates[1], label="coordinates[1]"),
        depth_km=_finite_number(coordinates[2], label="coordinates[2]"),
        magnitude=_finite_number(properties.get("mag"), label="properties.mag"),
        feature_canonical_sha256=_sha256(canonical_json_bytes(dict(feature_mapping))),
    )


def parse_comcat_geojson(
    raw_response_bytes: bytes,
    *,
    observed_at_utc: datetime,
    origin_start_exclusive_utc: datetime = LOCAL_CATALOG_CUTOFF_UTC,
    origin_end_inclusive_utc: datetime,
) -> ParsedComCatCatalogue:
    """Parse exact bytes and apply the local ``(start, Q]`` origin-time rule.

    Excluded rows remain recoverable through the caller's immutable raw bytes and
    are counted explicitly.  Provider ``updated`` is the conservative available-at
    field: a revision stamped after Q cannot enter that Q's model.
    """

    if not isinstance(raw_response_bytes, bytes) or not raw_response_bytes:
        raise ValueError("HTTP 200 GeoJSON must be non-empty exact bytes")
    observed = _utc(observed_at_utc, label="observed_at_utc")
    start = _utc(origin_start_exclusive_utc, label="origin_start_exclusive_utc")
    end = _utc(origin_end_inclusive_utc, label="origin_end_inclusive_utc")
    if end <= start:
        raise ValueError("origin_end_inclusive_utc must be after the source cutover")
    try:
        document = json.loads(raw_response_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("ComCat response must be valid UTF-8 JSON") from exc
    if not isinstance(document, Mapping) or document.get("type") != "FeatureCollection":
        raise ValueError("ComCat response must be a GeoJSON FeatureCollection")
    features_value = document.get("features")
    if not isinstance(features_value, list):
        raise ValueError("GeoJSON FeatureCollection.features must be a list")
    metadata_value = document.get("metadata")
    if metadata_value is not None:
        if not isinstance(metadata_value, Mapping):
            raise ValueError("GeoJSON metadata must be an object when present")
        declared_count = metadata_value.get("count")
        if declared_count is not None and (
            isinstance(declared_count, bool)
            or not isinstance(declared_count, int)
            or declared_count != len(features_value)
        ):
            raise ValueError("GeoJSON metadata.count must equal the exact feature count")

    retained: list[ComCatEvent] = []
    before_start = 0
    after_end = 0
    unavailable_at_q = 0
    for feature in features_value:
        event = _parse_feature(feature, observed_at_utc=observed)
        if event.origin_time_utc <= start:
            before_start += 1
        elif event.origin_time_utc > end:
            after_end += 1
        elif event.provider_updated_at_utc > end:
            unavailable_at_q += 1
        else:
            retained.append(event)
    retained.sort(key=lambda event: (event.origin_time_utc, event.event_id.encode()))
    return ParsedComCatCatalogue(
        feature_count=len(features_value),
        before_start_excluded_count=before_start,
        after_end_excluded_count=after_end,
        unavailable_at_Q_excluded_count=unavailable_at_q,
        events=tuple(retained),
    )


def deduplicate_comcat_revisions(events: Iterable[ComCatEvent]) -> tuple[ComCatEvent, ...]:
    """Deduplicate repeated queries by connected associated-ID components.

    The selected revision is the greatest provider ``updated`` time; exact ties
    use the smallest canonical feature SHA-256.  The component's earliest sealed
    observation is retained as ``first_seen_at_utc`` and is never backdated to the
    earthquake origin or provider update.
    """

    rows = tuple(events)
    if any(event.source_id != COMCAT_SOURCE_ID for event in rows):
        raise ValueError("cross-query deduplication accepts only real ComCat events")
    if not rows:
        return ()

    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            if left_root < right_root:
                parent[right_root] = left_root
            else:
                parent[left_root] = right_root

    identifier_owner: dict[str, int] = {}
    for index, event in enumerate(rows):
        for identifier in event.associated_ids:
            previous = identifier_owner.setdefault(identifier, index)
            union(index, previous)

    components: dict[int, list[ComCatEvent]] = {}
    for index, event in enumerate(rows):
        components.setdefault(find(index), []).append(event)

    selected: list[ComCatEvent] = []
    for component in components.values():
        all_ids = tuple(
            sorted(
                {identifier for event in component for identifier in event.associated_ids},
                key=lambda item: item.encode(),
            )
        )
        greatest_updated = max(event.provider_updated_at_utc for event in component)
        newest = [event for event in component if event.provider_updated_at_utc == greatest_updated]
        winner = min(
            newest,
            key=lambda event: (
                event.feature_canonical_sha256,
                event.event_id.encode(),
                event.observed_at_utc,
            ),
        )
        selected.append(
            replace(
                winner,
                associated_ids=all_ids,
                first_seen_at_utc=min(event.first_seen_at_utc for event in component),
            )
        )
    selected.sort(key=lambda event: (event.origin_time_utc, event.event_id.encode()))
    return tuple(selected)


def capture_response_headers(headers: Mapping[str, str]) -> CapturedHeaders:
    """Capture only the five frozen response headers, case-insensitively."""

    normalized: dict[str, str] = {}
    for name, value in headers.items():
        if not isinstance(name, str) or not isinstance(value, str):
            raise ValueError("HTTP response headers must be string pairs")
        lower = name.strip().lower()
        if lower in normalized:
            raise ValueError(f"duplicate case-insensitive HTTP response header: {lower}")
        normalized[lower] = value.strip()
    return {target: normalized.get(source) for source, target in _CAPTURED_HEADER_NAMES}


def build_issue_query_url(schedule: P1IssueSchedule) -> str:
    """Build the single canonical M3.9 ComCat issue-input query URL."""

    pairs = (
        ("starttime", _utc_text(LOCAL_CATALOG_CUTOFF_UTC)),
        ("endtime", _utc_text(schedule.query_cutoff_utc)),
        *_COMMON_REQUEST_PAIRS,
        *_QUERY_ONLY_REQUEST_PAIRS,
        ("includedeleted", "false"),
        ("offset", "1"),
        ("jsonerror", "true"),
        ("nodata", "204"),
    )
    return _canonical_url(COMCAT_QUERY_ENDPOINT, pairs)


def build_issue_count_url(schedule: P1IssueSchedule) -> str:
    """Build the matching operational count-preflight URL.

    ``format=geojson`` and the ``>=20,000`` fail-closed threshold are conservative
    acquisition safeguards.  They do not alter the frozen P1 model or endpoint.
    """

    pairs = (
        ("starttime", _utc_text(LOCAL_CATALOG_CUTOFF_UTC)),
        ("endtime", _utc_text(schedule.query_cutoff_utc)),
        *_COMMON_REQUEST_PAIRS,
        *_COMMON_TRAILING_REQUEST_PAIRS,
    )
    return _canonical_url(COMCAT_COUNT_ENDPOINT, pairs)


def validate_issue_query_url(request_url: str, schedule: P1IssueSchedule) -> None:
    """Require the exact canonical endpoint, parameter set, order, and encoding."""

    if request_url != build_issue_query_url(schedule):
        raise ValueError("request_url differs from the complete canonical ComCat query")


def validate_issue_count_url(request_url: str, schedule: P1IssueSchedule) -> None:
    """Require the exact matching count-preflight request identity."""

    if request_url != build_issue_count_url(schedule):
        raise ValueError("request_url differs from the complete canonical ComCat count query")


@dataclass(frozen=True, slots=True)
class ComCatHttpExchange:
    """Injected transport result; no concrete network implementation is provided."""

    request_url: str
    fetch_started_at_utc: datetime
    fetch_completed_at_utc: datetime
    http_status: int
    response_headers: Mapping[str, str]
    raw_response_bytes: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.raw_response_bytes, bytes):
            raise ValueError("raw_response_bytes must preserve the exact bytes")
        started = _utc(self.fetch_started_at_utc, label="fetch_started_at_utc")
        completed = _utc(self.fetch_completed_at_utc, label="fetch_completed_at_utc")
        if completed < started:
            raise ValueError("fetch completion cannot precede fetch start")
        if isinstance(self.http_status, bool) or not isinstance(self.http_status, int):
            raise ValueError("http_status must be an integer")
        object.__setattr__(self, "fetch_started_at_utc", started)
        object.__setattr__(self, "fetch_completed_at_utc", completed)


class ComCatTransport(Protocol):
    """Dependency-injected transport boundary used by an external issue runner."""

    def __call__(self, request_url: str) -> ComCatHttpExchange: ...


def _validate_success_exchange(
    exchange: ComCatHttpExchange,
    *,
    schedule: P1IssueSchedule,
) -> CapturedHeaders:
    if exchange.fetch_started_at_utc < schedule.query_cutoff_utc:
        raise ValueError("real issue acquisition must not start before Q")
    if exchange.fetch_completed_at_utc >= schedule.scheduled_issue_time_utc:
        raise ValueError("real issue acquisition must complete strictly before T")
    if exchange.http_status not in {200, 204}:
        raise ValueError("only successful ComCat HTTP 200/204 responses can be sealed")
    captured = capture_response_headers(exchange.response_headers)
    content_length = captured["content_length"]
    if content_length is not None and (
        not content_length.isdecimal() or int(content_length) != len(exchange.raw_response_bytes)
    ):
        raise ValueError("captured Content-Length does not match the exact response bytes")
    if exchange.http_status == 204 and exchange.raw_response_bytes:
        raise ValueError("HTTP 204 must preserve the exact empty byte sequence")
    return captured


def parse_comcat_count_geojson(raw_response_bytes: bytes) -> int:
    """Parse the count endpoint's GeoJSON-format JSON object without coercion."""

    if not isinstance(raw_response_bytes, bytes) or not raw_response_bytes:
        raise ValueError("HTTP 200 count response must be non-empty exact bytes")
    try:
        value = json.loads(raw_response_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("ComCat count response must be valid UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("ComCat count response must be a JSON object")
    count = value.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("ComCat count field must be a non-negative JSON integer")
    return count


@dataclass(frozen=True, slots=True)
class ComCatCountSnapshot:
    """Exact-byte identity and parsed integer from the count preflight."""

    source_id: Literal["usgs_anss_comcat_fdsn_event_api_v1"]
    request_url: str
    request_url_utf8_sha256: str
    fetch_started_at_utc: datetime
    fetch_completed_at_utc: datetime
    http_status: int
    captured_response_headers: CapturedHeaders
    response_headers_sha256: str
    raw_response_sha256: str
    response_body_byte_count: int
    parsed_count: int
    snapshot_sha256: str
    raw_response_bytes: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if self.source_id != COMCAT_SOURCE_ID:
            raise ValueError("count snapshot source_id must be the frozen ComCat identity")
        started = _utc(self.fetch_started_at_utc, label="fetch_started_at_utc")
        completed = _utc(self.fetch_completed_at_utc, label="fetch_completed_at_utc")
        object.__setattr__(self, "fetch_started_at_utc", started)
        object.__setattr__(self, "fetch_completed_at_utc", completed)
        if self.request_url_utf8_sha256 != _sha256(self.request_url.encode("utf-8")):
            raise ValueError("request_url_utf8_sha256 does not match request_url")
        if self.response_headers_sha256 != _sha256(
            canonical_json_bytes(self.captured_response_headers)
        ):
            raise ValueError("response_headers_sha256 does not match captured headers")
        if self.response_body_byte_count != len(self.raw_response_bytes):
            raise ValueError("response_body_byte_count does not match the exact body")
        if self.raw_response_sha256 != _sha256(self.raw_response_bytes):
            raise ValueError("raw_response_sha256 does not match the exact body")
        if isinstance(self.parsed_count, bool) or not isinstance(self.parsed_count, int):
            raise ValueError("parsed_count must be an integer")
        if self.parsed_count < 0:
            raise ValueError("parsed_count must be non-negative")
        if self.snapshot_sha256 != _sha256(canonical_json_bytes(self._identity_mapping())):
            raise ValueError("snapshot_sha256 does not match the count snapshot identity")

    def _identity_mapping(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "request_url": self.request_url,
            "request_url_utf8_sha256": self.request_url_utf8_sha256,
            "fetch_started_at_utc": _utc_text(self.fetch_started_at_utc),
            "fetch_completed_at_utc": _utc_text(self.fetch_completed_at_utc),
            "http_status": self.http_status,
            "captured_response_headers": self.captured_response_headers,
            "response_headers_sha256": self.response_headers_sha256,
            "raw_response_sha256": self.raw_response_sha256,
            "response_body_byte_count": self.response_body_byte_count,
            "parsed_count": self.parsed_count,
        }

    def as_mapping(self) -> dict[str, object]:
        result = self._identity_mapping()
        result["snapshot_sha256"] = self.snapshot_sha256
        return result


def build_comcat_count_snapshot(
    exchange: ComCatHttpExchange,
    *,
    schedule: P1IssueSchedule,
) -> ComCatCountSnapshot:
    """Validate and seal one injected count-preflight response."""

    validate_issue_count_url(exchange.request_url, schedule)
    captured = _validate_success_exchange(exchange, schedule=schedule)
    content_type = captured["content_type"]
    if exchange.http_status == 204:
        parsed_count = 0
    else:
        if content_type is None or content_type.split(";", 1)[0].strip().lower() not in {
            "application/json",
            "application/geo+json",
        }:
            raise ValueError("HTTP 200 count response must declare a JSON content type")
        parsed_count = parse_comcat_count_geojson(exchange.raw_response_bytes)
    request_sha = _sha256(exchange.request_url.encode("utf-8"))
    headers_sha = _sha256(canonical_json_bytes(captured))
    raw_sha = _sha256(exchange.raw_response_bytes)
    identity: dict[str, object] = {
        "source_id": COMCAT_SOURCE_ID,
        "request_url": exchange.request_url,
        "request_url_utf8_sha256": request_sha,
        "fetch_started_at_utc": _utc_text(exchange.fetch_started_at_utc),
        "fetch_completed_at_utc": _utc_text(exchange.fetch_completed_at_utc),
        "http_status": exchange.http_status,
        "captured_response_headers": captured,
        "response_headers_sha256": headers_sha,
        "raw_response_sha256": raw_sha,
        "response_body_byte_count": len(exchange.raw_response_bytes),
        "parsed_count": parsed_count,
    }
    return ComCatCountSnapshot(
        source_id=COMCAT_SOURCE_ID,
        request_url=exchange.request_url,
        request_url_utf8_sha256=request_sha,
        fetch_started_at_utc=exchange.fetch_started_at_utc,
        fetch_completed_at_utc=exchange.fetch_completed_at_utc,
        http_status=exchange.http_status,
        captured_response_headers=captured,
        response_headers_sha256=headers_sha,
        raw_response_sha256=raw_sha,
        response_body_byte_count=len(exchange.raw_response_bytes),
        parsed_count=parsed_count,
        snapshot_sha256=_sha256(canonical_json_bytes(identity)),
        raw_response_bytes=exchange.raw_response_bytes,
    )


@dataclass(frozen=True, slots=True)
class RawComCatSnapshot:
    """Content-addressed, exact-byte identity of one causal issue input response."""

    source_id: Literal["usgs_anss_comcat_fdsn_event_api_v1"]
    request_url: str
    request_url_utf8_sha256: str
    query_start_exclusive_utc: datetime
    query_end_inclusive_utc: datetime
    fetch_started_at_utc: datetime
    fetch_completed_at_utc: datetime
    http_status: int
    captured_response_headers: CapturedHeaders
    response_headers_sha256: str
    raw_response_sha256: str
    response_body_byte_count: int
    feature_count: int
    before_start_excluded_count: int
    after_end_excluded_count: int
    unavailable_at_Q_excluded_count: int
    deduplicated_event_count: int
    event_preimage_sha256: str
    snapshot_sha256: str
    events: tuple[ComCatEvent, ...]
    raw_response_bytes: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if self.source_id != COMCAT_SOURCE_ID:
            raise ValueError("snapshot source_id must be the frozen ComCat identity")
        start = _utc(self.query_start_exclusive_utc, label="query_start_exclusive_utc")
        end = _utc(self.query_end_inclusive_utc, label="query_end_inclusive_utc")
        fetch_start = _utc(self.fetch_started_at_utc, label="fetch_started_at_utc")
        fetch_end = _utc(self.fetch_completed_at_utc, label="fetch_completed_at_utc")
        object.__setattr__(self, "query_start_exclusive_utc", start)
        object.__setattr__(self, "query_end_inclusive_utc", end)
        object.__setattr__(self, "fetch_started_at_utc", fetch_start)
        object.__setattr__(self, "fetch_completed_at_utc", fetch_end)
        if not isinstance(self.raw_response_bytes, bytes):
            raise ValueError("raw_response_bytes must be exact bytes")
        if self.request_url_utf8_sha256 != _sha256(self.request_url.encode("utf-8")):
            raise ValueError("request_url_utf8_sha256 does not match request_url")
        if self.response_body_byte_count != len(self.raw_response_bytes):
            raise ValueError("response_body_byte_count does not match the exact body")
        if self.raw_response_sha256 != _sha256(self.raw_response_bytes):
            raise ValueError("raw_response_sha256 does not match the exact body")
        if self.response_headers_sha256 != _sha256(
            canonical_json_bytes(self.captured_response_headers)
        ):
            raise ValueError("response_headers_sha256 does not match captured headers")
        event_preimage = [event.as_mapping() for event in self.events]
        if self.event_preimage_sha256 != _sha256(canonical_json_bytes(event_preimage)):
            raise ValueError("event_preimage_sha256 does not match the deduplicated events")
        if self.deduplicated_event_count != len(self.events):
            raise ValueError("deduplicated_event_count does not match events")
        if self.snapshot_sha256 != _sha256(canonical_json_bytes(self._identity_mapping())):
            raise ValueError("snapshot_sha256 does not match the snapshot identity")

    def _identity_mapping(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "request_url": self.request_url,
            "request_url_utf8_sha256": self.request_url_utf8_sha256,
            "query_start_exclusive_utc": _utc_text(self.query_start_exclusive_utc),
            "query_end_inclusive_utc": _utc_text(self.query_end_inclusive_utc),
            "fetch_started_at_utc": _utc_text(self.fetch_started_at_utc),
            "fetch_completed_at_utc": _utc_text(self.fetch_completed_at_utc),
            "http_status": self.http_status,
            "captured_response_headers": self.captured_response_headers,
            "response_headers_sha256": self.response_headers_sha256,
            "raw_response_sha256": self.raw_response_sha256,
            "response_body_byte_count": self.response_body_byte_count,
            "feature_count": self.feature_count,
            "before_start_excluded_count": self.before_start_excluded_count,
            "after_end_excluded_count": self.after_end_excluded_count,
            "unavailable_at_Q_excluded_count": self.unavailable_at_Q_excluded_count,
            "deduplicated_event_count": self.deduplicated_event_count,
            "event_preimage_sha256": self.event_preimage_sha256,
        }

    def as_mapping(self) -> dict[str, object]:
        result = self._identity_mapping()
        result["snapshot_sha256"] = self.snapshot_sha256
        result["events"] = [event.as_mapping() for event in self.events]
        return result


def build_comcat_snapshot(
    exchange: ComCatHttpExchange,
    *,
    schedule: P1IssueSchedule,
    origin_start_exclusive_utc: datetime = LOCAL_CATALOG_CUTOFF_UTC,
) -> RawComCatSnapshot:
    """Validate an injected exchange and bind it to one on-time P1 issue."""

    validate_issue_query_url(exchange.request_url, schedule)
    start = _utc(origin_start_exclusive_utc, label="origin_start_exclusive_utc")
    captured = _validate_success_exchange(exchange, schedule=schedule)
    content_type = captured["content_type"]
    if exchange.http_status == 204:
        parsed = ParsedComCatCatalogue(0, 0, 0, 0, ())
    else:
        if content_type is None or content_type.split(";", 1)[0].strip().lower() not in {
            "application/json",
            "application/geo+json",
        }:
            raise ValueError("HTTP 200 ComCat response must declare a JSON content type")
        parsed = parse_comcat_geojson(
            exchange.raw_response_bytes,
            observed_at_utc=exchange.fetch_completed_at_utc,
            origin_start_exclusive_utc=start,
            origin_end_inclusive_utc=schedule.query_cutoff_utc,
        )
    events = deduplicate_comcat_revisions(parsed.events)
    request_sha = _sha256(exchange.request_url.encode("utf-8"))
    headers_sha = _sha256(canonical_json_bytes(captured))
    raw_sha = _sha256(exchange.raw_response_bytes)
    event_sha = _sha256(canonical_json_bytes([event.as_mapping() for event in events]))
    identity: dict[str, object] = {
        "source_id": COMCAT_SOURCE_ID,
        "request_url": exchange.request_url,
        "request_url_utf8_sha256": request_sha,
        "query_start_exclusive_utc": _utc_text(start),
        "query_end_inclusive_utc": _utc_text(schedule.query_cutoff_utc),
        "fetch_started_at_utc": _utc_text(exchange.fetch_started_at_utc),
        "fetch_completed_at_utc": _utc_text(exchange.fetch_completed_at_utc),
        "http_status": exchange.http_status,
        "captured_response_headers": captured,
        "response_headers_sha256": headers_sha,
        "raw_response_sha256": raw_sha,
        "response_body_byte_count": len(exchange.raw_response_bytes),
        "feature_count": parsed.feature_count,
        "before_start_excluded_count": parsed.before_start_excluded_count,
        "after_end_excluded_count": parsed.after_end_excluded_count,
        "unavailable_at_Q_excluded_count": parsed.unavailable_at_Q_excluded_count,
        "deduplicated_event_count": len(events),
        "event_preimage_sha256": event_sha,
    }
    return RawComCatSnapshot(
        source_id=COMCAT_SOURCE_ID,
        request_url=exchange.request_url,
        request_url_utf8_sha256=request_sha,
        query_start_exclusive_utc=start,
        query_end_inclusive_utc=schedule.query_cutoff_utc,
        fetch_started_at_utc=exchange.fetch_started_at_utc,
        fetch_completed_at_utc=exchange.fetch_completed_at_utc,
        http_status=exchange.http_status,
        captured_response_headers=captured,
        response_headers_sha256=headers_sha,
        raw_response_sha256=raw_sha,
        response_body_byte_count=len(exchange.raw_response_bytes),
        feature_count=parsed.feature_count,
        before_start_excluded_count=parsed.before_start_excluded_count,
        after_end_excluded_count=parsed.after_end_excluded_count,
        unavailable_at_Q_excluded_count=parsed.unavailable_at_Q_excluded_count,
        deduplicated_event_count=len(events),
        event_preimage_sha256=event_sha,
        snapshot_sha256=_sha256(canonical_json_bytes(identity)),
        events=events,
        raw_response_bytes=exchange.raw_response_bytes,
    )


def acquire_comcat_snapshot(
    request_url: str,
    *,
    schedule: P1IssueSchedule,
    transport: ComCatTransport | Callable[[str], ComCatHttpExchange],
    origin_start_exclusive_utc: datetime = LOCAL_CATALOG_CUTOFF_UTC,
) -> RawComCatSnapshot:
    """Acquire through an explicit injected transport and build a sealed snapshot.

    There is intentionally no default transport: tests and rehearsals inject a
    fixture, while a separately authorized issue runner must supply the real
    transport only during the legal ``[Q,T)`` window.
    """

    validate_issue_query_url(request_url, schedule)
    exchange = transport(request_url)
    if exchange.request_url != request_url:
        raise ValueError("transport exchange request_url differs from the authorized request")
    return build_comcat_snapshot(
        exchange,
        schedule=schedule,
        origin_start_exclusive_utc=origin_start_exclusive_utc,
    )


@dataclass(frozen=True, slots=True)
class ComCatIssueInputAcquisition:
    """Count-first outcome; an unavailable limit result never contains a query."""

    status: Literal["available", "unavailable_count_limit"]
    count_snapshot: ComCatCountSnapshot
    query_snapshot: RawComCatSnapshot | None
    unavailable_reason: str | None

    def __post_init__(self) -> None:
        limit_reached = self.count_snapshot.parsed_count >= COMCAT_RESPONSE_LIMIT
        if self.status == "unavailable_count_limit":
            if not limit_reached or self.query_snapshot is not None:
                raise ValueError("count-limit outcome must stop before the catalogue query")
            if self.unavailable_reason != "count_gte_20000_query_forbidden":
                raise ValueError("count-limit outcome requires the frozen structured reason")
        elif self.status == "available":
            if limit_reached or self.query_snapshot is None or self.unavailable_reason is not None:
                raise ValueError("available outcome requires a below-limit matching query snapshot")
            if self.query_snapshot.feature_count != self.count_snapshot.parsed_count:
                raise ValueError("count preflight and query feature counts differ")
            if (
                self.query_snapshot.fetch_started_at_utc
                < self.count_snapshot.fetch_completed_at_utc
            ):
                raise ValueError("catalogue query started before count preflight completed")
        else:
            raise ValueError("unsupported issue input acquisition status")

    def as_mapping(self) -> dict[str, object]:
        return {
            "status": self.status,
            "count_snapshot": self.count_snapshot.as_mapping(),
            "query_snapshot": (
                None if self.query_snapshot is None else self.query_snapshot.as_mapping()
            ),
            "unavailable_reason": self.unavailable_reason,
        }


def acquire_issue_input(
    *,
    schedule: P1IssueSchedule,
    transport: ComCatTransport | Callable[[str], ComCatHttpExchange],
) -> ComCatIssueInputAcquisition:
    """Run count first, fail closed at the limit, otherwise acquire one query."""

    count_url = build_issue_count_url(schedule)
    count_exchange = transport(count_url)
    if count_exchange.request_url != count_url:
        raise ValueError("transport count exchange differs from the authorized request")
    count_snapshot = build_comcat_count_snapshot(count_exchange, schedule=schedule)
    if count_snapshot.parsed_count >= COMCAT_RESPONSE_LIMIT:
        return ComCatIssueInputAcquisition(
            status="unavailable_count_limit",
            count_snapshot=count_snapshot,
            query_snapshot=None,
            unavailable_reason="count_gte_20000_query_forbidden",
        )

    query_url = build_issue_query_url(schedule)
    query_exchange = transport(query_url)
    if query_exchange.request_url != query_url:
        raise ValueError("transport query exchange differs from the authorized request")
    if query_exchange.fetch_started_at_utc < count_exchange.fetch_completed_at_utc:
        raise ValueError("catalogue query must start after count preflight completion")
    query_snapshot = build_comcat_snapshot(query_exchange, schedule=schedule)
    return ComCatIssueInputAcquisition(
        status="available",
        count_snapshot=count_snapshot,
        query_snapshot=query_snapshot,
        unavailable_reason=None,
    )


__all__ = [
    "COMCAT_RESPONSE_LIMIT",
    "COMCAT_SOURCE_ID",
    "P1_FIRST_ISSUE_UTC",
    "ComCatCountSnapshot",
    "ComCatEvent",
    "ComCatHttpExchange",
    "ComCatIssueInputAcquisition",
    "ComCatTransport",
    "P1IssueSchedule",
    "ParsedComCatCatalogue",
    "RawComCatSnapshot",
    "acquire_comcat_snapshot",
    "acquire_issue_input",
    "build_comcat_count_snapshot",
    "build_comcat_snapshot",
    "build_issue_count_url",
    "build_issue_query_url",
    "capture_response_headers",
    "deduplicate_comcat_revisions",
    "issue_schedule",
    "next_issue_schedule",
    "parse_comcat_count_geojson",
    "parse_comcat_geojson",
    "validate_issue_count_url",
    "validate_issue_query_url",
]
