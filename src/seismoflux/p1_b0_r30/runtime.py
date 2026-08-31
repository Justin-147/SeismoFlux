"""Authorized, time-gated runtime boundary for one real P1 issue.

The frozen scientific model remains in :mod:`prospective`.  This module only
checks that the public record chain already contains a remotely closed
authorization for the next unrecorded weekly issue and provides a narrowly
scoped ComCat HTTP transport.  No network request can start outside ``[Q,T)``.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final, Protocol, cast
from urllib.parse import urlsplit

from seismoflux.p1_b0_r30.ledger import read_p1_ledger
from seismoflux.p1_b0_r30.production import (
    COMCAT_COUNT_ENDPOINT,
    COMCAT_QUERY_ENDPOINT,
    ComCatHttpExchange,
    P1IssueSchedule,
    build_issue_count_url,
    build_issue_query_url,
)

JsonRecord = dict[str, object]
Clock = Callable[[], datetime]
PreRequestGate = Callable[[], None]

P1_SCHEMA_RELATIVE_PATH: Final = Path("data/contracts/p1_prospective_records_v1.json")
MAX_COMCAT_RESPONSE_BYTES: Final = 128 * 1024 * 1024
_ALLOWED_ENDPOINTS: Final = frozenset({COMCAT_COUNT_ENDPOINT, COMCAT_QUERY_ENDPOINT})


def _utc(value: datetime, *, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
    return value.astimezone(UTC)


def _parse_utc(value: object, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{label} must be a UTC timestamp ending in Z")
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00").astimezone(UTC)
    except ValueError as exc:
        raise ValueError(f"{label} is not a valid UTC timestamp") from exc


def _git_sha(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase 40-character Git SHA")
    return value


def _load_schema(repository_root: Path) -> Mapping[str, object]:
    root = repository_root.resolve()
    path = (root / P1_SCHEMA_RELATIVE_PATH).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError("the fixed P1 record schema is missing or unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("the fixed P1 record schema is unreadable") from exc
    if not isinstance(value, Mapping):
        raise ValueError("the fixed P1 record schema must be a JSON object")
    return cast(Mapping[str, object], value)


@dataclass(frozen=True, slots=True)
class AuthorizedIssueContext:
    """Validated public authorization for the next legal issue only."""

    schedule: P1IssueSchedule
    protocol_definition: JsonRecord
    authorization_record: JsonRecord
    code_commit: str

    @property
    def protocol_definition_sha256(self) -> str:
        return cast(str, self.protocol_definition["content_sha256"])

    @property
    def authorization_record_sha256(self) -> str:
        return cast(str, self.authorization_record["content_sha256"])


def _load_authorized_issue_context_at(
    repository_root: Path,
    *,
    schedule: P1IssueSchedule,
    code_commit: str,
    now_utc: datetime,
) -> AuthorizedIssueContext:
    """Fail closed unless this is the authorized next issue during ``[Q,T)``."""

    now = _utc(now_utc, label="now_utc")
    if not schedule.query_cutoff_utc <= now < schedule.scheduled_issue_time_utc:
        raise ValueError("real P1 issue execution is allowed only during [Q,T)")
    expected_code_commit = _git_sha(code_commit, label="code_commit")
    schema = _load_schema(repository_root)
    records = read_p1_ledger(repository_root, schema=schema, require_exists=True)
    protocol = records[0]
    authorizations = tuple(
        record for record in records if record.get("record_type") == "RealIssueAuthorizationRecord"
    )
    if len(authorizations) != 1:
        raise ValueError("the public P1 chain must contain exactly one authorization record")
    authorization = authorizations[0]
    if authorization.get("real_issue_authorized") is not True:
        raise ValueError("the public P1 authorization is not active")
    if _git_sha(authorization.get("code_commit"), label="authorization.code_commit") != (
        expected_code_commit
    ):
        raise ValueError("runtime code commit differs from the public authorization")
    authorized_from = _parse_utc(
        authorization.get("authorized_from_scheduled_issue_utc"),
        label="authorized_from_scheduled_issue_utc",
    )
    if schedule.scheduled_issue_time_utc < authorized_from:
        raise ValueError("scheduled issue precedes the authorization effective issue")
    for label in ("recorded_at_utc", "remote_verified_at_utc"):
        if _parse_utc(authorization.get(label), label=f"authorization.{label}") > now:
            raise ValueError("authorization evidence cannot be future-dated at execution")

    scheduled_records = tuple(
        record
        for record in records
        if record.get("record_type") in {"ForecastIssueRecord", "MissedIssueRecord"}
    )
    if scheduled_records:
        last_scheduled = _parse_utc(
            scheduled_records[-1].get("scheduled_issue_time_utc"),
            label="last scheduled issue",
        )
        expected_schedule = last_scheduled + timedelta(days=7)
    else:
        expected_schedule = _parse_utc(protocol.get("valid_from_utc"), label="valid_from_utc")
    if schedule.scheduled_issue_time_utc != expected_schedule:
        raise ValueError("requested issue is not the next unrecorded frozen weekly issue")
    final_reviews = tuple(
        record
        for record in records
        if record.get("record_type") == "SequentialReviewRecord"
        and (
            record.get("review_trigger") in {"cluster_30", "time_36_months"}
            or record.get("decision") == "pause_scientific_integrity_failure"
        )
    )
    if final_reviews:
        raise ValueError("the P1 experiment has already reached a final review")
    return AuthorizedIssueContext(
        schedule=schedule,
        protocol_definition=dict(protocol),
        authorization_record=dict(authorization),
        code_commit=expected_code_commit,
    )


def load_authorized_issue_context(
    repository_root: Path,
    *,
    schedule: P1IssueSchedule,
    code_commit: str,
) -> AuthorizedIssueContext:
    """Validate authorization using the production host's actual UTC clock."""

    return _load_authorized_issue_context_at(
        repository_root,
        schedule=schedule,
        code_commit=code_commit,
        now_utc=datetime.now(UTC),
    )


@dataclass(frozen=True, slots=True)
class HttpFetchResult:
    """Exact response returned by the small injectable HTTP boundary."""

    final_url: str
    http_status: int
    headers: Mapping[str, str]
    body: bytes


class HttpFetcher(Protocol):
    def __call__(self, request_url: str, timeout_seconds: float) -> HttpFetchResult: ...


def _default_http_fetch(request_url: str, timeout_seconds: float) -> HttpFetchResult:
    request = urllib.request.Request(
        request_url,
        headers={
            "Accept": "application/geo+json, application/json",
            "Accept-Encoding": "identity",
            "User-Agent": "SeismoFlux-P1-v0.2.7 (scientific prospective forecast)",
        },
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        status = int(response.status)
        final_url = response.geturl()
        headers = {str(name): str(value) for name, value in response.headers.items()}
        body = response.read(MAX_COMCAT_RESPONSE_BYTES + 1)
    if len(body) > MAX_COMCAT_RESPONSE_BYTES:
        raise ValueError("ComCat response exceeds the fixed operational byte ceiling")
    return HttpFetchResult(final_url=final_url, http_status=status, headers=headers, body=body)


@dataclass(frozen=True, slots=True)
class _TimeGatedComCatTransport:
    """Concrete transport that rechecks schedule and endpoint for every request."""

    schedule: P1IssueSchedule
    _clock: Clock
    _fetcher: HttpFetcher
    _pre_request_gate: PreRequestGate
    timeout_seconds: float = 45.0

    def __post_init__(self) -> None:
        if not 0.0 < self.timeout_seconds <= 120.0:
            raise ValueError("timeout_seconds must be in (0,120]")

    def __call__(self, request_url: str) -> ComCatHttpExchange:
        if request_url not in {
            build_issue_count_url(self.schedule),
            build_issue_query_url(self.schedule),
        }:
            raise ValueError("request URL is not one of this issue's two canonical ComCat URLs")
        split = urlsplit(request_url)
        endpoint = f"{split.scheme}://{split.netloc}{split.path}"
        if (
            split.scheme != "https"
            or split.username is not None
            or split.password is not None
            or split.fragment
            or endpoint not in _ALLOWED_ENDPOINTS
        ):
            raise ValueError("request URL escapes the frozen HTTPS ComCat endpoints")
        started = _utc(self._clock(), label="transport start time")
        if not self.schedule.query_cutoff_utc <= started < self.schedule.scheduled_issue_time_utc:
            raise ValueError("ComCat request cannot start outside [Q,T)")
        self._pre_request_gate()
        fetch_allowed_at = _utc(self._clock(), label="post-gate transport time")
        if (
            not self.schedule.query_cutoff_utc
            <= fetch_allowed_at
            < (self.schedule.scheduled_issue_time_utc)
        ):
            raise ValueError("ComCat request gate did not finish inside [Q,T)")
        fetched = self._fetcher(request_url, self.timeout_seconds)
        completed = _utc(self._clock(), label="transport completion time")
        if fetched.final_url != request_url:
            raise ValueError("redirected ComCat responses are not accepted")
        if not self.schedule.query_cutoff_utc <= completed < self.schedule.scheduled_issue_time_utc:
            raise ValueError("ComCat request did not finish inside [Q,T)")
        if not isinstance(fetched.body, bytes):
            raise ValueError("HTTP fetcher must preserve exact response bytes")
        if len(fetched.body) > MAX_COMCAT_RESPONSE_BYTES:
            raise ValueError("ComCat response exceeds the fixed operational byte ceiling")
        return ComCatHttpExchange(
            request_url=request_url,
            fetch_started_at_utc=started,
            fetch_completed_at_utc=completed,
            http_status=fetched.http_status,
            response_headers=fetched.headers,
            raw_response_bytes=fetched.body,
        )


def _build_guarded_time_gated_comcat_transport(
    *,
    schedule: P1IssueSchedule,
    pre_request_gate: PreRequestGate,
    timeout_seconds: float = 45.0,
) -> _TimeGatedComCatTransport:
    """Build the real transport; the private caller must supply a remote gate."""

    return _TimeGatedComCatTransport(
        schedule=schedule,
        _clock=lambda: datetime.now(UTC),
        _fetcher=_default_http_fetch,
        _pre_request_gate=pre_request_gate,
        timeout_seconds=timeout_seconds,
    )


def _build_time_gated_comcat_transport_for_test(
    *,
    schedule: P1IssueSchedule,
    clock: Clock,
    fetcher: HttpFetcher,
    pre_request_gate: PreRequestGate,
    timeout_seconds: float = 45.0,
) -> _TimeGatedComCatTransport:
    """Build a fixture-only transport that can never use the real HTTP fetcher."""

    if fetcher is _default_http_fetch:
        raise ValueError("the test transport cannot use the real HTTP fetcher")
    return _TimeGatedComCatTransport(
        schedule=schedule,
        _clock=clock,
        _fetcher=fetcher,
        _pre_request_gate=pre_request_gate,
        timeout_seconds=timeout_seconds,
    )


__all__ = [
    "MAX_COMCAT_RESPONSE_BYTES",
    "AuthorizedIssueContext",
    "HttpFetchResult",
    "load_authorized_issue_context",
]
