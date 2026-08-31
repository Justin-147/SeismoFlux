from __future__ import annotations

import hashlib
import inspect
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from seismoflux.p1_b0_r30 import runtime as runtime_module
from seismoflux.p1_b0_r30.ledger import append_new_p1_record
from seismoflux.p1_b0_r30.production import (
    build_issue_count_url,
    build_issue_query_url,
    issue_schedule,
)
from seismoflux.p1_b0_r30.runtime import (
    HttpFetchResult,
    load_authorized_issue_context,
)

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_SOURCE = ROOT / "data" / "contracts" / "p1_prospective_records_v1.json"
T = datetime(2026, 9, 9, 16, 0, tzinfo=UTC)
Q = T - timedelta(minutes=15)
CODE_COMMIT = "a" * 40


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _schema() -> dict[str, object]:
    value = json.loads(SCHEMA_SOURCE.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _authorized_root(tmp_path: Path) -> Path:
    schema_target = tmp_path / "data" / "contracts" / SCHEMA_SOURCE.name
    schema_target.parent.mkdir(parents=True)
    schema_target.write_bytes(SCHEMA_SOURCE.read_bytes())
    protocol = append_new_p1_record(
        tmp_path,
        "ProtocolDefinition",
        recorded_at_utc="2026-08-31T02:00:00Z",
        fields={
            "protocol_id": "p1-b0-r30-prospective-v1",
            "protocol_tag": "v0.2.7-p1-b0-r30-protocol",
            "code_tag": "v0.2.7-p1-b0-r30-code",
            "valid_from_utc": "2026-09-09T16:00:00Z",
            "historical_catalog_cutoff_utc": "2026-07-09T04:25:56Z",
            "source_boundary_manifest_sha256": _sha("source"),
            "model_manifest_sha256": _sha("model"),
            "protocol_commit": "b" * 40,
            "real_issue_authorized": False,
        },
        schema=_schema(),
    )
    append_new_p1_record(
        tmp_path,
        "RealIssueAuthorizationRecord",
        recorded_at_utc="2026-08-31T03:00:00Z",
        fields={
            "protocol_definition_sha256": protocol["content_sha256"],
            "authorization_commit": "c" * 40,
            "code_commit": CODE_COMMIT,
            "remote_verified_at_utc": "2026-08-31T02:59:00Z",
            "authorized_from_scheduled_issue_utc": "2026-09-09T16:00:00Z",
            "real_issue_authorized": True,
        },
        schema=_schema(),
    )
    return tmp_path


def test_authorized_context_accepts_only_the_next_issue_during_q_t(tmp_path: Path) -> None:
    root = _authorized_root(tmp_path)
    context = runtime_module._load_authorized_issue_context_at(
        root,
        schedule=issue_schedule(T),
        code_commit=CODE_COMMIT,
        now_utc=Q + timedelta(seconds=1),
    )

    assert context.authorization_record["real_issue_authorized"] is True
    assert context.code_commit == CODE_COMMIT


def test_authorized_context_fails_before_q_and_on_wrong_code(tmp_path: Path) -> None:
    root = _authorized_root(tmp_path)
    with pytest.raises(ValueError, match=r"only during \[Q,T\)"):
        runtime_module._load_authorized_issue_context_at(
            root,
            schedule=issue_schedule(T),
            code_commit=CODE_COMMIT,
            now_utc=Q - timedelta(microseconds=1),
        )
    with pytest.raises(ValueError, match="differs from the public authorization"):
        runtime_module._load_authorized_issue_context_at(
            root,
            schedule=issue_schedule(T),
            code_commit="d" * 40,
            now_utc=Q,
        )


def test_transport_fetches_exact_canonical_url_only_inside_window() -> None:
    schedule = issue_schedule(T)
    times = iter(
        (
            Q + timedelta(seconds=1),
            Q + timedelta(seconds=2),
            Q + timedelta(seconds=3),
        )
    )
    payload = b'{"count":0}'

    def fetcher(request_url: str, timeout_seconds: float) -> HttpFetchResult:
        assert timeout_seconds == 45.0
        return HttpFetchResult(
            final_url=request_url,
            http_status=200,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(payload)),
            },
            body=payload,
        )

    transport = runtime_module._build_time_gated_comcat_transport_for_test(
        schedule=schedule,
        clock=lambda: next(times),
        fetcher=fetcher,
        pre_request_gate=lambda: None,
    )
    exchange = transport(build_issue_count_url(schedule))

    assert exchange.raw_response_bytes == payload
    assert exchange.fetch_started_at_utc == Q + timedelta(seconds=1)
    assert exchange.fetch_completed_at_utc == Q + timedelta(seconds=3)


def test_transport_rejects_outside_window_before_fetch() -> None:
    schedule = issue_schedule(T)
    called = False

    def fetcher(request_url: str, timeout_seconds: float) -> HttpFetchResult:
        nonlocal called
        called = True
        raise AssertionError((request_url, timeout_seconds))

    transport = runtime_module._build_time_gated_comcat_transport_for_test(
        schedule=schedule,
        clock=lambda: Q - timedelta(seconds=1),
        fetcher=fetcher,
        pre_request_gate=lambda: None,
    )
    with pytest.raises(ValueError, match=r"outside \[Q,T\)"):
        transport(build_issue_count_url(schedule))
    assert called is False


@pytest.mark.parametrize(
    ("review_trigger", "decision"),
    [
        ("cluster_30", "report_uncertain_at_final_review"),
        ("time_36_months", "stop_B0_R30_retain_B0"),
        ("cluster_10", "pause_scientific_integrity_failure"),
    ],
)
def test_authorized_context_rejects_every_frozen_terminal_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    review_trigger: str,
    decision: str,
) -> None:
    root = _authorized_root(tmp_path)
    records = list(runtime_module.read_p1_ledger(root, schema=_schema(), require_exists=True))
    records.append(
        {
            "record_type": "SequentialReviewRecord",
            "review_trigger": review_trigger,
            "decision": decision,
        }
    )
    monkeypatch.setattr(runtime_module, "read_p1_ledger", lambda *args, **kwargs: tuple(records))

    with pytest.raises(ValueError, match="already reached a final review"):
        runtime_module._load_authorized_issue_context_at(
            root,
            schedule=issue_schedule(T),
            code_commit=CODE_COMMIT,
            now_utc=Q,
        )


def test_each_transport_request_is_gated_and_second_gate_failure_prevents_query_fetch() -> None:
    schedule = issue_schedule(T)
    current = Q + timedelta(seconds=1)
    gate_calls = 0
    fetch_calls: list[str] = []

    def gate() -> None:
        nonlocal gate_calls
        gate_calls += 1
        if gate_calls == 2:
            raise ValueError("remote ledger changed")

    def fetcher(request_url: str, timeout_seconds: float) -> HttpFetchResult:
        fetch_calls.append(request_url)
        return HttpFetchResult(
            final_url=request_url,
            http_status=200,
            headers={},
            body=b"0" if request_url == build_issue_count_url(schedule) else b'{"features":[]}',
        )

    transport = runtime_module._build_time_gated_comcat_transport_for_test(
        schedule=schedule,
        clock=lambda: current,
        fetcher=fetcher,
        pre_request_gate=gate,
    )
    transport(build_issue_count_url(schedule))
    with pytest.raises(ValueError, match="remote ledger changed"):
        transport(build_issue_query_url(schedule))

    assert gate_calls == 2
    assert fetch_calls == [build_issue_count_url(schedule)]


def test_public_real_entrypoints_do_not_accept_injected_time_or_unguarded_http() -> None:
    context_parameters = inspect.signature(load_authorized_issue_context).parameters

    assert "now_utc" not in context_parameters
    assert "build_time_gated_comcat_transport" not in runtime_module.__all__
    assert not hasattr(runtime_module, "build_time_gated_comcat_transport")
    guarded_parameters = inspect.signature(
        runtime_module._build_guarded_time_gated_comcat_transport
    ).parameters
    assert "pre_request_gate" in guarded_parameters
    assert "clock" not in guarded_parameters
    assert "fetcher" not in guarded_parameters
