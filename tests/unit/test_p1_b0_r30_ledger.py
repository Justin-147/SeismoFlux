from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from seismoflux.data.common import canonical_json_bytes
from seismoflux.p1_b0_r30 import ledger as ledger_module
from seismoflux.p1_b0_r30.ledger import (
    P1_LEDGER_RELATIVE_PATH,
    P1LedgerBusyError,
    P1LedgerError,
    P1LedgerIntegrityError,
    append_new_p1_record,
    append_p1_record,
    build_next_p1_record,
    p1_ledger_lock_path,
    p1_ledger_path,
    read_p1_ledger,
)
from seismoflux.p1_b0_r30.records import seal_record

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "data" / "contracts" / "p1_prospective_records_v1.json"


def _schema() -> dict[str, object]:
    value = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _git_sha(label: str) -> str:
    return _sha(label)[:40]


def _protocol_fields() -> dict[str, object]:
    return {
        "protocol_id": "p1-b0-r30-prospective-v1",
        "protocol_tag": "v0.2.7-p1-b0-r30-protocol",
        "code_tag": "v0.2.7-p1-b0-r30-code",
        "valid_from_utc": "2026-09-09T16:00:00Z",
        "historical_catalog_cutoff_utc": "2026-07-09T04:25:56Z",
        "source_boundary_manifest_sha256": _sha("source-boundary"),
        "model_manifest_sha256": _sha("model-manifest"),
        "protocol_commit": _git_sha("protocol-commit"),
        "real_issue_authorized": False,
    }


def _authorization_fields(protocol: dict[str, Any]) -> dict[str, object]:
    return {
        "protocol_definition_sha256": protocol["content_sha256"],
        "authorization_commit": _git_sha("authorization-commit"),
        "code_commit": _git_sha("real-issue-code-commit"),
        "remote_verified_at_utc": "2026-08-31T02:59:00Z",
        "authorized_from_scheduled_issue_utc": "2026-09-09T16:00:00Z",
        "real_issue_authorized": True,
    }


def _append_protocol(root: Path) -> dict[str, Any]:
    return append_new_p1_record(
        root,
        "ProtocolDefinition",
        recorded_at_utc="2026-08-31T02:00:00Z",
        fields=_protocol_fields(),
        schema=_schema(),
    )


def test_fixed_public_path_and_prepare_are_read_only(tmp_path: Path) -> None:
    path = p1_ledger_path(tmp_path)
    assert path == tmp_path / P1_LEDGER_RELATIVE_PATH

    prepared = build_next_p1_record(
        tmp_path,
        "ProtocolDefinition",
        recorded_at_utc="2026-08-31T02:00:00Z",
        fields=_protocol_fields(),
        schema=_schema(),
    )

    assert prepared["chain_sequence"] == 0
    assert prepared["previous_record_sha256"] is None
    assert not path.exists()
    assert read_p1_ledger(tmp_path, schema=_schema()) == ()


def test_genesis_is_one_canonical_fsynced_jsonl_record(tmp_path: Path) -> None:
    protocol = _append_protocol(tmp_path)
    path = p1_ledger_path(tmp_path)

    assert path.read_bytes() == canonical_json_bytes(protocol) + b"\n"
    assert read_p1_ledger(tmp_path, schema=_schema()) == (protocol,)
    assert not p1_ledger_lock_path(tmp_path).exists()


def test_authorization_appends_to_the_same_chain(tmp_path: Path) -> None:
    protocol = _append_protocol(tmp_path)
    authorization = append_new_p1_record(
        tmp_path,
        "RealIssueAuthorizationRecord",
        recorded_at_utc="2026-08-31T03:00:00Z",
        fields=_authorization_fields(protocol),
        schema=_schema(),
    )

    records = read_p1_ledger(tmp_path, schema=_schema())
    assert records == (protocol, authorization)
    assert authorization["chain_sequence"] == 1
    assert authorization["previous_record_sha256"] == protocol["content_sha256"]
    assert p1_ledger_path(tmp_path).read_bytes() == (
        canonical_json_bytes(protocol) + b"\n" + canonical_json_bytes(authorization) + b"\n"
    )


def test_duplicate_genesis_and_duplicate_authorization_are_rejected(tmp_path: Path) -> None:
    protocol = _append_protocol(tmp_path)
    with pytest.raises(P1LedgerError, match="genesis already exists"):
        append_new_p1_record(
            tmp_path,
            "ProtocolDefinition",
            recorded_at_utc="2026-08-31T02:30:00Z",
            fields=_protocol_fields(),
            schema=_schema(),
        )

    append_new_p1_record(
        tmp_path,
        "RealIssueAuthorizationRecord",
        recorded_at_utc="2026-08-31T03:00:00Z",
        fields=_authorization_fields(protocol),
        schema=_schema(),
    )
    with pytest.raises(P1LedgerError, match="AuthorizationRecord already exists"):
        append_new_p1_record(
            tmp_path,
            "RealIssueAuthorizationRecord",
            recorded_at_utc="2026-08-31T04:00:00Z",
            fields=_authorization_fields(protocol),
            schema=_schema(),
        )


def test_prebuilt_fork_and_duplicate_record_are_rejected(tmp_path: Path) -> None:
    protocol = _append_protocol(tmp_path)
    authorization = build_next_p1_record(
        tmp_path,
        "RealIssueAuthorizationRecord",
        recorded_at_utc="2026-08-31T03:00:00Z",
        fields=_authorization_fields(protocol),
        schema=_schema(),
    )
    fork = dict(authorization)
    fork["previous_record_sha256"] = "0" * 64
    fork = seal_record(fork)

    with pytest.raises(ValueError, match="previous_record_sha256"):
        append_p1_record(tmp_path, fork, schema=_schema())

    append_p1_record(tmp_path, authorization, schema=_schema())
    with pytest.raises(P1LedgerError, match="already exists"):
        append_p1_record(tmp_path, authorization, schema=_schema())


@pytest.mark.parametrize(
    "mutation",
    [
        lambda record: json.dumps(record, ensure_ascii=False, indent=2).encode("utf-8") + b"\n",
        lambda record: canonical_json_bytes(record),
        lambda record: b"\n" + canonical_json_bytes(record) + b"\n",
    ],
)
def test_noncanonical_incomplete_or_blank_ledger_is_rejected(tmp_path: Path, mutation: Any) -> None:
    protocol = _append_protocol(tmp_path)
    p1_ledger_path(tmp_path).write_bytes(mutation(protocol))

    with pytest.raises(P1LedgerIntegrityError):
        read_p1_ledger(tmp_path, schema=_schema(), require_exists=True)


def test_existing_writer_lock_fails_closed_without_creating_ledger(tmp_path: Path) -> None:
    lock_path = p1_ledger_lock_path(tmp_path)
    lock_path.parent.mkdir(parents=True)
    lock_path.write_bytes(b'{"pid":999999}\n')

    with pytest.raises(P1LedgerBusyError, match="process audit"):
        append_new_p1_record(
            tmp_path,
            "ProtocolDefinition",
            recorded_at_utc="2026-08-31T02:00:00Z",
            fields=_protocol_fields(),
            schema=_schema(),
        )

    assert not p1_ledger_path(tmp_path).exists()


def test_destructive_uncoordinated_mutation_between_validation_and_append_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol = _append_protocol(tmp_path)
    authorization = build_next_p1_record(
        tmp_path,
        "RealIssueAuthorizationRecord",
        recorded_at_utc="2026-08-31T03:00:00Z",
        fields=_authorization_fields(protocol),
        schema=_schema(),
    )
    original_append = ledger_module._append_bytes_cas

    def coordinated_attack(
        path: Path,
        expected_prefix: bytes,
        expected_signature: tuple[int, int, int, int, int, int],
        payload: bytes,
    ) -> None:
        path.write_bytes(expected_prefix + b"{}\n")
        original_append(path, expected_prefix, expected_signature, payload)

    monkeypatch.setattr(ledger_module, "_append_bytes_cas", coordinated_attack)
    with pytest.raises(P1LedgerIntegrityError, match="changed before append"):
        append_p1_record(tmp_path, authorization, schema=_schema())

    assert canonical_json_bytes(authorization) not in p1_ledger_path(tmp_path).read_bytes()
    assert not p1_ledger_lock_path(tmp_path).exists()


def test_validation_failure_never_appends_or_leaves_a_writer_lock(tmp_path: Path) -> None:
    protocol = _append_protocol(tmp_path)
    before = p1_ledger_path(tmp_path).read_bytes()
    invalid = build_next_p1_record(
        tmp_path,
        "RealIssueAuthorizationRecord",
        recorded_at_utc="2026-08-31T03:00:00Z",
        fields=_authorization_fields(protocol),
        schema=_schema(),
    )
    invalid["content_sha256"] = "f" * 64

    with pytest.raises(ValueError, match="content_sha256"):
        append_p1_record(tmp_path, invalid, schema=_schema())

    assert p1_ledger_path(tmp_path).read_bytes() == before
    assert not p1_ledger_lock_path(tmp_path).exists()
