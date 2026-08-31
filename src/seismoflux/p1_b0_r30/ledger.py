"""Fail-closed persistence for the single public P1 prospective record chain.

The scientific semantics live in :mod:`seismoflux.p1_b0_r30.records`.  This
module only gives those records one fixed JSONL location and an append-only
write boundary.  It deliberately has no recovery mode that truncates or
rewrites an existing ledger: a partial write, stale lock, non-canonical byte,
or coordinated external mutation requires human inspection.
"""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

from seismoflux.data.common import canonical_json_bytes
from seismoflux.p1_b0_r30.records import (
    JsonRecord,
    RecordType,
    build_record,
    validate_record_chain,
)

P1_LEDGER_RELATIVE_PATH = Path("outputs/prospective/p1_b0_r30_records_v1.jsonl")
P1_LEDGER_LOCK_RELATIVE_PATH = Path("outputs/prospective/.p1_b0_r30_records_v1.jsonl.lock")
P1_RECORD_SCHEMA_RELATIVE_PATH = Path("data/contracts/p1_prospective_records_v1.json")

# Keeping one record below this bound lets the append be issued as one local-file
# write.  The frozen record schema has no field that needs remotely large inline
# content; scientific artifacts are content-addressed instead.
MAX_RECORD_BYTES = 1_048_576

ScoreRegistryMap = Mapping[str, Sequence[Mapping[str, object]]]


class P1LedgerError(ValueError):
    """Base class for a fail-closed ledger refusal."""


class P1LedgerBusyError(P1LedgerError):
    """Another writer, or an unaudited stale writer lock, is present."""


class P1LedgerIntegrityError(P1LedgerError):
    """The public ledger bytes or their filesystem identity are not trustworthy."""


def p1_ledger_path(repository_root: Path) -> Path:
    """Return the only supported public P1 ledger path below ``repository_root``."""

    return _fixed_path(repository_root, P1_LEDGER_RELATIVE_PATH)


def p1_ledger_lock_path(repository_root: Path) -> Path:
    """Return the process-mutex path paired with the fixed public ledger."""

    return _fixed_path(repository_root, P1_LEDGER_LOCK_RELATIVE_PATH)


def read_p1_ledger(
    repository_root: Path,
    *,
    schema: Mapping[str, object] | None = None,
    require_exists: bool = False,
    score_registries_by_sha256: ScoreRegistryMap | None = None,
) -> tuple[JsonRecord, ...]:
    """Read and fully validate the canonical JSONL chain without changing it."""

    root = _repository_root(repository_root)
    path = _fixed_path(root, P1_LEDGER_RELATIVE_PATH)
    resolved_schema = _resolve_schema(root, schema)
    state = _read_state(path, allow_absent=not require_exists)
    if state is None:
        return ()
    raw_bytes, _ = state
    records = _decode_canonical_jsonl(raw_bytes)
    validate_record_chain(
        records,
        resolved_schema,
        score_registries_by_sha256=score_registries_by_sha256,
    )
    return tuple(dict(record) for record in records)


def build_next_p1_record(
    repository_root: Path,
    record_type: RecordType,
    *,
    recorded_at_utc: str,
    fields: Mapping[str, object],
    schema: Mapping[str, object] | None = None,
    score_registries_by_sha256: ScoreRegistryMap | None = None,
) -> JsonRecord:
    """Construct and validate the next record without persisting any bytes.

    A later append still compares the predecessor and the full existing ledger,
    so a record prepared here cannot silently fork a chain that advanced in the
    meantime.
    """

    root = _repository_root(repository_root)
    records = read_p1_ledger(
        root,
        schema=schema,
        score_registries_by_sha256=score_registries_by_sha256,
    )
    record = build_record(
        record_type,
        recorded_at_utc=recorded_at_utc,
        previous_record=records[-1] if records else None,
        fields=fields,
    )
    resolved_schema = _resolve_schema(root, schema)
    validate_record_chain(
        (*records, record),
        resolved_schema,
        score_registries_by_sha256=score_registries_by_sha256,
    )
    return record


def append_new_p1_record(
    repository_root: Path,
    record_type: RecordType,
    *,
    recorded_at_utc: str,
    fields: Mapping[str, object],
    schema: Mapping[str, object] | None = None,
    score_registries_by_sha256: ScoreRegistryMap | None = None,
) -> JsonRecord:
    """Build and append exactly one next record while holding the writer mutex."""

    root = _repository_root(repository_root)
    resolved_schema = _resolve_schema(root, schema)
    with _writer_lock(root):
        state = _read_state(_fixed_path(root, P1_LEDGER_RELATIVE_PATH), allow_absent=True)
        records = _validated_records(
            state,
            resolved_schema,
            score_registries_by_sha256=score_registries_by_sha256,
        )
        record = build_record(
            record_type,
            recorded_at_utc=recorded_at_utc,
            previous_record=records[-1] if records else None,
            fields=fields,
        )
        _append_validated_record(
            root,
            state,
            records,
            record,
            resolved_schema,
            score_registries_by_sha256=score_registries_by_sha256,
        )
    return record


def append_p1_record(
    repository_root: Path,
    record: Mapping[str, object],
    *,
    schema: Mapping[str, object] | None = None,
    score_registries_by_sha256: ScoreRegistryMap | None = None,
) -> JsonRecord:
    """Append a prebuilt sealed record only if it is the unique next chain link."""

    root = _repository_root(repository_root)
    resolved_schema = _resolve_schema(root, schema)
    candidate = dict(record)
    with _writer_lock(root):
        state = _read_state(_fixed_path(root, P1_LEDGER_RELATIVE_PATH), allow_absent=True)
        records = _validated_records(
            state,
            resolved_schema,
            score_registries_by_sha256=score_registries_by_sha256,
        )
        _append_validated_record(
            root,
            state,
            records,
            candidate,
            resolved_schema,
            score_registries_by_sha256=score_registries_by_sha256,
        )
    return candidate


def _repository_root(repository_root: Path) -> Path:
    root = Path(repository_root).resolve(strict=True)
    if not root.is_dir():
        raise P1LedgerIntegrityError("repository_root must be an existing directory")
    return root


def _fixed_path(repository_root: Path, relative_path: Path) -> Path:
    root = _repository_root(repository_root)
    candidate = root.joinpath(relative_path)
    try:
        candidate.relative_to(root)
    except ValueError as exc:  # pragma: no cover - constants make this defensive only
        raise P1LedgerIntegrityError("fixed P1 ledger path escaped repository_root") from exc
    return candidate


def _resolve_schema(
    repository_root: Path, schema: Mapping[str, object] | None
) -> Mapping[str, object]:
    if schema is not None:
        return schema
    schema_path = _fixed_path(repository_root, P1_RECORD_SCHEMA_RELATIVE_PATH)
    state = _read_state(schema_path, allow_absent=False)
    if state is None:  # pragma: no cover - allow_absent=False already fails
        raise P1LedgerIntegrityError("P1 record schema is missing")
    raw_bytes, _ = state
    try:
        decoded = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise P1LedgerIntegrityError("P1 record schema is not valid UTF-8 JSON") from exc
    if not isinstance(decoded, dict):
        raise P1LedgerIntegrityError("P1 record schema must be a JSON object")
    return cast(dict[str, object], decoded)


def _path_is_reparse_point(path: Path) -> bool:
    if path.is_symlink():
        return True
    metadata = path.stat(follow_symlinks=False)
    attributes = getattr(metadata, "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _reject_reparse_ancestors(repository_root: Path, path: Path) -> None:
    current = repository_root
    for part in path.relative_to(repository_root).parts:
        current = current / part
        if (current.exists() or current.is_symlink()) and _path_is_reparse_point(current):
            raise P1LedgerIntegrityError(
                f"reparse points are forbidden in the P1 ledger path: {current}"
            )


def _stat_signature(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_state(
    path: Path, *, allow_absent: bool
) -> tuple[bytes, tuple[int, int, int, int, int, int]] | None:
    root = path.parents[2]
    _reject_reparse_ancestors(root, path)
    if not path.exists():
        if allow_absent:
            return None
        raise P1LedgerIntegrityError(f"required file is missing: {path}")
    metadata = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode):
        raise P1LedgerIntegrityError(f"P1 ledger input is not a regular file: {path}")
    with path.open("rb") as handle:
        opened_before = os.fstat(handle.fileno())
        raw_bytes = handle.read()
        opened_after = os.fstat(handle.fileno())
    final = path.stat(follow_symlinks=False)
    signatures = {
        _stat_signature(metadata),
        _stat_signature(opened_before),
        _stat_signature(opened_after),
        _stat_signature(final),
    }
    if len(signatures) != 1:
        raise P1LedgerIntegrityError(f"file changed while it was read: {path}")
    return raw_bytes, _stat_signature(final)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise P1LedgerIntegrityError(f"duplicate JSON object key in P1 ledger: {key}")
        value[key] = item
    return value


def _decode_canonical_jsonl(raw_bytes: bytes) -> list[JsonRecord]:
    if not raw_bytes:
        raise P1LedgerIntegrityError("an existing P1 ledger may not be empty")
    if not raw_bytes.endswith(b"\n"):
        raise P1LedgerIntegrityError("P1 ledger must end with exactly one complete JSONL record")
    lines = raw_bytes.splitlines(keepends=True)
    records: list[JsonRecord] = []
    for line_number, line in enumerate(lines, start=1):
        if line == b"\n" or not line.endswith(b"\n"):
            raise P1LedgerIntegrityError(f"P1 ledger line {line_number} is blank or incomplete")
        payload = line[:-1]
        try:
            decoded = json.loads(payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
        except P1LedgerIntegrityError:
            raise
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise P1LedgerIntegrityError(
                f"P1 ledger line {line_number} is not valid UTF-8 JSON"
            ) from exc
        if not isinstance(decoded, dict):
            raise P1LedgerIntegrityError(f"P1 ledger line {line_number} must be a JSON object")
        if canonical_json_bytes(decoded) != payload:
            raise P1LedgerIntegrityError(f"P1 ledger line {line_number} is not canonical JSON")
        records.append(cast(JsonRecord, decoded))
    return records


def _validated_records(
    state: tuple[bytes, tuple[int, int, int, int, int, int]] | None,
    schema: Mapping[str, object],
    *,
    score_registries_by_sha256: ScoreRegistryMap | None,
) -> list[JsonRecord]:
    if state is None:
        return []
    records = _decode_canonical_jsonl(state[0])
    validate_record_chain(
        records,
        schema,
        score_registries_by_sha256=score_registries_by_sha256,
    )
    return records


def _append_validated_record(
    repository_root: Path,
    state: tuple[bytes, tuple[int, int, int, int, int, int]] | None,
    records: list[JsonRecord],
    candidate: JsonRecord,
    schema: Mapping[str, object],
    *,
    score_registries_by_sha256: ScoreRegistryMap | None,
) -> None:
    record_type = candidate.get("record_type")
    if records and record_type == "ProtocolDefinition":
        raise P1LedgerError("ProtocolDefinition genesis already exists")
    if not records and record_type != "ProtocolDefinition":
        raise P1LedgerError("the first P1 ledger record must be ProtocolDefinition")
    if record_type == "RealIssueAuthorizationRecord" and any(
        record.get("record_type") == "RealIssueAuthorizationRecord" for record in records
    ):
        raise P1LedgerError("RealIssueAuthorizationRecord already exists")
    content_sha256 = candidate.get("content_sha256")
    if isinstance(content_sha256, str) and any(
        record.get("content_sha256") == content_sha256 for record in records
    ):
        raise P1LedgerError("the candidate record already exists in the P1 ledger")

    candidate_chain = [*records, candidate]
    validate_record_chain(
        candidate_chain,
        schema,
        score_registries_by_sha256=score_registries_by_sha256,
    )
    payload = canonical_json_bytes(candidate) + b"\n"
    if len(payload) > MAX_RECORD_BYTES:
        raise P1LedgerError("P1 ledger record exceeds the one-write safety bound")

    path = _fixed_path(repository_root, P1_LEDGER_RELATIVE_PATH)
    if state is None:
        _create_ledger_exclusive(path, payload)
        expected = payload
    else:
        _append_bytes_cas(path, state[0], state[1], payload)
        expected = state[0] + payload

    observed_state = _read_state(path, allow_absent=False)
    if observed_state is None or observed_state[0] != expected:  # pragma: no cover - defensive
        raise P1LedgerIntegrityError("P1 ledger bytes changed immediately after append")
    observed_records = _decode_canonical_jsonl(observed_state[0])
    validate_record_chain(
        observed_records,
        schema,
        score_registries_by_sha256=score_registries_by_sha256,
    )
    if observed_records != candidate_chain:
        raise P1LedgerIntegrityError("P1 ledger semantic chain differs after append")


def _binary_flag() -> int:
    return cast(int, getattr(os, "O_BINARY", 0))


def _create_ledger_exclusive(path: Path, payload: bytes) -> None:
    _reject_reparse_ancestors(path.parents[2], path)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _binary_flag()
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise P1LedgerIntegrityError(
            "P1 ledger appeared after validation; refusing a competing genesis"
        ) from exc
    try:
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise P1LedgerIntegrityError("P1 genesis did not complete in one filesystem write")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_descriptor(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 65_536)
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


def _append_bytes_cas(
    path: Path,
    expected_prefix: bytes,
    expected_signature: tuple[int, int, int, int, int, int],
    payload: bytes,
) -> None:
    """Compare the exact prefix and issue one append write on the same descriptor."""

    flags = os.O_RDWR | os.O_APPEND | _binary_flag()
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if _stat_signature(opened) != expected_signature:
            raise P1LedgerIntegrityError("P1 ledger identity changed before append")
        if _read_descriptor(descriptor) != expected_prefix:
            raise P1LedgerIntegrityError("P1 ledger bytes changed before append")
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise P1LedgerIntegrityError("P1 record did not complete in one append write")
        os.fsync(descriptor)
        if _read_descriptor(descriptor) != expected_prefix + payload:
            raise P1LedgerIntegrityError("P1 ledger changed during coordinated append")
    finally:
        os.close(descriptor)


@contextmanager
def _writer_lock(repository_root: Path) -> Iterator[None]:
    lock_path = _fixed_path(repository_root, P1_LEDGER_LOCK_RELATIVE_PATH)
    # Check before mkdir so an existing reparse-point ancestor cannot redirect
    # even directory creation outside the repository.
    _reject_reparse_ancestors(repository_root, lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    _reject_reparse_ancestors(repository_root, lock_path)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _binary_flag()
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except FileExistsError as exc:
        raise P1LedgerBusyError(
            "P1 ledger writer lock already exists; do not delete it without process audit"
        ) from exc
    try:
        payload = canonical_json_bytes({"pid": os.getpid()}) + b"\n"
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise P1LedgerIntegrityError("P1 writer lock was not created completely")
        os.fsync(descriptor)
        lock_signature = _stat_signature(os.fstat(descriptor))
    finally:
        os.close(descriptor)

    try:
        yield
    finally:
        try:
            final_signature = _stat_signature(lock_path.stat(follow_symlinks=False))
        except FileNotFoundError as exc:
            raise P1LedgerIntegrityError("P1 writer lock disappeared during the append") from exc
        if final_signature != lock_signature:
            raise P1LedgerIntegrityError("P1 writer lock changed during the append")
        lock_path.unlink()


__all__ = [
    "MAX_RECORD_BYTES",
    "P1_LEDGER_LOCK_RELATIVE_PATH",
    "P1_LEDGER_RELATIVE_PATH",
    "P1_RECORD_SCHEMA_RELATIVE_PATH",
    "P1LedgerBusyError",
    "P1LedgerError",
    "P1LedgerIntegrityError",
    "append_new_p1_record",
    "append_p1_record",
    "build_next_p1_record",
    "p1_ledger_lock_path",
    "p1_ledger_path",
    "read_p1_ledger",
]
