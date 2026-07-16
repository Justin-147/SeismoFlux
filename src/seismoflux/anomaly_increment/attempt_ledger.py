"""Crash-safe local audit ledgers for the frozen stage-4 execution.

The ledgers are deliberately target agnostic.  They record *that* an operation was
reserved before work starts, and then atomically replace that reservation with a
terminal state.  A process failure therefore leaves either a durable ``started``
record or an explicit ``failed`` record; it can never erase an attempted target read
or formal scientific run.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import sys
import tempfile
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal, TypeAlias, cast

from seismoflux.anomaly_increment.immutable_file import (
    UnsafeImmutableFileError,
    ensure_real_directory_tree,
    open_existing_single_link_descriptor,
    read_existing_immutable_bytes,
    verify_opened_single_link_descriptor,
)
from seismoflux.anomaly_increment.preregistration import (
    verify_content_sha256,
    with_content_sha256,
)
from seismoflux.data.common import canonical_json_bytes

LedgerKind: TypeAlias = Literal["formal_attempt", "target_read"]
LedgerStatus: TypeAlias = Literal["started", "succeeded", "failed"]
Clock: TypeAlias = Callable[[], datetime]

STAGE4_LEDGER_SCHEMA_VERSION: Final[int] = 1
STAGE4_PROTOCOL_VERSION: Final[str] = "0.4.1"
STAGE4_ATTEMPT_SCOPES: Final[tuple[str, ...]] = (
    "development-fold-1",
    "development-fold-2",
    "development-fold-3",
    "formal-validation",
)
STAGE4_TARGET_SCOPE: Final[str] = "stage4-development-and-formal-validation-target"

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
_FAILURE_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,95}")
_THREAD_LOCK_GUARD = threading.Lock()
_THREAD_LOCKS: dict[str, threading.Lock] = {}


class Stage4LedgerError(RuntimeError):
    """Raised when a ledger is missing, altered, or used outside its binding."""


class Stage4OperationAlreadyConsumedError(Stage4LedgerError):
    """Raised when a one-shot stage-4 scope has already been consumed."""


def _sha256(value: str, *, label: str) -> str:
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 string")
    return value


def _identifier(value: str, *, label: str) -> str:
    if _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} is not a stable stage-4 identifier")
    return value


def _utc_text(value: datetime, *, label: str) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise TypeError(f"{label} must be a timezone-aware datetime")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _validated_utc_text(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{label} must be a canonical UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{label} must be a canonical UTC timestamp") from exc
    if _utc_text(parsed, label=label) != value:
        raise ValueError(f"{label} must be a canonical UTC timestamp")
    return value


def _parsed_utc_text(value: str, *, label: str) -> datetime:
    _validated_utc_text(value, label=label)
    return datetime.fromisoformat(value[:-1] + "+00:00")


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class Stage4LedgerRecord:
    """One durable reservation and its optional terminal outcome."""

    sequence: int
    operation_id: str
    scope: str
    authorization_id: str
    status: LedgerStatus
    started_at_utc: str
    completed_at_utc: str | None = None
    failure_code: str | None = None
    result_sha256: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or self.sequence < 1:
            raise ValueError("ledger record sequence must be a positive integer")
        _identifier(self.operation_id, label="operation_id")
        _identifier(self.scope, label="scope")
        _sha256(self.authorization_id, label="authorization_id")
        started_at = _parsed_utc_text(self.started_at_utc, label="started_at_utc")
        if self.status not in {"started", "succeeded", "failed"}:
            raise ValueError("unknown ledger record status")
        if self.status == "started":
            if any(
                value is not None
                for value in (self.completed_at_utc, self.failure_code, self.result_sha256)
            ):
                raise ValueError("a started ledger record cannot contain terminal fields")
            return
        if self.completed_at_utc is None:
            raise ValueError("a terminal ledger record requires completed_at_utc")
        completed_at = _parsed_utc_text(self.completed_at_utc, label="completed_at_utc")
        if completed_at < started_at:
            raise ValueError("ledger completion cannot precede its reservation")
        if self.status == "succeeded":
            if self.failure_code is not None:
                raise ValueError("a successful ledger record cannot contain a failure code")
            if self.result_sha256 is not None:
                _sha256(self.result_sha256, label="result_sha256")
            return
        if self.failure_code is None or _FAILURE_PATTERN.fullmatch(self.failure_code) is None:
            raise ValueError("a failed ledger record requires a normalized failure code")
        if self.result_sha256 is not None:
            raise ValueError("a failed ledger record cannot contain a result hash")

    def as_mapping(self) -> dict[str, object]:
        return {
            "authorization_id": self.authorization_id,
            "completed_at_utc": self.completed_at_utc,
            "failure_code": self.failure_code,
            "operation_id": self.operation_id,
            "result_sha256": self.result_sha256,
            "scope": self.scope,
            "sequence": self.sequence,
            "started_at_utc": self.started_at_utc,
            "status": self.status,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Stage4LedgerRecord:
        expected = {
            "authorization_id",
            "completed_at_utc",
            "failure_code",
            "operation_id",
            "result_sha256",
            "scope",
            "sequence",
            "started_at_utc",
            "status",
        }
        if set(value) != expected:
            raise Stage4LedgerError("ledger record fields changed")
        try:
            return cls(
                sequence=cast(int, value["sequence"]),
                operation_id=cast(str, value["operation_id"]),
                scope=cast(str, value["scope"]),
                authorization_id=cast(str, value["authorization_id"]),
                status=cast(LedgerStatus, value["status"]),
                started_at_utc=cast(str, value["started_at_utc"]),
                completed_at_utc=cast(str | None, value["completed_at_utc"]),
                failure_code=cast(str | None, value["failure_code"]),
                result_sha256=cast(str | None, value["result_sha256"]),
            )
        except (TypeError, ValueError) as exc:
            raise Stage4LedgerError("ledger record is malformed") from exc


@dataclass(frozen=True, slots=True)
class Stage4AuditLedger:
    """A complete content-addressed projection of one local audit ledger."""

    kind: LedgerKind
    execution_binding_id: str
    records: tuple[Stage4LedgerRecord, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in {"formal_attempt", "target_read"}:
            raise ValueError("unknown stage-4 ledger kind")
        _sha256(self.execution_binding_id, label="execution_binding_id")
        if any(not isinstance(item, Stage4LedgerRecord) for item in self.records):
            raise TypeError("ledger records must be Stage4LedgerRecord instances")
        expected_sequences = tuple(range(1, len(self.records) + 1))
        if tuple(item.sequence for item in self.records) != expected_sequences:
            raise ValueError("ledger record sequences must be contiguous and ordered")
        operation_ids = tuple(item.operation_id for item in self.records)
        if len(operation_ids) != len(set(operation_ids)):
            raise ValueError("ledger operation IDs must be unique")
        scopes = tuple(item.scope for item in self.records)
        allowed = (
            set(STAGE4_ATTEMPT_SCOPES) if self.kind == "formal_attempt" else {STAGE4_TARGET_SCOPE}
        )
        if not set(scopes) <= allowed:
            raise ValueError("ledger contains a scope outside the frozen stage-4 execution")
        if len(scopes) != len(set(scopes)):
            raise ValueError("each frozen stage-4 scope may be reserved at most once")
        if self.kind == "target_read" and len(self.records) > 1:
            raise ValueError("the sole stage-4 target entrance may be consumed at most once")

    @property
    def operation_count(self) -> int:
        return len(self.records)

    @property
    def started_count(self) -> int:
        return sum(item.status == "started" for item in self.records)

    @property
    def failed_count(self) -> int:
        return sum(item.status == "failed" for item in self.records)

    @property
    def succeeded_count(self) -> int:
        return sum(item.status == "succeeded" for item in self.records)

    def as_mapping(self) -> dict[str, object]:
        return with_content_sha256(
            {
                "execution_binding_id": self.execution_binding_id,
                "kind": self.kind,
                "protocol_version": STAGE4_PROTOCOL_VERSION,
                "records": [item.as_mapping() for item in self.records],
                "schema_version": STAGE4_LEDGER_SCHEMA_VERSION,
            }
        )

    @property
    def content_sha256(self) -> str:
        return cast(str, self.as_mapping()["content_sha256"])

    @property
    def ledger_id(self) -> str:
        return f"stage4-{self.kind}-ledger-{self.content_sha256[:16]}"

    def record(self, operation_id: str) -> Stage4LedgerRecord | None:
        return next((item for item in self.records if item.operation_id == operation_id), None)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> Stage4AuditLedger:
        expected = {
            "content_sha256",
            "execution_binding_id",
            "kind",
            "protocol_version",
            "records",
            "schema_version",
        }
        if set(value) != expected or not verify_content_sha256(value):
            raise Stage4LedgerError("ledger content hash or schema is invalid")
        if (
            value.get("schema_version") != STAGE4_LEDGER_SCHEMA_VERSION
            or value.get("protocol_version") != STAGE4_PROTOCOL_VERSION
        ):
            raise Stage4LedgerError("ledger version differs from the frozen stage-4 contract")
        raw_records = value.get("records")
        if not isinstance(raw_records, Sequence) or isinstance(raw_records, str | bytes):
            raise Stage4LedgerError("ledger records must be a sequence")
        records: list[Stage4LedgerRecord] = []
        for raw in raw_records:
            if not isinstance(raw, Mapping) or any(not isinstance(key, str) for key in raw):
                raise Stage4LedgerError("ledger record must be a string-keyed mapping")
            records.append(Stage4LedgerRecord.from_mapping(cast(Mapping[str, object], raw)))
        try:
            return cls(
                kind=cast(LedgerKind, value["kind"]),
                execution_binding_id=cast(str, value["execution_binding_id"]),
                records=tuple(records),
            )
        except (TypeError, ValueError) as exc:
            raise Stage4LedgerError("ledger invariants are invalid") from exc


@dataclass(frozen=True, slots=True)
class Stage4LedgerMutation:
    """Result of an idempotent reservation or terminal-state mutation."""

    ledger: Stage4AuditLedger
    record: Stage4LedgerRecord
    changed: bool


def _thread_lock(path: Path) -> threading.Lock:
    key = os.path.abspath(os.fspath(path))
    with _THREAD_LOCK_GUARD:
        return _THREAD_LOCKS.setdefault(key, threading.Lock())


@contextmanager
def _exclusive_file_lock(path: Path) -> Iterator[None]:
    try:
        ensure_real_directory_tree(
            Path(path.anchor) if path.anchor else Path.cwd(),
            path.parent,
            label="stage-4 ledger parent directory",
        )
    except UnsafeImmutableFileError as exc:
        raise Stage4LedgerError("stage-4 ledger parent path is unsafe") from exc
    lock_key = hashlib.sha256(os.path.abspath(os.fspath(path)).encode("utf-8")).hexdigest()
    lock_path = Path(tempfile.gettempdir()) / "seismoflux-stage4-ledger-locks" / lock_key
    try:
        ensure_real_directory_tree(
            Path(lock_path.anchor) if lock_path.anchor else Path.cwd(),
            lock_path.parent,
            label="stage-4 ledger lock directory",
        )
    except UnsafeImmutableFileError as exc:
        raise Stage4LedgerError("stage-4 ledger lock directory is unsafe") from exc
    with _thread_lock(lock_path):
        try:
            descriptor = os.open(
                lock_path,
                os.O_RDWR | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError:
            try:
                descriptor = open_existing_single_link_descriptor(
                    lock_path,
                    flags=os.O_RDWR,
                    label="stage-4 ledger process lock",
                )
            except UnsafeImmutableFileError as exc:
                raise Stage4LedgerError("stage-4 ledger process lock is unsafe") from exc
        try:
            if os.fstat(descriptor).st_size == 0:
                os.write(descriptor, b"\0")
                os.fsync(descriptor)
            os.lseek(descriptor, 0, os.SEEK_SET)
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
            else:
                fcntl = importlib.import_module("fcntl")
                fcntl.flock(descriptor, fcntl.LOCK_EX)
            try:
                try:
                    verify_opened_single_link_descriptor(
                        lock_path,
                        descriptor,
                        label="stage-4 ledger process lock",
                    )
                except UnsafeImmutableFileError as exc:
                    raise Stage4LedgerError(
                        "stage-4 ledger process lock changed before use"
                    ) from exc
                yield
            finally:
                os.lseek(descriptor, 0, os.SEEK_SET)
                if sys.platform == "win32":
                    import msvcrt

                    msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
                else:
                    fcntl = importlib.import_module("fcntl")
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _read_unlocked(path: Path) -> Stage4AuditLedger:
    try:
        payload = read_existing_immutable_bytes(
            path,
            label=f"stage-4 {path.name} ledger",
        )
        raw = json.loads(payload.decode("utf-8"))
    except UnsafeImmutableFileError as exc:
        if isinstance(exc.__cause__, FileNotFoundError):
            raise Stage4LedgerError(f"required stage-4 ledger is missing: {path.name}") from exc
        raise Stage4LedgerError(
            f"stage-4 ledger is not a safe single-link file: {path.name}"
        ) from exc
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Stage4LedgerError(f"stage-4 ledger cannot be decoded: {path.name}") from exc
    if not isinstance(raw, Mapping) or any(not isinstance(key, str) for key in raw):
        raise Stage4LedgerError("stage-4 ledger root must be a string-keyed mapping")
    return Stage4AuditLedger.from_mapping(cast(Mapping[str, object], raw))


def _write_unlocked(
    path: Path,
    ledger: Stage4AuditLedger,
    *,
    previous: Stage4AuditLedger | None,
) -> None:
    serialized = canonical_json_bytes(ledger.as_mapping()) + b"\n"
    if previous is not None and _read_unlocked(path) != previous:
        raise Stage4LedgerError("stage-4 ledger changed before its atomic update")
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        if previous is None:
            try:
                os.link(temporary_name, path)
            except FileExistsError as exc:
                raise Stage4LedgerError(
                    "stage-4 ledger appeared during create-only initialization"
                ) from exc
            Path(temporary_name).unlink()
        else:
            os.replace(temporary_name, path)
        temporary_name = None
    except Exception:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise
    try:
        observed = read_existing_immutable_bytes(
            path,
            label=f"updated stage-4 {path.name} ledger",
        )
    except UnsafeImmutableFileError as exc:
        raise Stage4LedgerError("updated stage-4 ledger is not a safe single-link file") from exc
    if observed != serialized:
        raise Stage4LedgerError("updated stage-4 ledger bytes changed after replacement")


def initialize_stage4_ledger(
    path: Path,
    *,
    kind: LedgerKind,
    execution_binding_id: str,
) -> Stage4AuditLedger:
    """Create one empty ledger, or idempotently verify the existing empty binding."""

    expected = Stage4AuditLedger(kind=kind, execution_binding_id=execution_binding_id)
    target = Path(path)
    with _exclusive_file_lock(target):
        try:
            os.lstat(target)
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise Stage4LedgerError("cannot safely inspect stage-4 ledger path") from exc
        else:
            current = _read_unlocked(target)
            if current != expected:
                raise Stage4LedgerError("existing stage-4 ledger is non-empty or differently bound")
            return current
        _write_unlocked(target, expected, previous=None)
        return expected


def read_stage4_ledger(
    path: Path,
    *,
    expected_kind: LedgerKind | None = None,
    expected_binding_id: str | None = None,
) -> Stage4AuditLedger:
    """Read and verify an atomic ledger snapshot."""

    target = Path(path)
    with _exclusive_file_lock(target):
        ledger = _read_unlocked(target)
    if expected_kind is not None and ledger.kind != expected_kind:
        raise Stage4LedgerError("stage-4 ledger kind changed")
    if expected_binding_id is not None and ledger.execution_binding_id != expected_binding_id:
        raise Stage4LedgerError("stage-4 ledger belongs to another execution binding")
    return ledger


def reserve_stage4_operation(
    path: Path,
    *,
    kind: LedgerKind,
    execution_binding_id: str,
    operation_id: str,
    scope: str,
    authorization_id: str,
    clock: Clock = _utc_now,
) -> Stage4LedgerMutation:
    """Durably reserve a one-shot scope before any target or score-bearing work."""

    _identifier(operation_id, label="operation_id")
    _identifier(scope, label="scope")
    _sha256(authorization_id, label="authorization_id")
    target = Path(path)
    with _exclusive_file_lock(target):
        ledger = _read_unlocked(target)
        if ledger.kind != kind or ledger.execution_binding_id != execution_binding_id:
            raise Stage4LedgerError("stage-4 operation uses another ledger binding")
        existing = ledger.record(operation_id)
        if existing is not None:
            if existing.scope != scope or existing.authorization_id != authorization_id:
                raise Stage4LedgerError("operation ID was already bound to different semantics")
            return Stage4LedgerMutation(ledger=ledger, record=existing, changed=False)
        if any(item.scope == scope for item in ledger.records):
            raise Stage4OperationAlreadyConsumedError(
                f"stage-4 one-shot scope has already been consumed: {scope}"
            )
        record = Stage4LedgerRecord(
            sequence=len(ledger.records) + 1,
            operation_id=operation_id,
            scope=scope,
            authorization_id=authorization_id,
            status="started",
            started_at_utc=_utc_text(clock(), label="clock result"),
        )
        updated = replace(ledger, records=(*ledger.records, record))
        _write_unlocked(target, updated, previous=ledger)
        return Stage4LedgerMutation(ledger=updated, record=record, changed=True)


def reserve_stage4_attempt_scopes(
    path: Path,
    *,
    execution_binding_id: str,
    operation_ids_by_scope: Mapping[str, str],
    authorization_id: str,
    clock: Clock = _utc_now,
) -> Stage4AuditLedger:
    """Atomically reserve all four frozen scientific scopes before target ingress."""

    operations = dict(operation_ids_by_scope)
    if set(operations) != set(STAGE4_ATTEMPT_SCOPES):
        raise ValueError("formal attempt reservation must cover exactly all four scopes")
    if len(set(operations.values())) != len(STAGE4_ATTEMPT_SCOPES):
        raise ValueError("formal attempt operation IDs must be unique")
    for scope in STAGE4_ATTEMPT_SCOPES:
        _identifier(operations[scope], label=f"operation_id for {scope}")
    _sha256(authorization_id, label="authorization_id")
    target = Path(path)
    with _exclusive_file_lock(target):
        ledger = _read_unlocked(target)
        if ledger.kind != "formal_attempt" or ledger.execution_binding_id != (execution_binding_id):
            raise Stage4LedgerError("formal attempt batch uses another ledger binding")
        if ledger.records:
            raise Stage4OperationAlreadyConsumedError(
                "one or more stage-4 formal attempt scopes were already consumed"
            )
        started_at = _utc_text(clock(), label="clock result")
        records = tuple(
            Stage4LedgerRecord(
                sequence=index,
                operation_id=operations[scope],
                scope=scope,
                authorization_id=authorization_id,
                status="started",
                started_at_utc=started_at,
            )
            for index, scope in enumerate(STAGE4_ATTEMPT_SCOPES, start=1)
        )
        updated = replace(ledger, records=records)
        _write_unlocked(target, updated, previous=ledger)
        return updated


def complete_stage4_operation(
    path: Path,
    *,
    kind: LedgerKind,
    execution_binding_id: str,
    operation_id: str,
    status: Literal["succeeded", "failed"],
    result_sha256: str | None = None,
    failure_code: str | None = None,
    clock: Clock = _utc_now,
) -> Stage4LedgerMutation:
    """Atomically complete one reservation; an identical completion is idempotent."""

    if status == "succeeded":
        if failure_code is not None:
            raise ValueError("successful completion cannot include failure_code")
        if result_sha256 is not None:
            _sha256(result_sha256, label="result_sha256")
    else:
        if failure_code is None or _FAILURE_PATTERN.fullmatch(failure_code) is None:
            raise ValueError("failed completion requires a normalized failure_code")
        if result_sha256 is not None:
            raise ValueError("failed completion cannot include result_sha256")

    target = Path(path)
    with _exclusive_file_lock(target):
        ledger = _read_unlocked(target)
        if ledger.kind != kind or ledger.execution_binding_id != execution_binding_id:
            raise Stage4LedgerError("stage-4 completion uses another ledger binding")
        existing = ledger.record(operation_id)
        if existing is None:
            raise Stage4LedgerError("cannot complete an unregistered stage-4 operation")
        if existing.status != "started":
            if (
                existing.status == status
                and existing.result_sha256 == result_sha256
                and existing.failure_code == failure_code
            ):
                return Stage4LedgerMutation(ledger=ledger, record=existing, changed=False)
            raise Stage4LedgerError("stage-4 operation already has a different terminal state")
        completed = replace(
            existing,
            status=status,
            completed_at_utc=_utc_text(clock(), label="clock result"),
            failure_code=failure_code,
            result_sha256=result_sha256,
        )
        records = tuple(
            completed if item.operation_id == operation_id else item for item in ledger.records
        )
        updated = replace(ledger, records=records)
        _write_unlocked(target, updated, previous=ledger)
        return Stage4LedgerMutation(ledger=updated, record=completed, changed=True)


def complete_stage4_attempt_scopes(
    path: Path,
    *,
    execution_binding_id: str,
    operation_ids_by_scope: Mapping[str, str],
    authorization_id: str,
    status: Literal["succeeded", "failed"],
    result_sha256: str | None = None,
    failure_code: str | None = None,
    clock: Clock = _utc_now,
) -> Stage4AuditLedger:
    """Atomically place all four formal-scope reservations in one terminal state."""

    operations = dict(operation_ids_by_scope)
    if set(operations) != set(STAGE4_ATTEMPT_SCOPES):
        raise ValueError("formal attempt completion must cover exactly all four scopes")
    if len(set(operations.values())) != len(STAGE4_ATTEMPT_SCOPES):
        raise ValueError("formal attempt completion operation IDs must be unique")
    for scope in STAGE4_ATTEMPT_SCOPES:
        _identifier(operations[scope], label=f"operation_id for {scope}")
    _sha256(authorization_id, label="authorization_id")
    if status == "succeeded":
        if failure_code is not None:
            raise ValueError("successful batch completion cannot include failure_code")
        if result_sha256 is None:
            raise ValueError("successful batch completion requires result_sha256")
        _sha256(result_sha256, label="result_sha256")
    else:
        if failure_code is None or _FAILURE_PATTERN.fullmatch(failure_code) is None:
            raise ValueError("failed batch completion requires a normalized failure_code")
        if result_sha256 is not None:
            raise ValueError("failed batch completion cannot include result_sha256")

    target = Path(path)
    with _exclusive_file_lock(target):
        ledger = _read_unlocked(target)
        if ledger.kind != "formal_attempt" or ledger.execution_binding_id != (execution_binding_id):
            raise Stage4LedgerError("formal attempt batch completion uses another binding")
        if tuple(item.scope for item in ledger.records) != STAGE4_ATTEMPT_SCOPES:
            raise Stage4LedgerError("formal attempt ledger does not contain the frozen batch")
        for record in ledger.records:
            if (
                record.operation_id != operations[record.scope]
                or record.authorization_id != authorization_id
            ):
                raise Stage4LedgerError("formal attempt batch identity changed")
        if all(item.status != "started" for item in ledger.records):
            if all(
                item.status == status
                and item.result_sha256 == result_sha256
                and item.failure_code == failure_code
                for item in ledger.records
            ):
                return ledger
            raise Stage4LedgerError("formal attempt batch already has another terminal state")
        if any(item.status != "started" for item in ledger.records):
            raise Stage4LedgerError("formal attempt batch has a partial terminal state")
        completed_at = _utc_text(clock(), label="clock result")
        completed = tuple(
            replace(
                item,
                status=status,
                completed_at_utc=completed_at,
                result_sha256=result_sha256,
                failure_code=failure_code,
            )
            for item in ledger.records
        )
        updated = replace(ledger, records=completed)
        _write_unlocked(target, updated, previous=ledger)
        return updated


def recover_interrupted_stage4_operations(
    path: Path,
    *,
    kind: LedgerKind,
    execution_binding_id: str,
    clock: Clock = _utc_now,
) -> Stage4AuditLedger:
    """Refuse automatic recovery because stage 4 has no frozen lease/owner protocol.

    A ``started`` record is durable evidence of a potentially live or interrupted
    formal run.  Automatically rewriting it could race a live worker and falsely
    authorize another scientific attempt.  Stage 4 therefore permits only read-only
    inspection; terminalization requires a future, separately frozen lease protocol.
    """

    target = Path(path)
    with _exclusive_file_lock(target):
        ledger = _read_unlocked(target)
        if ledger.kind != kind or ledger.execution_binding_id != execution_binding_id:
            raise Stage4LedgerError("stage-4 recovery uses another ledger binding")
        del clock
        if any(item.status == "started" for item in ledger.records):
            raise Stage4LedgerError(
                "automatic recovery is forbidden without a frozen owner/lease/staleness proof"
            )
        return ledger


@contextmanager
def registered_stage4_attempt(
    path: Path,
    *,
    execution_binding_id: str,
    operation_id: str,
    scope: str,
    authorization_id: str,
    clock: Clock = _utc_now,
) -> Iterator[None]:
    """Register one fold/validation attempt and retain failures without error text."""

    mutation = reserve_stage4_operation(
        path,
        kind="formal_attempt",
        execution_binding_id=execution_binding_id,
        operation_id=operation_id,
        scope=scope,
        authorization_id=authorization_id,
        clock=clock,
    )
    if not mutation.changed:
        raise Stage4OperationAlreadyConsumedError(
            f"stage-4 attempt operation ID has already been consumed: {operation_id}"
        )
    try:
        yield
    except BaseException as exc:
        failure_name = re.sub(r"(?<!^)(?=[A-Z])", "_", type(exc).__name__).casefold()
        failure_code = re.sub(r"[^a-z0-9_]", "_", failure_name)[:96]
        if _FAILURE_PATTERN.fullmatch(failure_code) is None:
            failure_code = "unclassified_execution_failure"
        complete_stage4_operation(
            path,
            kind="formal_attempt",
            execution_binding_id=execution_binding_id,
            operation_id=operation_id,
            status="failed",
            failure_code=failure_code,
            clock=clock,
        )
        raise
    else:
        complete_stage4_operation(
            path,
            kind="formal_attempt",
            execution_binding_id=execution_binding_id,
            operation_id=operation_id,
            status="succeeded",
            result_sha256=hashlib.sha256(
                canonical_json_bytes(
                    {
                        "authorization_id": authorization_id,
                        "operation_id": operation_id,
                        "scope": scope,
                        "status": "succeeded",
                    }
                )
            ).hexdigest(),
            clock=clock,
        )


__all__ = [
    "STAGE4_ATTEMPT_SCOPES",
    "STAGE4_LEDGER_SCHEMA_VERSION",
    "STAGE4_PROTOCOL_VERSION",
    "STAGE4_TARGET_SCOPE",
    "Stage4AuditLedger",
    "Stage4LedgerError",
    "Stage4LedgerMutation",
    "Stage4LedgerRecord",
    "Stage4OperationAlreadyConsumedError",
    "complete_stage4_attempt_scopes",
    "complete_stage4_operation",
    "initialize_stage4_ledger",
    "read_stage4_ledger",
    "recover_interrupted_stage4_operations",
    "registered_stage4_attempt",
    "reserve_stage4_attempt_scopes",
    "reserve_stage4_operation",
]
