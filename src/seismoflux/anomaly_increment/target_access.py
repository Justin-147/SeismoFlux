"""The sole authorized target entrance for stage 4.

No other stage-4 module should open, stat, hash, or parse the earthquake target.  This
entrance reserves the target-read ledger first, reads the file once into memory,
verifies the preregistered file digest, delegates parsing to an injected consumer, and
then records success or failure atomically.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Generic, TypeAlias, TypeVar

from seismoflux.anomaly_increment.attempt_ledger import (
    STAGE4_TARGET_SCOPE,
    Stage4OperationAlreadyConsumedError,
    complete_stage4_operation,
    reserve_stage4_operation,
)
from seismoflux.anomaly_increment.authorization import (
    Stage4ScoringNotAuthorizedError,
    Stage4TargetAuthorization,
    require_stage4_target_authorization,
)

T = TypeVar("T")
TargetBytesConsumer: TypeAlias = Callable[[bytes], T]


class Stage4TargetAccessError(RuntimeError):
    """Raised after authorization when the one-shot target entrance cannot complete."""


class Stage4LockedTestForbiddenError(Stage4TargetAccessError):
    """Raised unconditionally for any stage-4 locked-test request."""


@dataclass(frozen=True, slots=True)
class ConsumedStage4Target(Generic[T]):
    """Parsed in-memory target value and its preregistered identity receipt."""

    value: T
    operation_id: str
    target_sha256: str
    authorization_id: str


def require_stage4_execution_scope(scope: str) -> str:
    """Accept only the four preregistered stage-4 scopes; locked test is never legal."""

    allowed = {
        "development-fold-1",
        "development-fold-2",
        "development-fold-3",
        "formal-validation",
    }
    if scope not in allowed:
        if "lock" in scope.casefold() or "test" in scope.casefold():
            raise Stage4LockedTestForbiddenError(
                "locked test belongs to stage 9 and is forbidden throughout stage 4"
            )
        raise Stage4TargetAccessError("scope is outside the frozen stage-4 execution")
    return scope


def forbid_stage4_locked_test_access() -> None:
    """Permanent hard-stop entrypoint for accidental locked-test routing."""

    raise Stage4LockedTestForbiddenError(
        "stage 4 may never run, read, stat, hash, or infer the locked test"
    )


def _safe_failure_code(exc: BaseException) -> str:
    if isinstance(exc, Stage4ScoringNotAuthorizedError):
        return "authorization_changed_before_target_read"
    if isinstance(exc, Stage4TargetAccessError):
        return "target_access_contract_failure"
    name = re.sub(r"(?<!^)(?=[A-Z])", "_", type(exc).__name__).casefold()
    normalized = re.sub(r"[^a-z0-9_]", "_", name)[:96]
    if not normalized or not normalized[0].isalpha():
        return "unclassified_target_read_failure"
    return normalized


def _target_path_after_reservation(
    project_root: Path,
    authorization: Stage4TargetAuthorization,
) -> tuple[Path, str]:
    identity = authorization.expected_target_mapping()
    relative = identity.get("path")
    expected_sha256 = identity.get("sha256")
    if relative is None or expected_sha256 is None:
        raise Stage4TargetAccessError("authorized target identity is incomplete")
    if "\\" in relative:
        raise Stage4TargetAccessError("authorized target path is not normalized")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != relative:
        raise Stage4TargetAccessError("authorized target path is not project relative")
    root = Path(project_root).resolve()
    target = root.joinpath(*pure.parts).resolve()
    if not target.is_relative_to(root):
        raise Stage4TargetAccessError("authorized target path escapes the project root")
    return target, expected_sha256


def _read_target_bytes_once(path: Path) -> bytes:
    # This is intentionally the only target file-open operation in stage-4 code.
    with path.open("rb") as handle:
        return handle.read()


def consume_authorized_stage4_target(
    project_root: Path,
    authorization: Stage4TargetAuthorization,
    *,
    operation_id: str,
    consumer: TargetBytesConsumer[T],
) -> ConsumedStage4Target[T]:
    """Consume the sole target entrance after atomically recording the reservation."""

    capability = require_stage4_target_authorization(
        authorization,
        project_root=project_root,
    )
    if not callable(consumer):
        raise TypeError("consumer must be callable")
    reservation = reserve_stage4_operation(
        capability.target_read_ledger_path,
        kind="target_read",
        execution_binding_id=capability.execution_binding_id,
        operation_id=operation_id,
        scope=STAGE4_TARGET_SCOPE,
        authorization_id=capability.authorization_id,
    )
    if not reservation.changed:
        raise Stage4OperationAlreadyConsumedError(
            "the stage-4 target operation ID was already consumed"
        )

    try:
        target_path, expected_sha256 = _target_path_after_reservation(project_root, capability)
        payload = _read_target_bytes_once(target_path)
        actual_sha256 = hashlib.sha256(payload).hexdigest()
        if actual_sha256 != expected_sha256:
            raise Stage4TargetAccessError(
                "earthquake target bytes differ from the preregistered identity"
            )
        value = consumer(payload)
    except BaseException as exc:
        complete_stage4_operation(
            capability.target_read_ledger_path,
            kind="target_read",
            execution_binding_id=capability.execution_binding_id,
            operation_id=operation_id,
            status="failed",
            failure_code=_safe_failure_code(exc),
        )
        raise
    complete_stage4_operation(
        capability.target_read_ledger_path,
        kind="target_read",
        execution_binding_id=capability.execution_binding_id,
        operation_id=operation_id,
        status="succeeded",
        result_sha256=actual_sha256,
    )
    return ConsumedStage4Target(
        value=value,
        operation_id=operation_id,
        target_sha256=actual_sha256,
        authorization_id=capability.authorization_id,
    )


__all__ = [
    "ConsumedStage4Target",
    "Stage4LockedTestForbiddenError",
    "Stage4TargetAccessError",
    "TargetBytesConsumer",
    "consume_authorized_stage4_target",
    "forbid_stage4_locked_test_access",
    "require_stage4_execution_scope",
]
