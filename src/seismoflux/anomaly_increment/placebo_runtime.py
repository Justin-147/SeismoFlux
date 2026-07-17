"""Crash-safe path runtime for lazy stage-4 permutation refits.

The scientific pipeline remains path-free.  This module owns checkpoint JSON,
heartbeats, resource telemetry, BLAS thread enforcement, and strict prefix resume.
It never constructs a target path and never reads a target catalogue.
"""

from __future__ import annotations

import ctypes
import json
import math
import os
import re
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal, TypeAlias, cast

from seismoflux.anomaly_increment.compute import Stage4WorkerPlan
from seismoflux.anomaly_increment.immutable_file import (
    UnsafeImmutableFileError,
    ensure_real_directory_tree,
    read_existing_immutable_file,
    require_existing_real_directory,
)
from seismoflux.anomaly_increment.placebo import PERMUTATION_REPLICATIONS
from seismoflux.anomaly_increment.preregistration import (
    verify_content_sha256,
    with_content_sha256,
)
from seismoflux.anomaly_increment.scoring_pipeline import (
    CompletedPlaceboReplication,
    PlaceboExecution,
    PlaceboInjection,
    PlaceboRequest,
    PlaceboSource,
)
from seismoflux.data.common import canonical_json_bytes

STAGE4_PLACEBO_CHECKPOINT_SCHEMA_VERSION: Final[int] = 1
STAGE4_PROTOCOL_VERSION: Final[str] = "0.4.1"
CHECKPOINT_EVERY_REPLICATIONS: Final[int] = 25
HEARTBEAT_SECONDS: Final[int] = 30
HEARTBEAT_STALE_AFTER_SECONDS: Final[int] = 90

CheckpointStatus: TypeAlias = Literal["running", "interrupted", "completed"]
BackendId: TypeAlias = Literal["cpu_float64", "gpu_float64"]
PlaceboSourceFactory: TypeAlias = Callable[[PlaceboRequest], PlaceboSource]
MemoryProbe: TypeAlias = Callable[[], int | None]
Clock: TypeAlias = Callable[[], datetime]
SecondsClock: TypeAlias = Callable[[], float]

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_FAILURE_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,95}")
_OWNER_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")


class PlaceboCheckpointError(RuntimeError):
    """A checkpoint is missing required identity or cannot be resumed safely."""


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = (
        ("cb", ctypes.c_ulong),
        ("page_fault_count", ctypes.c_ulong),
        ("peak_working_set_size", ctypes.c_size_t),
        ("working_set_size", ctypes.c_size_t),
        ("quota_peak_paged_pool_usage", ctypes.c_size_t),
        ("quota_paged_pool_usage", ctypes.c_size_t),
        ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
        ("quota_non_paged_pool_usage", ctypes.c_size_t),
        ("pagefile_usage", ctypes.c_size_t),
        ("peak_pagefile_usage", ctypes.c_size_t),
    )


def process_peak_working_set_bytes() -> int | None:
    """Return peak resident process memory when the platform exposes it."""

    if os.name == "posix":
        try:
            import resource
        except ImportError:
            return None
        resource_module = cast(Any, resource)
        peak = int(resource_module.getrusage(resource_module.RUSAGE_SELF).ru_maxrss)
        if peak <= 0:
            return None
        return peak if sys.platform == "darwin" else peak * 1_024
    if os.name != "nt":
        return None
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    try:
        loader = getattr(ctypes, "windll", None)
        if loader is None:
            return None
        get_current_process = loader.kernel32.GetCurrentProcess
        get_current_process.restype = ctypes.c_void_p
        get_process_memory_info = loader.psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(_ProcessMemoryCounters),
            ctypes.c_ulong,
        )
        get_process_memory_info.restype = ctypes.c_int
        process = get_current_process()
        succeeded = get_process_memory_info(process, ctypes.byref(counters), counters.cb)
    except (AttributeError, OSError):
        return None
    return int(counters.peak_working_set_size) if succeeded else None


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise PlaceboCheckpointError(f"{label} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise PlaceboCheckpointError(f"{label} must be a lowercase SHA-256")
    return value


def _timestamp(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise TypeError("checkpoint clock must return a timezone-aware datetime")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: object, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise PlaceboCheckpointError(f"{label} must be a canonical UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise PlaceboCheckpointError(f"{label} must be a canonical UTC timestamp") from exc
    if _timestamp(parsed) != value:
        raise PlaceboCheckpointError(f"{label} must be a canonical UTC timestamp")
    return parsed


def _nonnegative_hex(value: object, *, label: str) -> float:
    if not isinstance(value, str):
        raise PlaceboCheckpointError(f"{label} must be a hexadecimal float")
    try:
        result = float.fromhex(value)
    except ValueError as exc:
        raise PlaceboCheckpointError(f"{label} must be a hexadecimal float") from exc
    if not math.isfinite(result) or result < 0.0 or result.hex() != value:
        raise PlaceboCheckpointError(f"{label} must be finite, nonnegative, and canonical")
    return result


def _worker_mapping(worker: Stage4WorkerPlan) -> dict[str, object]:
    return {
        "blas_threads_per_worker": worker.blas_threads_per_worker,
        "configured_max_workers": worker.configured_max_workers,
        "effective_workers": worker.effective_workers,
        "logical_processors": worker.logical_processors,
        "nested_parallelism": worker.nested_parallelism,
        "physical_cores": worker.physical_cores,
        "reserve_physical_cores": worker.reserve_physical_cores,
    }


def _worker_from_mapping(value: object) -> Stage4WorkerPlan:
    raw = _mapping(value, label="checkpoint worker plan")
    expected = {
        "blas_threads_per_worker",
        "configured_max_workers",
        "effective_workers",
        "logical_processors",
        "nested_parallelism",
        "physical_cores",
        "reserve_physical_cores",
    }
    if set(raw) != expected:
        raise PlaceboCheckpointError("checkpoint worker-plan schema changed")
    try:
        return Stage4WorkerPlan(
            physical_cores=cast(int, raw["physical_cores"]),
            logical_processors=cast(int, raw["logical_processors"]),
            reserve_physical_cores=cast(int, raw["reserve_physical_cores"]),
            configured_max_workers=cast(int, raw["configured_max_workers"]),
            effective_workers=cast(int, raw["effective_workers"]),
            blas_threads_per_worker=cast(int, raw["blas_threads_per_worker"]),
            nested_parallelism=cast(bool, raw["nested_parallelism"]),
        )
    except (TypeError, ValueError) as exc:
        raise PlaceboCheckpointError("checkpoint worker plan is invalid") from exc


def _completed_from_mapping(value: object) -> CompletedPlaceboReplication:
    raw = _mapping(value, label="completed placebo result")
    expected = {
        "converged",
        "mapping_sha256",
        "replication_index",
        "scientific_failure_code",
        "statistic_hex",
    }
    if set(raw) != expected or not isinstance(raw.get("converged"), bool):
        raise PlaceboCheckpointError("completed placebo result schema changed")
    statistic_raw = raw.get("statistic_hex")
    if statistic_raw is not None and not isinstance(statistic_raw, str):
        raise PlaceboCheckpointError("completed placebo statistic is malformed")
    try:
        statistic = None if statistic_raw is None else float.fromhex(statistic_raw)
        if statistic is not None and statistic.hex() != statistic_raw:
            raise ValueError("noncanonical statistic")
        return CompletedPlaceboReplication(
            replication_index=cast(int, raw["replication_index"]),
            mapping_sha256=cast(str, raw["mapping_sha256"]),
            statistic=statistic,
            converged=cast(bool, raw["converged"]),
            scientific_failure_code=cast(str | None, raw["scientific_failure_code"]),
        )
    except (TypeError, ValueError) as exc:
        raise PlaceboCheckpointError("completed placebo result is invalid") from exc


@dataclass(frozen=True, slots=True)
class PlaceboCheckpoint:
    execution_binding_id: str
    request_identity: tuple[tuple[str, str], ...]
    request_sha256: str
    source_id_sha256: str
    worker_plan: Stage4WorkerPlan
    max_in_flight: int
    blas_environment: tuple[tuple[str, str], ...]
    backend: BackendId
    gpu_device: str | None
    gpu_memory_used_bytes: int | None
    results: tuple[CompletedPlaceboReplication, ...]
    status: CheckpointStatus
    owner_id: str
    checkpoint_sequence: int
    created_at_utc: str
    updated_at_utc: str
    last_heartbeat_utc: str
    elapsed_seconds: float
    cpu_seconds: float
    process_peak_working_set_bytes: int | None
    infrastructure_failure_code: str | None = None

    def __post_init__(self) -> None:
        _sha256(self.execution_binding_id, label="execution_binding_id")
        _sha256(self.request_sha256, label="request_sha256")
        _sha256(self.source_id_sha256, label="source_id_sha256")
        if self.request_identity != tuple(sorted(self.request_identity)) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in self.request_identity
        ):
            raise ValueError("checkpoint request identity must be sorted string pairs")
        if (
            self.request_sha256
            != PlaceboRequest(
                kind=cast(Any, dict(self.request_identity)["kind"]),
                evaluation_id=dict(self.request_identity)["evaluation_id"],
                model_variant=cast(Any, dict(self.request_identity)["model_variant"]),
                observed_statistic=float.fromhex(
                    dict(self.request_identity)["observed_statistic_hex"]
                ),
                frozen_rate_head_sha256=dict(self.request_identity)["frozen_rate_head_sha256"],
            ).content_sha256
        ):
            raise ValueError("checkpoint request identity hash changed")
        if (
            self.worker_plan.reserve_physical_cores < 2
            or self.worker_plan.configured_max_workers > 12
            or self.worker_plan.blas_threads_per_worker != 1
            or self.worker_plan.nested_parallelism
        ):
            raise ValueError("checkpoint worker plan violates the frozen policy")
        if (
            not isinstance(self.max_in_flight, int)
            or isinstance(self.max_in_flight, bool)
            or not 1 <= self.max_in_flight <= self.worker_plan.effective_workers
        ):
            raise ValueError("checkpoint max_in_flight is outside the worker plan")
        if dict(self.blas_environment) != self.worker_plan.blas_environment() or set(
            dict(self.blas_environment).values()
        ) != {"1"}:
            raise ValueError("checkpoint BLAS environment is not frozen to one thread")
        if self.backend not in {"cpu_float64", "gpu_float64"}:
            raise ValueError("checkpoint backend is invalid")
        if self.gpu_device is not None and (
            not self.gpu_device or self.gpu_device != self.gpu_device.strip()
        ):
            raise ValueError("checkpoint GPU device is malformed")
        if self.gpu_memory_used_bytes is not None and (
            isinstance(self.gpu_memory_used_bytes, bool) or self.gpu_memory_used_bytes < 0
        ):
            raise ValueError("checkpoint GPU memory must be nonnegative")
        results = tuple(self.results)
        if tuple(item.replication_index for item in results) != tuple(range(len(results))):
            raise ValueError("checkpoint results must be one contiguous prefix")
        if len(results) > PERMUTATION_REPLICATIONS:
            raise ValueError("checkpoint contains too many permutation results")
        if self.status not in {"running", "interrupted", "completed"}:
            raise ValueError("checkpoint status is invalid")
        if self.status == "completed" and len(results) != PERMUTATION_REPLICATIONS:
            raise ValueError("completed checkpoint must contain all 1000 results")
        if self.status == "interrupted":
            if (
                self.infrastructure_failure_code is None
                or _FAILURE_PATTERN.fullmatch(self.infrastructure_failure_code) is None
            ):
                raise ValueError("interrupted checkpoint requires a failure code")
        elif self.infrastructure_failure_code is not None:
            raise ValueError("non-interrupted checkpoint cannot contain a failure code")
        if _OWNER_PATTERN.fullmatch(self.owner_id) is None:
            raise ValueError("checkpoint owner_id is invalid")
        if isinstance(self.checkpoint_sequence, bool) or self.checkpoint_sequence < 1:
            raise ValueError("checkpoint sequence must be positive")
        created = _parse_timestamp(self.created_at_utc, label="created_at_utc")
        updated = _parse_timestamp(self.updated_at_utc, label="updated_at_utc")
        heartbeat = _parse_timestamp(self.last_heartbeat_utc, label="last_heartbeat_utc")
        if updated < created or heartbeat < created or heartbeat > updated:
            raise ValueError("checkpoint timestamps are out of order")
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0.0:
            raise ValueError("checkpoint elapsed_seconds must be finite and nonnegative")
        if not math.isfinite(self.cpu_seconds) or self.cpu_seconds < 0.0:
            raise ValueError("checkpoint cpu_seconds must be finite and nonnegative")
        if self.process_peak_working_set_bytes is not None and (
            isinstance(self.process_peak_working_set_bytes, bool)
            or self.process_peak_working_set_bytes <= 0
        ):
            raise ValueError("checkpoint process memory must be positive when known")
        object.__setattr__(self, "results", results)

    def as_mapping(self) -> dict[str, object]:
        return with_content_sha256(
            {
                "backend": self.backend,
                "blas_environment": dict(self.blas_environment),
                "checkpoint_every_replications": CHECKPOINT_EVERY_REPLICATIONS,
                "checkpoint_sequence": self.checkpoint_sequence,
                "cpu_seconds_hex": self.cpu_seconds.hex(),
                "created_at_utc": self.created_at_utc,
                "elapsed_seconds_hex": self.elapsed_seconds.hex(),
                "execution_binding_id": self.execution_binding_id,
                "gpu_device": self.gpu_device,
                "gpu_memory_used_bytes": self.gpu_memory_used_bytes,
                "heartbeat_seconds": HEARTBEAT_SECONDS,
                "infrastructure_failure_code": self.infrastructure_failure_code,
                "last_heartbeat_utc": self.last_heartbeat_utc,
                "max_in_flight": self.max_in_flight,
                "owner_id": self.owner_id,
                "process_peak_working_set_bytes": self.process_peak_working_set_bytes,
                "protocol_version": STAGE4_PROTOCOL_VERSION,
                "request": dict(self.request_identity),
                "request_sha256": self.request_sha256,
                "results": [item.as_mapping() for item in self.results],
                "schema_version": STAGE4_PLACEBO_CHECKPOINT_SCHEMA_VERSION,
                "source_id_sha256": self.source_id_sha256,
                "status": self.status,
                "updated_at_utc": self.updated_at_utc,
                "worker_plan": _worker_mapping(self.worker_plan),
            }
        )

    @property
    def content_sha256(self) -> str:
        return cast(str, self.as_mapping()["content_sha256"])

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> PlaceboCheckpoint:
        expected = {
            "backend",
            "blas_environment",
            "checkpoint_every_replications",
            "checkpoint_sequence",
            "content_sha256",
            "cpu_seconds_hex",
            "created_at_utc",
            "elapsed_seconds_hex",
            "execution_binding_id",
            "gpu_device",
            "gpu_memory_used_bytes",
            "heartbeat_seconds",
            "infrastructure_failure_code",
            "last_heartbeat_utc",
            "max_in_flight",
            "owner_id",
            "process_peak_working_set_bytes",
            "protocol_version",
            "request",
            "request_sha256",
            "results",
            "schema_version",
            "source_id_sha256",
            "status",
            "updated_at_utc",
            "worker_plan",
        }
        if set(value) != expected or not verify_content_sha256(value):
            raise PlaceboCheckpointError("placebo checkpoint hash or schema is invalid")
        if (
            value.get("schema_version") != STAGE4_PLACEBO_CHECKPOINT_SCHEMA_VERSION
            or value.get("protocol_version") != STAGE4_PROTOCOL_VERSION
            or value.get("checkpoint_every_replications") != CHECKPOINT_EVERY_REPLICATIONS
            or value.get("heartbeat_seconds") != HEARTBEAT_SECONDS
        ):
            raise PlaceboCheckpointError("placebo checkpoint policy changed")
        request = _mapping(value.get("request"), label="checkpoint request")
        if any(not isinstance(item, str) for item in request.values()):
            raise PlaceboCheckpointError("checkpoint request identity is malformed")
        blas = _mapping(value.get("blas_environment"), label="checkpoint BLAS environment")
        if any(not isinstance(item, str) for item in blas.values()):
            raise PlaceboCheckpointError("checkpoint BLAS environment is malformed")
        raw_results = value.get("results")
        if not isinstance(raw_results, list):
            raise PlaceboCheckpointError("checkpoint results must be a list")
        try:
            return cls(
                execution_binding_id=cast(str, value["execution_binding_id"]),
                request_identity=tuple(sorted(cast(dict[str, str], dict(request)).items())),
                request_sha256=cast(str, value["request_sha256"]),
                source_id_sha256=cast(str, value["source_id_sha256"]),
                worker_plan=_worker_from_mapping(value["worker_plan"]),
                max_in_flight=cast(int, value["max_in_flight"]),
                blas_environment=tuple(sorted(cast(dict[str, str], dict(blas)).items())),
                backend=cast(BackendId, value["backend"]),
                gpu_device=cast(str | None, value["gpu_device"]),
                gpu_memory_used_bytes=cast(int | None, value["gpu_memory_used_bytes"]),
                results=tuple(_completed_from_mapping(item) for item in raw_results),
                status=cast(CheckpointStatus, value["status"]),
                owner_id=cast(str, value["owner_id"]),
                checkpoint_sequence=cast(int, value["checkpoint_sequence"]),
                created_at_utc=cast(str, value["created_at_utc"]),
                updated_at_utc=cast(str, value["updated_at_utc"]),
                last_heartbeat_utc=cast(str, value["last_heartbeat_utc"]),
                elapsed_seconds=_nonnegative_hex(
                    value["elapsed_seconds_hex"], label="elapsed_seconds"
                ),
                cpu_seconds=_nonnegative_hex(value["cpu_seconds_hex"], label="cpu_seconds"),
                process_peak_working_set_bytes=cast(
                    int | None, value["process_peak_working_set_bytes"]
                ),
                infrastructure_failure_code=cast(str | None, value["infrastructure_failure_code"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PlaceboCheckpointError("placebo checkpoint invariants failed") from exc


def load_placebo_checkpoint(path: Path) -> PlaceboCheckpoint:
    try:
        payload, _ = read_existing_immutable_file(
            Path(path),
            label="placebo checkpoint",
        )
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnsafeImmutableFileError, UnicodeError, json.JSONDecodeError) as exc:
        raise PlaceboCheckpointError("cannot read placebo checkpoint") from exc
    return PlaceboCheckpoint.from_mapping(_mapping(value, label="placebo checkpoint"))


def _directory_identity(value: os.stat_result) -> tuple[int, int]:
    return value.st_dev, value.st_ino


def _safe_unlink_entry(path: Path) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        return


def _write_checkpoint_atomic(path: Path, checkpoint: PlaceboCheckpoint) -> None:
    target = Path(os.path.abspath(os.fspath(path)))
    serialized = canonical_json_bytes(checkpoint.as_mapping()) + b"\n"
    try:
        ensure_real_directory_tree(
            Path(target.anchor) if target.anchor else Path.cwd(),
            target.parent,
            label="placebo checkpoint parent directory",
        )
        parent_before = require_existing_real_directory(
            target.parent,
            label="placebo checkpoint parent directory",
        )
        try:
            os.lstat(target)
        except FileNotFoundError:
            pass
        else:
            # Reject symlinks, reparse points, hardlinks, and concurrent mutation
            # before replacing a mutable checkpoint entry.
            read_existing_immutable_file(target, label="existing placebo checkpoint")
    except (OSError, UnsafeImmutableFileError) as exc:
        raise PlaceboCheckpointError("unsafe existing placebo checkpoint") from exc
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        temporary = Path(temporary_name)
        staged_payload, staged_snapshot = read_existing_immutable_file(
            temporary,
            label="staged placebo checkpoint",
        )
        if staged_payload != serialized:
            raise PlaceboCheckpointError("staged placebo checkpoint changed")
        os.replace(temporary_name, target)
        temporary_name = None
        observed, installed_snapshot = read_existing_immutable_file(
            target,
            label="installed placebo checkpoint",
        )
        parent_after = require_existing_real_directory(
            target.parent,
            label="placebo checkpoint parent directory",
        )
        if (
            observed != serialized
            or (installed_snapshot.device, installed_snapshot.inode)
            != (staged_snapshot.device, staged_snapshot.inode)
            or _directory_identity(parent_after) != _directory_identity(parent_before)
        ):
            raise PlaceboCheckpointError(
                "installed placebo checkpoint failed identity verification"
            )
    except (OSError, UnsafeImmutableFileError) as exc:
        if temporary_name is not None:
            _safe_unlink_entry(Path(temporary_name))
        raise PlaceboCheckpointError("cannot safely update placebo checkpoint") from exc
    except Exception:
        if temporary_name is not None:
            _safe_unlink_entry(Path(temporary_name))
        raise


def _failure_code(exc: BaseException) -> str:
    name = re.sub(r"(?<!^)(?=[A-Z])", "_", type(exc).__name__).casefold()
    normalized = re.sub(r"[^a-z0-9_]", "_", name)[:96]
    return normalized if _FAILURE_PATTERN.fullmatch(normalized) is not None else "runtime_failure"


@dataclass(slots=True)
class _RuntimeSession:
    runtime: PlaceboRuntime
    request: PlaceboRequest
    source: PlaceboSource
    path: Path
    checkpoint: PlaceboCheckpoint | None
    owner_id: str
    results: list[CompletedPlaceboReplication] = field(init=False)
    lock: threading.Lock = field(init=False, default_factory=threading.Lock)
    stop: threading.Event = field(init=False, default_factory=threading.Event)
    thread: threading.Thread | None = field(init=False, default=None)
    wall_start: float = field(init=False)
    cpu_start: float = field(init=False)

    def __post_init__(self) -> None:
        self.results = [] if self.checkpoint is None else list(self.checkpoint.results)
        self.wall_start = self.runtime.seconds_clock()
        self.cpu_start = self.runtime.cpu_clock()

    def _checkpoint(
        self,
        status: CheckpointStatus,
        *,
        failure_code: str | None,
    ) -> PlaceboCheckpoint:
        previous = self.checkpoint
        now = _timestamp(self.runtime.clock())
        peak = self.runtime.memory_probe()
        prior_peak = None if previous is None else previous.process_peak_working_set_bytes
        known_peaks = [value for value in (peak, prior_peak) if value is not None]
        gpu_memory = self.runtime.gpu_memory_probe()
        if gpu_memory is not None and (
            isinstance(gpu_memory, bool) or not isinstance(gpu_memory, int) or gpu_memory < 0
        ):
            raise PlaceboCheckpointError("GPU memory probe returned an invalid value")
        return PlaceboCheckpoint(
            execution_binding_id=self.runtime.execution_binding_id,
            request_identity=tuple(
                sorted((key, cast(str, value)) for key, value in self.request.as_mapping().items())
            ),
            request_sha256=self.request.content_sha256,
            source_id_sha256=self.source.source_id_sha256,
            worker_plan=self.runtime.worker_plan,
            max_in_flight=self.runtime.max_in_flight,
            blas_environment=tuple(sorted(self.runtime.worker_plan.blas_environment().items())),
            backend=self.runtime.backend,
            gpu_device=self.runtime.gpu_device,
            gpu_memory_used_bytes=gpu_memory,
            results=tuple(self.results),
            status=status,
            owner_id=self.owner_id,
            checkpoint_sequence=(0 if previous is None else previous.checkpoint_sequence) + 1,
            created_at_utc=now if previous is None else previous.created_at_utc,
            updated_at_utc=now,
            last_heartbeat_utc=now,
            elapsed_seconds=(0.0 if previous is None else previous.elapsed_seconds)
            + max(0.0, self.runtime.seconds_clock() - self.wall_start),
            cpu_seconds=(0.0 if previous is None else previous.cpu_seconds)
            + max(0.0, self.runtime.cpu_clock() - self.cpu_start),
            process_peak_working_set_bytes=max(known_peaks) if known_peaks else None,
            infrastructure_failure_code=failure_code,
        )

    def _write(self, status: CheckpointStatus, *, failure_code: str | None = None) -> None:
        checkpoint = self._checkpoint(status, failure_code=failure_code)
        _write_checkpoint_atomic(self.path, checkpoint)
        self.checkpoint = checkpoint
        self.wall_start = self.runtime.seconds_clock()
        self.cpu_start = self.runtime.cpu_clock()

    def _heartbeat(self) -> None:
        while not self.stop.wait(HEARTBEAT_SECONDS):
            with self.lock:
                if not self.stop.is_set():
                    self._write("running")

    def start(self) -> None:
        with self.lock:
            status: CheckpointStatus = (
                "completed" if len(self.results) == PERMUTATION_REPLICATIONS else "running"
            )
            self._write(status)
        if status != "completed":
            self.thread = threading.Thread(
                target=self._heartbeat,
                name=f"stage4-placebo-heartbeat-{self.request.kind}-{self.request.model_variant}",
                daemon=True,
            )
            self.thread.start()

    def on_result(
        self,
        request: PlaceboRequest,
        result: CompletedPlaceboReplication,
    ) -> None:
        if request != self.request:
            raise PlaceboCheckpointError("result callback used another placebo request")
        with self.lock:
            index = result.replication_index
            if index < len(self.results):
                if self.results[index] != result:
                    raise PlaceboCheckpointError("completed placebo result cannot be replaced")
                return
            if index != len(self.results):
                raise PlaceboCheckpointError("completed placebo result cannot skip an index")
            self.results.append(result)
            if len(self.results) % CHECKPOINT_EVERY_REPLICATIONS == 0:
                self._write("running")

    def _stop_thread(self) -> None:
        self.stop.set()
        thread = self.thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)

    def on_complete(
        self,
        request: PlaceboRequest,
        results: tuple[CompletedPlaceboReplication, ...],
    ) -> None:
        if request != self.request or results != tuple(self.results):
            raise PlaceboCheckpointError("completion callback changed recovered results")
        if len(results) != PERMUTATION_REPLICATIONS:
            raise PlaceboCheckpointError("completion callback requires all 1000 results")
        self._stop_thread()
        with self.lock:
            self._write("completed")

    def on_interruption(self, request: PlaceboRequest, exc: BaseException) -> None:
        if request != self.request:
            raise PlaceboCheckpointError("interruption callback used another placebo request")
        self._stop_thread()
        with self.lock:
            self._write("interrupted", failure_code=_failure_code(exc))


@dataclass(frozen=True, slots=True)
class PlaceboRuntime:
    """Build request-specific lazy executions backed by atomic local checkpoints."""

    checkpoint_directory: Path
    execution_binding_id: str
    source_factory: PlaceboSourceFactory = field(repr=False, compare=False)
    worker_plan: Stage4WorkerPlan
    max_in_flight: int
    backend: BackendId = "cpu_float64"
    gpu_device: str | None = None
    memory_probe: MemoryProbe = field(
        default=process_peak_working_set_bytes,
        repr=False,
        compare=False,
    )
    gpu_memory_probe: MemoryProbe = field(default=lambda: None, repr=False, compare=False)
    clock: Clock = field(default=lambda: datetime.now(UTC), repr=False, compare=False)
    seconds_clock: SecondsClock = field(default=time.perf_counter, repr=False, compare=False)
    cpu_clock: SecondsClock = field(default=time.process_time, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.worker_plan, Stage4WorkerPlan):
            raise TypeError("placebo runtime requires a Stage4WorkerPlan")
        _sha256(self.execution_binding_id, label="execution_binding_id")
        if not callable(self.source_factory):
            raise TypeError("placebo runtime source_factory must be callable")
        if (
            self.worker_plan.reserve_physical_cores < 2
            or self.worker_plan.configured_max_workers > 12
            or self.worker_plan.blas_threads_per_worker != 1
            or self.worker_plan.nested_parallelism
        ):
            raise ValueError("placebo runtime violates the frozen worker policy")
        if (
            not isinstance(self.max_in_flight, int)
            or isinstance(self.max_in_flight, bool)
            or not 1 <= self.max_in_flight <= self.worker_plan.effective_workers
        ):
            raise ValueError("placebo runtime max_in_flight must be within the worker count")
        if self.backend not in {"cpu_float64", "gpu_float64"}:
            raise ValueError("placebo runtime backend is invalid")
        if self.gpu_device is not None and (
            not self.gpu_device or self.gpu_device != self.gpu_device.strip()
        ):
            raise ValueError("placebo runtime GPU device is malformed")
        for callback in (
            self.memory_probe,
            self.gpu_memory_probe,
            self.clock,
            self.seconds_clock,
            self.cpu_clock,
        ):
            if not callable(callback):
                raise TypeError("placebo runtime probe/clock must be callable")
        object.__setattr__(self, "checkpoint_directory", Path(self.checkpoint_directory))

    def checkpoint_path_for(self, request: PlaceboRequest) -> Path:
        return self.checkpoint_directory / (
            f"{request.kind}-{request.model_variant}-permutations.json"
        )

    def _resume_checkpoint(
        self,
        path: Path,
        request: PlaceboRequest,
        source: PlaceboSource,
    ) -> PlaceboCheckpoint | None:
        try:
            os.lstat(path)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise PlaceboCheckpointError("cannot inspect placebo checkpoint") from exc
        checkpoint = load_placebo_checkpoint(path)
        if checkpoint.execution_binding_id != self.execution_binding_id:
            raise PlaceboCheckpointError("checkpoint uses another execution binding")
        if (
            checkpoint.request_sha256 != request.content_sha256
            or dict(checkpoint.request_identity) != request.as_mapping()
        ):
            raise PlaceboCheckpointError("checkpoint uses another placebo request")
        if checkpoint.source_id_sha256 != source.source_id_sha256:
            raise PlaceboCheckpointError("checkpoint uses another lazy source identity")
        if checkpoint.worker_plan != self.worker_plan:
            raise PlaceboCheckpointError("checkpoint uses another worker plan")
        if checkpoint.max_in_flight != self.max_in_flight:
            raise PlaceboCheckpointError("checkpoint uses another sealed max_in_flight")
        if checkpoint.backend != self.backend or checkpoint.gpu_device != self.gpu_device:
            raise PlaceboCheckpointError("checkpoint uses another backend identity")
        if checkpoint.status == "running":
            age = (
                self.clock().astimezone(UTC)
                - _parse_timestamp(
                    checkpoint.last_heartbeat_utc,
                    label="last_heartbeat_utc",
                )
            ).total_seconds()
            if age < HEARTBEAT_STALE_AFTER_SECONDS:
                raise PlaceboCheckpointError(
                    "running checkpoint still has a live heartbeat; concurrent resume forbidden"
                )
        return checkpoint

    def __call__(self, request: PlaceboRequest) -> PlaceboExecution:
        if not isinstance(request, PlaceboRequest):
            raise TypeError("placebo runtime requires PlaceboRequest")
        source = self.source_factory(request)
        if not isinstance(source, PlaceboSource):
            raise TypeError("placebo source_factory must return PlaceboSource")
        source.validate_for(request)
        for name, value in self.worker_plan.blas_environment().items():
            os.environ[name] = value
        path = self.checkpoint_path_for(request)
        checkpoint = self._resume_checkpoint(path, request, source)
        session = _RuntimeSession(
            runtime=self,
            request=request,
            source=source,
            path=path,
            checkpoint=checkpoint,
            owner_id=f"pid-{os.getpid()}-{uuid.uuid4().hex}",
        )
        session.start()
        return PlaceboExecution(
            source=source,
            completed_results=() if checkpoint is None else checkpoint.results,
            worker_plan=self.worker_plan,
            max_in_flight=self.max_in_flight,
            on_result=session.on_result,
            on_complete=session.on_complete,
            on_interruption=session.on_interruption,
        )


def build_placebo_runtime_injection(runtime: PlaceboRuntime) -> PlaceboInjection:
    if not isinstance(runtime, PlaceboRuntime):
        raise TypeError("runtime must be PlaceboRuntime")
    return runtime


__all__ = [
    "CHECKPOINT_EVERY_REPLICATIONS",
    "HEARTBEAT_SECONDS",
    "HEARTBEAT_STALE_AFTER_SECONDS",
    "STAGE4_PLACEBO_CHECKPOINT_SCHEMA_VERSION",
    "PlaceboCheckpoint",
    "PlaceboCheckpointError",
    "PlaceboRuntime",
    "PlaceboSourceFactory",
    "build_placebo_runtime_injection",
    "load_placebo_checkpoint",
    "process_peak_working_set_bytes",
]
