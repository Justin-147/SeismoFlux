"""Pre-import resource controls for the one-shot Stage 2S formal runner.

This module intentionally uses only the Python standard library.  The formal
launcher imports and executes it before importing NumPy, PyArrow, SciPy, or the
Stage 2S production module.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path

FORMAL_WORKER_COUNT = 1
MINIMUM_RESERVED_PHYSICAL_CORES = 2
MINIMUM_PHYSICAL_CORE_COUNT = FORMAL_WORKER_COUNT + MINIMUM_RESERVED_PHYSICAL_CORES
THREAD_LIMIT_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
_PREIMPORT_FORBIDDEN_MODULE_ROOTS = (
    "numpy",
    "pyarrow",
    "scipy",
    "pandas",
    "matplotlib",
    "seismoflux.stage2s.production",
)

PhysicalCoreProbe = Callable[[], int | None]


class Stage2SExecutionEnvironmentError(RuntimeError):
    """Raised when the frozen formal resource contract cannot be proven."""


@dataclass(frozen=True, slots=True)
class FormalExecutionEnvironmentEvidence:
    """Immutable evidence that the formal process was constrained before imports."""

    worker_count: int
    physical_core_count: int
    reserved_physical_core_count: int
    thread_limits: tuple[tuple[str, str], ...]
    configured_before_numeric_imports: bool
    evidence_sha256: str

    def receipt_bindings(self) -> dict[str, object]:
        """Return the JSON-safe fragment that production should bind."""

        return {
            "workers": self.worker_count,
            "physical_core_count": self.physical_core_count,
            "reserved_physical_cores_minimum": self.reserved_physical_core_count,
            "available_physical_cores_after_reservation": (
                self.physical_core_count - self.reserved_physical_core_count
            ),
            "thread_environment": dict(self.thread_limits),
            "configured_before_numeric_imports": self.configured_before_numeric_imports,
            "evidence_sha256": self.evidence_sha256,
        }


_PREPARED_EVIDENCE: FormalExecutionEnvironmentEvidence | None = None


def _loaded_forbidden_modules(module_names: Iterable[str]) -> tuple[str, ...]:
    observed = {
        root
        for name in module_names
        for root in _PREIMPORT_FORBIDDEN_MODULE_ROOTS
        if name == root or name.startswith(f"{root}.")
    }
    return tuple(sorted(observed))


def _windows_physical_core_count() -> int | None:
    win_dll_factory = getattr(ctypes, "WinDLL", None)
    get_last_error = getattr(ctypes, "get_last_error", None)
    if win_dll_factory is None or get_last_error is None:
        return None
    try:
        kernel32 = win_dll_factory("kernel32", use_last_error=True)
        query = kernel32.GetLogicalProcessorInformationEx
    except (AttributeError, OSError):
        return None
    query.argtypes = [
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    query.restype = ctypes.c_int
    relation_processor_core = 0
    required_bytes = ctypes.c_uint32(0)
    first_result = query(
        relation_processor_core,
        None,
        ctypes.byref(required_bytes),
    )
    error_insufficient_buffer = 122
    if first_result or int(get_last_error()) != error_insufficient_buffer:
        return None
    if required_bytes.value < 8:
        return None
    buffer = (ctypes.c_ubyte * required_bytes.value)()
    if not query(
        relation_processor_core,
        ctypes.byref(buffer),
        ctypes.byref(required_bytes),
    ):
        return None
    offset = 0
    physical_cores = 0
    while offset < required_bytes.value:
        if required_bytes.value - offset < 8:
            return None
        relationship = int.from_bytes(bytes(buffer[offset : offset + 4]), "little")
        record_size = int.from_bytes(bytes(buffer[offset + 4 : offset + 8]), "little")
        if record_size < 8 or offset + record_size > required_bytes.value:
            return None
        if relationship == relation_processor_core:
            physical_cores += 1
        offset += record_size
    return physical_cores or None


def _linux_physical_core_count() -> int | None:
    cpuinfo = Path("/proc/cpuinfo")
    try:
        payload = cpuinfo.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None
    identities: set[tuple[str, str]] = set()
    for block in re.split(r"\n\s*\n", payload.strip()):
        fields: dict[str, str] = {}
        for line in block.splitlines():
            key, separator, value = line.partition(":")
            if separator:
                fields[key.strip()] = value.strip()
        physical_id = fields.get("physical id")
        core_id = fields.get("core id")
        if physical_id is not None and core_id is not None:
            identities.add((physical_id, core_id))
    return len(identities) or None


def _macos_physical_core_count() -> int | None:
    try:
        completed = subprocess.run(
            ("sysctl", "-n", "hw.physicalcpu"),
            check=True,
            capture_output=True,
            text=True,
            encoding="ascii",
            errors="strict",
        )
        value = int(completed.stdout.strip())
    except (OSError, subprocess.CalledProcessError, UnicodeError, ValueError):
        return None
    return value if value > 0 else None


def detect_physical_core_count() -> int | None:
    """Return a physical-core count using only platform/stdlib facilities."""

    if os.name == "nt":
        return _windows_physical_core_count()
    if sys.platform.startswith("linux"):
        return _linux_physical_core_count()
    if sys.platform == "darwin":
        return _macos_physical_core_count()
    return None


def _evidence_sha256(
    *,
    physical_core_count: int,
    thread_limits: tuple[tuple[str, str], ...],
) -> str:
    payload = {
        "workers": FORMAL_WORKER_COUNT,
        "physical_core_count": physical_core_count,
        "reserved_physical_cores_minimum": MINIMUM_RESERVED_PHYSICAL_CORES,
        "thread_environment": dict(thread_limits),
        "configured_before_numeric_imports": True,
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def prepare_formal_execution_environment(
    *,
    environ: MutableMapping[str, str] | None = None,
    loaded_module_names: Iterable[str] | None = None,
    physical_core_probe: PhysicalCoreProbe | None = None,
) -> FormalExecutionEnvironmentEvidence:
    """Force frozen thread limits and prove that at least two cores remain free."""

    global _PREPARED_EVIDENCE

    environment = os.environ if environ is None else environ
    module_names = tuple(sys.modules) if loaded_module_names is None else tuple(loaded_module_names)
    forbidden = _loaded_forbidden_modules(module_names)
    if forbidden:
        raise Stage2SExecutionEnvironmentError(
            "formal resource controls were requested after numeric imports: " + ", ".join(forbidden)
        )
    for variable in THREAD_LIMIT_ENVIRONMENT_VARIABLES:
        environment[variable] = "1"
    probe = detect_physical_core_count if physical_core_probe is None else physical_core_probe
    try:
        physical_core_count = probe()
    except Exception as exc:
        raise Stage2SExecutionEnvironmentError(
            "physical-core detection failed; formal execution is closed"
        ) from exc
    if type(physical_core_count) is not int or physical_core_count < MINIMUM_PHYSICAL_CORE_COUNT:
        raise Stage2SExecutionEnvironmentError(
            "at least three verified physical cores are required to reserve two"
        )
    thread_limits = tuple(
        (variable, environment[variable]) for variable in THREAD_LIMIT_ENVIRONMENT_VARIABLES
    )
    evidence = FormalExecutionEnvironmentEvidence(
        worker_count=FORMAL_WORKER_COUNT,
        physical_core_count=physical_core_count,
        reserved_physical_core_count=MINIMUM_RESERVED_PHYSICAL_CORES,
        thread_limits=thread_limits,
        configured_before_numeric_imports=True,
        evidence_sha256=_evidence_sha256(
            physical_core_count=physical_core_count,
            thread_limits=thread_limits,
        ),
    )
    if environ is None and loaded_module_names is None and physical_core_probe is None:
        _PREPARED_EVIDENCE = evidence
    return evidence


def require_prepared_formal_execution_environment(
    *,
    environ: Mapping[str, str] | None = None,
) -> FormalExecutionEnvironmentEvidence:
    """Return the launch evidence or fail before production is imported."""

    environment = os.environ if environ is None else environ
    evidence = _PREPARED_EVIDENCE
    if evidence is None:
        raise Stage2SExecutionEnvironmentError(
            "formal execution must start through scripts/run_stage2s_once.py"
        )
    for variable in THREAD_LIMIT_ENVIRONMENT_VARIABLES:
        if environment.get(variable) != "1":
            raise Stage2SExecutionEnvironmentError(
                f"formal thread limit changed after launch: {variable}"
            )
    if evidence.worker_count != FORMAL_WORKER_COUNT:
        raise Stage2SExecutionEnvironmentError("formal worker count evidence changed")
    if evidence.physical_core_count - evidence.reserved_physical_core_count < FORMAL_WORKER_COUNT:
        raise Stage2SExecutionEnvironmentError("formal physical-core reservation is invalid")
    return evidence


__all__ = [
    "FORMAL_WORKER_COUNT",
    "MINIMUM_PHYSICAL_CORE_COUNT",
    "MINIMUM_RESERVED_PHYSICAL_CORES",
    "THREAD_LIMIT_ENVIRONMENT_VARIABLES",
    "FormalExecutionEnvironmentEvidence",
    "Stage2SExecutionEnvironmentError",
    "detect_physical_core_count",
    "prepare_formal_execution_environment",
    "require_prepared_formal_execution_environment",
]
