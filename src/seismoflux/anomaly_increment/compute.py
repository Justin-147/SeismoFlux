"""Deterministic stage-4 compute planning and optional-backend qualification.

The accepted formal backend is CPU ``float64``.  This module exposes an explicit
backend boundary so a GPU implementation can be qualified before any target read,
but it never imports a system-wide GPU package or silently changes backend.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Final, Generic, Literal, Protocol, TypeVar, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from seismoflux.anomaly_increment.config import Stage4ProtocolBundle

BackendId = Literal["cpu_float64", "gpu_float64"]
FloatArray = NDArray[np.float64]
T = TypeVar("T")
R = TypeVar("R")

CPU_BACKEND_ID: Final[BackendId] = "cpu_float64"
GPU_BACKEND_ID: Final[BackendId] = "gpu_float64"


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _positive_integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _finite_nonnegative(value: float, *, label: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return numeric


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{label} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be finite")
    return numeric


@dataclass(frozen=True, slots=True)
class Stage4WorkerPlan:
    """One reserved-core, non-nested worker plan."""

    physical_cores: int
    logical_processors: int
    reserve_physical_cores: int
    configured_max_workers: int
    effective_workers: int
    blas_threads_per_worker: int
    nested_parallelism: bool

    def __post_init__(self) -> None:
        physical = _positive_integer(self.physical_cores, label="physical_cores")
        logical = _positive_integer(self.logical_processors, label="logical_processors")
        reserve = _positive_integer(self.reserve_physical_cores, label="reserve_physical_cores")
        maximum = _positive_integer(self.configured_max_workers, label="configured_max_workers")
        _positive_integer(self.blas_threads_per_worker, label="blas_threads_per_worker")
        if reserve >= physical:
            raise ValueError("reserved physical cores must be fewer than detected cores")
        expected = min(maximum, max(1, physical - reserve))
        if self.effective_workers != expected:
            raise ValueError("effective stage-4 workers violate the reserved-core rule")
        if logical < physical:
            raise ValueError("logical processor count cannot be below physical cores")
        if self.nested_parallelism:
            raise ValueError("nested stage-4 parallelism is forbidden")

    def blas_environment(self) -> dict[str, str]:
        value = str(self.blas_threads_per_worker)
        return {
            "BLIS_NUM_THREADS": value,
            "MKL_NUM_THREADS": value,
            "NUMEXPR_NUM_THREADS": value,
            "OMP_NUM_THREADS": value,
            "OPENBLAS_NUM_THREADS": value,
            "VECLIB_MAXIMUM_THREADS": value,
        }


@dataclass(frozen=True, slots=True)
class BackendEquivalenceEvidence:
    """Target-blind CPU/GPU numerical comparison frozen before scoring."""

    candidate_backend: BackendId
    objective_relative_error: float
    gradient_max_abs_error: float
    coefficient_max_abs_error: float
    integrated_intensity_relative_error: float
    random_mapping_byte_identity: bool
    scientific_decision_identity: bool
    repeated_run_identity: bool
    worker_count_identity: bool

    def __post_init__(self) -> None:
        if self.candidate_backend != GPU_BACKEND_ID:
            raise ValueError("only gpu_float64 needs optional-backend qualification")
        for label, value in (
            ("objective_relative_error", self.objective_relative_error),
            ("gradient_max_abs_error", self.gradient_max_abs_error),
            ("coefficient_max_abs_error", self.coefficient_max_abs_error),
            ("integrated_intensity_relative_error", self.integrated_intensity_relative_error),
        ):
            _finite_nonnegative(value, label=label)

    def passes(self, protocol: Stage4ProtocolBundle) -> bool:
        compute = _mapping(protocol.protocol.get("compute"), label="compute")
        tolerances = _mapping(compute.get("gpu_equivalence"), label="gpu_equivalence")
        return (
            self.objective_relative_error
            <= _finite_number(
                tolerances["objective_relative_tolerance"],
                label="objective_relative_tolerance",
            )
            and self.gradient_max_abs_error
            <= _finite_number(
                tolerances["gradient_max_abs_tolerance"],
                label="gradient_max_abs_tolerance",
            )
            and self.coefficient_max_abs_error
            <= _finite_number(
                tolerances["coefficient_max_abs_tolerance"],
                label="coefficient_max_abs_tolerance",
            )
            and self.integrated_intensity_relative_error
            <= _finite_number(
                tolerances["integrated_intensity_relative_tolerance"],
                label="integrated_intensity_relative_tolerance",
            )
            and self.random_mapping_byte_identity
            and self.scientific_decision_identity
            and self.repeated_run_identity
            and self.worker_count_identity
        )

    def content_sha256(self) -> str:
        payload = {
            "candidate_backend": self.candidate_backend,
            "coefficient_max_abs_error_hex": float(self.coefficient_max_abs_error).hex(),
            "gradient_max_abs_error_hex": float(self.gradient_max_abs_error).hex(),
            "integrated_intensity_relative_error_hex": float(
                self.integrated_intensity_relative_error
            ).hex(),
            "objective_relative_error_hex": float(self.objective_relative_error).hex(),
            "random_mapping_byte_identity": self.random_mapping_byte_identity,
            "repeated_run_identity": self.repeated_run_identity,
            "scientific_decision_identity": self.scientific_decision_identity,
            "worker_count_identity": self.worker_count_identity,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class Stage4ComputePlan:
    """Backend and resources that become part of the execution seal."""

    backend: BackendId
    workers: Stage4WorkerPlan
    gpu_equivalence_sha256: str | None
    gpu_fallback_reason: str | None

    def __post_init__(self) -> None:
        if self.backend == CPU_BACKEND_ID:
            if self.gpu_equivalence_sha256 is not None:
                raise ValueError("CPU formal plan must not claim accepted GPU evidence")
        elif self.backend == GPU_BACKEND_ID:
            if self.gpu_equivalence_sha256 is None or self.gpu_fallback_reason is not None:
                raise ValueError(
                    "GPU formal plan requires accepted evidence and no fallback reason"
                )
        else:
            raise ValueError("unknown stage-4 compute backend")


class Float64Backend(Protocol):
    """Small numeric surface used by the point-process implementation."""

    backend_id: BackendId

    def as_float64(self, values: ArrayLike) -> FloatArray: ...

    def matvec(self, matrix: ArrayLike, vector: ArrayLike) -> FloatArray: ...

    def exp(self, values: ArrayLike) -> FloatArray: ...


@dataclass(frozen=True, slots=True)
class NumpyFloat64Backend:
    """Authoritative deterministic NumPy implementation."""

    backend_id: BackendId = CPU_BACKEND_ID

    def as_float64(self, values: ArrayLike) -> FloatArray:
        array = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(array)):
            raise ValueError("stage-4 numeric arrays must contain only finite values")
        return array

    def matvec(self, matrix: ArrayLike, vector: ArrayLike) -> FloatArray:
        left = self.as_float64(matrix)
        right = self.as_float64(vector)
        if left.ndim != 2 or right.ndim != 1 or left.shape[1] != right.shape[0]:
            raise ValueError("stage-4 matrix-vector shapes are incompatible")
        result = np.asarray(left @ right, dtype=np.float64)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("stage-4 matrix-vector result is nonfinite")
        return result

    def exp(self, values: ArrayLike) -> FloatArray:
        array = self.as_float64(values)
        with np.errstate(over="raise", invalid="raise"):
            result = np.asarray(np.exp(array), dtype=np.float64)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("stage-4 exponential result is nonfinite")
        return result


def build_compute_plan(
    protocol: Stage4ProtocolBundle,
    *,
    requested_backend: BackendId = CPU_BACKEND_ID,
    gpu_evidence: BackendEquivalenceEvidence | None = None,
    detected_physical_cores: int | None = None,
    detected_logical_processors: int | None = None,
) -> Stage4ComputePlan:
    """Resolve resources without importing or probing an unpinned GPU runtime."""

    compute = _mapping(protocol.protocol.get("compute"), label="compute")
    configured_physical = _positive_integer(
        compute.get("physical_cores_expected"), label="physical expected"
    )
    detected_logical = (
        os.cpu_count() if detected_logical_processors is None else detected_logical_processors
    )
    logical = (
        _positive_integer(compute.get("logical_processors_observed"), label="logical observed")
        if detected_logical is None
        else _positive_integer(detected_logical, label="detected_logical_processors")
    )
    physical = (
        min(configured_physical, logical)
        if detected_physical_cores is None
        else _positive_integer(detected_physical_cores, label="detected_physical_cores")
    )
    workers = Stage4WorkerPlan(
        physical_cores=physical,
        logical_processors=logical,
        reserve_physical_cores=_positive_integer(
            compute.get("reserve_physical_cores"), label="reserve_physical_cores"
        ),
        configured_max_workers=_positive_integer(compute.get("max_workers"), label="max_workers"),
        effective_workers=min(
            _positive_integer(compute.get("max_workers"), label="max_workers"),
            max(
                1,
                physical
                - _positive_integer(
                    compute.get("reserve_physical_cores"), label="reserve_physical_cores"
                ),
            ),
        ),
        blas_threads_per_worker=_positive_integer(
            compute.get("blas_threads_per_worker"), label="blas_threads_per_worker"
        ),
        nested_parallelism=bool(compute.get("nested_parallelism")),
    )
    gpu_reference = _mapping(
        compute.get("gpu_available_reference"), label="gpu_available_reference"
    )
    if requested_backend == CPU_BACKEND_ID:
        return Stage4ComputePlan(
            backend=CPU_BACKEND_ID,
            workers=workers,
            gpu_equivalence_sha256=None,
            gpu_fallback_reason="project_environment_has_no_frozen_gpu_backend",
        )
    if gpu_reference.get("python_gpu_backend_installed_at_freeze") is not True:
        return Stage4ComputePlan(
            backend=CPU_BACKEND_ID,
            workers=workers,
            gpu_equivalence_sha256=None,
            gpu_fallback_reason="project_environment_has_no_frozen_gpu_backend",
        )
    if gpu_evidence is None or not gpu_evidence.passes(protocol):
        return Stage4ComputePlan(
            backend=CPU_BACKEND_ID,
            workers=workers,
            gpu_equivalence_sha256=None,
            gpu_fallback_reason="gpu_equivalence_not_passed_before_scoring_code_freeze",
        )
    return Stage4ComputePlan(
        backend=GPU_BACKEND_ID,
        workers=workers,
        gpu_equivalence_sha256=gpu_evidence.content_sha256(),
        gpu_fallback_reason=None,
    )


def stable_parallel_map(
    function: Callable[[T], R],
    values: Sequence[T],
    *,
    workers: int,
) -> tuple[R, ...]:
    """Evaluate independent work in input order, never completion order."""

    count = _positive_integer(workers, label="workers")
    if count == 1 or len(values) <= 1:
        return tuple(map(function, values))
    with ThreadPoolExecutor(max_workers=count) as executor:
        return tuple(executor.map(function, values))


@dataclass(frozen=True, slots=True)
class WorkerInvarianceEvidence(Generic[R]):
    """Exact digest evidence for one target-blind worker-count comparison."""

    worker_counts: tuple[int, ...]
    result_digests: tuple[str, ...]
    reference_results: tuple[R, ...]

    @property
    def passed(self) -> bool:
        return bool(self.result_digests) and len(set(self.result_digests)) == 1


def _canonical_result_digest(values: Sequence[object]) -> str:
    def normalize(value: object) -> object:
        if isinstance(value, np.ndarray):
            array = np.asarray(value)
            return {
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "bytes_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
            }
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("worker-invariance results must be finite")
            return {"float_hex": value.hex()}
        if isinstance(value, str | int | bool) or value is None:
            return value
        if isinstance(value, Mapping):
            return {str(key): normalize(item) for key, item in sorted(value.items())}
        if isinstance(value, Sequence) and not isinstance(value, str | bytes):
            return [normalize(item) for item in value]
        raise TypeError(f"unsupported worker-invariance result type: {type(value)!r}")

    raw = json.dumps(normalize(list(values)), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def qualify_worker_invariance(
    function: Callable[[T], R],
    values: Sequence[T],
    *,
    worker_counts: Sequence[int] = (1, 2, 4, 12),
) -> WorkerInvarianceEvidence[R]:
    """Run the same predeclared work with each frozen worker count."""

    counts = tuple(_positive_integer(value, label="worker_count") for value in worker_counts)
    if not counts or len(set(counts)) != len(counts):
        raise ValueError("worker-count qualification requires unique counts")
    all_results = tuple(stable_parallel_map(function, values, workers=count) for count in counts)
    digests = tuple(
        _canonical_result_digest(cast(Sequence[object], result)) for result in all_results
    )
    return WorkerInvarianceEvidence(
        worker_counts=counts,
        result_digests=digests,
        reference_results=all_results[0],
    )


__all__ = [
    "CPU_BACKEND_ID",
    "GPU_BACKEND_ID",
    "BackendEquivalenceEvidence",
    "BackendId",
    "Float64Backend",
    "NumpyFloat64Backend",
    "Stage4ComputePlan",
    "Stage4WorkerPlan",
    "WorkerInvarianceEvidence",
    "build_compute_plan",
    "qualify_worker_invariance",
    "stable_parallel_map",
]
