"""Recoverable observed-value runner for the preregistered D1 replay.

The runner is intentionally limited to the six observed models.  It does not
run either placebo family, bootstrap uncertainty, visualisation, a prospective
issue, or the locked test.  Every learnt quantity is selected within an outer
fit fold, and every reported assessment score is produced only after the fold's
parameters have been frozen and refitted on all of its M4+ fit issues.
"""

# ruff: noqa: E402, I001

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from functools import wraps
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, ParamSpec, TypeVar, cast

# Set the process boundary before importing NumPy/SciPy-backed D1 modules.  The
# observed replay is serial; ``workers`` is recorded for the later placebo
# stage and is never used here to fan out BLAS work.
_BLAS_THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)
for _environment_name in _BLAS_THREAD_ENV:
    os.environ[_environment_name] = "1"

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from numpy.typing import NDArray

try:
    from threadpoolctl import threadpool_limits as _threadpool_limits  # type: ignore[import-not-found]
except ImportError:  # Optional: native OpenBLAS/MKL setters below are the fail-closed fallback.
    _threadpool_limits = None

from seismoflux.d1_replay.evaluation import (
    D1_AREA_BUDGETS_KM2,
    D1_FOLD_IDS,
    D1_HORIZONS_DAYS,
    D1_MODEL_ORDER,
    D1ClusterModelOutcome,
    D1IssueAlarmOutcome,
)
from seismoflux.d1_replay.features import (
    D1_SOURCE_COLUMNS,
    D1FeatureContract,
    D1GroupPreprocessor,
    D1IssueFeatures,
    D1StaticGrid,
    load_d1_feature_contract,
)
from seismoflux.d1_replay.model import (
    ConditionalSpatialRidgeFit,
    ConditionalSpatialTrainingIssue,
    fit_conditional_spatial_ridge,
)
from seismoflux.d1_replay.protocol import (
    D1Protocol,
    EXPECTED_MODEL_IDS,
    load_d1_protocol,
    sha256_file,
)
from seismoflux.d1_replay.spatial import (
    FROZEN_R30_ALPHA_CANDIDATES,
    D1AlarmPrefix,
    D1CausalBackground,
    D1SpatialDomain,
    build_causal_background_components,
    build_d1_spatial_domain_from_bytes,
    select_alarm_prefixes,
)
from seismoflux.d1_replay.targets import (
    D1TargetLayer,
    TargetSet,
    build_score_blind_target_layer,
)
from seismoflux.data.common import canonical_json_bytes, write_json_atomic
from seismoflux.stage2s.catalog import (
    Stage2SEarthquakeCatalog,
    parse_frozen_catalog_bytes,
)
from seismoflux.stage2s.contracts import MASS_SUM_ABSOLUTE_TOLERANCE, SpatialGrid
from seismoflux.background.execution import detect_physical_core_count

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
ProgressCallback = Callable[[Mapping[str, object]], None]
P = ParamSpec("P")
R = TypeVar("R")

_UTC_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_INNER_TARGET_MINIMUM = 10
_ALPHA_TIE_TOLERANCE = 1.0e-12
_RIDGE_TIE_TOLERANCE = 1.0e-12
_RIDGE_CANDIDATES = (0.1, 1.0, 10.0)
_FEATURE_GRID_COLUMNS = (
    "issue_time_utc",
    "issue_report_id",
    "grid_id",
    "cell_id",
    "cell_row",
    "cell_column",
    "query_x_m",
    "query_y_m",
    "clipped_area_km2",
)
_STATE_FILE = "state.json"
_PARTIAL_RESULT_FILE = "observed_result.partial.json"
_FINAL_RESULT_FILE = "observed_result.json"
_FINAL_CELL_FILE = "d1_cell_scores.parquet"
_MAX_WORKERS = 4
_RESERVED_PHYSICAL_CORES = 2


def _readonly_float(values: object) -> FloatArray:
    result = np.array(values, dtype=np.float64, copy=True, order="C")
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError("D1 replay vector must be one-dimensional and finite")
    result.setflags(write=False)
    return result


def _readonly_int(values: object) -> IntArray:
    result = np.array(values, dtype=np.int64, copy=True, order="C")
    if result.ndim != 1:
        raise ValueError("D1 replay indices must be one-dimensional")
    result.setflags(write=False)
    return result


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be a string-keyed mapping")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, *, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise TypeError(f"{label} must be a sequence")
    return cast(Sequence[object], value)


def _datetime_to_us(value: datetime) -> int:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("D1 issue time must be timezone-aware")
    delta = value.astimezone(UTC) - _UTC_EPOCH
    return (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds


def _us_to_datetime(value: int) -> datetime:
    return _UTC_EPOCH + timedelta(microseconds=int(value))


def _iso_utc_from_us(value: int) -> str:
    return _us_to_datetime(value).isoformat().replace("+00:00", "Z")


def _emit(progress: ProgressCallback | None, **payload: object) -> None:
    if progress is not None:
        progress(MappingProxyType(dict(payload)))


@dataclass(frozen=True, slots=True)
class D1ParameterSelection:
    """One fold-local alpha or ridge selection with auditable prefix support."""

    parameter: Literal["alpha", "ridge"]
    selected_value: float
    status: Literal["selected", "evidence_insufficient_for_tuning"]
    validation_issue_times_us: tuple[int, ...]
    training_prefix_issue_times_us: tuple[tuple[int, ...], ...]
    validation_event_count: int
    mean_log_density_by_candidate: Mapping[str, float | None]

    def __post_init__(self) -> None:
        if self.parameter not in {"alpha", "ridge"}:
            raise ValueError("D1 parameter selection must describe alpha or ridge")
        if self.status not in {"selected", "evidence_insufficient_for_tuning"}:
            raise ValueError("unknown D1 tuning status")
        if self.validation_event_count < 0:
            raise ValueError("validation event count cannot be negative")
        if len(self.validation_issue_times_us) != len(self.training_prefix_issue_times_us):
            raise ValueError("each validation issue must expose its earlier training prefix")
        for validation, prefix in zip(
            self.validation_issue_times_us,
            self.training_prefix_issue_times_us,
            strict=True,
        ):
            if tuple(sorted(set(prefix))) != prefix or any(item >= validation for item in prefix):
                raise ValueError("inner training prefix contains a non-earlier issue")
        scores = dict(self.mean_log_density_by_candidate)
        if any(value is not None and not math.isfinite(float(value)) for value in scores.values()):
            raise ValueError("D1 tuning scores must be finite or null")
        object.__setattr__(self, "mean_log_density_by_candidate", MappingProxyType(scores))

    def as_mapping(self) -> dict[str, object]:
        return {
            "parameter": self.parameter,
            "selected_value": self.selected_value,
            "status": self.status,
            "validation_issue_times_utc": [
                _iso_utc_from_us(value) for value in self.validation_issue_times_us
            ],
            "training_prefix_issue_times_utc": [
                [_iso_utc_from_us(value) for value in prefix]
                for prefix in self.training_prefix_issue_times_us
            ],
            "validation_event_count": self.validation_event_count,
            "mean_log_density_by_candidate": dict(self.mean_log_density_by_candidate),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> D1ParameterSelection:
        validation = tuple(
            _datetime_to_us(_parse_iso_datetime(item, label="validation issue"))
            for item in _sequence(value.get("validation_issue_times_utc"), label="validation")
        )
        prefixes = tuple(
            tuple(
                _datetime_to_us(_parse_iso_datetime(item, label="training prefix issue"))
                for item in _sequence(prefix, label="training prefix")
            )
            for prefix in _sequence(
                value.get("training_prefix_issue_times_utc"), label="training prefixes"
            )
        )
        parameter = value.get("parameter")
        status = value.get("status")
        if parameter not in {"alpha", "ridge"} or status not in {
            "selected",
            "evidence_insufficient_for_tuning",
        }:
            raise ValueError("stored D1 tuning selection is invalid")
        return cls(
            parameter=cast(Literal["alpha", "ridge"], parameter),
            selected_value=float(value["selected_value"]),
            status=cast(Literal["selected", "evidence_insufficient_for_tuning"], status),
            validation_issue_times_us=validation,
            training_prefix_issue_times_us=prefixes,
            validation_event_count=int(value["validation_event_count"]),
            mean_log_density_by_candidate={
                str(key): None if item is None else float(item)
                for key, item in _mapping(
                    value.get("mean_log_density_by_candidate"), label="candidate scores"
                ).items()
            },
        )


@dataclass(frozen=True, slots=True)
class D1TrainingDiagnostic:
    """Fit-window diagnostics; catalog counts are not called learnt labels.

    Background-only models have no fitted coefficients, so their M4+ count is
    recorded as fit-window scientific context rather than a "training label"
    count.  Feature models use the same field for an apples-to-apples audit.
    """

    fit_issue_count: int
    fit_catalog_m4plus_event_count: int
    coefficient_names: tuple[str, ...]
    active_coefficients: tuple[bool, ...]
    coefficients: tuple[float, ...]
    objective: float | None
    iteration_count: int

    def __post_init__(self) -> None:
        if self.fit_issue_count <= 0 or self.fit_catalog_m4plus_event_count <= 0:
            raise ValueError("D1 final fit must contain issues and M4+ events")
        if not (
            len(self.coefficient_names) == len(self.active_coefficients) == len(self.coefficients)
        ):
            raise ValueError("D1 coefficient diagnostics do not align")
        if any(not math.isfinite(value) for value in self.coefficients):
            raise ValueError("D1 fitted coefficients must be finite")
        if self.objective is not None and not math.isfinite(self.objective):
            raise ValueError("D1 fitted objective must be finite or null")
        if self.iteration_count < 0:
            raise ValueError("D1 fit iteration count cannot be negative")

    def as_mapping(self) -> dict[str, object]:
        return {
            "fit_issue_count": self.fit_issue_count,
            "fit_catalog_m4plus_event_count": self.fit_catalog_m4plus_event_count,
            "coefficient_names": list(self.coefficient_names),
            "active_coefficients": list(self.active_coefficients),
            "coefficients": list(self.coefficients),
            "objective": self.objective,
            "iteration_count": self.iteration_count,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> D1TrainingDiagnostic:
        return cls(
            fit_issue_count=int(value["fit_issue_count"]),
            fit_catalog_m4plus_event_count=int(value["fit_catalog_m4plus_event_count"]),
            coefficient_names=tuple(str(item) for item in value["coefficient_names"]),
            active_coefficients=tuple(bool(item) for item in value["active_coefficients"]),
            coefficients=tuple(float(item) for item in value["coefficients"]),
            objective=(None if value.get("objective") is None else float(value["objective"])),
            iteration_count=int(value["iteration_count"]),
        )


@dataclass(frozen=True, slots=True)
class D1FoldModelResult:
    fold_id: str
    model_id: str
    base: Literal["B0", "R30"]
    feature_groups: tuple[str, ...]
    alpha_selection: D1ParameterSelection
    ridge_selection: D1ParameterSelection | None
    training_diagnostic: D1TrainingDiagnostic

    def __post_init__(self) -> None:
        if self.fold_id not in D1_FOLD_IDS or self.model_id not in D1_MODEL_ORDER:
            raise ValueError("D1 fold/model result identity is not preregistered")
        if self.base not in {"B0", "R30"}:
            raise ValueError("D1 model base must be B0 or R30")
        if bool(self.feature_groups) != (self.ridge_selection is not None):
            raise ValueError("only feature models may carry a ridge selection")

    def as_mapping(self) -> dict[str, object]:
        return {
            "fold_id": self.fold_id,
            "model_id": self.model_id,
            "base": self.base,
            "feature_groups": list(self.feature_groups),
            "selected_alpha": self.alpha_selection.selected_value,
            "alpha_tuning": self.alpha_selection.as_mapping(),
            "selected_ridge": (
                None if self.ridge_selection is None else self.ridge_selection.selected_value
            ),
            "ridge_tuning": (
                None if self.ridge_selection is None else self.ridge_selection.as_mapping()
            ),
            "training_diagnostic": self.training_diagnostic.as_mapping(),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> D1FoldModelResult:
        base = value.get("base")
        if base not in {"B0", "R30"}:
            raise ValueError("stored D1 base is invalid")
        ridge = value.get("ridge_tuning")
        return cls(
            fold_id=str(value["fold_id"]),
            model_id=str(value["model_id"]),
            base=cast(Literal["B0", "R30"], base),
            feature_groups=tuple(str(item) for item in value["feature_groups"]),
            alpha_selection=D1ParameterSelection.from_mapping(
                _mapping(value.get("alpha_tuning"), label="alpha tuning")
            ),
            ridge_selection=(
                None
                if ridge is None
                else D1ParameterSelection.from_mapping(_mapping(ridge, label="ridge tuning"))
            ),
            training_diagnostic=D1TrainingDiagnostic.from_mapping(
                _mapping(value.get("training_diagnostic"), label="training diagnostic")
            ),
        )


@dataclass(frozen=True, slots=True)
class D1IssueModelScore:
    """One complete assessment frame of relative cell mass and alarm prefixes."""

    fold_id: str
    issue_id: str
    issue_time_us: int
    horizon_days: int
    model_id: str
    cell_mass: FloatArray
    ranking_indices: IntArray
    alarm_prefixes: tuple[D1AlarmPrefix, ...]

    def __post_init__(self) -> None:
        if self.fold_id not in D1_FOLD_IDS or self.model_id not in D1_MODEL_ORDER:
            raise ValueError("D1 assessment frame identity is invalid")
        if not self.issue_id or self.horizon_days not in D1_HORIZONS_DAYS:
            raise ValueError("D1 assessment issue/horizon is invalid")
        mass = _readonly_float(self.cell_mass)
        ranking = _readonly_int(self.ranking_indices)
        if mass.size == 0 or np.any(mass <= 0.0):
            raise ValueError("D1 cell masses must be strictly positive")
        if not math.isclose(
            math.fsum(float(value) for value in mass),
            1.0,
            rel_tol=0.0,
            abs_tol=MASS_SUM_ABSOLUTE_TOLERANCE,
        ):
            raise ValueError("D1 assessment mass must sum to one")
        if ranking.size != mass.size or set(int(item) for item in ranking) != set(range(mass.size)):
            raise ValueError("D1 assessment ranking must be a full cell permutation")
        prefixes = tuple(self.alarm_prefixes)
        if tuple(item.budget_km2 for item in prefixes) != D1_AREA_BUDGETS_KM2:
            raise ValueError("D1 assessment frame must contain all five alarm budgets")
        for prefix in prefixes:
            count = prefix.selected_indices.size
            if not np.array_equal(prefix.selected_indices, ranking[:count]):
                raise ValueError("D1 alarm must be an exact prefix of the full ranking")
        object.__setattr__(self, "cell_mass", mass)
        object.__setattr__(self, "ranking_indices", ranking)
        object.__setattr__(self, "alarm_prefixes", prefixes)

    @property
    def actual_area_km2(self) -> tuple[float, ...]:
        return tuple(item.actual_area_km2 for item in self.alarm_prefixes)

    @property
    def prefix_counts(self) -> tuple[int, ...]:
        return tuple(item.selected_indices.size for item in self.alarm_prefixes)

    def as_mapping(self, *, include_cell_mass: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "fold_id": self.fold_id,
            "issue_id": self.issue_id,
            "issue_time_utc": _iso_utc_from_us(self.issue_time_us),
            "horizon_days": self.horizon_days,
            "model_id": self.model_id,
            "cell_count": self.cell_mass.size,
            "alarm_prefix_counts": list(self.prefix_counts),
            "actual_area_km2": list(self.actual_area_km2),
        }
        if include_cell_mass:
            result["relative_cell_mass"] = self.cell_mass.tolist()
            result["ranking_indices"] = self.ranking_indices.tolist()
        return result


@dataclass(frozen=True, slots=True)
class D1ObservedClusterOutcome:
    cluster_id: str
    fold_id: str
    issue_id: str
    issue_time_us: int
    horizon_days: int
    model_id: str
    representative_cell_index: int | None
    outside_support: bool
    log_density: float | None
    hit_by_area: tuple[bool, ...]

    def __post_init__(self) -> None:
        if self.outside_support:
            if (
                self.representative_cell_index is not None
                or self.log_density is not None
                or any(self.hit_by_area)
            ):
                raise ValueError("outside-support D1 cluster must be an explicit forced miss")
        elif (
            self.representative_cell_index is None
            or self.representative_cell_index < 0
            or self.log_density is None
            or not math.isfinite(self.log_density)
        ):
            raise ValueError("inside-support cluster outcome is incomplete")
        self.evaluation_outcome()

    def evaluation_outcome(self) -> D1ClusterModelOutcome:
        return D1ClusterModelOutcome(
            cluster_id=self.cluster_id,
            fold_id=self.fold_id,
            issue_id=self.issue_id,
            horizon_days=self.horizon_days,
            model_id=self.model_id,
            log_density=self.log_density,
            outside_support=self.outside_support,
            hit_by_area=self.hit_by_area,
        )

    def as_mapping(self) -> dict[str, object]:
        return {
            "cluster_id": self.cluster_id,
            "fold_id": self.fold_id,
            "issue_id": self.issue_id,
            "issue_time_utc": _iso_utc_from_us(self.issue_time_us),
            "horizon_days": self.horizon_days,
            "model_id": self.model_id,
            "representative_cell_index": self.representative_cell_index,
            "outside_support": self.outside_support,
            "log_density": self.log_density,
            "hit_by_area": list(self.hit_by_area),
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> D1ObservedClusterOutcome:
        return cls(
            cluster_id=str(value["cluster_id"]),
            fold_id=str(value["fold_id"]),
            issue_id=str(value["issue_id"]),
            issue_time_us=_datetime_to_us(
                _parse_iso_datetime(value["issue_time_utc"], label="outcome issue")
            ),
            horizon_days=int(value["horizon_days"]),
            model_id=str(value["model_id"]),
            representative_cell_index=(
                None
                if value.get("representative_cell_index") is None
                else int(value["representative_cell_index"])
            ),
            outside_support=bool(value.get("outside_support", False)),
            log_density=(None if value.get("log_density") is None else float(value["log_density"])),
            hit_by_area=tuple(bool(item) for item in value["hit_by_area"]),
        )


@dataclass(frozen=True, slots=True)
class D1ObservedReplayResult:
    protocol_version: str
    identities: Mapping[str, object]
    workers: int
    expected_support_by_horizon: Mapping[int, tuple[tuple[str, str, str], ...]]
    fold_models: tuple[D1FoldModelResult, ...]
    outcomes: tuple[D1ObservedClusterOutcome, ...]
    issues: tuple[D1IssueModelScore, ...]
    status: Literal["completed"] = "completed"

    def __post_init__(self) -> None:
        if self.protocol_version != "d1.0.0" or self.status != "completed":
            raise ValueError("D1 observed result has an invalid protocol/status")
        expected_units = {(fold, model) for fold in D1_FOLD_IDS for model in D1_MODEL_ORDER}
        observed_units = {(item.fold_id, item.model_id) for item in self.fold_models}
        if observed_units != expected_units or len(self.fold_models) != len(expected_units):
            raise ValueError("D1 observed result does not contain all 18 fold/model units")
        issue_keys = {
            (item.fold_id, item.issue_time_us, item.horizon_days, item.model_id)
            for item in self.issues
        }
        if len(issue_keys) != len(self.issues) or len(self.issues) != 198:
            raise ValueError("D1 result must contain exactly 33 assessment issues x 6 models")
        horizon_frame_counts = {
            horizon: sum(item.horizon_days == horizon for item in self.issues)
            for horizon in D1_HORIZONS_DAYS
        }
        if horizon_frame_counts != {30: 24 * 6, 90: 9 * 6}:
            raise ValueError("D1 issue alarm table omitted a zero-target assessment issue")
        models_by_frame: dict[tuple[str, int, int], set[str]] = {}
        for item in self.issues:
            models_by_frame.setdefault(
                (item.fold_id, item.issue_time_us, item.horizon_days), set()
            ).add(item.model_id)
        if len(models_by_frame) != 33 or any(
            models != set(D1_MODEL_ORDER) for models in models_by_frame.values()
        ):
            raise ValueError("D1 assessment frame axis must contain all six models exactly once")
        issue_axis_counts = {
            (fold, horizon): sum(key[0] == fold and key[2] == horizon for key in models_by_frame)
            for fold in D1_FOLD_IDS
            for horizon in D1_HORIZONS_DAYS
        }
        if issue_axis_counts != {
            ("fold_1", 30): 8,
            ("fold_2", 30): 8,
            ("fold_3", 30): 8,
            ("fold_1", 90): 3,
            ("fold_2", 90): 3,
            ("fold_3", 90): 3,
        }:
            raise ValueError("D1 assessment issue support differs from the frozen 24/9 axis")
        expected_horizons = set(D1_HORIZONS_DAYS)
        if set(self.expected_support_by_horizon) != expected_horizons:
            raise ValueError("D1 result omitted an expected horizon support set")
        expected_support_counts = {
            30: {"fold_1": 8, "fold_2": 6, "fold_3": 7},
            90: {"fold_1": 8, "fold_2": 6, "fold_3": 8},
        }
        for horizon, support in self.expected_support_by_horizon.items():
            expected = set(support)
            if len(expected) != len(support) or len(expected) != sum(
                expected_support_counts[horizon].values()
            ):
                raise ValueError("D1 expected cluster support differs from the frozen 21/22 sets")
            fold_support_counts = {
                fold: sum(item[1] == fold for item in support) for fold in D1_FOLD_IDS
            }
            if fold_support_counts != expected_support_counts[horizon]:
                raise ValueError("D1 expected cluster support changed its fold allocation")
            for model in D1_MODEL_ORDER:
                model_outcomes = tuple(
                    item
                    for item in self.outcomes
                    if item.horizon_days == horizon and item.model_id == model
                )
                observed = {
                    (item.cluster_id, item.fold_id, item.issue_id) for item in model_outcomes
                }
                if observed != expected or len(model_outcomes) != len(expected):
                    raise ValueError("D1 model omitted or added an assessment cluster")
        object.__setattr__(self, "identities", MappingProxyType(dict(self.identities)))
        object.__setattr__(
            self,
            "expected_support_by_horizon",
            MappingProxyType(
                {key: tuple(value) for key, value in self.expected_support_by_horizon.items()}
            ),
        )

    @property
    def evaluation_outcomes(self) -> tuple[D1ClusterModelOutcome, ...]:
        return tuple(item.evaluation_outcome() for item in self.outcomes)

    @property
    def alarm_outcomes(self) -> tuple[D1IssueAlarmOutcome, ...]:
        return tuple(
            D1IssueAlarmOutcome(
                fold_id=item.fold_id,
                issue_id=item.issue_id,
                horizon_days=item.horizon_days,
                model_id=item.model_id,
                actual_area_km2=item.actual_area_km2,
            )
            for item in self.issues
        )

    @property
    def expected_issues_by_horizon(self) -> Mapping[int, tuple[tuple[str, str], ...]]:
        return MappingProxyType(
            {
                horizon: tuple(
                    sorted(
                        {
                            (item.fold_id, item.issue_id)
                            for item in self.issues
                            if item.horizon_days == horizon and item.model_id == "B0"
                        }
                    )
                )
                for horizon in D1_HORIZONS_DAYS
            }
        )

    def as_mapping(self, *, include_cell_mass: bool = True) -> dict[str, object]:
        folds = []
        for fold_id in D1_FOLD_IDS:
            models = [item.as_mapping() for item in self.fold_models if item.fold_id == fold_id]
            folds.append({"fold_id": fold_id, "models": models})
        return {
            "schema_version": 1,
            "protocol_version": self.protocol_version,
            "result_kind": "observed_replay",
            "retrospective_only": True,
            "relative_strength_not_absolute_probability": True,
            "status": self.status,
            "identities": dict(self.identities),
            "workers": self.workers,
            "blas_threads_per_worker": 1,
            "expected_support_by_horizon": {
                str(horizon): [
                    {"cluster_id": cluster, "fold_id": fold, "issue_id": issue}
                    for cluster, fold, issue in support
                ]
                for horizon, support in self.expected_support_by_horizon.items()
            },
            "expected_issues_by_horizon": {
                str(horizon): [{"fold_id": fold, "issue_id": issue} for fold, issue in support]
                for horizon, support in self.expected_issues_by_horizon.items()
            },
            "folds": folds,
            "outcomes": [item.as_mapping() for item in self.outcomes],
            "issue_alarm_outcomes": [
                item.as_mapping(include_cell_mass=include_cell_mass) for item in self.issues
            ],
        }


def _parse_iso_datetime(value: object, *, label: str) -> datetime:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be an ISO-8601 string")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include an offset")
    return parsed.astimezone(UTC)


def _inner_validation_indices(issue_count: int) -> tuple[int, ...]:
    """Return the frozen last max(3, ceil(n/3)) validation positions."""

    if isinstance(issue_count, bool) or issue_count <= 0:
        raise ValueError("outer fit issue count must be positive")
    validation_count = max(3, math.ceil(issue_count / 3))
    validation_count = min(validation_count, issue_count)
    return tuple(range(issue_count - validation_count, issue_count))


def _training_prefix_indices(issue_count: int) -> tuple[tuple[int, tuple[int, ...]], ...]:
    return tuple(
        (validation, tuple(range(validation)))
        for validation in _inner_validation_indices(issue_count)
    )


def _choose_alpha(score_by_candidate: Mapping[float, float]) -> float:
    if tuple(sorted(score_by_candidate)) != FROZEN_R30_ALPHA_CANDIDATES:
        raise ValueError("alpha scores do not cover the frozen candidate set")
    maximum = max(float(value) for value in score_by_candidate.values())
    eligible = [
        candidate
        for candidate, score in score_by_candidate.items()
        if maximum - float(score) <= _ALPHA_TIE_TOLERANCE
    ]
    return min(eligible)


def _choose_ridge(score_by_candidate: Mapping[float, float]) -> float:
    if tuple(sorted(score_by_candidate)) != _RIDGE_CANDIDATES:
        raise ValueError("ridge scores do not cover the frozen candidate set")
    maximum = max(float(value) for value in score_by_candidate.values())
    eligible = [
        candidate
        for candidate, score in score_by_candidate.items()
        if maximum - float(score) <= _RIDGE_TIE_TOLERANCE
    ]
    return max(eligible)


def _log_mass(mass: object) -> FloatArray:
    values = _readonly_float(mass)
    if np.any(values <= 0.0):
        raise FloatingPointError("D1 base mass contains a zero; no epsilon is permitted")
    logged = np.log(values)
    logged.setflags(write=False)
    return logged


def _pooled_log_density_sum(
    mass: object,
    counts: object,
    clipped_area_km2: object,
) -> tuple[float, int]:
    probability = np.asarray(mass, dtype=np.float64)
    response = np.asarray(counts, dtype=np.float64)
    area = np.asarray(clipped_area_km2, dtype=np.float64)
    if not (probability.shape == response.shape == area.shape) or probability.ndim != 1:
        raise ValueError("D1 validation mass/count/area vectors do not align")
    if np.any(response < 0.0) or not np.array_equal(response, np.floor(response)):
        raise ValueError("D1 validation response must contain event counts")
    selected = response > 0.0
    event_count = int(np.sum(response, dtype=np.float64))
    if event_count == 0:
        return 0.0, 0
    if (
        np.any(probability[selected] <= 0.0)
        or np.any(area[selected] <= 0.0)
        or not np.isfinite(probability[selected]).all()
        or not np.isfinite(area[selected]).all()
    ):
        raise FloatingPointError("a scored D1 target has zero or non-finite density")
    log_density = np.log(probability[selected]) - np.log(area[selected])
    return float(np.dot(response[selected], log_density)), event_count


def _full_ranking(mass: object, grid: SpatialGrid) -> IntArray:
    probability = _readonly_float(mass)
    if probability.size != grid.cell_count:
        raise ValueError("D1 mass does not match the operational grid")
    strength = probability / grid.clipped_area_km2
    ranking = sorted(
        range(grid.cell_count),
        key=lambda index: (
            -float(strength[index]),
            int(grid.rows[index]),
            int(grid.columns[index]),
            grid.cell_ids[index].encode("utf-8"),
        ),
    )
    return _readonly_int(ranking)


def _model_specs(protocol: D1Protocol) -> tuple[Mapping[str, Any], ...]:
    raw = _sequence(protocol.config.get("models"), label="D1 models")
    specs = tuple(_mapping(item, label="D1 model") for item in raw)
    if tuple(str(item.get("id")) for item in specs) != EXPECTED_MODEL_IDS:
        raise ValueError("D1 model order changed after protocol validation")
    return specs


def _base_mass(background: D1CausalBackground, base: str, alpha: float) -> FloatArray:
    if base == "B0":
        return background.b0_mass_25km
    if base == "R30":
        return background.mass_for_alpha(alpha)
    raise ValueError("D1 model names an unknown background")


def _select_alpha(
    fit_issue_times_us: tuple[int, ...],
    counts_by_issue: Mapping[int, FloatArray],
    backgrounds: Mapping[int, D1CausalBackground],
    grid: SpatialGrid,
) -> D1ParameterSelection:
    prefixes = _training_prefix_indices(len(fit_issue_times_us))
    validation_times = tuple(fit_issue_times_us[index] for index, _ in prefixes)
    training_times = tuple(
        tuple(fit_issue_times_us[index] for index in prefix) for _, prefix in prefixes
    )
    totals = {candidate: 0.0 for candidate in FROZEN_R30_ALPHA_CANDIDATES}
    event_count = 0
    for validation_index, _ in prefixes:
        issue_time = fit_issue_times_us[validation_index]
        counts = counts_by_issue[issue_time]
        local_events = int(np.sum(counts, dtype=np.float64))
        event_count += local_events
        for candidate in FROZEN_R30_ALPHA_CANDIDATES:
            value, observed_events = _pooled_log_density_sum(
                backgrounds[issue_time].mass_for_alpha(candidate),
                counts,
                grid.clipped_area_km2,
            )
            if observed_events != local_events:
                raise AssertionError("alpha validation event count changed by candidate")
            totals[candidate] += value
    if event_count < _INNER_TARGET_MINIMUM:
        return D1ParameterSelection(
            parameter="alpha",
            selected_value=0.0,
            status="evidence_insufficient_for_tuning",
            validation_issue_times_us=validation_times,
            training_prefix_issue_times_us=training_times,
            validation_event_count=event_count,
            mean_log_density_by_candidate={
                format(candidate, "g"): None for candidate in FROZEN_R30_ALPHA_CANDIDATES
            },
        )
    means = {candidate: total / event_count for candidate, total in totals.items()}
    selected = _choose_alpha(means)
    return D1ParameterSelection(
        parameter="alpha",
        selected_value=selected,
        status="selected",
        validation_issue_times_us=validation_times,
        training_prefix_issue_times_us=training_times,
        validation_event_count=event_count,
        mean_log_density_by_candidate={
            format(candidate, "g"): means[candidate] for candidate in FROZEN_R30_ALPHA_CANDIDATES
        },
    )


def _fit_inner_preprocessors(
    contract: D1FeatureContract,
    selected_groups: tuple[str, ...],
    fit_issue_times_us: tuple[int, ...],
    features_by_issue: Mapping[int, D1IssueFeatures],
) -> tuple[tuple[int, D1GroupPreprocessor], ...]:
    """Fit one preprocessor per validation issue on its strict past prefix."""

    result: list[tuple[int, D1GroupPreprocessor]] = []
    for validation_index, prefix in _training_prefix_indices(len(fit_issue_times_us)):
        if not prefix:
            raise ValueError("D1 inner validation has no earlier training issue")
        training = tuple(features_by_issue[fit_issue_times_us[index]] for index in prefix)
        result.append(
            (validation_index, D1GroupPreprocessor.fit(contract, selected_groups, training))
        )
        _validate_preprocessor_fit_axis(
            result[-1][1],
            tuple(fit_issue_times_us[index] for index in prefix),
        )
    return tuple(result)


def _training_issue(
    issue_time_us: int,
    *,
    base_mass: FloatArray,
    design: FloatArray,
    counts: FloatArray,
) -> ConditionalSpatialTrainingIssue:
    return ConditionalSpatialTrainingIssue(
        issue_id=_iso_utc_from_us(issue_time_us),
        log_base_mass=_log_mass(base_mass),
        design=design,
        future_counts=counts,
    )


def _select_ridge(
    *,
    contract: D1FeatureContract,
    selected_groups: tuple[str, ...],
    base: str,
    selected_alpha: float,
    fit_issue_times_us: tuple[int, ...],
    features_by_issue: Mapping[int, D1IssueFeatures],
    counts_by_issue: Mapping[int, FloatArray],
    backgrounds: Mapping[int, D1CausalBackground],
    grid: SpatialGrid,
) -> D1ParameterSelection:
    prefixes = _training_prefix_indices(len(fit_issue_times_us))
    validation_times = tuple(fit_issue_times_us[index] for index, _ in prefixes)
    training_times = tuple(
        tuple(fit_issue_times_us[index] for index in prefix) for _, prefix in prefixes
    )
    event_count = sum(
        int(np.sum(counts_by_issue[fit_issue_times_us[index]], dtype=np.float64))
        for index, _ in prefixes
    )
    if event_count < _INNER_TARGET_MINIMUM:
        return D1ParameterSelection(
            parameter="ridge",
            selected_value=10.0,
            status="evidence_insufficient_for_tuning",
            validation_issue_times_us=validation_times,
            training_prefix_issue_times_us=training_times,
            validation_event_count=event_count,
            mean_log_density_by_candidate={
                format(candidate, "g"): None for candidate in _RIDGE_CANDIDATES
            },
        )

    totals = {candidate: 0.0 for candidate in _RIDGE_CANDIDATES}
    preprocessors = dict(
        _fit_inner_preprocessors(
            contract,
            selected_groups,
            fit_issue_times_us,
            features_by_issue,
        )
    )
    for validation_index, prefix in prefixes:
        issue_time = fit_issue_times_us[validation_index]
        preprocessor = preprocessors[validation_index]
        designs = {
            fit_issue_times_us[index]: preprocessor.transform(
                features_by_issue[fit_issue_times_us[index]]
            )
            for index in (*prefix, validation_index)
        }
        training = tuple(
            _training_issue(
                fit_issue_times_us[index],
                base_mass=_base_mass(backgrounds[fit_issue_times_us[index]], base, selected_alpha),
                design=designs[fit_issue_times_us[index]].values,
                counts=counts_by_issue[fit_issue_times_us[index]],
            )
            for index in prefix
        )
        for candidate in _RIDGE_CANDIDATES:
            fitted = fit_conditional_spatial_ridge(
                training,
                ridge_lambda=candidate,
                active_coefficients=preprocessor.active_coefficients,
            )
            predicted = fitted.predict(
                _log_mass(_base_mass(backgrounds[issue_time], base, selected_alpha)),
                designs[issue_time].values,
            )
            value, _ = _pooled_log_density_sum(
                predicted,
                counts_by_issue[issue_time],
                grid.clipped_area_km2,
            )
            totals[candidate] += value
    means = {candidate: total / event_count for candidate, total in totals.items()}
    selected = _choose_ridge(means)
    return D1ParameterSelection(
        parameter="ridge",
        selected_value=selected,
        status="selected",
        validation_issue_times_us=validation_times,
        training_prefix_issue_times_us=training_times,
        validation_event_count=event_count,
        mean_log_density_by_candidate={
            format(candidate, "g"): means[candidate] for candidate in _RIDGE_CANDIDATES
        },
    )


def _resolve_supplied_path(project_root: Path, value: Path) -> Path:
    candidate = value if value.is_absolute() else project_root / value
    return candidate.resolve()


def _resolve_data_path(
    project_root: Path,
    relative: str,
    *,
    configured_data_root: Path,
) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute():
        raise ValueError("D1 input path must remain repository-relative")
    local_candidate = (project_root / relative_path).resolve()
    if local_candidate.exists():
        return local_candidate
    parts = relative_path.parts
    if not parts or parts[0].lower() != "data":
        raise FileNotFoundError(local_candidate)
    fallback = (configured_data_root / Path(*parts[1:])).resolve()
    try:
        fallback.relative_to(configured_data_root.resolve())
    except ValueError as exc:
        raise ValueError("D1 configured data path escapes its data root") from exc
    return fallback


def _verified_input_paths(protocol: D1Protocol) -> tuple[dict[str, Path], dict[str, str]]:
    data = _mapping(protocol.config.get("data"), label="D1 data")
    configured_root_raw = data.get("local_data_root_current_machine")
    if not isinstance(configured_root_raw, str) or not configured_root_raw:
        raise ValueError("D1 configuration omitted its local data root")
    configured_root = Path(configured_root_raw).resolve()
    bindings = {
        "earthquake_event": (
            _mapping(data.get("earthquake_event"), label="earthquake_event"),
            "file_sha256",
        ),
        "study_area": (
            _mapping(data.get("study_area"), label="study_area"),
            "file_sha256",
        ),
        "anomaly_features": (
            _mapping(data.get("anomaly_features"), label="anomaly_features"),
            "feature_store_file_sha256",
        ),
    }
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for name, (binding, hash_field) in bindings.items():
        path_field = "path"
        raw_path = binding.get(path_field)
        expected_hash = binding.get(hash_field)
        if not isinstance(raw_path, str) or not isinstance(expected_hash, str):
            raise ValueError(f"D1 {name} binding omitted path or SHA-256")
        path = _resolve_data_path(
            protocol.repository_root,
            raw_path,
            configured_data_root=configured_root,
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        observed_hash = sha256_file(path)
        if observed_hash != expected_hash:
            raise ValueError(f"D1 {name} bytes differ from the preregistered SHA-256")
        paths[name] = path
        hashes[name] = observed_hash
    return paths, hashes


def _git_commit(project_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    commit = completed.stdout.strip()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("D1 replay could not establish a full Git commit identity")
    status = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--",
            "src/seismoflux/d1_replay",
            "configs/d1_retrospective_development.yaml",
            "data/manifests/d1_fold_water_level_manifest.json",
            "docs/d1_executable_scientific_contract.md",
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if status.stdout.strip():
        raise ValueError("D1 replay implementation/contract must be committed before scoring")
    return commit


@dataclass(frozen=True, slots=True)
class _D1ResourceBoundary:
    logical_cores: int
    physical_cores: int
    max_workers_after_reservation: int


def _native_blas_libraries() -> tuple[Path, ...]:
    """Locate loaded-wheel BLAS libraries for the no-threadpoolctl fallback."""

    import scipy  # type: ignore[import-untyped]

    roots = (
        Path(np.__file__).resolve().parent.parent / "numpy.libs",
        Path(scipy.__file__).resolve().parent.parent / "scipy.libs",
    )
    patterns = (
        "*openblas*.dll",
        "*openblas*.so*",
        "*openblas*.dylib",
        "*mkl_rt*.dll",
        "*mkl_rt*.so*",
        "*mkl_rt*.dylib",
    )
    return tuple(
        sorted(
            {
                candidate.resolve()
                for root in roots
                if root.is_dir()
                for pattern in patterns
                for candidate in root.glob(pattern)
                if candidate.is_file()
            },
            key=lambda value: str(value).casefold(),
        )
    )


def _set_native_blas_threads_to_one() -> None:
    """Apply and verify a one-thread limit through OpenBLAS/MKL native APIs.

    Environment variables are set before NumPy imports above.  This fallback is
    additionally required because another caller may have imported NumPy first;
    changing an environment variable alone would then be too late.
    """

    import ctypes

    symbol_pairs = (
        ("scipy_openblas_set_num_threads64_", "scipy_openblas_get_num_threads64_"),
        ("scipy_openblas_set_num_threads", "scipy_openblas_get_num_threads"),
        ("openblas_set_num_threads64_", "openblas_get_num_threads64_"),
        ("openblas_set_num_threads", "openblas_get_num_threads"),
        ("MKL_Set_Num_Threads", "MKL_Get_Max_Threads"),
    )
    constrained = 0
    for path in _native_blas_libraries():
        try:
            library = ctypes.CDLL(str(path))
        except OSError:
            continue
        for setter_name, getter_name in symbol_pairs:
            try:
                setter = getattr(library, setter_name)
                getter = getattr(library, getter_name)
            except AttributeError:
                continue
            setter.argtypes = [ctypes.c_int]
            setter.restype = None
            getter.argtypes = []
            getter.restype = ctypes.c_int
            setter(1)
            if int(getter()) != 1:
                raise RuntimeError(f"failed to constrain {path.name} to one BLAS thread")
            constrained += 1
            break
    if constrained == 0:
        raise RuntimeError(
            "threadpoolctl is unavailable and no supported NumPy/SciPy BLAS runtime "
            "could be constrained; install threadpoolctl or use a supported OpenBLAS/MKL build"
        )


@contextmanager
def _single_thread_math_runtime() -> Iterator[None]:
    """Limit already-loaded numerical runtimes, not only future imports."""

    if _threadpool_limits is not None:
        with _threadpool_limits(limits=1):
            yield
        return
    _set_native_blas_threads_to_one()
    try:
        yield
    finally:
        # Keep the conservative global limit in force after a callback returns.
        _set_native_blas_threads_to_one()


def _single_threaded(function: Callable[P, R]) -> Callable[P, R]:
    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        with _single_thread_math_runtime():
            return function(*args, **kwargs)

    return wrapped


def _assert_resource_boundary(workers: int) -> _D1ResourceBoundary:
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("D1 workers must be a positive integer")
    if workers > _MAX_WORKERS:
        raise ValueError(f"D1 workers cannot exceed the frozen maximum of {_MAX_WORKERS}")
    logical_cores = os.cpu_count() or 1
    physical_cores = detect_physical_core_count()
    if (
        isinstance(physical_cores, bool)
        or not isinstance(physical_cores, int)
        or physical_cores <= _RESERVED_PHYSICAL_CORES
    ):
        raise RuntimeError(
            "D1 requires at least three verified physical cores so two can remain free"
        )
    available = physical_cores - _RESERVED_PHYSICAL_CORES
    maximum = min(_MAX_WORKERS, available)
    if workers > maximum:
        raise ValueError("D1 workers would leave fewer than two physical CPU cores free")
    for name in _BLAS_THREAD_ENV:
        os.environ[name] = "1"
        if os.environ.get(name) != "1":
            raise RuntimeError(f"failed to constrain {name} to one thread")
    return _D1ResourceBoundary(
        logical_cores=logical_cores,
        physical_cores=physical_cores,
        max_workers_after_reservation=maximum,
    )


def _single_value(table: pa.Table, name: str, *, label: str) -> object:
    column = table[name].combine_chunks()
    unique = pc.unique(column)
    if len(unique) != 1 or unique.null_count:
        raise ValueError(f"{label} must contain exactly one non-null value")
    return unique[0].as_py()


def _expected_static_grid(domain: D1SpatialDomain) -> D1StaticGrid:
    grid = domain.stage3_grid
    operational = domain.operational_grid
    comparisons = (
        (grid.cell_ids, operational.cell_ids),
        (grid.rows, operational.rows),
        (grid.columns, operational.columns),
        (grid.query_xy_m / 1_000.0, operational.query_xy_km),
        (grid.clipped_area_km2, operational.clipped_area_km2),
    )
    for left, right in comparisons:
        equal = left == right if isinstance(left, tuple) else np.array_equal(left, right)
        if not equal:
            raise ValueError("Stage-3 feature grid differs from the D1 operational grid")
    return D1StaticGrid(
        grid_id=grid.grid_id,
        cell_ids=grid.cell_ids,
        rows=grid.rows,
        columns=grid.columns,
        query_x_m=grid.query_xy_m[:, 0],
        query_y_m=grid.query_xy_m[:, 1],
        clipped_area_km2=grid.clipped_area_km2,
    )


def _verify_fold_bridge(protocol: D1Protocol) -> None:
    """Freeze config F1/F2/F3 to manifest fold_1/fold_2/fold_3 explicitly."""

    time_node = _mapping(protocol.config.get("time"), label="D1 time")
    config_folds = _sequence(time_node.get("folds"), label="D1 config folds")
    manifest_folds = _sequence(protocol.water_level.get("folds"), label="D1 manifest folds")
    if len(config_folds) != 3 or len(manifest_folds) != 3:
        raise ValueError("D1 requires exactly three outer folds")
    for ordinal, (raw_config, raw_manifest) in enumerate(
        zip(config_folds, manifest_folds, strict=True), start=1
    ):
        config_fold = _mapping(raw_config, label=f"config fold {ordinal}")
        manifest_fold = _mapping(raw_manifest, label=f"manifest fold {ordinal}")
        expected_config = f"F{ordinal}"
        expected_manifest = f"fold_{ordinal}"
        if (
            config_fold.get("id") != expected_config
            or manifest_fold.get("fold_id") != expected_manifest
        ):
            raise ValueError("D1 config-to-manifest fold bridge changed")
        interval = _mapping(
            manifest_fold.get("assessment_interval_local"),
            label=f"{expected_manifest} assessment interval",
        )
        if config_fold.get("assessment_start_local") != interval.get(
            "start_inclusive"
        ) or config_fold.get("assessment_end_exclusive_local") != interval.get("end_exclusive"):
            raise ValueError("D1 config and manifest assessment intervals differ")


def _table_numpy(table: pa.Table, name: str, dtype: np.dtype[Any]) -> NDArray[Any]:
    return np.asarray(
        table[name].combine_chunks().to_numpy(zero_copy_only=False),
        dtype=dtype,
    )


def _issue_from_row_group(
    parquet: pq.ParquetFile,
    row_group_index: int,
    expected_grid: D1StaticGrid,
) -> D1IssueFeatures:
    selected_columns = (*_FEATURE_GRID_COLUMNS, *D1_SOURCE_COLUMNS)
    table = parquet.read_row_group(row_group_index, columns=list(selected_columns))
    issue_time = _single_value(table, "issue_time_utc", label="issue_time_utc")
    report_id = _single_value(table, "issue_report_id", label="issue_report_id")
    grid_id = _single_value(table, "grid_id", label="grid_id")
    if (
        not isinstance(issue_time, datetime)
        or issue_time.tzinfo is None
        or not isinstance(report_id, str)
        or not report_id
        or grid_id != expected_grid.grid_id
    ):
        raise ValueError("D1 feature row group changed its issue/grid identity")
    cell_ids = tuple(cast(list[str], table["cell_id"].combine_chunks().to_pylist()))
    comparisons = (
        (cell_ids, expected_grid.cell_ids, "cell_id"),
        (
            _table_numpy(table, "cell_row", np.dtype(np.int64)),
            expected_grid.rows,
            "cell_row",
        ),
        (
            _table_numpy(table, "cell_column", np.dtype(np.int64)),
            expected_grid.columns,
            "cell_column",
        ),
        (
            _table_numpy(table, "query_x_m", np.dtype(np.float64)),
            expected_grid.query_x_m,
            "query_x_m",
        ),
        (
            _table_numpy(table, "query_y_m", np.dtype(np.float64)),
            expected_grid.query_y_m,
            "query_y_m",
        ),
        (
            _table_numpy(table, "clipped_area_km2", np.dtype(np.float64)),
            expected_grid.clipped_area_km2,
            "clipped_area_km2",
        ),
    )
    for observed, expected, name in comparisons:
        if isinstance(observed, tuple):
            equal = observed == expected
        else:
            equal = np.array_equal(observed, expected)
        if not equal:
            raise ValueError(f"D1 feature row group changed frozen grid column {name}")
    values = np.empty((expected_grid.cell_count, len(D1_SOURCE_COLUMNS)), dtype=np.float64)
    nulls = np.empty(values.shape, dtype=np.bool_)
    for column_index, name in enumerate(D1_SOURCE_COLUMNS):
        column = table[name].combine_chunks().cast(pa.float64())
        missing = np.asarray(column.is_null().to_numpy(zero_copy_only=False), dtype=np.bool_)
        raw = np.asarray(column.to_numpy(zero_copy_only=False), dtype=np.float64)
        if not np.isfinite(raw[~missing]).all():
            raise ValueError(f"D1 feature source {name} has a non-finite valid value")
        values[:, column_index] = raw
        nulls[:, column_index] = missing
    return D1IssueFeatures(
        issue_time_utc=issue_time.astimezone(UTC),
        issue_report_id=report_id,
        grid=expected_grid,
        source_columns=D1_SOURCE_COLUMNS,
        values=values,
        null_mask=nulls,
    )


def _load_needed_issue_features(
    path: Path,
    needed_issue_times_us: Sequence[int],
    domain: D1SpatialDomain,
    *,
    expected_issue_count: int,
    expected_cell_count: int,
) -> Mapping[int, D1IssueFeatures]:
    """Read light issue timestamps for all row groups, then only needed 15-field groups."""

    needed = tuple(sorted(set(int(value) for value in needed_issue_times_us)))
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_row_groups != expected_issue_count:
        raise ValueError("D1 feature store issue count differs from the contract")
    required_columns = set((*_FEATURE_GRID_COLUMNS, *D1_SOURCE_COLUMNS))
    missing = sorted(required_columns - set(parquet.schema_arrow.names))
    if missing:
        raise ValueError(f"D1 feature store is missing columns: {missing}")
    row_group_by_issue: dict[int, int] = {}
    previous: int | None = None
    for row_group_index in range(parquet.metadata.num_row_groups):
        metadata = parquet.metadata.row_group(row_group_index)
        if metadata.num_rows != expected_cell_count:
            raise ValueError("D1 feature store row group has the wrong cell count")
        small = parquet.read_row_group(row_group_index, columns=["issue_time_utc"])
        issue = _single_value(small, "issue_time_utc", label="issue_time_utc")
        if not isinstance(issue, datetime) or issue.tzinfo is None:
            raise ValueError("D1 feature issue time must be timezone-aware")
        issue_us = _datetime_to_us(issue)
        if previous is not None and issue_us <= previous:
            raise ValueError("D1 feature row groups are not strictly chronological")
        previous = issue_us
        if issue_us in needed:
            row_group_by_issue[issue_us] = row_group_index
    if set(row_group_by_issue) != set(needed):
        absent = sorted(set(needed) - set(row_group_by_issue))
        raise ValueError(f"D1 feature store omitted needed issue times: {absent}")
    expected_grid = _expected_static_grid(domain)
    result = {
        issue_us: _issue_from_row_group(parquet, row_group_by_issue[issue_us], expected_grid)
        for issue_us in needed
    }
    return MappingProxyType(result)


def _target_counts_by_issue(
    targets: TargetSet,
    catalog: Stage2SEarthquakeCatalog,
    domain: D1SpatialDomain,
) -> Mapping[int, FloatArray]:
    cell_count = domain.operational_grid.cell_count
    counts = {issue: np.zeros(cell_count, dtype=np.float64) for issue in targets.issue_times_us}
    for event_index, issue_time in zip(
        targets.event_indices,
        targets.assigned_issue_times_us,
        strict=True,
    ):
        cell_index = domain.locator.locate_lonlat(
            float(catalog.longitude[event_index]),
            float(catalog.latitude[event_index]),
        )
        if cell_index is None:
            raise ValueError("inside-study-area D1 fit event misses the frozen grid")
        counts[issue_time][cell_index] += 1.0
    return MappingProxyType({issue: _readonly_float(values) for issue, values in counts.items()})


@dataclass(frozen=True, slots=True)
class D1PreparedReplay:
    protocol: D1Protocol
    identities: Mapping[str, object]
    catalog: Stage2SEarthquakeCatalog
    target_layer: D1TargetLayer
    domain: D1SpatialDomain
    feature_contract: D1FeatureContract
    features_by_issue: Mapping[int, D1IssueFeatures]
    backgrounds: Mapping[int, D1CausalBackground]
    counts_by_fold: Mapping[str, Mapping[int, FloatArray]]


@dataclass(frozen=True, slots=True)
class D1FeatureVariantFoldResult:
    """Fold-local sufficient statistics returned to D1 placebo callbacks."""

    fold_id: str
    cluster_count: int
    hit_count_by_model: Mapping[str, int]
    hit_gain_by_comparison: Mapping[str, int]
    recall_difference_by_comparison: Mapping[str, float]
    selected_alpha: float
    selected_ridge_by_model: Mapping[str, float]
    preprocessing_refit: bool = True
    alpha_reselected: bool = True
    ridge_reselected: bool = True
    models_refit: bool = True

    def __post_init__(self) -> None:
        if self.fold_id not in D1_FOLD_IDS or self.cluster_count <= 0:
            raise ValueError("D1 feature-variant fold result has invalid support")
        expected_models = {"B0_C", "B0_C_A_snapshot", "B0_C_A_dynamic"}
        if (
            set(self.hit_count_by_model) != expected_models
            or set(self.selected_ridge_by_model) != expected_models
        ):
            raise ValueError("D1 feature-variant result omitted an attribution model")
        if any(
            isinstance(value, bool) or value < 0 or value > self.cluster_count
            for value in self.hit_count_by_model.values()
        ):
            raise ValueError("D1 feature-variant hit count is outside its support")
        expected_comparisons = {
            "B0_C_A_snapshot_minus_B0_C",
            "B0_C_A_dynamic_minus_B0_C",
            "B0_C_A_dynamic_minus_B0_C_A_snapshot",
        }
        if (
            set(self.hit_gain_by_comparison) != expected_comparisons
            or set(self.recall_difference_by_comparison) != expected_comparisons
        ):
            raise ValueError("D1 feature-variant result omitted a registered comparison")
        for key, gain in self.hit_gain_by_comparison.items():
            if self.recall_difference_by_comparison[key] != gain / self.cluster_count:
                raise ValueError("D1 feature-variant recall difference is not pooled by cluster")
        for field_name in (
            "preprocessing_refit",
            "alpha_reselected",
            "ridge_reselected",
            "models_refit",
        ):
            if getattr(self, field_name) is not True:
                raise ValueError(f"D1 feature-variant callback must set {field_name}=True")
        object.__setattr__(
            self, "hit_count_by_model", MappingProxyType(dict(self.hit_count_by_model))
        )
        object.__setattr__(
            self,
            "hit_gain_by_comparison",
            MappingProxyType(dict(self.hit_gain_by_comparison)),
        )
        object.__setattr__(
            self,
            "recall_difference_by_comparison",
            MappingProxyType(dict(self.recall_difference_by_comparison)),
        )
        object.__setattr__(
            self,
            "selected_ridge_by_model",
            MappingProxyType(dict(self.selected_ridge_by_model)),
        )

    @property
    def support_count(self) -> int:
        return self.cluster_count

    @property
    def model_hit_counts(self) -> Mapping[str, int]:
        return self.hit_count_by_model

    def as_mapping(self) -> dict[str, object]:
        return {
            "fold_id": self.fold_id,
            "cluster_count": self.cluster_count,
            "support_count": self.support_count,
            "hit_count_by_model": dict(self.hit_count_by_model),
            "model_hit_counts": dict(self.model_hit_counts),
            "hit_gain_by_comparison": dict(self.hit_gain_by_comparison),
            "recall_difference_by_comparison": dict(self.recall_difference_by_comparison),
            "selected_alpha": self.selected_alpha,
            "selected_ridge_by_model": dict(self.selected_ridge_by_model),
            "preprocessing_refit": self.preprocessing_refit,
            "alpha_reselected": self.alpha_reselected,
            "ridge_reselected": self.ridge_reselected,
            "models_refit": self.models_refit,
        }


def _prepare_replay(
    protocol: D1Protocol,
    paths: Mapping[str, Path],
    identities: Mapping[str, object],
    *,
    progress: ProgressCallback | None,
) -> D1PreparedReplay:
    _verify_fold_bridge(protocol)
    _emit(progress, phase="prepare", item="earthquake_catalog")
    catalog = parse_frozen_catalog_bytes(paths["earthquake_event"].read_bytes())
    _emit(progress, phase="prepare", item="study_area_and_grid")
    domain = build_d1_spatial_domain_from_bytes(paths["study_area"].read_bytes())
    target_layer = build_score_blind_target_layer(protocol, catalog)
    needed_times = sorted(
        {
            issue
            for targets in (*target_layer.fit_targets, *target_layer.assessment_targets)
            for issue in targets.issue_times_us
        }
    )
    anomaly_binding = _mapping(
        _mapping(protocol.config.get("data"), label="D1 data").get("anomaly_features"),
        label="anomaly_features",
    )
    expected_issue_count = int(anomaly_binding.get("snapshot_count", 0))
    expected_cell_count = int(anomaly_binding.get("query_cell_count", 0))
    _emit(
        progress,
        phase="prepare",
        item="selected_feature_row_groups",
        needed_issue_count=len(needed_times),
    )
    features = _load_needed_issue_features(
        paths["anomaly_features"],
        needed_times,
        domain,
        expected_issue_count=expected_issue_count,
        expected_cell_count=expected_cell_count,
    )
    feature_contract = load_d1_feature_contract(protocol.config_path)
    backgrounds: dict[int, D1CausalBackground] = {}
    for index, issue_time in enumerate(needed_times, start=1):
        _emit(
            progress,
            phase="prepare",
            item="causal_background",
            issue_index=index,
            issue_count=len(needed_times),
            issue_time_utc=_iso_utc_from_us(issue_time),
        )
        backgrounds[issue_time] = build_causal_background_components(
            catalog,
            issue_time,
            domain,
        )
    counts = {
        fold_id: _target_counts_by_issue(target_layer.fit_for(fold_id), catalog, domain)
        for fold_id in D1_FOLD_IDS
    }
    return D1PreparedReplay(
        protocol=protocol,
        identities=MappingProxyType(dict(identities)),
        catalog=catalog,
        target_layer=target_layer,
        domain=domain,
        feature_contract=feature_contract,
        features_by_issue=features,
        backgrounds=MappingProxyType(backgrounds),
        counts_by_fold=MappingProxyType(counts),
    )


def _validate_preprocessor_fit_axis(
    preprocessor: D1GroupPreprocessor,
    expected_issue_times_us: Sequence[int],
) -> None:
    observed = tuple(_datetime_to_us(value) for value in preprocessor.fitted_issue_times_utc)
    expected = tuple(int(value) for value in expected_issue_times_us)
    if observed != expected:
        raise ValueError("D1 preprocessor was not fitted on the exact frozen issue axis")


def _fit_outer_model(
    prepared: D1PreparedReplay,
    *,
    fold_id: str,
    model_spec: Mapping[str, Any],
    alpha_selection: D1ParameterSelection,
    ridge_selection: D1ParameterSelection | None,
) -> tuple[D1FoldModelResult, D1GroupPreprocessor | None, ConditionalSpatialRidgeFit | None]:
    model_id = str(model_spec.get("id"))
    base = str(model_spec.get("base"))
    if base not in {"B0", "R30"}:
        raise ValueError("D1 model base changed")
    selected_groups = tuple(
        str(item) for item in _sequence(model_spec.get("feature_groups"), label="feature groups")
    )
    fit_targets = prepared.target_layer.fit_for(fold_id)
    fit_times = fit_targets.issue_times_us
    counts = prepared.counts_by_fold[fold_id]
    if not selected_groups:
        if ridge_selection is not None:
            raise ValueError("background-only D1 model unexpectedly selected ridge")
        diagnostic = D1TrainingDiagnostic(
            fit_issue_count=len(fit_times),
            fit_catalog_m4plus_event_count=fit_targets.event_count,
            coefficient_names=(),
            active_coefficients=(),
            coefficients=(),
            objective=None,
            iteration_count=0,
        )
        result = D1FoldModelResult(
            fold_id=fold_id,
            model_id=model_id,
            base=cast(Literal["B0", "R30"], base),
            feature_groups=(),
            alpha_selection=alpha_selection,
            ridge_selection=None,
            training_diagnostic=diagnostic,
        )
        return result, None, None
    if ridge_selection is None:
        raise ValueError("D1 feature model omitted ridge selection")
    training_features = tuple(prepared.features_by_issue[issue] for issue in fit_times)
    preprocessor = D1GroupPreprocessor.fit(
        prepared.feature_contract,
        selected_groups,
        training_features,
    )
    _validate_preprocessor_fit_axis(preprocessor, fit_times)
    designs = {
        issue: preprocessor.transform(prepared.features_by_issue[issue]) for issue in fit_times
    }
    training_issues = tuple(
        _training_issue(
            issue,
            base_mass=_base_mass(prepared.backgrounds[issue], base, alpha_selection.selected_value),
            design=designs[issue].values,
            counts=counts[issue],
        )
        for issue in fit_times
    )
    fitted = fit_conditional_spatial_ridge(
        training_issues,
        ridge_lambda=ridge_selection.selected_value,
        active_coefficients=preprocessor.active_coefficients,
    )
    diagnostic = D1TrainingDiagnostic(
        fit_issue_count=len(training_issues),
        fit_catalog_m4plus_event_count=fitted.training_event_count,
        coefficient_names=preprocessor.output_column_names,
        active_coefficients=tuple(bool(value) for value in fitted.active_coefficients),
        coefficients=tuple(float(value) for value in fitted.coefficients),
        objective=float(fitted.objective),
        iteration_count=fitted.iteration_count,
    )
    result = D1FoldModelResult(
        fold_id=fold_id,
        model_id=model_id,
        base=cast(Literal["B0", "R30"], base),
        feature_groups=selected_groups,
        alpha_selection=alpha_selection,
        ridge_selection=ridge_selection,
        training_diagnostic=diagnostic,
    )
    return result, preprocessor, fitted


def _predict_model_mass(
    prepared: D1PreparedReplay,
    *,
    issue_time_us: int,
    base: str,
    selected_alpha: float,
    preprocessor: D1GroupPreprocessor | None,
    fitted: ConditionalSpatialRidgeFit | None,
) -> FloatArray:
    base_mass = _base_mass(prepared.backgrounds[issue_time_us], base, selected_alpha)
    if preprocessor is None or fitted is None:
        if preprocessor is not None or fitted is not None:
            raise ValueError(
                "D1 model preprocessing and fit must either both exist or both be absent"
            )
        return _readonly_float(base_mass)
    design = preprocessor.transform(prepared.features_by_issue[issue_time_us])
    return fitted.predict(_log_mass(base_mass), design.values)


def _make_issue_model_score(
    *,
    fold_id: str,
    issue: D1IssueFeatures,
    horizon_days: int,
    model_id: str,
    mass: FloatArray,
    grid: SpatialGrid,
) -> D1IssueModelScore:
    ranking = _full_ranking(mass, grid)
    prefixes = select_alarm_prefixes(mass, grid)
    return D1IssueModelScore(
        fold_id=fold_id,
        issue_id=issue.issue_report_id,
        issue_time_us=_datetime_to_us(issue.issue_time_utc),
        horizon_days=horizon_days,
        model_id=model_id,
        cell_mass=mass,
        ranking_indices=ranking,
        alarm_prefixes=prefixes,
    )


def _outcomes_for_unit(
    prepared: D1PreparedReplay,
    *,
    fold_id: str,
    model_id: str,
    scores_by_key: Mapping[tuple[int, int], D1IssueModelScore],
) -> tuple[D1ObservedClusterOutcome, ...]:
    outcomes: list[D1ObservedClusterOutcome] = []
    catalog = prepared.catalog
    grid = prepared.domain.operational_grid
    for horizon in D1_HORIZONS_DAYS:
        active = prepared.target_layer.active_clusters(fold_id, horizon)
        for cluster in active:
            representative = cluster.representative(horizon)
            if representative is None or representative.fold_id != fold_id:
                raise AssertionError("D1 active cluster lost its horizon representative")
            issue_time = representative.assigned_issue_time_us
            score = scores_by_key[(horizon, issue_time)]
            event_index = representative.event_index
            cell_index = prepared.domain.locator.locate_lonlat(
                float(catalog.longitude[event_index]),
                float(catalog.latitude[event_index]),
            )
            if cell_index is None:
                outcomes.append(
                    D1ObservedClusterOutcome(
                        cluster_id=cluster.identity_sha256,
                        fold_id=fold_id,
                        issue_id=score.issue_id,
                        issue_time_us=issue_time,
                        horizon_days=horizon,
                        model_id=model_id,
                        representative_cell_index=None,
                        outside_support=True,
                        log_density=None,
                        hit_by_area=tuple(False for _ in D1_AREA_BUDGETS_KM2),
                    )
                )
                continue
            probability = float(score.cell_mass[cell_index])
            area = float(grid.clipped_area_km2[cell_index])
            if probability <= 0.0 or area <= 0.0:
                raise FloatingPointError("inside-support cluster has zero D1 density")
            hit_by_area = tuple(
                bool(np.any(prefix.selected_indices == cell_index))
                for prefix in score.alarm_prefixes
            )
            outcomes.append(
                D1ObservedClusterOutcome(
                    cluster_id=cluster.identity_sha256,
                    fold_id=fold_id,
                    issue_id=score.issue_id,
                    issue_time_us=issue_time,
                    horizon_days=horizon,
                    model_id=model_id,
                    representative_cell_index=cell_index,
                    outside_support=False,
                    log_density=math.log(probability) - math.log(area),
                    hit_by_area=hit_by_area,
                )
            )
    return tuple(outcomes)


def _score_model_unit(
    prepared: D1PreparedReplay,
    *,
    fold_id: str,
    fold_model: D1FoldModelResult,
    preprocessor: D1GroupPreprocessor | None,
    fitted: ConditionalSpatialRidgeFit | None,
) -> tuple[tuple[D1IssueModelScore, ...], tuple[D1ObservedClusterOutcome, ...]]:
    scores: list[D1IssueModelScore] = []
    scores_by_key: dict[tuple[int, int], D1IssueModelScore] = {}
    grid = prepared.domain.operational_grid
    for horizon in D1_HORIZONS_DAYS:
        assessment = prepared.target_layer.assessment_for(fold_id, horizon)
        for issue_time in assessment.issue_times_us:
            mass = _predict_model_mass(
                prepared,
                issue_time_us=issue_time,
                base=fold_model.base,
                selected_alpha=fold_model.alpha_selection.selected_value,
                preprocessor=preprocessor,
                fitted=fitted,
            )
            score = _make_issue_model_score(
                fold_id=fold_id,
                issue=prepared.features_by_issue[issue_time],
                horizon_days=horizon,
                model_id=fold_model.model_id,
                mass=mass,
                grid=grid,
            )
            key = (horizon, issue_time)
            if key in scores_by_key:
                raise ValueError("duplicate D1 assessment issue/horizon frame")
            scores_by_key[key] = score
            scores.append(score)
    expected_frame_count = sum(
        len(prepared.target_layer.assessment_for(fold_id, horizon).issue_times_us)
        for horizon in D1_HORIZONS_DAYS
    )
    if len(scores) != expected_frame_count:
        raise AssertionError("D1 assessment frame count changed")
    outcomes = _outcomes_for_unit(
        prepared,
        fold_id=fold_id,
        model_id=fold_model.model_id,
        scores_by_key=scores_by_key,
    )
    return tuple(scores), outcomes


def _expected_support(
    prepared: D1PreparedReplay,
) -> Mapping[int, tuple[tuple[str, str, str], ...]]:
    support: dict[int, tuple[tuple[str, str, str], ...]] = {}
    for horizon in D1_HORIZONS_DAYS:
        items: list[tuple[str, str, str]] = []
        for cluster in prepared.target_layer.clusters:
            representative = cluster.representative(horizon)
            if representative is None:
                continue
            issue = prepared.features_by_issue[representative.assigned_issue_time_us]
            items.append(
                (
                    cluster.identity_sha256,
                    representative.fold_id,
                    issue.issue_report_id,
                )
            )
        support[horizon] = tuple(sorted(items))
    return MappingProxyType(support)


@_single_threaded
def evaluate_d1_feature_variant_fold(
    prepared: D1PreparedReplay,
    fold_id: str,
    pseudo_features: Mapping[int, D1IssueFeatures],
) -> D1FeatureVariantFoldResult:
    """Refit one pseudo-feature fold and return the three registered placebo statistics.

    The result contains integer hit gains as sufficient statistics so the
    placebo orchestrator can pool the three folds without averaging fold
    recalls.  No checkpoint or observed-result file is written by this callback.
    """

    if not isinstance(prepared, D1PreparedReplay) or fold_id not in D1_FOLD_IDS:
        raise TypeError("D1 feature-variant callback requires a prepared replay and fold")
    fit_times = prepared.target_layer.fit_for(fold_id).issue_times_us
    assessment_times = {
        issue
        for horizon in D1_HORIZONS_DAYS
        for issue in prepared.target_layer.assessment_for(fold_id, horizon).issue_times_us
    }
    needed = set(fit_times) | assessment_times
    if not needed <= set(pseudo_features):
        raise ValueError("D1 pseudo features omit a fit or assessment issue")
    replacement = dict(prepared.features_by_issue)
    for issue_time in needed:
        pseudo = pseudo_features[issue_time]
        expected = prepared.features_by_issue[issue_time]
        if (
            not isinstance(pseudo, D1IssueFeatures)
            or _datetime_to_us(pseudo.issue_time_utc) != issue_time
            or pseudo.issue_report_id != expected.issue_report_id
        ):
            raise ValueError("D1 pseudo feature issue identity changed")
        grid_pairs = (
            (pseudo.grid.grid_id, expected.grid.grid_id),
            (pseudo.grid.cell_ids, expected.grid.cell_ids),
            (pseudo.grid.rows, expected.grid.rows),
            (pseudo.grid.columns, expected.grid.columns),
            (pseudo.grid.query_x_m, expected.grid.query_x_m),
            (pseudo.grid.query_y_m, expected.grid.query_y_m),
            (pseudo.grid.clipped_area_km2, expected.grid.clipped_area_km2),
        )
        for left, right in grid_pairs:
            equal = left == right if isinstance(left, str | tuple) else np.array_equal(left, right)
            if not equal:
                raise ValueError("D1 pseudo feature changed the frozen spatial grid")
        replacement[issue_time] = pseudo
    variant = replace(prepared, features_by_issue=MappingProxyType(replacement))
    alpha = _select_alpha(
        fit_times,
        variant.counts_by_fold[fold_id],
        variant.backgrounds,
        variant.domain.operational_grid,
    )
    required_models = ("B0_C", "B0_C_A_snapshot", "B0_C_A_dynamic")
    specs = {str(item["id"]): item for item in _model_specs(variant.protocol)}
    hit_counts: dict[str, int] = {}
    selected_ridge: dict[str, float] = {}
    expected_clusters = {
        item.identity_sha256 for item in variant.target_layer.active_clusters(fold_id, 30)
    }
    for model_id in required_models:
        model_spec = specs[model_id]
        groups = tuple(
            str(item)
            for item in _sequence(model_spec.get("feature_groups"), label="feature groups")
        )
        ridge = _select_ridge(
            contract=variant.feature_contract,
            selected_groups=groups,
            base=str(model_spec["base"]),
            selected_alpha=alpha.selected_value,
            fit_issue_times_us=fit_times,
            features_by_issue=variant.features_by_issue,
            counts_by_issue=variant.counts_by_fold[fold_id],
            backgrounds=variant.backgrounds,
            grid=variant.domain.operational_grid,
        )
        fold_model, preprocessor, fitted = _fit_outer_model(
            variant,
            fold_id=fold_id,
            model_spec=model_spec,
            alpha_selection=alpha,
            ridge_selection=ridge,
        )
        _, outcomes = _score_model_unit(
            variant,
            fold_id=fold_id,
            fold_model=fold_model,
            preprocessor=preprocessor,
            fitted=fitted,
        )
        primary = tuple(item for item in outcomes if item.horizon_days == 30)
        if {item.cluster_id for item in primary} != expected_clusters:
            raise ValueError("D1 feature-variant model changed primary cluster support")
        hit_counts[model_id] = sum(item.hit_by_area[2] for item in primary)
        selected_ridge[model_id] = ridge.selected_value
    comparisons = {
        "B0_C_A_snapshot_minus_B0_C": hit_counts["B0_C_A_snapshot"] - hit_counts["B0_C"],
        "B0_C_A_dynamic_minus_B0_C": hit_counts["B0_C_A_dynamic"] - hit_counts["B0_C"],
        "B0_C_A_dynamic_minus_B0_C_A_snapshot": hit_counts["B0_C_A_dynamic"]
        - hit_counts["B0_C_A_snapshot"],
    }
    cluster_count = len(expected_clusters)
    return D1FeatureVariantFoldResult(
        fold_id=fold_id,
        cluster_count=cluster_count,
        hit_count_by_model=hit_counts,
        hit_gain_by_comparison=comparisons,
        recall_difference_by_comparison={
            key: value / cluster_count for key, value in comparisons.items()
        },
        selected_alpha=alpha.selected_value,
        selected_ridge_by_model=selected_ridge,
    )


def _frame_table(score: D1IssueModelScore, grid: SpatialGrid) -> pa.Table:
    order = np.asarray(score.ranking_indices, dtype=np.int64)
    count = order.size
    ranks = np.arange(1, count + 1, dtype=np.int32)
    strength = score.cell_mass / grid.clipped_area_km2
    data: dict[str, object] = {
        "fold_id": pa.array([score.fold_id] * count, type=pa.string()),
        "issue_id": pa.array([score.issue_id] * count, type=pa.string()),
        "issue_time_utc": pa.array(
            [score.issue_time_us] * count,
            type=pa.timestamp("us", tz="UTC"),
        ),
        "horizon_days": pa.array([score.horizon_days] * count, type=pa.int16()),
        "model_id": pa.array([score.model_id] * count, type=pa.string()),
        "cell_index": pa.array(order, type=pa.int32()),
        "cell_id": pa.array([grid.cell_ids[index] for index in order], type=pa.string()),
        "row": pa.array(grid.rows[order], type=pa.int32()),
        "column": pa.array(grid.columns[order], type=pa.int32()),
        "query_x_m": pa.array(grid.query_xy_km[order, 0] * 1_000.0, type=pa.float64()),
        "query_y_m": pa.array(grid.query_xy_km[order, 1] * 1_000.0, type=pa.float64()),
        "clipped_area_km2": pa.array(grid.clipped_area_km2[order], type=pa.float64()),
        "relative_cell_mass": pa.array(score.cell_mass[order], type=pa.float64()),
        "relative_strength_per_km2": pa.array(strength[order], type=pa.float64()),
        "rank": pa.array(ranks, type=pa.int32()),
    }
    for prefix in score.alarm_prefixes:
        label = str(int(prefix.budget_km2))
        data[f"alarm_{label}"] = pa.array(
            ranks <= prefix.selected_indices.size,
            type=pa.bool_(),
        )
    return pa.table(data)


def _atomic_parquet_writer(
    destination: Path,
    tables: Sequence[pa.Table],
) -> None:
    if not tables:
        raise ValueError("cannot write an empty D1 cell-score parquet")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    writer: pq.ParquetWriter | None = None
    try:
        for table in tables:
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary,
                    table.schema,
                    compression="zstd",
                    use_dictionary=["fold_id", "issue_id", "model_id", "cell_id"],
                )
            writer.write_table(table, row_group_size=table.num_rows)
        assert writer is not None
        writer.close()
        writer = None
        os.replace(temporary, destination)
    except Exception:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
        raise


def _unit_key(fold_id: str, model_id: str) -> str:
    if fold_id not in D1_FOLD_IDS or model_id not in D1_MODEL_ORDER:
        raise ValueError("invalid D1 checkpoint unit identity")
    return f"{fold_id}__{model_id}"


def _unit_paths(output_root: Path, fold_id: str, model_id: str) -> tuple[Path, Path]:
    base = output_root / "checkpoints" / "observed_units" / _unit_key(fold_id, model_id)
    return base.with_suffix(".json"), base.with_suffix(".parquet")


def _write_unit_checkpoint(
    output_root: Path,
    identities: Mapping[str, object],
    fold_model: D1FoldModelResult,
    issues: tuple[D1IssueModelScore, ...],
    outcomes: tuple[D1ObservedClusterOutcome, ...],
    grid: SpatialGrid,
) -> None:
    metadata_path, cell_path = _unit_paths(output_root, fold_model.fold_id, fold_model.model_id)
    ordered_issues = tuple(sorted(issues, key=lambda item: (item.issue_time_us, item.horizon_days)))
    _atomic_parquet_writer(
        cell_path,
        tuple(_frame_table(item, grid) for item in ordered_issues),
    )
    issue_summaries = []
    for row_group_index, issue in enumerate(ordered_issues):
        summary = issue.as_mapping(include_cell_mass=False)
        summary["row_group_index"] = row_group_index
        issue_summaries.append(summary)
    scientific_summary = {
        "fold_model": fold_model.as_mapping(),
        "outcomes": [item.as_mapping() for item in outcomes],
        "issue_alarm_outcomes": issue_summaries,
    }
    write_json_atomic(
        metadata_path,
        {
            "schema_version": 1,
            "result_kind": "d1_observed_fold_model_checkpoint",
            "identities": dict(identities),
            "unit": _unit_key(fold_model.fold_id, fold_model.model_id),
            **scientific_summary,
            "scientific_summary_sha256": hashlib.sha256(
                canonical_json_bytes(scientific_summary)
            ).hexdigest(),
            "cell_scores": {
                "path": cell_path.name,
                "file_sha256": sha256_file(cell_path),
                "row_group_count": len(ordered_issues),
            },
            "status": "completed",
        },
    )


def _constant_table_value(table: pa.Table, name: str) -> object:
    return _single_value(table, name, label=f"checkpoint {name}")


def _score_from_checkpoint_table(
    summary: Mapping[str, Any],
    table: pa.Table,
    grid: SpatialGrid,
) -> D1IssueModelScore:
    if table.num_rows != grid.cell_count:
        raise ValueError("D1 checkpoint frame row count changed")
    cell_indices = np.asarray(
        table["cell_index"].combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.int64,
    )
    ranks = np.asarray(
        table["rank"].combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.int64,
    )
    if not np.array_equal(ranks, np.arange(1, grid.cell_count + 1)):
        raise ValueError("D1 checkpoint frame rank is not one-based and complete")
    if set(int(value) for value in cell_indices) != set(range(grid.cell_count)):
        raise ValueError("D1 checkpoint frame cell index is not a full grid permutation")
    ranked_mass = np.asarray(
        table["relative_cell_mass"].combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.float64,
    )
    mass = np.empty(grid.cell_count, dtype=np.float64)
    mass[cell_indices] = ranked_mass
    expected_ranking = _full_ranking(mass, grid)
    if not np.array_equal(cell_indices, expected_ranking):
        raise ValueError("D1 checkpoint ranking is not the frozen mass/area ranking")
    if int(summary.get("cell_count", -1)) != grid.cell_count:
        raise ValueError("D1 checkpoint frame cell count changed")
    expected_cell_ids = [grid.cell_ids[index] for index in cell_indices]
    if table["cell_id"].combine_chunks().to_pylist() != expected_cell_ids:
        raise ValueError("D1 checkpoint ranked cell identities changed")
    grid_vectors = {
        "row": np.asarray(grid.rows[cell_indices], dtype=np.int64),
        "column": np.asarray(grid.columns[cell_indices], dtype=np.int64),
        "clipped_area_km2": np.asarray(grid.clipped_area_km2[cell_indices], dtype=np.float64),
    }
    for name, expected in grid_vectors.items():
        observed = np.asarray(
            table[name].combine_chunks().to_numpy(zero_copy_only=False),
            dtype=expected.dtype,
        )
        if not np.array_equal(observed, expected):
            raise ValueError(f"D1 checkpoint ranked grid field {name} changed")
    stored_strength = np.asarray(
        table["relative_strength_per_km2"].combine_chunks().to_numpy(zero_copy_only=False),
        dtype=np.float64,
    )
    expected_strength = ranked_mass / grid.clipped_area_km2[cell_indices]
    if not np.array_equal(stored_strength, expected_strength):
        raise ValueError("D1 checkpoint relative strength is inconsistent with mass/area")
    issue_time = _parse_iso_datetime(summary["issue_time_utc"], label="checkpoint issue")
    fold_id = str(summary["fold_id"])
    issue_id = str(summary["issue_id"])
    horizon = int(summary["horizon_days"])
    model_id = str(summary["model_id"])
    constants = {
        "fold_id": fold_id,
        "issue_id": issue_id,
        "horizon_days": horizon,
        "model_id": model_id,
    }
    for name, expected_value in constants.items():
        if _constant_table_value(table, name) != expected_value:
            raise ValueError(f"D1 checkpoint frame changed constant {name}")
    table_issue = _constant_table_value(table, "issue_time_utc")
    if not isinstance(table_issue, datetime) or _datetime_to_us(table_issue) != _datetime_to_us(
        issue_time
    ):
        raise ValueError("D1 checkpoint frame changed constant issue_time_utc")
    prefix_counts = tuple(int(value) for value in summary["alarm_prefix_counts"])
    actual_areas = tuple(float(value) for value in summary["actual_area_km2"])
    if len(prefix_counts) != len(D1_AREA_BUDGETS_KM2) or len(actual_areas) != len(
        D1_AREA_BUDGETS_KM2
    ):
        raise ValueError("D1 checkpoint frame omitted an alarm budget")
    prefixes = select_alarm_prefixes(mass, grid)
    if prefix_counts != tuple(item.selected_indices.size for item in prefixes):
        raise ValueError("D1 checkpoint alarm prefix counts changed")
    for budget, stored_area, expected_prefix in zip(
        D1_AREA_BUDGETS_KM2, actual_areas, prefixes, strict=True
    ):
        if not math.isclose(
            stored_area,
            expected_prefix.actual_area_km2,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError("D1 checkpoint actual alarm area changed")
        alarm_column = f"alarm_{int(budget)}"
        stored_alarm = np.asarray(
            table[alarm_column].combine_chunks().to_numpy(zero_copy_only=False),
            dtype=np.bool_,
        )
        expected_alarm = ranks <= expected_prefix.selected_indices.size
        if not np.array_equal(stored_alarm, expected_alarm):
            raise ValueError("D1 checkpoint alarm membership changed")
    return D1IssueModelScore(
        fold_id=fold_id,
        issue_id=issue_id,
        issue_time_us=_datetime_to_us(issue_time),
        horizon_days=horizon,
        model_id=model_id,
        cell_mass=mass,
        ranking_indices=cell_indices,
        alarm_prefixes=prefixes,
    )


def _read_unit_checkpoint(
    output_root: Path,
    identities: Mapping[str, object],
    fold_id: str,
    model_id: str,
    grid: SpatialGrid,
    *,
    prepared: D1PreparedReplay | None = None,
) -> tuple[D1FoldModelResult, tuple[D1IssueModelScore, ...], tuple[D1ObservedClusterOutcome, ...]]:
    metadata_path, expected_cell_path = _unit_paths(output_root, fold_id, model_id)
    document = _mapping(
        json.loads(metadata_path.read_text(encoding="utf-8")), label="D1 unit checkpoint"
    )
    if document.get("identities") != dict(identities):
        raise ValueError("D1 unit checkpoint identities do not match this replay")
    if (
        document.get("unit") != _unit_key(fold_id, model_id)
        or document.get("status") != "completed"
    ):
        raise ValueError("D1 unit checkpoint identity/status is invalid")
    scientific_summary = {
        "fold_model": document.get("fold_model"),
        "outcomes": document.get("outcomes"),
        "issue_alarm_outcomes": document.get("issue_alarm_outcomes"),
    }
    observed_summary_hash = hashlib.sha256(canonical_json_bytes(scientific_summary)).hexdigest()
    if document.get("scientific_summary_sha256") != observed_summary_hash:
        raise ValueError("D1 unit checkpoint scientific summary changed")
    cell_binding = _mapping(document.get("cell_scores"), label="unit cell scores")
    cell_path = metadata_path.parent / str(cell_binding.get("path"))
    if cell_path.resolve() != expected_cell_path.resolve():
        raise ValueError("D1 unit checkpoint points to another cell-score file")
    if sha256_file(cell_path) != cell_binding.get("file_sha256"):
        raise ValueError("D1 unit checkpoint cell-score SHA-256 changed")
    parquet = pq.ParquetFile(cell_path)
    issue_nodes = _sequence(document.get("issue_alarm_outcomes"), label="unit issue outcomes")
    if parquet.metadata.num_row_groups != len(issue_nodes):
        raise ValueError("D1 unit checkpoint row-group count changed")
    issues = []
    for expected_index, raw_summary in enumerate(issue_nodes):
        summary = _mapping(raw_summary, label="unit issue outcome")
        if int(summary.get("row_group_index", -1)) != expected_index:
            raise ValueError("D1 unit checkpoint row-group ordering changed")
        issues.append(
            _score_from_checkpoint_table(
                summary,
                parquet.read_row_group(expected_index),
                grid,
            )
        )
    fold_model = D1FoldModelResult.from_mapping(
        _mapping(document.get("fold_model"), label="fold model")
    )
    outcomes = tuple(
        D1ObservedClusterOutcome.from_mapping(_mapping(item, label="cluster outcome"))
        for item in _sequence(document.get("outcomes"), label="unit outcomes")
    )
    if fold_model.fold_id != fold_id or fold_model.model_id != model_id:
        raise ValueError("D1 checkpoint fold/model result changed unit identity")
    if any(item.fold_id != fold_id or item.model_id != model_id for item in issues) or any(
        item.fold_id != fold_id or item.model_id != model_id for item in outcomes
    ):
        raise ValueError("D1 checkpoint frame/outcome changed unit identity")
    if prepared is not None:
        spec = next(
            item for item in _model_specs(prepared.protocol) if str(item.get("id")) == model_id
        )
        expected_groups = tuple(
            str(item) for item in _sequence(spec.get("feature_groups"), label="feature groups")
        )
        if fold_model.base != str(spec.get("base")) or fold_model.feature_groups != expected_groups:
            raise ValueError("D1 checkpoint fold model differs from the frozen model contract")
        scores_by_key = {(item.horizon_days, item.issue_time_us): item for item in issues}
        expected_axes = {
            (horizon, issue_time)
            for horizon in D1_HORIZONS_DAYS
            for issue_time in prepared.target_layer.assessment_for(fold_id, horizon).issue_times_us
        }
        if set(scores_by_key) != expected_axes or len(scores_by_key) != len(issues):
            raise ValueError("D1 checkpoint changed the frozen assessment frame axis")
        for (_, issue_time), score in scores_by_key.items():
            if score.issue_id != prepared.features_by_issue[issue_time].issue_report_id:
                raise ValueError("D1 checkpoint frame changed its frozen issue identity")
        expected_outcomes = _outcomes_for_unit(
            prepared,
            fold_id=fold_id,
            model_id=model_id,
            scores_by_key=scores_by_key,
        )
        if [item.as_mapping() for item in outcomes] != [
            item.as_mapping() for item in expected_outcomes
        ]:
            raise ValueError("D1 checkpoint cluster outcomes do not match their score frames")
    return fold_model, tuple(issues), outcomes


def _identity_state_fields(identities: Mapping[str, object]) -> dict[str, object]:
    required = ("contract_sha256", "input_sha256", "git_commit")
    if any(key not in identities for key in required):
        raise ValueError("D1 replay identities omitted a recovery field")
    return {key: identities[key] for key in required}


def _load_or_initialize_state(
    output_root: Path,
    identities: Mapping[str, object],
    *,
    workers: int,
) -> dict[str, object]:
    """Load a compatible recovery state or atomically create ``prepared``."""

    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / _STATE_FILE
    identity_fields = _identity_state_fields(identities)
    if state_path.exists():
        state = dict(
            _mapping(
                json.loads(state_path.read_text(encoding="utf-8")),
                label="D1 recovery state",
            )
        )
        for key, expected in identity_fields.items():
            if state.get(key) != expected:
                raise ValueError(f"D1 recovery refused: {key} changed")
        if state.get("status") not in {"prepared", "running", "invalid_run", "completed"}:
            raise ValueError("D1 recovery state has an unknown status")
        if state.get("workers") != workers or state.get("blas_threads_per_worker") != 1:
            raise ValueError("D1 recovery refused: resource plan changed")
        completed = state.get("completed_units")
        if not isinstance(completed, list) or len(completed) != len(set(completed)):
            raise ValueError("D1 recovery completed_units is invalid")
        valid_units = {_unit_key(fold, model) for fold in D1_FOLD_IDS for model in D1_MODEL_ORDER}
        if not set(completed) <= valid_units:
            raise ValueError("D1 recovery names an unknown fold/model unit")
        return state
    checkpoint_root = output_root / "checkpoints" / "observed_units"
    if checkpoint_root.exists() and any(checkpoint_root.iterdir()):
        raise ValueError("D1 checkpoint files exist without an authoritative state.json")
    state = {
        **identity_fields,
        "fold": None,
        "model": None,
        "placebo_kind": None,
        "last_replication": None,
        "completed_units": [],
        "status": "prepared",
        "workers": workers,
        "blas_threads_per_worker": 1,
    }
    write_json_atomic(state_path, state)
    return state


def _write_state(output_root: Path, state: Mapping[str, object]) -> None:
    write_json_atomic(output_root / _STATE_FILE, dict(state))


def _write_partial_index(
    output_root: Path,
    identities: Mapping[str, object],
    units: Mapping[
        str,
        tuple[
            D1FoldModelResult,
            tuple[D1IssueModelScore, ...],
            tuple[D1ObservedClusterOutcome, ...],
        ],
    ],
) -> None:
    fold_models = [value[0].as_mapping() for _, value in sorted(units.items())]
    write_json_atomic(
        output_root / _PARTIAL_RESULT_FILE,
        {
            "schema_version": 1,
            "result_kind": "observed_replay_partial",
            "identities": dict(identities),
            "completed_units": sorted(units),
            "fold_models": fold_models,
            "status": "running",
        },
    )


def _write_final_cell_scores(
    output_root: Path,
    issues: tuple[D1IssueModelScore, ...],
    grid: SpatialGrid,
) -> Path:
    by_key = {
        (item.fold_id, item.issue_time_us, item.horizon_days, item.model_id): item
        for item in issues
    }
    if len(by_key) != len(issues):
        raise ValueError("D1 final cell table received duplicate assessment frames")
    ordered: list[D1IssueModelScore] = []
    for fold_id in D1_FOLD_IDS:
        frame_axes = sorted(
            {(item.issue_time_us, item.horizon_days) for item in issues if item.fold_id == fold_id}
        )
        for issue_time, horizon in frame_axes:
            for model_id in D1_MODEL_ORDER:
                try:
                    ordered.append(by_key[(fold_id, issue_time, horizon, model_id)])
                except KeyError as exc:
                    raise ValueError("D1 final cell table omitted one model frame") from exc
    destination = output_root / _FINAL_CELL_FILE
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    writer: pq.ParquetWriter | None = None
    try:
        for score in ordered:
            table = _frame_table(score, grid)
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary,
                    table.schema,
                    compression="zstd",
                    use_dictionary=["fold_id", "issue_id", "model_id", "cell_id"],
                )
            writer.write_table(table, row_group_size=table.num_rows)
        if writer is None:
            raise ValueError("D1 final cell table has no assessment frames")
        writer.close()
        writer = None
        os.replace(temporary, destination)
    except Exception:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _assemble_result(
    prepared: D1PreparedReplay,
    units: Mapping[
        str,
        tuple[
            D1FoldModelResult,
            tuple[D1IssueModelScore, ...],
            tuple[D1ObservedClusterOutcome, ...],
        ],
    ],
    *,
    workers: int,
) -> D1ObservedReplayResult:
    expected_keys = {_unit_key(fold, model) for fold in D1_FOLD_IDS for model in D1_MODEL_ORDER}
    if set(units) != expected_keys:
        raise ValueError("D1 observed replay cannot assemble before all 18 units complete")
    model_order = {model: index for index, model in enumerate(D1_MODEL_ORDER)}
    fold_order = {fold: index for index, fold in enumerate(D1_FOLD_IDS)}
    fold_models = tuple(
        sorted(
            (value[0] for value in units.values()),
            key=lambda item: (fold_order[item.fold_id], model_order[item.model_id]),
        )
    )
    issues = tuple(
        sorted(
            (item for value in units.values() for item in value[1]),
            key=lambda item: (
                fold_order[item.fold_id],
                item.issue_time_us,
                item.horizon_days,
                model_order[item.model_id],
            ),
        )
    )
    outcomes = tuple(
        sorted(
            (item for value in units.values() for item in value[2]),
            key=lambda item: (
                item.horizon_days,
                item.cluster_id,
                model_order[item.model_id],
            ),
        )
    )
    return D1ObservedReplayResult(
        protocol_version="d1.0.0",
        identities=prepared.identities,
        workers=workers,
        expected_support_by_horizon=_expected_support(prepared),
        fold_models=fold_models,
        outcomes=outcomes,
        issues=issues,
    )


@_single_threaded
def prepare_d1_replay(
    project_root: Path,
    config_path: Path,
    manifest_path: Path,
    *,
    progress: ProgressCallback | None = None,
) -> D1PreparedReplay:
    """Verify and prepare the shared causal D1 inputs without scoring or writing output."""

    root = Path(project_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    _assert_resource_boundary(1)
    protocol = load_d1_protocol(root)
    supplied_config = _resolve_supplied_path(root, Path(config_path))
    supplied_manifest = _resolve_supplied_path(root, Path(manifest_path))
    if supplied_config != protocol.config_path or supplied_manifest != protocol.water_level_path:
        raise ValueError("D1 preparation accepts only the frozen config and water-level manifest")
    git_commit = _git_commit(root)
    _emit(progress, phase="identity", item="hash_inputs")
    paths, input_hashes = _verified_input_paths(protocol)
    input_sha = hashlib.sha256(canonical_json_bytes(input_hashes)).hexdigest()
    identities: dict[str, object] = {
        "contract_sha256": protocol.config_sha256,
        "manifest_content_sha256": protocol.water_level_content_sha256,
        "input_sha256": input_sha,
        "input_files": input_hashes,
        "git_commit": git_commit,
    }
    return _prepare_replay(protocol, paths, identities, progress=progress)


@_single_threaded
def run_d1_observed_replay(
    project_root: Path,
    config_path: Path,
    manifest_path: Path,
    output_root: Path,
    *,
    workers: int = 4,
    progress: ProgressCallback | None = None,
) -> D1ObservedReplayResult:
    """Run or resume the six-model observed D1 causal historical replay.

    The function writes one immutable checkpoint per outer-fold/model unit,
    then a compact ``observed_result.json`` and the complete assessment-frame
    ``d1_cell_scores.parquet``.  It never evaluates a placebo or locked test.
    """

    root = Path(project_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    resource_boundary = _assert_resource_boundary(workers)
    _emit(
        progress,
        phase="resource",
        workers=workers,
        observed_execution="serial",
        logical_cores_visible=resource_boundary.logical_cores,
        physical_cores_detected=resource_boundary.physical_cores,
        reserved_physical_cores=_RESERVED_PHYSICAL_CORES,
        max_workers_after_reservation=resource_boundary.max_workers_after_reservation,
        blas_threads_per_worker=1,
    )
    prepared = prepare_d1_replay(
        root,
        config_path,
        manifest_path,
        progress=progress,
    )
    protocol = prepared.protocol
    identities = dict(prepared.identities)
    destination = Path(output_root).resolve()
    state = _load_or_initialize_state(destination, identities, workers=workers)
    units: dict[
        str,
        tuple[
            D1FoldModelResult,
            tuple[D1IssueModelScore, ...],
            tuple[D1ObservedClusterOutcome, ...],
        ],
    ] = {}
    try:
        completed_units = [str(item) for item in cast(list[object], state["completed_units"])]
        for unit in completed_units:
            fold_id, model_id = unit.split("__", maxsplit=1)
            units[unit] = _read_unit_checkpoint(
                destination,
                identities,
                fold_id,
                model_id,
                prepared.domain.operational_grid,
                prepared=prepared,
            )
        if state.get("status") == "completed":
            result = _assemble_result(prepared, units, workers=workers)
            final_cell = destination / _FINAL_CELL_FILE
            final_json = destination / _FINAL_RESULT_FILE
            if not final_cell.is_file() or not final_json.is_file():
                raise ValueError("completed D1 state is missing a final observed artifact")
            final_document = _mapping(
                json.loads(final_json.read_text(encoding="utf-8")),
                label="completed D1 observed result",
            )
            if final_document.get("identities") != identities:
                raise ValueError("completed D1 result identities changed")
            final_cell_binding = _mapping(
                final_document.get("cell_scores"), label="completed D1 cell scores"
            )
            if sha256_file(final_cell) != final_cell_binding.get("file_sha256"):
                raise ValueError("completed D1 cell-score SHA-256 changed")
            return result

        specs = _model_specs(protocol)
        alpha_by_fold: dict[str, D1ParameterSelection] = {}
        for fold_id in D1_FOLD_IDS:
            prior = [
                value[0].alpha_selection for value in units.values() if value[0].fold_id == fold_id
            ]
            if prior:
                first = prior[0]
                if any(item.as_mapping() != first.as_mapping() for item in prior[1:]):
                    raise ValueError("D1 completed fold units disagree on selected alpha")
                alpha_by_fold[fold_id] = first
            else:
                fit_times = prepared.target_layer.fit_for(fold_id).issue_times_us
                alpha_by_fold[fold_id] = _select_alpha(
                    fit_times,
                    prepared.counts_by_fold[fold_id],
                    prepared.backgrounds,
                    prepared.domain.operational_grid,
                )

        for fold_id in D1_FOLD_IDS:
            fit_times = prepared.target_layer.fit_for(fold_id).issue_times_us
            for model_spec in specs:
                model_id = str(model_spec["id"])
                unit = _unit_key(fold_id, model_id)
                if unit in units:
                    continue
                selected_groups = tuple(
                    str(item)
                    for item in _sequence(model_spec.get("feature_groups"), label="feature groups")
                )
                ridge_selection = (
                    None
                    if not selected_groups
                    else _select_ridge(
                        contract=prepared.feature_contract,
                        selected_groups=selected_groups,
                        base=str(model_spec["base"]),
                        selected_alpha=alpha_by_fold[fold_id].selected_value,
                        fit_issue_times_us=fit_times,
                        features_by_issue=prepared.features_by_issue,
                        counts_by_issue=prepared.counts_by_fold[fold_id],
                        backgrounds=prepared.backgrounds,
                        grid=prepared.domain.operational_grid,
                    )
                )
                state.update(
                    {
                        "fold": fold_id,
                        "model": model_id,
                        "status": "running",
                    }
                )
                _write_state(destination, state)
                _emit(progress, phase="observed", fold=fold_id, model=model_id)
                fold_model, preprocessor, fitted = _fit_outer_model(
                    prepared,
                    fold_id=fold_id,
                    model_spec=model_spec,
                    alpha_selection=alpha_by_fold[fold_id],
                    ridge_selection=ridge_selection,
                )
                issue_scores, outcomes = _score_model_unit(
                    prepared,
                    fold_id=fold_id,
                    fold_model=fold_model,
                    preprocessor=preprocessor,
                    fitted=fitted,
                )
                _write_unit_checkpoint(
                    destination,
                    identities,
                    fold_model,
                    issue_scores,
                    outcomes,
                    prepared.domain.operational_grid,
                )
                units[unit] = (fold_model, issue_scores, outcomes)
                completed_units.append(unit)
                state.update(
                    {
                        "completed_units": completed_units,
                        "fold": fold_id,
                        "model": model_id,
                        "status": "running",
                    }
                )
                _write_state(destination, state)
                _write_partial_index(destination, identities, units)

        result = _assemble_result(prepared, units, workers=workers)
        cell_path = _write_final_cell_scores(
            destination,
            result.issues,
            prepared.domain.operational_grid,
        )
        compact = result.as_mapping(include_cell_mass=False)
        compact["study_area_km2"] = math.fsum(
            float(value) for value in prepared.domain.operational_grid.clipped_area_km2
        )
        compact["cell_scores"] = {
            "path": cell_path.name,
            "file_sha256": sha256_file(cell_path),
            "frame_count": len(result.issues),
            "row_count": len(result.issues) * prepared.domain.operational_grid.cell_count,
            "ordering": [
                "fold_id",
                "issue_time_utc",
                "horizon_days",
                "model_id_fixed_order",
                "rank",
            ],
        }
        write_json_atomic(destination / _FINAL_RESULT_FILE, compact)
        state.update(
            {
                "fold": None,
                "model": None,
                "completed_units": [
                    _unit_key(fold, model) for fold in D1_FOLD_IDS for model in D1_MODEL_ORDER
                ],
                "status": "completed",
            }
        )
        _write_state(destination, state)
        _emit(progress, phase="observed", status="completed")
        return result
    except Exception:
        if state.get("status") != "completed":
            state["status"] = "invalid_run"
            _write_state(destination, state)
        raise


__all__ = [
    "D1FeatureVariantFoldResult",
    "D1FoldModelResult",
    "D1IssueModelScore",
    "D1ObservedClusterOutcome",
    "D1ObservedReplayResult",
    "D1ParameterSelection",
    "D1PreparedReplay",
    "D1TrainingDiagnostic",
    "evaluate_d1_feature_variant_fold",
    "prepare_d1_replay",
    "run_d1_observed_replay",
]
