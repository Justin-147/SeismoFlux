"""Pure, score-blind location baselines frozen for S1-B.

Every function accepts only an explicitly supplied frozen grid, causally
visible history, or explicitly earlier inner-validation summary.  Returned
cell values are normalized relative spatial mass, not absolute probability.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, cast

import numpy as np
from numpy.typing import NDArray

from seismoflux.background.poisson import GaussianMixtureFamily

FloatArray = NDArray[np.float64]

FROZEN_REGIONAL_TAU_YEARS: Final[tuple[float, ...]] = (1.0, 5.0, 10.0)
FROZEN_KDE_BANDWIDTHS_KM: Final[tuple[float, ...]] = (75.0, 100.0, 150.0, 200.0, 300.0)
FROZEN_R30_ALPHA_CANDIDATES: Final[tuple[float, ...]] = (0.0, 0.25, 0.5, 0.75)
FROZEN_REGIONAL_SIZE_KM: Final = 500.0
INNER_TARGET_MINIMUM: Final = 10

_PARAMETER_TIE_TOLERANCE: Final = 1.0e-12
_BEST_MEAN_TIE_TOLERANCE: Final = 1.0e-15
_MASS_SUM_TOLERANCE: Final = 1.0e-12
_MICROSECONDS_PER_DAY: Final = 86_400_000_000
_MAIN_CATALOG_DELAY_US: Final = _MICROSECONDS_PER_DAY
_RECENT_WINDOW_US: Final = 30 * _MICROSECONDS_PER_DAY
_PARAMETER_SELECTION_EMBARGO_US: Final = 30 * _MICROSECONDS_PER_DAY


def _readonly_float_vector(
    name: str,
    values: object,
    *,
    allow_empty: bool = False,
    positive: bool = False,
) -> FloatArray:
    result = np.array(values, dtype=np.float64, copy=True, order="C")
    if result.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if not allow_empty and result.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    if positive and np.any(result <= 0.0):
        raise ValueError(f"{name} must contain only positive values")
    result.setflags(write=False)
    return result


def _readonly_relative_mass(values: object) -> FloatArray:
    result = _readonly_float_vector("cell_relative_mass", values)
    if np.any(result < 0.0):
        raise ValueError("cell relative mass must be non-negative")
    if not math.isclose(
        math.fsum(float(value) for value in result),
        1.0,
        rel_tol=0.0,
        abs_tol=_MASS_SUM_TOLERANCE,
    ):
        raise ValueError("cell relative mass must sum to one")
    return result


@dataclass(frozen=True, slots=True)
class FrozenSpatialGrid:
    """Target-independent representative points and exact clipped cell areas."""

    x_km: FloatArray
    y_km: FloatArray
    area_km2: FloatArray

    def __post_init__(self) -> None:
        x = _readonly_float_vector("grid x_km", self.x_km)
        y = _readonly_float_vector("grid y_km", self.y_km)
        area = _readonly_float_vector("grid area_km2", self.area_km2, positive=True)
        if not (x.shape == y.shape == area.shape):
            raise ValueError("grid coordinates and areas must have one common shape")
        object.__setattr__(self, "x_km", x)
        object.__setattr__(self, "y_km", y)
        object.__setattr__(self, "area_km2", area)

    @property
    def cell_count(self) -> int:
        return int(self.area_km2.size)

    @property
    def total_area_km2(self) -> float:
        return math.fsum(float(value) for value in self.area_km2)


@dataclass(frozen=True, slots=True)
class CausalSpatialHistory:
    """Coordinates already restricted by the caller to causally visible history.

    The replay layer must construct this only after applying frozen origin-time,
    availability-time, magnitude, completeness, and study-area rules.
    """

    x_km: FloatArray
    y_km: FloatArray

    def __post_init__(self) -> None:
        x = _readonly_float_vector("causal history x_km", self.x_km, allow_empty=True)
        y = _readonly_float_vector("causal history y_km", self.y_km, allow_empty=True)
        if x.shape != y.shape:
            raise ValueError("causal history coordinates must have one common shape")
        object.__setattr__(self, "x_km", x)
        object.__setattr__(self, "y_km", y)

    @property
    def event_count(self) -> int:
        return int(self.x_km.size)


def _readonly_int64_vector(name: str, values: object, *, allow_empty: bool) -> NDArray[np.int64]:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if not allow_empty and raw.size == 0:
        raise ValueError(f"{name} must not be empty")
    if raw.dtype.kind not in {"i", "u"}:
        raise TypeError(f"{name} must contain integer epoch microseconds")
    if raw.dtype.kind == "u" and raw.size and np.any(raw > np.iinfo(np.int64).max):
        raise ValueError(f"{name} exceeds the int64 epoch range")
    result = np.array(raw, dtype=np.int64, copy=True, order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class CausalRecent30History:
    """Recent events with explicit, separately supplied issue and data cutoffs.

    The frozen main interval is ``(T-30d, T-24h]`` by origin time, while every
    row's availability time must also be no later than ``T-24h``.  This avoids
    the old single-cutoff ambiguity that could either leak 24 hours or shift the
    recent window back by one day.
    """

    x_km: FloatArray
    y_km: FloatArray
    origin_time_us: NDArray[np.int64]
    available_at_us: NDArray[np.int64]
    issue_time_us: int
    data_cutoff_us: int

    def __post_init__(self) -> None:
        if isinstance(self.issue_time_us, bool) or not isinstance(self.issue_time_us, int):
            raise TypeError("issue_time_us must be integer epoch microseconds")
        if isinstance(self.data_cutoff_us, bool) or not isinstance(self.data_cutoff_us, int):
            raise TypeError("data_cutoff_us must be integer epoch microseconds")
        if self.data_cutoff_us != self.issue_time_us - _MAIN_CATALOG_DELAY_US:
            raise ValueError("main S1 recent data cutoff must equal issue time minus 24 hours")
        x = _readonly_float_vector("recent history x_km", self.x_km, allow_empty=True)
        y = _readonly_float_vector("recent history y_km", self.y_km, allow_empty=True)
        origin = _readonly_int64_vector(
            "recent history origin_time_us",
            self.origin_time_us,
            allow_empty=True,
        )
        available = _readonly_int64_vector(
            "recent history available_at_us",
            self.available_at_us,
            allow_empty=True,
        )
        if not (x.shape == y.shape == origin.shape == available.shape):
            raise ValueError("recent history coordinates and times must have one common shape")
        lower = self.issue_time_us - _RECENT_WINDOW_US
        if origin.size and (np.any(origin <= lower) or np.any(origin > self.data_cutoff_us)):
            raise ValueError("recent origins must lie in the frozen (T-30d, T-24h] interval")
        if available.size and np.any(available > self.data_cutoff_us):
            raise ValueError("recent events must be available by the frozen T-24h cutoff")
        object.__setattr__(self, "x_km", x)
        object.__setattr__(self, "y_km", y)
        object.__setattr__(self, "origin_time_us", origin)
        object.__setattr__(self, "available_at_us", available)

    @property
    def event_count(self) -> int:
        return int(self.x_km.size)

    def as_spatial_history(self) -> CausalSpatialHistory:
        return CausalSpatialHistory(self.x_km, self.y_km)


@dataclass(frozen=True, slots=True)
class EarlierInnerBoundary:
    """Proof that inner labels end at least 30 days before outer evaluation."""

    latest_inner_target_end_us: int
    outer_evaluation_start_us: int

    def __post_init__(self) -> None:
        for name, value in (
            ("latest_inner_target_end_us", self.latest_inner_target_end_us),
            ("outer_evaluation_start_us", self.outer_evaluation_start_us),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be integer epoch microseconds")
        separation_us = self.outer_evaluation_start_us - self.latest_inner_target_end_us
        if separation_us < _PARAMETER_SELECTION_EMBARGO_US:
            raise ValueError("inner targets must end at least 30 days before outer evaluation")


@dataclass(frozen=True, slots=True)
class LocationSurface:
    """One normalized relative spatial mass surface, never an absolute probability."""

    model_id: str
    cell_relative_mass: FloatArray
    source_event_count: int
    bandwidth_km: float | None = None
    alpha: float | None = None
    recent_fallback_to_long: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        if (
            isinstance(self.source_event_count, bool)
            or not isinstance(self.source_event_count, int)
            or self.source_event_count < 0
        ):
            raise ValueError("source_event_count must be a non-negative integer")
        mass = _readonly_relative_mass(self.cell_relative_mass)
        if self.bandwidth_km is not None and (
            not math.isfinite(self.bandwidth_km) or self.bandwidth_km <= 0.0
        ):
            raise ValueError("bandwidth_km must be finite and positive")
        if self.alpha is not None and (
            not math.isfinite(self.alpha) or not 0.0 <= self.alpha <= 1.0
        ):
            raise ValueError("alpha must be finite and within [0, 1]")
        object.__setattr__(self, "model_id", self.model_id.strip())
        object.__setattr__(self, "cell_relative_mass", mass)


@dataclass(frozen=True, slots=True)
class KDEBandwidthCandidateAudit:
    bandwidth_km: float
    inner_fold_scores: tuple[float, ...]
    mean_score: float
    paired_standard_error: float
    eligible: bool


@dataclass(frozen=True, slots=True)
class KDEBandwidthSelection:
    best_mean_bandwidth_km: float
    selected_bandwidth_km: float
    candidates: tuple[KDEBandwidthCandidateAudit, ...]
    boundary: EarlierInnerBoundary


def fixed_origin_region_indices(x_km: object, y_km: object) -> NDArray[np.int64]:
    """Assign points to origin-fixed 500 km squares using high-side floor rules."""

    x = _readonly_float_vector("x_km", x_km, allow_empty=True)
    y = _readonly_float_vector("y_km", y_km, allow_empty=True)
    if x.shape != y.shape:
        raise ValueError("x_km and y_km must have one common shape")
    scaled = np.column_stack((y / FROZEN_REGIONAL_SIZE_KM, x / FROZEN_REGIONAL_SIZE_KM))
    floored = np.floor(scaled)
    int64_info = np.iinfo(np.int64)
    if floored.size and (np.any(floored < int64_info.min) or np.any(floored > int64_info.max)):
        raise ValueError("coordinates exceed the fixed-region integer index range")
    result = np.asarray(floored, dtype=np.int64)
    result.setflags(write=False)
    return result


def l0_uniform_relative_mass(grid: FrozenSpatialGrid) -> LocationSurface:
    """L0: distribute relative mass exactly in proportion to clipped area."""

    if not isinstance(grid, FrozenSpatialGrid):
        raise TypeError("grid must be a FrozenSpatialGrid")
    return LocationSurface(
        model_id="L0_UNIFORM",
        cell_relative_mass=grid.area_km2 / grid.total_area_km2,
        source_event_count=0,
    )


def _region_keys(indices: NDArray[np.int64]) -> tuple[tuple[int, int], ...]:
    return tuple((int(row), int(column)) for row, column in indices)


def l1_regional_constant_relative_mass(
    causal_history: CausalSpatialHistory,
    grid: FrozenSpatialGrid,
    *,
    exposure_years: float,
    tau_years: float,
) -> LocationSurface:
    """L1: origin-fixed 500 km regional rates with Gamma--Poisson shrinkage."""

    if not isinstance(causal_history, CausalSpatialHistory):
        raise TypeError("causal_history must be a CausalSpatialHistory")
    if not isinstance(grid, FrozenSpatialGrid):
        raise TypeError("grid must be a FrozenSpatialGrid")
    if causal_history.event_count == 0:
        raise ValueError("L1 requires at least one causally visible training event")
    duration = float(exposure_years)
    tau = float(tau_years)
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("exposure_years must be finite and positive")
    if tau not in FROZEN_REGIONAL_TAU_YEARS:
        raise ValueError("tau_years must be one of 1, 5, or 10")

    grid_keys = _region_keys(fixed_origin_region_indices(grid.x_km, grid.y_km))
    event_keys = _region_keys(fixed_origin_region_indices(causal_history.x_km, causal_history.y_km))
    grid_regions = set(grid_keys)
    if set(event_keys).difference(grid_regions):
        raise ValueError("causal training history contains an event outside grid regions")

    area_by_region: dict[tuple[int, int], float] = {}
    for key, area in zip(grid_keys, grid.area_km2, strict=True):
        area_by_region[key] = math.fsum((area_by_region.get(key, 0.0), float(area)))
    count_by_region = dict.fromkeys(area_by_region, 0)
    for key in event_keys:
        count_by_region[key] += 1

    national_rate = causal_history.event_count / (grid.total_area_km2 * duration)
    rate_by_region = {
        key: (count_by_region[key] + tau * area * national_rate) / ((duration + tau) * area)
        for key, area in area_by_region.items()
    }
    unnormalized = np.asarray(
        [
            rate_by_region[key] * float(area)
            for key, area in zip(grid_keys, grid.area_km2, strict=True)
        ],
        dtype=np.float64,
    )
    total = math.fsum(float(value) for value in unnormalized)
    if not math.isfinite(total) or total <= 0.0:
        raise FloatingPointError("L1 regional rates produced no finite positive national mass")
    return LocationSurface(
        model_id="L1_REGIONAL_CONSTANT",
        cell_relative_mass=unnormalized / total,
        source_event_count=causal_history.event_count,
    )


def _validate_frozen_bandwidth(bandwidth_km: float) -> float:
    bandwidth = float(bandwidth_km)
    if bandwidth not in FROZEN_KDE_BANDWIDTHS_KM:
        raise ValueError("bandwidth_km must be one of 75, 100, 150, 200, or 300")
    return bandwidth


def l2_gaussian_kde_relative_mass(
    causal_history: CausalSpatialHistory,
    grid: FrozenSpatialGrid,
    *,
    bandwidth_km: float,
    model_id: str = "L2_KDE_CAUSAL",
) -> LocationSurface:
    """L2: equal-event Gaussian KDE, area-integrated over the full frozen grid."""

    if not isinstance(causal_history, CausalSpatialHistory):
        raise TypeError("causal_history must be a CausalSpatialHistory")
    if not isinstance(grid, FrozenSpatialGrid):
        raise TypeError("grid must be a FrozenSpatialGrid")
    if causal_history.event_count == 0:
        raise ValueError("L2 requires at least one causally visible training event")
    bandwidth = _validate_frozen_bandwidth(bandwidth_km)
    mixture = GaussianMixtureFamily(causal_history.x_km, causal_history.y_km)
    raw_density = mixture.raw_densities(
        grid.x_km,
        grid.y_km,
        bandwidths_km=(bandwidth,),
    )[bandwidth]
    raw_mass = np.asarray(raw_density * grid.area_km2, dtype=np.float64)
    normalization_mass = math.fsum(float(value) for value in raw_mass)
    if not math.isfinite(normalization_mass) or normalization_mass <= 0.0:
        raise FloatingPointError("L2 KDE produced no finite positive mass on the frozen grid")
    return LocationSurface(
        model_id=model_id,
        cell_relative_mass=raw_mass / normalization_mass,
        source_event_count=causal_history.event_count,
        bandwidth_km=bandwidth,
    )


def l3_b0_r30_relative_mass(
    causal_long_history: CausalSpatialHistory,
    causal_recent30_history: CausalRecent30History,
    grid: FrozenSpatialGrid,
    *,
    bandwidth_km: float,
    alpha: float,
) -> LocationSurface:
    """L3: mix long and recent-30-day KDE masses at one common bandwidth."""

    if not isinstance(causal_recent30_history, CausalRecent30History):
        raise TypeError("causal_recent30_history must be a CausalRecent30History")
    weight = float(alpha)
    if weight not in FROZEN_R30_ALPHA_CANDIDATES:
        raise ValueError("alpha must be one of 0, 0.25, 0.5, or 0.75")
    long_surface = l2_gaussian_kde_relative_mass(
        causal_long_history,
        grid,
        bandwidth_km=bandwidth_km,
        model_id="L2_KDE_CAUSAL",
    )
    if causal_recent30_history.event_count == 0 or weight == 0.0:
        mixed_mass = long_surface.cell_relative_mass
        fallback = causal_recent30_history.event_count == 0
    else:
        recent_surface = l2_gaussian_kde_relative_mass(
            causal_recent30_history.as_spatial_history(),
            grid,
            bandwidth_km=bandwidth_km,
            model_id="R30_COMPONENT",
        )
        mixed_mass = np.asarray(
            (1.0 - weight) * long_surface.cell_relative_mass
            + weight * recent_surface.cell_relative_mass,
            dtype=np.float64,
        )
        fallback = False
    return LocationSurface(
        model_id="L3_B0_R30_CAUSAL",
        cell_relative_mass=mixed_mass,
        source_event_count=causal_long_history.event_count,
        bandwidth_km=long_surface.bandwidth_km,
        alpha=weight,
        recent_fallback_to_long=fallback,
    )


def _normalized_scalar_scores(
    score_by_candidate: Mapping[float, float],
    *,
    candidates: tuple[float, ...],
) -> Mapping[float, float]:
    normalized: dict[float, float] = {}
    for raw_candidate, raw_score in score_by_candidate.items():
        candidate = float(raw_candidate)
        score = float(raw_score)
        if candidate in normalized:
            raise ValueError("candidate score keys must be unique")
        if not math.isfinite(score):
            raise ValueError("inner scores must be finite")
        normalized[candidate] = score
    if set(normalized) != set(candidates):
        raise ValueError("inner scores must cover exactly the frozen candidate set")
    return MappingProxyType(normalized)


def select_regional_tau(
    inner_mean_log_score_by_tau: Mapping[float, float],
    *,
    boundary: EarlierInnerBoundary,
) -> float:
    """Select L1 shrinkage from explicitly earlier inner scores; ties use larger tau."""

    if not isinstance(boundary, EarlierInnerBoundary):
        raise TypeError("boundary must prove that inner targets are earlier than the outer fold")
    scores = _normalized_scalar_scores(
        inner_mean_log_score_by_tau,
        candidates=FROZEN_REGIONAL_TAU_YEARS,
    )
    maximum = max(scores.values())
    return max(
        candidate
        for candidate, score in scores.items()
        if maximum - score <= _PARAMETER_TIE_TOLERANCE
    )


def select_recent_alpha(
    inner_mean_log_score_by_alpha: Mapping[float, float],
    *,
    inner_target_count: int,
    boundary: EarlierInnerBoundary,
) -> float:
    """Select L3 alpha from earlier targets, with the frozen <10 fallback and tie rule."""

    if not isinstance(boundary, EarlierInnerBoundary):
        raise TypeError("boundary must prove that inner targets are earlier than the outer fold")
    if (
        isinstance(inner_target_count, bool)
        or not isinstance(inner_target_count, int)
        or inner_target_count < 0
    ):
        raise ValueError("inner_target_count must be a non-negative integer")
    scores = _normalized_scalar_scores(
        inner_mean_log_score_by_alpha,
        candidates=FROZEN_R30_ALPHA_CANDIDATES,
    )
    if inner_target_count < INNER_TARGET_MINIMUM:
        return 0.0
    maximum = max(scores.values())
    return min(
        candidate
        for candidate, score in scores.items()
        if maximum - score <= _PARAMETER_TIE_TOLERANCE
    )


def _normalized_fold_scores(
    inner_fold_scores_by_bandwidth: Mapping[float, Sequence[float]],
) -> Mapping[float, FloatArray]:
    normalized: dict[float, FloatArray] = {}
    fold_count: int | None = None
    for raw_bandwidth, raw_scores in inner_fold_scores_by_bandwidth.items():
        bandwidth = float(raw_bandwidth)
        if bandwidth in normalized:
            raise ValueError("bandwidth score keys must be unique")
        scores = _readonly_float_vector("inner fold scores", raw_scores)
        if scores.size < 2:
            raise ValueError("one-SE selection requires at least two earlier inner folds")
        if fold_count is None:
            fold_count = int(scores.size)
        elif scores.size != fold_count:
            raise ValueError("all bandwidths must have the same earlier inner folds")
        normalized[bandwidth] = scores
    if set(normalized) != set(FROZEN_KDE_BANDWIDTHS_KM):
        raise ValueError("inner scores must cover exactly the five frozen bandwidths")
    return MappingProxyType(normalized)


def select_kde_bandwidth_one_se(
    inner_fold_scores_by_bandwidth: Mapping[float, Sequence[float]],
    *,
    boundary: EarlierInnerBoundary,
) -> KDEBandwidthSelection:
    """Apply the existing paired one-SE rule to explicitly earlier time folds.

    This generalizes the validated four-fold math in
    :mod:`seismoflux.background.poisson` to any common count of at least two
    earlier inner folds.  Candidate-minus-best paired fold differences define
    each candidate's standard error; the largest eligible bandwidth is chosen.
    """

    if not isinstance(boundary, EarlierInnerBoundary):
        raise TypeError("boundary must prove that inner targets are earlier than the outer fold")
    scores = _normalized_fold_scores(inner_fold_scores_by_bandwidth)
    means = {
        bandwidth: float(np.mean(values, dtype=np.float64)) for bandwidth, values in scores.items()
    }
    best_mean = max(means.values())
    best_bandwidth = max(
        bandwidth
        for bandwidth, mean_score in means.items()
        if math.isclose(
            mean_score,
            best_mean,
            rel_tol=0.0,
            abs_tol=_BEST_MEAN_TIE_TOLERANCE,
        )
    )
    best_scores = scores[best_bandwidth]
    fold_count = int(best_scores.size)
    audits: list[KDEBandwidthCandidateAudit] = []
    for bandwidth in FROZEN_KDE_BANDWIDTHS_KM:
        candidate_scores = scores[bandwidth]
        differences = candidate_scores - best_scores
        paired_se = float(np.std(differences, ddof=1) / math.sqrt(fold_count))
        audits.append(
            KDEBandwidthCandidateAudit(
                bandwidth_km=bandwidth,
                inner_fold_scores=cast(tuple[float, ...], tuple(candidate_scores)),
                mean_score=means[bandwidth],
                paired_standard_error=paired_se,
                eligible=means[bandwidth] >= best_mean - paired_se,
            )
        )
    selected = max(item.bandwidth_km for item in audits if item.eligible)
    return KDEBandwidthSelection(
        best_mean_bandwidth_km=best_bandwidth,
        selected_bandwidth_km=selected,
        candidates=tuple(audits),
        boundary=boundary,
    )


__all__ = [
    "FROZEN_KDE_BANDWIDTHS_KM",
    "FROZEN_R30_ALPHA_CANDIDATES",
    "FROZEN_REGIONAL_TAU_YEARS",
    "CausalRecent30History",
    "CausalSpatialHistory",
    "EarlierInnerBoundary",
    "FrozenSpatialGrid",
    "KDEBandwidthCandidateAudit",
    "KDEBandwidthSelection",
    "LocationSurface",
    "fixed_origin_region_indices",
    "l0_uniform_relative_mass",
    "l1_regional_constant_relative_mass",
    "l2_gaussian_kde_relative_mass",
    "l3_b0_r30_relative_mass",
    "select_kde_bandwidth_one_se",
    "select_recent_alpha",
    "select_regional_tau",
]
