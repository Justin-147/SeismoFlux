"""Score-blind prediction core for the frozen four-fold S1-C0 screen.

The module has a deliberately sharp two-layer boundary:

* functions whose names and types contain ``inner`` may read the three strictly
  earlier I1--I3 label blocks and return aggregate parameter-selection evidence;
* development prediction functions accept only the resulting frozen selection
  and causal history ending at ``issue - 24 hours``.  Their returned objects and
  NPZ-ready arrays contain no outer-fold observations or scores.

This module performs no file I/O and starts no worker processes.  The runner is
responsible for serialising each fold, sealing all four files, and only then
authorising the separate outer scoring phase.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from functools import partial
from types import MappingProxyType
from typing import Any, Final, Literal, cast

import numpy as np
from numpy.typing import NDArray

from seismoflux.multitask_s1.development_contract import (
    DEVELOPMENT_FOLD_IDS,
    HORIZONS_DAYS,
)
from seismoflux.multitask_s1.location import (
    FROZEN_KDE_BANDWIDTHS_KM,
    FROZEN_R30_ALPHA_CANDIDATES,
    FROZEN_REGIONAL_TAU_YEARS,
    CausalRecent30History,
    EarlierInnerBoundary,
    FrozenSpatialGrid,
    KDEBandwidthSelection,
    LocationSurface,
    l0_uniform_relative_mass,
    l1_regional_constant_relative_mass,
    l2_gaussian_kde_relative_mass,
    select_kde_bandwidth_one_se,
    select_recent_alpha,
    select_regional_tau,
)
from seismoflux.multitask_s1.runner_inputs import (
    CATALOG_HISTORY_START_UTC,
    EXPECTED_25KM_CELL_COUNT,
    CatalogEventTable,
    CausalMagnitudeHistory,
    InnerExposure,
    S1RunnerInputs,
    causal_catalog_histories,
)
from seismoflux.multitask_s1.time_magnitude import (
    NB2DispersionQualification,
    TruncatedGRMagnitudeModel,
    fit_expanding_poisson,
    fit_m0_gr_global,
    fit_m3_gr_long_m5,
    fit_nb2_dispersion,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
TimeBand = Literal["m5_6", "m6_plus", "m5_plus_1970_for_joint"]
LocateLonLat = Callable[[float, float], int | None]

TIME_BANDS: Final[tuple[TimeBand, ...]] = (
    "m5_6",
    "m6_plus",
    "m5_plus_1970_for_joint",
)
LOCATION_MODEL_IDS: Final[tuple[str, ...]] = (
    "L0_UNIFORM",
    "L1_REGIONAL_CONSTANT",
    "L2_KDE_CAUSAL",
    "L2_KDE75_LEGACY",
    "L3_B0_R30_CAUSAL",
)
MAGNITUDE_MODEL_IDS: Final[tuple[str, ...]] = ("M0_GR_GLOBAL", "M3_GR_LONG_M5")
PREDICTION_ARRAY_SCHEMA_VERSION: Final = 1
NB2_STATUS_CODES: Final[tuple[str, ...]] = (
    "not_evaluable",
    "poisson_limit",
    "evaluable",
)
NB2_REASON_CODES: Final[tuple[str, ...]] = (
    "no_non_overlapping_history_blocks",
    "fewer_than_two_non_overlapping_history_blocks",
    "positive_count_with_zero_poisson_expectation",
    "sample_variance_not_greater_than_sample_mean",
    "nonfinite_method_of_moments_start",
    "nonfinite_nb2_score_during_bracketing",
    "finite_dispersion_mle_not_bracketed",
    "nb2_maximum_likelihood_did_not_converge",
    "nb2_observed_information_not_positive_finite",
    "finite_mle_and_positive_observed_information",
    "shared_M5_6_k_under_frozen_M6_plus_7d_rule",
    "frozen_M6_plus_7d_poisson_fallback_no_independent_k_fit",
)

_DAY_US: Final = 86_400_000_000
_CATALOG_DELAY_US: Final = _DAY_US
_RECENT_DAYS: Final = 30
_DAYS_PER_YEAR: Final = 365.2425
_UTC_EPOCH: Final = datetime(1970, 1, 1, tzinfo=UTC)
_INNER_BLOCK_IDS: Final = ("I1", "I2", "I3")
_NB2_STATUS_CODE: Final[Mapping[str, int]] = MappingProxyType(
    {value: index for index, value in enumerate(NB2_STATUS_CODES)}
)
_NB2_REASON_CODE: Final[Mapping[str, int]] = MappingProxyType(
    {value: index for index, value in enumerate(NB2_REASON_CODES)}
)
_SHANGHAI: Final = timezone(timedelta(hours=8))
_FOLD_YEAR_BOUNDS: Final[Mapping[str, tuple[int, int]]] = MappingProxyType(
    {
        "C_DEV_2000_2004": (2000, 2005),
        "C_DEV_2005_2009": (2005, 2010),
        "C_DEV_2010_2014": (2010, 2015),
        "C_DEV_2015_2019": (2015, 2020),
    }
)
_EXPECTED_WEEKLY_COUNT: Final[Mapping[str, int]] = MappingProxyType(
    {
        "C_DEV_2000_2004": 261,
        "C_DEV_2005_2009": 261,
        "C_DEV_2010_2014": 260,
        "C_DEV_2015_2019": 261,
    }
)
_EXPECTED_PRIMARY_COUNT: Final = 99


class DevelopmentPredictionError(ValueError):
    """Raised when a score-blind S1-C0 prediction boundary is violated."""


def _utc(value: datetime, *, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise DevelopmentPredictionError(f"{label} must be timezone aware")
    return value.astimezone(UTC)


def _epoch_us(value: datetime) -> int:
    delta = _utc(value, label="timestamp") - _UTC_EPOCH
    return (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds


def _readonly(values: object, *, dtype: np.dtype[Any]) -> NDArray[Any]:
    result = np.array(values, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


def _require_development_fold(fold_id: str) -> str:
    if fold_id not in DEVELOPMENT_FOLD_IDS:
        raise DevelopmentPredictionError("S1-C0 accepts only the four frozen development folds")
    return fold_id


def _require_horizon(horizon_days: int) -> int:
    if isinstance(horizon_days, bool) or horizon_days not in HORIZONS_DAYS:
        raise DevelopmentPredictionError("horizon must be one of 7, 30, 90, 180, or 365 days")
    return int(horizon_days)


@dataclass(frozen=True, slots=True)
class StrictlyEarlierInnerLocationWindow:
    """M4+ labels for one explicitly earlier primary inner exposure only."""

    fold_id: str
    block_id: str
    horizon_days: int
    issue_time_utc: datetime
    interval_end_utc: datetime
    event_ids: tuple[str, ...]
    event_cell_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_development_fold(self.fold_id)
        _require_horizon(self.horizon_days)
        if self.block_id not in _INNER_BLOCK_IDS:
            raise DevelopmentPredictionError("inner block must be I1, I2, or I3")
        issue = _utc(self.issue_time_utc, label="inner issue")
        interval_end = _utc(self.interval_end_utc, label="inner interval end")
        if interval_end != issue + timedelta(days=self.horizon_days):
            raise DevelopmentPredictionError("inner interval must be exactly (issue, issue+h]")
        identifiers = tuple(self.event_ids)
        indices = tuple(self.event_cell_indices)
        if len(identifiers) != len(indices) or len(set(identifiers)) != len(identifiers):
            raise DevelopmentPredictionError("inner location event IDs must be unique and aligned")
        if any(not value for value in identifiers) or any(index < 0 for index in indices):
            raise DevelopmentPredictionError("inner location events must have valid IDs and cells")
        object.__setattr__(self, "issue_time_utc", issue)
        object.__setattr__(self, "interval_end_utc", interval_end)
        object.__setattr__(self, "event_ids", identifiers)
        object.__setattr__(self, "event_cell_indices", indices)


@dataclass(frozen=True, slots=True)
class StrictlyEarlierInnerLocationBlock:
    """All primary windows from one earlier block at one fixed horizon."""

    fold_id: str
    block_id: str
    horizon_days: int
    block_end_utc: datetime
    windows: tuple[StrictlyEarlierInnerLocationWindow, ...]

    def __post_init__(self) -> None:
        _require_development_fold(self.fold_id)
        _require_horizon(self.horizon_days)
        end = _utc(self.block_end_utc, label="inner block end")
        windows = tuple(self.windows)
        if self.block_id not in _INNER_BLOCK_IDS or not windows:
            raise DevelopmentPredictionError("inner block must be a non-empty I1, I2, or I3")
        if any(
            item.fold_id != self.fold_id
            or item.block_id != self.block_id
            or item.horizon_days != self.horizon_days
            or item.interval_end_utc > end
            for item in windows
        ):
            raise DevelopmentPredictionError("inner windows do not belong to one mature block")
        issues = tuple(item.issue_time_utc for item in windows)
        if issues != tuple(sorted(issues)) or len(set(issues)) != len(issues):
            raise DevelopmentPredictionError("inner primary issues must be unique and ordered")
        identifiers = tuple(event_id for item in windows for event_id in item.event_ids)
        if len(set(identifiers)) != len(identifiers):
            raise DevelopmentPredictionError(
                "one event_id appeared in multiple primary inner windows at the same horizon"
            )
        object.__setattr__(self, "block_end_utc", end)
        object.__setattr__(self, "windows", windows)

    @property
    def event_count(self) -> int:
        return sum(len(item.event_ids) for item in self.windows)


def build_strictly_earlier_inner_m4_location_block(
    catalog: CatalogEventTable,
    inner_exposure: InnerExposure,
    *,
    block_end_utc: datetime,
    locate_lonlat: LocateLonLat,
) -> StrictlyEarlierInnerLocationBlock:
    """Build one mature ``(issue, issue+h]`` inner M4+ location block.

    The second argument must be :class:`InnerExposure`; an ``OuterIssueRow``
    cannot be passed accidentally.  The block-end availability check is an
    additional fail-closed condition and does not change the present catalog,
    whose historical ``available_at`` proxy equals origin time.
    """

    if not isinstance(catalog, CatalogEventTable):
        raise TypeError("catalog must be a CatalogEventTable")
    if not isinstance(inner_exposure, InnerExposure):
        raise TypeError("inner_exposure must be an InnerExposure, never an outer issue row")
    _require_development_fold(inner_exposure.fold_id)
    horizon = _require_horizon(inner_exposure.horizon_days)
    if inner_exposure.block_id not in _INNER_BLOCK_IDS:
        raise DevelopmentPredictionError("inner exposure block must be I1, I2, or I3")
    block_end = _utc(block_end_utc, label="inner block end")
    block_end_us = _epoch_us(block_end)
    windows: list[StrictlyEarlierInnerLocationWindow] = []
    for raw_issue in inner_exposure.issue_times_utc:
        issue = _utc(raw_issue, label="inner issue")
        interval_end = issue + timedelta(days=horizon)
        if interval_end > block_end:
            raise DevelopmentPredictionError("inner exposure is not mature by its block end")
        issue_us = _epoch_us(issue)
        interval_end_us = _epoch_us(interval_end)
        selected = np.flatnonzero(
            catalog.inside_study_area
            & (catalog.magnitude >= 4.0)
            & (catalog.origin_time_us > issue_us)
            & (catalog.origin_time_us <= interval_end_us)
            & (catalog.available_at_us <= block_end_us)
        )
        identifiers: list[str] = []
        cell_indices: list[int] = []
        for raw_index in selected:
            index = int(raw_index)
            cell = locate_lonlat(
                float(catalog.longitude[index]),
                float(catalog.latitude[index]),
            )
            if cell is None:
                raise DevelopmentPredictionError(
                    "an inside-study-area inner M4+ event did not map to the frozen grid"
                )
            identifiers.append(catalog.event_ids[index])
            cell_indices.append(int(cell))
        windows.append(
            StrictlyEarlierInnerLocationWindow(
                fold_id=inner_exposure.fold_id,
                block_id=inner_exposure.block_id,
                horizon_days=horizon,
                issue_time_utc=issue,
                interval_end_utc=interval_end,
                event_ids=tuple(identifiers),
                event_cell_indices=tuple(cell_indices),
            )
        )
    return StrictlyEarlierInnerLocationBlock(
        fold_id=inner_exposure.fold_id,
        block_id=inner_exposure.block_id,
        horizon_days=horizon,
        block_end_utc=block_end,
        windows=tuple(windows),
    )


@dataclass(frozen=True, slots=True)
class CandidateInnerBlockLogDensities:
    parameter_value: float
    block_mean_log_density: tuple[float, float, float]

    def __post_init__(self) -> None:
        value = float(self.parameter_value)
        block_scores = tuple(float(item) for item in self.block_mean_log_density)
        if not math.isfinite(value) or len(block_scores) != 3:
            raise DevelopmentPredictionError("candidate evidence must contain exactly three blocks")
        if not all(math.isfinite(item) for item in block_scores):
            raise DevelopmentPredictionError("inner spatial log densities must be finite")
        object.__setattr__(self, "parameter_value", value)
        object.__setattr__(self, "block_mean_log_density", block_scores)


@dataclass(frozen=True, slots=True)
class LocationParameterSelection:
    """Aggregate inner evidence; never an outer score or target population."""

    fold_id: str
    horizon_days: int
    boundary: EarlierInnerBoundary
    inner_block_event_counts: tuple[int, int, int]
    regional_tau_years: float
    kde_bandwidth: KDEBandwidthSelection
    recent_alpha: float
    regional_candidates: tuple[CandidateInnerBlockLogDensities, ...]
    kde_candidates: tuple[CandidateInnerBlockLogDensities, ...]
    recent_candidates: tuple[CandidateInnerBlockLogDensities, ...]

    @property
    def selected_bandwidth_km(self) -> float:
        return self.kde_bandwidth.selected_bandwidth_km

    @property
    def inner_event_count(self) -> int:
        return sum(self.inner_block_event_counts)


@dataclass(frozen=True, slots=True)
class _InnerIssueContext:
    window: StrictlyEarlierInnerLocationWindow
    history: CausalMagnitudeHistory
    recent: CausalRecent30History


def _recent_m4_history(history: CausalMagnitudeHistory) -> CausalRecent30History:
    issue_us = _epoch_us(history.issue_time_utc)
    cutoff_us = _epoch_us(history.data_cutoff_utc)
    lower_us = issue_us - _RECENT_DAYS * _DAY_US
    selected = (history.origin_time_us > lower_us) & (history.origin_time_us <= cutoff_us)
    return CausalRecent30History(
        x_km=history.spatial.x_km[selected],
        y_km=history.spatial.y_km[selected],
        origin_time_us=history.origin_time_us[selected],
        available_at_us=history.available_at_us[selected],
        issue_time_us=issue_us,
        data_cutoff_us=cutoff_us,
    )


def _history_exposure_days(history: CausalMagnitudeHistory) -> float:
    elapsed = (history.data_cutoff_utc - CATALOG_HISTORY_START_UTC).total_seconds() / 86_400.0
    if not math.isfinite(elapsed) or elapsed <= 0.0:
        raise DevelopmentPredictionError("causal 1970+ history has no positive elapsed exposure")
    return elapsed


def _mean_context_log_density(
    contexts: Sequence[_InnerIssueContext],
    surface_for_context: Callable[[_InnerIssueContext], LocationSurface],
    grid: FrozenSpatialGrid,
) -> float:
    total = 0.0
    count = 0
    for context in contexts:
        surface = surface_for_context(context)
        for cell_index in context.window.event_cell_indices:
            if not 0 <= cell_index < grid.cell_count:
                raise DevelopmentPredictionError("inner target cell is outside the frozen grid")
            mass = float(surface.cell_relative_mass[cell_index])
            if mass <= 0.0:
                raise DevelopmentPredictionError(
                    "an inner target received zero numerical mass; parameter selection is undefined"
                )
            total += math.log(mass) - math.log(float(grid.area_km2[cell_index]))
            count += 1
    if count == 0:
        raise DevelopmentPredictionError(
            "each of I1, I2, and I3 needs at least one M4+ target for spatial selection"
        )
    result = total / count
    if not math.isfinite(result):
        raise DevelopmentPredictionError("inner block mean spatial log density is not finite")
    return result


def _regional_surface(
    context: _InnerIssueContext,
    *,
    grid: FrozenSpatialGrid,
    tau_years: float,
) -> LocationSurface:
    return l1_regional_constant_relative_mass(
        context.history.spatial,
        grid,
        exposure_years=_history_exposure_days(context.history) / _DAYS_PER_YEAR,
        tau_years=tau_years,
    )


def _target_masses(
    surface: LocationSurface,
    window: StrictlyEarlierInnerLocationWindow,
    grid: FrozenSpatialGrid,
    *,
    allow_numerical_zero: bool = False,
) -> tuple[float, ...]:
    """Read target-cell masses, optionally retaining Gaussian underflow zeros.

    A Gaussian has infinite support, but a sparse recent-only component can
    underflow to numerical zero at a distant target.  That zero is admissible
    only before L3 mixes the component with its strictly positive long-term
    background.  Stand-alone L1/L2 target masses and final mixtures must remain
    strictly positive.
    """

    result: list[float] = []
    for cell_index in window.event_cell_indices:
        if not 0 <= cell_index < grid.cell_count:
            raise DevelopmentPredictionError("inner target cell is outside the frozen grid")
        mass = float(surface.cell_relative_mass[cell_index])
        if not math.isfinite(mass) or mass < 0.0:
            raise DevelopmentPredictionError("an inner target received invalid numerical mass")
        if mass == 0.0 and not allow_numerical_zero:
            raise DevelopmentPredictionError(
                "an inner target received zero numerical mass; parameter selection is undefined"
            )
        result.append(mass)
    return tuple(result)


def _l3_from_cached_surfaces(
    long_surface: LocationSurface,
    recent_surface: LocationSurface | None,
    *,
    recent_event_count: int,
    alpha: float,
) -> LocationSurface:
    """Apply the frozen L3 mixture to already computed same-issue KDE surfaces."""

    weight = float(alpha)
    if weight not in FROZEN_R30_ALPHA_CANDIDATES:
        raise DevelopmentPredictionError("cached L3 alpha is outside the frozen candidates")
    if recent_event_count < 0:
        raise DevelopmentPredictionError("recent_event_count must be non-negative")
    if recent_event_count == 0 or weight == 0.0:
        mixed_mass = long_surface.cell_relative_mass
        fallback = recent_event_count == 0
    else:
        if recent_surface is None:
            raise DevelopmentPredictionError("non-empty cached L3 requires a recent surface")
        mixed_mass = np.asarray(
            (1.0 - weight) * long_surface.cell_relative_mass
            + weight * recent_surface.cell_relative_mass,
            dtype=np.float64,
        )
        fallback = False
    return LocationSurface(
        model_id="L3_B0_R30_CAUSAL",
        cell_relative_mass=mixed_mass,
        source_event_count=long_surface.source_event_count,
        bandwidth_km=long_surface.bandwidth_km,
        alpha=weight,
        recent_fallback_to_long=fallback,
    )


def _surface_with_model_id(surface: LocationSurface, *, model_id: str) -> LocationSurface:
    """Reuse one deterministic KDE mass under a preregistered reference model ID."""

    return LocationSurface(
        model_id=model_id,
        cell_relative_mass=surface.cell_relative_mass,
        source_event_count=surface.source_event_count,
        bandwidth_km=surface.bandwidth_km,
        alpha=surface.alpha,
        recent_fallback_to_long=surface.recent_fallback_to_long,
    )


def select_location_parameters_from_strictly_earlier_inner_blocks(
    catalog: CatalogEventTable,
    grid: FrozenSpatialGrid,
    inner_blocks: Sequence[StrictlyEarlierInnerLocationBlock],
    *,
    outer_start_utc: datetime,
) -> LocationParameterSelection:
    """Select L1/L2/L3 from exactly three earlier M4+ validation blocks."""

    blocks = tuple(inner_blocks)
    if len(blocks) != 3 or tuple(item.block_id for item in blocks) != _INNER_BLOCK_IDS:
        raise DevelopmentPredictionError("location selection requires ordered I1, I2, and I3")
    fold_id = _require_development_fold(blocks[0].fold_id)
    horizon = _require_horizon(blocks[0].horizon_days)
    if any(item.fold_id != fold_id or item.horizon_days != horizon for item in blocks):
        raise DevelopmentPredictionError("inner blocks must share one development fold and horizon")
    identifiers = tuple(
        event_id for block in blocks for window in block.windows for event_id in window.event_ids
    )
    if len(set(identifiers)) != len(identifiers):
        raise DevelopmentPredictionError(
            "one event_id appeared twice across same-horizon primary inner blocks"
        )
    boundary = EarlierInnerBoundary(
        latest_inner_target_end_us=_epoch_us(max(item.block_end_utc for item in blocks)),
        outer_evaluation_start_us=_epoch_us(_utc(outer_start_utc, label="outer start")),
    )
    contexts_by_block: list[tuple[_InnerIssueContext, ...]] = []
    for block in blocks:
        contexts: list[_InnerIssueContext] = []
        for window in block.windows:
            history = causal_catalog_histories(catalog, window.issue_time_utc)["m4_plus"]
            if history.event_count == 0:
                raise DevelopmentPredictionError("L1/L2 require visible M4+ history")
            contexts.append(
                _InnerIssueContext(
                    window=window,
                    history=history,
                    recent=_recent_m4_history(history),
                )
            )
        contexts_by_block.append(tuple(contexts))

    regional_evidence: list[CandidateInnerBlockLogDensities] = []
    for tau in FROZEN_REGIONAL_TAU_YEARS:
        block_scores = tuple(
            _mean_context_log_density(
                contexts,
                partial(_regional_surface, grid=grid, tau_years=tau),
                grid,
            )
            for contexts in contexts_by_block
        )
        regional_evidence.append(
            CandidateInnerBlockLogDensities(tau, cast(tuple[float, float, float], block_scores))
        )
    regional_tau = select_regional_tau(
        {
            item.parameter_value: math.fsum(item.block_mean_log_density) / 3.0
            for item in regional_evidence
        },
        boundary=boundary,
    )

    kde_evidence: list[CandidateInnerBlockLogDensities] = []
    target_mass_cache: dict[tuple[int, int, float], tuple[float, ...]] = {}
    for bandwidth in FROZEN_KDE_BANDWIDTHS_KM:
        candidate_block_scores: list[float] = []
        for block_index, block_contexts in enumerate(contexts_by_block):
            total = 0.0
            count = 0
            for context_index, context in enumerate(block_contexts):
                surface = l2_gaussian_kde_relative_mass(
                    context.history.spatial,
                    grid,
                    bandwidth_km=bandwidth,
                )
                target_masses = _target_masses(surface, context.window, grid)
                target_mass_cache[(block_index, context_index, bandwidth)] = target_masses
                total += math.fsum(math.log(value) for value in target_masses)
                total -= math.fsum(
                    math.log(float(grid.area_km2[cell_index]))
                    for cell_index in context.window.event_cell_indices
                )
                count += len(target_masses)
            if count == 0:
                raise DevelopmentPredictionError(
                    "each inner block needs an M4+ target for KDE selection"
                )
            candidate_block_scores.append(total / count)
        block_scores = cast(tuple[float, float, float], tuple(candidate_block_scores))
        kde_evidence.append(
            CandidateInnerBlockLogDensities(
                bandwidth,
                block_scores,
            )
        )
    kde_selection = select_kde_bandwidth_one_se(
        {item.parameter_value: item.block_mean_log_density for item in kde_evidence},
        boundary=boundary,
    )

    recent_scores_by_alpha: dict[float, list[float]] = {
        alpha: [] for alpha in FROZEN_R30_ALPHA_CANDIDATES
    }
    for block_index, block_contexts in enumerate(contexts_by_block):
        sum_by_alpha = {alpha: 0.0 for alpha in FROZEN_R30_ALPHA_CANDIDATES}
        count = 0
        for context_index, context in enumerate(block_contexts):
            long_masses = target_mass_cache[
                (block_index, context_index, kde_selection.selected_bandwidth_km)
            ]
            if context.recent.event_count == 0:
                recent_masses = long_masses
            else:
                recent_component = l2_gaussian_kde_relative_mass(
                    context.recent.as_spatial_history(),
                    grid,
                    bandwidth_km=kde_selection.selected_bandwidth_km,
                    model_id="R30_COMPONENT",
                )
                recent_masses = _target_masses(
                    recent_component,
                    context.window,
                    grid,
                    allow_numerical_zero=True,
                )
            for target_position, cell_index in enumerate(context.window.event_cell_indices):
                long_mass = long_masses[target_position]
                recent_mass = recent_masses[target_position]
                log_area = math.log(float(grid.area_km2[cell_index]))
                for alpha in FROZEN_R30_ALPHA_CANDIDATES:
                    mixed_mass = (
                        long_mass
                        if context.recent.event_count == 0 or alpha == 0.0
                        else (1.0 - alpha) * long_mass + alpha * recent_mass
                    )
                    if mixed_mass <= 0.0:
                        raise DevelopmentPredictionError(
                            "cached inner L3 mixture assigned zero target mass"
                        )
                    sum_by_alpha[alpha] += math.log(mixed_mass) - log_area
                count += 1
        if count == 0:
            raise DevelopmentPredictionError(
                "each inner block needs an M4+ target for recent-weight selection"
            )
        for alpha in FROZEN_R30_ALPHA_CANDIDATES:
            recent_scores_by_alpha[alpha].append(sum_by_alpha[alpha] / count)

    recent_evidence = tuple(
        CandidateInnerBlockLogDensities(
            alpha,
            cast(tuple[float, float, float], tuple(recent_scores_by_alpha[alpha])),
        )
        for alpha in FROZEN_R30_ALPHA_CANDIDATES
    )
    total_inner_events = sum(block.event_count for block in blocks)
    recent_alpha = select_recent_alpha(
        {
            item.parameter_value: math.fsum(item.block_mean_log_density) / 3.0
            for item in recent_evidence
        },
        inner_target_count=total_inner_events,
        boundary=boundary,
    )
    return LocationParameterSelection(
        fold_id=fold_id,
        horizon_days=horizon,
        boundary=boundary,
        inner_block_event_counts=cast(
            tuple[int, int, int], tuple(block.event_count for block in blocks)
        ),
        regional_tau_years=regional_tau,
        kde_bandwidth=kde_selection,
        recent_alpha=recent_alpha,
        regional_candidates=tuple(regional_evidence),
        kde_candidates=tuple(kde_evidence),
        recent_candidates=recent_evidence,
    )


@dataclass(frozen=True, slots=True)
class InnerTimeCountSeries:
    """Pooled non-overlapping inner counts and matching causal T0 means."""

    band: TimeBand
    counts: tuple[int, ...]
    poisson_expected_counts: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.band not in TIME_BANDS:
            raise DevelopmentPredictionError("unknown frozen time band")
        counts = tuple(self.counts)
        means = tuple(float(item) for item in self.poisson_expected_counts)
        if not counts or len(counts) != len(means):
            raise DevelopmentPredictionError("inner count series must be non-empty and aligned")
        if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in counts):
            raise DevelopmentPredictionError("inner counts must be non-negative integers")
        if any(not math.isfinite(item) or item < 0.0 for item in means):
            raise DevelopmentPredictionError("inner T0 means must be finite and non-negative")
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "poisson_expected_counts", means)


@dataclass(frozen=True, slots=True)
class BandNB2Qualification:
    band: TimeBand
    qualification: NB2DispersionQualification


def fit_t1_qualifications_from_inner_count_series(
    series: Sequence[InnerTimeCountSeries],
    *,
    horizon_days: int,
) -> tuple[BandNB2Qualification, ...]:
    """Fit T1 from pooled I1--I3 primary counts, including frozen M6+ 7d sharing."""

    horizon = _require_horizon(horizon_days)
    by_band = {item.band: item for item in series}
    if len(by_band) != len(series) or tuple(by_band) != TIME_BANDS:
        raise DevelopmentPredictionError("inner T1 series must cover the three bands in order")
    series_lengths = {len(item.counts) for item in series}
    if len(series_lengths) != 1:
        raise DevelopmentPredictionError(
            "inner T1 series must share one common non-overlapping exposure axis"
        )
    fitted: dict[TimeBand, NB2DispersionQualification] = {}
    for band in TIME_BANDS:
        if band == "m6_plus" and horizon == 7:
            continue
        item = by_band[band]
        fitted[band] = fit_nb2_dispersion(item.counts, item.poisson_expected_counts)
    if horizon == 7:
        source = fitted["m5_6"]
        m6_series = by_band["m6_plus"]
        m6_mean = math.fsum(m6_series.counts) / len(m6_series.counts)
        m6_variance = (
            None
            if len(m6_series.counts) < 2
            else math.fsum((item - m6_mean) ** 2 for item in m6_series.counts)
            / (len(m6_series.counts) - 1)
        )
        if source.status == "evaluable":
            fitted["m6_plus"] = NB2DispersionQualification(
                status="evaluable",
                reason="shared_M5_6_k_under_frozen_M6_plus_7d_rule",
                historical_block_count=len(m6_series.counts),
                sample_mean_count=m6_mean,
                sample_variance_count=m6_variance,
                dispersion_k=source.dispersion_k,
                observed_information_k=source.observed_information_k,
                standard_error_k=source.standard_error_k,
            )
        else:
            fitted["m6_plus"] = NB2DispersionQualification(
                status="poisson_limit",
                reason="frozen_M6_plus_7d_poisson_fallback_no_independent_k_fit",
                historical_block_count=len(m6_series.counts),
                sample_mean_count=m6_mean,
                sample_variance_count=m6_variance,
            )
    return tuple(BandNB2Qualification(band, fitted[band]) for band in TIME_BANDS)


def _band_mask(catalog: CatalogEventTable, band: TimeBand) -> NDArray[np.bool_]:
    if band == "m5_6":
        return (catalog.magnitude >= 5.0) & (catalog.magnitude < 6.0)
    if band == "m6_plus":
        return catalog.magnitude >= 6.0
    return catalog.magnitude >= 5.0


def build_strictly_earlier_inner_time_count_series(
    catalog: CatalogEventTable,
    inner_exposures: Sequence[InnerExposure],
    *,
    block_end_by_id: Mapping[str, datetime],
) -> tuple[InnerTimeCountSeries, ...]:
    """Build pooled non-overlapping I1--I3 count series with matching T0 means."""

    exposures = tuple(inner_exposures)
    if len(exposures) != 3 or tuple(item.block_id for item in exposures) != _INNER_BLOCK_IDS:
        raise DevelopmentPredictionError("inner time series requires ordered I1, I2, and I3")
    fold_id = _require_development_fold(exposures[0].fold_id)
    horizon = _require_horizon(exposures[0].horizon_days)
    if any(item.fold_id != fold_id or item.horizon_days != horizon for item in exposures):
        raise DevelopmentPredictionError("inner time exposures must share one fold and horizon")
    if set(block_end_by_id) != set(_INNER_BLOCK_IDS):
        raise DevelopmentPredictionError("inner block ends must cover exactly I1, I2, and I3")
    counts: dict[TimeBand, list[int]] = {band: [] for band in TIME_BANDS}
    means: dict[TimeBand, list[float]] = {band: [] for band in TIME_BANDS}
    seen: dict[TimeBand, set[str]] = {band: set() for band in TIME_BANDS}
    for exposure in exposures:
        block_end = _utc(block_end_by_id[exposure.block_id], label="inner block end")
        block_end_us = _epoch_us(block_end)
        for issue in exposure.issue_times_utc:
            issue_utc = _utc(issue, label="inner issue")
            interval_end = issue_utc + timedelta(days=horizon)
            if interval_end > block_end:
                raise DevelopmentPredictionError("inner time exposure is not mature by block end")
            issue_us = _epoch_us(issue_utc)
            interval_end_us = _epoch_us(interval_end)
            histories = causal_catalog_histories(catalog, issue_utc)
            for band in TIME_BANDS:
                selected = np.flatnonzero(
                    catalog.inside_study_area
                    & _band_mask(catalog, band)
                    & (catalog.origin_time_us > issue_us)
                    & (catalog.origin_time_us <= interval_end_us)
                    & (catalog.available_at_us <= block_end_us)
                )
                event_ids = tuple(catalog.event_ids[int(index)] for index in selected)
                if seen[band].intersection(event_ids):
                    raise DevelopmentPredictionError(
                        "one event_id appeared in multiple same-horizon primary inner count windows"
                    )
                seen[band].update(event_ids)
                history = histories[band]
                rate = fit_expanding_poisson(
                    history_event_count=history.event_count,
                    history_exposure_days=_history_exposure_days(history),
                )
                counts[band].append(len(event_ids))
                means[band].append(rate.expected_count(horizon))
    return tuple(
        InnerTimeCountSeries(band, tuple(counts[band]), tuple(means[band])) for band in TIME_BANDS
    )


@dataclass(frozen=True, slots=True)
class FoldHorizonParameterSelection:
    fold_id: str
    horizon_days: int
    location: LocationParameterSelection
    t1: tuple[BandNB2Qualification, ...]

    def __post_init__(self) -> None:
        _require_development_fold(self.fold_id)
        _require_horizon(self.horizon_days)
        if self.location.fold_id != self.fold_id or self.location.horizon_days != self.horizon_days:
            raise DevelopmentPredictionError("location selection changed fold or horizon")
        if tuple(item.band for item in self.t1) != TIME_BANDS:
            raise DevelopmentPredictionError("T1 qualification order changed")


def _contract_fold(inputs: S1RunnerInputs, fold_id: str) -> Mapping[str, Any]:
    raw_folds = inputs.contract.get("outer_folds")
    if not isinstance(raw_folds, Sequence) or isinstance(raw_folds, str | bytes):
        raise DevelopmentPredictionError("runner contract outer_folds is invalid")
    matches = [
        cast(Mapping[str, Any], item)
        for item in raw_folds
        if isinstance(item, Mapping) and str(item.get("id")) == fold_id
    ]
    if len(matches) != 1:
        raise DevelopmentPredictionError("development fold is absent or duplicated in contract")
    return matches[0]


def _contract_inner_block_ends(
    inputs: S1RunnerInputs, fold_id: str
) -> tuple[datetime, Mapping[str, datetime]]:
    fold = _contract_fold(inputs, fold_id)
    outer_start_raw = fold.get("outer_start")
    blocks_raw = fold.get("inner_blocks")
    if not isinstance(outer_start_raw, str) or not isinstance(blocks_raw, Sequence):
        raise DevelopmentPredictionError("development fold calendar is invalid")
    try:
        outer_start = datetime.fromisoformat(outer_start_raw).astimezone(UTC)
    except ValueError as exc:
        raise DevelopmentPredictionError("outer start is not a valid contract timestamp") from exc
    result: dict[str, datetime] = {}
    for raw_block in blocks_raw:
        if not isinstance(raw_block, Mapping):
            raise DevelopmentPredictionError("inner block contract row is invalid")
        block_id = str(raw_block.get("id"))
        end_raw = raw_block.get("end")
        if block_id not in _INNER_BLOCK_IDS or not isinstance(end_raw, str):
            raise DevelopmentPredictionError("inner block contract identity is invalid")
        try:
            result[block_id] = datetime.fromisoformat(end_raw).astimezone(UTC)
        except ValueError as exc:
            raise DevelopmentPredictionError("inner block end is invalid") from exc
    if tuple(result) != _INNER_BLOCK_IDS:
        raise DevelopmentPredictionError("contract must contain ordered I1, I2, and I3")
    return outer_start, MappingProxyType(result)


def select_fold_horizon_parameters(
    inputs: S1RunnerInputs,
    *,
    fold_id: str,
    horizon_days: int,
) -> FoldHorizonParameterSelection:
    """Run the permitted I1--I3 selection without constructing any outer label."""

    if not isinstance(inputs, S1RunnerInputs):
        raise TypeError("inputs must be verified S1RunnerInputs")
    fold = _require_development_fold(fold_id)
    horizon = _require_horizon(horizon_days)
    exposures = tuple(
        item
        for item in inputs.inner_exposures
        if item.fold_id == fold and item.horizon_days == horizon
    )
    if len(exposures) != 3 or tuple(item.block_id for item in exposures) != _INNER_BLOCK_IDS:
        raise DevelopmentPredictionError("verified inputs lack exactly three inner exposures")
    outer_start, block_ends = _contract_inner_block_ends(inputs, fold)
    blocks = tuple(
        build_strictly_earlier_inner_m4_location_block(
            inputs.catalog,
            exposure,
            block_end_utc=block_ends[exposure.block_id],
            locate_lonlat=inputs.spatial_domain.locator.locate_lonlat,
        )
        for exposure in exposures
    )
    location = select_location_parameters_from_strictly_earlier_inner_blocks(
        inputs.catalog,
        inputs.location_grid,
        blocks,
        outer_start_utc=outer_start,
    )
    time_series = build_strictly_earlier_inner_time_count_series(
        inputs.catalog,
        exposures,
        block_end_by_id=block_ends,
    )
    t1 = fit_t1_qualifications_from_inner_count_series(time_series, horizon_days=horizon)
    return FoldHorizonParameterSelection(fold, horizon, location, t1)


@dataclass(frozen=True, slots=True)
class BandCountForecast:
    band: TimeBand
    poisson_expected_count: float
    t1_qualification: NB2DispersionQualification


@dataclass(frozen=True, slots=True)
class PrimaryIssuePrediction:
    """One target-free primary issue forecast aligned across all components."""

    fold_id: str
    horizon_days: int
    issue_time_utc: datetime
    regional_tau_years: float
    location_surfaces: tuple[LocationSurface, ...]
    count_forecasts: tuple[BandCountForecast, ...]

    def __post_init__(self) -> None:
        _require_development_fold(self.fold_id)
        _require_horizon(self.horizon_days)
        issue = _utc(self.issue_time_utc, label="primary issue")
        tau = float(self.regional_tau_years)
        if tau not in FROZEN_REGIONAL_TAU_YEARS:
            raise DevelopmentPredictionError("regional tau left the frozen candidate set")
        if tuple(item.model_id for item in self.location_surfaces) != LOCATION_MODEL_IDS:
            raise DevelopmentPredictionError("location model alignment changed")
        if tuple(item.band for item in self.count_forecasts) != TIME_BANDS:
            raise DevelopmentPredictionError("time-band alignment changed")
        object.__setattr__(self, "issue_time_utc", issue)
        object.__setattr__(self, "regional_tau_years", tau)


@dataclass(frozen=True, slots=True)
class WeeklyMagnitudeSnapshot:
    """Causal M0/M3 models at one weekly issue; no event is assigned here."""

    fold_id: str
    issue_time_utc: datetime
    m0: TruncatedGRMagnitudeModel
    m3: TruncatedGRMagnitudeModel

    def __post_init__(self) -> None:
        _require_development_fold(self.fold_id)
        object.__setattr__(self, "issue_time_utc", _utc(self.issue_time_utc, label="weekly issue"))
        if self.m0.model_id != "M0_GR_GLOBAL" or self.m3.model_id != "M3_GR_LONG_M5":
            raise DevelopmentPredictionError("magnitude model alignment changed")


def build_primary_issue_prediction(
    catalog: CatalogEventTable,
    grid: FrozenSpatialGrid,
    selection: FoldHorizonParameterSelection,
    *,
    issue_time_utc: datetime,
) -> PrimaryIssuePrediction:
    """Build five location surfaces and three T0 means from T-minus-24h history."""

    issue = _utc(issue_time_utc, label="primary issue")
    histories = causal_catalog_histories(catalog, issue)
    m4 = histories["m4_plus"]
    if m4.event_count == 0:
        raise DevelopmentPredictionError("outer prediction has no causal M4+ training history")
    recent = _recent_m4_history(m4)
    location = selection.location
    selected_long = l2_gaussian_kde_relative_mass(
        m4.spatial,
        grid,
        bandwidth_km=location.selected_bandwidth_km,
        model_id="L2_KDE_CAUSAL",
    )
    legacy = (
        _surface_with_model_id(selected_long, model_id="L2_KDE75_LEGACY")
        if location.selected_bandwidth_km == 75.0
        else l2_gaussian_kde_relative_mass(
            m4.spatial,
            grid,
            bandwidth_km=75.0,
            model_id="L2_KDE75_LEGACY",
        )
    )
    recent_component = (
        l2_gaussian_kde_relative_mass(
            recent.as_spatial_history(),
            grid,
            bandwidth_km=location.selected_bandwidth_km,
            model_id="R30_COMPONENT",
        )
        if recent.event_count > 0 and location.recent_alpha > 0.0
        else None
    )
    cached_l3 = _l3_from_cached_surfaces(
        selected_long,
        recent_component,
        recent_event_count=recent.event_count,
        alpha=location.recent_alpha,
    )
    surfaces = (
        l0_uniform_relative_mass(grid),
        l1_regional_constant_relative_mass(
            m4.spatial,
            grid,
            exposure_years=_history_exposure_days(m4) / _DAYS_PER_YEAR,
            tau_years=location.regional_tau_years,
        ),
        selected_long,
        legacy,
        cached_l3,
    )
    qualifications = {item.band: item.qualification for item in selection.t1}
    count_forecasts: list[BandCountForecast] = []
    for band in TIME_BANDS:
        history = histories[band]
        rate = fit_expanding_poisson(
            history_event_count=history.event_count,
            history_exposure_days=_history_exposure_days(history),
        )
        count_forecasts.append(
            BandCountForecast(
                band=band,
                poisson_expected_count=rate.expected_count(selection.horizon_days),
                t1_qualification=qualifications[band],
            )
        )
    return PrimaryIssuePrediction(
        fold_id=selection.fold_id,
        horizon_days=selection.horizon_days,
        issue_time_utc=issue,
        regional_tau_years=location.regional_tau_years,
        location_surfaces=surfaces,
        count_forecasts=tuple(count_forecasts),
    )


def build_weekly_magnitude_snapshot(
    catalog: CatalogEventTable,
    *,
    fold_id: str,
    issue_time_utc: datetime,
) -> WeeklyMagnitudeSnapshot:
    """Fit causal M0 (1970+ M4+) and M3 (1900+ M5+) at one weekly issue."""

    fold = _require_development_fold(fold_id)
    issue = _utc(issue_time_utc, label="weekly issue")
    histories = causal_catalog_histories(catalog, issue)
    m0 = fit_m0_gr_global(histories["m4_plus"].magnitude)
    m3 = fit_m3_gr_long_m5(histories["m5_plus_1900_for_m3"].magnitude)
    return WeeklyMagnitudeSnapshot(fold, issue, m0, m3)


@dataclass(frozen=True, slots=True)
class DevelopmentFoldPrediction:
    """A complete, target-free fold payload ready for deterministic array conversion."""

    fold_id: str
    primary_predictions: tuple[PrimaryIssuePrediction, ...]
    magnitude_snapshots: tuple[WeeklyMagnitudeSnapshot, ...]

    def __post_init__(self) -> None:
        fold = _require_development_fold(self.fold_id)
        primary = tuple(self.primary_predictions)
        magnitude = tuple(self.magnitude_snapshots)
        if not primary or not magnitude:
            raise DevelopmentPredictionError("a fold prediction cannot be empty")
        if any(item.fold_id != fold for item in primary) or any(
            item.fold_id != fold for item in magnitude
        ):
            raise DevelopmentPredictionError("fold prediction contains a foreign fold")
        primary_keys = tuple((item.horizon_days, item.issue_time_utc) for item in primary)
        expected_keys = tuple(
            sorted(primary_keys, key=lambda item: (HORIZONS_DAYS.index(item[0]), item[1]))
        )
        if primary_keys != expected_keys or len(set(primary_keys)) != len(primary_keys):
            raise DevelopmentPredictionError("primary predictions must be unique and ordered")
        magnitude_issues = tuple(item.issue_time_utc for item in magnitude)
        if magnitude_issues != tuple(sorted(magnitude_issues)) or len(set(magnitude_issues)) != len(
            magnitude_issues
        ):
            raise DevelopmentPredictionError(
                "weekly magnitude snapshots must be unique and ordered"
            )
        if not {item.issue_time_utc for item in primary} <= set(magnitude_issues):
            raise DevelopmentPredictionError(
                "each primary issue must bind to its same-issue causal M0 snapshot"
            )


def build_development_fold_prediction(
    inputs: S1RunnerInputs,
    *,
    fold_id: str,
    selections: Sequence[FoldHorizonParameterSelection],
) -> DevelopmentFoldPrediction:
    """Generate one fold after its five inner selections have been frozen."""

    if not isinstance(inputs, S1RunnerInputs):
        raise TypeError("inputs must be verified S1RunnerInputs")
    fold = _require_development_fold(fold_id)
    selected = tuple(selections)
    if tuple((item.fold_id, item.horizon_days) for item in selected) != tuple(
        (fold, horizon) for horizon in HORIZONS_DAYS
    ):
        raise DevelopmentPredictionError("one fold requires exactly five ordered selections")
    by_horizon = {item.horizon_days: item for item in selected}
    rows = tuple(
        item
        for item in inputs.outer_issues
        if item.fold_id == fold and item.primary_exposure_selected
    )
    primary = tuple(
        build_primary_issue_prediction(
            inputs.catalog,
            inputs.location_grid,
            by_horizon[row.horizon_days],
            issue_time_utc=row.issue_time_utc,
        )
        for row in rows
    )
    weekly_issues = tuple(
        sorted({row.issue_time_utc for row in inputs.outer_issues if row.fold_id == fold})
    )
    magnitude = tuple(
        build_weekly_magnitude_snapshot(inputs.catalog, fold_id=fold, issue_time_utc=issue)
        for issue in weekly_issues
    )
    return DevelopmentFoldPrediction(fold, primary, magnitude)


def select_all_development_parameters(
    inputs: S1RunnerInputs,
) -> Mapping[str, tuple[FoldHorizonParameterSelection, ...]]:
    """Select exactly four development folds; no caller-supplied fold subset exists."""

    observed = {item.fold_id for item in inputs.outer_issues}
    if observed != set(DEVELOPMENT_FOLD_IDS):
        raise DevelopmentPredictionError("verified outer issues must contain exactly four folds")
    result = {
        fold: tuple(
            select_fold_horizon_parameters(inputs, fold_id=fold, horizon_days=horizon)
            for horizon in HORIZONS_DAYS
        )
        for fold in DEVELOPMENT_FOLD_IDS
    }
    return MappingProxyType(result)


def build_all_development_fold_predictions(
    inputs: S1RunnerInputs,
    selections: Mapping[str, Sequence[FoldHorizonParameterSelection]],
) -> tuple[DevelopmentFoldPrediction, ...]:
    """Build all four target-free fold payloads, rejecting any subset or extra fold."""

    if tuple(selections) != DEVELOPMENT_FOLD_IDS:
        raise DevelopmentPredictionError(
            "prediction requires exactly four ordered development folds"
        )
    return tuple(
        build_development_fold_prediction(
            inputs,
            fold_id=fold,
            selections=selections[fold],
        )
        for fold in DEVELOPMENT_FOLD_IDS
    )


def _frozen_fold_issue_axes(
    fold_id: str,
) -> tuple[tuple[datetime, ...], tuple[tuple[int, datetime], ...]]:
    """Rebuild the frozen local-Thursday weekly and primary axes without a ledger."""

    fold = _require_development_fold(fold_id)
    start_year, end_year = _FOLD_YEAR_BOUNDS[fold]
    start = datetime(start_year, 1, 1, tzinfo=_SHANGHAI)
    end = datetime(end_year, 1, 1, tzinfo=_SHANGHAI)
    candidate = start + timedelta(days=(3 - start.weekday()) % 7)
    weekly: list[datetime] = []
    while candidate < end:
        weekly.append(candidate.astimezone(UTC))
        candidate += timedelta(days=7)
    if len(weekly) != _EXPECTED_WEEKLY_COUNT[fold]:
        raise AssertionError("frozen weekly issue count changed")
    end_utc = end.astimezone(UTC)
    primary: list[tuple[int, datetime]] = []
    for horizon in HORIZONS_DAYS:
        mature = tuple(issue for issue in weekly if issue + timedelta(days=horizon) <= end_utc)
        selected: list[datetime] = []
        separation = timedelta(days=horizon + 30)
        for issue in mature:
            if not selected or issue >= selected[-1] + separation:
                selected.append(issue)
        primary.extend((horizon, issue) for issue in selected)
    if len(primary) != _EXPECTED_PRIMARY_COUNT:
        raise AssertionError("frozen primary issue count changed")
    return tuple(weekly), tuple(primary)


def frozen_fold_prediction_npz_schema(
    fold_id: str,
    *,
    cell_count: int = EXPECTED_25KM_CELL_COUNT,
) -> Mapping[str, Mapping[str, object]]:
    """Return the sole trusted numeric NPZ schema for one frozen development fold."""

    fold = _require_development_fold(fold_id)
    if isinstance(cell_count, bool) or not isinstance(cell_count, int) or cell_count <= 0:
        raise DevelopmentPredictionError("cell_count must be a positive integer")
    weekly_count = _EXPECTED_WEEKLY_COUNT[fold]
    primary_count = _EXPECTED_PRIMARY_COUNT

    def spec(shape: Sequence[int], dtype: str) -> Mapping[str, object]:
        return MappingProxyType({"shape": list(shape), "dtype": str(np.dtype(dtype))})

    schema = {
        "schema_version": spec((1,), "int16"),
        "fold_index": spec((1,), "int8"),
        "location_model_index": spec((len(LOCATION_MODEL_IDS),), "int8"),
        "time_band_index": spec((len(TIME_BANDS),), "int8"),
        "magnitude_model_index": spec((len(MAGNITUDE_MODEL_IDS),), "int8"),
        "primary_fold_index": spec((primary_count,), "int8"),
        "primary_issue_time_us": spec((primary_count,), "int64"),
        "primary_horizon_days": spec((primary_count,), "int16"),
        "primary_magnitude_snapshot_index": spec((primary_count,), "int32"),
        "location_regional_tau_years": spec((primary_count,), "float64"),
        "location_relative_mass": spec(
            (primary_count, len(LOCATION_MODEL_IDS), cell_count), "float64"
        ),
        "location_source_event_count": spec((primary_count, len(LOCATION_MODEL_IDS)), "int32"),
        "location_bandwidth_km": spec((primary_count, len(LOCATION_MODEL_IDS)), "float64"),
        "location_bandwidth_applicable": spec((primary_count, len(LOCATION_MODEL_IDS)), "uint8"),
        "location_alpha": spec((primary_count, len(LOCATION_MODEL_IDS)), "float64"),
        "location_alpha_applicable": spec((primary_count, len(LOCATION_MODEL_IDS)), "uint8"),
        "location_recent_fallback": spec((primary_count, len(LOCATION_MODEL_IDS)), "uint8"),
        "t0_expected_count": spec((primary_count, len(TIME_BANDS)), "float64"),
        "t1_status_code": spec((primary_count, len(TIME_BANDS)), "int8"),
        "t1_reason_code": spec((primary_count, len(TIME_BANDS)), "int8"),
        "t1_historical_block_count": spec((primary_count, len(TIME_BANDS)), "int32"),
        "t1_sample_mean_count": spec((primary_count, len(TIME_BANDS)), "float64"),
        "t1_sample_variance_count": spec((primary_count, len(TIME_BANDS)), "float64"),
        "t1_sample_variance_applicable": spec((primary_count, len(TIME_BANDS)), "uint8"),
        "t1_dispersion_k": spec((primary_count, len(TIME_BANDS)), "float64"),
        "t1_dispersion_k_applicable": spec((primary_count, len(TIME_BANDS)), "uint8"),
        "t1_observed_information_k": spec((primary_count, len(TIME_BANDS)), "float64"),
        "t1_observed_information_k_applicable": spec((primary_count, len(TIME_BANDS)), "uint8"),
        "t1_standard_error_k": spec((primary_count, len(TIME_BANDS)), "float64"),
        "t1_standard_error_k_applicable": spec((primary_count, len(TIME_BANDS)), "uint8"),
        "magnitude_fold_index": spec((weekly_count,), "int8"),
        "magnitude_issue_time_us": spec((weekly_count,), "int64"),
        "m0_training_event_count": spec((weekly_count,), "int32"),
        "m0_b_value": spec((weekly_count,), "float64"),
        "m0_bin_probability_mass": spec((weekly_count, 55), "float64"),
        "m3_training_event_count": spec((weekly_count,), "int32"),
        "m3_b_value": spec((weekly_count,), "float64"),
        "m3_bin_probability_mass": spec((weekly_count, 45), "float64"),
    }
    return MappingProxyType(schema)


def fold_prediction_npz_arrays(
    prediction: DevelopmentFoldPrediction,
    *,
    cell_count: int,
) -> Mapping[str, NDArray[Any]]:
    """Return a deterministic, numeric-only, outer-answer-free NPZ payload.

    Model axes are bound by ``LOCATION_MODEL_IDS``, ``TIME_BANDS``, and
    ``MAGNITUDE_MODEL_IDS``.  ``fold_index`` binds the per-file fold to
    ``DEVELOPMENT_FOLD_IDS``.  The caller may serialise this mapping with
    ``numpy.savez`` only after its own no-overwrite and seal checks.
    """

    if not isinstance(prediction, DevelopmentFoldPrediction):
        raise TypeError("prediction must be a DevelopmentFoldPrediction")
    if isinstance(cell_count, bool) or not isinstance(cell_count, int) or cell_count <= 0:
        raise DevelopmentPredictionError("cell_count must be a positive integer")
    primary = prediction.primary_predictions
    magnitude = prediction.magnitude_snapshots
    fold_index = DEVELOPMENT_FOLD_IDS.index(prediction.fold_id)
    location_mass = np.empty((len(primary), len(LOCATION_MODEL_IDS), cell_count), dtype=np.float64)
    location_source_count = np.empty((len(primary), len(LOCATION_MODEL_IDS)), dtype=np.int32)
    location_bandwidth = np.zeros((len(primary), len(LOCATION_MODEL_IDS)), dtype=np.float64)
    location_bandwidth_applicable = np.zeros(
        (len(primary), len(LOCATION_MODEL_IDS)), dtype=np.uint8
    )
    location_alpha = np.zeros((len(primary), len(LOCATION_MODEL_IDS)), dtype=np.float64)
    location_alpha_applicable = np.zeros((len(primary), len(LOCATION_MODEL_IDS)), dtype=np.uint8)
    location_recent_fallback = np.zeros((len(primary), len(LOCATION_MODEL_IDS)), dtype=np.uint8)
    t0_mean = np.empty((len(primary), len(TIME_BANDS)), dtype=np.float64)
    t1_status = np.empty((len(primary), len(TIME_BANDS)), dtype=np.int8)
    t1_reason = np.empty((len(primary), len(TIME_BANDS)), dtype=np.int8)
    t1_block_count = np.empty((len(primary), len(TIME_BANDS)), dtype=np.int32)
    t1_sample_mean = np.empty((len(primary), len(TIME_BANDS)), dtype=np.float64)
    t1_sample_variance = np.zeros((len(primary), len(TIME_BANDS)), dtype=np.float64)
    t1_sample_variance_applicable = np.zeros((len(primary), len(TIME_BANDS)), dtype=np.uint8)
    t1_k = np.zeros((len(primary), len(TIME_BANDS)), dtype=np.float64)
    t1_k_applicable = np.zeros((len(primary), len(TIME_BANDS)), dtype=np.uint8)
    t1_information = np.zeros((len(primary), len(TIME_BANDS)), dtype=np.float64)
    t1_information_applicable = np.zeros((len(primary), len(TIME_BANDS)), dtype=np.uint8)
    t1_standard_error = np.zeros((len(primary), len(TIME_BANDS)), dtype=np.float64)
    t1_standard_error_applicable = np.zeros((len(primary), len(TIME_BANDS)), dtype=np.uint8)
    for row_index, row in enumerate(primary):
        for model_index, surface in enumerate(row.location_surfaces):
            if surface.cell_relative_mass.shape != (cell_count,):
                raise DevelopmentPredictionError("location surface cell count changed")
            location_mass[row_index, model_index, :] = surface.cell_relative_mass
            location_source_count[row_index, model_index] = surface.source_event_count
            if surface.bandwidth_km is not None:
                location_bandwidth[row_index, model_index] = surface.bandwidth_km
                location_bandwidth_applicable[row_index, model_index] = 1
            if surface.alpha is not None:
                location_alpha[row_index, model_index] = surface.alpha
                location_alpha_applicable[row_index, model_index] = 1
            location_recent_fallback[row_index, model_index] = int(surface.recent_fallback_to_long)
        for band_index, forecast in enumerate(row.count_forecasts):
            t0_mean[row_index, band_index] = forecast.poisson_expected_count
            qualification = forecast.t1_qualification
            t1_status[row_index, band_index] = _NB2_STATUS_CODE[qualification.status]
            try:
                t1_reason[row_index, band_index] = _NB2_REASON_CODE[qualification.reason]
            except KeyError as exc:
                raise DevelopmentPredictionError(
                    "T1 qualification reason is absent from the frozen numeric codebook"
                ) from exc
            t1_block_count[row_index, band_index] = qualification.historical_block_count
            t1_sample_mean[row_index, band_index] = qualification.sample_mean_count
            if qualification.sample_variance_count is not None:
                t1_sample_variance[row_index, band_index] = qualification.sample_variance_count
                t1_sample_variance_applicable[row_index, band_index] = 1
            if qualification.dispersion_k is not None:
                t1_k[row_index, band_index] = qualification.dispersion_k
                t1_k_applicable[row_index, band_index] = 1
            if qualification.observed_information_k is not None:
                t1_information[row_index, band_index] = qualification.observed_information_k
                t1_information_applicable[row_index, band_index] = 1
            if qualification.standard_error_k is not None:
                t1_standard_error[row_index, band_index] = qualification.standard_error_k
                t1_standard_error_applicable[row_index, band_index] = 1

    magnitude_index_by_issue = {item.issue_time_utc: index for index, item in enumerate(magnitude)}

    arrays: dict[str, NDArray[Any]] = {
        "schema_version": _readonly([PREDICTION_ARRAY_SCHEMA_VERSION], dtype=np.dtype("int16")),
        "fold_index": _readonly([fold_index], dtype=np.dtype("int8")),
        "location_model_index": _readonly(
            np.arange(len(LOCATION_MODEL_IDS)), dtype=np.dtype("int8")
        ),
        "time_band_index": _readonly(np.arange(len(TIME_BANDS)), dtype=np.dtype("int8")),
        "magnitude_model_index": _readonly(
            np.arange(len(MAGNITUDE_MODEL_IDS)), dtype=np.dtype("int8")
        ),
        "primary_fold_index": _readonly(np.full(len(primary), fold_index), dtype=np.dtype("int8")),
        "primary_issue_time_us": _readonly(
            [_epoch_us(item.issue_time_utc) for item in primary], dtype=np.dtype("int64")
        ),
        "primary_horizon_days": _readonly(
            [item.horizon_days for item in primary], dtype=np.dtype("int16")
        ),
        "primary_magnitude_snapshot_index": _readonly(
            [magnitude_index_by_issue[item.issue_time_utc] for item in primary],
            dtype=np.dtype("int32"),
        ),
        "location_regional_tau_years": _readonly(
            [item.regional_tau_years for item in primary], dtype=np.dtype("float64")
        ),
        "location_relative_mass": _readonly(location_mass, dtype=np.dtype("float64")),
        "location_source_event_count": _readonly(location_source_count, dtype=np.dtype("int32")),
        "location_bandwidth_km": _readonly(location_bandwidth, dtype=np.dtype("float64")),
        "location_bandwidth_applicable": _readonly(
            location_bandwidth_applicable, dtype=np.dtype("uint8")
        ),
        "location_alpha": _readonly(location_alpha, dtype=np.dtype("float64")),
        "location_alpha_applicable": _readonly(location_alpha_applicable, dtype=np.dtype("uint8")),
        "location_recent_fallback": _readonly(location_recent_fallback, dtype=np.dtype("uint8")),
        "t0_expected_count": _readonly(t0_mean, dtype=np.dtype("float64")),
        "t1_status_code": _readonly(t1_status, dtype=np.dtype("int8")),
        "t1_reason_code": _readonly(t1_reason, dtype=np.dtype("int8")),
        "t1_historical_block_count": _readonly(t1_block_count, dtype=np.dtype("int32")),
        "t1_sample_mean_count": _readonly(t1_sample_mean, dtype=np.dtype("float64")),
        "t1_sample_variance_count": _readonly(t1_sample_variance, dtype=np.dtype("float64")),
        "t1_sample_variance_applicable": _readonly(
            t1_sample_variance_applicable, dtype=np.dtype("uint8")
        ),
        "t1_dispersion_k": _readonly(t1_k, dtype=np.dtype("float64")),
        "t1_dispersion_k_applicable": _readonly(t1_k_applicable, dtype=np.dtype("uint8")),
        "t1_observed_information_k": _readonly(t1_information, dtype=np.dtype("float64")),
        "t1_observed_information_k_applicable": _readonly(
            t1_information_applicable, dtype=np.dtype("uint8")
        ),
        "t1_standard_error_k": _readonly(t1_standard_error, dtype=np.dtype("float64")),
        "t1_standard_error_k_applicable": _readonly(
            t1_standard_error_applicable, dtype=np.dtype("uint8")
        ),
        "magnitude_fold_index": _readonly(
            np.full(len(magnitude), fold_index), dtype=np.dtype("int8")
        ),
        "magnitude_issue_time_us": _readonly(
            [_epoch_us(item.issue_time_utc) for item in magnitude], dtype=np.dtype("int64")
        ),
        "m0_training_event_count": _readonly(
            [item.m0.training_event_count for item in magnitude], dtype=np.dtype("int32")
        ),
        "m0_b_value": _readonly([item.m0.b_value for item in magnitude], dtype=np.dtype("float64")),
        "m0_bin_probability_mass": _readonly(
            [item.m0.bin_probability_masses for item in magnitude], dtype=np.dtype("float64")
        ),
        "m3_training_event_count": _readonly(
            [item.m3.training_event_count for item in magnitude], dtype=np.dtype("int32")
        ),
        "m3_b_value": _readonly([item.m3.b_value for item in magnitude], dtype=np.dtype("float64")),
        "m3_bin_probability_mass": _readonly(
            [item.m3.bin_probability_masses for item in magnitude], dtype=np.dtype("float64")
        ),
    }
    result = MappingProxyType(arrays)
    validate_frozen_fold_prediction_npz_arrays(
        prediction.fold_id,
        result,
        cell_count=cell_count,
    )
    return result


def validate_frozen_fold_prediction_npz_arrays(
    fold_id: str,
    arrays: Mapping[str, object],
    *,
    cell_count: int = EXPECTED_25KM_CELL_COUNT,
) -> None:
    """Validate one in-memory or reloaded NPZ against frozen axes and science invariants."""

    fold = _require_development_fold(fold_id)
    if not isinstance(arrays, Mapping):
        raise TypeError("arrays must be a mapping of numeric numpy arrays")
    schema = frozen_fold_prediction_npz_schema(fold, cell_count=cell_count)
    if set(arrays) != set(schema):
        raise DevelopmentPredictionError("prediction NPZ keys differ from the frozen schema")
    materialized: dict[str, NDArray[Any]] = {}
    for key, specification in schema.items():
        array = np.asarray(arrays[key])
        expected_shape = tuple(cast(list[int], specification["shape"]))
        expected_dtype = np.dtype(cast(str, specification["dtype"]))
        if array.shape != expected_shape or array.dtype != expected_dtype or array.dtype.hasobject:
            raise DevelopmentPredictionError(
                f"prediction NPZ shape or dtype differs from frozen schema: {key}"
            )
        materialized[key] = array

    weekly, primary_axis = _frozen_fold_issue_axes(fold)
    fold_index = DEVELOPMENT_FOLD_IDS.index(fold)
    expected_primary_issue = np.asarray(
        [_epoch_us(issue) for _, issue in primary_axis], dtype=np.int64
    )
    expected_primary_horizon = np.asarray([horizon for horizon, _ in primary_axis], dtype=np.int16)
    expected_weekly_issue = np.asarray([_epoch_us(issue) for issue in weekly], dtype=np.int64)
    weekly_index_by_issue = {value: index for index, value in enumerate(expected_weekly_issue)}
    expected_snapshot_index = np.asarray(
        [weekly_index_by_issue[value] for value in expected_primary_issue], dtype=np.int32
    )

    exact_vectors: tuple[tuple[str, NDArray[Any]], ...] = (
        ("schema_version", np.asarray([PREDICTION_ARRAY_SCHEMA_VERSION], dtype=np.int16)),
        ("fold_index", np.asarray([fold_index], dtype=np.int8)),
        (
            "location_model_index",
            np.arange(len(LOCATION_MODEL_IDS), dtype=np.int8),
        ),
        ("time_band_index", np.arange(len(TIME_BANDS), dtype=np.int8)),
        (
            "magnitude_model_index",
            np.arange(len(MAGNITUDE_MODEL_IDS), dtype=np.int8),
        ),
        (
            "primary_fold_index",
            np.full(_EXPECTED_PRIMARY_COUNT, fold_index, dtype=np.int8),
        ),
        ("primary_issue_time_us", expected_primary_issue),
        ("primary_horizon_days", expected_primary_horizon),
        ("primary_magnitude_snapshot_index", expected_snapshot_index),
        (
            "magnitude_fold_index",
            np.full(len(weekly), fold_index, dtype=np.int8),
        ),
        ("magnitude_issue_time_us", expected_weekly_issue),
    )
    for key, expected in exact_vectors:
        if not np.array_equal(materialized[key], expected):
            raise DevelopmentPredictionError(f"prediction NPZ frozen alignment changed: {key}")

    regional_tau = cast(FloatArray, materialized["location_regional_tau_years"])
    if not np.isfinite(regional_tau).all() or not set(
        float(value) for value in regional_tau
    ) <= set(FROZEN_REGIONAL_TAU_YEARS):
        raise DevelopmentPredictionError("regional tau values left the frozen candidate set")
    for horizon in HORIZONS_DAYS:
        rows = np.flatnonzero(expected_primary_horizon == horizon)
        if not np.all(regional_tau[rows] == regional_tau[rows][0]):
            raise DevelopmentPredictionError("regional tau changed within one fold-horizon")

    location_mass = cast(FloatArray, materialized["location_relative_mass"])
    if (
        not np.isfinite(location_mass).all()
        or np.any(location_mass < 0.0)
        or not np.allclose(
            np.sum(location_mass, axis=2, dtype=np.float64),
            1.0,
            rtol=0.0,
            atol=1.0e-12,
        )
    ):
        raise DevelopmentPredictionError(
            "every location surface must be finite, non-negative, and unit-normalized"
        )
    source_count = materialized["location_source_event_count"]
    if np.any(source_count < 0):
        raise DevelopmentPredictionError("location source counts must be non-negative")
    if np.any(source_count[:, 0] != 0) or not np.all(source_count[:, 1:] == source_count[:, 1:2]):
        raise DevelopmentPredictionError(
            "location source counts must keep L0 empty and L1/L2/L2_75/L3 aligned"
        )

    def binary_mask(key: str) -> NDArray[np.uint8]:
        value = cast(NDArray[np.uint8], materialized[key])
        if np.any((value != 0) & (value != 1)):
            raise DevelopmentPredictionError(f"prediction applicability mask is not binary: {key}")
        return value

    bandwidth = cast(FloatArray, materialized["location_bandwidth_km"])
    bandwidth_mask = binary_mask("location_bandwidth_applicable")
    expected_bandwidth_mask = np.asarray([0, 0, 1, 1, 1], dtype=np.uint8)
    if not np.array_equal(
        bandwidth_mask,
        np.broadcast_to(expected_bandwidth_mask, bandwidth_mask.shape),
    ):
        raise DevelopmentPredictionError("location bandwidth applicability changed")
    if (
        not np.isfinite(bandwidth).all()
        or np.any(bandwidth[bandwidth_mask == 0] != 0.0)
        or np.any(bandwidth[bandwidth_mask == 1] <= 0.0)
        or np.any(bandwidth[:, 3] != 75.0)
        or not np.array_equal(bandwidth[:, 2], bandwidth[:, 4])
        or not set(float(value) for value in bandwidth[:, 2]) <= set(FROZEN_KDE_BANDWIDTHS_KM)
    ):
        raise DevelopmentPredictionError("location bandwidth values or masks are inconsistent")
    alpha = cast(FloatArray, materialized["location_alpha"])
    alpha_mask = binary_mask("location_alpha_applicable")
    expected_alpha_mask = np.asarray([0, 0, 0, 0, 1], dtype=np.uint8)
    if not np.array_equal(alpha_mask, np.broadcast_to(expected_alpha_mask, alpha_mask.shape)):
        raise DevelopmentPredictionError("location alpha applicability changed")
    if (
        not np.isfinite(alpha).all()
        or np.any(alpha[alpha_mask == 0] != 0.0)
        or not set(float(value) for value in alpha[:, 4]) <= set(FROZEN_R30_ALPHA_CANDIDATES)
    ):
        raise DevelopmentPredictionError("location alpha values or masks are inconsistent")
    fallback = binary_mask("location_recent_fallback")
    if np.any(fallback[:, :4] != 0):
        raise DevelopmentPredictionError("recent fallback flag may occur only on L3")

    t0_mean = cast(FloatArray, materialized["t0_expected_count"])
    if not np.isfinite(t0_mean).all() or np.any(t0_mean < 0.0):
        raise DevelopmentPredictionError("T0 expected counts must be finite and non-negative")
    if not np.allclose(
        t0_mean[:, 2],
        t0_mean[:, 0] + t0_mean[:, 1],
        rtol=1.0e-14,
        atol=1.0e-15,
    ):
        raise DevelopmentPredictionError(
            "T0 M5+ expectation must equal the M5-6 plus M6+ expectations"
        )
    status = materialized["t1_status_code"]
    reason = materialized["t1_reason_code"]
    if np.any((status < 0) | (status >= len(NB2_STATUS_CODES))):
        raise DevelopmentPredictionError("T1 status code is outside the frozen codebook")
    if np.any((reason < 0) | (reason >= len(NB2_REASON_CODES))):
        raise DevelopmentPredictionError("T1 reason code is outside the frozen codebook")
    expected_status_by_reason = np.asarray([0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 2, 1], dtype=np.int8)
    if not np.array_equal(status, expected_status_by_reason[reason]):
        raise DevelopmentPredictionError("T1 status and reason code are inconsistent")
    block_count = materialized["t1_historical_block_count"]
    sample_mean = cast(FloatArray, materialized["t1_sample_mean_count"])
    if np.any(block_count <= 0) or not np.isfinite(sample_mean).all() or np.any(sample_mean < 0.0):
        raise DevelopmentPredictionError("T1 pooled history statistics are invalid")

    optional_fields = (
        ("t1_sample_variance_count", "t1_sample_variance_applicable", False),
        ("t1_dispersion_k", "t1_dispersion_k_applicable", True),
        (
            "t1_observed_information_k",
            "t1_observed_information_k_applicable",
            True,
        ),
        ("t1_standard_error_k", "t1_standard_error_k_applicable", True),
    )
    masks: dict[str, NDArray[np.uint8]] = {}
    for value_key, mask_key, positive_when_present in optional_fields:
        values = cast(FloatArray, materialized[value_key])
        mask = binary_mask(mask_key)
        masks[mask_key] = mask
        if not np.isfinite(values).all() or np.any(values[mask == 0] != 0.0):
            raise DevelopmentPredictionError(f"T1 optional field mask is inconsistent: {value_key}")
        if positive_when_present:
            invalid_present = np.any(values[mask == 1] <= 0.0)
        else:
            invalid_present = np.any(values[mask == 1] < 0.0)
        if invalid_present:
            raise DevelopmentPredictionError(f"T1 optional field values are invalid: {value_key}")
    evaluable = status == _NB2_STATUS_CODE["evaluable"]
    for mask_key in (
        "t1_dispersion_k_applicable",
        "t1_observed_information_k_applicable",
        "t1_standard_error_k_applicable",
    ):
        if not np.array_equal(masks[mask_key].astype(bool), evaluable):
            raise DevelopmentPredictionError("T1 evaluable status and fitted-field masks disagree")

    seven_day_rows = expected_primary_horizon == 7
    m5_status = status[seven_day_rows, 0]
    m6_status = status[seven_day_rows, 1]
    m6_reason = reason[seven_day_rows, 1]
    m5_k = materialized["t1_dispersion_k"][seven_day_rows, 0]
    m6_k = materialized["t1_dispersion_k"][seven_day_rows, 1]
    source_evaluable = m5_status == _NB2_STATUS_CODE["evaluable"]
    if (
        np.any(m6_status[source_evaluable] != _NB2_STATUS_CODE["evaluable"])
        or np.any(m6_reason[source_evaluable] != _NB2_REASON_CODE[NB2_REASON_CODES[10]])
        or not np.array_equal(m6_k[source_evaluable], m5_k[source_evaluable])
        or np.any(m6_status[~source_evaluable] != _NB2_STATUS_CODE["poisson_limit"])
        or np.any(m6_reason[~source_evaluable] != _NB2_REASON_CODE[NB2_REASON_CODES[11]])
    ):
        raise DevelopmentPredictionError("frozen M6+ seven-day T1 sharing rule changed")

    qualification_fields = (
        "t1_status_code",
        "t1_reason_code",
        "t1_historical_block_count",
        "t1_sample_mean_count",
        "t1_sample_variance_count",
        "t1_sample_variance_applicable",
        "t1_dispersion_k",
        "t1_dispersion_k_applicable",
        "t1_observed_information_k",
        "t1_observed_information_k_applicable",
        "t1_standard_error_k",
        "t1_standard_error_k_applicable",
    )
    for horizon in HORIZONS_DAYS:
        rows = np.flatnonzero(expected_primary_horizon == horizon)
        for key in qualification_fields:
            values = materialized[key][rows]
            if not np.all(values == values[0]):
                raise DevelopmentPredictionError(
                    "T1 qualification changed between issues in one fold-horizon"
                )

    for prefix, expected_bins in (("m0", 55), ("m3", 45)):
        training_count = materialized[f"{prefix}_training_event_count"]
        b_value = cast(FloatArray, materialized[f"{prefix}_b_value"])
        mass = cast(FloatArray, materialized[f"{prefix}_bin_probability_mass"])
        if mass.shape[1] != expected_bins:
            raise AssertionError("frozen magnitude bin count changed")
        if (
            np.any(training_count <= 0)
            or not np.isfinite(b_value).all()
            or np.any(b_value <= 0.0)
            or not np.isfinite(mass).all()
            or np.any(mass <= 0.0)
            or not np.allclose(
                np.sum(mass, axis=1, dtype=np.float64),
                1.0,
                rtol=0.0,
                atol=1.0e-12,
            )
        ):
            raise DevelopmentPredictionError(
                f"{prefix.upper()} magnitude snapshots are invalid or not normalized"
            )


__all__ = [
    "LOCATION_MODEL_IDS",
    "MAGNITUDE_MODEL_IDS",
    "NB2_REASON_CODES",
    "NB2_STATUS_CODES",
    "PREDICTION_ARRAY_SCHEMA_VERSION",
    "TIME_BANDS",
    "BandCountForecast",
    "BandNB2Qualification",
    "CandidateInnerBlockLogDensities",
    "DevelopmentFoldPrediction",
    "DevelopmentPredictionError",
    "FoldHorizonParameterSelection",
    "InnerTimeCountSeries",
    "LocationParameterSelection",
    "PrimaryIssuePrediction",
    "StrictlyEarlierInnerLocationBlock",
    "StrictlyEarlierInnerLocationWindow",
    "WeeklyMagnitudeSnapshot",
    "build_all_development_fold_predictions",
    "build_development_fold_prediction",
    "build_primary_issue_prediction",
    "build_strictly_earlier_inner_m4_location_block",
    "build_strictly_earlier_inner_time_count_series",
    "build_weekly_magnitude_snapshot",
    "fit_t1_qualifications_from_inner_count_series",
    "fold_prediction_npz_arrays",
    "frozen_fold_prediction_npz_schema",
    "select_all_development_parameters",
    "select_fold_horizon_parameters",
    "select_location_parameters_from_strictly_earlier_inner_blocks",
    "validate_frozen_fold_prediction_npz_arrays",
]
