"""Score-free catalog conditioning for the frozen S3-A anomaly experiment.

The caller supplies an authenticated catalog and the fixed national grid.  Only
history visible by ``Q = T - 24h`` enters these calculations; this module neither
loads files nor constructs future targets.  The fixed C2B weights are transferred
from the latest old C-development fit, not selected again on the A period.

Spatial outputs are normalized relative cell masses.  The separate T0 output is
a Poisson model expectation, not an empirically validated event probability.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Final

import numpy as np
from numpy.typing import NDArray

from seismoflux.multitask_s1.c2b_models import gaussian_log_masses, mix_log_masses
from seismoflux.multitask_s1.location import FrozenSpatialGrid
from seismoflux.multitask_s1.runner_inputs import (
    CATALOG_HISTORY_START_UTC,
    MAIN_CATALOG_DELAY,
    CatalogEventTable,
    causal_catalog_histories,
)
from seismoflux.multitask_s1.time_magnitude import ExpandingPoissonRate, fit_expanding_poisson

FloatArray = NDArray[np.float64]

KERNEL_SCALES_KM: Final = (25.0, 75.0, 150.0)
SPATIAL_WEIGHTS_BY_HORIZON: Final[Mapping[int, tuple[float, float, float]]] = MappingProxyType(
    {
        7: (0.5, 0.5, 0.0),
        30: (0.5, 0.5, 0.0),
        90: (0.5, 0.5, 0.0),
        180: (0.5, 0.5, 0.0),
        365: (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    }
)
COUNT_HISTORY_NAMES: Final = MappingProxyType(
    {
        "Ms5_6": "m5_6",
        "Ms6_plus": "m6_plus",
        "Ms5_plus": "m5_plus_1970_for_joint",
    }
)
_DAY_US: Final = 86_400_000_000
_UTC_EPOCH: Final = datetime(1970, 1, 1, tzinfo=UTC)


def _aware_utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("issue_time must be a timezone-aware datetime")
    return value.astimezone(UTC)


def _epoch_us(value: datetime) -> int:
    delta = value - _UTC_EPOCH
    return (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds


@dataclass(frozen=True, slots=True)
class CatalogHistoryWaterlevel:
    """Counts and time bounds of conditioning history only, never target counts."""

    issue_time_utc: datetime
    data_cutoff_utc: datetime
    history_start_utc: datetime
    history_exposure_days: float
    history_counts: Mapping[str, int]
    recent_m4_event_count: int
    earliest_m4_origin_time_us: int | None
    latest_m4_origin_time_us: int | None
    latest_m4_available_at_us: int | None
    availability_basis: str = "canonical_availability_only"
    recent_interval: str = "(T-30d,T-24h]"


@dataclass(frozen=True, slots=True)
class CatalogBackgroundPrediction:
    """One horizon's frozen spatial references and T0 count expectations."""

    horizon_days: int
    primary_log_mass: FloatArray
    r30_reference_log_mass: FloatArray
    expected_counts: Mapping[str, float]
    poisson_at_least_one: Mapping[str, float]
    waterlevel: CatalogHistoryWaterlevel


@dataclass(frozen=True, slots=True)
class CatalogBackgroundComponents:
    """Compute kernels once per issue, then cheaply materialize all five horizons."""

    kernel_log_masses: Mapping[float, FloatArray]
    r30_reference_log_mass: FloatArray
    poisson_rates: Mapping[str, ExpandingPoissonRate]
    waterlevel: CatalogHistoryWaterlevel

    def for_horizon(self, horizon_days: int) -> CatalogBackgroundPrediction:
        if isinstance(horizon_days, bool) or not isinstance(horizon_days, int):
            raise TypeError("horizon_days must be one of the five frozen integer horizons")
        horizon = int(horizon_days)
        if horizon not in SPATIAL_WEIGHTS_BY_HORIZON:
            raise ValueError("horizon_days must be one of 7, 30, 90, 180, 365")
        primary = mix_log_masses(
            [self.kernel_log_masses[scale] for scale in KERNEL_SCALES_KM],
            SPATIAL_WEIGHTS_BY_HORIZON[horizon],
        )
        means = {band: rate.expected_count(horizon) for band, rate in self.poisson_rates.items()}
        probabilities = {
            band: rate.at_least_one_probability(horizon)
            for band, rate in self.poisson_rates.items()
        }
        return CatalogBackgroundPrediction(
            horizon_days=horizon,
            primary_log_mass=primary,
            r30_reference_log_mass=self.r30_reference_log_mass,
            expected_counts=MappingProxyType(means),
            poisson_at_least_one=MappingProxyType(probabilities),
            waterlevel=self.waterlevel,
        )


def build_catalog_background_components(
    catalog: CatalogEventTable,
    grid: FrozenSpatialGrid,
    issue_time: datetime,
    *,
    chunk_size: int = 256,
) -> CatalogBackgroundComponents:
    """Condition the frozen D0 background on history available at a new A issue.

    D0 means inside-country, raw user-declared Ms >= 4, origin >= local 1970,
    and origin/availability <= Q.  Canonical availability retains the S0/S1
    source semantics; no new source-membership or completeness gate is added.
    R30 keeps the inherited interval ``(T-30d, Q]`` (not ``(Q-30d, Q]``).

    Actual authorized report dates are supplied by the S3 calendar layer.  This
    pure function does not infer a calendar or reuse old prediction arrays.
    """
    if not isinstance(grid, FrozenSpatialGrid):
        raise TypeError("grid must be a FrozenSpatialGrid")
    issue_utc = _aware_utc(issue_time)
    cutoff = issue_utc - MAIN_CATALOG_DELAY
    exposure_days = (cutoff - CATALOG_HISTORY_START_UTC).total_seconds() / 86_400.0
    if exposure_days <= 0.0:
        raise ValueError("catalog cutoff must follow the local 1970 history start")
    histories = causal_catalog_histories(catalog, issue_utc)
    long_history = histories["m4_plus"]
    training_xy = np.column_stack((long_history.spatial.x_km, long_history.spatial.y_km))
    query_xy = np.column_stack((grid.x_km, grid.y_km))
    kernels = gaussian_log_masses(
        training_xy,
        query_xy,
        grid.area_km2,
        bandwidths_km=KERNEL_SCALES_KM,
        chunk_size=chunk_size,
    )
    recent = long_history.origin_time_us > _epoch_us(issue_utc) - 30 * _DAY_US
    recent_count = int(np.count_nonzero(recent))
    if recent_count:
        recent_75 = gaussian_log_masses(
            training_xy[recent],
            query_xy,
            grid.area_km2,
            bandwidths_km=(75.0,),
            chunk_size=chunk_size,
        )[75.0]
        r30 = mix_log_masses([kernels[75.0], recent_75], [0.75, 0.25])
    else:
        # The inherited empty-recent fallback is the long-history KDE75, not uniform.
        r30 = kernels[75.0]

    history_counts = {"Ms4_plus": long_history.event_count}
    rates: dict[str, ExpandingPoissonRate] = {}
    for band, history_name in COUNT_HISTORY_NAMES.items():
        count = histories[history_name].event_count
        history_counts[band] = count
        rates[band] = fit_expanding_poisson(
            history_event_count=count,
            history_exposure_days=exposure_days,
        )
    waterlevel = CatalogHistoryWaterlevel(
        issue_time_utc=issue_utc,
        data_cutoff_utc=cutoff,
        history_start_utc=CATALOG_HISTORY_START_UTC,
        history_exposure_days=exposure_days,
        history_counts=MappingProxyType(history_counts),
        recent_m4_event_count=recent_count,
        earliest_m4_origin_time_us=(
            int(long_history.origin_time_us[0]) if long_history.event_count else None
        ),
        latest_m4_origin_time_us=(
            int(long_history.origin_time_us[-1]) if long_history.event_count else None
        ),
        latest_m4_available_at_us=(
            int(np.max(long_history.available_at_us)) if long_history.event_count else None
        ),
    )
    return CatalogBackgroundComponents(
        kernel_log_masses=MappingProxyType(kernels),
        r30_reference_log_mass=r30,
        poisson_rates=MappingProxyType(rates),
        waterlevel=waterlevel,
    )


def build_catalog_background(
    catalog: CatalogEventTable,
    grid: FrozenSpatialGrid,
    issue_time: datetime,
    horizon_days: int,
    *,
    chunk_size: int = 256,
) -> CatalogBackgroundPrediction:
    """Convenience wrapper; reuse components for multiple horizons of one issue."""
    return build_catalog_background_components(
        catalog, grid, issue_time, chunk_size=chunk_size
    ).for_horizon(horizon_days)


__all__ = [
    "COUNT_HISTORY_NAMES",
    "KERNEL_SCALES_KM",
    "SPATIAL_WEIGHTS_BY_HORIZON",
    "CatalogBackgroundComponents",
    "CatalogBackgroundPrediction",
    "CatalogHistoryWaterlevel",
    "build_catalog_background",
    "build_catalog_background_components",
]
