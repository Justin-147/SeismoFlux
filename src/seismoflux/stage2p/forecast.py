"""P0/P1/PP synthetic forecasts using the frozen Stage 2S spatial kernels."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import numpy as np

from seismoflux.stage2p.catalog import (
    CausalCatalogWindows,
    SyntheticEvent,
    select_causal_windows,
)
from seismoflux.stage2s.contracts import (
    AlarmMask,
    NormalizedSpatialDensity,
    SpatialQuadratureFamily,
)
from seismoflux.stage2s.spatial import (
    build_normalized_kde,
    build_recent_component,
    mix_density,
    select_alarm_prefix,
)

ScienceModelId = Literal["P0", "P1", "PP"]
FROZEN_BANDWIDTH_KM = 75.0
FROZEN_MIXTURE_WEIGHT = 0.5
ALARM_BUDGET_KM2 = 600_000.0
MAXIMUM_COMPLETE_CELL_AREA_KM2 = 625.0
MAXIMUM_PAIRWISE_ALARM_AREA_DIFFERENCE_KM2 = 625.0


@dataclass(frozen=True, slots=True)
class ScienceModelForecast:
    """One scientifically labelled forecast and its complete alarm prefix."""

    model_id: ScienceModelId
    spatial_density: NormalizedSpatialDensity
    alarm: AlarmMask
    component_event_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        allowed_density_ids = {
            "P0": {"S0"},
            "P1": {"S0", "S1"},
            "PP": {"S0", "SP"},
        }
        if self.spatial_density.model_id not in allowed_density_ids[self.model_id]:
            raise ValueError("scientific model label does not match its spatial density")
        if self.alarm.model_id != self.spatial_density.model_id:
            raise ValueError("alarm and spatial density model ids must match")
        event_ids = tuple(self.component_event_ids)
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("component event ids must be unique")
        object.__setattr__(self, "component_event_ids", event_ids)


@dataclass(frozen=True, slots=True)
class ScienceForecastBundle:
    """The one P0/P1/PP comparison generated from a common causal snapshot."""

    windows: CausalCatalogWindows
    p0: ScienceModelForecast
    p1: ScienceModelForecast
    pp: ScienceModelForecast

    def __post_init__(self) -> None:
        if (self.p0.model_id, self.p1.model_id, self.pp.model_id) != (
            "P0",
            "P1",
            "PP",
        ):
            raise ValueError("forecasts must be ordered as P0, P1, and PP")
        family = self.p0.spatial_density.grid_family
        if not (
            self.p1.spatial_density.grid_family is family
            and self.pp.spatial_density.grid_family is family
        ):
            raise ValueError("P0, P1, and PP must share one grid-family object")
        areas = (
            self.p0.alarm.actual_area_km2,
            self.p1.alarm.actual_area_km2,
            self.pp.alarm.actual_area_km2,
        )
        if any(area > ALARM_BUDGET_KM2 for area in areas):
            raise ValueError("an alarm exceeds the frozen 600000 km2 budget")
        if max(areas) - min(areas) > MAXIMUM_PAIRWISE_ALARM_AREA_DIFFERENCE_KM2:
            raise ValueError("P0/P1/PP actual alarm areas differ by more than 625 km2")

    @property
    def models(self) -> tuple[ScienceModelForecast, ...]:
        return (self.p0, self.p1, self.pp)

    def at(self, model_id: ScienceModelId) -> ScienceModelForecast:
        return {"P0": self.p0, "P1": self.p1, "PP": self.pp}[model_id]


def _xy(events: tuple[SyntheticEvent, ...]) -> np.ndarray:
    result = np.asarray([(event.x_km, event.y_km) for event in events], dtype=np.float64)
    if result.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    return result.reshape((-1, 2))


def _validate_alarm_grid(grid_family: SpatialQuadratureFamily) -> None:
    grid = grid_family.at(25.0)
    maximum_area = max(float(area) for area in grid.clipped_area_km2)
    if maximum_area > MAXIMUM_COMPLETE_CELL_AREA_KM2:
        raise ValueError("every complete 25 km alarm cell must have area at most 625 km2")


def build_forecast_from_windows(
    windows: CausalCatalogWindows,
    grid_family: SpatialQuadratureFamily,
) -> ScienceForecastBundle:
    """Build fixed 75 km P0/P1/PP forecasts from target-free windows."""

    if not isinstance(windows, CausalCatalogWindows):
        raise TypeError("windows must be CausalCatalogWindows")
    if not isinstance(grid_family, SpatialQuadratureFamily):
        raise TypeError("grid_family must be SpatialQuadratureFamily")
    safe_windows = CausalCatalogWindows(
        issue_time=windows.issue_time,
        query_cutoff=windows.query_cutoff,
        training_start=windows.training_start,
        p0_events=tuple(windows.p0_events),
        r30_events=tuple(windows.r30_events),
        rp30_events=tuple(windows.rp30_events),
    )
    if not safe_windows.p0_events:
        raise ValueError("P0 requires at least one causal training event")
    _validate_alarm_grid(grid_family)

    p0_density = build_normalized_kde(
        _xy(safe_windows.p0_events),
        grid_family,
        model_id="S0",
        bandwidth_km=FROZEN_BANDWIDTH_KM,
    )
    if safe_windows.r30_events:
        recent = build_recent_component(
            _xy(safe_windows.r30_events),
            grid_family,
            component_id="R",
            empty_fallback_s0=p0_density,
        )
        p1_density = mix_density(
            p0_density,
            recent,
            FROZEN_MIXTURE_WEIGHT,
            model_id="S1",
        )
    else:
        p1_density = p0_density
    if safe_windows.rp30_events:
        preceding = build_recent_component(
            _xy(safe_windows.rp30_events),
            grid_family,
            component_id="RP",
            empty_fallback_s0=p0_density,
        )
        pp_density = mix_density(
            p0_density,
            preceding,
            FROZEN_MIXTURE_WEIGHT,
            model_id="SP",
        )
    else:
        pp_density = p0_density

    p0_alarm = select_alarm_prefix(p0_density, budget_km2=ALARM_BUDGET_KM2)
    p1_alarm = (
        p0_alarm
        if p1_density is p0_density
        else select_alarm_prefix(p1_density, budget_km2=ALARM_BUDGET_KM2)
    )
    pp_alarm = (
        p0_alarm
        if pp_density is p0_density
        else select_alarm_prefix(pp_density, budget_km2=ALARM_BUDGET_KM2)
    )
    p0 = ScienceModelForecast(
        model_id="P0",
        spatial_density=p0_density,
        alarm=p0_alarm,
        component_event_ids=tuple(event.id for event in safe_windows.p0_events),
    )
    p1 = ScienceModelForecast(
        model_id="P1",
        spatial_density=p1_density,
        alarm=p1_alarm,
        component_event_ids=tuple(event.id for event in safe_windows.r30_events),
    )
    pp = ScienceModelForecast(
        model_id="PP",
        spatial_density=pp_density,
        alarm=pp_alarm,
        component_event_ids=tuple(event.id for event in safe_windows.rp30_events),
    )
    return ScienceForecastBundle(windows=safe_windows, p0=p0, p1=p1, pp=pp)


def build_science_forecast(
    events: Iterable[SyntheticEvent],
    *,
    issue_time: datetime,
    query_cutoff: datetime,
    training_start: datetime,
    grid_family: SpatialQuadratureFamily,
) -> ScienceForecastBundle:
    """Select causal windows and build all three forecasts in one call."""

    windows = select_causal_windows(
        events,
        issue_time=issue_time,
        query_cutoff=query_cutoff,
        training_start=training_start,
    )
    return build_forecast_from_windows(windows, grid_family)


def alarm_area_spread_km2(bundle: ScienceForecastBundle) -> float:
    """Return the largest pairwise actual-area difference."""

    areas = tuple(model.alarm.actual_area_km2 for model in bundle.models)
    return math.fsum((max(areas), -min(areas)))


__all__ = [
    "ALARM_BUDGET_KM2",
    "FROZEN_BANDWIDTH_KM",
    "FROZEN_MIXTURE_WEIGHT",
    "MAXIMUM_COMPLETE_CELL_AREA_KM2",
    "MAXIMUM_PAIRWISE_ALARM_AREA_DIFFERENCE_KM2",
    "ScienceForecastBundle",
    "ScienceModelForecast",
    "ScienceModelId",
    "alarm_area_spread_km2",
    "build_forecast_from_windows",
    "build_science_forecast",
]
