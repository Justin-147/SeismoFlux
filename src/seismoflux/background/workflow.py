"""Pure stage-2 snapshot construction and background-model workflow helpers."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import numpy as np
from numpy.typing import NDArray

from seismoflux.background.catalog import EarthquakeCatalog, utc_timestamp_to_day
from seismoflux.background.completeness import (
    CATALOG_ANCHOR_UTC,
    CompletenessAnalysis,
    CompletenessEvent,
    analyze_catalog_completeness,
)
from seismoflux.background.config import BackgroundConfig
from seismoflux.background.etas_fit import ETASEvent

ProgressCallback = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class SnapshotDefinition:
    """One frozen fit cutoff and its subsequent assessment interval."""

    snapshot_id: str
    fit_end_utc: str
    fit_end_day: float
    assessment_start_utc: str
    assessment_start_day: float
    assessment_end_utc: str
    assessment_end_day: float
    optimizer_model_id: str
    is_validation: bool

    def __post_init__(self) -> None:
        values = (self.fit_end_day, self.assessment_start_day, self.assessment_end_day)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("snapshot days must be finite")
        if not self.fit_end_day < self.assessment_start_day < self.assessment_end_day:
            raise ValueError(
                "snapshot dates must satisfy fit_end < assessment_start < assessment_end"
            )
        if self.assessment_start_day - self.fit_end_day < 365.0:
            raise ValueError("snapshot must retain at least the frozen 365-day purge")

    @property
    def assessment_duration_days(self) -> float:
        return self.assessment_end_day - self.assessment_start_day


def _validation_local_to_utc_day(value: str) -> tuple[str, float]:
    timestamp = datetime.fromisoformat(value).replace(tzinfo=None)
    local = timestamp.replace(tzinfo=ZoneInfo("Asia/Shanghai"))
    utc_value = local.astimezone(UTC)
    text = utc_value.isoformat().replace("+00:00", "Z")
    return text, utc_timestamp_to_day(text)


def build_snapshot_definitions(config: BackgroundConfig) -> tuple[SnapshotDefinition, ...]:
    """Materialize the four historical folds plus the final validation snapshot."""

    snapshots: list[SnapshotDefinition] = []
    for fold in config.parameter_selection_folds.folds:
        snapshots.append(
            SnapshotDefinition(
                snapshot_id=fold.id,
                fit_end_utc=fold.fit_end_utc,
                fit_end_day=utc_timestamp_to_day(fold.fit_end_utc),
                assessment_start_utc=fold.assessment_start_utc,
                assessment_start_day=utc_timestamp_to_day(fold.assessment_start_utc),
                assessment_end_utc=fold.assessment_end_utc,
                assessment_end_day=utc_timestamp_to_day(fold.assessment_end_utc),
                optimizer_model_id=f"etas/{fold.id}",
                is_validation=False,
            )
        )
    validation_start_utc, validation_start_day = _validation_local_to_utc_day(
        f"{config.time.validation_start_local}T00:00:00"
    )
    validation_end_utc, validation_end_day = _validation_local_to_utc_day(
        f"{config.time.validation_end_local}T00:00:00"
    )
    snapshots.append(
        SnapshotDefinition(
            snapshot_id="final_validation",
            fit_end_utc=config.time.final_parameter_fit_end_utc,
            fit_end_day=utc_timestamp_to_day(config.time.final_parameter_fit_end_utc),
            assessment_start_utc=validation_start_utc,
            assessment_start_day=validation_start_day,
            assessment_end_utc=validation_end_utc,
            assessment_end_day=validation_end_day,
            optimizer_model_id="etas/final_validation",
            is_validation=True,
        )
    )
    return tuple(snapshots)


def _day_to_utc_datetime(day: float) -> datetime:
    return datetime.fromtimestamp(day * 86_400.0, tz=UTC)


def catalog_completeness_events(catalog: EarthquakeCatalog) -> tuple[CompletenessEvent, ...]:
    """Adapt only the frozen catalog allowlist to the completeness API."""

    return tuple(
        CompletenessEvent(
            event_id=str(catalog.event_id[index]),
            origin_time_utc=_day_to_utc_datetime(float(catalog.origin_day[index])),
            available_at=_day_to_utc_datetime(float(catalog.available_day[index])),
            magnitude=float(catalog.magnitude[index]),
            inside_study_area=bool(catalog.inside_study_area[index]),
            x_m=float(catalog.x_km[index]) * 1_000.0,
            y_m=float(catalog.y_km[index]) * 1_000.0,
        )
        for index in range(len(catalog))
    )


@dataclass(frozen=True, slots=True)
class CompletenessSnapshot:
    definition: SnapshotDefinition
    analysis: CompletenessAnalysis


def analyze_snapshot_completeness(
    catalog: EarthquakeCatalog,
    snapshots: tuple[SnapshotDefinition, ...],
    *,
    progress: ProgressCallback | None = None,
) -> tuple[CompletenessSnapshot, ...]:
    """Estimate completeness independently at every fit cutoff."""

    events = catalog_completeness_events(catalog)
    results: list[CompletenessSnapshot] = []
    for snapshot in snapshots:
        if progress is not None:
            progress(f"completeness:{snapshot.snapshot_id}:start")
        analysis = analyze_catalog_completeness(
            events,
            cutoff_utc=_day_to_utc_datetime(snapshot.fit_end_day),
        )
        results.append(CompletenessSnapshot(definition=snapshot, analysis=analysis))
        if progress is not None:
            progress(f"completeness:{snapshot.snapshot_id}:done:Mc={analysis.selected_mc:g}")
    return tuple(results)


def event_mask(
    catalog: EarthquakeCatalog,
    *,
    minimum_magnitude: float,
    origin_after_day: float | None = None,
    origin_through_day: float,
    available_through_day: float,
    spatial_domain: str,
) -> NDArray[np.bool_]:
    """Build one causal event mask without inspecting any disallowed column."""

    if spatial_domain == "inside":
        spatial = catalog.inside_study_area
    elif spatial_domain == "parent_buffer":
        spatial = catalog.inside_external_buffer
    else:
        raise ValueError("spatial_domain must be inside or parent_buffer")
    mask = (
        (catalog.magnitude >= float(minimum_magnitude))
        & (catalog.origin_day <= float(origin_through_day))
        & (catalog.available_day <= float(available_through_day))
        & spatial
    )
    if origin_after_day is not None:
        mask &= catalog.origin_day > float(origin_after_day)
    return np.asarray(mask, dtype=np.bool_)


def physical_target_mask(
    catalog: EarthquakeCatalog,
    *,
    minimum_magnitude: float,
    origin_after_day: float,
    origin_through_day: float,
) -> NDArray[np.bool_]:
    """Select matured physical labels by origin without using publication as a feature."""

    minimum = float(minimum_magnitude)
    start = float(origin_after_day)
    end = float(origin_through_day)
    if not all(math.isfinite(value) for value in (minimum, start, end)):
        raise ValueError("physical target thresholds must be finite")
    if minimum <= 0.0 or not start < end:
        raise ValueError("physical target magnitude and interval must be positive")
    return np.asarray(
        catalog.inside_study_area
        & (catalog.magnitude >= minimum)
        & (catalog.origin_day > start)
        & (catalog.origin_day <= end),
        dtype=np.bool_,
    )


def catalog_etas_events(
    catalog: EarthquakeCatalog,
    mask: NDArray[np.bool_],
    *,
    time_origin_day: float = 0.0,
    publication_delay_days: float = 0.0,
) -> tuple[ETASEvent, ...]:
    """Adapt catalog columns to ETAS events with separate origin and availability."""

    boolean_mask = np.asarray(mask, dtype=np.bool_)
    if boolean_mask.shape != (len(catalog),):
        raise ValueError("ETAS event mask has the wrong shape")
    origin = float(time_origin_day)
    delay = float(publication_delay_days)
    if not math.isfinite(origin) or not math.isfinite(delay) or delay < 0.0:
        raise ValueError("ETAS time origin and publication delay must be finite and non-negative")
    indices = np.flatnonzero(boolean_mask)
    return tuple(
        ETASEvent(
            event_id=str(catalog.event_id[index]),
            time_days=float(catalog.origin_day[index]) - origin,
            available_time_days=float(catalog.available_day[index]) + delay - origin,
            x_km=float(catalog.x_km[index]),
            y_km=float(catalog.y_km[index]),
            magnitude=float(catalog.magnitude[index]),
            inside_study_area=bool(catalog.inside_study_area[index]),
            inside_parent_domain=bool(catalog.inside_external_buffer[index]),
        )
        for index in indices
    )


def historical_training_mask(
    catalog: EarthquakeCatalog,
    *,
    minimum_magnitude: float,
    fit_end_day: float,
    start_utc: datetime = CATALOG_ANCHOR_UTC,
) -> NDArray[np.bool_]:
    """Select inside training events in ``[start, fit_end]`` known by the cutoff."""

    start_day = start_utc.timestamp() / 86_400.0
    mask = event_mask(
        catalog,
        minimum_magnitude=minimum_magnitude,
        origin_through_day=fit_end_day,
        available_through_day=fit_end_day,
        spatial_domain="inside",
    )
    mask &= catalog.origin_day >= start_day
    return np.asarray(mask, dtype=np.bool_)


__all__ = [
    "CompletenessSnapshot",
    "ProgressCallback",
    "SnapshotDefinition",
    "analyze_snapshot_completeness",
    "build_snapshot_definitions",
    "catalog_completeness_events",
    "catalog_etas_events",
    "event_mask",
    "historical_training_mask",
    "physical_target_mask",
]
