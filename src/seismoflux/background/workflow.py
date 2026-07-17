"""Pure stage-2 snapshot construction and background-model workflow helpers."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
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
_UNIX_EPOCH_UTC = datetime(1970, 1, 1, tzinfo=UTC)


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
    return _UNIX_EPOCH_UTC + timedelta(days=day)


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


def _readonly_bool_mask(
    catalog: EarthquakeCatalog,
    values: NDArray[np.bool_],
    *,
    label: str,
) -> NDArray[np.bool_]:
    result = np.array(values, dtype=np.bool_, copy=True)
    if result.shape != (len(catalog),):
        raise ValueError(f"{label} has the wrong shape")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class ETASParentRoleMasks:
    """Disjoint local-support ETAS parent roles before causal time filtering.

    ``supported_parent`` contains common-Mc events in the retained support.
    ``true_external_buffer_parent`` contains only events outside the original
    study area but inside its frozen 300-km buffer.  ``unsupported_conditional_parent``
    contains only events in unsupported study-area cells that satisfy their
    caller-supplied frozen local Mc.  Keeping the roles separate prevents an
    unsupported study-area event from leaking through the external-buffer path.
    """

    supported_parent: NDArray[np.bool_]
    true_external_buffer_parent: NDArray[np.bool_]
    unsupported_conditional_parent: NDArray[np.bool_]

    def __post_init__(self) -> None:
        arrays = tuple(
            np.array(values, dtype=np.bool_, copy=True)
            for values in (
                self.supported_parent,
                self.true_external_buffer_parent,
                self.unsupported_conditional_parent,
            )
        )
        shapes = {values.shape for values in arrays}
        if len(shapes) != 1 or arrays[0].ndim != 1:
            raise ValueError("ETAS parent role masks must be aligned vectors")
        combined_counts = sum(values.astype(np.uint8, copy=False) for values in arrays)
        if np.any(combined_counts > 1):
            raise ValueError("ETAS parent role masks must be pairwise disjoint")
        for values in arrays:
            values.setflags(write=False)
        object.__setattr__(self, "supported_parent", arrays[0])
        object.__setattr__(self, "true_external_buffer_parent", arrays[1])
        object.__setattr__(self, "unsupported_conditional_parent", arrays[2])

    @property
    def parent_mask(self) -> NDArray[np.bool_]:
        """Return the immutable union of all three parent roles."""

        result = np.asarray(
            self.supported_parent
            | self.true_external_buffer_parent
            | self.unsupported_conditional_parent,
            dtype=np.bool_,
        )
        result.setflags(write=False)
        return result

    def excluding_unsupported_parents(self) -> ETASParentRoleMasks:
        """Return the preregistered complete-exclusion sensitivity roles."""

        return ETASParentRoleMasks(
            supported_parent=self.supported_parent,
            true_external_buffer_parent=self.true_external_buffer_parent,
            unsupported_conditional_parent=np.zeros(
                self.unsupported_conditional_parent.shape,
                dtype=np.bool_,
            ),
        )


def build_local_support_etas_parent_roles(
    catalog: EarthquakeCatalog,
    *,
    supported_domain_mask: NDArray[np.bool_],
    unsupported_domain_mask: NDArray[np.bool_],
    common_mc: float,
    unsupported_local_mc: NDArray[np.float64] | None = None,
    prevalidated_unsupported_parent_mask: NDArray[np.bool_] | None = None,
) -> ETASParentRoleMasks:
    """Build disjoint parent roles for one frozen local-support snapshot.

    Exactly one unsupported-parent eligibility source is required.  Callers may
    provide an aligned local-Mc vector, with finite thresholds on unsupported
    events and NaN elsewhere, or a mask already validated against the frozen
    cell-local thresholds.  Causal origin/availability interval filtering is
    intentionally left to the caller so the same roles can be reused for fit,
    assessment, horizon, and issue-time histories.
    """

    supported = _readonly_bool_mask(
        catalog,
        supported_domain_mask,
        label="supported_domain_mask",
    )
    unsupported = _readonly_bool_mask(
        catalog,
        unsupported_domain_mask,
        label="unsupported_domain_mask",
    )
    if np.any(supported & unsupported):
        raise ValueError("supported and unsupported domain masks must be disjoint")
    inside = np.asarray(catalog.inside_study_area, dtype=np.bool_)
    if not np.array_equal(supported | unsupported, inside):
        raise ValueError(
            "supported and unsupported domain masks must exactly partition "
            "original study-area events"
        )

    minimum = float(common_mc)
    if not math.isfinite(minimum) or minimum <= 0.0:
        raise ValueError("common_mc must be finite and positive")
    if (unsupported_local_mc is None) == (prevalidated_unsupported_parent_mask is None):
        raise ValueError(
            "provide exactly one of unsupported_local_mc or prevalidated_unsupported_parent_mask"
        )

    supported_parent = np.asarray(
        supported & (catalog.magnitude >= minimum),
        dtype=np.bool_,
    )
    true_external_parent = np.asarray(
        ~inside & catalog.inside_external_buffer & (catalog.magnitude >= minimum),
        dtype=np.bool_,
    )
    if unsupported_local_mc is not None:
        local_mc = np.asarray(unsupported_local_mc, dtype=np.float64)
        if local_mc.shape != (len(catalog),):
            raise ValueError("unsupported_local_mc has the wrong shape")
        if np.any(~np.isnan(local_mc[~unsupported])):
            raise ValueError("unsupported_local_mc must be NaN outside unsupported events")
        unsupported_thresholds = local_mc[unsupported]
        if np.any(~np.isfinite(unsupported_thresholds)) or np.any(
            unsupported_thresholds <= minimum
        ):
            raise ValueError("unsupported local Mc values must be finite and above common_mc")
        unsupported_parent = np.asarray(
            unsupported & (catalog.magnitude >= local_mc),
            dtype=np.bool_,
        )
    else:
        if prevalidated_unsupported_parent_mask is None:
            raise AssertionError("unsupported parent eligibility source was not resolved")
        unsupported_parent = _readonly_bool_mask(
            catalog,
            prevalidated_unsupported_parent_mask,
            label="prevalidated_unsupported_parent_mask",
        )
        if np.any(unsupported_parent & ~unsupported):
            raise ValueError(
                "prevalidated unsupported parent mask must be contained in unsupported_domain_mask"
            )
        if np.any(unsupported_parent & (catalog.magnitude < minimum)):
            raise ValueError(
                "prevalidated unsupported parents must also satisfy the common Mc floor"
            )

    return ETASParentRoleMasks(
        supported_parent=supported_parent,
        true_external_buffer_parent=true_external_parent,
        unsupported_conditional_parent=unsupported_parent,
    )


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
    inside_target_domain_mask: NDArray[np.bool_] | None = None,
    inside_parent_domain_mask: NDArray[np.bool_] | None = None,
) -> tuple[ETASEvent, ...]:
    """Adapt catalog rows with optional scoring-target and parent-domain flags.

    Omitting both domain masks preserves the original v0.2.0 adapter byte for
    byte: ``inside_study_area`` and ``inside_external_buffer`` are copied from
    the catalog.  Local-support callers pass the retained target domain and the
    explicit union of supported, true-external, and eligible-unsupported parent
    roles.  The role identities remain in the caller's audit masks rather than
    changing :class:`ETASEvent`, so existing parameter/evidence hashes do not
    acquire a new field.
    """

    boolean_mask = np.asarray(mask, dtype=np.bool_)
    if boolean_mask.shape != (len(catalog),):
        raise ValueError("ETAS event mask has the wrong shape")
    origin = float(time_origin_day)
    delay = float(publication_delay_days)
    if not math.isfinite(origin) or not math.isfinite(delay) or delay < 0.0:
        raise ValueError("ETAS time origin and publication delay must be finite and non-negative")
    custom_domains = inside_target_domain_mask is not None or inside_parent_domain_mask is not None
    target_domain = (
        np.asarray(catalog.inside_study_area, dtype=np.bool_)
        if inside_target_domain_mask is None
        else _readonly_bool_mask(
            catalog,
            inside_target_domain_mask,
            label="inside_target_domain_mask",
        )
    )
    parent_domain = (
        np.asarray(catalog.inside_external_buffer, dtype=np.bool_)
        if inside_parent_domain_mask is None
        else _readonly_bool_mask(
            catalog,
            inside_parent_domain_mask,
            label="inside_parent_domain_mask",
        )
    )
    if custom_domains:
        if np.any(boolean_mask & target_domain & ~parent_domain):
            raise ValueError(
                "selected custom ETAS targets must be contained in their parent domain"
            )
        if np.any(boolean_mask & ~parent_domain):
            raise ValueError("custom ETAS event mask must be contained in its parent domain")
    indices = np.flatnonzero(boolean_mask)
    return tuple(
        ETASEvent(
            event_id=str(catalog.event_id[index]),
            time_days=float(catalog.origin_day[index]) - origin,
            available_time_days=float(catalog.available_day[index]) + delay - origin,
            x_km=float(catalog.x_km[index]),
            y_km=float(catalog.y_km[index]),
            magnitude=float(catalog.magnitude[index]),
            inside_study_area=bool(target_domain[index]),
            inside_parent_domain=bool(parent_domain[index]),
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
    "ETASParentRoleMasks",
    "ProgressCallback",
    "SnapshotDefinition",
    "analyze_snapshot_completeness",
    "build_local_support_etas_parent_roles",
    "build_snapshot_definitions",
    "catalog_completeness_events",
    "catalog_etas_events",
    "event_mask",
    "historical_training_mask",
    "physical_target_mask",
]
