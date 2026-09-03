"""Pure targets for complete, authorized S3-A development windows.

The caller supplies the already bounded catalog and its positional mapping to
the independently frozen grid.  This module reads no files and constructs no
grid from target locations.  Inner/outer calendar membership remains the
caller's responsibility; the supplied availability cutoff is applied to every
label.  Episode anchors are prepared once from the full bounded history, not
recreated inside each forecast window.
"""

from __future__ import annotations

from collections.abc import Mapping, Set
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Final, TypeVar, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from seismoflux.multitask_s0 import CATALOG_COLUMNS, build_episodes
from seismoflux.multitask_s3.calendar import HORIZONS, REPORT_END, REPORT_START

IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
ArrayScalar = TypeVar("ArrayScalar", bound=np.generic)
FORMAL_BANDS: Final = ("Ms5_6", "Ms6_plus")
_TARGET_COLUMNS: Final = (
    "event_id",
    "origin_time_utc",
    "available_at",
    "magnitude",
    "inside_study_area",
)


class UnevaluableTargetWindowError(ValueError):
    """A complete legal target cannot be formed; do not substitute zero counts."""


def _readonly(values: NDArray[ArrayScalar]) -> NDArray[ArrayScalar]:
    result = np.array(values, copy=True, order="C")
    result.setflags(write=False)
    return cast(NDArray[ArrayScalar], result)


def _aware_utc(value: datetime, *, label: str) -> datetime:
    if (
        not isinstance(value, datetime)
        or pd.isna(value)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(f"{label} must be a finite timezone-aware datetime")
    return value.astimezone(UTC)


def _aware_column(values: pd.Series, *, label: str) -> pd.DatetimeIndex:
    if values.isna().any():
        raise ValueError(f"{label} cannot contain missing timestamps")
    if not isinstance(values.dtype, pd.DatetimeTZDtype):
        # Never silently interpret naive catalog values as UTC.  The canonical
        # reader already gives a timezone-aware dtype; this also handles small
        # injected object arrays with different aware timezones.
        for value in values:
            _aware_utc(value, label=label)
    return pd.DatetimeIndex(pd.to_datetime(values, utc=True, errors="raise"))


def _catalog_arrays(
    frame: pd.DataFrame,
) -> tuple[tuple[str, ...], pd.DatetimeIndex, pd.DatetimeIndex, NDArray[np.float64], BoolArray]:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if frame.columns.duplicated().any():
        raise ValueError("catalog columns must be unique")
    missing = [name for name in _TARGET_COLUMNS if name not in frame]
    if missing:
        raise ValueError(f"target catalog is missing columns: {missing}")
    identifiers = frame["event_id"].astype("string")
    if (
        identifiers.isna().any()
        or (identifiers.str.len() == 0).any()
        or identifiers.duplicated().any()
    ):
        raise ValueError("event_id values must be non-empty and unique")
    origins = _aware_column(frame["origin_time_utc"], label="origin_time_utc")
    available = _aware_column(frame["available_at"], label="available_at")
    if (available < origins).any():
        raise ValueError("available_at cannot precede origin_time_utc")
    magnitude = np.asarray(pd.to_numeric(frame["magnitude"], errors="raise"), dtype=np.float64)
    if not np.isfinite(magnitude).all():
        raise ValueError("magnitude must contain finite raw values")
    inside_values = frame["inside_study_area"]
    if inside_values.isna().any() or any(
        not isinstance(value, bool | np.bool_) for value in inside_values
    ):
        raise ValueError("inside_study_area must contain non-missing booleans")
    return (
        tuple(str(value) for value in identifiers),
        origins,
        available,
        magnitude,
        np.asarray(inside_values, dtype=np.bool_),
    )


@dataclass(frozen=True, slots=True)
class S3BandTargets:
    """Formal-band events in input-row order, with full-history anchor flags."""

    event_ids: tuple[str, ...]
    cell_indices: IntArray
    anchor_mask: BoolArray

    @property
    def event_count(self) -> int:
        return len(self.event_ids)

    @property
    def anchor_count(self) -> int:
        return int(np.count_nonzero(self.anchor_mask))


@dataclass(frozen=True, slots=True)
class S3WindowTargets:
    """One actual complete exposure; a legal zero-event window stays present."""

    issue_time_utc: datetime
    target_end_utc: datetime
    available_by_utc: datetime
    horizon_days: int
    spatial_counts_ms4: IntArray
    count_ms4plus: int
    count_ms5plus: int
    bands: Mapping[str, S3BandTargets]


def prepare_anchor_ids(frame: pd.DataFrame) -> dict[str, set[str]]:
    """Reuse S0 fixed-first 30-day/75-km anchors separately for the two bands.

    Supply the complete authorized historical frame from the bounded catalog
    reader, including events before the A report period.  No issue, fold, or
    availability trimming is performed here.  This matches the existing S3
    input waterlevel: later-window members do not become new anchors merely
    because their actual anchor lies before a forecast window.
    """
    identifiers, origins, available, magnitude, inside = _catalog_arrays(frame)
    missing = [name for name in CATALOG_COLUMNS if name not in frame]
    if missing:
        raise ValueError(f"episode catalog is missing columns: {missing}")
    normalized = frame.loc[:, list(CATALOG_COLUMNS)].copy()
    normalized["event_id"] = identifiers
    normalized["origin_time_utc"] = origins
    normalized["available_at"] = available
    normalized["magnitude"] = magnitude
    result: dict[str, set[str]] = {}
    for name in FORMAL_BANDS:
        selected = inside & (magnitude >= (5.0 if name == "Ms5_6" else 6.0))
        if name == "Ms5_6":
            selected &= magnitude < 6.0
        episodes = build_episodes(normalized.loc[selected], max_time_days=30, max_distance_km=75.0)
        result[name] = {str(episode["anchor_event_id"]) for episode in episodes}
    return result


def build_window_targets(
    frame: pd.DataFrame,
    *,
    issue_time: datetime,
    horizon_days: int,
    available_by: datetime,
    cell_indices: IntArray,
    cell_count: int,
    anchor_ids_by_band: Mapping[str, set[str]],
) -> S3WindowTargets:
    """Select ``(T,T+h]`` labels whose canonical availability is ``<= cutoff``.

    ``cell_indices[k]`` belongs to ``frame.iloc[k]``; index labels and temporal
    sorting never change this association.  ``-1`` may describe an irrelevant
    row but is an explicit error for any selected in-country Ms4+ event.  The
    function never drops an unmapped target, manufactures a shorter exposure,
    or substitutes a zero target for an immature/unauthorized window.
    """
    if isinstance(horizon_days, bool) or not isinstance(horizon_days, int | np.integer):
        raise ValueError("horizon_days must be one of the five registered integer horizons")
    if horizon_days not in HORIZONS:
        raise ValueError("horizon_days must be one of the five registered integer horizons")
    issue = _aware_utc(issue_time, label="issue_time")
    cutoff = _aware_utc(available_by, label="available_by")
    end = issue + timedelta(days=int(horizon_days))
    if not REPORT_START <= issue < end < REPORT_END:
        raise UnevaluableTargetWindowError("window is outside S3-A development authorization")
    if end > cutoff:
        raise UnevaluableTargetWindowError("complete target window is not mature by available_by")
    if (
        isinstance(cell_count, bool)
        or not isinstance(cell_count, int | np.integer)
        or cell_count <= 0
    ):
        raise ValueError("cell_count must be a positive integer from the frozen grid")
    identifiers, origins, available, magnitude, inside = _catalog_arrays(frame)
    positions = np.asarray(cell_indices)
    if positions.shape != (len(frame),) or not np.issubdtype(positions.dtype, np.integer):
        raise ValueError(
            "cell_indices must be a one-dimensional integer array aligned to frame rows"
        )
    if np.any(positions < -1) or np.any(positions >= cell_count):
        raise ValueError("cell_indices must be -1 or a valid frozen-grid position")
    if set(anchor_ids_by_band) != set(FORMAL_BANDS):
        raise ValueError("anchor_ids_by_band must contain exactly Ms5_6 and Ms6_plus")
    for anchor_ids in anchor_ids_by_band.values():
        if not isinstance(anchor_ids, Set) or any(
            not isinstance(value, str) or not value for value in anchor_ids
        ):
            raise ValueError("each anchor band must contain a set of non-empty event identifiers")
    if anchor_ids_by_band["Ms5_6"] & anchor_ids_by_band["Ms6_plus"]:
        raise ValueError("one event cannot be an anchor in both disjoint magnitude bands")

    selected = (
        (origins > issue) & (origins <= end) & (available <= cutoff) & inside & (magnitude >= 4.0)
    )
    target_cells = np.asarray(positions[selected], dtype=np.int64)
    unmapped_count = int(np.count_nonzero(target_cells == -1))
    if unmapped_count:
        raise UnevaluableTargetWindowError(
            f"{unmapped_count} eligible Ms4+ targets lack a frozen-grid cell; do not drop them"
        )
    spatial_counts = np.bincount(target_cells, minlength=int(cell_count)).astype(np.int64)
    bands: dict[str, S3BandTargets] = {}
    for name in FORMAL_BANDS:
        band_selected = selected & (magnitude >= (5.0 if name == "Ms5_6" else 6.0))
        if name == "Ms5_6":
            band_selected &= magnitude < 6.0
        rows = np.flatnonzero(band_selected)
        event_ids = tuple(identifiers[int(index)] for index in rows)
        anchors = np.array(
            [event_id in anchor_ids_by_band[name] for event_id in event_ids], dtype=np.bool_
        )
        bands[name] = S3BandTargets(
            event_ids,
            _readonly(np.asarray(positions[rows], dtype=np.int64)),
            _readonly(anchors),
        )
    return S3WindowTargets(
        issue_time_utc=issue,
        target_end_utc=end,
        available_by_utc=cutoff,
        horizon_days=int(horizon_days),
        spatial_counts_ms4=_readonly(spatial_counts),
        count_ms4plus=int(np.count_nonzero(selected)),
        count_ms5plus=sum(band.event_count for band in bands.values()),
        bands=MappingProxyType(bands),
    )
