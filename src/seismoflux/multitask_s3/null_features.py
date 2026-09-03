"""Frozen S3 offline null-world feature reconstruction; no fitting or scoring.

Time donors may be later within their registered partition. These arrays must
never be used as causal forecasts. Coverage stays with the recipient report.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from typing import Any

import numpy as np

from seismoflux.features.anomaly.trajectory import compute_trajectory_features
from seismoflux.multitask_s3.calendar import (
    FOLDS,
    HORIZONS,
    REPORT_START,
    time_null_partitions,
)

SNAPSHOT_INDICES = (0, 1, 2, 3, 4, 5, 6, 7, 11)
SNAPSHOT_AND_MASK_INDICES = (*SNAPSHOT_INDICES, 16, 18)
DYNAMIC_INDICES = (8, 9, 10)
FIXED_COVERAGE_INDICES = (12, 13, 14, 15, 19)
RADIUS_BASE_COLUMNS = ("radius_200km__listed_count", "radius_200km__first_seen_count")


def _issue_times(values: Sequence[datetime]) -> tuple[datetime, ...]:
    if not values or any(
        not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None
        for value in values
    ):
        raise ValueError("null reconstruction requires aware actual issue timestamps")
    times = tuple(value.astimezone(UTC) for value in values)
    if any(right <= left for left, right in pairwise(times)):
        raise ValueError("null issue timestamps must be strictly increasing")
    return times


def rebuild_dynamic_values(
    issue_times_utc: Sequence[datetime], radius_bases: np.ndarray, *, cell_chunk_size: int = 512
) -> np.ndarray:
    """Reuse the historical formulas on actual pseudo-prefixes and raw counts.

    Returns transformed slope(listed), acceleration(listed), slope(first_seen).
    Chunking cells changes memory use only, not the temporal window or formulas.
    """
    times = _issue_times(issue_times_utc)
    bases = np.asarray(radius_bases, dtype=np.float64)
    if (
        bases.ndim != 3
        or bases.shape[0] != len(times)
        or bases.shape[1] == 0
        or bases.shape[2] != 2
        or np.isinf(bases).any()
        or np.any(bases < 0)
    ):
        raise ValueError("raw radius bases must have shape (issues, cells, 2), nonnegative or NaN")
    if (
        not isinstance(cell_chunk_size, int)
        or isinstance(cell_chunk_size, bool)
        or cell_chunk_size < 1
    ):
        raise ValueError("cell chunk size must be a positive integer")
    # Convert datetime objects explicitly, never reinterpret epoch integers as ns.
    time_axis = np.array([time.replace(tzinfo=None) for time in times], dtype="datetime64[ns]")
    dynamic = np.empty((*bases.shape[:2], 3), dtype=np.float64)
    for first in range(0, bases.shape[1], cell_chunk_size):
        last = min(first + cell_chunk_size, bases.shape[1])
        width = last - first
        result = compute_trajectory_features(
            time_axis, bases[:, first:last].reshape(len(times), -1)
        )
        for output_index, (feature_name, series_index) in enumerate(
            (
                ("slope_13w_per_week", 0),
                ("acceleration_4v13_per_week2", 0),
                ("slope_13w_per_week", 1),
            )
        ):
            values = result.features[feature_name].reshape(len(times), width, 2)[:, :, series_index]
            valid = result.valid_masks[feature_name].reshape(len(times), width, 2)[
                :, :, series_index
            ]
            dynamic[:, first:last, output_index] = np.arcsinh(np.where(valid, values, np.nan))
    dynamic.setflags(write=False)
    return dynamic


@dataclass(frozen=True, slots=True)
class S3TimeNullFeatures:
    issue_times_utc: tuple[datetime, ...]
    features: np.ndarray
    radius_bases: np.ndarray
    donor_indices: np.ndarray
    diagnostics: dict[str, Any]


def permute_time_features(
    *,
    issue_times_utc: Sequence[datetime],
    features: np.ndarray,
    radius_bases: np.ndarray,
    fold_id: str,
    horizon_days: int,
    truth_cutoff: datetime,
    rng: np.random.Generator,
    cell_chunk_size: int = 512,
) -> S3TimeNullFeatures:
    """Jointly move nine snapshots/two bases, then reconstruct three dynamics.

    Each source block is a bijection, including identity for singleton pools.
    Snapshot missing controls travel with donors; coverage and its mask do not.
    The caller must refit all training preprocessing and model selection later.
    """
    times = _issue_times(issue_times_utc)
    if fold_id not in FOLDS or horizon_days not in HORIZONS:
        raise ValueError("unregistered S3 null fold or horizon")
    if any(not REPORT_START <= time < FOLDS[fold_id][1] for time in times):
        raise ValueError("null inputs contain reports outside this fold's authorized pool")
    original = np.asarray(features, dtype=np.float64)
    bases = np.asarray(radius_bases, dtype=np.float64)
    if (
        original.ndim != 3
        or original.shape[0] != len(times)
        or original.shape[1] == 0
        or original.shape[2] != 20
        or np.isinf(original).any()
        or bases.shape != (*original.shape[:2], 2)
    ):
        raise ValueError("null values require aligned 20-column features and two raw bases")
    partitions = time_null_partitions(
        times, fold_id=fold_id, horizon_days=horizon_days, truth_cutoff=truth_cutoff
    )
    by_time = {time: index for index, time in enumerate(times)}
    donor = np.arange(len(times), dtype=np.int64)
    pool_sizes = []
    for partition in partitions:
        indices = np.array([by_time[time] for time in partition], dtype=np.int64)
        donor[indices] = rng.permutation(indices)
        pool_sizes.append(len(indices))
    shuffled = original.copy()
    for column in SNAPSHOT_AND_MASK_INDICES:
        shuffled[:, :, column] = original[donor, :, column]
    pseudo_bases = bases[donor].copy()
    dynamic = rebuild_dynamic_values(times, pseudo_bases, cell_chunk_size=cell_chunk_size)
    shuffled[:, :, DYNAMIC_INDICES] = dynamic
    shuffled[:, :, 17] = np.isnan(dynamic).mean(axis=2)
    moved = int(np.count_nonzero(donor != np.arange(len(times))))
    diagnostics = {
        "role": "offline_time_counterfactual_not_causal_prediction",
        "fold_id": fold_id,
        "horizon_days": horizon_days,
        "report_count": len(times),
        "pool_sizes": pool_sizes,
        "singleton_report_count": sum(size == 1 for size in pool_sizes),
        "permutable_report_fraction": sum(size for size in pool_sizes if size > 1) / len(times),
        "moved_report_count": moved,
        "effective_permutation_fraction": moved / len(times),
        "later_donor_count": int(np.count_nonzero(donor > np.arange(len(times)))),
        "coverage_kept_at_recipient": True,
        "dynamic_rebuilt_from_pseudo_prefix": True,
        "models_refitted": False,
    }
    for array in (shuffled, pseudo_bases, donor):
        array.setflags(write=False)
    return S3TimeNullFeatures(times, shuffled, pseudo_bases, donor, diagnostics)
