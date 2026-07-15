"""Causal global and local reporting-coverage proxies for stage 3."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
from numpy.typing import NDArray

from seismoflux.features.anomaly.nulls import NullReasonCode
from seismoflux.features.anomaly.snapshot import Stage3IssueSnapshot
from seismoflux.features.anomaly.spatial import SpatialEntityArrays, SpatialFeatureResult

TRAILING_REPORTING_REFERENCE_DAYS = 364
KNOWN_MISSING_REPORT_AVAILABLE_AT_UTC: tuple[datetime, ...] = (
    datetime(2024, 9, 4, 16, tzinfo=UTC),
    datetime(2025, 10, 29, 16, tzinfo=UTC),
)


@dataclass(frozen=True, slots=True)
class CoverageEntityBatch:
    """Coordinate-valid raw reporting entities from one actual issue."""

    issue_time_utc: datetime
    xy_m: NDArray[np.float64]
    station_id: NDArray[np.object_]
    measurement_id: NDArray[np.object_]
    discipline_code: NDArray[np.int64]
    discipline_membership: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class LocalCoverageFeatures:
    """Local current/trailing counts and ratios shaped ``(query, scale)``."""

    radius: dict[str, NDArray[np.float64]]
    gaussian: dict[str, NDArray[np.float64]]
    radius_null_reason: dict[str, NDArray[np.int8]]
    gaussian_null_reason: dict[str, NDArray[np.int8]]


def coverage_batch_from_spatial_arrays(
    issue_time_utc: datetime,
    arrays: SpatialEntityArrays,
) -> CoverageEntityBatch:
    """Extract the raw reporting rows used by the causal 364-day reference."""

    xy_m = np.asarray(arrays.xy_m, dtype=np.float64)
    reporting = np.asarray(arrays.reporting_listed, dtype=np.bool_)
    if reporting.shape != (xy_m.shape[0],):
        raise ValueError("reporting_listed must be explicit for coverage replay")
    mask = reporting & np.isfinite(xy_m).all(axis=1)
    membership = np.asarray(arrays.discipline_membership, dtype=np.bool_)
    return CoverageEntityBatch(
        issue_time_utc=issue_time_utc,
        xy_m=np.asarray(xy_m[mask], dtype=np.float64),
        station_id=np.asarray(np.asarray(arrays.station_id, dtype=object)[mask], dtype=object),
        measurement_id=np.asarray(
            np.asarray(arrays.measurement_id, dtype=object)[mask],
            dtype=object,
        ),
        discipline_code=np.asarray(
            np.asarray(arrays.discipline_code, dtype=np.int64)[mask],
            dtype=np.int64,
        ),
        discipline_membership=np.asarray(membership[mask], dtype=np.bool_),
    )


def select_trailing_coverage_batches(
    batches: tuple[CoverageEntityBatch, ...],
    issue_time_utc: datetime,
) -> tuple[CoverageEntityBatch, ...]:
    """Select the closed causal interval ``[T-364 days, T]``."""

    lower = issue_time_utc - timedelta(days=TRAILING_REPORTING_REFERENCE_DAYS)
    selected = tuple(batch for batch in batches if lower <= batch.issue_time_utc <= issue_time_utc)
    if any(batch.issue_time_utc > issue_time_utc for batch in selected):
        raise AssertionError("future reporting batch entered a causal trailing window")
    return selected


def trailing_coverage_entity_arrays(
    batches: tuple[CoverageEntityBatch, ...],
) -> SpatialEntityArrays:
    """Build raw reporting-proxy arrays; longitudinal entity statuses remain false."""

    size = sum(batch.xy_m.shape[0] for batch in batches)
    if batches:
        xy_m = np.concatenate([batch.xy_m for batch in batches], axis=0)
        station_id = np.concatenate([batch.station_id for batch in batches])
        measurement_id = np.concatenate([batch.measurement_id for batch in batches])
        discipline_code = np.concatenate([batch.discipline_code for batch in batches])
        discipline_membership = np.concatenate(
            [batch.discipline_membership for batch in batches],
            axis=0,
        )
    else:
        xy_m = np.empty((0, 2), dtype=np.float64)
        station_id = np.empty(0, dtype=object)
        measurement_id = np.empty(0, dtype=object)
        discipline_code = np.empty(0, dtype=np.int64)
        discipline_membership = np.empty((0, 4), dtype=np.bool_)
    false = np.zeros(size, dtype=np.bool_)
    return SpatialEntityArrays(
        xy_m=xy_m,
        listed=false,
        source_new=false,
        first_seen=false,
        explicit_end=false,
        not_continued=false,
        relisted=false,
        right_censored=false,
        reliability_high=false,
        reliability_cautious=false,
        station_id=station_id,
        measurement_id=measurement_id,
        discipline_code=discipline_code,
        age_days=np.full(size, np.nan, dtype=np.float64),
        known_duration_days=np.full(size, np.nan, dtype=np.float64),
        discipline_membership=discipline_membership,
        reporting_listed=np.ones(size, dtype=np.bool_),
        coverage_unique_only=True,
    )


def _coverage_family(
    current: dict[str, NDArray[np.float64]],
    trailing: dict[str, NDArray[np.float64]],
) -> tuple[dict[str, NDArray[np.float64]], dict[str, NDArray[np.int8]]]:
    current_station = current["unique_station_count"]
    current_measurement = current["unique_measurement_count"]
    trailing_station = trailing["unique_station_count"]
    trailing_measurement = trailing["unique_measurement_count"]
    output = {
        "trailing_station_count_reporting_coverage_proxy": trailing_station.copy(),
        "trailing_measurement_count_reporting_coverage_proxy": trailing_measurement.copy(),
        "current_to_trailing_station_reporting_coverage_proxy": np.full(
            trailing_station.shape,
            np.nan,
            dtype=np.float64,
        ),
        "current_to_trailing_measurement_reporting_coverage_proxy": np.full(
            trailing_measurement.shape,
            np.nan,
            dtype=np.float64,
        ),
    }
    reason = {
        key: np.full(value.shape, int(NullReasonCode.VALID), dtype=np.int8)
        for key, value in output.items()
    }
    station_valid = trailing_station > 0.0
    measurement_valid = trailing_measurement > 0.0
    station_name = "current_to_trailing_station_reporting_coverage_proxy"
    measurement_name = "current_to_trailing_measurement_reporting_coverage_proxy"
    output[station_name][station_valid] = (
        current_station[station_valid] / trailing_station[station_valid]
    )
    output[measurement_name][measurement_valid] = (
        current_measurement[measurement_valid] / trailing_measurement[measurement_valid]
    )
    reason[station_name][~station_valid] = int(NullReasonCode.ZERO_DENOMINATOR)
    reason[measurement_name][~measurement_valid] = int(NullReasonCode.ZERO_DENOMINATOR)
    return output, reason


def local_coverage_features(
    current: SpatialFeatureResult,
    trailing: SpatialFeatureResult,
) -> LocalCoverageFeatures:
    """Combine current and causal trailing spatial aggregations without global masking."""

    radius, radius_reason = _coverage_family(
        current.radius_features,
        trailing.radius_features,
    )
    gaussian, gaussian_reason = _coverage_family(
        current.gaussian_features,
        trailing.gaussian_features,
    )
    return LocalCoverageFeatures(
        radius=radius,
        gaussian=gaussian,
        radius_null_reason=radius_reason,
        gaussian_null_reason=gaussian_reason,
    )


def global_reporting_coverage_proxy(
    snapshots: tuple[Stage3IssueSnapshot, ...],
    issue_index: int,
) -> dict[str, object]:
    """Return report-level proxies that are repeated across local query rows."""

    snapshot = snapshots[issue_index]
    summary = snapshot.summary
    if issue_index == 0:
        days_since_previous: int | None = None
        days_reason = int(NullReasonCode.INSUFFICIENT_ACTUAL_SNAPSHOTS)
    else:
        delta = snapshot.issue_time_utc - snapshots[issue_index - 1].issue_time_utc
        days_since_previous = delta.days
        days_reason = int(NullReasonCode.VALID)
    output: dict[str, object] = {
        "report_present_reporting_coverage_proxy": True,
        "report_row_count_reporting_coverage_proxy": summary.row_count,
        "days_since_previous_actual_report_reporting_coverage_proxy": days_since_previous,
        "days_since_previous_actual_report_reporting_coverage_proxy__null_reason_code": (
            days_reason
        ),
    }
    for weeks in (4, 8, 13, 26, 52):
        lower = snapshot.issue_time_utc - timedelta(weeks=weeks)
        count = sum(
            lower < missing_time <= snapshot.issue_time_utc
            for missing_time in KNOWN_MISSING_REPORT_AVAILABLE_AT_UTC
        )
        output[f"missing_expected_period_count_{weeks}w_reporting_coverage_proxy"] = count
    return output


__all__ = [
    "KNOWN_MISSING_REPORT_AVAILABLE_AT_UTC",
    "TRAILING_REPORTING_REFERENCE_DAYS",
    "CoverageEntityBatch",
    "LocalCoverageFeatures",
    "coverage_batch_from_spatial_arrays",
    "global_reporting_coverage_proxy",
    "local_coverage_features",
    "select_trailing_coverage_batches",
    "trailing_coverage_entity_arrays",
]
