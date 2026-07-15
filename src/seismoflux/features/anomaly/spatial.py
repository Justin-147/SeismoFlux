"""Target-independent spatial aggregation for causal anomaly entity snapshots.

The API intentionally has no earthquake, target-event, completeness-magnitude, or
model-score input.  Missing entity coordinates are retained in the input audit counts
but cannot contribute to a query-location spatial aggregate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_matrix  # type: ignore[import-untyped]
from scipy.spatial import cKDTree  # type: ignore[import-untyped]

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

SPATIAL_SCALES_KM: Final[tuple[float, ...]] = (50.0, 100.0, 200.0, 300.0, 500.0)
GAUSSIAN_TRUNCATION_SIGMA: Final[float] = 3.0
HIGH_RELIABILITY_WEIGHT: Final[float] = 1.0
CAUTIOUS_RELIABILITY_WEIGHT: Final[float] = 0.5

DISCIPLINE_NAMES: Final[tuple[str, ...]] = (
    "deformation",
    "fluid",
    "electromagnetic",
    "cross_fault",
)

_CORE_STATUS_FIELDS: Final[tuple[str, ...]] = (
    "listed",
    "source_new",
    "first_seen",
    "explicit_end",
    "not_continued",
    "relisted",
    "right_censored",
    "left_truncated",
    "late_entry",
    "temporary_entity",
)

_STATUS_INPUT_FIELDS: Final[dict[str, str]] = {
    "temporary_entity": "temporary",
}


@dataclass(frozen=True, slots=True)
class SpatialEntityArrays:
    """Parallel arrays describing entities at one causal issue time.

    ``discipline_code`` uses ``0..3`` in :data:`DISCIPLINE_NAMES` order.  An entity
    may be high or cautious; an entity in neither tier is explicitly excluded with
    reliability weight zero.  The two positive tiers may not overlap.
    ``known_duration_days`` is NaN until an explicit duration is causally known.
    NaN coordinates are permitted and audited as spatially unavailable.
    """

    xy_m: ArrayLike
    listed: ArrayLike
    source_new: ArrayLike
    first_seen: ArrayLike
    explicit_end: ArrayLike
    not_continued: ArrayLike
    relisted: ArrayLike
    right_censored: ArrayLike
    reliability_high: ArrayLike
    reliability_cautious: ArrayLike
    station_id: ArrayLike
    measurement_id: ArrayLike
    discipline_code: ArrayLike
    age_days: ArrayLike
    known_duration_days: ArrayLike
    left_truncated: ArrayLike | None = None
    late_entry: ArrayLike | None = None
    temporary: ArrayLike | None = None
    multidisciplinary: ArrayLike | None = None
    discipline_membership: ArrayLike | None = None
    reporting_listed: ArrayLike | None = None
    coverage_unique_only: bool = False


@dataclass(frozen=True, slots=True)
class SpatialFeatureResult:
    """Spatial features with arrays shaped ``(n_query, n_scale)``."""

    query_xy_m: FloatArray
    scales_km: FloatArray
    radius_features: dict[str, FloatArray]
    gaussian_features: dict[str, FloatArray]
    input_entity_count: int
    spatial_entity_count: int
    missing_coordinate_count: int


@dataclass(frozen=True, slots=True)
class Stage4PlaceboSpatialFeatureResult:
    """Exact 200 km fields needed by the frozen stage-4 spatial placebo."""

    query_xy_m: FloatArray
    scales_km: FloatArray
    radius_features: dict[str, FloatArray]
    gaussian_features: dict[str, FloatArray]
    input_entity_count: int
    spatial_entity_count: int
    missing_coordinate_count: int


@dataclass(frozen=True, slots=True)
class _PreparedEntities:
    xy_m: FloatArray
    statuses: dict[str, BoolArray]
    reliability_high: BoolArray
    reliability_cautious: BoolArray
    station_id: NDArray[np.object_]
    measurement_id: NDArray[np.object_]
    discipline_code: NDArray[np.int64]
    discipline_membership: BoolArray
    age_days: FloatArray
    known_duration_days: FloatArray
    multidisciplinary: BoolArray
    reporting_listed: BoolArray
    station_code: NDArray[np.int64]
    measurement_code: NDArray[np.int64]
    station_measurement_code: NDArray[np.int64]
    input_count: int
    spatial_count: int
    missing_coordinate_count: int
    coverage_unique_only: bool


def _one_dimensional(
    value: ArrayLike,
    *,
    name: str,
    size: int,
    dtype: np.dtype[np.generic] | type[np.generic],
) -> NDArray[np.generic]:
    array = np.asarray(value, dtype=dtype)
    if array.ndim != 1 or array.shape[0] != size:
        raise ValueError(f"{name} must have shape ({size},)")
    return array


def _boolean_array(value: ArrayLike, *, name: str, size: int) -> BoolArray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.shape[0] != size:
        raise ValueError(f"{name} must have shape ({size},)")
    if raw.dtype != np.dtype(np.bool_):
        raise TypeError(f"{name} must be a boolean array")
    return np.asarray(raw, dtype=np.bool_)


def _optional_boolean_array(
    value: ArrayLike | None,
    *,
    name: str,
    size: int,
) -> BoolArray:
    if value is None:
        return np.zeros(size, dtype=np.bool_)
    return _boolean_array(value, name=name, size=size)


def _prepare_entities(entities: SpatialEntityArrays) -> _PreparedEntities:
    xy_m = np.asarray(entities.xy_m, dtype=np.float64)
    if xy_m.ndim != 2 or xy_m.shape[1] != 2:
        raise ValueError("entity xy_m must have shape (n_entity, 2)")
    if np.isinf(xy_m).any():
        raise ValueError("entity xy_m cannot contain infinity")
    size = int(xy_m.shape[0])

    statuses = {
        name: _optional_boolean_array(
            getattr(entities, _STATUS_INPUT_FIELDS.get(name, name)),
            name=name,
            size=size,
        )
        for name in _CORE_STATUS_FIELDS
    }
    reliability_high = _boolean_array(
        entities.reliability_high,
        name="reliability_high",
        size=size,
    )
    reliability_cautious = _boolean_array(
        entities.reliability_cautious,
        name="reliability_cautious",
        size=size,
    )
    if np.any(reliability_high & reliability_cautious):
        raise ValueError("high and cautious reliability tiers cannot overlap")
    station_id = _one_dimensional(
        entities.station_id,
        name="station_id",
        size=size,
        dtype=np.object_,
    )
    measurement_id = _one_dimensional(
        entities.measurement_id,
        name="measurement_id",
        size=size,
        dtype=np.object_,
    )

    raw_discipline = np.asarray(entities.discipline_code)
    if raw_discipline.ndim != 1 or raw_discipline.shape[0] != size:
        raise ValueError(f"discipline_code must have shape ({size},)")
    if not np.issubdtype(raw_discipline.dtype, np.integer):
        raise TypeError("discipline_code must use integer codes")
    discipline_code = np.asarray(raw_discipline, dtype=np.int64)
    if np.any((discipline_code < 0) | (discipline_code >= len(DISCIPLINE_NAMES))):
        raise ValueError("discipline_code values must be in the closed interval [0, 3]")
    if entities.discipline_membership is None:
        discipline_membership = np.zeros(
            (size, len(DISCIPLINE_NAMES)),
            dtype=np.bool_,
        )
        discipline_membership[np.arange(size), discipline_code] = True
    else:
        raw_membership = np.asarray(entities.discipline_membership)
        expected_shape = (size, len(DISCIPLINE_NAMES))
        if raw_membership.shape != expected_shape:
            raise ValueError(f"discipline_membership must have shape {expected_shape}")
        if raw_membership.dtype != np.dtype(np.bool_):
            raise TypeError("discipline_membership must be a boolean array")
        if np.any(np.sum(raw_membership, axis=1) < 1):
            raise ValueError("every entity must have at least one discipline membership")
        discipline_membership = np.asarray(raw_membership, dtype=np.bool_)

    age_days = np.asarray(
        _one_dimensional(entities.age_days, name="age_days", size=size, dtype=np.float64),
        dtype=np.float64,
    )
    known_duration_days = np.asarray(
        _one_dimensional(
            entities.known_duration_days,
            name="known_duration_days",
            size=size,
            dtype=np.float64,
        ),
        dtype=np.float64,
    )
    for name, values in (("age_days", age_days), ("known_duration_days", known_duration_days)):
        if np.isinf(values).any() or np.any(values[np.isfinite(values)] < 0.0):
            raise ValueError(f"{name} must contain only non-negative finite values or NaN")
    multidisciplinary = _optional_boolean_array(
        entities.multidisciplinary,
        name="multidisciplinary",
        size=size,
    )
    reporting_listed = (
        statuses["listed"].copy()
        if entities.reporting_listed is None
        else _boolean_array(
            entities.reporting_listed,
            name="reporting_listed",
            size=size,
        )
    )
    if not isinstance(entities.coverage_unique_only, bool):
        raise TypeError("coverage_unique_only must be boolean")

    coordinate_mask = np.isfinite(xy_m).all(axis=1)
    spatial_count = int(np.count_nonzero(coordinate_mask))
    selected_xy = np.asarray(xy_m[coordinate_mask], dtype=np.float64)
    selected_statuses = {name: values[coordinate_mask] for name, values in statuses.items()}
    selected_high = reliability_high[coordinate_mask]
    selected_cautious = reliability_cautious[coordinate_mask]
    selected_station = np.asarray(station_id[coordinate_mask], dtype=np.object_)
    selected_measurement = np.asarray(measurement_id[coordinate_mask], dtype=np.object_)
    selected_discipline = discipline_code[coordinate_mask]
    selected_membership = discipline_membership[coordinate_mask]
    selected_age = age_days[coordinate_mask]
    selected_duration = known_duration_days[coordinate_mask]
    selected_multidisciplinary = multidisciplinary[coordinate_mask]
    selected_reporting = reporting_listed[coordinate_mask]
    if entities.coverage_unique_only and selected_xy.shape[0]:
        keep = _coverage_unique_row_indices(
            selected_xy,
            selected_station,
            selected_measurement,
        )
        selected_xy = selected_xy[keep]
        selected_statuses = {name: values[keep] for name, values in selected_statuses.items()}
        selected_high = selected_high[keep]
        selected_cautious = selected_cautious[keep]
        selected_station = selected_station[keep]
        selected_measurement = selected_measurement[keep]
        selected_discipline = selected_discipline[keep]
        selected_membership = selected_membership[keep]
        selected_age = selected_age[keep]
        selected_duration = selected_duration[keep]
        selected_multidisciplinary = selected_multidisciplinary[keep]
        selected_reporting = selected_reporting[keep]
    station_codes, measurement_codes, pair_codes = _identifier_code_arrays(
        selected_station,
        selected_measurement,
    )
    return _PreparedEntities(
        xy_m=selected_xy,
        statuses=selected_statuses,
        reliability_high=selected_high,
        reliability_cautious=selected_cautious,
        station_id=selected_station,
        measurement_id=selected_measurement,
        discipline_code=selected_discipline,
        discipline_membership=selected_membership,
        age_days=selected_age,
        known_duration_days=selected_duration,
        multidisciplinary=selected_multidisciplinary,
        reporting_listed=selected_reporting,
        station_code=station_codes,
        measurement_code=measurement_codes,
        station_measurement_code=pair_codes,
        input_count=size,
        spatial_count=spatial_count,
        missing_coordinate_count=size - spatial_count,
        coverage_unique_only=entities.coverage_unique_only,
    )


def _empty_feature_arrays(n_query: int, n_scale: int) -> dict[str, FloatArray]:
    shape = (n_query, n_scale)
    features: dict[str, FloatArray] = {}
    for status in _CORE_STATUS_FIELDS:
        features[f"{status}_count"] = np.zeros(shape, dtype=np.float64)
        features[f"{status}_weighted_count"] = np.zeros(shape, dtype=np.float64)
    for name in (
        "high_reliability_listed_count",
        "cautious_reliability_listed_count",
        "excluded_reliability_listed_count",
        "reliability_weighted_listed_count",
        "spatial_weight_mass",
        "unique_station_count",
        "unique_measurement_count",
        "unique_station_measurement_count",
        "discipline_count",
        "multidisciplinary_entity_count",
        "multidisciplinary_entity_weighted_count",
    ):
        features[name] = np.zeros(shape, dtype=np.float64)
    for discipline in DISCIPLINE_NAMES:
        features[f"discipline_{discipline}_count"] = np.zeros(shape, dtype=np.float64)
    for name in (
        "discipline_shannon_normalized",
        "age_mean_days",
        "known_duration_mean_days",
        "mean_distance_km",
        "diffusion_radius_km",
        "concentration",
        "principal_direction_deg",
        "anisotropy",
        "first_seen_rate",
        "first_seen_weighted_rate",
        "right_censored_fraction",
        "right_censored_weighted_fraction",
        "multidisciplinary_entity_fraction",
        "multidisciplinary_entity_weighted_fraction",
    ):
        features[name] = np.full(shape, np.nan, dtype=np.float64)
    features["age_known_count"] = np.zeros(shape, dtype=np.float64)
    features["known_duration_count"] = np.zeros(shape, dtype=np.float64)
    return features


def _empty_coverage_unique_arrays(n_query: int, n_scale: int) -> dict[str, FloatArray]:
    shape = (n_query, n_scale)
    return {
        "unique_station_count": np.zeros(shape, dtype=np.float64),
        "unique_measurement_count": np.zeros(shape, dtype=np.float64),
    }


def _identifier(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coverage_unique_row_indices(
    xy_m: FloatArray,
    station_id: NDArray[np.object_],
    measurement_id: NDArray[np.object_],
) -> NDArray[np.int64]:
    """Remove exact repeated reporting rows without changing any unique-ID maximum."""

    seen: set[tuple[float, float, str | None, str | None]] = set()
    keep: list[int] = []
    for index, (point, station_value, measurement_value) in enumerate(
        zip(xy_m, station_id, measurement_id, strict=True)
    ):
        station = _identifier(station_value)
        measurement = _identifier(measurement_value)
        if station is None and measurement is None:
            continue
        key = (float(point[0]), float(point[1]), station, measurement)
        if key in seen:
            continue
        seen.add(key)
        keep.append(index)
    return np.asarray(keep, dtype=np.int64)


def _identifier_code_arrays(
    station_id: NDArray[np.object_],
    measurement_id: NDArray[np.object_],
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    station_codes = np.full(station_id.shape[0], -1, dtype=np.int64)
    measurement_codes = np.full(measurement_id.shape[0], -1, dtype=np.int64)
    pair_codes = np.full(station_id.shape[0], -1, dtype=np.int64)
    station_mapping: dict[str, int] = {}
    measurement_mapping: dict[str, int] = {}
    pair_mapping: dict[tuple[str, str], int] = {}
    for index, (station_value, measurement_value) in enumerate(
        zip(station_id, measurement_id, strict=True)
    ):
        station = _identifier(station_value)
        measurement = _identifier(measurement_value)
        if station is not None:
            station_codes[index] = station_mapping.setdefault(station, len(station_mapping))
        if measurement is not None:
            measurement_codes[index] = measurement_mapping.setdefault(
                measurement,
                len(measurement_mapping),
            )
        if station is not None and measurement is not None:
            pair = (station, measurement)
            pair_codes[index] = pair_mapping.setdefault(pair, len(pair_mapping))
    return station_codes, measurement_codes, pair_codes


def _unique_weight(
    first_ids: NDArray[np.object_],
    second_ids: NDArray[np.object_] | None,
    weights: FloatArray,
) -> float:
    maximum_by_identifier: dict[str | tuple[str, str], float] = {}
    for index, first_value in enumerate(first_ids):
        first = _identifier(first_value)
        if first is None:
            continue
        if second_ids is None:
            key: str | tuple[str, str] = first
        else:
            second = _identifier(second_ids[index])
            if second is None:
                continue
            key = (first, second)
        weight = float(weights[index])
        previous = maximum_by_identifier.get(key)
        if previous is None or weight > previous:
            maximum_by_identifier[key] = weight
    return float(sum(maximum_by_identifier.values()))


def _weighted_mean(values: FloatArray, weights: FloatArray) -> tuple[float, float]:
    valid = np.isfinite(values) & (weights > 0.0)
    count = float(np.count_nonzero(valid))
    if count == 0.0:
        return np.nan, 0.0
    selected_weights = weights[valid]
    denominator = float(np.sum(selected_weights))
    if denominator <= 0.0:
        return np.nan, count
    return float(np.sum(values[valid] * selected_weights) / denominator), count


def _populate_one(
    features: dict[str, FloatArray],
    *,
    query_index: int,
    scale_index: int,
    selected: BoolArray,
    base_weights: FloatArray,
    distances_m: FloatArray,
    support_radius_m: float,
    entities: _PreparedEntities,
) -> None:
    if not np.any(selected):
        return
    weights = np.asarray(base_weights[selected], dtype=np.float64)
    listed = entities.statuses["listed"][selected]
    high = entities.reliability_high[selected]
    cautious = entities.reliability_cautious[selected]
    excluded = ~(high | cautious)
    reliability_weights = (
        high.astype(np.float64) * HIGH_RELIABILITY_WEIGHT
        + cautious.astype(np.float64) * CAUTIOUS_RELIABILITY_WEIGHT
    )
    features["spatial_weight_mass"][query_index, scale_index] = float(np.sum(weights))

    for status_name in _CORE_STATUS_FIELDS:
        status = entities.statuses[status_name][selected]
        features[f"{status_name}_count"][query_index, scale_index] = float(np.sum(weights * status))
        features[f"{status_name}_weighted_count"][query_index, scale_index] = float(
            np.sum(weights * status * reliability_weights)
        )

    listed_weights = weights * listed
    features["high_reliability_listed_count"][query_index, scale_index] = float(
        np.sum(weights * listed * high)
    )
    features["cautious_reliability_listed_count"][query_index, scale_index] = float(
        np.sum(weights * listed * cautious)
    )
    features["excluded_reliability_listed_count"][query_index, scale_index] = float(
        np.sum(weights * listed * excluded)
    )
    features["reliability_weighted_listed_count"][query_index, scale_index] = float(
        np.sum(listed_weights * reliability_weights)
    )
    listed_total = float(np.sum(listed_weights))
    reliability_listed_total = float(np.sum(listed_weights * reliability_weights))

    eligible_first_seen_denominator = (
        listed_total - features["left_truncated_count"][query_index, scale_index]
    )
    if eligible_first_seen_denominator > 0.0:
        features["first_seen_rate"][query_index, scale_index] = (
            features["first_seen_count"][query_index, scale_index] / eligible_first_seen_denominator
        )
    eligible_first_seen_weighted_denominator = (
        reliability_listed_total
        - features["left_truncated_weighted_count"][query_index, scale_index]
    )
    if eligible_first_seen_weighted_denominator > 0.0:
        features["first_seen_weighted_rate"][query_index, scale_index] = (
            features["first_seen_weighted_count"][query_index, scale_index]
            / eligible_first_seen_weighted_denominator
        )
    if listed_total > 0.0:
        features["right_censored_fraction"][query_index, scale_index] = (
            features["right_censored_count"][query_index, scale_index] / listed_total
        )
    if reliability_listed_total > 0.0:
        features["right_censored_weighted_fraction"][query_index, scale_index] = (
            features["right_censored_weighted_count"][query_index, scale_index]
            / reliability_listed_total
        )

    multidisciplinary = entities.multidisciplinary[selected]
    multidisciplinary_count = float(np.sum(listed_weights * multidisciplinary))
    multidisciplinary_weighted_count = float(
        np.sum(listed_weights * multidisciplinary * reliability_weights)
    )
    features["multidisciplinary_entity_count"][query_index, scale_index] = multidisciplinary_count
    features["multidisciplinary_entity_weighted_count"][query_index, scale_index] = (
        multidisciplinary_weighted_count
    )
    if listed_total > 0.0:
        features["multidisciplinary_entity_fraction"][query_index, scale_index] = (
            multidisciplinary_count / listed_total
        )
    if reliability_listed_total > 0.0:
        features["multidisciplinary_entity_weighted_fraction"][query_index, scale_index] = (
            multidisciplinary_weighted_count / reliability_listed_total
        )

    listed_selection = listed & (weights > 0.0)
    reporting_selection = entities.reporting_listed[selected] & (weights > 0.0)
    listed_ids = entities.station_id[selected][reporting_selection]
    listed_measurements = entities.measurement_id[selected][reporting_selection]
    listed_unique_weights = weights[reporting_selection]
    features["unique_station_count"][query_index, scale_index] = _unique_weight(
        listed_ids,
        None,
        listed_unique_weights,
    )
    features["unique_measurement_count"][query_index, scale_index] = _unique_weight(
        listed_measurements,
        None,
        listed_unique_weights,
    )
    features["unique_station_measurement_count"][query_index, scale_index] = _unique_weight(
        listed_ids,
        listed_measurements,
        listed_unique_weights,
    )

    discipline_membership = entities.discipline_membership[selected]
    discipline_totals = np.zeros(len(DISCIPLINE_NAMES), dtype=np.float64)
    for code, discipline_name in enumerate(DISCIPLINE_NAMES):
        total = float(np.sum(listed_weights * discipline_membership[:, code]))
        discipline_totals[code] = total
        features[f"discipline_{discipline_name}_count"][query_index, scale_index] = total
    features["discipline_count"][query_index, scale_index] = float(
        np.count_nonzero(discipline_totals > 0.0)
    )
    discipline_total = float(np.sum(discipline_totals))
    if discipline_total > 0.0:
        probabilities = discipline_totals[discipline_totals > 0.0] / discipline_total
        features["discipline_shannon_normalized"][query_index, scale_index] = float(
            -np.sum(probabilities * np.log(probabilities)) / np.log(len(DISCIPLINE_NAMES))
        )

    age_mean, age_count = _weighted_mean(entities.age_days[selected], listed_weights)
    duration_mean, duration_count = _weighted_mean(
        entities.known_duration_days[selected],
        listed_weights,
    )
    features["age_mean_days"][query_index, scale_index] = age_mean
    features["age_known_count"][query_index, scale_index] = age_count
    features["known_duration_mean_days"][query_index, scale_index] = duration_mean
    features["known_duration_count"][query_index, scale_index] = duration_count

    if not np.any(listed_selection):
        return
    geometry_weights = weights[listed_selection]
    geometry_weight_sum = float(np.sum(geometry_weights))
    if geometry_weight_sum <= 0.0:
        return
    selected_distances = distances_m[selected][listed_selection]
    features["mean_distance_km"][query_index, scale_index] = float(
        np.sum(selected_distances * geometry_weights) / geometry_weight_sum / 1000.0
    )

    points = entities.xy_m[selected][listed_selection]
    centroid = np.sum(points * geometry_weights[:, None], axis=0) / geometry_weight_sum
    centered = points - centroid
    covariance = (centered * geometry_weights[:, None]).T @ centered / geometry_weight_sum
    covariance = np.asarray(covariance, dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    trace = float(np.sum(eigenvalues))
    diffusion_radius_m = float(np.sqrt(trace))
    features["diffusion_radius_km"][query_index, scale_index] = diffusion_radius_m / 1000.0
    features["concentration"][query_index, scale_index] = float(
        np.clip(1.0 - diffusion_radius_m / support_radius_m, 0.0, 1.0)
    )
    if trace <= np.finfo(np.float64).eps:
        features["anisotropy"][query_index, scale_index] = 0.0
        return
    anisotropy = float((eigenvalues[-1] - eigenvalues[0]) / trace)
    features["anisotropy"][query_index, scale_index] = anisotropy
    if anisotropy <= 1e-12:
        return
    principal = eigenvectors[:, -1]
    direction = float(np.degrees(np.arctan2(principal[1], principal[0])) % 180.0)
    features["principal_direction_deg"][query_index, scale_index] = direction


def _compute_spatial_features_dense_reference(
    query_xy_m: ArrayLike,
    entities: SpatialEntityArrays,
    *,
    query_chunk_size: int = 256,
) -> SpatialFeatureResult:
    """Aggregate one causal entity snapshot around fixed projected query points.

    Radius features use closed balls at 50/100/200/300/500 km.  Gaussian features
    use matching bandwidths and the unnormalised weight ``exp(-d^2/(2h^2))``, with
    weights deterministically set to zero outside ``3h``.  Gaussian unique-ID
    proxies count each ID once using its maximum contributing kernel weight.

    Geometry, discipline, age, duration, and coverage proxies describe entities
    marked ``listed``.  Core transition counts use their corresponding status masks.
    """

    queries = np.asarray(query_xy_m, dtype=np.float64)
    if queries.ndim != 2 or queries.shape[1] != 2:
        raise ValueError("query_xy_m must have shape (n_query, 2)")
    if not np.isfinite(queries).all():
        raise ValueError("query_xy_m must contain only finite projected metre coordinates")
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")

    prepared = _prepare_entities(entities)
    scales_km = np.asarray(SPATIAL_SCALES_KM, dtype=np.float64)
    scales_m = scales_km * 1000.0
    n_query = int(queries.shape[0])
    n_scale = int(scales_m.size)
    radius_features = _empty_feature_arrays(n_query, n_scale)
    gaussian_features = _empty_feature_arrays(n_query, n_scale)

    for chunk_start in range(0, n_query, query_chunk_size):
        chunk_stop = min(chunk_start + query_chunk_size, n_query)
        deltas = queries[chunk_start:chunk_stop, None, :] - prepared.xy_m[None, :, :]
        distance_squared = np.einsum("qni,qni->qn", deltas, deltas)
        distances_m = np.sqrt(distance_squared)
        for local_query_index in range(chunk_stop - chunk_start):
            query_index = chunk_start + local_query_index
            query_distances = np.asarray(distances_m[local_query_index], dtype=np.float64)
            query_distance_squared = np.asarray(
                distance_squared[local_query_index],
                dtype=np.float64,
            )
            for scale_index, scale_m in enumerate(scales_m):
                radius_selected = query_distance_squared <= scale_m * scale_m
                radius_weights = np.ones(prepared.xy_m.shape[0], dtype=np.float64)
                _populate_one(
                    radius_features,
                    query_index=query_index,
                    scale_index=scale_index,
                    selected=radius_selected,
                    base_weights=radius_weights,
                    distances_m=query_distances,
                    support_radius_m=float(scale_m),
                    entities=prepared,
                )

                gaussian_selected = query_distances <= GAUSSIAN_TRUNCATION_SIGMA * scale_m
                gaussian_weights = np.zeros(prepared.xy_m.shape[0], dtype=np.float64)
                gaussian_weights[gaussian_selected] = np.exp(
                    -0.5 * query_distance_squared[gaussian_selected] / (scale_m * scale_m)
                )
                _populate_one(
                    gaussian_features,
                    query_index=query_index,
                    scale_index=scale_index,
                    selected=gaussian_selected,
                    base_weights=gaussian_weights,
                    distances_m=query_distances,
                    support_radius_m=float(GAUSSIAN_TRUNCATION_SIGMA * scale_m),
                    entities=prepared,
                )

    return SpatialFeatureResult(
        query_xy_m=np.asarray(queries, dtype=np.float64),
        scales_km=scales_km,
        radius_features=radius_features,
        gaussian_features=gaussian_features,
        input_entity_count=prepared.input_count,
        spatial_entity_count=prepared.spatial_count,
        missing_coordinate_count=prepared.missing_coordinate_count,
    )


def _aggregation_signals(
    entities: _PreparedEntities,
) -> tuple[FloatArray, tuple[str, ...], int, int]:
    """Pack all linear entity signals for one sparse matrix multiplication."""

    high = entities.reliability_high.astype(np.float64)
    cautious = entities.reliability_cautious.astype(np.float64)
    reliability = high + cautious * CAUTIOUS_RELIABILITY_WEIGHT
    listed = entities.statuses["listed"].astype(np.float64)
    columns: list[FloatArray] = []
    names: list[str] = []

    def add(name: str, values: ArrayLike) -> int:
        names.append(name)
        columns.append(np.asarray(values, dtype=np.float64))
        return len(columns) - 1

    add("spatial_weight_mass", np.ones(entities.xy_m.shape[0], dtype=np.float64))
    for status_name in _CORE_STATUS_FIELDS:
        status = entities.statuses[status_name].astype(np.float64)
        add(f"{status_name}_count", status)
        add(f"{status_name}_weighted_count", status * reliability)
    add("high_reliability_listed_count", listed * high)
    add("cautious_reliability_listed_count", listed * cautious)
    add("excluded_reliability_listed_count", listed * ((high + cautious) == 0.0))
    add("reliability_weighted_listed_count", listed * reliability)
    multidisciplinary = entities.multidisciplinary.astype(np.float64)
    add("multidisciplinary_entity_count", listed * multidisciplinary)
    add(
        "multidisciplinary_entity_weighted_count",
        listed * multidisciplinary * reliability,
    )
    for code, discipline_name in enumerate(DISCIPLINE_NAMES):
        add(
            f"discipline_{discipline_name}_count",
            listed * entities.discipline_membership[:, code],
        )

    age_valid = listed.astype(np.bool_) & np.isfinite(entities.age_days)
    duration_valid = listed.astype(np.bool_) & np.isfinite(entities.known_duration_days)
    add("__age_weighted_value", np.where(age_valid, entities.age_days, 0.0))
    age_valid_index = add("__age_valid", age_valid)
    add(
        "__duration_weighted_value",
        np.where(duration_valid, entities.known_duration_days, 0.0),
    )
    duration_valid_index = add("__duration_valid", duration_valid)
    matrix = (
        np.column_stack(columns)
        if columns
        else np.empty((entities.xy_m.shape[0], 0), dtype=np.float64)
    )
    return matrix, tuple(names), age_valid_index, duration_valid_index


def _assign_ratio(
    target: FloatArray,
    numerator: FloatArray,
    denominator: FloatArray,
) -> None:
    valid = denominator > 0.0
    target[valid] = numerator[valid] / denominator[valid]


def _populate_linear_and_derived_features(
    features: dict[str, FloatArray],
    *,
    query_start: int,
    query_stop: int,
    scale_index: int,
    weighted_matrix: Any,
    support_matrix: Any,
    signal_matrix: FloatArray,
    signal_names: tuple[str, ...],
    age_valid_index: int,
    duration_valid_index: int,
) -> None:
    aggregated = np.asarray(weighted_matrix @ signal_matrix, dtype=np.float64)
    lookup = {name: index for index, name in enumerate(signal_names)}
    row_slice = slice(query_start, query_stop)
    for name, index in lookup.items():
        if not name.startswith("__"):
            features[name][row_slice, scale_index] = aggregated[:, index]

    listed = aggregated[:, lookup["listed_count"]]
    reliability_listed = aggregated[:, lookup["reliability_weighted_listed_count"]]
    left_truncated = aggregated[:, lookup["left_truncated_count"]]
    left_truncated_weighted = aggregated[:, lookup["left_truncated_weighted_count"]]
    _assign_ratio(
        features["first_seen_rate"][row_slice, scale_index],
        aggregated[:, lookup["first_seen_count"]],
        listed - left_truncated,
    )
    _assign_ratio(
        features["first_seen_weighted_rate"][row_slice, scale_index],
        aggregated[:, lookup["first_seen_weighted_count"]],
        reliability_listed - left_truncated_weighted,
    )
    _assign_ratio(
        features["right_censored_fraction"][row_slice, scale_index],
        aggregated[:, lookup["right_censored_count"]],
        listed,
    )
    _assign_ratio(
        features["right_censored_weighted_fraction"][row_slice, scale_index],
        aggregated[:, lookup["right_censored_weighted_count"]],
        reliability_listed,
    )
    _assign_ratio(
        features["multidisciplinary_entity_fraction"][row_slice, scale_index],
        aggregated[:, lookup["multidisciplinary_entity_count"]],
        listed,
    )
    _assign_ratio(
        features["multidisciplinary_entity_weighted_fraction"][row_slice, scale_index],
        aggregated[:, lookup["multidisciplinary_entity_weighted_count"]],
        reliability_listed,
    )

    discipline_totals = np.column_stack(
        [aggregated[:, lookup[f"discipline_{name}_count"]] for name in DISCIPLINE_NAMES]
    )
    features["discipline_count"][row_slice, scale_index] = np.count_nonzero(
        discipline_totals > 0.0,
        axis=1,
    )
    discipline_sum = np.sum(discipline_totals, axis=1)
    entropy_target = features["discipline_shannon_normalized"][row_slice, scale_index]
    valid_discipline = discipline_sum > 0.0
    if np.any(valid_discipline):
        probabilities = np.divide(
            discipline_totals[valid_discipline],
            discipline_sum[valid_discipline, None],
        )
        terms = np.zeros_like(probabilities)
        positive = probabilities > 0.0
        terms[positive] = probabilities[positive] * np.log(probabilities[positive])
        entropy_target[valid_discipline] = -np.sum(terms, axis=1) / np.log(len(DISCIPLINE_NAMES))

    age_denominator = aggregated[:, age_valid_index]
    duration_denominator = aggregated[:, duration_valid_index]
    age_count = np.asarray(support_matrix @ signal_matrix[:, age_valid_index], dtype=np.float64)
    duration_count = np.asarray(
        support_matrix @ signal_matrix[:, duration_valid_index],
        dtype=np.float64,
    )
    features["age_known_count"][row_slice, scale_index] = age_count
    features["known_duration_count"][row_slice, scale_index] = duration_count
    _assign_ratio(
        features["age_mean_days"][row_slice, scale_index],
        aggregated[:, lookup["__age_weighted_value"]],
        age_denominator,
    )
    _assign_ratio(
        features["known_duration_mean_days"][row_slice, scale_index],
        aggregated[:, lookup["__duration_weighted_value"]],
        duration_denominator,
    )


def _populate_geometry_features(
    features: dict[str, FloatArray],
    *,
    query_start: int,
    query_stop: int,
    scale_index: int,
    pair_rows: NDArray[np.int64],
    pair_entities: NDArray[np.int64],
    pair_distances_m: FloatArray,
    pair_weights: FloatArray,
    support_radius_m: float,
    entities: _PreparedEntities,
) -> None:
    chunk_size = query_stop - query_start
    if pair_rows.size == 0:
        return
    listed_weights = pair_weights * entities.statuses["listed"][pair_entities]
    positive = listed_weights > 0.0
    if not np.any(positive):
        return
    rows = pair_rows[positive]
    weights = listed_weights[positive]
    distances = pair_distances_m[positive]
    total = np.bincount(rows, weights=weights, minlength=chunk_size).astype(np.float64)
    valid = total > 0.0
    if not np.any(valid):
        return
    row_slice = slice(query_start, query_stop)
    mean_distance = np.bincount(
        rows,
        weights=weights * distances,
        minlength=chunk_size,
    )
    features["mean_distance_km"][row_slice, scale_index][valid] = (
        mean_distance[valid] / total[valid] / 1000.0
    )

    points = entities.xy_m[pair_entities[positive]]
    sum_x = np.bincount(rows, weights=weights * points[:, 0], minlength=chunk_size)
    sum_y = np.bincount(rows, weights=weights * points[:, 1], minlength=chunk_size)
    valid_indices = np.flatnonzero(valid)
    mean_x = np.zeros(chunk_size, dtype=np.float64)
    mean_y = np.zeros(chunk_size, dtype=np.float64)
    mean_x[valid] = sum_x[valid] / total[valid]
    mean_y[valid] = sum_y[valid] / total[valid]
    centered_x = points[:, 0] - mean_x[rows]
    centered_y = points[:, 1] - mean_y[rows]
    sum_xx = np.bincount(rows, weights=weights * centered_x**2, minlength=chunk_size)
    sum_xy = np.bincount(
        rows,
        weights=weights * centered_x * centered_y,
        minlength=chunk_size,
    )
    sum_yy = np.bincount(rows, weights=weights * centered_y**2, minlength=chunk_size)
    covariance = np.empty((valid_indices.size, 2, 2), dtype=np.float64)
    covariance[:, 0, 0] = sum_xx[valid] / total[valid]
    covariance[:, 0, 1] = sum_xy[valid] / total[valid]
    covariance[:, 1, 0] = covariance[:, 0, 1]
    covariance[:, 1, 1] = sum_yy[valid] / total[valid]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    trace = np.sum(eigenvalues, axis=1)
    diffusion_radius_m = np.sqrt(trace)
    diffusion = features["diffusion_radius_km"][row_slice, scale_index]
    concentration = features["concentration"][row_slice, scale_index]
    anisotropy_output = features["anisotropy"][row_slice, scale_index]
    direction_output = features["principal_direction_deg"][row_slice, scale_index]
    diffusion[valid_indices] = diffusion_radius_m / 1000.0
    concentration[valid_indices] = np.clip(
        1.0 - diffusion_radius_m / support_radius_m,
        0.0,
        1.0,
    )
    positive_trace = trace > np.finfo(np.float64).eps
    anisotropy = np.zeros(trace.shape, dtype=np.float64)
    anisotropy[positive_trace] = (
        eigenvalues[positive_trace, 1] - eigenvalues[positive_trace, 0]
    ) / trace[positive_trace]
    anisotropy_output[valid_indices] = anisotropy
    directional = anisotropy > 1e-12
    if np.any(directional):
        principal = eigenvectors[directional, :, 1]
        direction_output[valid_indices[directional]] = (
            np.degrees(np.arctan2(principal[:, 1], principal[:, 0])) % 180.0
        )


def _minimum_distance_squared_by_code(
    codes: NDArray[np.int64],
    distance_squared: FloatArray,
) -> FloatArray:
    valid = codes >= 0
    if not np.any(valid):
        return np.empty(0, dtype=np.float64)
    _, inverse = np.unique(codes[valid], return_inverse=True)
    minimum = np.full(int(np.max(inverse)) + 1, np.inf, dtype=np.float64)
    np.minimum.at(minimum, inverse, distance_squared[valid])
    return minimum


def _populate_unique_identifier_features(
    radius_features: dict[str, FloatArray],
    gaussian_features: dict[str, FloatArray],
    *,
    query_start: int,
    query_stop: int,
    pair_rows: NDArray[np.int64],
    pair_entities: NDArray[np.int64],
    pair_distance_squared: FloatArray,
    scales_m: FloatArray,
    entities: _PreparedEntities,
) -> None:
    boundaries = np.searchsorted(
        pair_rows,
        np.arange(query_stop - query_start + 1, dtype=np.int64),
    )
    families = (
        ("unique_station_count", entities.station_code),
        ("unique_measurement_count", entities.measurement_code),
        ("unique_station_measurement_count", entities.station_measurement_code),
    )
    radius_squared = scales_m**2
    gaussian_support_squared = (GAUSSIAN_TRUNCATION_SIGMA * scales_m) ** 2
    for local_query in range(query_stop - query_start):
        start = int(boundaries[local_query])
        stop = int(boundaries[local_query + 1])
        if start == stop:
            continue
        entity_indices = pair_entities[start:stop]
        reporting = entities.reporting_listed[entity_indices]
        if not np.any(reporting):
            continue
        distance_squared = pair_distance_squared[start:stop][reporting]
        query_index = query_start + local_query
        for feature_name, all_codes in families:
            if feature_name not in radius_features:
                continue
            minimum = _minimum_distance_squared_by_code(
                all_codes[entity_indices][reporting],
                distance_squared,
            )
            if minimum.size == 0:
                continue
            radius_features[feature_name][query_index, :] = np.count_nonzero(
                minimum[:, None] <= radius_squared[None, :],
                axis=0,
            )
            inside = minimum[:, None] <= gaussian_support_squared[None, :]
            gaussian_weights = np.zeros(inside.shape, dtype=np.float64)
            exponent = -0.5 * minimum[:, None] / radius_squared[None, :]
            gaussian_weights[inside] = np.exp(exponent[inside])
            gaussian_features[feature_name][query_index, :] = np.sum(
                gaussian_weights,
                axis=0,
            )


def _candidate_pairs(
    tree: Any,
    queries: FloatArray,
    entity_xy_m: FloatArray,
    *,
    maximum_support_m: float,
) -> tuple[NDArray[np.int64], NDArray[np.int64], FloatArray]:
    raw_neighbors = tree.query_ball_point(
        queries,
        r=np.nextafter(maximum_support_m, np.inf),
        workers=1,
        return_sorted=True,
    )
    neighbors = cast(Sequence[Sequence[int]], raw_neighbors)
    counts = np.fromiter((len(item) for item in neighbors), dtype=np.int64, count=len(neighbors))
    if not np.any(counts):
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
        )
    pair_rows = np.repeat(np.arange(len(neighbors), dtype=np.int64), counts)
    pair_entities = np.concatenate([np.asarray(item, dtype=np.int64) for item in neighbors if item])
    deltas = entity_xy_m[pair_entities] - queries[pair_rows]
    distance_squared = np.einsum("ni,ni->n", deltas, deltas)
    exact = distance_squared <= maximum_support_m * maximum_support_m
    return (
        pair_rows[exact],
        pair_entities[exact],
        np.asarray(distance_squared[exact], dtype=np.float64),
    )


def _populate_kernel_scale(
    features: dict[str, FloatArray],
    *,
    query_start: int,
    query_stop: int,
    scale_index: int,
    scale_m: float,
    gaussian: bool,
    candidate_rows: NDArray[np.int64],
    candidate_entities: NDArray[np.int64],
    candidate_distance_squared: FloatArray,
    entity_count: int,
    entities: _PreparedEntities,
    signal_matrix: FloatArray,
    signal_names: tuple[str, ...],
    age_valid_index: int,
    duration_valid_index: int,
) -> None:
    support_radius_m = GAUSSIAN_TRUNCATION_SIGMA * scale_m if gaussian else scale_m
    selected = candidate_distance_squared <= support_radius_m * support_radius_m
    rows = candidate_rows[selected]
    entity_indices = candidate_entities[selected]
    distance_squared = candidate_distance_squared[selected]
    if gaussian:
        weights = np.exp(-0.5 * distance_squared / (scale_m * scale_m))
    else:
        weights = np.ones(distance_squared.shape[0], dtype=np.float64)
    chunk_size = query_stop - query_start
    support_matrix = csr_matrix(
        (np.ones(weights.shape[0], dtype=np.float64), (rows, entity_indices)),
        shape=(chunk_size, entity_count),
    )
    weighted_matrix = (
        support_matrix
        if not gaussian
        else csr_matrix(
            (weights, (rows, entity_indices)),
            shape=(chunk_size, entity_count),
        )
    )
    _populate_linear_and_derived_features(
        features,
        query_start=query_start,
        query_stop=query_stop,
        scale_index=scale_index,
        weighted_matrix=weighted_matrix,
        support_matrix=support_matrix,
        signal_matrix=signal_matrix,
        signal_names=signal_names,
        age_valid_index=age_valid_index,
        duration_valid_index=duration_valid_index,
    )
    _populate_geometry_features(
        features,
        query_start=query_start,
        query_stop=query_stop,
        scale_index=scale_index,
        pair_rows=rows,
        pair_entities=entity_indices,
        pair_distances_m=np.sqrt(distance_squared),
        pair_weights=weights,
        support_radius_m=support_radius_m,
        entities=entities,
    )


def _compute_spatial_features_sparse(
    query_xy_m: ArrayLike,
    entities: SpatialEntityArrays,
    *,
    query_chunk_size: int,
    selected_scales_km: Sequence[float],
) -> SpatialFeatureResult:
    queries = np.asarray(query_xy_m, dtype=np.float64)
    if queries.ndim != 2 or queries.shape[1] != 2:
        raise ValueError("query_xy_m must have shape (n_query, 2)")
    if not np.isfinite(queries).all():
        raise ValueError("query_xy_m must contain only finite projected metre coordinates")
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")

    frozen_scales = tuple(float(value) for value in selected_scales_km)
    if (
        not frozen_scales
        or len(set(frozen_scales)) != len(frozen_scales)
        or tuple(sorted(frozen_scales)) != frozen_scales
        or any(value not in SPATIAL_SCALES_KM for value in frozen_scales)
    ):
        raise ValueError("selected spatial scales must be a sorted nonempty frozen subset")

    prepared = _prepare_entities(entities)
    scales_km = np.asarray(frozen_scales, dtype=np.float64)
    scales_m = scales_km * 1000.0
    n_query = int(queries.shape[0])
    n_scale = int(scales_m.size)
    if prepared.coverage_unique_only:
        radius_features = _empty_coverage_unique_arrays(n_query, n_scale)
        gaussian_features = _empty_coverage_unique_arrays(n_query, n_scale)
    else:
        radius_features = _empty_feature_arrays(n_query, n_scale)
        gaussian_features = _empty_feature_arrays(n_query, n_scale)
    if n_query and prepared.xy_m.shape[0]:
        tree = cKDTree(
            prepared.xy_m,
            compact_nodes=True,
            balanced_tree=True,
            copy_data=True,
        )
        if not prepared.coverage_unique_only:
            signal_matrix, signal_names, age_valid_index, duration_valid_index = (
                _aggregation_signals(prepared)
            )
        for chunk_start in range(0, n_query, query_chunk_size):
            chunk_stop = min(chunk_start + query_chunk_size, n_query)
            rows, entity_indices, distance_squared = _candidate_pairs(
                tree,
                np.asarray(queries[chunk_start:chunk_stop], dtype=np.float64),
                prepared.xy_m,
                maximum_support_m=float(GAUSSIAN_TRUNCATION_SIGMA * scales_m[-1]),
            )
            _populate_unique_identifier_features(
                radius_features,
                gaussian_features,
                query_start=chunk_start,
                query_stop=chunk_stop,
                pair_rows=rows,
                pair_entities=entity_indices,
                pair_distance_squared=distance_squared,
                scales_m=scales_m,
                entities=prepared,
            )
            if prepared.coverage_unique_only:
                continue
            for scale_index, scale_m in enumerate(scales_m):
                _populate_kernel_scale(
                    radius_features,
                    query_start=chunk_start,
                    query_stop=chunk_stop,
                    scale_index=scale_index,
                    scale_m=float(scale_m),
                    gaussian=False,
                    candidate_rows=rows,
                    candidate_entities=entity_indices,
                    candidate_distance_squared=distance_squared,
                    entity_count=prepared.xy_m.shape[0],
                    entities=prepared,
                    signal_matrix=signal_matrix,
                    signal_names=signal_names,
                    age_valid_index=age_valid_index,
                    duration_valid_index=duration_valid_index,
                )
                _populate_kernel_scale(
                    gaussian_features,
                    query_start=chunk_start,
                    query_stop=chunk_stop,
                    scale_index=scale_index,
                    scale_m=float(scale_m),
                    gaussian=True,
                    candidate_rows=rows,
                    candidate_entities=entity_indices,
                    candidate_distance_squared=distance_squared,
                    entity_count=prepared.xy_m.shape[0],
                    entities=prepared,
                    signal_matrix=signal_matrix,
                    signal_names=signal_names,
                    age_valid_index=age_valid_index,
                    duration_valid_index=duration_valid_index,
                )

    return SpatialFeatureResult(
        query_xy_m=np.asarray(queries, dtype=np.float64),
        scales_km=scales_km,
        radius_features=radius_features,
        gaussian_features=gaussian_features,
        input_entity_count=prepared.input_count,
        spatial_entity_count=prepared.spatial_count,
        missing_coordinate_count=prepared.missing_coordinate_count,
    )


def compute_spatial_features(
    query_xy_m: ArrayLike,
    entities: SpatialEntityArrays,
    *,
    query_chunk_size: int = 256,
) -> SpatialFeatureResult:
    """Aggregate fixed-grid features with deterministic sparse neighbourhoods."""

    return _compute_spatial_features_sparse(
        query_xy_m,
        entities,
        query_chunk_size=query_chunk_size,
        selected_scales_km=SPATIAL_SCALES_KM,
    )


def compute_stage4_placebo_spatial_features(
    query_xy_m: ArrayLike,
    entities: SpatialEntityArrays,
    *,
    query_chunk_size: int = 256,
) -> Stage4PlaceboSpatialFeatureResult:
    """Compute only the exact frozen 200 km fields used by the stage-4 placebo.

    This is a scientific-field projection of :func:`compute_spatial_features`, not
    a new formula.  It intentionally omits expensive identifier and unused geometry
    aggregates while retaining the same sparse neighbourhood, kernels, reliability
    weights, discipline entropy, age mean, and covariance calculations.
    """

    queries = np.asarray(query_xy_m, dtype=np.float64)
    if queries.ndim != 2 or queries.shape[1] != 2:
        raise ValueError("query_xy_m must have shape (n_query, 2)")
    if not np.isfinite(queries).all():
        raise ValueError("query_xy_m must contain only finite projected metre coordinates")
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")

    prepared = _prepare_entities(entities)
    if prepared.coverage_unique_only:
        raise ValueError("stage-4 placebo fields require scientific anomaly entities")
    n_query = int(queries.shape[0])
    shape = (n_query, 1)
    radius_features = {
        "listed_count": np.zeros(shape, dtype=np.float64),
        "first_seen_count": np.zeros(shape, dtype=np.float64),
    }
    gaussian_features = {
        "reliability_weighted_listed_count": np.zeros(shape, dtype=np.float64),
        "first_seen_weighted_count": np.zeros(shape, dtype=np.float64),
        "not_continued_weighted_count": np.zeros(shape, dtype=np.float64),
        "age_mean_days": np.full(shape, np.nan, dtype=np.float64),
        "discipline_shannon_normalized": np.full(shape, np.nan, dtype=np.float64),
        "multidisciplinary_entity_weighted_fraction": np.full(
            shape,
            np.nan,
            dtype=np.float64,
        ),
        "concentration": np.full(shape, np.nan, dtype=np.float64),
        "diffusion_radius_km": np.full(shape, np.nan, dtype=np.float64),
    }
    if n_query and prepared.xy_m.shape[0]:
        high = prepared.reliability_high.astype(np.float64)
        cautious = prepared.reliability_cautious.astype(np.float64)
        reliability = high + cautious * CAUTIOUS_RELIABILITY_WEIGHT
        listed = prepared.statuses["listed"].astype(np.float64)
        first_seen = prepared.statuses["first_seen"].astype(np.float64)
        not_continued = prepared.statuses["not_continued"].astype(np.float64)
        age_valid = listed.astype(np.bool_) & np.isfinite(prepared.age_days)
        radius_signals = np.column_stack((listed, first_seen))
        gaussian_signals = np.column_stack(
            (
                listed * reliability,
                first_seen * reliability,
                not_continued * reliability,
                np.where(age_valid, prepared.age_days, 0.0),
                age_valid,
                listed * prepared.multidisciplinary * reliability,
                *(
                    listed * prepared.discipline_membership[:, code]
                    for code in range(len(DISCIPLINE_NAMES))
                ),
            )
        )
        tree = cKDTree(
            prepared.xy_m,
            compact_nodes=True,
            balanced_tree=True,
            copy_data=True,
        )
        scale_m = 200_000.0
        support_m = GAUSSIAN_TRUNCATION_SIGMA * scale_m
        entity_count = prepared.xy_m.shape[0]
        for chunk_start in range(0, n_query, query_chunk_size):
            chunk_stop = min(chunk_start + query_chunk_size, n_query)
            rows, entity_indices, distance_squared = _candidate_pairs(
                tree,
                np.asarray(queries[chunk_start:chunk_stop], dtype=np.float64),
                prepared.xy_m,
                maximum_support_m=support_m,
            )
            chunk_size = chunk_stop - chunk_start

            radius_selected = distance_squared <= scale_m * scale_m
            radius_matrix = csr_matrix(
                (
                    np.ones(np.count_nonzero(radius_selected), dtype=np.float64),
                    (rows[radius_selected], entity_indices[radius_selected]),
                ),
                shape=(chunk_size, entity_count),
            )
            radius_values = np.asarray(radius_matrix @ radius_signals, dtype=np.float64)
            radius_features["listed_count"][chunk_start:chunk_stop, 0] = radius_values[:, 0]
            radius_features["first_seen_count"][chunk_start:chunk_stop, 0] = radius_values[:, 1]

            weights = np.exp(-0.5 * distance_squared / (scale_m * scale_m))
            gaussian_matrix = csr_matrix(
                (weights, (rows, entity_indices)),
                shape=(chunk_size, entity_count),
            )
            aggregated = np.asarray(gaussian_matrix @ gaussian_signals, dtype=np.float64)
            gaussian_features["reliability_weighted_listed_count"][chunk_start:chunk_stop, 0] = (
                aggregated[:, 0]
            )
            gaussian_features["first_seen_weighted_count"][chunk_start:chunk_stop, 0] = aggregated[
                :, 1
            ]
            gaussian_features["not_continued_weighted_count"][chunk_start:chunk_stop, 0] = (
                aggregated[:, 2]
            )
            _assign_ratio(
                gaussian_features["age_mean_days"][chunk_start:chunk_stop, 0],
                aggregated[:, 3],
                aggregated[:, 4],
            )
            _assign_ratio(
                gaussian_features["multidisciplinary_entity_weighted_fraction"][
                    chunk_start:chunk_stop,
                    0,
                ],
                aggregated[:, 5],
                aggregated[:, 0],
            )
            discipline_totals = aggregated[:, 6:10]
            discipline_sum = np.sum(discipline_totals, axis=1)
            entropy_target = gaussian_features["discipline_shannon_normalized"][
                chunk_start:chunk_stop,
                0,
            ]
            valid_discipline = discipline_sum > 0.0
            if np.any(valid_discipline):
                probabilities = np.divide(
                    discipline_totals[valid_discipline],
                    discipline_sum[valid_discipline, None],
                )
                terms = np.zeros_like(probabilities)
                positive_probability = probabilities > 0.0
                terms[positive_probability] = probabilities[positive_probability] * np.log(
                    probabilities[positive_probability]
                )
                entropy_target[valid_discipline] = -np.sum(terms, axis=1) / np.log(
                    len(DISCIPLINE_NAMES)
                )

            listed_weights = weights * listed[entity_indices]
            positive = listed_weights > 0.0
            if not np.any(positive):
                continue
            geometry_rows = rows[positive]
            geometry_weights = listed_weights[positive]
            points = prepared.xy_m[entity_indices[positive]]
            total = np.bincount(
                geometry_rows,
                weights=geometry_weights,
                minlength=chunk_size,
            ).astype(np.float64)
            valid = total > 0.0
            if not np.any(valid):
                continue
            sum_x = np.bincount(
                geometry_rows,
                weights=geometry_weights * points[:, 0],
                minlength=chunk_size,
            )
            sum_y = np.bincount(
                geometry_rows,
                weights=geometry_weights * points[:, 1],
                minlength=chunk_size,
            )
            mean_x = np.zeros(chunk_size, dtype=np.float64)
            mean_y = np.zeros(chunk_size, dtype=np.float64)
            mean_x[valid] = sum_x[valid] / total[valid]
            mean_y[valid] = sum_y[valid] / total[valid]
            centered_x = points[:, 0] - mean_x[geometry_rows]
            centered_y = points[:, 1] - mean_y[geometry_rows]
            sum_xx = np.bincount(
                geometry_rows,
                weights=geometry_weights * centered_x**2,
                minlength=chunk_size,
            )
            sum_xy = np.bincount(
                geometry_rows,
                weights=geometry_weights * centered_x * centered_y,
                minlength=chunk_size,
            )
            sum_yy = np.bincount(
                geometry_rows,
                weights=geometry_weights * centered_y**2,
                minlength=chunk_size,
            )
            valid_indices = np.flatnonzero(valid)
            covariance = np.empty((valid_indices.size, 2, 2), dtype=np.float64)
            covariance[:, 0, 0] = sum_xx[valid] / total[valid]
            covariance[:, 0, 1] = sum_xy[valid] / total[valid]
            covariance[:, 1, 0] = covariance[:, 0, 1]
            covariance[:, 1, 1] = sum_yy[valid] / total[valid]
            eigenvalues, _eigenvectors = np.linalg.eigh(covariance)
            trace = np.sum(np.maximum(eigenvalues, 0.0), axis=1)
            diffusion_radius_m = np.sqrt(trace)
            gaussian_features["diffusion_radius_km"][chunk_start:chunk_stop, 0][valid_indices] = (
                diffusion_radius_m / 1000.0
            )
            gaussian_features["concentration"][chunk_start:chunk_stop, 0][valid_indices] = np.clip(
                1.0 - diffusion_radius_m / support_m, 0.0, 1.0
            )

    return Stage4PlaceboSpatialFeatureResult(
        query_xy_m=np.asarray(queries, dtype=np.float64),
        scales_km=np.asarray([200.0], dtype=np.float64),
        radius_features=radius_features,
        gaussian_features=gaussian_features,
        input_entity_count=prepared.input_count,
        spatial_entity_count=prepared.spatial_count,
        missing_coordinate_count=prepared.missing_coordinate_count,
    )


def compute_selected_spatial_features(
    query_xy_m: ArrayLike,
    entities: SpatialEntityArrays,
    *,
    scales_km: Sequence[float],
    query_chunk_size: int = 256,
) -> SpatialFeatureResult:
    """Compute only an explicitly frozen scale subset with the accepted formulas."""

    return _compute_spatial_features_sparse(
        query_xy_m,
        entities,
        query_chunk_size=query_chunk_size,
        selected_scales_km=scales_km,
    )


__all__ = [
    "CAUTIOUS_RELIABILITY_WEIGHT",
    "DISCIPLINE_NAMES",
    "GAUSSIAN_TRUNCATION_SIGMA",
    "HIGH_RELIABILITY_WEIGHT",
    "SPATIAL_SCALES_KM",
    "SpatialEntityArrays",
    "SpatialFeatureResult",
    "Stage4PlaceboSpatialFeatureResult",
    "compute_selected_spatial_features",
    "compute_spatial_features",
    "compute_stage4_placebo_spatial_features",
]
