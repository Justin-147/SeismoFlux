"""Leakage-safe earthquake catalog loading for stage-2 background models."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pyproj import CRS, Transformer
from shapely import covers, points
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform

SECONDS_PER_DAY = 86_400.0
NANOSECONDS_PER_DAY = 86_400_000_000_000.0

EARTHQUAKE_COLUMN_ALLOWLIST = (
    "event_id",
    "origin_time_utc",
    "available_at",
    "longitude",
    "latitude",
    "magnitude",
    "inside_study_area",
)


def _readonly(array: NDArray[Any]) -> NDArray[Any]:
    result = np.ascontiguousarray(array)
    result.setflags(write=False)
    return result


def utc_timestamp_to_day(value: str | pd.Timestamp) -> float:
    """Convert one UTC timestamp to days since the Unix epoch."""

    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("background timestamps must be timezone-aware")
    return float(timestamp.tz_convert("UTC").value) / NANOSECONDS_PER_DAY


def _load_geojson_geometry(path: Path) -> BaseGeometry:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read study-area GeoJSON: {path}") from exc
    if not isinstance(document, dict):
        raise ValueError("study-area GeoJSON root must be a mapping")
    kind = document.get("type")
    if kind == "Feature":
        geometry_value = document.get("geometry")
        if not isinstance(geometry_value, dict):
            raise ValueError("study-area feature has no geometry")
        geometry = shape(geometry_value)
    elif kind == "FeatureCollection":
        features_value = document.get("features")
        if not isinstance(features_value, list) or len(features_value) != 1:
            raise ValueError("study-area FeatureCollection must contain exactly one feature")
        feature = features_value[0]
        if not isinstance(feature, dict) or not isinstance(feature.get("geometry"), dict):
            raise ValueError("study-area feature has no geometry")
        geometry = shape(feature["geometry"])
    else:
        geometry = shape(document)
    if geometry.is_empty or not geometry.is_valid or geometry.area <= 0.0:
        raise ValueError("study-area geometry must be non-empty, valid, and positive-area")
    return cast(BaseGeometry, geometry)


@dataclass(frozen=True, slots=True)
class StudyArea:
    """Frozen study-area geometry in geographic and equal-area coordinates."""

    geographic: BaseGeometry
    projected: BaseGeometry
    equal_area_crs: str
    area_km2: float


def load_study_area(path: Path, equal_area_crs: str) -> StudyArea:
    """Load and project the target-independent study-area polygon."""

    geographic = _load_geojson_geometry(path)
    source = CRS.from_epsg(4326)
    target = CRS.from_user_input(equal_area_crs)
    transformer = Transformer.from_crs(source, target, always_xy=True)
    projected = cast(BaseGeometry, transform(transformer.transform, geographic))
    if projected.is_empty or not projected.is_valid or projected.area <= 0.0:
        raise ValueError("projected study-area geometry is invalid")
    return StudyArea(
        geographic=geographic,
        projected=projected,
        equal_area_crs=target.to_string(),
        area_km2=float(projected.area) / 1_000_000.0,
    )


@dataclass(frozen=True, slots=True)
class EarthquakeCatalog:
    """Columnar immutable earthquake data allowed by the stage-2 protocol."""

    event_id: NDArray[np.str_]
    origin_day: NDArray[np.float64]
    available_day: NDArray[np.float64]
    longitude: NDArray[np.float64]
    latitude: NDArray[np.float64]
    x_km: NDArray[np.float64]
    y_km: NDArray[np.float64]
    magnitude: NDArray[np.float64]
    inside_study_area: NDArray[np.bool_]
    inside_external_buffer: NDArray[np.bool_]

    def __post_init__(self) -> None:
        lengths = {len(cast(NDArray[Any], getattr(self, item.name))) for item in fields(self)}
        if len(lengths) != 1:
            raise ValueError("earthquake catalog columns must have one common length")

    def __len__(self) -> int:
        return len(self.event_id)

    def subset(self, mask: NDArray[np.bool_]) -> EarthquakeCatalog:
        """Return an immutable, order-preserving subset."""

        boolean_mask = np.asarray(mask, dtype=np.bool_)
        if boolean_mask.shape != (len(self),):
            raise ValueError("catalog subset mask has the wrong shape")
        values: dict[str, NDArray[Any]] = {}
        for item in fields(self):
            values[item.name] = _readonly(
                cast(NDArray[Any], getattr(self, item.name))[boolean_mask]
            )
        return EarthquakeCatalog(**values)

    def eligible(
        self,
        *,
        available_through_day: float,
        minimum_magnitude: float,
        inside_only: bool,
    ) -> EarthquakeCatalog:
        """Select events available by a cutoff without consulting future rows."""

        cutoff = float(available_through_day)
        minimum = float(minimum_magnitude)
        if not math.isfinite(cutoff) or not math.isfinite(minimum):
            raise ValueError("catalog eligibility thresholds must be finite")
        spatial = self.inside_study_area if inside_only else self.inside_external_buffer
        mask = (self.available_day <= cutoff) & (self.magnitude >= minimum) & spatial
        return self.subset(mask)


def load_earthquake_catalog(
    path: Path,
    *,
    study_area: StudyArea,
    external_buffer_km: float,
    maximum_event_time_utc: str | pd.Timestamp,
) -> EarthquakeCatalog:
    """Read the frozen allowlist through the locked validation-maturity cutoff."""

    buffer_km = float(external_buffer_km)
    if not math.isfinite(buffer_km) or buffer_km <= 0.0:
        raise ValueError("external trigger buffer must be positive and finite")
    maximum_time = pd.Timestamp(maximum_event_time_utc)
    if maximum_time.tzinfo is None:
        raise ValueError("maximum earthquake event time must be timezone-aware")
    maximum_time = maximum_time.tz_convert("UTC")
    frame = pd.read_parquet(
        path,
        columns=list(EARTHQUAKE_COLUMN_ALLOWLIST),
        filters=[
            ("origin_time_utc", "<=", maximum_time),
            ("available_at", "<=", maximum_time),
        ],
    )
    if tuple(frame.columns) != EARTHQUAKE_COLUMN_ALLOWLIST:
        raise ValueError("earthquake dataset columns differ from the stage-2 allowlist")
    if frame.empty:
        raise ValueError("earthquake dataset is empty")
    if frame[list(EARTHQUAKE_COLUMN_ALLOWLIST)].isna().any().any():
        raise ValueError("stage-2 earthquake columns must not contain missing values")

    event_id = frame["event_id"].astype("string").to_numpy(dtype=np.str_)
    origin = pd.to_datetime(frame["origin_time_utc"], utc=True, errors="raise")
    available = pd.to_datetime(frame["available_at"], utc=True, errors="raise")
    origin_ns = pd.DatetimeIndex(origin).as_unit("ns").asi8
    available_ns = pd.DatetimeIndex(available).as_unit("ns").asi8
    origin_day = origin_ns.astype(np.float64) / NANOSECONDS_PER_DAY
    available_day = available_ns.astype(np.float64) / NANOSECONDS_PER_DAY
    if np.any(available_day < origin_day):
        raise ValueError("earthquake availability cannot precede origin time")

    longitude = frame["longitude"].to_numpy(dtype=np.float64)
    latitude = frame["latitude"].to_numpy(dtype=np.float64)
    magnitude = frame["magnitude"].to_numpy(dtype=np.float64)
    if not (
        np.isfinite(longitude).all()
        and np.isfinite(latitude).all()
        and np.isfinite(magnitude).all()
    ):
        raise ValueError("earthquake coordinates and magnitudes must be finite")
    if np.any((longitude < -180.0) | (longitude > 180.0)) or np.any(
        (latitude < -90.0) | (latitude > 90.0)
    ):
        raise ValueError("earthquake coordinates are outside geographic bounds")

    transformer = Transformer.from_crs(
        CRS.from_epsg(4326), CRS.from_user_input(study_area.equal_area_crs), always_xy=True
    )
    x_m_raw, y_m_raw = transformer.transform(longitude, latitude)
    x_m = np.asarray(x_m_raw, dtype=np.float64)
    y_m = np.asarray(y_m_raw, dtype=np.float64)
    point_array = points(x_m, y_m)
    inside_computed = np.asarray(covers(study_area.projected, point_array), dtype=np.bool_)
    inside_recorded = frame["inside_study_area"].to_numpy(dtype=np.bool_)
    if not np.array_equal(inside_computed, inside_recorded):
        mismatch_count = int(np.count_nonzero(inside_computed != inside_recorded))
        raise ValueError(
            f"earthquake study-area flags disagree with the frozen polygon: {mismatch_count} rows"
        )
    buffer_geometry = study_area.projected.buffer(buffer_km * 1_000.0)
    inside_external_buffer = np.asarray(covers(buffer_geometry, point_array), dtype=np.bool_)

    order = np.lexsort((event_id, origin_day))
    if not np.array_equal(order, np.arange(len(frame))):
        raise ValueError("earthquake dataset must remain sorted by origin_time_utc and event_id")

    return EarthquakeCatalog(
        event_id=cast(NDArray[np.str_], _readonly(event_id)),
        origin_day=cast(NDArray[np.float64], _readonly(origin_day)),
        available_day=cast(NDArray[np.float64], _readonly(available_day)),
        longitude=cast(NDArray[np.float64], _readonly(longitude)),
        latitude=cast(NDArray[np.float64], _readonly(latitude)),
        x_km=cast(NDArray[np.float64], _readonly(x_m / 1_000.0)),
        y_km=cast(NDArray[np.float64], _readonly(y_m / 1_000.0)),
        magnitude=cast(NDArray[np.float64], _readonly(magnitude)),
        inside_study_area=cast(NDArray[np.bool_], _readonly(inside_recorded)),
        inside_external_buffer=cast(NDArray[np.bool_], _readonly(inside_external_buffer)),
    )
