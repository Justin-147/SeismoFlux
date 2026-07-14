from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from seismoflux.background.catalog import (
    EARTHQUAKE_COLUMN_ALLOWLIST,
    load_earthquake_catalog,
    load_study_area,
    utc_timestamp_to_day,
)

EQUAL_AREA_CRS = (
    "+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 +datum=WGS84 +units=m +no_defs +type=crs"
)


def _write_study_area(path: Path) -> None:
    document = {
        "type": "Feature",
        "properties": {},
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [[99.0, 29.0], [101.0, 29.0], [101.0, 31.0], [99.0, 31.0], [99.0, 29.0]]
            ],
        },
    }
    path.write_text(json.dumps(document), encoding="utf-8")


def _write_catalog(path: Path, *, flip_inside: bool = False) -> None:
    inside = [not flip_inside, False, False]
    frame = pd.DataFrame(
        {
            "event_id": ["event-1", "event-2", "event-3"],
            "origin_time_utc": pd.to_datetime(
                ["2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z", "2020-01-03T00:00:00Z"],
                utc=True,
            ),
            "available_at": pd.to_datetime(
                ["2020-01-01T00:00:00Z", "2020-01-03T00:00:00Z", "2020-01-03T00:00:00Z"],
                utc=True,
            ),
            "longitude": [100.0, 102.0, 120.0],
            "latitude": [30.0, 30.0, 30.0],
            "magnitude": [3.5, 4.0, 5.0],
            "inside_study_area": inside,
            "future_label_forbidden": [1, 1, 1],
        }
    )
    frame.to_parquet(path, index=False)


def test_catalog_loader_uses_only_allowlisted_columns_and_fixed_buffer(tmp_path: Path) -> None:
    study_path = tmp_path / "study.geojson"
    catalog_path = tmp_path / "earthquakes.parquet"
    _write_study_area(study_path)
    _write_catalog(catalog_path)
    study_area = load_study_area(study_path, EQUAL_AREA_CRS)

    catalog = load_earthquake_catalog(
        catalog_path,
        study_area=study_area,
        external_buffer_km=300.0,
        maximum_event_time_utc="2020-01-04T00:00:00Z",
    )

    assert len(catalog) == 3
    assert catalog.inside_study_area.tolist() == [True, False, False]
    assert catalog.inside_external_buffer.tolist() == [True, True, False]
    assert set(EARTHQUAKE_COLUMN_ALLOWLIST).isdisjoint({"future_label_forbidden"})
    assert study_area.area_km2 > 0.0
    assert not catalog.magnitude.flags.writeable


def test_catalog_eligibility_uses_available_at_and_preserves_order(tmp_path: Path) -> None:
    study_path = tmp_path / "study.geojson"
    catalog_path = tmp_path / "earthquakes.parquet"
    _write_study_area(study_path)
    _write_catalog(catalog_path)
    study_area = load_study_area(study_path, EQUAL_AREA_CRS)
    catalog = load_earthquake_catalog(
        catalog_path,
        study_area=study_area,
        external_buffer_km=300.0,
        maximum_event_time_utc="2020-01-04T00:00:00Z",
    )

    cutoff = utc_timestamp_to_day("2020-01-02T12:00:00Z")
    inside = catalog.eligible(
        available_through_day=cutoff,
        minimum_magnitude=3.0,
        inside_only=True,
    )
    parents = catalog.eligible(
        available_through_day=cutoff,
        minimum_magnitude=3.0,
        inside_only=False,
    )

    assert inside.event_id.tolist() == ["event-1"]
    assert parents.event_id.tolist() == ["event-1"]
    assert np.all(np.diff(catalog.origin_day) > 0.0)


def test_catalog_rejects_spatial_flag_drift(tmp_path: Path) -> None:
    study_path = tmp_path / "study.geojson"
    catalog_path = tmp_path / "earthquakes.parquet"
    _write_study_area(study_path)
    _write_catalog(catalog_path, flip_inside=True)
    study_area = load_study_area(study_path, EQUAL_AREA_CRS)

    with pytest.raises(ValueError, match="flags disagree"):
        load_earthquake_catalog(
            catalog_path,
            study_area=study_area,
            external_buffer_km=300.0,
            maximum_event_time_utc="2020-01-04T00:00:00Z",
        )


def test_catalog_loader_excludes_rows_beyond_maturity_before_catalog_creation(
    tmp_path: Path,
) -> None:
    study_path = tmp_path / "study.geojson"
    catalog_path = tmp_path / "earthquakes.parquet"
    _write_study_area(study_path)
    _write_catalog(catalog_path)
    study_area = load_study_area(study_path, EQUAL_AREA_CRS)

    catalog = load_earthquake_catalog(
        catalog_path,
        study_area=study_area,
        external_buffer_km=300.0,
        maximum_event_time_utc="2020-01-02T12:00:00Z",
    )

    assert catalog.event_id.tolist() == ["event-1"]
    cutoff = utc_timestamp_to_day("2020-01-02T12:00:00Z")
    assert np.all(catalog.origin_day <= cutoff)
    assert np.all(catalog.available_day <= cutoff)


def test_catalog_loader_requires_timezone_aware_maturity_cutoff(tmp_path: Path) -> None:
    study_path = tmp_path / "study.geojson"
    catalog_path = tmp_path / "earthquakes.parquet"
    _write_study_area(study_path)
    _write_catalog(catalog_path)
    study_area = load_study_area(study_path, EQUAL_AREA_CRS)

    with pytest.raises(ValueError, match="maximum earthquake event time"):
        load_earthquake_catalog(
            catalog_path,
            study_area=study_area,
            external_buffer_km=300.0,
            maximum_event_time_utc="2020-01-04T00:00:00",
        )


def test_timestamp_conversion_requires_timezone() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        utc_timestamp_to_day("2020-01-01T00:00:00")
