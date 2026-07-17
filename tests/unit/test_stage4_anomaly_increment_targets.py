from __future__ import annotations

import inspect
from datetime import UTC, datetime, timedelta

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from shapely.geometry import box

from seismoflux.anomaly_increment.grid_features import build_stage4_integration_grid
from seismoflux.anomaly_increment.runner import ExposurePlan
from seismoflux.anomaly_increment.targets import (
    exposure_target_view,
    map_targets_to_frozen_primary_grid,
    parse_authorized_stage4_target_bytes,
)
from seismoflux.background.catalog import StudyArea
from seismoflux.background.grid import EQUAL_AREA_CRS, project_study_area_to_equal_area
from seismoflux.data.parquet import schema_sha256, table_content_sha256


def _study_area() -> StudyArea:
    geographic = box(105.0, 34.0, 105.1, 34.1)
    projected = project_study_area_to_equal_area(geographic)
    return StudyArea(
        geographic=geographic,
        projected=projected,
        equal_area_crs=EQUAL_AREA_CRS,
        area_km2=float(projected.area) / 1_000_000.0,
    )


def _target_payload() -> tuple[bytes, str, str]:
    origins = [
        datetime(2025, 4, 3, 12, tzinfo=UTC),
        datetime(2025, 4, 5, 0, tzinfo=UTC),
        datetime(2025, 4, 6, 0, tzinfo=UTC),
    ]
    table = pa.table(
        {
            "event_id": pa.array(["event-1", "event-2", "event-outside"], pa.string()),
            "origin_time_utc": pa.array(origins, pa.timestamp("us", tz="UTC")),
            "available_at": pa.array(
                [value + timedelta(hours=1) for value in origins],
                pa.timestamp("us", tz="UTC"),
            ),
            "longitude": pa.array([105.02, 105.08, 110.0], pa.float64()),
            "latitude": pa.array([34.02, 34.08, 40.0], pa.float64()),
            "magnitude": pa.array([5.5, 6.2, 5.8], pa.float64()),
            "inside_study_area": pa.array([True, True, False], pa.bool_()),
        }
    )
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    payload = sink.getvalue().to_pybytes()
    persisted = pq.read_table(pa.BufferReader(payload))
    return payload, table_content_sha256(persisted), schema_sha256(persisted.schema)


def test_target_parser_accepts_only_bytes_and_verifies_arrow_identities() -> None:
    parameters = inspect.signature(parse_authorized_stage4_target_bytes).parameters
    assert tuple(parameters) == (
        "payload",
        "expected_content_sha256",
        "expected_schema_sha256",
        "study_area",
    )
    payload, content, schema = _target_payload()
    catalog = parse_authorized_stage4_target_bytes(
        payload,
        expected_content_sha256=content,
        expected_schema_sha256=schema,
        study_area=_study_area(),
    )

    assert len(catalog) == 3
    assert catalog.event_id.tolist() == ["event-1", "event-2", "event-outside"]
    assert catalog.inside_study_area.tolist() == [True, True, False]
    assert catalog.event_id.flags.writeable is False


def test_target_parser_fails_closed_on_content_or_inside_flag_drift() -> None:
    payload, content, schema = _target_payload()
    with pytest.raises(ValueError, match="content hash"):
        parse_authorized_stage4_target_bytes(
            payload,
            expected_content_sha256="0" * 64,
            expected_schema_sha256=schema,
            study_area=_study_area(),
        )

    table = pq.read_table(pa.BufferReader(payload)).set_column(
        6,
        "inside_study_area",
        pa.array([False, True, False], pa.bool_()),
    )
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    changed = sink.getvalue().to_pybytes()
    persisted = pq.read_table(pa.BufferReader(changed))
    with pytest.raises(ValueError, match="inside flags"):
        parse_authorized_stage4_target_bytes(
            changed,
            expected_content_sha256=table_content_sha256(persisted),
            expected_schema_sha256=schema_sha256(persisted.schema),
            study_area=_study_area(),
        )
    assert content != table_content_sha256(persisted)


def test_exposure_selection_uses_strict_open_closed_time_and_magnitude_bins() -> None:
    payload, content, schema = _target_payload()
    catalog = parse_authorized_stage4_target_bytes(
        payload,
        expected_content_sha256=content,
        expected_schema_sha256=schema,
        study_area=_study_area(),
    )
    exposure = ExposurePlan.parse("validation-h007-2025-04-03")

    m5 = exposure_target_view(catalog, exposure, magnitude_bin_id="M5_6")
    m6 = exposure_target_view(catalog, exposure, magnitude_bin_id="M6_plus")
    assert m5.event_ids == ("event-1",)
    assert m6.event_ids == ("event-2",)
    assert 0.0 < float(m5.lead_days[0]) <= 7.0
    assert 0.0 < float(m6.lead_days[0]) <= 7.0


def test_true_locations_only_map_to_an_already_frozen_25km_grid() -> None:
    study_area = _study_area()
    grid = build_stage4_integration_grid(study_area.geographic, cell_size_km=25.0)
    frozen_grid_id = grid.grid_id
    frozen_cells = grid.cell_ids
    payload, content, schema = _target_payload()
    catalog = parse_authorized_stage4_target_bytes(
        payload,
        expected_content_sha256=content,
        expected_schema_sha256=schema,
        study_area=study_area,
    )

    assignments = map_targets_to_frozen_primary_grid(
        catalog,
        study_area=study_area,
        primary_grid=grid,
    )
    assert assignments.event_ids == ("event-1", "event-2")
    assert assignments.grid_id == frozen_grid_id
    assert all(cell_id in frozen_cells for cell_id in assignments.cell_ids)
    assert grid.grid_id == frozen_grid_id
    assert grid.cell_ids == frozen_cells
