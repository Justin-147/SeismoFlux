"""Synthetic geometry and cell-mass checks; no real catalogs or fault maps."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pyproj import Transformer
from scipy.special import logsumexp
from shapely import to_wkb
from shapely.geometry import LineString, Point

from seismoflux.background.grid import EQUAL_AREA_CRS
from seismoflux.d1_replay.spatial import D1SpatialDomain
from seismoflux.features.anomaly.grid import Stage3QueryGrid
from seismoflux.multitask_s2.fault_geometry import (
    _minimum_projected_distance_km,
    _project_lines_to_equal_area,
    _source_lines,
    _verified_payload,
    blend_log_masses,
    coarsen_log_mass,
    load_fault_surfaces,
    proximity_log_mass,
)
from seismoflux.stage2s.contracts import SpatialGrid
from seismoflux.stage2s.inputs import EXPECTED_MAPPING_FIELDS


def test_proximity_matches_formula_and_actual_area_not_cell_count() -> None:
    distance = np.array([0.0, 25.0, 50.0])
    area = np.array([1.0, 3.0, 2.0])
    result = proximity_log_mass(distance, area, 25.0)
    expected = np.array([1.0, 1.5, 0.4])
    expected /= expected.sum()
    assert_allclose(np.exp(result), expected, rtol=1.0e-14)
    assert_allclose(logsumexp(result), 0.0, atol=1.0e-14)
    assert not result.flags.writeable
    uniform_distance = proximity_log_mass(np.zeros(3), area, 75.0)
    assert_allclose(np.exp(uniform_distance), area / area.sum())


def test_proximity_extremes_stay_finite_without_an_artificial_floor() -> None:
    result = proximity_log_mass(np.array([0.0, 1.0e308]), np.array([1.0e-300, 1.0e300]), 1.0e-300)
    assert np.isfinite(result).all()
    assert_allclose(logsumexp(result), 0.0, atol=1.0e-12)
    assert result[1] < -1_000.0


def test_pure_fault_density_ranking_does_not_change_with_scale() -> None:
    distance = np.array([50.0, 0.0, 150.0, 25.0])
    area = np.array([1.0, 4.0, 2.0, 3.0])
    for scale in (25.0, 75.0, 150.0):
        log_density = proximity_log_mass(distance, area, scale) - np.log(area)
        assert_array_equal(np.argsort(-log_density, kind="stable"), np.array([1, 3, 0, 2]))


@pytest.mark.parametrize("area", [[0.0, 1.0], [-1.0, 1.0], [np.nan, 1.0], [np.inf, 1.0], [1.0]])
def test_nonpositive_nonfinite_or_unaligned_areas_rejected(area: list[float]) -> None:
    with pytest.raises(ValueError, match="areas"):
        proximity_log_mass(np.array([0.0, 1.0]), np.array(area), 25.0)
    with pytest.raises(ValueError, match="areas"):
        coarsen_log_mass(np.log([0.5, 0.5]), np.array(area), np.array([0, 0]))


@pytest.mark.parametrize("distance", [[-1.0, 0.0], [np.inf, 0.0], [np.nan, 0.0], []])
def test_bad_distances_rejected(distance: list[float]) -> None:
    with pytest.raises(ValueError):
        proximity_log_mass(np.array(distance), np.ones(len(distance)), 25.0)


@pytest.mark.parametrize("scale", [0.0, -1.0, np.inf, np.nan])
def test_bad_proximity_scale_rejected(scale: float) -> None:
    with pytest.raises(ValueError):
        proximity_log_mass(np.array([1.0]), np.array([1.0]), scale)


def test_coarsening_preserves_every_block_mass_and_only_removes_local_detail() -> None:
    mass = np.array([0.05, 0.15, 0.30, 0.50])
    area = np.array([1.0, 3.0, 1.0, 4.0])
    blocks = np.array([0, 1, 0, 1])
    result = coarsen_log_mass(np.log(mass), area, blocks)
    assert_allclose(np.exp(result), [0.175, 0.65 * 3 / 7, 0.175, 0.65 * 4 / 7])
    for block in (0, 1):
        selected = blocks == block
        assert_allclose(np.exp(result[selected]).sum(), mass[selected].sum(), rtol=1.0e-14)
        density = np.exp(result[selected]) / area[selected]
        assert_allclose(density, np.full(density.shape, density[0]), rtol=1.0e-14)
    assert_allclose(logsumexp(result), 0.0, atol=1.0e-14)
    assert not result.flags.writeable


def test_coarsening_keeps_log_mass_even_when_exp_underflows() -> None:
    log_mass = np.array([-10000.0, -10001.0, 0.0])
    result = coarsen_log_mass(log_mass, np.array([1.0, 2.0, 1.0]), np.array([0, 0, 1]))
    assert np.isfinite(result).all()
    assert_allclose(logsumexp(result[:2]), logsumexp(log_mass[:2]), atol=1.0e-12)


@pytest.mark.parametrize("blocks", [[0.0, 1.0], [True, False], [-1, 0], [0]])
def test_bad_block_indices_rejected(blocks: list[Any]) -> None:
    with pytest.raises(ValueError, match="block indices"):
        coarsen_log_mass(np.log([0.4, 0.6]), np.ones(2), np.asarray(blocks))


def test_blend_endpoints_are_bitwise_original_and_middle_is_linear_mass() -> None:
    catalog = np.array([-1000.0, -0.0], dtype=np.float64)
    fault = np.log(np.array([0.8, 0.2]))
    assert blend_log_masses(catalog, fault, 0.0).tobytes() == catalog.tobytes()
    assert blend_log_masses(catalog, fault, 1.0).tobytes() == fault.tobytes()
    middle = blend_log_masses(catalog, fault, 0.25)
    assert_allclose(np.exp(middle), 0.75 * np.exp(catalog) + 0.25 * np.exp(fault))
    assert_allclose(logsumexp(middle), 0.0, atol=1.0e-14)
    assert not middle.flags.writeable
    assert catalog[1].tobytes() == np.float64(-0.0).tobytes()


@pytest.mark.parametrize("alpha", [-0.01, 1.01, np.inf, np.nan])
def test_bad_blend_weight_rejected(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        blend_log_masses(np.log([0.4, 0.6]), np.log([0.2, 0.8]), alpha)


def test_non_normalized_or_misaligned_log_masses_rejected() -> None:
    with pytest.raises(ValueError, match="normalized"):
        blend_log_masses(np.zeros(2), np.log([0.4, 0.6]), 0.0)
    with pytest.raises(ValueError, match="normalized"):
        coarsen_log_mass(np.zeros(2), np.ones(2), np.array([0, 1]))
    with pytest.raises(ValueError, match="same shape"):
        blend_log_masses(np.zeros(1), np.log([0.4, 0.6]), 0.0)
    with pytest.raises(ValueError, match="finite"):
        blend_log_masses(np.array([-np.inf, 0.0]), np.log([0.4, 0.6]), 0.0)


def test_nearest_lines_are_in_metres_queries_in_km_and_duplicates_do_not_count() -> None:
    line = LineString([(0.0, 0.0), (0.0, 10000.0)])
    farther = LineString([(20000.0, 0.0), (20000.0, 10000.0)])
    query = np.array([[3.0, 4.0], [0.0, 11.0], [19.0, 4.0]])
    expected = np.array([3.0, 1.0, 1.0])
    result = _minimum_projected_distance_km((line, farther), query)
    duplicates = _minimum_projected_distance_km((line, line, farther, farther), query)
    assert_allclose(result, expected, rtol=0.0, atol=1.0e-14)
    assert_array_equal(result, duplicates)


def test_projection_uses_original_wgs84_albers_and_not_degree_distances() -> None:
    line = LineString([(105.0, 30.0), (105.0, 31.0)])
    projected = _project_lines_to_equal_area((line,))
    transformer = Transformer.from_crs("EPSG:4326", EQUAL_AREA_CRS, always_xy=True)
    expected = np.array([transformer.transform(*point) for point in line.coords])
    assert_allclose(np.asarray(projected[0].coords), expected, rtol=0.0, atol=1.0e-8)
    query_m = np.array(transformer.transform(105.5, 30.5))
    expected_km = Point(query_m).distance(projected[0]) / 1000.0
    result = _minimum_projected_distance_km(projected, query_m.reshape(1, 2) / 1000.0)
    assert_allclose(result, [expected_km], rtol=1.0e-13)
    assert 40.0 < result[0] < 55.0
    with pytest.raises(ValueError, match="LineString"):
        _project_lines_to_equal_area((Point(105.0, 30.0),))


def _parquet_bytes(table: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    return cast(bytes, sink.getvalue().to_pybytes())


def _synthetic_inputs(tmp_path: Path) -> tuple[dict[str, Any], D1SpatialDomain]:
    transformer = Transformer.from_crs("EPSG:4326", EQUAL_AREA_CRS, always_xy=True)
    query_m = np.asarray([transformer.transform(lon, 30.5) for lon in (105.0, 105.2, 107.0)])
    stage3 = Stage3QueryGrid(
        grid_id="synthetic-grid",
        equal_area_crs=EQUAL_AREA_CRS,
        cell_size_km=25.0,
        cell_ids=("cell-a", "cell-b", "cell-c"),
        rows=np.zeros(3, dtype=np.int64),
        columns=np.arange(3, dtype=np.int64),
        query_xy_m=query_m,
        clipped_area_km2=np.array([1.0, 3.0, 2.0]),
    )
    grid = SpatialGrid(
        grid_id=stage3.grid_id,
        cell_size_km=25.0,
        cell_ids=stage3.cell_ids,
        rows=stage3.rows,
        columns=stage3.columns,
        query_xy_km=query_m / 1000.0,
        clipped_area_km2=stage3.clipped_area_km2,
    )
    domain = cast(D1SpatialDomain, SimpleNamespace(operational_grid=grid, stage3_grid=stage3))
    mapping = pa.Table.from_arrays(
        [
            pa.array([stage3.grid_id] * 3),
            pa.array(stage3.cell_ids),
            pa.array(stage3.rows),
            pa.array(stage3.columns),
            pa.array(query_m[:, 0]),
            pa.array(query_m[:, 1]),
            pa.array(["乙", "A", "乙"]),
        ],
        schema=pa.schema(EXPECTED_MAPPING_FIELDS),
    )
    sources = {}
    line = to_wkb(LineString([(105.0, 30.0), (105.0, 31.0)]))
    zero_length = to_wkb(LineString([(105.0, 30.0), (105.0, 30.0)]))
    for source, id_col, geom_col, usable_col in (
        ("SIMPLE", "fault_segment_id", "simplified_geometry_wkb", None),
        ("TRACE", "trace_id", "geometry_wkb", "usable_for_geometry"),
    ):
        values = {id_col: ["a", "b"], geom_col: [line, line]}
        columns = [id_col, geom_col]
        if usable_col is not None:
            values[id_col].append("zero")
            values[geom_col].append(zero_length)
            values[usable_col] = [True, True, False]
            columns.append(usable_col)
        values["long_term_hazard_score"] = [999.0] * len(values[id_col])
        payload = _parquet_bytes(pa.table(values))
        relative = f"{source}.parquet"
        (tmp_path / relative).write_bytes(payload)
        sources[source] = {
            "path": relative,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "id_column": id_col,
            "geometry_column": geom_col,
            "usable_column": usable_col,
            "allowed_columns": columns,
            "expected_rows": len(values[id_col]),
            "expected_usable_lines": 2,
        }
    mapping_payload = _parquet_bytes(mapping)
    (tmp_path / "mapping.parquet").write_bytes(mapping_payload)
    protocol = {
        "scientific_role": "synthetic_retrospective",
        "inputs": {
            "grid_id": stage3.grid_id,
            "grid_cells": 3,
            "grid_cell_size_km": 25.0,
            "grid_area_km2": 6.0,
            "geometry_sources": sources,
            "static_snapshot_available_at": "2026-07-12T16:00:00+00:00",
            "original_historical_model_eligible": False,
            "cell_to_block_mapping": {
                "path": "mapping.parquet",
                "sha256": hashlib.sha256(mapping_payload).hexdigest(),
                "blocks": 2,
            },
        },
        "geometry_math": {
            "scales_km": [25.0, 75.0, 150.0],
            "power": 2.0,
            "source_crs_assumption": "EPSG_4326_WGS84_existing_interpretation",
        },
    }
    return protocol, domain


def test_synthetic_loader_uses_white_list_mapping_order_and_original_usability(
    tmp_path: Path,
) -> None:
    protocol, domain = _synthetic_inputs(tmp_path)
    surfaces = load_fault_surfaces(tmp_path, protocol, domain)
    assert_array_equal(surfaces.block_index, [1, 0, 1])
    assert surfaces.audit["zone_ids_utf8_order"] == ["A", "乙"]
    assert surfaces.audit["sources"]["TRACE"]["excluded_by_existing_usability_flag"] == 1
    assert not surfaces.audit["sources"]["SIMPLE"]["attribute_fields_read"]
    assert not surfaces.audit["target_coordinates_read"]
    assert not surfaces.audit["original_historical_model_eligible"]
    assert not surfaces.block_index.flags.writeable
    json.dumps(surfaces.audit, allow_nan=False)
    for scale in (25.0, 75.0, 150.0):
        assert_array_equal(surfaces.fine["SIMPLE"][scale], surfaces.fine["TRACE"][scale])
        for representation in (surfaces.fine, surfaces.coarse):
            assert_allclose(logsumexp(representation["SIMPLE"][scale]), 0.0, atol=1.0e-14)


def test_loader_refuses_changed_mapping_or_source_hash_and_grid(tmp_path: Path) -> None:
    protocol, domain = _synthetic_inputs(tmp_path)
    protocol["inputs"]["grid_cells"] = 4
    with pytest.raises(ValueError, match="grid differs"):
        load_fault_surfaces(tmp_path, protocol, domain)
    protocol["inputs"]["grid_cells"] = 3
    protocol["inputs"]["cell_to_block_mapping"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        load_fault_surfaces(tmp_path, protocol, domain)
    protocol, domain = _synthetic_inputs(tmp_path)
    protocol["inputs"]["geometry_sources"]["TRACE"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        load_fault_surfaces(tmp_path, protocol, domain)


def test_geometry_whitelist_cannot_expand_to_attributes(tmp_path: Path) -> None:
    protocol, _ = _synthetic_inputs(tmp_path)
    source = protocol["inputs"]["geometry_sources"]["SIMPLE"]
    source["allowed_columns"].append("long_term_hazard_score")
    with pytest.raises(ValueError, match="allowlist"):
        _source_lines((tmp_path / source["path"]).read_bytes(), "SIMPLE", source)


def test_source_rejects_wrong_row_count_or_false_usable_count(tmp_path: Path) -> None:
    protocol, _ = _synthetic_inputs(tmp_path)
    source = protocol["inputs"]["geometry_sources"]["TRACE"]
    payload = (tmp_path / source["path"]).read_bytes()
    source["expected_rows"] = 4
    with pytest.raises(ValueError, match="row count"):
        _source_lines(payload, "TRACE", source)
    source["expected_rows"] = 3
    source["expected_usable_lines"] = 3
    with pytest.raises(ValueError, match="usable geometry count"):
        _source_lines(payload, "TRACE", source)


def test_input_path_cannot_escape_data_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="data root"):
        _verified_payload(tmp_path, "../outside.parquet", "0" * 64)


@pytest.mark.parametrize("null_row_is_usable", [False, True])
def test_null_wkb_is_allowed_only_when_existing_flag_excludes_row(
    null_row_is_usable: bool,
) -> None:
    line = to_wkb(LineString([(105.0, 30.0), (105.0, 31.0)]))
    table = pa.table(
        {
            "trace_id": ["valid-line", "null-geometry"],
            "geometry_wkb": [line, None],
            "usable_for_geometry": [True, null_row_is_usable],
        }
    )
    specification = {
        "allowed_columns": ["trace_id", "geometry_wkb", "usable_for_geometry"],
        "id_column": "trace_id",
        "geometry_column": "geometry_wkb",
        "usable_column": "usable_for_geometry",
        "expected_rows": 2,
        "expected_usable_lines": 2 if null_row_is_usable else 1,
    }
    if null_row_is_usable:
        with pytest.raises(ValueError, match="usable geometry WKB must not be null"):
            _source_lines(_parquet_bytes(table), "TRACE", specification)
    else:
        lines, audit = _source_lines(_parquet_bytes(table), "TRACE", specification)
        assert len(lines) == 1
        assert lines[0].equals(LineString([(105.0, 30.0), (105.0, 31.0)]))
        assert audit["excluded_by_existing_usability_flag"] == 1
