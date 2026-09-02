"""Synthetic S2-B line/rate checks; no real faults, catalogs, or outcomes."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pyproj import Transformer
from scipy.special import logsumexp, ndtr
from shapely import to_wkb
from shapely.geometry import LineString

from seismoflux.background.grid import EQUAL_AREA_CRS
from seismoflux.d1_replay.spatial import D1SpatialDomain
from seismoflux.features.anomaly.grid import Stage3QueryGrid
from seismoflux.multitask_s2.slip_rate import (
    GD_RATE,
    GEO_RATE,
    _integration_diagnostics,
    _source_attributes,
    gaussian_line_log_masses,
    load_slip_rate_surfaces,
    midpoint_quadrature,
)
from seismoflux.stage2s.contracts import SpatialGrid


def test_midpoint_rule_preserves_lengths_zero_edges_and_metres_to_km() -> None:
    line = LineString([(0, 0), (0, 0), (6250, 0), (6250, 3125)])
    quadrature = midpoint_quadrature((line,), 3.125)
    assert_array_equal(quadrature.points_xy_km, [[1.5625, 0], [4.6875, 0], [6.25, 1.5625]])
    assert_array_equal(quadrature.represented_length_km, [3.125, 3.125, 3.125])
    assert_array_equal(quadrature.line_index, [0, 0, 0])
    assert_allclose(quadrature.line_length_km, [9.375], rtol=0.0, atol=0.0)
    assert_allclose(quadrature.represented_length_km.sum() * 7.0, 9.375 * 7.0)
    assert not quadrature.points_xy_km.flags.writeable
    assert not quadrature.line_index.flags.writeable


def _line_formula(query: np.ndarray, scale: float) -> np.ndarray:
    """Exact finite line integral for the synthetic x=[0,100] km, y=0 line."""

    along = query[:, 0]
    cross = query[:, 1]
    cdf_difference = ndtr((100.0 - along) / scale) - ndtr(-along / scale)
    return np.exp(-0.5 * (cross / scale) ** 2) * cdf_difference / (math.sqrt(2 * math.pi) * scale)


def test_gaussian_midpoint_matches_finite_line_integral_at_inside_and_outside_points() -> None:
    quadrature = midpoint_quadrature((LineString([(0, 0), (100000, 0)]),), 3.125)
    query = np.array([[0.0, 0.0], [50.0, 30.0], [100.0, 0.0], [-40.0, 10.0], [140.0, 10.0]])
    area = np.array([1.0, 2.0, 3.0, 4.0, 2.0])
    result = gaussian_line_log_masses(query, area, quadrature, {"unit": np.ones(1)}, (25.0,))
    expected = _line_formula(query, 25.0) * area
    expected /= expected.sum()
    # This checks the fixed quadrature approximation, not a new model-selection gate.
    assert_allclose(np.exp(result["unit"][25.0]), expected, rtol=0.002, atol=0.0)
    assert_allclose(logsumexp(result["unit"][25.0]), 0.0, atol=1.0e-14)


def test_length_weighting_is_invariant_to_aligned_line_splitting_not_segment_count() -> None:
    whole = (LineString([(0, 0), (100000, 0)]), LineString([(0, 100000), (50000, 100000)]))
    split = (
        LineString([(0, 0), (50000, 0)]),
        LineString([(50000, 0), (100000, 0)]),
        whole[1],
    )
    query = np.array([[20.0, 0.0], [50.0, 25.0], [20.0, 100.0]])
    area = np.array([2.0, 1.0, 3.0])
    whole_quad = midpoint_quadrature(whole, 3.125)
    split_quad = midpoint_quadrature(split, 3.125)
    assert_allclose(whole_quad.line_length_km, [100.0, 50.0])
    assert_allclose(split_quad.line_length_km, [50.0, 50.0, 50.0])
    a = gaussian_line_log_masses(
        query, area, whole_quad, {"weighted": np.array([4.0, 2.0])}, (25.0,)
    )
    b = gaussian_line_log_masses(
        query, area, split_quad, {"weighted": np.array([4.0, 4.0, 2.0])}, (25.0,)
    )
    assert_allclose(a["weighted"][25.0], b["weighted"][25.0], rtol=0.0, atol=2.0e-14)


def test_constant_rates_unit_conversion_and_actual_area_normalization() -> None:
    lines = (LineString([(0, 0), (40000, 0)]), LineString([(0, 50000), (20000, 50000)]))
    quadrature = midpoint_quadrature(lines, 3.125)
    query = np.array([[0.0, 0.0], [25.0, 20.0], [100.0, 100.0]])
    area = np.array([1.0, 3.0, 2.0])
    fields = gaussian_line_log_masses(
        query,
        area,
        quadrature,
        {
            "unit": np.ones(2),
            "constant_rate": np.full(2, 7.0),
            "rates": np.array([2.0, 9.0]),
            "converted_rates": np.array([2.0, 9.0]) * 1.0e-6,
        },
        (25.0, 75.0, 150.0),
    )
    for scale in (25.0, 75.0, 150.0):
        assert_allclose(fields["unit"][scale], fields["constant_rate"][scale], atol=2.0e-14)
        assert_allclose(fields["rates"][scale], fields["converted_rates"][scale], atol=2.0e-14)
        for layer in fields.values():
            assert_allclose(np.exp(layer[scale]).sum(), 1.0, atol=2.0e-14)
            assert not layer[scale].flags.writeable
    same_points = np.repeat([[25.0, 0.0]], 3, axis=0)
    equal_density = gaussian_line_log_masses(
        same_points, area, quadrature, {"unit": np.ones(2)}, (25.0,)
    )
    assert_allclose(np.exp(equal_density["unit"][25.0]), area / area.sum(), atol=1.0e-14)


def test_far_tails_and_chunks_preserve_finite_log_mass_without_floor() -> None:
    lines = (LineString([(0, 0), (50000, 0)]), LineString([(0, 100000), (100000, 100000)]))
    quadrature = midpoint_quadrature(lines, 3.125)
    query = np.array([[0.0, 0.0], [10_000.0, 10_000.0], [100_000.0, 100_000.0]])
    weights = {"common": np.array([2.0, np.nan]), "native": np.array([2.0, 3.0])}
    small_chunks = gaussian_line_log_masses(
        query, np.ones(3), quadrature, weights, (25.0,), query_chunk_size=1, point_chunk_size=3
    )
    single_chunk = gaussian_line_log_masses(
        query, np.ones(3), quadrature, weights, (25.0,), query_chunk_size=50, point_chunk_size=1000
    )
    for layer in weights:
        result = small_chunks[layer][25.0]
        assert np.isfinite(result).all()
        assert result[-1] < -1_000_000.0
        assert np.exp(result[-1]) == 0.0  # Log mass, not an artificial floor, carries this tail.
        assert_allclose(logsumexp(result), 0.0, atol=1.0e-14)
        assert_allclose(result, single_chunk[layer][25.0], rtol=2.0e-15, atol=1.0e-10)


@pytest.mark.parametrize("weights", [[0.0], [-1.0], [np.inf], [np.nan], [1.0, 2.0]])
def test_invalid_or_unsupported_weights_cannot_be_silently_imputed(weights: list[float]) -> None:
    quadrature = midpoint_quadrature((LineString([(0, 0), (50000, 0)]),), 3.125)
    with pytest.raises(ValueError, match="weights|weight"):
        gaussian_line_log_masses(
            np.zeros((1, 2)), np.ones(1), quadrature, {"bad": np.array(weights)}, (25.0,)
        )


def _parquet_bytes(table: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    return cast(bytes, sink.getvalue().to_pybytes())


def _fixture(tmp_path: Path) -> tuple[dict[str, Any], D1SpatialDomain]:
    lines = [LineString([(lon, 30), (lon, 31)]) for lon in (105, 107, 109)]
    table = pa.table(
        {
            "fault_segment_id": ["common", "geodetic-only", "unknown"],
            "simplified_geometry_wkb": [to_wkb(line) for line in lines],
            GEO_RATE: [2.0, None, None],
            GD_RATE: [4.0, 8.0, None],
            "last_strong_earthquake_year_raw": [2050, 2040, 2030],
            "long_term_hazard_score": [1000.0, 2000.0, 3000.0],
        }
    )
    payload = _parquet_bytes(table)
    (tmp_path / "faults.parquet").write_bytes(payload)
    transformer = Transformer.from_crs("EPSG:4326", EQUAL_AREA_CRS, always_xy=True)
    query = np.asarray([transformer.transform(lon, 30.5) for lon in (105, 107, 109)])
    stage3 = Stage3QueryGrid(
        grid_id="synthetic",
        equal_area_crs=EQUAL_AREA_CRS,
        cell_size_km=25.0,
        cell_ids=("a", "b", "c"),
        rows=np.zeros(3, dtype=np.int64),
        columns=np.arange(3, dtype=np.int64),
        query_xy_m=query,
        clipped_area_km2=np.array([1.0, 3.0, 2.0]),
    )
    grid = SpatialGrid(
        grid_id="synthetic",
        cell_size_km=25.0,
        cell_ids=stage3.cell_ids,
        rows=stage3.rows,
        columns=stage3.columns,
        query_xy_km=query / 1_000.0,
        clipped_area_km2=stage3.clipped_area_km2,
    )
    domain = cast(D1SpatialDomain, SimpleNamespace(operational_grid=grid, stage3_grid=stage3))
    protocol: dict[str, Any] = {
        "scientific_role": "synthetic_current_static_rates",
        "inputs": {
            "grid_id": "synthetic",
            "grid_cells": 3,
            "grid_cell_size_km": 25.0,
            "grid_area_km2": 6.0,
            "static_snapshot_available_at": "2026-07-12T16:00:00+00:00",
            "original_historical_model_eligible": False,
            "fault_segments": {
                "path": "faults.parquet",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "id_column": "fault_segment_id",
                "geometry_column": "simplified_geometry_wkb",
                "allowed_columns": [
                    "fault_segment_id",
                    "simplified_geometry_wkb",
                    GEO_RATE,
                    GD_RATE,
                ],
                "expected_rows": 3,
                "expected_unique_ids": 3,
                "expected_usable_lines": 3,
                "rate_units": "mm_per_year",
                "expected_geologic_nonnull_and_positive": 1,
                "expected_geodetic_nonnull_and_positive": 2,
                "expected_both_nonnull": 1,
                "expected_geodetic_only": 1,
                "expected_both_missing": 1,
                "geologic_rate_range_mm_per_year": [2.0, 2.0],
                "geodetic_rate_range_mm_per_year": [4.0, 8.0],
            },
        },
        "panels": {
            "COMMON": {"required_nonnull_columns": [GEO_RATE, GD_RATE], "expected_segments": 1},
            "NATIVE_GD": {"required_nonnull_columns": [GD_RATE], "expected_segments": 2},
        },
        "layers": {
            "COMMON_UNIT": {"panel": "COMMON", "weight_column": None, "constant_weight": 1.0},
            "COMMON_GEO": {"panel": "COMMON", "weight_column": GEO_RATE, "constant_weight": None},
            "COMMON_GD": {"panel": "COMMON", "weight_column": GD_RATE, "constant_weight": None},
            "NATIVE_UNIT": {"panel": "NATIVE_GD", "weight_column": None, "constant_weight": 1.0},
            "NATIVE_GD": {"panel": "NATIVE_GD", "weight_column": GD_RATE, "constant_weight": None},
        },
        "geometry_math": {
            "scales_km": [25.0, 75.0, 150.0],
            "source_crs_assumption": "existing_WGS84_interpretation",
        },
        "numerical_integration": {
            "method": "midpoint_quadrature_on_each_original_projected_straight_edge",
            "production_maximum_step_km": 3.125,
            "diagnostic_maximum_step_km": 6.25,
            "diagnostic_runs": 1,
            "diagnostic_uses_earthquake_targets": False,
            "production_always_uses_fine_step": True,
        },
        "evaluation": {"area_budgets_km2": [1.0, 2.0, 3.0, 4.0, 6.0]},
    }
    return protocol, domain


def test_loader_common_native_support_all_grid_cells_and_one_fixed_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol, domain = _fixture(tmp_path)
    observed_reads: list[list[str]] = []
    observed_steps: list[float] = []
    original_read = pq.read_table
    original_quadrature = midpoint_quadrature

    def guarded_read(*args: Any, **kwargs: Any) -> pa.Table:
        observed_reads.append(kwargs["columns"])
        assert kwargs["use_threads"] is False
        return original_read(*args, **kwargs)

    def recorded_quadrature(lines: tuple[Any, ...], step: float) -> Any:
        observed_steps.append(step)
        return original_quadrature(lines, step)

    monkeypatch.setattr(pq, "read_table", guarded_read)
    monkeypatch.setattr(
        "seismoflux.multitask_s2.slip_rate.midpoint_quadrature", recorded_quadrature
    )
    surfaces = load_slip_rate_surfaces(tmp_path, protocol, domain)
    assert observed_reads == [["fault_segment_id", "simplified_geometry_wkb", GEO_RATE, GD_RATE]]
    assert observed_steps == [3.125, 6.25]
    assert set(surfaces.layers) == set(protocol["layers"])
    for scale in (25.0, 75.0, 150.0):
        assert_allclose(surfaces.layers["COMMON_UNIT"][scale], surfaces.layers["COMMON_GEO"][scale])
        assert_allclose(surfaces.layers["COMMON_UNIT"][scale], surfaces.layers["COMMON_GD"][scale])
        for fields in surfaces.layers.values():
            assert fields[scale].shape == (3,)
            assert np.isfinite(fields[scale]).all()
            assert not fields[scale].flags.writeable
            assert_allclose(logsumexp(fields[scale]), 0.0, atol=1.0e-14)
    assert not np.allclose(surfaces.layers["NATIVE_UNIT"][25.0], surfaces.layers["NATIVE_GD"][25.0])
    assert not np.allclose(surfaces.layers["COMMON_GD"][25.0], surfaces.layers["NATIVE_GD"][25.0])
    assert surfaces.audit["source"]["both_rates_missing_segments"] == 1
    assert surfaces.audit["layers"]["COMMON_UNIT"]["segments"] == 1
    assert surfaces.audit["layers"]["NATIVE_UNIT"]["segments"] == 2
    assert not surfaces.audit["target_coordinates_read"]
    assert not surfaces.audit["grid_or_targets_filtered_by_rate_support"]
    assert not surfaces.audit["original_historical_model_eligible"]
    assert not surfaces.audit["source_eligibility_flags_modified"]
    diagnostic = surfaces.audit["numerical_integration"]
    assert diagnostic["diagnostic_runs"] == 1
    assert diagnostic["national_percentage_agreement_gate"] is None
    assert len(diagnostic["diagnostics"]) == 15
    assert all(
        len(row["alarm_cell_differences_at_fixed_budgets"]) == 5
        for row in diagnostic["diagnostics"]
    )
    json.dumps(surfaces.audit, allow_nan=False)


def test_loader_always_returns_fine_not_coarse_fields(tmp_path: Path) -> None:
    protocol, domain = _fixture(tmp_path)
    fields = load_slip_rate_surfaces(tmp_path, protocol, domain)
    from seismoflux.multitask_s2.fault_geometry import _project_lines_to_equal_area

    source = _source_attributes(
        (tmp_path / "faults.parquet").read_bytes(), protocol["inputs"]["fault_segments"]
    )
    native = _project_lines_to_equal_area(source.lines[:2])
    quadrature = midpoint_quadrature(native, 3.125)
    expected = gaussian_line_log_masses(
        domain.operational_grid.query_xy_km,
        domain.operational_grid.clipped_area_km2,
        quadrature,
        {"NATIVE_GD": np.array([4.0, 8.0])},
        (25.0, 75.0, 150.0),
    )
    for scale in (25.0, 75.0, 150.0):
        assert_array_equal(fields.layers["NATIVE_GD"][scale], expected["NATIVE_GD"][scale])


def test_prefix_diagnostic_uses_density_ties_and_stops_before_overspending(tmp_path: Path) -> None:
    _, domain = _fixture(tmp_path)
    fine = {"layer": {25.0: np.log(np.array([0.6, 0.3, 0.1]))}}
    coarse = {"layer": {25.0: np.log(np.array([0.2, 0.6, 0.2]))}}
    result = _integration_diagnostics(fine, coarse, domain.operational_grid, [2.0, 4.0])
    assert_allclose(result[0]["normalized_layer_total_variation"], 0.4)
    small = result[0]["alarm_cell_differences_at_fixed_budgets"][0]
    # Fine picks a (area 1), then stops at b (area 3), never skipping to c.
    assert small["fine_actual_area_km2"] == 1.0
    assert small["fine_selected_cells"] == 1
    assert small["fine_actual_area_km2"] <= small["area_budget_km2"]


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("inputs", "grid_cells", 4, "grid differs"),
        ("geometry_math", "scales_km", [20.0, 75.0, 150.0], "layers or scales"),
        ("numerical_integration", "production_maximum_step_km", 6.25, "fixed fine"),
        ("numerical_integration", "diagnostic_runs", 2, "fixed fine"),
        ("numerical_integration", "diagnostic_uses_earthquake_targets", True, "target-free"),
    ],
)
def test_loader_rejects_changed_frozen_geometry_or_numeric_definition(
    tmp_path: Path, section: str, field: str, value: Any, message: str
) -> None:
    protocol, domain = _fixture(tmp_path)
    protocol[section][field] = value
    with pytest.raises(ValueError, match=message):
        load_slip_rate_surfaces(tmp_path, protocol, domain)


def test_input_hash_column_allowlist_and_support_identity_are_verified(tmp_path: Path) -> None:
    protocol, domain = _fixture(tmp_path)
    bad_hash = copy.deepcopy(protocol)
    bad_hash["inputs"]["fault_segments"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        load_slip_rate_surfaces(tmp_path, bad_hash, domain)
    specification = copy.deepcopy(protocol["inputs"]["fault_segments"])
    specification["allowed_columns"].append("last_strong_earthquake_year_raw")
    payload = (tmp_path / "faults.parquet").read_bytes()
    with pytest.raises(ValueError, match="allowlist"):
        _source_attributes(payload, specification)
    specification = copy.deepcopy(protocol["inputs"]["fault_segments"])
    specification["expected_both_nonnull"] = 2
    with pytest.raises(ValueError, match="support counts"):
        _source_attributes(payload, specification)
    specification = copy.deepcopy(protocol["inputs"]["fault_segments"])
    specification["expected_unique_ids"] = 2
    with pytest.raises(ValueError, match="unique identity"):
        _source_attributes(payload, specification)
