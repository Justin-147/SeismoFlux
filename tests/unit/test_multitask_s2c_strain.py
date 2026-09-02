"""Synthetic geometry only: no real GSRM remapping or earthquake targets."""

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
from pyproj import Transformer
from shapely.geometry import box
from shapely.ops import transform

from seismoflux.background.grid import EQUAL_AREA_CRS
from seismoflux.multitask_s2.strain import (
    _densified_rectangle,
    _remap_projected,
    _source_rectangles,
    _strain_scalar,
    _study_source_indices,
    blend_log_masses,
    load_strain_surfaces,
    validate_log_mass,
)


def test_area_accumulation_not_per_source_normalization():
    targets = [box(0, 0, 1000, 1000), box(1000, 0, 3000, 1000), box(3000, 0, 4000, 1000)]
    sources = [box(0, 0, 1500, 1000), box(1500, 0, 3000, 500)]
    layers, areas, weighted = _remap_projected(sources, np.array([2.0, 0.0]), targets)
    np.testing.assert_allclose(areas, [1, 1.25, 0])
    np.testing.assert_allclose(weighted, [2, 1, 0])
    np.testing.assert_allclose(np.exp(layers["UNIT"]), [1 / 2.25, 1.25 / 2.25, 0])
    np.testing.assert_allclose(np.exp(layers["STRAIN"]), [2 / 3, 1 / 3, 0])
    assert np.isneginf(layers["STRAIN"][2])
    assert not layers["UNIT"].flags.writeable


def test_tensor_shear_not_engineering_shear():
    np.testing.assert_allclose(_strain_scalar([0, 3], [0, 4], [3, 0]), [np.sqrt(18), 5])
    with pytest.raises(ValueError, match="nonfinite"):
        _strain_scalar([np.nan], [0], [0])


@pytest.mark.parametrize(
    "values", [[np.nan, 0], [np.inf, 0], [-np.inf, -np.inf], [0, 0], [], [[0]]]
)
def test_invalid_log_mass(values):
    with pytest.raises(ValueError):
        validate_log_mass(values)


def test_true_zero_and_exact_endpoints():
    c = np.log([0.25, 0.75])
    s = np.array([0.0, -np.inf])
    assert blend_log_masses(c, s, 0).tobytes() == c.tobytes()
    assert blend_log_masses(c, s, 1).tobytes() == s.tobytes()
    np.testing.assert_allclose(np.exp(blend_log_masses(c, s, 0.5)), [0.625, 0.375])
    for alpha in [-0.1, 1.1, np.nan]:
        with pytest.raises(ValueError):
            blend_log_masses(c, s, alpha)
    with pytest.raises(ValueError, match="count"):
        validate_log_mass(c, 3)


def test_source_duplicates_overlap_and_roundoff():
    for lon in ([0.0, 0.0], [0.0, 0.2]):
        with pytest.raises(ValueError, match="duplicate|overlapping"):
            _source_rectangles(np.array(lon), np.array([30.0, 30.0]), 0.25, 0.2)
    for delta in [0.0, -1e-12]:
        assert (
            len(
                _source_rectangles(np.array([0.0, 0.25 + delta]), np.array([30.0, 30.0]), 0.25, 0.2)
            )
            == 2
        )


def test_densified_edges():
    rectangle = _densified_rectangle((100, 30, 100.25, 30.2), 0.05)
    points = np.asarray(rectangle.exterior.coords)
    assert np.max(np.abs(np.diff(points, axis=0))) <= 0.05 + 1e-12
    assert rectangle.bounds == (100, 30, 100.25, 30.2)


def test_study_prefilter_retains_boundary_crossing_cells_not_remote_duplicates():
    lon = np.array([0.125, 0.125, 72.9, 100.0, 140.0])
    lat = np.full(5, 30.0)
    indices = _study_source_indices(lon, lat, 0.25, 0.2, (73.0, 18.0, 135.0, 54.0))
    np.testing.assert_array_equal(indices, [2, 3])
    assert len(_source_rectangles(lon[indices], lat[indices], 0.25, 0.2)) == 2


def test_all_zero_strain_rejected_not_floored():
    with pytest.raises(ValueError, match="STRAIN"):
        _remap_projected([box(0, 0, 1000, 1000)], np.array([0.0]), [box(0, 0, 1000, 1000)])


@pytest.mark.parametrize("outside_duplicates", [False, True])
def test_loader_synthetic_source_and_identity(tmp_path, outside_duplicates):
    path = tmp_path / "tiny.txt"
    content = "# synthetic only\n30 100 3 4 0 0 0 0 0 0 0 3 4 0\n"
    if outside_duplicates:
        content += "30 0.125 3 4 0 0 0 0 0 0 0 3 4 0\n"
        content += "30 0.125 5 6 0 0 0 0 0 0 0 5 6 0\n"
    path.write_text(content)
    polygon = transform(
        Transformer.from_crs("EPSG:4326", EQUAL_AREA_CRS, always_xy=True).transform,
        _densified_rectangle((99.875, 29.9, 100.125, 30.1), 0.05),
    )
    domain = SimpleNamespace(
        operational_grid=SimpleNamespace(
            grid_id="synthetic",
            cell_count=1,
            cell_size_km=25.0,
            clipped_area_km2=np.array([polygon.area / 1e6]),
        ),
        locator=SimpleNamespace(clipped_geometries=(polygon,)),
        study_area_wgs84=box(99.875, 29.9, 100.125, 30.1),
    )
    protocol = {
        "scientific_role": "synthetic",
        "inputs": {
            "grid_id": "synthetic",
            "grid_cells": 1,
            "grid_cell_size_km": 25.0,
            "grid_area_km2": polygon.area / 1e6,
            "strain_source": {
                "path": "tiny.txt",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "expected_rows": 3 if outside_duplicates else 1,
                "expected_columns": 14,
                "license": "synthetic",
                "attribution": "synthetic",
                "header_release_date": "2014-03-01",
                "historical_exact_bytes_confirmed": False,
            },
        },
        "spatial_math": {
            "longitude_width_degrees": 0.25,
            "latitude_height_degrees": 0.2,
            "geodetic_edge_maximum_segment_degrees": 0.05,
            "strain_scalar": "official",
            "uncovered_domain_role": "modeled rigid",
        },
    }
    result = load_strain_surfaces(data_root=tmp_path, domain=domain, protocol=protocol)
    np.testing.assert_allclose(result.layers["STRAIN"], [0])
    assert result.audit["source_rows"] == (3 if outside_duplicates else 1)
    assert result.audit["candidate_source_cells"] == 1
    assert result.audit["global_duplicate_centers"] == int(outside_duplicates)
    assert result.audit["study_candidate_duplicate_centers"] == 0
    assert not result.audit["global_records_deduplicated_or_averaged"]
    assert result.audit["principal_norm_max_absolute_difference"] == 0
    protocol["inputs"]["strain_source"]["sha256"] = "bad"
    with pytest.raises(ValueError, match="SHA256"):
        load_strain_surfaces(data_root=tmp_path, domain=domain, protocol=protocol)
