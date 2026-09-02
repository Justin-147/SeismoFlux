"""Target-blind, speed-weighted line fields for the frozen S2-B experiment.

The two supplied total rates weight the original simplified fault geometry.
They are relative spatial covariates from a current snapshot, not earthquake
moment rates, absolute probabilities, or historically available forecasts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import NDArray
from scipy.spatial.distance import cdist  # type: ignore[import-untyped]
from scipy.special import logsumexp  # type: ignore[import-untyped]
from shapely import from_wkb
from shapely.geometry.base import BaseGeometry

from seismoflux.background.grid import EQUAL_AREA_CRS
from seismoflux.d1_replay.spatial import D1SpatialDomain
from seismoflux.multitask_s1.c2b_score import log_alarm_prefixes
from seismoflux.multitask_s2.fault_geometry import (
    _areas,
    _project_lines_to_equal_area,
    _readonly,
    _verified_payload,
)
from seismoflux.stage2s.contracts import SpatialGrid

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
GEO_RATE = "geologic_total_rate_mm_per_year"
GD_RATE = "geodetic_total_rate_mm_per_year"
_SOURCE_COLUMNS = ["fault_segment_id", "simplified_geometry_wkb", GEO_RATE, GD_RATE]
_SCALES = (25.0, 75.0, 150.0)
_LAYERS: dict[str, dict[str, Any]] = {
    "COMMON_UNIT": {"panel": "COMMON", "weight_column": None, "constant_weight": 1.0},
    "COMMON_GEO": {"panel": "COMMON", "weight_column": GEO_RATE, "constant_weight": None},
    "COMMON_GD": {"panel": "COMMON", "weight_column": GD_RATE, "constant_weight": None},
    "NATIVE_UNIT": {"panel": "NATIVE_GD", "weight_column": None, "constant_weight": 1.0},
    "NATIVE_GD": {"panel": "NATIVE_GD", "weight_column": GD_RATE, "constant_weight": None},
}


@dataclass(frozen=True, slots=True)
class SlipRateSurfaces:
    """The fifteen production log cell-mass fields and target-free diagnostics."""

    layers: dict[str, dict[float, FloatArray]]
    audit: dict[str, Any]


@dataclass(frozen=True, slots=True)
class LineQuadrature:
    """Original-edge midpoint locations, with unnormalized represented lengths."""

    points_xy_km: FloatArray
    represented_length_km: FloatArray
    line_index: IntArray
    line_length_km: FloatArray
    maximum_step_km: float


@dataclass(frozen=True, slots=True)
class _SlipRateSource:
    lines: tuple[BaseGeometry, ...]
    rates: dict[str, FloatArray]
    panels: dict[str, BoolArray]
    audit: dict[str, Any]


def _source_attributes(payload: bytes, specification: dict[str, Any]) -> _SlipRateSource:
    """Read only the frozen ID, geometry and two original total-rate columns."""

    if (
        specification["allowed_columns"] != _SOURCE_COLUMNS
        or specification["id_column"] != _SOURCE_COLUMNS[0]
        or specification["geometry_column"] != _SOURCE_COLUMNS[1]
        or specification["rate_units"] != "mm_per_year"
    ):
        raise ValueError("S2-B source must use its exact four-column allowlist and original units")
    table = pq.read_table(pa.BufferReader(payload), columns=_SOURCE_COLUMNS, use_threads=False)
    if table.num_rows != specification["expected_rows"]:
        raise ValueError("S2-B source row count changed")
    identifiers = table[_SOURCE_COLUMNS[0]].to_pylist()
    if any(not isinstance(value, str) or not value for value in identifiers):
        raise ValueError("S2-B source IDs must be nonempty strings")
    if (
        len(set(identifiers)) != len(identifiers)
        or len(set(identifiers)) != specification["expected_unique_ids"]
    ):
        raise ValueError("S2-B source IDs must have the frozen unique identity count")
    wkbs = table[_SOURCE_COLUMNS[1]].to_pylist()
    if any(value is None for value in wkbs):
        raise ValueError("S2-B source geometry WKB must not be null")
    lines = tuple(from_wkb(value) for value in wkbs)
    if len(lines) != specification["expected_usable_lines"]:
        raise ValueError("S2-B usable line count changed")
    for line in lines:
        if (
            line is None
            or line.geom_type != "LineString"
            or line.is_empty
            or not line.is_valid
            or not math.isfinite(float(line.length))
            or line.length <= 0.0
            or not np.isfinite(np.asarray(line.coords)).all()
        ):
            raise ValueError("S2-B source requires finite valid positive-length LineStrings")
    rates: dict[str, FloatArray] = {}
    rate_audit: dict[str, Any] = {}
    for field, short in ((GEO_RATE, "geologic"), (GD_RATE, "geodetic")):
        # NaN represents the absent attribute. It is never replaced by a zero rate.
        values = np.asarray(table[field].to_pylist(), dtype=np.float64)
        present = ~np.isnan(values)
        if np.any(~np.isfinite(values[present])) or np.any(values[present] <= 0.0):
            raise ValueError("S2-B supplied nonmissing total rates must be finite and positive")
        expected_count = int(specification[f"expected_{short}_nonnull_and_positive"])
        if int(np.count_nonzero(present)) != expected_count or expected_count == 0:
            raise ValueError(f"S2-B {short} nonnull/positive rate count changed")
        observed_range = [float(np.min(values[present])), float(np.max(values[present]))]
        expected_range = specification[f"{short}_rate_range_mm_per_year"]
        if not np.allclose(observed_range, expected_range, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"S2-B {short} original rate range changed")
        rates[field] = _readonly(values)
        rate_audit[field] = {
            "nonnull_and_positive": expected_count,
            "missing": int(values.size - expected_count),
            "range_mm_per_year": observed_range,
            "values_transformed_or_imputed": False,
        }
    geo_present = np.isfinite(rates[GEO_RATE])
    gd_present = np.isfinite(rates[GD_RATE])
    common = geo_present & gd_present
    counts = {
        "expected_both_nonnull": int(np.count_nonzero(common)),
        "expected_geodetic_only": int(np.count_nonzero(~geo_present & gd_present)),
        "expected_both_missing": int(np.count_nonzero(~geo_present & ~gd_present)),
    }
    if any(count != int(specification[key]) for key, count in counts.items()):
        raise ValueError("S2-B common/native rate support counts changed")
    if np.any(geo_present & ~gd_present):
        raise ValueError("S2-B frozen geological support must be contained in geodetic support")
    common.setflags(write=False)
    gd_present.setflags(write=False)
    return _SlipRateSource(
        lines=lines,
        rates=rates,
        panels={"COMMON": common, "NATIVE_GD": gd_present},
        audit={
            "rows": table.num_rows,
            "unique_ids": len(set(identifiers)),
            "usable_lines": len(lines),
            "read_columns": list(_SOURCE_COLUMNS),
            "rate_units": "mm_per_year",
            "rates": rate_audit,
            "common_segments": counts["expected_both_nonnull"],
            "geodetic_only_segments": counts["expected_geodetic_only"],
            "both_rates_missing_segments": counts["expected_both_missing"],
            "missing_rates_treated_as_zero": False,
            "attribute_transfer_to_detailed_traces": False,
            "original_line_order_and_duplicates_preserved": True,
        },
    )


def midpoint_quadrature(
    projected_lines_m: tuple[BaseGeometry, ...], maximum_step_km: float
) -> LineQuadrature:
    """Integrate every original projected straight edge, without line normalization."""

    step = float(maximum_step_km)
    if not math.isfinite(step) or step <= 0.0 or not projected_lines_m:
        raise ValueError("quadrature requires original lines and a finite positive step")
    point_parts: list[FloatArray] = []
    length_parts: list[FloatArray] = []
    index_parts: list[IntArray] = []
    line_lengths: list[float] = []
    for index, line in enumerate(projected_lines_m):
        if line.geom_type != "LineString" or line.is_empty or not line.is_valid:
            raise ValueError("quadrature requires valid original LineStrings")
        coordinates = np.asarray(line.coords, dtype=np.float64)[:, :2] / 1_000.0
        if not np.isfinite(coordinates).all():
            raise ValueError("quadrature line coordinates must be finite")
        edge_lengths: list[float] = []
        for start, end in pairwise(coordinates):
            delta = end - start
            length = math.hypot(float(delta[0]), float(delta[1]))
            if not math.isfinite(length):
                raise ValueError("quadrature projected edge length must be finite")
            if length == 0.0:
                continue
            count = math.ceil(length / step)
            fractions = (np.arange(count, dtype=np.float64) + 0.5) / count
            point_parts.append(start + fractions[:, np.newaxis] * delta)
            length_parts.append(np.full(count, length / count, dtype=np.float64))
            index_parts.append(np.full(count, index, dtype=np.int64))
            edge_lengths.append(length)
        total = math.fsum(edge_lengths)
        if total <= 0.0:
            raise ValueError("quadrature requires each source line to have positive length")
        line_lengths.append(total)
    line_index = np.concatenate(index_parts)
    line_index.setflags(write=False)
    return LineQuadrature(
        points_xy_km=_readonly(np.concatenate(point_parts)),
        represented_length_km=_readonly(np.concatenate(length_parts)),
        line_index=line_index,
        line_length_km=_readonly(np.asarray(line_lengths, dtype=np.float64)),
        maximum_step_km=step,
    )


def gaussian_line_log_masses(
    query_xy_km: FloatArray,
    area_km2: FloatArray,
    quadrature: LineQuadrature,
    layer_line_weights: dict[str, FloatArray],
    scales_km: tuple[float, ...],
    *,
    query_chunk_size: int = 128,
    point_chunk_size: int = 4096,
) -> dict[str, dict[float, FloatArray]]:
    """Length-weighted Gaussian sums, retaining distant tails in log space.

    NaN in a layer's line-weight vector means the line is outside that layer's
    explicit attribute support, not a zero observed rate. Chunk sizes affect
    memory only; no radius truncation, background floor, or extra model is used.
    The NumPy/SciPy elementwise reductions here do not launch fold workers or
    BLAS products. The caller retains the existing single-thread numeric setup.
    """

    query = np.asarray(query_xy_km, dtype=np.float64)
    if (
        query.ndim != 2
        or query.shape[1] != 2
        or query.shape[0] == 0
        or not np.isfinite(query).all()
    ):
        raise ValueError("query points must be finite with nonempty shape (cells, 2)")
    area = _areas(area_km2, query.shape[0])
    if query_chunk_size < 1 or point_chunk_size < 1:
        raise ValueError("quadrature chunk sizes must be positive")
    scales = tuple(float(scale) for scale in scales_km)
    if not scales or any(not math.isfinite(scale) or scale <= 0.0 for scale in scales):
        raise ValueError("Gaussian scales must be finite and positive")
    if len(set(scales)) != len(scales) or not layer_line_weights:
        raise ValueError("unique scales and at least one weighted layer are required")
    log_point_weights: dict[str, FloatArray] = {}
    for layer, weights in layer_line_weights.items():
        values = np.asarray(weights, dtype=np.float64)
        if values.shape != quadrature.line_length_km.shape:
            raise ValueError("layer weights must align with the original quadrature lines")
        present = ~np.isnan(values)
        if (
            not np.any(present)
            or not np.isfinite(values[present]).all()
            or np.any(values[present] <= 0.0)
        ):
            raise ValueError("every included line weight must be finite and positive")
        point_present = present[quadrature.line_index]
        log_weights = np.full(quadrature.line_index.shape, -np.inf, dtype=np.float64)
        log_weights[point_present] = np.log(
            quadrature.represented_length_km[point_present]
        ) + np.log(values[quadrature.line_index[point_present]])
        log_point_weights[layer] = log_weights
    log_density = {
        layer: {scale: np.full(query.shape[0], -np.inf, dtype=np.float64) for scale in scales}
        for layer in layer_line_weights
    }
    point_count = quadrature.points_xy_km.shape[0]
    for query_start in range(0, query.shape[0], query_chunk_size):
        query_slice = slice(query_start, query_start + query_chunk_size)
        for point_start in range(0, point_count, point_chunk_size):
            point_slice = slice(point_start, point_start + point_chunk_size)
            squared_distance = cdist(
                query[query_slice], quadrature.points_xy_km[point_slice], metric="sqeuclidean"
            )
            if not np.isfinite(squared_distance).all():
                raise ValueError("projected squared distances exceed finite float64 range")
            for scale in scales:
                kernel = -0.5 * squared_distance / (scale * scale)
                kernel -= math.log(2.0 * math.pi) + 2.0 * math.log(scale)
                for layer, weights in log_point_weights.items():
                    selected_weights = weights[point_slice]
                    included = np.isfinite(selected_weights)
                    if not np.any(included):
                        continue
                    subtotal = logsumexp(kernel[:, included] + selected_weights[included], axis=1)
                    log_density[layer][scale][query_slice] = np.logaddexp(
                        log_density[layer][scale][query_slice], subtotal
                    )
    result: dict[str, dict[float, FloatArray]] = {}
    for layer, fields in log_density.items():
        result[layer] = {}
        for scale, density in fields.items():
            if not np.isfinite(density).all():
                raise FloatingPointError("Gaussian line field lost a finite far-tail density")
            unnormalized = density + np.log(area)
            normalized = unnormalized - logsumexp(unnormalized)
            # A second normalization removes subtraction roundoff for very far queries.
            normalized -= logsumexp(normalized)
            result[layer][scale] = _readonly(normalized)
    return result


def _integration_diagnostics(
    fine: dict[str, dict[float, FloatArray]],
    diagnostic: dict[str, dict[float, FloatArray]],
    grid: SpatialGrid,
    budgets: list[float],
) -> list[dict[str, Any]]:
    """One field-only comparison using the unchanged no-skip area-prefix rule."""

    rows: list[dict[str, Any]] = []
    for layer, scales in fine.items():
        for scale, fine_mass in scales.items():
            coarse_mass = diagnostic[layer][scale]
            fine_prefixes = log_alarm_prefixes(fine_mass, grid, budgets)
            coarse_prefixes = log_alarm_prefixes(coarse_mass, grid, budgets)
            alarm_differences = []
            for fine_prefix, coarse_prefix in zip(fine_prefixes, coarse_prefixes, strict=True):
                fine_selected = set(fine_prefix["selected"])
                coarse_selected = set(coarse_prefix["selected"])
                changed = fine_selected ^ coarse_selected
                alarm_differences.append(
                    {
                        "area_budget_km2": fine_prefix["area_budget_km2"],
                        "fine_actual_area_km2": fine_prefix["actual_area_km2"],
                        "diagnostic_actual_area_km2": coarse_prefix["actual_area_km2"],
                        "fine_selected_cells": len(fine_selected),
                        "diagnostic_selected_cells": len(coarse_selected),
                        "fine_only_cells": len(fine_selected - coarse_selected),
                        "diagnostic_only_cells": len(coarse_selected - fine_selected),
                        "symmetric_difference_cells": len(changed),
                        "symmetric_difference_area_km2": math.fsum(
                            float(grid.clipped_area_km2[index]) for index in sorted(changed)
                        ),
                    }
                )
            rows.append(
                {
                    "layer": layer,
                    "scale_km": scale,
                    "normalized_layer_total_variation": 0.5
                    * float(np.abs(np.exp(fine_mass) - np.exp(coarse_mass)).sum()),
                    "alarm_cell_differences_at_fixed_budgets": alarm_differences,
                }
            )
    return rows


def load_slip_rate_surfaces(
    data_root: Path, protocol: dict[str, Any], domain: D1SpatialDomain
) -> SlipRateSurfaces:
    """Build exactly the frozen fine fields and one target-free coarse diagnostic.

    No catalogs, earthquake targets, source-version flags, or excluded fault
    attributes are loaded. This loader writes no files. The prediction caller
    saves its returned fields and audit once in the new S2-B run for resumption.
    """

    root = Path(data_root).resolve(strict=True)
    inputs = protocol["inputs"]
    geometry_math = protocol["geometry_math"]
    integration = protocol["numerical_integration"]
    grid = domain.operational_grid
    stage3_grid = domain.stage3_grid
    if (
        grid.grid_id != inputs["grid_id"]
        or grid.cell_count != inputs["grid_cells"]
        or grid.cell_size_km != inputs["grid_cell_size_km"]
        or stage3_grid.grid_id != grid.grid_id
        or stage3_grid.cell_ids != grid.cell_ids
        or not np.array_equal(stage3_grid.query_xy_m / 1_000.0, grid.query_xy_km)
        or not np.array_equal(stage3_grid.clipped_area_km2, grid.clipped_area_km2)
        or not math.isclose(
            math.fsum(float(value) for value in grid.clipped_area_km2),
            float(inputs["grid_area_km2"]),
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
    ):
        raise ValueError(
            "S2-B grid differs from its frozen original representative points or areas"
        )
    scales = tuple(float(value) for value in geometry_math["scales_km"])
    if scales != _SCALES or protocol["layers"] != _LAYERS:
        raise ValueError("S2-B layers or scales differ from the frozen finite experiment")
    if (
        integration["production_maximum_step_km"] != 3.125
        or integration["diagnostic_maximum_step_km"] != 6.25
        or integration["diagnostic_runs"] != 1
        or integration["diagnostic_uses_earthquake_targets"] is not False
        or integration["production_always_uses_fine_step"] is not True
    ):
        raise ValueError(
            "S2-B requires fixed fine production and one target-free coarse diagnostic"
        )
    specification = inputs["fault_segments"]
    payload = _verified_payload(root, specification["path"], specification["sha256"])
    source = _source_attributes(payload, specification)
    if set(protocol["panels"]) != {"COMMON", "NATIVE_GD"}:
        raise ValueError("S2-B requires exactly its COMMON and NATIVE_GD supports")
    for panel, expected_columns in (
        ("COMMON", [GEO_RATE, GD_RATE]),
        ("NATIVE_GD", [GD_RATE]),
    ):
        panel_spec = protocol["panels"][panel]
        if (
            panel_spec["required_nonnull_columns"] != expected_columns
            or int(np.count_nonzero(source.panels[panel])) != panel_spec["expected_segments"]
        ):
            raise ValueError("S2-B panel definitions or their original support counts changed")
    projected = _project_lines_to_equal_area(source.lines)
    native = source.panels["NATIVE_GD"]
    native_lines = tuple(line for line, keep in zip(projected, native, strict=True) if keep)
    layer_weights: dict[str, FloatArray] = {}
    for layer, layer_spec in _LAYERS.items():
        selected = source.panels[layer_spec["panel"]][native]
        weights = np.full(len(native_lines), np.nan, dtype=np.float64)
        field = layer_spec["weight_column"]
        if field is None:
            weights[selected] = 1.0
        else:
            weights[selected] = source.rates[field][native][selected]
        layer_weights[layer] = weights
    fine_quadrature = midpoint_quadrature(native_lines, 3.125)
    fine = gaussian_line_log_masses(
        grid.query_xy_km, grid.clipped_area_km2, fine_quadrature, layer_weights, scales
    )
    coarse_quadrature = midpoint_quadrature(native_lines, 6.25)
    coarse = gaussian_line_log_masses(
        grid.query_xy_km, grid.clipped_area_km2, coarse_quadrature, layer_weights, scales
    )
    diagnostics = _integration_diagnostics(
        fine, coarse, grid, [float(value) for value in protocol["evaluation"]["area_budgets_km2"]]
    )
    layer_audit = {}
    for layer, weights in layer_weights.items():
        selected = np.isfinite(weights)
        layer_audit[layer] = {
            **_LAYERS[layer],
            "segments": int(np.count_nonzero(selected)),
            "projected_line_length_km": math.fsum(fine_quadrature.line_length_km[selected]),
            "length_times_input_weight": math.fsum(
                fine_quadrature.line_length_km[selected] * weights[selected]
            ),
        }
    return SlipRateSurfaces(
        layers=fine,
        audit={
            "scientific_role": protocol["scientific_role"],
            "static_snapshot_available_at": inputs["static_snapshot_available_at"],
            "original_historical_model_eligible": inputs["original_historical_model_eligible"],
            "source_eligibility_flags_modified": False,
            "source_crs_assumption": geometry_math["source_crs_assumption"],
            "projected_crs": EQUAL_AREA_CRS,
            "distance_and_length_units": "km",
            "grid_id": grid.grid_id,
            "grid_cells": grid.cell_count,
            "grid_area_km2": math.fsum(grid.clipped_area_km2),
            "scales_km": list(scales),
            "source": {
                **source.audit,
                "path": specification["path"],
                "sha256": specification["sha256"],
            },
            "layers": layer_audit,
            "numerical_integration": {
                "method": integration["method"],
                "production_maximum_step_km": 3.125,
                "diagnostic_maximum_step_km": 6.25,
                "production_quadrature_points": int(fine_quadrature.line_index.size),
                "diagnostic_quadrature_points": int(coarse_quadrature.line_index.size),
                "diagnostic_runs": 1,
                "diagnostic_uses_earthquake_targets": False,
                "production_always_uses_fine_step": True,
                "national_percentage_agreement_gate": None,
                "diagnostics": diagnostics,
            },
            "target_coordinates_read": False,
            "grid_or_targets_filtered_by_rate_support": False,
            "per_line_unit_mass_normalization": False,
            "Gaussian_tail_truncation_radius_km": None,
            "artificial_probability_floor": None,
            "interpretation": "relative_spatial_mass_not_absolute_probability_or_moment_rate",
        },
    )
