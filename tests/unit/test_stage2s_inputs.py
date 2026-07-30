from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from shapely import normalize as normalize_geometry
from shapely import to_wkb
from shapely.geometry import box, mapping

from seismoflux.background.artifacts import canonical_json_bytes
from seismoflux.background.catalog import load_study_area_bytes
from seismoflux.background.grid import EQUAL_AREA_CRS
from seismoflux.features.anomaly.grid import build_stage3_query_grid
from seismoflux.stage2s.catalog import (
    FROZEN_EARTHQUAKE_CATALOG_CONTRACT,
    parse_catalog_bytes,
)
from seismoflux.stage2s.inputs import (
    GRID_LAYER_ORDER,
    PARENT_RELATION_ORDER,
    REPRESENTATIVE_POINT_ALGORITHM,
    NonTargetExpectations,
    NonTargetSpatialIdentity,
    Stage2SInputError,
    build_non_target_adapter_from_bytes,
    to_spatial_quadrature_family,
    validated_non_target_preflight_receipt_bindings,
)


def _geometry_hash(study_area_bytes: bytes) -> tuple[str, float]:
    study_area = load_study_area_bytes(study_area_bytes, EQUAL_AREA_CRS)
    payload = cast(
        bytes,
        to_wkb(
            normalize_geometry(study_area.projected),
            hex=False,
            output_dimension=2,
            byte_order=1,
            include_srid=False,
        ),
    )
    return hashlib.sha256(payload).hexdigest(), float(study_area.projected.area)


def _mapping_bytes(
    study_area_bytes: bytes,
    *,
    perturb_x: bool = False,
) -> tuple[bytes, str, int, int]:
    study_area = load_study_area_bytes(study_area_bytes, EQUAL_AREA_CRS)
    grid = build_stage3_query_grid(study_area.geographic)
    zone_ids = [f"zone-{index % 2 + 1}" for index in range(grid.cell_count)]
    x_m = np.array(grid.query_xy_m[:, 0], copy=True)
    if perturb_x:
        x_m[0] += 1.0
    table = pa.table(
        {
            "grid_id": [grid.grid_id] * grid.cell_count,
            "cell_id": list(grid.cell_ids),
            "cell_row": grid.rows,
            "cell_column": grid.columns,
            "query_x_m": x_m,
            "query_y_m": grid.query_xy_m[:, 1],
            "construction_zone_id": zone_ids,
        },
        schema=pa.schema(
            [
                pa.field("grid_id", pa.string()),
                pa.field("cell_id", pa.string()),
                pa.field("cell_row", pa.int64()),
                pa.field("cell_column", pa.int64()),
                pa.field("query_x_m", pa.float64()),
                pa.field("query_y_m", pa.float64()),
                pa.field("construction_zone_id", pa.string()),
            ]
        ),
    )
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    payload = sink.getvalue().to_pybytes()
    return payload, grid.grid_id, grid.cell_count, len(set(zone_ids))


def _synthetic_payloads() -> tuple[bytes, bytes, NonTargetExpectations]:
    document = {
        "type": "Feature",
        "properties": {},
        "geometry": mapping(box(109.5, 34.5, 110.5, 35.5)),
    }
    study_bytes = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    mapping_bytes, grid_id, row_count, zone_count = _mapping_bytes(study_bytes)
    geometry_hash, area_m2 = _geometry_hash(study_bytes)
    expectations = NonTargetExpectations(
        study_area_file_sha256=hashlib.sha256(study_bytes).hexdigest(),
        projected_geometry_sha256=geometry_hash,
        projected_area_m2=area_m2,
        projected_area_absolute_tolerance_m2=max(1.0e-6, area_m2 * 1.0e-12),
        mapping_file_sha256=hashlib.sha256(mapping_bytes).hexdigest(),
        mapping_row_count=row_count,
        zone_count=zone_count,
        expected_grid_id=grid_id,
    )
    return study_bytes, mapping_bytes, expectations


def test_byte_only_adapter_returns_grid_and_mapping_together() -> None:
    study_bytes, mapping_bytes, expectations = _synthetic_payloads()

    preflight = build_non_target_adapter_from_bytes(
        study_area_bytes=study_bytes,
        mapping_bytes=mapping_bytes,
        expectations=expectations,
    )

    query_grid = preflight.adapter.query_grid
    assert query_grid.cell_count == expectations.mapping_row_count
    assert len(preflight.adapter.construction_zone_id_by_cell_id) == query_grid.cell_count
    assert set(preflight.adapter.construction_zone_id_by_cell_id) == set(query_grid.cell_ids)
    assert len(preflight.zone_ids) == expectations.zone_count
    assert preflight.grid_cell_counts[1] == query_grid.cell_count
    quadrature = to_spatial_quadrature_family(preflight.adapter)
    assert quadrature.at(25.0).grid_id == query_grid.grid_id
    np.testing.assert_array_equal(
        quadrature.at(25.0).query_xy_km,
        query_grid.query_xy_m / 1_000.0,
    )
    np.testing.assert_array_equal(
        query_grid.query_xy_m,
        np.asarray(
            [
                (
                    cell.representative_point.x,
                    cell.representative_point.y,
                )
                for cell in preflight.adapter.grid_family.at(25.0).cells
            ]
        ),
    )


def test_preflight_receipt_binds_complete_aligned_grid_identity() -> None:
    study_bytes, mapping_bytes, expectations = _synthetic_payloads()
    preflight = build_non_target_adapter_from_bytes(
        study_area_bytes=study_bytes,
        mapping_bytes=mapping_bytes,
        expectations=expectations,
    )

    bindings = validated_non_target_preflight_receipt_bindings(preflight)
    identity = cast(dict[str, Any], bindings["aligned_grid_identity"])
    assert identity["layer_order"] == list(GRID_LAYER_ORDER)
    assert identity["parent_relation_order"] == list(PARENT_RELATION_ORDER)
    assert identity["representative_point_algorithm"] == REPRESENTATIVE_POINT_ALGORITHM
    assert identity["target_or_score_input_count"] == 0
    layers = cast(dict[str, dict[str, object]], identity["layers"])
    for index, layer_name in enumerate(GRID_LAYER_ORDER):
        layer = layers[layer_name]
        assert layer["cell_count"] == preflight.grid_cell_counts[index]
        assert layer["total_clipped_area_km2"] == preflight.grid_total_area_km2[index]
        assert layer["representative_point_count"] == layer["cell_count"]
        assert layer["representative_point_algorithm"] == REPRESENTATIVE_POINT_ALGORITHM
        assert len(cast(str, layer["grid_id"])) == 64
        assert len(cast(str, layer["operational_cell_identity_sha256"])) == 64
        assert len(cast(str, layer["ordered_clipped_area_sha256"])) == 64
        assert len(cast(str, layer["representative_point_sha256"])) == 64
    assert layers["25"]["grid_id"] == preflight.adapter.query_grid.grid_id

    relations = cast(dict[str, dict[str, object]], identity["parent_relations"])
    for relation_name, fine_name, coarse_name in (
        ("25_to_50", "25", "50"),
        ("12.5_to_25", "12.5", "25"),
    ):
        relation = relations[relation_name]
        assert relation["fine_grid_id"] == layers[fine_name]["grid_id"]
        assert relation["coarse_grid_id"] == layers[coarse_name]["grid_id"]
        assert relation["aligned_size_ratio"] == 2
        assert relation["mapped_fine_cell_count"] == relation["fine_cell_count"]
        assert relation["covered_coarse_parent_count"] == relation["coarse_cell_count"]
        assert cast(float, relation["maximum_parent_area_absolute_error_km2"]) <= cast(
            float,
            relation["parent_area_absolute_tolerance_km2"],
        )
        assert len(cast(str, relation["parent_mapping_sha256"])) == 64
        assert len(cast(str, relation["parent_area_relation_sha256"])) == 64

    unsigned = dict(identity)
    observed_sha256 = cast(str, unsigned.pop("identity_sha256"))
    assert hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() == observed_sha256


def test_preflight_spatial_identity_tampering_fails_before_receipt() -> None:
    study_bytes, mapping_bytes, expectations = _synthetic_payloads()
    preflight = build_non_target_adapter_from_bytes(
        study_area_bytes=study_bytes,
        mapping_bytes=mapping_bytes,
        expectations=expectations,
    )
    identity = cast(NonTargetSpatialIdentity, preflight.spatial_identity)
    changed_layer = replace(
        identity.layers[0],
        representative_point_sha256="0" * 64,
    )

    with pytest.raises(Stage2SInputError, match="aggregate spatial identity"):
        replace(
            identity,
            layers=(changed_layer, *identity.layers[1:]),
        )
    with pytest.raises(Stage2SInputError, match="grid summaries"):
        replace(
            preflight,
            grid_cell_counts=(
                preflight.grid_cell_counts[0] + 1,
                *preflight.grid_cell_counts[1:],
            ),
        )
    relation = identity.parent_relations[0]
    with pytest.raises(Stage2SInputError, match="parent cell area"):
        replace(
            relation,
            maximum_parent_area_absolute_error_km2=(
                relation.parent_area_absolute_tolerance_km2 + 1.0
            ),
        )


def test_byte_only_adapter_never_opens_or_stats_a_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study_bytes, mapping_bytes, expectations = _synthetic_payloads()

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("byte-only adapter touched a filesystem path")

    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(Path, "stat", forbidden)
    preflight = build_non_target_adapter_from_bytes(
        study_area_bytes=study_bytes,
        mapping_bytes=mapping_bytes,
        expectations=expectations,
    )

    assert preflight.receipt_bindings()["earthquake_catalog_bytes_read"] is False


def test_mapping_parquet_parse_is_single_threaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study_bytes, mapping_bytes, expectations = _synthetic_payloads()
    original_read_table = pq.read_table
    observed_use_threads: list[object] = []

    def guarded_read_table(*args: Any, **kwargs: Any) -> pa.Table:
        observed_use_threads.append(kwargs.get("use_threads"))
        return original_read_table(*args, **kwargs)

    monkeypatch.setattr(pq, "read_table", guarded_read_table)
    build_non_target_adapter_from_bytes(
        study_area_bytes=study_bytes,
        mapping_bytes=mapping_bytes,
        expectations=expectations,
    )

    assert observed_use_threads == [False]


def test_catalog_parquet_parse_is_single_threaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"synthetic-invalid-parquet"
    contract = replace(
        FROZEN_EARTHQUAKE_CATALOG_CONTRACT,
        file_sha256=hashlib.sha256(payload).hexdigest(),
    )
    observed_use_threads: list[object] = []

    def guarded_read_table(*args: Any, **kwargs: Any) -> pa.Table:
        observed_use_threads.append(kwargs.get("use_threads"))
        raise pa.ArrowInvalid("synthetic parser stop after thread-contract observation")

    monkeypatch.setattr(pq, "read_table", guarded_read_table)
    with pytest.raises(ValueError, match="valid Parquet"):
        parse_catalog_bytes(payload, contract=contract)

    assert observed_use_threads == [False]


def test_mapping_coordinates_must_match_query_grid_bitwise() -> None:
    study_bytes, original_mapping, expectations = _synthetic_payloads()
    changed_mapping, _, _, _ = _mapping_bytes(study_bytes, perturb_x=True)
    changed_expectations = NonTargetExpectations(
        study_area_file_sha256=expectations.study_area_file_sha256,
        projected_geometry_sha256=expectations.projected_geometry_sha256,
        projected_area_m2=expectations.projected_area_m2,
        projected_area_absolute_tolerance_m2=(expectations.projected_area_absolute_tolerance_m2),
        mapping_file_sha256=hashlib.sha256(changed_mapping).hexdigest(),
        mapping_row_count=expectations.mapping_row_count,
        zone_count=expectations.zone_count,
        expected_grid_id=expectations.expected_grid_id,
    )

    assert changed_mapping != original_mapping
    with pytest.raises(Stage2SInputError, match="coordinates differ bitwise"):
        build_non_target_adapter_from_bytes(
            study_area_bytes=study_bytes,
            mapping_bytes=changed_mapping,
            expectations=changed_expectations,
        )


def test_study_area_and_mapping_hashes_fail_closed() -> None:
    study_bytes, mapping_bytes, expectations = _synthetic_payloads()
    wrong = NonTargetExpectations(
        study_area_file_sha256="0" * 64,
        projected_geometry_sha256=expectations.projected_geometry_sha256,
        projected_area_m2=expectations.projected_area_m2,
        projected_area_absolute_tolerance_m2=(expectations.projected_area_absolute_tolerance_m2),
        mapping_file_sha256=expectations.mapping_file_sha256,
        mapping_row_count=expectations.mapping_row_count,
        zone_count=expectations.zone_count,
        expected_grid_id=expectations.expected_grid_id,
    )

    with pytest.raises(Stage2SInputError, match="study area file hash"):
        build_non_target_adapter_from_bytes(
            study_area_bytes=study_bytes,
            mapping_bytes=mapping_bytes,
            expectations=wrong,
        )
