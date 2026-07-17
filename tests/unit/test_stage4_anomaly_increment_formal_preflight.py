from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from stage4_formal_preflight_fixture import make_formal_preflight_receipt

from seismoflux.anomaly_increment.config import load_stage4_protocol_bundle
from seismoflux.anomaly_increment.formal_preflight import (
    FORMAL_ASSESSMENT_ISSUE_COUNT,
    FORMAL_FEATURE_COLUMNS,
    FORMAL_FIT_ISSUE_COUNT,
    FORMAL_POOL_ISSUE_COUNT,
    FormalIssueCalendar,
    FormalPreflightReceipt,
    GridIdentityBridge,
    _assert_columns_exact,
    _host_memory_observation,
    _logical_selected_table_identity_sha256,
    resolve_score_blind_input_path,
)
from seismoflux.anomaly_increment.grid_features import (
    Stage4IntegrationGrid,
    build_stage4_grid_family,
    selected_table_identity_sha256,
)
from seismoflux.background.catalog import load_study_area
from seismoflux.background.grid import EQUAL_AREA_CRS

ROOT = Path(__file__).resolve().parents[2]
ACTUAL_FEATURE_STORE = ROOT / (
    "data/processed/stage3/anomaly_history/"
    "anomaly-feature-bundle-de7547faa9f87541/anomaly_feature_store.parquet"
)
STAGE3_GRID_ID = "3aacdbdda04fed652dd5ee3674906f674c127cb735dea5d5e989527b20809763"


def _grid() -> Stage4IntegrationGrid:
    return Stage4IntegrationGrid(
        grid_id="4" * 64,
        equal_area_crs=EQUAL_AREA_CRS,
        cell_size_km=25.0,
        cell_ids=("cell-a", "cell-b"),
        rows=np.asarray([-1, 0], dtype=np.int64),
        columns=np.asarray([3, 4], dtype=np.int64),
        query_xy_m=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        clipped_area_km2=np.asarray([625.0, 125.0], dtype=np.float64),
    )


def _science_array(name: str) -> pa.Array:
    if name.endswith("__valid"):
        return pa.array([True, False], type=pa.bool_())
    if name.endswith("__sample_count"):
        return pa.array([3, 0], type=pa.int64())
    if name.endswith("__null_reason_code"):
        return pa.array([0, 2], type=pa.int8())
    return pa.array([1.25, None], type=pa.float64())


def _table() -> pa.Table:
    grid = _grid()
    issue_time = datetime(2022, 7, 20, 16, tzinfo=UTC)
    columns: dict[str, pa.Array] = {}
    for name in FORMAL_FEATURE_COLUMNS:
        if name == "issue_index":
            columns[name] = pa.array([0, 0], type=pa.int64())
        elif name == "issue_time_utc":
            columns[name] = pa.array([issue_time, issue_time], type=pa.timestamp("us", tz="UTC"))
        elif name == "issue_report_id":
            columns[name] = pa.array(["report-1", "report-1"], type=pa.string())
        elif name == "grid_id":
            columns[name] = pa.array([STAGE3_GRID_ID, STAGE3_GRID_ID], type=pa.string())
        elif name == "equal_area_crs":
            columns[name] = pa.array([EQUAL_AREA_CRS, EQUAL_AREA_CRS], type=pa.string())
        elif name == "cell_size_km":
            columns[name] = pa.array([25.0, 25.0], type=pa.float64())
        elif name == "cell_id":
            columns[name] = pa.array(grid.cell_ids, type=pa.string())
        elif name == "cell_row":
            columns[name] = pa.array(grid.rows, type=pa.int64())
        elif name == "cell_column":
            columns[name] = pa.array(grid.columns, type=pa.int64())
        elif name == "query_x_m":
            columns[name] = pa.array(grid.query_xy_m[:, 0], type=pa.float64())
        elif name == "query_y_m":
            columns[name] = pa.array(grid.query_xy_m[:, 1], type=pa.float64())
        elif name == "clipped_area_km2":
            columns[name] = pa.array(grid.clipped_area_km2, type=pa.float64())
        else:
            columns[name] = _science_array(name)
    return pa.table(columns).select(list(FORMAL_FEATURE_COLUMNS))


def test_frozen_formal_calendar_is_exactly_fit50_plus_assessment103() -> None:
    protocol = load_stage4_protocol_bundle(ROOT)
    calendar = FormalIssueCalendar.from_protocol(protocol)

    assert len(calendar.fit_issue_ids) == FORMAL_FIT_ISSUE_COUNT
    assert len(calendar.assessment_issue_ids) == FORMAL_ASSESSMENT_ISSUE_COUNT
    assert len(calendar.issue_ids) == FORMAL_POOL_ISSUE_COUNT
    assert calendar.issue_ids[0] == "anomaly-issue-2022-07-21"
    assert calendar.issue_ids[-1] == "anomaly-issue-2025-06-26"
    assert all(left < right for left, right in pairwise(calendar.issue_times_utc))


def test_calendar_rejects_missing_reordered_and_hindsight_periods() -> None:
    protocol = load_stage4_protocol_bundle(ROOT)
    calendar = FormalIssueCalendar.from_protocol(protocol)

    with pytest.raises(ValueError, match="exactly 50"):
        FormalIssueCalendar(
            fit_issue_ids=calendar.fit_issue_ids[:-1],
            assessment_issue_ids=calendar.assessment_issue_ids,
        )
    reordered = list(calendar.assessment_issue_ids)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(ValueError, match="reordered"):
        FormalIssueCalendar(
            fit_issue_ids=calendar.fit_issue_ids,
            assessment_issue_ids=tuple(reordered),
        )
    hindsight = (*calendar.assessment_issue_ids[:-1], "anomaly-issue-2025-07-03")
    with pytest.raises(ValueError, match="boundaries"):
        FormalIssueCalendar(
            fit_issue_ids=calendar.fit_issue_ids,
            assessment_issue_ids=hindsight,
        )


def test_grid_bridge_changes_only_role_digest_without_reordering() -> None:
    table = _table()
    grid = _grid()
    bridge = GridIdentityBridge.from_accepted_table(
        table,
        stage3_grid_id=STAGE3_GRID_ID,
        stage4_grid=grid,
    )

    projected = bridge.project(
        table,
        issue_time=datetime(2022, 7, 20, 16, tzinfo=UTC),
    )

    assert projected["grid_id"].combine_chunks().unique().to_pylist() == [grid.grid_id]
    unchanged = tuple(name for name in FORMAL_FEATURE_COLUMNS if name != "grid_id")
    assert selected_table_identity_sha256(table, unchanged) == (
        selected_table_identity_sha256(projected, unchanged)
    )
    assert tuple(projected["cell_id"].to_pylist()) == grid.cell_ids


def test_grid_bridge_rejects_cell_tampering_reordering_and_column_drift() -> None:
    table = _table()
    grid = _grid()
    bridge = GridIdentityBridge.from_accepted_table(
        table,
        stage3_grid_id=STAGE3_GRID_ID,
        stage4_grid=grid,
    )
    issue_time = datetime(2022, 7, 20, 16, tzinfo=UTC)

    query_index = table.schema.get_field_index("query_x_m")
    tampered = table.set_column(
        query_index,
        table.schema.field(query_index),
        pa.array([1.0, 3.5], type=pa.float64()),
    )
    with pytest.raises(ValueError, match="query_x_m"):
        bridge.project(tampered, issue_time=issue_time)
    with pytest.raises(ValueError, match="cell order"):
        bridge.project(table.take(pa.array([1, 0])), issue_time=issue_time)
    with pytest.raises(ValueError, match="missing, reordered, or drifted"):
        bridge.project(table.drop([FORMAL_FEATURE_COLUMNS[-1]]), issue_time=issue_time)
    with pytest.raises(ValueError, match="missing, reordered, or drifted"):
        bridge.project(
            table.select(list(reversed(FORMAL_FEATURE_COLUMNS))),
            issue_time=issue_time,
        )


def test_preflight_path_boundary_rejects_target_or_arbitrary_path_before_access() -> None:
    with pytest.raises(ValueError, match="target-blind allowlist"):
        resolve_score_blind_input_path(
            cast(Any, None),
            cast(Any, None),
            input_id=cast(Any, "earthquake_target"),
        )


def test_typed_receipt_round_trip_rejects_deterministic_and_resource_tampering() -> None:
    receipt = make_formal_preflight_receipt()
    document = receipt.as_mapping()

    loaded = FormalPreflightReceipt.from_mapping(document)
    assert loaded.content_sha256 == receipt.content_sha256
    assert loaded.as_mapping() == document

    changed_resource = replace(
        receipt.space_placebo_resource_observation,
        elapsed_seconds=2.0,
    )
    with_changed_resource = replace(
        receipt,
        space_placebo_resource_observation=changed_resource,
    )
    assert with_changed_resource.content_sha256 == receipt.content_sha256
    assert changed_resource.content_sha256 != (
        receipt.space_placebo_resource_observation.content_sha256
    )

    deterministic_tamper = deepcopy(document)
    deterministic_tamper["projected_formal_tables_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="deterministic content digest"):
        FormalPreflightReceipt.from_mapping(deterministic_tamper)

    resource_tamper = deepcopy(document)
    cast(dict[str, object], resource_tamper["space_placebo_resource_observation"])[
        "elapsed_seconds_hex"
    ] = (3.0).hex()
    with pytest.raises(ValueError, match="resource observation content digest"):
        FormalPreflightReceipt.from_mapping(resource_tamper)

    target_tamper = deepcopy(document)
    target_tamper["target_bytes_read"] = True
    with pytest.raises(ValueError, match="must remain false"):
        FormalPreflightReceipt.from_mapping(target_tamper)


def test_logical_identity_normalizes_only_payload_bits_hidden_beneath_nulls() -> None:
    null_bitmap = pa.py_buffer(b"\x00")
    null_zero = pa.Array.from_buffers(
        pa.float64(),
        1,
        [null_bitmap, pa.py_buffer(bytes.fromhex("0000000000000000"))],
        null_count=1,
    )
    null_nan = pa.Array.from_buffers(
        pa.float64(),
        1,
        [null_bitmap, pa.py_buffer(bytes.fromhex("000000000000f87f"))],
        null_count=1,
    )
    accepted = pa.table({"value": null_zero})
    rebuilt = pa.table({"value": null_nan})

    assert null_zero.equals(null_nan)
    assert selected_table_identity_sha256(accepted, ("value",)) != (
        selected_table_identity_sha256(rebuilt, ("value",))
    )
    assert _logical_selected_table_identity_sha256(accepted, ("value",)) == (
        _logical_selected_table_identity_sha256(rebuilt, ("value",))
    )
    _assert_columns_exact(accepted, rebuilt, columns=("value",), label="null payload")

    valid_zero = pa.table({"value": pa.array([0.0], type=pa.float64())})
    valid_negative_zero = pa.table({"value": pa.array([-0.0], type=pa.float64())})
    valid_nan_payload_0 = pa.table(
        {
            "value": pa.Array.from_buffers(
                pa.float64(),
                1,
                [None, pa.py_buffer(bytes.fromhex("000000000000f87f"))],
            )
        }
    )
    valid_nan_payload_1 = pa.table(
        {
            "value": pa.Array.from_buffers(
                pa.float64(),
                1,
                [None, pa.py_buffer(bytes.fromhex("010000000000f87f"))],
            )
        }
    )
    for changed in (
        valid_zero,
        valid_negative_zero,
        valid_nan_payload_0,
        valid_nan_payload_1,
    ):
        assert _logical_selected_table_identity_sha256(changed, ("value",)) != (
            _logical_selected_table_identity_sha256(rebuilt, ("value",))
        )
    assert _logical_selected_table_identity_sha256(valid_zero, ("value",)) != (
        _logical_selected_table_identity_sha256(valid_negative_zero, ("value",))
    )
    assert _logical_selected_table_identity_sha256(valid_nan_payload_0, ("value",)) != (
        _logical_selected_table_identity_sha256(valid_nan_payload_1, ("value",))
    )
    with pytest.raises(ValueError, match="values, validity"):
        _assert_columns_exact(accepted, valid_zero, columns=("value",), label="validity")
    with pytest.raises(ValueError, match="types"):
        _assert_columns_exact(
            valid_zero,
            pa.table({"value": pa.array([0.0], type=pa.float32())}),
            columns=("value",),
            label="type",
        )

    ordered = pa.table({"left": pa.array([1]), "right": pa.array([2])})
    assert _logical_selected_table_identity_sha256(ordered, ("left", "right")) != (
        _logical_selected_table_identity_sha256(ordered, ("right", "left"))
    )


def test_host_memory_observation_is_available_on_windows() -> None:
    observation = _host_memory_observation()
    if os.name == "nt":
        assert observation.process_working_set_bytes is not None
        assert observation.process_working_set_bytes > 0
        assert observation.process_peak_working_set_bytes is not None
        assert observation.process_peak_working_set_bytes > 0
        assert observation.system_available_memory_bytes is not None
        assert observation.system_available_memory_bytes > 0


@pytest.mark.skipif(
    not ACTUAL_FEATURE_STORE.is_file(),
    reason="accepted local stage-3 feature store is not distributed with the public repository",
)
def test_actual_first_stage3_row_group_passes_exact_grid_identity_bridge() -> None:
    study_area = load_study_area(ROOT / "data/processed/china_mainland.geojson", EQUAL_AREA_CRS)
    primary = build_stage4_grid_family(study_area.geographic).primary_25km
    parquet = pq.ParquetFile(ACTUAL_FEATURE_STORE)
    table = parquet.read_row_group(0, columns=list(FORMAL_FEATURE_COLUMNS))

    bridge = GridIdentityBridge.from_accepted_table(
        table,
        stage3_grid_id=STAGE3_GRID_ID,
        stage4_grid=primary,
    )
    projected = bridge.project(
        table,
        issue_time=cast(datetime, table["issue_time_utc"][0].as_py()),
    )

    assert bridge.stage3_grid_id == STAGE3_GRID_ID
    assert primary.grid_id == ("5c47e6af48158136d3ab18bf63d5744c1cdb80c107b6d9d412cb071cc3b7b5bf")
    assert projected.num_rows == 15_697
    assert projected["grid_id"][0].as_py() == primary.grid_id
