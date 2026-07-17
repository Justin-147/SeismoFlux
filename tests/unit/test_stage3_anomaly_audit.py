from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
from shapely.geometry import box

from seismoflux.features.anomaly.audit import (
    AuditResult,
    parse_lineage_audit_plan,
    run_lineage_replay_audit,
)
from seismoflux.features.anomaly.config import (
    AnomalyHistoryConfig,
    load_anomaly_history_config,
)
from seismoflux.features.anomaly.contracts import ANOMALY_STATE_HISTORY_CONTRACT
from seismoflux.features.anomaly.dictionary import DEFAULT_FEATURE_DICTIONARY
from seismoflux.features.anomaly.engine import Stage3FeatureEngine
from seismoflux.features.anomaly.grid import Stage3QueryGrid, build_stage3_query_grid
from seismoflux.features.anomaly.snapshot import build_issue_snapshots
from seismoflux.features.anomaly.state import (
    build_anomaly_state_history,
    state_records,
)

BEIJING = ZoneInfo("Asia/Shanghai")


@dataclass(frozen=True, slots=True)
class SyntheticAuditArtifacts:
    config: AnomalyHistoryConfig
    state_path: Path
    feature_path: Path
    query_grid: Stage3QueryGrid
    observation_table: pa.Table
    report_period_table: pa.Table


def _source_records(
    issue_dates: tuple[date, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    observations: list[dict[str, Any]] = []
    report_periods: list[dict[str, Any]] = []
    start_time = datetime.combine(
        min(issue_dates) - timedelta(days=60),
        time.min,
        BEIJING,
    )
    for index, issue_date in enumerate(sorted(issue_dates)):
        report_date = issue_date - timedelta(days=1)
        available_at = datetime.combine(issue_date, time.min, BEIJING).astimezone(UTC)
        report_periods.append(
            {
                "report_id": f"report-{index:02d}",
                "source_file": f"anomaly/report-{index:02d}.xlsx",
                "report_year": report_date.year,
                "report_period": index + 1,
                "report_date": report_date,
                "available_at": available_at,
                "row_count": 1,
                "row_report_date_mismatch_count": 0,
                "row_report_date_before_count": 0,
                "row_report_date_after_count": 0,
                "deformation_row_count": 1,
                "fluid_row_count": 0,
                "electromagnetic_row_count": 0,
                "cross_fault_row_count": 0,
            }
        )
        observations.append(
            {
                "observation_id": f"observation-{index:02d}",
                "anomaly_id": "synthetic-entity",
                "identity_complete": True,
                "report_date": report_date,
                "source_file": f"anomaly/report-{index:02d}.xlsx",
                "available_at": available_at,
                "station_id": "synthetic-station",
                "longitude": 110.0,
                "latitude": 35.0,
                "discipline": "形变",
                "measurement": "synthetic-measurement",
                "start_time": start_time,
                "is_listed": True,
                "report_state": "新增" if index == 0 else "持续",
                "reported_end_time": None,
                "right_censored": True,
                "reliability_flags": (),
            }
        )
    return observations, report_periods


def _build_artifacts(root: Path) -> SyntheticAuditArtifacts:
    config = load_anomaly_history_config("configs/anomaly_history.yaml")
    plan = parse_lineage_audit_plan(config)
    observations, report_periods = _source_records(plan.issue_dates_local)
    states = build_anomaly_state_history(
        observations,
        report_periods,
        expected_report_period_count=12,
    )
    snapshots = build_issue_snapshots(states, expected_issue_count=12)
    query_grid = build_stage3_query_grid(box(109.7, 34.7, 110.3, 35.3))
    engine = Stage3FeatureEngine(snapshots, query_grid, query_chunk_size=2)
    feature_groups = [engine.build_next_issue().table for _ in snapshots]

    state_path = root / "anomaly_state_history.parquet"
    feature_path = root / "anomaly_feature_store.parquet"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_table = pa.Table.from_pylist(
        list(state_records(states)),
        schema=ANOMALY_STATE_HISTORY_CONTRACT.schema,
    )
    pq.write_table(state_table, state_path, row_group_size=2)
    feature_table = pa.concat_tables(feature_groups)
    pq.write_table(
        feature_table,
        feature_path,
        row_group_size=query_grid.cell_count,
    )
    return SyntheticAuditArtifacts(
        config=config,
        state_path=state_path,
        feature_path=feature_path,
        query_grid=query_grid,
        observation_table=pa.Table.from_pylist(observations),
        report_period_table=pa.Table.from_pylist(report_periods),
    )


@pytest.fixture(scope="module")
def audit_artifacts(tmp_path_factory: pytest.TempPathFactory) -> SyntheticAuditArtifacts:
    return _build_artifacts(tmp_path_factory.mktemp("stage3-lineage-audit"))


def _copy_artifacts(
    source: SyntheticAuditArtifacts,
    root: Path,
) -> SyntheticAuditArtifacts:
    state_path = root / source.state_path.name
    feature_path = root / source.feature_path.name
    shutil.copyfile(source.state_path, state_path)
    shutil.copyfile(source.feature_path, feature_path)
    return replace(source, state_path=state_path, feature_path=feature_path)


def _run(artifacts: SyntheticAuditArtifacts, **changes: object) -> AuditResult:
    arguments: dict[str, object] = {
        "config": artifacts.config,
        "anomaly_state_history_path": artifacts.state_path,
        "anomaly_feature_store_path": artifacts.feature_path,
        "query_grid": artifacts.query_grid,
        "observation_table": artifacts.observation_table,
        "report_period_table": artifacts.report_period_table,
        "query_chunk_size": 2,
    }
    arguments.update(changes)
    return run_lineage_replay_audit(**arguments)  # type: ignore[arg-type]


def _mutate_first_issue_feature(path: Path, column_name: str, replacement: object) -> None:
    table = pq.read_table(path)
    field = table.schema.field(column_name)
    mask = pc.equal(table["issue_index"], pa.scalar(0, type=pa.int16()))
    column = pc.if_else(mask, pa.scalar(replacement, type=field.type), table[column_name])
    mutated = table.set_column(table.schema.get_field_index(column_name), field, column)
    pq.write_table(mutated, path)


def test_plan_parses_exactly_twelve_frozen_local_dates() -> None:
    plan = parse_lineage_audit_plan(load_anomaly_history_config("configs/anomaly_history.yaml"))

    assert len(plan.issue_dates_local) == 12
    assert len(set(plan.issue_dates_local)) == 12
    assert {item.isoformat() for item in plan.issue_dates_local} == {
        "2022-07-21",
        "2022-08-25",
        "2023-03-16",
        "2023-04-27",
        "2023-05-18",
        "2023-06-29",
        "2024-07-04",
        "2024-10-17",
        "2024-11-07",
        "2025-02-27",
        "2025-04-17",
        "2025-06-26",
    }


def test_audit_rehydrates_replays_and_returns_only_public_aggregates(
    audit_artifacts: SyntheticAuditArtifacts,
) -> None:
    result = _run(audit_artifacts)
    payload = asdict(result)

    assert result.passed is True
    assert result.selected_issue_count == 12
    assert result.selected_feature_row_count == 12
    assert 1 <= result.unique_selected_cell_count <= 12
    assert result.state_row_count_checked == 24
    assert result.observation_reference_count_checked == 12
    assert result.report_reference_count_checked == 12
    assert result.dictionary_definition_count == len(DEFAULT_FEATURE_DICTIONARY.definitions)
    assert result.dictionary_value_column_count == len(
        DEFAULT_FEATURE_DICTIONARY.storage_value_columns()
    )
    assert result.feature_scalar_count_compared == 12 * result.feature_field_count
    assert result.nullable_value_count_checked > result.null_value_count_checked > 0
    assert result.trajectory_value_count_checked > 0
    assert all(isinstance(value, bool | int) for value in payload.values())
    assert not any(
        token in key
        for key in payload
        for token in ("cell_id", "coordinate", "source_id", "observation_id", "report_id")
    )


def test_audit_rejects_a_dangling_state_source_reference(
    audit_artifacts: SyntheticAuditArtifacts,
    tmp_path: Path,
) -> None:
    artifacts = _copy_artifacts(audit_artifacts, tmp_path)
    table = pq.read_table(artifacts.state_path)
    rows = table.to_pylist()
    entity = next(row for row in rows if row["state_row_kind"] == "entity_state")
    entity["source_observation_ids"] = ["missing-observation"]
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), artifacts.state_path)

    with pytest.raises(ValueError, match="lineage aliases disagree"):
        _run(artifacts)


def test_audit_rejects_source_available_after_issue(
    audit_artifacts: SyntheticAuditArtifacts,
) -> None:
    rows = audit_artifacts.observation_table.to_pylist()
    rows[0]["available_at"] = rows[0]["available_at"] + timedelta(days=1)
    future_table = pa.Table.from_pylist(rows, schema=audit_artifacts.observation_table.schema)

    with pytest.raises(ValueError, match="observation later than its issue"):
        _run(audit_artifacts, observation_table=future_table)


def test_audit_rejects_feature_value_that_does_not_exactly_replay(
    audit_artifacts: SyntheticAuditArtifacts,
    tmp_path: Path,
) -> None:
    artifacts = _copy_artifacts(audit_artifacts, tmp_path)
    _mutate_first_issue_feature(
        artifacts.feature_path,
        "radius_50km__listed_count",
        999.0,
    )

    with pytest.raises(ValueError, match="feature replay differs"):
        _run(artifacts)


def test_audit_rejects_null_with_zero_reason(
    audit_artifacts: SyntheticAuditArtifacts,
    tmp_path: Path,
) -> None:
    artifacts = _copy_artifacts(audit_artifacts, tmp_path)
    name = "radius_50km__listed_count__slope_4w_per_week__null_reason_code"
    _mutate_first_issue_feature(artifacts.feature_path, name, 0)

    with pytest.raises(ValueError, match="null feature value must use a nonzero reason"):
        _run(artifacts)


def test_audit_rejects_inconsistent_trajectory_sample_count(
    audit_artifacts: SyntheticAuditArtifacts,
    tmp_path: Path,
) -> None:
    artifacts = _copy_artifacts(audit_artifacts, tmp_path)
    name = "radius_50km__listed_count__slope_4w_per_week__sample_count"
    _mutate_first_issue_feature(artifacts.feature_path, name, -1)

    with pytest.raises(ValueError, match="validity and sample count are inconsistent"):
        _run(artifacts)


def test_audit_rejects_dictionary_column_omission(
    audit_artifacts: SyntheticAuditArtifacts,
    tmp_path: Path,
) -> None:
    artifacts = _copy_artifacts(audit_artifacts, tmp_path)
    table = pq.read_table(artifacts.feature_path).drop(["radius_50km__listed_count"])
    pq.write_table(table, artifacts.feature_path)

    with pytest.raises(ValueError, match="schema differs from the dictionary"):
        _run(artifacts)
