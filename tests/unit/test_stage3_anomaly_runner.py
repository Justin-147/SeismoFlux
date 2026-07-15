from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import seismoflux.features.anomaly.runner as runner_module
from seismoflux.background.execution import RepositoryIdentity
from seismoflux.background.publication import FixedFileConflictError
from seismoflux.config import sha256_file
from seismoflux.features.anomaly.audit import AuditResult
from seismoflux.features.anomaly.dictionary import (
    DEFAULT_FEATURE_DICTIONARY,
    build_feature_store_schema,
)
from seismoflux.features.anomaly.engine import (
    Stage3IssueFeatureAudit,
    Stage3IssueFeatureTable,
)
from seismoflux.features.anomaly.grid import Stage3QueryGrid
from seismoflux.features.anomaly.input import LoadedStage3Inputs
from seismoflux.features.anomaly.public_deliverables import (
    PUBLIC_AUDIT_SVG_PATH,
    PUBLIC_DICTIONARY_PATH,
    PUBLIC_REGISTRY_PATH,
    PUBLIC_REPORT_PATH,
)
from seismoflux.features.anomaly.storage import Stage3DatasetArtifact
from seismoflux.features.anomaly.synthetic_audit import SyntheticPrefixAuditResult


def _ready_repository(freeze_tag: str) -> RepositoryIdentity:
    return RepositoryIdentity(
        code_commit="a" * 40,
        branch="codex/stage3-anomaly-features",
        upstream="origin/codex/stage3-anomaly-features",
        upstream_commit="a" * 40,
        freeze_tag=freeze_tag,
        freeze_tag_commit="b" * 40,
        git_available=True,
        worktree_clean=True,
        tag_is_ancestor=True,
        upstream_matches_head=True,
    )


def _one_cell_grid() -> Stage3QueryGrid:
    return Stage3QueryGrid(
        grid_id="c" * 64,
        equal_area_crs="synthetic_equal_area",
        cell_size_km=25.0,
        cell_ids=("cell-0",),
        rows=np.asarray([0], dtype=np.int64),
        columns=np.asarray([0], dtype=np.int64),
        query_xy_m=np.asarray([[0.0, 0.0]], dtype=np.float64),
        clipped_area_km2=np.asarray([625.0], dtype=np.float64),
    )


def test_stage3_plan_is_score_free_machine_readable_and_reserves_cores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_: object, **__: object) -> object:
        raise AssertionError("stage-3 plan must not load Parquet rows or write an artifact")

    monkeypatch.setattr(pq, "read_table", forbidden)
    monkeypatch.setattr(runner_module, "write_parquet_row_groups_atomic", forbidden)
    monkeypatch.setattr(runner_module, "stage3_bundle_workspace", forbidden)

    plan = runner_module.build_stage3_plan(physical_core_probe=lambda: 8)
    details = plan.to_manifest_details()

    assert plan.input_spec_dataset_names == (
        "anomaly_observation",
        "anomaly_report_period",
    )
    assert plan.resources.detected_physical_cores == 8
    assert plan.resources.reserve_physical_cores == 2
    assert plan.resources.effective_workers == 6
    assert plan.resources.spatial_workers == 2
    assert details["expected_actual_snapshot_count"] == 205
    assert details["locked_test"] == {
        "action": "do_not_run",
        "run": False,
        "target_count": None,
        "target_ids": [],
        "score_ids": [],
        "artifact_ids": [],
        "result": None,
    }
    forbidden_inputs = cast(list[str], details["forbidden_inputs"])
    assert "earthquake_targets_or_labels" in forbidden_inputs


@pytest.mark.parametrize("detected_physical_cores", (None, 1, 2))
def test_stage3_resource_plan_fails_closed_when_two_core_reserve_is_unverifiable(
    detected_physical_cores: int | None,
) -> None:
    with pytest.raises(ValueError, match="physical-core|reserved-core"):
        runner_module.build_stage3_plan(physical_core_probe=lambda: detected_physical_cores)


def test_stage3_resource_plan_uses_one_worker_at_minimum_safe_core_count() -> None:
    plan = runner_module.build_stage3_plan(physical_core_probe=lambda: 3)

    assert plan.resources.detected_physical_cores == 3
    assert plan.resources.reserve_physical_cores == 2
    assert plan.resources.effective_workers == 1
    assert plan.resources.spatial_workers == 1


def test_stage3_identity_has_exact_eleven_frozen_input_hashes() -> None:
    plan = runner_module.build_stage3_plan(physical_core_probe=lambda: 4)
    repository = _ready_repository(plan.config.freeze_tag)
    grid = _one_cell_grid()

    identity = runner_module._identity_payload(
        plan,
        repository,
        dictionary=DEFAULT_FEATURE_DICTIONARY,
        query_grid=grid,
    )

    assert set(identity) == {
        "schema_version",
        "protocol_version",
        "execution_mode",
        "protocol_freeze_tag",
        "code_commit",
        "feature_dictionary_sha256",
        "grid",
        "input_hashes",
    }
    assert identity["grid"] == {
        "grid_id": "c" * 64,
        "cell_count": 1,
        "cell_size_km": 25.0,
    }
    input_hashes = cast(dict[str, str], identity["input_hashes"])
    assert set(input_hashes) == {
        "protocol_bytes_sha256",
        "anomaly_history_config_bytes_sha256",
        "environment_lock_sha256",
        "data_catalog_sha256",
        "anomaly_observation_file_sha256",
        "anomaly_observation_content_sha256",
        "anomaly_observation_schema_sha256",
        "anomaly_report_period_file_sha256",
        "anomaly_report_period_content_sha256",
        "anomaly_report_period_schema_sha256",
        "study_area_sha256",
    }


def _install_synthetic_formal_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    audit_passes: bool = True,
) -> tuple[runner_module.Stage3RunPlan, list[str]]:
    actual_plan = runner_module.build_stage3_plan(physical_core_probe=lambda: 8)
    plan = replace(actual_plan, project_root=tmp_path)
    repository = _ready_repository(plan.config.freeze_tag)
    events: list[str] = []

    monkeypatch.setattr(runner_module, "build_stage3_plan", lambda *_args, **_kwargs: plan)

    def repository_identity(*_: object, **__: object) -> RepositoryIdentity:
        events.append("repository")
        return repository

    monkeypatch.setattr(runner_module, "require_repository_identity", repository_identity)
    monkeypatch.setattr(runner_module, "_verify_frozen_files", lambda _plan: None)

    loaded = LoadedStage3Inputs(
        observation_table=pa.table({"observation_id": pa.array([], type=pa.string())}),
        report_period_table=pa.table({"report_id": pa.array([], type=pa.string())}),
        study_area_path=tmp_path / "study.geojson",
        study_area_document={
            "type": "Feature",
            "properties": {"target_independent": True, "predictor_feature": False},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
            },
        },
        data_catalog_sha256=plan.input_spec.data_catalog_sha256,
        stage1_snapshot_id=plan.input_spec.expected_stage1_snapshot_id,
    )

    def load_inputs(*_: object, **__: object) -> LoadedStage3Inputs:
        events.append("inputs")
        return loaded

    monkeypatch.setattr(runner_module, "load_stage3_inputs", load_inputs)

    first = date(2022, 7, 20)
    report_dates = [first + timedelta(days=index) for index in range(205)]
    report_dates[-1] = date(2026, 7, 1)
    states = tuple(
        SimpleNamespace(
            state_row_kind="report_period_summary",
            identity_complete=False,
            reliability_grade="excluded",
            issue_report_date=report_dates[index],
            issue_time_utc=datetime.combine(report_dates[index], datetime.min.time(), UTC),
            max_source_available_at=datetime.combine(
                report_dates[index],
                datetime.min.time(),
                UTC,
            ),
        )
        for index in range(205)
    )
    monkeypatch.setattr(
        runner_module,
        "build_anomaly_state_history",
        lambda *_args, **_kwargs: states,
    )
    snapshots = tuple(range(205))
    monkeypatch.setattr(
        runner_module,
        "build_issue_snapshots",
        lambda *_args, **_kwargs: snapshots,
    )
    grid = _one_cell_grid()
    monkeypatch.setattr(runner_module, "build_stage3_query_grid", lambda _geometry: grid)

    class FakeEngine:
        def __init__(self, *_args: object, **kwargs: object) -> None:
            dictionary = kwargs["dictionary"]
            self.schema = build_feature_store_schema(dictionary)  # type: ignore[arg-type]
            self._next = 0

        @property
        def next_issue_index(self) -> int:
            return self._next

        def build_next_issue(self) -> Stage3IssueFeatureTable:
            index = self._next
            self._next += 1
            return Stage3IssueFeatureTable(
                table=pa.Table.from_batches([], schema=self.schema),
                audit=Stage3IssueFeatureAudit(
                    issue_index=index,
                    issue_report_id=f"report-{index:03d}",
                    row_count=1,
                    entity_state_count=0,
                    spatial_entity_count=0,
                    missing_coordinate_count=0,
                    trailing_reporting_record_count=0,
                    nullable_value_count=0,
                    null_value_count=0,
                ),
            )

    monkeypatch.setattr(runner_module, "Stage3FeatureEngine", FakeEngine)
    schemas: dict[Path, pa.Schema] = {}

    def fake_write(**kwargs: Any) -> Stage3DatasetArtifact:
        name = kwargs["name"]
        output_path = Path(kwargs["output_path"])
        schema = kwargs["schema"]
        if name == "anomaly_feature_store":
            list(kwargs["row_groups"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(name.encode("ascii"))
        schemas[output_path] = schema
        row_count = 205
        return Stage3DatasetArtifact(
            name=name,
            path=output_path.resolve().relative_to(tmp_path).as_posix(),
            row_count=row_count,
            row_group_count=205,
            file_size_bytes=output_path.stat().st_size,
            file_sha256=sha256_file(output_path),
            content_sha256=("d" if name == "anomaly_state_history" else "e") * 64,
            schema_sha256=("f" if name == "anomaly_state_history" else "1") * 64,
            sort_keys=tuple(kwargs["sort_keys"]),
        )

    monkeypatch.setattr(runner_module, "write_parquet_row_groups_atomic", fake_write)
    monkeypatch.setattr(pq, "read_schema", lambda path: schemas[Path(path)])
    replay = AuditResult(
        passed=audit_passes,
        selected_issue_count=12,
        selected_feature_row_count=12,
        unique_selected_cell_count=12,
        state_row_count_checked=12,
        observation_reference_count_checked=0,
        report_reference_count_checked=12,
        dictionary_definition_count=len(DEFAULT_FEATURE_DICTIONARY.definitions),
        dictionary_value_column_count=len(DEFAULT_FEATURE_DICTIONARY.storage_value_columns()),
        feature_field_count=len(build_feature_store_schema()),
        feature_scalar_count_compared=12 * len(build_feature_store_schema()),
        nullable_value_count_checked=0,
        null_value_count_checked=0,
        trajectory_value_count_checked=0,
    )
    monkeypatch.setattr(runner_module, "run_lineage_replay_audit", lambda **_kwargs: replay)
    monkeypatch.setattr(
        runner_module,
        "run_synthetic_prefix_audit",
        lambda _config: SyntheticPrefixAuditResult(
            passed=True,
            seed_count=32,
            invariant_count=7,
            check_count=224,
            failure_count=0,
        ),
    )
    monkeypatch.setattr(
        runner_module,
        "verify_stage3_parquet_artifact",
        lambda **_kwargs: [],
    )
    return plan, events


def test_formal_stage3_run_gates_before_inputs_and_uses_final_manifest_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan, events = _install_synthetic_formal_run(monkeypatch, tmp_path)

    result = runner_module.run_anomaly_history_stage3(
        memory_probe=lambda: 1024,
        physical_core_probe=lambda: 8,
    )

    assert events[:2] == ["repository", "inputs"]
    assert result.reused_existing_bundle is False
    assert result.audit["actual_snapshot_count"] == 205
    assert result.audit["known_missing_period_count"] == 2
    assert result.audit["missing_period_zero_imputation_count"] == 0
    expected_root = f"data/processed/stage3/anomaly_history/{result.bundle_id}"
    assert result.state_artifact.path == f"{expected_root}/anomaly_state_history.parquet"
    assert result.feature_artifact.path == f"{expected_root}/anomaly_feature_store.parquet"
    assert ".tmp" not in result.state_artifact.path
    assert ".tmp" not in result.feature_artifact.path
    destination = tmp_path / expected_root
    assert destination.is_dir()
    assert result.bundle_path == expected_root
    assert result.repository.freeze_tag == plan.config.freeze_tag
    assert result.public_publication is not None
    assert result.public_publication.registry.created is True
    for reference in (
        PUBLIC_REGISTRY_PATH,
        PUBLIC_REPORT_PATH,
        PUBLIC_AUDIT_SVG_PATH,
        PUBLIC_DICTIONARY_PATH,
    ):
        assert (tmp_path / reference).is_file()

    reused = runner_module.run_anomaly_history_stage3(
        memory_probe=lambda: 1024,
        physical_core_probe=lambda: 8,
    )
    assert reused.reused_existing_bundle is True
    assert reused.manifest_sha256 == result.manifest_sha256
    assert reused.public_publication is not None
    assert reused.public_publication.registry.created is False


def test_formal_stage3_public_conflict_is_never_overwritten(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_synthetic_formal_run(monkeypatch, tmp_path)
    runner_module.run_anomaly_history_stage3(
        memory_probe=lambda: 1024,
        physical_core_probe=lambda: 8,
    )
    report = tmp_path / PUBLIC_REPORT_PATH
    report.write_text("conflicting public report", encoding="utf-8")

    with pytest.raises(FixedFileConflictError):
        runner_module.run_anomaly_history_stage3(
            memory_probe=lambda: 1024,
            physical_core_probe=lambda: 8,
        )

    assert report.read_text(encoding="utf-8") == "conflicting public report"


def test_failed_replay_audit_publishes_no_stage3_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_synthetic_formal_run(monkeypatch, tmp_path, audit_passes=False)

    with pytest.raises(ValueError, match="lineage replay audit did not pass"):
        runner_module.run_anomaly_history_stage3(
            memory_probe=lambda: 1024,
            physical_core_probe=lambda: 8,
        )

    output_root = tmp_path / "data" / "processed" / "stage3" / "anomaly_history"
    assert not list(output_root.glob("anomaly-feature-bundle-*"))
    assert not list(output_root.glob(".*.lock"))
    assert not list(output_root.glob(".*.tmp"))
