from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from seismoflux.multitask_s1 import input_sensitivity_score as scorer
from seismoflux.multitask_s1.development_contract import DEVELOPMENT_FOLD_IDS
from seismoflux.stage2s.contracts import SpatialGrid


def _grid() -> SpatialGrid:
    return SpatialGrid(
        grid_id="synthetic-input-sensitivity-grid",
        cell_size_km=25.0,
        cell_ids=("c0", "c1", "c2", "c3"),
        rows=np.zeros(4, dtype=np.int64),
        columns=np.arange(4, dtype=np.int64),
        query_xy_km=np.asarray([[0.0, 0.0], [25.0, 0.0], [50.0, 0.0], [75.0, 0.0]]),
        clipped_area_km2=np.asarray([200_000.0, 250_000.0, 300_000.0, 300_000.0]),
    )


def _targets(index: int | None = None) -> dict[str, list]:
    if index is None:
        return {field: [] for field in scorer.TARGET_FIELDS}
    return {
        "event_ids": [f"event-{index}"],
        "event_cell_indices": [index],
        "episode_ids": [f"episode-{index}"],
        "global_episode_member_counts": [1],
        "is_episode_anchor": [True],
        "event_longitudes": [100.0 + index],
        "event_latitudes": [30.0],
    }


def _raw_targets() -> tuple[pd.DataFrame, dict[str, tuple[int, ...]]]:
    rows = []
    issues = {}
    for fold_index, fold in enumerate(DEVELOPMENT_FOLD_IDS):
        issue = pd.Timestamp(f"{2000 + 5 * fold_index}-01-05T16:00:00+00:00")
        elapsed = issue.to_pydatetime() - datetime(1970, 1, 1, tzinfo=UTC)
        issues[fold] = ((elapsed.days * 86400 + elapsed.seconds) * 1_000_000,)
        targets = _targets(fold_index if fold_index < 2 else None)
        for model in scorer.BASE_MODELS:
            payload = {
                "fold_id": fold,
                "issue_time_utc": issue.isoformat(),
                "model_id": model,
                "horizon_days": 30,
                "catalog_delay_hours": 24,
                "hit_tolerance_km": 0.0,
                "metric": "spatial_log_density",
                "magnitude_bin": "M5_6",
                "event_count": len(targets["event_ids"]),
                **targets,
            }
            rows.append(
                {
                    "score_family": "location",
                    "fold_id": fold,
                    "issue_time_utc": issue,
                    "horizon_days": 30,
                    "model_id": model,
                    "payload_json": json.dumps(payload),
                }
            )
    return pd.DataFrame(rows), issues


@pytest.mark.parametrize(
    ("timestamp", "expected_microseconds"),
    [
        ("2000-01-05T16:00:00+00:00", 947_088_000_000_000),
        ("2000-01-06T00:00:00+08:00", 947_088_000_000_000),
        ("1970-01-01T00:00:00.000001+00:00", 1),
        ("1969-12-31T23:59:59.999999+00:00", -1),
    ],
)
def test_issue_keys_use_epoch_microseconds_not_timestamp_value_nanoseconds(
    timestamp: str, expected_microseconds: int
) -> None:
    assert scorer._issue_us(timestamp) == expected_microseconds
    assert scorer._issue_us(pd.Timestamp(timestamp)) == expected_microseconds


def test_target_identity_preserves_empty_periods_and_same_three_model_population() -> None:
    rows, issues = _raw_targets()
    result = scorer.validate_target_rows(
        rows, expected_issues=issues, cell_count=4, expected_anchor_count=2
    )
    assert len(result) == 4
    assert sum(not item["event_ids"] for item in result.values()) == 2
    changed = rows.copy()
    payload = json.loads(changed.loc[1, "payload_json"])
    payload["event_cell_indices"] = [3]
    changed.loc[1, "payload_json"] = json.dumps(payload)
    with pytest.raises(scorer.InputSensitivityScoreError, match="target keys or metadata differ"):
        scorer.validate_target_rows(
            changed, expected_issues=issues, cell_count=4, expected_anchor_count=2
        )


def test_real_duplicate_target_exposure_is_still_rejected() -> None:
    rows, issues = _raw_targets()
    duplicated = pd.concat([rows, rows.iloc[:1]], ignore_index=True)
    with pytest.raises(scorer.InputSensitivityScoreError, match="duplicate"):
        scorer.validate_target_rows(
            duplicated, expected_issues=issues, cell_count=4, expected_anchor_count=2
        )


def test_missing_empty_period_is_rejected() -> None:
    rows, issues = _raw_targets()
    with pytest.raises(scorer.InputSensitivityScoreError, match="including empty"):
        scorer.validate_target_rows(
            rows.iloc[:-1], expected_issues=issues, cell_count=4, expected_anchor_count=2
        )


def test_future_fold_and_changed_delay_are_rejected() -> None:
    rows, issues = _raw_targets()
    rows.loc[0, "fold_id"] = "C_HOLDOUT_2020_2022"
    with pytest.raises(scorer.InputSensitivityScoreError, match="non-development"):
        scorer.validate_target_rows(
            rows, expected_issues=issues, cell_count=4, expected_anchor_count=2
        )
    rows, issues = _raw_targets()
    payload = json.loads(rows.loc[0, "payload_json"])
    payload["catalog_delay_hours"] = 0
    rows.loc[0, "payload_json"] = json.dumps(payload)
    with pytest.raises(scorer.InputSensitivityScoreError, match="causal boundary"):
        scorer.validate_target_rows(
            rows, expected_issues=issues, cell_count=4, expected_anchor_count=2
        )


def test_metrics_and_alarm_prefixes_share_exact_model_specific_area_with_empty_period() -> None:
    mass = np.asarray([0.4, 0.3, 0.2, 0.1])
    exposures, events, alarms = scorer.score_exposure(
        mass,
        _grid(),
        _targets(0),
        fold_id=DEVELOPMENT_FOLD_IDS[0],
        issue_time_us=0,
        model_id=scorer.ALL_MODELS[0],
    )
    assert len(exposures) == len(events) == len(alarms) == 5
    main = next(row for row in exposures if row["area_budget_km2"] == 600_000.0)
    assert main["actual_area_km2"] == 450_000.0
    assert main["anchor_hits"] == main["anchor_total"] == 1
    assert next(row for row in alarms if row["area_budget_km2"] == 600_000.0)[
        "selected_cell_indices"
    ] == [0, 1]
    empty, no_events, empty_alarms = scorer.score_exposure(
        mass,
        _grid(),
        _targets(),
        fold_id=DEVELOPMENT_FOLD_IDS[0],
        issue_time_us=0,
        model_id=scorer.ALL_MODELS[0],
    )
    assert len(empty) == len(empty_alarms) == 5 and not no_events
    assert all(row["anchor_recall"] is None for row in empty)
    assert [row["actual_area_km2"] for row in empty] == [
        row["actual_area_km2"] for row in exposures
    ]


def test_paired_gained_lost_bootstrap_and_fold_directions_are_transparent() -> None:
    exposures, events = [], []
    for model in scorer.ALL_MODELS:
        mass = (
            np.asarray([0.1, 0.2, 0.3, 0.4])
            if model.startswith("A_")
            else np.asarray([0.4, 0.3, 0.2, 0.1])
        )
        for fold_index, target_index in enumerate((0, 2, None, None)):
            exposure, event, _ = scorer.score_exposure(
                mass,
                _grid(),
                _targets(target_index),
                fold_id=DEVELOPMENT_FOLD_IDS[fold_index],
                issue_time_us=fold_index,
                model_id=model,
            )
            exposures.extend(exposure)
            events.extend(event)
    curves, comparisons, paired = scorer.summarize_results(
        pd.DataFrame(exposures), pd.DataFrame(events)
    )
    assert len(curves) == 45 and len(comparisons) == 60
    main = next(
        row
        for row in comparisons
        if row["candidate_model_id"] == "A_L1_REGIONAL_CONSTANT"
        and row["reference_model_id"] == "C0_L1_REGIONAL_CONSTANT"
        and row["area_budget_km2"] == 600_000.0
    )
    assert main["gained"] == main["lost"] == 1
    assert main["net_hits"] == main["delta_recall_pp"] == 0
    assert main["bootstrap_ci95_pp"] == [-100.0, 100.0]
    assert [row["direction"] for row in main["per_fold"]] == [
        "negative",
        "positive",
        "not_evaluable",
        "not_evaluable",
    ]
    assert set(paired.outcome) >= {"gained", "lost", "unchanged_hit", "unchanged_miss"}
    repeated = scorer.summarize_results(pd.DataFrame(exposures), pd.DataFrame(events))[1]
    assert comparisons == repeated


def test_prediction_gate_runs_before_any_target_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / scorer.PROTOCOL_PATH).write_text(
        "execution:\n  output_root: outputs/synthetic\n", encoding="utf-8"
    )
    module = ModuleType("seismoflux.multitask_s1.input_sensitivity_predict")

    def reject_prediction(project_root: Path, output_root: Path) -> dict:
        raise scorer.InputSensitivityScoreError("predictions are incomplete")

    module.verify_prediction_manifest = reject_prediction
    monkeypatch.setitem(sys.modules, module.__name__, module)

    def forbidden_read(*args: object, **kwargs: object) -> None:
        pytest.fail("target table was opened before the prediction gate")

    monkeypatch.setattr(pd, "read_parquet", forbidden_read)
    with pytest.raises(scorer.InputSensitivityScoreError, match="predictions are incomplete"):
        scorer.run_score_phase(project_root=tmp_path, data_root=tmp_path)
    assert not (tmp_path / "outputs/synthetic/score_phase").exists()


def test_incomplete_output_is_not_overwritten_and_complete_output_is_hash_verified(
    tmp_path: Path,
) -> None:
    assert not scorer._reuse_completed_score(tmp_path / "absent", {})
    with pytest.raises(scorer.InputSensitivityScoreError, match="incomplete score output"):
        scorer._reuse_completed_score(tmp_path, {})
    result = tmp_path / "summary.json"
    result.write_text("{}", encoding="utf-8")
    identity = {"synthetic": True}
    required = (
        "summary.json",
        "exposure_scores.csv",
        "event_results.parquet",
        "paired_episode_results.csv",
        "alarm_prefixes.parquet",
        "grid_cells.csv",
    )
    for name in required[1:]:
        (tmp_path / name).write_text("synthetic", encoding="utf-8")
    scorer._write_json(
        tmp_path / "score_manifest.json",
        {
            "complete": True,
            "identity": identity,
            "artifacts": [
                {"path": name, "sha256": scorer._sha256(tmp_path / name)} for name in required
            ],
        },
    )
    assert scorer._reuse_completed_score(tmp_path, identity)
    result.write_text("changed", encoding="utf-8")
    with pytest.raises(scorer.InputSensitivityScoreError, match="SHA-256 changed"):
        scorer._reuse_completed_score(tmp_path, identity)
