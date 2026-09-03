"""Synthetic identity, empty-window, weighting and checkpoint checks only."""

import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from seismoflux.multitask_s0 import CATALOG_COLUMNS
from seismoflux.multitask_s3 import strata_runner as runner
from seismoflux.multitask_s3.calendar import FOLDS, HORIZONS, build_fold_calendar
from seismoflux.multitask_s3.case_ledger import VARIANTS
from seismoflux.multitask_s3.preparation import sha256, write_json
from seismoflux.multitask_s3.targets import prepare_anchor_ids


def catalog_row(event_id, date, magnitude=5.4):
    return {
        "event_id": event_id,
        "origin_time_utc": pd.Timestamp(date, tz="UTC"),
        "available_at": pd.Timestamp(date, tz="UTC"),
        "magnitude": magnitude,
        "longitude": 105.0,
        "latitude": 35.0,
        "inside_study_area": True,
    }


def paired_row(event_id="member", *, anchor=False, size=2, region="r1", gain=True):
    return {
        "event_id": event_id,
        "episode_id": "episode_" + event_id,
        "global_member_count": size,
        "is_anchor": anchor,
        "event_view": "anchor" if anchor else "subsequent",
        "fold_id": "A_DEV_2023_2024",
        "horizon_days": 7,
        "magnitude_band": "Ms5_6",
        "issue_time_utc": "2023-07-06T00:00:00+00:00",
        "primary_nonoverlap": True,
        "candidate": "CAT_DYN",
        "reference": "CATALOG",
        "mode": "strict",
        "area_budget_km2": 300000,
        "candidate_actual_area_km2": 299900.0,
        "reference_actual_area_km2": 299800.0,
        "candidate_hit": gain,
        "reference_hit": not gain,
        "target_cell_index": 0,
        "region_id": region,
    }


def test_membership_uses_full_history_per_band_and_does_not_chain_extend():
    frame = pd.DataFrame(
        [
            catalog_row("early", "2023-07-01"),
            catalog_row("member", "2023-07-20"),
            catalog_row("not_chained", "2023-08-05"),
            catalog_row("larger_band", "2023-07-10", 6.4),
        ]
    )
    membership = runner.episode_membership(frame)
    assert membership["member"]["global_member_count"] == 2
    assert membership["member"]["is_anchor"] is False
    assert membership["not_chained"]["is_anchor"] is True
    assert membership["larger_band"]["is_anchor"] is True
    existing = prepare_anchor_ids(frame)
    for band in runner.BANDS:
        assert {
            event
            for event, info in membership.items()
            if info["magnitude_band"] == band and info["is_anchor"]
        } == existing[band]
    attached = runner.attach_membership([paired_row()], membership, ["r1"])
    assert attached[0]["candidate_hit"] is True
    assert attached[0]["candidate_actual_area_km2"] == 299900
    assert runner.view_rows(attached, "episode_balanced")[0]["weight"] == 0.5
    assert runner.view_rows(attached, "anchor") == []


def test_no_regional_or_window_renormalization_and_empty_strata_are_na(monkeypatch):
    rows = runner.view_rows(
        [
            paired_row("gain", size=2),
            paired_row("loss", size=1, region="r2", gain=False),
        ],
        "episode_balanced",
    )
    captured = []
    monkeypatch.setattr(
        runner, "paired_uncertainty", lambda summary, **kw: captured.append(kw) or {"checked": True}
    )
    keys = ["A_DEV_2023_2024|2023-07-06T00:00:00+00:00", "A_DEV_2023_2024|empty"]
    result = runner.summarize_task(
        rows,
        issue_keys=keys,
        all_regions=["r1", "r2", "r3"],
        identity={"task": "fixed"},
        primary=True,
        member_counts={"episode_gain": 2, "episode_loss": 1},
    )
    assert result["national"]["total_weight"] == 1.5
    assert result["national"]["delta_weighted_hits"] == -0.5
    assert result["regions"][0]["total_weight"] == 0.5
    assert result["regions"][0]["gained_weight"] == 0.5
    assert result["empty_region_ids_NA"] == ["r3"]
    assert result["issues_without_selected_events"] == 1
    assert captured[0]["issue_keys"] == keys
    assert "_local" not in result["national"]
    runner.summarize_task(
        rows,
        issue_keys=keys,
        all_regions=["r1", "r2"],
        identity={},
        primary=False,
        member_counts={},
    )
    assert len(captured) == 1


def test_membership_mismatch_is_not_silently_relabelled_or_dropped():
    row = paired_row()
    info = {
        "member": {
            "magnitude_band": "Ms5_6",
            "is_anchor": True,
            "episode_id": "ep",
            "global_member_count": 2,
        }
    }
    with pytest.raises(ValueError, match="band or anchor"):
        runner.attach_membership([row], info, ["r1"])
    info["member"]["is_anchor"] = False
    with pytest.raises(ValueError, match="outside"):
        runner.attach_membership([{**row, "target_cell_index": -1}], info, ["r1"])


def test_mapping_is_complete_static_and_aligned_by_cell_not_row_order():
    cells = [f"cell_{index:02d}" for index in range(39)]
    mapping = pd.DataFrame(
        {"cell_id": cells, "construction_zone_id": [f"zone_{index:02d}" for index in range(39)]}
    )
    assert runner.align_region_aliases(mapping.iloc[::-1], cells)[0] == "atomic_block_01"
    assert runner.align_region_aliases(mapping, cells[::-1])[0] == "atomic_block_39"
    with pytest.raises(ValueError, match="differs"):
        runner.align_region_aliases(mapping.iloc[:-1], cells)
    with pytest.raises(ValueError, match="duplicate"):
        runner.align_region_aliases(pd.concat([mapping, mapping.iloc[:1]]), cells)


@pytest.fixture
def synthetic_trial(tmp_path, monkeypatch):
    project = tmp_path / "project"
    config = project / "configs/multitask_s3_anomaly.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        yaml.safe_dump(
            {
                "access": {"catalog": "synthetic.parquet"},
                "support": {"alarm_area_budgets_km2": [300000]},
            }
        ),
        encoding="utf-8",
    )
    root = project / "outputs/multitask_s3"
    prediction, scores, cases, output = (
        root / "pred",
        root / "score",
        root / "score/case",
        root / "score/strata",
    )
    prediction.mkdir(parents=True)
    cases.mkdir(parents=True)
    for name in ("strata_runner", "strata_summary", "strata_uncertainty"):
        path = project / f"src/seismoflux/multitask_s3/{name}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic source identity\n", encoding="utf-8")
    issue = datetime(2023, 7, 6, tzinfo=UTC)
    cutoff = datetime(2025, 7, 1, tzinfo=UTC)
    prepared = {
        "protocol_sha256": sha256(config),
        "catalog_identity": {"id": "synthetic"},
        "grid_id": "grid",
        "grid_cells": 39,
        "study_area_sha256": "area",
    }
    manifest = {
        "status": "predictions_complete",
        "identity": {"prepared_inputs": prepared},
        "truth_cutoff_utc": cutoff.isoformat(),
        "issue_times_utc": [issue.isoformat()],
    }
    write_json(prediction / "prediction_manifest.json", manifest)
    prediction_hash = sha256(prediction / "prediction_manifest.json")
    write_json(
        scores / "score_progress.json",
        {"status": "complete", "prediction_manifest_sha256": prediction_hash},
    )
    write_json(scores / "science_scores.json", {"synthetic": True})
    records = []
    for fold in FOLDS:
        for horizon in HORIZONS:
            cal = build_fold_calendar(
                [issue], fold_id=fold, horizon_days=horizon, truth_cutoff=cutoff
            )
            for date in cal.evaluation_issues:
                for band in runner.BANDS:
                    records.append(
                        {
                            "fold_id": fold,
                            "horizon_days": horizon,
                            "magnitude_band": band,
                            "issue_time_utc": date.isoformat(),
                            "primary_nonoverlap": date in cal.primary_evaluation_issues,
                            "target_event_ids": [],
                            "anchor_mask": [],
                            "spatial": {
                                variant: {
                                    "alarms": [
                                        {"area_budget_km2": 300000, "actual_area_km2": 299800}
                                    ]
                                }
                                for variant in VARIANTS
                            },
                        }
                    )
    diagnostic_path = scores / "event_diagnostics_local.json"
    write_json(diagnostic_path, {"local_only": True, "records": records})
    write_json(
        cases / "case_ledger_local.json",
        {
            "local_only": True,
            "status": "ledger_complete_no_case_selection",
            "events": {},
            "rows": [],
            "provenance": {
                "prediction_manifest_sha256": prediction_hash,
                "event_diagnostics_sha256": sha256(diagnostic_path),
                "science_scores_sha256": sha256(scores / "science_scores.json"),
                "catalog_identity": prepared["catalog_identity"],
                "truth_cutoff_utc": cutoff.isoformat(),
                "model_refitted": False,
                "hits_recomputed": False,
                "new_evaluation_role_accessed": False,
            },
        },
    )
    calls = []
    monkeypatch.setattr(
        runner, "verify_authoritative_catalog_identity", lambda _: prepared["catalog_identity"]
    )
    monkeypatch.setattr(
        runner,
        "load_development_catalog",
        lambda path, **kw: calls.append(kw) or pd.DataFrame(columns=list(CATALOG_COLUMNS)),
    )
    cells = [f"cell_{index:02d}" for index in range(39)]
    domain = SimpleNamespace(
        operational_grid=SimpleNamespace(grid_id="grid", cell_count=39, cell_ids=cells)
    )
    monkeypatch.setattr(runner, "load_verified_spatial_inputs", lambda _: (domain, None, "area"))
    mapping = tmp_path / runner.MAPPING_PATH
    mapping.parent.mkdir(parents=True)
    pd.DataFrame(
        {"cell_id": cells, "construction_zone_id": [f"zone_{i:02d}" for i in range(39)]}
    ).to_parquet(mapping)
    monkeypatch.setattr(runner, "MAPPING_SHA256", sha256(mapping))
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **kw: "synthetic_commit\n")
    kwargs = dict(
        project_root=project,
        data_root=tmp_path,
        prediction_dir=prediction,
        score_dir=scores,
        case_dir=cases,
        output_dir=output,
    )
    return kwargs, calls


def test_empty_windows_and_365_na_survive_complete_synthetic_trial(synthetic_trial):
    kwargs, calls = synthetic_trial
    result = runner.summarize_trial(**kwargs)
    assert result["status"] == "strata_complete"
    assert result["model_refitted"] is False and result["null_scores_read"] is False
    assert len(result["completed"]) == 15
    path = kwargs["output_dir"] / "A_DEV_2023_2024__h007.json"
    block = json.loads(path.read_text(encoding="utf-8"))
    assert {r["event_view"] for r in block["rows"]} == set(runner.VIEWS)
    assert {r["axis"] for r in block["rows"]} == set(runner.AXES)
    assert all(
        r["issue_count"] == 1 and r["national"]["delta_recall_pp"] is None for r in block["rows"]
    )
    assert all(len(r["empty_region_ids_NA"]) == 39 for r in block["rows"])
    assert (
        result["completed"]["A_DEV_2023_2024__h365"]["status"]
        == "no_complete_evaluation_windows_NA"
    )
    assert len(calls) == 1
    with pytest.raises(FileExistsError):
        runner.summarize_trial(**kwargs, resume=True)


def test_changed_source_identity_fails_before_catalog(synthetic_trial):
    kwargs, calls = synthetic_trial
    write_json(kwargs["score_dir"] / "science_scores.json", {"changed": True})
    with pytest.raises(ValueError, match="same completed"):
        runner.summarize_trial(**kwargs)
    assert not calls
    assert not kwargs["output_dir"].exists()


def test_omitted_empty_window_fails_before_catalog_even_with_updated_hash(synthetic_trial):
    kwargs, calls = synthetic_trial
    diagnostic = kwargs["score_dir"] / "event_diagnostics_local.json"
    data = json.loads(diagnostic.read_text())
    data["records"].pop()
    write_json(diagnostic, data)
    ledger_path = kwargs["case_dir"] / "case_ledger_local.json"
    ledger = json.loads(ledger_path.read_text())
    ledger["provenance"]["event_diagnostics_sha256"] = sha256(diagnostic)
    write_json(ledger_path, ledger)
    with pytest.raises(ValueError, match="complete frozen calendar"):
        runner.summarize_trial(**kwargs)
    assert not calls


def test_resume_preserves_completed_block_and_same_evidence(synthetic_trial, monkeypatch):
    kwargs, _ = synthetic_trial
    original = runner.summarize_block
    count = 0

    def interrupt(*a, **kw):
        nonlocal count
        count += 1
        if count == 2:
            raise InterruptedError("synthetic interruption")
        return original(*a, **kw)

    monkeypatch.setattr(runner, "summarize_block", interrupt)
    with pytest.raises(InterruptedError):
        runner.summarize_trial(**kwargs)
    path = kwargs["output_dir"] / "A_DEV_2023_2024__h007.json"
    before = path.stat().st_mtime_ns
    monkeypatch.setattr(runner, "summarize_block", original)
    result = runner.summarize_trial(**kwargs, resume=True)
    assert path.stat().st_mtime_ns == before
    assert len(result["completed"]) == 15


def test_no_output_can_escape_s3_workspace(synthetic_trial, tmp_path):
    kwargs, calls = synthetic_trial
    with pytest.raises(ValueError, match="local S3"):
        runner.summarize_trial(**{**kwargs, "output_dir": tmp_path / "outside"})
    assert not calls
