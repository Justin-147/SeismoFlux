from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from seismoflux.multitask_s1 import input_sensitivity_score as scorer
from seismoflux.multitask_s1.development_contract import DEVELOPMENT_FOLD_IDS
from seismoflux.stage2s.contracts import SpatialGrid


def test_renderer_preserves_all_models_empty_periods_and_offline_data(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[2] / "scripts/render_multitask_s1_c2a.py"
    spec = importlib.util.spec_from_file_location("c2a_renderer_synthetic_test", script)
    assert spec is not None and spec.loader is not None
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    grid = SpatialGrid(
        grid_id="synthetic-not-a-scientific-result",
        cell_size_km=25.0,
        cell_ids=("c0", "c1", "c2", "c3"),
        rows=np.zeros(4, dtype=np.int64),
        columns=np.arange(4, dtype=np.int64),
        query_xy_km=np.asarray([[0.0, 0.0], [25.0, 0.0], [50.0, 0.0], [75.0, 0.0]]),
        clipped_area_km2=np.asarray([200000.0, 250000.0, 300000.0, 300000.0]),
    )
    exposures, events, alarms = [], [], []
    for fold_index, fold in enumerate(DEVELOPMENT_FOLD_IDS):
        targets = {field: [] for field in scorer.TARGET_FIELDS}
        if fold_index < 2:
            index = fold_index * 2
            targets = {
                "event_ids": [f"synthetic-{index}"],
                "episode_ids": [f"episode-{index}"],
                "event_cell_indices": [index],
                "global_episode_member_counts": [1],
                "is_episode_anchor": [True],
                "event_longitudes": [100.125 + index],
                "event_latitudes": [30.25],
            }
        for model in scorer.ALL_MODELS:
            mass = np.asarray(
                [0.1, 0.2, 0.3, 0.4] if model.startswith("A_") else [0.4, 0.3, 0.2, 0.1]
            )
            exposure, event, alarm = scorer.score_exposure(
                mass,
                grid,
                targets,
                fold_id=fold,
                issue_time_us=fold_index * 86400000000,
                model_id=model,
            )
            exposures.extend(exposure)
            events.extend(event)
            alarms.extend(alarm)
    event_frame = pd.DataFrame(events)
    curves, pairings, _ = scorer.summarize_results(pd.DataFrame(exposures), event_frame)
    score_root = tmp_path / "synthetic_fixture_only/score_phase"
    score_root.mkdir(parents=True)
    summary = {
        "synthetic_fixture": True,
        "model_ids": list(scorer.ALL_MODELS),
        "horizon_days": 30,
        "fixed_anchor_episode_count": 2,
        "primary_exposure_count": 4,
        "curves": curves,
        "pairings": pairings,
        "holdout_read": False,
        "locked_test_run": False,
    }
    (score_root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    event_frame.to_parquet(score_root / "event_results.parquet", index=False)
    pd.DataFrame(alarms).to_parquet(score_root / "alarm_prefixes.parquet", index=False)
    pd.DataFrame(
        {
            "cell_index": range(4),
            "cell_id": grid.cell_ids,
            "longitude": [100.0, 101.0, 102.0, 103.0],
            "latitude": [30.0] * 4,
            "clipped_area_km2": grid.clipped_area_km2,
        }
    ).to_csv(score_root / "grid_cells.csv", index=False)
    references = [
        {"path": path.name, "sha256": scorer._sha256(path)} for path in score_root.iterdir()
    ]
    (score_root / "score_manifest.json").write_text(
        json.dumps({"complete": True, "artifacts": references}), encoding="utf-8"
    )
    page = renderer.render(score_root.parent)
    assert all((page.parent / name).is_file() for name in renderer.FILENAMES)
    text = page.read_text(encoding="utf-8")
    assert "合成测试示例" in text
    assert "fetch(" not in text and "cdn." not in text and "<script src=" not in text
    assert '"lon":100.125' in text  # Original epicentre is not replaced by grid longitude 100.
    data = renderer._replay_data(
        summary, event_frame, pd.DataFrame(alarms), pd.read_csv(score_root / "grid_cells.csv")
    )
    assert len(data["issues"]) == 4
    assert sum(not item["events"] for item in data["issues"]) == 2
    assert all(len(item["forecasts"]) == 9 for item in data["issues"])
    assert len(data["issues"][0]["forecasts"][scorer.ALL_MODELS[0]]["areas"]) == 5
