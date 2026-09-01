from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def _load_runner() -> object:
    path = Path(__file__).parents[2] / "scripts" / "run_multitask_s0.py"
    spec = importlib.util.spec_from_file_location("seismoflux_s0_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


def test_runner_forces_numeric_libraries_to_one_thread_and_rejects_scores() -> None:
    assert all(os.environ[name] == "1" for name in runner.NUMERIC_THREAD_ENVIRONMENT)
    runner._assert_score_blind({"sample_count": 3, "score_blind": True})
    with pytest.raises(runner.S0RunnerError, match="forbidden model-result key"):
        runner._assert_score_blind({"recall": 0.5})


def test_flatten_fold_horizon_preserves_unavailable_as_null() -> None:
    values = {
        "issue_target_pair_count": None,
        "unique_event_count": None,
        "episode_sampling_status": "unavailable_no_mature_issue",
        "touched_episode_count": None,
        "anchor_target_count": None,
        "subsequent_target_event_count": None,
        "episode_balanced_total_weight": None,
    }
    axis = {
        "statistical_status": "descriptive",
        "availability_status": "unavailable_no_mature_issue",
        "evaluable": False,
        "issue_count": 0,
        "magnitude_bins": {"m5_6": values, "m6_plus": values},
    }
    ledger = {
        "fold_maturity": [
            {
                "fold_id": "fold",
                "role": "audit",
                "horizons": {
                    "365": {
                        "operational_weekly": axis,
                        "primary_exposure": axis,
                    }
                },
            }
        ]
    }
    rows = runner.flatten_fold_horizon_ledger(ledger)
    assert len(rows) == 4
    assert all(row["evaluable"] is False for row in rows)
    assert all(row["unique_event_count"] is None for row in rows)
    assert all(row["anchor_target_count"] is None for row in rows)


class _FakeLocator:
    def locate_lonlat(self, longitude: float, latitude: float) -> int | None:
        del latitude
        index = int(longitude)
        return index if 0 <= index < 39 else None


def _six_time_blocks() -> dict[str, object]:
    folds = []
    for index in range(6):
        start_year = 2000 + index
        folds.append(
            {
                "id": f"fold_{index + 1}",
                "role": "development" if index < 4 else "audit",
                "target_block_start": f"{start_year}-01-01T00:00:00Z",
                "target_block_end_exclusive": (
                    "derived_from_catalog_truth_cutoff"
                    if index == 5
                    else f"{start_year + 1}-01-01T00:00:00Z"
                ),
            }
        )
    return {"catalog_time_folds": {"outer_folds": folds}}


def test_atomic_block_rows_are_anonymous_deterministic_and_score_blind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell_ids = tuple(f"cell_{index:02d}" for index in range(39))
    raw_zones = tuple(f"private_zone_{39 - index:02d}" for index in range(39))
    mapping_path = tmp_path / "cell_mapping.parquet"
    mapping = pd.DataFrame({"cell_id": cell_ids, "construction_zone_id": raw_zones})
    monkeypatch.setattr(runner.pd, "read_parquet", lambda *args, **kwargs: mapping.copy())
    grid = SimpleNamespace(
        cell_count=39,
        cell_ids=cell_ids,
        clipped_area_km2=np.arange(1.0, 40.0),
    )
    domain = SimpleNamespace(operational_grid=grid, locator=_FakeLocator())
    origins = pd.to_datetime(
        ["2000-02-01T00:00:00Z", "2000-03-01T00:00:00Z", "2000-04-01T00:00:00Z"],
        utc=True,
    )
    catalog = pd.DataFrame(
        {
            "event_id": ["m4", "m5", "m6"],
            "origin_time_utc": origins,
            "available_at": origins,
            "longitude": [0.0, 1.0, 2.0],
            "latitude": [0.0, 0.0, 0.0],
            "magnitude": [4.5, 5.5, 6.5],
            "inside_study_area": [True, True, True],
        }
    )
    arguments = {
        "config": _six_time_blocks(),
        "catalog": catalog,
        "catalog_cutoff": pd.Timestamp("2006-01-01T00:00:00Z"),
        "magnitude_bins": {
            "m4_plus": (4.0, None),
            "m5_6": (5.0, 6.0),
            "m6_plus": (6.0, None),
        },
        "spatial_domain": domain,
        "cell_mapping_path": mapping_path,
        "cell_mapping_sha256": "f" * 64,
    }
    first = runner.build_atomic_block_sample_water_levels(**arguments)
    second = runner.build_atomic_block_sample_water_levels(**arguments)
    assert first["public_row_content_sha256"] == second["public_row_content_sha256"]
    assert first["public_row_count"] == 39 * 7
    assert first["unlocated_event_count"] == 0
    rows = first["rows"]
    overall = [row for row in rows if row["time_block_id"] == "ALL_1970_PLUS"]
    assert sum(row["all_event_count"] for row in overall) == 3
    assert sum(row["m5_6_episode_anchor_count"] for row in overall) == 1
    assert sum(row["m6_plus_episode_anchor_count"] for row in overall) == 1
    encoded = json.dumps(rows, ensure_ascii=False)
    assert "private_zone" not in encoded
    assert '"cell_id"' not in encoded
    assert "longitude" not in encoded
    assert "latitude" not in encoded
    assert all(tuple(row) == runner.ATOMIC_BLOCK_LEDGER_FIELDS for row in rows)
