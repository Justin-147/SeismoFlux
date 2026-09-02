from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
from pyproj import Transformer
from shapely.geometry import box
from shapely.strtree import STRtree

from seismoflux.multitask_s1.c2b_score import projected_near_cells, score_exposure
from seismoflux.multitask_s1.development_contract import DEVELOPMENT_FOLD_IDS
from seismoflux.multitask_s2 import slip_score as scorer
from seismoflux.multitask_s2.fault_geometry import blend_log_masses
from seismoflux.stage2s.contracts import SpatialGrid

MODELS = tuple(
    f"S2B_{layer}_{suffix}"
    for layer in ("COMMON_UNIT", "COMMON_GEO", "COMMON_GD", "NATIVE_UNIT", "NATIVE_GD")
    for suffix in ("ONLY", "CATALOG_MIX")
)

REFERENCES = (
    "C2B_D0_MULTISCALE",
    "C2B_D0_AGE_WEIGHTED",
    "C0_L3_B0_R30_CAUSAL",
    "C0_L0_UNIFORM",
)
BUDGETS = [300000.0, 450000.0, 600000.0, 750000.0, 960000.0]


@pytest.fixture(autouse=True)
def single_numeric_thread(monkeypatch):
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        monkeypatch.setenv(name, "1")


def _grid() -> SpatialGrid:
    return SpatialGrid(
        grid_id="synthetic-s2b-not-real",
        cell_size_km=25.0,
        cell_ids=("c0", "c1", "c2", "c3"),
        rows=np.zeros(4, dtype=np.int64),
        columns=np.arange(4, dtype=np.int64),
        query_xy_km=np.array([[0, 0], [25, 0], [50, 0], [75, 0]], dtype=float),
        clipped_area_km2=np.array([200000.0, 250000.0, 300000.0, 300000.0]),
    )


def _target(empty: bool = False) -> dict:
    target = {
        "event_ids": ["synthetic-event"],
        "episode_ids": ["synthetic-episode"],
        "global_episode_member_counts": [3],
        "is_episode_anchor": [True],
        "event_cell_indices": [2],
        "event_longitudes": [100.0],
        "event_latitudes": [30.0],
    }
    return {key: [] for key in target} if empty else target


def _reference_fixture() -> tuple[dict, dict, np.ndarray, STRtree]:
    fold = DEVELOPMENT_FOLD_IDS[0]
    targets = {
        (fold, 30, issue, band): _target(issue != 0 or band != "M5_6")
        for issue in (0, 86400000000)
        for band in scorer.BANDS
    }
    logs = np.log([0.4, 0.3, 0.2, 0.1])
    tree = STRtree(
        [
            box(0, 0, 25000, 25000),
            box(25000, 0, 50000, 25000),
            box(50000, 0, 75000, 25000),
            box(75000, 0, 100000, 25000),
        ]
    )
    near = projected_near_cells(tree, [100.0], [30.0])
    collections = {name: [] for name in scorer.TABLE_NAMES}
    for (fold_id, horizon, issue, band), target in targets.items():
        for model in REFERENCES:
            result = score_exposure(
                log_mass=logs,
                grid=_grid(),
                target=target,
                fold_id=fold_id,
                horizon_days=horizon,
                issue_time_us=issue,
                magnitude_bin=band,
                model_id=model,
                budgets=BUDGETS,
                near_cells=near if target["event_ids"] else [],
            )
            for name, rows in zip(scorer.TABLE_NAMES, result, strict=True):
                collections[name].extend(rows)
    return {name: pd.DataFrame(rows) for name, rows in collections.items()}, targets, logs, tree


@pytest.mark.parametrize("failure", ["verifier", "three_folds"])
def test_incomplete_four_fold_gate_precedes_every_target_or_old_score_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    module = ModuleType("seismoflux.multitask_s2.slip_predict")
    module.MODEL_IDS, module.HORIZONS, module.PROTOCOL_PATH = (
        MODELS,
        (7, 30, 90, 180, 365),
        Path("unused"),
    )
    module.load_protocol = lambda project: {"outputs": {"root": "outputs/multitask_s2/synthetic"}}

    def verify(*args):
        if failure == "verifier":
            raise ValueError("four folds incomplete")
        return {"folds": [{"fold_id": fold} for fold in DEVELOPMENT_FOLD_IDS[:3]]}

    module.verify_prediction_manifest = verify
    module.load_fold_arrays = lambda *args: pytest.fail("must not use an incomplete fold set")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(
        pd, "read_parquet", lambda *a, **k: pytest.fail("target-bearing table opened")
    )
    with pytest.raises(ValueError, match="four"):
        scorer.run_score_phase(project_root=tmp_path, data_root=tmp_path)
    assert not (tmp_path / "outputs/multitask_s2/synthetic/score_phase").exists()


def test_reference_reuse_preserves_empty_periods_targets_and_actual_area() -> None:
    tables, targets, _, _ = _reference_fixture()
    original = {name: frame.copy(deep=True) for name, frame in tables.items()}
    scorer._validate_references(tables, targets, REFERENCES, BUDGETS)
    for name, frame in tables.items():
        pd.testing.assert_frame_equal(frame, original[name])
    assert (tables[scorer.TABLE_NAMES[0]].event_count == 0).sum() == 120
    assert set(tables[scorer.TABLE_NAMES[0]].hit_tolerance_km) == {0.0, 70.0}
    damaged = {**tables, scorer.TABLE_NAMES[0]: tables[scorer.TABLE_NAMES[0]].iloc[:-1]}
    with pytest.raises(ValueError, match="zero-event"):
        scorer._validate_references(damaged, targets, REFERENCES, BUDGETS)


def test_alpha_zero_new_scores_are_numerically_identical_to_saved_catalog_component() -> None:
    references, targets, logs, tree = _reference_fixture()
    mixed = blend_log_masses(logs, np.log([0.1, 0.2, 0.3, 0.4]), 0.0)
    assert np.array_equal(mixed, logs)
    arrays = {
        "horizons_days": np.array([30, 30]),
        "issue_times_us": np.array([0, 86400000000]),
        "log_cell_mass": np.tile(mixed, (2, len(MODELS), 1)),
    }
    scored = scorer._score_fold(
        fold=DEVELOPMENT_FOLD_IDS[0],
        arrays=arrays,
        targets=targets,
        grid=_grid(),
        tree=tree,
        transformer=Transformer.from_pipeline("+proj=noop"),
        model_ids=MODELS,
        budgets=BUDGETS,
        reference_tables=references,
    )
    for name in scorer.TABLE_NAMES:
        expected = (
            references[name]
            .loc[references[name].model_id == REFERENCES[0]]
            .drop(columns="model_id")
            .reset_index(drop=True)
        )
        for model in MODELS:
            actual = (
                scored[name]
                .loc[scored[name].model_id == model]
                .drop(columns="model_id")
                .reset_index(drop=True)
            )
            pd.testing.assert_frame_equal(actual, expected)


def test_exactly_fourteen_pairs_do_not_append_old_c2b_comparisons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planned = [[f"model-{i}", "reference", f"purpose-{i}"] for i in range(14)]
    captured = {}

    def summarize(exposures, events, pairs, new_models):
        captured.update(pairs=pairs, new_models=new_models)
        return [], [], pd.DataFrame()

    monkeypatch.setattr(scorer.c2b_score, "summarize", summarize)
    scorer._summarize(pd.DataFrame(), pd.DataFrame(), {"planned_pairs": planned})
    assert captured == {"pairs": planned, "new_models": ()}
    with pytest.raises(ValueError, match="fourteen"):
        scorer._summarize(pd.DataFrame(), pd.DataFrame(), {"planned_pairs": planned[:-1]})


def test_resume_unsealed_tables_compares_content_without_overwriting_then_reuses_checkpoint(
    tmp_path: Path,
) -> None:
    frames, _, _, _ = _reference_fixture()
    fold, identity = DEVELOPMENT_FOLD_IDS[0], {"protocol": "synthetic-only"}
    first = tmp_path / "folds" / fold / scorer.TABLE_NAMES[0]
    scorer._write_or_compare_table(first, frames[scorer.TABLE_NAMES[0]])
    original, modified = first.read_bytes(), first.stat().st_mtime_ns
    scorer._fold_checkpoint(tmp_path, fold, identity, lambda: frames)
    assert first.read_bytes() == original and first.stat().st_mtime_ns == modified
    manifest = first.parent / "fold_score_manifest.json"
    assert b"\r\n" not in manifest.read_bytes()
    assert json.loads(manifest.read_text("utf-8"))["complete"] is True
    scorer._fold_checkpoint(
        tmp_path, fold, identity, lambda: pytest.fail("completed fold was recomputed")
    )
    assert first.read_bytes() == original and first.stat().st_mtime_ns == modified
    changed = frames[scorer.TABLE_NAMES[0]].copy()
    changed.loc[0, "anchor_hits"] = 999
    with pytest.raises(ValueError, match="differs"):
        scorer._write_or_compare_table(first, changed)
    assert first.read_bytes() == original


def test_grid_copy_is_byte_preserving_and_old_s1_cannot_be_output(tmp_path: Path) -> None:
    original = tmp_path / "source-grid"
    original.write_bytes(b"unchanged synthetic geometry")
    copy = tmp_path / "score/grid_geometry.parquet"
    scorer._copy_once(original, copy)
    assert original.read_bytes() == copy.read_bytes()
    modified = copy.stat().st_mtime_ns
    scorer._copy_once(original, copy)
    assert copy.stat().st_mtime_ns == modified
    with pytest.raises(ValueError, match="old S1 and S2A"):
        scorer._output_root(tmp_path, {}, tmp_path / "outputs/multitask_s1/old")
