"""Synthetic S2A selection and checkpoint checks; no scientific run or target scoring."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from seismoflux.multitask_s1 import c2b_predict as catalog_runner
from seismoflux.multitask_s2 import fault_predict as runner
from seismoflux.multitask_s2.fault_geometry import FaultSurfaces

ROOT = Path(__file__).resolve().parents[2]


def _components():
    return {name: np.log(np.array([0.25, 0.75])) for name in runner.COMPONENT_IDS}


def _sample(block="I1", issue="1988-01-01T00:00:00+00:00", *, available=None, cells=(0, 1)):
    time = datetime.fromisoformat(issue)
    end = time + timedelta(days=30)
    visible = end if available is None else available
    return runner.InnerSample(
        "C_DEV_2000_2004",
        block,
        30,
        time,
        end,
        _components(),
        np.array(cells, dtype=np.int64),
        np.full(len(cells), runner._epoch_us(visible), dtype=np.int64),
    )


def _fold():
    return {
        "id": "C_DEV_2000_2004",
        "outer_start": "2000-01-01T00:00:00+00:00",
        "inner_blocks": [
            {"id": "I1", "start": "1985-01-01T00:00:00+00:00", "end": "1989-12-31T00:00:00+00:00"},
            {"id": "I2", "start": "1990-01-01T00:00:00+00:00", "end": "1994-12-31T00:00:00+00:00"},
            {"id": "I3", "start": "1995-01-01T00:00:00+00:00", "end": "1999-12-31T00:00:00+00:00"},
        ],
    }


def _surfaces(probabilities=(0.75, 0.25)):
    fine = {
        source: {scale: np.log(np.array(probabilities)) for scale in (25.0, 75.0, 150.0)}
        for source in ("SIMPLE", "TRACE")
    }
    coarse = {
        source: {scale: np.log(np.array([0.5, 0.5])) for scale in (25.0, 75.0, 150.0)}
        for source in ("SIMPLE", "TRACE")
    }
    return FaultSurfaces(
        fine,
        coarse,
        {source: np.array([0.0, 75.0]) for source in fine},
        np.zeros(2, dtype=np.int64),
        {"role": "synthetic_geometry"},
    )


def _validation(block, counts, probabilities=(0.25, 0.75)):
    issue = datetime(1993 if block == "I2" else 1998, 1, 1, tzinfo=UTC)
    return runner.GeometryValidationSample(
        block,
        issue,
        issue + timedelta(days=30),
        np.log(np.array(probabilities)),
        np.array(counts, dtype=np.float64),
    )


def test_frozen_protocol_six_models_and_unchanged_396_pair_calendar():
    protocol = runner.load_protocol(ROOT)
    assert tuple(protocol["models"]) == runner.MODEL_IDS
    assert [
        sum(
            len(runner._expected_horizon_axis(ROOT, fold, horizon))
            for fold in runner.DEVELOPMENT_FOLD_IDS
        )
        for horizon in runner.HORIZONS
    ] == [176, 116, 56, 32, 16]


def test_finite_candidates_deduplicate_alpha_zero_and_preserve_tie_order():
    protocol = runner.load_protocol(ROOT)
    pure = runner.geometry_candidates("fault_only", protocol)
    mixed = runner.geometry_candidates("catalog_mixture", protocol)
    assert pure == [{"alpha": 1.0, "scale_km": scale} for scale in (75.0, 150.0, 25.0)]
    assert len(mixed) == 13
    assert mixed[0] == {"alpha": 0.0, "scale_km": None}
    assert sum(row["alpha"] == 0.0 for row in mixed) == 1
    assert mixed[1:4] == [{"alpha": 0.25, "scale_km": scale} for scale in (75.0, 150.0, 25.0)]
    with pytest.raises(ValueError, match="family"):
        runner.geometry_candidates("new_family", protocol)


def test_inner_catalog_separate_end_availability_and_training_block_cutoffs():
    fold = _fold()
    i2_cutoff = datetime(1989, 12, 2, tzinfo=UTC)
    i3_cutoff = datetime(1994, 12, 2, tzinfo=UTC)
    outer_cutoff = datetime(1999, 12, 2, tzinfo=UTC)
    samples = [
        _sample(),
        _sample(issue="1989-11-02T00:00:00+00:00", available=i2_cutoff),
        _sample(issue="1989-11-03T00:00:00+00:00"),
        _sample(available=i2_cutoff + timedelta(microseconds=1)),
        _sample("I2", "1993-01-01T00:00:00+00:00", available=i3_cutoff),
        _sample("I2", "1993-02-01T00:00:00+00:00", available=i3_cutoff + timedelta(microseconds=1)),
        _sample("I3", "1998-01-01T00:00:00+00:00", available=outer_cutoff),
        _sample(
            "I3", "1998-02-01T00:00:00+00:00", available=outer_cutoff + timedelta(microseconds=1)
        ),
    ]
    validation, branches = runner.build_inner_catalog_validation(
        samples, fold, np.ones(2), catalog_runner.load_protocol(ROOT)
    )
    i2, i3 = branches
    assert i2["train_blocks"] == ["I1"]
    assert i2["training"]["issue_count"] == 3  # late end removed, late labels retained as empty
    assert i2["training"]["target_count"] == 4
    assert i2["training"]["empty_issue_count"] == 1
    assert i2["catalog_multiscale_selection"]["selected"] == "K75"
    assert i2["catalog_multiscale_selection"]["status"] == "insufficient_nonempty_blocks_fixed_K75"
    assert i2["I2_explicit_K75_fallback"] is True
    assert i2["training_label_cutoff_utc"] == i2_cutoff.isoformat()
    assert i3["train_blocks"] == ["I1", "I2"]
    assert i3["training"]["issue_count"] == 6
    assert i3["training"]["target_count"] == 10
    assert i3["catalog_multiscale_selection"]["status"] == "selected_from_earlier_blocks"
    assert i3["training_label_cutoff_utc"] == i3_cutoff.isoformat()
    assert i3["validation_label_cutoff_utc"] == outer_cutoff.isoformat()
    assert i3["validation"]["issue_count"] == 2
    assert i3["validation"]["target_count"] == 2
    assert i3["validation"]["empty_issue_count"] == 1
    assert [row.target_counts.sum() for row in validation] == [2.0, 2.0, 2.0, 0.0]
    for row in validation[:2]:
        np.testing.assert_array_equal(row.catalog_log_mass, _components()["K75"])


def test_i3_catalog_is_selected_on_earlier_targets_not_its_own_validation_targets():
    samples = [
        _sample(cells=(0, 0, 0)),
        _sample("I2", "1993-01-01T00:00:00+00:00", cells=(0, 0, 0)),
        _sample("I3", "1998-01-01T00:00:00+00:00", cells=(1, 1, 1)),
    ]
    for sample in samples:
        sample.components["K25"] = np.log(np.array([0.99, 0.01]))
        sample.components["K150"] = np.log(np.array([0.01, 0.99]))
    validation, branches = runner.build_inner_catalog_validation(
        samples, _fold(), np.ones(2), catalog_runner.load_protocol(ROOT)
    )
    assert branches[0]["catalog_multiscale_selection"]["selected"] == "K75"
    assert branches[1]["catalog_multiscale_selection"]["selected"] == "fine"
    np.testing.assert_allclose(np.exp(validation[1].catalog_log_mass), [0.62, 0.38])


def test_invalid_validation_end_cannot_cross_outer_embargo():
    with pytest.raises(ValueError, match="outer embargo"):
        runner.build_inner_catalog_validation(
            [_sample("I3", "1999-11-03T00:00:00+00:00")],
            _fold(),
            np.ones(2),
            catalog_runner.load_protocol(ROOT),
        )


def test_one_nonempty_geometry_block_can_select_and_empty_periods_remain():
    result = runner.select_geometry_parameters(
        [_validation("I2", [1, 0]), _validation("I2", [0, 0]), _validation("I3", [0, 0])],
        _surfaces(),
        np.array([1.0, 2.0]),
        runner.load_protocol(ROOT),
    )
    selected = result["S2A_SIMPLE_CATALOG_MIX"]
    assert selected["selected"] == {"alpha": 1.0, "scale_km": 75.0}
    assert selected["validation_issue_counts"] == [2, 1]
    assert selected["validation_target_counts"] == [1, 0]
    assert selected["candidates"][0]["block_scores"][1] is None
    assert selected["status"] == "selected_from_nonempty_earlier_validation_blocks"


def test_selection_uses_equal_block_means_not_pooled_events_and_actual_area():
    result = runner.select_geometry_parameters(
        [_validation("I2", [100, 0]), _validation("I3", [0, 1])],
        _surfaces(),
        np.array([2.0, 4.0]),
        runner.load_protocol(ROOT),
    )["S2A_SIMPLE_CATALOG_MIX"]
    assert result["selected"] == {"alpha": 0.5, "scale_km": 75.0}
    base = result["candidates"][0]
    assert base["block_scores"] == pytest.approx([np.log(0.25 / 2), np.log(0.75 / 4)])
    assert base["mean"] == pytest.approx((np.log(0.25 / 2) + np.log(0.75 / 4)) / 2)


def test_no_geometry_labels_uses_fixed_scale_and_exact_catalog_without_dropping_empty_issues():
    protocol = runner.load_protocol(ROOT)
    base = np.log(np.array([0.25, 0.75]))
    surfaces = _surfaces()
    result = runner.select_geometry_parameters(
        [_validation("I2", [0, 0]), _validation("I3", [0, 0])],
        surfaces,
        np.ones(2),
        protocol,
    )
    assert result[runner.MODEL_IDS[0]]["selected"] == {"alpha": 1.0, "scale_km": 75.0}
    output = runner._prediction_models(base, surfaces, result, protocol)
    assert output.shape == (6, 2)
    for index in (1, 2, 4, 5):
        assert result[runner.MODEL_IDS[index]]["selected"] == {"alpha": 0.0, "scale_km": None}
        np.testing.assert_array_equal(output[index], base)
        assert output[index].tobytes() == base.tobytes()
    assert all(row["mean"] is None for row in result[runner.MODEL_IDS[0]]["candidates"])
    assert result[runner.MODEL_IDS[0]]["validation_issue_counts"] == [1, 1]


def test_numerical_ties_choose_small_alpha_then_scale_75():
    result = runner.select_geometry_parameters(
        [_validation("I2", [1, 1], probabilities=(0.75, 0.25))],
        _surfaces(),
        np.ones(2),
        runner.load_protocol(ROOT),
    )
    assert result["S2A_SIMPLE_CATALOG_MIX"]["selected"] == {"alpha": 0.0, "scale_km": None}
    assert result["S2A_SIMPLE_FAULT_ONLY"]["selected"] == {"alpha": 1.0, "scale_km": 75.0}


def _write_old_cache(root, *, identity=None, issue_us=0, saved_issue_us=None, logs=None):
    identity = {"old_run": "synthetic"} if identity is None else identity
    path = root / "component_cache" / f"issue_{issue_us}.npz"
    runner._atomic_npz(
        path,
        {
            "identity": np.asarray(
                hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            ),
            "issue_time_us": np.int64(issue_us if saved_issue_us is None else saved_issue_us),
            "log_masses": np.stack(list(_components().values())) if logs is None else logs,
            "history_counts": np.zeros(6, dtype=np.int64),
        },
    )
    return path, identity


def test_existing_component_cache_is_only_read_and_missing_issue_never_written(tmp_path):
    path, identity = _write_old_cache(tmp_path)
    before = (runner._sha(path), path.stat().st_mtime_ns)
    cache = runner.ReadOnlyComponentCache(tmp_path, identity, np.ones(2), expected_files=1)
    values = cache.get(datetime(1970, 1, 1, tzinfo=UTC))
    assert cache.audit["completed_component_count"] == 1
    assert all(not values[name].flags.writeable for name in runner.COMPONENT_IDS)
    with pytest.raises(FileNotFoundError, match="do not calculate"):
        cache.get(datetime(1970, 1, 2, tzinfo=UTC))
    assert len(list(path.parent.iterdir())) == 1
    assert (runner._sha(path), path.stat().st_mtime_ns) == before


@pytest.mark.parametrize("bad", ["identity", "issue", "shape", "changed_payload"])
def test_old_component_identity_issue_shape_and_post_inventory_mutation_are_rejected(tmp_path, bad):
    path, identity = _write_old_cache(
        tmp_path,
        saved_issue_us=1 if bad == "issue" else None,
        logs=np.zeros((10, 1)) if bad == "shape" else None,
    )
    if bad == "changed_payload":
        cache = runner.ReadOnlyComponentCache(tmp_path, identity, np.ones(2), expected_files=1)
        path.write_bytes(path.read_bytes() + b"changed")
        with pytest.raises(ValueError, match="identity"):
            cache.get(datetime(1970, 1, 1, tzinfo=UTC))
    else:
        with pytest.raises(ValueError):
            runner.ReadOnlyComponentCache(
                tmp_path,
                {"wrong": "identity"} if bad == "identity" else identity,
                np.ones(2),
                expected_files=1,
            )


def test_missing_or_incomplete_component_cache_does_not_create_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        runner.ReadOnlyComponentCache(tmp_path, {}, np.ones(2))
    assert not (tmp_path / "component_cache").exists()
    path, identity = _write_old_cache(tmp_path)
    with pytest.raises(ValueError, match="inventory"):
        runner.ReadOnlyComponentCache(tmp_path, identity, np.ones(2))
    assert list(path.parent.iterdir()) == [path]


def test_orphan_prediction_can_only_be_sealed_if_exactly_unchanged(tmp_path):
    path = tmp_path / "predictions.npz"
    arrays = {"log_cell_mass": np.log(np.array([[0.25, 0.75]]))}
    runner._save_prediction_payload(path, arrays)
    before = (runner._sha(path), path.stat().st_mtime_ns)
    runner._save_prediction_payload(path, arrays)
    with pytest.raises(ValueError, match="preserve the original"):
        runner._save_prediction_payload(path, {"log_cell_mass": np.log(np.array([[0.5, 0.5]]))})
    assert (runner._sha(path), path.stat().st_mtime_ns) == before


def test_partial_final_manifest_never_unlocks_target_scoring(tmp_path, monkeypatch):
    identity = {"protocol_id": "synthetic"}
    monkeypatch.setattr(runner, "load_protocol", lambda root: {})
    monkeypatch.setattr(runner, "_identity", lambda protocol: identity)
    (tmp_path / "prediction_manifest.json").write_text(
        json.dumps(
            {
                **identity,
                "folds": [{"fold_id": fold} for fold in runner.DEVELOPMENT_FOLD_IDS[:3]],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "_verify_fold", lambda *args: pytest.fail("partial run"))
    with pytest.raises(ValueError, match="all four"):
        runner.verify_prediction_manifest(tmp_path, tmp_path)


@pytest.mark.parametrize("workers", [0, 4, True, False])
def test_invalid_worker_counts_rejected_before_io(tmp_path, workers):
    with pytest.raises(ValueError, match="at most three"):
        runner.run_prediction_phase(project_root=tmp_path, data_root=tmp_path, workers=workers)


def test_output_root_cannot_point_at_old_s1_run(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "load_protocol", lambda project: {"outputs": {"root": "unused"}})
    for name in runner._NUMERICAL_ENV:
        monkeypatch.setenv(name, "1")
    path = tmp_path / "outputs" / "multitask_s1" / "old"
    with pytest.raises(ValueError, match="old S1 is read-only"):
        runner.run_prediction_phase(project_root=tmp_path, data_root=tmp_path, output_root=path)
    assert not path.exists()


def test_synthetic_four_fold_396_pairs_preserves_old_predictions_and_completed_resume(
    tmp_path, monkeypatch
):
    protocol = runner.load_protocol(ROOT)
    protocol["inputs"]["grid_cells"] = 2
    protocol["inputs"]["catalog_run"] = "outputs/multitask_s1/synthetic"
    old_root = tmp_path / protocol["inputs"]["catalog_run"]
    old_root.mkdir(parents=True)
    old_identity = {"protocol_id": "synthetic_old_run"}
    old_manifest = {"folds": [{"fold_id": fold} for fold in runner.DEVELOPMENT_FOLD_IDS]}
    runner._write_json(old_root / "prediction_manifest.json", old_manifest)
    runner._write_json(old_root / "score_phase" / "score_manifest.json", {"synthetic": True})
    runner._write_json(old_root / "run_identity.json", old_identity)
    for filename, field in (
        ("prediction_manifest.json", "catalog_prediction_manifest_sha256"),
        ("score_phase/score_manifest.json", "catalog_score_manifest_sha256"),
    ):
        protocol["inputs"][field] = runner._sha(old_root / filename)
    old_hashes = {
        path: (runner._sha(path), path.stat().st_mtime_ns) for path in old_root.rglob("*.json")
    }
    expected_axis = runner._expected_horizon_axis
    monkeypatch.setattr(
        runner,
        "_expected_horizon_axis",
        lambda project, fold, horizon: expected_axis(ROOT, fold, horizon),
    )
    monkeypatch.setattr(runner, "load_protocol", lambda project: protocol)
    monkeypatch.setattr(runner, "_identity", lambda value: {"protocol_id": "synthetic_s2a"})
    monkeypatch.setattr(catalog_runner, "load_protocol", lambda project: {})
    monkeypatch.setattr(catalog_runner, "_identity", lambda value: old_identity)
    monkeypatch.setattr(catalog_runner, "verify_prediction_manifest", lambda *args: old_manifest)
    folds = [{**_fold(), "id": fold} for fold in runner.DEVELOPMENT_FOLD_IDS]
    inputs = SimpleNamespace(
        project_root=tmp_path,
        location_grid=SimpleNamespace(area_km2=np.ones(2)),
        spatial_domain=None,
        contract={"outer_folds": folds},
    )
    monkeypatch.setattr(catalog_runner, "load_inputs", lambda *args: (inputs, None))
    monkeypatch.setattr(
        runner,
        "ReadOnlyComponentCache",
        lambda *args: SimpleNamespace(
            area=np.ones(2),
            audit={"synthetic": True},
        ),
    )
    monkeypatch.setattr(runner, "load_fault_surfaces", lambda *args: _surfaces())
    monkeypatch.setattr(runner, "_inner_samples", lambda *args: [])
    # Empty geometry validation falls back without needing a catalogue candidate fit.
    monkeypatch.setattr(runner, "build_inner_catalog_validation", lambda *args: ([], []))
    originals = {}
    for fold in runner.DEVELOPMENT_FOLD_IDS:
        issues, horizons = [], []
        for horizon in runner.HORIZONS:
            axis = expected_axis(ROOT, fold, horizon)
            issues.extend(axis)
            horizons.extend([horizon] * len(axis))
        logs = np.broadcast_to(np.log([0.25, 0.75]), (99, 9, 2)).copy()
        logs.setflags(write=False)
        originals[fold] = {
            "fold_id": np.asarray(fold),
            "model_ids": np.asarray(catalog_runner.MODEL_IDS),
            "issue_times_us": np.asarray(issues, dtype=np.int64),
            "horizons_days": np.asarray(horizons, dtype=np.int64),
            "log_cell_mass": logs,
        }
    old_reader = catalog_runner.load_fold_arrays
    monkeypatch.setattr(
        catalog_runner, "load_fold_arrays", lambda root, record: originals[record["fold_id"]]
    )
    for name in runner._NUMERICAL_ENV:
        monkeypatch.setenv(name, "1")
    output = tmp_path / "outputs" / "multitask_s2" / "synthetic"
    path = runner.run_prediction_phase(
        project_root=tmp_path,
        data_root=tmp_path,
        output_root=output,
        workers=2,
    )
    manifest = runner.verify_prediction_manifest(tmp_path, output)
    assert path == output / "prediction_manifest.json"
    assert manifest["issue_horizon_pairs"] == 396
    assert manifest["outer_targets_read"] is False
    for record in manifest["folds"]:
        values = old_reader(output, record)
        assert values["log_cell_mass"].shape == (99, 6, 2)
        assert values["model_ids"].tolist() == list(runner.MODEL_IDS)
        for index in (1, 2, 4, 5):
            np.testing.assert_array_equal(
                values["log_cell_mass"][:, index],
                originals[record["fold_id"]]["log_cell_mass"][:, 5],
            )
    new_hashes = {
        file: (runner._sha(file), file.stat().st_mtime_ns)
        for file in output.rglob("*")
        if file.is_file() and file.name != "run.lock"
    }
    monkeypatch.setattr(
        runner, "ReadOnlyComponentCache", lambda *args: pytest.fail("completed run")
    )
    assert (
        runner.run_prediction_phase(
            project_root=tmp_path,
            data_root=tmp_path,
            output_root=output,
            workers=1,
        )
        == path
    )
    for file, before in {**old_hashes, **new_hashes}.items():
        assert (runner._sha(file), file.stat().st_mtime_ns) == before
