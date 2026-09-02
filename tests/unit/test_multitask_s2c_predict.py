"""Synthetic S2-C prediction/selection checks; no real outer targets or new scores."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from seismoflux.multitask_s1 import c2b_predict as catalog_runner
from seismoflux.multitask_s2 import strain_predict as runner
from seismoflux.multitask_s2.strain import StrainSurfaces

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def single_numeric_thread(monkeypatch):
    for name in runner._NUMERICAL_ENV:
        monkeypatch.setenv(name, "1")


def _logs(values):
    with np.errstate(divide="ignore"):
        return np.log(np.asarray(values, dtype=np.float64))


def _surfaces(values=(0.75, 0.25, 0.0)):
    return StrainSurfaces(
        layers={layer: _logs(values) for layer in runner.LAYER_IDS},
        audit={"synthetic": True, "retains_zero_mass": True},
    )


def _validation(block, counts, probabilities=(0.25, 0.5, 0.25)):
    issue = datetime(1993 if block == "I2" else 1998, 1, 1, tzinfo=UTC)
    return runner.GeometryValidationSample(
        block,
        issue,
        issue + timedelta(days=30),
        _logs(probabilities),
        np.asarray(counts, dtype=np.float64),
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


def _sample(block, issue, available=None):
    time = datetime.fromisoformat(issue)
    end = time + timedelta(days=30)
    components = {name: _logs([0.25, 0.5, 0.25]) for name in catalog_runner.COMPONENT_IDS}
    return catalog_runner.InnerSample(
        _fold()["id"],
        block,
        30,
        time,
        end,
        components,
        np.array([0], dtype=np.int64),
        np.array(
            [catalog_runner._epoch_us(end if available is None else available)], dtype=np.int64
        ),
    )


def test_frozen_protocol_four_models_no_scales_and_396_unchanged_pairs():
    protocol = runner.load_protocol(ROOT)
    assert protocol["status"] == "protocol_frozen_before_S2C_predictions"
    assert (
        runner.PROTOCOL_SHA256 == "e9e0800279b9db64722c35763b0dae4d228d90173ddceb660e614ffbcc78876e"
    )
    assert tuple(protocol["models"]) == runner.MODEL_IDS
    assert len(runner.MODEL_IDS) == 4
    assert runner.strain_candidates("static_only", protocol) == [{"alpha": 1.0}]
    assert runner.strain_candidates("catalog_mixture", protocol) == [
        {"alpha": value} for value in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    assert [
        sum(
            len(runner._expected_horizon_axis(ROOT, fold, h))
            for fold in runner.DEVELOPMENT_FOLD_IDS
        )
        for h in runner.HORIZONS
    ] == [176, 116, 56, 32, 16]
    with pytest.raises(ValueError, match="family"):
        runner.strain_candidates("new_family", protocol)


def test_inner_catalog_uses_only_earlier_blocks_and_mature_visible_labels():
    cutoff = datetime(1989, 12, 2, tzinfo=UTC)
    outer_cutoff = datetime(1999, 12, 2, tzinfo=UTC)
    samples = [
        _sample("I1", "1988-01-01T00:00:00+00:00"),
        _sample("I1", "1988-02-01T00:00:00+00:00", cutoff + timedelta(microseconds=1)),
        _sample("I1", "1989-11-03T00:00:00+00:00"),
        _sample("I2", "1993-01-01T00:00:00+00:00"),
        _sample("I3", "1998-01-01T00:00:00+00:00", outer_cutoff + timedelta(microseconds=1)),
    ]
    validation, branches = runner.build_inner_catalog_validation(
        samples,
        _fold(),
        np.ones(3),
        catalog_runner.load_protocol(ROOT),
    )
    assert branches[0]["training"]["issue_count"] == 2
    assert branches[0]["training"]["target_count"] == 1
    assert branches[0]["catalog_multiscale_selection"]["selected"] == "K75"
    assert branches[0]["I2_explicit_K75_fallback"] is True
    assert branches[1]["train_blocks"] == ["I1", "I2"]
    assert branches[1]["training"]["target_count"] == 4
    assert branches[1]["validation"]["issue_count"] == 1
    assert branches[1]["validation"]["target_count"] == 0
    assert [sample.target_counts.sum() for sample in validation] == [1.0, 0.0]


def test_no_events_exact_catalog_and_fixed_pure_layer_with_explicit_no_events():
    protocol, surfaces = runner.load_protocol(ROOT), _surfaces()
    base = _logs([0.25, 0.5, 0.25])
    choices = runner.select_strain_parameters([], surfaces, np.ones(3), protocol)
    logs = runner._prediction_models(base, surfaces, choices, protocol)
    assert logs.shape == (4, 3)
    for index, model in enumerate(runner.MODEL_IDS):
        pure = model.endswith("_ONLY")
        assert choices[model]["selected"] == {"alpha": 1.0 if pure else 0.0}
        assert len(choices[model]["candidates"]) == (1 if pure else 5)
        assert all(
            row["mean"] is None and row["mean_status"] == "no_events"
            for row in choices[model]["candidates"]
        )
        expected = surfaces.layers[protocol["models"][model]["layer"]] if pure else base
        assert logs[index].tobytes() == expected.tobytes()
        assert "scale" not in json.dumps(choices[model])
    assert np.isneginf(logs[0, 2])
    json.dumps(choices, allow_nan=False)


def test_zero_count_cells_never_multiply_negative_infinity():
    choices = runner.select_strain_parameters(
        [_validation("I2", [1, 0, 0]), _validation("I3", [0, 0, 0])],
        _surfaces(),
        np.ones(3),
        runner.load_protocol(ROOT),
    )
    for choice in choices.values():
        assert choice["selected"] == {"alpha": 1.0}
        assert choice["validation_target_counts"] == [1, 0]
        assert choice["candidates"][-1]["block_score_statuses"] == ["finite", "no_events"]
        assert choice["candidates"][-1]["mean"] == pytest.approx(np.log(0.75))
    json.dumps(choices, allow_nan=False)


def test_observed_zero_retained_as_negative_infinity_not_discarded_block():
    choices = runner.select_strain_parameters(
        [_validation("I2", [1, 0, 0]), _validation("I3", [0, 0, 1])],
        _surfaces(),
        np.ones(3),
        runner.load_protocol(ROOT),
    )
    pure = choices["S2C_STRAIN_ONLY"]
    row = pure["candidates"][0]
    assert row["block_score_statuses"] == ["finite", "negative_infinity_from_zero_mass"]
    assert row["mean"] is None
    assert row["mean_status"] == "negative_infinity_from_zero_mass"
    assert pure["validation_target_counts"] == [1, 1]
    assert pure["status"] == "fixed_static_field_no_fitted_parameters"
    assert choices["S2C_STRAIN_CATALOG_MIX"]["selected"]["alpha"] < 1.0
    json.dumps(choices, allow_nan=False)


def test_all_candidates_negative_infinity_has_explicit_first_alpha_not_nan_tie():
    choices = runner.select_strain_parameters(
        [_validation("I2", [0, 0, 1], (0.25, 0.75, 0.0))],
        _surfaces(),
        np.ones(3),
        runner.load_protocol(ROOT),
    )
    for model in ("S2C_UNIT_CATALOG_MIX", "S2C_STRAIN_CATALOG_MIX"):
        assert choices[model]["selected"] == {"alpha": 0.0}
        assert choices[model]["status"] == "all_candidates_negative_infinity_first_registered_alpha"
        assert all(
            row["mean_status"] == "negative_infinity_from_zero_mass"
            for row in choices[model]["candidates"]
        )
    json.dumps(choices, allow_nan=False)


def test_selection_equal_means_of_blocks_not_event_pooled_and_smaller_alpha_ties():
    protocol = runner.load_protocol(ROOT)
    choices = runner.select_strain_parameters(
        [_validation("I2", [0, 100], (0.9, 0.1)), _validation("I3", [1, 0], (0.9, 0.1))],
        _surfaces((0.1, 0.9)),
        np.ones(2),
        protocol,
    )
    assert choices["S2C_STRAIN_CATALOG_MIX"]["selected"] == {"alpha": 0.5}
    same = runner.select_strain_parameters(
        [_validation("I2", [0, 1], (0.1, 0.9))],
        _surfaces((0.1, 0.9)),
        np.ones(2),
        protocol,
    )
    assert same["S2C_STRAIN_CATALOG_MIX"]["selected"] == {"alpha": 0.0}


def test_invalid_counts_and_extra_blocks_rejected():
    protocol = runner.load_protocol(ROOT)
    for sample in (
        _validation("I1", [1, 0, 0]),
        _validation("I2", [-1, 0, 0]),
        _validation("I2", [np.nan, 0, 0]),
    ):
        with pytest.raises(ValueError):
            runner.select_strain_parameters([sample], _surfaces(), np.ones(3), protocol)


def test_surface_checkpoint_preserves_true_zero_and_unsealed_payload(tmp_path, monkeypatch):
    protocol = runner.load_protocol(ROOT)
    protocol["inputs"]["grid_cells"] = 3
    identity, calls = {"synthetic_identity": True}, []

    def compute(**kwargs):
        calls.append(kwargs)
        return _surfaces()

    monkeypatch.setattr(runner, "load_strain_surfaces", compute)
    _, manifest = runner._load_or_build_surfaces(tmp_path, tmp_path, protocol, None, identity)
    payload = manifest.parent / "surfaces.npz"
    before = (runner._sha(payload), payload.stat().st_mtime_ns)
    with np.load(payload, allow_pickle=False) as saved:
        assert saved["log_cell_mass"].shape == (2, 3)
        assert "scales_km" not in saved.files
        assert np.isneginf(saved["log_cell_mass"][:, 2]).all()
    saved, same_manifest = runner._load_or_build_surfaces(
        tmp_path, tmp_path, protocol, None, identity
    )
    assert same_manifest == manifest and len(calls) == 1
    assert all(not mass.flags.writeable for mass in saved.layers.values())
    manifest.unlink()
    (manifest.parent / "audit.json").unlink()
    runner._load_or_build_surfaces(tmp_path, tmp_path, protocol, None, identity)
    assert len(calls) == 1
    assert (runner._sha(payload), payload.stat().st_mtime_ns) == before
    with pytest.raises(ValueError, match="identity"):
        runner._load_or_build_surfaces(tmp_path, tmp_path, protocol, None, {"changed": True})


@pytest.mark.parametrize(
    "suffix",
    [
        "multitask_s1/old",
        "multitask_s2/s2a_fault_geometry_v1",
        "multitask_s2/s2b_slip_rate_v1",
        "multitask_s2/s2b_slip_rate_v1/nested",
        "multitask_s2",
        "multitask_s2/unknown",
    ],
)
def test_old_runs_rejected_before_creating_lock(tmp_path, suffix):
    root = tmp_path / "outputs" / suffix
    with pytest.raises(ValueError, match="old S1, S2A and S2B"):
        runner._output_root(tmp_path, {}, root)
    assert not root.exists()


def test_log_arrays_allow_negative_infinity_but_reject_invalid_shape_dtype_or_nonfinite():
    logs = np.broadcast_to(_logs([0.75, 0.25, 0.0]), (2, 4, 3)).copy()
    runner._validate_prediction_log_mass(logs, (2, 4, 3))
    for invalid in (logs.astype(np.float32), logs[:, :3], np.full((2, 4, 3), -np.inf)):
        with pytest.raises(ValueError):
            runner._validate_prediction_log_mass(invalid, (2, 4, 3))
    for value in (np.nan, np.inf):
        invalid = logs.copy()
        invalid[0, 0, 0] = value
        with pytest.raises(ValueError):
            runner._validate_prediction_log_mass(invalid, (2, 4, 3))


def test_synthetic_396_pairs_and_partial_resume_preserve_old_and_complete_payloads(
    tmp_path, monkeypatch
):
    protocol, catalog_protocol = runner.load_protocol(ROOT), catalog_runner.load_protocol(ROOT)
    protocol["inputs"]["grid_cells"] = 3
    protocol["inputs"]["catalog_run"] = "outputs/multitask_s1/synthetic"
    old_root = tmp_path / protocol["inputs"]["catalog_run"]
    old_identity = {"synthetic_old_identity": True}
    old_manifest = {"folds": [{"fold_id": fold} for fold in runner.DEVELOPMENT_FOLD_IDS]}
    runner._write_json(old_root / "prediction_manifest.json", old_manifest)
    runner._write_json(old_root / "score_phase/score_manifest.json", {"synthetic": True})
    runner._write_json(old_root / "run_identity.json", old_identity)
    for path, field in (
        ("prediction_manifest.json", "catalog_prediction_manifest_sha256"),
        ("score_phase/score_manifest.json", "catalog_score_manifest_sha256"),
    ):
        protocol["inputs"][field] = runner._sha(old_root / path)
    old_hashes = {
        path: (runner._sha(path), path.stat().st_mtime_ns) for path in old_root.rglob("*.json")
    }
    real_axis = runner._expected_horizon_axis
    monkeypatch.setattr(runner, "_expected_horizon_axis", lambda p, f, h: real_axis(ROOT, f, h))
    monkeypatch.setattr(runner, "load_protocol", lambda p: protocol)
    monkeypatch.setattr(runner, "_identity", lambda p: {"synthetic_s2c_identity": True})
    monkeypatch.setattr(catalog_runner, "load_protocol", lambda p: catalog_protocol)
    monkeypatch.setattr(catalog_runner, "_identity", lambda p: old_identity)
    monkeypatch.setattr(catalog_runner, "verify_prediction_manifest", lambda *a: old_manifest)
    inputs = SimpleNamespace(
        project_root=tmp_path,
        location_grid=SimpleNamespace(area_km2=np.ones(3)),
        spatial_domain=None,
        contract={"outer_folds": [{**_fold(), "id": fold} for fold in runner.DEVELOPMENT_FOLD_IDS]},
    )
    monkeypatch.setattr(catalog_runner, "load_inputs", lambda *a: (inputs, None))
    monkeypatch.setattr(
        runner,
        "ReadOnlyComponentCache",
        lambda *a: SimpleNamespace(
            area=np.ones(3),
            audit={"synthetic_readonly_cache": True},
        ),
    )
    static_calls, inner_calls = [], []

    def static(**kwargs):
        static_calls.append(True)
        return _surfaces()

    def inner(*args):
        inner_calls.append(True)
        return []

    monkeypatch.setattr(runner, "load_strain_surfaces", static)
    monkeypatch.setattr(runner, "_inner_samples", inner)
    originals = {}
    for fold in runner.DEVELOPMENT_FOLD_IDS:
        issues, horizons = [], []
        for horizon in runner.HORIZONS:
            axis = real_axis(ROOT, fold, horizon)
            issues.extend(axis)
            horizons.extend([horizon] * len(axis))
        logs = np.broadcast_to(_logs([0.25, 0.5, 0.25]), (99, 9, 3)).copy()
        logs.setflags(write=False)
        originals[fold] = {
            "fold_id": np.asarray(fold),
            "model_ids": np.asarray(catalog_runner.MODEL_IDS),
            "issue_times_us": np.array(issues, dtype=np.int64),
            "horizons_days": np.array(horizons, dtype=np.int64),
            "log_cell_mass": logs,
        }
    old_reader = catalog_runner.load_fold_arrays
    monkeypatch.setattr(
        catalog_runner, "load_fold_arrays", lambda r, rec: originals[rec["fold_id"]]
    )
    output = tmp_path / "outputs/multitask_s2/s2c_synthetic"
    manifest_path = runner.run_prediction_phase(
        project_root=tmp_path,
        data_root=tmp_path,
        output_root=output,
        workers=2,
    )
    manifest = runner.verify_prediction_manifest(tmp_path, output)
    assert manifest["issue_horizon_pairs"] == 396 and manifest["outer_targets_read"] is False
    assert len(inner_calls) == 20 and len(static_calls) == 1
    for record in manifest["folds"]:
        arrays = old_reader(output, record)
        assert arrays["log_cell_mass"].shape == (99, 4, 3)
        for index in (1, 3):
            np.testing.assert_array_equal(
                arrays["log_cell_mass"][:, index],
                originals[record["fold_id"]]["log_cell_mass"][:, 5],
            )
        assert np.isneginf(arrays["log_cell_mass"][:, 0, 2]).all()
        assert np.isneginf(arrays["log_cell_mass"][:, 2, 2]).all()
    payloads = {
        path: (runner._sha(path), path.stat().st_mtime_ns) for path in output.rglob("*.npz")
    }
    last = manifest["folds"][-1]
    fold_path = output / last["path"]
    last_horizon = runner._read_json(fold_path)["horizons"][-1]
    (output / last_horizon["path"]).unlink()
    fold_path.unlink()
    manifest_path.unlink()
    runner.run_prediction_phase(project_root=tmp_path, data_root=tmp_path, output_root=output)
    assert len(inner_calls) == 21 and len(static_calls) == 1
    runner.run_prediction_phase(project_root=tmp_path, data_root=tmp_path, output_root=output)
    assert len(inner_calls) == 21 and len(static_calls) == 1
    for path, before in {**old_hashes, **payloads}.items():
        assert (runner._sha(path), path.stat().st_mtime_ns) == before
