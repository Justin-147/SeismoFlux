"""Tiny S2-B prediction fixtures: no real catalogue, fault attributes or outer scores."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from seismoflux.multitask_s1 import c2b_predict as catalog_runner
from seismoflux.multitask_s2 import slip_predict as runner
from seismoflux.multitask_s2.slip_rate import SlipRateSurfaces

ROOT = Path(__file__).resolve().parents[2]
LAYER_IDS = ("COMMON_UNIT", "COMMON_GEO", "COMMON_GD", "NATIVE_UNIT", "NATIVE_GD")


@pytest.fixture(autouse=True)
def single_numeric_thread(monkeypatch):
    for name in runner._NUMERICAL_ENV:
        monkeypatch.setenv(name, "1")


def _surfaces():
    return SlipRateSurfaces(
        layers={
            layer: {scale: np.log(np.array([0.75, 0.25])) for scale in (25.0, 75.0, 150.0)}
            for layer in LAYER_IDS
        },
        audit={"synthetic": True, "diagnostic_runs": 1},
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
    components = {name: np.log(np.array([0.25, 0.75])) for name in catalog_runner.COMPONENT_IDS}
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


def test_frozen_protocol_ten_models_and_396_unchanged_pairs():
    protocol = runner.load_protocol(ROOT)
    assert protocol["status"] == "protocol_frozen_before_S2B_predictions"
    assert (
        runner.PROTOCOL_SHA256 == "a57218c9fbedc21ae28a62de4980fc26b17e990aa32f3613b9ae95fc1db19d29"
    )
    assert tuple(protocol["models"]) == runner.MODEL_IDS
    assert len(runner.MODEL_IDS) == 10
    assert [
        sum(
            len(runner._expected_horizon_axis(ROOT, fold, h))
            for fold in runner.DEVELOPMENT_FOLD_IDS
        )
        for h in runner.HORIZONS
    ] == [176, 116, 56, 32, 16]


def test_reused_inner_oof_has_separate_earlier_end_and_visibility_cutoffs():
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
        np.ones(2),
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


def test_ten_model_empty_selection_keeps_exact_old_catalog_for_all_five_mixtures():
    protocol, surfaces = runner.load_protocol(ROOT), _surfaces()
    base = np.log(np.array([0.25, 0.75]))
    choices = runner.select_slip_parameters([], surfaces, np.ones(2), protocol)
    logs = runner._prediction_models(base, surfaces, choices, protocol)
    assert logs.shape == (10, 2)
    for index, model in enumerate(runner.MODEL_IDS):
        pure = model.endswith("_ONLY")
        assert choices[model]["selected"] == {
            "alpha": 1.0 if pure else 0.0,
            "scale_km": 75.0 if pure else None,
        }
        assert len(choices[model]["candidates"]) == (3 if pure else 13)
        assert all(row["mean"] is None for row in choices[model]["candidates"])
        np.testing.assert_array_equal(
            logs[index], surfaces.layers[protocol["models"][model]["layer"]][75.0] if pure else base
        )
        if not pure:
            assert logs[index].tobytes() == base.tobytes()


def test_one_nonempty_block_selects_and_ties_preserve_registered_scale():
    protocol, surfaces = runner.load_protocol(ROOT), _surfaces()
    issue = datetime(1993, 1, 1, tzinfo=UTC)
    sample = runner.GeometryValidationSample(
        "I2",
        issue,
        issue + timedelta(days=30),
        np.log(np.array([0.25, 0.75])),
        np.array([1.0, 0.0]),
    )
    selected = runner.select_slip_parameters([sample], surfaces, np.ones(2), protocol)
    for value in selected.values():
        assert value["selected"] == {"alpha": 1.0, "scale_km": 75.0}
        assert value["validation_target_counts"] == [1, 0]
        assert value["candidates"][0]["block_scores"][1] is None


def test_static_layers_and_one_diagnostic_resume_even_after_unsealed_payload(tmp_path, monkeypatch):
    protocol = runner.load_protocol(ROOT)
    protocol["inputs"]["grid_cells"] = 2
    identity, calls = {"synthetic_identity": True}, []

    def compute(*args):
        calls.append(True)
        return _surfaces()

    monkeypatch.setattr(runner, "load_slip_rate_surfaces", compute)
    _, manifest = runner._load_or_build_surfaces(tmp_path, tmp_path, protocol, None, identity)
    payload = manifest.parent / "surfaces.npz"
    before = (runner._sha(payload), payload.stat().st_mtime_ns)
    saved, same_manifest = runner._load_or_build_surfaces(
        tmp_path, tmp_path, protocol, None, identity
    )
    assert same_manifest == manifest and len(calls) == 1
    assert all(
        not mass.flags.writeable for layer in saved.layers.values() for mass in layer.values()
    )
    # Simulate interruption after the complete payload, before its derived records.
    manifest.unlink()
    (manifest.parent / "audit.json").unlink()
    runner._load_or_build_surfaces(tmp_path, tmp_path, protocol, None, identity)
    assert len(calls) == 1
    assert (runner._sha(payload), payload.stat().st_mtime_ns) == before
    with pytest.raises(ValueError, match="identity"):
        runner._load_or_build_surfaces(tmp_path, tmp_path, protocol, None, {"changed": True})
    assert len(calls) == 1


@pytest.mark.parametrize(
    "suffix",
    [
        "multitask_s1/old",
        "multitask_s2/s2a_fault_geometry_v1",
        "multitask_s2/s2a_fault_geometry_v1_attempt2",
        "multitask_s2/s2a_fault_geometry_v1_attempt2/nested",
    ],
)
def test_old_s1_and_every_s2a_run_are_rejected_before_creating_lock(tmp_path, suffix):
    root = tmp_path / "outputs" / suffix
    with pytest.raises(ValueError, match="old S1 and S2A"):
        runner._output_root(tmp_path, {}, root)
    assert not root.exists()


def test_synthetic_396_pairs_partial_fold_and_completed_resume_do_not_recompute_old_or_static(
    tmp_path,
    monkeypatch,
):
    protocol, catalog_protocol = runner.load_protocol(ROOT), catalog_runner.load_protocol(ROOT)
    protocol["inputs"]["grid_cells"] = 2
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
    monkeypatch.setattr(runner, "_identity", lambda p: {"synthetic_s2b_identity": True})
    monkeypatch.setattr(catalog_runner, "load_protocol", lambda p: catalog_protocol)
    monkeypatch.setattr(catalog_runner, "_identity", lambda p: old_identity)
    monkeypatch.setattr(catalog_runner, "verify_prediction_manifest", lambda *a: old_manifest)
    inputs = SimpleNamespace(
        project_root=tmp_path,
        location_grid=SimpleNamespace(area_km2=np.ones(2)),
        spatial_domain=None,
        contract={"outer_folds": [{**_fold(), "id": fold} for fold in runner.DEVELOPMENT_FOLD_IDS]},
    )
    monkeypatch.setattr(catalog_runner, "load_inputs", lambda *a: (inputs, None))
    monkeypatch.setattr(
        runner,
        "ReadOnlyComponentCache",
        lambda *a: SimpleNamespace(
            area=np.ones(2),
            audit={"synthetic_readonly_cache": True},
        ),
    )
    static_calls, inner_calls = [], []

    def static(*args):
        static_calls.append(True)
        return _surfaces()

    def inner(*args):
        inner_calls.append(True)
        return []

    monkeypatch.setattr(runner, "load_slip_rate_surfaces", static)
    monkeypatch.setattr(runner, "_inner_samples", inner)
    originals = {}
    for fold in runner.DEVELOPMENT_FOLD_IDS:
        issues, horizons = [], []
        for horizon in runner.HORIZONS:
            axis = real_axis(ROOT, fold, horizon)
            issues.extend(axis)
            horizons.extend([horizon] * len(axis))
        logs = np.broadcast_to(np.log([0.25, 0.75]), (99, 9, 2)).copy()
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
    output = tmp_path / "outputs/multitask_s2/s2b_synthetic"
    manifest_path = runner.run_prediction_phase(
        project_root=tmp_path, data_root=tmp_path, output_root=output, workers=2
    )
    manifest = runner.verify_prediction_manifest(tmp_path, output)
    assert manifest["issue_horizon_pairs"] == 396
    assert len(inner_calls) == 20 and len(static_calls) == 1
    for record in manifest["folds"]:
        arrays = old_reader(output, record)
        assert arrays["log_cell_mass"].shape == (99, 10, 2)
        for index in (1, 3, 5, 7, 9):
            np.testing.assert_array_equal(
                arrays["log_cell_mass"][:, index],
                originals[record["fold_id"]]["log_cell_mass"][:, 5],
            )
    payloads = {p: (runner._sha(p), p.stat().st_mtime_ns) for p in output.rglob("*.npz")}
    # Restore from 3 completed folds and 4 completed horizons in the last fold.
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
