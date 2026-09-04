"""Synthetic readonly authentication checks; no real inputs or outer scores."""

import json
from copy import deepcopy
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from seismoflux.multitask_s3 import null_parallel_inputs as inputs
from seismoflux.multitask_s3.null_runner import NullTask


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    project = tmp_path / "project"
    data = tmp_path / "data"
    prepared = project / "outputs/multitask_s3/prepared"
    reference = project / "outputs/multitask_s3/reference"
    output = project / "outputs/multitask_s3/null"
    for path in (project / "configs", data, prepared, reference, output):
        path.mkdir(parents=True, exist_ok=True)
    issue = datetime(2023, 7, 20, 16, tzinfo=UTC)
    truth = datetime(2026, 7, 9, 4, 25, 56, tzinfo=UTC)
    protocol = {
        "placebos": {
            "root_seed": 147,
            "time_replicates_per_fold": 200,
            "space_replicates_per_fold": 200,
        },
        "access": {
            "feature_store": "store.parquet",
            "feature_store_sha256": "frozen",
            "state_history_sha256": "frozen",
            "catalog": "catalog.parquet",
        },
    }
    spatial = {
        "local_coordinate_artifacts_not_committed": {
            "cell_mapping": "data/cells.parquet",
            "entity_mapping": "data/entities.parquet",
        },
        "local_artifact_sha256": {"cell_mapping": "frozen", "entity_mapping": "frozen"},
    }
    (project / "configs/multitask_s3_anomaly.yaml").write_text(
        yaml.safe_dump(protocol), encoding="utf-8"
    )
    (project / "configs/d1_retrospective_development.yaml").write_text(
        yaml.safe_dump(
            {
                "data": {
                    "spatial_strata": spatial,
                    "anomaly_features": {"state_history_path": "data/states.parquet"},
                }
            }
        ),
        encoding="utf-8",
    )
    source = {
        "status": "complete",
        "failures": {},
        "identity": {
            "protocol_sha256": "frozen",
            "catalog_identity": "catalog",
            "grid_id": "grid",
            "grid_cells": 2,
            "study_area_sha256": "area",
        },
        "issue_times_utc": [issue.isoformat()],
        "truth_cutoff_utc": truth.isoformat(),
        "completed": {issue.isoformat(): {"file": "issue.npz", "sha256": "frozen"}},
    }
    original = {
        "status": "predictions_complete",
        "identity": {
            "prepared_inputs": source["identity"],
            "implementation_sha256": {"multitask_s3/runner": "frozen"},
            "prepared_report_sha256": {issue.isoformat(): "frozen"},
        },
    }
    (prepared / "preparation.json").write_text(json.dumps(source), encoding="utf-8")
    (reference / "prediction_manifest.json").write_text(json.dumps(original), encoding="utf-8")
    names = (
        "multitask_s3/null_runner",
        "multitask_s3/null_features",
        "multitask_s3/null_inputs",
        "multitask_s3/null_space",
        "multitask_s3/null_state_inputs",
        "features/anomaly/trajectory",
        "features/anomaly/snapshot",
        "features/anomaly/state",
        "features/anomaly/spatial",
        "d1_replay/placebos",
    )
    task = NullTask("time", "A_DEV_2023_2024", 0, 7)
    manifest = {
        "identity": {
            **original["identity"],
            "reference_prediction_manifest_sha256": "frozen",
            "null_source_sha256": {
                key: "frozen"
                for key in ("feature_store", "state_history", "cell_mapping", "entity_mapping")
            },
            "null_implementation_sha256": {name: "frozen" for name in names},
            "seed_namespace": {
                "root": 147,
                "kinds": ["time", "space"],
                "folds": ["A_DEV_2023_2024", "A_DEV_2024_2025"],
                "replicate_indices": [0, 199],
                "words": [
                    "root",
                    "kind_index",
                    "fold_index",
                    "replicate",
                    "horizon_if_time_else_0",
                ],
            },
        },
        "source_reconstruction_check": {"time_identity": "passed", "space_identity": "passed"},
        "completed": {task.key: {"file": f"{task.key}.npz", "sha256": "frozen"}},
        "failures": {},
        "issue_times_utc": source["issue_times_utc"],
        "truth_cutoff_utc": source["truth_cutoff_utc"],
    }
    hashes = {}
    monkeypatch.setattr(inputs, "sha256", lambda path: hashes.get(path.resolve(), "frozen"))
    monkeypatch.setattr(inputs, "verify_authoritative_catalog_identity", lambda path: "catalog")
    areas = np.array([100.0, 150.0])
    domain = SimpleNamespace(
        operational_grid=SimpleNamespace(
            grid_id="grid", cell_ids=("one", "two"), query_xy_km=np.array([[1.0, 2.0], [3.0, 4.0]])
        ),
        locator=SimpleNamespace(locate_lonlat=lambda lon, lat: 1 if lon > 0 else None),
    )
    monkeypatch.setattr(
        inputs,
        "load_verified_spatial_inputs",
        lambda root: (domain, SimpleNamespace(cell_count=2, area_km2=areas), "area"),
    )
    calendar_calls = []

    def calendar(issues, **kwargs):
        calendar_calls.append((issues, kwargs))
        return SimpleNamespace(report_issues=issues)

    monkeypatch.setattr(inputs, "build_fold_calendar", calendar)
    cache = {
        "features": np.arange(40, dtype=float).reshape(2, 20),
        "kernel_25": np.array([-1.0, -1.0]),
        "metadata": {},
    }
    monkeypatch.setattr(inputs, "read_issue_cache", lambda *args, **kwargs: cache)
    bounded_calls = {}

    def bounded(name, result):
        def call(*args, **kwargs):
            bounded_calls[name] = (args, kwargs)
            return result

        return call

    monkeypatch.setattr(
        inputs, "load_radius_bases", bounded("radius", {issue: np.array([[1.0, 2.0], [3.0, 4.0]])})
    )
    snapshots = {issue: "snapshot"}
    monkeypatch.setattr(inputs, "load_issue_snapshots", bounded("snapshots", snapshots))
    monkeypatch.setattr(inputs, "load_construction_strata", bounded("strata", {"entity": "zone"}))
    monkeypatch.setattr(inputs, "load_all_zone_ids", bounded("zones", ("zone",)))
    frame = {"longitude": [-1.0, 1.0], "latitude": [0.0, 0.0]}
    monkeypatch.setattr(inputs, "load_development_catalog", bounded("catalog", frame))
    anchors = np.array([0, 1])
    monkeypatch.setattr(inputs, "prepare_anchor_ids", bounded("anchors", anchors))
    return SimpleNamespace(
        args=dict(
            project_root=project,
            data_root=data,
            prepared_dir=prepared,
            reference_prediction_dir=reference,
            output_dir=output,
            manifest=manifest,
        ),
        issue=issue,
        truth=truth,
        task=task,
        hashes=hashes,
        cache=cache,
        areas=areas,
        calls=bounded_calls,
        calendar_calls=calendar_calls,
        frame=frame,
        anchors=anchors,
        source=source,
        original=original,
    )


def test_load_is_readonly_and_preserves_original_inputs_and_boundaries(frozen):
    manifest_before = deepcopy(frozen.args["manifest"])
    files_before = {
        path: path.read_bytes() for path in frozen.args["project_root"].rglob("*") if path.is_file()
    }
    context = inputs.load_frozen_context(**frozen.args)
    assert set(context) == {
        "identity",
        "issues",
        "truth",
        "calendars",
        "caches",
        "bases",
        "features",
        "snapshots",
        "strata",
        "zones",
        "query_xy_m",
        "frame",
        "cells",
        "anchors",
        "areas_km2",
    }
    assert frozen.args["manifest"] == manifest_before
    assert files_before == {
        path: path.read_bytes() for path in frozen.args["project_root"].rglob("*") if path.is_file()
    }
    assert context["identity"] == manifest_before["identity"]
    assert context["frame"] is frozen.frame and context["anchors"] is frozen.anchors
    assert context["areas_km2"] is frozen.areas
    assert context["caches"][frozen.issue] is frozen.cache
    assert not frozen.cache["features"].flags.writeable
    assert not frozen.cache["kernel_25"].flags.writeable
    np.testing.assert_array_equal(context["cells"], [-1, 1])
    np.testing.assert_array_equal(context["query_xy_m"], [[1000.0, 2000.0], [3000.0, 4000.0]])
    np.testing.assert_array_equal(context["features"][0], frozen.cache["features"])
    assert len(frozen.calendar_calls) == 10
    for _, kwargs in frozen.calendar_calls:
        assert kwargs["truth_cutoff"] == frozen.truth
    for key in ("radius", "snapshots"):
        assert frozen.calls[key][1]["issue_times_utc"] == (frozen.issue,)
        assert frozen.calls[key][1]["report_end_exclusive"] == inputs.REPORT_END
    assert frozen.calls["catalog"][1] == {"truth_cutoff": frozen.truth}
    assert frozen.calls["anchors"][0] == (frozen.frame,)


@pytest.mark.parametrize("kind", ["time_identity", "space_identity"])
def test_requires_both_original_source_checks(frozen, kind):
    frozen.args["manifest"]["source_reconstruction_check"][kind] = "pending"
    with pytest.raises(ValueError, match="passed time and space"):
        inputs.load_frozen_context(**frozen.args)


@pytest.mark.parametrize(
    "target,relative,message",
    [
        ("project_root", "configs/multitask_s3_anomaly.yaml", "protocol changed"),
        ("project_root", "src/seismoflux/multitask_s3/runner.py", "real-feature model"),
        ("project_root", "src/seismoflux/multitask_s3/null_features.py", "same frozen"),
        ("reference_prediction_dir", "prediction_manifest.json", "same frozen"),
        ("data_root", "store.parquet", "immutable null source"),
        ("data_root", "states.parquet", "immutable null source"),
        ("data_root", "cells.parquet", "immutable null source"),
        ("prepared_dir", "issue.npz", "prepared report changed"),
    ],
)
def test_rejects_changed_frozen_hash(frozen, target, relative, message):
    frozen.hashes[(frozen.args[target] / relative).resolve()] = "changed"
    with pytest.raises(ValueError, match=message):
        inputs.load_frozen_context(**frozen.args)


def test_rejects_changed_completed_block(frozen):
    frozen.hashes[(frozen.args["output_dir"] / f"{frozen.task.key}.npz").resolve()] = "changed"
    with pytest.raises(ValueError, match="completed null block changed"):
        inputs.load_frozen_context(**frozen.args)


@pytest.mark.parametrize("field", ["root", "replicate_indices", "words"])
def test_rejects_changed_seed_namespace(frozen, field):
    frozen.args["manifest"]["identity"]["seed_namespace"][field] = "changed"
    with pytest.raises(ValueError, match="same frozen"):
        inputs.load_frozen_context(**frozen.args)


def test_rejects_changed_report_axis(frozen):
    frozen.args["manifest"]["issue_times_utc"] = []
    with pytest.raises(ValueError, match="report axis or truth cutoff"):
        inputs.load_frozen_context(**frozen.args)


def test_rejects_nonprefix_calendar(frozen, monkeypatch):
    monkeypatch.setattr(
        inputs,
        "build_fold_calendar",
        lambda *args, **kwargs: SimpleNamespace(report_issues=(frozen.truth,)),
    )
    with pytest.raises(ValueError, match="complete registered prefix"):
        inputs.load_frozen_context(**frozen.args)


def test_rejects_unregistered_completed_key(frozen):
    frozen.args["manifest"]["completed"]["not_registered"] = {}
    with pytest.raises(ValueError, match="unregistered"):
        inputs.load_frozen_context(**frozen.args)


def test_rejects_output_outside_current_s3(frozen, tmp_path):
    frozen.args["output_dir"] = tmp_path / "other"
    with pytest.raises(ValueError, match="distinct local S3"):
        inputs.load_frozen_context(**frozen.args)


def test_local_file_cannot_escape(tmp_path):
    with pytest.raises(ValueError, match="escaped"):
        inputs._local_file(tmp_path, "../outside.npz")
