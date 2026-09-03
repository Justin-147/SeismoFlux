"""Thin case assembly tests using synthetic metadata and no new score execution."""

import json
from datetime import UTC, datetime

import pandas as pd
import pytest
import yaml

from seismoflux.multitask_s3 import case_runner
from seismoflux.multitask_s3.preparation import sha256, write_json


@pytest.fixture
def case_inputs(tmp_path, monkeypatch):
    project = tmp_path / "project"
    config = project / "configs/multitask_s3_anomaly.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        yaml.safe_dump({"access": {"catalog": "synthetic.parquet"}}), encoding="utf-8"
    )
    root = project / "outputs/multitask_s3"
    prediction, scores, output = root / "prediction", root / "scores", root / "scores/cases"
    prediction.mkdir(parents=True)
    scores.mkdir()
    cutoff = datetime(2025, 6, 30, 16, tzinfo=UTC).isoformat()
    identity = {
        "prepared_inputs": {"protocol_sha256": sha256(config), "catalog_identity": "catalog"}
    }
    manifest = {"status": "predictions_complete", "identity": identity, "truth_cutoff_utc": cutoff}
    write_json(prediction / "prediction_manifest.json", manifest)
    prediction_hash = sha256(prediction / "prediction_manifest.json")
    write_json(
        scores / "score_progress.json",
        {"status": "complete", "prediction_manifest_sha256": prediction_hash},
    )
    write_json(
        scores / "science_scores.json",
        {
            "prediction_identity": identity,
            "prediction_manifest_sha256": prediction_hash,
            "truth_cutoff_utc": cutoff,
        },
    )
    write_json(
        scores / "event_diagnostics_local.json",
        {"local_only": True, "records": [{"target_event_ids": ["e1"]}]},
    )
    calls = []

    def read_catalog(path, *, truth_cutoff):
        calls.append((path.name, truth_cutoff))
        return pd.DataFrame(
            [
                {
                    "event_id": event_id,
                    "origin_time_utc": pd.Timestamp("2023-08-01", tz="UTC"),
                    "available_at": pd.Timestamp("2023-08-02", tz="UTC"),
                    "magnitude": 5.3,
                    "longitude": 101.0,
                    "latitude": 30.0,
                }
                for event_id in ("e1", "unused")
            ]
        )

    def build(records, *, catalog_metadata):
        assert records == [{"target_event_ids": ["e1"]}]
        assert set(catalog_metadata) == {"e1"}
        return {"local_only": True, "rows": [], "events": catalog_metadata, "summary": {}}

    monkeypatch.setattr(case_runner, "load_development_catalog", read_catalog)
    monkeypatch.setattr(case_runner, "verify_authoritative_catalog_identity", lambda _: "catalog")
    monkeypatch.setattr(case_runner, "build_case_ledger", build)
    return dict(
        project_root=project,
        data_root=tmp_path,
        prediction_dir=prediction,
        score_dir=scores,
        output_dir=output,
    ), calls


def test_case_assembly_uses_same_completed_evidence_and_only_needed_metadata(case_inputs):
    kwargs, calls = case_inputs
    ledger = case_runner.assemble_cases(**kwargs)
    assert len(calls) == 1
    assert calls[0][1] == datetime(2025, 6, 30, 16, tzinfo=UTC)
    assert ledger["provenance"]["model_refitted"] is False
    assert ledger["provenance"]["hits_recomputed"] is False
    assert ledger["provenance"]["new_evaluation_role_accessed"] is False
    assert (kwargs["output_dir"] / "case_ledger_local.json").exists()
    with pytest.raises(FileExistsError):
        case_runner.assemble_cases(**kwargs)
    assert len(calls) == 1


def test_case_assembly_does_not_read_catalog_when_prediction_identity_changed(case_inputs):
    kwargs, calls = case_inputs
    path = kwargs["prediction_dir"] / "prediction_manifest.json"
    value = json.loads(path.read_text())
    value["status"] = "incomplete"
    write_json(path, value)
    with pytest.raises(ValueError, match="same completed"):
        case_runner.assemble_cases(**kwargs)
    assert not calls


def test_case_assembly_refuses_output_outside_s3_and_changed_cutoff(case_inputs, tmp_path):
    kwargs, calls = case_inputs
    with pytest.raises(ValueError, match="local S3"):
        case_runner.assemble_cases(**{**kwargs, "output_dir": tmp_path / "outside"})
    path = kwargs["score_dir"] / "science_scores.json"
    value = json.loads(path.read_text())
    value["truth_cutoff_utc"] = "2026-01-01T00:00:00+00:00"
    write_json(path, value)
    with pytest.raises(ValueError, match="cutoff differs"):
        case_runner.assemble_cases(**kwargs)
    assert not calls
