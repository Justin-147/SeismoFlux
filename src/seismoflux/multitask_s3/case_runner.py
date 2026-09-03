"""Assemble local earthquake case records from already saved S3 diagnostics.

This never trains a model, recomputes a hit, or opens a new evaluation role.
It only joins the bounded catalog metadata needed to explain existing results.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import yaml

from seismoflux.multitask_s0 import verify_authoritative_catalog_identity
from seismoflux.multitask_s3.case_ledger import build_case_ledger
from seismoflux.multitask_s3.input_waterlevel import load_development_catalog
from seismoflux.multitask_s3.preparation import sha256, write_json


def assemble_cases(
    *, project_root: Path, data_root: Path, prediction_dir: Path, score_dir: Path, output_dir: Path
) -> dict[str, Any]:
    project = project_root.resolve()
    prediction, scores, output = (p.resolve() for p in (prediction_dir, score_dir, output_dir))
    allowed = project / "outputs/multitask_s3"
    if any(not p.is_relative_to(allowed) for p in (prediction, scores, output)):
        raise ValueError("case inputs and outputs must remain in the local S3 workspace")
    if output.exists():
        raise FileExistsError("preserve an existing case ledger; do not overwrite its evidence")
    progress = json.loads((scores / "score_progress.json").read_text(encoding="utf-8"))
    science_path = scores / "science_scores.json"
    science = json.loads(science_path.read_text(encoding="utf-8"))
    prediction_path = prediction / "prediction_manifest.json"
    manifest = json.loads(prediction_path.read_text(encoding="utf-8"))
    prediction_hash = sha256(prediction_path)
    if (
        progress["status"] != "complete"
        or manifest["status"] != "predictions_complete"
        or progress["prediction_manifest_sha256"] != prediction_hash
        or science["prediction_manifest_sha256"] != prediction_hash
        or science["prediction_identity"] != manifest["identity"]
    ):
        raise ValueError("case explanation requires the same completed predictions and scores")
    protocol_path = project / "configs/multitask_s3_anomaly.yaml"
    prepared_identity = manifest["identity"]["prepared_inputs"]
    if sha256(protocol_path) != prepared_identity["protocol_sha256"]:
        raise ValueError("frozen protocol changed since the scored predictions")
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    catalog_path = data_root / protocol["access"]["catalog"]
    if verify_authoritative_catalog_identity(catalog_path) != prepared_identity["catalog_identity"]:
        raise ValueError("case metadata must use the original catalog identity")
    truth_cutoff = datetime.fromisoformat(manifest["truth_cutoff_utc"])
    if science["truth_cutoff_utc"] != manifest["truth_cutoff_utc"]:
        raise ValueError("case metadata cutoff differs from the scored trial")
    diagnostic_path = scores / "event_diagnostics_local.json"
    diagnostics = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    if diagnostics.get("local_only") is not True:
        raise ValueError("event diagnostics must remain local-only")
    records = diagnostics["records"]
    event_ids = {event_id for record in records for event_id in record["target_event_ids"]}
    frame = load_development_catalog(catalog_path, truth_cutoff=truth_cutoff)
    selected = frame[frame["event_id"].astype(str).isin(event_ids)]
    if selected["event_id"].astype(str).duplicated().any():
        raise ValueError("duplicate earthquake metadata identifier")
    metadata = {
        str(row.event_id): {
            "origin_time_utc": row.origin_time_utc.isoformat(),
            "available_at": row.available_at.isoformat(),
            "magnitude": float(row.magnitude),
            "longitude": float(row.longitude),
            "latitude": float(row.latitude),
        }
        for row in selected.itertuples(index=False)
    }
    ledger = build_case_ledger(records, catalog_metadata=metadata)
    ledger["provenance"] = {
        "assembled_at_utc": datetime.now(UTC).isoformat(),
        "prediction_manifest_sha256": prediction_hash,
        "science_scores_sha256": sha256(science_path),
        "event_diagnostics_sha256": sha256(diagnostic_path),
        "catalog_identity": prepared_identity["catalog_identity"],
        "truth_cutoff_utc": truth_cutoff.isoformat(),
        "model_refitted": False,
        "hits_recomputed": False,
        "new_evaluation_role_accessed": False,
    }
    output.mkdir(parents=True)
    write_json(output / "case_ledger_local.json", ledger)
    return ledger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project-root", "data-root", "prediction-dir", "score-dir", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if any(
        os.environ.get(name) != "1"
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    ):
        raise RuntimeError("launch numerical libraries with one thread each")
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    ledger = assemble_cases(
        project_root=args.project_root,
        data_root=args.data_root,
        prediction_dir=args.prediction_dir,
        score_dir=args.score_dir,
        output_dir=args.output_dir,
    )
    print(f"Local case ledger assembled: {len(ledger['rows'])} comparison rows; no refit/rescore.")


if __name__ == "__main__":
    main()
