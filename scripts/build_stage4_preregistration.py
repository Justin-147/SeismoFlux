"""Generate or verify the target-blind stage-4 preregistration artifacts."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import pyarrow.parquet as pq
import yaml

from seismoflux.anomaly_increment import (
    build_construction_strata,
    build_exposure_preregistration,
    build_random_input_seal,
    build_randomness_manifest,
    protocol_design_sha256,
    select_feature_storage_columns,
    validate_stage4_protocol_bundle,
    verify_content_sha256,
    write_public_manifest_atomic,
)
from seismoflux.config import sha256_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT_ROOT / "configs" / "anomaly_increment.yaml"
LOCAL_OUTPUT_DIR = PROJECT_ROOT / "data" / "interim" / "stage4" / "anomaly_increment"


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be a string-keyed mapping")
    return cast(dict[str, Any], value)


def _load_yaml(path: Path) -> dict[str, Any]:
    return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), label=str(path))


def _load_json(path: Path) -> dict[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), label=str(path))


def _project_path(value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{label} must be a normalized project-relative POSIX path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or value != relative.as_posix():
        raise ValueError(f"{label} must be a normalized project-relative POSIX path")
    resolved = (PROJECT_ROOT / relative).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"{label} escapes the project root")
    return resolved


def _external_root(value: str | None) -> Path:
    raw = value or os.environ.get("SEISMOFLUX_LOCATIONPRED_ROOT")
    if not raw:
        raise ValueError(
            "set SEISMOFLUX_LOCATIONPRED_ROOT or pass --external-root for L1/L2 linework"
        )
    root = Path(raw).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    return root


def _input(protocol: dict[str, Any], input_id: str) -> dict[str, Any]:
    return _mapping(_mapping(protocol["inputs"], label="inputs")[input_id], label=input_id)


def _verify_file(path: Path, expected_sha256: object, *, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if not isinstance(expected_sha256, str) or sha256_file(path) != expected_sha256:
        raise ValueError(f"{label} SHA-256 does not match the frozen protocol")


def _anomaly_issue_dates(report_period_path: Path) -> tuple[Any, ...]:
    table = pq.read_table(report_period_path, columns=["available_at"])
    timezone = ZoneInfo("Asia/Shanghai")
    values = []
    for raw in table["available_at"].to_pylist():
        if not isinstance(raw, datetime) or raw.tzinfo is None:
            raise ValueError("anomaly report available_at must be timezone-aware")
        values.append(raw.astimezone(timezone).date())
    result = tuple(sorted(set(values)))
    if len(result) != table.num_rows:
        raise ValueError("anomaly report periods must resolve to unique local issue dates")
    return result


def _source_paths(
    protocol: dict[str, Any], external_root: Path
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    ids = (
        "background_fold_manifest",
        "feature_dictionary",
        "stage3_feature_store",
        "stage3_anomaly_state_history",
        "anomaly_report_period",
        "study_area",
    )
    entries = {input_id: _input(protocol, input_id) for input_id in ids}
    paths = {
        input_id: _project_path(entry["path"], label=f"inputs.{input_id}.path")
        for input_id, entry in entries.items()
    }
    for input_id, path in paths.items():
        _verify_file(path, entries[input_id]["sha256"], label=input_id)

    for input_id in ("construction_linework_l1", "construction_linework_l2"):
        entry = _input(protocol, input_id)
        relative = Path(cast(str, entry["source_inventory_relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"{input_id} external relative path is invalid")
        path = (external_root / relative).resolve()
        if not path.is_relative_to(external_root):
            raise ValueError(f"{input_id} escapes the external root")
        _verify_file(path, entry["sha256"], label=input_id)
        paths[input_id] = path
        entries[input_id] = entry
    return paths, entries


def _generated_paths(protocol: dict[str, Any]) -> dict[str, Path]:
    generated = _mapping(protocol["generated_manifests"], label="generated_manifests")
    return {
        manifest_id: _project_path(
            _mapping(entry, label=f"generated_manifests.{manifest_id}")["path"],
            label=f"generated_manifests.{manifest_id}.path",
        )
        for manifest_id, entry in generated.items()
    }


def _update_protocol_manifest_hashes(file_hashes: dict[str, str]) -> None:
    lines = PROTOCOL_PATH.read_text(encoding="utf-8").splitlines()
    in_generated = False
    current_manifest: str | None = None
    updated: set[str] = set()
    rewritten: list[str] = []
    for line in lines:
        if line == "generated_manifests:":
            in_generated = True
            rewritten.append(line)
            continue
        if in_generated and line and not line.startswith(" "):
            in_generated = False
            current_manifest = None
        if in_generated and line.startswith("  ") and not line.startswith("    "):
            candidate = line.strip()
            if candidate.endswith(":"):
                current_manifest = candidate[:-1]
        if in_generated and current_manifest in file_hashes and line.startswith("    sha256:"):
            rewritten.append(f"    sha256: {file_hashes[current_manifest]}")
            updated.add(current_manifest)
        else:
            rewritten.append(line)
    if updated != set(file_hashes):
        raise ValueError(
            f"could not update every generated manifest hash: updated={sorted(updated)}"
        )
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=PROTOCOL_PATH.parent,
            prefix=f".{PROTOCOL_PATH.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write("\n".join(rewritten) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, PROTOCOL_PATH)
        temporary_name = None
    except Exception:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise


def generate(protocol: dict[str, Any], external_root: Path) -> dict[str, Any]:
    paths, entries = _source_paths(protocol, external_root)
    output_paths = _generated_paths(protocol)

    fold = build_exposure_preregistration(
        _load_json(paths["background_fold_manifest"]),
        anomaly_issue_dates_local=_anomaly_issue_dates(paths["anomaly_report_period"]),
    )
    feature = select_feature_storage_columns(
        _load_json(paths["feature_dictionary"]),
        protocol,
    )
    spatial_result = build_construction_strata(
        l1_gmt_path=paths["construction_linework_l1"],
        l2_gmt_path=paths["construction_linework_l2"],
        study_area_path=paths["study_area"],
        stage3_feature_store_path=paths["stage3_feature_store"],
        stage3_state_history_path=paths["stage3_anomaly_state_history"],
        local_output_dir=LOCAL_OUTPUT_DIR,
        project_root=PROJECT_ROOT,
        expected_l1_sha256=cast(str, entries["construction_linework_l1"]["sha256"]),
        expected_l2_sha256=cast(str, entries["construction_linework_l2"]["sha256"]),
        expected_study_area_sha256=cast(str, entries["study_area"]["sha256"]),
        expected_stage3_feature_store_sha256=cast(str, entries["stage3_feature_store"]["sha256"]),
        expected_stage3_state_history_sha256=cast(
            str, entries["stage3_anomaly_state_history"]["sha256"]
        ),
    )
    spatial = spatial_result.public_summary
    seal = build_random_input_seal(
        protocol,
        fold_manifest=fold,
        feature_manifest=feature,
        spatial_manifest=spatial,
    )
    randomness = build_randomness_manifest(
        frozen_input_seal_sha256=cast(str, seal["content_sha256"])
    )
    payloads = {
        "fold": fold,
        "feature_set": feature,
        "randomness": randomness,
        "spatial_strata": spatial,
    }
    file_hashes = {
        manifest_id: write_public_manifest_atomic(output_paths[manifest_id], payload)
        for manifest_id, payload in payloads.items()
    }
    _update_protocol_manifest_hashes(file_hashes)
    return {
        "action": "generate",
        "file_sha256": file_hashes,
        "protocol_design_sha256": protocol_design_sha256(protocol),
        "protocol_generated_manifest_hashes_updated": True,
        "random_input_seal": seal,
        "target_read_count": 0,
    }


def check(protocol: dict[str, Any]) -> dict[str, Any]:
    output_paths = _generated_paths(protocol)
    generated = _mapping(protocol["generated_manifests"], label="generated_manifests")
    documents: dict[str, dict[str, Any]] = {}
    for manifest_id, path in output_paths.items():
        document = _load_json(path)
        if not verify_content_sha256(document):
            raise ValueError(f"{manifest_id} has an invalid content_sha256")
        expected = _mapping(generated[manifest_id], label=manifest_id)["sha256"]
        _verify_file(path, expected, label=f"generated manifest {manifest_id}")
        documents[manifest_id] = document

    spatial = documents["spatial_strata"]
    local_artifacts = _mapping(spatial["local_artifacts"], label="local_artifacts")
    topology = _mapping(
        protocol["spatial_permutation_topology"], label="spatial_permutation_topology"
    )
    configured = _mapping(
        topology["local_restricted_artifacts"], label="local_restricted_artifacts"
    )
    path_keys = {
        "cell_mapping": "cell_mapping",
        "entity_mapping": "entity_mapping",
        "connectors": "connectors",
        "zone_geometry": "zone_geometry",
    }
    for artifact_id, config_key in path_keys.items():
        artifact = _mapping(local_artifacts[artifact_id], label=artifact_id)
        path = _project_path(configured[config_key], label=config_key)
        _verify_file(path, artifact["sha256"], label=f"local artifact {artifact_id}")
        if path.stat().st_size != artifact["byte_count"]:
            raise ValueError(f"local artifact {artifact_id} byte count changed")

    seal = build_random_input_seal(
        protocol,
        fold_manifest=documents["fold"],
        feature_manifest=documents["feature_set"],
        spatial_manifest=spatial,
    )
    randomness = documents["randomness"]
    if randomness["frozen_input_seal_sha256"] != seal["content_sha256"]:
        raise ValueError("randomness manifest is not bound to the current frozen design")
    validation = validate_stage4_protocol_bundle(
        protocol,
        fold_manifest=documents["fold"],
        feature_manifest=documents["feature_set"],
        randomness_manifest=randomness,
        spatial_manifest=spatial,
    )
    return {
        "action": "check",
        "local_artifact_count": len(path_keys),
        "manifest_count": len(documents),
        "protocol_design_sha256": protocol_design_sha256(protocol),
        "random_input_seal_sha256": seal["content_sha256"],
        "target_read_count": 0,
        "validation_content_sha256": validation["content_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("generate", "check"))
    parser.add_argument("--external-root")
    args = parser.parse_args()
    protocol = _load_yaml(PROTOCOL_PATH)
    if protocol.get("protocol_version") != "0.4.0":
        raise ValueError("stage-4 preregistration builder requires protocol_version 0.4.0")
    result = (
        generate(protocol, _external_root(args.external_root))
        if args.action == "generate"
        else check(protocol)
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
