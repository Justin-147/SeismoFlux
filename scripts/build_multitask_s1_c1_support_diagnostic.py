"""Build the frozen C1-P1 causal local-completeness support diagnostic.

The command authenticates the two preregistration files and the S1 parent
inputs, then writes only deterministic support summaries and cell masks.  It
does not construct earthquake outcomes or run any forecasting method.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import yaml
from pyproj import CRS, Transformer

from seismoflux.background.grid import EQUAL_AREA_CRS
from seismoflux.background.local_support import build_local_support_base_partition
from seismoflux.data.common import canonical_json_bytes
from seismoflux.multitask_s1.local_completeness import (
    LocalCompletenessEvent,
    LocalCompletenessSnapshot,
    build_completeness_snapshot_anchors,
    build_local_completeness_snapshot,
    locate_completeness_events,
    snapshot_cell_records,
    snapshot_summary_record,
)
from seismoflux.multitask_s1.runner_inputs import (
    CATALOG_RELATIVE_PATH,
    CONTRACT_RELATIVE_PATH,
    EXPECTED_25KM_CELL_COUNT,
    EXPECTED_25KM_GRID_ID,
    STUDY_AREA_RELATIVE_PATH,
    S1RunnerInputs,
    load_s1_runner_inputs,
)

CONFIG_RELATIVE_PATH: Final = Path("configs/multitask_s1_c1_local_completeness.yaml")
DOCUMENT_RELATIVE_PATH: Final = Path("docs/s1c1_blind_protocol_2026-09-02.md")
EXPECTED_CONFIG_SHA256: Final = "303c99b280d1e62b644c2ebd02e04881026c280f5a8ee08366467763a5d58e4c"
EXPECTED_DOCUMENT_SHA256: Final = "47af15439c4d8dd48af55a7cf181fc9591bf7b9da08ab6055465dbc9245d054c"
EXPECTED_PARENT_CONTRACT_SHA256: Final = (
    "8501371eee2912547861840a43ed699885635b8934b9dfdb67e412e265674505"
)
EXPECTED_PARENT_RUN_SHA256: Final = (
    "af5ece932988827cb7f9d3db474c2541f92e54224fdefc3d820dc01d15fc0e85"
)
EXPECTED_PARENT_ACCEPTANCE_SHA256: Final = (
    "2bd16041f7d8cdaf1361b661024938f7ac0c1c6828ee4022efe98318904e9f7f"
)
SUMMARY_NAME: Final = "support_summary.json"
CELL_TABLE_NAME: Final = "support_cells.csv"
MANIFEST_NAME: Final = "manifest.json"
CELL_COLUMNS: Final = (
    "snapshot_id",
    "fold_id",
    "anchor_role",
    "block_id",
    "anchor_utc",
    "cutoff_utc",
    "cell_id",
    "row",
    "column",
    "clipped_area_m2",
    "base_event_count",
    "parent_row",
    "parent_column",
    "parent_event_count",
    "estimate_source",
    "status",
    "raw_mc",
    "main_common_mc4_training_allowed",
    "exclude_indeterminate_training_allowed",
    "supported_area_contributor",
)


class SupportDiagnosticError(RuntimeError):
    """Raised before any inconsistent diagnostic artifact can be installed."""


@dataclass(frozen=True, slots=True)
class SupportArtifactBytes:
    summary: bytes
    cells_csv: bytes
    manifest: bytes


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SupportDiagnosticError(f"{label} must be a mapping")
    return cast(Mapping[str, Any], value)


def _frozen_project_file(
    project_root: Path,
    relative_path: Path,
    expected_sha256: str,
    *,
    label: str,
) -> Path:
    path = (project_root / relative_path).resolve()
    try:
        path.relative_to(project_root)
    except ValueError as exc:
        raise SupportDiagnosticError(f"{label} escaped project_root") from exc
    if not path.is_file() or _sha256(path) != expected_sha256:
        raise SupportDiagnosticError(f"{label} is missing or changed from its frozen SHA-256")
    return path


def load_frozen_c1_protocol(project_root: Path) -> Mapping[str, Any]:
    """Authenticate both C1 preregistration files before computing diagnostics."""

    config_path = _frozen_project_file(
        project_root,
        CONFIG_RELATIVE_PATH,
        EXPECTED_CONFIG_SHA256,
        label="C1 machine protocol",
    )
    _frozen_project_file(
        project_root,
        DOCUMENT_RELATIVE_PATH,
        EXPECTED_DOCUMENT_SHA256,
        label="C1 plain-language protocol",
    )
    try:
        protocol = _mapping(yaml.safe_load(config_path.read_text(encoding="utf-8")), label="C1")
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise SupportDiagnosticError("C1 machine protocol is not valid UTF-8 YAML") from exc
    if (
        protocol.get("protocol_id") != "multitask-s1-c1-causal-local-completeness-v1"
        or protocol.get("stage") != "S1-C1"
        or protocol.get("status")
        != "score_blind_protocol_frozen_before_support_diagnostics_predictions_and_scores"
    ):
        raise SupportDiagnosticError("C1 protocol identity or frozen status changed")
    parent = _mapping(protocol.get("parent_bindings"), label="parent_bindings")
    expected_parent = {
        "S1_C0_run_contract": (
            Path("configs/multitask_s1_development_run.yaml"),
            EXPECTED_PARENT_RUN_SHA256,
        ),
        "S1_development_contract": (CONTRACT_RELATIVE_PATH, EXPECTED_PARENT_CONTRACT_SHA256),
        "S1_C0_scientific_acceptance": (
            Path("docs/s1c0_scientific_acceptance_2026-09-02.md"),
            EXPECTED_PARENT_ACCEPTANCE_SHA256,
        ),
    }
    for key, (relative, expected_hash) in expected_parent.items():
        record = _mapping(parent.get(key), label=key)
        if record.get("path") != relative.as_posix() or record.get("sha256") != expected_hash:
            raise SupportDiagnosticError(f"C1 parent binding changed: {key}")
        _frozen_project_file(project_root, relative, expected_hash, label=key)
    return protocol


def _event_time(microseconds: int) -> datetime:
    return datetime(1970, 1, 1, tzinfo=UTC) + timedelta(microseconds=int(microseconds))


def _catalog_events(inputs: S1RunnerInputs) -> tuple[LocalCompletenessEvent, ...]:
    catalog = inputs.catalog
    selected = np.flatnonzero(catalog.inside_study_area)
    transformer = Transformer.from_crs(
        CRS.from_epsg(4326), CRS.from_user_input(EQUAL_AREA_CRS), always_xy=True
    )
    x_raw, y_raw = transformer.transform(catalog.longitude[selected], catalog.latitude[selected])
    x_m = np.asarray(x_raw, dtype=np.float64)
    y_m = np.asarray(y_raw, dtype=np.float64)
    return tuple(
        LocalCompletenessEvent(
            event_id=catalog.event_ids[int(index)],
            origin_time_utc=_event_time(int(catalog.origin_time_us[index])),
            available_at_utc=_event_time(int(catalog.available_at_us[index])),
            magnitude=float(catalog.magnitude[index]),
            x_m=float(x_value),
            y_m=float(y_value),
        )
        for index, x_value, y_value in zip(selected, x_m, y_m, strict=True)
    )


def compute_support_snapshots(inputs: S1RunnerInputs) -> tuple[LocalCompletenessSnapshot, ...]:
    """Compute all 16 causal support surfaces without opening any outcome table."""

    partition = build_local_support_base_partition(inputs.spatial_domain.study_area_equal_area)
    if not math.isclose(
        partition.total_area_m2,
        inputs.location_grid.total_area_km2 * 1_000_000.0,
        rel_tol=1.0e-12,
        abs_tol=1.0e-4,
    ):
        raise SupportDiagnosticError("500 km partition changed the national area denominator")
    located = locate_completeness_events(_catalog_events(inputs), partition)
    anchors = build_completeness_snapshot_anchors(inputs.contract)
    return tuple(
        build_local_completeness_snapshot(located, anchor=anchor, partition=partition)
        for anchor in anchors
    )


def _csv_bytes(records: Sequence[Mapping[str, object]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CELL_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for record in records:
        if tuple(record) != CELL_COLUMNS:
            raise SupportDiagnosticError("support cell record columns or order changed")
        writer.writerow(record)
    return stream.getvalue().encode("utf-8")


def _artifact_identity(
    name: str, payload: bytes, *, row_count: int | None = None
) -> dict[str, object]:
    identity: dict[str, object] = {
        "path": name,
        "sha256": _sha256_bytes(payload),
        "size_bytes": len(payload),
    }
    if row_count is not None:
        identity["row_count"] = row_count
    return identity


def build_support_artifact_bytes(
    snapshots: Sequence[LocalCompletenessSnapshot],
    *,
    inputs: S1RunnerInputs,
    output_root_relative: str,
) -> SupportArtifactBytes:
    """Serialize deterministic JSON/CSV bytes and their identity manifest."""

    values = tuple(snapshots)
    if len(values) != 16 or len({item.anchor.snapshot_id for item in values}) != 16:
        raise SupportDiagnosticError("artifact build requires exactly 16 frozen snapshots")
    study_identities = {item.study_area_sha256 for item in values}
    if len(study_identities) != 1:
        raise SupportDiagnosticError("snapshots do not share one study-area geometry")
    summary_record = {
        "schema_version": 1,
        "artifact_kind": "s1_c1_causal_local_completeness_support_diagnostic",
        "protocol_id": "multitask-s1-c1-causal-local-completeness-v1",
        "spatial_rule": {
            "base_cell_km": 500.0,
            "sparse_parent_cell_km": 1000.0,
            "minimum_events": 200,
            "estimator": "maximum_curvature_peak_plus_0.2",
            "maximum_supported_raw_mc": 4.0,
            "national_domain_unchanged": True,
        },
        "snapshot_count": len(values),
        "all_snapshot_support_gates_passed": all(item.support_gate_passed for item in values),
        "snapshots": [snapshot_summary_record(item) for item in values],
    }
    summary_bytes = canonical_json_bytes(summary_record)
    cell_records = tuple(record for item in values for record in snapshot_cell_records(item))
    cell_bytes = _csv_bytes(cell_records)
    manifest_record = {
        "schema_version": 1,
        "artifact_kind": "s1_c1_causal_local_completeness_support_manifest",
        "protocol_files": {
            "machine": {
                "path": CONFIG_RELATIVE_PATH.as_posix(),
                "sha256": EXPECTED_CONFIG_SHA256,
            },
            "plain_language": {
                "path": DOCUMENT_RELATIVE_PATH.as_posix(),
                "sha256": EXPECTED_DOCUMENT_SHA256,
            },
        },
        "parent_files": {
            "development_contract": {
                "path": CONTRACT_RELATIVE_PATH.as_posix(),
                "sha256": EXPECTED_PARENT_CONTRACT_SHA256,
            },
            "development_run": {
                "path": "configs/multitask_s1_development_run.yaml",
                "sha256": EXPECTED_PARENT_RUN_SHA256,
            },
            "scientific_acceptance": {
                "path": "docs/s1c0_scientific_acceptance_2026-09-02.md",
                "sha256": EXPECTED_PARENT_ACCEPTANCE_SHA256,
            },
        },
        "input_identities": {
            "earthquake_catalog": {
                "path_from_data_root": CATALOG_RELATIVE_PATH.as_posix(),
                **dict(inputs.catalog_identity),
            },
            "study_area_file": {
                "path_from_data_root": STUDY_AREA_RELATIVE_PATH.as_posix(),
                "sha256": inputs.study_area_sha256,
            },
            "study_area_equal_area_geometry_sha256": next(iter(study_identities)),
            "operational_grid": {
                "grid_id": EXPECTED_25KM_GRID_ID,
                "cell_count": EXPECTED_25KM_CELL_COUNT,
                "area_km2": inputs.location_grid.total_area_km2,
            },
        },
        "output_root": output_root_relative,
        "snapshot_count": len(values),
        "artifacts": [
            _artifact_identity(SUMMARY_NAME, summary_bytes),
            _artifact_identity(CELL_TABLE_NAME, cell_bytes, row_count=len(cell_records)),
        ],
    }
    return SupportArtifactBytes(
        summary=summary_bytes,
        cells_csv=cell_bytes,
        manifest=canonical_json_bytes(manifest_record),
    )


def _install_exact(path: Path, payload: bytes) -> None:
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise SupportDiagnosticError(f"refusing to overwrite different bytes: {path}")
        return
    path.write_bytes(payload)
    if path.read_bytes() != payload:
        raise SupportDiagnosticError(f"installed artifact failed byte verification: {path}")


def run_support_diagnostic(*, project_root: Path, data_root: Path) -> tuple[Path, Path, Path]:
    """Authenticate, compute, and install the three frozen C1-P1 artifacts."""

    project = project_root.resolve()
    data = data_root.resolve()
    if not project.is_dir() or not data.is_dir():
        raise SupportDiagnosticError("project_root and data_root must be existing directories")
    protocol = load_frozen_c1_protocol(project)
    planned = _mapping(protocol.get("planned_artifacts"), label="planned_artifacts")
    raw_output = planned.get("support_diagnostic_root")
    if not isinstance(raw_output, str) or not raw_output:
        raise SupportDiagnosticError("C1 support diagnostic root is invalid")
    relative_output = Path(raw_output)
    if relative_output.is_absolute() or ".." in relative_output.parts:
        raise SupportDiagnosticError("C1 support diagnostic root must stay under project_root")
    output = (project / relative_output).resolve()
    try:
        output.relative_to(project)
    except ValueError as exc:
        raise SupportDiagnosticError("C1 support diagnostic root escaped project_root") from exc
    inputs = load_s1_runner_inputs(project_root=project, data_root=data)
    snapshots = compute_support_snapshots(inputs)
    artifacts = build_support_artifact_bytes(
        snapshots,
        inputs=inputs,
        output_root_relative=relative_output.as_posix(),
    )
    output.mkdir(parents=True, exist_ok=True)
    expected_names = {SUMMARY_NAME, CELL_TABLE_NAME, MANIFEST_NAME}
    unexpected = {item.name for item in output.iterdir()} - expected_names
    if unexpected:
        raise SupportDiagnosticError(
            f"support diagnostic root contains unexpected files: {sorted(unexpected)}"
        )
    paths = output / SUMMARY_NAME, output / CELL_TABLE_NAME, output / MANIFEST_NAME
    for path, payload in zip(
        paths, (artifacts.summary, artifacts.cells_csv, artifacts.manifest), strict=True
    ):
        _install_exact(path, payload)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    paths = run_support_diagnostic(project_root=args.project_root, data_root=args.data_root)
    print(json.dumps([path.as_posix() for path in paths], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
