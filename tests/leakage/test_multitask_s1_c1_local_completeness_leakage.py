from __future__ import annotations

import csv
import importlib.util
import io
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

from shapely.geometry import box

from seismoflux.background.local_support import build_local_support_base_partition
from seismoflux.multitask_s1.local_completeness import (
    CompletenessSnapshotAnchor,
    LocalCompletenessEvent,
    build_local_completeness_snapshot,
    locate_completeness_events,
)

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/build_multitask_s1_c1_support_diagnostic.py"
FORBIDDEN_FIELD_PARTS = (
    "target",
    "prediction",
    "score",
    "model_selection",
    "holdout",
    "audit",
    "locked",
)


def _script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("s1c1_support_diagnostic", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _anchor(index: int) -> CompletenessSnapshotAnchor:
    anchor = datetime(2000, 1, 2, tzinfo=UTC) + timedelta(days=index)
    return CompletenessSnapshotAnchor(
        snapshot_id=f"C_DEV_SYNTH_{index:02d}__I1",
        fold_id=f"C_DEV_SYNTH_{index:02d}",
        role="inner_block_start",
        block_id="I1",
        anchor_utc=anchor,
        cutoff_utc=anchor - timedelta(hours=24),
    )


def _collect_keys(value: object) -> list[str]:
    if isinstance(value, dict):
        return [
            *(str(key).lower() for key in value),
            *(item for child in value.values() for item in _collect_keys(child)),
        ]
    if isinstance(value, list):
        return [item for child in value for item in _collect_keys(child)]
    return []


def test_dual_cutoff_excludes_future_and_delayed_rows_before_any_cell_count() -> None:
    cutoff = datetime(2000, 1, 1, tzinfo=UTC)
    events = (
        LocalCompletenessEvent("visible", cutoff, cutoff, 3.5, 250_000.0, 250_000.0),
        LocalCompletenessEvent(
            "delayed",
            cutoff - timedelta(days=1),
            cutoff + timedelta(microseconds=1),
            3.5,
            250_000.0,
            250_000.0,
        ),
        LocalCompletenessEvent(
            "future",
            cutoff + timedelta(microseconds=1),
            cutoff + timedelta(microseconds=1),
            3.5,
            250_000.0,
            250_000.0,
        ),
    )
    partition = build_local_support_base_partition(box(0.0, 0.0, 500_000.0, 500_000.0))
    snapshot = build_local_completeness_snapshot(
        locate_completeness_events(events, partition),
        anchor=CompletenessSnapshotAnchor(
            "C_DEV_SYNTH__I1",
            "C_DEV_SYNTH",
            "inner_block_start",
            "I1",
            cutoff + timedelta(hours=24),
            cutoff,
        ),
        partition=partition,
    )
    assert snapshot.visible_event_count == 1
    assert snapshot.cells[0].base_event_count == 1


def test_serialized_c1_p1_artifacts_bind_both_protocols_and_expose_no_forbidden_fields() -> None:
    module = _script_module()
    partition = build_local_support_base_partition(box(0.0, 0.0, 500_000.0, 500_000.0))
    located = locate_completeness_events((), partition)
    snapshots = tuple(
        build_local_completeness_snapshot(located, anchor=_anchor(index), partition=partition)
        for index in range(16)
    )
    inputs = SimpleNamespace(
        catalog_identity={
            "status": "verified_authoritative_fail_closed",
            "file_sha256": "1" * 64,
            "row_count": 40_898,
            "arrow_schema": [],
        },
        study_area_sha256="2" * 64,
        location_grid=SimpleNamespace(total_area_km2=250_000.0),
    )
    artifacts = module.build_support_artifact_bytes(
        snapshots,
        inputs=cast(Any, inputs),
        output_root_relative="outputs/multitask_s1/s1c1_local_completeness_v1/support_diagnostic",
    )
    summary = json.loads(artifacts.summary)
    manifest = json.loads(artifacts.manifest)
    assert manifest["protocol_files"]["machine"]["sha256"] == (
        "303c99b280d1e62b644c2ebd02e04881026c280f5a8ee08366467763a5d58e4c"
    )
    assert manifest["protocol_files"]["plain_language"]["sha256"] == (
        "47af15439c4d8dd48af55a7cf181fc9591bf7b9da08ab6055465dbc9245d054c"
    )
    keys = _collect_keys(summary) + _collect_keys(manifest)
    rows = list(csv.DictReader(io.StringIO(artifacts.cells_csv.decode("utf-8"))))
    assert len(rows) == 16
    keys.extend(name.lower() for name in rows[0])
    assert not [key for key in keys if any(forbidden in key for forbidden in FORBIDDEN_FIELD_PARTS)]
    assert all(row["main_common_mc4_training_allowed"] == "True" for row in rows)
    assert all(row["exclude_indeterminate_training_allowed"] == "False" for row in rows)


def test_different_existing_bytes_are_never_overwritten(tmp_path: Path) -> None:
    module = _script_module()
    path = tmp_path / "manifest.json"
    path.write_bytes(b"existing")
    try:
        module._install_exact(path, b"different")
    except module.SupportDiagnosticError as error:
        assert "refusing to overwrite different bytes" in str(error)
    else:
        raise AssertionError("different bytes were unexpectedly accepted")
    assert path.read_bytes() == b"existing"
