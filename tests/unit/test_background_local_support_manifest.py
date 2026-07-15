from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError
from shapely.geometry import box

from seismoflux.background.completeness import CompletenessEvent
from seismoflux.background.local_support_manifest import (
    EXPECTED_LOCAL_SUPPORT_SNAPSHOTS,
    LocalSupportSourceFile,
    LocalSupportSources,
    background_local_support_manifest_bytes,
    build_background_local_support_manifest,
    load_background_local_support_manifest,
    validate_background_local_support_study_area,
)


def _events(magnitudes: Sequence[float]) -> tuple[CompletenessEvent, ...]:
    start = datetime(1970, 1, 2, tzinfo=UTC)
    return tuple(
        CompletenessEvent(
            event_id=f"history-{index:04d}",
            origin_time_utc=start + timedelta(minutes=index),
            available_at=start + timedelta(minutes=index),
            magnitude=magnitude,
            inside_study_area=True,
            x_m=1_000.0,
            y_m=1_000.0,
        )
        for index, magnitude in enumerate(magnitudes)
    )


def _sources() -> LocalSupportSources:
    return LocalSupportSources(
        earthquake_dataset=LocalSupportSourceFile(
            path="data/processed/stage1/snapshot/earthquake_event.parquet",
            sha256="1" * 64,
        ),
        study_area=LocalSupportSourceFile(
            path="data/processed/china_mainland.geojson",
            sha256="2" * 64,
        ),
    )


@pytest.fixture(scope="module")
def manifest_bytes() -> bytes:
    manifest = build_background_local_support_manifest(
        _events([3.0] * 150 + [3.2] * 50),
        study_area_equal_area=box(0.0, 0.0, 400_000.0, 400_000.0),
        sources=_sources(),
    )
    return background_local_support_manifest_bytes(manifest)


def test_five_snapshot_manifest_round_trip_is_deterministic_and_target_free(
    tmp_path: Path,
    manifest_bytes: bytes,
) -> None:
    path = tmp_path / "support.json"
    path.write_bytes(manifest_bytes)

    loaded = load_background_local_support_manifest(path)

    assert (
        tuple((entry.snapshot_id, entry.support.fit_end_utc) for entry in loaded.snapshots)
        == EXPECTED_LOCAL_SUPPORT_SNAPSHOTS
    )
    assert background_local_support_manifest_bytes(loaded) == manifest_bytes
    assert loaded.manifest_id.startswith("local-support-bundle-")
    decoded = manifest_bytes.decode("utf-8").casefold()
    assert "clipped_geometry_wkb_hex" not in decoded
    assert '"coordinates"' not in decoded
    for forbidden in (
        "assessment_target_count",
        "validation_target_locations",
        "model_score",
        "information_gain",
        "hit_result",
        "model_selection",
        "score_id",
    ):
        assert forbidden not in decoded


def test_manifest_rejects_hidden_result_fields_before_schema_validation(
    tmp_path: Path,
    manifest_bytes: bytes,
) -> None:
    payload = json.loads(manifest_bytes)
    payload["snapshots"][0]["support"]["model_score"] = 1.0
    path = tmp_path / "contaminated.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="forbidden result field"):
        load_background_local_support_manifest(path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("support_id", "support_id does not match"),
        ("snapshot_order", "five frozen snapshots"),
        ("cell_area", "fixed-cell areas must sum"),
        ("manifest_id", "manifest_id does not match"),
    ],
)
def test_manifest_rejects_identity_or_geometry_drift(
    tmp_path: Path,
    manifest_bytes: bytes,
    mutation: str,
    message: str,
) -> None:
    payload = json.loads(manifest_bytes)
    if mutation == "support_id":
        payload["snapshots"][0]["support"]["support_id"] = "local-support-0000000000000000"
    elif mutation == "snapshot_order":
        payload["snapshots"][0], payload["snapshots"][1] = (
            payload["snapshots"][1],
            payload["snapshots"][0],
        )
    elif mutation == "cell_area":
        payload["fixed_cells"][0]["clipped_area_m2"] += 1.0
    else:
        payload["manifest_id"] = "local-support-bundle-0000000000000000"
    path = tmp_path / f"{mutation}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises((ValueError, ValidationError), match=message):
        load_background_local_support_manifest(path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("dense_count_below_minimum", "dense base support"),
        ("cell_candidate_mapping", "frozen upward mapping"),
        ("indeterminate_parent_dense", "inconsistent with its fixed 1000 km parent"),
        ("nonfinite_cell_mc", "raw Mc must be finite"),
        ("temporal_candidate_mapping", "frozen upward mapping"),
        ("temporal_count_eligibility", "eligibility differs"),
        ("temporal_calendar", "frozen calendar"),
    ],
)
def test_manifest_rejects_scientifically_inconsistent_causal_evidence(
    tmp_path: Path,
    manifest_bytes: bytes,
    mutation: str,
    message: str,
) -> None:
    payload = json.loads(manifest_bytes)
    support = payload["snapshots"][0]["support"]
    cell = support["cells"][0]
    temporal = support["temporal_blocks"][0]
    if mutation == "dense_count_below_minimum":
        cell["base_event_count"] = 199
        cell["source_event_count"] = 199
    elif mutation == "cell_candidate_mapping":
        cell["raw_mc"] = 3.0
    elif mutation == "indeterminate_parent_dense":
        cell.update(
            {
                "status": "indeterminate",
                "source": "fixed_1000km_parent",
                "base_event_count": 199,
                "source_event_count": 200,
                "parent_cell_id": "g1000000000_r+0000000_c+0000000",
                "raw_mc": None,
                "candidate_mc": None,
                "applied_mc": support["common_mc"],
            }
        )
    elif mutation == "nonfinite_cell_mc":
        cell["raw_mc"] = float("nan")
    elif mutation == "temporal_candidate_mapping":
        temporal["raw_mc"] = 3.0
    elif mutation == "temporal_count_eligibility":
        temporal["event_count"] = 199
    else:
        temporal["block_id"] = "1970-wrong"
    path = tmp_path / f"semantic-{mutation}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises((ValueError, ValidationError), match=message):
        load_background_local_support_manifest(path)


def test_source_paths_must_be_project_relative() -> None:
    with pytest.raises(ValidationError, match="project-relative"):
        LocalSupportSourceFile(path="../outside.json", sha256="1" * 64)


def test_manifest_is_bound_to_locally_reconstructed_study_area(
    tmp_path: Path,
    manifest_bytes: bytes,
) -> None:
    path = tmp_path / "support.json"
    path.write_bytes(manifest_bytes)
    loaded = load_background_local_support_manifest(path)

    validate_background_local_support_study_area(
        loaded,
        box(0.0, 0.0, 400_000.0, 400_000.0),
    )

    with pytest.raises(ValueError, match="fixed-cell IDs, indices, or clipped areas"):
        validate_background_local_support_study_area(
            loaded,
            box(0.0, 0.0, 399_000.0, 400_000.0),
        )

    with pytest.raises(ValueError, match="study-area digest"):
        validate_background_local_support_study_area(
            loaded,
            box(10_000.0, 0.0, 410_000.0, 400_000.0),
        )
