from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest

from seismoflux.data.common import canonical_json_bytes
from seismoflux.p1_b0_r30 import operations
from seismoflux.p1_b0_r30.operations import (
    COUNT_BODY_FILENAME,
    FORECAST_GRID_FILENAME,
    OFFLINE_HTML_FILENAME,
    SOURCE_RECEIPT_FILENAME,
    STATIC_SVG_FILENAME,
    P1IssueArtifactError,
    build_forecast_issue_record_fields,
    prepare_production_issue_artifacts,
    verify_prepared_issue_against_frozen_inputs,
    verify_prepared_issue_artifacts,
)
from seismoflux.p1_b0_r30.preflight import EXPECTED_SUPPORT_MANIFEST_SHA256
from seismoflux.p1_b0_r30.production import (
    ComCatHttpExchange,
    ComCatIssueInputAcquisition,
    P1IssueSchedule,
    build_comcat_count_snapshot,
    build_comcat_snapshot,
    build_issue_count_url,
    build_issue_query_url,
    issue_schedule,
)
from seismoflux.p1_b0_r30.production_rendering import parse_production_forecast_view
from seismoflux.p1_b0_r30.prospective import (
    P1_MODEL_MANIFEST_SHA256,
    P1_SOURCE_BOUNDARY_MANIFEST_SHA256,
    ProductionForecastBundle,
)
from seismoflux.p1_b0_r30.records import build_record, validate_record_chain


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _acquisition(schedule: P1IssueSchedule) -> ComCatIssueInputAcquisition:
    count_bytes = canonical_json_bytes({"count": 0})
    count = build_comcat_count_snapshot(
        ComCatHttpExchange(
            request_url=build_issue_count_url(schedule),
            fetch_started_at_utc=schedule.query_cutoff_utc,
            fetch_completed_at_utc=schedule.query_cutoff_utc + timedelta(seconds=1),
            http_status=200,
            response_headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(count_bytes)),
            },
            raw_response_bytes=count_bytes,
        ),
        schedule=schedule,
    )
    query_bytes = canonical_json_bytes(
        {"features": [], "metadata": {"count": 0}, "type": "FeatureCollection"}
    )
    query = build_comcat_snapshot(
        ComCatHttpExchange(
            request_url=build_issue_query_url(schedule),
            fetch_started_at_utc=schedule.query_cutoff_utc + timedelta(seconds=2),
            fetch_completed_at_utc=schedule.query_cutoff_utc + timedelta(seconds=3),
            http_status=200,
            response_headers={
                "Content-Type": "application/geo+json",
                "Content-Length": str(len(query_bytes)),
            },
            raw_response_bytes=query_bytes,
        ),
        schedule=schedule,
    )
    return ComCatIssueInputAcquisition(
        status="available",
        count_snapshot=count,
        query_snapshot=query,
        unavailable_reason=None,
    )


@dataclass(frozen=True)
class _CatalogIdentity:
    row_count: int = 40_898
    file_sha256: str = "5" * 64
    content_sha256: str = "6" * 64
    schema_sha256: str = "7" * 64


class _CatalogArtifact:
    def __init__(self) -> None:
        self.catalog = type("SyntheticCatalog", (), {"identity": _CatalogIdentity()})()

    def audit_mapping(self) -> dict[str, object]:
        return {
            "combined_catalog_sha256": "5" * 64,
            "combined_catalog_row_count": 40_898,
            "retained_comcat_event_count": 0,
            "cutover_match_count": 0,
            "cutover_matches": [],
            "available_at_semantics": "ComCat_provider_updated_at_conservative_lte_Q",
        }


def _record_forecasts(mapping: dict[str, object]) -> list[dict[str, object]]:
    view = parse_production_forecast_view(mapping)
    cell_ids = [cell.cell_id for cell in view.cells]
    result: list[dict[str, object]] = []
    for model in view.models:
        result.append(
            {
                "model_id": model.model_id,
                "relative_intensity_grid_sha256": _sha(
                    canonical_json_bytes(
                        {
                            "domain": "seismoflux.p1.relative-intensity-grid.v1",
                            "model_id": model.model_id,
                            "cell_ids": cell_ids,
                            "normalized_cell_mass_hex": [
                                value.hex() for value in model.normalized_cell_mass
                            ],
                        }
                    )
                ),
                "alarm_mask_sha256": _sha(
                    canonical_json_bytes(
                        {
                            "domain": "seismoflux.p1.real-history-alarm-mask.v1",
                            "model_id": model.model_id,
                            "cell_ids": list(model.alarm_cell_ids),
                            "actual_area_km2_hex": model.actual_alarm_area_km2.hex(),
                        }
                    )
                ),
                "alarm_ranking_sha256": _sha(
                    canonical_json_bytes(
                        {
                            "domain": "seismoflux.p1.real-history-ranking.v1",
                            "model_id": model.model_id,
                            "cell_ids": list(model.ranked_cell_ids),
                        }
                    )
                ),
                "actual_alarm_area_km2": model.actual_alarm_area_km2,
            }
        )
    return result


class _SyntheticBundle:
    def __init__(self) -> None:
        self.schedule = issue_schedule(datetime(2026, 9, 9, 16, tzinfo=UTC))
        self.acquisition = _acquisition(self.schedule)
        self.catalog_artifact = _CatalogArtifact()
        self.b0_source_count = 5_991
        self.recent_source_count = 0
        cells = [
            {
                "cell_id": f"r{row:02d}c{column:02d}",
                "row": row,
                "column": column,
                "area_km2": 625.0,
                "support_status": "supported",
            }
            for row in range(31)
            for column in range(31)
        ]
        weights = [float(len(cells) - index) for index in range(len(cells))]
        total = sum(weights)
        mass = [value / total for value in weights]
        selected = [cast(str, cell["cell_id"]) for cell in cells[:960]]
        source_snapshot = _sha(canonical_json_bytes(self.acquisition.as_mapping()))
        query = self.acquisition.query_snapshot
        assert query is not None
        source_request = _sha(
            canonical_json_bytes(
                {
                    "count_request_url": self.acquisition.count_snapshot.request_url,
                    "query_request_url": query.request_url,
                }
            )
        )
        self._mapping: dict[str, object] = {
            "issue_id": self.schedule.issue_id,
            "scheduled_issue_time_utc": "2026-09-09T16:00:00Z",
            "query_cutoff_utc": "2026-09-09T15:45:00Z",
            "source_snapshot_sha256": source_snapshot,
            "source_request_sha256": source_request,
            "support_manifest_sha256": EXPECTED_SUPPORT_MANIFEST_SHA256,
            "code_commit": "a" * 40,
            "B0_source_count": self.b0_source_count,
            "R30_source_count": self.recent_source_count,
            "grid": {"cell_size_km": 25.0, "cells": cells},
            "models": {
                "B0": {
                    "normalized_cell_mass": mass,
                    "alarm_cell_ids": selected,
                    "actual_alarm_area_km2": 600_000.0,
                    "next_complete_cell_area_km2": 625.0,
                },
                "B0_R30": {
                    "normalized_cell_mass": mass,
                    "alarm_cell_ids": selected,
                    "actual_alarm_area_km2": 600_000.0,
                    "next_complete_cell_area_km2": 625.0,
                },
            },
        }

    def forecast_mapping(self, *, code_commit: str) -> dict[str, object]:
        mapping = copy.deepcopy(self._mapping)
        mapping["code_commit"] = code_commit
        return mapping

    def record_forecasts(self) -> list[dict[str, object]]:
        return _record_forecasts(self.forecast_mapping(code_commit="a" * 40))

    @property
    def source_snapshot_sha256(self) -> str:
        value = self._mapping["source_snapshot_sha256"]
        assert isinstance(value, str)
        return value


def _bundle() -> ProductionForecastBundle:
    return cast(ProductionForecastBundle, cast(object, _SyntheticBundle()))


def _created() -> datetime:
    return datetime(2026, 9, 9, 15, 46, tzinfo=UTC)


def test_prepare_is_write_once_future_blind_and_builds_record_fields(tmp_path: Path) -> None:
    bundle = _bundle()
    prepared = prepare_production_issue_artifacts(
        bundle,
        issue_parent=tmp_path,
        code_commit="a" * 40,
        forecast_created_at_utc=_created(),
    )

    expected_names = {
        operations.ARTIFACT_MANIFEST_FILENAME,
        operations.COUNT_BODY_FILENAME,
        operations.COUNT_HEADERS_FILENAME,
        operations.QUERY_BODY_FILENAME,
        operations.QUERY_HEADERS_FILENAME,
        operations.SOURCE_RECEIPT_FILENAME,
        operations.FORECAST_GRID_FILENAME,
        operations.STATIC_SVG_FILENAME,
        operations.OFFLINE_HTML_FILENAME,
        operations.PREPARED_RECEIPT_FILENAME,
    }
    assert {item.name for item in prepared.issue_directory.iterdir()} == expected_names
    assert (prepared.issue_directory / COUNT_BODY_FILENAME).read_bytes() == canonical_json_bytes(
        {"count": 0}
    )
    assert (prepared.issue_directory / STATIC_SVG_FILENAME).read_bytes().startswith(b"<svg")
    html = (prepared.issue_directory / OFFLINE_HTML_FILENAME).read_text(encoding="utf-8")
    assert "http://" not in html and "https://" not in html and "fetch(" not in html
    assert "长期目录入选数 <strong>5,991</strong>" in html
    assert "最近30天M4+入选数 <strong>0</strong>" in html
    source_receipt = (prepared.issue_directory / SOURCE_RECEIPT_FILENAME).read_text(
        encoding="utf-8"
    )
    assert str(tmp_path) not in source_receipt
    assert 'combined_catalog_bytes_published":false' in source_receipt
    assert 'private_local_catalog_bytes_published":false' in source_receipt
    assert "local_event_id" not in source_receipt and "cutover_matches" not in source_receipt
    assert ".parquet" not in " ".join(expected_names)

    verified = verify_prepared_issue_artifacts(prepared.issue_directory)
    assert verified.issue_id == "p1-20260909T160000Z"
    assert verified.artifact_manifest_sha256 == prepared.artifact_manifest_sha256

    def synthetic_rebuilder(
        *,
        schedule: P1IssueSchedule,
        acquisition: ComCatIssueInputAcquisition,
        local_catalog_bytes: bytes,
        study_area_bytes: bytes,
        support_manifest_bytes: bytes,
    ) -> ProductionForecastBundle:
        assert schedule == bundle.schedule
        assert acquisition.as_mapping() == bundle.acquisition.as_mapping()
        assert local_catalog_bytes == b"frozen-local-fixture"
        assert study_area_bytes == b"frozen-study-area-fixture"
        assert support_manifest_bytes == b"frozen-support-fixture"
        return bundle

    replayed = verify_prepared_issue_against_frozen_inputs(
        prepared.issue_directory,
        local_catalog_bytes=b"frozen-local-fixture",
        study_area_bytes=b"frozen-study-area-fixture",
        support_manifest_bytes=b"frozen-support-fixture",
        forecast_rebuilder=synthetic_rebuilder,
    )
    assert replayed == verified
    fields = build_forecast_issue_record_fields(
        verified,
        protocol_definition_sha256="b" * 64,
        authorization_record_sha256="c" * 64,
        publication_completed_at_utc=datetime(2026, 9, 9, 15, 50, tzinfo=UTC),
        recorded_at_utc=datetime(2026, 9, 9, 15, 51, tzinfo=UTC),
    )
    assert fields["status"] == "on_time"
    assert fields["source_snapshot_sha256"] == verified.source_snapshot_sha256
    assert fields["forecasts"] == [dict(item) for item in verified.forecasts]
    assert fields["B0_reference_area_km2"] == 600_000.0
    assert fields["actual_area_difference_km2"] == 0.0

    with pytest.raises(P1IssueArtifactError, match="already exists"):
        prepare_production_issue_artifacts(
            bundle,
            issue_parent=tmp_path,
            code_commit="a" * 40,
            forecast_created_at_utc=_created(),
        )


def test_prepare_rejects_source_counts_that_differ_from_frozen_bundle(tmp_path: Path) -> None:
    bundle = _bundle()
    bundle._mapping["R30_source_count"] = 1

    with pytest.raises(P1IssueArtifactError, match="source counts differ"):
        prepare_production_issue_artifacts(
            bundle,
            issue_parent=tmp_path,
            code_commit="a" * 40,
            forecast_created_at_utc=_created(),
        )


def test_frozen_replay_rejects_recomputed_source_count_mismatch(tmp_path: Path) -> None:
    bundle = _bundle()
    prepared = prepare_production_issue_artifacts(
        bundle,
        issue_parent=tmp_path,
        code_commit="a" * 40,
        forecast_created_at_utc=_created(),
    )
    bundle.recent_source_count = 1

    def mismatched_rebuilder(**_: object) -> ProductionForecastBundle:
        return bundle

    with pytest.raises(P1IssueArtifactError, match="cannot reproduce the forecast"):
        verify_prepared_issue_against_frozen_inputs(
            prepared.issue_directory,
            local_catalog_bytes=b"frozen-local-fixture",
            study_area_bytes=b"frozen-study-area-fixture",
            support_manifest_bytes=b"frozen-support-fixture",
            forecast_rebuilder=mismatched_rebuilder,
        )


def test_verification_detects_any_artifact_mutation(tmp_path: Path) -> None:
    prepared = prepare_production_issue_artifacts(
        _bundle(),
        issue_parent=tmp_path,
        code_commit="a" * 40,
        forecast_created_at_utc=_created(),
    )
    forecast_path = prepared.issue_directory / FORECAST_GRID_FILENAME
    forecast_path.write_bytes(forecast_path.read_bytes() + b" ")

    with pytest.raises(P1IssueArtifactError, match="hash mismatch"):
        verify_prepared_issue_artifacts(prepared.issue_directory)


def test_failed_preparation_cleans_only_its_temp_and_never_creates_issue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = operations._write_new_file
    calls = 0

    def fail_after_one(path: Path, payload: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic write failure")
        original(path, payload)

    monkeypatch.setattr(operations, "_write_new_file", fail_after_one)
    with pytest.raises(OSError, match="synthetic write failure"):
        prepare_production_issue_artifacts(
            _bundle(),
            issue_parent=tmp_path,
            code_commit="a" * 40,
            forecast_created_at_utc=_created(),
        )

    assert list(tmp_path.iterdir()) == []


def test_existing_issue_is_untouched_and_record_time_order_fails_closed(tmp_path: Path) -> None:
    issue = tmp_path / "p1-20260909T160000Z"
    issue.mkdir()
    sentinel = issue / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(P1IssueArtifactError, match="already exists"):
        prepare_production_issue_artifacts(
            _bundle(),
            issue_parent=tmp_path,
            code_commit="a" * 40,
            forecast_created_at_utc=_created(),
        )
    assert sentinel.read_text(encoding="utf-8") == "keep"

    other = tmp_path / "other"
    other.mkdir()
    prepared = prepare_production_issue_artifacts(
        _bundle(),
        issue_parent=other,
        code_commit="a" * 40,
        forecast_created_at_utc=_created(),
    )
    verified = verify_prepared_issue_artifacts(prepared.issue_directory)
    with pytest.raises(P1IssueArtifactError, match="Q<=created<=published<=recorded<T"):
        build_forecast_issue_record_fields(
            verified,
            protocol_definition_sha256="b" * 64,
            authorization_record_sha256="c" * 64,
            publication_completed_at_utc=datetime(2026, 9, 9, 15, 50, tzinfo=UTC),
            recorded_at_utc=datetime(2026, 9, 9, 16, tzinfo=UTC),
        )


def test_record_preimage_closes_against_frozen_schema_and_chain(tmp_path: Path) -> None:
    prepared = prepare_production_issue_artifacts(
        _bundle(),
        issue_parent=tmp_path,
        code_commit="a" * 40,
        forecast_created_at_utc=_created(),
    )
    verified = verify_prepared_issue_artifacts(prepared.issue_directory)
    genesis = build_record(
        "ProtocolDefinition",
        recorded_at_utc="2026-08-31T08:00:00Z",
        previous_record=None,
        fields={
            "protocol_id": "p1-b0-r30-prospective-v1",
            "protocol_tag": "v0.2.7-p1-b0-r30-protocol",
            "code_tag": "v0.2.7-p1-b0-r30-code",
            "valid_from_utc": "2026-09-09T16:00:00Z",
            "historical_catalog_cutoff_utc": "2026-07-09T04:25:56Z",
            "source_boundary_manifest_sha256": P1_SOURCE_BOUNDARY_MANIFEST_SHA256,
            "model_manifest_sha256": P1_MODEL_MANIFEST_SHA256,
            "protocol_commit": "d" * 40,
            "real_issue_authorized": False,
        },
    )
    authorization = build_record(
        "RealIssueAuthorizationRecord",
        recorded_at_utc="2026-09-01T08:01:00Z",
        previous_record=genesis,
        fields={
            "protocol_definition_sha256": genesis["content_sha256"],
            "authorization_commit": "e" * 40,
            "code_commit": "a" * 40,
            "remote_verified_at_utc": "2026-09-01T08:00:30Z",
            "authorized_from_scheduled_issue_utc": "2026-09-09T16:00:00Z",
            "real_issue_authorized": True,
        },
    )
    fields = build_forecast_issue_record_fields(
        verified,
        protocol_definition_sha256=cast(str, genesis["content_sha256"]),
        authorization_record_sha256=cast(str, authorization["content_sha256"]),
        publication_completed_at_utc=datetime(2026, 9, 9, 15, 50, tzinfo=UTC),
        recorded_at_utc=datetime(2026, 9, 9, 15, 51, tzinfo=UTC),
    )
    forecast = build_record(
        "ForecastIssueRecord",
        recorded_at_utc="2026-09-09T15:51:00Z",
        previous_record=authorization,
        fields=fields,
    )
    schema = json.loads(Path("data/contracts/p1_prospective_records_v1.json").read_text("utf-8"))

    validate_record_chain([genesis, authorization, forecast], schema)
