"""Calendar-only and synthetic-catalog tests for Stage 2S role membership."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from seismoflux.data.contracts import CONTRACTS
from seismoflux.data.parquet import (
    schema_sha256,
    table_content_sha256,
    table_from_records,
)
from seismoflux.stage2s.calendar import (
    assessment_target_memberships,
    causal_source_membership,
    fit_target_membership,
    parse_fold_manifest_bytes,
    parse_frozen_fold_manifest_bytes,
)
from seismoflux.stage2s.catalog import (
    ArrowFieldContract,
    CatalogByteContract,
    Stage2SEarthquakeCatalog,
    parse_catalog_bytes,
)
from seismoflux.stage2s.seals import write_o_excl_record

ROOT = Path(__file__).resolve().parents[2]
FOLD_MANIFEST = ROOT / "data" / "manifests" / "causal_seismicity_screen_fold_manifest.json"
_LOCAL_OFFSET = timezone(timedelta(hours=8))


def _utc_text(local_date: date) -> str:
    issue = datetime.combine(local_date, datetime.min.time(), _LOCAL_OFFSET).astimezone(UTC)
    return issue.strftime("%Y-%m-%dT%H:%M:%SZ")


def _fold(
    fold_index: int,
    *,
    fit_dates: tuple[date, ...],
    assessment_start: date,
) -> dict[str, object]:
    assessment_end = assessment_start + timedelta(days=90)
    return {
        "fold_index": fold_index,
        "fit_scope_id": f"stage2s-development-fold-{fold_index}",
        "fit_issue_dates_local_h007": [value.isoformat() for value in fit_dates],
        "fit_target_end_inclusive_utc": _utc_text(fit_dates[-1] + timedelta(days=7)),
        "assessment_band": {
            "start_exclusive_local": assessment_start.isoformat(),
            "start_exclusive_utc": _utc_text(assessment_start),
            "end_inclusive_local": assessment_end.isoformat(),
            "end_inclusive_utc": _utc_text(assessment_end),
        },
        "assessment_issue_dates_local_by_horizon": {
            "7": [
                assessment_start.isoformat(),
                (assessment_start + timedelta(days=7)).isoformat(),
            ],
            "30": [assessment_start.isoformat()],
            "90": [assessment_start.isoformat()],
        },
    }


def _manifest_mapping() -> dict[str, object]:
    first_fit = (date(2022, 1, 1), date(2022, 1, 8))
    second_fit = (*first_fit, date(2022, 1, 15))
    third_fit = (*second_fit, date(2022, 1, 22))
    return {
        "schema_version": 1,
        "protocol_version": "0.2.3",
        "experiment_id": "stage2s-causal-seismicity-development-v1",
        "status": "target_blind_calendar_only_no_execution_or_scoring_authority",
        "source_design": {
            "path": "synthetic-source-design.json",
            "file_sha256": "1" * 64,
            "content_sha256": "2" * 64,
            "allowed_pointers": [
                "/joint_macro_rolling_folds",
                "/target_window_rule",
                "/training_target_end_must_be_strictly_before_assessment_target_start",
            ],
            "inherited_anomaly_ids_pools_execution_attempt_randomness_or_results": False,
        },
        "issue_semantics": {
            "timezone": "Asia/Shanghai",
            "local_time": "00:00:00",
            "utc_offset": "+08:00",
            "target_window": "(T,T+h]",
            "fit_horizon_days": 7,
            "assessment_horizons_days": [7, 30, 90],
            "training_target_end_strictly_before_assessment_target_start": True,
            "random_split": False,
        },
        "target_bands_mutually_disjoint": True,
        "rolling_rule": "three_disjoint_90d_target_bands_with_expanding_nonoverlapping_h007_fit",
        "folds": [
            _fold(1, fit_dates=first_fit, assessment_start=date(2022, 2, 1)),
            _fold(2, fit_dates=second_fit, assessment_start=date(2022, 6, 1)),
            _fold(3, fit_dates=third_fit, assessment_start=date(2022, 10, 1)),
        ],
        "security": {
            "contains_target_ids_coordinates_scores_hits_or_model_results": False,
            "contains_anomaly_values_features_or_entity_ids": False,
            "development_target_read_authorized": False,
            "independent_validation_or_locked_test_authorized": False,
        },
    }


def _manifest_bytes(mapping: dict[str, object] | None = None) -> tuple[bytes, str]:
    payload = json.dumps(
        _manifest_mapping() if mapping is None else mapping,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return payload, hashlib.sha256(payload).hexdigest()


def _record(
    event_id: str,
    origin: datetime,
    *,
    available_at: datetime | None = None,
    magnitude: float = 5.2,
    inside: bool = True,
) -> dict[str, object]:
    available = origin if available_at is None else available_at
    return {
        "event_id": event_id,
        "origin_time_utc": origin,
        "available_at": available,
        "origin_time_local": origin.astimezone(_LOCAL_OFFSET),
        "longitude": 105.0,
        "latitude": 35.0,
        "depth_km": 10.0,
        "magnitude": magnitude,
        "magnitude_type": "M",
        "place": "synthetic",
        "catalog_sources": ["synthetic"],
        "inside_study_area": inside,
        "dedup_confidence": "exact",
        "anchor_source_record_id": f"source-{event_id}",
        "quality_flags": [],
    }


def _catalog(records: list[dict[str, object]]) -> Stage2SEarthquakeCatalog:
    stage1 = CONTRACTS["earthquake_event"]
    table = table_from_records(records, stage1.schema, stage1.sort_keys)
    sink = pa.BufferOutputStream()
    pq.write_table(
        table,
        sink,
        compression="zstd",
        compression_level=9,
        version="2.6",
        data_page_version="1.0",
        use_dictionary=False,
        write_statistics=True,
        coerce_timestamps="us",
        allow_truncated_timestamps=False,
    )
    payload = sink.getvalue().to_pybytes()
    persisted = pq.read_table(pa.BufferReader(payload), use_threads=False)
    contract = CatalogByteContract(
        row_count=persisted.num_rows,
        file_sha256=hashlib.sha256(payload).hexdigest(),
        content_sha256=table_content_sha256(persisted),
        schema_sha256=schema_sha256(persisted.schema),
        fields=tuple(ArrowFieldContract.from_arrow(field) for field in persisted.schema),
    )
    return parse_catalog_bytes(payload, contract=contract)


def test_actual_frozen_manifest_parses_to_strict_1_2_3_calendar() -> None:
    calendar = parse_frozen_fold_manifest_bytes(FOLD_MANIFEST.read_bytes())
    assert tuple(fold.fold_index for fold in calendar.folds) == (1, 2, 3)
    assert tuple(len(fold.fit_exposures) for fold in calendar.folds) == (12, 25, 38)
    assert tuple(len(fold.assessment_issues) for fold in calendar.folds) == (12, 12, 11)
    assert tuple(len(fold.assessment_exposures) for fold in calendar.folds) == (15, 14, 14)
    assert calendar.folds[0].assessment_issues[0].horizons_days == (7, 90)


def test_synthetic_manifest_groups_horizons_and_preserves_expanding_fit_prefixes() -> None:
    payload, digest = _manifest_bytes()
    calendar = parse_fold_manifest_bytes(payload, expected_sha256=digest)
    assert tuple(fold.fold_index for fold in calendar.folds) == (1, 2, 3)
    assert tuple(len(fold.fit_exposures) for fold in calendar.folds) == (2, 3, 4)
    assert calendar.folds[0].assessment_issues[0].horizons_days == (7, 30, 90)
    assert calendar.folds[0].assessment_issues[1].horizons_days == (7,)
    assert calendar.folds[0].fit_exposures[0].target_end_inclusive_utc == (
        calendar.folds[0].fit_exposures[1].issue_time_utc
    )


def test_manifest_hash_and_half_open_calendar_semantics_fail_closed() -> None:
    payload, digest = _manifest_bytes()
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        parse_fold_manifest_bytes(payload, expected_sha256="0" * 64)

    changed = _manifest_mapping()
    issue_semantics = changed["issue_semantics"]
    assert isinstance(issue_semantics, dict)
    issue_semantics["target_window"] = "[T,T+h]"
    changed_payload, changed_digest = _manifest_bytes(changed)
    with pytest.raises(ValueError, match="target_window mismatch"):
        parse_fold_manifest_bytes(changed_payload, expected_sha256=changed_digest)

    overlapping = _manifest_mapping()
    folds = overlapping["folds"]
    assert isinstance(folds, list)
    first = folds[0]
    assert isinstance(first, dict)
    by_horizon = first["assessment_issue_dates_local_by_horizon"]
    assert isinstance(by_horizon, dict)
    by_horizon["7"] = ["2022-02-01", "2022-02-07"]
    overlap_payload, overlap_digest = _manifest_bytes(overlapping)
    with pytest.raises(ValueError, match="disjoint targets"):
        parse_fold_manifest_bytes(overlap_payload, expected_sha256=overlap_digest)


def test_causal_r_and_rp_memberships_use_exact_origin_and_availability_boundaries() -> None:
    issue = datetime(2022, 4, 1, tzinfo=UTC)
    catalog = _catalog(
        [
            _record("rp-lower-excluded", issue - timedelta(days=60), magnitude=4.0),
            _record(
                "rp-inside",
                issue - timedelta(days=60) + timedelta(microseconds=1),
                magnitude=4.0,
            ),
            _record("rp-upper-r-lower", issue - timedelta(days=30), magnitude=4.0),
            _record(
                "r-inside",
                issue - timedelta(days=30) + timedelta(microseconds=1),
                magnitude=4.0,
            ),
            _record("r-upper", issue, magnitude=4.0),
            _record(
                "r-delayed",
                issue - timedelta(days=1),
                available_at=issue + timedelta(microseconds=1),
                magnitude=4.0,
            ),
            _record("below-m4", issue - timedelta(days=1), magnitude=3.999),
            _record(
                "outside",
                issue - timedelta(days=1),
                magnitude=4.0,
                inside=False,
            ),
        ]
    )
    recent = causal_source_membership(catalog, issue_time_utc=issue, component_id="R")
    preceding = causal_source_membership(catalog, issue_time_utc=issue, component_id="RP")
    assert recent.event_ids == ("r-inside", "r-upper")
    assert preceding.event_ids == ("rp-inside", "rp-upper-r-lower")
    assert recent.catalog is catalog
    assert preceding.catalog is catalog
    assert recent.additional_delay_days == 0
    assert preceding.additional_delay_days == 0
    assert not recent.event_indices.flags.writeable


def test_causal_membership_delay_only_moves_the_availability_cutoff() -> None:
    issue = datetime(2022, 4, 1, tzinfo=UTC)
    catalog = _catalog(
        [
            _record(
                "rp-delay7-cutoff-inclusive",
                issue - timedelta(days=45),
                available_at=issue - timedelta(days=37),
                magnitude=4.0,
            ),
            _record(
                "rp-after-delay7-cutoff",
                issue - timedelta(days=45),
                available_at=issue - timedelta(days=37) + timedelta(microseconds=1),
                magnitude=4.0,
            ),
            _record(
                "rp-delay1-cutoff-inclusive",
                issue - timedelta(days=31),
                available_at=issue - timedelta(days=31),
                magnitude=4.0,
            ),
            _record(
                "rp-origin-upper-fixed",
                issue - timedelta(days=30),
                available_at=issue - timedelta(days=30),
                magnitude=4.0,
            ),
            _record(
                "r-delay7-cutoff-inclusive",
                issue - timedelta(days=10),
                available_at=issue - timedelta(days=7),
                magnitude=4.0,
            ),
            _record(
                "r-after-delay7-cutoff",
                issue - timedelta(days=10),
                available_at=issue - timedelta(days=7) + timedelta(microseconds=1),
                magnitude=4.0,
            ),
            _record(
                "r-delay1-cutoff-inclusive",
                issue - timedelta(days=2),
                available_at=issue - timedelta(days=1),
                magnitude=4.0,
            ),
            _record(
                "r-origin-upper-fixed",
                issue,
                available_at=issue,
                magnitude=4.0,
            ),
        ]
    )

    recent_delay_1 = causal_source_membership(
        catalog,
        issue_time_utc=issue,
        component_id="R",
        additional_delay_days=1,
    )
    recent_delay_7 = causal_source_membership(
        catalog,
        issue_time_utc=issue,
        component_id="R",
        additional_delay_days=7,
    )
    preceding_delay_1 = causal_source_membership(
        catalog,
        issue_time_utc=issue,
        component_id="RP",
        additional_delay_days=1,
    )
    preceding_delay_7 = causal_source_membership(
        catalog,
        issue_time_utc=issue,
        component_id="RP",
        additional_delay_days=7,
    )

    assert recent_delay_1.event_ids == (
        "r-after-delay7-cutoff",
        "r-delay7-cutoff-inclusive",
        "r-delay1-cutoff-inclusive",
    )
    assert recent_delay_7.event_ids == ("r-delay7-cutoff-inclusive",)
    assert preceding_delay_1.event_ids == (
        "rp-after-delay7-cutoff",
        "rp-delay7-cutoff-inclusive",
        "rp-delay1-cutoff-inclusive",
    )
    assert preceding_delay_7.event_ids == ("rp-delay7-cutoff-inclusive",)
    assert recent_delay_1.additional_delay_days == 1
    assert recent_delay_7.additional_delay_days == 7
    assert preceding_delay_1.additional_delay_days == 1
    assert preceding_delay_7.additional_delay_days == 7


@pytest.mark.parametrize("delay", [-1, 2, 8, True, 1.0, "1"])
def test_causal_membership_rejects_non_frozen_delay_values(delay: object) -> None:
    issue = datetime(2022, 4, 1, tzinfo=UTC)
    catalog = _catalog([_record("event", issue - timedelta(days=2), magnitude=4.0)])
    with pytest.raises(ValueError, match="exactly 0, 1, or 7"):
        causal_source_membership(
            catalog,
            issue_time_utc=issue,
            component_id="R",
            additional_delay_days=delay,  # type: ignore[arg-type]
        )


def test_fit_membership_assigns_each_event_once_with_t_t_plus_h_closed_end() -> None:
    payload, digest = _manifest_bytes()
    fold = parse_fold_manifest_bytes(payload, expected_sha256=digest).fold(1)
    first_issue = fold.fit_exposures[0].issue_time_utc
    boundary = fold.fit_exposures[1].issue_time_utc
    fit_end = fold.fit_target_end_inclusive_utc
    catalog = _catalog(
        [
            _record("at-first-start-excluded", first_issue, magnitude=5.0),
            _record(
                "inside-first",
                first_issue + timedelta(microseconds=1),
                magnitude=5.0,
            ),
            _record("shared-boundary", boundary, magnitude=5.0),
            _record("fit-end-inclusive", fit_end, magnitude=5.999),
            _record(
                "reported-after-fit-cutoff",
                fit_end - timedelta(days=1),
                available_at=fit_end + timedelta(microseconds=1),
                magnitude=5.2,
            ),
            _record("m6-excluded", fit_end - timedelta(days=1), magnitude=6.0),
        ]
    )
    membership = fit_target_membership(catalog, fold)
    assert membership.event_ids == (
        "inside-first",
        "shared-boundary",
        "fit-end-inclusive",
    )
    assigned = dict(zip(membership.event_ids, membership.assigned_issue_time_us, strict=True))
    assert assigned["inside-first"] == fold.fit_exposures[0].issue_time_us
    assert assigned["shared-boundary"] == fold.fit_exposures[0].issue_time_us
    assert assigned["fit-end-inclusive"] == fold.fit_exposures[1].issue_time_us
    assert membership.exposure_days == 14.0
    assert membership.catalog is catalog


def test_assessment_membership_requires_master_seal_and_is_unique_per_horizon(
    tmp_path: Path,
) -> None:
    payload, digest = _manifest_bytes()
    calendar = parse_fold_manifest_bytes(payload, expected_sha256=digest)
    fold = calendar.fold(1)
    issue = fold.assessment_start_exclusive_utc
    catalog = _catalog(
        [
            _record("at-start-excluded", issue, magnitude=5.0),
            _record("inside-first", issue + timedelta(days=1), magnitude=5.0),
            _record("shared-seven-day-boundary", issue + timedelta(days=7), magnitude=5.1),
            _record("inside-second-seven", issue + timedelta(days=8), magnitude=5.2),
            _record("thirty-day-end", issue + timedelta(days=30), magnitude=5.3),
            _record(
                "after-thirty-before-ninety",
                issue + timedelta(days=30, microseconds=1),
                magnitude=5.4,
            ),
            _record("ninety-day-end", issue + timedelta(days=90), magnitude=5.9),
            _record(
                "after-ninety",
                issue + timedelta(days=90, microseconds=1),
                magnitude=5.5,
            ),
            _record("m6-excluded", issue + timedelta(days=1), magnitude=6.0),
            _record(
                "outside-excluded",
                issue + timedelta(days=1),
                magnitude=5.2,
                inside=False,
            ),
        ]
    )
    wrong_record = write_o_excl_record(
        tmp_path / "issue.json",
        record_type="stage2s_issue_prediction_seal",
        bindings={},
    )
    with pytest.raises(ValueError, match="master prediction seal"):
        assessment_target_memberships(catalog, calendar, master_seal=wrong_record)

    master = write_o_excl_record(
        tmp_path / "prediction_seal.json",
        record_type="stage2s_master_prediction_seal",
        bindings={
            "assessment_target_role_or_score_exposed_before_master_seal": False,
        },
    )
    memberships = assessment_target_memberships(
        catalog,
        calendar,
        master_seal=master,
    )
    assert len(memberships) == 9
    fold1_h7 = next(item for item in memberships if item.fold_index == 1 and item.horizon_days == 7)
    fold1_h30 = next(
        item for item in memberships if item.fold_index == 1 and item.horizon_days == 30
    )
    fold1_h90 = next(
        item for item in memberships if item.fold_index == 1 and item.horizon_days == 90
    )
    assert fold1_h7.event_ids == (
        "inside-first",
        "shared-seven-day-boundary",
        "inside-second-seven",
    )
    h7_assignment = dict(zip(fold1_h7.event_ids, fold1_h7.assigned_issue_time_us, strict=True))
    assert h7_assignment["shared-seven-day-boundary"] == fold.assessment_exposures[0].issue_time_us
    assert fold1_h30.event_ids == (
        "inside-first",
        "shared-seven-day-boundary",
        "inside-second-seven",
        "thirty-day-end",
    )
    assert fold1_h90.event_ids == (
        "inside-first",
        "shared-seven-day-boundary",
        "inside-second-seven",
        "thirty-day-end",
        "after-thirty-before-ninety",
        "ninety-day-end",
    )
    for membership in memberships:
        assert membership.catalog is catalog
        assert membership.master_seal_file_sha256 == master.file_sha256
        assert len(membership.event_ids) == len(set(membership.event_ids))
        assert not membership.event_indices.flags.writeable

    forged = replace(master, file_sha256="0" * 64)
    with pytest.raises(ValueError, match="file_sha256"):
        assessment_target_memberships(catalog, calendar, master_seal=forged)
