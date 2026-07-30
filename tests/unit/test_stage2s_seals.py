from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from seismoflux.stage2s.seals import (
    RoleSession,
    Stage2SSealError,
    Stage2SSealExists,
    write_o_excl_record,
)


def _row(
    event_id: str,
    *,
    origin: str = "2022-01-01T00:00:00Z",
    available: str | None = None,
) -> dict[str, object]:
    return {
        "event_id": event_id,
        "origin_time_utc": origin,
        "available_at": origin if available is None else available,
        "longitude": 110.0,
        "latitude": 35.0,
        "magnitude": 5.2,
        "inside_study_area": True,
    }


def _session(tmp_path: Path) -> RoleSession:
    return RoleSession(
        seal_root=(tmp_path / "seals").resolve(),
        issue_dates_by_fold={
            1: ("2022-01-10", "2022-01-17"),
            2: ("2022-02-10",),
            3: ("2022-03-10",),
        },
    )


def test_o_excl_record_is_canonical_immutable_and_not_rewritten(tmp_path: Path) -> None:
    path = (tmp_path / "record.json").resolve()
    record = write_o_excl_record(
        path,
        record_type="synthetic",
        bindings={"fold": 1, "value": 0.5},
    )
    original = path.read_bytes()

    assert original.endswith(b"\n")
    assert hashlib.sha256(original).hexdigest() == record.file_sha256
    assert record.payload["content_sha256"] == record.content_sha256
    with pytest.raises(Stage2SSealExists):
        write_o_excl_record(
            path,
            record_type="synthetic",
            bindings={"fold": 2, "value": 0.7},
        )
    assert path.read_bytes() == original


def test_fit_view_requires_raw_allowlist_and_prior_fold_seal(tmp_path: Path) -> None:
    session = _session(tmp_path)
    contaminated = _row("e1") | {"hit": True}

    with pytest.raises(Stage2SSealError, match="raw allowlist"):
        session.open_fit_view(
            fold_index=1,
            rows=[contaminated],
            fit_cutoff_utc="2022-01-05T00:00:00Z",
        )
    with pytest.raises(Stage2SSealError, match="prior fold"):
        session.open_fit_view(
            fold_index=2,
            rows=[_row("e1")],
            fit_cutoff_utc="2022-02-05T00:00:00Z",
        )


def test_assessment_and_issue_order_fail_closed(tmp_path: Path) -> None:
    session = _session(tmp_path)
    with pytest.raises(Stage2SSealError, match="master"):
        session.open_assessment_view(rows=[_row("e1")])

    session.open_fit_view(
        fold_index=1,
        rows=[_row("fit")],
        fit_cutoff_utc="2022-01-05T00:00:00Z",
    )
    session.seal_fit(fold_index=1, bindings={"alpha_R": 0.25, "rate": 0.1})
    with pytest.raises(Stage2SSealError, match="frozen ascending order"):
        session.open_causal_source_view(
            fold_index=1,
            issue_date="2022-01-17",
            issue_time_utc="2022-01-17T00:00:00Z",
            rows=[_row("history")],
        )
    session.open_causal_source_view(
        fold_index=1,
        issue_date="2022-01-10",
        issue_time_utc="2022-01-10T00:00:00Z",
        rows=[_row("history")],
    )
    with pytest.raises(Stage2SSealError, match="prior issue"):
        session.open_causal_source_view(
            fold_index=1,
            issue_date="2022-01-17",
            issue_time_utc="2022-01-17T00:00:00Z",
            rows=[_row("history")],
        )
    with pytest.raises(Stage2SSealError, match="open issue"):
        session.seal_fold(fold_index=1, bindings={"grid_id": "g"})


def test_complete_three_fold_chain_opens_assessment_once(tmp_path: Path) -> None:
    session = _session(tmp_path)
    schedules = {
        1: ("2022-01-10", "2022-01-17"),
        2: ("2022-02-10",),
        3: ("2022-03-10",),
    }
    cutoffs = {
        1: "2022-01-05T00:00:00Z",
        2: "2022-02-05T00:00:00Z",
        3: "2022-03-05T00:00:00Z",
    }
    for fold_index in (1, 2, 3):
        fit_rows = session.open_fit_view(
            fold_index=fold_index,
            rows=[
                _row(
                    f"fit-{fold_index}",
                    origin=f"2022-0{fold_index}-01T00:00:00Z",
                )
            ],
            fit_cutoff_utc=cutoffs[fold_index],
        )
        assert tuple(row.event_id for row in fit_rows) == (f"fit-{fold_index}",)
        fit = session.seal_fit(
            fold_index=fold_index,
            bindings={"fit_event_ids": [row.event_id for row in fit_rows]},
        )
        for issue_date in schedules[fold_index]:
            source = session.open_causal_source_view(
                fold_index=fold_index,
                issue_date=issue_date,
                issue_time_utc=f"{issue_date}T00:00:00Z",
                rows=[_row("shared-history")],
            )
            assert tuple(row.event_id for row in source) == ("shared-history",)
            issue = session.seal_issue(
                fold_index=fold_index,
                issue_date=issue_date,
                bindings={"models": ["S0", "S1", "SP"], "horizons": [7, 30, 90]},
            )
            assert issue.payload["record_type"] == "stage2s_issue_prediction_seal"
        fold = session.seal_fold(
            fold_index=fold_index,
            bindings={"fit_receipt_sha256": fit.file_sha256},
        )
        assert fold.payload["record_type"] == "stage2s_fold_prediction_seal"

    master = session.seal_master(
        bindings={
            "attempt_ledger_sha256": "a" * 64,
            "target_read_receipt_sha256": "b" * 64,
        }
    )
    assessment = session.open_assessment_view(rows=[_row("target")])

    assert master.payload["record_type"] == "stage2s_master_prediction_seal"
    assert tuple(row.event_id for row in assessment) == ("target",)
    with pytest.raises(Stage2SSealError, match="only once"):
        session.open_assessment_view(rows=[_row("target")])


def test_prior_assessment_row_can_only_reenter_as_raw_later_fit_data(tmp_path: Path) -> None:
    session = _session(tmp_path)
    session.open_fit_view(
        fold_index=1,
        rows=[_row("base")],
        fit_cutoff_utc="2022-01-05T00:00:00Z",
    )
    session.seal_fit(fold_index=1, bindings={"fit": 1})
    for issue_date in ("2022-01-10", "2022-01-17"):
        session.open_causal_source_view(
            fold_index=1,
            issue_date=issue_date,
            issue_time_utc=f"{issue_date}T00:00:00Z",
            rows=[_row("base")],
        )
        session.seal_issue(
            fold_index=1,
            issue_date=issue_date,
            bindings={"prediction": issue_date},
        )
    session.seal_fold(fold_index=1, bindings={"fold": 1})

    later_raw = _row("earlier-target", origin="2022-01-12T00:00:00Z")
    view = session.open_fit_view(
        fold_index=2,
        rows=[later_raw],
        fit_cutoff_utc="2022-02-05T00:00:00Z",
    )
    assert tuple(row.event_id for row in view) == ("earlier-target",)

    contaminated = later_raw | {"prior_hit": True}
    other = _session(tmp_path / "other")
    with pytest.raises(Stage2SSealError, match="raw allowlist"):
        other.open_fit_view(
            fold_index=1,
            rows=[contaminated],
            fit_cutoff_utc="2022-02-05T00:00:00Z",
        )
