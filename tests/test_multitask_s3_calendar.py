"""Synthetic actual-report dates; no real reports, labels, or scores are read."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from seismoflux.multitask_s3.calendar import (
    FOLDS,
    REPORT_END,
    REPORT_START,
    build_fold_calendar,
    time_null_partitions,
)


def _weekly():
    return tuple(REPORT_START + timedelta(days=19 + 7 * i) for i in range(153))


@pytest.mark.parametrize("horizon", [7, 30, 90, 180, 365])
@pytest.mark.parametrize("fold", ["A_DEV_2023_2024", "A_DEV_2024_2025"])
def test_calendar_maturity_embargo_and_nonoverlap(fold, horizon):
    c = build_fold_calendar(_weekly(), fold_id=fold, horizon_days=horizon, truth_cutoff=REPORT_END)
    start, end, _ = FOLDS[fold]
    assert all(t + timedelta(days=horizon) <= start - timedelta(days=30) for t in c.training_issues)
    assert all(start <= t and t + timedelta(days=horizon) < end for t in c.evaluation_issues)
    assert all(
        b - a >= timedelta(days=horizon)
        for a, b in zip(c.training_issues, c.training_issues[1:], strict=False)
    )
    assert all(
        b - a >= timedelta(days=horizon + 30)
        for a, b in zip(c.primary_evaluation_issues, c.primary_evaluation_issues[1:], strict=False)
    )
    assert set(c.primary_evaluation_issues) <= set(c.evaluation_issues)
    for inner in c.inner:
        assert all(
            t + timedelta(days=horizon) <= inner.label_fit_cutoff for t in inner.training_issues
        )


def test_no_fabricated_report_or_shortened_long_window():
    only_report = REPORT_START + timedelta(days=19)
    c = build_fold_calendar(
        [only_report], fold_id="A_DEV_2023_2024", horizon_days=365, truth_cutoff=REPORT_END
    )
    assert not c.training_issues
    assert not c.evaluation_issues
    assert c.report_issues == (only_report,)


def test_endpoint_training_inclusive_target_block_exclusive():
    start, end, _ = FOLDS["A_DEV_2023_2024"]
    train_last = start - timedelta(days=60)
    issues = [train_last, train_last + timedelta(microseconds=1), end - timedelta(days=30)]
    c = build_fold_calendar(
        issues, fold_id="A_DEV_2023_2024", horizon_days=30, truth_cutoff=REPORT_END
    )
    assert c.training_issues == (train_last,)
    assert not c.evaluation_issues
    pools = time_null_partitions(
        issues, fold_id=c.fold_id, horizon_days=30, truth_cutoff=REPORT_END
    )
    assert not any(train_last in pool and issues[1] in pool for pool in pools)


def test_null_pools_never_mix_actual_inner_training_sides():
    c = build_fold_calendar(
        _weekly(), fold_id="A_DEV_2024_2025", horizon_days=90, truth_cutoff=REPORT_END
    )
    pools = time_null_partitions(
        _weekly(), fold_id=c.fold_id, horizon_days=90, truth_cutoff=REPORT_END
    )
    assert tuple(t for pool in pools for t in pool) == c.report_issues
    for pool in pools:
        for inner in c.inner:
            flags = {t + timedelta(days=90) <= inner.label_fit_cutoff for t in pool}
            assert len(flags) == 1


def test_reject_unregistered_or_naive_dates_before_any_target_access():
    with pytest.raises(ValueError, match="authorization"):
        build_fold_calendar(
            [REPORT_END], fold_id="A_DEV_2024_2025", horizon_days=7, truth_cutoff=REPORT_END
        )
    with pytest.raises(ValueError, match="timezone-aware"):
        build_fold_calendar(
            [datetime(2023, 1, 1)],
            fold_id="A_DEV_2024_2025",
            horizon_days=7,
            truth_cutoff=REPORT_END,
        )
    with pytest.raises(ValueError, match="unregistered"):
        build_fold_calendar(
            [], fold_id="A_AUDIT_2025_2026", horizon_days=7, truth_cutoff=REPORT_END
        )
