"""Actual-report S3 calendars and score-blind counterfactual pool boundaries."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from itertools import groupby

REPORT_START = datetime(2022, 6, 30, 16, tzinfo=UTC)
REPORT_END = datetime(2025, 6, 30, 16, tzinfo=UTC)
HORIZONS = (7, 30, 90, 180, 365)
FOLDS = {
    "A_DEV_2023_2024": (
        datetime(2023, 6, 30, 16, tzinfo=UTC),
        datetime(2024, 6, 30, 16, tzinfo=UTC),
        ("A_I1",),
    ),
    "A_DEV_2024_2025": (
        datetime(2024, 6, 30, 16, tzinfo=UTC),
        REPORT_END,
        ("A_I1", "A_I2", "A_I3"),
    ),
}
INNER_BLOCKS = {
    "A_I1": (datetime(2022, 12, 31, 16, tzinfo=UTC), datetime(2023, 5, 31, 16, tzinfo=UTC)),
    "A_I2": (datetime(2023, 6, 30, 16, tzinfo=UTC), datetime(2023, 12, 31, 16, tzinfo=UTC)),
    "A_I3": (datetime(2023, 12, 31, 16, tzinfo=UTC), datetime(2024, 5, 31, 16, tzinfo=UTC)),
}


def _utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("report and cutoff times must be timezone-aware datetimes")
    return value.astimezone(UTC)


def _reports(values: Iterable[datetime], end: datetime) -> tuple[datetime, ...]:
    issues = tuple(_utc(value) for value in values)
    if len(set(issues)) != len(issues):
        raise ValueError("duplicate actual report issue")
    if any(not REPORT_START <= issue < REPORT_END for issue in issues):
        raise ValueError("reports outside the S3-A development authorization")
    return tuple(sorted(issue for issue in issues if issue < end))


def _greedy(issues: Iterable[datetime], days: int) -> tuple[datetime, ...]:
    selected: list[datetime] = []
    gap = timedelta(days=days)
    for issue in issues:
        if not selected or issue >= selected[-1] + gap:
            selected.append(issue)
    return tuple(selected)


@dataclass(frozen=True, slots=True)
class S3InnerCalendar:
    block_id: str
    label_fit_cutoff: datetime
    training_issues: tuple[datetime, ...]
    validation_issues: tuple[datetime, ...]


@dataclass(frozen=True, slots=True)
class S3FoldCalendar:
    fold_id: str
    horizon_days: int
    report_issues: tuple[datetime, ...]
    label_fit_cutoff: datetime
    training_issues: tuple[datetime, ...]
    evaluation_issues: tuple[datetime, ...]
    primary_evaluation_issues: tuple[datetime, ...]
    inner: tuple[S3InnerCalendar, ...]


def build_fold_calendar(
    report_issues: Iterable[datetime], *, fold_id: str, horizon_days: int, truth_cutoff: datetime
) -> S3FoldCalendar:
    """Select windows by timestamps only, never by earthquakes or model scores.

    Target blocks are right-open: a window whose closed target end equals the
    block end is excluded. Fit embargo and frozen-truth cutoffs are inclusive.
    Labels supplied later must additionally satisfy their availability cutoff.
    """
    if fold_id not in FOLDS or horizon_days not in HORIZONS:
        raise ValueError("unregistered S3-A fold or horizon")
    start, end, inner_ids = FOLDS[fold_id]
    reports = _reports(report_issues, end)
    maturity = _utc(truth_cutoff)
    horizon = timedelta(days=horizon_days)
    fit_cutoff = min(start - timedelta(days=30), maturity)
    training = _greedy((issue for issue in reports if issue + horizon <= fit_cutoff), horizon_days)
    evaluation = tuple(
        issue
        for issue in reports
        if start <= issue and issue + horizon < end and issue + horizon <= maturity
    )
    inner: list[S3InnerCalendar] = []
    for block_id in inner_ids:
        block_start, block_end = INNER_BLOCKS[block_id]
        cutoff = min(block_start - timedelta(days=30), maturity)
        inner.append(
            S3InnerCalendar(
                block_id=block_id,
                label_fit_cutoff=cutoff,
                training_issues=_greedy(
                    (issue for issue in reports if issue + horizon <= cutoff), horizon_days
                ),
                validation_issues=_greedy(
                    (
                        issue
                        for issue in reports
                        if block_start <= issue
                        and issue + horizon < block_end
                        and issue + horizon <= fit_cutoff
                    ),
                    horizon_days + 30,
                ),
            )
        )
    return S3FoldCalendar(
        fold_id,
        horizon_days,
        reports,
        fit_cutoff,
        training,
        evaluation,
        _greedy(evaluation, horizon_days + 30),
        tuple(inner),
    )


def time_null_partitions(
    report_issues: Iterable[datetime], *, fold_id: str, horizon_days: int, truth_cutoff: datetime
) -> tuple[tuple[datetime, ...], ...]:
    """Counterfactual donor pools cannot cross any inner/outer eligibility cut.

    A membership signature encodes both calendar boundaries and the actual last
    issue whose full target window may train or validate. Inclusive training
    endpoints stay on the training side. This does NOT make future donors causal.
    """
    calendar = build_fold_calendar(
        report_issues, fold_id=fold_id, horizon_days=horizon_days, truth_cutoff=truth_cutoff
    )
    start, end, inner_ids = FOLDS[fold_id]
    horizon = timedelta(days=horizon_days)
    maturity = _utc(truth_cutoff)

    def signature(issue: datetime) -> tuple[bool, ...]:
        values = [
            issue < start,
            issue < end,
            issue + horizon <= calendar.label_fit_cutoff,
            issue + horizon < end,
            issue + horizon <= maturity,
        ]
        for block_id in inner_ids:
            block_start, block_end = INNER_BLOCKS[block_id]
            values.extend(
                [
                    issue < block_start,
                    issue < block_end,
                    issue + horizon <= min(block_start - timedelta(days=30), maturity),
                    issue + horizon < block_end,
                ]
            )
        return tuple(values)

    return tuple(tuple(group) for _, group in groupby(calendar.report_issues, signature))
