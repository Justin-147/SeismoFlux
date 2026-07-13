from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from datetime import UTC, date, datetime
from typing import Any

import pytest

from seismoflux.data.common import fixed_local_midnight
from seismoflux.data.quality import build_quality_report, count_leakage_violations


def _valid_observation() -> dict[str, Any]:
    return {
        "report_date": date(2026, 7, 1),
        "raw_row_report_date": date(2026, 7, 1),
        # 2026-07-02 00:00 at fixed UTC+08 is 2026-07-01 16:00 UTC.
        "available_at": datetime(2026, 7, 1, 16, tzinfo=UTC),
        "start_time": fixed_local_midnight(date(2026, 6, 1)),
        "reported_end_time": None,
        "right_censored": True,
        "reliability_flags": [
            "anomaly_time_date_only_assumed_fixed_utc_plus_08_midnight",
            "manual_prediction_fields_excluded",
        ],
    }


def _valid_earthquake() -> dict[str, Any]:
    origin_time = datetime(2026, 7, 1, 8, tzinfo=UTC)
    return {
        "origin_time_utc": origin_time,
        "available_at": origin_time,
        "quality_flags": ["publication_time_assumed_origin_time"],
    }


def test_standardized_boundary_proves_zero_future_information_leakage() -> None:
    observations = [_valid_observation()]
    geology = [
        {"historical_model_eligible": False},
        {"historical_model_eligible": False},
    ]

    violations, checks = count_leakage_violations(
        observations,
        geology,
        [_valid_earthquake()],
        [{"model_feature_eligible": False}],
    )

    assert violations == 0
    assert checks == {
        "anomaly_available_at_rule": 0,
        "anomaly_end_or_manual_field_backfill": 0,
        "anomaly_occurrence_time_precision": 0,
        "earthquake_available_at_assumption": 0,
        "historical_geology_early_use": 0,
        "basemap_model_feature_use": 0,
    }


@pytest.mark.parametrize(
    ("mutate", "expected_check"),
    [
        (
            lambda record: record.__setitem__(
                "available_at", datetime(2026, 7, 1, 15, 59, 59, tzinfo=UTC)
            ),
            "anomaly_available_at_rule",
        ),
        (
            lambda record: record.__setitem__(
                "available_at", datetime(2026, 7, 1, 16, 0, 0, 1, tzinfo=UTC)
            ),
            "anomaly_available_at_rule",
        ),
        (
            lambda record: record.__setitem__("start_time", datetime(2026, 6, 1, 0, tzinfo=UTC)),
            "anomaly_occurrence_time_precision",
        ),
        (
            lambda record: record.__setitem__("right_censored", False),
            "anomaly_end_or_manual_field_backfill",
        ),
        (
            lambda record: record.__setitem__(
                "reported_end_time", fixed_local_midnight(date(2026, 7, 20))
            ),
            "anomaly_end_or_manual_field_backfill",
        ),
        (
            lambda record: record.__setitem__(
                "reliability_flags",
                ["anomaly_time_date_only_assumed_fixed_utc_plus_08_midnight"],
            ),
            "anomaly_end_or_manual_field_backfill",
        ),
    ],
)
def test_each_deliberate_anomaly_violation_is_counted(
    mutate: Callable[[dict[str, Any]], None], expected_check: str
) -> None:
    record = deepcopy(_valid_observation())
    mutate(record)

    violations, checks = count_leakage_violations([record], [])

    assert violations == 1
    assert checks[expected_check] == 1
    assert sum(checks.values()) == 1


def test_historical_geology_early_use_is_a_blocking_leakage_violation() -> None:
    violations, checks = count_leakage_violations(
        [_valid_observation()],
        [
            {"historical_model_eligible": False},
            {"historical_model_eligible": True},
        ],
    )

    assert violations == 1
    assert checks["historical_geology_early_use"] == 1

    report = build_quality_report(
        source_inventory_sha256="a" * 64,
        anomaly_quality={},
        earthquake_quality={},
        geology_quality={},
        leakage_checks=checks,
        leakage_violations=violations,
    )
    assert report["status"] == "fail"
    assert report["blocking_errors"] == ["future_information_leakage"]


@pytest.mark.parametrize("mutation", ["available_at", "flag"])
def test_earthquake_publication_assumption_is_enforced(mutation: str) -> None:
    earthquake = _valid_earthquake()
    if mutation == "available_at":
        earthquake["available_at"] = datetime(2026, 7, 1, 8, 0, 1, tzinfo=UTC)
    else:
        earthquake["quality_flags"] = []

    violations, checks = count_leakage_violations([], [], [earthquake], [])

    assert violations == 1
    assert checks["earthquake_available_at_assumption"] == 1


def test_basemap_cannot_silently_become_a_model_feature() -> None:
    violations, checks = count_leakage_violations([], [], [], [{"model_feature_eligible": True}])

    assert violations == 1
    assert checks["basemap_model_feature_use"] == 1
