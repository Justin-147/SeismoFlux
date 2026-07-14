from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest

from seismoflux.background.config import load_background_protocol
from seismoflux.background.issues import load_frozen_issue_calendar

BACKGROUND_CONFIG = Path("configs/background.yaml")
ISSUE_MANIFEST = Path("data/manifests/background_fold_manifest.json")


def _manifest() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(ISSUE_MANIFEST.read_text(encoding="utf-8")))


def _write_manifest(tmp_path: Path, value: dict[str, Any]) -> Path:
    destination = tmp_path / "background_fold_manifest.json"
    destination.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return destination


def test_frozen_issue_calendar_loads_all_dates_and_greedy_exposures() -> None:
    config = load_background_protocol(BACKGROUND_CONFIG)
    calendar = load_frozen_issue_calendar(ISSUE_MANIFEST, config=config)

    assert len(calendar.development.actual_issue_dates_local) == 50
    assert len(calendar.validation.actual_issue_dates_local) == 51
    assert tuple(
        len(calendar.validation.exposures(horizon)) for horizon in config.time.horizons_days
    ) == (51, 11, 4, 2, 1)
    assert calendar.validation.exposures(365)[0].issue_date_local == "2024-07-04"
    assert calendar.validation.exposures(365)[0].end_day == pytest.approx(
        calendar.validation.exposures(365)[0].issue_day + 365.0
    )
    assert "2025-06-26" in calendar.validation.actual_issue_dates_local


def test_issue_calendar_rejects_non_greedy_or_selectively_deleted_exposure(
    tmp_path: Path,
) -> None:
    raw = deepcopy(_manifest())
    validation = cast(dict[str, Any], cast(dict[str, Any], raw["partitions"])["validation"])
    exposures = cast(dict[str, Any], validation["non_overlapping_exposures"])
    thirty = cast(dict[str, Any], exposures["30"])
    dates = cast(list[str], thirty["issue_dates_local"])
    dates.pop(2)
    thirty["count"] = len(dates)

    with pytest.raises(ValueError, match="frozen greedy rule"):
        load_frozen_issue_calendar(
            _write_manifest(tmp_path, raw),
            config=load_background_protocol(BACKGROUND_CONFIG),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("future_scores", "not frozen before background scores"),
        ("wrong_column", "available_at-only boundary"),
        ("missing_partition", "issue partitions keys"),
        ("wrong_timezone", "timezone differs"),
    ],
)
def test_issue_calendar_rejects_boundary_and_schema_drift(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    raw = deepcopy(_manifest())
    if mutation == "future_scores":
        raw["scores_seen_before_freeze"] = True
    elif mutation == "wrong_column":
        cast(dict[str, Any], raw["source"])["column_read"] = "anomaly_value"
    elif mutation == "missing_partition":
        del cast(dict[str, Any], raw["partitions"])["development"]
    else:
        cast(dict[str, Any], raw["semantics"])["issue_timezone"] = "UTC"

    with pytest.raises(ValueError, match=message):
        load_frozen_issue_calendar(
            _write_manifest(tmp_path, raw),
            config=load_background_protocol(BACKGROUND_CONFIG),
        )
