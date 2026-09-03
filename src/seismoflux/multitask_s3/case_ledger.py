"""Local-only event comparison ledger from existing diagnostics, with no scoring.

Rows refer to the deduplicated ``events`` metadata table. Every saved exposure,
contrast, alarm budget and strict/70-km classification is retained. No best-case
selection, target construction, model computation, or file access occurs here.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

VARIANTS = ("CATALOG", "R30_REFERENCE", "CAT_COV", "CAT_SNAP", "CAT_DYN")
CONTRASTS = (
    *((variant, "CATALOG") for variant in VARIANTS if variant != "CATALOG"),
    ("CAT_DYN", "CAT_COV"),
    ("CAT_DYN", "CAT_SNAP"),
)
MODES = (("strict", "strict_hits"), ("secondary_70km", "secondary_70km_hits"))
CLASSIFICATIONS = ("gained", "lost", "both_hit", "both_miss")


def _aware_text(value: Any, label: str) -> str:
    # Preserve original fractional-second precision and timezone representation.
    text = value.isoformat() if isinstance(value, datetime) else value
    if not isinstance(text, str):
        raise ValueError(f"{label} must be an aware ISO timestamp or datetime")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
    return text


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _bools(values: Any, size: int, label: str) -> list[bool]:
    if (
        not isinstance(values, list | tuple)
        or len(values) != size
        or any(not isinstance(value, bool) for value in values)
    ):
        raise ValueError(f"{label} must be aligned boolean values, not missing or numeric flags")
    return list(values)


def _event_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "origin_time_utc": _aware_text(value["origin_time_utc"], "event origin"),
        "available_at": _aware_text(value["available_at"], "event availability"),
        "magnitude": _finite(value["magnitude"], "raw magnitude"),
        "longitude": _finite(value["longitude"], "longitude"),
        "latitude": _finite(value["latitude"], "latitude"),
    }
    if not -180 <= result["longitude"] <= 180 or not -90 <= result["latitude"] <= 90:
        raise ValueError("event coordinates must be longitude/latitude degrees")
    return result


def build_case_ledger(
    records: Sequence[Mapping[str, Any]],
    *,
    catalog_metadata: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Join only already-scored events, preserving every positive and negative case.

    ``catalog_metadata`` uses canonical event_id keys and origin_time_utc,
    available_at, magnitude, longitude, latitude fields. Unused entries are never
    inspected. Missing metadata or unaligned diagnostics fails explicitly rather
    than dropping an earthquake or inventing a hit. Summary counts distinguish
    unique earthquakes, event exposures, and expanded comparison rows; they are
    inventory counts, not a new prediction score.
    """
    events: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    exposures: set[tuple[str, int, str, str]] = set()
    issues: set[tuple[str, str]] = set()
    event_exposures = primary_event_exposures = 0
    expected_grid: str | None = None
    expected_budgets: set[float] | None = None
    for record in records:
        fold, horizon, band = record["fold_id"], record["horizon_days"], record["magnitude_band"]
        if not isinstance(fold, str) or not fold or band not in ("Ms5_6", "Ms6_plus"):
            raise ValueError("exposure must identify its fold and formal magnitude band")
        if isinstance(horizon, bool) or horizon not in (7, 30, 90, 180, 365):
            raise ValueError("exposure horizon must be registered")
        issue = _aware_text(record["issue_time_utc"], "issue")
        key = (fold, horizon, band, issue)
        if key in exposures:
            raise ValueError("duplicate fold/horizon/band/issue exposure")
        exposures.add(key)
        issues.add((fold, issue))
        primary = record["primary_nonoverlap"]
        if not isinstance(primary, bool):
            raise ValueError("primary_nonoverlap must be boolean")
        identifiers = list(record["target_event_ids"])
        cells = list(record["target_cell_indices"])
        size = len(identifiers)
        if len(set(identifiers)) != size or any(
            not isinstance(value, str) or not value for value in identifiers
        ):
            raise ValueError("target event identifiers must be unique nonempty strings")
        if len(cells) != size or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in cells
        ):
            raise ValueError("target cell positions must be aligned nonnegative integers")
        anchor = _bools(record["anchor_mask"], size, "anchor mask")
        fingerprint = hashlib.sha256(
            json.dumps(
                list(zip(identifiers, cells, anchor, strict=True)),
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        event_exposures += size
        primary_event_exposures += size if primary else 0
        for identifier in identifiers:
            if identifier not in catalog_metadata:
                raise ValueError(f"missing metadata for already-scored event: {identifier}")
            if identifier not in events:
                events[identifier] = _event_metadata(catalog_metadata[identifier])
        spatial = record["spatial"]
        if set(spatial) != set(VARIANTS):
            raise ValueError("all five registered spatial variants must be present")
        alarms: dict[str, dict[float, Any]] = {}
        for variant in VARIANTS:
            score = spatial[variant]
            grid = score["grid_id"]
            if not isinstance(grid, str) or not grid:
                raise ValueError("scored grid identity must be nonempty")
            expected_grid = grid if expected_grid is None else expected_grid
            if grid != expected_grid:
                raise ValueError("spatial variants/exposures must share the frozen grid")
            if (
                score["event_count"] != size
                or score["anchor_count"] != sum(anchor)
                or score["_local"]["target_order_sha256"] != fingerprint
                or _bools(score["_local"]["anchor_mask"], size, "variant anchor mask") != anchor
            ):
                raise ValueError("variant target fingerprint/order/anchor counts are not aligned")
            by_budget: dict[float, Any] = {}
            for alarm in score["alarms"]:
                budget = _finite(alarm["area_budget_km2"], "budget")
                paid = _finite(alarm["actual_area_km2"], "actual paid area")
                if budget < 0 or paid < 0 or budget in by_budget:
                    raise ValueError(
                        "budgets must be unique nonnegative values with nonnegative cost"
                    )
                if alarm["secondary_70km"]["status"] != "scored":
                    raise ValueError("70-km diagnostic was not scored; cannot classify or omit it")
                by_budget[budget] = {
                    "paid": paid,
                    **{
                        mode: _bools(alarm["_local"][hit_key], size, f"{mode} hit array")
                        for mode, hit_key in MODES
                    },
                }
            if not by_budget:
                raise ValueError("every scored exposure must retain its registered alarm budgets")
            expected_budgets = set(by_budget) if expected_budgets is None else expected_budgets
            if set(by_budget) != expected_budgets:
                raise ValueError("variant/exposure alarm budgets are not aligned")
            alarms[variant] = by_budget
        for candidate, reference in CONTRASTS:
            for budget in alarms[candidate]:
                c_alarm, r_alarm = alarms[candidate][budget], alarms[reference][budget]
                for mode, _ in MODES:
                    for index, identifier in enumerate(identifiers):
                        c_hit, r_hit = c_alarm[mode][index], r_alarm[mode][index]
                        classification = (
                            "both_hit"
                            if c_hit and r_hit
                            else "gained"
                            if c_hit
                            else "lost"
                            if r_hit
                            else "both_miss"
                        )
                        rows.append(
                            {
                                "event_id": identifier,
                                "target_cell_index": cells[index],
                                "fold_id": fold,
                                "horizon_days": horizon,
                                "magnitude_band": band,
                                "issue_time_utc": issue,
                                "primary_nonoverlap": primary,
                                "event_view": "anchor" if anchor[index] else "subsequent",
                                "candidate": candidate,
                                "reference": reference,
                                "comparison": f"{candidate}_minus_{reference}",
                                "area_budget_km2": budget,
                                "candidate_actual_area_km2": c_alarm["paid"],
                                "reference_actual_area_km2": r_alarm["paid"],
                                "mode": mode,
                                "candidate_hit": c_hit,
                                "reference_hit": r_hit,
                                "classification": classification,
                            }
                        )
    return {
        "local_only": True,
        "status": "ledger_complete_no_case_selection",
        "source": "existing_scored_event_diagnostics_only_no_new_targets_or_scores",
        "events": events,
        "rows": rows,
        "summary": {
            "exposure_count_fold_horizon_band_issue": len(exposures),
            "distinct_fold_issue_count": len(issues),
            "unique_event_count": len(events),
            "event_exposure_count": event_exposures,
            "primary_event_exposure_count": primary_event_exposures,
            "comparison_row_count": len(rows),
            "classification_row_counts": {
                label: sum(row["classification"] == label for row in rows)
                for label in CLASSIFICATIONS
            },
            "note": "inventory_not_skill_score; repeated_windows_and_contrasts_are_not_new_events",
        },
    }
