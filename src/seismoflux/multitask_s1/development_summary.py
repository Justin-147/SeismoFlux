"""Preregistered, JSON-compatible scientific summaries for the S1-C0 screen.

This module consumes only the raw rows already produced by authorized scoring.
It performs no file I/O and never pools horizons, alarm areas, or magnitude
bands to enlarge the apparent independent sample size.  Its conclusions are a
directional development screen, never a champion or holdout authorization.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, TypeAlias, cast

import numpy as np

from seismoflux.multitask_s1.development_contract import (
    DEVELOPMENT_FOLD_IDS,
    HORIZONS_DAYS,
)
from seismoflux.multitask_s1.development_score import (
    FIXED_EPISODE_DEFINITION,
    MAIN_SCIENTIFIC_ANCHOR,
    DevelopmentRawScores,
    JointRawScoreRow,
    LocationRawScoreRow,
    MagnitudeRawScoreRow,
    TimeRawScoreRow,
)

# JSON compatibility is enforced by construction and serialization tests.  An
# explicit ``Any`` alias avoids recursive-container invariance obscuring the
# scientific code under strict static checking.
JsonValue: TypeAlias = Any

BOOTSTRAP_REPLICATES: Final = 2_000
BOOTSTRAP_ROOT_SEED: Final = 147
BASE_LOCATION_MODEL: Final = "L0_UNIFORM"
MAIN_LOCATION_MODELS: Final[tuple[str, ...]] = (
    "L1_REGIONAL_CONSTANT",
    "L2_KDE_CAUSAL",
    "L2_KDE75_LEGACY",
    "L3_B0_R30_CAUSAL",
)
BASE_TIME_MODEL: Final = "T0_POISSON_EXPANDING"
NB2_TIME_MODEL: Final = "T1_NEGATIVE_BINOMIAL"
BASE_JOINT_MODEL: Final = "J0_U_P_GR"
JOINT_CANDIDATE_MODELS: Final[tuple[str, ...]] = (
    "J1_R_P_GR",
    "J2_KDE_P_GR",
    "J3_R30_P_GR",
    "J4_KDE_NB_GR",
)
_M0_MAIN_SUPPORT: Final = "M>=4 unique physical events"
_M0_M5_SUPPORT: Final = "M>=5 unique physical events, M0 re-normalized tail"
_M3_M5_SUPPORT: Final = "M>=5 unique physical events, conditional tail"
_TIME_BANDS: Final[tuple[str, ...]] = ("M5_6", "M6_plus")


class DevelopmentSummaryError(ValueError):
    """Raised when raw score identities cannot support a mechanical summary."""


@dataclass(frozen=True, slots=True)
class _EpisodeUnit:
    unit_id: str
    fold_id: str
    episode_id: str
    global_member_count: int
    candidate_hits: float
    baseline_hits: float
    total_weight: float


def _fold_order(fold_id: str) -> int:
    try:
        return DEVELOPMENT_FOLD_IDS.index(fold_id)
    except ValueError as exc:
        raise DevelopmentSummaryError("raw score contains a non-development fold") from exc


def _metric(value: float | None, *, reason: str | None = None) -> dict[str, JsonValue]:
    if value is None:
        return {
            "status": "not_evaluable",
            "value": None,
            "reason": reason or "no_evaluable_independent_units",
        }
    value = float(value)
    if math.isnan(value) or value == math.inf:
        raise DevelopmentSummaryError("summary metric is NaN or positive infinity")
    if value == -math.inf:
        return {
            "status": "negative_infinity",
            "value": None,
            "reason": "model_assigned_zero_density_or_probability",
        }
    return {"status": "evaluable", "value": value, "reason": None}


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    if any(math.isnan(value) or value == math.inf for value in values):
        raise DevelopmentSummaryError("raw score contains NaN or positive infinity")
    return math.fsum(values) / len(values)


def _derived_seed(label: str) -> int:
    payload = f"{BOOTSTRAP_ROOT_SEED}|{label}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def _interval(values: Sequence[float]) -> list[JsonValue]:
    ordered = sorted(float(value) for value in values)
    if not ordered or any(not math.isfinite(value) for value in ordered):
        raise DevelopmentSummaryError("bootstrap produced a non-finite distribution")
    lower_index = math.floor(0.025 * (len(ordered) - 1))
    upper_index = math.ceil(0.975 * (len(ordered) - 1))
    return [ordered[lower_index], ordered[upper_index]]


def _bootstrap_mean(values: Sequence[float], *, label: str, unit: str) -> dict[str, JsonValue]:
    samples = tuple(float(value) for value in values)
    if len(samples) < 2:
        return {
            "status": "not_evaluable",
            "reason": "fewer_than_two_independent_units",
            "resampling_unit": unit,
            "root_seed": BOOTSTRAP_ROOT_SEED,
            "derived_seed": _derived_seed(label),
            "replicates_requested": BOOTSTRAP_REPLICATES,
            "replicates_evaluable": 0,
            "confidence_interval_95": None,
        }
    if any(not math.isfinite(value) for value in samples):
        return {
            "status": "not_evaluable",
            "reason": "nonfinite_paired_effect",
            "resampling_unit": unit,
            "root_seed": BOOTSTRAP_ROOT_SEED,
            "derived_seed": _derived_seed(label),
            "replicates_requested": BOOTSTRAP_REPLICATES,
            "replicates_evaluable": 0,
            "confidence_interval_95": None,
        }
    generator = np.random.default_rng(_derived_seed(label))
    array = np.asarray(samples, dtype=np.float64)
    draws: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        indices = generator.integers(0, len(array), size=len(array))
        draws.append(float(np.mean(array[indices])))
    return {
        "status": "evaluable",
        "reason": None,
        "resampling_unit": unit,
        "root_seed": BOOTSTRAP_ROOT_SEED,
        "derived_seed": _derived_seed(label),
        "replicates_requested": BOOTSTRAP_REPLICATES,
        "replicates_evaluable": BOOTSTRAP_REPLICATES,
        "confidence_interval_95": _interval(draws),
    }


def _ratio_difference(units: Sequence[_EpisodeUnit]) -> float | None:
    denominator = math.fsum(unit.total_weight for unit in units)
    if denominator <= 0.0:
        return None
    candidate = math.fsum(unit.candidate_hits for unit in units) / denominator
    baseline = math.fsum(unit.baseline_hits for unit in units) / denominator
    return candidate - baseline


def _bootstrap_episode_ratio(units: Sequence[_EpisodeUnit], *, label: str) -> dict[str, JsonValue]:
    positive = tuple(unit for unit in units if unit.total_weight > 0.0)
    if len(positive) < 2:
        return {
            "status": "not_evaluable",
            "reason": "fewer_than_two_independent_episodes",
            "resampling_unit": "full_catalog_fixed_anchor_episode",
            "root_seed": BOOTSTRAP_ROOT_SEED,
            "derived_seed": _derived_seed(label),
            "replicates_requested": BOOTSTRAP_REPLICATES,
            "replicates_evaluable": 0,
            "confidence_interval_95": None,
        }
    generator = np.random.default_rng(_derived_seed(label))
    draws: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES):
        indices = generator.integers(0, len(positive), size=len(positive))
        sampled = tuple(positive[int(index)] for index in indices)
        effect = _ratio_difference(sampled)
        if effect is None or not math.isfinite(effect):
            raise DevelopmentSummaryError("episode bootstrap produced an invalid ratio")
        draws.append(effect)
    return {
        "status": "evaluable",
        "reason": None,
        "resampling_unit": "full_catalog_fixed_anchor_episode",
        "root_seed": BOOTSTRAP_ROOT_SEED,
        "derived_seed": _derived_seed(label),
        "replicates_requested": BOOTSTRAP_REPLICATES,
        "replicates_evaluable": BOOTSTRAP_REPLICATES,
        "confidence_interval_95": _interval(draws),
    }


def _validate_location_row(row: LocationRawScoreRow) -> None:
    _fold_order(row.fold_id)
    lengths = {
        row.event_count,
        len(row.event_ids),
        len(row.event_log_densities_per_km2),
        len(row.episode_ids),
        len(row.global_episode_member_counts),
        len(row.is_episode_anchor),
        len(row.event_cell_indices),
        len(row.event_longitudes),
        len(row.event_latitudes),
        len(row.event_weights),
    }
    if len(lengths) != 1 or len(set(row.event_ids)) != row.event_count:
        raise DevelopmentSummaryError("location raw event vectors are not aligned and unique")
    if row.hit_flags is not None and len(row.hit_flags) != row.event_count:
        raise DevelopmentSummaryError("location hit flags are not event-aligned")
    if any(count <= 0 for count in row.global_episode_member_counts):
        raise DevelopmentSummaryError("episode member count must remain positive")
    if any(not episode_id for episode_id in row.episode_ids):
        raise DevelopmentSummaryError("episode IDs must remain non-empty")
    if row.metric == "strict_recall":
        if (
            row.basis not in {"all", "anchor", "episode_balanced", "subsequent"}
            or row.area_budget_km2 is None
            or row.actual_area_km2 is None
            or row.hit_weight is None
            or row.total_weight is None
            or row.hit_flags is None
        ):
            raise DevelopmentSummaryError("recall row lost its frozen weighting identity")
        total = math.fsum(row.event_weights)
        hits = math.fsum(
            weight for weight, hit in zip(row.event_weights, row.hit_flags, strict=True) if hit
        )
        if not math.isclose(
            total, row.total_weight, rel_tol=0.0, abs_tol=1.0e-12
        ) or not math.isclose(hits, row.hit_weight, rel_tol=0.0, abs_tol=1.0e-12):
            raise DevelopmentSummaryError("recall aggregate disagrees with raw event payload")
    elif row.metric == "spatial_log_density":
        if row.area_budget_km2 is not None or row.hit_flags is not None:
            raise DevelopmentSummaryError("density row unexpectedly contains an alarm prefix")
    else:
        raise DevelopmentSummaryError("unknown location metric")


def _location_recall_summary(rows: Sequence[LocationRawScoreRow]) -> list[JsonValue]:
    grouped: dict[tuple[str, int, str, str, float, str], list[LocationRawScoreRow]] = defaultdict(
        list
    )
    for row in rows:
        if row.metric != "strict_recall":
            continue
        assert row.basis is not None and row.area_budget_km2 is not None
        grouped[
            (
                row.fold_id,
                row.horizon_days,
                row.magnitude_bin,
                row.basis,
                row.area_budget_km2,
                row.model_id,
            )
        ].append(row)
    output: list[JsonValue] = []
    for key in sorted(
        grouped,
        key=lambda item: (
            _fold_order(item[0]),
            item[1],
            item[2],
            item[3],
            item[4],
            item[5],
        ),
    ):
        fold_id, horizon, magnitude_bin, basis, area, model_id = key
        values = grouped[key]
        total = math.fsum(cast(float, row.total_weight) for row in values)
        hits = math.fsum(cast(float, row.hit_weight) for row in values)
        pooled = None if total <= 0.0 else hits / total
        actual_areas = tuple(cast(float, row.actual_area_km2) for row in values)
        output.append(
            {
                "fold_id": fold_id,
                "horizon_days": horizon,
                "magnitude_bin": magnitude_bin,
                "basis": basis,
                "area_budget_km2": area,
                "model_id": model_id,
                "exposure_count": len(values),
                "event_count": sum(row.event_count for row in values),
                "hit_weight": hits,
                "total_weight": total,
                "pooled_recall": _metric(pooled),
                "mean_actual_area_km2": cast(float, _mean(actual_areas)),
            }
        )
    return output


def _same_location_targets(first: LocationRawScoreRow, second: LocationRawScoreRow) -> bool:
    return (
        first.event_ids,
        first.episode_ids,
        first.global_episode_member_counts,
        first.is_episode_anchor,
        first.event_cell_indices,
        first.event_longitudes,
        first.event_latitudes,
    ) == (
        second.event_ids,
        second.episode_ids,
        second.global_episode_member_counts,
        second.is_episode_anchor,
        second.event_cell_indices,
        second.event_longitudes,
        second.event_latitudes,
    )


def _location_density_summary(rows: Sequence[LocationRawScoreRow]) -> list[JsonValue]:
    density = tuple(row for row in rows if row.metric == "spatial_log_density")
    by_exposure: dict[tuple[str, object, int, str, str], LocationRawScoreRow] = {}
    for row in density:
        key = (
            row.fold_id,
            row.issue_time_utc,
            row.horizon_days,
            row.magnitude_bin,
            row.model_id,
        )
        if key in by_exposure:
            raise DevelopmentSummaryError("duplicate location density exposure")
        by_exposure[key] = row
    grouped: dict[tuple[str, int, str, str], list[LocationRawScoreRow]] = defaultdict(list)
    for row in density:
        grouped[(row.fold_id, row.horizon_days, row.magnitude_bin, row.model_id)].append(row)
    output: list[JsonValue] = []
    for group_key in sorted(
        grouped,
        key=lambda item: (_fold_order(item[0]), item[1], item[2], item[3]),
    ):
        fold_id, horizon, magnitude_bin, model_id = group_key
        candidate_logs: list[float] = []
        baseline_logs: list[float] = []
        missing_baseline = False
        for row in grouped[group_key]:
            baseline = by_exposure.get(
                (
                    row.fold_id,
                    row.issue_time_utc,
                    row.horizon_days,
                    row.magnitude_bin,
                    BASE_LOCATION_MODEL,
                )
            )
            if baseline is None:
                missing_baseline = True
                continue
            if not _same_location_targets(row, baseline):
                raise DevelopmentSummaryError("paired location models use different target events")
            candidate_logs.extend(row.event_log_densities_per_km2)
            baseline_logs.extend(baseline.event_log_densities_per_km2)
        mean_density = _mean(candidate_logs)
        if missing_baseline or len(candidate_logs) != len(baseline_logs):
            information_gain = None
            ig_reason = "paired_L0_exposure_missing"
        else:
            differences = tuple(
                candidate - baseline
                for candidate, baseline in zip(candidate_logs, baseline_logs, strict=True)
            )
            information_gain = _mean(differences)
            ig_reason = None
        output.append(
            {
                "fold_id": fold_id,
                "horizon_days": horizon,
                "magnitude_bin": magnitude_bin,
                "model_id": model_id,
                "event_count": len(candidate_logs),
                "mean_log_density_per_event": _metric(mean_density),
                "paired_information_gain_vs_L0_nats_per_event": _metric(
                    information_gain, reason=ig_reason
                ),
            }
        )
    return output


def _main_anchor_rows(rows: Sequence[LocationRawScoreRow]) -> dict[str, list[LocationRawScoreRow]]:
    result: dict[str, list[LocationRawScoreRow]] = defaultdict(list)
    for row in rows:
        expected_main = (
            row.metric == "strict_recall"
            and row.magnitude_bin == "M5_6"
            and row.horizon_days == 30
            and row.basis == "anchor"
            and row.area_budget_km2 == 600_000.0
            and row.catalog_delay_hours == 24
            and row.hit_tolerance_km == 0.0
            and row.episode_definition == FIXED_EPISODE_DEFINITION
        )
        if row.is_main_scientific_anchor != expected_main:
            raise DevelopmentSummaryError("main scientific anchor flag changed")
        if not expected_main:
            continue
        if row.scientific_anchor_id != MAIN_SCIENTIFIC_ANCHOR:
            raise DevelopmentSummaryError("main scientific anchor identity changed")
        result[row.model_id].append(row)
    return result


def _episode_units(
    candidate_rows: Sequence[LocationRawScoreRow],
    baseline_rows: Sequence[LocationRawScoreRow],
) -> tuple[_EpisodeUnit, ...]:
    def key(row: LocationRawScoreRow) -> tuple[str, object]:
        return (row.fold_id, row.issue_time_utc)

    candidate_map = {key(row): row for row in candidate_rows}
    baseline_map = {key(row): row for row in baseline_rows}
    if len(candidate_map) != len(candidate_rows) or len(baseline_map) != len(baseline_rows):
        raise DevelopmentSummaryError("duplicate main-anchor exposure")
    if set(candidate_map) != set(baseline_map):
        raise DevelopmentSummaryError("main-anchor models have different exposure sets")
    accumulators: dict[str, list[object]] = {}
    seen_events: set[str] = set()
    for exposure_key in sorted(candidate_map, key=lambda item: (_fold_order(item[0]), item[1])):
        candidate = candidate_map[exposure_key]
        baseline = baseline_map[exposure_key]
        if (
            not _same_location_targets(candidate, baseline)
            or candidate.event_weights != baseline.event_weights
        ):
            raise DevelopmentSummaryError("main-anchor paired target payload differs by model")
        assert candidate.hit_flags is not None and baseline.hit_flags is not None
        for index, event_id in enumerate(candidate.event_ids):
            if event_id in seen_events:
                raise DevelopmentSummaryError("main-anchor physical event is duplicated")
            seen_events.add(event_id)
            weight = candidate.event_weights[index]
            if weight <= 0.0:
                continue
            episode_id = candidate.episode_ids[index]
            unit_id = f"{candidate.fold_id}|{episode_id}"
            values = accumulators.setdefault(
                unit_id,
                [
                    candidate.fold_id,
                    episode_id,
                    candidate.global_episode_member_counts[index],
                    0.0,
                    0.0,
                    0.0,
                ],
            )
            if values[0] != candidate.fold_id or values[1] != episode_id:
                raise DevelopmentSummaryError("episode unit identity changed")
            if cast(int, values[2]) != candidate.global_episode_member_counts[index]:
                raise DevelopmentSummaryError("global episode member count changed")
            values[3] = cast(float, values[3]) + (weight if candidate.hit_flags[index] else 0.0)
            values[4] = cast(float, values[4]) + (weight if baseline.hit_flags[index] else 0.0)
            values[5] = cast(float, values[5]) + weight
    return tuple(
        _EpisodeUnit(
            unit_id=unit_id,
            fold_id=cast(str, values[0]),
            episode_id=cast(str, values[1]),
            global_member_count=cast(int, values[2]),
            candidate_hits=cast(float, values[3]),
            baseline_hits=cast(float, values[4]),
            total_weight=cast(float, values[5]),
        )
        for unit_id, values in sorted(accumulators.items())
    )


def _fold_ratio_differences(units: Sequence[_EpisodeUnit]) -> list[JsonValue]:
    output: list[JsonValue] = []
    for fold_id in DEVELOPMENT_FOLD_IDS:
        selected = tuple(unit for unit in units if unit.fold_id == fold_id)
        output.append(
            {
                "fold_id": fold_id,
                "independent_episode_count": len(selected),
                "difference": _metric(
                    _ratio_difference(selected), reason="no_weighted_anchor_in_fold"
                ),
            }
        )
    return output


def _main_location_comparison(
    candidate_model: str,
    main_rows: Mapping[str, Sequence[LocationRawScoreRow]],
) -> dict[str, JsonValue]:
    candidate = tuple(main_rows.get(candidate_model, ()))
    baseline = tuple(main_rows.get(BASE_LOCATION_MODEL, ()))
    region_sensitivity: dict[str, JsonValue] = {
        "status": "not_evaluable",
        "reason": (
            "raw score rows retain lon/lat but not the frozen equal-area x/y or "
            "target-blind origin-fixed 500 km region ID"
        ),
        "definition": "target_blind_origin_fixed_500km_L1_region",
    }
    if not candidate or not baseline:
        return {
            "candidate_model_id": candidate_model,
            "baseline_model_id": BASE_LOCATION_MODEL,
            "status": "not_evaluable",
            "reason": "candidate_or_L0_main_anchor_rows_missing",
            "independent_episode_count": 0,
            "pooled_candidate_recall": _metric(None),
            "pooled_baseline_recall": _metric(None),
            "pooled_difference": _metric(None),
            "fold_differences": _fold_ratio_differences(()),
            "positive_fold_count": 0,
            "paired_bootstrap": _bootstrap_episode_ratio((), label=candidate_model),
            "leave_largest_episode": {
                "status": "not_evaluable",
                "reason": "no_independent_episode",
            },
            "fixed_origin_500km_region_sensitivity": region_sensitivity,
            "direction": "not_evaluable",
        }
    units = _episode_units(candidate, baseline)
    denominator = math.fsum(unit.total_weight for unit in units)
    candidate_recall = (
        None
        if denominator <= 0.0
        else math.fsum(unit.candidate_hits for unit in units) / denominator
    )
    baseline_recall = (
        None
        if denominator <= 0.0
        else math.fsum(unit.baseline_hits for unit in units) / denominator
    )
    effect = _ratio_difference(units)
    fold_differences = _fold_ratio_differences(units)
    positive_folds = sum(
        1
        for value in fold_differences
        if isinstance(value, dict)
        and isinstance(value.get("difference"), dict)
        and value["difference"].get("status") == "evaluable"
        and cast(float, value["difference"]["value"]) > 0.0
    )
    if units:
        largest = sorted(units, key=lambda unit: (-unit.global_member_count, unit.unit_id))[0]
        remaining = tuple(unit for unit in units if unit.unit_id != largest.unit_id)
        leave_effect = _ratio_difference(remaining)
        leave_largest: dict[str, JsonValue] = {
            "status": "evaluable" if leave_effect is not None else "not_evaluable",
            "removed_episode_unit_id": largest.unit_id,
            "removed_global_member_count": largest.global_member_count,
            "remaining_independent_episode_count": len(remaining),
            "pooled_difference_after_removal": _metric(
                leave_effect, reason="no_weight_after_removing_largest_episode"
            ),
        }
    else:
        leave_effect = None
        leave_largest = {"status": "not_evaluable", "reason": "no_independent_episode"}
    if effect is None:
        direction = "not_evaluable"
    elif effect > 0.0 and positive_folds >= 2 and leave_effect is not None and leave_effect >= 0.0:
        direction = "positive"
    elif effect <= 0.0:
        direction = "non_positive"
    else:
        direction = "mixed"
    return {
        "candidate_model_id": candidate_model,
        "baseline_model_id": BASE_LOCATION_MODEL,
        "status": "evaluable" if effect is not None else "not_evaluable",
        "reason": None if effect is not None else "no_weighted_anchor_episode",
        "independent_episode_count": len(units),
        "pooled_candidate_recall": _metric(candidate_recall),
        "pooled_baseline_recall": _metric(baseline_recall),
        "pooled_difference": _metric(effect),
        "fold_differences": fold_differences,
        "positive_fold_count": positive_folds,
        "paired_bootstrap": _bootstrap_episode_ratio(
            units, label=f"location-main|{candidate_model}"
        ),
        "leave_largest_episode": leave_largest,
        "fixed_origin_500km_region_sensitivity": region_sensitivity,
        "direction": direction,
    }


def _location_summary(rows: Sequence[LocationRawScoreRow]) -> dict[str, JsonValue]:
    for row in rows:
        _validate_location_row(row)
    main = _main_anchor_rows(rows)
    comparisons = [_main_location_comparison(model_id, main) for model_id in MAIN_LOCATION_MODELS]
    return {
        "pooling_rule": "fold_horizon_magnitude_bin_basis_area_model_kept_separate",
        "recall_groups": _location_recall_summary(rows),
        "log_density_and_information_gain_groups": _location_density_summary(rows),
        "main_scientific_anchor": {
            "anchor_id": MAIN_SCIENTIFIC_ANCHOR,
            "catalog_delay_hours": 24,
            "magnitude_bin": "M5_6",
            "horizon_days": 30,
            "area_budget_km2": 600_000.0,
            "hit_tolerance_km": 0.0,
            "target_basis": "fixed_anchor_episode",
            "comparisons": comparisons,
        },
    }


def _time_map(
    rows: Sequence[TimeRawScoreRow],
) -> dict[tuple[str, object, int, str, str], TimeRawScoreRow]:
    result: dict[tuple[str, object, int, str, str], TimeRawScoreRow] = {}
    for row in rows:
        _fold_order(row.fold_id)
        key = (
            row.fold_id,
            row.issue_time_utc,
            row.horizon_days,
            row.magnitude_band,
            row.model_id,
        )
        if key in result:
            raise DevelopmentSummaryError("duplicate time exposure score")
        result[key] = row
    return result


def _fold_mean_differences(
    values: Sequence[tuple[str, float]], *, unit_label: str
) -> list[JsonValue]:
    output: list[JsonValue] = []
    for fold_id in DEVELOPMENT_FOLD_IDS:
        selected = tuple(value for fold, value in values if fold == fold_id)
        output.append(
            {
                "fold_id": fold_id,
                f"{unit_label}_count": len(selected),
                "difference": _metric(_mean(selected)),
            }
        )
    return output


def _time_comparison(
    rows: Mapping[tuple[str, object, int, str, str], TimeRawScoreRow],
    *,
    band: str,
    horizon: int,
) -> dict[str, JsonValue]:
    baseline = {
        key[:-1]: value
        for key, value in rows.items()
        if key[2] == horizon and key[3] == band and key[4] == BASE_TIME_MODEL
    }
    candidate = {
        key[:-1]: value
        for key, value in rows.items()
        if key[2] == horizon and key[3] == band and key[4] == NB2_TIME_MODEL
    }
    if not baseline and not candidate:
        return {
            "magnitude_band": band,
            "horizon_days": horizon,
            "status": "not_evaluable",
            "reason": "no_time_exposures",
            "exposure_count": 0,
            "evaluable_paired_exposure_count": 0,
            "zero_event_exposure_count": 0,
            "pooled_count_log_score_difference_T1_minus_T0": _metric(None),
            "fold_differences": _fold_mean_differences((), unit_label="exposure"),
            "paired_bootstrap": _bootstrap_mean(
                (), label=f"time|{band}|{horizon}", unit="nonoverlapping_primary_exposure"
            ),
        }
    if set(baseline) != set(candidate):
        raise DevelopmentSummaryError("T1 and T0 exposure identities differ")
    log_differences: list[tuple[str, float]] = []
    baseline_bias: list[float] = []
    candidate_bias: list[float] = []
    baseline_brier: list[float] = []
    candidate_brier: list[float] = []
    zero_count = 0
    for key in sorted(baseline, key=lambda item: (_fold_order(item[0]), item[1])):
        t0 = baseline[key]
        t1 = candidate[key]
        if t0.observed_count != t1.observed_count:
            raise DevelopmentSummaryError("paired T1/T0 observed count differs")
        zero_count += int(t0.observed_count == 0)
        baseline_bias.append(t0.count_bias)
        candidate_bias.append(t1.count_bias)
        if t0.occurrence_brier is not None and t1.occurrence_brier is not None:
            baseline_brier.append(t0.occurrence_brier)
            candidate_brier.append(t1.occurrence_brier)
        if t0.count_log_score is not None and t1.count_log_score is not None:
            log_differences.append((t0.fold_id, t1.count_log_score - t0.count_log_score))
    delta_values = tuple(value for _, value in log_differences)
    fold_differences = _fold_mean_differences(log_differences, unit_label="exposure")
    positive_folds = sum(
        1
        for item in fold_differences
        if isinstance(item, dict)
        and isinstance(item.get("difference"), dict)
        and item["difference"].get("status") == "evaluable"
        and cast(float, item["difference"]["value"]) > 0.0
    )
    pooled = _mean(delta_values)
    return {
        "magnitude_band": band,
        "horizon_days": horizon,
        "status": "evaluable" if pooled is not None else "not_evaluable",
        "reason": None if pooled is not None else "T1_not_evaluable_on_all_paired_exposures",
        "exposure_count": len(baseline),
        "evaluable_paired_exposure_count": len(delta_values),
        "zero_event_exposure_count": zero_count,
        "pooled_count_log_score_difference_T1_minus_T0": _metric(pooled),
        "fold_differences": fold_differences,
        "positive_fold_count": positive_folds,
        "mean_expected_count_bias": {
            "T0": _metric(_mean(baseline_bias)),
            "T1": _metric(_mean(candidate_bias)),
            "T1_minus_T0": _metric(
                None
                if not baseline_bias
                else _mean(
                    tuple(
                        candidate_value - baseline_value
                        for candidate_value, baseline_value in zip(
                            candidate_bias, baseline_bias, strict=True
                        )
                    )
                )
            ),
        },
        "mean_occurrence_brier": {
            "T0": _metric(_mean(baseline_brier)),
            "T1": _metric(_mean(candidate_brier)),
            "T1_minus_T0": _metric(
                None
                if not baseline_brier
                else _mean(
                    tuple(
                        candidate_value - baseline_value
                        for candidate_value, baseline_value in zip(
                            candidate_brier, baseline_brier, strict=True
                        )
                    )
                )
            ),
        },
        "paired_bootstrap": _bootstrap_mean(
            delta_values,
            label=f"time|{band}|{horizon}",
            unit="nonoverlapping_primary_exposure",
        ),
    }


def _time_summary(rows: Sequence[TimeRawScoreRow]) -> dict[str, JsonValue]:
    mapped = _time_map(rows)
    comparisons = [
        _time_comparison(mapped, band=band, horizon=horizon)
        for band in _TIME_BANDS
        for horizon in HORIZONS_DAYS
    ]
    return {
        "pooling_rule": "magnitude_band_and_horizon_kept_separate",
        "zero_event_windows_retained": True,
        "comparisons": comparisons,
    }


def _magnitude_summary(rows: Sequence[MagnitudeRawScoreRow]) -> dict[str, JsonValue]:
    mapped: dict[tuple[str, object, str, str], MagnitudeRawScoreRow] = {}
    for row in rows:
        _fold_order(row.fold_id)
        if len(row.event_ids) != len(row.event_log_probabilities):
            raise DevelopmentSummaryError("magnitude event scores are not aligned")
        key = (row.fold_id, row.forecast_issue_time_utc, row.model_id, row.conditional_support)
        if key in mapped:
            raise DevelopmentSummaryError("duplicate magnitude issue score")
        mapped[key] = row

    m0_main_rows = tuple(
        row
        for row in rows
        if row.model_id == "M0_GR_GLOBAL" and row.conditional_support == _M0_MAIN_SUPPORT
    )
    main_event_ids: list[str] = []
    main_logs: list[float] = []
    main_brier_sum = 0.0
    main_brier_count = 0
    for row in m0_main_rows:
        main_event_ids.extend(row.event_ids)
        main_logs.extend(row.event_log_probabilities)
        if row.mean_m6_plus_brier is not None:
            main_brier_sum += row.mean_m6_plus_brier * len(row.event_ids)
            main_brier_count += len(row.event_ids)
    if len(set(main_event_ids)) != len(main_event_ids):
        raise DevelopmentSummaryError("M0 main description duplicated a unique event")

    m0_tail = {
        (row.fold_id, row.forecast_issue_time_utc): row
        for row in rows
        if row.model_id == "M0_GR_GLOBAL" and row.conditional_support == _M0_M5_SUPPORT
    }
    m3_tail = {
        (row.fold_id, row.forecast_issue_time_utc): row
        for row in rows
        if row.model_id == "M3_GR_LONG_M5" and row.conditional_support == _M3_M5_SUPPORT
    }
    if len(m0_tail) != sum(
        row.model_id == "M0_GR_GLOBAL" and row.conditional_support == _M0_M5_SUPPORT for row in rows
    ) or len(m3_tail) != sum(
        row.model_id == "M3_GR_LONG_M5" and row.conditional_support == _M3_M5_SUPPORT
        for row in rows
    ):
        raise DevelopmentSummaryError("duplicate M5-tail issue score")
    if set(m0_tail) != set(m3_tail) and (m0_tail or m3_tail):
        raise DevelopmentSummaryError("M3 and conditional M0 issue populations differ")
    paired_differences: list[tuple[str, float]] = []
    common_event_ids: list[str] = []
    m0_brier_sum = 0.0
    m3_brier_sum = 0.0
    brier_count = 0
    for tail_key in sorted(m0_tail, key=lambda item: (_fold_order(item[0]), item[1])):
        m0 = m0_tail[tail_key]
        m3 = m3_tail[tail_key]
        if m0.event_ids != m3.event_ids:
            raise DevelopmentSummaryError("M3 and conditional M0 event identities differ")
        common_event_ids.extend(m0.event_ids)
        paired_differences.extend(
            (m0.fold_id, m3_value - m0_value)
            for m0_value, m3_value in zip(
                m0.event_log_probabilities,
                m3.event_log_probabilities,
                strict=True,
            )
        )
        if m0.mean_m6_plus_brier is not None and m3.mean_m6_plus_brier is not None:
            m0_brier_sum += m0.mean_m6_plus_brier * len(m0.event_ids)
            m3_brier_sum += m3.mean_m6_plus_brier * len(m3.event_ids)
            brier_count += len(m0.event_ids)
    if len(set(common_event_ids)) != len(common_event_ids):
        raise DevelopmentSummaryError("M5-tail sensitivity duplicated a unique event")
    difference_values = tuple(value for _, value in paired_differences)
    return {
        "M0_main_description": {
            "model_id": "M0_GR_GLOBAL",
            "conditional_support": _M0_MAIN_SUPPORT,
            "unique_event_count": len(main_event_ids),
            "mean_log_probability": _metric(_mean(main_logs)),
            "mean_M6_plus_brier": _metric(
                None if main_brier_count == 0 else main_brier_sum / main_brier_count
            ),
        },
        "M3_vs_M0_common_M5_sensitivity": {
            "status": "evaluable" if difference_values else "not_evaluable",
            "reason": None if difference_values else "no_common_unique_M5_events",
            "common_support": "same_unique_M5_plus_events",
            "direct_comparison_to_M0_M4_support_allowed": False,
            "unique_event_count": len(common_event_ids),
            "mean_log_probability_difference_M3_minus_M0": _metric(_mean(difference_values)),
            "fold_differences": _fold_mean_differences(
                paired_differences, unit_label="unique_event"
            ),
            "mean_M6_plus_brier_difference_M3_minus_M0": _metric(
                None if brier_count == 0 else (m3_brier_sum - m0_brier_sum) / brier_count
            ),
            "paired_bootstrap": _bootstrap_mean(
                difference_values,
                label="magnitude|M3-vs-M0|common-M5",
                unit="unique_physical_M5_plus_event",
            ),
        },
    }


def _joint_summary(rows: Sequence[JointRawScoreRow]) -> dict[str, JsonValue]:
    mapped: dict[tuple[str, object, int, str], JointRawScoreRow] = {}
    for row in rows:
        _fold_order(row.fold_id)
        key = (row.fold_id, row.issue_time_utc, row.horizon_days, row.joint_model_id)
        if key in mapped:
            raise DevelopmentSummaryError("duplicate joint exposure score")
        mapped[key] = row
    comparisons: list[JsonValue] = []
    for horizon in HORIZONS_DAYS:
        baseline = {
            key[:-1]: value
            for key, value in mapped.items()
            if key[2] == horizon and key[3] == BASE_JOINT_MODEL
        }
        for candidate_model in JOINT_CANDIDATE_MODELS:
            candidate = {
                key[:-1]: value
                for key, value in mapped.items()
                if key[2] == horizon and key[3] == candidate_model
            }
            if not baseline and not candidate:
                differences: list[tuple[str, float]] = []
                exposure_count = 0
                zero_count = 0
                reason: str | None = "no_joint_exposures"
            else:
                if set(baseline) != set(candidate):
                    raise DevelopmentSummaryError(
                        "candidate and J0 joint exposure identities differ"
                    )
                differences = []
                zero_count = 0
                for joint_key in sorted(baseline, key=lambda item: (_fold_order(item[0]), item[1])):
                    j0 = baseline[joint_key]
                    model = candidate[joint_key]
                    if j0.event_count != model.event_count:
                        raise DevelopmentSummaryError("paired joint event count differs")
                    zero_count += int(j0.event_count == 0)
                    if j0.joint_log_score is not None and model.joint_log_score is not None:
                        differences.append((j0.fold_id, model.joint_log_score - j0.joint_log_score))
                exposure_count = len(baseline)
                reason = None if differences else "no_evaluable_paired_joint_exposure"
            delta_values = tuple(value for _, value in differences)
            fold_differences = _fold_mean_differences(differences, unit_label="exposure")
            comparisons.append(
                {
                    "horizon_days": horizon,
                    "candidate_model_id": candidate_model,
                    "baseline_model_id": BASE_JOINT_MODEL,
                    "status": "evaluable" if differences else "not_evaluable",
                    "reason": reason,
                    "exposure_count": exposure_count,
                    "evaluable_paired_exposure_count": len(differences),
                    "zero_event_exposure_count": zero_count,
                    "mean_joint_log_score_difference": _metric(_mean(delta_values)),
                    "fold_differences": fold_differences,
                    "positive_fold_count": sum(
                        1
                        for item in fold_differences
                        if isinstance(item, dict)
                        and isinstance(item.get("difference"), dict)
                        and item["difference"].get("status") == "evaluable"
                        and cast(float, item["difference"]["value"]) > 0.0
                    ),
                    "paired_bootstrap": _bootstrap_mean(
                        delta_values,
                        label=f"joint|{horizon}|{candidate_model}",
                        unit="nonoverlapping_primary_exposure",
                    ),
                }
            )
    return {
        "pooling_rule": "horizon_kept_separate",
        "target": "total_M5_plus_count_once_plus_conditional_location_and_M0_magnitude",
        "M5_6_and_M6_plus_count_terms_summed": False,
        "comparisons": comparisons,
    }


def _science_gate(location: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    anchor = location.get("main_scientific_anchor")
    comparisons: Sequence[JsonValue] = ()
    if isinstance(anchor, dict) and isinstance(anchor.get("comparisons"), list):
        comparisons = cast(list[JsonValue], anchor["comparisons"])
    directions: list[JsonValue] = []
    evaluable = 0
    for value in comparisons:
        if not isinstance(value, dict):
            continue
        direction = value.get("direction")
        if direction != "not_evaluable":
            evaluable += 1
        directions.append(
            {
                "model_id": cast(str, value.get("candidate_model_id")),
                "direction": cast(str, direction),
            }
        )
    return {
        "role": "direction_and_evidence_screen_only",
        "directional_comparisons": directions,
        "direction_status": (
            "evidence_insufficient" if evaluable == 0 else "directional_screen_available"
        ),
        "evidence_status": "insufficient_for_champion_or_holdout",
        "evidence_limitations": [
            "S1-C0 uses the preliminary all-M4 screen",
            "fixed-origin 500 km region sensitivity is not mechanically available in raw rows",
            "S1-C1 causal local-completeness evaluation is mandatory regardless of direction",
        ],
        "champion_selection_allowed": False,
        "holdout_opening_allowed": False,
        "small_stable_gain_may_be_reported": True,
        "effect_inflation_by_area_or_horizon_pooling_allowed": False,
    }


def summarize_development_scores(scores: DevelopmentRawScores) -> dict[str, JsonValue]:
    """Build the complete preregistered S1-C0 scientific summary in memory."""

    if not isinstance(scores, DevelopmentRawScores):
        raise TypeError("scores must be DevelopmentRawScores")
    location = _location_summary(scores.location)
    time = _time_summary(scores.time)
    magnitude = _magnitude_summary(scores.magnitude)
    joint = _joint_summary(scores.joint)
    return {
        "schema_version": 1,
        "record_type": "s1_c0_preregistered_development_scientific_summary",
        "scientific_role": "preliminary_directional_screen_not_champion_selection",
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "root_seed": BOOTSTRAP_ROOT_SEED,
            "method": "deterministic_paired_percentile",
        },
        "pooling_prohibitions": {
            "horizons_pooled": False,
            "alarm_areas_pooled": False,
            "magnitude_bins_pooled": False,
            "grid_cells_treated_as_independent_units": False,
        },
        "location": location,
        "time": time,
        "magnitude": magnitude,
        "joint": joint,
        "science_gate": _science_gate(location),
        "champion_selection_allowed": False,
        "holdout_opening_allowed": False,
        "mandatory_next_stage_regardless_of_screen_result": (
            "S1-C1_causal_local_completeness_main"
        ),
    }


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_ROOT_SEED",
    "DevelopmentSummaryError",
    "summarize_development_scores",
]
