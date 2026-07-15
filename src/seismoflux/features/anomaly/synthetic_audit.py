"""Deterministic synthetic-prefix property audit for the frozen stage-3 protocol."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta, timezone
from typing import Final, cast

import numpy as np

from seismoflux.features.anomaly.config import AnomalyHistoryConfig
from seismoflux.features.anomaly.coverage import local_coverage_features
from seismoflux.features.anomaly.nulls import NullReasonCode
from seismoflux.features.anomaly.spatial import (
    SpatialEntityArrays,
    compute_spatial_features,
)
from seismoflux.features.anomaly.state import AnomalyState, build_anomaly_state_history

_GENERATOR: Final[str] = "deterministic_coherent_anomaly_trajectory_v1"
_SEED_START: Final[int] = 0
_SEED_STOP: Final[int] = 32
_INVARIANTS: Final[tuple[str, ...]] = (
    "full_input_equals_available_prefix_scientific_payload",
    "future_mutations_do_not_change_prior_snapshot",
    "input_permutation_does_not_change_identity_or_values",
    "available_at_equal_issue_is_included",
    "available_at_after_issue_is_excluded",
    "multiscale_closed_ball_counts_are_nested",
    "local_reporting_gap_does_not_mask_unrelated_cells",
)
_SYNTHETIC_PROPERTY_KEYS: Final[frozenset[str]] = frozenset(
    {
        "generator",
        "seed_start_inclusive",
        "seed_stop_exclusive",
        "seed_count",
        "invariants",
    }
)
_LOCAL_TIMEZONE: Final[timezone] = timezone(timedelta(hours=8))
_CONTINUED_REPORT_STATE: Final[str] = "持续"
_PERIOD_COUNT: Final[int] = 6
_BOUNDARY_ISSUE_INDEX: Final[int] = 2


@dataclass(frozen=True, slots=True)
class SyntheticPrefixAuditResult:
    """Public-safe aggregate; no synthetic identities or locations escape."""

    passed: bool
    seed_count: int
    invariant_count: int
    check_count: int
    failure_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.passed, bool):
            raise ValueError("synthetic-prefix audit passed flag must be boolean")
        counts = (
            self.seed_count,
            self.invariant_count,
            self.check_count,
            self.failure_count,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts
        ):
            raise ValueError("synthetic-prefix audit counts must be non-negative integers")
        if self.seed_count == 0 or self.invariant_count == 0:
            raise ValueError("synthetic-prefix audit must execute seeds and invariants")
        if self.check_count != self.seed_count * self.invariant_count:
            raise ValueError("synthetic-prefix audit check count is inconsistent")
        if self.failure_count > self.check_count:
            raise ValueError("synthetic-prefix audit failure count exceeds checks")
        if self.passed is not (self.failure_count == 0):
            raise ValueError("synthetic-prefix audit passed flag is inconsistent")


@dataclass(frozen=True, slots=True)
class _SyntheticPrefixPlan:
    seeds: tuple[int, ...]
    invariants: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _SyntheticTrajectory:
    reports: tuple[dict[str, object], ...]
    observations: tuple[dict[str, object], ...]
    issue_times: tuple[datetime, ...]
    boundary_issue_time: datetime
    equal_boundary_entity_id: str
    after_boundary_entity_id: str


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _integer(mapping: Mapping[str, object], key: str, *, label: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label}.{key} must be an integer")
    return value


def _parse_plan(config: AnomalyHistoryConfig) -> _SyntheticPrefixPlan:
    audit = _mapping(config.audit, label="audit")
    raw = _mapping(
        audit.get("synthetic_prefix_property"),
        label="audit.synthetic_prefix_property",
    )
    if set(raw) != _SYNTHETIC_PROPERTY_KEYS:
        raise ValueError("synthetic-prefix property fields differ from the frozen protocol")
    if raw.get("generator") != _GENERATOR:
        raise ValueError("synthetic-prefix generator differs from the frozen protocol")
    start = _integer(raw, "seed_start_inclusive", label="synthetic-prefix property")
    stop = _integer(raw, "seed_stop_exclusive", label="synthetic-prefix property")
    count = _integer(raw, "seed_count", label="synthetic-prefix property")
    if (start, stop, count) != (_SEED_START, _SEED_STOP, _SEED_STOP - _SEED_START):
        raise ValueError("synthetic-prefix seeds must be exactly 0 through 31")
    invariants = raw.get("invariants")
    if not isinstance(invariants, list | tuple) or any(
        not isinstance(item, str) for item in invariants
    ):
        raise ValueError("synthetic-prefix invariants must be a string sequence")
    parsed_invariants = tuple(invariants)
    if parsed_invariants != _INVARIANTS:
        raise ValueError("synthetic-prefix invariants differ from the seven frozen checks")
    return _SyntheticPrefixPlan(
        seeds=tuple(range(start, stop)),
        invariants=parsed_invariants,
    )


def _report(report_date: date, report_number: int) -> dict[str, object]:
    available_at = datetime.combine(report_date, time(hour=16), UTC)
    return {
        "report_id": f"synthetic-report-{report_number}",
        "source_file": f"synthetic/report-{report_number}.xls",
        "report_year": report_date.year,
        "report_period": report_number,
        "report_date": report_date,
        "available_at": available_at,
        "row_count": 0,
        "row_report_date_mismatch_count": 0,
        "row_report_date_before_count": 0,
        "row_report_date_after_count": 0,
        "deformation_row_count": 0,
        "fluid_row_count": 0,
        "electromagnetic_row_count": 0,
        "cross_fault_row_count": 0,
    }


def _observation(
    report: Mapping[str, object],
    *,
    observation_id: str,
    anomaly_id: str,
    station_id: str,
    measurement: str,
    longitude: float,
    latitude: float,
    available_at: datetime,
) -> dict[str, object]:
    report_date = report.get("report_date")
    source_file = report.get("source_file")
    if isinstance(report_date, datetime) or not isinstance(report_date, date):
        raise AssertionError("generated report date is invalid")
    if not isinstance(source_file, str):
        raise AssertionError("generated report source file is invalid")
    return {
        "observation_id": observation_id,
        "anomaly_id": anomaly_id,
        "identity_complete": True,
        "report_date": report_date,
        "source_file": source_file,
        "available_at": available_at,
        "station_id": station_id,
        "longitude": longitude,
        "latitude": latitude,
        "discipline": "deformation",
        "measurement": measurement,
        "start_time": datetime.combine(
            report_date - timedelta(days=28),
            time.min,
            _LOCAL_TIMEZONE,
        ),
        "is_listed": True,
        "report_state": _CONTINUED_REPORT_STATE,
        "reported_end_time": None,
        "right_censored": True,
        "reliability_flags": (),
    }


def _generated_datetime(value: object, *, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise AssertionError(f"generated {label} is not an aware datetime")
    return value.astimezone(UTC)


def _seeded_rng(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed))


def _generate_trajectory(seed: int) -> _SyntheticTrajectory:
    rng = _seeded_rng(seed)
    first_date = date(2024, 1, 1) + timedelta(days=7 * (seed % 4))
    reports = [_report(first_date + timedelta(days=7 * index), index + 1) for index in range(6)]
    observations: list[dict[str, object]] = []
    for period_index, report in enumerate(reports):
        report_available_at = _generated_datetime(
            report["available_at"],
            label="report available_at",
        )
        for entity_index in range(3):
            present = entity_index == 0 or period_index == 0 or bool(rng.integers(0, 2))
            if not present:
                continue
            observations.append(
                _observation(
                    report,
                    observation_id=(f"seed-{seed}-entity-{entity_index}-period-{period_index}"),
                    anomaly_id=f"seed-{seed}-entity-{entity_index}",
                    station_id=f"station-{seed}-{entity_index}",
                    measurement=f"measurement-{entity_index}",
                    longitude=102.0 + 0.2 * entity_index + 0.001 * seed,
                    latitude=33.0 + 0.15 * entity_index,
                    available_at=report_available_at,
                )
            )

    boundary_report = reports[_BOUNDARY_ISSUE_INDEX]
    boundary_issue_time = _generated_datetime(
        boundary_report["available_at"],
        label="boundary issue time",
    )
    equal_entity = f"seed-{seed}-equal-boundary"
    after_entity = f"seed-{seed}-after-boundary"
    observations.extend(
        (
            _observation(
                boundary_report,
                observation_id=f"seed-{seed}-equal-boundary-observation",
                anomaly_id=equal_entity,
                station_id=f"station-{seed}-equal",
                measurement="measurement-equal",
                longitude=104.0 + 0.001 * seed,
                latitude=34.0,
                available_at=boundary_issue_time,
            ),
            _observation(
                boundary_report,
                observation_id=f"seed-{seed}-after-boundary-observation",
                anomaly_id=after_entity,
                station_id=f"station-{seed}-after",
                measurement="measurement-after",
                longitude=105.0 + 0.001 * seed,
                latitude=35.0,
                available_at=boundary_issue_time + timedelta(microseconds=1),
            ),
        )
    )

    for report in reports:
        source_file = report["source_file"]
        row_count = sum(item["source_file"] == source_file for item in observations)
        report["row_count"] = row_count
        report["deformation_row_count"] = row_count

    return _SyntheticTrajectory(
        reports=tuple(reports),
        observations=tuple(observations),
        issue_times=tuple(
            _generated_datetime(report["available_at"], label="issue time") for report in reports
        ),
        boundary_issue_time=boundary_issue_time,
        equal_boundary_entity_id=equal_entity,
        after_boundary_entity_id=after_entity,
    )


def _states_at(
    states: tuple[AnomalyState, ...],
    issue_time: datetime,
) -> tuple[AnomalyState, ...]:
    return tuple(state for state in states if state.issue_time_utc == issue_time)


def _prefix_payload_matches(
    trajectory: _SyntheticTrajectory,
    full_states: tuple[AnomalyState, ...],
) -> bool:
    for issue_time in trajectory.issue_times:
        prefix_reports = tuple(
            report
            for report in trajectory.reports
            if _generated_datetime(report["available_at"], label="report available_at")
            <= issue_time
        )
        prefix_observations = tuple(
            observation
            for observation in trajectory.observations
            if _generated_datetime(
                observation["available_at"],
                label="observation available_at",
            )
            <= issue_time
        )
        prefix_states = build_anomaly_state_history(prefix_observations, prefix_reports)
        if _states_at(full_states, issue_time) != _states_at(prefix_states, issue_time):
            return False
    return True


def _future_mutations_preserve_prefix(
    trajectory: _SyntheticTrajectory,
    full_states: tuple[AnomalyState, ...],
) -> bool:
    for issue_index, issue_time in enumerate(trajectory.issue_times[:-1]):
        mutated_observations: list[dict[str, object]] = []
        for original in trajectory.observations:
            mutation = dict(original)
            available_at = _generated_datetime(
                mutation["available_at"],
                label="observation available_at",
            )
            if available_at > issue_time:
                longitude = mutation["longitude"]
                if isinstance(longitude, bool) or not isinstance(longitude, int | float):
                    raise AssertionError("generated longitude is invalid")
                mutation["longitude"] = float(longitude) + 0.25
                mutation["station_id"] = f"{mutation['station_id']}-future-mutation-{issue_index}"
                mutation["measurement"] = f"{mutation['measurement']}-future-mutation-{issue_index}"
                mutation["is_listed"] = not cast(bool, mutation["is_listed"])
            mutated_observations.append(mutation)

        mutated_reports: list[dict[str, object]] = []
        for original in trajectory.reports:
            mutation = dict(original)
            available_at = _generated_datetime(
                mutation["available_at"],
                label="report available_at",
            )
            if available_at > issue_time:
                mutation["report_id"] = f"{mutation['report_id']}-future-mutation-{issue_index}"
            mutated_reports.append(mutation)
        mutated_states = build_anomaly_state_history(mutated_observations, mutated_reports)
        original_prefix = tuple(
            state for state in full_states if state.issue_time_utc <= issue_time
        )
        mutated_prefix = tuple(
            state for state in mutated_states if state.issue_time_utc <= issue_time
        )
        if original_prefix != mutated_prefix:
            return False
    return True


def _permutation_preserves_states(
    seed: int,
    trajectory: _SyntheticTrajectory,
    full_states: tuple[AnomalyState, ...],
) -> bool:
    rng = _seeded_rng(seed + 10_000)
    report_order = rng.permutation(len(trajectory.reports))
    observation_order = rng.permutation(len(trajectory.observations))
    reports = tuple(trajectory.reports[int(index)] for index in report_order)
    observations = tuple(trajectory.observations[int(index)] for index in observation_order)
    return build_anomaly_state_history(observations, reports) == full_states


def _equal_boundary_is_included(
    trajectory: _SyntheticTrajectory,
    full_states: tuple[AnomalyState, ...],
) -> bool:
    matches = tuple(
        state
        for state in _states_at(full_states, trajectory.boundary_issue_time)
        if state.anomaly_id == trajectory.equal_boundary_entity_id
    )
    return (
        len(matches) == 1
        and matches[0].current_report_listed
        and matches[0].max_source_available_at == trajectory.boundary_issue_time
    )


def _after_boundary_is_excluded(
    trajectory: _SyntheticTrajectory,
    full_states: tuple[AnomalyState, ...],
) -> bool:
    at_boundary = _states_at(full_states, trajectory.boundary_issue_time)
    absent_at_boundary = all(
        state.anomaly_id != trajectory.after_boundary_entity_id for state in at_boundary
    )
    later_present = any(
        state.anomaly_id == trajectory.after_boundary_entity_id
        and state.issue_time_utc > trajectory.boundary_issue_time
        for state in full_states
    )
    return absent_at_boundary and later_present


def _spatial_entities(
    xy_m: np.ndarray[tuple[int, int], np.dtype[np.float64]], seed: int
) -> SpatialEntityArrays:
    size = xy_m.shape[0]
    false = np.zeros(size, dtype=np.bool_)
    return SpatialEntityArrays(
        xy_m=xy_m,
        listed=np.ones(size, dtype=np.bool_),
        source_new=false,
        first_seen=false,
        explicit_end=false,
        not_continued=false,
        relisted=false,
        right_censored=np.ones(size, dtype=np.bool_),
        reliability_high=np.ones(size, dtype=np.bool_),
        reliability_cautious=false,
        station_id=np.asarray(
            [f"synthetic-station-{seed}-{index}" for index in range(size)],
            dtype=object,
        ),
        measurement_id=np.asarray(
            [f"synthetic-measurement-{seed}-{index}" for index in range(size)],
            dtype=object,
        ),
        discipline_code=np.zeros(size, dtype=np.int64),
        age_days=np.ones(size, dtype=np.float64),
        known_duration_days=np.full(size, np.nan, dtype=np.float64),
    )


def _closed_ball_counts_are_nested(seed: int) -> bool:
    rng = _seeded_rng(seed + 20_000)
    radii_m = np.concatenate(
        (
            np.asarray([0.0, 50_000.0, 100_000.0, 200_000.0, 300_000.0, 500_000.0]),
            rng.uniform(0.0, 650_000.0, size=12),
        )
    )
    angles = rng.uniform(0.0, 2.0 * np.pi, size=radii_m.size)
    xy_m = np.column_stack((radii_m * np.cos(angles), radii_m * np.sin(angles))).astype(np.float64)
    features = compute_spatial_features(
        np.asarray([[0.0, 0.0]], dtype=np.float64),
        _spatial_entities(xy_m, seed),
    ).radius_features["listed_count"][0]
    return bool(np.all(np.diff(features) >= -1e-12))


def _local_gap_does_not_mask_other_cell(seed: int) -> bool:
    entity_xy = np.asarray([[4_000_000.0, 0.0]], dtype=np.float64)
    query_xy = np.asarray([[0.0, 0.0], [4_000_000.0, 0.0]], dtype=np.float64)
    spatial = compute_spatial_features(query_xy, _spatial_entities(entity_xy, seed))
    local = local_coverage_features(spatial, spatial)
    ratio_names = (
        "current_to_trailing_station_reporting_coverage_proxy",
        "current_to_trailing_measurement_reporting_coverage_proxy",
    )
    for values, reasons in (
        (local.radius, local.radius_null_reason),
        (local.gaussian, local.gaussian_null_reason),
    ):
        for name in ratio_names:
            ratio = values[name]
            reason = reasons[name]
            if not np.isnan(ratio[0]).all():
                return False
            if not np.all(reason[0] == int(NullReasonCode.ZERO_DENOMINATOR)):
                return False
            if not np.allclose(ratio[1], 1.0, rtol=0.0, atol=1e-12):
                return False
            if not np.all(reason[1] == int(NullReasonCode.VALID)):
                return False
    return True


def _evaluate_seed(seed: int, invariants: tuple[str, ...]) -> tuple[bool, ...]:
    trajectory = _generate_trajectory(seed)
    full_states = build_anomaly_state_history(
        trajectory.observations,
        trajectory.reports,
    )
    outcomes = {
        "full_input_equals_available_prefix_scientific_payload": _prefix_payload_matches(
            trajectory,
            full_states,
        ),
        "future_mutations_do_not_change_prior_snapshot": _future_mutations_preserve_prefix(
            trajectory,
            full_states,
        ),
        "input_permutation_does_not_change_identity_or_values": _permutation_preserves_states(
            seed,
            trajectory,
            full_states,
        ),
        "available_at_equal_issue_is_included": _equal_boundary_is_included(
            trajectory,
            full_states,
        ),
        "available_at_after_issue_is_excluded": _after_boundary_is_excluded(
            trajectory,
            full_states,
        ),
        "multiscale_closed_ball_counts_are_nested": _closed_ball_counts_are_nested(seed),
        "local_reporting_gap_does_not_mask_unrelated_cells": (
            _local_gap_does_not_mask_other_cell(seed)
        ),
    }
    if set(outcomes) != set(invariants):
        raise AssertionError("synthetic-prefix invariant implementation drifted")
    return tuple(outcomes[name] for name in invariants)


def run_synthetic_prefix_audit(
    config: AnomalyHistoryConfig,
) -> SyntheticPrefixAuditResult:
    """Run all 32 synthetic seeds and return aggregate counts only."""

    plan = _parse_plan(config)
    check_count = 0
    failure_count = 0
    for seed in plan.seeds:
        outcomes = _evaluate_seed(seed, plan.invariants)
        check_count += len(outcomes)
        failure_count += sum(not outcome for outcome in outcomes)
    return SyntheticPrefixAuditResult(
        passed=failure_count == 0,
        seed_count=len(plan.seeds),
        invariant_count=len(plan.invariants),
        check_count=check_count,
        failure_count=failure_count,
    )


__all__ = [
    "SyntheticPrefixAuditResult",
    "run_synthetic_prefix_audit",
]
