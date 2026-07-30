"""Purely synthetic same-path acceptance for the Stage 2S code snapshot."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from seismoflux.background.artifacts import canonical_json_bytes
from seismoflux.stage2s.contracts import (
    AlarmMask,
    FitEventOrder,
    NormalizedSpatialDensity,
    SharedRate,
    SpatialGrid,
    SpatialQuadratureFamily,
    Stage2SModels,
)
from seismoflux.stage2s.evaluation import (
    CONTRASTS,
    HORIZONS,
    METRICS,
    BootstrapFamilies,
    CellScore,
    EventBlock,
    GateAssessment,
    LatencyMetrics,
    MetricKey,
    Model,
    RegionContribution,
    RegionRobustness,
    SequenceDiagnostic,
    SequenceEvent,
    bootstrap_families,
    build_sequence_closure_evidence,
    compute_region_robustness,
    compute_sequence_diagnostic,
    descriptive_sp_minus_s0_point_estimates,
    evaluate_stage2s_gate,
    score_fold_horizon,
)
from seismoflux.stage2s.protocol import Stage2SProtocolBundle
from seismoflux.stage2s.records import (
    Stage2SSyntheticAcceptanceRecord,
    Stage2SWholeRunRecord,
)
from seismoflux.stage2s.seals import CatalogRoleRow, RoleSession, SealedRecord
from seismoflux.stage2s.spatial import (
    build_normalized_kde,
    build_recent_component,
    build_stage2s_models,
    estimate_shared_m5_6_rate,
    event_cell_index_25km,
    fit_alpha,
    select_alarm_prefix,
)

_FOLDS = (1, 2, 3)
_DELAYS = (0, 1, 7)
_MODEL_ORDER: tuple[Model, ...] = ("S0", "S1", "SP")
_BOOTSTRAP_COLUMN_ORDER: tuple[MetricKey, ...] = (
    ("S1_minus_S0", "IG"),
    ("S1_minus_S0", "recall"),
    ("S1_minus_SP", "IG"),
    ("S1_minus_SP", "recall"),
)


@dataclass(frozen=True, slots=True)
class _SyntheticEvent:
    event_id: str
    origin_time_utc: datetime
    available_at: datetime
    longitude: float
    latitude: float
    x_km: float
    y_km: float
    magnitude: float
    inside_study_area: bool
    fold_index: int | None
    role: str

    def raw_mapping(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "origin_time_utc": self.origin_time_utc,
            "available_at": self.available_at,
            "longitude": self.longitude,
            "latitude": self.latitude,
            "magnitude": self.magnitude,
            "inside_study_area": self.inside_study_area,
        }


@dataclass(frozen=True, slots=True)
class _FoldFixture:
    fold_index: int
    issue_date: str
    issue_time_utc: datetime
    fit_issue_time_utc: datetime
    fit_cutoff_utc: datetime
    fit_target_ids: tuple[str, ...]
    assessment_target_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _PredictionVariant:
    delay_days: int
    models: Stage2SModels
    alarms: Mapping[Model, AlarmMask]
    alpha_r: float
    alpha_p: float


@dataclass(frozen=True, slots=True)
class _FoldRun:
    fixture: _FoldFixture
    shared_rate: SharedRate
    variants: Mapping[int, _PredictionVariant]
    fit_receipt: SealedRecord
    issue_seal: SealedRecord
    fold_seal: SealedRecord


def _grid(
    *,
    cell_size_km: float,
    side_count: int,
) -> SpatialGrid:
    rows = np.repeat(np.arange(side_count, dtype=np.int64), side_count)
    columns = np.tile(np.arange(side_count, dtype=np.int64), side_count)
    query_xy = np.column_stack(
        (
            (columns.astype(np.float64) + 0.5) * cell_size_km,
            (rows.astype(np.float64) + 0.5) * cell_size_km,
        )
    )
    areas = np.full(side_count * side_count, cell_size_km**2, dtype=np.float64)
    size_text = str(cell_size_km).replace(".", "_")
    cell_ids = tuple(
        f"synthetic_{size_text}_r{int(row):03d}_c{int(column):03d}"
        for row, column in zip(rows, columns, strict=True)
    )
    identity = hashlib.sha256(
        canonical_json_bytes(
            {
                "role": "stage2s_synthetic_target_independent_grid",
                "cell_size_km": cell_size_km,
                "side_count": side_count,
                "cell_ids": cell_ids,
            }
        )
    ).hexdigest()
    return SpatialGrid(
        grid_id=identity,
        cell_size_km=cell_size_km,
        cell_ids=cell_ids,
        rows=rows,
        columns=columns,
        query_xy_km=query_xy,
        clipped_area_km2=areas,
    )


def _quadrature_family() -> SpatialQuadratureFamily:
    return SpatialQuadratureFamily(
        grids=(
            _grid(cell_size_km=50.0, side_count=20),
            _grid(cell_size_km=25.0, side_count=40),
            _grid(cell_size_km=12.5, side_count=80),
        )
    )


def _event(
    event_id: str,
    *,
    origin: datetime,
    x_km: float,
    y_km: float,
    magnitude: float,
    fold_index: int | None,
    role: str,
) -> _SyntheticEvent:
    return _SyntheticEvent(
        event_id=event_id,
        origin_time_utc=origin,
        available_at=origin,
        longitude=100.0 + x_km / 100.0,
        latitude=25.0 + y_km / 100.0,
        x_km=x_km,
        y_km=y_km,
        magnitude=magnitude,
        inside_study_area=True,
        fold_index=fold_index,
        role=role,
    )


def _fixture() -> tuple[
    tuple[_SyntheticEvent, ...],
    tuple[_FoldFixture, ...],
    tuple[tuple[float, float], ...],
]:
    events: list[_SyntheticEvent] = []
    s0_xy = tuple(
        (50.0 + 100.0 * column, 50.0 + 100.0 * row) for row in range(10) for column in range(10)
    )
    base = datetime(2019, 1, 1, tzinfo=UTC)
    for index, (x_km, y_km) in enumerate(s0_xy):
        events.append(
            _event(
                f"s0-{index:03d}",
                origin=base + timedelta(hours=index),
                x_km=x_km,
                y_km=y_km,
                magnitude=4.2,
                fold_index=None,
                role="s0_source",
            )
        )
    target_xy = (
        (837.5, 837.5),
        (787.5, 887.5),
        (887.5, 787.5),
        (862.5, 762.5),
    )
    opposite_xy = (
        (137.5, 137.5),
        (187.5, 112.5),
        (112.5, 187.5),
        (212.5, 162.5),
    )
    issue_specs = (
        ("2022-10-20", datetime(2022, 10, 19, 16, tzinfo=UTC)),
        ("2023-01-19", datetime(2023, 1, 18, 16, tzinfo=UTC)),
        ("2023-04-20", datetime(2023, 4, 19, 16, tzinfo=UTC)),
    )
    folds: list[_FoldFixture] = []
    for fold_index, (issue_date, issue_time) in enumerate(issue_specs, start=1):
        fit_issue = issue_time - timedelta(days=21)
        fit_cutoff = fit_issue + timedelta(days=7)
        for index in range(12):
            x_km, y_km = target_xy[index % len(target_xy)]
            events.append(
                _event(
                    f"f{fold_index}-fit-recent-{index:02d}",
                    origin=fit_issue - timedelta(days=10, minutes=index),
                    x_km=x_km,
                    y_km=y_km,
                    magnitude=4.2,
                    fold_index=fold_index,
                    role="fit_recent_source",
                )
            )
            old_x, old_y = opposite_xy[index % len(opposite_xy)]
            events.append(
                _event(
                    f"f{fold_index}-fit-preceding-{index:02d}",
                    origin=fit_issue - timedelta(days=35, minutes=index),
                    x_km=old_x,
                    y_km=old_y,
                    magnitude=4.2,
                    fold_index=fold_index,
                    role="fit_preceding_source",
                )
            )
            events.append(
                _event(
                    f"f{fold_index}-fit-target-{index:02d}",
                    origin=fit_issue + timedelta(days=1, minutes=index),
                    x_km=x_km,
                    y_km=y_km,
                    magnitude=5.2,
                    fold_index=fold_index,
                    role="fit_target",
                )
            )
            events.append(
                _event(
                    f"f{fold_index}-recent-{index:02d}",
                    origin=issue_time - timedelta(days=10, minutes=index),
                    x_km=x_km,
                    y_km=y_km,
                    magnitude=4.3,
                    fold_index=fold_index,
                    role="assessment_recent_source",
                )
            )
            events.append(
                _event(
                    f"f{fold_index}-assessment-{index:02d}",
                    origin=issue_time + timedelta(days=1, minutes=index),
                    x_km=x_km,
                    y_km=y_km,
                    magnitude=5.3,
                    fold_index=fold_index,
                    role="assessment_target",
                )
            )
        fit_target_ids = tuple(f"f{fold_index}-fit-target-{index:02d}" for index in range(12))
        target_ids = tuple(f"f{fold_index}-assessment-{index:02d}" for index in range(12))
        folds.append(
            _FoldFixture(
                fold_index=fold_index,
                issue_date=issue_date,
                issue_time_utc=issue_time,
                fit_issue_time_utc=fit_issue,
                fit_cutoff_utc=fit_cutoff,
                fit_target_ids=fit_target_ids,
                assessment_target_ids=target_ids,
            )
        )
    events.sort(key=lambda item: (item.origin_time_utc, item.event_id.encode("utf-8")))
    return tuple(events), tuple(folds), s0_xy


def _xy(events: Sequence[_SyntheticEvent]) -> NDArray[np.float64]:
    if not events:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(
        [(event.x_km, event.y_km) for event in events],
        dtype=np.float64,
    )


def _authorized_events(
    role_rows: Sequence[CatalogRoleRow],
    by_id: Mapping[str, _SyntheticEvent],
) -> tuple[_SyntheticEvent, ...]:
    return tuple(by_id[row.event_id] for row in role_rows)


def _window_events(
    events: Sequence[_SyntheticEvent],
    *,
    issue_time_utc: datetime,
    delay_days: int,
    preceding: bool,
) -> tuple[_SyntheticEvent, ...]:
    upper = issue_time_utc - timedelta(days=30) if preceding else issue_time_utc
    lower = issue_time_utc - timedelta(days=60 if preceding else 30)
    availability_cutoff = upper - timedelta(days=delay_days)
    return tuple(
        event
        for event in events
        if lower < event.origin_time_utc <= upper
        and event.available_at <= availability_cutoff
        and event.magnitude >= 4.0
        and event.inside_study_area
    )


def _model(model_set: Stage2SModels, model_id: str) -> NormalizedSpatialDensity:
    if model_id == "S0":
        return model_set.s0
    if model_id == "S1":
        return model_set.s1
    if model_id == "SP":
        return model_set.sp
    raise KeyError(model_id)


def _array_sha256(values: NDArray[np.float64]) -> str:
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(values.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def _model_receipt(model: NormalizedSpatialDensity) -> dict[str, object]:
    return {
        "model_id": model.model_id,
        "alpha": model.alpha,
        "source_event_count": model.source_event_count,
        "mass_12_5km_sha256": _array_sha256(model.mass_12_5km),
        "mass_25km_sha256": _array_sha256(model.mass_25km),
        "direct_mass_25km_sha256": _array_sha256(model.direct_mass_25km),
        "direct_mass_50km_sha256": _array_sha256(model.direct_mass_50km),
        "convergence": {
            "primary_relative_count_difference": (
                model.convergence.primary_relative_count_difference
            ),
            "primary_density_l1": model.convergence.primary_density_l1,
            "diagnostic_relative_count_difference": (
                model.convergence.diagnostic_relative_count_difference
            ),
            "diagnostic_density_l1": model.convergence.diagnostic_density_l1,
            "passed": model.convergence.passed,
        },
    }


def _fit_weights(
    *,
    s0: NormalizedSpatialDensity,
    family: SpatialQuadratureFamily,
    fit_view: Sequence[_SyntheticEvent],
    fit_targets: Sequence[_SyntheticEvent],
    fit_issue_time_utc: datetime,
    delay_days: int,
) -> tuple[float, float]:
    recent_events = _window_events(
        fit_view,
        issue_time_utc=fit_issue_time_utc,
        delay_days=delay_days,
        preceding=False,
    )
    preceding_events = _window_events(
        fit_view,
        issue_time_utc=fit_issue_time_utc,
        delay_days=delay_days,
        preceding=True,
    )
    recent = build_recent_component(
        _xy(recent_events),
        family,
        component_id="R",
        empty_fallback_s0=s0,
    )
    preceding = build_recent_component(
        _xy(preceding_events),
        family,
        component_id="RP",
        empty_fallback_s0=s0,
    )
    target_xy = _xy(fit_targets)
    order = FitEventOrder(
        origin_time_ns=np.asarray(
            [int(event.origin_time_utc.timestamp() * 1_000_000_000) for event in fit_targets],
            dtype=np.int64,
        ),
        event_ids=tuple(event.event_id for event in fit_targets),
    )
    q0 = s0.density(target_xy[:, 0], target_xy[:, 1])
    alpha_r = fit_alpha(
        q0,
        recent.density(target_xy[:, 0], target_xy[:, 1]),
        order,
    )
    alpha_p = fit_alpha(
        q0,
        preceding.density(target_xy[:, 0], target_xy[:, 1]),
        order,
    )
    return alpha_r.alpha, alpha_p.alpha


def _prediction_variant(
    *,
    delay_days: int,
    s0: NormalizedSpatialDensity,
    family: SpatialQuadratureFamily,
    source_view: Sequence[_SyntheticEvent],
    issue_time_utc: datetime,
    alpha_r: float,
    alpha_p: float,
) -> _PredictionVariant:
    recent = build_recent_component(
        _xy(
            _window_events(
                source_view,
                issue_time_utc=issue_time_utc,
                delay_days=delay_days,
                preceding=False,
            )
        ),
        family,
        component_id="R",
        empty_fallback_s0=s0,
    )
    preceding = build_recent_component(
        _xy(
            _window_events(
                source_view,
                issue_time_utc=issue_time_utc,
                delay_days=delay_days,
                preceding=True,
            )
        ),
        family,
        component_id="RP",
        empty_fallback_s0=s0,
    )
    models = build_stage2s_models(
        s0,
        recent,
        preceding,
        alpha_r=alpha_r,
        alpha_p=alpha_p,
    )
    alarms = {model_id: select_alarm_prefix(_model(models, model_id)) for model_id in _MODEL_ORDER}
    return _PredictionVariant(
        delay_days=delay_days,
        models=models,
        alarms=alarms,
        alpha_r=alpha_r,
        alpha_p=alpha_p,
    )


def _score_variant(
    *,
    fold_index: int,
    horizon_days: int,
    targets: Sequence[_SyntheticEvent],
    variant: _PredictionVariant,
    shared_rate: SharedRate,
) -> CellScore:
    xy = _xy(targets)
    event_ids = tuple(event.event_id for event in targets)
    log_density = {
        model_id: _model(variant.models, model_id).log_density(
            xy[:, 0],
            xy[:, 1],
        )
        for model_id in _MODEL_ORDER
    }
    hits: dict[str, NDArray[np.bool_]] = {}
    for model_id in _MODEL_ORDER:
        selected = set(variant.alarms[model_id].selected_cell_ids)
        grid = _model(variant.models, model_id).grid_family.at(25.0)
        lookup = {
            (int(row), int(column)): identifier
            for identifier, row, column in zip(
                grid.cell_ids,
                grid.rows,
                grid.columns,
                strict=True,
            )
        }
        hits[model_id] = np.asarray(
            [
                lookup[event_cell_index_25km(event.x_km * 1_000.0, event.y_km * 1_000.0)]
                in selected
                for event in targets
            ],
            dtype=np.bool_,
        )
    masses = {
        model_id: np.asarray(
            [_model(variant.models, model_id).mass_25km],
            dtype=np.float64,
        )
        for model_id in _MODEL_ORDER
    }
    return score_fold_horizon(
        fold_index=fold_index,
        horizon_days=horizon_days,
        event_ids=event_ids,
        supported_ig=np.ones(len(targets), dtype=np.bool_),
        log_density_by_model=cast(Mapping[str, object], log_density),
        hit_by_model=hits,
        operational_mass_by_model=cast(Mapping[str, object], masses),
        shared_rate_per_day=shared_rate.rate_per_day,
    )


def _primary_metrics(
    scores: Mapping[tuple[int, int], CellScore],
) -> dict[MetricKey, float]:
    return {
        (contrast, metric): math.fsum(
            (score.information_gain[contrast] if metric == "IG" else score.recall_gain[contrast])
            for score in scores.values()
        )
        / 9.0
        for contrast in CONTRASTS
        for metric in METRICS
    }


def _event_blocks(
    scores: Mapping[tuple[int, int], CellScore],
    events_by_id: Mapping[str, _SyntheticEvent],
) -> tuple[EventBlock, ...]:
    blocks: list[EventBlock] = []
    for fold_index in _FOLDS:
        reference = scores[(fold_index, 7)]
        for event_index, event_id in enumerate(reference.event_ids):
            blocks.append(
                EventBlock(
                    event_id=event_id,
                    origin_time_utc=events_by_id[event_id].origin_time_utc,
                    fold_index=fold_index,
                    horizons=HORIZONS,
                    supported_ig=True,
                    ig_by_contrast_horizon={
                        (contrast, horizon): float(
                            scores[(fold_index, horizon)].ig_event_log_ratios[contrast][event_index]
                        )
                        for contrast in CONTRASTS
                        for horizon in HORIZONS
                    },
                    recall_by_contrast_horizon={
                        (contrast, horizon): float(
                            scores[(fold_index, horizon)].recall_hit_differences[contrast][
                                event_index
                            ]
                        )
                        for contrast in CONTRASTS
                        for horizon in HORIZONS
                    },
                )
            )
    return tuple(blocks)


def _zone_mapping(grid: SpatialGrid) -> dict[str, str]:
    return {
        identifier: f"zone-{(int(row) * 7 + int(column) * 11) % 39:02d}"
        for identifier, row, column in zip(
            grid.cell_ids,
            grid.rows,
            grid.columns,
            strict=True,
        )
    }


def _target_zone(
    event: _SyntheticEvent,
    *,
    grid: SpatialGrid,
    zone_by_cell: Mapping[str, str],
) -> str:
    lookup = {
        (int(row), int(column)): identifier
        for identifier, row, column in zip(
            grid.cell_ids,
            grid.rows,
            grid.columns,
            strict=True,
        )
    }
    cell = lookup[event_cell_index_25km(event.x_km * 1_000.0, event.y_km * 1_000.0)]
    return zone_by_cell[cell]


def _regional_evidence(
    *,
    scores: Mapping[tuple[int, int], CellScore],
    fold_runs: Mapping[int, _FoldRun],
    events_by_id: Mapping[str, _SyntheticEvent],
    primary_metrics: Mapping[MetricKey, float],
) -> tuple[tuple[RegionContribution, ...], RegionRobustness]:
    grid = next(iter(fold_runs.values())).variants[0].models.s0.grid_family.at(25.0)
    zone_by_cell = _zone_mapping(grid)
    zone_ids = tuple(f"zone-{index:02d}" for index in range(39))
    contributions = {
        zone_id: {(contrast, metric): 0.0 for contrast in CONTRASTS for metric in METRICS}
        for zone_id in zone_ids
    }
    event_ids_by_zone: dict[str, set[str]] = {zone_id: set() for zone_id in zone_ids}
    cell_indices_by_zone = {
        zone_id: np.asarray(
            [
                index
                for index, identifier in enumerate(grid.cell_ids)
                if zone_by_cell[identifier] == zone_id
            ],
            dtype=np.int64,
        )
        for zone_id in zone_ids
    }
    for fold_index in _FOLDS:
        run = fold_runs[fold_index]
        primary = run.variants[0]
        for horizon in HORIZONS:
            score = scores[(fold_index, horizon)]
            denominator = len(score.event_ids)
            for event_index, event_id in enumerate(score.event_ids):
                zone_id = _target_zone(
                    events_by_id[event_id],
                    grid=grid,
                    zone_by_cell=zone_by_cell,
                )
                event_ids_by_zone[zone_id].add(event_id)
                for contrast in CONTRASTS:
                    contributions[zone_id][(contrast, "IG")] += (
                        float(score.ig_event_log_ratios[contrast][event_index]) / denominator / 9.0
                    )
                    contributions[zone_id][(contrast, "recall")] += (
                        float(score.recall_hit_differences[contrast][event_index])
                        / denominator
                        / 9.0
                    )
            for zone_id in zone_ids:
                indices = cell_indices_by_zone[zone_id]
                for contrast, comparator_id in (
                    ("S1_minus_S0", "S0"),
                    ("S1_minus_SP", "SP"),
                ):
                    candidate_mass = math.fsum(
                        float(primary.models.s1.mass_25km[index]) for index in indices
                    )
                    comparator_mass = math.fsum(
                        float(_model(primary.models, comparator_id).mass_25km[index])
                        for index in indices
                    )
                    regional_compensator = (
                        run.shared_rate.rate_per_day * horizon * (candidate_mass - comparator_mass)
                    )
                    contributions[zone_id][(contrast, "IG")] -= (
                        regional_compensator / denominator / 9.0
                    )
    regions = tuple(
        RegionContribution(
            zone_id=zone_id,
            ig_event_count=len(event_ids_by_zone[zone_id]),
            recall_event_count=len(event_ids_by_zone[zone_id]),
            contributions=contributions[zone_id],
        )
        for zone_id in zone_ids
    )
    return regions, compute_region_robustness(
        regions,
        primary_metrics=primary_metrics,
    )


def _sequence_evidence(
    *,
    scores: Mapping[tuple[int, int], CellScore],
    events_by_id: Mapping[str, _SyntheticEvent],
    primary_metrics: Mapping[MetricKey, float],
) -> SequenceDiagnostic:
    contributions_by_event: dict[str, dict[MetricKey, float]] = {}
    model_hits_by_event: dict[str, dict[Model, float]] = {}
    for fold_index in _FOLDS:
        for horizon in HORIZONS:
            score = scores[(fold_index, horizon)]
            supported_count = int(np.count_nonzero(score.supported_ig))
            recall_count = len(score.event_ids)
            ig_by_contrast: dict[str, dict[str, float]] = {}
            for contrast in CONTRASTS:
                ig_by_contrast[contrast] = {
                    event_id: float(value)
                    for event_id, value in zip(
                        (
                            event_id
                            for event_id, supported in zip(
                                score.event_ids,
                                score.supported_ig,
                                strict=True,
                            )
                            if bool(supported)
                        ),
                        score.ig_event_log_ratios[contrast],
                        strict=True,
                    )
                }
            for position, (event_id, supported) in enumerate(
                zip(score.event_ids, score.supported_ig, strict=True)
            ):
                contributions = contributions_by_event.setdefault(
                    event_id,
                    {(contrast, metric): 0.0 for contrast in CONTRASTS for metric in METRICS},
                )
                model_hits = model_hits_by_event.setdefault(
                    event_id,
                    {model_id: 0.0 for model_id in _MODEL_ORDER},
                )
                for contrast in CONTRASTS:
                    if bool(supported):
                        contributions[(contrast, "IG")] += (
                            ig_by_contrast[contrast][event_id] / supported_count / 9.0
                        )
                    contributions[(contrast, "recall")] += (
                        float(score.recall_hit_differences[contrast][position]) / recall_count / 9.0
                    )
                for model_id in _MODEL_ORDER:
                    model_hits[model_id] += (
                        float(score.hit_by_model[model_id][position]) / recall_count / 9.0
                    )
    sequence_events = tuple(
        SequenceEvent(
            event_id=event_id,
            origin_time_utc=events_by_id[event_id].origin_time_utc,
            longitude=events_by_id[event_id].longitude,
            latitude=events_by_id[event_id].latitude,
            contributions=contributions,
            model_hit_contributions=model_hits_by_event[event_id],
        )
        for event_id, contributions in sorted(
            contributions_by_event.items(),
            key=lambda item: (
                events_by_id[item[0]].origin_time_utc,
                item[0].encode("utf-8"),
            ),
        )
    )
    return compute_sequence_diagnostic(
        sequence_events,
        primary_metrics=primary_metrics,
        closure=build_sequence_closure_evidence(scores),
    )


def _cell_record(score: CellScore) -> dict[str, object]:
    return {
        "fold_index": score.fold_index,
        "horizon_days": score.horizon_days,
        "issue_count": score.issue_count,
        "event_ids": list(score.event_ids),
        "supported_ig": [bool(value) for value in score.supported_ig],
        "hit_by_model": {
            model_id: [bool(value) for value in score.hit_by_model[model_id]]
            for model_id in _MODEL_ORDER
        },
        "ig_event_log_ratios": {
            contrast: [float(value) for value in score.ig_event_log_ratios[contrast]]
            for contrast in CONTRASTS
        },
        "recall_hit_differences": {
            contrast: [float(value) for value in score.recall_hit_differences[contrast]]
            for contrast in CONTRASTS
        },
        "compensator_differences": dict(score.compensator_differences),
        "information_gain": dict(score.information_gain),
        "recall_gain": dict(score.recall_gain),
    }


def _bootstrap_records(
    bootstrap: BootstrapFamilies,
) -> tuple[dict[str, object], tuple[tuple[float, float, float, float], ...]]:
    summary = {
        "entropy_uint128": bootstrap.entropy_uint128,
        "replications": 2000,
        "column_order": [f"{contrast}:{metric}" for contrast, metric in _BOOTSTRAP_COLUMN_ORDER],
        "intervals": {
            f"{contrast}:{metric}": {
                "point": bootstrap.intervals[(contrast, metric)].point,
                "lower": bootstrap.intervals[(contrast, metric)].lower,
                "upper": bootstrap.intervals[(contrast, metric)].upper,
            }
            for contrast, metric in _BOOTSTRAP_COLUMN_ORDER
        },
    }
    row_values: list[tuple[float, float, float, float]] = []
    for replicate_index in range(2000):
        values = tuple(
            bootstrap.intervals[key].replicates[replicate_index] for key in _BOOTSTRAP_COLUMN_ORDER
        )
        row_values.append((values[0], values[1], values[2], values[3]))
    return summary, tuple(row_values)


def _regional_record(
    regions: Sequence[RegionContribution],
    robustness: RegionRobustness,
) -> dict[str, object]:
    return {
        "regions": [
            {
                "zone_id": region.zone_id,
                "ig_event_count": region.ig_event_count,
                "recall_event_count": region.recall_event_count,
                "contributions": {
                    f"{contrast}:{metric}": region.contributions[(contrast, metric)]
                    for contrast in CONTRASTS
                    for metric in METRICS
                },
            }
            for region in regions
        ],
        "results": {
            f"{contrast}:{metric}": {
                "event_bearing_zone_count": result.event_bearing_zone_count,
                "positive_event_bearing_zone_count": (result.positive_event_bearing_zone_count),
                "strongest_positive_zone_id": result.strongest_positive_zone_id,
                "strongest_positive_contribution": (result.strongest_positive_contribution),
                "leave_strongest_out_residual": result.leave_strongest_out_residual,
                "passed": result.passed,
            }
            for (contrast, metric), result in robustness.results.items()
        },
        "failures": list(robustness.failures),
        "passed": robustness.passed,
    }


def _sequence_record(sequence: SequenceDiagnostic) -> dict[str, object]:
    largest_count = next(
        component
        for component in sequence.components
        if component.component_id == sequence.largest_count_component_id
    )

    def component_record(component: object) -> dict[str, object]:
        typed = cast(Any, component)
        return {
            "component_id": typed.component_id,
            "event_ids": list(typed.event_ids),
            "event_count": len(typed.event_ids),
            "event_fraction": typed.event_fraction,
            "origin_time_span_days": typed.origin_time_span_days,
            "max_pairwise_geodesic_distance_km": (typed.max_pairwise_geodesic_distance_km),
            "contributions": {
                f"{contrast}:{metric}": typed.contributions[(contrast, metric)]
                for contrast in CONTRASTS
                for metric in METRICS
            },
            "model_hits": {
                model_id: {
                    "raw": typed.model_hit_contributions[model_id],
                    "fraction": typed.model_hit_fractions[model_id],
                }
                for model_id in _MODEL_ORDER
            },
            "information_gain": {
                contrast: {
                    "raw": typed.contributions[(contrast, "IG")],
                    "fraction": typed.ig_fractions[contrast],
                }
                for contrast in CONTRASTS
            },
        }

    return {
        "component_count": len(sequence.components),
        "event_resampling_unit_count": sum(
            len(component.event_ids) for component in sequence.components
        ),
        "global_residual": {
            f"{contrast}:{metric}": sequence.global_residual[(contrast, metric)]
            for contrast in CONTRASTS
            for metric in METRICS
        },
        "primary_model_recall": dict(sequence.primary_model_recall),
        "components": [component_record(component) for component in sequence.components],
        "largest_count_component_id": sequence.largest_count_component_id,
        "largest_count_component": {
            **component_record(largest_count),
            "leave_out": {
                f"{contrast}:{metric}": sequence.leave_largest_count_out[(contrast, metric)]
                for contrast in CONTRASTS
                for metric in METRICS
            },
        },
        "largest_gain_component_id": {
            f"{contrast}:{metric}": component_id
            for (contrast, metric), component_id in sequence.largest_gain_component_id.items()
        },
        "largest_gain_component": {
            f"{contrast}:{metric}": {
                "component_id": sequence.largest_gain_component_id[(contrast, metric)],
                "raw_contribution": next(
                    component.contributions[(contrast, metric)]
                    for component in sequence.components
                    if component.component_id
                    == sequence.largest_gain_component_id[(contrast, metric)]
                ),
                "leave_out": sequence.leave_largest_gain_out[(contrast, metric)],
            }
            for contrast in CONTRASTS
            for metric in METRICS
        },
        "leave_out_residual": {
            f"{contrast}:{metric}": value
            for (contrast, metric), value in sequence.leave_largest_gain_out.items()
        },
        "leave_largest_count_out": {
            f"{contrast}:{metric}": value
            for (contrast, metric), value in sequence.leave_largest_count_out.items()
        },
        "leave_largest_gain_out": {
            f"{contrast}:{metric}": value
            for (contrast, metric), value in sequence.leave_largest_gain_out.items()
        },
        "claim_limited": sequence.claim_limited,
        "interpretation_limit": sequence.interpretation_limit,
    }


def _gate_record(assessment: GateAssessment) -> dict[str, object]:
    passed = assessment.decision.status == "passed_development_signal"
    return {
        "status": assessment.decision.status,
        "reasons": list(assessment.decision.reasons),
        "supported_event_union_count": assessment.supported_event_union_count,
        "recall_event_union_count": assessment.recall_event_union_count,
        "fold_macros": {
            f"{contrast}:{metric}:fold{fold_index}": value
            for (contrast, metric, fold_index), value in assessment.fold_macros.items()
        },
        "horizon_macros": {
            f"{contrast}:{metric}:h{horizon}": value
            for (contrast, metric, horizon), value in assessment.horizon_macros.items()
        },
        "overall_macros": {
            f"{contrast}:{metric}": value
            for (contrast, metric), value in assessment.overall_macros.items()
        },
        "claim_limited": assessment.claim_limited,
        "interpretation_limit": assessment.interpretation_limit,
        "interpretation_scope": (
            "sequence_associated_continuation_only"
            if assessment.claim_limited
            else (
                "broad_regional_gain_not_sequence_limited"
                if passed
                else "no_sequence_interpretation_limit"
            )
        ),
        "science_value_category": "necessary_enabler",
        "direct_prediction_improvement": "none",
        "evidence_scope": "synthetic_engineering_acceptance_not_prediction_improvement",
    }


def run_synthetic_acceptance(
    *,
    protocol: Stage2SProtocolBundle,
    scratch_root: Path,
) -> Stage2SSyntheticAcceptanceRecord:
    """Execute the complete target-free synthetic chain with immutable seals."""

    if not scratch_root.is_absolute():
        raise ValueError("scratch_root must be absolute")
    family = _quadrature_family()
    events, fold_fixtures, s0_xy = _fixture()
    by_id = {event.event_id: event for event in events}
    raw_rows = [event.raw_mapping() for event in events]
    fixture_sha256 = hashlib.sha256(
        canonical_json_bytes(
            {
                "fixture_version": 1,
                "grid_ids": [grid.grid_id for grid in family.grids],
                "event_ids": [event.event_id for event in events],
                "fold_issues": [fold.issue_date for fold in fold_fixtures],
            }
        )
    ).hexdigest()
    s0 = build_normalized_kde(
        np.asarray(s0_xy, dtype=np.float64),
        family,
        model_id="S0",
    )
    issue_dates = {fold.fold_index: (fold.issue_date,) for fold in fold_fixtures}
    session = RoleSession(
        seal_root=(scratch_root / "causal_seismicity_screen").resolve(),
        issue_dates_by_fold=issue_dates,
    )
    fold_runs: dict[int, _FoldRun] = {}
    fit_records: list[dict[str, object]] = []
    issue_records: list[dict[str, object]] = []
    fit_seal_hashes: list[str] = []
    issue_seal_hashes: list[str] = []
    fold_seal_hashes: list[str] = []
    for fold in fold_fixtures:
        fit_rows = session.open_fit_view(
            fold_index=fold.fold_index,
            rows=raw_rows,
            fit_cutoff_utc=fold.fit_cutoff_utc,
        )
        fit_view = _authorized_events(fit_rows, by_id)
        fit_targets = tuple(by_id[event_id] for event_id in fold.fit_target_ids)
        weights = {
            delay_days: _fit_weights(
                s0=s0,
                family=family,
                fit_view=fit_view,
                fit_targets=fit_targets,
                fit_issue_time_utc=fold.fit_issue_time_utc,
                delay_days=delay_days,
            )
            for delay_days in _DELAYS
        }
        shared_rate = estimate_shared_m5_6_rate(
            fold.fit_target_ids,
            total_exposure_days=7.0,
        )
        fit_receipt = session.seal_fit(
            fold_index=fold.fold_index,
            bindings={
                "fixture_sha256": fixture_sha256,
                "fit_cutoff_utc": fold.fit_cutoff_utc.isoformat(),
                "fit_target_event_ids": list(fold.fit_target_ids),
                "weights_by_delay": {
                    str(delay): {
                        "alpha_R": weights[delay][0],
                        "alpha_P": weights[delay][1],
                    }
                    for delay in _DELAYS
                },
                "shared_rate_per_day": shared_rate.rate_per_day,
                "grid_ids_50_25_12_5km": [grid.grid_id for grid in family.grids],
            },
        )
        fit_seal_hashes.append(fit_receipt.file_sha256)
        source_rows = session.open_causal_source_view(
            fold_index=fold.fold_index,
            issue_date=fold.issue_date,
            issue_time_utc=fold.issue_time_utc,
            rows=raw_rows,
        )
        source_view = _authorized_events(source_rows, by_id)
        variants = {
            delay_days: _prediction_variant(
                delay_days=delay_days,
                s0=s0,
                family=family,
                source_view=source_view,
                issue_time_utc=fold.issue_time_utc,
                alpha_r=weights[delay_days][0],
                alpha_p=weights[delay_days][1],
            )
            for delay_days in _DELAYS
        }
        issue_binding = {
            "fixture_sha256": fixture_sha256,
            "issue_time_utc": fold.issue_time_utc.isoformat(),
            "horizons_days": list(HORIZONS),
            "variants": {
                str(delay): {
                    "models": {
                        model_id: _model_receipt(_model(variants[delay].models, model_id))
                        for model_id in _MODEL_ORDER
                    },
                    "alarms": {
                        model_id: {
                            "ranking_sha256": variants[delay].alarms[model_id].ranking_sha256,
                            "actual_area_km2": variants[delay].alarms[model_id].actual_area_km2,
                        }
                        for model_id in _MODEL_ORDER
                    },
                }
                for delay in _DELAYS
            },
        }
        issue_seal = session.seal_issue(
            fold_index=fold.fold_index,
            issue_date=fold.issue_date,
            bindings=issue_binding,
        )
        issue_seal_hashes.append(issue_seal.file_sha256)
        fold_seal = session.seal_fold(
            fold_index=fold.fold_index,
            bindings={
                "fixture_sha256": fixture_sha256,
                "fit_receipt_sha256": fit_receipt.file_sha256,
                "issue_seal_sha256": issue_seal.file_sha256,
            },
        )
        fold_seal_hashes.append(fold_seal.file_sha256)
        fold_runs[fold.fold_index] = _FoldRun(
            fixture=fold,
            shared_rate=shared_rate,
            variants=variants,
            fit_receipt=fit_receipt,
            issue_seal=issue_seal,
            fold_seal=fold_seal,
        )
        fit_records.append(
            {
                "fold_index": fold.fold_index,
                "fit_receipt_sha256": fit_receipt.file_sha256,
                "alpha_R_by_delay": {str(delay): weights[delay][0] for delay in _DELAYS},
                "alpha_P_by_delay": {str(delay): weights[delay][1] for delay in _DELAYS},
                "shared_rate_per_day": shared_rate.rate_per_day,
                "fit_event_count": shared_rate.event_count,
            }
        )
        issue_records.append(
            {
                "fold_index": fold.fold_index,
                "issue_date": fold.issue_date,
                "issue_prediction_seal_sha256": issue_seal.file_sha256,
                "actual_alarm_area_km2": {
                    f"delay{delay}:{model_id}": variants[delay].alarms[model_id].actual_area_km2
                    for delay in _DELAYS
                    for model_id in _MODEL_ORDER
                },
            }
        )
    master = session.seal_master(
        bindings={
            "fixture_sha256": fixture_sha256,
            "protocol_identity": protocol.identity_mapping(),
            "synthetic_attempt_ledger_sha256": hashlib.sha256(
                b"stage2s-synthetic-no-real-attempt"
            ).hexdigest(),
            "synthetic_target_read_receipt_sha256": hashlib.sha256(
                b"stage2s-synthetic-no-real-target-read"
            ).hexdigest(),
        }
    )
    assessment_rows = session.open_assessment_view(rows=raw_rows)
    assessment_view = _authorized_events(assessment_rows, by_id)
    assessment_ids = {event.event_id for event in assessment_view}
    if any(
        event_id not in assessment_ids
        for fold in fold_fixtures
        for event_id in fold.assessment_target_ids
    ):
        raise RuntimeError("synthetic master-sealed assessment view is incomplete")

    primary_scores: dict[tuple[int, int], CellScore] = {}
    delayed_scores: dict[int, dict[tuple[int, int], CellScore]] = {
        1: {},
        7: {},
    }
    for fold in fold_fixtures:
        targets = tuple(by_id[event_id] for event_id in fold.assessment_target_ids)
        run = fold_runs[fold.fold_index]
        for horizon in HORIZONS:
            primary_scores[(fold.fold_index, horizon)] = _score_variant(
                fold_index=fold.fold_index,
                horizon_days=horizon,
                targets=targets,
                variant=run.variants[0],
                shared_rate=run.shared_rate,
            )
            for delay_days in (1, 7):
                delayed_scores[delay_days][(fold.fold_index, horizon)] = _score_variant(
                    fold_index=fold.fold_index,
                    horizon_days=horizon,
                    targets=targets,
                    variant=run.variants[delay_days],
                    shared_rate=run.shared_rate,
                )
    blocks = _event_blocks(primary_scores, by_id)
    compensators = {
        (contrast, fold_index, horizon): primary_scores[
            (fold_index, horizon)
        ].compensator_differences[contrast]
        for contrast in CONTRASTS
        for fold_index in _FOLDS
        for horizon in HORIZONS
    }
    bootstrap = bootstrap_families(blocks, compensators=compensators)
    primary_metrics = _primary_metrics(primary_scores)
    regions, regional = _regional_evidence(
        scores=primary_scores,
        fold_runs=fold_runs,
        events_by_id=by_id,
        primary_metrics=primary_metrics,
    )
    sequence = _sequence_evidence(
        scores=primary_scores,
        events_by_id=by_id,
        primary_metrics=primary_metrics,
    )
    latency = tuple(
        LatencyMetrics(
            delay_days=delay_days,
            values=_primary_metrics(delayed_scores[delay_days]),
        )
        for delay_days in (1, 7)
    )
    gate = evaluate_stage2s_gate(
        primary_scores,
        bootstrap=bootstrap,
        regional=regional,
        latency=latency,
        sequence=sequence,
    )
    bootstrap_summary, bootstrap_rows = _bootstrap_records(bootstrap)
    whole_run = Stage2SWholeRunRecord(
        mode="synthetic_acceptance",
        identity={
            **protocol.identity_mapping(),
            "execution_role": "target_free_synthetic_code_acceptance",
            "fixture_sha256": fixture_sha256,
        },
        input_receipts={
            "synthetic_fixture_sha256": fixture_sha256,
            "real_study_area_open_count": 0,
            "real_cell_mapping_open_count": 0,
            "real_catalog_open_count": 0,
            "real_target_byte_read_count": 0,
        },
        fold_fit_summaries=fit_records,
        issue_prediction_summaries=issue_records,
        seal_chain={
            "fold_fit_receipt_sha256": fit_seal_hashes,
            "issue_prediction_seal_sha256": issue_seal_hashes,
            "fold_prediction_seal_sha256": fold_seal_hashes,
            "master_prediction_seal_sha256": master.file_sha256,
        },
        cell_scores=[
            _cell_record(primary_scores[(fold_index, horizon)])
            for fold_index in _FOLDS
            for horizon in HORIZONS
        ],
        bootstrap_summary=bootstrap_summary,
        bootstrap_rows=bootstrap_rows,
        regional_evidence=_regional_record(regions, regional),
        sequence_evidence=_sequence_record(sequence),
        descriptive_point_estimates={
            "SP_minus_S0": {
                "information_gain": descriptive_sp_minus_s0_point_estimates(gate.overall_macros)[
                    "IG"
                ],
                "recall_gain": descriptive_sp_minus_s0_point_estimates(gate.overall_macros)[
                    "recall"
                ],
                "derivation": "S1_minus_S0_minus_S1_minus_SP",
                "inferential_status": "descriptive_point_estimate_only",
                "included_in_bootstrap_ci": False,
                "included_in_gate": False,
            }
        },
        latency_evidence=[
            {
                "delay_days": item.delay_days,
                "metrics": {
                    f"{contrast}:{metric}": item.values[(contrast, metric)]
                    for contrast in CONTRASTS
                    for metric in METRICS
                },
            }
            for item in latency
        ],
        gate_evidence=_gate_record(gate),
        artifact_sha256_by_name={},
    )
    return Stage2SSyntheticAcceptanceRecord(whole_run=whole_run)


__all__ = ["run_synthetic_acceptance"]
