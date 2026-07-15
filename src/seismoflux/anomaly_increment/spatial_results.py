"""Pure-memory adapters for local, target-safe stage-4 spatial results.

This module never opens a catalogue path and never writes an output.  Forecast
frames are derived from the actual in-memory score or fitted scope against the
pre-existing 25 km grid.  Authorized retrospective targets enter only through a
separate overlay function after every target-free frame has been built.
"""

# ruff: noqa: E501, RUF001

from __future__ import annotations

import dataclasses
import math
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from html import escape
from typing import Literal, cast
from zoneinfo import ZoneInfo

import numpy as np
from numpy.typing import NDArray
from pyproj import CRS, Transformer

from seismoflux.anomaly_increment.contracts import (
    FloatArray,
    canonical_mapping_sha256,
)
from seismoflux.anomaly_increment.evaluation import frozen_same_recall_budget_grid
from seismoflux.anomaly_increment.formal_assembly import BoundTargetCellAssignments
from seismoflux.anomaly_increment.formal_execution import AuthorizedFormalMaterialization
from seismoflux.anomaly_increment.grid_features import Stage4IntegrationGrid
from seismoflux.anomaly_increment.integration import composite_midpoint_quadrature, lead_decay
from seismoflux.anomaly_increment.scoring_pipeline import (
    PRIMARY_ALARM_AREA_KM2,
    AssembledEvaluationExposure,
    AssembledProspectiveIssue,
    ExposureVariantScore,
    FittedScope,
    FittedVariant,
    PipelineResult,
    Stage4InMemoryPlan,
)
from seismoflux.anomaly_increment.spatial_dashboard import (
    DisplayContextLayer,
    DisplayStudyArea,
    ForecastGridFrame,
    RetrospectiveTargetFrame,
    build_local_spatial_dashboard_html,
)
from seismoflux.anomaly_increment.targets import Stage4TargetCatalog

_SHANGHAI = ZoneInfo("Asia/Shanghai")
_DISPLAY_HORIZONS = (7, 30, 90)
_DISPLAY_MAGNITUDE_BIN: Literal["M5_6"] = "M5_6"
_DISPLAY_VARIANT: Literal["dynamic"] = "dynamic"
_SELECTION_RULE: Literal["formal-calendar-landmarks-with-immediate-predecessors-v2"] = (
    "formal-calendar-landmarks-with-immediate-predecessors-v2"
)
_LOCAL_CLASSIFICATION: Literal["local_restricted_target_bearing_spatial_visualization"] = (
    "local_restricted_target_bearing_spatial_visualization"
)
_FORBIDDEN_PROSPECTIVE_TOKENS = (
    "catalog",
    "covered_by",
    "event_id",
    "future",
    "hit",
    "recall",
    "score",
    "target",
)


def _sha256(value: str, *, label: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _identifier(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


def _readonly_float(values: object, *, label: str) -> FloatArray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be a finite vector")
    output = np.array(result, dtype=np.float64, copy=True)
    output.setflags(write=False)
    return output


@dataclass(frozen=True, slots=True)
class SpatialCalendarSelection:
    """Score-blind display keys selected only from frozen issue calendars."""

    retrospective_keys: tuple[tuple[str, int], ...]
    retrospective_support_keys: tuple[tuple[str, int], ...]
    prospective_keys: tuple[tuple[str, int], ...]
    prospective_support_keys: tuple[tuple[str, int], ...]
    selection_sha256: str
    rule: Literal["formal-calendar-landmarks-with-immediate-predecessors-v2"] = _SELECTION_RULE
    magnitude_bin: Literal["M5_6"] = _DISPLAY_MAGNITUDE_BIN
    model_variant: Literal["dynamic"] = _DISPLAY_VARIANT
    score_or_target_consulted: Literal[False] = False

    def __post_init__(self) -> None:
        retrospective = tuple(self.retrospective_keys)
        retrospective_support = tuple(self.retrospective_support_keys)
        prospective = tuple(self.prospective_keys)
        prospective_support = tuple(self.prospective_support_keys)
        for label, keys in (
            ("retrospective", retrospective),
            ("prospective", prospective),
        ):
            if not keys or len(keys) != len(set(keys)):
                raise ValueError(f"spatial {label} calendar keys must be non-empty and unique")
            for issue_date, horizon in keys:
                date.fromisoformat(issue_date)
                if horizon not in _DISPLAY_HORIZONS:
                    raise ValueError("spatial calendar selection horizon changed")
        for label, support, primary in (
            ("retrospective", retrospective_support, retrospective),
            ("prospective", prospective_support, prospective),
        ):
            if len(support) != len(set(support)) or set(support) & set(primary):
                raise ValueError(
                    f"spatial {label} support keys must be unique and disjoint from primary keys"
                )
            for issue_date, horizon in support:
                date.fromisoformat(issue_date)
                if horizon not in _DISPLAY_HORIZONS:
                    raise ValueError("spatial calendar support horizon changed")
        if self.rule != _SELECTION_RULE or self.score_or_target_consulted is not False:
            raise ValueError("spatial selection escaped its score-blind calendar rule")
        if self.magnitude_bin != _DISPLAY_MAGNITUDE_BIN or self.model_variant != _DISPLAY_VARIANT:
            raise ValueError("spatial selection changed the frozen main display")
        expected = canonical_mapping_sha256(
            {
                "magnitude_bin": self.magnitude_bin,
                "model_variant": self.model_variant,
                "prospective_keys": [list(item) for item in prospective],
                "prospective_support_keys": [list(item) for item in prospective_support],
                "retrospective_keys": [list(item) for item in retrospective],
                "retrospective_support_keys": [list(item) for item in retrospective_support],
                "rule": self.rule,
                "score_or_target_consulted": False,
            }
        )
        if self.selection_sha256 != expected:
            raise ValueError("spatial calendar selection differs from its receipt")
        object.__setattr__(self, "retrospective_keys", retrospective)
        object.__setattr__(self, "retrospective_support_keys", retrospective_support)
        object.__setattr__(self, "prospective_keys", prospective)
        object.__setattr__(self, "prospective_support_keys", prospective_support)


def build_score_blind_calendar_selection(plan: Stage4InMemoryPlan) -> SpatialCalendarSelection:
    """Select calendar landmarks plus immediate predecessors without scores or targets."""

    retrospective: list[tuple[str, int]] = []
    retrospective_support: list[tuple[str, int]] = []
    prospective: list[tuple[str, int]] = []
    prospective_support: list[tuple[str, int]] = []
    for horizon in _DISPLAY_HORIZONS:
        formal_dates = tuple(
            sorted(
                {
                    item.issue_date
                    for item in plan.formal_scope.exposures
                    if item.horizon_days == horizon and item.magnitude_bin == _DISPLAY_MAGNITUDE_BIN
                }
            )
        )
        if not formal_dates:
            raise ValueError(f"formal calendar has no M5 dynamic display date for {horizon} days")
        positions = (0, len(formal_dates) // 2, len(formal_dates) - 1)
        primary_positions = tuple(dict.fromkeys(positions))
        primary_dates = tuple(formal_dates[index] for index in primary_positions)
        retrospective.extend((value, horizon) for value in primary_dates)
        retrospective_support.extend(
            (formal_dates[index - 1], horizon)
            for index in primary_positions
            if index > 0 and formal_dates[index - 1] not in primary_dates
        )
        prospective_dates = tuple(
            sorted(
                {
                    item.issue_date
                    for item in plan.prospective_issues
                    if item.horizon_days == horizon and item.magnitude_bin == _DISPLAY_MAGNITUDE_BIN
                }
            )
        )
        if not prospective_dates:
            raise ValueError(
                f"target-free issue calendar has no M5 display date for {horizon} days"
            )
        prospective.append((prospective_dates[-1], horizon))
        if len(prospective_dates) > 1:
            prospective_support.append((prospective_dates[-2], horizon))
    payload = {
        "magnitude_bin": _DISPLAY_MAGNITUDE_BIN,
        "model_variant": _DISPLAY_VARIANT,
        "prospective_keys": [list(item) for item in prospective],
        "prospective_support_keys": [list(item) for item in prospective_support],
        "retrospective_keys": [list(item) for item in retrospective],
        "retrospective_support_keys": [list(item) for item in retrospective_support],
        "rule": _SELECTION_RULE,
        "score_or_target_consulted": False,
    }
    return SpatialCalendarSelection(
        retrospective_keys=tuple(retrospective),
        retrospective_support_keys=tuple(retrospective_support),
        prospective_keys=tuple(prospective),
        prospective_support_keys=tuple(prospective_support),
        selection_sha256=canonical_mapping_sha256(payload),
    )


def genuine_prospective_archive_content_sha256(
    *,
    archive_id: str,
    target_blind_frame_bundle_sha256: str,
    model_version: str,
    issue_dates: Sequence[str],
    archived_at_utc: datetime,
) -> str:
    """Bind the immutable archive receipt to the exact target-blind frame bundle."""

    _identifier(archive_id, label="prospective archive_id")
    _identifier(model_version, label="prospective archive model_version")
    _sha256(
        target_blind_frame_bundle_sha256,
        label="target_blind_frame_bundle_sha256",
    )
    normalized_dates = tuple(issue_dates)
    if not normalized_dates or len(normalized_dates) != len(set(normalized_dates)):
        raise ValueError("prospective archive issue dates must be non-empty and unique")
    for value in normalized_dates:
        date.fromisoformat(value)
    if archived_at_utc.tzinfo is None or archived_at_utc.utcoffset() != timedelta(0):
        raise ValueError("prospective archive time must be UTC aware")
    return canonical_mapping_sha256(
        {
            "archive_id": archive_id,
            "archived_at_utc": archived_at_utc.astimezone(UTC).isoformat().replace("+00:00", "Z"),
            "archived_before_any_target_observation": True,
            "issue_dates": list(normalized_dates),
            "model_version": model_version,
            "schema_version": 1,
            "target_blind_frame_bundle_sha256": target_blind_frame_bundle_sha256,
        }
    )


@dataclass(frozen=True, slots=True)
class GenuineProspectiveArchive:
    """Explicit receipt proving that the exact target-blind frame bundle was archived."""

    archive_id: str
    archive_content_sha256: str
    target_blind_frame_bundle_sha256: str
    model_version: str
    issue_dates: tuple[str, ...]
    archived_at_utc: datetime
    archived_before_any_target_observation: Literal[True] = True

    def __post_init__(self) -> None:
        _identifier(self.archive_id, label="prospective archive_id")
        _identifier(self.model_version, label="prospective archive model_version")
        _sha256(self.archive_content_sha256, label="archive_content_sha256")
        _sha256(
            self.target_blind_frame_bundle_sha256,
            label="target_blind_frame_bundle_sha256",
        )
        dates = tuple(self.issue_dates)
        if not dates or len(dates) != len(set(dates)):
            raise ValueError("prospective archive issue dates must be non-empty and unique")
        for value in dates:
            date.fromisoformat(value)
        if self.archived_at_utc.tzinfo is None or self.archived_at_utc.utcoffset() != timedelta(0):
            raise ValueError("prospective archive time must be UTC aware")
        if self.archived_before_any_target_observation is not True:
            raise ValueError("prospective archive must predate every target observation")
        normalized_time = self.archived_at_utc.astimezone(UTC)
        expected_content = genuine_prospective_archive_content_sha256(
            archive_id=self.archive_id,
            target_blind_frame_bundle_sha256=self.target_blind_frame_bundle_sha256,
            model_version=self.model_version,
            issue_dates=dates,
            archived_at_utc=normalized_time,
        )
        if self.archive_content_sha256 != expected_content:
            raise ValueError("prospective archive content hash does not bind its receipt")
        object.__setattr__(self, "issue_dates", dates)
        object.__setattr__(self, "archived_at_utc", normalized_time)


@dataclass(frozen=True, slots=True)
class _DerivedGridValues:
    relative_strength: FloatArray
    rank_percentile: FloatArray
    alarm_selected: NDArray[np.bool_]
    selected_area_km2: float
    threshold_rank_percentile: float
    selected_cell_ids: tuple[str, ...]
    prefix_count: int


def _derive_grid_values(
    intensities: object,
    *,
    cell_ids: tuple[str, ...],
    cell_rows: tuple[int, ...],
    cell_columns: tuple[int, ...],
    cell_area_km2: object,
) -> _DerivedGridValues:
    values = _readonly_float(intensities, label="cell integrated intensities")
    area = _readonly_float(cell_area_km2, label="cell areas")
    size = len(cell_ids)
    if (
        values.shape != (size,)
        or area.shape != (size,)
        or len(cell_rows) != size
        or len(cell_columns) != size
    ):
        raise ValueError("spatial intensity, area, row, column, and cell IDs must align")
    if np.any(values < 0.0) or np.any(area <= 0.0):
        raise ValueError("spatial intensities must be non-negative and cell areas positive")
    order = tuple(
        sorted(
            range(size),
            key=lambda index: (
                -float(values[index]),
                cell_rows[index],
                cell_columns[index],
                cell_ids[index],
            ),
        )
    )
    ordered = np.asarray(order, dtype=np.int64)
    cumulative = np.cumsum(area[ordered], dtype=np.float64)
    prefix_count = int(np.searchsorted(cumulative, PRIMARY_ALARM_AREA_KM2 + 1.0e-9, side="right"))
    selected_indices = order[:prefix_count]
    selected = np.zeros(size, dtype=np.bool_)
    selected[np.asarray(selected_indices, dtype=np.int64)] = True
    selected.setflags(write=False)
    ranks = np.empty(size, dtype=np.float64)
    ranks[ordered] = 100.0 * (size - np.arange(size, dtype=np.float64)) / size
    ranks.setflags(write=False)
    total = math.fsum(float(value) for value in values)
    if total == 0.0:
        strength = np.zeros(size, dtype=np.float64)
    else:
        strength = np.asarray(values / (total / size), dtype=np.float64)
    strength.setflags(write=False)
    selected_area = 0.0 if prefix_count == 0 else float(cumulative[prefix_count - 1])
    threshold = 0.0 if prefix_count == 0 else float(np.min(ranks[selected]))
    return _DerivedGridValues(
        relative_strength=cast(FloatArray, strength),
        rank_percentile=cast(FloatArray, ranks),
        alarm_selected=selected,
        selected_area_km2=selected_area,
        threshold_rank_percentile=threshold,
        selected_cell_ids=tuple(cell_ids[index] for index in selected_indices),
        prefix_count=prefix_count,
    )


def _grid_positions(
    grid: Stage4IntegrationGrid,
    *,
    cell_ids: tuple[str, ...],
    cell_rows: tuple[int, ...] | None = None,
    cell_columns: tuple[int, ...] | None = None,
    cell_area_km2: object | None = None,
) -> tuple[FloatArray, FloatArray, tuple[int, ...], tuple[int, ...], FloatArray]:
    if grid.cell_size_km != 25.0:
        raise ValueError("spatial stage-4 results require the frozen 25 km grid")
    index_by_id = {cell_id: index for index, cell_id in enumerate(grid.cell_ids)}
    if len(index_by_id) != len(grid.cell_ids):
        raise ValueError("frozen grid cell IDs are duplicated")
    try:
        indices = tuple(index_by_id[cell_id] for cell_id in cell_ids)
    except KeyError as exc:
        raise ValueError("forecast contains a target-derived or foreign grid cell") from exc
    expected_rows = tuple(int(grid.rows[index]) for index in indices)
    expected_columns = tuple(int(grid.columns[index]) for index in indices)
    expected_area = np.asarray(
        [float(grid.clipped_area_km2[index]) for index in indices],
        dtype=np.float64,
    )
    if cell_rows is not None and tuple(cell_rows) != expected_rows:
        raise ValueError("forecast cell rows differ from the frozen target-independent grid")
    if cell_columns is not None and tuple(cell_columns) != expected_columns:
        raise ValueError("forecast cell columns differ from the frozen target-independent grid")
    if cell_area_km2 is not None and not np.array_equal(
        np.asarray(cell_area_km2, dtype=np.float64),
        expected_area,
    ):
        raise ValueError("forecast cell areas differ from the frozen target-independent grid")
    xy = grid.query_xy_m[np.asarray(indices, dtype=np.int64)]
    transformer = Transformer.from_crs(
        CRS.from_user_input(grid.equal_area_crs),
        CRS.from_epsg(4326),
        always_xy=True,
    )
    longitude_raw, latitude_raw = transformer.transform(xy[:, 0], xy[:, 1])
    longitude = _readonly_float(longitude_raw, label="forecast longitude")
    latitude = _readonly_float(latitude_raw, label="forecast latitude")
    if np.any((longitude < -180.0) | (longitude > 180.0)) or np.any(
        (latitude < -90.0) | (latitude > 90.0)
    ):
        raise ValueError("frozen grid representative points do not project geographically")
    expected_area.setflags(write=False)
    return (
        longitude,
        latitude,
        expected_rows,
        expected_columns,
        expected_area,
    )


def _formal_fit(result: PipelineResult) -> FittedScope:
    matches = tuple(
        item for item in result.fitted_scopes if item.evaluation_id == "formal-validation"
    )
    if len(matches) != 1:
        raise ValueError("spatial results require exactly one formal fitted scope")
    return matches[0]


def _formal_model_fingerprint(fitted: FittedScope, plan: Stage4InMemoryPlan) -> str:
    """Hash only fitted training state and frozen target-blind model inputs."""

    return canonical_mapping_sha256(
        {
            "evaluation_id": fitted.evaluation_id,
            "feature_layout": {
                "coverage_sources": list(plan.feature_layout.coverage_sources),
                "dynamic_sources": list(plan.feature_layout.dynamic_sources),
                "snapshot_sources": list(plan.feature_layout.snapshot_sources),
            },
            "frozen_input_seal_sha256": plan.frozen_input_seal_sha256,
            "model_version": plan.model_version,
            "preprocessor": fitted.preprocessor.as_mapping(),
            "rate_head": fitted.rate_head.as_mapping(),
            "role": "target_blind_formal_fitted_model",
            "variants": [
                {
                    "beta_hex": [float(value).hex() for value in item.beta],
                    "design_column_indices": list(item.design_column_indices),
                    "fit_result_sha256": (
                        None if item.fit_result is None else item.fit_result.sha256
                    ),
                    "variant": item.variant,
                }
                for item in fitted.variants
            ],
        }
    )


def _dynamic_evidence_status(
    result: PipelineResult,
) -> Literal["passed", "failed", "evidence_insufficient"]:
    status = result.dynamic_g2.status
    if status not in {"passed", "failed", "evidence_insufficient"}:
        raise ValueError("spatial result received an unknown dynamic evidence status")
    return status


def _training_issue_count(fitted: FittedScope, cell_count: int) -> int:
    if cell_count <= 0 or fitted.preprocessor.fit_row_count % cell_count:
        raise ValueError("formal preprocessing rows are not an issue-by-grid matrix")
    return fitted.preprocessor.fit_row_count // cell_count


def _knowledge_cutoff(issue_date: str) -> datetime:
    parsed = date.fromisoformat(issue_date)
    return datetime.combine(parsed, time.min, tzinfo=_SHANGHAI).astimezone(UTC)


def _score_key(score: ExposureVariantScore) -> tuple[str, int, str, str]:
    return (score.issue_date, score.horizon_days, score.magnitude_bin, score.variant)


def _exposure_key(exposure: AssembledEvaluationExposure) -> tuple[str, int, str]:
    return (exposure.issue_date, exposure.horizon_days, exposure.magnitude_bin)


def _validate_score_against_exposure(
    score: ExposureVariantScore,
    exposure: AssembledEvaluationExposure,
) -> _DerivedGridValues:
    if score.evaluation_id != "formal-validation" or (
        score.issue_date,
        score.horizon_days,
        score.magnitude_bin,
    ) != _exposure_key(exposure):
        raise ValueError("spatial score does not correspond to its formal exposure")
    if (
        score.supported_event_ids != exposure.supported_event_ids
        or score.all_study_area_event_ids != exposure.all_study_area_event_ids
    ):
        raise ValueError("spatial score changed formal target memberships")
    intensities = _readonly_float(
        score.cell_integrated_intensities,
        label="scored cell integrated intensities",
    )
    derived = _derive_grid_values(
        intensities,
        cell_ids=exposure.cell_order_ids,
        cell_rows=exposure.cell_rows,
        cell_columns=exposure.cell_columns,
        cell_area_km2=exposure.cell_area_km2,
    )
    if not math.isclose(
        math.fsum(float(value) for value in intensities),
        score.integrated_intensity,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        raise ValueError("spatial score integrated intensity was fabricated or altered")
    primary_index = frozen_same_recall_budget_grid().index(int(PRIMARY_ALARM_AREA_KM2))
    if (
        derived.selected_cell_ids != score.selected_cell_ids_at_600000_km2
        or derived.prefix_count != score.alarm_prefix_cell_counts[primary_index]
        or not math.isclose(
            derived.selected_area_km2,
            float(score.alarm_exact_selected_area_km2[primary_index]),
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
    ):
        raise ValueError("spatial score alarm prefix was fabricated or altered")
    selected_set = set(derived.selected_cell_ids)
    expected_hits = tuple(
        event_id
        for event_id, local_cell_index in zip(
            exposure.supported_event_ids,
            exposure.event_cell_indices,
            strict=True,
        )
        if exposure.cell_order_ids[local_cell_index] in selected_set
    )
    if expected_hits != score.strict_hit_event_ids_at_600000_km2:
        raise ValueError("spatial score hit IDs differ from the real alarm prefix")
    return derived


def build_retrospective_forecast_frames(
    result: PipelineResult,
    plan: Stage4InMemoryPlan,
    primary_grid: Stage4IntegrationGrid,
    *,
    selection: SpatialCalendarSelection,
) -> tuple[ForecastGridFrame, ...]:
    """Adapt selected real formal scores into complete per-cell retrospective frames."""

    if selection != build_score_blind_calendar_selection(plan):
        raise ValueError("retrospective spatial selection is not the score-blind calendar rule")
    fitted = _formal_fit(result)
    model_fingerprint = _formal_model_fingerprint(fitted, plan)
    exposure_by_key = {_exposure_key(item): item for item in plan.formal_scope.exposures}
    if len(exposure_by_key) != len(plan.formal_scope.exposures):
        raise ValueError("formal exposures have duplicate spatial identities")
    formal_scores = tuple(
        item for item in result.exposure_scores if item.evaluation_id == "formal-validation"
    )
    score_by_key = {_score_key(item): item for item in formal_scores}
    if len(score_by_key) != len(formal_scores):
        raise ValueError("formal scores have duplicate spatial identities")
    output: list[ForecastGridFrame] = []
    primary_keys = set(selection.retrospective_keys)
    frame_keys = (*selection.retrospective_keys, *selection.retrospective_support_keys)
    for issue_date, horizon in frame_keys:
        exposure_key = (issue_date, horizon, selection.magnitude_bin)
        try:
            exposure = exposure_by_key[exposure_key]
            score = score_by_key[(*exposure_key, selection.model_variant)]
        except KeyError as exc:
            raise ValueError(
                "score-blind display key is missing a formal exposure or score"
            ) from exc
        derived = _validate_score_against_exposure(score, exposure)
        longitude, latitude, _, _, _ = _grid_positions(
            primary_grid,
            cell_ids=exposure.cell_order_ids,
            cell_rows=exposure.cell_rows,
            cell_columns=exposure.cell_columns,
            cell_area_km2=exposure.cell_area_km2,
        )
        output.append(
            ForecastGridFrame(
                issue_date=issue_date,
                magnitude_bin=selection.magnitude_bin,
                horizon_days=horizon,
                model_variant=selection.model_variant,
                model_version=plan.model_version,
                training_sample_size=_training_issue_count(fitted, len(exposure.cell_order_ids)),
                grid_id=primary_grid.grid_id,
                cell_size_km=primary_grid.cell_size_km,
                alarm_threshold_rank_percentile=derived.threshold_rank_percentile,
                selected_alarm_area_km2=derived.selected_area_km2,
                displayed_domain_area_km2=math.fsum(
                    float(value) for value in exposure.cell_area_km2
                ),
                cell_ids=exposure.cell_order_ids,
                longitude=longitude,
                latitude=latitude,
                relative_strength=derived.relative_strength,
                rank_percentile=derived.rank_percentile,
                alarm_selected=derived.alarm_selected,
                forecast_status="retrospective_evaluation",
                evidence_status=_dynamic_evidence_status(result),
                display_role=(
                    "primary" if (issue_date, horizon) in primary_keys else "previous_context"
                ),
                knowledge_cutoff_utc=_knowledge_cutoff(issue_date),
                source_model_fingerprint_sha256=model_fingerprint,
                source_result_fingerprint_sha256=result.result_fingerprint_sha256,
            )
        )
    return tuple(output)


def _prospective_schema_is_target_free(issue: AssembledProspectiveIssue) -> None:
    field_names = tuple(field.name for field in dataclasses.fields(issue))
    column_names = tuple(str(name) for name in issue.cell_feature_columns)
    for name in (*field_names, *column_names):
        folded = name.casefold()
        if any(token in folded for token in _FORBIDDEN_PROSPECTIVE_TOKENS):
            raise ValueError(f"prospective spatial inference contains a forbidden field: {name}")


def _cell_integrated_intensities(
    *,
    background_mass: FloatArray,
    linear_predictor: FloatArray,
    rate_multiplier: float,
    horizon_days: int,
) -> FloatArray:
    if rate_multiplier == 0.0:
        output = np.zeros(background_mass.size, dtype=np.float64)
        output.setflags(write=False)
        return cast(FloatArray, output)
    if not np.any(linear_predictor):
        output = np.asarray(background_mass * rate_multiplier * horizon_days, dtype=np.float64)
        output.setflags(write=False)
        return cast(FloatArray, output)
    quadrature = composite_midpoint_quadrature(float(horizon_days))
    temporal_multiplier = np.zeros(linear_predictor.size, dtype=np.float64)
    for midpoint, width in zip(
        quadrature.lead_midpoints_days,
        quadrature.widths_days,
        strict=True,
    ):
        decay = float(lead_decay(float(midpoint)))
        with np.errstate(over="raise", invalid="raise"):
            temporal_multiplier += np.exp(decay * linear_predictor) * float(width)
    output = np.asarray(background_mass * rate_multiplier * temporal_multiplier, dtype=np.float64)
    if not np.isfinite(output).all() or np.any(output < 0.0):
        raise FloatingPointError("prospective spatial intensity is non-finite")
    output.setflags(write=False)
    return cast(FloatArray, output)


def _prospective_intensities(
    issue: AssembledProspectiveIssue,
    *,
    fitted: FittedScope,
    variant: FittedVariant,
) -> FloatArray:
    full_design = fitted.preprocessor.transform(issue.cell_feature_columns)
    if variant.variant == "background_no_increment":
        linear = np.zeros(full_design.row_count, dtype=np.float64)
    else:
        indices = np.asarray(variant.design_column_indices, dtype=np.int64)
        linear = np.asarray(full_design.values[:, indices] @ variant.beta, dtype=np.float64)
    return _cell_integrated_intensities(
        background_mass=issue.background_spatial_mass,
        linear_predictor=cast(FloatArray, linear),
        rate_multiplier=fitted.rate_head.by_id(issue.magnitude_bin).rate_multiplier,
        horizon_days=issue.horizon_days,
    )


def target_blind_frame_bundle_sha256(frames: Sequence[ForecastGridFrame]) -> str:
    """Hash complete target-free scientific frame values, independent of display status."""

    selected = tuple(frames)
    if not selected:
        raise ValueError("target-blind frame bundle cannot be empty")
    if any(
        item.forecast_status == "retrospective_evaluation"
        or item.evidence_status != "not_evaluated_target_blind"
        or item.source_result_fingerprint_sha256 is not None
        for item in selected
    ):
        raise ValueError("target-blind frame bundle contains target-derived result metadata")
    return canonical_mapping_sha256(
        {
            "schema_version": 1,
            "role": "complete_target_blind_spatial_frame_bundle",
            "frames": [
                {
                    "alarm_selected": item.alarm_selected.tolist(),
                    "cell_ids": list(item.cell_ids),
                    "display_role": item.display_role,
                    "grid_id": item.grid_id,
                    "horizon_days": item.horizon_days,
                    "issue_date": item.issue_date,
                    "magnitude_bin": item.magnitude_bin,
                    "model_variant": item.model_variant,
                    "model_version": item.model_version,
                    "displayed_domain_area_km2_hex": (item.displayed_domain_area_km2.hex()),
                    "knowledge_cutoff_utc": (
                        item.knowledge_cutoff_utc.isoformat().replace("+00:00", "Z")
                    ),
                    "rank_percentile_hex": [float(value).hex() for value in item.rank_percentile],
                    "relative_strength_hex": [
                        float(value).hex() for value in item.relative_strength
                    ],
                    "alarm_threshold_rank_percentile_hex": (
                        item.alarm_threshold_rank_percentile.hex()
                    ),
                    "selected_alarm_area_km2_hex": item.selected_alarm_area_km2.hex(),
                    "source_model_fingerprint_sha256": (item.source_model_fingerprint_sha256),
                    "training_sample_size": item.training_sample_size,
                }
                for item in selected
            ],
        }
    )


def build_target_blind_forecast_frames(
    result: PipelineResult,
    plan: Stage4InMemoryPlan,
    primary_grid: Stage4IntegrationGrid,
    *,
    selection: SpatialCalendarSelection,
    genuine_prospective_archive: GenuineProspectiveArchive | None = None,
) -> tuple[ForecastGridFrame, ...]:
    """Run real target-free inference; default status is explicitly post-hoc shadow."""

    if selection != build_score_blind_calendar_selection(plan):
        raise ValueError("prospective spatial selection is not the score-blind calendar rule")
    fitted = _formal_fit(result)
    model_fingerprint = _formal_model_fingerprint(fitted, plan)
    variant = fitted.variant(selection.model_variant)
    issue_by_key = {
        (item.issue_date, item.horizon_days, item.magnitude_bin): item
        for item in plan.prospective_issues
    }
    if len(issue_by_key) != len(plan.prospective_issues):
        raise ValueError("target-free issues have duplicate spatial identities")
    output: list[ForecastGridFrame] = []
    primary_keys = set(selection.prospective_keys)
    frame_keys = (*selection.prospective_keys, *selection.prospective_support_keys)
    for issue_date, horizon in frame_keys:
        key = (issue_date, horizon, selection.magnitude_bin)
        try:
            issue = issue_by_key[key]
        except KeyError as exc:
            raise ValueError("score-blind display key is missing a target-free issue") from exc
        _prospective_schema_is_target_free(issue)
        if tuple(issue.cell_feature_columns) != tuple(plan.feature_layout.dynamic_sources):
            raise ValueError("target-free issue feature order differs from the frozen model")
        longitude, latitude, rows, columns, area = _grid_positions(
            primary_grid,
            cell_ids=issue.cell_order_ids,
            cell_area_km2=issue.cell_area_km2,
        )
        intensities = _prospective_intensities(issue, fitted=fitted, variant=variant)
        derived = _derive_grid_values(
            intensities,
            cell_ids=issue.cell_order_ids,
            cell_rows=rows,
            cell_columns=columns,
            cell_area_km2=area,
        )
        output.append(
            ForecastGridFrame(
                issue_date=issue_date,
                magnitude_bin=selection.magnitude_bin,
                horizon_days=horizon,
                model_variant=selection.model_variant,
                model_version=plan.model_version,
                training_sample_size=_training_issue_count(fitted, len(issue.cell_order_ids)),
                grid_id=primary_grid.grid_id,
                cell_size_km=primary_grid.cell_size_km,
                alarm_threshold_rank_percentile=derived.threshold_rank_percentile,
                selected_alarm_area_km2=derived.selected_area_km2,
                displayed_domain_area_km2=math.fsum(float(value) for value in area),
                cell_ids=issue.cell_order_ids,
                longitude=longitude,
                latitude=latitude,
                relative_strength=derived.relative_strength,
                rank_percentile=derived.rank_percentile,
                alarm_selected=derived.alarm_selected,
                forecast_status="retrospective_generated_target_blind_shadow",
                evidence_status="not_evaluated_target_blind",
                display_role=(
                    "primary" if (issue_date, horizon) in primary_keys else "previous_context"
                ),
                knowledge_cutoff_utc=_knowledge_cutoff(issue_date),
                source_model_fingerprint_sha256=model_fingerprint,
                source_result_fingerprint_sha256=None,
            )
        )
    shadow_frames = tuple(output)
    if genuine_prospective_archive is None:
        return shadow_frames
    expected_dates = tuple(sorted({item.issue_date for item in shadow_frames}))
    if (
        genuine_prospective_archive.model_version != plan.model_version
        or genuine_prospective_archive.issue_dates != expected_dates
        or genuine_prospective_archive.target_blind_frame_bundle_sha256
        != target_blind_frame_bundle_sha256(shadow_frames)
    ):
        raise ValueError("genuine prospective archive does not bind the exact target-blind frames")
    earliest_issue_time = min(item.knowledge_cutoff_utc for item in shadow_frames)
    if genuine_prospective_archive.archived_at_utc > earliest_issue_time:
        raise ValueError("genuine prospective archive was created after its issue time")
    return tuple(
        dataclasses.replace(item, forecast_status="genuine_prospective_archive")
        for item in shadow_frames
    )


def _expected_catalog_event_ids(
    catalog: Stage4TargetCatalog,
    exposure: AssembledEvaluationExposure,
) -> tuple[str, ...]:
    issue_local = datetime.combine(
        date.fromisoformat(exposure.issue_date),
        time.min,
        tzinfo=_SHANGHAI,
    )
    start = issue_local.astimezone(UTC)
    end = start + timedelta(days=exposure.horizon_days)
    output: list[str] = []
    for index, event_id in enumerate(catalog.event_id):
        magnitude = float(catalog.magnitude[index])
        in_bin = 5.0 <= magnitude < 6.0 if exposure.magnitude_bin == "M5_6" else magnitude >= 6.0
        if (
            start < catalog.origin_time_utc[index] <= end
            and in_bin
            and bool(catalog.inside_study_area[index])
        ):
            output.append(str(event_id))
    return tuple(output)


def build_retrospective_target_overlays(
    result: PipelineResult,
    plan: Stage4InMemoryPlan,
    primary_grid: Stage4IntegrationGrid,
    catalog: Stage4TargetCatalog,
    target_assignments: BoundTargetCellAssignments,
    *,
    selection: SpatialCalendarSelection,
) -> tuple[RetrospectiveTargetFrame, ...]:
    """Build target-bearing overlays only from the authorized catalogue and real hit IDs."""

    target_assignments.verify(catalog, primary_grid=primary_grid)
    if target_assignments.catalog_source_content_sha256 != catalog.source_content_sha256:
        raise ValueError("target overlay catalogue differs from the authorized assignment receipt")
    exposure_by_key = {_exposure_key(item): item for item in plan.formal_scope.exposures}
    score_by_key = {
        _score_key(item): item
        for item in result.exposure_scores
        if item.evaluation_id == "formal-validation"
    }
    catalog_index = {str(event_id): index for index, event_id in enumerate(catalog.event_id)}
    output: list[RetrospectiveTargetFrame] = []
    for issue_date, horizon in selection.retrospective_keys:
        key = (issue_date, horizon, selection.magnitude_bin)
        try:
            exposure = exposure_by_key[key]
            score = score_by_key[(*key, selection.model_variant)]
        except KeyError as exc:
            raise ValueError(
                "target overlay key is missing a formal exposure or dynamic score"
            ) from exc
        _validate_score_against_exposure(score, exposure)
        expected_event_ids = _expected_catalog_event_ids(catalog, exposure)
        if exposure.all_study_area_event_ids != expected_event_ids:
            raise ValueError(
                "formal target membership differs from the authorized in-memory catalogue"
            )
        try:
            indices = tuple(catalog_index[event_id] for event_id in expected_event_ids)
        except KeyError as exc:  # pragma: no cover - exact set checked above
            raise ValueError("formal target event is absent from the authorized catalogue") from exc
        hit_ids = score.strict_hit_event_ids_at_600000_km2
        output.append(
            RetrospectiveTargetFrame(
                issue_date=issue_date,
                magnitude_bin=selection.magnitude_bin,
                horizon_days=horizon,
                event_ids=expected_event_ids,
                longitude=np.asarray(
                    [float(catalog.longitude[index]) for index in indices],
                    dtype=np.float64,
                ),
                latitude=np.asarray(
                    [float(catalog.latitude[index]) for index in indices],
                    dtype=np.float64,
                ),
                covered_by_600000_km2=tuple(
                    event_id in set(hit_ids) for event_id in expected_event_ids
                ),
                hit_event_ids_at_600000_km2=hit_ids,
                source_catalog_content_sha256=catalog.source_content_sha256,
                source_result_fingerprint_sha256=result.result_fingerprint_sha256,
            )
        )
    return tuple(output)


def _color(rank: float) -> str:
    if rank >= 90.0:
        return "#d73027"
    if rank >= 70.0:
        return "#f2d35c"
    if rank >= 40.0:
        return "#6bb7d6"
    return "#2459a6"


def _evidence_label(status: str) -> str:
    try:
        return {
            "passed": "通过",
            "failed": "未通过",
            "evidence_insufficient": "证据不足",
            "not_evaluated_target_blind": "目标盲帧不携带回溯证据",
        }[status]
    except KeyError as exc:  # pragma: no cover - frame validation already enforces this
        raise ValueError("unknown spatial evidence status") from exc


def _static_indices(frame: ForecastGridFrame, *, non_alarm_limit: int = 1_200) -> tuple[int, ...]:
    alarm = tuple(int(value) for value in np.flatnonzero(frame.alarm_selected))
    non_alarm = tuple(int(value) for value in np.flatnonzero(~frame.alarm_selected))
    stride = max(1, math.ceil(len(non_alarm) / non_alarm_limit))
    sampled = non_alarm[::stride]
    return tuple(sorted((*alarm, *sampled)))


def _project_panel(
    longitude: float,
    latitude: float,
    *,
    bounds: tuple[float, float, float, float],
    x: float,
    y: float,
    width: float,
    height: float,
) -> tuple[float, float]:
    minimum_lon, minimum_lat, maximum_lon, maximum_lat = bounds
    span_lon = max(maximum_lon - minimum_lon, 1.0e-12)
    span_lat = max(maximum_lat - minimum_lat, 1.0e-12)
    padding = 10.0
    scale = min((width - 2 * padding) / span_lon, (height - 2 * padding) / span_lat)
    center_lon = (minimum_lon + maximum_lon) / 2.0
    center_lat = (minimum_lat + maximum_lat) / 2.0
    return (
        x + width / 2.0 + (longitude - center_lon) * scale,
        y + height / 2.0 - (latitude - center_lat) * scale,
    )


def _context_paths(layer: DisplayContextLayer) -> tuple[tuple[tuple[float, float], ...], ...]:
    geometry = layer.geometry_geojson
    geometry_type = geometry["type"]
    coordinates = geometry["coordinates"]
    raw_paths: Sequence[object]
    if geometry_type == "LineString":
        raw_paths = (coordinates,)
    elif geometry_type in {"MultiLineString", "Polygon"}:
        raw_paths = cast(Sequence[object], coordinates)
    else:
        raw_paths = tuple(
            ring for polygon in cast(Sequence[Sequence[object]], coordinates) for ring in polygon
        )
    return tuple(
        tuple((float(point[0]), float(point[1])) for point in cast(Sequence[Sequence[float]], path))
        for path in raw_paths
    )


def _study_area_rings(
    study_area: DisplayStudyArea,
) -> tuple[tuple[tuple[float, float], ...], ...]:
    geometry = study_area.geometry_geojson
    coordinates = geometry["coordinates"]
    raw_rings: Sequence[object]
    if geometry["type"] == "Polygon":
        raw_rings = cast(Sequence[object], coordinates)
    else:
        raw_rings = tuple(
            ring for polygon in cast(Sequence[Sequence[object]], coordinates) for ring in polygon
        )
    return tuple(
        tuple((float(point[0]), float(point[1])) for point in cast(Sequence[Sequence[float]], ring))
        for ring in raw_rings
    )


def _study_area_bounds(study_area: DisplayStudyArea) -> tuple[float, float, float, float]:
    positions = tuple(point for ring in _study_area_rings(study_area) for point in ring)
    if not positions:  # pragma: no cover - DisplayStudyArea validates non-empty polygons
        raise ValueError("display study area has no positions")
    longitude = tuple(item[0] for item in positions)
    latitude = tuple(item[1] for item in positions)
    return min(longitude), min(latitude), max(longitude), max(latitude)


def _study_area_svg_path(
    study_area: DisplayStudyArea,
    *,
    bounds: tuple[float, float, float, float],
    x: float,
    y: float,
    width: float,
    height: float,
) -> str:
    commands: list[str] = []
    for ring in _study_area_rings(study_area):
        projected = tuple(
            _project_panel(
                longitude,
                latitude,
                bounds=bounds,
                x=x,
                y=y,
                width=width,
                height=height,
            )
            for longitude, latitude in ring
        )
        commands.append(
            "M "
            + " L ".join(f"{point_x:.2f} {point_y:.2f}" for point_x, point_y in projected)
            + " Z"
        )
    return " ".join(commands)


def _static_panel(
    frame: ForecastGridFrame,
    *,
    previous_frame: ForecastGridFrame | None,
    target: RetrospectiveTargetFrame | None,
    study_area: DisplayStudyArea,
    context_layers: tuple[DisplayContextLayer, ...],
    x: float,
    y: float,
    width: float,
    height: float,
) -> str:
    plot_y = y + 58.0
    plot_height = height - 79.0
    bounds = _study_area_bounds(study_area)
    marks: list[str] = []
    for layer in context_layers:
        for path in _context_paths(layer):
            points = " ".join(
                f"{px:.2f},{py:.2f}"
                for longitude, latitude in path
                for px, py in (
                    _project_panel(
                        longitude,
                        latitude,
                        bounds=bounds,
                        x=x,
                        y=plot_y,
                        width=width,
                        height=plot_height,
                    ),
                )
            )
            marks.append(
                f'<polyline points="{points}" fill="none" stroke="#75634d" stroke-width="0.8"/>'
            )
    if previous_frame is not None:
        for previous_raw_index in np.flatnonzero(previous_frame.alarm_selected):
            previous_index = int(previous_raw_index)
            px, py = _project_panel(
                float(previous_frame.longitude[previous_index]),
                float(previous_frame.latitude[previous_index]),
                bounds=bounds,
                x=x,
                y=plot_y,
                width=width,
                height=plot_height,
            )
            marks.append(
                f'<rect x="{px - 2.4:.2f}" y="{py - 2.4:.2f}" width="4.8" height="4.8" fill="none" stroke="#7b2cbf" stroke-width="0.9" stroke-dasharray="3 2"/>'
            )
    displayed = _static_indices(frame)
    for index in displayed:
        px, py = _project_panel(
            float(frame.longitude[index]),
            float(frame.latitude[index]),
            bounds=bounds,
            x=x,
            y=plot_y,
            width=width,
            height=plot_height,
        )
        size = 3.2 if frame.alarm_selected[index] else 1.8
        stroke = ' stroke="#172033" stroke-width="0.8"' if frame.alarm_selected[index] else ""
        marks.append(
            f'<rect x="{px - size / 2:.2f}" y="{py - size / 2:.2f}" width="{size:.2f}" height="{size:.2f}" fill="{_color(float(frame.rank_percentile[index]))}"{stroke}/>'
        )
    if target is not None:
        for longitude, latitude, covered in zip(
            target.longitude,
            target.latitude,
            target.covered_by_600000_km2,
            strict=True,
        ):
            px, py = _project_panel(
                float(longitude),
                float(latitude),
                bounds=bounds,
                x=x,
                y=plot_y,
                width=width,
                height=plot_height,
            )
            fill = "#172033" if covered else "#ffffff"
            marks.append(
                f'<circle cx="{px:.2f}" cy="{py:.2f}" r="3.1" fill="{fill}" stroke="#172033" stroke-width="1"/>'
            )
    previous_note = "无上一期" if previous_frame is None else f"上一期 {previous_frame.issue_date}"
    sample_note = f"静态显示 {len(displayed)}/{len(frame.cell_ids)} 格（报警格全量；其余按网格顺序等距） · {previous_note}"
    clip_id = (
        "panel-clip-"
        + canonical_mapping_sha256(
            {
                "issue_date": frame.issue_date,
                "horizon_days": frame.horizon_days,
                "forecast_status": frame.forecast_status,
            }
        )[:12]
    )
    study_area_path = _study_area_svg_path(
        study_area,
        bounds=bounds,
        x=x,
        y=plot_y,
        width=width,
        height=plot_height,
    )
    peak = max(float(value) for value in frame.relative_strength)
    return (
        f'<g data-horizon-days="{frame.horizon_days}" data-forecast-status="{frame.forecast_status}">'
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" fill="#ffffff" stroke="#a7adb7"/>'
        f'<defs><clipPath id="{clip_id}"><path d="{study_area_path}" fill-rule="evenodd" clip-rule="evenodd"/></clipPath></defs>'
        f'<text x="{x + 10:.1f}" y="{y + 18:.1f}" font-size="13" font-weight="500">{escape(frame.issue_date)} · {frame.horizon_days}天 · 动态模型</text>'
        f'<text x="{x + 10:.1f}" y="{y + 33:.1f}" font-size="9" fill="#5d6878">报警 {frame.selected_alarm_area_km2:,.0f} / 展示域 {frame.displayed_domain_area_km2:,.0f} km² · 相对强度峰值 {peak:.2f}</text>'
        f'<text x="{x + 10:.1f}" y="{y + 47:.1f}" font-size="9" fill="#5d6878">截止 {frame.knowledge_cutoff_utc.isoformat().replace("+00:00", "Z")} · {escape(frame.model_version)} · 证据 {_evidence_label(frame.evidence_status)}</text>'
        + f'<g clip-path="url(#{clip_id})">'
        + "".join(marks)
        + "</g>"
        + f'<path d="{study_area_path}" fill="none" stroke="#657181" stroke-width="1" fill-rule="evenodd"/>'
        + f'<text x="{x + 10:.1f}" y="{y + height - 7:.1f}" font-size="9" fill="#5d6878">{sample_note}</text></g>'
    )


def build_static_spatial_svg(
    retrospective_frames: Sequence[ForecastGridFrame],
    prospective_frames: Sequence[ForecastGridFrame],
    retrospective_targets: Sequence[RetrospectiveTargetFrame],
    *,
    study_area: DisplayStudyArea,
    display_context_layers: Sequence[DisplayContextLayer] = (),
) -> str:
    """Render six compact, calendar-selected M5 dynamic maps for local review."""

    retro = tuple(retrospective_frames)
    prospective = tuple(prospective_frames)
    targets = tuple(retrospective_targets)
    contexts = tuple(display_context_layers)
    if not isinstance(study_area, DisplayStudyArea):
        raise TypeError("static spatial SVG requires a frozen target-independent study area")
    panels: list[str] = []
    target_by_key = {
        (item.issue_date, item.horizon_days, item.magnitude_bin): item for item in targets
    }
    for row, horizon in enumerate(_DISPLAY_HORIZONS):
        retro_matches = tuple(
            item
            for item in retro
            if item.horizon_days == horizon
            and item.magnitude_bin == _DISPLAY_MAGNITUDE_BIN
            and item.model_variant == _DISPLAY_VARIANT
            and item.display_role == "primary"
        )
        prospective_matches = tuple(
            item
            for item in prospective
            if item.horizon_days == horizon
            and item.magnitude_bin == _DISPLAY_MAGNITUDE_BIN
            and item.model_variant == _DISPLAY_VARIANT
            and item.display_role == "primary"
        )
        if not retro_matches or not prospective_matches:
            raise ValueError("static spatial SVG requires M5 dynamic 7/30/90-day frames")
        retrospective_frame = max(retro_matches, key=lambda item: item.issue_date)
        prospective_frame = max(prospective_matches, key=lambda item: item.issue_date)
        ordered_retrospective = tuple(
            sorted(
                (
                    item
                    for item in retro
                    if item.horizon_days == horizon
                    and item.magnitude_bin == _DISPLAY_MAGNITUDE_BIN
                    and item.model_variant == _DISPLAY_VARIANT
                ),
                key=lambda item: item.issue_date,
            )
        )
        ordered_prospective = tuple(
            sorted(
                (
                    item
                    for item in prospective
                    if item.horizon_days == horizon
                    and item.magnitude_bin == _DISPLAY_MAGNITUDE_BIN
                    and item.model_variant == _DISPLAY_VARIANT
                ),
                key=lambda item: item.issue_date,
            )
        )
        panels.append(
            _static_panel(
                retrospective_frame,
                previous_frame=(
                    None if len(ordered_retrospective) < 2 else ordered_retrospective[-2]
                ),
                target=target_by_key.get(
                    (
                        retrospective_frame.issue_date,
                        retrospective_frame.horizon_days,
                        retrospective_frame.magnitude_bin,
                    )
                ),
                study_area=study_area,
                context_layers=contexts,
                x=24.0,
                y=112.0 + row * 252.0,
                width=564.0,
                height=232.0,
            )
        )
        panels.append(
            _static_panel(
                prospective_frame,
                previous_frame=(None if len(ordered_prospective) < 2 else ordered_prospective[-2]),
                target=None,
                study_area=study_area,
                context_layers=contexts,
                x=612.0,
                y=112.0 + row * 252.0,
                width=564.0,
                height=232.0,
            )
        )
    prospective_statuses = {item.forecast_status for item in prospective}
    if prospective_statuses == {"retrospective_generated_target_blind_shadow"}:
        prospective_label = "回溯生成的目标盲影子预测"
    elif prospective_statuses == {"genuine_prospective_archive"}:
        prospective_label = "真正前瞻归档"
    else:
        raise ValueError("static spatial SVG received mixed prospective archive statuses")
    context_label = (
        "上下文层未提供"
        if not contexts
        else "显示专用上下文：" + "、".join(item.label for item in contexts)
    )
    fingerprint = retro[0].source_result_fingerprint_sha256
    if fingerprint is None:  # pragma: no cover - ForecastGridFrame enforces retrospective status
        raise ValueError("static retrospective frames lack a result fingerprint")
    model_fingerprint = retro[0].source_model_fingerprint_sha256
    document = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="888" viewBox="0 0 1200 888" role="img" aria-labelledby="title desc" data-classification="{_LOCAL_CLASSIFICATION}" data-result-fingerprint-sha256="{fingerprint}" data-model-fingerprint-sha256="{model_fingerprint}" data-study-area-geometry-sha256="{study_area.geometry_content_sha256}">
<title id="title">SeismoFlux M5 动态模型空间回溯与目标盲预测</title><desc id="desc">7、30、90天窗口；左侧是授权目标叠加的回溯评估，右侧是目标盲预测。条件相对强度不能解释为绝对发生率估计。</desc>
<rect width="1200" height="888" fill="#f7f8fb"/><text x="24" y="28" font-family="system-ui,sans-serif" font-size="20" font-weight="500" fill="#172033">M5 动态模型：条件相对强度与受控报警面积</text>
<text x="24" y="50" font-family="system-ui,sans-serif" font-size="11" fill="#5d6878">条件相对强度/顺位不能解释为绝对发生率估计；日期按冻结日历规则选取，不依据结果好坏。</text>
<text x="24" y="73" font-family="system-ui,sans-serif" font-size="13" font-weight="500" fill="#172033">回溯评估（独立目标仅在此叠加）</text><text x="612" y="73" font-family="system-ui,sans-serif" font-size="13" font-weight="500" fill="#172033">{escape(prospective_label)}</text>
<g font-family="system-ui,sans-serif">{"".join(panels)}</g>
<g font-family="system-ui,sans-serif" font-size="10" fill="#5d6878"><text x="24" y="876">图例顺序：低→高相对顺位　▧ 上一期报警格（紫色虚线）　▣ 当前报警格（深色实线）　● 回溯覆盖目标　○ 回溯未覆盖目标　— 显示专用上下文　· {escape(context_label)}</text></g></svg>"""
    if len(document.encode("utf-8")) > 64 * 1024 * 1024:
        raise ValueError("static spatial SVG exceeds the 64 MiB local payload cap")
    try:
        root = ET.fromstring(document)
    except ET.ParseError as exc:  # pragma: no cover - deterministic template regression guard
        raise ValueError("static spatial SVG is not well-formed XML") from exc
    if root.tag != "{http://www.w3.org/2000/svg}svg":
        raise ValueError("static spatial result root is not SVG")
    return document


@dataclass(frozen=True, slots=True)
class Stage4SpatialResults:
    selection: SpatialCalendarSelection
    retrospective_frames: tuple[ForecastGridFrame, ...]
    target_blind_frames: tuple[ForecastGridFrame, ...]
    retrospective_targets: tuple[RetrospectiveTargetFrame, ...]
    static_svg: str
    interactive_html: str
    interactive_html_bytes: int
    target_blind_frame_bundle_sha256: str
    local_only_classification: Literal["local_restricted_target_bearing_spatial_visualization"] = (
        _LOCAL_CLASSIFICATION
    )

    def __post_init__(self) -> None:
        retrospective = tuple(self.retrospective_frames)
        target_blind = tuple(self.target_blind_frames)
        targets = tuple(self.retrospective_targets)
        if not retrospective or not target_blind:
            raise ValueError("spatial result requires retrospective and target-blind frames")
        model_fingerprints = {
            item.source_model_fingerprint_sha256 for item in (*retrospective, *target_blind)
        }
        if len(model_fingerprints) != 1:
            raise ValueError("spatial result mixes fitted-model fingerprints")
        retrospective_result_fingerprints = {
            item.source_result_fingerprint_sha256 for item in retrospective
        }
        if None in retrospective_result_fingerprints or len(retrospective_result_fingerprints) != 1:
            raise ValueError("retrospective spatial result mixes scoring-result fingerprints")
        retrospective_result_fingerprint = next(iter(retrospective_result_fingerprints))
        if any(
            item.source_result_fingerprint_sha256 is not None
            or item.evidence_status != "not_evaluated_target_blind"
            for item in target_blind
        ):
            raise ValueError("target-blind spatial result contains target-derived metadata")
        if any(
            item.source_result_fingerprint_sha256 != retrospective_result_fingerprint
            for item in targets
        ):
            raise ValueError("retrospective target overlays use another scoring result")
        if self.interactive_html_bytes != len(self.interactive_html.encode("utf-8")):
            raise ValueError("spatial interactive byte receipt changed")
        if self.interactive_html_bytes > 64 * 1024 * 1024:
            raise ValueError("spatial interactive result exceeds the 64 MiB local cap")
        if self.target_blind_frame_bundle_sha256 != target_blind_frame_bundle_sha256(target_blind):
            raise ValueError("target-blind spatial frame bundle receipt changed")
        if self.local_only_classification != _LOCAL_CLASSIFICATION:
            raise ValueError("target-bearing spatial result escaped local-only classification")
        if len(self.static_svg.encode("utf-8")) > 64 * 1024 * 1024:
            raise ValueError("spatial static result exceeds the 64 MiB local cap")
        try:
            svg_root = ET.fromstring(self.static_svg)
        except ET.ParseError as exc:
            raise ValueError("spatial static result is not well-formed SVG") from exc
        if (
            svg_root.tag != "{http://www.w3.org/2000/svg}svg"
            or svg_root.attrib.get("data-classification") != _LOCAL_CLASSIFICATION
        ):
            raise ValueError("spatial static result lost its local-only classification")
        required_html_fragments = (
            '<meta charset="utf-8"/>',
            'data-classification="local_restricted_target_bearing_spatial_visualization"',
            'id="prospective-spatial-payload"',
            'id="retrospective-target-overlay-payload"',
        )
        if any(fragment not in self.interactive_html for fragment in required_html_fragments):
            raise ValueError("spatial interactive result is incomplete or not valid UTF-8 HTML")
        object.__setattr__(self, "retrospective_frames", retrospective)
        object.__setattr__(self, "target_blind_frames", target_blind)
        object.__setattr__(self, "retrospective_targets", targets)


def _verify_materialized_formal_scope(
    materialization: AuthorizedFormalMaterialization,
    plan: Stage4InMemoryPlan,
) -> None:
    materialized = materialization.assembly.formal_scope
    if materialized.fit.evaluation_id != plan.formal_scope.fit.evaluation_id:
        raise ValueError("spatial plan uses another materialized formal fit")
    left = tuple(
        (
            _exposure_key(item),
            item.cell_order_ids,
            item.supported_event_ids,
            item.all_study_area_event_ids,
        )
        for item in materialized.exposures
    )
    right = tuple(
        (
            _exposure_key(item),
            item.cell_order_ids,
            item.supported_event_ids,
            item.all_study_area_event_ids,
        )
        for item in plan.formal_scope.exposures
    )
    if left != right:
        raise ValueError("spatial plan differs from the authorized formal materialization")


def build_stage4_spatial_results(
    result: PipelineResult,
    plan: Stage4InMemoryPlan,
    materialization: AuthorizedFormalMaterialization,
    primary_grid: Stage4IntegrationGrid,
    catalog: Stage4TargetCatalog,
    *,
    study_area: DisplayStudyArea,
    display_context_layers: Sequence[DisplayContextLayer] = (),
    genuine_prospective_archive: GenuineProspectiveArchive | None = None,
) -> Stage4SpatialResults:
    """Wire real score, target-blind inference, authorized overlay, and local visuals."""

    if not isinstance(materialization, AuthorizedFormalMaterialization):
        raise TypeError("spatial results require an authorized formal materialization")
    _verify_materialized_formal_scope(materialization, plan)
    selection = build_score_blind_calendar_selection(plan)
    retrospective = build_retrospective_forecast_frames(
        result,
        plan,
        primary_grid,
        selection=selection,
    )
    target_blind = build_target_blind_forecast_frames(
        result,
        plan,
        primary_grid,
        selection=selection,
        genuine_prospective_archive=genuine_prospective_archive,
    )
    targets = build_retrospective_target_overlays(
        result,
        plan,
        primary_grid,
        catalog,
        materialization.cell_assignments,
        selection=selection,
    )
    contexts = tuple(display_context_layers)
    static_svg = build_static_spatial_svg(
        retrospective,
        target_blind,
        targets,
        study_area=study_area,
        display_context_layers=contexts,
    )
    interactive_html = build_local_spatial_dashboard_html(
        study_area=study_area,
        retrospective_fields=retrospective,
        prospective_fields=target_blind,
        retrospective_targets=targets,
        display_context_layers=contexts,
    )
    return Stage4SpatialResults(
        selection=selection,
        retrospective_frames=retrospective,
        target_blind_frames=target_blind,
        retrospective_targets=targets,
        static_svg=static_svg,
        interactive_html=interactive_html,
        interactive_html_bytes=len(interactive_html.encode("utf-8")),
        target_blind_frame_bundle_sha256=target_blind_frame_bundle_sha256(target_blind),
    )


__all__ = [
    "GenuineProspectiveArchive",
    "SpatialCalendarSelection",
    "Stage4SpatialResults",
    "build_retrospective_forecast_frames",
    "build_retrospective_target_overlays",
    "build_score_blind_calendar_selection",
    "build_stage4_spatial_results",
    "build_static_spatial_svg",
    "build_target_blind_forecast_frames",
    "genuine_prospective_archive_content_sha256",
    "target_blind_frame_bundle_sha256",
]
