"""Content-bound whole-run records for Stage 2S synthetic and formal execution."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, SupportsFloat, SupportsIndex, cast

from seismoflux.background.artifacts import canonical_json_bytes

RecordMode = Literal["synthetic_acceptance", "formal_development"]
_VALID_GATE_STATUSES = {
    "invalid",
    "evidence_insufficient",
    "failed",
    "passed_development_signal",
}
_METRIC_KEYS = (
    "S1_minus_S0:IG",
    "S1_minus_S0:recall",
    "S1_minus_SP:IG",
    "S1_minus_SP:recall",
)
_MODELS = ("S0", "S1", "SP")


class Stage2SRecordError(ValueError):
    """Raised when a whole-run record is incomplete or mutable."""


def _freeze(value: object, *, location: str = "$") -> object:
    if value is None or isinstance(value, bool | str | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise Stage2SRecordError(f"non-finite float at {location}")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise Stage2SRecordError(f"non-string mapping key at {location}")
            frozen[key] = _freeze(item, location=f"{location}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return tuple(
            _freeze(item, location=f"{location}[{index}]") for index, item in enumerate(value)
        )
    raise Stage2SRecordError(f"unsupported whole-run value at {location}: {type(value).__name__}")


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    frozen = _freeze(value, location=label)
    if not isinstance(frozen, Mapping):
        raise Stage2SRecordError(f"{label} must be a mapping")
    return cast(Mapping[str, object], frozen)


def _mapping_sequence(value: object, *, label: str) -> tuple[Mapping[str, object], ...]:
    frozen = _freeze(value, location=label)
    if not isinstance(frozen, tuple) or not all(isinstance(item, Mapping) for item in frozen):
        raise Stage2SRecordError(f"{label} must be a sequence of mappings")
    return cast(tuple[Mapping[str, object], ...], frozen)


def _number(value: object, *, label: str) -> float:
    if isinstance(value, bool | str | bytes | bytearray):
        raise Stage2SRecordError(f"{label} must be numeric")
    try:
        result = float(cast(SupportsFloat | SupportsIndex, value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise Stage2SRecordError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise Stage2SRecordError(f"{label} must be finite")
    return result


def _integer(value: object, *, label: str, positive: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise Stage2SRecordError(f"{label} must be an integer")
    if positive and value <= 0:
        raise Stage2SRecordError(f"{label} must be positive")
    return value


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise Stage2SRecordError(f"{label} must be non-empty text")
    return value


def _metric_mapping(value: object, *, label: str) -> Mapping[str, object]:
    mapping = _mapping(value, label=label)
    if tuple(mapping) != _METRIC_KEYS:
        raise Stage2SRecordError(f"{label} must contain the four ordered primary metrics")
    for key, item in mapping.items():
        _number(item, label=f"{label}.{key}")
    return mapping


def _validate_gate(gate: Mapping[str, object]) -> Mapping[str, object]:
    overall = _metric_mapping(gate.get("overall_macros"), label="gate overall_macros")
    claim_limited = gate.get("claim_limited")
    if not isinstance(claim_limited, bool):
        raise Stage2SRecordError("gate claim_limited must be boolean")
    expected_limit = (
        "claim_limited_to_sequence_associated_continuation_not_broad_regional_gain"
        if claim_limited
        else "no_sequence_interpretation_limit"
    )
    if gate.get("interpretation_limit") != expected_limit:
        raise Stage2SRecordError("gate interpretation_limit differs from claim_limited")
    expected_scope = (
        "sequence_associated_continuation_only"
        if claim_limited
        else (
            "broad_regional_gain_not_sequence_limited"
            if gate.get("status") == "passed_development_signal"
            else "no_sequence_interpretation_limit"
        )
    )
    if gate.get("interpretation_scope") != expected_scope:
        raise Stage2SRecordError("gate interpretation_scope differs from status and claim_limited")
    return overall


def _validate_sequence(
    sequence: Mapping[str, object],
    *,
    gate: Mapping[str, object],
    overall: Mapping[str, object],
) -> None:
    component_count = _integer(
        sequence.get("component_count"),
        label="sequence component_count",
        positive=True,
    )
    event_count = _integer(
        sequence.get("event_resampling_unit_count"),
        label="sequence event_resampling_unit_count",
        positive=True,
    )
    residual = _metric_mapping(
        sequence.get("global_residual"),
        label="sequence global_residual",
    )
    if any(
        _number(residual[key], label=f"sequence global_residual.{key}") != 0.0
        for key in ("S1_minus_S0:recall", "S1_minus_SP:recall")
    ):
        raise Stage2SRecordError("sequence recall residual must be exactly zero")
    primary_model_recall = _mapping(
        sequence.get("primary_model_recall"),
        label="sequence primary_model_recall",
    )
    if tuple(primary_model_recall) != _MODELS:
        raise Stage2SRecordError("sequence primary_model_recall must be ordered S0/S1/SP")
    model_totals = {
        model: _number(
            primary_model_recall[model],
            label=f"sequence primary_model_recall.{model}",
        )
        for model in _MODELS
    }
    if any(value < 0.0 for value in model_totals.values()):
        raise Stage2SRecordError("sequence primary model recall must be non-negative")
    components = _mapping_sequence(sequence.get("components"), label="sequence components")
    if len(components) != component_count:
        raise Stage2SRecordError("sequence component_count differs from components")
    component_ids: list[str] = []
    component_event_counts: dict[str, int] = {}
    component_contributions: dict[str, Mapping[str, object]] = {}
    component_model_hits: dict[str, Mapping[str, object]] = {}
    all_event_ids: set[str] = set()
    event_fraction_sum = 0.0
    for index, component in enumerate(components):
        label = f"sequence components[{index}]"
        component_id = _text(component.get("component_id"), label=f"{label}.component_id")
        event_ids_raw = component.get("event_ids")
        if not isinstance(event_ids_raw, tuple) or not event_ids_raw:
            raise Stage2SRecordError(f"{label}.event_ids must be a non-empty sequence")
        event_ids = tuple(_text(value, label=f"{label}.event_ids") for value in event_ids_raw)
        if (
            len(set(event_ids)) != len(event_ids)
            or component_id != min(event_ids, key=lambda value: value.encode("utf-8"))
            or any(event_id in all_event_ids for event_id in event_ids)
        ):
            raise Stage2SRecordError(f"{label} event identities are invalid or duplicated")
        all_event_ids.update(event_ids)
        declared_count = _integer(
            component.get("event_count"),
            label=f"{label}.event_count",
            positive=True,
        )
        if declared_count != len(event_ids):
            raise Stage2SRecordError(f"{label}.event_count differs from event_ids")
        event_fraction = _number(
            component.get("event_fraction"),
            label=f"{label}.event_fraction",
        )
        if not math.isclose(
            event_fraction,
            declared_count / event_count,
            rel_tol=0.0,
            abs_tol=1.0e-10,
        ):
            raise Stage2SRecordError(f"{label}.event_fraction differs from event count")
        event_fraction_sum += event_fraction
        if (
            _number(
                component.get("origin_time_span_days"),
                label=f"{label}.origin_time_span_days",
            )
            < 0.0
            or _number(
                component.get("max_pairwise_geodesic_distance_km"),
                label=f"{label}.max_pairwise_geodesic_distance_km",
            )
            < 0.0
        ):
            raise Stage2SRecordError(f"{label} span/distance must be non-negative")
        contributions = _metric_mapping(
            component.get("contributions"),
            label=f"{label}.contributions",
        )
        model_hits = _mapping(component.get("model_hits"), label=f"{label}.model_hits")
        if tuple(model_hits) != _MODELS:
            raise Stage2SRecordError(f"{label}.model_hits must be ordered S0/S1/SP")
        for model in _MODELS:
            hit = _mapping(
                model_hits[model],
                label=f"{label}.model_hits.{model}",
            )
            if set(hit) != {"raw", "fraction"}:
                raise Stage2SRecordError(f"{label}.model_hits.{model} fields are incomplete")
            raw = _number(hit["raw"], label=f"{label}.model_hits.{model}.raw")
            if raw < 0.0:
                raise Stage2SRecordError(f"{label}.model hit raw must be non-negative")
            fraction = hit["fraction"]
            if fraction is None:
                if model_totals[model] > 1.0e-12:
                    raise Stage2SRecordError(
                        f"{label}.model hit fraction cannot be null with positive macro"
                    )
            elif not math.isclose(
                _number(
                    fraction,
                    label=f"{label}.model_hits.{model}.fraction",
                ),
                raw / model_totals[model],
                rel_tol=0.0,
                abs_tol=1.0e-10,
            ):
                raise Stage2SRecordError(f"{label}.model hit fraction differs from raw/macro")
        information_gain = _mapping(
            component.get("information_gain"),
            label=f"{label}.information_gain",
        )
        if tuple(information_gain) != ("S1_minus_S0", "S1_minus_SP"):
            raise Stage2SRecordError(f"{label}.information_gain contrasts are incomplete")
        for contrast in ("S1_minus_S0", "S1_minus_SP"):
            value = _mapping(
                information_gain[contrast],
                label=f"{label}.information_gain.{contrast}",
            )
            if set(value) != {"raw", "fraction"}:
                raise Stage2SRecordError(
                    f"{label}.information_gain.{contrast} fields are incomplete"
                )
            raw = _number(
                value["raw"],
                label=f"{label}.information_gain.{contrast}.raw",
            )
            if raw != _number(
                contributions[f"{contrast}:IG"],
                label=f"{label}.contributions.{contrast}:IG",
            ):
                raise Stage2SRecordError(f"{label}.information_gain raw differs from contribution")
            macro = _number(overall[f"{contrast}:IG"], label=f"gate {contrast}:IG")
            fraction = value["fraction"]
            if fraction is None:
                if macro > 1.0e-12:
                    raise Stage2SRecordError(f"{label}.information_gain fraction cannot be null")
            elif not math.isclose(
                _number(
                    fraction,
                    label=f"{label}.information_gain.{contrast}.fraction",
                ),
                raw / macro,
                rel_tol=0.0,
                abs_tol=1.0e-10,
            ):
                raise Stage2SRecordError(
                    f"{label}.information_gain fraction differs from raw/macro"
                )
        component_ids.append(component_id)
        component_event_counts[component_id] = declared_count
        component_contributions[component_id] = contributions
        component_model_hits[component_id] = model_hits
    if (
        component_ids != sorted(component_ids, key=lambda value: value.encode("utf-8"))
        or len(all_event_ids) != event_count
        or not math.isclose(
            event_fraction_sum,
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-10,
        )
    ):
        raise Stage2SRecordError("sequence component partition does not close")
    for model in _MODELS:
        observed = math.fsum(
            _number(
                cast(Mapping[str, object], component_model_hits[component_id][model])["raw"],
                label=f"sequence {component_id} {model} raw",
            )
            for component_id in component_ids
        )
        if not math.isclose(
            observed,
            model_totals[model],
            rel_tol=0.0,
            abs_tol=1.0e-10,
        ):
            raise Stage2SRecordError(f"sequence {model} hit raw does not close")
    for key in _METRIC_KEYS:
        observed = math.fsum(
            _number(
                component_contributions[component_id][key],
                label=f"sequence {component_id} {key}",
            )
            for component_id in component_ids
        ) + _number(residual[key], label=f"sequence residual {key}")
        if not math.isclose(
            observed,
            _number(overall[key], label=f"gate overall {key}"),
            rel_tol=0.0,
            abs_tol=1.0e-10,
        ):
            raise Stage2SRecordError(f"sequence additive closure failed for {key}")
    largest_count_id = min(
        component_ids,
        key=lambda component_id: (
            -component_event_counts[component_id],
            component_id.encode("utf-8"),
        ),
    )
    if sequence.get("largest_count_component_id") != largest_count_id:
        raise Stage2SRecordError("sequence largest-count component is invalid")
    largest_count = _mapping(
        sequence.get("largest_count_component"),
        label="sequence largest_count_component",
    )
    if largest_count.get("component_id") != largest_count_id:
        raise Stage2SRecordError("sequence largest-count detail identity differs")
    largest_gain_ids = _metric_mapping_ids(
        sequence.get("largest_gain_component_id"),
        component_ids=component_ids,
        label="sequence largest_gain_component_id",
    )
    leave_count = _metric_mapping(
        sequence.get("leave_largest_count_out"),
        label="sequence leave_largest_count_out",
    )
    leave_gain = _metric_mapping(
        sequence.get("leave_largest_gain_out"),
        label="sequence leave_largest_gain_out",
    )
    alias = _metric_mapping(
        sequence.get("leave_out_residual"),
        label="sequence leave_out_residual",
    )
    if any(
        _number(alias[key], label=f"sequence leave alias {key}")
        != _number(leave_gain[key], label=f"sequence leave gain {key}")
        for key in _METRIC_KEYS
    ):
        raise Stage2SRecordError("sequence leave-out alias differs from largest-gain")
    largest_gain_detail = _mapping(
        sequence.get("largest_gain_component"),
        label="sequence largest_gain_component",
    )
    if tuple(largest_gain_detail) != _METRIC_KEYS:
        raise Stage2SRecordError("sequence largest-gain detail family is incomplete")
    for key in _METRIC_KEYS:
        strongest = min(
            component_ids,
            key=lambda component_id: (
                -_number(
                    component_contributions[component_id][key],
                    label=f"sequence {component_id} {key}",
                ),
                component_id.encode("utf-8"),
            ),
        )
        if largest_gain_ids[key] != strongest:
            raise Stage2SRecordError(f"sequence largest gain identity is invalid for {key}")
        strongest_raw = _number(
            component_contributions[strongest][key],
            label=f"sequence strongest {key}",
        )
        expected_gain_leave = _number(overall[key], label=f"gate overall {key}") - strongest_raw
        expected_count_leave = _number(
            overall[key],
            label=f"gate overall {key}",
        ) - _number(
            component_contributions[largest_count_id][key],
            label=f"sequence largest-count {key}",
        )
        if not math.isclose(
            _number(leave_gain[key], label=f"sequence leave gain {key}"),
            expected_gain_leave,
            rel_tol=0.0,
            abs_tol=1.0e-10,
        ) or not math.isclose(
            _number(leave_count[key], label=f"sequence leave count {key}"),
            expected_count_leave,
            rel_tol=0.0,
            abs_tol=1.0e-10,
        ):
            raise Stage2SRecordError(f"sequence leave-out differs for {key}")
        detail = _mapping(
            largest_gain_detail[key],
            label=f"sequence largest_gain_component.{key}",
        )
        if (
            detail.get("component_id") != strongest
            or _number(detail.get("raw_contribution"), label=f"sequence raw {key}") != strongest_raw
            or not math.isclose(
                _number(detail.get("leave_out"), label=f"sequence detail leave {key}"),
                expected_gain_leave,
                rel_tol=0.0,
                abs_tol=1.0e-10,
            )
        ):
            raise Stage2SRecordError(f"sequence largest-gain detail differs for {key}")
    claim_limited = any(
        _number(leave_gain[key], label=f"sequence leave gain {key}") <= 1.0e-12
        for key in _METRIC_KEYS
    )
    if sequence.get("claim_limited") is not claim_limited:
        raise Stage2SRecordError("sequence claim_limited differs from leave-outs")
    expected_limit = (
        "claim_limited_to_sequence_associated_continuation_not_broad_regional_gain"
        if claim_limited
        else "no_sequence_interpretation_limit"
    )
    if (
        sequence.get("interpretation_limit") != expected_limit
        or gate.get("claim_limited") is not claim_limited
        or gate.get("interpretation_limit") != expected_limit
    ):
        raise Stage2SRecordError("sequence and gate interpretation limits differ")


def _metric_mapping_ids(
    value: object,
    *,
    component_ids: Sequence[str],
    label: str,
) -> Mapping[str, str]:
    mapping = _mapping(value, label=label)
    if tuple(mapping) != _METRIC_KEYS:
        raise Stage2SRecordError(f"{label} must contain four ordered metrics")
    allowed = set(component_ids)
    result: dict[str, str] = {}
    for key, item in mapping.items():
        component_id = _text(item, label=f"{label}.{key}")
        if component_id not in allowed:
            raise Stage2SRecordError(f"{label}.{key} references an unknown component")
        result[key] = component_id
    return MappingProxyType(result)


def _validate_descriptive(
    descriptive: Mapping[str, object],
    *,
    overall: Mapping[str, object],
) -> None:
    if tuple(descriptive) != ("SP_minus_S0",):
        raise Stage2SRecordError("descriptive point estimates must contain only SP_minus_S0")
    comparison = _mapping(
        descriptive["SP_minus_S0"],
        label="descriptive SP_minus_S0",
    )
    expected_fields = {
        "information_gain",
        "recall_gain",
        "derivation",
        "inferential_status",
        "included_in_bootstrap_ci",
        "included_in_gate",
    }
    if set(comparison) != expected_fields:
        raise Stage2SRecordError("descriptive SP_minus_S0 fields are incomplete")
    expected_ig = _number(
        overall["S1_minus_S0:IG"],
        label="gate S1_minus_S0:IG",
    ) - _number(
        overall["S1_minus_SP:IG"],
        label="gate S1_minus_SP:IG",
    )
    expected_recall = _number(
        overall["S1_minus_S0:recall"],
        label="gate S1_minus_S0:recall",
    ) - _number(
        overall["S1_minus_SP:recall"],
        label="gate S1_minus_SP:recall",
    )
    if not math.isclose(
        _number(comparison["information_gain"], label="descriptive information_gain"),
        expected_ig,
        rel_tol=0.0,
        abs_tol=1.0e-10,
    ) or not math.isclose(
        _number(comparison["recall_gain"], label="descriptive recall_gain"),
        expected_recall,
        rel_tol=0.0,
        abs_tol=1.0e-10,
    ):
        raise Stage2SRecordError("descriptive SP_minus_S0 does not match linear derivation")
    if (
        comparison["derivation"] != "S1_minus_S0_minus_S1_minus_SP"
        or comparison["inferential_status"] != "descriptive_point_estimate_only"
        or comparison["included_in_bootstrap_ci"] is not False
        or comparison["included_in_gate"] is not False
    ):
        raise Stage2SRecordError("descriptive SP_minus_S0 interpretation is invalid")


@dataclass(frozen=True, slots=True)
class Stage2SWholeRunRecord:
    """One complete immutable result binding, including all 2,000 Bootstrap rows."""

    mode: RecordMode
    identity: Mapping[str, object]
    input_receipts: Mapping[str, object]
    fold_fit_summaries: Sequence[Mapping[str, object]]
    issue_prediction_summaries: Sequence[Mapping[str, object]]
    seal_chain: Mapping[str, object]
    cell_scores: Sequence[Mapping[str, object]]
    bootstrap_summary: Mapping[str, object]
    bootstrap_rows: Sequence[Sequence[float]]
    regional_evidence: Mapping[str, object]
    sequence_evidence: Mapping[str, object]
    descriptive_point_estimates: Mapping[str, object]
    latency_evidence: Sequence[Mapping[str, object]]
    gate_evidence: Mapping[str, object]
    artifact_sha256_by_name: Mapping[str, object]
    run_record_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.mode not in {"synthetic_acceptance", "formal_development"}:
            raise Stage2SRecordError("unknown Stage 2S whole-run mode")
        identity = _mapping(self.identity, label="identity")
        inputs = _mapping(self.input_receipts, label="input_receipts")
        fold_fits = _mapping_sequence(
            self.fold_fit_summaries,
            label="fold_fit_summaries",
        )
        issues = _mapping_sequence(
            self.issue_prediction_summaries,
            label="issue_prediction_summaries",
        )
        seals = _mapping(self.seal_chain, label="seal_chain")
        cells = _mapping_sequence(self.cell_scores, label="cell_scores")
        bootstrap_summary = _mapping(
            self.bootstrap_summary,
            label="bootstrap_summary",
        )
        rows_frozen = _freeze(self.bootstrap_rows, location="bootstrap_rows")
        if not isinstance(rows_frozen, tuple) or not all(
            isinstance(row, tuple) for row in rows_frozen
        ):
            raise Stage2SRecordError("bootstrap_rows must be a sequence of rows")
        bootstrap_rows = cast(tuple[tuple[object, ...], ...], rows_frozen)
        regional = _mapping(self.regional_evidence, label="regional_evidence")
        sequence = _mapping(self.sequence_evidence, label="sequence_evidence")
        descriptive = _mapping(
            self.descriptive_point_estimates,
            label="descriptive_point_estimates",
        )
        latency = _mapping_sequence(self.latency_evidence, label="latency_evidence")
        gate = _mapping(self.gate_evidence, label="gate_evidence")
        artifacts = _mapping(
            self.artifact_sha256_by_name,
            label="artifact_sha256_by_name",
        )
        if len(fold_fits) != 3:
            raise Stage2SRecordError("whole-run record requires three fold fits")
        if not issues:
            raise Stage2SRecordError("whole-run record requires issue predictions")
        if len(cells) != 9:
            raise Stage2SRecordError("whole-run record requires all nine cell scores")
        cell_identities = tuple(
            (item.get("fold_index"), item.get("horizon_days")) for item in cells
        )
        expected_cells = tuple(
            (fold_index, horizon_days) for fold_index in (1, 2, 3) for horizon_days in (7, 30, 90)
        )
        if cell_identities != expected_cells:
            raise Stage2SRecordError("cell score records must be ordered fold then 7/30/90 days")
        for index, cell in enumerate(cells):
            event_ids = cell.get("event_ids")
            if (
                not isinstance(event_ids, tuple)
                or not event_ids
                or any(not isinstance(value, str) or not value for value in event_ids)
                or len(set(event_ids)) != len(event_ids)
            ):
                raise Stage2SRecordError(
                    f"cell_scores[{index}].event_ids must be non-empty and unique"
                )
            model_hits = _mapping(
                cell.get("hit_by_model"),
                label=f"cell_scores[{index}].hit_by_model",
            )
            if tuple(model_hits) != _MODELS:
                raise Stage2SRecordError(
                    f"cell_scores[{index}] model hits must be ordered S0/S1/SP"
                )
            for model in _MODELS:
                hits = model_hits[model]
                if (
                    not isinstance(hits, tuple)
                    or len(hits) != len(event_ids)
                    or any(not isinstance(value, bool) for value in hits)
                ):
                    raise Stage2SRecordError(
                        f"cell_scores[{index}] {model} hits must be aligned booleans"
                    )
        if len(bootstrap_rows) != 2000 or any(len(row) != 4 for row in bootstrap_rows):
            raise Stage2SRecordError(
                "whole-run record requires exactly 2,000 four-family Bootstrap rows"
            )
        for row_index, row in enumerate(bootstrap_rows):
            for column_index, value in enumerate(row):
                _number(
                    value,
                    label=f"bootstrap_rows[{row_index}][{column_index}]",
                )
        if len(latency) != 2 or tuple(item.get("delay_days") for item in latency) != (
            1,
            7,
        ):
            raise Stage2SRecordError("latency evidence must be ordered 1 then 7 days")
        status = gate.get("status")
        if status not in _VALID_GATE_STATUSES:
            raise Stage2SRecordError("whole-run gate status is invalid")
        overall = _validate_gate(gate)
        _validate_sequence(sequence, gate=gate, overall=overall)
        _validate_descriptive(descriptive, overall=overall)
        column_order = bootstrap_summary.get("column_order")
        if column_order is not None and (
            not isinstance(column_order, tuple) or tuple(column_order) != _METRIC_KEYS
        ):
            raise Stage2SRecordError(
                "Bootstrap column order must contain only the four primary families"
            )
        intervals = bootstrap_summary.get("intervals")
        if intervals is not None:
            interval_mapping = _mapping(intervals, label="bootstrap intervals")
            if tuple(interval_mapping) != _METRIC_KEYS:
                raise Stage2SRecordError("Bootstrap intervals must exclude descriptive SP_minus_S0")
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "input_receipts", inputs)
        object.__setattr__(self, "fold_fit_summaries", fold_fits)
        object.__setattr__(self, "issue_prediction_summaries", issues)
        object.__setattr__(self, "seal_chain", seals)
        object.__setattr__(self, "cell_scores", cells)
        object.__setattr__(self, "bootstrap_summary", bootstrap_summary)
        object.__setattr__(self, "bootstrap_rows", bootstrap_rows)
        object.__setattr__(self, "regional_evidence", regional)
        object.__setattr__(self, "sequence_evidence", sequence)
        object.__setattr__(self, "descriptive_point_estimates", descriptive)
        object.__setattr__(self, "latency_evidence", latency)
        object.__setattr__(self, "gate_evidence", gate)
        object.__setattr__(self, "artifact_sha256_by_name", artifacts)
        digest = hashlib.sha256(
            canonical_json_bytes(self.as_mapping(include_hash=False))
        ).hexdigest()
        object.__setattr__(self, "run_record_sha256", digest)

    def as_mapping(self, *, include_hash: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": 1,
            "record_type": "stage2s_whole_run_record",
            "mode": self.mode,
            "identity": self.identity,
            "input_receipts": self.input_receipts,
            "fold_fit_summaries": self.fold_fit_summaries,
            "issue_prediction_summaries": self.issue_prediction_summaries,
            "seal_chain": self.seal_chain,
            "cell_scores": self.cell_scores,
            "bootstrap_summary": self.bootstrap_summary,
            "bootstrap_rows": self.bootstrap_rows,
            "regional_evidence": self.regional_evidence,
            "sequence_evidence": self.sequence_evidence,
            "descriptive_point_estimates": self.descriptive_point_estimates,
            "latency_evidence": self.latency_evidence,
            "gate_evidence": self.gate_evidence,
            "artifact_sha256_by_name": self.artifact_sha256_by_name,
        }
        if include_hash:
            value["run_record_sha256"] = self.run_record_sha256
        return value

    def to_canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.as_mapping()) + b"\n"


@dataclass(frozen=True, slots=True)
class Stage2SSyntheticAcceptanceRecord:
    """Target-free acceptance wrapper for one synthetic whole-run record."""

    whole_run: Stage2SWholeRunRecord
    real_input_path_open_count: int = 0
    real_target_byte_read_count: int = 0
    acceptance_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.whole_run.mode != "synthetic_acceptance":
            raise Stage2SRecordError("synthetic acceptance requires a synthetic whole-run record")
        if self.real_input_path_open_count != 0 or self.real_target_byte_read_count != 0:
            raise Stage2SRecordError("synthetic acceptance cannot read real inputs or targets")
        digest = hashlib.sha256(
            canonical_json_bytes(
                {
                    "schema_version": 1,
                    "record_type": "stage2s_synthetic_acceptance_record",
                    "whole_run_record_sha256": self.whole_run.run_record_sha256,
                    "real_input_path_open_count": self.real_input_path_open_count,
                    "real_target_byte_read_count": self.real_target_byte_read_count,
                }
            )
        ).hexdigest()
        object.__setattr__(self, "acceptance_sha256", digest)

    def as_mapping(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "record_type": "stage2s_synthetic_acceptance_record",
            "whole_run_record_sha256": self.whole_run.run_record_sha256,
            "real_input_path_open_count": self.real_input_path_open_count,
            "real_target_byte_read_count": self.real_target_byte_read_count,
            "acceptance_sha256": self.acceptance_sha256,
        }


__all__ = [
    "RecordMode",
    "Stage2SRecordError",
    "Stage2SSyntheticAcceptanceRecord",
    "Stage2SWholeRunRecord",
]
