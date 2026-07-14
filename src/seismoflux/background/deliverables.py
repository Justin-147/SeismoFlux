"""Pure adaptation from stage-2 science results to immutable publications.

This module deliberately performs no catalog reads, fitting, simulation, scoring, or
fixed-file publication.  It converts one already-computed
``BackgroundPipelineResult`` into the four frozen bundle families, derives every model
attempt and adoption conclusion from the retained audit evidence, and publishes only
those content-addressed bundles.  The caller remains responsible for the final sealed
registry/report write.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

from seismoflux.background.artifacts import canonical_json_bytes
from seismoflux.background.config import BackgroundConfig
from seismoflux.background.evidence import (
    EXPECTED_SNAPSHOTS,
    MODEL_SIMPLICITY_ORDER,
    AuditedBackgroundModelEvidence,
    PairedInformationGainEvidence,
    assess_audited_g1,
    select_audited_background_model,
)
from seismoflux.background.execution import (
    ExecutionSeal,
    GitCommandRunner,
    build_address_inputs,
    require_execution_seal_unchanged,
    subprocess_git_runner,
)
from seismoflux.background.future import FutureHorizonSummary, FutureIssueEnsemble
from seismoflux.background.pipeline import (
    BackgroundPipelineFailure,
    BackgroundPipelineOutcome,
    BackgroundPipelineResult,
)
from seismoflux.background.pipeline_etas import ETASSnapshotAttempt
from seismoflux.background.publication import (
    BackgroundRegistry,
    BackgroundScientificSummary,
    BacktestBundle,
    BundleBinary,
    BundleDocument,
    BundlePublication,
    ExperimentBundle,
    FutureScientificSummary,
    G1Conclusion,
    GateOutcome,
    HorizonScientificSummary,
    ModelAttemptRecord,
    ModelBundle,
    ModelSnapshotScientificSummary,
    ProcessedBundle,
    RepresentativeScientificSummary,
    ScientificFailureSummary,
    SelectionConclusion,
    ValidationBootstrapScientificSummary,
    build_background_registry,
    publish_backtest_bundle,
    publish_experiment_bundle,
    publish_model_bundle,
    publish_processed_bundle,
)
from seismoflux.background.scientific import ScientificJson, scientific_json, scientific_mapping

SnapshotId = Literal["fold_1", "fold_2", "fold_3", "fold_4", "final_validation"]
ModelId = Literal["uniform_poisson", "spatial_poisson", "etas"]
CandidateModelId = Literal["spatial_poisson", "etas"]
_ARROW_MEDIA_TYPE = "application/vnd.apache.arrow.file"
_ARROW_BATCH_SIZE = 65_536
_INTEGRATION_GRID_SCHEMA = pa.schema(
    (
        pa.field("cell_size_km", pa.float64(), nullable=False),
        pa.field("cell_id", pa.string(), nullable=False),
        pa.field("row", pa.int64(), nullable=False),
        pa.field("column", pa.int64(), nullable=False),
        pa.field("representative_x_km", pa.float64(), nullable=False),
        pa.field("representative_y_km", pa.float64(), nullable=False),
        pa.field("clipped_area_km2", pa.float64(), nullable=False),
    )
)
_REPRESENTATIVE_INTENSITY_SCHEMA = pa.schema(
    (
        pa.field("cell_id", pa.string(), nullable=False),
        pa.field("row", pa.int64(), nullable=False),
        pa.field("column", pa.int64(), nullable=False),
        pa.field("representative_x_km", pa.float64(), nullable=False),
        pa.field("representative_y_km", pa.float64(), nullable=False),
        pa.field("clipped_area_km2", pa.float64(), nullable=False),
        pa.field("background_intensity", pa.float64(), nullable=False),
        pa.field("triggering_intensity", pa.float64(), nullable=False),
        pa.field("total_intensity", pa.float64(), nullable=False),
    )
)
_FUTURE_SCHEMA = pa.schema(
    (
        pa.field("record_kind", pa.string(), nullable=False),
        pa.field("issue_date_local", pa.string(), nullable=False),
        pa.field("issue_id", pa.string(), nullable=False),
        pa.field("horizon_days", pa.int32(), nullable=False),
        pa.field("replicate_index", pa.int16(), nullable=True),
        pa.field("replicate_count", pa.int32(), nullable=True),
        pa.field("cell_size_km", pa.float64(), nullable=True),
        pa.field("cell_id", pa.string(), nullable=True),
        pa.field("expected_count", pa.float64(), nullable=True),
    )
)


def _scientific_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(scientific_json(value))).hexdigest()


def _protocol_sha256(config: BackgroundConfig) -> str:
    return hashlib.sha256(canonical_json_bytes(config.model_dump(mode="python"))).hexdigest()


def _document(relative_path: str, value: object) -> BundleDocument:
    return BundleDocument(
        relative_path,
        scientific_mapping(value, location=f"bundle:{relative_path}"),
    )


def _arrow_file_bytes(schema: pa.Schema, columns: dict[str, object]) -> bytes:
    """Encode one explicit-schema, metadata-free Arrow IPC file deterministically."""

    if tuple(columns) != tuple(field.name for field in schema):
        raise ValueError("Arrow columns must follow the explicit schema order")
    arrays = [pa.array(columns[field.name], type=field.type) for field in schema]
    table = pa.Table.from_arrays(arrays, schema=schema)
    sink = pa.BufferOutputStream()
    options = ipc.IpcWriteOptions(
        metadata_version=ipc.MetadataVersion.V5,
        use_legacy_format=False,
        compression=None,
        use_threads=False,
        emit_dictionary_deltas=False,
        unify_dictionaries=False,
    )
    with ipc.new_file(sink, schema, options=options) as writer:
        writer.write_table(table, max_chunksize=_ARROW_BATCH_SIZE)
    return cast(bytes, sink.getvalue().to_pybytes())


def _arrow_batch_file_bytes(
    schema: pa.Schema,
    column_batches: Iterable[dict[str, object]],
) -> bytes:
    """Stream deterministic logical batches without materializing all sparse rows."""

    sink = pa.BufferOutputStream()
    options = ipc.IpcWriteOptions(
        metadata_version=ipc.MetadataVersion.V5,
        use_legacy_format=False,
        compression=None,
        use_threads=False,
        emit_dictionary_deltas=False,
        unify_dictionaries=False,
    )
    expected_names = tuple(field.name for field in schema)
    batch_count = 0
    with ipc.new_file(sink, schema, options=options) as writer:
        for columns in column_batches:
            if tuple(columns) != expected_names:
                raise ValueError("Arrow batch columns must follow the explicit schema order")
            arrays = [pa.array(columns[field.name], type=field.type) for field in schema]
            batch = pa.RecordBatch.from_arrays(arrays, schema=schema)
            if batch.num_rows <= 0:
                raise ValueError("Arrow record batches must contain at least one row")
            writer.write_batch(batch)
            batch_count += 1
    if batch_count == 0:
        raise ValueError("Arrow IPC file must contain at least one record batch")
    return cast(bytes, sink.getvalue().to_pybytes())


def _arrow_schema_document(schema: pa.Schema) -> dict[str, object]:
    return {
        "fields": [
            {
                "name": field.name,
                "nullable": field.nullable,
                "type": str(field.type),
            }
            for field in schema
        ],
        "format": "Apache Arrow IPC file",
        "schema_metadata": None,
    }


def _paired_by_snapshot(
    evidence: AuditedBackgroundModelEvidence,
) -> dict[str, PairedInformationGainEvidence]:
    paired = {item.candidate.snapshot_id: item for item in evidence.development_folds}
    if evidence.validation is not None:
        paired[evidence.validation.candidate.snapshot_id] = evidence.validation
    return paired


def _failure_by_snapshot(
    evidence: AuditedBackgroundModelEvidence,
) -> dict[str, tuple[str, ...]]:
    return {
        snapshot_id: tuple(sorted(set(reasons)))
        for snapshot_id, reasons in evidence.failed_snapshot_reasons
    }


def _successful_attempt(
    *,
    model_id: ModelId,
    snapshot_id: SnapshotId,
    evidence: AuditedBackgroundModelEvidence,
    paired: PairedInformationGainEvidence,
    gate_names: tuple[str, ...],
) -> ModelAttemptRecord:
    score = paired.candidate
    if score.model_id != model_id or score.snapshot_id != snapshot_id:
        raise ValueError("attempt score does not match its model/snapshot identity")
    if len(gate_names) != len(score.numerical_gate_evidence_ids):
        raise ValueError("attempt gate names do not align with numerical evidence IDs")
    gates = tuple(
        sorted(
            (
                GateOutcome(gate_id=name, status="passed", evidence_id=evidence_id)
                for name, evidence_id in zip(
                    gate_names,
                    score.numerical_gate_evidence_ids,
                    strict=True,
                )
            ),
            key=lambda item: item.gate_id,
        )
    )
    return ModelAttemptRecord(
        model_id=model_id,
        snapshot_id=snapshot_id,
        status="succeeded",
        failure_reasons=(),
        variant=evidence.model_variant_id,
        parameter_identity={
            "model_id": model_id,
            "model_variant_id": evidence.model_variant_id,
            "parameter_snapshot_id": score.parameter_snapshot_id,
            "protocol_sha256": score.protocol_sha256,
            "selected_mc": score.selected_mc,
            "snapshot_id": snapshot_id,
        },
        gates=gates,
        score_ids=(score.score_id,),
    )


def _failed_attempt(
    *,
    model_id: ModelId,
    snapshot_id: SnapshotId,
    evidence: AuditedBackgroundModelEvidence,
    reasons: tuple[str, ...],
) -> ModelAttemptRecord:
    normalized_reasons = tuple(sorted(set(reasons)))
    failure_evidence_id = _scientific_sha256(
        {
            "model_id": model_id,
            "model_variant_id": evidence.model_variant_id,
            "snapshot_id": snapshot_id,
            "failure_reasons": normalized_reasons,
        }
    )
    return ModelAttemptRecord(
        model_id=model_id,
        snapshot_id=snapshot_id,
        status="failed",
        failure_reasons=normalized_reasons,
        variant=evidence.model_variant_id,
        parameter_identity={
            "model_id": model_id,
            "model_variant_id": evidence.model_variant_id,
            "parameter_snapshot_id": None,
            "protocol_sha256": evidence.protocol_sha256,
            "snapshot_id": snapshot_id,
        },
        gates=(
            GateOutcome(
                gate_id="snapshot_completion",
                status="failed",
                evidence_id=failure_evidence_id,
            ),
        ),
        score_ids=(),
    )


def _poisson_attempts(
    evidence: AuditedBackgroundModelEvidence,
) -> tuple[ModelAttemptRecord, ...]:
    model_id = evidence.model_id
    if model_id not in {"uniform_poisson", "spatial_poisson"}:
        raise ValueError("Poisson attempt adaptation requires a Poisson model family")
    paired = _paired_by_snapshot(evidence)
    failures = _failure_by_snapshot(evidence)
    gate_names = (
        ("causal_training_selection",)
        if model_id == "uniform_poisson"
        else ("snapshot_grid_convergence", "global_bandwidth_pre_score")
    )
    attempts: list[ModelAttemptRecord] = []
    for value in EXPECTED_SNAPSHOTS:
        snapshot_id = cast(SnapshotId, value)
        score = paired.get(snapshot_id)
        if score is not None:
            attempts.append(
                _successful_attempt(
                    model_id=model_id,
                    snapshot_id=snapshot_id,
                    evidence=evidence,
                    paired=score,
                    gate_names=gate_names,
                )
            )
        else:
            attempts.append(
                _failed_attempt(
                    model_id=model_id,
                    snapshot_id=snapshot_id,
                    evidence=evidence,
                    reasons=failures[snapshot_id],
                )
            )
    return tuple(attempts)


def _selection_audit_id(attempt: ETASSnapshotAttempt) -> str:
    return _scientific_sha256(
        {
            "fit_selection": attempt.fit_selection,
            "score_selection": attempt.score_selection,
        }
    )


def _etas_attempt_record(
    attempt: ETASSnapshotAttempt,
    evidence: AuditedBackgroundModelEvidence,
) -> ModelAttemptRecord:
    snapshot_id = cast(SnapshotId, attempt.definition.snapshot_id)
    causal_gate = GateOutcome(
        gate_id="causal_event_selection",
        status="passed",
        evidence_id=_selection_audit_id(attempt),
    )
    gates: list[GateOutcome] = [causal_gate]
    fit_result = attempt.fit_result
    if fit_result is None:
        gates.append(
            GateOutcome(
                gate_id="numerical_stability",
                status="failed",
                evidence_id=_scientific_sha256(
                    {
                        "snapshot_id": snapshot_id,
                        "failure_reasons": attempt.failure_reasons,
                        "fit_result": None,
                    }
                ),
            )
        )
    else:
        stability_status: Literal["passed", "failed"] = (
            "passed" if fit_result.stability.stable else "failed"
        )
        gates.append(
            GateOutcome(
                gate_id="numerical_stability",
                status=stability_status,
                evidence_id=_scientific_sha256(fit_result.stability),
            )
        )

    grid_gate = attempt.grid_gate_evidence
    if grid_gate is not None:
        gates.append(
            GateOutcome(
                gate_id="grid_convergence",
                status="passed" if grid_gate.passed else "failed",
                evidence_id=grid_gate.numerical_evidence_id,
            )
        )
    elif fit_result is not None and fit_result.stability.stable:
        gates.append(
            GateOutcome(
                gate_id="grid_convergence",
                status="failed",
                evidence_id=_scientific_sha256(
                    {
                        "snapshot_id": snapshot_id,
                        "failure_reasons": attempt.failure_reasons,
                        "grid_gate_evidence": None,
                    }
                ),
            )
        )
    else:
        gates.append(
            GateOutcome(
                gate_id="grid_convergence",
                status="not_applicable",
                evidence_id=_scientific_sha256(
                    {
                        "snapshot_id": snapshot_id,
                        "reason": "numerical_stability_not_passed",
                    }
                ),
            )
        )

    paired = attempt.paired_evidence
    upstream_failed = any(gate.status == "failed" for gate in gates)
    scoring_status: Literal["passed", "failed", "not_applicable"]
    if paired is not None:
        scoring_status = "passed"
        scoring_evidence_id = paired.candidate.score_id
    elif upstream_failed:
        scoring_status = "not_applicable"
        scoring_evidence_id = _scientific_sha256(
            {
                "snapshot_id": snapshot_id,
                "reason": "upstream_gate_not_passed",
            }
        )
    else:
        scoring_status = "failed"
        scoring_evidence_id = _scientific_sha256(
            {
                "snapshot_id": snapshot_id,
                "failure_reasons": attempt.failure_reasons,
                "paired_evidence": None,
            }
        )
    gates.append(
        GateOutcome(
            gate_id="scoring_completion",
            status=scoring_status,
            evidence_id=scoring_evidence_id,
        )
    )

    parameter_identity: dict[str, object] = {
        "fit_selection_evidence_id": _scientific_sha256(attempt.fit_selection),
        "model_id": "etas",
        "model_variant_id": evidence.model_variant_id,
        "parameter_snapshot_id": attempt.parameter_snapshot_id,
        "protocol_sha256": evidence.protocol_sha256,
        "score_selection_evidence_id": _scientific_sha256(attempt.score_selection),
        "selected_mc": attempt.selected_mc,
        "snapshot_id": snapshot_id,
    }
    if paired is not None:
        if attempt.failure_reasons:
            raise ValueError("successful ETAS attempt retains contradictory failure reasons")
        return ModelAttemptRecord(
            model_id="etas",
            snapshot_id=snapshot_id,
            status="succeeded",
            failure_reasons=(),
            variant=evidence.model_variant_id,
            parameter_identity=parameter_identity,
            gates=tuple(sorted(gates, key=lambda item: item.gate_id)),
            score_ids=(paired.candidate.score_id,),
        )
    return ModelAttemptRecord(
        model_id="etas",
        snapshot_id=snapshot_id,
        status="failed",
        failure_reasons=tuple(sorted(set(attempt.failure_reasons))),
        variant=evidence.model_variant_id,
        parameter_identity=parameter_identity,
        gates=tuple(sorted(gates, key=lambda item: item.gate_id)),
        score_ids=(),
    )


def _model_attempts(result: BackgroundPipelineResult) -> tuple[ModelAttemptRecord, ...]:
    attempts = (
        *_poisson_attempts(result.poisson.uniform_evidence),
        *_poisson_attempts(result.poisson.spatial_evidence),
        *(
            _etas_attempt_record(attempt, result.etas.etas_evidence)
            for attempt in result.etas.attempts
        ),
    )
    return tuple(attempts)


def _conclusions(
    result: BackgroundPipelineResult,
) -> tuple[G1Conclusion, SelectionConclusion]:
    audited = (
        result.poisson.uniform_evidence,
        result.poisson.spatial_evidence,
        result.etas.etas_evidence,
    )
    computed_g1 = assess_audited_g1(audited)
    if computed_g1 != result.g1:
        raise ValueError("pipeline G1 conclusion differs from its audited model evidence")
    computed_selection = select_audited_background_model(audited)
    if computed_selection != result.selection:
        raise ValueError("pipeline selection differs from its audited model evidence")

    by_variant = {item.model_variant_id: item.model_id for item in audited}
    if len(by_variant) != len(audited):
        raise ValueError("background model variants must be unique across model families")
    passing_models = tuple(
        cast(Literal["spatial_poisson", "etas"], model_id)
        for model_id, _variant_id, passed in computed_g1.model_pass
        if passed
    )
    g1 = G1Conclusion(
        passed=computed_g1.passed,
        passing_models=passing_models,
        evidence_ids=tuple(sorted(_scientific_sha256(item) for item in audited[1:])),
    )
    eligible_model_ids = tuple(
        model_id
        for model_id in MODEL_SIMPLICITY_ORDER
        if next(item for item in audited if item.model_id == model_id).eligible_for_selection
    )
    try:
        selected_model_id = by_variant[computed_selection.selected_model_variant_id]
        validation_best_model_id = by_variant[computed_selection.validation_best_model_variant_id]
    except KeyError as error:
        raise ValueError("selection names an unknown audited model variant") from error
    selection = SelectionConclusion(
        selected_model_id=selected_model_id,
        validation_best_model_id=validation_best_model_id,
        eligible_model_ids=eligible_model_ids,
        evidence_id=_scientific_sha256(
            {
                "audited_model_evidence_sha256": tuple(
                    _scientific_sha256(item) for item in audited
                ),
                "selection": computed_selection,
            }
        ),
    )
    if result.representative.selected_model_id != selected_model_id:
        raise ValueError("representative intensity does not use the derived selected model")
    return g1, selection


def _scientific_summary(
    result: BackgroundPipelineResult,
    model_attempts: tuple[ModelAttemptRecord, ...],
) -> BackgroundScientificSummary:
    """Project the retained audits into the small tracked stage-2 result surface."""

    evidence_by_model = {
        "uniform_poisson": result.poisson.uniform_evidence,
        "spatial_poisson": result.poisson.spatial_evidence,
        "etas": result.etas.etas_evidence,
    }
    paired_by_model = {
        model_id: _paired_by_snapshot(evidence) for model_id, evidence in evidence_by_model.items()
    }
    poisson_target_count = {
        cast(SnapshotId, snapshot.definition.snapshot_id): len(snapshot.target_event_ids)
        for snapshot in result.poisson.snapshots
    }
    etas_target_count = {
        cast(SnapshotId, attempt.definition.snapshot_id): len(
            attempt.score_selection.target_event_ids
        )
        for attempt in result.etas.attempts
    }
    if tuple(poisson_target_count) != EXPECTED_SNAPSHOTS or tuple(etas_target_count) != (
        EXPECTED_SNAPSHOTS
    ):
        raise ValueError("scientific summary target counts require all five frozen snapshots")

    snapshot_summaries: list[ModelSnapshotScientificSummary] = []
    for attempt in model_attempts:
        paired = paired_by_model[attempt.model_id].get(attempt.snapshot_id)
        target_event_count = (
            etas_target_count[attempt.snapshot_id]
            if attempt.model_id == "etas"
            else poisson_target_count[attempt.snapshot_id]
        )
        if paired is not None and len(paired.candidate.target_event_ids) != target_event_count:
            raise ValueError("scientific summary target count differs from retained score evidence")
        snapshot_summaries.append(
            ModelSnapshotScientificSummary(
                model_id=attempt.model_id,
                snapshot_id=attempt.snapshot_id,
                status=attempt.status,
                target_event_count=target_event_count,
                information_gain_nats_per_event=(
                    paired.information_gain_per_event if paired is not None else None
                ),
                score_id=paired.candidate.score_id if paired is not None else None,
            )
        )

    bootstrap_summaries: list[ValidationBootstrapScientificSummary] = []
    for outcome in (result.bootstrap.spatial_poisson, result.bootstrap.etas):
        interval = outcome.interval
        bootstrap_summaries.append(
            ValidationBootstrapScientificSummary(
                model_id=outcome.model_id,
                status="completed" if interval is not None else "skipped",
                point_estimate=(interval.point_estimate if interval is not None else None),
                lower=interval.lower if interval is not None else None,
                upper=interval.upper if interval is not None else None,
                replications=(
                    cast(Literal[2000], interval.replications) if interval is not None else None
                ),
                confidence_level=(interval.confidence_level if interval is not None else None),
                not_run_reason=outcome.not_run_reason,
            )
        )

    horizon_summaries: list[HorizonScientificSummary] = []
    for horizon_outcome in (result.horizons.spatial_poisson, result.horizons.etas):
        comparisons = horizon_outcome.comparisons
        horizon_summaries.append(
            HorizonScientificSummary(
                model_id=horizon_outcome.model_id,
                status="completed" if comparisons is not None else "skipped",
                comparison_count=len(comparisons) if comparisons is not None else 0,
                not_run_reason=horizon_outcome.not_run_reason,
            )
        )

    ensembles = result.future.ensembles
    return BackgroundScientificSummary(
        final_selected_mc=result.poisson.snapshots[-1].selected_mc,
        selected_kde_bandwidth_km=result.poisson.selected_bandwidth_km,
        snapshots=tuple(snapshot_summaries),
        validation_bootstrap=tuple(bootstrap_summaries),
        horizons=tuple(horizon_summaries),
        future=FutureScientificSummary(
            status="completed" if ensembles is not None else "skipped",
            issue_count=len(ensembles.issues) if ensembles is not None else 0,
            not_run_reason=result.future.not_run_reason,
        ),
        representative=RepresentativeScientificSummary(
            issue_date_local=cast(
                Literal["2025-06-26"],
                result.representative.issue_date_local,
            ),
            grid_cell_size_km=result.representative.primary_grid_cell_size_km,
            selected_model_id=result.representative.selected_model_id,
        ),
    )


def _integration_grid_artifacts(
    config: BackgroundConfig,
    result: BackgroundPipelineOutcome,
) -> tuple[bytes, dict[str, ScientificJson]]:
    cell_sizes: list[float] = []
    cell_ids: list[str] = []
    rows: list[int] = []
    columns: list[int] = []
    x_km: list[float] = []
    y_km: list[float] = []
    areas: list[float] = []
    summaries: list[dict[str, object]] = []
    for resolution in result.integration_grids.resolutions:
        summaries.append(
            {
                "cell_count": len(resolution.cells),
                "cell_size_km": resolution.cell_size_km,
                "total_clipped_area_km2": resolution.total_clipped_area_km2,
            }
        )
        for cell in resolution.cells:
            cell_sizes.append(resolution.cell_size_km)
            cell_ids.append(cell.cell_id)
            rows.append(cell.row)
            columns.append(cell.column)
            x_km.append(cell.representative_x_km)
            y_km.append(cell.representative_y_km)
            areas.append(cell.clipped_area_km2)
    arrow = _arrow_file_bytes(
        _INTEGRATION_GRID_SCHEMA,
        {
            "cell_size_km": cell_sizes,
            "cell_id": cell_ids,
            "row": rows,
            "column": columns,
            "representative_x_km": x_km,
            "representative_y_km": y_km,
            "clipped_area_km2": areas,
        },
    )
    summary = scientific_mapping(
        {
            "arrow": {
                "byte_count": len(arrow),
                "path": "integration_grids.arrow",
                "row_count": len(cell_ids),
                "schema": _arrow_schema_document(_INTEGRATION_GRID_SCHEMA),
                "sha256": hashlib.sha256(arrow).hexdigest(),
            },
            "equal_area_crs": config.integration.equal_area_crs,
            "primary_grid_cell_size_km": config.integration.primary_grid_cell_km,
            "resolutions": summaries,
            "study_area_km2": result.integration_grids.study_area_km2,
            "units": {
                "cell_size_km": "km",
                "clipped_area_km2": "km2",
                "representative_x_km": "km in equal_area_crs",
                "representative_y_km": "km in equal_area_crs",
            },
        },
        location="integration_grid_summary",
    )
    return arrow, summary


def _validate_representative_grid_alignment(result: BackgroundPipelineResult) -> None:
    representative = result.representative
    primary = result.integration_grids.at(representative.primary_grid_cell_size_km)
    cells = primary.cells
    if (
        representative.cell_ids != tuple(cell.cell_id for cell in cells)
        or representative.rows != tuple(cell.row for cell in cells)
        or representative.columns != tuple(cell.column for cell in cells)
        or not np.array_equal(
            representative.representative_x_km,
            np.asarray([cell.representative_x_km for cell in cells], dtype=np.float64),
        )
        or not np.array_equal(
            representative.representative_y_km,
            np.asarray([cell.representative_y_km for cell in cells], dtype=np.float64),
        )
        or not np.array_equal(
            representative.clipped_area_km2,
            np.asarray([cell.clipped_area_km2 for cell in cells], dtype=np.float64),
        )
    ):
        raise ValueError("representative intensity rows differ from the frozen 25-km grid")


def _representative_artifacts(
    config: BackgroundConfig,
    result: BackgroundPipelineResult,
) -> tuple[bytes, dict[str, ScientificJson]]:
    _validate_representative_grid_alignment(result)
    representative = result.representative
    render = representative.render
    arrow = _arrow_file_bytes(
        _REPRESENTATIVE_INTENSITY_SCHEMA,
        {
            "cell_id": representative.cell_ids,
            "row": representative.rows,
            "column": representative.columns,
            "representative_x_km": representative.representative_x_km,
            "representative_y_km": representative.representative_y_km,
            "clipped_area_km2": representative.clipped_area_km2,
            "background_intensity": representative.background_intensity,
            "triggering_intensity": representative.triggering_intensity,
            "total_intensity": representative.total_intensity,
        },
    )
    layer_summary = {
        name: {
            "integrated_events_per_day": math.fsum(
                float(intensity) * float(area)
                for intensity, area in zip(
                    values,
                    representative.clipped_area_km2,
                    strict=True,
                )
            ),
            "maximum_events_per_km2_per_day": float(np.max(values)),
            "minimum_events_per_km2_per_day": float(np.min(values)),
        }
        for name, values in (
            ("background", representative.background_intensity),
            ("triggering", representative.triggering_intensity),
            ("total", representative.total_intensity),
        )
    }
    summary = scientific_mapping(
        {
            "arrow": {
                "byte_count": len(arrow),
                "path": "conditional_intensity/representative_intensity.arrow",
                "row_count": len(representative.cell_ids),
                "schema": _arrow_schema_document(_REPRESENTATIVE_INTENSITY_SCHEMA),
                "sha256": hashlib.sha256(arrow).hexdigest(),
            },
            "eligible_parent_event_ids": representative.eligible_parent_event_ids,
            "equal_area_crs": config.integration.equal_area_crs,
            "issue_date_local": representative.issue_date_local,
            "issue_time_utc": representative.issue_time_utc,
            "layer_summary": layer_summary,
            "primary_grid_cell_size_km": representative.primary_grid_cell_size_km,
            "render": {
                "colorbar_label": render.colorbar_label,
                "colormap_name": render.colormap_name,
                "dpi": render.dpi,
                "font_family": render.font_family,
                "footer_label": render.footer_label,
                "height_px": render.height_px,
                "panel_titles": render.panel_titles,
                "png_byte_count": len(render.png_bytes),
                "png_metadata": render.png_metadata,
                "png_sha256": hashlib.sha256(render.png_bytes).hexdigest(),
                "width_px": render.width_px,
            },
            "selected_model_id": representative.selected_model_id,
            "selected_model_variant_id": representative.selected_model_variant_id,
            "units": {
                "clipped_area_km2": "km2",
                "intensity": "expected_events_per_day_per_km2",
                "representative_x_km": "km in equal_area_crs",
                "representative_y_km": "km in equal_area_crs",
            },
        },
        location="representative_conditional_intensity",
    )
    return arrow, summary


def _future_batch_columns(
    issue: FutureIssueEnsemble,
    horizon: FutureHorizonSummary,
) -> dict[str, object]:
    issue_date_local = issue.issue_date_local
    issue_id = issue.issue_id
    horizon_days = horizon.horizon_days
    replicate_counts = horizon.replicate_counts
    grids = horizon.grids
    record_kind = ["replicate_count"] * len(replicate_counts)
    issue_dates = [issue_date_local] * len(replicate_counts)
    issue_ids = [issue_id] * len(replicate_counts)
    horizons = [horizon_days] * len(replicate_counts)
    replicate_indices: list[int | None] = list(range(len(replicate_counts)))
    counts: list[int | None] = list(replicate_counts)
    cell_sizes: list[float | None] = [None] * len(replicate_counts)
    cell_ids: list[str | None] = [None] * len(replicate_counts)
    expected_counts: list[float | None] = [None] * len(replicate_counts)
    for grid in grids:
        for cell in grid.cells:
            record_kind.append("sparse_expected_cell")
            issue_dates.append(issue_date_local)
            issue_ids.append(issue_id)
            horizons.append(horizon_days)
            replicate_indices.append(None)
            counts.append(None)
            cell_sizes.append(grid.cell_size_km)
            cell_ids.append(cell.cell_id)
            expected_counts.append(cell.expected_count)
    return {
        "record_kind": record_kind,
        "issue_date_local": issue_dates,
        "issue_id": issue_ids,
        "horizon_days": horizons,
        "replicate_index": replicate_indices,
        "replicate_count": counts,
        "cell_size_km": cell_sizes,
        "cell_id": cell_ids,
        "expected_count": expected_counts,
    }


def _future_artifacts(
    result: BackgroundPipelineResult,
) -> tuple[bytes | None, dict[str, ScientificJson]]:
    outcome = result.future
    ensembles = outcome.ensembles
    if ensembles is None:
        return None, scientific_mapping(
            {
                "arrow": None,
                "not_run_reason": outcome.not_run_reason,
                "status": "skipped",
            },
            location="future_summary",
        )
    issue_summaries: list[dict[str, object]] = []
    row_count = 0

    def batches() -> Iterable[dict[str, object]]:
        nonlocal row_count
        for issue in ensembles.issues:
            horizon_summaries: list[dict[str, object]] = []
            for horizon in issue.horizons:
                batch = _future_batch_columns(issue, horizon)
                row_count += len(cast(list[str], batch["record_kind"]))
                horizon_summaries.append(
                    {
                        "grids": tuple(
                            {
                                "cell_size_km": grid.cell_size_km,
                                "expected_total": grid.expected_total,
                                "nonzero_cell_count": len(grid.cells),
                            }
                            for grid in horizon.grids
                        ),
                        "horizon_days": horizon.horizon_days,
                        "mean_count": horizon.mean_count,
                        "quantiles": horizon.quantiles,
                        "replicate_count": len(horizon.replicate_counts),
                    }
                )
                yield batch
            issue_summaries.append(
                {
                    "horizons": horizon_summaries,
                    "issue_date_local": issue.issue_date_local,
                    "issue_id": issue.issue_id,
                }
            )

    arrow = _arrow_batch_file_bytes(_FUTURE_SCHEMA, batches())
    return arrow, scientific_mapping(
        {
            "arrow": {
                "byte_count": len(arrow),
                "path": "future/future_sparse_counts.arrow",
                "row_count": row_count,
                "schema": _arrow_schema_document(_FUTURE_SCHEMA),
                "sha256": hashlib.sha256(arrow).hexdigest(),
            },
            "issue_count": len(ensembles.issues),
            "issues": issue_summaries,
            "status": "completed",
            "units": {
                "cell_size_km": "km",
                "expected_count": "expected_events_in_horizon",
                "replicate_count": "simulated_events_in_horizon",
            },
        },
        location="future_summary",
    )


def _poisson_publication_document(
    result: BackgroundPipelineResult,
) -> dict[str, ScientificJson]:
    """Retain model/audit state without repeating grid and KDE work arrays in JSON."""

    snapshots: list[dict[str, object]] = []
    for snapshot in result.poisson.snapshots:
        snapshots.append(
            {
                "definition": snapshot.definition,
                "grid_gate_evidence": snapshot.grid_gate_evidence,
                "kde_family": tuple(
                    {
                        "bandwidth_km": bandwidth,
                        "normalization_mass": model.normalization_mass,
                        "rate_per_day": model.rate_per_day,
                        "training_event_count": model.mixture.training_event_count,
                    }
                    for bandwidth, model in snapshot.kde_family
                ),
                "rate_per_day": snapshot.rate_per_day,
                "selected_mc": snapshot.selected_mc,
                "target_event_ids": snapshot.target_event_ids,
                "training_duration_days": snapshot.training_duration_days,
                "training_event_count": snapshot.training_event_count,
                "training_event_ids": snapshot.training_event_ids,
                "training_evidence_id": snapshot.training_evidence_id,
                "uniform_model": snapshot.uniform_model,
            }
        )
    return scientific_mapping(
        {
            "bandwidth_fold_audits": result.poisson.bandwidth_fold_audits,
            "bandwidth_selection": result.poisson.bandwidth_selection,
            "pre_score_gate_evidence": result.poisson.pre_score_gate_evidence,
            "protocol_sha256": result.poisson.protocol_sha256,
            "snapshots": snapshots,
            "spatial_evidence": result.poisson.spatial_evidence,
            "uniform_evidence": result.poisson.uniform_evidence,
        },
        location="poisson_publication",
    )


def _pipeline_failure_attempts(
    failure: BackgroundPipelineFailure,
) -> tuple[ModelAttemptRecord, ...]:
    variants = {
        "uniform_poisson": "uniform_poisson/not_fitted",
        "spatial_poisson": "spatial_poisson/not_fitted",
        "etas": "etas/not_fitted",
    }
    attempts: list[ModelAttemptRecord] = []
    for model_id in MODEL_SIMPLICITY_ORDER:
        for snapshot_value in EXPECTED_SNAPSHOTS:
            snapshot_id = cast(SnapshotId, snapshot_value)
            gate_evidence_id = _scientific_sha256(
                {
                    "failure_stage": failure.failure_stage,
                    "failure_reason_code": failure.failure_reason_code,
                    "failure_reasons": failure.failure_reasons,
                    "model_id": model_id,
                    "protocol_sha256": failure.protocol_sha256,
                    "snapshot_id": snapshot_id,
                }
            )
            attempts.append(
                ModelAttemptRecord(
                    model_id=model_id,
                    snapshot_id=snapshot_id,
                    status="not_run",
                    failure_reasons=failure.failure_reasons,
                    variant=variants[model_id],
                    parameter_identity={
                        "failure_stage": failure.failure_stage,
                        "model_id": model_id,
                        "parameter_snapshot_id": None,
                        "protocol_sha256": failure.protocol_sha256,
                        "snapshot_id": snapshot_id,
                    },
                    gates=(
                        GateOutcome(
                            gate_id=f"upstream_{failure.failure_stage}",
                            status="not_applicable",
                            evidence_id=gate_evidence_id,
                        ),
                    ),
                    score_ids=(),
                )
            )
    return tuple(attempts)


def _pipeline_failure_document(
    failure: BackgroundPipelineFailure,
) -> dict[str, ScientificJson]:
    return scientific_mapping(
        {
            "completeness_evidence_sha256": _scientific_sha256(failure.completeness),
            "failure_reasons": failure.failure_reasons,
            "failure_reason_code": failure.failure_reason_code,
            "failure_stage": failure.failure_stage,
            "integration_grid_evidence_sha256": _scientific_sha256(failure.integration_grids),
            "numerical_regression_evidence_sha256": _scientific_sha256(failure.regressions),
            "pre_score_gate_evidence": failure.pre_score_gate_evidence,
            "protocol_sha256": failure.protocol_sha256,
        },
        location="background_pipeline_failure",
    )


def _pipeline_failure_conclusions(
    failure: BackgroundPipelineFailure,
) -> tuple[
    BackgroundScientificSummary,
    tuple[ModelAttemptRecord, ...],
    G1Conclusion,
    SelectionConclusion,
]:
    attempts = _pipeline_failure_attempts(failure)
    failure_evidence_id = _scientific_sha256(_pipeline_failure_document(failure))
    skip_reason = "; ".join(failure.failure_reasons)
    candidate_model_ids: tuple[CandidateModelId, ...] = ("spatial_poisson", "etas")
    summary = BackgroundScientificSummary(
        outcome_status="scientific_gate_failed",
        failure=ScientificFailureSummary(
            failure_stage=failure.failure_stage,
            failure_reason_code=failure.failure_reason_code,
            failure_reasons=failure.failure_reasons,
            evidence_id=failure_evidence_id,
        ),
        final_selected_mc=None,
        selected_kde_bandwidth_km=None,
        snapshots=tuple(
            ModelSnapshotScientificSummary(
                model_id=attempt.model_id,
                snapshot_id=attempt.snapshot_id,
                status="not_run",
                target_event_count=None,
                information_gain_nats_per_event=None,
                score_id=None,
            )
            for attempt in attempts
        ),
        validation_bootstrap=tuple(
            ValidationBootstrapScientificSummary(
                model_id=model_id,
                status="skipped",
                point_estimate=None,
                lower=None,
                upper=None,
                replications=None,
                confidence_level=None,
                not_run_reason=skip_reason,
            )
            for model_id in candidate_model_ids
        ),
        horizons=tuple(
            HorizonScientificSummary(
                model_id=model_id,
                status="skipped",
                comparison_count=0,
                not_run_reason=skip_reason,
            )
            for model_id in candidate_model_ids
        ),
        future=FutureScientificSummary(
            status="skipped",
            issue_count=0,
            not_run_reason=skip_reason,
        ),
        representative=None,
    )
    g1 = G1Conclusion(
        status="not_evaluable",
        passed=False,
        passing_models=(),
        evidence_ids=(failure_evidence_id,),
    )
    selection = SelectionConclusion(
        status="not_evaluable",
        selected_model_id=None,
        validation_best_model_id=None,
        eligible_model_ids=(),
        evidence_id=failure_evidence_id,
    )
    return summary, attempts, g1, selection


def _build_background_failure_deliverables(
    config: BackgroundConfig,
    failure: BackgroundPipelineFailure,
) -> BackgroundDeliverables:
    scientific_summary, model_attempts, g1, selection = _pipeline_failure_conclusions(failure)
    integration_arrow, integration_summary = _integration_grid_artifacts(config, failure)
    failure_document = _pipeline_failure_document(failure)
    common_identity = {
        "failure_evidence_sha256": _scientific_sha256(failure_document),
        "failure_reason_code": failure.failure_reason_code,
        "failure_stage": failure.failure_stage,
        "protocol_sha256": failure.protocol_sha256,
        "protocol_version": config.protocol_version,
    }
    processed = ProcessedBundle(
        scientific_mapping(
            {
                **common_identity,
                "integration_grid_arrow_sha256": hashlib.sha256(integration_arrow).hexdigest(),
            },
            location="failed_processed_parameter_identity",
        ),
        (
            _document("failure.json", failure_document),
            _document("completeness.json", {"snapshots": failure.completeness}),
            _document("integration_grid.json", integration_summary),
            BundleBinary(
                "integration_grids.arrow",
                integration_arrow,
                media_type=_ARROW_MEDIA_TYPE,
            ),
            _document("numerical_regressions.json", failure.regressions),
        ),
    )
    model = ModelBundle(
        scientific_mapping(
            {**common_identity, "attempts": model_attempts},
            location="failed_model_parameter_identity",
        ),
        (
            _document("attempts.json", {"model_attempts": model_attempts}),
            _document("failure.json", failure_document),
            _document(
                "poisson_pre_score_gate.json",
                {"evidence": failure.pre_score_gate_evidence},
            ),
        ),
    )
    backtest = BacktestBundle(
        scientific_mapping(
            {
                **common_identity,
                "g1": g1,
                "scientific_summary_sha256": _scientific_sha256(scientific_summary),
                "selection": selection,
            },
            location="failed_backtest_parameter_identity",
        ),
        (
            _document("failure.json", failure_document),
            _document("g1.json", {"conclusion": g1}),
            _document("scientific_summary.json", scientific_summary),
            _document("selection.json", {"conclusion": selection}),
        ),
    )
    experiment = ExperimentBundle(
        scientific_mapping(
            {
                **common_identity,
                "conditional_intensity_status": "skipped",
                "future_status": "skipped",
            },
            location="failed_experiment_parameter_identity",
        ),
        (
            _document(
                "conditional_intensity/status.json",
                {"status": "skipped", "reason": failure.failure_reasons},
            ),
            _document("failure.json", failure_document),
            _document(
                "future/status.json",
                {"status": "skipped", "reason": failure.failure_reasons},
            ),
        ),
    )
    return BackgroundDeliverables(
        processed=processed,
        model=model,
        backtest=backtest,
        experiment=experiment,
        model_attempts=model_attempts,
        scientific_summary=scientific_summary,
        g1=g1,
        selection=selection,
        stage3_allowed=False,
    )


@dataclass(frozen=True, slots=True)
class BackgroundDeliverables:
    """The four bundles and the conclusions from which the registry must be built."""

    processed: ProcessedBundle
    model: ModelBundle
    backtest: BacktestBundle
    experiment: ExperimentBundle
    model_attempts: tuple[ModelAttemptRecord, ...]
    scientific_summary: BackgroundScientificSummary
    g1: G1Conclusion
    selection: SelectionConclusion
    stage3_allowed: bool

    def __post_init__(self) -> None:
        expected = tuple(
            (model_id, snapshot_id)
            for model_id in MODEL_SIMPLICITY_ORDER
            for snapshot_id in EXPECTED_SNAPSHOTS
        )
        actual = tuple((item.model_id, item.snapshot_id) for item in self.model_attempts)
        if actual != expected:
            raise ValueError("deliverables must retain all 3x5 attempts in frozen order")
        for attempt, summary in zip(
            self.model_attempts,
            self.scientific_summary.snapshots,
            strict=True,
        ):
            if attempt.status != summary.status:
                raise ValueError("deliverable summary status differs from its model attempt")
            if attempt.score_ids != ((summary.score_id,) if summary.score_id is not None else ()):
                raise ValueError("deliverable summary score differs from its model attempt")
        complete_success = {
            model_id: all(
                attempt.status == "succeeded"
                for attempt in self.model_attempts
                if attempt.model_id == model_id
            )
            for model_id in MODEL_SIMPLICITY_ORDER
        }
        if any(not complete_success[model_id] for model_id in self.g1.passing_models):
            raise ValueError("G1 passing models must retain five successful attempts")
        if any(not complete_success[model_id] for model_id in self.selection.eligible_model_ids):
            raise ValueError("selection-eligible models must retain five successful attempts")
        failure = self.scientific_summary.failure
        if failure is not None and any(
            attempt.failure_reasons != failure.failure_reasons for attempt in self.model_attempts
        ):
            raise ValueError("failed deliverable attempts differ from scientific failure")
        if self.stage3_allowed != self.g1.passed:
            raise ValueError("stage3 allowance must be derived exactly from G1")


@dataclass(frozen=True, slots=True)
class PublishedBackgroundDeliverables:
    """Four published bundles plus an in-memory registry ready for sealed writing."""

    processed: BundlePublication
    model: BundlePublication
    backtest: BundlePublication
    experiment: BundlePublication
    registry: BackgroundRegistry

    @property
    def bundle_publications(self) -> tuple[BundlePublication, ...]:
        return (self.processed, self.model, self.backtest, self.experiment)


def build_background_deliverables(
    config: BackgroundConfig,
    result: BackgroundPipelineOutcome,
) -> BackgroundDeliverables:
    """Adapt one complete or expected-negative result without recomputing scores."""

    expected_protocol = _protocol_sha256(config)
    if result.protocol_sha256 != expected_protocol:
        raise ValueError("pipeline result does not match the frozen background protocol")
    if isinstance(result, BackgroundPipelineFailure):
        return _build_background_failure_deliverables(config, result)
    model_attempts = _model_attempts(result)
    g1, selection = _conclusions(result)
    scientific_summary = _scientific_summary(result, model_attempts)

    if tuple(item.cell_size_km for item in result.integration_grids.resolutions) != (
        config.integration.grid_cells_km
    ):
        raise ValueError("pipeline integration grids differ from the frozen protocol")
    integration_arrow, integration_summary = _integration_grid_artifacts(config, result)

    processed_documents = (
        _document("completeness.json", {"snapshots": result.completeness}),
        _document("integration_grid.json", integration_summary),
        BundleBinary(
            "integration_grids.arrow",
            integration_arrow,
            media_type=_ARROW_MEDIA_TYPE,
        ),
        _document("numerical_regressions.json", result.regressions),
    )
    processed = ProcessedBundle(
        scientific_mapping(
            {
                "grid_cell_sizes_km": config.integration.grid_cells_km,
                "integration_grid_arrow_sha256": hashlib.sha256(integration_arrow).hexdigest(),
                "protocol_sha256": result.protocol_sha256,
                "protocol_version": config.protocol_version,
                "regression_evidence_sha256": _scientific_sha256(result.regressions),
                "snapshot_ids": EXPECTED_SNAPSHOTS,
            },
            location="processed_parameter_identity",
        ),
        processed_documents,
    )

    poisson_document = _poisson_publication_document(result)
    etas_document = scientific_mapping(result.etas, location="etas_publication")
    model_documents = (
        _document("attempts.json", {"model_attempts": model_attempts}),
        _document("etas.json", etas_document),
        _document("poisson.json", poisson_document),
    )
    model = ModelBundle(
        scientific_mapping(
            {
                "attempts": model_attempts,
                "etas_result_sha256": _scientific_sha256(etas_document),
                "model_variants": {
                    "etas": result.etas.etas_evidence.model_variant_id,
                    "spatial_poisson": result.poisson.spatial_evidence.model_variant_id,
                    "uniform_poisson": result.poisson.uniform_evidence.model_variant_id,
                },
                "poisson_result_sha256": _scientific_sha256(poisson_document),
                "protocol_sha256": result.protocol_sha256,
                "protocol_version": config.protocol_version,
                "selected_bandwidth_km": result.poisson.selected_bandwidth_km,
                "snapshot_ids": EXPECTED_SNAPSHOTS,
            },
            location="model_parameter_identity",
        ),
        model_documents,
    )

    audited_evidence = {
        "etas": result.etas.etas_evidence,
        "spatial_poisson": result.poisson.spatial_evidence,
        "uniform_poisson": result.poisson.uniform_evidence,
    }
    backtest_documents = (
        _document("audited_scores.json", audited_evidence),
        _document("bootstrap.json", result.bootstrap),
        _document("g1.json", {"conclusion": g1, "pipeline_assessment": result.g1}),
        _document("horizons.json", result.horizons),
        _document("scientific_summary.json", scientific_summary),
        _document(
            "selection.json",
            {"conclusion": selection, "pipeline_selection": result.selection},
        ),
    )
    backtest = BacktestBundle(
        scientific_mapping(
            {
                "audited_evidence_sha256": {
                    key: _scientific_sha256(value) for key, value in audited_evidence.items()
                },
                "bootstrap_sha256": _scientific_sha256(result.bootstrap),
                "g1": g1,
                "horizons_sha256": _scientific_sha256(result.horizons),
                "protocol_sha256": result.protocol_sha256,
                "protocol_version": config.protocol_version,
                "scientific_summary_sha256": _scientific_sha256(scientific_summary),
                "selection": selection,
            },
            location="backtest_parameter_identity",
        ),
        backtest_documents,
    )

    representative_arrow, representative_document = _representative_artifacts(config, result)
    future_arrow, future_document = _future_artifacts(result)
    png_payload = result.representative.render.png_bytes
    experiment_documents: list[BundleDocument | BundleBinary] = [
        _document(
            "conditional_intensity/conditional_intensity.json",
            representative_document,
        ),
        BundleBinary(
            "conditional_intensity/representative_intensity.arrow",
            representative_arrow,
            media_type=_ARROW_MEDIA_TYPE,
        ),
        BundleBinary(
            "conditional_intensity/conditional_intensity.png",
            png_payload,
            media_type="image/png",
        ),
        _document("future/future_summary.json", future_document),
    ]
    if future_arrow is not None:
        experiment_documents.append(
            BundleBinary(
                "future/future_sparse_counts.arrow",
                future_arrow,
                media_type=_ARROW_MEDIA_TYPE,
            )
        )
    experiment = ExperimentBundle(
        scientific_mapping(
            {
                "future_arrow_sha256": (
                    hashlib.sha256(future_arrow).hexdigest() if future_arrow is not None else None
                ),
                "future_summary_sha256": _scientific_sha256(future_document),
                "future_status": (
                    "completed" if result.future.ensembles is not None else "skipped"
                ),
                "png_sha256": hashlib.sha256(png_payload).hexdigest(),
                "protocol_sha256": result.protocol_sha256,
                "protocol_version": config.protocol_version,
                "representative_arrow_sha256": hashlib.sha256(representative_arrow).hexdigest(),
                "representative_issue_date_local": result.representative.issue_date_local,
                "representative_result_sha256": _scientific_sha256(representative_document),
                "selected_model_id": selection.selected_model_id,
                "selected_model_variant_id": (result.representative.selected_model_variant_id),
            },
            location="experiment_parameter_identity",
        ),
        tuple(experiment_documents),
    )
    return BackgroundDeliverables(
        processed=processed,
        model=model,
        backtest=backtest,
        experiment=experiment,
        model_attempts=model_attempts,
        scientific_summary=scientific_summary,
        g1=g1,
        selection=selection,
        stage3_allowed=g1.passed,
    )


def publish_background_deliverables(
    project_root: Path,
    config: BackgroundConfig,
    execution_seal: ExecutionSeal,
    deliverables: BackgroundDeliverables,
    *,
    runner: GitCommandRunner = subprocess_git_runner,
) -> PublishedBackgroundDeliverables:
    """Publish only the four bundles and build (but do not write) their registry."""

    if not isinstance(project_root, Path):
        raise TypeError("project_root must be pathlib.Path")
    if not isinstance(execution_seal, ExecutionSeal):
        raise TypeError("execution_seal must be ExecutionSeal")
    if not isinstance(deliverables, BackgroundDeliverables):
        raise TypeError("deliverables must be BackgroundDeliverables")
    expected_protocol = _protocol_sha256(config)
    if execution_seal.protocol_sha256 != expected_protocol:
        raise ValueError("execution seal protocol differs from the background config")
    address_inputs = build_address_inputs(
        config,
        execution_seal.repository,
        {"artifact_role": "execution_seal_validation", "schema_version": 1},
        uv_lock_sha256=config.inputs.environment_lock_sha256,
    )
    expected_input_hashes = cast(dict[str, str], address_inputs["input_hashes"])
    if expected_input_hashes != execution_seal.input_hash_mapping():
        raise ValueError("execution seal input hashes differ from the frozen config")
    require_execution_seal_unchanged(
        project_root,
        config,
        execution_seal,
        runner=runner,
    )

    identity = execution_seal.repository
    lock_hash = config.inputs.environment_lock_sha256
    processed = publish_processed_bundle(
        project_root,
        config,
        identity,
        deliverables.processed,
        uv_lock_sha256=lock_hash,
    )
    model = publish_model_bundle(
        project_root,
        config,
        identity,
        deliverables.model,
        uv_lock_sha256=lock_hash,
    )
    backtest = publish_backtest_bundle(
        project_root,
        config,
        identity,
        deliverables.backtest,
        uv_lock_sha256=lock_hash,
    )
    experiment = publish_experiment_bundle(
        project_root,
        config,
        identity,
        deliverables.experiment,
        uv_lock_sha256=lock_hash,
    )
    publications = (processed, model, backtest, experiment)
    registry = build_background_registry(
        config,
        identity,
        publications,
        deliverables.model_attempts,
        scientific_summary=deliverables.scientific_summary,
        g1=deliverables.g1,
        selection=deliverables.selection,
        stage3_allowed=deliverables.stage3_allowed,
        uv_lock_sha256=lock_hash,
    )
    if registry.code_commit != identity.code_commit:
        raise ValueError("registry code commit differs from the execution seal")
    if registry.input_hashes != execution_seal.input_hash_mapping():
        raise ValueError("registry input hashes differ from the execution seal")
    return PublishedBackgroundDeliverables(
        processed=processed,
        model=model,
        backtest=backtest,
        experiment=experiment,
        registry=registry,
    )


__all__ = [
    "BackgroundDeliverables",
    "PublishedBackgroundDeliverables",
    "build_background_deliverables",
    "publish_background_deliverables",
]
