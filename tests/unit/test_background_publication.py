from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Literal, cast

import pytest
import yaml

import seismoflux.background.execution as execution_module
import seismoflux.background.publication as publication_module
from seismoflux.background.artifacts import (
    ArtifactConflictError,
    ProjectRelativePath,
    canonical_json_bytes,
)
from seismoflux.background.config import BackgroundConfig
from seismoflux.background.execution import (
    CommandResult,
    ExecutionSeal,
    ExecutionSealError,
    GitCommandRunner,
    RepositoryIdentity,
)
from seismoflux.background.publication import (
    BackgroundRegistry,
    BackgroundScientificSummary,
    BacktestBundle,
    BundleBinary,
    BundleDocument,
    BundlePublication,
    ExperimentBundle,
    FixedFileConflictError,
    FixedFilePublicationError,
    FutureScientificSummary,
    G1Conclusion,
    GateOutcome,
    HorizonScientificSummary,
    ModelAttemptRecord,
    ModelBundle,
    ModelSnapshotScientificSummary,
    ProcessedBundle,
    RegistryValidationError,
    RepresentativeScientificSummary,
    ScientificFailureSummary,
    SelectionConclusion,
    ValidationBootstrapScientificSummary,
    build_background_registry,
    publish_backtest_bundle,
    publish_experiment_bundle,
    publish_fixed_project_file,
    publish_model_bundle,
    publish_processed_bundle,
    publish_registry_and_report,
    publish_registry_and_report_sealed,
    registry_payload_bytes,
    render_report_from_registry_payload,
    validate_registry_payload,
)

HEAD = "a" * 40
TAG_COMMIT = "1" * 40
FREEZE_TAG: Literal["v0.2.0-background-protocol"] = "v0.2.0-background-protocol"
MODEL_ORDER: tuple[
    Literal["uniform_poisson", "spatial_poisson", "etas"],
    ...,
] = ("uniform_poisson", "spatial_poisson", "etas")
SNAPSHOT_ORDER: tuple[
    Literal["fold_1", "fold_2", "fold_3", "fold_4", "final_validation"],
    ...,
] = ("fold_1", "fold_2", "fold_3", "fold_4", "final_validation")
BUNDLE_ORDER: tuple[
    Literal["processed", "model", "backtest", "experiment"],
    ...,
] = ("processed", "model", "backtest", "experiment")


@pytest.fixture(scope="module")
def background() -> BackgroundConfig:
    raw = yaml.safe_load(Path("configs/background.yaml").read_text(encoding="utf-8"))
    return BackgroundConfig.model_validate(raw)


@pytest.fixture(scope="module")
def identity() -> RepositoryIdentity:
    return RepositoryIdentity(
        code_commit=HEAD,
        branch="main",
        upstream="origin/main",
        upstream_commit=HEAD,
        freeze_tag=FREEZE_TAG,
        freeze_tag_commit=TAG_COMMIT,
        git_available=True,
        worktree_clean=True,
        tag_is_ancestor=True,
        upstream_matches_head=True,
    )


def _bundles() -> tuple[ProcessedBundle, ModelBundle, BacktestBundle, ExperimentBundle]:
    scientific_summary = _scientific_summary()
    scientific_summary_sha256 = hashlib.sha256(
        canonical_json_bytes(scientific_summary.model_dump(mode="python"))
    ).hexdigest()
    return (
        ProcessedBundle(
            {"grid_cells_km": [50.0, 25.0, 12.5], "snapshot_count": 5},
            (
                BundleDocument(
                    "grids.json",
                    {"cell_ids": ["g12500000_r+0000000_c+0000000"], "area_km2": [1.0]},
                ),
            ),
        ),
        ModelBundle(
            {"model_family": "background", "snapshot_count": 5},
            (BundleDocument("models.json", {"attempt_count": 15, "status": "complete"}),),
        ),
        BacktestBundle(
            {
                "endpoint": "g1",
                "fold_count": 4,
                "scientific_summary_sha256": scientific_summary_sha256,
            },
            (
                BundleDocument(
                    "metrics.json",
                    {"metric": "information_gain", "value": 0.1},
                ),
                BundleDocument(
                    "scientific_summary.json",
                    scientific_summary.model_dump(mode="python"),
                ),
            ),
        ),
        ExperimentBundle(
            {"representative_issue": "2025-06-26", "replicates": 128},
            (
                BundleDocument(
                    "conditional_intensity.json",
                    {"layer": "conditional_intensity", "relative_values": [0.25, 0.75]},
                ),
            ),
        ),
    )


def _publish_all(
    project_root: Path,
    background: BackgroundConfig,
    identity: RepositoryIdentity,
) -> tuple[BundlePublication, ...]:
    processed, model, backtest, experiment = _bundles()
    lock_hash = background.inputs.environment_lock_sha256
    return (
        publish_processed_bundle(
            project_root,
            background,
            identity,
            processed,
            uv_lock_sha256=lock_hash,
        ),
        publish_model_bundle(
            project_root,
            background,
            identity,
            model,
            uv_lock_sha256=lock_hash,
        ),
        publish_backtest_bundle(
            project_root,
            background,
            identity,
            backtest,
            uv_lock_sha256=lock_hash,
        ),
        publish_experiment_bundle(
            project_root,
            background,
            identity,
            experiment,
            uv_lock_sha256=lock_hash,
        ),
    )


def _attempts() -> tuple[ModelAttemptRecord, ...]:
    attempts: list[ModelAttemptRecord] = []
    for model_id in MODEL_ORDER:
        for snapshot_id in SNAPSHOT_ORDER:
            failed = model_id == "etas" and snapshot_id == "fold_2"
            gate_id = "numerical_stability" if model_id == "etas" else "grid_convergence"
            attempts.append(
                ModelAttemptRecord(
                    model_id=model_id,
                    snapshot_id=snapshot_id,
                    status="failed" if failed else "succeeded",
                    failure_reasons=("numerical_stability",) if failed else (),
                    variant=f"{model_id}/frozen-v1",
                    parameter_identity={
                        "model_id": model_id,
                        "snapshot_id": snapshot_id,
                        "variant": "frozen-v1",
                    },
                    gates=(
                        GateOutcome(
                            gate_id=gate_id,
                            status="failed" if failed else "passed",
                            evidence_id=f"gate/{model_id}/{snapshot_id}/{gate_id}",
                        ),
                    ),
                    score_ids=() if failed else (f"score/{model_id}/{snapshot_id}",),
                )
            )
    return tuple(attempts)


def _scientific_summary() -> BackgroundScientificSummary:
    information_gains: dict[
        Literal["uniform_poisson", "spatial_poisson", "etas"],
        tuple[float | None, ...],
    ] = {
        "uniform_poisson": (0.0, 0.0, 0.0, 0.0, 0.0),
        "spatial_poisson": (0.11, 0.12, 0.13, 0.14, 0.2),
        "etas": (0.31, None, 0.33, 0.34, 0.3),
    }
    snapshots: list[ModelSnapshotScientificSummary] = []
    for model_id in MODEL_ORDER:
        for snapshot_index, snapshot_id in enumerate(SNAPSHOT_ORDER):
            failed = model_id == "etas" and snapshot_id == "fold_2"
            snapshots.append(
                ModelSnapshotScientificSummary(
                    model_id=model_id,
                    snapshot_id=snapshot_id,
                    status="failed" if failed else "succeeded",
                    target_event_count=1,
                    information_gain_nats_per_event=information_gains[model_id][snapshot_index],
                    score_id=(None if failed else f"score/{model_id}/{snapshot_id}"),
                )
            )
    return BackgroundScientificSummary(
        final_selected_mc=2.6,
        selected_kde_bandwidth_km=80.0,
        snapshots=tuple(snapshots),
        validation_bootstrap=(
            ValidationBootstrapScientificSummary(
                model_id="spatial_poisson",
                status="completed",
                point_estimate=0.2,
                lower=0.17,
                upper=0.29,
                replications=2000,
                confidence_level=0.95,
                not_run_reason=None,
            ),
            ValidationBootstrapScientificSummary(
                model_id="etas",
                status="completed",
                point_estimate=0.3,
                lower=0.24,
                upper=0.41,
                replications=2000,
                confidence_level=0.95,
                not_run_reason=None,
            ),
        ),
        horizons=(
            HorizonScientificSummary(
                model_id="spatial_poisson",
                status="completed",
                comparison_count=15,
                not_run_reason=None,
            ),
            HorizonScientificSummary(
                model_id="etas",
                status="completed",
                comparison_count=15,
                not_run_reason=None,
            ),
        ),
        future=FutureScientificSummary(
            status="completed",
            issue_count=51,
            not_run_reason=None,
        ),
        representative=RepresentativeScientificSummary(
            issue_date_local="2025-06-26",
            grid_cell_size_km=25.0,
            selected_model_id="spatial_poisson",
        ),
    )


def _scientific_failure_projection() -> tuple[
    tuple[ModelAttemptRecord, ...],
    BackgroundScientificSummary,
    G1Conclusion,
    SelectionConclusion,
]:
    reason = "no eligible temporal completeness stratum"
    failure_document = {
        "failure_reason_code": "no_eligible_temporal_stratum",
        "failure_reasons": (reason,),
        "failure_stage": "completeness",
    }
    evidence_id = hashlib.sha256(canonical_json_bytes(failure_document)).hexdigest()
    attempts = tuple(
        ModelAttemptRecord(
            model_id=model_id,
            snapshot_id=snapshot_id,
            status="not_run",
            failure_reasons=(reason,),
            variant=f"{model_id}/not_run_upstream",
            parameter_identity={
                "failure_stage": "completeness",
                "model_id": model_id,
                "snapshot_id": snapshot_id,
            },
            gates=(
                GateOutcome(
                    gate_id="upstream_completeness",
                    status="not_applicable",
                    evidence_id=evidence_id,
                ),
            ),
            score_ids=(),
        )
        for model_id in MODEL_ORDER
        for snapshot_id in SNAPSHOT_ORDER
    )
    summary = BackgroundScientificSummary(
        outcome_status="scientific_gate_failed",
        failure=ScientificFailureSummary(
            failure_stage="completeness",
            failure_reason_code="no_eligible_temporal_stratum",
            failure_reasons=(reason,),
            evidence_id=evidence_id,
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
                not_run_reason=reason,
            )
            for model_id in cast(
                tuple[Literal["spatial_poisson", "etas"], ...],
                ("spatial_poisson", "etas"),
            )
        ),
        horizons=tuple(
            HorizonScientificSummary(
                model_id=model_id,
                status="skipped",
                comparison_count=0,
                not_run_reason=reason,
            )
            for model_id in cast(
                tuple[Literal["spatial_poisson", "etas"], ...],
                ("spatial_poisson", "etas"),
            )
        ),
        future=FutureScientificSummary(
            status="skipped",
            issue_count=0,
            not_run_reason=reason,
        ),
        representative=None,
    )
    return (
        attempts,
        summary,
        G1Conclusion(
            status="not_evaluable",
            passed=False,
            passing_models=(),
            evidence_ids=(evidence_id,),
        ),
        SelectionConclusion(
            status="not_evaluable",
            selected_model_id=None,
            validation_best_model_id=None,
            eligible_model_ids=(),
            evidence_id=evidence_id,
        ),
    )


def _registry(
    background: BackgroundConfig,
    identity: RepositoryIdentity,
    publications: tuple[BundlePublication, ...],
    *,
    attempts: tuple[ModelAttemptRecord, ...] | None = None,
    scientific_summary: BackgroundScientificSummary | None = None,
    g1: G1Conclusion | None = None,
    selection: SelectionConclusion | None = None,
    stage3_allowed: bool = True,
) -> BackgroundRegistry:
    return build_background_registry(
        background,
        identity,
        publications,
        attempts or _attempts(),
        scientific_summary=scientific_summary or _scientific_summary(),
        g1=g1
        or G1Conclusion(
            passed=True,
            passing_models=("spatial_poisson",),
            evidence_ids=("g1/spatial_poisson",),
        ),
        selection=selection
        or SelectionConclusion(
            selected_model_id="spatial_poisson",
            validation_best_model_id="spatial_poisson",
            eligible_model_ids=("uniform_poisson", "spatial_poisson"),
            evidence_id="selection/one_standard_error",
        ),
        stage3_allowed=stage3_allowed,
        uv_lock_sha256=background.inputs.environment_lock_sha256,
    )


def _execution_seal(identity: RepositoryIdentity, registry: BackgroundRegistry) -> ExecutionSeal:
    return ExecutionSeal(
        repository=identity,
        protocol_sha256=registry.protocol_fingerprint_sha256,
        input_hashes=tuple(
            (key, registry.input_hashes[key]) for key in sorted(registry.input_hashes)
        ),
    )


def _fixed_output_paths(project_root: Path, background: BackgroundConfig) -> tuple[Path, Path]:
    return (
        project_root.joinpath(*background.outputs.registry.split("/")),
        project_root.joinpath(*background.outputs.report.split("/")),
    )


def _synthetic_git_runner(*, commit: str = HEAD) -> GitCommandRunner:
    def run(command: tuple[str, ...], cwd: Path) -> CommandResult:
        del cwd
        if command == ("git", "rev-parse", "--is-inside-work-tree"):
            return CommandResult(0, "true\n")
        if command == ("git", "rev-parse", "--verify", "HEAD^{commit}"):
            return CommandResult(0, f"{commit}\n")
        if command == ("git", "status", "--porcelain=v1", "--untracked-files=all"):
            return CommandResult(0, "")
        if command == (
            "git",
            "rev-parse",
            "--verify",
            f"refs/tags/{FREEZE_TAG}^{{commit}}",
        ):
            return CommandResult(0, f"{TAG_COMMIT}\n")
        if command[:3] == ("git", "merge-base", "--is-ancestor"):
            return CommandResult(0)
        if command == ("git", "symbolic-ref", "--quiet", "--short", "HEAD"):
            return CommandResult(0, "main\n")
        if command == (
            "git",
            "rev-parse",
            "--abbrev-ref",
            "--symbolic-full-name",
            "@{upstream}",
        ):
            return CommandResult(0, "origin/main\n")
        if command == ("git", "rev-parse", "--verify", "@{upstream}^{commit}"):
            return CommandResult(0, f"{commit}\n")
        raise AssertionError(f"unexpected synthetic Git command: {command}")

    return run


def _materialize_synthetic_seal_inputs(
    project_root: Path,
    background: BackgroundConfig,
    monkeypatch: pytest.MonkeyPatch,
    *,
    changed_key: str | None = None,
) -> None:
    references = {
        "environment_lock": (
            background.inputs.environment_lock,
            background.inputs.environment_lock_sha256,
        ),
        "data_catalog": (
            background.inputs.data_catalog,
            background.inputs.data_catalog_sha256,
        ),
        "earthquake_dataset": (
            background.inputs.earthquake_dataset_path,
            background.inputs.earthquake_dataset_sha256,
        ),
        "study_area": (
            background.inputs.study_area,
            background.inputs.study_area_sha256,
        ),
        "issue_manifest": (
            background.inputs.issue_manifest,
            background.inputs.issue_manifest_sha256,
        ),
        "production_fixture": (
            background.numerical_regression.production_fixture,
            background.numerical_regression.production_fixture_sha256,
        ),
        "oracle_metadata": (
            background.numerical_regression.oracle_metadata,
            background.numerical_regression.oracle_metadata_sha256,
        ),
    }
    observed: dict[Path, str] = {}
    for key, (reference, expected_sha256) in references.items():
        path = project_root.joinpath(*reference.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"synthetic-{key}".encode())
        observed[path.resolve()] = "f" * 64 if key == changed_key else expected_sha256

    def observed_sha256(path: Path) -> str:
        return observed[path.resolve()]

    monkeypatch.setattr(execution_module, "sha256_file", observed_sha256)


def _allow_unchanged_seal(
    monkeypatch: pytest.MonkeyPatch,
    execution_seal: ExecutionSeal,
) -> None:
    def unchanged(*_: object, **__: object) -> ExecutionSeal:
        return execution_seal

    monkeypatch.setattr(publication_module, "require_execution_seal_unchanged", unchanged)


@pytest.mark.parametrize(
    "document",
    (
        {"generated_at_utc": "2026-07-13T00:00:00Z"},
        {"run_id": "synthetic-run"},
        {"pid": 147},
        {"uuid": "00000000-0000-0000-0000-000000000000"},
    ),
)
def test_bundle_documents_reject_volatile_runtime_metadata(
    document: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="volatile scientific metadata"):
        BundleDocument("invalid.json", document)


def test_bundle_documents_reject_nonfinite_values_and_unsafe_paths() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        BundleDocument("invalid.json", {"value": float("nan")})
    with pytest.raises(ValueError, match="POSIX-relative"):
        BundleDocument("../escape.json", {"value": 1})
    with pytest.raises(ValueError, match=".json extension"):
        BundleDocument("payload.bin", {"value": 1})


def test_bundle_documents_and_bundles_are_immutable_snapshots() -> None:
    document = BundleDocument("payload.json", {"value": 1})
    bundle = ProcessedBundle({"grid_km": 25.0}, (document,))

    with pytest.raises(AttributeError, match="immutable"):
        document.relative_path = ProjectRelativePath("other.json")
    with pytest.raises(AttributeError, match="immutable"):
        bundle._documents = ()


def test_binary_bundle_file_publishes_exact_png_bytes(
    tmp_path: Path,
    background: BackgroundConfig,
    identity: RepositoryIdentity,
) -> None:
    png_payload = b"\x89PNG\r\n\x1a\nsynthetic-deterministic-payload"
    binary = BundleBinary("maps/conditional_intensity.png", png_payload, media_type="image/png")
    bundle = ExperimentBundle(
        {"representative_issue": "2025-06-26", "render": "deterministic"},
        (
            BundleDocument("maps/conditional_intensity.json", {"label": "条件强度"}),
            binary,
        ),
    )

    published = publish_experiment_bundle(
        tmp_path,
        background,
        identity,
        bundle,
        uv_lock_sha256=background.inputs.environment_lock_sha256,
    )

    output = published.artifact.directory / "maps" / "conditional_intensity.png"
    assert output.read_bytes() == png_payload
    manifest = json.loads(published.artifact.manifest_path.read_text(encoding="utf-8"))
    png_entry = next(item for item in manifest["files"] if item["relative_path"].endswith(".png"))
    assert png_entry["media_type"] == "image/png"
    with pytest.raises(AttributeError, match="immutable"):
        binary.media_type = "application/octet-stream"


def test_binary_bundle_file_rejects_json_and_invalid_png() -> None:
    with pytest.raises(ValueError, match="BundleDocument"):
        BundleBinary("payload.json", b"{}", media_type="application/json")
    with pytest.raises(ValueError, match="PNG signature"):
        BundleBinary("map.png", b"not-png", media_type="image/png")


def test_four_bundle_types_publish_to_frozen_roots_and_reuse_identical_bytes(
    tmp_path: Path,
    background: BackgroundConfig,
    identity: RepositoryIdentity,
) -> None:
    first = _publish_all(tmp_path, background, identity)

    assert tuple(item.bundle_kind for item in first) == (
        "processed",
        "model",
        "backtest",
        "experiment",
    )
    assert tuple(item.root.value for item in first) == (
        background.outputs.processed_root,
        background.outputs.model_root,
        background.outputs.backtest_root,
        background.outputs.experiment_root,
    )
    assert len({item.artifact.artifact_id for item in first}) == 4
    assert all(item.artifact.created for item in first)

    fixed_ns = 1_700_000_000_000_000_000
    for item in first:
        os.utime(item.artifact.manifest_path, ns=(fixed_ns, fixed_ns))
        manifest = json.loads(item.artifact.manifest_path.read_text(encoding="utf-8"))
        parameters = manifest["content_address"]["inputs"]["model_parameters"]
        assert parameters["artifact_role"] == item.bundle_kind
        assert manifest["content_address"]["inputs"]["code_commit"] == HEAD

    second = _publish_all(tmp_path, background, identity)

    assert all(not item.artifact.created for item in second)
    assert tuple(item.artifact.artifact_id for item in second) == tuple(
        item.artifact.artifact_id for item in first
    )
    assert all(item.artifact.manifest_path.stat().st_mtime_ns == fixed_ns for item in second)


def test_registry_requires_every_model_snapshot_exactly_once(
    tmp_path: Path,
    background: BackgroundConfig,
    identity: RepositoryIdentity,
) -> None:
    publications = _publish_all(tmp_path, background, identity)

    with pytest.raises(ValueError, match="every model and snapshot exactly once"):
        _registry(background, identity, publications, attempts=_attempts()[:-1])

    duplicated = (*_attempts()[:-1], _attempts()[-2])
    with pytest.raises(ValueError, match="every model and snapshot exactly once"):
        _registry(background, identity, publications, attempts=duplicated)


def test_scientific_summary_rejects_wrong_snapshot_order_and_nonfinite_values() -> None:
    summary = _scientific_summary()
    payload = summary.model_dump(mode="python")

    with pytest.raises(ValueError, match="fixed 3x5 snapshots"):
        BackgroundScientificSummary.model_validate(
            {**payload, "snapshots": tuple(reversed(summary.snapshots))}
        )

    with pytest.raises(ValueError, match="final selected Mc must be finite"):
        BackgroundScientificSummary.model_validate({**payload, "final_selected_mc": float("nan")})

    with pytest.raises(ValueError, match="information gain must be finite"):
        ModelSnapshotScientificSummary(
            model_id="spatial_poisson",
            snapshot_id="fold_1",
            status="succeeded",
            target_event_count=1,
            information_gain_nats_per_event=float("inf"),
            score_id="score/spatial_poisson/fold_1",
        )


@pytest.mark.parametrize("mismatch", ("status", "score_id"))
def test_registry_rejects_scientific_summary_attempt_mismatch(
    tmp_path: Path,
    background: BackgroundConfig,
    identity: RepositoryIdentity,
    mismatch: str,
) -> None:
    publications = _publish_all(tmp_path, background, identity)
    summary = _scientific_summary()
    first = summary.snapshots[0]
    replacement = ModelSnapshotScientificSummary(
        model_id=first.model_id,
        snapshot_id=first.snapshot_id,
        status="failed" if mismatch == "status" else first.status,
        target_event_count=first.target_event_count,
        information_gain_nats_per_event=(
            None if mismatch == "status" else first.information_gain_nats_per_event
        ),
        score_id=(None if mismatch == "status" else "score/uniform_poisson/fold_1/tampered"),
    )
    tampered = BackgroundScientificSummary.model_validate(
        {
            **summary.model_dump(mode="python"),
            "snapshots": (replacement, *summary.snapshots[1:]),
        }
    )

    expected = "status differs" if mismatch == "status" else "score ID differs"
    with pytest.raises(ValueError, match=expected):
        _registry(
            background,
            identity,
            publications,
            scientific_summary=tampered,
        )


def test_registry_rejects_g1_conclusion_tampered_against_scientific_summary(
    tmp_path: Path,
    background: BackgroundConfig,
    identity: RepositoryIdentity,
) -> None:
    publications = _publish_all(tmp_path, background, identity)

    with pytest.raises(ValueError, match="G1 conclusion differs"):
        _registry(
            background,
            identity,
            publications,
            g1=G1Conclusion(
                passed=False,
                passing_models=(),
                evidence_ids=("g1/tampered",),
            ),
            stage3_allowed=False,
        )


def test_registry_rejects_bootstrap_point_tampered_against_validation_score(
    tmp_path: Path,
    background: BackgroundConfig,
    identity: RepositoryIdentity,
) -> None:
    publications = _publish_all(tmp_path, background, identity)
    summary = _scientific_summary()
    spatial = summary.validation_bootstrap[0]
    tampered_bootstrap = ValidationBootstrapScientificSummary(
        model_id=spatial.model_id,
        status=spatial.status,
        point_estimate=0.21,
        lower=spatial.lower,
        upper=spatial.upper,
        replications=spatial.replications,
        confidence_level=spatial.confidence_level,
        not_run_reason=spatial.not_run_reason,
    )
    tampered = BackgroundScientificSummary.model_validate(
        {
            **summary.model_dump(mode="python"),
            "validation_bootstrap": (
                tampered_bootstrap,
                summary.validation_bootstrap[1],
            ),
        }
    )

    with pytest.raises(ValueError, match="point estimate differs"):
        _registry(
            background,
            identity,
            publications,
            scientific_summary=tampered,
        )


def test_registry_rejects_inconsistent_attempt_and_stage3_states() -> None:
    with pytest.raises(ValueError, match="successful attempts need scores"):
        ModelAttemptRecord(
            model_id="uniform_poisson",
            snapshot_id="fold_1",
            status="succeeded",
            failure_reasons=(),
            variant="uniform/frozen-v1",
            parameter_identity={"model_id": "uniform_poisson"},
            gates=(GateOutcome(gate_id="grid", status="passed", evidence_id="gate/grid"),),
            score_ids=(),
        )

    with pytest.raises(ValueError, match="exactly equal the G1"):
        BackgroundRegistry(
            protocol_version="0.2.0",
            protocol_fingerprint_sha256="a" * 64,
            freeze_tag=FREEZE_TAG,
            code_commit=HEAD,
            input_hashes={key: "b" * 64 for key in publication_module._REQUIRED_INPUT_HASH_KEYS},
            bundles=tuple(
                publication_module.BundleReference(
                    bundle_kind=kind,
                    artifact_id=f"{index:016x}",
                    manifest_sha256=f"{index + 10:064x}",
                )
                for index, kind in enumerate(BUNDLE_ORDER, start=1)
            ),
            model_attempts=_attempts(),
            scientific_summary=_scientific_summary(),
            g1=G1Conclusion(
                passed=True,
                passing_models=("spatial_poisson",),
                evidence_ids=("g1/spatial_poisson",),
            ),
            selection=SelectionConclusion(
                selected_model_id="spatial_poisson",
                validation_best_model_id="spatial_poisson",
                eligible_model_ids=("uniform_poisson", "spatial_poisson"),
                evidence_id="selection/one_standard_error",
            ),
            stage3_allowed=False,
        )

    with pytest.raises(ValueError, match="five successful snapshot attempts"):
        BackgroundRegistry(
            protocol_version="0.2.0",
            protocol_fingerprint_sha256="a" * 64,
            freeze_tag=FREEZE_TAG,
            code_commit=HEAD,
            input_hashes={key: "b" * 64 for key in publication_module._REQUIRED_INPUT_HASH_KEYS},
            bundles=tuple(
                publication_module.BundleReference(
                    bundle_kind=kind,
                    artifact_id=f"{index:016x}",
                    manifest_sha256=f"{index + 10:064x}",
                )
                for index, kind in enumerate(BUNDLE_ORDER, start=1)
            ),
            model_attempts=_attempts(),
            scientific_summary=_scientific_summary(),
            g1=G1Conclusion(
                passed=False,
                passing_models=(),
                evidence_ids=("g1/failed",),
            ),
            selection=SelectionConclusion(
                selected_model_id="spatial_poisson",
                validation_best_model_id="spatial_poisson",
                eligible_model_ids=("uniform_poisson", "spatial_poisson", "etas"),
                evidence_id="selection/invalid-failed-model",
            ),
            stage3_allowed=False,
        )


def test_synthetic_publication_lifecycle_is_byte_stable_and_report_is_registry_only(
    tmp_path: Path,
    background: BackgroundConfig,
    identity: RepositoryIdentity,
) -> None:
    first_bundles = _publish_all(tmp_path, background, identity)
    registry = _registry(background, identity, first_bundles)
    first_fixed = publish_registry_and_report(tmp_path, background, registry)

    registry_payload = first_fixed.registry.path.read_bytes()
    report_payload = first_fixed.report.path.read_bytes()
    validated = validate_registry_payload(registry_payload)
    assert validated == registry
    assert len(validated.model_attempts) == 15
    assert tuple((item.model_id, item.snapshot_id) for item in validated.model_attempts) == tuple(
        (model_id, snapshot_id) for model_id in MODEL_ORDER for snapshot_id in SNAPSHOT_ORDER
    )
    assert render_report_from_registry_payload(registry_payload) == report_payload
    report_text = report_payload.decode("utf-8")
    assert "条件强度" in report_text
    assert "相对强度" in report_text
    assert "信息增益" in report_text
    assert "Mc" in report_text
    assert "2.6" in report_text
    assert "KDE" in report_text
    assert "80" in report_text
    assert all(
        value in report_text for value in ("0.11", "0.12", "0.13", "0.14", "0.31", "0.33", "0.34")
    )
    assert "bootstrap" in report_text.casefold()
    assert all(value in report_text for value in ("0.2", "0.17", "0.29", "0.24", "0.3", "0.41"))
    assert "2000" in report_text
    assert "0.95" in report_text
    assert "15" in report_text
    assert "延迟与时窗覆盖" in report_text
    assert "51" in report_text
    assert "未来集合覆盖" in report_text
    assert "2025-06-26" in report_text
    assert "25" in report_text
    assert "代表日条件强度" in report_text
    assert "probability" not in report_text.casefold()
    assert "概率" not in report_text
    registry_text = registry_payload.decode("utf-8")
    assert all(
        forbidden not in registry_text
        for forbidden in ("generated_at", "run_id", '"pid"', '"uuid"')
    )

    fixed_ns = 1_700_000_000_000_000_000
    for item in first_bundles:
        os.utime(item.artifact.manifest_path, ns=(fixed_ns, fixed_ns))
    os.utime(first_fixed.registry.path, ns=(fixed_ns, fixed_ns))
    os.utime(first_fixed.report.path, ns=(fixed_ns, fixed_ns))

    second_bundles = _publish_all(tmp_path, background, identity)
    second_registry = _registry(background, identity, second_bundles)
    second_fixed = publish_registry_and_report(tmp_path, background, second_registry)

    assert second_registry == registry
    assert all(not item.artifact.created for item in second_bundles)
    assert second_fixed.registry.created is False
    assert second_fixed.report.created is False
    assert second_fixed.registry.path.stat().st_mtime_ns == fixed_ns
    assert second_fixed.report.path.stat().st_mtime_ns == fixed_ns
    assert all(
        item.artifact.manifest_path.stat().st_mtime_ns == fixed_ns for item in second_bundles
    )


def test_sealed_publication_reseals_verifies_bundles_and_publishes_fixed_outputs(
    tmp_path: Path,
    background: BackgroundConfig,
    identity: RepositoryIdentity,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publications = _publish_all(tmp_path, background, identity)
    registry = _registry(background, identity, publications)
    execution_seal = _execution_seal(identity, registry)
    _materialize_synthetic_seal_inputs(tmp_path, background, monkeypatch)
    published = publish_registry_and_report_sealed(
        tmp_path,
        background,
        registry,
        execution_seal,
        runner=_synthetic_git_runner(),
    )

    assert published.registry.created is True
    assert published.report.created is True
    assert validate_registry_payload(published.registry.path.read_bytes()) == registry
    assert render_report_from_registry_payload(published.registry.path.read_bytes()) == (
        published.report.path.read_bytes()
    )


def test_sealed_publication_archives_scientific_failure_and_renders_stop_report(
    tmp_path: Path,
    background: BackgroundConfig,
    identity: RepositoryIdentity,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts, summary, g1, selection = _scientific_failure_projection()
    summary_sha256 = hashlib.sha256(
        canonical_json_bytes(summary.model_dump(mode="python"))
    ).hexdigest()
    failure = cast(ScientificFailureSummary, summary.failure)
    failure_parameters = {
        "failure_evidence_sha256": failure.evidence_id,
        "failure_reason_code": failure.failure_reason_code,
        "failure_stage": failure.failure_stage,
    }
    failure_document = {
        "failure_reason_code": failure.failure_reason_code,
        "failure_reasons": failure.failure_reasons,
        "failure_stage": failure.failure_stage,
    }
    processed_bundle = ProcessedBundle(
        failure_parameters,
        (
            BundleDocument(
                "failure.json",
                failure_document,
            ),
        ),
    )
    model_bundle = ModelBundle(
        failure_parameters,
        (BundleDocument("failure.json", failure_document),),
    )
    experiment_bundle = ExperimentBundle(
        failure_parameters,
        (BundleDocument("failure.json", failure_document),),
    )
    backtest_bundle = BacktestBundle(
        {**failure_parameters, "scientific_summary_sha256": summary_sha256},
        (
            BundleDocument("failure.json", failure_document),
            BundleDocument(
                "scientific_summary.json",
                summary.model_dump(mode="python"),
            ),
        ),
    )
    lock_hash = background.inputs.environment_lock_sha256
    publications = (
        publish_processed_bundle(
            tmp_path,
            background,
            identity,
            processed_bundle,
            uv_lock_sha256=lock_hash,
        ),
        publish_model_bundle(
            tmp_path,
            background,
            identity,
            model_bundle,
            uv_lock_sha256=lock_hash,
        ),
        publish_backtest_bundle(
            tmp_path,
            background,
            identity,
            backtest_bundle,
            uv_lock_sha256=lock_hash,
        ),
        publish_experiment_bundle(
            tmp_path,
            background,
            identity,
            experiment_bundle,
            uv_lock_sha256=lock_hash,
        ),
    )
    registry = _registry(
        background,
        identity,
        publications,
        attempts=attempts,
        scientific_summary=summary,
        g1=g1,
        selection=selection,
        stage3_allowed=False,
    )
    execution_seal = _execution_seal(identity, registry)
    _allow_unchanged_seal(monkeypatch, execution_seal)

    published = publish_registry_and_report_sealed(
        tmp_path,
        background,
        registry,
        execution_seal,
    )

    validated = validate_registry_payload(published.registry.path.read_bytes())
    report = published.report.path.read_text(encoding="utf-8")
    assert validated.scientific_summary.outcome_status == "scientific_gate_failed"
    assert all(attempt.status == "not_run" for attempt in validated.model_attempts)
    assert validated.g1.status == "not_evaluable"
    assert validated.selection.status == "not_evaluable"
    assert not validated.stage3_allowed
    assert "科学门控失败" in report
    assert "未运行" in report
    assert "| uniform_poisson | fold_1 | 未运行 |" in report
    assert "未评估 (上游科学门控失败)" in report
    assert "阶段3: 停止" in report


def test_sealed_publication_binds_registry_summary_to_backtest_address(
    tmp_path: Path,
    background: BackgroundConfig,
    identity: RepositoryIdentity,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publications = _publish_all(tmp_path, background, identity)
    summary = _scientific_summary()
    spatial_fold_1 = summary.snapshots[len(SNAPSHOT_ORDER)]
    changed = ModelSnapshotScientificSummary(
        **{
            **spatial_fold_1.model_dump(mode="python"),
            "information_gain_nats_per_event": 0.19,
        }
    )
    tampered_summary = BackgroundScientificSummary.model_validate(
        {
            **summary.model_dump(mode="python"),
            "snapshots": (
                *summary.snapshots[: len(SNAPSHOT_ORDER)],
                changed,
                *summary.snapshots[len(SNAPSHOT_ORDER) + 1 :],
            ),
        }
    )
    registry = _registry(
        background,
        identity,
        publications,
        scientific_summary=tampered_summary,
    )
    execution_seal = _execution_seal(identity, registry)
    _allow_unchanged_seal(monkeypatch, execution_seal)

    with pytest.raises(FixedFilePublicationError, match="summary differs.*address"):
        publish_registry_and_report_sealed(
            tmp_path,
            background,
            registry,
            execution_seal,
        )

    assert all(not path.exists() for path in _fixed_output_paths(tmp_path, background))


def test_sealed_publication_binds_registry_summary_to_backtest_payload(
    tmp_path: Path,
    background: BackgroundConfig,
    identity: RepositoryIdentity,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processed_bundle, model_bundle, _, experiment_bundle = _bundles()
    lock_hash = background.inputs.environment_lock_sha256
    summary = _scientific_summary()
    summary_sha256 = hashlib.sha256(
        canonical_json_bytes(summary.model_dump(mode="python"))
    ).hexdigest()
    publications = (
        publish_processed_bundle(
            tmp_path,
            background,
            identity,
            processed_bundle,
            uv_lock_sha256=lock_hash,
        ),
        publish_model_bundle(
            tmp_path,
            background,
            identity,
            model_bundle,
            uv_lock_sha256=lock_hash,
        ),
        publish_backtest_bundle(
            tmp_path,
            background,
            identity,
            BacktestBundle(
                {"scientific_summary_sha256": summary_sha256},
                (BundleDocument("scientific_summary.json", {"tampered": True}),),
            ),
            uv_lock_sha256=lock_hash,
        ),
        publish_experiment_bundle(
            tmp_path,
            background,
            identity,
            experiment_bundle,
            uv_lock_sha256=lock_hash,
        ),
    )
    registry = _registry(background, identity, publications)
    execution_seal = _execution_seal(identity, registry)
    _allow_unchanged_seal(monkeypatch, execution_seal)

    with pytest.raises(FixedFilePublicationError, match="summary differs.*payload"):
        publish_registry_and_report_sealed(
            tmp_path,
            background,
            registry,
            execution_seal,
        )

    assert all(not path.exists() for path in _fixed_output_paths(tmp_path, background))


@pytest.mark.parametrize(
    ("changed_surface", "match"),
    (
        ("identity", "repository identity changed"),
        ("input", "earthquake_dataset"),
    ),
)
def test_sealed_publication_rejects_identity_or_input_reseal_failure_before_fixed_writes(
    tmp_path: Path,
    background: BackgroundConfig,
    identity: RepositoryIdentity,
    monkeypatch: pytest.MonkeyPatch,
    changed_surface: str,
    match: str,
) -> None:
    publications = _publish_all(tmp_path, background, identity)
    registry = _registry(background, identity, publications)
    execution_seal = _execution_seal(identity, registry)
    _materialize_synthetic_seal_inputs(
        tmp_path,
        background,
        monkeypatch,
        changed_key="earthquake_dataset" if changed_surface == "input" else None,
    )
    with pytest.raises(ExecutionSealError, match=match):
        publish_registry_and_report_sealed(
            tmp_path,
            background,
            registry,
            execution_seal,
            runner=_synthetic_git_runner(
                commit="b" * 40 if changed_surface == "identity" else HEAD
            ),
        )

    assert all(not path.exists() for path in _fixed_output_paths(tmp_path, background))


def test_sealed_publication_reseals_again_after_bundle_verification(
    tmp_path: Path,
    background: BackgroundConfig,
    identity: RepositoryIdentity,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publications = _publish_all(tmp_path, background, identity)
    registry = _registry(background, identity, publications)
    execution_seal = _execution_seal(identity, registry)
    calls = 0

    def recheck(*_: object, **__: object) -> ExecutionSeal:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ExecutionSealError("synthetic drift during bundle verification")
        return execution_seal

    monkeypatch.setattr(
        publication_module,
        "require_execution_seal_unchanged",
        recheck,
    )

    with pytest.raises(ExecutionSealError, match="during bundle verification"):
        publish_registry_and_report_sealed(
            tmp_path,
            background,
            registry,
            execution_seal,
        )

    assert calls == 2
    assert all(not path.exists() for path in _fixed_output_paths(tmp_path, background))


def test_sealed_publication_rejects_tampered_bundle_payload_before_fixed_writes(
    tmp_path: Path,
    background: BackgroundConfig,
    identity: RepositoryIdentity,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publications = _publish_all(tmp_path, background, identity)
    registry = _registry(background, identity, publications)
    execution_seal = _execution_seal(identity, registry)
    _allow_unchanged_seal(monkeypatch, execution_seal)
    target = publications[0].artifact.directory / "grids.json"
    target.write_bytes(b"tampered-bundle-payload")

    with pytest.raises(FixedFilePublicationError, match="payload"):
        publish_registry_and_report_sealed(
            tmp_path,
            background,
            registry,
            execution_seal,
        )

    assert all(not path.exists() for path in _fixed_output_paths(tmp_path, background))


def test_sealed_publication_rejects_extra_bundle_file_before_fixed_writes(
    tmp_path: Path,
    background: BackgroundConfig,
    identity: RepositoryIdentity,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publications = _publish_all(tmp_path, background, identity)
    registry = _registry(background, identity, publications)
    execution_seal = _execution_seal(identity, registry)
    _allow_unchanged_seal(monkeypatch, execution_seal)
    (publications[1].artifact.directory / "unexpected.bin").write_bytes(b"unexpected")

    with pytest.raises(FixedFilePublicationError, match="file set differs"):
        publish_registry_and_report_sealed(
            tmp_path,
            background,
            registry,
            execution_seal,
        )

    assert all(not path.exists() for path in _fixed_output_paths(tmp_path, background))


def test_sealed_publication_rejects_nested_reparse_point_before_fixed_writes(
    tmp_path: Path,
    background: BackgroundConfig,
    identity: RepositoryIdentity,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    publications = _publish_all(tmp_path, background, identity)
    registry = _registry(background, identity, publications)
    execution_seal = _execution_seal(identity, registry)
    _allow_unchanged_seal(monkeypatch, execution_seal)
    simulated_reparse = publications[2].artifact.directory / "simulated-reparse"
    simulated_reparse.mkdir()
    original = publication_module._is_reparse_point

    def is_reparse(path: Path) -> bool:
        return path == simulated_reparse or original(path)

    monkeypatch.setattr(publication_module, "_is_reparse_point", is_reparse)
    with pytest.raises(FixedFilePublicationError, match="symlink or reparse point"):
        publish_registry_and_report_sealed(
            tmp_path,
            background,
            registry,
            execution_seal,
        )

    assert all(not path.exists() for path in _fixed_output_paths(tmp_path, background))


def test_changed_bundle_bytes_conflict_under_the_same_address(
    tmp_path: Path,
    background: BackgroundConfig,
    identity: RepositoryIdentity,
) -> None:
    parameters = {"grid_cells_km": [50.0, 25.0, 12.5]}
    first = ProcessedBundle(
        parameters,
        (BundleDocument("grids.json", {"area_km2": [1.0]}),),
    )
    changed = ProcessedBundle(
        parameters,
        (BundleDocument("grids.json", {"area_km2": [2.0]}),),
    )
    publish_processed_bundle(
        tmp_path,
        background,
        identity,
        first,
        uv_lock_sha256=background.inputs.environment_lock_sha256,
    )

    with pytest.raises(ArtifactConflictError, match="manifest bytes differ"):
        publish_processed_bundle(
            tmp_path,
            background,
            identity,
            changed,
            uv_lock_sha256=background.inputs.environment_lock_sha256,
        )


def test_fixed_file_different_existing_bytes_are_never_overwritten(tmp_path: Path) -> None:
    first = publish_fixed_project_file(tmp_path, "data/manifests/fixed.json", b"first")
    before = first.path.read_bytes()

    with pytest.raises(FixedFileConflictError, match="different bytes"):
        publish_fixed_project_file(tmp_path, "data/manifests/fixed.json", b"second")

    assert first.path.read_bytes() == before


def test_fixed_file_rejects_escape_and_reparse_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="POSIX-relative"):
        publish_fixed_project_file(tmp_path, "../escape.json", b"payload")

    reparse_parent = tmp_path / "data"
    reparse_parent.mkdir()
    original = publication_module._is_reparse_point

    def simulated_reparse(path: Path) -> bool:
        return path == reparse_parent or original(path)

    monkeypatch.setattr(publication_module, "_is_reparse_point", simulated_reparse)
    with pytest.raises(ValueError, match="symlink or reparse point"):
        publish_fixed_project_file(tmp_path, "data/manifests/fixed.json", b"payload")


def test_fixed_file_atomic_failure_leaves_no_destination_or_temporary_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "data" / "manifests" / "fixed.json"

    def fail_link(_: Path, __: Path) -> None:
        raise OSError("synthetic atomic-link failure")

    monkeypatch.setattr(os, "link", fail_link)
    with pytest.raises(FixedFilePublicationError, match="atomically create"):
        publish_fixed_project_file(tmp_path, "data/manifests/fixed.json", b"payload")

    assert not destination.exists()
    assert not list(destination.parent.glob(".fixed.json.*.tmp"))


def test_fixed_file_success_is_not_reclassified_by_temporary_cleanup_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_unlink = Path.unlink

    def fail_temporary_unlink(path: Path, missing_ok: bool = False) -> None:
        if path.suffix == ".tmp":
            raise PermissionError("synthetic temporary cleanup failure")
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fail_temporary_unlink)
    publication = publish_fixed_project_file(
        tmp_path,
        "data/manifests/fixed.json",
        b"payload",
    )

    assert publication.created
    assert publication.path.read_bytes() == b"payload"
    for temporary in publication.path.parent.glob(".fixed.json.*.tmp"):
        original_unlink(temporary)


def test_registry_is_rolled_back_when_report_fixed_publication_conflicts(
    tmp_path: Path,
    background: BackgroundConfig,
    identity: RepositoryIdentity,
) -> None:
    publications = _publish_all(tmp_path, background, identity)
    registry = _registry(background, identity, publications)
    report_path = tmp_path / background.outputs.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_bytes(b"preexisting-conflict")

    with pytest.raises(FixedFileConflictError, match="different bytes"):
        publish_registry_and_report(tmp_path, background, registry)

    assert not (tmp_path / background.outputs.registry).exists()
    assert report_path.read_bytes() == b"preexisting-conflict"


def test_report_rejects_tampered_or_noncanonical_registry_payload(
    tmp_path: Path,
    background: BackgroundConfig,
    identity: RepositoryIdentity,
) -> None:
    publications = _publish_all(tmp_path, background, identity)
    registry = _registry(background, identity, publications)
    payload = registry_payload_bytes(registry)
    raw = cast(dict[str, object], json.loads(payload))
    attempts = cast(list[object], raw["model_attempts"])
    attempts.pop()
    tampered = (json.dumps(raw, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()

    with pytest.raises(RegistryValidationError, match="invalid background registry"):
        render_report_from_registry_payload(tampered)
    with pytest.raises(RegistryValidationError, match="canonical form"):
        render_report_from_registry_payload(payload.rstrip())
