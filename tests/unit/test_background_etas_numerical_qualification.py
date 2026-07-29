from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from seismoflux.background import etas_numerical_qualification as qualification
from seismoflux.background.catalog import EarthquakeCatalog
from seismoflux.background.config import load_background_protocol
from seismoflux.background.etas_fit import (
    ETASEvent,
    ETASFitResult,
    ETASLikelihoodProblem,
    ETASModelSpec,
    ETASParameterBounds,
    ETASParameters,
    ETASStartResult,
    HessianAudit,
    OptimizerOptions,
    PointAreaQuadrature,
    QuadraturePoint,
    StabilityAudit,
    StabilityThresholds,
    audit_stability,
    etas_objective,
    optimizer_start,
)
from seismoflux.background.etas_numerical_qualification import (
    PreparedQualification,
    PreparedSnapshot,
    QualificationProtocol,
    atomic_create,
    build_html,
    build_markdown,
    build_svg,
    canonical_bytes,
    causal_fit_masks,
    classify_snapshot,
    file_sha256,
    load_protocol,
    run_prepared,
    verify_prepared,
)


@dataclass(frozen=True)
class _FlatDensity:
    def __call__(self, x_km: float, y_km: float) -> float:
        return 1.0

    def density_many(self, x_km: object, y_km: object) -> np.ndarray[Any, Any]:
        return np.ones(np.asarray(x_km).shape, dtype=np.float64)


def _problem() -> ETASLikelihoodProblem:
    event = ETASEvent(
        event_id="private-synthetic-event",
        time_days=1.0,
        available_time_days=1.0,
        x_km=0.0,
        y_km=0.0,
        magnitude=4.2,
        inside_study_area=True,
        inside_parent_domain=True,
    )
    return ETASLikelihoodProblem(
        assessment_start_days=0.0,
        assessment_end_days=2.0,
        target_events=(event,),
        parent_events=(event,),
        background_density=_FlatDensity(),
        spatial_integrator=PointAreaQuadrature.from_points(
            (QuadraturePoint(x_km=0.0, y_km=0.0, area_km2=1.0),)
        ),
    )


def _protocol(tmp_path: Path) -> QualificationProtocol:
    outputs = {
        "input_manifest": "public/input.json",
        "result_manifest": "public/result.json",
        "verification_manifest": "public/verification.json",
        "report": "public/report.md",
        "static_figure": "public/diagnostic.svg",
        "interactive_report": "public/interactive/index.html",
    }
    raw = {
        "publication": {"code_tag": "v-test-code"},
        "attempt": {"root": "restricted/attempt"},
        "outputs": outputs,
        "frozen_identity": {"uv_lock_sha256": "e" * 64},
    }
    return QualificationProtocol(
        path=tmp_path / "protocol.yaml",
        root=tmp_path,
        raw=raw,
        sha256="1" * 64,
    )


def _prepared(tmp_path: Path) -> PreparedQualification:
    bounds = ETASParameterBounds()
    options = OptimizerOptions()
    thresholds = StabilityThresholds()
    problem = _problem()
    spec = ETASModelSpec(mc=4.0, beta=2.0)
    snapshots: list[PreparedSnapshot] = []
    for snapshot_id in ("fold_1", "fold_2", "fold_3", "fold_4", "final_validation"):
        starts = tuple(
            tuple(
                float(value)
                for value in optimizer_start(
                    bounds.transformed(),
                    root_seed=147,
                    protocol_version="0.2.1",
                    model_id=f"etas/{snapshot_id}",
                    start_index=index,
                )
            )
            for index in range(5)
        )
        snapshots.append(
            PreparedSnapshot(
                snapshot_id=snapshot_id,
                fit_end_utc="2004-12-31T16:00:00Z",
                scientific_fit_input_sha256=(str(len(snapshots) + 2) * 64)[:64],
                membership_sha256=(str(len(snapshots) + 3) * 64)[:64],
                maximum_origin_time="2004-01-01T00:00:00Z",
                maximum_available_at="2004-01-02T00:00:00Z",
                fit_event_count=1,
                parent_event_count=1,
                kde_training_event_count=1,
                support_id=f"local-support-{len(snapshots):016x}",
                compensator_domain_id=(str(len(snapshots) + 4) * 64)[:64],
                model_id=f"etas/{snapshot_id}",
                starts=starts,
                problem=problem,
                sensitivity_problem=problem if snapshot_id in {"fold_1", "fold_3"} else None,
                spec=spec,
                bounds=bounds,
                options=options,
                thresholds=thresholds,
                grid_family=None,
            )
        )
    return PreparedQualification(
        protocol=_protocol(tmp_path),
        snapshots=tuple(snapshots),
        catalog_sha256="a" * 64,
        study_area_sha256="b" * 64,
        support_manifest_sha256="c" * 64,
        start_manifest_sha256="d" * 64,
    )


def _negative_fit(snapshot: PreparedSnapshot) -> ETASFitResult:
    objective = etas_objective(snapshot.problem, snapshot.spec, snapshot.bounds)
    rows = tuple(
        ETASStartResult(
            start_index=index,
            initial_transformed=start,
            final_transformed=start,
            objective=float(objective(np.asarray(start))),
            scipy_converged=False,
            gradient_infinity_norm=math.inf,
            iterations=1,
            function_evaluations=2,
            message="synthetic non-convergence",
        )
        for index, start in enumerate(snapshot.starts)
    )
    stability = audit_stability(
        objective,
        rows,
        snapshot.bounds.transformed(),
        thresholds=snapshot.thresholds,
    )
    return ETASFitResult(
        best_parameters=None,
        best_objective=None,
        start_results=rows,
        stability=stability,
    )


def test_protocol_loads_the_frozen_target_blind_contract() -> None:
    protocol = load_protocol("configs/background_etas_numerical_qualification.yaml")
    assert protocol.raw["stage"] == "2-ETAS-Q"
    assert protocol.sha256 == "dc602e6f3e543d124e7e3d4b363ac45bdfa1ba7d3773f13bf816035cf10b51c6"
    assert protocol.raw["target_blindness"]["assessment_event_read"] is False
    assert protocol.raw["snapshots"]["unsupported_parent_sensitivity_optimizer_call_count"] == 0


def test_target_blind_config_loader_matches_the_frozen_numeric_contract() -> None:
    protocol = load_protocol("configs/background_etas_numerical_qualification.yaml")
    background = qualification._load_target_blind_background(
        Path("configs/base_local_support.yaml").resolve(),
        Path("configs/background_local_support.yaml").resolve(),
    )
    assert background == load_background_protocol("configs/background_local_support.yaml")
    qualification._validate_runtime_numeric_contract(protocol, background)


def test_cli_imports_the_tagged_worktree_without_inherited_pythonpath(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            str(Path("scripts/run_background_etas_numerical_qualification.py").resolve()),
            "--help",
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode == 0, result.stderr
    assert "--protocol" not in result.stdout


def test_causal_masks_exclude_future_and_not_yet_available_rows() -> None:
    def values(items: Any, dtype: Any) -> np.ndarray[Any, Any]:
        return cast(np.ndarray[Any, Any], np.asarray(items, dtype=dtype))

    catalog = EarthquakeCatalog(
        event_id=values(["old", "fit", "late_report", "future", "external"], np.str_),
        origin_day=values([1.0, 5.0, 6.0, 11.0, 3.0], np.float64),
        available_day=values([1.0, 5.0, 11.0, 11.0, 3.0], np.float64),
        longitude=values([0, 0, 0, 0, 0], np.float64),
        latitude=values([0, 0, 0, 0, 0], np.float64),
        x_km=values([0, 0, 0, 0, 1], np.float64),
        y_km=values([0, 0, 0, 0, 1], np.float64),
        magnitude=values([4.2, 4.2, 4.2, 4.2, 4.2], np.float64),
        inside_study_area=values([True, True, True, True, False], np.bool_),
        inside_external_buffer=values([True, True, True, True, True], np.bool_),
    )
    targets, parents = causal_fit_masks(
        catalog,
        supported=[True, True, True, True, False],
        parents=[True, True, True, True, True],
        mc=4.0,
        fit_start_day=4.0,
        fit_end_day=10.0,
        history_start_day=0.0,
        parent_cutoff_days=100.0,
    )
    assert np.flatnonzero(targets).tolist() == [1]
    assert np.flatnonzero(parents).tolist() == [0, 1, 4]


@pytest.mark.parametrize(
    ("gates", "expected"),
    [
        ((True, True, True), ("evaluable", ())),
        (
            (False, True, False),
            (
                "not_evaluable",
                ("numerical_stability_failed", "three_grid_failed_or_unavailable"),
            ),
        ),
    ],
)
def test_classification_is_the_frozen_conjunction(
    gates: tuple[bool, bool, bool], expected: tuple[str, tuple[str, ...]]
) -> None:
    assert classify_snapshot(*gates) == expected


def test_five_by_five_run_is_atomic_and_resume_fits_only_missing_snapshot(
    tmp_path: Path,
) -> None:
    prepared = _prepared(tmp_path)
    by_model = {item.model_id: item for item in prepared.snapshots}
    calls: list[str] = []
    interrupted = True

    def fitter(*args: object, **kwargs: object) -> ETASFitResult:
        model_id = str(kwargs["model_id"])
        calls.append(model_id)
        if interrupted and model_id == "etas/fold_3":
            raise RuntimeError("synthetic interruption before snapshot persistence")
        return _negative_fit(by_model[model_id])

    with pytest.raises(RuntimeError, match="synthetic interruption"):
        run_prepared(
            prepared,
            code_commit="0" * 40,
            code_tag="v-test-code",
            fitter=fitter,
        )
    assert calls == ["etas/fold_1", "etas/fold_2", "etas/fold_3"]

    interrupted = False
    calls.clear()
    result = run_prepared(
        prepared,
        code_commit="0" * 40,
        code_tag="v-test-code",
        fitter=fitter,
    )
    assert calls == ["etas/fold_3", "etas/fold_4", "etas/final_validation"]
    assert result["qualification_status"] == "not_evaluable"
    assert result["completed_start_row_count"] == 25
    public_bytes = b"".join(
        path.read_bytes() for path in (tmp_path / "public").rglob("*") if path.is_file()
    )
    assert b"private-synthetic-event" not in public_bytes

    missing = prepared.protocol.attempt_root / "snapshots" / "fold_3.json"
    missing.unlink()
    calls.clear()
    with pytest.raises(ValueError, match="completed snapshot was deleted"):
        run_prepared(
            prepared,
            code_commit="0" * 40,
            code_tag="v-test-code",
            fitter=fitter,
        )
    assert calls == []
    assert not missing.exists()


def test_positive_path_uses_fit_problem_computes_sensitivity_without_refit_and_verifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = _prepared(tmp_path)
    snapshots = tuple(
        replace(
            snapshot,
            grid_family=cast(Any, object()),
            sensitivity_problem=(
                _problem() if snapshot.snapshot_id in {"fold_1", "fold_3"} else None
            ),
        )
        for snapshot in original.snapshots
    )
    prepared = replace(original, snapshots=snapshots)
    sensitivity_problem_ids = {
        id(snapshot.sensitivity_problem)
        for snapshot in snapshots
        if snapshot.sensitivity_problem is not None
    }
    stable = StabilityAudit(
        stable=True,
        converged_start_count=5,
        best_three_relative_objective_range=0.0,
        best_three_transformed_parameter_range=0.0,
        hessian=HessianAudit(
            success=True,
            minimum_eigenvalue=1.0,
            condition_number=1.0,
            matrix=tuple(
                tuple(1.0 if row == column else 0.0 for column in range(5)) for row in range(5)
            ),
            failure_reason=None,
        ),
        failure_reasons=(),
    )
    parameters = ETASParameters(
        background_rate_per_day=0.1,
        productivity_k=0.001,
        alpha=0.5,
        c_days=0.1,
        p=1.2,
    )
    grid_problems: list[ETASLikelihoodProblem] = []
    fit_calls: list[str] = []

    def objective(problem: ETASLikelihoodProblem, *args: object) -> Any:
        value = 2.0 if id(problem) in sensitivity_problem_ids else 1.0
        return lambda terminal: value

    def grid_gate(**kwargs: object) -> object:
        grid_problems.append(cast(ETASLikelihoodProblem, kwargs["problem"]))
        return SimpleNamespace(
            passed=True,
            resolutions=(),
            convergence=SimpleNamespace(comparisons=()),
        )

    monkeypatch.setattr(qualification, "etas_objective", objective)
    monkeypatch.setattr(
        qualification,
        "three_point_gradient",
        lambda *args, **kwargs: np.zeros(5, dtype=np.float64),
    )
    monkeypatch.setattr(qualification, "audit_stability", lambda *args, **kwargs: stable)
    monkeypatch.setattr(qualification, "_grid_gate_evidence", grid_gate)

    by_model = {snapshot.model_id: snapshot for snapshot in snapshots}

    def fitter(*args: object, **kwargs: object) -> Any:
        model_id = str(kwargs["model_id"])
        fit_calls.append(model_id)
        snapshot = by_model[model_id]
        terminal = tuple(snapshot.bounds.to_transformed(parameters))
        rows = tuple(
            ETASStartResult(
                start_index=index,
                initial_transformed=start,
                final_transformed=terminal,
                objective=1.0,
                scipy_converged=True,
                gradient_infinity_norm=0.0,
                iterations=2,
                function_evaluations=3,
                message="synthetic convergence",
            )
            for index, start in enumerate(snapshot.starts)
        )
        return SimpleNamespace(start_results=rows, stability=stable)

    manifest = run_prepared(
        prepared,
        code_commit="0" * 40,
        code_tag="v-test-code",
        fitter=fitter,
    )
    assert manifest["qualification_status"] == "evaluable"
    assert fit_calls == [snapshot.model_id for snapshot in snapshots]
    expected_grid_problems = [snapshot.problem for snapshot in snapshots]
    assert len(grid_problems) == len(expected_grid_problems)
    assert all(
        actual is expected
        for actual, expected in zip(grid_problems, expected_grid_problems, strict=True)
    )
    fold_1 = json.loads(
        (prepared.protocol.attempt_root / "snapshots" / "fold_1.json").read_text(encoding="utf-8")
    )
    assert fold_1["unsupported_parent_objective_sensitivity"] == {
        "status": "computed",
        "objective_difference": 1.0,
    }
    verification = verify_prepared(
        prepared,
        code_commit="0" * 40,
        code_tag="v-test-code",
        persist=False,
    )
    assert verification["qualification_status"] == "evaluable"
    assert fit_calls == [snapshot.model_id for snapshot in snapshots]


def test_atomic_create_ignores_stale_temp_and_never_replaces(tmp_path: Path) -> None:
    destination = tmp_path / "result.json"
    stale = tmp_path / ".result.json.interrupted.tmp"
    stale.write_bytes(b"incomplete")
    atomic_create(destination, b"complete")
    assert destination.read_bytes() == b"complete"
    atomic_create(destination, b"complete")
    with pytest.raises(ValueError, match="immutable artifact differs"):
        atomic_create(destination, b"different")
    assert stale.read_bytes() == b"incomplete"


def test_verify_recalculates_terminal_values_and_rejects_coordinated_tamper(
    tmp_path: Path,
) -> None:
    prepared = _prepared(tmp_path)
    by_model = {item.model_id: item for item in prepared.snapshots}
    run_prepared(
        prepared,
        code_commit="0" * 40,
        code_tag="v-test-code",
        fitter=lambda *args, **kwargs: _negative_fit(by_model[str(kwargs["model_id"])]),
    )
    assert (
        verify_prepared(
            prepared,
            code_commit="0" * 40,
            code_tag="v-test-code",
            persist=False,
        )["verification_status"]
        == "passed"
    )
    snapshot_path = prepared.protocol.attempt_root / "snapshots" / "fold_2.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["start_rows"][0]["objective"] += 1.0
    snapshot_path.write_bytes(canonical_bytes(snapshot))
    result_path = prepared.protocol.output("result_manifest")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["snapshot_results_sha256"]["fold_2"] = file_sha256(snapshot_path)
    result_path.write_bytes(canonical_bytes(result))
    with pytest.raises(ValueError, match="independent recalculation failed"):
        verify_prepared(
            prepared,
            code_commit="0" * 40,
            code_tag="v-test-code",
            persist=False,
        )


def test_verify_rebuilds_input_manifest_and_rejects_coordinated_identity_tamper(
    tmp_path: Path,
) -> None:
    prepared = _prepared(tmp_path)
    by_model = {item.model_id: item for item in prepared.snapshots}
    run_prepared(
        prepared,
        code_commit="0" * 40,
        code_tag="v-test-code",
        fitter=lambda *args, **kwargs: _negative_fit(by_model[str(kwargs["model_id"])]),
    )
    input_path = prepared.protocol.output("input_manifest")
    input_manifest = json.loads(input_path.read_text(encoding="utf-8"))
    input_manifest["code_commit"] = "1" * 40
    input_path.write_bytes(canonical_bytes(input_manifest))

    with pytest.raises(ValueError, match="independently rebuilt inputs"):
        verify_prepared(
            prepared,
            code_commit="0" * 40,
            code_tag="v-test-code",
            persist=False,
        )


def test_resume_rejects_a_mixed_or_tampered_numerical_environment(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    by_model = {item.model_id: item for item in prepared.snapshots}
    run_prepared(
        prepared,
        code_commit="0" * 40,
        code_tag="v-test-code",
        fitter=lambda *args, **kwargs: _negative_fit(by_model[str(kwargs["model_id"])]),
    )
    seal_path = prepared.protocol.attempt_root / "environment_seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    seal["numpy_version"] = "tampered-version"
    seal_path.write_bytes(canonical_bytes(seal))

    with pytest.raises(ValueError, match="immutable artifact differs"):
        run_prepared(
            prepared,
            code_commit="0" * 40,
            code_tag="v-test-code",
            fitter=lambda *args, **kwargs: _negative_fit(by_model[str(kwargs["model_id"])]),
        )


def test_public_views_are_offline_coordinate_free_and_explain_the_gate(
    tmp_path: Path,
) -> None:
    prepared = _prepared(tmp_path)
    by_model = {item.model_id: item for item in prepared.snapshots}
    manifest = run_prepared(
        prepared,
        code_commit="0" * 40,
        code_tag="v-test-code",
        fitter=lambda *args, **kwargs: _negative_fit(by_model[str(kwargs["model_id"])]),
    )
    results = tuple(
        json.loads(
            (prepared.protocol.attempt_root / "snapshots" / f"{item.snapshot_id}.json").read_text(
                encoding="utf-8"
            )
        )
        for item in prepared.snapshots
    )
    markdown = build_markdown(manifest, results)
    svg = build_svg(manifest, results)
    interactive = build_html(manifest, results)
    assert "不是预测命中率" in markdown
    assert "事件编号和坐标" in svg
    assert "完全离线" in interactive
    assert "private-synthetic-event" not in markdown + svg + interactive
    assert "https://" not in interactive and "src=" not in interactive
    prepared.protocol.output("report").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="public view differs"):
        verify_prepared(
            prepared,
            code_commit="0" * 40,
            code_tag="v-test-code",
            persist=False,
        )
