from __future__ import annotations

import ast
import inspect
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import seismoflux.multitask_s1.development_runtime as runtime
import seismoflux.multitask_s1.development_score as score
import seismoflux.multitask_s1.runner_inputs as runner_inputs
from seismoflux.multitask_s1.development_contract import DEVELOPMENT_FOLD_IDS
from seismoflux.multitask_s1.prediction_seal import (
    DevelopmentScoringAuthorization,
    PredictionInputIdentities,
    PredictionSealError,
    SealedArtifact,
)


def _identities() -> PredictionInputIdentities:
    return PredictionInputIdentities(
        run_contract_sha256="0" * 64,
        parent_contract_sha256="1" * 64,
        catalog_sha256="2" * 64,
        study_sha256="3" * 64,
        grid_sha256="4" * 64,
        issue_ledger_sha256="5" * 64,
        code_sha256="6" * 64,
        git_commit_oid="7" * 40,
    )


def _authorization(root: Path, digest: str = "a" * 64) -> DevelopmentScoringAuthorization:
    return DevelopmentScoringAuthorization(
        seal=SealedArtifact(root / "four_fold_prediction_seal.json", digest, 17),
        input_identities=_identities(),
        ordered_fold_sha256=tuple(
            (fold_id, str(index) * 64) for index, fold_id in enumerate(DEVELOPMENT_FOLD_IDS, 1)
        ),
    )


def _minimal_source_arrays(issue_us: int = 0) -> dict[str, np.ndarray[Any, Any]]:
    shape = (1, 3)
    masses_m0 = np.full((1, 55), 1.0 / 55.0, dtype=np.float64)
    masses_m3 = np.full((1, 45), 1.0 / 45.0, dtype=np.float64)
    return {
        "primary_issue_time_us": np.asarray([issue_us], dtype=np.int64),
        "primary_horizon_days": np.asarray([30], dtype=np.int16),
        "magnitude_issue_time_us": np.asarray([issue_us], dtype=np.int64),
        "location_relative_mass": np.asarray(
            [[[0.5, 0.5], [0.6, 0.4], [0.7, 0.3], [0.8, 0.2], [0.9, 0.1]]],
            dtype=np.float64,
        ),
        "t0_expected_count": np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64),
        "t1_status_code": np.full(shape, 2, dtype=np.int8),
        "t1_reason_code": np.full(shape, 9, dtype=np.int8),
        "t1_historical_block_count": np.full(shape, 7, dtype=np.int32),
        "t1_sample_mean_count": np.full(shape, 1.5, dtype=np.float64),
        "t1_sample_variance_count": np.full(shape, 2.5, dtype=np.float64),
        "t1_sample_variance_applicable": np.ones(shape, dtype=np.uint8),
        "t1_dispersion_k": np.full(shape, 4.0, dtype=np.float64),
        "t1_dispersion_k_applicable": np.ones(shape, dtype=np.uint8),
        "t1_observed_information_k": np.full(shape, 5.0, dtype=np.float64),
        "t1_observed_information_k_applicable": np.ones(shape, dtype=np.uint8),
        "t1_standard_error_k": np.full(shape, 0.5, dtype=np.float64),
        "t1_standard_error_k_applicable": np.ones(shape, dtype=np.uint8),
        "m0_training_event_count": np.asarray([20], dtype=np.int32),
        "m0_b_value": np.asarray([1.0], dtype=np.float64),
        "m0_bin_probability_mass": masses_m0,
        "m3_training_event_count": np.asarray([10], dtype=np.int32),
        "m3_b_value": np.asarray([0.9], dtype=np.float64),
        "m3_bin_probability_mass": masses_m3,
    }


def test_phase_roots_are_nonoverlapping_siblings(tmp_path: Path) -> None:
    roots = runtime.phase_roots(tmp_path / "overall")
    assert roots.prediction.parent == roots.overall
    assert roots.score.parent == roots.overall
    assert roots.prediction != roots.score
    with pytest.raises(runtime.DevelopmentRuntimeError, match="overall root"):
        runtime.phase_roots(tmp_path / "prediction_phase")


def test_prediction_module_has_no_top_level_score_import() -> None:
    tree = ast.parse(Path(runtime.__file__).read_text(encoding="utf-8"))
    top_level_imports = [
        node for node in tree.body if isinstance(node, ast.Import | ast.ImportFrom)
    ]
    assert all("development_score" not in ast.unparse(node) for node in top_level_imports)


def test_npz_writer_never_overwrites_and_safely_reloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        runtime, "validate_frozen_fold_prediction_npz_arrays", lambda *args, **kwargs: None
    )
    path = tmp_path / "predictions.npz"
    arrays = {"x": np.asarray([1], dtype=np.int64)}
    runtime._write_npz_exclusive(
        path,
        fold_id=DEVELOPMENT_FOLD_IDS[0],
        arrays=arrays,
        cell_count=1,
    )
    with pytest.raises(runtime.RuntimeArtifactExistsError, match="cannot be overwritten"):
        runtime._write_npz_exclusive(
            path,
            fold_id=DEVELOPMENT_FOLD_IDS[0],
            arrays=arrays,
            cell_count=1,
        )


def test_global_checkpoint_reuse_requires_exact_canonical_bytes(tmp_path: Path) -> None:
    path = tmp_path / "run_manifest.json"
    payload = {"schema_version": 1, "role": "development_prediction_only"}
    runtime._write_or_verify_canonical_json(path, payload)
    before = path.read_bytes()
    runtime._write_or_verify_canonical_json(path, payload)
    assert path.read_bytes() == before
    with pytest.raises(runtime.DevelopmentRuntimeError, match="byte-for-byte"):
        runtime._write_or_verify_canonical_json(path, {**payload, "schema_version": 2})


def test_npz_prediction_source_restores_frozen_lookup_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime, "validate_frozen_fold_prediction_npz_arrays", lambda *args, **kwargs: None
    )
    fold_id = DEVELOPMENT_FOLD_IDS[0]
    source = runtime.NPZDevelopmentPredictionSource(
        {fold_id: _minimal_source_arrays()}, cell_count=2
    )
    issue = datetime(1970, 1, 1, tzinfo=UTC)

    location = source.location_forecast(
        fold_id=fold_id,
        issue_time_utc=issue,
        horizon_days=30,
        model_id="L2_KDE_CAUSAL",
    )
    assert np.array_equal(np.asarray(location.cell_relative_mass), np.asarray([0.7, 0.3]))
    assert not np.asarray(location.cell_relative_mass).flags.writeable

    poisson = source.count_forecast(
        fold_id=fold_id,
        issue_time_utc=issue,
        horizon_days=30,
        model_id="T0_POISSON_EXPANDING",
        magnitude_band="M5_plus_for_joint",
    )
    assert poisson.expected_count == 3.0
    assert poisson.distribution == "poisson"
    assert poisson.nb2_qualification is None

    nb2 = source.count_forecast(
        fold_id=fold_id,
        issue_time_utc=issue,
        horizon_days=30,
        model_id="T1_NEGATIVE_BINOMIAL",
        magnitude_band="M5_6",
    )
    assert nb2.nb2_qualification is not None
    assert nb2.nb2_qualification.status == "evaluable"
    assert nb2.nb2_qualification.dispersion_k == 4.0

    magnitude = source.magnitude_forecast(
        fold_id=fold_id,
        issue_time_utc=issue,
        model_id="M0_GR_GLOBAL",
    )
    assert magnitude.model.training_event_count == 20
    assert len(magnitude.model.bin_probability_masses) == 55
    assert math.isclose(sum(magnitude.model.bin_probability_masses), 1.0)


def test_interrupted_prediction_reuses_payload_and_only_builds_missing_folds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    data = tmp_path / "data"
    project.mkdir()
    data.mkdir()
    overall = tmp_path / "overall"
    roots = runtime.phase_roots(overall)
    roots.prediction.mkdir(parents=True)
    identities = _identities()
    selection_payload = {
        "schema_version": 1,
        "record_type": "s1_c0_inner_parameter_selection",
        "run_contract_id": runtime.RUN_CONTRACT_ID,
        "role": "strictly_earlier_inner_selection_only",
        "folds": {fold_id: [] for fold_id in DEVELOPMENT_FOLD_IDS},
    }
    runtime._write_exclusive_json(
        roots.prediction / "run_manifest.json",
        runtime._run_manifest_payload(identities, maximum_fold_workers=1),
    )
    runtime._write_exclusive_json(roots.prediction / "parameter_selection.json", selection_payload)
    existing_fold = DEVELOPMENT_FOLD_IDS[0]
    existing_path = roots.prediction / "folds" / existing_fold / "predictions.npz"
    existing_path.parent.mkdir(parents=True)
    with existing_path.open("xb") as stream:
        np.savez(stream, x=np.asarray([99], dtype=np.int64))

    monkeypatch.setattr(runtime, "_require_configured_overall_root", lambda *args: None)
    monkeypatch.setattr(
        runtime, "recompute_prediction_input_identities", lambda **kwargs: identities
    )
    monkeypatch.setattr(
        runtime,
        "load_s1_runner_inputs",
        lambda **kwargs: SimpleNamespace(
            location_grid=SimpleNamespace(cell_count=runner_inputs.EXPECTED_25KM_CELL_COUNT)
        ),
    )
    monkeypatch.setattr(
        runtime, "validate_frozen_fold_prediction_npz_arrays", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(runtime, "_select_one_fold", lambda inputs, fold_id: (fold_id,))
    monkeypatch.setattr(runtime, "_parameter_selection_payload", lambda values: selection_payload)
    built: list[str] = []

    def fake_build(inputs: object, *, fold_id: str, selections: object) -> str:
        built.append(fold_id)
        return fold_id

    monkeypatch.setattr(runtime, "build_development_fold_prediction", fake_build)
    monkeypatch.setattr(
        runtime,
        "fold_prediction_npz_arrays",
        lambda prediction, cell_count: {
            "x": np.asarray([DEVELOPMENT_FOLD_IDS.index(prediction)], dtype=np.int64)
        },
    )
    monkeypatch.setattr(
        runtime,
        "frozen_fold_prediction_npz_schema",
        lambda fold_id, cell_count: {"x": {"shape": [1], "dtype": "int64"}},
    )
    sealed: list[str] = []

    def fake_seal(root: Path, fold_id: str, **kwargs: object) -> None:
        sealed.append(fold_id)
        bundle = root / "folds" / fold_id / "prediction_bundle.json"
        bundle.write_bytes(b"{}")

    monkeypatch.setattr(runtime, "seal_fold_prediction", fake_seal)
    monkeypatch.setattr(
        runtime,
        "seal_four_fold_predictions",
        lambda *args, **kwargs: _authorization(roots.prediction),
    )

    result = runtime.run_prediction_phase(
        project_root=project,
        data_root=data,
        output_root=overall,
        maximum_fold_workers=1,
    )
    assert result.master_seal_sha256 == "a" * 64
    assert existing_fold not in built
    assert set(built) == set(DEVELOPMENT_FOLD_IDS[1:])
    assert set(sealed) == set(DEVELOPMENT_FOLD_IDS)

    built.clear()
    sealed.clear()
    verified: list[str] = []
    monkeypatch.setattr(
        runtime,
        "_verify_existing_fold",
        lambda prediction_root, fold_id, **kwargs: verified.append(fold_id),
    )
    resumed = runtime.run_prediction_phase(
        project_root=project,
        data_root=data,
        output_root=overall,
        maximum_fold_workers=1,
    )
    assert resumed.master_seal_sha256 == "a" * 64
    assert built == []
    assert sealed == []
    assert verified == list(DEVELOPMENT_FOLD_IDS)


def test_official_context_path_loads_actual_npz_into_concrete_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prediction_root = tmp_path / "prediction_phase"
    for fold_id in DEVELOPMENT_FOLD_IDS:
        path = prediction_root / "folds" / fold_id / "predictions.npz"
        path.parent.mkdir(parents=True)
        with path.open("xb") as stream:
            np.savez(stream, **_minimal_source_arrays())  # type: ignore[arg-type]
    monkeypatch.setattr(
        runtime, "validate_frozen_fold_prediction_npz_arrays", lambda *args, **kwargs: None
    )
    authorization = _authorization(prediction_root)
    authorization_calls: list[object] = []

    def fake_reauthorize(context: object) -> DevelopmentScoringAuthorization:
        authorization_calls.append(context)
        return authorization

    monkeypatch.setattr(score, "_reauthorize", fake_reauthorize)
    primary_issue = SimpleNamespace(
        primary_exposure_selected=True,
        maturity_status="mature",
    )
    mature_nonprimary = SimpleNamespace(
        primary_exposure_selected=False,
        maturity_status="mature",
    )
    unavailable_nonprimary = SimpleNamespace(
        primary_exposure_selected=False,
        maturity_status="unavailable",
    )
    fake_inputs = SimpleNamespace(
        location_grid=SimpleNamespace(cell_count=runner_inputs.EXPECTED_25KM_CELL_COUNT),
        spatial_domain=SimpleNamespace(
            operational_grid=SimpleNamespace(cell_count=runner_inputs.EXPECTED_25KM_CELL_COUNT),
            locator=object(),
        ),
        catalog=object(),
        outer_issues=(primary_issue, mature_nonprimary, unavailable_nonprimary),
    )
    monkeypatch.setattr(runner_inputs, "load_s1_runner_inputs", lambda **kwargs: fake_inputs)
    captured_primary_rows: list[object] = []

    def fake_build_targets(*args: object, **kwargs: object) -> object:
        rows = kwargs["primary_issue_rows"]
        assert isinstance(rows, tuple)
        captured_primary_rows.extend(rows)
        return object()

    monkeypatch.setattr(score, "_build_primary_exposure_targets", fake_build_targets)
    monkeypatch.setattr(
        score, "_assign_standalone_magnitude_targets", lambda *args, **kwargs: object()
    )

    def fake_score(*args: object, **kwargs: object) -> score.DevelopmentRawScores:
        assert isinstance(kwargs["predictions"], runtime.NPZDevelopmentPredictionSource)
        return score.DevelopmentRawScores((), (), (), ())

    monkeypatch.setattr(score, "_score_development", fake_score)
    context = score.ScoringContext(
        output_root=prediction_root,
        expected_seal_sha256="a" * 64,
        project_root=tmp_path,
        data_root=tmp_path,
    )
    result = score.score_authorized_development_from_context(context)
    assert result.authorization == authorization
    assert len(authorization_calls) == 3
    assert captured_primary_rows == [primary_issue]


def test_run_score_rejects_before_writing_when_official_authorization_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    data = tmp_path / "data"
    project.mkdir()
    data.mkdir()
    overall = tmp_path / "overall"
    monkeypatch.setattr(runtime, "_require_configured_overall_root", lambda *args: None)

    official_called = False

    def reject_authorization(*args: object, **kwargs: object) -> object:
        raise PredictionSealError("missing or changed master seal")

    def should_not_score(context: object) -> object:
        nonlocal official_called
        official_called = True
        raise AssertionError("official scorer must not run before authorization")

    monkeypatch.setattr(runtime, "authorize_development_scoring", reject_authorization)
    monkeypatch.setattr(score, "score_authorized_development_from_context", should_not_score)
    with pytest.raises(PredictionSealError, match="master seal"):
        runtime.run_score_phase(
            project_root=project,
            data_root=data,
            output_root=overall,
            expected_seal_sha256="a" * 64,
        )
    assert not runtime.phase_roots(overall).score.exists()
    assert official_called is False


def test_run_score_writes_preregistered_summary_after_authorized_raw_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    data = tmp_path / "data"
    project.mkdir()
    data.mkdir()
    overall = project / "outputs" / "run"
    authorization = _authorization(overall / "prediction_phase")
    raw_scores = score.DevelopmentRawScores(location=(), time=(), magnitude=(), joint=())
    expected_summary = {
        "schema_version": 1,
        "record_type": "s1_c0_preregistered_development_scientific_summary",
        "champion_selection_allowed": False,
        "holdout_opening_allowed": False,
        "mandatory_next_stage_regardless_of_screen_result": (
            "S1-C1_causal_local_completeness_main"
        ),
    }
    events: list[str] = []

    monkeypatch.setattr(runtime, "_require_configured_overall_root", lambda *args: None)
    monkeypatch.setattr(
        runtime,
        "authorize_development_scoring",
        lambda *args, **kwargs: authorization,
    )
    monkeypatch.setattr(
        score,
        "score_authorized_development_from_context",
        lambda context: score.AuthorizedDevelopmentScores(
            authorization=authorization,
            scores=raw_scores,
        ),
    )

    import seismoflux.multitask_s1.development_summary as development_summary

    def summarize(received: score.DevelopmentRawScores) -> dict[str, object]:
        assert received is raw_scores
        events.append("summarize")
        return expected_summary

    def write_raw(path: Path, received: score.DevelopmentRawScores) -> dict[str, int]:
        assert received is raw_scores
        events.append("raw")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic-parquet")
        return {"location": 0, "time": 0, "magnitude": 0, "joint": 0}

    monkeypatch.setattr(development_summary, "summarize_development_scores", summarize)
    monkeypatch.setattr(runtime, "_write_raw_scores_parquet", write_raw)

    result = runtime.run_score_phase(
        project_root=project,
        data_root=data,
        output_root=overall,
        expected_seal_sha256="a" * 64,
    )

    assert events == ["summarize", "raw"]
    assert json.loads(result.summary_path.read_text(encoding="utf-8")) == expected_summary
    assert result.raw_score_row_counts == {
        "location": 0,
        "time": 0,
        "magnitude": 0,
        "joint": 0,
    }


def test_score_runtime_accepts_no_injected_scientific_objects() -> None:
    parameters = inspect.signature(runtime.run_score_phase).parameters
    assert tuple(parameters) == (
        "project_root",
        "data_root",
        "output_root",
        "expected_seal_sha256",
    )
    assert not {"source", "catalog", "grid", "targets", "issue_rows"}.intersection(parameters)
