from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import yaml

import seismoflux.multitask_s1.prediction_seal as seal_module
from seismoflux.multitask_s1.development_contract import DEVELOPMENT_FOLD_IDS, HORIZONS_DAYS
from seismoflux.multitask_s1.development_predict import TIME_BANDS
from seismoflux.multitask_s1.prediction_seal import (
    DevelopmentScoringAuthorization,
    PredictionArtifactInput,
    PredictionInputIdentities,
    PredictionSealError,
    PredictionSealExistsError,
    authorize_development_scoring,
    canonical_json_bytes,
    frozen_fold_prediction_manifest,
    recompute_prediction_input_identities,
    seal_fold_prediction,
    seal_four_fold_predictions,
    verify_fold_prediction,
)


def _identities(seed: str = "1") -> PredictionInputIdentities:
    return PredictionInputIdentities(
        run_contract_sha256=seed * 64,
        parent_contract_sha256="2" * 64,
        catalog_sha256="3" * 64,
        study_sha256="4" * 64,
        grid_sha256="5" * 64,
        issue_ledger_sha256="6" * 64,
        code_sha256="7" * 64,
        git_commit_oid="8" * 40,
    )


def _small_schema(fold_id: str, *, cell_count: int = 15_697) -> Mapping[str, Mapping[str, object]]:
    assert fold_id in DEVELOPMENT_FOLD_IDS
    assert cell_count == 15_697
    return {
        "schema_version": {"shape": [1], "dtype": "int16"},
        "fold_index": {"shape": [1], "dtype": "int8"},
        "forecast_probability": {"shape": [2, 3], "dtype": "float64"},
        "primary_horizon_days": {"shape": [5], "dtype": "int16"},
        "location_regional_tau_years": {"shape": [5], "dtype": "float64"},
        "location_bandwidth_km": {"shape": [5, 5], "dtype": "float64"},
        "location_alpha": {"shape": [5, 5], "dtype": "float64"},
        "t1_status_code": {"shape": [5, 3], "dtype": "int8"},
        "t1_reason_code": {"shape": [5, 3], "dtype": "int8"},
        "t1_historical_block_count": {"shape": [5, 3], "dtype": "int16"},
        "t1_sample_mean_count": {"shape": [5, 3], "dtype": "float64"},
        "t1_sample_variance_count": {"shape": [5, 3], "dtype": "float64"},
        "t1_sample_variance_applicable": {"shape": [5, 3], "dtype": "int8"},
        "t1_dispersion_k": {"shape": [5, 3], "dtype": "float64"},
        "t1_dispersion_k_applicable": {"shape": [5, 3], "dtype": "int8"},
        "t1_observed_information_k": {"shape": [5, 3], "dtype": "float64"},
        "t1_observed_information_k_applicable": {"shape": [5, 3], "dtype": "int8"},
        "t1_standard_error_k": {"shape": [5, 3], "dtype": "float64"},
        "t1_standard_error_k_applicable": {"shape": [5, 3], "dtype": "int8"},
    }


def _small_validator(
    fold_id: str,
    arrays: Mapping[str, object],
    *,
    cell_count: int = 15_697,
) -> None:
    schema = _small_schema(fold_id, cell_count=cell_count)
    if set(arrays) != set(schema):
        raise ValueError("keys differ from frozen schema")
    materialized = {key: np.asarray(value) for key, value in arrays.items()}
    for key, specification in schema.items():
        if materialized[key].shape != tuple(cast(list[int], specification["shape"])):
            raise ValueError("shape differs from frozen schema")
        if materialized[key].dtype != np.dtype(cast(str, specification["dtype"])):
            raise ValueError("dtype differs from frozen schema")
        if not np.isfinite(materialized[key]).all():
            raise ValueError("prediction arrays must be finite")
    fold_index = DEVELOPMENT_FOLD_IDS.index(fold_id)
    if not np.array_equal(materialized["schema_version"], np.asarray([1], dtype=np.int16)):
        raise ValueError("schema version changed")
    if not np.array_equal(materialized["fold_index"], np.asarray([fold_index], dtype=np.int8)):
        raise ValueError("foreign fold payload")
    probability = materialized["forecast_probability"]
    if (probability < 0.0).any() or not np.allclose(probability.sum(axis=1), 1.0):
        raise ValueError("probability mass is not normalized")


@pytest.fixture
def sealed_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, PredictionInputIdentities]:
    state = {"identities": _identities()}
    monkeypatch.setattr(
        seal_module,
        "recompute_prediction_input_identities",
        lambda **_kwargs: state["identities"],
    )
    monkeypatch.setattr(seal_module, "frozen_fold_prediction_npz_schema", _small_schema)
    monkeypatch.setattr(
        seal_module,
        "validate_frozen_fold_prediction_npz_arrays",
        _small_validator,
    )
    monkeypatch.setattr(
        seal_module,
        "_require_configured_prediction_root",
        lambda _project_root, output_root: Path(output_root).resolve(),
    )
    return state


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))


def _mutate_selection(root: Path, mutate: Any) -> None:
    path = root / "parameter_selection.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    _write_json(path, payload)


def _write_phase_documents(root: Path, identities: PredictionInputIdentities) -> None:
    _write_json(
        root / "run_manifest.json",
        {
            "schema_version": 1,
            "record_type": "s1_c0_prediction_run_manifest",
            "run_contract_id": "multitask-s1-c0-all-m4-screen-v1",
            "stage": "S1-C0",
            "role": "development_prediction_only",
            "git_commit_oid": identities.git_commit_oid,
            "input_identities": identities.as_mapping(),
            "fold_ids": list(DEVELOPMENT_FOLD_IDS),
            "maximum_fold_workers": 3,
            "numerical_threads_per_worker": 1,
            "outer_targets_constructed": False,
            "model_scores_read": False,
            "locked_test_run": False,
        },
    )
    _write_json(
        root / "parameter_selection.json",
        {
            "schema_version": 1,
            "record_type": "s1_c0_inner_parameter_selection",
            "run_contract_id": "multitask-s1-c0-all-m4-screen-v1",
            "role": "strictly_earlier_inner_selection_only",
            "folds": {fold_id: _selection_entries() for fold_id in DEVELOPMENT_FOLD_IDS},
        },
    )


def _candidate_rows(axis: tuple[float, ...], best: float | None = None) -> list[dict[str, object]]:
    return [
        {
            "parameter_value": value,
            "inner_block_mean_log_density": [1.0 if best == value else 0.0] * 3,
        }
        for value in axis
    ]


def _selection_entries() -> list[dict[str, object]]:
    qualification = {
        "status": "poisson_limit",
        "reason": "sample_variance_not_greater_than_sample_mean",
        "historical_block_count": 3,
        "sample_mean_count": 1.0,
        "sample_variance_count": 1.0,
        "dispersion_k": None,
        "observed_information_k": None,
        "standard_error_k": None,
    }
    return [
        {
            "horizon_days": horizon,
            "inner_evidence": {
                "location": {
                    "latest_inner_target_end_us": 0,
                    "evaluation_start_boundary_us": 31 * 86_400_000_000,
                    "inner_block_event_counts": [4, 3, 3],
                    "selected_regional_tau_years": 10.0,
                    "selected_kde_bandwidth_km": 75.0,
                    "selected_recent_alpha": 0.25,
                    "regional_candidates": _candidate_rows((1.0, 5.0, 10.0)),
                    "kde_candidates": _candidate_rows((75.0, 100.0, 150.0, 200.0, 300.0), 75.0),
                    "recent_candidates": _candidate_rows((0.0, 0.25, 0.5, 0.75), 0.25),
                },
                "time": [{"band": band, "qualification": qualification} for band in TIME_BANDS],
            },
        }
        for horizon in HORIZONS_DAYS
    ]


def _write_prediction_npz(
    root: Path,
    fold_id: str,
    *,
    probability: np.ndarray[Any, np.dtype[np.float64]] | None = None,
    destination: Path | None = None,
) -> PredictionArtifactInput:
    path = destination or root / "folds" / fold_id / "predictions.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    fold_index = DEVELOPMENT_FOLD_IDS.index(fold_id)
    if probability is None:
        probability = np.asarray(
            [[0.2 + 0.01 * fold_index, 0.3, 0.5 - 0.01 * fold_index], [0.1, 0.2, 0.7]],
            dtype=np.float64,
        )
    np.savez(
        path,
        schema_version=np.asarray([1], dtype=np.int16),
        fold_index=np.asarray([fold_index], dtype=np.int8),
        forecast_probability=probability,
        primary_horizon_days=np.asarray(HORIZONS_DAYS, dtype=np.int16),
        location_regional_tau_years=np.full(5, 10.0),
        location_bandwidth_km=np.column_stack(
            (np.zeros(5), np.zeros(5), np.full(5, 75.0), np.full(5, 75.0), np.full(5, 75.0))
        ),
        location_alpha=np.column_stack(
            (np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(5), np.full(5, 0.25))
        ),
        t1_status_code=np.ones((5, 3), dtype=np.int8),
        t1_reason_code=np.full((5, 3), 3, dtype=np.int8),
        t1_historical_block_count=np.full((5, 3), 3, dtype=np.int16),
        t1_sample_mean_count=np.ones((5, 3)),
        t1_sample_variance_count=np.ones((5, 3)),
        t1_sample_variance_applicable=np.ones((5, 3), dtype=np.int8),
        t1_dispersion_k=np.zeros((5, 3)),
        t1_dispersion_k_applicable=np.zeros((5, 3), dtype=np.int8),
        t1_observed_information_k=np.zeros((5, 3)),
        t1_observed_information_k_applicable=np.zeros((5, 3), dtype=np.int8),
        t1_standard_error_k=np.zeros((5, 3)),
        t1_standard_error_k_applicable=np.zeros((5, 3), dtype=np.int8),
    )
    return PredictionArtifactInput(path=path, schema=_small_schema(fold_id))


def _seal_fold(root: Path, fold_id: str) -> None:
    seal_fold_prediction(
        root,
        fold_id,
        prediction_manifest=frozen_fold_prediction_manifest(fold_id),
        prediction_artifacts=[_write_prediction_npz(root, fold_id)],
        project_root=root / "synthetic_project",
        data_root=root / "synthetic_data",
    )


def _seal_all(root: Path, identities: PredictionInputIdentities) -> None:
    _write_phase_documents(root, identities)
    for fold_id in DEVELOPMENT_FOLD_IDS:
        _seal_fold(root, fold_id)


def _master(root: Path) -> DevelopmentScoringAuthorization:
    return seal_four_fold_predictions(
        root,
        project_root=root / "synthetic_project",
        data_root=root / "synthetic_data",
    )


def _authorize(root: Path, seal_hash: str) -> DevelopmentScoringAuthorization:
    return authorize_development_scoring(
        root,
        expected_seal_sha256=seal_hash,
        project_root=root / "synthetic_project",
        data_root=root / "synthetic_data",
    )


def test_canonical_json_is_stable_and_rejects_nonfinite() -> None:
    assert canonical_json_bytes({"b": 2, "a": "地震"}) == '{"a":"地震","b":2}'.encode()
    with pytest.raises(PredictionSealError, match="non-finite"):
        canonical_json_bytes({"bad": float("nan")})


def test_legal_four_fold_chain_binds_payloads_and_phase_json(
    tmp_path: Path,
    sealed_dependencies: dict[str, PredictionInputIdentities],
) -> None:
    root = tmp_path / "prediction_phase"
    _seal_all(root, sealed_dependencies["identities"])
    authorization = _master(root)
    assert isinstance(authorization, DevelopmentScoringAuthorization)
    master = json.loads(authorization.seal.path.read_text(encoding="utf-8"))
    assert [item["relative_path"] for item in master["prediction_phase_artifacts"]] == [
        "run_manifest.json",
        "parameter_selection.json",
    ]
    assert _authorize(root, authorization.seal.sha256) == authorization


def test_fold_seal_is_exclusive_and_requires_exact_manifest(
    tmp_path: Path,
    sealed_dependencies: dict[str, PredictionInputIdentities],
) -> None:
    root = tmp_path / "prediction_phase"
    _write_phase_documents(root, sealed_dependencies["identities"])
    fold_id = DEVELOPMENT_FOLD_IDS[0]
    artifact = _write_prediction_npz(root, fold_id)
    seal_fold_prediction(
        root,
        fold_id,
        prediction_manifest=frozen_fold_prediction_manifest(fold_id),
        prediction_artifacts=[artifact],
        project_root=root,
        data_root=root,
    )
    with pytest.raises(PredictionSealExistsError):
        seal_fold_prediction(
            root,
            fold_id,
            prediction_manifest=frozen_fold_prediction_manifest(fold_id),
            prediction_artifacts=[artifact],
            project_root=root,
            data_root=root,
        )

    second_root = tmp_path / "second_prediction_phase"
    _write_phase_documents(second_root, sealed_dependencies["identities"])
    bad = dict(frozen_fold_prediction_manifest(fold_id))
    bad["outer_score"] = 1.0
    with pytest.raises(PredictionSealError, match="frozen fold manifest"):
        seal_fold_prediction(
            second_root,
            fold_id,
            prediction_manifest=bad,
            prediction_artifacts=[_write_prediction_npz(second_root, fold_id)],
            project_root=second_root,
            data_root=second_root,
        )


@pytest.mark.parametrize("count", [0, 2])
def test_each_fold_requires_exactly_one_payload(
    tmp_path: Path,
    sealed_dependencies: dict[str, PredictionInputIdentities],
    count: int,
) -> None:
    root = tmp_path / "prediction_phase"
    _write_phase_documents(root, sealed_dependencies["identities"])
    fold_id = DEVELOPMENT_FOLD_IDS[0]
    artifact = _write_prediction_npz(root, fold_id)
    with pytest.raises(PredictionSealError, match="exactly one"):
        seal_fold_prediction(
            root,
            fold_id,
            prediction_manifest=frozen_fold_prediction_manifest(fold_id),
            prediction_artifacts=[artifact] * count,
            project_root=root,
            data_root=root,
        )


def test_payload_must_be_exact_canonical_fold_npz_and_not_symlink(
    tmp_path: Path,
    sealed_dependencies: dict[str, PredictionInputIdentities],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "prediction_phase"
    _write_phase_documents(root, sealed_dependencies["identities"])
    fold_id = DEVELOPMENT_FOLD_IDS[0]
    wrong = _write_prediction_npz(
        root,
        fold_id,
        destination=root / "folds" / DEVELOPMENT_FOLD_IDS[1] / "predictions.npz",
    )
    with pytest.raises(PredictionSealError, match="wrong fold"):
        seal_fold_prediction(
            root,
            fold_id,
            prediction_manifest=frozen_fold_prediction_manifest(fold_id),
            prediction_artifacts=[wrong],
            project_root=root,
            data_root=root,
        )

    canonical = root / "folds" / fold_id / "predictions.npz"
    artifact = _write_prediction_npz(root, fold_id)
    original_is_symlink = Path.is_symlink

    def report_canonical_as_symlink(path: Path) -> bool:
        if path == canonical:
            return True
        return original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", report_canonical_as_symlink)
    with pytest.raises(PredictionSealError, match="symlink"):
        seal_fold_prediction(
            root,
            fold_id,
            prediction_manifest=frozen_fold_prediction_manifest(fold_id),
            prediction_artifacts=[artifact],
            project_root=root,
            data_root=root,
        )


def test_schema_is_three_way_frozen_and_zero_dimensions_are_rejected(
    tmp_path: Path,
    sealed_dependencies: dict[str, PredictionInputIdentities],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "prediction_phase"
    _write_phase_documents(root, sealed_dependencies["identities"])
    fold_id = DEVELOPMENT_FOLD_IDS[0]
    artifact = _write_prediction_npz(root, fold_id)
    wrong_schema = dict(_small_schema(fold_id))
    wrong_schema["forecast_probability"] = {"shape": [3, 2], "dtype": "float64"}
    with pytest.raises(PredictionSealError, match="caller schema"):
        seal_fold_prediction(
            root,
            fold_id,
            prediction_manifest=frozen_fold_prediction_manifest(fold_id),
            prediction_artifacts=[PredictionArtifactInput(artifact.path, wrong_schema)],
            project_root=root,
            data_root=root,
        )

    zero_schema = dict(_small_schema(fold_id))
    zero_schema["forecast_probability"] = {"shape": [0, 3], "dtype": "float64"}
    monkeypatch.setattr(
        seal_module,
        "frozen_fold_prediction_npz_schema",
        lambda *_args, **_kwargs: zero_schema,
    )
    with pytest.raises(PredictionSealError, match="zero/invalid"):
        seal_fold_prediction(
            root,
            fold_id,
            prediction_manifest=frozen_fold_prediction_manifest(fold_id),
            prediction_artifacts=[PredictionArtifactInput(artifact.path, zero_schema)],
            project_root=root,
            data_root=root,
        )


@pytest.mark.parametrize(
    "probability",
    [
        np.asarray([[np.nan, 0.0, 1.0], [0.1, 0.2, 0.7]], dtype=np.float64),
        np.asarray([[0.2, 0.2, 0.2], [0.1, 0.2, 0.7]], dtype=np.float64),
    ],
)
def test_actual_npz_science_invariants_are_rechecked(
    tmp_path: Path,
    sealed_dependencies: dict[str, PredictionInputIdentities],
    probability: np.ndarray[Any, np.dtype[np.float64]],
) -> None:
    root = tmp_path / "prediction_phase"
    _write_phase_documents(root, sealed_dependencies["identities"])
    fold_id = DEVELOPMENT_FOLD_IDS[0]
    artifact = _write_prediction_npz(root, fold_id, probability=probability)
    with pytest.raises(PredictionSealError, match="prediction (NPZ|payload)"):
        seal_fold_prediction(
            root,
            fold_id,
            prediction_manifest=frozen_fold_prediction_manifest(fold_id),
            prediction_artifacts=[artifact],
            project_root=root,
            data_root=root,
        )


def test_object_pickle_npz_is_rejected(
    tmp_path: Path,
    sealed_dependencies: dict[str, PredictionInputIdentities],
) -> None:
    root = tmp_path / "prediction_phase"
    _write_phase_documents(root, sealed_dependencies["identities"])
    fold_id = DEVELOPMENT_FOLD_IDS[0]
    path = root / "folds" / fold_id / "predictions.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        schema_version=np.asarray([1], dtype=np.int16),
        fold_index=np.asarray([0], dtype=np.int8),
        forecast_probability=np.asarray([{"bad": 1}], dtype=object),
    )
    with pytest.raises(PredictionSealError, match="prediction NPZ"):
        seal_fold_prediction(
            root,
            fold_id,
            prediction_manifest=frozen_fold_prediction_manifest(fold_id),
            prediction_artifacts=[PredictionArtifactInput(path, _small_schema(fold_id))],
            project_root=root,
            data_root=root,
        )


@pytest.mark.parametrize("mutation", ["tamper", "delete", "valid_replacement"])
def test_payload_mutation_after_master_revokes_authorization(
    tmp_path: Path,
    sealed_dependencies: dict[str, PredictionInputIdentities],
    mutation: str,
) -> None:
    root = tmp_path / "prediction_phase"
    _seal_all(root, sealed_dependencies["identities"])
    authorization = _master(root)
    path = root / "folds" / DEVELOPMENT_FOLD_IDS[0] / "predictions.npz"
    if mutation == "delete":
        path.unlink()
    elif mutation == "tamper":
        path.write_bytes(b"not-an-npz")
    else:
        _write_prediction_npz(
            root,
            DEVELOPMENT_FOLD_IDS[0],
            probability=np.asarray([[0.25, 0.25, 0.5], [0.2, 0.2, 0.6]], dtype=np.float64),
        )
    with pytest.raises(PredictionSealError):
        _authorize(root, authorization.seal.sha256)


@pytest.mark.parametrize(
    "relative_path",
    ["notes.txt", "targets.json", "scores/result.json", "score_authorization.json"],
)
def test_output_tree_is_an_exact_prediction_phase_allowlist(
    tmp_path: Path,
    sealed_dependencies: dict[str, PredictionInputIdentities],
    relative_path: str,
) -> None:
    root = tmp_path / "prediction_phase"
    _seal_all(root, sealed_dependencies["identities"])
    authorization = _master(root)
    extra = root / relative_path
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_text("{}", encoding="utf-8")
    with pytest.raises(PredictionSealError, match="unexpected output"):
        _authorize(root, authorization.seal.sha256)


def test_phase_json_outer_field_and_noncanonical_json_are_rejected(
    tmp_path: Path,
    sealed_dependencies: dict[str, PredictionInputIdentities],
) -> None:
    root = tmp_path / "prediction_phase"
    _seal_all(root, sealed_dependencies["identities"])
    selection_path = root / "parameter_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["folds"][DEVELOPMENT_FOLD_IDS[0]][0]["outer_score"] = 1.0
    _write_json(selection_path, selection)
    with pytest.raises(PredictionSealError, match="outer-fold"):
        _master(root)

    clean_root = tmp_path / "clean_prediction_phase"
    _seal_all(clean_root, sealed_dependencies["identities"])
    (clean_root / "run_manifest.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(PredictionSealError, match="canonical"):
        _master(clean_root)


def test_recomputed_identity_drift_revokes_authorization(
    tmp_path: Path,
    sealed_dependencies: dict[str, PredictionInputIdentities],
) -> None:
    root = tmp_path / "prediction_phase"
    _seal_all(root, sealed_dependencies["identities"])
    authorization = _master(root)
    sealed_dependencies["identities"] = _identities("9")
    with pytest.raises(PredictionSealError, match="identity"):
        _authorize(root, authorization.seal.sha256)


def _run_git(root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(["git", *arguments], cwd=root, check=True, capture_output=True)
    return completed.stdout


def test_identity_recompute_binds_files_grid_git_object_and_clean_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    data = tmp_path / "data"
    project.mkdir()
    data.mkdir()
    parent = project / "configs" / "multitask_s1_development_contract.yaml"
    catalog = data / "processed" / "stage1" / "debc98054172a4a1" / "earthquake_event.parquet"
    study = data / "processed" / "china_mainland.geojson"
    ledger = (
        project
        / "outputs"
        / "multitask_s0"
        / "s0_score_blind_20260901"
        / "issue_maturity_ledger.csv"
    )
    for path, payload in (
        (parent, b"parent-contract"),
        (catalog, b"catalog"),
        (study, b"study-area"),
        (ledger, b"issue-ledger"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    grid_id = "a" * 64
    run_document = {
        "run_contract_id": "multitask-s1-c0-all-m4-screen-v1",
        "parent_contract": {
            "path": "configs/multitask_s1_development_contract.yaml",
            "sha256": hashlib.sha256(parent.read_bytes()).hexdigest(),
        },
        "input_identities": {
            "authoritative_catalog": {
                "root": "data_root",
                "path": "processed/stage1/debc98054172a4a1/earthquake_event.parquet",
                "sha256": hashlib.sha256(catalog.read_bytes()).hexdigest(),
            },
            "study_area": {
                "root": "data_root",
                "path": "processed/china_mainland.geojson",
                "sha256": hashlib.sha256(study.read_bytes()).hexdigest(),
            },
            "operational_grid": {
                "cell_size_km": 25.0,
                "grid_id": grid_id,
                "cell_count": 2,
                "exact_area_km2": 3.0,
            },
            "issue_ledger": {
                "root": "project_root",
                "path": "outputs/multitask_s0/s0_score_blind_20260901/issue_maturity_ledger.csv",
                "sha256": hashlib.sha256(ledger.read_bytes()).hexdigest(),
            },
        },
    }
    run_path = project / "configs" / "multitask_s1_development_run.yaml"
    run_path.write_text(yaml.safe_dump(run_document, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(seal_module, "load_development_contract", lambda *_a, **_k: ({}, None))
    monkeypatch.setattr(
        seal_module,
        "verify_authoritative_catalog_identity",
        lambda path: {"file_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()},
    )
    fake_grid = SimpleNamespace(grid_id=grid_id, cell_count=2, clipped_area_km2=[1.0, 2.0])
    monkeypatch.setattr(
        seal_module,
        "build_d1_spatial_domain_from_bytes",
        lambda _payload: SimpleNamespace(operational_grid=fake_grid),
    )
    monkeypatch.setattr(
        seal_module,
        "EXPECTED_STUDY_AREA_SHA256",
        hashlib.sha256(b"study-area").hexdigest(),
    )
    monkeypatch.setattr(seal_module, "EXPECTED_25KM_GRID_ID", grid_id)
    monkeypatch.setattr(seal_module, "EXPECTED_25KM_CELL_COUNT", 2)
    monkeypatch.setattr(seal_module, "EXPECTED_TOTAL_AREA_KM2", 3.0)
    _run_git(project, "init")
    _run_git(project, "config", "user.email", "science@example.invalid")
    _run_git(project, "config", "user.name", "Science Test")
    _run_git(project, "add", ".")
    _run_git(project, "commit", "-m", "freeze inputs")

    identities = recompute_prediction_input_identities(project_root=project, data_root=data)
    oid = _run_git(project, "rev-parse", "HEAD").decode().strip()
    payload = _run_git(project, "cat-file", "commit", oid)
    preimage = b"commit " + str(len(payload)).encode("ascii") + b"\0" + payload
    assert identities.git_commit_oid == oid
    assert identities.code_sha256 == hashlib.sha256(preimage).hexdigest()
    assert identities.run_contract_sha256 == hashlib.sha256(run_path.read_bytes()).hexdigest()
    assert identities.parent_contract_sha256 == hashlib.sha256(parent.read_bytes()).hexdigest()
    assert identities.issue_ledger_sha256 == hashlib.sha256(ledger.read_bytes()).hexdigest()

    (project / "dirty.txt").write_text("dirty", encoding="utf-8")
    with pytest.raises(PredictionSealError, match="completely clean"):
        recompute_prediction_input_identities(project_root=project, data_root=data)


@pytest.mark.parametrize("bad", [{}, {"arbitrary": True}])
def test_inner_selection_structure_is_frozen(
    tmp_path: Path,
    sealed_dependencies: dict[str, PredictionInputIdentities],
    bad: object,
) -> None:
    root = tmp_path / "prediction_phase"
    _write_phase_documents(root, sealed_dependencies["identities"])
    fold_id = DEVELOPMENT_FOLD_IDS[0]
    _write_prediction_npz(root, fold_id)
    selection = json.loads((root / "parameter_selection.json").read_text(encoding="utf-8"))
    selection["folds"][fold_id] = bad
    _write_json(root / "parameter_selection.json", selection)
    with pytest.raises(PredictionSealError, match="selection|horizons"):
        _seal_fold(root, fold_id)


def test_selection_rule_and_npz_cross_binding_are_enforced(
    tmp_path: Path, sealed_dependencies: dict[str, PredictionInputIdentities]
) -> None:
    root = tmp_path / "prediction_phase"
    _write_phase_documents(root, sealed_dependencies["identities"])
    fold_id = DEVELOPMENT_FOLD_IDS[0]
    _write_prediction_npz(root, fold_id)
    path = root / "parameter_selection.json"
    selection = json.loads(path.read_text(encoding="utf-8"))
    location = selection["folds"][fold_id][0]["inner_evidence"]["location"]
    location["selected_recent_alpha"] = 0.5
    _write_json(path, selection)
    with pytest.raises(PredictionSealError, match="frozen selection rule"):
        _seal_fold(root, fold_id)

    _write_phase_documents(root, sealed_dependencies["identities"])
    npz = root / "folds" / fold_id / "predictions.npz"
    with np.load(npz, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    arrays["location_regional_tau_years"] = np.full(5, 5.0)
    np.savez(npz, **arrays)
    with pytest.raises(PredictionSealError, match="selection differs"):
        seal_fold_prediction(
            root,
            fold_id,
            prediction_manifest=frozen_fold_prediction_manifest(fold_id),
            prediction_artifacts=[PredictionArtifactInput(npz, _small_schema(fold_id))],
            project_root=root,
            data_root=root,
        )


def test_verify_fold_accepts_legal_payload_and_rejects_tamper(
    tmp_path: Path, sealed_dependencies: dict[str, PredictionInputIdentities]
) -> None:
    root = tmp_path / "prediction_phase"
    _write_phase_documents(root, sealed_dependencies["identities"])
    fold_id = DEVELOPMENT_FOLD_IDS[0]
    _seal_fold(root, fold_id)
    sealed = verify_fold_prediction(root, fold_id, project_root=root, data_root=root)
    assert sealed.path == root / "folds" / fold_id / "prediction_bundle.json"
    npz = root / "folds" / fold_id / "predictions.npz"
    npz.write_bytes(npz.read_bytes() + b"tamper")
    with pytest.raises(PredictionSealError):
        verify_fold_prediction(root, fold_id, project_root=root, data_root=root)


def test_configured_prediction_root_rejects_alias(tmp_path: Path) -> None:
    project = tmp_path / "project"
    config = project / "configs" / "multitask_s1_development_run.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        yaml.safe_dump(
            {
                "outputs": {
                    "root": "outputs/multitask_s1/s1c0_all_m4_screen_v1_attempt2",
                    "prediction_root": (
                        "outputs/multitask_s1/s1c0_all_m4_screen_v1_attempt2/prediction_phase"
                    ),
                    "score_root": (
                        "outputs/multitask_s1/s1c0_all_m4_screen_v1_attempt2/score_phase"
                    ),
                    "phase_roots_must_be_siblings_and_nonoverlapping": True,
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(PredictionSealError, match="configured prediction_phase"):
        seal_module._require_configured_prediction_root(
            project, project / "outputs" / "development_predictions"
        )
