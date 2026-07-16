from __future__ import annotations

import inspect
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest
from shapely.geometry import box

from seismoflux.anomaly_increment.compute import (
    GPU_BACKEND_ID,
    BackendEquivalenceEvidence,
    NumpyFloat64Backend,
    build_compute_plan,
    qualify_worker_invariance,
)
from seismoflux.anomaly_increment.config import (
    STAGE4_PROTOCOL_PATH,
    STAGE4_PROTOCOL_TAG,
    STAGE4_RESULT_TAG,
    STAGE4_SCORING_CODE_TAG,
    load_stage4_protocol_bundle,
    validate_stage4_r2_execution_contract,
)
from seismoflux.anomaly_increment.grid_features import (
    assert_selected_columns_exact,
    build_stage4_grid_family,
    build_stage4_integration_grid,
    extract_raw_feature_matrix,
    selected_table_identity_sha256,
    source_columns_for_variant,
)
from seismoflux.anomaly_increment.runner import (
    ExposurePlan,
    build_stage4_scoring_plan,
)


def test_protocol_bundle_loading_never_probes_the_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guarded_fragment = "earthquake_event"
    original_open = Path.open
    original_exists = Path.exists
    original_stat = Path.stat
    original_read_bytes = Path.read_bytes

    def assert_not_target(path: Path) -> None:
        assert guarded_fragment not in path.as_posix()

    def guarded_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        assert_not_target(path)
        return original_open(path, *args, **kwargs)

    def guarded_exists(path: Path) -> bool:
        assert_not_target(path)
        return original_exists(path)

    def guarded_stat(path: Path, *args: Any, **kwargs: Any) -> Any:
        assert_not_target(path)
        return original_stat(path, *args, **kwargs)

    def guarded_read_bytes(path: Path) -> bytes:
        assert_not_target(path)
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(Path, "exists", guarded_exists)
    monkeypatch.setattr(Path, "stat", guarded_stat)
    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)

    bundle = load_stage4_protocol_bundle(Path.cwd())
    assert bundle.expected_target.relative_path.endswith("earthquake_event.parquet")
    assert bundle.validation_receipt["target_read_count"] == 0


def test_score_free_bundle_identity_is_complete_and_unobserved() -> None:
    bundle = load_stage4_protocol_bundle(Path.cwd())
    identity = bundle.score_free_identity()

    assert identity["protocol_tag"] == STAGE4_PROTOCOL_TAG
    assert identity["expected_scoring_code_tag"] == STAGE4_SCORING_CODE_TAG
    assert identity["formal_attempt_count"] == 0
    assert identity["target_read_count"] == 0
    assert identity["locked_test_run"] is False
    target = identity["expected_target"]
    assert isinstance(target, dict)
    assert target["observed"] is False
    assert len(identity["manifest_file_sha256"]) == 4  # type: ignore[arg-type]
    assert len(identity["manifest_content_sha256"]) == 4  # type: ignore[arg-type]


def test_runtime_defaults_and_machine_contract_are_locked_to_execution_r2() -> None:
    bundle = load_stage4_protocol_bundle(Path.cwd())
    freeze = bundle.protocol["freeze"]
    assert isinstance(freeze, dict)
    assert STAGE4_PROTOCOL_PATH.as_posix() == "configs/anomaly_increment_r2.yaml"
    assert STAGE4_PROTOCOL_TAG.endswith("-protocol-r2")
    assert STAGE4_SCORING_CODE_TAG.endswith("-scoring-code-r2")
    assert STAGE4_RESULT_TAG.endswith("-increment-r2")
    assert freeze["execution_revision"] == "r2"
    validate_stage4_r2_execution_contract(bundle.protocol)


@pytest.mark.parametrize(
    ("field", "legacy_value"),
    (
        ("execution_revision", "r1"),
        ("pre_score_tag", "v0.3.0-anomaly-increment-protocol-r1"),
        ("results_tag", "v0.3.0-anomaly-increment-r1"),
    ),
)
def test_runtime_machine_contract_rejects_r1_freeze_identity(
    field: str,
    legacy_value: str,
) -> None:
    protocol = deepcopy(dict(load_stage4_protocol_bundle(Path.cwd()).protocol))
    freeze = protocol["freeze"]
    assert isinstance(freeze, dict)
    freeze[field] = legacy_value

    with pytest.raises(ValueError, match="stage-4"):
        validate_stage4_r2_execution_contract(protocol)


@pytest.mark.parametrize(
    ("path", "drifted_value"),
    (
        (("compute", "max_workers"), 7),
        (("evaluation", "permutations", "formal_requests"), []),
        (("evaluation", "permutations", "exact_request_set_required"), False),
        (("evaluation", "gates", "G2", "primary_space_permutation_p_lte"), 0.10),
        (
            (
                "inputs",
                "earthquake_target",
                "frozen_catalog_coverage",
                "observed_available_at_max_utc",
            ),
            "2025-07-01T00:00:00Z",
        ),
        (("publication", "result_identity_requires"), ["dynamic_G2"]),
        (("publication", "display_semantics", "coverage_only_option_required"), False),
        (("publication", "spatial_output_isolation", "physical_file_count"), 2),
        (
            (
                "publication",
                "spatial_output_isolation",
                "public_forecast_artifact_validator",
                "keyword_scan_or_ui_hiding_sufficient",
            ),
            True,
        ),
        (
            (
                "evaluation",
                "gates",
                "G2",
                "reporting_confound_guard_applies_independently_to_variants",
            ),
            ["dynamic"],
        ),
    ),
)
def test_runtime_machine_contract_rejects_r2_critical_drift(
    path: tuple[str, ...],
    drifted_value: object,
) -> None:
    protocol = deepcopy(dict(load_stage4_protocol_bundle(Path.cwd()).protocol))
    node: dict[str, Any] = protocol
    for key in path[:-1]:
        child = node[key]
        assert isinstance(child, dict)
        node = child
    node[path[-1]] = drifted_value

    with pytest.raises(ValueError, match="stage-4 R2"):
        validate_stage4_r2_execution_contract(protocol)


def test_target_blind_scoring_plan_freezes_all_fit_scopes_and_counts() -> None:
    bundle = load_stage4_protocol_bundle(Path.cwd())
    plan = build_stage4_scoring_plan(bundle)

    assert [len(scope.fit_exposures_7d) for scope in plan.fit_scopes] == [12, 25, 38, 50]
    assert [[len(item.exposures) for item in scope.assessments] for scope in plan.fit_scopes] == [
        [12, 2, 1],
        [12, 1, 1],
        [11, 2, 1],
        [51, 11, 4, 2, 1],
    ]
    assert plan.formal_attempt_count == 0
    assert plan.target_read_count == 0
    assert plan.locked_test_run is False
    assert len(plan.content_sha256) == 64
    assert plan.content_sha256 == build_stage4_scoring_plan(bundle).content_sha256

    serialized = json.dumps(plan.as_dict(), sort_keys=True)
    assert "earthquake_event" not in serialized
    assert "epicenter" not in serialized


def test_exposure_parser_preserves_open_closed_window() -> None:
    exposure = ExposurePlan.parse("validation-h090-2025-04-03")
    assert exposure.identifier == "validation-h090-2025-04-03"
    assert exposure.target_start_exclusive_local.isoformat() == "2025-04-03"
    assert exposure.target_end_inclusive_local.isoformat() == "2025-07-02"

    with pytest.raises(ValueError, match="exposure horizon changed"):
        ExposurePlan.parse("validation-h091-2025-04-03")


def test_compute_plan_reserves_cores_and_keeps_current_protocol_cpu_only() -> None:
    bundle = load_stage4_protocol_bundle(Path.cwd())
    evidence = BackendEquivalenceEvidence(
        candidate_backend=GPU_BACKEND_ID,
        objective_relative_error=0.0,
        gradient_max_abs_error=0.0,
        coefficient_max_abs_error=0.0,
        integrated_intensity_relative_error=0.0,
        random_mapping_byte_identity=True,
        scientific_decision_identity=True,
        repeated_run_identity=True,
        worker_count_identity=True,
    )
    plan = build_compute_plan(
        bundle,
        requested_backend=GPU_BACKEND_ID,
        gpu_evidence=evidence,
        detected_physical_cores=24,
        detected_logical_processors=48,
    )

    assert plan.backend == "cpu_float64"
    assert plan.gpu_fallback_reason == "project_environment_has_no_frozen_gpu_backend"
    assert plan.workers.effective_workers == 6
    assert plan.workers.reserve_physical_cores == 2
    assert set(plan.workers.blas_environment().values()) == {"1"}


def test_compute_plan_caps_configured_physical_cores_at_available_logical_cores() -> None:
    bundle = load_stage4_protocol_bundle(Path.cwd())
    plan = build_compute_plan(bundle, detected_logical_processors=4)

    assert plan.workers.physical_cores == 4
    assert plan.workers.logical_processors == 4
    assert plan.workers.reserve_physical_cores == 2
    assert plan.workers.effective_workers == 2


def test_compute_plan_rejects_explicitly_inconsistent_detected_core_counts() -> None:
    bundle = load_stage4_protocol_bundle(Path.cwd())

    with pytest.raises(ValueError, match="logical processor count cannot be below physical"):
        build_compute_plan(
            bundle,
            detected_physical_cores=24,
            detected_logical_processors=4,
        )


def test_compute_plan_fails_closed_when_two_cores_cannot_be_reserved() -> None:
    bundle = load_stage4_protocol_bundle(Path.cwd())

    with pytest.raises(ValueError, match="reserved physical cores must be fewer"):
        build_compute_plan(bundle, detected_logical_processors=2)


def test_numpy_float64_backend_rejects_nonfinite_and_shape_drift() -> None:
    backend = NumpyFloat64Backend()
    result = backend.matvec([[1.0, 2.0], [3.0, 4.0]], [0.5, -0.25])
    np.testing.assert_array_equal(result, np.asarray([0.0, 0.5], dtype=np.float64))
    np.testing.assert_allclose(backend.exp([0.0, 1.0]), [1.0, np.e])

    with pytest.raises(ValueError, match="finite"):
        backend.as_float64([np.nan])
    with pytest.raises(ValueError, match="shapes"):
        backend.matvec([[1.0, 2.0]], [1.0])


def test_worker_count_qualification_preserves_result_order_and_bytes() -> None:
    evidence = qualify_worker_invariance(
        lambda value: np.asarray([value, value * value], dtype=np.float64),
        tuple(range(8)),
        worker_counts=(1, 2, 4),
    )

    assert evidence.passed is True
    assert evidence.worker_counts == (1, 2, 4)
    assert len(set(evidence.result_digests)) == 1
    np.testing.assert_array_equal(evidence.reference_results[3], [3.0, 9.0])


def test_multigrid_builder_has_no_target_argument_and_is_deterministic() -> None:
    parameters = set(inspect.signature(build_stage4_integration_grid).parameters)
    assert not any(
        "target" in value or "event" in value or "epicenter" in value for value in parameters
    )

    study_area = box(105.0, 34.0, 105.05, 34.05)
    family = build_stage4_grid_family(study_area)
    repeated = build_stage4_integration_grid(study_area, cell_size_km=25.0)

    assert repeated.grid_id == family.primary_25km.grid_id
    assert repeated.cell_ids == family.primary_25km.cell_ids
    assert repeated.cell_count > 0
    assert family.coarse_50km.total_area_km2 == pytest.approx(
        family.reference_12_5km.total_area_km2,
        rel=1e-12,
        abs=1e-6,
    )


def test_feature_manifest_excludes_quality_companions_from_sources() -> None:
    bundle = load_stage4_protocol_bundle(Path.cwd())
    coverage = source_columns_for_variant(bundle, "coverage_only")
    snapshot = source_columns_for_variant(bundle, "snapshot")
    dynamic = source_columns_for_variant(bundle, "dynamic")

    assert len(coverage) == 9
    assert len(snapshot) == 17
    assert len(dynamic) == 27
    assert set(coverage) < set(snapshot) < set(dynamic)
    assert not any(
        token in name for name in dynamic for token in ("null_reason", "sample_count", "valid")
    )


def test_raw_feature_extraction_preserves_missingness_and_positive_area() -> None:
    table = pa.table(
        {
            "signal": pa.array([1.0, None, 3.0], type=pa.float64()),
            "binary": pa.array([True, False, True], type=pa.bool_()),
            "clipped_area_km2": pa.array([1.0, 2.0, 3.0], type=pa.float64()),
        }
    )
    matrix = extract_raw_feature_matrix(table, source_columns=("signal", "binary"))

    assert matrix.values.shape == (3, 2)
    assert matrix.missing.tolist() == [[False, False], [True, False], [False, False]]
    assert np.isnan(matrix.values[1, 0])
    assert matrix.values.flags.writeable is False


def test_arrow_identity_hash_includes_the_null_bitmap() -> None:
    accepted = pa.table(
        {
            "value": pa.array([1.0, None], type=pa.float64()),
            "valid": pa.array([True, False], type=pa.bool_()),
        }
    )
    same = pa.table(
        {
            "value": pa.array([1.0, None], type=pa.float64()),
            "valid": pa.array([True, False], type=pa.bool_()),
        }
    )
    changed_bitmap = pa.table(
        {
            "value": pa.array([1.0, 0.0], type=pa.float64()),
            "valid": pa.array([True, False], type=pa.bool_()),
        }
    )

    identity = assert_selected_columns_exact(
        accepted,
        same,
        columns=("value", "valid"),
    )
    assert identity == selected_table_identity_sha256(accepted, ("value", "valid"))
    with pytest.raises(ValueError, match="identity reconstruction"):
        assert_selected_columns_exact(
            accepted,
            changed_bitmap,
            columns=("value", "valid"),
        )


def test_grid_builder_rejects_unfrozen_resolution() -> None:
    with pytest.raises(ValueError, match="exactly 50, 25, or 12.5"):
        build_stage4_integration_grid(box(105.0, 34.0, 105.05, 34.05), cell_size_km=10.0)
