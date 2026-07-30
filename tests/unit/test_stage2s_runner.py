from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypeVar, cast

import numpy as np
import pytest
from shapely import box

from seismoflux.stage2s.catalog import Stage2SEarthquakeCatalog
from seismoflux.stage2s.evaluation import GateAssessment, GateDecision
from seismoflux.stage2s.production import (
    FormalExecutionServices,
    FormalPreflightContext,
    FormalScienceInputs,
    Stage2SFormalError,
    _CatalogRoleAccess,
    _formal_path_specs,
    _gate_record,
    _render_payload,
    _require_supported_fit_targets,
    _validate_formal_preflight_receipt,
    execute_formal_once,
)
from seismoflux.stage2s.protocol import Stage2SProtocolBundle
from seismoflux.stage2s.records import Stage2SWholeRunRecord
from seismoflux.stage2s.rendering import (
    DISPLAY_ALARM_BUDGETS_KM2,
    Stage2SMapFrame,
)

_T = TypeVar("_T")
_METRIC_KEYS = (
    "S1_minus_S0:IG",
    "S1_minus_S0:recall",
    "S1_minus_SP:IG",
    "S1_minus_SP:recall",
)
_PREFLIGHT_REQUIRED_BINDINGS = (
    "protocol_commit_tag_and_config_fold_input_hashes",
    "code_commit_and_tag",
    "study_area_file_projected_geometry_area_and_identity_algorithm",
    "loader_query_and_operational_grid_builder_source_hashes",
    "fold4_support_identity_and_manifest_hash",
    "cell_mapping_file_hash_schema_grid_id_15697_cells_and_39_zones",
    "aligned_12_5_25_50km_grid_ids_areas_parent_relations_and_representative_points",
    "query_grid_and_cell_zone_mapping_returned_together",
)


class _CoordinateCanary:
    def __init__(
        self,
        values: list[float],
        *,
        canary_index: int,
        allowed: Callable[[], bool],
    ) -> None:
        self._values = np.asarray(values, dtype=np.float64)
        self._canary_index = canary_index
        self._allowed = allowed
        self.accessed_indices: list[int] = []

    def __getitem__(self, key: Any) -> Any:
        selected = np.asarray(np.arange(self._values.size)[key], dtype=np.int64).reshape(-1)
        indices = [int(value) for value in selected]
        if self._canary_index in indices and not self._allowed():
            raise AssertionError("assessment coordinate canary opened before master")
        self.accessed_indices.extend(indices)
        return self._values[key]


def _record_call(calls: list[str], name: str, result: _T) -> _T:
    calls.append(name)
    return result


def _protocol(root: Path, catalog_payload: bytes) -> Stage2SProtocolBundle:
    folds = [
        {
            "fold_index": fold_index,
            "assessment_issue_dates_local_by_horizon": {
                "7": [f"202{fold_index}-01-01"],
                "30": [f"202{fold_index}-01-01"],
                "90": [f"202{fold_index}-01-01"],
            },
        }
        for fold_index in (1, 2, 3)
    ]
    return Stage2SProtocolBundle(
        repository_root=root,
        config={
            "execution_control": {
                "non_target_preflight_receipt": {
                    "path": "state/non_target_preflight.json",
                    "required_bindings": list(_PREFLIGHT_REQUIRED_BINDINGS),
                    "security_assertions": {
                        "earthquake_catalog_bytes_read": False,
                        "assessment_target_view_read": False,
                        "score_or_candidate_metric_read": False,
                    },
                },
                "attempt_ledger": {"path": "state/attempt.json"},
                "target_read_ledger": {"path": "state/target_read.json"},
            },
            "source_contracts": {
                "study_area": {
                    "projected_geometry_identity_algorithm": {
                        "normalize": "shapely.normalize",
                        "serialize": "shapely.to_wkb",
                        "digest": "SHA256",
                    },
                    "operational_quadrature_representative_point": (
                        "shapely.point_on_surface_of_exact_support_clipped_geometry"
                    ),
                },
                "cell_zone_mapping": {
                    "row_count": 1,
                    "required_nonempty_zone_count": 1,
                },
                "earthquake_catalog": {
                    "path": "forbidden-real-catalog.parquet",
                    "file_sha256": hashlib.sha256(catalog_payload).hexdigest(),
                    "content_sha256": "1" * 64,
                    "schema_sha256": "2" * 64,
                    "row_count": 1,
                },
            },
        },
        fold_manifest={"folds": folds},
        input_contract={},
        config_sha256="3" * 64,
        fold_manifest_sha256="4" * 64,
        input_contract_sha256="5" * 64,
    )


def _formal_record() -> Stage2SWholeRunRecord:
    primary_metrics = {key: 1.0 for key in _METRIC_KEYS}
    zero_metrics = {key: 0.0 for key in _METRIC_KEYS}
    component_id = "event-a"
    limited_interpretation = (
        "claim_limited_to_sequence_associated_continuation_not_broad_regional_gain"
    )
    return Stage2SWholeRunRecord(
        mode="formal_development",
        identity={"execution_role": "synthetic_injected_transaction_test"},
        input_receipts={"real_catalog_open_count": 0},
        fold_fit_summaries=[{"fold_index": fold_index} for fold_index in (1, 2, 3)],
        issue_prediction_summaries=[{"issue": "synthetic"}],
        seal_chain={"master": "synthetic"},
        cell_scores=[
            {
                "fold_index": fold_index,
                "horizon_days": horizon,
                "event_ids": [f"event-{fold_index}-{horizon}"],
                "hit_by_model": {
                    "S0": [True],
                    "S1": [True],
                    "SP": [True],
                },
            }
            for fold_index in (1, 2, 3)
            for horizon in (7, 30, 90)
        ],
        bootstrap_summary={"replications": 2_000},
        bootstrap_rows=[(0.0, 0.0, 0.0, 0.0)] * 2_000,
        regional_evidence={"synthetic": True},
        sequence_evidence={
            "component_count": 1,
            "event_resampling_unit_count": 1,
            "global_residual": zero_metrics,
            "primary_model_recall": {
                "S0": 1.0,
                "S1": 1.0,
                "SP": 1.0,
            },
            "components": [
                {
                    "component_id": component_id,
                    "event_ids": [component_id],
                    "event_count": 1,
                    "event_fraction": 1.0,
                    "origin_time_span_days": 0.0,
                    "max_pairwise_geodesic_distance_km": 0.0,
                    "contributions": primary_metrics,
                    "model_hits": {
                        "S0": {"raw": 1.0, "fraction": 1.0},
                        "S1": {"raw": 1.0, "fraction": 1.0},
                        "SP": {"raw": 1.0, "fraction": 1.0},
                    },
                    "information_gain": {
                        "S1_minus_S0": {"raw": 1.0, "fraction": 1.0},
                        "S1_minus_SP": {"raw": 1.0, "fraction": 1.0},
                    },
                }
            ],
            "largest_count_component_id": component_id,
            "largest_count_component": {"component_id": component_id},
            "largest_gain_component_id": {key: component_id for key in _METRIC_KEYS},
            "largest_gain_component": {
                key: {
                    "component_id": component_id,
                    "raw_contribution": 1.0,
                    "leave_out": 0.0,
                }
                for key in _METRIC_KEYS
            },
            "leave_largest_count_out": zero_metrics,
            "leave_largest_gain_out": zero_metrics,
            "leave_out_residual": zero_metrics,
            "claim_limited": True,
            "interpretation_limit": limited_interpretation,
        },
        descriptive_point_estimates={
            "SP_minus_S0": {
                "information_gain": 0.0,
                "recall_gain": 0.0,
                "derivation": "S1_minus_S0_minus_S1_minus_SP",
                "inferential_status": "descriptive_point_estimate_only",
                "included_in_bootstrap_ci": False,
                "included_in_gate": False,
            }
        },
        latency_evidence=[
            {"delay_days": 1},
            {"delay_days": 7},
        ],
        gate_evidence={
            "status": "failed",
            "overall_macros": primary_metrics,
            "claim_limited": True,
            "interpretation_limit": limited_interpretation,
            "interpretation_scope": "sequence_associated_continuation_only",
        },
        artifact_sha256_by_name={},
    )


def _preflight() -> FormalPreflightContext:
    return FormalPreflightContext(
        spatial=cast(Any, object()),
        support_manifest=cast(Any, object()),
        calendar=cast(Any, SimpleNamespace(folds=())),
        receipt_bindings={
            "earthquake_catalog_bytes_read": False,
            "assessment_target_view_read": False,
        },
    )


def test_injected_formal_transaction_claims_before_one_catalog_read(
    tmp_path: Path,
) -> None:
    payload = b"synthetic-catalog-bytes"
    protocol = _protocol(tmp_path, payload)
    phases: list[str] = []
    calls: list[str] = []
    catalog_reads = 0

    def read_catalog(path: Path) -> bytes:
        nonlocal catalog_reads
        catalog_reads += 1
        calls.append("catalog_read")
        assert path == tmp_path / "forbidden-real-catalog.parquet"
        assert (tmp_path / "state/attempt.json").is_file()
        assert (tmp_path / "state/target_read.json").is_file()
        return payload

    services = FormalExecutionServices(
        verify_release=lambda _protocol: _record_call(calls, "release", "a" * 40),
        run_code_acceptance=lambda _protocol: _record_call(calls, "synthetic", "b" * 64),
        audit_imports=lambda _root: "f" * 64,
        build_preflight=lambda *_args: _record_call(calls, "preflight", _preflight()),
        read_catalog_once=read_catalog,
        parse_catalog=lambda _payload: cast(Stage2SEarthquakeCatalog, object()),
        run_science=lambda _inputs, _catalog: _record_call(calls, "science", _formal_record()),
        audit_imports_release=lambda _root, _commit: _record_call(
            calls,
            "audit",
            {
                "head_commit": "a" * 40,
                "code_commit": "a" * 40,
                "visited_path_sha256": {"src/seismoflux/stage2s/production.py": "d" * 64},
                "visited_path_count": 1,
                "git_tracked": True,
                "path_scoped_status_clean": True,
                "working_tree_equals_head_and_code_commit": True,
                "evidence_sha256": "c" * 64,
            },
        ),
        execution_environment_bindings={
            "workers": 1,
            "physical_core_count": 4,
            "reserved_physical_cores_minimum": 2,
            "available_physical_cores_after_reservation": 2,
            "thread_environment": {"OMP_NUM_THREADS": "1"},
            "configured_before_numeric_imports": True,
            "evidence_sha256": "e" * 64,
        },
    )
    result = execute_formal_once(
        protocol,
        services=services,
        progress=phases.append,
    )

    assert result.mode == "formal_development"
    assert catalog_reads == 1
    assert calls == [
        "release",
        "synthetic",
        "audit",
        "preflight",
        "catalog_read",
        "science",
    ]
    assert phases[:5] == [
        "remote_tags_verified",
        "non_target_preflight_passed",
        "attempt_claimed",
        "target_read_claimed",
        "catalog_parsed",
    ]
    assert phases[-1] == "result_written"
    result_path = (
        tmp_path / "data/interim/stage2s/causal_seismicity_screen/stage2s_whole_run_record.json"
    )
    assert result_path.read_bytes() == result.to_canonical_bytes()
    preflight = json.loads((tmp_path / "state/non_target_preflight.json").read_text("utf-8"))
    bindings = preflight["bindings"]
    assert bindings["stage2s_import_closure_release"]["visited_path_count"] == 1
    assert bindings["formal_execution_environment"]["workers"] == 1
    absence = bindings["initial_formal_output_absence"]
    expected_paths = [spec.relative_path.as_posix() for spec in _formal_path_specs(protocol)]
    assert absence["relative_paths"] == expected_paths
    assert absence["path_count"] == len(expected_paths)
    assert absence["all_absent"] is True

    with pytest.raises(Stage2SFormalError, match="completed Stage2S result already exists"):
        execute_formal_once(protocol, services=services, progress=None)
    assert catalog_reads == 1


def test_injected_evidence_failure_writes_terminal_record_without_retry(
    tmp_path: Path,
) -> None:
    payload = b"synthetic-catalog-bytes"
    protocol = _protocol(tmp_path, payload)
    catalog_reads = 0

    def read_catalog(_path: Path) -> bytes:
        nonlocal catalog_reads
        catalog_reads += 1
        return payload

    def insufficient(
        _inputs: object,
        _catalog: Stage2SEarthquakeCatalog,
    ) -> Stage2SWholeRunRecord:
        _require_supported_fit_targets(0)
        raise AssertionError("zero supported fit targets must be terminal")

    services = FormalExecutionServices(
        verify_release=lambda _protocol: "a" * 40,
        run_code_acceptance=lambda _protocol: "b" * 64,
        audit_imports=lambda _root: "c" * 64,
        build_preflight=lambda *_args: _preflight(),
        read_catalog_once=read_catalog,
        parse_catalog=lambda _payload: cast(Stage2SEarthquakeCatalog, object()),
        run_science=insufficient,
    )
    with pytest.raises(Stage2SFormalError, match="terminal record"):
        execute_formal_once(protocol, services=services, progress=None)

    assert catalog_reads == 1
    terminal_path = (
        tmp_path / "data/interim/stage2s/causal_seismicity_screen/terminal_failure_record.json"
    )
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    assert terminal["bindings"]["terminal_status"] == "evidence_insufficient"
    assert terminal["bindings"]["exception_message"] == (
        "formal evidence was insufficient under the frozen protocol"
    )
    assert terminal["bindings"]["attempt_consumed"] is True
    assert terminal["bindings"]["retry_or_rollback_authorized"] is False
    assert not (
        tmp_path / "data/interim/stage2s/causal_seismicity_screen/stage2s_whole_run_record.json"
    ).exists()

    with pytest.raises(Stage2SFormalError, match="terminal record already exists"):
        execute_formal_once(protocol, services=services, progress=None)
    assert catalog_reads == 1


def test_every_existing_formal_sink_fails_before_preflight_or_catalog_open(
    tmp_path: Path,
) -> None:
    def make_forbidden_preflight(
        counter: dict[str, int],
    ) -> Callable[..., FormalPreflightContext]:
        def forbidden_preflight(*_args: object) -> FormalPreflightContext:
            counter["preflight"] += 1
            raise AssertionError("old sink must stop before study-area preflight")

        return forbidden_preflight

    def make_forbidden_catalog(
        counter: dict[str, int],
    ) -> Callable[[Path], bytes]:
        def forbidden_catalog(_path: Path) -> bytes:
            counter["catalog"] += 1
            raise AssertionError("old sink must stop before catalog open")

        return forbidden_catalog

    payload = b"synthetic-catalog-bytes"
    template_protocol = _protocol(tmp_path / "template", payload)
    labels = tuple(spec.label for spec in _formal_path_specs(template_protocol))

    for label in labels:
        root = tmp_path / label.replace("/", "_")
        root.mkdir()
        protocol = _protocol(root, payload)
        spec = next(item for item in _formal_path_specs(protocol) if item.label == label)
        sink = root / spec.relative_path
        sink.parent.mkdir(parents=True, exist_ok=True)
        sink.write_bytes(b"partial-old-state")
        calls = {"preflight": 0, "catalog": 0}

        services = FormalExecutionServices(
            verify_release=lambda _protocol: "a" * 40,
            run_code_acceptance=lambda _protocol: "b" * 64,
            audit_imports=lambda _root: "c" * 64,
            build_preflight=make_forbidden_preflight(calls),
            read_catalog_once=make_forbidden_catalog(calls),
            parse_catalog=lambda _payload: cast(Stage2SEarthquakeCatalog, object()),
            run_science=lambda _inputs, _catalog: _formal_record(),
        )
        with pytest.raises(Stage2SFormalError):
            execute_formal_once(protocol, services=services, progress=None)

        assert calls == {"preflight": 0, "catalog": 0}
        terminal = (
            root / "data/interim/stage2s/causal_seismicity_screen/terminal_failure_record.json"
        )
        if label == "terminal":
            assert terminal.read_bytes() == b"partial-old-state"
        else:
            terminal_payload = json.loads(terminal.read_text(encoding="utf-8"))
            assert terminal_payload["bindings"]["terminal_status"] == "interrupted"
            if label == "attempt":
                assert (
                    terminal_payload["bindings"]["seal_status_by_label"]["attempt"]["status"]
                    == "invalid"
                )
            if label.startswith("artifact_"):
                assert (
                    terminal_payload["bindings"]["artifact_status_by_name"][
                        spec.relative_path.name
                    ]["status"]
                    == "present_unverified"
                )


def test_keyboard_interrupt_is_terminal_and_restart_never_reopens_catalog(
    tmp_path: Path,
) -> None:
    payload = b"synthetic-catalog-bytes"
    protocol = _protocol(tmp_path, payload)
    catalog_reads = 0

    def read_catalog(_path: Path) -> bytes:
        nonlocal catalog_reads
        catalog_reads += 1
        return payload

    def interrupted(
        _inputs: FormalScienceInputs,
        _catalog: Stage2SEarthquakeCatalog,
    ) -> Stage2SWholeRunRecord:
        raise KeyboardInterrupt

    services = FormalExecutionServices(
        verify_release=lambda _protocol: "a" * 40,
        run_code_acceptance=lambda _protocol: "b" * 64,
        audit_imports=lambda _root: "c" * 64,
        build_preflight=lambda *_args: _preflight(),
        read_catalog_once=read_catalog,
        parse_catalog=lambda _payload: cast(Stage2SEarthquakeCatalog, object()),
        run_science=interrupted,
    )
    with pytest.raises(KeyboardInterrupt):
        execute_formal_once(protocol, services=services, progress=None)
    assert catalog_reads == 1

    terminal_path = (
        tmp_path / "data/interim/stage2s/causal_seismicity_screen/terminal_failure_record.json"
    )
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    assert terminal["bindings"]["terminal_status"] == "interrupted"
    assert terminal["bindings"]["exception_category"] == "keyboard_interrupt"
    assert terminal["bindings"]["seal_status_by_label"]["attempt"]["status"] == "valid"
    assert terminal["bindings"]["seal_status_by_label"]["target_read"]["status"] == "valid"

    with pytest.raises(Stage2SFormalError, match="terminal record already exists"):
        execute_formal_once(protocol, services=services, progress=None)
    assert catalog_reads == 1


def test_terminal_exception_message_suppresses_target_text(tmp_path: Path) -> None:
    payload = b"synthetic-catalog-bytes"
    protocol = _protocol(tmp_path, payload)
    secret = "SECRET_TARGET_EVENT_earthquake-42_at_101.5_23.5"

    def fail(
        _inputs: FormalScienceInputs,
        _catalog: Stage2SEarthquakeCatalog,
    ) -> Stage2SWholeRunRecord:
        raise RuntimeError(secret)

    services = FormalExecutionServices(
        verify_release=lambda _protocol: "a" * 40,
        run_code_acceptance=lambda _protocol: "b" * 64,
        audit_imports=lambda _root: "c" * 64,
        build_preflight=lambda *_args: _preflight(),
        read_catalog_once=lambda _path: payload,
        parse_catalog=lambda _payload: cast(Stage2SEarthquakeCatalog, object()),
        run_science=fail,
    )
    with pytest.raises(Stage2SFormalError, match="terminal record"):
        execute_formal_once(protocol, services=services, progress=None)

    terminal_bytes = (
        tmp_path / "data/interim/stage2s/causal_seismicity_screen/terminal_failure_record.json"
    ).read_bytes()
    assert secret.encode("utf-8") not in terminal_bytes
    terminal = json.loads(terminal_bytes)
    assert terminal["bindings"]["terminal_status"] == "invalid"
    assert terminal["bindings"]["exception_category"] == "exception"


@pytest.mark.parametrize(
    ("status", "expected_scope"),
    [
        ("passed_development_signal", "broad_regional_gain_not_sequence_limited"),
        ("failed", "no_sequence_interpretation_limit"),
        ("evidence_insufficient", "no_sequence_interpretation_limit"),
        ("invalid", "no_sequence_interpretation_limit"),
    ],
)
def test_unlimited_sequence_scope_is_positive_only_after_full_gate_pass(
    status: str,
    expected_scope: str,
) -> None:
    assessment = GateAssessment(
        decision=GateDecision(status=cast(Any, status), reasons=()),
        supported_event_union_count=20,
        recall_event_union_count=20,
        fold_macros={},
        horizon_macros={},
        overall_macros={},
        claim_limited=False,
        interpretation_limit="no_sequence_interpretation_limit",
    )

    assert _gate_record(assessment)["interpretation_scope"] == expected_scope


def test_formal_preflight_receipt_rejects_incomplete_spatial_bindings(
    tmp_path: Path,
) -> None:
    protocol = _protocol(tmp_path, b"catalog")
    layers = {
        layer: {
            "grid_id": character * 64,
        }
        for layer, character in (("50", "5"), ("25", "2"), ("12.5", "1"))
    }
    bindings: dict[str, object] = {
        **protocol.identity_mapping(),
        "code_commit": "a" * 40,
        "code_tag": "v0.2.3-causal-seismicity-screen-code",
        "source_sha256": {
            "projected_geometry_identity_source": "1" * 64,
            "study_area_loader_source": "2" * 64,
            "query_grid_builder_source": "3" * 64,
            "grid_primitives_source": "4" * 64,
        },
        "projected_geometry_identity_algorithm": {
            "normalize": "shapely.normalize",
            "serialize": "shapely.to_wkb",
            "digest": "SHA256",
        },
        "operational_quadrature_representative_point": (
            "shapely.point_on_surface_of_exact_support_clipped_geometry"
        ),
        "query_grid_id": "2" * 64,
        "query_grid_cell_count": 1,
        "construction_zone_ids": ["zone-1"],
        "aligned_grid_identity": {
            "layer_order": ["50", "25", "12.5"],
            "layers": layers,
            "parent_relation_order": ["25_to_50", "12.5_to_25"],
            "parent_relations": {
                "25_to_50": {},
                "12.5_to_25": {},
            },
            "identity_sha256": "f" * 64,
            "target_or_score_input_count": 0,
        },
        "query_grid_and_cell_zone_mapping_returned_together": True,
        "aligned_grid_parent_relations_verified": True,
        "representative_point_identities_verified": True,
        "earthquake_catalog_bytes_read": False,
        "assessment_target_view_read": False,
        "score_or_candidate_metric_read": False,
    }
    _validate_formal_preflight_receipt(protocol, bindings)

    missing_layer = deepcopy(bindings)
    cast(dict[str, object], missing_layer["aligned_grid_identity"])["layer_order"] = [
        "50",
        "25",
    ]
    with pytest.raises(Stage2SFormalError, match="aligned grid identity"):
        _validate_formal_preflight_receipt(protocol, missing_layer)

    wrong_algorithm = deepcopy(bindings)
    wrong_algorithm["operational_quadrature_representative_point"] = "centroid"
    with pytest.raises(Stage2SFormalError, match="grid/mapping relationship"):
        _validate_formal_preflight_receipt(protocol, wrong_algorithm)

    opened_target = deepcopy(bindings)
    opened_target["assessment_target_view_read"] = True
    with pytest.raises(Stage2SFormalError, match="security assertions"):
        _validate_formal_preflight_receipt(protocol, opened_target)


def test_render_payload_keeps_every_fold_issue_horizon_model_group() -> None:
    groups = (
        (3, 90, "2025-03-01T00:00:00+00:00"),
        (1, 7, "2023-01-01T00:00:00+00:00"),
        (2, 30, "2024-02-01T00:00:00+00:00"),
    )
    frames = tuple(
        Stage2SMapFrame(
            issue_time_utc=issue_time,
            data_cutoff_utc=issue_time,
            fold_index=fold_index,
            horizon_days=horizon_days,
            model_id=cast(Any, model_id),
            relative_intensity_rank=((0.0, 1.0), (None, 0.5)),
            study_area_km2=((625.0, 625.0), (None, 625.0)),
            alarm_area_fraction_by_budget_km2={
                str(budget): ((1.0, 1.0), (None, 1.0)) for budget in DISPLAY_ALARM_BUDGETS_KM2
            },
            actual_alarm_area_km2_by_budget={
                str(budget): 1_875.0 for budget in DISPLAY_ALARM_BUDGETS_KM2
            },
        )
        for fold_index, horizon_days, issue_time in groups
        for model_id in ("S0", "S1", "SP")
    )
    inputs = cast(
        Any,
        SimpleNamespace(
            protocol=SimpleNamespace(
                config={"long_term_background": {"fit_end_utc": "2022-12-31T23:59:59+00:00"}}
            )
        ),
    )

    payload = _render_payload(
        inputs=inputs,
        record=_formal_record(),
        map_frames=frames,
    )

    observed = {
        (frame.fold_index, frame.horizon_days, frame.issue_time_utc) for frame in payload.map_frames
    }
    assert observed == set(groups)
    for group in observed:
        assert {
            frame.model_id
            for frame in payload.map_frames
            if (frame.fold_index, frame.horizon_days, frame.issue_time_utc) == group
        } == {"S0", "S1", "SP"}


def test_role_gateway_defers_assessment_coordinate_canary_until_master() -> None:
    session = SimpleNamespace(master_record=None)
    master = object()

    def allowed() -> bool:
        return session.master_record is master

    longitude = _CoordinateCanary(
        [100.0, 101.0, 102.0, 103.0],
        canary_index=3,
        allowed=allowed,
    )
    latitude = _CoordinateCanary(
        [20.0, 21.0, 22.0, 23.0],
        canary_index=3,
        allowed=allowed,
    )
    origins = np.asarray(
        [
            int(
                (
                    datetime(year, 1, 1, tzinfo=UTC) - datetime(1970, 1, 1, tzinfo=UTC)
                ).total_seconds()
                * 1_000_000
            )
            for year in (2018, 2020, 2021, 2022)
        ],
        dtype=np.int64,
    )
    catalog = cast(
        Any,
        SimpleNamespace(
            row_count=4,
            event_ids=("support", "fit", "source", "assessment"),
            origin_time_us=origins,
            available_at_us=origins.copy(),
            longitude=longitude,
            latitude=latitude,
            magnitude=np.asarray([3.0, 5.2, 4.3, 5.4]),
            inside_study_area=np.asarray([True, True, True, True]),
        ),
    )
    preflight = cast(
        Any,
        SimpleNamespace(
            adapter=SimpleNamespace(
                grid_family=SimpleNamespace(
                    study_area_equal_area=box(
                        -20_000_000.0,
                        -20_000_000.0,
                        20_000_000.0,
                        20_000_000.0,
                    )
                ),
                query_grid=SimpleNamespace(equal_area_crs="EPSG:3857"),
            )
        ),
    )
    access = _CatalogRoleAccess(catalog, preflight)

    support = access.open_before(
        role_id="support",
        cutoff_utc=datetime(2018, 12, 31, tzinfo=UTC),
    )
    fit = access.open_before(
        role_id="fit",
        cutoff_utc=datetime(2020, 12, 31, tzinfo=UTC),
    )
    source = access.open_before(
        role_id="source",
        cutoff_utc=datetime(2021, 12, 31, tzinfo=UTC),
    )
    assert support.indices.tolist() == [0]
    assert fit.indices.tolist() == [0, 1]
    assert source.indices.tolist() == [0, 1, 2]
    assert 3 not in longitude.accessed_indices
    assert 3 not in latitude.accessed_indices

    with pytest.raises(Stage2SFormalError, match="master seal"):
        access.open_assessment(
            session=cast(Any, session),
            master_seal=cast(Any, master),
            target_indices=np.asarray([3], dtype=np.int64),
        )
    assert 3 not in longitude.accessed_indices

    session.master_record = master
    assessment = access.open_assessment(
        session=cast(Any, session),
        master_seal=cast(Any, master),
        target_indices=np.asarray([3], dtype=np.int64),
    )
    assert assessment.indices.tolist() == [3]
    assert longitude.accessed_indices[-1] == 3
    assert latitude.accessed_indices[-1] == 3
