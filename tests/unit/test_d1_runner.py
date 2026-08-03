from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, Literal, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import seismoflux.d1_replay.runner as runner_module
from seismoflux.d1_replay.evaluation import D1_MODEL_ORDER
from seismoflux.d1_replay.features import (
    D1_SOURCE_COLUMNS,
    D1IssueFeatures,
    D1StaticGrid,
    load_d1_feature_contract,
)
from seismoflux.d1_replay.protocol import sha256_file
from seismoflux.d1_replay.runner import (
    D1FoldModelResult,
    D1ObservedClusterOutcome,
    D1ObservedReplayResult,
    D1ParameterSelection,
    D1PreparedReplay,
    D1TrainingDiagnostic,
    _assert_resource_boundary,
    _choose_alpha,
    _choose_ridge,
    _fit_inner_preprocessors,
    _load_or_initialize_state,
    _make_issue_model_score,
    _read_unit_checkpoint,
    _target_counts_by_issue,
    _training_prefix_indices,
    _unit_paths,
    _write_unit_checkpoint,
    evaluate_d1_feature_variant_fold,
)
from seismoflux.d1_replay.targets import TargetSet
from seismoflux.stage2s.contracts import SpatialGrid

ROOT = Path(__file__).resolve().parents[2]


def _static_grid(cell_count: int = 3) -> D1StaticGrid:
    return D1StaticGrid(
        grid_id="synthetic-grid",
        cell_ids=tuple(f"cell-{index}" for index in range(cell_count)),
        rows=np.arange(cell_count, dtype=np.int64),
        columns=np.zeros(cell_count, dtype=np.int64),
        query_x_m=np.zeros(cell_count, dtype=np.float64),
        query_y_m=np.arange(cell_count, dtype=np.float64) * 25_000.0,
        clipped_area_km2=np.full(cell_count, 200_000.0, dtype=np.float64),
    )


def _issue(index: int, *, future_multiplier: float = 1.0) -> D1IssueFeatures:
    grid = _static_grid(3)
    values = np.empty((grid.cell_count, len(D1_SOURCE_COLUMNS)), dtype=np.float64)
    for column in range(len(D1_SOURCE_COLUMNS)):
        values[:, column] = (
            1.0 + column + index + future_multiplier * np.arange(grid.cell_count, dtype=np.float64)
        )
    return D1IssueFeatures(
        issue_time_utc=datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=35 * index),
        issue_report_id=f"issue-{index}",
        grid=grid,
        source_columns=D1_SOURCE_COLUMNS,
        values=values,
        null_mask=np.zeros(values.shape, dtype=np.bool_),
    )


def _operational_grid() -> SpatialGrid:
    return SpatialGrid(
        grid_id="synthetic-grid",
        cell_size_km=25.0,
        cell_ids=("cell-0", "cell-1", "cell-2"),
        rows=np.asarray([0, 1, 2], dtype=np.int64),
        columns=np.asarray([0, 0, 0], dtype=np.int64),
        query_xy_km=np.asarray([[0.0, 0.0], [0.0, 25.0], [0.0, 50.0]]),
        clipped_area_km2=np.asarray([200_000.0, 200_000.0, 200_000.0]),
    )


def _selection(parameter: str = "alpha") -> D1ParameterSelection:
    return D1ParameterSelection(
        parameter=cast(Literal["alpha", "ridge"], parameter),
        selected_value=0.0 if parameter == "alpha" else 10.0,
        status="evidence_insufficient_for_tuning",
        validation_issue_times_us=(),
        training_prefix_issue_times_us=(),
        validation_event_count=0,
        mean_log_density_by_candidate={},
    )


def _fold_model(fold_id: str, model_id: str) -> D1FoldModelResult:
    feature_groups = () if model_id in {"B0", "B0_R30"} else ("C1",)
    base = "R30" if model_id in {"B0_R30", "B0_R30_C_A_dynamic"} else "B0"
    return D1FoldModelResult(
        fold_id=fold_id,
        model_id=model_id,
        base=cast(Literal["B0", "R30"], base),
        feature_groups=feature_groups,
        alpha_selection=_selection(),
        ridge_selection=None if not feature_groups else _selection("ridge"),
        training_diagnostic=D1TrainingDiagnostic(
            fit_issue_count=5,
            fit_catalog_m4plus_event_count=10,
            coefficient_names=() if not feature_groups else ("C1",),
            active_coefficients=() if not feature_groups else (True,),
            coefficients=() if not feature_groups else (0.1,),
            objective=None if not feature_groups else 1.0,
            iteration_count=0 if not feature_groups else 2,
        ),
    )


def test_inner_validation_uses_only_strictly_earlier_issues() -> None:
    prefixes = _training_prefix_indices(8)
    assert tuple(index for index, _ in prefixes) == (5, 6, 7)
    assert prefixes[0][1] == (0, 1, 2, 3, 4)
    assert all(all(training < validation for training in prefix) for validation, prefix in prefixes)


def test_alpha_and_ridge_ties_apply_preregistered_opposite_rules() -> None:
    alpha_scores = {0.0: -2.0, 0.25: -2.0 + 5.0e-13, 0.5: -3.0, 0.75: -4.0}
    ridge_scores = {0.1: -2.0, 1.0: -2.0 + 5.0e-13, 10.0: -2.0}
    assert _choose_alpha(alpha_scores) == 0.0
    assert _choose_ridge(ridge_scores) == 10.0


def test_worker_boundary_uses_physical_cores_reserves_two_and_caps_four(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner_module, "detect_physical_core_count", lambda: 8)
    boundary = _assert_resource_boundary(4)
    assert boundary.physical_cores == 8
    assert boundary.max_workers_after_reservation == 4
    with pytest.raises(ValueError, match="maximum of 4"):
        _assert_resource_boundary(5)

    monkeypatch.setattr(runner_module, "detect_physical_core_count", lambda: 5)
    with pytest.raises(ValueError, match="physical CPU cores"):
        _assert_resource_boundary(4)

    monkeypatch.setattr(runner_module, "detect_physical_core_count", lambda: None)
    with pytest.raises(RuntimeError, match="verified physical cores"):
        _assert_resource_boundary(1)


def test_later_fit_change_cannot_affect_earlier_inner_preprocessor() -> None:
    contract = load_d1_feature_contract(ROOT / "configs/d1_retrospective_development.yaml")
    original = tuple(_issue(index) for index in range(6))
    changed = (*original[:4], _issue(4, future_multiplier=10_000.0), original[5])
    issue_times = tuple(int(item.issue_time_utc.timestamp() * 1_000_000) for item in original)
    original_by_time = dict(zip(issue_times, original, strict=True))
    changed_by_time = dict(zip(issue_times, changed, strict=True))
    original_fits = dict(_fit_inner_preprocessors(contract, ("C1",), issue_times, original_by_time))
    changed_fits = dict(_fit_inner_preprocessors(contract, ("C1",), issue_times, changed_by_time))
    first_validation = _training_prefix_indices(len(issue_times))[0][0]
    assert original_fits[first_validation].source_fits == changed_fits[first_validation].source_fits
    assert original_fits[first_validation].fitted_issue_times_utc == tuple(
        item.issue_time_utc for item in original[:first_validation]
    )


def test_target_assignment_and_all_six_model_frames_are_complete() -> None:
    issue_time = int(datetime(2020, 1, 1, tzinfo=UTC).timestamp() * 1_000_000)
    targets = TargetSet(
        role="fit",
        fold_id="fold_1",
        horizon_days=30,
        issue_times_us=(issue_time,),
        event_indices=(0, 1, 2),
        assigned_issue_times_us=(issue_time, issue_time, issue_time),
        event_identity_set_sha256="synthetic",
        late_available_target_count=0,
    )

    class Locator:
        def locate_lonlat(self, longitude: float, latitude: float) -> int | None:
            del latitude
            return int(longitude)

    catalog = SimpleNamespace(
        longitude=np.asarray([0.0, 1.0, 1.0]),
        latitude=np.zeros(3),
    )
    domain = SimpleNamespace(
        operational_grid=SimpleNamespace(cell_count=3),
        locator=Locator(),
    )
    counts = _target_counts_by_issue(targets, cast(Any, catalog), cast(Any, domain))[issue_time]
    assert counts.tolist() == [1.0, 2.0, 0.0]

    model_ids = (
        "B0",
        "B0_R30",
        "B0_C",
        "B0_C_A_snapshot",
        "B0_C_A_dynamic",
        "B0_R30_C_A_dynamic",
    )
    issue = _issue(0)
    grid = _operational_grid()
    frames = tuple(
        _make_issue_model_score(
            fold_id="fold_1",
            issue=issue,
            horizon_days=30,
            model_id=model_id,
            mass=np.asarray([0.6, 0.3, 0.1]),
            grid=grid,
        )
        for model_id in model_ids
    )
    assert tuple(item.model_id for item in frames) == model_ids
    assert all(item.ranking_indices.tolist() == [0, 1, 2] for item in frames)
    assert all(len(item.alarm_prefixes) == 5 for item in frames)
    outcomes = tuple(
        D1ObservedClusterOutcome(
            cluster_id="cluster-1",
            fold_id="fold_1",
            issue_id=frame.issue_id,
            issue_time_us=frame.issue_time_us,
            horizon_days=30,
            model_id=frame.model_id,
            representative_cell_index=1,
            outside_support=False,
            log_density=float(np.log(frame.cell_mass[1] / 200_000.0)),
            hit_by_area=tuple(
                bool(np.any(prefix.selected_indices == 1)) for prefix in frame.alarm_prefixes
            ),
        )
        for frame in frames
    )
    assert tuple(item.evaluation_outcome().model_id for item in outcomes) == model_ids


def test_complete_result_freezes_198_frames_six_models_and_21_22_cluster_support() -> None:
    grid = _operational_grid()
    issue_counts = {30: 8, 90: 3}
    issues = []
    frame_lookup = {}
    issue_index = 0
    issue_ids_by_axis: dict[tuple[str, int], list[str]] = {}
    for fold_id in ("fold_1", "fold_2", "fold_3"):
        for horizon, count in issue_counts.items():
            issue_ids_by_axis[(fold_id, horizon)] = []
            for _ in range(count):
                issue = _issue(issue_index)
                issue_index += 1
                issue_ids_by_axis[(fold_id, horizon)].append(issue.issue_report_id)
                for model_id in D1_MODEL_ORDER:
                    score = _make_issue_model_score(
                        fold_id=fold_id,
                        issue=issue,
                        horizon_days=horizon,
                        model_id=model_id,
                        mass=np.asarray([0.6, 0.3, 0.1]),
                        grid=grid,
                    )
                    issues.append(score)
                    frame_lookup[(fold_id, horizon, issue.issue_report_id, model_id)] = score

    support_counts = {
        30: {"fold_1": 8, "fold_2": 6, "fold_3": 7},
        90: {"fold_1": 8, "fold_2": 6, "fold_3": 8},
    }
    support: dict[int, tuple[tuple[str, str, str], ...]] = {}
    outcomes = []
    for horizon, by_fold in support_counts.items():
        triples = []
        for fold_id, cluster_count in by_fold.items():
            issue_ids = issue_ids_by_axis[(fold_id, horizon)]
            for cluster_index in range(cluster_count):
                issue_id = issue_ids[cluster_index % len(issue_ids)]
                cluster_id = f"cluster-{horizon}-{fold_id}-{cluster_index}"
                triples.append((cluster_id, fold_id, issue_id))
                for model_id in D1_MODEL_ORDER:
                    frame = frame_lookup[(fold_id, horizon, issue_id, model_id)]
                    outcomes.append(
                        D1ObservedClusterOutcome(
                            cluster_id=cluster_id,
                            fold_id=fold_id,
                            issue_id=issue_id,
                            issue_time_us=frame.issue_time_us,
                            horizon_days=horizon,
                            model_id=model_id,
                            representative_cell_index=0,
                            outside_support=False,
                            log_density=float(np.log(frame.cell_mass[0] / 200_000.0)),
                            hit_by_area=tuple(
                                bool(np.any(prefix.selected_indices == 0))
                                for prefix in frame.alarm_prefixes
                            ),
                        )
                    )
        support[horizon] = tuple(sorted(triples))

    result = D1ObservedReplayResult(
        protocol_version="d1.0.0",
        identities={"synthetic": True},
        workers=1,
        expected_support_by_horizon=support,
        fold_models=tuple(
            _fold_model(fold_id, model_id)
            for fold_id in ("fold_1", "fold_2", "fold_3")
            for model_id in D1_MODEL_ORDER
        ),
        outcomes=tuple(outcomes),
        issues=tuple(issues),
    )
    assert len(result.issues) == 198
    assert {horizon: len(items) for horizon, items in support.items()} == {30: 21, 90: 22}
    assert len(result.fold_models) == 18


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("contract_sha256", "d" * 64),
        ("input_sha256", "e" * 64),
        ("git_commit", "f" * 40),
    ),
)
def test_checkpoint_refuses_contract_input_or_git_identity_drift(
    tmp_path: Path,
    field: str,
    replacement: str,
) -> None:
    identities = {
        "contract_sha256": "a" * 64,
        "input_sha256": "b" * 64,
        "git_commit": "c" * 40,
    }
    state = _load_or_initialize_state(tmp_path, identities, workers=1)
    assert state["status"] == "prepared"
    changed = dict(identities)
    changed[field] = replacement
    with pytest.raises(ValueError, match=rf"{field} changed"):
        _load_or_initialize_state(tmp_path, changed, workers=1)


def test_checkpoint_resume_roundtrip_is_equivalent_and_rejects_rebound_ranking(
    tmp_path: Path,
) -> None:
    identities = {
        "contract_sha256": "a" * 64,
        "input_sha256": "b" * 64,
        "git_commit": "c" * 40,
    }
    grid = _operational_grid()
    issue = _issue(0)
    score = _make_issue_model_score(
        fold_id="fold_1",
        issue=issue,
        horizon_days=30,
        model_id="B0",
        mass=np.asarray([0.6, 0.3, 0.1]),
        grid=grid,
    )
    outcome = D1ObservedClusterOutcome(
        cluster_id="cluster-1",
        fold_id="fold_1",
        issue_id=score.issue_id,
        issue_time_us=score.issue_time_us,
        horizon_days=30,
        model_id="B0",
        representative_cell_index=0,
        outside_support=False,
        log_density=float(np.log(score.cell_mass[0] / 200_000.0)),
        hit_by_area=tuple(
            bool(np.any(prefix.selected_indices == 0)) for prefix in score.alarm_prefixes
        ),
    )
    fold_model = _fold_model("fold_1", "B0")
    _write_unit_checkpoint(tmp_path, identities, fold_model, (score,), (outcome,), grid)
    recovered = _read_unit_checkpoint(tmp_path, identities, "fold_1", "B0", grid)
    assert recovered[0].as_mapping() == fold_model.as_mapping()
    assert [item.as_mapping() for item in recovered[1]] == [score.as_mapping()]
    assert [item.as_mapping() for item in recovered[2]] == [outcome.as_mapping()]

    metadata_path, cell_path = _unit_paths(tmp_path, "fold_1", "B0")
    table = pq.read_table(cell_path)
    changed_mass = table["relative_cell_mass"].combine_chunks().to_pylist()
    changed_mass[0], changed_mass[-1] = changed_mass[-1], changed_mass[0]
    table = table.set_column(
        table.schema.get_field_index("relative_cell_mass"),
        "relative_cell_mass",
        pa.array(changed_mass, type=pa.float64()),
    )
    pq.write_table(table, cell_path, row_group_size=table.num_rows)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["cell_scores"]["file_sha256"] = sha256_file(cell_path)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="mass/area ranking"):
        _read_unit_checkpoint(tmp_path, identities, "fold_1", "B0", grid)


def test_checkpoint_resume_refuses_worker_plan_drift(tmp_path: Path) -> None:
    identities = {
        "contract_sha256": "a" * 64,
        "input_sha256": "b" * 64,
        "git_commit": "c" * 40,
    }
    _load_or_initialize_state(tmp_path, identities, workers=1)
    with pytest.raises(ValueError, match="resource plan changed"):
        _load_or_initialize_state(tmp_path, identities, workers=2)


def test_each_placebo_callback_reselects_parameters_and_refits_all_attribution_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    issue = _issue(0)
    issue_time = int(issue.issue_time_utc.timestamp() * 1_000_000)
    specs = (
        {"id": "B0", "base": "B0", "feature_groups": []},
        {"id": "B0_R30", "base": "R30", "feature_groups": []},
        {"id": "B0_C", "base": "B0", "feature_groups": ["C1"]},
        {
            "id": "B0_C_A_snapshot",
            "base": "B0",
            "feature_groups": ["C1", "S1"],
        },
        {
            "id": "B0_C_A_dynamic",
            "base": "B0",
            "feature_groups": ["C1", "D1"],
        },
        {
            "id": "B0_R30_C_A_dynamic",
            "base": "R30",
            "feature_groups": ["C1", "D1"],
        },
    )

    class TargetLayer:
        def fit_for(self, fold_id: str) -> SimpleNamespace:
            assert fold_id == "fold_1"
            return SimpleNamespace(issue_times_us=(issue_time,))

        def assessment_for(self, fold_id: str, horizon: int) -> SimpleNamespace:
            assert fold_id == "fold_1"
            assert horizon in {30, 90}
            return SimpleNamespace(issue_times_us=(issue_time,))

        def active_clusters(self, fold_id: str, horizon: int) -> tuple[SimpleNamespace, ...]:
            assert fold_id == "fold_1"
            return (SimpleNamespace(identity_sha256="cluster-1"),) if horizon == 30 else ()

    prepared = D1PreparedReplay(
        protocol=SimpleNamespace(config={"models": list(specs)}),  # type: ignore[arg-type]
        identities=MappingProxyType({}),
        catalog=SimpleNamespace(),  # type: ignore[arg-type]
        target_layer=TargetLayer(),  # type: ignore[arg-type]
        domain=SimpleNamespace(operational_grid=_operational_grid()),  # type: ignore[arg-type]
        feature_contract=SimpleNamespace(),  # type: ignore[arg-type]
        features_by_issue=MappingProxyType({issue_time: issue}),
        backgrounds=MappingProxyType({}),
        counts_by_fold=MappingProxyType({"fold_1": MappingProxyType({})}),
    )
    calls = {"alpha": 0, "ridge": 0, "fit": 0, "score": 0}

    @contextmanager
    def no_runtime_threads() -> Iterator[None]:
        yield

    def select_alpha(*args: object, **kwargs: object) -> D1ParameterSelection:
        del args, kwargs
        calls["alpha"] += 1
        return _selection()

    def select_ridge(*args: object, **kwargs: object) -> D1ParameterSelection:
        del args, kwargs
        calls["ridge"] += 1
        return _selection("ridge")

    def fit_outer(*args: object, **kwargs: object) -> tuple[SimpleNamespace, None, None]:
        del args
        calls["fit"] += 1
        model_spec = kwargs["model_spec"]
        assert isinstance(model_spec, dict)
        return SimpleNamespace(model_id=model_spec["id"]), None, None

    def score_unit(*args: object, **kwargs: object) -> tuple[tuple[()], tuple[object, ...]]:
        del args
        calls["score"] += 1
        fold_model = cast(SimpleNamespace, kwargs["fold_model"])
        model_id = str(fold_model.model_id)
        hit = model_id != "B0_C"
        return (), (
            D1ObservedClusterOutcome(
                cluster_id="cluster-1",
                fold_id="fold_1",
                issue_id=issue.issue_report_id,
                issue_time_us=issue_time,
                horizon_days=30,
                model_id=model_id,
                representative_cell_index=0,
                outside_support=False,
                log_density=-1.0,
                hit_by_area=(False, False, hit, hit, hit),
            ),
        )

    monkeypatch.setattr(runner_module, "_single_thread_math_runtime", no_runtime_threads)
    monkeypatch.setattr(runner_module, "_select_alpha", select_alpha)
    monkeypatch.setattr(runner_module, "_select_ridge", select_ridge)
    monkeypatch.setattr(runner_module, "_fit_outer_model", fit_outer)
    monkeypatch.setattr(runner_module, "_score_model_unit", score_unit)

    for _ in range(2):
        result = evaluate_d1_feature_variant_fold(
            prepared,
            "fold_1",
            MappingProxyType({issue_time: issue}),
        )
        assert result.alpha_reselected is True
        assert result.ridge_reselected is True
        assert result.models_refit is True
    assert calls == {"alpha": 2, "ridge": 6, "fit": 6, "score": 6}
