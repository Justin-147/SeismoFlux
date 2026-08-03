from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from seismoflux.d1_replay.evaluation import (
    D1_AREA_BUDGETS_KM2,
    D1_FOLD_IDS,
    D1_MODEL_ORDER,
)
from seismoflux.d1_replay.rendering import (
    RETROSPECTIVE_LABEL,
    STRENGTH_LABEL,
    D1RenderingError,
    render_d1_deliverables,
)
from seismoflux.d1_replay.robustness import build_d1_robustness_result
from seismoflux.data.common import canonical_json_bytes

_MODEL_ORDERS = {
    "B0": (0, 3, 4, 5, 6, 7, 1, 2, 8, 9, 10, 11),
    "B0_R30": (0, 3, 4, 5, 6, 1, 7, 2, 8, 9, 10, 11),
    "B0_C": (0, 1, 3, 4, 5, 6, 7, 2, 8, 9, 10, 11),
    "B0_C_A_snapshot": (1, 0, 3, 4, 5, 6, 7, 2, 8, 9, 10, 11),
    "B0_C_A_dynamic": (0, 3, 4, 5, 6, 7, 8, 1, 2, 9, 10, 11),
    "B0_R30_C_A_dynamic": (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
}
_PREFIX_COUNTS = (3, 4, 6, 7, 9)
_ACTUAL_AREAS = (300_000.0, 400_000.0, 600_000.0, 700_000.0, 900_000.0)
_CLUSTERS_BY_FOLD = {
    30: {"fold_1": 8, "fold_2": 6, "fold_3": 7},
    90: {"fold_1": 8, "fold_2": 6, "fold_3": 8},
}
_IDENTITIES = {
    "contract_sha256": "a" * 64,
    "manifest_content_sha256": "b" * 64,
    "input_sha256": "c" * 64,
    "git_commit": "d" * 40,
}
_FEATURES = {
    "B0": [],
    "B0_R30": [],
    "B0_C": ["C1", "C2"],
    "B0_C_A_snapshot": ["C1", "C2", "S1", "S2", "S3", "S4", "S5"],
    "B0_C_A_dynamic": ["C1", "C2", "S1", "S2", "S3", "S4", "S5", "D1", "D2"],
    "B0_R30_C_A_dynamic": [
        "C1",
        "C2",
        "S1",
        "S2",
        "S3",
        "S4",
        "S5",
        "D1",
        "D2",
    ],
}


def _time(base: datetime, days: int) -> str:
    return (base + timedelta(days=days)).astimezone(UTC).isoformat().replace("+00:00", "Z")


def _synthetic_bundle(root: Path) -> tuple[Path, Path, dict[str, Any]]:
    base = datetime(2023, 1, 1, tzinfo=UTC)
    issue_rows: list[dict[str, Any]] = []
    issue_axis: dict[tuple[str, int], list[tuple[str, str]]] = {}
    for fold_index, fold in enumerate(D1_FOLD_IDS):
        for horizon, count in ((30, 8), (90, 3)):
            values: list[tuple[str, str]] = []
            for issue_index in range(count):
                issue_id = f"issue_{horizon}_{fold}_{issue_index}"
                issue_time = _time(base, fold_index * 400 + issue_index * horizon)
                values.append((issue_id, issue_time))
                for model in D1_MODEL_ORDER:
                    issue_rows.append(
                        {
                            "fold_id": fold,
                            "issue_id": issue_id,
                            "issue_time_utc": issue_time,
                            "horizon_days": horizon,
                            "model_id": model,
                            "alarm_prefix_counts": list(_PREFIX_COUNTS),
                            "actual_area_km2": list(_ACTUAL_AREAS),
                        }
                    )
            issue_axis[(fold, horizon)] = values

    outcomes: list[dict[str, Any]] = []
    expected_support: dict[str, list[dict[str, str]]] = {"30": [], "90": []}
    for horizon in (30, 90):
        for fold in D1_FOLD_IDS:
            issues = issue_axis[(fold, horizon)]
            for cluster_index in range(_CLUSTERS_BY_FOLD[horizon][fold]):
                issue_id, issue_time = issues[cluster_index % len(issues)]
                cluster_id = hashlib.sha256(
                    f"cluster_{horizon}_{fold}_{cluster_index}".encode()
                ).hexdigest()
                # Mostly cell 1 (coverage-sensitive), with deterministic cell-8 misses.
                representative = 8 if cluster_index % 5 == 4 else 1
                expected_support[str(horizon)].append(
                    {
                        "cluster_id": cluster_id,
                        "fold_id": fold,
                        "issue_id": issue_id,
                    }
                )
                for model in D1_MODEL_ORDER:
                    rank = _MODEL_ORDERS[model].index(representative) + 1
                    hits = [rank <= count for count in _PREFIX_COUNTS]
                    outcomes.append(
                        {
                            "cluster_id": cluster_id,
                            "fold_id": fold,
                            "issue_id": issue_id,
                            "issue_time_utc": issue_time,
                            "horizon_days": horizon,
                            "model_id": model,
                            "representative_cell_index": representative,
                            "log_density": -8.0 + 0.01 * D1_MODEL_ORDER.index(model),
                            "outside_support": False,
                            "hit_by_area": hits,
                        }
                    )

    for support_rows in expected_support.values():
        support_rows.sort(key=lambda row: row["cluster_id"])

    folds = []
    for fold in D1_FOLD_IDS:
        folds.append(
            {
                "fold_id": fold,
                "models": [
                    {
                        "fold_id": fold,
                        "model_id": model,
                        "base": "R30" if model in {"B0_R30", "B0_R30_C_A_dynamic"} else "B0",
                        "feature_groups": _FEATURES[model],
                        "selected_alpha": 0.25,
                        "selected_ridge": None if model in {"B0", "B0_R30"} else 1.0,
                        "training_diagnostic": {
                            "fit_issue_count": 8,
                            "fit_catalog_m4plus_event_count": 60,
                            "coefficient_names": [],
                            "active_coefficients": [],
                            "coefficients": [],
                            "iteration_count": 0 if model in {"B0", "B0_R30"} else 4,
                            "objective": None if model in {"B0", "B0_R30"} else 2.5,
                        },
                    }
                    for model in D1_MODEL_ORDER
                ],
            }
        )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol_version": "d1.0.0",
        "result_kind": "observed_replay",
        "retrospective_only": True,
        "relative_strength_not_absolute_probability": True,
        "status": "completed",
        "identities": dict(_IDENTITIES),
        "workers": 2,
        "blas_threads_per_worker": 1,
        "folds": folds,
        "outcomes": outcomes,
        "issue_alarm_outcomes": issue_rows,
        "expected_support_by_horizon": expected_support,
        "expected_issues_by_horizon": {
            str(horizon): [
                {"fold_id": fold, "issue_id": issue_id}
                for fold in D1_FOLD_IDS
                for issue_id, _ in issue_axis[(fold, horizon)]
            ]
            for horizon in (30, 90)
        },
        "study_area_km2": 1_200_000.0,
    }
    observed = root / "observed_result.json"
    observed.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    cell_rows: list[dict[str, Any]] = []
    sorted_issues = sorted(
        issue_rows,
        key=lambda row: (
            row["fold_id"],
            row["issue_time_utc"],
            row["horizon_days"],
            D1_MODEL_ORDER.index(row["model_id"]),
        ),
    )
    for issue in sorted_issues:
        model = issue["model_id"]
        order = _MODEL_ORDERS[model]
        weight_total = sum(range(1, 13))
        for rank, cell_index in enumerate(order, start=1):
            mass = (13 - rank) / weight_total
            row: dict[str, Any] = {
                "fold_id": issue["fold_id"],
                "issue_id": issue["issue_id"],
                "issue_time_utc": issue["issue_time_utc"],
                "horizon_days": issue["horizon_days"],
                "model_id": model,
                "cell_index": cell_index,
                "cell_id": f"cell_{cell_index}",
                "row": cell_index // 4,
                "column": cell_index % 4,
                "query_x_m": float((cell_index % 4) * 25_000),
                "query_y_m": float((cell_index // 4) * 25_000),
                "clipped_area_km2": 100_000.0,
                "relative_cell_mass": mass,
                "relative_strength_per_km2": mass / 100_000.0,
                "rank": rank,
            }
            for suffix, count in zip(
                ("300000", "450000", "600000", "750000", "960000"),
                _PREFIX_COUNTS,
                strict=True,
            ):
                row[f"alarm_{suffix}"] = rank <= count
            cell_rows.append(row)
    scores = root / "d1_cell_scores.parquet"
    pq.write_table(pa.Table.from_pylist(cell_rows), scores)
    payload["cell_scores"] = {
        "path": scores.name,
        "file_sha256": hashlib.sha256(scores.read_bytes()).hexdigest(),
        "frame_count": len(issue_rows),
        "row_count": len(cell_rows),
    }
    observed.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return observed, scores, payload


def _component_statistics(payload: dict[str, Any]) -> dict[str, float]:
    hits = {
        model: sum(
            row["hit_by_area"][2]
            for row in payload["outcomes"]
            if row["horizon_days"] == 30 and row["model_id"] == model
        )
        for model in ("B0_C", "B0_C_A_snapshot", "B0_C_A_dynamic")
    }
    return {
        "B0_C_A_snapshot_minus_B0_C": (hits["B0_C_A_snapshot"] - hits["B0_C"]) / 21,
        "B0_C_A_dynamic_minus_B0_C": (hits["B0_C_A_dynamic"] - hits["B0_C"]) / 21,
        "B0_C_A_dynamic_minus_B0_C_A_snapshot": (hits["B0_C_A_dynamic"] - hits["B0_C_A_snapshot"])
        / 21,
    }


def _supplemental_results(
    root: Path,
    payload: dict[str, Any],
) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    statistics = _component_statistics(payload)
    kinds: dict[str, Any] = {}
    for kind in ("time", "space"):
        contrasts: dict[str, Any] = {}
        for contrast, observed in statistics.items():
            nulls = [observed - 0.01] * 200
            contrasts[contrast] = {
                "contrast": contrast,
                "observed_statistic": observed,
                "null_statistics": nulls,
                "null_greater_or_equal_count": 0,
                "scientific_failure_count": 0,
                "scientific_failure_fraction": 0.0,
                "monte_carlo_p_value": 1 / 201,
                "denominator": 201,
                "observed_exceeds_fraction": 1.0,
                "status": "passed",
                "mechanism_promising_for_kind": True,
            }
        kinds[kind] = {
            "kind": kind,
            "contrasts": contrasts,
            "fold_scientific_failure_counts": {fold: 0 for fold in D1_FOLD_IDS},
        }
    placebo_identities = {
        **_IDENTITIES,
        "observed_input_sha256": _IDENTITIES["input_sha256"],
        "input_sha256": "e" * 64,
    }
    placebo: dict[str, Any] = {
        "schema_version": 1,
        "protocol_version": "d1.0.0",
        "result_kind": "d1_time_and_space_placebos",
        "retrospective_only": True,
        "status": "completed",
        "identities": placebo_identities,
        "replications_each": 200,
        "schedule_by_kind_and_fold": {
            kind: {fold: 200 for fold in D1_FOLD_IDS} for kind in ("time", "space")
        },
        "observed_statistics": statistics,
        "kinds": kinds,
        "anomaly_mechanism_promising_by_contrast": {contrast: True for contrast in statistics},
    }
    zone_ids = sorted(
        hashlib.sha256(f"target-independent-zone-{index}".encode()).hexdigest()
        for index in range(39)
    )
    robustness = build_d1_robustness_result(
        payload,
        expected_contract_sha256=_IDENTITIES["contract_sha256"],
        expected_manifest_content_sha256=_IDENTITIES["manifest_content_sha256"],
        zone_ids=zone_ids,
        zone_id_by_cell_index=zone_ids,
        spatial_strata_identity={
            "public_manifest_content_sha256": "1" * 64,
            "artifact_sha256": {
                "cell_mapping": "2" * 64,
                "entity_mapping": "3" * 64,
                "zone_geometry": "4" * 64,
                "connectors": "5" * 64,
            },
            "operational_grid_id": "synthetic_grid",
            "operational_cell_count": 15_697,
            "nonempty_zone_count": 39,
            "geometry_zone_count": 65,
            "zero_cell_geometry_zone_count": 26,
        },
    )
    placebo_path = root / "d1_placebo_result.json"
    robustness_path = root / "d1_robustness_result.json"
    placebo_path.write_text(json.dumps(placebo, ensure_ascii=False, indent=2), encoding="utf-8")
    robustness_path.write_text(
        json.dumps(robustness, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return placebo_path, robustness_path, placebo, robustness


def test_rendered_bundle_has_four_controls_labels_models_and_verified_hashes(
    tmp_path: Path,
) -> None:
    observed, scores, _ = _synthetic_bundle(tmp_path)
    output = tmp_path / "outputs" / "d1"
    rendered = render_d1_deliverables(tmp_path, observed, scores, output)

    assert rendered.best_intermediate_model == "B0_C"
    for path in (
        rendered.effects_svg_path,
        rendered.maps_svg_path,
        rendered.science_report_path,
        rendered.explorer_html_path,
        rendered.science_summary_path,
        rendered.manifest_path,
    ):
        assert path.is_file()

    effects = rendered.effects_svg_path.read_text(encoding="utf-8")
    maps = rendered.maps_svg_path.read_text(encoding="utf-8")
    report = rendered.science_report_path.read_text(encoding="utf-8")
    explorer = rendered.explorer_html_path.read_text(encoding="utf-8")
    for document in (effects, maps, report, explorer):
        assert RETROSPECTIVE_LABEL in document
    for document in (maps, report, explorer):
        assert STRENGTH_LABEL in document
    assert "时间置乱、空间置乱" in effects
    assert "原始预测效果预分类" in report
    assert "最终科学结论" in report
    assert "支持域内真实命中/漏报案例" in report
    assert "最早支持域内命中" in maps
    assert "最早支持域内漏报" in maps
    for control in ("issueControl", "modelControl", "horizonControl", "areaControl"):
        assert f'id="{control}"' in explorer
    for model in D1_MODEL_ORDER:
        assert model in effects
        assert model in explorer
    assert "textContent" in explorer
    assert "fetch(" not in explorer
    assert "<script src=" not in explorer.casefold()
    assert "http://" not in explorer.casefold()
    assert "https://" not in explorer.casefold()

    manifest = json.loads(rendered.manifest_path.read_text(encoding="utf-8"))
    identity = manifest.pop("manifest_identity_sha256")
    assert identity == hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    assert identity == rendered.manifest_identity_sha256
    assert (
        rendered.manifest_file_sha256
        == hashlib.sha256(rendered.manifest_path.read_bytes()).hexdigest()
    )
    for artifact in manifest["outputs"]:
        artifact_path = tmp_path / artifact["path"]
        assert artifact["sha256"] == hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        assert artifact["size_bytes"] == artifact_path.stat().st_size

    summary = json.loads(rendered.science_summary_path.read_text(encoding="utf-8"))
    assert summary["result_kind"] == "d1_science_summary"
    assert summary["frozen_axis"]["cluster_counts_by_horizon"] == {"30": 21, "90": 22}
    assert summary["frozen_axis"]["issue_counts_by_horizon"] == {"30": 24, "90": 9}
    assert summary["frozen_axis"]["issue_model_frame_count"] == 198
    assert len(summary["component_contributions"]) == 3
    assert set(summary["deterministic_inside_support_cases"]) == {"命中", "漏报"}
    assert all(summary["deterministic_inside_support_cases"].values())
    assert summary["supplemental_evidence"]["final_attribution_ready"] is False


def test_canonical_optional_placebo_and_robustness_are_bound_and_rendered(
    tmp_path: Path,
) -> None:
    observed, scores, payload = _synthetic_bundle(tmp_path)
    placebo_path, robustness_path, _, _ = _supplemental_results(tmp_path, payload)
    rendered = render_d1_deliverables(
        tmp_path,
        observed,
        scores,
        tmp_path / "complete",
        placebo_result_path=placebo_path,
        robustness_result_path=robustness_path,
    )

    summary = json.loads(rendered.science_summary_path.read_text(encoding="utf-8"))
    evidence = summary["supplemental_evidence"]
    assert evidence["final_attribution_ready"] is True
    assert evidence["time_and_space_placebos"]["status"] == "completed"
    assert evidence["regional_and_leave_one_cluster_robustness"]["status"] == "completed"
    robustness_summary = evidence["regional_and_leave_one_cluster_robustness"]["diagnostic_summary"]
    assert robustness_summary["contrast_count"] == 3
    assert robustness_summary["primary_endpoint"]["cluster_count"] == 21
    assert all(
        row["leave_one_cluster_out"]["replication_count"] == 21
        for row in robustness_summary["contrasts"]
    )
    assert all(
        component["time_placebo_p_value"] == pytest.approx(1 / 201)
        and component["space_placebo_p_value"] == pytest.approx(1 / 201)
        for component in summary["component_contributions"]
    )
    effects = rendered.effects_svg_path.read_text(encoding="utf-8")
    report = rendered.science_report_path.read_text(encoding="utf-8")
    explorer = rendered.explorer_html_path.read_text(encoding="utf-8")
    assert "时间/空间置乱归因已完成" in effects
    assert "时间/空间置乱和稳健性诊断均已完成" in report
    assert "区域与逐震群稳健性" in report
    assert "robustnessRows" in explorer
    manifest = json.loads(rendered.manifest_path.read_text(encoding="utf-8"))
    assert manifest["placebo_attribution_complete"] is True
    assert manifest["robustness_diagnostics_complete"] is True
    assert (
        manifest["inputs"]["placebo_result"]["sha256"]
        == hashlib.sha256(placebo_path.read_bytes()).hexdigest()
    )


@pytest.mark.parametrize("corruption", ("p_value", "replications", "identity"))
def test_placebo_validation_fails_closed_on_p_value_replications_or_identity(
    tmp_path: Path,
    corruption: str,
) -> None:
    observed, scores, payload = _synthetic_bundle(tmp_path)
    placebo_path, _, placebo, _ = _supplemental_results(tmp_path, payload)
    if corruption == "p_value":
        placebo["kinds"]["time"]["contrasts"]["B0_C_A_snapshot_minus_B0_C"][
            "monte_carlo_p_value"
        ] = 0.5
    elif corruption == "replications":
        placebo["kinds"]["space"]["contrasts"]["B0_C_A_dynamic_minus_B0_C"]["null_statistics"].pop()
    else:
        placebo["identities"]["contract_sha256"] = "f" * 64
    placebo_path.write_text(json.dumps(placebo), encoding="utf-8")

    with pytest.raises(D1RenderingError, match="placebo"):
        render_d1_deliverables(
            tmp_path,
            observed,
            scores,
            tmp_path / "must_not_render",
            placebo_result_path=placebo_path,
        )
    assert not (tmp_path / "must_not_render").exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda payload: payload.update(status="prepared"), "completed observed replay"),
        (lambda payload: payload.update(result_kind="forecast"), "observed_replay only"),
        (lambda payload: payload.update(outcomes=[]), "no real cluster outcomes"),
        (
            lambda payload: payload.pop("issue_alarm_outcomes"),
            "issue_alarm_outcomes",
        ),
    ),
)
def test_rendering_fails_closed_without_completed_real_effect_or_required_fields(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    observed, scores, payload = _synthetic_bundle(tmp_path)
    mutation(payload)
    observed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(D1RenderingError, match=message):
        render_d1_deliverables(tmp_path, observed, scores, tmp_path / "out")


def test_rendering_recomputes_hits_from_the_ranked_parquet_prefixes(tmp_path: Path) -> None:
    observed, scores, payload = _synthetic_bundle(tmp_path)
    outcome = next(
        row for row in payload["outcomes"] if row["horizon_days"] == 30 and row["model_id"] == "B0"
    )
    outcome["hit_by_area"][2] = not outcome["hit_by_area"][2]
    observed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(D1RenderingError, match="hit_by_area differs"):
        render_d1_deliverables(tmp_path, observed, scores, tmp_path / "tampered_hit")
    assert not (tmp_path / "tampered_hit").exists()


def test_rendering_rejects_target_mapping_drift_across_models(tmp_path: Path) -> None:
    observed, scores, payload = _synthetic_bundle(tmp_path)
    reference = next(
        row for row in payload["outcomes"] if row["horizon_days"] == 30 and row["model_id"] == "B0"
    )
    candidate = next(
        row
        for row in payload["outcomes"]
        if row["horizon_days"] == 30
        and row["model_id"] == "B0_R30"
        and row["cluster_id"] == reference["cluster_id"]
    )
    candidate["representative_cell_index"] = 2
    observed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(D1RenderingError, match="models disagree on representative_cell_index"):
        render_d1_deliverables(tmp_path, observed, scores, tmp_path / "drifted_target")
    assert not (tmp_path / "drifted_target").exists()


def test_empty_robustness_shell_cannot_mark_final_attribution_ready(tmp_path: Path) -> None:
    observed, scores, payload = _synthetic_bundle(tmp_path)
    placebo_path, robustness_path, _, robustness = _supplemental_results(tmp_path, payload)
    robustness.pop("contrasts")
    robustness_path.write_text(json.dumps(robustness), encoding="utf-8")

    with pytest.raises(D1RenderingError, match="robustness"):
        render_d1_deliverables(
            tmp_path,
            observed,
            scores,
            tmp_path / "empty_robustness",
            placebo_result_path=placebo_path,
            robustness_result_path=robustness_path,
        )
    assert not (tmp_path / "empty_robustness").exists()


def test_registered_area_axis_remains_visible_in_effect_output(tmp_path: Path) -> None:
    observed, scores, _ = _synthetic_bundle(tmp_path)
    rendered = render_d1_deliverables(tmp_path, observed, scores, tmp_path / "rendered")
    effects = rendered.effects_svg_path.read_text(encoding="utf-8")
    for area in D1_AREA_BUDGETS_KM2:
        assert f">{area / 1000:.0f}<" in effects
    assert "30天" in effects
    assert "90天" in effects
    assert "2000次配对震群Bootstrap" in effects
