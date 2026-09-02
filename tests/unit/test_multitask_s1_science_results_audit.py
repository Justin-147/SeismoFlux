from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "audit_multitask_s1_science_results.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("audit_multitask_s1_science_results", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(family: str, payload: dict[str, object]) -> dict[str, object]:
    return {
        "score_family": family,
        "fold_id": payload["fold_id"],
        "issue_time_utc": pd.Timestamp(payload["issue_time_utc"]),
        "horizon_days": payload.get("horizon_days"),
        "model_id": payload["model_id"],
        "status": "evaluable",
        "payload_json": json.dumps(payload, separators=(",", ":")),
    }


def _main_location(
    *, fold: str, issue: str, event: str, episode: str, model: str, hit: bool
) -> dict[str, object]:
    return {
        "fold_id": fold,
        "issue_time_utc": issue,
        "horizon_days": 30,
        "magnitude_bin": "M5_6",
        "model_id": model,
        "metric": "strict_recall",
        "basis": "anchor",
        "area_budget_km2": 600_000.0,
        "actual_area_km2": 500_000.0,
        "catalog_delay_hours": 24,
        "hit_tolerance_km": 0.0,
        "is_main_scientific_anchor": True,
        "event_ids": [event],
        "episode_ids": [episode],
        "event_weights": [1.0],
        "hit_flags": [hit],
    }


def _synthetic_raw_scores() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    common_density = -3.0
    records.append(
        _record(
            "location",
            {
                "fold_id": "2000_2004",
                "issue_time_utc": "2000-01-06T16:00:00Z",
                "horizon_days": 7,
                "magnitude_bin": "M5_6",
                "model_id": "L0_UNIFORM",
                "metric": "spatial_log_density",
                "event_log_densities_per_km2": [common_density] * 4,
            },
        )
    )
    folds = ("2000_2004", "2005_2009", "2010_2014", "2015_2019")
    l1_hits = (False, False, True, False)
    l2_hits = (True, False, True, True)
    l3_hits = (True, True, True, True)
    for index, fold in enumerate(folds):
        issue = f"{2000 + index * 5}-01-06T16:00:00Z"
        event = f"e{index + 1}"
        episode = f"target_episode_{index + 1}"
        for model, hit in (
            ("L0_UNIFORM", False),
            ("L1_REGIONAL_CONSTANT", l1_hits[index]),
            ("L2_KDE_CAUSAL", l2_hits[index]),
            ("L3_B0_R30_CAUSAL", l3_hits[index]),
        ):
            records.append(
                _record(
                    "location",
                    _main_location(
                        fold=fold,
                        issue=issue,
                        event=event,
                        episode=episode,
                        model=model,
                        hit=hit,
                    ),
                )
            )
        m0 = {
            "fold_id": fold,
            "forecast_issue_time_utc": issue,
            "issue_time_utc": issue,
            "model_id": "M0_GR_GLOBAL",
            "conditional_support": "M>=5 unique physical events, M0 re-normalized tail",
            "event_ids": [event],
            "event_log_probabilities": [-1.0],
        }
        m3 = {
            **m0,
            "model_id": "M3_GR_LONG_M5",
            "conditional_support": "M>=5 unique physical events, conditional tail",
            "event_log_probabilities": [-0.9 + index * 0.1],
        }
        records.extend((_record("magnitude", m0), _record("magnitude", m3)))
    return pd.DataFrame.from_records(records)


def _synthetic_catalog() -> pd.DataFrame:
    # e1/e2 are one fixed-anchor cluster; e3 and e4 are separate clusters.
    return pd.DataFrame(
        {
            "event_id": ["e1", "e2", "e3", "e4"],
            "origin_time_utc": pd.to_datetime(
                [
                    "2000-01-10T00:00:00Z",
                    "2000-01-20T00:00:00Z",
                    "2010-01-10T00:00:00Z",
                    "2015-01-10T00:00:00Z",
                ],
                utc=True,
            ),
            "available_at": pd.to_datetime(
                [
                    "2000-01-11T00:00:00Z",
                    "2000-01-21T00:00:00Z",
                    "2010-01-11T00:00:00Z",
                    "2015-01-11T00:00:00Z",
                ],
                utc=True,
            ),
            "longitude": [100.0, 100.1, 110.0, 120.0],
            "latitude": [30.0, 30.1, 35.0, 40.0],
            "magnitude": [5.1, 5.2, 5.3, 6.1],
            "inside_study_area": [True, True, True, True],
        }
    )


def test_synthetic_diagnostic_reports_point_effects_and_cluster_interval() -> None:
    module = _module()
    result = module.compute_science_diagnostic(
        _synthetic_raw_scores(),
        _synthetic_catalog(),
        study_area_km2=1_000_000.0,
        max_time_days=30,
        max_distance_km=75.0,
        bootstrap_replicates=500,
        bootstrap_seed=20260902,
    )

    uniform = result["uniform_tie_diagnostic"]
    assert uniform["all_event_log_densities_exactly_identical"] is True
    assert uniform["actual_alarm_area_is_exactly_fixed"] is True
    assert uniform["random_area_expected_recall"] == pytest.approx(0.5)
    assert uniform["random_area_expected_hit_count"] == pytest.approx(2.0)
    assert uniform["sealed_fixed_prefix_role"] == "audit_record_only_not_uniform_random_baseline"

    l2, l3 = result["location_point_differences_vs_interpretable_L1"]
    assert l2["pooled"]["point_difference_candidate_minus_L1"] == pytest.approx(0.5)
    assert l3["pooled"]["point_difference_candidate_minus_L1"] == pytest.approx(0.75)
    assert l2["inference"] == "posthoc_point_difference_only_no_pairwise_CI"

    magnitude = result["magnitude_cluster_dependence_diagnostic"]
    assert magnitude["event_count"] == 4
    assert magnitude["combined_M5_plus_episode_count"] == 3
    assert magnitude["cluster_bootstrap"]["strictly_positive"] is True
    assert magnitude["wording_gate"] == "cluster_robust_small_development_signal"
    assert magnitude["remove_largest_combined_episode"]["full_catalog_member_count"] == 2
    assert [row["event_count"] for row in magnitude["four_outer_fold_mean_effects"]] == [1] * 4
    assert result["holdout_opened"] is False
    assert result["locked_test_run"] is False


def test_nonuniform_L0_density_is_rejected_before_any_inference() -> None:
    module = _module()
    raw = _synthetic_raw_scores()
    location_index = raw.index[raw["score_family"] == "location"][0]
    payload = json.loads(raw.at[location_index, "payload_json"])
    payload["event_log_densities_per_km2"][-1] = -2.9
    raw.at[location_index, "payload_json"] = json.dumps(payload)

    with pytest.raises(module.ScienceDiagnosticError, match="not exactly identical"):
        module.compute_science_diagnostic(
            raw,
            _synthetic_catalog(),
            study_area_km2=1_000_000.0,
            max_time_days=30,
            max_distance_km=75.0,
            bootstrap_replicates=20,
            bootstrap_seed=1,
        )


def test_changed_config_hash_is_rejected_without_reading_real_scores(tmp_path: Path) -> None:
    module = _module()
    config = tmp_path / "changed.yaml"
    config.write_text("role: changed\n", encoding="utf-8")

    with pytest.raises(module.ScienceDiagnosticError, match="config SHA-256 changed"):
        module.run_diagnostic(
            config_path=config,
            project_root=tmp_path,
            data_root=tmp_path,
            output_dir=tmp_path / "out",
        )
