from __future__ import annotations

import hashlib
import json
import math
import tomllib
from datetime import date, timedelta
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from seismoflux.background.etas import (
    ETASParent,
    aki_b_value,
    branching_ratio,
    conditional_intensity,
    inverse_power_cutoff_mass,
    inverse_power_density,
    inverse_power_mass,
    inverse_power_scale,
    omori_cdf,
    omori_density,
    productivity,
    truncated_gr_exp_expectation,
)

BACKGROUND_CONFIG = Path("configs/background.yaml")
FOLD_MANIFEST = Path("data/manifests/background_fold_manifest.json")
DATA_CATALOG = Path("data/manifests/data_catalog.json")
MICRO_FIXTURE = Path("tests/fixtures/background/etas_micro_reference.json")
JAPAN_FIXTURE = Path("tests/fixtures/background/jss_japan_reference.json")
PROJECT_METADATA = Path("pyproject.toml")
ENVIRONMENT_LOCK = Path("uv.lock")
RESEARCH_CONFIG = Path("configs/research_protocol.yaml")
RESEARCH_DOCUMENT = Path("docs/research_protocol.md")


def _load_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _load_yaml(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], yaml.safe_load(path.read_text(encoding="utf-8")))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_background_protocol_is_frozen_before_scores() -> None:
    config = _load_yaml(BACKGROUND_CONFIG)

    assert config["status"] == "preregistered_before_any_background_score"
    assert config["frozen_before_background_scores"] is True
    assert config["background_scores_seen_before_freeze"] is False
    assert config["freeze_tag"] == "v0.2.0-background-protocol"

    inputs = config["inputs"]
    catalog = _load_json(DATA_CATALOG)
    assert inputs["data_catalog_sha256"] == _sha256(DATA_CATALOG)
    assert inputs["expected_stage1_snapshot_id"] == catalog["snapshot_id"]
    assert (
        inputs["issue_manifest_source_sha256"]
        == catalog["datasets"]["anomaly_report_period"]["file_sha256"]
    )
    assert inputs["study_area_sha256"] == catalog["study_area"]["sha256"]
    assert inputs["issue_manifest_column_allowlist"] == ["available_at"]
    assert inputs["issue_manifest_feature_values_forbidden"] is True


def test_actual_issue_manifest_and_nonoverlapping_exposures_are_exact() -> None:
    manifest = _load_json(FOLD_MANIFEST)
    expected = {
        "development": (50, "2022-07-21", "2023-06-29"),
        "validation": (51, "2024-07-04", "2025-06-26"),
    }

    for partition_id, (count, first, last) in expected.items():
        partition = manifest["partitions"][partition_id]
        issue_dates = partition["actual_issue_dates_local"]
        assert len(issue_dates) == partition["actual_issue_date_count"] == count
        assert issue_dates == sorted(set(issue_dates))
        assert issue_dates[0] == first
        assert issue_dates[-1] == last

        parsed = [date.fromisoformat(value) for value in issue_dates]
        for horizon in (7, 30, 90, 180, 365):
            selected: list[date] = []
            for candidate in parsed:
                if not selected or candidate >= selected[-1] + timedelta(days=horizon):
                    selected.append(candidate)
            recorded = partition["non_overlapping_exposures"][str(horizon)]
            assert recorded["count"] == len(selected)
            assert recorded["issue_dates_local"] == [value.isoformat() for value in selected]

    assert manifest["source"]["column_read"] == "available_at"
    assert manifest["source"]["all_other_anomaly_columns_forbidden"] is True


def test_etas_cutoffs_rng_and_grid_are_unambiguous() -> None:
    config = _load_yaml(BACKGROUND_CONFIG)
    inputs = config["inputs"]
    etas = config["etas"]
    randomness = config["randomness"]
    integration = config["integration"]
    evaluation = config["evaluation"]

    assert (
        etas["spatial_kernel"]["support_radius_km"] == inputs["include_external_trigger_buffer_km"]
    )
    assert etas["spatial_kernel"]["renormalize_radially_on_support"] is True
    assert etas["temporal_kernel"]["cutoff_role"] == (
        "omit_older_history_parents_without_kernel_renormalization"
    )
    assert etas["magnitude_model"]["upper_magnitude"] == 9.5
    assert etas["branching_ratio"]["maximum"] == 0.95
    assert etas["likelihood"]["compensator_domain"] == "study_area_only"
    assert etas["likelihood"]["external_parent_role"] == (
        "conditional_history_only_never_target_or_background_training_event"
    )
    assert etas["simulation"]["buffer_events_in_outputs_or_metrics"] is False

    fields = [
        "seismoflux",
        "147",
        "0.2.0",
        "simulation_regression",
        "etas_inverse_power_cut300_v1",
        "-",
        "00000000",
    ]
    digest = hashlib.sha256(b"\x00".join(value.encode("utf-8") for value in fields)).digest()
    assert digest.hex() == randomness["seed_derivation"]["reference_sha256"]
    assert (
        int.from_bytes(digest[:16], "big")
        == randomness["seed_derivation"]["reference_entropy_integer"]
    )
    contexts = randomness["seed_derivation"]["namespace_contexts"]
    assert contexts["optimizer_start"] == {
        "model_ids": [
            "etas/fold_1",
            "etas/fold_2",
            "etas/fold_3",
            "etas/fold_4",
            "etas/final_validation",
        ],
        "issue_id_or_dash": "-",
        "replicate_index_role": "optimizer_start_index",
        "replicate_index_first": 0,
        "replicate_index_last_inclusive": 4,
    }
    assert contexts["future_simulation"]["model_id"] == "etas/final_validation"
    assert contexts["future_simulation"]["replicate_index_last_inclusive"] == 127
    assert contexts["simulation_regression"]["model_id"] == "etas_inverse_power_cut300_v1"
    assert contexts["simulation_regression"]["replicate_index_last_inclusive"] == 8191
    assert contexts["bootstrap"]["model_ids"] == [
        "spatial_poisson_vs_uniform_poisson",
        "etas_vs_uniform_poisson",
    ]
    assert contexts["bootstrap"]["issue_id"] == ("g1_primary_validation_2024-07-01_2025-07-01")
    assert contexts["bootstrap"]["replicate_index_last_inclusive"] == 1999
    assert randomness["future_simulation"]["maximum_events_per_replicate"] == 100000

    assert integration["grid_origin_m"] == [0.0, 0.0]
    assert integration["grid_cells_km"] == [50, 25, 12.5]
    assert integration["primary_convergence_pair_km"] == [25, 12.5]
    assert integration["refinement_rule"] == "target_independent_full_grid_only"
    assert evaluation["g1_primary_endpoint"]["horizon_aggregation"] == "none"
    assert evaluation["g1_primary_endpoint"]["development_folds"] == ("parameter_selection_folds")
    assert evaluation["issue_based_horizon_backtests"]["cross_horizon_aggregation"] == (
        "none_report_each_horizon_only"
    )
    assert evaluation["issue_based_horizon_backtests"]["actual_development_dates_role"] == (
        "reserved_for_later_anomaly_comparison_not_scored_with_final_background_snapshot"
    )


def test_phase2_randomness_has_one_namespaced_cross_protocol_contract() -> None:
    research = _load_yaml(RESEARCH_CONFIG)
    document = RESEARCH_DOCUMENT.read_text(encoding="utf-8")
    background = _load_yaml(BACKGROUND_CONFIG)

    assert "stability_seeds" not in research
    assert research["randomness"]["stage_2_authority"] == "configs/background.yaml"
    assert research["randomness"]["stage_2_direct_integer_subseeds_forbidden"] is True
    assert research["evaluation"]["bootstrap"]["seed_source"] == ("sha256_bootstrap_namespace")
    assert "seed" not in research["evaluation"]["bootstrap"]
    assert research["evaluation"]["permutations"]["seed_rule"] == (
        "sha256_time_or_space_permutation_namespace_then_PCG64"
    )
    assert background["randomness"]["root_seed"] == research["random_seed"] == 147
    assert "10147" not in document
    assert "147+i" not in document


def test_kde_and_final_model_one_standard_error_rules_are_fully_paired() -> None:
    config = _load_yaml(BACKGROUND_CONFIG)
    kde = config["spatial_poisson"]
    selection = config["evaluation"]["model_selection"]

    assert kde["best_bandwidth_rule"] == (
        "highest_equal_weight_mean_of_four_development_fold_scores"
    )
    assert kde["paired_difference"] == (
        "candidate_minus_best_bandwidth_on_same_fold_events_and_compensator"
    )
    assert kde["standard_error_formula"] == (
        "sample_stddev_of_four_paired_fold_differences_ddof1_div_sqrt4"
    )
    assert kde["one_standard_error_eligibility"] == (
        "candidate_mean_fold_score_gte_best_mean_fold_score_minus_paired_standard_error"
    )
    assert kde["selection_rule"] == "largest_eligible_bandwidth_km"
    assert kde["exact_tie_rule"] == "largest_bandwidth_km"
    assert selection["eligible_model_pool"] == (
        "only_models_with_successful_fit_and_all_model_specific_numerical_stability_and_"
        "grid_convergence_gates"
    )
    assert selection["failed_model_role"] == (
        "excluded_from_validation_best_one_standard_error_threshold_and_final_selection"
    )


def test_phase2_numerical_dependencies_are_directly_pinned_and_locked() -> None:
    metadata = tomllib.loads(PROJECT_METADATA.read_text(encoding="utf-8"))
    direct_dependencies = set(metadata["project"]["dependencies"])
    assert {"numpy==2.4.6", "scipy==1.17.1", "matplotlib==3.11.0"} <= direct_dependencies

    lock = tomllib.loads(ENVIRONMENT_LOCK.read_text(encoding="utf-8"))
    locked_versions = {item["name"]: item["version"] for item in lock["package"]}
    assert locked_versions["numpy"] == "2.4.6"
    assert locked_versions["scipy"] == "1.17.1"
    assert locked_versions["matplotlib"] == "3.11.0"

    config = _load_yaml(BACKGROUND_CONFIG)
    assert config["inputs"]["environment_lock"] == "uv.lock"
    assert config["inputs"]["environment_lock_sha256"] == _sha256(ENVIRONMENT_LOCK)


def test_fixture_tolerances_match_machine_protocol() -> None:
    config = _load_yaml(BACKGROUND_CONFIG)["numerical_regression"]
    fixture = _load_json(MICRO_FIXTURE)["tolerances"]

    assert config["production_scalar_absolute_tolerance"] == fixture["scalar_absolute"]
    assert config["production_scalar_relative_tolerance"] == fixture["scalar_relative"]
    assert (
        config["production_log_likelihood_absolute_tolerance"] == fixture["log_likelihood_absolute"]
    )


def test_micro_etas_fixture_matches_closed_form_reference() -> None:
    fixture = _load_json(MICRO_FIXTURE)
    parameters = fixture["parameters"]
    expected = fixture["expected"]
    tolerances = fixture["tolerances"]

    mc = float(parameters["Mc"])
    background = float(parameters["background_density_per_day_km2"])
    productivity_k = float(parameters["K"])
    alpha = float(parameters["alpha_per_magnitude"])
    c_days = float(parameters["c_days"])
    p_value = float(parameters["p"])
    d_km2 = float(parameters["D_km2"])
    q_value = float(parameters["q"])
    gamma = float(parameters["gamma_per_magnitude"])
    cutoff = float(parameters["spatial_cutoff_km"])

    def productivity_at(magnitude: float) -> float:
        return productivity(magnitude, k=productivity_k, alpha=alpha, mc=mc)

    def temporal_density_at(delta_days: float) -> float:
        return omori_density(delta_days, c_days=c_days, p=p_value)

    def temporal_cdf_at(delta_days: float) -> float:
        return omori_cdf(delta_days, c_days=c_days, p=p_value)

    def spatial_scale_at(magnitude: float) -> float:
        return inverse_power_scale(magnitude, d_km2=d_km2, gamma=gamma, mc=mc)

    def cutoff_mass_at(magnitude: float) -> float:
        return inverse_power_cutoff_mass(
            magnitude,
            d_km2=d_km2,
            q=q_value,
            gamma=gamma,
            mc=mc,
            cutoff_radius_km=cutoff,
        )

    def spatial_density_at(radius_km: float, magnitude: float) -> float:
        return inverse_power_density(
            radius_km,
            magnitude,
            d_km2=d_km2,
            q=q_value,
            gamma=gamma,
            mc=mc,
            cutoff_radius_km=cutoff,
        )

    def spatial_mass_at(radius_km: float, magnitude: float) -> float:
        return inverse_power_mass(
            radius_km,
            magnitude,
            d_km2=d_km2,
            q=q_value,
            gamma=gamma,
            mc=mc,
            cutoff_radius_km=cutoff,
        )

    relative = float(tolerances["scalar_relative"])
    absolute = float(tolerances["scalar_absolute"])

    def assert_close(actual: float, key: str) -> None:
        assert math.isclose(actual, float(expected[key]), rel_tol=relative, abs_tol=absolute)

    assert_close(cutoff_mass_at(4.0), "cutoff_mass_m4")
    assert_close(cutoff_mass_at(3.5), "cutoff_mass_m3p5")
    assert_close(spatial_scale_at(4.0), "sigma_m4_km2")
    assert_close(spatial_scale_at(3.5), "sigma_m3p5_km2")
    assert_close(productivity_at(4.0), "productivity_m4")
    assert_close(productivity_at(3.5), "productivity_m3p5")
    assert_close(temporal_density_at(1.0), "temporal_density_dt1_per_day")
    assert_close(temporal_density_at(2.0), "temporal_density_dt2_per_day")
    assert_close(temporal_cdf_at(0.5), "temporal_cdf_dt0p5")
    assert_close(temporal_cdf_at(1.0), "temporal_cdf_dt1")
    assert_close(temporal_cdf_at(2.0), "temporal_cdf_dt2")
    assert_close(spatial_density_at(0.0, 4.0), "spatial_density_r0_m4_per_km2")
    assert_close(spatial_density_at(5.0, 4.0), "spatial_density_r5_m4_per_km2")
    assert_close(spatial_density_at(5.0, 3.5), "spatial_density_r5_m3p5_per_km2")
    assert_close(spatial_mass_at(30.0, 4.0), "spatial_mass_r30_m4")
    assert_close(spatial_mass_at(30.0, 3.5), "spatial_mass_r30_m3p5")

    root = ETASParent(time_days=0.0, x_km=0.0, y_km=0.0, magnitude=4.0)
    observed = ETASParent(time_days=1.0, x_km=0.0, y_km=0.0, magnitude=3.5)
    lambda_event = conditional_intensity(
        time_days=1.0,
        x_km=0.0,
        y_km=0.0,
        background_density_per_day_km2=background,
        parents=[root],
        mc=mc,
        k=productivity_k,
        alpha=alpha,
        c_days=c_days,
        p=p_value,
        d_km2=d_km2,
        q=q_value,
        gamma=gamma,
        spatial_cutoff_km=cutoff,
    )
    assert_close(lambda_event, "lambda_event_t1_r0_per_day_km2")

    area = math.pi * 30.0**2
    background_compensator = background * area * (2.0 - 0.5)
    root_compensator = (
        productivity_at(4.0)
        * (temporal_cdf_at(2.0) - temporal_cdf_at(0.5))
        * spatial_mass_at(30.0, 4.0)
    )
    event1_compensator = productivity_at(3.5) * temporal_cdf_at(1.0) * spatial_mass_at(30.0, 3.5)
    total_compensator = background_compensator + root_compensator + event1_compensator
    assert_close(background_compensator, "background_compensator")
    assert_close(root_compensator, "root_compensator")
    assert_close(event1_compensator, "event1_postevent_compensator")
    assert_close(total_compensator, "total_compensator")
    assert math.isclose(
        math.log(lambda_event) - total_compensator,
        float(expected["unmarked_log_likelihood"]),
        abs_tol=float(tolerances["log_likelihood_absolute"]),
    )

    lambda_probe = conditional_intensity(
        time_days=2.0,
        x_km=3.0,
        y_km=4.0,
        background_density_per_day_km2=background,
        parents=[root, observed],
        mc=mc,
        k=productivity_k,
        alpha=alpha,
        c_days=c_days,
        p=p_value,
        d_km2=d_km2,
        q=q_value,
        gamma=gamma,
        spatial_cutoff_km=cutoff,
    )
    assert_close(lambda_probe, "lambda_probe_t2_r5_per_day_km2")


def test_omori_primitives_are_normalized_monotone_and_differentiable() -> None:
    c_days = 0.05
    p_value = 1.2
    elapsed = [0.0, 0.01, 0.1, 1.0, 10.0, 1.0e6]
    masses = [omori_cdf(value, c_days=c_days, p=p_value) for value in elapsed]

    assert masses[0] == 0.0
    assert masses == sorted(masses)
    assert all(0.0 <= value < 1.0 for value in masses)
    assert masses[-1] > 0.96

    for value in (0.01, 0.1, 1.0, 10.0):
        step = value * 1.0e-5
        derivative = (
            omori_cdf(value + step, c_days=c_days, p=p_value)
            - omori_cdf(value - step, c_days=c_days, p=p_value)
        ) / (2.0 * step)
        assert math.isclose(
            derivative,
            omori_density(value, c_days=c_days, p=p_value),
            rel_tol=2.0e-9,
        )


def test_inverse_power_primitives_are_normalized_and_radially_consistent() -> None:
    kernel = {
        "d_km2": 25.0,
        "q": 1.5,
        "gamma": 1.0,
        "mc": 3.0,
        "cutoff_radius_km": 300.0,
    }
    radii = [0.0, 1.0, 10.0, 100.0, 300.0, 500.0]
    masses = [inverse_power_mass(radius, 4.0, **kernel) for radius in radii]

    assert masses[0] == 0.0
    assert masses == sorted(masses)
    assert masses[-2:] == [1.0, 1.0]
    assert inverse_power_density(301.0, 4.0, **kernel) == 0.0
    assert 0.0 < inverse_power_cutoff_mass(4.0, **kernel) < 1.0

    for radius in (1.0, 10.0, 100.0):
        step = radius * 1.0e-5
        radial_derivative = (
            inverse_power_mass(radius + step, 4.0, **kernel)
            - inverse_power_mass(radius - step, 4.0, **kernel)
        ) / (2.0 * step)
        area_ring_density = 2.0 * math.pi * radius * inverse_power_density(radius, 4.0, **kernel)
        assert math.isclose(radial_derivative, area_ring_density, rel_tol=2.0e-9)


def test_productivity_aki_and_truncated_gr_properties() -> None:
    productivity_values = [
        productivity(magnitude, k=0.2, alpha=1.0, mc=3.0) for magnitude in (3.0, 3.5, 4.0)
    ]
    assert productivity_values == sorted(productivity_values)
    assert productivity_values[0] == 0.2

    b_value = aki_b_value((value for value in [3.0, 3.5, 4.0]), mc=3.0)
    assert math.isclose(b_value, 0.7896263307331853, rel_tol=1.0e-15)

    assert truncated_gr_exp_expectation(alpha=0.0, beta=2.0, magnitude_span=4.0) == 1.0
    generic = truncated_gr_exp_expectation(alpha=1.0, beta=2.0, magnitude_span=4.0)
    assert math.isclose(generic, 1.964027580075817, rel_tol=1.0e-15)
    equality = truncated_gr_exp_expectation(alpha=2.0, beta=2.0, magnitude_span=4.0)
    near_equality = truncated_gr_exp_expectation(
        alpha=2.0 + 5.0e-13,
        beta=2.0,
        magnitude_span=4.0,
    )
    assert equality == near_equality
    assert math.isclose(
        branching_ratio(k=0.2, alpha=1.0, beta=2.0, magnitude_span=4.0),
        0.3928055160151634,
        rel_tol=1.0e-15,
    )


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_etas_primitives_reject_nonfinite_inputs(bad: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        productivity(bad, k=0.2, alpha=1.0, mc=3.0)
    with pytest.raises(ValueError, match="finite"):
        omori_density(bad, c_days=0.05, p=1.2)
    with pytest.raises(ValueError, match="finite"):
        omori_cdf(bad, c_days=0.05, p=1.2)
    with pytest.raises(ValueError, match="finite"):
        inverse_power_scale(bad, d_km2=25.0, gamma=1.0, mc=3.0)
    with pytest.raises(ValueError, match="finite"):
        inverse_power_density(
            bad,
            4.0,
            d_km2=25.0,
            q=1.5,
            gamma=1.0,
            mc=3.0,
            cutoff_radius_km=300.0,
        )
    with pytest.raises(ValueError, match="finite"):
        aki_b_value([bad], mc=3.0)
    with pytest.raises(ValueError, match="finite"):
        truncated_gr_exp_expectation(alpha=bad, beta=2.0, magnitude_span=4.0)
    with pytest.raises(ValueError, match="finite"):
        ETASParent(time_days=bad, x_km=0.0, y_km=0.0, magnitude=4.0)


def test_etas_primitives_reject_invalid_domains_and_noncausal_history() -> None:
    with pytest.raises(ValueError, match="greater than or equal"):
        productivity(2.9, k=0.2, alpha=1.0, mc=3.0)
    with pytest.raises(ValueError, match="non-negative"):
        productivity(4.0, k=-0.2, alpha=1.0, mc=3.0)
    with pytest.raises(ValueError, match="positive"):
        omori_density(0.0, c_days=0.05, p=1.2)
    with pytest.raises(ValueError, match="greater than 1"):
        omori_cdf(1.0, c_days=0.05, p=1.0)
    with pytest.raises(ValueError, match="greater than 1"):
        inverse_power_cutoff_mass(
            4.0,
            d_km2=25.0,
            q=1.0,
            gamma=1.0,
            mc=3.0,
            cutoff_radius_km=300.0,
        )
    with pytest.raises(ValueError, match="at least one"):
        aki_b_value([], mc=3.0)
    with pytest.raises(ValueError, match="greater than or equal"):
        aki_b_value([2.9, 3.0], mc=3.0)
    with pytest.raises(ValueError, match="positive"):
        truncated_gr_exp_expectation(alpha=1.0, beta=0.0, magnitude_span=4.0)

    parent = ETASParent(time_days=1.0, x_km=0.0, y_km=0.0, magnitude=4.0)
    with pytest.raises(ValueError, match="strictly earlier"):
        conditional_intensity(
            time_days=1.0,
            x_km=0.0,
            y_km=0.0,
            background_density_per_day_km2=0.0001,
            parents=[parent],
            mc=3.0,
            k=0.2,
            alpha=1.0,
            c_days=0.05,
            p=1.2,
            d_km2=25.0,
            q=1.5,
            gamma=1.0,
            spatial_cutoff_km=300.0,
        )


def test_external_oracle_metadata_is_content_addressed_and_unit_safe() -> None:
    oracle = _load_json(JAPAN_FIXTURE)

    assert oracle["production_dependency"] is False
    assert oracle["code_or_data_redistribution"] is False
    assert oracle["sources"]["tarball_sha256"] == (
        "53819f5043c4d33fe688b33a8f8e40d3cd69376613283468c18bbd4a67c43359"
    )
    assert oracle["sources"]["rda_sha256"] == (
        "873cf037a8754728980e0f765543f6151369b7757ea3f0041bbb06864892a4ec"
    )
    assert oracle["raw_dataset"]["rda_rows"] == 13724
    assert oracle["catalog_construction"]["constructed_revents_rows"] == 10072
    assert oracle["catalog_construction"]["target_event_rows"] == 4656
    assert oracle["units"]["D"] == "degree^2"
    assert oracle["expected"]["log_likelihood"] == -15310.96
    assert oracle["expected"]["aic"] == 30637.91
