"""Strict loader for the preregistered stage-2 background protocol."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from seismoflux.config import (
    load_config,
    load_yaml_mapping,
    normalize_relative_path,
    project_root_for,
    resolve_project_path,
    sha256_file,
)

EXPECTED_HORIZONS = (7, 30, 90, 180, 365)
EXPECTED_GRIDS_KM = (50.0, 25.0, 12.5)
EXPECTED_EQUAL_AREA_CRS = (
    "+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 +datum=WGS84 +units=m +no_defs +type=crs"
)
EXPECTED_BACKGROUND_PROTOCOL_CANONICAL_SHA256 = (
    "b36ce50f7f4df6d712c743bb28ce4f1fd05dfdbc3d5026b4bf75d00477765c6d"
)


class StrictModel(BaseModel):
    """Immutable protocol node that rejects undocumented keys."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def _relative_path(value: str) -> str:
    return normalize_relative_path(value)


class InputsConfig(StrictModel):
    environment_lock: str
    environment_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    data_catalog: str
    data_catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_stage1_snapshot_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    earthquake_dataset: str
    earthquake_dataset_path: str
    earthquake_dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    study_area: str
    study_area_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    issue_manifest: str
    issue_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    issue_manifest_source_dataset: str
    issue_manifest_source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    issue_manifest_column_allowlist: tuple[str, ...]
    issue_manifest_feature_values_forbidden: bool
    include_external_trigger_buffer_km: float = Field(gt=0)
    model_feature_allowlist: tuple[str, ...]
    forbidden_model_dataset_prefixes: tuple[str, ...]
    anomaly_schedule_metadata_exception: str

    _environment_lock_relative = field_validator("environment_lock")(_relative_path)
    _catalog_relative = field_validator("data_catalog")(_relative_path)
    _earthquake_dataset_relative = field_validator("earthquake_dataset_path")(_relative_path)
    _study_area_relative = field_validator("study_area")(_relative_path)
    _manifest_relative = field_validator("issue_manifest")(_relative_path)

    @model_validator(mode="after")
    def validate_feature_boundary(self) -> Self:
        if self.earthquake_dataset != "earthquake_event":
            raise ValueError("stage 2 may read only the frozen earthquake event dataset")
        if self.issue_manifest_source_dataset != "anomaly_report_period":
            raise ValueError("the issue calendar must come from anomaly_report_period metadata")
        if self.issue_manifest_column_allowlist != ("available_at",):
            raise ValueError("the issue manifest may read only available_at")
        if not self.issue_manifest_feature_values_forbidden:
            raise ValueError("anomaly feature values must remain forbidden")
        if self.model_feature_allowlist != (
            "origin_time_utc",
            "available_at",
            "longitude",
            "latitude",
            "magnitude",
            "inside_study_area",
        ):
            raise ValueError("background model feature allowlist differs from the frozen boundary")
        if self.forbidden_model_dataset_prefixes != ("anomaly_", "fault_", "basemap_"):
            raise ValueError(
                "background forbidden dataset prefixes differ from the frozen boundary"
            )
        return self


class ParameterSnapshotMappingConfig(StrictModel):
    historical_fold_assessment: Literal["fit_only_through_that_folds_fit_end_utc"]
    validation: Literal["fit_only_through_final_parameter_fit_end_utc"]
    actual_development_issue_dates: Literal["schedule_only_not_scored_by_stage2_final_snapshot"]
    online_history_update: Literal[
        "fixed_parameters_may_condition_on_events_available_after_fit_and_before_evaluation_time"
    ]


class TimeConfig(StrictModel):
    timezone: str
    issue_time_local: str
    forecast_interval: str
    catalog_start_utc: str
    final_parameter_fit_end_utc: Literal["2023-06-30T16:00:00Z"]
    purge_start_local: str
    purge_end_local: str
    validation_start_local: str
    validation_end_local: str
    validation_maturity_end_local: str
    representative_issue_date_local: str
    horizons_days: tuple[int, ...]
    issue_schedule: str
    issue_date_bounds_inclusive: bool
    event_availability_rule: str
    earthquake_publication_time_assumption: str
    publication_delay_sensitivity_days: tuple[int, ...]
    parameter_snapshot_mapping: ParameterSnapshotMappingConfig


class ParameterSelectionFoldConfig(StrictModel):
    id: Literal["fold_1", "fold_2", "fold_3", "fold_4"]
    fit_end_utc: str
    assessment_start_utc: str
    assessment_end_utc: str


class ParameterSelectionFoldsConfig(StrictModel):
    role: str
    selection_rule: str
    assessment_interval: str
    target_events_must_not_enter_fit: Literal[True]
    folds: tuple[ParameterSelectionFoldConfig, ...]

    @model_validator(mode="after")
    def validate_causal_folds(self) -> Self:
        if tuple(fold.id for fold in self.folds) != ("fold_1", "fold_2", "fold_3", "fold_4"):
            raise ValueError("background parameter folds must be fold_1 through fold_4 in order")
        previous_start: datetime | None = None
        for fold in self.folds:
            fit_end = datetime.fromisoformat(fold.fit_end_utc.replace("Z", "+00:00"))
            assessment_start = datetime.fromisoformat(
                fold.assessment_start_utc.replace("Z", "+00:00")
            )
            assessment_end = datetime.fromisoformat(fold.assessment_end_utc.replace("Z", "+00:00"))
            if assessment_start - fit_end < timedelta(days=365):
                raise ValueError("each background fold must retain at least a 365-day purge")
            if assessment_end <= assessment_start:
                raise ValueError("each background fold assessment must have positive duration")
            if previous_start is not None and assessment_start <= previous_start:
                raise ValueError("background fold assessments must be ordered in time")
            previous_start = assessment_start
        return self


class CompletenessCutoffMappingConfig(StrictModel):
    historical_fold_assessment: Literal["that_folds_fit_end_utc"]
    validation: Literal["2023-06-30T16:00:00Z"]


class TemporalDiagnosticConfig(StrictModel):
    block_anchor_utc: str
    block_years: int = Field(gt=0)
    minimum_events_per_block: int = Field(gt=0)
    final_partial_block_minimum_years: int = Field(gt=0)
    final_partial_block_rule: str
    regime_change_flag_mc_difference: float = Field(ge=0)
    regime_change_flag_annual_count_ratio: float = Field(gt=0)
    interpretation: str


class SpatialDiagnosticConfig(StrictModel):
    equal_area_cell_km: float = Field(gt=0)
    sparse_parent_cell_km: float = Field(gt=0)
    grid_origin_m: tuple[float, float]
    minimum_events_per_stratum: int = Field(gt=0)
    sparse_rule: str
    minimum_eligible_strata: int = Field(gt=0)
    indeterminate_strata_use_global_selected_mc: bool


class CompletenessConfig(StrictModel):
    method: str
    magnitude_bin_width: float = Field(gt=0)
    maximum_curvature_correction: float = Field(ge=0)
    candidate_magnitudes: tuple[float, ...]
    sensitivity_magnitudes: tuple[float, ...]
    selection_rule: str
    estimate_above_maximum_candidate_action: Literal["fail_stage"]
    no_eligible_temporal_or_spatial_strata_action: Literal["fail_stage"]
    target_event_domain: Literal["inside_study_area_only"]
    parent_buffer_events_excluded: Literal[True]
    cutoff_mapping: CompletenessCutoffMappingConfig
    temporal_diagnostic: TemporalDiagnosticConfig
    spatial_diagnostic: SpatialDiagnosticConfig
    gr_estimator: str
    magnitude_type_policy: str
    pre_1970_m5_catalog_for_completeness_forbidden: Literal[True]

    @model_validator(mode="after")
    def validate_frozen_candidates(self) -> Self:
        expected = (3.0, 3.2, 3.5, 4.0)
        if self.candidate_magnitudes != expected or self.sensitivity_magnitudes != expected:
            raise ValueError("completeness candidates must remain 3.0, 3.2, 3.5, and 4.0")
        return self


class UniformPoissonConfig(StrictModel):
    rate_estimator: str


class SpatialPoissonFittingCutoffMappingConfig(StrictModel):
    historical_fold_assessment: Literal["that_folds_fit_end_utc"]
    validation: Literal["2023-06-30T16:00:00Z"]


class SpatialPoissonConfig(StrictModel):
    method: Literal["equal_area_gaussian_kde"]
    mixture_boundary_normalization: Literal["normalize_complete_mixture_once_over_study_area"]
    gaussian_support: Literal["infinite"]
    rate_component: Literal["inside_training_event_count_over_training_days"]
    spatial_density_integral_over_study_area: float
    normalization_grid_km: float = Field(gt=0)
    normalization_quadrature_rule: Literal[
        "density_at_clipped_cell_representative_point_times_exact_clipped_area"
    ]
    convergence_failure_action: Literal["fail_model_and_G1_for_that_model"]
    bandwidth_candidates_km: tuple[float, ...]
    fold_score: Literal[
        "continuous_time_information_gain_per_physical_event_on_same_fold_events_and_compensator"
    ]
    best_bandwidth_rule: Literal["highest_equal_weight_mean_of_four_development_fold_scores"]
    paired_difference: Literal["candidate_minus_best_bandwidth_on_same_fold_events_and_compensator"]
    standard_error_formula: Literal["sample_stddev_of_four_paired_fold_differences_ddof1_div_sqrt4"]
    one_standard_error_eligibility: Literal[
        "candidate_mean_fold_score_gte_best_mean_fold_score_minus_paired_standard_error"
    ]
    selection_rule: Literal["largest_eligible_bandwidth_km"]
    exact_tie_rule: Literal["largest_bandwidth_km"]
    candidate_pool: Literal["only_bandwidths_passing_normalization_and_grid_convergence"]
    fitting_events: Literal["inside_study_area_only"]
    fitting_cutoff_mapping: SpatialPoissonFittingCutoffMappingConfig


class EtasBackgroundComponentConfig(StrictModel):
    spatial_density: Literal["selected_boundary_normalized_kde"]
    density_domain: Literal["study_area"]
    density_integral_over_study_area: float
    immigrant_rate_parameter: Literal["background_rate_per_day_inside_study_area"]
    exogenous_immigrants_outside_study_area: Literal[False]


class EtasLikelihoodConfig(StrictModel):
    target_events: Literal["magnitude_gte_selected_mc_and_inside_study_area_only"]
    parent_events: Literal["magnitude_gte_selected_mc_inside_study_area_or_within_300km_buffer"]
    external_parent_role: Literal[
        "conditional_history_only_never_target_or_background_training_event"
    ]
    compensator_domain: Literal["study_area_only"]
    assessment_history_update: Literal[
        "sequential_events_with_available_at_lte_evaluation_time_without_parameter_refit"
    ]
    background_normalization_grid_km: float = Field(gt=0)
    compensator_grid_km: float = Field(gt=0)
    quadrature_rule: Literal[
        "density_at_clipped_cell_representative_point_times_exact_clipped_area"
    ]
    convergence_failure_action: Literal["fail_model_and_G1_for_that_model"]


class EtasTemporalKernelConfig(StrictModel):
    form: Literal["normalized_infinite_support_omori_utsu"]
    history_parent_cutoff_days: float = Field(gt=0)
    cutoff_role: Literal["omit_older_history_parents_without_kernel_renormalization"]


class EtasSpatialKernelConfig(StrictModel):
    form: Literal["normalized_truncated_isotropic_inverse_power"]
    support_radius_km: float = Field(gt=0)
    renormalize_radially_on_support: Literal[True]
    d_km2: float = Field(gt=0)
    q: float = Field(gt=1)
    gamma: float = Field(gt=0)
    sensitivity_d_km2: tuple[float, ...]


class EtasMagnitudeModelConfig(StrictModel):
    family: str
    lower_magnitude: str
    upper_magnitude: float
    interval: Literal["closed"]
    observed_magnitude_above_upper_action: Literal["fail_stage"]
    b_estimator: str
    effective_lower_edge: str
    b_formula: str
    beta_definition: str
    density: str
    productivity_expectation: str


class EtasBranchingRatioConfig(StrictModel):
    formula: str
    expectation_if_alpha_ne_beta: str
    expectation_if_abs_alpha_minus_beta_lte_1e_12: str
    equality_tolerance: float = Field(gt=0)
    temporal_kernel_total_mass: str
    spatial_kernel_total_mass: str
    maximum: float = Field(gt=0, lt=1)
    violation_action: Literal["reject_fit"]


class EtasParameterBoundsConfig(StrictModel):
    background_rate_per_day: tuple[float, float]
    productivity_k: tuple[float, float]
    alpha: tuple[float, float]
    c_days: tuple[float, float]
    p: tuple[float, float]


class EtasOptimizerParameterizationConfig(StrictModel):
    variables: tuple[str, ...]
    transform: str
    transformed_bounds: str
    start_generation: Literal[
        "sha256_optimizer_start_namespace_then_PCG64_uniform_on_transformed_bounds_for_each_index"
    ]
    objective_dtype: Literal["float64"]

    @model_validator(mode="after")
    def validate_variables(self) -> Self:
        if self.variables != (
            "background_rate_per_day",
            "productivity_k",
            "alpha",
            "c_days",
            "p_minus_one",
        ):
            raise ValueError("ETAS optimizer variables differ from the frozen order")
        return self


class EtasOptimizerOptionsConfig(StrictModel):
    scipy_tol: None
    ftol: float = Field(gt=0)
    gtol: float = Field(gt=0)
    maxiter: Literal[500]
    maxfun: Literal[100000]
    maxls: Literal[20]
    gradient_source: Literal[
        "numerical_three_point_objective_including_background_trigger_and_compensator"
    ]
    gradient_relative_step: float = Field(gt=0)
    invalid_physical_or_branching_point_objective: Literal["positive_infinity"]
    gradient_invalid_stencil: Literal["second_order_one_sided_inward_or_fail_start"]


class EtasNumericalStabilityConfig(StrictModel):
    minimum_converged_starts: Literal[4]
    gradient_infinity_norm_maximum: float = Field(gt=0)
    best_three_relative_objective_range_maximum: float = Field(gt=0)
    best_three_relative_objective_denominator: Literal["max_1_abs_best_objective"]
    transformed_parameter_maximum_range: float = Field(gt=0)
    transformed_parameter_range_statistic: Literal[
        "maximum_over_parameters_of_max_minus_min_across_best_three_transformed_vectors"
    ]
    hessian_minimum_eigenvalue: float = Field(gt=0)
    hessian_condition_number_maximum: float = Field(gt=0)
    hessian_method: str
    hessian_relative_step: float = Field(gt=0)
    hessian_step_formula: str
    hessian_symmetrize: Literal[True]
    hessian_invalid_stencil_action: Literal["fail_etas_stability_without_one_sided_substitution"]
    hessian_condition_number_definition: Literal[
        "largest_eigenvalue_divided_by_smallest_eigenvalue_of_symmetric_positive_definite_hessian"
    ]
    finite_nonnegative_intensity_required: Literal[True]
    failure_action: str


class EtasParameterUncertaintyConfig(StrictModel):
    method: str
    confidence_level: float = Field(gt=0, lt=1)
    non_positive_definite_hessian_action: str


class EtasSimulationConfig(StrictModel):
    target_domain: Literal["study_area_only"]
    propagation_domain: Literal["study_area_plus_300km_buffer_with_absorbing_outer_boundary"]
    background_immigrant_domain: Literal["study_area_only"]
    retain_buffer_descendants_as_parents: Literal[True]
    buffer_events_in_outputs_or_metrics: Literal[False]


class EtasConfig(StrictModel):
    family: str
    implementation: str
    direct_external_code_copy_forbidden: Literal[True]
    final_fit_start_utc: Literal["2000-01-01T00:00:00Z"]
    historical_fold_fit_start_utc: Literal["2000-01-01T00:00:00Z"]
    history_start_utc: Literal["1970-01-01T00:00:00Z"]
    background_component: EtasBackgroundComponentConfig
    likelihood: EtasLikelihoodConfig
    temporal_kernel: EtasTemporalKernelConfig
    spatial_kernel: EtasSpatialKernelConfig
    magnitude_model: EtasMagnitudeModelConfig
    branching_ratio: EtasBranchingRatioConfig
    parameter_bounds: EtasParameterBoundsConfig
    multi_start_indices: tuple[int, ...]
    optimizer: str
    optimizer_parameterization: EtasOptimizerParameterizationConfig
    optimizer_options: EtasOptimizerOptionsConfig
    maximum_iterations: int = Field(gt=0)
    numerical_stability: EtasNumericalStabilityConfig
    parameter_uncertainty: EtasParameterUncertaintyConfig
    future_simulation_replicates: int = Field(gt=0)
    future_descendant_generations_included: Literal[True]
    simulation: EtasSimulationConfig


class OptimizerStartSeedContextConfig(StrictModel):
    model_ids: tuple[str, ...]
    issue_id_or_dash: Literal["-"]
    replicate_index_role: Literal["optimizer_start_index"]
    replicate_index_first: Literal[0]
    replicate_index_last_inclusive: Literal[4]


class FutureSimulationSeedContextConfig(StrictModel):
    model_id: Literal["etas/final_validation"]
    issue_id_rule: Literal["validation_slash_actual_issue_date_yyyy_mm_dd_from_frozen_manifest"]
    replicate_index_role: Literal["future_catalog_replicate"]
    replicate_index_first: Literal[0]
    replicate_index_last_inclusive: Literal[127]


class SimulationRegressionSeedContextConfig(StrictModel):
    model_id: Literal["etas_inverse_power_cut300_v1"]
    issue_id_or_dash: Literal["-"]
    replicate_index_role: Literal["analytic_fixture_replicate"]
    replicate_index_first: Literal[0]
    replicate_index_last_inclusive: Literal[8191]


class BootstrapSeedContextConfig(StrictModel):
    model_ids: tuple[str, ...]
    issue_id: Literal["g1_primary_validation_2024-07-01_2025-07-01"]
    replicate_index_role: Literal["bootstrap_replicate"]
    replicate_index_first: Literal[0]
    replicate_index_last_inclusive: Literal[1999]


class SeedNamespaceContextsConfig(StrictModel):
    optimizer_start: OptimizerStartSeedContextConfig
    future_simulation: FutureSimulationSeedContextConfig
    simulation_regression: SimulationRegressionSeedContextConfig
    bootstrap: BootstrapSeedContextConfig

    @model_validator(mode="after")
    def validate_model_ids(self) -> Self:
        if self.bootstrap.model_ids != (
            "spatial_poisson_vs_uniform_poisson",
            "etas_vs_uniform_poisson",
        ):
            raise ValueError("bootstrap seed model IDs differ from the frozen comparisons")
        return self


class SeedDerivationConfig(StrictModel):
    encoding: str
    separator_hex: str
    ordered_fields: tuple[str, ...]
    digest: str
    entropy: str
    generator: str
    namespaces: tuple[str, ...]
    worker_count_invariant: bool
    gather_order: str
    reference_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_entropy_integer: int = Field(gt=0)
    namespace_contexts: SeedNamespaceContextsConfig

    @model_validator(mode="after")
    def validate_seed_contract(self) -> Self:
        if self.ordered_fields != (
            "literal_seismoflux",
            "root_seed_decimal",
            "protocol_version",
            "namespace",
            "model_id",
            "issue_id_or_dash",
            "replicate_index_decimal_8",
        ):
            raise ValueError("seed derivation fields differ from the frozen order")
        if self.namespaces != (
            "optimizer_start",
            "future_simulation",
            "simulation_regression",
            "bootstrap",
        ):
            raise ValueError("seed derivation namespaces differ from the frozen set")
        return self


class FutureSimulationRandomnessConfig(StrictModel):
    longest_horizon_days: int = Field(gt=0)
    simulate_once_per_replicate: Literal[True]
    reuse_same_catalog_for_horizons_days: tuple[int, ...]
    reuse_same_catalog_for_grid_cells_km: tuple[float, ...]
    maximum_events_per_replicate: int = Field(gt=0)
    event_cap_hit_is_failure: Literal[True]
    stable_event_order: tuple[str, ...]


class RandomnessConfig(StrictModel):
    root_seed: int = Field(ge=0)
    bit_generator: Literal["numpy.random.PCG64"]
    float_dtype: Literal["float64"]
    python_hash_forbidden: Literal[True]
    global_rng_forbidden: Literal[True]
    seed_derivation: SeedDerivationConfig
    future_simulation: FutureSimulationRandomnessConfig


class BackgroundIntegrationConfig(StrictModel):
    equal_area_crs: str
    grid_cells_km: tuple[float, ...]
    primary_grid_cell_km: float = Field(gt=0)
    grid_origin_m: tuple[float, float]
    cell_index_formula: str
    cell_bounds: str
    cell_order: str
    cell_id_template: str
    inclusion_rule: str
    boundary_rule: str
    quadrature_point: str
    cell_mass_formula: str
    refinement_rule: str
    primary_convergence_pair_km: tuple[float, float]
    diagnostic_convergence_pair_km: tuple[float, float]
    convergence_common_parent_rule: str
    relative_expected_count_formula: str
    density_l1_formula: str
    relative_expected_count_tolerance: float = Field(gt=0)
    density_l1_tolerance: float = Field(gt=0)
    relative_denominator_floor: float = Field(gt=0)
    both_totals_below_floor_relative_difference: float = Field(ge=0)
    target_independent_grid: Literal[True]
    convergence_failure_action: Literal["fail_model_and_G1_for_that_model"]


class G1PrimaryEndpointConfig(StrictModel):
    type: Literal["continuous_time_sequential_conditional_point_process_log_likelihood"]
    validation_interval_local: Literal["(2024-07-01T00:00:00,2025-07-01T00:00:00]"]
    validation_parameter_snapshot: Literal["final_fit_through_2023-06-30T16:00:00Z"]
    validation_history_rule: Literal[
        "update_with_available_events_before_each_event_time_without_refitting"
    ]
    horizon_aggregation: Literal["none"]
    score_formula: Literal[
        "nonuniform_log_likelihood_minus_uniform_log_likelihood_divided_by_"
        "inside_physical_target_event_count"
    ]
    zero_target_event_action: Literal["evidence_insufficient"]
    development_folds: Literal["parameter_selection_folds"]
    development_fold_score: Literal[
        "same_continuous_time_score_over_each_folds_assessment_interval"
    ]
    development_fold_summary: Literal["equal_weight_mean_and_sample_standard_error_ddof1"]


class IssueBasedHorizonBacktestsConfig(StrictModel):
    role: str
    partitions: tuple[str, ...]
    actual_development_dates_role: str
    issue_dates_source: str
    exposure_selection: str
    retain_zero_event_exposures: bool
    zero_event_exposure_contribution: str
    denominator: str
    zero_total_target_event_action: str
    event_multiplicity_per_partition_horizon: str
    cross_horizon_aggregation: str


class G1PassRuleConfig(StrictModel):
    eligible_nonuniform_models: tuple[Literal["spatial_poisson", "etas"], ...]
    conjunction_unit: Literal["same_nonuniform_model_must_satisfy_every_requirement"]
    any_nonuniform_model_validation_information_gain_gt_zero: Literal[True]
    minimum_positive_development_folds: Literal[3]
    required_development_folds: Literal[4]
    development_folds_reference: Literal["parameter_selection_folds"]
    model_must_be_numerically_stable: Literal[True]

    @model_validator(mode="after")
    def validate_eligible_models(self) -> Self:
        if self.eligible_nonuniform_models != ("spatial_poisson", "etas"):
            raise ValueError("G1 eligible nonuniform models must remain spatial Poisson and ETAS")
        return self


class ModelSelectionConfig(StrictModel):
    eligible_model_pool: Literal[
        "only_models_with_successful_fit_and_all_model_specific_numerical_stability_and_grid_convergence_gates"
    ]
    failed_model_role: Literal[
        "excluded_from_validation_best_one_standard_error_threshold_and_final_selection"
    ]
    validation_best_rule: Literal["highest_validation_information_gain_per_physical_event"]
    simplicity_order: tuple[Literal["uniform_poisson", "spatial_poisson", "etas"], ...]
    paired_difference: Literal[
        "candidate_minus_validation_best_on_same_fold_events_and_compensator"
    ]
    standard_error_formula: Literal["sample_stddev_of_four_paired_fold_differences_ddof1_div_sqrt4"]
    one_standard_error_eligibility: Literal[
        "candidate_validation_ig_gte_best_validation_ig_minus_paired_standard_error"
    ]
    final_rule: Literal["first_eligible_model_in_simplicity_order"]

    @model_validator(mode="after")
    def validate_simplicity_order(self) -> Self:
        if self.simplicity_order != ("uniform_poisson", "spatial_poisson", "etas"):
            raise ValueError("background model simplicity order is frozen")
        return self


class EvaluationConfig(StrictModel):
    primary_target: Literal[
        "unmarked_events_at_or_above_selected_completeness_magnitude_inside_study_area_only"
    ]
    information_gain_unit: Literal["nats_per_physical_event"]
    includes_integrated_intensity_compensator: Literal[True]
    g1_primary_endpoint: G1PrimaryEndpointConfig
    issue_based_horizon_backtests: IssueBasedHorizonBacktestsConfig
    bootstrap_replications: int = Field(gt=0)
    bootstrap_seed_source: Literal["sha256_bootstrap_namespace"]
    confidence_level: float = Field(gt=0, lt=1)
    g1_pass_rule: G1PassRuleConfig
    model_selection: ModelSelectionConfig
    target_magnitude_bins_role: str
    minimum_target_events_for_non_exploratory_claim: int = Field(gt=0)
    insufficient_nonoverlapping_exposures_action: str


class JapanReferenceConfig(StrictModel):
    total_events: int = Field(gt=0)
    target_events: int = Field(gt=0)
    log_likelihood: float
    aic: float
    parameter_max_absolute_tolerance: float = Field(gt=0)
    likelihood_absolute_tolerance: float = Field(gt=0)


class AnalyticMagnitudeDistributionConfig(StrictModel):
    family: str
    lower_magnitude: float
    upper_magnitude: float
    beta: float = Field(gt=0)


class AnalyticRootEventConfig(StrictModel):
    time_days: float
    x_km: float
    y_km: float
    magnitude: float


class AnalyticSimulationConfig(StrictModel):
    fixture_id: str
    replicate_count: int = Field(gt=0)
    magnitude_distribution: AnalyticMagnitudeDistributionConfig
    root_event: AnalyticRootEventConfig
    simulation_end: str
    temporal_horizon_days: float | None
    spatial_cutoff_km: float = Field(gt=0)
    expected_mean_exp_alpha_delta_m: float
    expected_generic_branching_ratio: float
    expected_generic_offspring_variance: float
    expected_root_direct_offspring_mean: float
    expected_root_total_descendants_mean: float
    expected_root_total_descendants_variance: float
    expected_root_zero_descendant_probability: float
    descendant_mean_absolute_tolerance: float = Field(gt=0)
    descendant_variance_relative_tolerance: float = Field(gt=0)
    zero_descendant_probability_absolute_tolerance: float = Field(gt=0)
    direct_offspring_pit_ks_maximum: float = Field(gt=0)
    minimum_pooled_direct_offspring: int = Field(gt=0)
    event_cap_hits_required: int = Field(ge=0)
    nonfinite_values_required: int = Field(ge=0)


class ExternalSimulationScenarioConfig(StrictModel):
    fixture_id: str
    temporal_horizon_days: int = Field(gt=0)
    include_all_descendant_generations: bool
    comparison_requires_adapted_common_kernel_conventions: bool


class ExternalSimulationOracleConfig(StrictModel):
    execution: str
    production_dependency: bool
    replicate_count_per_implementation: int = Field(gt=0)
    scenario: ExternalSimulationScenarioConfig
    sampling_for_continuous_comparison: str
    statistics: tuple[str, ...]
    descendant_count_two_sample_ks_maximum: float = Field(gt=0)
    descendant_mean_relative_tolerance: float = Field(gt=0)
    descendant_mean_absolute_floor: float = Field(gt=0)
    descendant_variance_relative_tolerance: float = Field(gt=0)
    probability_absolute_tolerance: float = Field(gt=0)
    continuous_two_sample_ks_maximum: float = Field(gt=0)
    event_cap_hits_required: int = Field(ge=0)
    nonfinite_values_required: int = Field(ge=0)
    unavailable_external_oracle_action: str


class NumericalRegressionConfig(StrictModel):
    production_fixture: str
    production_fixture_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    production_scalar_absolute_tolerance: float = Field(gt=0)
    production_scalar_relative_tolerance: float = Field(gt=0)
    production_log_likelihood_absolute_tolerance: float = Field(gt=0)
    authoritative_oracle: str
    oracle_metadata: str
    oracle_metadata_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    oracle_execution: str
    oracle_code_copy_forbidden: bool
    japan_reference: JapanReferenceConfig
    simulation_oracle: str
    cross_implementation_comparison: str
    analytic_simulation: AnalyticSimulationConfig
    external_simulation_oracle: ExternalSimulationOracleConfig

    _fixture_relative = field_validator("production_fixture")(_relative_path)
    _oracle_relative = field_validator("oracle_metadata")(_relative_path)


class CanonicalJsonConfig(StrictModel):
    mapping_keys: str
    string_values: str
    integers: str
    floats: str
    nonfinite_float_action: str
    mapping_separators: str
    encoding: str
    project_paths: str


class WriteProtocolConfig(StrictModel):
    staging: str
    file_manifest_fields: tuple[str, ...]
    durability: str
    publish: str
    existing_identical_id_action: str


class OutputsConfig(StrictModel):
    content_address_algorithm: str
    canonical_json: CanonicalJsonConfig
    content_address_inputs: tuple[str, ...]
    input_hashes_required_keys: tuple[str, ...]
    existing_id_different_bytes_action: str
    write_protocol: WriteProtocolConfig
    processed_root: str
    backtest_root: str
    experiment_root: str
    model_root: str
    registry: str
    fold_manifest: str
    report: str

    @field_validator(
        "processed_root",
        "backtest_root",
        "experiment_root",
        "model_root",
        "registry",
        "fold_manifest",
        "report",
    )
    @classmethod
    def validate_output_path(cls, value: str) -> str:
        return _relative_path(value)


class BackgroundConfig(StrictModel):
    """Complete, frozen stage-2 background protocol."""

    schema_version: Literal[1]
    protocol_version: Literal["0.2.0"]
    data_contract_version: Literal["0.1.0"]
    status: Literal["preregistered_before_any_background_score"]
    frozen_before_background_scores: Literal[True]
    frozen_on: str
    freeze_tag: Literal["v0.2.0-background-protocol"]
    background_scores_seen_before_freeze: Literal[False]
    inputs: InputsConfig
    time: TimeConfig
    parameter_selection_folds: ParameterSelectionFoldsConfig
    completeness: CompletenessConfig
    uniform_poisson: UniformPoissonConfig
    spatial_poisson: SpatialPoissonConfig
    etas: EtasConfig
    randomness: RandomnessConfig
    integration: BackgroundIntegrationConfig
    evaluation: EvaluationConfig
    numerical_regression: NumericalRegressionConfig
    outputs: OutputsConfig

    @model_validator(mode="after")
    def validate_internal_contract(self) -> Self:
        if self.time.horizons_days != EXPECTED_HORIZONS:
            raise ValueError("background horizons must be exactly 7, 30, 90, 180, and 365 days")
        if self.integration.grid_cells_km != EXPECTED_GRIDS_KM:
            raise ValueError("background grids must be exactly 50, 25, and 12.5 km")
        if self.integration.primary_grid_cell_km != 25.0:
            raise ValueError("the primary background grid must be 25 km")
        if self.integration.primary_convergence_pair_km != (25.0, 12.5):
            raise ValueError("primary grid convergence must compare 25 km with 12.5 km")
        if self.integration.diagnostic_convergence_pair_km != (50.0, 25.0):
            raise ValueError("diagnostic grid convergence must compare 50 km with 25 km")
        if self.integration.equal_area_crs != EXPECTED_EQUAL_AREA_CRS:
            raise ValueError("background equal-area CRS differs from the frozen project CRS")
        finest_grid = min(self.integration.grid_cells_km)
        if self.spatial_poisson.spatial_density_integral_over_study_area != 1.0:
            raise ValueError("spatial Poisson density must integrate to one over the study area")
        if self.spatial_poisson.normalization_grid_km != finest_grid:
            raise ValueError("KDE normalization must use the frozen finest integration grid")
        if self.etas.background_component.density_integral_over_study_area != 1.0:
            raise ValueError("ETAS background density must integrate to one over the study area")
        if (
            self.etas.likelihood.background_normalization_grid_km != finest_grid
            or self.etas.likelihood.compensator_grid_km != finest_grid
        ):
            raise ValueError("ETAS background and compensator must use the finest grid")
        if self.etas.temporal_kernel.history_parent_cutoff_days != 3650.0:
            raise ValueError("ETAS history-parent cutoff must remain 3650 days")
        if self.etas.spatial_kernel.support_radius_km != (
            self.inputs.include_external_trigger_buffer_km
        ):
            raise ValueError("ETAS spatial cutoff must equal the external trigger buffer")
        if self.etas.magnitude_model.upper_magnitude != 9.5:
            raise ValueError("ETAS upper magnitude must remain 9.5")
        if self.etas.branching_ratio.equality_tolerance != 1.0e-12:
            raise ValueError("ETAS alpha-beta equality tolerance must remain 1e-12")
        if self.etas.branching_ratio.maximum != 0.95:
            raise ValueError("ETAS branching-ratio maximum must remain 0.95")
        if self.etas.multi_start_indices != (0, 1, 2, 3, 4):
            raise ValueError("ETAS optimizer starts must remain indices 0 through 4")
        if self.etas.numerical_stability.minimum_converged_starts != 4:
            raise ValueError("ETAS stability requires four of five converged starts")
        if self.etas.optimizer_options.maxiter != self.etas.maximum_iterations:
            raise ValueError("ETAS optimizer iteration limits disagree")
        if self.etas.optimizer_options.ftol != 1.0e-12:
            raise ValueError("ETAS optimizer ftol must remain 1e-12")
        if self.etas.optimizer_options.gtol != 1.0e-6:
            raise ValueError("ETAS optimizer gtol must remain 1e-6")
        if self.etas.optimizer_options.gradient_relative_step != 1.0e-6:
            raise ValueError("ETAS optimizer gradient step must remain 1e-6")
        if "optimizer_start" not in self.randomness.seed_derivation.namespaces:
            raise ValueError("ETAS optimizer starts must use the SHA-256 seed namespace")
        if "bootstrap" not in self.randomness.seed_derivation.namespaces:
            raise ValueError("bootstrap must use the SHA-256 seed namespace")
        seed_contexts = self.randomness.seed_derivation.namespace_contexts
        expected_optimizer_model_ids = (
            *(f"etas/{fold.id}" for fold in self.parameter_selection_folds.folds),
            "etas/final_validation",
        )
        if seed_contexts.optimizer_start.model_ids != expected_optimizer_model_ids:
            raise ValueError("optimizer seed model IDs must match every fold and the final fit")
        if (
            seed_contexts.future_simulation.replicate_index_last_inclusive + 1
            != self.etas.future_simulation_replicates
        ):
            raise ValueError("future-simulation seed index range disagrees with replicate count")
        if (
            seed_contexts.simulation_regression.replicate_index_last_inclusive + 1
            != self.numerical_regression.analytic_simulation.replicate_count
        ):
            raise ValueError(
                "simulation-regression seed index range disagrees with replicate count"
            )
        if (
            seed_contexts.bootstrap.replicate_index_last_inclusive + 1
            != self.evaluation.bootstrap_replications
        ):
            raise ValueError("bootstrap seed index range disagrees with replicate count")
        if self.randomness.future_simulation.reuse_same_catalog_for_horizons_days != (
            self.time.horizons_days
        ):
            raise ValueError("simulation horizons must match the frozen forecast horizons")
        if self.randomness.future_simulation.reuse_same_catalog_for_grid_cells_km != (
            self.integration.grid_cells_km
        ):
            raise ValueError("simulation grids must match the frozen integration grids")
        if self.randomness.future_simulation.longest_horizon_days != max(self.time.horizons_days):
            raise ValueError("the simulation horizon must cover the longest forecast horizon")
        if self.outputs.fold_manifest != self.inputs.issue_manifest:
            raise ValueError("input and output references must identify one frozen fold manifest")
        if self.outputs.content_address_inputs != (
            "protocol",
            "input_hashes",
            "model_parameters",
            "code_commit",
            "uv_lock_sha256",
        ):
            raise ValueError("content-address inputs differ from the frozen protocol")
        if self.outputs.input_hashes_required_keys != (
            "environment_lock",
            "data_catalog",
            "earthquake_dataset",
            "study_area",
            "issue_manifest",
            "production_fixture",
            "oracle_metadata",
        ):
            raise ValueError("required content-address input hashes differ from the frozen set")
        if (
            self.numerical_regression.analytic_simulation.spatial_cutoff_km
            != self.etas.spatial_kernel.support_radius_km
        ):
            raise ValueError("analytic regression cutoff must match the ETAS spatial cutoff")
        fold_count = len(self.parameter_selection_folds.folds)
        if self.evaluation.g1_pass_rule.required_development_folds != fold_count:
            raise ValueError("G1 must require all preregistered development folds")
        if (
            self.evaluation.g1_pass_rule.minimum_positive_development_folds
            > self.evaluation.g1_pass_rule.required_development_folds
        ):
            raise ValueError("positive development fold threshold exceeds available folds")
        canonical_payload = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        canonical_sha256 = hashlib.sha256(canonical_payload).hexdigest()
        if canonical_sha256 != EXPECTED_BACKGROUND_PROTOCOL_CANONICAL_SHA256:
            raise ValueError("background protocol differs from its frozen canonical fingerprint")
        return self


def _load_json_mapping(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read JSON metadata: {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"JSON metadata root must be a mapping: {path}")
    return cast(dict[str, Any], raw)


def _require_equal(actual: object, expected: object, description: str) -> None:
    if actual != expected:
        raise ValueError(f"{description} does not match the frozen background protocol")


def _require_file_sha256(path: Path, expected: str, description: str) -> None:
    if not path.is_file():
        raise ValueError(f"{description} file is missing: {path}")
    _require_equal(sha256_file(path), expected, f"{description} SHA-256")


def _validate_referenced_metadata(config_path: Path, config: BackgroundConfig) -> None:
    environment_lock_path = resolve_project_path(config_path, config.inputs.environment_lock)
    catalog_path = resolve_project_path(config_path, config.inputs.data_catalog)
    study_area_path = resolve_project_path(config_path, config.inputs.study_area)
    manifest_path = resolve_project_path(config_path, config.inputs.issue_manifest)
    earthquake_path = resolve_project_path(config_path, config.inputs.earthquake_dataset_path)
    fixture_path = resolve_project_path(config_path, config.numerical_regression.production_fixture)
    oracle_path = resolve_project_path(config_path, config.numerical_regression.oracle_metadata)

    _require_equal(
        sha256_file(environment_lock_path),
        config.inputs.environment_lock_sha256,
        "environment lock SHA-256",
    )
    _require_equal(
        sha256_file(catalog_path),
        config.inputs.data_catalog_sha256,
        "data catalog SHA-256",
    )
    _require_file_sha256(
        study_area_path,
        config.inputs.study_area_sha256,
        "study-area",
    )
    _require_equal(
        sha256_file(manifest_path),
        config.inputs.issue_manifest_sha256,
        "issue manifest SHA-256",
    )
    _require_equal(
        sha256_file(fixture_path),
        config.numerical_regression.production_fixture_sha256,
        "production fixture SHA-256",
    )
    _require_equal(
        sha256_file(oracle_path),
        config.numerical_regression.oracle_metadata_sha256,
        "oracle metadata SHA-256",
    )

    catalog = _load_json_mapping(catalog_path)
    manifest = _load_json_mapping(manifest_path)
    datasets = catalog.get("datasets")
    study_area = catalog.get("study_area")
    source = manifest.get("source")
    if not isinstance(datasets, dict) or not isinstance(study_area, dict):
        raise ValueError("data catalog is missing required stage-1 metadata")
    if not isinstance(source, dict):
        raise ValueError("background fold manifest is missing source metadata")

    source_dataset = datasets.get(config.inputs.issue_manifest_source_dataset)
    earthquake_dataset = datasets.get(config.inputs.earthquake_dataset)
    if not isinstance(source_dataset, dict):
        raise ValueError("issue-manifest source dataset is absent from the data catalog")
    if not isinstance(earthquake_dataset, dict):
        raise ValueError("earthquake event dataset is absent from the data catalog")
    _require_equal(
        catalog.get("snapshot_id"),
        config.inputs.expected_stage1_snapshot_id,
        "stage-1 snapshot ID",
    )
    _require_equal(
        source_dataset.get("file_sha256"),
        config.inputs.issue_manifest_source_sha256,
        "issue-manifest source SHA-256",
    )
    _require_equal(
        earthquake_dataset.get("path"),
        config.inputs.earthquake_dataset_path,
        "earthquake dataset path",
    )
    _require_equal(
        earthquake_dataset.get("file_sha256"),
        config.inputs.earthquake_dataset_sha256,
        "earthquake dataset SHA-256",
    )
    _require_file_sha256(
        earthquake_path,
        config.inputs.earthquake_dataset_sha256,
        "local earthquake dataset",
    )
    _require_equal(
        study_area.get("sha256"),
        config.inputs.study_area_sha256,
        "catalog study-area SHA-256",
    )
    _require_equal(
        source.get("data_catalog_path"), config.inputs.data_catalog, "manifest catalog path"
    )
    _require_equal(
        source.get("data_catalog_sha256"),
        config.inputs.data_catalog_sha256,
        "manifest catalog SHA-256",
    )
    _require_equal(
        source.get("study_area_path"), config.inputs.study_area, "manifest study-area path"
    )
    _require_equal(
        source.get("study_area_sha256"),
        config.inputs.study_area_sha256,
        "manifest study-area SHA-256",
    )
    _require_equal(
        source.get("anomaly_report_period_sha256"),
        config.inputs.issue_manifest_source_sha256,
        "manifest source-dataset SHA-256",
    )
    source_dataset_path_value = source.get("anomaly_report_period_path")
    if not isinstance(source_dataset_path_value, str):
        raise ValueError("manifest source-dataset path is missing")
    source_dataset_path = resolve_project_path(config_path, source_dataset_path_value)
    _require_equal(
        source_dataset_path_value,
        source_dataset.get("path"),
        "manifest source-dataset path",
    )
    _require_file_sha256(
        source_dataset_path,
        config.inputs.issue_manifest_source_sha256,
        "source dataset",
    )
    _require_equal(
        source.get("stage1_snapshot_id"),
        config.inputs.expected_stage1_snapshot_id,
        "manifest stage-1 snapshot ID",
    )
    _require_equal(manifest.get("freeze_tag"), config.freeze_tag, "manifest freeze tag")
    _require_equal(
        manifest.get("scores_seen_before_freeze"),
        config.background_scores_seen_before_freeze,
        "manifest pre-freeze score state",
    )
    _require_equal(source.get("column_read"), "available_at", "manifest source column")
    _require_equal(
        source.get("all_other_anomaly_columns_forbidden"),
        True,
        "manifest anomaly-feature prohibition",
    )

    partitions = manifest.get("partitions")
    if not isinstance(partitions, dict):
        raise ValueError("background fold manifest is missing partitions")
    validation = partitions.get("validation")
    if not isinstance(validation, dict):
        raise ValueError("background fold manifest is missing the validation partition")
    issue_dates = validation.get("actual_issue_dates_local")
    if not isinstance(issue_dates, list) or not issue_dates:
        raise ValueError("validation issue-date manifest is empty")
    _require_equal(
        issue_dates[-1],
        config.time.representative_issue_date_local,
        "representative validation issue date",
    )


def load_background_config(path: str | Path) -> BackgroundConfig:
    """Load the stage-2 protocol and verify all content-addressed inputs."""

    config_path = Path(path)
    config = BackgroundConfig.model_validate(load_yaml_mapping(config_path))
    _validate_referenced_metadata(config_path, config)
    return config


def load_project_background_config(
    config_path: str | Path = Path("configs/base.yaml"),
) -> BackgroundConfig:
    """Load the project and its background protocol, enforcing shared invariants."""

    main_path = Path(config_path)
    project = load_config(main_path)
    background_path = resolve_project_path(main_path, project.config_files.background)
    background = load_background_config(background_path)

    _require_equal(
        background.randomness.root_seed,
        project.project.random_seed,
        "background root seed",
    )
    _require_equal(
        background.integration.equal_area_crs,
        project.study_area.equal_area_crs,
        "background equal-area CRS",
    )
    _require_equal(
        background.inputs.include_external_trigger_buffer_km,
        project.study_area.include_external_trigger_buffer_km,
        "background external trigger buffer",
    )
    _require_equal(
        background.time.horizons_days,
        project.forecast.horizons_days,
        "background forecast horizons",
    )
    _require_equal(
        background.integration.grid_cells_km,
        project.integration.convergence_cells_km,
        "background integration grids",
    )

    expected_root = project_root_for(main_path).resolve()
    actual_root = project_root_for(background_path).resolve()
    _require_equal(actual_root, expected_root, "background project root")
    return background
