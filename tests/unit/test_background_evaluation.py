from __future__ import annotations

import numpy as np
import pytest

from seismoflux.background.evaluation import (
    BackgroundModelEvidence,
    FoldInformationGain,
    InformationGainContributions,
    ModelId,
    assess_g1,
    bootstrap_information_gain,
    select_background_model,
)


def _evidence(
    model_id: ModelId,
    *,
    validation_ig: float,
    fold_igs: tuple[float, float, float, float],
    fit: bool = True,
    stable: bool = True,
    converged: bool = True,
) -> BackgroundModelEvidence:
    uniform_log_likelihood = -100.0
    event_count = 10
    folds = tuple(
        FoldInformationGain.build(
            fold_id=f"fold_{index + 1}",
            physical_event_count=event_count,
            candidate_log_likelihood=uniform_log_likelihood + ig * event_count,
            uniform_log_likelihood=uniform_log_likelihood,
        )
        for index, ig in enumerate(fold_igs)
    )
    return BackgroundModelEvidence.build(
        model_id=model_id,
        validation_physical_event_count=event_count,
        validation_log_likelihood=uniform_log_likelihood + validation_ig * event_count,
        validation_uniform_log_likelihood=uniform_log_likelihood,
        development_folds=folds,
        fit_succeeded=fit,
        numerically_stable=stable,
        grid_converged=converged,
    )


def test_g1_requires_one_same_nonuniform_model_to_pass_every_gate() -> None:
    uniform = _evidence("uniform_poisson", validation_ig=0.0, fold_igs=(0.0, 0.0, 0.0, 0.0))
    spatial = _evidence(
        "spatial_poisson",
        validation_ig=0.1,
        fold_igs=(0.2, 0.1, 0.05, -0.01),
    )
    etas = _evidence(
        "etas",
        validation_ig=0.5,
        fold_igs=(0.5, 0.4, 0.3, 0.2),
        stable=False,
    )

    assessment = assess_g1((uniform, spatial, etas))

    assert assessment.passed is True
    assert assessment.passing_models == ("spatial_poisson",)
    assert dict(assessment.model_pass) == {"spatial_poisson": True, "etas": False}


def test_cross_model_partial_success_cannot_be_combined_for_g1() -> None:
    uniform = _evidence("uniform_poisson", validation_ig=0.0, fold_igs=(0.0, 0.0, 0.0, 0.0))
    spatial = _evidence(
        "spatial_poisson",
        validation_ig=-0.1,
        fold_igs=(0.2, 0.2, 0.2, 0.2),
    )
    etas = _evidence(
        "etas",
        validation_ig=0.2,
        fold_igs=(0.2, -0.1, -0.1, -0.1),
    )

    assert assess_g1((uniform, spatial, etas)).passed is False


def test_model_selection_excludes_numerically_failed_high_score() -> None:
    uniform = _evidence("uniform_poisson", validation_ig=0.0, fold_igs=(0.0, 0.0, 0.0, 0.0))
    spatial = _evidence(
        "spatial_poisson",
        validation_ig=0.05,
        fold_igs=(0.2, -0.2, 0.2, -0.2),
    )
    failed_etas = _evidence(
        "etas",
        validation_ig=10.0,
        fold_igs=(10.0, 10.0, 10.0, 10.0),
        stable=False,
    )

    selection = select_background_model((uniform, spatial, failed_etas))

    assert selection.validation_best_model_id == "spatial_poisson"
    assert selection.selected_model_id == "uniform_poisson"
    assert selection.excluded_failed_models == ("etas",)


def test_bootstrap_is_namespaced_deterministic_and_keeps_fixed_compensator() -> None:
    contributions = InformationGainContributions(
        physical_event_ids=("e1", "e2", "e3", "e4"),
        event_log_intensity_differences=np.asarray([1.0, 2.0, 3.0, 4.0]),
        compensator_difference=2.0,
    )

    first = bootstrap_information_gain(
        contributions,
        model_seed_id="spatial_poisson_vs_uniform_poisson",
    )
    second = bootstrap_information_gain(
        contributions,
        model_seed_id="spatial_poisson_vs_uniform_poisson",
    )
    etas = bootstrap_information_gain(
        contributions,
        model_seed_id="etas_vs_uniform_poisson",
    )

    assert first.point_estimate == pytest.approx(2.0)
    assert first.replicate_values == second.replicate_values
    assert first.replicate_values != etas.replicate_values
    assert first.lower <= first.point_estimate <= first.upper
    assert len(first.replicate_values) == 2000


def test_bootstrap_rejects_duplicate_physical_events() -> None:
    with pytest.raises(ValueError, match="unique"):
        InformationGainContributions(
            physical_event_ids=("same", "same"),
            event_log_intensity_differences=np.asarray([1.0, 1.0]),
            compensator_difference=0.0,
        )
