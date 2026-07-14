from __future__ import annotations

import numpy as np
import pytest

from seismoflux.background.evidence import (
    AuditedBackgroundModelEvidence,
    PairedInformationGainEvidence,
    PointProcessScoreEvidence,
    assess_audited_g1,
    select_audited_background_model,
)

PROTOCOL = "a" * 64


def _score(
    model_id: str,
    snapshot_id: str,
    *,
    variant: str,
    intensity: float,
    event_ids: tuple[str, ...] = ("e1", "e2"),
) -> PointProcessScoreEvidence:
    index = int(snapshot_id.removeprefix("fold_")) if snapshot_id.startswith("fold_") else 5
    return PointProcessScoreEvidence(
        protocol_sha256=PROTOCOL,
        model_id=model_id,  # type: ignore[arg-type]
        model_variant_id=variant,
        parameter_snapshot_id=f"params/{snapshot_id}",
        snapshot_id=snapshot_id,
        fit_end_utc=f"200{index}-01-01T00:00:00Z",
        assessment_start_utc=f"201{index}-01-01T00:00:00Z",
        assessment_end_utc=f"201{index + 1}-01-01T00:00:00Z",
        selected_mc=3.5,
        target_event_ids=event_ids,
        event_log_intensities=np.full(len(event_ids), np.log(intensity)),
        compensator=2.0,
        numerical_gate_evidence_ids=(f"gate/{snapshot_id}",),
    )


def _pair(
    model_id: str, snapshot_id: str, *, variant: str, ratio: float
) -> PairedInformationGainEvidence:
    uniform = _score("uniform_poisson", snapshot_id, variant="uniform/v1", intensity=1.0)
    candidate = _score(model_id, snapshot_id, variant=variant, intensity=ratio)
    return PairedInformationGainEvidence.build(candidate=candidate, uniform=uniform)


def _model(
    model_id: str, variant: str, ratios: tuple[float, ...]
) -> AuditedBackgroundModelEvidence:
    snapshots = ("fold_1", "fold_2", "fold_3", "fold_4", "final_validation")
    pairs = tuple(
        _pair(model_id, snapshot, variant=variant, ratio=ratio)
        for snapshot, ratio in zip(snapshots, ratios, strict=True)
    )
    return AuditedBackgroundModelEvidence(
        model_id=model_id,  # type: ignore[arg-type]
        model_variant_id=variant,
        protocol_sha256=PROTOCOL,
        development_folds=pairs[:4],
        validation=pairs[4],
        failed_snapshot_reasons=(),
    )


def test_paired_evidence_requires_identical_physical_targets_and_interval() -> None:
    uniform = _score("uniform_poisson", "fold_1", variant="uniform/v1", intensity=1.0)
    changed_targets = _score(
        "etas",
        "fold_1",
        variant="etas/d25",
        intensity=1.2,
        event_ids=("e1", "other"),
    )
    with pytest.raises(ValueError, match="target_event_ids"):
        PairedInformationGainEvidence.build(candidate=changed_targets, uniform=uniform)

    candidate = _score("etas", "fold_1", variant="etas/d25", intensity=1.2)
    changed_interval = PointProcessScoreEvidence(
        **{
            **{field: getattr(uniform, field) for field in uniform.__dataclass_fields__},
            "assessment_end_utc": "2099-01-01T00:00:00Z",
        }
    )
    with pytest.raises(ValueError, match="assessment_end_utc"):
        PairedInformationGainEvidence.build(candidate=candidate, uniform=changed_interval)


def test_same_model_variant_is_enforced_across_all_five_snapshots() -> None:
    folds = tuple(
        _pair("etas", f"fold_{index}", variant="etas/d25", ratio=1.1) for index in range(1, 5)
    )
    validation = _pair("etas", "final_validation", variant="etas/d100", ratio=1.1)
    with pytest.raises(ValueError, match="mixes different model variants"):
        AuditedBackgroundModelEvidence(
            model_id="etas",
            model_variant_id="etas/d25",
            protocol_sha256=PROTOCOL,
            development_folds=folds,
            validation=validation,
            failed_snapshot_reasons=(),
        )


def test_failed_model_accounts_for_every_snapshot_and_is_excluded() -> None:
    evidence = AuditedBackgroundModelEvidence(
        model_id="etas",
        model_variant_id="etas/d25",
        protocol_sha256=PROTOCOL,
        development_folds=(),
        validation=None,
        failed_snapshot_reasons=tuple(
            (snapshot, ("fit failed",))
            for snapshot in ("fold_1", "fold_2", "fold_3", "fold_4", "final_validation")
        ),
    )
    assert evidence.eligible_for_selection is False
    assert evidence.passes_g1_as_same_model is False


def test_non_leading_successful_fold_subset_remains_auditable_but_ineligible() -> None:
    scored_folds = (
        _pair("etas", "fold_1", variant="etas/d25", ratio=1.1),
        _pair("etas", "fold_3", variant="etas/d25", ratio=1.1),
        _pair("etas", "fold_4", variant="etas/d25", ratio=1.1),
    )
    validation = _pair("etas", "final_validation", variant="etas/d25", ratio=1.1)
    evidence = AuditedBackgroundModelEvidence(
        model_id="etas",
        model_variant_id="etas/d25",
        protocol_sha256=PROTOCOL,
        development_folds=scored_folds,
        validation=validation,
        failed_snapshot_reasons=(("fold_2", ("unstable Hessian",)),),
    )

    assert evidence.positive_development_fold_count == 3
    assert evidence.eligible_for_selection is False
    assert evidence.passes_g1_as_same_model is False


def test_audited_g1_and_one_se_use_complete_same_variant_evidence() -> None:
    uniform = _model("uniform_poisson", "uniform/v1", (1.0, 1.0, 1.0, 1.0, 1.0))
    spatial = _model("spatial_poisson", "spatial/bw300", (1.2, 1.1, 1.3, 0.9, 1.2))
    etas = _model("etas", "etas/d25/bw300", (1.3, 1.2, 1.1, 1.2, 1.3))

    assessment = assess_audited_g1((uniform, spatial, etas))
    assert assessment.passed is True
    assert assessment.passing_model_variants == ("spatial/bw300", "etas/d25/bw300")

    selection = select_audited_background_model((uniform, spatial, etas))
    assert selection.validation_best_model_variant_id == "etas/d25/bw300"
    assert selection.selected_model_variant_id in {
        "uniform/v1",
        "spatial/bw300",
        "etas/d25/bw300",
    }


def test_score_id_changes_with_event_intensity_or_parameter_identity() -> None:
    first = _score("etas", "fold_1", variant="etas/d25", intensity=1.1)
    second = _score("etas", "fold_1", variant="etas/d25", intensity=1.2)
    changed_parameter = PointProcessScoreEvidence(
        **{
            **{field: getattr(first, field) for field in first.__dataclass_fields__},
            "parameter_snapshot_id": "params/changed",
        }
    )
    assert first.score_id != second.score_id
    assert first.score_id != changed_parameter.score_id
