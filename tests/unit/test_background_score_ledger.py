from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from seismoflux.background.config import load_background_protocol
from seismoflux.background.evidence import PointProcessScoreEvidence
from seismoflux.background.issues import FrozenIssueCalendar, load_frozen_issue_calendar
from seismoflux.background.score_ledger import (
    FROZEN_PRIMARY_INTERVALS,
    SCORE_SCOPE_ORDER,
    ScoreLedger,
    ScoreLedgerEntry,
    ScoreLedgerReferenceError,
    secondary_evaluation_context_id,
    validate_generated_score_collection,
    validate_registry_score_references,
)

CONFIG = Path("configs/background_local_support.yaml")
ISSUE_MANIFEST = Path("data/manifests/background_local_support_fold_manifest.json")
PROTOCOL = "a" * 64
AUTHORIZATION = "b" * 64
ISSUE_MANIFEST_SHA256 = "d" * 64
SUPPORT_IDS = {
    "fold_1": "local-support-1111111111111111",
    "fold_2": "local-support-2222222222222222",
    "fold_3": "local-support-3333333333333333",
    "fold_4": "local-support-4444444444444444",
    "final_validation": "local-support-5555555555555555",
}
DOMAIN_IDS = {
    snapshot_id: f"{index}" * 64
    for index, snapshot_id in enumerate(FROZEN_PRIMARY_INTERVALS, start=1)
}
_AUTO_SCORE = object()


@pytest.fixture(scope="module")
def calendar() -> FrozenIssueCalendar:
    config = load_background_protocol(CONFIG)
    return load_frozen_issue_calendar(ISSUE_MANIFEST, config=config)


def _score(
    *,
    model_id: str,
    snapshot_id: str,
    variant: str,
    interval: tuple[str, str, str] | None = None,
    evaluation_context_id: str | None = None,
) -> PointProcessScoreEvidence:
    fit_end, assessment_start, assessment_end = (
        FROZEN_PRIMARY_INTERVALS[snapshot_id] if interval is None else interval
    )
    return PointProcessScoreEvidence(
        protocol_sha256=PROTOCOL,
        model_id=model_id,  # type: ignore[arg-type]
        model_variant_id=variant,
        parameter_snapshot_id=f"parameters/{variant}/{snapshot_id}",
        snapshot_id=snapshot_id,
        fit_end_utc=fit_end,
        assessment_start_utc=assessment_start,
        assessment_end_utc=assessment_end,
        selected_mc=4.0,
        target_event_ids=(f"target/{snapshot_id}",),
        event_log_intensities=np.asarray([0.25], dtype=np.float64),
        compensator=0.5,
        numerical_gate_evidence_ids=(f"gate/{variant}/{snapshot_id}",),
        support_id=SUPPORT_IDS[snapshot_id],
        supported_area_km2=9_500_000.0,
        compensator_domain_id=DOMAIN_IDS[snapshot_id],
        authorization_id=AUTHORIZATION,
        evaluation_context_id=evaluation_context_id,
    )


def _entry(
    *,
    scope: str = "primary_snapshot",
    status: str = "succeeded",
    model_id: str = "uniform_poisson",
    snapshot_id: str = "fold_1",
    variant: str | None = None,
    interval: tuple[str, str, str] | None = None,
    score: object = _AUTO_SCORE,
    failure_reasons: tuple[str, ...] = (),
    partition_id: str | None = None,
    issue_date_local: str | None = None,
    horizon_days: int | None = None,
    publication_delay_days: int | None = None,
    kde_bandwidth_km: int | None = None,
    etas_parent_variant: str | None = None,
) -> ScoreLedgerEntry:
    chosen_variant = variant or f"{model_id}/v1"
    chosen_interval = FROZEN_PRIMARY_INTERVALS[snapshot_id] if interval is None else interval
    evaluation_context = (
        secondary_evaluation_context_id(
            protocol_sha256=PROTOCOL,
            authorization_id=AUTHORIZATION,
            partition_id=partition_id or "",
            issue_date_local=issue_date_local or "",
            issue_time_utc=chosen_interval[1],
            horizon_days=horizon_days or 0,
            publication_delay_days=(
                publication_delay_days if publication_delay_days is not None else -1
            ),
        )
        if scope == "secondary_validation_horizon" and status == "succeeded"
        else None
    )
    if score is _AUTO_SCORE:
        chosen_score = (
            _score(
                model_id=model_id,
                snapshot_id=snapshot_id,
                variant=chosen_variant,
                interval=chosen_interval,
                evaluation_context_id=evaluation_context,
            )
            if status == "succeeded"
            else None
        )
    else:
        if score is not None and not isinstance(score, PointProcessScoreEvidence):
            raise TypeError("test score must be PointProcessScoreEvidence or None")
        chosen_score = score
    return ScoreLedgerEntry(
        scope=scope,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        protocol_sha256=PROTOCOL,
        authorization_id=AUTHORIZATION,
        support_id=SUPPORT_IDS[snapshot_id],
        supported_area_km2=9_500_000.0,
        compensator_domain_id=DOMAIN_IDS[snapshot_id],
        model_id=model_id,  # type: ignore[arg-type]
        model_variant_id=chosen_variant,
        parameter_snapshot_id=f"parameters/{chosen_variant}/{snapshot_id}",
        snapshot_id=snapshot_id,
        fit_end_utc=chosen_interval[0],
        assessment_start_utc=chosen_interval[1],
        assessment_end_utc=chosen_interval[2],
        selected_mc=4.0,
        score=chosen_score,
        failure_reasons=failure_reasons,
        partition_id=partition_id,
        issue_date_local=issue_date_local,
        horizon_days=horizon_days,
        publication_delay_days=publication_delay_days,
        kde_bandwidth_km=kde_bandwidth_km,
        etas_parent_variant=etas_parent_variant,  # type: ignore[arg-type]
    )


def _ledger(calendar: FrozenIssueCalendar, entries: tuple[ScoreLedgerEntry, ...]) -> ScoreLedger:
    return ScoreLedger(
        protocol_sha256=PROTOCOL,
        authorization_id=AUTHORIZATION,
        issue_manifest_sha256=ISSUE_MANIFEST_SHA256,
        calendar=calendar,
        entries=entries,
        coverage="fragment",
    )


def test_complete_ledger_rejects_missing_preregistered_coverage(
    calendar: FrozenIssueCalendar,
) -> None:
    with pytest.raises(ValueError, match="primary attempts"):
        ScoreLedger(
            protocol_sha256=PROTOCOL,
            authorization_id=AUTHORIZATION,
            issue_manifest_sha256=ISSUE_MANIFEST_SHA256,
            calendar=calendar,
            entries=(),
        )

    fragment = ScoreLedger(
        protocol_sha256=PROTOCOL,
        authorization_id=AUTHORIZATION,
        issue_manifest_sha256=ISSUE_MANIFEST_SHA256,
        calendar=calendar,
        entries=(),
        coverage="fragment",
    )
    assert fragment.semantic_payload()["coverage"] == "fragment"


def _utc_plus_days(value: str, days: int) -> str:
    parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    assert parsed.tzinfo == UTC
    return (parsed + timedelta(days=days)).isoformat(timespec="seconds").replace("+00:00", "Z")


def test_primary_scope_accepts_only_the_five_exact_frozen_intervals() -> None:
    entries = tuple(_entry(snapshot_id=snapshot_id) for snapshot_id in FROZEN_PRIMARY_INTERVALS)
    assert tuple(entry.snapshot_id for entry in entries) == tuple(FROZEN_PRIMARY_INTERVALS)

    start = FROZEN_PRIMARY_INTERVALS["fold_1"][1]
    wrong = (
        FROZEN_PRIMARY_INTERVALS["fold_1"][0],
        start,
        "2009-12-30T16:00:00Z",
    )
    with pytest.raises(ValueError, match="frozen primary snapshot"):
        _entry(snapshot_id="fold_1", interval=wrong)


def test_kde_candidates_are_development_only_and_use_frozen_bandwidths() -> None:
    candidate = _entry(
        scope="kde_development_candidate",
        model_id="spatial_poisson",
        snapshot_id="fold_4",
        variant="spatial_poisson/bw300",
        kde_bandwidth_km=300,
    )
    assert candidate.kde_bandwidth_km == 300

    with pytest.raises(ValueError, match="four development folds"):
        _entry(
            scope="kde_development_candidate",
            model_id="spatial_poisson",
            snapshot_id="final_validation",
            variant="spatial_poisson/bw300",
            kde_bandwidth_km=300,
        )
    with pytest.raises(ValueError, match="frozen candidates"):
        _entry(
            scope="kde_development_candidate",
            model_id="spatial_poisson",
            variant="spatial_poisson/bw125",
            kde_bandwidth_km=125,
        )


def test_selected_kde_development_score_is_registered_once_and_may_be_registry_referenced(
    calendar: FrozenIssueCalendar,
) -> None:
    selected_candidate = _entry(
        scope="kde_development_candidate",
        model_id="spatial_poisson",
        snapshot_id="fold_2",
        variant="spatial_poisson/bw200/selected",
        kde_bandwidth_km=200,
    )
    ledger = _ledger(calendar, (selected_candidate,))
    selected_score_id = selected_candidate.score_id
    assert selected_score_id is not None
    validate_registry_score_references(ledger, {selected_score_id})
    with pytest.raises(ScoreLedgerReferenceError, match="duplicate"):
        validate_registry_score_references(
            ledger,
            (selected_score_id, selected_score_id),
        )

    duplicate_primary_reference = replace(
        selected_candidate,
        scope="primary_snapshot",
        kde_bandwidth_km=None,
    )
    with pytest.raises(ValueError, match="Score ID must occur exactly once"):
        _ledger(calendar, (selected_candidate, duplicate_primary_reference))


def test_etas_sensitivity_is_fold1_fold3_only_and_has_explicit_primary_pair(
    calendar: FrozenIssueCalendar,
) -> None:
    primary = _entry(
        model_id="etas",
        snapshot_id="fold_1",
        variant="etas/primary",
        etas_parent_variant="primary_include_eligible_unsupported",
    )
    excluded = _entry(
        scope="etas_unsupported_parent_sensitivity",
        model_id="etas",
        snapshot_id="fold_1",
        variant="etas/exclude_unsupported",
        etas_parent_variant="exclude_all_unsupported",
    )
    ledger = _ledger(calendar, (excluded, primary))
    assert len(ledger.score_ids) == 2

    with pytest.raises(ValueError, match="only for ETAS fold_1/fold_3"):
        _entry(
            scope="etas_unsupported_parent_sensitivity",
            model_id="etas",
            snapshot_id="fold_2",
            variant="etas/exclude_unsupported",
            etas_parent_variant="exclude_all_unsupported",
        )
    with pytest.raises(ValueError, match="exactly one explicit primary"):
        _ledger(calendar, (excluded,))


def test_secondary_windows_match_manifest_issue_and_exact_horizon_not_partition_end(
    calendar: FrozenIssueCalendar,
) -> None:
    exposure = calendar.validation.exposures(365)[0]
    interval = (
        FROZEN_PRIMARY_INTERVALS["final_validation"][0],
        exposure.issue_time_utc,
        _utc_plus_days(exposure.issue_time_utc, 365),
    )
    assert interval[2] > FROZEN_PRIMARY_INTERVALS["final_validation"][2]
    entry = _entry(
        scope="secondary_validation_horizon",
        model_id="etas",
        snapshot_id="final_validation",
        variant="etas/horizon365/delay0",
        interval=interval,
        partition_id="validation",
        issue_date_local=exposure.issue_date_local,
        horizon_days=365,
        publication_delay_days=0,
    )
    assert _ledger(calendar, (entry,)).score_ids == (entry.score_id,)

    wrong_end = (interval[0], interval[1], _utc_plus_days(interval[1], 364))
    wrong_entry = _entry(
        scope="secondary_validation_horizon",
        model_id="etas",
        snapshot_id="final_validation",
        variant="etas/wrong-horizon",
        interval=wrong_end,
        partition_id="validation",
        issue_date_local=exposure.issue_date_local,
        horizon_days=365,
        publication_delay_days=0,
    )
    with pytest.raises(ValueError, match="exact horizon"):
        _ledger(calendar, (wrong_entry,))


def test_development_locked_and_unknown_issue_partitions_are_never_scoreable(
    calendar: FrozenIssueCalendar,
) -> None:
    development_date = calendar.development.exposures(7)[0]
    interval = (
        FROZEN_PRIMARY_INTERVALS["final_validation"][0],
        development_date.issue_time_utc,
        _utc_plus_days(development_date.issue_time_utc, 7),
    )
    with pytest.raises(ValueError, match="fit_end < start < end"):
        _entry(
            scope="secondary_validation_horizon",
            snapshot_id="final_validation",
            interval=interval,
            partition_id="validation",
            issue_date_local=development_date.issue_date_local,
            horizon_days=7,
            publication_delay_days=0,
        )

    validation_exposure = calendar.validation.exposures(7)[0]
    validation_interval = (
        FROZEN_PRIMARY_INTERVALS["final_validation"][0],
        validation_exposure.issue_time_utc,
        _utc_plus_days(validation_exposure.issue_time_utc, 7),
    )
    for partition in ("development", "locked", "unknown"):
        with pytest.raises(ValueError, match="partition must be exactly validation"):
            _entry(
                scope="secondary_validation_horizon",
                snapshot_id="final_validation",
                interval=validation_interval,
                partition_id=partition,
                issue_date_local=validation_exposure.issue_date_local,
                horizon_days=7,
                publication_delay_days=0,
            )


def test_success_failed_and_not_run_score_contracts() -> None:
    succeeded = _entry()
    assert succeeded.score_id is not None
    failed = _entry(status="failed", score=None, failure_reasons=("optimizer_failed",))
    not_run = replace(
        _entry(status="not_run", score=None, failure_reasons=("upstream_gate",)),
        parameter_snapshot_id="not-run/upstream-gate/fold_1",
    )
    assert failed.score_id is None and not_run.score_id is None
    assert not_run.parameter_snapshot_id == "not-run/upstream-gate/fold_1"

    with pytest.raises(ValueError, match="require evidence"):
        _entry(status="succeeded", score=None)
    with pytest.raises(ValueError, match="must not have score"):
        _entry(
            status="failed",
            score=_score(
                model_id="uniform_poisson",
                snapshot_id="fold_1",
                variant="uniform_poisson/v1",
            ),
            failure_reasons=("optimizer_failed",),
        )
    with pytest.raises(ValueError, match="require reasons"):
        _entry(status="not_run", score=None)


def test_score_binding_rejects_authorization_support_domain_or_model_drift() -> None:
    score = _score(model_id="uniform_poisson", snapshot_id="fold_1", variant="uniform/v1")
    with pytest.raises(ValueError, match="authorization_id"):
        ScoreLedgerEntry(
            **{
                **{
                    field: getattr(_entry(variant="uniform/v1"), field)
                    for field in _entry(variant="uniform/v1").__dataclass_fields__
                },
                "authorization_id": "c" * 64,
                "score": score,
            }
        )


def test_ledger_is_content_addressed_order_invariant_and_reports_all_scope_counts(
    calendar: FrozenIssueCalendar,
) -> None:
    succeeded = _entry(snapshot_id="fold_2")
    failed = _entry(
        status="failed",
        snapshot_id="fold_4",
        score=None,
        failure_reasons=("numerical_gate",),
    )
    kde = _entry(
        scope="kde_development_candidate",
        model_id="spatial_poisson",
        snapshot_id="fold_3",
        variant="spatial_poisson/bw75",
        kde_bandwidth_km=75,
    )
    first = _ledger(calendar, (kde, failed, succeeded))
    second = _ledger(calendar, (succeeded, kde, failed))

    assert first.ledger_id == second.ledger_id
    assert len(first.ledger_sha256) == 64
    assert first.ledger_id == f"score-ledger-{first.ledger_sha256[:16]}"
    assert tuple(count.scope for count in first.classification_counts) == SCORE_SCOPE_ORDER
    primary = first.classification_counts[0]
    assert (primary.total, primary.succeeded, primary.failed, primary.not_run) == (2, 1, 1, 0)
    assert primary.score_count == 1


def test_locked_test_assertion_is_exactly_false_empty_null(
    calendar: FrozenIssueCalendar,
) -> None:
    ledger = _ledger(calendar, (_entry(),))
    assertion = ledger.locked_test_assertion
    assert assertion.run is False
    assert assertion.score_ids == ()
    assert assertion.result is None
    ledger.assert_locked_test_not_run()

    with pytest.raises(ValueError, match="false/empty/null"):
        replace(ledger, locked_test_run=True)
    with pytest.raises(ValueError, match="false/empty/null"):
        replace(ledger, locked_test_score_ids=("f" * 64,))
    with pytest.raises(ValueError, match="false/empty/null"):
        replace(ledger, locked_test_result={"passed": True})


def test_generated_and_registry_score_sets_have_no_missing_extra_or_orphans(
    calendar: FrozenIssueCalendar,
) -> None:
    entries = (_entry(snapshot_id="fold_1"), _entry(snapshot_id="fold_2"))
    ledger = _ledger(calendar, entries)
    evidence = tuple(entry.score for entry in entries if entry.score is not None)
    validate_generated_score_collection(ledger, evidence)
    validate_registry_score_references(ledger, set(ledger.score_ids))

    with pytest.raises(ScoreLedgerReferenceError, match="missing="):
        validate_registry_score_references(ledger, ledger.score_ids[:1])
    with pytest.raises(ScoreLedgerReferenceError, match="extra="):
        validate_registry_score_references(ledger, (*ledger.score_ids, "f" * 64))
    with pytest.raises(ScoreLedgerReferenceError, match="duplicate"):
        validate_generated_score_collection(ledger, (evidence[0], evidence[0]))


def test_ledger_rejects_duplicate_score_ids_and_cross_model_domain_drift(
    calendar: FrozenIssueCalendar,
) -> None:
    entry = _entry()
    with pytest.raises(ValueError, match="entry IDs must be unique"):
        _ledger(calendar, (entry, entry))

    spatial = _entry(model_id="spatial_poisson", variant="spatial/v1")
    changed_score = _score(
        model_id="spatial_poisson",
        snapshot_id="fold_1",
        variant="spatial/different-domain",
    )
    changed_score = replace(changed_score, compensator_domain_id="e" * 64)
    changed = replace(
        spatial,
        model_variant_id="spatial/different-domain",
        parameter_snapshot_id="parameters/spatial/different-domain/fold_1",
        compensator_domain_id="e" * 64,
        score=replace(
            changed_score,
            parameter_snapshot_id="parameters/spatial/different-domain/fold_1",
        ),
    )
    with pytest.raises(ValueError, match="one support/domain identity"):
        _ledger(calendar, (entry, changed))
