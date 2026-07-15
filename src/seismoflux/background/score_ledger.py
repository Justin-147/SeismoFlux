"""Immutable score ledger for the stage-2R-1 local-support comparison.

The ledger is deliberately separate from the original v0.2.0 publication path.  It
records every attempted v0.2.1 score, keeps successful score evidence in memory, and
uses each evidence object's content-addressed ``score_id`` in its own immutable
address.  Calendar-aware validation prevents a final-parameter snapshot from being
used on development issue dates and accepts only exact validation exposures frozen in
the issue manifest.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal, cast

from seismoflux.background.artifacts import canonical_json_bytes
from seismoflux.background.evidence import ModelId, PointProcessScoreEvidence
from seismoflux.background.issues import FrozenIssueCalendar

ScoreScope = Literal[
    "primary_snapshot",
    "kde_development_candidate",
    "secondary_validation_horizon",
    "etas_unsupported_parent_sensitivity",
]
ScoreStatus = Literal["succeeded", "failed", "not_run"]
ScoreLedgerCoverage = Literal["complete", "fragment"]
ETASUnsupportedParentVariant = Literal[
    "primary_include_eligible_unsupported",
    "exclude_all_unsupported",
]

SCORE_SCOPE_ORDER: tuple[ScoreScope, ...] = (
    "primary_snapshot",
    "kde_development_candidate",
    "secondary_validation_horizon",
    "etas_unsupported_parent_sensitivity",
)
_SCORE_STATUS_ORDER: tuple[ScoreStatus, ...] = ("succeeded", "failed", "not_run")
_SNAPSHOT_ORDER = ("fold_1", "fold_2", "fold_3", "fold_4", "final_validation")
_MODEL_ORDER: tuple[ModelId, ...] = ("uniform_poisson", "spatial_poisson", "etas")
_DEVELOPMENT_SNAPSHOTS = frozenset(_SNAPSHOT_ORDER[:4])
_UNSUPPORTED_PARENT_SNAPSHOTS = frozenset(("fold_1", "fold_3"))
_KDE_BANDWIDTHS_KM = frozenset((75, 100, 150, 200, 300))
_PUBLICATION_DELAYS_DAYS = frozenset((0, 1, 7))
_LOCAL_SUPPORT_FREEZE_TAG = "v0.2.1-background-local-support-protocol"
_LOCAL_SUPPORT_FROZEN_ON = "2026-07-14"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_LOCAL_SUPPORT_ID_PATTERN = re.compile(r"local-support-[0-9a-f]{16}")

# Exact protocol intervals.  These constants are intentionally independent of any
# target catalog and therefore cannot be moved after seeing a score.
FROZEN_PRIMARY_INTERVALS: dict[str, tuple[str, str, str]] = {
    "fold_1": (
        "2004-12-31T16:00:00Z",
        "2005-12-31T16:00:00Z",
        "2009-12-31T16:00:00Z",
    ),
    "fold_2": (
        "2009-12-31T16:00:00Z",
        "2010-12-31T16:00:00Z",
        "2014-12-31T16:00:00Z",
    ),
    "fold_3": (
        "2014-12-31T16:00:00Z",
        "2015-12-31T16:00:00Z",
        "2019-12-31T16:00:00Z",
    ),
    "fold_4": (
        "2019-12-31T16:00:00Z",
        "2020-12-31T16:00:00Z",
        "2023-06-30T16:00:00Z",
    ),
    "final_validation": (
        "2023-06-30T16:00:00Z",
        "2024-06-30T16:00:00Z",
        "2025-06-30T16:00:00Z",
    ),
}


class ScoreLedgerReferenceError(ValueError):
    """A score collection has missing or unregistered ledger references."""


def _nonempty(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty trimmed string")
    return value


def _sha256(value: str, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 string")
    return value


def _canonical_utc(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{name} must be a canonical UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise ValueError(f"{name} must be a canonical UTC timestamp ending in Z") from error
    if parsed.tzinfo != UTC or parsed.microsecond != 0:
        raise ValueError(f"{name} must use whole UTC seconds")
    canonical = parsed.isoformat(timespec="seconds").replace("+00:00", "Z")
    if canonical != value:
        raise ValueError(f"{name} must use canonical whole-second UTC form")
    return value


def _utc_plus_days(value: str, days: int) -> str:
    parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    return (parsed + timedelta(days=days)).isoformat(timespec="seconds").replace("+00:00", "Z")


def _ordered_unique_reasons(values: tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(_nonempty(value, name="score outcome reason") for value in values)
    if normalized != tuple(sorted(set(normalized))):
        raise ValueError("score outcome reasons must be sorted and unique")
    return normalized


def secondary_evaluation_context_id(
    *,
    protocol_sha256: str,
    authorization_id: str,
    partition_id: str,
    issue_date_local: str,
    issue_time_utc: str,
    horizon_days: int,
    publication_delay_days: int,
) -> str:
    """Address one exact preregistered secondary evaluation context."""

    _sha256(protocol_sha256, name="secondary context protocol_sha256")
    _sha256(authorization_id, name="secondary context authorization_id")
    _nonempty(partition_id, name="secondary context partition_id")
    _nonempty(issue_date_local, name="secondary context issue_date_local")
    _canonical_utc(issue_time_utc, name="secondary context issue_time_utc")
    if horizon_days not in {7, 30, 90, 180, 365}:
        raise ValueError("secondary context horizon is not frozen")
    if publication_delay_days not in _PUBLICATION_DELAYS_DAYS:
        raise ValueError("secondary context publication delay is not frozen")
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "schema": "seismoflux_g1_ls_secondary_context_v1",
                "protocol_sha256": protocol_sha256,
                "authorization_id": authorization_id,
                "partition_id": partition_id,
                "issue_date_local": issue_date_local,
                "issue_time_utc": issue_time_utc,
                "horizon_days": horizon_days,
                "publication_delay_days": publication_delay_days,
            }
        )
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class ScoreLedgerEntry:
    """One successful score or one explicit failed/not-run scoring attempt."""

    scope: ScoreScope
    status: ScoreStatus
    protocol_sha256: str
    authorization_id: str
    support_id: str
    supported_area_km2: float
    compensator_domain_id: str
    model_id: ModelId
    model_variant_id: str
    parameter_snapshot_id: str
    snapshot_id: str
    fit_end_utc: str
    assessment_start_utc: str
    assessment_end_utc: str
    selected_mc: float
    score: PointProcessScoreEvidence | None
    failure_reasons: tuple[str, ...] = ()
    partition_id: str | None = None
    issue_date_local: str | None = None
    horizon_days: int | None = None
    publication_delay_days: int | None = None
    kde_bandwidth_km: int | None = None
    etas_parent_variant: ETASUnsupportedParentVariant | None = None

    def __post_init__(self) -> None:
        if self.scope not in SCORE_SCOPE_ORDER:
            raise ValueError("unknown score-ledger scope")
        if self.status not in _SCORE_STATUS_ORDER:
            raise ValueError("unknown score-ledger status")
        _sha256(self.protocol_sha256, name="protocol_sha256")
        _sha256(self.authorization_id, name="authorization_id")
        if _LOCAL_SUPPORT_ID_PATTERN.fullmatch(self.support_id) is None:
            raise ValueError("support_id must be a frozen local-support identifier")
        if not math.isfinite(self.supported_area_km2) or self.supported_area_km2 <= 0.0:
            raise ValueError("supported_area_km2 must be finite and positive")
        _sha256(self.compensator_domain_id, name="compensator_domain_id")
        if self.model_id not in _MODEL_ORDER:
            raise ValueError("unknown background model_id")
        _nonempty(self.model_variant_id, name="model_variant_id")
        _nonempty(self.parameter_snapshot_id, name="parameter_snapshot_id")
        if self.snapshot_id not in _SNAPSHOT_ORDER:
            raise ValueError("score entry snapshot is not one of the five frozen snapshots")
        _canonical_utc(self.fit_end_utc, name="fit_end_utc")
        _canonical_utc(self.assessment_start_utc, name="assessment_start_utc")
        _canonical_utc(self.assessment_end_utc, name="assessment_end_utc")
        if not self.fit_end_utc < self.assessment_start_utc < self.assessment_end_utc:
            raise ValueError("score entry intervals must satisfy fit_end < start < end")
        if not math.isfinite(self.selected_mc) or self.selected_mc <= 0.0:
            raise ValueError("selected_mc must be finite and positive")
        reasons = _ordered_unique_reasons(tuple(self.failure_reasons))
        object.__setattr__(self, "failure_reasons", reasons)

        if self.status == "succeeded":
            if self.score is None or reasons:
                raise ValueError("successful score entries require evidence and no failure reason")
            self._validate_score_binding(self.score)
        elif self.score is not None or not reasons:
            raise ValueError("failed/not-run score entries require reasons and must not have score")

        if self.scope == "primary_snapshot":
            self._validate_primary_scope()
        elif self.scope == "kde_development_candidate":
            self._validate_kde_candidate_scope()
        elif self.scope == "secondary_validation_horizon":
            self._validate_secondary_shape()
        else:
            self._validate_etas_sensitivity_scope()

    def _validate_score_binding(self, score: PointProcessScoreEvidence) -> None:
        if not isinstance(score, PointProcessScoreEvidence):
            raise TypeError("score entries may contain only PointProcessScoreEvidence")
        bindings = {
            "protocol_sha256": self.protocol_sha256,
            "authorization_id": self.authorization_id,
            "support_id": self.support_id,
            "supported_area_km2": self.supported_area_km2,
            "compensator_domain_id": self.compensator_domain_id,
            "model_id": self.model_id,
            "model_variant_id": self.model_variant_id,
            "parameter_snapshot_id": self.parameter_snapshot_id,
            "snapshot_id": self.snapshot_id,
            "fit_end_utc": self.fit_end_utc,
            "assessment_start_utc": self.assessment_start_utc,
            "assessment_end_utc": self.assessment_end_utc,
            "selected_mc": self.selected_mc,
        }
        for field_name, expected in bindings.items():
            if getattr(score, field_name) != expected:
                raise ValueError(f"score evidence differs from ledger binding {field_name}")

    def _forbid_issue_fields(self) -> None:
        if any(
            value is not None
            for value in (
                self.partition_id,
                self.issue_date_local,
                self.horizon_days,
                self.publication_delay_days,
            )
        ):
            raise ValueError("non-secondary scores must not claim an issue partition or horizon")
        if self.score is not None and self.score.evaluation_context_id is not None:
            raise ValueError("non-secondary scores must not claim an evaluation context")

    def _validate_frozen_primary_interval(self) -> None:
        if (
            self.fit_end_utc,
            self.assessment_start_utc,
            self.assessment_end_utc,
        ) != FROZEN_PRIMARY_INTERVALS[self.snapshot_id]:
            raise ValueError("score interval differs from its frozen primary snapshot")

    def _validate_primary_scope(self) -> None:
        self._forbid_issue_fields()
        self._validate_frozen_primary_interval()
        if self.kde_bandwidth_km is not None:
            raise ValueError("primary scores must not claim a KDE candidate bandwidth")
        if self.model_id == "etas" and self.snapshot_id in _UNSUPPORTED_PARENT_SNAPSHOTS:
            if self.etas_parent_variant != "primary_include_eligible_unsupported":
                raise ValueError(
                    "primary ETAS fold_1/fold_3 entries must identify the unsupported-parent "
                    "primary variant"
                )
        elif self.etas_parent_variant is not None:
            raise ValueError("only fold_1/fold_3 ETAS entries may identify a parent variant")

    def _validate_kde_candidate_scope(self) -> None:
        self._forbid_issue_fields()
        self._validate_frozen_primary_interval()
        if self.snapshot_id not in _DEVELOPMENT_SNAPSHOTS:
            raise ValueError("KDE candidate scores are allowed only on the four development folds")
        if self.model_id != "spatial_poisson":
            raise ValueError("KDE candidate scope requires spatial_poisson")
        if self.kde_bandwidth_km not in _KDE_BANDWIDTHS_KM:
            raise ValueError("KDE candidate bandwidth is not one of the frozen candidates")
        if self.etas_parent_variant is not None:
            raise ValueError("KDE candidate scores cannot claim an ETAS parent variant")

    def _validate_secondary_shape(self) -> None:
        if self.partition_id != "validation":
            raise ValueError("secondary score partition must be exactly validation")
        _nonempty(cast(str, self.issue_date_local), name="secondary issue_date_local")
        if self.horizon_days not in {7, 30, 90, 180, 365}:
            raise ValueError("secondary horizon is not one of the frozen horizons")
        if self.publication_delay_days not in _PUBLICATION_DELAYS_DAYS:
            raise ValueError("secondary publication delay must be 0, 1, or 7 days")
        if self.snapshot_id != "final_validation":
            raise ValueError("secondary validation horizons require final_validation parameters")
        if self.fit_end_utc != FROZEN_PRIMARY_INTERVALS["final_validation"][0]:
            raise ValueError("secondary scores use the wrong frozen final fit cutoff")
        if self.kde_bandwidth_km is not None or self.etas_parent_variant is not None:
            raise ValueError("secondary scores cannot claim candidate or sensitivity fields")
        if self.score is not None:
            expected_context = secondary_evaluation_context_id(
                protocol_sha256=self.protocol_sha256,
                authorization_id=self.authorization_id,
                partition_id=self.partition_id,
                issue_date_local=cast(str, self.issue_date_local),
                issue_time_utc=self.assessment_start_utc,
                horizon_days=self.horizon_days,
                publication_delay_days=self.publication_delay_days,
            )
            if self.score.evaluation_context_id != expected_context:
                raise ValueError("secondary score uses another evaluation context")

    def _validate_etas_sensitivity_scope(self) -> None:
        self._forbid_issue_fields()
        self._validate_frozen_primary_interval()
        if self.model_id != "etas" or self.snapshot_id not in _UNSUPPORTED_PARENT_SNAPSHOTS:
            raise ValueError(
                "unsupported-parent sensitivity is allowed only for ETAS fold_1/fold_3"
            )
        if self.etas_parent_variant != "exclude_all_unsupported":
            raise ValueError("ETAS sensitivity entries must identify the exclude variant")
        if self.kde_bandwidth_km is not None:
            raise ValueError("ETAS sensitivity scores cannot claim a KDE candidate bandwidth")

    @property
    def score_id(self) -> str | None:
        return self.score.score_id if self.score is not None else None

    @property
    def entry_id(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.semantic_payload())).hexdigest()

    def semantic_payload(self) -> dict[str, object]:
        """Return the complete deterministic ledger projection for this entry."""

        return {
            "scope": self.scope,
            "status": self.status,
            "protocol_sha256": self.protocol_sha256,
            "authorization_id": self.authorization_id,
            "support_id": self.support_id,
            "supported_area_km2": self.supported_area_km2,
            "compensator_domain_id": self.compensator_domain_id,
            "model_id": self.model_id,
            "model_variant_id": self.model_variant_id,
            "parameter_snapshot_id": self.parameter_snapshot_id,
            "snapshot_id": self.snapshot_id,
            "fit_end_utc": self.fit_end_utc,
            "assessment_start_utc": self.assessment_start_utc,
            "assessment_end_utc": self.assessment_end_utc,
            "selected_mc": self.selected_mc,
            "score_id": self.score_id,
            "failure_reasons": self.failure_reasons,
            "partition_id": self.partition_id,
            "issue_date_local": self.issue_date_local,
            "horizon_days": self.horizon_days,
            "publication_delay_days": self.publication_delay_days,
            "kde_bandwidth_km": self.kde_bandwidth_km,
            "etas_parent_variant": self.etas_parent_variant,
        }


@dataclass(frozen=True, slots=True)
class ScoreLedgerClassificationCount:
    """Attempt and score counts for one frozen ledger scope."""

    scope: ScoreScope
    total: int
    succeeded: int
    failed: int
    not_run: int
    score_count: int


@dataclass(frozen=True, slots=True)
class LockedTestAssertion:
    """Machine-readable proof that this stage did not touch the locked test."""

    run: Literal[False] = False
    score_ids: tuple[str, ...] = ()
    result: None = None


@dataclass(frozen=True, slots=True)
class ScoreLedger:
    """Complete, immutable, content-addressed stage-2R-1 score ledger."""

    protocol_sha256: str
    authorization_id: str
    issue_manifest_sha256: str
    calendar: FrozenIssueCalendar
    entries: tuple[ScoreLedgerEntry, ...]
    coverage: ScoreLedgerCoverage = "complete"
    locked_test_run: bool = False
    locked_test_score_ids: tuple[str, ...] = ()
    locked_test_result: object | None = None

    def __post_init__(self) -> None:
        _sha256(self.protocol_sha256, name="ledger protocol_sha256")
        _sha256(self.authorization_id, name="ledger authorization_id")
        _sha256(self.issue_manifest_sha256, name="ledger issue_manifest_sha256")
        if self.coverage not in {"complete", "fragment"}:
            raise ValueError("score ledger coverage must be complete or fragment")
        self._validate_calendar()
        if (
            self.locked_test_run is not False
            or tuple(self.locked_test_score_ids) != ()
            or self.locked_test_result is not None
        ):
            raise ValueError("stage-2R-1 locked-test assertion must remain false/empty/null")
        object.__setattr__(self, "locked_test_score_ids", ())

        entries = tuple(self.entries)
        if any(not isinstance(entry, ScoreLedgerEntry) for entry in entries):
            raise TypeError("score ledger entries must be ScoreLedgerEntry instances")
        for entry in entries:
            if entry.protocol_sha256 != self.protocol_sha256:
                raise ValueError("score ledger entry uses another protocol fingerprint")
            if entry.authorization_id != self.authorization_id:
                raise ValueError("score ledger entry uses another scoring authorization")
            if entry.scope == "secondary_validation_horizon":
                self._validate_secondary_window(entry)

        entry_ids = tuple(entry.entry_id for entry in entries)
        if len(set(entry_ids)) != len(entry_ids):
            raise ValueError("score ledger entry IDs must be unique")
        score_ids = tuple(entry.score_id for entry in entries if entry.score_id is not None)
        if len(set(score_ids)) != len(score_ids):
            raise ValueError("every generated Score ID must occur exactly once in the ledger")
        self._validate_common_snapshot_domains(entries)
        self._validate_sensitivity_primary_entries(entries)
        if self.coverage == "complete":
            self._validate_complete_coverage(entries)

        ordered = tuple(sorted(entries, key=_entry_sort_key))
        object.__setattr__(self, "entries", ordered)

    def _validate_calendar(self) -> None:
        if not isinstance(self.calendar, FrozenIssueCalendar):
            raise TypeError("score ledger requires a FrozenIssueCalendar")
        if (
            self.calendar.schema_version != "1.0.0"
            or self.calendar.frozen_on != _LOCAL_SUPPORT_FROZEN_ON
            or self.calendar.freeze_tag != _LOCAL_SUPPORT_FREEZE_TAG
        ):
            raise ValueError("score ledger calendar is not the frozen local-support manifest")
        if (
            self.calendar.development.partition_id != "development"
            or self.calendar.validation.partition_id != "validation"
        ):
            raise ValueError("score ledger calendar partitions are not frozen")

    def _validate_secondary_window(self, entry: ScoreLedgerEntry) -> None:
        horizon = cast(int, entry.horizon_days)
        issue_date = cast(str, entry.issue_date_local)
        exposures = self.calendar.validation.exposures(horizon)
        matches = tuple(
            exposure for exposure in exposures if exposure.issue_date_local == issue_date
        )
        if len(matches) != 1:
            if issue_date in self.calendar.development.actual_issue_dates_local:
                raise ValueError(
                    "development issue dates must not be scored with final_validation parameters"
                )
            raise ValueError("secondary score is absent from the frozen validation exposure list")
        exposure = matches[0]
        expected_end = _utc_plus_days(exposure.issue_time_utc, exposure.horizon_days)
        if entry.assessment_start_utc != exposure.issue_time_utc:
            raise ValueError("secondary score start differs from its frozen issue time")
        if entry.assessment_end_utc != expected_end:
            raise ValueError("secondary score end differs from issue time plus exact horizon")

    @staticmethod
    def _validate_common_snapshot_domains(entries: tuple[ScoreLedgerEntry, ...]) -> None:
        for snapshot_id in _SNAPSHOT_ORDER:
            members = tuple(entry for entry in entries if entry.snapshot_id == snapshot_id)
            identities = {
                (
                    entry.support_id,
                    entry.supported_area_km2,
                    entry.compensator_domain_id,
                    entry.selected_mc,
                )
                for entry in members
            }
            if len(identities) > 1:
                raise ValueError(
                    f"ledger entries do not share one support/domain identity for {snapshot_id}"
                )

    @staticmethod
    def _validate_sensitivity_primary_entries(entries: tuple[ScoreLedgerEntry, ...]) -> None:
        for sensitivity in (
            entry for entry in entries if entry.scope == "etas_unsupported_parent_sensitivity"
        ):
            primary = tuple(
                entry
                for entry in entries
                if entry.scope == "primary_snapshot"
                and entry.model_id == "etas"
                and entry.snapshot_id == sensitivity.snapshot_id
                and entry.etas_parent_variant == "primary_include_eligible_unsupported"
            )
            if len(primary) != 1:
                raise ValueError(
                    "each ETAS unsupported-parent sensitivity needs exactly one explicit "
                    "primary variant entry"
                )

    def _validate_complete_coverage(self, entries: tuple[ScoreLedgerEntry, ...]) -> None:
        """Require every preregistered primary, candidate, sensitivity, and issue attempt."""

        primary = tuple(entry for entry in entries if entry.scope == "primary_snapshot")
        expected_primary = {
            *(("uniform_poisson", snapshot_id) for snapshot_id in _SNAPSHOT_ORDER),
            ("spatial_poisson", "final_validation"),
            *(("etas", snapshot_id) for snapshot_id in _SNAPSHOT_ORDER),
        }
        observed_primary = tuple((entry.model_id, entry.snapshot_id) for entry in primary)
        if len(observed_primary) != len(set(observed_primary)) or set(observed_primary) != (
            expected_primary
        ):
            raise ValueError("complete score ledger has incomplete or extra primary attempts")

        kde = tuple(entry for entry in entries if entry.scope == "kde_development_candidate")
        expected_kde = {
            (snapshot_id, bandwidth)
            for snapshot_id in _SNAPSHOT_ORDER[:4]
            for bandwidth in _KDE_BANDWIDTHS_KM
        }
        observed_kde = tuple((entry.snapshot_id, entry.kde_bandwidth_km) for entry in kde)
        if len(observed_kde) != len(set(observed_kde)) or set(observed_kde) != expected_kde:
            raise ValueError("complete score ledger has incomplete or extra KDE candidates")

        sensitivity = tuple(
            entry for entry in entries if entry.scope == "etas_unsupported_parent_sensitivity"
        )
        observed_sensitivity = tuple(entry.snapshot_id for entry in sensitivity)
        if len(observed_sensitivity) != len(set(observed_sensitivity)) or set(
            observed_sensitivity
        ) != set(_UNSUPPORTED_PARENT_SNAPSHOTS):
            raise ValueError("complete score ledger has incomplete ETAS parent sensitivity")

        secondary = tuple(
            entry for entry in entries if entry.scope == "secondary_validation_horizon"
        )
        expected_secondary = {
            (model_id, delay, horizon, exposure.issue_date_local)
            for delay in _PUBLICATION_DELAYS_DAYS
            for model_id in _MODEL_ORDER
            for horizon in (7, 30, 90, 180, 365)
            for exposure in self.calendar.validation.exposures(horizon)
        }
        observed_secondary = tuple(
            (
                entry.model_id,
                entry.publication_delay_days,
                entry.horizon_days,
                entry.issue_date_local,
            )
            for entry in secondary
        )
        if (
            len(observed_secondary) != len(set(observed_secondary))
            or set(observed_secondary) != expected_secondary
        ):
            raise ValueError(
                "complete score ledger has incomplete or extra secondary validation attempts"
            )

    @property
    def score_ids(self) -> tuple[str, ...]:
        return tuple(entry.score_id for entry in self.entries if entry.score_id is not None)

    @property
    def scored_evidence(self) -> tuple[PointProcessScoreEvidence, ...]:
        return tuple(entry.score for entry in self.entries if entry.score is not None)

    @property
    def classification_counts(self) -> tuple[ScoreLedgerClassificationCount, ...]:
        counts: list[ScoreLedgerClassificationCount] = []
        for scope in SCORE_SCOPE_ORDER:
            entries = tuple(entry for entry in self.entries if entry.scope == scope)
            counts.append(
                ScoreLedgerClassificationCount(
                    scope=scope,
                    total=len(entries),
                    succeeded=sum(entry.status == "succeeded" for entry in entries),
                    failed=sum(entry.status == "failed" for entry in entries),
                    not_run=sum(entry.status == "not_run" for entry in entries),
                    score_count=sum(entry.score_id is not None for entry in entries),
                )
            )
        return tuple(counts)

    @property
    def locked_test_assertion(self) -> LockedTestAssertion:
        return LockedTestAssertion()

    def assert_locked_test_not_run(self) -> None:
        """Raise if the immutable false/empty/null lock-test assertion ever drifts."""

        if self.locked_test_assertion != LockedTestAssertion():
            raise ValueError("locked test was not left untouched")

    def semantic_payload(self) -> dict[str, object]:
        validation_windows = tuple(
            {
                "issue_date_local": exposure.issue_date_local,
                "issue_time_utc": exposure.issue_time_utc,
                "horizon_days": horizon,
                "assessment_end_utc": _utc_plus_days(
                    exposure.issue_time_utc, exposure.horizon_days
                ),
            }
            for horizon, exposures in self.calendar.validation.exposures_by_horizon
            for exposure in exposures
        )
        return {
            "schema_version": "1.0.0",
            "protocol_version": "0.2.1",
            "protocol_sha256": self.protocol_sha256,
            "authorization_id": self.authorization_id,
            "issue_manifest_sha256": self.issue_manifest_sha256,
            "coverage": self.coverage,
            "calendar": {
                "schema_version": self.calendar.schema_version,
                "frozen_on": self.calendar.frozen_on,
                "freeze_tag": self.calendar.freeze_tag,
                "validation_windows": validation_windows,
            },
            "entries": tuple(entry.semantic_payload() for entry in self.entries),
            "classification_counts": tuple(
                {
                    "scope": count.scope,
                    "total": count.total,
                    "succeeded": count.succeeded,
                    "failed": count.failed,
                    "not_run": count.not_run,
                    "score_count": count.score_count,
                }
                for count in self.classification_counts
            ),
            "locked_test": {"run": False, "score_ids": (), "result": None},
        }

    @property
    def ledger_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.semantic_payload())).hexdigest()

    @property
    def ledger_id(self) -> str:
        return f"score-ledger-{self.ledger_sha256[:16]}"


def _entry_sort_key(entry: ScoreLedgerEntry) -> tuple[object, ...]:
    return (
        SCORE_SCOPE_ORDER.index(entry.scope),
        _SNAPSHOT_ORDER.index(entry.snapshot_id),
        _MODEL_ORDER.index(entry.model_id),
        entry.issue_date_local or "",
        entry.horizon_days or 0,
        entry.publication_delay_days or 0,
        entry.kde_bandwidth_km or 0,
        entry.etas_parent_variant or "",
        entry.model_variant_id,
        entry.status,
        entry.entry_id,
    )


def _score_id_set(values: Iterable[str], *, label: str) -> frozenset[str]:
    materialized = tuple(values)
    for value in materialized:
        _sha256(value, name=f"{label} Score ID")
    if len(set(materialized)) != len(materialized):
        raise ScoreLedgerReferenceError(f"{label} contains duplicate Score IDs")
    return frozenset(materialized)


def _validate_exact_score_references(
    ledger: ScoreLedger,
    references: Iterable[str],
    *,
    label: str,
) -> None:
    expected = frozenset(ledger.score_ids)
    actual = _score_id_set(references, label=label)
    missing = tuple(sorted(expected - actual))
    extra = tuple(sorted(actual - expected))
    if missing or extra:
        raise ScoreLedgerReferenceError(
            f"{label} Score ID set differs from the ledger; missing={missing}, extra={extra}"
        )


def validate_generated_score_collection(
    ledger: ScoreLedger,
    generated: Iterable[PointProcessScoreEvidence],
) -> None:
    """Prove that every generated evidence object occurs in the ledger exactly once."""

    scores = tuple(generated)
    if any(not isinstance(score, PointProcessScoreEvidence) for score in scores):
        raise TypeError("generated score collection must contain PointProcessScoreEvidence")
    identifiers = tuple(score.score_id for score in scores)
    if len(set(identifiers)) != len(identifiers):
        raise ScoreLedgerReferenceError("generated score collection contains duplicate Score IDs")
    _validate_exact_score_references(ledger, identifiers, label="generated evidence")


def validate_registry_score_references(
    ledger: ScoreLedger,
    registry_score_ids: Iterable[str],
) -> None:
    """Reject both orphan ledger scores and registry references to nonexistent scores."""

    _validate_exact_score_references(ledger, registry_score_ids, label="registry reference")


__all__ = [
    "FROZEN_PRIMARY_INTERVALS",
    "SCORE_SCOPE_ORDER",
    "ETASUnsupportedParentVariant",
    "LockedTestAssertion",
    "ScoreLedger",
    "ScoreLedgerClassificationCount",
    "ScoreLedgerCoverage",
    "ScoreLedgerEntry",
    "ScoreLedgerReferenceError",
    "ScoreScope",
    "ScoreStatus",
    "secondary_evaluation_context_id",
    "validate_generated_score_collection",
    "validate_registry_score_references",
]
