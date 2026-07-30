"""Fail-closed role separation and immutable Stage 2S prediction seals."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import cast

from seismoflux.background.artifacts import canonical_json_bytes

RAW_CATALOG_FIELDS = (
    "event_id",
    "origin_time_utc",
    "available_at",
    "longitude",
    "latitude",
    "magnitude",
    "inside_study_area",
)
FOLD_ORDER = (1, 2, 3)


class Stage2SSealError(RuntimeError):
    """Raised when a seal or role transition violates the frozen order."""


class Stage2SSealExists(Stage2SSealError):
    """Raised when an immutable O_EXCL record already exists."""


def _record_payload(record_type: str, bindings: Mapping[str, object]) -> dict[str, object]:
    if not record_type:
        raise ValueError("record_type must not be empty")
    base: dict[str, object] = {
        "schema_version": 1,
        "record_type": record_type,
        "bindings": dict(bindings),
    }
    base["content_sha256"] = hashlib.sha256(canonical_json_bytes(base)).hexdigest()
    return base


@dataclass(frozen=True, slots=True)
class SealedRecord:
    """Identity of one newly-created canonical O_EXCL JSON record."""

    path: Path
    record_type: str
    content_sha256: str
    file_sha256: str
    payload: Mapping[str, object]


def write_o_excl_record(
    path: Path,
    *,
    record_type: str,
    bindings: Mapping[str, object],
) -> SealedRecord:
    """Create one immutable canonical JSON+LF record without rollback or rewrite."""

    if not path.is_absolute():
        raise ValueError("seal path must be absolute")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _record_payload(record_type, bindings)
    serialized = canonical_json_bytes(payload) + b"\n"
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o600,
        )
    except FileExistsError as exc:
        raise Stage2SSealExists(f"immutable Stage 2S record already exists: {path}") from exc
    try:
        offset = 0
        while offset < len(serialized):
            written = os.write(descriptor, serialized[offset:])
            if written <= 0:
                raise Stage2SSealError("short write while creating immutable Stage 2S record")
            offset += written
        os.fsync(descriptor)
    except OSError as exc:
        raise Stage2SSealError(
            "immutable Stage 2S record creation failed and is intentionally not rolled back"
        ) from exc
    finally:
        os.close(descriptor)
    return SealedRecord(
        path=path,
        record_type=record_type,
        content_sha256=cast(str, payload["content_sha256"]),
        file_sha256=hashlib.sha256(serialized).hexdigest(),
        payload=MappingProxyType(payload),
    )


def _utc_datetime(value: object, *, label: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise Stage2SSealError(f"{label} is not an ISO-8601 timestamp") from exc
    else:
        raise Stage2SSealError(f"{label} must be an ISO-8601 string or datetime")
    if parsed.tzinfo is None:
        raise Stage2SSealError(f"{label} must be timezone-aware")
    return parsed.astimezone(UTC)


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool | str | bytes | bytearray):
        raise Stage2SSealError(f"{label} must be numeric")
    try:
        converted = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise Stage2SSealError(f"{label} must be numeric") from exc
    if not (-float("inf") < converted < float("inf")):
        raise Stage2SSealError(f"{label} must be finite")
    return converted


@dataclass(frozen=True, slots=True)
class CatalogRoleRow:
    """The exact raw-field allowlist exposed by every Stage 2S role view."""

    event_id: str
    origin_time_utc: datetime
    available_at: datetime
    longitude: float
    latitude: float
    magnitude: float
    inside_study_area: bool


def _catalog_rows(rows: Sequence[Mapping[str, object]]) -> tuple[CatalogRoleRow, ...]:
    converted: list[CatalogRoleRow] = []
    event_ids: set[str] = set()
    expected_fields = set(RAW_CATALOG_FIELDS)
    for row in rows:
        if set(row) != expected_fields:
            extra = sorted(set(row) - expected_fields)
            missing = sorted(expected_fields - set(row))
            raise Stage2SSealError(
                f"role row must expose only the raw allowlist; extra={extra}, missing={missing}"
            )
        event_id_value = row["event_id"]
        if not isinstance(event_id_value, str) or not event_id_value:
            raise Stage2SSealError("event_id must be a non-empty string")
        if event_id_value in event_ids:
            raise Stage2SSealError("role view event IDs must be unique")
        event_ids.add(event_id_value)
        origin = _utc_datetime(row["origin_time_utc"], label="origin_time_utc")
        available = _utc_datetime(row["available_at"], label="available_at")
        if available < origin:
            raise Stage2SSealError("available_at cannot precede origin_time_utc")
        longitude = _finite_number(row["longitude"], label="longitude")
        latitude = _finite_number(row["latitude"], label="latitude")
        magnitude = _finite_number(row["magnitude"], label="magnitude")
        if not -180.0 <= longitude <= 180.0 or not -90.0 <= latitude <= 90.0:
            raise Stage2SSealError("role row coordinates are outside geographic bounds")
        inside = row["inside_study_area"]
        if not isinstance(inside, bool):
            raise Stage2SSealError("inside_study_area must be boolean")
        converted.append(
            CatalogRoleRow(
                event_id=event_id_value,
                origin_time_utc=origin,
                available_at=available,
                longitude=longitude,
                latitude=latitude,
                magnitude=magnitude,
                inside_study_area=inside,
            )
        )
    converted.sort(key=lambda item: (item.origin_time_utc, item.event_id.encode("utf-8")))
    return tuple(converted)


class RoleSession:
    """Enforce fold-fit → issues → fold → master → assessment ordering."""

    def __init__(
        self,
        *,
        seal_root: Path,
        issue_dates_by_fold: Mapping[int, Sequence[str]],
    ) -> None:
        if not seal_root.is_absolute():
            raise ValueError("seal_root must be absolute")
        if tuple(sorted(issue_dates_by_fold)) != FOLD_ORDER:
            raise ValueError("issue_dates_by_fold must contain folds 1, 2, and 3")
        schedules: dict[int, tuple[str, ...]] = {}
        for fold_index in FOLD_ORDER:
            values = tuple(issue_dates_by_fold[fold_index])
            if not values or values != tuple(sorted(set(values))):
                raise ValueError("each fold issue schedule must be unique and ascending")
            schedules[fold_index] = values
        self._seal_root = seal_root
        self._schedules = MappingProxyType(schedules)
        self._fit_view_opened: set[int] = set()
        self._fit_receipts: dict[int, SealedRecord] = {}
        self._pending_issue: dict[int, str] = {}
        self._issue_records: dict[int, list[SealedRecord]] = {
            fold_index: [] for fold_index in FOLD_ORDER
        }
        self._fold_records: dict[int, SealedRecord] = {}
        self._master_record: SealedRecord | None = None
        self._assessment_opened = False

    @property
    def master_record(self) -> SealedRecord | None:
        return self._master_record

    def _require_prior_fold(self, fold_index: int) -> None:
        if fold_index not in FOLD_ORDER:
            raise Stage2SSealError("fold index must be 1, 2, or 3")
        if fold_index > 1 and fold_index - 1 not in self._fold_records:
            raise Stage2SSealError("later fold view requires the prior fold prediction seal")
        if any(index >= fold_index for index in self._fold_records):
            raise Stage2SSealError("fold role cannot be reopened after prediction sealing")

    def open_fit_view(
        self,
        *,
        fold_index: int,
        rows: Sequence[Mapping[str, object]],
        fit_cutoff_utc: object,
    ) -> tuple[CatalogRoleRow, ...]:
        """Open one raw-only fit view after the prior fold has been sealed."""

        self._require_prior_fold(fold_index)
        if self._master_record is not None or self._assessment_opened:
            raise Stage2SSealError("fit views cannot open after the master seal")
        if fold_index in self._fit_view_opened:
            raise Stage2SSealError("a fold fit view can open only once")
        cutoff = _utc_datetime(fit_cutoff_utc, label="fit_cutoff_utc")
        view = tuple(
            row
            for row in _catalog_rows(rows)
            if row.origin_time_utc <= cutoff and row.available_at <= cutoff
        )
        self._fit_view_opened.add(fold_index)
        return view

    def seal_fit(
        self,
        *,
        fold_index: int,
        bindings: Mapping[str, object],
    ) -> SealedRecord:
        """Seal weights/rates and raw fit identities before the first issue."""

        self._require_prior_fold(fold_index)
        if fold_index not in self._fit_view_opened:
            raise Stage2SSealError("fold fit view must open before its fit receipt")
        if fold_index in self._fit_receipts:
            raise Stage2SSealError("fold fit receipt is already sealed")
        previous = self._fold_records.get(fold_index - 1)
        complete_bindings = {
            **bindings,
            "fold_index": fold_index,
            "previous_fold_prediction_seal_sha256": (
                None if previous is None else previous.file_sha256
            ),
            "assessment_membership_score_hit_or_metric_exposed": False,
        }
        path = self._seal_root / f"fold_{fold_index}" / "fit_receipt.json"
        record = write_o_excl_record(
            path,
            record_type="stage2s_fold_fit_receipt",
            bindings=complete_bindings,
        )
        self._fit_receipts[fold_index] = record
        return record

    def open_causal_source_view(
        self,
        *,
        fold_index: int,
        issue_date: str,
        issue_time_utc: object,
        rows: Sequence[Mapping[str, object]],
    ) -> tuple[CatalogRoleRow, ...]:
        """Open only source rows already originated and available at issue time."""

        self._require_prior_fold(fold_index)
        if fold_index not in self._fit_receipts:
            raise Stage2SSealError("causal source view requires the fold fit receipt")
        if fold_index in self._pending_issue:
            raise Stage2SSealError("the prior issue must be sealed before another source view")
        completed = len(self._issue_records[fold_index])
        schedule = self._schedules[fold_index]
        if completed >= len(schedule) or issue_date != schedule[completed]:
            raise Stage2SSealError("issues must be opened in the frozen ascending order")
        cutoff = _utc_datetime(issue_time_utc, label="issue_time_utc")
        view = tuple(
            row
            for row in _catalog_rows(rows)
            if row.origin_time_utc <= cutoff and row.available_at <= cutoff
        )
        self._pending_issue[fold_index] = issue_date
        return view

    def seal_issue(
        self,
        *,
        fold_index: int,
        issue_date: str,
        bindings: Mapping[str, object],
    ) -> SealedRecord:
        """Seal all horizons/models/delays for the currently opened issue."""

        if self._pending_issue.get(fold_index) != issue_date:
            raise Stage2SSealError("issue seal has no matching opened causal source view")
        fit = self._fit_receipts[fold_index]
        previous = self._issue_records[fold_index][-1] if self._issue_records[fold_index] else None
        complete_bindings = {
            **bindings,
            "fold_index": fold_index,
            "issue_date": issue_date,
            "fold_fit_receipt_sha256": fit.file_sha256,
            "previous_issue_prediction_seal_sha256": (
                None if previous is None else previous.file_sha256
            ),
        }
        path = (
            self._seal_root / f"fold_{fold_index}" / f"issue_{issue_date}" / "prediction_seal.json"
        )
        record = write_o_excl_record(
            path,
            record_type="stage2s_issue_prediction_seal",
            bindings=complete_bindings,
        )
        self._issue_records[fold_index].append(record)
        del self._pending_issue[fold_index]
        return record

    def seal_fold(
        self,
        *,
        fold_index: int,
        bindings: Mapping[str, object],
    ) -> SealedRecord:
        """Seal the complete ordered issue chain for one fold."""

        self._require_prior_fold(fold_index)
        if fold_index not in self._fit_receipts:
            raise Stage2SSealError("fold prediction seal requires the fit receipt")
        if fold_index in self._pending_issue:
            raise Stage2SSealError("cannot seal a fold with an open issue")
        issues = self._issue_records[fold_index]
        if len(issues) != len(self._schedules[fold_index]):
            raise Stage2SSealError("every frozen issue must be sealed before its fold")
        previous = self._fold_records.get(fold_index - 1)
        complete_bindings = {
            **bindings,
            "fold_index": fold_index,
            "fold_fit_receipt_sha256": self._fit_receipts[fold_index].file_sha256,
            "ordered_issue_prediction_seal_sha256": [record.file_sha256 for record in issues],
            "previous_fold_prediction_seal_sha256": (
                None if previous is None else previous.file_sha256
            ),
        }
        path = self._seal_root / f"fold_{fold_index}" / "fold_prediction_seal.json"
        record = write_o_excl_record(
            path,
            record_type="stage2s_fold_prediction_seal",
            bindings=complete_bindings,
        )
        self._fold_records[fold_index] = record
        return record

    def seal_master(self, *, bindings: Mapping[str, object]) -> SealedRecord:
        """Seal all three folds before any assessment membership is exposed."""

        if tuple(sorted(self._fold_records)) != FOLD_ORDER:
            raise Stage2SSealError("master prediction seal requires folds 1, 2, and 3")
        if self._master_record is not None:
            raise Stage2SSealError("master prediction seal is already present")
        complete_bindings = {
            **bindings,
            "ordered_fold_prediction_seal_sha256": [
                self._fold_records[index].file_sha256 for index in FOLD_ORDER
            ],
            "assessment_target_role_or_score_exposed_before_master_seal": False,
        }
        path = self._seal_root / "prediction_seal.json"
        record = write_o_excl_record(
            path,
            record_type="stage2s_master_prediction_seal",
            bindings=complete_bindings,
        )
        self._master_record = record
        return record

    def open_assessment_view(
        self,
        *,
        rows: Sequence[Mapping[str, object]],
    ) -> tuple[CatalogRoleRow, ...]:
        """Expose raw assessment fields only after the O_EXCL master seal."""

        if self._master_record is None:
            raise Stage2SSealError("assessment view requires the master prediction seal")
        if self._assessment_opened:
            raise Stage2SSealError("assessment view can open only once")
        self._assessment_opened = True
        return _catalog_rows(rows)


__all__ = [
    "FOLD_ORDER",
    "RAW_CATALOG_FIELDS",
    "CatalogRoleRow",
    "RoleSession",
    "SealedRecord",
    "Stage2SSealError",
    "Stage2SSealExists",
    "write_o_excl_record",
]
