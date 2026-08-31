"""Write-once public artifacts for one frozen P1 prospective issue.

This module is the narrow persistence boundary between a future-blind
``ProductionForecastBundle`` and the public prospective ledger.  It has no
network transport and never opens a truth catalogue.  Exact public ComCat
responses, the issue-time forecast, and deterministic visualisations are
sealed in a new issue directory; the private frozen local catalogue and the
derived combined catalogue remain hash-only identities.

Preparing artifacts is intentionally separate from building a
``ForecastIssueRecord`` preimage.  Publication is complete only after a caller
has committed, pushed, and remotely read back the prepared directory, and then
passes that later timestamp to :func:`build_forecast_issue_record_fields`.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final, Protocol, TypeAlias, cast

from seismoflux.data.common import canonical_json_bytes
from seismoflux.p1_b0_r30.preflight import EXPECTED_SUPPORT_MANIFEST_SHA256
from seismoflux.p1_b0_r30.production import (
    ComCatHttpExchange,
    ComCatIssueInputAcquisition,
    P1IssueSchedule,
    build_comcat_count_snapshot,
    build_comcat_snapshot,
    issue_schedule,
)
from seismoflux.p1_b0_r30.production_rendering import (
    ProductionForecastView,
    build_offline_production_forecast_html,
    parse_production_forecast_view,
    render_production_forecast_svg,
)
from seismoflux.p1_b0_r30.prospective import (
    P1_MODEL_MANIFEST_SHA256,
    P1_SOURCE_BOUNDARY_MANIFEST_SHA256,
    ProductionForecastBundle,
    build_production_forecast,
)
from seismoflux.stage2s.catalog import (
    FROZEN_CATALOG_CONTENT_SHA256,
    FROZEN_CATALOG_FILE_SHA256,
    FROZEN_CATALOG_ROW_COUNT,
    FROZEN_CATALOG_SCHEMA_SHA256,
)

JsonObject: TypeAlias = dict[str, object]

_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}")
_COMMIT_RE: Final = re.compile(r"[0-9a-f]{40}")
_ISSUE_RE: Final = re.compile(r"p1-[0-9]{8}T[0-9]{6}Z")
_TEMP_PREFIX: Final = ".p1-issue-preparing-"

COUNT_BODY_FILENAME: Final = "comcat_count_response.bin"
COUNT_HEADERS_FILENAME: Final = "comcat_count_response_headers.json"
QUERY_BODY_FILENAME: Final = "comcat_query_response.geojson"
QUERY_HEADERS_FILENAME: Final = "comcat_query_response_headers.json"
SOURCE_RECEIPT_FILENAME: Final = "source_receipt.json"
FORECAST_GRID_FILENAME: Final = "forecast_grid.json"
STATIC_SVG_FILENAME: Final = "forecast_static.svg"
OFFLINE_HTML_FILENAME: Final = "forecast_interactive.html"
PREPARED_RECEIPT_FILENAME: Final = "prepared_receipt.json"
ARTIFACT_MANIFEST_FILENAME: Final = "artifact_manifest.json"

_CORE_ARTIFACT_FILENAMES: Final = (
    COUNT_BODY_FILENAME,
    COUNT_HEADERS_FILENAME,
    QUERY_BODY_FILENAME,
    QUERY_HEADERS_FILENAME,
    SOURCE_RECEIPT_FILENAME,
    FORECAST_GRID_FILENAME,
    STATIC_SVG_FILENAME,
    OFFLINE_HTML_FILENAME,
)
_MANIFESTED_FILENAMES: Final = (*_CORE_ARTIFACT_FILENAMES, PREPARED_RECEIPT_FILENAME)
_ALL_FILENAMES: Final = (*_MANIFESTED_FILENAMES, ARTIFACT_MANIFEST_FILENAME)


class P1IssueArtifactError(ValueError):
    """A prepared issue package is unsafe, non-canonical, or internally inconsistent."""


class ForecastRebuilder(Protocol):
    """Injected only by unit tests; production uses the frozen real builder."""

    def __call__(
        self,
        *,
        schedule: P1IssueSchedule,
        acquisition: ComCatIssueInputAcquisition,
        local_catalog_bytes: bytes,
        study_area_bytes: bytes,
        support_manifest_bytes: bytes,
    ) -> ProductionForecastBundle: ...


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _utc(value: datetime, *, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise P1IssueArtifactError(f"{label} must be timezone-aware")
    return value.astimezone(UTC)


def _utc_text(value: datetime) -> str:
    return _utc(value, label="timestamp").isoformat().replace("+00:00", "Z")


def _parse_utc(value: object, *, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise P1IssueArtifactError(f"{label} must be an ISO-8601 UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise P1IssueArtifactError(f"{label} is not a valid UTC timestamp") from exc
    return parsed.astimezone(UTC)


def _sha_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise P1IssueArtifactError(f"{label} must be a lowercase SHA-256")
    return value


def _commit_text(value: object, *, label: str = "code_commit") -> str:
    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
        raise P1IssueArtifactError(f"{label} must be a lowercase 40-character Git SHA")
    return value


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise P1IssueArtifactError(f"{label} must be a JSON object")
    return cast(Mapping[str, object], value)


def _sequence(value: object, *, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise P1IssueArtifactError(f"{label} must be a JSON array")
    return cast(Sequence[object], value)


def _exact_keys(value: Mapping[str, object], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise P1IssueArtifactError(f"{label} fields differ from the frozen artifact contract")


def _expected_issue_id(scheduled: datetime) -> str:
    return f"p1-{scheduled.strftime('%Y%m%dT%H%M%SZ')}"


def _validate_issue_identity(view: ProductionForecastView) -> None:
    if _ISSUE_RE.fullmatch(view.issue_id) is None:
        raise P1IssueArtifactError("issue_id is not a frozen P1 issue identity")
    if view.issue_id != _expected_issue_id(view.scheduled_issue_time):
        raise P1IssueArtifactError("issue_id does not exactly encode scheduled issue time T")


def _validate_bundle_source_counts(
    bundle: ProductionForecastBundle, view: ProductionForecastView
) -> None:
    """Bind displayed source counts to the exact frozen bundle used for modelling."""

    if type(bundle.b0_source_count) is not int or type(bundle.recent_source_count) is not int:
        raise P1IssueArtifactError("forecast bundle source counts must be integers")
    if (
        bundle.b0_source_count != view.B0_source_count
        or bundle.recent_source_count != view.R30_source_count
    ):
        raise P1IssueArtifactError("forecast source counts differ from the frozen bundle")


def _canonical_json_object(raw_bytes: bytes, *, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise P1IssueArtifactError(f"{label} is not valid UTF-8 JSON") from exc
    mapping = _mapping(value, label=label)
    if canonical_json_bytes(mapping) != raw_bytes:
        raise P1IssueArtifactError(f"{label} is not canonical JSON")
    return mapping


def _path_is_reparse_point(path: Path) -> bool:
    if path.is_symlink():
        return True
    metadata = path.stat(follow_symlinks=False)
    attributes = getattr(metadata, "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _regular_directory(path: Path, *, label: str) -> Path:
    candidate = Path(path).absolute()
    if not candidate.exists() or not candidate.is_dir():
        raise P1IssueArtifactError(f"{label} must be an existing non-reparse directory")
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current /= part
        if _path_is_reparse_point(current):
            raise P1IssueArtifactError(f"{label} may not traverse a reparse point")
    return candidate.resolve(strict=True)


def _read_regular_file(path: Path) -> bytes:
    if not path.exists() or _path_is_reparse_point(path):
        raise P1IssueArtifactError(f"required artifact is missing or redirected: {path.name}")
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise P1IssueArtifactError(f"artifact is not a regular file: {path.name}")
    with path.open("rb") as handle:
        opened_before = os.fstat(handle.fileno())
        payload = handle.read()
        opened_after = os.fstat(handle.fileno())
    after = path.stat(follow_symlinks=False)
    signatures = {
        (item.st_dev, item.st_ino, item.st_mode, item.st_size, item.st_mtime_ns)
        for item in (before, opened_before, opened_after, after)
    }
    if len(signatures) != 1:
        raise P1IssueArtifactError(f"artifact changed while being read: {path.name}")
    return payload


def _write_new_file(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | cast(int, getattr(os, "O_BINARY", 0))
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        total = 0
        while total < len(payload):
            written = os.write(descriptor, view[total:])
            if written <= 0:
                raise P1IssueArtifactError(f"artifact write stopped early: {path.name}")
            total += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _cleanup_owned_temp(parent: Path, temporary: Path) -> None:
    """Remove only the direct, internally named temporary directory we created."""

    if (
        temporary.parent != parent
        or not temporary.name.startswith(_TEMP_PREFIX)
        or temporary.is_symlink()
    ):
        raise P1IssueArtifactError("refusing cleanup outside the owned issue temporary path")
    if temporary.exists():
        shutil.rmtree(temporary)


def _file_entry(relative_path: str, payload: bytes) -> JsonObject:
    return {
        "relative_path": relative_path,
        "sha256": _sha256(payload),
        "byte_count": len(payload),
    }


def _record_forecast_preimages(view: ProductionForecastView) -> tuple[JsonObject, JsonObject]:
    cell_ids = [cell.cell_id for cell in view.cells]
    values: list[JsonObject] = []
    for model in view.models:
        grid_sha = _sha256(
            canonical_json_bytes(
                {
                    "domain": "seismoflux.p1.relative-intensity-grid.v1",
                    "model_id": model.model_id,
                    "cell_ids": cell_ids,
                    "normalized_cell_mass_hex": [
                        value.hex() for value in model.normalized_cell_mass
                    ],
                }
            )
        )
        ranking_sha = _sha256(
            canonical_json_bytes(
                {
                    "domain": "seismoflux.p1.real-history-ranking.v1",
                    "model_id": model.model_id,
                    "cell_ids": list(model.ranked_cell_ids),
                }
            )
        )
        mask_sha = _sha256(
            canonical_json_bytes(
                {
                    "domain": "seismoflux.p1.real-history-alarm-mask.v1",
                    "model_id": model.model_id,
                    "cell_ids": list(model.alarm_cell_ids),
                    "actual_area_km2_hex": model.actual_alarm_area_km2.hex(),
                }
            )
        )
        values.append(
            {
                "model_id": model.model_id,
                "relative_intensity_grid_sha256": grid_sha,
                "alarm_mask_sha256": mask_sha,
                "alarm_ranking_sha256": ranking_sha,
                "actual_alarm_area_km2": model.actual_alarm_area_km2,
            }
        )
    return values[0], values[1]


def _catalog_derivation_receipt(bundle: ProductionForecastBundle) -> JsonObject:
    """Reduce catalogue joining metadata to counts and content identities only."""

    audit = _mapping(bundle.catalog_artifact.audit_mapping(), label="catalogue derivation audit")
    _exact_keys(
        audit,
        {
            "combined_catalog_sha256",
            "combined_catalog_row_count",
            "retained_comcat_event_count",
            "cutover_match_count",
            "cutover_matches",
            "available_at_semantics",
        },
        label="catalogue derivation audit",
    )
    matches = _sequence(audit.get("cutover_matches"), label="cutover matches")
    counts: dict[str, int] = {}
    for key in (
        "combined_catalog_row_count",
        "retained_comcat_event_count",
        "cutover_match_count",
    ):
        value = audit.get(key)
        if type(value) is not int or value < 0:
            raise P1IssueArtifactError(f"catalogue derivation {key} must be non-negative")
        counts[key] = value
    if counts["cutover_match_count"] != len(matches):
        raise P1IssueArtifactError("catalogue cutover match count does not reconcile")
    if audit.get("available_at_semantics") != "ComCat_provider_updated_at_conservative_lte_Q":
        raise P1IssueArtifactError("catalogue available-at semantics differ from frozen P1")
    return {
        "combined_catalog_sha256": _sha_text(
            audit.get("combined_catalog_sha256"), label="combined catalogue SHA-256"
        ),
        **counts,
        # Match rows can carry local event IDs.  They remain private; the public
        # receipt binds only their deterministic canonical hash.
        "cutover_match_preimage_sha256": _sha256(canonical_json_bytes(matches)),
        "available_at_semantics": audit.get("available_at_semantics"),
    }


def _source_receipt(
    bundle: ProductionForecastBundle,
    *,
    view: ProductionForecastView,
    core_payloads: Mapping[str, bytes],
) -> JsonObject:
    acquisition = bundle.acquisition
    query = acquisition.query_snapshot
    if acquisition.status != "available" or query is None:
        raise P1IssueArtifactError("only an available count-first acquisition can be published")
    catalog_identity = bundle.catalog_artifact.catalog.identity
    derivation_audit = _catalog_derivation_receipt(bundle)
    _reject_private_catalog_disclosure(derivation_audit, label="catalogue derivation audit")
    return {
        "schema_version": 1,
        "artifact_type": "p1_real_prospective_source_receipt_v1",
        "issue_id": view.issue_id,
        "scheduled_issue_time_utc": view.scheduled_issue_time_utc,
        "query_cutoff_utc": view.query_cutoff_utc,
        "source_snapshot_sha256": view.source_snapshot_sha256,
        "source_request_sha256": view.source_request_sha256,
        "source_boundary_manifest_sha256": P1_SOURCE_BOUNDARY_MANIFEST_SHA256,
        "comcat_acquisition": acquisition.as_mapping(),
        "raw_comcat_artifacts": {
            "count_response": _file_entry(COUNT_BODY_FILENAME, core_payloads[COUNT_BODY_FILENAME]),
            "count_headers": _file_entry(
                COUNT_HEADERS_FILENAME, core_payloads[COUNT_HEADERS_FILENAME]
            ),
            "query_response": _file_entry(QUERY_BODY_FILENAME, core_payloads[QUERY_BODY_FILENAME]),
            "query_headers": _file_entry(
                QUERY_HEADERS_FILENAME, core_payloads[QUERY_HEADERS_FILENAME]
            ),
        },
        "frozen_local_catalog_identity": {
            "row_count": FROZEN_CATALOG_ROW_COUNT,
            "file_sha256": FROZEN_CATALOG_FILE_SHA256,
            "content_sha256": FROZEN_CATALOG_CONTENT_SHA256,
            "schema_sha256": FROZEN_CATALOG_SCHEMA_SHA256,
        },
        "derived_combined_catalog_identity": {
            "row_count": catalog_identity.row_count,
            "file_sha256": catalog_identity.file_sha256,
            "content_sha256": catalog_identity.content_sha256,
            "schema_sha256": catalog_identity.schema_sha256,
        },
        "derivation_audit": derivation_audit,
        "private_local_catalog_bytes_published": False,
        "derived_combined_catalog_bytes_published": False,
    }


def _prepared_receipt(
    bundle: ProductionForecastBundle,
    *,
    view: ProductionForecastView,
    forecast_created_at: datetime,
    core_payloads: Mapping[str, bytes],
) -> JsonObject:
    record_forecasts = _record_forecast_preimages(view)
    if list(record_forecasts) != bundle.record_forecasts():
        raise P1IssueArtifactError("forecast record identities differ from the frozen bundle")
    return {
        "schema_version": 1,
        "artifact_type": "p1_real_prospective_prepared_receipt_v1",
        "issue_id": view.issue_id,
        "scheduled_issue_time_utc": view.scheduled_issue_time_utc,
        "query_cutoff_utc": view.query_cutoff_utc,
        "forecast_created_at_utc": _utc_text(forecast_created_at),
        "source_snapshot_sha256": view.source_snapshot_sha256,
        "source_request_sha256": view.source_request_sha256,
        "source_boundary_manifest_sha256": P1_SOURCE_BOUNDARY_MANIFEST_SHA256,
        "model_manifest_sha256": P1_MODEL_MANIFEST_SHA256,
        "support_manifest_sha256": view.support_manifest_sha256,
        "code_commit": view.code_commit,
        "record_forecasts": list(record_forecasts),
        "model_summary": [
            {
                "model_id": model.model_id,
                "grid_cell_count": len(view.cells),
                "alarm_cell_count": len(model.alarm_cell_ids),
                "actual_alarm_area_km2": model.actual_alarm_area_km2,
            }
            for model in view.models
        ],
        "B0_reference_area_km2": view.models[0].actual_alarm_area_km2,
        "B0_R30_next_complete_cell_area_km2": (view.models[1].next_complete_cell_area_km2),
        "actual_area_difference_km2": (
            view.models[0].actual_alarm_area_km2 - view.models[1].actual_alarm_area_km2
        ),
        "artifact_sha256": {
            name: _sha256(core_payloads[name]) for name in _CORE_ARTIFACT_FILENAMES
        },
        "future_outcomes_absent": True,
        "value_semantics": "relative_intensity_not_absolute_probability",
        "original_artifacts_immutable": True,
    }


@dataclass(frozen=True, slots=True)
class PreparedIssueArtifacts:
    """Identity of a newly published, write-once local issue directory."""

    issue_directory: Path
    prepared_receipt_sha256: str
    artifact_manifest_sha256: str


@dataclass(frozen=True, slots=True)
class VerifiedPreparedIssue:
    """Immutable values needed for the later ForecastIssueRecord preimage."""

    issue_id: str
    scheduled_issue_time_utc: datetime
    query_cutoff_utc: datetime
    forecast_created_at_utc: datetime
    protocol_model_manifest_sha256: str
    source_boundary_manifest_sha256: str
    source_snapshot_sha256: str
    code_commit: str
    forecasts: tuple[JsonObject, JsonObject]
    static_svg_sha256: str
    offline_interactive_html_sha256: str
    B0_reference_area_km2: float
    B0_R30_next_complete_cell_area_km2: float
    actual_area_difference_km2: float
    prepared_receipt_sha256: str
    artifact_manifest_sha256: str


def prepare_production_issue_artifacts(
    bundle: ProductionForecastBundle,
    *,
    issue_parent: Path,
    code_commit: str,
    forecast_created_at_utc: datetime,
) -> PreparedIssueArtifacts:
    """Seal one bundle into a new public issue directory without overwriting.

    The directory is assembled under a same-parent temporary name and then
    renamed once.  This project's production host is Windows, where ``rename``
    refuses an existing destination.  We also fail before rename if the issue
    directory already exists; no recovery path reuses or repairs it.
    """

    commit = _commit_text(code_commit)
    created = _utc(forecast_created_at_utc, label="forecast_created_at_utc")
    forecast_mapping = bundle.forecast_mapping(code_commit=commit)
    view = parse_production_forecast_view(forecast_mapping)
    _validate_issue_identity(view)
    _validate_bundle_source_counts(bundle, view)
    if not view.query_cutoff <= created < view.scheduled_issue_time:
        raise P1IssueArtifactError("forecast must be created in the legal [Q,T) window")
    query = bundle.acquisition.query_snapshot
    if query is None or created < query.fetch_completed_at_utc:
        raise P1IssueArtifactError("forecast creation cannot precede the sealed ComCat query")
    if view.support_manifest_sha256 != EXPECTED_SUPPORT_MANIFEST_SHA256:
        raise P1IssueArtifactError("support manifest differs from the frozen P1 identity")

    parent = _regular_directory(issue_parent, label="issue_parent")
    target = parent / view.issue_id
    if target.exists() or target.is_symlink():
        raise P1IssueArtifactError("issue directory already exists and may never be overwritten")

    count = bundle.acquisition.count_snapshot
    core_payloads: dict[str, bytes] = {
        COUNT_BODY_FILENAME: count.raw_response_bytes,
        COUNT_HEADERS_FILENAME: canonical_json_bytes(count.captured_response_headers),
        QUERY_BODY_FILENAME: query.raw_response_bytes,
        QUERY_HEADERS_FILENAME: canonical_json_bytes(query.captured_response_headers),
    }
    source_receipt = _source_receipt(bundle, view=view, core_payloads=core_payloads)
    core_payloads[SOURCE_RECEIPT_FILENAME] = canonical_json_bytes(source_receipt)
    core_payloads[FORECAST_GRID_FILENAME] = canonical_json_bytes(forecast_mapping)
    core_payloads[STATIC_SVG_FILENAME] = render_production_forecast_svg(forecast_mapping)
    core_payloads[OFFLINE_HTML_FILENAME] = build_offline_production_forecast_html(
        forecast_mapping
    ).encode("utf-8")
    receipt = _prepared_receipt(
        bundle,
        view=view,
        forecast_created_at=created,
        core_payloads=core_payloads,
    )
    receipt_bytes = canonical_json_bytes(receipt)
    payloads = {**core_payloads, PREPARED_RECEIPT_FILENAME: receipt_bytes}
    manifest: JsonObject = {
        "schema_version": 1,
        "artifact_type": "p1_real_prospective_artifact_manifest_v1",
        "issue_id": view.issue_id,
        "files": [_file_entry(name, payloads[name]) for name in _MANIFESTED_FILENAMES],
    }
    manifest_bytes = canonical_json_bytes(manifest)
    payloads[ARTIFACT_MANIFEST_FILENAME] = manifest_bytes

    temporary = Path(tempfile.mkdtemp(prefix=_TEMP_PREFIX, dir=parent))
    try:
        for name in _ALL_FILENAMES:
            _write_new_file(temporary / name, payloads[name])
        if target.exists() or target.is_symlink():
            raise P1IssueArtifactError(
                "issue directory appeared during preparation; refusing to overwrite it"
            )
        try:
            os.rename(temporary, target)
        except FileExistsError as exc:
            raise P1IssueArtifactError(
                "issue directory appeared during publication; refusing to overwrite it"
            ) from exc
    except BaseException:
        if temporary.exists():
            _cleanup_owned_temp(parent, temporary)
        raise

    verified = verify_prepared_issue_artifacts(target)
    if verified.prepared_receipt_sha256 != _sha256(receipt_bytes):
        raise P1IssueArtifactError("prepared receipt changed after issue directory publication")
    return PreparedIssueArtifacts(
        issue_directory=target,
        prepared_receipt_sha256=verified.prepared_receipt_sha256,
        artifact_manifest_sha256=verified.artifact_manifest_sha256,
    )


def _verify_manifest(issue_directory: Path) -> tuple[Mapping[str, object], dict[str, bytes]]:
    actual_names = {item.name for item in issue_directory.iterdir()}
    if actual_names != set(_ALL_FILENAMES):
        raise P1IssueArtifactError("issue directory files differ from the frozen package layout")
    manifest_bytes = _read_regular_file(issue_directory / ARTIFACT_MANIFEST_FILENAME)
    manifest = _canonical_json_object(manifest_bytes, label="artifact manifest")
    _exact_keys(
        manifest,
        {"schema_version", "artifact_type", "issue_id", "files"},
        label="artifact manifest",
    )
    if (
        manifest.get("schema_version") != 1
        or manifest.get("artifact_type") != "p1_real_prospective_artifact_manifest_v1"
    ):
        raise P1IssueArtifactError("artifact manifest identity differs")
    entries = _sequence(manifest.get("files"), label="artifact manifest files")
    if len(entries) != len(_MANIFESTED_FILENAMES):
        raise P1IssueArtifactError("artifact manifest file count differs")
    payloads: dict[str, bytes] = {}
    observed_names: list[str] = []
    for index, raw_entry in enumerate(entries):
        entry = _mapping(raw_entry, label=f"artifact manifest file {index}")
        _exact_keys(
            entry,
            {"relative_path", "sha256", "byte_count"},
            label=f"artifact manifest file {index}",
        )
        name = entry.get("relative_path")
        if not isinstance(name, str) or name not in _MANIFESTED_FILENAMES:
            raise P1IssueArtifactError("artifact manifest contains an unsafe relative path")
        if name != _MANIFESTED_FILENAMES[index]:
            raise P1IssueArtifactError("artifact manifest order differs from the frozen layout")
        payload = _read_regular_file(issue_directory / name)
        if _sha_text(entry.get("sha256"), label=f"{name} SHA-256") != _sha256(payload):
            raise P1IssueArtifactError(f"artifact hash mismatch: {name}")
        byte_count = entry.get("byte_count")
        if type(byte_count) is not int or byte_count != len(payload):
            raise P1IssueArtifactError(f"artifact byte count mismatch: {name}")
        observed_names.append(name)
        payloads[name] = payload
    if tuple(observed_names) != _MANIFESTED_FILENAMES:
        raise P1IssueArtifactError("artifact manifest contains duplicate or missing files")
    payloads[ARTIFACT_MANIFEST_FILENAME] = manifest_bytes
    return manifest, payloads


def _verified_record_forecasts(
    receipt: Mapping[str, object], view: ProductionForecastView
) -> tuple[JsonObject, JsonObject]:
    expected = _record_forecast_preimages(view)
    raw = _sequence(receipt.get("record_forecasts"), label="record_forecasts")
    if list(raw) != list(expected):
        raise P1IssueArtifactError("prepared record forecast identities are inconsistent")
    return expected


def _reject_private_catalog_disclosure(value: object, *, label: str) -> None:
    """Reject path-like fields in the hash-only local/derived catalogue section."""

    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            if not isinstance(raw_key, str):
                raise P1IssueArtifactError(f"{label} contains a non-string key")
            if "path" in raw_key.casefold() or "bytes" in raw_key.casefold():
                raise P1IssueArtifactError(f"{label} may not disclose catalogue paths or bytes")
            _reject_private_catalog_disclosure(item, label=label)
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for item in value:
            _reject_private_catalog_disclosure(item, label=label)
    elif isinstance(value, str) and (
        re.match(r"^[A-Za-z]:[\\/]", value) is not None or value.startswith(("/", "\\\\"))
    ):
        raise P1IssueArtifactError(f"{label} contains an absolute catalogue path")


def _verify_source_receipt(
    source_receipt: Mapping[str, object],
    *,
    view: ProductionForecastView,
    payloads: Mapping[str, bytes],
) -> None:
    _exact_keys(
        source_receipt,
        {
            "schema_version",
            "artifact_type",
            "issue_id",
            "scheduled_issue_time_utc",
            "query_cutoff_utc",
            "source_snapshot_sha256",
            "source_request_sha256",
            "source_boundary_manifest_sha256",
            "comcat_acquisition",
            "raw_comcat_artifacts",
            "frozen_local_catalog_identity",
            "derived_combined_catalog_identity",
            "derivation_audit",
            "private_local_catalog_bytes_published",
            "derived_combined_catalog_bytes_published",
        },
        label="source receipt",
    )
    if (
        source_receipt.get("schema_version") != 1
        or source_receipt.get("artifact_type") != "p1_real_prospective_source_receipt_v1"
        or source_receipt.get("issue_id") != view.issue_id
        or source_receipt.get("scheduled_issue_time_utc") != view.scheduled_issue_time_utc
        or source_receipt.get("query_cutoff_utc") != view.query_cutoff_utc
        or source_receipt.get("source_snapshot_sha256") != view.source_snapshot_sha256
        or source_receipt.get("source_request_sha256") != view.source_request_sha256
        or source_receipt.get("source_boundary_manifest_sha256")
        != P1_SOURCE_BOUNDARY_MANIFEST_SHA256
        or source_receipt.get("private_local_catalog_bytes_published") is not False
        or source_receipt.get("derived_combined_catalog_bytes_published") is not False
    ):
        raise P1IssueArtifactError("source receipt violates the frozen hash-only source boundary")

    local_identity = _mapping(
        source_receipt.get("frozen_local_catalog_identity"),
        label="frozen local catalogue identity",
    )
    if local_identity != {
        "row_count": FROZEN_CATALOG_ROW_COUNT,
        "file_sha256": FROZEN_CATALOG_FILE_SHA256,
        "content_sha256": FROZEN_CATALOG_CONTENT_SHA256,
        "schema_sha256": FROZEN_CATALOG_SCHEMA_SHA256,
    }:
        raise P1IssueArtifactError("frozen local catalogue identity differs")
    combined_identity = _mapping(
        source_receipt.get("derived_combined_catalog_identity"),
        label="derived combined catalogue identity",
    )
    _exact_keys(
        combined_identity,
        {"row_count", "file_sha256", "content_sha256", "schema_sha256"},
        label="derived combined catalogue identity",
    )
    if (
        type(combined_identity.get("row_count")) is not int
        or cast(int, combined_identity.get("row_count")) <= 0
    ):
        raise P1IssueArtifactError("derived combined catalogue row count must be positive")
    for key in ("file_sha256", "content_sha256", "schema_sha256"):
        _sha_text(combined_identity.get(key), label=f"derived combined catalogue {key}")
    _reject_private_catalog_disclosure(local_identity, label="local catalogue identity")
    _reject_private_catalog_disclosure(combined_identity, label="combined catalogue identity")
    _reject_private_catalog_disclosure(
        source_receipt.get("derivation_audit"), label="catalogue derivation audit"
    )

    acquisition = _mapping(source_receipt.get("comcat_acquisition"), label="ComCat acquisition")
    if _sha256(canonical_json_bytes(acquisition)) != view.source_snapshot_sha256:
        raise P1IssueArtifactError("ComCat acquisition does not match source snapshot SHA-256")
    _exact_keys(
        acquisition,
        {"status", "count_snapshot", "query_snapshot", "unavailable_reason"},
        label="ComCat acquisition",
    )
    if (
        acquisition.get("status") != "available"
        or acquisition.get("unavailable_reason") is not None
    ):
        raise P1IssueArtifactError(
            "prepared issue does not contain an available ComCat acquisition"
        )
    count_snapshot = _mapping(acquisition.get("count_snapshot"), label="count snapshot")
    query_snapshot = _mapping(acquisition.get("query_snapshot"), label="query snapshot")
    request_identity = {
        "count_request_url": count_snapshot.get("request_url"),
        "query_request_url": query_snapshot.get("request_url"),
    }
    if _sha256(canonical_json_bytes(request_identity)) != view.source_request_sha256:
        raise P1IssueArtifactError("ComCat request URLs do not match source request SHA-256")
    for snapshot, body_name, header_name, snapshot_label in (
        (
            count_snapshot,
            COUNT_BODY_FILENAME,
            COUNT_HEADERS_FILENAME,
            "count snapshot",
        ),
        (
            query_snapshot,
            QUERY_BODY_FILENAME,
            QUERY_HEADERS_FILENAME,
            "query snapshot",
        ),
    ):
        if snapshot.get("raw_response_sha256") != _sha256(payloads[body_name]):
            raise P1IssueArtifactError(f"{snapshot_label} raw response hash differs")
        if snapshot.get("response_body_byte_count") != len(payloads[body_name]):
            raise P1IssueArtifactError(f"{snapshot_label} raw response byte count differs")
        headers = _canonical_json_object(payloads[header_name], label=f"{snapshot_label} headers")
        if snapshot.get("captured_response_headers") != headers:
            raise P1IssueArtifactError(f"{snapshot_label} captured headers differ")
        if snapshot.get("response_headers_sha256") != _sha256(payloads[header_name]):
            raise P1IssueArtifactError(f"{snapshot_label} response header hash differs")

    raw_entries = _mapping(source_receipt.get("raw_comcat_artifacts"), label="raw ComCat artifacts")
    raw_names = {
        "count_response": COUNT_BODY_FILENAME,
        "count_headers": COUNT_HEADERS_FILENAME,
        "query_response": QUERY_BODY_FILENAME,
        "query_headers": QUERY_HEADERS_FILENAME,
    }
    if set(raw_entries) != set(raw_names):
        raise P1IssueArtifactError("source receipt raw artifact entries differ")
    for key, name in raw_names.items():
        entry = _mapping(raw_entries.get(key), label=f"raw artifact {key}")
        if (
            entry.get("relative_path") != name
            or entry.get("sha256") != _sha256(payloads[name])
            or entry.get("byte_count") != len(payloads[name])
        ):
            raise P1IssueArtifactError(f"source receipt raw artifact mismatch: {key}")


def verify_prepared_issue_artifacts(issue_directory: Path) -> VerifiedPreparedIssue:
    """Read and fully verify one already prepared issue package without mutation."""

    issue = _regular_directory(issue_directory, label="issue_directory")
    manifest, payloads = _verify_manifest(issue)
    receipt = _canonical_json_object(payloads[PREPARED_RECEIPT_FILENAME], label="prepared receipt")
    _exact_keys(
        receipt,
        {
            "schema_version",
            "artifact_type",
            "issue_id",
            "scheduled_issue_time_utc",
            "query_cutoff_utc",
            "forecast_created_at_utc",
            "source_snapshot_sha256",
            "source_request_sha256",
            "source_boundary_manifest_sha256",
            "model_manifest_sha256",
            "support_manifest_sha256",
            "code_commit",
            "record_forecasts",
            "model_summary",
            "B0_reference_area_km2",
            "B0_R30_next_complete_cell_area_km2",
            "actual_area_difference_km2",
            "artifact_sha256",
            "future_outcomes_absent",
            "value_semantics",
            "original_artifacts_immutable",
        },
        label="prepared receipt",
    )
    if (
        receipt.get("schema_version") != 1
        or receipt.get("artifact_type") != "p1_real_prospective_prepared_receipt_v1"
        or receipt.get("future_outcomes_absent") is not True
        or receipt.get("value_semantics") != "relative_intensity_not_absolute_probability"
        or receipt.get("original_artifacts_immutable") is not True
    ):
        raise P1IssueArtifactError("prepared receipt scientific identity differs")
    artifact_hashes = _mapping(receipt.get("artifact_sha256"), label="artifact_sha256")
    if set(artifact_hashes) != set(_CORE_ARTIFACT_FILENAMES):
        raise P1IssueArtifactError("prepared receipt artifact hashes are incomplete")
    for name in _CORE_ARTIFACT_FILENAMES:
        if _sha_text(artifact_hashes.get(name), label=f"{name} receipt SHA-256") != _sha256(
            payloads[name]
        ):
            raise P1IssueArtifactError(f"prepared receipt hash mismatch: {name}")

    forecast_mapping = _canonical_json_object(
        payloads[FORECAST_GRID_FILENAME], label="forecast grid"
    )
    view = parse_production_forecast_view(forecast_mapping)
    _validate_issue_identity(view)
    if render_production_forecast_svg(forecast_mapping) != payloads[STATIC_SVG_FILENAME]:
        raise P1IssueArtifactError("static SVG differs from deterministic issue rendering")
    expected_html = build_offline_production_forecast_html(forecast_mapping).encode("utf-8")
    if expected_html != payloads[OFFLINE_HTML_FILENAME]:
        raise P1IssueArtifactError("offline HTML differs from deterministic issue rendering")

    issue_id = receipt.get("issue_id")
    if (
        issue_id != view.issue_id
        or issue_id != manifest.get("issue_id")
        or issue.name != view.issue_id
    ):
        raise P1IssueArtifactError("issue identity differs across directory and receipts")
    if (
        receipt.get("scheduled_issue_time_utc") != view.scheduled_issue_time_utc
        or receipt.get("query_cutoff_utc") != view.query_cutoff_utc
        or receipt.get("source_snapshot_sha256") != view.source_snapshot_sha256
        or receipt.get("source_request_sha256") != view.source_request_sha256
        or receipt.get("support_manifest_sha256") != view.support_manifest_sha256
        or receipt.get("code_commit") != view.code_commit
    ):
        raise P1IssueArtifactError("prepared receipt differs from the forecast grid identity")
    if view.support_manifest_sha256 != EXPECTED_SUPPORT_MANIFEST_SHA256:
        raise P1IssueArtifactError("forecast support manifest differs from frozen P1")
    if receipt.get("model_manifest_sha256") != P1_MODEL_MANIFEST_SHA256:
        raise P1IssueArtifactError("forecast model manifest differs from frozen P1")
    if receipt.get("source_boundary_manifest_sha256") != P1_SOURCE_BOUNDARY_MANIFEST_SHA256:
        raise P1IssueArtifactError("forecast source manifest differs from frozen P1")

    source_receipt = _canonical_json_object(
        payloads[SOURCE_RECEIPT_FILENAME], label="source receipt"
    )
    _verify_source_receipt(source_receipt, view=view, payloads=payloads)

    expected_summary = [
        {
            "model_id": model.model_id,
            "grid_cell_count": len(view.cells),
            "alarm_cell_count": len(model.alarm_cell_ids),
            "actual_alarm_area_km2": model.actual_alarm_area_km2,
        }
        for model in view.models
    ]
    if receipt.get("model_summary") != expected_summary:
        raise P1IssueArtifactError("prepared model cell counts or alarm areas differ")

    created = _parse_utc(receipt.get("forecast_created_at_utc"), label="forecast_created_at_utc")
    if not view.query_cutoff <= created < view.scheduled_issue_time:
        raise P1IssueArtifactError("prepared forecast timestamp is outside [Q,T)")
    record_forecasts = _verified_record_forecasts(receipt, view)
    b0_area = receipt.get("B0_reference_area_km2")
    next_area = receipt.get("B0_R30_next_complete_cell_area_km2")
    difference = receipt.get("actual_area_difference_km2")
    if any(
        isinstance(value, bool) or not isinstance(value, int | float)
        for value in (
            b0_area,
            next_area,
            difference,
        )
    ):
        raise P1IssueArtifactError("prepared alarm-area values must be numeric")
    b0_area_float = float(cast(int | float, b0_area))
    next_area_float = float(cast(int | float, next_area))
    difference_float = float(cast(int | float, difference))
    if (
        b0_area_float != view.models[0].actual_alarm_area_km2
        or next_area_float != view.models[1].next_complete_cell_area_km2
        or difference_float
        != view.models[0].actual_alarm_area_km2 - view.models[1].actual_alarm_area_km2
        or not 0.0 <= difference_float < next_area_float <= 625.0
    ):
        raise P1IssueArtifactError("prepared alarm-area fairness values are inconsistent")

    return VerifiedPreparedIssue(
        issue_id=view.issue_id,
        scheduled_issue_time_utc=view.scheduled_issue_time,
        query_cutoff_utc=view.query_cutoff,
        forecast_created_at_utc=created,
        protocol_model_manifest_sha256=P1_MODEL_MANIFEST_SHA256,
        source_boundary_manifest_sha256=P1_SOURCE_BOUNDARY_MANIFEST_SHA256,
        source_snapshot_sha256=view.source_snapshot_sha256,
        code_commit=_commit_text(view.code_commit),
        forecasts=record_forecasts,
        static_svg_sha256=_sha256(payloads[STATIC_SVG_FILENAME]),
        offline_interactive_html_sha256=_sha256(payloads[OFFLINE_HTML_FILENAME]),
        B0_reference_area_km2=b0_area_float,
        B0_R30_next_complete_cell_area_km2=next_area_float,
        actual_area_difference_km2=difference_float,
        prepared_receipt_sha256=_sha256(payloads[PREPARED_RECEIPT_FILENAME]),
        artifact_manifest_sha256=_sha256(payloads[ARTIFACT_MANIFEST_FILENAME]),
    )


def _http_headers_from_captured(value: object, *, label: str) -> dict[str, str]:
    captured = _mapping(value, label=label)
    names = {
        "date": "Date",
        "etag": "ETag",
        "last_modified": "Last-Modified",
        "content_type": "Content-Type",
        "content_length": "Content-Length",
    }
    if set(captured) != set(names):
        raise P1IssueArtifactError(f"{label} fields differ from the frozen header capture")
    headers: dict[str, str] = {}
    for key, output_name in names.items():
        item = captured.get(key)
        if item is None:
            continue
        if not isinstance(item, str):
            raise P1IssueArtifactError(f"{label}.{key} must be a string or null")
        headers[output_name] = item
    return headers


def _exchange_from_snapshot_mapping(
    snapshot: Mapping[str, object],
    *,
    raw_response_bytes: bytes,
    label: str,
) -> ComCatHttpExchange:
    request_url = snapshot.get("request_url")
    http_status = snapshot.get("http_status")
    if not isinstance(request_url, str) or not request_url:
        raise P1IssueArtifactError(f"{label}.request_url must be a non-empty string")
    if type(http_status) is not int:
        raise P1IssueArtifactError(f"{label}.http_status must be an integer")
    return ComCatHttpExchange(
        request_url=request_url,
        fetch_started_at_utc=_parse_utc(
            snapshot.get("fetch_started_at_utc"), label=f"{label}.fetch_started_at_utc"
        ),
        fetch_completed_at_utc=_parse_utc(
            snapshot.get("fetch_completed_at_utc"),
            label=f"{label}.fetch_completed_at_utc",
        ),
        http_status=http_status,
        response_headers=_http_headers_from_captured(
            snapshot.get("captured_response_headers"),
            label=f"{label}.captured_response_headers",
        ),
        raw_response_bytes=raw_response_bytes,
    )


def _rebuild_public_acquisition(
    source_receipt: Mapping[str, object],
    *,
    schedule: P1IssueSchedule,
    payloads: Mapping[str, bytes],
) -> ComCatIssueInputAcquisition:
    acquisition_mapping = _mapping(
        source_receipt.get("comcat_acquisition"), label="ComCat acquisition"
    )
    count_mapping = _mapping(acquisition_mapping.get("count_snapshot"), label="count snapshot")
    query_mapping = _mapping(acquisition_mapping.get("query_snapshot"), label="query snapshot")
    try:
        count = build_comcat_count_snapshot(
            _exchange_from_snapshot_mapping(
                count_mapping,
                raw_response_bytes=payloads[COUNT_BODY_FILENAME],
                label="count snapshot",
            ),
            schedule=schedule,
        )
        query = build_comcat_snapshot(
            _exchange_from_snapshot_mapping(
                query_mapping,
                raw_response_bytes=payloads[QUERY_BODY_FILENAME],
                label="query snapshot",
            ),
            schedule=schedule,
        )
        rebuilt = ComCatIssueInputAcquisition(
            status="available",
            count_snapshot=count,
            query_snapshot=query,
            unavailable_reason=None,
        )
    except (TypeError, ValueError) as exc:
        raise P1IssueArtifactError(
            "public exact ComCat responses cannot reconstruct the sealed acquisition"
        ) from exc
    if count.as_mapping() != count_mapping or query.as_mapping() != query_mapping:
        raise P1IssueArtifactError(
            "reconstructed ComCat snapshots differ from the public source receipt"
        )
    if rebuilt.as_mapping() != acquisition_mapping:
        raise P1IssueArtifactError(
            "reconstructed count-first acquisition differs from the public source receipt"
        )
    return rebuilt


def verify_prepared_issue_against_frozen_inputs(
    issue_directory: Path,
    *,
    local_catalog_bytes: bytes,
    study_area_bytes: bytes,
    support_manifest_bytes: bytes,
    forecast_rebuilder: ForecastRebuilder = build_production_forecast,
) -> VerifiedPreparedIssue:
    """Mechanically replay public raw input through the frozen forecast path.

    This is the remote-readback gate before a ``ForecastIssueRecord`` is built.
    It is stronger than package self-consistency: exact public count/query bytes
    and captured response metadata are parsed again, then joined to caller-held
    frozen local catalogue, study-area, and support bytes.  The recomputed grid,
    visualisations, source identity, and record preimages must match byte for
    byte.  No path or network transport is accepted.

    ``forecast_rebuilder`` exists only so a small synthetic unit fixture can
    exercise this orchestration.  Production callers must leave the default,
    which is :func:`build_production_forecast`.
    """

    for label, payload in (
        ("local_catalog_bytes", local_catalog_bytes),
        ("study_area_bytes", study_area_bytes),
        ("support_manifest_bytes", support_manifest_bytes),
    ):
        if not isinstance(payload, bytes) or not payload:
            raise P1IssueArtifactError(f"{label} must be non-empty exact bytes")
    verified = verify_prepared_issue_artifacts(issue_directory)
    issue = _regular_directory(issue_directory, label="issue_directory")
    _, payloads = _verify_manifest(issue)
    source_receipt = _canonical_json_object(
        payloads[SOURCE_RECEIPT_FILENAME], label="source receipt"
    )
    forecast_mapping = _canonical_json_object(
        payloads[FORECAST_GRID_FILENAME], label="forecast grid"
    )
    schedule = issue_schedule(verified.scheduled_issue_time_utc)
    if schedule.issue_id != verified.issue_id or schedule.query_cutoff_utc != (
        verified.query_cutoff_utc
    ):
        raise P1IssueArtifactError("prepared issue is not on the frozen weekly schedule")
    acquisition = _rebuild_public_acquisition(
        source_receipt,
        schedule=schedule,
        payloads=payloads,
    )
    try:
        rebuilt_bundle = forecast_rebuilder(
            schedule=schedule,
            acquisition=acquisition,
            local_catalog_bytes=local_catalog_bytes,
            study_area_bytes=study_area_bytes,
            support_manifest_bytes=support_manifest_bytes,
        )
        rebuilt_mapping = rebuilt_bundle.forecast_mapping(code_commit=verified.code_commit)
        rebuilt_view = parse_production_forecast_view(rebuilt_mapping)
        _validate_bundle_source_counts(rebuilt_bundle, rebuilt_view)
    except (TypeError, ValueError) as exc:
        raise P1IssueArtifactError(
            "frozen local inputs and public ComCat bytes cannot reproduce the forecast"
        ) from exc
    if canonical_json_bytes(rebuilt_mapping) != payloads[FORECAST_GRID_FILENAME]:
        raise P1IssueArtifactError("frozen input replay produced a different forecast grid")
    if rebuilt_mapping != forecast_mapping:
        raise P1IssueArtifactError("replayed forecast mapping differs semantically")
    if rebuilt_bundle.source_snapshot_sha256 != verified.source_snapshot_sha256:
        raise P1IssueArtifactError("replayed forecast produced a different source identity")
    rebuilt_catalog_identity = rebuilt_bundle.catalog_artifact.catalog.identity
    expected_catalog_identity = {
        "row_count": rebuilt_catalog_identity.row_count,
        "file_sha256": rebuilt_catalog_identity.file_sha256,
        "content_sha256": rebuilt_catalog_identity.content_sha256,
        "schema_sha256": rebuilt_catalog_identity.schema_sha256,
    }
    if source_receipt.get("derived_combined_catalog_identity") != expected_catalog_identity:
        raise P1IssueArtifactError(
            "replayed forecast produced a different combined catalogue identity"
        )
    if source_receipt.get("derivation_audit") != _catalog_derivation_receipt(rebuilt_bundle):
        raise P1IssueArtifactError("replayed forecast produced a different derivation audit")
    if rebuilt_bundle.record_forecasts() != [dict(item) for item in verified.forecasts]:
        raise P1IssueArtifactError("replayed forecast produced different ledger preimages")
    if render_production_forecast_svg(rebuilt_mapping) != payloads[STATIC_SVG_FILENAME]:
        raise P1IssueArtifactError("replayed forecast produced a different static SVG")
    if (
        build_offline_production_forecast_html(rebuilt_mapping).encode("utf-8")
        != payloads[OFFLINE_HTML_FILENAME]
    ):
        raise P1IssueArtifactError("replayed forecast produced a different offline HTML")
    return verified


def build_forecast_issue_record_fields(
    prepared: VerifiedPreparedIssue,
    *,
    protocol_definition_sha256: str,
    authorization_record_sha256: str,
    publication_completed_at_utc: datetime,
    recorded_at_utc: datetime,
) -> JsonObject:
    """Purely build schema fields after remote publication readback.

    ``recorded_at_utc`` is checked here but remains a chain-header argument for
    ``build_next_p1_record``/``append_new_p1_record`` and is therefore not
    duplicated in the returned fields.
    """

    protocol_sha = _sha_text(protocol_definition_sha256, label="protocol definition SHA-256")
    authorization_sha = _sha_text(authorization_record_sha256, label="authorization record SHA-256")
    published = _utc(publication_completed_at_utc, label="publication_completed_at_utc")
    recorded = _utc(recorded_at_utc, label="recorded_at_utc")
    if not (
        prepared.query_cutoff_utc
        <= prepared.forecast_created_at_utc
        <= published
        <= recorded
        < prepared.scheduled_issue_time_utc
    ):
        raise P1IssueArtifactError(
            "forecast timestamps must satisfy Q<=created<=published<=recorded<T"
        )
    if prepared.query_cutoff_utc != prepared.scheduled_issue_time_utc - timedelta(minutes=15):
        raise P1IssueArtifactError("prepared query cutoff is not T minus 15 minutes")
    return {
        "issue_id": prepared.issue_id,
        "status": "on_time",
        "scheduled_issue_time_utc": _utc_text(prepared.scheduled_issue_time_utc),
        "query_cutoff_utc": _utc_text(prepared.query_cutoff_utc),
        "forecast_created_at_utc": _utc_text(prepared.forecast_created_at_utc),
        "publication_completed_at_utc": _utc_text(published),
        "protocol_definition_sha256": protocol_sha,
        "authorization_record_sha256": authorization_sha,
        "model_manifest_sha256": prepared.protocol_model_manifest_sha256,
        "source_boundary_manifest_sha256": prepared.source_boundary_manifest_sha256,
        "source_snapshot_sha256": prepared.source_snapshot_sha256,
        "code_commit": prepared.code_commit,
        "forecasts": [dict(item) for item in prepared.forecasts],
        "static_svg_sha256": prepared.static_svg_sha256,
        "offline_interactive_html_sha256": prepared.offline_interactive_html_sha256,
        "B0_reference_area_km2": prepared.B0_reference_area_km2,
        "B0_R30_next_complete_cell_area_km2": (prepared.B0_R30_next_complete_cell_area_km2),
        "actual_area_difference_km2": prepared.actual_area_difference_km2,
        "area_fairness_status": "passed",
        "original_artifacts_immutable": True,
    }


__all__ = [
    "ARTIFACT_MANIFEST_FILENAME",
    "COUNT_BODY_FILENAME",
    "COUNT_HEADERS_FILENAME",
    "FORECAST_GRID_FILENAME",
    "OFFLINE_HTML_FILENAME",
    "PREPARED_RECEIPT_FILENAME",
    "QUERY_BODY_FILENAME",
    "QUERY_HEADERS_FILENAME",
    "SOURCE_RECEIPT_FILENAME",
    "STATIC_SVG_FILENAME",
    "ForecastRebuilder",
    "P1IssueArtifactError",
    "PreparedIssueArtifacts",
    "VerifiedPreparedIssue",
    "build_forecast_issue_record_fields",
    "prepare_production_issue_artifacts",
    "verify_prepared_issue_against_frozen_inputs",
    "verify_prepared_issue_artifacts",
]
