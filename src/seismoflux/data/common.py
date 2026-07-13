"""Shared deterministic and read-only primitives for data ingestion."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import stat
import tempfile
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

PROJECT_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "https://github.com/Justin-147/SeismoFlux")
BEIJING_FIXED_OFFSET = timezone(timedelta(hours=8), name="UTC+08:00")


@dataclass(frozen=True, slots=True)
class SourceFile:
    """A source path bound to the immutable inventory entry that authorized it."""

    source_id: str
    relative_path: str
    path: Path
    sha256: str
    modified_at_utc: str


def normalize_text(value: object | None) -> str | None:
    """Apply conservative Unicode and whitespace normalization without changing semantics."""

    if value is None:
        return None
    text = unicodedata.normalize("NFKC", str(value))
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def canonical_decimal(value: object, places: str | None = None) -> str:
    """Return a locale-independent decimal representation suitable for stable identifiers."""

    try:
        decimal_value = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"not a finite decimal: {value!r}") from exc
    if not decimal_value.is_finite():
        raise ValueError(f"not a finite decimal: {value!r}")
    if places is not None:
        decimal_value = decimal_value.quantize(Decimal(places))
    normalized = format(decimal_value, "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized or "0"


def canonical_json_bytes(value: Any) -> bytes:
    """Encode JSON with a stable key order and no insignificant whitespace."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def stable_uuid(kind: str, *parts: object) -> str:
    """Create a namespaced UUIDv5 from explicitly ordered semantic parts."""

    name = "\0".join([kind, *("" if part is None else str(part) for part in parts)])
    return str(uuid.uuid5(PROJECT_NAMESPACE, name))


def stable_token(prefix: str, *parts: object, length: int = 26) -> str:
    """Create a short base32 SHA-256 token for stable public identifiers."""

    payload = "\0".join([prefix, *("" if part is None else str(part) for part in parts)])
    token = base64.b32encode(hashlib.sha256(payload.encode("utf-8")).digest()).decode("ascii")
    return f"{prefix}_{token.rstrip('=').lower()[:length]}"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_reparse_point(path: Path) -> bool:
    if path.is_symlink():
        return True
    attributes = getattr(path.stat(follow_symlinks=False), "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def read_stable_bytes(source: SourceFile) -> bytes:
    """Read a source once and prove that its bytes and metadata match the inventory."""

    if _is_reparse_point(source.path):
        raise ValueError(f"reparse points are forbidden in raw inputs: {source.relative_path}")
    with source.path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        payload = handle.read()
        after = os.fstat(handle.fileno())
    final = source.path.stat()

    def signature(metadata: os.stat_result) -> tuple[int, int, int, int]:
        return (metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns, metadata.st_ino)

    if signature(before) != signature(after) or signature(before) != signature(final):
        raise ValueError(f"source changed while it was being read: {source.relative_path}")
    digest = sha256_bytes(payload)
    if digest != source.sha256:
        raise ValueError(f"source hash differs from inventory: {source.relative_path}")
    return payload


def conservative_available_at(*report_dates: date) -> datetime:
    """Use the day after the latest reported date at fixed UTC+08 midnight."""

    latest = max(report_dates)
    local = datetime.combine(latest + timedelta(days=1), time.min, BEIJING_FIXED_OFFSET)
    return local.astimezone(UTC)


def fixed_local_midnight(value: date) -> datetime:
    """Represent a source date at explicitly assumed fixed UTC+08 midnight."""

    return datetime.combine(value, time.min, BEIJING_FIXED_OFFSET)


def write_json_atomic(path: Path, value: Any) -> None:
    """Write stable UTF-8 JSON without leaving a partial destination."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(payload)
        os.replace(temporary_name, path)
    except Exception:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise
