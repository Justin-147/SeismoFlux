"""Deterministic byte-level inventory of external raw inputs."""

from __future__ import annotations

import csv
import hashlib
import os
import stat
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath
from typing import Any, Literal, Self

from pydantic import Field, field_validator, model_validator

from seismoflux.config import StrictModel, load_yaml_mapping, normalize_relative_path


class InventorySource(StrictModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    category: str = Field(min_length=1)
    kind: Literal["file", "directory"]
    path: str
    recursive: bool
    license_status: Literal["unknown_no_redistribution"]
    include_extensions: tuple[str, ...] | None = None

    @field_validator("path")
    @classmethod
    def validate_relative_source_path(cls, value: str) -> str:
        try:
            return normalize_relative_path(value)
        except ValueError as exc:
            raise ValueError("source paths must be relative to source_root") from exc

    @field_validator("include_extensions")
    @classmethod
    def validate_extensions(cls, value: tuple[str, ...] | None) -> tuple[str, ...] | None:
        if value is None:
            return None
        normalized = tuple(extension.lower() for extension in value)
        if not normalized or len(normalized) != len(set(normalized)):
            raise ValueError("include_extensions must be non-empty and unique")
        if any(not extension.startswith(".") or len(extension) < 2 for extension in normalized):
            raise ValueError("each included extension must begin with a dot")
        return normalized

    @model_validator(mode="after")
    def validate_recursive_flag(self) -> Self:
        if self.kind == "file" and self.recursive:
            raise ValueError("file sources cannot be recursive")
        return self


class DataSourcesConfig(StrictModel):
    schema_version: Literal[1]
    source_root_env: str = Field(pattern=r"^[A-Z][A-Z0-9_]*$")
    source_root: str = Field(min_length=1)
    source_root_resolution: Literal["environment_overrides_config"] = "environment_overrides_config"
    inventory_output: str
    sources: tuple[InventorySource, ...]

    @field_validator("inventory_output")
    @classmethod
    def validate_inventory_output(cls, value: str) -> str:
        try:
            return normalize_relative_path(value)
        except ValueError as exc:
            raise ValueError("inventory output must be project-relative") from exc

    @model_validator(mode="after")
    def validate_unique_sources(self) -> Self:
        source_ids = [source.id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("inventory source IDs must be unique")
        if not source_ids:
            raise ValueError("at least one inventory source is required")
        return self

    def resolved_source_root(self) -> Path:
        """Use an environment override without writing it to generated manifests."""

        raw_value = os.environ.get(self.source_root_env, self.source_root)
        path = Path(raw_value)
        if not raw_value or not (path.is_absolute() or PureWindowsPath(raw_value).is_absolute()):
            raise ValueError("resolved source_root must be an absolute path")
        return path


@dataclass(frozen=True, slots=True)
class InventoryRecord:
    source_id: str
    source_category: str
    relative_path: str
    file_extension: str
    size_bytes: int
    modified_at_utc: str
    sha256: str
    license_status: str

    def as_row(self) -> dict[str, str | int]:
        return {
            "source_id": self.source_id,
            "source_category": self.source_category,
            "relative_path": self.relative_path,
            "file_extension": self.file_extension,
            "size_bytes": self.size_bytes,
            "modified_at_utc": self.modified_at_utc,
            "sha256": self.sha256,
            "license_status": self.license_status,
        }


INVENTORY_COLUMNS = tuple(InventoryRecord.__dataclass_fields__)


def load_data_sources(path: str | Path) -> DataSourcesConfig:
    """Load the migration-only source configuration."""

    return DataSourcesConfig.model_validate(load_yaml_mapping(Path(path)))


def _is_reparse_point(path: Path) -> bool:
    if path.is_symlink():
        return True
    file_attributes = getattr(path.stat(follow_symlinks=False), "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(file_attributes & reparse_flag)


def _iter_source_files(root: Path, source: InventorySource) -> list[Path]:
    raw_target = root / Path(source.path)
    if not raw_target.exists():
        raise FileNotFoundError(f"configured source does not exist: {source.id}")
    if _is_reparse_point(raw_target):
        raise ValueError(f"reparse points are forbidden in raw inputs: {source.id}")
    target = raw_target.resolve()
    resolved_root = root.resolve()
    if not target.is_relative_to(resolved_root):
        raise ValueError(f"configured source escapes source_root: {source.id}")
    if source.kind == "file":
        if not target.is_file():
            raise ValueError(f"configured file source is not a file: {source.id}")
        return [target]
    if not target.is_dir():
        raise ValueError(f"configured directory source is not a directory: {source.id}")

    discovered: list[Path] = []
    for current_dir, dir_names, file_names in os.walk(target, followlinks=False):
        current = Path(current_dir)
        dir_names.sort(key=str.casefold)
        file_names.sort(key=str.casefold)
        for directory_name in dir_names:
            candidate = current / directory_name
            if _is_reparse_point(candidate):
                raise ValueError(f"reparse points are forbidden in raw inputs: {source.id}")
        for file_name in file_names:
            candidate = current / file_name
            if _is_reparse_point(candidate):
                raise ValueError(f"reparse points are forbidden in raw inputs: {source.id}")
            if candidate.is_file() and (
                source.include_extensions is None
                or candidate.suffix.lower() in source.include_extensions
            ):
                discovered.append(candidate)
        if not source.recursive:
            dir_names.clear()
    return discovered


def _stable_file_snapshot(path: Path, chunk_size: int = 1024 * 1024) -> tuple[os.stat_result, str]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
        after = os.fstat(handle.fileno())
    final = path.stat()

    def signature(metadata: os.stat_result) -> tuple[int, int, int, int]:
        return (metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns, metadata.st_ino)

    if signature(before) != signature(after) or signature(before) != signature(final):
        raise ValueError(f"source file changed while it was being inventoried: {path.name}")
    return before, digest.hexdigest()


def build_inventory(config: DataSourcesConfig) -> list[InventoryRecord]:
    """Hash every configured source file without parsing or modifying it."""

    root = config.resolved_source_root()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError("configured source_root is unavailable")
    if _is_reparse_point(root):
        raise ValueError("source_root may not be a reparse point")
    root = root.resolve()

    records: list[InventoryRecord] = []
    for source in config.sources:
        for path in _iter_source_files(root, source):
            metadata, digest = _stable_file_snapshot(path)
            modified = datetime.fromtimestamp(metadata.st_mtime, UTC).isoformat(timespec="seconds")
            records.append(
                InventoryRecord(
                    source_id=source.id,
                    source_category=source.category,
                    relative_path=path.relative_to(root).as_posix(),
                    file_extension=path.suffix.lower(),
                    size_bytes=metadata.st_size,
                    modified_at_utc=modified.replace("+00:00", "Z"),
                    sha256=digest,
                    license_status=source.license_status,
                )
            )
    records.sort(
        key=lambda item: (item.source_id, item.relative_path.casefold(), item.relative_path)
    )
    return records


def write_inventory(records: list[InventoryRecord], output_path: Path) -> None:
    """Atomically write a stable UTF-8 CSV without a volatile generation timestamp."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            writer = csv.DictWriter(handle, fieldnames=INVENTORY_COLUMNS, lineterminator="\n")
            writer.writeheader()
            for record in records:
                writer.writerow(record.as_row())
        os.replace(temporary_name, output_path)
    except Exception:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise


def inventory_summary(records: list[InventoryRecord]) -> dict[str, Any]:
    """Return non-sensitive counts suitable for a run manifest."""

    return {
        "file_count": len(records),
        "total_size_bytes": sum(record.size_bytes for record in records),
        "source_counts": {
            source_id: sum(record.source_id == source_id for record in records)
            for source_id in sorted({record.source_id for record in records})
        },
    }
