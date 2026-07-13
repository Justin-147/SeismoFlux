"""Bind configured raw sources to their tracked inventory entries."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from seismoflux.data.common import SourceFile
from seismoflux.inventory import DataSourcesConfig

EXPECTED_INVENTORY_COLUMNS = (
    "source_id",
    "source_category",
    "relative_path",
    "file_extension",
    "size_bytes",
    "modified_at_utc",
    "sha256",
    "license_status",
)


def load_inventory_sources(
    source_config: DataSourcesConfig,
    inventory_path: Path,
) -> dict[str, tuple[SourceFile, ...]]:
    """Resolve each inventory row beneath source_root and reject catalog drift."""

    source_root = source_config.resolved_source_root().resolve()
    configured_ids = {source.id for source in source_config.sources}
    grouped: dict[str, list[SourceFile]] = defaultdict(list)
    seen_paths: set[str] = set()

    with inventory_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != EXPECTED_INVENTORY_COLUMNS:
            raise ValueError("source inventory columns do not match the stage-0 contract")
        for row in reader:
            source_id = row["source_id"]
            relative_path = row["relative_path"]
            digest = row["sha256"]
            if source_id not in configured_ids:
                raise ValueError(f"inventory uses an unknown source ID: {source_id}")
            if relative_path in seen_paths:
                raise ValueError(f"duplicate path in source inventory: {relative_path}")
            invalid_digest = len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            )
            if invalid_digest:
                raise ValueError(f"invalid SHA-256 in source inventory: {relative_path}")
            path = (source_root / Path(relative_path)).resolve()
            if not path.is_relative_to(source_root):
                raise ValueError(f"inventory path escapes source_root: {relative_path}")
            grouped[source_id].append(
                SourceFile(
                    source_id=source_id,
                    relative_path=relative_path,
                    path=path,
                    sha256=digest,
                    modified_at_utc=row["modified_at_utc"],
                )
            )
            seen_paths.add(relative_path)

    if set(grouped) != configured_ids:
        missing = sorted(configured_ids - set(grouped))
        raise ValueError(f"source inventory is missing configured source IDs: {missing}")
    return {
        source_id: tuple(sorted(files, key=lambda item: item.relative_path.casefold()))
        for source_id, files in sorted(grouped.items())
    }


def require_single_source(
    source_map: dict[str, tuple[SourceFile, ...]], source_id: str
) -> SourceFile:
    files = source_map.get(source_id, ())
    if len(files) != 1:
        raise ValueError(f"source role requires exactly one file: {source_id}")
    return files[0]
