from __future__ import annotations

from pathlib import Path

import pytest

from seismoflux.config import sha256_file
from seismoflux.inventory import (
    DataSourcesConfig,
    build_inventory,
    inventory_summary,
    write_inventory,
)


def _source_config(root: Path) -> DataSourcesConfig:
    return DataSourcesConfig.model_validate(
        {
            "schema_version": 1,
            "source_root_env": "SEISMOFLUX_TEST_SOURCE_ROOT",
            "source_root": str(root),
            "inventory_output": "data/manifests/source_inventory.csv",
            "sources": [
                {
                    "id": "tables",
                    "category": "synthetic",
                    "kind": "directory",
                    "path": "中文目录",
                    "recursive": True,
                    "license_status": "unknown_no_redistribution",
                },
                {
                    "id": "catalog",
                    "category": "synthetic",
                    "kind": "file",
                    "path": "目录.eqt",
                    "recursive": False,
                    "license_status": "unknown_no_redistribution",
                },
            ],
        }
    )


def test_inventory_is_sorted_relative_and_byte_stable(tmp_path: Path) -> None:
    source_root = tmp_path / "原始数据"
    table_dir = source_root / "中文目录"
    table_dir.mkdir(parents=True)
    (table_dir / "乙.XLS").write_bytes(b"second")
    (table_dir / "甲.XLS").write_bytes(b"first")
    (source_root / "目录.eqt").write_bytes(b"catalog")

    records = build_inventory(_source_config(source_root))

    assert [record.source_id for record in records] == ["catalog", "tables", "tables"]
    assert all(not Path(record.relative_path).is_absolute() for record in records)
    assert {record.relative_path for record in records} == {
        "目录.eqt",
        "中文目录/乙.XLS",
        "中文目录/甲.XLS",
    }
    assert all(len(record.sha256) == 64 for record in records)

    first_output = tmp_path / "first.csv"
    second_output = tmp_path / "second.csv"
    write_inventory(records, first_output)
    write_inventory(build_inventory(_source_config(source_root)), second_output)

    assert first_output.read_bytes() == second_output.read_bytes()
    assert sha256_file(first_output) == sha256_file(second_output)
    assert inventory_summary(records) == {
        "file_count": 3,
        "total_size_bytes": 18,
        "source_counts": {"catalog": 1, "tables": 2},
    }


def test_inventory_fails_when_a_source_is_missing(tmp_path: Path) -> None:
    source_root = tmp_path / "raw"
    (source_root / "中文目录").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="catalog"):
        build_inventory(_source_config(source_root))


def test_environment_can_override_migration_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = tmp_path / "configured"
    override = tmp_path / "override"
    (override / "中文目录").mkdir(parents=True)
    (override / "目录.eqt").write_bytes(b"catalog")
    monkeypatch.setenv("SEISMOFLUX_TEST_SOURCE_ROOT", str(override))

    records = build_inventory(_source_config(configured))

    assert len(records) == 1
    assert records[0].relative_path == "目录.eqt"


def test_inventory_extension_allowlist_excludes_legacy_output(tmp_path: Path) -> None:
    source_root = tmp_path / "raw"
    plotbd = source_root / "PlotBD"
    plotbd.mkdir(parents=True)
    (plotbd / "fault.gmt").write_bytes(b"geometry")
    (plotbd / "All_abn.png").write_bytes(b"legacy output")
    config = DataSourcesConfig.model_validate(
        {
            "schema_version": 1,
            "source_root_env": "SEISMOFLUX_UNUSED_SOURCE_ROOT",
            "source_root": str(source_root),
            "inventory_output": "data/manifests/source_inventory.csv",
            "sources": [
                {
                    "id": "plotbd",
                    "category": "geometry",
                    "kind": "directory",
                    "path": "PlotBD",
                    "recursive": True,
                    "license_status": "unknown_no_redistribution",
                    "include_extensions": [".GMT", ".dat"],
                }
            ],
        }
    )

    records = build_inventory(config)

    assert [record.relative_path for record in records] == ["PlotBD/fault.gmt"]


@pytest.mark.parametrize("invalid_path", [r"\outside", "C:relative", "/outside"])
def test_source_paths_reject_rooted_and_drive_relative_forms(
    tmp_path: Path, invalid_path: str
) -> None:
    raw = _source_config(tmp_path).model_dump(mode="json")
    raw["sources"][0]["path"] = invalid_path

    with pytest.raises(ValueError, match="relative"):
        DataSourcesConfig.model_validate(raw)


def test_resolved_source_root_must_be_absolute(tmp_path: Path) -> None:
    config = _source_config(tmp_path)
    raw = config.model_dump(mode="json")
    raw["source_root"] = "relative/raw"
    relative = DataSourcesConfig.model_validate(raw)

    with pytest.raises(ValueError, match="absolute"):
        relative.resolved_source_root()
