"""Generate or verify the score-free stage-2R local-support manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, cast

import yaml

from seismoflux.background.catalog import load_earthquake_catalog, load_study_area
from seismoflux.background.local_support_manifest import (
    LocalSupportSourceFile,
    LocalSupportSources,
    background_local_support_manifest_bytes,
    build_background_local_support_manifest,
)
from seismoflux.background.workflow import catalog_completeness_events
from seismoflux.config import sha256_file


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return cast(dict[str, Any], value)


def _load_protocol(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"unable to read local-support protocol: {path}") from exc
    protocol = _mapping(raw, name="local-support protocol")
    if protocol.get("protocol_version") != "0.2.1":
        raise ValueError("manifest generator requires protocol_version 0.2.1")
    return protocol


def _project_path(project_root: Path, value: object, *, name: str) -> tuple[str, Path]:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{name} must be a project-relative POSIX path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or value != relative.as_posix():
        raise ValueError(f"{name} must be a normalized project-relative path")
    return value, (project_root / relative).resolve(strict=True)


def _require_hash(path: Path, expected: object, *, name: str) -> str:
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"{name} expected SHA-256 is invalid")
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"{name} SHA-256 differs from the frozen protocol")
    return observed


def generate(config_path: Path) -> bytes:
    config = config_path.resolve(strict=True)
    project_root = config.parents[1]
    protocol = _load_protocol(config)
    inputs = _mapping(protocol.get("inputs"), name="inputs")
    time_config = _mapping(protocol.get("time"), name="time")
    integration = _mapping(protocol.get("integration"), name="integration")

    earthquake_reference, earthquake_path = _project_path(
        project_root,
        inputs.get("earthquake_dataset_path"),
        name="earthquake dataset",
    )
    study_reference, study_path = _project_path(
        project_root,
        inputs.get("study_area"),
        name="study area",
    )
    earthquake_sha256 = _require_hash(
        earthquake_path,
        inputs.get("earthquake_dataset_sha256"),
        name="earthquake dataset",
    )
    study_sha256 = _require_hash(
        study_path,
        inputs.get("study_area_sha256"),
        name="study area",
    )
    equal_area_crs = integration.get("equal_area_crs")
    final_fit_end_utc = time_config.get("final_parameter_fit_end_utc")
    external_buffer_km = inputs.get("include_external_trigger_buffer_km")
    if not isinstance(equal_area_crs, str) or not isinstance(final_fit_end_utc, str):
        raise ValueError("local-support projection or final fit cutoff is invalid")
    if not isinstance(external_buffer_km, int | float):
        raise ValueError("local-support external buffer is invalid")
    study_area = load_study_area(study_path, equal_area_crs)
    catalog = load_earthquake_catalog(
        earthquake_path,
        study_area=study_area,
        external_buffer_km=float(external_buffer_km),
        maximum_event_time_utc=final_fit_end_utc,
    )
    sources = LocalSupportSources(
        earthquake_dataset=LocalSupportSourceFile(
            path=earthquake_reference,
            sha256=earthquake_sha256,
        ),
        study_area=LocalSupportSourceFile(path=study_reference, sha256=study_sha256),
    )
    manifest = build_background_local_support_manifest(
        catalog_completeness_events(catalog),
        study_area_equal_area=study_area.projected,
        sources=sources,
    )
    return background_local_support_manifest_bytes(manifest)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/background_local_support.yaml"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.check is not None:
        parser.error("--output and --check are mutually exclusive")

    payload = generate(args.config)
    if args.check is not None:
        if args.check.read_bytes() != payload:
            raise SystemExit("local-support manifest differs from deterministic regeneration")
        return 0
    if args.output is None:
        sys.stdout.buffer.write(payload)
    else:
        args.output.write_bytes(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
