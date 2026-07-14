"""Command-line contract for the staged SeismoFlux implementation."""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Protocol

from seismoflux import __version__
from seismoflux.background.config import BackgroundConfig, load_project_background_config
from seismoflux.background.execution import build_background_plan
from seismoflux.config import (
    SeismoFluxConfig,
    load_config,
    normalize_relative_path,
    project_root_for,
    resolve_project_path,
    sha256_file,
)
from seismoflux.data.pipeline import ingest_stage1, validate_stage1_data
from seismoflux.data.settings import IngestionSettings, load_ingestion_settings
from seismoflux.inventory import (
    DataSourcesConfig,
    build_inventory,
    inventory_summary,
    load_data_sources,
    write_inventory,
)
from seismoflux.run_manifest import build_run_manifest, emit_run_manifest


@dataclass(frozen=True, slots=True)
class CommandSpec:
    stage: int
    implemented: bool
    planned_inputs: tuple[str, ...]
    planned_outputs: tuple[str, ...]


class _BackgroundManifestResult(Protocol):
    def to_manifest_details(self) -> dict[str, object]: ...


def run_background_stage2(
    config_path: Path,
    *,
    progress: Callable[[str], None],
) -> _BackgroundManifestResult:
    """Lazily import the scientific runner only for an actual stage-2 execution."""

    from seismoflux.background.runner import run_background_stage2 as execute

    return execute(config_path, progress=progress)


COMMAND_SPECS: dict[str, CommandSpec] = {
    "inventory": CommandSpec(0, True, ("external raw files",), ("source inventory CSV",)),
    "ingest": CommandSpec(1, True, ("external raw files",), ("standardized data",)),
    "validate-data": CommandSpec(1, True, ("standardized data",), ("data quality report",)),
    "build-background": CommandSpec(2, True, ("earthquake catalog",), ("background registry",)),
    "build-anomaly-history": CommandSpec(
        3, False, ("anomaly observations",), ("anomaly state history",)
    ),
    "train": CommandSpec(4, False, ("feature stores",), ("experiment model",)),
    "backtest": CommandSpec(4, False, ("experiment model",), ("backtest metrics",)),
    "optimize-regions": CommandSpec(8, False, ("continuous intensity",), ("forecast regions",)),
    "freeze": CommandSpec(9, False, ("validated experiment",), ("frozen model",)),
    "forecast": CommandSpec(9, False, ("frozen model and current data",), ("forecast archive",)),
    "mature": CommandSpec(9, False, ("forecast archive and events",), ("maturity status",)),
    "render": CommandSpec(10, False, ("forecast archive",), ("static and interactive maps",)),
    "validate-release": CommandSpec(10, False, ("forecast archive",), ("release validation",)),
}


def _iso_date(value: str) -> str:
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None:
        raise argparse.ArgumentTypeError("expected an ISO date in YYYY-MM-DD form")
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected an ISO date in YYYY-MM-DD form") from exc
    return value


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dry-run", action="store_true", help="validate and emit a plan only")
    parser.add_argument(
        "--manifest",
        metavar="PATH",
        help="also write the machine-readable run manifest; '-' means stdout only",
    )


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/base.yaml", help="main project YAML")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="seismoflux",
        description="Auditable, anomaly-informed seismicity forecasting research tools.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    inventory = subparsers.add_parser("inventory", help="hash external raw inputs without parsing")
    _add_config(inventory)
    inventory.add_argument("--output", help="override the project-relative inventory CSV path")
    _add_common_options(inventory)

    for command in ("ingest", "validate-data", "build-background", "build-anomaly-history"):
        command_parser = subparsers.add_parser(command)
        _add_config(command_parser)
        _add_common_options(command_parser)

    for command in ("train", "backtest", "optimize-regions", "freeze"):
        command_parser = subparsers.add_parser(command)
        _add_config(command_parser)
        command_parser.add_argument("--experiment", required=True)
        _add_common_options(command_parser)

    forecast = subparsers.add_parser("forecast")
    _add_config(forecast)
    forecast.add_argument("--issue-date", required=True, type=_iso_date)
    forecast.add_argument("--model", required=True)
    _add_common_options(forecast)

    mature = subparsers.add_parser("mature")
    _add_config(mature)
    mature.add_argument("--as-of", required=True, type=_iso_date)
    _add_common_options(mature)

    for command in ("render", "validate-release"):
        command_parser = subparsers.add_parser(command)
        _add_config(command_parser)
        command_parser.add_argument("--issue-id", required=True)
        _add_common_options(command_parser)

    return parser


def _safe_arguments(namespace: argparse.Namespace) -> dict[str, Any]:
    return {
        key: value
        for key, value in vars(namespace).items()
        if key not in {"command", "config", "manifest"} and value is not None
    }


def _validated_manifest_destination(
    destination: str | None,
    *,
    config_path: Path,
    protected_paths: list[Path],
    protected_directories: list[Path] | None = None,
) -> str | None:
    if destination is None or destination == "-":
        return destination
    raw_path = Path(destination)
    if raw_path.suffix.lower() != ".json":
        raise ValueError("run manifest destination must use a .json extension")
    if raw_path.is_absolute():
        resolved = raw_path.resolve()
    else:
        resolved = resolve_project_path(config_path, normalize_relative_path(destination))

    project_root = project_root_for(config_path).resolve()
    standard_protected = [
        config_path.resolve(),
        project_root / "uv.lock",
    ]
    if any(resolved == path.resolve() for path in [*standard_protected, *protected_paths]):
        raise ValueError("run manifest destination collides with a protected project artifact")
    for directory in protected_directories or []:
        protected_root = directory.resolve()
        if resolved == protected_root or resolved.is_relative_to(protected_root):
            raise ValueError("run manifest destination collides with a protected project artifact")
    return str(resolved)


def _run_inventory(namespace: argparse.Namespace) -> int:
    config_path = Path(namespace.config)
    config = load_config(config_path)
    data_sources_path = resolve_project_path(config_path, config.config_files.data_sources)
    source_config = load_data_sources(data_sources_path)
    output_reference = normalize_relative_path(namespace.output or source_config.inventory_output)
    output_path = resolve_project_path(config_path, output_reference)
    manifest_destination = _validated_manifest_destination(
        namespace.manifest,
        config_path=config_path,
        protected_paths=[
            output_path,
            data_sources_path,
            resolve_project_path(config_path, config.config_files.research_protocol),
            resolve_project_path(config_path, config.config_files.operating_points),
        ],
    )

    details: dict[str, Any] = {
        "source_count": len(source_config.sources),
        "source_ids": [source.id for source in source_config.sources],
    }
    status = "planned"
    if not namespace.dry_run:
        records = build_inventory(source_config)
        write_inventory(records, output_path)
        details.update(inventory_summary(records))
        details["inventory_sha256"] = sha256_file(output_path)
        status = "completed"

    manifest = build_run_manifest(
        command="inventory",
        dry_run=namespace.dry_run,
        implementation_stage=0,
        implementation_status="implemented",
        status=status,
        arguments=_safe_arguments(namespace),
        config_path=config_path,
        config=config,
        planned_inputs=[config.config_files.data_sources, "configured external source IDs"],
        planned_outputs=[output_reference],
        details=details,
    )
    emit_run_manifest(manifest, manifest_destination)
    return 0


def _load_stage1_context(
    config_path: Path,
) -> tuple[
    SeismoFluxConfig,
    Path,
    IngestionSettings,
    Path,
    DataSourcesConfig,
]:
    config = load_config(config_path)
    ingestion_path = resolve_project_path(config_path, config.config_files.ingestion)
    settings = load_ingestion_settings(ingestion_path)
    data_sources_path = resolve_project_path(config_path, config.config_files.data_sources)
    source_config = load_data_sources(data_sources_path)

    configured_source_ids = {source.id for source in source_config.sources}
    role_source_ids = set(settings.source_roles.model_dump().values())
    if role_source_ids != configured_source_ids:
        missing_roles = sorted(configured_source_ids - role_source_ids)
        unknown_roles = sorted(role_source_ids - configured_source_ids)
        raise ValueError(
            "ingestion source roles do not exactly cover configured sources; "
            f"missing={missing_roles}, unknown={unknown_roles}"
        )
    if settings.source_inventory != source_config.inventory_output:
        raise ValueError("ingestion and source configurations disagree on the inventory path")
    if config.study_area.polygon != settings.outputs.study_area_geojson:
        raise ValueError("base and ingestion configurations disagree on the study-area path")
    return config, ingestion_path, settings, data_sources_path, source_config


def _stage1_output_references(settings: IngestionSettings) -> list[str]:
    outputs = settings.outputs
    return [
        outputs.processed_root,
        outputs.contracts_root,
        outputs.data_catalog,
        outputs.quality_json,
        outputs.quality_markdown,
        outputs.study_area_geojson,
    ]


def _run_stage1(namespace: argparse.Namespace) -> int:
    command = str(namespace.command)
    config_path = Path(namespace.config)
    config, ingestion_path, settings, data_sources_path, source_config = _load_stage1_context(
        config_path
    )
    project_paths = [
        data_sources_path,
        ingestion_path,
        resolve_project_path(config_path, config.config_files.research_protocol),
        resolve_project_path(config_path, config.config_files.operating_points),
        resolve_project_path(config_path, settings.source_inventory),
        resolve_project_path(config_path, settings.outputs.data_catalog),
        resolve_project_path(config_path, settings.outputs.quality_json),
        resolve_project_path(config_path, settings.outputs.quality_markdown),
        resolve_project_path(config_path, settings.outputs.study_area_geojson),
    ]
    manifest_destination = _validated_manifest_destination(
        namespace.manifest,
        config_path=config_path,
        protected_paths=project_paths,
        protected_directories=[
            resolve_project_path(config_path, settings.outputs.processed_root),
            resolve_project_path(config_path, settings.outputs.contracts_root),
        ],
    )

    source_ids = sorted(source.id for source in source_config.sources)
    details: dict[str, Any] = {
        "contract_version": settings.contract_version,
        "source_count": len(source_ids),
        "source_ids": source_ids,
        "source_roles": settings.source_roles.model_dump(),
    }
    dry_run = bool(namespace.dry_run)
    status = "planned"
    if not dry_run:
        if command == "ingest":
            execution_details = ingest_stage1(
                config_path=config_path,
                config=config,
                ingestion_path=ingestion_path,
                settings=settings,
                source_config=source_config,
            )
        else:
            execution_details = validate_stage1_data(
                config_path=config_path,
                settings=settings,
            )
        details.update(execution_details)
        status = "completed"

    output_references = _stage1_output_references(settings)
    planned_inputs = [settings.source_inventory, *source_ids]
    if command == "validate-data":
        planned_inputs = [
            settings.outputs.data_catalog,
            settings.outputs.quality_json,
            settings.outputs.processed_root,
            settings.outputs.contracts_root,
        ]
    manifest = build_run_manifest(
        command=command,
        dry_run=dry_run,
        implementation_stage=1,
        implementation_status="implemented",
        status=status,
        arguments=_safe_arguments(namespace),
        config_path=config_path,
        config=config,
        planned_inputs=planned_inputs,
        planned_outputs=output_references if command == "ingest" else ["validation result"],
        details=details,
    )
    emit_run_manifest(manifest, manifest_destination)
    return 0


def _background_input_references(background: BackgroundConfig) -> list[str]:
    return [
        background.inputs.environment_lock,
        background.inputs.data_catalog,
        background.inputs.earthquake_dataset_path,
        background.inputs.study_area,
        background.inputs.issue_manifest,
        background.numerical_regression.production_fixture,
        background.numerical_regression.oracle_metadata,
    ]


def _background_output_references(background: BackgroundConfig) -> list[str]:
    return [
        background.outputs.processed_root,
        background.outputs.model_root,
        background.outputs.backtest_root,
        background.outputs.experiment_root,
        background.outputs.registry,
        background.outputs.report,
    ]


def _background_progress(message: str) -> None:
    sys.stderr.write(f"阶段2背景基线: {message}\n")
    sys.stderr.flush()


def _stderr_best_effort(message: str) -> None:
    """Report a secondary CLI condition without changing the scientific outcome."""

    with suppress(Exception):
        sys.stderr.write(message)


def _run_background(namespace: argparse.Namespace) -> int:
    config_path = Path(namespace.config)
    project = load_config(config_path)
    background = load_project_background_config(config_path)
    input_references = _background_input_references(background)
    output_references = _background_output_references(background)
    manifest_destination = _validated_manifest_destination(
        namespace.manifest,
        config_path=config_path,
        protected_paths=[
            resolve_project_path(config_path, project.config_files.background),
            *(resolve_project_path(config_path, item) for item in input_references),
            resolve_project_path(config_path, background.outputs.registry),
            resolve_project_path(config_path, background.outputs.report),
        ],
        protected_directories=[
            resolve_project_path(config_path, background.outputs.processed_root),
            resolve_project_path(config_path, background.outputs.model_root),
            resolve_project_path(config_path, background.outputs.backtest_root),
            resolve_project_path(config_path, background.outputs.experiment_root),
        ],
    )
    plan = build_background_plan(background, project)
    dry_run = bool(namespace.dry_run)
    details = plan.to_manifest_details()
    if dry_run:
        manifest = build_run_manifest(
            command="build-background",
            dry_run=True,
            implementation_stage=2,
            implementation_status="implemented",
            status="planned",
            arguments=_safe_arguments(namespace),
            config_path=config_path,
            config=project,
            planned_inputs=[project.config_files.background, *input_references],
            planned_outputs=output_references,
            details=details,
        )
        emit_run_manifest(manifest, manifest_destination)
        return 0

    try:
        result = run_background_stage2(
            config_path,
            progress=_background_progress,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        failed_details = {
            **details,
            "failure": {
                "error_type": type(exc).__name__,
                "message": str(exc),
                "fixed_delivery_confirmed": False,
                "partial_content_addressed_bundles_may_exist": True,
            },
        }
        reporting_error: Exception | None = None
        try:
            manifest = build_run_manifest(
                command="build-background",
                dry_run=False,
                implementation_stage=2,
                implementation_status="implemented",
                status="failed",
                arguments=_safe_arguments(namespace),
                config_path=config_path,
                config=project,
                planned_inputs=[project.config_files.background, *input_references],
                planned_outputs=output_references,
                details=failed_details,
            )
            emit_run_manifest(manifest, manifest_destination)
        except Exception as manifest_exc:
            reporting_error = manifest_exc
        _stderr_best_effort(f"seismoflux: {exc}\n")
        if reporting_error is not None:
            _stderr_best_effort(
                "seismoflux: 失败运行清单报告也失败"
                f" ({type(reporting_error).__name__}: {reporting_error}); "
                "原始阶段2错误保持有效, 固定交付未确认。\n"
            )
        return 2

    # The production runner returns only after all content-addressed bundles and
    # the fixed registry/report pair are durably published. Reporting below is
    # auxiliary and must never reclassify that completed scientific delivery.
    try:
        details = result.to_manifest_details()
        manifest = build_run_manifest(
            command="build-background",
            dry_run=False,
            implementation_stage=2,
            implementation_status="implemented",
            status="completed",
            arguments=_safe_arguments(namespace),
            config_path=config_path,
            config=project,
            planned_inputs=[project.config_files.background, *input_references],
            planned_outputs=output_references,
            details=details,
        )
        emit_run_manifest(manifest, manifest_destination)
    except Exception as exc:
        _stderr_best_effort(
            "seismoflux: 阶段2固定交付已确认; 辅助运行清单报告失败"
            f" ({type(exc).__name__}: {exc})。\n"
        )
    return 0


def _run_deferred(namespace: argparse.Namespace) -> int:
    command = str(namespace.command)
    spec = COMMAND_SPECS[command]
    config_path = Path(namespace.config)
    config = load_config(config_path)
    manifest_destination = _validated_manifest_destination(
        namespace.manifest,
        config_path=config_path,
        protected_paths=[
            resolve_project_path(config_path, config.config_files.data_sources),
            resolve_project_path(config_path, config.config_files.research_protocol),
            resolve_project_path(config_path, config.config_files.operating_points),
        ],
    )
    dry_run = bool(namespace.dry_run)
    manifest = build_run_manifest(
        command=command,
        dry_run=dry_run,
        implementation_stage=spec.stage,
        implementation_status=f"deferred_to_stage_{spec.stage}",
        status="planned" if dry_run else "blocked",
        arguments=_safe_arguments(namespace),
        config_path=config_path,
        config=config,
        planned_inputs=list(spec.planned_inputs),
        planned_outputs=list(spec.planned_outputs),
        details={"reason": f"implementation is gated until stage {spec.stage}"},
    )
    emit_run_manifest(manifest, manifest_destination)
    if dry_run:
        return 0
    sys.stderr.write(
        f"seismoflux: '{command}' is deferred to stage {spec.stage}; "
        "no placeholder output was made.\n"
    )
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""

    parser = build_parser()
    namespace = parser.parse_args(argv)
    if namespace.command is None:
        parser.print_help()
        return 0
    try:
        if namespace.command == "inventory":
            return _run_inventory(namespace)
        if namespace.command in {"ingest", "validate-data"}:
            return _run_stage1(namespace)
        if namespace.command == "build-background":
            return _run_background(namespace)
        return _run_deferred(namespace)
    except (OSError, RuntimeError, ValueError) as exc:
        sys.stderr.write(f"seismoflux: {exc}\n")
        return 2
