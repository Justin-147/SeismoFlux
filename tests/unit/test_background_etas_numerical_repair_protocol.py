from __future__ import annotations

import ast
import base64
import hashlib
import json
import os
import re
import subprocess
import types
import unicodedata
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import distribution
from itertools import count
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import numpy as np
import pytest
import shapely
import shapely._geometry as shapely_geometry
import shapely.creation as shapely_creation
import shapely.geometry.base as shapely_base
import shapely.lib as shapely_lib
import shapely.predicates as shapely_predicates
import yaml
from shapely.geometry import Point
from shapely.geometry import point as shapely_point

from seismoflux.background.artifacts import canonical_json_bytes
from seismoflux.background.etas_fit import ETASParameterBounds, optimizer_start
from seismoflux.background.grid import (
    GridConvergenceDiagnostics,
    GridSpec,
    ThreeGridConvergenceGateEvidence,
    cell_id,
)
from seismoflux.background.pipeline_etas import (
    ETASGridGateEvidence,
    ETASGridResolutionEvidence,
)
from seismoflux.background.randomness import SeedContext

PROTOCOL_PATH = Path("configs/background_etas_numerical_repair.yaml")
START_MANIFEST_PATH = Path("data/manifests/etas_numerical_repair_start_manifest.json")
PROTOCOL_DOCUMENT_PATH = Path("docs/background_etas_numerical_repair_protocol.md")
PROTOCOL_ACCEPTANCE_PATH = Path("docs/phase2_etas_numerical_repair_protocol_acceptance.md")
RESTART_HANDOFF_PATH = Path("docs/restart_handoff_2026-07-19_stage2_etas_repair_protocol.md")
STAGE4_R2_PROTOCOL_PATH = Path("configs/anomaly_increment_r2.yaml")
R1_PROTOCOL_COMMIT = "da916454c908e0cbe4a7526f56a8f837331a3c7c"
R2_PROTOCOL_TAG = "v0.2.2-background-etas-repair-protocol-r2"
R2_PROTOCOL_TAG_OBJECT = "903c80ed64295311f8d7870b4847f56d67caee51"
R2_PROTOCOL_COMMIT = "5a5902a83645c217ea11a3bd99eb70b535f0e4df"

SNAPSHOT_ORDER = ("fold_1", "fold_2", "fold_3", "fold_4", "final_validation")
SYNTHETIC_WINDOWS_VOLUME_GUID_PREFIX = r"\\?\Volume{00000000-0000-0000-0000-000000000001}"
WINDOWS_VOLUME_GUID_PATH_PATTERN = re.compile(
    r"^\\\\\?\\Volume\{[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
    r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}\}\\(.+)$"
)


@dataclass(frozen=True, slots=True)
class _SyntheticLiveDirectoryIdentityCapture:
    source_kind: str
    capture_id: str
    opened_handle_count: int
    records: tuple[dict[str, object], ...]
    _provenance_capability: object


_LIVE_CAPTURE_FACTORY_TOKEN = object()
_LIVE_CAPTURE_ID_COUNTER = count()
_REGISTERED_LIVE_CAPTURE_RECEIPTS: dict[
    int, tuple[_SyntheticLiveDirectoryIdentityCapture, str, str, int, bytes]
] = {}
_CONSUMED_LIVE_CAPTURE_IDS: set[str] = set()
_UNBOUND_LIVE_HANDLE_SOURCES: dict[int, tuple[object, tuple[object, ...]]] = {}
_REGISTERED_LIVE_HANDLE_SOURCES: dict[int, tuple[object, object, tuple[object, ...]]] = {}
_CONSUMED_LIVE_HANDLE_SOURCES: dict[int, object] = {}
_REGISTERED_LIVE_CAPTURE_PROVIDERS: dict[int, tuple[object, object]] = {}
_CONSUMED_LIVE_CAPTURE_PROVIDERS: dict[int, object] = {}


class _SyntheticLiveDirectoryHandleSource:
    __slots__ = (
        "_attempt_id",
        "_identity_override",
        "_opened_handle_limit_for_test",
        "_path",
        "_platform",
        "_posix_parent_st_ino",
        "_posix_st_dev",
        "_windows_parent_file_id",
        "_windows_volume_serial",
    )

    def __init__(
        self,
        *,
        factory_token: object,
        attempt_id: str,
        platform: str | None,
        path: Path | None,
        windows_volume_serial: str | None,
        windows_parent_file_id: str | None,
        posix_st_dev: str | None,
        posix_parent_st_ino: str | None,
        identity_override: tuple[int, str, object] | None,
        opened_handle_limit_for_test: int | None,
    ) -> None:
        if factory_token is not _LIVE_CAPTURE_FACTORY_TOKEN:
            raise ValueError("live handle source requires the controlled factory")
        if type(self) is not _SyntheticLiveDirectoryHandleSource:
            raise ValueError("live handle source requires the exact controlled factory class")
        if (path is None) == (platform is None):
            raise ValueError("live handle source requires exactly one source mode")
        if platform is not None and platform not in {"posix", "windows"}:
            raise ValueError("live handle source platform is invalid")
        if opened_handle_limit_for_test is not None and opened_handle_limit_for_test < 0:
            raise ValueError("live handle source test limit is invalid")
        self._attempt_id = attempt_id
        self._platform = platform
        self._path = path
        self._windows_volume_serial = windows_volume_serial
        self._windows_parent_file_id = windows_parent_file_id
        self._posix_st_dev = posix_st_dev
        self._posix_parent_st_ino = posix_parent_st_ino
        self._identity_override = identity_override
        self._opened_handle_limit_for_test = opened_handle_limit_for_test
        _UNBOUND_LIVE_HANDLE_SOURCES[id(self)] = (
            self,
            _synthetic_live_handle_source_configuration_snapshot(self),
        )

    def observe_each_live_directory_handle(
        self,
        *,
        requesting_provider: object,
    ) -> tuple[dict[str, object], ...]:
        registered_source = _REGISTERED_LIVE_HANDLE_SOURCES.get(id(self))
        if (
            registered_source is None
            or registered_source[0] is not self
            or registered_source[1] is not requesting_provider
        ):
            if _CONSUMED_LIVE_HANDLE_SOURCES.get(id(self)) is self:
                raise ValueError("live handle source is single-use")
            raise ValueError("live handle source is not bound to the requesting provider")
        del _REGISTERED_LIVE_HANDLE_SOURCES[id(self)]
        _CONSUMED_LIVE_HANDLE_SOURCES[id(self)] = self
        if _synthetic_live_handle_source_configuration_snapshot(self) != registered_source[2]:
            raise ValueError("live handle source configuration changed after binding")
        if self._path is not None:
            parent_stat = self._path.parent.stat()
            platform = "windows" if os.name == "nt" else "posix"
            parent_file_id = (
                _synthetic_file_id_128_hex(role="parent", identity_value=parent_stat.st_ino)
                if platform == "windows"
                else str(parent_stat.st_ino)
            )
            windows_volume_serial = str(parent_stat.st_dev) if platform == "windows" else None
            windows_parent_file_id = parent_file_id if platform == "windows" else None
            posix_st_dev = str(parent_stat.st_dev) if platform == "posix" else None
            posix_parent_st_ino = parent_file_id if platform == "posix" else None
        else:
            if self._platform is None:
                raise AssertionError("raw live handle source platform disappeared")
            platform = self._platform
            windows_volume_serial = self._windows_volume_serial
            windows_parent_file_id = self._windows_parent_file_id
            posix_st_dev = self._posix_st_dev
            posix_parent_st_ino = self._posix_parent_st_ino
        raw_observations = _independent_synthetic_live_handle_records(
            attempt_id=self._attempt_id,
            platform=platform,
            windows_volume_serial=windows_volume_serial,
            windows_parent_file_id=windows_parent_file_id,
            posix_st_dev=posix_st_dev,
            posix_parent_st_ino=posix_parent_st_ino,
            identity_override=self._identity_override,
        )
        observations: list[dict[str, object]] = []
        for record in raw_observations:
            if (
                self._opened_handle_limit_for_test is not None
                and len(observations) >= self._opened_handle_limit_for_test
            ):
                break
            observations.append(deepcopy(record))
        return tuple(observations)


def _synthetic_live_handle_source_configuration_snapshot(
    source: _SyntheticLiveDirectoryHandleSource,
) -> tuple[object, ...]:
    path = source._path
    identity_override = source._identity_override
    return (
        (type(source._attempt_id), source._attempt_id),
        (type(source._platform), source._platform),
        (type(path), id(path), path),
        (type(source._windows_volume_serial), source._windows_volume_serial),
        (type(source._windows_parent_file_id), source._windows_parent_file_id),
        (type(source._posix_st_dev), source._posix_st_dev),
        (type(source._posix_parent_st_ino), source._posix_parent_st_ino),
        (
            type(identity_override),
            id(identity_override),
            canonical_json_bytes(list(identity_override))
            if identity_override is not None
            else None,
        ),
        (
            type(source._opened_handle_limit_for_test),
            source._opened_handle_limit_for_test,
        ),
    )


class _SyntheticLiveDirectoryCaptureProvider:
    __slots__ = ("_live_handle_source",)

    def __init__(
        self,
        *,
        factory_token: object,
        live_handle_source: _SyntheticLiveDirectoryHandleSource,
    ) -> None:
        if factory_token is not _LIVE_CAPTURE_FACTORY_TOKEN:
            raise ValueError("live capture provider requires the controlled factory")
        if type(self) is not _SyntheticLiveDirectoryCaptureProvider:
            raise ValueError("live capture provider requires the exact controlled factory class")
        if type(live_handle_source) is not _SyntheticLiveDirectoryHandleSource:
            raise ValueError("live capture provider requires a controlled handle source")
        unbound_source = _UNBOUND_LIVE_HANDLE_SOURCES.get(id(live_handle_source))
        if unbound_source is None or unbound_source[0] is not live_handle_source:
            raise ValueError("live capture provider requires a new unbound handle source")
        del _UNBOUND_LIVE_HANDLE_SOURCES[id(live_handle_source)]
        construction_snapshot = unbound_source[1]
        if (
            _synthetic_live_handle_source_configuration_snapshot(live_handle_source)
            != construction_snapshot
        ):
            raise ValueError("live handle source configuration changed before binding")
        self._live_handle_source = live_handle_source
        _REGISTERED_LIVE_CAPTURE_PROVIDERS[id(self)] = (self, live_handle_source)
        _REGISTERED_LIVE_HANDLE_SOURCES[id(live_handle_source)] = (
            live_handle_source,
            self,
            construction_snapshot,
        )

    def capture_before_evidence_read(self) -> _SyntheticLiveDirectoryIdentityCapture:
        registered_provider = _REGISTERED_LIVE_CAPTURE_PROVIDERS.get(id(self))
        if registered_provider is None or registered_provider[0] is not self:
            if _CONSUMED_LIVE_CAPTURE_PROVIDERS.get(id(self)) is self:
                raise ValueError("live capture provider is single-use")
            raise ValueError("live capture provider is not registered")
        del _REGISTERED_LIVE_CAPTURE_PROVIDERS[id(self)]
        _CONSUMED_LIVE_CAPTURE_PROVIDERS[id(self)] = self
        registered_source = registered_provider[1]
        if (
            type(self._live_handle_source) is not _SyntheticLiveDirectoryHandleSource
            or self._live_handle_source is not registered_source
        ):
            raise ValueError("live capture provider source changed after construction")
        observed_records = self._live_handle_source.observe_each_live_directory_handle(
            requesting_provider=self
        )
        capability = object()
        receipt = _SyntheticLiveDirectoryIdentityCapture(
            source_kind="independent_live_directory_handle_capture",
            capture_id=f"synthetic-live-capture-{next(_LIVE_CAPTURE_ID_COUNTER):08d}",
            opened_handle_count=len(observed_records),
            records=observed_records,
            _provenance_capability=capability,
        )
        _REGISTERED_LIVE_CAPTURE_RECEIPTS[id(capability)] = (
            receipt,
            receipt.source_kind,
            receipt.capture_id,
            receipt.opened_handle_count,
            canonical_json_bytes(list(observed_records)),
        )
        return receipt


FIT_ENDS = (
    "2004-12-31T16:00:00Z",
    "2009-12-31T16:00:00Z",
    "2014-12-31T16:00:00Z",
    "2019-12-31T16:00:00Z",
    "2023-06-30T16:00:00Z",
)
SUPPORT_IDS = (
    "local-support-f06e7c7496ea2357",
    "local-support-eaee903b28c55ace",
    "local-support-f86126dbec5bb79b",
    "local-support-788851371baf0e3b",
    "local-support-f6816ab6c6581306",
)


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Reject duplicate YAML mapping keys instead of silently keeping the last value."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = cast(Any, loader).construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = cast(Any, loader).construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)
COMPENSATOR_DOMAIN_IDS = (
    "154062341fe6e2b68625f90b832219617ce5c61b418a3ff31f4df36f50e8fb1f",
    "33a9095704a09f8661c48061f9febec0342a9db671d6384fe7dcbeb3cf3aed55",
    "8e41c306592739c634ca85bf2540dbcfeb2086e64e908a41b4f90db7cd1f94f1",
    "33a9095704a09f8661c48061f9febec0342a9db671d6384fe7dcbeb3cf3aed55",
    "33a9095704a09f8661c48061f9febec0342a9db671d6384fe7dcbeb3cf3aed55",
)
RETAINED_AREA_FRACTIONS = (0.9734474900209907, 1.0, 0.9972058595099415, 1.0, 1.0)
PARENT_ROLES = (
    "include_prevalidated_eligible_unsupported_history",
    "supported_and_external_buffer_history",
    "include_prevalidated_eligible_unsupported_history",
    "supported_and_external_buffer_history",
    "supported_and_external_buffer_history",
)


def _load_yaml(path: Path) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeySafeLoader),
    )


def _load_yaml_at_revision(revision: str, path: Path) -> dict[str, Any]:
    result = subprocess.run(
        ["git", "show", f"{revision}:{path.as_posix()}"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return cast(
        dict[str, Any],
        yaml.load(result.stdout, Loader=_UniqueKeySafeLoader),
    )


def _load_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_payload_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_v1_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _float_from_exact_hex(value: object) -> float:
    if not isinstance(value, str):
        raise ValueError("expected a float.hex string")
    parsed = float.fromhex(value)
    if not np.isfinite(parsed) or parsed.hex() != value:
        raise ValueError("float hex is non-finite or non-canonical")
    return parsed


def _synthetic_resolution_payload(
    resolution: ETASGridResolutionEvidence,
) -> dict[str, object]:
    return {
        "cell_size_km_hex": resolution.cell_size_km.hex(),
        "cell_count": resolution.cell_count,
        "background_total_hex": resolution.background_total.hex(),
        "triggering_total_hex": resolution.triggering_total.hex(),
        "total_hex": resolution.total.hex(),
        "ordered_cell_masses_sha256": resolution.ordered_cell_masses_sha256,
    }


def _synthetic_pair_payload(
    diagnostics: GridConvergenceDiagnostics,
) -> dict[str, object]:
    identity: dict[str, object] = {
        "coarse_grid_size_km_hex": diagnostics.coarse_cell_size_km.hex(),
        "fine_grid_size_km_hex": diagnostics.fine_cell_size_km.hex(),
        "coarse_total_expected_count_hex": diagnostics.coarse_total.hex(),
        "fine_total_expected_count_hex": diagnostics.fine_total.hex(),
        "relative_expected_count_difference_hex": (
            diagnostics.relative_expected_count_difference.hex()
        ),
        "density_l1_difference_hex": diagnostics.density_l1_difference.hex(),
        "passes_frozen_tolerances": diagnostics.passed,
    }
    return {**identity, "diagnostic_payload_sha256": _canonical_v1_sha256(identity)}


def _resolution_from_local_payload(
    payload: dict[str, object],
) -> ETASGridResolutionEvidence:
    cell_count = payload["cell_count"]
    if not isinstance(cell_count, int) or isinstance(cell_count, bool) or cell_count <= 0:
        raise ValueError("cell_count must be a positive strict integer")
    reconstructed = {
        "cell_size_km": _float_from_exact_hex(payload["cell_size_km_hex"]),
        "cell_count": cell_count,
        "background_total": _float_from_exact_hex(payload["background_total_hex"]),
        "triggering_total": _float_from_exact_hex(payload["triggering_total_hex"]),
        "total": _float_from_exact_hex(payload["total_hex"]),
        "ordered_cell_masses_sha256": payload["ordered_cell_masses_sha256"],
    }
    return ETASGridResolutionEvidence(**cast(Any, reconstructed))


def _comparison_from_local_payload(
    payload: dict[str, object],
) -> GridConvergenceDiagnostics:
    reconstructed = {
        "coarse_cell_size_km": _float_from_exact_hex(payload["coarse_grid_size_km_hex"]),
        "fine_cell_size_km": _float_from_exact_hex(payload["fine_grid_size_km_hex"]),
        "coarse_total": _float_from_exact_hex(payload["coarse_total_expected_count_hex"]),
        "fine_total": _float_from_exact_hex(payload["fine_total_expected_count_hex"]),
        "relative_expected_count_difference": _float_from_exact_hex(
            payload["relative_expected_count_difference_hex"]
        ),
        "density_l1_difference": _float_from_exact_hex(payload["density_l1_difference_hex"]),
    }
    diagnostics = GridConvergenceDiagnostics(**cast(Any, reconstructed))
    if payload["passes_frozen_tolerances"] is not diagnostics.passed:
        raise ValueError("pair pass flag does not match frozen tolerances")
    return diagnostics


def _independent_numerical_evidence_id(
    *,
    protocol_sha256: str,
    snapshot_id: str,
    parameter_snapshot_id: str,
    resolutions: tuple[ETASGridResolutionEvidence, ...],
    comparisons: tuple[GridConvergenceDiagnostics, ...],
) -> str:
    preimage = {
        "protocol_sha256": protocol_sha256,
        "snapshot_id": snapshot_id,
        "parameter_snapshot_id": parameter_snapshot_id,
        "resolutions": tuple(asdict(item) for item in resolutions),
        "comparisons": tuple(asdict(item) for item in comparisons),
    }
    return _canonical_v1_sha256(preimage)


def _reconstruct_gate_from_envelope_payload(
    envelope: dict[str, object],
    persistence: dict[str, Any],
) -> tuple[ETASGridGateEvidence, str]:
    sealed = cast(dict[str, Any], envelope["sealed_three_grid_gate_evidence"])
    identity = cast(dict[str, str], sealed["existing_evaluator_return_identity"])
    resolution_map = cast(
        dict[str, dict[str, object]], envelope["grid_resolution_payload_by_grid_size"]
    )
    crosswalk = persistence["numerical_evidence_id_crosswalk_exact"]
    resolution_keys = tuple(
        path.rsplit(".", 1)[1] for path in crosswalk["resolution_source_paths_in_order_exact"]
    )
    pair_fields = tuple(
        path.rsplit(".", 1)[1] for path in crosswalk["comparison_source_paths_in_order_exact"]
    )
    resolutions = tuple(
        _resolution_from_local_payload(resolution_map[key]) for key in resolution_keys
    )
    comparisons = tuple(
        _comparison_from_local_payload(cast(dict[str, object], sealed[field]))
        for field in pair_fields
    )
    if len(comparisons) != 2:
        raise ValueError("three-grid evidence requires exactly two comparison payloads")
    convergence = ThreeGridConvergenceGateEvidence(
        diagnostic_50_to_25=comparisons[0],
        primary_25_to_12_5=comparisons[1],
    )
    gate = ETASGridGateEvidence(
        protocol_sha256=identity["protocol_sha256"],
        snapshot_id=identity["snapshot_id"],
        parameter_snapshot_id=identity["parameter_snapshot_id"],
        resolutions=resolutions,
        convergence=convergence,
    )
    independent = _independent_numerical_evidence_id(
        protocol_sha256=gate.protocol_sha256,
        snapshot_id=gate.snapshot_id,
        parameter_snapshot_id=gate.parameter_snapshot_id,
        resolutions=gate.resolutions,
        comparisons=gate.convergence.comparisons,
    )
    if independent != gate.numerical_evidence_id:
        raise ValueError("independent numerical evidence formula disagrees with pipeline property")
    return gate, independent


def _numerical_evidence_id_from_envelope_payload(
    envelope: dict[str, object],
    persistence: dict[str, Any],
) -> str:
    gate, independent = _reconstruct_gate_from_envelope_payload(envelope, persistence)
    crosswalk = persistence["numerical_evidence_id_crosswalk_exact"]
    assert (
        list(
            {
                "protocol_sha256": gate.protocol_sha256,
                "snapshot_id": gate.snapshot_id,
                "parameter_snapshot_id": gate.parameter_snapshot_id,
                "resolutions": gate.resolutions,
                "comparisons": gate.convergence.comparisons,
            }
        )
        == crosswalk["numerical_evidence_preimage_fields_exact"]
    )
    return independent


def _unchecked_numerical_evidence_id_from_envelope_payload(
    envelope: dict[str, object],
    persistence: dict[str, Any],
) -> str:
    """Rehash an adversarial payload without enforcing wrapper invariants."""

    sealed = cast(dict[str, Any], envelope["sealed_three_grid_gate_evidence"])
    identity = cast(dict[str, str], sealed["existing_evaluator_return_identity"])
    resolution_map = cast(
        dict[str, dict[str, object]], envelope["grid_resolution_payload_by_grid_size"]
    )
    crosswalk = persistence["numerical_evidence_id_crosswalk_exact"]
    resolution_keys = tuple(
        path.rsplit(".", 1)[1] for path in crosswalk["resolution_source_paths_in_order_exact"]
    )
    pair_fields = tuple(
        path.rsplit(".", 1)[1] for path in crosswalk["comparison_source_paths_in_order_exact"]
    )
    return _independent_numerical_evidence_id(
        protocol_sha256=identity["protocol_sha256"],
        snapshot_id=identity["snapshot_id"],
        parameter_snapshot_id=identity["parameter_snapshot_id"],
        resolutions=tuple(
            _resolution_from_local_payload(resolution_map[key]) for key in resolution_keys
        ),
        comparisons=tuple(
            _comparison_from_local_payload(cast(dict[str, object], sealed[field]))
            for field in pair_fields
        ),
    )


def _directory_chain_relative_paths(attempt_id: str) -> list[str]:
    return [
        ".",
        "data",
        "data/processed",
        "data/processed/stage2R",
        "data/processed/stage2R/etas_numerical_repair_fit_input",
        "data/processed/stage2R/etas_numerical_repair_fit_input/attempts",
        f"data/processed/stage2R/etas_numerical_repair_fit_input/attempts/{attempt_id}",
        (
            "data/processed/stage2R/etas_numerical_repair_fit_input/attempts/"
            f"{attempt_id}/local_restricted"
        ),
        (
            "data/processed/stage2R/etas_numerical_repair_fit_input/attempts/"
            f"{attempt_id}/local_restricted/three_grid_gate_evidence"
        ),
    ]


def _synthetic_canonical_directory_path(*, platform: str, relative_path: str) -> str:
    components = ["synthetic_workspace"]
    if relative_path != ".":
        components.extend(relative_path.split("/"))
    separator = "\\" if platform == "windows" else "/"
    prefix = SYNTHETIC_WINDOWS_VOLUME_GUID_PREFIX if platform == "windows" else ""
    return f"{prefix}{separator}{separator.join(components)}"


def _synthetic_directory_identity_chain(
    *,
    attempt_id: str,
    platform: str,
    windows_volume_serial: str | None,
    windows_parent_file_id: str | None,
    posix_st_dev: str | None,
    posix_parent_st_ino: str | None,
) -> list[dict[str, object]]:
    paths = _directory_chain_relative_paths(attempt_id)
    records: list[dict[str, object]] = []
    for index, relative_path in enumerate(paths):
        is_parent = index == len(paths) - 1
        components = [] if relative_path == "." else relative_path.split("/")
        record: dict[str, object]
        if platform == "windows":
            file_id = (
                windows_parent_file_id
                if is_parent
                else _synthetic_file_id_128_hex(
                    role="directory-chain",
                    identity_value=f"{index}:{relative_path}",
                )
            )
            record = {
                "repository_relative_directory_path": relative_path,
                "exact_case_relative_components_from_workspace_root": components,
                "canonical_directory_path": _synthetic_canonical_directory_path(
                    platform=platform,
                    relative_path=relative_path,
                ),
                "platform": platform,
                "windows_volume_serial_u64_decimal_or_null": windows_volume_serial,
                "windows_directory_file_id_16_bytes_hex_or_null": file_id,
                "posix_st_dev_decimal_or_null": None,
                "posix_st_ino_decimal_or_null": None,
            }
        else:
            inode = posix_parent_st_ino if is_parent else str(1000 + index)
            record = {
                "repository_relative_directory_path": relative_path,
                "exact_case_relative_components_from_workspace_root": components,
                "canonical_directory_path": _synthetic_canonical_directory_path(
                    platform=platform,
                    relative_path=relative_path,
                ),
                "platform": platform,
                "windows_volume_serial_u64_decimal_or_null": None,
                "windows_directory_file_id_16_bytes_hex_or_null": None,
                "posix_st_dev_decimal_or_null": posix_st_dev,
                "posix_st_ino_decimal_or_null": inode,
            }
        records.append(record)
    return records


def _independent_synthetic_live_handle_records(
    *,
    attempt_id: str,
    platform: str,
    windows_volume_serial: str | None,
    windows_parent_file_id: str | None,
    posix_st_dev: str | None,
    posix_parent_st_ino: str | None,
    identity_override: tuple[int, str, object] | None = None,
) -> list[dict[str, object]]:
    paths = _directory_chain_relative_paths(attempt_id)
    source_records: list[dict[str, object]] = []
    for index, relative_path in enumerate(paths):
        is_parent = index == len(paths) - 1
        components = [] if relative_path == "." else relative_path.split("/")
        record: dict[str, object] = {
            "repository_relative_directory_path": relative_path,
            "exact_case_relative_components_from_workspace_root": components,
            "canonical_directory_path": _synthetic_canonical_directory_path(
                platform=platform,
                relative_path=relative_path,
            ),
            "platform": platform,
            "windows_volume_serial_u64_decimal_or_null": (
                windows_volume_serial if platform == "windows" else None
            ),
            "windows_directory_file_id_16_bytes_hex_or_null": (
                (
                    windows_parent_file_id
                    if is_parent
                    else _synthetic_file_id_128_hex(
                        role="directory-chain",
                        identity_value=f"{index}:{relative_path}",
                    )
                )
                if platform == "windows"
                else None
            ),
            "posix_st_dev_decimal_or_null": (posix_st_dev if platform == "posix" else None),
            "posix_st_ino_decimal_or_null": (
                (posix_parent_st_ino if is_parent else str(1000 + index))
                if platform == "posix"
                else None
            ),
        }
        source_records.append(record)
    if identity_override is not None:
        record_index, field_name, value = identity_override
        source_records[record_index][field_name] = value
    return source_records


def _synthetic_live_directory_capture_provider(
    *,
    attempt_id: str,
    platform: str,
    windows_volume_serial: str | None,
    windows_parent_file_id: str | None,
    posix_st_dev: str | None,
    posix_parent_st_ino: str | None,
    identity_override: tuple[int, str, object] | None = None,
    opened_handle_limit_for_test: int | None = None,
) -> _SyntheticLiveDirectoryCaptureProvider:
    live_handle_source = _SyntheticLiveDirectoryHandleSource(
        factory_token=_LIVE_CAPTURE_FACTORY_TOKEN,
        attempt_id=attempt_id,
        platform=platform,
        path=None,
        windows_volume_serial=windows_volume_serial,
        windows_parent_file_id=windows_parent_file_id,
        posix_st_dev=posix_st_dev,
        posix_parent_st_ino=posix_parent_st_ino,
        identity_override=identity_override,
        opened_handle_limit_for_test=opened_handle_limit_for_test,
    )
    return _SyntheticLiveDirectoryCaptureProvider(
        factory_token=_LIVE_CAPTURE_FACTORY_TOKEN,
        live_handle_source=live_handle_source,
    )


def _synthetic_live_directory_capture_provider_from_path(
    path: Path,
    *,
    attempt_id: str,
    identity_override: tuple[int, str, object] | None = None,
) -> _SyntheticLiveDirectoryCaptureProvider:
    live_handle_source = _SyntheticLiveDirectoryHandleSource(
        factory_token=_LIVE_CAPTURE_FACTORY_TOKEN,
        attempt_id=attempt_id,
        platform=None,
        path=path,
        windows_volume_serial=None,
        windows_parent_file_id=None,
        posix_st_dev=None,
        posix_parent_st_ino=None,
        identity_override=identity_override,
        opened_handle_limit_for_test=None,
    )
    return _SyntheticLiveDirectoryCaptureProvider(
        factory_token=_LIVE_CAPTURE_FACTORY_TOKEN,
        live_handle_source=live_handle_source,
    )


def _synthetic_three_grid_envelope(
    protocol: dict[str, Any],
    installed_file_identity: dict[str, object] | None = None,
    *,
    attempt_id: str = "synthetic-r3-envelope",
    snapshot_id: str = "fold_1",
    protocol_sha256: str = "a" * 64,
    parameter_snapshot_id: str = "b" * 64,
) -> tuple[ETASGridGateEvidence, dict[str, object], dict[str, object]]:
    resolutions = (
        ETASGridResolutionEvidence(50.0, 10, 2.0, 1.0, 3.0, "1" * 64),
        ETASGridResolutionEvidence(25.0, 40, 2.25, 0.75, 3.0, "2" * 64),
        ETASGridResolutionEvidence(12.5, 160, 2.5, 0.5, 3.0, "3" * 64),
    )
    convergence = ThreeGridConvergenceGateEvidence(
        diagnostic_50_to_25=GridConvergenceDiagnostics(50.0, 25.0, 3.0, 3.0, 0.0, 0.01),
        primary_25_to_12_5=GridConvergenceDiagnostics(25.0, 12.5, 3.0, 3.0, 0.0, 0.02),
    )
    gate = ETASGridGateEvidence(
        protocol_sha256=protocol_sha256,
        snapshot_id=snapshot_id,
        parameter_snapshot_id=parameter_snapshot_id,
        resolutions=resolutions,
        convergence=convergence,
    )
    resolution_keys = ("50_km", "25_km", "12_5_km")
    resolution_payloads = {
        key: _synthetic_resolution_payload(value)
        for key, value in zip(resolution_keys, resolutions, strict=True)
    }
    resolution_hashes = {
        key: _canonical_v1_sha256(value) for key, value in resolution_payloads.items()
    }
    sealed_identity: dict[str, object] = {
        "schema_version": 1,
        "attempt_id": attempt_id,
        "snapshot_id": gate.snapshot_id,
        "scientific_fit_input_sha256": "c" * 64,
        "selected_start_index": 0,
        "selected_terminal_transformed_sha256": "d" * 64,
        "selected_physical_parameters_sha256": gate.parameter_snapshot_id,
        "evaluator_callable_identity_sha256": "e" * 64,
        "existing_evaluator_return_identity": {
            "protocol_sha256": gate.protocol_sha256,
            "snapshot_id": gate.snapshot_id,
            "parameter_snapshot_id": gate.parameter_snapshot_id,
            "numerical_evidence_id": gate.numerical_evidence_id,
        },
        "grid_resolution_payload_sha256_by_grid_size": resolution_hashes,
        "diagnostic_50_to_25": _synthetic_pair_payload(convergence.diagnostic_50_to_25),
        "primary_25_to_12_5": _synthetic_pair_payload(convergence.primary_25_to_12_5),
    }
    sealed = {
        **sealed_identity,
        "three_grid_gate_evidence_sha256": _canonical_v1_sha256(sealed_identity),
    }
    persistence = protocol["qualification"]["three_grid_gate_evidence_protocol"][
        "local_restricted_persistence"
    ]
    if installed_file_identity is None:
        directory_chain = _synthetic_directory_identity_chain(
            attempt_id=attempt_id,
            platform="posix",
            windows_volume_serial=None,
            windows_parent_file_id=None,
            posix_st_dev="1",
            posix_parent_st_ino="9",
        )
        evidence_parent = directory_chain[-1]
        installed_file_identity = {
            "profile_id": "posix_linkat_v1",
            "filesystem_name": "synthetic-posix",
            "parent_identity": {
                "platform": "posix",
                "canonical_final_path": evidence_parent["canonical_directory_path"],
                "exact_case_relative_components": [
                    "local_restricted",
                    "three_grid_gate_evidence",
                ],
                "windows_volume_serial_u64_decimal_or_null": None,
                "windows_directory_file_id_16_bytes_hex_or_null": None,
                "posix_st_dev_decimal_or_null": "1",
                "posix_st_ino_decimal_or_null": "9",
            },
            "ordered_directory_identity_chain": directory_chain,
            "ordered_directory_identity_chain_sha256": _canonical_v1_sha256(directory_chain),
            "final_repository_relative_path": (
                "data/processed/stage2R/etas_numerical_repair_fit_input/attempts/"
                f"{attempt_id}/local_restricted/three_grid_gate_evidence/{snapshot_id}.json"
            ),
            "temp_leaf_utf8_hex": f".{snapshot_id}.envelope.tmp".encode().hex(),
            "final_leaf_utf8_hex": f"{snapshot_id}.json".encode().hex(),
            "canonical_final_path": (
                f"{evidence_parent['canonical_directory_path']}/{snapshot_id}.json"
            ),
            "exact_case_relative_components": [
                "local_restricted",
                "three_grid_gate_evidence",
                f"{snapshot_id}.json",
            ],
            "platform": "posix",
            "windows_volume_serial_u64_decimal_or_null": None,
            "windows_file_id_16_bytes_hex_or_null": None,
            "windows_number_of_links_at_temp_capture_or_null": None,
            "posix_st_dev_decimal_or_null": "1",
            "posix_st_ino_decimal_or_null": "2",
            "posix_st_nlink_at_temp_capture_or_null": 1,
            "required_final_nlink": 1,
        }
    envelope_identity: dict[str, object] = {
        "envelope_schema_version": persistence["envelope_schema_version_exact"],
        "installed_file_identity": installed_file_identity,
        "sealed_three_grid_gate_evidence": sealed,
        "grid_resolution_payload_by_grid_size": resolution_payloads,
        "numerical_evidence_id_crosswalk": deepcopy(
            persistence["numerical_evidence_id_crosswalk_exact"]
        ),
    }
    envelope = {
        **envelope_identity,
        "three_grid_gate_evidence_envelope_sha256": _canonical_v1_sha256(envelope_identity),
    }
    public_crosswalk = _synthetic_public_crosswalk_from_envelope(envelope)
    return gate, envelope, public_crosswalk


def _convert_installed_identity_to_valid_synthetic_windows(
    installed_identity: dict[str, object],
    *,
    attempt_id: str,
) -> None:
    windows_volume_serial = "1"
    windows_file_id = _file_id_128_identifier_raw_bytes_hex(bytes(range(16)))
    windows_parent_file_id = _file_id_128_identifier_raw_bytes_hex(bytes(range(16, 32)))
    directory_chain = _synthetic_directory_identity_chain(
        attempt_id=attempt_id,
        platform="windows",
        windows_volume_serial=windows_volume_serial,
        windows_parent_file_id=windows_parent_file_id,
        posix_st_dev=None,
        posix_parent_st_ino=None,
    )
    evidence_parent = directory_chain[-1]
    parent_identity = cast(dict[str, object], installed_identity["parent_identity"])
    parent_identity.update(
        {
            "platform": "windows",
            "canonical_final_path": evidence_parent["canonical_directory_path"],
            "windows_volume_serial_u64_decimal_or_null": windows_volume_serial,
            "windows_directory_file_id_16_bytes_hex_or_null": windows_parent_file_id,
            "posix_st_dev_decimal_or_null": None,
            "posix_st_ino_decimal_or_null": None,
        }
    )
    final_leaf = _decode_canonical_utf8_hex_component(
        installed_identity["final_leaf_utf8_hex"],
        field_name="final leaf",
    )
    installed_identity.update(
        {
            "profile_id": "windows_ntfs_ntcreatefile_filerenameinfo_v1",
            "filesystem_name": "NTFS",
            "ordered_directory_identity_chain": directory_chain,
            "ordered_directory_identity_chain_sha256": _canonical_v1_sha256(directory_chain),
            "canonical_final_path": (
                f"{evidence_parent['canonical_directory_path']}\\{final_leaf}"
            ),
            "platform": "windows",
            "windows_volume_serial_u64_decimal_or_null": windows_volume_serial,
            "windows_file_id_16_bytes_hex_or_null": windows_file_id,
            "windows_number_of_links_at_temp_capture_or_null": 1,
            "posix_st_dev_decimal_or_null": None,
            "posix_st_ino_decimal_or_null": None,
            "posix_st_nlink_at_temp_capture_or_null": None,
        }
    )


def _synthetic_public_crosswalk_from_envelope(
    envelope: dict[str, object],
) -> dict[str, object]:
    sealed = cast(dict[str, Any], envelope["sealed_three_grid_gate_evidence"])
    envelope_sha = envelope["three_grid_gate_evidence_envelope_sha256"]
    return {
        "snapshot_gate_three_grid_gate_evidence_sha256_or_null": envelope_sha,
        "fit_attempt_three_grid_gate_evidence_sha256_or_null": envelope_sha,
        "staged_local_presence_map_value": envelope_sha,
        "snapshot_gate_grid_50_to_25_diagnostic_payload_sha256_or_null": cast(
            dict[str, object], sealed["diagnostic_50_to_25"]
        )["diagnostic_payload_sha256"],
        "snapshot_gate_grid_25_to_12_5_expected_count_relative_difference_hex_or_null": cast(
            dict[str, object], sealed["primary_25_to_12_5"]
        )["relative_expected_count_difference_hex"],
        "snapshot_gate_grid_25_to_12_5_density_l1_hex_or_null": cast(
            dict[str, object], sealed["primary_25_to_12_5"]
        )["density_l1_difference_hex"],
    }


def _cascade_rehash_synthetic_three_grid_envelope(
    protocol: dict[str, Any],
    envelope: dict[str, object],
) -> dict[str, object]:
    persistence = protocol["qualification"]["three_grid_gate_evidence_protocol"][
        "local_restricted_persistence"
    ]
    sealed = cast(dict[str, Any], envelope["sealed_three_grid_gate_evidence"])
    resolution_payloads = cast(
        dict[str, dict[str, object]], envelope["grid_resolution_payload_by_grid_size"]
    )
    sealed["grid_resolution_payload_sha256_by_grid_size"] = {
        key: _canonical_v1_sha256(value) for key, value in resolution_payloads.items()
    }
    for pair_field in ("diagnostic_50_to_25", "primary_25_to_12_5"):
        pair = cast(dict[str, object], sealed[pair_field])
        pair["diagnostic_payload_sha256"] = _canonical_v1_sha256(
            {key: value for key, value in pair.items() if key != "diagnostic_payload_sha256"}
        )
    returned_identity = cast(dict[str, str], sealed["existing_evaluator_return_identity"])
    returned_identity["numerical_evidence_id"] = (
        _unchecked_numerical_evidence_id_from_envelope_payload(envelope, persistence)
    )
    sealed["three_grid_gate_evidence_sha256"] = _canonical_v1_sha256(
        {key: value for key, value in sealed.items() if key != "three_grid_gate_evidence_sha256"}
    )
    envelope["three_grid_gate_evidence_envelope_sha256"] = _canonical_v1_sha256(
        {
            key: value
            for key, value in envelope.items()
            if key != "three_grid_gate_evidence_envelope_sha256"
        }
    )
    return _synthetic_public_crosswalk_from_envelope(envelope)


def _is_canonical_unsigned_decimal(
    value: object,
    *,
    maximum: int | None = None,
) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = int(value, 10)
    except ValueError:
        return False
    return parsed >= 0 and (maximum is None or parsed <= maximum) and str(parsed) == value


def _is_lowercase_hex(value: object, *, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _file_id_128_identifier_raw_bytes_hex(identifier: bytes) -> str:
    if not isinstance(identifier, bytes) or len(identifier) != 16:
        raise ValueError("FILE_ID_128 Identifier must be exactly 16 raw bytes")
    return identifier.hex()


def _synthetic_file_id_128_hex(*, role: str, identity_value: object) -> str:
    return hashlib.sha256(f"{role}:{identity_value}".encode()).digest()[:16].hex()


def _decode_canonical_utf8_hex_component(value: object, *, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) % 2 != 0
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be canonical lowercase UTF-8 hex")
    try:
        decoded = bytes.fromhex(value).decode("utf-8")
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"{field_name} must be canonical lowercase UTF-8 hex") from exc
    if decoded.encode("utf-8").hex() != value:
        raise ValueError(f"{field_name} UTF-8 hex does not round-trip")
    if decoded != unicodedata.normalize("NFC", decoded):
        raise ValueError(f"{field_name} must decode to NFC")
    if (
        not decoded
        or decoded in {".", ".."}
        or "/" in decoded
        or "\\" in decoded
        or "\0" in decoded
    ):
        raise ValueError(f"{field_name} must decode to one safe component")
    return decoded


def _validate_exact_nfc_components(
    value: object,
    *,
    expected_length: int,
    field_name: str,
) -> list[str]:
    if not isinstance(value, list) or len(value) != expected_length:
        raise ValueError(f"{field_name} must be an exact {expected_length}-item list")
    if any(not isinstance(component, str) for component in value):
        raise ValueError(f"{field_name} components must be strings")
    components = cast(list[str], value)
    if any(
        not component
        or component in {".", ".."}
        or component != unicodedata.normalize("NFC", component)
        or "/" in component
        or "\\" in component
        or "\0" in component
        for component in components
    ):
        raise ValueError(f"{field_name} components must be safe UTF-8 NFC names")
    return components


def _validate_platform_canonical_path(
    value: object,
    *,
    platform: object,
    field_name: str,
) -> str:
    if not isinstance(value, str) or not value or value != unicodedata.normalize("NFC", value):
        raise ValueError(f"{field_name} must be a non-empty NFC string")
    if platform == "windows":
        match = WINDOWS_VOLUME_GUID_PATH_PATTERN.fullmatch(value)
        if match is None or "/" in value or "\0" in value:
            raise ValueError(f"{field_name} must be a canonical absolute Windows volume path")
        components = match.group(1).split("\\")
        if any(
            not component
            or component in {".", ".."}
            or component != unicodedata.normalize("NFC", component)
            for component in components
        ):
            raise ValueError(f"{field_name} must contain safe exact-case Windows components")
    elif platform == "posix":
        if not value.startswith("/") or "\\" in value or "//" in value or "\0" in value:
            raise ValueError(f"{field_name} must be a canonical absolute POSIX path")
        if any(component in {".", ".."} for component in value.split("/")[1:]):
            raise ValueError(f"{field_name} must contain safe POSIX components")
    else:
        raise ValueError("wrong installed identity platform")
    return value


def _verify_directory_identity_chain(
    persistence: dict[str, Any],
    installed_identity: dict[str, object],
    *,
    attempt_id: str,
    platform: str,
    parent_identity: dict[str, object],
) -> None:
    chain_value = installed_identity["ordered_directory_identity_chain"]
    if not isinstance(chain_value, list):
        raise ValueError("directory identity chain must be a list")
    chain = cast(list[dict[str, object]], chain_value)
    if len(chain) != persistence["directory_identity_chain_length_exact"]:
        raise ValueError("directory identity chain has wrong length")
    expected_paths = [
        value.format(attempt_id=attempt_id)
        for value in persistence[
            "directory_identity_chain_repository_relative_path_templates_exact"
        ]
    ]
    if expected_paths != _directory_chain_relative_paths(attempt_id):
        raise ValueError("protocol directory identity chain templates drifted")
    if [record.get("repository_relative_directory_path") for record in chain] != expected_paths:
        raise ValueError("directory identity chain paths are missing, extra, or out of order")
    expected_fields = set(persistence["directory_identity_chain_record_fields_exact"])
    root_canonical_path: str | None = None
    seen_platform_ids: set[str] = set()
    for relative_path, record in zip(expected_paths, chain, strict=True):
        if not isinstance(record, dict) or set(record) != expected_fields:
            raise ValueError("directory identity chain record has missing or extra fields")
        expected_components = [] if relative_path == "." else relative_path.split("/")
        components = _validate_exact_nfc_components(
            record["exact_case_relative_components_from_workspace_root"],
            expected_length=len(expected_components),
            field_name="directory chain exact-case components",
        )
        if components != expected_components:
            raise ValueError("directory identity chain components disagree with path")
        if record["platform"] != platform:
            raise ValueError("directory identity chain platform mismatch")
        canonical_path = _validate_platform_canonical_path(
            record["canonical_directory_path"],
            platform=platform,
            field_name="directory chain canonical path",
        )
        if root_canonical_path is None:
            root_canonical_path = canonical_path
        else:
            separator = "\\" if platform == "windows" else "/"
            expected_canonical_path = (
                f"{root_canonical_path.rstrip(separator)}{separator}"
                f"{separator.join(expected_components)}"
            )
            if canonical_path != expected_canonical_path:
                raise ValueError("directory identity chain canonical path mismatch")
        windows_values = (
            record["windows_volume_serial_u64_decimal_or_null"],
            record["windows_directory_file_id_16_bytes_hex_or_null"],
        )
        posix_values = (
            record["posix_st_dev_decimal_or_null"],
            record["posix_st_ino_decimal_or_null"],
        )
        if platform == "windows":
            if any(value is None for value in windows_values) or any(
                value is not None for value in posix_values
            ):
                raise ValueError("invalid Windows directory chain null branch")
            if not _is_canonical_unsigned_decimal(
                windows_values[0], maximum=(1 << 64) - 1
            ) or not _is_lowercase_hex(windows_values[1], length=32):
                raise ValueError("invalid Windows directory chain identity encoding")
            if windows_values[0] != installed_identity["windows_volume_serial_u64_decimal_or_null"]:
                raise ValueError("directory chain and file Windows volume serial disagree")
            platform_id = cast(str, windows_values[1])
        else:
            if any(value is None for value in posix_values) or any(
                value is not None for value in windows_values
            ):
                raise ValueError("invalid POSIX directory chain null branch")
            if not all(_is_canonical_unsigned_decimal(value) for value in posix_values):
                raise ValueError("invalid POSIX directory chain identity encoding")
            if posix_values[0] != installed_identity["posix_st_dev_decimal_or_null"]:
                raise ValueError("directory chain and file POSIX device disagree")
            platform_id = cast(str, posix_values[1])
        if platform_id in seen_platform_ids:
            raise ValueError("directory identity chain reuses a directory identity")
        seen_platform_ids.add(platform_id)
    if _canonical_v1_sha256(chain) != installed_identity["ordered_directory_identity_chain_sha256"]:
        raise ValueError("directory identity chain hash mismatch")
    final_record = chain[-1]
    if (
        final_record["canonical_directory_path"] != parent_identity["canonical_final_path"]
        or cast(list[str], final_record["exact_case_relative_components_from_workspace_root"])[-2:]
        != parent_identity["exact_case_relative_components"]
    ):
        raise ValueError("parent identity does not project from final directory chain record")
    for key in (
        "windows_volume_serial_u64_decimal_or_null",
        "windows_directory_file_id_16_bytes_hex_or_null",
        "posix_st_dev_decimal_or_null",
        "posix_st_ino_decimal_or_null",
    ):
        if final_record[key] != parent_identity[key]:
            raise ValueError("parent platform identity disagrees with directory chain")


def _verify_independently_observed_live_directory_identity_chain(
    installed_identity: dict[str, object],
    observed_live_directory_identity_capture: object,
) -> None:
    expected_value = installed_identity["ordered_directory_identity_chain"]
    if not isinstance(expected_value, list):
        raise ValueError("embedded expected directory identity chain must be a list")
    expected_chain = cast(list[dict[str, object]], expected_value)
    if type(observed_live_directory_identity_capture) is not (
        _SyntheticLiveDirectoryIdentityCapture
    ):
        raise ValueError("observed live directory identity requires an independent capture receipt")
    capability_id = id(observed_live_directory_identity_capture._provenance_capability)
    registered_entry = _REGISTERED_LIVE_CAPTURE_RECEIPTS.get(capability_id)
    if (
        registered_entry is None
        or registered_entry[0] is not observed_live_directory_identity_capture
    ):
        if observed_live_directory_identity_capture.capture_id in _CONSUMED_LIVE_CAPTURE_IDS:
            raise ValueError("observed live directory identity capture ID was already consumed")
        raise ValueError(
            "observed live directory identity capture receipt is not the registered original"
        )
    issuance_source_kind = registered_entry[1]
    issuance_capture_id = registered_entry[2]
    issuance_opened_handle_count = registered_entry[3]
    issuance_record_bytes = registered_entry[4]
    if issuance_capture_id in _CONSUMED_LIVE_CAPTURE_IDS:
        raise ValueError("observed live directory identity capture ID was already consumed")
    del _REGISTERED_LIVE_CAPTURE_RECEIPTS[capability_id]
    _CONSUMED_LIVE_CAPTURE_IDS.add(issuance_capture_id)
    if (
        observed_live_directory_identity_capture.source_kind != issuance_source_kind
        or observed_live_directory_identity_capture.capture_id != issuance_capture_id
        or observed_live_directory_identity_capture.opened_handle_count
        != issuance_opened_handle_count
    ):
        raise ValueError("observed live directory identity capture metadata changed after issuance")
    if (
        issuance_source_kind != "independent_live_directory_handle_capture"
        or not issuance_capture_id
    ):
        raise ValueError("observed live directory identity capture source is invalid")
    if issuance_opened_handle_count != len(expected_chain):
        raise ValueError("observed live directory identity capture did not open all handles")
    observed_chain = list(observed_live_directory_identity_capture.records)
    if canonical_json_bytes(observed_chain) != issuance_record_bytes:
        raise ValueError("observed live directory identity capture changed after issuance")
    if observed_chain is expected_chain or any(
        observed is expected
        for observed, expected in zip(observed_chain, expected_chain, strict=False)
    ):
        raise ValueError("observed live directory identity chain must be independent")
    if len(observed_chain) != len(expected_chain):
        raise ValueError("observed live directory identity chain length drifted")
    if canonical_json_bytes(observed_chain) != canonical_json_bytes(expected_chain):
        raise ValueError("observed live directory identity chain drifted")
    if (
        _canonical_v1_sha256(observed_chain)
        != installed_identity["ordered_directory_identity_chain_sha256"]
    ):
        raise ValueError("observed live directory identity chain hash drifted")


def _verify_synthetic_three_grid_envelope(
    protocol: dict[str, Any],
    envelope_bytes: bytes,
    public_crosswalk: dict[str, object],
    *,
    expected_attempt_id: str | None = None,
    expected_snapshot_id: str | None = None,
    expected_final_repository_relative_path: str | None = None,
    expected_protocol_sha256: str | None = None,
    expected_installed_file_identity: dict[str, object] | None = None,
) -> dict[str, str]:
    grid_protocol = protocol["qualification"]["three_grid_gate_evidence_protocol"]
    persistence = grid_protocol["local_restricted_persistence"]
    parsed = cast(dict[str, object], json.loads(envelope_bytes))
    if canonical_json_bytes(parsed) != envelope_bytes:
        raise ValueError("envelope bytes are not canonical JSON v1")
    if set(parsed) != set(persistence["envelope_fields_exact"]):
        raise ValueError("missing or extra envelope field")
    if parsed["envelope_schema_version"] != persistence["envelope_schema_version_exact"]:
        raise ValueError("wrong envelope schema version")
    installed_identity = cast(dict[str, object], parsed["installed_file_identity"])
    if set(installed_identity) != set(persistence["installed_file_identity_fields_exact"]):
        raise ValueError("missing or extra installed identity field")
    if (
        expected_installed_file_identity is not None
        and installed_identity != expected_installed_file_identity
    ):
        raise ValueError("embedded installed identity differs from the captured external identity")
    if installed_identity["platform"] not in {"windows", "posix"}:
        raise ValueError("wrong installed identity platform")
    parent_identity = cast(dict[str, object], installed_identity["parent_identity"])
    if set(parent_identity) != set(persistence["installed_file_parent_identity_fields_exact"]):
        raise ValueError("missing or extra parent identity field")
    if parent_identity["platform"] != installed_identity["platform"]:
        raise ValueError("parent and file platform disagree")
    required_nlink = installed_identity["required_final_nlink"]
    if required_nlink != 1 or isinstance(required_nlink, bool):
        raise ValueError("required final link count must be exactly one")
    platform = installed_identity["platform"]
    windows_fields = (
        installed_identity["windows_volume_serial_u64_decimal_or_null"],
        installed_identity["windows_file_id_16_bytes_hex_or_null"],
        installed_identity["windows_number_of_links_at_temp_capture_or_null"],
    )
    posix_fields = (
        installed_identity["posix_st_dev_decimal_or_null"],
        installed_identity["posix_st_ino_decimal_or_null"],
        installed_identity["posix_st_nlink_at_temp_capture_or_null"],
    )
    expected_profile = (
        "windows_ntfs_ntcreatefile_filerenameinfo_v1"
        if platform == "windows"
        else "posix_linkat_v1"
    )
    if installed_identity["profile_id"] != expected_profile:
        raise ValueError("installed identity profile mismatch")
    if platform == "windows" and installed_identity["filesystem_name"] != "NTFS":
        raise ValueError("Windows profile requires NTFS")
    if (
        not isinstance(installed_identity["filesystem_name"], str)
        or not installed_identity["filesystem_name"]
    ):
        raise ValueError("filesystem name is required")
    if platform == "windows":
        if any(value is None for value in windows_fields) or any(
            value is not None for value in posix_fields
        ):
            raise ValueError("invalid Windows identity null branch")
        if windows_fields[2] != 1 or isinstance(windows_fields[2], bool):
            raise ValueError("Windows temp link count must be one")
        if not _is_canonical_unsigned_decimal(
            windows_fields[0],
            maximum=(1 << 64) - 1,
        ) or not _is_lowercase_hex(windows_fields[1], length=32):
            raise ValueError("invalid Windows installed file identity encoding")
    else:
        if any(value is None for value in posix_fields) or any(
            value is not None for value in windows_fields
        ):
            raise ValueError("invalid POSIX identity null branch")
        if posix_fields[2] != 1 or isinstance(posix_fields[2], bool):
            raise ValueError("POSIX temp link count must be one")
        if not all(_is_canonical_unsigned_decimal(value) for value in posix_fields[:2]):
            raise ValueError("invalid POSIX installed file identity encoding")

    parent_windows_fields = (
        parent_identity["windows_volume_serial_u64_decimal_or_null"],
        parent_identity["windows_directory_file_id_16_bytes_hex_or_null"],
    )
    parent_posix_fields = (
        parent_identity["posix_st_dev_decimal_or_null"],
        parent_identity["posix_st_ino_decimal_or_null"],
    )
    if platform == "windows":
        if any(value is None for value in parent_windows_fields) or any(
            value is not None for value in parent_posix_fields
        ):
            raise ValueError("invalid Windows parent identity null branch")
        if not _is_canonical_unsigned_decimal(
            parent_windows_fields[0],
            maximum=(1 << 64) - 1,
        ) or not _is_lowercase_hex(parent_windows_fields[1], length=32):
            raise ValueError("invalid Windows parent identity encoding")
    else:
        if any(value is None for value in parent_posix_fields) or any(
            value is not None for value in parent_windows_fields
        ):
            raise ValueError("invalid POSIX parent identity null branch")
        if not all(_is_canonical_unsigned_decimal(value) for value in parent_posix_fields):
            raise ValueError("invalid POSIX parent identity encoding")
    parent_components = _validate_exact_nfc_components(
        parent_identity["exact_case_relative_components"],
        expected_length=2,
        field_name="parent exact-case components",
    )
    components = _validate_exact_nfc_components(
        installed_identity["exact_case_relative_components"],
        expected_length=3,
        field_name="file exact-case components",
    )
    if parent_components != components[:-1]:
        raise ValueError("parent and file exact-case components disagree")
    if platform == "windows":
        if parent_windows_fields[0] != windows_fields[0]:
            raise ValueError("parent and file Windows volume serial disagree")
    elif parent_posix_fields[0] != posix_fields[0]:
        raise ValueError("parent and file POSIX device identity disagree")
    _decode_canonical_utf8_hex_component(
        installed_identity["temp_leaf_utf8_hex"],
        field_name="temp leaf",
    )
    final_leaf = _decode_canonical_utf8_hex_component(
        installed_identity["final_leaf_utf8_hex"],
        field_name="final leaf",
    )
    if final_leaf != components[-1]:
        raise ValueError("final leaf and exact-case components disagree")
    parent_canonical_path = _validate_platform_canonical_path(
        parent_identity["canonical_final_path"],
        platform=platform,
        field_name="parent canonical path",
    )
    file_canonical_path = _validate_platform_canonical_path(
        installed_identity["canonical_final_path"],
        platform=platform,
        field_name="file canonical path",
    )
    path_separator = "\\" if platform == "windows" else "/"
    if (
        file_canonical_path
        != f"{parent_canonical_path.rstrip(path_separator)}{path_separator}{final_leaf}"
    ):
        raise ValueError("parent and file canonical paths disagree")
    final_repository_relative_path = installed_identity["final_repository_relative_path"]
    if not isinstance(final_repository_relative_path, str):
        raise ValueError("final repository-relative path must be a string")
    repository_components = final_repository_relative_path.split("/")
    if (
        final_repository_relative_path.startswith("/")
        or "\\" in final_repository_relative_path
        or any(not component or component in {".", ".."} for component in repository_components)
        or repository_components[-len(components) :] != components
    ):
        raise ValueError("final repository-relative path mismatch")
    if (
        expected_final_repository_relative_path is not None
        and installed_identity["final_repository_relative_path"]
        != expected_final_repository_relative_path
    ):
        raise ValueError("installed path differs from the externally bound final path")
    if (
        parsed["numerical_evidence_id_crosswalk"]
        != persistence["numerical_evidence_id_crosswalk_exact"]
    ):
        raise ValueError("numerical evidence crosswalk mismatch")

    sealed = cast(dict[str, Any], parsed["sealed_three_grid_gate_evidence"])
    if set(sealed) != set(grid_protocol["evidence_fields_exact"]):
        raise ValueError("missing or extra sealed evidence field")
    returned_identity = cast(dict[str, str], sealed["existing_evaluator_return_identity"])
    if set(returned_identity) != set(
        grid_protocol["existing_evaluator_return_identity_fields_exact"]
    ):
        raise ValueError("missing or extra evaluator return identity field")
    if sealed["snapshot_id"] != returned_identity["snapshot_id"]:
        raise ValueError("sealed snapshot and evaluator return identity disagree")
    if sealed["selected_physical_parameters_sha256"] != returned_identity["parameter_snapshot_id"]:
        raise ValueError("selected physical parameters and evaluator parameter identity disagree")
    attempt_id = sealed["attempt_id"]
    snapshot_id = sealed["snapshot_id"]
    if not isinstance(attempt_id, str) or not attempt_id:
        raise ValueError("sealed attempt ID must be a non-empty string")
    if snapshot_id not in SNAPSHOT_ORDER:
        raise ValueError("sealed snapshot ID is not frozen")
    exact_path = persistence["file_path_template"].format(
        attempt_id=attempt_id,
        snapshot_id=snapshot_id,
    )
    if installed_identity["final_repository_relative_path"] != exact_path:
        raise ValueError("sealed attempt/snapshot do not match the installed file path")
    if components != ["local_restricted", "three_grid_gate_evidence", f"{snapshot_id}.json"]:
        raise ValueError("sealed snapshot does not match exact-case path components")
    _verify_directory_identity_chain(
        persistence,
        installed_identity,
        attempt_id=attempt_id,
        platform=cast(str, platform),
        parent_identity=parent_identity,
    )
    if expected_attempt_id is not None and attempt_id != expected_attempt_id:
        raise ValueError("sealed attempt ID differs from the external attempt binding")
    if expected_snapshot_id is not None and snapshot_id != expected_snapshot_id:
        raise ValueError("sealed snapshot ID differs from the external snapshot binding")
    if (
        expected_protocol_sha256 is not None
        and returned_identity["protocol_sha256"] != expected_protocol_sha256
    ):
        raise ValueError("evaluator protocol SHA differs from the external protocol binding")
    resolutions = cast(dict[str, dict[str, object]], parsed["grid_resolution_payload_by_grid_size"])
    resolution_keys = grid_protocol["grid_resolution_payload_sha256_keys_exact"]
    if set(resolutions) != set(resolution_keys):
        raise ValueError("missing or extra resolution payload")
    resolution_hashes = cast(dict[str, str], sealed["grid_resolution_payload_sha256_by_grid_size"])
    if set(resolution_hashes) != set(resolution_keys):
        raise ValueError("missing or extra sealed resolution hash")
    for key in resolution_keys:
        payload = resolutions[key]
        if set(payload) != set(grid_protocol["grid_resolution_payload_fields_exact"]):
            raise ValueError("missing or extra resolution field")
        _resolution_from_local_payload(payload)
        if _canonical_v1_sha256(payload) != resolution_hashes[key]:
            raise ValueError("resolution payload hash mismatch")

    for pair_field in ("diagnostic_50_to_25", "primary_25_to_12_5"):
        pair = cast(dict[str, object], sealed[pair_field])
        if set(pair) != set(grid_protocol["grid_pair_diagnostic_fields_exact"]):
            raise ValueError("missing or extra pair diagnostic field")
        pair_identity = {
            key: value for key, value in pair.items() if key != "diagnostic_payload_sha256"
        }
        if _canonical_v1_sha256(pair_identity) != pair["diagnostic_payload_sha256"]:
            raise ValueError("pair diagnostic hash mismatch")
        _comparison_from_local_payload(pair)

    sealed_identity = {
        key: value for key, value in sealed.items() if key != "three_grid_gate_evidence_sha256"
    }
    sealed_sha = _canonical_v1_sha256(sealed_identity)
    if sealed_sha != sealed["three_grid_gate_evidence_sha256"]:
        raise ValueError("sealed evidence hash mismatch")
    reconstructed_gate, numerical_evidence_id = _reconstruct_gate_from_envelope_payload(
        parsed,
        persistence,
    )
    if numerical_evidence_id != returned_identity["numerical_evidence_id"]:
        raise ValueError("pipeline numerical evidence ID mismatch")
    if numerical_evidence_id != reconstructed_gate.numerical_evidence_id:
        raise ValueError("reconstructed pipeline property mismatch")
    envelope_identity = {
        key: value
        for key, value in parsed.items()
        if key != "three_grid_gate_evidence_envelope_sha256"
    }
    envelope_sha = _canonical_v1_sha256(envelope_identity)
    if envelope_sha != parsed["three_grid_gate_evidence_envelope_sha256"]:
        raise ValueError("envelope hash mismatch")

    diagnostic = cast(dict[str, object], sealed["diagnostic_50_to_25"])
    primary = cast(dict[str, object], sealed["primary_25_to_12_5"])
    expected_public = {
        "snapshot_gate_three_grid_gate_evidence_sha256_or_null": envelope_sha,
        "fit_attempt_three_grid_gate_evidence_sha256_or_null": envelope_sha,
        "staged_local_presence_map_value": envelope_sha,
        "snapshot_gate_grid_50_to_25_diagnostic_payload_sha256_or_null": diagnostic[
            "diagnostic_payload_sha256"
        ],
        "snapshot_gate_grid_25_to_12_5_expected_count_relative_difference_hex_or_null": primary[
            "relative_expected_count_difference_hex"
        ],
        "snapshot_gate_grid_25_to_12_5_density_l1_hex_or_null": primary[
            "density_l1_difference_hex"
        ],
    }
    if public_crosswalk != expected_public:
        raise ValueError("public gate/fit-attempt/presence crosswalk mismatch")
    return {
        "sealed_sha256": sealed_sha,
        "envelope_sha256": envelope_sha,
        "numerical_evidence_id": numerical_evidence_id,
    }


def _synthetic_installed_file_identity(
    source_path: Path,
    final_path: Path,
    *,
    attempt_id: str = "synthetic-r3-envelope",
) -> dict[str, object]:
    stat_result = source_path.stat()
    parent_stat = source_path.parent.stat()
    platform = "windows" if os.name == "nt" else "posix"
    parent_file_id = (
        _synthetic_file_id_128_hex(role="parent", identity_value=parent_stat.st_ino)
        if platform == "windows"
        else str(parent_stat.st_ino)
    )
    components = ["local_restricted", "three_grid_gate_evidence", final_path.name]
    directory_chain = _synthetic_directory_identity_chain(
        attempt_id=attempt_id,
        platform=platform,
        windows_volume_serial=(str(parent_stat.st_dev) if platform == "windows" else None),
        windows_parent_file_id=(parent_file_id if platform == "windows" else None),
        posix_st_dev=(str(parent_stat.st_dev) if platform == "posix" else None),
        posix_parent_st_ino=(parent_file_id if platform == "posix" else None),
    )
    evidence_parent = directory_chain[-1]
    separator = "\\" if platform == "windows" else "/"
    return {
        "profile_id": (
            "windows_ntfs_ntcreatefile_filerenameinfo_v1"
            if platform == "windows"
            else "posix_linkat_v1"
        ),
        "filesystem_name": "NTFS" if platform == "windows" else "synthetic-posix",
        "parent_identity": {
            "platform": platform,
            "canonical_final_path": evidence_parent["canonical_directory_path"],
            "exact_case_relative_components": components[:-1],
            "windows_volume_serial_u64_decimal_or_null": (
                str(parent_stat.st_dev) if platform == "windows" else None
            ),
            "windows_directory_file_id_16_bytes_hex_or_null": (
                parent_file_id if platform == "windows" else None
            ),
            "posix_st_dev_decimal_or_null": (
                str(parent_stat.st_dev) if platform == "posix" else None
            ),
            "posix_st_ino_decimal_or_null": (parent_file_id if platform == "posix" else None),
        },
        "ordered_directory_identity_chain": directory_chain,
        "ordered_directory_identity_chain_sha256": _canonical_v1_sha256(directory_chain),
        "final_repository_relative_path": (
            "data/processed/stage2R/etas_numerical_repair_fit_input/attempts/"
            f"{attempt_id}/{'/'.join(components)}"
        ),
        "temp_leaf_utf8_hex": source_path.name.encode("utf-8").hex(),
        "final_leaf_utf8_hex": final_path.name.encode("utf-8").hex(),
        "canonical_final_path": (
            f"{evidence_parent['canonical_directory_path']}{separator}{final_path.name}"
        ),
        "exact_case_relative_components": components,
        "platform": platform,
        "windows_volume_serial_u64_decimal_or_null": (
            str(stat_result.st_dev) if platform == "windows" else None
        ),
        "windows_file_id_16_bytes_hex_or_null": (
            _synthetic_file_id_128_hex(role="file", identity_value=stat_result.st_ino)
            if platform == "windows"
            else None
        ),
        "windows_number_of_links_at_temp_capture_or_null": (
            stat_result.st_nlink if platform == "windows" else None
        ),
        "posix_st_dev_decimal_or_null": str(stat_result.st_dev) if platform == "posix" else None,
        "posix_st_ino_decimal_or_null": str(stat_result.st_ino) if platform == "posix" else None,
        "posix_st_nlink_at_temp_capture_or_null": (
            stat_result.st_nlink if platform == "posix" else None
        ),
        "required_final_nlink": 1,
    }


def _assert_installed_file_identity(path: Path, expected: dict[str, object]) -> None:
    stat_result = path.stat()
    if expected["platform"] == "windows":
        actual_file_id = _synthetic_file_id_128_hex(
            role="file",
            identity_value=stat_result.st_ino,
        )
        expected_volume = expected["windows_volume_serial_u64_decimal_or_null"]
        expected_file_id = expected["windows_file_id_16_bytes_hex_or_null"]
    else:
        actual_file_id = str(stat_result.st_ino)
        expected_volume = expected["posix_st_dev_decimal_or_null"]
        expected_file_id = expected["posix_st_ino_decimal_or_null"]
    if str(stat_result.st_dev) != expected_volume:
        raise ValueError("volume/device identity changed")
    if actual_file_id != expected_file_id:
        raise ValueError("file identity changed")
    if stat_result.st_nlink != expected["required_final_nlink"]:
        raise ValueError("link count changed")
    components = cast(list[str], expected["exact_case_relative_components"])
    if path.name != components[-1]:
        raise ValueError("exact-case destination component changed")
    parent = cast(dict[str, object], expected["parent_identity"])
    parent_stat = path.parent.stat()
    if expected["platform"] == "windows":
        expected_parent_volume = parent["windows_volume_serial_u64_decimal_or_null"]
        expected_parent_id = parent["windows_directory_file_id_16_bytes_hex_or_null"]
        actual_parent_id = _synthetic_file_id_128_hex(
            role="parent",
            identity_value=parent_stat.st_ino,
        )
    else:
        expected_parent_volume = parent["posix_st_dev_decimal_or_null"]
        expected_parent_id = parent["posix_st_ino_decimal_or_null"]
        actual_parent_id = str(parent_stat.st_ino)
    if str(parent_stat.st_dev) != expected_parent_volume or actual_parent_id != expected_parent_id:
        raise ValueError("parent identity changed")
    platform = cast(str, expected["platform"])
    parent_path = _validate_platform_canonical_path(
        parent["canonical_final_path"],
        platform=platform,
        field_name="synthetic parent canonical path",
    )
    file_path = _validate_platform_canonical_path(
        expected["canonical_final_path"],
        platform=platform,
        field_name="synthetic file canonical path",
    )
    separator = "\\" if platform == "windows" else "/"
    if file_path != f"{parent_path}{separator}{path.name}":
        raise ValueError("canonical final path changed")


def _reopen_and_verify_synthetic_three_grid_envelope(
    protocol: dict[str, Any],
    path: Path,
    public_crosswalk: dict[str, object],
    *,
    live_directory_capture_provider: _SyntheticLiveDirectoryCaptureProvider,
    expected_attempt_id: str,
    expected_snapshot_id: str,
    expected_protocol_sha256: str,
    verification_order_events: list[str] | None = None,
) -> dict[str, str]:
    if type(live_directory_capture_provider) is not _SyntheticLiveDirectoryCaptureProvider:
        raise ValueError("fresh checkpoint requires a controlled live capture provider")
    captured_observed_live_chain = live_directory_capture_provider.capture_before_evidence_read()
    if verification_order_events is not None:
        verification_order_events.append("capture_independent_live_directory_chain")
    before = path.stat()
    reopened_bytes = path.read_bytes()
    after = path.stat()
    if verification_order_events is not None:
        verification_order_events.append("open_and_read_untrusted_complete_file")
    if (before.st_dev, before.st_ino, before.st_nlink, before.st_size) != (
        after.st_dev,
        after.st_ino,
        after.st_nlink,
        after.st_size,
    ) or len(reopened_bytes) != before.st_size:
        raise ValueError("installed envelope size changed during complete read")
    persistence = protocol["qualification"]["three_grid_gate_evidence_protocol"][
        "local_restricted_persistence"
    ]
    verified = _verify_synthetic_three_grid_envelope(
        protocol,
        reopened_bytes,
        public_crosswalk,
        expected_attempt_id=expected_attempt_id,
        expected_snapshot_id=expected_snapshot_id,
        expected_final_repository_relative_path=persistence["snapshot_path_by_id_exact"][
            expected_snapshot_id
        ].format(attempt_id=expected_attempt_id),
        expected_protocol_sha256=expected_protocol_sha256,
    )
    if verification_order_events is not None:
        verification_order_events.append("outer_sha_matches_independent_public_anchors")
    parsed = cast(dict[str, object], json.loads(reopened_bytes))
    authenticated_installed_identity = cast(
        dict[str, object],
        parsed["installed_file_identity"],
    )
    _verify_independently_observed_live_directory_identity_chain(
        authenticated_installed_identity,
        captured_observed_live_chain,
    )
    if verification_order_events is not None:
        verification_order_events.append("compare_live_chain_to_authenticated_embedded_chain")
    _assert_installed_file_identity(path, authenticated_installed_identity)
    if verification_order_events is not None:
        verification_order_events.append("accept_authenticated_file")
    return verified


def _null_three_grid_public_crosswalk() -> dict[str, object]:
    return {
        "snapshot_gate_three_grid_gate_evidence_sha256_or_null": None,
        "fit_attempt_three_grid_gate_evidence_sha256_or_null": None,
        "staged_local_presence_map_value": None,
        "snapshot_gate_grid_50_to_25_diagnostic_payload_sha256_or_null": None,
        "snapshot_gate_grid_25_to_12_5_expected_count_relative_difference_hex_or_null": None,
        "snapshot_gate_grid_25_to_12_5_density_l1_hex_or_null": None,
    }


def _verify_synthetic_three_grid_presence_file_set(
    protocol: dict[str, Any],
    evidence_directory: Path,
    public_crosswalk_by_snapshot: dict[str, dict[str, object]],
    installed_identity_by_snapshot: dict[str, dict[str, object]],
    *,
    expected_attempt_id: str,
    expected_protocol_sha256: str,
) -> tuple[dict[str, str | None], str]:
    if set(public_crosswalk_by_snapshot) != set(SNAPSHOT_ORDER):
        raise ValueError("presence crosswalk must contain exactly five snapshots")
    present_snapshots: set[str] = set()
    for snapshot_id in SNAPSHOT_ORDER:
        crosswalk = public_crosswalk_by_snapshot[snapshot_id]
        anchor_values = (
            crosswalk["snapshot_gate_three_grid_gate_evidence_sha256_or_null"],
            crosswalk["fit_attempt_three_grid_gate_evidence_sha256_or_null"],
            crosswalk["staged_local_presence_map_value"],
        )
        if all(value is None for value in anchor_values):
            if crosswalk != _null_three_grid_public_crosswalk():
                raise ValueError("absent evidence requires every grid crosswalk value to be null")
        elif len(set(anchor_values)) == 1 and all(
            _is_lowercase_hex(value, length=64) for value in anchor_values
        ):
            present_snapshots.add(snapshot_id)
        else:
            raise ValueError("gate/fit/presence outer envelope anchors disagree")
    observed_entries = {entry.name for entry in evidence_directory.iterdir()}
    expected_entries = {f"{snapshot_id}.json" for snapshot_id in present_snapshots}
    if observed_entries != expected_entries:
        raise ValueError("three-grid evidence file set does not match presence projection")
    if set(installed_identity_by_snapshot) != present_snapshots:
        raise ValueError("installed identity set does not match presence projection")

    presence_map: dict[str, str | None] = {}
    for snapshot_id in SNAPSHOT_ORDER:
        crosswalk = public_crosswalk_by_snapshot[snapshot_id]
        if snapshot_id not in present_snapshots:
            presence_map[snapshot_id] = None
            continue
        result = _reopen_and_verify_synthetic_three_grid_envelope(
            protocol,
            evidence_directory / f"{snapshot_id}.json",
            crosswalk,
            live_directory_capture_provider=(
                _synthetic_live_directory_capture_provider_from_path(
                    evidence_directory / f"{snapshot_id}.json",
                    attempt_id=expected_attempt_id,
                )
            ),
            expected_attempt_id=expected_attempt_id,
            expected_snapshot_id=snapshot_id,
            expected_protocol_sha256=expected_protocol_sha256,
        )
        presence_map[snapshot_id] = result["envelope_sha256"]
    return presence_map, _canonical_v1_sha256(presence_map)


def _ledger_entry_sha256(entry: dict[str, Any]) -> str:
    return _canonical_payload_sha256(
        {key: value for key, value in entry.items() if key != "entry_sha256"}
    )


def _ast_sha256(node: ast.AST) -> str:
    payload = ast.dump(
        node,
        annotate_fields=True,
        include_attributes=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _recursive_code_names(code: types.CodeType) -> set[str]:
    """Collect global/attribute names from a function and all nested code objects."""

    names = set(code.co_names)
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            names.update(_recursive_code_names(constant))
    return names


def _without_named_class_method(
    source_text: str,
    *,
    class_name: str,
    method_name: str,
) -> tuple[ast.Module, str]:
    """Remove one allowed method while preserving every other source byte and AST node."""

    module = ast.parse(source_text)
    matching_classes = [
        node for node in module.body if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    assert len(matching_classes) == 1
    class_node = matching_classes[0]
    matching_methods = [
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == method_name
    ]
    assert len(matching_methods) == 1
    method_node = matching_methods[0]
    assert method_node.end_lineno is not None

    first_line = min(
        [method_node.lineno, *(decorator.lineno for decorator in method_node.decorator_list)]
    )
    source_lines = source_text.splitlines(keepends=True)
    source_without_method = "".join(
        source_lines[: first_line - 1] + source_lines[method_node.end_lineno :]
    )
    class_node.body = [node for node in class_node.body if node is not method_node]
    return module, source_without_method


def test_protocol_yaml_loader_rejects_duplicate_mapping_keys() -> None:
    duplicate = "root:\n  value: 1\n  value: 2\n"
    try:
        yaml.load(duplicate, Loader=_UniqueKeySafeLoader)
    except yaml.constructor.ConstructorError:
        return
    raise AssertionError("duplicate YAML keys must fail closed")


def test_all_declared_dotted_protocol_refs_resolve() -> None:
    protocol = _load_yaml(PROTOCOL_PATH)
    resolved: list[tuple[str, str]] = []

    def resolve(reference: str) -> None:
        current: object = protocol
        for component in reference.split("."):
            assert isinstance(current, dict), reference
            assert component in current, (reference, component)
            current = current[component]

    def walk(value: object, path: tuple[str, ...] = ()) -> None:
        if isinstance(value, dict):
            for raw_key, item in value.items():
                key = str(raw_key)
                item_path = (*path, key)
                if isinstance(item, str) and (key.endswith("_ref") or key.endswith("_schema_ref")):
                    resolve(item)
                    resolved.append((".".join(item_path), item))
                if key.endswith("_refs") or key.endswith("_schema_refs"):
                    assert isinstance(item, dict), item_path
                    for nested_key, nested_reference in item.items():
                        assert isinstance(nested_reference, str), (*item_path, nested_key)
                        resolve(nested_reference)
                        resolved.append((".".join((*item_path, str(nested_key))), nested_reference))
                walk(item, item_path)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, (*path, str(index)))

    walk(protocol)
    assert len(resolved) >= 110


def test_repair_protocol_is_independent_target_blind_and_not_yet_executed() -> None:
    protocol = _load_yaml(PROTOCOL_PATH)

    assert protocol["protocol_version"] == "0.2.2"
    assert protocol["protocol_revision"] == "r3"
    assert protocol["stage"] == "2-ETAS-R"
    assert protocol["status"] == "preregistered_target_blind_before_any_repair_fit"
    assert protocol["preregistered_on"] == "2026-07-17"
    assert protocol["revised_on"] == "2026-07-19"
    assert protocol["revision_reason"] == (
        "freeze_attempt_local_complete_three_grid_envelope_identity_durability_and_"
        "fresh_disk_reopen_without_changing_any_scientific_input_public_schema_or_fit_rule"
    )
    revision_base = protocol["revision_base"]
    assert revision_base == {
        "protocol_tag": R2_PROTOCOL_TAG,
        "protocol_tag_object": R2_PROTOCOL_TAG_OBJECT,
        "protocol_commit": R2_PROTOCOL_COMMIT,
        "protocol_tag_object_type_exact": "annotated_tag",
        "protocol_tag_must_peel_exactly_to_protocol_commit": True,
        "remote_protocol_tag_and_peeled_commit_must_be_verified_before_r3_freeze": True,
    }
    tag_type = subprocess.run(
        ["git", "cat-file", "-t", R2_PROTOCOL_TAG],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    peeled_commit = subprocess.run(
        ["git", "rev-parse", f"{R2_PROTOCOL_TAG}^{{}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert tag_type == "tag"
    assert (
        subprocess.run(
            ["git", "rev-parse", R2_PROTOCOL_TAG],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        == R2_PROTOCOL_TAG_OBJECT
    )
    assert peeled_commit == R2_PROTOCOL_COMMIT
    publication = protocol["publication"]
    assert publication["protocol_tag"] == "v0.2.2-background-etas-repair-protocol-r3"
    assert publication["qualification_code_tag"] == "v0.2.2-background-etas-repair-code"
    assert publication["qualification_result_tag"] == (
        "v0.2.2-background-etas-numerical-qualification"
    )
    assert publication["comparator_adapter_code_tag"] == (
        "v0.2.2-background-etas-comparator-adapter"
    )
    assert publication["comparator_receipt_tag"] == "v0.2.2-background-etas-comparator"
    assert publication["qualification_result_tag_freezes_evaluable_or_not_evaluable"] is True
    assert publication["negative_result_requires_same_qualification_result_tag"] is True
    assert publication["adapter_code_tag_allowed_only_after_positive_qualification_result_tag"]
    assert publication["new_stage4_revision_requires_comparator_receipt_tag"] is True
    assert protocol["repair_code_scope_from_protocol_tag"]["comparison_base"] == (
        "v0.2.2-background-etas-repair-protocol-r3"
    )
    assert publication["exact_order"] == [
        "protocol_commit_push_and_remote_tag_verification",
        "repair_code_and_tests_commit_push_and_remote_tag_verification",
        "qualification_execution_and_positive_or_negative_result_commit_push_and_remote_tag_verification",
        "positive_only_adapter_code_and_tests_commit_push_and_remote_tag_verification",
        "positive_only_adapter_artifact_and_global_receipt_commit_push_and_remote_tag_verification",
        "positive_only_new_stage4_revision",
    ]
    assert protocol["parent"]["etas_status"] == "not_evaluable"
    assert protocol["parent"]["etas_primary_snapshot_converged_starts"] == [0] * 5
    assert protocol["parent"]["parent_scientific_result_may_not_be_reinterpreted"] is True

    target_blind = protocol["target_blindness"]
    assert target_blind["mode"] == "fit_only_before_any_stage4_target_read"
    assert target_blind["stage4_formal_target_consumer_read_count_required"] == 0
    assert target_blind["stage4_assessment_row_materialization_count_required"] == 0
    assert target_blind["stage2_causal_fit_source_access_must_be_separately_ledgered"] is True
    assert target_blind["locked_test_run_required"] is False
    for forbidden in (
        "anomaly_feature_read_allowed",
        "anomaly_result_read_allowed",
        "stage4_formal_target_read_allowed",
        "stage4_score_read_allowed",
        "stage9_locked_test_allowed",
        "stage2_holdout_assessment_interval_construction_allowed",
        "stage2_assessment_target_event_read_allowed",
        "information_gain_computation_allowed",
        "score_id_creation_allowed",
        "model_selection_allowed",
        "prior_stage2_scores_as_tuning_inputs_allowed",
        "parameter_bound_or_threshold_relaxation_allowed",
        "new_or_replacement_optimizer_starts_allowed",
        "failed_snapshot_omission_allowed",
    ):
        assert target_blind[forbidden] is False

    assert protocol["locked_test"] == {
        "run": False,
        "target_count": None,
        "score_ids": [],
        "artifact_ids": [],
        "result": None,
    }


def test_all_repair_input_bindings_match_current_frozen_bytes() -> None:
    protocol = _load_yaml(PROTOCOL_PATH)
    bindings = protocol["input_bindings"]
    for name, binding in bindings.items():
        path = Path(binding["path"])
        assert path.is_file(), name
        assert _sha256(path) == binding["sha256"], name

    assert bindings["parent_model_registry"]["role"] == (
        "provenance_only_not_read_by_qualification"
    )
    assert bindings["local_support_fold_manifest"]["role"] == (
        "provenance_only_not_read_by_fit_or_qualification"
    )

    local = protocol["local_restricted_input_identities"]
    assert local["ci_policy"] == "verify_frozen_metadata_only_and_never_require_local_files"
    assert local["local_acceptance_policy"] == (
        "every_file_must_exist_and_match_after_code_tag_and_before_execution"
    )
    for name, binding in local.items():
        if name in {"ci_policy", "local_acceptance_policy"}:
            continue
        path_text = binding.get("path", binding.get("path_from_parent_config"))
        assert isinstance(path_text, str) and path_text
        assert isinstance(binding["sha256"], str) and len(binding["sha256"]) == 64

    source = local["stage2_catalog_source"]
    assert source["same_physical_file_as_later_stage4_catalog"] is True
    assert source["allowed_action_after_code_tag"] == "stage2_causal_fit_source_query_only"
    assert source["stage4_formal_target_consumer_action_forbidden"] is True
    assert source["source_to_internal_field_mapping"] == {
        "event_id": "physical_event_id",
        "origin_time_utc": "origin_time",
        "available_at": "available_time",
        "longitude": "longitude",
        "latitude": "latitude",
        "magnitude": "magnitude",
        "inside_study_area": "inside_study_area",
    }
    assert local["frozen_kde_payload"]["role"] == (
        "provenance_only_not_read_by_fit_or_qualification"
    )


def test_five_fit_only_snapshots_are_frozen_without_assessment_intervals() -> None:
    snapshots = _load_yaml(PROTOCOL_PATH)["snapshots"]

    assert tuple(snapshots["order"]) == SNAPSHOT_ORDER
    assert snapshots["fit_start_utc"] == "2000-01-01T00:00:00Z"
    assert snapshots["history_start_utc"] == "1970-01-01T00:00:00Z"
    assert snapshots["common_mc"] == 4.0
    entries = snapshots["entries"]
    assert tuple(item["snapshot_id"] for item in entries) == SNAPSHOT_ORDER
    assert tuple(item["fit_end_utc"] for item in entries) == FIT_ENDS
    assert tuple(item["support_id"] for item in entries) == SUPPORT_IDS
    assert tuple(item["compensator_domain_id"] for item in entries) == COMPENSATOR_DOMAIN_IDS
    assert tuple(item["retained_area_fraction"] for item in entries) == (RETAINED_AREA_FRACTIONS)
    assert tuple(item["parent_role"] for item in entries) == PARENT_ROLES
    assert all("assessment_start_utc" not in item for item in entries)
    assert all("assessment_end_utc" not in item for item in entries)

    bundle = _load_yaml(PROTOCOL_PATH)["fit_input_bundle"]
    assert bundle["source_query_may_not_construct_or_return_assessment_rows"] is True
    assert bundle["column_projection_and_timestamp_predicate_must_be_applied_at_reader_boundary"]
    assert bundle["full_source_table_materialization_then_filter_forbidden"] is True
    assert bundle["reader_spy_must_prove_no_later_or_unavailable_row_was_materialized"]
    assert bundle["local_bundle_classification"] == "local_restricted"
    assert (
        bundle[
            "post_protocol_freeze_code_tag_required_before_any_new_qualification_source_open_stat_hash_query_or_bundle_inspection"
        ]
        is True
    )
    disclosed_probe = bundle["prefreeze_disclosed_source_probe"]
    probe_guard = (
        "no_additional_probe_allowed_after_protocol_freeze_before_remote_code_tag_verification"
    )
    assert disclosed_probe[probe_guard] is True
    assert {key: value for key, value in disclosed_probe.items() if key != probe_guard} == {
        "purpose": "protocol_drafting_verification_of_frozen_file_identity",
        "action": "one_read_only_file_level_sha256_check",
        "decoded_or_returned_row_count": 0,
        "fit_or_assessment_cohort_constructed": False,
        "stage4_formal_target_consumer_count": 0,
        "stage4_assessment_row_materialization_count": 0,
        "qualification_attempt_or_fit_input_bundle_created": False,
    }
    assert bundle["event_order"] == "origin_time_then_physical_event_id"
    selection = bundle["selection"]
    assert selection["fit_events"] == (
        "inside_snapshot_supported_domain_and_magnitude_gte_common_mc_and_origin_gt_fit_start_"
        "and_origin_lte_fit_end_and_available_at_lte_fit_end"
    )
    assert selection["parent_time"] == (
        "origin_gte_max_history_start_and_fit_start_minus_3650_days_and_origin_lte_fit_end_and_"
        "available_at_lte_fit_end"
    )
    assert selection["later_or_unavailable_event_count_required"] == 0
    expected_forbidden = {
        "event_id",
        "physical_event_id",
        "event_coordinates",
        "origin_time",
        "available_time",
        "longitude",
        "latitude",
        "projected_x",
        "projected_y",
        "assessment_event_count",
        "assessment_event_ids",
        "model_score",
        "information_gain",
        "score_id",
    }
    assert expected_forbidden == set(bundle["public_manifest_forbidden_fields"])
    counts = bundle["expected_parent_result_counts"]
    assert tuple(counts["snapshot_order"]) == SNAPSHOT_ORDER
    assert counts["fit_event_count"] == [385, 1287, 1828, 2342, 2734]
    assert counts["parent_event_count"] == [1875, 2874, 3592, 4263, 4802]
    assert counts["immigrant_kde_training_event_count"] == [3189, 4182, 4722, 5237, 5629]
    assert bundle["scientific_fit_input_sha256"]["must_be_bound_by_every_parameter_snapshot"]
    scientific_fields = [
        "snapshot_id",
        "fit_interval",
        "support_id",
        "compensator_domain_id",
        "common_mc_hex",
        "aki_b_hex",
        "beta_hex",
        "model_spec",
        "parameter_bounds",
        "optimizer_options",
        "stability_thresholds",
        "exact_start_vectors",
        "ordered_fit_events",
        "ordered_parent_events_and_roles",
        "ordered_quadrature_containers",
        "immigrant_kde_payload",
    ]
    assert bundle["scientific_fit_input_sha256"]["includes"] == scientific_fields
    assert (
        bundle["scientific_fit_input_record_schemas"]["scientific_fit_input_payload_fields_exact"]
        == scientific_fields
    )
    assert bundle["persistence"]["content_addressed_no_overwrite"] is True

    ledger = bundle["source_access_ledger"]
    assert ledger["path"] == "data/manifests/etas_numerical_repair_source_access_ledger.json"
    assert ledger["classification"] == "local_restricted_gitignored"
    assert ledger["selected_complete_acquisition_attempt_required_completed_action_counts"] == {
        "stage2_fit_source_metadata": 1,
        "stage2_fit_source_rows": 5,
    }
    assert ledger["selected_complete_acquisition_attempt_allowed_actions_exact"] == [
        "stage2_fit_source_metadata",
        "stage2_fit_source_rows",
    ]
    assert ledger["selected_complete_acquisition_attempt_exact_access_pair_count"] == 6
    assert ledger["selected_complete_acquisition_attempt_exact_ledger_entry_count"] == 12
    access_pairs = ledger["selected_complete_acquisition_attempt_ordered_access_pairs"]
    assert [item["pair_index"] for item in access_pairs] == list(range(6))
    assert [item["snapshot_id_or_null"] for item in access_pairs] == [None, *SNAPSHOT_ORDER]
    assert ledger["event_type_values_exact"] == ["intent", "completed", "aborted"]
    assert ledger["any_missing_extra_alias_duplicate_or_unknown_entry_field_forbidden"]
    reader_contract = bundle["reader_boundary_contract"]
    projection = dict(reader_contract["projection_payload"])
    projection_sha256 = projection.pop("canonical_sha256")
    assert _canonical_payload_sha256(projection) == projection_sha256
    predicates = reader_contract["predicate_payload_and_sha256_by_snapshot"]
    assert tuple(predicates) == SNAPSHOT_ORDER
    for index, snapshot_id in enumerate(SNAPSHOT_ORDER, start=1):
        predicate = dict(predicates[snapshot_id])
        predicate_sha256 = predicate.pop("canonical_sha256")
        assert _canonical_payload_sha256(predicate) == predicate_sha256
        assert access_pairs[index]["reader_projection_sha256_or_null"] == projection_sha256
        assert access_pairs[index]["reader_predicate_sha256_or_null"] == predicate_sha256
    pair_payload = {"ordered_access_pairs": access_pairs}
    pair_identity = bundle["public_source_access_receipt"][
        "selected_attempt_ordered_access_pair_identity"
    ]
    assert _canonical_payload_sha256(pair_payload) == pair_identity["reference_sha256"]
    assert ledger[
        "global_ledger_action_counts_may_exceed_selected_attempt_counts_after_interruption"
    ]
    two_phase = ledger["source_access_two_phase_protocol"]
    assert two_phase["intent_entry_fsynced_before_any_source_open_stat_hash_or_reader_call"]
    assert two_phase["existing_entries_may_not_be_deleted_truncated_or_rewritten"] is True
    assert two_phase["sealed_bundle_retry_must_reuse_verified_bundle_without_reopening_source"]
    assert two_phase["exactly_one_intent_and_exactly_one_completed_xor_aborted_terminal_per_pair"]
    assert two_phase["terminal_must_bind_intent_entry_sha256"] is True
    assert two_phase[
        "terminal_must_match_intent_action_snapshot_projection_predicate_code_commit_and_source_sha256"
    ]
    assert ledger["zero_counts_required_on_every_entry"] == {
        "stage4_formal_target_consumer_count_after": 0,
        "stage4_assessment_row_materialization_count_after": 0,
    }
    assert "full_source_table_materialization_then_filter" in ledger["forbidden_actions"]
    receipt = bundle["public_source_access_receipt"]
    assert receipt["stage4_formal_target_consumer_count_required"] == 0
    assert receipt["stage4_assessment_row_materialization_count_required"] == 0
    assert "local_ledger_content_sha256" in receipt["fields_exact"]

    replay = bundle["parent_replay_membership_equivalence"]
    assert replay["source_commit"] == "34fa7b4a491a062ff6e86daecf5568539661b42f"
    assert replay["counts_only_equivalence_forbidden"] is True
    assert replay["repair_and_parent_replay_ordered_identities_must_match_exactly"] is True
    assert replay["parent_replay_scientific_fit_input_sha256_required"] is True
    assert replay[
        "repair_scientific_fit_input_sha256_must_equal_parent_replay_scientific_fit_input_sha256"
    ]
    assert replay["full_value_payloads_required_by_snapshot"] == scientific_fields
    assert replay["full_value_payloads_must_equal_scientific_fit_input_includes_exactly"]
    assert replay[
        "parent_replay_must_construct_the_same_canonical_payload_schema_and_identical_bytes_before_hashing"
    ]
    assert replay["frozen_source_blobs"] == {
        "src/seismoflux/background/pipeline_local_support_etas.py": (
            "11b0b70ff900694780281e8da21123269c6463f1"
        ),
        "src/seismoflux/background/pipeline_poisson.py": (
            "63eab3bf4a62a0052ac05f287b9941fff5a946e5"
        ),
    }


def test_source_access_ledger_hash_chain_reference_is_non_self_referential() -> None:
    ledger = _load_yaml(PROTOCOL_PATH)["fit_input_bundle"]["source_access_ledger"]
    reference = ledger["hash_chain_reference_vector"]
    entry_0: dict[str, Any] = {
        "sequence": 0,
        "occurred_at_utc": "2026-07-17T00:00:00Z",
        "previous_entry_sha256_or_null": None,
        "intent_entry_sha256_or_null": None,
        "acquisition_attempt_id": "acq-0001",
        "access_id": "access-0001",
        "event_type": "intent",
        "code_tag_commit": "0" * 40,
        "source_sha256": "1" * 64,
        "action": "stage2_fit_source_metadata",
        "snapshot_id_or_null": None,
        "reader_projection_sha256_or_null": None,
        "reader_predicate_sha256_or_null": None,
        "materialized_row_count_or_null": None,
        "returned_row_count_or_null": None,
        "stage4_formal_target_consumer_count_after": 0,
        "stage4_assessment_row_materialization_count_after": 0,
    }
    entry_0["entry_sha256"] = _ledger_entry_sha256(entry_0)
    assert entry_0["entry_sha256"] == reference["entry_0_sha256"]

    entry_1: dict[str, Any] = {
        "sequence": 1,
        "occurred_at_utc": "2026-07-17T00:00:01Z",
        "previous_entry_sha256_or_null": entry_0["entry_sha256"],
        "intent_entry_sha256_or_null": entry_0["entry_sha256"],
        "acquisition_attempt_id": "acq-0001",
        "access_id": "access-0001",
        "event_type": "completed",
        "code_tag_commit": "0" * 40,
        "source_sha256": "1" * 64,
        "action": "stage2_fit_source_metadata",
        "snapshot_id_or_null": None,
        "reader_projection_sha256_or_null": None,
        "reader_predicate_sha256_or_null": None,
        "materialized_row_count_or_null": 0,
        "returned_row_count_or_null": 0,
        "stage4_formal_target_consumer_count_after": 0,
        "stage4_assessment_row_materialization_count_after": 0,
    }
    entry_1["entry_sha256"] = _ledger_entry_sha256(entry_1)
    assert entry_1["entry_sha256"] == reference["entry_1_sha256"]
    payload = {"schema_version": 1, "entries": [entry_0, entry_1]}
    assert _canonical_payload_sha256(payload) == reference["final_ledger_content_sha256"]

    mutated_0 = dict(entry_0)
    mutated_0["returned_row_count_or_null"] = 1
    mutated_0["entry_sha256"] = _ledger_entry_sha256(mutated_0)
    assert mutated_0["entry_sha256"] != reference["entry_0_sha256"]
    mutated_1 = dict(entry_1)
    mutated_1["previous_entry_sha256_or_null"] = mutated_0["entry_sha256"]
    mutated_1["intent_entry_sha256_or_null"] = mutated_0["entry_sha256"]
    mutated_1["entry_sha256"] = _ledger_entry_sha256(mutated_1)
    assert mutated_1["entry_sha256"] != reference["entry_1_sha256"]
    mutated_payload = {"schema_version": 1, "entries": [mutated_0, mutated_1]}
    assert _canonical_payload_sha256(mutated_payload) != reference["final_ledger_content_sha256"]


def test_scientific_fit_input_integer_encoding_keeps_only_grid_indices_signed() -> None:
    schemas = _load_yaml(PROTOCOL_PATH)["fit_input_bundle"]["scientific_fit_input_record_schemas"]
    assert schemas["quadrature_cell_integer_field_types"] == {
        "row": "strict_base10_integer",
        "column": "strict_base10_integer",
    }
    assert schemas["integer_encoding"] == {
        "signed_fields_exact": [
            "ordered_quadrature_containers.cells.row",
            "ordered_quadrature_containers.cells.column",
        ],
        "signed_field_encoding": "strict_base10_integer",
        "every_other_integer_field_encoding": "strict_nonnegative_base10_integer",
        "python_bool_is_not_an_integer": True,
    }
    spec = GridSpec(25.0)
    assert cell_id(spec, row=-1, column=2) == "g25000000_r-0000001_c+0000002"
    with pytest.raises(TypeError, match="row must be an integer"):
        cell_id(spec, row=True, column=2)


def test_parent_protocol_twenty_five_optimizer_starts_are_hex_exact() -> None:
    protocol = _load_yaml(PROTOCOL_PATH)
    manifest = _load_json(START_MANIFEST_PATH)
    without_identity = dict(manifest)
    recorded_identity = without_identity.pop("vector_payload_sha256")
    encoded = json.dumps(
        without_identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == recorded_identity
    assert (
        recorded_identity
        == protocol["input_bindings"]["optimizer_start_manifest"]["vector_payload_sha256"]
    )

    assert manifest["seed_protocol_version"] == "0.2.1"
    assert manifest["root_seed"] == 147
    assert manifest["bit_generator"] == "numpy.random.PCG64"
    assert tuple(item["snapshot_id"] for item in manifest["snapshots"]) == SNAPSHOT_ORDER
    assert sum(len(item["starts"]) for item in manifest["snapshots"]) == 25

    bounds = ETASParameterBounds().transformed()
    for snapshot in manifest["snapshots"]:
        assert snapshot["model_id"] == f"etas/{snapshot['snapshot_id']}"
        assert [item["start_index"] for item in snapshot["starts"]] == list(range(5))
        for item in snapshot["starts"]:
            regenerated = optimizer_start(
                bounds,
                root_seed=147,
                protocol_version="0.2.1",
                model_id=snapshot["model_id"],
                start_index=item["start_index"],
            )
            assert [float(value).hex() for value in regenerated] == item["transformed_hex"]


def test_repair_does_not_widen_bounds_or_change_the_optimizer() -> None:
    protocol = _load_yaml(PROTOCOL_PATH)
    model = protocol["etas_model"]
    assert model["parameter_bounds"] == {
        "background_rate_per_day": [0.01, 10.0],
        "productivity_k": [0.0001, 0.5],
        "alpha": [0.05, 2.0],
        "c_days": [0.001, 30.0],
        "p": [1.01, 2.5],
    }
    assert model["spatial_kernel"] == {
        "d_km2": 25.0,
        "q": 1.5,
        "gamma": 1.0,
        "cutoff_km": 300.0,
    }
    assert model["branching_ratio_maximum"] == 0.95

    repair = protocol["repair"]
    assert repair["affected_upper_bounds"] == {
        "background_rate_per_day": {
            "exact": "0x1.4000000000000p+3",
            "old_decoded": "0x1.4000000000001p+3",
        },
        "c_days": {
            "exact": "0x1.e000000000000p+4",
            "old_decoded": "0x1.e000000000001p+4",
        },
    }
    assert {
        "widen_physical_bounds",
        "tolerance_based_contains",
        "clip_out_of_domain_transformed_coordinates",
        "move_transformed_bounds_inward",
        "nextafter_expand_physical_domain",
    } == set(repair["forbidden_implementation"])
    assert protocol["randomness"]["seed_protocol_version"] == "0.2.1"
    assert protocol["optimizer"]["retry_with_new_seed_allowed"] is False
    assert protocol["optimizer"]["alternate_optimizer_allowed"] is False
    scope = protocol["repair_code_scope_from_protocol_tag"]
    etas_fit_rule = scope["existing_source_change_rules"]["src/seismoflux/background/etas_fit.py"]
    assert etas_fit_rule["only_symbol_allowed_to_change"] == (
        "ETASParameterBounds.from_transformed"
    )
    assert etas_fit_rule["every_other_ast_node_and_source_byte_range_must_match_protocol_tag"]
    assert scope["any_other_tracked_path_change_forbidden"] is True
    static_receipt = scope["etas_fit_static_call_graph_receipt"]
    assert static_receipt["protocol_tag_etas_fit_git_blob_oid"] == (
        "827ff2b8801c46ed5059231a5df64ce15320c0cf"
    )
    assert static_receipt["symbol_ast_sha256"] == {
        "fit_etas": "9c5da8d64c4f71424184056d2962e013703fd286a4f89ee739956ffbd5bf6caf",
        "etas_objective": "62778f464a77ff3b5ba08db421da2e807121c04b058ecd675e22b18527406b6d",
        "run_five_start_lbfgsb": "05a16648d6b5356db855db004e16d8032092de35884a72b5fd046b28f162a368",
        "three_point_gradient": "6be88b0786084ac637b5cff9aa1f2ad161ace87a594c6b4827df71c2d2db3e3d",
        "optimizer_start": "75ad95aff2c97e86ac1c526a21b18f7c78329121f818e74a97ab4f7cb4d50e87",
        "audit_stability": "1847456b17de82d88003cc0b0672600bac9fe34f8a52f6218a4d64209feec41d",
        "scipy_optimize_minimize_import": (
            "e396c4c61341edb49816428cb3f47557febea6867899a88cdaa91e2b8a786fc4"
        ),
    }
    assert static_receipt["all_other_module_ast_and_remaining_source_bytes_must_equal_protocol_tag"]
    assert static_receipt[
        "run_five_start_must_have_exactly_one_syntactic_minimize_call_inside_range_5"
    ]
    etas_fit_path = Path("src/seismoflux/background/etas_fit.py")
    baseline_bytes = subprocess.run(
        ["git", "cat-file", "blob", static_receipt["protocol_tag_etas_fit_git_blob_oid"]],
        check=True,
        capture_output=True,
    ).stdout
    assert (
        hashlib.sha256(baseline_bytes).hexdigest()
        == static_receipt["protocol_tag_etas_fit_file_sha256"]
    )
    baseline_text = baseline_bytes.decode("utf-8")
    current_text = etas_fit_path.read_bytes().decode("utf-8")
    baseline_without_allowed_ast, baseline_without_allowed_source = _without_named_class_method(
        baseline_text,
        class_name="ETASParameterBounds",
        method_name="from_transformed",
    )
    current_without_allowed_ast, current_without_allowed_source = _without_named_class_method(
        current_text,
        class_name="ETASParameterBounds",
        method_name="from_transformed",
    )
    assert current_without_allowed_source == baseline_without_allowed_source
    assert _ast_sha256(current_without_allowed_ast) == _ast_sha256(baseline_without_allowed_ast)

    baseline_module = ast.parse(baseline_text)
    current_module = ast.parse(current_text)
    baseline_symbol_nodes = {
        node.name: node
        for node in baseline_module.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    }
    current_symbol_nodes = {
        node.name: node
        for node in current_module.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    }
    baseline_import_node = next(
        node
        for node in baseline_module.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "scipy.optimize"
        and any(alias.name == "minimize" for alias in node.names)
    )
    current_import_node = next(
        node
        for node in current_module.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "scipy.optimize"
        and any(alias.name == "minimize" for alias in node.names)
    )
    for symbol in (
        "fit_etas",
        "etas_objective",
        "run_five_start_lbfgsb",
        "three_point_gradient",
        "optimizer_start",
        "audit_stability",
    ):
        expected_ast_sha256 = static_receipt["symbol_ast_sha256"][symbol]
        assert _ast_sha256(baseline_symbol_nodes[symbol]) == expected_ast_sha256
        assert _ast_sha256(current_symbol_nodes[symbol]) == expected_ast_sha256
    expected_import_ast_sha256 = static_receipt["symbol_ast_sha256"][
        "scipy_optimize_minimize_import"
    ]
    assert _ast_sha256(baseline_import_node) == expected_import_ast_sha256
    assert _ast_sha256(current_import_node) == expected_import_ast_sha256

    runtime_baseline = scope["repair_code_tag_prerequisite_public_artifacts"][
        "optimizer_runtime_baseline"
    ]
    expected_baseline_paths = [
        "src/seismoflux/background/etas_fit.py",
        "src/seismoflux/background/etas_numerical_repair.py",
        "src/seismoflux/background/etas_numerical_repair_io.py",
        "src/seismoflux/background/etas_numerical_repair_evidence.py",
        "src/seismoflux/background/visualization_etas_numerical_repair.py",
        "scripts/run_background_etas_numerical_repair.py",
        "tests/unit/test_etas_fit.py",
        "tests/unit/test_background_etas_numerical_repair.py",
    ]
    assert (
        runtime_baseline["prospective_project_blob_oid_and_file_sha256_map_exact_paths"]
        == expected_baseline_paths
    )
    assert set(runtime_baseline["prospective_project_blob_map_excludes_exact_paths"]) == {
        "data/manifests/etas_numerical_repair_code_diff_receipt.json",
        "data/manifests/etas_numerical_repair_optimizer_runtime_baseline.json",
    }
    assert runtime_baseline[
        "baseline_is_completed_before_code_diff_receipt_and_neither_artifact_hashes_itself"
    ]
    runtime_file_contract = runtime_baseline["python_runtime_file_map_contract"]
    assert runtime_file_contract["record_fields_exact"] == [
        "runtime_role",
        "origin_kind",
        "root_role",
        "canonical_root_relative_path",
        "file_sha256",
        "file_size",
    ]
    assert runtime_file_contract["root_role_values_exact"] == [
        "base_prefix",
        "venv_prefix",
        "windows_system_root",
    ]
    assert (
        runtime_file_contract["required_role_root_contract"]["active_venv_python_executable"]
        == "venv_prefix"
    )
    assert runtime_file_contract["runtime_role_values_exact"] == [
        "base_python_executable",
        "active_venv_python_executable",
        "python_shared_library",
        "runtime_dependency_module_origin",
        "loaded_native_dependency",
    ]
    assert runtime_file_contract["allowed_runtime_role_origin_kind_combinations_exact"] == {
        "base_python_executable": ["python_executable"],
        "active_venv_python_executable": ["python_executable"],
        "python_shared_library": ["python_shared_library"],
        "runtime_dependency_module_origin": ["stdlib_source", "stdlib_extension"],
        "loaded_native_dependency": ["loaded_native_dependency"],
    }
    assert (
        "every_loaded_native_dependency_of_verified_python_numpy_scipy_shapely_runtime_closure_once"
        in runtime_file_contract["required_runtime_coverage_requirements"]
    )
    assert (
        "every_loaded_shapely_geos_and_geos_c_native_image_once"
        in runtime_file_contract["required_runtime_coverage_requirements"]
    )
    assert (
        "every_loaded_python_shared_library_once"
        in runtime_file_contract["required_runtime_coverage_requirements"]
    )
    assert runtime_file_contract["native_dependency_capture_contract"][
        "libcrypto_vcruntime_and_transitive_BLAS_LAPACK_runtime_dependencies_may_not_be_omitted"
    ]
    native_capture = runtime_file_contract["native_dependency_capture_contract"]
    assert native_capture["qualification_preflight_must_run_exact_same_fixed_warmup_before_capture"]
    warmup = native_capture["synthetic_runtime_warmup_receipt_contract"]
    evidence_rules = scope["new_repair_module_execution_rules"][
        "src/seismoflux/background/etas_numerical_repair_evidence.py"
    ]
    assert evidence_rules["optimizer_call_forbidden_except_exact_runtime_warmup_callable"]
    assert evidence_rules["exact_runtime_warmup_optimizer_exception"] == (
        "one_direct_scipy_optimize_minimize_LBFGSB_call_with_frozen_synthetic_receipt_only"
    )
    assert evidence_rules["exact_runtime_warmup_geometry_exception"] == (
        "construct_two_fixed_synthetic_shapely_Points_read_x_y_and_call_"
        "BaseGeometry_equals_once_without_real_geometry"
    )
    assert evidence_rules[
        "runtime_warmup_may_not_import_or_call_etas_fit_etas_objective_run_five_start_or_any_real_scientific_orchestration"
    ]
    assert warmup["exact_callable_qualified_name"] == (
        "seismoflux.background.etas_numerical_repair_evidence._run_fixed_optimizer_runtime_warmup"
    )
    assert warmup["invocation_order_exact"] == [
        "scipy_optimize_minimize_lbfgsb",
        "numpy_linalg_solve",
        "scipy_spatial_ckdtree_query_ball_point",
        "shapely_point_xy_and_base_geometry_equals",
    ]
    assert warmup["receipt_fields_exact"] == [
        "schema_version",
        "callable_identity",
        "invocation_order",
        "canonical_input_payload",
        "canonical_input_payload_sha256",
        "canonical_output_payload",
        "canonical_output_payload_sha256",
        "shapely_runtime_binding_identity",
        "synthetic_runtime_warmup_receipt_sha256",
    ]
    assert warmup["shapely_runtime_binding_identity_fields_exact"] == [
        "fixed_path_branch_decision_receipt_sha256",
        "point_public_alias_dependency_record_sha256",
        "ordered_point_constructor_chain_dependency_record_sha256",
        "ordered_point_x_chain_dependency_record_sha256",
        "ordered_point_y_chain_dependency_record_sha256",
        "ordered_equals_chain_dependency_record_sha256",
        "shapely_runtime_binding_identity_sha256",
    ]
    branch_receipt = warmup["shapely_fixed_path_branch_decision_receipt"]
    assert branch_receipt["exact_values"] == {
        "point_argument_count": 2,
        "point_coordinate_array_dtype": "float64",
        "point_coordinate_array_ndim": 1,
        "point_numeric_dtype_branch": True,
        "deprecation_warn_from_comparison_executed": True,
        "deprecation_category_and_make_msg_branch_executed": False,
        "deprecation_warning_branch_taken": False,
        "multithreading_object_array_count": 0,
        "points_y_is_none": True,
        "points_z_is_none": True,
        "points_indices_is_none": True,
    }
    assert _canonical_payload_sha256(branch_receipt["exact_values"]) == (
        "4d010d7cdb5f1b7d35502d7b8c52db79e94f9c2a2b6b37fdfaac9925f80def37"
    )
    chain_preimages = warmup["shapely_runtime_binding_identity_chain_preimages"]
    assert chain_preimages["chain_preimage_item_fields_exact"] == [
        "dependency_record_id",
        "dependency_record_identity_sha256",
    ]
    assert chain_preimages["point_public_alias_dependency_record_id_exact"] == (
        "shapely.geometry.point.Point@direct_binding"
    )
    assert chain_preimages["ordered_point_constructor_chain_dependency_record_ids_exact"] == [
        "shapely.geometry.point.Point.__new__@direct_descriptor",
        "numpy.array@direct_binding",
        "numpy.ndarray.squeeze@direct_descriptor",
        "numpy.ndarray.ndim@direct_descriptor",
        "numpy.issubdtype@direct_binding",
        "numpy.ndarray.dtype@direct_descriptor",
        "numpy.number@direct_binding",
        "shapely.creation.points@deprecation_wrapper",
        "shapely.creation.points@multithreading_wrapper",
        "numpy.ndarray@direct_binding",
        "shapely.creation.points@wrapped_python_function",
        "shapely.creation._xyz_to_coords@direct_binding",
        "numpy.intc@direct_binding",
        "shapely.lib.points@native_numpy_ufunc",
        "shapely.geometry.point.Point@direct_binding",
    ]
    assert chain_preimages["ordered_point_x_chain_dependency_record_ids_exact"] == [
        "shapely.geometry.point.Point.x@direct_descriptor",
        "shapely._geometry.get_x@multithreading_wrapper",
        "numpy.ndarray@direct_binding",
        "shapely._geometry.get_x@wrapped_python_function",
        "shapely.lib.get_x@native_numpy_ufunc",
    ]
    assert chain_preimages["ordered_point_y_chain_dependency_record_ids_exact"] == [
        "shapely.geometry.point.Point.y@direct_descriptor",
        "shapely._geometry.get_y@multithreading_wrapper",
        "numpy.ndarray@direct_binding",
        "shapely._geometry.get_y@wrapped_python_function",
        "shapely.lib.get_y@native_numpy_ufunc",
    ]
    assert chain_preimages["ordered_equals_chain_dependency_record_ids_exact"] == [
        "shapely.geometry.base.BaseGeometry.equals@direct_descriptor",
        "shapely.predicates.equals@multithreading_wrapper",
        "numpy.ndarray@direct_binding",
        "shapely.predicates.equals@wrapped_python_function",
        "shapely.lib.equals@native_numpy_ufunc",
        "shapely.geometry.base._maybe_unpack@direct_binding",
        "numpy.generic.ndim@direct_descriptor",
        "numpy.generic.item@direct_descriptor",
    ]
    assert chain_preimages["ordered_chain_dependency_record_sha256_formula"] == (
        "sha256_seismoflux_canonical_json_v1_of_complete_ordered_chain_preimage_items"
    )
    assert chain_preimages["identity_field_to_preimage_exact"] == {
        "ordered_point_constructor_chain_dependency_record_sha256": (
            "ordered_point_constructor_chain_dependency_record_ids_exact"
        ),
        "ordered_point_x_chain_dependency_record_sha256": (
            "ordered_point_x_chain_dependency_record_ids_exact"
        ),
        "ordered_point_y_chain_dependency_record_sha256": (
            "ordered_point_y_chain_dependency_record_ids_exact"
        ),
        "ordered_equals_chain_dependency_record_sha256": (
            "ordered_equals_chain_dependency_record_ids_exact"
        ),
    }
    chain_fields = [
        "ordered_point_constructor_chain_dependency_record_ids_exact",
        "ordered_point_x_chain_dependency_record_ids_exact",
        "ordered_point_y_chain_dependency_record_ids_exact",
        "ordered_equals_chain_dependency_record_ids_exact",
    ]
    fake_runtime_map = {
        record_id: hashlib.sha256(record_id.encode("utf-8")).hexdigest()
        for field in chain_fields
        for record_id in chain_preimages[field]
    }
    expected_chain_aggregate_sha256 = {
        "ordered_point_constructor_chain_dependency_record_ids_exact": (
            "f8514cede60afcbf9db4b476b730a0d2bd4c2a815519b4d848d59d4f9ba365c7"
        ),
        "ordered_point_x_chain_dependency_record_ids_exact": (
            "7e705d74b231f9627a742107e3ef21b2a2d27774f6580bc1b4796e29ab229d35"
        ),
        "ordered_point_y_chain_dependency_record_ids_exact": (
            "136d1e9a00b8575cda707eb632d4a53640af47f99887b0e165b9b45273ee6124"
        ),
        "ordered_equals_chain_dependency_record_ids_exact": (
            "9b8ca09c557665ce1f907ca948ff828de0b82a27078b72af5844779ac4740cd0"
        ),
    }
    for field in chain_fields:
        preimage = [
            {
                "dependency_record_id": record_id,
                "dependency_record_identity_sha256": fake_runtime_map[record_id],
            }
            for record_id in chain_preimages[field]
        ]
        assert _canonical_payload_sha256(preimage) == expected_chain_aggregate_sha256[field]
        assert (
            _canonical_payload_sha256(list(reversed(preimage)))
            != (expected_chain_aggregate_sha256[field])
        )
    assert chain_preimages[
        "every_chain_item_dependency_record_id_must_resolve_exactly_once_in_same_runtime_callable_dependency_map_and_item_sha256_must_equal_that_record_dependency_record_identity_sha256"
    ]
    assert warmup["baseline_and_qualification_full_receipt_bytes_must_be_equal"]
    assert warmup[
        "baseline_and_qualification_shapely_input_output_and_descriptor_identity_must_be_equal_byte_for_byte"
    ]
    assert warmup[
        "warmup_must_force_first_use_loading_of_shapely_geometry_predicates_geos_and_geos_c_before_capture"
    ]
    shapely_warmup = warmup["canonical_input_payload_exact"][
        "shapely_point_xy_and_base_geometry_equals"
    ]
    assert shapely_warmup["defining_descriptor_qualified_names_exact"] == [
        "shapely.geometry.point.Point.x",
        "shapely.geometry.point.Point.y",
    ]
    assert warmup[
        "warmup_optimizer_calls_are_preflight_only_and_must_not_enter_five_snapshot_twenty_five_start_diagnostics"
    ]
    assert "synthetic_runtime_warmup_receipt" in runtime_baseline["fields_exact"]
    assert native_capture["single_file_classification_precedence_exact"] == [
        "active_venv_python_executable__python_executable",
        "base_python_executable__python_executable",
        "python_shared_library__python_shared_library",
        "runtime_dependency_module_origin__stdlib_extension",
        "loaded_native_dependency__loaded_native_dependency",
    ]
    assert native_capture["classification_examples"] == {
        "python311_dll": "python_shared_library__python_shared_library_only",
        "python3_dll": "python_shared_library__python_shared_library_only",
        "_hashlib_pyd": "runtime_dependency_module_origin__stdlib_extension_only",
        "_ctypes_pyd": "runtime_dependency_module_origin__stdlib_extension_only",
        "base_prefix_Library_bin_libcrypto_dll": (
            "loaded_native_dependency__loaded_native_dependency"
        ),
    }
    assert native_capture["loaded_image_coverage_record_predicate_exact"] == (
        "runtime_role_is_active_venv_python_executable_or_python_shared_library_or_"
        "loaded_native_dependency_OR_runtime_role_is_runtime_dependency_module_origin_"
        "and_origin_kind_is_stdlib_extension"
    )
    assert runtime_file_contract["canonical_record_key_exact"] == [
        "root_role",
        "canonical_root_relative_path",
    ]
    assert runtime_file_contract["canonical_root_relative_path_normalization_exact"] == (
        "resolve_final_path_then_selected_root_relative_then_forward_slash_then_unicode_NFC_"
        "then_windows_ordinal_lowercase"
    )
    dependency_contract = runtime_baseline["runtime_callable_dependency_map_contract"]
    assert {
        "project_class",
        "project_dunder_method",
        "project_property",
        "scipy_class",
        "shapely_callable",
        "shapely_property",
    } <= set(dependency_contract["dependency_kind_values_exact"])
    assert dependency_contract["closure_membership_values_exact"] == [
        "optimizer_fit_runtime_closure",
        "synthetic_runtime_warmup_closure",
        "three_grid_runtime_closure",
        "adapter_artifact_runtime_closure",
    ]
    assert "verified_shapely_RECORD_file" in dependency_contract["origin_kind_values_exact"]
    assert dependency_contract[
        "project_class_requires_defining_module_attribute_identity_project_blob_and_class_ast_sha256_with_null_code_object_sha256"
    ]
    assert dependency_contract[
        "project_property_requires_exact_class_descriptor_identity_and_fget_ast_and_canonical_code_object_sha256"
    ]
    assert dependency_contract["property_may_not_be_classified_as_project_class_method"]
    assert dependency_contract["record_key_exact"] == [
        "canonical_binding_path",
        "callable_layer",
    ]
    assert {
        "dependency_record_id",
        "binding_alias_paths_exact",
        "wrapped_target_dependency_record_id_or_null",
        "closure_cell_bindings",
        "native_ufunc_identity_or_null",
        "dependency_record_identity_sha256",
    } <= set(dependency_contract["record_fields_exact"])
    assert {
        "deprecation_wrapper",
        "multithreading_wrapper",
        "wrapped_python_function",
        "native_numpy_ufunc",
    } <= set(dependency_contract["callable_layer_values_exact"])
    assert dependency_contract[
        "wrapper___wrapped___must_be_identical_to_target_record_object_and_multithreading_closure_func_cell"
    ]
    assert dependency_contract["closure_cell_role_values_exact"] == [
        "callable_traversal_target",
        "executed_fixed_path_noncallable_configuration",
        "inert_unexecuted_branch_decorator_configuration",
    ]
    noncallable_cells = dependency_contract["deprecation_wrapper_noncallable_closure_cells_exact"]
    assert set(noncallable_cells) == {
        "category",
        "make_msg",
        "warn_from",
    }
    assert noncallable_cells["warn_from"]["cell_role"] == (
        "executed_fixed_path_noncallable_configuration"
    )
    for cell_name in ("category", "make_msg"):
        assert noncallable_cells[cell_name]["cell_role"] == (
            "inert_unexecuted_branch_decorator_configuration"
        )
    for cell in noncallable_cells.values():
        assert set(cell["nontraversed_value_identity"]) == set(
            dependency_contract["nontraversed_value_identity_fields_exact"]
        )
    assert {
        "shapely_distribution_name_and_version",
        "shapely_dist_info_RECORD_sha256",
        "shapely_complete_installed_distribution_verification_map",
        "shapely_complete_installed_distribution_verification_map_sha256",
        "shapely_geos_loaded_native_dependency_file_sha256_and_size_map",
        "shapely_geos_loaded_native_dependency_file_sha256_and_size_map_sha256",
        "runtime_callable_dependency_map",
        "runtime_callable_dependency_map_sha256",
        "three_grid_runtime_dependency_closure_sha256",
    } <= set(runtime_baseline["fields_exact"])
    test_rule = scope["existing_test_change_rules"]["tests/unit/test_etas_fit.py"]
    assert test_rule["only_new_test_functions_may_be_appended"]
    assert test_rule["deletion_rename_skip_xfail_or_assertion_weakening_forbidden"]
    current_test_module = ast.parse(Path("tests/unit/test_etas_fit.py").read_text(encoding="utf-8"))
    current_test_names = [
        node.name
        for node in current_test_module.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and node.name.startswith("test_")
    ]
    frozen_test_names = test_rule["frozen_protocol_tag_test_function_names_exact"]
    assert test_rule["frozen_test_count_exact"] == 19
    assert current_test_names[: len(frozen_test_names)] == frozen_test_names
    assert len(current_test_names) == len(set(current_test_names))
    assert test_rule["duplicate_definition_shadow_collection_disappearance_skip_or_xfail_forbidden"]


def test_qualification_and_stage4_receipt_are_fail_closed() -> None:
    protocol = _load_yaml(PROTOCOL_PATH)
    diagnostics = protocol["diagnostics"]
    assert diagnostics["primary_record_count"] == 25
    assert diagnostics["exact_row_key"] == ["snapshot_id", "start_index"]
    assert diagnostics["exact_snapshot_and_start_cartesian_product_required"] is True
    assert diagnostics["nonfinite_serialization"] == "null_plus_explicit_failure_code"
    assert diagnostics["missing_failed_start_row_action"] == (
        "invalidate_attempt_without_qualification_result"
    )
    assert "attempt_id" in diagnostics["required_fields"]
    assert "optimizer_invocation_receipt_sha256" in diagnostics["required_fields"]
    assert diagnostics["any_unlisted_failure_code_invalidates_attempt"] is True
    assert diagnostics[
        "required_fields_are_exact_and_any_extra_alias_duplicate_or_unknown_field_is_forbidden"
    ]
    assert "raw_scipy_success" in diagnostics["required_fields"]
    assert "etas_start_scipy_converged" in diagnostics["required_fields"]
    assert diagnostics["start_failure_code_first_match_precedence"] == [
        "terminal_vector_missing_or_nonfinite",
        "terminal_physical_decode_invalid",
        "objective_missing_or_nonfinite",
        "gradient_missing_or_nonfinite",
        "scipy_not_success",
        "gradient_threshold_exceeded",
    ]
    assert diagnostics["every_row_must_bind_one_actual_started_and_completed_optimizer_invocation"]
    assert "fit_etas_call_closing_receipt_sha256" in diagnostics["required_fields"]
    assert "diagnostic_row_sha256" in diagnostics["required_fields"]
    assert diagnostics["diagnostic_row_sha256_formula"].endswith(
        "row_without_diagnostic_row_sha256"
    )
    invocation = protocol["optimizer_invocation_receipt_protocol"]
    assert invocation["exact_fit_call_count"] == 5
    assert invocation["exact_optimizer_invocation_count"] == 25
    assert invocation["completion_kind_for_valid_execution"] == "returned_normally"
    assert invocation[
        "any_exception_or_missing_minimize_call_is_invalid_execution_not_numerical_negative"
    ]
    assert invocation["transparent_wrapper"][
        "wrapper_must_call_captured_original_scipy_minimize_exactly_once_with_same_objects_and_values"
    ]
    assert invocation["transparent_wrapper"][
        "wrapper_must_return_exact_same_OptimizeResult_object_without_copy_or_mutation"
    ]
    assert invocation["transparent_wrapper"][
        "original_module_global_minimize_must_be_restored_in_finally_and_identity_rechecked"
    ]
    invocation_fields = invocation["optimizer_invocation_receipt_fields"]
    assert "raw_OptimizeResult_canonical_payload" in invocation_fields
    assert "raw_OptimizeResult_canonical_sha256" in invocation_fields
    assert not {
        "raw_OptimizeResult_x_hex",
        "raw_OptimizeResult_fun_hex_or_null",
        "raw_OptimizeResult_success",
        "raw_OptimizeResult_status",
        "raw_OptimizeResult_nit",
        "raw_OptimizeResult_nfev",
        "raw_OptimizeResult_njev_or_null",
        "raw_OptimizeResult_message",
    } & set(invocation_fields)
    raw_schema = invocation["canonical_subpayload_schemas"]["raw_OptimizeResult_canonical_sha256"]
    assert raw_schema["receipt_must_embed_complete_payload_and_separate_recomputable_sha256"]
    assert raw_schema["exact_keys_required"] == [
        "fun",
        "hess_inv",
        "jac",
        "message",
        "nfev",
        "nit",
        "njev",
        "status",
        "success",
        "x",
    ]
    assert raw_schema["numeric_scalar_wrapper_fields_exact"] == [
        "value_hex_or_null",
        "numeric_state",
        "original_nonfinite_kind_or_null",
    ]
    assert raw_schema["numeric_state_values_exact"] == ["finite", "nonfinite", "absent_null"]
    assert raw_schema["vector_shapes_if_present_exact"] == {"x": [5], "jac": [5]}
    assert raw_schema["present_vector_runtime_type_dtype_layout_exact"] == {
        "type": "numpy.ndarray",
        "dtype": "float64",
        "C_contiguous": True,
    }
    assert raw_schema[
        "hess_inv_array_element_wrappers_may_use_only_finite_or_nonfinite_never_absent_null"
    ]
    assert invocation["valid_execution_requires"] == [
        "five_fit_etas_calls_started_and_returned_normally",
        "twenty_five_original_scipy_minimize_calls_started_and_returned_normally",
        "exact_five_calls_per_snapshot_in_start_index_order",
        "no_preoptimizer_initial_objective_short_circuit",
        "no_wrapper_observed_exception",
        "original_callable_restored_after_every_snapshot",
        "returned_fit_result_contains_exactly_the_same_five_observed_start_results_in_order",
    ]
    assert invocation["receipt_hash_DAG_order"] == (
        "fit_call_opening_then_optimizer_invocations_then_fit_call_closing_then_"
        "diagnostic_rows_then_three_grid_gate_evidence_then_snapshot_gate_result_then_"
        "fit_attempt_snapshot"
    )
    closing_fields = invocation["fit_etas_call_closing_receipt_fields"]
    assert closing_fields[0] == "schema_version"
    assert {
        "returned_five_start_results_canonical_payload",
        "returned_five_start_results_canonical_sha256",
        "returned_fit_result_canonical_payload",
        "returned_fit_result_canonical_sha256",
    } <= set(closing_fields)
    returned_fit_schema = invocation["canonical_subpayload_schemas"][
        "returned_fit_result_canonical_payload"
    ]
    assert returned_fit_schema["result_branch_values_exact"] == [
        "no_stability_eligible_start",
        "stability_eligible_but_unstable",
        "stable",
    ]
    assert returned_fit_schema["hessian_metric_state_values_exact"] == [
        "finite",
        "nonfinite",
        "absent",
    ]
    closure = invocation["runtime_callable_preconditions"]["runtime_global_dependency_closure"]
    assert {
        "ETASModelSpec.validate_event",
        "ETASModelSpec.magnitude_span",
        "PointAreaQuadrature.integrate",
        "PointAreaQuadrature.inverse_power_masses",
        "SeedContext.fields",
        "SeedContext.digest",
        "SeedContext.entropy",
        "SeedContext.generator",
    } <= set(closure["critical_class_methods_and_properties"])
    expected_edges = {
        "fit_etas": [
            "ETASParameterBounds.transformed",
            "ETASParameterBounds.from_transformed",
            "ETASModelSpec.validate_parameters",
        ],
        "etas_objective.<locals>.objective": ["ETASParameterBounds.from_transformed"],
        "evaluate_prepared_likelihood": ["ETASModelSpec.validate_parameters"],
        "observed_hessian_delta_uncertainty": [
            "ETASModelSpec.validate_parameters",
            "ETASModelSpec.branching_ratio",
        ],
        "prepare_etas_likelihood": [
            "ETASModelSpec.validate_event",
            "PointAreaQuadrature.integrate",
        ],
        "_prepare_compensator_arrays": ["PointAreaQuadrature.inverse_power_masses"],
        "_spatial_parent_mass": ["PointAreaQuadrature.integrate"],
        "ETASParameterBounds.from_transformed": [
            "ETASParameterBounds.transformed",
            "ETASParameterBounds.contains",
        ],
        "ETASModelSpec.validate_parameters": ["ETASModelSpec.branching_ratio"],
        "ETASModelSpec.branching_ratio": ["ETASModelSpec.magnitude_span"],
        "_truncated_gr_expectation_alpha_derivative": ["ETASModelSpec.magnitude_span"],
        "optimizer_start": ["SeedContext.generator"],
        "SeedContext.generator": ["SeedContext.entropy"],
        "SeedContext.entropy": ["SeedContext.digest"],
        "SeedContext.digest": ["SeedContext.fields"],
        "_evaluate_background_density_many": [
            "seismoflux.background.pipeline_etas._KDEBackgroundDensity.density_many",
            "seismoflux.background.pipeline_etas._KDEBackgroundDensity.__call__",
        ],
        "_validated_background_density": [
            "seismoflux.background.pipeline_etas._KDEBackgroundDensity.__call__"
        ],
        "seismoflux.background.pipeline_etas._KDEBackgroundDensity.density_many": [
            "seismoflux.background.poisson.SpatialPoissonModel.density"
        ],
        "seismoflux.background.pipeline_etas._KDEBackgroundDensity.__call__": [
            "seismoflux.background.poisson.SpatialPoissonModel.density_scalar"
        ],
        "seismoflux.background.poisson.SpatialPoissonModel.density": [
            "seismoflux.background.poisson.GaussianMixtureFamily.raw_densities"
        ],
        "seismoflux.background.poisson.SpatialPoissonModel.density_scalar": [
            "seismoflux.background.poisson.SpatialPoissonModel.density"
        ],
        "seismoflux.background.poisson.GaussianMixtureFamily.raw_densities": [
            "seismoflux.background.poisson.GaussianMixtureFamily.training_event_count"
        ],
    }
    assert closure["explicit_scientific_instance_method_edges"] == expected_edges
    assert {target for targets in expected_edges.values() for target in targets} == set(
        closure["critical_class_methods_and_properties"]
    )
    assert closure[
        "protocol_tag_pre_repair_edge_absent_but_repair_code_tag_edge_required_exact"
    ] == {"ETASParameterBounds.from_transformed": ["ETASParameterBounds.transformed"]}
    assert closure["third_party_scientific_instance_descriptors_exact"] == [
        "numpy.random.Generator.uniform",
        "scipy.spatial.cKDTree.query_ball_point",
    ]
    assert closure["explicit_third_party_scientific_instance_descriptor_edges"] == {
        "optimizer_start": ["numpy.random.Generator.uniform"],
        "_query_ball_indices": ["scipy.spatial.cKDTree.query_ball_point"],
    }
    assert invocation["runtime_callable_preconditions"]["critical_global_object_identity_edges"][
        "optimizer_start"
    ] == ["SeedContext"]
    assert "run_etas_numerical_repair_qualification" not in closure["roots"]
    assert "_validate_optimizer_runtime" not in closure["roots"]
    assert closure["nonrecursive_orchestration_identity_only_roots"] == [
        "run_etas_numerical_repair_qualification",
        "_validate_optimizer_runtime",
    ]
    assert closure[
        "every_unresolved_frozen_project_scientific_instance_LOAD_ATTR_LOAD_METHOD_or_call_must_be_in_explicit_edges_exactly_once"
    ]
    assert closure[
        "every_separately_enumerated_third_party_scientific_descriptor_call_must_be_in_explicit_third_party_edges_exactly_once"
    ]

    qualification = protocol["qualification"]
    assert qualification["evaluable_requires_all_five_primary_snapshots_pass"] is True
    assert qualification["partial_success_adoption_allowed"] is False
    assert qualification["threshold_relaxation_after_results_allowed"] is False
    assert qualification["any_valid_numerical_gate_failure_action"] == (
        "publish_target_blind_numerical_negative_and_keep_stage4_blocked"
    )
    classification = qualification["outcome_classification"]
    assert set(classification) == {"evaluable", "not_evaluable", "invalid_execution"}
    assert qualification["invalid_execution_may_not_publish_qualification_manifest_or_result_tag"]
    assert qualification["implementation_exception_may_not_be_reclassified_as_numerical_negative"]
    fit_payload = protocol["optimizer_invocation_receipt_protocol"]["canonical_subpayload_schemas"][
        "returned_fit_result_canonical_payload"
    ]
    stability_fields = fit_payload["stability_fields_exact"]
    assert "best_three_relative_objective_range_nonfinite_kind_or_null" in stability_fields
    assert "best_three_transformed_parameter_range_nonfinite_kind_or_null" in stability_fields
    assert fit_payload["stability_range_metric_state_contract"] == {
        "finite": {"value_hex": "required", "nonfinite_kind": None},
        "nonfinite": {"value_hex": None, "nonfinite_kind": "required"},
        "absent": {"value_hex": None, "nonfinite_kind": None},
    }
    assert fit_payload[
        "either_stability_range_metric_nonfinite_requires_stable_false_and_corresponding_frozen_spread_failure_reason"
    ]
    requirements = qualification["per_snapshot_conjunctive_requirements"]
    assert requirements == {
        "exact_start_count": 5,
        "minimum_converged_starts": 4,
        "every_counted_converged_gradient_infinity_norm_lte": 1.0e-4,
        "best_three_relative_objective_range_lte": 1.0e-4,
        "best_three_transformed_parameter_maximum_range_lte": 0.1,
        "hessian_minimum_eigenvalue_gte": 1.0e-8,
        "hessian_condition_number_lte": 1.0e10,
        "branching_ratio_lt": 0.95,
        "fit_only_25_to_12_5km_expected_count_relative_difference_lte": 0.02,
        "fit_only_25_to_12_5km_density_l1_lte": 0.05,
    }
    selected = qualification["selected_start_rule"]
    assert selected["exact_order"] == "objective_ascending_then_start_index_ascending"
    assert selected["selected_start"] == "first_stability_eligible_row_in_exact_order"
    assert selected["selected_start_must_equal_hessian_evaluation_point"] is True
    assert selected["selected_start_must_equal_fit_result_best_parameters_and_objective"] is True
    assert selected["mismatch_action"] == "invalid_execution_without_qualification_result"
    assert protocol["numerical_regression"]["primary_grid_gate"] == "25_to_12_5km"
    assert protocol["numerical_regression"]["diagnostic_grid_pair"] == (
        "50_to_25km_record_required_not_a_pass_gate"
    )

    receipt = protocol["stage4_receipt"]
    assert receipt["required_only_if_qualification_status"] == "evaluable"
    assert receipt["role_order"] == ["development", "formal_validation", "prospective"]
    assert receipt["exact_global_frozen_comparator_receipt_count"] == 1
    assert receipt[
        "global_receipt_contains_complete_ordered_role_mapping_and_role_parameter_hashes"
    ]
    assert receipt["selected_role_field_in_global_receipt_forbidden"] is True
    assert receipt[
        "required_global_hashes_are_sibling_bindings_in_stage4_evidence_not_fields_nested_inside_each_other"
    ]
    assert receipt[
        "frozen_etas_comparator_receipt_may_not_include_adapter_artifact_closing_seal_sha256"
    ]
    assert receipt["required_global_hashes"] == [
        "etas_artifact_sha256",
        "etas_parameter_set_sha256",
        "etas_numerical_qualification_evidence_sha256",
        "frozen_etas_comparator_receipt_sha256",
        "adapter_artifact_closing_seal_sha256",
    ]
    assert receipt["required_per_role_hashes"] == ["etas_parameter_snapshot_sha256"]
    assert receipt["parameter_set_role_mapping"] == {
        "development": "fold_4",
        "formal_validation": "final_validation",
        "prospective": "final_validation",
    }
    assert receipt["qualification_must_bind_all_five_snapshot_parameter_hashes"] is True
    assert receipt["adapter_has_no_file_io"] is True
    assert receipt["window_events_may_not_backfill_issue_time_forecast_map"] is True
    assert receipt["static_kde_adapter_may_not_be_renamed_or_substituted_as_etas"] is True
    assert receipt["new_stage4_execution_revision_required"] is True
    assert receipt["current_stage4_r2_remains_blocked"] is True
    expected_stage4_binding_chain = [
        "protocol_design_sha256",
        "random_input_seal",
        "ScoreBlindInputEvidence",
        "FormalPreflightReceipt",
        "Stage4QualificationEvidence",
        "Stage4ScoringSeal_and_execution_binding",
        "TargetBlindFormalContext",
        "Stage4InMemoryPlan",
        "PlaceboRequest_and_PlaceboSource",
        "placebo_checkpoint_and_result",
        "final_registry_model_card_and_fingerprint",
    ]
    assert receipt["future_stage4_revision_required_binding_chain"] == (
        expected_stage4_binding_chain
    )
    stage4_object_contract = receipt["future_stage4_every_binding_object_contract"]
    assert stage4_object_contract[
        "object_names_must_equal_future_stage4_revision_required_binding_chain_exactly"
    ]
    assert stage4_object_contract["every_object_must_bind_as_sibling_external_hashes"] == [
        "frozen_etas_comparator_receipt_sha256",
        "adapter_artifact_closing_seal_sha256",
    ]
    assert stage4_object_contract[
        "every_object_must_also_bind_etas_artifact_and_parameter_set_sha256"
    ]

    stage4_r2 = _load_yaml(STAGE4_R2_PROTOCOL_PATH)
    r2_required = stage4_r2["evaluation"]["multiple_comparisons"]["confirmatory_gatekeeping"][
        "future_post_etas_preregistered_design"
    ]["comparator_contract"]["mandatory_primary_scientific_comparator"]["required_bindings"]
    repair_role_required = {
        "etas_artifact_sha256",
        "etas_parameter_snapshot_sha256",
        "etas_numerical_qualification_evidence_sha256",
    }
    assert set(r2_required) == repair_role_required
    assert repair_role_required <= (
        set(receipt["required_global_hashes"]) | set(receipt["required_per_role_hashes"])
    )


def test_three_grid_runtime_dependency_closure_includes_shapely_and_post_return_chain() -> None:
    protocol = _load_yaml(PROTOCOL_PATH)
    runtime = protocol["optimizer_invocation_receipt_protocol"]["runtime_callable_preconditions"]
    closure = runtime["three_grid_runtime_dependency_closure"]

    assert closure["closure_membership_name_exact"] == "three_grid_runtime_closure"
    assert closure["primary_root_exact"] == (
        "seismoflux.background.pipeline_etas._grid_gate_evidence"
    )
    assert set(closure["post_return_property_roots_exact"]) == {
        "seismoflux.background.pipeline_etas.ETASGridGateEvidence.passed",
        "seismoflux.background.pipeline_etas.ETASGridGateEvidence.failure_reasons",
        "seismoflux.background.pipeline_etas.ETASGridGateEvidence.numerical_evidence_id",
        "seismoflux.background.grid.ThreeGridConvergenceGateEvidence.passed",
    }

    edges = closure["project_dependency_edges_exact"]
    assert edges["seismoflux.background.pipeline_etas._grid_gate_evidence"] == [
        "seismoflux.background.grid.EqualAreaGridFamily.at",
        "seismoflux.background.adapters.point_area_quadrature_from_grid",
        "seismoflux.background.etas_fit.evaluate_etas_cell_expected_masses",
        "seismoflux.background.pipeline_etas._canonical_sha256",
        "seismoflux.background.pipeline_etas.ETASGridResolutionEvidence",
        "seismoflux.background.grid.diagnose_three_grid_convergence",
        "seismoflux.background.pipeline_etas.ETASGridGateEvidence",
    ]
    assert edges[
        "seismoflux.background.etas_fit.PointAreaQuadrature.inverse_power_weighted_point_sums"
    ] == [
        "seismoflux.background.etas_fit._query_ball_indices",
        "seismoflux.background.etas_fit._vectorized_inverse_power_density_squared",
        "seismoflux.background.etas_fit._readonly_float",
    ]
    assert edges[
        "seismoflux.background.pipeline_etas.ETASGridGateEvidence.numerical_evidence_id"
    ] == [
        "seismoflux.background.pipeline_etas._canonical_sha256",
        "seismoflux.background.grid.ThreeGridConvergenceGateEvidence.comparisons",
    ]
    assert edges["seismoflux.background.grid.EqualAreaGrid.cell_ids"] == [
        "seismoflux.background.grid.GridCell.id"
    ]
    assert edges["seismoflux.background.grid.GridCell.id"] == ["seismoflux.background.grid.cell_id"]
    assert edges["seismoflux.background.artifacts._canonicalize"] == [
        "seismoflux.background.artifacts._canonicalize"
    ]

    expected_dunders = {
        "seismoflux.background.etas_fit.QuadraturePoint.__post_init__",
        "seismoflux.background.etas_fit.PointAreaQuadrature.__post_init__",
        "seismoflux.background.etas_fit.ETASExpectedCellMasses.__post_init__",
        "seismoflux.background.pipeline_etas.ETASGridResolutionEvidence.__post_init__",
        "seismoflux.background.pipeline_etas.ETASGridGateEvidence.__post_init__",
        "seismoflux.background.grid.ThreeGridConvergenceGateEvidence.__post_init__",
    }
    assert set(closure["project_dunder_methods_required_exact"]) == expected_dunders
    assert {
        "seismoflux.background.etas_fit.PointAreaQuadrature.inverse_power_weighted_point_sums",
        "seismoflux.background.grid.EqualAreaGridFamily.at",
        "seismoflux.background.pipeline_etas._KDEBackgroundDensity.density_many",
        "seismoflux.background.poisson.GaussianMixtureFamily.raw_densities",
    } <= set(closure["project_class_methods_required_exact"])
    assert {
        "seismoflux.background.grid.EqualAreaGrid.cell_ids",
        "seismoflux.background.grid.GridCell.id",
        "seismoflux.background.grid.GridSpec.cell_size_mm",
        "seismoflux.background.grid.GridConvergenceDiagnostics.passed",
        "seismoflux.background.grid.ThreeGridConvergenceGateEvidence.comparisons",
        "seismoflux.background.grid.ThreeGridConvergenceGateEvidence.passed",
        "seismoflux.background.pipeline_etas.ETASGridGateEvidence.passed",
        "seismoflux.background.pipeline_etas.ETASGridGateEvidence.failure_reasons",
        "seismoflux.background.pipeline_etas.ETASGridGateEvidence.numerical_evidence_id",
    } <= set(closure["project_properties_required_exact"])

    scipy = closure["scipy_dependencies_exact"]
    assert scipy["class_and_constructor"] == "scipy.spatial.cKDTree"
    assert scipy["instance_method"] == "scipy.spatial.cKDTree.query_ball_point"
    assert scipy["exact_edges"] == {
        "seismoflux.background.etas_fit.PointAreaQuadrature.__post_init__": [
            "scipy.spatial.cKDTree"
        ],
        "seismoflux.background.etas_fit._query_ball_indices": [
            "scipy.spatial.cKDTree.query_ball_point"
        ],
    }

    shapely_contract = closure["shapely_dependencies_exact"]
    assert shapely_contract["public_class_binding"] == "shapely.geometry.Point"
    assert shapely_contract["defining_class_qualified_name"] == "shapely.geometry.point.Point"
    assert shapely_contract["properties"] == [
        "shapely.geometry.point.Point.x",
        "shapely.geometry.point.Point.y",
    ]
    assert shapely_contract["instance_methods"] == ["shapely.geometry.base.BaseGeometry.equals"]
    assert shapely_contract["public_alias_bindings_exact"] == {
        "shapely.geometry.Point": "shapely.geometry.point.Point",
        "shapely.points": "shapely.creation.points",
        "shapely.get_x": "shapely._geometry.get_x",
        "shapely.get_y": "shapely._geometry.get_y",
        "shapely.equals": "shapely.predicates.equals",
    }
    warmup_ids = set(shapely_contract["synthetic_runtime_warmup_dependency_record_ids_exact"])
    three_grid_ids = set(shapely_contract["three_grid_runtime_dependency_record_ids_exact"])
    assert {
        "shapely.geometry.point.Point@direct_binding",
        "shapely.geometry.point.Point.__new__@direct_descriptor",
        "numpy.array@direct_binding",
        "numpy.ndarray.squeeze@direct_descriptor",
        "numpy.ndarray.ndim@direct_descriptor",
        "numpy.issubdtype@direct_binding",
        "numpy.number@direct_binding",
        "shapely.creation.points@deprecation_wrapper",
        "shapely.creation.points@multithreading_wrapper",
        "numpy.ndarray@direct_binding",
        "numpy.ndarray.dtype@direct_descriptor",
        "shapely.creation.points@wrapped_python_function",
        "numpy.intc@direct_binding",
        "shapely.lib.points@native_numpy_ufunc",
    } <= warmup_ids
    assert {
        "numpy.array@direct_binding",
        "numpy.ndarray@direct_binding",
        "shapely._geometry.get_x@multithreading_wrapper",
        "shapely._geometry.get_x@wrapped_python_function",
        "shapely.lib.get_x@native_numpy_ufunc",
        "shapely._geometry.get_y@multithreading_wrapper",
        "shapely._geometry.get_y@wrapped_python_function",
        "shapely.lib.get_y@native_numpy_ufunc",
        "shapely.predicates.equals@multithreading_wrapper",
        "shapely.predicates.equals@wrapped_python_function",
        "shapely.lib.equals@native_numpy_ufunc",
        "numpy.generic.ndim@direct_descriptor",
        "numpy.generic.item@direct_descriptor",
    } <= three_grid_ids
    constructor_only = warmup_ids - three_grid_ids
    assert "shapely.geometry.point.Point@direct_binding" in constructor_only
    assert "shapely.lib.points@native_numpy_ufunc" in constructor_only
    assert not any("Point.__new__" in record_id for record_id in three_grid_ids)
    assert shapely_contract[
        "point_constructor_dependency_record_ids_except_shared_numpy_array_and_ndarray_bindings_must_be_disjoint_from_three_grid_runtime_dependency_record_ids"
    ]
    assert shapely_contract[
        "three_grid_runtime_dependency_record_ids_must_equal_exact_intersection_of_synthetic_warmup_and_three_grid_lists"
    ]
    layered_edges = shapely_contract["layered_dependency_record_edges_exact"]
    assert layered_edges["seismoflux.background.adapters.point_area_quadrature_from_grid"] == [
        "shapely.geometry.point.Point.x@direct_descriptor",
        "shapely.geometry.point.Point.y@direct_descriptor",
    ]
    assert (
        "shapely.geometry.point.Point@direct_binding"
        in layered_edges[
            "seismoflux.background.etas_numerical_repair_evidence."
            "_run_fixed_optimizer_runtime_warmup"
        ]
    )
    assert layered_edges["shapely.geometry.point.Point.__new__@direct_descriptor"] == [
        "numpy.array@direct_binding",
        "numpy.ndarray.squeeze@direct_descriptor",
        "numpy.ndarray.ndim@direct_descriptor",
        "numpy.issubdtype@direct_binding",
        "numpy.ndarray.dtype@direct_descriptor",
        "numpy.number@direct_binding",
        "shapely.creation.points@deprecation_wrapper",
        "shapely.geometry.point.Point@direct_binding",
    ]
    assert layered_edges["shapely._geometry.get_x@multithreading_wrapper"] == [
        "numpy.ndarray@direct_binding",
        "shapely._geometry.get_x@wrapped_python_function",
    ]
    assert layered_edges["shapely._geometry.get_x@wrapped_python_function"] == [
        "shapely.lib.get_x@native_numpy_ufunc"
    ]
    assert layered_edges["shapely._geometry.get_y@multithreading_wrapper"] == [
        "numpy.ndarray@direct_binding",
        "shapely._geometry.get_y@wrapped_python_function",
    ]
    assert layered_edges["shapely.predicates.equals@multithreading_wrapper"] == [
        "numpy.ndarray@direct_binding",
        "shapely.predicates.equals@wrapped_python_function",
    ]
    assert layered_edges["shapely.predicates.equals@wrapped_python_function"] == [
        "shapely.lib.equals@native_numpy_ufunc"
    ]
    assert layered_edges["shapely.geometry.base._maybe_unpack@direct_binding"] == [
        "numpy.generic.ndim@direct_descriptor",
        "numpy.generic.item@direct_descriptor",
    ]

    assert shapely.geometry.Point is shapely_point.Point
    assert shapely.points is shapely_creation.points
    assert shapely.get_x is shapely_geometry.get_x
    assert shapely.get_y is shapely_geometry.get_y
    assert shapely.equals is shapely_predicates.equals

    synthetic_left = Point(1.25, -0.5)
    synthetic_right = Point(1.25, -0.5)
    assert synthetic_left.x.hex() == "0x1.4000000000000p+0"
    assert synthetic_left.y.hex() == "-0x1.0000000000000p-1"
    assert synthetic_left.equals(synthetic_right) is True
    assert Point.x.fget is not None
    assert Point.y.fget is not None
    assert Point.x.fget.__globals__["shapely"].get_x is shapely_geometry.get_x
    assert Point.y.fget.__globals__["shapely"].get_y is shapely_geometry.get_y
    assert Point.equals is shapely_base.BaseGeometry.equals
    assert Point.equals.__globals__["shapely"].equals is shapely_predicates.equals
    assert Point.equals.__globals__["_maybe_unpack"] is shapely_base._maybe_unpack
    raw_equals = shapely_predicates.equals(synthetic_left, synthetic_right)
    assert isinstance(raw_equals, np.generic)
    assert raw_equals.ndim == 0
    assert raw_equals.item() is True

    points_outer = cast(Any, shapely_creation.points)
    points_multithreading = cast(Any, points_outer.__wrapped__)
    points_original = cast(Any, points_multithreading.__wrapped__)
    assert points_outer is not points_multithreading
    assert points_multithreading is not points_original
    points_outer_cells = dict(
        zip(points_outer.__code__.co_freevars, points_outer.__closure__, strict=True)
    )
    points_multithreading_cells = dict(
        zip(
            points_multithreading.__code__.co_freevars,
            points_multithreading.__closure__,
            strict=True,
        )
    )
    assert points_outer_cells["func"].cell_contents is points_multithreading
    assert points_multithreading_cells["func"].cell_contents is points_original
    assert points_outer_cells["category"].cell_contents is DeprecationWarning
    assert points_outer_cells["warn_from"].cell_contents == 3
    make_msg = points_outer_cells["make_msg"].cell_contents
    assert type(make_msg).__module__ == "functools"
    assert type(make_msg).__qualname__ == "_lru_cache_wrapper"
    assert make_msg.__wrapped__.__module__ == "shapely.decorators"
    assert make_msg.__wrapped__.__qualname__.endswith("<locals>.make_msg")
    assert {"array", "ndim", "issubdtype", "dtype", "number", "points", "Point"} <= set(
        Point.__new__.__code__.co_names
    )
    assert Point.__new__.__globals__["Point"] is Point
    assert Point.__new__.__globals__["np"].ndarray is np.ndarray
    assert {"ndarray", "dtype", "flags", "writeable"} <= _recursive_code_names(
        points_multithreading.__code__
    )
    assert points_original.__globals__["lib"].points is shapely_lib.points

    for wrapper, native_name in (
        (cast(Any, shapely_geometry.get_x), "get_x"),
        (cast(Any, shapely_geometry.get_y), "get_y"),
        (cast(Any, shapely_predicates.equals), "equals"),
    ):
        original = cast(Any, wrapper.__wrapped__)
        closure_cells = dict(zip(wrapper.__code__.co_freevars, wrapper.__closure__, strict=True))
        assert closure_cells["func"].cell_contents is original
        assert {"ndarray", "dtype", "flags", "writeable"} <= _recursive_code_names(wrapper.__code__)
        assert wrapper.__globals__["np"].ndarray is np.ndarray
        assert original.__globals__["lib"].__dict__[native_name] is getattr(
            shapely_lib, native_name
        )

    native_expectations = {
        "points": {
            "nin": 2,
            "nout": 1,
            "nargs": 3,
            "ntypes": 1,
            "types": ["di->O"],
            "identity": None,
            "signature": "(d),()->()",
        },
        "get_x": {
            "nin": 1,
            "nout": 1,
            "nargs": 2,
            "ntypes": 1,
            "types": ["O->d"],
            "identity": None,
            "signature": None,
        },
        "get_y": {
            "nin": 1,
            "nout": 1,
            "nargs": 2,
            "ntypes": 1,
            "types": ["O->d"],
            "identity": None,
            "signature": None,
        },
        "equals": {
            "nin": 2,
            "nout": 1,
            "nargs": 3,
            "ntypes": 1,
            "types": ["OO->?"],
            "identity": None,
            "signature": None,
        },
    }
    for native_name, expected in native_expectations.items():
        native = getattr(shapely_lib, native_name)
        assert isinstance(native, np.ufunc)
        assert native.__name__ == native_name
        for field, value in expected.items():
            assert getattr(native, field) == value

    shapely_distribution = distribution("shapely")
    shapely_extension_path = Path(shapely_lib.__file__).resolve()
    matching_record_rows = [
        file
        for file in shapely_distribution.files or []
        if Path(str(shapely_distribution.locate_file(file))).resolve() == shapely_extension_path
    ]
    assert len(matching_record_rows) == 1
    shapely_extension_record = matching_record_rows[0]
    assert shapely_extension_record.hash is not None
    assert shapely_extension_record.hash.mode == "sha256"
    encoded_digest = shapely_extension_record.hash.value
    padding = "=" * (-len(encoded_digest) % 4)
    assert base64.urlsafe_b64decode(encoded_digest + padding).hex() == _sha256(
        shapely_extension_path
    )
    assert shapely_extension_record.size == shapely_extension_path.stat().st_size
    assert isinstance(Point.x, property)
    assert isinstance(Point.y, property)
    assert shapely_contract[
        "every_native_backed_call_requires_complete_shapely_RECORD_and_GEOS_loaded_image_map_match"
    ]
    memberships = shapely_contract["exact_closure_memberships_by_dependency_record_id"]
    assert set(memberships) == warmup_ids | three_grid_ids
    for record_id in constructor_only:
        assert memberships[record_id] == ["synthetic_runtime_warmup_closure"]
    for record_id in warmup_ids & three_grid_ids:
        assert memberships[record_id] == [
            "synthetic_runtime_warmup_closure",
            "three_grid_runtime_closure",
        ]
    assert closure[
        "runtime_callable_dependency_map_closure_membership_for_every_three_grid_reachable_record_must_include_three_grid_runtime_closure"
    ]
    assert closure[
        "synthetic_warmup_only_records_must_include_synthetic_runtime_warmup_closure_and_must_not_claim_three_grid_runtime_closure"
    ]
    assert closure[
        "no_reachable_record_may_be_missing_and_no_unreachable_record_may_claim_three_grid_runtime_closure_membership"
    ]

    runtime_seal = protocol["qualification_execution_seal"]["optimizer_runtime_code_seal"]
    assert runtime_seal["expected_shapely_distribution_version_from_uv_lock"] == "2.1.2"
    assert runtime_seal["complete_installed_distribution_RECORD_validation"]["distributions"] == [
        "numpy",
        "scipy",
        "shapely",
    ]
    required_runtime_fields = {
        "shapely_distribution_name_and_version",
        "shapely_dist_info_RECORD_sha256",
        "shapely_complete_installed_distribution_verification_map",
        "shapely_complete_installed_distribution_verification_map_sha256",
        "shapely_geos_loaded_native_dependency_file_sha256_and_size_map",
        "shapely_geos_loaded_native_dependency_file_sha256_and_size_map_sha256",
        "runtime_callable_dependency_map",
        "runtime_callable_dependency_map_sha256",
        "three_grid_runtime_dependency_closure_sha256",
    }
    assert required_runtime_fields <= set(runtime_seal["fields_exact"])
    public_runtime = protocol["outputs"]["public_qualification_seal_schema"]
    assert runtime_seal["fields_exact"] == public_runtime["runtime_fields_exact"]
    runtime_baseline = protocol["repair_code_scope_from_protocol_tag"][
        "repair_code_tag_prerequisite_public_artifacts"
    ]["optimizer_runtime_baseline"]
    full_hash_pairs = {
        "numpy_complete_installed_distribution_verification_map": (
            "numpy_complete_installed_distribution_verification_map_sha256"
        ),
        "scipy_complete_installed_distribution_verification_map": (
            "scipy_complete_installed_distribution_verification_map_sha256"
        ),
        "shapely_complete_installed_distribution_verification_map": (
            "shapely_complete_installed_distribution_verification_map_sha256"
        ),
        "shapely_geos_loaded_native_dependency_file_sha256_and_size_map": (
            "shapely_geos_loaded_native_dependency_file_sha256_and_size_map_sha256"
        ),
        "numpy_runtime_config_safe_projection": "numpy_runtime_config_canonical_sha256",
        "runtime_callable_dependency_map": "runtime_callable_dependency_map_sha256",
    }
    baseline_fields = set(runtime_baseline["fields_exact"])
    runtime_fields = set(runtime_seal["fields_exact"])
    for full_field, hash_field in full_hash_pairs.items():
        assert {full_field, hash_field} <= baseline_fields
        assert {full_field, hash_field} <= runtime_fields
    crosswalks = runtime_seal["code_tag_baseline_equality"][
        "exact_full_object_and_sibling_hash_crosswalks"
    ]
    assert set(crosswalks) == set(full_hash_pairs)
    for full_field, hash_field in full_hash_pairs.items():
        assert crosswalks[full_field]["baseline_full_field"] == full_field
        assert crosswalks[full_field]["runtime_full_field"] == full_field
        assert crosswalks[full_field]["baseline_sha256_field"] == hash_field
        assert crosswalks[full_field]["runtime_sha256_field"] == hash_field
    assert runtime_seal["code_tag_baseline_equality"][
        "every_baseline_full_object_must_equal_runtime_full_object_byte_for_byte_and_each_sibling_sha256_must_equal_recomputed_hash_of_both"
    ]
    assert public_runtime["opening_runtime_baseline_identity_crosswalk"][
        "all_three_pairs_must_be_equal_and_recompute_from_the_exact_remote_repair_code_tag_baseline_blob"
    ]
    assert public_runtime["runtime_full_object_and_sibling_hash_crosswalk_ref"] == (
        "qualification_execution_seal.optimizer_runtime_code_seal.code_tag_baseline_equality."
        "exact_full_object_and_sibling_hash_crosswalks"
    )
    assert (
        public_runtime["runtime_nested_field_schema_refs"]["shapely_distribution_name_and_version"]
        == "outputs.canonical_nested_schemas.distribution_name_and_version"
    )
    assert public_runtime["runtime_nested_field_schema_refs"][
        "shapely_geos_loaded_native_dependency_file_sha256_and_size_map"
    ] == (
        "outputs.canonical_nested_schemas."
        "shapely_geos_loaded_native_dependency_file_sha256_and_size_map"
    )


def test_content_identities_and_typed_adapter_contract_are_closed() -> None:
    protocol = _load_yaml(PROTOCOL_PATH)
    identities = protocol["content_addressing"]["identities"]
    assert set(identities) == {
        "optimizer_runtime_code_seal_sha256",
        "fit_etas_call_opening_receipt_sha256",
        "optimizer_invocation_receipt_sha256",
        "fit_etas_call_closing_receipt_sha256",
        "fit_attempt_snapshot_sha256",
        "three_grid_gate_evidence_sha256",
        "etas_parameter_snapshot_sha256",
        "etas_parameter_set_sha256",
        "etas_numerical_negative_evidence_sha256",
        "etas_numerical_qualification_evidence_sha256",
        "etas_artifact_sha256",
        "frozen_etas_comparator_receipt_sha256",
        "etas_issue_forecast_query_nodes_sha256",
        "etas_issue_simulation_context_sha256",
        "etas_issue_simulation_batch_payload_sha256",
        "etas_issue_simulation_catalog_receipt_sha256",
        "etas_issue_forecast_input_sha256",
        "etas_issue_forecast_projection_receipt_sha256",
        "etas_retrospective_likelihood_input_sha256",
        "etas_causal_parent_receipt_sha256",
        "etas_retrospective_likelihood_terms_sha256",
    }
    adapter_public_schema = protocol["outputs"]["adapter_public_artifact_schemas"]
    assert (
        identities["etas_artifact_sha256"]["includes"]
        == adapter_public_schema["artifact_manifest_fields_exact"][:-1]
    )
    assert "five_etas_parameter_snapshot_sha256" in identities["etas_artifact_sha256"]["includes"]
    assert "five_parameter_snapshot_sha256" not in identities["etas_artifact_sha256"]["includes"]
    assert "model_spec_by_snapshot" in identities["etas_artifact_sha256"]["includes"]
    assert "model_spec" not in identities["etas_artifact_sha256"]["includes"]
    assert (
        identities["frozen_etas_comparator_receipt_sha256"]["includes"]
        == (adapter_public_schema["global_receipt_fields_exact"][:-1])
    )
    invocation_protocol = protocol["optimizer_invocation_receipt_protocol"]
    assert (
        invocation_protocol["fit_etas_call_opening_receipt_identity_fields_exact"]
        == (invocation_protocol["fit_etas_call_opening_receipt_fields"][:-1])
    )
    assert (
        invocation_protocol["optimizer_invocation_receipt_identity_fields_exact"]
        == (invocation_protocol["optimizer_invocation_receipt_fields"][:-1])
    )
    assert (
        invocation_protocol["fit_etas_call_closing_receipt_identity_fields_exact"]
        == (invocation_protocol["fit_etas_call_closing_receipt_fields"][:-1])
    )
    fit_attempt_fields = invocation_protocol["fit_attempt_snapshot_payload_fields_exact"]
    assert identities["fit_attempt_snapshot_sha256"]["includes_ref"] == (
        "optimizer_invocation_receipt_protocol.fit_attempt_snapshot_payload_fields_exact"
    )
    assert {
        "fit_etas_call_opening_receipt_sha256",
        "ordered_five_optimizer_invocation_receipt_sha256",
        "fit_etas_call_closing_receipt_sha256",
        "ordered_five_diagnostic_row_sha256",
        "three_grid_gate_evidence_sha256_or_null",
        "snapshot_gate_result_sha256",
    } <= set(fit_attempt_fields)
    assert "aki_b_beta_two_bin_masses" in identities["etas_parameter_snapshot_sha256"]["includes"]
    assert (
        identities["etas_parameter_set_sha256"]["includes"]
        == protocol["outputs"]["public_parameter_registry_schema"][
            "parameter_set_identity_payload_fields_exact"
        ]
    )
    qualification_identity = identities["etas_numerical_qualification_evidence_sha256"]
    qualification_manifest_schema = protocol["outputs"]["public_qualification_manifest_schema"]
    qualification_projection = qualification_manifest_schema[
        "qualification_evidence_projection_fields_exact"
    ]
    assert qualification_projection == qualification_manifest_schema["top_level_fields_exact"][:-2]
    assert {
        "opening_execution_seal_sha256",
        "qualification_input_seal_sha256",
        "public_source_access_receipt",
        "parent_replay_membership_identity_sha256_by_snapshot",
        "fit_attempt_snapshot_sha256_by_snapshot",
        "diagnostic_rows",
        "snapshot_gate_results",
    } <= set(qualification_projection)
    assert qualification_identity["identity_projection_fields_exact_ref"] == (
        "outputs.public_qualification_manifest_schema.qualification_evidence_projection_fields_exact"
    )
    assert qualification_identity["branch_invariants"] == {
        "evaluable": {
            "etas_parameter_set_sha256": "required",
            "etas_numerical_negative_evidence_sha256": None,
        },
        "not_evaluable": {
            "etas_parameter_set_sha256": None,
            "etas_numerical_negative_evidence_sha256": "required",
        },
    }
    assert (
        identities["etas_numerical_negative_evidence_sha256"][
            "parameter_snapshot_or_set_artifacts_allowed"
        ]
        is False
    )
    assert (
        identities["etas_numerical_negative_evidence_sha256"]["includes"]
        == (qualification_manifest_schema["numerical_negative_evidence_fields_exact"][1:])
    )
    assert (
        identities["frozen_etas_comparator_receipt_sha256"]["selected_role_field_allowed"] is False
    )
    simulation_context = identities["etas_issue_simulation_context_sha256"]
    assert set(simulation_context["explicitly_excludes"]) == {
        "grid_family",
        "horizons_days",
        "query_nodes",
        "magnitude_output_bins",
        "stage4_targets_or_results",
    }
    assert {
        "etas_issue_simulation_context_sha256",
        "etas_issue_simulation_batch_payload_sha256",
        "etas_issue_simulation_catalog_receipt_sha256",
        "grid_family_sha256",
        "horizons_days",
        "etas_issue_forecast_query_nodes_sha256",
    } <= set(identities["etas_issue_forecast_input_sha256"]["includes"])
    projection_receipt = identities["etas_issue_forecast_projection_receipt_sha256"]
    assert projection_receipt["circular_hash_reference_forbidden"] is True
    simulation_receipt_identity = identities["etas_issue_simulation_catalog_receipt_sha256"]

    contract = protocol["adapter_contract"]
    interfaces = contract["typed_interfaces"]
    assert set(interfaces) == {
        "ETASKnownParentEvent",
        "ETASRetrospectiveWindow",
        "ETASRetrospectiveTargetEvent",
        "ETASRetrospectiveEventQueryNode",
        "ETASBaselineQueryNodes",
        "ETASBaselineMeasure",
        "ETASRetrospectiveLikelihoodInput",
        "ETASRetrospectiveEventTerm",
        "ETASRetrospectiveLikelihoodTerms",
        "ETASIssueSimulationContext",
        "ETASIssueSimulationInput",
        "ETASFuturePropagationEvent",
        "ETASIssueSimulationBatch",
        "ETASIssueSimulationOutput",
        "ETASIssueForecastQueryNodes",
        "ETASIssueForecastInput",
        "ETASIssueForecastMeasure",
        "ETASIssueForecastField",
        "ETASIssueSimulationCatalogReceipt",
        "ETASIssueForecastReplicateNodeDiagnostic",
        "ETASIssueForecastProjectionReceipt",
    }
    assert interfaces["ETASIssueSimulationInput"]["target_or_event_result_fields_allowed"] is False
    assert interfaces["ETASIssueForecastInput"]["target_or_event_result_fields_allowed"] is False
    assert interfaces["ETASKnownParentEvent"]["field_aliases_allowed"] is False
    assert interfaces["ETASKnownParentEvent"]["fields"] == [
        "physical_event_id",
        "origin_time_utc",
        "available_time_utc",
        "x_y_hex",
        "magnitude_hex",
        "inside_supported_domain",
        "inside_study_area",
        "inside_parent_domain",
        "parent_role",
    ]
    assert interfaces["ETASKnownParentEvent"]["parent_role_values"] == [
        "supported",
        "true_external_buffer",
        "unsupported_conditional",
    ]
    assert interfaces["ETASKnownParentEvent"]["missing_extra_unknown_or_alias_field_rejected"]
    assert interfaces["ETASRetrospectiveTargetEvent"][
        "missing_extra_unknown_or_alias_field_rejected"
    ]
    assert interfaces["ETASRetrospectiveLikelihoodInput"][
        "ordered_targets_use_ETASRetrospectiveTargetEvent_schema"
    ]
    assert interfaces["ETASRetrospectiveWindow"]["fields"] == [
        "window_start_utc",
        "window_end_utc",
    ]
    assert interfaces["ETASRetrospectiveLikelihoodInput"]["nested_schema_refs"] == {
        "window": "adapter_contract.typed_interfaces.ETASRetrospectiveWindow",
        "ordered_targets": "adapter_contract.typed_interfaces.ETASRetrospectiveTargetEvent",
        "ordered_event_query_nodes": (
            "adapter_contract.typed_interfaces.ETASRetrospectiveEventQueryNode"
        ),
        "ordered_known_parent_events": "adapter_contract.typed_interfaces.ETASKnownParentEvent",
        "baseline_query_nodes": "adapter_contract.typed_interfaces.ETASBaselineQueryNodes",
    }
    assert (
        identities["etas_retrospective_likelihood_input_sha256"]["includes"]
        == interfaces["ETASRetrospectiveLikelihoodInput"]["fields"][:-1]
    )
    assert interfaces["ETASRetrospectiveLikelihoodInput"][
        "selected_role_snapshot_and_parameter_sha_must_match_global_receipt_mapping"
    ]
    assert interfaces["ETASRetrospectiveLikelihoodInput"]["window_interval"] == (
        "(window_start_utc,window_end_utc]"
    )
    assert interfaces["ETASRetrospectiveTargetEvent"]["target_to_cell_mapping"] == (
        "exact_covers_then_row_column_cell_id_tie_break_from_frozen_stage4_R2_25km_grid"
    )
    assert interfaces["ETASRetrospectiveTargetEvent"][
        "exact_coordinates_used_only_for_frozen_cell_mapping_and_never_for_query_refinement_or_point_intensity"
    ]
    assert interfaces["ETASRetrospectiveEventQueryNode"][
        "target_micro_move_within_same_cell_must_leave_query_node_and_intensity_bytes_unchanged"
    ]
    assert interfaces["ETASRetrospectiveEventQueryNode"][
        "target_query_node_crosswalk_fields_exact"
    ] == [
        "target.event_query_node_id_equals_node.event_query_node_id",
        "target.physical_event_id_equals_node.physical_event_id",
        "target.frozen_25km_grid_id_equals_node.frozen_25km_grid_id",
        "target.frozen_25km_cell_id_equals_node.frozen_25km_cell_id",
        "target.frozen_25km_row_equals_node.frozen_25km_row",
        "target.frozen_25km_column_equals_node.frozen_25km_column",
        "target.origin_time_utc_equals_node.event_time_utc",
        "target.magnitude_bin_equals_node.magnitude_bin",
    ]
    assert interfaces["ETASRetrospectiveLikelihoodInput"]["known_parent_collection_semantics"] == (
        "complete_authorized_M4_plus_causal_catalog_through_window_end_including_scored_"
        "targets_and_unscored_M4_to_lt_M5_events"
    )
    assert (
        interfaces["ETASIssueSimulationInput"][
            "every_parent_available_time_lte_knowledge_cutoff_required"
        ]
        is True
    )
    assert interfaces["ETASIssueSimulationInput"]["every_parent_origin_lte_issue_time_required"]
    assert interfaces["ETASIssueSimulationBatch"]["hidden_global_cache_allowed"] is False
    assert (
        identities["etas_issue_simulation_batch_payload_sha256"]["includes"]
        == interfaces["ETASIssueSimulationBatch"]["fields"][:-1]
    )
    simulation_batch = interfaces["ETASIssueSimulationBatch"]
    assert simulation_batch["replicate_catalog_fields_exact"] == [
        "replicate_index",
        "ordered_future_events",
        "replicate_catalog_sha256",
    ]
    assert simulation_batch[
        "every_ordered_replicate_catalog_sha256_must_recompute_from_and_equal_same_index_catalog_own_sha"
    ]
    assert (
        interfaces["ETASIssueSimulationBatch"]["public_serialization_or_event_row_exposure_allowed"]
        is False
    )
    assert interfaces["ETASIssueSimulationContext"][
        "ordered_known_parent_events_use_ETASKnownParentEvent_schema"
    ]
    assert interfaces["ETASIssueSimulationBatch"][
        "context_object_contains_known_parent_events_once_and_every_replicate_catalog_contains_future_events_only"
    ]
    future_event = interfaces["ETASFuturePropagationEvent"]
    assert future_event["output_eligible_iff_inside_supported_domain_and_magnitude_gte_5"]
    assert set(future_event["domain_role_truth_table"]) == {
        "supported",
        "true_external_buffer",
        "eligible_unsupported",
    }
    assert future_event["replicate_index_must_equal_containing_replicate_catalog_replicate_index"]
    assert future_event[
        "origin_time_must_be_strictly_after_context_issue_time_and_lte_context_issue_time_plus_365_days"
    ]
    assert interfaces["ETASIssueSimulationBatch"][
        "projection_node_intensity_uses_context_known_parents_plus_same_replicate_future_parents_strictly_before_node_time"
    ]
    assert interfaces["ETASIssueForecastInput"]["projection_may_not_resimulate_or_use_hidden_cache"]
    assert interfaces["ETASRetrospectiveLikelihoodTerms"]["alarm_ranking_consumer_allowed"] is False
    assert (
        interfaces["ETASIssueForecastField"]["retrospective_likelihood_consumer_allowed"] is False
    )
    assert (
        interfaces["ETASIssueForecastField"]["permutation_duplicate_or_missing_cell_rejected"]
        is True
    )
    assert (
        "ordered_query_node_measure_payload_sha256_excluding_projection_receipt"
        in (interfaces["ETASIssueForecastField"]["fields"])
    )
    assert (
        "ordered_query_node_measure_sha256" not in (interfaces["ETASIssueForecastField"]["fields"])
    )
    assert interfaces["ETASIssueForecastProjectionReceipt"]["acyclic_construction_order"] == [
        "build_and_hash_ordered_replicate_node_intensity_diagnostic_rows_with_no_projection_receipt_field",
        "hash_ordered_node_measure_rows_with_projection_receipt_field_omitted",
        "build_and_hash_ordered_cell_rows_with_projection_receipt_field_omitted_and_only_the_receipt_free_node_payload_sha",
        "hash_projection_receipt_from_input_catalog_receipt_adapter_replicate_diagnostic_and_both_receipt_free_payload_hashes",
        "fill_projection_receipt_sha_into_node_and_cell_rows_without_rehashing_payload_identities",
    ]
    assert (
        projection_receipt["includes"] == interfaces["ETASIssueForecastProjectionReceipt"]["fields"]
    )
    assert interfaces["ETASIssueForecastMeasure"]["variant_id_exact"] == (
        "etas_background_no_increment"
    )
    assert (
        interfaces["ETASIssueForecastMeasure"]["anomaly_factor_input_or_weighting_allowed"] is False
    )
    assert interfaces["ETASIssueForecastMeasure"][
        "outside_selected_support_positive_zero_fields_exact"
    ] == [
        "conditional_ground_intensity_mean_per_day_km2_hex",
        "conditional_ground_intensity_standard_error_hex",
        "conditional_bin_intensity_mean_per_day_km2_hex",
        "weighted_expected_count_hex",
    ]
    replicate_diagnostic = interfaces["ETASIssueForecastReplicateNodeDiagnostic"]
    assert replicate_diagnostic["outside_selected_support_positive_zero_measure_fields_exact"] == [
        "conditional_ground_intensity_per_day_km2_hex",
        "conditional_bin_intensity_per_day_km2_hex",
        "weighted_expected_count_hex",
    ]
    assert replicate_diagnostic[
        "replicate_index_query_node_id_inside_selected_support_and_magnitude_bin_mass_are_not_zeroed"
    ]
    assert (
        interfaces["ETASRetrospectiveLikelihoodTerms"][
            "exact_event_intensity_count_must_equal_ordered_target_count"
        ]
        is True
    )
    assert interfaces["ETASRetrospectiveLikelihoodTerms"]["terms_must_bind_complete_input_sha256"]
    assert (
        "ordered_baseline_node_measures" in interfaces["ETASRetrospectiveLikelihoodTerms"]["fields"]
    )
    assert (
        "node_level_baseline_measure"
        not in interfaces["ETASRetrospectiveLikelihoodTerms"]["fields"]
    )
    assert interfaces["ETASRetrospectiveLikelihoodTerms"][
        "ordered_baseline_node_measures_use_ETASBaselineMeasure_schema"
    ]
    assert (
        identities["etas_retrospective_likelihood_terms_sha256"]["includes"]
        == interfaces["ETASRetrospectiveLikelihoodTerms"]["fields"][:-1]
    )
    causal_parent_identity = identities["etas_causal_parent_receipt_sha256"]
    assert causal_parent_identity["parent_identity_map_item_fields_exact"] == [
        "query_node_id",
        "ordered_causal_parent_identity_sha256",
    ]
    assert causal_parent_identity[
        "target_parent_identity_map_exact_order_must_equal_input_ordered_event_query_nodes"
    ]
    retrospective_terms = interfaces["ETASRetrospectiveLikelihoodTerms"]
    assert retrospective_terms["ordered_target_identity_sha256_formula"] == (
        "sha256_seismoflux_canonical_json_v1_of_complete_input_ordered_targets"
    )
    assert retrospective_terms[
        "causal_parent_receipt_sha256_must_equal_recomputed_etas_causal_parent_receipt_sha256"
    ]
    assert set(interfaces["ETASRetrospectiveEventTerm"]["field_types"]) == set(
        interfaces["ETASRetrospectiveEventTerm"]["fields"]
    )
    assert (
        "etas_retrospective_likelihood_input_sha256"
        in interfaces["ETASRetrospectiveLikelihoodTerms"]["fields"]
    )
    catalog_receipt = interfaces["ETASIssueSimulationCatalogReceipt"]
    assert simulation_receipt_identity["includes"] == catalog_receipt["fields"]
    assert "branching_process_domain_and_marks" in catalog_receipt["fields"]
    assert "simulation_controls" in catalog_receipt["fields"]
    assert catalog_receipt["ordered_seed_context_digests_exact_count"] == 128
    assert catalog_receipt["ordered_seed_context_digest_item_fields_exact"] == [
        "replicate_index",
        "seed_context_digest_hex",
    ]
    assert catalog_receipt[
        "each_seed_digest_item_must_equal_same_index_replicate_diagnostic_seed_context_digest"
    ]
    assert catalog_receipt[
        "each_replicate_diagnostic_replicate_catalog_sha256_must_equal_same_index_batch_catalog_sha256"
    ]
    assert catalog_receipt["event_cap_status_fields_exact"] == [
        "maximum_events_per_replicate",
        "any_event_cap_hit",
        "ordered_hit_replicate_indices",
        "forecast_valid",
    ]
    assert catalog_receipt["forecast_valid_iff_any_event_cap_hit_false"]
    assert (
        identities["etas_issue_simulation_context_sha256"]["includes"]
        == interfaces["ETASIssueSimulationContext"]["identity_projection_fields_exact"]
    )
    assert (
        identities["etas_issue_forecast_input_sha256"]["includes"]
        == interfaces["ETASIssueForecastInput"]["identity_projection_fields_exact"]
    )
    forecast = contract["issue_forecast_definition"]
    assert forecast["all_future_descendant_generations_included"] is True
    assert forecast["simulation_replicates"] == 128
    assert forecast["sliced_horizons_days"] == [7, 30, 90, 180, 365]
    assert forecast["replicate_index_first"] == 0
    assert forecast["replicate_index_last_inclusive"] == 127
    assert (
        forecast[
            "one_simulated_catalog_per_role_issue_replicate_reused_across_all_horizons_grids_and_magnitude_bins"
        ]
        is True
    )
    domain = forecast["branching_process_domain_and_marks"]
    assert domain["M4_to_lt_M5_events_are_latent_propagating_not_output_events"] is True
    assert domain["every_nonabsorbed_future_event_with_magnitude_gte_4_is_a_propagating_parent"]
    assert domain["propagation_outer_boundary"] == (
        "absorbing_without_spatial_kernel_renormalization"
    )
    assert domain[
        "event_cap_counts_all_attempted_future_ground_events_including_absorbed_and_M4_to_lt_M5"
    ]
    assert domain["output_measure_domain"] == (
        "every_preregistered_full_grid_query_node_with_exact_positive_zero_outside_selected_snapshot_support"
    )
    assert domain["support_partial_cell_measure_uses_frozen_clipped_area_weight"]
    downstream = contract["downstream_dynamic_anomaly_composition_boundary"]
    assert downstream["owner"] == "future_stage4_candidate_pipeline_not_etas_adapter"
    assert downstream["weighting_must_be_per_query_node_before_cell_aggregation"]
    assert downstream[
        "pure_etas_artifact_global_receipt_simulation_and_projection_receipts_must_remain_unchanged"
    ]
    assert contract["magnitude_bins"]["learned_stage4_bin_rate_head_on_etas_track_allowed"] is False
    assert set(contract["separation_property_tests"]) == {
        "mutate_or_append_future_window_targets_leaves_issue_forecast_bytes_and_sha_unchanged",
        "issue_forecast_rejects_any_parent_available_time_after_knowledge_cutoff",
        "alarm_order_accepts_only_ETASIssueForecastField",
        "retrospective_terms_cannot_be_passed_as_alarm_field",
        "baseline_measure_retains_issue_cell_time_bin_node_identity",
        "issue_forecast_exactly_preserves_query_node_and_cell_order",
        "future_target_mutation_cannot_change_seed_context_or_simulation_receipt",
        "same_simulated_catalog_is_reused_across_horizons_grids_and_magnitude_bins",
        "pure_etas_adapter_rejects_anomaly_factor_input_and_preserves_no_increment_variant",
        "outside_support_nodes_and_cells_are_present_with_exact_positive_zero_values",
        "target_micro_move_within_same_frozen_25km_cell_leaves_retrospective_event_intensity_unchanged",
        "unscored_M4_to_lt_M5_window_event_changes_later_intensity_without_creating_event_term",
    }


def test_issue_simulation_output_has_exact_batch_receipt_crosswalk() -> None:
    protocol = _load_yaml(PROTOCOL_PATH)
    contract = protocol["adapter_contract"]
    interfaces = contract["typed_interfaces"]
    output = interfaces["ETASIssueSimulationOutput"]
    receipt = interfaces["ETASIssueSimulationCatalogReceipt"]
    crosswalk = output["batch_and_receipt_exact_crosswalk"]

    assert set(crosswalk) == {
        "batch_payload_sha256_equal_across",
        "catalog_receipt_sha256_equal_across",
        "direct_receipt_batch_field_equalities_exact",
        "global_receipt_sha256_equal_across",
        "selected_role_snapshot_parameter_must_match_global_receipt_role_mapping_exactly",
        "adapter_code_blob_sha256_equal_across",
        "environment_lock_sha256_equal_across",
        "branching_process_domain_and_marks_exact_crosswalk",
        "simulation_controls_exact_crosswalk",
        "ordered_seed_context_digests_must_have_exact_indices_zero_through_127_and_each_digest_must_recompute_from_frozen_seed_context_and_same_batch_context",
        "each_seed_digest_must_equal_same_index_replicate_diagnostic_seed_context_digest",
        "each_replicate_diagnostic_seed_entropy_must_equal_first_16_digest_bytes_as_unsigned_big_endian_decimal",
        "ordered_replicate_catalog_sha256_exact_crosswalk",
        "every_replicate_catalog_and_diagnostic_index_must_equal_zero_through_127_once_each_in_ascending_order",
        "event_cap_status_must_recompute_exactly_from_ordered_replicate_catalog_diagnostics_and_frozen_maximum",
        "event_cap_hit_requires_forecast_valid_false_and_forbids_any_forecast_projection",
        "any_missing_extra_alias_duplicate_order_context_issue_role_snapshot_parameter_batch_receipt_model_artifact_control_seed_catalog_or_environment_mismatch_action",
    }
    assert len(crosswalk["batch_payload_sha256_equal_across"]) == 4
    assert len(crosswalk["catalog_receipt_sha256_equal_across"]) == 2
    assert set(crosswalk["direct_receipt_batch_field_equalities_exact"]) == {
        "etas_issue_simulation_context_sha256",
        "issue_id",
        "selected_role",
        "selected_snapshot_id",
        "selected_parameter_snapshot_sha256",
    }
    for values in crosswalk["direct_receipt_batch_field_equalities_exact"].values():
        assert len(values) == 3
        assert any(value.startswith("issue_simulation_batch.") for value in values)
        assert any(value.startswith("etas_issue_simulation_catalog_receipt.") for value in values)

    marks = crosswalk["branching_process_domain_and_marks_exact_crosswalk"]
    assert set(marks) == set(receipt["branching_process_domain_and_marks_fields_exact"])
    assert len(marks["maximum_magnitude_hex"]) == 3
    assert len(marks["beta_hex"]) == 4
    assert len(marks["immigrant_density_artifact_sha256"]) == 2
    assert len(marks["propagation_domain_artifact_sha256"]) == 3
    assert marks["ground_magnitude_lower_hex"] == [
        "etas_issue_simulation_catalog_receipt.branching_process_domain_and_marks.ground_magnitude_lower_hex",
        "canonical_python_float64_hex_of_ETASFuturePropagationEvent_magnitude_range_inclusive_lower",
    ]

    controls = crosswalk["simulation_controls_exact_crosswalk"]
    assert set(controls) == set(receipt["simulation_controls_fields_exact"])
    assert receipt["simulation_controls_exact_values"] == {
        "simulation_replicates": 128,
        "longest_horizon_days": 365,
        "maximum_events_per_replicate": 100000,
        "bit_generator": "numpy.random.PCG64",
        "seed_namespace": "etas_issue_forecast",
    }
    assert controls["seed_namespace"] == [
        "etas_issue_simulation_catalog_receipt.simulation_controls.seed_namespace",
        "adapter_contract.issue_forecast_definition.seed_context_contract.namespace",
    ]
    assert (
        receipt["simulation_controls_exact_values"]["seed_namespace"]
        == contract["issue_forecast_definition"]["seed_context_contract"]["namespace"]
    )
    assert len(crosswalk["ordered_replicate_catalog_sha256_exact_crosswalk"]) == 3
    for required in (
        "selected_role_snapshot_parameter_must_match_global_receipt_role_mapping_exactly",
        "ordered_seed_context_digests_must_have_exact_indices_zero_through_127_and_each_digest_must_recompute_from_frozen_seed_context_and_same_batch_context",
        "each_seed_digest_must_equal_same_index_replicate_diagnostic_seed_context_digest",
        "each_replicate_diagnostic_seed_entropy_must_equal_first_16_digest_bytes_as_unsigned_big_endian_decimal",
        "every_replicate_catalog_and_diagnostic_index_must_equal_zero_through_127_once_each_in_ascending_order",
        "event_cap_status_must_recompute_exactly_from_ordered_replicate_catalog_diagnostics_and_frozen_maximum",
        "event_cap_hit_requires_forecast_valid_false_and_forbids_any_forecast_projection",
    ):
        assert crosswalk[required] is True
    assert (
        crosswalk[
            "any_missing_extra_alias_duplicate_order_context_issue_role_snapshot_parameter_batch_receipt_model_artifact_control_seed_catalog_or_environment_mismatch_action"
        ]
        == "reject_atomic_simulation_output_and_generate_no_projection"
    )


def test_qualification_execution_seals_require_clean_remote_frozen_code() -> None:
    protocol = _load_yaml(PROTOCOL_PATH)
    seal = protocol["qualification_execution_seal"]
    opening = seal["opening_seal"]
    assert opening[
        "created_before_any_new_qualification_attempt_source_open_stat_hash_query_or_bundle_inspection_after_protocol_freeze"
    ]
    repository = opening["repository_requirements"]
    for required in (
        "worktree_clean",
        "head_equals_repair_code_tag_commit",
        "named_upstream_exists",
        "upstream_commit_equals_head",
        "remote_repair_code_tag_resolves_to_head_and_is_verified",
        "protocol_tag_commit_and_remote_tag_verified",
        "protocol_package_paths_and_git_blob_oids_equal_protocol_tag_commit",
        "repair_code_diff_from_protocol_tag_matches_exact_allowlist_and_receipt",
        "remote_identity_and_tag_objects_verified_with_network",
    ):
        assert repository[required] is True
    assert repository["remote_repository_slug"] == "Justin-147/SeismoFlux"
    assert repository["upstream_branch"] == "origin/codex/stage2-etas-numerical-repair"
    assert opening["protocol_package_paths"] == [
        ".gitignore",
        "configs/background_etas_numerical_repair.yaml",
        "data/manifests/etas_numerical_repair_start_manifest.json",
        "docs/background_etas_numerical_repair_protocol.md",
        "docs/phase2_etas_numerical_repair_protocol_acceptance.md",
        "tests/unit/test_background_etas_numerical_repair_protocol.py",
    ]
    assert (
        "tests/unit/test_stage4_anomaly_increment_runtime.py"
        not in opening["protocol_package_paths"]
    )
    assert (
        "docs/restart_handoff_2026-07-19_stage2_etas_repair_protocol.md"
        not in opening["protocol_package_paths"]
    )
    assert opening["every_protocol_package_path_must_exist_as_regular_file_before_protocol_commit"]
    assert opening[
        "every_protocol_package_path_must_resolve_to_git_blob_in_protocol_commit_and_remote_tag"
    ]
    for package_path in opening["protocol_package_paths"]:
        assert Path(package_path).is_file()
    assert PROTOCOL_ACCEPTANCE_PATH in {
        Path(package_path) for package_path in opening["protocol_package_paths"]
    }
    runtime = seal["optimizer_runtime_code_seal"]
    assert runtime["expected_numpy_distribution_version_from_uv_lock"] == "2.4.6"
    assert runtime["expected_scipy_distribution_version_from_uv_lock"] == "1.17.1"
    assert runtime["ordinary_system_python_or_any_other_distribution_version_must_fail_closed"]
    assert runtime["runtime_module_global_minimize_must_be_same_object_as_scipy_optimize_minimize"]
    assert runtime["absolute_paths_hostnames_usernames_or_environment_secrets_allowed"] is False
    assert (
        runtime["fields_exact"]
        == _load_yaml(PROTOCOL_PATH)["outputs"]["public_qualification_seal_schema"][
            "runtime_fields_exact"
        ]
    )
    assert (
        protocol["content_addressing"]["identities"]["optimizer_runtime_code_seal_sha256"][
            "includes"
        ]
        == runtime["fields_exact"][:-1]
    )
    assert {
        "numpy_runtime_config_safe_projection",
        "runtime_callable_dependency_map_sha256",
        "python_runtime_file_sha256_and_size_map",
        "attempt_id",
        "checked_at_utc",
    } <= set(runtime["fields_exact"])
    live_runtime_rechecks = runtime["live_runtime_module_and_native_image_reenumeration"]
    assert live_runtime_rechecks["checkpoints"] == [
        "before_each_snapshot_fit",
        "after_each_snapshot_fit",
        "before_qualification_closing_seal",
    ]
    assert live_runtime_rechecks[
        "canonical_runtime_file_map_must_equal_code_tag_baseline_and_optimizer_runtime_code_seal_byte_for_byte"
    ]
    local_acceptance = seal["local_restricted_input_acceptance_receipt"]
    assert local_acceptance["every_local_restricted_input_must_exist_and_match_frozen_sha256"]
    assert local_acceptance["canonical_local_restricted_input_acceptance_receipt_sha256_required"]
    qualification = seal["qualification_input_seal"]
    assert qualification[
        "created_after_fit_bundle_and_source_access_ledger_are_sealed_before_any_fit"
    ]
    assert {
        "public_source_access_receipt_sha256",
        "local_source_access_ledger_content_sha256",
        "fit_input_manifest_file_and_content_sha256",
        "parent_replay_membership_identity_sha256_by_snapshot",
        "parent_replay_scientific_fit_input_sha256_by_snapshot",
        "local_restricted_input_acceptance_receipt_sha256",
        "optimizer_runtime_code_seal_sha256",
    } <= set(qualification["includes"])
    assert qualification["unchanged_rechecks"] == [
        "before_each_snapshot_fit",
        "after_each_snapshot_fit",
        "before_qualification_result_finalization",
        "before_public_artifact_materialization",
    ]
    interrupted = seal["interrupted_attempts"]
    assert interrupted["retry_may_not_replace_or_delete_prior_diagnostic_rows"] is True
    assert interrupted["selecting_better_retry_result_forbidden"] is True
    assert interrupted["qualification_uses_first_complete_protocol_valid_attempt_only"] is True
    failure_receipt = interrupted["invalid_execution_local_failure_receipt"]
    assert failure_receipt[
        "every_completed_or_observation_list_is_append_only_and_preserves_actual_logical_order"
    ]
    assert failure_receipt[
        "completed_optimizer_and_diagnostic_lists_may_be_sparse_ordered_subsequences_after_a_caught_start_failure"
    ]
    assert failure_receipt["completed_receipt_list_cardinalities"] == {
        "ordered_completed_fit_call_opening_receipt_sha256": "zero_to_five",
        "ordered_completed_optimizer_invocation_receipt_sha256": "zero_to_twenty_five",
        "ordered_completed_fit_call_closing_receipt_sha256": "zero_to_five",
        "ordered_completed_fit_attempt_snapshot_sha256": "zero_to_five",
        "ordered_completed_diagnostic_row_sha256": "zero_to_twenty_five",
        "ordered_optimizer_call_observation_log": "zero_to_twenty_five",
    }
    assert "ordered_optimizer_call_observation_log" in failure_receipt["exact_fields"]
    assert failure_receipt["optimizer_call_observation_status_values_exact"] == [
        "completed_valid",
        "failed",
    ]
    assert {
        "fit_etas_raised_after_optimizer_return_before_returned_result_crosswalk",
        "returned_start_result_crosswalk_failed",
    } <= set(failure_receipt["incomplete_wrapper_failure_phase_values_exact"])
    assert failure_receipt["safe_failure_evidence_kind_values_exact"] == [
        "none",
        "complete_raw_OptimizeResult",
        "raw_schema_failure_type_state_projection",
    ]
    allowed_failure_evidence_kinds = set(failure_receipt["safe_failure_evidence_kind_values_exact"])
    assert all(
        set(contract["allowed_kinds"]) <= allowed_failure_evidence_kinds
        for contract in failure_receipt["safe_failure_evidence_phase_contract"].values()
    )
    assert failure_receipt[
        "every_failed_observation_sha256_must_resolve_to_exactly_one_canonical_observation_file_with_matching_recomputed_preimage"
    ]
    assert failure_receipt[
        "observation_file_set_must_equal_failed_projection_of_observation_log_with_no_missing_extra_or_overwrite"
    ]
    assert failure_receipt[
        "evidence_layer_may_not_reimplement_fit_postprocessing_or_fabricate_completed_invocation_receipt_after_fit_failure"
    ]
    assert failure_receipt[
        "ordered_completed_optimizer_invocation_receipt_sha256_must_equal_completed_valid_projection_of_observation_log"
    ]
    closing = seal["closing_seal"]
    assert closing["all_25_rows_and_five_snapshot_attempts_must_share_exact_attempt_id"]
    assert closing["preclosing_invalid_execution_has_no_closing_qualification_seal"]
    assert {
        "optimizer_runtime_code_seal_sha256",
        "ordered_five_fit_etas_call_opening_receipt_sha256",
        "ordered_25_optimizer_invocation_receipt_sha256",
        "ordered_five_fit_etas_call_closing_receipt_sha256",
    } <= set(closing["includes"])
    assert closing["self_or_mutual_hash_reference_forbidden"] is True
    assert (
        "staged_public_payload_identity_excluding_closing_seal_and_qualification_evidence"
        in (closing["includes"])
    )


def test_qualification_public_result_staging_paths_are_attempt_local_and_fail_closed() -> None:
    protocol = _load_yaml(PROTOCOL_PATH)
    seal = protocol["qualification_execution_seal"]
    staging = seal["qualification_public_result_staging"]
    outputs = protocol["outputs"]

    local_root = protocol["fit_input_bundle"]["local_root"]
    attempt_root = f"{local_root}/attempts/{{attempt_id}}"
    staged_root = f"{attempt_root}/staged_public"
    assert staging["fit_input_local_root_ref"] == "fit_input_bundle.local_root"
    assert staging["attempt_root_path_template"] == attempt_root
    assert staging["staged_public_root_path_template"] == staged_root
    attempt_id_contract = staging["attempt_id_path_component_contract"]
    assert attempt_id_contract["fullmatch_regex"] == "[A-Za-z0-9][A-Za-z0-9._-]{0,127}"
    assert attempt_id_contract["single_ascii_component_only"] is True
    assert (
        attempt_id_contract[
            "slash_backslash_colon_control_character_NUL_drive_UNC_empty_dot_and_dot_dot_forbidden"
        ]
        is True
    )
    assert (
        attempt_id_contract[
            "trailing_dot_or_space_and_case_insensitive_windows_reserved_device_basename_forbidden"
        ]
        is True
    )
    assert (
        attempt_id_contract[
            "exact_case_must_match_the_existing_same_attempt_fit_input_directory_name"
        ]
        is True
    )
    assert staging["staged_to_final_record_fields_exact"] == ["staged_path", "final_path"]

    common = staging["common_staged_to_final_path_mapping_exact"]
    expected_common_final_paths = {
        "fit_input_manifest": outputs["fit_input_manifest"],
        "opening_execution_seal": outputs["opening_execution_seal"],
        "optimizer_runtime_code_seal": outputs["optimizer_runtime_code_seal"],
        "qualification_input_seal": outputs["qualification_input_seal"],
        "report": outputs["report"],
        "static_diagnostic": outputs["static_diagnostic"],
        "interactive_diagnostic": outputs["interactive_diagnostic"],
        "qualification_closing_seal": outputs["qualification_closing_seal"],
        "qualification_manifest": outputs["qualification_manifest"],
    }
    assert len(common) == 9
    assert {key: record["final_path"] for key, record in common.items()} == (
        expected_common_final_paths
    )

    evaluable = staging["evaluable_additional_staged_to_final_path_mapping_exact"]
    expected_evaluable_final_paths = {
        "parameter_snapshots": (
            "models/registry/background_etas_numerical_repair/parameter_snapshots.json"
        ),
        "parameter_set_manifest": (
            "models/registry/background_etas_numerical_repair/parameter_set_manifest.json"
        ),
    }
    assert len(evaluable) == 2
    assert {key: record["final_path"] for key, record in evaluable.items()} == (
        expected_evaluable_final_paths
    )
    assert staging["not_evaluable_additional_staged_to_final_path_mapping_exact"] == {}

    preclosing_common_keys = staging["preclosing_common_logical_artifact_keys_exact"]
    assert preclosing_common_keys == [
        "fit_input_manifest",
        "opening_execution_seal",
        "optimizer_runtime_code_seal",
        "qualification_input_seal",
        "report",
        "static_diagnostic",
        "interactive_diagnostic",
    ]
    assert [common[key]["final_path"] for key in preclosing_common_keys] == outputs[
        "canonical_nested_schemas"
    ]["staged_public_payload_identity"]["common_complete_file_paths_exact"]
    assert staging["preclosing_evaluable_additional_logical_artifact_keys_exact"] == [
        "parameter_snapshots",
        "parameter_set_manifest",
    ]
    assert [
        evaluable[key]["final_path"]
        for key in staging["preclosing_evaluable_additional_logical_artifact_keys_exact"]
    ] == outputs["canonical_nested_schemas"]["staged_public_payload_identity"][
        "evaluable_additional_complete_file_paths_exact"
    ]
    assert staging["preclosing_not_evaluable_additional_logical_artifact_keys_exact"] == []

    closing_key = staging["qualification_closing_seal_logical_artifact_key_exact"]
    manifest_key = staging["qualification_manifest_logical_artifact_key_exact"]
    assert closing_key == "qualification_closing_seal"
    assert manifest_key == "qualification_manifest"
    assert closing_key not in preclosing_common_keys
    assert manifest_key not in preclosing_common_keys
    not_evaluable_order = [*preclosing_common_keys, closing_key, manifest_key]
    evaluable_order = [
        *preclosing_common_keys,
        *staging["preclosing_evaluable_additional_logical_artifact_keys_exact"],
        closing_key,
        manifest_key,
    ]
    assert staging["not_evaluable_public_materialization_logical_artifact_order_exact"] == (
        not_evaluable_order
    )
    assert staging["evaluable_public_materialization_logical_artifact_order_exact"] == (
        evaluable_order
    )
    assert not_evaluable_order[-1] == manifest_key
    assert evaluable_order[-1] == manifest_key

    all_records = [*common.values(), *evaluable.values()]
    assert len({record["final_path"] for record in all_records}) == 11
    assert len({record["staged_path"] for record in all_records}) == 11
    for record in all_records:
        assert set(record) == set(staging["staged_to_final_record_fields_exact"])
        assert record["staged_path"] == f"{staged_root}/{record['final_path']}"
        assert not record["final_path"].startswith(("/", "\\"))
        assert ".." not in record["final_path"].split("/")
        assert "\\" not in record["final_path"]

    ignore_probe_attempt_id = "r2-ignore-probe"
    ignored_staged_probe = f"{staged_root}/probe".format(attempt_id=ignore_probe_attempt_id)
    ignored_staged_check = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", ignored_staged_probe],
        check=False,
    )
    assert ignored_staged_check.returncode == 0
    for record in all_records:
        staged_path = record["staged_path"].format(attempt_id=ignore_probe_attempt_id)
        staged_path_check = subprocess.run(
            ["git", "check-ignore", "--no-index", "-q", staged_path],
            check=False,
        )
        assert staged_path_check.returncode == 0, staged_path
        public_path_check = subprocess.run(
            ["git", "check-ignore", "--no-index", "-q", record["final_path"]],
            check=False,
        )
        assert public_path_check.returncode == 1, record["final_path"]
        base_tree_check = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", R1_PROTOCOL_COMMIT, "--", record["final_path"]],
            check=False,
            capture_output=True,
            text=True,
        )
        assert base_tree_check.returncode == 0, base_tree_check.stderr
        assert base_tree_check.stdout == "", record["final_path"]

    result_diff = seal["qualification_result_tag_diff_from_repair_code_tag"]
    assert set(expected_common_final_paths.values()) == set(result_diff["common_exact_added_paths"])
    assert set(expected_evaluable_final_paths.values()) == set(
        result_diff["evaluable_additional_exact_paths"]
    )
    assert result_diff["not_evaluable_additional_exact_paths"] == []
    assert result_diff["invalid_execution_tracked_added_paths"] == []
    assert result_diff[
        "every_evaluable_or_not_evaluable_result_path_must_be_absent_in_repair_code_tag_and_have_exact_git_name_status_A_with_null_old_blob_oid"
    ]
    assert result_diff["overwrite_delete_rename_copy_or_other_modified_name_status_forbidden"]
    assert result_diff["exact_name_status_A_blob_oid_and_binary_patch_receipt_required"]

    staged_identity = outputs["canonical_nested_schemas"]["staged_public_payload_identity"]
    assert staged_identity[
        "ordered_complete_file_path_sha256_map_keys_must_equal_common_plus_qualification_status_branch_additional_paths_exactly"
    ]
    assert staged_identity[
        "every_map_key_must_equal_the_final_path_of_exactly_one_same_branch_qualification_public_result_staging_preclosing_record"
    ]
    assert staged_identity[
        "every_map_value_must_equal_sha256_of_complete_reopened_bytes_at_that_same_record_staged_path"
    ]
    assert staged_identity[
        "every_reopened_staged_path_must_be_non_reparse_regular_file_with_stable_size_hash_bytes_and_declared_schema_or_visual_contract"
    ]
    assert staged_identity[
        "aggregate_content_sha256_must_recompute_after_all_mapped_staged_files_are_reopened_and_before_closing_seal"
    ]

    for required in (
        "staged_public_root_must_equal_fit_input_local_root_plus_attempts_attempt_id_staged_public",
        "attempt_root_must_be_the_existing_non_reparse_fit_input_attempt_directory_strictly_below_fit_input_local_root_attempts",
        "staged_public_root_must_be_gitignored_attempt_exclusive_absent_before_staging_and_created_new",
        "preclosing_mappings_must_equal_staged_public_payload_identity_common_and_branch_additional_paths_exactly",
        "branch_materialization_order_must_equal_exact_preclosing_common_then_branch_additional_then_closing_then_manifest_and_manifest_must_be_last",
        "rollback_order_must_be_the_exact_reverse_of_the_successfully_created_prefix_of_the_same_branch_materialization_order",
        "staged_public_payload_identity_ordered_path_sha256_map_key_set_must_equal_same_branch_preclosing_mapping_final_paths_exactly",
        "staged_public_payload_identity_ordered_path_sha256_map_key_must_equal_each_mapped_record_final_path_and_value_must_equal_sha256_of_complete_reopened_bytes_at_same_record_staged_path",
        "staged_public_payload_identity_every_mapped_staged_file_must_reopen_as_non_reparse_regular_file_with_stable_size_sha256_and_complete_bytes_before_and_after_identity_construction",
        "staged_public_payload_identity_every_reopened_staged_file_must_validate_against_its_declared_strict_public_file_schema_or_visualization_contract",
        "staged_public_payload_identity_aggregate_must_recompute_from_exact_branch_status_ordered_path_sha256_map_and_preclosing_manifest_projection_sha256",
        "qualification_closing_seal_must_use_its_independent_staged_path_only_after_preclosing_identity_and_final_clean_repository_identity_are_frozen",
        "qualification_manifest_must_use_its_independent_staged_path_only_after_closing_seal_and_complete_qualification_evidence_and_manifest_content_sha256_are_frozen",
        "closing_seal_and_qualification_manifest_must_not_enter_staged_public_payload_identity",
        "every_staged_path_must_equal_staged_public_root_plus_its_exact_final_repository_relative_path",
        "staged_root_and_every_staged_or_final_path_must_resolve_strictly_inside_its_declared_root_without_symlink_junction_mount_point_or_other_reparse_escape",
        "every_existing_ancestor_must_be_reopened_and_verified_as_the_same_non_reparse_directory_before_each_create_copy_or_remove",
        "every_staged_and_final_file_must_be_absent_before_its_first_creation_and_no_historical_result_may_be_overwritten",
        "every_staged_file_must_use_exclusive_sibling_temp_write_flush_fsync_atomic_no_clobber_install_then_reopen_hash_and_byte_verification",
        "qualification_manifest_staged_file_must_strict_parse_recompute_all_own_and_cross_file_hashes_reserialize_byte_identically_and_reopen_before_public_materialization",
        "public_materialization_may_begin_only_after_all_branch_required_staged_files_closing_seal_and_qualification_manifest_reopen_checks_and_final_clean_repository_recheck_pass",
        "public_materialization_source_bytes_must_be_read_only_from_the_exact_mapped_staged_file_and_never_regenerated",
        "public_materialization_must_use_exclusive_destination_sibling_temp_byte_copy_flush_fsync_reopen_hash_and_byte_verification_atomic_no_clobber_install_then_final_reopen_verification",
        "every_materialized_final_file_must_equal_its_staged_source_in_size_sha256_and_complete_bytes",
        "every_final_result_path_must_be_absent_in_repair_code_tag_and_prepublication_worktree_then_have_git_status_A_only_after_successful_materialization",
        "tracked_status_for_every_materialized_path_must_be_exact_A_with_no_overwrite_delete_rename_or_other_modified_path_allowed",
        "prepublication_recheck_must_verify_clean_repository_unchanged_head_and_upstream_all_final_paths_absent_and_complete_staged_nine_or_eleven_file_set_byte_exact",
        "evaluable_must_materialize_exactly_all_nine_common_and_two_evaluable_additional_files",
        "not_evaluable_must_materialize_exactly_all_nine_common_and_no_additional_files",
        "not_evaluable_parameter_artifact_root_must_be_absent_before_during_and_after_materialization",
        "any_materialization_failure_must_rollback_in_reverse_creation_order_only_same_invocation_new_final_files_that_reopen_as_non_reparse_regular_files_and_still_match_their_exact_staged_bytes",
        "rollback_must_retain_all_attempt_staging_and_local_failure_evidence_and_must_never_remove_preexisting_mismatched_ambiguous_or_unverified_paths",
        "rollback_may_remove_only_attempt_unique_destination_sibling_temps_after_strict_parent_and_name_reverification_and_may_not_recursively_remove_any_directory",
        "rollback_must_reopen_and_verify_every_removed_final_path_is_absent_else_publication_failure_requires_manual_remediation_without_result_commit_or_tag",
        "post_closing_publication_failure_must_preserve_the_reopened_staged_closing_seal_and_any_installed_manifest_or_attempt_unique_manifest_temp_failure_evidence_and_publish_no_result_or_tag",
        "same_attempt_byte_exact_public_materialization_retry_allowed_only_after_complete_verified_rollback_clean_repository_unchanged_head_and_upstream_all_final_paths_absent_and_all_staged_bytes_and_hashes_unchanged",
        "same_attempt_publication_retry_may_not_rerun_fit_recompute_replace_or_modify_any_staged_payload_closing_seal_or_qualification_manifest_or_select_a_new_result",
        "failed_retry_precondition_permanently_invalidates_attempt_for_publication_and_forbids_result_commit_or_tag",
    ):
        assert staging[required] is True
    assert staging[
        "any_preclosing_path_creation_reopen_hash_byte_or_cross_file_mismatch_action"
    ] == ("invalid_execution_without_closing_seal_public_result_commit_or_qualification_result_tag")
    assert staging[
        "any_post_closing_copy_reopen_hash_byte_cross_file_or_rollback_mismatch_action"
    ] == ("publication_failure_without_public_result_commit_or_qualification_result_tag")

    failure_receipt = staging["publication_failure_receipt"]
    assert failure_receipt["path_template"] == (
        f"{attempt_root}/publication_failures/"
        "{publication_failure_sequence_decimal_zero_padded_6}.json"
    )
    assert failure_receipt["classification"] == "local_restricted_gitignored_append_only"
    assert failure_receipt["schema_version_exact"] == 1
    assert failure_receipt["fields_exact"] == [
        "schema_version",
        "attempt_id",
        "publication_failure_sequence",
        "failed_at_utc",
        "failure_phase",
        "failure_code",
        "exception_type_or_null",
        "qualification_closing_seal_sha256",
        "qualification_manifest_staging_state",
        "qualification_manifest_content_sha256_or_null",
        "qualification_manifest_file_sha256_or_null",
        "ordered_created_final_paths",
        "ordered_rolled_back_final_paths",
        "staged_file_size_and_sha256_or_null_by_final_path",
        "repository_identity_after_rollback",
        "rollback_complete",
        "retry_eligible",
        "previous_publication_failure_receipt_sha256_or_null",
        "publication_failure_receipt_sha256",
    ]
    assert failure_receipt["failure_phase_values_exact"] == [
        "qualification_manifest_construction",
        "qualification_manifest_sibling_temp_write",
        "qualification_manifest_atomic_no_clobber_install",
        "qualification_manifest_reopen",
        "qualification_manifest_schema_byte_cross_file_validation",
        "pre_materialization_recheck",
        "destination_sibling_temp_copy",
        "destination_sibling_temp_reopen",
        "atomic_no_clobber_install",
        "final_file_reopen",
        "rollback",
        "post_rollback_recheck",
    ]
    assert failure_receipt["qualification_manifest_staging_state_values_exact"] == [
        "not_constructed",
        "canonical_bytes_constructed_not_validly_reopened",
        "reopened_bytes_not_validated",
        "reopened_valid",
    ]
    manifest_state_by_phase = failure_receipt[
        "qualification_manifest_staging_state_required_by_failure_phase_exact"
    ]
    assert set(manifest_state_by_phase) == set(failure_receipt["failure_phase_values_exact"])
    assert manifest_state_by_phase["qualification_manifest_construction"] == "not_constructed"
    for phase in (
        "qualification_manifest_sibling_temp_write",
        "qualification_manifest_atomic_no_clobber_install",
        "qualification_manifest_reopen",
    ):
        assert manifest_state_by_phase[phase] == "canonical_bytes_constructed_not_validly_reopened"
    assert (
        manifest_state_by_phase["qualification_manifest_schema_byte_cross_file_validation"]
        == "reopened_bytes_not_validated"
    )
    for phase in failure_receipt["failure_phase_values_exact"][5:]:
        assert manifest_state_by_phase[phase] == "reopened_valid"
    assert failure_receipt[
        "qualification_manifest_sha_required_null_state_by_staging_state_exact"
    ] == {
        "not_constructed": {"content_sha256": None, "file_sha256": None},
        "canonical_bytes_constructed_not_validly_reopened": {
            "content_sha256": None,
            "file_sha256": None,
        },
        "reopened_bytes_not_validated": {
            "content_sha256": None,
            "file_sha256": "required",
        },
        "reopened_valid": {"content_sha256": "required", "file_sha256": "required"},
    }
    assert failure_receipt["first_sequence_exact"] == 0
    assert failure_receipt["maximum_sequence_exact"] == 999999
    assert failure_receipt["sequence_exhaustion_action"] == (
        "publication_failure_requires_manual_remediation_without_retry_result_commit_or_tag"
    )
    assert failure_receipt["attempt_id_schema_ref"] == (
        "qualification_execution_seal.qualification_public_result_staging."
        "attempt_id_path_component_contract"
    )
    assert failure_receipt["failed_at_utc_type"] == "canonical_RFC3339_UTC_instant_with_Z"
    assert failure_receipt["failure_code_fullmatch_regex"] == "[a-z][a-z0-9_]{0,127}"
    assert failure_receipt["staged_file_size_or_null_type"] == (
        "strict_nonnegative_base10_integer_or_null_only_for_manifest_before_reopened_bytes"
    )
    assert failure_receipt["staged_file_sha256_or_null_type"] == (
        "lowercase_hex_length_64_or_null_only_for_manifest_before_reopened_bytes"
    )
    assert failure_receipt["rollback_complete_and_retry_eligible_type"] == "strict_boolean"
    assert failure_receipt[
        "staged_file_size_and_sha256_or_null_by_final_path_value_fields_exact"
    ] == [
        "file_size_or_null",
        "file_sha256_or_null",
    ]
    assert (
        failure_receipt["staged_file_size_and_sha256_or_null_by_final_path_key_order"]
        == "unicode_codepoint_ascending"
    )
    assert failure_receipt["repository_identity_after_rollback_schema_ref"] == (
        "outputs.canonical_nested_schemas.final_clean_repository_identity"
    )
    for required in (
        "path_sequence_component_must_equal_sequence_as_exact_six_digit_zero_padded_decimal",
        "each_later_sequence_must_equal_previous_sequence_plus_one",
        "previous_receipt_sha256_is_null_only_for_sequence_zero_else_must_equal_immediately_previous_reopened_receipt_own_sha256",
        "ordered_created_final_paths_must_equal_successfully_created_prefix_of_same_branch_materialization_order",
        "ordered_rolled_back_final_paths_must_equal_exact_successful_reverse_rollback_projection_of_ordered_created_final_paths",
        "any_extra_missing_alias_duplicate_unknown_or_noncanonical_top_level_or_nested_field_forbidden",
        "staged_file_size_and_sha256_or_null_by_final_path_key_set_must_equal_complete_same_branch_nine_or_eleven_final_paths",
        "every_non_manifest_staged_file_size_and_sha256_value_must_be_nonnull_and_equal_reopened_same_attempt_mapped_staged_regular_file_complete_bytes",
        "manifest_staged_file_size_and_sha256_value_must_be_null_null_before_reopened_bytes_and_nonnull_equal_reopened_complete_bytes_for_reopened_bytes_not_validated_or_reopened_valid",
        "qualification_closing_seal_sha256_must_equal_reopened_same_attempt_staged_closing_seal_own_sha256",
        "nonnull_qualification_manifest_content_sha256_must_equal_reopened_valid_same_attempt_staged_manifest_own_content_sha256",
        "nonnull_qualification_manifest_file_sha256_must_equal_sha256_of_complete_reopened_same_attempt_staged_manifest_file_bytes",
        "manifest_staging_failure_phases_require_empty_created_and_rolled_back_final_paths_retry_eligible_false_and_no_same_attempt_materialization_retry",
        "retry_eligible_true_iff_rollback_complete_repository_clean_head_upstream_unchanged_all_final_paths_absent_and_all_staged_sizes_hashes_and_bytes_unchanged",
        "same_attempt_receipt_files_must_be_complete_gapless_append_only_hash_chain_and_may_never_be_overwritten_truncated_deleted_or_reordered",
    ):
        assert failure_receipt[required] is True
    failure_receipt_probe = failure_receipt["path_template"].format(
        attempt_id=ignore_probe_attempt_id,
        publication_failure_sequence_decimal_zero_padded_6="000000",
    )
    failure_receipt_ignore_check = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", failure_receipt_probe],
        check=False,
    )
    assert failure_receipt_ignore_check.returncode == 0


def test_three_grid_evidence_is_attempt_local_create_once_durable_and_reopened() -> None:
    protocol = _load_yaml(PROTOCOL_PATH)
    grid = protocol["qualification"]["three_grid_gate_evidence_protocol"]
    persistence = grid["local_restricted_persistence"]
    staging = protocol["qualification_execution_seal"]["qualification_public_result_staging"]
    attempt_root = staging["attempt_root_path_template"]
    expected_root = f"{attempt_root}/local_restricted/three_grid_gate_evidence"
    expected_template = f"{expected_root}/{{snapshot_id}}.json"

    assert len(grid["evidence_fields_exact"]) == 13
    assert persistence["persisted_file_is_complete_envelope_not_bare_13_field_evidence"]
    assert persistence["envelope_fields_exact"] == [
        "envelope_schema_version",
        "installed_file_identity",
        "sealed_three_grid_gate_evidence",
        "grid_resolution_payload_by_grid_size",
        "numerical_evidence_id_crosswalk",
        "three_grid_gate_evidence_envelope_sha256",
    ]
    assert (
        persistence["envelope_identity_fields_exact"] == persistence["envelope_fields_exact"][:-1]
    )
    assert persistence[
        "envelope_identity_fields_must_equal_envelope_fields_excluding_three_grid_gate_"
        "evidence_envelope_sha256"
    ]
    assert persistence["installed_file_identity_fields_exact"] == [
        "profile_id",
        "filesystem_name",
        "parent_identity",
        "ordered_directory_identity_chain",
        "ordered_directory_identity_chain_sha256",
        "final_repository_relative_path",
        "temp_leaf_utf8_hex",
        "final_leaf_utf8_hex",
        "canonical_final_path",
        "exact_case_relative_components",
        "platform",
        "windows_volume_serial_u64_decimal_or_null",
        "windows_file_id_16_bytes_hex_or_null",
        "windows_number_of_links_at_temp_capture_or_null",
        "posix_st_dev_decimal_or_null",
        "posix_st_ino_decimal_or_null",
        "posix_st_nlink_at_temp_capture_or_null",
        "required_final_nlink",
    ]
    assert persistence["installed_file_parent_identity_fields_exact"] == [
        "platform",
        "canonical_final_path",
        "exact_case_relative_components",
        "windows_volume_serial_u64_decimal_or_null",
        "windows_directory_file_id_16_bytes_hex_or_null",
        "posix_st_dev_decimal_or_null",
        "posix_st_ino_decimal_or_null",
    ]
    assert persistence["directory_identity_chain_record_fields_exact"] == [
        "repository_relative_directory_path",
        "exact_case_relative_components_from_workspace_root",
        "canonical_directory_path",
        "platform",
        "windows_volume_serial_u64_decimal_or_null",
        "windows_directory_file_id_16_bytes_hex_or_null",
        "posix_st_dev_decimal_or_null",
        "posix_st_ino_decimal_or_null",
    ]
    assert persistence["directory_identity_chain_length_exact"] == 9
    assert persistence["directory_identity_chain_repository_relative_path_templates_exact"] == (
        _directory_chain_relative_paths("{attempt_id}")
    )
    raw_file_id_type = (
        "lowercase_hex_of_FILE_ID_128_Identifier_bytes_index_0_through_15_in_raw_array_"
        "order_no_integer_or_GUID_endian_conversion_or_null"
    )
    assert (
        persistence["installed_file_identity_field_types_exact"][
            "windows_file_id_16_bytes_hex_or_null"
        ]
        == raw_file_id_type
    )
    assert (
        persistence["installed_file_parent_identity_field_types_exact"][
            "windows_directory_file_id_16_bytes_hex_or_null"
        ]
        == raw_file_id_type
    )
    assert (
        persistence["directory_identity_chain_record_field_types_exact"][
            "windows_directory_file_id_16_bytes_hex_or_null"
        ]
        == raw_file_id_type
    )
    assert persistence[
        "every_windows_FILE_ID_128_hex_must_serialize_Identifier_raw_bytes_in_index_0_"
        "through_15_order_without_GUID_or_integer_endian_conversion"
    ]
    assert persistence["initial_install_directory_identity_chain_source_exact"].startswith(
        "one_live_verified_handle_per_workspace_root"
    )
    assert persistence["before_seal_return_expected_directory_identity_source_exact"].startswith(
        "installer_live_handle_chain"
    )
    assert persistence["fresh_checkpoint_workspace_root_namespace_source_exact"].startswith(
        "isolated_launcher_fixed_worktree_root_argument"
    )
    assert persistence["fresh_checkpoint_expected_directory_identity_source_exact"].startswith(
        "complete_envelope_chain_whose_outer_sha_must_equal_independently_frozen"
    )
    assert persistence["fresh_checkpoint_observed_live_directory_identity_source_exact"].startswith(
        "independent_launcher_root_handle"
    )
    assert persistence[
        "fresh_checkpoint_reference_verifier_requires_controlled_single_use_live_directory_"
        "capture_provider_and_invokes_it_before_evidence_stat_or_read"
    ]
    assert persistence["synthetic_reference_live_capture_receipt_contract_exact"] == {
        "fields_exact": ["source_kind", "capture_id", "opened_handle_count", "records"],
        "source_kind_exact": "independent_live_directory_handle_capture",
        "capture_id": (
            "nonempty_process_unique_monotonic_synthetic_checkpoint_identifier_consumed_"
            "exactly_once"
        ),
        "opened_handle_count": "exact_directory_identity_chain_length",
        "records": (
            "independently_constructed_nine_record_live_chain_equal_only_by_value_to_"
            "authenticated_embedded_chain"
        ),
        "opaque_runtime_provenance_capability": (
            "minted_only_by_controlled_provider_factory_registered_by_object_identity_"
            "consumed_exactly_once_and_not_serialized_or_value_forgeable"
        ),
        "controlled_provider_input": (
            "independent_raw_live_handle_source_values_only_never_embedded_chain_or_receipt_records"
        ),
        "provider_constructor_accepts_only_controlled_live_handle_source_and_never_a_"
        "prebuilt_record_collection": True,
        "controlled_live_handle_source_capture_provider_and_receipt_must_have_exact_factory_"
        "class_and_reject_subclasses_or_structural_matches": True,
        "provider_and_source_factory_bindings_and_single_use_state_must_live_in_external_"
        "object_identity_registries_not_mutable_instance_flags": True,
        "unbound_source_registry_must_freeze_the_complete_construction_snapshot_"
        "immediately_when_the_exact_factory_finishes_source_construction": True,
        "provider_constructor_must_atomically_consume_the_unbound_source_then_compare_the_"
        "original_construction_snapshot_before_binding": True,
        "provider_capture_must_atomically_consume_its_registered_original_object_then_"
        "revalidate_the_exact_original_bound_source_object_before_observation": True,
        "source_binding_registry_must_freeze_every_construction_field_type_and_value_plus_"
        "path_object_identity_and_identity_override_canonical_bytes": True,
        "source_observation_must_atomically_consume_then_reject_any_post_binding_source_"
        "configuration_mutation_before_path_stat_or_record_construction": True,
        "controlled_provider_must_be_single_use_and_called_by_reference_verifier_before_"
        "file_stat_open_or_read": True,
        "registered_original_receipt_binding": (
            "exact_receipt_object_identity_capture_id_opened_handle_count_and_issuance_"
            "canonical_record_bytes_until_first_verification_attempt"
        ),
        "first_registered_original_verification_attempt_must_atomically_consume_capture_id_"
        "and_capability_before_count_or_content_validation": True,
        "passing_embedded_chain_directly_or_by_shallow_or_deep_copy_must_be_rejected": True,
        "wrapping_a_deep_copy_in_a_structurally_valid_receipt_with_self_claimed_source_and_"
        "capture_id_must_be_rejected": True,
        "post_construction_source_swap_or_provider_or_source_used_state_reset_must_be_"
        "rejected": True,
        "post_construction_pre_binding_path_to_raw_mode_or_raw_scalar_mutation_must_be_"
        "rejected": True,
        "post_binding_path_to_raw_mode_or_any_other_source_field_mutation_must_be_rejected": True,
        "stealing_a_registered_capability_into_a_new_wrapper_or_mutating_any_scalar_"
        "metadata_or_nested_records_after_issuance_must_be_rejected": True,
        "duplicate_capture_id_capability_reuse_or_fewer_than_nine_opened_handles_must_be_"
        "rejected": True,
    }
    assert persistence[
        "observed_live_directory_identity_chain_must_not_alias_or_be_derived_from_the_"
        "embedded_expected_chain"
    ]
    assert persistence["fresh_checkpoint_verification_order_exact"].startswith(
        "capture_independent_live_directory_chain_without_reading_envelope"
    )
    assert persistence[
        "compare_every_previously_captured_observed_live_directory_record_to_the_"
        "outer_SHA_authenticated_embedded_expected_record_before_accepting_or_"
        "trusting_the_evidence_file"
    ]
    root_argument = persistence["fixed_workspace_root_argument_contract_exact"]
    assert root_argument["cli_option"] == "--workspace-root"
    assert root_argument["occurrence_count_exact"] == 1
    assert root_argument["public_artifact_value"] == "forbidden_local_restricted_only"
    assert root_argument["windows_nt_object_name_example"] == (
        r"\??\Volume{11111111-2222-3333-4444-555555555555}\repo"
    )
    assert root_argument[
        "any_drive_UNC_DOS_device_alias_environment_current_directory_or_envelope_"
        "derived_root_forbidden"
    ]
    assert (
        "opening_seal_verifies_HEAD_upstream_repair_code_tag"
        in persistence["opening_repository_root_crosswalk_exact"]
    )
    opening_requirements = protocol["qualification_execution_seal"]["opening_seal"][
        "repository_requirements"
    ]
    assert opening_requirements[
        "fixed_workspace_root_argument_must_open_the_exact_worktree_whose_HEAD_upstream_"
        "remote_code_tag_and_protocol_package_blobs_are_verified_here"
    ]
    assert opening_requirements[
        "absolute_workspace_root_path_remains_local_restricted_and_may_not_enter_opening_"
        "or_public_artifacts"
    ]
    launcher = protocol["qualification_execution_seal"]["optimizer_runtime_code_seal"][
        "isolated_launcher_contract"
    ]
    assert launcher["qualification_local_restricted_workspace_root_cli_option_exact"] == (
        "--workspace-root"
    )
    assert launcher[
        "workspace_root_option_must_occur_exactly_once_and_equal_the_fixed_root_argument_"
        "contract_before_any_project_import_or_evidence_open"
    ]
    assert launcher[
        "workspace_root_argument_must_not_come_from_environment_current_directory_or_envelope"
    ]
    assert persistence[
        "every_fresh_checkpoint_must_capture_live_chain_from_the_launcher_root_handle_"
        "and_compare_every_record_to_the_outer_SHA_anchored_embedded_chain_before_"
        "trusting_file_content"
    ]
    assert (
        persistence["installed_file_identity_field_types_exact"][
            "windows_volume_serial_u64_decimal_or_null"
        ]
        == "canonical_unsigned_u64_base10_string_or_null"
    )
    assert (
        persistence["installed_file_parent_identity_field_types_exact"][
            "windows_volume_serial_u64_decimal_or_null"
        ]
        == "canonical_unsigned_u64_base10_string_or_null"
    )
    assert persistence["sealed_three_grid_gate_evidence_field_count_exact"] == 13
    assert persistence["sealed_three_grid_gate_evidence_fields_exact_ref"] == (
        "qualification.three_grid_gate_evidence_protocol.evidence_fields_exact"
    )
    assert persistence["each_grid_resolution_payload_field_count_exact"] == 6
    assert persistence["each_grid_resolution_payload_fields_exact_ref"] == (
        "qualification.three_grid_gate_evidence_protocol.grid_resolution_payload_fields_exact"
    )
    crosswalk = persistence["numerical_evidence_id_crosswalk_exact"]
    assert list(crosswalk) == persistence["numerical_evidence_id_crosswalk_fields_exact"]
    assert crosswalk["implementation_qualified_name"] == (
        "seismoflux.background.pipeline_etas.ETASGridGateEvidence.numerical_evidence_id"
    )
    assert crosswalk["numerical_evidence_preimage_fields_exact"] == [
        "protocol_sha256",
        "snapshot_id",
        "parameter_snapshot_id",
        "resolutions",
        "comparisons",
    ]
    assert crosswalk["resolution_asdict_fields_exact"] == list(
        asdict(ETASGridResolutionEvidence(50.0, 1, 1.0, 0.0, 1.0, "0" * 64))
    )
    assert crosswalk["comparison_asdict_fields_exact"] == list(
        asdict(GridConvergenceDiagnostics(50.0, 25.0, 1.0, 1.0, 0.0, 0.0))
    )
    assert persistence["attempt_root_path_template_ref"] == (
        "qualification_execution_seal.qualification_public_result_staging."
        "attempt_root_path_template"
    )
    assert persistence["root_path_template"] == expected_root
    assert persistence["file_path_template"] == expected_template
    assert persistence["snapshot_key_order_exact"] == list(SNAPSHOT_ORDER)
    expected_paths = {
        snapshot_id: f"{expected_root}/{snapshot_id}.json" for snapshot_id in SNAPSHOT_ORDER
    }
    assert persistence["snapshot_path_by_id_exact"] == expected_paths
    assert persistence["directory_components_below_attempt_root_exact"] == [
        "local_restricted",
        "three_grid_gate_evidence",
    ]
    for required in (
        "snapshot_id_path_substitution_must_use_exact_allowlisted_snapshot_id_without_alias_case_or_separator_normalization",
        "every_path_must_equal_attempt_root_plus_local_restricted_three_grid_gate_evidence_and_exact_snapshot_filename",
        "every_path_and_existing_ancestor_must_resolve_strictly_inside_same_non_reparse_attempt_root_without_symlink_junction_mount_hardlink_alias_drive_UNC_dot_dot_case_or_separator_escape",
        "each_existing_directory_component_must_be_reopened_by_parent_relative_handle_and_verified_same_non_reparse_directory_before_every_file_operation",
        "each_missing_directory_component_must_be_created_once_without_replacement_then_durably_bound_to_its_verified_parent_before_use",
        "three_grid_gate_evidence_directory_may_preexist_only_as_the_same_verified_directory_created_by_same_attempt_initialization_or_retained_from_an_earlier_same_attempt_snapshot_and_may_never_be_recreated_or_replaced",
        "directory_entries_and_installed_files_are_append_only_and_successful_snapshot_files_must_follow_frozen_snapshot_order",
        "complete_directory_tree_must_be_precreated_durably_reopened_and_identity_verified_before_any_same_attempt_three_grid_evaluator_call",
        "any_directory_create_reopen_identity_or_durability_capability_failure_must_fail_closed_before_grid_evaluator_and_require_new_attempt_id",
        "sealed_three_grid_gate_evidence_must_be_byte_identical_to_the_existing_13_field_object_before_embedding",
        "numerical_evidence_id_reconstruction_must_match_pipeline_etas_ETASGridGateEvidence_property_field_for_field_without_wrapper_passed_failure_reasons_or_diagnostic_hash_fields",
        "each_resolution_sha256_must_recompute_from_matching_embedded_exact_six_field_payload_and_equal_sealed_grid_resolution_payload_sha256_by_grid_size",
        "each_pair_sha256_must_recompute_from_matching_embedded_pair_payload_without_diagnostic_payload_sha256",
        "sealed_13_field_own_sha256_must_recompute_without_three_grid_gate_evidence_sha256",
        "persisted_file_must_strict_parse_recompute_all_three_resolution_sha256_values_both_pair_sha256_values_sealed_13_field_own_sha256_numerical_evidence_id_envelope_own_sha256_and_reserialize_byte_identically",
        "file_must_be_absent_before_same_snapshot_grid_evaluator_call_and_before_no_clobber_install",
        "any_preexisting_file_even_if_byte_identical_is_invalid_execution_and_may_not_be_adopted",
        "exactly_zero_or_one_file_per_snapshot_and_exact_file_set_must_equal_presence_truth_table_projection",
        "seal_must_close_renamed_file_handle_after_flush_retain_verified_parent_chain_for_"
        "handle_relative_final_reopen_then_close_reopened_file_and_all_parent_handles_after_"
        "complete_verification",
        "seal_returned_13_field_payload_envelope_sha256_and_public_crosswalk_values_must_be_derived_only_from_complete_reopened_envelope_bytes_not_from_preinstall_memory_or_temp_bytes",
        "seal_may_return_only_after_path_regular_file_identity_stable_size_complete_bytes_strict_envelope_schema_all_nested_hashes_both_own_hashes_numerical_evidence_id_crosswalk_and_byte_reserialization_checks_pass",
        "every_checkpoint_must_discard_in_memory_payload_temp_bytes_and_cached_file_handle_then_reopen_the_exact_installed_path_from_disk",
        "every_checkpoint_must_revalidate_exact_file_set_presence_stable_identity_and_link_count_complete_bytes_strict_envelope_and_embedded_13_field_schema_three_resolution_hashes_two_pair_hashes_both_own_hashes_numerical_evidence_id_canonical_reserialization_and_cross_snapshot_bindings",
        "before_first_external_anchor_checkpoint_must_return_reopened_envelope_sha_for_immediate_durable_snapshot_gate_anchor_without_permitting_restart_reanchor",
        "every_checkpoint_at_or_after_first_external_anchor_must_rederive_envelope_sha_from_reopened_file_and_compare_to_independently_frozen_snapshot_gate_fit_attempt_and_staged_local_presence_values",
        "checkpoint_replay_of_grid_evaluator_or_reconstruction_from_fit_result_gate_manifest_or_public_sha_forbidden",
        "missing_corrupt_changed_or_extra_file_may_not_be_recreated_replaced_substituted_or_repaired_under_same_attempt",
        "same_attempt_grid_evaluator_rerun_or_evidence_recompute_reseal_replace_substitute_or_alternate_path_forbidden",
        "all_installed_evidence_files_and_attempt_directories_must_be_retained_after_invalid_execution_publication_failure_successful_public_materialization_result_commit_and_remote_tag",
        "public_schema_public_path_and_public_file_count_must_remain_unchanged_and_local_file_path_or_bytes_may_not_be_published",
    ):
        assert persistence[required] is True
    assert (
        "seal_returned_13_field_payload_envelope_sha256_and_initial_external_crosswalk_"
        "values_must_be_derived_only_from_complete_reopened_envelope_bytes_not_from_"
        "preinstall_memory_or_temp_bytes" not in persistence
    )

    durability = persistence["durable_create_once_file_protocol"]
    assert durability["selected_install_profile_exact"] == (
        "windows_ntfs_ntcreatefile_filerenameinfo_v1"
    )
    assert durability[
        "posix_linkat_v1_is_defined_for_portability_but_is_not_selectable_in_this_"
        "frozen_windows_qualification_runtime"
    ]
    assert durability["scope_exact"] == (
        "local_restricted_three_grid_evidence_envelope_files_and_their_local_restricted_"
        "and_three_grid_gate_evidence_directory_components_only"
    )
    assert durability[
        "staged_public_closing_manifest_failure_receipt_and_public_destination_install_"
        "protocols_remain_exactly_R2_and_are_not_governed_or_changed_by_this_local_"
        "durable_contract"
    ]
    assert durability["install_failure_state_machine"] == {
        "preexisting_final_before_evaluator_or_install": (
            "collision_invalid_execution_never_adopt_even_if_byte_identical"
        ),
        "failure_before_install_success": (
            "retain_temp_and_attempt_evidence_no_primitive_fallback_no_same_final_retry"
        ),
        "install_success_until_complete_postinstall_flush_parent_sync_and_reopen_verification": (
            "any_crash_or_failure_is_indeterminate_after_install_retain_everything_for_audit_"
            "read_only_forensic_reopen_only_no_delete_rollback_overwrite_or_same_final_retry_"
            "same_attempt_may_never_resume_reanchor_publish_or_advance_qualification_new_"
            "attempt_id_only_after_manual_audit"
        ),
        "complete_postinstall_reopen_verification_before_first_external_envelope_sha_anchor_"
        "is_durably_persisted": (
            "any_crash_or_failure_is_invalid_execution_retain_everything_same_attempt_may_"
            "never_resume_reanchor_or_publish_new_attempt_only_after_manual_audit"
        ),
        "at_or_after_first_external_envelope_sha_anchor": (
            "any_missing_changed_or_mismatched_local_envelope_or_external_anchor_is_invalid_"
            "execution_retain_everything_without_same_attempt_repair_reanchor_or_scientific_"
            "retry"
        ),
        "any_scientific_retry": "new_attempt_id_only_after_manual_audit",
    }
    assert durability["common_order_exact"] == [
        "strict_parent_reopen",
        "missing_directory_create_once_and_directory_entry_durable_sync",
        "exclusive_sibling_temp_create",
        "temp_file_identity_capture_and_destination_path_case_prebind",
        "complete_canonical_envelope_byte_write",
        "file_flush_and_durable_sync",
        "close_initial_temp_write_handle",
        "preinstall_temp_handle_relative_reopen_and_verification",
        "atomic_no_clobber_install",
        "installed_file_flush",
        "parent_directory_entry_durable_sync",
        "close_renamed_file_handle_while_retaining_verified_parent_chain",
        "installed_file_handle_relative_reopen_and_complete_verification",
        "close_reopened_file_and_all_parent_handles",
    ]
    windows = durability["windows"]
    assert windows["profile_id_exact"] == "windows_ntfs_ntcreatefile_filerenameinfo_v1"
    assert windows["trusted_namespace_bootstrap_exact"] == (
        "one_absolute_NT_path_open_of_trusted_workspace_root_per_initial_install_or_fresh_"
        "checkpoint_verification_session_then_attempt_root_and_all_descendants_opened_only_"
        "by_RootDirectory_relative_descent"
    )
    assert windows["trusted_workspace_root_bootstrap_object_attributes_exact"] == (
        "RootDirectory_NULL_ObjectName_exact_NT_path_derived_from_fixed_launcher_workspace_"
        "root_argument_and_bound_to_opening_repository_HEAD_Attributes_OBJ_DONT_REPARSE_"
        "0x1000_without_OBJ_CASE_INSENSITIVE_0x40"
    )
    assert "trusted_workspace_root_bootstrap_create_disposition_and_options_exact" not in windows
    assert windows["trusted_workspace_root_bootstrap_create_disposition_exact"] == {
        "symbol": "FILE_OPEN",
        "uint32_hex": "00000001",
    }
    assert windows["trusted_workspace_root_bootstrap_create_options_exact"] == {
        "symbolic_or": (
            "FILE_DIRECTORY_FILE_0x00000001_bitwise_OR_FILE_OPEN_REPARSE_POINT_0x00200000_"
            "bitwise_OR_FILE_SYNCHRONOUS_IO_NONALERT_0x00000020"
        ),
        "uint32_hex": "00200021",
    }
    assert windows[
        "every_fresh_checkpoint_session_must_close_all_prior_handles_open_workspace_root_once_"
        "then_parent_relative_redescend_and_reverify_every_directory_identity_before_final_"
        "file_open"
    ]
    assert windows[
        "share_access_for_every_workspace_directory_temp_preinstall_and_final_reopen_exact"
    ] == (
        "FILE_SHARE_READ_0x1_bitwise_OR_FILE_SHARE_WRITE_0x2_equal_0x3_and_explicitly_"
        "without_FILE_SHARE_DELETE_0x4"
    )
    assert windows[
        "share_delete_absence_must_be_mechanically_verified_for_every_open_and_no_handle_"
        "with_DELETE_desired_access_may_overlap_an_incompatible_second_open"
    ]
    directory_access_mask = 0x20000 | 0x100000 | 0x1 | 0x2 | 0x4 | 0x8 | 0x10 | 0x20 | 0x80 | 0x100
    directory_access = {
        "symbolic_or": (
            "READ_CONTROL_0x00020000_bitwise_OR_SYNCHRONIZE_0x00100000_bitwise_OR_FILE_"
            "LIST_DIRECTORY_0x00000001_bitwise_OR_FILE_ADD_FILE_0x00000002_bitwise_OR_FILE_"
            "ADD_SUBDIRECTORY_0x00000004_bitwise_OR_FILE_READ_EA_0x00000008_bitwise_OR_FILE_"
            "WRITE_EA_0x00000010_bitwise_OR_FILE_TRAVERSE_0x00000020_bitwise_OR_FILE_READ_"
            "ATTRIBUTES_0x00000080_bitwise_OR_FILE_WRITE_ATTRIBUTES_0x00000100"
        ),
        "uint32_hex": f"{directory_access_mask:08x}",
    }
    temp_access = {
        "symbolic_or": (
            "GENERIC_READ_0x80000000_bitwise_OR_GENERIC_WRITE_0x40000000_bitwise_OR_"
            "DELETE_0x00010000_bitwise_OR_FILE_READ_ATTRIBUTES_0x00000080_bitwise_OR_"
            "SYNCHRONIZE_0x00100000"
        ),
        "uint32_hex": f"{0x80000000 | 0x40000000 | 0x10000 | 0x80 | 0x100000:08x}",
    }
    final_access = {
        "symbolic_or": (
            "GENERIC_READ_0x80000000_bitwise_OR_FILE_READ_ATTRIBUTES_0x00000080_bitwise_"
            "OR_SYNCHRONIZE_0x00100000"
        ),
        "uint32_hex": f"{0x80000000 | 0x80 | 0x100000:08x}",
    }
    assert windows["desired_access_by_open_kind_exact"] == {
        "trusted_workspace_root_bootstrap": directory_access,
        "attempt_root_or_descendant_directory_create_or_reopen": directory_access,
        "exclusive_temp_create_and_preinstall_reopen_for_same_handle_install": temp_access,
        "installed_final_read_only_reopen": final_access,
    }
    assert windows[
        "FILE_SYNCHRONOUS_IO_NONALERT_requires_SYNCHRONIZE_in_every_matching_desired_access_mask"
    ]
    assert windows["generic_access_high_bits_for_directory_NtCreateFile_forbidden"]
    assert int(directory_access["uint32_hex"], 16) & 0xC0000000 == 0
    assert int(directory_access["uint32_hex"], 16) & 0x00120116 == 0x00120116
    assert windows[
        "every_directory_handle_passed_to_FlushFileBuffers_must_have_the_fully_expanded_FILE_"
        "GENERIC_WRITE_specific_rights_0x00120116_as_a_subset_of_granted_access"
    ]
    assert windows[
        "every_file_handle_passed_to_FlushFileBuffers_requires_GENERIC_WRITE_in_its_desired_"
        "access_mask"
    ]
    assert windows["filesystem_capability_gate_exact"] == (
        "GetVolumeInformationByHandleW_on_workspace_attempt_root_and_evidence_parent_"
        "handles_must_report_exact_NTFS_and_identical_volume_serial_plus_"
        "NtQueryVolumeInformationFile_FileFsDeviceInformation_DeviceType_FILE_DEVICE_"
        "DISK_0x00000007_and_Characteristics_bitwise_AND_FILE_REMOTE_DEVICE_0x00000010_"
        "equal_zero"
    )
    assert windows["local_nonremote_predicate_exact"] == (
        "DeviceType_equal_FILE_DEVICE_DISK_0x00000007_and_Characteristics_bitwise_AND_"
        "FILE_REMOTE_DEVICE_0x00000010_equal_zero_on_workspace_attempt_root_and_evidence_"
        "parent_handles"
    )
    assert windows["remote_SMB_NFS_ReFS_FAT_unknown_or_nonlocal_filesystem_action"] == (
        "fail_closed_before_grid_evaluator"
    )
    assert windows["descendant_namespace_primitives_exact"] == (
        "NtCreateFile_RootDirectory_relative_open_and_create_plus_"
        "SetFileInformationByHandle_FileRenameInfo_RootDirectory_relative_install"
    )
    assert windows[
        "path_based_CreateDirectoryW_CreateFileW_or_MoveFileExW_for_descendant_"
        "directory_temp_destination_or_install_forbidden"
    ]
    assert windows["object_attributes_exact"] == (
        "RootDirectory_verified_parent_handle_ObjectName_single_UTF16_leaf_component_"
        "Attributes_OBJ_DONT_REPARSE_0x1000_without_OBJ_CASE_INSENSITIVE_0x40"
    )
    assert windows["ntcreatefile_common_structure_contract_exact"] == {
        "OBJECT_ATTRIBUTES_Length": "sizeof_OBJECT_ATTRIBUTES_for_active_architecture",
        "OBJECT_ATTRIBUTES_SecurityDescriptor": None,
        "OBJECT_ATTRIBUTES_SecurityQualityOfService": None,
        "OBJECT_ATTRIBUTES_Attributes_uint32_hex": "00001000",
        "UNICODE_STRING_Length": (
            "exact_ObjectName_UTF16LE_byte_length_without_terminator_reject_odd_or_over_65534"
        ),
        "UNICODE_STRING_MaximumLength": ("exactly_equal_Length_no_implicit_terminator_capacity"),
        "UNICODE_STRING_Buffer": (
            "exact_absolute_workspace_root_or_single_leaf_UTF16_code_units_matching_ObjectName_kind"
        ),
        "AllocationSize": None,
        "EaBuffer": None,
        "EaLength_uint32_hex": "00000000",
        "IO_STATUS_BLOCK_Status_initial": (
            "zero_initialized_before_call_then_NT_SUCCESS_required_after_call"
        ),
    }
    parameter_fields = [
        "root_directory_kind",
        "object_name_kind",
        "desired_access_uint32_hex",
        "share_access_uint32_hex",
        "file_attributes_symbol",
        "file_attributes_uint32_hex",
        "create_disposition_symbol",
        "create_disposition_uint32_hex",
        "create_options_symbolic_or",
        "create_options_uint32_hex",
        "allocation_size",
        "ea_buffer",
        "ea_length_uint32_hex",
        "expected_io_status_information_symbol",
        "expected_io_status_information_uint64_hex",
    ]
    assert windows["ntcreatefile_call_parameter_fields_exact"] == parameter_fields
    directory_options = (
        "FILE_DIRECTORY_FILE_0x00000001_bitwise_OR_FILE_OPEN_REPARSE_POINT_0x00200000_"
        "bitwise_OR_FILE_SYNCHRONOUS_IO_NONALERT_0x00000020"
    )
    temp_create_options = (
        "FILE_NON_DIRECTORY_FILE_0x00000040_bitwise_OR_FILE_OPEN_REPARSE_POINT_0x00200000_"
        "bitwise_OR_FILE_WRITE_THROUGH_0x00000002"
    )
    preinstall_options = f"{temp_create_options}_bitwise_OR_FILE_SYNCHRONOUS_IO_NONALERT_0x00000020"
    final_options = (
        "FILE_NON_DIRECTORY_FILE_0x00000040_bitwise_OR_FILE_OPEN_REPARSE_POINT_0x00200000_"
        "bitwise_OR_FILE_SYNCHRONOUS_IO_NONALERT_0x00000020"
    )
    row_defaults: dict[str, object] = {
        "share_access_uint32_hex": "00000003",
        "allocation_size": None,
        "ea_buffer": None,
        "ea_length_uint32_hex": "00000000",
    }
    expected_call_rows: dict[str, dict[str, object]] = {
        "trusted_workspace_root_bootstrap": {
            "root_directory_kind": None,
            "object_name_kind": (
                "exact_NT_ObjectName_derived_only_from_fixed_launcher_workspace_root_"
                "argument_and_crosschecked_to_opening_repository_HEAD"
            ),
            "desired_access_uint32_hex": "001201bf",
            "file_attributes_symbol": "zero_ignored_for_FILE_OPEN",
            "file_attributes_uint32_hex": "00000000",
            "create_disposition_symbol": "FILE_OPEN",
            "create_disposition_uint32_hex": "00000001",
            "create_options_symbolic_or": directory_options,
            "create_options_uint32_hex": "00200021",
            "expected_io_status_information_symbol": "FILE_OPENED",
            "expected_io_status_information_uint64_hex": "0000000000000001",
        },
        "existing_ancestor_or_evidence_directory_reopen": {
            "root_directory_kind": "verified_parent_directory_handle",
            "object_name_kind": "exact_safe_single_UTF16_leaf",
            "desired_access_uint32_hex": "001201bf",
            "file_attributes_symbol": "zero_ignored_for_FILE_OPEN",
            "file_attributes_uint32_hex": "00000000",
            "create_disposition_symbol": "FILE_OPEN",
            "create_disposition_uint32_hex": "00000001",
            "create_options_symbolic_or": directory_options,
            "create_options_uint32_hex": "00200021",
            "expected_io_status_information_symbol": "FILE_OPENED",
            "expected_io_status_information_uint64_hex": "0000000000000001",
        },
        "missing_directory_create": {
            "root_directory_kind": "verified_parent_directory_handle",
            "object_name_kind": "exact_safe_single_UTF16_leaf",
            "desired_access_uint32_hex": "001201bf",
            "file_attributes_symbol": "FILE_ATTRIBUTE_DIRECTORY",
            "file_attributes_uint32_hex": "00000010",
            "create_disposition_symbol": "FILE_CREATE",
            "create_disposition_uint32_hex": "00000002",
            "create_options_symbolic_or": directory_options,
            "create_options_uint32_hex": "00200021",
            "expected_io_status_information_symbol": "FILE_CREATED",
            "expected_io_status_information_uint64_hex": "0000000000000002",
        },
        "exclusive_temp_create": {
            "root_directory_kind": "verified_evidence_parent_directory_handle",
            "object_name_kind": "exact_attempt_unique_safe_single_UTF16_temp_leaf",
            "desired_access_uint32_hex": "c0110080",
            "file_attributes_symbol": "FILE_ATTRIBUTE_NORMAL",
            "file_attributes_uint32_hex": "00000080",
            "create_disposition_symbol": "FILE_CREATE",
            "create_disposition_uint32_hex": "00000002",
            "create_options_symbolic_or": temp_create_options,
            "create_options_uint32_hex": "00200042",
            "expected_io_status_information_symbol": "FILE_CREATED",
            "expected_io_status_information_uint64_hex": "0000000000000002",
        },
        "temp_preinstall_reopen": {
            "root_directory_kind": "verified_evidence_parent_directory_handle",
            "object_name_kind": "exact_stored_case_safe_single_UTF16_temp_leaf",
            "desired_access_uint32_hex": "c0110080",
            "file_attributes_symbol": "zero_ignored_for_FILE_OPEN",
            "file_attributes_uint32_hex": "00000000",
            "create_disposition_symbol": "FILE_OPEN",
            "create_disposition_uint32_hex": "00000001",
            "create_options_symbolic_or": preinstall_options,
            "create_options_uint32_hex": "00200062",
            "expected_io_status_information_symbol": "FILE_OPENED",
            "expected_io_status_information_uint64_hex": "0000000000000001",
        },
        "installed_final_read_only_reopen": {
            "root_directory_kind": "verified_evidence_parent_directory_handle",
            "object_name_kind": "exact_stored_case_safe_single_UTF16_final_leaf",
            "desired_access_uint32_hex": "80100080",
            "file_attributes_symbol": "zero_ignored_for_FILE_OPEN",
            "file_attributes_uint32_hex": "00000000",
            "create_disposition_symbol": "FILE_OPEN",
            "create_disposition_uint32_hex": "00000001",
            "create_options_symbolic_or": final_options,
            "create_options_uint32_hex": "00200060",
            "expected_io_status_information_symbol": "FILE_OPENED",
            "expected_io_status_information_uint64_hex": "0000000000000001",
        },
    }
    matrix = windows["ntcreatefile_call_parameter_matrix_exact"]
    assert list(matrix) == list(expected_call_rows)
    for call_kind, expected_specific in expected_call_rows.items():
        expected_row = {**row_defaults, **expected_specific}
        assert set(matrix[call_kind]) == set(parameter_fields)
        assert matrix[call_kind] == expected_row
    assert windows[
        "every_NtCreateFile_call_must_match_exactly_one_parameter_matrix_row_and_no_"
        "unlisted_default_or_implicit_argument_is_permitted"
    ]
    assert "NtCreateFile_with_exact_object_attributes" in windows["directory_create_exact"]
    assert "NtCreateFile_with_exact_object_attributes" in windows["temp_create_exact"]
    assert windows["temp_create_result_exact"] == (
        "NT_SUCCESS_and_IoStatusBlock_Information_FILE_CREATED"
    )
    assert "EndOfFile_zero" in windows["temp_initial_state_exact"]
    assert "FileIdInfo" in windows["temp_preinstall_reopen_exact"]
    for required_option in (
        "FILE_NON_DIRECTORY_FILE",
        "FILE_OPEN_REPARSE_POINT",
        "FILE_WRITE_THROUGH",
        "FILE_SYNCHRONOUS_IO_NONALERT",
    ):
        assert required_option in windows["temp_preinstall_reopen_exact"]
    assert windows["atomic_no_clobber_install_exact"] == (
        "SetFileInformationByHandle_FileRenameInfo_on_open_temp_handle_with_FILE_RENAME_"
        "INFO_ReplaceIfExists_FALSE_RootDirectory_verified_parent_handle_and_exact_"
        "destination_component"
    )
    assert windows["durability_flush_api_exact"] == "kernel32_FlushFileBuffers_only"
    assert windows["installed_file_flush_exact"] == (
        "kernel32_FlushFileBuffers_on_the_original_renamed_file_handle_must_succeed_"
        "before_parent_FlushFileBuffers"
    )
    assert windows[
        "NtFlushBuffersFile_NtFlushBuffersFileEx_or_any_other_NtFlush_durability_call_forbidden"
    ]
    assert "VolumeSerialNumber_and_FileId" in windows["identity_capture_exact"]
    assert "NumberOfLinks" in windows["identity_capture_exact"]
    assert windows["parent_identity_capture_exact"].startswith(
        "GetFileInformationByHandleEx_FileIdInfo_on_each_verified_workspace_attempt"
    )
    assert "FILE_ID_128" in windows["parent_identity_source_and_crosscheck_exact"]
    assert (
        "path_stat_or_synthetic_st_ino_source_forbidden"
        in windows["parent_identity_source_and_crosscheck_exact"]
    )
    assert windows["canonical_volume_guid_path_grammar_exact"] == (
        "windows_extended_length_prefix_then_Volume_braced_GUID_8_4_4_4_12_hex_then_"
        "backslash_then_one_or_more_exact_case_UTF16_NFC_safe_components"
    )
    assert windows[
        "fixed_launcher_Win32_Volume_GUID_path_to_NtCreateFile_NT_ObjectName_conversion_exact"
    ] == (
        "replace_exact_leading_two_backslashes_question_mark_backslash_with_single_"
        "backslash_question_mark_question_mark_backslash_preserve_all_remaining_UTF16_"
        "code_units_and_reject_any_other_prefix"
    )
    assert windows[
        "DOS_drive_UNC_incomplete_volume_GUID_dot_dot_dot_empty_or_repeated_separator_"
        "forward_slash_NUL_or_non_NFC_canonical_path_forbidden"
    ]
    assert windows[
        "synthetic_windows_path_fixture_must_use_a_complete_fake_volume_GUID_path_and_"
        "may_not_use_Path_resolve_drive_letter_output"
    ]
    assert windows[
        "volume_serial_file_id_final_path_and_exact_case_components_must_match_install_"
        "record_at_every_checkpoint"
    ]
    posix = durability["posix"]
    assert posix["profile_id_exact"] == "posix_linkat_v1"
    assert posix["selectable_in_current_windows_qualification_runtime"] is False
    assert posix["trusted_workspace_root_bootstrap_exact"] == (
        "open_exact_fixed_launcher_workspace_root_argument_once_with_O_RDONLY_O_DIRECTORY_"
        "O_NOFOLLOW_O_CLOEXEC_then_fstat_and_bind_to_opening_repository_HEAD"
    )
    assert posix[
        "every_fresh_checkpoint_session_must_close_all_prior_handles_open_the_fixed_"
        "workspace_root_once_then_openat_parent_relative_redescend_every_frozen_"
        "directory_component"
    ]
    assert posix[
        "each_of_nine_directory_identity_chain_records_must_come_from_fstat_on_its_"
        "independently_open_live_directory_handle_with_canonical_st_dev_and_st_ino_"
        "before_opening_the_next_child"
    ]
    assert posix[
        "absolute_open_below_root_chdir_environment_current_directory_envelope_path_or_"
        "path_stat_identity_source_forbidden"
    ]
    assert posix["directory_create_exact"] == (
        "mkdirat_verified_parent_once_then_openat_child_O_DIRECTORY_O_NOFOLLOW"
    )
    assert posix["atomic_no_clobber_install_exact"] == (
        "linkat_verified_parent_temp_component_to_same_verified_parent_destination_"
        "component_flags_zero_must_fail_EEXIST_if_destination_exists_then_unlinkat_"
        "temp_component"
    )
    assert posix["ordinary_rename_renameat_or_replace_install_forbidden"]
    assert posix[
        "install_profile_must_be_selected_and_capability_verified_before_first_grid_"
        "evaluator_and_may_not_fallback_during_attempt"
    ]
    assert posix[
        "parent_directory_fsync_required_after_each_mkdirat_linkat_success_and_unlinkat_"
        "temp_cleanup"
    ]
    assert "st_dev_and_st_ino" in posix["installed_identity_exact"]
    assert "st_nlink_equal_one" in posix["installed_identity_exact"]
    assert persistence["disk_reopen_checkpoints_exact"] == [
        "before_seal_return",
        "qualification_result_finalize",
        "staged_local_scientific_payload_identity_construction",
        "qualification_closing_seal_construction",
        "qualification_manifest_construction",
        "qualification_manifest_staged_reopen_validation",
        "before_initial_public_materialization",
        "before_every_same_attempt_public_materialization_retry",
    ]

    truth = persistence["presence_sha_null_truth_table_exact"]
    required_state = truth["grid_evidence_required"]
    assert required_state["exact_local_file_presence"] == "required"
    assert required_state["exact_local_file_payload"] == (
        "complete_six_field_envelope_with_installed_identity_embedded_sealed_13_fields_and_three_"
        "resolution_preimages_required"
    )
    for field in (
        "snapshot_gate_three_grid_gate_evidence_sha256_or_null",
        "fit_attempt_three_grid_gate_evidence_sha256_or_null",
        "staged_local_presence_map_value",
    ):
        assert required_state[field] == "required_equal_complete_envelope_own_sha256"
    forbidden_state = truth["grid_evidence_forbidden"]
    assert forbidden_state["exact_local_file_presence"] == "forbidden"
    assert forbidden_state["exact_local_file_payload"] == "absent"
    for field in (
        "snapshot_gate_three_grid_gate_evidence_sha256_or_null",
        "fit_attempt_three_grid_gate_evidence_sha256_or_null",
        "staged_local_presence_map_value",
        "snapshot_gate_grid_50_to_25_diagnostic_payload_sha256_or_null",
        "snapshot_gate_grid_25_to_12_5_expected_count_relative_difference_hex_or_null",
        "snapshot_gate_grid_25_to_12_5_density_l1_hex_or_null",
    ):
        assert forbidden_state[field] is None

    r3_probe_attempt = "r3-three-grid-evidence-probe"
    for path in expected_paths.values():
        formatted = path.format(attempt_id=r3_probe_attempt)
        ignore_check = subprocess.run(
            ["git", "check-ignore", "--no-index", "-q", formatted],
            check=False,
        )
        assert ignore_check.returncode == 0, formatted

    r2 = _load_yaml_at_revision(R2_PROTOCOL_COMMIT, PROTOCOL_PATH)
    assert staging == r2["qualification_execution_seal"]["qualification_public_result_staging"]

    execution_rules = protocol["repair_code_scope_from_protocol_tag"][
        "new_repair_module_execution_rules"
    ]
    responsibility = (
        "complete_three_grid_local_restricted_create_once_persistence_and_disk_reopen_validation"
    )
    assert (
        responsibility
        in execution_rules["src/seismoflux/background/etas_numerical_repair_io.py"][
            "allowed_responsibilities"
        ]
    )
    assert (
        responsibility
        in execution_rules["src/seismoflux/background/etas_numerical_repair_evidence.py"][
            "allowed_responsibilities"
        ]
    )


def test_three_grid_envelope_synthetic_canonical_round_trip_matches_pipeline_formula(
    tmp_path: Path,
) -> None:
    protocol = _load_yaml(PROTOCOL_PATH)
    temp_path = tmp_path / ".fold_1.envelope.tmp"
    final_path = tmp_path / "fold_1.json"
    with temp_path.open("xb"):
        pass
    installed_identity = _synthetic_installed_file_identity(temp_path, final_path)
    gate, envelope, public_crosswalk = _synthetic_three_grid_envelope(
        protocol,
        installed_identity,
    )
    envelope_bytes = canonical_json_bytes(envelope)
    with temp_path.open("r+b") as handle:
        handle.write(envelope_bytes)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temp_path, final_path)
    assert temp_path.stat().st_nlink == 2
    temp_path.unlink()
    assert final_path.stat().st_nlink == 1
    _assert_installed_file_identity(final_path, installed_identity)

    verification_order_events: list[str] = []
    verified = _reopen_and_verify_synthetic_three_grid_envelope(
        protocol,
        final_path,
        public_crosswalk,
        live_directory_capture_provider=(
            _synthetic_live_directory_capture_provider_from_path(
                final_path,
                attempt_id="synthetic-r3-envelope",
            )
        ),
        expected_attempt_id="synthetic-r3-envelope",
        expected_snapshot_id="fold_1",
        expected_protocol_sha256=gate.protocol_sha256,
        verification_order_events=verification_order_events,
    )
    assert verification_order_events == [
        "capture_independent_live_directory_chain",
        "open_and_read_untrusted_complete_file",
        "outer_sha_matches_independent_public_anchors",
        "compare_live_chain_to_authenticated_embedded_chain",
        "accept_authenticated_file",
    ]
    reopened_bytes = final_path.read_bytes()
    assert reopened_bytes == envelope_bytes
    assert verified["numerical_evidence_id"] == gate.numerical_evidence_id
    assert (
        verified["envelope_sha256"]
        == public_crosswalk["snapshot_gate_three_grid_gate_evidence_sha256_or_null"]
    )
    assert (
        verified["sealed_sha256"]
        == cast(dict[str, object], envelope["sealed_three_grid_gate_evidence"])[
            "three_grid_gate_evidence_sha256"
        ]
    )
    assert verified["envelope_sha256"] == envelope["three_grid_gate_evidence_envelope_sha256"]


def test_three_grid_envelope_rejects_cascade_rehash_tamper_against_public_crosswalk() -> None:
    protocol = _load_yaml(PROTOCOL_PATH)
    _, envelope, public_crosswalk = _synthetic_three_grid_envelope(protocol)
    tampered = deepcopy(envelope)
    resolution_map = cast(
        dict[str, dict[str, object]], tampered["grid_resolution_payload_by_grid_size"]
    )
    resolution_25 = resolution_map["25_km"]
    resolution_25["background_total_hex"] = (2.0).hex()
    resolution_25["triggering_total_hex"] = (1.0).hex()
    _cascade_rehash_synthetic_three_grid_envelope(protocol, tampered)

    with pytest.raises(ValueError, match="public gate/fit-attempt/presence crosswalk"):
        _verify_synthetic_three_grid_envelope(
            protocol,
            canonical_json_bytes(tampered),
            public_crosswalk,
        )


@pytest.mark.parametrize("platform", ("posix", "windows"))
def test_three_grid_envelope_rejects_identity_replacement_outer_cascade_and_each_old_anchor(
    platform: str,
) -> None:
    protocol = _load_yaml(PROTOCOL_PATH)
    _, envelope, _ = _synthetic_three_grid_envelope(protocol)
    original_identity = cast(dict[str, object], envelope["installed_file_identity"])
    if platform == "windows":
        _convert_installed_identity_to_valid_synthetic_windows(
            original_identity,
            attempt_id="synthetic-r3-envelope",
        )
    assert original_identity["platform"] == platform
    envelope["three_grid_gate_evidence_envelope_sha256"] = _canonical_v1_sha256(
        {
            key: value
            for key, value in envelope.items()
            if key != "three_grid_gate_evidence_envelope_sha256"
        }
    )
    frozen_external_anchors = _synthetic_public_crosswalk_from_envelope(envelope)
    replacement = deepcopy(envelope)
    original_sealed = deepcopy(replacement["sealed_three_grid_gate_evidence"])
    replacement_identity = cast(dict[str, object], replacement["installed_file_identity"])
    changed_field = (
        "windows_file_id_16_bytes_hex_or_null"
        if platform == "windows"
        else "posix_st_ino_decimal_or_null"
    )
    replacement_identity[changed_field] = (
        _file_id_128_identifier_raw_bytes_hex(bytes(range(15, -1, -1)))
        if platform == "windows"
        else "777"
    )
    restored_identity = deepcopy(replacement_identity)
    restored_identity[changed_field] = original_identity[changed_field]
    assert restored_identity == original_identity
    replacement["three_grid_gate_evidence_envelope_sha256"] = _canonical_v1_sha256(
        {
            key: value
            for key, value in replacement.items()
            if key != "three_grid_gate_evidence_envelope_sha256"
        }
    )
    replacement_anchors = _synthetic_public_crosswalk_from_envelope(replacement)

    assert replacement["sealed_three_grid_gate_evidence"] == original_sealed
    assert (
        cast(dict[str, object], replacement["sealed_three_grid_gate_evidence"])[
            "three_grid_gate_evidence_sha256"
        ]
        == cast(dict[str, object], envelope["sealed_three_grid_gate_evidence"])[
            "three_grid_gate_evidence_sha256"
        ]
    )
    assert (
        replacement["three_grid_gate_evidence_envelope_sha256"]
        != envelope["three_grid_gate_evidence_envelope_sha256"]
    )
    with pytest.raises(ValueError, match="public gate/fit-attempt/presence crosswalk"):
        _verify_synthetic_three_grid_envelope(
            protocol,
            canonical_json_bytes(replacement),
            frozen_external_anchors,
        )

    anchor_fields = (
        "snapshot_gate_three_grid_gate_evidence_sha256_or_null",
        "fit_attempt_three_grid_gate_evidence_sha256_or_null",
        "staged_local_presence_map_value",
    )
    for anchor_field in anchor_fields:
        one_stale_anchor = deepcopy(replacement_anchors)
        one_stale_anchor[anchor_field] = frozen_external_anchors[anchor_field]
        with pytest.raises(ValueError, match="public gate/fit-attempt/presence crosswalk"):
            _verify_synthetic_three_grid_envelope(
                protocol,
                canonical_json_bytes(replacement),
                one_stale_anchor,
            )

    verified_only_when_all_external_anchors_are_replaced = _verify_synthetic_three_grid_envelope(
        protocol,
        canonical_json_bytes(replacement),
        replacement_anchors,
    )
    assert (
        verified_only_when_all_external_anchors_are_replaced["envelope_sha256"]
        == replacement["three_grid_gate_evidence_envelope_sha256"]
    )


@pytest.mark.parametrize(
    ("tamper_kind", "error_pattern"),
    (
        ("resolution_order", "50, 25, and 12.5 km in order"),
        ("pair_direction", "exactly 50-to-25 diagnostic"),
        ("snapshot_binding", "sealed snapshot and evaluator return identity disagree"),
        (
            "parameter_binding",
            "selected physical parameters and evaluator parameter identity disagree",
        ),
    ),
)
def test_three_grid_envelope_rejects_cascade_rehashed_source_impossible_bindings(
    tamper_kind: str,
    error_pattern: str,
) -> None:
    protocol = _load_yaml(PROTOCOL_PATH)
    _, envelope, _ = _synthetic_three_grid_envelope(protocol)
    tampered = deepcopy(envelope)
    sealed = cast(dict[str, Any], tampered["sealed_three_grid_gate_evidence"])
    if tamper_kind == "resolution_order":
        resolution_map = cast(
            dict[str, dict[str, object]], tampered["grid_resolution_payload_by_grid_size"]
        )
        resolution_map["25_km"], resolution_map["12_5_km"] = (
            resolution_map["12_5_km"],
            resolution_map["25_km"],
        )
    elif tamper_kind == "pair_direction":
        diagnostic = cast(dict[str, object], sealed["diagnostic_50_to_25"])
        diagnostic["coarse_grid_size_km_hex"] = (40.0).hex()
    elif tamper_kind == "snapshot_binding":
        sealed["snapshot_id"] = "fold_2"
    elif tamper_kind == "parameter_binding":
        sealed["selected_physical_parameters_sha256"] = "f" * 64
    else:  # pragma: no cover - the parameterization is frozen above
        raise AssertionError(tamper_kind)
    attacker_crosswalk = _cascade_rehash_synthetic_three_grid_envelope(protocol, tampered)

    with pytest.raises(ValueError, match=error_pattern):
        _verify_synthetic_three_grid_envelope(
            protocol,
            canonical_json_bytes(tampered),
            attacker_crosswalk,
        )


def test_three_grid_envelope_rejects_missing_extra_and_no_clobber(tmp_path: Path) -> None:
    protocol = _load_yaml(PROTOCOL_PATH)
    _, envelope, public_crosswalk = _synthetic_three_grid_envelope(protocol)

    missing = deepcopy(envelope)
    missing.pop("grid_resolution_payload_by_grid_size")
    with pytest.raises(ValueError, match="missing or extra envelope field"):
        _verify_synthetic_three_grid_envelope(
            protocol,
            canonical_json_bytes(missing),
            public_crosswalk,
        )

    extra = deepcopy(envelope)
    resolution_map = cast(
        dict[str, dict[str, object]], extra["grid_resolution_payload_by_grid_size"]
    )
    resolution_map["50_km"]["unexpected"] = "forbidden"
    with pytest.raises(ValueError, match="missing or extra resolution field"):
        _verify_synthetic_three_grid_envelope(
            protocol,
            canonical_json_bytes(extra),
            public_crosswalk,
        )

    destination = tmp_path / "fold_1.json"
    challenger = tmp_path / ".challenger.tmp"
    original_bytes = canonical_json_bytes(envelope)
    destination.write_bytes(original_bytes)
    challenger.write_bytes(original_bytes)
    with pytest.raises(FileExistsError):
        os.link(challenger, destination)
    assert destination.read_bytes() == original_bytes
    assert destination.stat().st_nlink == 1


def test_three_grid_envelope_detects_same_bytes_replacement_and_hardlink_identity_tamper(
    tmp_path: Path,
) -> None:
    original = tmp_path / "fold_1.json"
    original.write_bytes(b"same canonical bytes")
    expected_identity = _synthetic_installed_file_identity(original, original)
    _assert_installed_file_identity(original, expected_identity)

    replacement = tmp_path / ".replacement.tmp"
    replacement.write_bytes(original.read_bytes())
    replacement_identity = _synthetic_installed_file_identity(replacement, original)
    assert replacement_identity != expected_identity
    os.replace(replacement, original)
    assert original.read_bytes() == b"same canonical bytes"
    with pytest.raises(ValueError, match="file identity changed"):
        _assert_installed_file_identity(original, expected_identity)

    hardlink_target = tmp_path / "fold_2.json"
    hardlink_target.write_bytes(b"hardlink probe")
    hardlink_identity = _synthetic_installed_file_identity(hardlink_target, hardlink_target)
    alias = tmp_path / "fold_2.alias"
    os.link(hardlink_target, alias)
    with pytest.raises(ValueError, match="link count changed"):
        _assert_installed_file_identity(hardlink_target, hardlink_identity)


@pytest.mark.parametrize(
    ("mutation_path", "invalid_value", "error_pattern"),
    (
        (
            ("windows_volume_serial_u64_decimal_or_null",),
            str(1 << 64),
            "Windows installed file identity encoding",
        ),
        (
            ("windows_volume_serial_u64_decimal_or_null",),
            "01",
            "Windows installed file identity encoding",
        ),
        (
            ("windows_file_id_16_bytes_hex_or_null",),
            "A" * 32,
            "Windows installed file identity encoding",
        ),
        (
            ("windows_file_id_16_bytes_hex_or_null",),
            "a" * 31,
            "Windows installed file identity encoding",
        ),
        (
            ("posix_st_dev_decimal_or_null",),
            "1",
            "Windows identity null branch",
        ),
        (
            ("parent_identity", "windows_volume_serial_u64_decimal_or_null"),
            str(1 << 64),
            "Windows parent identity encoding",
        ),
        (
            ("parent_identity", "windows_directory_file_id_16_bytes_hex_or_null"),
            "g" * 32,
            "Windows parent identity encoding",
        ),
        (
            ("parent_identity", "windows_volume_serial_u64_decimal_or_null"),
            "2",
            "parent and file Windows volume serial disagree",
        ),
        (
            ("parent_identity", "exact_case_relative_components"),
            ["local_restricted", "wrong_case"],
            "parent and file exact-case components disagree",
        ),
        (
            ("temp_leaf_utf8_hex",),
            b".fold_1.envelope.tmp".hex().upper(),
            "temp leaf must be canonical lowercase UTF-8 hex",
        ),
        (
            ("final_leaf_utf8_hex",),
            b"fold_1.json".hex() + " ",
            "final leaf must be canonical lowercase UTF-8 hex",
        ),
        (
            ("temp_leaf_utf8_hex",),
            123,
            "temp leaf must be canonical lowercase UTF-8 hex",
        ),
        (
            ("exact_case_relative_components",),
            ["local_restricted", "three_grid_gate_evidence", "e\u0301.json"],
            "final leaf and exact-case components disagree",
        ),
        (
            ("exact_case_relative_components",),
            "local_restricted/three_grid_gate_evidence/fold_1.json",
            "file exact-case components must be an exact 3-item list",
        ),
        (
            ("parent_identity", "exact_case_relative_components"),
            ["three_grid_gate_evidence"],
            "parent exact-case components must be an exact 2-item list",
        ),
        (
            ("canonical_final_path",),
            "/synthetic/three_grid_gate_evidence/fold_1.json",
            "file canonical path must be a canonical absolute Windows volume path",
        ),
        (
            ("canonical_final_path",),
            r"D:\synthetic\three_grid_gate_evidence\fold_1.json",
            "file canonical path must be a canonical absolute Windows volume path",
        ),
        (
            ("canonical_final_path",),
            r"\\?\Volume{00000000}\synthetic\fold_1.json",
            "file canonical path must be a canonical absolute Windows volume path",
        ),
        (
            ("canonical_final_path",),
            (
                f"{SYNTHETIC_WINDOWS_VOLUME_GUID_PREFIX}\\synthetic_workspace"
                r"\\fold_1.json"
            ),
            "file canonical path must contain safe exact-case Windows components",
        ),
        (
            ("canonical_final_path",),
            f"{SYNTHETIC_WINDOWS_VOLUME_GUID_PREFIX}\\synthetic_workspace\\..\\fold_1.json",
            "file canonical path must contain safe exact-case Windows components",
        ),
        (
            ("final_repository_relative_path",),
            123,
            "final repository-relative path must be a string",
        ),
    ),
)
def test_three_grid_envelope_rejects_malformed_windows_file_and_parent_identity(
    mutation_path: tuple[str, ...],
    invalid_value: object,
    error_pattern: str,
) -> None:
    protocol = _load_yaml(PROTOCOL_PATH)
    _, envelope, _ = _synthetic_three_grid_envelope(protocol)
    installed_identity = cast(dict[str, object], envelope["installed_file_identity"])
    _convert_installed_identity_to_valid_synthetic_windows(
        installed_identity,
        attempt_id="synthetic-r3-envelope",
    )
    target: dict[str, object] = installed_identity
    for component in mutation_path[:-1]:
        target = cast(dict[str, object], target[component])
    target[mutation_path[-1]] = invalid_value
    envelope["three_grid_gate_evidence_envelope_sha256"] = _canonical_v1_sha256(
        {
            key: value
            for key, value in envelope.items()
            if key != "three_grid_gate_evidence_envelope_sha256"
        }
    )
    public_crosswalk = _synthetic_public_crosswalk_from_envelope(envelope)
    with pytest.raises(ValueError, match=error_pattern):
        _verify_synthetic_three_grid_envelope(
            protocol,
            canonical_json_bytes(envelope),
            public_crosswalk,
        )


def test_three_grid_identity_component_validator_rejects_non_nfc_input_before_serialization() -> (
    None
):
    with pytest.raises(ValueError, match="safe UTF-8 NFC names"):
        _validate_exact_nfc_components(
            ["local_restricted", "e\u0301"],
            expected_length=2,
            field_name="parent exact-case components",
        )


def test_file_id_128_serialization_preserves_raw_identifier_array_order() -> None:
    identifier = bytes(range(16))
    assert _file_id_128_identifier_raw_bytes_hex(identifier) == ("000102030405060708090a0b0c0d0e0f")
    assert _file_id_128_identifier_raw_bytes_hex(identifier[::-1]) == (
        "0f0e0d0c0b0a09080706050403020100"
    )
    with pytest.raises(ValueError, match="exactly 16 raw bytes"):
        _file_id_128_identifier_raw_bytes_hex(identifier[:-1])


def test_directory_identity_chain_requires_independent_live_capture_receipt() -> None:
    protocol = _load_yaml(PROTOCOL_PATH)
    _, envelope, _ = _synthetic_three_grid_envelope(protocol)
    installed_identity = cast(dict[str, object], envelope["installed_file_identity"])
    expected_chain = cast(
        list[dict[str, object]],
        installed_identity["ordered_directory_identity_chain"],
    )
    for derived_candidate in (
        expected_chain,
        list(expected_chain),
        deepcopy(expected_chain),
    ):
        with pytest.raises(ValueError, match="independent capture receipt"):
            _verify_independently_observed_live_directory_identity_chain(
                installed_identity,
                derived_candidate,
            )
    forged_wrapped_deepcopy = _SyntheticLiveDirectoryIdentityCapture(
        source_kind="independent_live_directory_handle_capture",
        capture_id="forged-wrapped-deepcopy",
        opened_handle_count=len(expected_chain),
        records=tuple(deepcopy(expected_chain)),
        _provenance_capability=object(),
    )
    with pytest.raises(ValueError, match="not the registered original"):
        _verify_independently_observed_live_directory_identity_chain(
            installed_identity,
            forged_wrapped_deepcopy,
        )

    with pytest.raises(ValueError, match="controlled factory"):
        _SyntheticLiveDirectoryCaptureProvider(
            factory_token=object(),
            live_handle_source=cast(_SyntheticLiveDirectoryHandleSource, object()),
        )
    with pytest.raises(ValueError, match="controlled handle source"):
        _SyntheticLiveDirectoryCaptureProvider(
            factory_token=_LIVE_CAPTURE_FACTORY_TOKEN,
            live_handle_source=cast(
                _SyntheticLiveDirectoryHandleSource,
                tuple(deepcopy(expected_chain)),
            ),
        )

    prebind_path_source = _SyntheticLiveDirectoryHandleSource(
        factory_token=_LIVE_CAPTURE_FACTORY_TOKEN,
        attempt_id="synthetic-r3-envelope",
        platform=None,
        path=Path("must-not-be-read-before-binding.json"),
        windows_volume_serial=None,
        windows_parent_file_id=None,
        posix_st_dev=None,
        posix_parent_st_ino=None,
        identity_override=None,
        opened_handle_limit_for_test=None,
    )
    object.__setattr__(prebind_path_source, "_path", None)
    object.__setattr__(prebind_path_source, "_platform", "posix")
    object.__setattr__(prebind_path_source, "_posix_st_dev", "1")
    object.__setattr__(prebind_path_source, "_posix_parent_st_ino", "9")
    with pytest.raises(ValueError, match="configuration changed before binding"):
        _SyntheticLiveDirectoryCaptureProvider(
            factory_token=_LIVE_CAPTURE_FACTORY_TOKEN,
            live_handle_source=prebind_path_source,
        )
    with pytest.raises(ValueError, match="new unbound handle source"):
        _SyntheticLiveDirectoryCaptureProvider(
            factory_token=_LIVE_CAPTURE_FACTORY_TOKEN,
            live_handle_source=prebind_path_source,
        )

    prebind_raw_source = _SyntheticLiveDirectoryHandleSource(
        factory_token=_LIVE_CAPTURE_FACTORY_TOKEN,
        attempt_id="synthetic-r3-envelope",
        platform="posix",
        path=None,
        windows_volume_serial=None,
        windows_parent_file_id=None,
        posix_st_dev="1",
        posix_parent_st_ino="9",
        identity_override=None,
        opened_handle_limit_for_test=None,
    )
    object.__setattr__(prebind_raw_source, "_attempt_id", "forged-before-binding")
    with pytest.raises(ValueError, match="configuration changed before binding"):
        _SyntheticLiveDirectoryCaptureProvider(
            factory_token=_LIVE_CAPTURE_FACTORY_TOKEN,
            live_handle_source=prebind_raw_source,
        )

    class _CopyingLiveHandleSource(_SyntheticLiveDirectoryHandleSource):
        def observe_each_live_directory_handle(
            self,
            *,
            requesting_provider: object,
        ) -> tuple[dict[str, object], ...]:
            del requesting_provider
            return tuple(deepcopy(expected_chain))

    with pytest.raises(ValueError, match="exact controlled factory class"):
        _CopyingLiveHandleSource(
            factory_token=_LIVE_CAPTURE_FACTORY_TOKEN,
            attempt_id="synthetic-r3-envelope",
            platform="posix",
            path=None,
            windows_volume_serial=None,
            windows_parent_file_id=None,
            posix_st_dev="1",
            posix_parent_st_ino="9",
            identity_override=None,
            opened_handle_limit_for_test=None,
        )
    copying_source = object.__new__(_CopyingLiveHandleSource)
    swapped_provider = _synthetic_live_directory_capture_provider(
        attempt_id="synthetic-r3-envelope",
        platform="posix",
        windows_volume_serial=None,
        windows_parent_file_id=None,
        posix_st_dev="1",
        posix_parent_st_ino="9",
    )
    object.__setattr__(swapped_provider, "_live_handle_source", copying_source)
    with pytest.raises(ValueError, match="source changed after construction"):
        swapped_provider.capture_before_evidence_read()
    with pytest.raises(ValueError, match="single-use"):
        swapped_provider.capture_before_evidence_read()

    mutated_source_provider = _synthetic_live_directory_capture_provider_from_path(
        Path("must-not-be-read.json"),
        attempt_id="synthetic-r3-envelope",
    )
    mutated_source = mutated_source_provider._live_handle_source
    object.__setattr__(mutated_source, "_path", None)
    object.__setattr__(mutated_source, "_platform", "posix")
    object.__setattr__(mutated_source, "_posix_st_dev", "1")
    object.__setattr__(mutated_source, "_posix_parent_st_ino", "9")
    with pytest.raises(ValueError, match="configuration changed after binding"):
        mutated_source_provider.capture_before_evidence_read()
    with pytest.raises(ValueError, match="single-use"):
        mutated_source_provider.capture_before_evidence_read()

    class _OverridingCaptureProvider(_SyntheticLiveDirectoryCaptureProvider):
        pass

    overriding_provider = object.__new__(_OverridingCaptureProvider)
    with pytest.raises(ValueError, match="controlled live capture provider"):
        _reopen_and_verify_synthetic_three_grid_envelope(
            protocol,
            Path("must-not-be-opened.json"),
            {},
            live_directory_capture_provider=overriding_provider,
            expected_attempt_id="must-not-be-used",
            expected_snapshot_id="must-not-be-used",
            expected_protocol_sha256="must-not-be-used",
        )

    short_provider = _synthetic_live_directory_capture_provider(
        attempt_id="synthetic-r3-envelope",
        platform="posix",
        windows_volume_serial=None,
        windows_parent_file_id=None,
        posix_st_dev="1",
        posix_parent_st_ino="9",
        opened_handle_limit_for_test=len(expected_chain) - 1,
    )
    short_capture = short_provider.capture_before_evidence_read()
    with pytest.raises(ValueError, match="did not open all handles"):
        _verify_independently_observed_live_directory_identity_chain(
            installed_identity,
            short_capture,
        )
    with pytest.raises(ValueError, match="ID was already consumed"):
        _verify_independently_observed_live_directory_identity_chain(
            installed_identity,
            short_capture,
        )

    mutated_provider = _synthetic_live_directory_capture_provider(
        attempt_id="synthetic-r3-envelope",
        platform="posix",
        windows_volume_serial=None,
        windows_parent_file_id=None,
        posix_st_dev="1",
        posix_parent_st_ino="9",
    )
    mutated_capture = mutated_provider.capture_before_evidence_read()
    mutated_capture.records[0]["posix_st_ino_decimal_or_null"] = "forged-after-issuance"
    with pytest.raises(ValueError, match="changed after issuance"):
        _verify_independently_observed_live_directory_identity_chain(
            installed_identity,
            mutated_capture,
        )
    with pytest.raises(ValueError, match="ID was already consumed"):
        _verify_independently_observed_live_directory_identity_chain(
            installed_identity,
            mutated_capture,
        )

    for field_name, forged_value in (
        ("source_kind", "forged-source-kind"),
        ("capture_id", "forged-capture-id"),
        ("opened_handle_count", len(expected_chain) + 1),
    ):
        scalar_provider = _synthetic_live_directory_capture_provider(
            attempt_id="synthetic-r3-envelope",
            platform="posix",
            windows_volume_serial=None,
            windows_parent_file_id=None,
            posix_st_dev="1",
            posix_parent_st_ino="9",
        )
        scalar_capture = scalar_provider.capture_before_evidence_read()
        assert not hasattr(scalar_capture, "__dict__")
        object.__setattr__(scalar_capture, field_name, forged_value)
        with pytest.raises(ValueError, match="metadata changed after issuance"):
            _verify_independently_observed_live_directory_identity_chain(
                installed_identity,
                scalar_capture,
            )

    provider = _synthetic_live_directory_capture_provider(
        attempt_id="synthetic-r3-envelope",
        platform="posix",
        windows_volume_serial=None,
        windows_parent_file_id=None,
        posix_st_dev="1",
        posix_parent_st_ino="9",
    )
    independent_capture = provider.capture_before_evidence_read()
    bound_source = provider._live_handle_source
    stolen_capability_wrapper = _SyntheticLiveDirectoryIdentityCapture(
        source_kind=independent_capture.source_kind,
        capture_id=independent_capture.capture_id,
        opened_handle_count=independent_capture.opened_handle_count,
        records=tuple(deepcopy(independent_capture.records)),
        _provenance_capability=independent_capture._provenance_capability,
    )
    with pytest.raises(ValueError, match="not the registered original"):
        _verify_independently_observed_live_directory_identity_chain(
            installed_identity,
            stolen_capability_wrapper,
        )
    _verify_independently_observed_live_directory_identity_chain(
        installed_identity,
        independent_capture,
    )
    with pytest.raises(ValueError, match="ID was already consumed"):
        _verify_independently_observed_live_directory_identity_chain(
            installed_identity,
            independent_capture,
        )
    with pytest.raises(AttributeError):
        object.__setattr__(provider, "_used", False)
    with pytest.raises(AttributeError):
        object.__setattr__(bound_source, "_used", False)
    with pytest.raises(ValueError, match="single-use"):
        bound_source.observe_each_live_directory_handle(requesting_provider=provider)
    with pytest.raises(ValueError, match="single-use"):
        provider.capture_before_evidence_read()


@pytest.mark.parametrize("platform", ("posix", "windows"))
def test_directory_identity_chain_ancestor_replacement_reanchors_outer_sha(
    platform: str,
) -> None:
    protocol = _load_yaml(PROTOCOL_PATH)
    _, envelope, _ = _synthetic_three_grid_envelope(protocol)
    installed_identity = cast(dict[str, object], envelope["installed_file_identity"])
    if platform == "windows":
        _convert_installed_identity_to_valid_synthetic_windows(
            installed_identity,
            attempt_id="synthetic-r3-envelope",
        )
    envelope["three_grid_gate_evidence_envelope_sha256"] = _canonical_v1_sha256(
        {
            key: value
            for key, value in envelope.items()
            if key != "three_grid_gate_evidence_envelope_sha256"
        }
    )
    frozen_anchors = _synthetic_public_crosswalk_from_envelope(envelope)
    replacement = deepcopy(envelope)
    replacement_identity = cast(dict[str, object], replacement["installed_file_identity"])
    replacement_chain = cast(
        list[dict[str, object]],
        replacement_identity["ordered_directory_identity_chain"],
    )
    target_record = replacement_chain[3]
    changed_field = (
        "windows_directory_file_id_16_bytes_hex_or_null"
        if platform == "windows"
        else "posix_st_ino_decimal_or_null"
    )
    original_value = target_record[changed_field]
    target_record[changed_field] = (
        _file_id_128_identifier_raw_bytes_hex(bytes(range(15, -1, -1)))
        if platform == "windows"
        else "7777"
    )
    restored_chain = deepcopy(replacement_chain)
    restored_chain[3][changed_field] = original_value
    assert restored_chain == installed_identity["ordered_directory_identity_chain"]
    replacement_identity["ordered_directory_identity_chain_sha256"] = _canonical_v1_sha256(
        replacement_chain
    )
    replacement["three_grid_gate_evidence_envelope_sha256"] = _canonical_v1_sha256(
        {
            key: value
            for key, value in replacement.items()
            if key != "three_grid_gate_evidence_envelope_sha256"
        }
    )
    replacement_anchors = _synthetic_public_crosswalk_from_envelope(replacement)
    assert (
        replacement["sealed_three_grid_gate_evidence"]
        == envelope["sealed_three_grid_gate_evidence"]
    )
    assert (
        replacement["three_grid_gate_evidence_envelope_sha256"]
        != envelope["three_grid_gate_evidence_envelope_sha256"]
    )
    with pytest.raises(ValueError, match="public gate/fit-attempt/presence crosswalk"):
        _verify_synthetic_three_grid_envelope(
            protocol,
            canonical_json_bytes(replacement),
            frozen_anchors,
        )
    _verify_synthetic_three_grid_envelope(
        protocol,
        canonical_json_bytes(replacement),
        replacement_anchors,
    )


def test_directory_identity_chain_rejects_stale_hash_missing_record_and_wrong_order() -> None:
    protocol = _load_yaml(PROTOCOL_PATH)
    _, envelope, public_crosswalk = _synthetic_three_grid_envelope(protocol)
    for mutation in ("stale_hash", "missing", "wrong_order"):
        tampered = deepcopy(envelope)
        identity = cast(dict[str, object], tampered["installed_file_identity"])
        chain = cast(list[dict[str, object]], identity["ordered_directory_identity_chain"])
        if mutation == "stale_hash":
            chain[2]["posix_st_ino_decimal_or_null"] = "8888"
        elif mutation == "missing":
            chain.pop(2)
            identity["ordered_directory_identity_chain_sha256"] = _canonical_v1_sha256(chain)
        else:
            chain[1], chain[2] = chain[2], chain[1]
            identity["ordered_directory_identity_chain_sha256"] = _canonical_v1_sha256(chain)
        tampered["three_grid_gate_evidence_envelope_sha256"] = _canonical_v1_sha256(
            {
                key: value
                for key, value in tampered.items()
                if key != "three_grid_gate_evidence_envelope_sha256"
            }
        )
        tampered_crosswalk = _synthetic_public_crosswalk_from_envelope(tampered)
        with pytest.raises(ValueError, match="directory identity chain"):
            _verify_synthetic_three_grid_envelope(
                protocol,
                canonical_json_bytes(tampered),
                tampered_crosswalk if mutation != "stale_hash" else public_crosswalk,
            )


@pytest.mark.parametrize("platform", ("posix", "windows"))
def test_fresh_checkpoint_rejects_independent_live_middle_ancestor_identity_drift(
    tmp_path: Path,
    platform: str,
) -> None:
    protocol = _load_yaml(PROTOCOL_PATH)
    temp_path = tmp_path / f".{platform}.fold_1.envelope.tmp"
    final_path = tmp_path / "fold_1.json"
    with temp_path.open("xb"):
        pass
    if platform == "windows":
        installed_identity = _synthetic_installed_file_identity(temp_path, final_path)
        _convert_installed_identity_to_valid_synthetic_windows(
            installed_identity,
            attempt_id="synthetic-r3-envelope",
        )
        gate, envelope, public_crosswalk = _synthetic_three_grid_envelope(
            protocol,
            installed_identity,
        )
    else:
        gate, envelope, public_crosswalk = _synthetic_three_grid_envelope(protocol)
        installed_identity = cast(dict[str, object], envelope["installed_file_identity"])
    frozen_envelope_bytes = canonical_json_bytes(envelope)
    frozen_outer_sha = cast(str, envelope["three_grid_gate_evidence_envelope_sha256"])
    frozen_anchors = deepcopy(public_crosswalk)
    with temp_path.open("r+b") as handle:
        handle.write(frozen_envelope_bytes)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temp_path, final_path)
    temp_path.unlink()

    changed_field = (
        "windows_directory_file_id_16_bytes_hex_or_null"
        if platform == "windows"
        else "posix_st_ino_decimal_or_null"
    )
    changed_value = "ffffffffffffffffffffffffffffffff" if platform == "windows" else "999999999"
    drifted_live_capture_provider = _synthetic_live_directory_capture_provider(
        attempt_id="synthetic-r3-envelope",
        platform=platform,
        windows_volume_serial="1" if platform == "windows" else None,
        windows_parent_file_id=(
            _file_id_128_identifier_raw_bytes_hex(bytes(range(16, 32)))
            if platform == "windows"
            else None
        ),
        posix_st_dev="1" if platform == "posix" else None,
        posix_parent_st_ino="9" if platform == "posix" else None,
        identity_override=(3, changed_field, changed_value),
    )

    with pytest.raises(ValueError, match="observed live directory identity chain drifted"):
        _reopen_and_verify_synthetic_three_grid_envelope(
            protocol,
            final_path,
            frozen_anchors,
            live_directory_capture_provider=drifted_live_capture_provider,
            expected_attempt_id="synthetic-r3-envelope",
            expected_snapshot_id="fold_1",
            expected_protocol_sha256=gate.protocol_sha256,
        )
    assert final_path.read_bytes() == frozen_envelope_bytes
    assert envelope["three_grid_gate_evidence_envelope_sha256"] == frozen_outer_sha
    assert public_crosswalk == frozen_anchors


@pytest.mark.parametrize(
    "present_snapshots",
    (
        frozenset(("fold_1", "fold_3", "final_validation")),
        frozenset(SNAPSHOT_ORDER),
    ),
    ids=("mixed_present_null", "all_present"),
)
def test_three_grid_presence_map_mixed_round_trip_and_exact_file_set(
    tmp_path: Path,
    present_snapshots: frozenset[str],
) -> None:
    protocol = _load_yaml(PROTOCOL_PATH)
    attempt_id = "synthetic-r3-five-snapshot-presence"
    protocol_sha256 = "a" * 64
    evidence_directory = (
        tmp_path / "attempts" / attempt_id / "local_restricted" / "three_grid_gate_evidence"
    )
    evidence_directory.mkdir(parents=True)
    public_crosswalk_by_snapshot = {
        snapshot_id: _null_three_grid_public_crosswalk() for snapshot_id in SNAPSHOT_ORDER
    }
    installed_identity_by_snapshot: dict[str, dict[str, object]] = {}
    outer_sha_by_snapshot: dict[str, str] = {}

    for snapshot_id in SNAPSHOT_ORDER:
        if snapshot_id not in present_snapshots:
            continue
        temp_path = evidence_directory / f".{snapshot_id}.envelope.tmp"
        final_path = evidence_directory / f"{snapshot_id}.json"
        with temp_path.open("xb"):
            pass
        installed_identity = _synthetic_installed_file_identity(
            temp_path,
            final_path,
            attempt_id=attempt_id,
        )
        _, envelope, public_crosswalk = _synthetic_three_grid_envelope(
            protocol,
            installed_identity,
            attempt_id=attempt_id,
            snapshot_id=snapshot_id,
            protocol_sha256=protocol_sha256,
        )
        envelope_bytes = canonical_json_bytes(envelope)
        with temp_path.open("r+b") as handle:
            handle.write(envelope_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temp_path, final_path)
        temp_path.unlink()
        installed_identity_by_snapshot[snapshot_id] = installed_identity
        public_crosswalk_by_snapshot[snapshot_id] = public_crosswalk
        outer_sha_by_snapshot[snapshot_id] = cast(
            str,
            envelope["three_grid_gate_evidence_envelope_sha256"],
        )

    presence_map, presence_map_sha256 = _verify_synthetic_three_grid_presence_file_set(
        protocol,
        evidence_directory,
        public_crosswalk_by_snapshot,
        installed_identity_by_snapshot,
        expected_attempt_id=attempt_id,
        expected_protocol_sha256=protocol_sha256,
    )
    assert list(presence_map) == list(SNAPSHOT_ORDER)
    assert presence_map == {
        snapshot_id: outer_sha_by_snapshot.get(snapshot_id) for snapshot_id in SNAPSHOT_ORDER
    }
    assert presence_map_sha256 == _canonical_v1_sha256(presence_map)

    extra_path = evidence_directory / "unexpected.json"
    extra_path.write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="file set does not match presence projection"):
        _verify_synthetic_three_grid_presence_file_set(
            protocol,
            evidence_directory,
            public_crosswalk_by_snapshot,
            installed_identity_by_snapshot,
            expected_attempt_id=attempt_id,
            expected_protocol_sha256=protocol_sha256,
        )
    extra_path.unlink()

    missing_snapshot = next(iter(sorted(present_snapshots)))
    (evidence_directory / f"{missing_snapshot}.json").unlink()
    with pytest.raises(ValueError, match="file set does not match presence projection"):
        _verify_synthetic_three_grid_presence_file_set(
            protocol,
            evidence_directory,
            public_crosswalk_by_snapshot,
            installed_identity_by_snapshot,
            expected_attempt_id=attempt_id,
            expected_protocol_sha256=protocol_sha256,
        )


def test_three_grid_indeterminate_after_install_is_terminal() -> None:
    protocol = _load_yaml(PROTOCOL_PATH)
    state_machine = protocol["qualification"]["three_grid_gate_evidence_protocol"][
        "local_restricted_persistence"
    ]["durable_create_once_file_protocol"]["install_failure_state_machine"]
    indeterminate_action = state_machine[
        "install_success_until_complete_postinstall_flush_parent_sync_and_reopen_verification"
    ]
    for terminal_requirement in (
        "indeterminate_after_install",
        "read_only_forensic_reopen_only",
        "no_delete_rollback_overwrite_or_same_final_retry",
        "same_attempt_may_never_resume_reanchor_publish_or_advance_qualification",
        "new_attempt_id_only_after_manual_audit",
    ):
        assert terminal_requirement in indeterminate_action


def test_three_grid_pre_anchor_restart_may_not_adopt_or_reanchor_installed_file(
    tmp_path: Path,
) -> None:
    protocol = _load_yaml(PROTOCOL_PATH)
    attempt_id = "synthetic-r3-pre-anchor-crash"
    evidence_directory = tmp_path / "three_grid_gate_evidence"
    evidence_directory.mkdir()
    _, envelope, _ = _synthetic_three_grid_envelope(
        protocol,
        attempt_id=attempt_id,
        snapshot_id="fold_1",
    )
    (evidence_directory / "fold_1.json").write_bytes(canonical_json_bytes(envelope))
    all_null_crosswalks = {
        snapshot_id: _null_three_grid_public_crosswalk() for snapshot_id in SNAPSHOT_ORDER
    }
    with pytest.raises(ValueError, match="file set does not match presence projection"):
        _verify_synthetic_three_grid_presence_file_set(
            protocol,
            evidence_directory,
            all_null_crosswalks,
            {},
            expected_attempt_id=attempt_id,
            expected_protocol_sha256="a" * 64,
        )

    state_machine = protocol["qualification"]["three_grid_gate_evidence_protocol"][
        "local_restricted_persistence"
    ]["durable_create_once_file_protocol"]["install_failure_state_machine"]
    assert state_machine[
        "complete_postinstall_reopen_verification_before_first_external_envelope_sha_anchor_"
        "is_durably_persisted"
    ].startswith("any_crash_or_failure_is_invalid_execution")


def test_r3_is_a_minimal_engineering_erratum_from_exact_r2_tag_and_commit() -> None:
    current = _load_yaml(PROTOCOL_PATH)
    r2 = _load_yaml_at_revision(R2_PROTOCOL_COMMIT, PROTOCOL_PATH)

    assert current["revision_base"]["protocol_tag"] == R2_PROTOCOL_TAG
    assert current["revision_base"]["protocol_tag_object"] == R2_PROTOCOL_TAG_OBJECT
    assert current["revision_base"]["protocol_commit"] == R2_PROTOCOL_COMMIT
    assert (
        current["qualification"]["per_snapshot_conjunctive_requirements"]
        == r2["qualification"]["per_snapshot_conjunctive_requirements"]
    )
    assert (
        current["qualification_execution_seal"][
            "qualification_result_tag_diff_from_repair_code_tag"
        ]
        == r2["qualification_execution_seal"]["qualification_result_tag_diff_from_repair_code_tag"]
    )

    missing = object()
    differences: set[tuple[str, ...]] = set()

    def collect_differences(
        left: object,
        right: object,
        path: tuple[str, ...] = (),
    ) -> None:
        if left is missing or right is missing:
            differences.add(path)
            return
        if isinstance(left, dict) and isinstance(right, dict):
            for key in set(left) | set(right):
                collect_differences(
                    left.get(key, missing),
                    right.get(key, missing),
                    (*path, str(key)),
                )
            return
        if left != right:
            differences.add(path)

    collect_differences(current, r2)
    responsibility_paths = {
        (
            "repair_code_scope_from_protocol_tag",
            "new_repair_module_execution_rules",
            module_path,
            "allowed_responsibilities",
        )
        for module_path in (
            "src/seismoflux/background/etas_numerical_repair_io.py",
            "src/seismoflux/background/etas_numerical_repair_evidence.py",
        )
    }
    hash_source_semantic_paths = {
        (
            "qualification",
            "three_grid_gate_evidence_protocol",
            "evidence_payload_is_local_restricted_but_its_sha256_is_public_in_same_snapshot_gate_and_fit_attempt",
        ),
        (
            "qualification",
            "three_grid_gate_evidence_protocol",
            "embedded_13_field_evidence_payload_is_local_restricted_and_its_own_sha256_is_not_an_external_identity_anchor",
        ),
        (
            "qualification",
            "three_grid_gate_evidence_protocol",
            "complete_six_field_envelope_own_sha256_is_public_in_same_snapshot_gate_fit_attempt_and_presence_map_without_changing_field_names_types_paths_or_file_counts",
        ),
        (
            "content_addressing",
            "identities",
            "three_grid_gate_evidence_sha256",
            "public_embedding",
        ),
        (
            "content_addressing",
            "identities",
            "three_grid_gate_evidence_sha256",
            "exact_bytes",
        ),
        (
            "content_addressing",
            "identities",
            "three_grid_gate_evidence_sha256",
            "includes_ref",
        ),
        (
            "content_addressing",
            "identities",
            "three_grid_gate_evidence_sha256",
            "fields_must_equal_evidence_fields_excluding_own_sha256",
        ),
        (
            "content_addressing",
            "identities",
            "three_grid_gate_evidence_sha256",
            "fields_must_equal_complete_envelope_fields_excluding_envelope_own_sha256",
        ),
        (
            "content_addressing",
            "identities",
            "three_grid_gate_evidence_sha256",
            "embedded_13_field_own_sha256_remains_recomputed_inside_envelope_but_is_not_the_public_crosswalk_value",
        ),
        (
            "outputs",
            "canonical_nested_schemas",
            "three_grid_gate_evidence_presence_and_sha256_by_snapshot",
            "R3_present_value_exact_source",
        ),
        (
            "outputs",
            "public_qualification_manifest_schema",
            "snapshot_gate_result_derivation_and_crosswalk",
            "three_grid_gate_evidence_sha256_exact_source",
        ),
    }
    root_binding_semantic_paths = {
        (
            "qualification_execution_seal",
            "opening_seal",
            "repository_requirements",
            "fixed_workspace_root_argument_must_open_the_exact_worktree_whose_HEAD_upstream_"
            "remote_code_tag_and_protocol_package_blobs_are_verified_here",
        ),
        (
            "qualification_execution_seal",
            "opening_seal",
            "repository_requirements",
            "absolute_workspace_root_path_remains_local_restricted_and_may_not_enter_"
            "opening_or_public_artifacts",
        ),
        (
            "qualification_execution_seal",
            "optimizer_runtime_code_seal",
            "isolated_launcher_contract",
            "qualification_local_restricted_workspace_root_cli_option_exact",
        ),
        (
            "qualification_execution_seal",
            "optimizer_runtime_code_seal",
            "isolated_launcher_contract",
            "workspace_root_option_must_occur_exactly_once_and_equal_the_fixed_root_"
            "argument_contract_before_any_project_import_or_evidence_open",
        ),
        (
            "qualification_execution_seal",
            "optimizer_runtime_code_seal",
            "isolated_launcher_contract",
            "workspace_root_argument_must_not_come_from_environment_current_directory_or_envelope",
        ),
    }
    exact_allowed_paths = {
        ("protocol_revision",),
        ("revision_reason",),
        ("revision_base",),
        ("publication", "protocol_tag"),
        ("repair_code_scope_from_protocol_tag", "comparison_base"),
        (
            "qualification",
            "three_grid_gate_evidence_protocol",
            "local_restricted_persistence",
        ),
        *hash_source_semantic_paths,
        *root_binding_semantic_paths,
        *responsibility_paths,
    }
    assert differences == exact_allowed_paths

    grid_protocol = current["qualification"]["three_grid_gate_evidence_protocol"]
    assert grid_protocol[
        "embedded_13_field_evidence_payload_is_local_restricted_and_its_own_sha256_is_not_an_external_identity_anchor"
    ]
    assert grid_protocol[
        "complete_six_field_envelope_own_sha256_is_public_in_same_snapshot_gate_fit_attempt_and_presence_map_without_changing_field_names_types_paths_or_file_counts"
    ]
    assert (
        "evidence_payload_is_local_restricted_but_its_sha256_is_public_in_same_snapshot_gate_and_fit_attempt"
        not in grid_protocol
    )
    three_grid_identity = current["content_addressing"]["identities"][
        "three_grid_gate_evidence_sha256"
    ]
    assert three_grid_identity["public_embedding"] == (
        "complete_envelope_own_sha256_only_in_same_snapshot_gate_fit_attempt_and_presence_map"
    )
    assert three_grid_identity["exact_bytes"] == (
        "canonical_json_v1_of_envelope_identity_fields_exact"
    )
    assert three_grid_identity["includes_ref"] == (
        "qualification.three_grid_gate_evidence_protocol.local_restricted_persistence."
        "envelope_identity_fields_exact"
    )
    persistence = current["qualification"]["three_grid_gate_evidence_protocol"][
        "local_restricted_persistence"
    ]
    assert (
        persistence["envelope_identity_fields_exact"] == persistence["envelope_fields_exact"][:-1]
    )
    assert three_grid_identity[
        "embedded_13_field_own_sha256_remains_recomputed_inside_envelope_but_is_not_the_public_crosswalk_value"
    ]
    presence_schema = current["outputs"]["canonical_nested_schemas"][
        "three_grid_gate_evidence_presence_and_sha256_by_snapshot"
    ]
    assert presence_schema["R3_present_value_exact_source"] == (
        "same_snapshot_complete_envelope_three_grid_gate_evidence_envelope_sha256"
    )

    for path in root_binding_semantic_paths:
        current_value: Any = current
        for component in path:
            current_value = current_value[component]
        if path[-1] == "qualification_local_restricted_workspace_root_cli_option_exact":
            assert current_value == "--workspace-root"
        else:
            assert current_value is True

    responsibility = (
        "complete_three_grid_local_restricted_create_once_persistence_and_disk_reopen_validation"
    )
    for path in responsibility_paths:
        current_values: Any = current
        r2_values: Any = r2
        for component in path:
            current_values = current_values[component]
            r2_values = r2_values[component]
        assert current_values == [*r2_values, responsibility]

    assert (
        current["qualification_execution_seal"]["qualification_public_result_staging"]
        == r2["qualification_execution_seal"]["qualification_public_result_staging"]
    )


def test_issue_forecast_seed_context_exact_bytes_and_reference_vector() -> None:
    forecast = _load_yaml(PROTOCOL_PATH)["adapter_contract"]["issue_forecast_definition"]
    seed = forecast["seed_context_contract"]
    reference = seed["reference_vector"]
    context = SeedContext(
        root_seed=reference["root_seed"],
        protocol_version=reference["protocol_version"],
        namespace=reference["namespace"],
        model_id=reference["model_id"],
        issue_id=reference["issue_id"],
        replicate_index=reference["replicate_index"],
    )
    assert list(context.fields()) == reference["expected_fields"]
    payload = b"\x00".join(field.encode("utf-8") for field in context.fields())
    assert hashlib.sha256(payload).hexdigest() == reference["digest_sha256"]
    assert str(context.entropy()) == reference["entropy_uint128_decimal"]
    assert f"{context.entropy():032x}" == reference["entropy_uint128_hex"]
    assert [float(value).hex() for value in context.generator().random(4)] == reference[
        "pcg64_first_four_uniform_float64_hex"
    ]
    assert seed["fields_in_exact_order"] == [
        "literal_seismoflux",
        "root_seed_base10_without_leading_zero",
        "protocol_version",
        "namespace",
        "model_id",
        "issue_id",
        "replicate_index_zero_padded_eight_decimal_digits",
    ]
    assert seed["replicate_output_order"] == "replicate_index_ascending"
    assert seed["grid_horizon_query_node_or_output_bin_changes_must_not_change_seed_context_digest"]

    local_issue = datetime(2020, 1, 1, tzinfo=ZoneInfo("Asia/Shanghai"))
    assert local_issue.astimezone(UTC) == datetime(2019, 12, 31, 16, tzinfo=UTC)


def test_adapter_artifacts_require_clean_remote_frozen_adapter_code() -> None:
    protocol = _load_yaml(PROTOCOL_PATH)
    scope = protocol["adapter_code_scope_from_positive_qualification_result_tag"]
    assert scope[
        "qualification_result_artifact_paths_must_be_blob_identical_to_positive_result_tag"
    ]
    assert scope["protocol_package_paths_must_be_blob_identical_to_protocol_tag"]
    assert scope["any_other_tracked_path_change_forbidden"] is True
    code_payload = scope["adapter_code_payload_identity"]
    assert code_payload["schema_version_exact"] == 1
    assert code_payload["field_types"] == {
        "schema_version": "strict_base10_integer",
        "adapter_code_tag_commit": "lowercase_git_commit_oid_length_40",
        "ordered_path_records": "ordered_list",
        "adapter_code_blob_sha256": "lowercase_hex_length_64",
    }
    assert code_payload["ordered_path_records_item_field_types"] == {
        "repository_relative_path": "exact_allowlisted_forward_slash_repository_relative_path",
        "git_blob_oid": "lowercase_git_blob_oid_length_40",
        "file_sha256": "lowercase_hex_length_64",
        "file_size": "nonnegative_strict_base10_integer",
    }
    assert code_payload["ordered_paths_exact_ref"] == (
        "adapter_code_scope_from_positive_qualification_result_tag.allowed_changed_or_added_paths"
    )
    assert code_payload["ordered_path_records_item_fields_exact"] == [
        "repository_relative_path",
        "git_blob_oid",
        "file_sha256",
        "file_size",
    ]
    assert code_payload["adapter_code_blob_sha256_formula"] == (
        "sha256_seismoflux_canonical_json_v1_of_schema_version_adapter_code_tag_commit_"
        "and_complete_ordered_path_records"
    )
    seal = protocol["adapter_artifact_execution_seal"]
    attempt = seal["attempt_protocol"]
    assert attempt["adapter_artifact_attempt_id_required"]
    assert attempt["independent_gitignored_staging_directory_per_attempt"]
    assert attempt["interrupted_attempt_directory_and_failure_receipt_must_be_retained"]
    assert attempt["overwrite_delete_or_reuse_prior_attempt_directory_forbidden"]
    assert attempt["only_first_complete_protocol_valid_attempt_may_be_published"]
    ledger = attempt["append_only_attempt_ledger"]
    assert ledger["event_type_values"] == ["intent", "ready_to_close", "completed", "aborted"]
    assert set(ledger["event_type_required_null_state_matrix"]) == {
        "intent",
        "ready_to_close",
        "completed",
        "aborted",
    }
    assert ledger["event_type_required_null_state_matrix"]["completed"]["protocol_valid_exact"]
    assert (
        ledger["event_type_required_null_state_matrix"]["aborted"]["protocol_valid_exact"] is False
    )
    assert set(ledger["aborted_failure_phase_sha_matrix"]) == {
        "before_opening",
        "after_opening_before_ready",
        "after_ready_before_closing",
        "after_closing_before_completed",
    }
    assert ledger["completed_requires_ready_to_close_immediately_before_it"]
    assert ledger["aborted_predecessor_and_sha_state_must_match_declared_failure_phase"]
    assert ledger["top_level_fields_exact"] == [
        "schema_version",
        "entries",
        "ledger_content_sha256",
    ]
    assert ledger["first_entry_sequence_exact"] == 0
    assert ledger[
        "every_subsequent_previous_entry_sha256_must_equal_immediately_preceding_entry_recomputed_sha256"
    ]
    empty_ledger_reference = ledger["empty_ledger_reference_vector"]
    assert (
        _canonical_payload_sha256(empty_ledger_reference["canonical_payload"])
        == (empty_ledger_reference["ledger_content_sha256"])
    )
    opening = seal["opening_seal"]
    repository = opening["repository_requirements"]
    for required in (
        "worktree_clean",
        "head_equals_adapter_code_tag_commit",
        "upstream_commit_equals_head",
        "remote_adapter_code_tag_resolves_to_head_and_is_verified_with_network",
        "remote_positive_qualification_result_tag_and_commit_verified",
        "adapter_source_and_test_blob_oids_equal_adapter_code_tag",
        "protocol_package_blob_oids_equal_protocol_tag",
        "adapter_code_diff_from_positive_result_tag_matches_exact_allowlist_and_receipt",
        "qualification_result_artifact_blob_oids_equal_positive_result_tag",
        "repair_and_core_dependency_blob_oids_equal_positive_result_tag",
        "every_other_tracked_path_is_unchanged_or_an_exact_allowed_adapter_path",
    ):
        assert repository[required] is True
    assert repository["remote_repository_slug"] == "Justin-147/SeismoFlux"
    runtime = seal["artifact_runtime_preflight"]
    assert runtime[
        "created_after_attempt_intent_fsync_and_before_opening_seal_or_any_local_or_public_artifact_generation"
    ]
    assert (
        runtime[
            "target_real_catalog_fit_input_event_row_anomaly_score_or_historical_result_access_allowed"
        ]
        is False
    )
    assert runtime["expected_shapely_distribution_version"] == "2.1.2"
    assert runtime["schema_version_exact"] == 1
    assert runtime["scalar_field_types"] == {
        "schema_version": "strict_base10_integer",
        "checked_at_utc": "canonical_RFC3339_UTC_instant_with_Z",
        "adapter_artifact_attempt_id": "nonempty_unicode_NFC_string",
        "adapter_code_tag_commit": "lowercase_git_commit_oid_length_40",
        "adapter_code_blob_sha256": "lowercase_hex_length_64",
        "environment_lock_sha256": "lowercase_hex_length_64",
        "shapely_dist_info_RECORD_sha256": "lowercase_hex_length_64",
        "shapely_complete_installed_distribution_verification_map_sha256": (
            "lowercase_hex_length_64"
        ),
        "shapely_geos_loaded_native_dependency_file_sha256_and_size_map_sha256": (
            "lowercase_hex_length_64"
        ),
        "adapter_runtime_file_sha256_and_size_map_sha256": "lowercase_hex_length_64",
        "adapter_geometry_callable_dependency_map_sha256": "lowercase_hex_length_64",
        "adapter_artifact_runtime_seal_sha256": "lowercase_hex_length_64",
    }
    assert {
        "adapter_code_blob_sha256",
        "isolated_launcher_identity",
        "synthetic_adapter_geometry_warmup_receipt",
        "shapely_complete_installed_distribution_verification_map",
        "shapely_complete_installed_distribution_verification_map_sha256",
        "shapely_geos_loaded_native_dependency_file_sha256_and_size_map",
        "shapely_geos_loaded_native_dependency_file_sha256_and_size_map_sha256",
        "adapter_runtime_file_sha256_and_size_map",
        "adapter_runtime_file_sha256_and_size_map_sha256",
        "adapter_geometry_callable_dependency_map",
        "adapter_geometry_callable_dependency_map_sha256",
        "qualification_runtime_shared_identity_crosswalk",
        "adapter_artifact_runtime_seal_sha256",
    } <= set(runtime["fields_exact"])
    assert runtime["synthetic_adapter_geometry_warmup"]["receipt_schema_version_exact"] == 1
    assert (
        "isolated_launcher_identity"
        in runtime["qualification_runtime_shared_identity_fields_exact"]
    )
    nested_runtime_object_fields = {
        "isolated_launcher_identity",
        "synthetic_adapter_geometry_warmup_receipt",
        "windows_runtime_identity",
        "python_implementation_version_abi_platform_and_executable_sha256",
        "shapely_distribution_name_and_version",
        "shapely_complete_installed_distribution_verification_map",
        "shapely_geos_loaded_native_dependency_file_sha256_and_size_map",
        "adapter_runtime_file_sha256_and_size_map",
        "adapter_geometry_callable_dependency_map",
        "qualification_runtime_shared_identity_crosswalk",
    }
    assert set(runtime["scalar_field_types"]).isdisjoint(nested_runtime_object_fields)
    assert set(runtime["scalar_field_types"]) | nested_runtime_object_fields == set(
        runtime["fields_exact"]
    )
    adapter_runtime_schema = protocol["outputs"]["canonical_nested_schemas"][
        "adapter_artifact_runtime_seal"
    ]
    assert adapter_runtime_schema["scalar_field_types_ref"] == (
        "adapter_artifact_execution_seal.artifact_runtime_preflight.scalar_field_types"
    )
    assert adapter_runtime_schema["python_runtime_identity_schema_ref"] == (
        "outputs.canonical_nested_schemas.python_runtime_identity"
    )
    shared_identity_crosswalk_schema = protocol["outputs"]["canonical_nested_schemas"][
        "qualification_runtime_shared_identity_crosswalk"
    ]
    assert shared_identity_crosswalk_schema["field_types"] == {
        "positive_qualification_optimizer_runtime_code_seal_sha256": ("lowercase_hex_length_64"),
        "shared_field_names_exact": "ordered_list_of_exact_runtime_field_names",
        "qualification_runtime_shared_identity_crosswalk_sha256": ("lowercase_hex_length_64"),
    }
    assert runtime[
        "same_isolated_launcher_and_single_thread_environment_contract_as_qualification_runtime_required"
    ]
    assert runtime["synthetic_adapter_geometry_warmup"]["operation_order_exact"] == [
        "construct_fixed_box_and_point",
        "fixed_buffer",
        "normalize_geometry",
        "serialize_big_endian_2D_WKB_without_SRID",
        "deserialize_WKB",
        "compare_roundtrip_geometry_equals_normalized_buffer",
        "covers_including_boundary",
    ]
    warmup = runtime["synthetic_adapter_geometry_warmup"]
    assert warmup["receipt_fields_exact"] == [
        "schema_version",
        "operation_order",
        "canonical_input_payload",
        "canonical_input_payload_sha256",
        "canonical_output_payload",
        "canonical_output_payload_sha256",
        "shapely_runtime_binding_identity",
        "synthetic_adapter_geometry_warmup_receipt_sha256",
    ]
    assert warmup["receipt_field_types"] == {
        "schema_version": "strict_base10_integer",
        "operation_order": "ordered_list_of_nonempty_unicode_NFC_strings",
        "canonical_input_payload": "strict_object",
        "canonical_input_payload_sha256": "lowercase_hex_length_64",
        "canonical_output_payload": "strict_object",
        "canonical_output_payload_sha256": "lowercase_hex_length_64",
        "shapely_runtime_binding_identity": "strict_object",
        "synthetic_adapter_geometry_warmup_receipt_sha256": "lowercase_hex_length_64",
    }
    expected_warmup_input = {
        "box_min_max_xy_python_float_hex": [
            "0x0.0p+0",
            "0x0.0p+0",
            "0x1.0000000000000p+1",
            "0x1.0000000000000p+1",
        ],
        "point_xy_python_float_hex": [
            "0x1.0000000000000p+0",
            "0x1.0000000000000p+0",
        ],
        "buffer_distance_python_float_hex": "0x1.0000000000000p+0",
        "buffer_quad_segs": 8,
        "wkb_byte_order": 0,
        "wkb_output_dimension": 2,
        "wkb_include_srid": False,
        "wkb_flavor": "extended",
    }
    assert warmup["canonical_input_payload_exact"] == expected_warmup_input
    assert list(expected_warmup_input) == warmup["canonical_input_payload_fields_exact"]
    assert warmup["canonical_input_payload_field_types"] == {
        "box_min_max_xy_python_float_hex": ("exact_length_4_ordered_python_float64_hex_list"),
        "point_xy_python_float_hex": "exact_length_2_ordered_python_float64_hex_list",
        "buffer_distance_python_float_hex": "finite_python_float64_hex",
        "buffer_quad_segs": "positive_strict_base10_integer",
        "wkb_byte_order": "strict_base10_integer",
        "wkb_output_dimension": "strict_base10_integer",
        "wkb_include_srid": "strict_boolean",
        "wkb_flavor": "nonempty_unicode_NFC_string",
    }
    assert warmup["canonical_output_payload_field_types"] == {
        "normalized_buffer_big_endian_2D_WKB_lowercase_hex": ("nonempty_even_length_lowercase_hex"),
        "roundtrip_geometry_big_endian_2D_WKB_lowercase_hex": (
            "nonempty_even_length_lowercase_hex"
        ),
        "roundtrip_wkb_byte_equal": "strict_boolean",
        "roundtrip_geometry_equals_normalized_buffer": "strict_boolean",
        "roundtrip_geometry_covers_fixed_point": "strict_boolean",
    }
    assert set(warmup["canonical_output_payload_derivation_exact"]) == set(
        warmup["canonical_output_payload_fields_exact"]
    )
    assert _canonical_payload_sha256(expected_warmup_input) == (
        "26bef17f5fc6fe2a3c6dc1690eeaa7de8faa9f43e735190ab2b5809caa6268a0"
    )
    assert warmup["canonical_input_and_output_payload_sha256_formula"] == (
        "sha256_seismoflux_canonical_json_v1_of_complete_payload"
    )
    assert warmup[
        "both_sibling_payload_sha256_values_must_be_recomputed_from_their_complete_exact_payload_before_receipt_sha256"
    ]
    assert warmup["callable_bindings_exact"] == [
        "shapely.geometry.box",
        "shapely.geometry.Point",
        "shapely.buffer",
        "shapely.normalize",
        "shapely.to_wkb",
        "shapely.from_wkb",
        "shapely.equals",
        "shapely.covers",
    ]
    box_coordinates = [
        float.fromhex(value)
        for value in cast(list[str], expected_warmup_input["box_min_max_xy_python_float_hex"])
    ]
    point_coordinates = [
        float.fromhex(value)
        for value in cast(list[str], expected_warmup_input["point_xy_python_float_hex"])
    ]
    fixed_box = shapely.box(*box_coordinates)
    fixed_point = Point(*point_coordinates)
    fixed_buffer = shapely.buffer(
        fixed_box,
        float.fromhex(cast(str, expected_warmup_input["buffer_distance_python_float_hex"])),
        quad_segs=cast(int, expected_warmup_input["buffer_quad_segs"]),
    )
    normalized_buffer = shapely.normalize(fixed_buffer)
    normalized_wkb = shapely.to_wkb(
        normalized_buffer,
        hex=False,
        output_dimension=expected_warmup_input["wkb_output_dimension"],
        byte_order=expected_warmup_input["wkb_byte_order"],
        include_srid=expected_warmup_input["wkb_include_srid"],
        flavor=expected_warmup_input["wkb_flavor"],
    )
    assert isinstance(normalized_wkb, bytes)
    roundtrip_geometry = shapely.from_wkb(normalized_wkb)
    roundtrip_wkb = shapely.to_wkb(
        roundtrip_geometry,
        hex=False,
        output_dimension=expected_warmup_input["wkb_output_dimension"],
        byte_order=expected_warmup_input["wkb_byte_order"],
        include_srid=expected_warmup_input["wkb_include_srid"],
        flavor=expected_warmup_input["wkb_flavor"],
    )
    assert isinstance(roundtrip_wkb, bytes)
    observed_warmup_output = {
        "normalized_buffer_big_endian_2D_WKB_lowercase_hex": normalized_wkb.hex(),
        "roundtrip_geometry_big_endian_2D_WKB_lowercase_hex": roundtrip_wkb.hex(),
        "roundtrip_wkb_byte_equal": roundtrip_wkb == normalized_wkb,
        "roundtrip_geometry_equals_normalized_buffer": bool(
            shapely.equals(roundtrip_geometry, normalized_buffer)
        ),
        "roundtrip_geometry_covers_fixed_point": bool(
            shapely.covers(roundtrip_geometry, fixed_point)
        ),
    }
    assert list(observed_warmup_output) == warmup["canonical_output_payload_fields_exact"]
    assert {
        key: observed_warmup_output[key] for key in warmup["canonical_output_boolean_values_exact"]
    } == warmup["canonical_output_boolean_values_exact"]
    assert len(_canonical_payload_sha256(observed_warmup_output)) == 64
    warmup_schema = protocol["outputs"]["canonical_nested_schemas"][
        "synthetic_adapter_geometry_warmup_receipt"
    ]
    assert warmup_schema["canonical_input_payload_fields_exact_ref"].endswith(
        ".canonical_input_payload_fields_exact"
    )
    assert warmup_schema["canonical_output_payload_fields_exact_ref"].endswith(
        ".canonical_output_payload_fields_exact"
    )
    assert warmup_schema["canonical_input_and_output_payload_sha256_formula_ref"].endswith(
        ".canonical_input_and_output_payload_sha256_formula"
    )
    assert runtime[
        "every_shared_identity_field_must_equal_same_field_in_positive_qualification_optimizer_runtime_code_seal_byte_for_byte"
    ]
    closing = seal["closing_seal"]
    assert closing["self_or_mutual_hash_reference_forbidden"] is True
    assert {
        "adapter_artifact_attempt_id",
        "adapter_artifact_opening_seal_sha256",
        "etas_artifact_sha256",
        "frozen_etas_comparator_receipt_sha256",
        "staged_adapter_payload_identity_excluding_closing_seal",
    } <= set(closing["includes"])
    outputs = protocol["outputs"]
    public_paths = outputs["adapter_public_artifacts"]
    assert public_paths["static_contract_visual"] == "docs/background_etas_comparator_contract.svg"
    assert public_paths["interactive_contract_visual"] == (
        "outputs/interactive/background_etas_comparator/index.html"
    )
    public_schema = outputs["adapter_public_artifact_schemas"]
    assert {
        "adapter_code_blob_sha256",
        "adapter_artifact_runtime_seal_sha256",
    } <= set(public_schema["artifact_manifest_fields_exact"])
    assert {
        "adapter_code_blob_sha256",
        "adapter_artifact_runtime_seal",
    } <= set(public_schema["opening_seal_fields_exact"])
    assert set(public_schema["artifact_manifest_nested_field_schema_refs"]) == {
        "model_spec_by_snapshot",
        "immigrant_density_artifact_sha256_by_snapshot",
        "propagation_domain_artifact_sha256_by_snapshot",
        "five_etas_parameter_snapshot_sha256",
    }
    model_specs = outputs["canonical_nested_schemas"]["adapter_model_spec_by_snapshot"]
    assert model_specs["keys_exact"] == [
        "fold_1",
        "fold_2",
        "fold_3",
        "fold_4",
        "final_validation",
    ]
    assert model_specs[
        "every_value_must_equal_same_snapshot_positive_qualification_parameter_snapshot_and_scientific_fit_input_model_spec_field_for_field_and_byte_for_byte"
    ]
    assert set(public_schema["opening_seal_nested_field_schema_refs"]) == {
        "repository_identity",
        "adapter_code_commit_and_remote_tag",
        "adapter_artifact_runtime_seal",
        "positive_qualification_result_tag_commit",
        "five_etas_parameter_snapshot_sha256",
        "protocol_package_tree_identity",
    }
    assert set(public_schema["global_receipt_nested_field_schema_refs"]) == {
        "ordered_complete_stage4_role_mapping",
        "etas_parameter_snapshot_sha256_by_role",
        "protocol_code_environment_hashes",
    }
    staged_adapter_schema = outputs["canonical_nested_schemas"]["staged_adapter_payload_identity"]
    assert staged_adapter_schema["logical_artifact_keys_exact"] == [
        "opening_seal",
        "artifact_manifest",
        "global_comparator_receipt",
        "report",
        "static_contract_visual",
        "interactive_contract_visual",
    ]
    assert public_schema["publication_manifest_self_path"] == (
        "data/manifests/background_etas_comparator_publication.json"
    )
    assert public_schema[
        "exact_public_path_file_sha256_map_keys_must_equal_comparator_receipt_tag_exact_added_paths_excluding_publication_manifest_self_path"
    ]
    assert public_schema[
        "publication_manifest_self_path_must_not_appear_in_its_internal_file_sha256_map"
    ]
    assert public_schema[
        "publication_manifest_final_file_sha256_is_frozen_only_by_result_commit_and_remote_annotated_tag_not_embedded_in_self"
    ]
    assert public_schema["contract_visual_any_unlisted_field_or_external_network_request_forbidden"]


def test_adapter_public_result_package_has_strict_cross_file_identity_closure() -> None:
    protocol = _load_yaml(PROTOCOL_PATH)
    outputs = protocol["outputs"]
    public_schema = outputs["adapter_public_artifact_schemas"]
    invariants = public_schema["adapter_public_result_cross_file_invariants"]

    assert invariants["adapter_artifact_attempt_id_equal_across"] == [
        "opening_seal.adapter_artifact_attempt_id",
        "artifact_manifest.adapter_artifact_attempt_id",
        "closing_seal.adapter_artifact_attempt_id",
        "publication_manifest.adapter_artifact_attempt_id",
        "selected_attempt_ledger.intent.adapter_artifact_attempt_id",
        "selected_attempt_ledger.ready_to_close.adapter_artifact_attempt_id",
        "selected_attempt_ledger.completed.adapter_artifact_attempt_id",
    ]
    assert invariants[
        "global_receipt_same_attempt_must_be_uniquely_bound_by_its_exact_opening_seal_and_artifact_manifest_hashes"
    ]
    assert set(invariants) >= {
        "adapter_artifact_opening_seal_sha256_equal_across",
        "adapter_code_blob_sha256_equal_across",
        "adapter_artifact_runtime_seal_sha256_equal_across",
        "etas_artifact_sha256_equal_across",
        "frozen_etas_comparator_receipt_sha256_equal_across",
        "adapter_artifact_closing_seal_sha256_equal_across",
        "publication_manifest_content_sha256_must_equal_recomputed_canonical_content_identity",
        "selected_attempt_ledger_entry_and_prefix_equalities",
        "parameter_and_qualification_equalities",
        "code_and_environment_equalities",
        "staged_logical_artifact_path_mapping_exact",
        "final_public_logical_artifact_path_mapping_exact",
        "complete_nonpublication_public_materialization_path_mapping_exact",
    }
    assert len(invariants["adapter_artifact_opening_seal_sha256_equal_across"]) == 8
    assert invariants["adapter_code_blob_sha256_equal_across"] == [
        "recomputed_adapter_code_payload_identity.adapter_code_blob_sha256",
        "opening_seal.adapter_code_blob_sha256",
        "opening_seal.adapter_artifact_runtime_seal.adapter_code_blob_sha256",
        "artifact_manifest.adapter_code_blob_sha256",
        "global_receipt.protocol_code_environment_hashes.adapter_code_blob_sha256",
    ]
    assert invariants["adapter_artifact_runtime_seal_sha256_equal_across"] == [
        "recomputed_opening_seal.adapter_artifact_runtime_seal."
        "adapter_artifact_runtime_seal_sha256",
        "opening_seal.adapter_artifact_runtime_seal.adapter_artifact_runtime_seal_sha256",
        "artifact_manifest.adapter_artifact_runtime_seal_sha256",
        "local_payload_manifest.adapter_artifact_runtime_seal_sha256",
        "every_immigrant_density_payload.adapter_artifact_runtime_seal_sha256",
        "every_propagation_domain_payload.adapter_artifact_runtime_seal_sha256",
        "global_receipt.protocol_code_environment_hashes.adapter_artifact_runtime_seal_sha256",
    ]
    assert len(invariants["etas_artifact_sha256_equal_across"]) == 7
    assert len(invariants["frozen_etas_comparator_receipt_sha256_equal_across"]) == 6
    assert len(invariants["adapter_artifact_closing_seal_sha256_equal_across"]) == 4

    parameters = invariants["parameter_and_qualification_equalities"]
    assert set(parameters) == {
        "etas_parameter_set_sha256_equal_across",
        "five_etas_parameter_snapshot_sha256_equal_across",
        "global_role_parameter_hashes_must_equal_frozen_role_projection_of_five_snapshot_map",
        "global_role_projection_exact",
        "etas_numerical_qualification_evidence_sha256_equal_across",
        "artifact_model_spec_by_snapshot_must_equal_same_snapshot_model_spec_in_each_positive_qualification_parameter_snapshot_and_scientific_fit_input",
        "artifact_adapter_code_blob_sha256_must_equal_recomputed_canonical_adapter_code_payload_at_verified_remote_adapter_code_tag_commit",
    }
    assert parameters["global_role_projection_exact"] == {
        "development": "fold_4",
        "formal_validation": "final_validation",
        "prospective": "final_validation",
    }
    code_environment = invariants["code_and_environment_equalities"]
    assert set(code_environment) == {
        "protocol_tag_commit",
        "repair_code_tag_commit",
        "positive_qualification_result_tag_commit",
        "adapter_code_tag_commit",
        "environment_lock_sha256",
        "adapter_code_diff_receipt_sha256",
        "adapter_code_blob_sha256",
        "adapter_artifact_runtime_seal_sha256",
    }
    repository_schema = outputs["canonical_nested_schemas"]["adapter_repository_identity"]
    assert {
        "remote_protocol_tag_commit",
        "remote_repair_code_tag_commit",
        "remote_positive_qualification_result_tag_commit",
        "remote_adapter_code_tag_commit",
    } <= set(repository_schema["fields_exact"])
    assert repository_schema["every_remote_tag_commit_must_be_network_verified_before_opening_seal"]
    assert invariants["acyclic_construction_and_reference_order_exact"] == [
        "attempt_ledger_intent_append_and_fsync",
        "adapter_runtime_preflight_and_staged_opening_seal",
        "ten_local_restricted_artifact_payloads",
        "local_restricted_payload_manifest",
        "artifact_manifest",
        "global_receipt",
        "report",
        "static_contract_visual",
        "interactive_contract_visual",
        "attempt_ledger_ready_to_close",
        "staged_closing_seal",
        "public_materialization_and_reopen_verification_of_seven_nonpublication_files",
        "attempt_ledger_completed",
        "publication_manifest",
        "result_commit_and_remote_annotated_tag",
    ]

    assert invariants["final_public_logical_artifact_path_mapping_exact"] == {
        key: outputs["adapter_public_artifacts"][key]
        for key in outputs["canonical_nested_schemas"]["staged_adapter_payload_identity"][
            "logical_artifact_keys_exact"
        ]
    }
    staged_root = invariants["staged_public_root_path_template"]
    assert staged_root.endswith("/{adapter_artifact_attempt_id}/staged_public")
    assert set(invariants["staged_logical_artifact_path_mapping_exact"]) == set(
        invariants["final_public_logical_artifact_path_mapping_exact"]
    )
    assert all(
        path.startswith(f"{staged_root}/")
        for path in invariants["staged_logical_artifact_path_mapping_exact"].values()
    )
    assert invariants["closing_seal_staged_path_template"].startswith(f"{staged_root}/")
    assert "closing_seal" not in invariants["staged_logical_artifact_path_mapping_exact"]
    assert len(invariants["staged_logical_artifact_path_mapping_exact"]) == 6
    expected_nonpublication_mapping = {
        "opening_seal": "data/manifests/background_etas_adapter_opening_seal.json",
        "artifact_manifest": (
            "models/registry/background_etas_comparator/etas_artifact_manifest.json"
        ),
        "global_comparator_receipt": "data/manifests/background_etas_comparator_receipt.json",
        "closing_seal": "data/manifests/background_etas_adapter_closing_seal.json",
        "report": "docs/background_etas_comparator_report.md",
        "static_contract_visual": "docs/background_etas_comparator_contract.svg",
        "interactive_contract_visual": "outputs/interactive/background_etas_comparator/index.html",
    }
    assert len(expected_nonpublication_mapping) == 7
    assert (
        invariants["complete_nonpublication_public_materialization_path_mapping_exact"]
        == expected_nonpublication_mapping
    )
    publication_self_path = outputs["adapter_public_artifact_schemas"][
        "publication_manifest_self_path"
    ]
    expected_nonpublication_paths = [
        path
        for path in outputs["adapter_public_artifacts"]["comparator_receipt_tag_exact_added_paths"]
        if path != publication_self_path
    ]
    assert len(expected_nonpublication_paths) == 7
    assert set(expected_nonpublication_mapping.values()) == set(expected_nonpublication_paths)
    assert (
        invariants["closing_seal_final_public_path_exact"]
        == outputs["adapter_public_artifacts"]["closing_seal"]
    )
    for required in (
        "every_staged_logical_artifact_hash_must_equal_sha256_of_exact_staged_file_bytes_at_mapped_path",
        "every_staged_logical_artifact_file_must_be_byte_identical_after_public_materialization",
        "staged_logical_hash_must_equal_publication_path_map_value_for_same_materialized_path",
        "staged_payload_identity_recomputed_aggregate_must_equal_closing_seal_staged_adapter_payload_identity",
        "publication_path_map_value_for_every_key_must_equal_sha256_of_exact_final_repository_relative_path_file_bytes",
        "publication_path_map_must_be_recomputed_after_materialization_and_completed_then_before_publication_manifest_atomic_write",
        "deterministic_publication_manifest_atomic_write_may_be_retried_after_completed_only_if_final_ledger_and_all_seven_file_bytes_are_unchanged",
        "publication_path_map_self_path_excluded_and_final_self_file_sha_frozen_only_by_result_commit_and_remote_tag",
        "all_seven_nonpublication_final_paths_must_be_absent_before_materialization_and_no_historical_result_may_be_overwritten",
        "seven_file_materialization_must_stage_destination_sibling_temps_flush_fsync_reopen_hash_then_atomic_replace_and_reopen_verify_each_exact_byte_payload",
        "any_materialization_failure_before_completed_must_remove_only_same_attempt_newly_created_final_files_after_exact_hash_and_attempt_identity_match_retain_all_attempt_staging_and_append_after_closing_before_completed_aborted",
        "completed_may_be_appended_only_after_all_seven_nonpublication_final_files_are_present_byte_identical_and_match_their_staged_or_closing_hashes",
        "opening_may_not_reference_artifact_global_closing_publication_or_final_ledger_hash",
        "artifact_may_not_reference_global_closing_publication_or_final_ledger_hash",
        "global_receipt_may_not_reference_closing_publication_or_final_ledger_hash",
        "closing_may_not_reference_publication_or_final_ledger_hash",
        "staged_logical_map_may_not_contain_closing_publication_attempt_ledger_or_its_own_aggregate_as_a_logical_artifact",
        "all_six_preclosing_staged_files_must_use_temp_write_flush_fsync_atomic_replace_reopen_hash_and_byte_verification_before_ready_to_close",
        "closing_seal_staged_file_must_use_temp_write_flush_fsync_atomic_replace_reopen_hash_and_byte_verification_after_ready_to_close",
    ):
        assert invariants[required] is True
    assert (
        invariants[
            "any_missing_extra_alias_duplicate_hash_byte_identity_attempt_parameter_model_code_environment_or_ledger_mismatch_action"
        ]
        == "invalidate_adapter_attempt_and_publish_nothing"
    )


def test_issue_forecast_rng_contexts_are_distinct_by_role_and_replicate() -> None:
    seed = _load_yaml(PROTOCOL_PATH)["adapter_contract"]["issue_forecast_definition"][
        "seed_context_contract"
    ]
    role_snapshots = {
        "development": "fold_4",
        "formal_validation": "final_validation",
        "prospective": "final_validation",
    }
    parameter_sha_by_snapshot = {
        snapshot: hashlib.sha256(snapshot.encode()).hexdigest()
        for snapshot in set(role_snapshots.values())
    }
    context_sha = "1" * 64
    digests = {
        SeedContext(
            root_seed=seed["root_seed"],
            protocol_version=seed["protocol_version"],
            namespace=seed["namespace"],
            model_id=(f"etas/{role}/{snapshot}/{parameter_sha_by_snapshot[snapshot]}"),
            issue_id=f"stage4/{role}/2024-01-01/{context_sha}",
            replicate_index=replicate,
        ).digest()
        for role, snapshot in role_snapshots.items()
        for replicate in range(128)
    }
    assert len(digests) == 3 * 128


def test_new_outputs_do_not_overlap_historical_stage2_paths() -> None:
    protocol = _load_yaml(PROTOCOL_PATH)
    new_paths = {protocol["fit_input_bundle"]["local_root"]}
    new_paths.update(
        value for value in protocol["outputs"].values() if isinstance(value, str) and "/" in value
    )
    historical_prefixes = {
        "data/processed/stage2R/local_support",
        "models/registry/background_local_support",
        "outputs/backtests/background_local_support",
        "outputs/experiments/background_local_support",
        "data/manifests/background_local_support_model_registry.json",
        "docs/background_local_support_report.md",
    }
    for new_path in new_paths:
        assert all(
            new_path != historical and not new_path.startswith(f"{historical}/")
            for historical in historical_prefixes
        )
    public_probe = "models/registry/background_etas_numerical_repair/probe.json"
    comparator_probe = "models/registry/background_etas_comparator/probe.json"
    local_probe = "data/processed/stage2R/etas_numerical_repair_adapter_payload/probe.json"
    adapter_ledger_probe = "data/manifests/etas_numerical_repair_adapter_attempt_ledger.json"
    source_ledger_probe = "data/manifests/etas_numerical_repair_source_access_ledger.json"
    public_check = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", public_probe],
        check=False,
    )
    local_check = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", local_probe],
        check=False,
    )
    comparator_check = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", comparator_probe],
        check=False,
    )
    adapter_ledger_check = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", adapter_ledger_probe],
        check=False,
    )
    source_ledger_check = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", source_ledger_probe],
        check=False,
    )
    assert public_check.returncode == 1
    assert comparator_check.returncode == 1
    assert local_check.returncode == 0
    assert adapter_ledger_check.returncode == 0
    assert source_ledger_check.returncode == 0
    outputs = protocol["outputs"]
    assert outputs["public_parameter_artifact_root_must_not_be_gitignored"]
    branch = outputs["parameter_artifact_branch_contract"]
    assert branch["evaluable"] == {
        "root_presence_in_qualification_result_tag": "required",
        "exact_files": ["parameter_snapshots.json", "parameter_set_manifest.json"],
    }
    assert branch["not_evaluable"]["root_presence_in_qualification_result_tag"] == "forbidden"
    assert branch["not_evaluable"]["exact_files"] == []
    assert branch["invalid_execution"]["public_qualification_result_allowed"] is False
    registry_schema = outputs["public_parameter_registry_schema"]
    assert registry_schema["any_unlisted_file_field_nested_field_or_alias_forbidden"]
    assert registry_schema["unknown_missing_duplicate_or_noncanonical_field_rejected"]
    assert (
        registry_schema[
            "event_identifier_coordinate_time_target_score_or_training_row_fields_allowed"
        ]
        is False
    )
    manifest_schema = outputs["public_qualification_manifest_schema"]
    cross_file = manifest_schema["cross_file_input_identity_invariants"]
    assert cross_file["every_mapping_key_order_exactly_snapshot_key_order_and_every_value_equal"]
    assert cross_file["any_missing_extra_or_mismatched_cross_file_identity_action"] == (
        "invalid_execution_without_public_result"
    )
    assert set(cross_file) >= {
        "public_source_access_receipt_canonical_sha256_equal_across",
        "fit_input_manifest_file_and_content_sha256_equal_across",
        "five_snapshot_scientific_fit_input_sha256_mapping_equal_across",
        "five_snapshot_parent_replay_scientific_fit_input_sha256_mapping_equal_across",
        "five_snapshot_parent_replay_membership_identity_sha256_mapping_equal_across",
        "start_manifest_file_and_vector_payload_sha256_equal_across",
        "source_and_reader_identity_equal_across",
        "global_source_ledger_identity_equal_across",
        "selected_source_acquisition_attempt_id_equal_across",
        "single_qualification_attempt_id_equal_across",
        "frozen_execution_identity_equal_across",
        "qualification_branch_values_equal_across",
    }
    start_pair = outputs["canonical_nested_schemas"][
        "start_manifest_file_and_vector_payload_sha256_pair"
    ]
    assert start_pair["fields_exact"] == ["file_sha256", "vector_payload_sha256"]
    seal_schema = outputs["public_qualification_seal_schema"]
    assert set(seal_schema["opening_nested_field_schema_refs"]) == {
        "repository_identity",
        "protocol_package_blob_oid_by_path",
        "repair_code_commit_and_remote_tag",
        "optimizer_runtime_baseline_blob_and_content_sha256",
        "all_public_input_binding_sha256",
        "frozen_local_restricted_input_identity_metadata",
    }
    assert (
        "python_runtime_file_sha256_and_size_map" in seal_schema["runtime_nested_field_schema_refs"]
    )
    assert (
        seal_schema["runtime_nested_field_schema_refs"]["synthetic_runtime_warmup_receipt"]
        == "outputs.canonical_nested_schemas.synthetic_runtime_warmup_receipt"
    )
    warmup_schema = outputs["canonical_nested_schemas"]["synthetic_runtime_warmup_receipt"]
    assert warmup_schema["full_receipt_must_equal_code_tag_baseline_byte_for_byte"]
    launcher_schema = outputs["canonical_nested_schemas"]["isolated_launcher_identity"]
    assert launcher_schema["sys_path_role_record_fields_exact"] == [
        "order_index",
        "role",
        "root_role",
        "canonical_root_relative_path",
        "entry_kind",
        "regular_file_sha256_or_null",
        "regular_file_size_or_null",
    ]
    assert launcher_schema["sys_path_root_role_values_exact"] == [
        "base_prefix",
        "venv_prefix",
        "workspace",
    ]
    assert launcher_schema["pth_record_fields_exact"] == [
        "root_role",
        "canonical_root_relative_path",
        "file_sha256",
    ]
    assert launcher_schema["pth_root_role_values_exact"] == ["venv_prefix"]
    startup_environment = launcher_schema["startup_environment_required_exact_values"]
    assert startup_environment["SETUPTOOLS_USE_DISTUTILS"] == "stdlib"
    assert {
        key: startup_environment[key]
        for key in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "BLIS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        )
    } == {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }
    assert {
        "COVERAGE_PROCESS_START",
        "COVERAGE_PROCESS_CONFIG",
        "COV_CORE_SOURCE",
        "COV_CORE_CONFIG",
        "COV_CORE_DATAFILE",
    } <= set(launcher_schema["startup_environment_required_absent_names_exact"])
    assert launcher_schema["meta_path_exact_ordered_finders"] == [
        "_frozen_importlib.BuiltinImporter",
        "_frozen_importlib.FrozenImporter",
        "_frozen_importlib_external.PathFinder",
    ]
    assert launcher_schema[
        "distutils_hack_coverage_pytest_cov_or_any_other_meta_path_finder_forbidden"
    ]
    assert (
        outputs["canonical_nested_schemas"]["windows_runtime_identity"]["platform_system_exact"]
        == "Windows"
    )
    assert registry_schema["evaluable_cross_file_identity_invariants"][
        "every_snapshot_record_hash_must_recompute_and_match_its_mapping_value"
    ]
    assert registry_schema["not_evaluable_cross_file_absence_invariants"] == {
        "qualification_manifest_parameter_set_and_snapshot_mapping_must_be_null": True,
        "parameter_artifact_root_and_both_registry_files_must_be_absent": True,
    }
    assert set(protocol["outputs"]["public_visual_forbidden_fields"]) == {
        "event_id",
        "physical_event_id",
        "event_coordinates",
        "coordinates",
        "longitude",
        "latitude",
        "projected_x",
        "projected_y",
        "target_rows",
        "assessment_event_id",
        "assessment_event_count",
        "assessment_scores",
        "model_score",
        "information_gain",
        "score_id",
    }
    assert protocol["outputs"]["any_public_visual_field_not_in_allowlist_forbidden"] is True
    report_contract = outputs["public_report_contract"]
    assert report_contract[
        "public_visual_forbidden_fields_apply_to_report_text_tables_links_alt_text_and_embedded_payloads"
    ]
    assert (
        report_contract[
            "absolute_path_hostname_username_environment_secret_or_local_ledger_detail_allowed"
        ]
        is False
    )
    assert set(protocol["outputs"]["public_visual_allowed_fields"]) == {
        "snapshot_id",
        "start_index",
        "numerical_status",
        "objective",
        "gradient_infinity_norm",
        "iterations",
        "function_evaluations",
        "parameter_name",
        "parameter_value",
        "gate_name",
        "gate_status",
        "failure_code",
    }
    assert protocol["outputs"][
        "allowlist_applies_to_embedded_json_tooltips_dom_attributes_downloads_and_accessibility_text"
    ]
    assert protocol["outputs"]["interactive_external_network_requests_allowed"] is False
    assert protocol["outputs"]["interactive_may_embed_only_allowlisted_static_payload"] is True
    domains = protocol["outputs"]["public_visual_value_domains"]
    assert set(domains["parameter_name"]) == {
        "background_rate_per_day",
        "productivity_k",
        "alpha",
        "c_days",
        "p",
        "branching_ratio",
    }


def test_parameter_snapshot_derivation_and_crosswalk_is_strict_and_complete() -> None:
    protocol = _load_yaml(PROTOCOL_PATH)
    registry = protocol["outputs"]["public_parameter_registry_schema"]
    crosswalk = registry["parameter_snapshot_derivation_and_crosswalk"]

    assert (
        "etas_numerical_qualification_evidence_sha256"
        not in registry["parameter_snapshots_json_top_level_fields"]
    )
    assert (
        "etas_numerical_qualification_evidence_sha256"
        not in registry["parameter_set_manifest_json_fields"]
    )
    registry_invariants = registry["evaluable_cross_file_identity_invariants"]
    assert "qualification_evidence_sha256_equal_across" not in registry_invariants
    assert registry_invariants[
        "parameter_registry_files_may_not_contain_qualification_closing_seal_qualification_evidence_or_manifest_content_sha256"
    ]

    assert crosswalk["applicability"] == (
        "evaluable_branch_only_after_complete_protocol_valid_five_snapshot_qualification"
    )
    assert crosswalk["source_records_exact"] == [
        "complete_scientific_fit_input_payload",
        "fit_etas_call_opening_receipt",
        "fit_etas_call_closing_receipt",
        "selected_diagnostic_row",
        "snapshot_gate_result",
    ]
    source_selection = crosswalk["source_selection"]
    assert source_selection[
        "closing_receipt_opening_receipt_sha256_must_resolve_to_the_same_selected_opening_receipt"
    ]
    assert source_selection["every_source_snapshot_id_must_equal_parameter_snapshot_snapshot_id"]
    assert source_selection["cross_snapshot_or_ambiguous_source_selection_forbidden"]

    field_sources = crosswalk["record_field_derivation_source_exact"]
    assert set(field_sources) == set(registry["parameter_snapshot_record_fields"])
    assert field_sources["scientific_fit_input_sha256"] == (
        "complete_scientific_fit_input_payload_canonical_sha256"
    )
    assert field_sources["model_spec"] == "complete_scientific_fit_input_payload.model_spec"
    assert field_sources["selected_start_index"] == (
        "snapshot_gate_result.selected_start_index_or_null"
    )
    assert field_sources["physical_parameters_hex"].endswith(
        "returned_fit_result_canonical_payload.best_parameters_hex_or_null"
    )
    assert field_sources["hessian_and_uncertainty"] == (
        "exact_projection_of_fit_etas_call_closing_receipt.returned_fit_result_canonical_payload"
    )

    scientific = crosswalk["scientific_input_and_model_spec_crosswalk"]
    assert scientific["scientific_fit_input_sha256_must_equal"] == [
        "canonical_complete_scientific_fit_input_payload_sha256",
        "fit_etas_call_opening_receipt.scientific_fit_input_sha256",
        "qualification_manifest.scientific_fit_input_sha256_by_snapshot_matching_value",
    ]
    assert scientific[
        "model_spec_must_equal_complete_scientific_fit_input_payload_model_spec_field_for_field_and_byte_for_byte"
    ]
    assert scientific["model_spec_fields_exact_ref"] == (
        "fit_input_bundle.scientific_fit_input_record_schemas.model_spec_fields_exact"
    )
    assert scientific[
        "fit_etas_opening_exact_fit_arguments_preimage_ETASModelSpec_payload_must_equal_parameter_snapshot_model_spec"
    ]

    selected = crosswalk["selected_start_and_diagnostic_crosswalk"]
    assert selected["snapshot_gate_result_requirements"] == {
        "numerical_status": "evaluable",
        "selected_start_index_or_null": "required",
        "ordered_failure_codes": "empty",
    }
    assert selected[
        "selected_start_index_must_equal_first_stability_eligible_row_under_frozen_objective_then_start_index_order"
    ]
    assert selected["selected_diagnostic_row_must_have"] == {
        "etas_start_scipy_converged": True,
        "terminal_transformed_hex_or_null": "required",
        "objective_hex_or_null": "required",
        "gradient_infinity_norm_hex_or_null": "required",
        "physical_parameters_hex_or_null": "required",
        "failure_code_or_null": None,
    }
    assert selected["selected_terminal_vector_must_equal_byte_for_byte"] == [
        "selected_diagnostic_row.terminal_transformed_hex_or_null",
        "matching_selected_returned_start_result.final_transformed_hex",
    ]
    assert selected["selected_terminal_transformed_sha256_formula"].endswith(
        "exact_ordered_five_selected_terminal_float64_hex_strings"
    )
    assert selected[
        "hessian_evaluation_point_sha256_must_equal_selected_terminal_transformed_sha256"
    ]

    parameter_values = crosswalk["parameter_value_crosswalk"]
    assert parameter_values["transformed_field_to_selected_terminal_index_exact"] == {
        "log_background_rate_per_day": 0,
        "log_productivity_k": 1,
        "log_alpha": 2,
        "log_c_days": 3,
        "log_p_minus_one": 4,
    }
    assert parameter_values["physical_parameters_hex_must_equal_byte_for_byte"] == [
        "fit_etas_call_closing_receipt.returned_fit_result_canonical_payload."
        "best_parameters_hex_or_null",
        "selected_diagnostic_row.physical_parameters_hex_or_null",
        "frozen_repaired_ETASParameterBounds.from_transformed_of_selected_terminal_vector",
    ]
    assert parameter_values[
        "endpoint_aware_decode_may_not_be_reimplemented_clipped_toleranced_or_recomputed_by_plain_exp"
    ]

    uncertainty = crosswalk["hessian_and_uncertainty_crosswalk"]
    assert uncertainty["returned_fit_requirements"] == {
        "stability_stable": True,
        "hessian_success": True,
        "uncertainty_or_null": "required",
    }
    assert uncertainty["field_mapping_exact"] == {
        "observed_hessian_transformed_hex": (
            "returned_fit_result.stability.hessian.matrix_hex_or_null"
        ),
        "minimum_eigenvalue_hex": (
            "returned_fit_result.stability.hessian.minimum_eigenvalue_hex_or_null"
        ),
        "condition_number_hex": (
            "returned_fit_result.stability.hessian.condition_number_hex_or_null"
        ),
        "transformed_covariance_hex": (
            "returned_fit_result.uncertainty_or_null.transformed_covariance_hex"
        ),
        "physical_covariance_hex": (
            "returned_fit_result.uncertainty_or_null.physical_covariance_hex"
        ),
        "confidence_level_hex": ("returned_fit_result.uncertainty_or_null.confidence_level_hex"),
        "parameter_delta_estimates": (
            "returned_fit_result.uncertainty_or_null.parameter_estimates"
        ),
        "branching_ratio_delta_estimate": (
            "returned_fit_result.uncertainty_or_null.branching_ratio"
        ),
    }
    assert uncertainty[
        "branching_ratio_delta_estimate_estimate_hex_must_equal_snapshot_gate_result_branching_ratio_hex_or_null"
    ]
    assert uncertainty["hessian_and_uncertainty_sha256_must_recompute_from_complete_exact_object"]

    masses = crosswalk["aki_b_beta_and_bin_mass_crosswalk"]
    assert masses["beta_hex_must_equal_byte_for_byte"] == [
        "complete_scientific_fit_input_payload.beta_hex",
        "complete_scientific_fit_input_payload.model_spec.beta_hex",
        "parameter_snapshot.model_spec.beta_hex",
    ]
    assert masses["magnitude_bin_definition_and_formula_ref"] == "adapter_contract.magnitude_bins"
    assert masses[
        "beta_hex_must_equal_python_float64_hex_of_float_aki_b_times_math_log_10_under_frozen_build_etas_model_spec"
    ]
    assert masses[
        "M5_6_mass_hex_must_be_recomputed_from_frozen_beta_mc_mmax_lower_5_upper_6_formula"
    ]
    assert masses[
        "M6_plus_mass_hex_must_be_recomputed_from_frozen_beta_mc_mmax_lower_6_upper_mmax_formula"
    ]
    assert masses["payload_sha256_must_recompute_from_complete_exact_object_without_payload_sha256"]

    closure = crosswalk["gate_and_record_closure"]
    assert closure[
        "snapshot_gate_result_sha256_must_recompute_from_complete_same_snapshot_gate_result"
    ]
    assert closure["snapshot_gate_result_sha256_must_equal_fit_attempt_snapshot_bound_gate_sha256"]
    assert closure[
        "all_parameter_snapshot_record_fields_except_own_sha256_must_be_derived_exactly_once_by_record_field_derivation_source_exact"
    ]
    assert (
        closure["any_missing_extra_mismatch_noncanonical_value_or_failed_recomputation_action"]
        == "invalid_execution_without_parameter_artifact_or_public_qualification_result"
    )


def test_qualification_gate_and_public_result_crosswalks_are_closed() -> None:
    protocol = _load_yaml(PROTOCOL_PATH)
    invocation = protocol["optimizer_invocation_receipt_protocol"]
    qualification = protocol["qualification"]
    outputs = protocol["outputs"]
    manifest = outputs["public_qualification_manifest_schema"]

    assert (
        "three_grid_gate_evidence_sha256_or_null"
        in invocation["fit_attempt_snapshot_payload_fields_exact"]
    )
    grid = qualification["three_grid_gate_evidence_protocol"]
    assert grid["exact_existing_evaluator"] == (
        "seismoflux.background.pipeline_etas._grid_gate_evidence"
    )
    assert grid["grid_size_order_km_exact"] == [50.0, 25.0, 12.5]
    assert grid[
        "evidence_present_iff_selected_start_index_is_nonnull_and_all_upstream_stability_gates_required_before_grid_evaluation_pass"
    ]
    failure_ordering = qualification["snapshot_failure_code_order_and_deduplication"]
    assert failure_ordering["append_each_failed_gate_once_in_exact_order"]
    skip = failure_ordering["prerequisite_skip_and_failure_code_semantics"]
    truth = failure_ordering["gate_dependency_and_status_truth_table"]
    assert truth["ordered_gate_names_exact"] == [
        "minimum_converged_starts",
        "converged_gradient",
        "best_three_objective_spread",
        "best_three_parameter_spread",
        "hessian_minimum_eigenvalue",
        "hessian_condition_number",
        "branching_ratio",
        "grid_25_to_12_5_expected_count_difference",
        "grid_25_to_12_5_density_l1",
    ]
    assert truth["gate_status_values_exact"] == [
        "passed",
        "failed",
        "not_run_upstream_gate",
    ]
    expected_gate_prerequisites_and_failure_codes = {
        "minimum_converged_starts": ([], "insufficient_converged_starts"),
        "converged_gradient": (
            ["minimum_converged_starts_passed"],
            "converged_gradient_threshold_exceeded",
        ),
        "best_three_objective_spread": (
            ["minimum_converged_starts_passed", "converged_gradient_passed"],
            "best_three_objective_spread_exceeded",
        ),
        "best_three_parameter_spread": (
            ["minimum_converged_starts_passed", "converged_gradient_passed"],
            "best_three_parameter_spread_exceeded",
        ),
        "hessian_minimum_eigenvalue": (
            ["both_best_three_spread_siblings_passed"],
            "hessian_invalid_or_minimum_eigenvalue_failed",
        ),
        "hessian_condition_number": (
            ["hessian_minimum_eigenvalue_passed"],
            "hessian_condition_number_exceeded",
        ),
        "branching_ratio": (
            ["hessian_condition_number_passed", "selected_start_nonnull"],
            "branching_ratio_gate_failed",
        ),
        "grid_25_to_12_5_expected_count_difference": (
            ["branching_ratio_passed"],
            "grid_25_to_12_5_expected_count_difference_exceeded",
        ),
        "grid_25_to_12_5_density_l1": (
            ["branching_ratio_passed"],
            "grid_25_to_12_5_density_l1_exceeded",
        ),
    }
    assert list(expected_gate_prerequisites_and_failure_codes) == truth["ordered_gate_names_exact"]
    for gate_name, (
        prerequisites,
        failure_code,
    ) in expected_gate_prerequisites_and_failure_codes.items():
        assert truth[gate_name]["prerequisites"] == prerequisites
        assert truth[gate_name]["failure_code"] == failure_code
    assert truth["best_three_objective_spread"]["evaluation_group"] == (
        "best_three_spread_siblings"
    )
    assert truth["best_three_parameter_spread"]["evaluation_group"] == (
        "best_three_spread_siblings"
    )
    assert truth["grid_25_to_12_5_expected_count_difference"]["evaluation_group"] == (
        "three_grid_metric_siblings"
    )
    assert truth["grid_25_to_12_5_density_l1"]["evaluation_group"] == ("three_grid_metric_siblings")
    assert truth["sibling_group_rule"] == (
        "once_all_shared_prerequisites_pass_both_sibling_metrics_are_evaluated_and_each_"
        "status_and_failure_code_is_derived_independently"
    )
    assert truth["downstream_rule"] == (
        "any_failed_or_not_run_prerequisite_makes_each_dependent_gate_not_run_upstream_"
        "gate_with_null_gate_metric_and_no_failure_code"
    )
    assert truth["hessian_failure_rule"] == (
        "invalid_absent_nonfinite_or_below_minimum_fails_only_hessian_minimum_"
        "eigenvalue_and_skips_condition_and_all_downstream_gates"
    )
    assert truth["ordered_gate_status_records_must_contain_every_ordered_gate_once_in_exact_order"]
    assert truth[
        "ordered_failure_codes_must_equal_exact_order_projection_of_gate_status_failed_to_declared_failure_code"
    ]
    assert skip == {
        "fewer_than_minimum_stability_eligible_starts": (
            "count_failed_and_every_downstream_gate_not_run_upstream_gate_with_null_"
            "snapshot_gate_metric"
        ),
        "best_three_metrics_in_closing_stability_payload_when_count_or_gradient_gate_failed": (
            "retained_only_in_closing_diagnostic_payload_but_snapshot_gate_metrics_are_null_"
            "and_both_spread_gates_are_not_run_upstream_gate"
        ),
        "hessian_metrics_absent_because_any_count_gradient_or_spread_prerequisite_failed": (
            "both_hessian_snapshot_gate_metrics_null_and_hessian_minimum_and_condition_"
            "gates_not_run_upstream_gate"
        ),
        "hessian_attempted_but_nonfinite_invalid_or_below_minimum": (
            "append_hessian_invalid_or_minimum_eigenvalue_failed_once"
        ),
        "hessian_finite_positive_but_condition_number_above_maximum": (
            "append_hessian_condition_number_exceeded_once"
        ),
        (
            "branching_ratio_absent_because_any_hessian_prerequisite_failed_or_selected_"
            "start_is_null"
        ): ("skipped_null_and_do_not_append_branching_ratio_gate_failed"),
        (
            "three_grid_evidence_absent_because_any_required_upstream_stability_or_"
            "branching_gate_failed"
        ): (
            "both_grid_gates_skipped_all_grid_metrics_null_and_do_not_append_either_grid_"
            "failure_code"
        ),
        "numerical_status_when_any_downstream_gate_is_skipped": (
            "not_evaluable_from_the_actual_recorded_upstream_failure_codes_only"
        ),
        "skipped_gate_status_value_exact": "not_run_upstream_gate",
    }

    gate_fields = manifest["snapshot_gate_result_fields_exact"]
    assert {
        "hessian_minimum_eigenvalue_nonfinite_kind_or_null",
        "hessian_condition_number_nonfinite_kind_or_null",
        "three_grid_gate_evidence_sha256_or_null",
        "ordered_gate_status_records",
    } <= set(gate_fields)
    gate_crosswalk = manifest["snapshot_gate_result_derivation_and_crosswalk"]
    assert gate_crosswalk["selected_start_index_exact_source"] == (
        "qualification.selected_start_rule_applied_to_same_snapshot_five_diagnostic_rows"
    )
    assert gate_crosswalk["three_grid_gate_evidence_sha256_exact_source"] == (
        "same_snapshot_reopened_complete_six_field_envelope_three_grid_gate_evidence_"
        "envelope_sha256"
    )
    assert gate_crosswalk["prerequisite_skip_and_failure_code_semantics_ref"] == (
        "qualification.snapshot_failure_code_order_and_deduplication."
        "prerequisite_skip_and_failure_code_semantics"
    )
    assert gate_crosswalk["ordered_gate_status_records_exact_source_ref"] == (
        "qualification.snapshot_failure_code_order_and_deduplication."
        "gate_dependency_and_status_truth_table"
    )
    assert manifest["snapshot_ordered_gate_status_records_schema_ref"] == (
        "outputs.canonical_nested_schemas.qualification_ordered_gate_status_records"
    )
    gate_status_schema = outputs["canonical_nested_schemas"][
        "qualification_ordered_gate_status_records"
    ]
    assert gate_status_schema["exact_length"] == 9
    assert gate_status_schema["canonical_container_type_exact"] == "ordered_list_not_mapping"
    assert gate_status_schema["item_fields_exact"] == ["gate_name", "gate_status"]
    assert gate_status_schema["gate_name_order_exact_ref"] == (
        "qualification.snapshot_failure_code_order_and_deduplication."
        "gate_dependency_and_status_truth_table.ordered_gate_names_exact"
    )
    assert gate_status_schema["gate_status_values_exact_ref"] == (
        "qualification.snapshot_failure_code_order_and_deduplication."
        "gate_dependency_and_status_truth_table.gate_status_values_exact"
    )
    assert gate_status_schema["every_gate_name_must_appear_exactly_once_in_declared_order"]
    assert (
        gate_crosswalk["any_source_hash_value_state_order_status_or_cross_snapshot_mismatch_action"]
        == "invalid_execution_without_qualification_result"
    )

    excluded = manifest[
        "staged_preclosing_projection_fields_must_equal_top_level_fields_minus_exact_three_identity_fields"
    ]
    assert excluded == [
        "qualification_closing_seal_sha256",
        "etas_numerical_qualification_evidence_sha256",
        "qualification_manifest_content_sha256",
    ]
    assert manifest["staged_preclosing_projection_fields_exact"] == [
        field for field in manifest["top_level_fields_exact"] if field not in excluded
    ]
    staged = outputs["canonical_nested_schemas"]["staged_public_payload_identity"]
    assert (
        "qualification_manifest_preclosing_projection_sha256_excluding_closing_seal_"
        "qualification_evidence_and_manifest_content" in staged["fields_exact"]
    )
    assert staged[
        "every_preclosing_staged_file_byte_payload_must_exclude_direct_and_transitive_references_to"
    ] == [
        "qualification_closing_seal_sha256",
        "etas_numerical_qualification_evidence_sha256",
        "qualification_manifest_content_sha256",
    ]
    closing = protocol["qualification_execution_seal"]["closing_seal"]
    assert closing["qualification_publication_hash_DAG_order_exact"].endswith(
        "etas_numerical_qualification_evidence_sha256_then_qualification_manifest_content_sha256"
    )

    public_crosswalk = manifest["qualification_public_result_crosswalk"]
    negative_scalar = public_crosswalk["closing_seal_scalar_siblings_must_equal_manifest"][
        "etas_numerical_negative_evidence_sha256_or_null"
    ]
    assert negative_scalar == {
        "evaluable": {"closing_value": None, "manifest_nested_value": None},
        "not_evaluable": (
            "closing_value_equals_qualification_manifest.etas_numerical_negative_"
            "evidence_or_null.etas_numerical_negative_evidence_sha256"
        ),
    }
    assert public_crosswalk[
        "closing_seal_ordered_25_optimizer_sha256_must_equal_manifest_receipt_own_sha256_projection"
    ]
    assert public_crosswalk[
        "closing_seal_ordered_25_diagnostic_sha256_must_equal_manifest_row_own_sha256_projection"
    ]
    assert public_crosswalk["not_evaluable_negative_evidence_crosswalk"][
        "ordered_failure_codes"
    ] == (
        "stable_first_occurrence_deduplication_of_gate_rows_ordered_by_snapshot_then_"
        "frozen_gate_order"
    )
    assert (
        public_crosswalk[
            "any_missing_extra_duplicate_hash_value_order_branch_or_cross_snapshot_mismatch_action"
        ]
        == "invalid_execution_without_public_materialization_or_result_tag"
    )
    input_crosswalk = manifest["cross_file_input_identity_invariants"]
    assert (
        "local_restricted_input_acceptance_receipt.observed_sha256_by_input."
        "stage2_catalog_source" in input_crosswalk["source_and_reader_identity_equal_across"]
    )
    assert (
        "qualification_evidence.public_source_access_receipt.local_ledger_content_sha256"
        in input_crosswalk["global_source_ledger_identity_equal_across"]
    )
    assert input_crosswalk[
        "qualification_input_seal_public_source_access_receipt_sha256_must_recompute_from_the_exact_public_receipt_in_this_crosswalk"
    ]


def test_adapter_local_restricted_payloads_have_strict_byte_and_source_closure() -> None:
    protocol = _load_yaml(PROTOCOL_PATH)
    outputs = protocol["outputs"]
    schemas = outputs["canonical_nested_schemas"]
    local_protocol = outputs["adapter_local_restricted_artifact_protocol"]
    public_schema = outputs["adapter_public_artifact_schemas"]
    invariants = public_schema["adapter_public_result_cross_file_invariants"]
    local_equalities = invariants["local_restricted_payload_equalities"]

    assert local_protocol["exact_payload_count"] == 10
    assert local_protocol["exact_payload_count_per_kind"] == 5
    staging_root = protocol["adapter_artifact_execution_seal"]["attempt_protocol"][
        "staging_directory_template"
    ]
    assert local_protocol["root_path_template"] == f"{staging_root}/local_restricted"
    assert local_protocol[
        "root_path_template_must_equal_adapter_execution_staging_directory_template_plus_local_restricted"
    ]
    assert local_protocol["payload_manifest_schema_ref"] == (
        "outputs.canonical_nested_schemas.adapter_local_restricted_payload_manifest"
    )
    assert local_protocol["immigrant_density_artifact_schema_ref"] == (
        "outputs.canonical_nested_schemas.adapter_immigrant_density_artifact_payload"
    )
    assert local_protocol["propagation_domain_artifact_schema_ref"] == (
        "outputs.canonical_nested_schemas.adapter_propagation_domain_artifact_payload"
    )
    assert local_protocol[
        "payload_files_and_manifest_must_be_write_temp_flush_fsync_atomic_replace_then_reopened_and_byte_verified_before_public_artifact_construction"
    ]

    density = schemas["adapter_immigrant_density_artifact_payload"]
    assert density["schema_version_exact"] == 1
    assert "adapter_artifact_runtime_seal_sha256" in density["fields_exact"]
    assert density["identity_fields_exact"] == [
        field for field in density["fields_exact"] if field != "immigrant_density_artifact_sha256"
    ]
    assert density["immigrant_kde_payload_fields_exact_ref"] == (
        "fit_input_bundle.scientific_fit_input_record_schemas.immigrant_kde_payload_fields_exact"
    )
    assert density[
        "immigrant_kde_payload_must_equal_same_snapshot_positive_qualification_complete_scientific_fit_input_immigrant_kde_payload_field_for_field_and_byte_for_byte"
    ]

    propagation = schemas["adapter_propagation_domain_artifact_payload"]
    assert propagation["schema_version_exact"] == 1
    assert "adapter_artifact_runtime_seal_sha256" in propagation["fields_exact"]
    assert propagation["identity_fields_exact"] == [
        field
        for field in propagation["fields_exact"]
        if field != "propagation_domain_artifact_sha256"
    ]
    assert propagation["parent_selection_source_commit_exact_ref"] == (
        "adapter_contract.issue_forecast_definition.branching_process_domain_and_marks."
        "parent_0_2_1_propagation_membership.source_commit"
    )
    assert propagation["parent_selection_source_blob_git_oid_by_path_exact_ref"] == (
        "fit_input_bundle.parent_replay_membership_equivalence.frozen_source_blobs"
    )
    assert propagation["frozen_study_area_source_sha256_exact_source"] == (
        "local_restricted_input_acceptance_receipt.observed_sha256_by_input.study_area"
    )
    assert propagation["exact_buffer_distance_km"] == 300.0
    assert propagation["geometry_canonicalization_exact"] == (
        "GEOSNormalize_then_OGC_WKB_big_endian_2D_without_SRID_lowercase_hex"
    )

    local_manifest = schemas["adapter_local_restricted_payload_manifest"]
    assert local_manifest["schema_version_exact"] == 1
    assert "adapter_artifact_runtime_seal_sha256" in local_manifest["fields_exact"]
    assert local_manifest["identity_fields_exact"] == [
        field
        for field in local_manifest["fields_exact"]
        if field != "local_restricted_payload_content_sha256"
    ]
    assert local_manifest["snapshot_order_exact"] == list(SNAPSHOT_ORDER)
    assert local_manifest["exact_local_artifact_path_file_sha256_map_keys"] == (
        "exact_ten_paths_from_both_snapshot_record_maps"
    )
    assert local_manifest[
        "every_record_file_sha256_must_equal_sha256_of_exact_fsynced_file_bytes_at_repository_relative_path_and_same_path_map_value"
    ]
    assert local_manifest[
        "every_artifact_file_must_parse_as_its_declared_strict_schema_and_reserialize_byte_identically"
    ]

    assert local_equalities["local_restricted_payload_content_sha256_equal_across"] == [
        "recomputed_local_payload_manifest_content_sha256",
        "local_payload_manifest.local_restricted_payload_content_sha256",
        "artifact_manifest.local_restricted_payload_content_sha256",
    ]
    assert local_equalities["payload_manifest_runtime_seal_sha256_equal_across"] == [
        "local_payload_manifest.adapter_artifact_runtime_seal_sha256",
        "opening_seal.adapter_artifact_runtime_seal.adapter_artifact_runtime_seal_sha256",
        "artifact_manifest.adapter_artifact_runtime_seal_sha256",
        "global_receipt.protocol_code_environment_hashes.adapter_artifact_runtime_seal_sha256",
    ]
    for key in (
        "immigrant_density_artifact_sha256_by_snapshot_crosswalk",
        "propagation_domain_artifact_sha256_by_snapshot_crosswalk",
    ):
        crosswalk = local_equalities[key]
        assert crosswalk["all_three_maps_must_be_key_and_value_identical_in_exact_snapshot_order"]
    assert local_equalities[
        "every_local_record_file_sha256_must_equal_sha256_of_exact_reopened_payload_file_bytes_and_same_exact_path_map_value"
    ]
    assert local_equalities[
        "every_reopened_payload_file_must_strict_parse_recompute_own_content_sha_and_reserialize_to_identical_bytes"
    ]
    assert invariants["acyclic_construction_and_reference_order_exact"][2:5] == [
        "ten_local_restricted_artifact_payloads",
        "local_restricted_payload_manifest",
        "artifact_manifest",
    ]


def test_protocol_document_states_the_same_stop_boundary() -> None:
    document = PROTOCOL_DOCUMENT_PATH.read_text(encoding="utf-8")
    for required in (
        "v0.2.2-background-etas-repair-protocol-r3",
        R2_PROTOCOL_TAG_OBJECT,
        R2_PROTOCOL_COMMIT,
        "Stage 4 formal target consumer 调用 0",
        "恰好有 25 行",
        "不得复用旧 `run_local_support_etas_pipeline`",
        "FrozenETASComparatorReceipt",
        "ETASIssueForecastInput/Field",
        "ETASIssueForecastQueryNodes/Measure",
        "SeedContext",
        "4≤M<5",
        "实际 Python shared library",
        "完整规范化的原始 `OptimizeResult` payload",
        "`ordered_optimizer_call_observation_log`",
        "四个公开 seal (opening、runtime、input、closing)",
        "只能包含规定章节中的聚合数值诊断和公开协议/工件 SHA",
        "当前阶段 4 R2 继续保持目标读取前硬停",
        "data/processed/stage2R/etas_numerical_repair_fit_input/attempts/{attempt_id}/staged_public",
        "pre-closing identity 只覆盖 7 个 common 文件",
        "同一 attempt 仅重试 byte-exact public materialization",
        "local_restricted/three_grid_gate_evidence/{snapshot_id}.json",
        "6 字段 complete envelope",
        "sealed 对象仍是包括自身 SHA 的原 13 字段",
        "presence/SHA/null 真值表",
        "OBJ_DONT_REPARSE(0x1000)",
        "FileIdExtdDirectoryInfo",
        "ordered_directory_identity_chain",
        "--workspace-root",
        "Identifier[0]",
        "受控、单次的 live-capture provider",
        "indeterminate_after_install",
        "成功公开、结果提交和远端标签后这些 evidence 仍永久保留",
    ):
        assert required in document


def test_acceptance_and_restart_handoff_share_the_frozen_boundaries() -> None:
    protocol = _load_yaml(PROTOCOL_PATH)
    acceptance = PROTOCOL_ACCEPTANCE_PATH.read_text(encoding="utf-8")
    handoff = RESTART_HANDOFF_PATH.read_text(encoding="utf-8")
    protocol_document = PROTOCOL_DOCUMENT_PATH.read_text(encoding="utf-8")
    fullwidth_colon = "\uff1a"
    fullwidth_left_parenthesis = "\uff08"

    package_paths = protocol["qualification_execution_seal"]["opening_seal"][
        "protocol_package_paths"
    ]
    assert package_paths == [
        ".gitignore",
        "configs/background_etas_numerical_repair.yaml",
        "data/manifests/etas_numerical_repair_start_manifest.json",
        "docs/background_etas_numerical_repair_protocol.md",
        "docs/phase2_etas_numerical_repair_protocol_acceptance.md",
        "tests/unit/test_background_etas_numerical_repair_protocol.py",
    ]
    for package_path in package_paths:
        assert f"`{package_path}`" in acceptance
        assert f"`{package_path}`" in handoff

    for document in (acceptance, handoff):
        assert "v0.2.2-background-etas-repair-protocol" in document
        assert "v0.2.2-background-etas-repair-protocol-r1" in document
        assert "v0.2.2-background-etas-repair-protocol-r2" in document
        assert "v0.2.2-background-etas-repair-protocol-r3" in document
        assert R2_PROTOCOL_TAG_OBJECT in document
        assert R2_PROTOCOL_COMMIT in document
        assert "codex/stage2-etas-numerical-repair" in document
        assert "dae6403" in document
        assert f"阶段 9 锁定测试{fullwidth_colon}未运行" in document
        assert "not_run_upstream_gate" in document
        assert "adapter runtime preflight" in document
        assert "artifact、global receipt、报告、静态 SVG、离线 HTML" in document
        assert (
            "data/processed/stage2R/etas_numerical_repair_fit_input/attempts/"
            "{attempt_id}/staged_public"
        ) in document
        assert "staged→final" in document
        assert "post-closing" in document
        assert "local_restricted/three_grid_gate_evidence" in document
        assert "6 字段" in document
        assert "13 字段" in document
        assert "presence/SHA/null" in document
        assert "windows_ntfs_ntcreatefile_filerenameinfo_v1" in document
        assert "envelope_identity_fields_exact" in document
        assert "before_seal_return" in document
        assert "0x001201bf" in document
        assert "0xc01000a1" not in document
        assert "P0=1/P1=2/P2=1" in document
        assert "P0=1/P1=4/P2=1" in document
        assert "P0=0/P1=1/P2=1" in document
        assert "P0=0/P1=0/P2=1" in document
        assert "P0=0/P1=3/P2=2" in document
        assert "ordered_directory_identity_chain" in document
        assert "--workspace-root" in document
        assert "Identifier[0" in document
        for second_round_count in (
            "P0=0/P1=2/P2=1",
            "P0=0/P1=2/P2=0",
            "P0=0/P1=3/P2=0",
        ):
            assert second_round_count in document
        assert "INVALID/STOPPED" in document
        assert "R2→R3" in document

    assert (
        f"状态{fullwidth_colon}未通过{fullwidth_left_parenthesis}第四轮独立终审仍发现 P1/P2"
        in acceptance
    )
    assert "之后补填" not in acceptance
    assert "文档同步后的最终限定复跑、三路独立复审" not in acceptance
    assert "最终限定复跑和静态检查已经完成" in acceptance
    for r3_evidence in (
        "ModuleNotFoundError: scripts",
        "1 error in 14.17s",
        "stdout 约 70%",
        "cascade rehash",
        "同字节替换",
        "qualification_public_result_staging",
        "10 passed in 13.31s",
        "31 passed in 35.49s",
        "31 passed in 35.10s",
        "63 passed in 33.55s",
        "25 passed, 33 deselected in 24.25s",
        "58 passed in 60.16s",
        "90 passed in 66.95s",
        "90 passed in 60.85s",
        "33 passed, 34 deselected in 32.34s",
        "67 passed in 60.76s",
        "99 passed in 61.24s",
        "7 passed, 63 deselected in 10.52s",
        "2 passed, 68 deselected in 7.63s",
        "70 passed in 67.63s",
        "102 passed in 67.52s",
        "Success: no issues found in 1 source file",
        "git diff --check` 通过",
        "仅修改配置、主协议、验收记录、重启交接和协议测试五个授权 tracked 文件",
    ):
        assert r3_evidence in acceptance
    for final_evidence in (
        "1213 passed in 329.82s",
        "failures=0",
        "errors=0",
        "skipped=0",
        "406edffc53ad4f49a83638611d9aa376026882c89e67d87adcc5712da9dc2916",
        "Success: no issues found in 216 source files",
        "Runtime/Shapely 闭包与四链前像",
        "Adapter runtime、九门真值表与七文件 DAG",
        "六文件 protocol package、引用和三文档一致性",
    ):
        assert final_evidence in acceptance

    assert f"Stage 4 formal target consumer 调用{fullwidth_colon}`0`" in acceptance
    assert f"Stage 4 assessment 行物化{fullwidth_colon}`0`" in acceptance
    assert f"Stage 4 formal target consumer 调用{fullwidth_colon}0" in handoff
    assert f"assessment row 物化{fullwidth_colon}0" in handoff
    assert "## 5. 精确续接步骤" in handoff
    assert "## 6. 后续阶段计划" in handoff
    for future_stage in ("阶段 2R-A", "阶段 2R-B", "阶段 2R-C", "新阶段 4 修订"):
        assert future_stage in handoff

    assert "固定双坐标 `Point` 构造只归 synthetic warmup" in acceptance
    assert "类型化 closure cell 和 native ufunc" in acceptance
    assert "canonical_binding_path + callable_layer" in handoff
    assert (
        "ledger intent append+fsync → adapter runtime preflight+staged opening" in protocol_document
    )
    assert "七个非-publication 文件公开物化并 reopen 验证 → ledger completed" in protocol_document
