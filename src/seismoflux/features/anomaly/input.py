"""Strict, target-blind loading of registered stage-3 anomaly inputs."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from seismoflux.config import normalize_relative_path, sha256_file
from seismoflux.data.parquet import schema_sha256, table_content_sha256
from seismoflux.features.anomaly.config import (
    ALLOWED_SCIENTIFIC_DATASETS,
    AnomalyHistoryConfig,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_OBSERVATION_COLUMNS = frozenset(
    {
        "observation_id",
        "anomaly_id",
        "report_date",
        "available_at",
        "longitude",
        "latitude",
        "measurement",
        "report_state",
        "is_listed",
        "right_censored",
        "identity_complete",
        "source_file",
    }
)
_REQUIRED_REPORT_COLUMNS = frozenset(
    {
        "report_id",
        "source_file",
        "report_year",
        "report_period",
        "report_date",
        "available_at",
    }
)
_REPORT_STATES = frozenset({"新增", "持续", "取消"})


@dataclass(frozen=True, slots=True)
class RegisteredParquetInput:
    """One exact catalog-registered Parquet identity and scientific projection."""

    dataset_name: str
    path: str
    row_count: int
    file_sha256: str
    content_sha256: str
    schema_sha256: str
    source_column_allowlist: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.dataset_name not in ALLOWED_SCIENTIFIC_DATASETS:
            raise ValueError(f"stage 3 forbids scientific dataset: {self.dataset_name}")
        object.__setattr__(self, "path", normalize_relative_path(self.path))
        if self.row_count < 0:
            raise ValueError("registered row_count must be non-negative")
        for label, value in (
            ("file_sha256", self.file_sha256),
            ("content_sha256", self.content_sha256),
            ("schema_sha256", self.schema_sha256),
        ):
            if _SHA256.fullmatch(value) is None:
                raise ValueError(f"{label} must be a lowercase SHA-256 digest")
        if not self.source_column_allowlist or len(self.source_column_allowlist) != len(
            set(self.source_column_allowlist)
        ):
            raise ValueError("source_column_allowlist must be non-empty and unique")
        required = (
            _REQUIRED_OBSERVATION_COLUMNS
            if self.dataset_name == "anomaly_observation"
            else _REQUIRED_REPORT_COLUMNS
        )
        missing = sorted(required - set(self.source_column_allowlist))
        if missing:
            raise ValueError(
                f"source_column_allowlist for {self.dataset_name} is missing required "
                f"columns: {missing}"
            )


@dataclass(frozen=True, slots=True)
class RegisteredStudyAreaInput:
    """Local target-independent geometry used only to construct the fixed query grid."""

    path: str
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", normalize_relative_path(self.path))
        if _SHA256.fullmatch(self.sha256) is None:
            raise ValueError("study-area sha256 must be a lowercase SHA-256 digest")


@dataclass(frozen=True, slots=True)
class Stage3InputSpec:
    """All and only the identities stage 3 is authorized to load."""

    data_catalog_path: str
    data_catalog_sha256: str
    expected_stage1_snapshot_id: str
    observation: RegisteredParquetInput
    report_period: RegisteredParquetInput
    study_area: RegisteredStudyAreaInput
    expected_report_period_count: int
    query_grid_cell_km: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "data_catalog_path",
            normalize_relative_path(self.data_catalog_path),
        )
        if _SHA256.fullmatch(self.data_catalog_sha256) is None:
            raise ValueError("data-catalog sha256 must be a lowercase SHA-256 digest")
        if not self.expected_stage1_snapshot_id:
            raise ValueError("expected_stage1_snapshot_id must not be empty")
        require_stage3_dataset_boundary(
            (self.observation.dataset_name, self.report_period.dataset_name)
        )
        if self.expected_report_period_count <= 0:
            raise ValueError("expected_report_period_count must be positive")
        if self.query_grid_cell_km != 25.0:
            raise ValueError("stage-3 target-independent query grid must remain 25 km")


@dataclass(frozen=True, slots=True)
class LoadedStage3Inputs:
    """Verified in-memory scientific payload with forbidden columns projected away."""

    observation_table: pa.Table
    report_period_table: pa.Table
    study_area_path: Path
    study_area_document: dict[str, object]
    data_catalog_sha256: str
    stage1_snapshot_id: str


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _string(mapping: Mapping[str, object], key: str, *, label: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label}.{key} must be a non-empty string")
    return value


def _integer(mapping: Mapping[str, object], key: str, *, label: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label}.{key} must be an integer")
    return value


def _string_tuple(mapping: Mapping[str, object], key: str, *, label: str) -> tuple[str, ...]:
    value = mapping.get(key)
    if not isinstance(value, list | tuple) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"{label}.{key} must be a string sequence")
    return tuple(value)


def require_stage3_dataset_boundary(dataset_names: Sequence[str]) -> None:
    """Reject earthquakes, labels, G1 support masks, Mc, and every extra dataset."""

    names = tuple(dataset_names)
    if names != ALLOWED_SCIENTIFIC_DATASETS:
        extras = sorted(set(names) - set(ALLOWED_SCIENTIFIC_DATASETS))
        raise ValueError(
            "stage 3 scientific inputs must be exactly anomaly_observation and "
            "anomaly_report_period; earthquake catalogs/labels, G1 support masks, "
            f"completeness Mc, and all extras are forbidden: {extras}"
        )


def stage3_input_spec_from_config(config: AnomalyHistoryConfig) -> Stage3InputSpec:
    """Extract the only authorized identities from the complete frozen protocol."""

    scientific = config.scientific_inputs
    dataset_names = _string_tuple(scientific, "exact_dataset_names", label="scientific_inputs")
    require_stage3_dataset_boundary(dataset_names)
    if scientific.get("every_other_scientific_dataset_forbidden") is not True:
        raise ValueError("every non-anomaly scientific dataset must remain forbidden")
    for key in (
        "earthquake_catalog_forbidden",
        "earthquake_target_labels_forbidden",
        "human_forecast_or_result_dataset_forbidden",
    ):
        if scientific.get(key) is not True:
            raise ValueError(f"scientific_inputs.{key} must remain true")

    forbidden_columns = frozenset(
        _string_tuple(
            config.forbidden_source_semantics, "fields", label="forbidden_source_semantics"
        )
    )

    def registered_dataset(name: str) -> RegisteredParquetInput:
        values = _mapping(scientific.get(name), label=f"scientific_inputs.{name}")
        catalog_name = _string(values, "catalog_dataset_name", label=name)
        if catalog_name != name:
            raise ValueError(f"catalog dataset name drift for {name}")
        allowlist = _string_tuple(values, "source_column_allowlist", label=name)
        disabled = sorted(set(allowlist) & forbidden_columns)
        if disabled:
            raise ValueError(f"disabled source columns entered the {name} allowlist: {disabled}")
        return RegisteredParquetInput(
            dataset_name=name,
            path=_string(values, "expected_path", label=name),
            row_count=_integer(values, "expected_row_count", label=name),
            file_sha256=_string(values, "expected_file_sha256", label=name),
            content_sha256=_string(values, "expected_content_sha256", label=name),
            schema_sha256=_string(values, "expected_schema_sha256", label=name),
            source_column_allowlist=allowlist,
        )

    study = _mapping(scientific.get("study_area"), label="scientific_inputs.study_area")
    if (
        study.get("role") != "local_query_grid_only"
        or study.get("public_redistribution") != "forbidden"
    ):
        raise ValueError("study area must remain local and query-grid-only")
    if config.query_grid.get("cell_size_km") != 25.0:
        raise ValueError("stage-3 target-independent query grid must remain 25 km")
    if config.query_grid.get("earthquake_or_target_derived_cells") != "forbidden":
        raise ValueError("earthquake- or target-derived query cells are forbidden")

    return Stage3InputSpec(
        data_catalog_path=_string(scientific, "data_catalog", label="scientific_inputs"),
        data_catalog_sha256=_string(
            scientific,
            "data_catalog_sha256",
            label="scientific_inputs",
        ),
        expected_stage1_snapshot_id=_string(
            scientific,
            "expected_stage1_snapshot_id",
            label="scientific_inputs",
        ),
        observation=registered_dataset("anomaly_observation"),
        report_period=registered_dataset("anomaly_report_period"),
        study_area=RegisteredStudyAreaInput(
            path=_string(study, "path", label="study_area"),
            sha256=_string(study, "sha256", label="study_area"),
        ),
        expected_report_period_count=config.expected_report_period_count,
        query_grid_cell_km=config.query_grid_cell_km,
    )


def _safe_file(project_root: Path, relative_path: str, *, label: str) -> Path:
    root = project_root.resolve()
    path = (root / normalize_relative_path(relative_path)).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"missing or unsafe {label}: {relative_path}")
    return path


def _load_catalog(spec: Stage3InputSpec, project_root: Path) -> Mapping[str, object]:
    path = _safe_file(project_root, spec.data_catalog_path, label="data catalog")
    actual_hash = sha256_file(path)
    if actual_hash != spec.data_catalog_sha256:
        raise ValueError(
            "data catalog hash mismatch: "
            f"expected={spec.data_catalog_sha256}, observed={actual_hash}"
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("data catalog must be valid UTF-8 JSON") from exc
    if sha256_file(path) != spec.data_catalog_sha256:
        raise ValueError("data catalog changed while being read")
    catalog = _mapping(raw, label="data catalog")
    if catalog.get("schema_version") != 1 or catalog.get("contract_version") != "0.1.0":
        raise ValueError("unsupported stage-1 data catalog version")
    if catalog.get("snapshot_id") != spec.expected_stage1_snapshot_id:
        raise ValueError("stage-1 snapshot identity mismatch")
    if catalog.get("standardized_data_committed_to_git") is not False:
        raise ValueError("stage-1 standardized data must remain local and uncommitted")
    return catalog


def _catalog_dataset_entry(
    catalog: Mapping[str, object],
    identity: RegisteredParquetInput,
) -> Mapping[str, object]:
    datasets = _mapping(catalog.get("datasets"), label="data catalog datasets")
    entry = _mapping(
        datasets.get(identity.dataset_name),
        label=f"catalog dataset {identity.dataset_name}",
    )
    expected: dict[str, object] = {
        "path": identity.path,
        "row_count": identity.row_count,
        "file_sha256": identity.file_sha256,
        "content_sha256": identity.content_sha256,
        "schema_sha256": identity.schema_sha256,
    }
    for key, expected_value in expected.items():
        if entry.get(key) != expected_value:
            raise ValueError(f"catalog identity mismatch for {identity.dataset_name}.{key}")
    return entry


def _field_documents(schema: pa.Schema) -> list[dict[str, object]]:
    return [
        {"name": field.name, "type": str(field.type), "nullable": field.nullable}
        for field in schema
    ]


def _load_registered_parquet(
    *,
    project_root: Path,
    identity: RegisteredParquetInput,
    catalog_entry: Mapping[str, object],
) -> pa.Table:
    path = _safe_file(project_root, identity.path, label=identity.dataset_name)
    observed_file_hash = sha256_file(path)
    if observed_file_hash != identity.file_sha256:
        raise ValueError(f"file hash mismatch for {identity.dataset_name}")

    full_table = pq.read_table(path)
    if sha256_file(path) != identity.file_sha256:
        raise ValueError(f"file changed while being read for {identity.dataset_name}")
    if full_table.num_rows != identity.row_count:
        raise ValueError(f"row count mismatch for {identity.dataset_name}")
    if table_content_sha256(full_table) != identity.content_sha256:
        raise ValueError(f"content hash mismatch for {identity.dataset_name}")
    if schema_sha256(full_table.schema) != identity.schema_sha256:
        raise ValueError(f"schema hash mismatch for {identity.dataset_name}")
    if catalog_entry.get("fields") != _field_documents(full_table.schema):
        raise ValueError(f"catalog field schema mismatch for {identity.dataset_name}")

    sort_keys_value = catalog_entry.get("sort_keys")
    if not isinstance(sort_keys_value, list) or not all(
        isinstance(key, str) for key in sort_keys_value
    ):
        raise ValueError(f"invalid catalog sort keys for {identity.dataset_name}")
    sort_keys = tuple(sort_keys_value)
    if not set(sort_keys) <= set(full_table.column_names):
        raise ValueError(f"missing catalog sort key for {identity.dataset_name}")
    if full_table.num_rows:
        sorted_table = full_table.sort_by([(key, "ascending") for key in sort_keys])
        if not full_table.equals(sorted_table, check_metadata=True):
            raise ValueError(f"row order mismatch for {identity.dataset_name}")

    missing_allowlist = sorted(set(identity.source_column_allowlist) - set(full_table.column_names))
    if missing_allowlist:
        raise ValueError(
            f"registered allowlist columns missing from {identity.dataset_name}: "
            f"{missing_allowlist}"
        )
    projected = full_table.select(list(identity.source_column_allowlist))
    if tuple(projected.column_names) != identity.source_column_allowlist:
        raise AssertionError("scientific payload projection differs from the frozen allowlist")
    return projected


def select_allowlisted_columns(
    table: pa.Table,
    requested_columns: Sequence[str],
    *,
    identity: RegisteredParquetInput,
) -> pa.Table:
    """Select a downstream payload only after proving every request was preregistered."""

    requested = tuple(requested_columns)
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("requested columns must be non-empty and unique")
    unauthorized = sorted(set(requested) - set(identity.source_column_allowlist))
    if unauthorized:
        raise ValueError(f"columns were not authorized for {identity.dataset_name}: {unauthorized}")
    missing = sorted(set(requested) - set(table.column_names))
    if missing:
        raise ValueError(f"authorized columns are missing from {identity.dataset_name}: {missing}")
    return table.select(list(requested))


def _aware_utc(value: object, *, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be a timezone-aware datetime")
    normalized = value.astimezone(UTC)
    offset = normalized.utcoffset()
    if offset is None or offset.total_seconds() != 0:
        raise ValueError(f"{label} must be convertible to UTC")
    return normalized


def _plain_date(value: object, *, label: str) -> date:
    if isinstance(value, datetime) or not isinstance(value, date):
        raise ValueError(f"{label} must be a date")
    return value


def _finite_coordinate(value: object, *, label: str, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or not lower <= numeric <= upper:
        raise ValueError(f"{label} is outside WGS84 bounds")
    return numeric


def _require_columns(table: pa.Table, required: frozenset[str], *, label: str) -> None:
    missing = sorted(required - set(table.column_names))
    if missing:
        raise ValueError(f"{label} is missing required scientific columns: {missing}")


def validate_stage3_input_tables(
    observation_table: pa.Table,
    report_period_table: pa.Table,
    *,
    expected_report_period_count: int,
) -> None:
    """Validate schedule, causal row availability, identities, and WGS84 semantics."""

    _require_columns(
        observation_table,
        _REQUIRED_OBSERVATION_COLUMNS,
        label="anomaly_observation",
    )
    _require_columns(
        report_period_table,
        _REQUIRED_REPORT_COLUMNS,
        label="anomaly_report_period",
    )
    if report_period_table.num_rows != expected_report_period_count:
        raise ValueError(
            "anomaly report-period count mismatch: "
            f"expected={expected_report_period_count}, observed={report_period_table.num_rows}"
        )

    period_rows = report_period_table.select(sorted(_REQUIRED_REPORT_COLUMNS)).to_pylist()
    report_ids: set[str] = set()
    report_keys: set[tuple[date, str]] = set()
    year_period_keys: set[tuple[int, int]] = set()
    issue_times: list[datetime] = []
    issue_by_key: dict[tuple[date, str], datetime] = {}
    previous_report_date: date | None = None
    previous_issue: datetime | None = None
    for row in period_rows:
        report_id = row["report_id"]
        source_file = row["source_file"]
        report_year = row["report_year"]
        report_period = row["report_period"]
        if not isinstance(report_id, str) or not report_id:
            raise ValueError("report_id must be a non-empty string")
        if not isinstance(source_file, str) or not source_file:
            raise ValueError("report source_file must be a non-empty string")
        if isinstance(report_year, bool) or not isinstance(report_year, int):
            raise ValueError("report_year must be an integer")
        if isinstance(report_period, bool) or not isinstance(report_period, int):
            raise ValueError("report_period must be an integer")
        report_date = _plain_date(row["report_date"], label="report_date")
        issue = _aware_utc(row["available_at"], label="report available_at")
        key = (report_date, source_file)
        if report_id in report_ids or key in report_keys:
            raise ValueError("report-period identities must be unique")
        if (report_year, report_period) in year_period_keys:
            raise ValueError("report_year/report_period identities must be unique")
        if previous_report_date is not None and report_date <= previous_report_date:
            raise ValueError("report-period dates must be strictly increasing")
        if previous_issue is not None and issue <= previous_issue:
            raise ValueError("report-period available_at values must be strictly increasing")
        report_ids.add(report_id)
        report_keys.add(key)
        year_period_keys.add((report_year, report_period))
        issue_times.append(issue)
        issue_by_key[key] = issue
        previous_report_date = report_date
        previous_issue = issue
    if len(set(issue_times)) != expected_report_period_count:
        raise ValueError("report-period available_at identities must be unique")

    observation_rows = observation_table.select(sorted(_REQUIRED_OBSERVATION_COLUMNS)).to_pylist()
    observation_ids: set[str] = set()
    temporary_anomaly_ids: set[str] = set()
    for row in observation_rows:
        observation_id = row["observation_id"]
        anomaly_id = row["anomaly_id"]
        source_file = row["source_file"]
        if not isinstance(observation_id, str) or not observation_id:
            raise ValueError("observation_id must be a non-empty string")
        if observation_id in observation_ids:
            raise ValueError("observation_id identities must be unique")
        observation_ids.add(observation_id)
        if not isinstance(anomaly_id, str) or not anomaly_id:
            raise ValueError("anomaly_id must be a non-empty string")
        if not isinstance(source_file, str) or not source_file:
            raise ValueError("observation source_file must be a non-empty string")
        report_date = _plain_date(row["report_date"], label="observation report_date")
        report_issue = issue_by_key.get((report_date, source_file))
        if report_issue is None:
            raise ValueError("observation does not resolve to a registered report period")
        available_at = _aware_utc(row["available_at"], label="observation available_at")
        if available_at < report_issue:
            raise ValueError("observation available_at precedes its report-period availability")

        identity_complete = row["identity_complete"]
        right_censored = row["right_censored"]
        is_listed = row["is_listed"]
        if not isinstance(identity_complete, bool) or not isinstance(right_censored, bool):
            raise ValueError("identity_complete and right_censored must be booleans")
        if is_listed is not True:
            raise ValueError("every anomaly_observation row must represent a listed row")
        if not identity_complete:
            if anomaly_id in temporary_anomaly_ids:
                raise ValueError("incomplete anomaly identities may not link across observations")
            temporary_anomaly_ids.add(anomaly_id)

        report_state = row["report_state"]
        if report_state not in _REPORT_STATES:
            raise ValueError(f"unsupported anomaly report_state: {report_state}")
        longitude = row["longitude"]
        latitude = row["latitude"]
        if (longitude is None) != (latitude is None):
            raise ValueError("longitude and latitude must be jointly present or missing")
        if longitude is not None and latitude is not None:
            _finite_coordinate(longitude, label="longitude", lower=-180.0, upper=180.0)
            _finite_coordinate(latitude, label="latitude", lower=-90.0, upper=90.0)
        if identity_complete and (
            longitude is None or latitude is None or row["measurement"] is None
        ):
            raise ValueError("complete anomaly identity requires coordinates and measurement")


def _load_study_area(
    *,
    project_root: Path,
    identity: RegisteredStudyAreaInput,
    catalog: Mapping[str, object],
) -> tuple[Path, dict[str, object]]:
    catalog_study = _mapping(catalog.get("study_area"), label="catalog study_area")
    if catalog_study.get("path") != identity.path or catalog_study.get("sha256") != identity.sha256:
        raise ValueError("catalog study-area identity mismatch")
    catalog_properties = _mapping(
        catalog_study.get("properties"),
        label="catalog study_area properties",
    )
    if catalog_properties.get("target_independent") is not True:
        raise ValueError("catalog study area must be explicitly target-independent")
    if catalog_properties.get("predictor_feature") is not False:
        raise ValueError("catalog study area must not be a predictor feature")
    path = _safe_file(project_root, identity.path, label="study area")
    if sha256_file(path) != identity.sha256:
        raise ValueError("study-area file hash mismatch")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("study area must be valid UTF-8 GeoJSON") from exc
    if sha256_file(path) != identity.sha256:
        raise ValueError("study-area file changed while being read")
    document = dict(_mapping(raw, label="study area GeoJSON"))
    geometry = _mapping(document.get("geometry"), label="study area geometry")
    properties = _mapping(document.get("properties"), label="study area properties")
    if document.get("type") != "Feature" or geometry.get("type") not in {
        "Polygon",
        "MultiPolygon",
    }:
        raise ValueError("study area must be one polygonal GeoJSON Feature")
    if not geometry.get("coordinates"):
        raise ValueError("study-area geometry must not be empty")
    if properties.get("target_independent") is not True:
        raise ValueError("study area must be explicitly target-independent")
    if properties.get("predictor_feature") is not False:
        raise ValueError("study-area geometry may define the query grid but not a predictor")
    return path, document


def load_stage3_inputs(spec: Stage3InputSpec, project_root: Path) -> LoadedStage3Inputs:
    """Verify registered identities and return only allowlisted anomaly columns."""

    require_stage3_dataset_boundary(
        (spec.observation.dataset_name, spec.report_period.dataset_name)
    )
    catalog = _load_catalog(spec, project_root)
    observation_entry = _catalog_dataset_entry(catalog, spec.observation)
    report_entry = _catalog_dataset_entry(catalog, spec.report_period)
    observation_table = _load_registered_parquet(
        project_root=project_root,
        identity=spec.observation,
        catalog_entry=observation_entry,
    )
    report_table = _load_registered_parquet(
        project_root=project_root,
        identity=spec.report_period,
        catalog_entry=report_entry,
    )
    validate_stage3_input_tables(
        observation_table,
        report_table,
        expected_report_period_count=spec.expected_report_period_count,
    )
    study_area_path, study_area_document = _load_study_area(
        project_root=project_root,
        identity=spec.study_area,
        catalog=catalog,
    )
    return LoadedStage3Inputs(
        observation_table=observation_table,
        report_period_table=report_table,
        study_area_path=study_area_path,
        study_area_document=study_area_document,
        data_catalog_sha256=spec.data_catalog_sha256,
        stage1_snapshot_id=spec.expected_stage1_snapshot_id,
    )


__all__ = [
    "LoadedStage3Inputs",
    "RegisteredParquetInput",
    "RegisteredStudyAreaInput",
    "Stage3InputSpec",
    "load_stage3_inputs",
    "require_stage3_dataset_boundary",
    "select_allowlisted_columns",
    "stage3_input_spec_from_config",
    "validate_stage3_input_tables",
]
