"""Two-phase runtime for the frozen, score-blind S1-C0 development screen.

The prediction entry point deliberately has no import of ``development_score``.
It authenticates the frozen inputs, selects parameters only on the strictly
earlier inner blocks, writes one numeric-only NPZ per development fold, and
seals all four predictions.  The scoring module is imported locally only after
the complete seal has been verified, the four NPZ files have been loaded with
``allow_pickle=False``, and the seal has been verified a second time.

This module does not bootstrap, choose a champion, open a holdout, run an audit,
or run a locked test.  Its score-phase summary is only an inventory of raw rows;
scientific aggregation and value review remain an explicit subsequent step.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Literal, cast

import numpy as np
import yaml
from numpy.typing import NDArray

from seismoflux.multitask_s1.development_contract import (
    DEVELOPMENT_FOLD_IDS,
    HORIZONS_DAYS,
)
from seismoflux.multitask_s1.development_predict import (
    LOCATION_MODEL_IDS,
    NB2_REASON_CODES,
    NB2_STATUS_CODES,
    TIME_BANDS,
    FoldHorizonParameterSelection,
    build_development_fold_prediction,
    fold_prediction_npz_arrays,
    frozen_fold_prediction_npz_schema,
    select_fold_horizon_parameters,
    validate_frozen_fold_prediction_npz_arrays,
)
from seismoflux.multitask_s1.prediction_seal import (
    DevelopmentScoringAuthorization,
    PredictionArtifactInput,
    PredictionInputIdentities,
    authorize_development_scoring,
    canonical_json_bytes,
    frozen_fold_prediction_manifest,
    recompute_prediction_input_identities,
    seal_fold_prediction,
    seal_four_fold_predictions,
    verify_fold_prediction,
)
from seismoflux.multitask_s1.runner_inputs import (
    EXPECTED_25KM_CELL_COUNT,
    S1RunnerInputs,
    load_s1_runner_inputs,
)
from seismoflux.multitask_s1.time_magnitude import (
    GR_BIN_WIDTH,
    GR_MAXIMUM_MAGNITUDE,
    LONG_M5_GR_MC,
    MAIN_GR_MC,
    NB2DispersionQualification,
    TruncatedGRMagnitudeModel,
)

if TYPE_CHECKING:
    from seismoflux.multitask_s1.development_score import (
        CountForecast,
        DevelopmentRawScores,
        LocationForecast,
        MagnitudeForecast,
    )

JsonScalar = None | bool | int | float | str
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
NumericArray = NDArray[np.generic]
Phase = Literal["predict", "score"]

RUN_CONTRACT_RELATIVE_PATH: Final = Path("configs/multitask_s1_development_run.yaml")
RUN_CONTRACT_ID: Final = "multitask-s1-c0-all-m4-screen-v1"
PREDICTION_DIRECTORY_NAME: Final = "prediction_phase"
SCORE_DIRECTORY_NAME: Final = "score_phase"
PREDICTION_FILE_NAME: Final = "predictions.npz"
RAW_SCORE_FILE_NAME: Final = "raw_scores.parquet"
_SHA256: Final = re.compile(r"[0-9a-f]{64}")
_UTC_EPOCH: Final = datetime(1970, 1, 1, tzinfo=UTC)
_NUMERICAL_THREAD_ENVIRONMENT: Final = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
_TIME_BAND_FROM_SCORE: Final[Mapping[str, str]] = MappingProxyType(
    {
        "M5_6": "m5_6",
        "M6_plus": "m6_plus",
        "M5_plus_for_joint": "m5_plus_1970_for_joint",
    }
)


class DevelopmentRuntimeError(RuntimeError):
    """Raised when the frozen two-phase runtime fails closed."""


class RuntimeArtifactExistsError(FileExistsError):
    """Raised rather than overwriting any historical prediction or score file."""


@dataclass(frozen=True, slots=True)
class PhaseRoots:
    """The non-overlapping sibling directories of one S1-C0 run."""

    overall: Path
    prediction: Path
    score: Path


@dataclass(frozen=True, slots=True)
class PredictionPhaseResult:
    """Identity of a completed four-fold prediction seal."""

    overall_root: Path
    prediction_root: Path
    master_seal_sha256: str
    fold_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ScorePhaseResult:
    """Paths and raw row counts from authorized, unaggregated development scoring."""

    overall_root: Path
    score_root: Path
    authorization_path: Path
    raw_scores_path: Path
    summary_path: Path
    raw_score_row_counts: Mapping[str, int]


def configure_single_thread_numerics() -> None:
    """Limit each fold worker's numerical libraries to one thread."""

    for name in _NUMERICAL_THREAD_ENVIRONMENT:
        os.environ[name] = "1"


def phase_roots(output_root: str | Path) -> PhaseRoots:
    """Resolve one explicit overall root into two distinct sibling phase roots."""

    raw = Path(output_root).expanduser()
    if not raw.is_absolute():
        raise DevelopmentRuntimeError("output_root must be an explicit absolute path")
    overall = raw.resolve()
    if overall.name in {PREDICTION_DIRECTORY_NAME, SCORE_DIRECTORY_NAME}:
        raise DevelopmentRuntimeError("output_root must be the overall root, not a phase root")
    prediction = (overall / PREDICTION_DIRECTORY_NAME).resolve()
    score = (overall / SCORE_DIRECTORY_NAME).resolve()
    if prediction == score or prediction.parent != overall or score.parent != overall:
        raise AssertionError("prediction and score roots must be distinct siblings")
    return PhaseRoots(overall=overall, prediction=prediction, score=score)


def _explicit_existing_root(value: str | Path, *, label: str) -> Path:
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        raise DevelopmentRuntimeError(f"{label} must be an explicit absolute path")
    try:
        result = raw.resolve(strict=True)
    except OSError as exc:
        raise DevelopmentRuntimeError(f"{label} does not exist or cannot be resolved") from exc
    if not result.is_dir():
        raise DevelopmentRuntimeError(f"{label} must be a directory")
    return result


def _require_configured_overall_root(project_root: Path, overall_root: Path) -> None:
    """Bind the explicit output root to the frozen run YAML, not caller convention."""

    path = project_root / RUN_CONTRACT_RELATIVE_PATH
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise DevelopmentRuntimeError("frozen S1-C0 run contract cannot be read") from exc
    if not isinstance(raw, dict) or raw.get("run_contract_id") != RUN_CONTRACT_ID:
        raise DevelopmentRuntimeError("frozen S1-C0 run contract identity changed")
    outputs = raw.get("outputs")
    if not isinstance(outputs, dict) or not isinstance(outputs.get("root"), str):
        raise DevelopmentRuntimeError("run contract does not define its overall output root")
    relative = Path(cast(str, outputs["root"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise DevelopmentRuntimeError("configured output root must be project-relative and scoped")
    expected = (project_root / relative).resolve()
    if overall_root != expected:
        raise DevelopmentRuntimeError(
            f"output_root differs from the frozen run contract: expected {expected}"
        )
    if outputs.get("prediction_root") != (relative / PREDICTION_DIRECTORY_NAME).as_posix():
        raise DevelopmentRuntimeError("configured prediction root changed")
    if outputs.get("score_root") != (relative / SCORE_DIRECTORY_NAME).as_posix():
        raise DevelopmentRuntimeError("configured score root changed")
    if outputs.get("phase_roots_must_be_siblings_and_nonoverlapping") is not True:
        raise DevelopmentRuntimeError("run contract no longer requires separate phase roots")


def _maximum_fold_workers(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 3:
        raise DevelopmentRuntimeError("maximum_fold_workers must be an integer from 1 to 3")
    logical = os.cpu_count()
    if logical is not None:
        conservative_physical = max(1, logical // 2)
        if conservative_physical <= 2 or value > conservative_physical - 2:
            raise DevelopmentRuntimeError(
                "requested fold workers would not conservatively reserve two physical cores"
            )
    return value


def _write_exclusive_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | cast(int, getattr(os, "O_BINARY", 0))
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise RuntimeArtifactExistsError(f"immutable artifact already exists: {path}") from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        # A partial file remains visible and blocks reuse of this historical root.
        # That is intentional fail-closed behavior; it is never silently replaced.
        raise


def _write_exclusive_json(path: Path, payload: object) -> None:
    _write_exclusive_bytes(path, canonical_json_bytes(payload))


def _write_or_verify_canonical_json(path: Path, payload: object) -> None:
    """Create once, or reuse only an exact canonical checkpoint from this run."""

    expected = canonical_json_bytes(payload)
    if not path.exists():
        _write_exclusive_bytes(path, expected)
        return
    try:
        actual = path.read_bytes()
    except OSError as exc:
        raise DevelopmentRuntimeError(f"checkpoint cannot be read: {path}") from exc
    if actual != expected:
        raise DevelopmentRuntimeError(
            f"existing checkpoint differs byte-for-byte from this frozen run: {path}"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise DevelopmentRuntimeError(f"checkpoint cannot be hashed: {path}") from exc
    return digest.hexdigest()


def _write_npz_exclusive(
    path: Path,
    *,
    fold_id: str,
    arrays: Mapping[str, object],
    cell_count: int,
) -> None:
    """Write one canonical NPZ once, then reload it safely and revalidate it."""

    validate_frozen_fold_prediction_npz_arrays(fold_id, arrays, cell_count=cell_count)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | cast(int, getattr(os, "O_BINARY", 0))
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise RuntimeArtifactExistsError(
            f"historical fold prediction cannot be overwritten: {path}"
        ) from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez(
                stream,
                **{  # type: ignore[arg-type]
                    name: np.asarray(value) for name, value in arrays.items()
                },
            )
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        raise
    reloaded = _load_one_npz(path, fold_id=fold_id, cell_count=cell_count)
    validate_frozen_fold_prediction_npz_arrays(fold_id, reloaded, cell_count=cell_count)


def _load_one_npz(path: Path, *, fold_id: str, cell_count: int) -> Mapping[str, NumericArray]:
    try:
        loaded = np.load(path, allow_pickle=False)
        with loaded as archive:
            names = tuple(archive.files)
            if len(names) != len(set(names)):
                raise DevelopmentRuntimeError(f"prediction NPZ has duplicate arrays: {path}")
            arrays = {name: np.array(archive[name], copy=True, order="C") for name in names}
    except (OSError, EOFError, ValueError) as exc:
        raise DevelopmentRuntimeError(f"prediction NPZ is unsafe or unreadable: {path}") from exc
    validate_frozen_fold_prediction_npz_arrays(fold_id, arrays, cell_count=cell_count)
    for array in arrays.values():
        array.setflags(write=False)
    return MappingProxyType(arrays)


def _candidate_evidence(values: Sequence[Any]) -> list[JsonValue]:
    result: list[JsonValue] = []
    for value in values:
        parameter = float(value.parameter_value)
        block_scores = tuple(float(item) for item in value.block_mean_log_density)
        if len(block_scores) != 3:
            raise DevelopmentRuntimeError("inner candidate evidence lost one of I1/I2/I3")
        result.append(
            {
                "parameter_value": parameter,
                "inner_block_mean_log_density": list(block_scores),
            }
        )
    return result


def _qualification_evidence(value: NB2DispersionQualification) -> dict[str, JsonValue]:
    return {
        "status": value.status,
        "reason": value.reason,
        "historical_block_count": value.historical_block_count,
        "sample_mean_count": value.sample_mean_count,
        "sample_variance_count": value.sample_variance_count,
        "dispersion_k": value.dispersion_k,
        "observed_information_k": value.observed_information_k,
        "standard_error_k": value.standard_error_k,
    }


def _selection_evidence(value: FoldHorizonParameterSelection) -> dict[str, JsonValue]:
    location = value.location
    if value.fold_id != location.fold_id or value.horizon_days != location.horizon_days:
        raise DevelopmentRuntimeError("inner parameter-selection identity changed")
    return {
        "horizon_days": value.horizon_days,
        "inner_evidence": {
            "location": {
                "latest_inner_target_end_us": location.boundary.latest_inner_target_end_us,
                "evaluation_start_boundary_us": location.boundary.outer_evaluation_start_us,
                "inner_block_event_counts": list(location.inner_block_event_counts),
                "selected_regional_tau_years": location.regional_tau_years,
                "selected_kde_bandwidth_km": location.selected_bandwidth_km,
                "selected_recent_alpha": location.recent_alpha,
                "regional_candidates": _candidate_evidence(location.regional_candidates),
                "kde_candidates": _candidate_evidence(location.kde_candidates),
                "recent_candidates": _candidate_evidence(location.recent_candidates),
            },
            "time": [
                {
                    "band": item.band,
                    "qualification": _qualification_evidence(item.qualification),
                }
                for item in value.t1
            ],
        },
    }


def _parameter_selection_payload(
    selections: Mapping[str, Sequence[FoldHorizonParameterSelection]],
) -> dict[str, JsonValue]:
    if tuple(selections) != DEVELOPMENT_FOLD_IDS:
        raise DevelopmentRuntimeError("inner selections must cover exactly four ordered folds")
    folds: dict[str, JsonValue] = {}
    for fold_id in DEVELOPMENT_FOLD_IDS:
        values = tuple(selections[fold_id])
        if tuple((item.fold_id, item.horizon_days) for item in values) != tuple(
            (fold_id, horizon) for horizon in HORIZONS_DAYS
        ):
            raise DevelopmentRuntimeError("each fold needs five ordered inner selections")
        folds[fold_id] = [_selection_evidence(item) for item in values]
    return {
        "schema_version": 1,
        "record_type": "s1_c0_inner_parameter_selection",
        "run_contract_id": RUN_CONTRACT_ID,
        "role": "strictly_earlier_inner_selection_only",
        "folds": folds,
    }


def _run_manifest_payload(
    identities: PredictionInputIdentities, *, maximum_fold_workers: int
) -> dict[str, JsonValue]:
    return {
        "schema_version": 1,
        "record_type": "s1_c0_prediction_run_manifest",
        "run_contract_id": RUN_CONTRACT_ID,
        "stage": "S1-C0",
        "role": "development_prediction_only",
        "git_commit_oid": identities.git_commit_oid,
        "input_identities": cast(dict[str, JsonValue], identities.as_mapping()),
        "fold_ids": list(DEVELOPMENT_FOLD_IDS),
        "maximum_fold_workers": maximum_fold_workers,
        "numerical_threads_per_worker": 1,
        "outer_targets_constructed": False,
        "model_scores_read": False,
        "locked_test_run": False,
    }


def _select_one_fold(
    inputs: S1RunnerInputs, fold_id: str
) -> tuple[FoldHorizonParameterSelection, ...]:
    return tuple(
        select_fold_horizon_parameters(inputs, fold_id=fold_id, horizon_days=horizon)
        for horizon in HORIZONS_DAYS
    )


def _verify_existing_fold(
    prediction_root: Path,
    fold_id: str,
    *,
    project_root: Path,
    data_root: Path,
) -> None:
    verify_fold_prediction(
        prediction_root,
        fold_id,
        project_root=project_root,
        data_root=data_root,
    )


def _checkpoint_fold_state(
    prediction_root: Path,
    fold_id: str,
    *,
    project_root: Path,
    data_root: Path,
    cell_count: int,
) -> Literal["complete", "payload_only", "missing"]:
    prediction_path = prediction_root / "folds" / fold_id / PREDICTION_FILE_NAME
    bundle_path = prediction_root / "folds" / fold_id / "prediction_bundle.json"
    prediction_exists = prediction_path.exists()
    bundle_exists = bundle_path.exists()
    if bundle_exists and not prediction_exists:
        raise DevelopmentRuntimeError(
            f"fold bundle exists without its prediction payload: {fold_id}"
        )
    if prediction_exists and bundle_exists:
        _verify_existing_fold(
            prediction_root,
            fold_id,
            project_root=project_root,
            data_root=data_root,
        )
        return "complete"
    if prediction_exists:
        _load_one_npz(prediction_path, fold_id=fold_id, cell_count=cell_count)
        return "payload_only"
    return "missing"


def run_prediction_phase(
    *,
    project_root: str | Path,
    data_root: str | Path,
    output_root: str | Path,
    maximum_fold_workers: int = 3,
) -> PredictionPhaseResult:
    """Generate and seal all four target-free development predictions."""

    configure_single_thread_numerics()
    workers = _maximum_fold_workers(maximum_fold_workers)
    project = _explicit_existing_root(project_root, label="project_root")
    data = _explicit_existing_root(data_root, label="data_root")
    roots = phase_roots(output_root)
    _require_configured_overall_root(project, roots.overall)
    roots.prediction.mkdir(parents=True, exist_ok=True)

    identities = recompute_prediction_input_identities(project_root=project, data_root=data)
    run_manifest = _run_manifest_payload(identities, maximum_fold_workers=workers)
    _write_or_verify_canonical_json(
        roots.prediction / "run_manifest.json",
        run_manifest,
    )
    master_path = roots.prediction / "four_fold_prediction_seal.json"
    if master_path.exists():
        authorization = authorize_development_scoring(
            roots.prediction,
            expected_seal_sha256=_sha256_file(master_path),
            project_root=project,
            data_root=data,
        )
        return PredictionPhaseResult(
            overall_root=roots.overall,
            prediction_root=roots.prediction,
            master_seal_sha256=authorization.seal.sha256,
            fold_ids=DEVELOPMENT_FOLD_IDS,
        )
    if roots.score.exists() and any(roots.score.iterdir()):
        raise DevelopmentRuntimeError(
            "incomplete prediction_phase cannot resume beside a non-empty score_phase"
        )
    inputs = load_s1_runner_inputs(project_root=project, data_root=data)
    if inputs.location_grid.cell_count != EXPECTED_25KM_CELL_COUNT:
        raise DevelopmentRuntimeError("verified prediction grid cell count changed")

    selections_by_fold: dict[str, tuple[FoldHorizonParameterSelection, ...]] = {}
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="s1c0-fold") as executor:
        selection_futures = {
            executor.submit(_select_one_fold, inputs, fold_id): fold_id
            for fold_id in DEVELOPMENT_FOLD_IDS
        }
        for selection_future in as_completed(selection_futures):
            fold_id = selection_futures[selection_future]
            selections_by_fold[fold_id] = selection_future.result()
    ordered_selections = {fold_id: selections_by_fold[fold_id] for fold_id in DEVELOPMENT_FOLD_IDS}
    _write_or_verify_canonical_json(
        roots.prediction / "parameter_selection.json",
        _parameter_selection_payload(ordered_selections),
    )

    missing_folds: list[str] = []
    for fold_id in DEVELOPMENT_FOLD_IDS:
        state = _checkpoint_fold_state(
            roots.prediction,
            fold_id,
            project_root=project,
            data_root=data,
            cell_count=inputs.location_grid.cell_count,
        )
        if state == "complete":
            continue
        path = roots.prediction / "folds" / fold_id / PREDICTION_FILE_NAME
        if state == "payload_only":
            seal_fold_prediction(
                roots.prediction,
                fold_id,
                prediction_manifest=frozen_fold_prediction_manifest(fold_id),
                prediction_artifacts=(
                    PredictionArtifactInput(
                        path=path,
                        schema=frozen_fold_prediction_npz_schema(
                            fold_id, cell_count=inputs.location_grid.cell_count
                        ),
                    ),
                ),
                project_root=project,
                data_root=data,
            )
            continue
        missing_folds.append(fold_id)

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="s1c0-fold") as executor:
        prediction_futures = {
            executor.submit(
                build_development_fold_prediction,
                inputs,
                fold_id=fold_id,
                selections=ordered_selections[fold_id],
            ): fold_id
            for fold_id in missing_folds
        }
        for prediction_future in as_completed(prediction_futures):
            fold_id = prediction_futures[prediction_future]
            prediction = prediction_future.result()
            arrays = fold_prediction_npz_arrays(
                prediction, cell_count=inputs.location_grid.cell_count
            )
            path = roots.prediction / "folds" / fold_id / PREDICTION_FILE_NAME
            _write_npz_exclusive(
                path,
                fold_id=fold_id,
                arrays=arrays,
                cell_count=inputs.location_grid.cell_count,
            )
            seal_fold_prediction(
                roots.prediction,
                fold_id,
                prediction_manifest=frozen_fold_prediction_manifest(fold_id),
                prediction_artifacts=(
                    PredictionArtifactInput(
                        path=path,
                        schema=frozen_fold_prediction_npz_schema(
                            fold_id, cell_count=inputs.location_grid.cell_count
                        ),
                    ),
                ),
                project_root=project,
                data_root=data,
            )
    authorization = seal_four_fold_predictions(
        roots.prediction,
        project_root=project,
        data_root=data,
    )
    return PredictionPhaseResult(
        overall_root=roots.overall,
        prediction_root=roots.prediction,
        master_seal_sha256=authorization.seal.sha256,
        fold_ids=DEVELOPMENT_FOLD_IDS,
    )


def _epoch_us(value: datetime) -> int:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise DevelopmentRuntimeError("forecast issue must be timezone-aware")
    delta = value.astimezone(UTC) - _UTC_EPOCH
    return (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds


def _utc_from_epoch_us(value: int) -> datetime:
    return _UTC_EPOCH + timedelta(microseconds=value)


def _optional_float(
    arrays: Mapping[str, NumericArray],
    value_key: str,
    mask_key: str,
    row: int,
    column: int,
) -> float | None:
    mask = int(arrays[mask_key][row, column])
    if mask == 0:
        return None
    if mask != 1:
        raise DevelopmentRuntimeError(f"prediction applicability mask changed: {mask_key}")
    return float(arrays[value_key][row, column])


class NPZDevelopmentPredictionSource:
    """In-memory, validated lookup adapter for the frozen numeric NPZ schema."""

    def __init__(
        self,
        arrays_by_fold: Mapping[str, Mapping[str, object]],
        *,
        cell_count: int = EXPECTED_25KM_CELL_COUNT,
    ) -> None:
        folds = tuple(arrays_by_fold)
        if not folds or any(fold_id not in DEVELOPMENT_FOLD_IDS for fold_id in folds):
            raise DevelopmentRuntimeError("prediction source contains an unknown or empty fold set")
        materialized: dict[str, Mapping[str, NumericArray]] = {}
        primary_lookup: dict[tuple[str, int, int], int] = {}
        magnitude_lookup: dict[tuple[str, int], int] = {}
        for fold_id in folds:
            raw = arrays_by_fold[fold_id]
            validate_frozen_fold_prediction_npz_arrays(fold_id, raw, cell_count=cell_count)
            arrays: dict[str, NumericArray] = {}
            for name, value in raw.items():
                array = np.array(value, copy=True, order="C")
                if array.dtype.hasobject:
                    raise DevelopmentRuntimeError("object arrays are prohibited")
                array.setflags(write=False)
                arrays[name] = cast(NumericArray, array)
            materialized[fold_id] = MappingProxyType(arrays)
            issue_values = arrays["primary_issue_time_us"]
            horizon_values = arrays["primary_horizon_days"]
            for row, (issue_us, horizon) in enumerate(
                zip(issue_values, horizon_values, strict=True)
            ):
                primary_key = (fold_id, int(issue_us), int(horizon))
                if primary_key in primary_lookup:
                    raise DevelopmentRuntimeError("primary prediction identity is duplicated")
                primary_lookup[primary_key] = row
            for row, issue_us in enumerate(arrays["magnitude_issue_time_us"]):
                magnitude_key = (fold_id, int(issue_us))
                if magnitude_key in magnitude_lookup:
                    raise DevelopmentRuntimeError("magnitude snapshot identity is duplicated")
                magnitude_lookup[magnitude_key] = row
        self._cell_count = cell_count
        self._arrays = MappingProxyType(materialized)
        self._primary_lookup = MappingProxyType(primary_lookup)
        self._magnitude_lookup = MappingProxyType(magnitude_lookup)

    @property
    def fold_ids(self) -> tuple[str, ...]:
        return tuple(self._arrays)

    def weekly_issue_times_by_fold(self) -> Mapping[str, tuple[datetime, ...]]:
        result = {
            fold_id: tuple(
                _utc_from_epoch_us(int(value)) for value in arrays["magnitude_issue_time_us"]
            )
            for fold_id, arrays in self._arrays.items()
        }
        return MappingProxyType(result)

    def _primary(
        self, fold_id: str, issue_time_utc: datetime, horizon_days: int
    ) -> tuple[Mapping[str, NumericArray], int]:
        try:
            arrays = self._arrays[fold_id]
            row = self._primary_lookup[(fold_id, _epoch_us(issue_time_utc), horizon_days)]
        except KeyError as exc:
            raise DevelopmentRuntimeError(
                "requested primary prediction snapshot does not exist"
            ) from exc
        return arrays, row

    def location_forecast(
        self,
        *,
        fold_id: str,
        issue_time_utc: datetime,
        horizon_days: int,
        model_id: str,
    ) -> LocationForecast:
        from seismoflux.multitask_s1.development_score import LocationForecast

        arrays, row = self._primary(fold_id, issue_time_utc, horizon_days)
        try:
            model_index = LOCATION_MODEL_IDS.index(model_id)
        except ValueError as exc:
            raise DevelopmentRuntimeError("requested location model is not frozen") from exc
        return LocationForecast(
            fold_id=fold_id,
            issue_time_utc=issue_time_utc.astimezone(UTC),
            horizon_days=horizon_days,
            model_id=model_id,
            cell_relative_mass=arrays["location_relative_mass"][row, model_index, :],
        )

    def count_forecast(
        self,
        *,
        fold_id: str,
        issue_time_utc: datetime,
        horizon_days: int,
        model_id: str,
        magnitude_band: str,
    ) -> CountForecast:
        from seismoflux.multitask_s1.development_score import CountForecast

        arrays, row = self._primary(fold_id, issue_time_utc, horizon_days)
        try:
            internal_band = _TIME_BAND_FROM_SCORE[magnitude_band]
            band_index = TIME_BANDS.index(cast(Any, internal_band))
        except (KeyError, ValueError) as exc:
            raise DevelopmentRuntimeError("requested time magnitude band is not frozen") from exc
        expected = float(arrays["t0_expected_count"][row, band_index])
        if model_id == "T0_POISSON_EXPANDING":
            return CountForecast(
                fold_id=fold_id,
                issue_time_utc=issue_time_utc.astimezone(UTC),
                horizon_days=horizon_days,
                model_id=model_id,
                magnitude_band=cast(Any, magnitude_band),
                expected_count=expected,
                distribution="poisson",
            )
        if model_id != "T1_NEGATIVE_BINOMIAL":
            raise DevelopmentRuntimeError("requested count model is not frozen")
        status_code = int(arrays["t1_status_code"][row, band_index])
        reason_code = int(arrays["t1_reason_code"][row, band_index])
        try:
            status = NB2_STATUS_CODES[status_code]
            reason = NB2_REASON_CODES[reason_code]
        except IndexError as exc:
            raise DevelopmentRuntimeError("T1 qualification code is outside its codebook") from exc
        qualification = NB2DispersionQualification(
            status=cast(Any, status),
            reason=reason,
            historical_block_count=int(arrays["t1_historical_block_count"][row, band_index]),
            sample_mean_count=float(arrays["t1_sample_mean_count"][row, band_index]),
            sample_variance_count=_optional_float(
                arrays,
                "t1_sample_variance_count",
                "t1_sample_variance_applicable",
                row,
                band_index,
            ),
            dispersion_k=_optional_float(
                arrays,
                "t1_dispersion_k",
                "t1_dispersion_k_applicable",
                row,
                band_index,
            ),
            observed_information_k=_optional_float(
                arrays,
                "t1_observed_information_k",
                "t1_observed_information_k_applicable",
                row,
                band_index,
            ),
            standard_error_k=_optional_float(
                arrays,
                "t1_standard_error_k",
                "t1_standard_error_k_applicable",
                row,
                band_index,
            ),
        )
        return CountForecast(
            fold_id=fold_id,
            issue_time_utc=issue_time_utc.astimezone(UTC),
            horizon_days=horizon_days,
            model_id=model_id,
            magnitude_band=cast(Any, magnitude_band),
            expected_count=expected,
            distribution="nb2",
            nb2_qualification=qualification,
        )

    def magnitude_forecast(
        self,
        *,
        fold_id: str,
        issue_time_utc: datetime,
        model_id: str,
    ) -> MagnitudeForecast:
        from seismoflux.multitask_s1.development_score import MagnitudeForecast

        try:
            arrays = self._arrays[fold_id]
            row = self._magnitude_lookup[(fold_id, _epoch_us(issue_time_utc))]
        except KeyError as exc:
            raise DevelopmentRuntimeError("requested magnitude snapshot does not exist") from exc
        if model_id == "M0_GR_GLOBAL":
            prefix = "m0"
            mc = MAIN_GR_MC
        elif model_id == "M3_GR_LONG_M5":
            prefix = "m3"
            mc = LONG_M5_GR_MC
        else:
            raise DevelopmentRuntimeError("requested magnitude model is not frozen")
        masses = tuple(float(value) for value in arrays[f"{prefix}_bin_probability_mass"][row])
        b_value = float(arrays[f"{prefix}_b_value"][row])
        lower = tuple(mc + index * GR_BIN_WIDTH for index in range(len(masses)))
        upper_values = [mc + (index + 1) * GR_BIN_WIDTH for index in range(len(masses))]
        upper_values[-1] = GR_MAXIMUM_MAGNITUDE
        m5_6_mass = math.fsum(
            mass for mass, edge in zip(masses, lower, strict=True) if 5.0 <= edge < 6.0
        )
        m6_plus_mass = math.fsum(
            mass for mass, edge in zip(masses, lower, strict=True) if edge >= 6.0
        )
        model = TruncatedGRMagnitudeModel(
            model_id=cast(Any, model_id),
            training_event_count=int(arrays[f"{prefix}_training_event_count"][row]),
            mc=mc,
            maximum_magnitude=GR_MAXIMUM_MAGNITUDE,
            bin_width=GR_BIN_WIDTH,
            b_value=b_value,
            beta=b_value * math.log(10.0),
            bin_lower_edges=lower,
            bin_upper_edges=tuple(upper_values),
            bin_probability_masses=masses,
            m5_6_probability_mass=m5_6_mass,
            m6_plus_probability_mass=m6_plus_mass,
        )
        return MagnitudeForecast(
            fold_id=fold_id,
            issue_time_utc=issue_time_utc.astimezone(UTC),
            model_id=model_id,
            model=model,
        )


def _load_all_prediction_arrays(
    prediction_root: Path, *, cell_count: int
) -> Mapping[str, Mapping[str, NumericArray]]:
    return MappingProxyType(
        {
            fold_id: _load_one_npz(
                prediction_root / "folds" / fold_id / PREDICTION_FILE_NAME,
                fold_id=fold_id,
                cell_count=cell_count,
            )
            for fold_id in DEVELOPMENT_FOLD_IDS
        }
    )


def _authorization_payload(
    authorization: DevelopmentScoringAuthorization,
    *,
    expected_seal_sha256: str,
) -> dict[str, JsonValue]:
    return {
        "schema_version": 1,
        "record_type": "s1_c0_development_score_authorization",
        "run_contract_id": RUN_CONTRACT_ID,
        "role": "development_scoring_after_complete_prediction_seal",
        "prediction_seal_relative_path": "../prediction_phase/four_fold_prediction_seal.json",
        "prediction_seal_sha256": expected_seal_sha256,
        "input_identities": cast(dict[str, JsonValue], authorization.input_identities.as_mapping()),
        "ordered_fold_sha256": [
            {"fold_id": fold_id, "sha256": digest}
            for fold_id, digest in authorization.ordered_fold_sha256
        ],
        "holdout_opened": False,
        "audit_opened": False,
        "locked_test_run": False,
    }


def _jsonable(value: object) -> JsonValue:
    if value is None or isinstance(value, bool | int | str):
        return cast(JsonScalar, value)
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if math.isnan(value):
            return {"nonfinite_float": "NaN"}
        return {"nonfinite_float": "Infinity" if value > 0 else "-Infinity"}
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise DevelopmentRuntimeError("raw score contains a naive datetime")
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        result: dict[str, JsonValue] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise DevelopmentRuntimeError("raw score mappings require string keys")
            result[key] = _jsonable(child)
        return result
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_jsonable(child) for child in value]
    raise DevelopmentRuntimeError(f"raw score contains an unsupported value: {type(value)!r}")


def _raw_score_records(scores: DevelopmentRawScores) -> Iterable[dict[str, object]]:
    families: tuple[tuple[str, Sequence[object]], ...] = (
        ("location", scores.location),
        ("time", scores.time),
        ("magnitude", scores.magnitude),
        ("joint", scores.joint),
    )
    for family, rows in families:
        for row in rows:
            raw = cast(Any, row)
            payload = _jsonable(row)
            if not isinstance(payload, dict):
                raise AssertionError("raw score dataclass must serialize to an object")
            issue = getattr(row, "issue_time_utc", None)
            if issue is None:
                issue = getattr(row, "forecast_issue_time_utc", None)
            model_id = getattr(row, "model_id", None)
            if model_id is None:
                model_id = getattr(row, "joint_model_id", None)
            yield {
                "score_family": family,
                "fold_id": raw.fold_id,
                "issue_time_utc": issue,
                "horizon_days": getattr(row, "horizon_days", None),
                "model_id": model_id,
                "status": raw.status,
                "payload_json": canonical_json_bytes(payload).decode("utf-8"),
            }


def _write_raw_scores_parquet(path: Path, scores: DevelopmentRawScores) -> Mapping[str, int]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    counts = {
        "location": len(scores.location),
        "time": len(scores.time),
        "magnitude": len(scores.magnitude),
        "joint": len(scores.joint),
    }
    if sum(counts.values()) == 0:
        raise DevelopmentRuntimeError("authorized scoring returned no raw rows")
    schema = pa.schema(
        [
            ("score_family", pa.string()),
            ("fold_id", pa.string()),
            ("issue_time_utc", pa.timestamp("us", tz="UTC")),
            ("horizon_days", pa.int16()),
            ("model_id", pa.string()),
            ("status", pa.string()),
            ("payload_json", pa.large_string()),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        stream = path.open("xb")
    except FileExistsError as exc:
        raise RuntimeArtifactExistsError(f"raw scores cannot be overwritten: {path}") from exc
    try:
        with stream:
            writer = pq.ParquetWriter(stream, schema=schema, compression="zstd")
            try:
                batch: list[dict[str, object]] = []
                for record in _raw_score_records(scores):
                    batch.append(record)
                    if len(batch) == 1024:
                        writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                        batch.clear()
                if batch:
                    writer.write_table(pa.Table.from_pylist(batch, schema=schema))
            finally:
                writer.close()
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        raise
    return MappingProxyType(counts)


def run_score_phase(
    *,
    project_root: str | Path,
    data_root: str | Path,
    output_root: str | Path,
    expected_seal_sha256: str,
) -> ScorePhaseResult:
    """Authorize, reload, re-authorize, and only then construct development truth."""

    configure_single_thread_numerics()
    if _SHA256.fullmatch(expected_seal_sha256) is None:
        raise DevelopmentRuntimeError("score phase requires an explicit lowercase seal SHA-256")
    project = _explicit_existing_root(project_root, label="project_root")
    data = _explicit_existing_root(data_root, label="data_root")
    roots = phase_roots(output_root)
    _require_configured_overall_root(project, roots.overall)

    # Importing the target/scoring layer is forbidden in prediction.  This sole
    # official score entry point owns authorization, sealed-NPZ loading,
    # authenticated inputs, target construction, and scoring; callers cannot
    # inject any of those scientific objects.
    from seismoflux.multitask_s1.development_score import (
        ScoringContext,
        score_authorized_development_from_context,
    )
    from seismoflux.multitask_s1.development_summary import summarize_development_scores

    context = ScoringContext(
        output_root=roots.prediction,
        expected_seal_sha256=expected_seal_sha256,
        project_root=project,
        data_root=data,
    )
    authorization = authorize_development_scoring(
        roots.prediction,
        expected_seal_sha256=expected_seal_sha256,
        project_root=project,
        data_root=data,
    )
    roots.score.mkdir(parents=True, exist_ok=True)
    authorization_path = roots.score / "score_authorization.json"
    _write_or_verify_canonical_json(
        authorization_path,
        _authorization_payload(authorization, expected_seal_sha256=expected_seal_sha256),
    )
    authorized = score_authorized_development_from_context(context)
    if authorized.authorization != authorization:
        raise DevelopmentRuntimeError(
            "score authorization changed after its persisted phase checkpoint"
        )
    scores = authorized.scores
    # Build and validate the preregistered scientific interpretation in memory
    # before making the raw-score file immutable.  This preserves the required
    # on-disk order (authorization -> raw -> summary) without leaving a valid
    # raw checkpoint stranded behind an unserializable summary.
    summary = summarize_development_scores(scores)
    raw_path = roots.score / RAW_SCORE_FILE_NAME
    row_counts = _write_raw_scores_parquet(raw_path, scores)
    summary_path = roots.score / "development_summary.json"
    _write_exclusive_json(summary_path, summary)
    return ScorePhaseResult(
        overall_root=roots.overall,
        score_root=roots.score,
        authorization_path=authorization_path,
        raw_scores_path=raw_path,
        summary_path=summary_path,
        raw_score_row_counts=row_counts,
    )


def run_phase(
    *,
    phase: Phase,
    project_root: str | Path,
    data_root: str | Path,
    output_root: str | Path,
    expected_seal_sha256: str | None = None,
    maximum_fold_workers: int = 3,
) -> PredictionPhaseResult | ScorePhaseResult:
    """Dispatch one explicit phase without guessing or auto-advancing the gate."""

    if phase == "predict":
        if expected_seal_sha256 is not None:
            raise DevelopmentRuntimeError("predict phase does not accept a score authorization")
        return run_prediction_phase(
            project_root=project_root,
            data_root=data_root,
            output_root=output_root,
            maximum_fold_workers=maximum_fold_workers,
        )
    if phase == "score":
        if expected_seal_sha256 is None:
            raise DevelopmentRuntimeError("score phase requires --expected-seal-sha256")
        return run_score_phase(
            project_root=project_root,
            data_root=data_root,
            output_root=output_root,
            expected_seal_sha256=expected_seal_sha256,
        )
    raise DevelopmentRuntimeError("phase must be exactly 'predict' or 'score'")


__all__ = [
    "DevelopmentRuntimeError",
    "NPZDevelopmentPredictionSource",
    "PhaseRoots",
    "PredictionPhaseResult",
    "RuntimeArtifactExistsError",
    "ScorePhaseResult",
    "configure_single_thread_numerics",
    "phase_roots",
    "run_phase",
    "run_prediction_phase",
    "run_score_phase",
]
