"""Finite, location-only C2B prediction with earlier training and small checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import yaml
from pyproj import Transformer
from scipy.special import logsumexp

from seismoflux.background.grid import EQUAL_AREA_CRS
from seismoflux.multitask_s1.c2b_inputs import C2BCatalog, load_c2b_catalog, panel_indices
from seismoflux.multitask_s1.c2b_models import (
    C2BFitError,
    C2BRidgeFit,
    C2BTrainingIssue,
    fit_spatial_ridge,
    gaussian_log_masses,
    mix_log_masses,
)
from seismoflux.multitask_s1.development_contract import (
    DEVELOPMENT_FOLD_IDS,
    load_development_contract,
)
from seismoflux.multitask_s1.runner_inputs import (
    S1RunnerInputs,
    build_inner_development_exposures,
    load_outer_development_issues,
    load_verified_spatial_inputs,
)

PROTOCOL_PATH = Path("configs/multitask_s1_c2b_catalog_models.yaml")
PROTOCOL_SHA256 = "4f497690643a64466cbd9eec358977f1c8c1d4655bc4cc9ca5a65dbc95859243"
MODEL_IDS = (
    "C2B_D0_K75",
    "C2B_D1_K75",
    "C2B_D2_K75",
    "C2B_D0_R30",
    "C2B_D1_R30",
    "C2B_D0_MULTISCALE",
    "C2B_D0_AGE_WEIGHTED",
    "C2B_D0_RIDGE_CORE",
    "C2B_D0_RIDGE_M5",
)
COMPONENT_IDS = ("K25", "K75", "K150", "D1K75", "D2K75", "R30", "D1R30", "E7", "E30", "E90")
HORIZONS = (7, 30, 90, 180, 365)
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_DAY_US = 86_400_000_000
_NUMERICAL_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Install a new complete record without overwriting any completed record."""
    if path.exists():
        raise FileExistsError(f"preserve completed record: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial.json")
    with partial.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(partial, path)


def _atomic_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"preserve completed prediction: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial.npz")
    with partial.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(partial, path)


def _epoch_us(value: datetime) -> int:
    if value.tzinfo is None:
        raise ValueError("issue must be timezone aware")
    delta = value.astimezone(UTC) - _EPOCH
    return (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(UTC)


def _emit(stage: str, **values: Any) -> None:
    print(
        json.dumps({"stage": stage, "time_utc": datetime.now(UTC).isoformat(), **values}),
        flush=True,
    )


def load_protocol(project: Path) -> dict[str, Any]:
    path = project / PROTOCOL_PATH
    if _sha(path) != PROTOCOL_SHA256:
        raise ValueError("C2B protocol changed; preserve existing run and review explicitly")
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    model_ids = (
        tuple(protocol["fixed_models"])
        + tuple(protocol["selected_models"])
        + tuple(protocol["ridge"]["models"])
    )
    if model_ids != MODEL_IDS or tuple(protocol["calendar"]["outer_folds"]) != DEVELOPMENT_FOLD_IDS:
        raise ValueError("finite C2B model or fold axis changed")
    if (
        _sha(project / protocol["inputs"]["panel_ledger"])
        != protocol["inputs"]["panel_ledger_sha256"]
    ):
        raise ValueError("C2B training panel ledger changed")
    return protocol


def _identity(protocol: dict[str, Any]) -> dict[str, Any]:
    directory = Path(__file__).parent
    return {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": PROTOCOL_SHA256,
        "grid_id": protocol["inputs"]["grid_id"],
        "catalog_sha256": protocol["inputs"]["canonical_catalog_sha256"],
        "source_records_sha256": protocol["inputs"]["source_records_sha256"],
        "model_ids": list(MODEL_IDS),
        "implementation_hashes": {
            name: _sha(directory / name)
            for name in ("c2b_predict.py", "c2b_inputs.py", "c2b_models.py")
        },
    }


def _record(root: Path, path: Path, **extra: Any) -> dict[str, Any]:
    return {"path": path.relative_to(root).as_posix(), "sha256": _sha(path), **extra}


def _checked(root: Path, record: Mapping[str, Any]) -> Path:
    path = (root / record["path"]).resolve()
    if (
        not path.is_relative_to(root.resolve())
        or not path.is_file()
        or _sha(path) != record["sha256"]
    ):
        raise ValueError("C2B completed artifact identity changed")
    return path


@contextmanager
def _run_lock(root: Path) -> Iterator[None]:
    """One OS-held lock; a crashed process releases it without deleting checkpoints."""
    path = root / "run.lock"
    with path.open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def load_inputs(
    project: Path, data_root: Path, protocol: dict[str, Any]
) -> tuple[S1RunnerInputs, C2BCatalog]:
    """Use existing calendars and geography; never materialize 2020+ catalog rows."""
    contract, summary = load_development_contract(
        project / protocol["parent_contract"], project_root=project
    )
    catalog = load_c2b_catalog(data_root=data_root, protocol=protocol)
    domain, grid, study_hash = load_verified_spatial_inputs(data_root)
    issue_path = project / contract["source_identities"]["issue_maturity_ledger"]["path"]
    return S1RunnerInputs(
        project_root=project,
        data_root=data_root,
        contract=contract,
        contract_summary=summary,
        catalog=catalog.table,
        catalog_identity={
            "file_sha256": protocol["inputs"]["canonical_catalog_sha256"],
            "row_count": 40898,
        },
        study_area_sha256=study_hash,
        spatial_domain=domain,
        location_grid=grid,
        outer_issues=load_outer_development_issues(issue_path, contract),
        inner_exposures=build_inner_development_exposures(contract),
    ), catalog


class ComponentCache:
    """Causal kernels shared across horizons; no labels are accepted by this API."""

    def __init__(
        self,
        inputs: S1RunnerInputs,
        catalog: C2BCatalog,
        protocol: dict[str, Any],
        root: Path,
        identity: dict[str, Any],
    ) -> None:
        self.inputs, self.catalog, self.protocol = inputs, catalog, protocol
        self.root = root / "component_cache"
        self.root.mkdir(parents=True, exist_ok=True)
        self.identity = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        transformer = Transformer.from_crs("EPSG:4326", EQUAL_AREA_CRS, always_xy=True)
        x, y = transformer.transform(catalog.table.longitude, catalog.table.latitude)
        self.xy = np.column_stack((x, y)) / 1000.0
        grid = inputs.location_grid
        self.query = np.column_stack((grid.x_km, grid.y_km))
        self.area = grid.area_km2
        self._locks: dict[int, Any] = {}
        self._guard = threading.Lock()

    def _kernel(
        self,
        selected: np.ndarray,
        bandwidths: tuple[float, ...] = (75.0,),
        weights: np.ndarray | None = None,
    ) -> dict[float, np.ndarray]:
        return gaussian_log_masses(
            self.xy[selected],
            self.query,
            self.area,
            bandwidths_km=bandwidths,
            log_event_weights=weights,
        )

    def get(self, issue: datetime) -> dict[str, np.ndarray]:
        issue_us = _epoch_us(issue)
        with self._guard:
            lock = self._locks.setdefault(issue_us, threading.Lock())
        with lock:
            path = self.root / f"issue_{issue_us}.npz"
            if path.is_file():
                with np.load(path, allow_pickle=False) as saved:
                    if (
                        str(saved["identity"].item()) != self.identity
                        or int(saved["issue_time_us"].item()) != issue_us
                    ):
                        raise ValueError("C2B component cache differs from the same causal run")
                    logs = saved["log_masses"].copy()
                    counts = saved["history_counts"].copy()
                _validate_log_mass(logs, (len(COMPONENT_IDS), self.area.size))
                result = dict(zip(COMPONENT_IDS, logs, strict=True))
                result["history_counts"] = counts
                return result
            panels = self.protocol["panels"]
            panel_specs = list(panels.values())
            long_indices = [panel_indices(self.catalog, spec, issue) for spec in panel_specs]
            base = self._kernel(long_indices[0], (25.0, 75.0, 150.0))
            result = {"K25": base[25.0], "K75": base[75.0], "K150": base[150.0]}
            result["D1K75"] = self._kernel(long_indices[1])[75.0]
            result["D2K75"] = self._kernel(long_indices[2])[75.0]
            counts = [len(indices) for indices in long_indices]
            for index, name, long_name in ((0, "R30", "K75"), (1, "D1R30", "D1K75")):
                recent = panel_indices(self.catalog, panel_specs[index], issue, recent_days=30)
                result[name] = self._kernel(recent)[75.0] if recent.size else result[long_name]
                counts.append(len(recent))
            recent = panel_indices(self.catalog, panel_specs[0], issue, recent_days=365)
            ages = (issue_us - self.catalog.table.origin_time_us[recent]) / _DAY_US
            for half_life in (7, 30, 90):
                result[f"E{half_life}"] = (
                    self._kernel(recent, weights=-np.log(2.0) * ages / half_life)[75.0]
                    if recent.size
                    else result["K75"]
                )
            counts.append(len(recent))
            logs = np.stack([result[name] for name in COMPONENT_IDS])
            _validate_log_mass(logs, (len(COMPONENT_IDS), self.area.size))
            _atomic_npz(
                path,
                {
                    "identity": np.asarray(self.identity),
                    "issue_time_us": np.int64(issue_us),
                    "log_masses": logs,
                    "history_counts": np.asarray(counts, dtype=np.int64),
                },
            )
            result["history_counts"] = np.asarray(counts, dtype=np.int64)
            _emit(
                "causal_components_saved", issue_time_utc=issue.isoformat(), training_counts=counts
            )
            return result


def _validate_log_mass(logs: np.ndarray, shape: tuple[int, ...]) -> None:
    if logs.shape != shape or logs.dtype != np.dtype("float64") or not np.isfinite(logs).all():
        raise ValueError("C2B log mass shape, dtype, or finiteness mismatch")
    if not np.allclose(logsumexp(logs, axis=-1), 0.0, atol=1e-10, rtol=0.0):
        raise ValueError("C2B full-domain log masses are not normalized")


def _mix_recent(base: np.ndarray, recent: np.ndarray, alpha: float) -> np.ndarray:
    if alpha == 0.0 or np.array_equal(base, recent):
        return base
    return mix_log_masses((base, recent), (1.0 - alpha, alpha))


def _features(components: Mapping[str, np.ndarray], with_m5: bool) -> np.ndarray:
    names = ("K25", "K150", "E30", "D2K75") if with_m5 else ("K25", "K150", "E30")
    return np.column_stack([components[name] - components["K75"] for name in names])


@dataclass(frozen=True)
class InnerSample:
    fold_id: str
    block_id: str
    horizon: int
    issue: datetime
    end: datetime
    components: Mapping[str, np.ndarray]
    target_cells: np.ndarray
    target_available_us: np.ndarray

    def counts(self, availability_cutoff: datetime) -> np.ndarray:
        selected = self.target_cells[self.target_available_us <= _epoch_us(availability_cutoff)]
        return np.bincount(selected, minlength=self.components["K75"].size).astype(np.float64)

    def training_issue(self, availability_cutoff: datetime, with_m5: bool) -> C2BTrainingIssue:
        return C2BTrainingIssue(
            f"{self.fold_id}:{self.block_id}:{self.horizon}:{self.issue.isoformat()}",
            self.components["K75"],
            _features(self.components, with_m5),
            self.counts(availability_cutoff),
        )


def _inner_samples(
    inputs: S1RunnerInputs, fold: Mapping[str, Any], horizon: int, cache: ComponentCache
) -> list[InnerSample]:
    outer_cutoff = _utc(fold["outer_start"]) - timedelta(days=30)
    blocks = {block["id"]: block for block in fold["inner_blocks"]}
    catalog = inputs.catalog
    result = []
    for exposure in inputs.inner_exposures:
        if exposure.fold_id != fold["id"] or exposure.horizon_days != horizon:
            continue
        for issue in exposure.issue_times_utc:
            end = issue + timedelta(days=horizon)
            if end > outer_cutoff or end > _utc(blocks[exposure.block_id]["end"]):
                raise ValueError("inner labels must finish before the declared embargo boundary")
            components = cache.get(issue)
            indices = np.flatnonzero(
                catalog.inside_study_area
                & (catalog.magnitude >= 4.0)
                & (catalog.origin_time_us > _epoch_us(issue))
                & (catalog.origin_time_us <= _epoch_us(end))
                & (catalog.available_at_us <= _epoch_us(outer_cutoff))
            )
            cells = []
            for index in indices:
                cell = inputs.spatial_domain.locator.locate_lonlat(
                    catalog.longitude[index], catalog.latitude[index]
                )
                if cell is None:
                    raise ValueError(
                        "an inner canonical in-domain event is outside the unchanged grid"
                    )
                cells.append(cell)
            result.append(
                InnerSample(
                    fold["id"],
                    exposure.block_id,
                    horizon,
                    issue,
                    end,
                    components,
                    np.asarray(cells, dtype=np.int64),
                    catalog.available_at_us[indices],
                )
            )
    return result


def _candidate_logmass(
    components: Mapping[str, np.ndarray], candidate: tuple[str, Any], protocol: dict[str, Any]
) -> np.ndarray:
    family, value = candidate
    if family == "multiscale":
        if value == "K75":
            return components["K75"]
        weights = protocol["selected_models"]["C2B_D0_MULTISCALE"]["candidates"][value]
        return mix_log_masses([components[name] for name in ("K25", "K75", "K150")], weights)
    alpha, half_life = value
    return _mix_recent(components["K75"], components[f"E{half_life}"], alpha)


def _block_score(
    samples: Sequence[InnerSample], cutoff: datetime, area: np.ndarray, predictor: Any
) -> float | None:
    total, count = 0.0, 0.0
    log_area = np.log(area)
    for sample in samples:
        counts = sample.counts(cutoff)
        n = float(counts.sum())
        if n:
            total += float(np.dot(counts, predictor(sample) - log_area))
            count += n
    return total / count if count else None


def select_kernel_parameters(
    samples: Sequence[InnerSample], cutoff: datetime, area: np.ndarray, protocol: dict[str, Any]
) -> dict[str, Any]:
    """Select only from earlier block means; targets never become features."""
    by_block = {
        block: [sample for sample in samples if sample.block_id == block]
        for block in ("I1", "I2", "I3")
    }
    choices: dict[str, list[Any]] = {
        "multiscale": list(protocol["selected_models"]["C2B_D0_MULTISCALE"]["tie_order"]),
        "age": [(0.0, 90)]
        + [(alpha, half_life) for alpha in (0.25, 0.5, 0.75) for half_life in (90, 30, 7)],
    }
    selected: dict[str, Any] = {}
    for family, values in choices.items():
        evidence = []
        for value in values:
            candidate = (family, value)
            scores = [
                _block_score(
                    block,
                    cutoff,
                    area,
                    lambda sample, choice=candidate: _candidate_logmass(
                        sample.components, choice, protocol
                    ),
                )
                for block in by_block.values()
            ]
            valid = [score for score in scores if score is not None]
            evidence.append(
                {
                    "candidate": value,
                    "block_scores": scores,
                    "mean": float(np.mean(valid)) if len(valid) >= 2 else None,
                }
            )
        evaluable = [row for row in evidence if row["mean"] is not None]
        if evaluable:
            best = max(row["mean"] for row in evaluable)
            winner = next(row["candidate"] for row in evaluable if best - row["mean"] <= 1e-12)
            status = "selected_from_earlier_blocks"
        else:
            winner, status = values[0], "insufficient_nonempty_blocks_fixed_K75"
        selected[family] = {"selected": winner, "status": status, "candidates": evidence}
    return selected


def legal_ridge_training(
    samples: Sequence[InnerSample], block_ids: Sequence[str], boundary: datetime
) -> list[InnerSample]:
    cutoff = boundary - timedelta(days=30)
    return [sample for sample in samples if sample.block_id in block_ids and sample.end <= cutoff]


def _fit_ridge(
    samples: Sequence[InnerSample],
    cutoff: datetime,
    area: np.ndarray,
    with_m5: bool,
    ridge_lambda: float,
) -> tuple[C2BRidgeFit | None, dict[str, Any]]:
    if not samples:
        return None, {"status": "no_training_issues_beta_zero", "event_count": 0}
    try:
        fit = fit_spatial_ridge(
            [sample.training_issue(cutoff, with_m5) for sample in samples],
            area,
            ridge_lambda=ridge_lambda,
        )
    except C2BFitError as error:
        return None, {"status": "not_evaluable_beta_zero", "reason": str(error)}
    return fit, fit.to_dict()


def select_and_fit_ridge(
    samples: Sequence[InnerSample], fold: Mapping[str, Any], area: np.ndarray, with_m5: bool
) -> tuple[C2BRidgeFit | None, dict[str, Any]]:
    blocks = {block["id"]: block for block in fold["inner_blocks"]}
    outer_start = _utc(fold["outer_start"])
    outer_cutoff = outer_start - timedelta(days=30)
    evidence = []
    for ridge_lambda in (10.0, 1.0, 0.1):
        scores, branch_records = [], []
        for train_blocks, validate_block in ((["I1"], "I2"), (["I1", "I2"], "I3")):
            validation_start = _utc(blocks[validate_block]["start"])
            cutoff = validation_start - timedelta(days=30)
            train = legal_ridge_training(samples, train_blocks, validation_start)
            fit, fit_record = _fit_ridge(train, cutoff, area, with_m5, ridge_lambda)
            validation = [sample for sample in samples if sample.block_id == validate_block]

            def predict(sample: InnerSample, fitted: C2BRidgeFit | None = fit) -> np.ndarray:
                return (
                    sample.components["K75"]
                    if fitted is None
                    else fitted.predict_log_mass(
                        sample.components["K75"], _features(sample.components, with_m5)
                    )
                )

            score = _block_score(validation, outer_cutoff, area, predict)
            if fit_record.get("status") == "not_evaluable_beta_zero":
                score = None
            scores.append(score)
            branch_records.append(
                {
                    "train_blocks": train_blocks,
                    "validate_block": validate_block,
                    "training_label_cutoff_utc": cutoff.isoformat(),
                    "training_issue_count": len(train),
                    "fit": fit_record,
                    "validation_log_density": score,
                }
            )
        mean = float(np.mean(scores)) if all(value is not None for value in scores) else None
        evidence.append({"lambda": ridge_lambda, "mean": mean, "branches": branch_records})
    valid = [row for row in evidence if row["mean"] is not None]
    if valid:
        best = max(row["mean"] for row in valid)
        selected = next(row["lambda"] for row in valid if best - row["mean"] <= 1e-12)
        status = "selected_from_two_earlier_validation_blocks"
    else:
        selected, status = 10.0, "insufficient_validation_fixed_lambda_10"
    final_train = legal_ridge_training(samples, ("I1", "I2", "I3"), outer_start)
    fit, final_record = _fit_ridge(final_train, outer_cutoff, area, with_m5, selected)
    return fit, {
        "selected_lambda": selected,
        "selection_status": status,
        "validation": evidence,
        "final_training_cutoff_utc": outer_cutoff.isoformat(),
        "final_fit": final_record,
    }


def _prediction_models(
    components: Mapping[str, np.ndarray],
    kernel_selection: dict[str, Any],
    fits: Sequence[C2BRidgeFit | None],
    protocol: dict[str, Any],
) -> np.ndarray:
    surfaces = [
        components["K75"],
        components["D1K75"],
        components["D2K75"],
        _mix_recent(components["K75"], components["R30"], 0.25),
        _mix_recent(components["D1K75"], components["D1R30"], 0.25),
        _candidate_logmass(
            components, ("multiscale", kernel_selection["multiscale"]["selected"]), protocol
        ),
        _candidate_logmass(components, ("age", kernel_selection["age"]["selected"]), protocol),
    ]
    for with_m5, fit in zip((False, True), fits, strict=True):
        surfaces.append(
            components["K75"]
            if fit is None
            else fit.predict_log_mass(components["K75"], _features(components, with_m5))
        )
    return np.stack(surfaces)


def _expected_horizon_axis(project: Path, fold_id: str, horizon: int) -> list[int]:
    contract = yaml.safe_load(
        (project / "configs/multitask_s1_development_contract.yaml").read_text(encoding="utf-8")
    )
    fold = next(fold for fold in contract["outer_folds"] if fold["id"] == fold_id)
    start, end = (
        datetime.fromisoformat(fold["outer_start"]),
        datetime.fromisoformat(fold["outer_end"]),
    )
    issue = start + timedelta(days=(3 - start.weekday()) % 7)
    selected = []
    while issue + timedelta(days=horizon) <= end:
        if not selected or _epoch_us(issue) >= selected[-1] + (horizon + 30) * _DAY_US:
            selected.append(_epoch_us(issue))
        issue += timedelta(days=7)
    return selected


def _verify_horizon(
    project: Path, root: Path, record: Mapping[str, Any], identity: dict[str, Any], fold_id: str
) -> dict[str, np.ndarray]:
    metadata = _read_json(_checked(root, record))
    if metadata["identity"] != identity or metadata["fold_id"] != fold_id:
        raise ValueError("C2B horizon identity differs")
    horizon = int(metadata["horizon_days"])
    if horizon not in HORIZONS or horizon != int(record["horizon_days"]):
        raise ValueError("C2B horizon is outside the finite protocol")
    with np.load(_checked(root, metadata["predictions"]), allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    expected = _expected_horizon_axis(project, fold_id, horizon)
    if set(arrays) != {"fold_id", "issue_times_us", "horizons_days", "model_ids", "log_cell_mass"}:
        raise ValueError("unexpected C2B prediction arrays")
    if (
        str(arrays["fold_id"].item()) != fold_id
        or arrays["issue_times_us"].tolist() != expected
        or arrays["issue_times_us"].dtype != np.dtype("int64")
        or arrays["horizons_days"].tolist() != [horizon] * len(expected)
        or arrays["model_ids"].tolist() != list(MODEL_IDS)
    ):
        raise ValueError("C2B saved issue, horizon, or model axis changed")
    cell_count = load_protocol(project)["inputs"]["grid_cells"]
    _validate_log_mass(arrays["log_cell_mass"], (len(expected), len(MODEL_IDS), cell_count))
    return arrays


def load_fold_arrays(output_root: Path, fold_record: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Read already verified horizon artifacts; caller verifies the four-fold manifest first."""
    metadata = _read_json(_checked(output_root, fold_record))
    horizons = []
    for record in metadata["horizons"]:
        horizon_record = _read_json(_checked(output_root, record))
        with np.load(
            _checked(output_root, horizon_record["predictions"]), allow_pickle=False
        ) as saved:
            horizons.append({key: saved[key].copy() for key in saved.files})
    return {
        "fold_id": horizons[0]["fold_id"],
        "model_ids": horizons[0]["model_ids"],
        **{
            name: np.concatenate([arrays[name] for arrays in horizons], axis=0)
            for name in ("issue_times_us", "horizons_days", "log_cell_mass")
        },
    }


def _verify_fold(
    project: Path, root: Path, record: Mapping[str, Any], identity: dict[str, Any]
) -> None:
    metadata = _read_json(_checked(root, record))
    if metadata["identity"] != identity or metadata["fold_id"] != record["fold_id"]:
        raise ValueError("C2B fold identity changed")
    if tuple(item["horizon_days"] for item in metadata["horizons"]) != HORIZONS:
        raise ValueError("C2B fold does not contain exactly the five horizons")
    for horizon in metadata["horizons"]:
        _verify_horizon(project, root, horizon, identity, record["fold_id"])


def verify_prediction_manifest(project_root: Path, output_root: Path) -> dict[str, Any]:
    project, root = project_root.resolve(), output_root.resolve()
    identity = _identity(load_protocol(project))
    manifest = _read_json(root / "prediction_manifest.json")
    if any(manifest.get(key) != value for key, value in identity.items()):
        raise ValueError("C2B final prediction identity changed")
    if tuple(item["fold_id"] for item in manifest["folds"]) != DEVELOPMENT_FOLD_IDS:
        raise ValueError("all four C2B folds must be saved before any outer scoring")
    for fold in manifest["folds"]:
        _verify_fold(project, root, fold, identity)
    return manifest


def _run_fold(
    inputs: S1RunnerInputs,
    protocol: dict[str, Any],
    cache: ComponentCache,
    root: Path,
    identity: dict[str, Any],
    fold: dict[str, Any],
) -> dict[str, Any]:
    fold_id = fold["id"]
    fold_root = root / "folds" / fold_id
    fold_root.mkdir(parents=True, exist_ok=True)
    fold_path = fold_root / "fold_manifest.json"
    if fold_path.exists():
        record = _record(root, fold_path, fold_id=fold_id, issue_count=99)
        _verify_fold(inputs.project_root, root, record, identity)
        _emit("fold_reused", fold_id=fold_id)
        return record
    records = []
    for horizon in HORIZONS:
        horizon_root = fold_root / f"horizon_{horizon:03d}"
        path = horizon_root / "horizon_manifest.json"
        if path.exists():
            record = _record(root, path, horizon_days=horizon)
            _verify_horizon(inputs.project_root, root, record, identity, fold_id)
            records.append(record)
            _emit("horizon_reused", fold_id=fold_id, horizon_days=horizon)
            continue
        samples = _inner_samples(inputs, fold, horizon, cache)
        cutoff = _utc(fold["outer_start"]) - timedelta(days=30)
        kernels = select_kernel_parameters(samples, cutoff, cache.area, protocol)
        fits, ridge_records = [], []
        for with_m5 in (False, True):
            fit, ridge_record = select_and_fit_ridge(samples, fold, cache.area, with_m5)
            fits.append(fit)
            ridge_records.append(ridge_record)
        issues = [
            row.issue_time_utc
            for row in inputs.outer_issues
            if row.fold_id == fold_id
            and row.horizon_days == horizon
            and row.primary_exposure_selected
        ]
        issue_us = [_epoch_us(issue) for issue in issues]
        if issue_us != _expected_horizon_axis(inputs.project_root, fold_id, horizon):
            raise ValueError("C0 issue ledger differs from the declared C2B issue axis")
        logs = np.stack(
            [_prediction_models(cache.get(issue), kernels, fits, protocol) for issue in issues]
        )
        _validate_log_mass(logs, (len(issues), len(MODEL_IDS), cache.area.size))
        prediction_path = horizon_root / "predictions.npz"
        arrays = {
            "fold_id": np.asarray(fold_id),
            "issue_times_us": np.asarray(issue_us, dtype=np.int64),
            "horizons_days": np.full(len(issues), horizon, dtype=np.int64),
            "model_ids": np.asarray(MODEL_IDS),
            "log_cell_mass": logs,
        }
        if prediction_path.exists():
            # A crash may occur after the payload but before the completion record.
            # Recompute from the same earlier-only inputs, require exact equality,
            # and seal the original payload; never overwrite or choose between runs.
            with np.load(prediction_path, allow_pickle=False) as saved:
                if set(saved.files) != set(arrays) or any(
                    not np.array_equal(saved[name], value) for name, value in arrays.items()
                ):
                    raise ValueError(
                        "unsealed prediction differs from deterministic recovery; "
                        "original preserved"
                    )
            _emit("orphan_payload_verified_unchanged", fold_id=fold_id, horizon_days=horizon)
        else:
            _atomic_npz(prediction_path, arrays)
        _write_json(
            path,
            {
                "identity": identity,
                "fold_id": fold_id,
                "horizon_days": horizon,
                "issue_count": len(issues),
                "completed_utc": datetime.now(UTC).isoformat(),
                "predictions": _record(root, prediction_path),
                "kernel_selection": kernels,
                "ridge_selection": ridge_records,
                "inner_target_counts": {
                    block: int(
                        sum(
                            sample.counts(cutoff).sum()
                            for sample in samples
                            if sample.block_id == block
                        )
                    )
                    for block in ("I1", "I2", "I3")
                },
                "outer_targets_read": False,
            },
        )
        records.append(_record(root, path, horizon_days=horizon))
        _emit(
            "horizon_complete",
            fold_id=fold_id,
            horizon_days=horizon,
            issue_count=len(issues),
            no_outer_scores=True,
        )
    _write_json(
        fold_path,
        {
            "identity": identity,
            "fold_id": fold_id,
            "horizons": records,
            "completed_utc": datetime.now(UTC).isoformat(),
        },
    )
    record = _record(root, fold_path, fold_id=fold_id, issue_count=99)
    _verify_fold(inputs.project_root, root, record, identity)
    _emit("fold_complete", fold_id=fold_id, issue_count=99)
    return record


def run_prediction_phase(
    *, project_root: Path, data_root: Path, output_root: Path | None = None, workers: int = 2
) -> Path:
    if isinstance(workers, bool) or workers not in (1, 2, 3):
        raise ValueError("C2B permits at most three fold workers")
    if any(os.environ.get(name) != "1" for name in _NUMERICAL_ENV):
        raise ValueError("set numerical-library thread limits to one before importing the runner")
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    project, data = project_root.resolve(), data_root.resolve()
    protocol = load_protocol(project)
    root = (output_root or project / protocol["outputs"]["root"]).resolve()
    if not root.is_relative_to(project / "outputs"):
        raise ValueError("C2B outputs must stay inside this worktree's outputs directory")
    root.mkdir(parents=True, exist_ok=True)
    with _run_lock(root):
        manifest_path = root / "prediction_manifest.json"
        if manifest_path.is_file():
            verify_prediction_manifest(project, root)
            return manifest_path
        identity = _identity(protocol)
        identity_path = root / "run_identity.json"
        if identity_path.exists():
            if _read_json(identity_path) != identity:
                raise ValueError("C2B saved run uses different code or input identity")
        else:
            _write_json(identity_path, identity)
        inputs, catalog = load_inputs(project, data, protocol)
        cache = ComponentCache(inputs, catalog, protocol, root, identity)
        _emit(
            "prediction_started",
            fold_workers=workers,
            total_issue_horizon_pairs=396,
            numerical_threads=1,
        )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_run_fold, inputs, protocol, cache, root, identity, fold)
                for fold in inputs.contract["outer_folds"]
            ]
            records = [future.result() for future in futures]
        _write_json(
            manifest_path,
            {
                **identity,
                "folds": records,
                "completed_utc": datetime.now(UTC).isoformat(),
                "issue_horizon_pairs": 396,
                "outer_targets_read": False,
            },
        )
        verify_prediction_manifest(project, root)
        return manifest_path
