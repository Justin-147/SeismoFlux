"""S2-A location evaluation: six new surfaces, four unchanged saved references.

All four development prediction folds must be complete before any target-bearing
table is opened. The 2026 fault snapshot is retrospective static information,
not a historical real-time covariate or evidence of time/magnitude skill.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from pyproj import Transformer
from scipy.special import logsumexp  # type: ignore[import-untyped]
from shapely.strtree import STRtree

from seismoflux.background.grid import EQUAL_AREA_CRS
from seismoflux.d1_replay.spatial import build_d1_spatial_domain_from_bytes
from seismoflux.multitask_s1 import c2b_score
from seismoflux.multitask_s1.c2b_score import (
    BANDS,
    PARENT_ARTIFACT_CONFIG,
    _json,
    _sha,
    _verified,
    projected_near_cells,
    score_exposure,
    validate_targets,
)
from seismoflux.multitask_s1.development_contract import DEVELOPMENT_FOLD_IDS
from seismoflux.multitask_s1.development_predict import LOCATION_MODEL_IDS
from seismoflux.stage2s.contracts import SpatialGrid

TABLE_NAMES = ("exposure_results.parquet", "event_results.parquet", "alarm_prefixes.parquet")
FINAL_NAMES = (
    *TABLE_NAMES,
    "paired_anchor_results.parquet",
    "grid_geometry.parquet",
    "summary.json",
)
AXES = ("fold_id", "horizon_days", "issue_time_us", "magnitude_bin")


def _output_root(project: Path, protocol: dict[str, Any], output_root: Path | None) -> Path:
    root = (
        output_root if output_root is not None else project / protocol["outputs"]["root"]
    ).resolve()
    base = (project / "outputs/multitask_s2").resolve()
    if root == base or not root.is_relative_to(base):
        raise ValueError("S2A score outputs must remain below project/outputs/multitask_s2")
    return root


def _json_bytes(value: dict[str, Any]) -> bytes:
    def scalar(item: Any) -> Any:
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"non-JSON score value: {type(item).__name__}")

    return (
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=scalar) + "\n"
    ).encode("utf-8")


def _publish_partial(partial: Path, final: Path) -> None:
    if final.exists():
        raise FileExistsError(f"completed score artifact cannot be replaced: {final}")
    os.replace(partial, final)


def _write_json_once(path: Path, value: dict[str, Any]) -> None:
    payload = _json_bytes(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"saved score JSON differs from recomputed content: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.{uuid.uuid4().hex}.partial")
    with partial.open("xb") as stream:
        stream.write(payload)
    _publish_partial(partial, path)


def _write_or_compare_table(path: Path, frame: pd.DataFrame) -> None:
    """Complete an interrupted fold without replacing an already written table."""

    table = pa.Table.from_pandas(frame, preserve_index=False)
    if path.exists():
        saved = pq.read_table(path, use_threads=False)
        if not saved.equals(table, check_metadata=False):
            raise ValueError(f"saved unsealed score table differs from recomputation: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.{uuid.uuid4().hex}.partial")
    with partial.open("xb") as stream:
        pq.write_table(table, stream, compression="zstd")
    _publish_partial(partial, path)


def _copy_once(source: Path, destination: Path) -> None:
    if destination.exists():
        if _sha(source) != _sha(destination):
            raise ValueError("saved grid geometry differs from the verified C2B geometry")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.{uuid.uuid4().hex}.partial")
    with source.open("rb") as reader, partial.open("xb") as writer:
        shutil.copyfileobj(reader, writer)
    _publish_partial(partial, destination)


def _artifact_records(root: Path, names: tuple[str, ...]) -> list[dict[str, Any]]:
    return [{"path": name, "sha256": _sha(root / name)} for name in names]


def _completed_manifest(
    root: Path,
    name: str,
    identity: dict[str, Any],
    names: tuple[str, ...],
) -> dict[str, Any] | None:
    path = root / name
    if not path.exists():
        return None
    record = _json(path)
    if not record.get("complete") or record.get("identity") != identity:
        raise ValueError("saved S2A score checkpoint belongs to different or incomplete inputs")
    if {item["path"] for item in record["artifacts"]} != set(names):
        raise ValueError("saved S2A score checkpoint artifact list is incomplete")
    for item in record["artifacts"]:
        _verified(root, item)
    return record


def _prediction_axes(
    manifest: dict[str, Any],
    arrays_by_fold: dict[str, dict[str, np.ndarray]],
    protocol: dict[str, Any],
    model_ids: tuple[str, ...],
    horizons: tuple[int, ...],
) -> set[tuple[str, int, int]]:
    if (
        tuple(item["fold_id"] for item in manifest["folds"]) != DEVELOPMENT_FOLD_IDS
        or tuple(arrays_by_fold) != DEVELOPMENT_FOLD_IDS
        or tuple(protocol["calendar"]["outer_folds"]) != DEVELOPMENT_FOLD_IDS
        or tuple(protocol["calendar"]["horizons_days"]) != horizons
        or tuple(protocol["models"]) != model_ids
    ):
        raise ValueError("S2A requires all four frozen development folds and all six models")
    axes: set[tuple[str, int, int]] = set()
    for fold, arrays in arrays_by_fold.items():
        axis = list(
            zip(arrays["horizons_days"].tolist(), arrays["issue_times_us"].tolist(), strict=True)
        )
        expected_fold_count = protocol["calendar"]["outer_total_issue_horizon_pairs"] // 4
        mass = arrays["log_cell_mass"]
        if (
            len(axis) != expected_fold_count
            or len(set(axis)) != len(axis)
            or set(h for h, _ in axis) != set(horizons)
            or tuple(arrays["model_ids"].tolist()) != model_ids
            or mass.shape != (expected_fold_count, len(model_ids), protocol["inputs"]["grid_cells"])
            or not np.isfinite(mass).all()
            or not np.allclose(logsumexp(mass, axis=-1), 0, atol=1e-9, rtol=0)
        ):
            raise ValueError("S2A fold shape, prediction axis, or normalized log mass changed")
        axes.update((fold, int(h), int(t)) for h, t in axis)
    if len(axes) != protocol["calendar"]["outer_total_issue_horizon_pairs"]:
        raise ValueError("S2A must preserve all 396 issue-horizon pairs")
    if [sum(h == horizon for _, h, _ in axes) for horizon in horizons] != protocol["calendar"][
        "outer_issue_counts_per_horizon"
    ]:
        raise ValueError("S2A horizon-specific exposure counts changed")
    return axes


def _reference_paths(
    project: Path, protocol: dict[str, Any]
) -> tuple[dict[str, Path], dict[str, Any]]:
    root = (project / protocol["inputs"]["catalog_run"] / "score_phase").resolve()
    record_path = root / "score_manifest.json"
    if (
        not root.is_relative_to(project)
        or _sha(record_path) != protocol["inputs"]["catalog_score_manifest_sha256"]
    ):
        raise ValueError("saved C2B score manifest differs from the frozen S2A reference")
    manifest = _json(record_path)
    refs = {item["path"]: item for item in manifest["artifacts"]}
    paths = {name: _verified(root, refs[name]) for name in (*TABLE_NAMES, "grid_geometry.parquet")}
    return paths, manifest


def _load_reference_tables(
    paths: dict[str, Path],
    reference_models: tuple[str, ...],
    horizons: tuple[int, ...],
) -> dict[str, pd.DataFrame]:
    filters = [
        ("model_id", "in", list(reference_models)),
        ("fold_id", "in", list(DEVELOPMENT_FOLD_IDS)),
        ("horizon_days", "in", list(horizons)),
        ("magnitude_bin", "in", list(BANDS)),
    ]
    return {name: pd.read_parquet(paths[name], filters=filters) for name in TABLE_NAMES}


def _validate_references(
    tables: dict[str, pd.DataFrame],
    targets: dict[tuple[str, int, int, str], dict[str, Any]],
    reference_models: tuple[str, ...],
    budgets: list[float],
) -> None:
    """Preserve the saved target population and each reference's actual paid area."""

    exposure = tables["exposure_results.parquet"]
    keys = [*AXES, "model_id", "area_budget_km2", "hit_tolerance_km"]
    expected = {
        (*axis, model, float(budget), tolerance)
        for axis in targets
        for model in reference_models
        for budget in budgets
        for tolerance in (0.0, 70.0)
    }
    actual = set(exposure[keys].itertuples(index=False, name=None))
    if len(exposure) != len(expected) or actual != expected:
        raise ValueError("saved reference exposure axes differ or omit zero-event periods")
    for row in exposure.itertuples(index=False):
        target = targets[
            (row.fold_id, int(row.horizon_days), int(row.issue_time_us), row.magnitude_bin)
        ]
        count, anchor_count = len(target["event_ids"]), sum(target["is_episode_anchor"])
        if (
            row.event_count != count
            or row.all_total != count
            or row.anchor_total != anchor_count
            or row.actual_area_km2 < 0
            or row.actual_area_km2 > row.area_budget_km2
        ):
            raise ValueError("saved reference target counts or actual alarm areas changed")
    alarms = tables["alarm_prefixes.parquet"]
    alarm_keys = [*AXES, "model_id", "area_budget_km2"]
    expected_alarms = {key[:-1] for key in expected}
    if (
        len(alarms) != len(expected_alarms)
        or set(alarms[alarm_keys].itertuples(index=False, name=None)) != expected_alarms
    ):
        raise ValueError("saved reference alarm prefixes omit an exposure or area budget")
    strict = exposure.loc[exposure.hit_tolerance_km == 0, [*alarm_keys, "actual_area_km2"]]
    area_pairs = strict.merge(
        alarms[[*alarm_keys, "actual_area_km2"]],
        on=alarm_keys,
        suffixes=("_score", "_alarm"),
        validate="one_to_one",
    )
    if not np.array_equal(area_pairs.actual_area_km2_score, area_pairs.actual_area_km2_alarm):
        raise ValueError("saved reference score and alarm actual areas differ")
    events = tables["event_results.parquet"]
    event_keys = [*keys, "event_id"]
    if events.duplicated(event_keys).any():
        raise ValueError("saved reference physical events are duplicated")
    expected_event_count = (
        sum(len(target["event_ids"]) for target in targets.values())
        * len(reference_models)
        * len(budgets)
        * 2
    )
    if len(events) != expected_event_count:
        raise ValueError("saved reference event population changed")
    # C2B's fixed manifest binds all numerical scores. Check the deduplicated
    # metadata against the same old C0 targets rather than rescoring references.
    meta = [
        *AXES,
        "event_id",
        "episode_id",
        "is_episode_anchor",
        "global_episode_member_count",
        "cell_index",
        "longitude",
        "latitude",
    ]
    expected_meta = set()
    for axis, target in targets.items():
        expected_meta.update(
            (*axis, *values)
            for values in zip(
                target["event_ids"],
                target["episode_ids"],
                target["is_episode_anchor"],
                target["global_episode_member_counts"],
                target["event_cell_indices"],
                target["event_longitudes"],
                target["event_latitudes"],
                strict=True,
            )
        )
    if set(events[meta].itertuples(index=False, name=None)) != expected_meta:
        raise ValueError("saved reference event identities differ from unchanged C0 targets")


def _score_fold(
    *,
    fold: str,
    arrays: dict[str, np.ndarray],
    targets: dict[tuple[str, int, int, str], dict[str, Any]],
    grid: SpatialGrid,
    tree: STRtree,
    transformer: Transformer,
    model_ids: tuple[str, ...],
    budgets: list[float],
    reference_tables: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    exposures, events, alarms = [], [], []
    for index, (horizon, issue) in enumerate(
        zip(arrays["horizons_days"], arrays["issue_times_us"], strict=True)
    ):
        horizon, issue = int(horizon), int(issue)
        for band in BANDS:
            target = targets[(fold, horizon, issue, band)]
            near = []
            if target["event_ids"]:
                x, y = transformer.transform(target["event_longitudes"], target["event_latitudes"])
                near = projected_near_cells(tree, x, y)
            for model_index, model in enumerate(model_ids):
                scored, event, alarm = score_exposure(
                    log_mass=arrays["log_cell_mass"][index, model_index],
                    grid=grid,
                    target=target,
                    fold_id=fold,
                    horizon_days=horizon,
                    issue_time_us=issue,
                    magnitude_bin=band,
                    model_id=model,
                    budgets=budgets,
                    near_cells=near,
                )
                exposures.extend(scored)
                events.extend(event)
                alarms.extend(alarm)
    return {
        name: pd.DataFrame(rows, columns=reference_tables[name].columns)
        for name, rows in zip(TABLE_NAMES, (exposures, events, alarms), strict=True)
    }


def _fold_checkpoint(
    root: Path,
    fold: str,
    identity: dict[str, Any],
    compute: Any,
) -> dict[str, pd.DataFrame]:
    path = root / "folds" / fold
    bound = {**identity, "fold_id": fold}
    if _completed_manifest(path, "fold_score_manifest.json", bound, TABLE_NAMES) is not None:
        return {name: pd.read_parquet(path / name) for name in TABLE_NAMES}
    frames = compute()
    for name in TABLE_NAMES:
        _write_or_compare_table(path / name, frames[name])
    _write_json_once(
        path / "fold_score_manifest.json",
        {
            "schema_version": 1,
            "complete": True,
            "identity": bound,
            "artifacts": _artifact_records(path, TABLE_NAMES),
        },
    )
    return frames


def _summarize(
    exposures: pd.DataFrame,
    events: pd.DataFrame,
    protocol: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], pd.DataFrame]:
    planned = protocol["planned_pairs"]
    if len(planned) != 11 or len({(pair[0], pair[1]) for pair in planned}) != 11:
        raise ValueError("S2A requires exactly the eleven registered comparisons")
    return c2b_score.summarize(exposures, events, planned, new_models=())


def _run_score_phase_locked(
    *,
    project_root: Path,
    data_root: Path,
    output_root: Path | None = None,
) -> Path:
    from seismoflux.multitask_s2.fault_predict import (
        HORIZONS,
        MODEL_IDS,
        PROTOCOL_PATH,
        load_fold_arrays,
        load_protocol,
        verify_prediction_manifest,
    )

    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    project = project_root.resolve()
    protocol = load_protocol(project)
    root = _output_root(project, protocol, output_root)
    manifest = verify_prediction_manifest(project, root)
    if tuple(record["fold_id"] for record in manifest["folds"]) != DEVELOPMENT_FOLD_IDS:
        raise ValueError("all four S2A prediction folds must be complete before target access")
    new = {record["fold_id"]: load_fold_arrays(root, record) for record in manifest["folds"]}
    axes = _prediction_axes(manifest, new, protocol, tuple(MODEL_IDS), tuple(HORIZONS))
    # No old evaluation/target parquet has been read above this line.
    paths, reference_manifest = _reference_paths(project, protocol)
    parents = yaml.safe_load((project / PARENT_ARTIFACT_CONFIG).read_text("utf-8"))[
        "parent_artifacts"
    ]
    raw_path = _verified(project, parents["C0_raw_scores_score_phase_only"])
    if reference_manifest["identity"]["C0_raw_scores_sha256"] != _sha(raw_path):
        raise ValueError("S2A and the saved C2B reference use different C0 targets")
    identity = {
        "protocol_sha256": _sha(project / PROTOCOL_PATH),
        "prediction_manifest_sha256": _sha(root / "prediction_manifest.json"),
        "catalog_score_manifest_sha256": protocol["inputs"]["catalog_score_manifest_sha256"],
        "C0_raw_scores_sha256": _sha(raw_path),
    }
    score_root = root / "score_phase"
    if _completed_manifest(score_root, "score_manifest.json", identity, FINAL_NAMES) is not None:
        return score_root / "summary.json"
    rows = pd.read_parquet(
        raw_path,
        columns=[
            "score_family",
            "fold_id",
            "issue_time_utc",
            "horizon_days",
            "model_id",
            "payload_json",
        ],
        filters=[
            ("score_family", "==", "location"),
            ("fold_id", "in", list(DEVELOPMENT_FOLD_IDS)),
            ("horizon_days", "in", list(HORIZONS)),
            ("model_id", "in", list(LOCATION_MODEL_IDS)),
        ],
    )
    targets = validate_targets(
        rows,
        axes,
        cell_count=protocol["inputs"]["grid_cells"],
        expected_main_anchors=protocol["evaluation"]["expected_main_anchor_count"],
    )
    reference_models = tuple(protocol["evaluation"]["references"])
    reference_tables = _load_reference_tables(paths, reference_models, tuple(HORIZONS))
    budgets = protocol["evaluation"]["area_budgets_km2"]
    _validate_references(reference_tables, targets, reference_models, budgets)
    parent_run = yaml.safe_load(_verified(project, parents["C0_run_contract"]).read_text("utf-8"))
    study_path = _verified(data_root.resolve(), parent_run["input_identities"]["study_area"])
    domain = build_d1_spatial_domain_from_bytes(study_path.read_bytes())
    grid = domain.operational_grid
    saved_grid = pd.read_parquet(paths["grid_geometry.parquet"]).sort_values("cell_index")
    if (
        grid.grid_id != protocol["inputs"]["grid_id"]
        or grid.cell_count != protocol["inputs"]["grid_cells"]
        or tuple(saved_grid.cell_id) != grid.cell_ids
        or not np.array_equal(saved_grid.area_km2.to_numpy(), grid.clipped_area_km2)
    ):
        raise ValueError("S2A evaluation grid differs from the saved reference grid")
    tree = STRtree(domain.locator.clipped_geometries)
    transformer = Transformer.from_crs("EPSG:4326", EQUAL_AREA_CRS, always_xy=True)
    completed = []
    for fold in DEVELOPMENT_FOLD_IDS:

        def compute(fold_id: str = fold) -> dict[str, pd.DataFrame]:
            return _score_fold(
                fold=fold_id,
                arrays=new[fold_id],
                targets=targets,
                grid=grid,
                tree=tree,
                transformer=transformer,
                model_ids=tuple(MODEL_IDS),
                budgets=budgets,
                reference_tables=reference_tables,
            )

        completed.append(_fold_checkpoint(score_root, fold, identity, compute))
    combined = {
        name: pd.concat(
            [reference_tables[name], *(fold[name] for fold in completed)], ignore_index=True
        )
        for name in TABLE_NAMES
    }
    curves, pairings, paired = _summarize(
        combined[TABLE_NAMES[0]], combined[TABLE_NAMES[1]], protocol
    )
    for curve in curves:
        value = curve["event_mean_log_density"]
        if value is not None and not math.isfinite(value):
            curve["event_mean_log_density"] = None
            curve["log_density_status"] = "negative_infinity_from_saved_reference_zero_mass"
        else:
            curve["log_density_status"] = "finite" if value is not None else "no_events"
    summary = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "scientific_role": protocol["scientific_role"],
        "model_ids": [*reference_models, *MODEL_IDS],
        "reference_model_ids": list(reference_models),
        "new_model_ids": list(MODEL_IDS),
        "horizons_days": list(HORIZONS),
        "magnitude_bins": list(BANDS),
        "primary_issue_horizon_count": len(axes),
        "target_exposure_band_count": len(targets),
        "main_anchor_count": protocol["evaluation"]["expected_main_anchor_count"],
        "curves": curves,
        "pairings": pairings,
        "planned_pair_count": 11,
        "reference_scores_reused_without_recalculation": True,
        "static_snapshot_available_at": protocol["inputs"]["static_snapshot_available_at"],
        "static_snapshot_role": (
            "2026_geometry_retrospective_spatial_information_not_known_at_old_issue_dates"
        ),
        "historical_prospective_evidence": False,
        "new_independent_test_evidence": False,
        "time_magnitude_or_joint_skill_claim": False,
        "secondary_70km_definition": (
            "projected_real_epicenter_distance_to_clipped_alarm_cell_polygon_le_70km"
        ),
        "holdout_read": False,
        "locked_test_run": False,
        "bootstrap": {
            "unit": "nonoverlapping_time_exposure_whole_episodes_retained",
            "replicates": 2000,
            "seed": 147,
        },
    }
    for name in TABLE_NAMES:
        _write_or_compare_table(score_root / name, combined[name])
    _write_or_compare_table(score_root / "paired_anchor_results.parquet", paired)
    _copy_once(paths["grid_geometry.parquet"], score_root / "grid_geometry.parquet")
    _write_json_once(score_root / "summary.json", summary)
    _write_json_once(
        score_root / "score_manifest.json",
        {
            "schema_version": 1,
            "complete": True,
            "identity": identity,
            "artifacts": _artifact_records(score_root, FINAL_NAMES),
        },
    )
    return score_root / "summary.json"


def run_score_phase(
    *,
    project_root: Path,
    data_root: Path,
    output_root: Path | None = None,
) -> Path:
    """Use the same OS-held run lock as prediction, not a separate score lock."""

    from seismoflux.multitask_s1.c2b_predict import _run_lock
    from seismoflux.multitask_s2.fault_predict import load_protocol

    project = project_root.resolve()
    protocol = load_protocol(project)
    root = _output_root(project, protocol, output_root)
    root.mkdir(parents=True, exist_ok=True)
    with _run_lock(root):
        return _run_score_phase_locked(project_root=project, data_root=data_root, output_root=root)
