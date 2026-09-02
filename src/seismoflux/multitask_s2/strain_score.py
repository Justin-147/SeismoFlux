"""S2-C evaluation with genuine zero mass and a predeclared later-era slice."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import yaml
from pyproj import Transformer
from scipy.special import logsumexp
from shapely.strtree import STRtree

from seismoflux.background.grid import EQUAL_AREA_CRS
from seismoflux.d1_replay.spatial import build_d1_spatial_domain_from_bytes
from seismoflux.multitask_s1 import c2b_score
from seismoflux.multitask_s1.c2b_score import (
    BANDS,
    PARENT_ARTIFACT_CONFIG,
    _sha,
    _verified,
    projected_near_cells,
    validate_targets,
)
from seismoflux.multitask_s1.development_contract import DEVELOPMENT_FOLD_IDS
from seismoflux.multitask_s1.development_predict import LOCATION_MODEL_IDS
from seismoflux.multitask_s2.fault_score import (
    FINAL_NAMES,
    TABLE_NAMES,
    _artifact_records,
    _completed_manifest,
    _copy_once,
    _fold_checkpoint,
    _load_reference_tables,
    _reference_paths,
    _validate_references,
    _write_json_once,
    _write_or_compare_table,
)
from seismoflux.multitask_s2.strain import validate_log_mass
from seismoflux.stage2s.contracts import SpatialGrid


def log_alarm_prefixes(
    log_mass: np.ndarray, grid: SpatialGrid, budgets: list[float]
) -> list[dict[str, Any]]:
    """Preserve the old exact log ranking, including row/column/ID zero ties."""
    values = validate_log_mass(log_mass, expected_cells=grid.cell_count)
    density = values - np.log(grid.clipped_area_km2)
    order = sorted(
        range(grid.cell_count),
        key=lambda i: (
            -float(density[i]),
            int(grid.rows[i]),
            int(grid.columns[i]),
            grid.cell_ids[i].encode("utf-8"),
        ),
    )
    result = []
    for budget in budgets:
        selected: list[int] = []
        areas: list[float] = []
        for index in order:
            area = float(grid.clipped_area_km2[index])
            if math.fsum([*areas, area]) > budget:
                break
            selected.append(index)
            areas.append(area)
        result.append(
            {
                "area_budget_km2": float(budget),
                "actual_area_km2": math.fsum(areas),
                "selected": selected,
            }
        )
    return result


def score_exposure(
    *,
    log_mass: np.ndarray,
    grid: SpatialGrid,
    target: dict[str, Any],
    fold_id: str,
    horizon_days: int,
    issue_time_us: int,
    magnitude_bin: str,
    model_id: str,
    budgets: list[float],
    near_cells: list[set[int]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Same S1 target aggregation; only the prefix validator now permits -inf."""
    prefixes = log_alarm_prefixes(log_mass, grid, budgets)
    cells = np.asarray(target["event_cell_indices"], dtype=np.int64)
    event_logs = np.asarray(log_mass[cells] - np.log(grid.clipped_area_km2[cells]))
    anchor = np.asarray(target["is_episode_anchor"], dtype=bool)
    weights = {
        "all": np.ones(len(cells)),
        "anchor": anchor.astype(float),
        "episode_balanced": 1.0 / np.asarray(target["global_episode_member_counts"], dtype=float),
        "subsequent": (~anchor).astype(float),
    }
    if len(near_cells) != len(cells):
        raise ValueError("secondary spatial neighborhoods are not target aligned")
    identity = {
        "fold_id": fold_id,
        "horizon_days": horizon_days,
        "issue_time_us": issue_time_us,
        "magnitude_bin": magnitude_bin,
        "model_id": model_id,
    }
    exposures, events, alarms = [], [], []
    for prefix in prefixes:
        selected = set(prefix["selected"])
        area = {key: prefix[key] for key in ("area_budget_km2", "actual_area_km2")}
        alarms.append({**identity, **area, "selected_cell_indices": prefix["selected"]})
        for tolerance in (0.0, 70.0):
            hit = np.asarray(
                [
                    int(cell) in selected if tolerance == 0 else bool(selected & nearby)
                    for cell, nearby in zip(cells, near_cells, strict=True)
                ],
                dtype=bool,
            )
            row = {
                **identity,
                **area,
                "hit_tolerance_km": tolerance,
                "event_count": len(cells),
                "event_log_density_sum": float(event_logs.sum()),
                "event_mean_log_density": float(event_logs.mean()) if len(cells) else None,
            }
            for view, weight in weights.items():
                row[f"{view}_hits"] = float(weight[hit].sum())
                row[f"{view}_total"] = float(weight.sum())
            exposures.append(row)
            for index, event_id in enumerate(target["event_ids"]):
                events.append(
                    {
                        **identity,
                        **area,
                        "hit_tolerance_km": tolerance,
                        "event_id": event_id,
                        "episode_id": target["episode_ids"][index],
                        "is_episode_anchor": bool(anchor[index]),
                        "global_episode_member_count": target["global_episode_member_counts"][
                            index
                        ],
                        "cell_index": int(cells[index]),
                        "hit": bool(hit[index]),
                        "longitude": target["event_longitudes"][index],
                        "latitude": target["event_latitudes"][index],
                        "log_density_per_km2": float(event_logs[index]),
                    }
                )
    return exposures, events, alarms


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
        raise ValueError("S2C requires all four frozen development folds and all four models")
    axes: set[tuple[str, int, int]] = set()
    for fold, arrays in arrays_by_fold.items():
        axis = list(
            zip(arrays["horizons_days"].tolist(), arrays["issue_times_us"].tolist(), strict=True)
        )
        expected = protocol["calendar"]["outer_total_issue_horizon_pairs"] // 4
        mass = arrays["log_cell_mass"]
        if (
            len(axis) != expected
            or len(set(axis)) != len(axis)
            or set(h for h, _ in axis) != set(horizons)
            or tuple(arrays["model_ids"].tolist()) != model_ids
            or mass.shape != (expected, len(model_ids), protocol["inputs"]["grid_cells"])
            or np.isnan(mass).any()
            or np.isposinf(mass).any()
            or not np.allclose(logsumexp(mass, axis=-1), 0, atol=1e-9, rtol=0)
        ):
            raise ValueError("S2C fold shape, prediction axis, or normalized log mass changed")
        axes.update((fold, int(h), int(t)) for h, t in axis)
    if len(axes) != protocol["calendar"]["outer_total_issue_horizon_pairs"]:
        raise ValueError("S2C must preserve all issue-horizon pairs")
    if [sum(h == horizon for _, h, _ in axes) for horizon in horizons] != protocol["calendar"][
        "outer_issue_counts_per_horizon"
    ]:
        raise ValueError("S2C horizon-specific exposure counts changed")
    return axes


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


def _json_log_scores(curves: list[dict[str, Any]]) -> None:
    for curve in curves:
        value = curve["event_mean_log_density"]
        if value is None:
            curve["log_density_status"] = "no_events"
        elif value == -math.inf:
            curve["event_mean_log_density"] = None
            curve["log_density_status"] = "negative_infinity_from_zero_mass"
        elif math.isfinite(value):
            curve["log_density_status"] = "finite"
        else:
            raise ValueError("NaN or positive infinite log score is not a zero-mass outcome")


def _summarize(
    exposures: pd.DataFrame,
    events: pd.DataFrame,
    protocol: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], pd.DataFrame]:
    planned = protocol["planned_pairs"]
    if len(planned) != 6 or len({(pair[0], pair[1]) for pair in planned}) != 6:
        raise ValueError("S2C requires exactly the six registered comparisons")
    values = exposures.event_log_density_sum.to_numpy(dtype=float)
    if np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError("invalid exposure log-density sum must not be silently omitted")
    curves, pairings, paired = c2b_score.summarize(exposures, events, planned, new_models=())
    _json_log_scores(curves)
    return curves, pairings, paired


def _slice_summary(
    exposures: pd.DataFrame,
    events: pd.DataFrame,
    protocol: dict[str, Any],
    *,
    folds: tuple[str, ...],
    expected_main_anchors: int,
    role: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if not folds or not set(folds).issubset(DEVELOPMENT_FOLD_IDS):
        raise ValueError("S2C slices may use only the frozen development folds")
    exposure = exposures.loc[exposures.fold_id.isin(folds)]
    event = events.loc[events.fold_id.isin(folds)]
    curves, pairings, paired = _summarize(exposure, event, protocol)
    if (
        len(curves) != protocol["evaluation"]["expected_curves_per_slice"]
        or len(pairings) != protocol["evaluation"]["expected_pairings_per_slice"]
    ):
        raise ValueError("S2C slice is missing a model, horizon, band, area, or planned pairing")
    main = [
        row
        for row in curves
        if row["model_id"] == protocol["inputs"]["catalog_main_model"]
        and row["horizon_days"] == 30
        and row["magnitude_bin"] == "M5_6"
        and row["hit_tolerance_km"] == 0
        and row["area_budget_km2"] == 600000
    ]
    if len(main) != 1 or main[0]["anchor_total"] != expected_main_anchors:
        raise ValueError("S2C evaluation slice main anchor population changed")
    return {
        "fold_ids": list(folds),
        "scientific_role": role,
        "main_anchor_count": expected_main_anchors,
        "primary_issue_horizon_count": len(
            exposure[["fold_id", "horizon_days", "issue_time_us"]].drop_duplicates()
        ),
        "target_exposure_band_count": len(
            exposure[
                ["fold_id", "horizon_days", "issue_time_us", "magnitude_bin"]
            ].drop_duplicates()
        ),
        "curves": curves,
        "pairings": pairings,
        "new_independent_test_evidence": False,
        "historical_prospective_evidence": False,
    }, paired


def _run_score_phase_locked(
    *, project_root: Path, data_root: Path, output_root: Path | None = None
) -> Path:
    from seismoflux.multitask_s2.strain_predict import (
        HORIZONS,
        MODEL_IDS,
        PROTOCOL_PATH,
        _output_root,
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
        raise ValueError("all four S2C prediction folds must be complete before target access")
    new = {record["fold_id"]: load_fold_arrays(root, record) for record in manifest["folds"]}
    axes = _prediction_axes(manifest, new, protocol, tuple(MODEL_IDS), tuple(HORIZONS))
    # No target-bearing table or old evaluation parquet is read before this point.
    paths, reference_manifest = _reference_paths(project, protocol)
    parents = yaml.safe_load((project / PARENT_ARTIFACT_CONFIG).read_text("utf-8"))[
        "parent_artifacts"
    ]
    raw_path = _verified(project, parents["C0_raw_scores_score_phase_only"])
    if reference_manifest["identity"]["C0_raw_scores_sha256"] != _sha(raw_path):
        raise ValueError("S2C and saved C2B references use different targets")
    identity = {
        "protocol_sha256": _sha(project / PROTOCOL_PATH),
        "prediction_manifest_sha256": _sha(root / "prediction_manifest.json"),
        "catalog_score_manifest_sha256": protocol["inputs"]["catalog_score_manifest_sha256"],
        "C0_raw_scores_sha256": _sha(raw_path),
        "implementation_hashes": {
            "strain_score.py": _sha(Path(__file__)),
            "strain.py": _sha(Path(__file__).with_name("strain.py")),
            "fault_score.py": _sha(Path(__file__).with_name("fault_score.py")),
            "c2b_score.py": _sha(Path(c2b_score.__file__)),
        },
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
        raise ValueError("S2C evaluation grid differs from the saved reference grid")
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
    full, paired = _slice_summary(
        combined[TABLE_NAMES[0]],
        combined[TABLE_NAMES[1]],
        protocol,
        folds=DEVELOPMENT_FOLD_IDS,
        expected_main_anchors=protocol["evaluation"]["expected_main_anchor_count"],
        role="all_four_development_folds_retrospective",
    )
    later, _ = _slice_summary(
        combined[TABLE_NAMES[0]],
        combined[TABLE_NAMES[1]],
        protocol,
        folds=tuple(protocol["calendar"]["post_release_development_folds"]),
        expected_main_anchors=protocol["evaluation"]["post_release_expected_main_anchor_count"],
        role="post_release_2015_2019_development_not_independent_test",
    )
    summary = {
        **full,
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "scientific_role": protocol["scientific_role"],
        "model_ids": [*reference_models, *MODEL_IDS],
        "reference_model_ids": list(reference_models),
        "new_model_ids": list(MODEL_IDS),
        "horizons_days": list(HORIZONS),
        "magnitude_bins": list(BANDS),
        "planned_pair_count": 6,
        "post_release_development": later,
        "evaluation_slices": [
            {"id": "all_four_development_folds_retrospective", "summary_location": "root"},
            {
                "id": "post_release_2015_2019_development",
                "summary_location": "post_release_development",
            },
        ],
        "reference_scores_reused_without_recalculation": True,
        "static_snapshot_available_at": protocol["inputs"]["static_snapshot_available_at"],
        "historical_exact_bytes_confirmed": False,
        "static_snapshot_role": "GSRM_2014_label_current_bytes_retrospective_development_covariate",
        "source_attribution": protocol["inputs"]["strain_source"]["attribution"],
        "strain_derived_artifact_license": "CC-BY-NC-SA-3.0",
        "zero_mass_log_score_status": "negative_infinity_from_zero_mass_not_missing",
        "negative_infinity_log_score_difference": "undefined_never_replaced_by_zero",
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
    *, project_root: Path, data_root: Path, output_root: Path | None = None
) -> Path:
    from seismoflux.multitask_s1.c2b_predict import _run_lock
    from seismoflux.multitask_s2.strain_predict import _output_root, load_protocol

    if any(
        os.environ.get(name) != "1"
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    ):
        raise ValueError("S2C score requires numerical-library thread limits of one")
    project = project_root.resolve()
    protocol = load_protocol(project)
    root = _output_root(project, protocol, output_root)
    root.mkdir(parents=True, exist_ok=True)
    with _run_lock(root):
        return _run_score_phase_locked(project_root=project, data_root=data_root, output_root=root)
