"""Location-only C2B evaluation reusing the unchanged sealed C0 targets."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import yaml
from pyproj import Transformer
from scipy.special import logsumexp  # type: ignore[import-untyped]
from shapely import points
from shapely.strtree import STRtree

from seismoflux.background.grid import EQUAL_AREA_CRS
from seismoflux.d1_replay.spatial import (
    build_d1_spatial_domain_from_bytes,
    select_alarm_prefixes,
)
from seismoflux.multitask_s1.development_contract import DEVELOPMENT_FOLD_IDS
from seismoflux.multitask_s1.development_predict import LOCATION_MODEL_IDS
from seismoflux.multitask_s1.input_sensitivity_score import TARGET_FIELDS, _issue_us
from seismoflux.stage2s.contracts import SpatialGrid

HORIZONS = (7, 30, 90, 180, 365)
BANDS = ("M5_6", "M6_plus")
C0_MODELS = tuple(f"C0_{name}" for name in LOCATION_MODEL_IDS)
PARENT_ARTIFACT_CONFIG = Path("configs/multitask_s1_c2a_input_sensitivity.yaml")
VIEWS = ("all", "anchor", "episode_balanced", "subsequent")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified(root: Path, record: dict[str, Any], key: str = "path") -> Path:
    path = (root / str(record[key])).resolve()
    if not path.is_relative_to(root.resolve()) or _sha(path) != record["sha256"]:
        raise ValueError("C2B parent artifact path or SHA changed")
    return path


def _json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    def scalar(item: Any) -> Any:
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"non-JSON artifact value: {type(item).__name__}")

    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False, default=scalar)
        stream.write("\n")


def projected_near_cells(tree: STRtree, x: Any, y: Any) -> list[set[int]]:
    """Event distance to clipped polygons in existing projected metre coordinates."""

    result: list[set[int]] = [set() for _ in x]
    if result:
        matches = tree.query(points(x, y), predicate="dwithin", distance=70000.0)
        for event_index, cell_index in matches.T:
            result[int(event_index)].add(int(cell_index))
    return result


def validate_targets(
    rows: pd.DataFrame,
    expected_axes: set[tuple[str, int, int]],
    *,
    cell_count: int,
    expected_main_anchors: int = 147,
) -> dict[tuple[str, int, int, str], dict[str, Any]]:
    """Reuse exact target vectors, preserving empty periods and separate horizons."""

    expected = {(*axis, band) for axis in expected_axes for band in BANDS}
    by_model: dict[str, dict[tuple[str, int, int, str], dict[str, Any]]] = {
        model: {} for model in LOCATION_MODEL_IDS
    }
    for row in rows.to_dict("records"):
        if row["score_family"] != "location" or row["model_id"] not in by_model:
            raise ValueError("unexpected C0 score family or reference model")
        payload = json.loads(row["payload_json"])
        if payload["metric"] != "spatial_log_density":
            continue
        band = payload["magnitude_bin"]
        key = (row["fold_id"], int(row["horizon_days"]), _issue_us(row["issue_time_utc"]), band)
        if key not in expected or key in by_model[row["model_id"]]:
            raise ValueError("extra or duplicate C0 target exposure")
        if (
            payload["fold_id"] != key[0]
            or payload["horizon_days"] != key[1]
            or _issue_us(payload["issue_time_utc"]) != key[2]
            or payload["model_id"] != row["model_id"]
            or payload["catalog_delay_hours"] != 24
            or payload["hit_tolerance_km"] != 0.0
            or payload["episode_definition"]
            != "full_catalog_fixed_anchor_30d_75km_by_magnitude_bin"
        ):
            raise ValueError("C0 target identity or causal boundary changed")
        target = {field: payload[field] for field in TARGET_FIELDS}
        if {len(values) for values in target.values()} != {payload["event_count"]}:
            raise ValueError("target vector lengths disagree")
        if any(type(value) is not bool for value in target["is_episode_anchor"]):
            raise ValueError("anchor flags must remain boolean")
        if any(
            type(value) is not int or not 0 <= value < cell_count
            for value in target["event_cell_indices"]
        ):
            raise ValueError("target cell index outside frozen grid")
        by_model[row["model_id"]][key] = target
    reference = by_model[LOCATION_MODEL_IDS[0]]
    if any(set(values) != expected or values != reference for values in by_model.values()):
        raise ValueError("C0 reference targets differ or omit empty exposures")
    seen: dict[tuple[int, str], set[str]] = {}
    anchors: dict[tuple[int, str], set[str]] = {}
    counts: dict[tuple[str, str], int] = {}
    for (_, horizon, _, band), target in reference.items():
        axis = (horizon, band)
        event_seen, anchor_seen = seen.setdefault(axis, set()), anchors.setdefault(axis, set())
        for event, episode, count, anchor in zip(
            target["event_ids"],
            target["episode_ids"],
            target["global_episode_member_counts"],
            target["is_episode_anchor"],
            strict=True,
        ):
            if event in event_seen:
                raise ValueError("duplicate event within one horizon and magnitude band")
            event_seen.add(event)
            if (
                type(count) is not int
                or count < 1
                or counts.setdefault((band, episode), count) != count
            ):
                raise ValueError("global episode size changed")
            if anchor:
                if episode in anchor_seen:
                    raise ValueError("duplicate fixed anchor within one horizon and band")
                anchor_seen.add(episode)
    if len(anchors.get((30, "M5_6"), set())) != expected_main_anchors:
        raise ValueError("the 30-day M5_6 fixed anchor population changed")
    return reference


def log_alarm_prefixes(
    log_mass: np.ndarray, grid: SpatialGrid, budgets: list[float]
) -> list[dict[str, Any]]:
    """The existing no-skip whole-cell rule, with exact log-density ranking."""

    values = np.asarray(log_mass, dtype=np.float64)
    if values.shape != (grid.cell_count,) or not np.isfinite(values).all():
        raise ValueError("C2B log mass must be finite and grid aligned")
    if abs(float(logsumexp(values))) > 1e-9:
        raise ValueError("C2B log mass must be normalized")
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
    c0_mass: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Score one unchanged population; 70km is secondary and never enlarges paid area."""

    if c0_mass is None:
        prefixes = log_alarm_prefixes(log_mass, grid, budgets)
    else:
        prefixes = [
            {
                "area_budget_km2": item.budget_km2,
                "actual_area_km2": item.actual_area_km2,
                "selected": item.selected_indices.tolist(),
            }
            for item in select_alarm_prefixes(c0_mass, grid)
        ]
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


def exposure_bootstrap(delta_hits: np.ndarray, totals: np.ndarray) -> list[float] | None:
    """Paired 2000-draw exposure-ratio bootstrap, retaining zero-event periods."""

    if len(delta_hits) != len(totals) or not len(totals) or totals.sum() == 0:
        return None
    rng = np.random.default_rng(147)
    sampled = rng.integers(0, len(totals), size=(2000, len(totals)))
    denominators = totals[sampled].sum(axis=1)
    valid = denominators > 0
    draws = delta_hits[sampled].sum(axis=1)[valid] / denominators[valid] * 100.0
    return np.quantile(draws, [0.025, 0.975]).tolist() if len(draws) else None


def summarize(
    exposures: pd.DataFrame,
    events: pd.DataFrame,
    planned_pairs: list[list[str]],
    new_models: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], pd.DataFrame]:
    axes = ["horizon_days", "magnitude_bin", "hit_tolerance_km", "area_budget_km2"]
    curves = []
    for key, group in exposures.groupby(["model_id", *axes], sort=True):
        row = dict(zip(["model_id", *axes], key, strict=True))
        row.update(
            {
                "actual_area_mean_km2": float(group.actual_area_km2.mean()),
                "actual_area_min_km2": float(group.actual_area_km2.min()),
                "actual_area_max_km2": float(group.actual_area_km2.max()),
                "exposure_count": len(group),
                "empty_exposure_count": int((group.event_count == 0).sum()),
            }
        )
        for view in VIEWS:
            hits, total = float(group[f"{view}_hits"].sum()), float(group[f"{view}_total"].sum())
            row.update(
                {
                    f"{view}_hits": hits,
                    f"{view}_total": total,
                    f"{view}_recall": hits / total if total else None,
                }
            )
        count = int(group.event_count.sum())
        row["event_mean_log_density"] = (
            float(group.event_log_density_sum.sum()) / count if count else None
        )
        row["per_fold"] = [
            {
                "fold_id": fold,
                "anchor_hits": float(part.anchor_hits.sum()),
                "anchor_total": float(part.anchor_total.sum()),
            }
            for fold, part in group.groupby("fold_id", sort=True)
        ]
        curves.append(row)
    pairs = [(pair[0], pair[1], pair[2], False) for pair in planned_pairs]
    pairs.extend(
        (model, baseline, "existing_reference_main_anchor_only", True)
        for model in new_models
        for baseline in ("C0_L3_B0_R30_CAUSAL", "C0_L2_KDE_CAUSAL")
    )
    summaries, details = [], []
    join_keys = ["fold_id", "issue_time_us", *axes]
    anchors = events.loc[events.is_episode_anchor].copy()
    for candidate, reference, purpose, main_only in pairs:
        left = exposures.loc[exposures.model_id == candidate]
        right = exposures.loc[exposures.model_id == reference]
        paired = left.merge(
            right,
            on=join_keys,
            suffixes=("_candidate", "_reference"),
            how="outer",
            validate="one_to_one",
            indicator=True,
        )
        if not paired._merge.eq("both").all() or not np.array_equal(
            paired.anchor_total_candidate, paired.anchor_total_reference
        ):
            raise ValueError("paired exposure populations differ")
        if main_only:
            paired = paired.loc[
                (paired.horizon_days == 30)
                & (paired.magnitude_bin == "M5_6")
                & (paired.hit_tolerance_km == 0)
                & (paired.area_budget_km2 == 600000)
            ]
        event_keys = [*join_keys, "event_id", "episode_id"]
        anchor_pair = anchors.loc[anchors.model_id == candidate].merge(
            anchors.loc[anchors.model_id == reference],
            on=event_keys,
            suffixes=("_candidate", "_reference"),
            how="outer",
            validate="one_to_one",
            indicator=True,
        )
        if not anchor_pair._merge.eq("both").all():
            raise ValueError("paired anchor identities differ")
        if main_only:
            anchor_pair = anchor_pair.loc[
                (anchor_pair.horizon_days == 30)
                & (anchor_pair.magnitude_bin == "M5_6")
                & (anchor_pair.hit_tolerance_km == 0)
                & (anchor_pair.area_budget_km2 == 600000)
            ]
        anchor_pair["candidate_model_id"], anchor_pair["reference_model_id"] = candidate, reference
        anchor_pair["net_hit"] = anchor_pair.hit_candidate.astype(
            int
        ) - anchor_pair.hit_reference.astype(int)
        details.append(anchor_pair.drop(columns="_merge"))
        for key, group in paired.groupby(axes, sort=True):
            subset = anchor_pair
            for name, value in zip(axes, key, strict=True):
                subset = subset.loc[subset[name] == value]
            delta = (group.anchor_hits_candidate - group.anchor_hits_reference).to_numpy(float)
            total = group.anchor_total_candidate.to_numpy(float)
            denominator = float(total.sum())
            row = {
                **dict(zip(axes, key, strict=True)),
                "candidate_model_id": candidate,
                "reference_model_id": reference,
                "purpose": purpose,
                "anchor_total": int(denominator),
                "net_hits": int(delta.sum()),
                "gained": int((subset.net_hit == 1).sum()),
                "lost": int((subset.net_hit == -1).sum()),
                "delta_recall_pp": float(delta.sum() / denominator * 100) if denominator else None,
                "bootstrap_ci95_pp": exposure_bootstrap(delta, total),
                "bootstrap_unit": "paired_nonoverlapping_time_exposure_including_empty",
                "per_fold": [],
            }
            for fold, part in group.groupby("fold_id", sort=True):
                net = float((part.anchor_hits_candidate - part.anchor_hits_reference).sum())
                row["per_fold"].append(
                    {
                        "fold_id": fold,
                        "net_hits": int(net),
                        "anchor_total": int(part.anchor_total_candidate.sum()),
                    }
                )
            summaries.append(row)
    return curves, summaries, pd.concat(details, ignore_index=True) if details else pd.DataFrame()


def _load_c0(project: Path, references: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    seal_path = _verified(project, references["C0_four_fold_prediction_seal"])
    seal = _json(seal_path)
    if tuple(item["fold_id"] for item in seal["ordered_fold_predictions"]) != DEVELOPMENT_FOLD_IDS:
        raise ValueError("C0 sealed fold identities changed")
    result = {}
    for record in seal["ordered_fold_predictions"]:
        bundle = _json(_verified(seal_path.parent, record, "relative_path"))
        if bundle["fold_id"] != record["fold_id"] or len(bundle["prediction_artifacts"]) != 1:
            raise ValueError("C0 fold bundle changed")
        path = _verified(seal_path.parent, bundle["prediction_artifacts"][0], "relative_path")
        with np.load(path, allow_pickle=False) as arrays:
            result[record["fold_id"]] = {
                key: np.asarray(arrays[key])
                for key in (
                    "primary_issue_time_us",
                    "primary_horizon_days",
                    "location_model_index",
                    "location_relative_mass",
                )
            }
        item = result[record["fold_id"]]
        if not np.array_equal(item["location_model_index"], np.arange(5)):
            raise ValueError("C0 reference model axis changed")
        mass = item["location_relative_mass"]
        if (
            not np.isfinite(mass).all()
            or np.any(mass < 0)
            or not np.allclose(mass.sum(axis=-1), 1, atol=1e-10, rtol=0)
        ):
            raise ValueError("C0 reference mass changed")
    return result


def run_score_phase(
    *, project_root: Path, data_root: Path, output_root: Path | None = None
) -> Path:
    from seismoflux.multitask_s1.c2b_predict import (
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
    root = (
        output_root.resolve() if output_root is not None else project / protocol["outputs"]["root"]
    )
    manifest = verify_prediction_manifest(project, root)
    new = {record["fold_id"]: load_fold_arrays(root, record) for record in manifest["folds"]}
    references = yaml.safe_load((project / PARENT_ARTIFACT_CONFIG).read_text("utf-8"))[
        "parent_artifacts"
    ]
    c0 = _load_c0(project, references)
    raw_path = _verified(project, references["C0_raw_scores_score_phase_only"])
    identity = {
        "protocol_sha256": _sha(project / PROTOCOL_PATH),
        "prediction_manifest": manifest,
        "C0_raw_scores_sha256": _sha(raw_path),
    }
    score_root = root / "score_phase"
    manifest_path = score_root / "score_manifest.json"
    if manifest_path.exists():
        saved = _json(manifest_path)
        if saved["identity"] != identity:
            raise ValueError("completed C2B score belongs to different inputs")
        for record in saved["artifacts"]:
            _verified(score_root, record)
        return score_root / "summary.json"
    if score_root.exists() and any(score_root.iterdir()):
        raise FileExistsError("incomplete C2B score output preserved; review before replacement")
    axes: set[tuple[str, int, int]] = set()
    for fold, arrays in new.items():
        log_mass = arrays["log_cell_mass"]
        if (
            tuple(arrays["model_ids"].tolist()) != tuple(MODEL_IDS)
            or log_mass.shape != (99, len(MODEL_IDS), protocol["inputs"]["grid_cells"])
            or not np.isfinite(log_mass).all()
            or not np.allclose(logsumexp(log_mass, axis=-1), 0.0, atol=1e-9, rtol=0)
            or c0[fold]["location_relative_mass"].shape != (99, 5, protocol["inputs"]["grid_cells"])
        ):
            raise ValueError("C2B or C0 prediction shape, model axis or normalization changed")
        new_axis = list(
            zip(arrays["horizons_days"].tolist(), arrays["issue_times_us"].tolist(), strict=True)
        )
        old_axis = list(
            zip(
                c0[fold]["primary_horizon_days"].tolist(),
                c0[fold]["primary_issue_time_us"].tolist(),
                strict=True,
            )
        )
        if len(new_axis) != 99 or len(set(new_axis)) != 99 or set(new_axis) != set(old_axis):
            raise ValueError("C2B and C0 issue-horizon axes differ")
        axes.update((fold, int(h), int(t)) for h, t in new_axis)
    if len(axes) != 396:
        raise ValueError("C2B exposure population changed")
    # No target-bearing table is opened before every new and C0 prediction was checked.
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
    targets = validate_targets(rows, axes, cell_count=protocol["inputs"]["grid_cells"])
    parent_run = yaml.safe_load(
        _verified(project, references["C0_run_contract"]).read_text("utf-8")
    )
    study_path = _verified(data_root.resolve(), parent_run["input_identities"]["study_area"])
    domain = build_d1_spatial_domain_from_bytes(study_path.read_bytes())
    grid = domain.operational_grid
    if grid.grid_id != protocol["inputs"]["grid_id"]:
        raise ValueError("C2B evaluation grid changed")
    tree = STRtree(domain.locator.clipped_geometries)
    transformer = Transformer.from_crs("EPSG:4326", EQUAL_AREA_CRS, always_xy=True)
    exposures, events, alarms = [], [], []
    for fold in DEVELOPMENT_FOLD_IDS:
        arrays = new[fold]
        old = c0[fold]
        old_rows = {
            (int(h), int(t)): i
            for i, (h, t) in enumerate(
                zip(old["primary_horizon_days"], old["primary_issue_time_us"], strict=True)
            )
        }
        for index, (horizon, issue) in enumerate(
            zip(arrays["horizons_days"], arrays["issue_times_us"], strict=True)
        ):
            horizon, issue = int(horizon), int(issue)
            old_index = old_rows[(horizon, issue)]
            for band in BANDS:
                target = targets[(fold, horizon, issue, band)]
                nearby: list[set[int]] = [set() for _ in target["event_ids"]]
                if nearby:
                    x, y = transformer.transform(
                        target["event_longitudes"], target["event_latitudes"]
                    )
                    nearby = projected_near_cells(tree, x, y)
                models = [*C0_MODELS, *MODEL_IDS]
                for model_index, model in enumerate(models):
                    mass = (
                        old["location_relative_mass"][old_index, model_index]
                        if model_index < 5
                        else None
                    )
                    if mass is None:
                        log_mass = arrays["log_cell_mass"][index, model_index - 5]
                    else:
                        with np.errstate(divide="ignore"):
                            log_mass = np.log(mass)
                    scored, event, alarm = score_exposure(
                        log_mass=log_mass,
                        grid=grid,
                        target=target,
                        fold_id=fold,
                        horizon_days=horizon,
                        issue_time_us=issue,
                        magnitude_bin=band,
                        model_id=model,
                        budgets=protocol["evaluation"]["area_budgets_km2"],
                        near_cells=nearby,
                        c0_mass=mass,
                    )
                    exposures.extend(scored)
                    events.extend(event)
                    alarms.extend(alarm)
    exposure_frame, event_frame = pd.DataFrame(exposures), pd.DataFrame(events)
    curves, pairings, paired = summarize(
        exposure_frame, event_frame, protocol["planned_pairs"], tuple(MODEL_IDS)
    )
    summary = {
        "protocol_id": protocol["protocol_id"],
        "scientific_role": protocol["scientific_role"],
        "model_ids": [*C0_MODELS, *MODEL_IDS],
        "horizons_days": list(HORIZONS),
        "magnitude_bins": list(BANDS),
        "primary_issue_horizon_count": len(axes),
        "target_exposure_band_count": len(targets),
        "curves": curves,
        "pairings": pairings,
        "new_independent_test_evidence": False,
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
    score_root.mkdir(parents=True, exist_ok=True)
    # JSON cannot represent legitimate -inf C0 log scores: write explicit null plus status.
    for curve in curves:
        value = curve["event_mean_log_density"]
        if value is not None and not math.isfinite(value):
            curve["event_mean_log_density"] = None
            curve["log_density_status"] = "negative_infinity_from_saved_C0_zero_mass"
        else:
            curve["log_density_status"] = "finite" if value is not None else "no_events"
    _write_json(score_root / "summary.json", summary)
    exposure_frame.to_parquet(score_root / "exposure_results.parquet", index=False)
    event_frame.to_parquet(score_root / "event_results.parquet", index=False)
    paired.to_parquet(score_root / "paired_anchor_results.parquet", index=False)
    pd.DataFrame(alarms).to_parquet(score_root / "alarm_prefixes.parquet", index=False)
    lon, lat = Transformer.from_crs(EQUAL_AREA_CRS, "EPSG:4326", always_xy=True).transform(
        grid.query_xy_km[:, 0] * 1000, grid.query_xy_km[:, 1] * 1000
    )
    pd.DataFrame(
        {
            "cell_index": np.arange(grid.cell_count),
            "cell_id": grid.cell_ids,
            "longitude": lon,
            "latitude": lat,
            "area_km2": grid.clipped_area_km2,
            "clipped_geometry_wkt_equal_area_m": [
                geometry.wkt for geometry in domain.locator.clipped_geometries
            ],
        }
    ).to_parquet(score_root / "grid_geometry.parquet", index=False)
    artifact_names = [
        "summary.json",
        "exposure_results.parquet",
        "event_results.parquet",
        "paired_anchor_results.parquet",
        "alarm_prefixes.parquet",
        "grid_geometry.parquet",
    ]
    _write_json(
        manifest_path,
        {
            "identity": identity,
            "artifacts": [
                {"path": name, "sha256": _sha(score_root / name)} for name in artifact_names
            ],
        },
    )
    return score_root / "summary.json"
