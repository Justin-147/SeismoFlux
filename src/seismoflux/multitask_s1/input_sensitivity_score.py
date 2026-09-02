"""Paired location-only scoring of the frozen S1-C2A input sensitivity.

No target is opened until all six predictions in all four folds are verified.
Targets are reused from C0, never rebuilt or selected from a newer catalog.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from seismoflux.d1_replay.spatial import (
    FROZEN_D1_AREA_BUDGETS_KM2,
    build_d1_spatial_domain_from_bytes,
    select_alarm_prefixes,
)
from seismoflux.multitask_s1.development_contract import DEVELOPMENT_FOLD_IDS
from seismoflux.multitask_s1.metrics import score_location_events
from seismoflux.stage2s.contracts import SpatialGrid

PROTOCOL_PATH = Path("configs/multitask_s1_c2a_input_sensitivity.yaml")
BASE_MODELS = ("L1_REGIONAL_CONSTANT", "L2_KDE_CAUSAL", "L3_B0_R30_CAUSAL")
NEW_MODELS = tuple(f"{treatment}_{model}" for treatment in ("A", "B") for model in BASE_MODELS)
ALL_MODELS = tuple(f"C0_{model}" for model in BASE_MODELS) + NEW_MODELS
TARGET_FIELDS = (
    "event_ids",
    "event_cell_indices",
    "episode_ids",
    "global_episode_member_counts",
    "is_episode_anchor",
    "event_longitudes",
    "event_latitudes",
)


class InputSensitivityScoreError(ValueError):
    """A changed prediction, target, or area identity cannot be scored."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bound_path(root: Path, relative_path: str) -> Path:
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root.resolve()):
        raise InputSensitivityScoreError("artifact path leaves its declared root")
    return path


def _verified_path(root: Path, reference: dict[str, Any], *, key: str = "path") -> Path:
    path = _bound_path(root, str(reference[key]))
    if _sha256(path) != reference["sha256"]:
        raise InputSensitivityScoreError(f"input SHA-256 changed: {path}")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _issue_us(value: object) -> int:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise InputSensitivityScoreError("target issue must be timezone aware")
    return int(stamp.tz_convert("UTC").as_unit("us").value)


def validate_target_rows(
    rows: pd.DataFrame,
    *,
    expected_issues: dict[str, tuple[int, ...]],
    cell_count: int,
    expected_anchor_count: int,
) -> dict[tuple[str, int], dict[str, Any]]:
    """Require identical C0 target vectors, including every empty exposure."""

    if tuple(expected_issues) != DEVELOPMENT_FOLD_IDS:
        raise InputSensitivityScoreError("only the four development folds may be scored")
    expected_keys = {(fold, issue) for fold, issues in expected_issues.items() for issue in issues}
    by_model: dict[str, dict[tuple[str, int], dict[str, Any]]] = {m: {} for m in BASE_MODELS}
    for row in rows.to_dict("records"):
        if row["score_family"] != "location" or int(row["horizon_days"]) != 30:
            raise InputSensitivityScoreError("target input must be location-only and 30 days")
        model = row["model_id"]
        if model not in by_model:
            raise InputSensitivityScoreError("unexpected C0 reference model")
        payload = json.loads(row["payload_json"])
        if payload.get("metric") != "spatial_log_density" or payload.get("magnitude_bin") != "M5_6":
            continue
        key = (row["fold_id"], _issue_us(row["issue_time_utc"]))
        if key not in expected_keys or key in by_model[model]:
            raise InputSensitivityScoreError("extra, duplicate, or non-development target issue")
        if (
            payload.get("fold_id") != key[0]
            or _issue_us(payload["issue_time_utc"]) != key[1]
            or payload.get("model_id") != model
            or payload.get("horizon_days") != 30
            or payload.get("catalog_delay_hours") != 24
            or payload.get("hit_tolerance_km") != 0.0
        ):
            raise InputSensitivityScoreError("C0 target identity or causal boundary changed")
        targets = {field: payload[field] for field in TARGET_FIELDS}
        lengths = {len(values) for values in targets.values()}
        if len(lengths) != 1 or len(targets["event_ids"]) != payload["event_count"]:
            raise InputSensitivityScoreError("target vectors have unequal lengths")
        if any(type(v) is not bool for v in targets["is_episode_anchor"]):
            raise InputSensitivityScoreError("anchor flags must be explicit booleans")
        if any(
            type(v) is not int or not 0 <= v < cell_count for v in targets["event_cell_indices"]
        ):
            raise InputSensitivityScoreError("target event is outside the unchanged grid")
        by_model[model][key] = targets
    if any(set(targets) != expected_keys for targets in by_model.values()):
        raise InputSensitivityScoreError("missing C0 target exposures, including empty exposures")
    reference = by_model[BASE_MODELS[0]]
    if any(targets != reference for targets in by_model.values()):
        raise InputSensitivityScoreError("C0 L1/L2/L3 target keys or metadata differ")
    event_ids: list[str] = []
    anchor_episodes: list[str] = []
    episode_counts: dict[str, int] = {}
    for targets in reference.values():
        event_ids.extend(targets["event_ids"])
        for episode, count, anchor in zip(
            targets["episode_ids"],
            targets["global_episode_member_counts"],
            targets["is_episode_anchor"],
            strict=True,
        ):
            if (
                type(count) is not int
                or count < 1
                or episode_counts.setdefault(episode, count) != count
            ):
                raise InputSensitivityScoreError("inconsistent full-catalog episode size")
            if anchor:
                anchor_episodes.append(episode)
    if len(event_ids) != len(set(event_ids)):
        raise InputSensitivityScoreError("physical event repeats across primary exposures")
    if len(anchor_episodes) != expected_anchor_count or len(set(anchor_episodes)) != len(
        anchor_episodes
    ):
        raise InputSensitivityScoreError("fixed first-anchor episode population changed")
    return reference


def _validate_mass(mass: np.ndarray, shape: tuple[int, ...]) -> None:
    if (
        mass.shape != shape
        or not np.isfinite(mass).all()
        or np.any(mass < 0)
        or not np.allclose(mass.sum(axis=-1), 1.0, rtol=0.0, atol=1e-10)
    ):
        raise InputSensitivityScoreError(
            "prediction mass is nonfinite, unnormalized, or wrong shape"
        )


def _load_predictions(
    project: Path,
    root: Path,
    protocol: dict[str, Any],
    manifest: dict[str, Any],
) -> tuple[dict[str, tuple[int, ...]], dict[str, np.ndarray]]:
    design = protocol["development_design"]
    if (
        manifest.get("schema_version") != 1
        or manifest.get("protocol_id") != protocol["protocol_id"]
        or manifest.get("run_id") != protocol["run_id"]
        or manifest.get("protocol_sha256") != _sha256(project / PROTOCOL_PATH)
        or manifest.get("grid_id") != design["grid_id"]
        or tuple(manifest.get("model_ids", [])) != NEW_MODELS
        or tuple(f["fold_id"] for f in manifest["folds"]) != DEVELOPMENT_FOLD_IDS
    ):
        raise InputSensitivityScoreError("C2A manifest does not describe the frozen four-fold run")
    seal_path = _verified_path(
        project, protocol["parent_artifacts"]["C0_four_fold_prediction_seal"]
    )
    seal = _read_json(seal_path)
    c0_folds = seal["ordered_fold_predictions"]
    if tuple(f["fold_id"] for f in c0_folds) != DEVELOPMENT_FOLD_IDS:
        raise InputSensitivityScoreError("C0 seal changed development folds")
    issues: dict[str, tuple[int, ...]] = {}
    arrays: dict[str, np.ndarray] = {}
    for reference, c0_reference in zip(manifest["folds"], c0_folds, strict=True):
        fold = reference["fold_id"]
        path = _verified_path(
            root, {"path": reference["npz_path"], "sha256": reference["npz_sha256"]}
        )
        with np.load(path, allow_pickle=False) as payload:
            issue_axis = np.asarray(payload["issue_time_us"], dtype=np.int64)
            mass = np.asarray(payload["location_relative_mass"], dtype=np.float64)
        count = design["primary_exposure_count_per_fold"]
        if (
            issue_axis.shape != (count,)
            or len(set(issue_axis.tolist())) != count
            or np.any(np.diff(issue_axis) <= 0)
        ):
            raise InputSensitivityScoreError("C2A primary issue axis is incomplete or duplicated")
        _validate_mass(mass, (count, len(NEW_MODELS), design["grid_cell_count"]))
        bundle_path = _verified_path(seal_path.parent, c0_reference, key="relative_path")
        bundle = _read_json(bundle_path)
        artifacts = bundle["prediction_artifacts"]
        if bundle["fold_id"] != fold or len(artifacts) != 1:
            raise InputSensitivityScoreError("C0 fold prediction identity changed")
        c0_path = _verified_path(seal_path.parent, artifacts[0], key="relative_path")
        with np.load(c0_path, allow_pickle=False) as c0:
            selected = np.asarray(c0["primary_horizon_days"]) == 30
            c0_issues = np.asarray(c0["primary_issue_time_us"])[selected]
            c0_mass = np.asarray(c0["location_relative_mass"])[selected][:, [1, 2, 4], :]
        if not np.array_equal(issue_axis, c0_issues):
            raise InputSensitivityScoreError("C2A and C0 issue axes differ")
        _validate_mass(c0_mass, (count, len(BASE_MODELS), design["grid_cell_count"]))
        issues[fold] = tuple(int(v) for v in issue_axis)
        arrays[fold] = np.concatenate((c0_mass, mass), axis=1)
    return issues, arrays


def score_exposure(
    mass: np.ndarray,
    grid: SpatialGrid,
    targets: dict[str, Any],
    *,
    fold_id: str,
    issue_time_us: int,
    model_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Use the existing metric and exact same complete-cell area prefixes."""

    evaluation = score_location_events(
        mass,
        grid,
        event_ids=targets["event_ids"],
        event_cell_indices=targets["event_cell_indices"],
        episode_ids=targets["episode_ids"],
        episode_member_counts=targets["global_episode_member_counts"],
        is_episode_anchor=targets["is_episode_anchor"],
    )
    identity = {
        "fold_id": fold_id,
        "issue_time_us": issue_time_us,
        "issue_time_utc": pd.Timestamp(issue_time_us, unit="us", tz="UTC").isoformat(),
        "model_id": model_id,
    }
    exposure_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    alarms: list[dict[str, Any]] = []
    for prefix in select_alarm_prefixes(mass, grid):
        selected = set(int(v) for v in prefix.selected_indices)
        scores = {
            score.basis: score
            for score in evaluation.alarm_recall
            if score.area_budget_km2 == prefix.budget_km2
        }
        row = {
            **identity,
            "area_budget_km2": prefix.budget_km2,
            "actual_area_km2": prefix.actual_area_km2,
            "event_count": len(targets["event_ids"]),
        }
        for basis in ("anchor", "all", "episode_balanced"):
            row[f"{basis}_hits"] = scores[basis].hit_weight
            row[f"{basis}_total"] = scores[basis].total_weight
            row[f"{basis}_recall"] = scores[basis].recall
        exposure_rows.append(row)
        alarms.append(
            {
                **identity,
                "area_budget_km2": prefix.budget_km2,
                "actual_area_km2": prefix.actual_area_km2,
                "selected_cell_indices": prefix.selected_indices.tolist(),
            }
        )
        for index, event in enumerate(targets["event_ids"]):
            event_rows.append(
                {
                    **identity,
                    "area_budget_km2": prefix.budget_km2,
                    "actual_area_km2": prefix.actual_area_km2,
                    "event_id": event,
                    "episode_id": targets["episode_ids"][index],
                    "is_episode_anchor": targets["is_episode_anchor"][index],
                    "global_episode_member_count": targets["global_episode_member_counts"][index],
                    "cell_index": targets["event_cell_indices"][index],
                    "longitude": targets["event_longitudes"][index],
                    "latitude": targets["event_latitudes"][index],
                    "hit": targets["event_cell_indices"][index] in selected,
                }
            )
    return exposure_rows, event_rows, alarms


def summarize_results(
    exposure_rows: pd.DataFrame,
    event_rows: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], pd.DataFrame]:
    """Effect sizes, fold directions and a paired 2,000 episode bootstrap."""

    curves: list[dict[str, Any]] = []
    for (model, area), frame in exposure_rows.groupby(["model_id", "area_budget_km2"], sort=True):
        hits, total = float(frame.anchor_hits.sum()), float(frame.anchor_total.sum())
        fold_rows = []
        for fold, subset in frame.groupby("fold_id", sort=True):
            fold_hits, fold_total = (
                float(subset.anchor_hits.sum()),
                float(subset.anchor_total.sum()),
            )
            fold_rows.append(
                {
                    "fold_id": fold,
                    "anchor_hits": int(fold_hits),
                    "anchor_total": int(fold_total),
                    "anchor_recall": fold_hits / fold_total if fold_total else None,
                }
            )
        curves.append(
            {
                "model_id": model,
                "area_budget_km2": float(area),
                "anchor_hits": int(hits),
                "anchor_total": int(total),
                "anchor_recall": hits / total if total else None,
                "actual_area_mean_km2": float(frame.actual_area_km2.mean()),
                "actual_area_min_km2": float(frame.actual_area_km2.min()),
                "actual_area_max_km2": float(frame.actual_area_km2.max()),
                "all_event_hits": int(frame.all_hits.sum()),
                "all_event_total": int(frame.all_total.sum()),
                "per_fold": fold_rows,
            }
        )
    pairing_models = [(f"{t}_{m}", f"C0_{m}") for t in ("A", "B") for m in BASE_MODELS]
    pairing_models += [
        (f"{t}_{BASE_MODELS[c]}", f"{t}_{BASE_MODELS[r]}")
        for t in ("A", "B")
        for c, r in ((1, 0), (2, 0), (2, 1))
    ]
    anchors = event_rows.loc[event_rows.is_episode_anchor].copy()
    summaries: list[dict[str, Any]] = []
    paired_frames: list[pd.DataFrame] = []
    keys = ["fold_id", "issue_time_us", "episode_id", "event_id", "area_budget_km2"]
    for candidate, reference in pairing_models:
        left = anchors.loc[anchors.model_id == candidate, [*keys, "hit", "actual_area_km2"]]
        right = anchors.loc[anchors.model_id == reference, [*keys, "hit", "actual_area_km2"]]
        paired = left.merge(
            right,
            on=keys,
            how="outer",
            validate="one_to_one",
            indicator=True,
            suffixes=("_candidate", "_reference"),
        )
        if not (paired._merge == "both").all():
            raise InputSensitivityScoreError(
                "paired models have different physical anchor populations"
            )
        paired = paired.drop(columns="_merge").sort_values(keys).reset_index(drop=True)
        paired["candidate_model_id"], paired["reference_model_id"] = candidate, reference
        paired["net_hit"] = paired.hit_candidate.astype(int) - paired.hit_reference.astype(int)
        paired["outcome"] = np.select(
            [paired.net_hit == 1, paired.net_hit == -1, paired.hit_candidate],
            ["gained", "lost", "unchanged_hit"],
            default="unchanged_miss",
        )
        paired_frames.append(paired)
        for area, frame in paired.groupby("area_budget_km2", sort=True):
            delta = frame.net_hit.to_numpy(dtype=np.int8)
            rng = np.random.default_rng(147)
            draws = delta[rng.integers(0, len(delta), size=(2000, len(delta)))].mean(axis=1) * 100.0
            fold_rows = []
            for fold in DEVELOPMENT_FOLD_IDS:
                subset = frame.loc[frame.fold_id == fold]
                net = int(subset.net_hit.sum())
                fold_rows.append(
                    {
                        "fold_id": fold,
                        "net_hits": net,
                        "anchor_total": len(subset),
                        "delta_recall_pp": float(subset.net_hit.mean() * 100.0)
                        if len(subset)
                        else None,
                        "direction": (
                            "not_evaluable"
                            if not len(subset)
                            else "positive"
                            if net > 0
                            else "negative"
                            if net < 0
                            else "zero"
                        ),
                    }
                )
            summaries.append(
                {
                    "candidate_model_id": candidate,
                    "reference_model_id": reference,
                    "area_budget_km2": float(area),
                    "anchor_total": len(frame),
                    **{
                        name: int((frame.outcome == name).sum())
                        for name in ("gained", "lost", "unchanged_hit", "unchanged_miss")
                    },
                    "net_hits": int(delta.sum()),
                    "delta_recall_pp": float(delta.mean() * 100.0),
                    "bootstrap_ci95_pp": np.quantile(draws, [0.025, 0.975]).tolist(),
                    "per_fold": fold_rows,
                }
            )
    return curves, summaries, pd.concat(paired_frames, ignore_index=True)


def _reuse_completed_score(score_root: Path, identity: dict[str, Any]) -> bool:
    if not score_root.exists():
        return False
    manifest_path = score_root / "score_manifest.json"
    if not manifest_path.is_file():
        raise InputSensitivityScoreError("incomplete score output exists; do not overwrite it")
    record = _read_json(manifest_path)
    if record.get("identity") != identity or not record.get("complete"):
        raise InputSensitivityScoreError("existing score identity differs or is incomplete")
    required = {
        "summary.json",
        "exposure_scores.csv",
        "event_results.parquet",
        "paired_episode_results.csv",
        "alarm_prefixes.parquet",
        "grid_cells.csv",
    }
    if {item["path"] for item in record["artifacts"]} != required:
        raise InputSensitivityScoreError("existing score artifact list is incomplete")
    for reference in record["artifacts"]:
        _verified_path(score_root, reference)
    return True


def run_score_phase(
    *,
    project_root: Path,
    data_root: Path,
    output_root: Path | None = None,
) -> Path:
    """Score one complete, unchanged C2A prediction run; never run a locked test."""

    from seismoflux.multitask_s1.input_sensitivity_predict import verify_prediction_manifest

    project, data = project_root.resolve(), data_root.resolve()
    protocol = yaml.safe_load((project / PROTOCOL_PATH).read_text(encoding="utf-8"))
    root = (
        output_root.resolve()
        if output_root is not None
        else project / protocol["execution"]["output_root"]
    )
    # This call must precede all access to the old target-bearing score table.
    manifest = verify_prediction_manifest(project, root)
    issues, masses = _load_predictions(project, root, protocol, manifest)
    raw_reference = protocol["parent_artifacts"]["C0_raw_scores_score_phase_only"]
    raw_path = _verified_path(project, raw_reference)
    identity = {
        "protocol_sha256": _sha256(project / PROTOCOL_PATH),
        "prediction_manifest_sha256": _sha256(root / "prediction_manifest.json"),
        "C0_raw_scores_sha256": raw_reference["sha256"],
    }
    score_root = root / "score_phase"
    if _reuse_completed_score(score_root, identity):
        return score_root / "summary.json"
    c0_contract_path = _verified_path(project, protocol["parent_artifacts"]["C0_run_contract"])
    c0_contract = yaml.safe_load(c0_contract_path.read_text(encoding="utf-8"))
    study_path = _verified_path(data, c0_contract["input_identities"]["study_area"])
    grid = build_d1_spatial_domain_from_bytes(study_path.read_bytes()).operational_grid
    design = protocol["development_design"]
    if (
        grid.grid_id != design["grid_id"]
        or grid.cell_count != design["grid_cell_count"]
        or tuple(design["alarm_area_budgets_km2"]) != FROZEN_D1_AREA_BUDGETS_KM2
    ):
        raise InputSensitivityScoreError("scoring grid or area budgets changed")
    rows = pd.read_parquet(
        raw_path,
        filters=[
            ("score_family", "==", "location"),
            ("horizon_days", "==", 30),
            ("model_id", "in", list(BASE_MODELS)),
        ],
    )
    targets = validate_target_rows(
        rows,
        expected_issues=issues,
        cell_count=grid.cell_count,
        expected_anchor_count=design["fixed_anchor_episode_count"],
    )
    if len(targets) != design["primary_exposure_count_total"]:
        raise InputSensitivityScoreError("total primary exposure count changed")
    exposures: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    alarms: list[dict[str, Any]] = []
    for fold, issue_axis in issues.items():
        for index, issue in enumerate(issue_axis):
            for model_index, model in enumerate(ALL_MODELS):
                exposure, event, alarm = score_exposure(
                    masses[fold][index, model_index],
                    grid,
                    targets[(fold, issue)],
                    fold_id=fold,
                    issue_time_us=issue,
                    model_id=model,
                )
                exposures.extend(exposure)
                events.extend(event)
                alarms.extend(alarm)
    exposure_frame, event_frame = pd.DataFrame(exposures), pd.DataFrame(events)
    curves, pairings, paired = summarize_results(exposure_frame, event_frame)
    summary = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "run_id": protocol["run_id"],
        "scientific_role": protocol["scientific_role"],
        "effective_magnitude_type": "Ms",
        "horizon_days": 30,
        "magnitude_interval": "Ms in [5,6)",
        "main_alarm_area_km2": design["main_alarm_area_km2"],
        "grid_id": grid.grid_id,
        "study_area_km2": design["study_area_km2"],
        "primary_exposure_count": len(targets),
        "fixed_anchor_episode_count": design["fixed_anchor_episode_count"],
        "unique_target_event_count": len(set(event_frame.event_id)),
        "empty_exposure_count": sum(not t["event_ids"] for t in targets.values()),
        "model_ids": list(ALL_MODELS),
        "curves": curves,
        "pairings": pairings,
        "bootstrap": {
            "unit": "paired_fixed_anchor_episode",
            "replicates": 2000,
            "root_seed": 147,
            "interval": "percentile_95",
            "research_stop_gate": False,
        },
        "interpretation": (
            "Development fixed-parameter input sensitivity, not independent confirmation, "
            "optimized capacity or absolute earthquake probability."
        ),
        "area_interpretation": (
            "Same nominal budgets, unchanged full national domain, "
            "model-specific actual whole-cell areas are reported."
        ),
        "holdout_read": False,
        "locked_test_run": False,
    }
    score_root.mkdir(parents=True, exist_ok=False)
    exposure_frame.to_csv(score_root / "exposure_scores.csv", index=False)
    event_frame.to_parquet(score_root / "event_results.parquet", index=False)
    paired.to_csv(score_root / "paired_episode_results.csv", index=False)
    pd.DataFrame(alarms).to_parquet(score_root / "alarm_prefixes.parquet", index=False)
    from pyproj import Transformer

    from seismoflux.background.grid import EQUAL_AREA_CRS

    transformer = Transformer.from_crs(EQUAL_AREA_CRS, "EPSG:4326", always_xy=True)
    longitude, latitude = transformer.transform(
        grid.query_xy_km[:, 0] * 1000.0, grid.query_xy_km[:, 1] * 1000.0
    )
    pd.DataFrame(
        {
            "cell_index": np.arange(grid.cell_count),
            "cell_id": grid.cell_ids,
            "longitude": longitude,
            "latitude": latitude,
            "clipped_area_km2": grid.clipped_area_km2,
        }
    ).to_csv(score_root / "grid_cells.csv", index=False)
    _write_json(score_root / "summary.json", summary)
    artifacts = [
        {"path": path.name, "sha256": _sha256(path)}
        for path in sorted(score_root.iterdir())
        if path.is_file()
    ]
    _write_json(
        score_root / "score_manifest.json",
        {"schema_version": 1, "complete": True, "identity": identity, "artifacts": artifacts},
    )
    return score_root / "summary.json"
