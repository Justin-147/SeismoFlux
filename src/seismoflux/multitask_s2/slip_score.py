"""S2-B scoring: ten saved slip-rate models and four unchanged catalogue references."""

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
from shapely.strtree import STRtree

from seismoflux.background.grid import EQUAL_AREA_CRS
from seismoflux.d1_replay.spatial import build_d1_spatial_domain_from_bytes
from seismoflux.multitask_s1 import c2b_score
from seismoflux.multitask_s1.c2b_score import (
    BANDS,
    PARENT_ARTIFACT_CONFIG,
    _sha,
    _verified,
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
    _prediction_axes,
    _reference_paths,
    _score_fold,
    _validate_references,
    _write_json_once,
    _write_or_compare_table,
)
from seismoflux.multitask_s2.slip_predict import _output_root


def _summarize(
    exposures: pd.DataFrame,
    events: pd.DataFrame,
    protocol: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], pd.DataFrame]:
    planned = protocol["planned_pairs"]
    if len(planned) != 14 or len({(pair[0], pair[1]) for pair in planned}) != 14:
        raise ValueError("S2B requires exactly the fourteen registered comparisons")
    return c2b_score.summarize(exposures, events, planned, new_models=())


def _run_score_phase_locked(
    *,
    project_root: Path,
    data_root: Path,
    output_root: Path | None = None,
) -> Path:
    from seismoflux.multitask_s2.slip_predict import (
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
        raise ValueError("all four S2B prediction folds must be complete before target access")
    new = {record["fold_id"]: load_fold_arrays(root, record) for record in manifest["folds"]}
    axes = _prediction_axes(manifest, new, protocol, tuple(MODEL_IDS), tuple(HORIZONS))
    # No old evaluation/target parquet has been read above this line.
    paths, reference_manifest = _reference_paths(project, protocol)
    parents = yaml.safe_load((project / PARENT_ARTIFACT_CONFIG).read_text("utf-8"))[
        "parent_artifacts"
    ]
    raw_path = _verified(project, parents["C0_raw_scores_score_phase_only"])
    if reference_manifest["identity"]["C0_raw_scores_sha256"] != _sha(raw_path):
        raise ValueError("S2B and the saved C2B reference use different C0 targets")
    identity = {
        "protocol_sha256": _sha(project / PROTOCOL_PATH),
        "prediction_manifest_sha256": _sha(root / "prediction_manifest.json"),
        "catalog_score_manifest_sha256": protocol["inputs"]["catalog_score_manifest_sha256"],
        "C0_raw_scores_sha256": _sha(raw_path),
        "implementation_hashes": {
            "slip_score.py": _sha(Path(__file__)),
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
        raise ValueError("S2B evaluation grid differs from the saved reference grid")
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
        "planned_pair_count": 14,
        "reference_scores_reused_without_recalculation": True,
        "static_snapshot_available_at": protocol["inputs"]["static_snapshot_available_at"],
        "static_snapshot_role": (
            "2026_slip_rate_retrospective_spatial_information_not_known_at_old_issue_dates"
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
    from seismoflux.multitask_s2.slip_predict import load_protocol

    if any(
        os.environ.get(name) != "1"
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    ):
        raise ValueError("S2B score requires numerical-library thread limits of one")
    project = project_root.resolve()
    protocol = load_protocol(project)
    root = _output_root(project, protocol, output_root)
    root.mkdir(parents=True, exist_ok=True)
    with _run_lock(root):
        return _run_score_phase_locked(project_root=project, data_root=data_root, output_root=root)
