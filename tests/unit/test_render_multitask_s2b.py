# ruff: noqa: E501, RUF001
"""Synthetic-only S2-B scientific figures and offline controls."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from pyproj import Transformer
from shapely.geometry import box

from seismoflux.background.grid import EQUAL_AREA_CRS
from seismoflux.multitask_s1.c2b_score import score_exposure, summarize
from seismoflux.stage2s.contracts import SpatialGrid


@pytest.fixture(scope="module")
def renderer():
    path = Path(__file__).resolve().parents[2] / "scripts/render_multitask_s2b.py"
    spec = importlib.util.spec_from_file_location("s2b_renderer_synthetic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def fixture_data(renderer):
    models = list(renderer.MODEL_LABELS)
    horizons, bands = [7, 30, 90, 180, 365], ["M5_6", "M6_plus"]
    transformer = Transformer.from_crs("EPSG:4326", EQUAL_AREA_CRS, always_xy=True)
    lon, lat = [98.0, 102.0, 106.0, 110.0], [30.0] * 4
    x, y = transformer.transform(lon, lat)
    grid = SpatialGrid(
        grid_id="synthetic",
        cell_size_km=25.0,
        cell_ids=("a", "b", "c", "d"),
        rows=np.zeros(4, dtype=np.int64),
        columns=np.arange(4, dtype=np.int64),
        query_xy_km=np.column_stack([x, y]) / 1000,
        clipped_area_km2=np.array([200000.0, 250000.0, 300000.0, 300000.0]),
    )
    exposures, events, alarms = [], [], []
    for horizon in horizons:
        for band in bands:
            for period in range(3):
                cell = 2 if period == 0 else 0
                target = {
                    "event_ids": [f"synthetic-</script>-{period}"],
                    "episode_ids": [f"episode-{period}"],
                    "event_cell_indices": [cell],
                    "global_episode_member_counts": [1],
                    "is_episode_anchor": [True],
                    "event_longitudes": [lon[cell] + 0.125],
                    "event_latitudes": [30.125],
                }
                if period == 2:
                    target = {key: [] for key in target}
                for model in models:
                    mass = [0.7, 0.2, 0.05, 0.05]
                    if model.startswith("S2B_"):
                        mass = [0.05, 0.05, 0.7, 0.2]
                    result = score_exposure(
                        log_mass=np.log(mass),
                        grid=grid,
                        target=target,
                        fold_id="synthetic-fold",
                        horizon_days=horizon,
                        issue_time_us=946742400000000 + period * 366 * 86400000000,
                        magnitude_bin=band,
                        model_id=model,
                        budgets=list(renderer.AREA_BUDGETS),
                        near_cells=[{0, 2}] if period != 2 else [],
                    )
                    exposures.extend(result[0])
                    events.extend(result[1])
                    alarms.extend(result[2])
    exposure_frame, event_frame = pd.DataFrame(exposures), pd.DataFrame(events)
    protocol_path = Path(__file__).resolve().parents[2] / "configs/multitask_s2_b_slip_rate.yaml"
    planned = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))["planned_pairs"]
    curves, pairs, details = summarize(exposure_frame, event_frame, planned, ())
    for row in curves:
        row["log_density_status"] = (
            "finite" if row["event_mean_log_density"] is not None else "no_events"
        )
    summary = {
        "synthetic_fixture": True,
        "model_ids": models,
        "horizons_days": horizons,
        "magnitude_bins": bands,
        "curves": curves,
        "pairings": pairs,
        "primary_issue_horizon_count": 15,
        "target_exposure_band_count": 30,
        "holdout_read": False,
        "locked_test_run": False,
        "new_independent_test_evidence": False,
        "scientific_role": "current_static_slip_rates_retrospective_development_not_historical_prospective",
    }
    geometry = pd.DataFrame(
        {
            "cell_index": range(4),
            "cell_id": grid.cell_ids,
            "longitude": lon,
            "latitude": lat,
            "area_km2": grid.clipped_area_km2,
            "clipped_geometry_wkt_equal_area_m": [
                box(xx - 12500, yy - 12500, xx + 12500, yy + 12500).wkt
                for xx, yy in zip(x, y, strict=True)
            ],
        }
    )
    return summary, event_frame, pd.DataFrame(alarms), geometry, exposure_frame, details


def _write_fixture(root, data, renderer):
    summary, events, alarms, grid, exposures, details = data
    score = root / "score_phase"
    score.mkdir(parents=True)
    (score / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    for name, frame in (
        ("event_results", events),
        ("alarm_prefixes", alarms),
        ("grid_geometry", grid),
        ("exposure_results", exposures),
        ("paired_anchor_results", details),
    ):
        frame.to_parquet(score / f"{name}.parquet", index=False)
    manifest = {
        "artifacts": [
            {"path": name, "sha256": renderer._sha256(score / name)}
            for name in renderer.REQUIRED_ARTIFACTS
        ]
    }
    (score / "score_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return score


def test_all_axes_and_empty_periods_preserved_with_s2_labels(renderer, fixture_data):
    data = renderer._replay_data(*fixture_data[:5])
    assert len(data["issues"]) == 30
    assert sum(not issue["events"] for issue in data["issues"]) == 10
    assert all(len(issue["forecasts"]) == 14 for issue in data["issues"])
    assert data["default_reference"] == "C2B_D0_MULTISCALE"
    assert set(data["model_labels"]) == set(renderer.MODEL_LABELS)
    assert all("D0｜" not in label for label in data["model_labels"].values())
    first = data["issues"][0]["events"][0]
    assert first["lon"] == 106.125
    assert first["lon"] != data["grid"][2]["lon"]


def test_case_choice_uses_only_five_mixtures_and_real_catalog_reference(renderer, fixture_data):
    data = renderer._replay_data(*fixture_data[:5])
    selection = renderer._select_cases(data)
    assert selection == renderer._select_cases(data)
    assert selection["candidate_model_id"] == "S2B_COMMON_UNIT_CATALOG_MIX"
    assert selection["reference_model_id"] == renderer.REFERENCE
    gain, failure = selection["cases"]
    assert gain["net_hits"] == 1 and failure["net_hits"] == -1
    assert gain["issue_index"] != failure["issue_index"]
    assert selection["scope"].startswith("development_illustration")
    # Even a maximal pure-fault summary score cannot enter the display-candidate pool.
    altered = deepcopy(data)
    for row in altered["summary"]["curves"]:
        if row["model_id"].endswith("_ONLY"):
            row["anchor_recall"] = 1.0
    assert renderer._select_cases(altered)["candidate_model_id"] == selection["candidate_model_id"]


def test_no_gain_or_failure_is_not_fabricated(renderer, fixture_data):
    data = renderer._replay_data(*fixture_data[:5])
    for issue in data["issues"]:
        for event in issue["events"]:
            for model in data["summary"]["model_ids"]:
                event["hits"][model]["0"] = 0
    selection = renderer._select_cases(data)
    assert all(case["net_hits"] == 0 for case in selection["cases"])
    assert all("共同漏报" in case["label"] for case in selection["cases"])
    assert selection["cases"][0]["issue_index"] != selection["cases"][1]["issue_index"]


def test_render_local_boundary_and_snapshot_caveat(renderer, fixture_data, tmp_path):
    _write_fixture(tmp_path, fixture_data, renderer)
    page = renderer.render(tmp_path)
    text = page.read_text(encoding="utf-8")
    assert "S2-B：断层活动速率" in text and "2026 年收集快照" in text
    assert "不代表当时可发布的预测" in text and "相对" in text
    assert "缺测不是零" in text and "共同 385" in text and "完整 515" in text
    assert "同覆盖" in text and "地震矩率" in text
    assert "S2-A" not in text and "粗化" not in text and "四个混合" not in text
    assert "D1：" not in text and "D2：" not in text and "C2B：" not in text
    assert "fetch(" not in text and "<script src=" not in text
    assert not re.search(r'(?:src|href)=["\']https?://', text)
    assert "synthetic-</script>-0" not in text
    payload = re.search(r'<script id="replay-data" type="application/json">(.*?)</script>', text)
    data = json.loads(payload.group(1))
    assert data["default_candidate"] == "S2B_COMMON_UNIT_CATALOG_MIX"
    assert data["default_reference"] == renderer.REFERENCE
    manifest = json.loads((page.parent / "render_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["artifacts"]) == 9
    assert sum(a["audience"].startswith("public_") for a in manifest["artifacts"]) == 6
    assert all(
        a["audience"].startswith("local_")
        for a in manifest["artifacts"]
        if a["path"].startswith(renderer.CASE_STEM) or a["path"].endswith(".html")
    )
    assert all((page.parent / name).stat().st_size > 1000 for name in renderer.FILENAMES)
    for stem in renderer.STATIC_STEMS:
        svg = (page.parent / f"{stem}.svg").read_text(encoding="utf-8")
        assert "<text" in svg and "2026" in svg
        assert "synthetic-" not in svg


def test_changed_hash_and_snapshot_role_are_refused(renderer, fixture_data, tmp_path):
    score = _write_fixture(tmp_path, fixture_data, renderer)
    summary = json.loads((score / "summary.json").read_text(encoding="utf-8"))
    summary["scientific_role"] = "prospective"
    (score / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="hash changed"):
        renderer._load(tmp_path)
    manifest = json.loads((score / "score_manifest.json").read_text(encoding="utf-8"))
    manifest["artifacts"][0]["sha256"] = renderer._sha256(score / "summary.json")
    (score / "score_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="snapshot role"):
        renderer._load(tmp_path)


def test_offline_script_all_axes_defaults_and_empty_replay(renderer, fixture_data):
    node = os.environ.get("SEISMOFLUX_TEST_NODE") or shutil.which("node")
    if node is None:
        pytest.skip("Node unavailable for offline DOM/canvas synthetic check")
    data = renderer._replay_data(*fixture_data[:5])
    data["default_candidate"] = renderer._select_cases(data)["candidate_model_id"]
    page = renderer.HTML_TEMPLATE.replace("__DATA__", renderer._json_for_script(data))
    javascript = r"""
const fs=require("fs"),vm=require("vm"),html=fs.readFileSync(0,"utf8");
const scripts=[...html.matchAll(/<script(?: [^>]*)?>([\s\S]*?)<\/script>/g)];
class Element {
 constructor(){this.value="0";this.options=[];this.children=[];this.style={};this.clientWidth=600;this.clientHeight=430;}
 set textContent(s){this.text=s;this.children=[];this.options=[];} get textContent(){return this.text||"";}
 add(x){this.options.push(x);} addEventListener(){} insertRow(){const e=new Element;this.children.push(e);return e;}
 insertCell(){const e=new Element;this.children.push(e);this.lastChild=e;return e;}
 getContext(){return new Proxy({},{get:()=>()=>{}});}
}
const elements=new Map,document={getElementById(id){if(!elements.has(id))elements.set(id,new Element);return elements.get(id);}};
document.getElementById("replay-data").textContent=scripts[0][1];document.getElementById("view").value="anchor";
const context=vm.createContext({document,console,window:{devicePixelRatio:1,addEventListener(){}},Option:function(label,value){this.label=label;this.value=value;}});
vm.runInContext(scripts[1][1],context);
vm.runInContext(`
if(reference.value!=="C2B_D0_MULTISCALE"||candidate.value!=="S2B_COMMON_UNIT_CATALOG_MIX"||tolerance.value!=="0")throw Error("wrong S2 defaults");
let count=0;
for(const h of S.horizons_days)for(const b of S.magnitude_bins){horizon.value=String(h);band.value=b;rebuildDates();
for(const m of S.model_ids)for(const a of D.area_budgets)for(const t of D.tolerances){candidate.value=m;area.value=String(a);tolerance.value=String(t);update();count++;}
slider.value="2";drawIssue();if(visibleIssues[2].event_count!==0)throw Error("empty period lost");}
console.log(count,"axes passed");`,context);
"""
    result = subprocess.run(
        [node, "-e", javascript],
        input=page,
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "1400 axes passed" in result.stdout


def test_renderer_preserves_frozen_models_and_rate_geometry_coverage_pair_axes(
    renderer, fixture_data
):
    path = Path(__file__).resolve().parents[2] / "configs/multitask_s2_b_slip_rate.yaml"
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert set(renderer.MODEL_LABELS) == set(protocol["models"]) | set(
        protocol["evaluation"]["references"]
    )
    assert len(renderer.MODEL_LABELS) == 14
    assert len(renderer.MIX_MODELS) == 5
    assert tuple(
        model for model in protocol["models"] if model.endswith("_CATALOG_MIX")
    ) == renderer.MIX_MODELS
    required = {tuple(protocol["planned_pairs"][index][:2]) for index in (0, 1, 2, 8, 9, 10)}
    assert {(a, b) for a, b, _ in renderer.PAIR_AXES} == required
    assert len(fixture_data[0]["curves"]) == 1400
    assert len(fixture_data[0]["pairings"]) == 1400


def test_replay_keeps_every_saved_hit_and_true_event_coordinate(renderer, fixture_data):
    data = renderer._replay_data(*fixture_data[:5])
    issue_lookup = {
        (issue["fold"], issue["horizon_days"], issue["magnitude_bin"], issue["issue_us"]): issue
        for issue in data["issues"]
    }
    checked = 0
    for row in fixture_data[1].itertuples(index=False):
        issue = issue_lookup[(row.fold_id, row.horizon_days, row.magnitude_bin, row.issue_time_us)]
        event = next(event for event in issue["events"] if event["id"] == row.event_id)
        bit = 1 << renderer.AREA_BUDGETS.index(row.area_budget_km2)
        assert bool(event["hits"][row.model_id][str(int(row.hit_tolerance_km))] & bit) == bool(
            row.hit
        )
        assert event["lon"] == row.longitude and event["lat"] == row.latitude
        checked += 1
    assert checked == len(fixture_data[1])
    assert checked > 0


def test_no_main_anchor_cases_are_not_invented(renderer, fixture_data):
    data = renderer._replay_data(*fixture_data[:5])
    for issue in data["issues"]:
        issue["events"] = []
    selection = renderer._select_cases(data)
    assert selection["status"] == "no_main_anchor_events"
    assert selection["cases"] == []
    assert "five_frozen_mixtures" in selection["candidate_selection"]


def test_secondary_tolerance_cannot_disappear_from_completed_replay(
    renderer, fixture_data, tmp_path
):
    altered = list(deepcopy(fixture_data))
    altered[0]["curves"] = [row for row in altered[0]["curves"] if row["hit_tolerance_km"] == 0]
    _write_fixture(tmp_path, altered, renderer)
    with pytest.raises(ValueError, match="both hit tolerances"):
        renderer._load(tmp_path)
