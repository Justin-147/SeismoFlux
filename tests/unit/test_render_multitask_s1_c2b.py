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
from pyproj import Transformer
from shapely.geometry import box

from seismoflux.background.grid import EQUAL_AREA_CRS
from seismoflux.multitask_s1.c2b_score import score_exposure, summarize
from seismoflux.stage2s.contracts import SpatialGrid


@pytest.fixture(scope="module")
def renderer():
    script = Path(__file__).resolve().parents[2] / "scripts/render_multitask_s1_c2b.py"
    spec = importlib.util.spec_from_file_location("c2b_renderer_synthetic_test", script)
    assert spec is not None and spec.loader is not None
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
        grid_id="synthetic_only",
        cell_size_km=25.0,
        cell_ids=("c0", "c1", "c2", "c3"),
        rows=np.zeros(4, dtype=np.int64),
        columns=np.arange(4, dtype=np.int64),
        query_xy_km=np.column_stack([x, y]) / 1000,
        clipped_area_km2=np.asarray([200000.0, 250000.0, 300000.0, 300000.0]),
    )
    # Nonzero, modern microseconds catch a silent ns/us reinterpretation.
    start_us = 946684800000000
    exposures, events, alarms = [], [], []
    for horizon in horizons:
        for band in bands:
            for offset in (0, 366 * 86400000000):
                target = {
                    "event_ids": ["synthetic-</script>-anchor", "synthetic-subsequent"],
                    "episode_ids": ["episode-a", "episode-a"],
                    "event_cell_indices": [2, 3],
                    "global_episode_member_counts": [2, 2],
                    "is_episode_anchor": [True, False],
                    "event_longitudes": [106.125, 110.25],
                    "event_latitudes": [30.125, 30.25],
                }
                if offset or band == "M6_plus":
                    target = {key: [] for key in target}
                for model_index, model in enumerate(models):
                    mass = [0.7, 0.2, 0.05, 0.05]
                    if model_index % 2:
                        mass = [0.05, 0.05, 0.7, 0.2]
                    exposure, event, alarm = score_exposure(
                        log_mass=np.log(mass),
                        grid=grid,
                        target=target,
                        fold_id="synthetic-fold",
                        horizon_days=horizon,
                        issue_time_us=start_us + offset,
                        magnitude_bin=band,
                        model_id=model,
                        budgets=list(renderer.AREA_BUDGETS),
                        near_cells=[{0, 2}, {1, 3}] if target["event_ids"] else [],
                    )
                    exposures.extend(exposure)
                    events.extend(event)
                    alarms.extend(alarm)
    exposure_frame, event_frame = pd.DataFrame(exposures), pd.DataFrame(events)
    planned = [
        ["C2B_D1_K75", "C2B_D0_K75", "panel_overall_difference"],
        ["C2B_D2_K75", "C2B_D0_K75", "panel_overall_difference"],
        ["C2B_D1_R30", "C2B_D0_R30", "panel_recent_difference"],
        ["C2B_D0_R30", "C2B_D0_K75", "fixed_recent30_contribution"],
        ["C2B_D0_MULTISCALE", "C2B_D0_K75", "spatial_scale_contribution"],
        ["C2B_D0_AGE_WEIGHTED", "C2B_D0_K75", "relative_age_contribution"],
        ["C2B_D0_AGE_WEIGHTED", "C2B_D0_R30", "age_vs_recent"],
        ["C2B_D0_RIDGE_CORE", "C2B_D0_K75", "learned_spatial_combination"],
        ["C2B_D0_RIDGE_M5", "C2B_D0_RIDGE_CORE", "added_M5_feature"],
    ]
    curves, pairs, details = summarize(exposure_frame, event_frame, planned, ())
    for row in curves:
        row["log_density_status"] = (
            "finite" if row["event_mean_log_density"] is not None else "no_events"
        )
    curves[0]["event_mean_log_density"] = None
    curves[0]["log_density_status"] = "negative_infinity_from_saved_C0_zero_mass"
    summary = {
        "synthetic_fixture": True,
        "model_ids": models,
        "horizons_days": horizons,
        "magnitude_bins": bands,
        "curves": curves,
        "pairings": pairs,
        "primary_issue_horizon_count": 10,
        "target_exposure_band_count": 20,
        "holdout_read": False,
        "locked_test_run": False,
        "new_independent_test_evidence": False,
        "bootstrap": {"replicates": 2000},
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


def _write_fixture(root, fixture_data, renderer):
    summary, events, alarms, grid, exposures, details = fixture_data
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
        "identity": {"synthetic_fixture": True},
        "artifacts": [
            {"path": name, "sha256": renderer._sha256(score / name)}
            for name in renderer.REQUIRED_ARTIFACTS
        ],
    }
    (score / "score_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return score


def test_replay_keeps_all_axes_empty_exposures_and_saved_tolerant_hits(renderer, fixture_data):
    summary, events, alarms, grid, exposures, _ = fixture_data
    data = renderer._replay_data(summary, events, alarms, grid, exposures)
    assert len(data["issues"]) == 20
    assert {row["horizon_days"] for row in data["issues"]} == {7, 30, 90, 180, 365}
    assert {row["magnitude_bin"] for row in data["issues"]} == {"M5_6", "M6_plus"}
    assert sum(not row["events"] for row in data["issues"]) == 15
    assert all(len(row["forecasts"]) == 14 for row in data["issues"])
    assert all(row["label"].startswith(("2000-", "2001-")) for row in data["issues"])
    assert len(data["plans"]) < len(data["issues"])  # Global prefix interning.
    issue = next(row for row in data["issues"] if row["events"])
    event = issue["events"][0]
    assert event["lon"] == 106.125 and data["grid"][2]["lon"] == 106
    reference = summary["model_ids"][0]
    bit = 1 << renderer.AREA_BUDGETS.index(600000)
    assert event["hits"][reference]["0"] & bit == 0
    assert event["hits"][reference]["70"] & bit != 0
    assert data["grid"][0]["polygons"][0][0][0] != [98.0, 30.0]
    assert all(len(plan["areas"]) == 5 for plan in data["plans"])
    assert any(row["bootstrap_ci95_pp"] is None for row in data["summary"]["pairings"])
    assert {row["net_hits"] for row in data["summary"]["pairings"]} >= {-1, 0, 1}


def test_render_is_self_contained_and_separates_public_from_local(renderer, fixture_data, tmp_path):
    root = tmp_path / "synthetic_only"
    _write_fixture(root, fixture_data, renderer)
    page = renderer.render(root)
    assert page.parent == root / "rendered"
    assert all((page.parent / name).stat().st_size > 1000 for name in renderer.FILENAMES)
    text = page.read_text(encoding="utf-8")
    assert "合成测试示例" in text and "配对非重叠时间曝光重采样" in text
    assert "fetch(" not in text and "<script src=" not in text and "cdn." not in text
    assert not re.search(r'(?:src|href)=["\']https?://', text)
    assert 'tolerance.value="0"' in text
    assert "event.hits[model][tolerance.value]" in text
    assert "selected.has(e.cell)" not in text
    assert "negative_infinity_from_saved_C0_zero_mass" in text
    assert "bootstrap_ci95_pp===null" in text
    assert "synthetic-</script>-anchor" not in text
    payload = re.search(r'<script id="replay-data" type="application/json">(.*?)</script>', text)
    assert payload is not None
    data = json.loads(payload.group(1))
    assert len(data["issues"]) == 20 and len(data["summary"]["model_ids"]) == 14
    assert data["issues"][0]["events"][0]["id"] == "synthetic-</script>-anchor"
    manifest = json.loads((page.parent / "render_manifest.json").read_text(encoding="utf-8"))
    assert manifest["network_resources"] == []
    assert manifest["timestamp_unit"] == "us" and manifest["empty_exposures_retained"]
    assert len([a for a in manifest["artifacts"] if a["audience"].startswith("public_")]) == 6
    assert next(a for a in manifest["artifacts"] if a["path"] == page.name)["audience"].startswith(
        "local_only"
    )
    assert all(
        a["audience"].startswith("local_only")
        for a in manifest["artifacts"]
        if a["path"].startswith(renderer.CASE_STEM)
    )
    assert manifest["case_selection"]["scope"].startswith("development_illustration")


def test_manifest_hash_gate_precedes_parquet_read(renderer, fixture_data, tmp_path, monkeypatch):
    score = _write_fixture(tmp_path, fixture_data, renderer)
    (score / "summary.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **kw: pytest.fail("unverified read"))
    with pytest.raises(ValueError, match="hash changed"):
        renderer._load(tmp_path)


def test_missing_model_and_empty_exposure_are_rejected(renderer, fixture_data):
    summary, events, alarms, grid, exposures, _ = fixture_data
    with pytest.raises(ValueError, match="missing an alarm model"):
        renderer._replay_data(
            summary, events, alarms[alarms.model_id != summary["model_ids"][0]], grid, exposures
        )
    with pytest.raises(ValueError, match="lost exposure axes"):
        renderer._replay_data(
            summary, events, alarms, grid, exposures[exposures.issue_time_us == 946684800000000]
        )
    with pytest.raises(ValueError, match="omits area budgets"):
        renderer._replay_data(summary, events.iloc[1:], alarms, grid, exposures)


def test_curve_axis_and_manifest_path_are_guarded(renderer, fixture_data, tmp_path):
    score = _write_fixture(tmp_path, fixture_data, renderer)
    summary = json.loads((score / "summary.json").read_text(encoding="utf-8"))
    summary["curves"].pop()
    (score / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    manifest = json.loads((score / "score_manifest.json").read_text(encoding="utf-8"))
    manifest["artifacts"][0]["sha256"] = renderer._sha256(score / "summary.json")
    (score / "score_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="curve axis"):
        renderer._load(tmp_path)
    manifest["artifacts"].append({"path": "../outside.json", "sha256": "not-allowed"})
    (score / "score_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact path"):
        renderer._load(tmp_path)


def test_all_models_are_in_scientific_panel_groups(renderer, fixture_data):
    models = fixture_data[0]["model_ids"]
    groups = renderer._panel_groups(models)
    assert {model for _, group in groups for model in group} == set(models)
    assert max(len(group) for _, group in groups) <= 6
    assert len(groups) == 3


def test_case_selection_is_deterministic_and_retains_failure_fallback(renderer, fixture_data):
    summary, events, alarms, grid, exposures, _ = fixture_data
    data = renderer._replay_data(summary, events, alarms, grid, exposures)
    selected = renderer._select_cases(data)
    assert selected == renderer._select_cases(data)
    assert selected["candidate_model_id"] == "C2B_D0_K75"
    assert selected["reference_model_id"] == "C0_L3_B0_R30_CAUSAL"
    assert selected["cases"][0]["gained"] == 1
    assert selected["cases"][1]["lost"] == 0
    assert "无净损失命中期" in selected["cases"][1]["label"]
    assert "独立" not in selected["cases"][0]["label"]
    miss = deepcopy(data)
    for issue in miss["issues"]:
        for event in issue["events"]:
            for model in (selected["candidate_model_id"], selected["reference_model_id"]):
                event["hits"][model]["0"] = 0
    fallback = renderer._select_cases(miss)
    assert all("共同漏报" in case["label"] for case in fallback["cases"])
    assert all(case["common_missed"] == 1 for case in fallback["cases"])


def test_case_selection_prefers_distinct_net_positive_and_negative_over_mixed_early_issue(
    renderer, fixture_data
):
    summary, events, alarms, grid, exposures, _ = fixture_data
    data = renderer._replay_data(summary, events, alarms, grid, exposures)
    selection = renderer._select_cases(data)
    candidate, reference = selection["candidate_model_id"], selection["reference_model_id"]
    base = next(
        issue
        for issue in data["issues"]
        if issue["horizon_days"] == 30 and issue["magnitude_bin"] == "M5_6" and issue["events"]
    )
    bit = 1 << renderer.AREA_BUDGETS.index(600000.0)
    issues = []
    for offset, outcomes in enumerate(
        ([(1, 0), (0, 1)], [(1, 0), (1, 0), (0, 1)], [(1, 0), (0, 1), (0, 1)])
    ):
        issue = deepcopy(base)
        issue["issue_us"] += offset * 86400000000
        issue["events"] = []
        for index, (left, right) in enumerate(outcomes):
            event = deepcopy(base["events"][0])
            event["id"] = f"synthetic-{offset}-{index}"
            event["hits"][candidate]["0"] = bit if left else 0
            event["hits"][reference]["0"] = bit if right else 0
            issue["events"].append(event)
        issues.append(issue)
    data["issues"] = issues
    result = renderer._select_cases(data)
    gain, failure = result["cases"]
    assert gain["issue_index"] == 1 and failure["issue_index"] == 2
    assert gain["net_hits"] == 1 and failure["net_hits"] == -1
    assert gain["highlight"] == "gained" and failure["highlight"] == "lost"
    assert result["display_rule_version"] == 2
    assert result == renderer._select_cases(data)

    # If no net-negative issue remains, a distinct mixed period is preferable
    # to repeating the already displayed net-positive period that also has loss.
    data["issues"] = issues[:2]
    fallback = renderer._select_cases(data)
    assert fallback["cases"][0]["issue_index"] == 1
    assert fallback["cases"][1]["issue_index"] == 0
    assert "无净损失" in fallback["cases"][1]["label"]


def test_embedded_script_controls_all_axes_without_network(renderer, fixture_data):
    node = os.environ.get("SEISMOFLUX_TEST_NODE") or shutil.which("node")
    if node is None:
        pytest.skip("Node is optional for the offline DOM/canvas mock regression")
    summary, events, alarms, grid, exposures, _ = fixture_data
    data = renderer._replay_data(summary, events, alarms, grid, exposures)
    page = renderer.HTML_TEMPLATE.replace("__DATA__", renderer._json_for_script(data))
    javascript = r"""
const fs=require("fs"),vm=require("vm"),html=fs.readFileSync(0,"utf8");
const scripts=[...html.matchAll(/<script(?: [^>]*)?>([\s\S]*?)<\/script>/g)];
class Element {
  constructor(){this.value="0";this.options=[];this.children=[];this.style={};
    this.clientWidth=600;this.clientHeight=430;}
  set textContent(s){this.text=s;this.children=[];this.options=[];}
  get textContent(){return this.text||"";}
  add(x){this.options.push(x);} addEventListener(){}
  insertRow(){const e=new Element;this.children.push(e);return e;}
  insertCell(){const e=new Element;this.children.push(e);this.lastChild=e;return e;}
  getContext(){return new Proxy({},{get:()=>()=>{}});}
}
const elements=new Map,document={getElementById(id){
  if(!elements.has(id))elements.set(id,new Element);return elements.get(id);}};
document.getElementById("replay-data").textContent=scripts[0][1];
document.getElementById("view").value="anchor";
const context=vm.createContext({document,console,
  window:{devicePixelRatio:1,addEventListener(){}},
  Option:function(label,value){this.label=label;this.value=value;}});
vm.runInContext(scripts[1][1],context);
vm.runInContext(`
if(horizon.value!=="30"||tolerance.value!=="0")throw Error("default axis wrong");
let count=0;
for(const h of S.horizons_days)for(const b of S.magnitude_bins){
  horizon.value=String(h);band.value=b;rebuildDates();
  for(const model of S.model_ids)for(const budget of D.area_budgets)
  for(const tol of D.tolerances){
    candidate.value=model;area.value=String(budget);tolerance.value=String(tol);
    update();count++;
  }
  slider.value="1";drawIssue();
  if(visibleIssues[1].event_count!==0)throw Error("lost empty period");
}
console.log(count,"axis combinations and empty-date replay passed");
`,context);
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
    assert "1400 axis combinations and empty-date replay passed" in result.stdout
