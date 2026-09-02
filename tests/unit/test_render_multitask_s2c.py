# ruff: noqa: E501, RUF001
"""Synthetic-only S2-C scientific figures and offline controls."""

from __future__ import annotations

import importlib.util
import json
import os
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
    path = Path(__file__).resolve().parents[2] / "scripts/render_multitask_s2c.py"
    spec = importlib.util.spec_from_file_location("s2c_renderer_synthetic", path)
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
                    if model.startswith("S2C_"):
                        mass = [0.05, 0.05, 0.7, 0.2]
                    result = score_exposure(
                        log_mass=np.log(mass),
                        grid=grid,
                        target=target,
                        fold_id="synthetic-early" if period == 0 else "synthetic-post",
                        horizon_days=horizon,
                        issue_time_us=946742400000000
                        if period == 0
                        else 1420070400000000 + period * 366 * 86400000000,
                        magnitude_bin=band,
                        model_id=model,
                        budgets=list(renderer.AREA_BUDGETS),
                        near_cells=[{0, 2}] if period != 2 else [],
                    )
                    exposures.extend(result[0])
                    events.extend(result[1])
                    alarms.extend(result[2])
    exposure_frame, event_frame = pd.DataFrame(exposures), pd.DataFrame(events)
    protocol_path = Path(__file__).resolve().parents[2] / "configs/multitask_s2_c_strain.yaml"
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
        "scientific_role": renderer.ROLE,
    }
    post_curves, post_pairs, _ = summarize(
        exposure_frame.loc[exposure_frame.fold_id == "synthetic-post"],
        event_frame.loc[event_frame.fold_id == "synthetic-post"],
        planned,
        (),
    )
    for row in post_curves:
        row["log_density_status"] = (
            "finite" if row["event_mean_log_density"] is not None else "no_events"
        )
    summary["main_anchor_count"] = 2
    summary["post_release_development"] = {
        "curves": post_curves,
        "pairings": post_pairs,
        "fold_ids": ["synthetic-post"],
        "main_anchor_count": 1,
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


def test_axes_and_cases(renderer, fixture_data):
    data = renderer._replay_data(*fixture_data[:5])
    assert len(data["issues"]) == 30
    assert sum(not i["events"] for i in data["issues"]) == 10
    assert all(len(i["forecasts"]) == 8 for i in data["issues"])
    selection = renderer._select_cases(data)
    assert selection["candidate_model_id"] == "S2C_UNIT_CATALOG_MIX"
    gain, failure = selection["cases"]
    assert gain["net_hits"] == 1 and failure["net_hits"] == -1
    for issue in data["issues"]:
        for event in issue["events"]:
            for model in renderer.MODEL_LABELS:
                event["hits"][model]["0"] = 0
    assert all(c["net_hits"] == 0 for c in renderer._select_cases(data)["cases"])


def test_render_slices_license_local_boundary(renderer, fixture_data, tmp_path, monkeypatch):
    save = renderer._shared._save_figure

    def check_percent_axes(fig, root, stem):
        if stem == renderer.STATIC_STEMS[0]:
            assert all(ax.get_xlim() == (0, 100) for ax in fig.axes)
            for ax in fig.axes:
                for text in ax.texts:
                    if text.get_text().startswith("100.0%"):
                        assert text.get_position()[0] < 100
                        assert text.get_ha() == "right"
        if stem == renderer.STATIC_STEMS[1]:
            assert all(ax.get_ylim() == (0, 100) for ax in fig.axes)
        save(fig, root, stem)

    monkeypatch.setattr(renderer._shared, "_save_figure", check_percent_axes)
    _write_fixture(tmp_path, fixture_data, renderer)
    page = renderer.render(tmp_path)
    text = page.read_text(encoding="utf-8")
    assert "S2-C：应变" in text and "2015—2019" in text and "独立测试" in text
    assert "CC-BY-NC-SA 3.0" in text and "GEM Foundation 2014" in text
    assert "S2B_" not in text and "共同 385" not in text
    assert "fetch(" not in text and "<script src=" not in text
    assert "synthetic-</script>-0" not in text
    manifest = json.loads((page.parent / "render_manifest.json").read_text(encoding="utf-8"))
    assert sum(a["audience"].startswith("public_") for a in manifest["artifacts"]) == 6
    assert all((page.parent / name).stat().st_size > 1000 for name in renderer.FILENAMES)
    for stem in renderer.STATIC_STEMS:
        svg = (page.parent / f"{stem}.svg").read_text(encoding="utf-8")
        assert "<text" in svg and "CC-BY-NC-SA" in svg and "synthetic-" not in svg
    first = (page.parent / f"{renderer.STATIC_STEMS[0]}.svg").read_text(encoding="utf-8")
    assert "/2" in first and "/1" in first


def test_incomplete_slice_and_changed_hash_rejected(renderer, fixture_data, tmp_path):
    data = deepcopy(fixture_data)
    data[0]["post_release_development"]["curves"].pop()
    score = _write_fixture(tmp_path, data, renderer)
    with pytest.raises(ValueError, match="slice"):
        renderer._load(tmp_path)
    with (score / "summary.json").open("a") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="hash changed"):
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
if(reference.value!=="C2B_D0_MULTISCALE"||candidate.value!=="S2C_UNIT_CATALOG_MIX"||tolerance.value!=="0")throw Error("wrong S2 defaults");
let count=0;
slice.value="post";rebuildDates();update();if(visibleIssues.length!==2||visibleIssues.some(i=>i.fold!=="synthetic-post"))throw Error("slice date mismatch");
const savedPost=S.post_release_development.curves.find(r=>r.model_id===candidate.value&&axis(r));
if(document.getElementById("metric-table").children[0].children[1].textContent!==(number(savedPost.anchor_hits,0)+" / "+number(savedPost.anchor_total,0)))throw Error("slice metric mismatch");
if(!logLabel({log_density_status:"negative_infinity_from_zero_mass",event_mean_log_density:null}).includes("−∞"))throw Error("zero mass lost");
slice.value="all";rebuildDates();update();

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
    assert "800 axes passed" in result.stdout
