# ruff: noqa: E501, RUF001
"""Two scientific figures and one self-contained, offline C2A replay.

Only completed C2A score artifacts are read. No model is fitted or scored, no
catalog is opened, and no external map or script is requested by the HTML.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_MODELS = ("L1_REGIONAL_CONSTANT", "L2_KDE_CAUSAL", "L3_B0_R30_CAUSAL")
MODEL_LABELS = ("L1 分区活动率", "L2 长期空间分布", "L3 长期＋近期活动")
TREATMENTS = ("C0", "A", "B")
LABELS = {"C0": "C0 原目录", "A": "A 保留未知地区", "B": "B 排除未知地区"}
COLORS = {"C0": "#264b73", "A": "#008f83", "B": "#dc8b22"}
AREA_BUDGETS = (300000.0, 450000.0, 600000.0, 750000.0, 960000.0)
FILENAMES = (
    "01_main_anchor_hits.png",
    "01_main_anchor_hits.svg",
    "02_l3_area_curves.png",
    "02_l3_area_curves.svg",
    "seismoflux_s1c2a_replay.html",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(output_root: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    score = output_root / "score_phase"
    record = json.loads((score / "score_manifest.json").read_text(encoding="utf-8"))
    if not record.get("complete"):
        raise ValueError("Only completed C2A scores may be rendered")
    references = {item["path"]: item["sha256"] for item in record["artifacts"]}
    for name in (
        "summary.json",
        "event_results.parquet",
        "alarm_prefixes.parquet",
        "grid_cells.csv",
    ):
        if references.get(name) != _sha256(score / name):
            raise ValueError(f"Score artifact changed: {name}")
    summary = json.loads((score / "summary.json").read_text(encoding="utf-8"))
    expected_models = {f"{t}_{m}" for t in TREATMENTS for m in BASE_MODELS}
    if set(summary["model_ids"]) != expected_models or summary["horizon_days"] != 30:
        raise ValueError("Renderer requires all nine C2A 30-day location curves")
    if summary.get("holdout_read") or summary.get("locked_test_run"):
        raise ValueError("This renderer is for development sensitivity only")
    return (
        summary,
        pd.read_parquet(score / "event_results.parquet"),
        pd.read_parquet(score / "alarm_prefixes.parquet"),
        pd.read_csv(score / "grid_cells.csv"),
    )


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "Arial", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "font.size": 11,
            "axes.titlesize": 16,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#acb8c2",
            "text.color": "#253642",
            "axes.labelcolor": "#253642",
            "xtick.color": "#253642",
            "ytick.color": "#253642",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _footnote(summary: dict[str, Any]) -> str:
    synthetic = "合成测试示例 · 不代表科学结果\n" if summary.get("synthetic_fixture") else ""
    return (
        synthetic + "开发期固定参数敏感性：不是独立确认，也不是模型重调后的性能上限。\n"
        "同一全国评价区域、同一目标；预算一致，完整格前缀的实际报警面积逐模型记录。"
    )


def _static_figures(summary: dict[str, Any], render_root: Path) -> None:
    _style()
    curves = {(row["model_id"], float(row["area_budget_km2"])): row for row in summary["curves"]}
    count = int(summary["fixed_anchor_episode_count"])
    fig, ax = plt.subplots(figsize=(12.0, 6.4))
    positions = np.arange(len(BASE_MODELS))
    width = 0.23
    max_hits = 0
    for treatment_index, treatment in enumerate(TREATMENTS):
        values = [
            curves[(f"{treatment}_{model}", 600000.0)]["anchor_hits"] for model in BASE_MODELS
        ]
        max_hits = max(max_hits, *values)
        bars = ax.bar(
            positions + (treatment_index - 1) * width,
            values,
            width,
            color=COLORS[treatment],
            label=LABELS[treatment],
            zorder=3,
        )
        ax.bar_label(bars, labels=[f"{value}/{count}" for value in values], padding=5, fontsize=11)
    ax.set_xticks(positions, MODEL_LABELS)
    ax.set_ylabel("命中的独立首震锚点数")
    ax.set_ylim(0, max(5, max_hits * 1.26))
    ax.set_title("改变历史目录的局地筛选，命中数有何变化？", loc="left", pad=38)
    ax.text(
        0,
        1.045,
        f"30 天 · Ms 5–6 · 60 万 km² 报警预算 · 同一批 {count} 个独立首震锚点",
        transform=ax.transAxes,
        fontsize=11,
        color="#526573",
    )
    ax.grid(axis="y", color="#e7ecf0", linewidth=0.8, zorder=0)
    ax.legend(frameon=False, loc="upper left", ncol=3, bbox_to_anchor=(0, 1.01))
    fig.text(0.075, 0.065, _footnote(summary), fontsize=10, linespacing=1.7, color="#526573")
    fig.subplots_adjust(left=0.085, right=0.98, top=0.79, bottom=0.24)
    for suffix in ("png", "svg"):
        fig.savefig(render_root / f"01_main_anchor_hits.{suffix}", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.4, 6.4))
    model = BASE_MODELS[2]
    styles = (("o", "-"), ("s", "--"), ("^", "-."))
    for treatment, (marker, line) in zip(TREATMENTS, styles, strict=True):
        values = [
            curves[(f"{treatment}_{model}", area)]["anchor_recall"] * 100.0 for area in AREA_BUDGETS
        ]
        ax.plot(
            np.asarray(AREA_BUDGETS) / 10000,
            values,
            marker=marker,
            linestyle=line,
            color=COLORS[treatment],
            label=LABELS[treatment],
            linewidth=2.3,
            markersize=7,
        )
    ax.set_xticks(np.asarray(AREA_BUDGETS) / 10000, ["30", "45", "60", "75", "96"])
    ax.set_xlabel("报警面积预算（万 km²）")
    ax.set_ylabel("独立首震锚点召回（%）")
    ax.set_ylim(bottom=0)
    ax.set_title("L3：扩大或缩小报警面积后，差异是否一致？", loc="left", pad=32)
    ax.text(
        0,
        1.045,
        f"同一批 {count} 个首震锚点；三个处理均完整展示，不只保留有利面积档。",
        transform=ax.transAxes,
        fontsize=11,
        color="#526573",
    )
    ax.grid(color="#e7ecf0", linewidth=0.8)
    ax.legend(frameon=False, loc="upper left")
    fig.text(0.085, 0.045, _footnote(summary), fontsize=10, linespacing=1.7, color="#526573")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.81, bottom=0.24)
    for suffix in ("png", "svg"):
        fig.savefig(render_root / f"02_l3_area_curves.{suffix}", dpi=180)
    plt.close(fig)


def _replay_data(
    summary: dict[str, Any],
    events: pd.DataFrame,
    alarms: pd.DataFrame,
    grid: pd.DataFrame,
) -> dict[str, Any]:
    cells = grid.sort_values("cell_index")
    if not np.array_equal(cells.cell_index.to_numpy(), np.arange(len(cells))):
        raise ValueError("Grid indices must remain contiguous and unchanged")
    # Five prefixes share one ranking. Embed the longest once, not five copies.
    issues: dict[tuple[str, int], dict[str, Any]] = {}
    for (fold, issue_us, model), group in alarms.groupby(
        ["fold_id", "issue_time_us", "model_id"], sort=True
    ):
        group = group.sort_values("area_budget_km2")
        if tuple(group.area_budget_km2) != AREA_BUDGETS:
            raise ValueError("All five area budgets must be retained in the replay")
        order = [int(v) for v in group.iloc[-1].selected_cell_indices]
        if any(not 0 <= value < len(cells) for value in order):
            raise ValueError("Alarm cell index leaves the provided grid")
        area_rows = []
        for row in group.itertuples(index=False):
            prefix = [int(v) for v in row.selected_cell_indices]
            if prefix != order[: len(prefix)]:
                raise ValueError("Area prefixes are not nested; cannot compact replay data")
            area_rows.append([float(row.area_budget_km2), float(row.actual_area_km2), len(prefix)])
        key = (str(fold), int(issue_us))
        issue = issues.setdefault(
            key,
            {
                "fold": str(fold),
                "issue_us": int(issue_us),
                "label": str(group.iloc[0].issue_time_utc),
                "events": [],
                "forecasts": {},
            },
        )
        issue["forecasts"][model] = {"order": order, "areas": area_rows}
    # Event coordinates are original catalog epicentres copied by the C0 scorer.
    unique = events.drop_duplicates(["fold_id", "issue_time_us", "event_id"])
    for row in unique.loc[unique.is_episode_anchor].itertuples(index=False):
        issues[(row.fold_id, int(row.issue_time_us))]["events"].append(
            {
                "id": row.event_id,
                "episode": row.episode_id,
                "cell": int(row.cell_index),
                "lon": float(row.longitude),
                "lat": float(row.latitude),
            }
        )
    ordered = sorted(issues.values(), key=lambda row: row["issue_us"])
    if len(ordered) != summary["primary_exposure_count"]:
        raise ValueError("Replay lost an exposure, including a possible empty period")
    if any(set(issue["forecasts"]) != set(summary["model_ids"]) for issue in ordered):
        raise ValueError("Replay is missing one of the nine model curves")
    return {
        "summary": summary,
        "issues": ordered,
        "grid": [
            [round(float(row.longitude), 6), round(float(row.latitude), 6)]
            for row in cells.itertuples(index=False)
        ],
        "colors": COLORS,
        "labels": LABELS,
        "base_models": list(BASE_MODELS),
        "model_labels": list(MODEL_LABELS),
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SeismoFlux｜局地目录筛选的预测效果</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f5f7f9;color:#243746;font:16px/1.6 "Microsoft YaHei",Arial,sans-serif}
main{max-width:1280px;margin:auto;padding:30px 24px}h1{font-size:28px;line-height:1.3;margin:0 0 12px}h2{font-size:20px;margin:28px 0 12px}p{margin:8px 0}.note{color:#536977;font-size:14px}.scope{padding:12px 16px;background:#eaf0f4;border-left:4px solid #264b73;margin:18px 0}.synthetic{color:#a12c33;font-weight:bold}
.figure{display:block;width:100%;background:white;margin:18px 0;border-radius:4px}.controls{display:flex;flex-wrap:wrap;align-items:center;gap:16px;margin:18px 0}.controls label{display:flex;align-items:center;gap:8px}select,button{font:inherit;padding:7px 10px;background:white;border:1px solid #aab8c2;border-radius:4px}button{cursor:pointer}button:disabled{opacity:.4;cursor:default}input[type=range]{flex:1;min-width:180px}svg{width:100%;height:auto;display:block;background:white}canvas{width:100%;height:310px;display:block;background:white}.maps{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}.map h3{font-size:16px;margin:4px 0}.map p{font-size:13px;min-height:42px}.legend{display:flex;flex-wrap:wrap;gap:18px;font-size:14px}.swatch{display:inline-block;width:12px;height:12px;margin-right:5px;vertical-align:middle}.tablewrap{overflow-x:auto}table{border-collapse:collapse;width:100%;background:white;font-size:14px}th,td{padding:9px 10px;border-bottom:1px solid #e0e7ec;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}thead{background:#eaf0f4}.positive{color:#007b6f}.negative{color:#ad3c43}.neutral{color:#536977}.twocol{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(0,1fr);gap:24px;align-items:start}#issue-label{font-weight:bold}footer{font-size:13px;color:#536977;margin-top:32px;padding-top:18px;border-top:1px solid #d4dee5}@media(max-width:850px){.maps,.twocol{grid-template-columns:1fr}main{padding:20px 14px}h1{font-size:24px}canvas{height:340px}.controls label{flex-wrap:wrap}.map p{min-height:0}}
</style></head><body><main>
<h1>历史目录怎样筛选，预测会更好吗？</h1>
<p>把同一个模型放在同一批地震、同样大小的报警预算下，比较原目录与两种局地筛选。</p>
<div class="scope"><span id="study-scope"></span><br><span class="note">A：仅排除明确 Mc &gt; 4 的历史训练中心，保留样本稀疏、Mc 未知地区。B：进一步排除未知地区。三种处理均预测和评价完整全国区域；模型参数不重新选择。</span><div id="synthetic" class="synthetic"></div></div>
<img class="figure" src="data:image/png;base64,__BAR_IMAGE__" alt="原目录、保留未知和排除未知三种处理，在三个模型中的全部九个命中数">
<div class="controls"><label for="model">模型 <select id="model"></select></label><label for="area">报警预算 <select id="area"></select></label></div>
<div class="twocol"><section><h2>五档面积下的整体召回</h2><svg id="curve" viewBox="0 0 660 350" role="img" aria-label="所选模型三种处理的面积与召回曲线"></svg><div class="legend" id="curve-legend"></div></section><section><h2>新增命中，也看新增漏报</h2><div class="tablewrap"><table><thead><tr><th>与原目录相比</th><th>新命中</th><th>新漏报</th><th>净变化</th><th>召回变化</th></tr></thead><tbody id="pair-table"></tbody></table></div><p class="note" id="pair-detail"></p><p class="note">不确定区间来自 2,000 次配对震序重采样；不把某个显著性阈值当作研究的一票否决条件。零变化和负变化同样保留。</p></section></div>
<h2>按起报日期回放：同一目标、同一面积预算</h2>
<div class="controls"><button id="previous" type="button">上一期</button><input id="issue" type="range" min="0" value="0" aria-label="起报期"><button id="next" type="button">下一期</button><label for="issue-select">日期 <select id="issue-select"></select></label></div>
<p id="issue-label" aria-live="polite"></p>
<div class="maps"><section class="map"><h3 id="title-C0"></h3><p id="stats-C0"></p><canvas id="map-C0" role="img" aria-label="原目录该期报警格与首震锚点"></canvas></section><section class="map"><h3 id="title-A"></h3><p id="stats-A"></p><canvas id="map-A" role="img" aria-label="保留未知地区该期报警格与首震锚点"></canvas></section><section class="map"><h3 id="title-B"></h3><p id="stats-B"></p><canvas id="map-B" role="img" aria-label="排除未知地区该期报警格与首震锚点"></canvas></section></div>
<p class="note">经纬度坐标图，不绘制或推断行政边界。灰点为全国评价格的代表点，彩色方点为报警格的代表点，不能当作完整格边界。● 表示命中的目录震中，× 表示漏报的目录震中；命中严格按震中所属 25 km 格判断，不按图上点间距判断。图中仅显示评价主指标的首震锚点。</p>
<div class="tablewrap"><table><thead><tr><th>本期首震锚点</th><th>经度</th><th>纬度</th><th>C0 原目录</th><th>A 保留未知</th><th>B 排除未知</th></tr></thead><tbody id="event-table"></tbody></table></div>
<footer>这是开发期固定参数输入敏感性，不是独立确认、最终优胜模型或绝对发震概率。报警预算相同，但完整格前缀的实际面积可能略有差异，已逐期标明。所有数据已嵌入本页，可断网使用；不读取未来地震，不回写历史预测。</footer>
</main><script id="replay-data" type="application/json">__DATA__</script><script>
"use strict";
const D=JSON.parse(document.getElementById("replay-data").textContent), S=D.summary, T=["C0","A","B"];
const $=id=>document.getElementById(id), model=$("model"),area=$("area"),slider=$("issue"),dateSelect=$("issue-select");
const number=(x,n=1)=>Number(x).toLocaleString("zh-CN",{maximumFractionDigits:n,minimumFractionDigits:n});
const signed=(x,n=1)=>(x>0?"+":"")+number(x,n), date=iso=>new Date(iso).toLocaleString("zh-CN",{timeZone:"Asia/Shanghai",year:"numeric",month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit",hour12:false});
$("study-scope").textContent=`30 天 · Ms 5–6 · ${S.fixed_anchor_episode_count} 个固定首震锚点 · ${S.primary_exposure_count} 期（含空期） · 24 小时目录延迟`;
if(S.synthetic_fixture)$("synthetic").textContent="合成测试示例：仅验证绘图和交互，不代表科学结果。";
D.base_models.forEach((id,i)=>model.add(new Option(D.model_labels[i],id)));model.value=D.base_models[2];
[300000,450000,600000,750000,960000].forEach(a=>area.add(new Option(`${a/10000} 万 km²`,a)));area.value="600000";
D.issues.forEach((issue,i)=>dateSelect.add(new Option(`${i+1}. ${date(issue.label)}`,i)));slider.max=D.issues.length-1;
$("curve-legend").innerHTML=T.map(t=>`<span><i class="swatch" style="background:${D.colors[t]}"></i>${D.labels[t]}</span>`).join("");
function drawCurve(){const W=660,H=350,l=68,r=26,top=25,b=60,groups=T.map(t=>S.curves.filter(c=>c.model_id===`${t}_${model.value}`).sort((a,b)=>a.area_budget_km2-b.area_budget_km2));
const ymax=Math.max(5,...groups.flat().map(c=>100*c.anchor_recall))*1.16,x=a=>l+(a-300000)/(960000-300000)*(W-l-r),y=v=>H-b-v/ymax*(H-b-top);
let html=`<title>报警面积预算与独立首震锚点召回</title><rect x="${l}" y="${top}" width="${W-l-r}" height="${H-b-top}" fill="none" stroke="#b7c4cd"/>`;
for(let i=0;i<=4;i++){const v=ymax*i/4,yy=y(v);html+=`<line x1="${l}" x2="${W-r}" y1="${yy}" y2="${yy}" stroke="#e6ecf0"/><text x="${l-9}" y="${yy+5}" text-anchor="end">${number(v,0)}</text>`;}
groups[0].forEach(c=>html+=`<text x="${x(c.area_budget_km2)}" y="${H-b+24}" text-anchor="middle">${c.area_budget_km2/10000}</text>`);
groups.forEach((g,i)=>{const points=g.map(c=>`${x(c.area_budget_km2)},${y(c.anchor_recall*100)}`).join(" ");html+=`<polyline points="${points}" fill="none" stroke="${D.colors[T[i]]}" stroke-width="2.5" ${i?`stroke-dasharray="${i===1?"7 4":"9 3 2 3"}"`:""}/>`;g.forEach(c=>html+=`<circle cx="${x(c.area_budget_km2)}" cy="${y(c.anchor_recall*100)}" r="4" fill="${D.colors[T[i]]}"><title>${D.labels[T[i]]}：${c.anchor_hits}/${c.anchor_total}，${number(c.anchor_recall*100)}%</title></circle>`);});
html+=`<line x1="${x(+area.value)}" x2="${x(+area.value)}" y1="${top}" y2="${H-b}" stroke="#6d7980" stroke-dasharray="3 4"/><text x="${(l+W-r)/2}" y="${H-14}" text-anchor="middle">报警预算（万 km²）</text><text transform="translate(18 ${H/2}) rotate(-90)" text-anchor="middle">首震锚点召回（%）</text>`;
$("curve").innerHTML=`<g font-family="Microsoft YaHei,Arial,sans-serif" font-size="14" fill="#243746">${html}</g>`;
}
function drawPairs(){const ps=["A","B"].map(t=>S.pairings.find(p=>p.candidate_model_id===`${t}_${model.value}`&&p.reference_model_id===`C0_${model.value}`&&p.area_budget_km2===+area.value));
$("pair-table").innerHTML=ps.map((p,i)=>`<tr><td>${D.labels[T[i+1]]}</td><td>${p.gained}</td><td>${p.lost}</td><td class="${p.net_hits>0?"positive":p.net_hits<0?"negative":"neutral"}">${signed(p.net_hits,0)}</td><td>${signed(p.delta_recall_pp)} 个百分点</td></tr>`).join("");
$("pair-detail").textContent=ps.map((p,i)=>`${T[i+1]}：95% 区间 [${signed(p.bootstrap_ci95_pp[0])}, ${signed(p.bootstrap_ci95_pp[1])}] 个百分点；各折净命中 ${p.per_fold.map(f=>f.direction==="not_evaluable"?"NA":signed(f.net_hits,0)).join(" / ")}`).join("。 ");}
const lonMin=Math.min(...D.grid.map(c=>c[0]))-1,lonMax=Math.max(...D.grid.map(c=>c[0]))+1,latMin=Math.min(...D.grid.map(c=>c[1]))-1,latMax=Math.max(...D.grid.map(c=>c[1]))+1;
function mapDraw(t,issue,selected){const canvas=$("map-"+t),ratio=window.devicePixelRatio||1,w=canvas.clientWidth,h=canvas.clientHeight;canvas.width=w*ratio;canvas.height=h*ratio;const c=canvas.getContext("2d");c.scale(ratio,ratio);c.fillStyle="white";c.fillRect(0,0,w,h);const l=42,r=12,top=12,b=34,x=lon=>l+(lon-lonMin)/(lonMax-lonMin)*(w-l-r),y=lat=>h-b-(lat-latMin)/(latMax-latMin)*(h-b-top);
c.font='11px "Microsoft YaHei",Arial,sans-serif';c.fillStyle="#536977";c.strokeStyle="#e6ecf0";c.lineWidth=.6;for(let i=0;i<=3;i++){let lon=lonMin+(lonMax-lonMin)*i/3,lat=latMin+(latMax-latMin)*i/3;c.beginPath();c.moveTo(x(lon),top);c.lineTo(x(lon),h-b);c.moveTo(l,y(lat));c.lineTo(w-r,y(lat));c.stroke();c.textAlign=i===0?"left":i===3?"right":"center";c.fillText(number(lon,0)+"°E",x(lon),h-10);c.textAlign="right";c.fillText(number(lat,0)+"°N",l-6,y(lat)+4);}
c.fillStyle="#dce3e8";D.grid.forEach(p=>c.fillRect(x(p[0])-.55,y(p[1])-.55,1.1,1.1));c.fillStyle=D.colors[t];selected.forEach(index=>{const p=D.grid[index];c.fillRect(x(p[0])-1,y(p[1])-1,2,2);});issue.events.forEach(e=>{const xx=x(e.lon),yy=y(e.lat);c.lineWidth=1.8;if(selected.has(e.cell)){c.fillStyle="#132b39";c.strokeStyle="white";c.beginPath();c.arc(xx,yy,4.2,0,Math.PI*2);c.fill();c.stroke();}else{c.strokeStyle="#b23f47";c.beginPath();c.moveTo(xx-4,yy-4);c.lineTo(xx+4,yy+4);c.moveTo(xx-4,yy+4);c.lineTo(xx+4,yy-4);c.stroke();}});}
function drawIssue(){const i=+slider.value,issue=D.issues[i];dateSelect.value=String(i);$("previous").disabled=i===0;$("next").disabled=i===D.issues.length-1;$("issue-label").textContent=`第 ${i+1}/${D.issues.length} 期 · ${date(issue.label)}（北京时间） · ${issue.fold} · 本期 ${issue.events.length} 个首震锚点`;
const sets={};T.forEach(t=>{const forecast=issue.forecasts[`${t}_${model.value}`],a=forecast.areas.find(row=>row[0]===+area.value),selected=new Set(forecast.order.slice(0,a[2]));sets[t]=selected;const hit=issue.events.filter(e=>selected.has(e.cell)).length;$("title-"+t).textContent=D.labels[t];$("title-"+t).style.color=D.colors[t];$("stats-"+t).textContent=`命中 ${hit}/${issue.events.length} · 实际报警 ${number(a[1]/10000,2)} 万 km² · ${selected.size} 格`;mapDraw(t,issue,selected);});
const body=$("event-table");body.textContent="";if(!issue.events.length){const tr=body.insertRow();const td=tr.insertCell();td.colSpan=6;td.textContent="本期无评价首震锚点；该期仍保留，不从面积和时间回放中删除。";}else issue.events.forEach(e=>{const row=body.insertRow();[e.id,number(e.lon,3),number(e.lat,3),...T.map(t=>sets[t].has(e.cell)?"命中":"漏报")].forEach(v=>row.insertCell().textContent=v);});}
function update(){drawCurve();drawPairs();drawIssue();}model.addEventListener("change",update);area.addEventListener("change",update);slider.addEventListener("input",drawIssue);dateSelect.addEventListener("change",()=>{slider.value=dateSelect.value;drawIssue();});$("previous").addEventListener("click",()=>{slider.value=+slider.value-1;drawIssue();});$("next").addEventListener("click",()=>{slider.value=+slider.value+1;drawIssue();});window.addEventListener("resize",drawIssue);update();
</script></body></html>"""


def render(output_root: Path, render_root: Path | None = None) -> Path:
    output = output_root.resolve()
    destination = render_root.resolve() if render_root is not None else output / "visualization"
    summary, events, alarms, grid = _load(output)
    data = _replay_data(summary, events, alarms, grid)
    destination.mkdir(parents=True, exist_ok=True)
    _static_figures(summary, destination)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace(
        "<", "\\u003c"
    )
    image = base64.b64encode((destination / "01_main_anchor_hits.png").read_bytes()).decode("ascii")
    page = HTML_TEMPLATE.replace("__BAR_IMAGE__", image).replace("__DATA__", payload)
    html_path = destination / "seismoflux_s1c2a_replay.html"
    html_path.write_text(page, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "source_score_manifest_sha256": _sha256(output / "score_phase/score_manifest.json"),
        "event_coordinate_semantics": "original_catalog_epicentre_from_C0_target_payload_not_grid_centre",
        "alarm_coordinate_semantics": "provided_grid_representative_points_not_cell_polygons",
        "network_resources": [],
        "synthetic_fixture": bool(summary.get("synthetic_fixture", False)),
        "artifacts": [{"path": name, "sha256": _sha256(destination / name)} for name in FILENAMES],
    }
    (destination / "render_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return html_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="渲染 C2A 固定参数目录敏感性：两张图及完全离线回放"
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--render-root", type=Path)
    args = parser.parse_args()
    print(render(args.output_root, args.render_root))


if __name__ == "__main__":
    main()
