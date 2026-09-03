"""Render a self-contained, read-only historical replay from saved predictions.

This module performs no data loading, training, score computation, or hit testing.
The payload contains saved alarm cells and saved event-level hit decisions.
"""

# Embedded HTML/JS is kept intact; Chinese publication text uses Chinese punctuation.
# ruff: noqa: E501, RUF001

from __future__ import annotations

import json
from typing import Any


def render_replay_html(payload: dict[str, Any]) -> str:
    """Return a complete offline HTML document for the supplied local payload.

    ``local_only`` must be true. JSON must be finite and serializable; text is
    embedded with HTML-significant characters escaped and inserted into the DOM
    only through ``textContent``. Empty frames and unevaluable tasks are retained.
    """
    if payload.get("local_only") is not True:
        raise ValueError("Historical event replay must be explicitly local_only")
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    for char, replacement in (
        ("&", "\\u0026"),
        ("<", "\\u003c"),
        (">", "\\u003e"),
        ("\u2028", "\\u2028"),
        ("\u2029", "\\u2029"),
    ):
        encoded = encoded.replace(char, replacement)
    return _HTML.replace("__REPLAY_PAYLOAD__", encoded)


_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'none'; font-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'">
<title>SeismoFlux｜历史预测回放</title>
<style>
:root{--ink:#183044;--muted:#61717f;--line:#dce4e9;--paper:#f2f5f6;--card:#fff;--teal:#087c72;--red:#bb493e;--blue:#356ca8;--grey:#74828b;--alarm:#d2a64c}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:"Microsoft YaHei","PingFang SC",Arial,sans-serif;line-height:1.6}header{background:#183044;color:#fff;padding:30px max(24px,calc((100vw - 1460px)/2)) 25px}.eyebrow{color:#b4ced9;font-size:12px;letter-spacing:2px}h1{font-size:29px;line-height:1.35;margin:7px 0 10px;font-weight:600}header p{color:#d6e3e8;margin:0;font-size:14px}.status{display:inline-block;background:#ffedc7;color:#684717;border-radius:5px;padding:5px 10px;font-size:13px;margin-top:17px;font-weight:bold}main{max-width:1508px;margin:auto;padding:22px 24px 40px}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;box-shadow:0 3px 14px #16344605}.controls{padding:18px;display:grid;grid-template-columns:repeat(5,minmax(120px,1fr));gap:13px 16px}label{display:block;font-size:12px;font-weight:600;color:var(--muted);margin-bottom:5px}select,button{font:inherit;color:var(--ink);border:1px solid #c8d4dc;background:#fff;border-radius:5px}select{width:100%;padding:8px 28px 8px 9px;font-size:13px;min-height:40px}button{padding:7px 12px;cursor:pointer}button:hover:not(:disabled){border-color:var(--teal);color:var(--teal)}button:disabled{opacity:.4;cursor:default}select:focus-visible,button:focus-visible,canvas:focus-visible{outline:3px solid #73b4c9;outline-offset:2px}.timeline{padding:15px 18px;margin-top:12px;display:flex;align-items:end;gap:12px}.timeline-field{flex:1}.timeline-meta{font-size:12px;color:var(--muted);padding:5px 0}.window-info{margin:17px 2px 12px;font-size:14px}.window-info strong{font-weight:600}.annotation{font-size:12px;color:var(--muted)}.warning{color:#83500b;background:#fff3db;padding:10px 13px;border-radius:6px;margin-bottom:12px}.maps{display:grid;grid-template-columns:1fr 1fr;gap:16px}.map-card{overflow:hidden}.map-head{padding:15px 17px 12px;border-bottom:1px solid var(--line)}.map-role{font-size:11px;color:var(--muted);letter-spacing:1px}.map-title{font-size:18px;font-weight:600;margin:3px 0}.map-area{font-size:12px;color:var(--muted)}canvas{display:block;width:100%;height:430px;background:#f7fafb;cursor:crosshair}.map-caption{padding:8px 17px 12px;font-size:11px;color:var(--muted)}.legend{display:flex;flex-wrap:wrap;gap:10px 24px;padding:14px 2px;font-size:13px}.legend span{white-space:nowrap}.mark{font-size:17px;font-weight:bold;display:inline-block;width:20px;text-align:center}.gain{color:var(--teal)}.loss{color:var(--red)}.both{color:var(--blue)}.miss{color:var(--grey)}.na{color:#9c64a3}.alarm-key{display:inline-block;width:13px;height:13px;margin-right:7px;background:#e7c777;border:1px solid #ba9240;vertical-align:-1px}.stats{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin:0 0 16px}.stat{padding:11px 15px;background:white;border:1px solid var(--line);border-radius:7px}.stat b{display:block;font-size:22px;font-weight:600;line-height:1.4}.stat span{font-size:12px;color:var(--muted)}.section{padding:18px;margin-top:16px}h2{font-size:17px;font-weight:600;margin:0 0 4px}.section p{font-size:12px;color:var(--muted);margin:3px 0 12px}.table-wrap{overflow:auto;max-height:340px}table{border-collapse:collapse;width:100%;font-size:13px;text-align:left}th{position:sticky;top:0;background:#f0f5f6;color:#4a606f;font-weight:600;white-space:nowrap}td,th{padding:9px 12px;border-bottom:1px solid #e5ebee}tbody tr:hover{background:#f4f9f9}tbody tr.selected{background:#e4f2f0}.event-button{border:none;background:transparent;padding:0;text-align:left;text-decoration:underline;text-underline-offset:3px;font-size:13px}.detail{padding:12px 14px;background:#f1f6f7;border-left:3px solid #80aeb3;font-size:13px;margin-top:14px;white-space:pre-line}.two-column{display:grid;grid-template-columns:1.3fr 1fr;gap:16px}.notes{font-size:12px;color:var(--muted)}.notes ul{padding-left:20px;margin:8px 0}.notes li{margin:5px 0}details{margin-top:12px}summary{cursor:pointer;color:#496676}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:300px;overflow:auto;font-size:11px;background:#f3f6f7;padding:12px}footer{padding-top:20px;font-size:11px;color:var(--muted)}[hidden]{display:none!important}@media(max-width:1000px){.controls{grid-template-columns:repeat(3,minmax(120px,1fr))}.two-column{grid-template-columns:1fr}canvas{height:350px}}@media(max-width:700px){header{padding:24px 17px}h1{font-size:24px}main{padding:15px 12px}.controls{grid-template-columns:repeat(2,minmax(110px,1fr));padding:13px;gap:11px}.maps{grid-template-columns:1fr}.stats{grid-template-columns:repeat(3,1fr)}.timeline{gap:7px;padding:12px}.timeline button{padding:8px}.map-title{font-size:16px}canvas{height:360px}.section{padding:13px}}
</style>
</head>
<body>
<header>
<div class="eyebrow">SEISMOFLUX · 地点 / 时间 / 震级</div>
<h1>SeismoFlux｜历史预测回放</h1>
<p>同一时间窗、同一批地震、相同报警面积预算：对照新增命中，也保留丢失命中。</p>
<div class="status">较晚历史开发；非独立测试、非真实前瞻</div>
</header>
<main>
<section class="controls card" aria-label="历史回放选择">
<div><label for="fold-select">较晚评价年份组</label><select id="fold-select"></select></div>
<div><label for="horizon-select">预报时限</label><select id="horizon-select"></select></div>
<div><label for="band-select">目标震级</label><select id="band-select"><option value="Ms5_6">5 ≤ Ms &lt; 6</option><option value="Ms6_plus">Ms ≥ 6</option></select></div>
<div><label for="axis-select">起报轴</label><select id="axis-select"><option value="primary">主起报（非重叠评价轴）</option><option value="all">全部报告（描述性回放）</option></select></div>
<div><label for="contrast-select">候选与参考模型</label><select id="contrast-select"></select></div>
<div><label for="budget-select">共同报警面积预算</label><select id="budget-select"></select></div>
<div><label for="mode-select">保存的命中判定</label><select id="mode-select"><option value="strict">严格：事件在报警网格内</option><option value="secondary_70km">70 km 辅助判定</option></select></div>
<div><label for="event-view-select">事件视图</label><select id="event-view-select"><option value="anchor">震序锚点</option><option value="all">全部目标事件</option><option value="subsequent">非锚点后续事件</option></select></div>
</section>
<section class="timeline card" aria-label="起报日期导航">
<button id="frame-prev" type="button" aria-label="上一个起报日期">← 上一期</button>
<div class="timeline-field"><label for="frame-select">起报日期（默认第一合法起报，不挑选命中案例）</label><select id="frame-select"></select><div id="frame-position" class="timeline-meta"></div></div>
<button id="frame-next" type="button" aria-label="下一个起报日期">下一期 →</button>
</section>
<div id="window-info" class="window-info" aria-live="polite"></div>
<div id="empty-notice" class="warning" hidden></div>
<div class="timeline-meta"><button id="focus-event" type="button" disabled>查看所选震例附近</button> <button id="national-view" type="button" disabled>返回全国</button> <span id="viewport-note">全国视域；点击事件后可查看周围 300 km。只改变显示，不改变报警范围或命中。</span></div>
<section class="maps" aria-label="相同全国视域的报警对比">
<article class="card map-card"><div class="map-head"><div class="map-role">参考模型</div><div id="reference-title" class="map-title"></div><div id="reference-area" class="map-area"></div></div><canvas id="reference-map" tabindex="0" aria-label="参考模型全国报警网格与历史地震，事件明细见下表"></canvas><div class="map-caption">全国等面积投影视图 · 金色网格为已保存的报警范围</div></article>
<article class="card map-card"><div class="map-head"><div class="map-role">候选模型</div><div id="candidate-title" class="map-title"></div><div id="candidate-area" class="map-area"></div></div><canvas id="candidate-map" tabindex="0" aria-label="候选模型全国报警网格与历史地震，事件明细见下表"></canvas><div class="map-caption">与左图共享范围与比例 · 点击事件或使用下方事件表</div></article>
</section>
<div class="legend" aria-label="图例"><span><i class="alarm-key"></i>实际报警网格</span><span><i class="mark gain">▲</i>新增命中</span><span><i class="mark loss">■</i>丢失命中</span><span><i class="mark both">●</i>两者命中</span><span><i class="mark miss">◆</i>两者漏报</span><span><i class="mark na">×</i>未评价</span></div>
<div class="stats" aria-label="本窗口当前事件视图的对照计数"><div class="stat"><b id="stat-total">0</b><span>当前视图事件</span></div><div class="stat"><b id="stat-gained" class="gain">0</b><span>新增命中</span></div><div class="stat"><b id="stat-lost" class="loss">0</b><span>丢失命中</span></div><div class="stat"><b id="stat-both" class="both">0</b><span>两者命中</span></div><div class="stat"><b id="stat-miss" class="miss">0</b><span>两者漏报</span></div></div>
<section class="section card">
<h2>本窗口的事件明细</h2>
<p>分类直接读取已保存的两组命中数组，不在页面重新判定。以下计数仅对应当前窗口与视图，不是全项目成绩。</p>
<div class="table-wrap"><table><thead><tr><th>发震时刻（含时区）</th><th>Ms</th><th>经度 / 纬度</th><th>震序角色</th><th>参考模型</th><th>候选模型</th><th>配对变化</th></tr></thead><tbody id="events-body"></tbody></table></div>
<div id="event-detail" class="detail">点击地图中的事件，或点击表格中的发震时刻，查看这一事件的保存结果。</div>
</section>
<div class="two-column">
<section class="section card"><h2>本时窗的次数预测</h2><p>展示本震级档全部已保存次数模型。期望次数是模型的平均次数预测，不是“必然会发生几次”；实际次数含全部目标事件，不随锚点视图切换。</p><div class="table-wrap"><table><thead><tr><th>次数模型</th><th>期望次数</th><th>实际次数</th></tr></thead><tbody id="counts-body"></tbody></table></div></section>
<section class="section card notes"><h2>如何阅读这份回放</h2><ul><li>主起报与全部报告有重叠差别，回放窗口不是独立地震样本。</li><li>全部报告仅供描述；不能把重叠窗口的命中次数相加当作新增独立震例。</li><li>报警网格并非最终≤10区域产品。</li><li>70km仅辅助判定，图上实际报警面积不扩张。</li><li>局部改善可以保留为候选；单个好的震例不等于普遍有效。</li><li>此页只读本机，不联网、不训练、不修改参数、不发布预测。</li></ul><ul id="payload-notes"></ul><details><summary>数据来源与保存依据</summary><pre id="provenance"></pre></details><details><summary>当前不可评价的任务</summary><ul id="unevaluable-list"></ul></details></section>
</div>
<footer id="footer-note"></footer>
</main>
<script id="replay-data" type="application/json">__REPLAY_PAYLOAD__</script>
<script>
"use strict";
(() => {
const data = JSON.parse(document.getElementById("replay-data").textContent);
const $ = id => document.getElementById(id);
const ids = ["fold-select","horizon-select","band-select","axis-select","contrast-select","budget-select","mode-select","event-view-select","frame-select"];
const controls = Object.fromEntries(ids.map(id => [id,$(id)]));
const colors = {gained:"#087c72",lost:"#bb493e",both_hit:"#356ca8",both_miss:"#74828b",unevaluated:"#9c64a3"};
const words = {gained:"▲ 新增命中",lost:"■ 丢失命中",both_hit:"● 两者命中",both_miss:"◆ 两者漏报",unevaluated:"× 未评价"};
const number = (v,d=0) => typeof v === "number" && Number.isFinite(v) ? v.toLocaleString("zh-CN",{maximumFractionDigits:d}) : "—";
const time = v => v ? String(v).replace("T"," ").replace(/Z$/," UTC") : "—";
const name = id => data.variants?.[id] || id || "未提供";
const make = (tag,text) => {const node=document.createElement(tag);if(text!==undefined)node.textContent=String(text);return node;};
function options(select,items,preferred){select.replaceChildren();for(const item of items){const o=make("option",item.label);o.value=String(item.value);select.append(o);}if(items.some(i=>String(i.value)===String(preferred)))select.value=String(preferred);}
const frames = [...(data.frames || [])].sort((a,b)=>String(a.issue_time_utc).localeCompare(String(b.issue_time_utc)) || String(a.id).localeCompare(String(b.id)));
const folds = [...new Set([...frames.map(f=>f.fold_id),...(data.unevaluable||[]).map(f=>f.fold_id)].filter(Boolean))];
options($("fold-select"),folds.map(v=>({value:v,label:v.replace("A_DEV_","开发评价 ").replaceAll("_","–")})),folds[0]);
options($("horizon-select"),[7,30,90,180,365].map(v=>({value:v,label:v+" 天"+(v===365?"（NA／不可评价）":"")})),30);
options($("contrast-select"),(data.contrasts||[]).map(v=>({value:v.id,label:v.label || name(v.candidate)+" 对 "+name(v.reference)})),"CAT_DYN_minus_CAT_COV");
options($("budget-select"),(data.budgets||[]).map(v=>({value:v,label:number(v/10000,1)+" 万 km²"})),600000);
for(const note of data.notes||[])$("payload-notes").append(make("li",note));
$("provenance").textContent=JSON.stringify(data.provenance||{},null,2);
for(const task of data.unevaluable||[])$("unevaluable-list").append(make("li",task.fold_id+" · "+task.horizon_days+" 天："+task.reason));
if(!(data.unevaluable||[]).length)$("unevaluable-list").append(make("li","未另列不可评价任务；缺失保存结果不会视作命中或漏报。"));
$("footer-note").textContent="只读本地历史开发回放 · "+String(data.version||"")+" · 地图与事件表包含位置记录，请仅在本机使用。";
let eligibleFrames=[],currentFrame=null,visibleEvents=[],selectedEvent=null,viewBounds=null;
const mapCaches=new Map();
const mapPoints=new Map();
const budget=()=>Number($("budget-select").value);
const contrast=()=>(data.contrasts||[]).find(c=>c.id===$("contrast-select").value)||{};
const savedAlarm=variant=>(currentFrame?.models?.[variant]?.alarms||[]).find(a=>Number(a.area_budget_km2)===budget());
const savedOutcome=variant=>(currentFrame?.bands?.[$("band-select").value]?.outcomes?.[variant]||[]).find(a=>Number(a.area_budget_km2)===budget());
function booleanOrNull(v){return v===true||v===1?true:v===false||v===0?false:null;}
function hit(outcome,index){return booleanOrNull(outcome?.[$("mode-select").value==="strict"?"strict_hits":"secondary_70km_hits"]?.[index]);}
function classify(ref,candidate){if(ref===null||candidate===null)return "unevaluated";return candidate?(ref?"both_hit":"gained"):(ref?"lost":"both_miss");}
function refreshFrames(){
 const previous=$("frame-select").value;
 eligibleFrames=frames.filter(f=>f.fold_id===$("fold-select").value && Number(f.horizon_days)===Number($("horizon-select").value) && ($("axis-select").value==="all"||f.primary_nonoverlap===true));
 options($("frame-select"),eligibleFrames.map(f=>({value:f.id,label:time(f.issue_time_utc)})),previous);
 if(!eligibleFrames.length)options($("frame-select"),[{value:"",label:"此组合没有可评价起报"}],"");
 render();
}
function writeCell(row,text){row.append(make("td",text));}
function selectEvent(id){selectedEvent=id;const item=visibleEvents.find(e=>e.id===id);if(!item)return;const e=item.event;
 $("focus-event").disabled=!Number.isFinite(e.x_m)||!Number.isFinite(e.y_m);
 $("event-detail").textContent=["事件："+item.id,"发震时刻："+time(e.origin_time_utc)+"；Ms "+number(e.magnitude,1),"经纬度："+number(e.longitude,3)+"° / "+number(e.latitude,3)+"°；"+(item.anchor?"震序锚点":"非锚点后续事件"),"参考："+hitWord(item.reference)+"；候选："+hitWord(item.candidate)+"；"+words[item.category],"起报："+time(currentFrame.issue_time_utc)+"；目标结束："+time(currentFrame.target_end_utc)].join("\n");
 for(const row of $("events-body").children)row.classList.toggle("selected",row.dataset.eventId===id);drawMaps();
}
const hitWord=v=>v===true?"命中":v===false?"漏报":"未评价";
function renderEvents(){
 const band=currentFrame?.bands?.[$("band-select").value];const c=contrast();const ref=savedOutcome(c.reference),cand=savedOutcome(c.candidate);const view=$("event-view-select").value;
 visibleEvents=(band?.event_ids||[]).map((id,index)=>{const anchor=booleanOrNull(band.anchor_mask?.[index])===true;const reference=hit(ref,index),candidate=hit(cand,index);return{id,index,event:data.events?.[id]||{},anchor,reference,candidate,category:classify(reference,candidate)};}).filter(e=>view==="all"||(view==="anchor"?e.anchor:!e.anchor));
 $("events-body").replaceChildren();
 const tally={gained:0,lost:0,both_hit:0,both_miss:0,unevaluated:0};
 for(const item of visibleEvents){tally[item.category]++;const e=item.event,row=make("tr");row.dataset.eventId=item.id;const cell=make("td"),button=make("button",time(e.origin_time_utc));button.type="button";button.className="event-button";button.addEventListener("click",()=>selectEvent(item.id));cell.append(button);row.append(cell);writeCell(row,number(e.magnitude,1));writeCell(row,number(e.longitude,3)+" / "+number(e.latitude,3));writeCell(row,item.anchor?"锚点":"后续事件");writeCell(row,hitWord(item.reference));writeCell(row,hitWord(item.candidate));const status=make("td",words[item.category]);status.style.color=colors[item.category];row.append(status);$("events-body").append(row);}
 if(!visibleEvents.length){const row=make("tr"),cell=make("td",currentFrame?"此窗口在当前事件视图下没有目标事件；空窗口仍保留在起报列表中。":"本组合没有可评价的保存结果。");cell.colSpan=7;row.append(cell);$("events-body").append(row);}
 $("stat-total").textContent=number(visibleEvents.length);$("stat-gained").textContent=number(tally.gained);$("stat-lost").textContent=number(tally.lost);$("stat-both").textContent=number(tally.both_hit);$("stat-miss").textContent=number(tally.both_miss);
 $("stat-total").title=tally.unevaluated?"其中 "+tally.unevaluated+" 个未评价，不计入命中或漏报。":"当前窗口事件数，不是跨窗口独立样本总数。";
 selectedEvent=null;$("focus-event").disabled=true;$("event-detail").textContent=tally.unevaluated?"当前有 "+tally.unevaluated+" 个事件缺少配对保存结果，显示为“未评价”，没有当作漏报。":"点击地图中的事件，或点击表格中的发震时刻，查看这一事件的保存结果。";
}
function renderCounts(){
 $("counts-body").replaceChildren();const counts=currentFrame?.bands?.[$("band-select").value]?.counts||{};
 for(const [variant,result] of Object.entries(counts)){const row=make("tr");writeCell(row,name(variant));writeCell(row,number(result?.expected_count,3));writeCell(row,number(result?.observed_count));$("counts-body").append(row);}
 if(!Object.keys(counts).length){const row=make("tr"),cell=make("td","此任务未提供可评价的次数预测。");cell.colSpan=3;row.append(cell);$("counts-body").append(row);}
}
function render(){
 viewBounds=null;$("national-view").disabled=true;$("viewport-note").textContent="全国视域；点击事件后可查看周围 300 km。只改变显示，不改变报警范围或命中。";
 currentFrame=eligibleFrames.find(f=>String(f.id)===$("frame-select").value)||null;
 const index=eligibleFrames.indexOf(currentFrame);$("frame-prev").disabled=index<=0;$("frame-next").disabled=index<0||index>=eligibleFrames.length-1;
 $("frame-position").textContent=currentFrame?"第 "+(index+1)+" / "+eligibleFrames.length+" 期 · "+($("axis-select").value==="primary"?"主起报评价轴":"全部报告描述轴；窗口可能重叠"):"没有起报被隐藏为成功案例。";
 const c=contrast();$("reference-title").textContent=name(c.reference);$("candidate-title").textContent=name(c.candidate);
 for(const [role,variant] of [["reference",c.reference],["candidate",c.candidate]]){const alarm=savedAlarm(variant);$(role+"-area").textContent="共同预算 "+number(budget())+" km² · 实际报警面积 "+number(alarm?.actual_area_km2,1)+" km²";}
 $("window-info").textContent=currentFrame?"起报 "+time(currentFrame.issue_time_utc)+" → 目标结束 "+time(currentFrame.target_end_utc)+" · "+$("band-select").selectedOptions[0].textContent:"当前任务不可评价（NA），不显示零成绩。";
 const reasons=(data.unevaluable||[]).filter(t=>t.fold_id===$("fold-select").value&&Number(t.horizon_days)===Number($("horizon-select").value));
 $("empty-notice").hidden=Boolean(currentFrame);$("empty-notice").textContent=reasons.length?reasons.map(t=>t.reason).join("；"):"该年份组、时限与起报轴尚无可评价的保存结果。空白不是预测失败。";
 renderEvents();renderCounts();drawMaps();
}
function mapCache(canvas){
 const rect=canvas.getBoundingClientRect(),width=Math.max(200,Math.round(rect.width)),height=Math.max(200,Math.round(rect.height));const ratio=Math.min(window.devicePixelRatio||1,2);
 canvas.width=Math.round(width*ratio);canvas.height=Math.round(height*ratio);
 const bounds=viewBounds||data.geometry?.bounds||[0,0,1,1];
 const key=width+"x"+height+"@"+ratio+":"+bounds.join(",");if(mapCaches.has(key))return mapCaches.get(key);
 const spanX=bounds[2]-bounds[0]||1,spanY=bounds[3]-bounds[1]||1,scale=Math.min((width-32)/spanX,(height-30)/spanY),left=(width-spanX*scale)/2,top=(height-spanY*scale)/2;
 const xy=(x,y)=>[left+(x-bounds[0])*scale,top+(bounds[3]-y)*scale];
 const base=document.createElement("canvas");base.width=canvas.width;base.height=canvas.height;const ctx=base.getContext("2d");ctx.scale(ratio,ratio);ctx.fillStyle="#f7fafb";ctx.fillRect(0,0,width,height);const paths=[];
 for(const cell of data.geometry?.cells||[]){const path=new Path2D();for(const polygon of cell||[])for(const ring of polygon||[]){let first=true;for(const point of ring||[]){if(!Array.isArray(point)||point.length<2)continue;const p=xy(point[0],point[1]);if(first){path.moveTo(...p);first=false;}else path.lineTo(...p);}if(!first)path.closePath();}paths.push(path);ctx.fillStyle="#e8edef";ctx.fill(path,"evenodd");ctx.strokeStyle="#d7e0e4";ctx.lineWidth=.25;ctx.stroke(path);}
 const cached={width,height,ratio,xy,base,paths};if(mapCaches.size>3)mapCaches.clear();mapCaches.set(key,cached);return cached;
}
function marker(ctx,x,y,category,size){ctx.beginPath();if(category==="gained"){ctx.moveTo(x,y-size);ctx.lineTo(x+size,y+size);ctx.lineTo(x-size,y+size);ctx.closePath();}else if(category==="lost"){ctx.rect(x-size,y-size,size*2,size*2);}else if(category==="both_miss"){ctx.moveTo(x,y-size);ctx.lineTo(x+size,y);ctx.lineTo(x,y+size);ctx.lineTo(x-size,y);ctx.closePath();}else if(category==="unevaluated"){ctx.moveTo(x-size,y-size);ctx.lineTo(x+size,y+size);ctx.moveTo(x-size,y+size);ctx.lineTo(x+size,y-size);}else ctx.arc(x,y,size,0,Math.PI*2);ctx.fillStyle=colors[category];ctx.strokeStyle=category==="unevaluated"?colors[category]:"#fff";ctx.lineWidth=category==="unevaluated"?2:1.2;if(category!=="unevaluated")ctx.fill();ctx.stroke();}
function drawMap(id,variant){
 const canvas=$(id),cached=mapCache(canvas),ctx=canvas.getContext("2d");ctx.drawImage(cached.base,0,0);ctx.scale(cached.ratio,cached.ratio);
 const alarm=savedAlarm(variant);for(const index of alarm?.selected||[]){const path=cached.paths[index];if(!path)continue;ctx.fillStyle="#e8c97f";ctx.fill(path,"evenodd");ctx.strokeStyle="#b39040";ctx.lineWidth=.38;ctx.stroke(path);}
 const points=[];for(const item of visibleEvents){const e=item.event;if(!Number.isFinite(e.x_m)||!Number.isFinite(e.y_m))continue;const [x,y]=cached.xy(e.x_m,e.y_m);const size=Math.max(4.3,Math.min(7.5,4.3+(Number(e.magnitude)||5)-5));if(item.id===selectedEvent){ctx.beginPath();ctx.arc(x,y,size+4,0,Math.PI*2);ctx.strokeStyle="#152e3e";ctx.lineWidth=2;ctx.stroke();}marker(ctx,x,y,item.category,size);points.push({id:item.id,x,y});}mapPoints.set(id,points);
 if(!currentFrame){ctx.fillStyle="#496170";ctx.font='15px "Microsoft YaHei",Arial,sans-serif';ctx.textAlign="center";ctx.fillText("NA · 没有可评价的预测窗口",cached.width/2,cached.height/2);}
}
function drawMaps(){const c=contrast();drawMap("reference-map",c.reference);drawMap("candidate-map",c.candidate);}
for(const id of ["reference-map","candidate-map"])$(id).addEventListener("click",event=>{const rect=$(id).getBoundingClientRect(),x=event.clientX-rect.left,y=event.clientY-rect.top;let nearest=null,distance=Infinity;for(const p of mapPoints.get(id)||[]){const d=(p.x-x)**2+(p.y-y)**2;if(d<distance){nearest=p;distance=d;}}if(nearest&&distance<=225)selectEvent(nearest.id);});
for(const id of ids)controls[id].addEventListener("change",()=>["fold-select","horizon-select","axis-select"].includes(id)?refreshFrames():render());
$("focus-event").addEventListener("click",()=>{const e=visibleEvents.find(v=>v.id===selectedEvent)?.event;if(!Number.isFinite(e?.x_m)||!Number.isFinite(e?.y_m))return;viewBounds=[e.x_m-300000,e.y_m-300000,e.x_m+300000,e.y_m+300000];$("national-view").disabled=false;$("viewport-note").textContent="所选震例周围 300 km 的显示视窗，左右同步；报警面积和保存的命中判定均未改变。";drawMaps();});
$("national-view").addEventListener("click",()=>{viewBounds=null;$("national-view").disabled=true;$("viewport-note").textContent="全国视域；点击事件后可查看周围 300 km。只改变显示，不改变报警范围或命中。";drawMaps();});
$("frame-prev").addEventListener("click",()=>{if($("frame-select").selectedIndex>0){$("frame-select").selectedIndex--;render();}});
$("frame-next").addEventListener("click",()=>{if($("frame-select").selectedIndex<eligibleFrames.length-1){$("frame-select").selectedIndex++;render();}});
let resizeTimer;window.addEventListener("resize",()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(drawMaps,150);});
refreshFrames();
})();
</script>
</body>
</html>
"""
