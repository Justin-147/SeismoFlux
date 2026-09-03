"""Execute the offline replay controls using synthetic data and a strict DOM mock."""

# The embedded runtime assertions intentionally preserve UI copy and compact JS.
# ruff: noqa: E501, RUF001

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any

import pytest

from seismoflux.multitask_s3.replay_html import render_replay_html


def _payload() -> dict[str, Any]:
    budgets = [300000, 600000]
    models = {
        variant: {
            "alarms": [
                {"area_budget_km2": budget, "actual_area_km2": budget - 1000, "selected": [0]}
                for budget in budgets
            ]
        }
        for variant in ("CAT_COV", "CAT_DYN", "CAT_SNAP")
    }

    def band(ids: list[str], anchors: list[bool]) -> dict[str, Any]:
        hits = {"CAT_COV": [False, True], "CAT_DYN": [True, False], "CAT_SNAP": [False, False]}
        return {
            "event_ids": ids,
            "anchor_mask": anchors,
            "outcomes": {
                variant: [
                    {
                        "area_budget_km2": budget,
                        "strict_hits": (values if budget == 600000 else [False, False])[:len(ids)],
                        "secondary_70km_hits": [True] * len(ids),
                    }
                    for budget in budgets
                ]
                for variant, values in hits.items()
            },
            "counts": {"T0": {"expected_count": 0.75, "observed_count": len(ids)}},
        }

    def frame(identifier: str, fold: str, horizon: int, day: str, *, empty: bool = False,
              primary: bool = True) -> dict[str, Any]:
        return {
            "id": identifier, "fold_id": fold, "horizon_days": horizon,
            "issue_time_utc": day + "T00:00:00Z", "target_end_utc": "2024-12-01T00:00:00Z",
            "primary_nonoverlap": primary, "models": models,
            "bands": {
                "Ms5_6": band([] if empty else ["anchor", "later"], [] if empty else [True, False]),
                "Ms6_plus": band([] if empty else ["large"], [] if empty else [True]),
            },
        }

    frames = [
        frame("empty-first", "A_DEV_2023_2024", 30, "2023-01-05", empty=True),
        frame("with-events", "A_DEV_2023_2024", 30, "2023-02-09"),
        frame("overlap", "A_DEV_2023_2024", 30, "2023-02-16", primary=False),
        frame("fold-two", "A_DEV_2024_2025", 30, "2024-02-08"),
    ]
    frames.extend(frame(f"h{h}", "A_DEV_2023_2024", h, "2023-03-02") for h in (7, 90, 180))
    return {
        "version": "synthetic-runtime", "local_only": True, "budgets": budgets,
        "contrasts": [
            {"id": "CAT_DYN_minus_CAT_COV", "candidate": "CAT_DYN", "reference": "CAT_COV"},
            {"id": "CAT_SNAP_minus_CAT_COV", "candidate": "CAT_SNAP", "reference": "CAT_COV"},
        ],
        "variants": {"CAT_DYN": "动态", "CAT_COV": "覆盖", "CAT_SNAP": "快照", "T0": "次数背景"},
        "geometry": {
            "bounds": [0, 0, 2000000, 2000000],
            "cells": [[[[[0, 0], [200000, 0], [200000, 200000], [0, 0]]]]],
        },
        "events": {
            key: {
                "origin_time_utc": "2023-02-10T00:00:00Z", "magnitude": magnitude,
                "longitude": 100, "latitude": 30, "x_m": x, "y_m": 100000,
            }
            for key, magnitude, x in (("anchor", 5.2, 100000), ("later", 5.1, 150000), ("large", 6.2, 200000))
        },
        # Deliberately reverse storage order: the first display must be chronological, not best.
        "frames": list(reversed(frames)), "notes": ["仅合成事件"], "provenance": {},
        "unevaluable": [
            {"fold_id": fold, "horizon_days": 365, "reason": "365 天样本尚未成熟"}
            for fold in ("A_DEV_2023_2024", "A_DEV_2024_2025")
        ],
    }


_RUNTIME = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');
const html=fs.readFileSync(0,'utf8');
class Element {
 constructor(tag,id='') {
  this.tagName=tag;this.id=id;this.children=[];this.dataset={};this.style={};
  this.listeners={};this.disabled=false;this.hidden=false;this._value='';this._text='';
  this.classList={toggle(){}};this.ctx=null;
 }
 set textContent(value){this._text=String(value);this.children=[];}
 get textContent(){return this._text+this.children.map(x=>x.textContent).join('');}
 append(...nodes){this.children.push(...nodes);}
 replaceChildren(...nodes){this.children=[...nodes];this._text='';this._value='';}
 get options(){return this.children;}
 get value(){if(this.tagName!=='select')return this._value;return this._value||this.children[0]?.value||'';}
 set value(value){this._value=String(value);}
 get selectedIndex(){return this.children.findIndex(x=>x.value===this.value);}
 set selectedIndex(index){this.value=this.children[index]?.value||'';}
 get selectedOptions(){return this.children.filter(x=>x.value===this.value);}
 addEventListener(kind,callback){(this.listeners[kind]??=[]).push(callback);}
 fire(kind,event={}){for(const callback of this.listeners[kind]||[])callback(event);}
 getBoundingClientRect(){return {width:600,height:430,left:0,top:0};}
 set width(value){this._width=value;if(this.ctx)this.ctx.calls=[];}
 get width(){return this._width;}
 getContext(kind){assert.equal(kind,'2d');if(!this.ctx){
  const calls=[];this.ctx={calls};
  for(const method of ['scale','fillRect','fill','stroke','drawImage','beginPath','moveTo','lineTo','closePath','rect','arc','fillText'])
   this.ctx[method]=(...args)=>this.ctx.calls.push([method,...args]);
 }return this.ctx;}
}
const elements=new Map();
for(const match of html.matchAll(/<([a-z][a-z0-9]*)\b[^>]*\bid="([^"]+)"[^>]*>/gi)){
 assert(!elements.has(match[2]),'duplicate HTML id '+match[2]);
 elements.set(match[2],new Element(match[1].toLowerCase(),match[2]));
}
const document={getElementById(id){assert(elements.has(id),'missing real HTML id '+id);return elements.get(id);},
 createElement(tag){return new Element(tag);}};
const el=id=>document.getElementById(id);
for(const select of html.matchAll(/<select\b[^>]*\bid="([^"]+)"[^>]*>([\s\S]*?)<\/select>/g))
 for(const option of select[2].matchAll(/<option value="([^"]+)"[^>]*>([\s\S]*?)<\/option>/g)){
  const node=new Element('option');node.value=option[1];node.textContent=option[2];el(select[1]).append(node);
 }
const scripts=[...html.matchAll(/<script\b([^>]*)>([\s\S]*?)<\/script>/g)];
assert.equal(scripts.length,2);el('replay-data').textContent=scripts[0][2];
class Path2D {moveTo(){}lineTo(){}closePath(){}}
const context=vm.createContext({document,Path2D,console,setTimeout,clearTimeout,
 window:{devicePixelRatio:1,addEventListener(){}}});
vm.runInContext(scripts[1][2],context,{timeout:10000});
const change=(id,value)=>{el(id).value=value;el(id).fire('change');};
const click=id=>{assert(!el(id).disabled,id+' unexpectedly disabled');el(id).fire('click');};
const text=id=>el(id).textContent;
const eventIds=()=>el('events-body').children.map(row=>row.dataset.eventId).filter(Boolean);
const markerCalls=id=>el(id).ctx.calls.filter(call=>['moveTo','lineTo','rect','arc'].includes(call[0]));
assert.throws(()=>document.getElementById('nonexistent-control'),/missing real HTML id/);
assert.equal(el('frame-select').value,'empty-first');
assert.equal(el('horizon-select').value,'30');assert.equal(el('budget-select').value,'600000');
assert.equal(el('contrast-select').value,'CAT_DYN_minus_CAT_COV');
assert.equal(text('stat-total'),'0');assert.match(text('events-body'),/空窗口仍保留/);
assert.equal(el('frame-select').options.length,2);assert(el('frame-prev').disabled);
click('frame-next');assert.equal(el('frame-select').value,'with-events');
assert.deepEqual(eventIds(),['anchor']);assert.equal(text('stat-gained'),'1');
change('event-view-select','all');assert.deepEqual(eventIds(),['anchor','later']);
assert.equal(text('stat-lost'),'1');
change('event-view-select','subsequent');assert.deepEqual(eventIds(),['later']);
assert.equal(text('stat-lost'),'1');assert.equal(text('stat-gained'),'0');
change('event-view-select','anchor');
const strictArea=text('candidate-area');const strictMarker=markerCalls('candidate-map');
change('mode-select','secondary_70km');assert.equal(text('stat-both'),'1');
assert.equal(text('stat-gained'),'0');assert.equal(text('candidate-area'),strictArea);
assert.notDeepEqual(markerCalls('candidate-map'),strictMarker);
change('mode-select','strict');assert.equal(text('stat-gained'),'1');
change('budget-select','300000');assert.equal(text('stat-miss'),'1');
assert.match(text('candidate-area'),/299,000/);
change('budget-select','600000');assert.equal(text('stat-gained'),'1');
change('contrast-select','CAT_SNAP_minus_CAT_COV');assert.equal(text('candidate-title'),'快照');
assert.equal(text('stat-miss'),'1');
change('contrast-select','CAT_DYN_minus_CAT_COV');assert.equal(text('stat-gained'),'1');
change('band-select','Ms6_plus');assert.deepEqual(eventIds(),['large']);
assert.match(text('window-info'),/Ms/);change('band-select','Ms5_6');
el('events-body').children[0].children[0].children[0].fire('click');
assert.match(text('event-detail'),/事件：anchor/);
const beforeFocus=markerCalls('reference-map');click('focus-event');
assert.match(text('viewport-note'),/左右同步/);
assert.deepEqual(markerCalls('reference-map'),markerCalls('candidate-map'));
assert.notDeepEqual(markerCalls('reference-map'),beforeFocus);
assert.equal(text('candidate-area'),strictArea);assert.equal(text('stat-gained'),'1');
el('candidate-map').fire('click',{clientX:300,clientY:215});
assert.match(text('event-detail'),/事件：anchor/);
click('national-view');assert.deepEqual(markerCalls('reference-map'),beforeFocus);
assert.deepEqual(markerCalls('reference-map'),markerCalls('candidate-map'));
change('axis-select','all');assert.equal(el('frame-select').options.length,3);
change('frame-select','overlap');assert.match(text('window-info'),/2023-02-16/);
assert.match(text('frame-position'),/可能重叠/);
change('axis-select','primary');assert.equal(el('frame-select').value,'empty-first');
click('frame-next');click('frame-prev');assert.equal(el('frame-select').value,'empty-first');
for(const horizon of [7,90,180]){
 change('horizon-select',String(horizon));assert.equal(el('frame-select').value,'h'+horizon);
 assert.equal(text('stat-total'),'1');
}
change('horizon-select','365');assert.match(text('window-info'),/不可评价（NA）/);
assert.match(text('empty-notice'),/365 天样本尚未成熟/);assert(!el('empty-notice').hidden);
assert.match(text('counts-body'),/未提供可评价/);assert.match(text('events-body'),/没有可评价/);
assert(el('reference-map').ctx.calls.some(c=>c[0]==='fillText'&&c[1].includes('NA')));
change('horizon-select','30');change('fold-select','A_DEV_2024_2025');
assert.equal(el('frame-select').value,'fold-two');assert.match(text('window-info'),/2024-02-08/);
console.log('all controls, saved outcomes, empty windows, NA and shared viewport passed');
"""


def test_offline_replay_controls_with_strict_synthetic_dom() -> None:
    node = os.environ.get("SEISMOFLUX_TEST_NODE") or shutil.which("node")
    if node is None:
        pytest.skip("Node unavailable for synthetic offline interaction checks")
    result = subprocess.run(
        [node, "-e", _RUNTIME], input=render_replay_html(_payload()),
        text=True, encoding="utf-8", capture_output=True, timeout=30, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "all controls, saved outcomes, empty windows, NA and shared viewport passed" in result.stdout
