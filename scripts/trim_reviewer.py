import argparse
import json
import mimetypes
import threading
import webbrowser
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict
from urllib.parse import unquote, urlparse

from zalia_ruta.bench.srt_io import parse_srt
from zalia_ruta.bench.trim import (
    ClipItem,
    discover_clips,
    is_effective_trim,
    load_trim_plan,
    save_trim_plan,
)

PLAN_LOCK = threading.Lock()

HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>zalia-ruta Trim Reviewer</title>
<style>
  :root {
    --bg: #0f1218;
    --panel: #171c26;
    --panel-2: #1e2533;
    --border: #2c3548;
    --text: #e6ebf5;
    --muted: #8f9bb3;
    --accent: #4f8cff;
    --green: #2ecc71;
    --amber: #f59e0b;
    --red: #ef4444;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    height: 100vh;
    display: grid;
    grid-template-columns: 320px 1fr;
    overflow: hidden;
  }
  aside {
    background: var(--panel);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    min-height: 0;
  }
  .sidebar-head {
    padding: 12px;
    border-bottom: 1px solid var(--border);
  }
  .sidebar-head h1 {
    font-size: 15px;
    margin: 0 0 8px 0;
  }
  .stats {
    font-size: 12px;
    color: var(--muted);
    margin-bottom: 8px;
  }
  .filters {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-bottom: 8px;
  }
  .filters button, .btn {
    background: var(--panel-2);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 5px 9px;
    font-size: 12px;
    cursor: pointer;
  }
  .filters button.active {
    background: var(--accent);
    border-color: var(--accent);
    color: #fff;
  }
  .search {
    width: 100%;
    padding: 6px 8px;
    border-radius: 6px;
    border: 1px solid var(--border);
    background: var(--bg);
    color: var(--text);
    font-size: 12px;
  }
  .clip-list {
    overflow-y: auto;
    flex: 1;
    padding: 6px;
  }
  .clip-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 6px;
    padding: 7px 8px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 12px;
    border: 1px solid transparent;
  }
  .clip-row:hover { background: var(--panel-2); }
  .clip-row.active {
    background: #1d2b44;
    border-color: var(--accent);
  }
  .clip-name {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .badge {
    font-size: 10px;
    padding: 2px 6px;
    border-radius: 999px;
    font-weight: 600;
    flex-shrink: 0;
  }
  .b-unreviewed { background: #2a3142; color: var(--muted); }
  .b-keep { background: rgba(46, 204, 113, 0.16); color: var(--green); }
  .b-trim { background: rgba(245, 158, 11, 0.2); color: var(--amber); }
  .b-applied { background: rgba(79, 140, 255, 0.2); color: var(--accent); }

  main {
    display: flex;
    flex-direction: column;
    min-height: 0;
    padding: 14px 18px;
    gap: 10px;
    overflow-y: auto;
  }
  .topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
  }
  .title-block h2 {
    margin: 0;
    font-size: 18px;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .bucket-tag {
    font-size: 11px;
    padding: 2px 7px;
    border-radius: 4px;
    background: var(--panel-2);
    border: 1px solid var(--border);
    color: var(--muted);
  }
  .banner {
    padding: 8px 12px;
    border-radius: 6px;
    font-size: 12px;
    border: 1px solid var(--border);
  }
  .banner-info { background: rgba(79, 140, 255, 0.12); border-color: rgba(79, 140, 255, 0.4); }
  .banner-warn { background: rgba(239, 68, 68, 0.15); border-color: rgba(239, 68, 68, 0.5); color: #fecaca; }
  .hidden { display: none !important; }

  .media-box {
    background: #07090d;
    border: 1px solid var(--border);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 80px;
    max-height: 300px;
    overflow: hidden;
    position: relative;
  }
  video {
    max-height: 296px;
    max-width: 100%;
    background: #000;
  }
  .audio-only-label {
    padding: 22px;
    color: var(--muted);
    font-size: 13px;
  }
  .waveform-wrap {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px;
  }
  .wave-controls {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
    flex-wrap: wrap;
    font-size: 12px;
  }
  canvas#wave {
    width: 100%;
    height: 150px;
    display: block;
    cursor: crosshair;
    border-radius: 4px;
    background: #0b0e14;
  }
  .controls-row {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }
  .btn-primary {
    background: var(--accent);
    border-color: var(--accent);
    color: #fff;
    font-weight: 600;
  }
  .btn-green {
    background: #1c7c47;
    border-color: #2ecc71;
    color: #fff;
    font-weight: 600;
  }
  .time-inputs {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    font-size: 13px;
  }
  .time-inputs input[type="number"] {
    width: 92px;
    padding: 4px 6px;
    border-radius: 5px;
    border: 1px solid var(--border);
    background: var(--bg);
    color: var(--text);
    font-family: monospace;
  }
  .note-input {
    flex: 1;
    min-width: 180px;
    padding: 5px 8px;
    border-radius: 5px;
    border: 1px solid var(--border);
    background: var(--bg);
    color: var(--text);
    font-size: 12px;
  }
  .cues-box {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px;
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    max-height: 140px;
    overflow-y: auto;
  }
  .cue-pill {
    font-size: 12px;
    padding: 3px 6px;
    border-radius: 4px;
    background: var(--panel-2);
    border: 1px solid var(--border);
    cursor: pointer;
    user-select: none;
  }
  .cue-pill.current { background: var(--accent); border-color: var(--accent); color: #fff; }
  .cue-pill.outside { opacity: 0.35; text-decoration: line-through; }
  .cue-pill.split { border-color: var(--red); background: rgba(239, 68, 68, 0.25); }
  .shortcuts {
    font-size: 11px;
    color: var(--muted);
    line-height: 1.5;
  }
  kbd {
    background: var(--panel-2);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1px 5px;
    font-family: monospace;
    color: var(--text);
  }
</style>
</head>
<body>
<aside>
  <div class="sidebar-head">
    <h1>Trim Reviewer</h1>
    <div class="stats" id="stats">Loading...</div>
    <div class="filters">
      <button class="active" data-filter="all">All</button>
      <button data-filter="incoming">Incoming</button>
      <button data-filter="benchmark">Benchmark</button>
      <button data-filter="unreviewed">Unreviewed</button>
      <button data-filter="trim">Trimmed</button>
    </div>
    <input class="search" id="search" placeholder="Filter by clip id..." />
  </div>
  <div class="clip-list" id="clipList"></div>
</aside>

<main>
  <div class="topbar">
    <div class="title-block">
      <h2>
        <span id="clipTitle">-</span>
        <span class="bucket-tag" id="clipBucket">-</span>
      </h2>
      <div class="stats" id="clipMeta">-</div>
    </div>
    <div class="controls-row">
      <button class="btn" id="prevBtn">[ Prev</button>
      <button class="btn" id="nextBtn">Next ]</button>
      <button class="btn btn-green" id="saveNextBtn">Save & Next (Enter)</button>
    </div>
  </div>

  <div class="banner banner-info hidden" id="benchBanner">
    Benchmark clip: keep clean leading/trailing silence. Only trim unwanted speech, music, another speaker, or broken edges.
  </div>
  <div class="banner banner-warn hidden" id="collisionBanner"></div>

  <div class="media-box" id="mediaBox">
    <video id="videoPlayer" class="hidden" controls preload="metadata"></video>
    <audio id="audioPlayer" class="hidden" preload="metadata"></audio>
    <div id="audioOnlyLabel" class="audio-only-label hidden">Audio-only clip (use waveform & hotkeys below)</div>
  </div>

  <div class="waveform-wrap">
    <div class="wave-controls">
      <div class="controls-row">
        <button class="btn" id="playBtn">Play/Pause (Space)</button>
        <button class="btn" id="previewBtn">Preview [IN..OUT] (P)</button>
        <button class="btn" id="tailBtn">Preview Tail (Shift+P)</button>
        <span>Zoom:</span>
        <button class="btn zoom-btn active" data-zoom="full">Full (1)</button>
        <button class="btn zoom-btn" data-zoom="start">Start Edge (2)</button>
        <button class="btn zoom-btn" data-zoom="end">End Edge (3)</button>
      </div>
      <div style="font-family: monospace;" id="clockLabel">0.000s / 0.000s</div>
    </div>
    <canvas id="wave" width="1200" height="150"></canvas>
  </div>

  <div class="time-inputs">
    <button class="btn" id="setInBtn">Set IN (I)</button>
    <label>Start: <input type="number" step="0.01" min="0" id="inInput" /></label>
    <button class="btn" id="setOutBtn">Set OUT (O)</button>
    <label>End: <input type="number" step="0.01" min="0" id="outInput" /></label>
    <button class="btn" id="resetBtn">Reset Full (R)</button>
    <span id="trimDelta" style="color: var(--muted); font-family: monospace;"></span>
    <input class="note-input" id="noteInput" placeholder="Optional note..." />
  </div>

  <div class="cues-box" id="cuesBox"></div>

  <div class="shortcuts">
    Hotkeys:
    <kbd>Space</kbd> Play/Pause ·
    <kbd>←</kbd>/<kbd>→</kbd> Nudge ±0.05s (<kbd>Shift</kbd> ±0.01s, <kbd>Ctrl</kbd> ±0.5s) ·
    <kbd>I</kbd> Set IN · <kbd>O</kbd> Set OUT ·
    <kbd>P</kbd> Preview kept range · <kbd>Shift+P</kbd> Preview last 2s ·
    <kbd>1</kbd>/<kbd>2</kbd>/<kbd>3</kbd> Zoom Full/Start/End ·
    <kbd>R</kbd> Reset · <kbd>Enter</kbd> Save & Next · <kbd>[</kbd>/<kbd>]</kbd> Prev/Next
  </div>
</main>

<script>
const state = {
  clips: [],
  filtered: [],
  filter: "all",
  search: "",
  currentKey: null,
  start: 0,
  end: 0,
  zoom: "full",
  waveCache: new Map(),
  peaks: null,
  previewStopAt: null,
  audioCtx: null,
};

const els = {
  stats: document.getElementById("stats"),
  clipList: document.getElementById("clipList"),
  search: document.getElementById("search"),
  clipTitle: document.getElementById("clipTitle"),
  clipBucket: document.getElementById("clipBucket"),
  clipMeta: document.getElementById("clipMeta"),
  benchBanner: document.getElementById("benchBanner"),
  collisionBanner: document.getElementById("collisionBanner"),
  video: document.getElementById("videoPlayer"),
  audio: document.getElementById("audioPlayer"),
  audioOnlyLabel: document.getElementById("audioOnlyLabel"),
  wave: document.getElementById("wave"),
  clockLabel: document.getElementById("clockLabel"),
  inInput: document.getElementById("inInput"),
  outInput: document.getElementById("outInput"),
  noteInput: document.getElementById("noteInput"),
  trimDelta: document.getElementById("trimDelta"),
  cuesBox: document.getElementById("cuesBox"),
};

function activePlayer() {
  const clip = currentClip();
  return clip && clip.has_video ? els.video : els.audio;
}

function currentClip() {
  return state.clips.find(c => c.key === state.currentKey) || null;
}

function effectiveTrim(clip, s, e) {
  return s > 0.005 || e < clip.duration - 0.005;
}

function getZoomWindow(clip) {
  if (!clip) return [0, 1];
  const d = clip.duration;
  if (state.zoom === "start") return [0, Math.min(d, Math.max(4.0, state.start + 2.0))];
  if (state.zoom === "end") return [Math.max(0, Math.min(d - 4.0, state.end - 2.0)), d];
  return [0, d];
}

function findCollisions(clip, s, e) {
  const tol = 0.02;
  const cols = [];
  for (const c of clip.cues) {
    if (c.start + tol < s && s < c.end - tol) {
      cols.push({ side: "start", cue: c, ms: Math.round((s - c.start) * 1000) });
    }
    if (c.start + tol < e && e < c.end - tol) {
      cols.push({ side: "end", cue: c, ms: Math.round((c.end - e) * 1000) });
    }
  }
  return cols;
}

async function fetchClips() {
  const res = await fetch("/api/clips");
  const data = await res.json();
  state.clips = data.clips;
  applyFilters();
  const firstUnrev = state.filtered.find(c => !c.plan || c.plan.status === "unreviewed") || state.filtered[0];
  if (firstUnrev) selectClip(firstUnrev.key);
  updateStats();
}

function updateStats() {
  const total = state.clips.length;
  const inc = state.clips.filter(c => c.bucket === "incoming");
  const ben = state.clips.filter(c => c.bucket === "benchmark");
  const reviewed = state.clips.filter(c => c.plan && c.plan.status && c.plan.status !== "unreviewed").length;
  const trimmed = state.clips.filter(c => c.plan && c.plan.status === "trim").length;
  els.stats.textContent = `${reviewed}/${total} reviewed · ${trimmed} trimmed (In: ${inc.length}, Bench: ${ben.length})`;
}

function applyFilters() {
  const q = state.search.trim().toLowerCase();
  state.filtered = state.clips.filter(c => {
    if (state.filter === "incoming" && c.bucket !== "incoming") return false;
    if (state.filter === "benchmark" && c.bucket !== "benchmark") return false;
    if (state.filter === "unreviewed" && c.plan && c.plan.status !== "unreviewed") return false;
    if (state.filter === "trim" && (!c.plan || c.plan.status !== "trim")) return false;
    if (q && !c.id.toLowerCase().includes(q)) return false;
    return true;
  });
  renderList();
}

function renderList() {
  els.clipList.innerHTML = "";
  for (const c of state.filtered) {
    const row = document.createElement("div");
    row.className = "clip-row" + (c.key === state.currentKey ? " active" : "");
    row.onclick = () => selectClip(c.key);
    const name = document.createElement("span");
    name.className = "clip-name";
    name.textContent = `${c.bucket === "incoming" ? "IN" : "BM"} · ${c.id}`;
    const badge = document.createElement("span");
    const st = (c.plan && c.plan.status) || "unreviewed";
    badge.className = "badge " + (
      st === "trim" ? "b-trim" :
      st === "keep" ? "b-keep" :
      st === "applied" ? "b-applied" : "b-unreviewed"
    );
    badge.textContent = st === "trim"
      ? `${c.plan.start.toFixed(2)}-${c.plan.end.toFixed(2)}`
      : st;
    row.appendChild(name);
    row.appendChild(badge);
    els.clipList.appendChild(row);
  }
}

async function selectClip(key) {
  els.video.pause();
  els.audio.pause();
  state.previewStopAt = null;
  state.currentKey = key;
  const clip = currentClip();
  if (!clip) return;

  state.start = clip.plan && typeof clip.plan.start === "number" ? clip.plan.start : 0;
  state.end = clip.plan && typeof clip.plan.end === "number" ? clip.plan.end : clip.duration;
  els.noteInput.value = (clip.plan && clip.plan.note) || "";
  els.clipTitle.textContent = clip.id;
  els.clipBucket.textContent = clip.bucket;
  els.clipMeta.textContent = `Duration: ${clip.duration.toFixed(3)}s · ${clip.cues.length} words · ${clip.has_video ? "Video + WAV" : "WAV only"}`;
  els.benchBanner.classList.toggle("hidden", clip.bucket !== "benchmark");

  const wavUrl = `/api/media/${clip.bucket}/${encodeURIComponent(clip.id)}.wav`;
  if (clip.has_video) {
    els.video.src = `/api/media/${clip.bucket}/${encodeURIComponent(clip.id)}.mp4`;
    els.video.classList.remove("hidden");
    els.audio.classList.add("hidden");
    els.audioOnlyLabel.classList.add("hidden");
    els.audio.src = "";
  } else {
    els.video.classList.add("hidden");
    els.video.src = "";
    els.audio.src = wavUrl;
    els.audio.classList.remove("hidden");
    els.audioOnlyLabel.classList.remove("hidden");
  }

  syncInputsAndCues();
  renderList();
  const activeRow = els.clipList.querySelector(".clip-row.active");
  if (activeRow) activeRow.scrollIntoView({ block: "nearest" });

  await loadWaveform(clip, wavUrl);
  drawWaveform();
}

async function loadWaveform(clip, wavUrl) {
  if (state.waveCache.has(clip.key)) {
    state.peaks = state.waveCache.get(clip.key);
    return;
  }
  state.peaks = null;
  drawWaveform();
  if (!state.audioCtx) state.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const resp = await fetch(wavUrl);
  const buf = await resp.arrayBuffer();
  const decoded = await state.audioCtx.decodeAudioData(buf);
  const ch = decoded.getChannelData(0);
  const buckets = 2400;
  const blockSize = Math.max(1, Math.floor(ch.length / buckets));
  const peaks = new Float32Array(buckets);
  for (let i = 0; i < buckets; i++) {
    let max = 0;
    const off = i * blockSize;
    const end = Math.min(ch.length, off + blockSize);
    for (let j = off; j < end; j++) {
      const v = Math.abs(ch[j]);
      if (v > max) max = v;
    }
    peaks[i] = max;
  }
  state.waveCache.set(clip.key, peaks);
  if (state.currentKey === clip.key) state.peaks = peaks;
}

function syncInputsAndCues() {
  const clip = currentClip();
  if (!clip) return;
  state.start = Math.max(0, Math.min(state.start, clip.duration - 0.01));
  state.end = Math.max(state.start + 0.01, Math.min(state.end, clip.duration));
  els.inInput.value = state.start.toFixed(3);
  els.outInput.value = state.end.toFixed(3);
  const kept = state.end - state.start;
  els.trimDelta.textContent = effectiveTrim(clip, state.start, state.end)
    ? `Kept: ${kept.toFixed(2)}s (cut -${state.start.toFixed(2)}s front, -${(clip.duration - state.end).toFixed(2)}s end)`
    : `Full clip (${clip.duration.toFixed(2)}s)`;

  const cols = findCollisions(clip, state.start, state.end);
  if (cols.length) {
    els.collisionBanner.classList.remove("hidden");
    els.collisionBanner.textContent = "Boundary slices inside word: " + cols.map(
      x => `${x.side.toUpperCase()} cuts '${x.cue.text}' [${x.cue.start.toFixed(2)}-${x.cue.end.toFixed(2)}] by ${x.ms}ms`
    ).join(" | ");
  } else {
    els.collisionBanner.classList.add("hidden");
  }
  renderCues(cols);
  drawWaveform();
}

function renderCues(cols = []) {
  const clip = currentClip();
  if (!clip) return;
  const splitSet = new Set(cols.map(x => x.cue.index));
  const t = activePlayer().currentTime || 0;
  els.cuesBox.innerHTML = "";
  for (const c of clip.cues) {
    const pill = document.createElement("span");
    pill.className = "cue-pill";
    if (c.end <= state.start + 0.005 || c.start >= state.end - 0.005) pill.classList.add("outside");
    if (splitSet.has(c.index)) pill.classList.add("split");
    if (t >= c.start && t <= c.end) pill.classList.add("current");
    pill.textContent = c.text;
    pill.title = `${c.start.toFixed(3)}s -> ${c.end.toFixed(3)}s`;
    pill.onclick = () => {
      const p = activePlayer();
      p.currentTime = c.start;
    };
    els.cuesBox.appendChild(pill);
  }
}

function drawWaveform() {
  const canvas = els.wave;
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  const clip = currentClip();
  if (!clip) return;
  const [winS, winE] = getZoomWindow(clip);
  const winDur = Math.max(0.01, winE - winS);
  const toX = t => ((t - winS) / winDur) * W;

  const waveH = H - 22;
  // Shaded cut-away areas
  ctx.fillStyle = "rgba(239, 68, 68, 0.18)";
  if (state.start > winS) ctx.fillRect(0, 0, Math.min(W, toX(state.start)), waveH);
  if (state.end < winE) {
    const xEnd = Math.max(0, toX(state.end));
    ctx.fillRect(xEnd, 0, W - xEnd, waveH);
  }

  // Waveform bars
  if (state.peaks) {
    ctx.fillStyle = "#5b8def";
    const mid = waveH / 2;
    const total = state.peaks.length;
    for (let x = 0; x < W; x++) {
      const t0 = winS + (x / W) * winDur;
      const t1 = winS + ((x + 1) / W) * winDur;
      const i0 = Math.max(0, Math.floor((t0 / clip.duration) * total));
      const i1 = Math.min(total - 1, Math.ceil((t1 / clip.duration) * total));
      let peak = 0;
      for (let i = i0; i <= i1; i++) {
        if (state.peaks[i] > peak) peak = state.peaks[i];
      }
      const h = Math.max(1, peak * (waveH * 0.9));
      ctx.fillRect(x, mid - h / 2, 1, h);
    }
  } else {
    ctx.fillStyle = "#8f9bb3";
    ctx.font = "12px sans-serif";
    ctx.fillText("Decoding waveform...", 12, 24);
  }

  // SRT word strip at bottom
  ctx.fillStyle = "#121722";
  ctx.fillRect(0, waveH, W, 22);
  for (const c of clip.cues) {
    if (c.end < winS || c.start > winE) continue;
    const x0 = Math.max(0, toX(c.start));
    const x1 = Math.min(W, toX(c.end));
    const w = Math.max(1, x1 - x0);
    ctx.fillStyle = "#2b3954";
    ctx.fillRect(x0, waveH + 2, w, 18);
    ctx.strokeStyle = "#4f8cff";
    ctx.strokeRect(x0, waveH + 2, w, 18);
    if (w > 22) {
      ctx.fillStyle = "#e6ebf5";
      ctx.font = "10px sans-serif";
      ctx.fillText(c.text, x0 + 3, waveH + 14, w - 5);
    }
  }

  // IN / OUT markers
  const xIn = toX(state.start);
  if (xIn >= 0 && xIn <= W) {
    ctx.strokeStyle = "#2ecc71";
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(xIn, 0); ctx.lineTo(xIn, H); ctx.stroke();
  }
  const xOut = toX(state.end);
  if (xOut >= 0 && xOut <= W) {
    ctx.strokeStyle = "#f59e0b";
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(xOut, 0); ctx.lineTo(xOut, H); ctx.stroke();
  }

  // Playhead
  const t = activePlayer().currentTime || 0;
  const xPlay = toX(t);
  if (xPlay >= 0 && xPlay <= W) {
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(xPlay, 0); ctx.lineTo(xPlay, H); ctx.stroke();
  }
  ctx.lineWidth = 1;
}

function updateClockLoop() {
  const clip = currentClip();
  const p = activePlayer();
  if (clip && p) {
    const t = p.currentTime || 0;
    if (state.previewStopAt !== null && t >= state.previewStopAt) {
      p.pause();
      state.previewStopAt = null;
    }
    els.clockLabel.textContent = `${t.toFixed(3)}s / ${clip.duration.toFixed(3)}s`;
    drawWaveform();
    const pills = els.cuesBox.children;
    for (let i = 0; i < clip.cues.length && i < pills.length; i++) {
      const c = clip.cues[i];
      pills[i].classList.toggle("current", t >= c.start && t <= c.end);
    }
  }
  requestAnimationFrame(updateClockLoop);
}

async function saveCurrent(advance = true) {
  const clip = currentClip();
  if (!clip) return;
  const status = effectiveTrim(clip, state.start, state.end) ? "trim" : "keep";
  const payload = {
    key: clip.key,
    bucket: clip.bucket,
    id: clip.id,
    duration: clip.duration,
    start: Number(state.start.toFixed(3)),
    end: Number(state.end.toFixed(3)),
    status,
    note: els.noteInput.value.trim(),
  };
  const res = await fetch("/api/plan", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const saved = await res.json();
  clip.plan = saved.entry;
  updateStats();
  renderList();
  if (advance) stepClip(1);
}

function stepClip(delta) {
  if (!state.filtered.length) return;
  const idx = state.filtered.findIndex(c => c.key === state.currentKey);
  const nextIdx = Math.max(0, Math.min(state.filtered.length - 1, idx + delta));
  if (nextIdx !== idx) selectClip(state.filtered[nextIdx].key);
}

function togglePlay() {
  const p = activePlayer();
  state.previewStopAt = null;
  if (p.paused) p.play(); else p.pause();
}

function setZoom(mode) {
  state.zoom = mode;
  document.querySelectorAll(".zoom-btn").forEach(b => b.classList.toggle("active", b.dataset.zoom === mode));
  drawWaveform();
}

// Event bindings
els.wave.addEventListener("mousedown", ev => {
  const clip = currentClip();
  if (!clip) return;
  const rect = els.wave.getBoundingClientRect();
  const frac = Math.max(0, Math.min(1, (ev.clientX - rect.left) / rect.width));
  const [winS, winE] = getZoomWindow(clip);
  const t = winS + frac * (winE - winS);
  activePlayer().currentTime = t;
});

document.querySelectorAll(".filters button").forEach(btn => {
  btn.onclick = () => {
    state.filter = btn.dataset.filter;
    document.querySelectorAll(".filters button").forEach(b => b.classList.toggle("active", b === btn));
    applyFilters();
  };
});
document.querySelectorAll(".zoom-btn").forEach(btn => {
  btn.onclick = () => setZoom(btn.dataset.zoom);
});
els.search.oninput = () => { state.search = els.search.value; applyFilters(); };
els.inInput.onchange = () => { state.start = parseFloat(els.inInput.value) || 0; syncInputsAndCues(); };
els.outInput.onchange = () => {
  const clip = currentClip();
  state.end = parseFloat(els.outInput.value) || (clip ? clip.duration : 0);
  syncInputsAndCues();
};

document.getElementById("playBtn").onclick = togglePlay;
document.getElementById("previewBtn").onclick = () => {
  const p = activePlayer();
  p.currentTime = state.start;
  state.previewStopAt = state.end;
  p.play();
};
document.getElementById("tailBtn").onclick = () => {
  const p = activePlayer();
  p.currentTime = Math.max(state.start, state.end - 2.0);
  state.previewStopAt = state.end;
  p.play();
};
document.getElementById("setInBtn").onclick = () => { state.start = activePlayer().currentTime || 0; syncInputsAndCues(); };
document.getElementById("setOutBtn").onclick = () => { state.end = activePlayer().currentTime || 0; syncInputsAndCues(); };
document.getElementById("resetBtn").onclick = () => {
  const clip = currentClip();
  if (!clip) return;
  state.start = 0; state.end = clip.duration;
  syncInputsAndCues();
};
document.getElementById("prevBtn").onclick = () => stepClip(-1);
document.getElementById("nextBtn").onclick = () => stepClip(1);
document.getElementById("saveNextBtn").onclick = () => saveCurrent(true);

window.addEventListener("keydown", ev => {
  if (["INPUT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
  const p = activePlayer();
  if (ev.code === "Space") { ev.preventDefault(); togglePlay(); }
  else if (ev.key === "ArrowLeft" || ev.key === "j" || ev.key === "J") {
    ev.preventDefault();
    const step = ev.ctrlKey ? 0.5 : (ev.shiftKey ? 0.01 : 0.05);
    p.currentTime = Math.max(0, (p.currentTime || 0) - step);
  } else if (ev.key === "ArrowRight" || ev.key === "l" || ev.key === "L") {
    ev.preventDefault();
    const step = ev.ctrlKey ? 0.5 : (ev.shiftKey ? 0.01 : 0.05);
    p.currentTime = Math.min(currentClip()?.duration || 9999, (p.currentTime || 0) + step);
  } else if (ev.key === "i" || ev.key === "I") {
    state.start = p.currentTime || 0; syncInputsAndCues();
  } else if (ev.key === "o" || ev.key === "O") {
    state.end = p.currentTime || 0; syncInputsAndCues();
  } else if (ev.key === "p" || ev.key === "P") {
    if (ev.shiftKey) document.getElementById("tailBtn").click();
    else document.getElementById("previewBtn").click();
  } else if (ev.key === "r" || ev.key === "R") {
    document.getElementById("resetBtn").click();
  } else if (ev.key === "Enter" || ev.key === "k" || ev.key === "K") {
    ev.preventDefault(); saveCurrent(true);
  } else if (ev.key === "[") { stepClip(-1); }
  else if (ev.key === "]") { stepClip(1); }
  else if (ev.key === "1") { setZoom("full"); }
  else if (ev.key === "2") { setZoom("start"); }
  else if (ev.key === "3") { setZoom("end"); }
});

fetchClips();
requestAnimationFrame(updateClockLoop);
</script>
</body>
</html>
"""


def build_clip_payload(clips: list[ClipItem], plan: Dict[str, Any]) -> list[dict]:
    plan_clips = plan.get("clips", {})
    payload = []
    for item in clips:
        cues = []
        if item.srt_path.exists():
            for c in parse_srt(item.srt_path.read_text(encoding="utf-8")):
                cues.append({"index": c.index, "start": c.start, "end": c.end, "text": c.text})
        payload.append(
            {
                "key": item.key,
                "bucket": item.bucket,
                "id": item.id,
                "duration": item.duration,
                "has_video": item.video_path is not None and item.video_path.exists(),
                "cues": cues,
                "plan": plan_clips.get(item.key),
            }
        )
    return payload


def make_handler(clips: list[ClipItem], plan_path: Path):
    clip_map = {c.key: c for c in clips}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # keep terminal clean

        def _send_json(self, obj: Any, status: int = 200):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _serve_file_range(self, file_path: Path):
            if not file_path.exists():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            size = file_path.stat().st_size
            ctype = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
            range_hdr = self.headers.get("Range")
            if range_hdr and range_hdr.startswith("bytes="):
                spec = range_hdr.replace("bytes=", "").split(",")[0].strip()
                start_s, end_s = spec.split("-")
                start = int(start_s) if start_s else 0
                end = int(end_s) if end_s else size - 1
                end = min(end, size - 1)
                length = end - start + 1
                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Type", ctype)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.send_header("Content-Length", str(length))
                self.end_headers()
                with file_path.open("rb") as f:
                    f.seek(start)
                    remaining = length
                    while remaining > 0:
                        chunk = f.read(min(65536, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            else:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", ctype)
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(size))
                self.end_headers()
                with file_path.open("rb") as f:
                    while True:
                        chunk = f.read(65536)
                        if not chunk:
                            break
                        self.wfile.write(chunk)

        def do_GET(self):
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            if path == "/":
                body = HTML_PAGE.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if path == "/api/clips":
                with PLAN_LOCK:
                    plan = load_trim_plan(plan_path)
                self._send_json({"clips": build_clip_payload(clips, plan)})
                return

            if path.startswith("/api/media/"):
                parts = path[len("/api/media/"):].split("/", 1)
                if len(parts) != 2:
                    self.send_error(HTTPStatus.BAD_REQUEST)
                    return
                bucket, filename = parts
                if filename.endswith(".wav"):
                    cid = filename[:-4]
                    item = clip_map.get(f"{bucket}:{cid}")
                    if item:
                        self._serve_file_range(item.wav_path)
                        return
                elif filename.endswith(".mp4"):
                    cid = filename[:-4]
                    item = clip_map.get(f"{bucket}:{cid}")
                    if item and item.video_path:
                        self._serve_file_range(item.video_path)
                        return
                self.send_error(HTTPStatus.NOT_FOUND)
                return

            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self):
            if self.path != "/api/plan":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            body = json.loads(raw.decode("utf-8"))
            key = body["key"]
            item = clip_map.get(key)
            if not item:
                self._send_json({"error": f"Unknown clip {key}"}, status=404)
                return

            start = max(0.0, float(body["start"]))
            end = min(item.duration, float(body["end"]))
            status = "trim" if is_effective_trim(start, end, item.duration) else "keep"
            entry = {
                "bucket": item.bucket,
                "id": item.id,
                "duration": item.duration,
                "start": round(start, 3),
                "end": round(end, 3),
                "status": status,
                "note": str(body.get("note", "")),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            with PLAN_LOCK:
                plan = load_trim_plan(plan_path)
                plan.setdefault("clips", {})[key] = entry
                save_trim_plan(plan_path, plan)
            self._send_json({"ok": True, "entry": entry})

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Interactive local web reviewer for trimming incoming + benchmark clips.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--plan", type=Path, default=Path("trim_plan.json"))
    parser.add_argument("--no-open", action="store_true", help="Do not automatically open the browser.")
    args = parser.parse_args()

    clips = discover_clips(Path("."))
    in_count = sum(1 for c in clips if c.bucket == "incoming")
    bm_count = sum(1 for c in clips if c.bucket == "benchmark")
    print(f"Discovered {len(clips)} clips ({in_count} incoming + {bm_count} benchmark).")
    print(f"Plan file: {args.plan.resolve()}")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(clips, args.plan))
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Serving trim reviewer at {url} (Ctrl+C to stop)")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

