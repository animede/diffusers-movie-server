/* diffusers-movie-server gateway シェルUI(Phase 3)
 *
 * 設計メモ:
 * - /api/v1/status のポーリングは「in-flight ガード付き」: gateway の
 *   manager.load() はヘルス待ちの間ロックを保持するため、その間 status は
 *   応答待ちになる。ポーリングごとに新規リクエストを積むとサーバ側スレッドが
 *   溜まるので、前回の応答が返るまで次を発行しない。
 * - バックエンド起動(load POST)は h3 の 96gb で数分かかりうる。fetch が
 *   ネットワーク都合で切れても、status ポーリングが「対象がアクティブ+
 *   ヘルスOK」を検知した時点で iframe 表示に切り替える。
 * - ジョブ一覧・状態カードは DOM 再構築せず差分更新(ちらつき防止)。
 */
"use strict";

const BACKEND_TABS = ["h3", "ltx25"];
const LABELS = { h3: "MiniMax-H3", ltx25: "LTX-2.5" };
const STATUS_INTERVAL_MS = 3000;
const JOBS_INTERVAL_MS = 5000;

const state = {
  catalog: {},            // backend name -> catalog entry(/api/v1/backends)
  status: null,           // 最新の /api/v1/status
  statusFetching: false,
  jobsFetching: false,
  loading: null,          // {backend, preset, note} 起動要求中
  currentTab: "h3",
  framePid: {},           // backend -> iframe を張った時点の pid
  jobRows: new Map(),     // job id -> {tr, cells...}
  gallery: {              // Phase 5b: ギャラリー(ポーリングなし)
    tiles: new Map(),     // "backend/filename" -> {el, item, ...}
    total: 0,
    fetching: false,
  },
};

function $(id) { return document.getElementById(id); }

async function fetchJSON(url, opts) {
  const res = await fetch(url, opts);
  let body = null;
  try { body = await res.json(); } catch (e) { /* 非JSON応答 */ }
  if (!res.ok) {
    let msg = body && (body.detail || body.error);
    if (msg && typeof msg !== "string") msg = JSON.stringify(msg);
    const err = new Error(msg || ("HTTP " + res.status));
    err.status = res.status;
    throw err;
  }
  return body;
}

// ---------------------------------------------------------------------------
// タブ切替
// ---------------------------------------------------------------------------

function initTabs() {
  $("tabs").addEventListener("click", (ev) => {
    const btn = ev.target.closest(".tab-btn");
    if (!btn) return;
    state.currentTab = btn.dataset.tab;
    document.querySelectorAll(".tab-btn").forEach((b) =>
      b.classList.toggle("active", b === btn));
    document.querySelectorAll(".tab-panel").forEach((p) =>
      p.classList.toggle("active", p.id === "panel-" + state.currentTab));
    renderAll();
    // ギャラリーはポーリングしない: タブ表示時と更新ボタンでのみ取得
    if (state.currentTab === "gallery") refreshGallery();
  });
}

// ---------------------------------------------------------------------------
// バックエンドタブ(オーバーレイ + iframe)
// ---------------------------------------------------------------------------

function backendOrigin(backend) {
  // LAN クライアント対応のためホスト名は動的に組み立てる。ポートはカタログ由来。
  const entry = state.catalog[backend];
  const port = entry ? entry.port : (backend === "h3" ? 8631 : 8632);
  return "http://" + window.location.hostname + ":" + port + "/";
}

function buildOverlay(backend) {
  const entry = state.catalog[backend];
  const box = document.createElement("div");
  box.className = "overlay-box";

  const h2 = document.createElement("h2");
  h2.textContent = LABELS[backend] + " は未起動です";
  box.appendChild(h2);

  const sub = document.createElement("div");
  sub.className = "overlay-sub";
  sub.textContent = entry ? entry.description : "";
  box.appendChild(sub);

  const err = document.createElement("div");
  err.className = "error-text";
  err.id = "ov-error-" + backend;
  err.hidden = true;
  box.appendChild(err);

  // プリセット選択(カタログから動的生成)
  const list = document.createElement("div");
  list.className = "preset-list";
  list.id = "ov-presets-" + backend;
  const presets = entry ? entry.presets : [];
  const defaultPreset = entry ? entry.default_preset : null;
  for (const p of presets) {
    const label = document.createElement("label");
    label.className = "preset-item" + (p.name === defaultPreset ? " selected" : "");
    const radio = document.createElement("input");
    radio.type = "radio";
    radio.name = "preset-" + backend;
    radio.value = p.name;
    radio.checked = p.name === defaultPreset;
    radio.addEventListener("change", () => {
      list.querySelectorAll(".preset-item").forEach((el) =>
        el.classList.toggle("selected", el === label));
    });
    const name = document.createElement("span");
    name.className = "preset-name";
    name.textContent = p.name + (p.name === defaultPreset ? "(既定)" : "");
    const vram = document.createElement("span");
    vram.className = "preset-vram";
    vram.textContent = p.vram_hint || "";
    const desc = document.createElement("div");
    desc.className = "preset-desc";
    desc.textContent = p.description || "";
    label.append(radio, name, vram, desc);
    list.appendChild(label);
  }
  box.appendChild(list);

  if (backend === "h3") {
    const note = document.createElement("div");
    note.className = "overlay-note";
    note.textContent = "注意: 96gb プリセットは起動時に全モデルをプリロードするため、"
      + "起動完了まで数分かかることがあります(初回はモデルダウンロード時間も加算)。";
    box.appendChild(note);
  }

  const warn = document.createElement("div");
  warn.className = "overlay-warn";
  warn.id = "ov-warn-" + backend;
  warn.hidden = true;
  box.appendChild(warn);

  const btn = document.createElement("button");
  btn.className = "btn primary";
  btn.id = "ov-start-" + backend;
  btn.textContent = "起動";
  btn.addEventListener("click", () => onStartClicked(backend));
  box.appendChild(btn);

  const spin = document.createElement("div");
  spin.className = "spinner-row";
  spin.id = "ov-spinner-" + backend;
  spin.hidden = true;
  const s = document.createElement("span");
  s.className = "spinner";
  const st = document.createElement("span");
  st.id = "ov-spinner-text-" + backend;
  st.textContent = "起動中…";
  spin.append(s, st);
  box.appendChild(spin);

  const overlay = $("overlay-" + backend);
  overlay.textContent = "";
  overlay.appendChild(box);
}

function selectedPreset(backend) {
  const checked = document.querySelector(
    'input[name="preset-' + backend + '"]:checked');
  return checked ? checked.value : null;
}

function onStartClicked(backend) {
  const status = state.status || {};
  const active = status.active_backend;
  if (active && active !== backend) {
    const ok = window.confirm(
      "現在の " + (LABELS[active] || active) + " を停止して "
      + LABELS[backend] + " を起動します。よろしいですか?");
    if (!ok) return;
  }
  startBackend(backend, selectedPreset(backend));
}

function selectedStrategy() {
  const sel = $("ctl-strategy");
  return sel ? sel.value : "resident";
}

async function startBackend(backend, preset, strategy) {
  if (state.loading) return;
  strategy = strategy || selectedStrategy();
  state.loading = { backend, preset, note: null };
  const errEl = $("ov-error-" + backend);
  if (errEl) { errEl.hidden = true; errEl.textContent = ""; }
  const ctlErr = $("ctl-error");
  ctlErr.hidden = true;
  renderAll();
  try {
    await fetchJSON("/api/v1/backend/load", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ backend, preset, strategy }),
    });
    state.loading = null;
  } catch (e) {
    if (e.status) {
      // gateway からの明示エラー(400/409/500)
      state.loading = null;
      showStartError(backend, e.message);
    } else {
      // ネットワーク断など。起動自体は継続中の可能性があるため
      // status ポーリングでの検知に切り替える(spinner は出したまま)。
      state.loading.note =
        "応答待ちが中断されました。起動は継続中の可能性があるため状態を監視しています…";
    }
  }
  await refreshStatus();
  renderAll();
}

function showStartError(backend, message) {
  const errEl = $("ov-error-" + backend);
  if (errEl) { errEl.textContent = "起動に失敗しました: " + message; errEl.hidden = false; }
  const ctlErr = $("ctl-error");
  ctlErr.textContent = "起動に失敗しました: " + message;
  ctlErr.hidden = false;
}

function backendReady(backend) {
  const st = state.status;
  return !!(st && st.active_backend === backend && st.backend_health);
}

function renderBackendTab(backend) {
  const overlay = $("overlay-" + backend);
  const frame = $("frame-" + backend);
  const st = state.status || {};
  const loading = state.loading && state.loading.backend === backend;

  if (backendReady(backend)) {
    // 起動検知 → loading 状態を解除して iframe を表示
    if (loading) state.loading = null;
    const pid = st.process ? st.process.pid : null;
    if (frame.hidden || state.framePid[backend] !== pid) {
      state.framePid[backend] = pid;
      frame.src = backendOrigin(backend);
      frame.hidden = false;
    }
    overlay.hidden = true;
    return;
  }

  // 未起動(または外部要因で停止)→ オーバーレイへ戻す
  if (!frame.hidden) {
    frame.hidden = true;
    frame.src = "about:blank";
    state.framePid[backend] = null;
  }
  overlay.hidden = false;

  const btn = $("ov-start-" + backend);
  const warn = $("ov-warn-" + backend);
  const spin = $("ov-spinner-" + backend);
  const spinText = $("ov-spinner-text-" + backend);
  if (!btn) return; // カタログ未取得でオーバーレイ未構築

  const active = st.active_backend;
  const busy = st.busy === true;
  const otherActive = active && active !== backend;

  if (loading) {
    btn.hidden = true;
    warn.hidden = true;
    spin.hidden = false;
    let text = LABELS[backend] + " を起動しています…";
    if (backend === "h3" && state.loading.preset === "96gb") {
      text += "(96gb プリセットは数分かかります)";
    }
    if (state.loading.note) text += " " + state.loading.note;
    spinText.textContent = text;
    return;
  }
  spin.hidden = true;
  btn.hidden = false;

  if (otherActive && busy) {
    btn.disabled = true;
    btn.textContent = "起動";
    warn.textContent = "現在 " + (LABELS[active] || active)
      + " が生成中(busy)のため切り替えできません。完了を待ってください。";
    warn.hidden = false;
  } else if (otherActive) {
    btn.disabled = false;
    btn.textContent = "切替(現在の " + (LABELS[active] || active) + " を停止して起動)";
    warn.textContent = "別のバックエンド(" + (LABELS[active] || active)
      + ")が起動中です。切替時は停止してからの起動になります。";
    warn.hidden = false;
  } else {
    btn.disabled = false;
    btn.textContent = "起動";
    warn.hidden = true;
  }
}

// ---------------------------------------------------------------------------
// 管理タブ: 状態カード
// ---------------------------------------------------------------------------

function renderStatusCard() {
  const st = state.status;
  const errEl = $("st-error");
  if (!st) {
    $("st-active").textContent = "-";
    return;
  }
  errEl.hidden = true;

  const proc = st.process;
  if (st.active_backend && proc) {
    $("st-active").textContent =
      (LABELS[st.active_backend] || st.active_backend)
      + " (" + st.active_backend + ", port " + proc.port + ")"
      + (proc.adopted ? " [adopted]" : "");
    $("st-preset").textContent = proc.preset || "-";
    $("st-pid").textContent = proc.pid + " / " + formatUptime(proc.uptime_s);
  } else {
    $("st-active").textContent = "なし(全バックエンド停止中)";
    $("st-preset").textContent = "-";
    $("st-pid").textContent = "-";
  }
  const busyEl = $("st-busy");
  if (st.busy === true) { busyEl.textContent = "はい(生成中)"; }
  else if (st.busy === false) { busyEl.textContent = "いいえ"; }
  else { busyEl.textContent = st.active_backend ? "不明(ヘルス未応答)" : "-"; }

  // Phase 5a: process alive / weights loaded の2軸表示
  const bk = st.backends || {};
  $("st-backends").textContent = BACKEND_TABS.map((name) => {
    const b = bk[name] || {};
    const procTxt = b.process_alive ? "生存" : "停止";
    const wTxt = b.weights_loaded ? "ロード済" : "未ロード";
    const vram = (b.vram_mb != null) ? " " + (b.vram_mb / 1024).toFixed(1) + "GB" : "";
    return name + ": プロセス" + procTxt + " / 重み" + wTxt + vram;
  }).join(" | ");

  renderVram(st.vram);

  if (st.foreign_listeners) {
    errEl.textContent = "管理外プロセスがポートを占有しています: "
      + JSON.stringify(st.foreign_listeners);
    errEl.hidden = false;
  }
}

function formatUptime(s) {
  if (s == null) return "-";
  s = Math.round(s);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  if (h > 0) return h + "h" + m + "m" + sec + "s";
  if (m > 0) return m + "m" + sec + "s";
  return sec + "s";
}

const vramRows = new Map(); // gpu index -> {row, fill, text}

function renderVram(vram) {
  const wrap = $("st-vram");
  if (!Array.isArray(vram)) {
    if (vramRows.size === 0) wrap.innerHTML = '<span class="muted">取得できません</span>';
    return;
  }
  if (vramRows.size === 0) wrap.textContent = "";
  for (const gpu of vram) {
    let row = vramRows.get(gpu.index);
    if (!row) {
      const div = document.createElement("div");
      div.className = "vram-row";
      const label = document.createElement("span");
      label.className = "vram-label";
      label.textContent = "GPU:" + gpu.index;
      const bar = document.createElement("div");
      bar.className = "vram-bar";
      const fill = document.createElement("div");
      fill.className = "vram-fill";
      bar.appendChild(fill);
      const text = document.createElement("span");
      text.className = "vram-text";
      div.append(label, bar, text);
      wrap.appendChild(div);
      row = { fill, text };
      vramRows.set(gpu.index, row);
    }
    const pct = gpu.memory_total_mb
      ? Math.round(100 * gpu.memory_used_mb / gpu.memory_total_mb) : 0;
    row.fill.style.width = pct + "%";
    row.fill.classList.toggle("hi", pct >= 85);
    row.text.textContent =
      (gpu.memory_used_mb / 1024).toFixed(1) + " / "
      + (gpu.memory_total_mb / 1024).toFixed(1) + " GB(" + pct + "%)";
  }
}

// ---------------------------------------------------------------------------
// 管理タブ: 起動/切替/アンロード
// ---------------------------------------------------------------------------

function initControls() {
  const backendSel = $("ctl-backend");
  for (const name of BACKEND_TABS) {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = LABELS[name] + " (" + name + ")";
    backendSel.appendChild(opt);
  }
  backendSel.addEventListener("change", populateCtlPresets);
  $("ctl-preset").addEventListener("change", updateCtlPresetDesc);

  $("ctl-load").addEventListener("click", () => {
    const backend = backendSel.value;
    const preset = $("ctl-preset").value || null;
    const st = state.status || {};
    if (st.active_backend && st.active_backend !== backend) {
      const ok = window.confirm(
        "現在の " + (LABELS[st.active_backend] || st.active_backend)
        + " を停止して " + LABELS[backend] + " を起動します。よろしいですか?");
      if (!ok) return;
    }
    startBackend(backend, preset);
  });

  $("ctl-unload").addEventListener("click", async () => {
    const ctlErr = $("ctl-error");
    ctlErr.hidden = true;
    try {
      await fetchJSON("/api/v1/backend/unload", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ strategy: selectedStrategy() }),
      });
    } catch (e) {
      ctlErr.textContent = "アンロードに失敗しました: " + e.message;
      ctlErr.hidden = false;
    }
    await refreshStatus();
    renderAll();
  });
}

function populateCtlPresets() {
  const backend = $("ctl-backend").value;
  const entry = state.catalog[backend];
  const sel = $("ctl-preset");
  sel.textContent = "";
  if (!entry) return;
  for (const p of entry.presets) {
    const opt = document.createElement("option");
    opt.value = p.name;
    opt.textContent = p.name + (p.name === entry.default_preset ? "(既定)" : "")
      + (p.vram_hint ? " — " + p.vram_hint : "");
    if (p.name === entry.default_preset) opt.selected = true;
    sel.appendChild(opt);
  }
  updateCtlPresetDesc();
}

function updateCtlPresetDesc() {
  const entry = state.catalog[$("ctl-backend").value];
  const name = $("ctl-preset").value;
  const p = entry && entry.presets.find((x) => x.name === name);
  $("ctl-preset-desc").textContent = p ? (p.description || "") : "";
}

function renderControls() {
  const st = state.status || {};
  const busy = st.busy === true;
  const loading = !!state.loading;
  const loadBtn = $("ctl-load");
  const unloadBtn = $("ctl-unload");
  const note = $("ctl-note");
  const spin = $("ctl-spinner");

  const targetBackend = $("ctl-backend").value;
  const otherActive = st.active_backend && st.active_backend !== targetBackend;
  loadBtn.textContent = otherActive
    ? "切替(" + (LABELS[st.active_backend] || st.active_backend) + " を停止して起動)"
    : "起動";
  loadBtn.disabled = loading || (busy && st.active_backend !== targetBackend);
  unloadBtn.disabled = loading || busy || !st.active_backend;

  const reasons = [];
  if (busy) reasons.push("生成中(busy)のため切替・アンロードはできません。");
  if (!st.active_backend) reasons.push("アクティブなバックエンドはありません(アンロード不要)。");
  note.textContent = reasons.join(" ");

  spin.hidden = !loading;
  if (loading) {
    let text = LABELS[state.loading.backend] + " を起動中…";
    if (state.loading.backend === "h3" && state.loading.preset === "96gb") {
      text += "(数分かかります)";
    }
    if (state.loading.note) text += " " + state.loading.note;
    $("ctl-spinner-text").textContent = text;
  }
}

// ---------------------------------------------------------------------------
// 管理タブ: 統一ジョブ一覧(差分更新)
// ---------------------------------------------------------------------------

function jobRow(job) {
  let row = state.jobRows.get(job.id);
  if (!row) {
    const tr = document.createElement("tr");
    const tdId = document.createElement("td");
    tdId.className = "mono";
    tdId.textContent = job.id.slice(0, 8);
    tdId.title = job.id;
    const tdBackend = document.createElement("td");
    const tdMode = document.createElement("td");
    const tdStatus = document.createElement("td");
    const badge = document.createElement("span");
    badge.className = "status-badge";
    tdStatus.appendChild(badge);
    const tdProgress = document.createElement("td");
    const wrap = document.createElement("div");
    wrap.className = "job-progress";
    const bar = document.createElement("div");
    bar.className = "job-bar";
    const fill = document.createElement("div");
    fill.className = "job-fill";
    bar.appendChild(fill);
    const pct = document.createElement("span");
    pct.className = "job-pct";
    wrap.append(bar, pct);
    tdProgress.appendChild(wrap);
    const tdElapsed = document.createElement("td");
    tdElapsed.className = "mono";
    const tdLinks = document.createElement("td");
    tdLinks.className = "job-links";
    tr.append(tdId, tdBackend, tdMode, tdStatus, tdProgress, tdElapsed, tdLinks);
    row = { tr, tdBackend, tdMode, badge, fill, pct, tdElapsed, tdLinks,
            links: {}, errorShown: false };
    state.jobRows.set(job.id, row);
  }
  // 差分更新(リンクは一度だけ作る)
  row.tdBackend.textContent = job.backend;
  row.tdMode.textContent = job.mode;
  row.badge.textContent = job.status;
  row.badge.className = "status-badge " + job.status;
  const pct = Math.round((job.progress || 0) * 100);
  row.fill.style.width = pct + "%";
  row.pct.textContent = pct + "%";
  row.tdElapsed.textContent = (job.elapsed_s != null ? job.elapsed_s.toFixed(0) : "-") + "s";
  const result = job.result || {};
  for (const [key, label] of [["image_url", "画像"], ["video_url", "動画"], ["audio_url", "音声"]]) {
    if (result[key] && !row.links[key]) {
      const a = document.createElement("a");
      a.href = result[key];
      a.target = "_blank";
      a.textContent = label;
      row.tdLinks.appendChild(a);
      row.links[key] = a;
    }
  }
  if (job.error && !row.errorShown) {
    const div = document.createElement("div");
    div.className = "job-error";
    div.textContent = job.error;
    div.title = job.error;
    row.tdLinks.appendChild(div);
    row.errorShown = true;
  }
  return row.tr;
}

function renderJobs(jobs) {
  const body = $("jobs-body");
  const empty = body.querySelector(".jobs-empty");
  if (!jobs.length) {
    if (empty) empty.hidden = false;
    return;
  }
  if (empty) empty.hidden = true;
  // API は新しい順。既存 tr は再利用し、並び順のみ合わせる(appendChild は移動)。
  let cursor = body.querySelector("tr:not(.jobs-empty)");
  for (const job of jobs) {
    const tr = jobRow(job);
    if (tr !== cursor) {
      body.insertBefore(tr, cursor);
    } else {
      cursor = cursor.nextElementSibling;
    }
  }
}

// ---------------------------------------------------------------------------
// ギャラリータブ(Phase 5b)。ポーリングなし・DOM は差分更新。
// ---------------------------------------------------------------------------

const GALLERY_FETCH_LIMIT = 200;

function galleryKey(item) { return item.backend + "/" + item.filename; }

function formatSize(bytes) {
  if (bytes >= 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  if (bytes >= 1024) return (bytes / 1024).toFixed(0) + " KB";
  return bytes + " B";
}

function buildGalleryMedia(item) {
  const media = document.createElement("div");
  media.className = "gallery-media";
  if (item.kind === "image") {
    const img = document.createElement("img");
    img.loading = "lazy";
    img.src = item.url;
    img.alt = item.filename;
    media.appendChild(img);
  } else if (item.kind === "video") {
    // 動画サムネイルは生成しない(preload="metadata" で先頭フレームを表示)
    const video = document.createElement("video");
    video.controls = true;
    video.preload = "metadata";
    video.src = item.url;
    media.appendChild(video);
  } else if (item.kind === "audio") {
    const audio = document.createElement("audio");
    audio.controls = true;
    audio.preload = "none";
    audio.src = item.url;
    media.appendChild(audio);
  } else {
    const a = document.createElement("a");
    a.href = item.url;
    a.target = "_blank";
    a.textContent = item.filename;
    media.appendChild(a);
  }
  return media;
}

function galleryTile(item) {
  const key = galleryKey(item);
  let tile = state.gallery.tiles.get(key);
  if (!tile) {
    const el = document.createElement("div");
    el.className = "gallery-tile";
    el.dataset.backend = item.backend;
    el.dataset.kind = item.kind;
    el.appendChild(buildGalleryMedia(item));

    const cap = document.createElement("div");
    cap.className = "gallery-caption";
    const name = document.createElement("a");
    name.className = "gallery-name";
    name.href = item.url;
    name.target = "_blank";
    name.textContent = item.filename;
    name.title = item.filename;
    const meta = document.createElement("div");
    meta.className = "gallery-meta";
    const info = document.createElement("span");
    info.className = "gallery-info";
    const del = document.createElement("button");
    del.className = "btn danger small";
    del.textContent = "削除";
    del.addEventListener("click", () => deleteOutput(key));
    meta.append(info, del);
    cap.append(name, meta);
    el.appendChild(cap);

    tile = { el, info, item };
    state.gallery.tiles.set(key, tile);
  }
  tile.item = item;
  tile.info.textContent = item.backend + " · " + formatSize(item.size) + " · "
    + new Date(item.mtime * 1000).toLocaleString("ja-JP");
  return tile.el;
}

function renderGallery(items) {
  const grid = $("gal-grid");
  const seen = new Set();
  // API は新しい順。既存タイルは再利用し、並び順のみ合わせる(appendChild は移動)。
  let cursor = grid.firstElementChild;
  for (const item of items) {
    seen.add(galleryKey(item));
    const el = galleryTile(item);
    if (el !== cursor) {
      grid.insertBefore(el, cursor);
    } else {
      cursor = cursor.nextElementSibling;
    }
  }
  // 消えたファイル(削除済み等)のタイルを除去
  for (const [key, tile] of Array.from(state.gallery.tiles)) {
    if (!seen.has(key)) {
      tile.el.remove();
      state.gallery.tiles.delete(key);
    }
  }
  applyGalleryFilter();
}

function applyGalleryFilter() {
  const backend = $("gal-backend").value;
  const kind = $("gal-kind").value;
  let shown = 0;
  for (const tile of state.gallery.tiles.values()) {
    const hide = (backend && tile.el.dataset.backend !== backend)
      || (kind && tile.el.dataset.kind !== kind);
    tile.el.hidden = hide;
    if (!hide) shown++;
  }
  $("gal-count").textContent = shown + " / " + state.gallery.total + " 件";
  $("gal-empty").hidden = shown > 0;
}

async function refreshGallery() {
  if (state.gallery.fetching) return;
  state.gallery.fetching = true;
  const errEl = $("gal-error");
  errEl.hidden = true;
  try {
    const body = await fetchJSON(
      "/api/v1/outputs?offset=0&limit=" + GALLERY_FETCH_LIMIT);
    state.gallery.total = body.total != null ? body.total
      : (body.items || []).length;
    renderGallery(body.items || []);
  } catch (e) {
    errEl.textContent = "成果物一覧の取得に失敗しました: " + e.message;
    errEl.hidden = false;
  } finally {
    state.gallery.fetching = false;
  }
}

async function deleteOutput(key) {
  const tile = state.gallery.tiles.get(key);
  if (!tile) return;
  const item = tile.item;
  if (!window.confirm(item.backend + "/" + item.filename
      + " を削除します。よろしいですか?(元に戻せません)")) return;
  const errEl = $("gal-error");
  errEl.hidden = true;
  try {
    await fetchJSON("/api/v1/outputs/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ backend: item.backend, filename: item.filename }),
    });
    tile.el.remove();
    state.gallery.tiles.delete(key);
    state.gallery.total = Math.max(0, state.gallery.total - 1);
    applyGalleryFilter();
  } catch (e) {
    errEl.textContent = "削除に失敗しました: " + e.message;
    errEl.hidden = false;
  }
}

function initGallery() {
  $("gal-refresh").addEventListener("click", refreshGallery);
  $("gal-backend").addEventListener("change", applyGalleryFilter);
  $("gal-kind").addEventListener("change", applyGalleryFilter);
}

// ---------------------------------------------------------------------------
// ポーリング
// ---------------------------------------------------------------------------

async function refreshStatus() {
  if (state.statusFetching) return;  // in-flight ガード(冒頭の設計メモ参照)
  state.statusFetching = true;
  try {
    state.status = await fetchJSON("/api/v1/status");
    updateTopbar(null);
  } catch (e) {
    updateTopbar(e.message);
  } finally {
    state.statusFetching = false;
  }
}

function updateTopbar(error) {
  const el = $("topbar-status");
  if (error) {
    el.textContent = "gateway に接続できません: " + error;
    return;
  }
  const st = state.status || {};
  el.textContent = "";
  const dot = document.createElement("span");
  dot.className = "dot" + (st.active_backend ? (st.busy ? " busy" : " on") : "");
  el.appendChild(dot);
  const label = st.active_backend
    ? (LABELS[st.active_backend] || st.active_backend)
      + (st.busy ? "(生成中)" : " 稼働中")
    : "バックエンド停止中";
  el.appendChild(document.createTextNode(label));
}

async function refreshJobs() {
  if (state.jobsFetching) return;
  state.jobsFetching = true;
  try {
    const body = await fetchJSON("/api/v1/jobs?limit=50");
    renderJobs(body.jobs || []);
  } catch (e) {
    /* 一過性のエラーは無視(次周期で再試行) */
  } finally {
    state.jobsFetching = false;
  }
}

function renderAll() {
  for (const backend of BACKEND_TABS) renderBackendTab(backend);
  renderStatusCard();
  renderControls();
}

// ---------------------------------------------------------------------------
// 初期化
// ---------------------------------------------------------------------------

async function init() {
  initTabs();
  initControls();
  initGallery();
  try {
    const body = await fetchJSON("/api/v1/backends");
    for (const entry of body.backends || []) state.catalog[entry.name] = entry;
  } catch (e) {
    updateTopbar("カタログ取得に失敗: " + e.message);
  }
  for (const backend of BACKEND_TABS) buildOverlay(backend);
  populateCtlPresets();
  await refreshStatus();
  renderAll();
  await refreshJobs();
  setInterval(async () => { await refreshStatus(); renderAll(); }, STATUS_INTERVAL_MS);
  setInterval(refreshJobs, JOBS_INTERVAL_MS);
}

document.addEventListener("DOMContentLoaded", init);
