"use strict";
// Outline Labeler: the labelling page. Labels are a class id per pixel (0 = unlabelled) on the image's working grid.

const PID = new URLSearchParams(location.search).get("p");
if (!PID) location.replace("/");
const API = "/api/p/" + PID;
const $ = (id) => document.getElementById(id);
const canvas = $("view"), ctx = canvas.getContext("2d");
const imgC = document.createElement("canvas"), ovC = document.createElement("canvas");
const TOOL_KEYS = {v: "pieces", b: "brush", e: "eraser", p: "polygon", s: "smart"};

const S = {
  P: null, images: [], idx: -1, cur: null,
  tool: "pieces", active: 0, radius: 12, protect: false, showLabels: true, showEdges: true, peek: false, alpha: 0.4,
  gran: localStorage.getItem("ol-gran") || "medium", method: localStorage.getItem("ol-pieces") || "sam",
  filter: "all", search: "", who: localStorage.getItem("ol-annotator") || "",
  view: {s: 1, tx: 0, ty: 0}, mouse: null, spaceDown: false,
};
let fillW = new Uint32Array(1), lineW = new Uint32Array(1), byKey = new Map(), byId = new Map();

// ------------------------------------------------------------------ helpers
function toast(msg, ms = 1800) {
  const t = $("toast"); t.textContent = msg; t.classList.remove("hidden");
  clearTimeout(toast.t); toast.t = setTimeout(() => t.classList.add("hidden"), ms);
}
const rgb = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
const word = (r, g, b, a) => ((Math.round(a * 255) << 24) | (b << 16) | (g << 8) | r) >>> 0;
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (r.status === 410) { location.replace("/?reopen=" + PID); throw new Error("reopening the project"); }   // server restarted
  if (!r.ok) {
    let msg = r.statusText;
    try { const j = await r.json(); msg = j.error || msg; r.body_json = j; } catch (e) { /* not json */ }
    const err = new Error(msg); err.status = r.status; err.json = r.body_json; throw err;
  }
  return r;
}

async function fetchArray(path) {
  const r = await api(path);
  const H = +r.headers.get("X-Height"), W = +r.headers.get("X-Width"), dt = r.headers.get("X-Dtype");
  const buf = await new Response(r.body.pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
  const T = dt.endsWith("u2") ? Uint16Array : dt.endsWith("i4") ? Int32Array : Uint8Array;
  return {a: new T(buf), W, H, headers: r.headers};
}

async function gzipBytes(typed) {
  return new Response(new Blob([typed]).stream().pipeThrough(new CompressionStream("gzip"))).arrayBuffer();
}

// ------------------------------------------------------------------ start
async function init() {
  S.P = await (await api(API + "/project")).json();
  S.images = S.P.images;
  document.title = `${S.P.name} · Outline Labeler`;
  if (!S.P.sam) {
    const b = document.querySelector('[data-tool="smart"]');
    b.disabled = true; b.title = "Smart click needs Segment Anything: start with --sam-checkpoint";
    if (S.method === "sam") S.method = "superpixels";
  }
  setClasses(S.P.classes);
  buildSeg("gran", S.P.granularities.map((g) => [g, g[0].toUpperCase() + g.slice(1)]), S.gran, (v) => { S.gran = v; localStorage.setItem("ol-gran", v); loadPieces(); });
  buildSeg("method", [["sam", "Segment Anything"], ["superpixels", "Superpixels"]], S.method, (v) => {
    if (v === "sam" && !S.P.sam) { toast("Segment Anything is not set up: start with --sam-checkpoint"); return false; }
    S.method = v; localStorage.setItem("ol-pieces", v); loadPieces();
  });
  setTool("pieces");
  $("who").textContent = S.who || "Set your name";
  buildList();
  if (!S.who) askName();
  const first = S.images.findIndex((r) => r.status !== "done");
  if (S.images.length) await openImage(first >= 0 ? first : 0);
  else $("meta").textContent = "No images found in " + S.P.images_dir;
}

function askName() {
  const d = $("dlgName"); $("nameInput").value = S.who; d.showModal();
  $("nameGo").onclick = () => {
    S.who = $("nameInput").value.trim() || "anonymous";
    localStorage.setItem("ol-annotator", S.who); $("who").textContent = S.who; d.close();
  };
  $("nameInput").onkeydown = (e) => { if (e.key === "Enter") { e.preventDefault(); $("nameGo").click(); } };
}

function buildSeg(id, opts, value, onPick) {
  const el = $(id); el.innerHTML = "";
  for (const [v, label] of opts) {
    const b = document.createElement("button"); b.textContent = label; b.dataset.v = v;
    if (v === value) b.classList.add("on");
    b.onclick = () => {
      if (onPick(v) === false) return;
      el.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
    };
    el.appendChild(b);
  }
}

// ------------------------------------------------------------------ classes
function setClasses(list) {
  S.P.classes = list;
  byKey = new Map(); byId = new Map();
  let maxId = 0;
  for (const c of list) { byId.set(c.id, c); if (c.key) byKey.set(c.key.toLowerCase(), c); maxId = Math.max(maxId, c.id); }
  fillW = new Uint32Array(maxId + 1); lineW = new Uint32Array(maxId + 1);
  recolour();
  if (!byId.has(S.active)) S.active = list.length ? list[0].id : 0;
  buildClasses();
}

function recolour() {
  for (const c of S.P.classes) { const [r, g, b] = rgb(c.color); fillW[c.id] = word(r, g, b, S.alpha); lineW[c.id] = word(r, g, b, 1); }
}

function buildClasses() {
  const el = $("classes"); el.innerHTML = "";
  const counts = classCounts();
  for (const c of S.P.classes) {
    const row = document.createElement("div");
    row.className = "cls" + (c.id === S.active ? " active" : "");
    row.title = "Click: give the selection this class (or make it the brush class). Double-click the name to rename.";
    row.innerHTML = `<span class="sw" style="background:${c.color}"><input type="color" value="${c.color}"></span>` +
      `<span class="name">${esc(c.name)}</span><span class="cnt">${counts[c.id] || ""}</span>` +
      `<span class="key" title="Click, then press a key">${esc(c.key || "")}</span><span class="del" title="Delete class">×</span>`;
    row.onclick = (e) => { if (!e.target.closest(".sw,.key,.del,[contenteditable=true]")) useClass(c.id); };
    const sw = row.querySelector("input");
    sw.onclick = (e) => e.stopPropagation();
    sw.onchange = () => { c.color = sw.value; pushClasses(); };
    const nm = row.querySelector(".name");
    nm.ondblclick = (e) => {
      e.stopPropagation(); nm.contentEditable = "true"; nm.focus(); document.getSelection().selectAllChildren(nm);
      const done = (ok) => { nm.contentEditable = "false"; const v = nm.textContent.trim(); if (ok && v && v !== c.name) { c.name = v; pushClasses(); } else nm.textContent = c.name; };
      nm.onkeydown = (k) => { k.stopPropagation(); if (k.key === "Enter") { k.preventDefault(); done(true); } if (k.key === "Escape") done(false); };
      nm.onblur = () => done(true);
    };
    const kb = row.querySelector(".key");
    kb.onclick = (e) => {
      e.stopPropagation(); kb.classList.add("wait"); kb.textContent = "…";
      const grab = (k) => {
        k.preventDefault(); k.stopPropagation(); window.removeEventListener("keydown", grab, true); kb.classList.remove("wait");
        const key = k.key.length === 1 ? k.key.toLowerCase() : "";
        if (k.key === "Escape") { kb.textContent = c.key || ""; return; }
        if (k.key === "Backspace" || k.key === "Delete") { c.key = ""; pushClasses(); return; }
        if (!key || "vbepshlo=[],. ".includes(key)) { toast(`"${k.key}" is used by the page; pick another key`); kb.textContent = c.key || ""; return; }
        for (const o of S.P.classes) if (o !== c && o.key === key) o.key = "";
        c.key = key; pushClasses();
      };
      window.addEventListener("keydown", grab, true);
    };
    row.querySelector(".del").onclick = (e) => { e.stopPropagation(); deleteClass(c); };
    el.appendChild(row);
  }
  $("classHint").textContent = S.P.classes.length ? `Brush class: ${byId.get(S.active)?.name || "—"}` : "Add the classes you want to label.";
}

function classCounts() {
  const out = {};
  if (!S.cur) return out;
  const n = new Float64Array(fillW.length), L = S.cur.labels;
  for (let p = 0; p < L.length; p++) if (L[p]) n[L[p]]++;
  for (const c of S.P.classes) if (n[c.id]) { const v = 100 * n[c.id] / L.length; out[c.id] = v < 0.1 ? "<0.1%" : v.toFixed(v < 10 ? 1 : 0) + "%"; }
  return out;
}

async function pushClasses(force = false) {
  try {
    const r = await api(API + "/classes", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({classes: S.P.classes, force})});
    setClasses((await r.json()).classes);
    if (S.cur) renderAll();
  } catch (e) {
    if (e.status === 409 && e.json && e.json.usage) {
      const what = Object.entries(e.json.usage).map(([n, px]) => `${n}: ${px.toLocaleString()} pixels`).join("\n");
      if (confirm(`These classes are used in saved labels:\n${what}\n\nDelete anyway? Their pixels become unlabelled.`)) return pushClasses(true);
    } else toast("Could not save classes: " + e.message, 3500);
    const r = await (await api(API + "/project")).json(); setClasses(r.classes);
  }
  if (force && S.cur) await reloadLabels();
}

function addClass() {
  const name = prompt("Name of the new class:");
  if (!name || !name.trim()) return;
  const used = new Set(S.P.classes.map((c) => c.key));
  const key = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"].find((k) => !used.has(k)) || "";
  S.P.classes.push({name: name.trim(), key, color: ""});
  pushClasses().then(() => { const c = S.P.classes[S.P.classes.length - 1]; if (c) { S.active = c.id; buildClasses(); } });
}

function deleteClass(c) {
  const here = S.cur ? S.cur.labels.some((v) => v === c.id) : false;
  if (here && !confirm(`"${c.name}" is used in this image. Delete it? Its pixels become unlabelled.`)) return;
  if (here) {                                                   // unsaved pixels of the class go too
    const L = S.cur.labels; for (let p = 0; p < L.length; p++) if (L[p] === c.id) L[p] = 0;
    renderAll(); markDirty();
  }
  S.P.classes = S.P.classes.filter((x) => x !== c);
  saveNow().then(() => pushClasses());
}

function useClass(id) {
  if (S.cur && S.cur.selN > 0) { assign(id); return; }
  S.active = id; buildClasses();
  toast(`Brush class: ${byId.get(id).name}`);
  if (S.tool === "pieces" || S.tool === "eraser") setTool("brush");
}

// ------------------------------------------------------------------ images
function buildList() {
  const L = $("list"); L.innerHTML = "";
  const q = S.search.toLowerCase();
  let shown = 0;
  S.images.forEach((r, i) => {
    if (S.filter === "done" && r.status !== "done") return;
    if (S.filter === "todo" && r.status === "done") return;
    if (q && !r.name.toLowerCase().includes(q)) return;
    const d = document.createElement("div"); d.className = i === S.idx ? "cur" : "";
    const st = r.status === "done" ? ["done", "✓"] : (r.status === "in progress" || r.status === "prelabelled") ? ["prog", "●"] : ["", ""];
    d.innerHTML = `<span class="st ${st[0]}">${st[1]}</span><span class="nm" title="${esc(r.name)}">&lrm;${esc(r.name)}</span><span class="pc">${r.labelled ? Math.round(100 * r.labelled) + "%" : ""}</span>`;
    d.onclick = () => openImage(i);
    L.appendChild(d); shown++;
  });
  const done = S.images.filter((r) => r.status === "done").length;
  $("progress").textContent = `${S.idx + 1} / ${S.images.length}`;
  $("count").textContent = `${done} of ${S.images.length} done`;
  const cur = L.querySelector(".cur"); if (cur) cur.scrollIntoView({block: "nearest"});
}

function loadImg(src) {
  return new Promise((res, rej) => { const im = new Image(); im.onload = () => res(im); im.onerror = () => rej(new Error("cannot load image")); im.src = src; });
}

async function openImage(i) {
  if (i < 0 || i >= S.images.length) return;
  await saveNow();
  const r = S.images[i], key = encodeURIComponent(r.key);
  $("meta").innerHTML = `loading <b>${esc(r.name)}</b>…`;
  const [info, im, lab] = await Promise.all([api(`${API}/image/${key}/info`).then((x) => x.json()), loadImg(`${API}/image/${key}/display`), fetchArray(`${API}/image/${key}/labels`)]);
  const W = im.naturalWidth, H = im.naturalHeight;
  if (lab.W !== W || lab.H !== H) throw new Error("label size does not match the image");
  imgC.width = ovC.width = W; imgC.height = ovC.height = H;
  const g = imgC.getContext("2d", {willReadFrequently: true}); g.drawImage(im, 0, 0);
  const px = g.getImageData(0, 0, W, H).data, bright = new Uint8Array(W * H);    // outlines: dark on bright pixels, light on dark
  for (let p = 0, q = 0; p < W * H; p++, q += 4) bright[p] = 0.299 * px[q] + 0.587 * px[q + 1] + 0.114 * px[q + 2] > 150 ? 1 : 0;
  S.idx = i;
  S.cur = {key: r.key, rec: r, W, H, labels: lab.a, version: +lab.headers.get("X-Version"), status: info.status === "new" ? "new" : info.status,
           seconds: info.seconds || 0, last: performance.now(), pieces: null, edge: new Uint8Array(W * H), sel: new Uint8Array(W * H), selN: 0,
           selBox: null, touch: new Uint8Array(W * H), undo: [], redo: [], dirty: false, poly: null, smart: [], bright, ov: ovC.getContext("2d").createImageData(W, H)};
  S.cur.u32 = new Uint32Array(S.cur.ov.data.buffer);
  for (let p = 0; p < W * H; p++) if (S.cur.labels[p] && !byId.has(S.cur.labels[p])) S.cur.labels[p] = 0;   // class deleted meanwhile
  fit(); renderAll(); buildList(); buildClasses(); updateMeta(); $("conflict").classList.add("hidden");
  loadPieces();
  if (S.images[i + 1]) new Image().src = `${API}/image/${encodeURIComponent(S.images[i + 1].key)}/display`;
}

function updateMeta() {
  const c = S.cur; if (!c) return;
  const st = c.status === "done" ? " · <b>done</b>" : c.status === "prelabelled" ? " · pre-labelled" : "";
  $("meta").innerHTML = `<b>${esc(c.rec.name)}</b> · ${c.W}×${c.H}${st}`;
  $("done").textContent = c.status === "done" ? "Done ✓ → next" : "Done ✓";
}

async function loadPieces() {
  const c = S.cur; if (!c) return;
  const key = c.key;
  $("busy").textContent = S.method === "sam" && S.P.sam ? "outlining grains (Segment Anything)…" : "finding pieces…";
  $("busy").classList.remove("hidden");
  try {
    const r = await fetchArray(`${API}/image/${encodeURIComponent(key)}/pieces?method=${S.method}&granularity=${S.gran}`);
    if (!S.cur || S.cur.key !== key) return;
    setPieces(r.a);
  } catch (e) { toast("Could not make pieces: " + e.message, 4000); }
  finally { if (S.cur && S.cur.key === key) $("busy").classList.add("hidden"); }
}

function setPieces(seg) {                 // seg: piece of every pixel, -1 = no piece there
  const c = S.cur, {W, H} = c, N = W * H;
  let n = 0, covered = 0; for (let p = 0; p < N; p++) if (seg[p] >= n) n = seg[p] + 1;
  const start = new Int32Array(n + 1);
  for (let p = 0; p < N; p++) if (seg[p] >= 0) { start[seg[p] + 1]++; covered++; }
  for (let j = 0; j < n; j++) start[j + 1] += start[j];
  const pos = start.slice(0, n), idx = new Int32Array(covered), box = new Int32Array(4 * n);
  for (let j = 0; j < n; j++) { box[4 * j] = W; box[4 * j + 1] = H; box[4 * j + 2] = -1; box[4 * j + 3] = -1; }
  for (let y = 0, p = 0; y < H; y++) for (let x = 0; x < W; x++, p++) {
    const j = seg[p]; if (j < 0) continue;
    idx[pos[j]++] = p;
    const b = 4 * j; if (x < box[b]) box[b] = x; if (y < box[b + 1]) box[b + 1] = y; if (x > box[b + 2]) box[b + 2] = x; if (y > box[b + 3]) box[b + 3] = y;
  }
  c.pieces = {seg, n, start, idx, box, picked: new Set(), covered: covered / N};
  computeEdges(); renderAll();
  if (S.tool === "pieces") $("info").textContent = `${n.toLocaleString()} pieces covering ${Math.round(100 * covered / N)}% of the image` +
    (covered < N ? "; label the rest with the brush (B), polygon (P) or smart click (S)" : "");
}

function computeEdges() {          // piece outlines, thick enough to stay visible when a large image is zoomed out
  const c = S.cur; if (!c || !c.pieces) return;
  const {W, H} = c, seg = c.pieces.seg, e = c.edge; e.fill(0);
  for (let y = 0, p = 0; y < H; y++) for (let x = 0; x < W; x++, p++) {
    if (x + 1 < W && seg[p + 1] !== seg[p]) e[p] = 1;
    if (y + 1 < H && seg[p + W] !== seg[p]) e[p] = 1;
  }
  const t = Math.max(1, Math.min(4, Math.round(1 / Math.max(S.view.s, 0.05))));
  for (let r = 1; r < t; r++) {
    const f = e.slice();
    for (let y = 0, p = 0; y < H; y++) for (let x = 0; x < W; x++, p++)
      if (!e[p] && ((x > 0 && e[p - 1]) || (y > 0 && e[p - W]))) f[p] = 1;
    e.set(f);
  }
}

// ------------------------------------------------------------------ drawing
function renderRect(x0, y0, x1, y1) {
  const c = S.cur; if (!c) return;
  const {W, H, labels: L, sel, edge, u32, bright} = c;
  x0 = Math.max(0, x0 | 0); y0 = Math.max(0, y0 | 0); x1 = Math.min(W, Math.ceil(x1)); y1 = Math.min(H, Math.ceil(y1));
  if (x1 <= x0 || y1 <= y0) return;
  const showL = S.showLabels, showE = S.showEdges && !!c.pieces;
  const edgeW = word(255, 255, 255, 0.6), edgeD = word(0, 0, 0, 0.5), stripe = word(255, 255, 255, 0.7), shade = word(0, 0, 0, 0.25);
  for (let y = y0; y < y1; y++) {
    let p = y * W + x0;
    for (let x = x0; x < x1; x++, p++) {
      const v = L[p];
      let w = 0, border = false;
      if (showL && v) {
        border = (x > 0 && L[p - 1] !== v) || (x + 1 < W && L[p + 1] !== v) || (y > 0 && L[p - W] !== v) || (y + 1 < H && L[p + W] !== v);
        w = border ? lineW[v] : fillW[v];
      }
      if (showE && edge[p] && !border) w = bright[p] ? edgeD : edgeW;     // piece outline (a class border wins)
      if (sel[p]) w = ((x + y) >> 2) & 1 ? stripe : (w || shade);
      u32[p] = w;
    }
  }
  ovC.getContext("2d").putImageData(c.ov, 0, 0, x0, y0, x1 - x0, y1 - y0);
}

function renderAll() { if (S.cur) { renderRect(0, 0, S.cur.W, S.cur.H); draw(); } }

function fit() {
  const c = S.cur; if (!c) return;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  S.view.s = Math.min(w / c.W, h / c.H) * 0.98;
  S.view.tx = (w - c.W * S.view.s) / 2; S.view.ty = (h - c.H * S.view.s) / 2;
}

function draw() {
  const dpr = window.devicePixelRatio || 1;
  const cw = Math.round(canvas.clientWidth * dpr), ch = Math.round(canvas.clientHeight * dpr);
  if (canvas.width !== cw || canvas.height !== ch) { canvas.width = cw; canvas.height = ch; }
  ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, cw, ch);
  const c = S.cur; if (!c) return;
  const {s, tx, ty} = S.view;
  ctx.setTransform(dpr * s, 0, 0, dpr * s, dpr * tx, dpr * ty);
  ctx.imageSmoothingEnabled = s < 2; ctx.drawImage(imgC, 0, 0);
  if (!S.peek) { ctx.imageSmoothingEnabled = s < 1; ctx.drawImage(ovC, 0, 0); }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const scr = (x, y) => [tx + x * s, ty + y * s];
  if (c.poly && c.poly.length) {                               // polygon being drawn
    ctx.lineWidth = 2; ctx.strokeStyle = "#fff"; ctx.setLineDash([5, 4]);
    ctx.beginPath(); c.poly.forEach(([x, y], k) => { const [a, b] = scr(x, y); k ? ctx.lineTo(a, b) : ctx.moveTo(a, b); });
    if (S.mouse) ctx.lineTo(S.mouse.sx, S.mouse.sy);
    ctx.stroke(); ctx.setLineDash([]);
    const [a, b] = scr(c.poly[0][0], c.poly[0][1]); ctx.fillStyle = "#fff"; ctx.beginPath(); ctx.arc(a, b, 5, 0, 7); ctx.fill();
  }
  for (const [x, y, pos] of c.smart) {                          // smart-click points
    const [a, b] = scr(x, y); ctx.beginPath(); ctx.arc(a, b, 6, 0, 7);
    ctx.fillStyle = pos ? "#2fbf5b" : "#e5484d"; ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = "#fff"; ctx.stroke();
  }
  if ((S.tool === "brush" || S.tool === "eraser") && S.mouse && S.mouse.inside) {
    ctx.beginPath(); ctx.arc(S.mouse.sx, S.mouse.sy, Math.max(2, S.radius * s), 0, 7);
    ctx.lineWidth = 1.5; ctx.strokeStyle = "#fff"; ctx.stroke(); ctx.strokeStyle = "rgba(0,0,0,.5)"; ctx.lineWidth = 0.75; ctx.stroke();
  }
}

// ------------------------------------------------------------------ selection
function clearSel(render = true) {
  const c = S.cur; if (!c || !c.selN) { if (c && c.pieces) c.pieces.picked.clear(); return; }
  const b = c.selBox; c.sel.fill(0); c.selN = 0; c.selBox = null;
  if (c.pieces) c.pieces.picked.clear();
  if (render && b) { renderRect(b[0], b[1], b[2] + 1, b[3] + 1); draw(); }
}

function growBox(c, x0, y0, x1, y1) {
  const b = c.selBox;
  c.selBox = b ? [Math.min(b[0], x0), Math.min(b[1], y0), Math.max(b[2], x1), Math.max(b[3], y1)] : [x0, y0, x1, y1];
}

function togglePiece(j, on) {
  const c = S.cur, P = c.pieces;
  const want = on === undefined ? !P.picked.has(j) : on;
  if (want === P.picked.has(j)) return;
  want ? P.picked.add(j) : P.picked.delete(j);
  for (let k = P.start[j]; k < P.start[j + 1]; k++) { const p = P.idx[k]; if (!!c.sel[p] !== want) { c.sel[p] = want ? 1 : 0; c.selN += want ? 1 : -1; } }
  const b = P.box.subarray(4 * j, 4 * j + 4);
  growBox(c, b[0], b[1], b[2], b[3]);
  renderRect(b[0], b[1], b[2] + 1, b[3] + 1);
}

function setSelMask(mask, add) {                               // mask: Uint8Array over the image
  const c = S.cur, {W, H} = c;
  if (!add) clearSel(false);
  let x0 = W, y0 = H, x1 = -1, y1 = -1;
  for (let y = 0, p = 0; y < H; y++) for (let x = 0; x < W; x++, p++) if (mask[p]) {
    if (!c.sel[p]) { c.sel[p] = 1; c.selN++; }
    if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
  }
  if (x1 >= 0) growBox(c, x0, y0, x1, y1);
  renderAll();
}

function polygonMask(pts) {
  const c = S.cur, xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const x0 = Math.max(0, Math.floor(Math.min(...xs))), y0 = Math.max(0, Math.floor(Math.min(...ys)));
  const x1 = Math.min(c.W, Math.ceil(Math.max(...xs)) + 1), y1 = Math.min(c.H, Math.ceil(Math.max(...ys)) + 1);
  const mask = new Uint8Array(c.W * c.H);
  if (x1 <= x0 || y1 <= y0) return mask;
  const t = document.createElement("canvas"); t.width = x1 - x0; t.height = y1 - y0;
  const g = t.getContext("2d", {willReadFrequently: true});
  g.fillStyle = "#fff"; g.beginPath(); pts.forEach(([x, y], k) => (k ? g.lineTo(x - x0 + 0.5, y - y0 + 0.5) : g.moveTo(x - x0 + 0.5, y - y0 + 0.5))); g.closePath(); g.fill();
  const d = g.getImageData(0, 0, t.width, t.height).data;
  for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) if (d[4 * ((y - y0) * t.width + (x - x0)) + 3] > 127) mask[y * c.W + x] = 1;
  return mask;
}

// ------------------------------------------------------------------ editing
function record(idx, old, val, box) {
  const c = S.cur; c.undo.push({idx, old, val, box}); c.redo = [];
  let bytes = 0; for (const u of c.undo) bytes += u.idx.byteLength + u.old.byteLength;
  while (bytes > 400e6 && c.undo.length > 1) { const u = c.undo.shift(); bytes -= u.idx.byteLength + u.old.byteLength; }
}

function assign(val) {
  const c = S.cur; if (!c || !c.selN) { toast("Select something first"); return; }
  const [x0, y0, x1, y1] = c.selBox, {W, labels: L, sel} = c, idx = [], old = [];
  for (let y = y0; y <= y1; y++) for (let x = x0, p = y * W + x0; x <= x1; x++, p++) {
    if (sel[p] && L[p] !== val && (val === 0 || !S.protect || L[p] === 0)) { idx.push(p); old.push(L[p]); L[p] = val; }
  }
  if (idx.length) record(Int32Array.from(idx), Uint16Array.from(old), val, [x0, y0, x1, y1]);
  clearSel(false); c.smart = []; c.poly = null;
  renderRect(x0 - 1, y0 - 1, x1 + 2, y1 + 2); draw();
  if (idx.length) { markDirty(); buildClasses(); }
  toast(idx.length ? `${val ? byId.get(val).name : "Unlabelled"}: ${idx.length.toLocaleString()} pixels` : "Nothing changed" + (S.protect ? " (protect is on)" : ""));
}

function applyHistory(from, to, undo) {
  const c = S.cur, u = from.pop(); if (!u) { toast(undo ? "Nothing to undo" : "Nothing to redo"); return; }
  const L = c.labels;
  if (undo) for (let k = 0; k < u.idx.length; k++) L[u.idx[k]] = u.old[k];
  else for (let k = 0; k < u.idx.length; k++) L[u.idx[k]] = u.val;
  to.push(u);
  const [x0, y0, x1, y1] = u.box; renderRect(x0 - 1, y0 - 1, x1 + 2, y1 + 2); draw(); markDirty(); buildClasses();
}

let stroke = null;
function paintAt(x, y) {
  const c = S.cur, {W, H, labels: L, touch} = c, r = S.radius, val = S.tool === "eraser" ? 0 : S.active;
  if (S.tool === "brush" && !byId.has(val)) { toast("Add a class first"); return; }
  const x0 = Math.max(0, Math.floor(x - r)), x1 = Math.min(W - 1, Math.ceil(x + r)), y0 = Math.max(0, Math.floor(y - r)), y1 = Math.min(H - 1, Math.ceil(y + r));
  for (let yy = y0; yy <= y1; yy++) for (let xx = x0; xx <= x1; xx++) {
    if ((xx - x) * (xx - x) + (yy - y) * (yy - y) > r * r) continue;
    const p = yy * W + xx;
    if (touch[p] || L[p] === val || (val !== 0 && S.protect && L[p] !== 0)) continue;
    touch[p] = 1; stroke.idx.push(p); stroke.old.push(L[p]); L[p] = val;
  }
  const b = stroke.box; b[0] = Math.min(b[0], x0); b[1] = Math.min(b[1], y0); b[2] = Math.max(b[2], x1); b[3] = Math.max(b[3], y1);
  renderRect(x0 - 1, y0 - 1, x1 + 2, y1 + 2);
}

function paintLine(a, b) {
  const d = Math.hypot(b[0] - a[0], b[1] - a[1]), n = Math.max(1, Math.ceil(d / Math.max(1, S.radius / 3)));
  for (let k = 1; k <= n; k++) paintAt(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n);
  draw();
}

function endStroke() {
  const c = S.cur; if (!stroke) return;
  for (const p of stroke.idx) c.touch[p] = 0;
  if (stroke.idx.length) {
    record(Int32Array.from(stroke.idx), Uint16Array.from(stroke.old), S.tool === "eraser" ? 0 : S.active, stroke.box);
    markDirty(); buildClasses();
  }
  stroke = null;
}

function closePolygon(add) {
  const c = S.cur; if (!c.poly) return;
  const pts = c.poly; c.poly = null;
  if (pts.length < 3) { draw(); return; }
  setSelMask(polygonMask(pts), add);
  toast("Now press a class key (or Del to unlabel)");
}

async function smartRequest() {
  const c = S.cur, key = c.key;
  if (!c.smart.length) { clearSel(); return; }
  $("busy").textContent = "outlining…"; $("busy").classList.remove("hidden");
  try {
    const r = await api(`${API}/image/${encodeURIComponent(key)}/smart`, {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({points: c.smart.map(([x, y, pos]) => [x, y, pos ? 1 : 0])})});
    const buf = await new Response(r.body.pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
    if (S.cur && S.cur.key === key) setSelMask(new Uint8Array(buf), false);
  } catch (e) { toast("Smart click failed: " + e.message, 3500); }
  finally { $("busy").classList.add("hidden"); $("busy").textContent = "finding pieces…"; }
}

// ------------------------------------------------------------------ saving
function markDirty() {
  const c = S.cur; c.dirty = true;
  if (c.status === "new") c.status = "in progress";
  $("save").textContent = "unsaved"; $("save").classList.remove("bad");
  clearTimeout(markDirty.t); markDirty.t = setTimeout(saveNow, 1200);
}

function tick() {                                                // time spent: activity only, idle gaps capped at a minute
  const c = S.cur; if (!c) return;
  const now = performance.now(); c.seconds += Math.min(60, (now - c.last) / 1000); c.last = now;
}

function saveQuery(c, base) {
  const q = new URLSearchParams({status: c.status === "new" ? "in progress" : c.status, annotator: S.who || "anonymous", seconds: Math.round(c.seconds)});
  if (base) q.set("base", c.version);
  return q;
}

async function saveNow(force = false) {
  clearTimeout(markDirty.t);
  const c = S.cur; if (!c || (!c.dirty && !force)) return true;
  tick(); c.dirty = false; $("save").textContent = "saving…";
  const q = saveQuery(c, true); q.set("enc", "gzip");
  try {
    const r = await api(`${API}/image/${encodeURIComponent(c.key)}/labels?${q}`, {method: "POST", headers: {"Content-Type": "application/octet-stream"}, body: await gzipBytes(c.labels)});
    const st = await r.json();
    c.version = st.version; $("save").textContent = "saved";
    const rec = S.images.find((x) => x.key === c.key);
    if (rec) { rec.status = st.status; rec.labelled = st.labelled_fraction; rec.annotator = st.annotator; }
    return true;
  } catch (e) {
    c.dirty = true;
    if (e.status === 409) { $("conflict").classList.remove("hidden"); $("save").textContent = "conflict"; }
    else { $("save").textContent = "NOT saved"; toast("Could not save: " + e.message + ". Is the server still running?", 4000); }
    $("save").classList.add("bad");
    return false;
  }
}

async function overwrite() {
  const c = S.cur; $("conflict").classList.add("hidden");
  const info = await (await api(`${API}/image/${encodeURIComponent(c.key)}/info`)).json();
  c.version = info.version; c.dirty = true; await saveNow();
}

async function reloadLabels() {
  const c = S.cur; $("conflict").classList.add("hidden");
  const lab = await fetchArray(`${API}/image/${encodeURIComponent(c.key)}/labels`);
  c.labels.set(lab.a); c.version = +lab.headers.get("X-Version"); c.dirty = false; c.undo = []; c.redo = [];
  $("save").textContent = "loaded"; $("save").classList.remove("bad"); renderAll(); buildClasses();
}

window.addEventListener("beforeunload", (e) => {
  const c = S.cur;
  if (c && c.dirty) {
    tick(); const q = saveQuery(c, true); q.set("enc", "raw");
    navigator.sendBeacon(`${API}/image/${encodeURIComponent(c.key)}/labels?${q}`, new Blob([c.labels]));
  }
});

async function finish() {
  const c = S.cur; if (!c) return;
  c.status = "done"; c.dirty = true;
  if (!(await saveNow())) return;
  buildList();
  let nxt = S.images.findIndex((r, i) => i > S.idx && r.status !== "done");
  if (nxt < 0) nxt = S.images.findIndex((r) => r.status !== "done");
  if (nxt < 0) { updateMeta(); toast("Every image is done. Export your labels from the panel on the right.", 4000); return; }
  openImage(nxt);
}

// ------------------------------------------------------------------ tools and view
function setTool(t) {
  if (t === "smart" && !S.P.sam) { toast("Smart click needs Segment Anything: start with --sam-checkpoint"); return; }
  const c = S.cur;
  if (c) { c.poly = null; if (t !== "smart") c.smart = []; }
  S.tool = t;
  document.querySelectorAll("#tools [data-tool]").forEach((b) => b.classList.toggle("on", b.dataset.tool === t));
  const hints = {pieces: "Click a piece (shift+click adds, drag paints a selection), then press a class key.",
    brush: "Paint with the brush class; [ and ] change the size.", eraser: "Rub out labels; [ and ] change the size.",
    polygon: "Click corners or hold and drag; close on the first point, double-click or Enter; then press a class key.",
    smart: "Click inside an object; right-click what is not part of it; then press a class key."};
  $("info").textContent = hints[t];
  draw();
}

function toImage(e) {
  const r = canvas.getBoundingClientRect(), sx = e.clientX - r.left, sy = e.clientY - r.top;
  const x = (sx - S.view.tx) / S.view.s, y = (sy - S.view.ty) / S.view.s;
  const c = S.cur;
  return {sx, sy, x, y, inside: !!c && x >= 0 && y >= 0 && x < c.W && y < c.H};
}

function pieceAt(m) { const c = S.cur; return c && c.pieces && m.inside ? c.pieces.seg[(m.y | 0) * c.W + (m.x | 0)] : -1; }

function describe(m) {
  const c = S.cur; if (!c || !m.inside) return;
  const p = (m.y | 0) * c.W + (m.x | 0), v = c.labels[p];
  const name = v ? `<b>${esc(byId.get(v)?.name || v)}</b>` : "unlabelled";
  const sel = c.selN ? ` · selected ${c.selN.toLocaleString()} px` : "";
  $("info").innerHTML = `x ${m.x | 0}, y ${m.y | 0} · ${name}${sel}`;
}

let drag = null;
canvas.addEventListener("contextmenu", (e) => e.preventDefault());
canvas.addEventListener("pointerdown", (e) => {
  const c = S.cur; if (!c) return;
  canvas.setPointerCapture(e.pointerId);
  const m = toImage(e);
  const panBtn = e.button === 1 || (e.button === 2 && S.tool !== "smart") || (e.button === 0 && S.spaceDown);
  if (panBtn) { drag = {pan: true, x: e.clientX, y: e.clientY, tx: S.view.tx, ty: S.view.ty}; $("stage").classList.add("pan"); return; }
  tick();
  if (S.tool === "pieces") {
    if (!c.pieces) { toast("Pieces are still being made…"); return; }
    const j = pieceAt(m);
    if (!e.shiftKey) clearSel(false);
    if (j >= 0) togglePiece(j, e.shiftKey ? undefined : true);
    drag = {pieces: true}; draw();
  } else if (S.tool === "brush" || S.tool === "eraser") {
    stroke = {idx: [], old: [], box: [c.W, c.H, 0, 0]};
    if (m.inside) paintAt(m.x, m.y);
    drag = {paint: true, last: [m.x, m.y]}; draw();
  } else if (S.tool === "polygon") {
    if (!c.poly) { c.poly = [[m.x, m.y]]; c.polyAdd = e.shiftKey; }
    else {
      const [fx, fy] = c.poly[0];
      if (c.poly.length >= 3 && Math.hypot((fx - m.x) * S.view.s, (fy - m.y) * S.view.s) < 9) { closePolygon(c.polyAdd); return; }
      c.poly.push([m.x, m.y]);
    }
    drag = {lasso: true, sx: m.sx, sy: m.sy}; draw();
  } else if (S.tool === "smart") {
    if (!m.inside) return;
    c.smart.push([m.x, m.y, !(e.button === 2 || e.altKey)]);
    draw(); smartRequest();
  }
});
canvas.addEventListener("pointermove", (e) => {
  const c = S.cur; if (!c) return;
  const m = toImage(e); S.mouse = m;
  if (drag && drag.pan) { S.view.tx = drag.tx + e.clientX - drag.x; S.view.ty = drag.ty + e.clientY - drag.y; draw(); return; }
  if (drag && drag.pieces) { const j = pieceAt(m); if (j >= 0 && !c.pieces.picked.has(j)) { togglePiece(j, true); draw(); } }
  else if (drag && drag.paint) { paintLine(drag.last, [m.x, m.y]); drag.last = [m.x, m.y]; }
  else if (drag && drag.lasso && c.poly && Math.hypot(m.sx - drag.sx, m.sy - drag.sy) > 4) { c.poly.push([m.x, m.y]); drag.sx = m.sx; drag.sy = m.sy; drag.moved = true; draw(); }
  else draw();
  describe(m);
});
canvas.addEventListener("pointerup", () => {
  const c = S.cur;
  if (drag && drag.paint) endStroke();
  if (drag && drag.lasso && drag.moved && c.poly && c.poly.length >= 3) closePolygon(c.polyAdd);   // freehand lasso closes on release
  drag = null; $("stage").classList.remove("pan");
});
canvas.addEventListener("pointerleave", () => { if (S.mouse) S.mouse.inside = false; draw(); });
canvas.addEventListener("dblclick", () => { if (S.tool === "polygon" && S.cur.poly) { S.cur.poly.pop(); closePolygon(S.cur.polyAdd); } });
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  const r = canvas.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
  const s = Math.min(64, Math.max(0.02, S.view.s * Math.exp(-e.deltaY * (e.deltaMode ? 0.05 : 0.0015))));
  S.view.tx = mx - (mx - S.view.tx) * (s / S.view.s); S.view.ty = my - (my - S.view.ty) * (s / S.view.s); S.view.s = s;
  draw();
}, {passive: false});
window.addEventListener("resize", draw);

window.addEventListener("keydown", (e) => {
  if (e.target.closest && e.target.closest("input, [contenteditable=true], dialog")) return;
  const c = S.cur; if (!c) return;
  const k = e.key.toLowerCase();
  if ((e.ctrlKey || e.metaKey) && k === "z") { e.preventDefault(); e.shiftKey ? applyHistory(c.redo, c.undo, false) : applyHistory(c.undo, c.redo, true); return; }
  if ((e.ctrlKey || e.metaKey) && k === "y") { e.preventDefault(); applyHistory(c.redo, c.undo, false); return; }
  if ((e.ctrlKey || e.metaKey) && k === "a") { e.preventDefault(); setSelMask(new Uint8Array(c.W * c.H).fill(1), false); toast("Whole image selected: press a class key"); return; }
  if ((e.ctrlKey || e.metaKey) && k === "s") { e.preventDefault(); c.dirty = true; saveNow(); return; }
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  if (k === " ") { S.spaceDown = true; e.preventDefault(); return; }
  if (k === "h") { if (!S.peek) { S.peek = true; draw(); } return; }
  if (e.repeat && !["[", "]"].includes(k)) return;
  tick();
  if (k === "enter") { e.preventDefault(); if (c.poly) closePolygon(c.polyAdd); else finish(); return; }
  if (k === "escape") { if (c.poly) c.poly = null; c.smart = []; clearSel(); draw(); return; }
  if (k === "delete" || k === "backspace") {
    e.preventDefault();
    if (c.poly) { c.poly.pop(); if (!c.poly.length) c.poly = null; draw(); return; }
    if (c.selN) assign(0); return;
  }
  if (k in TOOL_KEYS) { setTool(TOOL_KEYS[k]); return; }
  if (k === "[" || k === "]") { S.radius = Math.max(1, Math.min(400, Math.round(S.radius * (k === "]" ? 1.25 : 0.8)) + (k === "]" ? 1 : 0))); $("sizeval").textContent = S.radius; draw(); return; }
  if (k === "l") { toggle("tLabels"); return; }
  if (k === "o") { toggle("tEdges"); return; }
  if (k === "=") { fit(); draw(); return; }
  if (k === ",") { openImage(S.idx - 1); return; }
  if (k === ".") { openImage(S.idx + 1); return; }
  if (k === "?") { $("dlgHelp").showModal(); return; }
  const cls = byKey.get(k);
  if (cls) { useClass(cls.id); return; }
});
window.addEventListener("keyup", (e) => {
  const k = e.key.toLowerCase();
  if (k === " ") S.spaceDown = false;
  if (k === "h") { S.peek = false; draw(); }
});

function toggle(id) {
  const b = $(id), on = !b.classList.contains("on"); b.classList.toggle("on", on);
  if (id === "tLabels") S.showLabels = on;
  if (id === "tEdges") S.showEdges = on;
  if (id === "protect") { S.protect = on; toast(on ? "Protect on: new labels only fill unlabelled pixels" : "Protect off: new labels overwrite"); return; }
  renderAll();
}

// ------------------------------------------------------------------ panel controls
document.querySelectorAll("#tools [data-tool]").forEach((b) => (b.onclick = () => setTool(b.dataset.tool)));
document.addEventListener("click", (e) => { const b = e.target.closest("button"); if (b && !b.closest("dialog")) b.blur(); });
$("protect").onclick = () => toggle("protect");
$("tLabels").onclick = () => toggle("tLabels");
$("tEdges").onclick = () => toggle("tEdges");
$("fit").onclick = () => { fit(); draw(); };
$("undo").onclick = () => S.cur && applyHistory(S.cur.undo, S.cur.redo, true);
$("redo").onclick = () => S.cur && applyHistory(S.cur.redo, S.cur.undo, false);
$("help").onclick = () => $("dlgHelp").showModal();
$("prev").onclick = () => openImage(S.idx - 1);
$("next").onclick = () => openImage(S.idx + 1);
$("done").onclick = finish;
$("who").onclick = askName;
$("addClass").onclick = addClass;
$("unlabel").onclick = () => assign(0);
$("theirs").onclick = reloadLabels;
$("mine").onclick = overwrite;
$("alpha").oninput = (e) => { S.alpha = parseFloat(e.target.value); recolour(); renderAll(); };
$("search").oninput = (e) => { S.search = e.target.value; buildList(); };
document.querySelectorAll("#filter button").forEach((b) => (b.onclick = () => {
  S.filter = b.dataset.f; document.querySelectorAll("#filter button").forEach((x) => x.classList.toggle("on", x === b)); buildList();
}));
$("export").onclick = () => $("dlgExport").showModal();
$("exCancel").onclick = () => $("dlgExport").close();
$("exGo").onclick = async () => {
  const f = (n) => document.querySelector(`#dlgExport [name=${n}]`).checked;
  const formats = ["masks", "color", "coco"].filter(f);
  if (!formats.length) { toast("Pick at least one format"); return; }
  $("exGo").disabled = true; $("exGo").textContent = "Exporting…";
  try {
    await saveNow();
    const r = await api(API + "/export", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({formats, full_size: f("full"), only_done: f("done")})});
    const blob = await r.blob(), a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = (r.headers.get("Content-Disposition") || "").match(/filename="(.+)"/)?.[1] || "labels.zip";
    a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 5000);
    $("dlgExport").close(); toast("Exported; also kept in " + (r.headers.get("X-Folder") || "the project's exports folder"), 4000);
  } catch (e) { toast("Export failed: " + e.message, 4000); }
  finally { $("exGo").disabled = false; $("exGo").textContent = "Export"; }
};
$("sizeval").textContent = S.radius;

init().catch((e) => { $("meta").textContent = "Could not start: " + e.message; console.error(e); });
