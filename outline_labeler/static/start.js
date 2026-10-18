"use strict";
// Start page: recent projects, a new project from an image folder, Segment Anything set-up.

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
let H = null, projEdited = false, samTimer = null;

async function post(path, body) {
  const r = await fetch(path, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}

async function load() {
  H = await (await fetch("/api/home")).json();
  $("ver").textContent = "version " + H.version;
  document.querySelectorAll(".local").forEach((e) => e.classList.toggle("hidden", !H.local));
  const reopen = new URLSearchParams(location.search).get("reopen");
  if (reopen) {
    const r = H.recent.find((x) => x.pid === reopen);
    history.replaceState(null, "", "/");
    if (r) return openProject({project: r.project});
  }
  showRecent(); showSam(H.sam);
}

function showRecent() {
  const el = $("recent");
  const items = H.local ? H.recent : H.recent.filter((r) => r.open);
  if (!items.length) {
    el.innerHTML = `<div class="empty">${H.local ? "No projects yet: start one on the right." : "No project is open on this server yet."}</div>`;
    return;
  }
  el.innerHTML = "";
  for (const r of items) {
    const d = document.createElement("div"); d.className = "proj";
    const total = r.total ?? null;
    d.innerHTML = `<div class="n">${esc(r.name)}</div>
      <div class="act"><button class="primary">Open</button>${H.local ? '<button class="ghost" title="Remove from this list (nothing is deleted)">×</button>' : ""}</div>
      <div class="p" title="${esc(r.images)}">&lrm;${esc(r.images)}</div>
      <div class="s">${r.done} done · ${r.labelled} started · ${r.classes} classes · last opened ${esc(r.opened)}</div>`;
    d.querySelector(".primary").onclick = () => openProject({project: r.project});
    const x = d.querySelector(".ghost");
    if (x) x.onclick = async () => { await post("/api/forget", {project: r.project}); H.recent = H.recent.filter((y) => y !== r); showRecent(); };
    el.appendChild(d);
  }
}

async function openProject(body, errEl) {
  try {
    const j = await post("/api/open", body);
    location.href = "/label?p=" + j.pid;
  } catch (e) {
    if (errEl) errEl.textContent = e.message; else alert(e.message);
  }
}

// ------------------------------------------------------------------ new project
$("imgPath").addEventListener("input", () => { suggestProject(); checkFolder(); });
$("projPath").addEventListener("input", () => { projEdited = true; });

function suggestProject() {
  if (projEdited) return;
  const p = $("imgPath").value.trim().replace(/[\\/]+$/, "");
  $("projPath").value = p ? p + "_labels" : "";
}

async function checkFolder() {
  const p = $("imgPath").value.trim(), note = $("imgNote");
  if (!p) { note.textContent = ""; return; }
  clearTimeout(checkFolder.t);
  checkFolder.t = setTimeout(async () => {
    const r = await (await fetch("/api/browse?path=" + encodeURIComponent(p))).json();
    if (!r.found) { note.textContent = "Folder not found"; return; }
    const sub = r.dirs.length ? ` and ${r.dirs.length} subfolder${r.dirs.length > 1 ? "s" : ""}` : "";
    note.textContent = `${r.images} image${r.images === 1 ? "" : "s"} in this folder${sub}`;
    const s = await (await fetch("/api/summary?path=" + encodeURIComponent($("projPath").value.trim()))).json();
    if (s && s.name) note.textContent += ` · already a project (${s.done} done): Start opens it`;
  }, 250);
}

$("create").onclick = () => {
  const images = $("imgPath").value.trim();
  if (!images) { $("createErr").textContent = "Choose the folder of images first."; return; }
  $("createErr").textContent = "";
  openProject({images, project: $("projPath").value.trim() || null, classes: $("classes").value, max_side: +$("maxSide").value || 3072}, $("createErr"));
};
$("openGo").onclick = () => {
  const p = $("openPath").value.trim();
  if (!p) { $("openErr").textContent = "Choose a project folder."; return; }
  openProject({project: p}, $("openErr"));
};

// ------------------------------------------------------------------ folder browser
let fbTarget = null;
document.querySelectorAll("[data-browse]").forEach((b) => (b.onclick = () => {
  fbTarget = b.dataset.browse;
  const start = $(fbTarget).value.trim() || (fbTarget === "projPath" ? $("imgPath").value.trim() : "");
  browse(start); $("fb").showModal();
}));

async function browse(path) {
  const r = await (await fetch("/api/browse?path=" + encodeURIComponent(path || ""))).json();
  if (r.error && !r.dirs) { $("fbInfo").textContent = r.error; return; }
  $("fbPath").value = r.path; $("fbUp").disabled = !r.parent; $("fbUp").onclick = () => browse(r.parent);
  $("fbHome").onclick = () => browse(r.home);
  const L = $("fbList"); L.innerHTML = "";
  for (const d of r.dirs) {
    const e = document.createElement("div"); e.textContent = d.name; e.onclick = () => browse(d.path); L.appendChild(e);
  }
  if (!r.dirs.length) L.innerHTML = '<div style="color:var(--muted);cursor:default" class="none">no subfolders</div>';
  $("fbInfo").textContent = r.error ? r.error : `${r.images} image${r.images === 1 ? "" : "s"} directly in this folder`;
  L.scrollTop = 0;
}
$("fbGo").onclick = () => browse($("fbPath").value.trim());
$("fbPath").onkeydown = (e) => { if (e.key === "Enter") { e.preventDefault(); browse($("fbPath").value.trim()); } };
$("fbCancel").onclick = () => $("fb").close();
$("fbPick").onclick = () => {
  $(fbTarget).value = $("fbPath").value; $("fb").close();
  if (fbTarget === "imgPath") { suggestProject(); checkFolder(); }
  if (fbTarget === "projPath") projEdited = true;
};

// ------------------------------------------------------------------ Segment Anything
function showSam(s) {
  const el = $("sam");
  const dl = s.download;
  let html;
  if (!s.libraries) {
    html = `<div class="samline"><span class="dot"></span><div>Not installed. Smart click and object pieces need it; everything else works without.
      <div class="note">Install with <code>pip install "./OutlineLabeler[sam]"</code> (or <code>pip install torch segment-anything</code>), then start again.</div></div></div>`;
  } else if (s.ready) {
    html = `<div class="samline"><span class="dot on"></span><div><b>Ready</b>: ${esc(s.checkpoint.split(/[\\/]/).pop())}
      <div class="note">${s.gpu ? "Runs on the GPU with the most free memory" : "No GPU found: runs on the CPU (the small ViT-B model is the practical one there)"}; loaded the first time you use smart click.</div></div></div>`;
  } else if (!s.enabled) {
    html = `<div class="samline"><span class="dot"></span><div>Turned off.</div></div>`;
  } else {
    html = `<div class="samline"><span class="dot"></span><div>Installed, but no model file (checkpoint) was found.
      <div class="note">Download one (saved in <code>~/.outline-labeler/models</code>), or give the path of one you have.</div></div></div>
      <div class="row"><button data-dl="vit_b">Download ViT-B (375 MB${s.gpu ? "" : ", for CPUs"})</button><button data-dl="vit_h">Download ViT-H (2.6 GB${s.gpu ? ", best on a GPU" : ""})</button></div>`;
  }
  if (dl && dl.state === "running") html += `<div class="note">Downloading ${dl.model}: ${(dl.done / 1e6).toFixed(0)} of ${(dl.total / 1e6).toFixed(0)} MB</div><div class="bar"><i style="width:${(100 * dl.done / dl.total).toFixed(1)}%"></i></div>`;
  if (dl && dl.state === "failed") html += `<div class="err">Download failed: ${esc(dl.error)}</div>`;
  if (s.libraries) {
    const other = s.found.filter((f) => f !== s.checkpoint);
    html += `<details${s.ready ? "" : " open"}><summary>Settings</summary>
      <label class="f">Model file (checkpoint)</label>
      <div class="inrow"><input class="t" id="samPath" value="${esc(s.checkpoint || "")}" placeholder="…/sam_vit_h_4b8939.pth"><button id="samUse">Use</button></div>
      ${other.length ? `<div class="note">Also found: ${other.map((f) => `<a href="#" data-use="${esc(f)}">${esc(f.split(/[\\/]/).pop())}</a>`).join(", ")}</div>` : ""}
      <label class="f">Device</label>
      <div class="inrow"><input class="t" id="samDev" value="${esc(s.device === "auto" ? "" : s.device)}" placeholder="automatic (or cpu, cuda:0, cuda:1 …)"><button id="samDevGo">Set</button></div>
      <label class="f"><input type="checkbox" id="samOn" ${s.enabled ? "checked" : ""}> Use Segment Anything</label>
      <div class="err" id="samErr"></div></details>`;
  }
  el.innerHTML = html;
  el.querySelectorAll("[data-dl]").forEach((b) => (b.onclick = async () => { showSam(await post("/api/sam", {download: b.dataset.dl})); pollSam(); }));
  el.querySelectorAll("[data-use]").forEach((a) => (a.onclick = async (e) => { e.preventDefault(); setSam({checkpoint: a.dataset.use}); }));
  if ($("samUse")) $("samUse").onclick = () => setSam({checkpoint: $("samPath").value.trim()});
  if ($("samDevGo")) $("samDevGo").onclick = () => setSam({device: $("samDev").value.trim()});
  if ($("samOn")) $("samOn").onchange = () => setSam({enabled: $("samOn").checked});
  if (dl && dl.state === "running") pollSam();
}

async function setSam(body) {
  try { showSam(await post("/api/sam", body)); } catch (e) { if ($("samErr")) $("samErr").textContent = e.message; }
}

function pollSam() {
  clearTimeout(samTimer);
  samTimer = setTimeout(async () => { const h = await (await fetch("/api/home")).json(); showSam(h.sam); }, 1000);
}

load().catch((e) => { document.querySelector(".page").insertAdjacentHTML("beforeend", `<p class="err">Could not load: ${esc(e.message)}</p>`); });
