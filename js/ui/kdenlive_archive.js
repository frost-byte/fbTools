/**
 * Kdenlive Archive tab.
 *
 * Make a .kdenlive project portable: copy its clips into one folder, rewrite
 * the project to relative paths, optionally strip the embedded ComfyUI
 * workflow metadata. Same engine as scripts/kdenlive_archive.py.
 * Long archive jobs report progress via "fbtools.status" (source
 * "kdenlive_archive"); status() lets a reloaded page pick a job back up.
 */

import { kdenliveApi } from "../api/kdenlive.js";
import { api } from "../../../scripts/api.js";

const LS_KEY = "fbt_kdenlive_archive_form";
const SOURCE = "kdenlive_archive";

function _mk(tag, props = {}, children = []) {
    const el = document.createElement(tag);
    Object.entries(props).forEach(([k, v]) => {
        if (k === "cls")        el.className = v;
        else if (k === "style") Object.assign(el.style, v);
        else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
        else el[k] = v;
    });
    children.filter(Boolean).forEach(c =>
        el.appendChild(typeof c === "string" ? document.createTextNode(c) : c));
    return el;
}

function _fmtBytes(n) {
    if (n < 1024) return `${n} B`;
    const units = ["KB", "MB", "GB", "TB"];
    let i = -1;
    do { n /= 1024; i++; } while (n >= 1024 && i < units.length - 1);
    return `${n.toFixed(1)} ${units[i]}`;
}

function _loadForm() {
    try { return JSON.parse(localStorage.getItem(LS_KEY) || "{}"); } catch { return {}; }
}
function _saveForm(v) {
    try { localStorage.setItem(LS_KEY, JSON.stringify(v)); } catch {}
}

export async function renderKdenliveArchive(parent) {
    const saved = _loadForm();
    const wrap = _mk("div", { cls: "fbt-ka-wrap" });
    parent.appendChild(wrap);

    wrap.appendChild(_mk("p", { cls: "fbt-ka-intro", textContent:
        "Archive a Kdenlive project into one portable folder (clips in media/, relative paths) " +
        "that opens on Windows, macOS and Linux. Paths are on the ComfyUI server." }));

    const field = (label, control, hint) =>
        _mk("div", { cls: "fbt-ka-field" }, [
            _mk("span", { cls: "fbt-ka-label", textContent: label }), control,
            hint ? _mk("span", { cls: "fbt-ka-hint", textContent: hint }) : null,
        ]);
    const input = (val, ph) => _mk("input", { cls: "fbt-ka-input", type: "text", value: val || "", placeholder: ph });
    const area = (val, ph, rows = 2) => _mk("textarea", { cls: "fbt-ka-textarea", value: val || "", placeholder: ph, rows });

    const projectEl = input(saved.project, "/path/to/project.kdenlive");
    const destEl    = input(saved.dest, "/path/to/archive_folder");
    const searchEl  = area(saved.search, "one folder per line");
    const mapsEl    = area(saved.maps, "Z:/=/mnt/data/   (one SRC=DST per line)");
    const stripEl   = _mk("input", { type: "checkbox", checked: saved.strip !== false });
    const dryEl     = _mk("input", { type: "checkbox", checked: !!saved.dry });

    wrap.appendChild(field("Project (.kdenlive)", projectEl));
    wrap.appendChild(field("Destination folder", destEl, "Created if missing. Existing identical files are skipped, so re-running resumes."));
    wrap.appendChild(field("Search folders for missing clips", searchEl,
        "Looked at by filename when a clip isn't at its recorded path (e.g. you moved clips)."));
    wrap.appendChild(field("Path maps", mapsEl,
        "Translate a path prefix from another machine, e.g. a Windows drive letter or network share."));
    wrap.appendChild(_mk("div", { cls: "fbt-ka-checks" }, [
        _mk("label", { cls: "fbt-ka-check" }, [stripEl, "Strip embedded workflow metadata"]),
        _mk("label", { cls: "fbt-ka-check" }, [dryEl, "Dry run (archive)"]),
    ]));

    const btn = (label, onclick, cls = "") => _mk("button", { cls: `fbt-ka-btn ${cls}`, textContent: label, onclick });
    const checkBtn   = btn("Check", () => _check());
    const archiveBtn = btn("Archive", () => _archive(), "fbt-ka-btn-primary");
    const stripBtn   = btn("Strip metadata only", () => _stripOnly());
    const cancelBtn  = btn("Cancel", () => _cancel());
    cancelBtn.style.display = "none";
    wrap.appendChild(_mk("div", { cls: "fbt-ka-actions" }, [checkBtn, archiveBtn, stripBtn, cancelBtn]));

    const barEl    = _mk("div", { cls: "fbt-ka-progress-bar" });
    const progEl   = _mk("div", { cls: "fbt-ka-progress" }, [barEl]);
    const statusEl = _mk("div", { cls: "fbt-ka-status" });
    const reportEl = _mk("div", { cls: "fbt-ka-report" });
    wrap.append(progEl, statusEl, reportEl);

    let jobId = null;
    let pollTimer = null;

    const persist = () => _saveForm({
        project: projectEl.value, dest: destEl.value, search: searchEl.value,
        maps: mapsEl.value, strip: stripEl.checked, dry: dryEl.checked,
    });
    [projectEl, destEl, searchEl, mapsEl, stripEl, dryEl].forEach(el => el.addEventListener("change", persist));

    const lines = (el) => el.value.split("\n").map(s => s.trim()).filter(Boolean);
    const opts = () => ({ project: projectEl.value.trim(), path_maps: lines(mapsEl), search_dirs: lines(searchEl) });
    const say = (msg, isErr = false) => { statusEl.textContent = msg; statusEl.classList.toggle("error", isErr); };
    const busy = (on) => {
        [checkBtn, archiveBtn, stripBtn].forEach(b => { b.disabled = on; });
        cancelBtn.style.display = on && jobId ? "" : "none";
    };
    const setProgress = (done, total) => {
        progEl.style.display = total ? "block" : "none";
        barEl.style.width = total ? `${Math.round((100 * done) / total)}%` : "0";
    };

    function renderReport(rep) {
        reportEl.innerHTML = "";
        if (!rep) return;
        const rows = [
            ["References", `${rep.references} (${rep.resolved} resolved, ${rep.unresolved_count} missing)`],
            ["Unique files", `${rep.unique_files} (${_fmtBytes(rep.total_bytes)})`],
        ];
        if (rep.by_method && Object.keys(rep.by_method).length)
            rows.push(["Resolved via", Object.entries(rep.by_method).map(([k, v]) => `${k}: ${v}`).join(", ")]);
        if (rep.output_project) {
            rows.push([rep.dry_run ? "Would write" : "Wrote", rep.output_project]);
            if (!rep.dry_run) rows.push(["Copied", `${rep.copied} files (${_fmtBytes(rep.bytes_copied)}), ${rep.skipped_existing} already present`]);
            if (rep.stripped?.removed) rows.push(["Metadata", `stripped ${rep.stripped.removed} properties; project ${_fmtBytes(rep.size_before)} → ${_fmtBytes(rep.size_after)}`]);
        }
        const dl = _mk("dl", { cls: "fbt-ka-summary" });
        rows.forEach(([k, v]) => dl.append(_mk("dt", { textContent: k }), _mk("dd", { textContent: v })));
        reportEl.appendChild(dl);

        const list = (title, items, cls) => {
            if (!items?.length) return;
            reportEl.appendChild(_mk("div", { cls: "fbt-ka-h", textContent: title }));
            reportEl.appendChild(_mk("ul", { cls: `fbt-ka-list ${cls}` }, items.map(t => _mk("li", { textContent: t }))));
        };
        list(`Missing clips (${rep.unresolved_count}) — left untouched`, rep.unresolved, "missing");
        list("Ambiguous matches (first candidate used)", (rep.ambiguous || []).map(a => `${a.reference} → ${a.chosen}`), "warn");
        list("Warnings", rep.warnings, "warn");
        list("Notes", rep.notes, "");
    }

    async function _run(label, fn) {
        say(`${label}…`);
        reportEl.innerHTML = "";
        setProgress(0, 0);
        busy(true);
        try { await fn(); }
        catch (e) { say(e?.message || String(e), true); }
        finally { if (!jobId) busy(false); }
    }

    function _check() {
        persist();
        return _run("Checking", async () => {
            const { report } = await kdenliveApi.check(opts());
            renderReport(report);
            say(report.unresolved_count ? `${report.unresolved_count} clip(s) could not be found.` : "All clips found.");
        });
    }

    function _stripOnly() {
        persist();
        return _run("Stripping", async () => {
            const inPlace = window.confirm(
                "Edit the project in place?\n\nOK = edit in place (a .bak of the original is kept)\nCancel = write <name>_stripped.kdenlive");
            const { report } = await kdenliveApi.strip({ project: projectEl.value.trim(), in_place: inPlace });
            reportEl.innerHTML = "";
            say(`Removed ${report.removed} properties: ${_fmtBytes(report.size_before)} → ${_fmtBytes(report.size_after)} (${report.output})`);
        });
    }

    function _archive() {
        persist();
        return _run("Starting archive", async () => {
            const res = await kdenliveApi.archive({
                ...opts(), dest: destEl.value.trim(), strip_metadata: stripEl.checked, dry_run: dryEl.checked,
            });
            jobId = res.job_id;
            busy(true);
            startPolling();
        });
    }

    async function _cancel() {
        if (!jobId) return;
        cancelBtn.disabled = true;
        say("Cancelling…");
        try { await kdenliveApi.cancel(jobId); } catch (e) { say(e?.message || String(e), true); }
    }

    function finish(job) {
        clearInterval(pollTimer);
        pollTimer = null;
        jobId = null;
        cancelBtn.disabled = false;
        busy(false);
        setProgress(0, 0);
        if (job.state === "error") { say(job.error || "Archive failed", true); return; }
        renderReport(job.report);
        const missing = job.report?.unresolved_count || 0;
        say(job.state === "cancelled" ? "Cancelled — project file was not written."
            : `${job.report?.dry_run ? "Dry run finished" : "Archive complete"}${missing ? ` — ${missing} clip(s) missing` : ""}.`);
    }

    async function pollOnce() {
        try {
            const { job } = await kdenliveApi.status(jobId || "");
            if (!job) return;
            if (job.state === "running") {
                jobId = job.id;
                const p = job.progress || {};
                if (p.total) setProgress(p.done || 0, p.total);
                busy(true);
            } else if (jobId === job.id) {
                finish(job);
            }
        } catch { /* transient; next tick retries */ }
    }

    function startPolling() {
        clearInterval(pollTimer);
        pollTimer = setInterval(pollOnce, 1500);
    }

    // Live progress between polls.
    api.addEventListener("fbtools.status", (ev) => {
        const d = ev.detail || {};
        // "kind" disambiguates from the Clean clips section below, which shares this same
        // websocket event (both are source: "kdenlive_archive") but runs as an independent job.
        if (d.source !== SOURCE || d.kind !== "archive") return;
        if (d.job_id && jobId && d.job_id !== jobId) return;
        if (d.status) say(d.status, d.level === "error");
        if (d.progress?.total) setProgress(d.progress.done || 0, d.progress.total);
        if (d.finished) pollOnce();
    });

    // Pick up an archive job already running (page reloaded mid-archive).
    try {
        const { job } = await kdenliveApi.status("", "archive");
        if (job?.state === "running") {
            jobId = job.id;
            busy(true);
            say("Archive in progress…");
            startPolling();
        }
    } catch { /* server unreachable; leave idle */ }

    await renderCleanClips(wrap);
}

// ── Clean clips ──────────────────────────────────────────────────────────────────
// Remux every video in a folder without embedded metadata (a ComfyUI-saved clip's own
// workflow/prompt JSON) before adding it to a project's media folder by hand. Never
// touches a .kdenlive file, so it runs independently of the archive job above — the
// server only refuses two jobs of the *same* kind at once.

const CLEAN_LS_KEY = "fbt_kdenlive_clean_form";

async function renderCleanClips(parent) {
    const saved = (() => { try { return JSON.parse(localStorage.getItem(CLEAN_LS_KEY) || "{}"); } catch { return {}; } })();
    const persistClean = (v) => { try { localStorage.setItem(CLEAN_LS_KEY, JSON.stringify(v)); } catch {} };

    parent.appendChild(_mk("hr", { cls: "fbt-ka-divider" }));
    parent.appendChild(_mk("p", { cls: "fbt-ka-intro", textContent:
        "Clean generated clips before adding them to a project's media folder: remux every video " +
        "directly under a source folder (not recursive) into a destination folder with its embedded " +
        "workflow/prompt metadata stripped. The originals are never touched." }));

    const field = (label, control, hint) =>
        _mk("div", { cls: "fbt-ka-field" }, [
            _mk("span", { cls: "fbt-ka-label", textContent: label }), control,
            hint ? _mk("span", { cls: "fbt-ka-hint", textContent: hint }) : null,
        ]);
    const input = (val, ph) => _mk("input", { cls: "fbt-ka-input", type: "text", value: val || "", placeholder: ph });

    const srcEl  = input(saved.src, "/path/to/output/video/process_me");
    const destEl = input(saved.dest, "/path/to/archive/media/comps");
    const dryEl  = _mk("input", { type: "checkbox", checked: !!saved.dry });

    parent.appendChild(field("Source folder", srcEl, "Clips to clean (not searched recursively)."));
    parent.appendChild(field("Destination folder", destEl, "Created if missing. Files already there are left alone, so re-running only cleans what's new."));
    parent.appendChild(_mk("div", { cls: "fbt-ka-checks" }, [
        _mk("label", { cls: "fbt-ka-check" }, [dryEl, "Dry run"]),
    ]));

    const btn = (label, onclick, cls = "") => _mk("button", { cls: `fbt-ka-btn ${cls}`, textContent: label, onclick });
    const cleanBtn  = btn("Clean", () => runClean(), "fbt-ka-btn-primary");
    const cancelBtn = btn("Cancel", () => cancelClean());
    cancelBtn.style.display = "none";
    parent.appendChild(_mk("div", { cls: "fbt-ka-actions" }, [cleanBtn, cancelBtn]));

    const barEl    = _mk("div", { cls: "fbt-ka-progress-bar" });
    const progEl   = _mk("div", { cls: "fbt-ka-progress" }, [barEl]);
    const statusEl = _mk("div", { cls: "fbt-ka-status" });
    const reportEl = _mk("div", { cls: "fbt-ka-report" });
    parent.append(progEl, statusEl, reportEl);

    let jobId = null;
    let pollTimer = null;

    const persist = () => persistClean({ src: srcEl.value, dest: destEl.value, dry: dryEl.checked });
    [srcEl, destEl, dryEl].forEach(el => el.addEventListener("change", persist));

    const say = (msg, isErr = false) => { statusEl.textContent = msg; statusEl.classList.toggle("error", isErr); };
    const busy = (on) => { cleanBtn.disabled = on; cancelBtn.style.display = on && jobId ? "" : "none"; };
    const setProgress = (done, total) => {
        progEl.style.display = total ? "block" : "none";
        barEl.style.width = total ? `${Math.round((100 * done) / total)}%` : "0";
    };

    function renderCleanReport(rep) {
        reportEl.innerHTML = "";
        if (!rep) return;
        const rows = [["Videos found", String(rep.files_total)]];
        if (rep.dry_run) {
            rows.push(["Would clean", String(rep.results.filter(r => r.status === "would_clean").length)]);
        } else {
            rows.push(["Cleaned", `${rep.cleaned} (${rep.skipped_existing} already present)`]);
            if (rep.cleaned) rows.push(["Size", `${_fmtBytes(rep.bytes_before)} → ${_fmtBytes(rep.bytes_after)} (${_fmtBytes(rep.bytes_saved)} saved)`]);
        }
        const dl = _mk("dl", { cls: "fbt-ka-summary" });
        rows.forEach(([k, v]) => dl.append(_mk("dt", { textContent: k }), _mk("dd", { textContent: v })));
        reportEl.appendChild(dl);

        const list = (title, items, cls) => {
            if (!items?.length) return;
            reportEl.appendChild(_mk("div", { cls: "fbt-ka-h", textContent: title }));
            reportEl.appendChild(_mk("ul", { cls: `fbt-ka-list ${cls}` }, items.map(t => _mk("li", { textContent: t }))));
        };
        list("Duplicate content (nothing removed — cleaned independently)",
            (rep.duplicates || []).map(g => g.map(p => p.split("/").pop()).join(", ")), "warn");
        list("Errors — left uncleaned", (rep.errors || []).map(e => `${e.file}: ${e.error}`), "missing");
    }

    async function runClean() {
        persist();
        say("Starting…");
        reportEl.innerHTML = "";
        setProgress(0, 0);
        busy(true);
        try {
            const res = await kdenliveApi.clean({
                src_dir: srcEl.value.trim(), dest_dir: destEl.value.trim(), dry_run: dryEl.checked,
            });
            jobId = res.job_id;
            busy(true);
            startPolling();
        } catch (e) { say(e?.message || String(e), true); busy(false); }
    }

    async function cancelClean() {
        if (!jobId) return;
        cancelBtn.disabled = true;
        say("Cancelling…");
        try { await kdenliveApi.cancel(jobId); } catch (e) { say(e?.message || String(e), true); }
    }

    function finishClean(job) {
        clearInterval(pollTimer);
        pollTimer = null;
        jobId = null;
        cancelBtn.disabled = false;
        busy(false);
        setProgress(0, 0);
        if (job.state === "error") { say(job.error || "Clean failed", true); return; }
        renderCleanReport(job.report);
        const errs = job.report?.errors?.length || 0;
        say(job.state === "cancelled" ? "Cancelled."
            : `${job.report?.dry_run ? "Dry run finished" : "Clean complete"}${errs ? ` — ${errs} error(s)` : ""}.`);
    }

    async function pollOnce() {
        try {
            const { job } = await kdenliveApi.status(jobId || "", "clean");
            if (!job) return;
            if (job.state === "running") {
                jobId = job.id;
                const p = job.progress || {};
                if (p.total) setProgress(p.done || 0, p.total);
                busy(true);
            } else if (jobId === job.id) {
                finishClean(job);
            }
        } catch { /* transient; next tick retries */ }
    }

    function startPolling() {
        clearInterval(pollTimer);
        pollTimer = setInterval(pollOnce, 1500);
    }

    api.addEventListener("fbtools.status", (ev) => {
        const d = ev.detail || {};
        if (d.source !== SOURCE || d.kind !== "clean") return;
        if (d.job_id && jobId && d.job_id !== jobId) return;
        if (d.status) say(d.status, d.level === "error");
        if (d.progress?.total) setProgress(d.progress.done || 0, d.progress.total);
        if (d.finished) pollOnce();
    });

    // Pick up a clean job already running (page reloaded mid-clean).
    try {
        const { job } = await kdenliveApi.status("", "clean");
        if (job?.state === "running") {
            jobId = job.id;
            busy(true);
            say("Clean in progress…");
            startPolling();
        }
    } catch { /* server unreachable; leave idle */ }
}
