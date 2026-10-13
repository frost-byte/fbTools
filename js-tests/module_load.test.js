/**
 * Every extension module must load in a browser-like environment.
 *
 * A top-level error or a bad import in any module (for example one imported by js/fb_tools.js)
 * stops the whole extension UI from loading, and `node --check` cannot see it. This copies js/ next
 * to stub ComfyUI scripts and imports each module in a child process, so a stray statement or a
 * missing export fails here instead of in the browser.
 */

import { execFileSync } from "child_process";
import { cpSync, mkdtempSync, mkdirSync, writeFileSync, rmSync, symlinkSync } from "fs";
import { tmpdir } from "os";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");

// Modules that cannot load under the stubs, with the reason.
const SKIP = new Set([
    "nodes/video.js",   // imports ComfyUI's app by absolute URL ("/scripts/app.js")
]);

const RUNNER = `
import { JSDOM } from "jsdom";
import { readdirSync, statSync } from "fs";
import { join } from "path";
const dom = new JSDOM("<!doctype html><body></body>", { pretendToBeVisual: true, url: "http://127.0.0.1:8188/" });
globalThis.window = dom.window; globalThis.document = dom.window.document;
for (const k of ["HTMLElement", "Node", "KeyboardEvent", "Event", "localStorage", "navigator"]) { try { globalThis[k] = dom.window[k]; } catch {} }
globalThis.ResizeObserver = class { observe() {} disconnect() {} };
globalThis.requestAnimationFrame = f => setTimeout(f, 0);
const skip = new Set(JSON.parse(process.argv[2]));
const walk = d => readdirSync(d).flatMap(f => { const p = join(d, f); return statSync(p).isDirectory() ? walk(p) : p.endsWith(".js") ? [p] : []; });
const failures = [];
let count = 0;
for (const f of walk("ext/comfyui-fbTools")) {
    const rel = f.replace("ext/comfyui-fbTools/", "");
    if (skip.has(rel)) continue;
    count++;
    try { await import("./" + f); } catch (e) { failures.push(rel + ": " + String(e.message).slice(0, 200)); }
}
console.log(JSON.stringify({ count, failures }));
process.exit(0);
`;

test("every extension module loads without throwing", () => {
    const dir = mkdtempSync(join(tmpdir(), "fbt-load-"));
    try {
        mkdirSync(join(dir, "scripts"));
        writeFileSync(join(dir, "scripts", "api.js"),
            "export const api = { addEventListener() {}, removeEventListener() {}, fetchApi: () => Promise.resolve({ ok: false, json: async () => ({}) }), apiURL: r => r, fileURL: r => r };");
        writeFileSync(join(dir, "scripts", "app.js"),
            "export const app = { graph: { links: {}, _nodes: [], getNodeById() { return null; }, setDirtyCanvas() {} }, canvas: {}, " +
            "registerExtension() {}, extensionManager: { registerSidebarTab() {} }, ui: { settings: { addSetting() {}, getSettingValue() {} } } };");
        writeFileSync(join(dir, "scripts", "ui.js"),
            "export const $el = (tag, props) => Object.assign(document.createElement(tag), props || {}); export const ComfyDialog = class {};");
        mkdirSync(join(dir, "ext"));
        cpSync(join(ROOT, "js"), join(dir, "ext", "comfyui-fbTools"), { recursive: true });
        writeFileSync(join(dir, "run.mjs"), RUNNER);
        symlinkSync(join(ROOT, "node_modules"), join(dir, "node_modules"));   // jsdom, from the repo
        const out = execFileSync(process.execPath, ["run.mjs", JSON.stringify([...SKIP])], { cwd: dir, encoding: "utf8" });
        const { count, failures } = JSON.parse(out.trim().split("\n").pop());
        expect(count).toBeGreaterThan(50);
        expect(failures).toEqual([]);
    } finally {
        rmSync(dir, { recursive: true, force: true });
    }
}, 60000);
