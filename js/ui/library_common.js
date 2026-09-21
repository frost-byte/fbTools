/** Small DOM/UI helpers shared by the asset editors (Compose, Assets tab). */

export function mk(tag, props = {}, children = []) {
    const el = document.createElement(tag);
    Object.entries(props).forEach(([k, v]) => {
        if (k === "cls") el.className = v;
        else if (k === "style") Object.assign(el.style, v);
        else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
        else el[k] = v;
    });
    children.forEach(c => c && el.appendChild(c));
    return el;
}

export function toast(msg, severity = "info") {
    try {
        const app = window._fbtApp;
        if (app?.extensionManager?.toast) {
            app.extensionManager.toast.add({ severity, summary: msg, life: 2500 });
        }
    } catch (_) {}
}

/** ComfyUI /view URL for "subdir/name.ext" inside input/output/temp. */
export function ceViewUrl(relPath, folder = "input") {
    const slash = relPath.lastIndexOf("/");
    const name  = slash === -1 ? relPath : relPath.slice(slash + 1);
    const sub   = slash === -1 ? ""      : relPath.slice(0, slash);
    return `/view?filename=${encodeURIComponent(name)}&type=${folder}&subfolder=${encodeURIComponent(sub)}`;
}

const VIDEO_EXTS = new Set([".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".wmv"]);
export function isVideoFile(name) {
    const dot = name.lastIndexOf(".");
    return dot >= 0 && VIDEO_EXTS.has(name.slice(dot).toLowerCase());
}

export const DEFAULT_OUTFIT_QUERY =
    "Describe the outfit in detail for video generation prompts. " +
    "Focus on garment types, colors, materials, textures, patterns, and accessories. " +
    "Do not describe the person's face, hair, or pose.";
