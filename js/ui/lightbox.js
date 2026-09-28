/**
 * Shared click-to-zoom image lightbox.
 *
 * Nothing like this existed yet — `.fbt-be-img-preview-clickable`/`.fbt-be-lightbox-img` were
 * dead CSS (defined, never referenced) until this module. Reuses the same full-screen backdrop
 * pattern every other modal in this codebase already uses (`.fbt-ce-modal-overlay`).
 */

function _mk(tag, props = {}, children = []) {
    const el = document.createElement(tag);
    Object.entries(props).forEach(([k, v]) => {
        if (k === "cls") el.className = v;
        else if (k === "style") Object.assign(el.style, v);
        else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
        else el[k] = v;
    });
    children.filter(Boolean).forEach(c => el.appendChild(c));
    return el;
}

/** Open `src` full-size in a click-to-dismiss overlay. */
export function openImageLightbox(src, alt = "") {
    if (!src) return;
    const img = _mk("img", { cls: "fbt-be-lightbox-img", src, alt });
    const overlay = _mk("div", { cls: "fbt-ce-modal-overlay" }, [img]);
    // Only the backdrop itself dismisses — clicking the enlarged image shouldn't close it.
    overlay.addEventListener("click", (e) => { if (e.target === overlay) overlay.remove(); });
    document.body.appendChild(overlay);
}

/** Wire `img` to open itself in the lightbox on click; adds the zoom-cursor class too. */
export function makeImageZoomable(img) {
    img.classList.add("fbt-be-img-preview-clickable");
    img.title = img.title || "Click to enlarge";
    img.addEventListener("click", () => openImageLightbox(img.src, img.alt || ""));
    return img;
}
