/**
 * Tests for js/utils/clip_preview_source.js — proxy-vs-fallback decision for SceneCastBuild's
 * clip video preview.
 */

import { resolveClipPreviewSource } from "../js/utils/clip_preview_source.js";

const CLIP = { id: "c1", start_time: 3.2, end_time: 9.75 };
const PROFILE_META = { media_filename: "raw.mp4", media_dir: "output" };

test("fresh proxy wins over the fallback", () => {
    const status = { clips: [{ clip_id: "c1", fresh: true, proxy_path: "/data/proxies/source_profiles/c1.mp4" }] };
    const r = resolveClipPreviewSource(status, CLIP, PROFILE_META);
    expect(r.kind).toBe("proxy");
    expect(r.proxyPath).toBe("/data/proxies/source_profiles/c1.mp4");
    expect(r.caption).toMatch(/silent/i);
});

test("stale/missing proxy falls back to the seeked full source", () => {
    const status = { clips: [{ clip_id: "c1", fresh: false, proxy_path: null }] };
    const r = resolveClipPreviewSource(status, CLIP, PROFILE_META);
    expect(r.kind).toBe("fallback");
    expect(r.filename).toBe("raw.mp4");
    expect(r.dir).toBe("output");
    expect(r.startTime).toBeCloseTo(3.2);
    expect(r.endTime).toBeCloseTo(9.75);
    expect(r.caption).toContain("3.2s");
    expect(r.caption).toContain("9.8s");
});

test("proxy entry absent entirely (clip not yet in a status response) falls back", () => {
    const status = { clips: [] };
    const r = resolveClipPreviewSource(status, CLIP, PROFILE_META);
    expect(r.kind).toBe("fallback");
});

test("no proxyStatus response yet (still loading / errored) falls back, not a crash", () => {
    const r = resolveClipPreviewSource(null, CLIP, PROFILE_META);
    expect(r.kind).toBe("fallback");
});

test("no clip selected is unavailable, not fallback", () => {
    const status = { clips: [{ clip_id: "c1", fresh: true, proxy_path: "/x.mp4" }] };
    expect(resolveClipPreviewSource(status, null, PROFILE_META).kind).toBe("unavailable");
});

test("missing profile media_filename is unavailable when no proxy exists either", () => {
    const status = { clips: [{ clip_id: "c1", fresh: false, proxy_path: null }] };
    const r = resolveClipPreviewSource(status, CLIP, { media_filename: "", media_dir: "input" });
    expect(r.kind).toBe("unavailable");
});

test("missing profile meta entirely still prefers a fresh proxy over unavailable", () => {
    const status = { clips: [{ clip_id: "c1", fresh: true, proxy_path: "/x.mp4" }] };
    const r = resolveClipPreviewSource(status, CLIP, null);
    expect(r.kind).toBe("proxy");
});

test("defaults media_dir to input when the profile omits it", () => {
    const status = { clips: [{ clip_id: "c1", fresh: false, proxy_path: null }] };
    const r = resolveClipPreviewSource(status, CLIP, { media_filename: "raw.mp4" });
    expect(r.dir).toBe("input");
});
