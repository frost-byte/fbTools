/**
 * Tests for the tracked-node title helpers (marker form + legacy [track:] form)
 */

import {
    TRACK_MARKER, parseTrackTitle, formatTrackTitle, migrateLegacyTrackTitle, getTrackTag, setTrackTag,
} from "../js/utils/run_tracker.js";

describe("parseTrackTitle", () => {
    test("marker form", () => {
        expect(parseTrackTitle(`${TRACK_MARKER} Video Shift`)).toBe("Video Shift");
        expect(parseTrackTitle(`  ${TRACK_MARKER}   Audio Shift  `)).toBe("Audio Shift");
    });
    test("bare or empty marker is not tracked", () => {
        expect(parseTrackTitle(TRACK_MARKER)).toBeNull();
        expect(parseTrackTitle(`${TRACK_MARKER}   `)).toBeNull();
    });
    test("marker must lead the title", () => {
        expect(parseTrackTitle(`KSampler ${TRACK_MARKER}`)).toBeNull();
    });
    test("legacy form still parses", () => {
        expect(parseTrackTitle("Scene Cast Build [track: Cast]")).toBe("Cast");
        expect(parseTrackTitle("[track:   Padded  ]")).toBe("Padded");
    });
    test("plain / missing titles", () => {
        expect(parseTrackTitle("KSampler")).toBeNull();
        expect(parseTrackTitle("")).toBeNull();
        expect(parseTrackTitle(null)).toBeNull();
    });
});

describe("migrateLegacyTrackTitle", () => {
    test("rewrites a legacy title to the marker form using the label", () => {
        expect(migrateLegacyTrackTitle("Scene Cast Build [track: Cast]")).toBe(`${TRACK_MARKER} Cast`);
    });
    test("empty legacy label falls back to the rest of the title", () => {
        expect(migrateLegacyTrackTitle("Video Shift [track: ]")).toBe(`${TRACK_MARKER} Video Shift`);
    });
    test("already-migrated or untracked titles need no change", () => {
        expect(migrateLegacyTrackTitle(`${TRACK_MARKER} Cast`)).toBeNull();
        expect(migrateLegacyTrackTitle("KSampler")).toBeNull();
    });
});

describe("get/setTrackTag", () => {
    test("set writes the marker form and get reads it back", () => {
        const node = { title: "KSampler" };
        setTrackTag(node, "Sampler A");
        expect(node.title).toBe(formatTrackTitle("Sampler A"));
        expect(getTrackTag(node)).toBe("Sampler A");
    });
    test("clearing untracks and leaves the label as the plain title", () => {
        const node = { title: formatTrackTitle("Sampler A") };
        setTrackTag(node, null);
        expect(node.title).toBe("Sampler A");
        expect(getTrackTag(node)).toBeNull();
    });
});
