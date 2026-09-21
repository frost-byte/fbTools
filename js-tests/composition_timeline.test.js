/**
 * Tests for composition timeline helpers
 */

import {
    parseShotTimestamp, shotsToSegments, shotDisplayText, substituteSlotPlaceholders,
} from "../js/utils/composition_timeline.js";

describe("parseShotTimestamp", () => {
    test("parses MM:SS.mmm, HH:MM:SS and bare seconds", () => {
        expect(parseShotTimestamp("00:03.500")).toBeCloseTo(3.5);
        expect(parseShotTimestamp("01:02.250")).toBeCloseTo(62.25);
        expect(parseShotTimestamp("1:00:00")).toBe(3600);
        expect(parseShotTimestamp("7.5")).toBeCloseTo(7.5);
    });

    test("null, empty and malformed values give null", () => {
        expect(parseShotTimestamp(null)).toBeNull();
        expect(parseShotTimestamp(undefined)).toBeNull();
        expect(parseShotTimestamp("")).toBeNull();
        expect(parseShotTimestamp("abc")).toBeNull();
        expect(parseShotTimestamp("1:2:3:4")).toBeNull();
        expect(parseShotTimestamp("00:-3")).toBeNull();
    });
});

describe("shotsToSegments", () => {
    const shot = (id, ts, action = "{A} walks in") => ({ id, timestamp: ts, action, camera: "", dialogue: null });

    test("increasing timestamps give contiguous segments, last one uses the median gap", () => {
        const segs = shotsToSegments([shot("s1", "00:00.000"), shot("s2", "00:04.000"), shot("s3", "00:10.000")]);
        expect(segs.map(s => [s.start_time, s.end_time])).toEqual([[0, 4], [4, 10], [10, 15]]);
    });

    test("single timestamped shot ends one second later", () => {
        const [seg] = shotsToSegments([shot("s1", "00:02.000")]);
        expect([seg.start_time, seg.end_time]).toEqual([2, 3]);
    });

    test("null timestamps leave times unset for equal-width bands", () => {
        const segs = shotsToSegments([shot("s1", null), shot("s2", null)]);
        expect(segs.every(s => s.start_time === undefined && s.end_time === undefined)).toBe(true);
    });

    test("mixed null and real timestamps fall back to equal bands", () => {
        const segs = shotsToSegments([shot("s1", "00:00.000"), shot("s2", null)]);
        expect(segs[0].start_time).toBeUndefined();
    });

    test("non-increasing timestamps fall back to equal bands", () => {
        const segs = shotsToSegments([shot("s1", "00:05.000"), shot("s2", "00:05.000")]);
        expect(segs[1].end_time).toBeUndefined();
    });

    test("labels are numbered with the first words of the action, placeholders removed", () => {
        const [seg] = shotsToSegments([shot("s1", null, "{A} walks slowly across the crowded dance floor tonight")]);
        expect(seg.label).toBe("1 · walks slowly across the crowded…");
        expect(seg.id).toBe("s1");
    });

    test("empty or missing input", () => {
        expect(shotsToSegments([])).toEqual([]);
        expect(shotsToSegments(null)).toEqual([]);
    });

    test("shots without an id get a stable fallback", () => {
        expect(shotsToSegments([{ action: "x" }, { action: "y" }]).map(s => s.id)).toEqual(["shot_1", "shot_2"]);
    });
});

describe("shotDisplayText", () => {
    test("joins camera, action and dialogue, keeping placeholders", () => {
        const text = shotDisplayText({
            camera: "Close-up of {A}", action: "{A} turns to {B}.",
            dialogue: { speaker: "A", text: "Hello." },
        });
        expect(text).toBe('Close-up of {A}\n{A} turns to {B}.\n{A}: "Hello."');
    });

    test("skips empty parts", () => {
        expect(shotDisplayText({ camera: "", action: "Only action", dialogue: null })).toBe("Only action");
        expect(shotDisplayText({})).toBe("");
    });
});

describe("substituteSlotPlaceholders", () => {
    test("replaces known slots from an object or a Map", () => {
        expect(substituteSlotPlaceholders("{A} meets {B}", { A: "Alex", B: "Bob" })).toBe("[Alex] meets [Bob]");
        expect(substituteSlotPlaceholders("{A} meets {B}", new Map([["A", "Alex"], ["B", "Bob"]]))).toBe("[Alex] meets [Bob]");
    });

    test("handles multi-letter slots and adjacent placeholders", () => {
        expect(substituteSlotPlaceholders("{AA}{B}", { AA: "Zed", B: "Bob" })).toBe("[Zed][Bob]");
    });

    test("leaves unknown or non-slot tokens alone", () => {
        expect(substituteSlotPlaceholders("{C} {BG} {a} {Fit_1}", { A: "x" })).toBe("{C} {BG} {a} {Fit_1}");
    });

    test("null text gives an empty string", () => {
        expect(substituteSlotPlaceholders(null, {})).toBe("");
    });
});
