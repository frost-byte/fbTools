import { findRunEndTs, formatDuration, formatRunTime } from "../js/utils/run_time.js";

describe("findRunEndTs", () => {
    test("uses success, error or interrupted messages", () => {
        expect(findRunEndTs([["execution_start", { timestamp: 1 }], ["execution_success", { timestamp: 9 }]])).toBe(9);
        expect(findRunEndTs([["execution_error", { timestamp: 7 }]])).toBe(7);
        expect(findRunEndTs([["execution_interrupted", { timestamp: 5 }]])).toBe(5);
    });
    test("null while running or with no messages", () => {
        expect(findRunEndTs([["execution_start", { timestamp: 1 }]])).toBeNull();
        expect(findRunEndTs(undefined)).toBeNull();
    });
});

describe("formatDuration", () => {
    test("seconds, minutes, hours", () => {
        expect(formatDuration(4000)).toBe("4s");
        expect(formatDuration(606000)).toBe("10m 6s");
        expect(formatDuration(3725000)).toBe("1h 2m 5s");
        expect(formatDuration(0)).toBe("0s");
    });
    test("rejects negatives", () => {
        expect(formatDuration(-1)).toBe("");
    });
});

describe("formatRunTime", () => {
    const start = new Date(2026, 8, 20, 23, 42, 4).getTime();
    test("start -> end (duration) on the same day drops the end date", () => {
        const s = formatRunTime(start, start + 606000, "en-US");
        expect(s).toBe("9/20/2026, 11:42:04 PM → 11:52:10 PM (10m 6s)");
    });
    test("end on another day keeps the date", () => {
        const s = formatRunTime(start, start + 3600 * 1000, "en-US");
        expect(s).toContain("→ 9/21/2026, 12:42:04 AM (1h 0m 0s)");
    });
    test("no end shows only the start; no start shows nothing", () => {
        expect(formatRunTime(start, null, "en-US")).toBe("9/20/2026, 11:42:04 PM");
        expect(formatRunTime(null, null)).toBe("");
    });
});
