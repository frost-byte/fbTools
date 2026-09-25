/**
 * Tests for js/utils/segment_grouping.js — client-side, advisory grouping of detected segments
 * by their (per-window, unreliable) setting_label suggestion.
 */

import { normalizeSettingLabel, groupSuggestionsBySetting } from "../js/utils/segment_grouping.js";

// ── normalizeSettingLabel ────────────────────────────────────────────────────

test("trims and lowercases", () => {
    expect(normalizeSettingLabel("  Kitchen  ")).toBe("kitchen");
});

test("collapses internal whitespace", () => {
    expect(normalizeSettingLabel("Rooftop   Patio")).toBe("rooftop patio");
});

test("empty/undefined/null all normalize to empty string", () => {
    expect(normalizeSettingLabel("")).toBe("");
    expect(normalizeSettingLabel(undefined)).toBe("");
    expect(normalizeSettingLabel(null)).toBe("");
});

// ── groupSuggestionsBySetting ────────────────────────────────────────────────

test("empty input yields no groups", () => {
    expect(groupSuggestionsBySetting([])).toEqual([]);
});

test("all segments sharing a label land in one group", () => {
    const segs = [
        { start_time: 0, end_time: 5, setting_label: "Kitchen" },
        { start_time: 5, end_time: 10, setting_label: "Kitchen" },
        { start_time: 10, end_time: 15, setting_label: "Kitchen" },
    ];
    const groups = groupSuggestionsBySetting(segs);
    expect(groups).toHaveLength(1);
    expect(groups[0].indices).toEqual([0, 1, 2]);
    expect(groups[0].representativeIndex).toBe(0);
});

test("mixed labels preserve first-seen group order", () => {
    const segs = [
        { start_time: 0, end_time: 5, setting_label: "Kitchen" },
        { start_time: 5, end_time: 10, setting_label: "Bedroom" },
        { start_time: 10, end_time: 15, setting_label: "Kitchen" },
    ];
    const groups = groupSuggestionsBySetting(segs);
    expect(groups.map(g => g.label)).toEqual(["Kitchen", "Bedroom"]);
    expect(groups[0].indices).toEqual([0, 2]);
    expect(groups[1].indices).toEqual([1]);
});

test("case and whitespace differences still merge into one group", () => {
    const segs = [
        { start_time: 0, end_time: 5, setting_label: "Rooftop Patio" },
        { start_time: 5, end_time: 10, setting_label: "  rooftop   patio " },
    ];
    const groups = groupSuggestionsBySetting(segs);
    expect(groups).toHaveLength(1);
    expect(groups[0].indices).toEqual([0, 1]);
    // Label text shown to the user comes from whichever segment was seen first.
    expect(groups[0].label).toBe("Rooftop Patio");
});

test("missing or empty setting_label groups under (unlabeled)", () => {
    const segs = [
        { start_time: 0, end_time: 5 },
        { start_time: 5, end_time: 10, setting_label: "" },
        { start_time: 10, end_time: 15, setting_label: "   " },
    ];
    const groups = groupSuggestionsBySetting(segs);
    expect(groups).toHaveLength(1);
    expect(groups[0].label).toBe("(unlabeled)");
    expect(groups[0].indices).toEqual([0, 1, 2]);
});

test("representativeIndex is the first member encountered, not necessarily index 0 overall", () => {
    const segs = [
        { start_time: 0, end_time: 5, setting_label: "Bedroom" },
        { start_time: 5, end_time: 10, setting_label: "Kitchen" },
        { start_time: 10, end_time: 15, setting_label: "Kitchen" },
    ];
    const groups = groupSuggestionsBySetting(segs);
    const kitchen = groups.find(g => g.key === "kitchen");
    expect(kitchen.representativeIndex).toBe(1);
});
