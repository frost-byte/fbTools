/**
 * Regression test for the shot-id collision bug: shot ids used to come from a session-global
 * counter (_S.shotSeq) that never re-synced to a loaded composition's own shots, so the first
 * "+ Add Shot" click after opening a composition could mint an id already used by one of its
 * existing shots (a real composition hit this: two shots both ended up "shot_1"). Ids are now
 * derived purely from the composition's own shot list.
 */
import { _nextShotId } from "../js/ui/composition_editor.js";

describe("_nextShotId", () => {
    test("shot_1 for an empty or missing shot list", () => {
        expect(_nextShotId([])).toBe("shot_1");
        expect(_nextShotId(undefined)).toBe("shot_1");
    });

    test("continues after the highest existing shot_N, regardless of order", () => {
        expect(_nextShotId([{ id: "shot_1" }])).toBe("shot_2");
        expect(_nextShotId([{ id: "shot_3" }, { id: "shot_1" }])).toBe("shot_4");
    });

    test("never collides with an existing id — the original bug", () => {
        // A composition freshly loaded into a session where nothing has reset any counter:
        // the next id must still account for what's already in THIS composition's shots.
        const shots = [{ id: "shot_1" }];
        const next = _nextShotId(shots);
        expect(shots.some(s => s.id === next)).toBe(false);
        expect(next).toBe("shot_2");
    });

    test("ignores non-matching or malformed ids", () => {
        expect(_nextShotId([{ id: "intro" }, { id: "shot_abc" }, { id: "shot_2" }])).toBe("shot_3");
    });
});
