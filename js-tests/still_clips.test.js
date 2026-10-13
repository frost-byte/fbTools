import { layoutStillClips, clipFrame } from "../js/utils/still_clips.js";

describe("layoutStillClips", () => {
    test("overlapping 0..5 clips are laid end to end", () => {
        const out = layoutStillClips([
            { id: "a", start_time: 0, end_time: 5 },
            { id: "b", start_time: 0, end_time: 5 },
            { id: "c", start_time: 0, end_time: 3 },
        ]);
        expect(out.map(c => [c.start_time, c.end_time])).toEqual([[0, 5], [5, 10], [10, 13]]);
    });

    test("a clip without a positive length gets the fallback", () => {
        const out = layoutStillClips([{ id: "a", start_time: 0, end_time: 0 }, { id: "b", start_time: 2, end_time: 2 }], 4);
        expect(out.map(c => c.end_time)).toEqual([4, 8]);
    });

    test("does not mutate the input and tolerates empty input", () => {
        const src = [{ id: "a", start_time: 7, end_time: 9 }];
        layoutStillClips(src);
        expect(src[0].start_time).toBe(7);
        expect(layoutStillClips(null)).toEqual([]);
    });
});

describe("clipFrame", () => {
    const profile = {
        media_filename: "p.png", media_dir: "output",
        frames: [{ id: "f1", filename: "a.png", dir: "output" }, { id: "f2", filename: "b.png", dir: "input", label: "B" }],
    };
    test("picks the clip's frame, then the first, then the primary media", () => {
        expect(clipFrame(profile, { frame_id: "f2" })).toEqual({ filename: "b.png", dir: "input", label: "B" });
        expect(clipFrame(profile, { frame_id: "gone" }).filename).toBe("a.png");
        expect(clipFrame({ media_filename: "p.png" }, {})).toMatchObject({ filename: "p.png", dir: "input" });
        expect(clipFrame({}, {})).toBeNull();
    });
});
