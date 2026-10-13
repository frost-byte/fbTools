import { clipSubjectsInOrder, clipSlotLetters } from "../js/utils/clip_subjects.js";

const profile = { subjects: [{ id: "girl" }, { id: "boy" }, { id: "dog" }] };

test("letters follow the profile's subject order, not the order subjects were ticked in the clip", () => {
    const clip = { subjects: ["dog", "girl"] };
    expect(clipSubjectsInOrder(profile, clip).map(s => s.id)).toEqual(["girl", "dog"]);
    expect(clipSlotLetters(profile, clip)).toEqual({ girl: "A", dog: "B" });
});

test("only the ticked subjects get letters, and they are consecutive", () => {
    expect(clipSlotLetters(profile, { subjects: ["dog", "boy"] })).toEqual({ boy: "A", dog: "B" });
});

test("ids the profile no longer has are skipped and repeats ignored", () => {
    expect(clipSubjectsInOrder(profile, { subjects: ["gone", "boy", "boy", "girl"] }).map(s => s.id)).toEqual(["girl", "boy"]);
});

test("empty or missing input", () => {
    expect(clipSubjectsInOrder(profile, {})).toEqual([]);
    expect(clipSubjectsInOrder(null, { subjects: ["a"] })).toEqual([]);
    expect(clipSlotLetters(profile, null)).toEqual({});
});

import { usedLetters, letterMoves, movesAffectText, remapLetters, setClipSubjects } from "../js/utils/clip_subjects.js";

describe("changing a clip's subjects", () => {
    const prof = { subjects: [{ id: "girl", label: "the girl" }, { id: "boy", label: "the boy" }, { id: "dog", label: "the dog" }] };

    test("usedLetters finds placeholders in every text field", () => {
        const clip = { action: "{A} runs", style: "of {C}", overall_soundscape: "", non_diegetic_music: "for {B}" };
        expect([...usedLetters(clip)].sort()).toEqual(["A", "B", "C"]);
    });

    test("letterMoves for removing the first subject shifts the rest up", () => {
        expect(letterMoves(["girl", "boy", "dog"], ["boy", "dog"])).toEqual([
            { id: "girl", from: "A", to: "" },
            { id: "boy", from: "B", to: "A" },
            { id: "dog", from: "C", to: "B" },
        ]);
    });

    test("a swap is rewritten in one pass and does not undo itself", () => {
        const moves = letterMoves(["girl", "boy"], ["boy", "girl"]);
        expect(remapLetters("{A} hugs {B}, then {A} waves", moves)).toBe("{B} hugs {A}, then {B} waves");
    });

    test("a removed subject's placeholder becomes its label, the others shift", () => {
        const moves = letterMoves(["girl", "boy", "dog"], ["boy", "dog"]);
        expect(remapLetters("{A} chases {B} and {C}", moves, { girl: "the girl" })).toBe("the girl chases {A} and {B}");
        expect(remapLetters("{A} waves", moves)).toBe("{A} waves");   // no label known: left alone
    });

    test("appending a subject changes no existing letter", () => {
        const moves = letterMoves(["girl", "boy"], ["girl", "boy", "dog"]);
        expect(moves.every(m => m.from === m.to)).toBe(true);
        expect(movesAffectText({ action: "{A} and {B}" }, moves)).toBe(false);
    });

    test("movesAffectText only cares about letters the text uses", () => {
        const moves = letterMoves(["girl", "boy", "dog"], ["girl", "dog"]);   // boy leaves, dog B<-C
        expect(movesAffectText({ action: "{A} waves" }, moves)).toBe(false);
        expect(movesAffectText({ action: "{A} waves at {C}" }, moves)).toBe(true);
        expect(movesAffectText({ style: "" }, moves)).toBe(false);
    });

    test("ticking a subject listed before the others shifts them; setClipSubjects rewrites all text fields", () => {
        const clip = { subjects: ["boy", "dog"], action: "{A} barks at {B}", style: "{B} alone", other: 1 };
        const out = setClipSubjects(prof, clip, ["dog", "boy", "girl"]);   // click order is irrelevant
        expect(out.subjects).toEqual(["girl", "boy", "dog"]);
        expect(out.action).toBe("{B} barks at {C}");
        expect(out.style).toBe("{C} alone");
        expect(out.other).toBe(1);
        expect(clip.action).toBe("{A} barks at {B}");   // input untouched
    });

    test("setClipSubjects with remap off changes only the subjects", () => {
        const clip = { subjects: ["boy", "dog"], action: "{A} barks at {B}" };
        expect(setClipSubjects(prof, clip, ["girl", "boy", "dog"], { remap: false }).action).toBe("{A} barks at {B}");
    });

    test("dropping a subject replaces its letter with its label", () => {
        const clip = { subjects: ["boy", "girl"], action: "{A} greets {B}" };   // girl=A, boy=B
        const out = setClipSubjects(prof, clip, ["boy"]);
        expect(out.action).toBe("the girl greets {A}");
        expect(out.subjects).toEqual(["boy"]);
    });

    test("merging: the second clip's letters are rewritten into the merged set", () => {
        const b = { subjects: ["dog", "boy"], action: "{A} barks at {B}" };   // boy=A, dog=B in b
        const mergedIds = ["girl", "boy", "dog"];                              // boy=B, dog=C when merged
        expect(setClipSubjects(prof, b, mergedIds).action).toBe("{B} barks at {C}");
    });
});
