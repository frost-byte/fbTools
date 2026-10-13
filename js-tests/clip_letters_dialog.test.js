import { openClipLettersDialog, openBulkLettersDialog } from "../js/ui/clip_letters_dialog.js";

const profile = { subjects: [{ id: "girl", label: "the girl" }, { id: "boy", label: "the boy" }, { id: "dog", label: "the dog" }] };
const buttons = () => [...document.querySelectorAll(".spe-letters-dialog button")];
const byText = t => buttons().find(b => b.textContent === t);
afterEach(() => { document.body.innerHTML = ""; });

describe("single clip dialog", () => {
    const clip = { id: "c1", subjects: ["boy", "dog"], action: "{A} barks at {B}" };

    test("shows the new letters in profile order and applies them with the text rewritten", async () => {
        // Ticking "the girl" (listed first in the profile) pushes the boy and the dog down a letter.
        const p = openClipLettersDialog({ profile, clip, newIds: ["dog", "boy", "girl"] });
        const rows = [...document.querySelectorAll(".spe-letters-row")].map(r => r.textContent);
        expect(rows[0]).toContain("{A}");
        expect(rows[0]).toContain("the girl");
        expect(rows[0]).toContain("new");
        expect(rows[1]).toContain("the boy");
        expect(rows[1]).toContain("was {A}");
        expect(document.querySelector(".spe-letters-preview").textContent).toContain("{B} barks at {C}");
        byText("Apply").click();
        expect(await p).toEqual({ ids: ["girl", "boy", "dog"], remap: true });
        expect(document.querySelector(".spe-letters-dialog")).toBeNull();
    });

    test("there is no way to reorder subjects in the dialog", () => {
        openClipLettersDialog({ profile, clip, newIds: ["girl", "boy", "dog"] });
        expect(document.querySelector("button[title='Move up']")).toBeNull();
    });

    test("the rewrite can be turned off", async () => {
        const p = openClipLettersDialog({ profile, clip, newIds: ["girl", "boy", "dog"] });
        document.querySelector(".spe-letters-check input").click();
        expect(document.querySelector(".spe-letters-note.warn").textContent).toMatch(/different subjects/);
        byText("Apply").click();
        expect((await p).remap).toBe(false);
    });

    test("cancel resolves null and Escape cancels too", async () => {
        const p = openClipLettersDialog({ profile, clip, newIds: ["girl", "boy", "dog"] });
        byText("Cancel").click();
        expect(await p).toBeNull();
        const q = openClipLettersDialog({ profile, clip, newIds: ["girl", "boy", "dog"] });
        document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
        expect(await q).toBeNull();
    });

    test("a removed subject is called out and its name replaces the letter", async () => {
        openClipLettersDialog({ profile, clip, newIds: ["dog"] });
        expect(document.querySelector(".spe-letters-dialog").textContent).toContain("Removed from this clip: the boy (was {A})");
        expect(document.querySelector(".spe-letters-preview").textContent).toContain("the boy barks at {A}");
    });
});

describe("bulk dialog", () => {
    const items = [
        { clip: { id: "c1", label: "one", subjects: ["girl", "boy"], action: "{A} hugs {B}" }, newIds: ["boy"] },
        { clip: { id: "c2", label: "two", subjects: ["girl", "dog"], action: "{B} barks" }, newIds: ["dog"] },
    ];

    test("applies to the ticked clips only", async () => {
        const p = openBulkLettersDialog({ profile, items, title: "Remove the girl" });
        expect(document.querySelectorAll(".spe-letters-row").length).toBe(2);
        document.querySelectorAll(".spe-letters-row input")[1].click();   // untick clip two
        byText("Apply").click();
        expect([...(await p)]).toEqual(["c1"]);
    });

    test("cancel resolves null", async () => {
        const p = openBulkLettersDialog({ profile, items, title: "x" });
        byText("Cancel").click();
        expect(await p).toBeNull();
    });
});
