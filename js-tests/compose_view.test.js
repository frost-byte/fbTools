/**
 * Smoke test: the Compose tab opens on the saved-compositions list and switches to the
 * editor for "+ New" / a card click, and "← Back" returns to the list.
 */
import { jest } from "@jest/globals";

const COMPS = [
    { id: "beach", name: "Beach walk", model_type: "h3_ref2va", subject_count: 2, shot_count: 3, background: "b1", updated_at: "2026-09-20T12:00:00Z" },
    { id: "cafe", name: "Cafe scene", model_type: "wan22", subject_count: 1, shot_count: 1, background: "" },
];

function jsonRes(body) {
    return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(body), text: () => Promise.resolve(JSON.stringify(body)) });
}

beforeAll(() => {
    global.fetch = jest.fn((url) => {
        const u = String(url);
        if (u.includes("/compositions/list")) return jsonRes({ compositions: COMPS });
        if (u.includes("/compositions/get")) return jsonRes({ id: "beach", name: "Beach walk", model_type: "h3_ref2va", subjects: { A: "s1" }, shots: [], background: "b1" });
        if (u.includes("/subjects/list") || u.includes("/subjects")) return jsonRes({ subjects: [{ id: "s1", name: "Mara" }] });
        if (u.includes("/backgrounds/list")) return jsonRes({ backgrounds: [{ id: "b1", name: "Beach" }] });
        return jsonRes({});
    });
});

test("list view -> editor -> back", async () => {
    const { renderCompositionEditor } = await import("../js/ui/composition_editor.js");
    const el = document.createElement("div");
    document.body.appendChild(el);
    await renderCompositionEditor(el);

    const list = el.querySelector(".fbt-ce-list-view");
    const editor = el.querySelector(".fbt-ce-editor-view");
    expect(list.style.display).toBe("");
    expect(editor.style.display).toBe("none");

    const cards = el.querySelectorAll(".fbt-ce-list-card");
    expect(cards.length).toBe(2);
    expect(cards[0].textContent).toContain("Beach walk");
    expect(cards[0].textContent).toContain("2 subj");
    expect(cards[0].textContent).toContain("3 shots");

    // + New opens the editor
    [...el.querySelectorAll("button")].find(b => b.textContent === "+ New").click();
    expect(list.style.display).toBe("none");
    expect(editor.style.display).toBe("");
    expect(el.querySelector(".fbt-ce-editor-title").textContent).toBe("New composition");

    // ← Back returns to the list (state is clean, so no confirm)
    [...el.querySelectorAll("button")].find(b => b.textContent === "← Back").click();
    expect(list.style.display).toBe("");

    // Search filters the cards
    const search = el.querySelector(".fbt-ce-list-search");
    search.value = "cafe";
    search.dispatchEvent(new Event("input"));
    expect(el.querySelectorAll(".fbt-ce-list-card").length).toBe(1);
});
