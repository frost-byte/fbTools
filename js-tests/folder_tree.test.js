/**
 * Tests for js/ui/folder_tree.js — the folder-picking sibling of file_tree.js.
 */
import { jest } from "@jest/globals";
import { buildFolderTree } from "../js/ui/folder_tree.js";

function names(el, selector = ".fbt-be-tree-file-name") {
    return [...el.querySelectorAll(selector)].map(n => n.textContent);
}

describe("buildFolderTree", () => {
    test("shows a root row plus every top-level folder, collapsed", () => {
        const { el } = buildFolderTree({ inputDirs: ["video", "video/clip1", "archives"] });
        expect(names(el)).toEqual(["(this folder)", "archives", "video"]);
        // Nested folder isn't shown until its parent is expanded.
        expect(names(el)).not.toContain("clip1");
    });

    test("an empty tab shows the empty-text message but still offers the root row", () => {
        const { el } = buildFolderTree({ inputDirs: [], emptyText: "No subfolders in {dir}/" });
        expect(el.textContent).toContain("No subfolders in input/");
        expect(names(el)).toEqual(["(this folder)"]);
    });

    test("clicking a folder both selects it and expands its children", () => {
        const onSelect = jest.fn();
        const { el } = buildFolderTree({ inputDirs: ["video", "video/clip1"], onSelect });

        const videoRow = [...el.querySelectorAll(".fbt-ft-dir-row")].find(r => r.textContent.includes("video"));
        videoRow.click();

        expect(onSelect).toHaveBeenCalledWith("video", "input");
        expect(videoRow.classList.contains("fbt-be-tree-file-cur")).toBe(true);
        expect(names(el)).toContain("clip1");
    });

    test("clicking the root row selects the empty-string path", () => {
        const onSelect = jest.fn();
        const { el } = buildFolderTree({ inputDirs: ["video"], onSelect });
        el.querySelector(".fbt-be-tree-file-name").parentElement.click();
        expect(onSelect).toHaveBeenCalledWith("", "input");
    });

    test("clicking a nested folder passes its full path", () => {
        const onSelect = jest.fn();
        const { el } = buildFolderTree({ inputDirs: ["video", "video/clip1"], onSelect });
        [...el.querySelectorAll(".fbt-ft-dir-row")].find(r => r.textContent.includes("video")).click();
        [...el.querySelectorAll(".fbt-ft-dir-row")].find(r => r.textContent.includes("clip1")).click();
        expect(onSelect).toHaveBeenCalledWith("video/clip1", "input");
    });

    test("isSelected highlights the matching row on first render", () => {
        const { el } = buildFolderTree({
            inputDirs: ["video"],
            isSelected: (path, dir) => path === "video" && dir === "input",
        });
        const row = [...el.querySelectorAll(".fbt-ft-dir-row")].find(r => r.textContent.includes("video"));
        expect(row.classList.contains("fbt-be-tree-file-cur")).toBe(true);
    });

    test("switching tabs shows the other side's folders", () => {
        const { el } = buildFolderTree({ inputDirs: ["in_folder"], outputDirs: ["out_folder"] });
        expect(names(el)).toContain("in_folder");
        [...el.querySelectorAll(".fbt-be-tree-tab")].find(b => b.textContent === "Output").click();
        expect(names(el)).toContain("out_folder");
        expect(names(el)).not.toContain("in_folder");
    });

    test("search filters to a flat list of matching paths", () => {
        const { el } = buildFolderTree({ inputDirs: ["video/clip1", "video/clip2", "archives"] });
        el.querySelector(".fbt-be-tree-search").dispatchEvent(Object.assign(new Event("input"), {}));
        const search = el.querySelector(".fbt-be-tree-search");
        search.value = "clip";
        search.dispatchEvent(new Event("input"));
        expect(names(el)).toEqual(["(this folder)", "clip1", "clip2"]);
    });

    test("rebuild() replaces the folder lists", () => {
        const { el, rebuild } = buildFolderTree({ inputDirs: ["old"] });
        expect(names(el)).toContain("old");
        rebuild({ inputDirs: ["new"] });
        expect(names(el)).toContain("new");
        expect(names(el)).not.toContain("old");
    });

    test("getDir reports the active tab", () => {
        const { getDir, el } = buildFolderTree({});
        expect(getDir()).toBe("input");
        [...el.querySelectorAll(".fbt-be-tree-tab")].find(b => b.textContent === "Output").click();
        expect(getDir()).toBe("output");
    });
});
