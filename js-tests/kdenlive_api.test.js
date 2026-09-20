/**
 * Tests for Kdenlive Archive API client
 */

import { KdenliveAPI } from "../js/api/kdenlive.js";
import { mockFetch } from "./test_utils.js";

describe("KdenliveAPI", () => {
    beforeEach(() => { mockFetch.setup(); });
    afterEach(() => { mockFetch.restore(); });

    test("check posts project, maps and search dirs to /check", async () => {
        mockFetch.mockResponse({ ok: true, report: { unresolved_count: 0 } });
        const result = await new KdenliveAPI().check({
            project: "/p/a.kdenlive", path_maps: ["Z:/=/mnt/x/"], search_dirs: ["/clips"],
        });
        expect(result.report.unresolved_count).toBe(0);
        const call = mockFetch.getCalls()[0];
        expect(call.url).toContain("/fbtools/kdenlive/check");
        expect(JSON.parse(call.body)).toEqual({
            project: "/p/a.kdenlive", path_maps: ["Z:/=/mnt/x/"], search_dirs: ["/clips"],
        });
    });

    test("archive returns the job id", async () => {
        mockFetch.mockResponse({ started: true, job_id: "abc123" });
        const result = await new KdenliveAPI().archive({ project: "/p/a.kdenlive", dest: "/out", dry_run: true });
        expect(result.job_id).toBe("abc123");
        const call = mockFetch.getCalls()[0];
        expect(call.url).toContain("/fbtools/kdenlive/archive");
        expect(JSON.parse(call.body).dry_run).toBe(true);
    });

    test("status passes job_id as a query parameter", async () => {
        mockFetch.mockResponse({ job: { id: "abc123", state: "running" } });
        const result = await new KdenliveAPI().status("abc123");
        expect(result.job.state).toBe("running");
        expect(String(mockFetch.getCalls()[0].url)).toContain("job_id=abc123");
    });

    test("status without an id omits the query", async () => {
        mockFetch.mockResponse({ job: null });
        await new KdenliveAPI().status();
        expect(String(mockFetch.getCalls()[0].url)).not.toContain("job_id");
    });

    test("cancel and strip post to their endpoints", async () => {
        mockFetch.mockResponse({ ok: true });
        mockFetch.mockResponse({ ok: true, report: { removed: 2 } });
        const api = new KdenliveAPI();
        await api.cancel("abc123");
        const res = await api.strip({ project: "/p/a.kdenlive", in_place: true });
        expect(res.report.removed).toBe(2);
        const calls = mockFetch.getCalls();
        expect(calls[0].url).toContain("/fbtools/kdenlive/cancel");
        expect(JSON.parse(calls[0].body)).toEqual({ job_id: "abc123" });
        expect(calls[1].url).toContain("/fbtools/kdenlive/strip");
    });
});
