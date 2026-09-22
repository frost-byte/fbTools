import { jest } from "@jest/globals";
let setupLlmQueueWarning;

function fakeApi() {
    const handlers = {};
    return {
        addEventListener: (name, fn) => { handlers[name] = fn; },
        fire: async (name) => { await handlers[name]?.(); },
    };
}

function jsonRes(body, ok = true) {
    return Promise.resolve({ ok, json: () => Promise.resolve(body) });
}

describe("setupLlmQueueWarning", () => {
    let app, api, toastAdd;

    beforeEach(async () => {
        jest.useFakeTimers({ doNotFake: ["nextTick"] }).setSystemTime(0);
        jest.resetModules();
        // _lastWarnTs is module-level (one real instance for the whole app session);
        // re-import fresh each test so the cooldown doesn't leak between test cases.
        ({ setupLlmQueueWarning } = await import("../js/utils/llm_queue_warning.js"));
        toastAdd = jest.fn();
        app = { extensionManager: { toast: { add: toastAdd } } };
        api = fakeApi();
        global.fetch = jest.fn();
    });

    test("warns when the local LLM is loaded", async () => {
        global.fetch.mockReturnValue(jsonRes({ loaded_model: "qwen2.5-vl-7b" }));
        setupLlmQueueWarning(app, api);
        await api.fire("promptQueued");
        expect(global.fetch).toHaveBeenCalledWith("/fbtools/llm/status");
        expect(toastAdd).toHaveBeenCalledTimes(1);
        const call = toastAdd.mock.calls[0][0];
        expect(call.severity).toBe("warn");
        expect(call.detail).toContain("qwen2.5-vl-7b");
    });

    test("stays quiet when no model is loaded", async () => {
        global.fetch.mockReturnValue(jsonRes({ loaded_model: null }));
        setupLlmQueueWarning(app, api);
        await api.fire("promptQueued");
        expect(toastAdd).not.toHaveBeenCalled();
    });

    test("stays quiet on a fetch error, never throws", async () => {
        global.fetch.mockReturnValue(jsonRes({}, false));
        setupLlmQueueWarning(app, api);
        await expect(api.fire("promptQueued")).resolves.toBeUndefined();
        expect(toastAdd).not.toHaveBeenCalled();

        global.fetch.mockRejectedValue(new Error("network down"));
        await expect(api.fire("promptQueued")).resolves.toBeUndefined();
        expect(toastAdd).not.toHaveBeenCalled();
    });

    test("is silent again within the cooldown window, then warns once it elapses", async () => {
        global.fetch.mockReturnValue(jsonRes({ loaded_model: "m" }));
        setupLlmQueueWarning(app, api);
        await api.fire("promptQueued");
        expect(toastAdd).toHaveBeenCalledTimes(1);

        jest.setSystemTime(5000);
        await api.fire("promptQueued");
        expect(toastAdd).toHaveBeenCalledTimes(1); // still within cooldown

        jest.setSystemTime(20000);
        await api.fire("promptQueued");
        expect(toastAdd).toHaveBeenCalledTimes(2);
    });
});
