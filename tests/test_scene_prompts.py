"""Tests for utils/scene_prompts.py (prompt resolution shared by Story nodes and the importer)."""
from conftest import import_test_module

sp = import_test_module("utils/scene_prompts.py")
prompt_models = import_test_module("prompt_models.py")
PromptCollection = prompt_models.PromptCollection


class _Libber:
    def __init__(self, libs):
        self.libs = libs

    def substitute(self, text):
        for k, v in self.libs.items():
            text = text.replace(f"%{k}%", v)
        return text


class _Manager:
    """Mimics LibberStateManager: ensure_libber loads on demand, get_libber only what is loaded."""

    def __init__(self, available, loaded=()):
        self.available = available
        self.loaded = {n: _Libber(available[n]) for n in loaded}

    def ensure_libber(self, name):
        if name not in self.loaded and name in self.available:
            self.loaded[name] = _Libber(self.available[name])
        return self.loaded.get(name)

    def get_libber(self, name):
        return self.loaded.get(name)


def _v2(prompts, compositions=None):
    return {
        "version": 2,
        "prompts": {k: {"value": v[0], "processing_type": v[1], "category": "x", "libber_name": v[2]}
                    for k, v in prompts.items()},
        "compositions": compositions or {},
    }


RAW = _v2(
    {
        "img": ("%img_text%", "libber", "story"),
        "vid": ("she walks to the window", "text", ""),
        "who": ("a woman", "text", ""),
    },
    {"img_comp": ["img", "who"], "vid_comp": ["vid", "who"]},
)


def _resolve(raw=RAW, mgr=None):
    mgr = mgr or _Manager({"story": {"img_text": "replace the person"}})
    return sp.resolve_scene_prompts(raw, PromptCollection.from_dict, mgr)


class TestResolveScenePrompts:
    def test_libber_prompts_are_substituted(self):
        prompts, comps = _resolve()
        assert prompts["img"] == "replace the person"
        assert comps["img_comp"] == "replace the person a woman"

    def test_libber_is_loaded_from_disk_when_not_already_loaded(self):
        mgr = _Manager({"story": {"img_text": "from disk"}})
        assert mgr.get_libber("story") is None
        prompts, _ = _resolve(mgr=mgr)
        assert prompts["img"] == "from disk"

    def test_missing_libber_leaves_the_token(self):
        prompts, _ = _resolve(mgr=_Manager({}))
        assert prompts["img"] == "%img_text%"

    def test_no_manager_needed_for_plain_text(self):
        raw = _v2({"a": ("plain", "text", "")})
        prompts, _ = sp.resolve_scene_prompts(raw, PromptCollection.from_dict, None)
        assert prompts == {"a": "plain"}

    def test_legacy_flat_dict_passes_through(self):
        prompts, comps = sp.resolve_scene_prompts({"a": "x"}, PromptCollection.from_dict, None)
        assert prompts == {"a": "x"} and comps == {}

    def test_empty_input(self):
        assert sp.resolve_scene_prompts(None, PromptCollection.from_dict) == ({}, {})
        assert sp.resolve_scene_prompts({}, PromptCollection.from_dict) == ({}, {})


class TestResolveVideoPrompt:
    def setup_method(self):
        self.prompts, self.comps = _resolve()

    def _scene(self, **kw):
        base = {"prompt_source": "composition", "prompt_key": "img_comp", "custom_prompt": "",
                "video_prompt_source": "auto", "video_prompt_key": "", "video_custom_prompt": ""}
        base.update(kw)
        return base

    def test_auto_falls_back_to_the_image_prompt(self):
        text, source = sp.resolve_video_prompt(self._scene(), self.prompts, self.comps)
        assert (text, source) == ("replace the person a woman", sp.SOURCE_IMAGE_PROMPT)

    def test_auto_with_a_custom_image_prompt(self):
        scene = self._scene(prompt_source="custom", custom_prompt="custom image text")
        assert sp.resolve_video_prompt(scene, self.prompts, self.comps) == ("custom image text", sp.SOURCE_IMAGE_PROMPT)

    def test_explicit_composition(self):
        scene = self._scene(video_prompt_source="composition", video_prompt_key="vid_comp")
        assert sp.resolve_video_prompt(scene, self.prompts, self.comps) == ("she walks to the window a woman", "composition")

    def test_explicit_prompt(self):
        scene = self._scene(video_prompt_source="prompt", video_prompt_key="vid")
        assert sp.resolve_video_prompt(scene, self.prompts, self.comps) == ("she walks to the window", "prompt")

    def test_explicit_custom(self):
        scene = self._scene(video_prompt_source="custom", video_custom_prompt="fast action")
        assert sp.resolve_video_prompt(scene, self.prompts, self.comps) == ("fast action", "custom")

    def test_unresolved_key_is_empty(self):
        scene = self._scene(video_prompt_source="composition", video_prompt_key="nope")
        assert sp.resolve_video_prompt(scene, self.prompts, self.comps) == ("", sp.SOURCE_EMPTY)

    def test_auto_with_nothing_is_empty(self):
        scene = self._scene(prompt_key="")
        assert sp.resolve_video_prompt(scene, self.prompts, self.comps) == ("", sp.SOURCE_EMPTY)

    def test_accepts_attribute_objects(self):
        class Scene:
            prompt_source, prompt_key, custom_prompt = "prompt", "who", ""
            video_prompt_source, video_prompt_key, video_custom_prompt = "auto", "", ""
        assert sp.resolve_video_prompt(Scene(), self.prompts, self.comps) == ("a woman", sp.SOURCE_IMAGE_PROMPT)


def test_unresolved_tokens():
    assert sp.unresolved_tokens("a %girl% and %girl% with %eyes_x%") == ["%eyes_x%", "%girl%"]
    assert sp.unresolved_tokens("100% sure, 50% off") == []
    assert sp.unresolved_tokens("") == []
