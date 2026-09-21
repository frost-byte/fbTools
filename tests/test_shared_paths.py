"""nodes/shared.py: package-root path derivation, default registry paths and the reload-counter registry.

shared.py imports ComfyUI's folder_paths and server at module level, so tests load it under a synthetic
package with those two modules stubbed (no real ComfyUI needed).
"""
import importlib
import os
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = "fbt_shared_test_pkg"


@pytest.fixture()
def shared(tmp_path, monkeypatch):
    fp = types.ModuleType("folder_paths")
    fp.get_user_directory = lambda: str(tmp_path / "user")
    fp.get_output_directory = lambda: str(tmp_path / "output")
    fp.base_path = str(tmp_path)
    fp.models_dir = str(tmp_path / "models")

    class _Server:
        routes = object()
        instance = None
    _Server.instance = _Server()
    srv = types.ModuleType("server")
    srv.PromptServer = _Server

    pkg = types.ModuleType(PKG)
    pkg.__path__ = [str(ROOT)]
    monkeypatch.setitem(sys.modules, "folder_paths", fp)
    monkeypatch.setitem(sys.modules, "server", srv)
    monkeypatch.setitem(sys.modules, PKG, pkg)
    for name in list(sys.modules):
        if name.startswith(PKG + "."):
            monkeypatch.delitem(sys.modules, name)
    return importlib.import_module(f"{PKG}.nodes.shared")


def test_package_root_is_the_package_not_the_nodes_folder(shared):
    assert Path(shared.PACKAGE_ROOT) == ROOT
    assert (Path(shared.PACKAGE_ROOT) / "extension.py").is_file()


def test_user_data_dir_is_named_after_the_package(shared, tmp_path):
    d = shared.user_data_dir()
    assert d == str(tmp_path / "user" / ROOT.name)
    assert os.path.isdir(d)


def test_default_registry_paths_live_in_user_data_dir(shared):
    base = shared.user_data_dir()
    assert shared.default_subject_profiles_path() == os.path.join(base, "subject_profiles.json")
    assert shared.default_bundle_registry_path() == os.path.join(base, "reference_bundles.json")
    assert shared.default_cast_registry_path() == os.path.join(base, "scene_casts.json")
    assert shared.default_outfit_registry_path() == os.path.join(base, "outfit_registry.json")


def test_scene_templates_dir_is_seeded_from_the_package_folder(shared):
    d = shared.default_scene_templates_dir()
    bundled = [f for f in os.listdir(ROOT / "scene_templates") if f.endswith(".json")]
    assert bundled and all((Path(d) / f).is_file() for f in bundled)


def test_routes_is_the_prompt_server_route_table(shared):
    assert shared.routes is sys.modules["server"].PromptServer.instance.routes


def test_reload_counters_start_at_zero_and_bump_independently(shared):
    assert shared.reload_counter("concept") == 0
    assert shared.bump_reload("concept") == 1
    assert shared.bump_reload("concept") == 2
    assert shared.reload_counter("concept") == 2
    assert shared.reload_counter("outfit") == 0
