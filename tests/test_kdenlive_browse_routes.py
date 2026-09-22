"""Route tests for the Kdenlive Archive tab's browse endpoints (browse_files, browse_dirs).

Loaded under a synthetic package with folder_paths/server stubbed to real temp directories (same
technique as tests/test_route_modules.py), rather than through conftest's global mocks — those mock
folder_paths as a bare MagicMock, which can't stand in for a real directory to os.walk().
"""
import asyncio
import importlib
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = "fbt_kdenlive_browse_test_pkg"


class _Routes:
    def __init__(self):
        self.handlers = {}

    def _decorator(self, verb):
        def deco(path):
            def wrap(fn):
                self.handlers[(verb, path)] = fn
                return fn
            return wrap
        return deco

    def __getattr__(self, name):
        if name in ("get", "post", "delete", "put"):
            return self._decorator(name)
        raise AttributeError(name)


class _Req:
    def __init__(self, query):
        self.rel_url = types.SimpleNamespace(query=query)


@pytest.fixture()
def routes(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    r = _Routes()
    fp = types.ModuleType("folder_paths")
    fp.get_input_directory = lambda: str(input_dir)
    fp.get_output_directory = lambda: str(output_dir)

    class _PS:
        instance = types.SimpleNamespace(routes=r, send_sync=lambda *a, **k: None)
    srv = types.ModuleType("server")
    srv.PromptServer = _PS

    pkg = types.ModuleType(PKG)
    pkg.__path__ = [str(ROOT)]
    monkeypatch.setitem(sys.modules, "folder_paths", fp)
    monkeypatch.setitem(sys.modules, "server", srv)
    monkeypatch.setitem(sys.modules, PKG, pkg)
    for name in list(sys.modules):
        if name.startswith(PKG + "."):
            monkeypatch.delitem(sys.modules, name)
    importlib.import_module(f"{PKG}.nodes.kdenlive_archive")

    r.input_dir, r.output_dir = input_dir, output_dir
    return r


def _call(routes, verb, path, **query):
    return asyncio.run(routes.handlers[(verb, path)](_Req(query)))


def _body(resp) -> dict:
    return json.loads(resp.text)


# ── browse_files ─────────────────────────────────────────────────────────────────

def test_browse_files_finds_kdenlive_recursively_as_absolute_paths(routes):
    sub = routes.output_dir / "projects" / "video1"
    sub.mkdir(parents=True)
    (sub / "yaoyao.kdenlive").write_text("<mlt/>")
    (routes.output_dir / "unrelated.mp4").write_text("not a project")

    resp = _call(routes, "get", "/fbtools/kdenlive/browse_files", folder="output")
    files = _body(resp)["files"]

    assert files == [str(sub / "yaoyao.kdenlive")]
    assert all(Path(f).is_absolute() for f in files)


def test_browse_files_skips_dot_directories(routes):
    hidden = routes.input_dir / ".cache"
    hidden.mkdir()
    (hidden / "backup.kdenlive").write_text("<mlt/>")

    resp = _call(routes, "get", "/fbtools/kdenlive/browse_files", folder="input")
    assert _body(resp)["files"] == []


def test_browse_files_defaults_to_input(routes):
    (routes.input_dir / "a.kdenlive").write_text("<mlt/>")
    (routes.output_dir / "b.kdenlive").write_text("<mlt/>")

    resp = _call(routes, "get", "/fbtools/kdenlive/browse_files")
    assert _body(resp)["files"] == [str(routes.input_dir / "a.kdenlive")]


def test_browse_files_rejects_unknown_folder(routes):
    resp = _call(routes, "get", "/fbtools/kdenlive/browse_files", folder="elsewhere")
    assert resp.status == 400
    assert "Invalid folder" in _body(resp)["error"]


# ── browse_dirs ──────────────────────────────────────────────────────────────────

def test_browse_dirs_includes_empty_directories(routes):
    empty = routes.output_dir / "new_archive"
    empty.mkdir()
    populated = routes.output_dir / "video" / "clip1"
    populated.mkdir(parents=True)
    (populated / "clip.mp4").write_text("x")

    resp = _call(routes, "get", "/fbtools/kdenlive/browse_dirs", folder="output")
    body = _body(resp)

    assert body["root"] == str(routes.output_dir)
    assert str(empty) in body["dirs"]
    assert str(populated) in body["dirs"]
    assert str(populated.parent) in body["dirs"]


def test_browse_dirs_skips_dot_directories(routes):
    (routes.input_dir / ".git").mkdir()
    (routes.input_dir / "real").mkdir()

    resp = _call(routes, "get", "/fbtools/kdenlive/browse_dirs", folder="input")
    dirs = _body(resp)["dirs"]

    assert str(routes.input_dir / "real") in dirs
    assert not any(".git" in d for d in dirs)


def test_browse_dirs_rejects_unknown_folder(routes):
    resp = _call(routes, "get", "/fbtools/kdenlive/browse_dirs", folder="nope")
    assert resp.status == 400
