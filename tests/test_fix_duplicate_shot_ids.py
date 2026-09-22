"""Tests for scripts/fix_duplicate_shot_ids.py (renumbers a composition's shots to shot_1..shot_N
when duplicate ids are found — the data-side counterpart of the composition_editor.js fix)."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "fix_duplicate_shot_ids", ROOT / "scripts" / "fix_duplicate_shot_ids.py"
)
fix_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fix_mod)


def _write(comps_dir: Path, comp_id: str, shot_ids: list[str]) -> Path:
    comps_dir.mkdir(parents=True, exist_ok=True)
    path = comps_dir / f"{comp_id}.json"
    path.write_text(json.dumps({
        "id": comp_id, "name": comp_id,
        "shots": [{"id": sid, "action": f"action for {sid}"} for sid in shot_ids],
    }), encoding="utf-8")
    return path


def test_has_duplicate_shot_ids():
    assert fix_mod.has_duplicate_shot_ids([{"id": "shot_1"}, {"id": "shot_1"}]) is True
    assert fix_mod.has_duplicate_shot_ids([{"id": "shot_1"}, {"id": "shot_2"}]) is False
    assert fix_mod.has_duplicate_shot_ids([]) is False


def test_renumber_shots_preserves_order_and_other_fields():
    comp = {"shots": [{"id": "shot_1", "action": "first"}, {"id": "shot_1", "action": "second"}]}
    fix_mod.renumber_shots(comp)
    assert [s["id"] for s in comp["shots"]] == ["shot_1", "shot_2"]
    assert [s["action"] for s in comp["shots"]] == ["first", "second"]


def test_cli_fixes_only_affected_files_and_backs_up(tmp_path):
    comps = tmp_path / "prompt_compositions"
    bad_path = _write(comps, "bad", ["shot_1", "shot_1", "shot_2"])
    good_path = _write(comps, "good", ["shot_1", "shot_2"])

    import sys
    old_argv = sys.argv
    try:
        sys.argv = ["fix_duplicate_shot_ids.py", str(tmp_path)]
        assert fix_mod.main() == 0
    finally:
        sys.argv = old_argv

    fixed = json.loads(bad_path.read_text())
    assert [s["id"] for s in fixed["shots"]] == ["shot_1", "shot_2", "shot_3"]
    assert (comps / "bad.json.pre-shot-id-fix").is_file()
    original = json.loads((comps / "bad.json.pre-shot-id-fix").read_text())
    assert [s["id"] for s in original["shots"]] == ["shot_1", "shot_1", "shot_2"]

    untouched = json.loads(good_path.read_text())
    assert [s["id"] for s in untouched["shots"]] == ["shot_1", "shot_2"]
    assert not (comps / "good.json.pre-shot-id-fix").exists()


def test_dry_run_writes_nothing(tmp_path):
    comps = tmp_path / "prompt_compositions"
    bad_path = _write(comps, "bad", ["shot_1", "shot_1"])
    before = bad_path.read_text()

    import sys
    old_argv = sys.argv
    try:
        sys.argv = ["fix_duplicate_shot_ids.py", str(tmp_path), "--dry-run"]
        assert fix_mod.main() == 0
    finally:
        sys.argv = old_argv

    assert bad_path.read_text() == before
    assert not (comps / "bad.json.pre-shot-id-fix").exists()


def test_rerun_is_idempotent(tmp_path):
    comps = tmp_path / "prompt_compositions"
    _write(comps, "bad", ["shot_1", "shot_1"])
    import sys
    old_argv = sys.argv
    try:
        sys.argv = ["fix_duplicate_shot_ids.py", str(tmp_path)]
        assert fix_mod.main() == 0
        assert fix_mod.main() == 0  # second run: nothing left to fix
    finally:
        sys.argv = old_argv
