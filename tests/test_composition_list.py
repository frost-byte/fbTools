"""list_compositions summaries carry the counts the Compose list view shows."""
import json

from conftest import import_test_module

pc = import_test_module("utils/prompt_compositions.py")


def _write(data_dir, comp):
    d = data_dir / "prompt_compositions"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{comp['id']}.json").write_text(json.dumps(comp), encoding="utf-8")


def test_list_includes_counts_and_background(tmp_path):
    _write(tmp_path, {"id": "b", "name": "Beta", "model_type": "wan22",
                      "subjects": {"A": "x", "B": "", "C": "y"},
                      "shots": [{"id": "s1"}, {"id": "s2"}], "background": "beach"})
    _write(tmp_path, {"id": "a", "name": "alpha", "model_type": "h3_ref2va"})
    items = pc.list_compositions(str(tmp_path))
    assert [i["name"] for i in items] == ["alpha", "Beta"]
    beta = items[1]
    assert beta["subject_count"] == 2   # blank slots are not counted
    assert beta["shot_count"] == 2
    assert beta["background"] == "beach"
    alpha = items[0]
    assert (alpha["subject_count"], alpha["shot_count"], alpha["background"]) == (0, 0, "")
