"""Tests for the spreadsheet-column-style slot letter generator."""
import pytest
from conftest import import_test_module

sl = import_test_module("utils/slot_letters.py")
slot_letter = sl.slot_letter
is_bare_slot_letter = sl.is_bare_slot_letter


@pytest.mark.parametrize("index,expected", [
    (0, "A"),
    (1, "B"),
    (25, "Z"),
    (26, "AA"),
    (27, "AB"),
    (51, "AZ"),
    (52, "BA"),
    (58, "BG"),
    (701, "ZZ"),
    (702, "AAA"),
])
def test_slot_letter_known_values(index, expected):
    assert slot_letter(index) == expected


def test_slot_letter_negative_index_raises():
    with pytest.raises(ValueError):
        slot_letter(-1)


def test_slot_letter_unique_over_range():
    letters = [slot_letter(i) for i in range(700)]
    assert len(letters) == len(set(letters))


class TestIsBareSlotLetter:
    def test_single_letter(self):
        assert is_bare_slot_letter("A") is True

    def test_double_letter(self):
        assert is_bare_slot_letter("AA") is True

    def test_bundle_suffix_is_not_bare(self):
        assert is_bare_slot_letter("A_bundle") is False

    def test_fit_key_is_not_bare(self):
        assert is_bare_slot_letter("Fit_1") is False

    def test_lowercase_is_not_bare(self):
        assert is_bare_slot_letter("a") is False

    def test_empty_string_is_not_bare(self):
        assert is_bare_slot_letter("") is False
