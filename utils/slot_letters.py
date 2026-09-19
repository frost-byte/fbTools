"""Spreadsheet-column-style slot letter generation: A, B, ..., Z, AA, AB, ...

Shared by Composition subject slots and (in a later phase) Source Profile
clip slots, so neither notation is capped at 26/9 entries and both agree on
the same generated shape.
"""
from __future__ import annotations


def slot_letter(index: int) -> str:
    """0-based index -> 'A', 'B', ..., 'Z', 'AA', 'AB', ..., 'AZ', 'BA', ...

    Bijective base-26 (spreadsheet column naming) — no artificial cap.
    """
    if index < 0:
        raise ValueError(f"slot_letter index must be >= 0, got {index}")
    n = index + 1
    letters = ""
    while n > 0:
        n, rem = divmod(n - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def is_bare_slot_letter(key: str) -> bool:
    """True if key is a pure generated slot letter (A, B, ..., AA, ...) —
    i.e. all uppercase ASCII letters, no underscore/digit. Used to distinguish
    base subject-slot keys from synthetic keys like 'A_bundle', 'Fit_1',
    or reserved non-letter keys."""
    return bool(key) and key.isalpha() and key.isupper()
