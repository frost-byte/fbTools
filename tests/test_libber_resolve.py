"""Tests for utils/libber_resolve.py"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.libber_resolve import resolve_libber_refs, extract_libber_names


REGISTRY = {
    "greet": {"hello": "Hello there!", "bye": "Goodbye!"},
    "rora":  {"line_a": "I never meant to hurt you.", "line_b": "We need to talk."},
}


# ── resolve_libber_refs ────────────────────────────────────────────────────────

class TestResolveLibberRefs:
    def test_named_key_resolved(self):
        assert resolve_libber_refs("%greet:hello%", REGISTRY) == "Hello there!"

    def test_named_key_other_libber(self):
        assert resolve_libber_refs("%rora:line_b%", REGISTRY) == "We need to talk."

    def test_no_token_unchanged(self):
        assert resolve_libber_refs("plain text", REGISTRY) == "plain text"

    def test_empty_string(self):
        assert resolve_libber_refs("", REGISTRY) == ""

    def test_none_safe(self):
        assert resolve_libber_refs(None, REGISTRY) == None  # type: ignore[arg-type]

    def test_missing_libber_left_as_is(self):
        assert resolve_libber_refs("%unknown:key%", REGISTRY) == "%unknown:key%"

    def test_missing_key_left_as_is(self):
        assert resolve_libber_refs("%greet:morning%", REGISTRY) == "%greet:morning%"

    def test_multiple_tokens_in_one_string(self):
        result = resolve_libber_refs("%greet:hello% said, %greet:bye%", REGISTRY)
        assert result == "Hello there! said, Goodbye!"

    def test_token_mixed_with_plain_text(self):
        result = resolve_libber_refs("She said: %greet:hello%", REGISTRY)
        assert result == "She said: Hello there!"

    def test_random_star_returns_a_value(self):
        result = resolve_libber_refs("%greet:*%", REGISTRY)
        assert result in ("Hello there!", "Goodbye!")

    def test_random_star_empty_libber_left_as_is(self):
        reg = {"empty": {}}
        assert resolve_libber_refs("%empty:*%", reg) == "%empty:*%"

    def test_key_with_whitespace_stripped(self):
        # Key has surrounding spaces in the token
        assert resolve_libber_refs("%greet: hello %", REGISTRY) == "Hello there!"

    def test_existing_percent_delimiters_not_consumed(self):
        # Standard libber %key% pattern (no colon) must not be touched
        result = resolve_libber_refs("%hello%", REGISTRY)
        assert result == "%hello%"

    def test_empty_registry(self):
        assert resolve_libber_refs("%greet:hello%", {}) == "%greet:hello%"


# ── extract_libber_names ───────────────────────────────────────────────────────

class TestExtractLibberNames:
    def test_single_name(self):
        assert extract_libber_names("%greet:hello%") == ["greet"]

    def test_two_different_libbers(self):
        assert extract_libber_names("%greet:hello% and %rora:*%") == ["greet", "rora"]

    def test_same_libber_deduplicated(self):
        assert extract_libber_names("%greet:hello% %greet:bye%") == ["greet"]

    def test_order_preserved(self):
        names = extract_libber_names("%rora:*% then %greet:hello%")
        assert names == ["rora", "greet"]

    def test_no_tokens_empty(self):
        assert extract_libber_names("plain text") == []

    def test_empty_string(self):
        assert extract_libber_names("") == []
