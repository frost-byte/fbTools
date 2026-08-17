"""Tests for the Outfit Registry system."""

import json
import os
import pytest
from conftest import import_test_module

or_mod = import_test_module("utils/outfit_registry.py")

OutfitRegistry         = or_mod.OutfitRegistry
load_outfit_registry   = or_mod.load_outfit_registry
save_outfit_registry   = or_mod.save_outfit_registry
_normalize_outfit_images = or_mod._normalize_outfit_images


# ── normalize_outfit_images ────────────────────────────────────────────────────

def test_normalize_plain_string():
    result = _normalize_outfit_images(["outfit.jpg"])
    assert result == [{"file": "outfit.jpg", "role": "costume detail"}]


def test_normalize_dict_passthrough():
    result = _normalize_outfit_images([{"file": "a.jpg", "role": "full body"}])
    assert result == [{"file": "a.jpg", "role": "full body"}]


def test_normalize_mixed():
    result = _normalize_outfit_images(["a.jpg", {"file": "b.jpg", "role": "portrait"}])
    assert result[0] == {"file": "a.jpg", "role": "costume detail"}
    assert result[1] == {"file": "b.jpg", "role": "portrait"}


def test_normalize_empty_list():
    assert _normalize_outfit_images([]) == []


def test_normalize_skips_falsy():
    assert _normalize_outfit_images(["", None, "a.jpg"]) == [{"file": "a.jpg", "role": "costume detail"}]


# ── OutfitRegistry.define ──────────────────────────────────────────────────────

def test_define_basic():
    reg = OutfitRegistry()
    r = reg.define("summer_dress", name="Summer Dress", description="A light floral dress.")
    assert r.outfits["summer_dress"]["name"] == "Summer Dress"
    assert r.outfits["summer_dress"]["description"] == "A light floral dress."


def test_define_with_reference_images():
    reg = OutfitRegistry()
    r = reg.define("summer_dress", name="Summer Dress", description="",
                   reference_images=["dress.jpg"])
    refs = r.outfits["summer_dress"]["reference_images"]
    assert refs == [{"file": "dress.jpg", "role": "costume detail"}]


def test_define_preserves_existing_reference_images():
    reg = OutfitRegistry(outfits={
        "dress": {"name": "Dress", "description": "", "reference_images": [
            {"file": "existing.jpg", "role": "costume detail"}
        ]}
    })
    # Not passing reference_images → should preserve
    r = reg.define("dress", name="Dress Updated", description="new desc")
    refs = r.outfits["dress"].get("reference_images", [])
    assert refs == [{"file": "existing.jpg", "role": "costume detail"}]


def test_define_replaces_reference_images_when_provided():
    reg = OutfitRegistry(outfits={
        "dress": {"name": "Dress", "description": "", "reference_images": [
            {"file": "old.jpg", "role": "costume detail"}
        ]}
    })
    r = reg.define("dress", name="Dress", description="", reference_images=["new.jpg"])
    refs = r.outfits["dress"]["reference_images"]
    assert refs == [{"file": "new.jpg", "role": "costume detail"}]


def test_define_empty_reference_images_clears_list():
    reg = OutfitRegistry(outfits={
        "dress": {"name": "Dress", "description": "", "reference_images": [
            {"file": "old.jpg", "role": "costume detail"}
        ]}
    })
    r = reg.define("dress", name="Dress", description="", reference_images=[])
    assert r.outfits["dress"]["reference_images"] == []


# ── from_dict migration ────────────────────────────────────────────────────────

def test_from_dict_normalizes_plain_string_refs():
    data = {"version": 1, "outfits": {
        "jacket": {"name": "Jacket", "description": "",
                   "reference_images": ["jacket.jpg"]}
    }}
    reg = OutfitRegistry.from_dict(data)
    assert reg.outfits["jacket"]["reference_images"] == [
        {"file": "jacket.jpg", "role": "costume detail"}
    ]


def test_from_dict_no_reference_images_ok():
    data = {"version": 1, "outfits": {
        "jacket": {"name": "Jacket", "description": ""}
    }}
    reg = OutfitRegistry.from_dict(data)
    assert "reference_images" not in reg.outfits["jacket"]


# ── round-trip persistence ─────────────────────────────────────────────────────

def test_round_trip(tmp_path):
    path = str(tmp_path / "outfits.json")
    reg = OutfitRegistry()
    reg = reg.define("coat", name="Winter Coat", description="Heavy wool coat.",
                     reference_images=[{"file": "coat.jpg", "role": "full body"}])
    save_outfit_registry(reg, path, backup=False)
    loaded = load_outfit_registry(path)
    refs = loaded.outfits["coat"]["reference_images"]
    assert refs == [{"file": "coat.jpg", "role": "full body"}]
