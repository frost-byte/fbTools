"""Pure logic for the Concept Registry system.

No ComfyUI dependencies — all I/O helpers that need folder_paths or
torch live in extension.py so this module is fully testable in CI.
"""
from __future__ import annotations

import copy
import json
import os
import shutil

# ── Model profiles ─────────────────────────────────────────────────────────────

#: Maps model-type ID → display name and whether it uses a split high/low model.
#: Adding a new model type only requires an entry here.
MODEL_PROFILES: dict[str, dict] = {
    "wan22":   {"split": True,  "display": "Wan 2.2"},
    "bernini": {"split": True,  "display": "BerniniR"},
    "ltx23":   {"split": False, "display": "LTX 2.3"},
    "flux2":   {"split": False, "display": "Flux 2"},
    "krea2":   {"split": False, "display": "Krea 2"},
    "qwen":    {"split": False, "display": "Qwen Image"},
}

MODEL_TYPE_IDS: list[str] = list(MODEL_PROFILES.keys())

_EMPTY_DATA: dict = {"version": 1, "concepts": {}}


# ── ConceptRegistry ────────────────────────────────────────────────────────────

class ConceptRegistry:
    """In-memory concept registry.  Wire between Load → Define → Resolve nodes."""

    def __init__(
        self,
        concepts: dict | None = None,
        file_path: str | None = None,
        version: int = 1,
    ) -> None:
        self.concepts: dict = concepts or {}
        self.file_path: str | None = file_path
        self.version: int = version

    # ── serialisation ──────────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, data: dict, file_path: str | None = None) -> "ConceptRegistry":
        return cls(
            concepts=copy.deepcopy(data.get("concepts", {})),
            file_path=file_path,
            version=data.get("version", 1),
        )

    def to_dict(self) -> dict:
        return {"version": self.version, "concepts": copy.deepcopy(self.concepts)}

    # ── mutation (returns new instance — safe for multi-branch wiring) ─────────

    def define(
        self,
        concept_id: str,
        name: str,
        description: str,
        model_type: str,
        model_entry: dict,
    ) -> "ConceptRegistry":
        """Return a NEW registry with the concept model entry added or updated.

        Different model types for the same concept_id accumulate.
        Same model_type for the same concept_id overwrites.
        """
        new_reg = copy.deepcopy(self)
        if concept_id not in new_reg.concepts:
            new_reg.concepts[concept_id] = {
                "name": name or concept_id,
                "description": description,
                "models": {},
            }
        else:
            if name:
                new_reg.concepts[concept_id]["name"] = name
            if description:
                new_reg.concepts[concept_id]["description"] = description
        new_reg.concepts[concept_id].setdefault("models", {})[model_type] = copy.deepcopy(model_entry)
        return new_reg

    # ── queries ────────────────────────────────────────────────────────────────

    def get_concept(self, concept_id: str) -> dict | None:
        return self.concepts.get(concept_id)

    def get_model_entry(self, concept_id: str, model_type: str) -> dict | None:
        concept = self.concepts.get(concept_id)
        if not concept:
            return None
        return concept.get("models", {}).get(model_type)

    def list_concepts(self, model_type: str | None = None) -> str:
        """Return a formatted human-readable concept list, optionally filtered."""
        if not self.concepts:
            return "No concepts defined."
        lines = []
        for concept_id in sorted(self.concepts):
            concept = self.concepts[concept_id]
            models = concept.get("models", {})
            if model_type and model_type not in models:
                continue
            model_tags = ", ".join(sorted(models.keys()))
            lines.append(f"[{concept_id}] {concept.get('name', concept_id)} ({model_tags})")
        if not lines:
            return f"No concepts defined for model type '{model_type}'."
        return "\n".join(lines)

    def save(self, path: str | None = None, backup: bool = True) -> str:
        """Save to file, return the path written."""
        target = path or self.file_path
        if not target:
            raise ValueError("No file path specified for registry save")
        save_registry(self, target, backup=backup)
        return target


# ── Persistence ────────────────────────────────────────────────────────────────

def load_registry(path: str) -> ConceptRegistry:
    """Load registry from JSON.  Returns empty registry if file absent."""
    if not os.path.exists(path):
        return ConceptRegistry(file_path=path)
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return ConceptRegistry.from_dict(data, file_path=path)


def save_registry(registry: ConceptRegistry, path: str, backup: bool = True) -> None:
    """Write registry to JSON, optionally creating a .bak first."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if backup and os.path.exists(path):
        shutil.copy2(path, path + ".bak")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(registry.to_dict(), fh, indent=2, ensure_ascii=False)
    registry.file_path = path


# ── Resolve logic ──────────────────────────────────────────────────────────────

class ResolvedConcept:
    """Outcome of resolving one concept ID for a specific model type."""

    def __init__(
        self,
        concept_id: str,
        name: str,
        model_type: str,
        lora_high: str | None,
        lora_low: str | None,
        weight_high: float,
        weight_low: float,
        trigger: str,
        error: str | None = None,
    ) -> None:
        self.concept_id = concept_id
        self.name = name
        self.model_type = model_type
        self.lora_high = lora_high
        self.lora_low = lora_low
        self.weight_high = weight_high
        self.weight_low = weight_low
        self.trigger = trigger
        self.error = error  # non-None → something was skipped/missing

    @property
    def is_split(self) -> bool:
        return MODEL_PROFILES.get(self.model_type, {}).get("split", False)

    def format_info(self) -> str:
        if self.error:
            return f"  [{self.concept_id}] {self.error}"
        lines = [f"  [{self.concept_id}] {self.name}"]
        if self.is_split:
            if self.lora_high:
                lines.append(f"    high: {self.lora_high} @ {self.weight_high}")
            if self.lora_low:
                lines.append(f"    low:  {self.lora_low} @ {self.weight_low}")
        else:
            if self.lora_high:
                lines.append(f"    lora: {self.lora_high} @ {self.weight_high}")
        if self.trigger:
            lines.append(f"    trigger: \"{self.trigger}\"")
        return "\n".join(lines)


def parse_concept_ids(concepts_str: str) -> list[str]:
    """Accept comma-separated or one-per-line concept IDs."""
    ids = []
    for part in concepts_str.replace(",", "\n").split("\n"):
        cid = part.strip()
        if cid:
            ids.append(cid)
    return ids


def resolve_concepts(
    registry: ConceptRegistry,
    concept_ids: list[str],
    model_type: str,
) -> list[ResolvedConcept]:
    """Resolve a list of concept IDs against the registry for one model type."""
    profile = MODEL_PROFILES.get(model_type, {})
    is_split = profile.get("split", False)
    results: list[ResolvedConcept] = []

    for concept_id in concept_ids:
        concept = registry.get_concept(concept_id)
        if concept is None:
            results.append(ResolvedConcept(
                concept_id=concept_id, name=concept_id, model_type=model_type,
                lora_high=None, lora_low=None, weight_high=1.0, weight_low=1.0,
                trigger="", error="NOT FOUND",
            ))
            continue

        entry = concept.get("models", {}).get(model_type)
        if entry is None:
            results.append(ResolvedConcept(
                concept_id=concept_id, name=concept.get("name", concept_id),
                model_type=model_type, lora_high=None, lora_low=None,
                weight_high=1.0, weight_low=1.0, trigger="",
                error=f"not defined for {model_type}",
            ))
            continue

        if is_split:
            results.append(ResolvedConcept(
                concept_id=concept_id,
                name=concept.get("name", concept_id),
                model_type=model_type,
                lora_high=entry.get("lora_high") or None,
                lora_low=entry.get("lora_low") or None,
                weight_high=entry.get("weight_high", 1.0),
                weight_low=entry.get("weight_low", 1.0),
                trigger=entry.get("trigger", ""),
            ))
        else:
            results.append(ResolvedConcept(
                concept_id=concept_id,
                name=concept.get("name", concept_id),
                model_type=model_type,
                lora_high=entry.get("lora") or None,
                lora_low=None,
                weight_high=entry.get("weight", 1.0),
                weight_low=1.0,
                trigger=entry.get("trigger", ""),
            ))

    return results


def assemble_prompt(triggers: list[str], base_prompt: str, trigger_position: str) -> str:
    """Prepend or append collected trigger texts to the base prompt."""
    trigger_text = ", ".join(t.strip() for t in triggers if t.strip())
    if not trigger_text:
        return base_prompt
    if not base_prompt.strip():
        return trigger_text
    if trigger_position == "prepend":
        return f"{trigger_text}, {base_prompt}"
    return f"{base_prompt}, {trigger_text}"


def format_resolved_info(
    model_type: str,
    resolved: list[ResolvedConcept],
    assembled_prompt: str,
) -> str:
    profile = MODEL_PROFILES.get(model_type, {})
    lines = [f"Model type: {model_type} ({profile.get('display', model_type)})"]
    lines.append("Resolved concepts:")
    for r in resolved:
        lines.append(r.format_info())
    if assembled_prompt:
        lines.append(f"Assembled prompt: \"{assembled_prompt}\"")
    return "\n".join(lines)


def build_model_entry(
    model_type: str,
    lora: str,
    lora_low: str,
    weight: float,
    weight_low: float,
    trigger: str,
) -> dict:
    """Build the model-specific entry dict for ConceptDefine."""
    is_split = MODEL_PROFILES.get(model_type, {}).get("split", False)
    lora_val = lora if lora and lora != "None" else ""
    if is_split:
        lora_low_val = lora_low if lora_low and lora_low != "None" else ""
        return {
            "lora_high": lora_val,
            "lora_low": lora_low_val,
            "weight_high": weight,
            "weight_low": weight_low,
            "trigger": trigger,
        }
    return {"lora": lora_val, "weight": weight, "trigger": trigger}
