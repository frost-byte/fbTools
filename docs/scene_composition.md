# Scene Composition Nodes

Scene templates define the structure of a scene — its slots (who appears), its shots (what happens), and its environment. Scene Compose fills those slots with subject profiles and dialogue, producing a SCENE_INSTANCE that Prompt Assemble turns into a model-ready prompt.

**Category:** `🧊 frost-byte/Scene`

---

## Nodes

### Scene Template Load

Loads a scene template from the `scene_templates/` directory and exposes it for Scene Compose.

| Input | Type | Notes |
|---|---|---|
| Template ID | COMBO | Template to load. Press R to refresh the list after adding new templates. |

| Output | Type | Notes |
|---|---|---|
| Scene Template | SCENE_TEMPLATE | Template object for wiring into Scene Compose. |
| Slot Info | STRING | Formatted summary of the template's slot requirements, shown in the node UI. |

The node re-executes automatically when the template file changes on disk. Use `POST /fbtools/scene_templates/reload` to force a refresh after external edits without restarting ComfyUI.

---

### Scene Template List

Scans the `scene_templates/` directory and lists all available templates.

| Input | Type | Notes |
|---|---|---|
| *(none)* | — | No inputs required. |

| Output | Type | Notes |
|---|---|---|
| Template List | STRING | Formatted list of all templates with their descriptions, shown in the node UI. |
| Template Count | INT | Number of templates found. |

---

### Scene Compose

Assigns subjects to a template's slots, maps dialogue lines to shots in order, and applies optional outfit overrides. Outputs a SCENE_INSTANCE ready for Prompt Assemble.

| Input | Type | Notes |
|---|---|---|
| Scene Template | SCENE_TEMPLATE | Template from Scene Template Load. |
| Slot A | SUBJECT_PROFILE | Subject assigned to slot A (typically the primary speaker). Optional. |
| Slot B | SUBJECT_PROFILE | Subject assigned to slot B. Optional. |
| Slot C | SUBJECT_PROFILE | Subject assigned to slot C. Optional. |
| Slot D | SUBJECT_PROFILE | Subject assigned to slot D. Optional. |
| Dialogue 1 | STRING | Fills the first placeholder dialogue shot in template shot order. Optional. |
| Dialogue 2 | STRING | Fills the second placeholder dialogue shot. Optional. |
| Dialogue 3 | STRING | Fills the third placeholder dialogue shot. Optional. |
| Dialogue 4 | STRING | Fills the fourth placeholder dialogue shot. Optional. |
| Outfit Override A | STRING | Replaces slot A's default outfit for this scene only. Optional. |
| Outfit Override B | STRING | Replaces slot B's default outfit for this scene only. Optional. |
| Outfit Override C | STRING | Replaces slot C's default outfit for this scene only. Optional. |
| Outfit Override D | STRING | Replaces slot D's default outfit for this scene only. Optional. |

| Output | Type | Notes |
|---|---|---|
| Scene Instance | SCENE_INSTANCE | Composed scene dict for wiring into Prompt Assemble. |
| Scene Summary | STRING | Human-readable summary of the composition, shown in the node UI. Includes any validation warnings. |

**Dialogue mapping** is positional: Dialogue 1 fills whichever shot in the template is marked `placeholder: true` first (by shot order), Dialogue 2 fills the second such shot, and so on. You don't need to know the shot IDs.

**Validation** warns but never hard-fails — if a required slot is missing or a voice/character-sheet requirement isn't met, the warning appears in the Scene Summary and the node still outputs a scene instance so you can inspect it.

---

## Scene Template Format

Templates are JSON files in `ComfyUI/user/default/comfyui-fbTools/scene_templates/`. Three bundled examples are seeded there automatically on first use:

| Template ID | Description |
|---|---|
| `monologue_indoor` | Single subject, indoor setting, 3 shots |
| `cafe_conversation_2p` | Two subjects in a café, 4 shots with placeholder dialogue |
| `meeting_room_3p` | Three subjects in a meeting room |

A template defines:
- **slots** — which roles appear (`A`, `B`, `C`, …) and whether each needs a voice reference or character sheet images
- **environment** — summary and lighting description
- **style** — visual style hint
- **shots** — ordered list of shot IDs, each with camera angle, action description, optional placeholder dialogue, and sound events

Placeholder substitution: `{A}`, `{B}`, `{C}`, `{D}` in action and camera fields are replaced at assembly time with the subject's appearance description.

---

## Typical Workflow

```
[Scene Template Load]          [Subject Profile Load]    [Subject Profile Load]
  template_id: cafe_conv_2p      subject_id: char_alice    subject_id: char_bob
       │ template                      │ subject_profile         │ subject_profile
       └─────────────────────────┬─────┘                         │
                                 ▼                               │
                          [Scene Compose]  ◄─────────────────────┘
                            slot_A ◄── char_alice
                            slot_B ◄── char_bob
                            dialogue_1: "Hi, long time no see."
                            dialogue_2: "Indeed! How have you been?"
                                 │ scene_instance
                                 ▼
                          [Prompt Assemble]
```

---

## Writing a Custom Template

Create a new JSON file in `scene_templates/`. Minimal structure:

```json
{
  "id": "my_template",
  "name": "My Template",
  "description": "Two subjects, outdoor, 2 shots",
  "slots": {
    "A": {"role": "speaker",   "needs_voice": true,  "needs_character_sheet": true},
    "B": {"role": "listener",  "needs_voice": false, "needs_character_sheet": true}
  },
  "environment": {
    "summary": "sunny park bench",
    "lighting": "natural afternoon light"
  },
  "style": "cinematic",
  "shots": [
    {
      "id": "shot_1",
      "timestamp": "0:00",
      "camera": "medium shot of {A}",
      "action": "{A} sits down next to {B}",
      "dialogue": {"placeholder": true, "speaker": "A"},
      "sound_events": []
    },
    {
      "id": "shot_2",
      "timestamp": "0:05",
      "camera": "reaction shot of {B}",
      "action": "{B} smiles and nods",
      "dialogue": {"placeholder": true, "speaker": "B"},
      "sound_events": ["birds chirping"]
    }
  ],
  "overall_soundscape": "gentle outdoor ambience",
  "non_diegetic_music": "soft piano"
}
```

Refresh the ComfyUI page after adding new template files to see them in the Scene Template Load dropdown.
