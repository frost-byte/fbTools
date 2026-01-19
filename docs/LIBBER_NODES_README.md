# Libber System Documentation

## Overview

The Libber system provides a powerful string templating and substitution mechanism for ComfyUI workflows. It allows you to define reusable text snippets (called "libs") that can be referenced in other strings using delimiters, with support for recursive substitution.

## What is a Libber?

A **Libber** is a collection of key-value pairs where:
- **Keys**: Short identifiers (e.g., `character`, `style`, `quality`)
- **Values**: Text snippets that can reference other keys
- **Delimiter**: Character(s) used to wrap key references (default: `%`)

### Example

```python
libs = {
    "quality": "masterpiece, best quality, ultra detailed",
    "lighting": "soft diffused lighting, golden hour",
    "character": "beautiful woman, %quality%, %lighting%"
}

# Using the libber:
"%character%, smiling"
# Result: "beautiful woman, masterpiece, best quality, ultra detailed, soft diffused lighting, golden hour, smiling"
```

### Recursive Substitution

Libbers support nested references with depth limiting to prevent infinite loops:

```python
libs = {
    "chunky": "incredibly thick, and %yummy%",
    "yummy": "delicious and %perfect%",
    "perfect": "absolutely amazing",
    "meal": "A %chunky% steak"
}

# "%meal%" becomes:
# "A incredibly thick, and delicious and absolutely amazing steak"
```

## Nodes

### LibberManager

**Category:** 🧊 frost-byte/Libber

Interactive table editor for creating and managing libbers.

#### Features

- **Interactive Table**: Edit key-value pairs directly in the node
- **Inline Actions**: Per-row buttons for Update (✏️) and Remove (➖)
- **Add New**: Bottom row for adding new key-value pairs (➕)
- **Sticky Controls**: Always-visible Load, Save, and Create buttons
- **Auto-Save**: Changes automatically saved after add/update/remove operations
- **Smart Loading**: Automatically loads libber from memory or disk on node creation

#### Inputs

- `libber_name`: Combo dropdown of available libbers (basenames only)
  - Shows all `.json` files from `libber_dir` without the extension
  - Select "none" to work with the table without a loaded libber
- `libber_dir`: Directory for libber files (default: `output/libbers`)
- `delimiter`: Character(s) to wrap key references (default: `%`)
- `max_depth`: Maximum recursion depth for substitutions (default: 10)

#### Outputs

- `status`: Status message about current operation
- `keys`: List of keys as text

#### UI Controls

1. **📂 Load Button**: Load the selected libber from disk into memory
2. **💾 Save Button**: Save current libber to disk
3. **Text Input + ➕ Create Button**: Create a new libber
   - Enter a name in the text field
   - Click Create to create and save the new libber
   - Automatically updates the dropdown and switches to the new libber

#### Table Structure

| Column | Description |
|--------|-------------|
| 🗝️ Key | The identifier for the lib entry (editable textarea) |
| 🪙 Value | The text content (editable textarea, supports multiline) |
| ⚡ Actions | Update (✏️) and Remove (➖) buttons |

**New Row (bottom):**
- Empty key and value textareas
- ➕ Add button to insert the new entry

#### Workflow

1. **Create a New Libber:**
   - Enter name in text field (e.g., "character_presets")
   - Click ➕ Create
   - Libber is created, saved, and loaded

2. **Add Entries:**
   - Type key in bottom row (e.g., "warrior")
   - Type value (e.g., "muscular fighter, armor, %quality%")
   - Click ➕ Add
   - Entry is automatically saved

3. **Edit Entries:**
   - Click in any Key or Value textarea to edit
   - Modify the text
   - Click ✏️ Update to save changes
   - Auto-saves to disk

4. **Remove Entries:**
   - Click ➖ Remove button on any row
   - Entry is deleted and auto-saved

5. **Load/Save:**
   - 📂 Load: Refresh from disk (discards unsaved changes)
   - 💾 Save: Manually save to disk (usually not needed due to auto-save)

#### Technical Details

- **Schema Simplified**: Single `libber_name` Combo widget (basenames only)
- **Backend Auto-Creation**: If libber doesn't exist, it's created automatically
- **Smart Loading Priority**: Memory → File → Create new
- **File Format**: JSON with structure:
  ```json
  {
    "delimiter": "%",
    "max_depth": 10,
    "lib_dict": {
      "key1": "value1",
      "key2": "value2 with %key1% reference"
    }
  }
  ```

### LibberApply

**Category:** 🧊 frost-byte/Libber

Apply libber substitutions to text with click-to-insert functionality.

#### Features

- **Click-to-Insert**: Click any row in the table to insert the key with delimiters
- **Cursor Tracking**: Maintains cursor position across focus changes
- **Native Undo/Redo**: Use Ctrl+Z / Cmd+Z to undo insertions
- **Always-Visible Refresh**: 🔄 Refresh button always shown (sticky at top)
- **Smart Libber Loading**: Automatically loads libber from disk if not in memory
- **Dynamic Table**: Shows all available keys and their values

#### Inputs

- `libber_name`: Combo dropdown of available libbers
- `text`: Multiline text input where substitutions are applied
- `skip_none`: Skip substitution if key not found (default: True)

#### Outputs

- `result`: Text with all substitutions applied

#### UI Display

**Refresh Button (Always Visible):**
- 🔄 Refresh: Refreshes the libber list from disk
- Lib Count: Shows number of entries (e.g., "15 libs")

**Table View:**
| Column | Description |
|--------|-------------|
| 🗝️ Lib | The key name |
| 🪙 Value | The resolved value (shows actual text) |

**Click Behavior:**
- Click any row to insert that key wrapped in delimiters
- Insertion happens at the last known cursor position in the `text` widget
- Text is inserted using `document.execCommand` for native undo support

#### Workflow

1. **Select Libber:**
   - Choose a libber from the `libber_name` dropdown
   - Table displays all available keys and values

2. **Insert Keys:**
   - Click in the `text` widget to set cursor position
   - Click any row in the table
   - Key is inserted as `%key%` (or with your custom delimiter)
   - Cursor remains at insertion point for continued editing

3. **Edit Text:**
   - Type additional text around inserted keys
   - Use Ctrl+Z to undo insertions
   - Keys are highlighted/recognizable by delimiters

4. **Execute:**
   - Click Execute or run the workflow
   - All key references are replaced with their values
   - Recursive substitutions are applied up to `max_depth`

5. **Refresh:**
   - Click 🔄 Refresh to update the libber list
   - Useful after creating new libbers in LibberManager
   - Re-selects the current libber or picks the first available

#### Example Usage

**Setup:**
```
Libber: character_presets
Keys:
  quality: "masterpiece, best quality, ultra detailed"
  warrior: "muscular fighter, armor, determined expression"
  mage: "wise spellcaster, flowing robes, glowing staff"
```

**Input Text:**
```
%warrior%, %quality%, standing in a castle
```

**Output:**
```
muscular fighter, armor, determined expression, masterpiece, best quality, ultra detailed, standing in a castle
```

#### Technical Details

- **Cursor Position Storage**: Tracks `selectionStart` and `selectionEnd` across focus events
- **Event Listeners**: `click`, `keyup`, `select`, `focus` on the text input
- **Insert Method**: `document.execCommand('insertText')` for undo stack support
- **Auto-Loading**: Checks memory first, loads from file if needed
- **Refresh Logic**:
  1. Fetch libber list from server
  2. Check for files on disk (`.json` in `libber_dir`)
  3. Update dropdown options
  4. Maintain current selection if still available
  5. Load selected libber into memory if needed

## REST API

The Libber system includes REST endpoints for JavaScript integration:

### Endpoints

#### `POST /fbtools/libber/create`
Create a new libber in memory.

**Request:**
```json
{
  "name": "my_libber",
  "delimiter": "%",
  "max_depth": 10
}
```

**Response:**
```json
{
  "name": "my_libber",
  "keys": [],
  "delimiter": "%"
}
```

#### `POST /fbtools/libber/load`
Load a libber from disk into memory.

**Request:**
```json
{
  "name": "my_libber",
  "filepath": "output/libbers/my_libber.json"
}
```

#### `POST /fbtools/libber/add_lib`
Add or update a key-value pair.

**Request:**
```json
{
  "name": "my_libber",
  "key": "warrior",
  "value": "muscular fighter, armor"
}
```

#### `POST /fbtools/libber/remove_lib`
Remove a key-value pair.

**Request:**
```json
{
  "name": "my_libber",
  "key": "warrior"
}
```

#### `POST /fbtools/libber/save`
Save libber to disk.

**Request:**
```json
{
  "name": "my_libber",
  "filepath": "output/libbers/my_libber.json"
}
```

#### `GET /fbtools/libber/list`
List all available libbers.

**Response:**
```json
{
  "libbers": ["my_libber", "presets"],
  "files": ["my_libber.json", "presets.json"]
}
```

#### `GET /fbtools/libber/get_data/{name}`
Get libber data including all keys and values.

**Response:**
```json
{
  "lib_dict": {
    "warrior": "muscular fighter",
    "quality": "best quality"
  },
  "delimiter": "%",
  "max_depth": 10
}
```

#### `POST /fbtools/libber/apply`
Apply substitutions to text.

**Request:**
```json
{
  "name": "my_libber",
  "text": "A %warrior% with %quality%",
  "skip_none": true
}
```

**Response:**
```json
{
  "result": "A muscular fighter with best quality"
}
```

## Use Cases

### Character Presets
```python
{
  "base_male": "handsome man, strong jawline, athletic build",
  "base_female": "beautiful woman, elegant features",
  "hero": "%base_male%, confident expression, heroic pose",
  "warrior": "%base_male%, armor, battle-worn, determined",
  "mage": "%base_female%, flowing robes, mystical aura"
}
```

### Quality Presets
```python
{
  "quality_high": "masterpiece, best quality, ultra detailed, 8k",
  "quality_med": "high quality, detailed",
  "quality_low": "simple, sketch",
  "style_realistic": "photorealistic, lifelike, cinematic lighting",
  "final": "%quality_high%, %style_realistic%"
}
```

### Scene Components
```python
{
  "lighting_day": "bright sunlight, clear sky, shadows",
  "lighting_night": "moonlight, stars, dark atmosphere",
  "location_castle": "medieval castle, stone walls, towers",
  "location_forest": "dense forest, trees, natural lighting",
  "scene_day_castle": "%location_castle%, %lighting_day%",
  "scene_night_forest": "%location_forest%, %lighting_night%"
}
```

### Workflow Variations
```python
{
  "lora_style_a": "<lora:style_a:0.8>",
  "lora_style_b": "<lora:style_b:0.6>",
  "lora_character": "<lora:character:1.0>",
  "workflow_a": "%lora_style_a%, %lora_character%",
  "workflow_b": "%lora_style_b%, %lora_character%"
}
```

## Best Practices

1. **Naming Keys**: Use descriptive, lowercase keys with underscores
   - Good: `character_warrior`, `quality_high`, `lighting_sunset`
   - Avoid: `w`, `q1`, `light`

2. **Organize by Category**: Group related libs together
   - Character presets in one libber
   - Quality settings in another
   - Scene components in a third

3. **Keep Values Focused**: Each lib should represent one concept
   - Don't: `"warrior": "strong warrior, best quality, in a castle"`
   - Do: `"warrior": "strong fighter, armor"` + reference quality/location separately

4. **Use Recursion Wisely**: Max 2-3 levels of nesting for readability
   - Good: `base → variant → final`
   - Avoid: Chains longer than 3-4 levels

5. **Test Substitutions**: Use LibberApply to verify output before using in workflows

6. **Backup Important Libbers**: Save copies of libber JSON files before major changes

7. **Document Complex Chains**: Add comments in your workflow about substitution logic

## Migration from Other Systems

If you're coming from other templating systems:

### From Simple Find/Replace
- Each find/replace rule becomes a lib entry
- Benefit: Reusability across multiple prompts

### From Wildcards
- Each wildcard file becomes a libber
- Wildcard `{term1|term2}` becomes separate lib entries
- Use libber keys instead of wildcard syntax

### From Prompt Presets
- Each preset becomes a lib entry
- Combine presets using references: `%preset_a%, %preset_b%`

## Troubleshooting

### "Libber not found" Error
- Check that the libber file exists in `libber_dir`
- Click 📂 Load in LibberManager
- Verify filename matches dropdown selection

### Infinite Recursion
- Check for circular references (A → B → A)
- Reduce `max_depth` to catch infinite loops earlier
- Review substitution chains for cycles

### Keys Not Substituting
- Verify delimiter matches (check `delimiter` setting)
- Ensure key names match exactly (case-sensitive)
- Check that `skip_none` is set appropriately

### Table Not Updating
- Click 🔄 Refresh to reload from disk
- Check console for JavaScript errors
- Restart ComfyUI if UI is unresponsive

### Auto-Save Not Working
- Verify `libber_dir` is writable
- Check ComfyUI console for permission errors
- Manually save with 💾 Save button as backup

## Advanced Features

### Custom Delimiters

Change the delimiter to avoid conflicts:

```python
# Using << >> as delimiter
libber_dict = {
    "quality": "best quality",
    "char": "warrior"
}
# Text: "A <<char>> with <<quality>>"
# Result: "A warrior with best quality"
```

### Skip None Behavior

Control what happens when a key is not found:

- `skip_none=True`: Leave placeholder as-is (`%unknown%` stays `%unknown%`)
- `skip_none=False`: Remove placeholder (`%unknown%` becomes empty string)

### Multiple Libbers

Use different libbers for different purposes:

1. **Characters Libber**: Character descriptions
2. **Styles Libber**: Art styles and quality settings
3. **Scenes Libber**: Scene and environment descriptions

Switch between them in LibberApply as needed, or use multiple LibberApply nodes in sequence.

## JavaScript Integration

See [js/api/libber.js](js/api/libber.js) for the JavaScript API client.

Example usage in ComfyUI frontend:

```javascript
import { libberAPI } from "./api/libber.js";

// Create a new libber
await libberAPI.createLibber("my_libber", "%", 10);

// Add entries
await libberAPI.addLib("my_libber", "quality", "best quality, ultra detailed");

// Save to disk
await libberAPI.saveLibber("my_libber", "output/libbers/my_libber.json");

// Apply substitutions
const result = await libberAPI.applySubstitutions("my_libber", "%quality%, warrior");
console.log(result); // "best quality, ultra detailed, warrior"
```

## Architecture

### Backend (Python)
- `Libber` class: Core substitution engine in `extension.py`
- `LibberStateManager`: Server-side session management
- REST endpoints: CRUD operations for libber data

### Frontend (JavaScript)
- `js/nodes/libber.js`: LibberManager and LibberApply UI implementations
- `js/api/libber.js`: REST API client
- Modular architecture with shared utilities

### Data Flow
1. User interacts with LibberManager table
2. JavaScript calls REST API
3. Backend updates in-memory libber
4. Changes saved to JSON file
5. LibberApply nodes reference the same in-memory data

## File Format

Libber JSON files are stored in `output/libbers/` with this structure:

```json
{
  "delimiter": "%",
  "max_depth": 10,
  "lib_dict": {
    "key1": "value1",
    "key2": "value2 with %key1% reference",
    "key3": "value3 with %key2% nested reference"
  }
}
```

**Fields:**
- `delimiter`: String used to wrap key references
- `max_depth`: Maximum recursion depth for substitutions
- `lib_dict`: Dictionary of key-value pairs

## See Also

- [Main README](README.md): Overview of all fbTools features
- [Story Nodes Documentation](STORY_NODES_README.md): Multi-scene storytelling
- [Scene Nodes Documentation](SCENE_NODES_README.md): Scene management
- [JavaScript Architecture](js/README.md): Frontend code organization
