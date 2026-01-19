# ScenePromptManager - Three-Tab Interface

## Overview
The ScenePromptManager now features a comprehensive three-tab interface for managing prompts and compositions in a single node.

## Tab Structure

### 1. Define Tab 📝
**Purpose:** Create and manage individual reusable prompt components.

**Features:**
- Add/Edit/Remove prompts
- Configure processing type (raw or libber)
- Assign libber for substitution
- Categorize prompts for organization
- Apply changes to collection_json

**Fields:**
- **Key:** Unique identifier for the prompt
- **Value:** The actual prompt text
- **Type:** raw (use as-is) or libber (with substitution)
- **Libber:** Which libber to use for substitution
- **Category:** Optional grouping/organization

### 2. Compose Tab 🎨
**Purpose:** Create compositions by combining multiple prompts.

**Features:**
- Add/Edit/Remove compositions
- Select multiple prompt keys for each composition
- Name each composition uniquely
- Apply compositions to collection_json

**Workflow:**
1. Enter composition name
2. Select prompt keys using multi-select dropdown
3. Click ➕ to add composition
4. Edit existing compositions by adding/removing keys
5. Click "Apply Compositions" to save

**Composition Structure:**
```json
{
  "compositions": {
    "final_prompt": ["char1", "setting", "quality"],
    "negative": ["bad_quality", "artifacts"]
  }
}
```

### 3. View Tab 👁️
**Purpose:** Preview fully processed composition outputs.

**Features:**
- Dropdown to select composition
- Large textarea showing processed output
- Character count display
- Copy to clipboard button

**What You See:**
- Final text after combining all prompts
- Libber substitutions applied automatically
- Real-time preview from backend processing

## Data Flow

### Frontend → Backend
1. User edits prompts/compositions in Define/Compose tabs
2. Clicks Apply button to update `collection_json` widget
3. Backend receives updated JSON when node executes

### Backend → Frontend
1. Backend processes prompts and compositions
2. Returns 6 text outputs:
   - `text[0]`: collection_json
   - `text[1]`: prompts_list (array of prompt objects)
   - `text[2]`: status message
   - `text[3]`: available_libbers (array)
   - `text[4]`: compositions_list (array of composition objects)
   - `text[5]`: prompt_dict (dict of composed outputs)
3. Frontend parses and updates all three tabs

## Backend Support

### PromptCollection Model
```python
class PromptCollection(BaseModel):
    version: int = 2
    prompts: dict[str, Prompt] = {}
    compositions: dict[str, List[str]] = {}  # New field
```

### ScenePromptManager Node
**Outputs:**
1. `SCENE_INFO`: Scene information dictionary
2. `DICT`: prompt_dict containing composed outputs
3. `STRING`: Status message

**UI Data Array:**
- Text widget array contains all data for frontend rendering
- Compositions automatically processed on execute()
- Uses LibberStateManager for substitutions

## Usage Examples

### Example 1: Complete Workflow
```json
// 1. Define prompts in Define tab
{
  "prompts": {
    "char1": {
      "value": "beautiful woman with long hair",
      "processing_type": "raw"
    },
    "setting": {
      "value": "in a %location%, %time_of_day%",
      "processing_type": "libber",
      "libber_name": "scene_libber"
    },
    "quality": {
      "value": "masterpiece, best quality, highly detailed",
      "processing_type": "raw"
    }
  }
}

// 2. Create composition in Compose tab
{
  "compositions": {
    "final_prompt": ["char1", "setting", "quality"]
  }
}

// 3. View output in View tab
// Output: "beautiful woman with long hair, in a forest, sunset, masterpiece, best quality, highly detailed"
```

### Example 2: Atomic Prompts
```json
{
  "prompts": {
    "base": {"value": "portrait photo", "processing_type": "raw"},
    "subject": {"value": "%character_name%", "processing_type": "libber", "libber_name": "char_libber"},
    "style": {"value": "cinematic lighting", "processing_type": "raw"}
  },
  "compositions": {
    "positive": ["base", "subject", "style"],
    "negative": ["bad_quality"]
  }
}
```

## Implementation Details

### State Management
```javascript
let currentPromptsData = [];          // Array of prompt objects
let currentCompositionsData = [];     // Array of composition objects
let currentPromptDict = {};           // Dict of composed outputs
let availableLibbers = ["none"];      // Available libbers from backend
let activeTab = "define";             // Current active tab
```

### Event Handlers
- **Tab switching:** Updates activeTab and re-renders
- **Define tab:** Add/remove/apply prompts, type selection, libber dropdown
- **Compose tab:** Add/remove compositions, add/remove keys, apply compositions
- **View tab:** Composition selection, copy to clipboard

### Rendering Pipeline
1. `renderTabs()` - Creates tab buttons with active styling
2. `renderDefineTab()` - Returns Define tab HTML
3. `renderComposeTab()` - Returns Compose tab HTML
4. `renderViewTab()` - Returns View tab HTML
5. `renderTable()` - Coordinates everything, sets container.innerHTML
6. `attachEventHandlers()` - Attaches all event listeners
7. `updateContainerHeight()` - Adjusts container size

## Data Persistence

All data is saved in `prompts.json`:
```json
{
  "version": 2,
  "prompts": {
    "key1": { "value": "...", "processing_type": "...", ... },
    "key2": { ... }
  },
  "compositions": {
    "comp1": ["key1", "key2"],
    "comp2": ["key1", "key3"]
  }
}
```

## Testing Checklist

### Define Tab
- [ ] Add new prompt
- [ ] Remove existing prompt
- [ ] Edit prompt value/type/libber/category
- [ ] Switch between raw/libber type
- [ ] Libber dropdown updates when type changes
- [ ] Apply button updates collection_json

### Compose Tab
- [ ] Add new composition with multiple keys
- [ ] Remove existing composition
- [ ] Add key to existing composition
- [ ] Remove key from composition
- [ ] Apply button updates collection_json
- [ ] Composition count updates correctly

### View Tab
- [ ] Dropdown shows all compositions
- [ ] Selecting composition updates textarea
- [ ] Character count updates correctly
- [ ] Copy button copies to clipboard
- [ ] Shows "no compositions" when empty

### Integration
- [ ] Tab switching preserves data
- [ ] Backend updates all tabs correctly
- [ ] Compositions persist in prompts.json
- [ ] Libber substitutions work in outputs
- [ ] prompt_dict output matches View tab

## Browser Console Commands

```javascript
// Check current state
console.log("Active Tab:", activeTab);
console.log("Prompts:", currentPromptsData);
console.log("Compositions:", currentCompositionsData);
console.log("Prompt Dict:", currentPromptDict);

// Force re-render
renderTable(currentPromptsData, availableLibbers, currentCompositionsData, currentPromptDict);
```

## Notes

- **No Service Restart Needed:** JavaScript changes only require browser refresh
- **Backward Compatible:** Existing scenes without compositions will work normally
- **Composition Format:** compositions is a dict with composition_name → [prompt_keys]
- **Output Format:** prompt_dict is a dict with composition_name → composed_text
- **Libber Integration:** Backend automatically applies libber substitutions to composed prompts
