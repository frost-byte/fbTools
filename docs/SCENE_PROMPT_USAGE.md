# Scene Prompt System - User Guide

## Quick Start

The Scene Prompt System allows you to create reusable prompt components and combine them flexibly for different workflows (image generation, video, etc.).

### Three-Node Workflow

1. **ScenePromptManager** - Create and organize reusable prompts
2. **PromptComposer** - Combine prompts into named outputs
3. **Your Workflow** - Use the composed prompts

---

## ScenePromptManager Node

**Purpose:** Create a library of reusable prompt components

### Table Columns Explained

| Column | Purpose | Example | Notes |
|--------|---------|---------|-------|
| **🗝️ Key** | Unique identifier | `char1`, `quality`, `setting` | Used to reference the prompt in composition |
| **💬 Value** | The prompt text | `beautiful woman, long hair` | The actual content that will be used |
| **🔧 Type** | Processing mode | `📝 raw` or `🔄 libber` | Raw = use as-is, Libber = substitute placeholders |
| **🪙 Libber** | Libber name | `char_lib`, `scene_lib` | Which libber to use (only for libber type) |
| **🏷️ Category** | Organization | `character`, `scene`, `quality` | Optional grouping for your reference |
| **⚡ Actions** | Add/Remove | ➕ ➖ | Add new prompt or remove existing |

### Usage Tips

- **Key**: Think of it like a variable name. Use descriptive names like `char1_face`, `background_forest`, `quality_high`
- **Value**: Write your prompt normally. For libber type, use placeholders like `%hair_color%`, `%age%`
- **Type**: 
  - Use **raw** for fixed prompts that won't change
  - Use **libber** when you want dynamic substitution (e.g., changing hair color across scenes)
- **Libber**: Only enabled when Type is libber. Must match an existing LibberManager name
- **Category**: Helps you organize. Examples: `character`, `scene`, `quality`, `style`, `technical`

### Workflow

1. Click in the bottom row to add a new prompt
2. Fill in at minimum: **Key** and **Value**
3. Click ➕ to add it to the collection
4. Edit any row by changing the fields
5. Click **✓ Apply Changes** to save all modifications
6. Remove prompts with the ➖ button

---

## PromptComposer Node

**Purpose:** Combine prompts into named outputs for your workflow

### Inputs

- **scene_info**: Connect from ScenePromptManager (or any scene node)
- **composition_json**: Define which prompts go into which outputs

### Composition JSON Format

```json
{
  "output_name": ["prompt_key1", "prompt_key2", "prompt_key3"]
}
```

### Examples

#### Image Generation Workflow
```json
{
  "qwen_main": ["char1", "char2", "setting", "quality", "style"]
}
```
This creates one output named `qwen_main` combining 5 prompts in order.

#### Video High/Low Quality
```json
{
  "video_high": ["subject", "quality_8k", "motion", "lighting"],
  "video_low": ["subject", "quality_720p"]
}
```
This creates two outputs with different quality settings.

#### Multi-Image Workflow
```json
{
  "image_closeup": ["char1", "facial_detail", "quality_high"],
  "image_wide": ["char1", "char2", "setting", "quality_high"],
  "image_background": ["setting", "lighting", "quality_medium"]
}
```
Each output gets different prompt combinations.

### Outputs

- **prompt_dict**: Dictionary you can access in other nodes
  - Example: `prompt_dict["qwen_main"]` gives you the composed text
- **composition_json**: The composition map (for saving/reference)
- **info**: Human-readable summary of what was composed

---

## Two Workflow Approaches

The system supports two different strategies:

### Approach 1: Complete Prompts (Recommended for Most Users)

**Philosophy:** Each prompt is a complete, self-contained prompt. Use Libber for dynamic parts.

**Example:**

| Key | Value | Type | Libber | Category |
|-----|-------|------|--------|----------|
| `wan_high` | `beautiful woman with %hair_color% and %eye_color%, standing in a %setting%, masterpiece quality, 8k, highly detailed` | libber | `scene_lib` | main |
| `wan_low` | `woman in %setting%, normal quality` | libber | `scene_lib` | main |
| `four_image` | `four characters in %location%, cinematic composition, dramatic lighting` | libber | `scene_lib` | main |

**PromptComposer:**
```json
{
  "qwen_main": ["wan_high"],
  "video_prompt": ["wan_low"],
  "multi_subject": ["four_image"]
}
```

**Result:** Each output gets one complete prompt with Libber substitutions applied.

**Benefits:**
- Simple and straightforward
- Each prompt is readable as a complete unit
- Easy to understand what each output will be
- Similar to traditional prompt management

---

### Approach 2: Atomic Composition (Advanced)

**Philosophy:** Break prompts into small reusable pieces. Combine them flexibly.

**Example:**

## Complete Example Workflow (Atomic Approach)

### Step 1: Create Prompts (ScenePromptManager)

| Key | Value | Type | Libber | Category |
|-----|-------|------|--------|----------|
| `char1` | `beautiful woman, long hair` | raw | - | character |
| `char2` | `handsome man, strong` | raw | - | character |
| `char_dynamic` | `A %age% %gender% with %hair%` | libber | `char_lib` | character |
| `setting` | `in a mystical forest` | raw | - | scene |
| `quality_high` | `masterpiece, 8k, highly detailed` | raw | - | quality |
| `quality_low` | `normal quality, 720p` | raw | - | quality |
| `style` | `oil painting style, dramatic lighting` | raw | - | style |

**Benefits:**
- Maximum flexibility and reusability
- Change one piece, affects all compositions using it
- Good for large projects with many variations
- Mix and match freely

**PromptComposer:**
```json
{
  "image_gen": ["char1", "char2", "setting", "quality_high", "style"],
  "video_high": ["char_dynamic", "setting", "quality_high"],
  "video_low": ["char_dynamic", "setting", "quality_low"]
}
```

**Result:** Multiple pieces concatenated together with Libber processing where needed.

---

### Which Approach Should You Use?

**Use Complete Prompts (Approach 1) when:**
- You have established prompt templates
- Each scene/output needs a specific prompt
- You want simple, readable prompts
- You're migrating from legacy systems (girl_pos, wan_prompt, etc.)

**Use Atomic Composition (Approach 2) when:**
- You need maximum flexibility
- Many outputs share common elements
- You want to experiment with combinations
- You're building a large prompt library

**Mix both:**
- Use complete prompts for main outputs
- Use atomic pieces for shared elements like quality, style
- Example: `["wan_high", "quality_8k", "style_cinematic"]`

---

## Complete Example: Traditional Workflow (Complete Prompts)

### Step 1: Create Complete Prompts

| Key | Value | Type | Libber | Category |
|-----|-------|------|--------|----------|
| `wan_high` | `%char_desc% woman with %hair% and %eyes%, in a %setting%, wearing %outfit%, masterpiece, 8k, highly detailed, perfect anatomy` | libber | `scene_lib` | image |
| `wan_low` | `%char_desc% woman in %setting%, normal quality` | libber | `scene_lib` | image |
| `girl_pos` | `beautiful %age% woman, %hair%, %pose%, professional photography` | libber | `char_lib` | character |
| `male_pos` | `handsome man, %body_type%, %expression%, cinematic lighting` | libber | `char_lib` | character |
| `four_image` | `four people in %location%, %composition%, dramatic scene` | libber | `scene_lib` | multi |

### Step 2: Setup Libber (LibberManager)

**scene_lib:**
```json
{
  "char_desc": "beautiful",
  "hair": "long flowing hair",
  "eyes": "piercing blue eyes",
  "setting": "mystical forest",
  "outfit": "elegant dress",
  "location": "ancient temple",
  "composition": "symmetrical composition"
}
```

**char_lib:**
```json
{
  "age": "young",
  "pose": "standing confidently",
  "body_type": "athletic build",
  "expression": "intense gaze"
}
```

### Step 3: Compose for Workflow

**Image Generation:**
```json
{
  "qwen_main": ["wan_high"]
}
```

**Video Generation:**
```json
{
  "video_high": ["wan_high"],
  "video_low": ["wan_low"]
}
```

**Multi-Character Scene:**
```json
{
  "main_prompt": ["girl_pos", "male_pos"],
  "environment": ["four_image"]
}
```

**Result:**
- `qwen_main`: "beautiful woman with long flowing hair and piercing blue eyes, in a mystical forest, wearing elegant dress, masterpiece, 8k, highly detailed, perfect anatomy"
- Each time you change the libber values, all prompts update automatically
- Clean, readable, traditional workflow

---

## Complete Example: Atomic Workflow (Building Blocks)

### Step 1: Create Atomic Prompts (ScenePromptManager)

```json
{
  "image_gen": ["char1", "char2", "setting", "quality_high", "style"],
  "video_high": ["char_dynamic", "setting", "quality_high"],
  "video_low": ["char_dynamic", "setting", "quality_low"]
}
```

| `char2` | `handsome man, strong` | raw | - | character |
| `char_dynamic` | `A %age% %gender% with %hair%` | libber | `char_lib` | character |
| `setting` | `in a mystical forest` | raw | - | scene |
| `quality_high` | `masterpiece, 8k, highly detailed` | raw | - | quality |
| `quality_low` | `normal quality, 720p` | raw | - | quality |
| `style` | `oil painting style, dramatic lighting` | raw | - | style |

### Step 2: Compose with Atomic Pieces

```json
{
  "image_gen": ["char1", "char2", "setting", "quality_high", "style"],
  "video_high": ["char_dynamic", "setting", "quality_high"],
  "video_low": ["char_dynamic", "setting", "quality_low"]
}
```

**Result:** Prompts assembled from multiple pieces, offering maximum flexibility.

---

## Comparison

| Aspect | Complete Prompts | Atomic Composition |
|--------|------------------|-------------------|
| **Complexity** | Simple | Advanced |
| **Readability** | High (each prompt standalone) | Lower (must mentally combine) |
| **Flexibility** | Medium | Very High |
| **Setup Time** | Fast | Slower |
| **Maintenance** | Update each prompt individually | Update shared pieces once |
| **Best For** | Traditional workflows, specific scenes | Large projects, experimentation |
| **Libber Usage** | Within complete prompts | Can use in any piece |

---

## Hybrid Approach (Best of Both)

Combine both strategies:

```json
{
  "main_output": ["wan_high", "quality_8k", "style_cinematic"],
  "background": ["setting_forest", "lighting_dramatic"]
}
```

- `wan_high`: Complete character prompt with libber
- `quality_8k`, `style_cinematic`: Shared technical tags
- Mix complete and atomic as needed

---

## Migration Strategy

If you're coming from legacy prompts:

**Legacy System:**
- `girl_pos`: Complete prompt
- `male_pos`: Complete prompt
- `wan_prompt`: Complete prompt with high quality
- `wan_low_prompt`: Complete prompt with low quality

**New System (Complete Approach - Recommended):**

1. Create prompts with same keys
2. Use libber for dynamic parts
3. Compose outputs selecting the prompts you need

**Example Migration:**

| Old Key | New Key | New Value | Type |
|---------|---------|-----------|------|
| `girl_pos` | `girl_pos` | `beautiful woman, %hair%, %pose%` | libber |
| `male_pos` | `male_pos` | `handsome man, %build%, %expression%` | libber |
| `wan_prompt` | `wan_high` | `%subject% in %setting%, masterpiece, 8k` | libber |
| `wan_low_prompt` | `wan_low` | `%subject% in %setting%, normal quality` | libber |

Then use PromptComposer to select which ones for which outputs!

---

### Step 3: Use in Workflow

The `prompt_dict` output contains:
- `image_gen`: "beautiful woman, long hair handsome man, strong in a mystical forest masterpiece, 8k, highly detailed oil painting style, dramatic lighting"
- `video_high`: "A magnificent warrior with flowing hair in a mystical forest masterpiece, 8k, highly detailed" (after libber substitution)
- `video_low`: "A magnificent warrior with flowing hair in a mystical forest normal quality, 720p"

The `prompt_dict` output contains:
- `image_gen`: "beautiful woman, long hair handsome man, strong in a mystical forest masterpiece, 8k, highly detailed oil painting style, dramatic lighting"
- `video_high`: "A magnificent warrior with flowing hair in a mystical forest masterpiece, 8k, highly detailed" (after libber substitution)
- `video_low`: "A magnificent warrior with flowing hair in a mystical forest normal quality, 720p"

---

## Real-World Usage Patterns

### Pattern 1: One Prompt Per Output (Simple)

**Use case:** Each output needs its own specific prompt

```json
{
  "qwen_main": ["wan_high"],
  "video_main": ["video_prompt"],
  "four_panel": ["four_image_prompt"]
}
```

Each key selects ONE complete prompt.

### Pattern 2: Prompt + Modifiers (Hybrid)

**Use case:** Complete prompt with shared technical tags

```json
{
  "render_high": ["wan_high", "quality_8k", "style_photo"],
  "render_low": ["wan_high", "quality_720p"],
  "artistic": ["wan_high", "style_painting", "lighting_dramatic"]
}
```

Same base prompt, different technical variations.

### Pattern 3: Multi-Character Assembly

**Use case:** Multiple characters in one scene

```json
{
  "two_char": ["girl_pos", "male_pos", "setting_forest"],
  "four_char": ["char1", "char2", "char3", "char4", "setting_temple"]
}
```

Combine multiple character prompts with environment.

---

## Advanced: Libber Integration

When you use `type=libber`, the PromptComposer will:

1. Get the prompt value (with placeholders)
2. Look up the specified libber by name
3. Substitute all placeholders like `%variable%`
4. Return the processed text

### Example with Libber

**LibberManager** named `char_lib`:
```json
{
  "age": "young",
  "gender": "warrior woman",
  "hair": "flowing silver hair"
}
```

**ScenePromptManager** prompt:
- Key: `templated_char`
- Value: `A %age% %gender% with %hair%`
- Type: `libber`
- Libber: `char_lib`

**PromptComposer** composition:
```json
{
  "main": ["templated_char", "setting"]
}
```

**Result**:
```
"A young warrior woman with flowing silver hair in a mystical forest"
```

Change the libber values, and the same composition produces different results!

---

## Tips & Best Practices

### Naming Conventions

- Use **snake_case** for keys: `char_main`, `quality_high`, `bg_forest`
- Be descriptive: `char1_closeup` is better than `c1`
- Use prefixes for organization: `char_*`, `bg_*`, `qual_*`, `tech_*`

### Organization Strategies

1. **By Category**: Group all characters, all qualities, all scenes
2. **By Scene**: `scene1_char`, `scene1_bg`, `scene2_char`, `scene2_bg`
3. **By Workflow**: `img_*` for image gen, `vid_*` for video

### Reusability

- Keep prompts **atomic** (small, single-purpose)
- Combine them differently for different needs
- Use libber for **variations** (same structure, different values)
- Use raw for **constants** (quality settings, style tags)

### Performance

- Start with **raw** type unless you need dynamic substitution
- Use **categories** to filter/organize large collections
- Keep prompt values **concise** but descriptive

---

## Troubleshooting

### "Key required" error
- You must provide a unique key for every prompt
- Keys cannot be empty

### "Key already exists" error
- Each key must be unique in the collection
- Edit the existing prompt or use a different key

### Libber input is disabled
- Check that Type is set to `🔄 libber`
- If Type is `📝 raw`, the Libber field is not used

### Prompts not composing correctly
- Check that your composition_json keys match the prompt keys exactly
- Keys are case-sensitive: `char1` ≠ `Char1`
- Missing keys are silently skipped (check the info output)

### Changes not saving
- Click **✓ Apply Changes** button after editing
- The collection_json widget updates automatically when you apply

---

## Migration from Legacy Prompts

If you have old scenes with legacy prompts (girl_pos, male_pos, etc.):

1. Load the scene with SceneSelect
2. Connect to ScenePromptManager
3. The prompts are **auto-migrated** to the new format
4. All legacy prompts become `type=raw`
5. Edit and organize them as needed
6. Save the scene

The original values are preserved in a backup field.

---

## Next Steps

- Create your first prompt collection
- Experiment with different compositions
- Try libber templates for dynamic content
- Share composition maps across projects

**Happy prompting!** 🎨
