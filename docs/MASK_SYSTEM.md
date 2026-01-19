# Generic Mask System

## Overview

The ComfyUI-fbTools extension now supports a **generic mask system** that allows users to define arbitrary masks for scenes, rather than being limited to hardcoded "girl", "male", and "combined" masks.

## Key Features

- **Arbitrary Mask Names**: Define masks with any name you choose
- **Flexible Mask Types**: 
  - `TRANSPARENT`: Alpha-based masks (traditional PNG transparency)
  - `COLOR`: Color-keyed masks with RGB values
- **Background Variants**: Each mask can have both background and no-background versions
- **JSON Configuration**: Masks defined in `masks.json` file per scene
- **Backward Compatible**: Legacy mask files still supported

## Mask Definition Structure

Each mask is defined by:

1. **name** (string): Unique identifier for the mask (e.g., "character1", "background", "props")
2. **type** (string): Either "transparent" or "color"
3. **has_background** (boolean): Whether this is the background variant (true) or no-background variant (false)
4. **color** (array): RGB values [R, G, B] (0-255) - only for COLOR type masks, null for TRANSPARENT

## File Format: masks.json

Located in each scene directory (e.g., `output/scenes/my_scene/masks.json`):

```json
{
  "version": 1,
  "masks": [
    {
      "name": "character",
      "type": "transparent",
      "has_background": true,
      "color": null
    },
    {
      "name": "environment",
      "type": "color",
      "has_background": false,
      "color": [255, 0, 0]
    }
  ]
}
```

### Version History

- **Version 1** (Current): Initial generic mask system with name, type, has_background, and color fields

## Mask Image Files

Mask images follow this naming convention:

```
{mask_name}_mask{_bkgd|_no_bkgd}.png
```

Examples:
- `character_mask_bkgd.png` - Character mask with background
- `character_mask_no_bkgd.png` - Character mask without background
- `environment_mask_bkgd.png` - Environment mask with background

### Legacy Naming (Still Supported)

Legacy mask files use the old naming convention:
- `girl_mask_bkgd.png`, `girl_mask_no_bkgd.png`
- `male_mask_bkgd.png`, `male_mask_no_bkgd.png`
- `combined_mask_bkgd.png`, `combined_mask_no_bkgd.png`

If no `masks.json` exists, the system automatically falls back to loading legacy files.

## Using Masks in Nodes

### SceneCreate Node

Create a new scene with custom masks:

1. **mask_definitions** (STRING): JSON array of mask definitions
   ```json
   [
     {"name": "hero", "type": "transparent", "has_background": true, "color": null},
     {"name": "villain", "type": "transparent", "has_background": false, "color": null}
   ]
   ```

2. **mask_images** (IMAGE, optional): Batch of mask images corresponding to definitions
   - First image in batch corresponds to first definition, etc.
   - Batch size should match number of definitions

**Example Workflow:**
```
ImageLoader (hero mask) → 
ImageLoader (villain mask) → 
BatchImages → mask_images input

TextNode (JSON definitions) → mask_definitions input
```

### SceneUpdate Node

Update masks in an existing scene:

1. **update_masks** (BOOLEAN): Set to `True` to enable mask updates

2. **mask_definitions** (STRING): JSON array of masks to add/update

3. **mask_images** (IMAGE, optional): Batch of images for new/updated masks

4. **remove_mask_names** (STRING): Comma-separated list of mask names to remove
   - Example: `"old_mask1, old_mask2"`

**Operations:**
- **Add**: Include new mask definition with corresponding image
- **Update**: Include existing mask name with new definition/image
- **Remove**: List mask name in `remove_mask_names`

### SceneSelect Node

Select a scene and specify which mask to use:

1. **mask_name** (STRING): Name of the mask to load from the scene
   - Leave empty to skip mask selection
   - For legacy scenes: "girl", "male", or "combined"
   - For new scenes: any mask name defined in masks.json

2. **mask_background** (BOOLEAN): 
   - `True`: Use the background variant (_bkgd)
   - `False`: Use the no-background variant (_no_bkgd)

### SceneView Node

Preview a scene with a specific mask:

1. **mask_name** (STRING): Name of the mask to preview
2. **include_mask_bg** (BOOLEAN): Whether to include background in mask

### StoryCreate Node

Create a story with a scene that uses a specific mask:

1. **mask_name** (STRING): Name of mask from the initial scene
2. **mask_background** (BOOLEAN): Background variant selection

## Migration Guide

### Migrating Existing Scenes

Run the migration script to convert legacy scenes:

```bash
# Dry run (see what would change)
python migrate_masks.py --dry-run

# Actual migration
python migrate_masks.py

# Custom directories
python migrate_masks.py --scenes-dir /path/to/scenes --stories-dir /path/to/stories
```

**What the migration script does:**

1. Scans all scene directories
2. Finds legacy mask files (girl_mask_bkgd.png, etc.)
3. Creates `masks.json` with definitions for found masks
4. Backs up original files to `backup_pre_migration/` directory
5. Updates `story.json` files to use `mask_name` instead of `mask_type`

### Manual Migration

If you prefer manual migration:

1. Create `masks.json` in your scene directory:
   ```json
   {
     "version": 1,
     "masks": [
       {
         "name": "girl",
         "type": "transparent",
         "has_background": true,
         "color": null
       },
       {
         "name": "male",
         "type": "transparent",
         "has_background": true,
         "color": null
       }
     ]
   }
   ```

2. (Optional) Rename mask files to new convention:
   - `girl_mask_bkgd.png` → already correct
   - `male_mask_no_bkgd.png` → already correct
   - Or keep legacy names (still supported)

3. Update any story.json files:
   - Change `"mask_type": "combined"` to `"mask_name": "combined"`
   - The system will auto-migrate on load

## Best Practices

### Naming Conventions

- Use descriptive names: `hero`, `background`, `props`, `character1`
- Avoid special characters; stick to alphanumeric and underscores
- Be consistent across scenes in a project

### Mask Types

- **Use TRANSPARENT for most cases**: Traditional alpha channel masks work with all ComfyUI nodes
- **Use COLOR for special workflows**: When you need to identify specific regions by color

### Background Variants

- Always create both variants when possible:
  - `_bkgd`: Full mask including background
  - `_no_bkgd`: Subject only, background removed
- This provides maximum flexibility in workflows

### Organization

For complex scenes with many masks:

```json
{
  "version": 1,
  "masks": [
    {"name": "main_character", "type": "transparent", "has_background": false, "color": null},
    {"name": "secondary_character", "type": "transparent", "has_background": false, "color": null},
    {"name": "environment", "type": "transparent", "has_background": true, "color": null},
    {"name": "props", "type": "transparent", "has_background": false, "color": null}
  ]
}
```

## Troubleshooting

### Scene Not Loading Masks

**Problem**: Scene loads but masks are missing

**Solutions**:
1. Check if `masks.json` exists in scene directory
2. Verify mask image files exist with correct naming
3. Check ComfyUI console for error messages
4. Try legacy naming if migration didn't complete

### Invalid Mask Definition

**Problem**: Error when loading scene with custom masks

**Solutions**:
1. Validate JSON syntax in masks.json
2. Ensure all required fields present (name, type, has_background)
3. For COLOR type, ensure `color` field has [R, G, B] array
4. For TRANSPARENT type, ensure `color` field is `null`

### Migration Script Issues

**Problem**: Migration script reports errors

**Solutions**:
1. Run with `--dry-run` first to preview changes
2. Check file permissions on scene directories
3. Use `--verbose` flag for detailed logging
4. Ensure no other processes are accessing the files

### Backward Compatibility

**Problem**: Old workflows broken after update

**Solutions**:
1. Legacy mask files are automatically loaded if no masks.json exists
2. SceneInStory automatically migrates `mask_type` to `mask_name`
3. All legacy mask names ("girl", "male", "combined") still work
4. Both old and new file naming conventions supported

## API Reference

### MaskType Enum

```python
class MaskType(str, Enum):
    TRANSPARENT = "transparent"  # Alpha-based transparency mask
    COLOR = "color"              # Color-keyed mask
```

### MaskDefinition Class

```python
@dataclass
class MaskDefinition:
    name: str                      # Unique mask identifier
    type: MaskType                 # TRANSPARENT or COLOR
    has_background: bool           # True for _bkgd, False for _no_bkgd
    color: Optional[Tuple[int, int, int]]  # RGB (0-255), None for transparent
    
    def validate(self) -> None:
        """Validates the mask definition"""
        
    def get_filename(self) -> str:
        """Returns the expected filename for this mask"""
        
    def to_dict(self) -> dict:
        """Converts to JSON-serializable dict"""
        
    @classmethod
    def from_dict(cls, data: dict) -> 'MaskDefinition':
        """Creates instance from dict"""
```

### Functions

```python
def load_masks_json(scene_dir: str) -> Optional[Dict[str, MaskDefinition]]:
    """Load masks.json from scene directory"""
    
def save_masks_json(scene_dir: str, masks: Dict[str, MaskDefinition]) -> bool:
    """Save masks to masks.json in scene directory"""
```

## Examples

### Creating a Character Scene

```python
# Define two character masks
mask_defs = [
    {
        "name": "alice",
        "type": "transparent",
        "has_background": false,
        "color": null
    },
    {
        "name": "bob",
        "type": "transparent",
        "has_background": false,
        "color": null
    }
]

# In SceneCreate node:
# - mask_definitions: JSON.stringify(mask_defs)
# - mask_images: Batch of 2 images (alice mask, bob mask)
```

### Using Color Masks for Region Control

```python
mask_defs = [
    {
        "name": "foreground",
        "type": "color",
        "has_background": false,
        "color": [255, 0, 0]  # Red
    },
    {
        "name": "midground",
        "type": "color",
        "has_background": false,
        "color": [0, 255, 0]  # Green
    },
    {
        "name": "background",
        "type": "color",
        "has_background": true,
        "color": [0, 0, 255]  # Blue
    }
]
```

### Updating Scene Masks

```python
# Add a new mask
new_mask = {
    "name": "prop",
    "type": "transparent",
    "has_background": false,
    "color": null
}

# In SceneUpdate node:
# - update_masks: True
# - mask_definitions: JSON.stringify([new_mask])
# - mask_images: Single image for the prop mask
# - remove_mask_names: ""  # Not removing anything
```

### Removing Old Masks

```python
# In SceneUpdate node:
# - update_masks: True
# - mask_definitions: ""  # Not adding anything
# - mask_images: (empty)
# - remove_mask_names: "old_mask1, old_mask2"
```

## Future Enhancements

Potential future additions to the mask system:

- **Mask Layers**: Support for multiple layered masks per scene
- **Procedural Masks**: Generate masks from prompts or other nodes
- **Mask Presets**: Library of common mask configurations
- **Mask Validation UI**: Visual editor for mask definitions
- **Advanced Mask Types**: Gradient masks, pattern masks, etc.

## Support

For issues, questions, or feature requests related to the mask system:

1. Check the [DEBUGGING.md](DEBUGGING.md) guide
2. Review ComfyUI console logs for error messages
3. Verify your masks.json file against the schema
4. Test with legacy mask files to isolate issues

## Changelog

### Version 1.0.0 (January 2026)
- Initial release of generic mask system
- MaskType enum (TRANSPARENT, COLOR)
- MaskDefinition dataclass with validation
- masks.json file format (version 1)
- SceneCreate/SceneUpdate/SceneSelect/SceneView node updates
- Migration script for legacy scenes
- Backward compatibility with legacy mask system
