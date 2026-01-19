#!/usr/bin/env python3
"""
Migration script to convert scenes from legacy hardcoded masks to new generic mask system.

This script:
1. Scans all scene directories
2. For each scene with legacy mask files but no masks.json:
   - Creates masks.json with definitions for found legacy masks
   - Renames mask files to new naming convention
3. Updates story.json files to use mask_name instead of mask_type
4. Creates backup of all modified files

Usage:
    python migrate_masks.py [--scenes-dir PATH] [--stories-dir PATH] [--dry-run]
"""

import argparse
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Legacy mask file mappings
LEGACY_MASK_FILES = {
    'girl': 'girl_mask_bkgd.png',
    'male': 'male_mask_bkgd.png',
    'combined': 'combined_mask_bkgd.png',
    'girl_no_bg': 'girl_mask_no_bkgd.png',
    'male_no_bg': 'male_mask_no_bkgd.png',
    'combined_no_bg': 'combined_mask_no_bkgd.png',
}

# New mask system structure
def create_mask_definition(name: str, has_background: bool, mask_type: str = "transparent") -> dict:
    """Create a mask definition dictionary"""
    return {
        "name": name,
        "type": mask_type,
        "has_background": has_background,
        "color": None if mask_type == "transparent" else [255, 255, 255]
    }


def get_default_scenes_dir() -> str:
    """Get default scenes directory"""
    output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "output")
    return os.path.join(output_dir, "scenes")


def get_default_stories_dir() -> str:
    """Get default stories directory"""
    output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "output")
    return os.path.join(output_dir, "stories")


def find_legacy_masks(scene_dir: Path) -> List[Tuple[str, str, bool]]:
    """Find legacy mask files in scene directory.
    
    Returns list of (mask_name, filename, has_background) tuples
    """
    found_masks = []
    
    # Check for legacy mask files
    for name, filename in LEGACY_MASK_FILES.items():
        file_path = scene_dir / filename
        if file_path.exists():
            # Determine if this is a background or no-background variant
            if name.endswith('_no_bg'):
                base_name = name.replace('_no_bg', '')
                has_background = False
            else:
                base_name = name
                has_background = True
            
            found_masks.append((base_name, filename, has_background))
            logger.debug(f"Found legacy mask: {filename} -> {base_name} (bg={has_background})")
    
    return found_masks


def create_masks_json(scene_dir: Path, legacy_masks: List[Tuple[str, str, bool]], dry_run: bool = False) -> bool:
    """Create masks.json from legacy mask files.
    
    Returns True if masks.json was created/would be created
    """
    masks_json_path = scene_dir / "masks.json"
    
    # Skip if masks.json already exists
    if masks_json_path.exists():
        logger.info(f"Skipping {scene_dir.name}: masks.json already exists")
        return False
    
    # Group masks by name (combine background and no-background variants)
    mask_defs = {}
    for base_name, filename, has_background in legacy_masks:
        if base_name not in mask_defs:
            # Create definition for first variant found
            mask_defs[base_name] = create_mask_definition(base_name, has_background)
    
    if not mask_defs:
        logger.debug(f"No legacy masks found in {scene_dir.name}")
        return False
    
    # Create masks.json structure
    masks_data = {
        "version": 1,
        "masks": list(mask_defs.values())
    }
    
    if dry_run:
        logger.info(f"[DRY RUN] Would create {masks_json_path}")
        logger.info(f"[DRY RUN] Content: {json.dumps(masks_data, indent=2)}")
        return True
    
    # Write masks.json
    try:
        with open(masks_json_path, 'w') as f:
            json.dump(masks_data, f, indent=2)
        logger.info(f"Created {masks_json_path} with {len(mask_defs)} mask definitions")
        return True
    except Exception as e:
        logger.error(f"Failed to create masks.json in {scene_dir}: {e}")
        return False


def rename_legacy_mask_files(scene_dir: Path, legacy_masks: List[Tuple[str, str, bool]], dry_run: bool = False):
    """Rename legacy mask files to new naming convention.
    
    New convention: {name}_mask{_bkgd|_no_bkgd}.png
    """
    for base_name, old_filename, has_background in legacy_masks:
        old_path = scene_dir / old_filename
        
        # Generate new filename
        suffix = "_bkgd" if has_background else "_no_bkgd"
        new_filename = f"{base_name}_mask{suffix}.png"
        new_path = scene_dir / new_filename
        
        # Skip if already using new naming
        if old_filename == new_filename:
            logger.debug(f"Skipping {old_filename}: already uses new naming")
            continue
        
        # Skip if new file already exists
        if new_path.exists():
            logger.warning(f"Skipping rename of {old_filename}: {new_filename} already exists")
            continue
        
        if dry_run:
            logger.info(f"[DRY RUN] Would rename {old_filename} -> {new_filename}")
        else:
            try:
                shutil.copy2(old_path, new_path)
                logger.info(f"Copied {old_filename} -> {new_filename}")
                # Keep original file for now (can be manually deleted after verification)
                logger.debug(f"Original file preserved: {old_filename}")
            except Exception as e:
                logger.error(f"Failed to copy {old_filename} to {new_filename}: {e}")


def migrate_scene(scene_dir: Path, dry_run: bool = False) -> bool:
    """Migrate a single scene to new mask system.
    
    Returns True if migration was performed/would be performed
    """
    logger.info(f"Processing scene: {scene_dir.name}")
    
    # Check if masks.json already exists
    masks_json_path = scene_dir / "masks.json"
    if masks_json_path.exists():
        logger.info(f"Scene {scene_dir.name} already migrated (masks.json exists)")
        return False
    
    # Find legacy mask files
    legacy_masks = find_legacy_masks(scene_dir)
    if not legacy_masks:
        logger.debug(f"No legacy masks found in {scene_dir.name}")
        return False
    
    # Create backup directory
    backup_dir = scene_dir / "backup_pre_migration"
    if not dry_run and not backup_dir.exists():
        backup_dir.mkdir(parents=True, exist_ok=True)
        # Backup legacy mask files
        for _, filename, _ in legacy_masks:
            src = scene_dir / filename
            dst = backup_dir / filename
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
                logger.info(f"Backed up {filename} to {backup_dir.name}/")
    
    # Create masks.json
    created = create_masks_json(scene_dir, legacy_masks, dry_run)
    
    # Rename mask files (optional - keeping legacy names for compatibility)
    # rename_legacy_mask_files(scene_dir, legacy_masks, dry_run)
    
    return created


def migrate_story_file(story_path: Path, dry_run: bool = False) -> bool:
    """Migrate story.json to use mask_name instead of mask_type.
    
    Returns True if migration was performed/would be performed
    """
    if not story_path.exists():
        return False
    
    logger.info(f"Processing story: {story_path}")
    
    try:
        with open(story_path, 'r') as f:
            story_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read {story_path}: {e}")
        return False
    
    # Check if migration needed
    modified = False
    scenes = story_data.get('scenes', [])
    
    for scene in scenes:
        if 'mask_type' in scene and 'mask_name' not in scene:
            mask_type = scene['mask_type']
            
            # Convert mask_type to mask_name
            if mask_type.endswith('_no_bg'):
                scene['mask_name'] = mask_type.replace('_no_bg', '')
                scene['mask_background'] = False
            else:
                scene['mask_name'] = mask_type
                scene['mask_background'] = scene.get('mask_background', True)
            
            # Keep mask_type for backward compatibility
            logger.info(f"Migrated scene {scene.get('scene_name', 'unknown')}: mask_type={mask_type} -> mask_name={scene['mask_name']}")
            modified = True
    
    if not modified:
        logger.debug(f"No migration needed for {story_path}")
        return False
    
    if dry_run:
        logger.info(f"[DRY RUN] Would update {story_path}")
        return True
    
    # Create backup
    backup_path = story_path.with_suffix('.json.backup')
    if not backup_path.exists():
        shutil.copy2(story_path, backup_path)
        logger.info(f"Created backup: {backup_path}")
    
    # Write updated story
    try:
        with open(story_path, 'w') as f:
            json.dump(story_data, f, indent=2)
        logger.info(f"Updated {story_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to write {story_path}: {e}")
        # Restore from backup
        if backup_path.exists():
            shutil.copy2(backup_path, story_path)
            logger.info(f"Restored from backup after error")
        return False


def migrate_all_scenes(scenes_dir: str, dry_run: bool = False) -> Tuple[int, int]:
    """Migrate all scenes in scenes directory.
    
    Returns (migrated_count, total_count)
    """
    scenes_path = Path(scenes_dir)
    if not scenes_path.exists():
        logger.error(f"Scenes directory not found: {scenes_dir}")
        return 0, 0
    
    logger.info(f"Scanning scenes directory: {scenes_dir}")
    
    scene_dirs = [d for d in scenes_path.iterdir() if d.is_dir()]
    migrated = 0
    
    for scene_dir in scene_dirs:
        if migrate_scene(scene_dir, dry_run):
            migrated += 1
    
    return migrated, len(scene_dirs)


def migrate_all_stories(stories_dir: str, dry_run: bool = False) -> Tuple[int, int]:
    """Migrate all story.json files in stories directory.
    
    Returns (migrated_count, total_count)
    """
    stories_path = Path(stories_dir)
    if not stories_path.exists():
        logger.warning(f"Stories directory not found: {stories_dir}")
        return 0, 0
    
    logger.info(f"Scanning stories directory: {stories_dir}")
    
    story_files = list(stories_path.glob("*/story.json"))
    migrated = 0
    
    for story_file in story_files:
        if migrate_story_file(story_file, dry_run):
            migrated += 1
    
    return migrated, len(story_files)


def main():
    parser = argparse.ArgumentParser(
        description="Migrate ComfyUI scenes from legacy mask system to new generic mask system"
    )
    parser.add_argument(
        '--scenes-dir',
        help='Path to scenes directory (default: auto-detect)',
        default=None
    )
    parser.add_argument(
        '--stories-dir',
        help='Path to stories directory (default: auto-detect)',
        default=None
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Determine directories
    scenes_dir = args.scenes_dir or get_default_scenes_dir()
    stories_dir = args.stories_dir or get_default_stories_dir()
    
    logger.info("=" * 80)
    logger.info("ComfyUI Mask System Migration")
    logger.info("=" * 80)
    logger.info(f"Scenes directory: {scenes_dir}")
    logger.info(f"Stories directory: {stories_dir}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info("=" * 80)
    
    # Migrate scenes
    logger.info("\n--- Migrating Scenes ---")
    migrated_scenes, total_scenes = migrate_all_scenes(scenes_dir, args.dry_run)
    logger.info(f"Migrated {migrated_scenes} of {total_scenes} scenes")
    
    # Migrate stories
    logger.info("\n--- Migrating Stories ---")
    migrated_stories, total_stories = migrate_all_stories(stories_dir, args.dry_run)
    logger.info(f"Migrated {migrated_stories} of {total_stories} stories")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Migration Summary")
    logger.info("=" * 80)
    logger.info(f"Scenes: {migrated_scenes}/{total_scenes} migrated")
    logger.info(f"Stories: {migrated_stories}/{total_stories} migrated")
    
    if args.dry_run:
        logger.info("\nThis was a dry run. No files were modified.")
        logger.info("Run without --dry-run to perform actual migration.")
    else:
        logger.info("\nMigration complete!")
        logger.info("Backups created for all modified files.")
        logger.info("Legacy mask files preserved for compatibility.")
    
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
