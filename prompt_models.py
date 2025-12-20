"""
Data models for prompt management system.

This module contains the PromptMetadata and PromptCollection classes
extracted to a separate file for easier testing and reusability.
"""

from typing import Optional, List
from pydantic import BaseModel, ConfigDict


class PromptMetadata(BaseModel):
    """Metadata for a single prompt entry with processing support."""
    value: str
    category: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    
    # Processing configuration
    processing_type: str = "raw"  # "raw" or "libber"
    libber_name: Optional[str] = None  # which libber to use for substitution
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


class PromptCollection(BaseModel):
    """
    Version 2 prompt system supporting unlimited named prompts.
    Maintains backward compatibility with v1 format through v1_backup field.
    """
    version: int = 2
    v1_backup: Optional[dict] = None
    prompts: dict[str, PromptMetadata] = {}
    
    def get_prompt_value(self, key: str) -> Optional[str]:
        """Get prompt value by key, returns None if not found."""
        prompt = self.prompts.get(key)
        return prompt.value if prompt else None
    
    def add_prompt(self, key: str, value: str, category: Optional[str] = None, 
                   description: Optional[str] = None, tags: Optional[List[str]] = None,
                   processing_type: str = "raw", libber_name: Optional[str] = None):
        """Add or update a prompt entry."""
        self.prompts[key] = PromptMetadata(
            value=value,
            category=category,
            description=description,
            tags=tags,
            processing_type=processing_type,
            libber_name=libber_name
        )
    
    def remove_prompt(self, key: str) -> bool:
        """Remove a prompt entry. Returns True if removed, False if not found."""
        if key in self.prompts:
            del self.prompts[key]
            return True
        return False
    
    def list_prompt_names(self) -> List[str]:
        """Return sorted list of all prompt keys."""
        return sorted(self.prompts.keys())
    
    @classmethod
    def from_legacy_dict(cls, legacy_data: dict) -> "PromptCollection":
        """
        Create PromptCollection from v1 format, preserving original data in v1_backup.
        
        Legacy format: {"girl_pos": "...", "male_pos": "...", ...}
        V2 format: {"version": 2, "v1_backup": {...}, "prompts": {...}}
        
        All legacy prompts are migrated as "raw" processing type.
        Output slot assignment happens at node level, not in metadata.
        """
        prompts = {}
        
        for key, value in legacy_data.items():
            if isinstance(value, str) and value:  # Only migrate non-empty strings
                prompts[key] = PromptMetadata(
                    value=value,
                    processing_type="raw"
                )
        
        return cls(
            version=2,
            v1_backup=legacy_data.copy(),
            prompts=prompts
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {
            "version": self.version,
            "prompts": {}
        }
        
        if self.v1_backup is not None:
            result["v1_backup"] = self.v1_backup
        
        for key, metadata in self.prompts.items():
            result["prompts"][key] = {
                "value": metadata.value,
                "processing_type": metadata.processing_type,
            }
            if metadata.category:
                result["prompts"][key]["category"] = metadata.category
            if metadata.description:
                result["prompts"][key]["description"] = metadata.description
            if metadata.tags:
                result["prompts"][key]["tags"] = metadata.tags
            if metadata.libber_name:
                result["prompts"][key]["libber_name"] = metadata.libber_name
        
        return result
    
    @classmethod
    def from_dict(cls, data: dict) -> "PromptCollection":
        """Load PromptCollection from dictionary."""
        version = data.get("version", 1)
        
        # Handle v1 format - auto-migrate
        if version == 1 or "prompts" not in data:
            return cls.from_legacy_dict(data)
        
        # Handle v2 format
        prompts = {}
        for key, prompt_data in data.get("prompts", {}).items():
            if isinstance(prompt_data, str):
                prompts[key] = PromptMetadata(value=prompt_data)
            elif isinstance(prompt_data, dict):
                prompts[key] = PromptMetadata(
                    value=prompt_data.get("value", ""),
                    category=prompt_data.get("category"),
                    description=prompt_data.get("description"),
                    tags=prompt_data.get("tags"),
                    processing_type=prompt_data.get("processing_type", "raw"),
                    libber_name=prompt_data.get("libber_name")
                )
        
        return cls(
            version=2,
            v1_backup=data.get("v1_backup"),
            prompts=prompts
        )
    
    def get_prompt_metadata(self, key: str) -> Optional[PromptMetadata]:
        """Get full metadata for a prompt by key."""
        return self.prompts.get(key)
    
    def get_prompts_by_category(self, category: str) -> dict[str, PromptMetadata]:
        """Get all prompts in a specific category."""
        return {
            key: metadata for key, metadata in self.prompts.items()
            if metadata.category == category
        }
    
    def compose_prompts(self, composition_map: dict, libber_manager=None) -> dict[str, str]:
        """
        Compose output prompts based on a composition map.
        
        Args:
            composition_map: Dict mapping output names to lists of prompt keys
                Example: {
                    "qwen_main": ["girl_pos", "male_pos", "quality"],
                    "video_high": ["wan_prompt", "style"]
                }
            libber_manager: Optional LibberStateManager instance for processing libber prompts
            
        Returns:
            Dict mapping output names to composed prompt strings
        """
        results = {}
        
        for output_name, prompt_keys in composition_map.items():
            parts = []
            
            for key in prompt_keys:
                if key not in self.prompts:
                    continue
                    
                metadata = self.prompts[key]
                value = metadata.value
                
                # Process libber substitution if needed
                if metadata.processing_type == "libber" and metadata.libber_name and libber_manager:
                    libber = libber_manager.get_libber(metadata.libber_name)
                    if libber:
                        value = libber.substitute(value)
                
                if value:
                    parts.append(value)
            
            results[output_name] = " ".join(parts).strip()
        
        return results
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
