"""Prompt collection session REST routes (/fbtools/prompts/*) and PromptCollectionStateManager.

Moved out of extension.py (pure code motion)."""
from __future__ import annotations

from typing import Optional
from ..prompt_models import PromptCollection
from datetime import datetime, timedelta
import time
from aiohttp import web
from .shared import routes
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class PromptCollectionStateManager:
    """
    Manages server-side PromptCollection instances for REST API operations.
    Sessions expire after 30 minutes of inactivity.
    """
    _instance = None
    
    def __init__(self):
        self.sessions = {}  # session_id -> {"collection": PromptCollection, "last_access": datetime}
        self.ttl_minutes = 30
    
    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def cleanup_expired(self):
        """Remove sessions older than TTL."""
        now = datetime.now()
        expired = [
            sid for sid, data in self.sessions.items()
            if now - data["last_access"] > timedelta(minutes=self.ttl_minutes)
        ]
        for sid in expired:
            del self.sessions[sid]
            logger.info("PromptCollectionStateManager: Expired session %s", sid)
    
    def create_session(self, session_id: str, collection: PromptCollection):
        """Create or update a session with a PromptCollection."""
        self.cleanup_expired()
        self.sessions[session_id] = {
            "collection": collection,
            "last_access": datetime.now()
        }
        logger.info("PromptCollectionStateManager: Created session %s", session_id)
    
    def get_collection(self, session_id: str) -> Optional[PromptCollection]:
        """Get PromptCollection for a session, updating last access time."""
        self.cleanup_expired()
        if session_id in self.sessions:
            self.sessions[session_id]["last_access"] = datetime.now()
            return self.sessions[session_id]["collection"]
        return None
    
    def update_collection(self, session_id: str, collection: PromptCollection):
        """Update the PromptCollection for a session."""
        if session_id in self.sessions:
            self.sessions[session_id]["collection"] = collection
            self.sessions[session_id]["last_access"] = datetime.now()


# Register REST API endpoints
@routes.post("/fbtools/prompts/create")
async def create_prompt_collection(request):
    """Create a new PromptCollection session."""
    try:
        data = await request.json()
        session_id = data.get("session_id", f"prompt_{int(time.time()*1000)}")
        
        # Create new empty collection or from legacy data
        legacy_data = data.get("legacy_data")
        if legacy_data:
            collection = PromptCollection.from_legacy_dict(legacy_data)
        else:
            collection = PromptCollection()
        
        manager = PromptCollectionStateManager.instance()
        manager.create_session(session_id, collection)
        
        return web.json_response({
            "success": True,
            "session_id": session_id,
            "collection": collection.to_dict()
        })
    except Exception as e:
        return web.json_response({
            "success": False,
            "error": str(e)
        }, status=500)


@routes.post("/fbtools/prompts/add")
async def add_prompt(request):
    """Add or update a prompt in a PromptCollection."""
    try:
        data = await request.json()
        session_id = data.get("session_id")
        prompt_name = data.get("prompt_name")
        prompt_value = data.get("prompt_value")
        category = data.get("category")
        description = data.get("description")
        tags = data.get("tags")
        
        if not session_id or not prompt_name:
            return web.json_response({
                "success": False,
                "error": "session_id and prompt_name required"
            }, status=400)
        
        manager = PromptCollectionStateManager.instance()
        collection = manager.get_collection(session_id)
        
        if not collection:
            return web.json_response({
                "success": False,
                "error": f"Session {session_id} not found"
            }, status=404)
        
        collection.add_prompt(prompt_name, prompt_value, category, description, tags)
        manager.update_collection(session_id, collection)
        
        return web.json_response({
            "success": True,
            "collection": collection.to_dict(),
            "prompt_names": collection.list_prompt_names()
        })
    except Exception as e:
        return web.json_response({
            "success": False,
            "error": str(e)
        }, status=500)


@routes.post("/fbtools/prompts/remove")
async def remove_prompt(request):
    """Remove a prompt from a PromptCollection."""
    try:
        data = await request.json()
        session_id = data.get("session_id")
        prompt_name = data.get("prompt_name")
        
        if not session_id or not prompt_name:
            return web.json_response({
                "success": False,
                "error": "session_id and prompt_name required"
            }, status=400)
        
        manager = PromptCollectionStateManager.instance()
        collection = manager.get_collection(session_id)
        
        if not collection:
            return web.json_response({
                "success": False,
                "error": f"Session {session_id} not found"
            }, status=404)
        
        removed = collection.remove_prompt(prompt_name)
        if removed:
            manager.update_collection(session_id, collection)
        
        return web.json_response({
            "success": True,
            "removed": removed,
            "collection": collection.to_dict(),
            "prompt_names": collection.list_prompt_names()
        })
    except Exception as e:
        return web.json_response({
            "success": False,
            "error": str(e)
        }, status=500)


@routes.get("/fbtools/prompts/list_names")
async def list_prompt_names(request):
    """Get list of all prompt names in a PromptCollection."""
    try:
        session_id = request.query.get("session_id")
        
        if not session_id:
            return web.json_response({
                "success": False,
                "error": "session_id required"
            }, status=400)
        
        manager = PromptCollectionStateManager.instance()
        collection = manager.get_collection(session_id)
        
        if not collection:
            return web.json_response({
                "success": False,
                "error": f"Session {session_id} not found"
            }, status=404)
        
        return web.json_response({
            "success": True,
            "prompt_names": collection.list_prompt_names(),
            "count": len(collection.prompts)
        })
    except Exception as e:
        return web.json_response({
            "success": False,
            "error": str(e)
        }, status=500)
