"""User API routes."""
from fastapi import APIRouter, Depends
from typing import Dict, Any

from ...core.database import get_db, Database
from ...core.security import get_current_user
from ...models.user import User

router = APIRouter()


@router.get("/me")
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information."""
    return {
        "id": current_user.id,
        "email": current_user.email,
        "role": current_user.role
    }


@router.get("/preferences")
async def get_user_preferences(current_user: User = Depends(get_current_user)):
    """Get user preferences."""
    return {
        "theme": "system",
        "notifications": True,
        "privacy_level": "high"
    }
