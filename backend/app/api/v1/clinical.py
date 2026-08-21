"""Clinical API routes."""
from fastapi import APIRouter, Depends
from typing import Dict, Any

from ...core.database import get_db, Database
from ...core.security import get_current_user
from ...models.user import User

router = APIRouter()


@router.get("/status")
async def get_clinical_status():
    """Get clinical integration status."""
    return {
        "status": "available",
        "providers_connected": 0,
        "last_sync": None
    }
