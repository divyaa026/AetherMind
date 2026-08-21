"""Crisis detection API routes."""
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, List

from ...core.database import get_db, Database
from ...core.security import get_current_user
from ...models.user import User

router = APIRouter()


@router.get("/resources")
async def get_crisis_resources():
    """Get crisis resources and hotlines."""
    return {
        "resources": [
            {
                "name": "988 Suicide & Crisis Lifeline",
                "number": "988",
                "description": "24/7 free, confidential support",
                "available": "24/7"
            },
            {
                "name": "Crisis Text Line",
                "number": "Text HOME to 741741",
                "description": "Free crisis counseling via text",
                "available": "24/7"
            },
            {
                "name": "SAMHSA National Helpline",
                "number": "1-800-662-4357",
                "description": "Treatment referral and information",
                "available": "24/7"
            }
        ]
    }
