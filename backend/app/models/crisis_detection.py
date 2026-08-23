"""Crisis detection models."""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class CrisisDetectionRequest(BaseModel):
    """Request model for crisis detection."""
    text: str = Field(..., description="Text to analyze for crisis indicators")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")


class CrisisDetectionResponse(BaseModel):
    """Response model for crisis detection."""
    risk_level: str = Field(..., description="Risk level: low, medium, high, immediate")
    risk_score: float = Field(..., ge=0, le=1, description="Risk score from 0 to 1")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in the assessment")
    flags: List[str] = Field(default_factory=list, description="Detected risk flags")
    recommended_actions: List[str] = Field(default_factory=list, description="Recommended actions")
    sentiment: str = Field(default="neutral", description="Overall sentiment")
    timestamp: str = Field(..., description="Timestamp of the analysis")
