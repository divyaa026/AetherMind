"""
MindGuard Backend API

FastAPI application for mental health crisis detection system.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import logging
import asyncio
from datetime import datetime, timedelta
import json
import uuid

# Import our modules
from .core.config import settings
from .core.security import get_current_user, create_access_token
from .core.database import get_db, Database
from .models.crisis_detection import CrisisDetectionRequest, CrisisDetectionResponse
from .models.user import User, UserCreate, UserLogin
from .services.crisis_service import CrisisDetectionService
from .services.user_service import UserService
from .services.notification_service import NotificationService
from .api.v1.crisis import router as crisis_router
from .api.v1.users import router as users_router
from .api.v1.clinical import router as clinical_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MindGuard API",
    description="Mental Health Crisis Detection System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Include routers
app.include_router(crisis_router, prefix="/api/v1/crisis", tags=["crisis"])
app.include_router(users_router, prefix="/api/v1/users", tags=["users"])
app.include_router(clinical_router, prefix="/api/v1/clinical", tags=["clinical"])


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting MindGuard API...")
    
    # Initialize database
    await Database.connect()
    
    # Initialize services
    app.state.crisis_service = CrisisDetectionService()
    app.state.user_service = UserService()
    app.state.notification_service = NotificationService()
    
    logger.info("MindGuard API started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down MindGuard API...")
    await Database.disconnect()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "MindGuard Mental Health Crisis Detection System",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "database": "connected",
            "ml_models": "loaded",
            "notification_service": "active"
        }
    }


@app.post("/api/v1/auth/register", response_model=Dict[str, Any])
async def register_user(user_data: UserCreate, db: Database = Depends(get_db)):
    """Register a new user"""
    try:
        user_service = UserService()
        user = await user_service.create_user(db, user_data)
        
        # Create access token
        access_token = create_access_token(data={"sub": user.email})
        
        return {
            "message": "User registered successfully",
            "user": {
                "id": user.id,
                "email": user.email,
                "role": user.role
            },
            "access_token": access_token,
            "token_type": "bearer"
        }
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/v1/auth/login", response_model=Dict[str, Any])
async def login_user(user_data: UserLogin, db: Database = Depends(get_db)):
    """Login user"""
    try:
        user_service = UserService()
        user = await user_service.authenticate_user(db, user_data.email, user_data.password)
        
        if not user:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Create access token
        access_token = create_access_token(data={"sub": user.email})
        
        return {
            "message": "Login successful",
            "user": {
                "id": user.id,
                "email": user.email,
                "role": user.role
            },
            "access_token": access_token,
            "token_type": "bearer"
        }
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=401, detail=str(e))


@app.post("/api/v1/detect-crisis", response_model=CrisisDetectionResponse)
async def detect_crisis(
    request: CrisisDetectionRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Database = Depends(get_db)
):
    """Detect crisis in text input"""
    try:
        crisis_service = CrisisDetectionService()
        
        # Perform crisis detection
        detection_result = await crisis_service.detect_crisis(
            text=request.text,
            user_id=current_user.id,
            context=request.context
        )
        
        # Store detection result
        await crisis_service.store_detection_result(db, detection_result)
        
        # Handle high-risk cases
        if detection_result["risk_level"] in ["high", "immediate"]:
            background_tasks.add_task(
                crisis_service.handle_high_risk_case,
                detection_result,
                current_user
            )
        
        return CrisisDetectionResponse(**detection_result)
        
    except Exception as e:
        logger.error(f"Crisis detection error: {e}")
        raise HTTPException(status_code=500, detail="Crisis detection failed")


@app.get("/api/v1/crisis-history", response_model=List[Dict[str, Any]])
async def get_crisis_history(
    current_user: User = Depends(get_current_user),
    db: Database = Depends(get_db),
    limit: int = 50,
    offset: int = 0
):
    """Get user's crisis detection history"""
    try:
        crisis_service = CrisisDetectionService()
        history = await crisis_service.get_user_history(
            db, current_user.id, limit, offset
        )
        return history
    except Exception as e:
        logger.error(f"Error fetching crisis history: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch history")


@app.post("/api/v1/emergency-contact")
async def add_emergency_contact(
    contact_data: Dict[str, Any],
    current_user: User = Depends(get_current_user),
    db: Database = Depends(get_db)
):
    """Add emergency contact for user"""
    try:
        user_service = UserService()
        contact = await user_service.add_emergency_contact(
            db, current_user.id, contact_data
        )
        return {"message": "Emergency contact added successfully", "contact": contact}
    except Exception as e:
        logger.error(f"Error adding emergency contact: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v1/analytics/overview")
async def get_analytics_overview(
    current_user: User = Depends(get_current_user),
    db: Database = Depends(get_db)
):
    """Get analytics overview for user"""
    try:
        crisis_service = CrisisDetectionService()
        analytics = await crisis_service.get_user_analytics(db, current_user.id)
        return analytics
    except Exception as e:
        logger.error(f"Error fetching analytics: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch analytics")


@app.post("/api/v1/feedback")
async def submit_feedback(
    feedback_data: Dict[str, Any],
    current_user: User = Depends(get_current_user),
    db: Database = Depends(get_db)
):
    """Submit feedback about crisis detection"""
    try:
        crisis_service = CrisisDetectionService()
        feedback = await crisis_service.submit_feedback(
            db, current_user.id, feedback_data
        )
        return {"message": "Feedback submitted successfully", "feedback": feedback}
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# WebSocket endpoint for real-time crisis monitoring
@app.websocket("/ws/crisis-monitor")
async def crisis_monitor_websocket(websocket, token: str):
    """WebSocket endpoint for real-time crisis monitoring"""
    try:
        # Validate token
        user = await get_current_user(token)
        
        # Add to monitoring connections
        app.state.notification_service.add_monitoring_connection(
            user.id, websocket
        )
        
        try:
            while True:
                # Keep connection alive
                await websocket.receive_text()
        except:
            # Remove from monitoring connections
            app.state.notification_service.remove_monitoring_connection(user.id)
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return {
        "error": exc.detail,
        "status_code": exc.status_code,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return {
        "error": "Internal server error",
        "status_code": 500,
        "timestamp": datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
