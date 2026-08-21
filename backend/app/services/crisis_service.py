"""Crisis detection service."""
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid
import re

from ..core.database import Database
from ..models.crisis_detection import CrisisDetectionResponse


class CrisisDetectionService:
    """Service for detecting crisis indicators in text."""
    
    # Keywords for crisis detection
    CRISIS_KEYWORDS = [
        'suicide', 'kill myself', 'end my life', 'want to die', 
        'hurt myself', 'self-harm', 'overdose', 'no reason to live'
    ]
    
    CONCERN_KEYWORDS = [
        'hopeless', 'worthless', 'can\'t go on', 'give up', 
        'nobody cares', 'burden', 'alone', 'depressed', 'anxious',
        'panic', 'scared', 'terrified', 'overwhelmed'
    ]
    
    POSITIVE_KEYWORDS = [
        'better', 'hopeful', 'grateful', 'improving', 'happy',
        'peaceful', 'calm', 'relaxed', 'excited', 'proud'
    ]
    
    async def detect_crisis(
        self, 
        text: str, 
        user_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze text for crisis indicators."""
        text_lower = text.lower()
        
        # Calculate risk score
        risk_score = 0.1
        flags = []
        
        # Check for crisis keywords
        for keyword in self.CRISIS_KEYWORDS:
            if keyword in text_lower:
                risk_score = max(risk_score, 0.9)
                flags.append('crisis_language_detected')
                break
        
        # Check for concern keywords
        concern_count = sum(1 for kw in self.CONCERN_KEYWORDS if kw in text_lower)
        if concern_count > 0:
            risk_score = max(risk_score, 0.4 + (concern_count * 0.1))
            flags.append('concerning_language')
        
        # Check for positive keywords
        positive_count = sum(1 for kw in self.POSITIVE_KEYWORDS if kw in text_lower)
        if positive_count > 0:
            risk_score = max(0.1, risk_score - (positive_count * 0.1))
            flags.append('positive_indicators')
        
        # Determine risk level
        if risk_score >= 0.8:
            risk_level = 'immediate'
        elif risk_score >= 0.6:
            risk_level = 'high'
        elif risk_score >= 0.4:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        # Generate recommended actions
        recommended_actions = []
        if risk_level in ['immediate', 'high']:
            recommended_actions = [
                'Consider reaching out to a crisis helpline (988)',
                'Talk to someone you trust',
                'Practice grounding techniques',
                'Remove access to harmful items'
            ]
        elif risk_level == 'medium':
            recommended_actions = [
                'Consider speaking with a mental health professional',
                'Practice self-care activities',
                'Reach out to supportive friends or family'
            ]
        
        # Determine sentiment
        if positive_count > concern_count:
            sentiment = 'positive'
        elif concern_count > positive_count:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'id': str(uuid.uuid4()),
            'user_id': user_id,
            'text': text[:100] + '...' if len(text) > 100 else text,
            'risk_level': risk_level,
            'risk_score': round(risk_score, 2),
            'confidence': 0.85,
            'flags': list(set(flags)),
            'recommended_actions': recommended_actions,
            'sentiment': sentiment,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def store_detection_result(self, db: Database, result: Dict[str, Any]):
        """Store a crisis detection result."""
        db.add_to_collection('crisis_history', result)
    
    async def get_user_history(
        self, 
        db: Database, 
        user_id: str, 
        limit: int = 50, 
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get user's crisis detection history."""
        all_history = db.get_collection('crisis_history')
        user_history = [h for h in all_history if h.get('user_id') == user_id]
        return user_history[offset:offset + limit]
    
    async def get_user_analytics(self, db: Database, user_id: str) -> Dict[str, Any]:
        """Get analytics for a user."""
        history = await self.get_user_history(db, user_id, limit=100)
        
        if not history:
            return {
                'total_analyses': 0,
                'avg_risk_score': 0,
                'risk_distribution': {'low': 0, 'medium': 0, 'high': 0, 'immediate': 0},
                'trend': 'stable'
            }
        
        # Calculate statistics
        total = len(history)
        avg_score = sum(h.get('risk_score', 0) for h in history) / total
        
        risk_dist = {'low': 0, 'medium': 0, 'high': 0, 'immediate': 0}
        for h in history:
            level = h.get('risk_level', 'low')
            risk_dist[level] = risk_dist.get(level, 0) + 1
        
        return {
            'total_analyses': total,
            'avg_risk_score': round(avg_score, 2),
            'risk_distribution': risk_dist,
            'trend': 'improving' if avg_score < 0.4 else 'stable'
        }
    
    async def submit_feedback(
        self, 
        db: Database, 
        user_id: str, 
        feedback_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Submit feedback on crisis detection."""
        feedback = {
            'id': str(uuid.uuid4()),
            'user_id': user_id,
            'feedback': feedback_data,
            'timestamp': datetime.utcnow().isoformat()
        }
        db.add_to_collection('feedback', feedback)
        return feedback
    
    async def handle_high_risk_case(
        self, 
        detection_result: Dict[str, Any], 
        user
    ):
        """Handle high-risk cases (background task)."""
        # In production, this would trigger alerts, notifications, etc.
        print(f"HIGH RISK ALERT for user {user.id}: {detection_result['risk_level']}")
