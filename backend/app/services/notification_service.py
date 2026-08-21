"""Notification service."""
from typing import Dict, Any, List


class NotificationService:
    """Service for sending notifications."""
    
    def __init__(self):
        self._monitoring_connections: Dict[str, Any] = {}
    
    def add_monitoring_connection(self, user_id: str, websocket):
        """Add a websocket connection for monitoring."""
        self._monitoring_connections[user_id] = websocket
    
    def remove_monitoring_connection(self, user_id: str):
        """Remove a websocket connection."""
        self._monitoring_connections.pop(user_id, None)
    
    async def send_notification(self, user_id: str, message: Dict[str, Any]):
        """Send a notification to a user."""
        ws = self._monitoring_connections.get(user_id)
        if ws:
            try:
                await ws.send_json(message)
            except Exception as e:
                print(f"Failed to send notification: {e}")
