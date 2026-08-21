"""User service."""
from typing import Dict, Any, Optional
from datetime import datetime
import uuid

from ..core.database import Database
from ..core.security import get_password_hash, verify_password
from ..models.user import User, UserCreate


class UserService:
    """Service for user management."""
    
    async def create_user(self, db: Database, user_data: UserCreate) -> User:
        """Create a new user."""
        # Check if user exists
        existing = db.find_in_collection('users', 'email', user_data.email)
        if existing:
            raise ValueError("User with this email already exists")
        
        # Create user
        user_dict = {
            'id': str(uuid.uuid4()),
            'email': user_data.email,
            'hashed_password': get_password_hash(user_data.password),
            'role': 'user',
            'created_at': datetime.utcnow().isoformat()
        }
        
        db.add_to_collection('users', user_dict)
        
        return User(id=user_dict['id'], email=user_dict['email'], role=user_dict['role'])
    
    async def authenticate_user(
        self, 
        db: Database, 
        email: str, 
        password: str
    ) -> Optional[User]:
        """Authenticate a user."""
        user_dict = db.find_in_collection('users', 'email', email)
        
        if not user_dict:
            return None
        
        if not verify_password(password, user_dict.get('hashed_password', '')):
            return None
        
        return User(
            id=user_dict['id'], 
            email=user_dict['email'], 
            role=user_dict.get('role', 'user')
        )
    
    async def add_emergency_contact(
        self, 
        db: Database, 
        user_id: str, 
        contact_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Add an emergency contact for a user."""
        contact = {
            'id': str(uuid.uuid4()),
            'user_id': user_id,
            'name': contact_data.get('name'),
            'phone': contact_data.get('phone'),
            'relationship': contact_data.get('relationship'),
            'created_at': datetime.utcnow().isoformat()
        }
        
        db.add_to_collection('emergency_contacts', contact)
        return contact
