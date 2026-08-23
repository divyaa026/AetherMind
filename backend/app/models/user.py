"""User models."""
from pydantic import BaseModel, EmailStr
from typing import Optional


class User(BaseModel):
    """User model."""
    id: str
    email: str
    role: str = "user"
    hashed_password: Optional[str] = None


class UserCreate(BaseModel):
    """User creation model."""
    email: EmailStr
    password: str
    name: Optional[str] = None


class UserLogin(BaseModel):
    """User login model."""
    email: EmailStr
    password: str
