"""Database connection and utilities."""
from typing import Dict, List, Any
import json
from pathlib import Path

# In-memory storage for demo
_storage: Dict[str, List[Any]] = {
    "users": [],
    "checkins": [],
    "crisis_history": [],
    "emergency_contacts": [],
    "feedback": []
}


class Database:
    """Simple in-memory database for demo purposes."""
    
    _connected = False
    
    @classmethod
    async def connect(cls):
        """Connect to the database."""
        cls._connected = True
        # Load from file if exists
        storage_path = Path("data/storage.json")
        if storage_path.exists():
            global _storage
            with open(storage_path) as f:
                _storage.update(json.load(f))
    
    @classmethod
    async def disconnect(cls):
        """Disconnect from the database."""
        cls._connected = False
        # Save to file
        storage_path = Path("data/storage.json")
        storage_path.parent.mkdir(exist_ok=True)
        with open(storage_path, "w") as f:
            json.dump(_storage, f)
    
    @classmethod
    def get_collection(cls, name: str) -> List[Any]:
        """Get a collection by name."""
        return _storage.get(name, [])
    
    @classmethod
    def add_to_collection(cls, name: str, item: Any):
        """Add an item to a collection."""
        if name not in _storage:
            _storage[name] = []
        _storage[name].append(item)
    
    @classmethod
    def find_in_collection(cls, name: str, key: str, value: Any) -> Any:
        """Find an item in a collection."""
        for item in _storage.get(name, []):
            if isinstance(item, dict) and item.get(key) == value:
                return item
        return None


async def get_db() -> Database:
    """Get database instance."""
    return Database()
