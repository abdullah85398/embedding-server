import hashlib
import json
import logging
from typing import List, Optional
import redis
from app.config.settings import settings

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self):
        self.enabled = settings.enable_cache
        self.redis_client = None
        self.local_cache = {}  # Simple dictionary for in-memory fallback
        
        if self.enabled and settings.redis_url:
            try:
                self.redis_client = redis.from_url(settings.redis_url, decode_responses=True)
                # Test connection
                self.redis_client.ping()
                logger.info("Connected to Redis cache")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis, falling back to in-memory cache: {e}")
                self.redis_client = None

    def _generate_key(self, model: str, text: str) -> str:
        # Create a deterministic hash of model + text
        content = f"{model}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get_embedding(self, model: str, text: str) -> Optional[List[float]]:
        if not self.enabled:
            return None

        key = self._generate_key(model, text)
        
        # Try Redis first
        if self.redis_client:
            try:
                data = self.redis_client.get(key)
                if data:
                    return json.loads(data)
            except Exception as e:
                logger.error(f"Redis get error: {e}")
        
        # Try Local Cache
        if key in self.local_cache:
            return self.local_cache[key]
            
        return None

    def set_embedding(self, model: str, text: str, vector: List[float]):
        if not self.enabled:
            return

        key = self._generate_key(model, text)
        
        # Save to Redis
        if self.redis_client:
            try:
                self.redis_client.setex(key, settings.cache_ttl, json.dumps(vector))
            except Exception as e:
                logger.error(f"Redis set error: {e}")
        
        # Save to Local Cache (Limit size to avoid OOM in production - rudimentary LRU could be added here)
        if len(self.local_cache) > 10000:
            self.local_cache.clear() # Simple wipe for MVP
        self.local_cache[key] = vector

cache_manager = CacheManager()
