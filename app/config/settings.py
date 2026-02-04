import yaml
import os
import json
from pathlib import Path
from enum import Enum
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Dict, Any, Optional, Set, Union

class AuthMode(str, Enum):
    NONE = "NONE"
    KEY = "KEY"
    JWT = "JWT"

class ModelConfig(BaseSettings):
    name: str
    preload: bool = True
    device: Optional[str] = None

class Settings(BaseSettings):
    app_name: str = "Embedding Server"
    
    # Auth Settings
    auth_mode: AuthMode = AuthMode.NONE
    api_key: str = "changeme"
    
    # JWT Settings
    jwt_secret: str = "please_change_this_secret_in_production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    registered_client_ids: Union[Set[str], str] = {"default_client"}
    
    @field_validator("registered_client_ids", mode="before")
    @classmethod
    def parse_registered_client_ids(cls, v: Any) -> Set[str]:
        if isinstance(v, str):
            # Check if it looks like a list/json but isn't quite valid, or just assume it is if starts with [
            if v.strip().startswith("["):
                 try:
                    parsed = json.loads(v)
                    if isinstance(parsed, list):
                        return set(parsed)
                 except json.JSONDecodeError:
                    pass # Fall through to comma split
            
            # Fallback to comma-separated (e.g. "a,b")
            return {x.strip() for x in v.split(",") if x.strip()}
        if isinstance(v, list):
            return set(v)
        if isinstance(v, set):
            return v
        return set()

    model_config_path: str = "models.yaml"
    
    # Cache Settings
    redis_url: Optional[str] = None
    enable_cache: bool = True
    cache_ttl: int = 3600
    
    # Concurrency / Backpressure
    max_inflight_requests: int = 100
    
    model_config = SettingsConfigDict(env_file=os.getenv("ENV_FILE", ".env"), protected_namespaces=('settings_',))

settings = Settings()

def load_model_config() -> Dict[str, Any]:
    path = Path(settings.model_config_path)
    if not path.exists():
        return {}
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config.get("models", {})

model_config = load_model_config()
model_manager_config = model_config # Alias for clarity in other files
