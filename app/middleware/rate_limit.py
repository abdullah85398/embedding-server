import time
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from collections import defaultdict, deque
from app.config.settings import settings, AuthMode
from app.core.security import decode_access_token

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_requests: int = 60, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        # Simple in-memory rate limiter (IP based)
        # In production with multiple workers, use Redis
        self.request_history = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        # Skip rate limit for health checks
        if request.url.path in ["/health", "/ready", "/metrics"]:
            return await call_next(request)

        # Identify Client (Token > IP)
        client_id = request.client.host
        
        api_key = request.headers.get("X-API-Key")
        auth_header = request.headers.get("Authorization")
        
        # 1. Check Master Key (Highest Priority)
        is_master = False
        if api_key and api_key == settings.api_key:
            is_master = True
        elif auth_header and auth_header.startswith("Bearer "):
             token = auth_header.split(" ")[1]
             if token == settings.api_key:
                 is_master = True
        
        if is_master:
            client_id = "master_key"
        
        # 2. Mode Specific
        elif settings.auth_mode == AuthMode.JWT:
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]
                payload = decode_access_token(token)
                if payload:
                    if "client_id" in payload:
                        client_id = f"client:{payload['client_id']}"
                    elif "sub" in payload:
                        client_id = f"user:{payload['sub']}"

        elif settings.auth_mode == AuthMode.KEY:
             if api_key:
                 client_id = f"apikey:{api_key}"
             # Also check bearer for legacy key support if needed, but strictly:
             elif auth_header and auth_header.startswith("Bearer "):
                 token = auth_header.split(" ")[1]
                 # If it wasn't master key (checked above), maybe it's another key if we supported multiple keys
                 # But for now we only have one api_key.
                 pass

        now = time.time()
        
        history = self.request_history[client_id]
        
        # Remove old requests
        while history and history[0] < now - self.window_seconds:
            history.popleft()
            
        if len(history) >= self.max_requests:
            return Response("Too Many Requests", status_code=429)
            
        history.append(now)
        
        response = await call_next(request)
        return response
