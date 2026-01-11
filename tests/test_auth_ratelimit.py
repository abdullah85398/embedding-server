from unittest.mock import patch
from app.middleware.rate_limit import RateLimitMiddleware
from collections import defaultdict, deque

def test_auth_failure(client, override_settings):
    # We need to override auth mode to KEY or JWT to trigger 403
    # because default is now NONE (open)
    from app.config.settings import AuthMode
    with override_settings(auth_mode=AuthMode.KEY):
        # No headers
        response = client.post("/embed", json={"model": "mini", "input": "test"})
        assert response.status_code == 403 

        # Wrong key
        response = client.post("/embed", json={"model": "mini", "input": "test"}, headers={"X-API-Key": "wrong"})
        assert response.status_code == 403

def test_rate_limit_allowance(client):
    """
    Verifies that requests within the limit are allowed.
    """
    # Normal requests should pass
    for _ in range(5):
        response = client.get("/health")
        assert response.status_code == 200

def test_rate_limit_exceeded(client):
    """
    Verifies that requests exceeding the limit are blocked with 429.
    We patch the max_requests on the middleware instance directly.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    
    # Create a fresh app with a tight limit
    test_app = FastAPI()
    test_app.add_middleware(RateLimitMiddleware, max_requests=2, window_seconds=60)
    
    # Define a test endpoint that is NOT excluded (unlike /health)
    @test_app.get("/test-limit")
    def test_limit():
        return {"status": "ok"}
    
    test_client = TestClient(test_app)
    
    # 1. First request (OK)
    response = test_client.get("/test-limit")
    assert response.status_code == 200
    
    # 2. Second request (OK)
    response = test_client.get("/test-limit")
    assert response.status_code == 200
    
    # 3. Third request (Blocked)
    response = test_client.get("/test-limit")
    assert response.status_code == 429
    assert response.text == "Too Many Requests"
