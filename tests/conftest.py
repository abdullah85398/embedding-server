import pytest
import os
from fastapi.testclient import TestClient
from app.config.settings import settings, AuthMode
from main import app

@pytest.fixture(scope="session", autouse=True)
def test_env_setup():
    """
    Sets up the environment variables for testing.
    Autouse ensures this runs for all tests.
    """
    # Store original values
    original_api_key = os.environ.get("API_KEY")
    original_inflight = os.environ.get("MAX_INFLIGHT_REQUESTS")
    original_cache = os.environ.get("ENABLE_CACHE")
    original_auth_mode = os.environ.get("AUTH_MODE")

    # Set test values
    os.environ["API_KEY"] = "test-secret"
    os.environ["MAX_INFLIGHT_REQUESTS"] = "10"
    os.environ["ENABLE_CACHE"] = "false"
    os.environ["AUTH_MODE"] = "NONE"
    
    # Also update the pydantic settings object directly
    settings.api_key = "test-secret"
    settings.max_inflight_requests = 10
    settings.enable_cache = False
    settings.auth_mode = AuthMode.NONE

    yield

    # Restore original values (best effort, though session scope means it ends with the process)
    if original_api_key: os.environ["API_KEY"] = original_api_key
    if original_inflight: os.environ["MAX_INFLIGHT_REQUESTS"] = original_inflight
    if original_cache: os.environ["ENABLE_CACHE"] = original_cache
    if original_auth_mode: os.environ["AUTH_MODE"] = original_auth_mode

@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c

@pytest.fixture
def auth_headers():
    return {"X-API-Key": "test-secret"}

@pytest.fixture
def override_settings():
    """
    Fixture to safely override settings for a single test.
    Usage:
        def test_foo(override_settings):
            with override_settings(auth_mode="JWT"):
                ...
    """
    from contextlib import contextmanager
    
    @contextmanager
    def _override(**kwargs):
        original_values = {}
        for k, v in kwargs.items():
            if hasattr(settings, k):
                original_values[k] = getattr(settings, k)
                setattr(settings, k, v)
        try:
            yield
        finally:
            for k, v in original_values.items():
                setattr(settings, k, v)

    return _override
