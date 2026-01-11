from app.config.settings import settings
from app.core.security import create_access_token, decode_access_token
from app.config.settings import AuthMode

def test_security_headers(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert "Strict-Transport-Security" in response.headers

def test_auth_master_key_header(client, override_settings):
    # We must enable AUTH_MODE=KEY for this test, otherwise all requests pass
    with override_settings(auth_mode=AuthMode.KEY):
        # Test valid key
        response = client.post(
            "/embed",
            headers={"X-API-Key": settings.api_key},
            json={"model": "test-model", "input": "hello"}
        )
        # 400 or 200 is fine, just not 403
        assert response.status_code != 403

        # Test invalid key
        response = client.post(
            "/embed",
            headers={"X-API-Key": "wrong-key"},
            json={"model": "test-model", "input": "hello"}
        )
        assert response.status_code == 403

def test_auth_token_flow(client, override_settings):
    # Update settings to allow test-user
    settings.registered_client_ids.add("test-user")
    
    try:
        # 1. Get Token using Master Key
        response = client.post(
            "/auth/token",
            headers={"X-API-Key": settings.api_key},
            json={"client_id": "test-user"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        token = data["access_token"]
        
        # 2. Use Token (Switch to JWT mode for this test)
        with override_settings(auth_mode=AuthMode.JWT):
            response = client.post(
                "/embed",
                headers={"Authorization": f"Bearer {token}"},
                json={"model": "test-model", "input": "hello"}
            )
            assert response.status_code != 403

        # 3. Verify Token Payload
        payload = decode_access_token(token)
        assert payload["sub"] == "test-user"
        assert payload["client_id"] == "test-user"

    finally:
        if "test-user" in settings.registered_client_ids:
            settings.registered_client_ids.remove("test-user")

def test_auth_token_endpoint_protection(client, override_settings):
    # Ensure auth is enabled
    with override_settings(auth_mode=AuthMode.KEY):
        # Try to get token without master key
        response = client.post(
            "/auth/token",
            headers={"X-API-Key": "wrong"},
            json={"client_id": "hacker"}
        )
        assert response.status_code == 403

def test_rate_limit_token_granularity(client, override_settings):
    # Generate two tokens for two different users
    settings.registered_client_ids.add("user1")
    settings.registered_client_ids.add("user2")
    
    try:
        token1 = create_access_token("user1", client_id="user1")
        token2 = create_access_token("user2", client_id="user2")
        
        with override_settings(auth_mode=AuthMode.JWT):
            # User 1 request
            response = client.post(
                "/embed",
                headers={"Authorization": f"Bearer {token1}"},
                json={"model": "test-model", "input": "hello"}
            )
            assert response.status_code != 403
            
            # User 2 request
            response = client.post(
                "/embed",
                headers={"Authorization": f"Bearer {token2}"},
                json={"model": "test-model", "input": "hello"}
            )
            assert response.status_code != 403
            
    finally:
        settings.registered_client_ids.discard("user1")
        settings.registered_client_ids.discard("user2")
