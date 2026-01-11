
from app.config.settings import settings, AuthMode
from app.core.security import create_access_token

def test_auth_mode_none(client, override_settings):
    with override_settings(auth_mode=AuthMode.NONE):
        # 1. No credentials
        response = client.post(
            "/embed",
            json={"model": "test-model", "input": "hello"}
        )
        assert response.status_code != 403

        # 2. Garbage credentials
        response = client.post(
            "/embed",
            headers={"Authorization": "Bearer garbage"},
            json={"model": "test-model", "input": "hello"}
        )
        assert response.status_code != 403

def test_auth_mode_key(client, override_settings):
    with override_settings(auth_mode=AuthMode.KEY):
        # 1. Valid X-API-Key
        response = client.post(
            "/embed",
            headers={"X-API-Key": settings.api_key},
            json={"model": "test-model", "input": "hello"}
        )
        assert response.status_code != 403

        # 2. Valid Bearer Key
        response = client.post(
            "/embed",
            headers={"Authorization": f"Bearer {settings.api_key}"},
            json={"model": "test-model", "input": "hello"}
        )
        assert response.status_code != 403

        # 3. Invalid Key
        response = client.post(
            "/embed",
            headers={"X-API-Key": "wrong"},
            json={"model": "test-model", "input": "hello"}
        )
        assert response.status_code == 403

        # 4. No Key
        response = client.post(
            "/embed",
            json={"model": "test-model", "input": "hello"}
        )
        assert response.status_code == 403

def test_auth_mode_jwt(client, override_settings):
    # Prepare test state
    settings.registered_client_ids.add("valid-client")
    token_valid = create_access_token("user", client_id="valid-client")
    token_invalid_client = create_access_token("user", client_id="invalid-client")
    
    try:
        with override_settings(auth_mode=AuthMode.JWT):
            # 1. Valid JWT
            response = client.post(
                "/embed",
                headers={"Authorization": f"Bearer {token_valid}"},
                json={"model": "test-model", "input": "hello"}
            )
            assert response.status_code != 403

            # 2. Invalid Client ID
            response = client.post(
                "/embed",
                headers={"Authorization": f"Bearer {token_invalid_client}"},
                json={"model": "test-model", "input": "hello"}
            )
            assert response.status_code == 403
            assert "Invalid Client ID" in response.text

            # 3. Invalid JWT (signature)
            response = client.post(
                "/embed",
                headers={"Authorization": "Bearer invalid.token.here"},
                json={"model": "test-model", "input": "hello"}
            )
            assert response.status_code == 403

            # 4. Master Key in Header (Should Fail in JWT mode for standard endpoints)
            response = client.post(
                "/embed",
                headers={"X-API-Key": settings.api_key},
                json={"model": "test-model", "input": "hello"}
            )
            assert response.status_code == 403

    finally:
        # Cleanup set modification (override_settings handles auth_mode, but not the set)
        if "valid-client" in settings.registered_client_ids:
            settings.registered_client_ids.remove("valid-client")
