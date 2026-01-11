def test_admin_load_unload_model(client, auth_headers):
    # 1. Load a new model (using a small one for test speed)
    # We'll use the same mini model but with a different alias to simulate "new" load
    payload_load = {
        "alias": "test-model",
        "model_name": "all-MiniLM-L6-v2",
        "device": "cpu"
    }
    response = client.post("/admin/load-model", json=payload_load, headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["status"] == "success"

    # 2. Verify it's loaded via ready check
    response = client.get("/ready")
    assert "test-model" in response.json()["models_loaded"]

    # 3. Use the new model
    payload_embed = {
        "model": "test-model",
        "input": "test"
    }
    response = client.post("/embed", json=payload_embed, headers=auth_headers)
    assert response.status_code == 200

    # 4. Unload the model
    payload_unload = {
        "alias": "test-model"
    }
    response = client.post("/admin/unload-model", json=payload_unload, headers=auth_headers)
    assert response.status_code == 200

    # 5. Verify it's gone
    response = client.get("/ready")
    assert "test-model" not in response.json()["models_loaded"]
