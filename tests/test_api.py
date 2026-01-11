def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_embed_single_string(client, auth_headers):
    payload = {
        "model": "mini",
        "input": "Hello world"
    }
    response = client.post("/embed", json=payload, headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "mini"
    assert len(data["vectors"]) == 1
    assert len(data["vectors"][0]) == 384 # MiniLM dimension

def test_embed_list_strings(client, auth_headers):
    payload = {
        "model": "mini",
        "input": ["Hello", "World"]
    }
    response = client.post("/embed", json=payload, headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert len(data["vectors"]) == 2

def test_embed_structured_input(client, auth_headers):
    payload = {
        "model": "mini",
        "input": {
            "title": "Test Title",
            "body": "Test Body",
            "tags": ["tag1", "tag2"]
        }
    }
    response = client.post("/embed", json=payload, headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert len(data["vectors"]) == 1

def test_embed_invalid_model(client, auth_headers):
    payload = {
        "model": "invalid-model",
        "input": "Hello"
    }
    # Currently implementation raises 400 for ValueError (model not found)
    response = client.post("/embed", json=payload, headers=auth_headers)
    assert response.status_code == 400
    assert "not found" in response.json()["detail"].lower()
