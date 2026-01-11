def test_openai_embeddings(client, auth_headers):
    payload = {
        "model": "mini",
        "input": "Hello OpenAI"
    }
    # OpenAI auth uses Bearer token, but our middleware checks X-API-Key OR Bearer
    # Let's test with Bearer token
    headers = {"Authorization": "Bearer test-secret"}
    
    response = client.post("/v1/embeddings", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["object"] == "embedding"
    assert data["model"] == "mini"
    assert "usage" in data
    assert data["usage"]["prompt_tokens"] > 0
    assert data["usage"]["total_tokens"] > 0

def test_openai_embeddings_batch(client, auth_headers):
    payload = {
        "model": "mini",
        "input": ["Hello", "World"]
    }
    headers = {"Authorization": "Bearer test-secret"}
    response = client.post("/v1/embeddings", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert len(data["data"]) == 2
    assert data["data"][0]["index"] == 0
    assert data["data"][1]["index"] == 1
