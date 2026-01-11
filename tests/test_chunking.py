def test_chunk_and_embed_token(client, auth_headers):
    # Create a long string
    long_text = "word " * 1000 
    payload = {
        "model": "mini",
        "input": long_text,
        "method": "token",
        "size": 100,
        "overlap": 0
    }
    response = client.post("/embed/chunk", json=payload, headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert len(data["chunks"]) > 1
    assert len(data["vectors"]) == len(data["chunks"])

def test_chunk_and_embed_char(client, auth_headers):
    text = "abcdefghij"
    payload = {
        "model": "mini",
        "input": text,
        "method": "char",
        "size": 5,
        "overlap": 0
    }
    response = client.post("/embed/chunk", json=payload, headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    # "abcdefghij" split by 5 -> "abcde", "fghij"
    assert len(data["chunks"]) == 2
    assert data["chunks"][0] == "abcde"
    assert data["chunks"][1] == "fghij"
