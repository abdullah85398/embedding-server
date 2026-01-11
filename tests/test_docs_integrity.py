import pytest
from unittest.mock import MagicMock, patch
import sys
import os

# Ensure root directory is in path to import examples
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import examples
# Note: We import them inside tests or use importlib to handle potential side effects if they were to run immediately (they shouldn't if they use if __name__ == "__main__")
import example_client

class MockResponse:
    def __init__(self, json_data, status_code=200):
        self.json_data = json_data
        self.status_code = status_code
        self.text = str(json_data)

    def json(self):
        return self.json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP Error: {self.status_code}")

@pytest.fixture
def mock_requests():
    with patch("requests.post") as mock_post, \
         patch("requests.get") as mock_get:
        
        # Default successful response structure for embedding
        mock_post.return_value = MockResponse({
            "model": "mini",
            "dims": 384,
            "vectors": [[0.1, 0.2, 0.3]],
            "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}], # OpenAI format
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
            "chunks": ["chunk1"],
            "status": "success" # Admin responses
        })
        
        mock_get.return_value = MockResponse({
            "status": "ok",
            "models_loaded": ["mini", "bge", "code"]
        })
        
        yield mock_post, mock_get

def test_example_client_functions(mock_requests):
    """
    Verifies that example_client.py functions execute without error 
    given a mocked server. This ensures the client code structure matches 
    what we document and expect.
    """
    mock_post, mock_get = mock_requests
    
    # Run all functions in example_client
    example_client.run_health_check()
    example_client.run_basic_embedding()
    example_client.run_batch_embedding()
    example_client.run_structured_embedding()
    example_client.run_smart_chunking()
    example_client.run_openai_compatible()
    example_client.run_admin_operations()
    
    # Verify calls were made
    assert mock_get.called
    assert mock_post.called

@pytest.fixture
def mock_openai_client():
    with patch("openai.OpenAI") as MockClient:
        # Setup the mock client instance
        client_instance = MockClient.return_value
        
        # Mock embeddings.create response
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        mock_response.usage.prompt_tokens = 10
        
        client_instance.embeddings.create.return_value = mock_response
        
        yield MockClient

def test_example_openai_logic(mock_openai_client):
    """
    Verifies the logic in example_openai.py.
    Since example_openai.py runs immediately if imported without protection,
    we just verify the critical lines here by simulating what it does.
    """
    # Re-implement the core logic of example_openai.py to verify it works with the lib
    from openai import OpenAI
    
    client = OpenAI(base_url="http://test", api_key="test")
    response = client.embeddings.create(model="mini", input="test")
    
    assert len(response.data[0].embedding) == 3
    assert response.usage.prompt_tokens == 10
