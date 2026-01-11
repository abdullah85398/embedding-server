import pytest
from unittest.mock import AsyncMock, MagicMock
from app.grpc.servicer import EmbeddingServicer
from app.grpc.generated.protos import embedding_pb2
from app.api.endpoints import router
from fastapi.testclient import TestClient
from fastapi import FastAPI

# Create a minimal app for testing HTTP
app = FastAPI()
app.include_router(router)
client = TestClient(app)

@pytest.fixture
def mock_service(mocker):
    # Patch the singleton in both places
    mock = mocker.patch("app.services.embedding_service.embedding_service", new_callable=AsyncMock)
    # Patch in endpoints.py import
    mocker.patch("app.api.endpoints.embedding_service", mock)
    # Patch in servicer.py import
    mocker.patch("app.grpc.servicer.embedding_service", mock)
    
    mock.get_embeddings.return_value = [[0.1, 0.2, 0.3]]
    return mock

@pytest.fixture
def auth_headers():
    return {"X-API-Key": "test-secret"}

@pytest.fixture(autouse=True)
def mock_auth(mocker):
    # Bypass auth for simplicity in interop test
    mocker.patch("app.api.endpoints.verify_api_key", return_value=True)

@pytest.mark.asyncio
async def test_protocol_consistency(mock_service, auth_headers):
    """
    Ensure both protocols map to the same service call and return compatible data.
    """
    model_name = "test-model"
    input_text = "hello"
    
    # 1. HTTP Call
    # Note: We need to mock verify_api_key dependency injection if we use TestClient
    # But since we patched endpoints.embedding_service, it should work if we handle auth.
    # The fixture mock_auth tries to patch the function, but FastAPI dependencies are resolved at startup.
    # So we better use override_dependency.
    app.dependency_overrides = {}
    from app.middleware.auth import verify_api_key
    app.dependency_overrides[verify_api_key] = lambda: True
    
    http_response = client.post("/embed", json={"model": model_name, "input": input_text})
    assert http_response.status_code == 200
    http_data = http_response.json()
    
    # 2. gRPC Call
    servicer = EmbeddingServicer()
    grpc_request = embedding_pb2.EmbedRequest(model=model_name, input=[input_text])
    context = MagicMock()
    grpc_response = await servicer.Embed(grpc_request, context)
    
    # 3. Comparison
    assert http_data["model"] == grpc_response.model
    # HTTP returns list of lists, gRPC returns list of Vector(values=list)
    assert http_data["vectors"][0] == pytest.approx(list(grpc_response.vectors[0].values))
    
    # Verify service was called twice
    assert mock_service.get_embeddings.call_count == 2
