import pytest
import sys
import os
from unittest.mock import MagicMock, AsyncMock

# Ensure paths are set up for generated code
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "app", "grpc", "generated"))

from app.grpc.servicer import EmbeddingServicer
from app.grpc.generated.protos import embedding_pb2

@pytest.fixture
def mock_embedding_service(mocker):
    # Patch the service used in servicer.py
    mock = mocker.patch("app.grpc.servicer.embedding_service", new_callable=AsyncMock)
    mock.get_embeddings.return_value = [[0.1, 0.2, 0.3]]
    mock.chunk_and_embed.return_value = (["chunk1"], [[0.1, 0.2]])
    return mock

@pytest.mark.asyncio
async def test_embed_grpc(mock_embedding_service):
    servicer = EmbeddingServicer()
    request = embedding_pb2.EmbedRequest(model="test-model", input=["hello"])
    
    # Create a mock context
    context = MagicMock()
    
    response = await servicer.Embed(request, context)
    
    assert response.model == "test-model"
    assert response.dims == 3
    assert len(response.vectors) == 1
    assert response.vectors[0].values == pytest.approx([0.1, 0.2, 0.3])
    
    mock_embedding_service.get_embeddings.assert_awaited_once_with("test-model", ["hello"])

@pytest.mark.asyncio
async def test_embed_stream_grpc(mock_embedding_service):
    servicer = EmbeddingServicer()
    
    async def request_iterator():
        yield embedding_pb2.EmbedRequest(model="test-model", input=["hello"])
        yield embedding_pb2.EmbedRequest(model="test-model", input=["world"])
        
    context = MagicMock()
    
    responses = []
    async for response in servicer.EmbedStream(request_iterator(), context):
        responses.append(response)
        
    assert len(responses) == 2
    assert responses[0].vectors[0].values == pytest.approx([0.1, 0.2, 0.3])
    assert mock_embedding_service.get_embeddings.call_count == 2

@pytest.mark.asyncio
async def test_chunk_and_embed_grpc(mock_embedding_service):
    servicer = EmbeddingServicer()
    request = embedding_pb2.ChunkRequest(
        model="test-model", 
        input=["long text"], 
        method="token", 
        size=512, 
        overlap=0
    )
    
    context = MagicMock()
    
    response = await servicer.ChunkAndEmbed(request, context)
    
    assert response.model == "test-model"
    assert len(response.chunks) == 1
    assert len(response.vectors) == 1
    assert response.vectors[0].values == pytest.approx([0.1, 0.2])
    
    mock_embedding_service.chunk_and_embed.assert_awaited_once()
