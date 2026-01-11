import grpc
import logging
import sys
import os

# Add generated code to path
sys.path.append(os.path.join(os.path.dirname(__file__), "generated"))

from protos import embedding_pb2
from protos import embedding_pb2_grpc
from app.services.embedding_service import embedding_service

logger = logging.getLogger(__name__)

class EmbeddingServicer(embedding_pb2_grpc.EmbeddingServiceServicer):
    async def Embed(self, request, context):
        try:
            vectors = await embedding_service.get_embeddings(request.model, request.input)
            
            # Convert list of lists to repeated Vector messages
            vector_msgs = [embedding_pb2.Vector(values=v) for v in vectors]
            
            return embedding_pb2.EmbedResponse(
                model=request.model,
                dims=len(vectors[0]) if vectors else 0,
                vectors=vector_msgs
            )
        except ValueError as e:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
        except Exception as e:
            logger.exception("gRPC Embed failed")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def EmbedStream(self, request_iterator, context):
        async for request in request_iterator:
            try:
                vectors = await embedding_service.get_embeddings(request.model, request.input)
                vector_msgs = [embedding_pb2.Vector(values=v) for v in vectors]
                yield embedding_pb2.EmbedResponse(
                    model=request.model,
                    dims=len(vectors[0]) if vectors else 0,
                    vectors=vector_msgs
                )
            except Exception as e:
                logger.exception("gRPC EmbedStream failed")
                await context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def ChunkAndEmbed(self, request, context):
        try:
            chunks, vectors = await embedding_service.chunk_and_embed(
                request.model,
                request.input,
                request.method,
                request.size,
                request.overlap
            )
            
            vector_msgs = [embedding_pb2.Vector(values=v) for v in vectors]
            
            return embedding_pb2.ChunkResponse(
                model=request.model,
                chunks=chunks,
                vectors=vector_msgs
            )
        except Exception as e:
            logger.exception("gRPC ChunkAndEmbed failed")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))
