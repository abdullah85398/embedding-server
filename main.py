from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from app.config.settings import settings
from app.api.endpoints import router
from app.middleware.rate_limit import RateLimitMiddleware
from app.middleware.security_headers import SecurityHeadersMiddleware
import logging
import asyncio
import grpc
from app.grpc.interceptors import LoggingInterceptor

# Configure logging
class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "health" in msg or "ready" in msg: # Reduce noise
            return False
        return True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("uvicorn.access").addFilter(EndpointFilter())

app = FastAPI(title=settings.app_name, version="1.0.0")

# Middleware
app.add_middleware(SecurityHeadersMiddleware)
# Add Rate Limiting (Default: 600 requests per minute per IP)
app.add_middleware(RateLimitMiddleware, max_requests=600, window_seconds=60)

# Include Router
app.include_router(router)

# Instrumentation for Prometheus
instrumentator = Instrumentator().instrument(app).expose(app)

async def serve_grpc():
    # Deferred import to avoid circular dependencies or path issues
    from app.grpc.servicer import EmbeddingServicer, embedding_pb2_grpc
    from app.grpc.generated.protos import embedding_pb2
    from grpc_reflection.v1alpha import reflection

    server = grpc.aio.server(interceptors=[LoggingInterceptor()])
    embedding_pb2_grpc.add_EmbeddingServiceServicer_to_server(EmbeddingServicer(), server)
    
    # Enable reflection
    SERVICE_NAMES = (
        embedding_pb2.DESCRIPTOR.services_by_name['EmbeddingService'].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)
    
    server.add_insecure_port('[::]:50051')
    logger.info("Starting gRPC server on [::]:50051")
    await server.start()
    await server.wait_for_termination()

async def main():
    import uvicorn
    # Use standard uvicorn configuration
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    
    logger.info("Starting Dual-Protocol Server (HTTP: 8000, gRPC: 50051)")
    
    await asyncio.gather(
        server.serve(),
        serve_grpc()
    )

if __name__ == "__main__":
    asyncio.run(main())
