from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from app.config.settings import settings
from app.api.endpoints import router
from app.middleware.rate_limit import RateLimitMiddleware
from app.middleware.security_headers import SecurityHeadersMiddleware
import logging
import asyncio
import os
import grpc
from app.grpc.interceptors import LoggingInterceptor

import sys

# Configure logging
class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "health" in msg or "ready" in msg: # Reduce noise
            return False
        return True

# Configure root logger to output to stdout
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

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
    
    grpc_port = int(os.getenv("GRPC_PORT", 50051))
    server.add_insecure_port(f'[::]:{grpc_port}')
    logger.info(f"Starting gRPC server on [::]:{grpc_port}")
    await server.start()
    await server.wait_for_termination()

async def main():
    import uvicorn
    
    http_port = int(os.getenv("PORT", 8000))
    grpc_port = int(os.getenv("GRPC_PORT", 50051))

    # Use standard uvicorn configuration
    config = uvicorn.Config(app, host="0.0.0.0", port=http_port, log_level="info")
    server = uvicorn.Server(config)
    
    logger.info(f"Starting Dual-Protocol Server (HTTP: {http_port}, gRPC: {grpc_port})")
    
    await asyncio.gather(
        server.serve(),
        serve_grpc()
    )

if __name__ == "__main__":
    asyncio.run(main())
