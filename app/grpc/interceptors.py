import grpc
import time
import logging

logger = logging.getLogger("app.grpc.access")

class LoggingInterceptor(grpc.aio.ServerInterceptor):
    async def intercept_service(self, continuation, handler_call_details):
        start_time = time.time()
        method = handler_call_details.method
        try:
            return await continuation(handler_call_details)
        finally:
            duration = time.time() - start_time
            logger.info(f"gRPC method={method} duration={duration:.4f}s")
