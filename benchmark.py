import time
import asyncio
import grpc
import httpx
import sys
import os
import random
import string
import logging

# Setup imports for gRPC
sys.path.append(os.path.join(os.path.dirname(__file__), "app", "grpc", "generated"))
# Handle the fact that generated code imports 'protos' which is in 'generated/protos'
sys.path.append(os.path.join(os.path.dirname(__file__), "app", "grpc", "generated"))

from protos import embedding_pb2
from protos import embedding_pb2_grpc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("benchmark")

HTTP_URL = "http://localhost:8000/embed"
GRPC_TARGET = "localhost:50051"
MODEL = "mini" # Default model
API_KEY = "changeme" 

def generate_text(length=50):
    return ''.join(random.choices(string.ascii_letters + " ", k=length))

async def benchmark_http(session, texts):
    start = time.time()
    response = await session.post(
        HTTP_URL, 
        json={"model": MODEL, "input": texts},
        headers={"X-API-Key": API_KEY}
    )
    if response.status_code != 200:
        logger.error(f"HTTP Error: {response.text}")
        response.raise_for_status()
    duration = time.time() - start
    return duration

async def benchmark_grpc(stub, texts):
    start = time.time()
    request = embedding_pb2.EmbedRequest(model=MODEL, input=texts)
    await stub.Embed(request)
    duration = time.time() - start
    return duration

async def run_benchmark(iterations=10, batch_size=1):
    texts = [generate_text() for _ in range(batch_size)]
    
    logger.info(f"Benchmarking with batch_size={batch_size}, iterations={iterations}")
    
    # HTTP Setup
    async with httpx.AsyncClient() as client:
        # Warmup
        try:
            await benchmark_http(client, texts)
        except Exception as e:
            logger.warning(f"HTTP Warmup failed (server might be down): {e}")
            return

        total_http = 0
        for _ in range(iterations):
            total_http += await benchmark_http(client, texts)
            
    # gRPC Setup
    async with grpc.aio.insecure_channel(GRPC_TARGET) as channel:
        stub = embedding_pb2_grpc.EmbeddingServiceStub(channel)
        # Warmup
        try:
            await benchmark_grpc(stub, texts)
        except Exception as e:
            logger.warning(f"gRPC Warmup failed (server might be down): {e}")
            return
            
        total_grpc = 0
        for _ in range(iterations):
            total_grpc += await benchmark_grpc(stub, texts)

    avg_http = (total_http / iterations) * 1000
    avg_grpc = (total_grpc / iterations) * 1000
    
    logger.info(f"HTTP Avg Latency: {avg_http:.2f} ms")
    logger.info(f"gRPC Avg Latency: {avg_grpc:.2f} ms")
    if avg_grpc > 0:
        logger.info(f"Speedup: {avg_http/avg_grpc:.2f}x")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        asyncio.run(run_benchmark(iterations=50, batch_size=1))
    else:
        print("Usage: python benchmark.py run")
        print("Make sure the server is running first (python main.py)")
