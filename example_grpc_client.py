"""
gRPC Client Example

This script demonstrates how to interact with the Embedding Server using gRPC.
It covers Unary calls, Streaming, and Chunking.

Prerequisites:
    pip install grpcio grpcio-tools
"""

import sys
import os
import grpc
import time

# --- Setup Import Paths ---
# Add the generated protos directory to python path so we can import them
current_dir = os.path.dirname(os.path.abspath(__file__))
generated_protos_dir = os.path.join(current_dir, "app", "grpc", "generated")
sys.path.append(generated_protos_dir)

# Now we can import the generated modules
# Note: The import structure depends on how protoc generated them.
# Usually it's 'protos.embedding_pb2' if the package was 'protos'.
# Let's assume the folder structure matches the package.
try:
    from protos import embedding_pb2
    from protos import embedding_pb2_grpc
except ImportError:
    # Fallback: try direct import if the path is added directly to the file folder
    sys.path.append(os.path.join(generated_protos_dir, "protos"))
    import embedding_pb2
    import embedding_pb2_grpc

# Configuration
SERVER_ADDRESS = "localhost:50051"

def print_section(title):
    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}")

def run_unary_embed(stub):
    print_section("1. Unary Embed")
    
    # Create Request
    req = embedding_pb2.EmbedRequest(
        model="mini",
        input=["Hello via gRPC", "Speed is key"]
    )
    
    print(f"Sending request with {len(req.input)} items...")
    start = time.time()
    
    # Call RPC
    resp = stub.Embed(req)
    
    duration = (time.time() - start) * 1000
    print(f"✅ Success in {duration:.2f}ms")
    print(f"Model: {resp.model}")
    print(f"Dimensions: {resp.dims}")
    print(f"Vectors received: {len(resp.vectors)}")
    if resp.vectors:
        print(f"First vector snippet: {resp.vectors[0].values[:3]}...")

def run_chunk_and_embed(stub):
    print_section("2. Chunk and Embed")
    
    long_text = "Word " * 1000
    
    req = embedding_pb2.ChunkRequest(
        model="mini",
        input=[long_text],
        method="token",
        size=256,
        overlap=20
    )
    
    print("Sending chunk request (Size: 256, Overlap: 20)...")
    resp = stub.ChunkAndEmbed(req)
    
    print("✅ Success")
    print(f"Total Chunks: {len(resp.chunks)}")
    print(f"Total Vectors: {len(resp.vectors)}")
    print(f"First chunk preview: {resp.chunks[0][:50]}...")

def generate_stream_requests():
    """Yields requests for the stream."""
    inputs = ["Stream Item 1", "Stream Item 2", "Stream Item 3"]
    for text in inputs:
        print(f"  -> Sending: {text}")
        yield embedding_pb2.EmbedRequest(
            model="mini",
            input=[text]
        )
        time.sleep(0.5) # Simulate delay

def run_bidirectional_stream(stub):
    print_section("3. Bidirectional Stream")
    
    # Call RPC with an iterator of requests
    responses = stub.EmbedStream(generate_stream_requests())
    
    # Iterate over responses as they arrive
    print("  <- Waiting for responses...")
    for resp in responses:
        print(f"  <- Received: {len(resp.vectors)} vector(s) from model '{resp.model}'")

if __name__ == "__main__":
    print(f"Connecting to gRPC server at {SERVER_ADDRESS}...")
    
    # Create Channel
    # Note: If you need Metadata/Auth, you can pass it here or in call_credentials
    # For now, assuming insecure or IP-based auth as per default
    with grpc.insecure_channel(SERVER_ADDRESS) as channel:
        stub = embedding_pb2_grpc.EmbeddingServiceStub(channel)
        
        try:
            run_unary_embed(stub)
            run_chunk_and_embed(stub)
            run_bidirectional_stream(stub)
            print("\n✅ All gRPC examples completed.")
            
        except grpc.RpcError as e:
            print(f"\n❌ gRPC Error: {e.code()}")
            print(f"Details: {e.details()}")
