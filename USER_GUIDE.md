# Embedding Server User Guide

Welcome to the **Embedding Server** user guide. This high-performance server provides a robust interface for generating vector embeddings from text and structured data. It is designed to be a drop-in replacement for OpenAI's embedding API while offering advanced features like native batching, smart chunking, and multi-model support.

## Table of Contents
1. [Overview](#1-overview)
2. [Quick Start](#2-quick-start)
3. [Configuration Manual](#3-configuration-manual)
4. [Feature Deep Dive](#4-feature-deep-dive)
5. [API Reference](#5-api-reference)
6. [Troubleshooting](#6-troubleshooting)

---

## 1. Overview

The Embedding Server bridges the gap between raw machine learning models and production applications. It exposes state-of-the-art embedding models (like Hugging Face transformers) via two high-performance interfaces:
- **HTTP API (FastAPI)**: Easy to integrate, web-standard, includes Swagger UI.
- **gRPC API**: Low-latency, strongly typed, ideal for microservices.

**Key Capabilities:**
- **OpenAI Compatibility**: Use standard OpenAI libraries to talk to this server.
- **Native Batching**: Process thousands of documents in parallel with optimized GPU usage.
- **Smart Chunking**: Automatically split long documents into overlapping windows to fit model context limits.
- **Hardware Acceleration**: Auto-detects and uses NVIDIA CUDA or Apple MPS (Metal Performance Shaders).
- **Caching**: Built-in Redis and Memory caching to prevent redundant computations.

---

## 2. Quick Start

### Prerequisites
- Python 3.10+
- (Optional) NVIDIA GPU with CUDA drivers or Mac with Apple Silicon.

### Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd embedding-server
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Server
Start the server using Uvicorn (HTTP) and the internal gRPC manager:
```bash
python main.py
```
*By default, the HTTP server runs on port **8000** and gRPC on **50051**.*

### "Hello World" Example
Generate your first embedding using `curl`:

```bash
curl -X POST "http://localhost:8000/embed" \
     -H "Content-Type: application/json" \
     -d '{
           "model": "mini",
           "input": "Hello, world!"
         }'
```

**Response:**
```json
{
  "model": "mini",
  "dims": 384,
  "vectors": [
    [0.12, -0.05, ...] 
  ]
}
```

---

## 3. Configuration Manual

The server is configured via **Environment Variables** (for server settings) and a **YAML file** (for model definitions).

### Server Settings (`.env`)
Create a `.env` file in the root directory to override defaults.

| Variable | Default | Description |
| :--- | :--- | :--- |
| `AUTH_MODE` | `NONE` | Authentication mode. Options: `NONE` (public), `KEY` (API Key), `JWT` (Token). |
| `API_KEY` | `secret-key` | Master API Key used for admin actions or simple auth. |
| `JWT_SECRET` | `secret` | Secret key for signing JWT tokens. |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `30` | Duration (minutes) before a JWT token expires. |
| `MAX_INFLIGHT_REQUESTS` | `100` | Maximum number of concurrent requests processed. |
| `ENABLE_CACHE` | `True` | Enable/Disable caching of embeddings. |
| `REDIS_URL` | `None` | URL for Redis (e.g., `redis://localhost:6379`). Uses local memory if empty. |
| `CACHE_TTL` | `3600` | Time-To-Live for cached items in seconds. |
| `MODEL_CONFIG_PATH` | `models.yaml` | Path to the model definition file. |

### Model Configuration (`models.yaml`)
This file defines which models are available to the API.

**Structure:**
```yaml
models:
  <alias>:
    name: <hugging-face-model-id>
    preload: <true|false>
    device: <cpu|cuda|mps|null>
```

**Example `models.yaml`:**
```yaml
models:
  # A lightweight CPU-friendly model
  mini:
    name: all-MiniLM-L6-v2
    preload: true
    device: cpu

  # A high-performance model for RAG
  bge:
    name: BAAI/bge-base-en-v1.5
    preload: true
    # 'device' is omitted to auto-detect (CUDA/MPS)
```

---

## 4. Feature Deep Dive

### Native Batching
Instead of sending one request per sentence, send a list of strings. The server processes them in parallel batches, significantly improving throughput on GPUs.

**Python Example:**
```python
import requests

payload = {
    "model": "mini",
    "input": [
        "The quick brown fox",
        "jumps over the lazy dog",
        "Machine learning is fascinating"
    ]
}
response = requests.post("http://localhost:8000/embed", json=payload).json()
print(f"Generated {len(response['vectors'])} vectors.")
```

### Smart Chunking
Models have a maximum token limit (e.g., 512 tokens). The `/embed/chunk` endpoint splits long text into manageable pieces with overlap to preserve context.

**Chunking Methods:**

1.  **Token Chunking (`method="token"`)** - *Default & Recommended*
    *   **How it works**: Uses a tokenizer (tiktoken) to count actual tokens (like GPT-4). It ensures chunks fit exactly within the model's context window.
    *   **Use Case**: Best for strict compliance with model limits (e.g., ensuring a chunk is exactly 512 tokens).
    *   **Performance**: Slightly slower due to tokenization overhead but most precise.

2.  **Character Chunking (`method="char"`)**
    *   **How it works**: Splits text based on raw character count (e.g., every 1000 characters).
    *   **Use Case**: Good for rough splitting, simple text processing, or when high speed is required and exact token limits are less critical.
    *   **Performance**: Very fast.

**Visualizing Chunking:**
*Text: "A B C D E F G H I J"*
*Size: 4, Overlap: 2*

1. Chunk 1: `[A, B, C, D]`
2. Chunk 2: `[C, D, E, F]` (Overlap 'C, D')
3. Chunk 3: `[E, F, G, H]` (Overlap 'E, F')
...

**Usage:**
```python
payload = {
    "model": "mini",
    "input": "Very long document content...",
    "method": "token",  # Options: "token" or "char"
    "size": 512,
    "overlap": 50
}
requests.post("http://localhost:8000/embed/chunk", json=payload)
```

### Structured Input
You can embed complex objects (like a blog post with a title and tags) directly. The server intelligently formats them into a single string before embedding.

**JSON Payload:**
```json
{
  "model": "mini",
  "input": {
    "title": "Understanding Embeddings",
    "body": "Embeddings are vector representations...",
    "tags": ["AI", "ML", "Vector DB"]
  }
}
```
*The server converts this to: "Title: Understanding Embeddings\nEmbeddings are vector representations...\nTags: AI, ML, Vector DB"*

---

## 5. API Reference

### Authentication
If `AUTH_MODE` is set to `KEY`:
- Add header: `X-API-Key: <your-api-key>`

If `AUTH_MODE` is set to `JWT`:
1. Call `POST /auth/token` with the Master Key to get a token.
2. Add header: `Authorization: Bearer <access_token>`

### Core Endpoints

#### `POST /embed`
Generate embeddings for text.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `model` | string | The alias of the model (e.g., "mini"). |
| `input` | string/list/dict | The data to embed. |

**Example:**
```bash
curl -X POST "http://localhost:8000/embed" \
     -d '{"model": "mini", "input": ["Text 1", "Text 2"]}'
```

#### `POST /embed/chunk`
Split and embed long text.

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model` | string | - | Model alias. |
| `input` | string | - | Long text to process. |
| `size` | int | 512 | Max tokens per chunk. |
| `overlap` | int | 0 | Overlap between chunks. |

#### `POST /v1/embeddings` (OpenAI Compatible)
Standard OpenAI format.

**Python Integration:**
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-dummy"  # Required by library, ignored if Auth is NONE
)

response = client.embeddings.create(
    model="mini",
    input="Hello from OpenAI client!"
)
print(response.data[0].embedding)
```

### System Endpoints
- `GET /health`: Returns `{"status": "ok"}`.
- `GET /ready`: Returns list of loaded models.
- `POST /admin/load-model`: Load a model dynamically.
  ```json
  {"alias": "new-model", "model_name": "bert-base-uncased"}
  ```

---

## 6. Troubleshooting

### Common Issues

**1. "Model not found" Error**
*   **Cause**: The model alias requested is not in `models.yaml` or hasn't been loaded.
*   **Fix**: Check `models.yaml` aliases. If `preload: false`, explicitly load it via `/admin/load-model`.

**2. Rate Limiting (429 Too Many Requests)**
*   **Cause**: You exceeded the `MAX_INFLIGHT_REQUESTS` semaphore.
*   **Fix**: Increase `MAX_INFLIGHT_REQUESTS` in `.env` or implement client-side backoff/retry logic.

**3. Out of Memory (OOM) on GPU**
*   **Cause**: Batch size is too large for the VRAM.
*   **Fix**: Reduce the number of items in the `input` list per request.

**4. 422 Validation Error**
*   **Cause**: Malformed JSON or invalid data types.
*   **Fix**: Ensure `input` matches the expected schema (e.g., not sending an integer when a string is expected).

### Getting Help
For logs and deeper debugging, check the console output where the server is running. The server uses Python's standard `logging` module to output detailed information about request processing and model loading.
