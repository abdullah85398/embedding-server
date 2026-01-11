import requests
import json

# Configuration
BASE_URL = "http://localhost:8000"
API_KEY = "changeme"  # Default key from settings.py
HEADERS = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

def print_section(title):
    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}")

def print_response(label, response):
    if response.status_code in [200, 201]:
        try:
            data = response.json()
            # Truncate long vectors for display
            if "vectors" in data:
                for i, vec in enumerate(data["vectors"]):
                    data["vectors"][i] = f"Vector[{len(vec)}] (truncated: {vec[:3]}...)"
            if "data" in data and isinstance(data["data"], list): # OpenAI format
                for item in data["data"]:
                    if "embedding" in item:
                        item["embedding"] = f"Vector[{len(item['embedding'])}] (truncated...)"
            
            print(f"✅ {label}: Success")
            print(json.dumps(data, indent=2))
        except Exception:
            print(f"✅ {label}: Success (Non-JSON)")
            print(response.text)
    else:
        print(f"❌ {label}: Failed ({response.status_code})")
        print(response.text)

def run_health_check():
    print_section("1. System Health")
    
    # Simple Health
    resp = requests.get(f"{BASE_URL}/health")
    print_response("Health Check", resp)
    
    # Readiness (Model Status)
    resp = requests.get(f"{BASE_URL}/ready")
    print_response("Readiness Check", resp)

def run_basic_embedding():
    print_section("2. Basic Embedding")
    
    payload = {
        "model": "mini",
        "input": "Hello, world!"
    }
    resp = requests.post(f"{BASE_URL}/embed", headers=HEADERS, json=payload)
    print_response("Single String", resp)

def run_batch_embedding():
    print_section("3. Native Batching")
    
    payload = {
        "model": "mini",
        "input": [
            "The quick brown fox",
            "jumps over the lazy dog",
            "Embeddings are useful"
        ]
    }
    resp = requests.post(f"{BASE_URL}/embed", headers=HEADERS, json=payload)
    print_response("Batch of 3", resp)

def run_structured_embedding():
    print_section("4. Structured Input")
    
    payload = {
        "model": "mini",
        "input": {
            "title": "Machine Learning Guide",
            "body": "This article explains how transformers work...",
            "tags": ["AI", "NLP", "Tutorial"]
        }
    }
    resp = requests.post(f"{BASE_URL}/embed", headers=HEADERS, json=payload)
    print_response("Structured Object", resp)

def run_smart_chunking():
    print_section("5. Smart Chunking")
    
    long_text = "Word " * 600  # Simulate long text
    
    # Case A: Token Chunking (Default/Recommended)
    print("\n--- Method: Token ---")
    payload_token = {
        "model": "mini",
        "input": long_text,
        "method": "token",
        "size": 50,  # Small size to force split
        "overlap": 10
    }
    resp = requests.post(f"{BASE_URL}/embed/chunk", headers=HEADERS, json=payload_token)
    print_response("Token Chunking", resp)
    
    # Case B: Character Chunking
    print("\n--- Method: Character ---")
    payload_char = {
        "model": "mini",
        "input": long_text,
        "method": "char",
        "size": 200,  # Characters
        "overlap": 20
    }
    resp = requests.post(f"{BASE_URL}/embed/chunk", headers=HEADERS, json=payload_char)
    print_response("Character Chunking", resp)

def run_openai_compatible():
    print_section("6. OpenAI Compatibility")
    
    # Simulating OpenAI Client request structure
    payload = {
        "model": "mini",
        "input": "This is an OpenAI-compatible request"
    }
    
    # Note endpoint: /v1/embeddings
    # Auth header typically: Authorization: Bearer <key>
    # But our server accepts X-API-Key too, or Bearer if configured. 
    # Let's use the standard headers we defined for simplicity, assuming server checks both or we are in KEY mode.
    # In KEY mode, X-API-Key is standard.
    
    resp = requests.post(f"{BASE_URL}/v1/embeddings", headers=HEADERS, json=payload)
    print_response("OpenAI Format", resp)

def run_admin_operations():
    print_section("7. Admin Operations")
    
    # 1. Load a new model (using 'code' alias from models.yaml which is preload: false)
    print("Loading model 'code'...")
    payload_load = {
        "alias": "code",
        "model_name": "BAAI/bge-code-v1" 
        # Note: model_name is technically optional if defined in yaml, 
        # but the endpoint schema asks for it. 
        # Actually, if it's in YAML, we might just need alias? 
        # Let's check schema: LoadModelRequest(alias: str, model_name: str, device: Optional[str])
        # So we must provide name.
    }
    resp = requests.post(f"{BASE_URL}/admin/load-model", headers=HEADERS, json=payload_load)
    print_response("Load Model", resp)
    
    # 2. Verify it's ready
    resp = requests.get(f"{BASE_URL}/ready")
    print("Checking ready status...")
    if "code" in resp.json().get("models_loaded", []):
        print("✅ Model 'code' is loaded!")
    else:
        print("❌ Model 'code' not found in ready list.")

    # 3. Unload it
    print("\nUnloading model 'code'...")
    payload_unload = {"alias": "code"}
    resp = requests.post(f"{BASE_URL}/admin/unload-model", headers=HEADERS, json=payload_unload)
    print_response("Unload Model", resp)

if __name__ == "__main__":
    try:
        # Check if server is up first
        requests.get(f"{BASE_URL}/health", timeout=2)
    except requests.exceptions.ConnectionError:
        print(f"❌ Could not connect to {BASE_URL}. Is the server running?")
        print("Run: python main.py")
        exit(1)

    run_health_check()
    run_basic_embedding()
    run_batch_embedding()
    run_structured_embedding()
    run_smart_chunking()
    run_openai_compatible()
    run_admin_operations()
    
    print("\n✅ All examples completed.")
