"""
OpenAI Compatibility Example

This script demonstrates how to use the standard OpenAI Python client library
to interact with this Embedding Server.

Prerequisites:
    pip install openai
"""

from openai import OpenAI
import sys

# Configuration
# If you are using the default settings:
BASE_URL = "http://localhost:8000/v1"
API_KEY = "changeme" # The server ignores this if AUTH_MODE is NONE. 
                     # If AUTH_MODE is KEY/JWT, use your actual key.

def main():
    try:
        print(f"Connecting to Embedding Server at {BASE_URL}...")
        
        # Initialize the OpenAI client pointing to our local server
        client = OpenAI(
            base_url=BASE_URL,
            api_key=API_KEY
        )

        # Define the input text and the model alias (must match models.yaml)
        model_alias = "mini"
        input_text = "Hello from OpenAI client!"

        print(f"Generating embedding for: '{input_text}' using model '{model_alias}'")

        # Call the embeddings endpoint
        response = client.embeddings.create(
            model=model_alias,
            input=input_text
        )

        # Extract the embedding vector
        embedding_vector = response.data[0].embedding
        
        print("\n✅ Success!")
        print(f"Vector Dimensions: {len(embedding_vector)}")
        print(f"First 5 values: {embedding_vector[:5]}...")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Ensure the server is running (python main.py) and the 'openai' library is installed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
