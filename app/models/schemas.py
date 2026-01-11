from pydantic import BaseModel, Field
from typing import List, Union, Optional, Dict, Any

# --- Auth Schemas ---

class TokenRequest(BaseModel):
    client_id: str = Field(..., description="Identifier for the client requesting the token")

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int

# --- Internal Schemas ---

class StructuredInput(BaseModel):
    title: Optional[str] = None
    body: str
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_text(self) -> str:
        parts = []
        if self.title:
            parts.append(f"Title: {self.title}")
        parts.append(self.body)
        if self.tags:
            parts.append(f"Tags: {', '.join(self.tags)}")
        return "\n".join(parts)

class EmbedRequest(BaseModel):
    model: str
    # Typed as Any to handle Pydantic Union complexity with FastAPI
    input: Any 

class EmbedResponse(BaseModel):
    model: str
    dims: int
    vectors: List[List[float]]

class ChunkRequest(BaseModel):
    input: Union[str, List[str]]
    method: str = "token"
    size: int = 512
    overlap: int = 0
    model: str

class ChunkResponse(BaseModel):
    model: str
    chunks: List[str]
    vectors: List[List[float]]

class LoadModelRequest(BaseModel):
    alias: str
    model_name: str
    device: Optional[str] = None

class UnloadModelRequest(BaseModel):
    alias: str

# --- OpenAI Compatibility Schemas ---

class OpenAIEmbedRequest(BaseModel):
    input: Union[str, List[str], List[int], List[List[int]]]
    model: str
    encoding_format: Optional[str] = "float" # float or base64 (we only support float for now)
    user: Optional[str] = None

class OpenAIEmbeddingObject(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int

class OpenAIUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int

class OpenAIEmbedResponse(BaseModel):
    object: str = "list"
    data: List[OpenAIEmbeddingObject]
    model: str
    usage: OpenAIUsage
