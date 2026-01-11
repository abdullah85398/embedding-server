import asyncio
import tiktoken
import logging
from fastapi import APIRouter, Depends, HTTPException
from typing import List
from app.models.schemas import (
    EmbedRequest, EmbedResponse, StructuredInput,
    ChunkRequest, ChunkResponse,
    LoadModelRequest, UnloadModelRequest,
    OpenAIEmbedRequest, OpenAIEmbedResponse, OpenAIEmbeddingObject, OpenAIUsage,
    TokenRequest, TokenResponse
)
from app.core.model_manager import model_manager
from app.services.embedding_service import embedding_service
from app.middleware.auth import verify_api_key, verify_master_key
from app.config.settings import settings
from app.core.security import create_access_token

# Setup logger
logger = logging.getLogger(__name__)

router = APIRouter()

concurrency_limiter = asyncio.Semaphore(settings.max_inflight_requests)

# Tokenizer for counting usage (approximate, using cl100k_base)
usage_tokenizer = tiktoken.get_encoding("cl100k_base")

@router.post("/auth/token", response_model=TokenResponse, dependencies=[Depends(verify_master_key)])
async def get_access_token(request: TokenRequest):
    """
    Generate a short-lived JWT access token.
    Requires the Master API Key.
    """
    access_token = create_access_token(subject=request.client_id, client_id=request.client_id)
    return TokenResponse(
        access_token=access_token,
        expires_in=settings.access_token_expire_minutes * 60
    )

@router.post("/embed", response_model=EmbedResponse, dependencies=[Depends(verify_api_key)])
async def embed(request: EmbedRequest):
    """
    Generate embeddings for a list of texts or structured inputs.
    
    Args:
        request (EmbedRequest): The embedding request containing model alias and input data.
        
    Returns:
        EmbedResponse: Object containing the model used, dimensions, and list of vectors.
        
    Raises:
        HTTPException: 422 for invalid input, 400 for model errors, 500 for internal errors.
    """
    async with concurrency_limiter:
        # 1. Normalize Input
        raw_inputs = request.input
        input_texts: List[str] = []
        
        # Manual Validation Logic because Pydantic Union was acting up
        if isinstance(raw_inputs, str):
            input_texts = [raw_inputs]
        elif isinstance(raw_inputs, dict):
            # Try to validate as StructuredInput
            try:
                si = StructuredInput(**raw_inputs)
                input_texts = [si.to_text()]
            except Exception as e:
                logger.warning(f"Invalid structured input: {e}")
                raise HTTPException(status_code=422, detail=f"Invalid structured input: {str(e)}")
        elif isinstance(raw_inputs, list):
            for item in raw_inputs:
                if isinstance(item, str):
                    input_texts.append(item)
                elif isinstance(item, dict):
                    try:
                        si = StructuredInput(**item)
                        input_texts.append(si.to_text())
                    except Exception as e:
                        logger.warning(f"Invalid item in list: {e}")
                        raise HTTPException(status_code=422, detail=f"Invalid item in list: {str(e)}")
                elif isinstance(item, StructuredInput): # Should not happen with Any but good for safety
                    input_texts.append(item.to_text())
                else:
                    logger.warning(f"Unsupported input type in list: {type(item)}")
                    raise HTTPException(status_code=422, detail=f"Unsupported input type in list: {type(item)}")
        elif isinstance(raw_inputs, StructuredInput):
             input_texts = [raw_inputs.to_text()]
        else:
            logger.warning(f"Unsupported input type: {type(raw_inputs)}")
            raise HTTPException(status_code=422, detail=f"Unsupported input type: {type(raw_inputs)}")
        
        # 2. Get Embeddings via Service
        try:
            final_vectors = await embedding_service.get_embeddings(request.model, input_texts)
        except ValueError as e:
             raise HTTPException(status_code=400, detail=str(e))
        except RuntimeError as e:
             raise HTTPException(status_code=500, detail=str(e))

        # 3. Construct Response
        dims = len(final_vectors[0]) if final_vectors else 0
        
        return EmbedResponse(
            model=request.model,
            dims=dims,
            vectors=final_vectors
        )

@router.post("/embed/chunk", response_model=ChunkResponse, dependencies=[Depends(verify_api_key)])
async def chunk_and_embed(request: ChunkRequest):
    """
    Split input text into chunks and generate embeddings for each chunk.
    
    Args:
        request (ChunkRequest): Request containing text, model, and chunking parameters.
        
    Returns:
        ChunkResponse: List of chunks and their corresponding embeddings.
    """
    async with concurrency_limiter:
        raw_inputs = request.input
        if isinstance(raw_inputs, str):
            raw_inputs = [raw_inputs]
        
        try:
            chunks, vectors = await embedding_service.chunk_and_embed(
                request.model,
                raw_inputs,
                request.method,
                request.size,
                request.overlap
            )
            
            return ChunkResponse(
                model=request.model,
                chunks=chunks,
                vectors=vectors
            )
        except Exception as e:
            logger.exception(f"Chunk embedding failed: {e}")
            raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

# --- OpenAI Compatible Endpoint ---

@router.post("/v1/embeddings", response_model=OpenAIEmbedResponse, dependencies=[Depends(verify_api_key)])
async def openai_embeddings(request: OpenAIEmbedRequest):
    """
    OpenAI-compatible endpoint for generating embeddings.
    
    Args:
        request (OpenAIEmbedRequest): Request following OpenAI's API format.
        
    Returns:
        OpenAIEmbedResponse: Response following OpenAI's API format.
    """
    async with concurrency_limiter:
        # 1. Normalize Input (OpenAI supports str, list[str], list[int])
        # We only support str and list[str] for now
        input_texts: List[str] = []
        if isinstance(request.input, str):
            input_texts = [request.input]
        elif isinstance(request.input, list):
            # Check if list of strings
            if request.input and isinstance(request.input[0], str):
                input_texts = request.input
            else:
                logger.warning("Token embeddings (list[int]) requested but not supported")
                raise HTTPException(status_code=400, detail="Token embeddings (list[int]) not supported yet.")
        else:
            logger.warning(f"Invalid input format: {type(request.input)}")
            raise HTTPException(status_code=400, detail="Invalid input format.")
        
        # 2. Compute Embeddings
        try:
             # Calculate Usage (approximate)
            prompt_tokens = 0
            for text in input_texts:
                prompt_tokens += len(usage_tokenizer.encode(text))

            final_vectors = await embedding_service.get_embeddings(request.model, input_texts)
        except Exception as e:
            logger.exception(f"OpenAI embedding failed: {e}")
            raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

        # 3. Construct OpenAI Response
        data_objects = []
        for i in range(len(input_texts)):
            data_objects.append(OpenAIEmbeddingObject(
                embedding=final_vectors[i],
                index=i
            ))
            
        return OpenAIEmbedResponse(
            data=data_objects,
            model=request.model,
            usage=OpenAIUsage(
                prompt_tokens=prompt_tokens,
                total_tokens=prompt_tokens # No completion, so total = prompt
            )
        )

# --- Admin Endpoints ---

@router.post("/admin/load-model", dependencies=[Depends(verify_api_key)])
async def load_model_admin(request: LoadModelRequest):
    """
    Admin endpoint to manually load a model into memory.
    """
    try:
        model_manager.load_model(request.alias, request.model_name, request.device)
        return {"status": "success", "message": f"Model {request.alias} ({request.model_name}) loaded on {request.device or 'default'}"}
    except Exception as e:
        logger.error(f"Failed to load model {request.alias}: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/admin/unload-model", dependencies=[Depends(verify_api_key)])
async def unload_model_admin(request: UnloadModelRequest):
    """
    Admin endpoint to unload a model from memory.
    """
    model_manager.unload_model(request.alias)
    return {"status": "success", "message": f"Model {request.alias} unloaded"}

@router.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}

@router.get("/ready")
async def ready():
    """Readiness probe checking if models are loaded."""
    return {"status": "ready", "models_loaded": list(model_manager.models.keys())}
