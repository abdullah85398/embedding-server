import asyncio
import logging
from typing import List
from app.core.model_manager import model_manager
from app.core.cache import cache_manager
from app.core.chunking import chunking_service

logger = logging.getLogger(__name__)

class EmbeddingService:
    @staticmethod
    async def get_embeddings(model_name: str, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of texts, handling caching and missing values.
        """
        vectors_map = {}
        missing_indices = []
        missing_texts = []

        # Check Cache
        for i, text in enumerate(texts):
            cached_vector = cache_manager.get_embedding(model_name, text)
            if cached_vector:
                vectors_map[i] = cached_vector
            else:
                missing_indices.append(i)
                missing_texts.append(text)

        # Compute Missing
        if missing_texts:
            try:
                # Offload blocking model inference to thread pool
                model = model_manager.get_model(model_name)
                
                loop = asyncio.get_event_loop()
                # model.encode usually returns numpy array
                new_vectors = await loop.run_in_executor(None, model.encode, missing_texts)
                
                # Convert to list if it's numpy array
                if hasattr(new_vectors, 'tolist'):
                    new_vectors_list = new_vectors.tolist()
                else:
                    new_vectors_list = new_vectors
                
                for i, vector in enumerate(new_vectors_list):
                    original_idx = missing_indices[i]
                    vectors_map[original_idx] = vector
                    # Cache the result
                    cache_manager.set_embedding(model_name, missing_texts[i], vector)
                    
            except ValueError as e:
                logger.error(f"Model error for {model_name}: {e}")
                raise ValueError(str(e))
            except Exception as e:
                logger.exception(f"Internal embedding error: {e}")
                raise RuntimeError(f"Internal embedding error: {str(e)}")

        # Construct Final List
        final_vectors = [vectors_map[i] for i in range(len(texts))]
        return final_vectors

    @staticmethod
    async def chunk_and_embed(
        model_name: str, 
        texts: List[str], 
        method: str = "token", 
        size: int = 512, 
        overlap: int = 0
    ) -> tuple[List[str], List[List[float]]]:
        """
        Chunk texts and return both chunks and their embeddings.
        """
        all_chunks = []
        
        for text in texts:
            chunks = chunking_service.chunk_text(
                text, 
                method=method, 
                size=size, 
                overlap=overlap
            )
            all_chunks.extend(chunks)
        
        if not all_chunks:
            return [], []

        vectors = await EmbeddingService.get_embeddings(model_name, all_chunks)
        return all_chunks, vectors

embedding_service = EmbeddingService()
