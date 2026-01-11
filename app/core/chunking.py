from typing import List
import tiktoken

class ChunkingService:
    def __init__(self):
        # Default tokenizer for general purpose English
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def chunk_text(self, text: str, method: str = "token", size: int = 512, overlap: int = 0) -> List[str]:
        if method == "char":
            return self._chunk_by_char(text, size, overlap)
        elif method == "token":
            return self._chunk_by_token(text, size, overlap)
        else:
            raise ValueError(f"Unknown chunking method: {method}")

    def _chunk_by_char(self, text: str, size: int, overlap: int) -> List[str]:
        if size <= overlap:
            raise ValueError("Chunk size must be greater than overlap")
        
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + size, text_len)
            chunks.append(text[start:end])
            if end == text_len:
                break
            start += (size - overlap)
            
        return chunks

    def _chunk_by_token(self, text: str, size: int, overlap: int) -> List[str]:
        if size <= overlap:
            raise ValueError("Chunk size must be greater than overlap")

        tokens = self.tokenizer.encode(text)
        total_tokens = len(tokens)
        chunks = []
        start = 0
        
        while start < total_tokens:
            end = min(start + size, total_tokens)
            chunk_tokens = tokens[start:end]
            chunks.append(self.tokenizer.decode(chunk_tokens))
            if end == total_tokens:
                break
            start += (size - overlap)
            
        return chunks

chunking_service = ChunkingService()
