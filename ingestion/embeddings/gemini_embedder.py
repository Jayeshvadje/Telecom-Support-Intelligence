import os
from typing import List, Dict, Any
from google import genai


class GeminiEmbedder:
    """
    Generates text embeddings using Google GenAI SDK (text-embedding-004).
    """

    def __init__(self, model_name: str = "text-embedding-004"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        texts = [c["content"] for c in chunks]

        if not texts:
            return []

        response = self.client.models.embed_content(
            model=self.model_name,
            contents=texts
        )

        for chunk, embedding_data in zip(chunks, response.embeddings):
            chunk["embedding"] = embedding_data.values

        return chunks