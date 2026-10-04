import os
from typing import List, Dict, Any
from google import genai
from google.genai import types


class GeminiEmbedder:
    """
    Generates text embeddings using Google GenAI SDK ("gemini-embedding-001).
    Configured for 768 dimensions to match Pinecone index schema.
    """

    def __init__(self, model_name: str = "gemini-embedding-001"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        texts = [c["content"] for c in chunks]

        if not texts:
            return []

        # Request 768 dimensions explicitly
        config = types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=768
        )

        response = self.client.models.embed_content(
            model=self.model_name,
            contents=texts,
            config=config
        )

        for chunk, embedding_data in zip(chunks, response.embeddings):
            chunk["embedding"] = embedding_data.values

        return chunks