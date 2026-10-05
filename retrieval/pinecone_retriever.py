import os
from typing import List, Dict, Any
from pinecone import Pinecone
from google import genai
from google.genai import types


class PineconeRetriever:
    """
    Handles semantic query embedding and top-k vector retrieval from Pinecone.
    """

    def __init__(
        self, 
        index_name: str = "telecom-policies", 
        model_name: str = "gemini-embedding-001",
        namespace: str = "default"  # Match namespace used in PineconeManager
    ):
        gemini_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        self.gemini_client = genai.Client(api_key=gemini_key)
        self.model_name = model_name
        self.namespace = namespace

        pinecone_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_key:
            raise ValueError("PINECONE_API_KEY environment variable is not set.")
        
        self.pc = Pinecone(api_key=pinecone_key)
        self.index = self.pc.Index(index_name)

    def _embed_query(self, query: str) -> List[float]:
        """Embeds a user query string into a 768-dim vector with RETRIEVAL_QUERY task type."""
        config = types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=768
        )
        response = self.gemini_client.models.embed_content(
            model=self.model_name,
            contents=query,
            config=config
        )
        return response.embeddings[0].values

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        query_vector = self._embed_query(query)

        response = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            namespace=self.namespace  # Target the namespace containing the vectors
        )

        # Support both dictionary and object response types
        matches = response.matches if hasattr(response, 'matches') else response.get("matches", [])

        results = []
        for match in matches:
            metadata = match.metadata if hasattr(match, 'metadata') else match.get("metadata", {})
            score = match.score if hasattr(match, 'score') else match.get("score", 0.0)

            results.append({
                "score": round(score, 4),
                "policy_id": metadata.get("policy_id", "N/A"),
                "document_name": metadata.get("document_name", "N/A"),
                "section_heading": metadata.get("section_heading", "N/A"),
                "content": metadata.get("content", "N/A")
            })

        return results


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    retriever = PineconeRetriever()
    test_query = "What is the policy for eSIM activation?"
    print(f"🔍 Searching for: '{test_query}'...\n")
    
    matches = retriever.search(query=test_query, top_k=2)
    for i, m in enumerate(matches, 1):
        print(f"--- Match {i} (Score: {m['score']}) ---")
        print(f"Policy: {m['policy_id']} | Doc: {m['document_name']}")
        print(f"Section: {m['section_heading']}")
        print(f"Content Sample: {m['content'][:150]}...\n")