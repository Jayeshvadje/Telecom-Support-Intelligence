import os
from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec


class PineconeManager:
    """
    Manages Pinecone serverless vector database connection, index creation,
    and metadata payload upserts.
    """

    def __init__(self, index_name: str = "telecom-policies", vector_dim: int = 3072):
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable is not set.")

        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.vector_dim = vector_dim
        self._ensure_index_exists()
        self.index = self.pc.Index(self.index_name)

    def _ensure_index_exists(self):
        """
        Creates a serverless Pinecone index if it doesn't already exist.
        `text-embedding-004` output dimension default = 3072.
        """
        existing_indexes = [i.name for i in self.pc.list_indexes()]

        if self.index_name not in existing_indexes:
            print(f"Creating Pinecone serverless index: '{self.index_name}'...")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.vector_dim,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            print(f"Index '{self.index_name}' created successfully.")

    def upsert_chunks(self, chunks: List[Dict[str, Any]], namespace: str = "default"):
        """
        Upserts embedded chunks into Pinecone with associated metadata payloads.
        """
        vectors_to_upsert = []

        for chunk in chunks:
            if "embedding" not in chunk:
                continue

            vectors_to_upsert.append({
                "id": chunk["chunk_id"],
                "values": chunk["embedding"],
                "metadata": {
                    "content": chunk["content"],
                    "policy_id": chunk["metadata"]["policy_id"],
                    "document_name": chunk["metadata"]["document_name"],
                    "section_heading": chunk["metadata"]["section_heading"],
                    "chunk_index": chunk["metadata"]["chunk_index"],
                    "token_count": chunk["metadata"]["token_count"]
                }
            })

        if vectors_to_upsert:
            # Batch upsert in chunks of 100 to respect request size limits
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                self.index.upsert(vectors=batch, namespace=namespace)

            print(f"Successfully upserted {len(vectors_to_upsert)} vectors into Pinecone index '{self.index_name}'.")