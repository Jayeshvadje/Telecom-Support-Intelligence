import re
from typing import Dict, Any


class MetadataExtractor:
    """
    Extracts policy identifiers and structures chunk payloads for vector indexing.
    """

    @staticmethod
    def extract_policy_id(file_name: str) -> str:
        # Tries to match common patterns like "Section 3", "Doc_07", "POL-102"
        match = re.search(r'(Section\s*\d+|Doc_\d+|POL[-_]?\d+)', file_name, re.IGNORECASE)
        return match.group(0) if match else file_name.replace(".docx", "")

    @classmethod
    def enrich_chunk(cls, chunk: Dict[str, Any]) -> Dict[str, Any]:
        policy_id = cls.extract_policy_id(chunk["file_name"])

        chunk["metadata"] = {
            "policy_id": policy_id,
            "document_name": chunk["file_name"],
            "section_heading": chunk["section_heading"],
            "chunk_index": chunk["chunk_index"],
            "token_count": chunk["token_count"]
        }
        return chunk