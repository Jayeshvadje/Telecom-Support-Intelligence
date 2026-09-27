from typing import List, Dict, Any
import tiktoken


class PolicyChunker:
    """
    Splits DOCX parsed elements into semantically coherent policy chunks 
    using token-based sliding windows with overlap.
    """

    def __init__(self, target_tokens: int = 600, overlap_tokens: int = 100):
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def create_chunks(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        elements = document.get("elements", [])
        file_name = document.get("file_name", "Unknown")

        chunks = []
        current_chunk_elements = []
        current_token_count = 0
        current_heading = "General"
        chunk_index = 0

        for elem in elements:
            text = elem["text"]
            elem_tokens = self.count_tokens(text)

            # Update heading context if element is a heading
            if elem["type"] == "heading":
                current_heading = text

            # If adding this element exceeds target tokens, flush current chunk
            if current_token_count + elem_tokens > self.target_tokens and current_chunk_elements:
                chunk_text = "\n\n".join([e["text"] for e in current_chunk_elements])
                
                chunks.append({
                    "chunk_id": f"{file_name}_chunk_{chunk_index}",
                    "chunk_index": chunk_index,
                    "content": chunk_text,
                    "token_count": current_token_count,
                    "section_heading": current_heading,
                    "file_name": file_name
                })
                
                chunk_index += 1

                # Keep overlap elements for continuity
                overlap_elements = []
                overlap_count = 0
                for e in reversed(current_chunk_elements):
                    t_count = self.count_tokens(e["text"])
                    if overlap_count + t_count <= self.overlap_tokens:
                        overlap_elements.insert(0, e)
                        overlap_count += t_count
                    else:
                        break

                current_chunk_elements = overlap_elements
                current_token_count = overlap_count

            current_chunk_elements.append(elem)
            current_token_count += elem_tokens

        # Flush final remaining elements
        if current_chunk_elements:
            chunk_text = "\n\n".join([e["text"] for e in current_chunk_elements])
            chunks.append({
                "chunk_id": f"{file_name}_chunk_{chunk_index}",
                "chunk_index": chunk_index,
                "content": chunk_text,
                "token_count": current_token_count,
                "section_heading": current_heading,
                "file_name": file_name
            })

        return chunks