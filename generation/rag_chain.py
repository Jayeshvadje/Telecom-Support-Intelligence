import os
from typing import List, Dict, Any
from google import genai
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from retrieval.pinecone_retriever import PineconeRetriever

SYSTEM_PROMPT = """You are a Tier-1 Telecom Support Assistant.
Answer the user's inquiry strictly using the provided policy context below. 

Guidelines:
1. Base your answer ONLY on the given context. Do not make up or assume steps.
2. If the context does not contain sufficient information, state clearly: "I cannot find specific guidance in the policy documentation for this query."
3. Format your response cleanly with bullet points or numbered steps where appropriate.
4. Include relevant Policy Document IDs (e.g., Doc_04) or section references when citing rules.

=== RETRIEVED POLICY CONTEXT ===
{context}
================================
"""


class RAGChain:
    """
    Combines PineconeRetriever and Gemini generation with automatic retry backoff.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        gemini_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        
        self.client = genai.Client(api_key=gemini_key)
        self.model_name = model_name
        self.retriever = PineconeRetriever()

    def _format_context(self, matches: List[Dict[str, Any]]) -> str:
        context_blocks = []
        for idx, match in enumerate(matches, 1):
            block = (
                f"--- Context Block {idx} ---\n"
                f"Policy ID: {match['policy_id']} ({match['document_name']})\n"
                f"Section: {match['section_heading']}\n"
                f"Content: {match['content']}\n"
            )
            context_blocks.append(block)
        return "\n".join(context_blocks)

    @retry(
        wait=wait_random_exponential(min=1, max=10),
        stop=stop_after_attempt(5)
    )
    def _call_gemini_with_retry(self, prompt: str, system_instruction: str):
        """Calls Gemini API with exponential backoff on 503/429 spikes."""
        return self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={
                "system_instruction": system_instruction,
                "temperature": 0.2
            }
        )

    def answer_query(self, user_query: str, top_k: int = 3) -> Dict[str, Any]:
        matches = self.retriever.search(query=user_query, top_k=top_k)
        
        if not matches:
            return {
                "answer": "No relevant policy documents were found for your query.",
                "sources": []
            }

        formatted_context = self._format_context(matches)
        prompt = f"User Question: {user_query}"
        sys_instruction = SYSTEM_PROMPT.format(context=formatted_context)

        # Execute generation call with automated retries
        response = self._call_gemini_with_retry(
            prompt=prompt, 
            system_instruction=sys_instruction
        )

        sources = [
            {
                "policy_id": m["policy_id"],
                "document_name": m["document_name"],
                "score": m["score"]
            }
            for m in matches
        ]

        return {
            "answer": response.text,
            "sources": sources
        }