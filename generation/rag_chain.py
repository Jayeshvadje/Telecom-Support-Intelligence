import os
from typing import List, Dict, Any
from google import genai
from retrieval.pinecone_retriever import PineconeRetriever


SYSTEM_PROMPT = """You are an Tier-1 Telecom Support Assistant.
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
    Combines PineconeRetriever and Gemini generation to execute 
    Context-Grounded Question Answering.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        gemini_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        
        self.client = genai.Client(api_key=gemini_key)
        self.model_name = model_name
        self.retriever = PineconeRetriever()

    def _format_context(self, matches: List[Dict[str, Any]]) -> str:
        """Formats Pinecone search matches into a clean string block for the LLM."""
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

    def answer_query(self, user_query: str, top_k: int = 3) -> Dict[str, Any]:
        # 1. Retrieve relevant policy chunks
        matches = self.retriever.search(query=user_query, top_k=top_k)
        
        if not matches:
            return {
                "answer": "No relevant policy documents were found for your query.",
                "sources": []
            }

        # 2. Format context and system prompt
        formatted_context = self._format_context(matches)
        prompt = f"User Question: {user_query}"

        # 3. Call Gemini Model
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={
                "system_instruction": SYSTEM_PROMPT.format(context=formatted_context),
                "temperature": 0.2  # Low temperature for strict factual adherence
            }
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


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    rag = RAGChain()
    query = "What is the policy for eSIM activation?"
    print(f"🤖 Processing Query: '{query}'\n")
    
    result = rag.answer_query(query)
    
    print("=== ANSWER ===")
    print(result["answer"])
    print("\n=== CITED SOURCES ===")
    for src in result["sources"]:
        print(f"- {src['policy_id']} ({src['document_name']}) | Score: {src['score']}")