import os
import json
from typing import Dict, Any, List
from google import genai
from generation.rag_chain import RAGChain


EVAL_PROMPT = """You are an AI Quality Evaluator for an Enterprise Telecom RAG system.
Evaluate the generated answer strictly against the provided context.

Query: {query}
Context: {context}
Generated Answer: {answer}

Provide your evaluation in JSON format with the following keys:
- "faithfulness_score": float between 0.0 and 1.0 (1.0 = completely faithful to context, 0.0 = contains hallucinations)
- "answer_relevant": boolean (true if it directly answers the query)
- "reasoning": brief explanation of the score
"""


class RAGEvaluator:
    def __init__(self):
        gemini_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=gemini_key)
        self.rag_chain = RAGChain()

    def evaluate_query(self, query: str) -> Dict[str, Any]:
        # Get RAG output
        rag_output = self.rag_chain.answer_query(query)
        matches = self.rag_chain.retriever.search(query, top_k=3)
        context_str = self.rag_chain._format_context(matches)

        prompt = EVAL_PROMPT.format(
            query=query,
            context=context_str,
            answer=rag_output["answer"]
        )

        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={"response_mime_type": "application/json"}
        )

        eval_result = json.loads(response.text)
        eval_result["rag_output"] = rag_output
        return eval_result


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    evaluator = RAGEvaluator()
    test_queries = [
        "What is the policy for eSIM activation?",
        "How do I unlock my 5G router using a master password?",  # Testing ungrounded/out-of-scope query
    ]

    for q in test_queries:
        print(f"\n🧪 Evaluating Query: '{q}'")
        res = evaluator.evaluate_query(q)
        print(f"Faithfulness Score: {res.get('faithfulness_score')}")
        print(f"Answer Relevant: {res.get('answer_relevant')}")
        print(f"Reasoning: {res.get('reasoning')}")