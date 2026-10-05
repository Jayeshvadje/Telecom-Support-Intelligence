from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from dotenv import load_dotenv

from generation.rag_chain import RAGChain

load_dotenv()

app = FastAPI(
    title="Telecom Support Intelligence RAG API",
    version="1.0.0",
    description="Enterprise RAG service for Telecom policy ingestion and semantic search."
)

# Initialize RAG Engine on application startup
rag_engine = RAGChain()


class QueryRequest(BaseModel):
    query: str = Field(..., json_schema_extra={"example": "How do I transfer an eSIM to a new phone?"})
    top_k: int = Field(default=3, ge=1, le=10)


class SourceMetadata(BaseModel):
    policy_id: str
    document_name: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceMetadata]


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "Telecom-Support-Intelligence"}


@app.post("/api/v1/query", response_model=QueryResponse)
def query_policy(request: QueryRequest):
    try:
        result = rag_engine.answer_query(
            user_query=request.query, 
            top_k=request.top_k
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))