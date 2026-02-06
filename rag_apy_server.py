# rag_api_server.py

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uvicorn
from rag_with_vllm_streaming import RIBRAGSystemStreaming
import json

# Initialize FastAPI app
app = FastAPI(
    title="RIB RAG API",
    description="Risk Improvement Benchmark Retrieval-Augmented Generation API",
    version="1.0.0"
)

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'tsdsrd',
    'user': 'amir',
    'password': 'amir123',
    'port': 5432
}

# Initialize RAG system
rag_system = None

@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup"""
    global rag_system
    print("Initializing RIB RAG System...")
    rag_system = RIBRAGSystemStreaming(
        db_config=DB_CONFIG,
        vllm_url="http://localhost:8000/v1/chat/completions",
        embedding_model="BAAI/bge-large-en-v1.5",
        temperature=0.7,
        max_tokens=2048
    )
    print("✓ RAG system initialized")

# Request/Response models
class QueryRequest(BaseModel):
    question: str = Field(..., description="The question to ask")
    num_results: int = Field(5, ge=1, le=20, description="Number of search results to retrieve")
    category_filter: Optional[str] = Field(None, description="Category to filter by")
    similarity_threshold: float = Field(0.3, ge=0.0, le=1.0, description="Minimum similarity score")
    stream: bool = Field(False, description="Whether to stream the response")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Override default temperature")
    max_tokens: Optional[int] = Field(None, ge=1, description="Override default max tokens")

class SourceInfo(BaseModel):
    id: str
    section_number: str
    title: str
    category: str
    similarity: float
    observation: Optional[str] = None
    recommendation: Optional[str] = None

class QueryResponse(BaseModel):
    success: bool
    answer: str
    sources: List[SourceInfo]
    model: str
    usage: Optional[Dict[str, Any]] = None

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RIB RAG API is running",
        "version": "1.0.0",
        "endpoints": {
            "query": "/query",
            "search": "/search",
            "categories": "/categories",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "rag_system": "initialized" if rag_system else "not initialized"
    }

@app.get("/categories")
async def get_categories():
    """Get list of available categories"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    categories = rag_system.search.get_categories()
    return {
        "categories": categories,
        "count": len(categories)
    }

@app.post("/search")
async def search_observations(request: QueryRequest):
    """Search for observations without generating a response"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        results = rag_system.search.search_observations(
            query=request.question,
            limit=request.num_results,
            category_filter=request.category_filter,
            similarity_threshold=request.similarity_threshold
        )
        
        sources = [
            SourceInfo(
                id=r['id'],
                section_number=r['section_number'],
                title=r['title'],
                category=r['category'],
                similarity=r['similarity'],
                observation=r.get('observation'),
                recommendation=r.get('recommendation')
            )
            for r in results
        ]
        
        return {
            "success": True,
            "count": len(sources),
            "sources": sources
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_rag(request: QueryRequest):
    """Query the RAG system (non-streaming)"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        # Get search results
        results = rag_system.search.search_observations(
            query=request.question,
            limit=request.num_results,
            category_filter=request.category_filter,
            similarity_threshold=request.similarity_threshold
        )
        
        if not results:
            return QueryResponse(
                success=False,
                answer="No relevant information found in the database.",
                sources=[],
                model="none"
            )
        
        # Format context
        context = rag_system.format_context_from_results(results)
        
        # Generate response (non-streaming)
        full_response = ""
        for chunk in rag_system.generate_response_stream(request.question, context):
            full_response += chunk
        
        # Format sources
        sources = [
            SourceInfo(
                id=r['id'],
                section_number=r['section_number'],
                title=r['title'],
                category=r['category'],
                similarity=r['similarity'],
                observation=r.get('observation'),
                recommendation=r.get('recommendation')
            )
            for r in results
        ]
        
        return QueryResponse(
            success=True,
            answer=full_response,
            sources=sources,
            model="Qwen/Qwen3-VL-8B-Instruct"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/stream")
async def query_rag_stream(request: QueryRequest):
    """Query the RAG system with streaming response"""
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    try:
        # Get streaming response and sources
        response_stream, sources = rag_system.query_stream(
            question=request.question,
            num_results=request.num_results,
            category_filter=request.category_filter,
            similarity_threshold=request.similarity_threshold
        )
        
        async def generate():
            # First send sources as metadata
            sources_data = [
                {
                    "id": r['id'],
                    "section_number": r['section_number'],
                    "title": r['title'],
                    "category": r['category'],
                    "similarity": r['similarity']
                }
                for r in sources
            ]
            
            # Send sources as first chunk
            yield f"data: {json.dumps({'type': 'sources', 'data': sources_data})}\n\n"
            
            # Stream the response
            for chunk in response_stream:
                yield f"data: {json.dumps({'type': 'content', 'data': chunk})}\n\n"
            
            # Send done signal
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Main
if __name__ == "__main__":
    print("Starting RIB RAG API Server...")
    print("Access the API at: http://localhost:8001")
    print("Interactive docs at: http://localhost:8001/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )