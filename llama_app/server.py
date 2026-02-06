import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor
from PIL import Image
import io
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import psycopg2
from sentence_transformers import SentenceTransformer
import time

print("="*80)
print("Llama-3.2-90B-Vision-Instruct RAG System for TSD RIB")
print("="*80)

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'tsdsrd',
    'user': 'amir',
    'password': 'amir123',
    'port': 5432
}

# Load models
print("\nLoading Llama Vision model (4-bit)...")
model_id = "unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

print(f"✓ Vision model loaded! Memory: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")

# Load embedding model
print("\nLoading embedding model...")
embedding_model_name = "BAAI/bge-large-en-v1.5"
device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = SentenceTransformer(embedding_model_name, device=device)
print(f"✓ Embedding model loaded on {device}")

print("\n" + "="*80)
print("SYSTEM READY")
print("="*80)

app = FastAPI(title="Llama Vision 90B RAG API for TSD RIB")

# Request/Response Models
class ImageDescriptionRequest(BaseModel):
    image_base64: str

class TextSearchRequest(BaseModel):
    query_text: str
    top_k: int = 3
    similarity_threshold: float = 0.4
    category_filter: Optional[str] = None

class RecommendationRequest(BaseModel):
    query_text: Optional[str] = None
    image_base64: Optional[str] = None
    top_k: int = 3
    similarity_threshold: float = 0.4
    category_filter: Optional[str] = None
    max_tokens: int = 2048
    temperature: float = 0.2

# Database functions
def connect_db():
    """Establish database connection"""
    return psycopg2.connect(
        host=DB_CONFIG['host'],
        database=DB_CONFIG['database'],
        user=DB_CONFIG['user'],
        password=DB_CONFIG['password'],
        port=DB_CONFIG['port']
    )

def generate_query_embedding(query: str) -> List[float]:
    """Generate embedding for text query"""
    if "bge" in embedding_model_name.lower():
        query = f"Represent this observation for searching relevant observations: {query}"
    
    embedding = embedding_model.encode(
        query,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    
    return embedding.tolist()

def search_observations(
    query_text: str,
    top_k: int = 5,
    similarity_threshold: float = 0.4,
    category_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Search for similar observations using semantic search"""
    print(f"\n🔎 Searching for matching observations...")
    
    query_embedding = generate_query_embedding(query_text)
    
    conn = connect_db()
    cursor = conn.cursor()
    
    try:
        sql = """
            SELECT 
                r.id,
                r.section_number,
                r.title,
                r.category,
                r.observation,
                r.recommendation,
                r.regulation,
                r.full_text,
                1 - (r.observation_embedding <=> %s::vector) AS similarity,
                m.observation_length,
                m.recommendation_length,
                m.regulation_length
            FROM tsd_rib r
            JOIN tsd_metadata m ON r.id = m.chunk_id
            WHERE m.has_observation = TRUE
              AND 1 - (r.observation_embedding <=> %s::vector) > %s
        """
        params = [query_embedding, query_embedding, similarity_threshold]
        
        if category_filter:
            sql += " AND r.category = %s"
            params.append(category_filter)
        
        sql += " ORDER BY r.observation_embedding <=> %s::vector LIMIT %s"
        params.extend([query_embedding, top_k])
        
        cursor.execute(sql, params)
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row[0],
                'section_number': row[1],
                'title': row[2],
                'category': row[3],
                'observation': row[4],
                'recommendation': row[5],
                'regulation': row[6],
                'full_text': row[7],
                'similarity': float(row[8]),
                'observation_length': row[9],
                'recommendation_length': row[10],
                'regulation_length': row[11]
            })
        
        if results:
            print(f"✓ Found {len(results)} matching observations")
            print(f"  🎯 Top match: Section {results[0]['section_number']} - {results[0]['title']}")
            print(f"     Similarity: {results[0]['similarity']:.3f}")
        else:
            print(f"⚠️  No observations found above similarity threshold {similarity_threshold}")
        
        return results
        
    finally:
        cursor.close()
        conn.close()

def format_context_for_recommendations(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved observations as context"""
    if not chunks:
        return "No relevant RIB documentation found."
    
    context_parts = []
    
    for i, chunk in enumerate(chunks, 1):
        context_part = f"""
{'='*80}
RIB REFERENCE {i}: Section {chunk['section_number']} - {chunk['title']}
Category: {chunk['category']}
Similarity Score: {chunk['similarity']:.3f}
{'='*80}

OBSERVATION:
{chunk['observation']}

RECOMMENDATION:
{chunk['recommendation']}

REGULATION/GUIDELINE:
{chunk['regulation']}
"""
        context_parts.append(context_part)
    
    return "\n".join(context_parts)

# API Endpoints

@app.post("/describe_image")
async def describe_image(request: ImageDescriptionRequest):
    """
    STAGE 1: Describe what's in the image
    """
    try:
        print(f"\n🔍 Stage 1: Describing image...")
        
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_data))
        
        system_prompt = """You are an image analysis assistant for industrial safety inspection. Your task is to describe what you see in the image in clear, objective detail.

When analyzing an image, always describe elements in this order of priority:
1. MOST PROMINENT FEATURE - What is the most striking, unusual, or attention-grabbing element?
2. PRIMARY SUBJECT - What is the main focus or central element of the image?
3. ENVIRONMENTAL CONTEXT - What is the setting, background, or surrounding environment?
4. SECONDARY DETAILS - Any additional objects, features, or characteristics

Provide a factual, objective description. Do not make assumptions about industrial safety, engineering implications, or risk assessments - just describe what you see."""

        user_prompt = """Describe this image systematically:

FIRST: What is the most prominent, striking, or unusual feature in this image? (e.g., weather phenomena, dramatic events, visible damage, unusual conditions)

THEN: Describe the other elements:
- What objects or structures are present
- Their condition and appearance  
- The setting and environment
- Any other notable details

Start with the most eye-catching or significant element, then work through the rest."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(
            images=image,
            text=input_text,
            return_tensors="pt"
        ).to(model.device)
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.3,
                do_sample=True,
            )
        
        end_time = time.time()
        
        description = processor.decode(outputs[0], skip_special_tokens=True)
        
        print(f"✓ Image description generated in {end_time - start_time:.2f}s")
        
        return {
            "description": description,
            "generation_time": end_time - start_time,
            "success": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search_observations")
async def search_observations_endpoint(request: TextSearchRequest):
    """
    STAGE 2: Search for matching observations
    """
    try:
        results = search_observations(
            query_text=request.query_text,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            category_filter=request.category_filter
        )
        
        return {
            "observations": results,
            "count": len(results),
            "success": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_recommendations")
async def generate_recommendations(request: RecommendationRequest):
    """
    STAGE 3: Generate recommendations with full RAG pipeline
    Supports both text-only and image+text queries
    """
    try:
        print(f"\n{'='*80}")
        print("MULTIMODAL RAG PIPELINE - TSD RIB")
        print(f"{'='*80}")
        
        # Determine query mode
        if request.image_base64:
            # Image mode: describe image first
            print("📸 Image query mode")
            image_data = base64.b64decode(request.image_base64)
            image = Image.open(io.BytesIO(image_data))
            
            # Stage 1: Describe image
            system_prompt_stage1 = """You are an image analysis assistant for industrial safety inspection. Your task is to describe what you see in the image in clear, objective detail.

When analyzing an image, always describe elements in this order of priority:
1. MOST PROMINENT FEATURE - What is the most striking, unusual, or attention-grabbing element?
2. PRIMARY SUBJECT - What is the main focus or central element of the image?
3. ENVIRONMENTAL CONTEXT - What is the setting, background, or surrounding environment?
4. SECONDARY DETAILS - Any additional objects, features, or characteristics

Provide a factual, objective description. Do not make assumptions about industrial safety, engineering implications, or risk assessments - just describe what you see."""

            user_prompt_stage1 = """Describe this image systematically:

FIRST: What is the most prominent, striking, or unusual feature in this image? (e.g., weather phenomena, dramatic events, visible damage, unusual conditions)

THEN: Describe the other elements:
- What objects or structures are present
- Their condition and appearance  
- The setting and environment
- Any other notable details

Start with the most eye-catching or significant element, then work through the rest."""

            messages_stage1 = [
                {"role": "system", "content": system_prompt_stage1},
                {"role": "user", "content": user_prompt_stage1}
            ]
            
            input_text = processor.apply_chat_template(messages_stage1, add_generation_prompt=True)
            inputs = processor(
                images=image,
                text=input_text,
                return_tensors="pt"
            ).to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.3,
                    do_sample=True,
                )
            
            image_description = processor.decode(outputs[0], skip_special_tokens=True)
            query_text = image_description
            
            print(f"\n{'='*80}")
            print("IMAGE DESCRIPTION:")
            print(f"{'='*80}")
            print(image_description)
            print(f"{'='*80}\n")
        else:
            # Text mode
            print("📝 Text query mode")
            query_text = request.query_text
            image_description = None
            image = None
        
        # Stage 2: Search observations
        matching_chunks = search_observations(
            query_text=query_text,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            category_filter=request.category_filter
        )
        
        if not matching_chunks:
            return {
                "answer": "No matching RIB documentation found for the described issue.",
                "num_sources": 0,
                "success": False
            }
        
        # Stage 3: Generate recommendations
        context = format_context_for_recommendations(matching_chunks)
        
        system_prompt_stage3 = """You are an expert industrial safety consultant providing Risk Improvement Benchmark (RIB) recommendations.

Your role is to:
1. Understand the safety/structural issue described
2. Provide specific, actionable recommendations from the matched RIB sections
3. Cite applicable regulations and guidelines
4. Be precise and reference specific section numbers

Structure your response as:
- WHAT I SEE (for images): Confirm what's visible and identify key safety elements
- MATCHED RIB SECTION: Which section(s) apply and why
- SPECIFIC RECOMMENDATIONS: Actionable steps from the RIB documentation
- APPLICABLE REGULATIONS: Relevant standards and guidelines
- PRIORITY ACTIONS: Most critical steps to take immediately"""

        if request.image_base64:
            user_prompt_stage3 = f"""Based on the image and the retrieved RIB documentation below, provide specific RIB recommendations and regulations.

IMAGE DESCRIPTION (from initial analysis):
{image_description}

{context}

QUESTION: Is there any lightning arrester or lightning event counter in the image? If yes, provide me with the specific RIB recommendations and regulations.

Instructions:
1. Verify the image content matches the RIB observations above
2. Provide detailed recommendations from the matched RIB sections
3. Cite specific regulations and guidelines mentioned
4. Be actionable and specific - reference exact requirements (e.g., testing frequencies, standards)
5. If multiple sections apply, explain which is most relevant and why"""
        else:
            user_prompt_stage3 = f"""Based on the situation described and the retrieved RIB documentation below, provide specific RIB recommendations and regulations.

USER DESCRIPTION:
{query_text}

{context}

QUESTION: Provide me with the specific RIB recommendations and regulations that apply to this situation.

Instructions:
1. Analyze how the description matches the RIB observations above
2. Provide detailed recommendations from the matched RIB sections
3. Cite specific regulations and guidelines mentioned
4. Be actionable and specific - reference exact requirements (e.g., testing frequencies, standards)
5. If multiple sections apply, explain which is most relevant and why"""

        messages_stage3 = [
            {"role": "system", "content": system_prompt_stage3},
            {"role": "user", "content": user_prompt_stage3}
        ]
        
        input_text = processor.apply_chat_template(messages_stage3, add_generation_prompt=True)
        inputs = processor(
            images=image if request.image_base64 else None,
            text=input_text,
            return_tensors="pt"
        ).to(model.device)
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                do_sample=True,
            )
        
        end_time = time.time()
        
        recommendations = processor.decode(outputs[0], skip_special_tokens=True)
        
        print(f"✓ Recommendations generated in {end_time - start_time:.2f}s")
        
        result = {
            "answer": recommendations,
            "num_sources": len(matching_chunks),
            "generation_time": end_time - start_time,
            "success": True,
            "sources": [
                {
                    'section': chunk['section_number'],
                    'title': chunk['title'],
                    'category': chunk['category'],
                    'similarity': round(chunk['similarity'], 3)
                }
                for chunk in matching_chunks
            ]
        }
        
        if image_description:
            result["image_description"] = image_description
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models():
    return {"data": [{"id": model_id, "object": "model"}]}

@app.get("/health")
async def health():
    return {"status": "healthy", "model": model_id}

if __name__ == "__main__":
    print("\nStarting API server on http://0.0.0.0:8000")
    print("Available endpoints:")
    print("  POST /describe_image - Describe image content")
    print("  POST /search_observations - Search RIB observations")
    print("  POST /generate_recommendations - Full RAG pipeline")
    print("  GET  /v1/models - List available models")
    print("  GET  /health - Health check")
    uvicorn.run(app, host="0.0.0.0", port=8000)