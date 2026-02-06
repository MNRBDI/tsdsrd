# multimodal_rag_vllm_serve.py

import psycopg2
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import torch
import requests
import json
import base64
from pathlib import Path
import time

class MultimodalRAGSystemVLLM:
    def __init__(
        self, 
        db_config: Dict[str, str], 
        vllm_url: str = "http://localhost:8000",
        embedding_model_name: str = "BAAI/bge-large-en-v1.5"
    ):
        """Initialize Multimodal RAG system with VLLM server"""
        self.db_config = db_config
        self.vllm_url = vllm_url.rstrip('/')
        self.embedding_model_name = embedding_model_name
        
        # VLLM endpoints
        self.chat_url = f"{self.vllm_url}/v1/chat/completions"
        self.models_url = f"{self.vllm_url}/v1/models"
        
        # Test connection and get model info
        print(f"Testing VLLM connection at {self.vllm_url}...")
        self._test_vllm_connection()
        
        # Load embedding model
        print(f"\nLoading embedding model: {embedding_model_name}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = SentenceTransformer(embedding_model_name, device=device)
        print(f"✓ Embedding model loaded on {device}")
        
    def _test_vllm_connection(self):
        """Test connection to VLLM server"""
        try:
            # Test health
            health_response = requests.get(f"{self.vllm_url}/health", timeout=5)
            if health_response.status_code != 200:
                raise ConnectionError(f"Health check failed: {health_response.status_code}")
            
            print(f"✓ VLLM server is healthy")
            
            # Get available models
            response = requests.get(self.models_url, timeout=5)
            response.raise_for_status()
            models = response.json()
            
            if 'data' in models and len(models['data']) > 0:
                self.model_name = models['data'][0]['id']
                print(f"✓ Connected to VLLM server")
                print(f"✓ Available model: {self.model_name}")
            else:
                # Fallback to expected model name
                self.model_name = "Qwen/Qwen3-VL-8B-Instruct"
                print(f"✓ Using model: {self.model_name}")
            
            return True
            
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"❌ Cannot connect to VLLM server at {self.vllm_url}\n"
                f"Make sure VLLM is running: docker-compose ps vllm-qwen"
            )
        except Exception as e:
            raise Exception(f"❌ Error connecting to VLLM: {e}")
    
    def connect_db(self):
        """Establish database connection"""
        return psycopg2.connect(
            host=self.db_config['host'],
            database=self.db_config['database'],
            user=self.db_config['user'],
            password=self.db_config['password'],
            port=self.db_config.get('port', 5432)
        )
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for text query"""
        if "bge" in self.embedding_model_name.lower():
            query = f"Represent this sentence for searching relevant passages: {query}"
        
        embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        return embedding.tolist()
    
    def search_similar_chunks(
        self, 
        query: str, 
        top_k: int = 5, 
        similarity_threshold: float = 0.3,
        category_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks using vector similarity"""
        query_embedding = self.generate_query_embedding(query)
        
        conn = self.connect_db()
        cursor = conn.cursor()
        
        try:
            sql = """
                SELECT 
                    c.id, c.category, c.title, c.content,
                    1 - (c.embedding <=> %s::vector) AS similarity,
                    m.section_number, m.regulations, m.keywords, m.risk_type
                FROM rib_chunks c
                LEFT JOIN chunk_metadata m ON c.id = m.chunk_id
                WHERE 1 - (c.embedding <=> %s::vector) > %s
            """
            params = [query_embedding, query_embedding, similarity_threshold]
            
            if category_filter:
                sql += " AND c.category = %s"
                params.append(category_filter)
            
            sql += " ORDER BY c.embedding <=> %s::vector LIMIT %s"
            params.extend([query_embedding, top_k])
            
            cursor.execute(sql, params)
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'id': row[0], 'category': row[1], 'title': row[2],
                    'content': row[3], 'similarity': float(row[4]),
                    'section_number': row[5], 'regulations': row[6],
                    'keywords': row[7], 'risk_type': row[8]
                })
            
            return results
        finally:
            cursor.close()
            conn.close()
    
    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into context string"""
        if not chunks:
            return "No relevant context found in the RIB documentation."
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Context {i} - Section {chunk['section_number']}: {chunk['title']}]\n"
                f"Category: {chunk['category']}\n"
                f"Risk Type: {chunk['risk_type']}\n"
                f"Similarity Score: {chunk['similarity']:.3f}\n\n"
                f"{chunk['content']}\n"
            )
        
        return "\n" + "="*80 + "\n".join(context_parts) + "="*80 + "\n"
    
    def image_to_base64(self, image_path: str) -> str:
        """Convert image to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def get_image_url(self, image_path: str) -> str:
        """Convert image to data URL for VLLM"""
        image_ext = Path(image_path).suffix.lower()
        mime_types = {
            '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
            '.png': 'image/png', '.gif': 'image/gif', '.webp': 'image/webp'
        }
        mime_type = mime_types.get(image_ext, 'image/jpeg')
        base64_image = self.image_to_base64(image_path)
        return f"data:{mime_type};base64,{base64_image}"
    
    def generate_response(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        image_path: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
        top_p: float = 0.9
    ) -> Dict[str, Any]:
        """Generate response using VLLM server with retrieved context"""
        
        context = self.format_context(context_chunks)
        
        system_prompt = """You are an expert risk management and industrial safety consultant specializing in the Risk Improvement Benchmark (RIB) documentation. Your role is to provide accurate, detailed, and actionable recommendations.

Guidelines:
1. Base your responses strictly on the provided context from the RIB documentation
2. Reference specific section numbers when citing recommendations
3. Include relevant regulations and guidelines when applicable
4. Provide clear, actionable steps and recommendations
5. If the context doesn't contain sufficient information, clearly state this
6. When analyzing images, relate your observations to safety risks and RIB recommendations
7. Be precise and technical when discussing safety measures
8. Structure your response clearly with proper sections if needed"""

        if context_chunks:
            user_prompt = f"""Based on the following excerpts from the Risk Improvement Benchmark (RIB) documentation, please answer the question.

RETRIEVED CONTEXT FROM RIB DOCUMENTATION:
{context}

USER QUESTION: {query}

Please provide a comprehensive answer based on the context above. Make sure to:
- Reference specific section numbers from the context
- Cite relevant regulations mentioned in the context
- Provide actionable recommendations
- Be specific and detailed in your response"""
        else:
            user_prompt = f"""USER QUESTION: {query}

Note: No specific RIB documentation context was found for this query. Please answer based on general risk management and safety principles, and clearly indicate that specific RIB recommendations should be consulted for detailed guidance."""
        
        # Prepare messages for VLLM
        messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]
        
        # Add user message with optional image
        if image_path and Path(image_path).exists():
            image_url = self.get_image_url(image_path)
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    },
                    {
                        "type": "text",
                        "text": user_prompt
                    }
                ]
            })
        else:
            messages.append({
                "role": "user",
                "content": user_prompt
            })
        
        # Prepare request payload
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False
        }
        
        # Make request to VLLM server
        try:
            print(f"🤖 Sending request to VLLM server...")
            print(f"   URL: {self.chat_url}")
            print(f"   Model: {self.model_name}")
            
            start_time = time.time()
            
            response = requests.post(
                self.chat_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=180  # 3 minute timeout for long generations
            )
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Debug: Print response details
            print(f"   Response status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"   Response body: {response.text[:500]}")
            
            response.raise_for_status()
            result = response.json()
            
            # Extract generated text
            generated_text = result['choices'][0]['message']['content']
            
            # Extract usage statistics
            usage = result.get('usage', {})
            completion_tokens = usage.get('completion_tokens', 0)
            
            # Calculate tokens per second
            tokens_per_second = completion_tokens / total_time if total_time > 0 else 0
            
            return {
                'text': generated_text,
                'usage': usage,
                'model': result.get('model', self.model_name),
                'generation_time': total_time,
                'tokens_per_second': tokens_per_second
            }
            
        except requests.exceptions.Timeout:
            return {
                'text': "❌ Request timed out. The model might be processing a complex query or the server is busy.",
                'error': 'timeout'
            }
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP Error {response.status_code}: {response.text[:500]}"
            return {
                'text': f"❌ Error communicating with VLLM server: {error_msg}",
                'error': error_msg
            }
        except requests.exceptions.RequestException as e:
            return {
                'text': f"❌ Error communicating with VLLM server: {str(e)}",
                'error': str(e)
            }
        except Exception as e:
            return {
                'text': f"❌ Unexpected error: {str(e)}",
                'error': str(e)
            }
    
    def query(
        self, question: str, image_path: Optional[str] = None,
        top_k: int = 5, category_filter: Optional[str] = None,
        max_tokens: int = 2048, temperature: float = 0.3,
        show_sources: bool = True, similarity_threshold: float = 0.3
    ) -> Dict[str, Any]:
        """Complete RAG pipeline: search + generate response"""
        
        print(f"🔍 Searching for relevant information...")
        
        chunks = self.search_similar_chunks(
            query=question, top_k=top_k,
            category_filter=category_filter,
            similarity_threshold=similarity_threshold
        )
        
        if chunks:
            print(f"✓ Found {len(chunks)} relevant sections")
            for i, chunk in enumerate(chunks[:3], 1):
                print(f"  {i}. Section {chunk['section_number']}: {chunk['title']} (similarity: {chunk['similarity']:.3f})")
        else:
            print(f"⚠️  No highly relevant sections found (threshold: {similarity_threshold})")
        
        generation_result = self.generate_response(
            query=question, context_chunks=chunks,
            image_path=image_path, max_tokens=max_tokens,
            temperature=temperature
        )
        
        if 'error' in generation_result:
            return {
                'answer': generation_result['text'],
                'num_sources': len(chunks),
                'has_image': image_path is not None and Path(image_path).exists(),
                'error': generation_result['error']
            }
        
        tps = generation_result.get('tokens_per_second', 0)
        gen_time = generation_result.get('generation_time', 0)
        completion_tokens = generation_result.get('usage', {}).get('completion_tokens', 0)
        
        print(f"✓ Response generated")
        print(f"  ⏱️  Generation time: {gen_time:.2f}s")
        print(f"  📝 Completion tokens: {completion_tokens}")
        print(f"  ⚡ Tokens/second: {tps:.2f}")
        
        response = {
            'answer': generation_result['text'],
            'num_sources': len(chunks),
            'has_image': image_path is not None and Path(image_path).exists(),
            'usage': generation_result.get('usage', {}),
            'model': generation_result.get('model', 'unknown'),
            'generation_time': gen_time,
            'tokens_per_second': tps
        }
        
        if show_sources and chunks:
            response['sources'] = [
                {
                    'section': chunk['section_number'],
                    'title': chunk['title'],
                    'category': chunk['category'],
                    'similarity': round(chunk['similarity'], 3),
                    'risk_type': chunk['risk_type'],
                    'regulations': chunk['regulations'][:3] if chunk['regulations'] else []
                }
                for chunk in chunks
            ]
        
        return response


def main():
    """Example usage"""
    
    db_config = {
        'host': 'localhost',
        'database': 'tsdsrd',
        'user': 'amir',
        'password': 'amir123',
        'port': 5432
    }
    
    print("="*80)
    print("MULTIMODAL RAG SYSTEM WITH VLLM SERVER (vllm serve)")
    print("="*80)
    
    try:
        rag = MultimodalRAGSystemVLLM(
            db_config=db_config,
            vllm_url="http://localhost:8000",
            embedding_model_name="BAAI/bge-large-en-v1.5"
        )
    except Exception as e:
        print(f"\n❌ Failed to initialize: {e}")
        return
    
    print("\n" + "="*80)
    print("SYSTEM READY")
    print("="*80)
    
    question = "What are the safety recommendations for LPG gas cylinder storage?"
    print(f"\n📝 Question: {question}\n")
    
    result = rag.query(
        question=question,
        top_k=3,
        temperature=0.3,
        show_sources=True
    )
    
    if 'error' not in result:
        print(f"\n💡 Answer:\n{result['answer']}\n")
        
        print(f"⚡ Performance Metrics:")
        print(f"   Generation Time: {result.get('generation_time', 0):.2f}s")
        print(f"   Tokens/Second: {result.get('tokens_per_second', 0):.2f}")
        
        if 'usage' in result:
            usage = result['usage']
            print(f"\n📊 Token Usage:")
            print(f"   Prompt Tokens: {usage.get('prompt_tokens', 0)}")
            print(f"   Completion Tokens: {usage.get('completion_tokens', 0)}")
            print(f"   Total Tokens: {usage.get('total_tokens', 0)}")
        
        if result.get('sources'):
            print(f"\n📚 Sources ({result['num_sources']}):")
            for i, source in enumerate(result['sources'], 1):
                print(f"  {i}. Section {source['section']}: {source['title']}")
                print(f"     Category: {source['category']} | Similarity: {source['similarity']}")


if __name__ == "__main__":
    main()