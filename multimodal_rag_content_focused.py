# multimodal_rag_vllm_focused.py

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
        owlv2_url: str = "http://localhost:8010",
        embedding_model_name: str = "BAAI/bge-large-en-v1.5"
    ):
        """Initialize Multimodal RAG system with VLLM server and OWLv2 detection"""
        self.db_config = db_config
        self.vllm_url = vllm_url.rstrip('/')
        self.owlv2_url = owlv2_url.rstrip('/')
        self.embedding_model_name = embedding_model_name
        
        # VLLM endpoints
        self.chat_url = f"{self.vllm_url}/v1/chat/completions"
        self.models_url = f"{self.vllm_url}/v1/models"
        
        # OWLv2 endpoints
        self.owlv2_detect_url = f"{self.owlv2_url}/detect-top5"
        
        # Test connections
        print(f"Testing VLLM connection at {self.vllm_url}...")
        self._test_vllm_connection()
        
        print(f"\nTesting OWLv2 connection at {self.owlv2_url}...")
        self._test_owlv2_connection()
        
        # Load embedding model
        print(f"\nLoading embedding model: {embedding_model_name}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedding_model = SentenceTransformer(embedding_model_name, device=device)
        print(f"✓ Embedding model loaded on {device}")
        
    def _test_vllm_connection(self):
        """Test connection to VLLM server"""
        try:
            health_response = requests.get(f"{self.vllm_url}/health", timeout=5)
            if health_response.status_code != 200:
                raise ConnectionError(f"Health check failed: {health_response.status_code}")
            
            print(f"✓ VLLM server is healthy")
            
            response = requests.get(self.models_url, timeout=5)
            response.raise_for_status()
            models = response.json()
            
            if 'data' in models and len(models['data']) > 0:
                self.model_name = models['data'][0]['id']
                print(f"✓ Connected to VLLM server")
                print(f"✓ Available model: {self.model_name}")
            else:
                self.model_name = "Qwen/Qwen3-VL-32B-Instruct-FP8"
                print(f"✓ Using model: {self.model_name}")
            
            return True
            
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"❌ Cannot connect to VLLM server at {self.vllm_url}\n"
                f"Make sure VLLM is running: docker-compose ps vllm-qwen"
            )
        except Exception as e:
            raise Exception(f"❌ Error connecting to VLLM: {e}")
    
    def _test_owlv2_connection(self):
        """Test connection to OWLv2 detection server"""
        try:
            health_response = requests.get(f"{self.owlv2_url}/health", timeout=10)
            if health_response.status_code == 200:
                data = health_response.json()
                if data.get('detector_ready', False):
                    print(f"✓ OWLv2 server is healthy and detector is ready")
                else:
                    print(f"⚠️ OWLv2 server is running but detector not ready")
            else:
                print(f"⚠️ OWLv2 health check returned: {health_response.status_code}")
            
            self.owlv2_available = True
            return True
            
        except requests.exceptions.ConnectionError:
            print(f"⚠️ Cannot connect to OWLv2 server at {self.owlv2_url}")
            print(f"   Object detection will be skipped, falling back to semantic search only")
            self.owlv2_available = False
            return False
        except Exception as e:
            print(f"⚠️ Error connecting to OWLv2: {e}")
            self.owlv2_available = False
            return False
    
    def detect_objects_in_image(self, image_path: str) -> Dict[str, Any]:
        """
        Send image to OWLv2 API for object detection.
        Returns top 5 RIB subsection detections.
        """
        if not self.owlv2_available:
            return {'success': False, 'error': 'OWLv2 not available'}
        
        try:
            print(f"\n🔍 Sending image to OWLv2 for object detection...")
            
            with open(image_path, 'rb') as f:
                files = {'file': (Path(image_path).name, f, 'image/png')}
                response = requests.post(
                    self.owlv2_detect_url,
                    files=files,
                    timeout=60
                )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    data = result.get('data', {})
                    print(f"✓ OWLv2 detection completed")
                    print(f"  🎯 Tier 1 objects detected: {data.get('tier1_detection_count', 0)}")
                    print(f"  📊 Broad categories: {', '.join(data.get('broad_categories', []))}")
                    
                    # Print detected tier1 objects
                    tier1_objects = data.get('tier1_objects', [])
                    if tier1_objects:
                        print(f"\n  📦 Top Detected Objects:")
                        for i, obj in enumerate(tier1_objects[:5], 1):
                            print(f"     {i}. {obj['object']} (confidence: {obj['confidence']:.1%}, category: {obj['category']})")
                    
                    # Print top 5 RIB subsections
                    top5 = data.get('top5_rib_subsections', [])
                    if top5:
                        print(f"\n  🏆 Top 5 RIB Subsections from Object Detection:")
                        print(f"  " + "-"*60)
                        for sub in top5:
                            print(f"     #{sub['rank']}: {sub['section']}")
                            print(f"         Score: {sub['score']:.3f} | Detections: {sub['detection_count']}")
                        print(f"  " + "-"*60)
                    else:
                        print(f"  ⚠️ No RIB subsections detected")
                    
                    return {
                        'success': True,
                        'data': data
                    }
                else:
                    return {'success': False, 'error': result.get('message', 'Unknown error')}
            else:
                return {'success': False, 'error': f'HTTP {response.status_code}'}
                
        except requests.exceptions.Timeout:
            print(f"⚠️ OWLv2 detection timed out")
            return {'success': False, 'error': 'timeout'}
        except Exception as e:
            print(f"⚠️ OWLv2 detection error: {e}")
            return {'success': False, 'error': str(e)}
    
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
    
    def format_context(self, chunks: List[Dict[str, Any]], focus_on_top: bool = True) -> str:
        """Format retrieved chunks into context string, optionally focusing on top matches"""
        if not chunks:
            return "No relevant context found in the RIB documentation."
        
        # If focus_on_top, only use chunks with similarity > 0.7
        if focus_on_top:
            relevant_chunks = [c for c in chunks if c['similarity'] > 0.7]
            if not relevant_chunks:
                # If no high-similarity chunks, use top 2
                relevant_chunks = chunks[:2]
        else:
            relevant_chunks = chunks
        
        context_parts = []
        for i, chunk in enumerate(relevant_chunks, 1):
            context_parts.append(
                f"[Context {i} - Section {chunk['section_number']}: {chunk['title']}]\n"
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
        owlv2_detections: Optional[Dict[str, Any]] = None,
        max_tokens: int = 2048,
        temperature: float = 0.3,
        top_p: float = 0.9
    ) -> Dict[str, Any]:
        """Generate response using VLLM server with retrieved context and OWLv2 detections"""
        
        # Focus on most relevant chunks only
        context = self.format_context(context_chunks, focus_on_top=True)
        
        # Get the top matching section for focused analysis
        top_section = ""
        if context_chunks:
            top_chunk = context_chunks[0]
            top_section = f"Section {top_chunk['section_number']}: {top_chunk['title']}"
        
        # Build OWLv2 detection context if available
        owlv2_context = ""
        if owlv2_detections and owlv2_detections.get('success'):
            data = owlv2_detections.get('data', {})
            top5 = data.get('top5_rib_subsections', [])
            tier1_objects = data.get('tier1_objects', [])
            
            if top5 or tier1_objects:
                owlv2_context = "\n\n=== OBJECT DETECTION RESULTS (OWLv2) ===\n"
                
                if tier1_objects:
                    owlv2_context += "Detected Objects in Image:\n"
                    for obj in tier1_objects[:5]:
                        owlv2_context += f"  - {obj['object']} (confidence: {obj['confidence']:.1%}, category: {obj['category']})\n"
                
                if top5:
                    owlv2_context += "\nTop 5 RIB Subsections Detected by Object Analysis:\n"
                    for sub in top5:
                        owlv2_context += f"  #{sub['rank']}: {sub['section']} (score: {sub['score']}, detections: {sub['detection_count']})\n"
                
                owlv2_context += "========================================\n"
        
        # More focused system prompt with OWLv2 integration
        system_prompt = """You are an expert risk management and industrial safety consultant specializing in the Risk Improvement Benchmark (RIB) documentation.

You have access to two sources of information:
1. OBJECT DETECTION RESULTS: An AI vision system has analyzed the image and detected specific objects and suggested RIB subsections
2. SEMANTIC SEARCH RESULTS: Text-based matching against the RIB documentation database

CRITICAL INSTRUCTIONS:
1. First, review the OBJECT DETECTION RESULTS to see what objects were visually detected in the image
2. Compare the detected RIB subsections from object detection with the semantic search results
3. Determine which RIB section BEST matches what is actually visible in the image
4. If NONE of the top 5 detected subsections seem to fit, describe what you see and explain why
5. ONLY reference RIB sections that directly match the image content
6. Be concise and specific - no general analysis of unrelated topics

Your response should follow this structure:
- WHAT I SEE: Describe only what's visible in the image
- OBJECT DETECTION ANALYSIS: Evaluate which of the top 5 detected subsections best fits
- SELECTED RIB SECTION: State the ONE most relevant section and why
- SPECIFIC OBSERVATIONS: Link visual evidence to that section's criteria
- ACTIONABLE RECOMMENDATIONS: Give specific steps from that section only
- APPLICABLE REGULATIONS: Cite only regulations mentioned in that section"""

        # Build user prompt with both OWLv2 and semantic search context
        if context_chunks:
            # Extract only the most relevant section info
            top_match = context_chunks[0]
            focused_context = f"""
SEMANTIC SEARCH RESULT - MOST RELEVANT RIB SECTION (Similarity: {top_match['similarity']:.3f}):
Section {top_match['section_number']}: {top_match['title']}
Category: {top_match['category']}
Risk Type: {top_match['risk_type']}

CONTENT:
{top_match['content']}
"""
            
            user_prompt = f"""Analyze this image using both object detection and semantic search results.
{owlv2_context}
{focused_context}

USER QUESTION: {query}

INSTRUCTIONS:
1. First, look at the OBJECT DETECTION RESULTS to see what objects were detected in the image
2. Review the top 5 RIB subsections suggested by object detection
3. Compare with the semantic search result above
4. Determine which RIB section BEST matches what you can see in the image
5. If the object detection suggests a different section than semantic search, explain which one fits better and why
6. If NONE of the detected subsections fit the image well, describe the image and suggest the correct RIB section
7. Provide recommendations from ONLY the section that best matches the image

Provide specific, actionable recommendations based on the best matching section."""
        else:
            user_prompt = f"""Analyze this image and identify the risk or safety issue shown.
{owlv2_context}

USER QUESTION: {query}

No specific RIB documentation was found via semantic search. 
Use the object detection results above to guide your analysis.
If none of the detected subsections fit, describe what you see and recommend general safety principles."""
        
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
            print(f"   Focusing on: {top_section}")
            
            start_time = time.time()
            
            response = requests.post(
                self.chat_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=180
            )
            
            end_time = time.time()
            total_time = end_time - start_time
            
            response.raise_for_status()
            result = response.json()
            
            generated_text = result['choices'][0]['message']['content']
            usage = result.get('usage', {})
            completion_tokens = usage.get('completion_tokens', 0)
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
                'text': "❌ Request timed out.",
                'error': 'timeout'
            }
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP Error {response.status_code}: {response.text[:500]}"
            return {
                'text': f"❌ Error: {error_msg}",
                'error': error_msg
            }
        except Exception as e:
            return {
                'text': f"❌ Unexpected error: {str(e)}",
                'error': str(e)
            }
    
    def query(
        self, question: str, image_path: Optional[str] = None,
        top_k: int = 5, category_filter: Optional[str] = None,
        max_tokens: int = 1500, temperature: float = 0.2,
        show_sources: bool = True, similarity_threshold: float = 0.5,
        use_object_detection: bool = True
    ) -> Dict[str, Any]:
        """Complete RAG pipeline: OWLv2 detection + semantic search + generate response"""
        
        # Step 1: Object detection with OWLv2 (if image provided)
        owlv2_detections = None
        if image_path and Path(image_path).exists() and use_object_detection:
            owlv2_detections = self.detect_objects_in_image(image_path)
        
        # Step 2: Semantic search
        print(f"\n🔍 Searching for relevant RIB documentation...")
        
        # If OWLv2 detected subsections, use the primary one to enhance the search
        enhanced_query = question
        if owlv2_detections and owlv2_detections.get('success'):
            data = owlv2_detections.get('data', {})
            primary_section = data.get('primary_rib_subsection')
            tier1_objects = data.get('tier1_objects', [])
            
            # Enhance query with detected objects
            if tier1_objects:
                detected_objects = ' '.join([obj['object'] for obj in tier1_objects[:3]])
                enhanced_query = f"{question} {detected_objects}"
                print(f"  📝 Enhanced query with detected objects: {detected_objects}")
        
        chunks = self.search_similar_chunks(
            query=enhanced_query, top_k=top_k,
            category_filter=category_filter,
            similarity_threshold=similarity_threshold
        )
        
        if chunks:
            print(f"✓ Found {len(chunks)} relevant sections from semantic search")
            print(f"  🎯 Top match: Section {chunks[0]['section_number']}: {chunks[0]['title']} (similarity: {chunks[0]['similarity']:.3f})")
            
            # Only show additional matches if they're also highly relevant
            high_relevance = [c for c in chunks[1:3] if c['similarity'] > 0.7]
            if high_relevance:
                for i, chunk in enumerate(high_relevance, 2):
                    print(f"     #{i}: Section {chunk['section_number']}: {chunk['title']} (similarity: {chunk['similarity']:.3f})")
        else:
            print(f"⚠️  No highly relevant sections found via semantic search (threshold: {similarity_threshold})")
        
        # Step 3: Generate response with both OWLv2 and semantic search context
        generation_result = self.generate_response(
            query=question, context_chunks=chunks,
            image_path=image_path, 
            owlv2_detections=owlv2_detections,
            max_tokens=max_tokens,
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
            # Only show highly relevant sources
            relevant_sources = [c for c in chunks if c['similarity'] > 0.7] or chunks[:1]
            response['sources'] = [
                {
                    'section': chunk['section_number'],
                    'title': chunk['title'],
                    'category': chunk['category'],
                    'similarity': round(chunk['similarity'], 3),
                    'risk_type': chunk['risk_type'],
                    'regulations': chunk['regulations'][:3] if chunk['regulations'] else []
                }
                for chunk in relevant_sources
            ]
        
        # Include OWLv2 detection results in response
        if owlv2_detections and owlv2_detections.get('success'):
            data = owlv2_detections.get('data', {})
            response['owlv2_detection'] = {
                'tier1_objects': data.get('tier1_objects', [])[:5],
                'top5_rib_subsections': data.get('top5_rib_subsections', []),
                'primary_rib_subsection': data.get('primary_rib_subsection'),
                'broad_categories': data.get('broad_categories', [])
            }
        
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
    print("MULTIMODAL RAG SYSTEM WITH OWLv2 OBJECT DETECTION")
    print("="*80)
    
    try:
        rag = MultimodalRAGSystemVLLM(
            db_config=db_config,
            vllm_url="http://localhost:8000",
            owlv2_url="http://localhost:8010",
            embedding_model_name="BAAI/bge-large-en-v1.5"
        )
    except Exception as e:
        print(f"\n❌ Failed to initialize: {e}")
        return
    
    print("\n" + "="*80)
    print("SYSTEM READY")
    print("="*80)
    
    # Test with subsidence image
    image_path = "/home/amir/Desktop/MRE TSD/RIB_images/page15_img1.png"
    question = "Analyze this image and provide specific RIB recommendations for what you observe."
    
    print(f"\n📝 Question: {question}")
    print(f"🖼️  Image: {image_path}\n")
    
    result = rag.query(
        question=question,
        image_path=image_path,
        top_k=5,
        temperature=0.2,
        similarity_threshold=0.4,
        show_sources=True,
        use_object_detection=True  # Enable OWLv2 detection
    )
    
    if 'error' not in result:
        print(f"\n💡 Answer:\n{result['answer']}\n")
        
        print(f"⚡ Performance:")
        print(f"   Time: {result.get('generation_time', 0):.2f}s")
        print(f"   Tokens/sec: {result.get('tokens_per_second', 0):.2f}")
        
        # Show OWLv2 detection results
        if result.get('owlv2_detection'):
            owlv2 = result['owlv2_detection']
            print(f"\n🔍 OWLv2 Object Detection Results:")
            if owlv2.get('tier1_objects'):
                print(f"   Detected objects: {[obj['object'] for obj in owlv2['tier1_objects']]}")
            if owlv2.get('top5_rib_subsections'):
                print(f"   Top 5 RIB subsections:")
                for sub in owlv2['top5_rib_subsections']:
                    print(f"     #{sub['rank']}: {sub['section']} (score: {sub['score']})")
        
        if result.get('sources'):
            print(f"\n📚 Semantic Search Sources ({len(result['sources'])}):")
            for i, source in enumerate(result['sources'], 1):
                print(f"  {i}. Section {source['section']}: {source['title']}")
                print(f"     Similarity: {source['similarity']} | Category: {source['category']}")


if __name__ == "__main__":
    main()