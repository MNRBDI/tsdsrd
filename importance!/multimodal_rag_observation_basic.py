import psycopg2
from psycopg2 import pool
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import torch
import requests
import json
import base64
from pathlib import Path
import time
from PIL import Image
import io
from functools import lru_cache

class MultimodalRAGSystemTSDRIB:
    def __init__(
        self, 
        db_config: Dict[str, str], 
        vllm_url: str = "http://localhost:8000",
        owlv2_url: str = "http://localhost:8010",
        embedding_model_name: str = "BAAI/bge-large-en-v1.5",
        max_image_size: int = 1536
    ):
        """Initialize Multimodal RAG system with VLLM server and OWLv2 detection for TSD RIB database"""
        self.db_config = db_config
        self.vllm_url = vllm_url.rstrip('/')
        self.owlv2_url = owlv2_url.rstrip('/')
        self.embedding_model_name = embedding_model_name
        self.max_image_size = max_image_size
        
        # Create connection pool for database
        print(f"Creating database connection pool...")
        self.db_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=5,
            host=db_config['host'],
            database=db_config['database'],
            user=db_config['user'],
            password=db_config['password'],
            port=db_config.get('port', 5432)
        )
        print(f"✓ Database connection pool created")
        
        # VLLM endpoints
        self.chat_url = f"{self.vllm_url}/v1/chat/completions"
        self.models_url = f"{self.vllm_url}/v1/models"
        
        # OWLv2 endpoints
        self.owlv2_detect_url = f"{self.owlv2_url}/detect-top5"
        
        # Image caching for multi-stage pipeline
        self._cached_image_url = None
        
        # Test connections
        print(f"Testing VLLM connection at {self.vllm_url}...")
        self._test_vllm_connection()
        
        print(f"\nTesting OWLv2 connection at {self.owlv2_url}...")
        self._test_owlv2_connection()
        
        # Load embedding model
        print(f"\nLoading embedding model: {embedding_model_name}")
        # Force CPU mode to avoid cuBLAS initialization issues on GPU
        device = "cpu"
        self.embedding_model = SentenceTransformer(embedding_model_name, device=device)
        print(f"✓ Embedding model loaded on {device} (forced for CUDA compatibility)")
        
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
                self.model_name = "Qwen/Qwen2-VL-7B-Instruct"
                print(f"✓ Using model: {self.model_name}")
            
            return True
            
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"❌ Cannot connect to VLLM server at {self.vllm_url}\n"
                f"Make sure VLLM is running with: docker-compose up -d vllm"
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
            print(f"   Object detection will be skipped, falling back to image description only")
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
        Uses original image (not resized) for better compatibility with OWLv2.
        """
        if not self.owlv2_available:
            return {'success': False, 'error': 'OWLv2 not available'}
        
        try:
            print(f"\n🔍 Sending image to OWLv2 for object detection...")
            
            # Send original image to OWLv2 (avoiding format issues from resizing)
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
                    print(f"  🎯 Tier 1 detections: {data.get('tier1_detection_count', 0)}")
                    print(f"  📊 Broad categories: {', '.join(data.get('broad_categories', []))}")
                    
                    # Get top 5 RIB subsections (already formatted by /detect-top5 endpoint)
                    top5_subsections = data.get('top5_rib_subsections', [])
                    
                    # Print top 5 RIB subsections
                    if top5_subsections:
                        print(f"\n  🏆 Top 5 RIB Subsections from Object Detection:")
                        print(f"  " + "-"*60)
                        for sub in top5_subsections:
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
                # Handle HTTP errors with more detail
                error_msg = f'HTTP {response.status_code}'
                try:
                    error_detail = response.json()
                    if 'detail' in error_detail:
                        error_msg += f": {error_detail['detail']}"
                except:
                    if response.text:
                        error_msg += f": {response.text[:200]}"
                
                print(f"❌ OWLv2 API error: {error_msg}")
                return {'success': False, 'error': error_msg}
                
        except requests.exceptions.Timeout:
            print(f"⚠️ OWLv2 detection timed out")
            return {'success': False, 'error': 'timeout'}
        except Exception as e:
            print(f"⚠️ OWLv2 detection error: {e}")
            return {'success': False, 'error': str(e)}
    
    def connect_db(self):
        """Get database connection from pool"""
        return self.db_pool.getconn()
    
    def return_db_connection(self, conn):
        """Return connection to pool"""
        self.db_pool.putconn(conn)
    
    @lru_cache(maxsize=100)
    def generate_query_embedding_cached(self, query: str) -> tuple:
        """Generate and cache embeddings for repeated queries"""
        embedding = self.generate_query_embedding(query)
        return tuple(embedding)  # Convert to tuple for caching
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for text query (observation description)"""
        if "bge" in self.embedding_model_name.lower():
            query = f"Represent this observation for searching relevant observations: {query}"
        
        embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        return embedding.tolist()
    
    def preprocess_image(self, image_path: str) -> Image.Image:
        """Resize image only if larger than 1024x1024 to speed up encoding and transmission"""
        img = Image.open(image_path).convert("RGB")
        width, height = img.size
        
        # Only resize if BOTH dimensions are larger than 1024px
        if width > 1024 and height > 1024:
            if width > height:
                new_width = self.max_image_size
                new_height = int(height * (self.max_image_size / width))
            else:
                new_height = self.max_image_size
                new_width = int(width * (self.max_image_size / height))
            
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            print(f"  ℹ️ Resized image from {width}x{height} to {new_width}x{new_height} for faster processing")
        else:
            print(f"  ℹ️ Image size {width}x{height} is within limits, no resize needed")
        
        return img
    
    def image_to_base64(self, image: Image.Image, quality: int = 85) -> str:
        """Convert PIL Image to optimized base64 string"""
        buffer = io.BytesIO()
        # Save as JPEG with quality setting for smaller size
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def get_image_url(self, image_path: str) -> str:
        """Convert image to optimized data URL for VLLM"""
        # Preprocess image (resize if needed)
        image = self.preprocess_image(image_path)
        base64_image = self.image_to_base64(image)
        return f"data:image/jpeg;base64,{base64_image}"
    
    def describe_image(self, image_path: str) -> Dict[str, Any]:
        """
        STAGE 1: Use VLLM to describe what's in the image
        This description will be used for semantic search against observations
        """
        print(f"\n🔍 Stage 1: Describing image with VLLM...")
        
        image_url = self.get_image_url(image_path)
        
        # Improved prompt with hierarchical structure to guide the model
        system_prompt = """You are an image analysis assistant. Your task is to describe what you see in the image in clear, objective detail.

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
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    },
                    {
                        "type": "text",
                        "text": user_prompt
                    }
                ]
            }
        ]
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": 250,  # Reduced from 350 for faster generation
            "temperature": 0.3,
            "top_p": 0.9,
            "stream": False
        }
        
        try:
            start_time = time.time()
            
            response = requests.post(
                self.chat_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=300
            )
            
            end_time = time.time()
            response.raise_for_status()
            result = response.json()
            
            description = result['choices'][0]['message']['content']
            
            print(f"✓ Image description generated in {end_time - start_time:.2f}s")
            print(f"\n{'='*80}")
            print("VLLM DESCRIPTION:")
            print(f"{'='*80}")
            print(description)
            print(f"{'='*80}\n")
            
            # Cache the image URL for Stage 3
            self._cached_image_url = image_url
            
            return {
                'description': description,
                'generation_time': end_time - start_time,
                'success': True
            }
            
        except Exception as e:
            print(f"❌ Error describing image: {e}")
            return {
                'description': None,
                'error': str(e),
                'success': False
            }
    
    def format_recommendations_to_paragraphs(self, structured_text: str) -> str:
        """
        Convert structured recommendation (with sections like SELECTED RIB SECTION, WHAT I SEE, etc.)
        into formal paragraph format for better readability.
        """
        lines = structured_text.split('\n')
        paragraphs = []
        current_section = None
        current_content = []
        
        for line in lines:
            # Preserve the SELECTED RIB SECTION line at top
            if line.startswith('SELECTED RIB SECTION:'):
                if current_content and current_section:
                    section_text = ' '.join([l.strip() for l in current_content if l.strip()])
                    if section_text:
                        paragraphs.append(f"**{current_section}**\n\n{section_text}\n")
                    current_content = []
                paragraphs.append(f"## {line}\n")
                current_section = None
            # Check for section headers (lines starting with "- " or section names in uppercase)
            elif line.startswith('- ') and ':' in line:
                if current_content and current_section:
                    section_text = ' '.join([l.strip() for l in current_content if l.strip()])
                    if section_text:
                        paragraphs.append(f"**{current_section}**\n\n{section_text}\n")
                current_section = line[2:].rstrip(':').strip()
                current_content = []
            elif line.strip() and current_section:
                current_content.append(line.strip())
        
        # Save last section
        if current_content and current_section:
            section_text = ' '.join([l.strip() for l in current_content if l.strip()])
            if section_text:
                paragraphs.append(f"**{current_section}**\n\n{section_text}\n")
        
        # Join paragraphs with proper spacing
        formatted = '\n'.join(paragraphs)
        return formatted if formatted.strip() else structured_text
    
    def search_observations(
        self, 
        query_text: str, 
        top_k: int = 5, 
        similarity_threshold: float = 0.4,
        category_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        STAGE 2: Search for similar observations using the query text
        Returns matching observations with their recommendations and regulations
        """
        print(f"\n🔎 Stage 2: Searching for matching observations...")
        
        # Try to use cached embedding first
        try:
            query_embedding = list(self.generate_query_embedding_cached(query_text))
        except:
            query_embedding = self.generate_query_embedding(query_text)
        
        conn = self.connect_db()
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
            self.return_db_connection(conn)
    
    def format_context_for_recommendations(self, chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved observations, recommendations, and regulations as context"""
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
    
    def generate_recommendations_text_only(
        self,
        user_query: str,
        context_chunks: List[Dict[str, Any]],
        max_tokens: int = 1024,
        temperature: float = 0.2
    ) -> Dict[str, Any]:
        """
        STAGE 3 (Text Mode): Generate recommendations using text query with retrieved context
        """
        print(f"\n💡 Stage 3: Generating recommendations with context...")
        
        # Format context
        context = self.format_context_for_recommendations(context_chunks)
        
        # System prompt for recommendation generation
        system_prompt = """You are an expert industrial safety consultant providing Risk Improvement Benchmark (RIB) recommendations.

Your role is to:
1. Understand the safety/structural issue described by the user
2. Provide specific, actionable recommendations from the matched RIB sections
3. Cite applicable regulations and guidelines
4. Be precise and reference specific section numbers

Structure your response as:
- ISSUE IDENTIFIED: Brief summary of the described issue
- MATCHED RIB SECTION: Which section(s) apply and why
- SPECIFIC RECOMMENDATIONS: Actionable steps from the RIB documentation
- APPLICABLE REGULATIONS: Relevant standards and guidelines
- PRIORITY ACTIONS: Most critical steps to take immediately"""

        # User prompt with description and context
        user_prompt = f"""Based on the situation described and the retrieved RIB documentation below, provide specific RIB recommendations and regulations.

USER DESCRIPTION:
{user_query}

{context}

QUESTION: Provide me with the specific RIB recommendations and regulations that apply to this situation.

Instructions:
1. Analyze how the user's description matches the RIB observations above
2. Provide detailed recommendations from the matched RIB sections
3. Cite specific regulations and guidelines mentioned
4. Be actionable and specific - reference exact requirements (e.g., testing frequencies, standards)
5. If multiple sections apply, explain which is most relevant and why"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "stream": False
        }
        
        try:
            start_time = time.time()
            
            response = requests.post(
                self.chat_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=600
            )
            
            end_time = time.time()
            total_time = end_time - start_time
            
            response.raise_for_status()
            result = response.json()
            
            generated_text = result['choices'][0]['message']['content']
            usage = result.get('usage', {})
            completion_tokens = usage.get('completion_tokens', 0)
            tokens_per_second = completion_tokens / total_time if total_time > 0 else 0
            
            print(f"✓ Recommendations generated in {total_time:.2f}s")
            print(f"  📝 Tokens: {completion_tokens}")
            print(f"  ⚡ Speed: {tokens_per_second:.2f} tokens/sec")
            
            return {
                'text': generated_text,
                'usage': usage,
                'generation_time': total_time,
                'tokens_per_second': tokens_per_second,
                'success': True
            }
            
        except Exception as e:
            print(f"❌ Error generating recommendations: {e}")
            return {
                'text': f"Error generating recommendations: {str(e)}",
                'error': str(e),
                'success': False
            }
    
    def generate_recommendations_with_image(
        self,
        image_description: str,
        context_chunks: List[Dict[str, Any]],
        image_path: str,
        owlv2_detections: Optional[Dict[str, Any]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.2
    ) -> Dict[str, Any]:
        """
        STAGE 3 (Image Mode): Generate recommendations using VLLM with retrieved context, OWLv2 detections, and image
        """
        print(f"\n💡 Stage 3: Generating recommendations with context, OWLv2 detections, and image...")
        
        # Format context
        context = self.format_context_for_recommendations(context_chunks)
        
        # Build OWLv2 detection context if available
        owlv2_context = ""
        if owlv2_detections and owlv2_detections.get('success'):
            data = owlv2_detections.get('data', {})
            top5 = data.get('top5_rib_subsections', [])
            broad_categories = data.get('broad_categories', [])
            primary_subsection = data.get('primary_rib_subsection', '')
            
            if top5:
                owlv2_context = "\n\n=== OBJECT DETECTION RESULTS (OWLv2) ===\n"
                
                if broad_categories:
                    owlv2_context += f"Detected Categories: {', '.join(broad_categories)}\n"
                
                if primary_subsection:
                    owlv2_context += f"Primary RIB Subsection: {primary_subsection}\n"
                
                owlv2_context += "\nTop 5 RIB Subsections Detected by Object Analysis:\n"
                for sub in top5:
                    owlv2_context += f"  #{sub['rank']}: {sub['section']} (score: {sub['score']:.3f}, detections: {sub['detection_count']})\n"
                
                owlv2_context += "========================================\n"
        
        # System prompt for recommendation generation with OWLv2 integration
        system_prompt = """You are an expert industrial safety consultant providing Risk Improvement Benchmark (RIB) recommendations.

You have access to THREE sources of information:
1. OBJECT DETECTION RESULTS (OWLv2): An AI vision system has analyzed the image and detected specific objects and suggested RIB subsections
2. IMAGE DESCRIPTION: A textual description of what's in the image
3. SEMANTIC SEARCH RESULTS: Text-based matching against the RIB observation database

CRITICAL INSTRUCTIONS:
1. First, review the OBJECT DETECTION RESULTS to see what objects were visually detected
2. Compare the top 5 detected RIB subsections with the semantic search results
3. Look at the image and verify which detection is most accurate
4. Determine which RIB section BEST matches what is actually visible in the image
5. If NONE of the top 5 detected subsections fit the image, describe what you see and explain why

Your response must START with a single line in this exact format:
SELECTED RIB SECTION: <section number and title>

Then follow this structure:
- WHAT I SEE: Describe what's visible in the image
- OBJECT DETECTION ANALYSIS: Evaluate which of the top 5 detected subsections best fits
- WHY SELECTED: 2-4 bullet points explaining why this section is the best match
- SPECIFIC RECOMMENDATIONS: Actionable steps from the RIB documentation
- APPLICABLE REGULATIONS: Relevant standards and guidelines
- PRIORITY ACTIONS: Most critical steps to take immediately"""

        # User prompt with OWLv2 detections, image description, and context
        user_prompt = f"""Based on the image, object detection results, and retrieved RIB documentation below, provide specific RIB recommendations.
{owlv2_context}

IMAGE DESCRIPTION (from initial analysis):
{image_description}

{context}

INSTRUCTIONS:
1. START your response with the single line:
    SELECTED RIB SECTION: <section number and title>
2. After that line, follow the required structure exactly.
3. First, look at the OBJECT DETECTION RESULTS to see what objects were detected in the image
4. Review the top 5 RIB subsections suggested by object detection
5. Compare with the semantic search results and image description
6. Determine which RIB section BEST matches what you can see in the image
7. If the object detection suggests a different section than semantic search, explain which one fits better and why
8. If NONE of the top 5 detected subsections fit the image well, describe the image and use the semantic search result instead
9. Provide recommendations from ONLY the section that best matches the image

Provide specific, actionable recommendations based on the best matching section."""

        # Prepare messages with image (reuse cached encoding from Stage 1)
        image_url = self._cached_image_url if self._cached_image_url else self.get_image_url(image_path)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    },
                    {
                        "type": "text",
                        "text": user_prompt
                    }
                ]
            }
        ]
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "stream": False
        }
        
        try:
            start_time = time.time()
            
            response = requests.post(
                self.chat_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=600
            )
            
            end_time = time.time()
            total_time = end_time - start_time
            
            response.raise_for_status()
            result = response.json()
            
            generated_text = result['choices'][0]['message']['content']
            usage = result.get('usage', {})
            completion_tokens = usage.get('completion_tokens', 0)
            tokens_per_second = completion_tokens / total_time if total_time > 0 else 0
            
            print(f"✓ Recommendations generated in {total_time:.2f}s")
            print(f"  📝 Tokens: {completion_tokens}")
            print(f"  ⚡ Speed: {tokens_per_second:.2f} tokens/sec")
            
            return {
                'text': generated_text,
                'usage': usage,
                'generation_time': total_time,
                'tokens_per_second': tokens_per_second,
                'success': True
            }
            
        except Exception as e:
            print(f"❌ Error generating recommendations: {e}")
            return {
                'text': f"Error generating recommendations: {str(e)}",
                'error': str(e),
                'success': False
            }
    
    def query_with_text(
        self,
        query_text: str,
        top_k: int = 3,
        category_filter: Optional[str] = None,
        similarity_threshold: float = 0.4,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        show_sources: bool = True
    ) -> Dict[str, Any]:
        """
        2-stage RAG pipeline for text input:
        1. Search for matching observations using the text query
        2. VLLM generates recommendations using retrieved context
        """
        
        print(f"\n{'='*80}")
        print(f"TEXT-BASED RAG PIPELINE - TSD RIB")
        print(f"{'='*80}")
        print(f"📝 Query: {query_text[:100]}{'...' if len(query_text) > 100 else ''}")
        
        # STAGE 1: Search for matching observations
        matching_chunks = self.search_observations(
            query_text=query_text,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            category_filter=category_filter
        )
        
        if not matching_chunks:
            print(f"\n⚠️  No matching RIB observations found.")
            print(f"   Try lowering similarity_threshold (current: {similarity_threshold})")
            return {
                'query': query_text,
                'answer': "No matching RIB documentation found for the described issue.",
                'num_sources': 0,
                'success': False
            }
        
        # STAGE 2: Generate recommendations
        recommendation_result = self.generate_recommendations_text_only(
            user_query=query_text,
            context_chunks=matching_chunks,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        if not recommendation_result['success']:
            return {
                'query': query_text,
                'error': 'Failed to generate recommendations',
                'stage': 2,
                'success': False
            }
        
        # Compile final result
        # Format recommendation text into formal paragraphs
        formatted_answer = self.format_recommendations_to_paragraphs(recommendation_result['text'])
        
        result = {
            'query': query_text,
            'answer': formatted_answer,
            'num_sources': len(matching_chunks),
            'total_time': recommendation_result.get('generation_time', 0),
            'tokens_per_second': recommendation_result.get('tokens_per_second', 0),
            'success': True
        }
        
        if show_sources:
            result['sources'] = [
                {
                    'section': chunk['section_number'],
                    'title': chunk['title'],
                    'category': chunk['category'],
                    'similarity': round(chunk['similarity'], 3)
                }
                for chunk in matching_chunks
            ]
        
        return result
    
    def query_with_image(
        self,
        image_path: str,
        top_k: int = 3,
        category_filter: Optional[str] = None,
        similarity_threshold: float = 0.4,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        show_sources: bool = True,
        use_object_detection: bool = True
    ) -> Dict[str, Any]:
        """
        Complete 4-stage RAG pipeline for image input with OWLv2 integration:
        1. OWLv2 detects objects and suggests top 5 RIB subsections
        2. VLLM describes the image
        3. Search for matching observations using the description (enhanced with detected objects)
        4. VLLM generates recommendations using OWLv2 detections, description, and retrieved context
        """
        
        if not Path(image_path).exists():
            return {
                'error': f"Image not found: {image_path}",
                'success': False
            }
        
        print(f"\n{'='*80}")
        print(f"MULTIMODAL RAG PIPELINE WITH OWLv2 - TSD RIB")
        print(f"{'='*80}")
        print(f"📸 Image: {image_path}")
        
        # STAGE 0: Object detection with OWLv2
        owlv2_detections = None
        if use_object_detection:
            owlv2_detections = self.detect_objects_in_image(image_path)
        
        # STAGE 1: Describe image
        description_result = self.describe_image(image_path)
        
        if not description_result['success']:
            return {
                'error': 'Failed to describe image',
                'stage': 1,
                'success': False
            }
        
        image_description = description_result['description']
        
        # STAGE 2: Search for matching observations (enhanced with OWLv2 detections)
        enhanced_query = image_description
        if owlv2_detections and owlv2_detections.get('success'):
            data = owlv2_detections.get('data', {})
            top5_subsections = data.get('top5_rib_subsections', [])
            primary_subsection = data.get('primary_rib_subsection', '')
            
            # Enhance query with primary detected subsection
            if primary_subsection:
                enhanced_query = f"{image_description} {primary_subsection}"
                print(f"\n  📝 Enhanced search query with primary subsection: {primary_subsection}")
        
        matching_chunks = self.search_observations(
            query_text=enhanced_query,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            category_filter=category_filter
        )
        
        if not matching_chunks:
            print(f"\n⚠️  No matching RIB observations found.")
            print(f"   Try lowering similarity_threshold (current: {similarity_threshold})")
            return {
                'image_description': image_description,
                'answer': "No matching RIB documentation found for the observed issue.",
                'num_sources': 0,
                'owlv2_detection': owlv2_detections.get('data') if owlv2_detections else None,
                'success': False
            }
        
        # STAGE 3: Generate recommendations with OWLv2 detections
        recommendation_result = self.generate_recommendations_with_image(
            image_description=image_description,
            context_chunks=matching_chunks,
            image_path=image_path,
            owlv2_detections=owlv2_detections,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        if not recommendation_result['success']:
            return {
                'image_description': image_description,
                'error': 'Failed to generate recommendations',
                'stage': 3,
                'success': False
            }
        
        # Compile final result
        top_match = matching_chunks[0] if matching_chunks else None
        
        # Format recommendation text into formal paragraphs
        formatted_answer = self.format_recommendations_to_paragraphs(recommendation_result['text'])
        
        result = {
            'image_description': image_description,
            'answer': formatted_answer,
            'num_sources': len(matching_chunks),
            'total_time': (
                description_result.get('generation_time', 0) +
                recommendation_result.get('generation_time', 0)
            ),
            'tokens_per_second': recommendation_result.get('tokens_per_second', 0),
            'success': True,
            'semantic_top_match': {
                'section': top_match['section_number'],
                'title': top_match['title'],
                'category': top_match['category'],
                'similarity': round(top_match['similarity'], 3)
            } if top_match else None
        }
        
        if show_sources:
            result['sources'] = [
                {
                    'section': chunk['section_number'],
                    'title': chunk['title'],
                    'category': chunk['category'],
                    'similarity': round(chunk['similarity'], 3)
                }
                for chunk in matching_chunks
            ]
        
        # Include OWLv2 detection results in response
        if owlv2_detections and owlv2_detections.get('success'):
            data = owlv2_detections.get('data', {})
            result['owlv2_detection'] = {
                'top5_rib_subsections': data.get('top5_rib_subsections', []),
                'primary_rib_subsection': data.get('primary_rib_subsection'),
                'broad_categories': data.get('broad_categories', []),
                'tier1_detections': data.get('tier1_detections', 0)
            }
        
        return result
    
    def interactive_query(
        self,
        user_input: str,
        top_k: int = 3,
        category_filter: Optional[str] = None,
        similarity_threshold: float = 0.4,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        show_sources: bool = True,
        use_object_detection: bool = True
    ) -> Dict[str, Any]:
        """
        Universal query interface that handles both text and image inputs
        Use /image <path> to query with an image
        Otherwise, treats input as text query
        
        Args:
            use_object_detection: If True, uses OWLv2 for object detection before VLM processing
        """
        
        # Check if input starts with /image command
        if user_input.strip().startswith('/image'):
            # Extract image path
            parts = user_input.strip().split(maxsplit=1)
            if len(parts) < 2:
                return {
                    'error': 'Please provide image path after /image command',
                    'usage': '/image <path/to/image.jpg>',
                    'success': False
                }
            
            image_path = parts[1].strip()
            
            # Use image query pipeline with optional object detection
            return self.query_with_image(
                image_path=image_path,
                top_k=top_k,
                category_filter=category_filter,
                similarity_threshold=similarity_threshold,
                max_tokens=max_tokens,
                temperature=temperature,
                show_sources=show_sources,
                use_object_detection=use_object_detection
            )
        else:
            # Use text query pipeline
            return self.query_with_text(
                query_text=user_input,
                top_k=top_k,
                category_filter=category_filter,
                similarity_threshold=similarity_threshold,
                max_tokens=max_tokens,
                temperature=temperature,
                show_sources=show_sources
            )


def main():
    """Interactive demo with both text and image support"""
    
    # Database configuration
    db_config = {
        'host': 'localhost',
        'database': 'tsdsrd',
        'user': 'amir',
        'password': 'amir123',
        'port': 5432
    }
    
    print("="*80)
    print("TSD RIB MULTIMODAL RAG SYSTEM")
    print("Supports both text queries and image analysis")
    print("With OWLv2 Object Detection Integration")
    print("="*80)
    
    try:
        rag = MultimodalRAGSystemTSDRIB(
            db_config=db_config,
            vllm_url="http://localhost:8000",
            embedding_model_name="BAAI/bge-large-en-v1.5",
            owlv2_url="http://localhost:8010"
        )
    except Exception as e:
        print(f"\n❌ Failed to initialize: {e}")
        return
    
    print("\n" + "="*80)
    print("SYSTEM READY")
    print("="*80)
    print("\nUsage:")
    print("  • Text query: Just type your description")
    print("  • Image query: /image <path/to/image.jpg>")
    print("  • Type 'quit' or 'exit' to stop")
    print("="*80)
    
    # Interactive loop
    while True:
        print("\n" + "-"*80)
        user_input = input("\n🔍 Enter your query (or 'quit' to exit): ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not user_input:
            continue
        
        # Process query
        result = rag.interactive_query(
            user_input=user_input,
            top_k=3,
            similarity_threshold=0.4,
            temperature=0.2,
            show_sources=True
        )
        
        # Display results
        if result.get('success'):
            print(f"\n{'='*80}")
            
            # Show description if it's an image query
            if 'image_description' in result:
                print("STAGE 1: IMAGE ANALYSIS")
                print(f"{'='*80}")
                print(result['image_description'])
                print(f"\n{'='*80}")
                print("FINAL RECOMMENDATIONS")
            else:
                print("RIB RECOMMENDATIONS")
            
            print(f"{'='*80}")
            print(result['answer'])
            
            print(f"\n{'='*80}")
            print("PERFORMANCE & SOURCES")
            print(f"{'='*80}")
            print(f"⏱️  Total time: {result.get('total_time', 0):.2f}s")
            print(f"⚡ Tokens/sec: {result.get('tokens_per_second', 0):.2f}")
            print(f"📚 Sources used: {result.get('num_sources', 0)}")
            
            if result.get('sources'):
                print(f"\n🔖 Matched RIB sections:")
                for i, source in enumerate(result['sources'], 1):
                    print(f"   {i}. Section {source['section']}: {source['title']}")
                    print(f"      Category: {source['category']} | Similarity: {source['similarity']}")
            
            # Display OWLv2 detection results
            if result.get('owlv2_detection'):
                owlv2 = result['owlv2_detection']
                print(f"\n{'='*80}")
                print("🔍 OWLv2 OBJECT DETECTION RESULTS")
                print(f"{'='*80}")
                
                # Display tier1 detection count
                tier1_count = owlv2.get('tier1_detections', 0)
                if tier1_count:
                    print(f"\n📦 Tier 1 Detections: {tier1_count}")
                
                # Display broad categories
                categories = owlv2.get('broad_categories', [])
                if categories:
                    print(f"📊 Broad Categories: {', '.join(categories)}")
                
                # Display primary subsection
                primary = owlv2.get('primary_rib_subsection')
                if primary:
                    print(f"\n🎯 Primary RIB Subsection: {primary}")
                
                # Display top 5 RIB subsections
                top5_rib = owlv2.get('top5_rib_subsections', [])
                if top5_rib:
                    print(f"\n🏆 Top 5 RIB Subsections from Object Detection:")
                    for sub in top5_rib:
                        print(f"   #{sub['rank']}: {sub['section']}")
                        print(f"         Score: {sub['score']:.3f} | Detections: {sub['detection_count']}")
        else:
            print(f"\n❌ Error: {result.get('error', 'Unknown error')}")
            if 'usage' in result:
                print(f"💡 Usage: {result['usage']}")


def demo_examples():
    """Run example queries to demonstrate both modes"""
    
    db_config = {
        'host': 'localhost',
        'database': 'tsdsrd',
        'user': 'amir',
        'password': 'amir123',
        'port': 5432
    }
    
    print("="*80)
    print("TSD RIB MULTIMODAL RAG SYSTEM - DEMO")
    print("="*80)
    
    try:
        rag = MultimodalRAGSystemTSDRIB(
            db_config=db_config,
            vllm_url="http://localhost:8000",
            embedding_model_name="BAAI/bge-large-en-v1.5",
            owlv2_url="http://localhost:8010"
        )
    except Exception as e:
        print(f"\n❌ Failed to initialize: {e}")
        return
    
    print("\n" + "="*80)
    print("EXAMPLE 1: TEXT QUERY")
    print("="*80)

    text_query = """The site is situated in an area that frequently experiences intense atmospheric disturbances, 
    creating a serious risk to both personnel safety and operational stability. These sudden high-energy events increase the 
    chances of damage to sensitive equipment, unexpected power interruptions, and even fire outbreaks. Recent incident reports 
    linked to these conditions show that the current protective measures are no longer adequate, highlighting the urgent need 
    to strengthen and modernize the facility's protection systems.
    """
    result1 = rag.query_with_text(
        query_text=text_query,
        top_k=3,
        similarity_threshold=0.4
    )
    
    if result1.get('success'):
        print(f"\n📝 Query: {text_query}")
        print(f"\n{'='*80}")
        print("RECOMMENDATIONS:")
        print(f"{'='*80}")
        print(result1['answer'])
    
    print("\n\n" + "="*80)
    print("EXAMPLE 2: IMAGE QUERY")
    print("="*80)
    
    image_path = "/home/amir/Desktop/MRE TSD/RIB_images/page34_img1.png"
    
    result2 = rag.query_with_image(
        image_path=image_path,
        top_k=3,
        similarity_threshold=0.4
    )
    
    if result2.get('success'):
        print(f"\n📸 Image: {image_path}")
        
        # Display OWLv2 detection results
        if result2.get('owlv2_detection'):
            owlv2 = result2['owlv2_detection']
            print(f"\n{'='*80}")
            print("🔍 OWLv2 OBJECT DETECTION RESULTS")
            print(f"{'='*80}")
            
            # Display tier1 detection count
            tier1_count = owlv2.get('tier1_detections', 0)
            if tier1_count:
                print(f"\n📦 Tier 1 Detections: {tier1_count}")
            
            # Display broad categories
            categories = owlv2.get('broad_categories', [])
            if categories:
                print(f"📊 Broad Categories: {', '.join(categories)}")
            
            # Display primary subsection
            primary = owlv2.get('primary_rib_subsection')
            if primary:
                print(f"\n🎯 Primary RIB Subsection: {primary}")
            
            # Display top 5 RIB subsections
            top5_rib = owlv2.get('top5_rib_subsections', [])
            if top5_rib:
                print(f"\n🏆 Top 5 RIB Subsections from Object Detection:")
                for sub in top5_rib:
                    print(f"   #{sub['rank']}: {sub['section']}")
                    print(f"         Score: {sub['score']:.3f} | Detections: {sub['detection_count']}")
        
        print(f"\n{'='*80}")
        print("IMAGE ANALYSIS:")
        print(f"{'='*80}")
        print(result2['image_description'])
        print(f"\n{'='*80}")
        print("RECOMMENDATIONS:")
        print(f"{'='*80}")
        print(result2['answer'])


if __name__ == "__main__":
    # Choose mode:
    # main()           # Interactive mode
    demo_examples()    # Run examples