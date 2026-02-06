import psycopg2
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import torch
import requests
import json
import base64
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image
import io

class MultimodalRAGSystemTSDRIB:
    def __init__(
        self, 
        db_config: Dict[str, str], 
        vllm_url: str = "http://localhost:8000",
        embedding_model_name: str = "BAAI/bge-large-en-v1.5"
    ):
        """Initialize Multimodal RAG system with VLLM server for TSD RIB database"""
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
    
    def detect_red_circle(self, image_path: str) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect red circle in the image and return bounding box of the circled area
        
        Returns:
            (x, y, width, height) of the red circle area, or None if no circle detected
        """
        print(f"\n🔍 Checking for red circle annotation in image...")
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠️  Could not read image: {image_path}")
            return None
        
        # Convert to HSV for better red detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define range for red color (red wraps around in HSV, so we need two ranges)
        # Lower red range (0-10 in hue)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        
        # Upper red range (170-180 in hue)
        lower_red2 = np.array([170, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        # Create masks for both red ranges
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        
        # Combine masks
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print(f"✓ No red circle detected - will use full image")
            return None
        
        # Find the largest circular contour
        best_circle = None
        best_circularity = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter out very small areas (noise)
            if area < 1000:  # Minimum area threshold
                continue
            
            # Calculate circularity: 4π × area / perimeter²
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # A perfect circle has circularity of 1.0
            # Accept shapes with circularity > 0.6 (allows for imperfect circles)
            if circularity > 0.6 and circularity > best_circularity:
                best_circularity = circularity
                best_circle = contour
        
        if best_circle is not None:
            # Get bounding box of the circle
            x, y, w, h = cv2.boundingRect(best_circle)
            
            # Add some padding (10% on each side)
            padding = int(min(w, h) * 0.1)
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)
            
            print(f"✓ Red circle detected! Circularity: {best_circularity:.2f}")
            print(f"  Region of Interest: x={x}, y={y}, width={w}, height={h}")
            print(f"  📌 VLM will focus ONLY on the circled area")
            
            return (x, y, w, h)
        
        print(f"✓ No clear red circle detected - will use full image")
        return None
    
    def extract_roi_from_image(self, image_path: str, bbox: Tuple[int, int, int, int]) -> str:
        """
        Extract the Region of Interest (ROI) from image based on bounding box
        
        Args:
            image_path: Path to original image
            bbox: (x, y, width, height) of the ROI
            
        Returns:
            Path to the extracted ROI image
        """
        x, y, w, h = bbox
        
        # Read image
        image = cv2.imread(image_path)
        
        # Extract ROI
        roi = image[y:y+h, x:x+w]
        
        # Save ROI to temporary file
        roi_path = image_path.replace('.png', '_roi.png').replace('.jpg', '_roi.jpg')
        cv2.imwrite(roi_path, roi)
        
        print(f"✓ Extracted ROI saved to: {roi_path}")
        
        return roi_path
    
    def preprocess_image_for_vllm(self, image_path: str) -> Tuple[str, bool, Optional[Tuple[int, int, int, int]]]:
        """
        Preprocess image: detect red circle and extract ROI if present
        
        Returns:
            (processed_image_path, has_red_circle, bbox)
        """
        # Check for red circle
        bbox = self.detect_red_circle(image_path)
        
        if bbox is not None:
            # Extract ROI
            roi_path = self.extract_roi_from_image(image_path, bbox)
            return roi_path, True, bbox
        else:
            # Use original image
            return image_path, False, None
    
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
        """Generate embedding for text query (observation description)"""
        if "bge" in self.embedding_model_name.lower():
            query = f"Represent this observation for searching relevant observations: {query}"
        
        embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        return embedding.tolist()
    
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
    
    def describe_image(self, image_path: str, focus_instruction: str = "") -> Dict[str, Any]:
        """
        STAGE 1: Use VLLM to describe what's in the image
        This description will be used for semantic search against observations
        
        Args:
            image_path: Path to the image (could be original or ROI)
            focus_instruction: Additional instruction if ROI was extracted
        """
        print(f"\n🔍 Stage 1: Describing image with VLLM...")
        
        image_url = self.get_image_url(image_path)
        
        # Build system prompt with optional focus instruction
        focus_context = ""
        if focus_instruction:
            focus_context = f"\n\nIMPORTANT: {focus_instruction}"
        
        # Improved prompt with hierarchical structure to guide the model
        system_prompt = f"""You are an image analysis assistant. Your task is to describe what you see in the image in clear, objective detail.{focus_context}

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
            "max_tokens": 512,
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
                timeout=120
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
            conn.close()
    
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
        max_tokens: int = 2048,
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
        max_tokens: int = 2048,
        temperature: float = 0.2,
        has_red_circle: bool = False
    ) -> Dict[str, Any]:
        """
        STAGE 3 (Image Mode): Generate recommendations using VLLM with retrieved context and image
        """
        print(f"\n💡 Stage 3: Generating recommendations with context and image...")
        
        # Format context
        context = self.format_context_for_recommendations(context_chunks)
        
        # Add context about red circle if present
        roi_instruction = ""
        if has_red_circle:
            roi_instruction = "\n\nNOTE: This image shows the specific area of concern that was marked with a red circle in the original image. Focus your analysis on this extracted region."
        
        # System prompt for recommendation generation
        system_prompt = f"""You are an expert industrial safety consultant providing Risk Improvement Benchmark (RIB) recommendations.{roi_instruction}

Your role is to:
1. Confirm what you see in the image matches the RIB observations
2. Provide specific, actionable recommendations from the matched RIB sections
3. Cite applicable regulations and guidelines
4. Be precise and reference specific section numbers

Structure your response as:
- WHAT I SEE: Look at the image and detect the objects in the image correctly. For example, 
1. If there is a flagpole in the image, are you sure its a flagpole and not a lightning arrester?
2. If there is a electrical board in the image, are you sure its an electrical board and not a lightning event counter?
- MATCHED RIB SECTION: Which section(s) apply and why
- SPECIFIC RECOMMENDATIONS: Actionable steps from the RIB documentation
- APPLICABLE REGULATIONS: Relevant standards and guidelines
- PRIORITY ACTIONS: Most critical steps to take immediately"""

        # User prompt with image, description, and context
        user_prompt = f"""Based on the image and the retrieved RIB documentation below, provide specific RIB recommendations and regulations.

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

        # Prepare messages with image
        image_url = self.get_image_url(image_path)
        
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
        result = {
            'query': query_text,
            'answer': recommendation_result['text'],
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
        show_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Complete 3-stage RAG pipeline for image input with red circle detection:
        0. Preprocess: Detect red circle and extract ROI if present
        1. VLLM describes the image (original or ROI)
        2. Search for matching observations using the description
        3. VLLM generates recommendations using retrieved context
        """
        
        if not Path(image_path).exists():
            return {
                'error': f"Image not found: {image_path}",
                'success': False
            }
        
        print(f"\n{'='*80}")
        print(f"MULTIMODAL RAG PIPELINE - TSD RIB")
        print(f"{'='*80}")
        print(f"📸 Image: {image_path}")
        
        # STAGE 0: Preprocess image - detect red circle and extract ROI
        processed_image_path, has_red_circle, bbox = self.preprocess_image_for_vllm(image_path)
        
        focus_instruction = ""
        if has_red_circle:
            focus_instruction = "This image shows ONLY the region that was marked with a red circle. Focus your entire description on what is visible in THIS specific area."
        
        # STAGE 1: Describe image (original or ROI)
        description_result = self.describe_image(processed_image_path, focus_instruction)
        
        if not description_result['success']:
            return {
                'error': 'Failed to describe image',
                'stage': 1,
                'success': False
            }
        
        image_description = description_result['description']
        
        # STAGE 2: Search for matching observations
        matching_chunks = self.search_observations(
            query_text=image_description,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            category_filter=category_filter
        )
        
        if not matching_chunks:
            print(f"\n⚠️  No matching RIB observations found.")
            print(f"   Try lowering similarity_threshold (current: {similarity_threshold})")
            return {
                'image_description': image_description,
                'has_red_circle': has_red_circle,
                'answer': "No matching RIB documentation found for the observed issue.",
                'num_sources': 0,
                'success': False
            }
        
        # STAGE 3: Generate recommendations
        recommendation_result = self.generate_recommendations_with_image(
            image_description=image_description,
            context_chunks=matching_chunks,
            image_path=processed_image_path,  # Use processed image (ROI or original)
            max_tokens=max_tokens,
            temperature=temperature,
            has_red_circle=has_red_circle
        )
        
        if not recommendation_result['success']:
            return {
                'image_description': image_description,
                'has_red_circle': has_red_circle,
                'error': 'Failed to generate recommendations',
                'stage': 3,
                'success': False
            }
        
        # Compile final result
        result = {
            'image_description': image_description,
            'has_red_circle': has_red_circle,
            'roi_bbox': bbox if has_red_circle else None,
            'answer': recommendation_result['text'],
            'num_sources': len(matching_chunks),
            'total_time': (
                description_result.get('generation_time', 0) +
                recommendation_result.get('generation_time', 0)
            ),
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
    
    def interactive_query(
        self,
        user_input: str,
        top_k: int = 3,
        category_filter: Optional[str] = None,
        similarity_threshold: float = 0.4,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        show_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Universal query interface that handles both text and image inputs
        Use /image <path> to query with an image
        Otherwise, treats input as text query
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
            
            # Use image query pipeline
            return self.query_with_image(
                image_path=image_path,
                top_k=top_k,
                category_filter=category_filter,
                similarity_threshold=similarity_threshold,
                max_tokens=max_tokens,
                temperature=temperature,
                show_sources=show_sources
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
    print("Now with RED CIRCLE detection!")
    print("="*80)
    
    try:
        rag = MultimodalRAGSystemTSDRIB(
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
    print("\nUsage:")
    print("  • Text query: Just type your description")
    print("  • Image query: /image <path/to/image.jpg>")
    print("  • Red circle: If image has a red circle, only that area will be analyzed!")
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
                if result.get('has_red_circle'):
                    print("⭕ RED CIRCLE DETECTED - Analyzed circled area only")
                    print(f"   ROI: {result.get('roi_bbox')}")
                    print(f"{'='*80}")
                
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
            embedding_model_name="BAAI/bge-large-en-v1.5"
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
    print("EXAMPLE 2: IMAGE QUERY (with potential red circle)")
    print("="*80)
    
    image_path = "RIB_images/page36_img1.png"
    result2 = rag.query_with_image(
        image_path=image_path,
        top_k=3,
        similarity_threshold=0.4
    )
    
    if result2.get('success'):
        print(f"\n📸 Image: {image_path}")
        
        if result2.get('has_red_circle'):
            print(f"⭕ RED CIRCLE DETECTED!")
            print(f"   ROI: {result2.get('roi_bbox')}")
        
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