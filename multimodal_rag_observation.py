import psycopg2
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import torch
import requests
import json
import base64
from pathlib import Path
import time

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
    
    def describe_image(self, image_path: str) -> Dict[str, Any]:
        """
        STAGE 1: Use VLLM to describe what's in the image
        This description will be used for semantic search against observations
        """
        print(f"\n🔍 Stage 1: Describing image with VLLM...")
        
        image_url = self.get_image_url(image_path)
        
        # System prompt for image description
        system_prompt = """You are an expert structural engineer and industrial safety consultant specializing in building integrity, environmental hazards, and operational risk in industrial and commercial facilities.

Your task is to analyze images and identify the SPECIFIC UNDERLYING RISK SCENARIO based on visible physical evidence — not just describe surface damage.

You must match observations to ONE of the predefined risk situations below and explain the ROOT CAUSE using engineering reasoning. 

***THINK TWICE BEFORE YOU MATCH THE OBSERVATION TO A RISK SCENARIO. MAKE SURE THE VISIBLE EVIDENCE SUPPORTS YOUR DIAGNOSIS. DO NOT BE MISTAKEN TREES WITH LIGHTNING ARRESTERS***

===============================
RISK SCENARIOS YOU MUST IDENTIFY
===============================

PERILS CATEGORY:

1. SUBSIDENCE  
   Indicators:
   - Evidence of subsidence were observed at several areas such as around the M&E rooms \nof Carpark Building.  \n \nSubsidence, which refers to the gradual sinking or settling of the ground, poses significant \nrisks to the structural integrity and operational functionality of the building

2. PIPE BURSTING  
   Indicators:
   - Regular occurrences of pipe bursting in the facility present significant risks, including \npotential damage to infrastructure, equipment, and inventory. This issue can also disrupt \noperations, increase maintenance costs, and pose safety hazards such as flooding, \nelectrical hazards, and compromised structural integrity.

3. LANDSLIDE / SLOPE INSTABILITY  
   Indicators:
   - The risk of landslides is present due to the slightly elevated land with a hill located within \nthe campus. Although no past incidents of landslides have been recorded, the proximity of \nthe hill slope to buildings increases the risk of soil movement which could lead to a \nlandslide, potentially causing structural damage, injury, or operational disruption. \n \nPotential Risks Associated with the Hill Slope \n \n1. Landslide & Soil Erosion Risks \n- Unstable slopes due to natural soil erosion, which may gradually weaken the slope \nstability. \n- Heavy rainfall or poor drainage systems may trigger soil displacement. \n- Vegetation overgrowth can cause uneven soil distribution, increasing the chances of \nslope failure. \n \n2. Infrastructure & Safety Risks \n- The proximity of the buildings and walkway to the hill slope means a landslide could \ndamage structures. \n- Falling rocks or debris may cause injury to individuals walking along the covered \nwalkway.

4. DRAINAGE SYSTEM FAILURE  
   Indicators:
   - Cracked or broken drains
   - Blocked drainage channels
   - Vegetation growing inside drains
   - Standing water near drainage lines  
   Root Cause Focus: Inadequate stormwater flow, structural drain damage, blockage, poor maintenance.

5. FLOOD RISK / POOR FLOOD MITIGATION  
   Indicators:
   - Water marks on walls
   - Equipment placed at low levels
   - Absence of flood barriers
   - Signs of previous inundation  
   Root Cause Focus: Inadequate flood protection, poor water diversion, insufficient drainage capacity.

6. LIGHTNING STRIKE RISK  
   Indicators:
   - The area is highly exposed to lightning activity, which poses a significant risk to both \npersonnel safety and operational integrity. The frequent occurrence of lightning strikes \nincreases the likelihood of damage to critical equipment, power disruptions, and potential \nfire hazards. Furthermore, recent losses recorded due to lightning incidents highlight the \ninadequacy of current protection measures and the urgent need for improved lightning \nprotection systems. \n \n \nLightning arestor and lightning strike counter

7. FALLING TREE HAZARD  
   Indicators:
   - The presence of large, mature trees within a developed area poses significant risks that \nmust be addressed to ensure safety and prevent potential losses. Trees provide essential \nbenefits such as shade, aesthetic value, and environmental enhancement. However, their \nsize, age, and structural condition can also pose hazards if not properly managed. This \nmay pose to following risks: \n \n1) Dead Trees: \n• \nTrees that have died due to diseases, pest infestations (such as termites or \nbeetles), or natural decay are at a high risk of falling. Dead trees are structurally \nweak and can collapse suddenly, causing injury, property damage, or even \nfatalities. \n \n2) Broken or Hanging Branches: \n• \nDetached or partially broken branches, especially those hanging from significant \nheights, may fall unexpectedly. This can be hazardous to pedestrians, vehicles, \nand nearby structures. \n \n3) Material Fall from Trees: \n• \nObjects such as nests, fruits, or broken branches can fall without warning, posing a \nthreat to people and property below. \n \n4) Signs of Disease or Structural Weakness: \n• \nPresence of fungi, mushrooms, or perennial fruiting bodies on the trunk or \nbranches indicates internal decay and structural weakness. \n• \nDisease symptoms such as leaf discoloration, premature shedding, or cankers can \nalso signal compromised health and stability. \n \n5) Leaning Trees: \n• \nTrees with a tilt greater than 10 degrees from the vertical pose a higher potential \nfor failure, especially during heavy rains, strong winds, or seismic activity. \n \n6) Cracks in Trunks and Branches: \n• \nVisible cracks are clear signs of compromised structural integrity, increasing the \nlikelihood of collapse during storms or high wind events. \n \n7) Soil Heaving or Uplift: \n• \nDisturbances around tree roots, such as soil heaving or uplift, indicate potential \nroot failure. Such trees may be ready to fall at any moment, particularly under \nadverse weather conditions

8. WINDSTORM DAMAGE RISK  
   Indicators:
   - Loose or detached roof components
   - Ceiling panels blown off
   - Lightweight materials poorly secured  
   Root Cause Focus: Wind uplift forces, inadequate fastening, poor wind-resistant design.

9. IMPACT DAMAGE FROM VEHICLES / LIFTING ACTIVITIES  
   Indicators:
   - Dents or cracks on walls at vehicle height
   - Damaged columns near loading bays
   - Roof or canopy impact marks  
   Root Cause Focus: Vehicle collision due to poor clearance, lack of barriers, unsafe maneuvering.

10. GAS INSTALLATION IMPACT RISK  
   Indicators:
   - Gas meters or regulators near vehicle paths
   - Weak or plastic barriers protecting gas systems
   - No crash-rated protection around gas equipment  
   Root Cause Focus: High consequence impact risk leading to gas leaks, fire, or explosion.

===============================
OTHER RISK CATEGORIES
===============================

ELECTRICAL  
- Exposed wiring, open panels, overloaded sockets  
Root Cause Focus: Electrical safety management failure

HOUSEKEEPING  
- Clutter, obstructed walkways, poor storage  
Root Cause Focus: Poor workplace organization and maintenance

HUMAN ELEMENT  
- Missing PPE, unsafe worker behavior  
Root Cause Focus: Training, supervision, or safety culture gaps

PROCESS  
- Unsafe industrial processes, poor equipment arrangement  
Root Cause Focus: Operational or procedural risk

===============================
YOUR OUTPUT MUST INCLUDE
===============================

For each image, provide:

1. **Identified Risk Scenario** (choose ONE from the list above)  
2. **Risk Category** (Perils / Electrical / Housekeeping / Human Element / Process)  
3. **Root Cause Diagnosis** (engineering-based explanation)  
4. **Key Visual Evidence** (what in the image supports your diagnosis)  
5. **Potential Consequences** (structural, safety, or operational impact)

Be diagnostic and technical. Focus on WHY the problem exists, not just WHAT is visible.
Use professional engineering and risk assessment terminology.
"""

        user_prompt = """Analyze this image and identify the root cause of any safety or structural issues.

EXAMPLES OF ROOT CAUSE IDENTIFICATION:

Example 1: Image shows cracked, sunken pavement with vegetation growing in gaps
→ ROOT CAUSE: Ground subsidence or settlement
→ DIAGNOSIS: "Evidence of ground subsidence with differential settlement of pavement surface. Vegetation growth in cracks indicates long-term ground movement and inadequate maintenance. This suggests underlying soil instability or foundation settlement."

Example 2: Image shows exposed electrical wiring with damaged insulation
→ ROOT CAUSE: Electrical safety hazard
→ DIAGNOSIS: "Electrical hazard due to exposed wiring with compromised insulation. Risk of electrical shock, short circuit, or fire."

Example 3: Image shows cracked concrete foundation with visible separation from wall
→ ROOT CAUSE: Foundation failure due to subsidence
→ DIAGNOSIS: "Structural subsidence evident from foundation cracking and wall separation. Indicates differential ground settlement or soil movement beneath foundation."

Now analyze the provided image:
- What is the UNDERLYING PROBLEM causing what I see?
- Is this subsidence, ground settlement, or structural movement?
- Is this an electrical, flooding, fire, or other hazard?
- What category does this fall under (Perils, Electrical, Housekeeping, Human Element, Process)?

Provide a technical diagnosis focusing on the root cause, using specific terminology like "subsidence", "ground settlement", "structural deterioration", "foundation failure" when applicable."""

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
            print(f"  📝 Description: {description[:150]}...")
            
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
        temperature: float = 0.2
    ) -> Dict[str, Any]:
        """
        STAGE 3 (Image Mode): Generate recommendations using VLLM with retrieved context and image
        """
        print(f"\n💡 Stage 3: Generating recommendations with context and image...")
        
        # Format context
        context = self.format_context_for_recommendations(context_chunks)
        
        # System prompt for recommendation generation
        system_prompt = """You are an expert industrial safety consultant providing Risk Improvement Benchmark (RIB) recommendations.

Your role is to:
1. Confirm what you see in the image matches the RIB observations
2. Provide specific, actionable recommendations from the matched RIB sections
3. Cite applicable regulations and guidelines
4. Be precise and reference specific section numbers

Structure your response as:
- WHAT I SEE: Brief confirmation of the hazard/issue in the image
- MATCHED RIB SECTION: Which section(s) apply and why
- SPECIFIC RECOMMENDATIONS: Actionable steps from the RIB documentation
- APPLICABLE REGULATIONS: Relevant standards and guidelines
- PRIORITY ACTIONS: Most critical steps to take immediately"""

        # User prompt with image, description, and context
        user_prompt = f"""Based on the image and the retrieved RIB documentation below, provide specific RIB recommendations and regulations.

IMAGE DESCRIPTION (from initial analysis):
{image_description}

{context}

QUESTION: Provide me with the specific RIB recommendations and regulations.

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
        Complete 3-stage RAG pipeline for image input:
        1. VLLM describes the image
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
        
        # STAGE 1: Describe image
        description_result = self.describe_image(image_path)
        
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
                'answer': "No matching RIB documentation found for the observed issue.",
                'num_sources': 0,
                'success': False
            }
        
        # STAGE 3: Generate recommendations
        recommendation_result = self.generate_recommendations_with_image(
            image_description=image_description,
            context_chunks=matching_chunks,
            image_path=image_path,
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
        result = {
            'image_description': image_description,
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
    
    text_query = """I noticed significant ground settlement around our building foundation. 
    The pavement has multiple cracks and there's visible separation between the foundation 
    and walls. What RIB recommendations apply to this situation?"""
    
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
    
    image_path = "/home/amir/Desktop/MRE TSD/1.1 subsidence.png"
    
    result2 = rag.query_with_image(
        image_path=image_path,
        top_k=3,
        similarity_threshold=0.4
    )
    
    if result2.get('success'):
        print(f"\n📸 Image: {image_path}")
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