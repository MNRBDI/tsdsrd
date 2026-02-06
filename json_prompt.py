"""
json_prompt.py

VLM-Based RIB Detection System
Replaces OWLv2 object detection with pure VLM analysis using tier vocabularies.

This system:
1. Loads RIB vocabulary and observation data
2. Uses VLLM to analyze images directly against all 47 subsections
3. Calculates weighted scores with safety-critical priority boosts
4. Returns top 5 subsections compatible with existing RAG pipeline
5. Integrates semantic matching for final recommendation selection

Usage:
    # Start API server (replaces OWLv2 server on port 8010)
    python json_prompt.py
    
    # Test standalone detection
    python json_prompt.py test <image_path>
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import base64
import json
import requests
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer


class VLMRIBDetector:
    """
    VLM-based RIB detector that replaces OWLv2.
    Uses VLLM to analyze images and match against RIB subsection vocabularies.
    """
    
    def __init__(
        self,
        vllm_url: str = "http://localhost:8000",
        observation_json_path: str = "./observation_chunks(ni betul).json",
        embedding_model_name: str = "BAAI/bge-large-en-v1.5"
    ):
        """Initialize VLM detector with vocabularies and VLLM connection"""
        self.vllm_url = vllm_url.rstrip('/')
        self.chat_url = f"{self.vllm_url}/v1/chat/completions"
        
        print(f"Loading VLM-based RIB detector...")
        
        # Load RIB observations and vocabularies
        print(f"Loading RIB observations from {observation_json_path}...")
        self.rib_data = self._load_rib_observations(observation_json_path)
        
        # Build detection vocabularies
        print(f"Building detection vocabularies...")
        self.vocabularies = self._build_vocabularies()
        
        # Load embedding model for semantic matching
        print(f"Loading embedding model: {embedding_model_name}")
        device = "cpu"  # Force CPU to avoid cuBLAS issues
        self.embedding_model = SentenceTransformer(embedding_model_name, device=device)
        print(f"✓ Embedding model loaded on {device}")
        
        # Test VLLM connection
        self._test_vllm_connection()
        
        print(f"✓ VLM-based RIB detector initialized with {len(self.rib_data)} subsections")
    
    def _test_vllm_connection(self):
        """Test connection to VLLM server"""
        try:
            response = requests.get(f"{self.vllm_url}/v1/models", timeout=5)
            if response.status_code == 200:
                models = response.json()
                model_name = models['data'][0]['id']
                self.model_name = model_name
                print(f"✓ VLLM server connected: {model_name}")
            else:
                raise Exception(f"VLLM server returned status {response.status_code}")
        except Exception as e:
            raise Exception(f"Failed to connect to VLLM server at {self.vllm_url}: {e}")
    
    def _load_rib_observations(self, json_path: str) -> List[Dict[str, Any]]:
        """Load RIB observation data from JSON file"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Ensure it's a list
        if isinstance(data, dict):
            data = [data]
        
        return data
    
    def _build_vocabularies(self) -> Dict[str, Any]:
        """
        Build comprehensive vocabulary data structure for VLM prompting.
        Includes tier1 general objects and tier2 subsection-specific vocabularies.
        """
        
        # Tier 1: General RIB indicators (from OWLv2_tiers_improv.py)
        tier1_objects = [
            # PERILS
            "crack", "ground crack", "wall crack", "floor crack",
            "burst pipe", "water leak", "pipe leak",
            "slope", "hillside", "retaining wall",
            "drain", "drainage", "blocked drain", "water puddle",
            "flood water", "standing water", "water damage",
            "lightning rod", "lightning arrester",
            "tree", "dead tree", "fallen tree",
            
            # ELECTRICAL
            "electrical panel", "switchboard", "open panel",
            "transformer", "circuit breaker",
            "exposed wire", "loose wire", "burn mark", "scorch mark",
            
            # HOUSEKEEPING
            "grease floor", "dirty floor", "oily surface",
            "metal scrap", "scrap pile",
            "lpg cylinder", "gas cylinder",
            "storage drum", "metal drum", "oil drum",
            "cardboard box", "boxes", "combustible material",
            "generator", "dike wall", "bund",
            
            # HUMAN ELEMENT
            "fire extinguisher", "exit sign", "emergency exit",
            "smoking area", "cigarette", "no smoking sign",
            "safety sign", "warning sign",
            
            # PROCESS
            "spray booth", "battery", "battery rack",
            "overhead crane", "crane",
            "pipe label", "storage tank",
            "solar panel", "pv panel"
        ]
        
        # Tier 2: RIB subsection-specific vocabularies (from OWLv2_tiers_improv.py)
        tier2_rib_vocabulary = {
            "RIB_1.1_SUBSIDENCE": {
                "section": "1.1",
                "title": "Subsidence",
                "category": "PERILS",
                "priority_weight": 1.0,
                "vocabulary": [
                    "ground crack", "floor crack", "foundation crack", "settlement crack",
                    "sinking ground", "uneven floor", "subsidence damage", "ground movement",
                    "structure settlement", "differential settlement", "soil subsidence",
                    "pavement crack", "wall crack", "tilted building", "foundation issue"
                ]
            },
            "RIB_1.2_PIPE_BURSTING": {
                "section": "1.2",
                "title": "Pipe Bursting (Main Supply Pipe)",
                "category": "PERILS",
                "priority_weight": 1.4,
                "vocabulary": [
                    "burst pipe", "broken pipe", "pipe leak", "water leak",
                    "pipe rupture", "damaged pipe", "leaking pipe", "pipe failure",
                    "water spray", "pipe crack", "corroded pipe", "pipe damage",
                    "water damage", "pipe joint leak", "main pipe"
                ]
            },
            "RIB_1.3_HILL_SLOPE": {
                "section": "1.3",
                "title": "Building Near Hill Slope",
                "category": "PERILS",
                "priority_weight": 1.2,
                "vocabulary": [
                    "hill slope", "hillside", "steep slope", "retaining wall",
                    "slope erosion", "landslide risk", "slope failure", "hill cut",
                    "unstable slope", "slope vegetation", "hill face", "elevated terrain",
                    "slope drainage", "slope protection", "hillside building"
                ]
            },
            "RIB_1.4_DRAINAGE": {
                "section": "1.4",
                "title": "Inadequate Drainage System",
                "category": "PERILS",
                "priority_weight": 1.2,
                "vocabulary": [
                    "blocked drain", "clogged drain", "poor drainage", "water puddle",
                    "standing water", "drainage issue", "overflow", "drain blockage",
                    "inadequate drain", "water pooling", "flood risk", "drain debris",
                    "vegetation in drain", "drain cover", "drainage grate"
                ]
            },
            "RIB_1.5_FLOOD": {
                "section": "1.5",
                "title": "Flood Prone Area",
                "category": "PERILS",
                "priority_weight": 1.3,
                "vocabulary": [
                    "flood water", "flooding", "water level", "standing water",
                    "water damage", "flood mark", "water stain", "submerged area",
                    "flood risk", "water accumulation", "waterlogged", "flood prone",
                    "low lying area", "water ingress", "flood protection"
                ]
            },
            "RIB_1.6_LIGHTNING": {
                "section": "1.6",
                "title": "Inadequate Lightning Protection",
                "category": "PERILS",
                "priority_weight": 1.8,
                "vocabulary": [
                    "lightning rod", "lightning arrester", "lightning protection", "air terminal",
                    "lightning conductor", "grounding system", "surge protector", "strike counter",
                    "lightning strike", "protection system", "earthing", "lightning damage",
                    "protective device", "lightning risk", "tall structure"
                ]
            },
            "RIB_1.7_TREE": {
                "section": "1.7",
                "title": "Large and Tall Trees Near Buildings",
                "category": "PERILS",
                "priority_weight": 1.2,
                "vocabulary": [
                    "large tree", "tall tree", "dead tree", "fallen tree",
                    "tree branch", "overhanging branch", "tree trunk", "tree hazard",
                    "tree near building", "dangerous tree", "tree limb", "tree root",
                    "weak tree", "leaning tree", "tree damage"
                ]
            },
            "RIB_1.8_STRUCTURE": {
                "section": "1.8",
                "title": "Structural Distress",
                "category": "PERILS",
                "priority_weight": 1.3,
                "vocabulary": [
                    "structure crack", "wall crack", "ceiling damage", "structural damage",
                    "building distress", "deformation", "sagging", "bulging wall",
                    "structural failure", "strip ceiling", "ceiling panel", "damaged structure",
                    "wall deterioration", "structural defect", "building crack"
                ]
            },
            "RIB_1.9_VEHICLE": {
                "section": "1.9",
                "title": "Parked Vehicles Within 5m From Building",
                "category": "PERILS",
                "priority_weight": 1.1,
                "vocabulary": [
                    "parked vehicle", "lorry", "truck", "vehicle near building",
                    "car park", "parking area", "vehicle too close", "large vehicle",
                    "heavy vehicle", "delivery truck", "parked lorry", "vehicle proximity",
                    "parking violation", "vehicle hazard", "blocked access"
                ]
            },
            "RIB_1.10_EXTERNAL": {
                "section": "1.10",
                "title": "External Fire Exposure",
                "category": "PERILS",
                "priority_weight": 1.4,
                "vocabulary": [
                    "external fire", "adjacent building", "fence", "barrier",
                    "fire exposure", "nearby structure", "combustible fence", "plastic barrier",
                    "fire spread", "chain link fence", "perimeter fence", "gas station nearby",
                    "external hazard", "fire risk", "neighbor fire"
                ]
            },
            
            # ELECTRICAL (2.x) - Higher priority weights due to fire/shock risks
            "RIB_2.1_TRANSFORMER": {
                "section": "2.1",
                "title": "Transformer Not Tested",
                "category": "ELECTRICAL",
                "priority_weight": 1.3,
                "vocabulary": [
                    "transformer", "power transformer", "electrical transformer", "transformer oil",
                    "transformer tank", "transformer inspection", "oil transformer", "transformer test",
                    "transformer maintenance", "transformer room", "transformer bay", "HV transformer",
                    "transformer insulation", "transformer cooling", "transformer age"
                ]
            },
            "RIB_2.2_ELECTRICAL_INSTALLATION": {
                "section": "2.2",
                "title": "Electrical Installation Not Tested",
                "category": "ELECTRICAL",
                "priority_weight": 1.4,
                "vocabulary": [
                    "electrical wiring", "electrical cable", "electrical installation", "wiring system",
                    "switchgear", "circuit breaker", "relay", "electrical equipment",
                    "electrical test", "wiring inspection", "electrical panel", "circuit protection",
                    "electrical safety", "wiring age", "electrical component"
                ]
            },
            "RIB_2.3_COMBUSTIBLE_BOARD": {
                "section": "2.3",
                "title": "Accumulation of Combustible Material at Electrical Board",
                "category": "ELECTRICAL",
                "priority_weight": 2.5,  # HIGH PRIORITY - fire risk
                "vocabulary": [
                    "combustible material", "cardboard near panel", "boxes at panel", "storage at board",
                    "clutter at panel", "material at switchboard", "blocked panel", "combustible storage",
                    "paper near electrical", "debris at board", "obstruction at panel", "flammable near panel",
                    "boxes near switchboard", "storage too close", "combustible accumulation"
                ]
            },
            "RIB_2.4_PANEL_NOT_CLOSED": {
                "section": "2.4",
                "title": "Electrical Board Is Not Properly Closed",
                "category": "ELECTRICAL",
                "priority_weight": 3.0,  # HIGHEST PRIORITY - immediate shock/fire risk
                "vocabulary": [
                    "open panel", "open switchboard", "exposed panel", "panel door open",
                    "unclosed panel", "open electrical box", "missing panel cover", "panel ajar",
                    "exposed switchboard", "panel not closed", "open cabinet", "door missing",
                    "exposed electrical", "unsecured panel", "open enclosure"
                ]
            },
            "RIB_2.5_EXPOSED_WIRING": {
                "section": "2.5",
                "title": "Exposed Wiring or Improper Electrical Installation",
                "category": "ELECTRICAL",
                "priority_weight": 2.2,  # HIGH PRIORITY - shock risk
                "vocabulary": [
                    "exposed wire", "loose wire", "bare wire", "damaged cable",
                    "improper wiring", "hanging wire", "exposed cable", "wire damage",
                    "poor installation", "unsafe wiring", "unprotected wire", "wire insulation damage",
                    "loose connection", "wire exposure", "improper termination"
                ]
            },
            "RIB_2.6_BURN_MARK": {
                "section": "2.6",
                "title": "Presence of Burn Marks on Electrical Installation",
                "category": "ELECTRICAL",
                "priority_weight": 2.5,  # HIGH PRIORITY - indicates past fault
                "vocabulary": [
                    "burn mark", "scorch mark", "charred surface", "burnt panel",
                    "thermal damage", "heat damage", "discoloration", "burnt wire",
                    "fire damage", "overheating mark", "carbon deposit", "arcing damage",
                    "burnt component", "heat mark", "electrical burn"
                ]
            },
            "RIB_2.7_NO_KILL_SWITCH": {
                "section": "2.7",
                "title": "No Readily Accessible Kill Switch for Electrical Equipment",
                "category": "ELECTRICAL",
                "priority_weight": 1.5,
                "vocabulary": [
                    "kill switch", "emergency switch", "emergency stop", "disconnect switch",
                    "main switch", "isolation switch", "power cutoff", "safety switch",
                    "emergency disconnect", "accessible switch", "main breaker", "shutdown button",
                    "emergency control", "quick disconnect", "panic button"
                ]
            },
            
            # HOUSEKEEPING (3.x)
            "RIB_3.1_DIRTY_FLOOR": {
                "section": "3.1",
                "title": "Dirty Floor / Machinery",
                "category": "HOUSEKEEPING",
                "priority_weight": 1.1,
                "vocabulary": [
                    "dirty floor", "oily floor", "greasy floor", "grimy floor",
                    "dirty machinery", "grease buildup", "oil spill", "contaminated surface",
                    "unclean floor", "slippery floor", "greasy machine", "floor contamination",
                    "oily surface", "grime accumulation", "dirty equipment"
                ]
            },
            "RIB_3.2_UNUSED_METALS": {
                "section": "3.2",
                "title": "Unused Metals / Materials Kept Near Critical Equipment",
                "category": "HOUSEKEEPING",
                "priority_weight": 1.3,
                "vocabulary": [
                    "metal scrap", "unused metal", "scrap pile", "metal storage",
                    "debris", "clutter", "stored material", "scrap near equipment",
                    "metal accumulation", "unused material", "waste metal", "scrap metal",
                    "material storage", "junk near equipment", "metal debris"
                ]
            },
            "RIB_3.3_LPG": {
                "section": "3.3",
                "title": "LPG Cylinder Kept At Chiller Plant",
                "category": "HOUSEKEEPING",
                "priority_weight": 1.7,  # Higher priority - gas hazard
                "vocabulary": [
                    "lpg cylinder", "gas cylinder", "propane tank", "gas bottle",
                    "lpg storage", "cylinder at chiller", "gas at plant", "refrigerant cylinder",
                    "lpg tank", "gas container", "compressed gas", "cylinder storage",
                    "lpg at equipment", "gas near chiller", "cylinder improper storage"
                ]
            },
            "RIB_3.4_DRUM_WITHOUT_LID": {
                "section": "3.4",
                "title": "Drum Kept Without Lid at Outlet Discharge",
                "category": "HOUSEKEEPING",
                "priority_weight": 1.6,
                "vocabulary": [
                    "open drum", "drum without lid", "uncovered drum", "drum no cover",
                    "exposed drum", "lidless drum", "open container", "drum at discharge",
                    "unsealed drum", "open barrel", "drum without top", "uncovered container",
                    "exposed barrel", "drum opening", "container without lid"
                ]
            },
            "RIB_3.5_DRUM_OUTSIDE_BUND": {
                "section": "3.5",
                "title": "Drum Kept Outside Bund / Dike",
                "category": "HOUSEKEEPING",
                "priority_weight": 1.6,
                "vocabulary": [
                    "drum outside bund", "drum outside dike", "no containment", "drum not contained",
                    "missing bund", "no spill containment", "drum outside barrier", "uncontained drum",
                    "no dike", "drum without containment", "missing containment", "drum exposed",
                    "no secondary containment", "drum outside protection", "unprotected drum"
                ]
            },
            "RIB_3.6_COMBUSTIBLE_NEAR_EQUIPMENT": {
                "section": "3.6",
                "title": "Combustible Material Kept Near Fire Equipment",
                "category": "HOUSEKEEPING",
                "priority_weight": 1.8,  # Higher priority - blocks fire safety
                "vocabulary": [
                    "combustible near extinguisher", "blocked fire equipment", "material at fire panel",
                    "obstruction at fire equipment", "boxes near extinguisher", "blocked fire access",
                    "combustible at fire station", "storage at fire equipment", "fire equipment blocked",
                    "material obstructing", "boxes at fire panel", "clutter at fire equipment",
                    "combustible blocking", "access blocked", "fire safety obstruction"
                ]
            },
            "RIB_3.7_BLOCKED_EXIT": {
                "section": "3.7",
                "title": "Emergency Exit Blocked",
                "category": "HOUSEKEEPING",
                "priority_weight": 1.9,  # HIGH PRIORITY - life safety
                "vocabulary": [
                    "blocked exit", "obstructed exit", "exit blocked", "emergency exit blocked",
                    "blocked door", "obstructed escape", "exit obstruction", "door blocked",
                    "blocked egress", "escape blocked", "exit not accessible", "blocked escape route",
                    "obstructed doorway", "exit impediment", "blocked fire exit"
                ]
            },
            "RIB_3.8_GENERATOR_NO_BUND": {
                "section": "3.8",
                "title": "Generator Not Provided With Bund / Dike",
                "category": "HOUSEKEEPING",
                "priority_weight": 1.5,
                "vocabulary": [
                    "generator", "generator no bund", "generator without dike", "no containment",
                    "generator no containment", "diesel generator", "generator without barrier",
                    "uncontained generator", "generator no protection", "missing bund",
                    "generator without spill control", "no dike", "generator exposed",
                    "no secondary containment", "generator base"
                ]
            },
            
            # HUMAN ELEMENT (4.x)
            "RIB_4.1_SMOKING_SHED": {
                "section": "4.1",
                "title": "No Proper Smoking Shed",
                "category": "HUMAN_ELEMENT",
                "priority_weight": 1.3,
                "vocabulary": [
                    "smoking area", "smoking shed", "designated smoking", "smoking zone",
                    "no smoking shed", "improper smoking area", "smoking location", "smoking facility",
                    "smoking shelter", "smoking station", "smoking point", "smoking designation",
                    "smoking signage", "smoking control", "no designated area"
                ]
            },
            "RIB_4.2_SMOKING_NEAR_COMBUSTIBLE": {
                "section": "4.2",
                "title": "Smoking Shed Located Near Combustible Materials",
                "category": "HUMAN_ELEMENT",
                "priority_weight": 1.7,
                "vocabulary": [
                    "smoking near paper", "smoking near combustible", "cigarette near materials",
                    "smoking too close", "smoking hazard", "smoking near storage", "unsafe smoking area",
                    "smoking near flammable", "cigarette risk", "smoking proximity", "smoking danger",
                    "smoking near stock", "smoking location unsafe", "combustible near smoking",
                    "paper stock nearby"
                ]
            },
            "RIB_4.3_HOT_WORK": {
                "section": "4.3",
                "title": "Hot Work (Welding / Cutting Operations)",
                "category": "HUMAN_ELEMENT",
                "priority_weight": 1.5,  # Higher priority - active fire risk
                "vocabulary": [
                    "welding", "cutting torch", "hot work", "welding equipment",
                    "grinding", "flame cutting", "welding operation", "spark generating",
                    "torch work", "hot cutting", "welding area", "welding machine",
                    "cutting operation", "thermal cutting", "welding setup"
                ]
            },
            "RIB_4.4_FLAMMABLE_LIQUID": {
                "section": "4.4",
                "title": "Flammable Liquid (Class I) Stored in Occupied Building",
                "category": "HUMAN_ELEMENT",
                "priority_weight": 1.6,
                "vocabulary": [
                    "flammable liquid", "class 1 liquid", "gasoline", "solvent",
                    "petrol", "volatile liquid", "flammable storage", "fuel storage",
                    "hazardous liquid", "flammable container", "solvent storage", "volatile storage",
                    "flammable in building", "fuel container", "class i storage"
                ]
            },
            "RIB_4.5_FLAMMABLE_GAS": {
                "section": "4.5",
                "title": "Flammable Gas Stored in Occupied Building",
                "category": "HUMAN_ELEMENT",
                "priority_weight": 1.7,
                "vocabulary": [
                    "flammable gas", "gas cylinder", "lpg", "compressed gas",
                    "gas storage", "gas bottle", "propane", "acetylene",
                    "gas container", "gas in building", "hazardous gas", "gas cylinder storage",
                    "gas improper storage", "gas risk", "cylinder in building"
                ]
            },
            "RIB_4.6_COMBUSTIBLE_DUST": {
                "section": "4.6",
                "title": "Combustible Dust",
                "category": "HUMAN_ELEMENT",
                "priority_weight": 1.5,
                "vocabulary": [
                    "dust", "combustible dust", "dust accumulation", "dust layer",
                    "powder", "fine particles", "dust buildup", "dust hazard",
                    "dust cloud", "dust explosion risk", "particulate", "dust deposit",
                    "dust contamination", "dust suspended", "airborne dust"
                ]
            },
            "RIB_4.7_FLAMMABLE_SOLID": {
                "section": "4.7",
                "title": "Flammable Solid",
                "category": "HUMAN_ELEMENT",
                "priority_weight": 1.4,
                "vocabulary": [
                    "flammable solid", "combustible solid", "solid fuel", "flammable material",
                    "combustible storage", "solid combustible", "flammable substance", "solid flammable",
                    "combustible product", "flammable goods", "solid fuel storage", "flammable stock",
                    "combustible inventory", "solid hazard", "flammable pile"
                ]
            },
            "RIB_4.8_PROCESS_FLAMMABLE": {
                "section": "4.8",
                "title": "Flammable Materials in Process",
                "category": "HUMAN_ELEMENT",
                "priority_weight": 1.6,
                "vocabulary": [
                    "flammable in process", "material handling", "flammable operation", "process material",
                    "flammable manufacturing", "material processing", "flammable production", "process hazard",
                    "flammable workflow", "handling flammable", "process flammable", "production material",
                    "flammable in use", "active process", "material in process"
                ]
            },
            "RIB_4.9_HOUSEKEEPING_COMBUSTIBLE": {
                "section": "4.9",
                "title": "Housekeeping (Combustible Material)",
                "category": "HUMAN_ELEMENT",
                "priority_weight": 1.4,
                "vocabulary": [
                    "combustible clutter", "combustible debris", "waste combustible", "trash accumulation",
                    "cardboard waste", "paper waste", "combustible garbage", "waste material",
                    "packaging waste", "combustible trash", "waste cardboard", "paper clutter",
                    "combustible waste", "garbage combustible", "waste accumulation"
                ]
            },
            
            # PROCESS (5.x) - Varied priorities based on process hazards
            "RIB_5.1_SPRAY_COATING": {
                "section": "5.1",
                "title": "Spray Coating Operation Without Proper Booth",
                "category": "PROCESS",
                "priority_weight": 1.5,
                "vocabulary": [
                    "spray coating", "spray booth", "coating operation", "spray painting",
                    "paint spraying", "spray area", "coating booth", "spray enclosure",
                    "painting operation", "spray chamber", "coating facility", "spray setup",
                    "painting booth", "spray room", "coating process"
                ]
            },
            "RIB_5.2_BATTERY_CHARGING": {
                "section": "5.2",
                "title": "Battery Charging Area Not Properly Segregated",
                "category": "PROCESS",
                "priority_weight": 1.6,
                "vocabulary": [
                    "battery charging", "charging station", "battery area", "charging room",
                    "battery rack", "charging bay", "battery maintenance", "charging socket",
                    "battery storage", "charging area", "battery room", "charging equipment",
                    "battery segregation", "charging zone", "battery facility"
                ]
            },
            "RIB_5.3_RACK_STORAGE": {
                "section": "5.3",
                "title": "Height of Rack Storage Greater Than 6m",
                "category": "PROCESS",
                "priority_weight": 1.2,
                "vocabulary": [
                    "high rack", "tall racking", "rack storage", "storage rack",
                    "high storage", "tall shelving", "rack system", "elevated storage",
                    "warehouse rack", "storage height", "racking system", "vertical storage",
                    "high bay", "tall rack", "storage system"
                ]
            },
            "RIB_5.4_RACK_COMBUSTIBLE": {
                "section": "5.4",
                "title": "Storage of Combustible in High Rack Without Proper Protection",
                "category": "PROCESS",
                "priority_weight": 1.7,
                "vocabulary": [
                    "combustible rack", "high rack combustible", "rack storage combustible", "stored combustible",
                    "rack fire risk", "combustible high storage", "material in rack", "combustible height",
                    "rack protection", "combustible elevated", "storage combustible", "rack material",
                    "high combustible storage", "rack fire protection", "combustible inventory"
                ]
            },
            "RIB_5.5_OVERHEAD_CRANE": {
                "section": "5.5",
                "title": "Overhead Crane Operating Near Critical Equipment",
                "category": "PROCESS",
                "priority_weight": 1.4,
                "vocabulary": [
                    "overhead crane", "bridge crane", "crane operation", "gantry crane",
                    "crane near equipment", "crane hazard", "crane proximity", "crane track",
                    "lifting equipment", "crane boom", "crane near critical", "crane operation area",
                    "crane movement", "crane zone", "crane danger"
                ]
            },
            "RIB_5.6_SPRAY_PAINTING": {
                "section": "5.6",
                "title": "Spray Painting Area Not Segregated",
                "category": "PROCESS",
                "priority_weight": 1.6,
                "vocabulary": [
                    "spray painting", "paint area", "painting zone", "spray room",
                    "painting not segregated", "paint operation", "spray facility", "painting area",
                    "paint booth", "spray segregation", "painting enclosure", "paint spraying",
                    "spray location", "painting space", "paint hazard"
                ]
            },
            "RIB_5.7_FIRE_RESISTANT_CABINET": {
                "section": "5.7",
                "title": "No Fire Resistant Cabinet for Flammable Liquid Storage",
                "category": "PROCESS",
                "priority_weight": 1.7,
                "vocabulary": [
                    "fire cabinet", "fire resistant cabinet", "safety cabinet", "flammable cabinet",
                    "storage cabinet", "chemical cabinet", "flammable storage", "liquid storage cabinet",
                    "fire rated cabinet", "safety storage", "flammable locker", "chemical storage",
                    "cabinet storage", "fire protection cabinet", "storage container"
                ]
            },
            "RIB_5.8_STORAGE_TANK": {
                "section": "5.8",
                "title": "No Regular Inspection and Testing For Storage Tank / Piping",
                "category": "PROCESS",
                "priority_weight": 1.5,
                "vocabulary": [
                    "storage tank", "tank inspection", "tank testing", "piping",
                    "tank maintenance", "pipe inspection", "tank condition", "pipe testing",
                    "tank system", "pipe network", "tank integrity", "pipe maintenance",
                    "tank age", "pipe corrosion", "tank safety"
                ]
            },
            "RIB_5.9_PALM_KERNEL": {
                "section": "5.9",
                "title": "Palm Kernel Silo",
                "category": "PROCESS",
                "priority_weight": 1.3,
                "vocabulary": [
                    "palm kernel", "kernel silo", "silo", "palm storage",
                    "kernel storage", "silo structure", "palm kernel storage", "bulk silo",
                    "kernel handling", "silo system", "palm processing", "kernel facility",
                    "silo operation", "palm kernel process", "silo tank"
                ]
            },
            "RIB_5.10_WEIR": {
                "section": "5.10",
                "title": "Possibility of Dust Explosion (Weir / Dust Extraction System)",
                "category": "PROCESS",
                "priority_weight": 1.6,
                "vocabulary": [
                    "weir", "dust extraction", "dust system", "dust collector",
                    "extraction system", "dust control", "dust ventilation", "dust explosion risk",
                    "dust removal", "extraction equipment", "dust hazard", "dust handling",
                    "dust prevention", "extraction fan", "dust management"
                ]
            },
            "RIB_5.11_SOLAR_FARM": {
                "section": "5.11",
                "title": "No Regular Inspection and Maintenance of PV Solar Farm",
                "category": "PROCESS",
                "priority_weight": 1.2,
                "vocabulary": [
                    "solar panel", "pv panel", "solar farm", "solar array",
                    "photovoltaic", "solar installation", "pv system", "solar module",
                    "panel array", "solar field", "pv farm", "solar maintenance",
                    "solar inspection", "panel condition", "solar equipment"
                ]
            },
            "RIB_5.12_GAS_PIPELINE": {
                "section": "5.12",
                "title": "No Regular Inspection and Maintenance for Gas Pipeline",
                "category": "PROCESS",
                "priority_weight": 1.6,
                "vocabulary": [
                    "gas pipeline", "gas pipe", "pipeline inspection", "gas line",
                    "pipeline maintenance", "gas network", "pipe inspection", "gas system",
                    "pipeline condition", "gas piping", "pipeline testing", "gas infrastructure",
                    "pipe corrosion", "gas leak risk", "pipeline integrity"
                ]
            },
            "RIB_5.13_UPS": {
                "section": "5.13",
                "title": "No Regular Inspection and Maintenance for UPS",
                "category": "PROCESS",
                "priority_weight": 1.3,
                "vocabulary": [
                    "ups", "uninterruptible power", "ups system", "backup power",
                    "ups battery", "power backup", "ups unit", "ups room",
                    "ups maintenance", "ups inspection", "battery backup", "ups equipment",
                    "ups testing", "ups condition", "power system"
                ]
            },
            "RIB_5.14_CABLE_JOINT": {
                "section": "5.14",
                "title": "No Regular Inspection and Maintenance for Cable Joint / Terminations",
                "category": "PROCESS",
                "priority_weight": 1.4,
                "vocabulary": [
                    "cable joint", "cable termination", "cable connection", "joint",
                    "termination", "cable splice", "cable connection point", "joint box",
                    "cable junction", "termination kit", "cable end", "joint inspection",
                    "cable maintenance", "joint condition", "termination inspection"
                ]
            },
            "RIB_5.15_BRIDGE_PIER": {
                "section": "5.15",
                "title": "Rust / Corrosion on Bridge Pier Supporting Pipe Rack",
                "category": "PROCESS",
                "priority_weight": 1.3,
                "vocabulary": [
                    "rust", "corrosion", "bridge pier", "pier",
                    "corroded pier", "rusted structure", "bridge support", "pier corrosion",
                    "structural rust", "pipe rack support", "pier deterioration", "corroded support",
                    "rust damage", "pier condition", "corrosion damage"
                ]
            },
            "RIB_5.16_STATIC_ELECTRICITY": {
                "section": "5.16",
                "title": "Static Electricity During the Unwinding and Winding of Plastic Sheets",
                "category": "PROCESS",
                "priority_weight": 1.4,
                "vocabulary": [
                    "static electricity", "plastic sheet", "unwinding", "winding",
                    "plastic roll", "static discharge", "plastic film", "roller",
                    "static charge", "plastic handling", "film roll", "static risk",
                    "plastic process", "static buildup", "film winding"
                ]
            }
        }
        
        # Build subsection lookup from RIB data
        subsection_lookup = {}
        for item in self.rib_data:
            section_num = item['section_number']
            subsection_lookup[section_num] = {
                'id': item['id'],
                'section_number': section_num,
                'title': item['title'],
                'category': item['category'],
                'observation': item['observation'],
                'recommendation': item['recommendation'],
                'regulation': item.get('regulation', '')
            }
        
        # Merge tier2 vocabulary with observation data
        enriched_subsections = {}
        for key, vocab_data in tier2_rib_vocabulary.items():
            section_num = vocab_data['section']
            if section_num in subsection_lookup:
                enriched_subsections[key] = {
                    **vocab_data,
                    **subsection_lookup[section_num]
                }
            else:
                enriched_subsections[key] = vocab_data
        
        return {
            'tier1_objects': tier1_objects,
            'tier2_subsections': enriched_subsections
        }
    
    def _image_to_base64(self, image: Image.Image, quality: int = 85) -> str:
        """Convert PIL Image to base64 string"""
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def _build_detection_prompt(self) -> str:
        """
        DEPRECATED - No longer used. Kept for reference only.
        Replaced by two-stage approach with _build_focused_detection_prompt.
        """
        return ""
    
    def detect_with_vlm(
        self,
        image: Image.Image,
        max_tokens: int = 1000,
        temperature: float = 0.1
    ) -> Dict[str, Any]:
        """
        Use VLM to detect RIB subsections in image.
        Returns structured detection results.
        Two-stage approach for efficiency:
        1. Stage 1: Describe image and detect general category
        2. Stage 2: Match against top subsections for that category
        """
        
        # Convert image to base64
        image_url = f"data:image/jpeg;base64,{self._image_to_base64(image)}"
        
        # STAGE 1: Image description and category detection (simpler, faster)
        print(f"    Stage 1: Analyzing image and detecting categories...")
        stage1_prompt = """You are an industrial safety inspector. Analyze this image and respond ONLY with valid JSON in this format:

{
  "image_description": "Brief description of what you see (2-3 sentences)",
  "visible_objects": ["object1", "object2", ...],
  "suspected_categories": ["CATEGORY1", "CATEGORY2"],
  "suspected_sections": ["1.x", "2.x", ...]
}

Categories: PERILS, ELECTRICAL, HOUSEKEEPING, HUMAN_ELEMENT, PROCESS

Only return JSON, nothing else."""

        messages_stage1 = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    },
                    {
                        "type": "text",
                        "text": stage1_prompt
                    }
                ]
            }
        ]
        
        payload_stage1 = {
            "model": self.model_name,
            "messages": messages_stage1,
            "max_tokens": 300,
            "temperature": temperature,
            "top_p": 0.9,
            "stream": False
        }
        
        try:
            response_stage1 = requests.post(self.chat_url, json=payload_stage1, timeout=60)
            response_stage1.raise_for_status()
            result_stage1 = response_stage1.json()
            content_stage1 = result_stage1['choices'][0]['message']['content']
            
            # Clean markdown if present
            if "```json" in content_stage1:
                content_stage1 = content_stage1.split("```json")[1].split("```")[0].strip()
            elif "```" in content_stage1:
                content_stage1 = content_stage1.split("```")[1].split("```")[0].strip()
            
            stage1_data = json.loads(content_stage1)
            image_description = stage1_data.get('image_description', '')
            visible_objects = stage1_data.get('visible_objects', [])
            suspected_categories = stage1_data.get('suspected_categories', [])
            suspected_sections = stage1_data.get('suspected_sections', [])
            
            print(f"    ✓ Detected categories: {', '.join(suspected_categories)}")
            
            # STAGE 2: Detailed scoring against relevant subsections
            print(f"    Stage 2: Scoring against relevant subsections...")
            detection_prompt = self._build_focused_detection_prompt(
                image_description, 
                suspected_categories, 
                suspected_sections
            )
            
            messages_stage2 = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url}
                        },
                        {
                            "type": "text",
                            "text": detection_prompt
                        }
                    ]
                }
            ]
            
            payload_stage2 = {
                "model": self.model_name,
                "messages": messages_stage2,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.9,
                "stream": False
            }
            
            response = requests.post(self.chat_url, json=payload_stage2, timeout=120)
            response.raise_for_status()
            result = response.json()
            
            # Extract response content
            content = result['choices'][0]['message']['content']
            print(f"    ✓ Stage 2 scoring complete")
            
            # Parse JSON response from VLM
            # Remove markdown code fences if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            detection_result = json.loads(content)
            
            # Merge stage1 and stage2 data
            detection_result['image_description'] = image_description
            detection_result['detected_tier1_objects'] = visible_objects
            
            return {
                'success': True,
                'data': detection_result,
                'raw_response': content
            }
            
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse VLM JSON response: {e}")
            print(f"Raw response: {content[:500]}...")
            return {
                'success': False,
                'error': f'JSON parse error: {str(e)}',
                'raw_response': content
            }
        except requests.exceptions.Timeout as e:
            print(f"❌ VLM request timeout: {e}")
            print(f"   Try restarting VLLM server: sudo systemctl restart vllm")
            return {
                'success': False,
                'error': f'Request timeout - VLLM server may be unresponsive: {str(e)}'
            }
        except Exception as e:
            print(f"❌ VLM detection error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _build_focused_detection_prompt(
        self,
        image_description: str,
        categories: List[str],
        suspected_sections: List[str]
    ) -> str:
        """
        Build optimized prompt focused only on relevant subsections.
        Significantly reduces token count compared to all 47 subsections.
        """
        
        # Filter subsections to only those matching detected categories and sections
        relevant_subsections = []
        
        for key, data in self.vocabularies['tier2_subsections'].items():
            if data['category'] in categories or data['section'] in suspected_sections:
                relevant_subsections.append({
                    'key': key,
                    'section': data['section'],
                    'title': data['title'],
                    'priority_weight': data['priority_weight'],
                    'vocabulary': data['vocabulary'][:6]  # Reduced from 8
                })
        
        # If no matches found, include critical sections anyway
        if not relevant_subsections:
            for key, data in self.vocabularies['tier2_subsections'].items():
                if data['priority_weight'] >= 1.5:  # Include critical items
                    relevant_subsections.append({
                        'key': key,
                        'section': data['section'],
                        'title': data['title'],
                        'priority_weight': data['priority_weight'],
                        'vocabulary': data['vocabulary'][:6]
                    })
        
        # Limit to top 20 most relevant to reduce token count
        relevant_subsections = relevant_subsections[:20]
        
        prompt = f"""Score each RIB subsection based on the image. Return TOP 5 matches ONLY as JSON.

IMAGE: {image_description}
CATEGORIES: {', '.join(categories)}

SUBSECTIONS TO SCORE ({len(relevant_subsections)}):
{json.dumps(relevant_subsections, indent=1)}

SCORING:
- Base score = (visible vocabulary terms / total terms) × 100
- Final score = base score × priority_weight
- Return exactly 5 top matches

RESPONSE (JSON ONLY):
{{
  "tier1_detection_count": <number>,
  "broad_categories": {json.dumps(categories)},
  "subsection_matches": [
    {{"rank": 1, "key": "RIB_X.Y", "section": "X.Y", "title": "...", "score": 85.5, "detection_count": 7, "justification": "..."}}
  ]
}}"""

        return prompt
    
    def format_for_rag_pipeline(self, vlm_detection: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format VLM detection results to match OWLv2 response structure.
        This ensures compatibility with existing RAG pipeline.
        """
        
        if not vlm_detection.get('success'):
            return {
                'success': False,
                'error': vlm_detection.get('error', 'Detection failed'),
                'data': {}
            }
        
        data = vlm_detection['data']
        subsection_matches = data.get('subsection_matches', [])
        
        # Format top 5 subsections
        top5_rib_subsections = []
        for match in subsection_matches[:5]:
            top5_rib_subsections.append({
                'rank': match['rank'],
                'section': f"{match['section']} - {match['title']}",
                'score': match['score'],
                'detection_count': match['detection_count']
            })
        
        # Determine primary subsection (highest score)
        primary_rib_subsection = top5_rib_subsections[0]['section'] if top5_rib_subsections else "Unknown"
        
        return {
            'success': True,
            'data': {
                'tier1_detection_count': data.get('tier1_detection_count', 0),
                'broad_categories': data.get('broad_categories', []),
                'primary_rib_subsection': primary_rib_subsection,
                'top5_rib_subsections': top5_rib_subsections,
                'image_description': data.get('image_description', ''),
                'detected_objects': data.get('detected_tier1_objects', [])
            }
        }
    
    def detect_image(self, image_path: str) -> Dict[str, Any]:
        """
        Main detection method - analyze image and return top 5 RIB subsections.
        This is the primary interface method called by the RAG pipeline.
        """
        
        print(f"\n🔍 VLM-based RIB Detection: {image_path}")
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            
            # Perform VLM detection
            print(f"  → Analyzing image with VLM...")
            vlm_result = self.detect_with_vlm(image)
            
            if not vlm_result.get('success'):
                print(f"  ❌ Detection failed: {vlm_result.get('error')}")
                return vlm_result
            
            # Format results for RAG pipeline
            formatted_result = self.format_for_rag_pipeline(vlm_result)
            
            # Print summary
            if formatted_result.get('success'):
                data = formatted_result['data']
                print(f"  ✓ Detection complete:")
                print(f"    - Tier 1 objects: {data['tier1_detection_count']}")
                print(f"    - Categories: {', '.join(data['broad_categories'])}")
                print(f"    - Primary: {data['primary_rib_subsection']}")
                print(f"    - Top 5 subsections identified")
            
            return formatted_result
            
        except Exception as e:
            print(f"  ❌ Detection error: {e}")
            return {
                'success': False,
                'error': str(e),
                'data': {}
            }


# FastAPI server application
app = FastAPI(title="VLM-Based RIB Detection API")

# Initialize detector (singleton)
detector = None

@app.on_event("startup")
async def startup_event():
    """Initialize detector on server startup"""
    global detector
    print("\n" + "="*80)
    print("INITIALIZING VLM-BASED RIB DETECTION SERVER")
    print("="*80)
    
    detector = VLMRIBDetector(
        vllm_url="http://localhost:8000",
        observation_json_path="./observation_chunks(ni betul).json",
        embedding_model_name="BAAI/bge-large-en-v1.5"
    )
    
    print("="*80)
    print("SERVER READY - VLM detector initialized")
    print("="*80)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "VLM-Based RIB Detection API",
        "endpoints": ["/detect-top5", "/health"]
    }

@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "detector_initialized": detector is not None,
        "vllm_url": detector.vllm_url if detector else None,
        "subsection_count": len(detector.rib_data) if detector else 0
    }

@app.post("/detect-top5")
async def detect_top5(file: UploadFile = File(...)):
    """
    Detect top 5 RIB subsections in uploaded image.
    Compatible with existing multimodal_rag_observation_basic.py integration.
    """
    
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not initialized")
    
    try:
        # Read uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Save temporarily
        temp_path = f"/tmp/{file.filename}"
        image.save(temp_path)
        
        # Perform detection
        result = detector.detect_image(temp_path)
        
        # Clean up
        Path(temp_path).unlink(missing_ok=True)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


# Standalone testing function
def test_detector(image_path: str):
    """Test the detector on a single image"""
    print("\n" + "="*80)
    print("VLM-BASED RIB DETECTOR - STANDALONE TEST")
    print("="*80)
    
    # Initialize detector
    detector = VLMRIBDetector(
        vllm_url="http://localhost:8000",
        observation_json_path="./observation_chunks(ni betul).json"
    )
    
    # Perform detection
    result = detector.detect_image(image_path)
    
    # Print results
    print("\n" + "="*80)
    print("DETECTION RESULTS")
    print("="*80)
    print(json.dumps(result, indent=2))
    
    return result


if __name__ == "__main__":
    import sys
    import uvicorn
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Test mode
        if len(sys.argv) < 3:
            print("Usage: python json_prompt.py test <image_path>")
            sys.exit(1)
        
        image_path = sys.argv[2]
        test_detector(image_path)
    
    else:
        # Server mode
        print("\nStarting VLM-Based RIB Detection API Server...")
        print("Run with 'python json_prompt.py test <image_path>' to test standalone")
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8010,  # Same port as OWLv2 server for drop-in replacement
            log_level="info"
        )
