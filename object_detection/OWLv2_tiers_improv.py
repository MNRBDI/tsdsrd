"""
RIB-Aligned Two-Tier OWLv2 Object Detection System
IMPROVED VERSION with optimized vocabularies for better accuracy

Key improvements over original:
- Shortened, concrete vocabulary phrases (2-4 words) for better OWLv2 performance
- Visual-focused descriptions instead of abstract states
- Lower thresholds for critical safety items
- Better category mapping
- Enhanced red circle detection
- Improved scoring logic with category-specific boosts

Installation:
    pip install transformers torch pillow matplotlib opencv-python

Usage:
    python rib_owlv2_detector_improved.py --image path/to/image.jpg
    
Red Circle Feature:
    - Images with a red circle will automatically crop to that region
    - Only objects within the red circle are detected
    - Improves accuracy for focused inspection areas
"""

from transformers import Owlv2Processor, Owlv2ForObjectDetection
from PIL import Image
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Any, Optional, Tuple
import json
import argparse
from pathlib import Path
import cv2
import numpy as np


class RIBAlignedDetector:
    """
    RIB-aligned two-tier object detection system with OPTIMIZED vocabularies
    """
    
    def __init__(self, model_name: str = "google/owlv2-base-patch16-ensemble", max_image_size: int = 1024):
        """Initialize detector with OWLv2 model"""
        print(f"Loading OWLv2 model: {model_name}...")
        self.processor = Owlv2Processor.from_pretrained(model_name)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_name)
        
        # Force CPU mode due to GPU cuBLAS initialization issues
        # OWLv2 is lightweight enough for CPU, VLLM remains on GPU
        self.device = "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        self.max_image_size = max_image_size
        print(f"✓ Model loaded successfully on {self.device}")
        
        self._setup_rib_detection_vocabulary()
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Resize image only if larger than 1024x1024 to speed up inference"""
        width, height = image.size
        # Only resize if BOTH dimensions are larger than 1024px
        if width > 1024 and height > 1024:
            # Calculate new dimensions maintaining aspect ratio
            if width > height:
                new_width = self.max_image_size
                new_height = int(height * (self.max_image_size / width))
            else:
                new_height = self.max_image_size
                new_width = int(width * (self.max_image_size / height))
            
            image = image.resize((new_width, new_height), Image.LANCZOS)
            print(f"  ℹ️ Resized image from {width}x{height} to {new_width}x{new_height} for faster processing")
        else:
            print(f"  ℹ️ Image size {width}x{height} is within limits, no resize needed")
        
        return image
    
    def detect_red_circle(self, image: Image.Image) -> Optional[Tuple[Image.Image, Dict[str, Any]]]:
        """
        Detect red circle in image and extract region within it.
        Returns: (cropped_image, circle_info) if circle found, else (None, None)
        """
        # Convert PIL image to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Convert to HSV for better red color detection
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        # Define range for red color (red wraps around in HSV, so we check two ranges)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        # Create mask for red colors
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        
        # Detect circles in the red mask
        circles = cv2.HoughCircles(
            red_mask,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=20,
            maxRadius=None
        )
        
        if circles is None or len(circles[0]) == 0:
            return None, None
        
        # Get the largest circle (most prominent)
        circles = circles[0]
        circles = sorted(circles, key=lambda c: c[2], reverse=True)  # Sort by radius
        x, y, radius = circles[0]
        x, y, radius = int(x), int(y), int(radius)
        
        # Create circular mask for cropping
        h, w = cv_image.shape[:2]
        circular_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(circular_mask, (x, y), radius, 255, -1)
        
        # Extract region within the circle
        result = cv2.bitwise_and(cv_image, cv_image, mask=circular_mask)
        
        # Crop to bounding box of circle
        x_min = max(0, x - radius)
        y_min = max(0, y - radius)
        x_max = min(w, x + radius)
        y_max = min(h, y + radius)
        
        cropped = result[y_min:y_max, x_min:x_max]
        
        # Convert back to PIL image
        cropped_pil = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        
        circle_info = {
            'center': (x, y),
            'radius': radius,
            'bbox': (x_min, y_min, x_max, y_max),
            'detected': True
        }
        
        print(f"\n✓ Red circle detected at center ({x}, {y}) with radius {radius}")
        print(f"  Detection will be limited to the red circle region only")
        
        return cropped_pil, circle_info
    
    def _setup_rib_detection_vocabulary(self):
        """Setup detection vocabulary with OPTIMIZED short, visual phrases"""
        
        # TIER 1: General detection vocabulary (IMPROVED - shorter, more visual)
        self.tier1_objects = [
            # 1.0 PERILS - General indicators
            "crack", "ground crack", "wall crack", "floor crack",
            "burst pipe", "water leak", "pipe leak",
            "slope", "hillside", "retaining wall",
            "drain", "drainage", "blocked drain", "water puddle",
            "flood water", "standing water", "water damage",
            "lightning rod", "lightning arrester",
            "tree", "dead tree", "fallen tree",
    
            # 2.0 ELECTRICAL - General indicators
            "electrical panel", "switchboard", "open panel",
            "transformer", "circuit breaker",
            "exposed wire", "loose wire", "burn mark", "scorch mark",
    
            # 3.0 HOUSEKEEPING - General indicators
            "grease floor", "dirty floor", "oily surface",
            "metal scrap", "scrap pile",
            "lpg cylinder", "gas cylinder",
            "storage drum", "metal drum", "oil drum",
            "cardboard box", "boxes", "combustible material",
            "generator", "dike wall", "bund",
    
            # 4.0 HUMAN ELEMENT - General indicators
            "fire extinguisher", "exit sign", "emergency exit",
            "smoking area", "cigarette", "no smoking sign",
            "safety sign", "warning sign",
    
            # 5.0 PROCESS - General indicators
            "spray booth", "battery", "battery rack",
            "overhead crane", "crane",
            "pipe label", "storage tank",
            "solar panel", "pv panel"
        ]
        
        # TIER 2: OPTIMIZED vocabularies (SHORT, VISUAL, CONCRETE)
        self.tier2_rib_vocabulary = {
            # ============================================================
            # 1.0 PERILS - 15 terms each
            # ============================================================
            'RIB_1.1_SUBSIDENCE': [
                # Ground/floor evidence
                "ground crack", "floor crack", "foundation crack",
                "settlement crack", "sinking ground", "uneven floor",
                # Structural evidence
                "subsided area", "ground depression", "floor settlement",
                "tilted floor", "sunken pavement", "settled foundation",
                # M&E room context (from observation)
                "mne room damage", "carpark subsidence", "building settlement"
            ],

            'RIB_1.2_PIPE_BURSTING': [
                # Visible pipe damage
                "burst pipe", "broken pipe", "ruptured pipe",
                "pipe leak", "water leak", "leaking pipe",
                # Corrosion and wear
                "corroded pipe", "damaged pipe", "pipe rupture",
                "rusted pipe", "pipe joint failure", "cracked pipe",
                # Water damage evidence
                "pipe water damage", "wet ceiling", "water stain"
            ],

            'RIB_1.3_LANDSLIDE': [
                # Terrain features
                "hillside slope", "steep slope", "unstable slope",
                "retaining wall", "embankment", "erosion",
                # Risk indicators
                "slope failure", "hill terrain", "elevated land",
                "slope crack", "unstable hillside", "eroding slope",
                # Mitigation structures
                "slope reinforcement", "retaining structure", "slope barrier"
            ],
            
            'RIB_1.4_DRAINAGE': [
                # Drain conditions
                "blocked drain", "clogged drain", "cracked drain",
                "debris in drain", "drain damage", "drainage channel",
                # Visible problems
                "broken drain", "damaged drain", "vegetation in drain",
                "dirty drain", "drain blockage", "overflow drain",
                # System issues
                "drainage grate", "drain cover", "storm drain"
            ],
            
            'RIB_1.5_FLOOD': [
                # Water presence
                "flood water", "standing water", "water damage",
                "water pooling", "wet floor", "water mark",
                # Mitigation
                "flood barrier", "sandbags", "water barrier",
                # Damage evidence
                "flood damage", "water stain", "wet wall",
                "water intrusion", "flooded area", "water ingress"
            ],
            
            'RIB_1.6_LIGHTNING': [
                # Lightning protection equipment (page15_img1.png shows this)
                "lightning rod", "lightning arrester", "strike counter",
                "air terminal", "lightning protection", "surge arrester",
                # System components
                "down conductor", "grounding rod", "lightning system",
                "earth termination", "lightning conductor", "protection rod",
                # Installation
                "roof lightning rod", "building lightning", "tall lightning rod"
            ],
            
            'RIB_1.7_FALLING_TREES': [
                # Tree condition
                "dead tree", "diseased tree", "decaying tree",
                "leaning tree", "tree fungus", "rotting tree",
                # Structural issues
                "hollow tree", "cracked tree", "unstable tree",
                "dry tree", "damaged tree", "weak tree",
                # Risk indicators
                "fallen branch", "tree near building", "hazardous tree"
            ],
            
            'RIB_1.8_WINDSTORM': [
                # Roof damage
                "roof damage", "ceiling damage", "damaged roof",
                "wind damage", "stripped ceiling", "roof strip",
                # Structural
                "gable bracing", "damaged ceiling", "storm damage",
                "blown roof", "roof uplift", "ceiling collapse",
                # Evidence
                "missing roof", "torn roofing", "roof debris"
            ],
            
            'RIB_1.9_IMPACT_DAMAGE': [
                # Visible damage
                "wall damage", "impact damage", "collision damage",
                "damaged wall", "structural damage", "wall crack",
                # Protection
                "bollard", "barrier", "impact barrier",
                # Causes
                "forklift damage", "vehicle impact", "crane damage",
                "loading damage", "impact mark", "dented wall"
            ],
            
            'RIB_1.10_GAS_STATION_IMPACT': [
                # Facility identification
                "gas station", "metering station", "gas facility",
                "gas meter", "station yard", "gas equipment",
                # Protection
                "plastic barrier", "chain fence", "inadequate barrier",
                # Risk areas
                "parking area", "gas pipeline", "metering equipment",
                "station barrier", "gas infrastructure", "meter yard"
            ],
            
            # ============================================================
            # 2.0 ELECTRICAL - 15 terms each
            # ============================================================
            'RIB_2.1_TRANSFORMER_INSPECTION': [
                # Transformer identification
                "transformer", "power transformer", "electrical transformer",
                "transformer unit", "transformer box", "transformer cabinet",
                # Condition indicators
                "unmaintained transformer", "dirty transformer", "old transformer",
                "rusted transformer", "weathered transformer", "outdoor transformer",
                # Inspection evidence
                "transformer no tag", "uninspected transformer", "transformer oxidation"
            ],
            
            'RIB_2.2_ELECTRICAL_INSPECTION': [
                # Panel types
                "electrical panel", "distribution board", "switchboard",
                "electrical meter", "panel box", "electrical board",
                # Inspection indicators
                "power panel", "breaker panel", "meter box",
                "panel door", "electrical cabinet", "switch panel",
                # Condition
                "old panel", "weathered panel", "outdoor panel"
            ],
            
            'RIB_2.3_COMBUSTIBLE_AT_BOARD': [
                # Cardboard boxes (page30_img1.png, page30_img2.png show this)
                "boxes near panel", "cardboard near board", "storage at panel",
                "boxes at switchboard", "cardboard boxes", "stacked boxes",
                # Clutter
                "clutter at panel", "items near electrical", "storage near board",
                "combustible at board", "boxes stacked", "materials at panel",
                # Risk scenario
                "boxes at electrical", "cardboard storage", "items near board"
            ],
            
            'RIB_2.4_PANEL_NOT_CLOSED': [
                # Open panel (page30_img1.png, page30_img2.png, page32_img1.png show open panels)
                "open panel", "open switchboard", "exposed panel",
                "panel door open", "unclosed panel", "panel without cover",
                # Missing components
                "open electrical box", "panel interior visible", "doorless panel",
                "missing panel door", "panel cover removed", "exposed electrical",
                # Interior visible
                "open distribution board", "visible panel interior", "panel ajar"
            ],
            
            'RIB_2.5_EXPOSED_WIRING': [
                # Wire conditions (page32_img1.png shows exposed wiring)
                "exposed wire", "bare wire", "loose wire",
                "unsheathed cable", "dangling wire", "wire without conduit",
                # Risk indicators
                "visible conductor", "exposed cable", "unprotected wire",
                "hanging wire", "open wiring", "wire no cover",
                # Installation issues
                "improper wiring", "unsafe wire", "exposed connection"
            ],
            
            'RIB_2.6_BURN_MARK': [
                # Burn evidence (page36_img1.png, page38_img1.png show burn marks)
                "burn mark", "scorch mark", "charred surface",
                "burnt panel", "thermal damage", "fire mark",
                # Electrical damage
                "scorched breaker", "burn damage", "charring",
                "burnt wire", "heat damage", "electrical burn",
                # Discoloration
                "black mark", "smoke damage", "burnt component"
            ],
            
            'RIB_2.7_KILL_SWITCH': [
                # Emergency controls
                "emergency switch", "kill switch", "master switch",
                "emergency cutoff", "main breaker", "power disconnect",
                # Safety equipment
                "emergency stop", "main switch", "shutdown button",
                "safety switch", "emergency disconnect", "power cutoff",
                # Installation
                "red emergency button", "kill button", "emergency control"
            ],
            
            # ============================================================
            # 3.0 HOUSEKEEPING - 15 terms each
            # ============================================================
            'RIB_3.1_GREASE_GRIME': [
                # Floor conditions
                "greasy floor", "grimy floor", "oily floor",
                "dirty floor", "grease buildup", "grime layer",
                # Surface types
                "oil stain", "greasy surface", "slippery floor",
                "stained floor", "dirty surface", "oily ground",
                # Industrial context
                "mill floor grease", "factory floor dirt", "workshop grime"
            ],
            
            'RIB_3.2_UNUSED_METAL': [
                # Metal accumulation
                "metal scrap", "scrap pile", "discarded metal",
                "metal debris", "scrap metal", "metal waste",
                # Storage
                "unused metal", "metal parts", "scrap yard",
                "metal storage", "steel scrap", "iron waste",
                # Types
                "metal clutter", "scrap material", "abandoned metal"
            ],
            
            'RIB_3.3_LPG_STORAGE': [
                # LPG containers (page44_img1.png shows LPG cylinders clearly)
                "lpg cylinder", "gas cylinder", "propane tank",
                "gas bottle", "lpg tank", "cylinder storage",
                # Storage setup
                "gas storage", "lpg bottle", "cylinder rack",
                "lpg cage", "gas cylinder cage", "cylinder area",
                # Context
                "blue cylinder", "orange cylinder", "cooking gas"
            ],
            
            'RIB_3.4_DRUM_WITHOUT_CONTAINMENT': [
                # Drums without trays (page46_img1.png shows drums/barrels)
                "drum no tray", "uncontained drum", "drum no pallet",
                "storage drum", "metal drum", "oil drum",
                # Risk scenario
                "chemical drum", "drum without spill tray", "barrel",
                "blue drum", "drum on floor", "drum no containment",
                # Types
                "200 liter drum", "steel drum", "industrial drum"
            ],
            
            'RIB_3.5_CONTAINMENT': [
                # Containment equipment
                "spill tray", "containment pallet", "drip tray",
                "bund wall", "catch pit", "containment bund",
                # Systems
                "spill containment", "secondary containment", "collection tray",
                "drum pallet", "spill pallet", "containment system",
                # Structures
                "dike wall", "bund structure", "spill barrier"
            ],
            
            'RIB_3.6_COMBUSTIBLE_STORAGE': [
                # High-piled storage (page50_img1.png, page50_img2.png, page55_img1.png show mattresses/cardboard)
                "mattress storage", "foam storage", "high pile storage",
                "combustible warehouse", "packaging material", "mattress pile",
                # Materials
                "flammable goods", "foam material", "high stock",
                "cardboard pile", "mattress stack", "foam stack",
                # Context
                "warehouse storage", "stacked mattresses", "piled combustibles"
            ],
            
            'RIB_3.7_GENERATOR_DIKE': [
                # Generator identification
                "generator", "generator set", "standby generator",
                "diesel generator", "generator unit", "backup generator",
                # Bund/dike
                "generator without bund", "generator no dike", "power generator",
                "generator fuel tank", "genset", "emergency generator",
                # Installation
                "outdoor generator", "generator without wall", "generator base"
            ],
            
            'RIB_3.8_COMBUSTIBLE_NEAR_PANEL': [
                # Items near controls (page55_img1.png shows clutter)
                "items near controls", "storage near panel", "boxes near control",
                "combustible near dcs", "clutter near controls", "materials at panel",
                # Control rooms
                "storage at controls", "items near equipment", "boxes at control",
                "clutter control room", "storage control panel", "items near instrumentation",
                # Process control
                "combustible near scada", "materials near controls", "clutter near dcs"
            ],
            
            # ============================================================
            # 4.0 HUMAN ELEMENT - 15 terms each
            # ============================================================
            'RIB_4.1_FIRE_EVACUATION': [
                # Evacuation planning
                "evacuation plan", "emergency route", "exit route",
                "assembly point", "evacuation map", "escape plan",
                # Signage
                "emergency exit", "evacuation sign", "exit sign",
                "fire escape", "evacuation route", "assembly area",
                # Preparedness
                "fire drill", "evacuation procedure", "emergency plan"
            ],
            
            'RIB_4.2_SMOKING_AREA': [
                # Designated areas (page60_img1.png shows context)
                "smoking area", "smoking zone", "designated smoking",
                "smoking spot", "smoking section", "cigarette area",
                # Facilities
                "smoking facility", "outdoor smoking", "smoking point",
                "smoking shelter", "ashtray", "smoking signage",
                # Context
                "smoking bay", "designated spot", "smoking corner"
            ],
            
            'RIB_4.3_HOT_WORK': [
                # Activities
                "welding", "cutting", "grinding",
                "hot work", "welding torch", "sparks",
                # Operations
                "flame cutting", "welding operation", "grinding work",
                "arc welding", "torch cutting", "metal cutting",
                # Equipment
                "welding machine", "angle grinder", "cutting torch"
            ],
            
            'RIB_4.4_GAS_TEST': [
                # Testing equipment
                "gas detector", "gas monitor", "lel meter",
                "gas testing", "gas sensor", "flammable gas detector",
                # Measurement
                "combustible gas meter", "gas measurement", "atmospheric monitor",
                "oxygen meter", "multi gas detector", "portable gas detector",
                # Usage
                "gas test device", "lel testing", "gas monitor handheld"
            ],
            
            'RIB_4.5_COOKING_FACILITIES': [
                # Kitchen areas (page67_img1.png shows residential LPG context)
                "cooking area", "kitchen", "cooking facility",
                "dormitory kitchen", "residential cooking", "cooking equipment",
                # Appliances
                "kitchen area", "cooking space", "food preparation",
                "gas stove", "kitchen stove", "cooking appliance",
                # Context
                "residential kitchen", "worker kitchen", "canteen"
            ],
            
            'RIB_4.6_CONSTRUCTION_AREAS': [
                # Construction activity
                "construction zone", "construction site", "construction area",
                "construction work", "renovation area", "construction barrier",
                # Work in progress
                "building work", "construction activity", "work site",
                "scaffolding", "construction fence", "work zone",
                # Safety
                "construction safety", "work area", "building site"
            ],
            
            'RIB_4.7_SMOKING_POLICY': [
                # Policy violations
                "smoking violation", "unauthorized smoking", "no smoking breach",
                "smoking breach", "smoking in prohibited area", "illegal smoking",
                # Non-compliance
                "smoking not allowed", "smoking infraction", "smoking rule breach",
                "cigarette butt", "smoking evidence", "prohibited smoking",
                # Enforcement
                "smoking ban", "no smoking area", "smoking restriction"
            ],
            
            'RIB_4.8_SAFETY_COMMITTEE': [
                # Committee structure
                "safety committee", "safety board", "safety team",
                "health committee", "workplace safety", "safety organization",
                # Operations
                "safety structure", "committee meeting", "safety group",
                "safety meeting", "committee board", "safety representatives",
                # Documentation
                "safety minutes", "committee record", "safety program"
            ],
            
            'RIB_4.9_PATROL_CLOCKING': [
                # Patrol systems
                "patrol checkpoint", "clocking station", "security point",
                "guard checkpoint", "patrol point", "clocking system",
                # Equipment
                "security checkpoint", "patrol station", "guard point",
                "clocking device", "patrol tag", "checkpoint scanner",
                # Implementation
                "security patrol", "guard tour", "patrol system"
            ],
            
            # ============================================================
            # 5.0 PROCESS - 15 terms each
            # ============================================================
            'RIB_5.1_SPRAY_COATING': [
                # Spray booth (page78_img1.png shows industrial area)
                "spray booth", "coating booth", "paint booth",
                "spray chamber", "coating facility", "spray painting",
                # Equipment
                "spray area", "painting booth", "coating area",
                "spray equipment", "paint chamber", "coating system",
                # Safety
                "explosion proof booth", "spray ventilation", "booth fan"
            ],
            
            'RIB_5.2_BATTERY_CHARGING': [
                # Storage and charging (page80_img1.png shows drums + meters)
                "metal drums", "storage drums", "drum storage",
                "electrical meter", "wall meter", "meter box",
                # Risk scenario
                "socket near drums", "charging near storage", "outlet warehouse",
                # Industrial context
                "barrel storage", "warehouse socket", "industrial charging",
                # Equipment
                "charging equipment", "battery charger", "wall outlet"
            ],
            
            'RIB_5.3_OVERHEAD_CRANE': [
                # Crane types (page82_img1.png shows overhead crane clearly)
                "overhead crane", "bridge crane", "gantry crane",
                "crane system", "hoist crane", "crane track",
                # Equipment
                "lifting crane", "crane bridge", "industrial crane",
                "crane hoist", "crane beam", "electric crane",
                # Installation
                "yellow crane", "factory crane", "warehouse crane"
            ],
            
            'RIB_5.4_PIPE_LABELLING': [
                # Unlabeled pipes
                "unlabeled pipe", "unmarked pipe", "pipe no label",
                "unidentified pipe", "pipe no marking", "pipe no tag",
                # Missing identification
                "unlabelled pipe", "pipe without label", "pipe system",
                "pipe no color code", "untagged pipe", "pipe no identification",
                # Process pipes
                "industrial pipe", "process pipe", "overhead pipe"
            ],
            
            'RIB_5.5_BATTERY_CHARGING_VENTILATION': [
                # Battery rooms
                "battery room", "battery area", "charging room",
                "battery space", "battery storage", "battery facility",
                # Ventilation issues
                "charging area", "battery chamber", "battery enclosure",
                "battery no ventilation", "unventilated battery", "battery room closed",
                # Context
                "ups room", "battery bank", "battery cabinet"
            ],
            
            'RIB_5.6_SPRAY_PAINTING_AREA': [
                # Indoor painting (page87_img1.png, page87_img2.png show cluttered areas)
                "indoor painting", "showroom painting", "painting indoors",
                "spray painting", "indoor spray", "painting work",
                # Inappropriate locations
                "spray work", "painting activity", "paint spraying",
                "showroom spray", "retail painting", "indoor paint",
                # Clutter
                "paint materials", "painting supplies", "spray equipment"
            ],
            
            'RIB_5.7_TANK_THICKNESS': [
                # Storage tanks
                "storage tank", "bulk tank", "tank shell",
                "corroded tank", "tank corrosion", "damaged tank",
                # Testing needs
                "tank bottom", "tank wall", "tank surface",
                "rusted tank", "tank deterioration", "tank integrity",
                # Industrial
                "industrial tank", "chemical tank", "fuel tank"
            ],
            
            'RIB_5.8_TEMPERATURE_MONITORING': [
                # Silo storage (from JSON: palm kernel storage)
                "silo", "storage silo", "grain silo",
                "bulk storage", "silo storage", "storage pile",
                # Monitoring equipment
                "silo structure", "storage facility", "silo temperature",
                "temperature sensor", "silo probe", "heat monitoring",
                # Materials
                "palm kernel silo", "grain storage", "combustible storage"
            ],
            
            'RIB_5.9_TANK_MANAGEMENT': [
                # Tank inspection
                "storage tank", "tank inspection", "bulk tank",
                "tank system", "tank facility", "tank structure",
                # Testing
                "industrial tank", "tank integrity", "tank maintenance",
                "ultrasonic testing", "tank testing", "ndt testing",
                # Program
                "tank program", "inspection program", "tank monitoring"
            ],
            
            'RIB_5.10_WEIR_INTEGRITY': [
                # Weir structures (from JSON: hydroelectric weir)
                "weir", "dam weir", "spillway",
                "weir structure", "hydroelectric weir", "weir dam",
                # Testing
                "water weir", "weir system", "weir facility",
                "weir integrity", "weir inspection", "weir testing",
                # Components
                "weir piezometer", "weir monitoring", "concrete weir"
            ],
            
            'RIB_5.11_DUST_EXTRACTION': [
                # Extraction equipment
                "dust extractor", "dust hood", "extraction system",
                "dust vent", "dust collector", "extraction hood",
                # Ductwork
                "dust duct", "ventilation duct", "dust filter",
                "extraction fan", "dust exhaust", "dust collection",
                # Installation
                "hood system", "extraction arm", "dust control"
            ],
            
            'RIB_5.12_SURGE_PROTECTION_PV': [
                # Solar installations
                "solar panel", "pv panel", "photovoltaic",
                "solar array", "pv system", "solar inverter",
                # Equipment
                "solar installation", "pv array", "solar farm",
                "solar roof", "pv installation", "solar power",
                # Protection
                "surge protector", "spd device", "lightning arrester"
            ],
            
            'RIB_5.13_GAS_PIPELINE': [
                # Pipeline types (from JSON: gas supply pipeline)
                "gas pipeline", "gas pipe", "pipeline",
                "gas line", "gas main", "overhead pipeline",
                # Installation
                "exposed pipeline", "gas distribution", "gas supply",
                "yellow pipe", "pipeline overhead", "gas network",
                # Components
                "pipe support", "pipeline valve", "gas piping"
            ],
            
            'RIB_5.14_PARTIAL_DISCHARGE': [
                # UPS equipment (from JSON: uninterruptible power supply)
                "ups", "uninterruptible power", "ups system",
                "ups unit", "backup power", "power supply",
                # Installation
                "ups equipment", "ups cabinet", "ups installation",
                "ups room", "ups battery", "ups rack",
                # Testing
                "ups panel", "ups transformer", "ups aging"
            ],
            
            'RIB_5.15_UNDERWATER_CORROSION': [
                # Bridge structures (from JSON: underwater bridge pilings)
                "bridge piling", "underwater piling", "marine piling",
                "underwater structure", "bridge support", "water piling",
                # Corrosion
                "submerged piling", "bridge foundation", "underwater support",
                "corroded piling", "piling rust", "marine corrosion",
                # Coastal
                "coastal bridge", "pier piling", "saltwater piling"
            ],
            
            'RIB_5.16_STATIC_ELECTRICITY': [
                # Static control equipment (from JSON: plastic film winding)
                "ionizing bar", "static eliminator", "ionizer",
                "anti-static bar", "static control", "ionizing blower",
                # Installation
                "static bar", "electrostatic control", "static device",
                "ionizing nozzle", "static neutralizer", "ionization system",
                # Process
                "plastic film", "winding machine", "unwinding roller"
            ]
        }
        
        # Comprehensive RIB section mapping
        self.rib_section_mapping = {
            'RIB_1.1_SUBSIDENCE': '1.1 - Subsidence',
            'RIB_1.2_PIPE_BURSTING': '1.2 - Pipe Bursting',
            'RIB_1.3_LANDSLIDE': '1.3 - Landslide and Risk Assessment',
            'RIB_1.4_DRAINAGE': '1.4 - Drainage System',
            'RIB_1.5_FLOOD': '1.5 - Flood Mitigation Measures',
            'RIB_1.6_LIGHTNING': '1.6 - Lightning Strike',
            'RIB_1.7_FALLING_TREES': '1.7 - Falling Trees',
            'RIB_1.8_WINDSTORM': '1.8 - Windstorm',
            'RIB_1.9_IMPACT_DAMAGE': '1.9 - Impact Damage from Lifting Activities',
            'RIB_1.10_GAS_STATION_IMPACT': '1.10 - Enhancing Impact Protection for Natural Gas Reducing and Metering Station',
            
            'RIB_2.1_TRANSFORMER_INSPECTION': '2.1 - No Inspection on Transformer',
            'RIB_2.2_ELECTRICAL_INSPECTION': '2.2 - Electrical Inspection',
            'RIB_2.3_COMBUSTIBLE_AT_BOARD': '2.3 - Accumulation of Combustible Material at Electrical Board',
            'RIB_2.4_PANEL_NOT_CLOSED': '2.4 - Electrical Board Is Not Properly Closed',
            'RIB_2.5_EXPOSED_WIRING': '2.5 - Exposed Wiring',
            'RIB_2.6_BURN_MARK': '2.6 - Burn Mark At The Circuit Breaker',
            'RIB_2.7_KILL_SWITCH': '2.7 - Kill Switch',
            
            'RIB_3.1_GREASE_GRIME': '3.1 - Floor Area Covered with Grease and Grime',
            'RIB_3.2_UNUSED_METAL': '3.2 - Storage of Unused Metal',
            'RIB_3.3_LPG_STORAGE': '3.3 - Storage of LPG Gas Cylinder',
            'RIB_3.4_DRUM_WITHOUT_CONTAINMENT': '3.4 - Storage of Drums Without Containment',
            'RIB_3.5_CONTAINMENT': '3.5 - Containment / Catch Pit / Outlet Discharge Containment',
            'RIB_3.6_COMBUSTIBLE_STORAGE': '3.6 - Storage Of Combustible Material',
            'RIB_3.7_GENERATOR_DIKE': '3.7 - Installation of Dike Around Generator Set',
            'RIB_3.8_COMBUSTIBLE_NEAR_PANEL': '3.8 - Combustible Storage Near Control Panels',
            
            'RIB_4.1_FIRE_EVACUATION': '4.1 - Fire Evacuation',
            'RIB_4.2_SMOKING_AREA': '4.2 - Designated Smoking Area',
            'RIB_4.3_HOT_WORK': '4.3 - Hot Work',
            'RIB_4.4_GAS_TEST': '4.4 - Gas Test Before Hot Work Operation',
            'RIB_4.5_COOKING_FACILITIES': '4.5 - Cooking Facilities',
            'RIB_4.6_CONSTRUCTION_AREAS': '4.6 - Managing Risks in Construction Areas',
            'RIB_4.7_SMOKING_POLICY': '4.7 - Smoking Policy Not Enforced',
            'RIB_4.8_SAFETY_COMMITTEE': '4.8 - Safety Committee',
            'RIB_4.9_PATROL_CLOCKING': '4.9 - Proposal of Patrol Clocking',
            
            'RIB_5.1_SPRAY_COATING': '5.1 - Spray Coating',
            'RIB_5.2_BATTERY_CHARGING': '5.2 - Battery Charging Socket',
            'RIB_5.3_OVERHEAD_CRANE': '5.3 - Overhead Crane',
            'RIB_5.4_PIPE_LABELLING': '5.4 - Labelling of Pipes and Tubes',
            'RIB_5.5_BATTERY_CHARGING_VENTILATION': '5.5 - Battery Charging in Non-Ventilated Room',
            'RIB_5.6_SPRAY_PAINTING_AREA': '5.6 - Spray Painting Within Showroom Space',
            'RIB_5.7_TANK_THICKNESS': '5.7 - Thickness Testing of Bulk Tanks',
            'RIB_5.8_TEMPERATURE_MONITORING': '5.8 - Temperature Monitoring for Palm Kernel Storage',
            'RIB_5.9_TANK_MANAGEMENT': '5.9 - Tank Management Program',
            'RIB_5.10_WEIR_INTEGRITY': '5.10 - Weir Integrity Testing',
            'RIB_5.11_DUST_EXTRACTION': '5.11 - Dust Extraction System',
            'RIB_5.12_SURGE_PROTECTION_PV': '5.12 - Surge Protection Device at PV Solar Farm',
            'RIB_5.13_GAS_PIPELINE': '5.13 - Pipeline for Gas Supply for Commercial Building/Apartment',
            'RIB_5.14_PARTIAL_DISCHARGE': '5.14 - Partial Discharge (PD) for Uninterruptible Power Supply (UPS)',
            'RIB_5.15_UNDERWATER_CORROSION': '5.15 - Corrosion of Underwater Bridge Pilings',
            'RIB_5.16_STATIC_ELECTRICITY': '5.16 - Static Electricity During the Unwinding and Winding of Plastic Sheets'
        }
        
        # IMPROVED Object to broad category mapping
        self.object_to_broad_category = {
            # Perils
            'crack': 'PERILS', 'ground crack': 'PERILS', 'wall crack': 'PERILS', 'floor crack': 'PERILS',
            'subsidence': 'PERILS',
            'pipe': 'PERILS', 'burst pipe': 'PERILS', 'pipe leak': 'PERILS', 'water leak': 'PERILS',
            'drain': 'PERILS', 'drainage': 'PERILS', 'blocked drain': 'PERILS',
            'flood water': 'PERILS', 'standing water': 'PERILS', 'water damage': 'PERILS', 'water puddle': 'PERILS',
            'lightning arrester': 'PERILS', 'lightning rod': 'PERILS',
            'tree': 'PERILS', 'dead tree': 'PERILS', 'fallen tree': 'PERILS',
            'slope': 'PERILS', 'hillside': 'PERILS', 'retaining wall': 'PERILS',
            
            # Electrical
            'electrical panel': 'ELECTRICAL', 'electrical board': 'ELECTRICAL', 'switchboard': 'ELECTRICAL',
            'open panel': 'ELECTRICAL',
            'transformer': 'ELECTRICAL', 'circuit breaker': 'ELECTRICAL',
            'exposed wire': 'ELECTRICAL', 'loose wire': 'ELECTRICAL',
            'burn mark': 'ELECTRICAL', 'scorch mark': 'ELECTRICAL',
            
            # Housekeeping
            'grease': 'HOUSEKEEPING', 'grease floor': 'HOUSEKEEPING', 'dirty floor': 'HOUSEKEEPING', 'oily surface': 'HOUSEKEEPING',
            'metal scrap': 'HOUSEKEEPING', 'scrap pile': 'HOUSEKEEPING',
            'lpg cylinder': 'HOUSEKEEPING', 'gas cylinder': 'HOUSEKEEPING',
            'storage drum': 'HOUSEKEEPING', 'metal drum': 'HOUSEKEEPING', 'oil drum': 'HOUSEKEEPING',
            'combustible material': 'HOUSEKEEPING', 'cardboard box': 'HOUSEKEEPING', 'boxes': 'HOUSEKEEPING',
            'generator': 'HOUSEKEEPING', 'dike wall': 'HOUSEKEEPING', 'bund': 'HOUSEKEEPING',
            
            # Human Element
            'fire extinguisher': 'HUMAN_ELEMENT', 'exit sign': 'HUMAN_ELEMENT', 'emergency exit': 'HUMAN_ELEMENT',
            'smoking area': 'HUMAN_ELEMENT', 'cigarette': 'HUMAN_ELEMENT', 'no smoking sign': 'HUMAN_ELEMENT',
            'safety sign': 'HUMAN_ELEMENT', 'warning sign': 'HUMAN_ELEMENT',
            
            # Process
            'spray booth': 'PROCESS', 'battery': 'PROCESS', 'battery rack': 'PROCESS',
            'overhead crane': 'PROCESS', 'crane': 'PROCESS',
            'pipe label': 'PROCESS', 'storage tank': 'PROCESS',
            'solar panel': 'PROCESS', 'pv panel': 'PROCESS'
        }
        
        # Define critical safety subsections for lower threshold
        self.critical_subsections = [
            'RIB_2.4', 'RIB_2.5', 'RIB_2.6',  # Electrical safety
            'RIB_2.3',  # Combustible at board
            'RIB_3.4',  # Drums without containment
            'RIB_4.3',  # Hot work
        ]
    
    def detect_tier1(self, image_path: str, threshold: float = 0.1) -> Dict[str, Any]:
        """Tier 1: General detection"""
        print(f"\n{'='*80}")
        print("TIER 1: GENERAL RIB INDICATOR DETECTION")
        print(f"{'='*80}")
        
        image = Image.open(image_path).convert("RGB")
        
        # Preprocess image (resize if needed)
        image = self.preprocess_image(image)
        
        # Check for red circle and extract if found
        circle_info = None
        detection_image = image
        red_circle_detected, circle_info = self.detect_red_circle(image)
        if red_circle_detected is not None:
            detection_image = red_circle_detected
            circle_info['in_circle'] = True
        else:
            circle_info = {'detected': False, 'in_circle': False}
        
        texts = [self.tier1_objects]
        inputs = self.processor(text=texts, images=detection_image, return_tensors="pt")
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        target_sizes = torch.Tensor([detection_image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=threshold
        )
        
        boxes = results[0]["boxes"]
        scores = results[0]["scores"]
        labels = results[0]["labels"]
        
        detections = []
        identified_categories = set()
        
        for box, score, label in zip(boxes, scores, labels):
            detected_object = texts[0][label]
            broad_category = self.object_to_broad_category.get(detected_object, 'UNKNOWN')
            
            detection = {
                'object': detected_object,
                'confidence': float(score),
                'bbox': [float(x) for x in box.tolist()],
                'broad_category': broad_category
            }
            
            detections.append(detection)
            identified_categories.add(broad_category)
        
        print(f"\n✓ Detected {len(detections)} objects")
        print(f"✓ Identified RIB categories: {', '.join(identified_categories)}")
        
        for i, det in enumerate(detections[:10], 1):  # Show top 10
            print(f"  {i}. {det['object']} ({det['confidence']:.2%}) - {det['broad_category']}")
        
        return {
            'image': image,
            'detection_image': detection_image,
            'circle_info': circle_info,
            'detections': detections,
            'identified_categories': list(identified_categories),
            'tier': 1
        }
    
    def detect_tier2_rib_subsections(
        self,
        image: Image.Image,
        broad_categories: List[str],
        threshold: float = 0.10,
        max_subsections: int = 10
    ) -> Dict[str, Any]:
        """Tier 2: Detect specific RIB subsections with adaptive thresholds"""
        print(f"\n{'='*80}")
        print("TIER 2: RIB SUBSECTION DETECTION")
        print(f"{'='*80}")
        
        # Filter RIB subsections based on broad categories
        # IMPORTANT: Use separate if statements (not elif) to collect ALL matching categories
        relevant_subsections = []
        
        for rib_key in self.tier2_rib_vocabulary.keys():
            # Check each category independently
            if 'ELECTRICAL' in broad_categories and rib_key.startswith('RIB_2'):
                relevant_subsections.append(rib_key)
            if 'HOUSEKEEPING' in broad_categories and rib_key.startswith('RIB_3'):
                relevant_subsections.append(rib_key)
            if 'PERILS' in broad_categories and rib_key.startswith('RIB_1'):
                relevant_subsections.append(rib_key)
            if 'HUMAN_ELEMENT' in broad_categories and rib_key.startswith('RIB_4'):
                relevant_subsections.append(rib_key)
            if 'PROCESS' in broad_categories and rib_key.startswith('RIB_5'):
                relevant_subsections.append(rib_key)
        
        # Limit number of subsections to process for speed
        # PRIORITIZE: Put critical electrical/safety subsections first
        critical_priority = []
        high_priority = []
        normal_priority = []
        
        for subsec in relevant_subsections:
            if subsec.startswith('RIB_2'):  # Electrical - CRITICAL
                critical_priority.append(subsec)
            elif subsec.startswith('RIB_4'):  # Human element - HIGH
                high_priority.append(subsec)
            elif subsec.startswith('RIB_3'):  # Housekeeping - HIGH
                high_priority.append(subsec)
            else:  # Others - NORMAL
                normal_priority.append(subsec)
        
        # Reorder: Critical first, then high priority, then normal
        relevant_subsections = critical_priority + high_priority + normal_priority
        
        print(f"  ℹ️ Found {len(relevant_subsections)} relevant subsections across {len(broad_categories)} categories")
        print(f"     Categories detected: {', '.join(broad_categories)}")
        print(f"     Breakdown: {len(critical_priority)} electrical, {len(high_priority)} housekeeping/human, {len(normal_priority)} perils/process")
        
        if len(relevant_subsections) > max_subsections:
            print(f"  ℹ️ Processing top {max_subsections} subsections (prioritizing electrical safety)")
            relevant_subsections = relevant_subsections[:max_subsections]
        
        subsection_detections = {}
        
        for rib_key in relevant_subsections:
            vocabulary = self.tier2_rib_vocabulary[rib_key]
            
            # Use lower threshold for critical safety items
            detection_threshold = threshold
            if any(crit in rib_key for crit in self.critical_subsections):
                detection_threshold = max(0.05, threshold * 0.5)  # 50% lower, minimum 5%
                print(f"  ℹ️ Using lower threshold {detection_threshold:.2%} for critical subsection {rib_key}")
            
            texts = [vocabulary]
            inputs = self.processor(text=texts, images=image, return_tensors="pt")
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            target_sizes = torch.Tensor([image.size[::-1]])
            results = self.processor.post_process_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=detection_threshold
            )
            
            boxes = results[0]["boxes"]
            scores = results[0]["scores"]
            labels = results[0]["labels"]
            
            detections = []
            for box, score, label in zip(boxes, scores, labels):
                detections.append({
                    'object': texts[0][label],
                    'confidence': float(score),
                    'bbox': [float(x) for x in box.tolist()]
                })
            
            if detections:
                subsection_detections[rib_key] = detections
                rib_section = self.rib_section_mapping[rib_key]
                print(f"\n  ✓ {rib_section}")
                print(f"    Found {len(detections)} indicators")
                for i, det in enumerate(detections[:3], 1):
                    print(f"      {i}. {det['object']} ({det['confidence']:.2%})")
        
        return {
            'detections': subsection_detections,
            'tier': 2
        }
    
    def determine_primary_rib_subsection(
        self,
        subsection_detections: Dict[str, List[Dict]]
    ) -> List[Dict[str, Any]]:
        """Determine primary RIB subsection based on detection scores with IMPROVED priority logic"""
        
        subsection_scores = {}
        
        for rib_key, detections in subsection_detections.items():
            if not detections:
                continue
            
            # Calculate weighted score
            total_confidence = sum(d['confidence'] for d in detections)
            count = len(detections)
            avg_confidence = total_confidence / count if count > 0 else 0
            
            # Weighted score: count × average confidence
            score = count * avg_confidence
            
            # IMPROVED Priority boosts for critical issues
            if 'RIB_2.4' in rib_key:  # Open panel - HIGHEST priority (safety critical)
                score *= 3.0
            elif 'RIB_2.3' in rib_key:  # Cardboard at electrical board - very high priority
                score *= 2.5
            elif 'RIB_2.6' in rib_key:  # Burn mark - very high priority
                score *= 2.5
            elif 'RIB_2.5' in rib_key:  # Exposed wiring - high priority
                score *= 2.2
            elif 'RIB_1.6' in rib_key:  # Lightning - high priority
                score *= 1.8
            elif 'RIB_3.4' in rib_key or 'RIB_3.5' in rib_key:  # Containment issues
                score *= 1.6
            elif 'RIB_4.3' in rib_key:  # Hot work
                score *= 1.5
            
            subsection_scores[rib_key] = {
                'score': score,
                'count': count,
                'avg_confidence': avg_confidence,
                'rib_section': self.rib_section_mapping.get(rib_key, 'Unknown'),
                'detections': detections
            }
        
        # Sort by score
        sorted_subsections = sorted(
            subsection_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )
        
        return sorted_subsections
    
    def detect_hierarchical(
        self,
        image_path: str,
        tier1_threshold: float = 0.1,
        tier2_threshold: float = 0.10,
        visualize: bool = True,
        save_viz_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Complete hierarchical RIB detection"""
        
        print(f"\n{'='*80}")
        print("RIB-ALIGNED HIERARCHICAL DETECTION (IMPROVED VERSION)")
        print(f"{'='*80}")
        print(f"Image: {image_path}")
        
        # Tier 1
        tier1_results = self.detect_tier1(image_path, tier1_threshold)
        
        # Tier 2
        subsection_analysis = None
        ranked_subsections = []
        
        if tier1_results['identified_categories']:
            tier2_results = self.detect_tier2_rib_subsections(
                tier1_results['detection_image'],
                tier1_results['identified_categories'],
                tier2_threshold
            )
            
            ranked_subsections = self.determine_primary_rib_subsection(
                tier2_results['detections']
            )
            
            if ranked_subsections:
                print(f"\n{'='*80}")
                print("RANKED RIB SUBSECTIONS")
                print(f"{'='*80}")
                for i, (key, data) in enumerate(ranked_subsections[:5], 1):
                    print(f"\n{i}. {data['rib_section']}")
                    print(f"   Score: {data['score']:.2f}")
                    print(f"   Detections: {data['count']}")
                    print(f"   Avg Confidence: {data['avg_confidence']:.2%}")
            
            subsection_analysis = {
                'detections': tier2_results['detections'],
                'ranked_subsections': ranked_subsections
            }
        
        final_results = {
            'image_path': image_path,
            'tier1': tier1_results,
            'rib_subsection_analysis': subsection_analysis,
            'summary': self._generate_rib_summary(tier1_results, subsection_analysis)
        }
        
        if visualize and subsection_analysis:
            self.visualize_rib_detections(
                tier1_results['detection_image'],
                subsection_analysis,
                save_path=save_viz_path
            )
        
        return final_results
    
    def _generate_rib_summary(
        self,
        tier1_results: Dict,
        subsection_analysis: Optional[Dict]
    ) -> Dict[str, Any]:
        """Generate RIB-aligned summary"""
        
        summary = {
            'total_tier1_detections': len(tier1_results['detections']),
            'identified_broad_categories': tier1_results['identified_categories']
        }
        
        if subsection_analysis and subsection_analysis['ranked_subsections']:
            top_subsection = subsection_analysis['ranked_subsections'][0]
            summary['primary_rib_subsection'] = top_subsection[1]['rib_section']
            summary['subsection_score'] = top_subsection[1]['score']
            summary['detection_count'] = top_subsection[1]['count']
            
            summary['all_applicable_subsections'] = [
                {
                    'section': data['rib_section'],
                    'score': data['score'],
                    'detections': data['count']
                }
                for key, data in subsection_analysis['ranked_subsections']
            ]
        
        return summary
    
    def visualize_rib_detections(
        self,
        image: Image.Image,
        subsection_analysis: Dict,
        save_path: Optional[str] = None
    ):
        """Visualize RIB subsection detections"""
        
        ranked = subsection_analysis['ranked_subsections']
        if not ranked:
            print("No subsection detections to visualize")
            return
        
        num_plots = min(3, len(ranked))
        fig, axes = plt.subplots(1, num_plots, figsize=(8*num_plots, 8))
        
        if num_plots == 1:
            axes = [axes]
        
        colors = plt.cm.Set1(range(10))
        
        for idx, (key, data) in enumerate(ranked[:num_plots]):
            ax = axes[idx]
            ax.imshow(image)
            ax.set_title(
                f"{data['rib_section']}\nScore: {data['score']:.2f} | Detections: {data['count']}",
                fontsize=11,
                weight='bold'
            )
            
            for i, det in enumerate(data['detections'][:10]):
                box = det['bbox']
                xmin, ymin, xmax, ymax = box
                width = xmax - xmin
                height = ymax - ymin
                
                rect = patches.Rectangle(
                    (xmin, ymin), width, height,
                    linewidth=2,
                    edgecolor=colors[i % 10],
                    facecolor='none'
                )
                ax.add_patch(rect)
                
                ax.text(
                    xmin, ymin - 5,
                    f"{det['object']}\n{det['confidence']:.1%}",
                    bbox=dict(boxstyle="round,pad=0.2", fc=colors[i % 10], ec="black", alpha=0.7),
                    fontsize=7,
                    weight='bold',
                    color='white'
                )
            
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n✓ Visualization saved to: {save_path}")
        
        plt.show()
    
    def print_summary(self, results: Dict[str, Any]):
        """Print RIB-aligned summary"""
        summary = results['summary']
        
        print(f"\n{'='*80}")
        print("RIB DETECTION SUMMARY")
        print(f"{'='*80}")
        
        print(f"\nTier 1 Detections: {summary['total_tier1_detections']}")
        print(f"Broad Categories: {', '.join(summary['identified_broad_categories'])}")
        
        if 'primary_rib_subsection' in summary:
            print(f"\n🎯 PRIMARY RIB SUBSECTION: {summary['primary_rib_subsection']}")
            print(f"   Confidence Score: {summary['subsection_score']:.2f}")
            print(f"   Detection Count: {summary['detection_count']}")
        
        if 'all_applicable_subsections' in summary and len(summary['all_applicable_subsections']) > 1:
            print(f"\n📌 OTHER APPLICABLE RIB SUBSECTIONS:")
            for subsec in summary['all_applicable_subsections'][1:5]:
                print(f"   • {subsec['section']}")
                print(f"     Score: {subsec['score']:.2f} | Detections: {subsec['detections']}")
        
        print(f"\n{'='*80}\n")
    
    def generate_text_report(self, results: Dict[str, Any]) -> str:
        """Generate detailed RIB-aligned text report"""
        lines = []
        lines.append("="*80)
        lines.append("RISK IMPROVEMENT BENCHMARK (RIB) DETECTION REPORT")
        lines.append("IMPROVED VERSION - Optimized Vocabularies")
        lines.append("="*80)
        lines.append(f"Image: {results['image_path']}")
        lines.append("")
        
        summary = results['summary']
        
        lines.append("EXECUTIVE SUMMARY")
        lines.append("-"*80)
        lines.append(f"Total Detections: {summary['total_tier1_detections']}")
        lines.append(f"Broad Categories: {', '.join(summary['identified_broad_categories'])}")
        
        if 'primary_rib_subsection' in summary:
            lines.append(f"\nPrimary RIB Subsection: {summary['primary_rib_subsection']}")
            lines.append(f"Confidence Score: {summary['subsection_score']:.2f}")
            lines.append(f"Detection Count: {summary['detection_count']}")
        
        if 'all_applicable_subsections' in summary:
            lines.append("")
            lines.append("ALL APPLICABLE RIB SUBSECTIONS (Ranked by Relevance)")
            lines.append("-"*80)
            for i, subsec in enumerate(summary['all_applicable_subsections'], 1):
                lines.append(f"\n{i}. {subsec['section']}")
                lines.append(f"   Score: {subsec['score']:.2f}")
                lines.append(f"   Detections: {subsec['detections']}")
        
        lines.append("")
        lines.append("="*80)
        lines.append("END OF REPORT")
        lines.append("="*80)
        
        return "\n".join(lines)


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description='RIB-Aligned Object Detection using OWLv2 (IMPROVED VERSION)'
    )
    parser.add_argument('--image', '-i', required=True, help='Path to image')
    parser.add_argument('--tier1-threshold', type=float, default=0.1)
    parser.add_argument('--tier2-threshold', type=float, default=0.10)
    parser.add_argument('--save-viz', help='Save visualization')
    parser.add_argument('--save-report', help='Save text report')
    parser.add_argument('--no-visualize', action='store_true')
    
    args = parser.parse_args()
    
    if not Path(args.image).exists():
        print(f"❌ Error: Image not found: {args.image}")
        return 1
    
    # Initialize detector
    detector = RIBAlignedDetector()
    
    # Run detection
    results = detector.detect_hierarchical(
        image_path=args.image,
        tier1_threshold=args.tier1_threshold,
        tier2_threshold=args.tier2_threshold,
        visualize=not args.no_visualize,
        save_viz_path=args.save_viz
    )
    
    # Print summary
    detector.print_summary(results)
    
    # Save report if requested
    if args.save_report:
        report = detector.generate_text_report(results)
        with open(args.save_report, 'w') as f:
            f.write(report)
        print(f"✓ Report saved to: {args.save_report}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())