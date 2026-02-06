"""
RIB-Aligned Two-Tier OWLv2 Object Detection System
Enhanced with UNIQUE, NON-OVERLAPPING subsection vocabularies
And red circle region-of-interest detection

Key improvements:
- Each subsection vocabulary is now explicitly unique to prevent cross-contamination
- Red circle detection: If image contains a red circle, detection is limited to that region
- Automatic cropping to circle boundaries for focused analysis

Installation:
    pip install transformers torch pillow matplotlib opencv-python

Usage:
    python rib_owlv2_detector_unique.py --image path/to/image.jpg
    
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
    RIB-aligned two-tier object detection system with UNIQUE subsection vocabularies
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
        
        # Note: torch.compile disabled due to GPU architecture compatibility issues
        # Uncomment below if you want to use torch.compile on compatible hardware
        # if hasattr(torch, 'compile') and self.device == "cuda":
        #     try:
        #         print("  Compiling model with torch.compile for faster inference...")
        #         self.model = torch.compile(self.model, mode="reduce-overhead")
        #         print("  ✓ Model compilation successful")
        #     except Exception as e:
        #         print(f"  ⚠️ Model compilation skipped: {e}")
        
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
        """Setup detection vocabulary with UNIQUE terms per subsection"""
        
        # TIER 1: General detection vocabulary (unchanged)
        self.tier1_objects = [
            # 1.0 PERILS - General indicators
            "crack", "ground crack", "wall crack", "subsidence",
            "burst pipe", "water leak",
            "slope", "landslide", "retaining wall",
            "drain", "drainage channel", "blocked drain", "water puddle",
            "flood water", "standing water",
            "lightning arrester", "lightning rod", "strike counter",
            "tree", "dead tree", "leaning tree", "fallen tree",
    
            # 2.0 ELECTRICAL - General indicators
            "electrical panel", "electrical board", "switchboard",
            "transformer", "circuit breaker",
            "exposed wire", "loose wire", "burn mark",
    
            # 3.0 HOUSEKEEPING - General indicators
            "grease", "grime", "dirty floor",
            "metal scrap", "unused metal",
            "lpg cylinder",
            "storage drum", "metal drum",
            "combustible material", "cardboard box",
            "generator", "dike", "bund wall",
    
            # 4.0 HUMAN ELEMENT - General indicators
            "fire extinguisher", "exit sign", "emergency exit",
            "smoking area", "cigarette", "no smoking sign",
            "safety sign", "warning sign",
    
            # 5.0 PROCESS - General indicators
            "spray booth", "battery", "battery rack",
            "overhead crane", "lifting equipment",
            "pipe label", "storage tank",
            "solar panel"
        ]
        
        # TIER 2: UNIQUE, NON-OVERLAPPING subsection vocabularies
        self.tier2_rib_vocabulary = {
            # ============================================================
            # 1.0 PERILS - 9 features each
            # ============================================================
            'RIB_1.1_SUBSIDENCE': [
                "ground subsidence crack", "foundation subsidence", "structural subsidence damage",
                "settlement crack in ground", "sinking foundation", "ground settlement crack",
                "subsided floor area", "uneven ground settlement", "building foundation crack"
            ],

            'RIB_1.2_PIPE_BURSTING': [
                "burst water pipe", "ruptured pipe leak", "broken water main",
                "pipe burst flood damage", "water pipe rupture", "pipe joint failure",
                "corroded burst pipe", "frequent pipe bursting", "leaking burst pipe"
            ],

            'RIB_1.3_LANDSLIDE': [
                "hillside landslide risk", "slope instability", "elevated land with hill",
                "retaining wall for slope", "slope failure area", "eroded embankment",
                "unstable slope terrain", "soil mass movement", "campus hill slope"
            ],
            
            'RIB_1.4_DRAINAGE': [
                "blocked storm drain", "clogged drainage channel", "cracked drain section",
                "drainage system blockage", "debris in drain", "vegetation blocking drain",
                "cracked drainage pipe", "damaged drain rear side", "non-functional drain"
            ],
            
            'RIB_1.5_FLOOD': [
                "flood water damage", "standing flood water", "recent flooding loss",
                "flood barrier installation", "sandbag flood protection", "equipment flood damage",
                "flood water mark", "inundated production area", "flood prone factory"
            ],
            
            'RIB_1.6_LIGHTNING': [
                "lightning strike counter", "lightning arrester installation", "lightning event counter",
                "air termination rod", "lightning down conductor", "lightning grounding system",
                "lightning protection system", "frequent lightning activity", "lightning equipment damage"
            ],
            
            'RIB_1.7_FALLING_TREES': [
                "dead standing tree", "diseased tree trunk", "decaying tree",
                "leaning unstable tree", "tree with fungal growth", "rotting tree branch",
                "tree structural crack", "hollow tree trunk", "dangerous overhanging tree"
            ],
            
            'RIB_1.8_WINDSTORM': [
                "windstorm roof damage", "wind damaged ceiling", "storm damaged roof strip",
                "gable end bracing", "wind uplift damage", "ceiling strip damage",
                "multiple wind incidents", "windstorm mitigation", "stripped ceiling wind"
            ],
            
            'RIB_1.9_IMPACT_DAMAGE': [
                "vehicle impact damage wall", "lorry collision damage roof", "lifting impact damage",
                "loading area collision", "structural impact damage", "bollard protection missing",
                "impact barrier inadequate", "forklift impact damage", "factory wall impact"
            ],
            
            'RIB_1.10_GAS_STATION_IMPACT': [
                "gas metering station yard", "gas reducing station barrier", "plastic impact barrier inadequate",
                "gas station vehicle protection", "chain link fence station", "metering station parking area",
                "car parking near gas station", "plastic barrier insufficient", "natural gas leak risk"
            ],
            
            # ============================================================
            # 2.0 ELECTRICAL - 9 features each
            # ============================================================
            'RIB_2.1_TRANSFORMER_INSPECTION': [
                "transformer no inspection", "unmaintained transformer", "transformer without inspection tag",
                "transformer maintenance absent", "transformer testing never done", "transformer service missing",
                "dirty unmaintained transformer", "transformer lacking inspection record", "no transformer inspection sticker"
            ],
            
            'RIB_2.2_ELECTRICAL_INSPECTION': [
                "electrical panel no inspection", "distribution board not tested", "electrical meter uninspected",
                "switchboard maintenance absent", "electrical inspection missing", "panel testing not done",
                "electrical system no inspection", "board inspection overdue", "no electrical testing record"
            ],
            
            'RIB_2.3_COMBUSTIBLE_AT_BOARD': [
                "cardboard boxes at electrical board", "stacked boxes near panel",
                "corrugated cardboard near switchboard", "storage boxes piled at electrical",
                "multiple boxes at electrical panel", "paper boxes at distribution board",
                "stacked packaging near electrical", "brown boxes accumulated at board", "shipping cartons at panel"
            ],
            
            'RIB_2.4_PANEL_NOT_CLOSED': [
                "open electrical panel door", "electrical panel door removed",
                "unclosed electrical panel", "electrical board missing cover",
                "exposed panel interior", "electrical cabinet door open",
                "open distribution board", "panel cover not installed", "switchboard door wide open"
            ],
            
            'RIB_2.5_EXPOSED_WIRING': [
                "exposed live electrical wire", "bare electrical conductor", "wire without conduit",
                "unsheathed electrical cable", "exposed copper wire", "unprotected electrical wiring",
                "dangling exposed wire", "wire lacking insulation", "visible bare conductor"
            ],
            
            'RIB_2.6_BURN_MARK': [
                "burn mark on panel", "scorch mark at breaker", "charred electrical component",
                "burnt panel surface", "thermal damage on board", "electrical fire mark",
                "scorched circuit breaker", "burn damage electrical", "charred breaker"
            ],
            
            'RIB_2.7_KILL_SWITCH': [
                "emergency kill switch missing", "main power kill switch absent", "no master shutdown",
                "emergency cutoff switch needed", "main breaker kill switch", "emergency disconnect required",
                "facility kill switch absent", "power emergency stop missing", "main electrical kill button needed"
            ],
            
            # ============================================================
            # 3.0 HOUSEKEEPING - 9 features each
            # ============================================================
            'RIB_3.1_GREASE_GRIME': [
                "grease covered mill floor", "grimy floor surface", "greasy industrial floor machinery",
                "oil stained floor", "grease accumulation floor", "slippery greasy surface",
                "grime buildup machinery", "oily dirty floor", "grease grime layer mill"
            ],
            
            'RIB_3.2_UNUSED_METAL': [
                "unused metal scrap pile", "discarded metal storage", "scrap metal accumulation outdoor",
                "metal debris pile", "obsolete metal parts", "scrap metal storage yard",
                "abandoned metal materials", "unused metal components", "scrap metal clutter outside"
            ],
            
            'RIB_3.3_LPG_STORAGE': [
                "lpg cylinder storage residential", "lpg gas bottle near kitchen", "propane cylinder storage",
                "lpg cylinder cage", "gas cylinder storage area", "lpg bottle rack",
                "lpg storage residential facility", "gas cylinder near cooking", "lpg tank storage dormitory"
            ],
            
            'RIB_3.4_DRUM_WITHOUT_CONTAINMENT': [
                "drum without spill tray chiller", "uncontained storage drum", "drum lacking containment",
                "chemical drum without tray", "oil drum without pallet", "metal drum no containment",
                "storage drum no secondary containment", "unknown liquid drum uncontained", "liquid drum no spill tray"
            ],
            
            'RIB_3.5_CONTAINMENT': [
                "spill containment tray", "secondary containment pallet", "drip containment tray drum",
                "containment bund structure", "catch pit for containment", "spill prevention tray",
                "liquid containment system", "drum containment pallet", "spill collection tray installed"
            ],
            
            'RIB_3.6_COMBUSTIBLE_STORAGE': [
                "combustible material warehouse", "mattress storage high pile", "foam material storage area",
                "packaging material warehouse", "combustible inventory high stock", "mattress stockpile storage",
                "flammable goods warehouse", "foam combustible storage", "mattress foam high pile"
            ],
            
            'RIB_3.7_GENERATOR_DIKE': [
                "generator without dike wall", "generator no bund installation", "generator lacking fuel containment",
                "standby generator no bund", "generator set no dike", "generator no containment wall",
                "generator fuel no bund", "diesel generator no containment", "generator no spill containment"
            ],
            
            'RIB_3.8_COMBUSTIBLE_NEAR_PANEL': [
                "combustible near control panel", "flammable items control room",
                "storage items near process control", "combustible goods near dcs",
                "materials stored at control panel", "items accumulated near controls",
                "storage near control system", "combustible near control equipment", "boxes near process control"
            ],
            
            # ============================================================
            # 4.0 HUMAN ELEMENT - 9 features each
            # ============================================================
            'RIB_4.1_FIRE_EVACUATION': [
                "fire evacuation plan absent", "no emergency evacuation route", "fire escape plan missing",
                "evacuation assembly point undefined", "fire evacuation not adopted", "emergency exit unclear",
                "evacuation map not displayed", "fire evacuation plan needed", "emergency egress not planned"
            ],
            
            'RIB_4.2_SMOKING_AREA': [
                "designated smoking area missing", "no smoking zone facility", "smoking area not designated",
                "outdoor smoking area absent", "smoking section not designated", "smoking area not defined",
                "employee smoking area needed", "smoking zone not provided", "designated smoking spot missing"
            ],
            
            'RIB_4.3_HOT_WORK': [
                "welding hot work uncontrolled", "cutting hot work without permit", "grinding hot work area",
                "hot work without fire watch", "welding torch operation", "hot work permit not used",
                "active welding without control", "spark producing activity", "hot work site unmanaged"
            ],
            
            'RIB_4.4_GAS_TEST': [
                "no gas testing before hot work", "flammable gas test not done", "pre-hot work gas detection absent",
                "lel gas testing missing", "combustible gas measurement absent", "hot work no gas monitoring",
                "gas test before welding skipped", "atmospheric gas testing not performed", "gas detector not used hot work"
            ],
            
            'RIB_4.5_COOKING_FACILITIES': [
                "residential cooking in dormitory", "cooking area fire protection missing", "kitchen cooking without suppression",
                "cooking facility dormitory", "residential kitchen area", "cooking in sleeping quarter",
                "cooking equipment no fire wall", "kitchen no fire protection", "dormitory cooking area"
            ],
            
            'RIB_4.6_CONSTRUCTION_AREAS': [
                "active construction zone", "construction site within facility", "construction work ongoing",
                "construction safety barrier", "renovation construction area", "construction site restriction",
                "construction activity zone", "construction site within premise", "ongoing construction work"
            ],
            
            'RIB_4.7_SMOKING_POLICY': [
                "smoking policy not enforced", "unauthorized smoking area", "smoking prohibition breach",
                "smoking restriction violation", "unenforced smoking ban", "smoking policy non-compliance",
                "smoking in prohibited area", "smoking policy not followed", "smoking restriction ignored"
            ],
            
            'RIB_4.8_SAFETY_COMMITTEE': [
                "safety committee not established", "no safety committee", "health safety committee absent",
                "safety committee not formed", "safety committee missing", "workplace safety committee needed",
                "safety committee not active", "safety committee not functional", "no safety committee structure"
            ],
            
            'RIB_4.9_PATROL_CLOCKING': [
                "security patrol clocking absent", "no patrol checkpoint system", "guard patrol clocking missing",
                "patrol clocking station not installed", "security patrol point absent", "patrol monitoring not implemented",
                "guard tour clocking missing", "patrol checkpoint not used", "security clocking not done"
            ],
            
            # ============================================================
            # 5.0 PROCESS - 9 features each
            # ============================================================
            'RIB_5.1_SPRAY_COATING': [
                "spray coating booth not explosion proof", "spray booth electrical not rated", "paint spray booth coating factory",
                "coating spray facility", "spray painting booth no ex rated", "spray coating chamber",
                "spray booth no explosion proof wiring", "spray coating equipment", "booth electrical not explosion proof"
            ],
            
            'RIB_5.2_BATTERY_CHARGING': [
                "battery charging socket residential", "battery charging in dormitory", "charging point sleeping quarter",
                "battery charger in bedroom", "battery charging outlet dormitory", "charging socket living area",
                "battery charge point residence", "electrical battery charging dormitory", "battery charging sleeping area"
            ],
            
            'RIB_5.3_OVERHEAD_CRANE': [
                "overhead bridge crane inspection overdue", "overhead crane not inspected", "gantry crane no maintenance",
                "overhead crane system uninspected", "overhead hoist crane no testing", "bridge crane no inspection",
                "overhead crane no service record", "overhead crane track no inspection", "crane overdue inspection"
            ],
            
            'RIB_5.4_PIPE_LABELLING': [
                "pipe no identification label", "unlabeled industrial pipe", "pipe no marking system",
                "pipe no color coding", "pipe content label missing", "unidentified pipe system",
                "pipe no labeling scheme", "pipe tag identification absent", "unmarked pipe system"
            ],
            
            'RIB_5.5_BATTERY_CHARGING_VENTILATION': [
                "battery room no ventilation", "battery charging area no exhaust", "battery room hydrogen gas risk",
                "battery room no exhaust fan", "no ventilation battery charging", "battery area no air circulation",
                "battery room no mechanical ventilation", "battery charging no ventilation duct", "battery room ventilation absent"
            ],
            
            'RIB_5.6_SPRAY_PAINTING_AREA': [
                "spray painting in showroom", "indoor spray painting improper", "painting in showroom area",
                "spray painting enclosed space", "painting activity indoors showroom", "showroom painting work",
                "indoor paint spraying retail", "painting in retail showroom", "spray painting inside building"
            ],
            
            'RIB_5.7_TANK_THICKNESS': [
                "bulk storage tank shell corrosion", "storage tank thickness not tested", "tank bottom no inspection",
                "storage tank corrosion visible", "tank shell thickness unknown", "bulk tank integrity not tested",
                "storage tank shell corroded", "tank thickness not measured", "corroded tank shell storage"
            ],
            
            'RIB_5.8_TEMPERATURE_MONITORING': [
                "silo no temperature monitoring", "palm kernel storage no temp sensor", "storage no temperature probe",
                "no temperature monitoring system storage", "silo no heat sensor", "combustible storage no temperature check",
                "no temperature alarm silo", "spontaneous combustion risk", "storage pile no temperature monitoring"
            ],
            
            'RIB_5.9_TANK_MANAGEMENT': [
                "no tank management program", "storage tank no inspection program", "tank integrity not tested",
                "ultrasonic tank testing absent", "ndt tank inspection missing", "tank maintenance program absent",
                "storage tank no monitoring program", "tank inspection not scheduled", "tank management system missing"
            ],
            
            'RIB_5.10_WEIR_INTEGRITY': [
                "weir structural integrity untested", "hydroelectric weir no inspection", "dam weir structure",
                "weir integrity not tested", "spillway weir no testing", "weir no inspection program",
                "weir structural testing absent", "weir no piezometer", "weir seepage not monitored"
            ],
            
            'RIB_5.11_DUST_EXTRACTION': [
                "no dust extraction system", "dust collection equipment missing", "no dust extraction hood",
                "dust extractor absent", "no dust extraction fan", "dust collection ventilation missing",
                "dust extraction ductwork absent", "no dust removal system", "dust extraction filter not installed"
            ],
            
            'RIB_5.12_SURGE_PROTECTION_PV': [
                "solar pv no surge protection", "photovoltaic surge protector missing", "solar array no surge device",
                "pv system no spd", "solar inverter no surge protection", "solar farm surge protector absent",
                "pv no surge protection device", "solar panel no surge arrester", "photovoltaic spd not installed"
            ],
            
            'RIB_5.13_GAS_PIPELINE': [
                "natural gas pipeline overhead", "gas supply pipeline exposed", "gas pipeline above ground",
                "gas main pipeline visible", "gas distribution pipeline", "gas pipeline inspection needed",
                "gas pipeline integrity assessment", "gas pipe network overhead", "gas transmission pipeline exposed"
            ],
            
            'RIB_5.14_PARTIAL_DISCHARGE': [
                "ups partial discharge detected", "uninterruptible power supply pd", "ups insulation deterioration",
                "partial discharge testing ups needed", "ups electrical discharge", "ups pd measurement required",
                "ups high frequency discharge", "ups hfct testing", "partial discharge monitoring ups needed"
            ],
            
            'RIB_5.15_UNDERWATER_CORROSION': [
                "underwater bridge piling corrosion", "marine piling corrosion", "underwater structural corrosion detected",
                "bridge piling underwater inspection needed", "underwater cathodic protection", "marine structure corrosion",
                "underwater piling condition deteriorated", "submerged piling corrosion", "underwater structural inspection required"
            ],
            
            'RIB_5.16_STATIC_ELECTRICITY': [
                "no static electricity ionizing bar", "static eliminator system absent", "electrostatic discharge risk",
                "plastic film static control missing", "ionizing blower not installed", "static electricity prevention absent",
                "winding machine no static eliminator", "anti-static ionizer missing", "static charge control system absent"
            ]
        }
        
        # Comprehensive RIB section mapping (unchanged)
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
        
        # Object to broad category mapping (unchanged)
        self.object_to_broad_category = {
            # Perils
            'crack': 'PERILS',
            'ground crack': 'PERILS',
            'subsidence': 'PERILS',
            'pipe': 'PERILS',
            'burst pipe': 'PERILS',
            'drain': 'PERILS',
            'flood water': 'PERILS',
            'lightning arrester': 'PERILS',
            'lightning rod': 'PERILS',
            'tree': 'PERILS',
            'dead tree': 'PERILS',
            
            # Electrical
            'electrical panel': 'ELECTRICAL',
            'electrical board': 'ELECTRICAL',
            'open panel': 'ELECTRICAL',
            'transformer': 'ELECTRICAL',
            'exposed wire': 'ELECTRICAL',
            'burn mark': 'ELECTRICAL',
            
            # Housekeeping
            'grease': 'HOUSEKEEPING',
            'metal scrap': 'HOUSEKEEPING',
            'lpg cylinder': 'HOUSEKEEPING',
            'gas cylinder': 'HOUSEKEEPING',
            'storage drum': 'HOUSEKEEPING',
            'combustible material': 'HOUSEKEEPING',
            'generator': 'HOUSEKEEPING',
            
            # Human Element
            'fire extinguisher': 'HUMAN_ELEMENT',
            'exit sign': 'HUMAN_ELEMENT',
            'smoking area': 'HUMAN_ELEMENT',
            
            # Process
            'spray booth': 'PROCESS',
            'battery': 'PROCESS',
            'overhead crane': 'PROCESS',
            'solar panel': 'PROCESS'
        }
    
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
        max_subsections: int = 5
    ) -> Dict[str, Any]:
        """Tier 2: Detect specific RIB subsections with batched processing"""
        print(f"\n{'='*80}")
        print("TIER 2: RIB SUBSECTION DETECTION")
        print(f"{'='*80}")
        
        # Filter RIB subsections based on broad categories
        relevant_subsections = []
        
        for rib_key in self.tier2_rib_vocabulary.keys():
            if 'ELECTRICAL' in broad_categories and rib_key.startswith('RIB_2'):
                relevant_subsections.append(rib_key)
            elif 'HOUSEKEEPING' in broad_categories and rib_key.startswith('RIB_3'):
                relevant_subsections.append(rib_key)
            elif 'PERILS' in broad_categories and rib_key.startswith('RIB_1'):
                relevant_subsections.append(rib_key)
            elif 'HUMAN_ELEMENT' in broad_categories and rib_key.startswith('RIB_4'):
                relevant_subsections.append(rib_key)
            elif 'PROCESS' in broad_categories and rib_key.startswith('RIB_5'):
                relevant_subsections.append(rib_key)
        
        # Limit number of subsections to process for speed
        if len(relevant_subsections) > max_subsections:
            print(f"  ℹ️ Limiting to top {max_subsections} most relevant subsections for speed")
            relevant_subsections = relevant_subsections[:max_subsections]
        
        subsection_detections = {}
        
        # Process subsections in batches for better GPU utilization
        batch_size = 3  # Process 3 subsections at a time
        for i in range(0, len(relevant_subsections), batch_size):
            batch_keys = relevant_subsections[i:i+batch_size]
            
            for rib_key in batch_keys:
                vocabulary = self.tier2_rib_vocabulary[rib_key]
                
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
                threshold=threshold
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
        """Determine primary RIB subsection based on detection scores"""
        
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
            
            # Priority boosts for critical issues
            if 'RIB_2.4' in rib_key:  # Open panel - HIGHEST priority (safety critical)
                score *= 2.2
            elif 'RIB_2.3' in rib_key:  # Cardboard at electrical board - high priority
                score *= 2.0
            elif 'RIB_2.6' in rib_key:  # Burn mark - high priority
                score *= 1.8
            elif 'RIB_1.6' in rib_key:  # Lightning - high priority
                score *= 1.5
            elif 'RIB_3.4' in rib_key or 'RIB_3.5' in rib_key:  # Containment issues
                score *= 1.4
            
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
        print("RIB-ALIGNED HIERARCHICAL DETECTION (UNIQUE VOCABULARIES)")
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
        description='RIB-Aligned Object Detection using OWLv2 (Unique Vocabularies)'
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