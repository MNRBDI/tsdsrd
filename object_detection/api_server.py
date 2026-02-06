"""
FastAPI server for RIB-Aligned OWLv2 Object Detection
Deploy with: uvicorn api_server:app --host 0.0.0.0 --port 8010
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from pathlib import Path
import tempfile
from typing import Optional, List
import json

from OWLv2_tiers_improv import RIBAlignedDetector

# Initialize FastAPI app
app = FastAPI(
    title="RIB-Aligned OWLv2 Detection API",
    description="Object detection system for Risk Improvement Benchmark indicators",
    version="1.0.0"
)

# Global detector instance
detector = None

class DetectionParams(BaseModel):
    tier1_threshold: float = 0.1
    tier2_threshold: float = 0.10

class DetectionResponse(BaseModel):
    success: bool
    message: str
    data: Optional[dict] = None
    error: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    """Initialize detector on startup"""
    global detector
    print("Loading OWLv2 detector...")
    detector = RIBAlignedDetector()
    print("✓ Detector loaded successfully")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "detector_ready": detector is not None
    }

@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    tier1_threshold: float = 0.1,
    tier2_threshold: float = 0.10
):
    """
    Upload an image and perform RIB detection.
    
    - **file**: Image file (PNG, JPG, etc.)
    - **tier1_threshold**: Confidence threshold for Tier 1 detection (0-1)
    - **tier2_threshold**: Confidence threshold for Tier 2 detection (0-1)
    """
    
    if detector is None:
        raise HTTPException(status_code=500, detail="Detector not initialized")
    
    try:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name
        
        print(f"\n📥 Processing image: {file.filename}")
        
        # Run detection
        results = detector.detect_hierarchical(
            image_path=tmp_path,
            tier1_threshold=tier1_threshold,
            tier2_threshold=tier2_threshold,
            visualize=False
        )
        
        # Extract key information
        summary = results['summary']
        
        response_data = {
            'image_name': file.filename,
            'tier1_detections': summary['total_tier1_detections'],
            'broad_categories': summary['identified_broad_categories'],
            'primary_rib_subsection': summary.get('primary_rib_subsection', None),
            'subsection_score': summary.get('subsection_score', None),
            'detection_count': summary.get('detection_count', None),
            'all_applicable_subsections': summary.get('all_applicable_subsections', [])
        }
        
        # Clean up temp file
        Path(tmp_path).unlink()
        
        return {
            "success": True,
            "message": "Detection completed successfully",
            "data": response_data
        }
    
    except Exception as e:
        print(f"❌ Error during detection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/detect-with-circle")
async def detect_with_circle(
    file: UploadFile = File(...),
    tier1_threshold: float = 0.1,
    tier2_threshold: float = 0.10
):
    """
    Upload an image with red circle indicator and perform focused RIB detection.
    
    Images with red circles will automatically crop to the circle region.
    
    - **file**: Image file (PNG, JPG, etc.)
    - **tier1_threshold**: Confidence threshold for Tier 1 detection (0-1)
    - **tier2_threshold**: Confidence threshold for Tier 2 detection (0-1)
    """
    
    if detector is None:
        raise HTTPException(status_code=500, detail="Detector not initialized")
    
    try:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name
        
        print(f"\n📥 Processing image with circle detection: {file.filename}")
        
        # Run detection
        results = detector.detect_hierarchical(
            image_path=tmp_path,
            tier1_threshold=tier1_threshold,
            tier2_threshold=tier2_threshold,
            visualize=False
        )
        
        # Extract key information
        summary = results['summary']
        circle_info = results['tier1'].get('circle_info', {})
        
        response_data = {
            'image_name': file.filename,
            'red_circle_detected': circle_info.get('detected', False),
            'circle_center': circle_info.get('center', None),
            'circle_radius': circle_info.get('radius', None),
            'tier1_detections': summary['total_tier1_detections'],
            'broad_categories': summary['identified_broad_categories'],
            'primary_rib_subsection': summary.get('primary_rib_subsection', None),
            'subsection_score': summary.get('subsection_score', None),
            'detection_count': summary.get('detection_count', None),
            'all_applicable_subsections': summary.get('all_applicable_subsections', [])
        }
        
        # Clean up temp file
        Path(tmp_path).unlink()
        
        return {
            "success": True,
            "message": "Detection completed successfully",
            "data": response_data
        }
    
    except Exception as e:
        print(f"❌ Error during detection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/detect-top5")
async def detect_top5(
    file: UploadFile = File(...),
    tier1_threshold: float = 0.1,
    tier2_threshold: float = 0.10
):
    """
    Upload an image and get top 5 RIB subsection detections for RAG integration.
    
    Returns detailed detection results including:
    - Top 5 ranked RIB subsections with scores
    - Detected indicators for each subsection
    - Tier 1 broad category detections
    
    - **file**: Image file (PNG, JPG, etc.)
    - **tier1_threshold**: Confidence threshold for Tier 1 detection (0-1)
    - **tier2_threshold**: Confidence threshold for Tier 2 detection (0-1)
    """
    
    if detector is None:
        raise HTTPException(status_code=500, detail="Detector not initialized")
    
    try:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name
        
        print(f"\n📥 Processing image for top 5 detection: {file.filename}")
        
        # Run detection
        results = detector.detect_hierarchical(
            image_path=tmp_path,
            tier1_threshold=tier1_threshold,
            tier2_threshold=tier2_threshold,
            visualize=False
        )
        
        # Extract key information
        summary = results['summary']
        circle_info = results['tier1'].get('circle_info', {})
        tier1_detections = results['tier1'].get('detections', [])
        
        # Get top 5 subsections with full details
        all_subsections = summary.get('all_applicable_subsections', [])
        top5_subsections = all_subsections[:5] if all_subsections else []
        
        # Get tier1 detected objects (top 10)
        tier1_objects = [
            {
                'object': det['object'],
                'confidence': round(det['confidence'], 3),
                'category': det['broad_category']
            }
            for det in sorted(tier1_detections, key=lambda x: x['confidence'], reverse=True)[:10]
        ]
        
        response_data = {
            'image_name': file.filename,
            'red_circle_detected': circle_info.get('detected', False),
            'tier1_detection_count': len(tier1_detections),
            'tier1_objects': tier1_objects,
            'broad_categories': summary['identified_broad_categories'],
            'top5_rib_subsections': [
                {
                    'rank': i + 1,
                    'section': sub['section'],
                    'score': round(sub['score'], 3),
                    'detection_count': sub['detections']
                }
                for i, sub in enumerate(top5_subsections)
            ],
            'primary_rib_subsection': summary.get('primary_rib_subsection', None),
            'primary_score': round(summary.get('subsection_score', 0), 3) if summary.get('subsection_score') else None,
            'has_detections': len(top5_subsections) > 0
        }
        
        # Clean up temp file
        Path(tmp_path).unlink()
        
        return {
            "success": True,
            "message": "Top 5 detection completed successfully",
            "data": response_data
        }
    
    except Exception as e:
        print(f"❌ Error during detection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.get("/info")
async def get_info():
    """Get detector information and available RIB subsections"""
    
    if detector is None:
        raise HTTPException(status_code=500, detail="Detector not initialized")
    
    return {
        "model": "google/owlv2-base-patch16-ensemble",
        "tier1_objects_count": len(detector.tier1_objects),
        "tier2_subsections": len(detector.tier2_rib_vocabulary),
        "subsection_mapping": detector.rib_section_mapping,
        "features": [
            "Tier 1: General RIB indicator detection",
            "Tier 2: Specific RIB subsection identification",
            "Red circle region-of-interest detection",
            "Safety-critical priority boosting"
        ]
    }

if __name__ == "__main__":
    print("🚀 Starting RIB-Aligned OWLv2 Detection API Server")
    print("📍 Server will be available at: http://0.0.0.0:8010")
    print("📚 API Documentation: http://localhost:8010/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8010,
        log_level="info"
    )
