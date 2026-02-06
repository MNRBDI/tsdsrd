#!/usr/bin/env python3
"""
Direct test of OWLv2 API to diagnose the 500 error.
This script tests both the original image and a 1024px resized version.
"""

import requests
import sys
from pathlib import Path
from PIL import Image
import tempfile
import os

def test_owlv2_api():
    """Test OWLv2 API with diagnostic output"""
    
    owlv2_url = "http://localhost:8010"
    detect_url = f"{owlv2_url}/detect"
    
    # Test image - use one from your RIB_images folder
    test_images = [
        "/home/amir/Desktop/MRE TSD/RIB_images/page34_img1.png",
        "/home/amir/Desktop/MRE TSD/extracted_images/0.png",
    ]
    
    # Find first existing image
    image_path = None
    for img in test_images:
        if Path(img).exists():
            image_path = img
            break
    
    if not image_path:
        print("❌ No test images found. Create one or update the paths.")
        return
    
    print(f"🧪 Testing OWLv2 API at {owlv2_url}")
    print(f"📸 Using image: {image_path}")
    
    # Test 1: Original image
    print("\n" + "="*80)
    print("TEST 1: Sending ORIGINAL IMAGE to OWLv2")
    print("="*80)
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (Path(image_path).name, f, 'image/png')}
            print(f"📤 POST {detect_url}")
            print(f"   File: {Path(image_path).name}")
            
            response = requests.post(
                detect_url,
                files=files,
                timeout=30
            )
            
            print(f"📥 Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    print(f"✅ Success!")
                    print(f"   Tier 1 detections: {result['data'].get('tier1_detections', 0)}")
                    print(f"   Subsections: {len(result['data'].get('all_applicable_subsections', []))}")
                else:
                    print(f"❌ API error: {result.get('message')}")
            else:
                print(f"❌ HTTP Error")
                try:
                    print(f"   Response: {response.json()}")
                except:
                    print(f"   Response text: {response.text[:500]}")
    
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    # Test 2: Resized image (1024px)
    print("\n" + "="*80)
    print("TEST 2: Sending RESIZED (1024px) IMAGE to OWLv2")
    print("="*80)
    
    try:
        img = Image.open(image_path).convert("RGB")
        width, height = img.size
        print(f"📏 Original size: {width}x{height}")
        
        # Resize to 1024px
        if max(width, height) > 1024:
            if width > height:
                new_width = 1024
                new_height = int((height / width) * 1024)
            else:
                new_height = 1024
                new_width = int((width / height) * 1024)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            print(f"📏 Resized to: {new_width}x{new_height}")
        
        # Save to temp file and send
        temp_fd, temp_path = tempfile.mkstemp(suffix=".png")
        try:
            img.save(temp_path, "PNG")
            file_size = os.path.getsize(temp_path)
            print(f"💾 Temp file: {Path(temp_path).name} ({file_size} bytes)")
            
            with open(temp_path, 'rb') as f:
                files = {'file': (Path(image_path).name, f, 'image/png')}
                print(f"📤 POST {detect_url}")
                
                response = requests.post(
                    detect_url,
                    files=files,
                    timeout=30
                )
                
                print(f"📥 Status: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get('success'):
                        print(f"✅ Success!")
                        print(f"   Tier 1 detections: {result['data'].get('tier1_detections', 0)}")
                        print(f"   Subsections: {len(result['data'].get('all_applicable_subsections', []))}")
                    else:
                        print(f"❌ API error: {result.get('message')}")
                else:
                    print(f"❌ HTTP Error")
                    try:
                        print(f"   Response: {response.json()}")
                    except:
                        print(f"   Response text: {response.text[:500]}")
        finally:
            os.close(temp_fd)
            os.unlink(temp_path)
    
    except Exception as e:
        print(f"❌ Exception: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("💡 DIAGNOSIS TIPS:")
    print("="*80)
    print("1. If TEST 1 works but TEST 2 fails: Issue is with image resizing/format")
    print("   → Solution: Skip resizing for OWLv2, send original image")
    print("2. If both fail: OWLv2 server issue or API endpoint problem")
    print("   → Check OWLv2 server logs: docker logs owlv2_server")
    print("3. Check OWLv2 server is running: curl http://localhost:8010/health")

if __name__ == "__main__":
    test_owlv2_api()
