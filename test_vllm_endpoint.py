# test_vllm_qwen.py

import requests
import json

def test_vllm_server():
    """Test VLLM server with correct endpoints"""
    
    base_url = "http://localhost:8000"
    
    print("="*80)
    print("TESTING VLLM SERVER (vllm serve)")
    print("="*80)
    
    # Test 1: Health check
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"✓ Health: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"✗ Health check failed: {e}")
    
    # Test 2: Models endpoint
    print("\n2. Testing models endpoint...")
    try:
        response = requests.get(f"{base_url}/v1/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            print(f"✓ Models: {json.dumps(models, indent=2)}")
        else:
            print(f"✗ Status: {response.status_code}")
    except Exception as e:
        print(f"✗ Models check failed: {e}")
    
    # Test 3: Chat completions (OpenAI-compatible)
    print("\n3. Testing chat completions endpoint...")
    try:
        payload = {
            "model": "Qwen/Qwen3-VL-8B-Instruct",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello! What is the capital city of France?"
                }
            ],
            "max_tokens": 50,
            "temperature": 0.7
        }
        
        print(f"Request payload: {json.dumps(payload, indent=2)}")
        
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Response: {json.dumps(result, indent=2)}")
        else:
            print(f"✗ Error: {response.text}")
            
    except Exception as e:
        print(f"✗ Chat completion failed: {e}")
    
    # Test 4: Completions endpoint (alternative)
    print("\n4. Testing completions endpoint...")
    try:
        payload = {
            "model": "Qwen/Qwen3-VL-8B-Instruct",
            "prompt": "Hello! Say yes.",
            "max_tokens": 10
        }
        
        response = requests.post(
            f"{base_url}/v1/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Response: {json.dumps(result, indent=2)}")
        else:
            print(f"✗ Error: {response.text}")
            
    except Exception as e:
        print(f"✗ Completions failed: {e}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    test_vllm_server()