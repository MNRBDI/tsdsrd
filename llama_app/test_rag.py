import requests
import base64
import json

API_URL = "http://localhost:8004"

def test_text_query():
    """Test text-based RAG query"""
    print("\n" + "="*80)
    print("TEST 1: TEXT QUERY")
    print("="*80)
    
    query_text = """The site is situated in an area that frequently experiences intense atmospheric disturbances, 
    creating a serious risk to both personnel safety and operational stability. These sudden high-energy events increase the 
    chances of damage to sensitive equipment, unexpected power interruptions, and even fire outbreaks."""
    
    response = requests.post(
        f"{API_URL}/generate_recommendations",
        json={
            "query_text": query_text,
            "top_k": 3,
            "similarity_threshold": 0.4,
            "max_tokens": 2048,
            "temperature": 0.2
        }
    )
    
    result = response.json()
    
    print(f"\n📝 Query: {query_text}")
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS:")
    print(f"{'='*80}")
    print(result['answer'])
    print(f"\n{'='*80}")
    print(f"📚 Sources: {result['num_sources']}")
    for source in result.get('sources', []):
        print(f"  • Section {source['section']}: {source['title']} (similarity: {source['similarity']})")

def test_image_query(image_path):
    """Test image-based RAG query"""
    print("\n" + "="*80)
    print("TEST 2: IMAGE QUERY")
    print("="*80)
    
    # Read and encode image
    with open(image_path, "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode()
    
    response = requests.post(
        f"{API_URL}/generate_recommendations",
        json={
            "image_base64": image_base64,
            "top_k": 3,
            "similarity_threshold": 0.4,
            "max_tokens": 2048,
            "temperature": 0.2
        }
    )
    
    result = response.json()
    
    print(f"\n📸 Image: {image_path}")
    print(f"\n{'='*80}")
    print("IMAGE DESCRIPTION:")
    print(f"{'='*80}")
    print(result.get('image_description', 'N/A'))
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS:")
    print(f"{'='*80}")
    print(result['answer'])
    print(f"\n{'='*80}")
    print(f"📚 Sources: {result['num_sources']}")
    for source in result.get('sources', []):
        print(f"  • Section {source['section']}: {source['title']} (similarity: {source['similarity']})")

if __name__ == "__main__":
    # Test text query
    test_text_query()
    
    # Test image query
    test_image_query("/home/amir/Desktop/MRE TSD/lightning-strike-2.png")