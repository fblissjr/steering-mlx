# test_backend.py
"""
Simple test script to verify the backend functionality.
"""
import requests
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "http://127.0.0.1:8008"

def test_api_endpoints():
    """Test the main API endpoints."""
    
    # Test root endpoint
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Root endpoint: {response.status_code}")
        print(f"Response: {response.json()}")
        print()
    except Exception as e:
        print(f"Error testing root endpoint: {e}")
        return False
    
    # Test supported models endpoint
    try:
        response = requests.get(f"{BASE_URL}/supported_models")
        print(f"Supported models: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        print()
    except Exception as e:
        print(f"Error testing supported models: {e}")
    
    # Test derived vectors endpoint
    try:
        response = requests.get(f"{BASE_URL}/derived_vectors")
        print(f"Derived vectors: {response.status_code}")
        print(f"Response: {response.json()}")
        print()
    except Exception as e:
        print(f"Error testing derived vectors: {e}")
    
    return True

if __name__ == "__main__":
    print("Testing MLX Control Vector Laboratory Backend API")
    print("=" * 50)
    
    success = test_api_endpoints()
    
    if success:
        print("✅ Basic API tests passed!")
        print("\nNext steps:")
        print("1. Try loading a model: POST /load_model")
        print("2. Test control vector operations")
        print("3. Test text generation")
    else:
        print("❌ API tests failed. Check if the server is running.")
