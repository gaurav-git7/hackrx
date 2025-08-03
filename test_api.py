#!/usr/bin/env python3
"""
Simple test script for HackRx API endpoints
"""

import requests
import json

# Replace with your actual Render URL
BASE_URL = "https://hackrx-rjzs.onrender.com"
AUTH_TOKEN = "02b1ad646a69f58d41c75bb9ea5f78bbaf30389258623d713ff4115b554377f0"

def test_get_endpoint(endpoint):
    """Test a GET endpoint"""
    url = f"{BASE_URL}{endpoint}"
    print(f"\n🔍 Testing GET {endpoint}")
    try:
        response = requests.get(url)
        print(f"✅ Status: {response.status_code}")
        print(f"📄 Response: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def test_post_endpoint(endpoint, data):
    """Test a POST endpoint"""
    url = f"{BASE_URL}{endpoint}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AUTH_TOKEN}"
    }
    print(f"\n🔍 Testing POST {endpoint}")
    print(f"📤 Data: {json.dumps(data, indent=2)}")
    try:
        response = requests.post(url, headers=headers, json=data)
        print(f"✅ Status: {response.status_code}")
        print(f"📄 Response: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def main():
    print("🚀 Testing HackRx API Endpoints")
    print("=" * 50)
    
    # Test GET endpoints
    test_get_endpoint("/test")
    test_get_endpoint("/health")
    
    # Test POST endpoints
    test_post_endpoint("/test-post", {"test": "data"})
    
    test_post_endpoint("/hackrx/simple", {
        "documents": "https://example.com/test.txt",
        "questions": ["What is this document about?"]
    })
    
    print("\n" + "=" * 50)
    print("🎉 Testing complete!")

if __name__ == "__main__":
    main() 