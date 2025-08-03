#!/usr/bin/env python3
"""
Test the fixed hackrx endpoint
"""

import requests
import json

BASE_URL = "https://hackrx-rjzs.onrender.com"
AUTH_TOKEN = "02b1ad646a69f58d41c75bb9ea5f78bbaf30389258623d713ff4115b554377f0"

def test_fixed_endpoint():
    """Test the fixed hackrx endpoint"""
    url = f"{BASE_URL}/hackrx/fixed"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AUTH_TOKEN}"
    }
    
    # Test data
    data = {
        "documents": "https://raw.githubusercontent.com/example/test/main/README.md",
        "questions": ["What is this document about?"]
    }
    
    print(f"🔍 Testing fixed endpoint: {url}")
    print(f"📤 Data: {json.dumps(data, indent=2)}")
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        print(f"✅ Status: {response.status_code}")
        print(f"📄 Response: {response.text}")
        return response.status_code == 200
    except requests.exceptions.Timeout:
        print("❌ Timeout error")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection error: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Fixed HackRx Endpoint")
    print("=" * 50)
    test_fixed_endpoint()
    print("=" * 50)
    print("�� Test complete!") 