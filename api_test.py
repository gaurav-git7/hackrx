#!/usr/bin/env python3
"""
Test script to verify API keys are working
"""

import os
import requests

def test_gemini_api():
    """Test Gemini API key"""
    print("🧪 Testing Gemini API...")
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not found in environment variables")
        return False
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    data = {
        "contents": [{
            "parts": [{"text": "Hello, this is a test message. Please respond with 'API working' if you can see this."}]
        }],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 50,
        }
    }
    
    try:
        response = requests.post(url, json=data, timeout=10)
        if response.status_code == 200:
            result = response.json()
            if "candidates" in result and len(result["candidates"]) > 0:
                print("✅ Gemini API is working!")
                return True
            else:
                print("❌ Gemini API returned no candidates")
                return False
        else:
            print(f"❌ Gemini API error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Gemini API test failed: {str(e)}")
        return False

def test_huggingface_api():
    """Test HuggingFace API token"""
    print("\n🧪 Testing HuggingFace API...")
    
    api_token = os.environ.get("HF_TOKEN")
    if not api_token:
        print("❌ HF_TOKEN not found in environment variables")
        return False
    
    url = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
    
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    
    data = {"inputs": "Hello, this is a test."}
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        if response.status_code == 200:
            print("✅ HuggingFace API is working!")
            return True
        else:
            print(f"❌ HuggingFace API error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ HuggingFace API test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 API Key Test Suite")
    print("=" * 40)
    
    gemini_ok = test_gemini_api()
    hf_ok = test_huggingface_api()
    
    print(f"\n📊 Results:")
    print(f"Gemini API: {'✅ Working' if gemini_ok else '❌ Failed'}")
    print(f"HuggingFace API: {'✅ Working' if hf_ok else '❌ Failed'}")
    
    if gemini_ok and hf_ok:
        print("\n🎉 All APIs are working!")
    else:
        print("\n⚠️ Some APIs are not working. Check your environment variables.") 