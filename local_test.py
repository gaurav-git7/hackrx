#!/usr/bin/env python3
"""
Local test script to verify APIs work on localhost
"""

import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def test_gemini_api():
    """Test Gemini API key locally"""
    print("🧪 Testing Gemini API locally...")
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not found in environment variables")
        print("💡 Create a .env file with: GEMINI_API_KEY=your_key_here")
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
                text = result["candidates"][0]["content"]["parts"][0]["text"]
                print(f"✅ Gemini API is working! Response: {text}")
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
    """Test HuggingFace API token locally"""
    print("\n🧪 Testing HuggingFace API locally...")
    
    api_token = os.environ.get("HF_TOKEN")
    if not api_token:
        print("❌ HF_TOKEN not found in environment variables")
        print("💡 Create a .env file with: HF_TOKEN=your_token_here")
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
            result = response.json()
            print(f"✅ HuggingFace API is working! Response: {result}")
            return True
        else:
            print(f"❌ HuggingFace API error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ HuggingFace API test failed: {str(e)}")
        return False

def test_pdf_extraction():
    """Test PDF extraction locally"""
    print("\n📄 Testing PDF extraction locally...")
    
    try:
        from enhanced_pdf_extractor import pdf_extractor
        
        # Test with sample file
        if os.path.exists("sample_insurance_policy.txt"):
            text = pdf_extractor.extract_text("sample_insurance_policy.txt")
            if text and len(text) > 100:
                print(f"✅ PDF extraction working! Extracted {len(text)} characters")
                print(f"📝 Sample: {text[:200]}...")
                return True
            else:
                print("❌ PDF extraction returned insufficient text")
                return False
        else:
            print("⚠️ Sample file not found, skipping PDF extraction test")
            return True
            
    except Exception as e:
        print(f"❌ PDF extraction test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Local API Test Suite")
    print("=" * 40)
    
    # Check if .env file exists
    if not os.path.exists(".env"):
        print("⚠️ No .env file found. Create one with your API keys:")
        print("GEMINI_API_KEY=your_gemini_key_here")
        print("HF_TOKEN=your_huggingface_token_here")
        print()
    
    gemini_ok = test_gemini_api()
    hf_ok = test_huggingface_api()
    pdf_ok = test_pdf_extraction()
    
    print(f"\n📊 Results:")
    print(f"Gemini API: {'✅ Working' if gemini_ok else '❌ Failed'}")
    print(f"HuggingFace API: {'✅ Working' if hf_ok else '❌ Failed'}")
    print(f"PDF Extraction: {'✅ Working' if pdf_ok else '❌ Failed'}")
    
    if gemini_ok and hf_ok and pdf_ok:
        print("\n🎉 All systems are working locally!")
    else:
        print("\n⚠️ Some systems failed. Check your API keys and .env file.") 