#!/usr/bin/env python3
"""
Test the API with a text file URL
"""

import requests
import json

BASE_URL = "https://hackrx-rjzs.onrender.com"
AUTH_TOKEN = "02b1ad646a69f58d41c75bb9ea5f78bbaf30389258623d713ff4115b554377f0"

def test_api_with_text_file():
    """Test the API with a text file URL"""
    url = f"{BASE_URL}/hackrx/run"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AUTH_TOKEN}"
    }
    
    # Use a simple text file URL (you can replace this with your own)
    # For now, let's use a GitHub raw URL for the sample policy
    documents_url = "https://raw.githubusercontent.com/gaurav-git7/hackrx/main/sample_policy.txt"
    
    # Test questions
    questions = [
        "What is the grace period for premium payments?",
        "What is the waiting period for pre-existing diseases?",
        "What is the waiting period for cataract surgery?",
        "Are medical expenses for organ donor covered?",
        "What is the No Claim Discount (NCD) offered?"
    ]
    
    data = {
        "documents": documents_url,
        "questions": questions
    }
    
    print("🧪 Testing API with Text File")
    print("=" * 50)
    print(f"📄 Document URL: {documents_url}")
    print(f"❓ Questions: {len(questions)}")
    print()
    
    try:
        print("🚀 Sending request to API...")
        response = requests.post(url, headers=headers, json=data, timeout=60)
        
        print(f"✅ Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("📋 Answers:")
            for i, answer in enumerate(result.get("answers", []), 1):
                print(f"{i}. {answer}")
                print()
        else:
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    test_api_with_text_file() 