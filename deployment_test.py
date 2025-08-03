#!/usr/bin/env python3
"""
Simple deployment test to verify the system works
"""

import os
import sys

def test_basic_imports():
    """Test that basic imports work"""
    print("🧪 Testing basic imports...")
    
    try:
        import requests
        print("✅ requests imported successfully")
    except ImportError as e:
        print(f"❌ requests import failed: {e}")
        return False
    
    try:
        import fastapi
        print("✅ fastapi imported successfully")
    except ImportError as e:
        print(f"❌ fastapi import failed: {e}")
        return False
    
    try:
        import uvicorn
        print("✅ uvicorn imported successfully")
    except ImportError as e:
        print(f"❌ uvicorn import failed: {e}")
        return False
    
    return True

def test_pdf_imports():
    """Test PDF processing imports"""
    print("\n📄 Testing PDF processing imports...")
    
    try:
        import PyPDF2
        print("✅ PyPDF2 imported successfully")
    except ImportError as e:
        print(f"❌ PyPDF2 import failed: {e}")
        return False
    
    try:
        import pdfplumber
        print("✅ pdfplumber imported successfully")
    except ImportError as e:
        print(f"❌ pdfplumber import failed: {e}")
        return False
    
    return True

def test_enhanced_extractor():
    """Test the enhanced PDF extractor"""
    print("\n🔧 Testing enhanced PDF extractor...")
    
    try:
        from enhanced_pdf_extractor import pdf_extractor
        print("✅ Enhanced PDF extractor imported successfully")
        
        # Test with sample file
        if os.path.exists("sample_insurance_policy.txt"):
            text = pdf_extractor.extract_text("sample_insurance_policy.txt")
            if text and len(text) > 100:
                print("✅ PDF extraction test passed")
                return True
            else:
                print("⚠️ PDF extraction returned insufficient text")
                return False
        else:
            print("⚠️ Sample file not found, skipping extraction test")
            return True
            
    except Exception as e:
        print(f"❌ Enhanced PDF extractor test failed: {e}")
        return False

def test_main_app():
    """Test that the main app can be imported"""
    print("\n🚀 Testing main app import...")
    
    try:
        from main import app
        print("✅ Main app imported successfully")
        return True
    except Exception as e:
        print(f"❌ Main app import failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Deployment Test Suite")
    print("=" * 40)
    
    tests = [
        test_basic_imports,
        test_pdf_imports,
        test_enhanced_extractor,
        test_main_app
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        else:
            print(f"❌ Test {test.__name__} failed")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Deployment should work.")
        sys.exit(0)
    else:
        print("⚠️ Some tests failed. Check the output above.")
        sys.exit(1) 