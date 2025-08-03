#!/usr/bin/env python3
"""
Test script for enhanced PDF extraction
"""

import os
import sys
from enhanced_pdf_extractor import pdf_extractor

def test_extraction():
    """Test the enhanced PDF extraction with the sample document"""
    
    # Test with the sample text file
    sample_file = "sample_insurance_policy.txt"
    
    if not os.path.exists(sample_file):
        print(f"❌ Sample file {sample_file} not found")
        return False
    
    print("🧪 Testing enhanced PDF extraction...")
    print(f"📄 Testing with file: {sample_file}")
    
    try:
        # Extract text
        extracted_text = pdf_extractor.extract_text(sample_file)
        
        # Clean the text
        cleaned_text = pdf_extractor.clean_extracted_text(extracted_text)
        
        print(f"✅ Extraction successful!")
        print(f"📊 Extracted {len(cleaned_text)} characters")
        print(f"📝 First 200 characters: {cleaned_text[:200]}...")
        
        # Check if key terms are present
        key_terms = ['grace period', 'waiting period', 'maternity', 'cataract', 'NCD']
        found_terms = [term for term in key_terms if term.lower() in cleaned_text.lower()]
        
        print(f"🔍 Found key terms: {found_terms}")
        
        if len(found_terms) >= 3:
            print("✅ Extraction quality looks good!")
            return True
        else:
            print("⚠️  Extraction quality may need improvement")
            return False
            
    except Exception as e:
        print(f"❌ Extraction failed: {str(e)}")
        return False

def test_available_methods():
    """Test which extraction methods are available"""
    print("\n🔧 Testing available extraction methods...")
    
    methods = [
        ("pdfplumber", "pdfplumber"),
        ("PyPDF2", "PyPDF2"),
        ("OCR", "pytesseract"),
        ("Basic", "re")
    ]
    
    for method_name, module_name in methods:
        try:
            __import__(module_name)
            print(f"✅ {method_name} available")
        except ImportError:
            print(f"❌ {method_name} not available")

if __name__ == "__main__":
    print("🚀 Enhanced PDF Extraction Test")
    print("=" * 40)
    
    # Test available methods
    test_available_methods()
    
    # Test extraction
    success = test_extraction()
    
    if success:
        print("\n🎉 All tests passed! The enhanced extraction is working.")
    else:
        print("\n⚠️  Some tests failed. Check the output above.")
    
    sys.exit(0 if success else 1) 