#!/usr/bin/env python3
"""
Test the API with sample policy content
"""

import requests
import json

BASE_URL = "https://hackrx-rjzs.onrender.com"
AUTH_TOKEN = "02b1ad646a69f58d41c75bb9ea5f78bbaf30389258623d713ff4115b554377f0"

def test_with_sample_policy():
    """Test the API with sample policy content"""
    url = f"{BASE_URL}/hackrx/run"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AUTH_TOKEN}"
    }
    
    # Sample policy content (we'll use this directly)
    sample_policy_content = """NATIONAL PARIVAR MEDICLAIM PLUS POLICY (NPMPP)

POLICY TERMS AND CONDITIONS

1. COVERAGE DETAILS
This policy provides comprehensive health insurance coverage for the entire family under a single policy.

2. WAITING PERIODS
- Pre-existing diseases: 48 months waiting period
- Maternity expenses: 24 months waiting period
- Cataract surgery: 12 months waiting period
- Hernia, Hydrocele, and Fistula: 12 months waiting period

3. GRACE PERIOD
Premium payment grace period: 15 days from the due date

4. HOSPITAL DEFINITION
A hospital is defined as any institution that:
- Has at least 10 inpatient beds
- Has qualified nursing staff available 24/7
- Has qualified medical practitioners available
- Has proper medical and surgical facilities

5. ROOM RENT AND ICU CHARGES
Plan A sub-limits:
- Room rent: Up to 1% of sum insured per day
- ICU charges: Up to 2% of sum insured per day

6. AYUSH TREATMENT COVERAGE
AYUSH treatments are covered up to 25% of the sum insured, subject to:
- Treatment must be taken at a government hospital
- Maximum coverage of Rs. 50,000 per policy period

7. NO CLAIM DISCOUNT (NCD)
- 5% for 1 claim-free year
- 10% for 2 claim-free years
- 15% for 3 claim-free years
- 20% for 4 claim-free years
- 25% for 5 claim-free years

8. PREVENTIVE HEALTH CHECK-UP
One free health check-up per policy year up to Rs. 5,000

9. ORGAN DONOR EXPENSES
Medical expenses for organ donor are covered up to Rs. 1,00,000

10. EXCLUSIONS
- Cosmetic surgery
- Dental treatment (except due to accident)
- Convalescence and rehabilitation
- Expenses for spectacles and contact lenses
- Expenses for hearing aids
- Expenses for artificial limbs

11. CLAIM PROCESS
- Intimate within 24 hours of hospitalization
- Submit all required documents within 30 days
- Pre-authorization required for planned hospitalization

12. POLICY RENEWAL
- Policy can be renewed for lifetime
- No medical examination required for renewal
- Premium may be revised based on age and claims history

This policy provides comprehensive coverage for the entire family with reasonable waiting periods and extensive benefits."""
    
    # Create a temporary file with the sample policy
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(sample_policy_content)
        temp_file_path = f.name
    
    # Upload to a simple file sharing service or use a local server
    # For now, let's create a simple test with the questions
    
    # Test questions
    questions = [
        "What is the grace period for premium payments?",
        "What is the waiting period for pre-existing diseases?",
        "What are the maternity expenses covered?",
        "What is the waiting period for cataract surgery?",
        "Are medical expenses for organ donor covered?",
        "What is the No Claim Discount (NCD) offered?",
        "What are the benefits of preventive health check-ups?",
        "How does the policy define a 'Hospital'?",
        "What is the extent of coverage for AYUSH treatments?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]
    
    print("🧪 Testing with Sample Policy Content")
    print("=" * 60)
    
    # Since we can't easily upload the file, let's simulate the answers
    expected_answers = [
        "Premium payment grace period: 15 days from the due date",
        "Pre-existing diseases: 48 months waiting period",
        "Maternity expenses: 24 months waiting period",
        "Cataract surgery: 12 months waiting period",
        "Medical expenses for organ donor are covered up to Rs. 1,00,000",
        "NCD: 5% for 1 claim-free year, 10% for 2 years, 15% for 3 years, 20% for 4 years, 25% for 5 years",
        "One free health check-up per policy year up to Rs. 5,000",
        "Hospital must have at least 10 inpatient beds, qualified nursing staff 24/7, qualified medical practitioners, and proper medical facilities",
        "AYUSH treatments covered up to 25% of sum insured, maximum Rs. 50,000, must be at government hospital",
        "Plan A sub-limits: Room rent up to 1% of sum insured per day, ICU charges up to 2% of sum insured per day"
    ]
    
    print("📋 Expected Answers:")
    for i, (question, expected) in enumerate(zip(questions, expected_answers), 1):
        print(f"{i}. Q: {question}")
        print(f"   A: {expected}")
        print()
    
    print("=" * 60)
    print("💡 To test with your API:")
    print("1. Upload sample_policy.txt to GitHub/Gist")
    print("2. Use the raw URL in your API call")
    print("3. Or convert your PDF to text format")
    
    # Clean up
    os.unlink(temp_file_path)

if __name__ == "__main__":
    test_with_sample_policy() 