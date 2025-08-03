#!/usr/bin/env python3
"""Test API authentication"""

import requests
import base64
import json

# Test configuration
API_URL = "http://localhost:4000"
CORRECT_KEY = "cgrfYmh5qQ86qv6udyWmm6sEq5eH5Bwmkmbxx6XyUX"
WRONG_KEY = "wrong-key-123"
TEST_FILE = "test_auth.txt"

# Create a simple test file
with open(TEST_FILE, "w") as f:
    f.write("Test content for API authentication")

print("Testing OCR API Authentication\n")

# Test 1: Health check (no auth required)
print("1. Testing health endpoint (no auth required):")
try:
    response = requests.get(f"{API_URL}/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
except Exception as e:
    print(f"   Error: {e}")

print("\n2. Testing file upload WITHOUT key (should fail):")
try:
    with open(TEST_FILE, "rb") as f:
        files = {"file": (TEST_FILE, f, "text/plain")}
        data = {"preserve_formatting": "true"}
        response = requests.post(f"{API_URL}/ocr/file", files=files, data=data)
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
except Exception as e:
    print(f"   Error: {e}")

print("\n3. Testing file upload with WRONG key (should fail):")
try:
    with open(TEST_FILE, "rb") as f:
        files = {"file": (TEST_FILE, f, "text/plain")}
        data = {
            "preserve_formatting": "true",
            "key": WRONG_KEY
        }
        response = requests.post(f"{API_URL}/ocr/file", files=files, data=data)
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
except Exception as e:
    print(f"   Error: {e}")

print("\n4. Testing file upload with CORRECT key (should succeed):")
try:
    with open(TEST_FILE, "rb") as f:
        files = {"file": (TEST_FILE, f, "text/plain")}
        data = {
            "preserve_formatting": "true",
            "key": CORRECT_KEY
        }
        response = requests.post(f"{API_URL}/ocr/file", files=files, data=data)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"   Success: {result['success']}")
        print(f"   Text extracted: {result['text'][:50]}...")
    else:
        print(f"   Response: {response.json()}")
except Exception as e:
    print(f"   Error: {e}")

print("\n5. Testing base64 endpoint WITHOUT key (should fail):")
try:
    with open(TEST_FILE, "rb") as f:
        content = base64.b64encode(f.read()).decode()
    
    payload = {
        "filename": TEST_FILE,
        "content": content,
        "preserve_formatting": True
    }
    response = requests.post(f"{API_URL}/ocr/base64", json=payload)
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
except Exception as e:
    print(f"   Error: {e}")

print("\n6. Testing base64 endpoint with CORRECT key (should succeed):")
try:
    with open(TEST_FILE, "rb") as f:
        content = base64.b64encode(f.read()).decode()
    
    payload = {
        "filename": TEST_FILE,
        "content": content,
        "preserve_formatting": True,
        "key": CORRECT_KEY
    }
    response = requests.post(f"{API_URL}/ocr/base64", json=payload)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"   Success: {result['success']}")
        print(f"   Text extracted: {result['text'][:50]}...")
    else:
        print(f"   Response: {response.json()}")
except Exception as e:
    print(f"   Error: {e}")

# Cleanup
import os
os.remove(TEST_FILE)
print("\n✅ Authentication tests completed!")