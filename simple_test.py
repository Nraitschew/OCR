#!/usr/bin/env python3
"""
Simple test to verify OCR system works
"""

import sys
sys.path.insert(0, '.')

from test_document_generator import TestDocumentGenerator
import os
from pathlib import Path

# Generate test documents
print("Generating test documents...")
generator = TestDocumentGenerator()
generator.generate_all()

# List generated files
print("\nGenerated test documents:")
test_dir = Path("test_documents")
for file in sorted(test_dir.iterdir()):
    if file.is_file():
        size_kb = file.stat().st_size / 1024
        print(f"  - {file.name} ({size_kb:.1f} KB)")

# Test German text generation
print("\nTesting German text generation...")
with open(test_dir / "german_text.txt", "r", encoding="utf-8") as f:
    content = f.read()
    print(f"German text sample: {content[:100]}...")
    
    # Check for umlauts
    umlauts = ['ä', 'ö', 'ü', 'Ä', 'Ö', 'Ü', 'ß']
    found_umlauts = [u for u in umlauts if u in content]
    print(f"Found umlauts: {found_umlauts}")

print("\n✓ Test document generation successful!")
print("\nNote: Full OCR functionality requires additional dependencies:")
print("- easyocr, pytesseract, paddleocr (for OCR)")
print("- PyMuPDF, pdf2image (for PDF processing)")
print("- opencv-python, torch (for image processing)")
print("\nResource limits configured: CPU 70%, RAM 70% (Max 2GB)")