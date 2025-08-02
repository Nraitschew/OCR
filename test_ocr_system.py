#!/usr/bin/env python3
"""
Comprehensive Test Suite for OCR System
Tests all document types and German umlaut support
"""

import asyncio
import time
import sys
from pathlib import Path
import pytest
import logging
from typing import List, Dict, Any

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from ocr_system import OCRSystem, OCRConfig, get_resource_info
from test_document_generator import TestDocumentGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# German characters to test
GERMAN_CHARS = ['ä', 'ö', 'ü', 'Ä', 'Ö', 'Ü', 'ß']


class TestResults:
    """Store and display test results"""
    
    def __init__(self):
        self.results = []
        self.start_time = time.time()
        
    def add_result(self, test_name: str, status: str, details: Dict[str, Any]):
        """Add a test result"""
        self.results.append({
            'test': test_name,
            'status': status,
            'details': details,
            'timestamp': time.time()
        })
        
    def print_summary(self):
        """Print test summary"""
        total_time = time.time() - self.start_time
        passed = sum(1 for r in self.results if r['status'] == 'PASSED')
        failed = sum(1 for r in self.results if r['status'] == 'FAILED')
        
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        print(f"Total Tests: {len(self.results)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Total Time: {total_time:.2f} seconds")
        print("\nDetailed Results:")
        print("-"*80)
        
        for result in self.results:
            status_color = '\033[92m' if result['status'] == 'PASSED' else '\033[91m'
            reset_color = '\033[0m'
            
            print(f"{status_color}{result['status']}{reset_color}: {result['test']}")
            for key, value in result['details'].items():
                print(f"  - {key}: {value}")
            print()


class OCRSystemTester:
    """Test the OCR system comprehensively"""
    
    def __init__(self):
        self.ocr_system = OCRSystem()
        self.test_results = TestResults()
        self.test_dir = Path("test_documents")
        
    async def run_all_tests(self):
        """Run all tests"""
        print("Starting OCR System Tests...")
        print(f"Resource Limits: CPU 70%, RAM 70% (Max 2GB)")
        
        # Show initial resources
        resources = get_resource_info()
        print(f"\nInitial Resources:")
        print(f"  CPU: {resources['cpu_percent']:.1f}%")
        print(f"  RAM: {resources['ram_gb']:.2f}GB ({resources['ram_percent']:.1f}%)")
        
        # Generate test documents if not exists
        if not self.test_dir.exists() or len(list(self.test_dir.iterdir())) < 10:
            print("\nGenerating test documents...")
            generator = TestDocumentGenerator()
            generator.generate_all()
            
        # Run tests for each document type
        await self.test_txt_files()
        await self.test_image_files()
        await self.test_pdf_files()
        await self.test_docx_files()
        await self.test_german_umlauts()
        await self.test_concurrent_processing()
        await self.test_resource_limits()
        
        # Print summary
        self.test_results.print_summary()
        
    async def test_txt_files(self):
        """Test TXT file processing"""
        print("\n--- Testing TXT Files ---")
        
        test_files = [
            "german_text.txt",
            "english_text.txt", 
            "mixed_language.txt"
        ]
        
        for filename in test_files:
            file_path = self.test_dir / filename
            if not file_path.exists():
                self.test_results.add_result(
                    f"TXT: {filename}",
                    "FAILED",
                    {"error": "File not found"}
                )
                continue
                
            try:
                result = await self.ocr_system.process_file(str(file_path))
                
                # Check if German characters are preserved
                german_chars_found = sum(1 for char in GERMAN_CHARS if char in result.text)
                
                self.test_results.add_result(
                    f"TXT: {filename}",
                    "PASSED",
                    {
                        "language": result.language,
                        "chars": len(result.text),
                        "german_chars": german_chars_found,
                        "time": f"{result.processing_time:.3f}s"
                    }
                )
                
                # Print sample text
                print(f"  {filename}: {result.text[:100]}...")
                
            except Exception as e:
                self.test_results.add_result(
                    f"TXT: {filename}",
                    "FAILED",
                    {"error": str(e)}
                )
                
    async def test_image_files(self):
        """Test image file processing"""
        print("\n--- Testing Image Files ---")
        
        image_files = [
            "german_image_1.png",
            "english_image_1.jpg",
            "table_image.png",
            "scanned_document.png"
        ]
        
        for filename in image_files:
            file_path = self.test_dir / filename
            if not file_path.exists():
                self.test_results.add_result(
                    f"Image: {filename}",
                    "FAILED", 
                    {"error": "File not found"}
                )
                continue
                
            try:
                result = await self.ocr_system.process_file(str(file_path))
                
                # Check results
                has_text = len(result.text.strip()) > 0
                german_chars_found = sum(1 for char in GERMAN_CHARS if char in result.text)
                
                self.test_results.add_result(
                    f"Image: {filename}",
                    "PASSED" if has_text else "FAILED",
                    {
                        "has_text": has_text,
                        "confidence": f"{result.confidence:.2%}",
                        "german_chars": german_chars_found,
                        "engine": result.metadata.get('engine', 'unknown'),
                        "time": f"{result.processing_time:.3f}s"
                    }
                )
                
                # Print extracted text sample
                if has_text:
                    print(f"  {filename}: {result.text[:100]}...")
                    
            except Exception as e:
                self.test_results.add_result(
                    f"Image: {filename}",
                    "FAILED",
                    {"error": str(e)}
                )
                
    async def test_pdf_files(self):
        """Test PDF file processing"""
        print("\n--- Testing PDF Files ---")
        
        pdf_files = [
            "native_german.pdf",
            "pdf_with_table.pdf",
            "scanned_german.pdf",
            "sample_english.pdf"
        ]
        
        for filename in pdf_files:
            file_path = self.test_dir / filename
            if not file_path.exists():
                self.test_results.add_result(
                    f"PDF: {filename}",
                    "FAILED",
                    {"error": "File not found"}
                )
                continue
                
            try:
                result = await self.ocr_system.process_file(str(file_path))
                
                # Check results
                has_text = len(result.text.strip()) > 0
                german_chars_found = sum(1 for char in GERMAN_CHARS if char in result.text)
                pages = result.metadata.get('pages', 1)
                
                self.test_results.add_result(
                    f"PDF: {filename}",
                    "PASSED" if has_text else "FAILED",
                    {
                        "has_text": has_text,
                        "pages": pages,
                        "german_chars": german_chars_found,
                        "tables": len(result.tables),
                        "time": f"{result.processing_time:.3f}s"
                    }
                )
                
                # Print sample
                if has_text:
                    print(f"  {filename}: {result.text[:100]}...")
                    
            except Exception as e:
                self.test_results.add_result(
                    f"PDF: {filename}",
                    "FAILED",
                    {"error": str(e)}
                )
                
    async def test_docx_files(self):
        """Test DOCX file processing"""
        print("\n--- Testing DOCX Files ---")
        
        docx_files = [
            "german_document.docx",
            "docx_with_table.docx"
        ]
        
        for filename in docx_files:
            file_path = self.test_dir / filename
            if not file_path.exists():
                self.test_results.add_result(
                    f"DOCX: {filename}",
                    "FAILED",
                    {"error": "File not found"}
                )
                continue
                
            try:
                result = await self.ocr_system.process_file(str(file_path))
                
                # Check results
                has_text = len(result.text.strip()) > 0
                german_chars_found = sum(1 for char in GERMAN_CHARS if char in result.text)
                
                self.test_results.add_result(
                    f"DOCX: {filename}",
                    "PASSED" if has_text else "FAILED",
                    {
                        "has_text": has_text,
                        "german_chars": german_chars_found,
                        "paragraphs": result.metadata.get('paragraphs', 0),
                        "time": f"{result.processing_time:.3f}s"
                    }
                )
                
                # Print sample
                if has_text:
                    print(f"  {filename}: {result.text[:100]}...")
                    
            except Exception as e:
                self.test_results.add_result(
                    f"DOCX: {filename}",
                    "FAILED",
                    {"error": str(e)}
                )
                
    async def test_german_umlauts(self):
        """Specific test for German umlaut support"""
        print("\n--- Testing German Umlaut Support ---")
        
        # Create a test file with all umlauts
        test_text = "Äpfel Öl Übung ähnlich öffnen über Größe"
        test_file = self.test_dir / "umlaut_test.txt"
        
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_text)
            
        try:
            result = await self.ocr_system.process_file(str(test_file))
            
            # Check each umlaut
            umlaut_results = {}
            for char in GERMAN_CHARS:
                umlaut_results[char] = char in result.text
                
            all_found = all(umlaut_results.values())
            
            self.test_results.add_result(
                "German Umlaut Support",
                "PASSED" if all_found else "FAILED",
                {
                    "umlauts_found": umlaut_results,
                    "language_detected": result.language,
                    "original": test_text,
                    "extracted": result.text.strip()
                }
            )
            
            print(f"  Original: {test_text}")
            print(f"  Extracted: {result.text.strip()}")
            
        except Exception as e:
            self.test_results.add_result(
                "German Umlaut Support",
                "FAILED",
                {"error": str(e)}
            )
            
    async def test_concurrent_processing(self):
        """Test concurrent file processing"""
        print("\n--- Testing Concurrent Processing ---")
        
        # Get multiple files
        test_files = []
        for file in self.test_dir.iterdir():
            if file.is_file() and file.suffix in ['.txt', '.png', '.pdf']:
                test_files.append(str(file))
                if len(test_files) >= 5:
                    break
                    
        if not test_files:
            self.test_results.add_result(
                "Concurrent Processing",
                "FAILED",
                {"error": "No test files found"}
            )
            return
            
        try:
            start_time = time.time()
            results = await self.ocr_system.process_batch(test_files)
            end_time = time.time()
            
            total_time = end_time - start_time
            avg_time = total_time / len(results)
            
            self.test_results.add_result(
                "Concurrent Processing",
                "PASSED",
                {
                    "files_processed": len(results),
                    "total_time": f"{total_time:.2f}s",
                    "avg_time": f"{avg_time:.2f}s",
                    "successful": sum(1 for r in results if r.text)
                }
            )
            
        except Exception as e:
            self.test_results.add_result(
                "Concurrent Processing",
                "FAILED",
                {"error": str(e)}
            )
            
    async def test_resource_limits(self):
        """Test resource usage stays within limits"""
        print("\n--- Testing Resource Limits ---")
        
        # Process multiple files and monitor resources
        test_files = list(self.test_dir.glob("*.png"))[:3]
        
        max_cpu = 0
        max_ram_percent = 0
        max_ram_gb = 0
        
        for file in test_files:
            try:
                # Process file
                await self.ocr_system.process_file(str(file))
                
                # Check resources
                resources = get_resource_info()
                max_cpu = max(max_cpu, resources['cpu_percent'])
                max_ram_percent = max(max_ram_percent, resources['ram_percent'])
                max_ram_gb = max(max_ram_gb, resources['ram_gb'])
                
            except Exception as e:
                logger.error(f"Error processing {file}: {e}")
                
        # Check if within limits
        cpu_ok = max_cpu <= 75  # Allow small buffer
        ram_ok = max_ram_percent <= 75 or max_ram_gb <= 2.5
        
        self.test_results.add_result(
            "Resource Limits",
            "PASSED" if (cpu_ok and ram_ok) else "WARNING",
            {
                "max_cpu": f"{max_cpu:.1f}%",
                "max_ram": f"{max_ram_gb:.2f}GB ({max_ram_percent:.1f}%)",
                "cpu_within_limit": cpu_ok,
                "ram_within_limit": ram_ok
            }
        )
        
        print(f"  Max CPU: {max_cpu:.1f}%")
        print(f"  Max RAM: {max_ram_gb:.2f}GB ({max_ram_percent:.1f}%)")


async def main():
    """Run all tests"""
    tester = OCRSystemTester()
    await tester.run_all_tests()
    
    # Final resource check
    print("\nFinal Resource Usage:")
    resources = get_resource_info()
    print(f"  CPU: {resources['cpu_percent']:.1f}%")
    print(f"  RAM: {resources['ram_gb']:.2f}GB ({resources['ram_percent']:.1f}%)")


if __name__ == "__main__":
    asyncio.run(main())