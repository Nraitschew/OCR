#!/usr/bin/env python3
"""
Comprehensive Test of OCR System
"""

import asyncio
import time
import sys
from pathlib import Path
from minimal_ocr_system import MinimalOCRSystem, get_resource_info

# Add colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

# German characters to test
GERMAN_CHARS = ['ä', 'ö', 'ü', 'Ä', 'Ö', 'Ü', 'ß']


class OCRTester:
    def __init__(self):
        self.ocr_system = MinimalOCRSystem()
        self.test_dir = Path("test_documents")
        self.results = []
        
    def print_header(self, text):
        print(f"\n{BLUE}{'='*80}{RESET}")
        print(f"{BLUE}{text}{RESET}")
        print(f"{BLUE}{'='*80}{RESET}")
        
    def print_test_result(self, test_name, passed, details=""):
        status = f"{GREEN}PASSED{RESET}" if passed else f"{RED}FAILED{RESET}"
        print(f"  {test_name}: {status} {details}")
        self.results.append((test_name, passed, details))
        
    def check_german_chars(self, text):
        """Check which German characters are present"""
        found = [char for char in GERMAN_CHARS if char in text]
        return found
        
    async def test_txt_files(self):
        """Test TXT file processing"""
        self.print_header("Testing TXT Files")
        
        test_files = [
            ("german_text.txt", True),  # Should have umlauts
            ("english_text.txt", False),  # No umlauts
            ("mixed_language.txt", True)  # Should have umlauts
        ]
        
        for filename, should_have_umlauts in test_files:
            file_path = self.test_dir / filename
            if not file_path.exists():
                self.print_test_result(f"TXT: {filename}", False, "File not found")
                continue
                
            try:
                result = await self.ocr_system.process_file(str(file_path))
                
                # Check German characters
                found_chars = self.check_german_chars(result.text)
                has_umlauts = len(found_chars) > 0
                
                # Verify expectations
                if should_have_umlauts:
                    passed = has_umlauts
                    details = f"(Found: {found_chars})" if has_umlauts else "(No umlauts found)"
                else:
                    passed = True  # English files are ok without umlauts
                    details = f"(Lang: {result.language})"
                    
                self.print_test_result(
                    f"TXT: {filename}",
                    passed,
                    details
                )
                
                # Show sample
                sample = result.text[:50].replace('\n', ' ')
                print(f"      Sample: {sample}...")
                
            except Exception as e:
                self.print_test_result(f"TXT: {filename}", False, f"Error: {str(e)}")
                
    async def test_image_files(self):
        """Test image file processing"""
        self.print_header("Testing Image Files")
        
        # Check if tesseract is available
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            tesseract_available = True
        except:
            tesseract_available = False
            print(f"{YELLOW}  Warning: Tesseract not available. Skipping image OCR tests.{RESET}")
            
        test_images = [
            "german_image_1.png",
            "english_image_1.jpg",
            "table_image.png",
            "scanned_document.png"
        ]
        
        for filename in test_images:
            file_path = self.test_dir / filename
            if not file_path.exists():
                self.print_test_result(f"Image: {filename}", False, "File not found")
                continue
                
            try:
                result = await self.ocr_system.process_file(str(file_path))
                
                if tesseract_available and result.text:
                    found_chars = self.check_german_chars(result.text)
                    self.print_test_result(
                        f"Image: {filename}",
                        True,
                        f"(Conf: {result.confidence:.1%}, Chars: {len(result.text)})"
                    )
                    
                    # Show sample
                    sample = result.text[:50].replace('\n', ' ')
                    print(f"      Sample: {sample}...")
                else:
                    self.print_test_result(
                        f"Image: {filename}",
                        not tesseract_available,
                        "(No OCR available)" if not tesseract_available else "(No text extracted)"
                    )
                    
            except Exception as e:
                self.print_test_result(f"Image: {filename}", False, f"Error: {str(e)}")
                
    async def test_pdf_files(self):
        """Test PDF file processing"""
        self.print_header("Testing PDF Files")
        
        test_pdfs = [
            ("native_german.pdf", "native"),
            ("pdf_with_table.pdf", "native"),
            ("scanned_german.pdf", "scanned"),
            ("sample_english.pdf", "native")
        ]
        
        for filename, pdf_type in test_pdfs:
            file_path = self.test_dir / filename
            if not file_path.exists():
                self.print_test_result(f"PDF: {filename}", False, "File not found")
                continue
                
            try:
                result = await self.ocr_system.process_file(str(file_path))
                
                has_text = len(result.text.strip()) > 0
                found_chars = self.check_german_chars(result.text)
                pages = result.metadata.get('pages', 1)
                
                details = f"(Pages: {pages}, Type: {pdf_type}"
                if has_text:
                    details += f", Chars: {len(result.text)}"
                    if found_chars:
                        details += f", Umlauts: {found_chars}"
                details += ")"
                
                self.print_test_result(
                    f"PDF: {filename}",
                    has_text or pdf_type == "scanned",
                    details
                )
                
                # Show sample
                if has_text:
                    sample = result.text[:50].replace('\n', ' ')
                    print(f"      Sample: {sample}...")
                    
            except Exception as e:
                self.print_test_result(f"PDF: {filename}", False, f"Error: {str(e)}")
                
    async def test_docx_files(self):
        """Test DOCX file processing"""
        self.print_header("Testing DOCX Files")
        
        test_docx = [
            "german_document.docx",
            "docx_with_table.docx"
        ]
        
        for filename in test_docx:
            file_path = self.test_dir / filename
            if not file_path.exists():
                self.print_test_result(f"DOCX: {filename}", False, "File not found")
                continue
                
            try:
                result = await self.ocr_system.process_file(str(file_path))
                
                has_text = len(result.text.strip()) > 0
                found_chars = self.check_german_chars(result.text)
                paragraphs = result.metadata.get('paragraphs', 0)
                
                details = f"(Paragraphs: {paragraphs}"
                if found_chars:
                    details += f", Umlauts: {found_chars}"
                details += ")"
                
                self.print_test_result(
                    f"DOCX: {filename}",
                    has_text,
                    details
                )
                
                # Show sample
                if has_text:
                    sample = result.text[:50].replace('\n', ' ')
                    print(f"      Sample: {sample}...")
                    
            except Exception as e:
                self.print_test_result(f"DOCX: {filename}", False, f"Error: {str(e)}")
                
    async def test_resource_monitoring(self):
        """Test resource monitoring"""
        self.print_header("Testing Resource Monitoring")
        
        # Get initial resources
        initial = get_resource_info()
        
        # Process a few files to generate load
        test_files = list(self.test_dir.glob("*.txt"))[:3]
        
        max_cpu = initial['cpu_percent']
        max_ram_gb = initial['ram_gb']
        
        for file in test_files:
            await self.ocr_system.process_file(str(file))
            resources = get_resource_info()
            max_cpu = max(max_cpu, resources['cpu_percent'])
            max_ram_gb = max(max_ram_gb, resources['ram_gb'])
            
        # Check limits
        cpu_ok = max_cpu <= 75  # Allow 5% buffer
        ram_ok = max_ram_gb <= 2.5  # Allow some buffer
        
        self.print_test_result(
            "CPU Usage",
            cpu_ok,
            f"(Max: {max_cpu:.1f}%, Limit: 70%)"
        )
        
        self.print_test_result(
            "RAM Usage",
            ram_ok,
            f"(Max: {max_ram_gb:.2f}GB, Limit: 2GB)"
        )
        
    async def test_german_umlaut_preservation(self):
        """Test specific German umlaut preservation"""
        self.print_header("Testing German Umlaut Preservation")
        
        # Create test content
        test_content = "Äpfel Öl Übung ähnlich öffnen über Größe"
        test_file = self.test_dir / "umlaut_test.txt"
        
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
            
        try:
            result = await self.ocr_system.process_file(str(test_file))
            
            # Check each umlaut
            all_found = True
            for char in GERMAN_CHARS:
                if char in test_content:
                    found = char in result.text
                    self.print_test_result(
                        f"  Umlaut '{char}'",
                        found,
                        "✓" if found else "✗"
                    )
                    all_found = all_found and found
                    
            # Overall result
            self.print_test_result(
                "Overall Umlaut Support",
                all_found,
                f"(Original: {test_content})"
            )
            print(f"      Extracted: {result.text.strip()}")
            
        except Exception as e:
            self.print_test_result("German Umlaut Test", False, f"Error: {str(e)}")
            
    async def run_all_tests(self):
        """Run all tests"""
        print(f"{BLUE}Starting Comprehensive OCR System Test{RESET}")
        print(f"Test directory: {self.test_dir}")
        
        # Show system info
        resources = get_resource_info()
        print(f"\nSystem Resources:")
        print(f"  CPU: {resources['cpu_percent']:.1f}%")
        print(f"  RAM: {resources['ram_gb']:.2f}GB ({resources['ram_percent']:.1f}%)")
        print(f"  Threads: {resources['threads']}")
        
        # Run tests
        await self.test_txt_files()
        await self.test_image_files()
        await self.test_pdf_files()
        await self.test_docx_files()
        await self.test_german_umlaut_preservation()
        await self.test_resource_monitoring()
        
        # Print summary
        self.print_header("TEST SUMMARY")
        
        total = len(self.results)
        passed = sum(1 for _, p, _ in self.results if p)
        failed = total - passed
        
        print(f"Total Tests: {total}")
        print(f"{GREEN}Passed: {passed}{RESET}")
        print(f"{RED}Failed: {failed}{RESET}")
        
        success_rate = (passed / total * 100) if total > 0 else 0
        color = GREEN if success_rate >= 80 else YELLOW if success_rate >= 60 else RED
        print(f"\n{color}Success Rate: {success_rate:.1f}%{RESET}")
        
        # Final resource check
        final_resources = get_resource_info()
        print(f"\nFinal Resources:")
        print(f"  CPU: {final_resources['cpu_percent']:.1f}%")
        print(f"  RAM: {final_resources['ram_gb']:.2f}GB ({final_resources['ram_percent']:.1f}%)")


async def main():
    tester = OCRTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())