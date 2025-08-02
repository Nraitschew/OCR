#!/usr/bin/env python3
"""
CPU-Only OCR System Test Suite
Tests the system with CPU-only configuration
"""

import asyncio
import time
import sys
from pathlib import Path
from typing import List, Dict, Any
import multiprocessing as mp

# Import CPU-only system
sys.path.append(str(Path(__file__).parent))
from cpu_only_ocr_system import CPUOnlyOCRSystem, CPUOptimizedConfig, get_resource_info

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
CYAN = '\033[96m'
RESET = '\033[0m'
BOLD = '\033[1m'

# German characters to test
GERMAN_CHARS = ['ä', 'ö', 'ü', 'Ä', 'Ö', 'Ü', 'ß']


class CPUOnlyTester:
    """Test suite for CPU-only OCR system"""
    
    def __init__(self):
        # Create CPU-optimized configuration
        self.config = CPUOptimizedConfig(
            max_workers=min(4, mp.cpu_count() - 1),
            tesseract_threads=4,
            batch_size=3,
            enable_preprocessing=True,
            dpi_for_pdf=150  # Lower DPI for faster CPU processing
        )
        
        self.ocr_system = CPUOnlyOCRSystem(self.config)
        self.test_dir = Path("test_documents")
        self.enhanced_test_dir = Path("test_documents_enhanced")
        self.results = []
        
    def print_banner(self):
        """Print test banner"""
        print(f"\n{BLUE}{'='*70}{RESET}")
        print(f"{BLUE}{BOLD}     CPU-ONLY OCR SYSTEM TEST{RESET}")
        print(f"{BLUE}{'='*70}{RESET}")
        print(f"{CYAN}Configuration:{RESET}")
        print(f"  - CPU cores: {mp.cpu_count()} available, using {self.config.max_workers}")
        print(f"  - Tesseract threads: {self.config.tesseract_threads}")
        print(f"  - Image preprocessing: {'Enabled' if self.config.enable_preprocessing else 'Disabled'}")
        print(f"  - PDF DPI: {self.config.dpi_for_pdf}")
        print(f"  - NO GPU DEPENDENCIES")
        print(f"{BLUE}{'='*70}{RESET}\n")
        
    def print_section(self, title: str):
        """Print section header"""
        print(f"\n{CYAN}{'─'*50}{RESET}")
        print(f"{CYAN}{title}{RESET}")
        print(f"{CYAN}{'─'*50}{RESET}")
        
    async def run_all_tests(self):
        """Run all CPU-only tests"""
        self.print_banner()
        
        # Show initial resources
        self.print_section("Initial System Resources")
        resources = get_resource_info()
        print(f"CPU Usage: {resources['cpu_percent']:.1f}%")
        print(f"RAM Usage: {resources['ram_gb']:.2f}GB ({resources['ram_percent']:.1f}%)")
        print(f"Threads: {resources['threads']}")
        
        # Ensure test documents exist
        if not self.test_dir.exists():
            print(f"{RED}Error: Test documents not found. Please run setup first.{RESET}")
            return
            
        # Run test categories
        await self.test_basic_functionality()
        await self.test_german_support()
        await self.test_table_extraction()
        await self.test_cpu_performance()
        await self.test_resource_limits()
        await self.test_batch_processing()
        
        # Print summary
        self.print_summary()
        
    async def test_basic_functionality(self):
        """Test basic OCR functionality with CPU"""
        self.print_section("Basic Functionality Tests")
        
        # Test text file
        print(f"\n{BOLD}Testing TXT file...{RESET}")
        txt_file = self.test_dir / "german_text.txt"
        
        try:
            start = time.time()
            result = await self.ocr_system.process_file(str(txt_file))
            elapsed = time.time() - start
            
            success = len(result.text) > 0
            status = f"{GREEN}✓ PASSED{RESET}" if success else f"{RED}✗ FAILED{RESET}"
            
            print(f"{status} Text file processing ({elapsed:.2f}s)")
            print(f"  - Characters: {len(result.text)}")
            print(f"  - Language: {result.language}")
            
            self.results.append(('TXT Processing', success))
            
        except Exception as e:
            print(f"{RED}✗ FAILED{RESET} Text file processing: {e}")
            self.results.append(('TXT Processing', False))
            
        # Test image file
        print(f"\n{BOLD}Testing Image OCR...{RESET}")
        img_file = self.test_dir / "german_image_1.png"
        
        try:
            start = time.time()
            result = await self.ocr_system.process_file(str(img_file))
            elapsed = time.time() - start
            
            success = len(result.text) > 0
            status = f"{GREEN}✓ PASSED{RESET}" if success else f"{RED}✗ FAILED{RESET}"
            
            print(f"{status} Image OCR processing ({elapsed:.2f}s)")
            print(f"  - Confidence: {result.confidence:.1%}")
            print(f"  - CPU usage: {result.cpu_usage:.1f}%")
            print(f"  - RAM usage: {result.ram_usage_mb:.1f}MB")
            
            self.results.append(('Image OCR', success))
            
        except Exception as e:
            print(f"{RED}✗ FAILED{RESET} Image OCR: {e}")
            self.results.append(('Image OCR', False))
            
    async def test_german_support(self):
        """Test German character support"""
        self.print_section("German Language Support")
        
        # Create test file
        test_content = "Äpfel Öl Übung ähnlich öffnen über Größe"
        test_file = self.test_dir / "cpu_german_test.txt"
        
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
            
        try:
            result = await self.ocr_system.process_file(str(test_file))
            
            # Check each character
            found_chars = []
            missing_chars = []
            
            for char in GERMAN_CHARS:
                if char in result.text:
                    found_chars.append(char)
                    print(f"{GREEN}✓{RESET} Found character: {char}")
                else:
                    missing_chars.append(char)
                    print(f"{RED}✗{RESET} Missing character: {char}")
                    
            success = len(missing_chars) == 0
            self.results.append(('German Characters', success))
            
            print(f"\nLanguage detected: {result.language}")
            
        except Exception as e:
            print(f"{RED}✗ FAILED{RESET} German support test: {e}")
            self.results.append(('German Characters', False))
            
    async def test_table_extraction(self):
        """Test table extraction with CPU"""
        self.print_section("Table Extraction Tests")
        
        # Test PDF with tables
        if self.enhanced_test_dir.exists():
            table_file = self.enhanced_test_dir / "complex_table_multicolumn.pdf"
        else:
            table_file = self.test_dir / "pdf_with_table.pdf"
            
        if table_file.exists():
            try:
                start = time.time()
                result = await self.ocr_system.process_file(str(table_file))
                elapsed = time.time() - start
                
                tables_found = len(result.tables)
                success = tables_found > 0
                status = f"{GREEN}✓ PASSED{RESET}" if success else f"{YELLOW}⚠ WARNING{RESET}"
                
                print(f"{status} Table extraction ({elapsed:.2f}s)")
                print(f"  - Tables found: {tables_found}")
                
                if tables_found > 0:
                    first_table = result.tables[0]
                    print(f"  - First table: {len(first_table.rows)} rows")
                    
                self.results.append(('Table Extraction', success))
                
            except Exception as e:
                print(f"{RED}✗ FAILED{RESET} Table extraction: {e}")
                self.results.append(('Table Extraction', False))
        else:
            print(f"{YELLOW}⚠ Skipping table test - file not found{RESET}")
            
    async def test_cpu_performance(self):
        """Test CPU performance characteristics"""
        self.print_section("CPU Performance Tests")
        
        # Test different file types and measure performance
        test_files = {
            'TXT': self.test_dir / "english_text.txt",
            'PNG': self.test_dir / "english_image_1.jpg",
            'PDF': self.test_dir / "native_german.pdf"
        }
        
        for file_type, file_path in test_files.items():
            if not file_path.exists():
                continue
                
            try:
                # Measure processing time and resources
                initial_resources = get_resource_info()
                
                start = time.time()
                result = await self.ocr_system.process_file(str(file_path))
                elapsed = time.time() - start
                
                final_resources = get_resource_info()
                
                print(f"\n{file_type} Performance:")
                print(f"  - Processing time: {elapsed:.2f}s")
                print(f"  - CPU usage during: {result.cpu_usage:.1f}%")
                print(f"  - RAM usage: {result.ram_usage_mb:.1f}MB")
                print(f"  - Confidence: {result.confidence:.1%}")
                
                # Performance is good if under reasonable time
                success = elapsed < 10.0  # 10 seconds max for CPU
                self.results.append((f'{file_type} Performance', success))
                
            except Exception as e:
                print(f"{RED}✗ Error testing {file_type}: {e}{RESET}")
                self.results.append((f'{file_type} Performance', False))
                
    async def test_resource_limits(self):
        """Test that resource limits are respected"""
        self.print_section("Resource Limit Tests")
        
        max_cpu_observed = 0
        max_ram_gb_observed = 0
        
        # Process several files and monitor resources
        test_files = list(self.test_dir.glob("*.png"))[:3]
        
        for file_path in test_files:
            try:
                result = await self.ocr_system.process_file(str(file_path))
                
                resources = get_resource_info()
                max_cpu_observed = max(max_cpu_observed, resources['cpu_percent'])
                max_ram_gb_observed = max(max_ram_gb_observed, resources['ram_gb'])
                
                # Small delay between files
                await asyncio.sleep(0.5)
                
            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")
                
        print(f"\nResource Usage Summary:")
        print(f"  - Max CPU observed: {max_cpu_observed:.1f}% (limit: 70%)")
        print(f"  - Max RAM observed: {max_ram_gb_observed:.2f}GB (limit: 2GB)")
        
        cpu_ok = max_cpu_observed <= 75  # Allow 5% buffer
        ram_ok = max_ram_gb_observed <= 2.5
        
        if cpu_ok:
            print(f"{GREEN}✓ CPU usage within limits{RESET}")
        else:
            print(f"{YELLOW}⚠ CPU usage exceeded soft limit{RESET}")
            
        if ram_ok:
            print(f"{GREEN}✓ RAM usage within limits{RESET}")
        else:
            print(f"{YELLOW}⚠ RAM usage exceeded soft limit{RESET}")
            
        self.results.append(('Resource Limits', cpu_ok and ram_ok))
        
    async def test_batch_processing(self):
        """Test batch processing efficiency"""
        self.print_section("Batch Processing Test")
        
        # Get multiple files
        batch_files = []
        for ext in ['*.txt', '*.png']:
            batch_files.extend(list(self.test_dir.glob(ext))[:2])
            
        if len(batch_files) < 2:
            print(f"{YELLOW}⚠ Not enough files for batch test{RESET}")
            return
            
        try:
            print(f"Processing batch of {len(batch_files)} files...")
            
            start = time.time()
            results = await self.ocr_system.process_batch([str(f) for f in batch_files])
            elapsed = time.time() - start
            
            successful = sum(1 for r in results if r.text)
            avg_time = elapsed / len(batch_files)
            
            print(f"\nBatch Results:")
            print(f"  - Files processed: {len(results)}")
            print(f"  - Successful: {successful}")
            print(f"  - Total time: {elapsed:.2f}s")
            print(f"  - Average per file: {avg_time:.2f}s")
            
            success = successful == len(batch_files)
            self.results.append(('Batch Processing', success))
            
        except Exception as e:
            print(f"{RED}✗ Batch processing failed: {e}{RESET}")
            self.results.append(('Batch Processing', False))
            
    def print_summary(self):
        """Print test summary"""
        self.print_section("TEST SUMMARY")
        
        total_tests = len(self.results)
        passed_tests = sum(1 for _, passed in self.results if passed)
        failed_tests = total_tests - passed_tests
        
        print(f"\nTotal Tests: {total_tests}")
        print(f"{GREEN}Passed: {passed_tests}{RESET}")
        print(f"{RED}Failed: {failed_tests}{RESET}")
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        if success_rate >= 80:
            color = GREEN
        elif success_rate >= 60:
            color = YELLOW
        else:
            color = RED
            
        print(f"\n{color}{BOLD}Success Rate: {success_rate:.1f}%{RESET}")
        
        # Final resource check
        final_resources = get_resource_info()
        print(f"\nFinal System Resources:")
        print(f"  - CPU: {final_resources['cpu_percent']:.1f}%")
        print(f"  - RAM: {final_resources['ram_gb']:.2f}GB")
        print(f"  - Threads: {final_resources['threads']}")
        
        print(f"\n{BLUE}CPU-Only OCR System Test Complete!{RESET}")


async def main():
    """Run CPU-only tests"""
    tester = CPUOnlyTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Test interrupted by user{RESET}")
    except Exception as e:
        print(f"\n{RED}Fatal error: {e}{RESET}")
        raise