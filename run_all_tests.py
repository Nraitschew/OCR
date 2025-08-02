#!/usr/bin/env python3
"""
Comprehensive Automated Test Runner for Enhanced OCR System
"""

import asyncio
import time
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json
from datetime import datetime

# Import both OCR systems
sys.path.append(str(Path(__file__).parent))
from enhanced_ocr_system import EnhancedOCRSystem, get_resource_info
from enhanced_test_generator import EnhancedTestDocumentGenerator

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
CYAN = '\033[96m'
MAGENTA = '\033[95m'
RESET = '\033[0m'
BOLD = '\033[1m'

# German characters to test
GERMAN_CHARS = ['ä', 'ö', 'ü', 'Ä', 'Ö', 'Ü', 'ß']


class DetailedTestResult:
    """Stores detailed test results"""
    
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.details = {}
        self.errors = []
        self.warnings = []
        self.execution_time = 0.0
        self.resource_usage = {}
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'passed': self.passed,
            'details': self.details,
            'errors': self.errors,
            'warnings': self.warnings,
            'execution_time': self.execution_time,
            'resource_usage': self.resource_usage
        }


class ComprehensiveOCRTester:
    """Comprehensive automated OCR tester"""
    
    def __init__(self):
        self.ocr_system = EnhancedOCRSystem()
        self.basic_test_dir = Path("test_documents")
        self.enhanced_test_dir = Path("test_documents_enhanced")
        self.results = []
        self.start_time = None
        
    def print_banner(self):
        """Print test banner"""
        print(f"\n{BLUE}{'='*80}{RESET}")
        print(f"{BLUE}{BOLD}     COMPREHENSIVE OCR SYSTEM TEST SUITE{RESET}")
        print(f"{BLUE}{'='*80}{RESET}")
        print(f"{CYAN}Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{RESET}")
        print(f"{CYAN}System: Enhanced OCR with Advanced Table Extraction{RESET}")
        print(f"{BLUE}{'='*80}{RESET}\n")
        
    def print_section(self, title: str):
        """Print section header"""
        print(f"\n{MAGENTA}{'─'*60}{RESET}")
        print(f"{MAGENTA}{BOLD}{title}{RESET}")
        print(f"{MAGENTA}{'─'*60}{RESET}\n")
        
    def print_test_status(self, test_name: str, result: DetailedTestResult):
        """Print test status with details"""
        status_icon = f"{GREEN}✓{RESET}" if result.passed else f"{RED}✗{RESET}"
        status_text = f"{GREEN}PASSED{RESET}" if result.passed else f"{RED}FAILED{RESET}"
        
        print(f"{status_icon} {test_name}: {status_text} ({result.execution_time:.2f}s)")
        
        # Print key details
        if result.details:
            for key, value in result.details.items():
                print(f"    {CYAN}{key}:{RESET} {value}")
                
        # Print warnings
        for warning in result.warnings:
            print(f"    {YELLOW}⚠ {warning}{RESET}")
            
        # Print errors
        for error in result.errors:
            print(f"    {RED}✗ {error}{RESET}")
            
    async def run_all_tests(self):
        """Run all comprehensive tests"""
        self.start_time = time.time()
        self.print_banner()
        
        # Show initial system resources
        self.print_section("System Information")
        resources = get_resource_info()
        print(f"CPU Usage: {resources['cpu_percent']:.1f}%")
        print(f"RAM Usage: {resources['ram_gb']:.2f}GB ({resources['ram_percent']:.1f}%)")
        print(f"Threads: {resources['threads']}")
        
        # Ensure test documents exist
        await self.ensure_test_documents()
        
        # Run test categories
        await self.test_basic_functionality()
        await self.test_table_extraction()
        await self.test_complex_documents()
        await self.test_german_language_support()
        await self.test_performance_and_resources()
        await self.test_edge_cases()
        
        # Print summary
        self.print_summary()
        
    async def ensure_test_documents(self):
        """Ensure all test documents are generated"""
        self.print_section("Test Document Generation")
        
        # Generate basic documents if needed
        if not self.basic_test_dir.exists() or len(list(self.basic_test_dir.iterdir())) < 10:
            print("Generating basic test documents...")
            import test_document_generator
            generator = test_document_generator.TestDocumentGenerator()
            generator.generate_all()
            print(f"{GREEN}✓ Basic test documents generated{RESET}")
        else:
            print(f"{GREEN}✓ Basic test documents already exist{RESET}")
            
        # Generate enhanced documents
        if not self.enhanced_test_dir.exists() or len(list(self.enhanced_test_dir.iterdir())) < 5:
            print("Generating enhanced test documents...")
            generator = EnhancedTestDocumentGenerator()
            generator.generate_all()
            print(f"{GREEN}✓ Enhanced test documents generated{RESET}")
        else:
            print(f"{GREEN}✓ Enhanced test documents already exist{RESET}")
            
    async def test_basic_functionality(self):
        """Test basic OCR functionality"""
        self.print_section("Basic Functionality Tests")
        
        # Test TXT files
        result = DetailedTestResult("Text File Processing")
        start = time.time()
        
        try:
            txt_file = self.basic_test_dir / "german_text.txt"
            ocr_result = await self.ocr_system.process_file(str(txt_file))
            
            result.passed = len(ocr_result.text) > 0
            result.details = {
                'characters': len(ocr_result.text),
                'language': ocr_result.language,
                'tables_found': len(ocr_result.tables)
            }
            
            # Check for German characters
            found_chars = [char for char in GERMAN_CHARS if char in ocr_result.text]
            if found_chars:
                result.details['german_chars'] = ', '.join(found_chars)
            else:
                result.warnings.append("No German characters found")
                
        except Exception as e:
            result.passed = False
            result.errors.append(str(e))
            
        result.execution_time = time.time() - start
        result.resource_usage = get_resource_info()
        self.results.append(result)
        self.print_test_status("Text File Processing", result)
        
        # Test image OCR
        result = DetailedTestResult("Image OCR Processing")
        start = time.time()
        
        try:
            img_file = self.basic_test_dir / "german_image_1.png"
            ocr_result = await self.ocr_system.process_file(str(img_file))
            
            result.passed = len(ocr_result.text) > 0
            result.details = {
                'confidence': f"{ocr_result.confidence:.1%}",
                'characters': len(ocr_result.text),
                'processing_time': f"{ocr_result.processing_time:.3f}s"
            }
            
        except Exception as e:
            result.passed = False
            result.errors.append(str(e))
            
        result.execution_time = time.time() - start
        self.results.append(result)
        self.print_test_status("Image OCR Processing", result)
        
    async def test_table_extraction(self):
        """Test advanced table extraction"""
        self.print_section("Table Extraction Tests")
        
        # Test complex multi-column table
        result = DetailedTestResult("Multi-Column Table Extraction")
        start = time.time()
        
        try:
            table_file = self.enhanced_test_dir / "complex_table_multicolumn.pdf"
            ocr_result = await self.ocr_system.process_file(str(table_file))
            
            result.passed = len(ocr_result.tables) > 0
            result.details = {
                'tables_found': len(ocr_result.tables),
                'total_rows': sum(len(t.rows) for t in ocr_result.tables),
                'confidence': f"{ocr_result.confidence:.1%}"
            }
            
            # Check table content
            if ocr_result.tables:
                first_table = ocr_result.tables[0]
                result.details['first_table_size'] = f"{len(first_table.rows)}×{len(first_table.rows[0]) if first_table.rows else 0}"
                
                # Check for German content in tables
                table_text = ' '.join([' '.join(row) for table in ocr_result.tables for row in table.rows])
                if any(char in table_text for char in GERMAN_CHARS):
                    result.details['german_in_tables'] = "Yes"
                    
        except Exception as e:
            result.passed = False
            result.errors.append(str(e))
            
        result.execution_time = time.time() - start
        self.results.append(result)
        self.print_test_status("Multi-Column Table Extraction", result)
        
        # Test invoice table extraction
        result = DetailedTestResult("Invoice Table Extraction")
        start = time.time()
        
        try:
            invoice_file = self.enhanced_test_dir / "invoice_complex.pdf"
            ocr_result = await self.ocr_system.process_file(str(invoice_file))
            
            result.passed = len(ocr_result.tables) > 0
            result.details = {
                'tables_found': len(ocr_result.tables),
                'invoice_items': "Found" if any(t for t in ocr_result.tables if len(t.rows) > 3) else "Not found"
            }
            
            # Check for currency symbols
            if '€' in ocr_result.text:
                result.details['currency_symbols'] = "€ found"
                
        except Exception as e:
            result.passed = False
            result.errors.append(str(e))
            
        result.execution_time = time.time() - start
        self.results.append(result)
        self.print_test_status("Invoice Table Extraction", result)
        
    async def test_complex_documents(self):
        """Test complex document structures"""
        self.print_section("Complex Document Tests")
        
        # Test scientific document
        result = DetailedTestResult("Scientific Document Processing")
        start = time.time()
        
        try:
            sci_file = self.enhanced_test_dir / "scientific_document.pdf"
            ocr_result = await self.ocr_system.process_file(str(sci_file))
            
            result.passed = len(ocr_result.text) > 0
            result.details = {
                'tables_found': len(ocr_result.tables),
                'has_formulas': "Yes" if any(c in ocr_result.text for c in ['→', '₂', '₄']) else "No",
                'language': ocr_result.language
            }
            
        except Exception as e:
            result.passed = False
            result.errors.append(str(e))
            
        result.execution_time = time.time() - start
        self.results.append(result)
        self.print_test_status("Scientific Document Processing", result)
        
        # Test financial report
        result = DetailedTestResult("Financial Report Processing")
        start = time.time()
        
        try:
            fin_file = self.enhanced_test_dir / "financial_report.docx"
            ocr_result = await self.ocr_system.process_file(str(fin_file))
            
            result.passed = len(ocr_result.tables) > 0
            result.details = {
                'tables_found': len(ocr_result.tables),
                'quarters_found': "Yes" if all(q in ocr_result.text for q in ['Q1', 'Q2', 'Q3']) else "No",
                'currency_data': "Yes" if '€' in ocr_result.text else "No"
            }
            
        except Exception as e:
            result.passed = False
            result.errors.append(str(e))
            
        result.execution_time = time.time() - start
        self.results.append(result)
        self.print_test_status("Financial Report Processing", result)
        
    async def test_german_language_support(self):
        """Test comprehensive German language support"""
        self.print_section("German Language Support Tests")
        
        # Create test file with all special characters
        test_content = """
        Umlaute: ä ö ü Ä Ö Ü ß
        Wörter: Müller, Größe, Übung, Äpfel, Öl, süß
        Sätze: Die schöne Müllerin ging über die Brücke.
        Zahlen: 1.234,56 € (Eintausendzweihundertvierunddreißig Euro)
        Adressen: Königstraße 42, 80331 München
        """
        
        test_file = self.enhanced_test_dir / "german_special_test.txt"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
            
        result = DetailedTestResult("German Special Characters")
        start = time.time()
        
        try:
            ocr_result = await self.ocr_system.process_file(str(test_file))
            
            # Check each character
            missing_chars = []
            found_chars = []
            for char in GERMAN_CHARS:
                if char in ocr_result.text:
                    found_chars.append(char)
                else:
                    missing_chars.append(char)
                    
            result.passed = len(missing_chars) == 0
            result.details = {
                'found_chars': ', '.join(found_chars) if found_chars else "None",
                'missing_chars': ', '.join(missing_chars) if missing_chars else "None",
                'language_detected': ocr_result.language
            }
            
            # Check specific words
            test_words = ['Müller', 'Größe', 'Übung', 'München']
            found_words = [word for word in test_words if word in ocr_result.text]
            result.details['german_words'] = f"{len(found_words)}/{len(test_words)} found"
            
        except Exception as e:
            result.passed = False
            result.errors.append(str(e))
            
        result.execution_time = time.time() - start
        self.results.append(result)
        self.print_test_status("German Special Characters", result)
        
    async def test_performance_and_resources(self):
        """Test performance and resource usage"""
        self.print_section("Performance & Resource Tests")
        
        # Batch processing test
        result = DetailedTestResult("Batch Processing Performance")
        start = time.time()
        
        try:
            # Process multiple files
            test_files = list(self.basic_test_dir.glob("*.txt"))[:5]
            
            batch_start = time.time()
            results = []
            max_cpu = 0
            max_ram = 0
            
            for file in test_files:
                ocr_result = await self.ocr_system.process_file(str(file))
                results.append(ocr_result)
                
                # Check resources
                resources = get_resource_info()
                max_cpu = max(max_cpu, resources['cpu_percent'])
                max_ram = max(max_ram, resources['ram_gb'])
                
            batch_time = time.time() - batch_start
            
            result.passed = all(r.text for r in results)
            result.details = {
                'files_processed': len(results),
                'total_time': f"{batch_time:.2f}s",
                'avg_time_per_file': f"{batch_time/len(results):.2f}s",
                'max_cpu': f"{max_cpu:.1f}%",
                'max_ram': f"{max_ram:.2f}GB"
            }
            
            # Check resource limits
            if max_cpu > 70:
                result.warnings.append(f"CPU usage exceeded limit: {max_cpu:.1f}%")
            if max_ram > 2:
                result.warnings.append(f"RAM usage exceeded limit: {max_ram:.2f}GB")
                
        except Exception as e:
            result.passed = False
            result.errors.append(str(e))
            
        result.execution_time = time.time() - start
        self.results.append(result)
        self.print_test_status("Batch Processing Performance", result)
        
    async def test_edge_cases(self):
        """Test edge cases and error handling"""
        self.print_section("Edge Case Tests")
        
        # Empty file test
        result = DetailedTestResult("Empty File Handling")
        start = time.time()
        
        try:
            empty_file = self.enhanced_test_dir / "empty_test.txt"
            with open(empty_file, 'w') as f:
                f.write("")
                
            ocr_result = await self.ocr_system.process_file(str(empty_file))
            
            result.passed = True  # Should handle gracefully
            result.details = {
                'text_length': len(ocr_result.text),
                'tables_found': len(ocr_result.tables),
                'handled_gracefully': "Yes"
            }
            
        except Exception as e:
            result.passed = False
            result.errors.append(str(e))
            
        result.execution_time = time.time() - start
        self.results.append(result)
        self.print_test_status("Empty File Handling", result)
        
        # Non-existent file test
        result = DetailedTestResult("Non-Existent File Handling")
        start = time.time()
        
        try:
            await self.ocr_system.process_file("non_existent_file.pdf")
            result.passed = False
            result.errors.append("Should have raised FileNotFoundError")
        except FileNotFoundError:
            result.passed = True
            result.details = {'error_handling': 'Correct exception raised'}
        except Exception as e:
            result.passed = False
            result.errors.append(f"Wrong exception: {type(e).__name__}")
            
        result.execution_time = time.time() - start
        self.results.append(result)
        self.print_test_status("Non-Existent File Handling", result)
        
    def print_summary(self):
        """Print comprehensive test summary"""
        total_time = time.time() - self.start_time
        passed_tests = sum(1 for r in self.results if r.passed)
        total_tests = len(self.results)
        
        self.print_section("TEST SUMMARY")
        
        # Overall results
        print(f"{BOLD}Total Tests:{RESET} {total_tests}")
        print(f"{BOLD}{GREEN}Passed:{RESET} {passed_tests}")
        print(f"{BOLD}{RED}Failed:{RESET} {total_tests - passed_tests}")
        print(f"{BOLD}Success Rate:{RESET} {passed_tests/total_tests*100:.1f}%")
        print(f"{BOLD}Total Time:{RESET} {total_time:.2f} seconds")
        
        # Resource usage summary
        print(f"\n{BOLD}Resource Usage Summary:{RESET}")
        final_resources = get_resource_info()
        print(f"  Final CPU: {final_resources['cpu_percent']:.1f}%")
        print(f"  Final RAM: {final_resources['ram_gb']:.2f}GB ({final_resources['ram_percent']:.1f}%)")
        
        # Failed tests details
        if passed_tests < total_tests:
            print(f"\n{BOLD}{RED}Failed Tests:{RESET}")
            for result in self.results:
                if not result.passed:
                    print(f"  - {result.name}")
                    for error in result.errors:
                        print(f"    {RED}Error: {error}{RESET}")
                        
        # Warnings summary
        warnings = [w for r in self.results for w in r.warnings]
        if warnings:
            print(f"\n{BOLD}{YELLOW}Warnings:{RESET}")
            for warning in set(warnings):
                print(f"  - {warning}")
                
        # Save detailed results
        self.save_detailed_results()
        
    def save_detailed_results(self):
        """Save detailed test results to file"""
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': len(self.results),
            'passed': sum(1 for r in self.results if r.passed),
            'total_time': time.time() - self.start_time,
            'tests': [r.to_dict() for r in self.results]
        }
        
        results_file = Path("test_results.json")
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
            
        print(f"\n{CYAN}Detailed results saved to: {results_file}{RESET}")


async def main():
    """Run all tests"""
    tester = ComprehensiveOCRTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    # Run with proper error handling
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Tests interrupted by user{RESET}")
    except Exception as e:
        print(f"\n{RED}Fatal error: {e}{RESET}")
        raise