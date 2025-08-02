#!/usr/bin/env python3
"""
Test Line Breaks and Paragraph Preservation
Tests if the OCR system correctly preserves formatting
"""

import asyncio
from pathlib import Path
from cpu_only_ocr_system import CPUOnlyOCRSystem
from fpdf import FPDF
from docx import Document
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'


class LineBreakTester:
    """Test line breaks and paragraph preservation"""
    
    def __init__(self):
        self.ocr_system = CPUOnlyOCRSystem()
        self.test_dir = Path("test_linebreaks")
        self.test_dir.mkdir(exist_ok=True)
        
    def create_test_documents(self):
        """Create test documents with various formatting"""
        print(f"{BLUE}Creating test documents with line breaks and paragraphs...{RESET}")
        
        # Test content with various formatting
        self.test_content = """Erste Zeile des Dokuments.

Dies ist der zweite Absatz mit mehr Text. Er enthält mehrere Sätze.
Diese Zeile gehört noch zum zweiten Absatz.

Dritter Absatz nach Leerzeile.


Vierter Absatz nach zwei Leerzeilen.

Liste mit Einträgen:
- Erster Eintrag
- Zweiter Eintrag  
- Dritter Eintrag

Tabelle:
Name        Alter   Stadt
Max Müller  35      München
Anna Klein  28      Köln

Letzter Absatz mit Sonderzeichen: äöüÄÖÜß €."""
        
        # Create text file
        self._create_text_file()
        
        # Create PDF with formatting
        self._create_pdf_file()
        
        # Create DOCX with formatting
        self._create_docx_file()
        
        # Create image with text
        self._create_image_file()
        
        print(f"{GREEN}✓ Test documents created{RESET}")
        
    def _create_text_file(self):
        """Create text file with line breaks"""
        with open(self.test_dir / "formatted_text.txt", "w", encoding="utf-8") as f:
            f.write(self.test_content)
            
    def _create_pdf_file(self):
        """Create PDF with formatted text"""
        from fpdf import FPDF
        
        # Create custom PDF class with proper unicode support
        class PDF(FPDF):
            pass
            
        pdf = PDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Use built-in font that works reliably
        pdf.set_font('helvetica', size=12)
        
        # Split content into paragraphs
        paragraphs = self.test_content.split('\n\n')
        
        for paragraph in paragraphs:
            if paragraph.strip():
                # Replace special characters for PDF compatibility
                safe_para = paragraph.replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue')
                safe_para = safe_para.replace('Ä', 'Ae').replace('Ö', 'Oe').replace('Ü', 'Ue')
                safe_para = safe_para.replace('ß', 'ss').replace('€', 'EUR')
                
                # Add each line of the paragraph
                lines = safe_para.split('\n')
                for line in lines:
                    if line.strip():
                        pdf.cell(0, 10, line, ln=True)
                
                # Add empty line between paragraphs
                pdf.ln(5)
        
        pdf.output(str(self.test_dir / "formatted_text.pdf"))
        
    def _create_docx_file(self):
        """Create DOCX with formatted text"""
        doc = Document()
        
        # Add paragraphs
        paragraphs = self.test_content.split('\n\n')
        for para in paragraphs:
            if para.strip():
                doc.add_paragraph(para)
                
        doc.save(str(self.test_dir / "formatted_text.docx"))
        
    def _create_image_file(self):
        """Create image with formatted text"""
        # Create white background
        img = Image.new('RGB', (800, 1000), color='white')
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 16)
        except:
            font = ImageFont.load_default()
            
        # Draw text with line breaks
        y_position = 50
        line_height = 25
        
        lines = self.test_content.split('\n')
        for line in lines:
            if line.strip():
                draw.text((50, y_position), line, fill='black', font=font)
            y_position += line_height
            
        img.save(self.test_dir / "formatted_text.png")
        
    async def test_line_break_preservation(self):
        """Test if line breaks are preserved correctly"""
        print(f"\n{BLUE}Testing Line Break and Paragraph Preservation{RESET}")
        print("="*60)
        
        results = {}
        
        # Test each file type
        for file_type in ["txt", "pdf", "docx", "png"]:
            file_path = self.test_dir / f"formatted_text.{file_type}"
            
            print(f"\n{BOLD}Testing {file_type.upper()} file...{RESET}")
            
            try:
                result = await self.ocr_system.process_file(str(file_path))
                
                # Analyze formatting preservation
                analysis = self._analyze_formatting(result.text)
                
                # Print results
                success = analysis['paragraphs_found'] >= 4  # We expect at least 4 paragraphs
                status = f"{GREEN}✓ PASSED{RESET}" if success else f"{RED}✗ FAILED{RESET}"
                
                print(f"{status} {file_type.upper()} formatting preservation")
                print(f"  - Paragraphs found: {analysis['paragraphs_found']}")
                print(f"  - Empty lines: {analysis['empty_lines']}")
                print(f"  - Lines with content: {analysis['content_lines']}")
                print(f"  - German chars preserved: {'Yes' if analysis['has_umlauts'] else 'No'}")
                
                if file_type == "txt":
                    # For text files, we expect exact preservation
                    exact_match = result.text.strip() == self.test_content.strip()
                    print(f"  - Exact match: {'Yes' if exact_match else 'No'}")
                    
                # Show sample of preserved formatting
                print(f"\n  Preview of formatting:")
                lines = result.text.split('\n')[:10]  # First 10 lines
                for i, line in enumerate(lines):
                    if line.strip():
                        print(f"    Line {i+1}: {line[:50]}...")
                    else:
                        print(f"    Line {i+1}: [empty line]")
                        
                results[file_type] = success
                
            except Exception as e:
                print(f"{RED}✗ Error processing {file_type}: {e}{RESET}")
                results[file_type] = False
                
        # Summary
        print(f"\n{BLUE}{'='*60}{RESET}")
        print(f"{BOLD}Summary:{RESET}")
        passed = sum(1 for v in results.values() if v)
        total = len(results)
        
        print(f"Total: {total} tests")
        print(f"{GREEN}Passed: {passed}{RESET}")
        print(f"{RED}Failed: {total - passed}{RESET}")
        
        success_rate = (passed / total * 100) if total > 0 else 0
        color = GREEN if success_rate >= 75 else YELLOW if success_rate >= 50 else RED
        print(f"\n{color}{BOLD}Success Rate: {success_rate:.1f}%{RESET}")
        
        return results
        
    def _analyze_formatting(self, text):
        """Analyze formatting preservation"""
        lines = text.split('\n')
        
        analysis = {
            'total_lines': len(lines),
            'empty_lines': sum(1 for line in lines if not line.strip()),
            'content_lines': sum(1 for line in lines if line.strip()),
            'paragraphs_found': 0,
            'has_umlauts': any(char in text for char in 'äöüÄÖÜß')
        }
        
        # Count paragraphs (separated by empty lines)
        in_paragraph = False
        for line in lines:
            if line.strip() and not in_paragraph:
                analysis['paragraphs_found'] += 1
                in_paragraph = True
            elif not line.strip():
                in_paragraph = False
                
        return analysis
        
    async def run_all_tests(self):
        """Run all line break tests"""
        print(f"\n{BLUE}{BOLD}Line Break and Paragraph Preservation Test Suite{RESET}")
        print(f"{BLUE}{'='*60}{RESET}")
        
        # Create test documents
        self.create_test_documents()
        
        # Run tests
        results = await self.test_line_break_preservation()
        
        # Create detailed test with complex formatting
        await self.test_complex_formatting()
        
        print(f"\n{BLUE}Test complete!{RESET}")
        
    async def test_complex_formatting(self):
        """Test complex formatting scenarios"""
        print(f"\n{BLUE}Testing Complex Formatting Scenarios{RESET}")
        print("="*60)
        
        # Create document with complex formatting
        complex_content = """ÜBERSCHRIFT

Einleitung:
Dies ist ein komplexes Dokument mit verschiedenen Formatierungen.

1. Nummerierte Liste:
   1.1 Unterpunkt eins
   1.2 Unterpunkt zwei
   
2. Zweiter Hauptpunkt
   - Aufzählung A
   - Aufzählung B

Code-Beispiel:
    def hello_world():
        print("Hallo Welt!")
        
Zitat:
"Dies ist ein wichtiges Zitat mit Anführungszeichen."

Kontaktdaten:
Max Mustermann
Musterstraße 123
80331 München
Tel: +49 89 12345678

Ende des Dokuments."""
        
        # Save as text file
        complex_file = self.test_dir / "complex_formatting.txt"
        with open(complex_file, "w", encoding="utf-8") as f:
            f.write(complex_content)
            
        # Test OCR
        result = await self.ocr_system.process_file(str(complex_file))
        
        # Check specific formatting elements
        checks = {
            "Indentation preserved": "   1.1" in result.text,
            "Code block spacing": "    def" in result.text,
            "Quote marks preserved": '"Dies ist ein wichtiges Zitat' in result.text,
            "Phone number format": "+49 89 12345678" in result.text,
            "Address formatting": "80331 München" in result.text
        }
        
        print("\nComplex formatting checks:")
        for check, passed in checks.items():
            status = f"{GREEN}✓{RESET}" if passed else f"{RED}✗{RESET}"
            print(f"{status} {check}")
            
        # Count preserved structure elements
        preserved_elements = sum(1 for passed in checks.values() if passed)
        print(f"\nPreserved elements: {preserved_elements}/{len(checks)}")


async def main():
    """Run line break tests"""
    tester = LineBreakTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())