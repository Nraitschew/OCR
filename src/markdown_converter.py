"""
Markdown converter for OCR output with table formatting
"""
import logging
from typing import List, Dict, Any, Optional
import re
import pandas as pd
from tabulate import tabulate

from .config import PRESERVE_FORMATTING, INCLUDE_CONFIDENCE_SCORES

logger = logging.getLogger(__name__)


class MarkdownConverter:
    """Convert OCR results to well-formatted Markdown"""
    
    def __init__(self):
        self.preserve_formatting = PRESERVE_FORMATTING
        self.include_confidence = INCLUDE_CONFIDENCE_SCORES
        
    def convert_to_markdown(self, ocr_result: Dict) -> str:
        """
        Convert OCR result to Markdown format
        
        Args:
            ocr_result: OCR result dictionary
            
        Returns:
            Markdown formatted string
        """
        markdown_parts = []
        
        # Add metadata header
        if self.include_confidence:
            markdown_parts.append(self._create_metadata_section(ocr_result))
            
        # Process based on file type
        file_type = ocr_result.get('file_type', 'unknown')
        
        if file_type == 'pdf':
            markdown_parts.append(self._convert_pdf_result(ocr_result))
        elif file_type == 'image':
            markdown_parts.append(self._convert_image_result(ocr_result))
        elif file_type == 'docx':
            markdown_parts.append(self._convert_docx_result(ocr_result))
        elif file_type == 'text':
            markdown_parts.append(self._convert_text_result(ocr_result))
        else:
            # Generic conversion
            markdown_parts.append(self._convert_generic_result(ocr_result))
            
        return '\n\n'.join(filter(None, markdown_parts))
        
    def _create_metadata_section(self, result: Dict) -> str:
        """Create metadata section"""
        metadata = []
        metadata.append("## Document Information")
        metadata.append("")
        
        # Basic info
        if 'file_path' in result:
            metadata.append(f"- **File**: `{result['file_path']}`")
        if 'file_type' in result:
            metadata.append(f"- **Type**: {result['file_type']}")
            
        # Language info
        if 'language' in result:
            lang = result['language']
            confidence = result.get('confidence', 0)
            metadata.append(f"- **Language**: {lang} ({confidence:.1%} confidence)")
            
        # Processing info
        if 'engine_used' in result:
            metadata.append(f"- **OCR Engine**: {result['engine_used']}")
        if 'processing_time' in result:
            metadata.append(f"- **Processing Time**: {result['processing_time']:.2f}s")
            
        # Resource usage
        if 'resource_usage' in result:
            usage = result['resource_usage']
            metadata.append(f"- **Memory Used**: {usage.get('memory_mb', 0):.1f} MB")
            metadata.append(f"- **CPU Usage**: {usage.get('cpu_percent', 0):.1f}%")
            
        metadata.append("")
        metadata.append("---")
        
        return '\n'.join(metadata)
        
    def _convert_pdf_result(self, result: Dict) -> str:
        """Convert PDF OCR result to Markdown"""
        parts = []
        
        # Document header
        parts.append("# PDF Document")
        parts.append("")
        parts.append(f"**Total Pages**: {result.get('total_pages', 0)}")
        parts.append(f"**Document Type**: {'Scanned' if result.get('is_scanned') else 'Native Text'}")
        parts.append("")
        
        # Process each page
        for page in result.get('pages', []):
            page_num = page.get('page_number', 0)
            parts.append(f"## Page {page_num}")
            parts.append("")
            
            # Add OCR info if applicable
            if page.get('ocr_applied'):
                confidence = page.get('ocr_confidence', 0)
                parts.append(f"*OCR applied (confidence: {confidence:.1%})*")
                parts.append("")
                
            # Add text content
            text = page.get('text', '').strip()
            if text:
                parts.append(self._format_text(text))
                parts.append("")
                
            # Add tables
            tables = page.get('tables', [])
            if tables:
                parts.append("### Tables")
                parts.append("")
                for i, table in enumerate(tables):
                    parts.append(self._format_table(table, i + 1))
                    parts.append("")
                    
            parts.append("---")
            parts.append("")
            
        return '\n'.join(parts)
        
    def _convert_image_result(self, result: Dict) -> str:
        """Convert image OCR result to Markdown"""
        parts = []
        
        # Document header
        parts.append("# Image OCR Result")
        parts.append("")
        parts.append(f"**Format**: {result.get('format', 'unknown').upper()}")
        
        if self.include_confidence and 'confidence' in result:
            parts.append(f"**OCR Confidence**: {result['confidence']:.1%}")
            
        parts.append("")
        parts.append("## Extracted Text")
        parts.append("")
        
        # Add text
        text = result.get('text', '').strip()
        if text:
            parts.append(self._format_text(text))
        else:
            parts.append("*No text detected*")
            
        # Add tables
        tables = result.get('tables', [])
        if tables:
            parts.append("")
            parts.append("## Tables")
            parts.append("")
            for i, table in enumerate(tables):
                parts.append(self._format_table(table, i + 1))
                parts.append("")
                
        return '\n'.join(parts)
        
    def _convert_docx_result(self, result: Dict) -> str:
        """Convert DOCX result to Markdown"""
        parts = []
        
        # Document header
        parts.append("# Word Document")
        parts.append("")
        
        # Add paragraphs
        paragraphs = result.get('paragraphs', [])
        if paragraphs:
            parts.append("## Content")
            parts.append("")
            for para in paragraphs:
                if para.strip():
                    parts.append(self._format_text(para))
                    parts.append("")
                    
        # Add tables
        tables = result.get('tables', [])
        if tables:
            parts.append("## Tables")
            parts.append("")
            for i, table in enumerate(tables):
                parts.append(self._format_table(table, i + 1))
                parts.append("")
                
        return '\n'.join(parts)
        
    def _convert_text_result(self, result: Dict) -> str:
        """Convert text file result to Markdown"""
        parts = []
        
        # Document header
        parts.append("# Text Document")
        parts.append("")
        parts.append(f"**Encoding**: {result.get('encoding', 'unknown')}")
        parts.append("")
        parts.append("## Content")
        parts.append("")
        
        # Add text
        text = result.get('text', '').strip()
        if text:
            parts.append(self._format_text(text))
        else:
            parts.append("*Empty file*")
            
        return '\n'.join(parts)
        
    def _convert_generic_result(self, result: Dict) -> str:
        """Generic conversion for unknown formats"""
        parts = []
        
        # Add any text content
        text = result.get('text', '').strip()
        if text:
            parts.append(self._format_text(text))
            
        # Add any tables
        tables = result.get('tables', [])
        for i, table in enumerate(tables):
            parts.append("")
            parts.append(self._format_table(table, i + 1))
            
        return '\n'.join(parts)
        
    def _format_text(self, text: str) -> str:
        """Format text content"""
        if not self.preserve_formatting:
            # Clean up text
            text = re.sub(r'\n\s*\n', '\n\n', text)  # Remove extra blank lines
            text = re.sub(r' +', ' ', text)  # Remove extra spaces
            
        # Escape markdown special characters if needed
        # But preserve actual markdown if it looks intentional
        if not self._looks_like_markdown(text):
            text = self._escape_markdown(text)
            
        return text
        
    def _format_table(self, table_data: Dict, table_num: int) -> str:
        """Format table data as Markdown table"""
        parts = []
        
        if self.include_confidence and 'confidence' in table_data:
            parts.append(f"*Table {table_num} (confidence: {table_data['confidence']:.1%})*")
            parts.append("")
            
        data = table_data.get('data', [])
        
        if not data:
            return ""
            
        # Use pandas for better table handling
        try:
            # Assume first row is header if it looks like headers
            if self._is_header_row(data[0]) and len(data) > 1:
                df = pd.DataFrame(data[1:], columns=data[0])
            else:
                # No header, create generic columns
                df = pd.DataFrame(data)
                df.columns = [f"Column {i+1}" for i in range(len(df.columns))]
                
            # Convert to markdown using tabulate
            table_md = tabulate(df, headers='keys', tablefmt='pipe', showindex=False)
            parts.append(table_md)
            
        except Exception as e:
            logger.error(f"Error formatting table: {e}")
            # Fallback to simple formatting
            parts.append(self._simple_table_format(data))
            
        return '\n'.join(parts)
        
    def _simple_table_format(self, data: List[List[str]]) -> str:
        """Simple table formatting fallback"""
        if not data:
            return ""
            
        # Calculate column widths
        col_widths = []
        for col_idx in range(len(data[0])):
            max_width = max(len(str(row[col_idx])) if col_idx < len(row) else 0 
                           for row in data)
            col_widths.append(max(max_width, 3))  # Minimum 3 chars
            
        lines = []
        
        # Format each row
        for row_idx, row in enumerate(data):
            cells = []
            for col_idx, cell in enumerate(row):
                width = col_widths[col_idx] if col_idx < len(col_widths) else 10
                cells.append(str(cell).ljust(width))
            lines.append("| " + " | ".join(cells) + " |")
            
            # Add separator after first row (assumed header)
            if row_idx == 0:
                separators = ["-" * width for width in col_widths[:len(row)]]
                lines.append("| " + " | ".join(separators) + " |")
                
        return '\n'.join(lines)
        
    def _is_header_row(self, row: List[str]) -> bool:
        """Check if a row looks like a header"""
        # Simple heuristic: headers are usually short and don't contain numbers only
        for cell in row:
            cell_str = str(cell).strip()
            if not cell_str or cell_str.replace('.', '').replace(',', '').isdigit():
                return False
        return True
        
    def _looks_like_markdown(self, text: str) -> bool:
        """Check if text already contains markdown"""
        markdown_patterns = [
            r'^#{1,6} ',  # Headers
            r'\*\*[^*]+\*\*',  # Bold
            r'\*[^*]+\*',  # Italic
            r'\[.+\]\(.+\)',  # Links
            r'^\* ',  # Bullet lists
            r'^\d+\. ',  # Numbered lists
            r'^\|.+\|',  # Tables
            r'^```',  # Code blocks
        ]
        
        for pattern in markdown_patterns:
            if re.search(pattern, text, re.MULTILINE):
                return True
                
        return False
        
    def _escape_markdown(self, text: str) -> str:
        """Escape markdown special characters"""
        # Only escape if not already escaped
        chars_to_escape = ['*', '_', '[', ']', '(', ')', '#', '+', '-', '.', '!']
        
        for char in chars_to_escape:
            # Don't escape if already escaped
            text = re.sub(f'(?<!\\\\)\\{char}', f'\\{char}', text)
            
        return text