# OCR System with Resource Management

A comprehensive OCR system that supports multiple document formats with configurable resource limits.

**Repository**: https://github.com/Nraitschew/OCR/  
**Author**: n.raitschew@grvity.de

## Features

✅ **Multi-format Support**: PDF, DOCX, PNG, JPEG, TXT
✅ **German Language Support**: Full umlaut support (ä, ö, ü, Ä, Ö, Ü, ß)
✅ **Resource Management**: CPU and RAM usage limited to 70% (configurable)
✅ **Scanned Document Processing**: Handles both native and scanned documents
✅ **Table Extraction**: Automatically detects and extracts tables
✅ **Concurrent Processing**: Handles multiple files simultaneously
✅ **GPU Support**: Uses GPU acceleration when available, falls back to CPU

## Resource Configuration

Easily configure resource limits at the top of `ocr_system.py`:

```python
MAX_CPU_PERCENT = 70  # Maximum CPU usage percentage
MAX_RAM_PERCENT = 70  # Maximum RAM usage percentage  
MAX_RAM_GB = 2        # Maximum RAM in GB (can be exceeded if needed)
```

## Installation

1. Create virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
# Basic dependencies (for testing)
pip install -r requirements.txt

# System dependencies (for full OCR)
./setup.sh
```

## Usage

### Generate Test Documents
```bash
python test_document_generator.py
```

This creates test documents in all supported formats with German text and umlauts.

### Run Tests
```bash
python simple_test.py  # Basic test
python test_ocr_system.py  # Full test suite (requires all dependencies)
```

### Use OCR System
```python
from ocr_system import OCRSystem

# Initialize
ocr = OCRSystem()

# Process single file
result = await ocr.process_file("document.pdf")
print(result.text)
print(f"Language: {result.language}")
print(f"Confidence: {result.confidence}")

# Process multiple files
results = await ocr.process_batch(["doc1.pdf", "doc2.png", "doc3.docx"])
```

## Project Structure

```
/home/nikolai/ocr/
├── ocr_system.py              # Main OCR system with resource management
├── test_document_generator.py # Generates/downloads test documents
├── test_ocr_system.py        # Comprehensive test suite
├── simple_test.py            # Basic functionality test
├── requirements.txt          # Python dependencies
├── setup.sh                 # System dependencies installer
├── test_documents/          # Generated test documents
└── README.md               # This file
```

## Test Results

✅ All test documents generated successfully
✅ German umlauts properly handled: ä, ö, ü, Ä, Ö, Ü, ß
✅ Resource limits implemented and configurable
✅ Multiple document formats supported
✅ Test suite created for verification

## Dependencies

### Required for Full OCR:
- `easyocr`: GPU-accelerated OCR
- `pytesseract`: CPU-based OCR fallback
- `paddleocr`: Table extraction
- `PyMuPDF`: PDF processing
- `pdf2image`: Scanned PDF conversion
- `torch`: Deep learning framework

### Basic Dependencies:
- `pillow`: Image processing
- `fpdf2`: PDF generation
- `python-docx`: DOCX handling
- `opencv-python`: Image preprocessing
- `psutil`: Resource monitoring
- `requests`: Document downloading
- `beautifulsoup4`: Web scraping

## Docker Deployment

### Quick Start with Docker Compose

1. Clone the repository:
```bash
git clone https://github.com/Nraitschew/OCR.git
cd OCR
```

2. Start the service:
```bash
docker-compose up -d
```

The OCR service will be available at `http://localhost:4000`

### Docker Configuration

The service runs with:
- **Port**: 4000
- **CPU Limit**: 70%
- **Memory Limit**: 2GB
- **Auto-restart**: Enabled

## API Usage

### Health Check
```bash
curl http://localhost:4000/health
```

### System Status
```bash
curl http://localhost:4000/status
```

### OCR File Upload
```bash
# Process a PDF file
curl -X POST -F "file=@document.pdf" \
  -F "preserve_formatting=true" \
  http://localhost:4000/ocr/file

# Process an image
curl -X POST -F "file=@image.png" \
  -F "preserve_formatting=true" \
  http://localhost:4000/ocr/file

# Process a Word document
curl -X POST -F "file=@document.docx" \
  -F "preserve_formatting=true" \
  http://localhost:4000/ocr/file
```

### OCR with Base64 Encoding
```bash
# Encode file to base64
base64 document.pdf > document.b64

# Send base64 encoded file
curl -X POST -H "Content-Type: application/json" \
  -d '{
    "filename": "document.pdf",
    "content": "'$(cat document.b64)'",
    "preserve_formatting": true
  }' \
  http://localhost:4000/ocr/base64
```

### Response Format
```json
{
  "success": true,
  "text": "Extracted text content...",
  "language": "de",
  "confidence": 0.95,
  "processing_time": 2.34,
  "formatting_preserved": true,
  "tables_found": 1,
  "metadata": {
    "filename": "document.pdf",
    "file_size_mb": 1.23,
    "cpu_usage": 45.2,
    "ram_usage_mb": 234.5
  }
}
```

## CPU-Only Setup

For systems without GPU:

### Automatic Setup
```bash
./setup_cpu_only.sh
```

### Manual Setup
```bash
python3 -m venv venv_cpu
source venv_cpu/bin/activate
pip install -r requirements_cpu.txt
```

### Run CPU-Only System
```bash
python cpu_only_ocr_system.py
```

## Testing

### Test Line Breaks and Formatting
```bash
python test_linebreaks.py
```

### Run All Tests
```bash
# Basic test
python simple_test.py

# Full test suite
python test_ocr_system.py

# CPU-only tests
python test_cpu_only.py
```

## Notes

- The system automatically manages resources to stay within configured limits
- GPU acceleration is used when available, with automatic CPU fallback
- All German special characters are fully supported
- The system can handle both native text and scanned documents
- Line breaks and paragraph formatting are preserved
- Tables are automatically detected and extracted