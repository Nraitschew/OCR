# OCR System with Resource Management

A comprehensive OCR system that supports multiple document formats with configurable resource limits.

**Repository**: https://github.com/Nraitschew/OCR/  
**Author**: n.raitschew@grvity.de

## Features

✅ **Unified API Endpoint**: Single `/ocr` endpoint for all file types
✅ **Extended Format Support**: PDF, DOCX, PNG, JPEG, TXT, RTF, ODT, HTML, XML, CSV, TIFF, BMP, GIF, WebP
✅ **Automatic File Type Detection**: Uses magic numbers for accurate MIME type detection
✅ **German Language Support**: Full umlaut support (ä, ö, ü, Ä, Ö, Ü, ß) with proper Unicode handling
✅ **Resource Management**: Optimized for maximum performance (95% CPU, 90% RAM)
✅ **Scanned Document Processing**: Handles both native and scanned documents
✅ **Table Extraction**: Automatically detects and extracts tables
✅ **Concurrent Processing**: Handles multiple files simultaneously
✅ **GPU Support**: Uses GPU acceleration when available, falls back to CPU
✅ **Multiple Upload Methods**: Supports both multipart form data and base64 JSON uploads

## Resource Configuration

### Maximum Performance Mode

The system is now configured for **maximum resource utilization** to deliver the best performance:

```python
MAX_CPU_PERCENT = 95  # Uses up to 95% CPU for maximum performance
MAX_RAM_PERCENT = 90  # Uses up to 90% RAM with safeguards
MAX_RAM_GB = 8        # Increased RAM limit for large documents
MAX_WORKERS = 8       # Uses all CPU cores (up to 8) for parallel processing
```

### Key Features:
- **Dynamic Resource Management**: Automatically adjusts processing based on available resources
- **Memory Safeguards**: Triggers garbage collection when memory usage is high
- **Adaptive Image Resizing**: Reduces image size when memory is constrained
- **Error Recovery**: Continues processing even if individual files fail
- **Parallel Processing**: Processes multiple files simultaneously for maximum throughput

### Safety Mechanisms:
- Leaves 5% CPU headroom for system stability
- Monitors RAM usage and prevents out-of-memory errors
- Exponential backoff when resources are constrained
- Automatic cleanup after each file processing

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

## What's New

### Version 2.0 - Unified API & Enhanced Format Support

- **Unified `/ocr` Endpoint**: Single endpoint for all file uploads
- **Automatic File Type Detection**: Uses `python-magic` for accurate MIME type detection
- **Extended Format Support**: Added RTF, ODT, HTML, XML, CSV, TIFF, BMP, GIF, WebP
- **Improved Unicode Handling**: Umlauts and special characters display correctly without escape sequences
- **Enhanced Image Processing**: Better preprocessing for improved OCR accuracy on images with text

## Project Structure

```
/home/nikolai/ocr/
├── ocr_system.py              # Main OCR system with resource management
├── ocr_api.py                 # FastAPI service with unified endpoint
├── cpu_only_ocr_system.py     # CPU-optimized OCR implementation
├── test_document_generator.py # Generates/downloads test documents
├── test_ocr_system.py        # Comprehensive test suite
├── simple_test.py            # Basic functionality test
├── requirements.txt          # Python dependencies
├── requirements_api.txt      # API service dependencies
├── requirements_cpu.txt      # CPU-only dependencies
├── docker-compose.yml        # Docker deployment configuration
├── Dockerfile               # Container build instructions
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

### New Dependencies (v2.0):
- `python-magic`: Automatic file type detection
- `striprtf`: RTF file support
- `odfpy`: ODT file support
- `lxml`: Enhanced XML/HTML processing

**Note**: On Linux/macOS, you may need to install libmagic:
```bash
# Ubuntu/Debian
sudo apt-get install libmagic1

# macOS
brew install libmagic
```

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
- **CPU Limit**: 95% (Maximum performance mode)
- **Memory Limit**: 8GB (Dynamically managed)
- **Auto-restart**: Enabled
- **Parallel Workers**: Up to 8 concurrent processes

## API Usage

### API Key Configuration

The API requires an API key for authentication. Create a `.env` file in the project root:

```bash
KEY="your-secret-key-here"
```

**Important**: Never commit the `.env` file to version control!

### Migration from v1.x to v2.0

If you're upgrading from the previous version, note these changes:

1. **Endpoint Change**: 
   - Old: `/ocr/file` and `/ocr/base64`
   - New: `/ocr` (handles both multipart and base64)

2. **No Changes Required for**:
   - Request parameters (same field names)
   - Response format (same structure with additional metadata)
   - Authentication method (same API key)

3. **New Features Available**:
   - Automatic file type detection
   - Support for more file formats
   - Better Unicode handling

### Health Check
```bash
curl http://localhost:4000/health
```

### System Status
```bash
curl http://localhost:4000/status
```

### Unified OCR Endpoint

The API now provides a single `/ocr` endpoint that accepts all supported file types and automatically detects the upload method.

#### Multipart File Upload
```bash
# Process any supported file type
curl -X POST -F "file=@document.pdf" \
  -F "preserve_formatting=true" \
  -F "key=your-secret-key-here" \
  http://localhost:4000/ocr

# Examples for different formats:
# PDF
curl -X POST -F "file=@document.pdf" -F "preserve_formatting=true" -F "key=your-secret-key-here" http://localhost:4000/ocr

# Images (PNG, JPG, TIFF, BMP, GIF, WebP)
curl -X POST -F "file=@image.png" -F "preserve_formatting=true" -F "key=your-secret-key-here" http://localhost:4000/ocr

# Documents (DOCX, RTF, ODT)
curl -X POST -F "file=@document.docx" -F "preserve_formatting=true" -F "key=your-secret-key-here" http://localhost:4000/ocr

# Text/Data files (TXT, CSV, HTML, XML)
curl -X POST -F "file=@data.csv" -F "preserve_formatting=true" -F "key=your-secret-key-here" http://localhost:4000/ocr
```

#### Base64 JSON Upload
```bash
# Encode file to base64
base64 -w 0 document.pdf > document.b64

# Send base64 encoded file
curl -X POST -H "Content-Type: application/json" \
  -d '{
    "filename": "document.pdf",
    "content": "'$(cat document.b64)'",
    "preserve_formatting": true,
    "key": "your-secret-key-here"
  }' \
  http://localhost:4000/ocr
```

### Response Format
```json
{
  "success": true,
  "text": "Extracted text content mit Umlauten: ä, ö, ü...",
  "language": "de",
  "confidence": 0.95,
  "processing_time": 2.34,
  "formatting_preserved": true,
  "tables_found": 1,
  "error": null,
  "metadata": {
    "filename": "document.pdf",
    "detected_mime_type": "application/pdf",
    "file_extension": ".pdf",
    "file_size_mb": 1.23,
    "cpu_usage": 45.2,
    "ram_usage_mb": 234.5
  }
}
```

#### Supported File Types

| Format | Extensions | MIME Type |
|--------|------------|-----------|
| PDF | .pdf | application/pdf |
| Images | .png, .jpg, .jpeg, .tiff, .tif, .bmp, .gif, .webp | image/* |
| Word | .docx | application/vnd.openxmlformats-officedocument.wordprocessingml.document |
| OpenDocument | .odt | application/vnd.oasis.opendocument.text |
| Rich Text | .rtf | application/rtf, text/rtf |
| Plain Text | .txt | text/plain |
| CSV | .csv | text/csv |
| HTML | .html, .htm | text/html |
| XML | .xml | text/xml, application/xml |

### Example Usage in Different Languages

#### Python
```python
import requests

# Multipart upload
with open('document.pdf', 'rb') as f:
    response = requests.post(
        'http://localhost:4000/ocr',
        files={'file': f},
        data={'key': 'your-secret-key-here', 'preserve_formatting': 'true'}
    )
    print(response.json()['text'])

# Base64 upload
import base64
with open('document.pdf', 'rb') as f:
    content = base64.b64encode(f.read()).decode('utf-8')
    response = requests.post(
        'http://localhost:4000/ocr',
        json={
            'filename': 'document.pdf',
            'content': content,
            'key': 'your-secret-key-here',
            'preserve_formatting': True
        }
    )
    print(response.json()['text'])
```

#### JavaScript/Node.js
```javascript
const FormData = require('form-data');
const fs = require('fs');

// Multipart upload
const form = new FormData();
form.append('file', fs.createReadStream('document.pdf'));
form.append('key', 'your-secret-key-here');
form.append('preserve_formatting', 'true');

fetch('http://localhost:4000/ocr', {
    method: 'POST',
    body: form
})
.then(res => res.json())
.then(data => console.log(data.text));
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