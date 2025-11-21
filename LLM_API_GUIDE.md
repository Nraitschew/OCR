# OCR Service API Guide for LLMs

## Service Information

**Base URL**: `https://ocr-server-u27285.vm.elestio.app`

**Authentication**: Required via API key in request payload

**Maximum File Size**: 250 MB per file

**Response Format**: JSON with UTF-8 encoding (umlauts and special characters preserved)

---

## Available Endpoints

### 1. Health Check
```
GET https://ocr-server-u27285.vm.elestio.app/health
```
No authentication required. Returns service health status.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-11-21T08:30:00.000000",
  "ocr_functional": true
}
```

### 2. System Status
```
GET https://ocr-server-u27285.vm.elestio.app/status
```
No authentication required. Returns resource usage and limits.

**Response**:
```json
{
  "timestamp": "2025-11-21T08:30:00.000000",
  "resources": {
    "cpu_percent": 45.2,
    "cpu_cores": 4,
    "ram_mb": 1234.56,
    "ram_gb": 1.21,
    "ram_percent": 15.4,
    "threads": 12
  },
  "limits": {
    "max_cpu_percent": 70,
    "max_ram_gb": 2,
    "max_file_size_mb": 250
  }
}
```

### 3. OCR Processing (Unified Endpoint)
```
POST https://ocr-server-u27285.vm.elestio.app/ocr
```
**Authentication**: Required (API key in request)

**Accepts**: Multipart form data OR JSON with base64 encoded file

---

## Supported File Formats

| Format | Extensions | Notes |
|--------|------------|-------|
| PDF | .pdf | Handles both native text and scanned PDFs |
| Images | .png, .jpg, .jpeg, .tiff, .tif, .bmp, .gif, .webp | OCR applied to extract text |
| Word | .docx | Microsoft Word documents |
| OpenDocument | .odt | LibreOffice/OpenOffice documents |
| Rich Text | .rtf | Rich Text Format |
| Plain Text | .txt | Direct text extraction |
| CSV | .csv | Comma-separated values |
| HTML | .html, .htm | Web pages |
| XML | .xml | XML documents |

---

## Request Methods

### Method 1: Multipart Form Upload (Recommended)

Use this method when you have direct file access or binary data.

**cURL Example**:
```bash
curl -X POST https://ocr-server-u27285.vm.elestio.app/ocr \
  -F "file=@/path/to/document.pdf" \
  -F "preserve_formatting=true" \
  -F "key=YOUR_API_KEY_HERE"
```

**Python Example**:
```python
import requests

url = "https://ocr-server-u27285.vm.elestio.app/ocr"

# Open and send file
with open("document.pdf", "rb") as f:
    files = {"file": f}
    data = {
        "preserve_formatting": "true",
        "key": "YOUR_API_KEY_HERE"
    }
    response = requests.post(url, files=files, data=data)

result = response.json()
print(result["text"])
```

**JavaScript/Node.js Example**:
```javascript
const FormData = require('form-data');
const fs = require('fs');
const fetch = require('node-fetch');

const url = 'https://ocr-server-u27285.vm.elestio.app/ocr';
const form = new FormData();

form.append('file', fs.createReadStream('document.pdf'));
form.append('preserve_formatting', 'true');
form.append('key', 'YOUR_API_KEY_HERE');

fetch(url, {
    method: 'POST',
    body: form
})
.then(res => res.json())
.then(data => console.log(data.text));
```

### Method 2: Base64 JSON Upload

Use this method when you have base64 encoded file content.

**cURL Example**:
```bash
# First, encode the file
FILE_CONTENT=$(base64 -w 0 document.pdf)

# Send as JSON
curl -X POST https://ocr-server-u27285.vm.elestio.app/ocr \
  -H "Content-Type: application/json" \
  -d "{
    \"filename\": \"document.pdf\",
    \"content\": \"$FILE_CONTENT\",
    \"preserve_formatting\": true,
    \"key\": \"YOUR_API_KEY_HERE\"
  }"
```

**Python Example**:
```python
import requests
import base64

url = "https://ocr-server-u27285.vm.elestio.app/ocr"

# Read and encode file
with open("document.pdf", "rb") as f:
    file_content = base64.b64encode(f.read()).decode('utf-8')

# Send as JSON
payload = {
    "filename": "document.pdf",
    "content": file_content,
    "preserve_formatting": True,
    "key": "YOUR_API_KEY_HERE"
}

response = requests.post(url, json=payload)
result = response.json()
print(result["text"])
```

**JavaScript Example**:
```javascript
const fs = require('fs');
const fetch = require('node-fetch');

const url = 'https://ocr-server-u27285.vm.elestio.app/ocr';

// Read and encode file
const fileBuffer = fs.readFileSync('document.pdf');
const fileContent = fileBuffer.toString('base64');

// Send as JSON
fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        filename: 'document.pdf',
        content: fileContent,
        preserve_formatting: true,
        key: 'YOUR_API_KEY_HERE'
    })
})
.then(res => res.json())
.then(data => console.log(data.text));
```

---

## Request Parameters

### Multipart Form Parameters:
- **file** (required): The file to process
- **preserve_formatting** (optional, default: true): Whether to preserve text formatting
- **key** (required): Your API authentication key

### JSON Parameters:
- **filename** (required): Original filename with extension
- **content** (required): Base64 encoded file content
- **preserve_formatting** (optional, default: true): Whether to preserve text formatting
- **key** (required): Your API authentication key

---

## Response Format

**Success Response** (HTTP 200):
```json
{
  "success": true,
  "text": "Extracted text content with proper German umlauts: ä, ö, ü, Ä, Ö, Ü, ß",
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

**Error Response** (HTTP 4xx/5xx):
```json
{
  "detail": "Error message describing what went wrong"
}
```

### Response Fields:

| Field | Type | Description |
|-------|------|-------------|
| success | boolean | Whether OCR processing succeeded |
| text | string | Extracted text content (UTF-8, umlauts preserved) |
| language | string | Detected language code (e.g., "de", "en") |
| confidence | float | OCR confidence score (0.0 to 1.0) |
| processing_time | float | Processing time in seconds |
| formatting_preserved | boolean | Whether formatting was preserved |
| tables_found | integer | Number of tables detected and extracted |
| error | string/null | Error message if processing failed |
| metadata | object | Additional file and processing information |

---

## Error Codes

| Code | Meaning | Cause |
|------|---------|-------|
| 400 | Bad Request | Missing file, invalid format, or malformed request |
| 401 | Unauthorized | Invalid or missing API key |
| 413 | Payload Too Large | File exceeds 250 MB limit |
| 500 | Internal Server Error | Processing error (returned with success: false) |

---

## Usage Guidelines for LLMs

### 1. When to Use This Service

Use this OCR service when you need to:
- Extract text from PDF documents (scanned or native)
- Read text from images (screenshots, photos of documents, etc.)
- Convert Word/RTF/ODT documents to plain text
- Extract tables from documents
- Process documents with German language content (umlauts fully supported)

### 2. File Size Considerations

- **Maximum**: 250 MB per file
- **Optimal for concurrent use**: 50-100 MB images, unlimited for PDFs/text documents
- **Processing time**: ~1-5 seconds for small files, up to 30+ seconds for 250 MB files

### 3. Best Practices

**DO**:
- ✅ Always include the API key
- ✅ Specify the correct filename with extension
- ✅ Handle both success and error responses
- ✅ Use multipart form for local files (more efficient)
- ✅ Use base64 JSON when you receive encoded content from users
- ✅ Check `/health` before processing multiple files
- ✅ Preserve the UTF-8 encoding in responses (umlauts are correctly encoded)

**DON'T**:
- ❌ Send files larger than 250 MB
- ❌ Send unsupported file types
- ❌ Hardcode API keys in public code
- ❌ Assume instant responses for large files
- ❌ Re-encode the response text (it's already properly UTF-8)

### 4. Example LLM Workflow

```
1. User provides a file or file path
2. Check file size (< 250 MB)
3. Verify file type is supported
4. Choose upload method (multipart for files, base64 for encoded content)
5. Make POST request to /ocr with API key
6. Parse JSON response
7. Extract "text" field for the OCR result
8. Handle errors gracefully if success = false
```

### 5. Handling Responses

**Success Case**:
```python
if response.json()["success"]:
    extracted_text = response.json()["text"]
    language = response.json()["language"]
    confidence = response.json()["confidence"]
    # Use extracted_text...
```

**Error Case**:
```python
if not response.json()["success"]:
    error_message = response.json().get("error", "Unknown error")
    # Handle error...
```

**HTTP Error**:
```python
if response.status_code != 200:
    error_detail = response.json().get("detail", "Request failed")
    # Handle HTTP error...
```

---

## Complete Working Examples

### Example 1: Process a Local PDF File

```python
import requests

def process_pdf(file_path, api_key):
    """Process a PDF file and return extracted text"""
    url = "https://ocr-server-u27285.vm.elestio.app/ocr"

    with open(file_path, "rb") as f:
        files = {"file": f}
        data = {
            "preserve_formatting": "true",
            "key": api_key
        }
        response = requests.post(url, files=files, data=data)

    if response.status_code == 200:
        result = response.json()
        if result["success"]:
            return {
                "text": result["text"],
                "language": result["language"],
                "confidence": result["confidence"],
                "tables": result["tables_found"]
            }
        else:
            return {"error": result.get("error", "Processing failed")}
    else:
        return {"error": f"HTTP {response.status_code}: {response.json().get('detail', 'Unknown error')}"}

# Usage
result = process_pdf("document.pdf", "YOUR_API_KEY_HERE")
if "error" in result:
    print(f"Error: {result['error']}")
else:
    print(f"Extracted text: {result['text']}")
    print(f"Language: {result['language']}")
    print(f"Confidence: {result['confidence']:.2%}")
```

### Example 2: Process Base64 Encoded Image

```python
import requests
import base64

def process_base64_image(base64_content, filename, api_key):
    """Process a base64 encoded image"""
    url = "https://ocr-server-u27285.vm.elestio.app/ocr"

    payload = {
        "filename": filename,
        "content": base64_content,
        "preserve_formatting": True,
        "key": api_key
    }

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        return response.json()
    else:
        return {
            "success": False,
            "error": f"HTTP {response.status_code}: {response.json().get('detail', 'Unknown error')}"
        }

# Usage with base64 string
result = process_base64_image("iVBORw0KGgoAAAANS...", "screenshot.png", "YOUR_API_KEY_HERE")
print(result["text"])
```

### Example 3: Batch Processing Multiple Files

```python
import requests
from concurrent.futures import ThreadPoolExecutor
import os

def process_single_file(file_path, api_key):
    """Process a single file"""
    url = "https://ocr-server-u27285.vm.elestio.app/ocr"

    try:
        # Check file size
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > 250:
            return {"file": file_path, "error": f"File too large: {file_size_mb:.1f} MB"}

        with open(file_path, "rb") as f:
            files = {"file": f}
            data = {"preserve_formatting": "true", "key": api_key}
            response = requests.post(url, files=files, data=data, timeout=120)

        if response.status_code == 200:
            result = response.json()
            return {
                "file": file_path,
                "success": result["success"],
                "text": result.get("text", ""),
                "error": result.get("error")
            }
        else:
            return {"file": file_path, "error": f"HTTP {response.status_code}"}

    except Exception as e:
        return {"file": file_path, "error": str(e)}

def process_batch(file_paths, api_key, max_workers=4):
    """Process multiple files concurrently"""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(
            lambda fp: process_single_file(fp, api_key),
            file_paths
        ))
    return results

# Usage
files = ["doc1.pdf", "doc2.png", "doc3.docx"]
results = process_batch(files, "YOUR_API_KEY_HERE", max_workers=4)

for result in results:
    if result.get("success"):
        print(f"✓ {result['file']}: {len(result['text'])} characters extracted")
    else:
        print(f"✗ {result['file']}: {result.get('error', 'Unknown error')}")
```

---

## Service Capabilities

### ✅ What This Service Does Well:

1. **German Language Support**: Perfect handling of äöüÄÖÜß
2. **Large Files**: Supports up to 250 MB with intelligent memory management
3. **Multiple Formats**: 15+ file formats supported
4. **Table Extraction**: Automatically detects and extracts tables
5. **Concurrent Processing**: Handles 4 simultaneous requests efficiently
6. **Smart Processing**: Automatic file type detection via MIME types
7. **Scanned Documents**: Handles both native text and scanned images

### ⚠️ Limitations:

1. **File Size**: 250 MB maximum per file
2. **Processing Time**: Large files may take 30+ seconds
3. **Image Quality**: Very low-quality scans may have reduced accuracy
4. **Handwriting**: Not optimized for handwritten text
5. **Complex Layouts**: Multi-column layouts may lose some formatting

---

## Troubleshooting

### Common Issues:

**"Invalid API key"**
- Verify the key parameter is included in your request
- Check for typos in the API key

**"File too large"**
- File exceeds 250 MB
- Compress or split the file

**"File type not supported"**
- Check the file extension is in the supported list
- Verify the file isn't corrupted

**Timeout errors**
- Large files take longer to process
- Increase your HTTP client timeout to 120+ seconds
- Consider splitting very large files

**Empty text returned**
- File may be blank or image-only with no text
- Image quality may be too low for OCR
- Check the confidence score in the response

---

## API Key Management

**Security Best Practices**:
1. Never commit API keys to version control
2. Use environment variables: `os.getenv('OCR_API_KEY')`
3. Rotate keys periodically
4. Use different keys for development and production
5. Monitor usage via the `/status` endpoint

---

## Rate Limiting & Performance

- **Concurrent Requests**: 4 simultaneous files processed in parallel
- **Queue Management**: Additional requests automatically queued
- **Resource Monitoring**: Check `/status` endpoint for current load
- **Recommended**: For high-volume use, limit concurrent requests to 4

---

## Support & Updates

**Service URL**: https://ocr-server-u27285.vm.elestio.app

**Health Check**: https://ocr-server-u27285.vm.elestio.app/health

**System Status**: https://ocr-server-u27285.vm.elestio.app/status

For issues or questions, check the health endpoint first to verify service availability.
