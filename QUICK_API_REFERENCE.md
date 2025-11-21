# OCR API Quick Reference for LLMs

## Essential Information

**Base URL**: `https://ocr-server-u27285.vm.elestio.app`
**Endpoint**: `POST /ocr`
**Auth**: API key required in request
**Max File Size**: 250 MB
**Formats**: PDF, PNG, JPG, DOCX, RTF, ODT, TXT, CSV, HTML, XML, TIFF, BMP, GIF, WebP

---

## Quick Start

### Python (Multipart)
```python
import requests

response = requests.post(
    "https://ocr-server-u27285.vm.elestio.app/ocr",
    files={"file": open("document.pdf", "rb")},
    data={"key": "YOUR_API_KEY", "preserve_formatting": "true"}
)

text = response.json()["text"]
```

### Python (Base64)
```python
import requests, base64

with open("document.pdf", "rb") as f:
    content = base64.b64encode(f.read()).decode()

response = requests.post(
    "https://ocr-server-u27285.vm.elestio.app/ocr",
    json={
        "filename": "document.pdf",
        "content": content,
        "key": "YOUR_API_KEY",
        "preserve_formatting": True
    }
)

text = response.json()["text"]
```

### cURL
```bash
curl -X POST https://ocr-server-u27285.vm.elestio.app/ocr \
  -F "file=@document.pdf" \
  -F "key=YOUR_API_KEY" \
  -F "preserve_formatting=true"
```

---

## Response

```json
{
  "success": true,
  "text": "Extracted text...",
  "language": "de",
  "confidence": 0.95,
  "processing_time": 2.34,
  "tables_found": 1,
  "error": null
}
```

**Extract text**: `response.json()["text"]`
**Check success**: `response.json()["success"]`
**Handle error**: `response.json().get("error")`

---

## Key Points

✅ **DO**: Include API key, check file size < 250 MB, use timeout 120+ seconds
❌ **DON'T**: Send > 250 MB, forget API key, expect instant responses for large files

**Health check**: `GET https://ocr-server-u27285.vm.elestio.app/health`
**Concurrent**: Handles 4 files simultaneously, queues additional requests
