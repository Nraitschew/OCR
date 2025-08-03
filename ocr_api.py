#!/usr/bin/env python3
"""
OCR API Service
Provides REST API for OCR processing on port 4000
"""

import os
import asyncio
import logging
from pathlib import Path
from datetime import datetime
import tempfile
import base64
import magic
import mimetypes
from typing import Optional, Dict, Any, Union, Tuple
from dotenv import load_dotenv

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import json

# Load environment variables
load_dotenv()

# Import OCR system
from cpu_only_ocr_system import CPUOnlyOCRSystem, get_resource_info

# Custom JSON response that doesn't escape Unicode characters
class UnicodeJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":")
        ).encode("utf-8")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="OCR Service API",
    description="CPU-optimized OCR service with German language support",
    version="1.0.0",
    default_response_class=UnicodeJSONResponse
)

# Initialize OCR system
ocr_system = CPUOnlyOCRSystem()

# Get API key from environment
API_KEY = os.getenv('KEY')
if not API_KEY:
    logger.warning("No API key found in environment. Authentication will be disabled.")


class OCRRequest(BaseModel):
    """Request model for base64 encoded files"""
    filename: str
    content: str  # base64 encoded file content
    preserve_formatting: bool = True
    key: str  # API key for authentication


class OCRResponse(BaseModel):
    """Response model for OCR results"""
    success: bool
    text: str
    language: str
    confidence: float
    processing_time: float
    formatting_preserved: bool
    tables_found: int
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "OCR API",
        "version": "1.0.0",
        "endpoints": {
            "POST /ocr": "Unified endpoint for all file uploads (multipart or base64)",
            "GET /health": "Health check endpoint",
            "GET /status": "System status and resource usage"
        },
        "supported_formats": [
            "pdf", "png", "jpg", "jpeg", "txt", "docx",
            "tiff", "tif", "bmp", "gif", "webp",
            "rtf", "odt", "html", "htm", "xml", "csv"
        ],
        "features": [
            "German language support (ä, ö, ü, Ä, Ö, Ü, ß)",
            "Line break and paragraph preservation",
            "Table extraction",
            "CPU-optimized processing"
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test OCR system
        test_text = "Test"
        test_file = Path("/tmp/health_test.txt")
        with open(test_file, "w") as f:
            f.write(test_text)
            
        result = await ocr_system.process_file(str(test_file))
        test_file.unlink()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "ocr_functional": result.text == test_text
        }
    except Exception as e:
        return UnicodeJSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


@app.get("/status")
async def system_status():
    """Get system status and resource usage"""
    resources = get_resource_info()
    
    return {
        "timestamp": datetime.now().isoformat(),
        "resources": {
            "cpu_percent": resources["cpu_percent"],
            "cpu_cores": resources["cpu_count"],
            "ram_mb": round(resources["ram_mb"], 2),
            "ram_gb": round(resources["ram_gb"], 2),
            "ram_percent": resources["ram_percent"],
            "threads": resources["threads"]
        },
        "limits": {
            "max_cpu_percent": 70,
            "max_ram_gb": 2,
            "max_file_size_mb": 50
        }
    }


def detect_file_type(content: bytes, filename: str = "") -> Tuple[str, str]:
    """
    Detect file type using magic numbers and filename extension
    Returns: (mime_type, file_extension)
    """
    # Try to detect MIME type using magic numbers
    try:
        mime = magic.Magic(mime=True)
        mime_type = mime.from_buffer(content)
    except:
        # Fallback to filename-based detection
        mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    
    # Map MIME types to extensions
    mime_to_ext = {
        'application/pdf': '.pdf',
        'image/png': '.png',
        'image/jpeg': '.jpg',
        'image/jpg': '.jpg',
        'image/tiff': '.tiff',
        'image/bmp': '.bmp',
        'image/gif': '.gif',
        'image/webp': '.webp',
        'text/plain': '.txt',
        'text/html': '.html',
        'text/xml': '.xml',
        'text/csv': '.csv',
        'application/rtf': '.rtf',
        'text/rtf': '.rtf',
        'application/vnd.oasis.opendocument.text': '.odt',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
        'application/xml': '.xml',
    }
    
    # Get extension from mime type or filename
    ext = mime_to_ext.get(mime_type, "")
    if not ext and filename:
        ext = Path(filename).suffix.lower()
    
    return mime_type, ext


@app.post("/ocr", response_model=OCRResponse)
async def unified_ocr(
    request: Request,
    # Optional file upload (multipart)
    file: Optional[UploadFile] = File(None),
    preserve_formatting: bool = Form(None),
    key: Optional[str] = Form(None),
):
    """
    Unified OCR endpoint - accepts both multipart file uploads and base64 encoded files
    Automatically detects file type and processes accordingly
    """
    
    # Check if request is JSON (base64)
    content_type = request.headers.get("content-type", "")
    
    if "application/json" in content_type:
        # Handle JSON request with base64
        try:
            json_data = await request.json()
            ocr_request = OCRRequest(**json_data)
            
            # Verify API key
            if API_KEY and ocr_request.key != API_KEY:
                raise HTTPException(status_code=401, detail="Invalid API key")
            
            contents = base64.b64decode(ocr_request.content)
            filename = ocr_request.filename
            preserve_fmt = ocr_request.preserve_formatting
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON/base64 data: {e}")
            
    elif file and file.filename:
        # Handle multipart file upload
        if not key:
            raise HTTPException(status_code=400, detail="API key required")
        
        # Verify API key
        if API_KEY and key != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")
        
        filename = file.filename
        contents = await file.read()
        preserve_fmt = preserve_formatting if preserve_formatting is not None else True
        
    else:
        raise HTTPException(
            status_code=400,
            detail="No file provided. Send either multipart file or base64 encoded content"
        )
    
    # Check file size (50MB limit)
    file_size_mb = len(contents) / (1024 * 1024)
    if file_size_mb > 50:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({file_size_mb:.1f}MB). Maximum size: 50MB"
        )
    
    # Detect file type
    mime_type, file_ext = detect_file_type(contents, filename)
    
    # Validate file type
    allowed_extensions = {
        ".pdf", ".png", ".jpg", ".jpeg", ".txt", ".docx",
        ".tiff", ".tif", ".bmp", ".gif", ".webp",
        ".rtf", ".odt", ".html", ".htm", ".xml", ".csv"
    }
    
    if file_ext not in allowed_extensions:
        # Try to get extension from filename if detection failed
        file_ext = Path(filename).suffix.lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type not supported. Detected: {mime_type}, Extension: {file_ext}. "
                       f"Allowed types: {sorted(allowed_extensions)}"
            )
    
    # Save temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        tmp_file.write(contents)
        tmp_path = tmp_file.name
    
    try:
        # Process with OCR
        result = await ocr_system.process_file(tmp_path)
        
        # Prepare response
        response = OCRResponse(
            success=True,
            text=result.text,
            language=result.language,
            confidence=result.confidence,
            processing_time=result.processing_time,
            formatting_preserved=preserve_fmt,
            tables_found=len(result.tables),
            metadata={
                "filename": filename,
                "detected_mime_type": mime_type,
                "file_extension": file_ext,
                "file_size_mb": round(file_size_mb, 2),
                "cpu_usage": result.cpu_usage,
                "ram_usage_mb": result.ram_usage_mb
            }
        )
        
        return response
        
    except Exception as e:
        logger.error(f"OCR processing failed: {e}")
        return OCRResponse(
            success=False,
            text="",
            language="unknown",
            confidence=0.0,
            processing_time=0.0,
            formatting_preserved=False,
            tables_found=0,
            error=str(e),
            metadata={
                "filename": filename,
                "detected_mime_type": mime_type,
                "file_extension": file_ext
            }
        )
    finally:
        # Clean up temporary file
        Path(tmp_path).unlink(missing_ok=True)


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return UnicodeJSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


def main():
    """Run the API server"""
    port = int(os.environ.get("OCR_API_PORT", 4000))
    host = os.environ.get("OCR_API_HOST", "0.0.0.0")
    
    logger.info(f"Starting OCR API on {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()