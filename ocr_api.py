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
from typing import Optional, Dict, Any

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Import OCR system
from cpu_only_ocr_system import CPUOnlyOCRSystem, get_resource_info

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
    version="1.0.0"
)

# Initialize OCR system
ocr_system = CPUOnlyOCRSystem()


class OCRRequest(BaseModel):
    """Request model for base64 encoded files"""
    filename: str
    content: str  # base64 encoded file content
    preserve_formatting: bool = True


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
            "POST /ocr/file": "Upload file for OCR processing",
            "POST /ocr/base64": "Send base64 encoded file for OCR",
            "GET /health": "Health check endpoint",
            "GET /status": "System status and resource usage"
        },
        "supported_formats": ["pdf", "png", "jpg", "jpeg", "txt", "docx"],
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
        return JSONResponse(
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


@app.post("/ocr/file", response_model=OCRResponse)
async def ocr_file(
    file: UploadFile = File(...),
    preserve_formatting: bool = Form(True)
):
    """Process uploaded file with OCR"""
    
    # Validate file type
    allowed_extensions = {".pdf", ".png", ".jpg", ".jpeg", ".txt", ".docx"}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file_ext} not supported. Allowed types: {allowed_extensions}"
        )
    
    # Check file size (50MB limit)
    contents = await file.read()
    file_size_mb = len(contents) / (1024 * 1024)
    
    if file_size_mb > 50:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({file_size_mb:.1f}MB). Maximum size: 50MB"
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
            formatting_preserved=preserve_formatting,
            tables_found=len(result.tables),
            metadata={
                "filename": file.filename,
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
            error=str(e)
        )
    finally:
        # Clean up temporary file
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/ocr/base64", response_model=OCRResponse)
async def ocr_base64(request: OCRRequest):
    """Process base64 encoded file with OCR"""
    
    # Decode base64 content
    try:
        file_content = base64.b64decode(request.content)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid base64 encoding: {e}"
        )
    
    # Determine file extension
    file_ext = Path(request.filename).suffix.lower()
    allowed_extensions = {".pdf", ".png", ".jpg", ".jpeg", ".txt", ".docx"}
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file_ext} not supported"
        )
    
    # Check file size
    file_size_mb = len(file_content) / (1024 * 1024)
    if file_size_mb > 50:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({file_size_mb:.1f}MB). Maximum size: 50MB"
        )
    
    # Save temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        tmp_file.write(file_content)
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
            formatting_preserved=request.preserve_formatting,
            tables_found=len(result.tables),
            metadata={
                "filename": request.filename,
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
            error=str(e)
        )
    finally:
        # Clean up
        Path(tmp_path).unlink(missing_ok=True)


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
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