"""
Configuration module for OCR system with resource constraints
"""
import os
import multiprocessing

# ============================================
# RESOURCE CONSTRAINTS - EASY TO CONFIGURE
# ============================================

# Memory constraints
MAX_MEMORY_GB = 2.0  # Maximum memory in GB
TARGET_MEMORY_PERCENT = 70  # Target memory usage percentage

# CPU constraints  
TARGET_CPU_PERCENT = 70  # Target CPU usage percentage
MAX_WORKERS = multiprocessing.cpu_count()  # Maximum number of workers

# GPU settings
USE_GPU = True  # Enable GPU if available
GPU_FALLBACK_TO_CPU = True  # Fallback to CPU if GPU fails

# ============================================
# OCR SETTINGS
# ============================================

# Supported languages
OCR_LANGUAGES = ['en', 'de']  # English and German
DEFAULT_LANGUAGE = 'de'  # Default to German for umlaut support

# Supported file formats
SUPPORTED_FORMATS = ['.pdf', '.png', '.jpg', '.jpeg', '.docx', '.txt']

# OCR Engine settings
EASYOCR_CONFIG = {
    'gpu': USE_GPU,
    'model_storage_directory': './models',
    'download_enabled': True,
    'verbose': False
}

TESSERACT_CONFIG = {
    'lang': 'deu+eng',  # German + English
    'config': '--oem 3 --psm 6'  # LSTM engine with uniform block of text
}

PADDLEOCR_CONFIG = {
    'use_gpu': USE_GPU,
    'lang': 'german',
    'use_angle_cls': True,
    'show_log': False
}

# ============================================
# PROCESSING SETTINGS
# ============================================

# Image processing
IMAGE_MAX_SIZE = (4096, 4096)  # Maximum image dimensions
IMAGE_DPI = 300  # DPI for PDF to image conversion

# Table detection
TABLE_DETECTION_ENABLED = True
TABLE_MIN_ROWS = 2  # Minimum rows to consider as table

# Output settings
OUTPUT_FORMAT = 'markdown'  # Output format
PRESERVE_FORMATTING = True  # Preserve original formatting
INCLUDE_CONFIDENCE_SCORES = True  # Include OCR confidence scores

# ============================================
# PERFORMANCE SETTINGS
# ============================================

# Batch processing
BATCH_SIZE = 10  # Number of pages to process in batch
QUEUE_MAX_SIZE = 100  # Maximum queue size for requests

# Timeouts
OCR_TIMEOUT_SECONDS = 300  # 5 minutes timeout per document
REQUEST_TIMEOUT_SECONDS = 600  # 10 minutes total timeout

# Caching
ENABLE_CACHE = True
CACHE_SIZE_MB = 500  # Cache size in MB

# ============================================
# PATHS
# ============================================

# Directory paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DOCUMENTS_DIR = os.path.join(BASE_DIR, 'test_documents')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
CACHE_DIR = os.path.join(BASE_DIR, '.cache')

# Create directories if they don't exist
for directory in [TEST_DOCUMENTS_DIR, OUTPUT_DIR, MODELS_DIR, CACHE_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================
# LOGGING
# ============================================

LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = os.path.join(BASE_DIR, 'ocr_system.log')