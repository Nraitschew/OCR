"""
Core OCR engine with GPU/CPU fallback support
"""
import os
import time
import logging
from typing import List, Dict, Optional, Union, Tuple
import numpy as np
from PIL import Image
import torch

from .config import (
    USE_GPU, GPU_FALLBACK_TO_CPU, OCR_LANGUAGES,
    EASYOCR_CONFIG, TESSERACT_CONFIG, PADDLEOCR_CONFIG
)
from .resource_manager import ResourceManager, ResourceContext

logger = logging.getLogger(__name__)


class OCREngine:
    """Main OCR engine with automatic GPU/CPU fallback"""
    
    def __init__(self, resource_manager: ResourceManager):
        self.resource_manager = resource_manager
        self.device = self._detect_device()
        self.engines = {}
        self._initialize_engines()
        
    def _detect_device(self) -> str:
        """Detect available device (GPU/CPU)"""
        if USE_GPU and torch.cuda.is_available():
            logger.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
            return 'cuda'
        else:
            logger.info("Using CPU for OCR processing")
            return 'cpu'
            
    def _initialize_engines(self):
        """Initialize OCR engines based on available resources"""
        # Try to initialize EasyOCR (preferred for GPU)
        if self._init_easyocr():
            self.engines['easyocr'] = True
            
        # Always initialize Tesseract as fallback
        if self._init_tesseract():
            self.engines['tesseract'] = True
            
        # Initialize PaddleOCR for table detection
        if self._init_paddleocr():
            self.engines['paddleocr'] = True
            
        if not self.engines:
            raise RuntimeError("No OCR engines could be initialized")
            
        logger.info(f"Initialized engines: {list(self.engines.keys())}")
        
    def _init_easyocr(self) -> bool:
        """Initialize EasyOCR engine"""
        try:
            import easyocr
            
            # Configure GPU usage
            gpu_use = self.device == 'cuda' and EASYOCR_CONFIG['gpu']
            
            self.easyocr_reader = easyocr.Reader(
                OCR_LANGUAGES,
                gpu=gpu_use,
                model_storage_directory=EASYOCR_CONFIG['model_storage_directory'],
                download_enabled=EASYOCR_CONFIG['download_enabled'],
                verbose=EASYOCR_CONFIG['verbose']
            )
            logger.info(f"EasyOCR initialized (GPU: {gpu_use})")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to initialize EasyOCR: {e}")
            return False
            
    def _init_tesseract(self) -> bool:
        """Initialize Tesseract OCR"""
        try:
            import pytesseract
            
            # Test if tesseract is installed
            pytesseract.get_tesseract_version()
            self.tesseract = pytesseract
            logger.info("Tesseract OCR initialized")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to initialize Tesseract: {e}")
            return False
            
    def _init_paddleocr(self) -> bool:
        """Initialize PaddleOCR for table detection"""
        try:
            from paddleocr import PaddleOCR
            
            use_gpu = self.device == 'cuda' and PADDLEOCR_CONFIG['use_gpu']
            
            self.paddleocr = PaddleOCR(
                use_gpu=use_gpu,
                lang=PADDLEOCR_CONFIG['lang'],
                use_angle_cls=PADDLEOCR_CONFIG['use_angle_cls'],
                show_log=PADDLEOCR_CONFIG['show_log']
            )
            logger.info(f"PaddleOCR initialized (GPU: {use_gpu})")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to initialize PaddleOCR: {e}")
            return False
            
    def process_image(self, image: Union[str, np.ndarray, Image.Image], 
                     detect_tables: bool = True) -> Dict:
        """
        Process image with OCR
        
        Args:
            image: Image path, numpy array, or PIL Image
            detect_tables: Whether to detect tables
            
        Returns:
            Dictionary with OCR results
        """
        start_time = time.time()
        
        # Convert image to appropriate format
        if isinstance(image, str):
            pil_image = Image.open(image)
            np_image = np.array(pil_image)
        elif isinstance(image, Image.Image):
            pil_image = image
            np_image = np.array(image)
        else:
            np_image = image
            pil_image = Image.fromarray(image)
            
        result = {
            'text': '',
            'confidence': 0.0,
            'language': '',
            'tables': [],
            'engine_used': '',
            'processing_time': 0.0,
            'resource_usage': {}
        }
        
        # Use resource context to ensure we have enough resources
        with ResourceContext(self.resource_manager, memory_mb=200):
            # Try EasyOCR first if available
            if 'easyocr' in self.engines:
                try:
                    result = self._process_with_easyocr(np_image, result)
                    result['engine_used'] = 'easyocr'
                except Exception as e:
                    logger.error(f"EasyOCR failed: {e}")
                    if GPU_FALLBACK_TO_CPU and 'tesseract' in self.engines:
                        result = self._process_with_tesseract(pil_image, result)
                        result['engine_used'] = 'tesseract'
                        
            elif 'tesseract' in self.engines:
                result = self._process_with_tesseract(pil_image, result)
                result['engine_used'] = 'tesseract'
                
            # Detect tables if requested
            if detect_tables and 'paddleocr' in self.engines:
                try:
                    tables = self._detect_tables(np_image)
                    result['tables'] = tables
                except Exception as e:
                    logger.error(f"Table detection failed: {e}")
                    
        # Record final metrics
        result['processing_time'] = time.time() - start_time
        result['resource_usage'] = self.resource_manager.get_current_usage()
        
        return result
        
    def _process_with_easyocr(self, image: np.ndarray, result: Dict) -> Dict:
        """Process image with EasyOCR"""
        ocr_results = self.easyocr_reader.readtext(image)
        
        texts = []
        confidences = []
        
        for (bbox, text, confidence) in ocr_results:
            texts.append(text)
            confidences.append(confidence)
            
        result['text'] = ' '.join(texts)
        result['confidence'] = np.mean(confidences) if confidences else 0.0
        result['details'] = ocr_results
        
        return result
        
    def _process_with_tesseract(self, image: Image.Image, result: Dict) -> Dict:
        """Process image with Tesseract"""
        # Get text
        text = self.tesseract.image_to_string(
            image,
            lang=TESSERACT_CONFIG['lang'],
            config=TESSERACT_CONFIG['config']
        )
        
        # Get detailed data including confidence
        data = self.tesseract.image_to_data(
            image,
            lang=TESSERACT_CONFIG['lang'],
            output_type=self.tesseract.Output.DICT
        )
        
        # Calculate average confidence
        confidences = [int(c) for c in data['conf'] if int(c) > 0]
        avg_confidence = np.mean(confidences) / 100 if confidences else 0.0
        
        result['text'] = text.strip()
        result['confidence'] = avg_confidence
        result['details'] = data
        
        return result
        
    def _detect_tables(self, image: np.ndarray) -> List[Dict]:
        """Detect and extract tables from image"""
        tables = []
        
        try:
            # Use PaddleOCR for structure detection
            ocr_results = self.paddleocr.ocr(image, cls=True)
            
            # Process results to identify table structures
            # This is a simplified version - real implementation would need
            # more sophisticated table detection algorithms
            if ocr_results and ocr_results[0]:
                # Group text by vertical position to identify rows
                rows = {}
                for line in ocr_results[0]:
                    bbox = line[0]
                    text = line[1][0]
                    confidence = line[1][1]
                    
                    # Get approximate row position
                    y_pos = int(bbox[0][1] / 20) * 20  # Round to nearest 20 pixels
                    
                    if y_pos not in rows:
                        rows[y_pos] = []
                    rows[y_pos].append({
                        'text': text,
                        'bbox': bbox,
                        'confidence': confidence
                    })
                    
                # Convert to table if multiple rows found
                if len(rows) > 1:
                    table_data = []
                    for y_pos in sorted(rows.keys()):
                        row_cells = sorted(rows[y_pos], key=lambda x: x['bbox'][0][0])
                        table_data.append([cell['text'] for cell in row_cells])
                        
                    tables.append({
                        'data': table_data,
                        'confidence': np.mean([cell['confidence'] for row in rows.values() for cell in row])
                    })
                    
        except Exception as e:
            logger.error(f"Table detection error: {e}")
            
        return tables
        
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        return OCR_LANGUAGES
        
    def shutdown(self):
        """Cleanup resources"""
        # EasyOCR and PaddleOCR handle their own cleanup
        pass