import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import pytesseract

class AdaptiveModelLoader:
    def __init__(self, vram_limit_bytes):
        self.vram_limit = vram_limit_bytes
        self.gpu_models = {}
        self.cpu_models = {}

    def load_gpu_models(self):
        processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-printed')
        model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-printed')
        model.to('cuda')
        self.gpu_models['ocr'] = (processor, model)

    def load_cpu_models(self):
        # pytesseract acts as cpu model
        self.cpu_models['ocr'] = pytesseract

    def load_for_device(self, device='auto'):
        if device == 'gpu':
            if not self.gpu_models:
                self.load_gpu_models()
            return self.gpu_models
        else:
            if not self.cpu_models:
                self.load_cpu_models()
            return self.cpu_models
