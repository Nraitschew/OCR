try:
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
except Exception:  # transformers or torch might be missing
    torch = None
    TrOCRProcessor = None
    VisionEncoderDecoderModel = None
import pytesseract

class AdaptiveModelLoader:
    def __init__(self, vram_limit_bytes):
        self.vram_limit = vram_limit_bytes
        self.gpu_models = {}
        self.cpu_models = {}

    def load_gpu_models(self):
        if torch is None or TrOCRProcessor is None:
            raise RuntimeError("GPU OCR libraries are unavailable")
        processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-printed')
        model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-printed')
        model.to('cuda')
        self.gpu_models['ocr'] = (processor, model)

    def load_cpu_models(self):
        # pytesseract acts as cpu model
        self.cpu_models['ocr'] = pytesseract

    def load_for_device(self, device='auto'):
        if device == 'auto':
            device = 'gpu' if torch is not None and torch.cuda.is_available() else 'cpu'

        if device == 'gpu':
            try:
                if not self.gpu_models:
                    self.load_gpu_models()
                return self.gpu_models
            except Exception:
                device = 'cpu'

        if not self.cpu_models:
            self.load_cpu_models()
        return self.cpu_models
