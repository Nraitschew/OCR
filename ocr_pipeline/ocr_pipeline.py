import torch
from .device_manager import DeviceManager
from .adaptive_model_loader import AdaptiveModelLoader
from .language_detector import LanguageDetector
from PIL import Image
import io

class OCRPipeline:
    def __init__(self, vram_limit_gb=1):
        self.vram_limit_bytes = int(vram_limit_gb * 1024 ** 3)
        self.device_manager = DeviceManager(vram_threshold=self.vram_limit_bytes)
        self.model_loader = AdaptiveModelLoader(self.vram_limit_bytes)
        self.language_detector = LanguageDetector()

    def process_document(self, image_bytes):
        device = self.device_manager.select_optimal_device()
        models = self.model_loader.load_for_device(device)
        if device == 'gpu':
            processor, model = models['ocr']
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            pixel_values = processor(images=image, return_tensors='pt').pixel_values
            pixel_values = pixel_values.to('cuda')
            with torch.no_grad():
                generated_ids = model.generate(pixel_values)
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        else:
            text = models['ocr'].image_to_string(Image.open(io.BytesIO(image_bytes)))

        lang = self.language_detector.detect_language(text)
        return {"text": text, "language": lang}
