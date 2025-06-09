# OCR Pipeline

This repository provides a minimal OCR pipeline demonstrating GPU/CPU fallback.
It uses a lightweight GPU model (TrOCR) if CUDA is available and falls back to
Tesseract otherwise.

## Usage

```bash
pip install -r requirements.txt
python main.py path_to_image.png
```
