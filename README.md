# OCR Pipeline

This repository provides a minimal OCR pipeline demonstrating GPU/CPU fallback.
It uses a lightweight GPU model (TrOCR) if CUDA and the required libraries are
available and falls back to Tesseract otherwise. The code tries to operate even
when optional dependencies like `torch`, `transformers` or `langdetect` are not
installed.

## Usage

```bash
pip install -r requirements.txt
python main.py path_to_image.png
```

If GPU-related packages fail to install you can still run the CPU mode with the
base dependencies that come with this repository.
