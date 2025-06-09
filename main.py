import sys
from pathlib import Path
from ocr_pipeline.ocr_pipeline import OCRPipeline


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_file>")
        return
    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"File not found: {image_path}")
        return

    data = image_path.read_bytes()
    pipeline = OCRPipeline()
    result = pipeline.process_document(data)
    print(result)

if __name__ == '__main__':
    main()
