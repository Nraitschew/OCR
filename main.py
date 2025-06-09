import sys
from pathlib import Path
from ocr_pipeline.ocr_pipeline import OCRPipeline


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_file>")
        return
    image_path = Path(sys.argv[1])
    data = image_path.read_bytes()
    pipeline = OCRPipeline()
    result = pipeline.process_document(data)
    print(result)

if __name__ == '__main__':
    main()
