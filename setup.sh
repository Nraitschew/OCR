#!/bin/bash
# Setup script for OCR System

echo "Setting up OCR System..."

# Update package list
sudo apt-get update

# Install system dependencies
echo "Installing system dependencies..."
sudo apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-deu \
    tesseract-ocr-eng \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    fonts-dejavu-core \
    fonts-liberation

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Download font for better rendering
if [ ! -f "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf" ]; then
    echo "Installing DejaVu fonts..."
    sudo apt-get install -y fonts-dejavu
fi

echo "Setup complete!"
echo ""
echo "To run tests: python test_ocr_system.py"
echo "To generate test documents: python test_document_generator.py"