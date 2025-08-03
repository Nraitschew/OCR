FROM python:3.9-slim

# Install system dependencies including git
RUN apt-get update && apt-get install -y \
    git \
    tesseract-ocr \
    tesseract-ocr-deu \
    tesseract-ocr-eng \
    libtesseract-dev \
    libpoppler-cpp-dev \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install additional fonts for better OCR
RUN apt-get update && apt-get install -y \
    fonts-dejavu-core \
    fonts-dejavu-extra \
    fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy local files
COPY . /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_api.txt

# Create necessary directories
RUN mkdir -p logs temp test_linebreaks

# Set permissions
RUN chmod +x auto_setup.sh setup_cpu_only.sh || true

# Expose port
EXPOSE 4000

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:4000/health || exit 1

# Run the API service
CMD ["python", "ocr_api.py"]