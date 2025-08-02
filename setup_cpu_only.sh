#!/bin/bash
# CPU-Only OCR System Setup Script
# No GPU dependencies, optimized for CPU usage

set -e  # Exit on error

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'
BOLD='\033[1m'

echo -e "${BLUE}${BOLD}=====================================${NC}"
echo -e "${BLUE}${BOLD}CPU-Only OCR System Setup${NC}"
echo -e "${BLUE}${BOLD}=====================================${NC}"
echo -e "${YELLOW}No GPU dependencies will be installed${NC}"
echo

# Function to print status
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Check Python
print_info "Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    print_status "Python3 found: $PYTHON_VERSION"
else
    print_error "Python3 not found. Please install Python 3.8 or higher"
    exit 1
fi

# Create virtual environment
if [ ! -d "venv_cpu" ]; then
    print_info "Creating CPU-optimized virtual environment..."
    python3 -m venv venv_cpu
    print_status "Virtual environment created (venv_cpu)"
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv_cpu/bin/activate
print_status "Virtual environment activated"

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip --quiet
print_status "Pip upgraded"

# Install CPU-only Python dependencies
print_info "Installing CPU-only Python dependencies..."
pip install --upgrade -r requirements_cpu.txt --quiet

# Additional CPU-specific optimizations
pip install --upgrade --quiet \
    pytesseract \
    opencv-python-headless \
    pdfplumber \
    fast-langdetect

print_status "CPU-only dependencies installed"

# Install system dependencies
print_info "Installing system dependencies..."

if [ -f /etc/debian_version ]; then
    print_info "Detected Debian/Ubuntu system"
    
    # Update package list
    sudo apt-get update -qq
    
    # Essential packages for CPU-only OCR
    PACKAGES="tesseract-ocr tesseract-ocr-deu tesseract-ocr-eng"
    PACKAGES="$PACKAGES poppler-utils ghostscript"
    PACKAGES="$PACKAGES libgl1-mesa-glx libglib2.0-0"
    PACKAGES="$PACKAGES fonts-dejavu-core fonts-liberation"
    
    for package in $PACKAGES; do
        if ! dpkg -l | grep -q "^ii  $package "; then
            print_info "Installing $package..."
            sudo apt-get install -y --no-upgrade $package -qq
        else
            print_status "$package already installed"
        fi
    done
    
    print_status "System dependencies installed"
else
    print_error "Non-Debian system detected. Please install tesseract manually"
fi

# Verify Tesseract
print_info "Verifying Tesseract installation..."
if command -v tesseract &> /dev/null; then
    TESS_VERSION=$(tesseract --version 2>&1 | head -n1)
    print_status "Tesseract found: $TESS_VERSION"
    
    # Check language support
    LANGS=$(tesseract --list-langs 2>&1)
    if echo "$LANGS" | grep -q "deu"; then
        print_status "German language support available"
    else
        print_info "Installing German language data..."
        sudo apt-get install -y tesseract-ocr-deu -qq
    fi
else
    print_error "Tesseract not found. OCR will not work!"
    exit 1
fi

# Generate test documents if needed
if [ ! -d "test_documents" ] || [ $(ls test_documents | wc -l) -lt 10 ]; then
    print_info "Generating test documents..."
    python3 test_document_generator.py > /dev/null 2>&1
    print_status "Test documents generated"
fi

# Create CPU test runner
cat > run_cpu_tests.sh << 'EOF'
#!/bin/bash
# Run CPU-only OCR tests
source venv_cpu/bin/activate
echo -e "\033[0;34mRunning CPU-only OCR tests...\033[0m"
python3 test_cpu_only_system.py
EOF
chmod +x run_cpu_tests.sh

# Create quick start script
cat > start_cpu_ocr.sh << 'EOF'
#!/bin/bash
# Start CPU-only OCR system
source venv_cpu/bin/activate
python3 -c "
from cpu_only_ocr_system import CPUOnlyOCRSystem, get_resource_info
import asyncio

print('CPU-Only OCR System Ready!')
print(f'CPU cores: {get_resource_info()[\"cpu_count\"]}')
print('No GPU dependencies loaded')
"
python3 -i -c "
from cpu_only_ocr_system import CPUOnlyOCRSystem
print('\nUse: ocr = CPUOnlyOCRSystem()')
print('Example: result = await ocr.process_file(\"document.pdf\")')
"
EOF
chmod +x start_cpu_ocr.sh

# Summary
echo
echo -e "${GREEN}${BOLD}=====================================${NC}"
echo -e "${GREEN}${BOLD}CPU-Only Setup Complete!${NC}"
echo -e "${GREEN}${BOLD}=====================================${NC}"
echo
echo "System Configuration:"
echo "  - OCR Engine: Tesseract (CPU-optimized)"
echo "  - No GPU dependencies installed"
echo "  - Resource limits: CPU 70%, RAM 70% (2GB soft limit)"
echo "  - Max workers: $(python3 -c 'import multiprocessing; print(min(4, multiprocessing.cpu_count()-1))')"
echo
echo "To run CPU tests:"
echo -e "  ${BLUE}./run_cpu_tests.sh${NC}"
echo
echo "To start OCR system:"
echo -e "  ${BLUE}./start_cpu_ocr.sh${NC}"
echo
echo "To use in scripts:"
echo -e "  ${BLUE}source venv_cpu/bin/activate${NC}"
echo -e "  ${BLUE}python3 your_script.py${NC}"
echo
print_info "CPU-only system ready for use!"