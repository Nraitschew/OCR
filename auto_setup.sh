#!/bin/bash
# Automatic Setup Script for OCR System
# This script sets up everything automatically without interrupting running processes

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}OCR System Automatic Setup${NC}"
echo -e "${BLUE}=====================================${NC}"

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

# Check if running with appropriate permissions
if [[ $EUID -eq 0 ]]; then
   print_error "Please don't run this script as root/sudo"
   print_info "The script will ask for sudo password when needed"
   exit 1
fi

# Step 1: Check Python installation
print_info "Checking Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    print_status "Python3 found: $PYTHON_VERSION"
else
    print_error "Python3 not found. Please install Python 3.8 or higher"
    exit 1
fi

# Step 2: Create virtual environment if not exists
if [ ! -d "venv" ]; then
    print_info "Creating virtual environment..."
    python3 -m venv venv
    print_status "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

# Step 3: Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate
print_status "Virtual environment activated"

# Step 4: Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip --quiet
print_status "Pip upgraded"

# Step 5: Install Python dependencies
print_info "Installing Python dependencies..."
pip install --upgrade -r requirements.txt --quiet
print_status "Python dependencies installed"

# Step 6: Install system dependencies (without interrupting processes)
print_info "Installing system dependencies (this may require sudo password)..."

# Check if running on Debian/Ubuntu
if [ -f /etc/debian_version ]; then
    print_info "Detected Debian/Ubuntu system"
    
    # Update package list without upgrading existing packages
    sudo apt-get update -qq
    
    # Install only missing packages without upgrading or removing anything
    PACKAGES="tesseract-ocr tesseract-ocr-deu tesseract-ocr-eng poppler-utils"
    PACKAGES="$PACKAGES libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev"
    PACKAGES="$PACKAGES libgomp1 wget fonts-dejavu-core fonts-liberation"
    
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
    print_error "Non-Debian system detected. Please install tesseract and other dependencies manually"
fi

# Step 7: Verify Tesseract installation
print_info "Verifying Tesseract installation..."
if command -v tesseract &> /dev/null; then
    TESS_VERSION=$(tesseract --version 2>&1 | head -n1)
    print_status "Tesseract found: $TESS_VERSION"
    
    # Check for language data
    if tesseract --list-langs 2>&1 | grep -q "deu"; then
        print_status "German language data found"
    else
        print_info "Installing German language data..."
        sudo apt-get install -y tesseract-ocr-deu -qq
    fi
else
    print_error "Tesseract not found. OCR functionality will be limited"
fi

# Step 8: Generate test documents
print_info "Generating test documents..."
python3 test_document_generator.py > /dev/null 2>&1
print_status "Test documents generated"

# Step 9: Create run script
cat > run_tests.sh << 'EOF'
#!/bin/bash
# Activate virtual environment and run tests
source venv/bin/activate
echo "Running comprehensive OCR tests..."
python3 run_comprehensive_test.py
EOF
chmod +x run_tests.sh
print_status "Test runner script created"

# Step 10: Create quick start script
cat > start_ocr.sh << 'EOF'
#!/bin/bash
# Quick start script for OCR system
source venv/bin/activate
python3 -c "
from minimal_ocr_system import MinimalOCRSystem
import asyncio

async def main():
    print('OCR System Ready!')
    print('Example usage:')
    print('  ocr = MinimalOCRSystem()')
    print('  result = await ocr.process_file(\"test_documents/german_text.txt\")')
    print('  print(result.text)')

asyncio.run(main())
"
python3 -i -c "
from minimal_ocr_system import MinimalOCRSystem
print('\nOCR system loaded. Use: ocr = MinimalOCRSystem()')
"
EOF
chmod +x start_ocr.sh
print_status "Quick start script created"

# Final summary
echo
echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}=====================================${NC}"
echo
echo "To run tests:"
echo -e "  ${BLUE}./run_tests.sh${NC}"
echo
echo "To start OCR system interactively:"
echo -e "  ${BLUE}./start_ocr.sh${NC}"
echo
echo "To use in your own scripts:"
echo -e "  ${BLUE}source venv/bin/activate${NC}"
echo -e "  ${BLUE}python3 your_script.py${NC}"
echo
print_info "No running processes were interrupted during setup"