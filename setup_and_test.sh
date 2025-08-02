#!/bin/bash
# Complete Setup and Test Script for Enhanced OCR System

set -e  # Exit on error

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'
BOLD='\033[1m'

echo -e "${BLUE}${BOLD}=====================================${NC}"
echo -e "${BLUE}${BOLD}Enhanced OCR System Setup & Test${NC}"
echo -e "${BLUE}${BOLD}=====================================${NC}"
echo

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Step 1: Run automatic setup
echo -e "${BLUE}Step 1: Running automatic setup...${NC}"
if [ -f "auto_setup.sh" ]; then
    ./auto_setup.sh
else
    echo -e "${RED}Error: auto_setup.sh not found${NC}"
    exit 1
fi

echo
echo -e "${GREEN}✓ Setup completed successfully${NC}"
echo

# Step 2: Activate virtual environment
echo -e "${BLUE}Step 2: Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo

# Step 3: Install additional dependencies for enhanced features
echo -e "${BLUE}Step 3: Installing enhanced dependencies...${NC}"
pip install --quiet pdfplumber tabulate

# Check if all imports work
python3 -c "
import enhanced_ocr_system
import enhanced_test_generator
import run_all_tests
print('✓ All modules imported successfully')
" || {
    echo -e "${RED}Error: Failed to import modules${NC}"
    exit 1
}

echo -e "${GREEN}✓ Enhanced dependencies installed${NC}"
echo

# Step 4: Generate enhanced test documents
echo -e "${BLUE}Step 4: Generating enhanced test documents...${NC}"
python3 enhanced_test_generator.py
echo -e "${GREEN}✓ Enhanced test documents generated${NC}"
echo

# Step 5: Run comprehensive tests
echo -e "${BLUE}Step 5: Running comprehensive tests...${NC}"
echo -e "${YELLOW}This will test:${NC}"
echo "  - Basic OCR functionality"
echo "  - Complex table extraction"
echo "  - German umlaut support (ä, ö, ü, Ä, Ö, Ü, ß)"
echo "  - Invoice and financial documents"
echo "  - Scientific documents with formulas"
echo "  - Resource management (CPU/RAM limits)"
echo "  - Performance and batch processing"
echo "  - Edge cases and error handling"
echo

# Give user a moment to read
sleep 2

# Run the comprehensive test
python3 run_all_tests.py

# Check test results
if [ -f "test_results.json" ]; then
    echo
    echo -e "${GREEN}${BOLD}All tests completed!${NC}"
    echo
    
    # Extract summary from JSON
    PASSED=$(python3 -c "import json; data=json.load(open('test_results.json')); print(data['passed'])")
    TOTAL=$(python3 -c "import json; data=json.load(open('test_results.json')); print(data['total_tests'])")
    
    if [ "$PASSED" -eq "$TOTAL" ]; then
        echo -e "${GREEN}${BOLD}✓ All $TOTAL tests PASSED!${NC}"
    else
        FAILED=$((TOTAL - PASSED))
        echo -e "${YELLOW}${BOLD}⚠ $PASSED/$TOTAL tests passed ($FAILED failed)${NC}"
    fi
else
    echo -e "${RED}Warning: Test results file not found${NC}"
fi

echo
echo -e "${BLUE}${BOLD}Setup and testing complete!${NC}"
echo
echo "To use the OCR system:"
echo -e "  1. Activate venv: ${BLUE}source venv/bin/activate${NC}"
echo -e "  2. Run OCR: ${BLUE}python3 -c \"from enhanced_ocr_system import EnhancedOCRSystem; ...\"${NC}"
echo
echo "To run tests again:"
echo -e "  ${BLUE}./run_tests.sh${NC}"
echo
echo "Resource limits configured:"
echo "  - CPU: 70% maximum"
echo "  - RAM: 70% maximum (2GB soft limit)"
echo
echo -e "${GREEN}${BOLD}The system is ready to use!${NC}"