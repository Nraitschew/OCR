#!/bin/bash

# Script to ensure a completely fresh build from GitHub

echo "🔄 Starting fresh build from GitHub repository..."

# Stop and remove existing containers
echo "🛑 Stopping existing containers..."
docker-compose down

# Remove the existing image to force rebuild
echo "🗑️  Removing existing image..."
docker rmi $(docker images -q ocr-ocr-api) 2>/dev/null || true

# Build with no cache to ensure fresh clone from GitHub
echo "🏗️  Building fresh image from GitHub..."
docker-compose build --no-cache

# Start the service
echo "🚀 Starting OCR service..."
docker-compose up -d

# Show logs
echo "📋 Showing logs (press Ctrl+C to exit)..."
docker-compose logs -f