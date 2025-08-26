#!/bin/bash

# Development setup script for AI Job Queue System

set -e

echo "🚀 Setting up AI Job Queue System for development..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install it first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "✅ Prerequisites check passed"

# Setup backend
echo "📦 Setting up backend..."
cd backend
uv sync
echo "✅ Backend dependencies installed"

# Setup AI worker
echo "🤖 Setting up AI worker..."
cd ../ai-worker
uv sync
echo "✅ AI worker dependencies installed"

# Return to root
cd ..

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cat > .env << EOF
# Development environment variables
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Backend settings
BACKEND_PORT=8000

# AI Worker settings
MODEL_STORAGE_PATH=./models
EOF
    echo "✅ .env file created"
fi

# Create models directory
mkdir -p models

echo ""
echo "🎉 Development setup complete!"
echo ""
echo "Next steps:"
echo "1. Start Redis: docker run -d -p 6379:6379 redis:7-alpine"
echo "2. Start backend: cd backend && uv run python -m uvicorn src.main:app --reload"
echo "3. Start AI worker: cd ai-worker && uv run python src/worker.py"
echo ""
echo "Or use Docker Compose for everything:"
echo "   docker-compose up -d"
echo ""
echo "Test the API:"
echo "   python scripts/test_api.py"
