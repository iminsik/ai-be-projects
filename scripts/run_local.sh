#!/bin/bash

# Local development script - No Docker required
set -e

echo "🚀 Starting AI Job Queue System (Local Mode)"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install it first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check if Redis is running locally
if ! redis-cli ping &> /dev/null; then
    echo "⚠️  Redis is not running. Starting Redis..."
    echo "   Please install Redis first:"
    echo "   - macOS: brew install redis && brew services start redis"
    echo "   - Ubuntu: sudo apt install redis-server && sudo systemctl start redis"
    echo "   - Or run: redis-server"
    exit 1
fi

echo "✅ Redis is running"

# Create models directory
mkdir -p models

# Function to cleanup background processes
cleanup() {
    echo "🛑 Shutting down services..."
    kill $BACKEND_PID $WORKER_PID 2>/dev/null || true
    exit 0
}

# Set trap to cleanup on exit
trap cleanup SIGINT SIGTERM

# Start backend
echo "📦 Starting FastAPI backend..."
cd backend
uv run python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
cd ..

# Wait a moment for backend to start
sleep 3

# Start AI worker
echo "🤖 Starting AI worker..."
cd ai-worker
uv run python run_worker.py &
WORKER_PID=$!
cd ..

echo ""
echo "🎉 System is running!"
echo ""
echo "Services:"
echo "  - Backend: http://localhost:8000"
echo "  - API Docs: http://localhost:8000/docs"
echo "  - Redis: localhost:6379"
echo ""
echo "Test the system:"
echo "  python scripts/test_api.py"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for background processes
wait
