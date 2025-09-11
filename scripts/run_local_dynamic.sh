#!/bin/bash

# Local dynamic worker development script
# No Docker required - uses local processes

set -e

echo "🚀 Starting AI Job Queue System with Local Dynamic Workers..."

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
    kill $BACKEND_PID $WORKER_MANAGER_PID 2>/dev/null || true
    # Kill any worker processes
    pkill -f "run_worker.py" 2>/dev/null || true
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

# Start local worker manager
echo "🤖 Starting Local Worker Manager..."
cd worker-manager
uv run python src/local_worker_manager.py &
WORKER_MANAGER_PID=$!
cd ..

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 5

# Check service health
echo "🔍 Checking service health..."

# Check Backend
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Backend API is healthy"
else
    echo "❌ Backend API is not responding"
    exit 1
fi

# Check Worker Manager
if curl -f http://localhost:8001/workers/status > /dev/null 2>&1; then
    echo "✅ Local Worker Manager is healthy"
else
    echo "❌ Local Worker Manager is not responding"
    exit 1
fi

echo ""
echo "🎉 AI Job Queue System with Local Dynamic Workers is running!"
echo ""
echo "📊 Service URLs:"
echo "  - Backend API: http://localhost:8000"
echo "  - Worker Manager: http://localhost:8001"
echo "  - Redis: localhost:6379"
echo ""
echo "🔧 Management Commands:"
echo "  - Worker status: curl http://localhost:8001/workers/status"
echo "  - Start worker: curl -X POST http://localhost:8001/workers/start/pytorch-2.1"
echo "  - Stop worker: curl -X POST http://localhost:8001/workers/stop/{worker_id}"
echo ""
echo "📝 Example Usage:"
echo "  # Submit a training job (will spawn a worker automatically)"
echo "  curl -X POST http://localhost:8000/jobs/training \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model_type\": \"bert\", \"data_path\": \"/data/train.csv\", \"requires_gpu\": false}'"
echo ""
echo "  # Check worker status"
echo "  curl http://localhost:8001/workers/status"
echo ""
echo "💡 Workers will be spawned automatically when jobs are submitted!"
echo "   They will be cleaned up after being idle for 5-10 minutes."
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for background processes
wait
