#!/bin/bash

# Local worker runner script
# Usage: ./scripts/run_worker_local.sh [worker_type]
# Example: ./scripts/run_worker_local.sh pytorch-2.1

set -e

WORKER_TYPE=${1:-pytorch-2.1}

echo "🤖 Starting AI worker locally (type: $WORKER_TYPE)..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install it first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check if Redis is running locally
if ! redis-cli ping &> /dev/null; then
    echo "❌ Redis is not running. Please start Redis first:"
    echo "   - macOS: brew install redis && brew services start redis"
    echo "   - Ubuntu: sudo apt install redis-server && sudo systemctl start redis"
    echo "   - Or run: redis-server"
    exit 1
fi

echo "✅ Redis is running"

# Navigate to ai-worker directory
cd ai-worker

# Set worker environment variables based on type
case $WORKER_TYPE in
    "pytorch-2.0")
        export WORKER_TYPE=pytorch-2.0
        export MODEL_FRAMEWORK=pytorch
        EXTRA=pytorch_2_0
        ;;
    "pytorch-2.1")
        export WORKER_TYPE=pytorch-2.1
        export MODEL_FRAMEWORK=pytorch
        EXTRA=pytorch_2_1
        ;;
    "pytorch-2.0-gpu")
        export WORKER_TYPE=pytorch-2.0-gpu
        export MODEL_FRAMEWORK=pytorch
        export USE_GPU=true
        EXTRA=pytorch_2_0_gpu
        ;;
    "pytorch-2.1-gpu")
        export WORKER_TYPE=pytorch-2.1-gpu
        export MODEL_FRAMEWORK=pytorch
        export USE_GPU=true
        EXTRA=pytorch_2_1_gpu
        ;;
    "tensorflow")
        export WORKER_TYPE=tensorflow
        export MODEL_FRAMEWORK=tensorflow
        EXTRA=tensorflow
        ;;
    "sklearn")
        export WORKER_TYPE=sklearn
        export MODEL_FRAMEWORK=sklearn
        EXTRA=sklearn
        ;;
    *)
        echo "❌ Unknown worker type: $WORKER_TYPE"
        echo "Available types: pytorch-2.0, pytorch-2.1, pytorch-2.0-gpu, pytorch-2.1-gpu, tensorflow, sklearn"
        exit 1
        ;;
esac

# Set common environment variables
export REDIS_HOST=localhost
export REDIS_PORT=6379
export REDIS_DB=0

echo "📦 Installing dependencies for $WORKER_TYPE..."
uv sync --extra $EXTRA

echo "🚀 Starting worker..."
uv run python run_worker.py
