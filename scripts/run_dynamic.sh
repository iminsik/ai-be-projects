#!/bin/bash

# Script to run the AI Job Queue System with dynamic worker spawning

set -e

echo "🚀 Starting AI Job Queue System with Dynamic Workers..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Use the docker-compose compatibility script
COMPOSE_CMD="./scripts/docker-compose.sh"

# Test if the compose command works
if ! $COMPOSE_CMD version &> /dev/null; then
    echo "❌ Docker Compose is not available. Please install Docker Compose first."
    echo "   - Docker Compose V1: https://docs.docker.com/compose/install/"
    echo "   - Docker Compose V2: https://docs.docker.com/compose/install/compose-plugin/"
    exit 1
fi

# Create network if it doesn't exist
echo "📡 Creating network..."
docker network create ai-job-queue-network 2>/dev/null || true

# Build worker images first (needed for dynamic spawning)
echo "🔨 Building worker images..."
docker build -t ai-worker-pytorch-2.0 -f ai-worker/Dockerfile.pytorch-2.0 ai-worker/
docker build -t ai-worker-pytorch-2.1 -f ai-worker/Dockerfile.pytorch-2.1 ai-worker/
docker build -t ai-worker-pytorch-2.0-gpu -f ai-worker/Dockerfile.pytorch-2.0-gpu ai-worker/
docker build -t ai-worker-pytorch-2.1-gpu -f ai-worker/Dockerfile.pytorch-2.1-gpu ai-worker/
docker build -t ai-worker-tensorflow -f ai-worker/Dockerfile.tensorflow ai-worker/
docker build -t ai-worker-sklearn -f ai-worker/Dockerfile.sklearn ai-worker/

# Start services
echo "🚀 Starting services..."
$COMPOSE_CMD -f docker-compose.dynamic.yml up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check service health
echo "🔍 Checking service health..."

# Check Redis
if $COMPOSE_CMD -f docker-compose.dynamic.yml exec -T redis redis-cli ping | grep -q "PONG"; then
    echo "✅ Redis is healthy"
else
    echo "❌ Redis is not responding"
    exit 1
fi

# Check Backend
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Backend API is healthy"
else
    echo "❌ Backend API is not responding"
    exit 1
fi

# Check Worker Manager
if curl -f http://localhost:8001/workers/status > /dev/null 2>&1; then
    echo "✅ Worker Manager is healthy"
else
    echo "❌ Worker Manager is not responding"
    exit 1
fi

echo ""
echo "🎉 AI Job Queue System with Dynamic Workers is running!"
echo ""
echo "📊 Service URLs:"
echo "  - Backend API: http://localhost:8000"
echo "  - Worker Manager: http://localhost:8001"
echo "  - Redis: localhost:6379"
echo ""
echo "🔧 Management Commands:"
echo "  - View logs: $COMPOSE_CMD -f docker-compose.dynamic.yml logs -f"
echo "  - Stop services: $COMPOSE_CMD -f docker-compose.dynamic.yml down"
echo "  - Worker status: curl http://localhost:8001/workers/status"
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
