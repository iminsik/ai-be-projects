#!/bin/bash

# Full Stack Development Script
# This script runs the complete system with frontend, backend, and workers

set -e

echo "🚀 Starting Full Stack AI Job Queue System..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if docker-compose is available
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
elif docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
else
    echo "❌ Neither docker-compose nor 'docker compose' is available."
    echo "   Please install Docker Compose."
    exit 1
fi

echo "🐳 Starting services with $COMPOSE_CMD..."

# Start all services
$COMPOSE_CMD up --build -d

echo ""
echo "✅ Services started successfully!"
echo ""
echo "🌐 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo "🗄️  Redis: localhost:6379"
echo ""
echo "📊 To view logs:"
echo "   $COMPOSE_CMD logs -f"
echo ""
echo "🛑 To stop services:"
echo "   $COMPOSE_CMD down"
echo ""
echo "🔄 To restart services:"
echo "   $COMPOSE_CMD restart"
echo ""

# Wait a moment for services to start
sleep 5

# Check if services are healthy
echo "🔍 Checking service health..."

# Check backend
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Backend is healthy"
else
    echo "⚠️  Backend is not responding yet (may still be starting)"
fi

# Check frontend
if curl -s http://localhost:3000 > /dev/null; then
    echo "✅ Frontend is healthy"
else
    echo "⚠️  Frontend is not responding yet (may still be starting)"
fi

echo ""
echo "🎉 Setup complete! Open http://localhost:3000 in your browser to get started."
