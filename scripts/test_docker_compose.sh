#!/bin/bash

# Test script to verify Docker Compose detection logic

echo "🧪 Testing Docker Compose detection..."

# Test the detection function
detect_docker_compose() {
    if docker compose version &> /dev/null; then
        echo "docker compose"
    elif command -v docker-compose &> /dev/null; then
        echo "docker-compose"
    else
        echo "error"
    fi
}

# Test detection
DOCKER_COMPOSE_CMD=$(detect_docker_compose)

echo "Detected command: $DOCKER_COMPOSE_CMD"

if [ "$DOCKER_COMPOSE_CMD" = "error" ]; then
    echo "❌ No Docker Compose found"
    echo "This is expected if Docker is not installed in this environment"
else
    echo "✅ Docker Compose detected: $DOCKER_COMPOSE_CMD"
fi

echo ""
echo "📝 The script will work correctly when Docker is installed with either:"
echo "  - Docker Compose V1: docker-compose"
echo "  - Docker Compose V2: docker compose"
