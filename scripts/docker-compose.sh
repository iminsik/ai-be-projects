#!/bin/bash

# Docker Compose compatibility script
# Automatically detects and uses the correct Docker Compose command

# Function to detect Docker Compose command
detect_docker_compose() {
    if docker compose version &> /dev/null; then
        echo "docker compose"
    elif command -v docker-compose &> /dev/null; then
        echo "docker-compose"
    else
        echo "error"
    fi
}

# Get the Docker Compose command
DOCKER_COMPOSE_CMD=$(detect_docker_compose)

if [ "$DOCKER_COMPOSE_CMD" = "error" ]; then
    echo "❌ Docker Compose is not installed or not accessible."
    echo "Please install Docker Compose first."
    exit 1
fi

echo "🔍 Using Docker Compose command: $DOCKER_COMPOSE_CMD"
echo ""

# Pass all arguments to the detected Docker Compose command
exec $DOCKER_COMPOSE_CMD "$@"
