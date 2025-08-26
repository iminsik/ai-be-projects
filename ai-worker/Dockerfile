FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy dependency files
COPY pyproject.toml ./

# Install dependencies (CPU version by default)
RUN uv sync --frozen

# Copy source code
COPY src/ ./src/

# Create model storage directory
RUN mkdir -p /app/models

# Run the worker
CMD ["uv", "run", "python", "src/worker.py"]
