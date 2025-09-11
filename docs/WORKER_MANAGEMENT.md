# Worker Management Guide

This guide covers how to manage and run multiple AI workers with different frameworks and configurations.

## Worker Types

The system supports the following worker types:

| Worker Type | Framework | GPU Support | Queue Names |
|-------------|-----------|-------------|-------------|
| `pytorch-2.0` | PyTorch 2.0 | CPU Only | `ai:training:pytorch-2.0:queue`, `ai:inference:pytorch-2.0:queue` |
| `pytorch-2.1` | PyTorch 2.1 | CPU Only | `ai:training:pytorch-2.1:queue`, `ai:inference:pytorch-2.1:queue` |
| `pytorch-2.0-gpu` | PyTorch 2.0 | CUDA 11.8 | `ai:training:pytorch-2.0-gpu:queue`, `ai:inference:pytorch-2.0-gpu:queue` |
| `pytorch-2.1-gpu` | PyTorch 2.1 | CUDA 12.1 | `ai:training:pytorch-2.1-gpu:queue`, `ai:inference:pytorch-2.1-gpu:queue` |
| `tensorflow` | TensorFlow 2.13 | CPU Only | `ai:training:tensorflow:queue`, `ai:inference:tensorflow:queue` |
| `sklearn` | Scikit-learn 1.3 | CPU Only | `ai:training:sklearn:queue`, `ai:inference:sklearn:queue` |

## Environment Variables

Each worker requires specific environment variables:

### Required Variables
- `WORKER_TYPE`: The type of worker (e.g., `pytorch-2.1-gpu`)
- `MODEL_FRAMEWORK`: The ML framework (e.g., `pytorch`, `tensorflow`, `sklearn`)

### Optional Variables
- `USE_GPU`: Set to `true` for GPU workers
- `REDIS_HOST`: Redis server host (default: `localhost`)
- `REDIS_PORT`: Redis server port (default: `6379`)
- `REDIS_DB`: Redis database number (default: `0`)
- `REDIS_PASSWORD`: Redis password (default: none)
- `MAX_CONCURRENT_JOBS`: Maximum concurrent jobs per worker (default: `2`)

## Running Workers

### Method 1: Manual Setup (Development)

#### Single Worker
```bash
cd ai-worker

# Set environment variables
export WORKER_TYPE=pytorch-2.1
export MODEL_FRAMEWORK=pytorch

# Install dependencies
uv sync --extra pytorch_2_1

# Start worker
uv run python src/worker.py
```

#### Multiple Workers (Different Terminals)

**Terminal 1 - PyTorch 2.0 CPU:**
```bash
cd ai-worker
export WORKER_TYPE=pytorch-2.0
export MODEL_FRAMEWORK=pytorch
uv sync --extra pytorch_2_0
uv run python src/worker.py
```

**Terminal 2 - PyTorch 2.1 GPU:**
```bash
cd ai-worker
export WORKER_TYPE=pytorch-2.1-gpu
export MODEL_FRAMEWORK=pytorch
export USE_GPU=true
uv sync --extra pytorch_2_1_gpu
uv run python src/worker.py
```

**Terminal 3 - TensorFlow:**
```bash
cd ai-worker
export WORKER_TYPE=tensorflow
export MODEL_FRAMEWORK=tensorflow
uv sync --extra tensorflow
uv run python src/worker.py
```

**Terminal 4 - Scikit-learn:**
```bash
cd ai-worker
export WORKER_TYPE=sklearn
export MODEL_FRAMEWORK=sklearn
uv sync --extra sklearn
uv run python src/worker.py
```

### Method 2: Docker Compose (Production)

#### Start All Workers
```bash
./scripts/docker-compose.sh up -d
```

#### Start Specific Workers
```bash
# CPU workers only
docker compose up -d redis backend ai-worker-pytorch-2.0 ai-worker-pytorch-2.1 ai-worker-tensorflow ai-worker-sklearn

# GPU workers only
docker compose up -d redis backend ai-worker-pytorch-2.0-gpu ai-worker-pytorch-2.1-gpu

# Mixed setup
docker compose up -d redis backend ai-worker-pytorch-2.1 ai-worker-pytorch-2.1-gpu ai-worker-tensorflow
```

### Method 3: Helper Script

Create a helper script for easy worker management:

```bash
#!/bin/bash
# scripts/run_workers.sh

WORKER_TYPE=$1
FRAMEWORK=$2
EXTRA=$3
USE_GPU=${4:-false}

if [ -z "$WORKER_TYPE" ]; then
    echo "Usage: $0 <worker_type> <framework> <extra> [use_gpu]"
    echo "Example: $0 pytorch-2.1-gpu pytorch pytorch_2_1_gpu true"
    exit 1
fi

cd ai-worker

export WORKER_TYPE=$WORKER_TYPE
export MODEL_FRAMEWORK=$FRAMEWORK
if [ "$USE_GPU" = "true" ]; then
    export USE_GPU=true
fi

uv sync --extra $EXTRA
uv run python src/worker.py
```

Usage:
```bash
chmod +x scripts/run_workers.sh

# Run PyTorch 2.1 GPU worker
./scripts/run_workers.sh pytorch-2.1-gpu pytorch pytorch_2_1_gpu true

# Run TensorFlow worker
./scripts/run_workers.sh tensorflow tensorflow tensorflow false
```

## Worker Configuration

### PyTorch Workers

#### CPU Workers
- **PyTorch 2.0**: Uses PyTorch 2.0.1 with CPU-only builds
- **PyTorch 2.1**: Uses PyTorch 2.1.0 with CPU-only builds
- **Dependencies**: transformers, datasets, accelerate, tqdm

#### GPU Workers
- **PyTorch 2.0 GPU**: Uses CUDA 11.8 builds
- **PyTorch 2.1 GPU**: Uses CUDA 12.1 builds
- **GPU Memory Management**: Automatic memory allocation and cleanup
- **CUDA Support**: Requires NVIDIA GPU with appropriate drivers

### TensorFlow Workers
- **Version**: TensorFlow 2.13.0
- **Dependencies**: tensorflow-hub, tensorflow-datasets
- **GPU Support**: Automatic GPU detection and usage

### Scikit-learn Workers
- **Version**: Scikit-learn 1.3.0+
- **Dependencies**: joblib
- **Use Case**: Traditional ML algorithms, tabular data

## Scaling Workers

### Horizontal Scaling

#### Docker Compose Scaling
```bash
# Scale specific worker types
docker compose up -d --scale ai-worker-pytorch-2.1=3
docker compose up -d --scale ai-worker-pytorch-2.1-gpu=2
```

#### Manual Scaling
Run multiple instances of the same worker type:

```bash
# Terminal 1
export WORKER_TYPE=pytorch-2.1
export MODEL_FRAMEWORK=pytorch
uv run python src/worker.py

# Terminal 2 (same worker type)
export WORKER_TYPE=pytorch-2.1
export MODEL_FRAMEWORK=pytorch
uv run python src/worker.py

# Terminal 3 (same worker type)
export WORKER_TYPE=pytorch-2.1
export MODEL_FRAMEWORK=pytorch
uv run python src/worker.py
```

### Vertical Scaling

#### Resource Limits
```bash
# Set memory limits for Docker containers
docker run -m 4g --cpus="2.0" ai-worker-pytorch-2.1-gpu

# Set concurrent job limits
export MAX_CONCURRENT_JOBS=4
```

## Monitoring Workers

### Health Checks

#### Worker Status
```bash
# Check if worker is running
ps aux | grep "python src/worker.py"

# Check worker logs
docker logs ai-worker-pytorch-2.1-gpu
```

#### Redis Queue Monitoring
```bash
# Connect to Redis
redis-cli

# Check queue lengths
LLEN ai:training:pytorch-2.1:queue
LLEN ai:inference:pytorch-2.1:queue

# List all queues
KEYS ai:*:queue
```

### Logging

#### Log Levels
- `INFO`: General information about job processing
- `WARNING`: Non-critical issues (e.g., GPU memory warnings)
- `ERROR`: Job failures and critical errors
- `DEBUG`: Detailed debugging information

#### Log Format
```
2024-01-15 10:30:00,123 - __main__ - INFO - Worker started for pytorch-2.1-gpu
2024-01-15 10:30:00,124 - __main__ - INFO - Listening on queues: ai:training:pytorch-2.1-gpu:queue, ai:inference:pytorch-2.1-gpu:queue
2024-01-15 10:30:15,456 - __main__ - INFO - Received training job: job_123
2024-01-15 10:30:15,457 - __main__ - INFO - Starting training job job_123 for model type: bert using pytorch
```

## Troubleshooting

### Common Issues

#### 1. Worker Not Starting
```bash
# Check environment variables
echo $WORKER_TYPE
echo $MODEL_FRAMEWORK

# Check dependencies
uv sync --extra pytorch_2_1

# Check Redis connection
redis-cli ping
```

#### 2. GPU Workers Not Using GPU
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Check GPU memory
nvidia-smi

# Verify GPU environment variables
echo $USE_GPU
```

#### 3. Jobs Not Being Processed
```bash
# Check queue status
redis-cli LLEN ai:training:pytorch-2.1:queue

# Check worker logs
tail -f worker.log

# Verify worker is listening
netstat -tlnp | grep python
```

#### 4. Memory Issues
```bash
# Check system memory
free -h

# Check GPU memory
nvidia-smi

# Reduce concurrent jobs
export MAX_CONCURRENT_JOBS=1
```

### Performance Optimization

#### CPU Workers
- Increase `MAX_CONCURRENT_JOBS` for CPU-bound tasks
- Use multiple CPU workers for parallel processing
- Optimize batch sizes for your hardware

#### GPU Workers
- Monitor GPU memory usage
- Use appropriate batch sizes
- Consider mixed precision training
- Implement gradient accumulation for large models

#### Memory Management
- Set appropriate memory limits
- Monitor memory usage patterns
- Implement job queuing for memory-intensive tasks

## Best Practices

### 1. Worker Distribution
- Run CPU workers for lightweight tasks
- Use GPU workers for compute-intensive training
- Balance worker types based on workload

### 2. Resource Management
- Monitor resource usage regularly
- Set appropriate limits for concurrent jobs
- Implement proper cleanup procedures

### 3. Error Handling
- Implement retry mechanisms
- Log errors for debugging
- Set up monitoring and alerting

### 4. Security
- Use environment variables for sensitive data
- Implement proper authentication
- Secure Redis connections

### 5. Deployment
- Use Docker for consistent environments
- Implement health checks
- Set up proper logging and monitoring

## Example Configurations

### Development Setup
```bash
# Lightweight development setup
docker compose up -d redis backend ai-worker-pytorch-2.1 ai-worker-sklearn
```

### Production Setup
```bash
# Full production setup with GPU support
docker compose up -d redis backend \
  ai-worker-pytorch-2.0 \
  ai-worker-pytorch-2.1 \
  ai-worker
```
